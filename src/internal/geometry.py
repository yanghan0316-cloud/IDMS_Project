"""
src.internal.geometry

几何工具函数：EAR（眼睛纵横比）、MAR（嘴部纵横比）、头部姿态解算（PnP->欧拉角）。

这些函数尽量“纯函数化”：
- 输入：关键点坐标 / 图像尺寸
- 输出：数值（ear/mar/角度）
这样做的好处是：后续你想换模型、换阈值、换 UI，都不会影响核心计算。

说明：
- MediaPipe FaceMesh 返回的是 0~1 的归一化坐标 (x,y)，这里统一转成像素坐标再算距离。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import math
import numpy as np
import cv2


Point2D = Tuple[float, float]


# -----------------------------
# 基础距离函数
# -----------------------------
def _to_pixel_xy(
    landmark,
    image_w: int,
    image_h: int,
) -> Point2D:
    """把 MediaPipe 的单个 landmark (含 x,y) 转成像素坐标。"""
    return float(landmark.x * image_w), float(landmark.y * image_h)


def euclidean(p1: Point2D, p2: Point2D) -> float:
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _get_points(
    landmarks: Sequence,
    indices: Sequence[int],
    image_w: int,
    image_h: int,
) -> List[Point2D]:
    return [_to_pixel_xy(landmarks[i], image_w, image_h) for i in indices]


# -----------------------------
# EAR / MAR
# -----------------------------
def calculate_ear(
    landmarks: Sequence,
    eye_indices: Sequence[int],
    image_w: int,
    image_h: int,
) -> float:
    """
    计算 Eye Aspect Ratio (EAR)

    参数:
        eye_indices: 6 个点的索引，按顺序 [p1, p2, p3, p4, p5, p6]
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    返回:
        ear: float（越小越闭眼）
    """
    if len(eye_indices) != 6:
        raise ValueError("eye_indices 必须是 6 个点的索引")

    p1, p2, p3, p4, p5, p6 = _get_points(landmarks, eye_indices, image_w, image_h)

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calculate_mar(
    landmarks: Sequence,
    mouth_indices: Dict[str, int],
    image_w: int,
    image_h: int,
) -> float:
    """
    计算 Mouth Aspect Ratio (MAR)

    我们使用“多条竖直线取平均 / 口角水平线”的经典做法：
        MAR = (d(upper_inner, lower_inner) + d(upper, lower)+ d(upper2, lower2)) / (3 * d(left_corner, right_corner))

    参数 mouth_indices: 一个 dict，至少包含：
        left_corner, right_corner, upper_inner, lower_inner
    可选：
        upper_mid_1, lower_mid_1, upper_mid_2, lower_mid_2
    """
    def pt(idx: int) -> Point2D:
        return _to_pixel_xy(landmarks[idx], image_w, image_h)

    left_corner = pt(mouth_indices["left_corner"])
    right_corner = pt(mouth_indices["right_corner"])
    upper_inner = pt(mouth_indices["upper_inner"])
    lower_inner = pt(mouth_indices["lower_inner"])

    # 两条可选的竖线（不提供就退化为只用一条）
    dlist = [euclidean(upper_inner, lower_inner)]

    if "upper_mid_1" in mouth_indices and "lower_mid_1" in mouth_indices:
        dlist.append(euclidean(pt(mouth_indices["upper_mid_1"]), pt(mouth_indices["lower_mid_1"])))
    if "upper_mid_2" in mouth_indices and "lower_mid_2" in mouth_indices:
        dlist.append(euclidean(pt(mouth_indices["upper_mid_2"]), pt(mouth_indices["lower_mid_2"])))

    horiz = euclidean(left_corner, right_corner)
    if horiz <= 1e-6:
        return 0.0

    return float(sum(dlist) / (len(dlist) * horiz))


# -----------------------------
# 头部姿态解算（PnP）
# -----------------------------
@dataclass(frozen=True)
class HeadPose:
    yaw: float   # 左右转头，单位：度（+向右，-向左，按 OpenCV 坐标系近似）
    pitch: float # 低头/抬头，单位：度（+抬头，-低头，近似）
    roll: float  # 歪头，单位：度（顺/逆时针，近似）


# MediaPipe FaceMesh 常用姿态关键点（2D）
DEFAULT_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# 通用 3D 模型点（单位：毫米）
# 这是“通用人脸几何”的近似，不需要非常精确，主要是用于得到稳定的 yaw/pitch/roll 相对值。
DEFAULT_MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),         # nose tip
        (0.0, -330.0, -65.0),    # chin
        (-225.0, 170.0, -135.0), # left eye outer
        (225.0, 170.0, -135.0),  # right eye outer
        (-150.0, -150.0, -125.0),# left mouth
        (150.0, -150.0, -125.0), # right mouth
    ],
    dtype=np.float64,
)


def solve_head_pose(
    landmarks: Sequence,
    image_w: int,
    image_h: int,
    pose_landmarks: Dict[str, int] = DEFAULT_POSE_LANDMARKS,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[HeadPose]:
    """
    用 OpenCV solvePnP 解算头部姿态，返回 yaw/pitch/roll（度）。

    返回 None 表示解算失败（比如点不够 / 数值不稳定）。
    """
    try:
        image_points = np.array(
            [
                _to_pixel_xy(landmarks[pose_landmarks["nose_tip"]], image_w, image_h),
                _to_pixel_xy(landmarks[pose_landmarks["chin"]], image_w, image_h),
                _to_pixel_xy(landmarks[pose_landmarks["left_eye_outer"]], image_w, image_h),
                _to_pixel_xy(landmarks[pose_landmarks["right_eye_outer"]], image_w, image_h),
                _to_pixel_xy(landmarks[pose_landmarks["left_mouth"]], image_w, image_h),
                _to_pixel_xy(landmarks[pose_landmarks["right_mouth"]], image_w, image_h),
            ],
            dtype=np.float64,
        )
    except Exception:
        return None

    if camera_matrix is None:
        focal_length = float(image_w)  # 近似：用图像宽作为焦距
        center = (image_w / 2.0, image_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1.0],
            ],
            dtype=np.float64,
        )

    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # solvePnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        DEFAULT_MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    # rot_vec -> rotation matrix
    rot_mtx, _ = cv2.Rodrigues(rot_vec)

    # rotation matrix -> Euler angles
    # 参考：OpenCV 坐标系下的常见解法
    sy = math.sqrt(rot_mtx[0, 0] * rot_mtx[0, 0] + rot_mtx[1, 0] * rot_mtx[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rot_mtx[2, 1], rot_mtx[2, 2])
        y = math.atan2(-rot_mtx[2, 0], sy)
        z = math.atan2(rot_mtx[1, 0], rot_mtx[0, 0])
    else:
        x = math.atan2(-rot_mtx[1, 2], rot_mtx[1, 1])
        y = math.atan2(-rot_mtx[2, 0], sy)
        z = 0.0

    # 转成角度
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
