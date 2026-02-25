"""
src.internal.face_mesh

MediaPipe FaceMesh 的封装：输入一帧 BGR 图像，输出驾驶员状态数据。

你最需要关心的接口只有一个：
    FaceMeshDetector.process(frame_bgr) -> dict

返回字段（最小集）：
- ear: float
- mar: float
- is_drowsy: bool
- is_yawning: bool

新增（已完成）：
- is_distracted: bool   # 分心（转头幅度过大且持续）
- is_nodding: bool      # 点头/低头（pitch 过阈值且持续）

扩展字段（方便后续迭代）：
- yaw/pitch/roll: float (degrees)
- blink: bool
- landmarks: List[(x,y)]  (可用于 UI 画点；注意这会占用更多内存/带宽)

提示：
- 如果你担心性能：把 landmarks 的返回关掉（return_landmarks=False）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None  # 让 import 不至于直接炸，方便在没有 mediapipe 的环境做静态检查

from .geometry import calculate_ear, calculate_mar, solve_head_pose, HeadPose
from .fatigue_logic import FatigueAnalyzer
from .attention_logic import AttentionAnalyzer


Point2D = Tuple[int, int]


# MediaPipe FaceMesh 指标点位（0~467）
# 眼睛：你们之前草稿里已经用过这一套 6 点 EAR
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# 嘴部：MAR（口角水平 / 多条竖直线平均）
MOUTH_IDX = {
    "left_corner": 61,
    "right_corner": 291,
    "upper_inner": 13,
    "lower_inner": 14,
    # 可选两条竖直线（让 MAR 更稳一点）
    "upper_mid_1": 81,
    "lower_mid_1": 311,
    "upper_mid_2": 82,
    "lower_mid_2": 312,
}


class FaceMeshDetector:
    def __init__(self, config: Dict):
        if mp is None:
            raise ImportError("mediapipe 未安装或导入失败，请先 pip install mediapipe")

        self.cfg = config or {}

        # 配置项
        self.max_num_faces = int(self.cfg.get("max_num_faces", 1))
        self.refine_landmarks = bool(self.cfg.get("refine_landmarks", True))
        self.min_det_conf = float(self.cfg.get("min_detection_confidence", 0.5))
        self.min_trk_conf = float(self.cfg.get("min_tracking_confidence", 0.5))

        self.return_landmarks = bool(self.cfg.get("return_landmarks", False))

        # 头部姿态开关/阈值（可选）
        self.enable_head_pose = bool(self.cfg.get("enable_head_pose", True))

        # 分心/点头逻辑开关（依赖头部姿态）
        self.enable_attention = bool(self.cfg.get("enable_attention", True))

        # Fatigue state machine
        self.analyzer = FatigueAnalyzer(self.cfg)

        # Attention (distraction / nodding) state machine
        self.attention = AttentionAnalyzer(self.cfg)

        # MediaPipe FaceMesh
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_det_conf,
            min_tracking_confidence=self.min_trk_conf,
        )

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()

    def process(self, frame_bgr: np.ndarray) -> Dict:
        """
        输入：BGR 图像
        输出：dict（驾驶员状态）
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "ear": 0.0,
                "mar": 0.0,
                "blink": False,
                "is_drowsy": False,
                "is_yawning": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "has_face": False,
            }

        h, w = frame_bgr.shape[:2]

        # MediaPipe 要 RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            # 没检测到脸就重置计数器，避免“离开画面后仍报警”
            self.analyzer.reset()
            self.attention.reset()
            return {
                "ear": 0.0,
                "mar": 0.0,
                "blink": False,
                "is_drowsy": False,
                "is_yawning": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "is_distracted": False,
                "is_nodding": False,
                "has_face": False,
            }

        # 只取第一张脸
        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- EAR / MAR ---
        left_ear = calculate_ear(face_landmarks, LEFT_EYE_IDX, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
        ear = (left_ear + right_ear) / 2.0

        mar = calculate_mar(face_landmarks, MOUTH_IDX, w, h)

        # --- 疲劳/哈欠状态机 ---
        state = self.analyzer.update(ear=ear, mar=mar)

        # --- Head Pose ---
        yaw = pitch = roll = 0.0
        pose: Optional[HeadPose] = None
        if self.enable_head_pose:
            pose = solve_head_pose(face_landmarks, w, h)
            if pose is not None:
                yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

        # --- Attention (Distraction / Nodding) ---
        is_distracted = False
        is_nodding = False
        distracted_frames = 0
        nod_frames = 0
        if self.enable_head_pose and self.enable_attention and pose is not None:
            att = self.attention.update(yaw=yaw, pitch=pitch)
            is_distracted = bool(att.is_distracted)
            is_nodding = bool(att.is_nodding)
            distracted_frames = int(att.distracted_frames)
            nod_frames = int(att.nod_frames)
            # 使用 EMA 后的角度更适合展示
            yaw, pitch = float(att.yaw_ema), float(att.pitch_ema)
        else:
            # 没有姿态结果时，重置计数器，避免上一帧状态“粘住”
            self.attention.reset()

        # --- 输出 ---
        out = {
            "ear": float(state.ear_ema),
            "mar": float(state.mar_ema),
            "blink": bool(state.blink),
            "is_drowsy": bool(state.is_drowsy),
            "is_yawning": bool(state.is_yawning),
            "drowsy_frames": int(state.drowsy_frames),
            "yawn_frames": int(state.yawn_frames),
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "is_distracted": bool(is_distracted),
            "is_nodding": bool(is_nodding),
            "distracted_frames": int(distracted_frames),
            "nod_frames": int(nod_frames),
            "has_face": True,
        }

        if self.return_landmarks:
            # 转为像素点便于画图
            pts: List[Point2D] = []
            for lm in face_landmarks:
                pts.append((int(lm.x * w), int(lm.y * h)))
            out["landmarks"] = pts

        return out
