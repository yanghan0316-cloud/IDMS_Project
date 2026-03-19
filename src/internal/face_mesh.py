"""
src.internal.face_mesh

MediaPipe FaceMesh 的封装：输入一帧 BGR 图像，输出驾驶员状态数据。

接口:
    FaceMeshDetector.process(frame_bgr) -> dict

v4 新增输出字段:
    - perclos: float            # 0.0~1.0, 滑动窗口内闭眼帧占比
    - is_perclos_fatigued: bool # PERCLOS 超阈值
    - blink_freq: float         # 每分钟眨眼次数
    - is_blink_freq_high: bool  # 眨眼频率过高
"""

from __future__ import annotations

from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

from .geometry import calculate_ear, calculate_mar, solve_head_pose, HeadPose
from .fatigue_logic import FatigueAnalyzer
from .attention_logic import AttentionAnalyzer


Point2D = Tuple[int, int]

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

MOUTH_IDX = {
    "left_corner": 61,
    "right_corner": 291,
    "upper_inner": 13,
    "lower_inner": 14,
    "upper_mid_1": 81,
    "lower_mid_1": 311,
    "upper_mid_2": 82,
    "lower_mid_2": 312,
}


# 无人脸时的默认返回值
_NO_FACE_RESULT = {
    "ear": 0.0, "mar": 0.0,
    "blink": False,
    "is_drowsy": False, "is_yawning": False,
    "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
    "is_distracted": False, "is_nodding": False,
    "has_face": False,
    # v4
    "perclos": 0.0, "is_perclos_fatigued": False,
    "blink_freq": 0.0, "is_blink_freq_high": False,
}


class FaceMeshDetector:
    def __init__(self, config: Dict):
        if mp is None:
            raise ImportError("mediapipe 未安装或导入失败")

        self.cfg = config or {}

        self.max_num_faces = int(self.cfg.get("max_num_faces", 1))
        self.refine_landmarks = bool(self.cfg.get("refine_landmarks", True))
        self.min_det_conf = float(self.cfg.get("min_detection_confidence", 0.5))
        self.min_trk_conf = float(self.cfg.get("min_tracking_confidence", 0.5))
        self.return_landmarks = bool(self.cfg.get("return_landmarks", False))
        self.enable_head_pose = bool(self.cfg.get("enable_head_pose", True))
        self.enable_attention = bool(self.cfg.get("enable_attention", True))

        self.analyzer = FatigueAnalyzer(self.cfg)
        self.attention = AttentionAnalyzer(self.cfg)

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
        if frame_bgr is None or frame_bgr.size == 0:
            return dict(_NO_FACE_RESULT)

        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            self.analyzer.reset()
            self.attention.reset()
            return dict(_NO_FACE_RESULT)

        face_landmarks = results.multi_face_landmarks[0].landmark

        # --- EAR / MAR ---
        left_ear = calculate_ear(face_landmarks, LEFT_EYE_IDX, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE_IDX, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(face_landmarks, MOUTH_IDX, w, h)

        # --- 疲劳/哈欠/PERCLOS/眨眼频率 状态机 ---
        state = self.analyzer.update(ear=ear, mar=mar)

        # --- Head Pose ---
        yaw = pitch = roll = 0.0
        pose: Optional[HeadPose] = None
        if self.enable_head_pose:
            pose = solve_head_pose(face_landmarks, w, h)
            if pose is not None:
                yaw, pitch, roll = pose.yaw, pose.pitch, pose.roll

        # --- Attention ---
        is_distracted = False
        is_nodding = False
        distracted_frames = 0
        nod_frames = 0
        yaw_grace_cnt = 0
        if self.enable_head_pose and self.enable_attention and pose is not None:
            att = self.attention.update(yaw=yaw, pitch=pitch)
            is_distracted = bool(att.is_distracted)
            is_nodding = bool(att.is_nodding)
            distracted_frames = int(att.distracted_frames)
            nod_frames = int(att.nod_frames)
            yaw_grace_cnt = int(self.attention._yaw_grace_cnt)
            yaw, pitch = float(att.yaw_ema), float(att.pitch_ema)
        else:
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
            "yaw_grace_cnt": int(yaw_grace_cnt),
            "has_face": True,

            # v4 新增
            "perclos": float(state.perclos),
            "is_perclos_fatigued": bool(state.is_perclos_fatigued),
            "blink_freq": float(state.blink_freq),
            "is_blink_freq_high": bool(state.is_blink_freq_high),
        }

        if self.return_landmarks:
            pts: List[Point2D] = []
            for lm in face_landmarks:
                pts.append((int(lm.x * w), int(lm.y * h)))
            out["landmarks"] = pts

        return out