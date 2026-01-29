"""
src.internal.fatigue_logic

把“连续帧阈值”的工程逻辑封装成一个小状态机。
你只需要每帧喂进去 ear/mar，就能得到:
- is_drowsy: 是否疲劳（闭眼时间过长）
- is_yawning: 是否哈欠（张嘴时间过长）
- blink: 是否眨眼（短暂闭眼事件，可选）

用法（在 FaceMeshDetector.process 里）：
    analyzer = FatigueAnalyzer(config["internal"])
    ...
    out = analyzer.update(ear, mar)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class FatigueState:
    ear: float
    mar: float
    ear_ema: float
    mar_ema: float
    blink: bool
    is_drowsy: bool
    is_yawning: bool
    drowsy_frames: int
    yawn_frames: int


class FatigueAnalyzer:
    def __init__(self, config: Dict):
        # 阈值
        self.ear_threshold = float(config.get("ear_threshold", 0.22))
        self.mar_threshold = float(config.get("mar_threshold", 0.60))

        # 连续帧阈值
        self.consecutive_frames_eye = int(config.get("consecutive_frames_eye", 45))
        self.consecutive_frames_mouth = int(config.get("consecutive_frames_mouth", 60))

        # 为“眨眼”留一个更短的窗口（可选，不影响疲劳报警）
        self.blink_max_frames = int(config.get("blink_max_frames", 8))

        # 平滑：指数滑动平均 (EMA)
        self.ema_alpha = float(config.get("ema_alpha", 0.4))  # 越大越跟随当前帧

        # 计数器
        self._eye_low_frames = 0
        self._mouth_high_frames = 0

        # 为眨眼检测记录闭眼段长度
        self._blink_segment = 0

        # 初始化 EMA
        self._ear_ema: Optional[float] = None
        self._mar_ema: Optional[float] = None

        # 状态
        self._is_drowsy = False
        self._is_yawning = False

    def reset(self) -> None:
        self._eye_low_frames = 0
        self._mouth_high_frames = 0
        self._blink_segment = 0
        self._ear_ema = None
        self._mar_ema = None
        self._is_drowsy = False
        self._is_yawning = False

    def _ema(self, prev: Optional[float], cur: float) -> float:
        if prev is None:
            return cur
        return (1 - self.ema_alpha) * prev + self.ema_alpha * cur

    def update(self, ear: float, mar: float) -> FatigueState:
        """
        每帧调用一次。

        返回 FatigueState：
        - ear_ema/mar_ema 是平滑后的指标（建议用它做阈值判断更稳）
        """
        ear = float(ear)
        mar = float(mar)

        # 1) 平滑
        self._ear_ema = self._ema(self._ear_ema, ear)
        self._mar_ema = self._ema(self._mar_ema, mar)

        ear_use = self._ear_ema
        mar_use = self._mar_ema

        # 2) 连续帧计数
        blink = False

        # --- 眼睛（低于阈值：闭眼）---
        if ear_use < self.ear_threshold:
            self._eye_low_frames += 1
            self._blink_segment += 1
        else:
            # 从“闭眼段”回到“睁眼”
            if 1 <= self._blink_segment <= self.blink_max_frames:
                blink = True
            self._blink_segment = 0
            self._eye_low_frames = 0  # 这里按“连续闭眼”定义疲劳

        # 疲劳判定
        self._is_drowsy = self._eye_low_frames >= self.consecutive_frames_eye

        # --- 嘴巴（高于阈值：张嘴）---
        if mar_use > self.mar_threshold:
            self._mouth_high_frames += 1
        else:
            self._mouth_high_frames = 0

        self._is_yawning = self._mouth_high_frames >= self.consecutive_frames_mouth

        return FatigueState(
            ear=ear,
            mar=mar,
            ear_ema=float(ear_use),
            mar_ema=float(mar_use),
            blink=blink,
            is_drowsy=self._is_drowsy,
            is_yawning=self._is_yawning,
            drowsy_frames=self._eye_low_frames,
            yawn_frames=self._mouth_high_frames,
        )
