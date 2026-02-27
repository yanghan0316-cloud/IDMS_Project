"""
src.internal.fatigue_logic

把“连续帧阈值”的工程逻辑封装成一个小状态机。
你只需要每帧喂进去 ear/mar，就能得到:
- is_drowsy: 是否疲劳（闭眼时间过长）
- is_yawning: 是否哈欠（张嘴时间过长）
- blink: 是否眨眼（短暂闭眼事件，可选）

【TODO(参数待定)】
- 如果你们后续测得真实 FPS，请在 config.yaml 里填 internal.fps。
  这样你就可以用 *_duration_sec（秒）来设置阈值；否则会退化为使用 *frames。
- 若驾驶员戴口罩导致嘴部关键点不稳定，可以在 config.yaml 里临时把 enable_yawn 设为 false。

用法（在 FaceMeshDetector.process 里）:
    analyzer = FatigueAnalyzer(config["internal"])
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
        cfg = config or {}

        # ====== 开关（口罩/特殊场景时可临时关闭） ======
        self.enable_drowsy = bool(cfg.get("enable_drowsy", True))
        self.enable_yawn = bool(cfg.get("enable_yawn", True))

        # ====== 指标阈值 ======
        self.ear_threshold = float(cfg.get("ear_threshold", 0.22))
        self.mar_threshold = float(cfg.get("mar_threshold", 0.60))

        # ====== FPS / 秒 -> 帧 的转换 ======
        # 【TODO(参数待定)】如果你们后续测得真实 FPS，建议在 config.yaml 里填 internal.fps
        self.fps = float(cfg.get("fps", 0.0) or 0.0)

        # 连续帧阈值（fallback）
        self.consecutive_frames_eye = int(cfg.get("consecutive_frames_eye", 45))
        self.consecutive_frames_mouth = int(cfg.get("consecutive_frames_mouth", 60))

        # 更直观的“持续秒数”配置（优先）
        self.drowsy_duration_sec = float(cfg.get("drowsy_duration_sec", 1.5))
        self.yawn_duration_sec = float(cfg.get("yawn_duration_sec", 2.0))

        # 为“眨眼”留一个更短的窗口（可选，不影响疲劳报警）
        self.blink_max_frames = int(cfg.get("blink_max_frames", 8))
        self.blink_max_sec = float(cfg.get("blink_max_sec", 0.3))

        # 平滑：指数滑动平均 (EMA)
        self.ema_alpha = float(cfg.get("ema_alpha", 0.4))  # 越大越跟随当前帧

        # 把秒转换成帧阈值（如果 fps 未知则回退为 config 里的 frames）
        self._eye_frames_th = self._sec_to_frames(self.drowsy_duration_sec, self.consecutive_frames_eye)
        self._mouth_frames_th = self._sec_to_frames(self.yawn_duration_sec, self.consecutive_frames_mouth)
        self._blink_frames_th = self._sec_to_frames(self.blink_max_sec, self.blink_max_frames)

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

    def _sec_to_frames(self, duration_sec: float, fallback_frames: int) -> int:
        if self.fps and self.fps > 1.0 and duration_sec and duration_sec > 0:
            return max(1, int(round(duration_sec * self.fps)))
        return max(1, int(fallback_frames))

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
        """每帧调用一次。"""
        ear = float(ear)
        mar = float(mar)

        # 1) 平滑
        self._ear_ema = self._ema(self._ear_ema, ear)
        self._mar_ema = self._ema(self._mar_ema, mar)

        ear_use = float(self._ear_ema)
        mar_use = float(self._mar_ema)

        # 2) 连续帧计数
        blink = False

        # --- 眼睛（低于阈值：闭眼）---
        if ear_use < self.ear_threshold:
            self._eye_low_frames += 1
            self._blink_segment += 1
        else:
            # 从“闭眼段”回到“睁眼”
            if 1 <= self._blink_segment <= self._blink_frames_th:
                blink = True
            self._blink_segment = 0
            self._eye_low_frames = 0  # 这里按“连续闭眼”定义疲劳

        # 疲劳判定
        if self.enable_drowsy:
            self._is_drowsy = self._eye_low_frames >= self._eye_frames_th
        else:
            self._is_drowsy = False

        # --- 嘴巴（高于阈值：张嘴）---
        if mar_use > self.mar_threshold:
            self._mouth_high_frames += 1
        else:
            self._mouth_high_frames = 0

        if self.enable_yawn:
            self._is_yawning = self._mouth_high_frames >= self._mouth_frames_th
        else:
            self._is_yawning = False

        return FatigueState(
            ear=ear,
            mar=mar,
            ear_ema=ear_use,
            mar_ema=mar_use,
            blink=blink,
            is_drowsy=self._is_drowsy,
            is_yawning=self._is_yawning,
            drowsy_frames=self._eye_low_frames,
            yawn_frames=self._mouth_high_frames,
        )
