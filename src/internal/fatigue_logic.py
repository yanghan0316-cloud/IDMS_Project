"""
src.internal.fatigue_logic

疲劳检测状态机。每帧喂入 ear/mar，输出疲劳/哈欠/眨眼/PERCLOS/眨眼频率。

v4 新增:
    - PERCLOS (Percentage of Eye Closure):
      在滑动窗口内统计 EAR 低于阈值的帧占比。FHWA 认定的疲劳检测金标准。
      当 PERCLOS > perclos_threshold (默认 0.15) 时判定疲劳。
      它能捕捉到"频繁短暂闭眼"这种连续帧检测器无法检出的疲劳模式。

    - 眨眼频率 (Blink Frequency):
      在滑动窗口内统计每分钟眨眼次数。
      正常人 15-20 次/分钟，疲劳时显著升高至 25-30 次/分钟。
      当 blink_freq > blink_freq_high_threshold (默认 25) 时标记异常。

配置项 (config.yaml -> internal):
    # --- v4 新增 ---
    perclos_window_sec: 60.0        # PERCLOS 统计窗口（秒）
    perclos_threshold: 0.15         # PERCLOS 报警阈值 (0.15 = 15%)
    blink_freq_window_sec: 60.0    # 眨眼频率统计窗口（秒）
    blink_freq_high_threshold: 25  # 高频眨眼报警阈值（次/分钟）
"""

from __future__ import annotations

import time
from collections import deque
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

    # v4 新增
    perclos: float              # 0.0 ~ 1.0, 当前窗口内眼睛闭合比例
    is_perclos_fatigued: bool   # PERCLOS 超过阈值
    blink_freq: float           # 每分钟眨眼次数
    is_blink_freq_high: bool    # 眨眼频率过高


class FatigueAnalyzer:
    def __init__(self, config: Dict):
        cfg = config or {}

        # ====== 开关 ======
        self.enable_drowsy = bool(cfg.get("enable_drowsy", True))
        self.enable_yawn = bool(cfg.get("enable_yawn", True))

        # ====== 指标阈值 ======
        self.ear_threshold = float(cfg.get("ear_threshold", 0.22))
        self.mar_threshold = float(cfg.get("mar_threshold", 0.60))

        # ====== FPS / 秒 -> 帧 的转换 ======
        self.fps = float(cfg.get("fps", 0.0) or 0.0)

        self.consecutive_frames_eye = int(cfg.get("consecutive_frames_eye", 45))
        self.consecutive_frames_mouth = int(cfg.get("consecutive_frames_mouth", 60))
        self.drowsy_duration_sec = float(cfg.get("drowsy_duration_sec", 1.5))
        self.yawn_duration_sec = float(cfg.get("yawn_duration_sec", 2.0))
        self.blink_max_frames = int(cfg.get("blink_max_frames", 8))
        self.blink_max_sec = float(cfg.get("blink_max_sec", 0.3))

        # EMA
        self.ema_alpha = float(cfg.get("ema_alpha", 0.4))

        # 帧阈值
        self._eye_frames_th = self._sec_to_frames(self.drowsy_duration_sec, self.consecutive_frames_eye)
        self._mouth_frames_th = self._sec_to_frames(self.yawn_duration_sec, self.consecutive_frames_mouth)
        self._blink_frames_th = self._sec_to_frames(self.blink_max_sec, self.blink_max_frames)

        # 计数器
        self._eye_low_frames = 0
        self._mouth_high_frames = 0
        self._blink_segment = 0
        self._eye_closed_since: Optional[float] = None
        self._mouth_open_since: Optional[float] = None
        self._blink_segment_since: Optional[float] = None
        self._ear_ema: Optional[float] = None
        self._mar_ema: Optional[float] = None
        self._is_drowsy = False
        self._is_yawning = False

        # ====== v4 新增: PERCLOS ======
        self.perclos_window_sec = float(cfg.get("perclos_window_sec", 60.0))
        self.perclos_threshold = float(cfg.get("perclos_threshold", 0.15))

        # PERCLOS 滑动窗口: 存储每帧是否闭眼 (1=闭眼, 0=睁眼)
        perclos_window_frames = self._sec_to_frames(
            self.perclos_window_sec,
            max(1, int(self.perclos_window_sec * 30))  # fallback: 假设30fps
        )
        self._perclos_buffer: deque = deque(maxlen=perclos_window_frames)
        self._perclos: float = 0.0
        self._is_perclos_fatigued: bool = False

        # ====== v4 新增: 眨眼频率 ======
        self.blink_freq_window_sec = float(cfg.get("blink_freq_window_sec", 60.0))
        self.blink_freq_high_threshold = float(cfg.get("blink_freq_high_threshold", 25.0))

        # 存储眨眼事件的时间戳
        self._blink_timestamps: deque = deque()
        self._blink_freq: float = 0.0
        self._is_blink_freq_high: bool = False

    def _sec_to_frames(self, duration_sec: float, fallback_frames: int) -> int:
        if self.fps and self.fps > 1.0 and duration_sec and duration_sec > 0:
            return max(1, int(round(duration_sec * self.fps)))
        return max(1, int(fallback_frames))

    def reset(self) -> None:
        self._eye_low_frames = 0
        self._mouth_high_frames = 0
        self._blink_segment = 0
        self._eye_closed_since = None
        self._mouth_open_since = None
        self._blink_segment_since = None
        self._ear_ema = None
        self._mar_ema = None
        self._is_drowsy = False
        self._is_yawning = False
        # v4
        self._perclos_buffer.clear()
        self._perclos = 0.0
        self._is_perclos_fatigued = False
        self._blink_timestamps.clear()
        self._blink_freq = 0.0
        self._is_blink_freq_high = False

    def _ema(self, prev: Optional[float], cur: float) -> float:
        if prev is None:
            return cur
        return (1 - self.ema_alpha) * prev + self.ema_alpha * cur

    def update(self, ear: float, mar: float) -> FatigueState:
        """每帧调用一次。"""
        ear = float(ear)
        mar = float(mar)
        now = time.time()

        # 1) 平滑
        self._ear_ema = self._ema(self._ear_ema, ear)
        self._mar_ema = self._ema(self._mar_ema, mar)

        ear_use = float(self._ear_ema)
        mar_use = float(self._mar_ema)

        # 2) 判断当前帧是否闭眼
        eye_closed = ear_use < self.ear_threshold

        # 3) 连续帧计数 + 眨眼检测（与原逻辑完全一致）
        blink = False

        if eye_closed:
            if self._eye_closed_since is None:
                self._eye_closed_since = now
            if self._blink_segment == 0:
                self._blink_segment_since = now
            self._eye_low_frames += 1
            self._blink_segment += 1
        else:
            blink_duration = 0.0
            if self._blink_segment_since is not None:
                blink_duration = now - self._blink_segment_since
            if (1 <= self._blink_segment
                    and (self._blink_segment <= self._blink_frames_th
                         or blink_duration <= self.blink_max_sec)):
                blink = True
            self._blink_segment = 0
            self._eye_low_frames = 0
            self._eye_closed_since = None
            self._blink_segment_since = None

        # 疲劳判定（连续闭眼）
        eye_closed_duration = 0.0
        if self._eye_closed_since is not None:
            eye_closed_duration = now - self._eye_closed_since
        if self.enable_drowsy:
            self._is_drowsy = (
                self._eye_low_frames >= self._eye_frames_th
                or (self.drowsy_duration_sec > 0 and eye_closed_duration >= self.drowsy_duration_sec)
            )
        else:
            self._is_drowsy = False

        # 哈欠判定
        if mar_use > self.mar_threshold:
            if self._mouth_open_since is None:
                self._mouth_open_since = now
            self._mouth_high_frames += 1
        else:
            self._mouth_high_frames = 0
            self._mouth_open_since = None

        mouth_open_duration = 0.0
        if self._mouth_open_since is not None:
            mouth_open_duration = now - self._mouth_open_since
        if self.enable_yawn:
            self._is_yawning = (
                self._mouth_high_frames >= self._mouth_frames_th
                or (self.yawn_duration_sec > 0 and mouth_open_duration >= self.yawn_duration_sec)
            )
        else:
            self._is_yawning = False

        # ====== v4: PERCLOS 计算 ======
        # 每帧推入 1(闭眼) 或 0(睁眼)，deque 自动淘汰旧帧
        self._perclos_buffer.append(1 if eye_closed else 0)

        if len(self._perclos_buffer) > 0:
            self._perclos = sum(self._perclos_buffer) / len(self._perclos_buffer)
        else:
            self._perclos = 0.0

        # 只有窗口积累到至少 30% 才开始判定（避免启动阶段误报）
        buffer_fill_ratio = len(self._perclos_buffer) / self._perclos_buffer.maxlen
        if buffer_fill_ratio >= 0.3:
            self._is_perclos_fatigued = self._perclos > self.perclos_threshold
        else:
            self._is_perclos_fatigued = False

        # ====== v4: 眨眼频率计算 ======
        if blink:
            self._blink_timestamps.append(now)

        # 清除窗口外的旧时间戳
        cutoff = now - self.blink_freq_window_sec
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        # 计算每分钟眨眼次数
        if self.blink_freq_window_sec > 0:
            # 实际覆盖的时间跨度
            if len(self._blink_timestamps) >= 2:
                actual_span = now - self._blink_timestamps[0]
                # 至少积累 10 秒数据才开始计算
                if actual_span >= 10.0:
                    self._blink_freq = len(self._blink_timestamps) / (actual_span / 60.0)
                else:
                    self._blink_freq = 0.0
            else:
                self._blink_freq = 0.0
        else:
            self._blink_freq = 0.0

        self._is_blink_freq_high = self._blink_freq > self.blink_freq_high_threshold

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
            # v4 新增
            perclos=round(self._perclos, 4),
            is_perclos_fatigued=self._is_perclos_fatigued,
            blink_freq=round(self._blink_freq, 1),
            is_blink_freq_high=self._is_blink_freq_high,
        )
