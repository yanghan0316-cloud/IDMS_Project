"""
src.internal.attention_logic

把"头部姿态 -> 分心/点头"封装成一个小状态机。

为什么单独拆出来？
- 疲劳（闭眼/哈欠）主要看 EAR/MAR。
- 分心/点头主要看 yaw/pitch（头部姿态）。

这样做的好处：后续你们想把"视线估计""手机检测"等加进来时，逻辑不会和 EAR/MAR 绞在一起。

配置项（来自 config.yaml 的 internal 部分）：

- fps: 摄像头 FPS（填 0 或不填表示未知；会退化为直接使用 *frames 阈值）

- distraction_yaw_threshold_deg: 触发分心的 yaw 角阈值（建议先用 30 度）
- distraction_yaw_release_deg: 解除分心的 yaw 阈值（迟滞，建议略小于触发阈值）
- distraction_duration_sec / distraction_frames: 持续时间（秒）或帧数
- distraction_grace_frames: yaw 短暂跌落时的容忍帧数（解决极端转头时 PnP 不稳定问题）

- nod_pitch_threshold_deg: 触发点头/低头的 pitch 角阈值
    注意：pitch 的正负号与相机/solvePnP 的坐标系有关。
- nod_pitch_release_deg: 解除阈值（迟滞）
- nod_duration_sec / nod_frames: 持续时间（秒）或帧数

输出：
- is_distracted: bool
- is_nodding: bool
- yaw_ema / pitch_ema: 平滑后的角度（可用于 UI 显示）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AttentionState:
    yaw: float
    pitch: float
    yaw_ema: float
    pitch_ema: float
    is_distracted: bool
    is_nodding: bool
    distracted_frames: int
    nod_frames: int


class AttentionAnalyzer:
    def __init__(self, config: Dict):
        cfg = config or {}

        # ====== FPS / 时长转换 ======
        self.fps = float(cfg.get("fps", 0.0) or 0.0)

        # ====== 分心 (Yaw) ======
        self.distraction_yaw_threshold_deg = float(cfg.get("distraction_yaw_threshold_deg", 30.0))
        self.distraction_yaw_release_deg = float(cfg.get("distraction_yaw_release_deg", 20.0))

        self.distraction_duration_sec = float(cfg.get("distraction_duration_sec", 1.0))
        self.distraction_frames = int(cfg.get("distraction_frames", 30))

        # v3 新增: Grace period —— 当 yaw 在阈值附近短暂跌落时，不立即清零计数器。
        # 解决的问题: 转头角度很大时，MediaPipe 关键点质量下降，PnP 解算的 yaw 会
        # 突然跳低甚至翻转，导致明明在大幅转头却无法触发分心报警。
        # 原理: 允许最多 grace_frames 帧的 yaw 低于阈值而不重置计数器。
        self.distraction_grace_frames = int(cfg.get("distraction_grace_frames", 8))

        # ====== 点头/低头 (Pitch) ======
        # v3 调整: 默认阈值更严格 (-25 → -30)，持续时间更长 (1.0s → 2.0s)
        self.nod_pitch_threshold_deg = float(cfg.get("nod_pitch_threshold_deg", -30.0))
        self.nod_pitch_release_deg = float(cfg.get("nod_pitch_release_deg", -18.0))

        self.nod_duration_sec = float(cfg.get("nod_duration_sec", 2.0))
        self.nod_frames = int(cfg.get("nod_frames", 60))

        # ====== 平滑（角度很抖，建议 EMA） ======
        self.ema_alpha = float(cfg.get("pose_ema_alpha", 0.35))

        # 计数器
        self._distracted_cnt = 0
        self._nod_cnt = 0

        # v3 新增: grace 计数器（yaw 低于阈值时累计，超过 grace 上限才清零 distracted_cnt）
        self._yaw_grace_cnt = 0

        # EMA
        self._yaw_ema: Optional[float] = None
        self._pitch_ema: Optional[float] = None

        # 输出状态
        self._is_distracted = False
        self._is_nodding = False

        # 把"秒"转换成"帧数"
        self._distraction_frames_th = self._sec_to_frames(self.distraction_duration_sec, self.distraction_frames)
        self._nod_frames_th = self._sec_to_frames(self.nod_duration_sec, self.nod_frames)

    def _sec_to_frames(self, duration_sec: float, fallback_frames: int) -> int:
        if self.fps and self.fps > 1.0 and duration_sec and duration_sec > 0:
            return max(1, int(round(duration_sec * self.fps)))
        return max(1, int(fallback_frames))

    def reset(self) -> None:
        self._distracted_cnt = 0
        self._nod_cnt = 0
        self._yaw_grace_cnt = 0
        self._yaw_ema = None
        self._pitch_ema = None
        self._is_distracted = False
        self._is_nodding = False

    def _ema(self, prev: Optional[float], cur: float) -> float:
        if prev is None:
            return cur
        return (1 - self.ema_alpha) * prev + self.ema_alpha * cur

    def update(self, yaw: float, pitch: float) -> AttentionState:
        yaw = float(yaw)
        pitch = float(pitch)

        # 平滑
        self._yaw_ema = self._ema(self._yaw_ema, yaw)
        self._pitch_ema = self._ema(self._pitch_ema, pitch)

        yaw_use = float(self._yaw_ema)
        pitch_use = float(self._pitch_ema)

        # ============ 分心判定（abs(yaw) 超阈值持续） ============
        # v3: 加入 grace period，解决极端转头时 PnP yaw 跳变的问题
        if abs(yaw_use) > self.distraction_yaw_threshold_deg:
            # yaw 在阈值以上：正常累计，grace 清零
            self._distracted_cnt += 1
            self._yaw_grace_cnt = 0
        else:
            # yaw 低于阈值
            if self._distracted_cnt > 0:
                # 之前有累计 —— 进入 grace 容忍期
                self._yaw_grace_cnt += 1
                if self._yaw_grace_cnt <= self.distraction_grace_frames:
                    # 还在 grace 期内：保持计数器不变（也不增加，只是不清零）
                    pass
                else:
                    # grace 期耗尽：需要判断是否用迟滞释放
                    if self._is_distracted and abs(yaw_use) > self.distraction_yaw_release_deg:
                        # 已处于分心状态且 yaw 仍高于释放阈值 → 保持
                        pass
                    else:
                        # 真正恢复正常 → 清零
                        self._distracted_cnt = 0
                        self._yaw_grace_cnt = 0
            else:
                # 之前没有累计，无需 grace
                self._yaw_grace_cnt = 0

        self._is_distracted = self._distracted_cnt >= self._distraction_frames_th

        # ============ 点头/低头判定（pitch 过阈值持续） ============
        if pitch_use < self.nod_pitch_threshold_deg:
            self._nod_cnt += 1
        else:
            if self._is_nodding and pitch_use < self.nod_pitch_release_deg:
                pass
            else:
                self._nod_cnt = 0

        self._is_nodding = self._nod_cnt >= self._nod_frames_th

        return AttentionState(
            yaw=yaw,
            pitch=pitch,
            yaw_ema=yaw_use,
            pitch_ema=pitch_use,
            is_distracted=self._is_distracted,
            is_nodding=self._is_nodding,
            distracted_frames=self._distracted_cnt,
            nod_frames=self._nod_cnt,
        )