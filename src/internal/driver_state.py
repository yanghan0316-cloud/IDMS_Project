"""
src.internal.driver_state

驾驶员状态综合评估器 (Driver State Assessor)
=============================================

核心思想:
    不再把 PERCLOS、点头、眨眼频率、分心等当作独立布尔报警，
    而是引入"信号互相印证"（corroboration）机制：

    - 多个疲劳信号同时出现 → 可信度倍增（互相印证）
    - 矛盾信号同时出现 → 可信度打折（可能是传感器误差）
    - 单一信号孤立出现 → 中等可信度（需要观察但不急报警）

    这解决了一个真实场景中的关键问题：
    PERCLOS 单独高可能是逆光/墨镜/眼型导致 EAR 偏低，
    但如果同时还在频繁点头、眨眼加快，那疲劳就几乎是确定的。

输出两个维度:
    - fatigue_score:  0~1，疲劳可信度（综合 PERCLOS + 点头 + 眨眼频率 + 闭眼）
    - attention_score: 0~1，注意力缺失度（综合分心 + 点头 + 哈欠）

协同规则（可在 config.yaml 中调节）:
    corroboration_boost: 1.5   多信号印证时的放大倍数
    contradiction_penalty: 0.5 矛盾信号的衰减倍数
    single_signal_cap: 0.6     单信号孤立出现时的上限

用法:
    assessor = DriverStateAssessor(config)
    result = assessor.evaluate(face_data)
    # result.fatigue_score, result.attention_score, result.driver_risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DriverState:
    """驾驶员状态评估结果"""
    fatigue_score: float = 0.0       # 疲劳可信度 0~1
    attention_score: float = 0.0     # 注意力缺失度 0~1
    driver_risk: float = 0.0         # 综合驾驶员风险 0~1

    # 调试信息
    fatigue_signals: int = 0         # 同时触发的疲劳信号数量
    attention_signals: int = 0       # 同时触发的注意力信号数量
    has_contradiction: bool = False  # 是否检测到矛盾信号
    confidence_label: str = "none"   # "none"/"single"/"corroborated"/"contradicted"


class DriverStateAssessor:
    """
    基于信号互相印证的驾驶员状态评估器。

    替代 RiskFusionEngine 中原来的 _compute_int_score 简单加权逻辑。
    """

    def __init__(self, config: Dict):
        cfg = config or {}

        # ====== 协同评估参数 ======
        # 多信号印证时的放大倍数 (>1.0 表示多信号比单信号可信得多)
        self.corroboration_boost = float(cfg.get("corroboration_boost", 1.5))
        # 矛盾信号的衰减倍数 (<1.0 表示矛盾时降低可信度)
        self.contradiction_penalty = float(cfg.get("contradiction_penalty", 0.5))
        # 单一信号孤立出现时的得分上限
        self.single_signal_cap = float(cfg.get("single_signal_cap", 0.6))

        # ====== 连续指标阈值 ======
        self.ear_threshold = float(cfg.get("ear_threshold", 0.22))
        self.ear_safe = float(cfg.get("ear_safe", 0.30))
        self.perclos_threshold = float(cfg.get("perclos_threshold", 0.15))
        self.blink_freq_normal = float(cfg.get("blink_freq_normal", 20.0))
        self.blink_freq_high = float(cfg.get("blink_freq_high_threshold", 25.0))
        self.yaw_threshold = float(cfg.get("distraction_yaw_threshold_deg", 20.0))
        self.yaw_safe = float(cfg.get("distraction_yaw_safe_deg", max(0.0, self.yaw_threshold * 0.5)))
        self.yaw_danger = float(cfg.get("distraction_yaw_danger_deg", max(self.yaw_threshold + 10.0, 40.0)))
        if self.yaw_danger <= self.yaw_safe:
            self.yaw_danger = self.yaw_safe + 1.0

        # ====== 疲劳维度各指标基础权重 ======
        self.w_perclos = float(cfg.get("w_perclos", 0.30))
        self.w_drowsy = float(cfg.get("w_drowsy", 0.25))
        self.w_nodding = float(cfg.get("w_nodding", 0.20))
        self.w_blink_freq = float(cfg.get("w_blink_freq", 0.10))
        self.w_yawn = float(cfg.get("w_yawn", 0.15))

        # ====== 注意力维度各指标基础权重 ======
        self.w_distracted = float(cfg.get("w_distracted", 0.50))
        self.w_nod_attention = float(cfg.get("w_nod_attention", 0.25))
        self.w_yawn_attention = float(cfg.get("w_yawn_attention", 0.25))

        # ====== 两维度合成驾驶员总风险的权重 ======
        self.fatigue_weight = float(cfg.get("fatigue_weight", 0.55))
        self.attention_weight = float(cfg.get("attention_weight", 0.45))

    def evaluate(self, face_data: Optional[Dict]) -> DriverState:
        """
        评估驾驶员状态。每帧调用一次。

        Args:
            face_data: FaceMeshDetector.process() 的输出 dict

        Returns:
            DriverState: 包含疲劳可信度、注意力缺失度、综合风险
        """
        result = DriverState()

        if not face_data or not face_data.get("has_face"):
            return result

        # ============================================================
        # 第一步: 提取原始信号并量化为连续分值
        # ============================================================

        # --- 疲劳相关信号 ---
        perclos = float(face_data.get("perclos", 0.0))
        is_drowsy = bool(face_data.get("is_drowsy", False))
        is_nodding = bool(face_data.get("is_nodding", False))
        is_yawning = bool(face_data.get("is_yawning", False))
        blink_freq = float(face_data.get("blink_freq", 0.0))
        ear = float(face_data.get("ear", 0.3))

        # --- 注意力相关信号 ---
        is_distracted = bool(face_data.get("is_distracted", False))
        yaw = abs(float(face_data.get("yaw", 0.0)))

        # PERCLOS 连续分值: 线性映射到 [0, 1]
        # 0% → 0.0,  15%(阈值) → 0.6,  30%+ → 1.0
        if perclos <= 0:
            perclos_score = 0.0
        elif perclos >= 0.30:
            perclos_score = 1.0
        else:
            perclos_score = min(1.0, perclos / 0.30)

        # EAR 接近度: EAR 越低于安全值，分值越高
        if ear >= self.ear_safe:
            ear_proximity = 0.0
        elif ear <= self.ear_threshold:
            ear_proximity = 1.0
        else:
            ear_proximity = (self.ear_safe - ear) / (self.ear_safe - self.ear_threshold)

        # 眨眼频率连续分值: 正常20次/min以下=0，25+=0.8，35+=1.0
        if blink_freq <= self.blink_freq_normal:
            blink_score = 0.0
        elif blink_freq >= 35.0:
            blink_score = 1.0
        else:
            blink_score = (blink_freq - self.blink_freq_normal) / (35.0 - self.blink_freq_normal)

        # 偏航角连续分值
        if yaw <= self.yaw_safe:
            yaw_score = 0.0
        elif yaw >= self.yaw_danger:
            yaw_score = 1.0
        else:
            yaw_score = (yaw - self.yaw_safe) / (self.yaw_danger - self.yaw_safe)

        # ============================================================
        # 第二步: 计算疲劳信号互相印证
        # ============================================================

        # 统计触发的疲劳布尔信号数量
        fatigue_bools = [
            perclos > self.perclos_threshold,  # PERCLOS 超阈值
            is_drowsy,                          # 连续闭眼
            is_nodding,                         # 持续点头
            blink_freq > self.blink_freq_high,  # 高频眨眼
            is_yawning,                         # 哈欠
        ]
        fatigue_signal_count = sum(fatigue_bools)
        result.fatigue_signals = fatigue_signal_count

        # 基础疲劳分值: 加权组合连续分值 + 布尔分值
        raw_fatigue = (
            self.w_perclos * perclos_score
            + self.w_drowsy * max(ear_proximity, float(is_drowsy))
            + self.w_nodding * float(is_nodding)
            + self.w_blink_freq * blink_score
            + self.w_yawn * float(is_yawning)
        )

        # 互相印证调节
        if fatigue_signal_count >= 3:
            # 3+ 个疲劳信号同时触发 → 强烈印证，放大
            raw_fatigue *= self.corroboration_boost
            result.confidence_label = "corroborated"
        elif fatigue_signal_count == 2:
            # 2 个信号印证 → 温和放大
            raw_fatigue *= (1.0 + (self.corroboration_boost - 1.0) * 0.5)
            result.confidence_label = "corroborated"
        elif fatigue_signal_count == 1:
            # 只有 1 个信号 → 封顶，防止单一故障导致误报
            raw_fatigue = min(raw_fatigue, self.single_signal_cap)
            result.confidence_label = "single"
        else:
            result.confidence_label = "none"

        # ============================================================
        # 第三步: 检测矛盾信号
        # ============================================================

        # 矛盾情况 1: 说"疲劳"又说"分心"
        # 真正昏昏欲睡的人不太可能同时大幅转头（除非是打瞌睡后惊醒的短暂反应）
        # 如果 PERCLOS 高但同时持续大幅转头 → 可能是 EAR 被头部角度干扰
        if (perclos > self.perclos_threshold and is_distracted
                and not is_nodding and not is_drowsy):
            # PERCLOS 高 + 分心，但没有其他疲劳佐证 → 可能是传感器噪声
            raw_fatigue *= self.contradiction_penalty
            result.has_contradiction = True
            result.confidence_label = "contradicted"

        # 矛盾情况 2: 同时闭眼（drowsy）和高频眨眼
        # 如果眼睛一直闭着，怎么可能还在频繁眨眼？说明检测链路有问题
        if is_drowsy and blink_freq > self.blink_freq_high:
            # 信号矛盾，以更强的信号（drowsy）为准，忽略 blink_freq 贡献
            raw_fatigue -= self.w_blink_freq * blink_score * 0.5

        result.fatigue_score = round(max(0.0, min(1.0, raw_fatigue)), 4)

        # ============================================================
        # 第四步: 计算注意力缺失分值
        # ============================================================

        attention_bools = [is_distracted, is_nodding, is_yawning]
        attention_signal_count = sum(attention_bools)
        result.attention_signals = attention_signal_count

        raw_attention = (
            self.w_distracted * max(yaw_score, float(is_distracted))
            + self.w_nod_attention * float(is_nodding)
            + self.w_yawn_attention * float(is_yawning)
        )

        # 注意力信号的互相印证
        if attention_signal_count >= 2:
            raw_attention *= (1.0 + (self.corroboration_boost - 1.0) * 0.5)
        elif attention_signal_count == 1:
            raw_attention = min(raw_attention, self.single_signal_cap)

        result.attention_score = round(max(0.0, min(1.0, raw_attention)), 4)

        # ============================================================
        # 第五步: 合成驾驶员综合风险
        # ============================================================
        # 取两个维度的加权和，但也考虑"两维度同时高"的叠加效应
        combined = (
            self.fatigue_weight * result.fatigue_score
            + self.attention_weight * result.attention_score
        )
        # 如果两个维度都较高，额外叠加（类似舱内外融合的交叉项思路）
        if result.fatigue_score > 0.3 and result.attention_score > 0.3:
            cross_boost = result.fatigue_score * result.attention_score * 0.2
            combined += cross_boost

        result.driver_risk = round(max(0.0, min(1.0, combined)), 4)

        return result


