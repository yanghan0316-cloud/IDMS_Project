"""
src.core.risk_fusion

多模态风险融合引擎 (Multimodal Risk Fusion Engine)
===================================================

核心思想:
    传统方案中，舱内（疲劳/分心）和舱外（碰撞预警）是独立报警的，
    这意味着"驾驶员打瞌睡 + 前车 3 米"和"驾驶员打瞌睡 + 前方空旷"
    触发的警报完全相同。

    本模块将两路信号量化为 0~1 的连续风险分值，并通过交叉项实现
    跨模态风险放大：

        R_fused = w_ext * S_ext + w_int * S_int + w_cross * S_ext * S_int

    交叉项 (S_ext * S_int) 只有在两路同时出现风险时才会显著贡献，
    实现了"叠加危险指数级放大"的效果。

融合后的风险等级:
    LEVEL 0 - SAFE     (R < 0.25)  正常驾驶
    LEVEL 1 - LOW      (R < 0.50)  轻度风险（单路低危）
    LEVEL 2 - HIGH     (R < 0.75)  高风险（单路高危或双路中危）
    LEVEL 3 - CRITICAL (R >= 0.75) 极高风险（双路叠加）

输入:
    - vehicle_data: list[dict]  舱外碰撞预警模块的输出
    - face_data: dict           舱内 FaceMesh 检测器的输出

输出:
    - FusionResult: dataclass，包含各项分值、融合分值、融合等级

配置项 (config.yaml -> fusion):
    w_ext: 0.35          舱外权重
    w_int: 0.35          舱内权重
    w_cross: 0.30        交叉项权重（关键创新）
    level_thresholds: [0.25, 0.50, 0.75]

用法:
    fusion = RiskFusionEngine(config.get('fusion', {}))
    result = fusion.evaluate(vehicle_data, face_data)
    print(result.fused_score, result.fused_level, result.fused_text)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ==================== 输出数据结构 ====================

@dataclass
class FusionResult:
    """融合评估结果"""

    # 各通道的连续风险分值 (0.0 ~ 1.0)
    ext_score: float = 0.0          # 舱外风险分值
    int_score: float = 0.0          # 舱内风险分值
    cross_score: float = 0.0        # 交叉项分值 (ext * int)

    # 融合后的综合风险
    fused_score: float = 0.0        # 加权融合分值
    fused_level: int = 0            # 融合风险等级 (0~3)
    fused_text: str = "SAFE"        # 风险等级文字描述

    # 细分指标（用于 UI 显示和调参）
    ext_min_ttc: float = 99.0       # 舱外最小 TTC
    ext_min_dist: float = 99.0      # 舱外最近距离
    ext_max_level: int = 0          # 舱外最高警告级别

    int_drowsy: bool = False        # 是否疲劳
    int_yawning: bool = False       # 是否哈欠
    int_distracted: bool = False    # 是否分心
    int_nodding: bool = False       # 是否点头

    # 用于报警模块判断
    should_alert: bool = False      # 是否应该触发报警
    alert_urgency: str = "none"     # "none" / "normal" / "urgent" / "emergency"


# 融合风险等级常量
LEVEL_SAFE = 0
LEVEL_LOW = 1
LEVEL_HIGH = 2
LEVEL_CRITICAL = 3

LEVEL_TEXT = {
    0: "SAFE",
    1: "LOW",
    2: "HIGH",
    3: "CRITICAL",
}

ALERT_URGENCY = {
    0: "none",
    1: "normal",
    2: "urgent",
    3: "emergency",
}


# ==================== 融合引擎 ====================

class RiskFusionEngine:
    """
    多模态风险融合引擎

    将舱外碰撞预警信号和舱内驾驶员状态信号融合为统一的风险评分。
    """

    def __init__(self, config: Dict):
        cfg = config or {}

        # ====== 融合权重 ======
        self.w_ext = float(cfg.get("w_ext", 0.35))
        self.w_int = float(cfg.get("w_int", 0.35))
        self.w_cross = float(cfg.get("w_cross", 0.30))

        # ====== 风险等级阈值 ======
        thresholds = cfg.get("level_thresholds", [0.25, 0.50, 0.75])
        self.thresh_low = float(thresholds[0])
        self.thresh_high = float(thresholds[1])
        self.thresh_critical = float(thresholds[2])

        # ====== 舱外评分参数 ======
        # TTC 映射区间: TTC <= ttc_danger 时 score=1.0, TTC >= ttc_safe 时 score=0.0
        self.ttc_danger = float(cfg.get("ttc_danger", 1.5))
        self.ttc_safe = float(cfg.get("ttc_safe", 6.0))

        # 距离映射区间
        self.dist_danger = float(cfg.get("dist_danger", 3.0))
        self.dist_safe = float(cfg.get("dist_safe", 30.0))

        # ====== 舱内评分权重（各子指标） ======
        self.int_weights = {
            "drowsy": float(cfg.get("int_w_drowsy", 0.40)),
            "yawning": float(cfg.get("int_w_yawning", 0.15)),
            "distracted": float(cfg.get("int_w_distracted", 0.30)),
            "nodding": float(cfg.get("int_w_nodding", 0.15)),
        }

        # EAR 连续性评分：即使未触发 is_drowsy，EAR 接近阈值也应贡献部分分值
        self.ear_threshold = float(cfg.get("ear_threshold", 0.22))
        self.ear_safe = float(cfg.get("ear_safe", 0.30))

        # ====== EMA 平滑（防止融合分值抖动） ======
        self.ema_alpha = float(cfg.get("ema_alpha", 0.4))
        self._fused_ema: Optional[float] = None

    def evaluate(
        self,
        vehicle_data: Optional[List[Dict]] = None,
        face_data: Optional[Dict] = None,
    ) -> FusionResult:
        """
        执行一次融合评估。每帧调用一次。

        Args:
            vehicle_data: 舱外碰撞预警模块的输出列表
            face_data: 舱内 FaceMesh 检测器的输出字典

        Returns:
            FusionResult: 融合评估结果
        """
        result = FusionResult()

        # --- 1. 计算舱外风险分值 ---
        result.ext_score, ext_details = self._compute_ext_score(vehicle_data)
        result.ext_min_ttc = ext_details["min_ttc"]
        result.ext_min_dist = ext_details["min_dist"]
        result.ext_max_level = ext_details["max_level"]

        # --- 2. 计算舱内风险分值 ---
        result.int_score, int_details = self._compute_int_score(face_data)
        result.int_drowsy = int_details["drowsy"]
        result.int_yawning = int_details["yawning"]
        result.int_distracted = int_details["distracted"]
        result.int_nodding = int_details["nodding"]

        # --- 3. 计算交叉项 ---
        result.cross_score = result.ext_score * result.int_score

        # --- 4. 加权融合 ---
        raw_fused = (
            self.w_ext * result.ext_score
            + self.w_int * result.int_score
            + self.w_cross * result.cross_score
        )
        # 钳位到 [0, 1]
        raw_fused = max(0.0, min(1.0, raw_fused))

        # EMA 平滑
        if self._fused_ema is None:
            self._fused_ema = raw_fused
        else:
            self._fused_ema = (
                self.ema_alpha * raw_fused
                + (1 - self.ema_alpha) * self._fused_ema
            )

        result.fused_score = round(float(self._fused_ema), 4)

        # --- 5. 风险分级 ---
        result.fused_level = self._classify_level(result.fused_score)
        result.fused_text = LEVEL_TEXT[result.fused_level]

        # --- 6. 报警决策 ---
        result.alert_urgency = ALERT_URGENCY[result.fused_level]
        result.should_alert = result.fused_level >= LEVEL_HIGH

        return result

    # ------------------------------------------------------------------
    #  舱外风险量化
    # ------------------------------------------------------------------

    def _compute_ext_score(self, vehicle_data: Optional[List[Dict]]) -> Tuple[float, Dict]:
        """
        将舱外检测结果量化为 0~1 的风险分值。

        策略: 取所有检测目标中风险最高的那个（最小 TTC / 最近距离）。
        """
        details = {"min_ttc": 99.0, "min_dist": 99.0, "max_level": 0}

        if not vehicle_data:
            return 0.0, details

        max_score = 0.0

        for obj in vehicle_data:
            ttc = obj.get("ttc", 99.0)
            dist = obj.get("distance", 99.0)
            level = obj.get("warning_level", 0)
            lane_rel = obj.get("lane_relevance", 1.0)

            # 更新统计
            if ttc < details["min_ttc"]:
                details["min_ttc"] = ttc
            if 0 < dist < details["min_dist"]:
                details["min_dist"] = dist
            if level > details["max_level"]:
                details["max_level"] = level

            # TTC 分值: 线性映射 [ttc_safe, ttc_danger] -> [0, 1]
            ttc_score = 0.0
            if ttc < self.ttc_safe:
                ttc_score = 1.0 - (ttc - self.ttc_danger) / (self.ttc_safe - self.ttc_danger)
                ttc_score = max(0.0, min(1.0, ttc_score))

            # 距离分值
            dist_score = 0.0
            if 0 < dist < self.dist_safe:
                dist_score = 1.0 - (dist - self.dist_danger) / (self.dist_safe - self.dist_danger)
                dist_score = max(0.0, min(1.0, dist_score))

            # 取 TTC 和距离中更危险的那个
            obj_score = max(ttc_score, dist_score)

            # 考虑车道相关性衰减
            obj_score *= lane_rel

            # 离散 warning_level 的保底分值
            # 确保 DANGER 级别的目标至少有 0.7 分
            level_floor = {0: 0.0, 1: 0.3, 2: 0.7}
            obj_score = max(obj_score, level_floor.get(level, 0.0))

            max_score = max(max_score, obj_score)

        return round(max_score, 4), details

    # ------------------------------------------------------------------
    #  舱内风险量化
    # ------------------------------------------------------------------

    def _compute_int_score(self, face_data: Optional[Dict]) -> Tuple[float, Dict]:
        """
        将舱内驾驶员状态量化为 0~1 的风险分值。

        策略: 加权组合各项布尔状态 + EAR 连续性补偿。
        """
        details = {
            "drowsy": False,
            "yawning": False,
            "distracted": False,
            "nodding": False,
        }

        if not face_data or not face_data.get("has_face"):
            return 0.0, details

        # 提取布尔状态
        is_drowsy = bool(face_data.get("is_drowsy", False))
        is_yawning = bool(face_data.get("is_yawning", False))
        is_distracted = bool(face_data.get("is_distracted", False))
        is_nodding = bool(face_data.get("is_nodding", False))

        details["drowsy"] = is_drowsy
        details["yawning"] = is_yawning
        details["distracted"] = is_distracted
        details["nodding"] = is_nodding

        # 加权求和（布尔部分）
        bool_score = 0.0
        if is_drowsy:
            bool_score += self.int_weights["drowsy"]
        if is_yawning:
            bool_score += self.int_weights["yawning"]
        if is_distracted:
            bool_score += self.int_weights["distracted"]
        if is_nodding:
            bool_score += self.int_weights["nodding"]

        # EAR 连续性补偿:
        # 即使 is_drowsy 还没触发（闭眼时间不够长），但 EAR 已经很低了，
        # 应该提前贡献部分分值，让融合分值提前上升。
        ear = float(face_data.get("ear", 0.3))
        ear_contrib = 0.0
        if ear < self.ear_safe:
            # 线性映射 [ear_threshold, ear_safe] -> [0.3, 0]
            ear_contrib = (self.ear_safe - ear) / (self.ear_safe - self.ear_threshold)
            ear_contrib = max(0.0, min(0.3, ear_contrib * 0.3))

        # 偏航角连续性补偿（类似 EAR 的思路）
        yaw = abs(float(face_data.get("yaw", 0.0)))
        yaw_contrib = 0.0
        if yaw > 10.0:  # 10° 以上开始有微弱贡献
            yaw_contrib = min(0.2, (yaw - 10.0) / 40.0 * 0.2)

        # 综合（布尔状态 + 连续性补偿，钳位到 1.0）
        total = min(1.0, bool_score + ear_contrib + yaw_contrib)

        return round(total, 4), details

    # ------------------------------------------------------------------
    #  风险分级
    # ------------------------------------------------------------------

    def _classify_level(self, score: float) -> int:
        """将融合分值映射到离散风险等级"""
        if score >= self.thresh_critical:
            return LEVEL_CRITICAL
        elif score >= self.thresh_high:
            return LEVEL_HIGH
        elif score >= self.thresh_low:
            return LEVEL_LOW
        else:
            return LEVEL_SAFE

    def reset(self) -> None:
        """重置 EMA 状态（用于重新初始化）"""
        self._fused_ema = None