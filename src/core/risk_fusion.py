"""
src.core.risk_fusion  (v5)

多模态风险融合引擎 — 集成 DriverStateAssessor 协同评估

v5 变化:
    - 舱内 int_score 不再来自简单布尔加权，
      而是由 DriverStateAssessor 的信号互相印证逻辑计算。
    - FusionResult 新增 int_fatigue_score / int_attention_score /
      int_confidence_label 等调试字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.internal.driver_state import DriverStateAssessor


@dataclass
class FusionResult:
    ext_score: float = 0.0
    int_score: float = 0.0
    cross_score: float = 0.0

    fused_score: float = 0.0
    fused_level: int = 0
    fused_text: str = "SAFE"

    ext_min_ttc: float = 99.0
    ext_min_dist: float = 99.0
    ext_max_level: int = 0

    # v5: 驾驶员状态细分
    int_fatigue_score: float = 0.0
    int_attention_score: float = 0.0
    int_fatigue_signals: int = 0
    int_confidence_label: str = "none"
    int_has_contradiction: bool = False

    int_drowsy: bool = False
    int_yawning: bool = False
    int_distracted: bool = False
    int_nodding: bool = False

    should_alert: bool = False
    alert_urgency: str = "none"


LEVEL_SAFE, LEVEL_LOW, LEVEL_HIGH, LEVEL_CRITICAL = 0, 1, 2, 3
LEVEL_TEXT = {0: "SAFE", 1: "LOW", 2: "HIGH", 3: "CRITICAL"}
ALERT_URGENCY = {0: "none", 1: "normal", 2: "urgent", 3: "emergency"}


class RiskFusionEngine:
    def __init__(self, config: Dict):
        cfg = config or {}
        self.w_ext = float(cfg.get("w_ext", 0.35))
        self.w_int = float(cfg.get("w_int", 0.35))
        self.w_cross = float(cfg.get("w_cross", 0.30))

        thresholds = cfg.get("level_thresholds", [0.25, 0.50, 0.75])
        if len(thresholds) != 3:
            thresholds = [0.25, 0.50, 0.75]
        self.thresh_low = float(thresholds[0])
        self.thresh_high = float(thresholds[1])
        self.thresh_critical = float(thresholds[2])

        self.ttc_danger = float(cfg.get("ttc_danger", 1.5))
        self.ttc_safe = float(cfg.get("ttc_safe", 6.0))
        self.dist_danger = float(cfg.get("dist_danger", 3.0))
        self.dist_safe = float(cfg.get("dist_safe", 30.0))

        # v5: 协同评估器
        self._driver_assessor = DriverStateAssessor(cfg)

        self.ema_alpha = float(cfg.get("ema_alpha", 0.4))
        self._fused_ema: Optional[float] = None

    def evaluate(
        self,
        vehicle_data: Optional[List[Dict]] = None,
        face_data: Optional[Dict] = None,
    ) -> FusionResult:
        result = FusionResult()

        # 1. 舱外
        result.ext_score, ext_d = self._compute_ext_score(vehicle_data)
        result.ext_min_ttc = ext_d["min_ttc"]
        result.ext_min_dist = ext_d["min_dist"]
        result.ext_max_level = ext_d["max_level"]

        # 2. 舱内 (v5: DriverStateAssessor)
        ds = self._driver_assessor.evaluate(face_data)
        result.int_score = ds.driver_risk
        result.int_fatigue_score = ds.fatigue_score
        result.int_attention_score = ds.attention_score
        result.int_fatigue_signals = ds.fatigue_signals
        result.int_confidence_label = ds.confidence_label
        result.int_has_contradiction = ds.has_contradiction

        if face_data and face_data.get("has_face"):
            result.int_drowsy = bool(face_data.get("is_drowsy"))
            result.int_yawning = bool(face_data.get("is_yawning"))
            result.int_distracted = bool(face_data.get("is_distracted"))
            result.int_nodding = bool(face_data.get("is_nodding"))

        # 3. 交叉项
        result.cross_score = result.ext_score * result.int_score

        # 4. 融合
        raw = (self.w_ext * result.ext_score
               + self.w_int * result.int_score
               + self.w_cross * result.cross_score)
        raw = max(0.0, min(1.0, raw))

        if self._fused_ema is None:
            self._fused_ema = raw
        else:
            self._fused_ema = self.ema_alpha * raw + (1 - self.ema_alpha) * self._fused_ema

        result.fused_score = round(float(self._fused_ema), 4)

        # 5. 分级
        result.fused_level = self._classify(result.fused_score)
        result.fused_text = LEVEL_TEXT[result.fused_level]
        result.alert_urgency = ALERT_URGENCY[result.fused_level]
        result.should_alert = result.fused_level >= LEVEL_HIGH
        return result

    # ---------- 舱外评分 ----------
    def _compute_ext_score(self, vd: Optional[List[Dict]]) -> Tuple[float, Dict]:
        d = {"min_ttc": 99.0, "min_dist": 99.0, "max_level": 0}
        if not vd:
            return 0.0, d
        mx = 0.0
        for o in vd:
            ttc, dist, lv = o.get("ttc", 99.0), o.get("distance", 99.0), o.get("warning_level", 0)
            lr = o.get("lane_relevance", 1.0)
            if ttc < d["min_ttc"]: d["min_ttc"] = ttc
            if 0 < dist < d["min_dist"]: d["min_dist"] = dist
            if lv > d["max_level"]: d["max_level"] = lv
            ts = max(0.0, min(1.0, 1.0 - (ttc - self.ttc_danger) / (self.ttc_safe - self.ttc_danger))) if ttc < self.ttc_safe else 0.0
            ds = max(0.0, min(1.0, 1.0 - (dist - self.dist_danger) / (self.dist_safe - self.dist_danger))) if 0 < dist < self.dist_safe else 0.0
            s = max(ts, ds) * lr
            s = max(s, {0: 0.0, 1: 0.3, 2: 0.7}.get(lv, 0.0))
            mx = max(mx, s)
        return round(mx, 4), d

    def _classify(self, s: float) -> int:
        if s >= self.thresh_critical: return LEVEL_CRITICAL
        if s >= self.thresh_high: return LEVEL_HIGH
        if s >= self.thresh_low: return LEVEL_LOW
        return LEVEL_SAFE

    def reset(self):
        self._fused_ema = None
