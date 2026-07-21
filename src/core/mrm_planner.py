from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ACTION_KEEP = "KEEP"
ACTION_SLOW_DOWN = "SLOW_DOWN"
ACTION_BRAKE = "BRAKE"
ACTION_EMERGENCY_BRAKE = "EMERGENCY_BRAKE"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CandidateAction:
    name: str
    target_decel: float
    comfort_cost: float
    rank: int
    response_delay_sec: float = 0.15


@dataclass
class WorldState:
    timestamp: float = 0.0

    # A valid empty external observation means "road observed, no target".
    # False means the external perception channel is unavailable or stale and
    # must never be interpreted as evidence that the road is clear.
    external_perception_valid: bool = True
    external_perception_age_sec: float = 0.0
    external_state_held: bool = False

    ext_score: float = 0.0
    int_score: float = 0.0
    cross_score: float = 0.0
    fused_score: float = 0.0
    fused_level: int = 0
    fused_text: str = "SAFE"

    has_lead_vehicle: bool = False
    lead_distance: float = 99.0
    lead_ttc: float = 99.0
    closing_speed: float = 0.0
    lane_relevance: float = 1.0
    lead_warning_level: int = 0
    lead_class_name: str = ""

    fatigue_score: float = 0.0
    attention_score: float = 0.0
    driver_flags: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RiskPrediction:
    action: CandidateAction
    peak_risk: float
    terminal_risk: float
    min_distance: float
    min_ttc: float
    trajectory: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class CandidateEvaluation:
    action: CandidateAction
    prediction: RiskPrediction
    cost: float
    terms: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionResult:
    timestamp: float
    action: str
    target_decel: float
    cost: float
    predicted_risk: float
    predicted_min_distance: float
    predicted_min_ttc: float
    reasons: List[str]
    world_state: WorldState
    candidates: List[CandidateEvaluation] = field(default_factory=list)

    prediction_source: str = "kinematic"
    model_uncertainty: float = 0.0

class CandidateActions:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        decel_cfg = cfg.get("decelerations", {}) or {}

        self._actions = [
            CandidateAction(
                ACTION_KEEP,
                _as_float(decel_cfg.get(ACTION_KEEP), 0.0),
                _as_float(cfg.get("keep_cost"), 0.00),
                0,
                _as_float(cfg.get("keep_delay_sec"), 0.0),
            ),
            CandidateAction(
                ACTION_SLOW_DOWN,
                _as_float(decel_cfg.get(ACTION_SLOW_DOWN), 1.5),
                _as_float(cfg.get("slow_down_cost"), 0.04),
                1,
                _as_float(cfg.get("slow_down_delay_sec"), 0.20),
            ),
            CandidateAction(
                ACTION_BRAKE,
                _as_float(decel_cfg.get(ACTION_BRAKE), 3.5),
                _as_float(cfg.get("brake_cost"), 0.12),
                2,
                _as_float(cfg.get("brake_delay_sec"), 0.15),
            ),
            CandidateAction(
                ACTION_EMERGENCY_BRAKE,
                _as_float(decel_cfg.get(ACTION_EMERGENCY_BRAKE), 6.5),
                _as_float(cfg.get("emergency_brake_cost"), 0.32),
                3,
                _as_float(cfg.get("emergency_brake_delay_sec"), 0.10),
            ),
        ]
        self._validate(cfg)

    def _validate(self, config: Dict[str, Any]) -> None:
        decelerations = [action.target_decel for action in self._actions]
        delays = [action.response_delay_sec for action in self._actions]
        comfort_costs = [action.comfort_cost for action in self._actions]
        if abs(decelerations[0]) > 1e-6:
            raise ValueError("KEEP deceleration must be exactly 0")
        if any(value < 0.0 for value in decelerations):
            raise ValueError("candidate decelerations must be non-negative")
        if any(
            right <= left
            for left, right in zip(decelerations, decelerations[1:])
        ):
            raise ValueError(
                "candidate decelerations must strictly increase from KEEP to EMERGENCY_BRAKE"
            )
        if any(value < 0.0 for value in comfort_costs):
            raise ValueError("candidate comfort costs must be non-negative")
        if any(
            right < left for left, right in zip(comfort_costs, comfort_costs[1:])
        ):
            raise ValueError("candidate comfort costs must not decrease with braking rank")


        rssm_cfg = config.get("rssm", {}) or {}
        max_decel = _as_float(rssm_cfg.get("max_decel"), 8.0)
        max_delay = _as_float(rssm_cfg.get("max_delay"), 0.5)
        if decelerations[-1] > max_decel:
            raise ValueError("candidate deceleration exceeds rssm.max_decel")
        if any(delay < 0.0 or delay > max_delay for delay in delays):
            raise ValueError("candidate response delay is outside [0, rssm.max_delay]")
        if str(config.get("predictor", "kinematic")).lower() == "hybrid":
            dt_sec = _as_float(rssm_cfg.get("dt_sec"), 0.25)
            if any(delay > dt_sec + 1e-6 for delay in delays):
                raise ValueError(
                    "hybrid RSSM requires candidate response delays <= rssm.dt_sec"
                )


    def generate(self, state: WorldState) -> List[CandidateAction]:
        return list(self._actions)


class RiskPredictor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.max_horizon_sec = _as_float(cfg.get("max_horizon_sec"), 10.0)
        if self.max_horizon_sec <= 0.0:
            raise ValueError("risk predictor max_horizon_sec must be positive")
        horizons = cfg.get("horizons_sec", [0.5, 1.0, 1.5])
        if not isinstance(horizons, (list, tuple, set)):
            raise ValueError("risk predictor horizons_sec must be a sequence")
        parsed_horizons = []
        for raw_horizon in horizons:
            try:
                horizon = float(raw_horizon)
            except (TypeError, ValueError) as exc:
                raise ValueError("risk predictor horizons must be numeric") from exc
            if not math.isfinite(horizon) or horizon <= 0.0:
                raise ValueError("risk predictor horizons must be finite and positive")
            if horizon > self.max_horizon_sec:
                raise ValueError("risk predictor horizon exceeds max_horizon_sec")
            parsed_horizons.append(max(0.05, horizon))
        self.horizons = sorted(set(parsed_horizons)) or [0.5, 1.0, 1.5]

        self.ttc_critical = _as_float(cfg.get("ttc_critical"), 1.2)
        self.ttc_safe = _as_float(cfg.get("ttc_safe"), 5.0)
        self.dist_critical = _as_float(cfg.get("dist_critical"), 3.0)
        self.dist_safe = _as_float(cfg.get("dist_safe"), 30.0)
        self.min_closing_speed = _as_float(cfg.get("min_closing_speed"), 0.1)
        thresholds = (
            self.ttc_critical, self.ttc_safe, self.dist_critical, self.dist_safe
        )
        if any(value < 0.0 for value in thresholds):
            raise ValueError("risk predictor TTC and distance thresholds must be non-negative")
        if self.ttc_safe <= self.ttc_critical or self.dist_safe <= self.dist_critical:
            raise ValueError("risk predictor safe thresholds must exceed critical thresholds")
        if self.min_closing_speed <= 0.0:
            raise ValueError("risk predictor min_closing_speed must be positive")

    def predict(
        self,
        state: WorldState,
        action: CandidateAction,
        response_delay_sec: Optional[float] = None,
    ) -> RiskPrediction:
        if not state.has_lead_vehicle:
            risk = self._external_floor(state)
            return RiskPrediction(
                action=action,
                peak_risk=risk,
                terminal_risk=risk,
                min_distance=state.lead_distance,
                min_ttc=state.lead_ttc,
                trajectory=[],
            )

        response_delay = (
            action.response_delay_sec
            if response_delay_sec is None
            else max(0.0, _as_float(response_delay_sec))
        )

        def evaluate_at(horizon: float) -> Dict[str, float]:
            effective_t = max(0.0, horizon - response_delay)
            distance = max(
                0.0,
                state.lead_distance
                - state.closing_speed * horizon
                + 0.5 * action.target_decel * effective_t * effective_t,
            )
            closing_speed = max(
                0.0,
                state.closing_speed - action.target_decel * effective_t,
            )
            if distance <= 0.0:
                ttc = 0.0
            elif closing_speed > self.min_closing_speed:
                ttc = distance / closing_speed
            else:
                ttc = 99.0
            return {
                "horizon": horizon,
                "distance": distance,
                "closing_speed": closing_speed,
                "ttc": ttc,
                "risk": self._risk_from_distance_ttc(distance, ttc, state),
            }

        # Requested horizons are presentation points. Safety extrema can occur
        # between them, especially when braking makes relative speed cross zero.
        analysis_times = {0.0, *self.horizons}
        max_horizon = max(self.horizons)
        if 0.0 < response_delay < max_horizon:
            analysis_times.add(response_delay)
        decel = action.target_decel
        closing = state.closing_speed
        if decel > 0.0 and closing > 0.0:
            turning_time = response_delay + closing / decel
            if 0.0 < turning_time < max_horizon:
                analysis_times.add(turning_time)

            # TTC can have a separate stationary point before relative speed
            # reaches zero. Include it so threshold crossings are not skipped.
            distance_at_onset = state.lead_distance - closing * response_delay
            discriminant = 2.0 * decel * distance_at_onset - closing * closing
            if discriminant >= 0.0:
                ttc_offset = (closing - math.sqrt(discriminant)) / decel
                ttc_time = response_delay + ttc_offset
                if (
                    0.0 < ttc_offset < closing / decel
                    and 0.0 < ttc_time < max_horizon
                ):
                    analysis_times.add(ttc_time)

        analysis = [evaluate_at(value) for value in sorted(analysis_times)]
        trajectory = [evaluate_at(horizon) for horizon in self.horizons]
        current_sensor_risk = self._risk_from_distance_ttc(
            state.lead_distance, state.lead_ttc, state
        )
        return RiskPrediction(
            action=action,
            peak_risk=round(
                max(current_sensor_risk, *(item["risk"] for item in analysis)), 4
            ),
            terminal_risk=round(trajectory[-1]["risk"], 4),
            min_distance=round(
                min(state.lead_distance, *(item["distance"] for item in analysis)),
                3,
            ),
            min_ttc=round(
                min(state.lead_ttc, *(item["ttc"] for item in analysis)),
                3,
            ),
            trajectory=trajectory,
        )

    def _risk_from_distance_ttc(
        self,
        distance: float,
        ttc: float,
        state: WorldState,
    ) -> float:
        if ttc <= self.ttc_critical:
            ttc_score = 1.0
        elif ttc >= self.ttc_safe:
            ttc_score = 0.0
        else:
            ttc_score = 1.0 - (ttc - self.ttc_critical) / (self.ttc_safe - self.ttc_critical)

        if distance <= self.dist_critical:
            dist_score = 1.0
        elif distance >= self.dist_safe:
            dist_score = 0.0
        else:
            dist_score = 1.0 - (distance - self.dist_critical) / (self.dist_safe - self.dist_critical)

        lane_weight = 0.4 + 0.6 * _clamp(state.lane_relevance)
        risk = max(ttc_score, dist_score) * lane_weight
        risk = max(risk, self._external_floor(state))
        return round(_clamp(risk), 4)

    @staticmethod
    def _external_floor(state: WorldState) -> float:
        floor = state.ext_score * 0.75
        if state.lead_warning_level >= 2:
            floor = max(floor, 0.75)
        elif state.lead_warning_level == 1:
            floor = max(floor, 0.35)
        return _clamp(floor)


class RiskCost:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.pred_weight = _as_float(cfg.get("pred_weight"), 0.65)
        self.terminal_weight = _as_float(cfg.get("terminal_weight"), 0.15)
        self.fusion_weight = _as_float(cfg.get("fusion_weight"), 0.10)
        self.driver_weight = _as_float(cfg.get("driver_weight"), 0.10)
        weights = (
            self.pred_weight, self.terminal_weight, self.fusion_weight, self.driver_weight
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("risk cost weights must be non-negative")

        self.driver_slowdown_threshold = _as_float(
            cfg.get("driver_slowdown_threshold"),
            0.55,
        )
        self.driver_high_threshold = _as_float(
            cfg.get("driver_high_threshold"),
            0.75,
        )
        if not (
            0.0 <= self.driver_slowdown_threshold <= self.driver_high_threshold <= 1.0
        ):
            raise ValueError(
                "driver thresholds must satisfy 0 <= slowdown <= high <= 1"
            )

    def evaluate(
        self,
        state: WorldState,
        action: CandidateAction,
        prediction: RiskPrediction,
    ) -> CandidateEvaluation:
        terms = {
            "pred": self.pred_weight * prediction.peak_risk,
            "terminal": self.terminal_weight * prediction.terminal_risk,
            "fusion": self.fusion_weight * state.fused_score,
            "driver": self.driver_weight * state.int_score,
            "comfort": action.comfort_cost,
            "underreaction": 0.0,
        }

        if prediction.peak_risk >= 0.75 and action.rank < 2:
            terms["underreaction"] += (prediction.peak_risk - 0.70) * 0.65
        if prediction.min_ttc <= 0.8 and action.rank < 3:
            terms["underreaction"] += (0.8 - prediction.min_ttc) * 0.75 + 0.20
        if prediction.min_distance <= 2.0 and action.rank < 3:
            terms["underreaction"] += (2.0 - prediction.min_distance) * 0.35 + 0.25
        if state.fused_level >= 3 and action.rank < 2:
            terms["underreaction"] += 0.22
        elif state.fused_level >= 2 and action.rank == 0:
            terms["underreaction"] += 0.12

        if state.int_score >= self.driver_slowdown_threshold:
            if action.name == ACTION_KEEP:
                terms["underreaction"] += 0.16
            elif action.name == ACTION_SLOW_DOWN and state.int_score >= self.driver_high_threshold:
                terms["underreaction"] += 0.04

        cost = round(sum(terms.values()), 4)
        return CandidateEvaluation(action=action, prediction=prediction, cost=cost, terms=terms)


class SafetyShield:
    """Non-learned lower bound on braking severity for critical scenes."""

    def __init__(self, config: Optional[Dict[str, Any]], predictor: RiskPredictor):
        root = config or {}
        cfg = root.get("safety_shield", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.emergency_ttc = _as_float(
            cfg.get("emergency_ttc"), min(0.65, predictor.ttc_critical)
        )
        self.emergency_distance = _as_float(
            cfg.get("emergency_distance"), min(1.25, predictor.dist_critical)
        )
        self.brake_ttc = _as_float(cfg.get("brake_ttc"), predictor.ttc_critical)
        self.brake_distance = _as_float(cfg.get("brake_distance"), predictor.dist_critical)
        self.caution_ttc = _as_float(
            cfg.get("caution_ttc"), max(3.0, predictor.ttc_critical)
        )
        self.caution_distance = _as_float(
            cfg.get("caution_distance"), max(10.0, predictor.dist_critical)
        )
        self.driver_slowdown_threshold = _as_float(
            cfg.get("driver_slowdown_threshold"),
            root.get("driver_slowdown_threshold", 0.55),
        )
        raw_loss_rank = cfg.get("perception_loss_min_rank", 1)
        if isinstance(raw_loss_rank, bool):
            raise ValueError("perception_loss_min_rank must be an integer in [1, 3]")
        try:
            self.perception_loss_min_rank = int(raw_loss_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "perception_loss_min_rank must be an integer in [1, 3]"
            ) from exc
        if self.perception_loss_min_rank != raw_loss_rank or not (
            1 <= self.perception_loss_min_rank <= 3
        ):
            raise ValueError("perception_loss_min_rank must be an integer in [1, 3]")
        raw_loss_delay = cfg.get("perception_loss_brake_after_sec", 0.5)
        if isinstance(raw_loss_delay, bool):
            raise ValueError(
                "perception_loss_brake_after_sec must be finite and non-negative"
            )
        try:
            self.perception_loss_brake_after_sec = float(raw_loss_delay)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "perception_loss_brake_after_sec must be finite and non-negative"
            ) from exc
        ttc_thresholds = (self.emergency_ttc, self.brake_ttc, self.caution_ttc)
        distance_thresholds = (
            self.emergency_distance,
            self.brake_distance,
            self.caution_distance,
        )
        if any(value < 0.0 for value in (*ttc_thresholds, *distance_thresholds)):
            raise ValueError("safety shield TTC and distance thresholds must be non-negative")
        if tuple(sorted(ttc_thresholds)) != ttc_thresholds:
            raise ValueError(
                "safety shield TTC thresholds must satisfy emergency <= brake <= caution"
            )
        if tuple(sorted(distance_thresholds)) != distance_thresholds:
            raise ValueError(
                "safety shield distance thresholds must satisfy emergency <= brake <= caution"
            )
        if not 0.0 <= self.driver_slowdown_threshold <= 1.0:
            raise ValueError("safety shield driver_slowdown_threshold must be in [0, 1]")
        if (
            not math.isfinite(self.perception_loss_brake_after_sec)
            or self.perception_loss_brake_after_sec < 0.0
        ):
            raise ValueError(
                "perception_loss_brake_after_sec must be finite and non-negative"
            )

    def minimum_rank(self, state: WorldState) -> int:
        if not self.enabled:
            return 0

        rank = 0
        if state.has_lead_vehicle:
            approaching = state.closing_speed > 0.1
            if (
                state.lead_distance <= self.emergency_distance
                or (approaching and state.lead_ttc <= self.emergency_ttc)
            ):
                rank = 3
            elif (
                state.lead_warning_level >= 2
                or state.lead_distance <= self.brake_distance
                or (approaching and state.lead_ttc <= self.brake_ttc)
            ):
                rank = 2
            elif (
                state.lead_warning_level >= 1
                or state.lead_distance <= self.caution_distance
                or (approaching and state.lead_ttc <= self.caution_ttc)
            ):
                rank = 1

        if state.fused_level >= 3:
            rank = max(rank, 2)
        elif state.fused_level >= 2:
            rank = max(rank, 1)
        if state.int_score >= self.driver_slowdown_threshold:
            rank = max(rank, 1)
        if not state.external_perception_valid:
            rank = max(rank, self.perception_loss_min_rank)
            if state.external_perception_age_sec >= self.perception_loss_brake_after_sec:
                rank = max(rank, 2)
        return rank


class MRMPlanner:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.actions = CandidateActions(self.cfg)
        self.predictor = RiskPredictor(self.cfg)
        self.kinematic_predictor = self.predictor
        self.cost = RiskCost(self.cfg)
        self.safety_shield = SafetyShield(self.cfg, self.predictor)
        self.predictor_mode = str(self.cfg.get("predictor", "kinematic")).lower()
        if self.predictor_mode not in {"kinematic", "hybrid"}:
            raise ValueError("decision.predictor must be 'kinematic' or 'hybrid'")
        self.rssm_can_deescalate = bool(self.cfg.get("rssm_can_deescalate", False))
        self.rssm = None
        self.model_status = "kinematic"
        self._last_valid_external_state: Optional[Dict[str, Any]] = None
        self._perception_loss_started_at: Optional[float] = None
        self._perception_loss_age_sec = 0.0
        self._last_valid_decision_rank = 0
        self._rssm_reset_for_perception_loss = False
        if self.predictor_mode == "hybrid":
            try:
                from src.core.rssm_world_model import (
                    ACTION_CONTRACT_VERSION,
                    MAX_IMAGINATION_STEPS,
                    MIN_RSSM_TRAINING_STEPS,
                    RSSMInferenceEngine,
                    build_fusion_feature_contract,
                )

                engine = RSSMInferenceEngine(self.cfg.get("rssm", {}) or {})
                if engine.ready:
                    fusion_contract = build_fusion_feature_contract(
                        self.cfg.get("_runtime_fusion_config"),
                        self.cfg.get("_runtime_internal_config"),
                    )
                    self._validate_rssm_action_catalog(
                        engine.metrics,
                        ACTION_CONTRACT_VERSION,
                        engine.config,
                        fusion_contract,
                        MAX_IMAGINATION_STEPS,
                        MIN_RSSM_TRAINING_STEPS,
                    )
                    self.rssm = engine
                self.model_status = engine.status
            except Exception as exc:
                self.rssm = None
                self.model_status = f"fallback ({type(exc).__name__}: {exc})"

    def _validate_rssm_action_catalog(
        self,
        metrics: Any,
        contract_version: int,
        model_config: Any,
        expected_fusion_contract: Mapping[str, Any],
        max_imagination_steps: int,
        absolute_minimum_training_steps: int,
    ) -> None:
        if not isinstance(metrics, dict):
            raise ValueError("RSSM checkpoint metrics must contain an action catalog")
        saved_fusion_contract = metrics.get("fusion_feature_contract")
        if saved_fusion_contract != dict(expected_fusion_contract):
            raise ValueError("RSSM checkpoint fusion feature contract mismatch")
        rssm_cfg = self.cfg.get("rssm", {}) or {}
        raw_minimum_steps = rssm_cfg.get(
            "min_training_steps", absolute_minimum_training_steps
        )
        if isinstance(raw_minimum_steps, bool):
            raise ValueError("rssm.min_training_steps must be a positive integer")
        try:
            minimum_steps = int(raw_minimum_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rssm.min_training_steps must be a positive integer"
            ) from exc
        if (
            minimum_steps != raw_minimum_steps
            or minimum_steps < absolute_minimum_training_steps
        ):
            raise ValueError(
                "rssm.min_training_steps cannot be below "
                f"{absolute_minimum_training_steps}"
            )
        trained_steps = metrics.get("steps")
        if (
            isinstance(trained_steps, bool)
            or not isinstance(trained_steps, int)
            or trained_steps < minimum_steps
        ):
            raise ValueError(
                f"RSSM checkpoint requires at least {minimum_steps} training steps"
            )
        required_steps = math.ceil(
            max(self.predictor.horizons) / model_config.dt_sec
        )
        if required_steps > max_imagination_steps:
            raise ValueError("RSSM planning horizon exceeds the imagination step limit")
        contract = metrics.get("action_contract")
        if not isinstance(contract, dict):
            raise ValueError("RSSM checkpoint action contract is missing")
        if contract.get("version") != contract_version:
            raise ValueError("RSSM checkpoint action contract version mismatch")
        if contract.get("unknown_action_supported") is not True:
            raise ValueError("RSSM checkpoint was not trained for unknown actions")
        expected_alignment = (
            "previous_observation_applied_action_current_observation"
        )
        if contract.get("alignment") != expected_alignment:
            raise ValueError("RSSM checkpoint action alignment mismatch")
        unknown_training = metrics.get("unknown_action_training")
        if not isinstance(unknown_training, dict):
            raise ValueError("RSSM checkpoint unknown-action training metadata is missing")
        unknown_fraction = _as_float(unknown_training.get("unknown_fraction"), float("nan"))
        if not 0.05 <= unknown_fraction <= 1.0:
            raise ValueError("RSSM checkpoint unknown-action coverage is invalid")
        catalog = metrics.get("action_catalog")
        expected = self.actions.generate(WorldState())
        if not isinstance(catalog, (list, tuple)) or len(catalog) != len(expected):
            raise ValueError("RSSM checkpoint action catalog is missing or incomplete")
        known_counts = metrics.get("known_executed_action_counts")
        if not isinstance(known_counts, dict):
            raise ValueError("RSSM checkpoint known-action coverage is missing")
        for action in expected:
            count = known_counts.get(action.name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"RSSM checkpoint lacks known-action coverage for {action.name}"
                )
        for saved, action in zip(catalog, expected):
            if not isinstance(saved, dict) or saved.get("name") != action.name:
                raise ValueError("RSSM checkpoint action catalog order/name mismatch")
            saved_decel = _as_float(saved.get("target_decel"), float("nan"))
            saved_delay = _as_float(saved.get("response_delay_sec"), float("nan"))
            if not (
                math.isclose(saved_decel, action.target_decel, abs_tol=1e-7)
                and math.isclose(saved_delay, action.response_delay_sec, abs_tol=1e-7)
            ):
                raise ValueError(
                    f"RSSM checkpoint action parameters mismatch for {action.name}"
                )
            if (
                action.target_decel > model_config.max_decel + 1e-7
                or action.response_delay_sec > model_config.max_delay + 1e-7
                or action.response_delay_sec > model_config.dt_sec + 1e-7
            ):
                raise ValueError(
                    f"RSSM checkpoint normalization cannot represent {action.name}"
                )

    def _applied_action_delay_override(
        self,
        candidate: CandidateAction,
        applied_action: Any,
    ) -> Optional[float]:
        """Return a future delay only when the receipt matches this candidate."""
        if applied_action is None:
            return None
        identity: Any = None
        has_remaining_delay = False
        has_response_delay = False
        explicit_changed: Optional[bool] = None
        if isinstance(applied_action, Mapping):
            if "target_decel" in applied_action:
                raw_decel = applied_action["target_decel"]
            elif "decel" in applied_action:
                raw_decel = applied_action["decel"]
            else:
                return None
            has_remaining_delay = "remaining_delay_sec" in applied_action
            has_response_delay = (
                "response_delay_sec" in applied_action or "delay" in applied_action
            )
            raw_delay = applied_action.get(
                "response_delay_sec", applied_action.get("delay", 0.0)
            )
            raw_remaining = applied_action.get("remaining_delay_sec", raw_delay)
            identity = applied_action.get("name", applied_action.get("stable_action_id"))
            if "action_changed" in applied_action:
                raw_changed = applied_action["action_changed"]
                if not isinstance(raw_changed, bool):
                    return None
                explicit_changed = raw_changed
        elif isinstance(applied_action, (int, float)) and not isinstance(applied_action, bool):
            raw_decel, raw_delay = applied_action, 0.0
            raw_remaining = raw_delay
        elif hasattr(applied_action, "target_decel"):
            raw_decel = getattr(applied_action, "target_decel")
            raw_delay = getattr(applied_action, "response_delay_sec", 0.0)
            has_response_delay = hasattr(applied_action, "response_delay_sec")
            has_remaining_delay = hasattr(applied_action, "remaining_delay_sec")
            raw_remaining = getattr(applied_action, "remaining_delay_sec", raw_delay)
            identity = getattr(
                applied_action, "name", getattr(applied_action, "stable_action_id", None)
            )
            if hasattr(applied_action, "action_changed"):
                raw_changed = getattr(applied_action, "action_changed")
                if not isinstance(raw_changed, bool):
                    return None
                explicit_changed = raw_changed
        else:
            return None
        try:
            decel = float(raw_decel)
            delay = float(raw_delay)
            remaining_delay = float(raw_remaining)
        except (TypeError, ValueError):
            return None
        rssm_cfg = self.cfg.get("rssm", {}) or {}
        max_decel = _as_float(rssm_cfg.get("max_decel"), 8.0)
        max_delay = _as_float(rssm_cfg.get("max_delay"), 0.5)
        dt_sec = _as_float(rssm_cfg.get("dt_sec"), 0.25)
        values = (decel, delay, remaining_delay)
        if (
            not all(math.isfinite(value) for value in values)
            or decel < 0.0
            or delay < 0.0
            or remaining_delay < 0.0
            or decel > max_decel
            or delay > max_delay
            or remaining_delay > max_delay
            or delay > dt_sec + 1e-6
        ):
            return None
        identity_matches = identity is None or str(identity) == candidate.name
        decel_matches = math.isclose(
            decel, candidate.target_decel, abs_tol=1e-3
        )
        if not identity_matches or not decel_matches:
            return None
        if has_remaining_delay:
            return remaining_delay
        # response_delay_sec describes the transition that produced the current
        # observation. Future prediction only waits when remaining delay is explicit.
        return 0.0

    def plan(
        self,
        fusion_result: Any,
        vehicle_data: Optional[Sequence[Dict[str, Any]]] = None,
        face_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        applied_action: Any = None,
        sequence_timestamp: Optional[float] = None,
        external_perception_valid: bool = True,
        external_perception_age_sec: Optional[float] = None,
    ) -> DecisionResult:
        if not isinstance(external_perception_valid, bool):
            raise ValueError("external_perception_valid must be a bool")
        state = self.build_world_state(
            fusion_result=fusion_result,
            vehicle_data=vehicle_data,
            face_data=face_data,
            timestamp=timestamp,
            external_perception_valid=external_perception_valid,
            external_perception_age_sec=external_perception_age_sec,
        )

        if not self.enabled:
            return DecisionResult(
                timestamp=state.timestamp,
                action=ACTION_KEEP,
                target_decel=0.0,
                cost=0.0,
                predicted_risk=0.0,
                predicted_min_distance=state.lead_distance,
                predicted_min_ttc=state.lead_ttc,
                reasons=["decision layer disabled"],
                world_state=state,
                candidates=[],
            )

        actions = self.actions.generate(state)
        delay_overrides = {
            action.name: self._applied_action_delay_override(action, applied_action)
            for action in actions
        }
        analytic_predictions = {
            action.name: self.kinematic_predictor.predict(
                state,
                action,
                response_delay_sec=delay_overrides[action.name],
            )
            for action in actions
        }
        analytic_evaluations = [
            self.cost.evaluate(state, action, analytic_predictions[action.name])
            for action in actions
        ]

        rssm_forecasts: Dict[str, Any] = {}
        model_used = False
        if not state.external_perception_valid:
            if self.rssm is not None and not self._rssm_reset_for_perception_loss:
                self.rssm.reset()
                self._rssm_reset_for_perception_loss = True
            self.model_status = "fallback (external perception unavailable)"
        elif self.rssm is not None and self.rssm.ready:
            self._rssm_reset_for_perception_loss = False
            try:
                model_time = state.timestamp if sequence_timestamp is None else sequence_timestamp
                self.rssm.observe(
                    state,
                    applied_action=applied_action,
                    timestamp=model_time,
                )
                rssm_forecasts = self.rssm.predict_many(
                    actions,
                    self.kinematic_predictor.horizons,
                )
                model_used = bool(rssm_forecasts)
                self.model_status = self.rssm.status
            except Exception as exc:
                self.model_status = f"fallback ({type(exc).__name__}: {exc})"
                rssm_forecasts = {}

        evaluations: List[CandidateEvaluation] = []
        for action in actions:
            prediction = analytic_predictions[action.name]
            forecast = rssm_forecasts.get(action.name)
            if forecast is not None:
                prediction = self._conservative_merge(prediction, forecast)
            evaluations.append(self.cost.evaluate(state, action, prediction))

        unconstrained = min(evaluations, key=lambda item: (item.cost, item.action.rank))
        minimum_rank = self.safety_shield.minimum_rank(state)
        if not state.external_perception_valid:
            minimum_rank = max(minimum_rank, self._last_valid_decision_rank)
        if model_used and not self.rssm_can_deescalate:
            analytic_best = min(
                analytic_evaluations,
                key=lambda item: (item.cost, item.action.rank),
            )
            minimum_rank = max(minimum_rank, analytic_best.action.rank)
        eligible = [item for item in evaluations if item.action.rank >= minimum_rank]
        best = min(eligible, key=lambda item: (item.cost, item.action.rank))
        if state.external_perception_valid:
            self._last_valid_decision_rank = best.action.rank
        reasons = self._explain(state, best, evaluations)
        if best.action.rank > unconstrained.action.rank:
            reasons.insert(0, f"safety shield minimum {best.action.name}")
            reasons = reasons[:4]
        if model_used and len(reasons) < 4:
            reasons.append("RSSM CVaR imagination")

        source = "rssm_hybrid" if model_used else "kinematic"
        best_forecast = rssm_forecasts.get(best.action.name)
        uncertainty = _as_float(getattr(best_forecast, "uncertainty", 0.0))

        return DecisionResult(
            timestamp=state.timestamp,
            action=best.action.name,
            target_decel=best.action.target_decel,
            cost=best.cost,
            predicted_risk=best.prediction.peak_risk,
            predicted_min_distance=best.prediction.min_distance,
            predicted_min_ttc=best.prediction.min_ttc,
            reasons=reasons,
            world_state=state,
            candidates=evaluations,
            prediction_source=source,
            model_uncertainty=uncertainty,
        )

    @staticmethod
    def _conservative_merge(
        analytic: RiskPrediction,
        forecast: Any,
    ) -> RiskPrediction:
        trajectory: List[Dict[str, float]] = []
        learned_trajectory = list(getattr(forecast, "trajectory", []) or [])
        if analytic.trajectory:
            for index, base in enumerate(analytic.trajectory):
                learned = learned_trajectory[index] if index < len(learned_trajectory) else {}
                trajectory.append({
                    "horizon": _as_float(base.get("horizon"), 0.0),
                    "distance": min(
                        _as_float(base.get("distance"), 99.0),
                        _as_float(learned.get("distance"), 99.0),
                    ),
                    "closing_speed": max(
                        _as_float(base.get("closing_speed"), 0.0),
                        _as_float(learned.get("closing_speed"), 0.0),
                    ),
                    "ttc": min(
                        _as_float(base.get("ttc"), 99.0),
                        _as_float(learned.get("ttc"), 99.0),
                    ),
                    "risk": max(
                        _as_float(base.get("risk"), 0.0),
                        _as_float(learned.get("risk"), 0.0),
                    ),
                })
        else:
            trajectory = [dict(item) for item in learned_trajectory]

        return RiskPrediction(
            action=analytic.action,
            peak_risk=round(
                max(analytic.peak_risk, _as_float(forecast.peak_risk)),
                4,
            ),
            terminal_risk=round(
                max(analytic.terminal_risk, _as_float(forecast.terminal_risk)),
                4,
            ),
            min_distance=round(
                min(analytic.min_distance, _as_float(forecast.min_distance, 99.0)),
                3,
            ),
            min_ttc=round(
                min(analytic.min_ttc, _as_float(forecast.min_ttc, 99.0)),
                3,
            ),
            trajectory=trajectory,
        )

    def reset(self) -> None:
        if self.rssm is not None:
            self.rssm.reset()
        self._last_valid_external_state = None
        self._perception_loss_started_at = None
        self._perception_loss_age_sec = 0.0
        self._last_valid_decision_rank = 0
        self._rssm_reset_for_perception_loss = False

    def build_world_state(
        self,
        fusion_result: Any,
        vehicle_data: Optional[Sequence[Dict[str, Any]]] = None,
        face_data: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        external_perception_valid: bool = True,
        external_perception_age_sec: Optional[float] = None,
    ) -> WorldState:
        if not isinstance(external_perception_valid, bool):
            raise ValueError("external_perception_valid must be a bool")
        if external_perception_age_sec is None:
            reported_perception_age = 0.0
        else:
            if isinstance(external_perception_age_sec, bool):
                raise ValueError(
                    "external_perception_age_sec must be finite and non-negative"
                )
            try:
                reported_perception_age = float(external_perception_age_sec)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "external_perception_age_sec must be finite and non-negative"
                ) from exc
            if (
                not math.isfinite(reported_perception_age)
                or reported_perception_age < 0.0
            ):
                raise ValueError(
                    "external_perception_age_sec must be finite and non-negative"
                )
        timestamp = time.time() if timestamp is None else timestamp
        vehicle_data = list(vehicle_data or [])
        primary = self._pick_primary_object(vehicle_data)

        ext_score = _as_float(getattr(fusion_result, "ext_score", 0.0))
        int_score = _as_float(getattr(fusion_result, "int_score", 0.0))

        state = WorldState(
            timestamp=timestamp,
            external_perception_valid=external_perception_valid,
            ext_score=ext_score,
            int_score=int_score,
            cross_score=_as_float(getattr(fusion_result, "cross_score", 0.0)),
            fused_score=_as_float(getattr(fusion_result, "fused_score", 0.0)),
            fused_level=_as_int(getattr(fusion_result, "fused_level", 0)),
            fused_text=str(getattr(fusion_result, "fused_text", "SAFE")),
            fatigue_score=_as_float(getattr(fusion_result, "int_fatigue_score", 0.0)),
            attention_score=_as_float(getattr(fusion_result, "int_attention_score", 0.0)),
            driver_flags=self._driver_flags(fusion_result, face_data),
        )

        if primary:
            distance = _as_float(primary.get("distance"), 99.0)
            ttc = _as_float(primary.get("ttc"), 99.0)
            if ttc <= 0.0:
                ttc = 99.0
            closing_speed = max(0.0, _as_float(primary.get("rel_speed"), 0.0))
            if 0.0 < distance < 99.0 and 0.0 < ttc < 99.0:
                # Distance/TTC and rel_speed can be temporarily inconsistent.
                # Use the larger closing estimate so planning never understates it.
                closing_speed = max(closing_speed, distance / max(ttc, 0.1))

            state.has_lead_vehicle = True
            state.lead_distance = distance
            state.lead_ttc = ttc
            state.closing_speed = max(0.0, closing_speed)
            state.lane_relevance = _clamp(_as_float(primary.get("lane_relevance"), 1.0))
            state.lead_warning_level = _as_int(primary.get("warning_level"), 0)
            state.lead_class_name = str(primary.get("class_name", "vehicle"))
        else:
            state.lead_distance = _as_float(getattr(fusion_result, "ext_min_dist", 99.0), 99.0)
            state.lead_ttc = _as_float(getattr(fusion_result, "ext_min_ttc", 99.0), 99.0)
            if state.lead_distance <= 0.0:
                state.lead_distance = 99.0
            if state.lead_ttc <= 0.0:
                state.lead_ttc = 99.0
            state.lead_warning_level = _as_int(getattr(fusion_result, "ext_max_level", 0), 0)
            state.has_lead_vehicle = (
                state.ext_score > 0.05
                or state.lead_warning_level > 0
                or state.lead_distance < 60.0
                or state.lead_ttc < 20.0
            )
            if state.has_lead_vehicle and state.lead_distance < 99.0 and state.lead_ttc < 99.0:
                state.closing_speed = state.lead_distance / max(state.lead_ttc, 0.1)

        if external_perception_valid:
            state.external_perception_age_sec = reported_perception_age
            self._last_valid_external_state = self._external_snapshot(state)
            self._perception_loss_started_at = None
            self._perception_loss_age_sec = 0.0
        else:
            if self._perception_loss_started_at is None:
                self._perception_loss_started_at = timestamp
            computed_age = max(0.0, timestamp - self._perception_loss_started_at)
            self._perception_loss_age_sec = max(
                self._perception_loss_age_sec,
                computed_age,
                reported_perception_age,
            )
            state.external_perception_age_sec = self._perception_loss_age_sec
            self._hold_last_external_state(state)

        return state

    @staticmethod
    def _external_snapshot(state: WorldState) -> Dict[str, Any]:
        return {
            "ext_score": state.ext_score,
            "fused_score": state.fused_score,
            "fused_level": state.fused_level,
            "fused_text": state.fused_text,
            "has_lead_vehicle": state.has_lead_vehicle,
            "lead_distance": state.lead_distance,
            "lead_ttc": state.lead_ttc,
            "closing_speed": state.closing_speed,
            "lane_relevance": state.lane_relevance,
            "lead_warning_level": state.lead_warning_level,
            "lead_class_name": state.lead_class_name,
        }

    def _hold_last_external_state(self, state: WorldState) -> None:
        saved = self._last_valid_external_state
        if saved is None:
            return
        state.external_state_held = True
        state.ext_score = max(state.ext_score, _as_float(saved.get("ext_score")))
        saved_fused = _as_float(saved.get("fused_score"))
        if saved_fused > state.fused_score:
            state.fused_score = saved_fused
        saved_level = _as_int(saved.get("fused_level"), 0)
        if saved_level > state.fused_level:
            state.fused_level = saved_level
            state.fused_text = str(saved.get("fused_text", state.fused_text))
        if not bool(saved.get("has_lead_vehicle")):
            return
        age = state.external_perception_age_sec
        closing_speed = max(0.0, _as_float(saved.get("closing_speed"), 0.0))
        saved_distance = max(0.0, _as_float(saved.get("lead_distance"), 99.0))
        saved_ttc = max(0.0, _as_float(saved.get("lead_ttc"), 99.0))
        state.has_lead_vehicle = True
        state.closing_speed = closing_speed
        state.lead_distance = max(0.0, saved_distance - closing_speed * age)
        state.lead_ttc = (
            max(0.0, saved_ttc - age)
            if closing_speed > 0.1 and saved_ttc < 99.0
            else saved_ttc
        )
        state.lane_relevance = _clamp(
            _as_float(saved.get("lane_relevance"), 1.0)
        )
        state.lead_warning_level = _as_int(
            saved.get("lead_warning_level"), 0
        )
        state.lead_class_name = str(saved.get("lead_class_name", "vehicle"))

    def _pick_primary_object(
        self,
        vehicle_data: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        valid_objects = [
            obj
            for obj in vehicle_data
            if isinstance(obj, Mapping)
            and _as_float(obj.get("distance"), -1.0) > 0.0
        ]
        if not valid_objects:
            return None

        def score(obj: Dict[str, Any]) -> float:
            distance = _as_float(obj.get("distance"), 99.0)
            ttc = _as_float(obj.get("ttc"), 99.0)
            if ttc <= 0.0:
                ttc = 99.0
            warning = _as_int(obj.get("warning_level"), 0)
            lane = _clamp(_as_float(obj.get("lane_relevance"), 1.0))
            closing = max(0.0, _as_float(obj.get("rel_speed"), 0.0))
            dist_score = 0.0 if distance >= 35.0 else 1.0 - max(0.0, distance - 2.0) / 33.0
            ttc_score = 0.0 if ttc >= 6.0 else 1.0 - max(0.0, ttc - 0.8) / 5.2
            return (warning * 1.5 + max(dist_score, ttc_score) + 0.1 * closing) * (0.4 + 0.6 * lane)

        return max(valid_objects, key=score)

    @staticmethod
    def _driver_flags(fusion_result: Any, face_data: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
        flags: List[str] = []
        checks = [
            ("drowsy", bool(getattr(fusion_result, "int_drowsy", False))),
            ("yawning", bool(getattr(fusion_result, "int_yawning", False))),
            ("distracted", bool(getattr(fusion_result, "int_distracted", False))),
            ("nodding", bool(getattr(fusion_result, "int_nodding", False))),
        ]
        for name, active in checks:
            if active:
                flags.append(name)

        if face_data and face_data.get("has_face"):
            if face_data.get("is_perclos_fatigued"):
                flags.append("perclos")
            if face_data.get("is_blink_freq_high"):
                flags.append("blink")

        return tuple(dict.fromkeys(flags))

    def _explain(
        self,
        state: WorldState,
        best: CandidateEvaluation,
        evaluations: Sequence[CandidateEvaluation],
    ) -> List[str]:
        reasons: List[str] = []
        if not state.external_perception_valid:
            if state.external_state_held:
                reasons.append(
                    "external perception unavailable; holding last state"
                )
            else:
                reasons.append("external perception unavailable")
        if state.has_lead_vehicle:
            if state.lead_ttc < self.predictor.ttc_critical:
                reasons.append(f"TTC {state.lead_ttc:.1f}s critical")
            elif state.lead_ttc < self.predictor.ttc_safe:
                reasons.append(f"TTC {state.lead_ttc:.1f}s")

            if state.lead_distance < self.predictor.dist_critical:
                reasons.append(f"distance {state.lead_distance:.1f}m critical")
            elif state.lead_distance < 12.0:
                reasons.append(f"distance {state.lead_distance:.1f}m")

            if state.closing_speed > 0.5:
                reasons.append(f"closing {state.closing_speed:.1f}m/s")

        if state.int_score >= 0.55:
            if state.driver_flags:
                reasons.append("driver " + "/".join(state.driver_flags[:2]))
            else:
                reasons.append(f"driver risk {state.int_score:.2f}")

        if state.cross_score >= 0.35:
            reasons.append(f"cross risk {state.cross_score:.2f}")

        if best.prediction.peak_risk >= 0.55:
            reasons.append(f"pred risk {best.prediction.peak_risk:.2f}")

        if not reasons:
            reasons.append("risk remains low")

        return reasons[:4]


class DecisionLogger:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("log_enable", True))
        self.interval_sec = _as_float(cfg.get("log_interval_sec"), 0.5)
        self.path = str(cfg.get("log_path", "logs/mrm_decisions.csv"))
        self._last_log_time = 0.0
        self._file = None
        self._writer = None

        if not self.enabled:
            return

        log_dir = os.path.dirname(self.path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_exists = os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "timestamp",
                "action",
                "target_decel",
                "cost",
                "predicted_risk",
                "predicted_min_distance",
                "predicted_min_ttc",
                "ext_score",
                "int_score",
                "cross_score",
                "fused_score",
                "fused_level",
                "lead_distance",
                "lead_ttc",
                "closing_speed",
                "lead_warning_level",
                "reasons",
            ],
        )
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, decision: DecisionResult, timestamp: Optional[float] = None) -> None:
        if not self.enabled or self._writer is None or self._file is None:
            return

        now = time.time() if timestamp is None else timestamp
        if now - self._last_log_time < self.interval_sec:
            return
        self._last_log_time = now

        state = decision.world_state
        self._writer.writerow({
            "timestamp": f"{decision.timestamp:.3f}",
            "action": decision.action,
            "target_decel": f"{decision.target_decel:.2f}",
            "cost": f"{decision.cost:.4f}",
            "predicted_risk": f"{decision.predicted_risk:.4f}",
            "predicted_min_distance": f"{decision.predicted_min_distance:.3f}",
            "predicted_min_ttc": f"{decision.predicted_min_ttc:.3f}",
            "ext_score": f"{state.ext_score:.4f}",
            "int_score": f"{state.int_score:.4f}",
            "cross_score": f"{state.cross_score:.4f}",
            "fused_score": f"{state.fused_score:.4f}",
            "fused_level": state.fused_level,
            "lead_distance": f"{state.lead_distance:.3f}",
            "lead_ttc": f"{state.lead_ttc:.3f}",
            "closing_speed": f"{state.closing_speed:.3f}",
            "lead_warning_level": state.lead_warning_level,
            "reasons": " | ".join(decision.reasons),
        })
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


__all__ = [
    "ACTION_KEEP",
    "ACTION_SLOW_DOWN",
    "ACTION_BRAKE",
    "ACTION_EMERGENCY_BRAKE",
    "CandidateAction",
    "WorldState",
    "RiskPrediction",
    "CandidateEvaluation",
    "DecisionResult",
    "CandidateActions",
    "RiskPredictor",
    "RiskCost",
    "SafetyShield",
    "MRMPlanner",
    "DecisionLogger",
]
