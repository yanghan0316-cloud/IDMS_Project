"""Tiny DreamerV3-style RSSM for feature-level minimum-risk planning.

The model operates on the already fused IDMS state rather than camera pixels.
It keeps DreamerV3's important world-model ingredients: a recurrent
deterministic state, discrete categorical stochastic variables, straight-
through samples, uniform probability mixing, and separated dynamics and
representation KL losses.

This module deliberately does not treat a planner recommendation as an action
that was executed.  Runtime callers must pass confirmed applied actions; an
unknown action is represented by an explicit invalid-action bit.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import math
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = 3
ACTION_CONTRACT_VERSION = 1
OBS_FIELDS: Tuple[str, ...] = (
    "has_lead_vehicle",
    "lead_distance",
    "lead_ttc",
    "closing_speed",
    "lane_relevance",
    "lead_warning_level",
    "ext_score",
    "int_score",
    "cross_score",
    "fused_score",
    "fused_level",
    "fatigue_score",
    "attention_score",
)
ACTION_FIELDS: Tuple[str, ...] = (
    "target_decel",
    "response_delay",
    "action_valid",
)
FUSION_FEATURE_CONTRACT_VERSION = 2
MAX_IMAGINATION_STEPS = 40
MAX_RSSM_SAMPLES = 64
MAX_RSSM_NETWORK_DIM = 1024
MAX_RSSM_CATEGORICAL_DIM = 64
MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
MIN_RSSM_TRAINING_STEPS = 1000


def _read_checkpoint_bytes(path: str | Path) -> bytes:
    """Read one bounded snapshot so verification and loading see identical bytes."""
    source = Path(path)
    with source.open("rb") as stream:
        stream.seek(0, 2)
        checkpoint_size = stream.tell()
        stream.seek(0)
        checkpoint_bytes = stream.read(MAX_CHECKPOINT_BYTES + 1)
    if checkpoint_size <= 0:
        raise ValueError("RSSM checkpoint is empty")
    if checkpoint_size > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"RSSM checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes"
        )
    if len(checkpoint_bytes) != checkpoint_size:
        raise ValueError("RSSM checkpoint changed while it was being read")
    return checkpoint_bytes


def checkpoint_sha256(path: str | Path) -> str:
    """Hash the same size-bounded byte snapshot used by the secure loader."""
    return hashlib.sha256(_read_checkpoint_bytes(path)).hexdigest()


def build_fusion_feature_contract(
    fusion_config: Optional[Mapping[str, Any]] = None,
    internal_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize every online scoring parameter recorded in a checkpoint.

    RiskFusionEngine passes its fusion configuration to DriverStateAssessor.
    The separate internal configuration controls upstream signal extraction;
    both resolved parameter sets are recorded without allowing one to silently
    override the other.
    """
    cfg = dict(fusion_config or {})
    internal_cfg = dict(internal_config or {})
    effective_cfg = dict(internal_cfg)
    effective_cfg.update(cfg)

    def number(name: str, default: float) -> float:
        try:
            value = float(effective_cfg.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fusion feature {name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"fusion feature {name} must be finite")
        return value

    def internal_number(name: str, default: float) -> float:
        raw = internal_cfg.get(name, default)
        if name == "fps" and not raw:
            raw = 0.0
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"internal feature {name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"internal feature {name} must be finite")
        return value

    def internal_integer(name: str, default: int) -> int:
        raw = internal_cfg.get(name, default)
        if isinstance(raw, bool):
            raise ValueError(f"internal feature {name} must be an integer")
        try:
            return int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"internal feature {name} must be an integer") from exc

    def internal_boolean(name: str, default: bool) -> bool:
        return bool(internal_cfg.get(name, default))

    thresholds = effective_cfg.get("level_thresholds", [0.25, 0.50, 0.75])
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 3:
        thresholds = [0.25, 0.50, 0.75]
    normalized_thresholds = []
    for value in thresholds:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("fusion level thresholds must be numeric") from exc
        if not math.isfinite(parsed):
            raise ValueError("fusion level thresholds must be finite")
        normalized_thresholds.append(parsed)

    yaw_threshold = number("distraction_yaw_threshold_deg", 20.0)
    yaw_safe = number(
        "distraction_yaw_safe_deg", max(0.0, yaw_threshold * 0.5)
    )
    yaw_danger = number(
        "distraction_yaw_danger_deg", max(yaw_threshold + 10.0, 40.0)
    )
    if yaw_danger <= yaw_safe:
        yaw_danger = yaw_safe + 1.0

    return {
        "version": FUSION_FEATURE_CONTRACT_VERSION,
        "external_mapping": "risk_fusion_piecewise_v1",
        "driver_mapping": "driver_state_assessor_v1",
        "upstream_signal_mapping": "face_mesh_fatigue_attention_v1",
        "upstream_internal": {
            "max_num_faces": internal_integer("max_num_faces", 1),
            "refine_landmarks": internal_boolean("refine_landmarks", True),
            "min_detection_confidence": internal_number(
                "min_detection_confidence", 0.5
            ),
            "min_tracking_confidence": internal_number(
                "min_tracking_confidence", 0.5
            ),
            "enable_head_pose": internal_boolean("enable_head_pose", True),
            "enable_attention": internal_boolean("enable_attention", True),
            "enable_drowsy": internal_boolean("enable_drowsy", True),
            "enable_yawn": internal_boolean("enable_yawn", True),
            "ear_threshold": internal_number("ear_threshold", 0.22),
            "mar_threshold": internal_number("mar_threshold", 0.60),
            "fps": internal_number("fps", 0.0),
            "consecutive_frames_eye": internal_integer(
                "consecutive_frames_eye", 45
            ),
            "consecutive_frames_mouth": internal_integer(
                "consecutive_frames_mouth", 60
            ),
            "drowsy_duration_sec": internal_number(
                "drowsy_duration_sec", 1.5
            ),
            "yawn_duration_sec": internal_number("yawn_duration_sec", 2.0),
            "blink_max_frames": internal_integer("blink_max_frames", 8),
            "blink_max_sec": internal_number("blink_max_sec", 0.30),
            "fatigue_ema_alpha": internal_number("ema_alpha", 0.40),
            "perclos_window_sec": internal_number(
                "perclos_window_sec", 60.0
            ),
            "perclos_threshold": internal_number("perclos_threshold", 0.15),
            "blink_freq_window_sec": internal_number(
                "blink_freq_window_sec", 60.0
            ),
            "blink_freq_high_threshold": internal_number(
                "blink_freq_high_threshold", 25.0
            ),
            "distraction_yaw_threshold_deg": internal_number(
                "distraction_yaw_threshold_deg", 30.0
            ),
            "distraction_yaw_release_deg": internal_number(
                "distraction_yaw_release_deg", 20.0
            ),
            "distraction_duration_sec": internal_number(
                "distraction_duration_sec", 1.0
            ),
            "distraction_frames": internal_integer("distraction_frames", 30),
            "distraction_grace_frames": internal_integer(
                "distraction_grace_frames", 8
            ),
            "nod_pitch_threshold_deg": internal_number(
                "nod_pitch_threshold_deg", -30.0
            ),
            "nod_pitch_release_deg": internal_number(
                "nod_pitch_release_deg", -18.0
            ),
            "nod_duration_sec": internal_number("nod_duration_sec", 2.0),
            "nod_frames": internal_integer("nod_frames", 60),
            "pose_ema_alpha": internal_number("pose_ema_alpha", 0.35),
        },
        "weights": {
            "external": number("w_ext", 0.35),
            "internal": number("w_int", 0.35),
            "cross": number("w_cross", 0.30),
        },
        "ema_alpha": number("ema_alpha", 0.40),
        "level_thresholds": normalized_thresholds,
        "ttc_thresholds": {
            "danger": number("ttc_danger", 1.5),
            "safe": number("ttc_safe", 6.0),
        },
        "distance_thresholds": {
            "danger": number("dist_danger", 3.0),
            "safe": number("dist_safe", 30.0),
        },
        "warning_floors": {"low": 0.30, "high": 0.70},
        "driver": {
            "corroboration_boost": number("corroboration_boost", 1.5),
            "contradiction_penalty": number("contradiction_penalty", 0.5),
            "single_signal_cap": number("single_signal_cap", 0.6),
            "ear_threshold": number("ear_threshold", 0.22),
            "ear_safe": number("ear_safe", 0.30),
            "perclos_threshold": number("perclos_threshold", 0.15),
            "perclos_saturation": 0.30,
            "blink_freq_normal": number("blink_freq_normal", 20.0),
            "blink_freq_high_threshold": number(
                "blink_freq_high_threshold", 25.0
            ),
            "blink_saturation": 35.0,
            "yaw_threshold": yaw_threshold,
            "yaw_safe": yaw_safe,
            "yaw_danger": yaw_danger,
            "w_perclos": number("w_perclos", 0.30),
            "w_drowsy": number("w_drowsy", 0.25),
            "w_nodding": number("w_nodding", 0.20),
            "w_blink_freq": number("w_blink_freq", 0.10),
            "w_yawn": number("w_yawn", 0.15),
            "w_distracted": number("w_distracted", 0.50),
            "w_nod_attention": number("w_nod_attention", 0.25),
            "w_yawn_attention": number("w_yawn_attention", 0.25),
            "fatigue_weight": number("fatigue_weight", 0.55),
            "attention_weight": number("attention_weight", 0.45),
            "cross_threshold": 0.30,
            "cross_scale": 0.20,
        },
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, _finite_float(value, lo)))


@dataclass(frozen=True)
class RSSMConfig:
    obs_dim: int = len(OBS_FIELDS)
    action_dim: int = len(ACTION_FIELDS)
    embed_dim: int = 64
    deter_dim: int = 64
    stoch_dim: int = 8
    classes: int = 8
    hidden_dim: int = 64
    unimix_ratio: float = 0.01
    free_nats: float = 1.0
    obs_loss_scale: float = 1.0
    risk_loss_scale: float = 2.0
    continue_loss_scale: float = 0.1
    dyn_loss_scale: float = 1.0
    rep_loss_scale: float = 0.1
    dt_sec: float = 0.25
    max_gap_sec: float = 1.5
    samples: int = 8
    cvar_alpha: float = 0.75
    max_decel: float = 8.0
    max_delay: float = 0.5

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None) -> "RSSMConfig":
        raw = dict(values or {})
        aliases = {
            "stoch_vars": "stoch_dim",
            "stoch_classes": "classes",
            "num_samples": "samples",
        }
        for old, new in aliases.items():
            if old in raw and new not in raw:
                raw[new] = raw[old]
        allowed = {item.name for item in fields(cls)}
        cfg = cls(**{key: value for key, value in raw.items() if key in allowed})
        integer_fields = (
            "obs_dim", "action_dim", "embed_dim", "deter_dim",
            "stoch_dim", "classes", "hidden_dim", "samples",
        )
        for name in integer_fields:
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("embed_dim", "deter_dim", "hidden_dim"):
            if getattr(cfg, name) > MAX_RSSM_NETWORK_DIM:
                raise ValueError(f"{name} exceeds the small-model limit")
        for name in ("stoch_dim", "classes"):
            if getattr(cfg, name) > MAX_RSSM_CATEGORICAL_DIM:
                raise ValueError(f"{name} exceeds the categorical-model limit")
        if cfg.samples > MAX_RSSM_SAMPLES:
            raise ValueError("samples exceeds the rollout safety limit")
        if cfg.obs_dim != len(OBS_FIELDS):
            raise ValueError(f"obs_dim must be {len(OBS_FIELDS)}, got {cfg.obs_dim}")
        if cfg.action_dim != len(ACTION_FIELDS):
            raise ValueError(f"action_dim must be {len(ACTION_FIELDS)}, got {cfg.action_dim}")

        float_fields = (
            "unimix_ratio", "free_nats", "obs_loss_scale", "risk_loss_scale",
            "continue_loss_scale", "dyn_loss_scale", "rep_loss_scale",
            "dt_sec", "max_gap_sec", "cvar_alpha", "max_decel", "max_delay",
        )
        for name in float_fields:
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= cfg.unimix_ratio < 1.0:
            raise ValueError("unimix_ratio must be in [0, 1)")
        nonnegative = (
            cfg.free_nats, cfg.obs_loss_scale, cfg.risk_loss_scale,
            cfg.continue_loss_scale, cfg.dyn_loss_scale, cfg.rep_loss_scale,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("free_nats and loss scales must be non-negative")
        if cfg.dt_sec <= 0.0 or cfg.max_gap_sec < cfg.dt_sec:
            raise ValueError("dt_sec must be positive and max_gap_sec >= dt_sec")
        if cfg.max_decel <= 0.0 or cfg.max_delay <= 0.0:
            raise ValueError("max_decel and max_delay must be positive")
        if not 0.0 <= cfg.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in [0, 1)")
        return cfg


class WorldStateCodec:
    """Stable 13-field schema shared by training and online inference."""

    OBS_FIELDS = OBS_FIELDS
    DISTANCE_SCALE = 100.0
    TTC_SCALE = 10.0
    CLOSING_SPEED_SCALE = 30.0

    @classmethod
    def encode(
        cls,
        state: Any,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        has_lead = bool(getattr(state, "has_lead_vehicle", False))
        distance = _finite_float(getattr(state, "lead_distance", 99.0), 99.0)
        ttc = _finite_float(getattr(state, "lead_ttc", 99.0), 99.0)
        closing = _finite_float(getattr(state, "closing_speed", 0.0), 0.0)
        if not has_lead:
            distance, ttc, closing = 99.0, 99.0, 0.0
        values = (
            float(has_lead),
            _clip(distance / cls.DISTANCE_SCALE),
            _clip(ttc / cls.TTC_SCALE),
            _clip(closing / cls.CLOSING_SPEED_SCALE),
            _clip(getattr(state, "lane_relevance", 1.0)),
            _clip(_finite_float(getattr(state, "lead_warning_level", 0.0)) / 3.0),
            _clip(getattr(state, "ext_score", 0.0)),
            _clip(getattr(state, "int_score", 0.0)),
            _clip(getattr(state, "cross_score", 0.0)),
            _clip(getattr(state, "fused_score", 0.0)),
            _clip(_finite_float(getattr(state, "fused_level", 0.0)) / 3.0),
            _clip(getattr(state, "fatigue_score", 0.0)),
            _clip(getattr(state, "attention_score", 0.0)),
        )
        result = torch.tensor(values, dtype=torch.float32, device=device)
        if not torch.isfinite(result).all():
            raise ValueError("encoded world state contains non-finite values")
        return result

    @classmethod
    def encode_batch(
        cls,
        states: Sequence[Any],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not states:
            return torch.empty((0, len(OBS_FIELDS)), dtype=torch.float32, device=device)
        return torch.stack([cls.encode(state, device=device) for state in states], dim=0)

    @classmethod
    def decode_geometry(cls, encoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        values = encoded.clamp(0.0, 1.0)
        has_lead = values[..., 0]
        distance = values[..., 1] * cls.DISTANCE_SCALE
        ttc = values[..., 2] * cls.TTC_SCALE
        closing = values[..., 3] * cls.CLOSING_SPEED_SCALE
        far = torch.full_like(distance, 99.0)
        distance = torch.where(has_lead >= 0.5, distance, far)
        ttc = torch.where(has_lead >= 0.5, ttc, far)
        closing = torch.where(has_lead >= 0.5, closing, torch.zeros_like(closing))
        return {
            "has_lead": has_lead,
            "distance": distance,
            "ttc": ttc,
            "closing_speed": closing,
        }

    @classmethod
    def risk_target(cls, encoded: torch.Tensor) -> torch.Tensor:
        values = encoded.clamp(0.0, 1.0)
        has_lead = values[..., 0]
        distance = values[..., 1] * cls.DISTANCE_SCALE
        ttc = values[..., 2] * cls.TTC_SCALE
        lane = 0.4 + 0.6 * values[..., 4]
        warning = values[..., 5] * 3.0
        ttc_risk = (1.0 - (ttc - 1.2) / (5.0 - 1.2)).clamp(0.0, 1.0)
        dist_risk = (1.0 - (distance - 3.0) / (30.0 - 3.0)).clamp(0.0, 1.0)
        external = torch.maximum(ttc_risk, dist_risk) * lane * has_lead
        external = torch.maximum(external, torch.where(warning >= 1.5, 0.75, 0.0))
        external = torch.maximum(external, torch.where((warning >= 0.5) & (warning < 1.5), 0.35, 0.0))
        fused = values[..., 9]
        driver = values[..., 7] * 0.35
        return torch.maximum(torch.maximum(external, fused), driver).clamp(0.0, 1.0)


class ActionCodec:
    ACTION_FIELDS = ACTION_FIELDS

    def __init__(self, max_decel: float = 8.0, max_delay: float = 0.5):
        self.max_decel = max(0.1, float(max_decel))
        self.max_delay = max(0.01, float(max_delay))

    @staticmethod
    def values(action: Any) -> Tuple[float, float]:
        if isinstance(action, Mapping):
            if "target_decel" in action:
                raw_decel = action["target_decel"]
            elif "decel" in action:
                raw_decel = action["decel"]
            else:
                raise ValueError("action mapping must include target_decel or decel")
            raw_delay = action.get(
                "remaining_delay_sec",
                action.get("response_delay_sec", action.get("delay", 0.0)),
            )
        elif isinstance(action, (int, float)) and not isinstance(action, bool):
            raw_decel, raw_delay = action, 0.0
        elif hasattr(action, "target_decel"):
            raw_decel = getattr(action, "target_decel")
            raw_delay = getattr(action, "response_delay_sec", 0.0)
        else:
            raise TypeError("action must expose target_decel and response_delay_sec")
        try:
            decel = float(raw_decel)
            delay = float(raw_delay)
        except (TypeError, ValueError) as exc:
            raise ValueError("action deceleration and delay must be numeric") from exc
        if not math.isfinite(decel) or not math.isfinite(delay):
            raise ValueError("action deceleration and delay must be finite")
        if decel < 0.0 or delay < 0.0:
            raise ValueError("action deceleration and delay must be non-negative")
        return decel, delay

    def encode(
        self,
        action: Any = None,
        valid: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = action is not None
        if not valid:
            return torch.zeros(len(ACTION_FIELDS), dtype=torch.float32, device=device)
        decel, delay = self.values(action)
        if decel > self.max_decel or delay > self.max_delay:
            raise ValueError("action exceeds codec normalization range")
        values = (
            _clip(decel / self.max_decel),
            _clip(delay / self.max_delay),
            1.0,
        )
        return torch.tensor(values, dtype=torch.float32, device=device)


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor

    def detach(self) -> "RSSMState":
        return RSSMState(self.deter.detach(), self.stoch.detach(), self.logits.detach())

    def repeat_interleave(self, repeats: int) -> "RSSMState":
        return RSSMState(
            self.deter.repeat_interleave(repeats, dim=0),
            self.stoch.repeat_interleave(repeats, dim=0),
            self.logits.repeat_interleave(repeats, dim=0),
        )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TinyRSSM(nn.Module):
    """Compact categorical recurrent state-space model."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any] | RSSMConfig] = None,
        **kwargs: Any,
    ):
        super().__init__()
        base = asdict(config) if isinstance(config, RSSMConfig) else dict(config or {})
        cfg = RSSMConfig.from_mapping({**base, **kwargs})
        self.config = cfg
        stochastic_size = cfg.stoch_dim * cfg.classes
        feature_size = cfg.deter_dim + stochastic_size

        self.encoder = _mlp(cfg.obs_dim, cfg.hidden_dim, cfg.embed_dim)
        self.action_encoder = _mlp(cfg.action_dim, cfg.hidden_dim, cfg.hidden_dim)
        self.dynamics_input = _mlp(
            stochastic_size + cfg.hidden_dim,
            cfg.hidden_dim,
            cfg.hidden_dim,
        )
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.deter_dim)
        self.prior_net = _mlp(
            cfg.deter_dim,
            cfg.hidden_dim,
            stochastic_size,
        )
        self.posterior_net = _mlp(
            cfg.deter_dim + cfg.embed_dim,
            cfg.hidden_dim,
            stochastic_size,
        )
        self.decoder = _mlp(feature_size, cfg.hidden_dim, cfg.obs_dim)
        self.risk_head = _mlp(feature_size, cfg.hidden_dim, 1)
        self.continue_head = _mlp(feature_size, cfg.hidden_dim, 1)

    @property
    def feature_dim(self) -> int:
        return self.config.deter_dim + self.config.stoch_dim * self.config.classes

    def get_config(self) -> Dict[str, Any]:
        result = asdict(self.config)
        result["action_fields"] = list(ACTION_FIELDS)
        result["normalization"] = {
            "distance_scale": WorldStateCodec.DISTANCE_SCALE,
            "ttc_scale": WorldStateCodec.TTC_SCALE,
            "closing_speed_scale": WorldStateCodec.CLOSING_SPEED_SCALE,
            "max_decel": self.config.max_decel,
            "max_delay": self.config.max_delay,
        }
        return result

    def initial(self, batch_size: int, device: Optional[torch.device] = None) -> RSSMState:
        if device is None:
            device = next(self.parameters()).device
        deter = torch.zeros(batch_size, self.config.deter_dim, device=device)
        logits = torch.zeros(
            batch_size,
            self.config.stoch_dim,
            self.config.classes,
            device=device,
        )
        indices = torch.zeros(batch_size, self.config.stoch_dim, dtype=torch.long, device=device)
        stoch = F.one_hot(indices, self.config.classes).float()
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def feature(self, state: RSSMState) -> torch.Tensor:
        return torch.cat((state.deter, state.stoch.flatten(-2)), dim=-1)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("RSSM categorical logits are non-finite")
        probs = torch.softmax(logits, dim=-1)
        mix = self.config.unimix_ratio
        mixed = (1.0 - mix) * probs + mix / self.config.classes
        if not torch.isfinite(mixed).all():
            raise FloatingPointError("RSSM categorical probabilities are non-finite")
        return mixed

    def _sample(self, logits: torch.Tensor, sample: bool = True) -> torch.Tensor:
        probs = self.probabilities(logits)
        if not sample:
            indices = probs.argmax(dim=-1)
            return F.one_hot(indices, self.config.classes).float()
        if self.training:
            return F.gumbel_softmax(torch.log(probs.clamp_min(1e-8)), tau=1.0, hard=True, dim=-1)
        indices = torch.distributions.Categorical(probs=probs).sample()
        return F.one_hot(indices, self.config.classes).float()

    def imagine_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        sample: bool = True,
    ) -> RSSMState:
        action_embedding = self.action_encoder(action)
        dynamics_input = self.dynamics_input(
            torch.cat((previous.stoch.flatten(-2), action_embedding), dim=-1)
        )
        deter = self.gru(dynamics_input, previous.deter)
        logits = self.prior_net(deter).reshape(
            -1,
            self.config.stoch_dim,
            self.config.classes,
        )
        return RSSMState(deter=deter, stoch=self._sample(logits, sample=sample), logits=logits)

    def observe_step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        observation: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[RSSMState, RSSMState]:
        prior = self.imagine_step(previous, action, sample=sample)
        embedding = self.encoder(observation)
        logits = self.posterior_net(torch.cat((prior.deter, embedding), dim=-1)).reshape(
            -1,
            self.config.stoch_dim,
            self.config.classes,
        )
        posterior = RSSMState(
            deter=prior.deter,
            stoch=self._sample(logits, sample=sample),
            logits=logits,
        )
        return posterior, prior

    def decode_observation(self, state: RSSMState) -> torch.Tensor:
        return torch.sigmoid(self.decoder(self.feature(state)))

    def predict_risk(self, state: RSSMState) -> torch.Tensor:
        return torch.sigmoid(self.risk_head(self.feature(state))).squeeze(-1)

    def predict_continue(self, state: RSSMState) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(self.feature(state))).squeeze(-1)

    def _reset_where(self, state: RSSMState, reset_mask: torch.Tensor) -> RSSMState:
        initial = self.initial(state.deter.shape[0], state.deter.device)
        deter_mask = reset_mask.reshape(-1, 1)
        stoch_mask = reset_mask.reshape(-1, 1, 1)
        return RSSMState(
            deter=torch.where(deter_mask, initial.deter, state.deter),
            stoch=torch.where(stoch_mask, initial.stoch, state.stoch),
            logits=torch.where(stoch_mask, initial.logits, state.logits),
        )

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        is_first: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Tuple[List[RSSMState], List[RSSMState]]:
        if observations.ndim != 3 or observations.shape[-1] != self.config.obs_dim:
            raise ValueError("observations must have shape [batch, time, obs_dim]")
        if observations.shape[1] == 0:
            raise ValueError("observation sequences must contain at least one step")
        if (
            actions.ndim != 3
            or actions.shape[:2] != observations.shape[:2]
            or actions.shape[-1] != self.config.action_dim
        ):
            raise ValueError("actions must have shape [batch, time, action_dim]")
        if is_first is not None and is_first.shape != observations.shape[:2]:
            raise ValueError("is_first must have shape [batch, time]")
        state = self.initial(observations.shape[0], observations.device)
        posteriors: List[RSSMState] = []
        priors: List[RSSMState] = []
        for step in range(observations.shape[1]):
            step_action = actions[:, step]
            if is_first is not None:
                reset_mask = is_first[:, step].bool()
                state = self._reset_where(state, reset_mask)
                step_action = torch.where(
                    reset_mask.unsqueeze(-1),
                    torch.zeros_like(step_action),
                    step_action,
                )
            state, prior = self.observe_step(
                state,
                step_action,
                observations[:, step],
                sample=sample,
            )
            posteriors.append(state)
            priors.append(prior)
        return posteriors, priors

    def _kl(self, posterior_logits: torch.Tensor, prior_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        posterior = self.probabilities(posterior_logits)
        prior = self.probabilities(prior_logits)
        log_posterior = posterior.clamp_min(1e-8).log()
        log_prior = prior.clamp_min(1e-8).log()
        dyn = (posterior.detach() * (log_posterior.detach() - log_prior)).sum(dim=(-1, -2))
        rep = (posterior * (log_posterior - log_prior.detach())).sum(dim=(-1, -2))
        return dyn, rep

    def loss_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        risk_targets: Optional[torch.Tensor] = None,
        continues: Optional[torch.Tensor] = None,
        is_first: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        posteriors, priors = self.observe_sequence(
            observations,
            actions,
            is_first=is_first,
            sample=True,
        )
        if risk_targets is None:
            risk_targets = WorldStateCodec.risk_target(observations)
        if continues is None:
            continues = torch.ones_like(risk_targets)
        if risk_targets.ndim == 3 and risk_targets.shape[-1] == 1:
            risk_targets = risk_targets.squeeze(-1)
        if continues.ndim == 3 and continues.shape[-1] == 1:
            continues = continues.squeeze(-1)
        expected_shape = observations.shape[:2]
        if risk_targets.shape != expected_shape or continues.shape != expected_shape:
            raise ValueError("risk_targets and continues must have shape [batch, time]")
        risk_targets = risk_targets.clamp(0.0, 1.0)
        continues = continues.clamp(0.0, 1.0)
        if valid_mask is None:
            valid_mask = torch.ones_like(risk_targets)
        if valid_mask.shape != expected_shape:
            raise ValueError("valid_mask must have shape [batch, time]")
        valid_mask = valid_mask.float().clamp(0.0, 1.0)
        denominator = valid_mask.sum().clamp_min(1.0)

        obs_terms: List[torch.Tensor] = []
        risk_terms: List[torch.Tensor] = []
        continue_terms: List[torch.Tensor] = []
        dyn_terms: List[torch.Tensor] = []
        rep_terms: List[torch.Tensor] = []
        for step, (posterior, prior) in enumerate(zip(posteriors, priors)):
            feature = self.feature(posterior)
            decoded = torch.sigmoid(self.decoder(feature))
            obs_terms.append(F.smooth_l1_loss(decoded, observations[:, step], reduction="none").mean(-1))
            risk_terms.append(
                F.binary_cross_entropy_with_logits(
                    self.risk_head(feature).squeeze(-1),
                    risk_targets[:, step],
                    reduction="none",
                )
            )
            continue_terms.append(
                F.binary_cross_entropy_with_logits(
                    self.continue_head(feature).squeeze(-1),
                    continues[:, step],
                    reduction="none",
                )
            )
            dyn, rep = self._kl(posterior.logits, prior.logits)
            dyn_terms.append(dyn.clamp_min(self.config.free_nats))
            rep_terms.append(rep.clamp_min(self.config.free_nats))

        def masked_mean(items: Sequence[torch.Tensor]) -> torch.Tensor:
            stacked = torch.stack(list(items), dim=1)
            return (stacked * valid_mask).sum() / denominator

        obs_loss = masked_mean(obs_terms)
        risk_loss = masked_mean(risk_terms)
        continue_loss = masked_mean(continue_terms)
        dyn_loss = masked_mean(dyn_terms)
        rep_loss = masked_mean(rep_terms)
        total = (
            self.config.obs_loss_scale * obs_loss
            + self.config.risk_loss_scale * risk_loss
            + self.config.continue_loss_scale * continue_loss
            + self.config.dyn_loss_scale * dyn_loss
            + self.config.rep_loss_scale * rep_loss
        )
        return {
            "loss": total,
            "total_loss": total,
            "obs_loss": obs_loss,
            "risk_loss": risk_loss,
            "continue_loss": continue_loss,
            "dyn_loss": dyn_loss,
            "rep_loss": rep_loss,
        }

    training_loss = loss_sequence
    compute_loss = loss_sequence
    loss = loss_sequence

    def checkpoint_payload(self, metrics: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "obs_fields": list(OBS_FIELDS),
            "model_config": self.get_config(),
            "model_state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "metrics": dict(metrics or {}),
        }

    def save_checkpoint(self, path: str | Path, metrics: Optional[Mapping[str, Any]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(metrics), destination)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        expected_sha256: Optional[str] = None,
    ) -> Tuple["TinyRSSM", Dict[str, Any]]:
        checkpoint_bytes = _read_checkpoint_bytes(path)
        if expected_sha256 is not None:
            actual_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
            if not hmac.compare_digest(actual_digest, expected_sha256):
                raise ValueError("RSSM checkpoint SHA-256 mismatch")
        payload = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location=device,
            weights_only=True,
        )
        if not isinstance(payload, Mapping):
            raise ValueError("RSSM checkpoint must contain a mapping")
        required = {"schema_version", "obs_fields", "model_config", "model_state_dict", "metrics"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"RSSM checkpoint missing keys: {sorted(missing)}")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("RSSM checkpoint schema_version mismatch")
        if tuple(payload["obs_fields"]) != OBS_FIELDS:
            raise ValueError("RSSM checkpoint observation schema mismatch")
        if not isinstance(payload["model_config"], Mapping):
            raise ValueError("RSSM checkpoint model_config must be a mapping")
        model_config = dict(payload["model_config"])
        if tuple(model_config.get("action_fields", ())) != ACTION_FIELDS:
            raise ValueError("RSSM checkpoint action schema mismatch")
        config = RSSMConfig.from_mapping(model_config)
        normalization = model_config.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("RSSM checkpoint normalization metadata missing")
        expected_normalization = {
            "distance_scale": WorldStateCodec.DISTANCE_SCALE,
            "ttc_scale": WorldStateCodec.TTC_SCALE,
            "closing_speed_scale": WorldStateCodec.CLOSING_SPEED_SCALE,
            "max_decel": config.max_decel,
            "max_delay": config.max_delay,
        }
        for name, expected in expected_normalization.items():
            actual = _finite_float(normalization.get(name), float("nan"))
            if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"RSSM checkpoint normalization mismatch for {name}"
                )
        model = cls(config).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        for parameter in model.state_dict().values():
            if not torch.isfinite(parameter).all():
                raise ValueError("RSSM checkpoint contains non-finite weights")
        model.eval()
        return model, dict(payload.get("metrics", {}))


@dataclass
class RSSMForecast:
    peak_risk: float
    terminal_risk: float
    min_distance: float
    min_ttc: float
    trajectory: List[Dict[str, float]]
    uncertainty: float



_LOCKED_RUNTIME_FIELDS = (
    "obs_dim",
    "action_dim",
    "embed_dim",
    "deter_dim",
    "stoch_dim",
    "classes",
    "hidden_dim",
    "unimix_ratio",
    "dt_sec",
    "max_decel",
    "max_delay",
)


def _validated_runtime_config(
    model: TinyRSSM,
    raw_config: Mapping[str, Any],
) -> RSSMConfig:
    raw = dict(raw_config)
    for old, new in {
        "stoch_vars": "stoch_dim",
        "stoch_classes": "classes",
        "num_samples": "samples",
    }.items():
        if old in raw and new not in raw:
            raw[new] = raw[old]

    trained = model.get_config()
    for name in _LOCKED_RUNTIME_FIELDS:
        if name not in raw:
            continue
        expected = trained[name]
        actual = raw[name]
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-7)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"runtime {name}={actual!r} does not match checkpoint value {expected!r}"
            )

    merged = dict(trained)
    for name in ("samples", "cvar_alpha", "max_gap_sec"):
        if name in raw:
            merged[name] = raw[name]
    return RSSMConfig.from_mapping(merged)

class RSSMInferenceEngine:
    """Stateful posterior update and risk-sensitive candidate imagination."""

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        model: Optional[TinyRSSM] = None,
    ):
        raw = dict(config or {})
        self.enabled = bool(raw.get("enabled", True))
        self.device = torch.device(str(raw.get("device", "cpu")))
        self.model: Optional[TinyRSSM] = None
        self.metrics: Dict[str, Any] = {}
        self.ready = False
        self.status = "disabled"

        if model is not None:
            self.config = _validated_runtime_config(model, raw)
            self.model = model.to(self.device).eval()
            self.ready = self.enabled
            self.status = "ready" if self.ready else "disabled"
        else:
            self.config = RSSMConfig.from_mapping(raw)
            checkpoint = str(raw.get("checkpoint", "")).strip()
            if self.enabled and checkpoint:
                try:
                    require_digest = raw.get("require_checkpoint_sha256", True)
                    if not isinstance(require_digest, bool):
                        raise ValueError(
                            "require_checkpoint_sha256 must be boolean"
                        )
                    expected_digest = str(
                        raw.get("checkpoint_sha256", "")
                    ).strip().lower()
                    if require_digest and not expected_digest:
                        raise ValueError(
                            "RSSM checkpoint_sha256 is required"
                        )
                    if expected_digest:
                        if len(expected_digest) != 64 or any(
                            char not in "0123456789abcdef"
                            for char in expected_digest
                        ):
                            raise ValueError(
                                "RSSM checkpoint_sha256 must be 64 hexadecimal characters"
                            )
                    loaded, metrics = TinyRSSM.load_checkpoint(
                        checkpoint,
                        self.device,
                        expected_sha256=expected_digest or None,
                    )
                    self.config = _validated_runtime_config(loaded, raw)
                    self.model = loaded
                    self.metrics = metrics
                    self.ready = True
                    self.status = "ready"
                except Exception as exc:
                    self.status = f"fallback ({type(exc).__name__}: {exc})"
            elif self.enabled:
                self.status = "fallback (checkpoint not configured)"

        self.codec = WorldStateCodec()
        self.action_codec = ActionCodec(self.config.max_decel, self.config.max_delay)
        self._belief: Optional[RSSMState] = None
        self._last_observe_time: Optional[float] = None
        self._last_applied_signature: Optional[Tuple[str, str]] = None
        self._last_applied_decel: Optional[float] = None
        self._last_applied_remaining_delay = 0.0
        self._last_seen_applied_signature: Optional[Tuple[str, str]] = None
        self._last_seen_applied_decel: Optional[float] = None
        self._has_seen_applied_action = False
        self._saw_throttled_receipt_since_observe = False
        self._pending_action_change = False
        self.observe_count = 0
        self.last_uncertainty = 0.0

    @classmethod
    def from_model(
        cls,
        model: TinyRSSM,
        config: Optional[Mapping[str, Any]] = None,
        device: str | torch.device = "cpu",
    ) -> "RSSMInferenceEngine":
        raw = dict(config or {})
        raw["device"] = str(device)
        return cls(raw, model=model)

    @property
    def has_belief(self) -> bool:
        return self._belief is not None

    def reset(self) -> None:
        self._belief = None
        self._last_observe_time = None
        self._last_applied_signature = None
        self._last_applied_decel = None
        self._last_applied_remaining_delay = 0.0
        self._last_seen_applied_signature = None
        self._last_seen_applied_decel = None
        self._has_seen_applied_action = False
        self._saw_throttled_receipt_since_observe = False
        self._pending_action_change = False
        self.observe_count = 0
        self.last_uncertainty = 0.0

    @staticmethod
    def _action_signature(action: Any, decel: float) -> Tuple[str, str]:
        if isinstance(action, Mapping):
            identity = action.get("name", action.get("stable_action_id"))
        else:
            identity = getattr(action, "name", getattr(action, "stable_action_id", None))
        if identity is not None:
            return "id", str(identity)
        return "decel", f"{decel:.2f}"

    def _validated_action_details(
        self,
        action: Any,
    ) -> Tuple[
        bool,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[Tuple[str, str]],
        Optional[bool],
        bool,
    ]:
        if action is None:
            return False, None, None, None, None, None, False
        decel, default_delay = self.action_codec.values(action)
        has_remaining_delay = False
        explicit_changed: Optional[bool] = None
        if isinstance(action, Mapping):
            has_remaining_delay = "remaining_delay_sec" in action
            if "response_delay_sec" in action:
                raw_response_delay = action["response_delay_sec"]
            elif "delay" in action:
                raw_response_delay = action["delay"]
            else:
                raw_response_delay = 0.0
            raw_remaining_delay = action.get(
                "remaining_delay_sec", raw_response_delay
            )
            if "action_changed" in action:
                raw_changed = action["action_changed"]
                if not isinstance(raw_changed, bool):
                    raise ValueError("action_changed must be boolean")
                explicit_changed = raw_changed
        elif hasattr(action, "target_decel"):
            has_remaining_delay = hasattr(action, "remaining_delay_sec")
            raw_response_delay = getattr(
                action, "response_delay_sec", default_delay
            )
            raw_remaining_delay = getattr(
                action, "remaining_delay_sec", raw_response_delay
            )
            if hasattr(action, "action_changed"):
                raw_changed = getattr(action, "action_changed")
                if not isinstance(raw_changed, bool):
                    raise ValueError("action_changed must be boolean")
                explicit_changed = raw_changed
        else:
            raw_response_delay = default_delay
            raw_remaining_delay = default_delay
        try:
            response_delay = float(raw_response_delay)
            remaining_delay = float(raw_remaining_delay)
        except (TypeError, ValueError) as exc:
            raise ValueError("action delays must be numeric") from exc
        delays = (response_delay, remaining_delay)
        if not all(math.isfinite(value) and value >= 0.0 for value in delays):
            raise ValueError("action delays must be finite and non-negative")
        if decel > self.config.max_decel or any(
            value > self.config.max_delay for value in delays
        ):
            raise ValueError("applied action exceeds normalization range")
        if response_delay > self.config.dt_sec + 1e-6:
            raise ValueError("applied action response delay exceeds training step")
        signature = self._action_signature(action, decel)
        return (
            True,
            decel,
            response_delay,
            remaining_delay,
            signature,
            explicit_changed,
            has_remaining_delay,
        )

    def _record_throttled_action(
        self,
        signature: Optional[Tuple[str, str]],
        decel: Optional[float],
        explicit_changed: Optional[bool],
    ) -> None:
        if decel is None and self._last_seen_applied_decel is None:
            decel_changed = False
        elif decel is None or self._last_seen_applied_decel is None:
            decel_changed = True
        else:
            decel_changed = not math.isclose(
                decel, self._last_seen_applied_decel, abs_tol=1e-3
            )
        if (
            self._has_seen_applied_action
            and (
                explicit_changed is True
                or signature != self._last_seen_applied_signature
                or decel_changed
            )
        ):
            self._pending_action_change = True
        self._last_seen_applied_signature = signature
        self._last_seen_applied_decel = decel
        self._has_seen_applied_action = True
        self._saw_throttled_receipt_since_observe = True

    def _matches_applied_action(self, action: Any, decel: float) -> bool:
        if self._last_applied_signature is None or self._last_applied_decel is None:
            return False
        candidate_signature = self._action_signature(action, decel)
        same_decel = math.isclose(decel, self._last_applied_decel, abs_tol=1e-3)
        return candidate_signature == self._last_applied_signature and same_decel

    def observe(
        self,
        world_state: Any,
        applied_action: Any = None,
        timestamp: Optional[float] = None,
    ) -> bool:
        if not self.ready or self.model is None:
            return False
        if timestamp is None:
            timestamp = _finite_float(getattr(world_state, "timestamp", time.monotonic()), time.monotonic())
        timestamp = _finite_float(timestamp, time.monotonic())
        try:
            (
                action_provided,
                applied_decel,
                response_delay,
                remaining_delay,
                action_signature,
                explicit_changed,
                has_remaining_delay,
            ) = self._validated_action_details(applied_action)
        except (TypeError, ValueError) as exc:
            self.reset()
            self.status = f"fallback ({exc})"
            return False
        belief = self._belief
        reanchored = belief is None
        if self._last_observe_time is not None:
            elapsed = timestamp - self._last_observe_time
            if elapsed < -1e-9 or elapsed > self.config.max_gap_sec:
                self.reset()
                belief = None
                reanchored = True
            elif elapsed <= 1e-9:
                # The caller may reuse a cached sensor frame. An exact duplicate
                # observation is not a new transition and must not reset/re-anchor
                # the belief or consume another posterior update.
                self._record_throttled_action(
                    action_signature, applied_decel, explicit_changed
                )
                return False
            elif elapsed < self.config.dt_sec:
                self._record_throttled_action(
                    action_signature, applied_decel, explicit_changed
                )
                return False
            elif elapsed - self.config.dt_sec > 0.40 * self.config.dt_sec:
                # A stale latest action cannot safely reconstruct the missing
                # actuator history. Re-anchor on the current observation.
                self.reset()
                belief = None
                reanchored = True

        if self._saw_throttled_receipt_since_observe and self._has_seen_applied_action:
            if applied_decel is None and self._last_seen_applied_decel is None:
                decel_changed = False
            elif applied_decel is None or self._last_seen_applied_decel is None:
                decel_changed = True
            else:
                decel_changed = not math.isclose(
                    applied_decel, self._last_seen_applied_decel, abs_tol=1e-3
                )
            if (
                explicit_changed is True
                or action_signature != self._last_seen_applied_signature
                or decel_changed
            ):
                self._pending_action_change = True

        if self._pending_action_change and belief is not None:
            self.reset()
            belief = None
            reanchored = True

        first_observation = belief is None
        action_valid = action_provided and not first_observation
        held_action = (
            action_valid
            and explicit_changed is not True
            and action_signature == self._last_applied_signature
            and self._last_applied_decel is not None
            and math.isclose(
                float(applied_decel), self._last_applied_decel, abs_tol=1e-3
            )
        )
        transition_delay = 0.0 if held_action else float(response_delay or 0.0)
        action_for_codec = {
            "target_decel": float(applied_decel or 0.0),
            "response_delay_sec": transition_delay,
        }

        observation = self.codec.encode(world_state, device=self.device).unsqueeze(0)
        action = self.action_codec.encode(
            action_for_codec,
            valid=action_valid,
            device=self.device,
        ).unsqueeze(0)
        try:
            belief = belief or self.model.initial(1, self.device)
            with torch.no_grad():
                posterior, _ = self.model.observe_step(
                    belief,
                    action,
                    observation,
                    sample=False,
                )
        except Exception as exc:
            self.reset()
            self.ready = False
            self.status = (
                f"fallback (RSSM posterior {type(exc).__name__}; model disabled)"
            )
            return False
        tensors = (posterior.deter, posterior.stoch, posterior.logits)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            self.reset()
            self.ready = False
            self.status = "fallback (non-finite posterior; model disabled)"
            return False
        self._belief = posterior.detach()
        if reanchored or self._last_observe_time is None:
            self._last_observe_time = timestamp
        else:
            self._last_observe_time += self.config.dt_sec
        if action_provided:
            if has_remaining_delay:
                future_delay = float(remaining_delay or 0.0)
            else:
                # response_delay_sec belongs to the just-observed transition;
                # only an explicit remaining delay carries into imagination.
                future_delay = 0.0
        else:
            future_delay = 0.0
        self._last_applied_signature = action_signature if action_provided else None
        self._last_applied_decel = applied_decel if action_provided else None
        self._last_applied_remaining_delay = future_delay
        self._last_seen_applied_signature = action_signature
        self._last_seen_applied_decel = applied_decel
        self._has_seen_applied_action = True
        self._saw_throttled_receipt_since_observe = False
        self._pending_action_change = False
        self.status = "ready"
        self.observe_count += 1
        return True

    def _cvar(self, values: torch.Tensor) -> torch.Tensor:
        """Empirical upper-tail CVaR with a fractional boundary sample."""
        sample_count = values.shape[-1]
        tail_mass = (1.0 - self.config.cvar_alpha) * sample_count
        ordered = values.sort(dim=-1, descending=True).values
        # A tail smaller than one empirical sample consists entirely of the
        # worst observation. This also avoids rounding a valid tiny tail to 0.
        if tail_mass <= 1.0:
            return ordered[..., 0]
        nearest_integer = round(tail_mass)
        if math.isclose(tail_mass, nearest_integer, abs_tol=1e-12):
            tail_mass = float(nearest_integer)
        full_count = int(math.floor(tail_mass))
        fractional = tail_mass - full_count
        total = ordered[..., :full_count].sum(dim=-1)
        if fractional > 0.0:
            total = total + fractional * ordered[..., full_count]
        return total / tail_mass

    def predict_many(
        self,
        actions: Sequence[Any],
        horizons_sec: Sequence[float],
    ) -> Dict[str, RSSMForecast]:
        if not self.ready or self.model is None or self._belief is None:
            return {}
        if not actions:
            return {}
        if self._pending_action_change:
            self.status = "fallback (pending action transition)"
            return {}
        horizons: List[float] = []
        for raw_horizon in horizons_sec:
            try:
                horizon = float(raw_horizon)
            except (TypeError, ValueError) as exc:
                raise ValueError("RSSM horizons must be numeric") from exc
            if not math.isfinite(horizon) or horizon <= 0.0:
                raise ValueError("RSSM horizons must be finite and positive")
            horizons.append(max(self.config.dt_sec, horizon))
        if not horizons:
            horizons = [self.config.dt_sec]
        max_horizon = max(horizons)
        if max_horizon > self.config.dt_sec * MAX_IMAGINATION_STEPS:
            raise ValueError("RSSM horizon exceeds the imagination step limit")
        steps = max(1, int(math.ceil(max_horizon / self.config.dt_sec)))
        samples = self.config.samples
        action_count = len(actions)
        latent = self._belief.repeat_interleave(action_count * samples)

        action_values = [self.action_codec.values(action) for action in actions]
        initial_delays = [
            self._last_applied_remaining_delay
            if self._matches_applied_action(action, decel)
            else delay
            for action, (decel, delay) in zip(actions, action_values)
        ]
        risk_steps: List[torch.Tensor] = []
        observation_steps: List[torch.Tensor] = []
        try:
            with torch.no_grad():
                for step in range(steps):
                    elapsed = step * self.config.dt_sec
                    vectors = []
                    for action_index, (decel, _) in enumerate(action_values):
                        remaining_delay = max(
                            0.0, initial_delays[action_index] - elapsed
                        )
                        vectors.append(
                            self.action_codec.encode(
                                {"target_decel": max(0.0, decel), "response_delay_sec": remaining_delay},
                                valid=True,
                                device=self.device,
                            )
                        )
                    action_tensor = torch.stack(vectors, dim=0).repeat_interleave(samples, dim=0)
                    latent = self.model.imagine_step(latent, action_tensor, sample=True)
                    observation = self.model.decode_observation(latent)
                    learned_risk = self.model.predict_risk(latent)
                    derived_risk = self.codec.risk_target(observation)
                    termination_risk = 1.0 - self.model.predict_continue(latent)
                    risk = torch.maximum(
                        torch.maximum(learned_risk, derived_risk), termination_risk
                    )
                    if not torch.isfinite(observation).all() or not torch.isfinite(risk).all():
                        self.reset()
                        self.ready = False
                        self.status = "fallback (non-finite imagination; model disabled)"
                        raise RuntimeError("RSSM imagination produced non-finite output")
                    observation_steps.append(observation.reshape(action_count, samples, -1))
                    risk_steps.append(risk.reshape(action_count, samples))
    
        except Exception as exc:
            self.reset()
            self.ready = False
            self.status = (
                f"fallback (RSSM imagination {type(exc).__name__}; model disabled)"
            )
            return {}

        risks = torch.stack(risk_steps, dim=0)
        observations = torch.stack(observation_steps, dim=0)
        geometry = self.codec.decode_geometry(observations)
        peak_per_rollout = risks.max(dim=0).values
        terminal_index = max(0, min(steps - 1, int(math.ceil(max(horizons) / self.config.dt_sec)) - 1))
        peak = self._cvar(peak_per_rollout)
        terminal = self._cvar(risks[terminal_index])
        uncertainty = peak_per_rollout.std(dim=-1, unbiased=False)

        forecasts: Dict[str, RSSMForecast] = {}
        for action_index, action in enumerate(actions):
            trajectory: List[Dict[str, float]] = []
            for horizon in horizons:
                index = max(0, min(steps - 1, int(math.ceil(horizon / self.config.dt_sec)) - 1))
                distance_values = geometry["distance"][index, action_index]
                ttc_values = geometry["ttc"][index, action_index]
                closing_values = geometry["closing_speed"][index, action_index]
                trajectory.append({
                    "horizon": float(horizon),
                    "distance": float(
                        torch.quantile(
                            distance_values, 0.10, interpolation="lower"
                        ).cpu()
                    ),
                    "closing_speed": float(closing_values.mean().cpu()),
                    "ttc": float(
                        torch.quantile(
                            ttc_values, 0.10, interpolation="lower"
                        ).cpu()
                    ),
                    "risk": float(self._cvar(risks[index, action_index].unsqueeze(0)).squeeze(0).cpu()),
                })
            min_distance = geometry["distance"][:, action_index].min(dim=0).values
            min_ttc = geometry["ttc"][:, action_index].min(dim=0).values
            name = str(getattr(action, "name", action_index))
            forecasts[name] = RSSMForecast(
                peak_risk=float(peak[action_index].clamp(0.0, 1.0).cpu()),
                terminal_risk=float(terminal[action_index].clamp(0.0, 1.0).cpu()),
                min_distance=float(
                    torch.quantile(
                        min_distance, 0.10, interpolation="lower"
                    ).clamp(0.0, 99.0).cpu()
                ),
                min_ttc=float(
                    torch.quantile(
                        min_ttc, 0.10, interpolation="lower"
                    ).clamp(0.0, 99.0).cpu()
                ),
                trajectory=trajectory,
                uncertainty=float(uncertainty[action_index].cpu()),
            )
        self.last_uncertainty = max(item.uncertainty for item in forecasts.values())
        return forecasts


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "FUSION_FEATURE_CONTRACT_VERSION",
    "MAX_IMAGINATION_STEPS",
    "MIN_RSSM_TRAINING_STEPS",
    "SCHEMA_VERSION",
    "OBS_FIELDS",
    "ACTION_FIELDS",
    "build_fusion_feature_contract",
    "checkpoint_sha256",
    "RSSMConfig",
    "WorldStateCodec",
    "ActionCodec",
    "RSSMState",
    "TinyRSSM",
    "RSSMForecast",
    "RSSMInferenceEngine",
]
