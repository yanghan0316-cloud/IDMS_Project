"""Pre-train the tiny RSSM on safe, synthetic closed-loop trajectories.

The decision CSVs in ``logs/`` contain *recommended* MRM actions.  They are
deliberately not consumed here: a world model must be conditioned on an action
that was actually executed and confirmed by the vehicle/actuator interface.
Until such telemetry is available, this script uses the four sampled actions
below as explicit inputs to a small, physically plausible simulator.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from src.core.rssm_world_model import (
    ACTION_CONTRACT_VERSION,
    ActionCodec,
    TinyRSSM,
    WorldStateCodec,
    build_fusion_feature_contract,
    checkpoint_sha256,
)


ACTION_NAMES: Tuple[str, ...] = (
    "KEEP",
    "SLOW_DOWN",
    "BRAKE",
    "EMERGENCY_BRAKE",
)
ACTION_DECEL_MPS2 = (0.0, 1.5, 3.5, 6.5)
ACTION_RESPONSE_DELAY_S = (0.0, 0.20, 0.15, 0.10)
UNKNOWN_ACTION_EPISODE_PROBABILITY = 0.20
UNKNOWN_ACTION_STEP_PROBABILITY = 0.10
OBS_FIELDS: Tuple[str, ...] = tuple(WorldStateCodec.OBS_FIELDS)


def load_feature_contract(config_path: str | None) -> Dict[str, Any]:
    """Load the exact online feature configuration attached to a checkpoint."""
    if config_path is None or str(config_path).strip().lower() in {
        "", "none", "null", "-"
    }:
        return build_fusion_feature_contract()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load --feature-config; "
            "install requirements.txt or pass --feature-config none"
        ) from exc
    source = Path(config_path)
    with source.open("r", encoding="utf-8") as stream:
        project = yaml.safe_load(stream) or {}
    if not isinstance(project, Mapping):
        raise ValueError("feature config root must be a mapping")
    fusion = project.get("fusion", {}) or {}
    internal = project.get("internal", {}) or {}
    if not isinstance(fusion, Mapping) or not isinstance(internal, Mapping):
        raise ValueError("feature config fusion/internal sections must be mappings")
    return build_fusion_feature_contract(fusion, internal)


def _uniform(
    shape: Sequence[int],
    low: float,
    high: float,
    generator: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(tuple(shape), generator=generator)

def _integrate_nonnegative_speed(
    speed: torch.Tensor,
    acceleration: torch.Tensor,
    duration: float | torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate constant acceleration without allowing reverse motion."""
    duration_tensor = torch.as_tensor(
        duration, dtype=speed.dtype, device=speed.device
    ).expand_as(speed)
    time_to_stop = speed / (-acceleration).clamp_min(1e-6)
    motion_time = torch.where(
        acceleration < 0.0,
        torch.minimum(duration_tensor, time_to_stop),
        duration_tensor,
    )
    displacement = (
        speed * motion_time + 0.5 * acceleration * motion_time.square()
    ).clamp_min(0.0)
    next_speed = (speed + acceleration * duration_tensor).clamp(0.0, 36.0)
    return next_speed, displacement


def _external_risk(
    distance_m: torch.Tensor,
    closing_speed_mps: torch.Tensor,
) -> torch.Tensor:
    """Smooth physical hazard target from gap, TTC and stopping margin."""
    safe_closing = closing_speed_mps.clamp_min(0.10)
    ttc_s = torch.where(
        closing_speed_mps > 0.10,
        distance_m / safe_closing,
        torch.full_like(distance_m, 30.0),
    )
    distance_risk = torch.sigmoid((12.0 - distance_m) / 3.5)
    ttc_risk = torch.sigmoid((3.0 - ttc_s) / 0.65)
    # This term becomes large when the remaining gap is below an approximate
    # comfortable stopping distance for the current closing rate.
    stopping_margin = distance_m - closing_speed_mps.square() / (2.0 * 3.5)
    stopping_risk = torch.sigmoid((2.0 - stopping_margin) / 2.5)
    external = torch.maximum(torch.maximum(distance_risk, ttc_risk), stopping_risk)
    return external.clamp(0.0, 1.0)


def _encode_observation(
    codec: WorldStateCodec,
    has_lead_vehicle: torch.Tensor,
    distance_m: torch.Tensor,
    closing_speed_mps: torch.Tensor,
    lane_relevance: torch.Tensor,
    fatigue_score: torch.Tensor,
    attention_score: torch.Tensor,
    previous_fused_score: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build features with the same default scoring equations as online fusion."""
    has_lead_vehicle = has_lead_vehicle.bool()
    lead_distance = torch.where(
        has_lead_vehicle,
        distance_m.clamp(0.0, 100.0),
        torch.full_like(distance_m, 99.0),
    )
    lead_closing = torch.where(
        has_lead_vehicle,
        closing_speed_mps.clamp(0.0, 30.0),
        torch.zeros_like(closing_speed_mps),
    )
    lead_ttc = torch.where(
        has_lead_vehicle & (lead_closing > 0.10),
        lead_distance / lead_closing.clamp_min(0.10),
        torch.full_like(lead_distance, 99.0),
    ).clamp(0.0, 99.0)

    danger = has_lead_vehicle & (lead_ttc < 1.5) & (lead_distance < 45.0)
    caution = has_lead_vehicle & ~danger & (
        ((lead_ttc < 6.0) & (lead_distance < 45.0))
        | (lead_distance < 2.0 * lead_closing)
    )
    warning_level = caution.long() + 2 * danger.long()

    ttc_score = torch.where(
        lead_ttc < 6.0,
        (1.0 - (lead_ttc - 1.5) / (6.0 - 1.5)).clamp(0.0, 1.0),
        torch.zeros_like(lead_ttc),
    )
    distance_score = torch.where(
        (lead_distance > 0.0) & (lead_distance < 30.0),
        (1.0 - (lead_distance - 3.0) / (30.0 - 3.0)).clamp(0.0, 1.0),
        torch.zeros_like(lead_distance),
    )
    geometric_score = torch.maximum(ttc_score, distance_score) * lane_relevance
    warning_floor = torch.where(
        warning_level >= 2,
        torch.full_like(geometric_score, 0.70),
        torch.where(
            warning_level == 1,
            torch.full_like(geometric_score, 0.30),
            torch.zeros_like(geometric_score),
        ),
    )
    ext_score = torch.where(
        has_lead_vehicle,
        torch.maximum(geometric_score, warning_floor),
        torch.zeros_like(geometric_score),
    ).clamp(0.0, 1.0)
    int_score = (0.55 * fatigue_score + 0.45 * attention_score).clamp(0.0, 1.0)
    driver_cross = torch.where(
        (fatigue_score > 0.30) & (attention_score > 0.30),
        0.20 * fatigue_score * attention_score,
        torch.zeros_like(int_score),
    )
    int_score = (int_score + driver_cross).clamp(0.0, 1.0)
    cross_score = (ext_score * int_score).clamp(0.0, 1.0)
    fused_raw = (
        0.35 * ext_score + 0.35 * int_score + 0.30 * cross_score
    ).clamp(0.0, 1.0)
    fused_score = (
        fused_raw
        if previous_fused_score is None
        else 0.40 * fused_raw + 0.60 * previous_fused_score
    ).clamp(0.0, 1.0)
    fused_level = (
        (fused_score >= 0.25).long()
        + (fused_score >= 0.50).long()
        + (fused_score >= 0.75).long()
    )

    columns = zip(
        has_lead_vehicle.tolist(),
        lead_distance.tolist(),
        lead_ttc.tolist(),
        lead_closing.tolist(),
        lane_relevance.tolist(),
        warning_level.tolist(),
        ext_score.tolist(),
        int_score.tolist(),
        cross_score.tolist(),
        fused_score.tolist(),
        fused_level.tolist(),
        fatigue_score.tolist(),
        attention_score.tolist(),
    )
    states = [
        SimpleNamespace(
            has_lead_vehicle=bool(has_lead),
            lead_distance=distance,
            lead_ttc=ttc,
            closing_speed=closing,
            lane_relevance=lane,
            lead_warning_level=int(warning),
            ext_score=ext,
            int_score=internal,
            cross_score=cross,
            fused_score=fused,
            fused_level=int(level),
            fatigue_score=fatigue,
            attention_score=attention,
        )
        for (
            has_lead,
            distance,
            ttc,
            closing,
            lane,
            warning,
            ext,
            internal,
            cross,
            fused,
            level,
            fatigue,
            attention,
        ) in columns
    ]
    encoded = codec.encode_batch(
        states,
        device=torch.device("cpu"),
    )
    encoded = torch.as_tensor(encoded, dtype=torch.float32).reshape(len(states), -1)
    expected_shape = (distance_m.shape[0], len(OBS_FIELDS))
    if tuple(encoded.shape) != expected_shape:
        raise RuntimeError(
            f"WorldStateCodec returned shape {tuple(encoded.shape)}, expected {expected_shape}"
        )
    return encoded, codec.risk_target(encoded), fused_score


def synthetic_batch(
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device,
    world_codec: WorldStateCodec,
    action_codec: ActionCodec,
    dt_s: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Generate transition-aligned states and actually executed actions.

    Observations contain T + 1 states while actions contain T transitions.
    Sampling stays on CPU for seed stability and tensors move to the training
    device only after codec validation.
    """
    bsz = int(batch_size)
    horizon = int(sequence_length)
    if bsz <= 0 or horizon <= 0:
        raise ValueError("batch_size and sequence_length must be positive")

    # Broad initial conditions cover ordinary following, receding traffic and
    # already-dangerous closing scenarios. Some trajectories have no lead
    # vehicle so the codec's 99-value sentinel path is also represented.
    ego_speed = _uniform((bsz,), 6.0, 33.0, generator)
    initial_relative = _uniform((bsz,), -4.0, 15.0, generator)
    lead_speed = (ego_speed - initial_relative).clamp(0.0, 35.0)
    distance = _uniform((bsz,), 5.0, 95.0, generator)
    has_lead = torch.rand((bsz,), generator=generator) < 0.88
    lane_relevance = _uniform((bsz,), 0.35, 1.0, generator)

    fatigue_baseline = _uniform((bsz,), 0.01, 0.78, generator)
    attention_baseline = _uniform((bsz,), 0.01, 0.78, generator)
    fatigue_score = (
        fatigue_baseline + 0.05 * torch.randn((bsz,), generator=generator)
    ).clamp(0.0, 1.0)
    attention_score = (
        attention_baseline + 0.08 * torch.randn((bsz,), generator=generator)
    ).clamp(0.0, 1.0)

    observations = []
    risks = []
    action_ids = []
    action_vectors = []
    continues = [torch.ones((bsz,), dtype=torch.float32)]
    valid_masks = [torch.ones((bsz,), dtype=torch.float32)]
    collided = torch.zeros((bsz,), dtype=torch.bool)

    closing_speed = (ego_speed - lead_speed).clamp_min(0.0)
    obs, risk, fused_ema = _encode_observation(
        world_codec,
        has_lead,
        distance,
        closing_speed,
        lane_relevance,
        fatigue_score,
        attention_score,
    )
    observations.append(obs)
    risks.append(risk)

    decel_table = torch.tensor(ACTION_DECEL_MPS2, dtype=torch.float32)
    delay_table = torch.tensor(ACTION_RESPONSE_DELAY_S, dtype=torch.float32)
    encoded_actions = []
    for name, decel, delay in zip(
        ACTION_NAMES,
        ACTION_DECEL_MPS2,
        ACTION_RESPONSE_DELAY_S,
    ):
        encoded = action_codec.encode(
            SimpleNamespace(
                name=name,
                target_decel=decel,
                response_delay_sec=delay,
            ),
            valid=True,
        )
        encoded_actions.append(torch.as_tensor(encoded, dtype=torch.float32).reshape(-1))
    action_table = torch.stack(encoded_actions)
    if tuple(action_table.shape) != (len(ACTION_NAMES), 3):
        raise RuntimeError(
            f"ActionCodec returned shape {tuple(action_table.shape)}, expected (4, 3)"
        )

    previous_action = torch.arange(bsz, dtype=torch.long) % len(ACTION_NAMES)
    unknown_action_episode = (
        torch.rand((bsz,), generator=generator) < UNKNOWN_ACTION_EPISODE_PROBABILITY
    )
    for step in range(horizon):
        active_transition = ~collided
        # Action persistence makes realistic brake pulses. The first step is
        # stratified across categories. These are simulator-executed commands,
        # never recommendations copied from a decision log.
        proposed = torch.randint(len(ACTION_NAMES), (bsz,), generator=generator)
        hold = torch.rand((bsz,), generator=generator) < 0.72
        executed_action = torch.where(hold, previous_action, proposed)
        if step == 0:
            executed_action = torch.arange(bsz, dtype=torch.long) % len(ACTION_NAMES)
            action_changed = torch.ones((bsz,), dtype=torch.bool)
        else:
            action_changed = executed_action != previous_action

        command_decel = decel_table[executed_action]
        response_delay = delay_table[executed_action]
        active_brake_time = torch.where(
            action_changed,
            (dt_s - response_delay).clamp(0.0, dt_s),
            torch.full_like(response_delay, dt_s),
        )
        previous_action = executed_action

        road_grip = _uniform((bsz,), 0.78, 1.05, generator)
        brake_decel = command_decel * road_grip

        # Integrate the onset wait and braking phase separately. This preserves
        # both terminal speed and the correct 0.5*a*(dt-delay)^2 displacement.
        keep_accel = _uniform((bsz,), -0.35, 0.65, generator)
        next_keep_speed, keep_displacement = _integrate_nonnegative_speed(
            ego_speed, keep_accel, dt_s
        )
        wait_time = dt_s - active_brake_time
        next_brake_speed, brake_displacement = _integrate_nonnegative_speed(
            ego_speed, -brake_decel, active_brake_time
        )
        brake_displacement = brake_displacement + ego_speed * wait_time
        use_keep = executed_action == 0
        next_ego_speed = torch.where(use_keep, next_keep_speed, next_brake_speed)
        ego_displacement = torch.where(
            use_keep, keep_displacement, brake_displacement
        )

        # The lead vehicle evolves independently and can perform short braking
        # events. Signed displacement lets the gap grow when the lead pulls away.
        lead_accel = 0.20 * torch.randn((bsz,), generator=generator)
        lead_brake_event = torch.rand((bsz,), generator=generator) < 0.035
        lead_brake = _uniform((bsz,), 1.0, 4.5, generator)
        lead_accel = torch.where(
            lead_brake_event,
            -lead_brake,
            lead_accel,
        ).clamp(-5.0, 1.5)
        next_lead_speed, lead_displacement = _integrate_nonnegative_speed(
            lead_speed, lead_accel, dt_s
        )

        next_closing = (next_ego_speed - next_lead_speed).clamp_min(0.0)
        raw_distance = distance + lead_displacement - ego_displacement
        collided = collided | (has_lead & (raw_distance <= 0.0))
        distance = torch.where(
            collided, torch.zeros_like(raw_distance), raw_distance.clamp(0.0, 100.0)
        )
        ego_speed, lead_speed = next_ego_speed, next_lead_speed

        # Fatigue and attention are exogenous, persistent driver processes.
        # Acute traffic hazard may increase workload, but the chosen action is
        # never used as a driver-risk label.
        hazard = _external_risk(distance, next_closing) * has_lead.float()
        fatigue_event = torch.rand((bsz,), generator=generator) < 0.010
        fatigue_score = (
            0.985 * fatigue_score
            + 0.015 * fatigue_baseline
            + 0.010 * torch.randn((bsz,), generator=generator)
            + fatigue_event * _uniform((bsz,), 0.03, 0.12, generator)
        ).clamp(0.0, 1.0)
        attention_lapse = torch.rand((bsz,), generator=generator) < 0.020
        attention_score = (
            0.90 * attention_score
            + 0.07 * attention_baseline
            + 0.03 * hazard
            + 0.025 * torch.randn((bsz,), generator=generator)
            + attention_lapse * _uniform((bsz,), 0.08, 0.28, generator)
        ).clamp(0.0, 1.0)
        lane_relevance = (
            lane_relevance + 0.015 * torch.randn((bsz,), generator=generator)
        ).clamp(0.25, 1.0)

        obs, risk, fused_ema = _encode_observation(
            world_codec,
            has_lead,
            distance,
            next_closing,
            lane_relevance,
            fatigue_score,
            attention_score,
            previous_fused_score=fused_ema,
        )
        observations.append(obs)
        risks.append(risk)
        continues.append((~collided).float())
        valid_masks.append(active_transition.float())
        action_ids.append(executed_action)
        transition_action = action_table[executed_action].clone()
        transition_action[:, 1] = torch.where(
            action_changed,
            transition_action[:, 1],
            torch.zeros_like(transition_action[:, 1]),
        )
        unknown_action = unknown_action_episode | (
            torch.rand((bsz,), generator=generator) < UNKNOWN_ACTION_STEP_PROBABILITY
        )
        transition_action[unknown_action] = 0.0

        action_vectors.append(transition_action)

    result = {
        "observations": torch.stack(observations, dim=1).to(device),
        "executed_actions": torch.stack(action_vectors, dim=1).to(device),
        "executed_action_ids": torch.stack(action_ids, dim=1).to(device),
        "continues": torch.stack(continues, dim=1).to(device),
        "valid_mask": torch.stack(valid_masks, dim=1).to(device),
        "risk_targets": torch.stack(risks, dim=1).to(device),
    }
    for name, tensor in result.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"synthetic generator produced non-finite {name}")
    return result


def build_model() -> Tuple[TinyRSSM, Dict[str, Any]]:
    """Construct the small categorical RSSM used by online inference."""
    requested = {
        "obs_dim": len(OBS_FIELDS),
        "action_dim": 3,
        "embed_dim": 64,
        "deter_dim": 64,
        "stoch_dim": 8,
        "classes": 8,
        "hidden_dim": 64,
        "max_decel": 8.0,
        "max_delay": 0.5,
    }
    model = TinyRSSM(requested)
    return model, model.get_config()


def _loss_mapping(result: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    if torch.is_tensor(result):
        return result, {"loss": float(result.detach())}
    if not isinstance(result, Mapping):
        raise TypeError("TinyRSSM loss method must return a tensor or a mapping")
    loss = result.get("loss", result.get("total_loss"))
    if not torch.is_tensor(loss):
        raise TypeError("TinyRSSM loss mapping must contain tensor key 'loss' or 'total_loss'")
    values: Dict[str, float] = {}
    for key, value in result.items():
        if torch.is_tensor(value) and value.numel() == 1:
            values[str(key)] = float(value.detach())
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            values[str(key)] = float(value)
    values.setdefault("loss", float(loss.detach()))
    return loss, values


def training_loss(
    model: TinyRSSM,
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Align each observation with the action that led into that state."""
    observations = batch["observations"]
    executed_actions = batch["executed_actions"]
    initial_unknown_action = torch.zeros(
        observations.shape[0],
        1,
        model.config.action_dim,
        dtype=observations.dtype,
        device=observations.device,
    )
    aligned_actions = torch.cat((initial_unknown_action, executed_actions), dim=1)
    if aligned_actions.shape[:2] != observations.shape[:2]:
        raise RuntimeError("synthetic observation/action timelines are not aligned")
    result = model.loss_sequence(
        observations=observations,
        actions=aligned_actions,
        risk_targets=batch["risk_targets"],
        continues=batch["continues"],
        valid_mask=batch["valid_mask"],
    )
    return _loss_mapping(result)


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def save_checkpoint(
    output: Path,
    model: TinyRSSM,
    metrics: Mapping[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint_payload(metrics)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    if not math.isfinite(args.lr) or not 0.0 < args.lr <= 0.1:
        raise ValueError("learning rate must be finite and in (0, 0.1]")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0.0:
        raise ValueError("gradient clip must be finite and positive")
    feature_contract = load_feature_contract(
        getattr(args, "feature_config", "config.yaml")
    )
    device = resolve_device(args.device)
    generator = seed_everything(args.seed)
    model, _ = build_model()
    model.to(device)
    model.train()

    world_codec = WorldStateCodec()
    action_codec = ActionCodec(
        max_decel=model.config.max_decel,
        max_delay=model.config.max_delay,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    started = time.perf_counter()
    recent_losses = []
    best_loss = float("inf")
    last_parts: Dict[str, float] = {}
    action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
    known_action_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.long)
    valid_action_count = 0
    transition_count = 0

    for step in range(1, args.steps + 1):
        batch = synthetic_batch(
            args.batch,
            args.seq,
            generator,
            device,
            world_codec,
            action_codec,
            dt_s=model.config.dt_sec,
        )
        transition_valid = batch["valid_mask"][:, 1:].bool()
        action_ids = batch["executed_action_ids"]
        known_action = batch["executed_actions"][..., -1] > 0.5
        action_counts += torch.bincount(
            action_ids[transition_valid].detach().cpu(),
            minlength=len(ACTION_NAMES),
        )
        known_action_counts += torch.bincount(
            action_ids[transition_valid & known_action].detach().cpu(),
            minlength=len(ACTION_NAMES),
        )
        valid_action_count += int((transition_valid & known_action).sum().item())
        transition_count += int(transition_valid.sum().item())
        optimizer.zero_grad(set_to_none=True)
        loss, parts = training_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        if not all(
            torch.isfinite(parameter).all().item()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"optimizer produced non-finite weights at step {step}")
        optimizer_tensors = (
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        if not all(torch.isfinite(value).all().item() for value in optimizer_tensors):
            raise FloatingPointError(f"optimizer produced non-finite state at step {step}")

        value = float(loss.detach())
        recent_losses.append(value)
        recent_losses = recent_losses[-100:]
        best_loss = min(best_loss, value)
        last_parts = parts
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(
                f"step={step:5d}/{args.steps} loss={value:.5f} "
                f"mean100={sum(recent_losses) / len(recent_losses):.5f} "
                f"grad={float(grad_norm):.3f}"
            )

    elapsed = time.perf_counter() - started
    unknown_fraction = 1.0 - valid_action_count / max(1, transition_count)
    if not 0.05 <= unknown_fraction <= 1.0:
        raise RuntimeError(
            "training did not achieve the required unknown-action coverage"
        )
    missing_known_actions = [
        name
        for name, count in zip(ACTION_NAMES, known_action_counts.tolist())
        if count <= 0
    ]
    if missing_known_actions:
        raise RuntimeError(
            "training lacks known executed-action coverage for: "
            + ", ".join(missing_known_actions)
        )

    metrics: Dict[str, Any] = {
        "steps": args.steps,
        "batch_size": args.batch,
        "sequence_length": args.seq,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": round(elapsed, 3),
        "final_loss": recent_losses[-1],
        "mean_loss_last_100": sum(recent_losses) / len(recent_losses),
        "best_loss": best_loss,
        "synthetic_pretraining": True,
        "action_source": "simulator_confirmed_command_setpoint",
        "action_contract": {
            "version": ACTION_CONTRACT_VERSION,
            "alignment": "previous_observation_applied_action_current_observation",
            "unknown_action_supported": True,
        },
        "unknown_action_training": {
            "episode_probability": UNKNOWN_ACTION_EPISODE_PROBABILITY,
            "step_probability": UNKNOWN_ACTION_STEP_PROBABILITY,
            "unknown_fraction": unknown_fraction,
        },
        "fusion_feature_contract": feature_contract,
        "action_catalog": [
            {
                "name": name,
                "target_decel": decel,
                "response_delay_sec": delay,
            }
            for name, decel, delay in zip(
                ACTION_NAMES,
                ACTION_DECEL_MPS2,
                ACTION_RESPONSE_DELAY_S,
            )
        ],
        "executed_action_counts": {
            name: int(count)
            for name, count in zip(ACTION_NAMES, action_counts.tolist())
        },
        "known_executed_action_counts": {
            name: int(count)
            for name, count in zip(ACTION_NAMES, known_action_counts.tolist())
        },
        "last_loss_components": last_parts,
    }
    save_checkpoint(Path(args.output), model, metrics)
    digest = checkpoint_sha256(args.output)
    print(f"saved checkpoint: {Path(args.output).resolve()}")
    print(f"checkpoint sha256: {digest}")
    return metrics


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-train TinyRSSM with physically simulated executed braking actions."
    )
    parser.add_argument("--steps", type=positive_int, default=2000, help="optimizer updates")
    parser.add_argument("--batch", type=positive_int, default=32, help="trajectories per update")
    parser.add_argument("--seq", type=positive_int, default=32, help="transitions per trajectory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--output", default="checkpoints/idms_rssm.pt", help="checkpoint path")
    parser.add_argument(
        "--feature-config",
        default="config.yaml",
        help="project YAML defining online fusion/internal feature semantics; use 'none' for defaults",
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--grad-clip", type=float, default=100.0, help="global gradient-norm cap")
    parser.add_argument("--log-every", type=positive_int, default=100, help="progress interval")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not math.isfinite(args.lr) or not 0.0 < args.lr <= 0.1:
        parser.error("--lr must be finite and in (0, 0.1]")
    if not math.isfinite(args.grad_clip) or args.grad_clip <= 0.0:
        parser.error("--grad-clip must be finite and positive")
    try:
        train(args)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
