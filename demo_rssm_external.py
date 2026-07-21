#!/usr/bin/env python3
"""External-only RSSM inference-chain validation demo.

The cabin perception channel is deliberately unavailable.  Its legacy RSSM
features remain zero-valued to preserve the trained schema, but the HUD labels
them N/A rather than claiming that the driver is safe.  This demo is open-loop:
planner recommendations are never fed back as executed actions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2

from src.core.mrm_planner import DecisionLogger, MRMPlanner
from src.core.risk_fusion import FusionResult, RiskFusionEngine
from src.external.collision_warn import CollisionWarner
from src.external.distance_est import DistanceEstimator
from src.external.yolo_detector import YoloDetector
from src.integration.external_source import FramePacket, OpenCVFrameSource
from src.ui.alert_system import AudioAlerter
from src.ui.visualizer import Visualizer


CABIN_STATUS = "UNAVAILABLE (external-only demo)"
DEFAULT_MIN_POSTERIOR_UPDATES = 4
DEFAULT_MIN_SOURCE_SPAN_SEC = 0.75


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the DreamerV3-style RSSM inference chain with one front camera "
            "while the cabin perception channel is explicitly unavailable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("camera", "video"), default="camera")
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index or video path. Camera defaults to system.camera_id_ext.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="Override YOLO device.")
    parser.add_argument("--rssm-device", default=None, help="Override RSSM device.")
    parser.add_argument("--model", default=None, help="Override YOLO model path.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument("--roi-top", type=float, default=None)
    parser.add_argument("--output", default=None, help="Optional annotated MP4 path.")
    parser.add_argument(
        "--summary-json",
        default="logs/rssm_external_summary.json",
        help="Validation summary path; use 'none' to disable.",
    )
    parser.add_argument(
        "--log-path",
        default="logs/rssm_external_decisions.csv",
        help="Per-decision CSV path.",
    )
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-pacing", action="store_true", help="Do not pace video to its FPS.")
    parser.add_argument("--loop", action="store_true", help="Loop a video source.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-posterior-updates",
        type=int,
        default=DEFAULT_MIN_POSTERIOR_UPDATES,
        help="Minimum contiguous RSSM posterior updates for strict validation.",
    )
    parser.add_argument(
        "--min-source-span-sec",
        type=float,
        default=DEFAULT_MIN_SOURCE_SPAN_SEC,
        help="Minimum fresh source-time span for strict validation.",
    )
    parser.add_argument(
        "--max-source-loss-sec",
        type=float,
        default=3.0,
        help="Stop after this much continuous camera read failure.",
    )
    parser.add_argument(
        "--allow-kinematic-fallback",
        action="store_true",
        help="Allow a run to succeed without RSSM forecasts.",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.min_posterior_updates < 2:
        parser.error("--min-posterior-updates must be at least 2")
    if not math.isfinite(args.min_source_span_sec) or args.min_source_span_sec <= 0.0:
        parser.error("--min-source-span-sec must be positive and finite")
    if not math.isfinite(args.max_source_loss_sec) or args.max_source_loss_sec <= 0.0:
        parser.error("--max-source-loss-sec must be positive and finite")
    if args.mode == "video" and not args.source:
        parser.error("--source is required in video mode")
    return args


def load_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is missing; run: python -m pip install -r requirements.txt"
        ) from exc
    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return loaded


def build_configs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    object,
]:
    config = load_config(args.config)
    system_cfg = dict(config.get("system", {}))
    external_cfg = dict(config.get("external", {}))
    ui_cfg = dict(config.get("ui", {}))
    fusion_cfg = dict(config.get("internal", {}))
    fusion_cfg.update(config.get("fusion", {}))
    decision_cfg = dict(config.get("decision", {}))
    # Preserve the exact feature contract used to train the checkpoint. Cabin
    # hardware is not constructed even though its configuration is recorded.
    decision_cfg["_runtime_fusion_config"] = dict(config.get("fusion", {}))
    decision_cfg["_runtime_internal_config"] = dict(config.get("internal", {}))

    overrides = {
        "device": args.device,
        "model_path": args.model,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "focal_length": args.focal,
        "roi_top_ratio": args.roi_top,
    }
    for name, value in overrides.items():
        if value is not None:
            external_cfg[name] = value
    if args.rssm_device is not None:
        rssm_cfg = dict(decision_cfg.get("rssm", {}))
        rssm_cfg["device"] = args.rssm_device
        decision_cfg["rssm"] = rssm_cfg
    if args.no_audio:
        alert_cfg = dict(ui_cfg.get("alert", {}))
        alert_cfg["enable"] = False
        ui_cfg["alert"] = alert_cfg
    decision_cfg["log_enable"] = True
    decision_cfg["log_path"] = args.log_path
    source = args.source
    if source is None:
        source = system_cfg.get("camera_id_ext", 0)
    return system_cfg, external_cfg, ui_cfg, fusion_cfg, decision_cfg, source


@dataclass
class ValidationStats:
    packets: int = 0
    valid_frames: int = 0
    sensor_loss_events: int = 0
    rssm_frames: int = 0
    fallback_valid_frames: int = 0
    max_contiguous_posterior_updates: int = 0
    max_source_time_span_sec: float = 0.0
    uncertainty_sum: float = 0.0
    max_uncertainty: float = 0.0
    actions: Counter[str] = field(default_factory=Counter)
    _sequence_start_timestamp: Optional[float] = field(default=None, repr=False)
    _last_sequence_timestamp: Optional[float] = field(default=None, repr=False)

    def record(
        self,
        decision: Any,
        *,
        valid: bool,
        rssm_observe_count: int = 0,
        sequence_timestamp: Optional[float] = None,
        reset: bool = False,
    ) -> None:
        self.packets += 1
        if valid:
            self.valid_frames += 1
            self.max_contiguous_posterior_updates = max(
                self.max_contiguous_posterior_updates,
                max(0, int(rssm_observe_count)),
            )
            if sequence_timestamp is not None:
                source_time = float(sequence_timestamp)
                if not math.isfinite(source_time) or source_time < 0.0:
                    raise ValueError("sequence_timestamp must be finite and non-negative")
                if (
                    reset
                    or self._sequence_start_timestamp is None
                    or (
                        self._last_sequence_timestamp is not None
                        and source_time < self._last_sequence_timestamp
                    )
                ):
                    self._sequence_start_timestamp = source_time
                self._last_sequence_timestamp = source_time
                self.max_source_time_span_sec = max(
                    self.max_source_time_span_sec,
                    source_time - self._sequence_start_timestamp,
                )
            model_used = getattr(decision, "prediction_source", "") == "rssm_hybrid"
            if model_used:
                self.rssm_frames += 1
            else:
                self.fallback_valid_frames += 1
            uncertainty = float(getattr(decision, "model_uncertainty", 0.0))
            if model_used and math.isfinite(uncertainty):
                self.uncertainty_sum += uncertainty
                self.max_uncertainty = max(self.max_uncertainty, uncertainty)
        else:
            self.sensor_loss_events += 1
        self.actions[str(getattr(decision, "action", "UNKNOWN"))] += 1

    def as_dict(self, *, planner: MRMPlanner, source_description: str) -> dict[str, Any]:
        average_uncertainty = (
            self.uncertainty_sum / self.rssm_frames if self.rssm_frames else 0.0
        )
        observe_count = (
            int(planner.rssm.observe_count) if planner.rssm is not None else 0
        )
        return {
            "source": source_description,
            "cabin_perception": "unavailable",
            "cabin_feature_encoding": "zero_placeholder_schema_v3",
            "control_mode": "open_loop_advisory",
            "planner_status": planner.model_status,
            "packets": self.packets,
            "valid_frames": self.valid_frames,
            "sensor_loss_events": self.sensor_loss_events,
            "rssm_frames": self.rssm_frames,
            "fallback_valid_frames": self.fallback_valid_frames,
            "rssm_observe_count": observe_count,
            "max_contiguous_posterior_updates": self.max_contiguous_posterior_updates,
            "max_source_time_span_sec": self.max_source_time_span_sec,
            "rssm_utilization": (
                self.rssm_frames / self.valid_frames if self.valid_frames else 0.0
            ),
            "mean_model_uncertainty": average_uncertainty,
            "max_model_uncertainty": self.max_uncertainty,
            "actions": dict(self.actions),
        }


def strict_validation_errors(
    stats: ValidationStats,
    *,
    min_posterior_updates: int = DEFAULT_MIN_POSTERIOR_UPDATES,
    min_source_span_sec: float = DEFAULT_MIN_SOURCE_SPAN_SEC,
) -> list[str]:
    """Return deterministic inference-chain failures for strict mode."""
    if min_posterior_updates < 2:
        raise ValueError("min_posterior_updates must be at least 2")
    if not math.isfinite(min_source_span_sec) or min_source_span_sec <= 0.0:
        raise ValueError("min_source_span_sec must be positive and finite")
    errors: list[str] = []
    if stats.valid_frames <= 0:
        errors.append("no fresh valid front-camera frames were processed")
    if stats.rssm_frames <= 0:
        errors.append("no valid frame used an RSSM hybrid forecast")
    if stats.fallback_valid_frames > 0:
        errors.append(
            f"{stats.fallback_valid_frames} valid frame(s) used a non-RSSM forecast"
        )
    if stats.max_contiguous_posterior_updates < min_posterior_updates:
        errors.append(
            "maximum contiguous posterior updates "
            f"{stats.max_contiguous_posterior_updates} < {min_posterior_updates}"
        )
    if stats.max_source_time_span_sec + 1e-9 < min_source_span_sec:
        errors.append(
            f"maximum fresh source-time span {stats.max_source_time_span_sec:.3f}s "
            f"< {min_source_span_sec:.3f}s"
        )
    return errors


def build_validation_outcome(
    stats: ValidationStats,
    *,
    allow_kinematic_fallback: bool,
    min_posterior_updates: int,
    min_source_span_sec: float,
    runtime_exit_code: int,
) -> tuple[dict[str, Any], list[str]]:
    """Separate diagnostic completion from strict RSSM-chain validation."""
    if allow_kinematic_fallback:
        completion_errors = (
            ["no fresh valid front-camera frames were processed"]
            if stats.valid_frames == 0
            else []
        )
    else:
        completion_errors = strict_validation_errors(
            stats,
            min_posterior_updates=min_posterior_updates,
            min_source_span_sec=min_source_span_sec,
        )
    if runtime_exit_code != 0:
        completion_errors.insert(
            0, f"runtime ended with exit code {runtime_exit_code}"
        )

    strict_mode = not allow_kinematic_fallback
    run_completed = runtime_exit_code == 0 and stats.valid_frames > 0
    rssm_validation_passed = strict_mode and not completion_errors
    diagnostic_passed: Optional[bool] = None
    if allow_kinematic_fallback:
        diagnostic_passed = not completion_errors
    outcome = {
        "validation_mode": (
            "fallback_diagnostic"
            if allow_kinematic_fallback
            else "strict_rssm_chain"
        ),
        "run_completed": run_completed,
        "diagnostic_passed": diagnostic_passed,
        "rssm_validation_passed": rssm_validation_passed,
        # Kept for existing readers, but only strict RSSM mode can set it true.
        "validation_passed": rssm_validation_passed,
        "strict_requirements": {
            "min_contiguous_posterior_updates": min_posterior_updates,
            "min_source_time_span_sec": min_source_span_sec,
        },
        "validation_errors": list(completion_errors),
    }
    return outcome, completion_errors


class ExternalOnlyPipeline:
    """Single-packet orchestration kept independent of camera/CARLA creation."""

    def __init__(
        self,
        detector: Any,
        distance_estimator: Any,
        collision_warner: Any,
        fusion_engine: Any,
        planner: MRMPlanner,
        decision_logger: Any,
    ) -> None:
        self.detector = detector
        self.distance_estimator = distance_estimator
        self.collision_warner = collision_warner
        self.fusion_engine = fusion_engine
        self.planner = planner
        self.decision_logger = decision_logger
        self.last_fusion: Any = FusionResult()
        self.last_vehicle_data: list[dict[str, Any]] = []

    def reset(self) -> None:
        reset_collision = getattr(self.collision_warner, "reset", None)
        if callable(reset_collision):
            reset_collision()
        else:
            self.collision_warner.last_frame_data = []
            self.collision_warner.last_timestamp = None
        self.fusion_engine.reset()
        self.planner.reset()
        self.last_fusion = FusionResult()
        self.last_vehicle_data = []

    def process(self, packet: FramePacket) -> tuple[Any, list[dict[str, Any]], Any]:
        if packet.eof:
            raise ValueError("EOF is not an observation")
        if packet.reset:
            self.reset()
        if packet.valid:
            raw_detections = self.detector.process(packet.frame)
            dist_detections = self.distance_estimator.calculate(raw_detections)
            vehicle_data = self.collision_warner.process(
                dist_detections,
                frame_width=packet.frame.shape[1],
                timestamp=packet.sequence_timestamp,
            )
            fusion_result = self.fusion_engine.evaluate(
                vehicle_data=vehicle_data,
                face_data=None,
            )
            self.last_vehicle_data = vehicle_data
            self.last_fusion = fusion_result
        else:
            vehicle_data = []
            fusion_result = self.last_fusion

        decision = self.planner.plan(
            fusion_result=fusion_result,
            vehicle_data=vehicle_data,
            face_data=None,
            timestamp=packet.wall_timestamp,
            applied_action=None,
            sequence_timestamp=packet.sequence_timestamp,
            external_perception_valid=packet.valid,
            external_perception_age_sec=packet.perception_age_sec,
        )
        self.decision_logger.log(decision, timestamp=packet.wall_timestamp)
        return fusion_result, vehicle_data, decision


def require_ready_planner(planner: MRMPlanner, allow_fallback: bool) -> None:
    ready = planner.rssm is not None and planner.rssm.ready and planner.model_status == "ready"
    if not ready and not allow_fallback:
        raise RuntimeError(
            "RSSM validation requires a ready checkpoint; planner status is "
            f"{planner.model_status!r}. Use --allow-kinematic-fallback only for diagnostics."
        )


def draw_status(frame: Any, decision: Any, planner: MRMPlanner, packet: FramePacket) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 96), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    source = getattr(decision, "prediction_source", "kinematic")
    uncertainty = float(getattr(decision, "model_uncertainty", 0.0))
    observe_count = planner.rssm.observe_count if planner.rssm is not None else 0
    lines = (
        "EXTERNAL-ONLY RSSM DEMO | CABIN: UNAVAILABLE / INT: N/A",
        f"MODEL: {planner.model_status} | prediction_source: {source} | uncertainty: {uncertainty:.3f}",
        f"OPEN LOOP / ADVISORY ONLY | applied_action: UNKNOWN | posterior updates: {observe_count}",
    )
    colors = ((0, 210, 255), (235, 235, 235), (160, 200, 255))
    for index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            frame,
            line,
            (10, 25 + index * 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1 if index else 2,
            cv2.LINE_AA,
        )
    if not packet.valid:
        cv2.putText(
            frame,
            f"FRONT CAMERA UNAVAILABLE ({packet.perception_age_sec:.2f}s)",
            (10, min(h - 16, 128)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def make_writer(path: Optional[str], fps: float, size: tuple[int, int]) -> Any:
    if not path:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 1.0 else 25.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {destination}")
    return writer


def write_summary(path: Optional[str], summary: dict[str, Any]) -> None:
    if not path or str(path).strip().lower() == "none":
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def run_demo(args: argparse.Namespace) -> int:
    (
        system_cfg,
        external_cfg,
        ui_cfg,
        fusion_cfg,
        decision_cfg,
        source_value,
    ) = build_configs(args)
    planner = MRMPlanner(decision_cfg)
    require_ready_planner(planner, args.allow_kinematic_fallback)
    stats = ValidationStats()
    source = None
    writer = None
    decision_logger = None
    alerter = None
    last_display = None
    source_loss_started: Optional[float] = None
    valid_frames_at_last_loop_reset: Optional[int] = None
    exit_code = 0

    try:
        detector = YoloDetector(external_cfg)
        distance_estimator = DistanceEstimator(external_cfg)
        collision_warner = CollisionWarner(external_cfg)
        fusion_engine = RiskFusionEngine(fusion_cfg)
        source = OpenCVFrameSource(
            args.mode,
            source_value,
            width=int(system_cfg.get("frame_width", 640)),
            height=int(system_cfg.get("frame_height", 480)),
        )
        decision_logger = DecisionLogger(decision_cfg)
        visualizer = Visualizer(ui_cfg)
        alerter = AudioAlerter(ui_cfg)
        pipeline = ExternalOnlyPipeline(
            detector,
            distance_estimator,
            collision_warner,
            fusion_engine,
            planner,
            decision_logger,
        )
        frame_interval = (
            1.0 / source.nominal_fps
            if args.mode == "video" and source.nominal_fps > 1.0 and not args.no_pacing
            else 0.0
        )

        print("=" * 78)
        print("IDMS external-only RSSM inference-chain validation")
        print(f"source          : {source.description}")
        print(f"cabin perception: {CABIN_STATUS}")
        print("control         : OPEN LOOP / ADVISORY ONLY")
        print(f"planner         : {planner.model_status}")
        print("=" * 78)

        while True:
            if frame_interval > 0.0:
                time.sleep(frame_interval)
            packet = source.read()
            if packet.eof:
                if args.loop:
                    if valid_frames_at_last_loop_reset == stats.valid_frames:
                        print("[ERROR] video loop reset produced no fresh frame")
                        exit_code = 2
                        break
                    if source.reset():
                        valid_frames_at_last_loop_reset = stats.valid_frames
                        # The first packet after seek carries reset=True, so the
                        # temporal pipeline is reset exactly once.
                        continue
                    print("[ERROR] video source could not seek to the first frame")
                    exit_code = 2
                break
            if packet.valid:
                source_loss_started = None
            else:
                source_check_time = time.monotonic()
                if source_loss_started is None:
                    source_loss_started = (
                        source_check_time - packet.perception_age_sec
                    )
                if source_check_time - source_loss_started >= args.max_source_loss_sec:
                    print("[ERROR] front-camera source remained unavailable")
                    exit_code = 2
                    break

            fusion_result, vehicle_data, decision = pipeline.process(packet)
            stats.record(
                decision,
                valid=packet.valid,
                rssm_observe_count=(planner.rssm.observe_count if planner.rssm else 0),
                sequence_timestamp=packet.sequence_timestamp,
                reset=packet.reset,
            )
            ext_danger = (
                float(getattr(fusion_result, "ext_score", 0.0)) >= 0.7
                or decision.action in {"BRAKE", "EMERGENCY_BRAKE"}
            )
            alerter.update(ext_danger=ext_danger, int_danger=False)

            if packet.valid:
                display = visualizer.draw_results(
                    packet.frame.copy(),
                    face_data=None,
                    vehicle_data=vehicle_data,
                )
                visualizer.draw_decision_panel(display, decision)
                draw_status(display, decision, planner, packet)
                last_display = display
                if writer is None and args.output:
                    height, width = display.shape[:2]
                    writer = make_writer(args.output, source.nominal_fps, (width, height))
                if writer is not None:
                    writer.write(display)
            elif last_display is not None:
                display = last_display.copy()
                draw_status(display, decision, planner, packet)
                last_display = display

            if stats.valid_frames and stats.valid_frames % 30 == 0 and packet.valid:
                print(
                    f"[RSSM] frames={stats.valid_frames} source={decision.prediction_source} "
                    f"updates={planner.rssm.observe_count if planner.rssm else 0} "
                    f"action={decision.action} risk={decision.predicted_risk:.3f} "
                    f"uncertainty={decision.model_uncertainty:.3f}"
                )

            if not args.no_display and last_display is not None:
                cv2.imshow("IDMS - External-only RSSM Inference Chain", last_display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    pipeline.reset()
                    print("[System] temporal state reset")
            elif not packet.valid:
                time.sleep(0.01)

            if args.max_frames is not None and stats.valid_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n[System] interrupted")
    finally:
        cleanup_actions = (
            ("frame source", source.close if source is not None else None),
            ("video writer", writer.release if writer is not None else None),
            (
                "decision logger",
                decision_logger.close if decision_logger is not None else None,
            ),
            ("audio alerter", alerter.close if alerter is not None else None),
            ("OpenCV windows", cv2.destroyAllWindows),
        )
        for label, cleanup in cleanup_actions:
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as exc:  # best effort: do not skip later resources
                print(f"[WARN] failed to close {label}: {exc}", file=sys.stderr)
                exit_code = 2

    source_description = (
        source.description if source is not None else f"opencv:{args.mode}:{source_value}"
    )
    summary = stats.as_dict(planner=planner, source_description=source_description)
    outcome, completion_errors = build_validation_outcome(
        stats,
        allow_kinematic_fallback=args.allow_kinematic_fallback,
        min_posterior_updates=args.min_posterior_updates,
        min_source_span_sec=args.min_source_span_sec,
        runtime_exit_code=exit_code,
    )
    summary.update(outcome)
    try:
        write_summary(args.summary_json, summary)
    except OSError as exc:
        exit_code = max(exit_code, 2)
        artifact_error = (
            f"failed to write validation summary: {exc}; any existing file may be stale"
        )
        completion_errors.append(artifact_error)
        summary["run_completed"] = False
        if summary["diagnostic_passed"] is not None:
            summary["diagnostic_passed"] = False
        summary["rssm_validation_passed"] = False
        summary["validation_passed"] = False
        summary["validation_errors"] = list(completion_errors)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[ERROR] validation: {artifact_error}", file=sys.stderr)
        return exit_code
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if completion_errors:
        for error in completion_errors:
            print(f"[ERROR] validation: {error}")
        return max(exit_code, 2)
    return exit_code

def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_demo(parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

