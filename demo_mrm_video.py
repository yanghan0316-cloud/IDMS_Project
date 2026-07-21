#!/usr/bin/env python3
#python demo_mrm_video.py --source E:\数据集\数据集\day-clear-1.mp4 --device cpu
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from src.core.mrm_planner import DecisionLogger, MRMPlanner
from src.core.risk_fusion import RiskFusionEngine
from src.external.collision_warn import CollisionWarner
from src.external.distance_est import DistanceEstimator
from src.external.yolo_detector import YoloDetector
from src.ui.alert_system import AudioAlerter
from src.ui.visualizer import Visualizer


RISK_COLORS = {
    0: (0, 200, 0),
    1: (0, 200, 255),
    2: (0, 128, 255),
    3: (0, 0, 255),
}


ACTION_COLORS = {
    "KEEP": (0, 200, 0),
    "SLOW_DOWN": (0, 210, 255),
    "BRAKE": (0, 140, 255),
    "EMERGENCY_BRAKE": (0, 0, 255),
}


class FPSCounter:
    def __init__(self, window: int = 30):
        self.window = int(window)
        self.timestamps: list[float] = []

    def tick(self) -> None:
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / dt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external perception + risk fusion + MRM decision on a front-camera video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Front-camera video path.")
    parser.add_argument("--config", default="config.yaml", help="Config file path.")
    parser.add_argument("--device", default=None, help="Override YOLO device, e.g. cpu or cuda:0.")
    parser.add_argument("--model", default=None, help="Override YOLO model path.")
    parser.add_argument("--conf", type=float, default=None, help="Override YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO inference image size.")
    parser.add_argument("--focal", type=float, default=None, help="Override monocular distance focal length.")
    parser.add_argument("--roi-top", type=float, default=None, help="Override external ROI top ratio.")
    parser.add_argument("--output", default=None, help="Optional path to save annotated video.")
    parser.add_argument("--log-path", default="logs/mrm_video_decisions.csv", help="MRM decision CSV log path.")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio alerts.")
    parser.add_argument("--no-display", action="store_true", help="Do not open an OpenCV display window.")
    parser.add_argument("--no-pacing", action="store_true", help="Process video as fast as possible.")
    parser.add_argument("--loop", action="store_true", help="Loop the video when it reaches the end.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames.")
    parser.add_argument("--save-dir", default="screenshots", help="Directory for screenshots.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] Missing dependency PyYAML. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_configs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = load_config(args.config)
    external_cfg = dict(cfg.get("external", {}))
    ui_cfg = dict(cfg.get("ui", {}))
    decision_cfg = dict(cfg.get("decision", {}))
    decision_cfg["_runtime_fusion_config"] = dict(cfg.get("fusion", {}))
    decision_cfg["_runtime_internal_config"] = dict(cfg.get("internal", {}))

    if args.device is not None:
        external_cfg["device"] = args.device
    if args.model is not None:
        external_cfg["model_path"] = args.model
    if args.conf is not None:
        external_cfg["conf_threshold"] = args.conf
    if args.imgsz is not None:
        external_cfg["imgsz"] = args.imgsz
    if args.focal is not None:
        external_cfg["focal_length"] = args.focal
    if args.roi_top is not None:
        external_cfg["roi_top_ratio"] = args.roi_top

    if args.no_audio:
        alert_cfg = dict(ui_cfg.get("alert", {}))
        alert_cfg["enable"] = False
        ui_cfg["alert"] = alert_cfg

    decision_cfg["log_enable"] = True
    decision_cfg["log_path"] = args.log_path

    fusion_cfg = dict(cfg.get("internal", {}))
    fusion_cfg.update(cfg.get("fusion", {}))
    return external_cfg, ui_cfg, fusion_cfg, decision_cfg


def open_video(path: str) -> cv2.VideoCapture:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"[ERROR] Video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {source}")
    return cap


def make_writer(path: Optional[str], fps: float, frame_size: tuple[int, int]) -> Optional[cv2.VideoWriter]:
    if not path:
        return None

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps > 1.0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, safe_fps, frame_size)
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Cannot create output video: {out_path}")
    return writer


def draw_risk_bar(frame, fusion_result) -> None:
    h, w = frame.shape[:2]
    color = RISK_COLORS.get(int(fusion_result.fused_level), (0, 200, 0))
    bar_w = min(220, max(150, w // 4))
    bar_h = 16
    x = w - bar_w - 18
    y = 12

    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (55, 55, 55), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, float(fusion_result.fused_score))))
    cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
    cv2.putText(
        frame,
        f"RISK: {fusion_result.fused_text} ({fusion_result.fused_score:.2f})",
        (x, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Ext:{fusion_result.ext_score:.2f} Int:{fusion_result.int_score:.2f} Cross:{fusion_result.cross_score:.2f}",
        (x, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )


def draw_header(frame, fps: float, frame_idx: int, source: str, paused: bool) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    state = "PAUSED" if paused else "RUNNING"
    src_name = Path(source).name
    cv2.putText(
        frame,
        f"IDMS MRM Video Demo | {src_name} | {state}",
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS:{fps:.1f}  Frame:{frame_idx}  Q/ESC:quit  P:pause  S:screenshot  R:reset  +/-:conf",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (205, 220, 235),
        1,
        cv2.LINE_AA,
    )


def draw_candidate_costs(frame, decision) -> None:
    candidates = list(getattr(decision, "candidates", []) or [])
    if not candidates:
        return

    h, w = frame.shape[:2]
    x = 12
    y = 92
    row_h = 22
    panel_w = min(300, max(240, w // 3))
    panel_h = 32 + row_h * len(candidates)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (90, 90, 90), 1)
    cv2.putText(
        frame,
        "Candidate action costs",
        (x + 10, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    for i, item in enumerate(sorted(candidates, key=lambda c: c.action.rank)):
        yy = y + 48 + i * row_h
        name = item.action.name
        color = ACTION_COLORS.get(name, (220, 220, 220))
        mark = "*" if name == decision.action else " "
        text = (
            f"{mark} {name:<16} cost {item.cost:.2f} "
            f"risk {item.prediction.peak_risk:.2f}"
        )
        cv2.putText(frame, text, (x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def save_screenshot(frame, save_dir: str) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mrm_video_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    print(f"[Screenshot] saved: {path}")


def reset_runtime(collision_warner, fusion_engine, mrm_planner, decision_logger) -> None:
    collision_warner.last_frame_data = []
    collision_warner.last_timestamp = time.time()
    fusion_engine.reset()
    mrm_planner.reset()
    if decision_logger is not None:
        decision_logger._last_log_time = 0.0


def main() -> int:
    args = parse_args()
    external_cfg, ui_cfg, fusion_cfg, decision_cfg = build_configs(args)

    cap = open_video(args.source)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_interval = 1.0 / src_fps if src_fps > 1.0 and not args.no_pacing else 0.0

    print("[System] Loading external perception and MRM modules...")
    detector = YoloDetector(external_cfg)
    distance_estimator = DistanceEstimator(external_cfg)
    collision_warner = CollisionWarner(external_cfg)
    fusion_engine = RiskFusionEngine(fusion_cfg)
    mrm_planner = MRMPlanner(decision_cfg)
    decision_logger = DecisionLogger(decision_cfg)
    visualizer = Visualizer(ui_cfg)
    alerter = AudioAlerter(ui_cfg)

    writer = None
    fps_counter = FPSCounter()
    frame_idx = 0
    paused = False
    last_display = None
    last_log_time = time.time()
    conf_threshold = float(external_cfg.get("conf_threshold", 0.5))

    print("=" * 72)
    print("IDMS MRM Video Demo")
    print(f"video       : {args.source}")
    print(f"model       : {external_cfg.get('model_path')}")
    print(f"device      : {external_cfg.get('device')}")
    print(f"conf        : {conf_threshold:.2f}")
    print(f"decision log: {decision_cfg.get('log_path')}")
    print(f"planner     : {mrm_planner.model_status}")
    if args.output:
        print(f"output      : {args.output}")
    print("=" * 72)

    try:
        while True:
            if not paused or last_display is None:
                if frame_interval > 0:
                    time.sleep(frame_interval)

                ok, frame = cap.read()
                if not ok:
                    if args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        reset_runtime(collision_warner, fusion_engine, mrm_planner, decision_logger)
                        ok, frame = cap.read()
                    if not ok:
                        break

                now = time.time()
                raw_detections = detector.process(frame)
                dist_detections = distance_estimator.calculate(raw_detections)
                vehicle_data = collision_warner.process(
                    dist_detections,
                    frame_width=frame.shape[1],
                )
                fusion_result = fusion_engine.evaluate(
                    vehicle_data=vehicle_data,
                    face_data=None,
                )
                video_time = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if video_time <= 0.0 and src_fps > 0.0:
                    video_time = frame_idx / src_fps
                decision_result = mrm_planner.plan(
                    fusion_result=fusion_result,
                    vehicle_data=vehicle_data,
                    face_data=None,
                    timestamp=now,
                    sequence_timestamp=video_time,
                )
                decision_logger.log(decision_result, timestamp=now)

                ext_has_danger = (
                    fusion_result.ext_score >= 0.7
                    or decision_result.action in ("BRAKE", "EMERGENCY_BRAKE")
                )
                alert_result = alerter.update(ext_danger=ext_has_danger, int_danger=False)

                display = visualizer.draw_results(frame.copy(), face_data=None, vehicle_data=vehicle_data)
                draw_header(display, fps_counter.fps, frame_idx, args.source, paused=False)
                draw_risk_bar(display, fusion_result)
                visualizer.draw_decision_panel(display, decision_result)
                draw_candidate_costs(display, decision_result)

                if alert_result.get("ext_alert_fired"):
                    cv2.putText(
                        display,
                        "COLLISION / MRM ALERT SOUND",
                        (12, min(display.shape[0] - 48, 280)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                last_display = display
                fps_counter.tick()
                frame_idx += 1

                if writer is None and args.output:
                    h, w = display.shape[:2]
                    writer = make_writer(args.output, src_fps, (w, h))
                if writer is not None:
                    writer.write(display)

                if now - last_log_time >= 1.0:
                    lead = decision_result.world_state
                    print(
                        f"[MRM] frame={frame_idx:05d} fps={fps_counter.fps:4.1f} "
                        f"obj={len(vehicle_data):2d} "
                        f"risk={fusion_result.fused_text}:{fusion_result.fused_score:.2f} "
                        f"action={decision_result.action:<15} "
                        f"pred={decision_result.predicted_risk:.2f} "
                        f"ttc={lead.lead_ttc:.1f}s dist={lead.lead_distance:.1f}m "
                        f"why={'; '.join(decision_result.reasons[:2])}"
                    )
                    last_log_time = now

            if args.no_display:
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                continue

            show_frame = last_display
            if paused and show_frame is not None:
                show_frame = show_frame.copy()
                draw_header(show_frame, fps_counter.fps, frame_idx, args.source, paused=True)

            cv2.imshow("IDMS - MRM Video Decision Demo", show_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
            elif key == ord("s") and last_display is not None:
                save_screenshot(last_display, args.save_dir)
            elif key == ord("r"):
                reset_runtime(collision_warner, fusion_engine, mrm_planner, decision_logger)
                print("[System] reset TTC tracking, fusion EMA and decision log timer.")
            elif key in (ord("+"), ord("=")):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                detector.conf_threshold = conf_threshold
                print(f"[Param] conf_threshold={conf_threshold:.2f}")
            elif key == ord("-"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                detector.conf_threshold = conf_threshold
                print(f"[Param] conf_threshold={conf_threshold:.2f}")

            if args.max_frames is not None and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[System] interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        decision_logger.close()
        alerter.close()
        cv2.destroyAllWindows()

    print(f"[Done] processed {frame_idx} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
