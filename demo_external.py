#!/usr/bin/env python3
"""
IDMS 舱外感知独立 Demo

本脚本只运行 main.py 中的舱外链路：
    YOLOv8 车辆检测 -> 单目测距 -> TTC/碰撞预警 -> 可视化 -> 风险融合 -> 声音报警

支持三种输入：
    python demo_external.py --mode camera
    python demo_external.py --mode video --source path/to/video.mp4
    python demo_external.py --mode sim

常用按键：
    q / ESC  退出
    p        暂停/继续
    s        截图
    + / -    调整 YOLO 置信度阈值
    r        重置 TTC 跟踪状态
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

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


class FPSCounter:
    """滑动窗口 FPS 计数器。"""

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
        description="IDMS 舱外感知独立 Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        choices=["camera", "video", "sim"],
        default="camera",
        help="输入模式：camera=摄像头，video=视频文件，sim=合成场景",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="摄像头编号或视频路径；camera 模式为空时使用 config.system.camera_id_ext",
    )
    parser.add_argument("--width", type=int, default=None, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=None, help="摄像头采集高度")
    parser.add_argument("--model", default=None, help="覆盖 external.model_path")
    parser.add_argument("--device", default=None, help="覆盖 external.device，例如 cpu 或 cuda:0")
    parser.add_argument("--conf", type=float, default=None, help="覆盖 YOLO 置信度阈值")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖 YOLO 推理尺寸")
    parser.add_argument("--focal", type=float, default=None, help="覆盖单目测距焦距常量")
    parser.add_argument("--roi-top", type=float, default=None, help="覆盖 ROI 上边界比例")
    parser.add_argument("--no-display", action="store_true", help="不打开 OpenCV 窗口")
    parser.add_argument("--no-audio", action="store_true", help="禁用声音报警")
    parser.add_argument("--no-pacing", action="store_true", help="视频模式不按原始 FPS 限速")
    parser.add_argument("--no-loop", action="store_true", help="视频播放到末尾后不循环")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    parser.add_argument("--save-dir", default="screenshots", help="截图保存目录")
    parser.add_argument("--test", action="store_true", help="运行舱外核心逻辑快速测试")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise SystemExit(f"[错误] 找不到配置文件: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"[错误] 配置文件格式错误: {exc}") from exc
    return data


def build_runtime_config(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    cfg = load_config(args.config)
    system_cfg = dict(cfg.get("system", {}))
    external_cfg = dict(cfg.get("external", {}))
    ui_cfg = dict(cfg.get("ui", {}))

    if args.model is not None:
        external_cfg["model_path"] = args.model
    if args.device is not None:
        external_cfg["device"] = args.device
    if args.conf is not None:
        external_cfg["conf_threshold"] = args.conf
    if args.imgsz is not None:
        external_cfg["imgsz"] = args.imgsz
    if args.focal is not None:
        external_cfg["focal_length"] = args.focal
    if args.roi_top is not None:
        external_cfg["roi_top_ratio"] = args.roi_top

    external_cfg.setdefault("model_path", "yolov8n.pt")
    external_cfg.setdefault("conf_threshold", 0.5)
    external_cfg.setdefault("imgsz", 640)
    external_cfg.setdefault("device", "cpu")
    external_cfg.setdefault("roi_top_ratio", 0.35)
    external_cfg.setdefault("focal_length", 600.0)
    external_cfg.setdefault("known_width", 1.8)
    external_cfg.setdefault("max_distance", 100.0)
    external_cfg.setdefault("min_distance", 0.5)
    external_cfg.setdefault("smoothing", 0.3)
    external_cfg.setdefault("ttc_threshold", 1.5)
    external_cfg.setdefault("safe_distance_time", 2.0)
    external_cfg.setdefault("match_pixel_base", 80)
    external_cfg.setdefault("cooldown_sec", 3.0)

    if args.no_audio:
        alert_cfg = dict(ui_cfg.get("alert", {}))
        alert_cfg["enable"] = False
        ui_cfg["alert"] = alert_cfg

    fusion_cfg = dict(cfg.get("internal", {}))
    fusion_cfg.update(cfg.get("fusion", {}))
    return system_cfg, external_cfg, ui_cfg, fusion_cfg


def parse_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


class ExternalPipeline:
    """与 main.py 舱外部分一致的处理管线。"""

    def __init__(self, config: dict, use_detector: bool):
        self.config = dict(config)
        self.use_detector = bool(use_detector)
        self.detector = YoloDetector(self.config) if self.use_detector else None
        self.distance_estimator = DistanceEstimator(self.config)
        self.collision_warner = CollisionWarner(self.config)

    def process(self, frame: np.ndarray, detections: list[dict] | None = None) -> list[dict]:
        if self.use_detector:
            detections = self.detector.process(frame)
            detections = self.distance_estimator.calculate(detections)
        else:
            detections = [dict(item) for item in (detections or [])]
            if any("distance" not in item for item in detections):
                detections = self.distance_estimator.calculate(detections)

        return self.collision_warner.process(
            detections,
            frame_width=frame.shape[1],
        )

    def set_conf_threshold(self, value: float) -> None:
        self.config["conf_threshold"] = float(value)
        if self.detector is not None:
            self.detector.conf_threshold = float(value)

    def reset_tracking(self) -> None:
        self.collision_warner = CollisionWarner(self.config)


class SimulatedExternalScenario:
    """合成前车接近/远离场景，用于无摄像头、无 YOLO 的逻辑验证。"""

    def __init__(self, width: int, height: int, config: dict):
        self.width = int(width)
        self.height = int(height)
        self.focal = float(config.get("focal_length", 600.0))
        self.default_width = float(config.get("known_width", 1.8))
        self.start_time = time.time()
        self._last_t = 0.0

    def read(self) -> tuple[np.ndarray, list[dict], bool]:
        t = (time.time() - self.start_time) % 24.0
        wrapped = t < self._last_t
        self._last_t = t

        frame = self._draw_background()
        detections: list[dict] = []

        lead_dist = self._lead_distance(t)
        lead_box = self._box_from_distance(lead_dist, cx_ratio=0.5, width_scale=1.0)
        if lead_box is not None:
            detections.append({
                "box": lead_box,
                "class_id": 2,
                "class_name": "car",
                "conf": 0.94,
                "distance": round(lead_dist, 2),
            })
            self._draw_vehicle(frame, lead_box, color=(80, 100, 150), label="car")

        if t >= 18.0:
            ratio = (t - 18.0) / 6.0
            moto_dist = 24.0 - 10.0 * ratio
            moto_box = self._box_from_distance(
                moto_dist,
                cx_ratio=0.72,
                width_scale=0.45,
                real_width=0.8,
            )
            if moto_box is not None:
                detections.append({
                    "box": moto_box,
                    "class_id": 3,
                    "class_name": "motorcycle",
                    "conf": 0.88,
                    "distance": round(moto_dist, 2),
                })
                self._draw_vehicle(frame, moto_box, color=(100, 120, 80), label="motorcycle")

        self._draw_scene_caption(frame, t)
        self._draw_timeline(frame, t)
        return frame, detections, wrapped

    def _lead_distance(self, t: float) -> float:
        if t < 5.0:
            return 28.0
        if t < 10.0:
            return 28.0 - (t - 5.0) / 5.0 * 16.0
        if t < 14.0:
            return 12.0 - (t - 10.0) / 4.0 * 9.5
        if t < 18.0:
            return 2.5 + (t - 14.0) / 4.0 * 28.0
        return 30.0

    def _box_from_distance(
        self,
        distance: float,
        cx_ratio: float,
        width_scale: float,
        real_width: float | None = None,
    ) -> list[int] | None:
        distance = max(0.6, float(distance))
        real_width = self.default_width if real_width is None else float(real_width)
        pixel_w = int(real_width * self.focal / distance * width_scale)
        pixel_h = int(pixel_w * 0.62)

        cx = int(self.width * cx_ratio)
        horizon_y = int(self.height * 0.40)
        bottom_y = int(self.height * 0.86)
        near_ratio = min(1.0, 5.5 / distance)
        cy = int(horizon_y + (bottom_y - horizon_y) * near_ratio)

        x1 = max(0, cx - pixel_w // 2)
        y1 = max(0, cy - pixel_h // 2)
        x2 = min(self.width - 1, cx + pixel_w // 2)
        y2 = min(self.height - 1, cy + pixel_h // 2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return [x1, y1, x2, y2]

    def _draw_background(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon_y = int(self.height * 0.40)

        for y in range(horizon_y):
            ratio = y / max(1, horizon_y)
            frame[y, :] = (
                int(95 + 40 * ratio),
                int(95 + 30 * ratio),
                int(80 + 20 * ratio),
            )

        frame[horizon_y:, :] = (45, 48, 52)
        vanish = (self.width // 2, horizon_y)
        road_bottom_left = (int(self.width * 0.06), self.height)
        road_bottom_right = (int(self.width * 0.94), self.height)
        cv2.line(frame, vanish, road_bottom_left, (95, 95, 95), 2)
        cv2.line(frame, vanish, road_bottom_right, (95, 95, 95), 2)

        for offset in (-0.18, 0.0, 0.18):
            for seg in range(0, 18, 2):
                t1 = seg / 18.0
                t2 = (seg + 1) / 18.0
                y1 = int(horizon_y + (self.height - horizon_y) * t1)
                y2 = int(horizon_y + (self.height - horizon_y) * t2)
                x1 = int(self.width / 2 + offset * self.width * t1 * 1.7)
                x2 = int(self.width / 2 + offset * self.width * t2 * 1.7)
                cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

        return frame

    @staticmethod
    def _draw_vehicle(frame: np.ndarray, box: list[int], color: tuple[int, int, int], label: str) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 35, 35), 2)

        win_y1 = y1 + max(3, (y2 - y1) // 5)
        win_y2 = y1 + max(5, (y2 - y1) // 2)
        win_x1 = x1 + max(4, (x2 - x1) // 6)
        win_x2 = x2 - max(4, (x2 - x1) // 6)
        cv2.rectangle(frame, (win_x1, win_y1), (win_x2, win_y2), (150, 160, 175), -1)

        if label == "motorcycle":
            cv2.circle(frame, (x1 + 4, y2), 4, (20, 20, 20), -1)
            cv2.circle(frame, (x2 - 4, y2), 4, (20, 20, 20), -1)
        else:
            light_w = max(4, (x2 - x1) // 8)
            light_h = max(3, (y2 - y1) // 10)
            cv2.rectangle(frame, (x1 + 4, y2 - light_h - 2), (x1 + 4 + light_w, y2 - 2), (0, 0, 230), -1)
            cv2.rectangle(frame, (x2 - 4 - light_w, y2 - light_h - 2), (x2 - 4, y2 - 2), (0, 0, 230), -1)

    def _draw_scene_caption(self, frame: np.ndarray, t: float) -> None:
        if t < 5.0:
            text = "Scene 1/5: safe following"
        elif t < 10.0:
            text = "Scene 2/5: lead vehicle braking"
        elif t < 14.0:
            text = "Scene 3/5: rapid approach"
        elif t < 18.0:
            text = "Scene 4/5: lead vehicle leaving"
        else:
            text = "Scene 5/5: adjacent motorcycle"
        cv2.putText(frame, f"{text}  {t:04.1f}s", (12, self.height - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_timeline(self, frame: np.ndarray, t: float) -> None:
        y = self.height - 24
        x = 12
        w = self.width - 24
        h = 8
        cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 35), -1)
        segments = [
            (0.0, 5.0, (0, 200, 0)),
            (5.0, 10.0, (0, 200, 255)),
            (10.0, 14.0, (0, 0, 255)),
            (14.0, 18.0, (0, 200, 0)),
            (18.0, 24.0, (0, 200, 255)),
        ]
        for start, end, color in segments:
            sx = x + int(start / 24.0 * w)
            ex = x + int(end / 24.0 * w)
            cv2.rectangle(frame, (sx, y), (ex, y + h), color, -1)
        cx = x + int(t / 24.0 * w)
        cv2.circle(frame, (cx, y + h // 2), 6, (255, 255, 255), -1)
        cv2.circle(frame, (cx, y + h // 2), 6, (0, 0, 0), 1)


class FrameSource:
    def __init__(self, args: argparse.Namespace, system_cfg: dict, external_cfg: dict):
        self.mode = args.mode
        self.loop_video = not args.no_loop
        self.no_pacing = args.no_pacing
        self.cap: cv2.VideoCapture | None = None
        self.sim: SimulatedExternalScenario | None = None
        self.frame_interval = 0.0
        self._last_emit_time = 0.0
        self.label = ""

        width = int(args.width or system_cfg.get("frame_width", 640))
        height = int(args.height or system_cfg.get("frame_height", 480))

        if self.mode == "sim":
            self.sim = SimulatedExternalScenario(width, height, external_cfg)
            self.label = f"sim {width}x{height}"
            return

        if self.mode == "camera":
            source = args.source if args.source is not None else system_cfg.get("camera_id_ext", 0)
            source = parse_source(source)
            self.label = f"camera {source}"
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            if args.source is None:
                raise SystemExit("[错误] video 模式必须传入 --source 视频路径")
            source = parse_source(args.source)
            self.label = f"video {source}"
            self.cap = cv2.VideoCapture(source)

        if self.cap is None or not self.cap.isOpened():
            raise SystemExit(f"[错误] 无法打开输入源: {self.label}")

        if self.mode == "video":
            src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps > 1.0:
                self.frame_interval = 1.0 / src_fps

    def read(self) -> tuple[bool, np.ndarray | None, list[dict] | None, bool]:
        if self.sim is not None:
            frame, detections, wrapped = self.sim.read()
            return True, frame, detections, wrapped

        if self.cap is None:
            return False, None, None, False

        if self.mode == "video" and self.frame_interval > 0 and not self.no_pacing:
            now = time.time()
            sleep_s = self.frame_interval - (now - self._last_emit_time)
            if sleep_s > 0:
                time.sleep(sleep_s)
            self._last_emit_time = time.time()

        ok, frame = self.cap.read()
        wrapped = False
        if not ok and self.mode == "video" and self.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            wrapped = ok
        return ok, frame if ok else None, None, wrapped

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def draw_fusion_overlay(
    frame: np.ndarray,
    fusion_result: Any,
    alert_result: dict[str, bool],
    fps: float,
    object_count: int,
    source_label: str,
    conf_threshold: float,
    paused: bool,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 74), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    state = "PAUSED" if paused else "RUNNING"
    cv2.putText(frame, f"IDMS External | {source_label} | {state}",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2)
    cv2.putText(frame, f"FPS:{fps:4.1f}  Vehicles:{object_count}  Conf:{conf_threshold:.2f}",
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

    color = RISK_COLORS.get(int(fusion_result.fused_level), (0, 200, 0))
    bar_w = 210
    bar_h = 15
    bar_x = max(12, w - bar_w - 24)
    bar_y = 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * max(0.0, min(1.0, float(fusion_result.fused_score))))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"RISK {fusion_result.fused_text} {fusion_result.fused_score:.2f}",
                (bar_x, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"E:{fusion_result.ext_score:.2f} I:{fusion_result.int_score:.2f} X:{fusion_result.cross_score:.2f}",
                (bar_x, bar_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    if alert_result.get("ext_alert_fired"):
        cv2.putText(frame, "COLLISION ALERT SOUND", (12, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if fusion_result.fused_level >= 3 or fusion_result.ext_score >= 0.7:
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    cv2.putText(frame, "[Q/ESC] Quit  [P] Pause  [S] Screenshot  [+/-] Conf  [R] Reset",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA)


def print_startup(args: argparse.Namespace, external_cfg: dict, source_label: str) -> None:
    print("=" * 72)
    print("IDMS 舱外感知 Demo")
    print("=" * 72)
    print(f"输入模式: {args.mode} ({source_label})")
    if args.mode != "sim":
        print(f"YOLO 模型: {external_cfg['model_path']} | device={external_cfg['device']}")
    else:
        print("模拟模式: 不加载 YOLO，不需要摄像头")
    print(f"置信度阈值: {external_cfg['conf_threshold']:.2f}")
    print(f"焦距常量: {external_cfg['focal_length']:.1f}")
    print(f"TTC 红色阈值: {external_cfg['ttc_threshold']:.2f}s")
    print("按 q 或 ESC 退出。")
    print("=" * 72)


def save_screenshot(frame: np.ndarray, save_dir: str, prefix: str = "external") -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    print(f"[截图] 已保存: {path}")


def run_unit_tests() -> bool:
    print("=" * 72)
    print("IDMS 舱外核心逻辑快速测试")
    print("=" * 72)

    cfg = {
        "focal_length": 600.0,
        "known_width": 1.8,
        "max_distance": 100.0,
        "min_distance": 0.5,
        "smoothing": 0.0,
        "ttc_threshold": 1.5,
        "safe_distance_time": 2.0,
        "cooldown_sec": 0.0,
    }

    passed = 0
    failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        mark = "[OK]" if ok else "[FAIL]"
        print(f"{mark} {name} {detail}".rstrip())
        if ok:
            passed += 1
        else:
            failed += 1

    estimator = DistanceEstimator(cfg)
    detections = [{"box": [100, 100, 208, 180], "class_id": 2, "class_name": "car"}]
    result = estimator.calculate(detections)
    check("DistanceEstimator 10m 标定", math.isclose(result[0]["distance"], 10.0, abs_tol=0.2),
          f"distance={result[0]['distance']:.2f}")

    focal = DistanceEstimator.calibration_helper(known_distance=5.0, pixel_width=150, real_width=1.8)
    check("calibration_helper", math.isclose(focal, 416.67, abs_tol=0.2), f"focal={focal:.2f}")

    warner = CollisionWarner(cfg)
    frame1 = [{"box": [300, 220, 420, 320], "class_id": 2, "class_name": "car", "distance": 20.0}]
    warner.process(frame1, frame_width=640)
    time.sleep(0.08)
    frame2 = [{"box": [295, 218, 430, 326], "class_id": 2, "class_name": "car", "distance": 15.0}]
    result2 = warner.process(frame2, frame_width=640)
    check("CollisionWarner 输出 TTC", result2[0].get("ttc", 99.0) < 99.0,
          f"ttc={result2[0].get('ttc', 99.0):.2f}")
    check("CollisionWarner 风险字段", "warning_level" in result2[0] and "warning_text" in result2[0])

    total = passed + failed
    print("-" * 72)
    print(f"测试结果: {passed}/{total} 通过")
    return failed == 0


def main() -> int:
    args = parse_args()
    if args.test:
        return 0 if run_unit_tests() else 1

    system_cfg, external_cfg, ui_cfg, fusion_cfg = build_runtime_config(args)
    source = FrameSource(args, system_cfg, external_cfg)
    print_startup(args, external_cfg, source.label)

    use_detector = args.mode != "sim"
    pipeline: ExternalPipeline | None = None
    alerter: AudioAlerter | None = None
    fps_counter = FPSCounter(window=30)
    frame_count = 0

    try:
        pipeline = ExternalPipeline(external_cfg, use_detector=use_detector)
        visualizer = Visualizer(ui_cfg)
        fusion_engine = RiskFusionEngine(fusion_cfg)
        alerter = AudioAlerter(ui_cfg)

        paused = False
        last_frame: np.ndarray | None = None
        log_timer = time.time()
        conf_threshold = float(external_cfg.get("conf_threshold", 0.5))

        max_frames = args.max_frames
        if args.no_display and max_frames is None:
            max_frames = 300

        while True:
            if not paused or last_frame is None:
                ok, frame, supplied_detections, reset_tracking = source.read()
                if not ok or frame is None:
                    print("[信息] 输入源结束或断开。")
                    break
                if reset_tracking:
                    pipeline.reset_tracking()
                    fusion_engine.reset()

                vehicle_data = pipeline.process(frame, supplied_detections)
                display = frame.copy()
                display = visualizer.draw_results(display, face_data=None, vehicle_data=vehicle_data)

                fusion_result = fusion_engine.evaluate(vehicle_data=vehicle_data, face_data=None)
                ext_has_danger = fusion_result.ext_score >= 0.7
                alert_result = alerter.update(ext_danger=ext_has_danger, int_danger=False)

                fps_counter.tick()
                frame_count += 1
                draw_fusion_overlay(
                    display,
                    fusion_result=fusion_result,
                    alert_result=alert_result,
                    fps=fps_counter.fps,
                    object_count=len(vehicle_data),
                    source_label=source.label,
                    conf_threshold=conf_threshold,
                    paused=paused,
                )
                last_frame = display

                now = time.time()
                if now - log_timer >= 1.0:
                    min_dist = min((obj.get("distance", 99.0) for obj in vehicle_data), default=99.0)
                    min_ttc = min((obj.get("ttc", 99.0) for obj in vehicle_data), default=99.0)
                    print(
                        f"[External] FPS:{fps_counter.fps:4.1f} | Obj:{len(vehicle_data):2d} | "
                        f"minDist:{min_dist:5.1f}m | minTTC:{min_ttc:5.1f}s | "
                        f"Risk:{fusion_result.fused_text}({fusion_result.fused_score:.2f}) "
                        f"E:{fusion_result.ext_score:.2f}"
                    )
                    log_timer = now

            if args.no_display:
                if max_frames is not None and frame_count >= max_frames:
                    break
                time.sleep(0.001)
                continue

            cv2.imshow("IDMS External Demo", last_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print(f"[状态] {'暂停' if paused else '继续'}")
            elif key == ord("s") and last_frame is not None:
                save_screenshot(last_frame, args.save_dir)
            elif key in (ord("+"), ord("=")):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                pipeline.set_conf_threshold(conf_threshold)
                print(f"[参数] conf_threshold={conf_threshold:.2f}")
            elif key == ord("-"):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                pipeline.set_conf_threshold(conf_threshold)
                print(f"[参数] conf_threshold={conf_threshold:.2f}")
            elif key == ord("r"):
                pipeline.reset_tracking()
                fusion_engine.reset()
                print("[状态] 已重置 TTC 跟踪与融合平滑状态")

            if max_frames is not None and frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在退出...")
    finally:
        source.close()
        if alerter is not None:
            alerter.close()
        cv2.destroyAllWindows()

    print(f"[完成] 共处理 {frame_count} 帧，平均 FPS(窗口): {fps_counter.fps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
