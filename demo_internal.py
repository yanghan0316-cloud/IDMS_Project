#!/usr/bin/env python3
"""
IDMS 舱内驾驶员监测独立 Demo

本脚本只运行 main.py 中的舱内链路：
    MediaPipe FaceMesh -> EAR/MAR -> PERCLOS/眨眼频率 -> 头部姿态 ->
    分心/点头状态机 -> 可视化 -> 风险融合 -> 声音报警

常用命令：
    python demo_internal.py
    python demo_internal.py --mode video --source path/to/driver.mp4
    python demo_internal.py --csv logs/internal.csv

常用按键：
    q / ESC  退出
    p        暂停/继续
    s        截图
    r        重置融合平滑状态
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.core.risk_fusion import RiskFusionEngine
from src.internal.face_mesh import FaceMeshDetector
from src.ui.alert_system import AudioAlerter
from src.ui.visualizer import Visualizer


RISK_COLORS = {
    0: (0, 200, 0),
    1: (0, 200, 255),
    2: (0, 128, 255),
    3: (0, 0, 255),
}


CSV_FIELDS = [
    "timestamp",
    "frame",
    "has_face",
    "ear",
    "mar",
    "blink",
    "is_drowsy",
    "is_yawning",
    "drowsy_frames",
    "yawn_frames",
    "yaw",
    "pitch",
    "roll",
    "is_distracted",
    "is_nodding",
    "distracted_frames",
    "nod_frames",
    "yaw_grace_cnt",
    "perclos",
    "is_perclos_fatigued",
    "blink_freq",
    "is_blink_freq_high",
    "int_score",
    "int_fatigue_score",
    "int_attention_score",
    "int_fatigue_signals",
    "int_confidence_label",
    "int_has_contradiction",
    "fused_score",
    "fused_level",
    "fused_text",
    "alert_fired",
]


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
        description="IDMS 舱内驾驶员监测独立 Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--mode",
        choices=["camera", "video"],
        default="camera",
        help="输入模式：camera=摄像头，video=视频文件",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="摄像头编号或视频路径；camera 模式为空时使用 config.system.camera_id_int",
    )
    parser.add_argument("--width", type=int, default=None, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=None, help="摄像头采集高度")
    parser.add_argument("--csv", default=None, help="可选：逐帧导出调参 CSV")
    parser.add_argument("--show-landmarks", action="store_true", help="强制显示 FaceMesh 网格点")
    parser.add_argument("--hide-landmarks", action="store_true", help="强制隐藏 FaceMesh 网格点")
    parser.add_argument("--no-display", action="store_true", help="不打开 OpenCV 窗口")
    parser.add_argument("--no-audio", action="store_true", help="禁用声音报警")
    parser.add_argument("--no-pacing", action="store_true", help="视频模式不按原始 FPS 限速")
    parser.add_argument("--no-loop", action="store_true", help="视频播放到末尾后不循环")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    parser.add_argument("--save-dir", default="screenshots", help="截图保存目录")
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
    internal_cfg = dict(cfg.get("internal", {}))
    ui_cfg = dict(cfg.get("ui", {}))

    show_landmarks = bool(ui_cfg.get("show_landmarks", False))
    if args.show_landmarks:
        show_landmarks = True
    if args.hide_landmarks:
        show_landmarks = False

    ui_cfg["show_landmarks"] = show_landmarks
    internal_cfg["return_landmarks"] = show_landmarks

    if args.no_audio:
        alert_cfg = dict(ui_cfg.get("alert", {}))
        alert_cfg["enable"] = False
        ui_cfg["alert"] = alert_cfg

    fusion_cfg = dict(internal_cfg)
    fusion_cfg.update(cfg.get("fusion", {}))
    return system_cfg, internal_cfg, ui_cfg, fusion_cfg


def parse_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


class FrameSource:
    def __init__(self, args: argparse.Namespace, system_cfg: dict):
        self.mode = args.mode
        self.loop_video = not args.no_loop
        self.no_pacing = args.no_pacing
        self.cap: cv2.VideoCapture | None = None
        self.frame_interval = 0.0
        self._last_emit_time = 0.0
        self.label = ""

        width = int(args.width or system_cfg.get("frame_width", 640))
        height = int(args.height or system_cfg.get("frame_height", 480))

        if self.mode == "camera":
            source = args.source if args.source is not None else system_cfg.get("camera_id_int", 0)
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

    def read(self) -> tuple[bool, Any, bool]:
        if self.cap is None:
            return False, None, False

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
        return ok, frame if ok else None, wrapped

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def draw_internal_overlay(
    frame: Any,
    face_data: dict,
    fusion_result: Any,
    alert_result: dict[str, bool],
    fps: float,
    source_label: str,
    paused: bool,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (22, 22, 22), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    state = "PAUSED" if paused else "RUNNING"
    face_state = "YES" if face_data.get("has_face") else "NO"
    cv2.putText(frame, f"IDMS Internal | {source_label} | {state}",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2)
    cv2.putText(
        frame,
        f"FPS:{fps:4.1f}  Face:{face_state}  PERCLOS:{face_data.get('perclos', 0.0):.1%}  "
        f"Blink:{face_data.get('blink_freq', 0.0):.1f}/min",
        (12, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

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
    cv2.putText(
        frame,
        f"I:{fusion_result.int_score:.2f} F:{fusion_result.int_fatigue_score:.2f} "
        f"A:{fusion_result.int_attention_score:.2f}",
        (bar_x, bar_y + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )

    detail_y = 104
    if fusion_result.int_confidence_label != "none":
        cv2.putText(
            frame,
            f"DriverState: {fusion_result.int_confidence_label} "
            f"signals={fusion_result.int_fatigue_signals}",
            (12, detail_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )
        detail_y += 26
    if fusion_result.int_has_contradiction:
        cv2.putText(frame, "DriverState contradiction detected", (12, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        detail_y += 26
    if alert_result.get("int_alert_fired"):
        cv2.putText(frame, "FATIGUE ALERT SOUND", (12, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if fusion_result.int_score >= 0.5 or fusion_result.fused_level >= 2:
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    cv2.putText(frame, "[Q/ESC] Quit  [P] Pause  [S] Screenshot  [R] Reset",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA)


def open_csv(path: str | None) -> tuple[Any | None, csv.writer | None]:
    if not path:
        return None, None
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    writer.writerow(CSV_FIELDS)
    print(f"[CSV] 已启用逐帧记录: {out_path}")
    return fp, writer


def write_csv_row(
    writer: csv.writer | None,
    frame_index: int,
    face_data: dict,
    fusion_result: Any,
    alert_result: dict[str, bool],
) -> None:
    if writer is None:
        return
    writer.writerow([
        time.time(),
        frame_index,
        int(bool(face_data.get("has_face"))),
        float(face_data.get("ear", 0.0)),
        float(face_data.get("mar", 0.0)),
        int(bool(face_data.get("blink"))),
        int(bool(face_data.get("is_drowsy"))),
        int(bool(face_data.get("is_yawning"))),
        int(face_data.get("drowsy_frames", 0)),
        int(face_data.get("yawn_frames", 0)),
        float(face_data.get("yaw", 0.0)),
        float(face_data.get("pitch", 0.0)),
        float(face_data.get("roll", 0.0)),
        int(bool(face_data.get("is_distracted"))),
        int(bool(face_data.get("is_nodding"))),
        int(face_data.get("distracted_frames", 0)),
        int(face_data.get("nod_frames", 0)),
        int(face_data.get("yaw_grace_cnt", 0)),
        float(face_data.get("perclos", 0.0)),
        int(bool(face_data.get("is_perclos_fatigued"))),
        float(face_data.get("blink_freq", 0.0)),
        int(bool(face_data.get("is_blink_freq_high"))),
        float(fusion_result.int_score),
        float(fusion_result.int_fatigue_score),
        float(fusion_result.int_attention_score),
        int(fusion_result.int_fatigue_signals),
        fusion_result.int_confidence_label,
        int(bool(fusion_result.int_has_contradiction)),
        float(fusion_result.fused_score),
        int(fusion_result.fused_level),
        fusion_result.fused_text,
        int(bool(alert_result.get("int_alert_fired"))),
    ])


def save_screenshot(frame: Any, save_dir: str, prefix: str = "internal") -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(path), frame)
    print(f"[截图] 已保存: {path}")


def print_startup(args: argparse.Namespace, internal_cfg: dict, source_label: str) -> None:
    print("=" * 72)
    print("IDMS 舱内驾驶员监测 Demo")
    print("=" * 72)
    print(f"输入模式: {args.mode} ({source_label})")
    print(f"EAR 阈值: {internal_cfg.get('ear_threshold', 0.22)}")
    print(f"MAR 阈值: {internal_cfg.get('mar_threshold', 0.60)}")
    print(f"PERCLOS 阈值: {internal_cfg.get('perclos_threshold', 0.15)}")
    print(f"眨眼频率阈值: {internal_cfg.get('blink_freq_high_threshold', 25)}/min")
    print(f"FaceMesh 网格显示: {bool(internal_cfg.get('return_landmarks'))}")
    print("按 q 或 ESC 退出。")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    system_cfg, internal_cfg, ui_cfg, fusion_cfg = build_runtime_config(args)

    source: FrameSource | None = None
    detector: FaceMeshDetector | None = None
    alerter: AudioAlerter | None = None
    csv_fp = None
    fps_counter = FPSCounter(window=30)
    frame_count = 0

    try:
        source = FrameSource(args, system_cfg)
        print_startup(args, internal_cfg, source.label)

        detector = FaceMeshDetector(internal_cfg)
        visualizer = Visualizer(ui_cfg)
        fusion_engine = RiskFusionEngine(fusion_cfg)
        alerter = AudioAlerter(ui_cfg)
        csv_fp, writer = open_csv(args.csv)

        paused = False
        last_frame = None
        log_timer = time.time()

        max_frames = args.max_frames
        if args.no_display and max_frames is None:
            max_frames = 300

        while True:
            if not paused or last_frame is None:
                ok, frame, wrapped = source.read()
                if not ok or frame is None:
                    print("[信息] 输入源结束或断开。")
                    break
                if wrapped:
                    fusion_engine.reset()

                face_data = detector.process(frame)
                display = frame.copy()
                display = visualizer.draw_results(display, face_data=face_data, vehicle_data=None)

                fusion_result = fusion_engine.evaluate(vehicle_data=[], face_data=face_data)
                int_has_danger = fusion_result.int_score >= 0.5
                alert_result = alerter.update(ext_danger=False, int_danger=int_has_danger)

                fps_counter.tick()
                frame_count += 1
                draw_internal_overlay(
                    display,
                    face_data=face_data,
                    fusion_result=fusion_result,
                    alert_result=alert_result,
                    fps=fps_counter.fps,
                    source_label=source.label,
                    paused=paused,
                )
                write_csv_row(writer, frame_count, face_data, fusion_result, alert_result)
                last_frame = display

                now = time.time()
                if now - log_timer >= 1.0:
                    print(
                        f"[Internal] FPS:{fps_counter.fps:4.1f} | "
                        f"Face:{bool(face_data.get('has_face'))} | "
                        f"EAR:{face_data.get('ear', 0.0):.3f} MAR:{face_data.get('mar', 0.0):.3f} | "
                        f"P:{face_data.get('perclos', 0.0):.1%} BF:{face_data.get('blink_freq', 0.0):.1f}/m | "
                        f"Risk:{fusion_result.fused_text}({fusion_result.fused_score:.2f}) "
                        f"I:{fusion_result.int_score:.2f} "
                        f"F:{fusion_result.int_fatigue_score:.2f} "
                        f"A:{fusion_result.int_attention_score:.2f}"
                    )
                    log_timer = now

            if args.no_display:
                if max_frames is not None and frame_count >= max_frames:
                    break
                time.sleep(0.001)
                continue

            cv2.imshow("IDMS Internal Demo", last_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print(f"[状态] {'暂停' if paused else '继续'}")
            elif key == ord("s") and last_frame is not None:
                save_screenshot(last_frame, args.save_dir)
            elif key == ord("r"):
                fusion_engine.reset()
                print("[状态] 已重置融合平滑状态")

            if max_frames is not None and frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在退出...")
    finally:
        if detector is not None:
            detector.close()
        if source is not None:
            source.close()
        if alerter is not None:
            alerter.close()
        if csv_fp is not None:
            csv_fp.close()
        cv2.destroyAllWindows()

    print(f"[完成] 共处理 {frame_count} 帧，平均 FPS(窗口): {fps_counter.fps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
