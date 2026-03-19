"""demo_internal.py

一个独立的舱内 Demo：只跑 MediaPipe + EAR/MAR + 状态机 + 头部姿态 + 声音报警

运行：
    python demo_internal.py
    python demo_internal.py --csv logs/internal.csv
按 q 退出。

v4 更新:
    - 显示 PERCLOS 百分比和 is_perclos_fatigued
    - 显示 blink_freq (次/分钟) 和 is_blink_freq_high
    - CSV 新增 perclos, is_perclos_fatigued, blink_freq, is_blink_freq_high 列
    - PERCLOS 和 blink_freq 高时纳入警报触发条件
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import yaml

from src.internal.face_mesh import FaceMeshDetector
from src.ui.alert_system import AudioAlerter


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--csv", default="", help="Optional: save per-frame metrics to CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)

    cfg_internal = cfg.get("internal", {})
    show_landmarks = bool(cfg.get("ui", {}).get("show_landmarks", False))
    cfg_internal["return_landmarks"] = show_landmarks

    detector = FaceMeshDetector(cfg_internal)

    cap = cv2.VideoCapture(cfg["system"].get("camera_id_int", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["system"].get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["system"].get("frame_height", 480))

    alerter = AudioAlerter(cfg.get("ui", {}))

    ui_cfg = cfg.get("ui", {})
    normal_color = tuple(ui_cfg.get("normal_color", [0, 255, 0]))
    warning_color = tuple(ui_cfg.get("warning_color", [0, 0, 255]))

    # CSV logger
    csv_fp = None
    writer = None
    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fp = out_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(csv_fp)
        writer.writerow([
            "ts", "has_face",
            "ear", "mar",
            "blink", "is_drowsy", "is_yawning",
            "yaw", "pitch", "roll",
            "is_distracted", "is_nodding",
            "drowsy_frames", "yawn_frames", "distracted_frames", "nod_frames",
            "yaw_grace_cnt",
            # v4 新增列
            "perclos", "is_perclos_fatigued",
            "blink_freq", "is_blink_freq_high",
        ])
        print(f"[Demo] CSV logging enabled: {out_path}")

    # FPS
    t0 = time.time()
    frames = 0
    fps_disp = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            data = detector.process(frame)

            # --- 绘制面部网格关键点 ---
            if show_landmarks and data.get("landmarks"):
                for (x, y) in data["landmarks"]:
                    cv2.circle(frame, (int(x), int(y)), 1, normal_color, -1)

            # --- 左侧指标文字 ---
            y = 28
            show_keys = [
                "has_face",
                "ear", "mar",
                "blink", "is_drowsy", "is_yawning",
                "yaw", "pitch", "roll",
                "is_distracted", "is_nodding",
                "distracted_frames", "nod_frames",
                "yaw_grace_cnt",
            ]
            for k in show_keys:
                v = data.get(k)
                txt = f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                y += 22

            # --- v4: PERCLOS 和眨眼频率显示 ---
            # PERCLOS 用百分比格式和进度条显示
            perclos = data.get("perclos", 0.0)
            is_perclos_fat = data.get("is_perclos_fatigued", False)
            perclos_color = (0, 0, 255) if is_perclos_fat else (0, 255, 0)

            cv2.putText(frame, f"PERCLOS: {perclos:.1%}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, perclos_color, 2)
            y += 22

            # PERCLOS 进度条
            bar_x, bar_y, bar_w, bar_h = 10, y - 4, 180, 12
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * min(1.0, perclos / 0.30))  # 0.30 为条满值
            fill_color = (0, 0, 255) if perclos > 0.15 else (0, 200, 255) if perclos > 0.08 else (0, 200, 0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
            # 15% 阈值标记线
            thresh_x = bar_x + int(bar_w * 0.15 / 0.30)
            cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 255), 1)
            y += 20

            # 眨眼频率
            blink_freq = data.get("blink_freq", 0.0)
            is_blink_high = data.get("is_blink_freq_high", False)
            bf_color = (0, 0, 255) if is_blink_high else (0, 255, 0)
            cv2.putText(frame, f"blink_freq: {blink_freq:.1f}/min", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bf_color, 2)
            y += 22

            if is_perclos_fat:
                cv2.putText(frame, "PERCLOS!", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                y += 22
            if is_blink_high:
                cv2.putText(frame, "BLINK HIGH!", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                y += 22

            # --- 右上角报警文字 ---
            warn = []
            if data.get("is_drowsy"):
                warn.append("DROWSY")
            if data.get("is_yawning"):
                warn.append("YAWN")
            if data.get("is_distracted"):
                warn.append("DISTRACT")
            if data.get("is_nodding"):
                warn.append("NOD")
            if data.get("is_perclos_fatigued"):
                warn.append("PERCLOS")
            if data.get("is_blink_freq_high"):
                warn.append("BLINK-HI")

            if warn:
                cv2.putText(frame, "WARNING: " + ",".join(warn), (frame.shape[1] - 480, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # --- 声音报警 ---
            int_has_danger = bool(
                data.get("has_face") and (
                    data.get("is_drowsy")
                    or data.get("is_yawning")
                    or data.get("is_distracted")
                    or data.get("is_nodding")
                    or data.get("is_perclos_fatigued")    # v4
                    or data.get("is_blink_freq_high")      # v4
                )
            )
            alert_result = alerter.update(int_danger=int_has_danger)
            if alert_result.get("int_alert_fired"):
                cv2.putText(frame, "ALERT SOUND!", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS
            frames += 1
            if time.time() - t0 >= 1.0:
                fps_disp = frames
                frames = 0
                t0 = time.time()

            cv2.putText(frame, f"FPS: {fps_disp}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # CSV
            if writer is not None:
                writer.writerow([
                    time.time(),
                    int(bool(data.get("has_face"))),
                    float(data.get("ear", 0.0)),
                    float(data.get("mar", 0.0)),
                    int(bool(data.get("blink"))),
                    int(bool(data.get("is_drowsy"))),
                    int(bool(data.get("is_yawning"))),
                    float(data.get("yaw", 0.0)),
                    float(data.get("pitch", 0.0)),
                    float(data.get("roll", 0.0)),
                    int(bool(data.get("is_distracted"))),
                    int(bool(data.get("is_nodding"))),
                    int(data.get("drowsy_frames", 0)),
                    int(data.get("yawn_frames", 0)),
                    int(data.get("distracted_frames", 0)),
                    int(data.get("nod_frames", 0)),
                    int(data.get("yaw_grace_cnt", 0)),
                    # v4 新增列
                    float(data.get("perclos", 0.0)),
                    int(bool(data.get("is_perclos_fatigued"))),
                    float(data.get("blink_freq", 0.0)),
                    int(bool(data.get("is_blink_freq_high"))),
                ])

            cv2.imshow("Internal Demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        alerter.close()
        if csv_fp is not None:
            csv_fp.close()


if __name__ == "__main__":
    main()