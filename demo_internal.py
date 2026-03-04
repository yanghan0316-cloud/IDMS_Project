"""demo_internal.py

一个独立的舱内 Demo：只跑 MediaPipe + EAR/MAR + 状态机 + 头部姿态 + 声音报警

运行：
    python demo_internal.py
    python demo_internal.py --csv logs/internal.csv
按 q 退出。

v3 更新:
    - 显示 yaw_grace_cnt 和 nod_frames，方便调参
    - CSV 增加 yaw_grace_cnt 列
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

    # --- 关键修复：根据 UI 配置开启 landmarks 返回（与 main.py 保持一致） ---
    cfg_internal = cfg.get("internal", {})
    show_landmarks = bool(cfg.get("ui", {}).get("show_landmarks", False))
    cfg_internal["return_landmarks"] = show_landmarks

    detector = FaceMeshDetector(cfg_internal)

    cap = cv2.VideoCapture(cfg["system"].get("camera_id_int", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["system"].get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["system"].get("frame_height", 480))

    # ---------- 初始化声音报警 ----------
    alerter = AudioAlerter(cfg.get("ui", {}))

    # 读取 UI 颜色配置
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

            # --- 绘制面部网格关键点（与 main.py 的 Visualizer 一致） ---
            if show_landmarks and data.get("landmarks"):
                for (x, y) in data["landmarks"]:
                    cv2.circle(frame, (int(x), int(y)), 1, normal_color, -1)

            # 画一些关键文字
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

            # 报警文字
            warn = []
            if data.get("is_drowsy"):
                warn.append("DROWSY")
            if data.get("is_yawning"):
                warn.append("YAWN")
            if data.get("is_distracted"):
                warn.append("DISTRACT")
            if data.get("is_nodding"):
                warn.append("NOD")

            if warn:
                cv2.putText(frame, "WARNING: " + ",".join(warn), (frame.shape[1] - 420, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            # ---------- 声音报警 ----------
            int_has_danger = bool(
                data.get("has_face") and (
                    data.get("is_drowsy")
                    or data.get("is_yawning")
                    or data.get("is_distracted")
                    or data.get("is_nodding")
                )
            )
            alert_result = alerter.update(int_danger=int_has_danger)
            if alert_result.get("int_alert_fired"):
                cv2.putText(frame, "ALERT SOUND!", (10, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS 统计
            frames += 1
            if time.time() - t0 >= 1.0:
                fps_disp = frames
                frames = 0
                t0 = time.time()

            cv2.putText(frame, f"FPS: {fps_disp}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 写 CSV
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