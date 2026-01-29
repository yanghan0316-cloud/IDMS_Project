"""
一个独立的舱内 Demo：只跑 MediaPipe + EAR/MAR + 状态机

运行：
    python demo_internal.py
按 q 退出。
"""
import cv2
import yaml
from src.internal.face_mesh import FaceMeshDetector


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    detector = FaceMeshDetector(cfg.get("internal", {}))

    cap = cv2.VideoCapture(cfg["system"].get("camera_id", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["system"].get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["system"].get("frame_height", 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data = detector.process(frame)

        # 画一些关键文字
        y = 30
        for k in ["has_face", "ear", "mar", "blink", "is_drowsy", "is_yawning", "yaw", "pitch", "roll"]:
            v = data.get(k)
            txt = f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 24

        # 简单红色报警
        if data.get("is_drowsy") or data.get("is_yawning"):
            cv2.putText(frame, "WARNING!", (frame.shape[1]-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

        cv2.imshow("Internal Demo (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
