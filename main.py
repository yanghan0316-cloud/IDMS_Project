import cv2
import time
import yaml
import traceback
import multiprocessing as mp
from queue import Empty, Full
import numpy as np
from src.ui.alert_system import AudioAlerter

# --- 导入我们之前写好的模块 ---
from src.external.yolo_detector import YoloDetector
from src.external.distance_est import DistanceEstimator
from src.external.collision_warn import CollisionWarner
from src.ui.visualizer import Visualizer
from src.internal.face_mesh import FaceMeshDetector

import torch
print(f"[DEBUG] torch={torch.__version__}, cuda={torch.cuda.is_available()}")


def load_config(path="config.yaml"):
    """加载全局配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[错误] 找不到 config.yaml，请确保它在项目根目录下！")
        exit(1)


# ==================== 滑动窗口 FPS 计算器 ====================

class FPSCounter:
    """滑动窗口 FPS 计数器，比每秒重置计数器更平滑。"""

    def __init__(self, window: int = 30):
        self.window = window
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


def camera_producer(queue, camera_id, width, height, label="Cam", ready_event=None):
    """
    [生产者进程]
    职责：只负责从摄像头读取图像，推送到队列。
    """
    print(f"[{label}] 正在打开摄像头 (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[错误] {label} 无法打开摄像头 ID={camera_id}！")
        return

    ret, _ = cap.read()
    if not ret:
        print(f"[错误] {label} 摄像头 ID={camera_id} 已打开但无法读取帧！")
        cap.release()
        return

    print(f"[{label}] 摄像头 ID={camera_id} 就绪 ✓")
    if ready_event is not None:
        ready_event.set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass
        try:
            queue.put(frame)
        except Full:
            pass

    cap.release()
    print(f"[{label}] 摄像头进程已退出。")


def main():
    # 1. 加载配置
    config = load_config()
    print("[System] 配置加载成功。")

    # 2. 初始化双路多进程通信队列
    queue_size = config['system'].get('queue_size', 2)
    frame_queue_ext = mp.Queue(maxsize=queue_size)
    frame_queue_int = mp.Queue(maxsize=queue_size)

    cam_id_ext = config['system'].get('camera_id_ext', 0)
    cam_id_int = config['system'].get('camera_id_int', 1)
    cam_w = config['system']['frame_width']
    cam_h = config['system']['frame_height']

    print(f"[System] 读取到的摄像头 ID: ext={cam_id_ext}, int={cam_id_int}")

    ext_ready = mp.Event()
    int_ready = mp.Event()

    # 3. 启动两个摄像头子进程
    p_camera_ext = mp.Process(
        target=camera_producer,
        args=(frame_queue_ext, cam_id_ext, cam_w, cam_h, "Cam-Ext", ext_ready),
    )
    p_camera_int = mp.Process(
        target=camera_producer,
        args=(frame_queue_int, cam_id_int, cam_w, cam_h, "Cam-Int", int_ready),
    )

    p_camera_ext.daemon = True
    p_camera_int.daemon = True
    p_camera_ext.start()
    p_camera_int.start()
    print(f"[System] 舱外摄像头进程 PID: {p_camera_ext.pid} | 舱内摄像头进程 PID: {p_camera_int.pid}")

    # 等待摄像头就绪
    CAMERA_TIMEOUT = 8.0
    print(f"[System] 等待摄像头就绪 (最多 {CAMERA_TIMEOUT} 秒)...")

    ext_ok = ext_ready.wait(timeout=CAMERA_TIMEOUT)
    int_ok = int_ready.wait(timeout=CAMERA_TIMEOUT)

    if not ext_ok:
        print(f"[警告] 舱外摄像头 (ID={cam_id_ext}) 启动超时！")
    if not int_ok:
        print(f"[警告] 舱内摄像头 (ID={cam_id_int}) 启动超时！")
    if not ext_ok and not int_ok:
        print("[错误] 两个摄像头都无法启动，程序退出。")
        p_camera_ext.terminate()
        p_camera_int.terminate()
        return

    use_ext = ext_ok
    use_int = int_ok
    if ext_ok and int_ok:
        print("[System] 双路摄像头均已就绪 ✓")
    elif ext_ok:
        print("[System] 仅舱外摄像头可用，单路模式运行。")
    else:
        print("[System] 仅舱内摄像头可用，单路模式运行。")

    # 4. 初始化 AI 算法模块
    print("[System] 正在加载 AI 模型 (这可能需要几秒钟)...")

    yolo_detector = None
    dist_estimator = None
    collision_warner = None
    if use_ext:
        try:
            yolo_detector = YoloDetector(config['external'])
            dist_estimator = DistanceEstimator(config['external'])
            collision_warner = CollisionWarner(config['external'])
        except Exception as e:
            print(f"[错误] YOLO 模型加载失败: {e}")
            use_ext = False

    visualizer = Visualizer(config['ui'])
    alerter = AudioAlerter(config.get('ui', {}))

    face_detector = None
    if use_int:
        config['internal']['return_landmarks'] = bool(config.get('ui', {}).get('show_landmarks', False))
        face_detector = FaceMeshDetector(config['internal'])

    print("[System] 系统就绪！按 'q' 键退出。")

    # 让队列积累几帧
    time.sleep(0.5)

    # --- 使用滑动窗口 FPS 计数器 ---
    fps_counter = FPSCounter(window=30)
    log_timer = time.time()  # 用于控制终端日志输出频率

    # --- 帧时间戳，用于过期检测 ---
    STALE_THRESHOLD = 1.0  # 超过 1 秒没有新帧，视为过期
    frame_ext = None
    frame_int = None
    frame_ext_time = 0.0
    frame_int_time = 0.0

    try:
        while True:
            # ==================== 子进程崩溃检测 ====================
            if use_ext and not p_camera_ext.is_alive():
                print("[警告] 舱外摄像头进程已意外退出，禁用舱外通道。")
                use_ext = False
            if use_int and not p_camera_int.is_alive():
                print("[警告] 舱内摄像头进程已意外退出，禁用舱内通道。")
                use_int = False
            if not use_ext and not use_int:
                print("[错误] 双路摄像头均已断开，程序退出。")
                break

            # 5. 非阻塞式获取双路最新帧
            if use_ext:
                try:
                    frame_ext = frame_queue_ext.get_nowait()
                    frame_ext_time = time.time()  # 修复 #2: 记录获取时间
                except Empty:
                    pass

            if use_int:
                try:
                    frame_int = frame_queue_int.get_nowait()
                    frame_int_time = time.time()  # 修复 #2: 记录获取时间
                except Empty:
                    pass

            has_any_frame = (use_ext and frame_ext is not None) or (use_int and frame_int is not None)
            if not has_any_frame:
                time.sleep(0.01)
                continue

            # ==================== 过期帧检测 ====================
            now = time.time()
            ext_frame_valid = (use_ext and frame_ext is not None
                               and (now - frame_ext_time) < STALE_THRESHOLD)
            int_frame_valid = (use_int and frame_int is not None
                               and (now - frame_int_time) < STALE_THRESHOLD)

            # ==========================================================
            #  核心处理流程 (Pipeline)
            # ==========================================================

            vehicle_data = []
            face_data = None

            # --- A. 舱外环境感知 ---
            vis_ext = None
            if ext_frame_valid:
                curr_frame_ext = frame_ext.copy()
                raw_detections = yolo_detector.process(curr_frame_ext)
                dist_detections = dist_estimator.calculate(raw_detections)
                vehicle_data = collision_warner.process(dist_detections)
                vis_ext = visualizer.draw_results(curr_frame_ext, face_data=None, vehicle_data=vehicle_data)
            elif use_ext and frame_ext is not None:
                # 帧已过期，显示灰色提示但不做 AI 推理
                vis_ext = frame_ext.copy()
                cv2.putText(vis_ext, "[EXT] Stale Frame", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- B. 舱内驾驶员监测 ---
            vis_int = None
            if int_frame_valid:
                curr_frame_int = frame_int.copy()
                face_data = face_detector.process(curr_frame_int)
                vis_int = visualizer.draw_results(curr_frame_int, face_data=face_data, vehicle_data=None)
            elif use_int and frame_int is not None:
                vis_int = frame_int.copy()
                cv2.putText(vis_int, "[INT] Stale Frame", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- C. 拼接画面 ---
            if vis_ext is not None and vis_int is not None:
                h_ext = vis_ext.shape[0]
                h_int = vis_int.shape[0]
                if h_ext != h_int:
                    scale = h_ext / h_int
                    new_w = int(vis_int.shape[1] * scale)
                    vis_int = cv2.resize(vis_int, (new_w, h_ext))
                combined_frame = np.hstack((vis_ext, vis_int))
            elif vis_ext is not None:
                combined_frame = vis_ext
            elif vis_int is not None:
                combined_frame = vis_int
            else:
                time.sleep(0.01)
                continue

            # --- 滑动窗口 FPS ---
            fps_counter.tick()
            cv2.putText(combined_frame, f"FPS: {fps_counter.fps:.1f}",
                        (10, combined_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- D. 声音报警 ---
            ext_has_danger = any(
                v.get('warning_level', 0) >= 2 for v in vehicle_data
            )
            int_has_danger = bool(
                face_data and face_data.get('has_face') and (
                    face_data.get('is_drowsy')
                    or face_data.get('is_yawning')
                    or face_data.get('is_distracted')
                    or face_data.get('is_nodding')
                )
            )

            # --- 使用 alerter 返回值，在画面上显示报警状态 ---
            alert_result = alerter.update(ext_danger=ext_has_danger, int_danger=int_has_danger)

            alert_y = 80
            if alert_result.get("ext_alert_fired"):
                cv2.putText(combined_frame, "!! COLLISION ALERT SOUND !!",
                            (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert_y += 30
            if alert_result.get("int_alert_fired"):
                cv2.putText(combined_frame, "!! FATIGUE ALERT SOUND !!",
                            (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 6. 显示最终拼接画面
            cv2.imshow('IDMS - Dual Camera Monitoring', combined_frame)

            # 7. 终端日志（每秒输出一次，避免刷屏）
            if now - log_timer >= 1.0:
                face_ok = bool(face_data and face_data.get('has_face')) if face_data else False
                print(f"[Running] FPS: {fps_counter.fps:.1f} | "
                      f"Objects: {len(vehicle_data)} | Face: {face_ok}")
                log_timer = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在停止...")
    except Exception as e:
        # ====== 关键：打印实际的崩溃原因，不再静默退出 ======
        print(f"\n{'='*60}")
        print(f"[错误] 主循环发生异常:")
        print(f"{'='*60}")
        traceback.print_exc()
        print(f"{'='*60}")
    finally:
        p_camera_ext.terminate()
        p_camera_int.terminate()
        p_camera_ext.join(timeout=2)
        p_camera_int.join(timeout=2)
        cv2.destroyAllWindows()
        alerter.close()
        if face_detector:
            face_detector.close()
        print("[System] 程序已安全退出。")


if __name__ == '__main__':
    mp.freeze_support()
    main()