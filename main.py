import cv2
import time
import yaml
import multiprocessing as mp
from queue import Empty, Full
import numpy as np  
from src.ui.alert_system import AudioAlerter

# --- 导入我们之前写好的模块 ---
# 1. 舱外感知模块 (External)
from src.external.yolo_detector import YoloDetector
from src.external.distance_est import DistanceEstimator
from src.external.collision_warn import CollisionWarner

# 2. UI 可视化模块 (UI)
from src.ui.visualizer import Visualizer

# 3. 舱内感知模块 (Internal)
from src.internal.face_mesh import FaceMeshDetector

def load_config(path="config.yaml"):
    """加载全局配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[错误] 找不到 config.yaml，请确保它在项目根目录下！")
        exit(1)

def camera_producer(queue, camera_id_int, width, height, label="Cam"):
    """
    [生产者进程] 
    职责：只负责从摄像头读取图像，推送到队列。
    """
    print(f"[{label}] 正在打开摄像头 (ID: {camera_id_int})...")
    cap = cv2.VideoCapture(camera_id_int)
    
    # 设置分辨率 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[错误] {label} 无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 实时性策略 ---
        # 扔掉旧帧 (LIFO)，只保留最新的
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

    # 读取配置文件中的双摄ID（如果没有则使用默认值0和1）
    cam_id_ext = config['system'].get('camera_id_ext', 0)
    cam_id_int = config['system'].get('camera_id_int', 1)
    cam_w = config['system']['frame_width']
    cam_h = config['system']['frame_height']

    # 3. 启动两个摄像头子进程
    p_camera_ext = mp.Process(target=camera_producer, args=(frame_queue_ext, cam_id_ext, cam_w, cam_h, "Cam-Ext"))
    p_camera_int = mp.Process(target=camera_producer, args=(frame_queue_int, cam_id_int, cam_w, cam_h, "Cam-Int"))
    
    p_camera_ext.daemon = True 
    p_camera_int.daemon = True 
    p_camera_ext.start()
    p_camera_int.start()
    print(f"[System] 舱外摄像头进程 PID: {p_camera_ext.pid} | 舱内摄像头进程 PID: {p_camera_int.pid}")

    # 4. 初始化 AI 算法模块 (消费者)
    print("[System] 正在加载 AI 模型 (这可能需要几秒钟)...")
    
    # --- 实例化舱外模块 ---
    try:
        yolo_detector = YoloDetector(config['external'])      
        dist_estimator = DistanceEstimator(config['external']) 
        collision_warner = CollisionWarner(config['external']) 
    except Exception as e:
        print(f"[错误] YOLO 模型加载失败: {e}")
        return

    # --- 实例化 UI 模块 ---
    visualizer = Visualizer(config['ui']) 

    # 新增：实例化声音报警模块
    alerter = AudioAlerter(config.get('ui', {}))

    # --- 实例化舱内模块 ---
    config['internal']['return_landmarks'] = bool(config.get('ui', {}).get('show_landmarks', False))
    face_detector = FaceMeshDetector(config['internal'])

    print("[System] 系统就绪！按 'q' 键退出。")
    
    fps_time = time.time()
    frame_count = 0
    fps_display = 0

    # 缓存最新帧
    frame_ext = None
    frame_int = None

    try:
        while True:
            # 5. 非阻塞式获取双路最新帧
            try:
                frame_ext = frame_queue_ext.get_nowait()
            except Empty:
                pass
                
            try:
                frame_int = frame_queue_int.get_nowait()
            except Empty:
                pass

            # 等待两个摄像头都至少成功读取到一帧
            if frame_ext is None or frame_int is None:
                time.sleep(0.01)
                continue

            # 拷贝一份用于处理，防止画面绘制互相干扰
            curr_frame_ext = frame_ext.copy()
            curr_frame_int = frame_int.copy()

            # ==========================================================
            #  核心处理流程 (Pipeline)
            # ==========================================================

            # --- A. 舱外环境感知 (处理 curr_frame_ext) ---
            raw_detections = yolo_detector.process(curr_frame_ext)
            dist_detections = dist_estimator.calculate(raw_detections)
            vehicle_data = collision_warner.process(dist_detections)

            # --- B. 舱内驾驶员监测 (处理 curr_frame_int) ---
            face_data = face_detector.process(curr_frame_int)

            # --- C. 结果可视化 (Visualization) ---
            # 巧妙利用原有的 visualizer：
            # 舱外画面：只传 vehicle_data，不传 face_data
            vis_ext = visualizer.draw_results(curr_frame_ext, face_data=None, vehicle_data=vehicle_data)
            # 舱内画面：只传 face_data，不传 vehicle_data
            vis_int = visualizer.draw_results(curr_frame_int, face_data=face_data, vehicle_data=None)

            # --- 强制高度对齐保护 ---
            h_ext, w_ext = vis_ext.shape[:2]
            h_int, w_int = vis_int.shape[:2]

            # 左右拼接两路视频 (要求两路高度一致)
            combined_frame = np.hstack((vis_ext, vis_int))

            # 显示 FPS
            cv2.putText(combined_frame, f"FPS: {fps_display}", (10, combined_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- D. 声音报警 ---
            # 舱外：检查是否有 DANGER 级别目标
            ext_has_danger = any(
                v.get('warning_level', 0) >= 2 for v in vehicle_data
            )
            # 舱内：检查疲劳/分心/点头任一触发
            int_has_danger = bool(
                face_data and (
                    face_data.get('is_drowsy')
                    or face_data.get('is_yawning')
                    or face_data.get('is_distracted')
                    or face_data.get('is_nodding')
                )
            )
            alerter.update(ext_danger=ext_has_danger, int_danger=int_has_danger)

            # ==========================================================

            # 6. 显示最终拼接画面
            cv2.imshow('IDMS - Dual Camera Monitoring', combined_frame)

            # 7. FPS 计算逻辑
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                fps_time = time.time()
                print(f"[Running] FPS: {fps_display} | Objects: {len(vehicle_data)} | Face Detected: {bool(face_data and face_data.get('has_face'))}")

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在停止...")
    finally:
        # 清理资源
        p_camera_ext.terminate()
        p_camera_int.terminate()
        p_camera_ext.join()
        p_camera_int.join()
        cv2.destroyAllWindows()
        alerter.close()
        print("[System] 程序已安全退出。")

if __name__ == '__main__':
    mp.freeze_support() 
    main()