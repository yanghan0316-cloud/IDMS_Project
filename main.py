import cv2
import time
import yaml
import multiprocessing as mp
from queue import Empty, Full

# --- 导入我们之前写好的模块 ---
# 1. 舱外感知模块 (External)
from src.external.yolo_detector import YoloDetector
from src.external.distance_est import DistanceEstimator
from src.external.collision_warn import CollisionWarner

# 2. UI 可视化模块 (UI)
from src.ui.visualizer import Visualizer

# 3. 舱内感知模块 (Internal)
# TODO: 等你写好 face_mesh.py 后，解开下面这行的注释
# from src.internal.face_mesh import FaceMeshDetector

def load_config(path="config.yaml"):
    """加载全局配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("[错误] 找不到 config.yaml，请确保它在项目根目录下！")
        exit(1)

def camera_producer(queue, camera_id, width, height):
    """
    [生产者进程] 
    职责：只负责从摄像头读取图像，推送到队列。
    [cite_start]利用多进程绕过 Python GIL 锁，防止视频卡顿 [cite: 34, 37]。
    """
    print(f"[CamProcess] 正在打开摄像头 (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    
    # [cite_start]设置分辨率 [cite: 17]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 实时性策略 ---
        # 如果队列满了，说明主程序处理得太慢。
        # [cite_start]必须扔掉旧帧 (LIFO)，只保留最新的，防止画面延迟 [cite: 42]。
        if queue.full():
            try:
                queue.get_nowait()  # 丢弃旧帧
            except Empty:
                pass
        
        try:
            queue.put(frame)
        except Full:
            pass

    cap.release()
    print("[CamProcess] 摄像头进程已退出。")

def main():
    # 1. 加载配置
    config = load_config()
    print("[System] 配置加载成功。")

    # 2. 初始化多进程通信队列
    # [cite_start]maxsize=2 限制缓冲区大小，强制实时处理 [cite: 42]
    frame_queue = mp.Queue(maxsize=config['system']['queue_size'])

    # 3. 启动摄像头子进程
    p_camera = mp.Process(
        target=camera_producer,
        args=(
            frame_queue, 
            config['system']['camera_id'],
            config['system']['frame_width'],
            config['system']['frame_height']
        )
    )
    p_camera.daemon = True # 设为守护进程，主程序关闭时它也会关闭
    p_camera.start()
    print(f"[System] 摄像头采集进程启动 (PID: {p_camera.pid})")

    # 4. 初始化 AI 算法模块 (消费者)
    print("[System] 正在加载 AI 模型 (这可能需要几秒钟)...")
    
    # --- 实例化舱外模块 ---
    try:
        [cite_start]yolo_detector = YoloDetector(config['external'])      # YOLOv8 [cite: 29]
        [cite_start]dist_estimator = DistanceEstimator(config['external']) # 单目测距 [cite: 104]
        [cite_start]collision_warner = CollisionWarner(config['external']) # 碰撞预警 [cite: 114]
    except Exception as e:
        print(f"[错误] YOLO 模型加载失败: {e}")
        print("请检查 data/models/ 目录下是否有 yolov8n.pt 文件。")
        return

    # --- 实例化 UI 模块 ---
    [cite_start]visualizer = Visualizer(config['ui']) # 绘图器 [cite: 46]

    # --- 实例化舱内模块 (TODO) ---
    # face_detector = FaceMeshDetector(config['internal']) # 待实现

    print("[System] 系统就绪！按 'q' 键退出。")
    
    # 性能统计变量
    fps_time = time.time()
    frame_count = 0
    fps_display = 0

    try:
        while True:
            # 5. 从队列获取最新帧
            try:
                # timeout=1 防止摄像头挂了导致主程序死锁
                frame = frame_queue.get(timeout=1)
            except Empty:
                # 如果队列空了 (摄像头没传数据过来)，就继续等
                continue

            # ==========================================================
            #  核心处理流程 (Pipeline)
            # ==========================================================

            # --- A. 舱外环境感知 (External Perception) ---
            # 1. 检测车辆 (YOLO)
            raw_detections = yolo_detector.process(frame)
            
            # 2. 估算距离 (Distance)
            dist_detections = dist_estimator.calculate(raw_detections)
            
            # 3. 评估风险 (Collision Warning)
            vehicle_data = collision_warner.process(dist_detections)

            # --- B. 舱内驾驶员监测 (Internal Monitoring) ---
            # TODO: 等你写好 face_mesh.py，请替换下面的模拟数据
            
            # face_data = face_detector.process(frame)  <-- 真实代码应该是这样
            
            # [模拟数据] 用于测试 UI 是否正常工作
            # 你可以手动修改这里的值，看看界面会不会变红
            face_data = {
                'ear': 0.32,          # 正常眼睛开度
                'mar': 0.05,          # 嘴巴闭合
                'is_drowsy': False,   # 无疲劳
                'is_yawning': False   # 无哈欠
            }

            # --- C. 结果可视化 (Visualization) ---
            # [cite_start]将所有数据绘制到图像上 [cite: 47]
            frame = visualizer.draw_results(frame, face_data, vehicle_data)

            # 显示 FPS
            cv2.putText(frame, f"FPS: {fps_display}", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ==========================================================

            # 6. 显示最终画面
            cv2.imshow('IDMS - Intelligent Driver Monitoring System', frame)

            # 7. FPS 计算逻辑
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                fps_time = time.time()
                # [cite_start]在终端打印状态，方便调试 [cite: 127]
                print(f"[Running] FPS: {fps_display} | Objects: {len(vehicle_data)}")

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[System] 用户中断，正在停止...")
    finally:
        # 清理资源
        p_camera.terminate()
        p_camera.join()
        cv2.destroyAllWindows()
        print("[System] 程序已安全退出。")

if __name__ == '__main__':
    # [cite_start]Windows 下使用 multiprocessing 必须放在此判断下 [cite: 33]
    mp.freeze_support() # 可选，防止打包成exe时出错
    main()