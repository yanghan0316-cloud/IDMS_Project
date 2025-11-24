import cv2
import time
import yaml
import multiprocessing as mp
from queue import Empty, Full

# 导入未来其他同学写的模块 (现在先注释掉，等他们写好再解开)
# from src.internal.face_mesh import FaceMeshDetector
# from src.external.yolo_detector import YoloDetector
# from src.ui.visualizer import draw_results

def load_config(path="config.yaml"):
    """加载配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def camera_producer(queue, camera_id, width, height):
    """
    [生产者进程]
    职责：独立于主进程，全速读取摄像头画面，放入共享队列。
    解决痛点：避免因图像处理耗时导致的视频流卡顿 (I/O阻塞) 
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- 关键策略：保持实时性 ---
        # 如果队列满了，说明算法处理不过来。
        # 此时必须取出旧帧扔掉，放入新帧，确保算法永远处理“当下”的画面 
        if queue.full():
            try:
                queue.get_nowait()  # 扔掉旧帧
            except Empty:
                pass
        
        try:
            queue.put(frame)
        except Full:
            pass

    cap.release()

def main():
    # 1. 加载配置
    config = load_config()
    print("[系统] 配置加载完毕...")

    # 2. 初始化共享队列 (跨进程通信)
    # maxsize设置为2，强制实现“实时优先” 
    frame_queue = mp.Queue(maxsize=config['system']['queue_size'])

    # 3. 启动摄像头子进程 (Producer)
    p_camera = mp.Process(
        target=camera_producer,
        args=(
            frame_queue, 
            config['system']['camera_id'],
            config['system']['frame_width'],
            config['system']['frame_height']
        )
    )
    p_camera.daemon = True # 设置为守护进程，主程序退出时它也会自动退出
    p_camera.start()
    print("[系统] 摄像头采集进程已启动 (PID: {})...".format(p_camera.pid))

    # 4. 初始化算法模块 (在这里实例化其他同学写的类)
    print("[系统] 正在初始化 AI 模型 (YOLOv8 & MediaPipe)...")
    # face_detector = FaceMeshDetector(config['internal'])  # 舱内算法工程师负责
    # yolo_detector = YoloDetector(config['external'])      # 舱外算法工程师负责

    print("[系统] 开始主循环...")
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            # [消费者] 从队列获取最新的一帧
            try:
                frame = frame_queue.get(timeout=1)
            except Empty:
                continue

            # ==========================================================
            #  这里是核心处理区域 (Process Zone)
            #  其他同学的代码将在这里被调用
            # ==========================================================

            # 1. 舱内检测 (TODO: 调用 internal 模块)
            # face_data = face_detector.process(frame)
            
            # 2. 舱外检测 (TODO: 调用 external 模块)
            # vehicle_data = yolo_detector.process(frame)

            # 3. 绘制结果 (TODO: 调用 ui 模块)
            # frame = draw_results(frame, face_data, vehicle_data)

            # ==========================================================

            # 计算并显示帧率 (FPS)
            frame_count += 1
            if time.time() - fps_time > 1:
                print(f"[性能监控] 当前FPS: {frame_count}") # 
                frame_count = 0
                fps_time = time.time()

            cv2.imshow('IDMS - Intelligent Driver Monitoring System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("停止中...")
    finally:
        # 清理资源
        p_camera.terminate()
        cv2.destroyAllWindows()
        print("[系统] 已安全退出。")

if __name__ == '__main__':
    # Windows下使用多进程必须放在 if __name__ == '__main__': 之下 
    main()