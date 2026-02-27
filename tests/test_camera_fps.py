import cv2
import time

def test_camera(cam_id, width=640, height=480):
    print(f"\n正在测试摄像头 ID: {cam_id} ...")
    cap = cv2.VideoCapture(cam_id)
    
    # 强制设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 ID: {cam_id}")
        return

    print(f"✅ 摄像头 {cam_id} 已启动。按 'q' 键退出当前测试。")

    frame_count = 0
    start_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败。")
            break

        frame_count += 1
        elapsed_time = time.time() - start_time

        # 每隔 1 秒计算并更新一次 FPS
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            print(f"摄像头 {cam_id} 当前帧率: {fps:.1f} FPS")

        # 将 FPS 写在画面上
        cv2.putText(frame, f"Cam {cam_id} | {width}x{height} | FPS: {fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(f'FPS Test - Cam {cam_id}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 假设你之前测出来舱内是0，舱外是1，可以在这里修改测试
    test_camera(0)
    test_camera(1)
    test_camera(2)