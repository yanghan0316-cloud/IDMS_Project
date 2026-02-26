import cv2

def find_camera_ids():
    print("开始扫描系统中的摄像头 (测试 ID 0 到 5)...\n")
    
    for cam_id in range(6):
        cap = cv2.VideoCapture(cam_id)
        
        # 强制设置为你需要的 640x480 分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            continue
            
        print(f"✅ 成功打开摄像头 ID: {cam_id}")
        print("👉 请查看弹出的视频窗口。用手在镜头前晃动，确认它是【舱内】还是【舱外】摄像头。")
        print("👉 确认完毕后，请在视频窗口处于激活状态时，按键盘的 'q' 键，继续测试下一个。\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 在画面左上角打印当前 ID，防止看错
            cv2.putText(frame, f"Testing Camera ID: {cam_id}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to next", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
            cv2.imshow(f"Camera Test - ID {cam_id}", frame)
            
            # 按 'q' 键关闭当前画面，进入下一个 ID 测试
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    print("🎉 所有摄像头扫描完毕！")

if __name__ == '__main__':
    find_camera_ids()


