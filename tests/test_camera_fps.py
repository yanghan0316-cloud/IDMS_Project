import cv2  # 导入OpenCV库

def main():
    # 1. 初始化摄像头
    # 参数 0 通常代表电脑自带的第一个摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查连接或权限。")
        return

    print("摄像头已启动。按 'q' 键退出程序。")

    while True:
        # 2. 逐帧读取视频流
        # ret (布尔值): 是否成功读取
        # frame (数组): 当前帧的图像数据（默认是BGR色彩空间）
        ret, frame = cap.read()

        if not ret:
            print("错误：无法接收帧（流结束？）。")
            break

        # 3. 数据预处理：色彩空间转换
        # 将 BGR (蓝绿红) 3通道图像转换为 灰度 (Grayscale) 1通道图像
        # 这一步能显著减少后续算法（如人脸检测）的计算量
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. 可视化显示
        # 创建两个窗口对比：原图 vs 灰度图
        cv2.imshow('Original (RGB)', frame)
        cv2.imshow('Processed (Grayscale)', gray_frame)

        # 5. 退出逻辑
        # cv2.waitKey(1) 等待1毫秒，如果有按键按下则返回按键的ASCII码
        # ord('q') 获取字符 'q' 的ASCII码
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. 资源释放
    # 释放摄像头硬件控制权
    cap.release()
    # 关闭所有OpenCV创建的窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()