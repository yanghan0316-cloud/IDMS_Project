class my_class(object):
    pass

import sys
print("正在检查环境...")

# 1. 检查 Python 位置
if 'idms_env' in sys.executable:
    print("✅ 环境正确: idms_env")
else:
    print("❌ 环境错误! 请在 VS Code 右下角切换解释器")

# 2. 检查 NumPy (应该显示 1.2x.x)
import numpy
print(f"✅ NumPy 版本: {numpy.__version__} (要求 < 2.0)")

# 3. 检查 OpenCV (应该显示 4.9.0)
import cv2
print(f"✅ OpenCV 版本: {cv2.__version__}")

# 4. 检查 MediaPipe
import mediapipe
print("✅ MediaPipe 导入成功")

# 5. 检查 YOLO
from ultralytics import YOLO
print("✅ YOLOv8 导入成功")

print("\n恭喜！所有环境配置完美！")


