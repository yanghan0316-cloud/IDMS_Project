from src.internal.geometry import calculate_ear, calculate_mar

# 假设你已经从 MediaPipe 拿到了 landmarks 列表
left_ear = calculate_ear(landmarks, [33, 160, 158, 133, 153, 144])
right_ear = calculate_ear(landmarks, [362, 385, 387, 263, 373, 380])

# 取平均值提高稳定性 
avg_ear = (left_ear + right_ear) / 2.0

if avg_ear < CONFIG['internal']['ear_threshold']:
    # 记录疲劳...