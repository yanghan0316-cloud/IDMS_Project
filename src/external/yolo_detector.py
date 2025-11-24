import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, config):
        """
        初始化 YOLOv8 检测器
        
        Args:
            config (dict): 来自 config.yaml 的 external 配置部分
        """
        print(f"[YOLO] 正在加载模型: {config['model_path']} ...")
        # 加载预训练模型 (yolov8n.pt) 
        self.model = YOLO(config['model_path'])
        
        # 设定置信度阈值 (如 0.5) 
        self.conf_threshold = config.get('conf_threshold', 0.5)
        
        # 定义我们需要关注的 COCO 数据集类别索引
        # COCO 标准索引: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck 
        self.target_classes = {2, 3, 5, 7}
        
        # 用于显示的标签映射
        self.class_names = self.model.names

    def process(self, frame):
        """
        执行推理并过滤结果
        
        Args:
            frame (numpy.ndarray): 输入的视频帧
            
        Returns:
            list: 检测结果列表，每项包含 {'box': [x1,y1,x2,y2], 'class_id': int, 'conf': float}
        """
        # 1. 执行推理 (Inference)
        # verbose=False 防止在终端打印大量日志
        # stream=True 在处理视频流时内存更高效
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        valid_detections = []

        # YOLOv8 返回的是一个 Results 对象列表 (因为可能是 batch 推理)
        # 我们只处理第一张图 (r[0])
        result = results[0]
        
        # 获取边界框数据 (Boxes object)
        boxes = result.boxes

        for box in boxes:
            # 获取类别 ID (转为整数)
            cls_id = int(box.cls[0])
            
            # 2. 类别过滤 (Class Filtering) 
            # 如果不是我们关心的车辆类别，直接跳过
            if cls_id not in self.target_classes:
                continue

            # 获取置信度
            conf = float(box.conf[0])
            
            # 获取边界框坐标 (x1, y1, x2, y2)
            # xyxy 格式适用于画框
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 将结果存入列表
            valid_detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'class_id': cls_id,
                'class_name': self.class_names[cls_id],
                'conf': conf
            })
            
        return valid_detections

# 单元测试代码 (仅当直接运行此文件时执行)
if __name__ == '__main__':
    # 模拟一个配置字典
    dummy_config = {
        'model_path': 'yolov8n.pt',  # 确保你有这个文件，或者库会自动下载
        'conf_threshold': 0.5
    }
    
    detector = YoloDetector(dummy_config)
    
    # 打开摄像头测试
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break
        
        detections = detector.process(img)
        
        # 简单绘制一下结果
        for d in detections:
            x1, y1, x2, y2 = d['box']
            label = f"{d['class_name']} {d['conf']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow('YOLO Test', img)
        if cv2.waitKey(1) == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()