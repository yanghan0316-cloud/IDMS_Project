"""
YOLOv8 车辆检测器
================
封装 ultralytics YOLOv8，仅保留驾驶场景中需要关注的目标类别。

"""

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # 允许在未安装 ultralytics 时加载模块（用于测试桩）


class YoloDetector:
    def __init__(self, config):
        """
        初始化 YOLOv8 检测器

        Args:
            config (dict): 来自 config.yaml 的 external 配置部分
                必需键:
                    model_path (str): 模型文件路径，如 'yolov8n.pt'
                可选键:
                    conf_threshold (float): 置信度阈值，默认 0.5
                    imgsz (int): 推理输入尺寸，默认 640。降低可提升速度
                    device (str): 'cpu' 或 'cuda:0'，默认 'cpu'
                    roi_top_ratio (float): ROI 上边界占画面高度的比例
                        默认 0.4，即只检测画面下方 60%（路面区域）
                        设为 0.0 则检测全画面
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics 未安装，请运行: pip install ultralytics"
            )

        model_path = config['model_path']
        print(f"[YOLO] 正在加载模型: {model_path} ...")
        self.model = YOLO(model_path)

        self.conf_threshold = config.get('conf_threshold', 0.5)
        self.imgsz = config.get('imgsz', 640)
        self.device = config.get('device', 'cpu')

        # ROI 裁剪：默认只检测下方 60% 区域（跳过天空和远景）
        self.roi_top_ratio = config.get('roi_top_ratio', 0.4)

        # COCO 类别索引: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = {2, 3, 5, 7}
        self.class_names = self.model.names

    def process(self, frame):
        """
        执行推理并过滤结果

        Args:
            frame (numpy.ndarray): 输入的视频帧 (BGR)

        Returns:
            list[dict]: 检测结果列表，每项包含:
                - box: [x1, y1, x2, y2] (相对于原始帧的坐标)
                - class_id: int
                - class_name: str
                - conf: float
        """
        h, w = frame.shape[:2]

        # === ROI 裁剪优化 ===
        roi_y_offset = 0
        if self.roi_top_ratio > 0:
            roi_y_offset = int(h * self.roi_top_ratio)
            roi_frame = frame[roi_y_offset:, :]
        else:
            roi_frame = frame

        # === 推理 ===
        results = self.model(
            roi_frame,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
        )

        valid_detections = []
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])

            # 类别过滤
            if cls_id not in self.target_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 将 ROI 坐标映射回原始帧坐标
            y1 += roi_y_offset
            y2 += roi_y_offset

            valid_detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'class_id': cls_id,
                'class_name': self.class_names[cls_id],
                'conf': conf,
            })

        return valid_detections
