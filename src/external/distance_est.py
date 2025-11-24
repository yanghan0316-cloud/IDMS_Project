import numpy as np

class DistanceEstimator:
    def __init__(self, config):
        """
        初始化距离估算器
        
        Args:
            config (dict): 对应 config.yaml 中的 'external' 部分
        """
        # 车辆的物理宽度 W (单位: 米)
        # 假设所有车辆宽度约为 1.8 米 (这是一个工程近似值) 
        self.known_width = config.get('known_width', 1.8)
        
        # 摄像头的焦距常量 F (单位: 像素)
        # 注意：这个值必须通过校准获得！公式: F = (P * D) / W 
        self.focal_length = config.get('focal_length', 600.0)

    def calculate(self, detections):
        """
        为检测到的每个物体增加距离信息
        
        Args:
            detections (list): 来自 YoloDetector.process() 的结果列表
                               [{'box': [x1,y1,x2,y2], ...}, ...]
        Returns:
            list: 增加 'distance' 键值后的列表
        """
        for obj in detections:
            x1, y1, x2, y2 = obj['box']
            
            # 1. 计算物体在图像中的像素宽度 P
            pixel_width = x2 - x1
            
            # 保护措施：防止像素宽度为0导致除以零错误
            if pixel_width <= 0:
                obj['distance'] = -1.0
                continue

            # 2. 应用相似三角形公式计算距离 D 
            # D = (W * F) / P
            distance = (self.known_width * self.focal_length) / pixel_width
            
            # 将结果存入字典 (保留2位小数)
            obj['distance'] = round(distance, 2)
            
        return detections

    @staticmethod
    def calibration_helper(known_distance, pixel_width, real_width=1.8):
        """
        [校准辅助工具]
        帮助你在项目开始前算出摄像头的焦距 F。
        
        操作步骤:
        1. 找一辆车，停在距离摄像头已知距离处 (例如 D=5.0米)。
        2. 用 YOLO 检测出这辆车，看它的框有多宽 (例如 P=150像素)。
        3. 调用此函数: focal_length = calibration_helper(5.0, 150)
        4. 把算出的值填入 config.yaml
        
        Args:
            known_distance (float): 物理距离 (米)
            pixel_width (int): 图像中的像素宽度
            real_width (float): 物理宽度 (米)
            
        Returns:
            float: 焦距常量 F
        """
        if real_width == 0: return 0
        return (pixel_width * known_distance) / real_width