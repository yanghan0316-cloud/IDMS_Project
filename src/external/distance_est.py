"""
单目距离估算器
==============
基于相似三角形原理 (Triangle Similarity) 的单目测距。

"""

import numpy as np


class DistanceEstimator:
    # 不同车辆类别的典型物理宽度 (米)
    DEFAULT_WIDTHS = {
        2: 1.8,   # Car
        3: 0.8,   # Motorcycle
        5: 2.5,   # Bus
        7: 2.4,   # Truck
    }

    def __init__(self, config):
        """
        初始化距离估算器

        Args:
            config (dict): 对应 config.yaml 中 'external' 部分
                focal_length (float): 焦距常量 F（像素），默认 600
                known_width (float): 默认物理宽度（米），默认 1.8
                class_widths (dict): 按类别覆盖宽度，如 {7: 2.4}
                max_distance (float): 距离上限（米），默认 100
                min_distance (float): 距离下限（米），默认 0.5
                smoothing (float): EMA 平滑系数 (0~1)，0 表示不平滑
        """
        self.focal_length = config.get('focal_length', 600.0)
        self.default_width = config.get('known_width', 1.8)
        self.max_distance = config.get('max_distance', 100.0)
        self.min_distance = config.get('min_distance', 0.5)
        self.smoothing = config.get('smoothing', 0.0)

        # 合并默认宽度表与用户覆盖
        self.class_widths = dict(self.DEFAULT_WIDTHS)
        user_widths = config.get('class_widths', {})
        self.class_widths.update(user_widths)

        # EMA 历史缓存 {track_key: last_distance}
        self._ema_cache = {}

    def calculate(self, detections):
        """
        为每个检测对象增加 'distance' 字段

        Args:
            detections (list[dict]): 来自 YoloDetector.process() 的列表

        Returns:
            list[dict]: 增加了 'distance' (float, 米) 的列表
                        -1.0 表示无法估算
        """
        for obj in detections:
            x1, y1, x2, y2 = obj['box']
            pixel_width = x2 - x1

            if pixel_width <= 0:
                obj['distance'] = -1.0
                continue

            # 根据类别选择物理宽度
            cls_id = obj.get('class_id', 2)
            real_width = self.class_widths.get(cls_id, self.default_width)

            # 相似三角形公式: D = (W × F) / P
            distance = (real_width * self.focal_length) / pixel_width

            # 钳位到合理范围
            distance = max(self.min_distance, min(distance, self.max_distance))

            # 可选 EMA 平滑（用 box 中心作为简易 key）
            if self.smoothing > 0:
                key = self._box_key(obj['box'])
                prev = self._ema_cache.get(key)
                if prev is not None:
                    alpha = self.smoothing
                    distance = alpha * distance + (1 - alpha) * prev
                self._ema_cache[key] = distance

            obj['distance'] = round(distance, 2)

        return detections

    def _box_key(self, box):
        """粗粒度空间哈希，用于 EMA 匹配"""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        # 量化到 50px 网格
        return (cx // 50, cy // 50)

    @staticmethod
    def calibration_helper(known_distance, pixel_width, real_width=1.8):
        """
        校准辅助工具 —— 计算焦距常量 F

        操作步骤:
        1. 将车停在已知距离处 (如 D=5.0m)
        2. 用 YOLO 检测，读取像素宽度 (如 P=150)
        3. F = calibration_helper(5.0, 150) → 416.67
        4. 填入 config.yaml 的 focal_length
        """
        if real_width <= 0 or pixel_width <= 0:
            return 0.0
        return (pixel_width * known_distance) / real_width