"""
碰撞预警系统 (Forward Collision Warning, FCW)
=============================================
通过帧间距离差计算相对速度 → TTC → 风险分级。

"""

import time
import math


class CollisionWarner:
    # 风险等级常量
    LEVEL_SAFE = 0       # 绿色
    LEVEL_CAUTION = 1    # 黄色
    LEVEL_DANGER = 2     # 红色

    LEVEL_TEXT = {
        0: "SAFE",
        1: "CAUTION",
        2: "DANGER",
    }

    def __init__(self, config):
        """
        初始化碰撞预警系统

        Args:
            config (dict): 对应 config.yaml 的 external 部分
                ttc_threshold (float): TTC 红色警报阈值（秒），默认 1.5
                safe_distance_time (float): 安全跟车时间（秒），默认 2.0
                match_pixel_base (int): 匹配像素基础阈值，默认 80
                cooldown_sec (float): 同目标最高警报冷却时间（秒），默认 3.0
        """
        self.ttc_threshold = config.get('ttc_threshold', 1.5)
        self.safe_distance_time = config.get('safe_distance_time', 2.0)
        self.match_pixel_base = config.get('match_pixel_base', 80)
        self.cooldown_sec = config.get('cooldown_sec', 3.0)

        # 上一帧数据
        self.last_frame_data = []
        self.last_timestamp = time.time()

        # 冷却记录: { (grid_x, grid_y): last_danger_time }
        self._cooldown_map = {}

    def process(self, detections):
        """
        计算相对速度、TTC，评估风险等级

        Args:
            detections (list[dict]): 含 'distance' 和 'box' 的检测列表

        Returns:
            list[dict]: 增加以下字段:
                rel_speed (float): 相对速度 m/s（正=靠近）
                ttc (float): 碰撞时间（秒）
                warning_level (int): 0/1/2
                warning_text (str): SAFE/CAUTION/DANGER
        """
        current_time = time.time()
        time_diff = current_time - self.last_timestamp

        if time_diff < 0.001:
            # 帧间隔太短，直接返回默认值
            for obj in detections:
                obj.setdefault('rel_speed', 0.0)
                obj.setdefault('ttc', 99.0)
                obj.setdefault('warning_level', self.LEVEL_SAFE)
                obj.setdefault('warning_text', self.LEVEL_TEXT[self.LEVEL_SAFE])
            return detections

        # 如果上一帧数据太老 (> 0.5s)，说明中间丢帧了，清空历史
        if time_diff > 0.5:
            self.last_frame_data = []

        for obj in detections:
            # --- 初始化基础状态 ---
            obj['rel_speed'] = 0.0
            obj['ttc'] = 99.0
            obj['warning_level'] = self.LEVEL_SAFE
            obj['raw_level'] = self.LEVEL_SAFE  # 新增：记录当前帧的"原始计算等级"
            obj['streak'] = 1                   # 新增：记录该原始等级"连续出现的帧数"
            obj['warning_text'] = self.LEVEL_TEXT[self.LEVEL_SAFE]

            if obj.get('distance', -1) <= 0:
                continue

            matched = self._find_best_match(obj, self.last_frame_data)

            if matched and matched.get('distance', -1) > 0:
                # 1. 计算相对速度 (保留上一版的 EMA 平滑)
                delta_dist = matched['distance'] - obj['distance']
                raw_rel_speed = delta_dist / time_diff
                prev_speed = matched.get('rel_speed', 0.0)
                rel_speed = 0.2 * raw_rel_speed + 0.8 * prev_speed 
                obj['rel_speed'] = round(rel_speed, 2)

                # 2. 计算当前帧纯粹的 "原始风险等级 (raw_level)"
                raw_level = self.LEVEL_SAFE
                if rel_speed > 1.5:
                    ttc = obj['distance'] / rel_speed
                    obj['ttc'] = round(ttc, 2)
                    if obj['distance'] < 45.0:
                        if ttc < self.ttc_threshold:
                            raw_level = self.LEVEL_DANGER
                        elif obj['distance'] < (rel_speed * self.safe_distance_time):
                            raw_level = self.LEVEL_CAUTION
                else:
                    if obj['distance'] < 2.0:
                        raw_level = self.LEVEL_CAUTION

                # 3. 核心机制：连续帧状态机防抖 (Debounce)
                prev_raw = matched.get('raw_level', self.LEVEL_SAFE)
                streak = matched.get('streak', 0)
                
                # 如果当前计算结果与上一帧相同，连续帧数 +1；否则重置为 1
                if raw_level == prev_raw:
                    streak += 1
                else:
                    streak = 1  
                    
                obj['raw_level'] = raw_level
                obj['streak'] = streak

                # 4. 决定最终输出的等级
                CONFIRM_FRAMES = 2  # 设置连续多少帧才生效
                if streak < CONFIRM_FRAMES:
                    # 状态尚未稳定 (例如突发的1帧误报)，强行继承上一帧的输出状态
                    obj['warning_level'] = matched.get('warning_level', self.LEVEL_SAFE)
                else:
                    # 状态已连续维持 2 帧，确认更新输出状态
                    obj['warning_level'] = raw_level

            # --- 下方保留你原有的冷却期逻辑，不做修改 ---
            grid_key = self._grid_key(obj['box'])
            if obj['warning_level'] == self.LEVEL_DANGER:
                last_danger = self._cooldown_map.get(grid_key, 0)
                if (current_time - last_danger) < self.cooldown_sec:
                    # 冷却期内降级为黄色
                    obj['warning_level'] = self.LEVEL_CAUTION
                else:
                    self._cooldown_map[grid_key] = current_time

            obj['warning_text'] = self.LEVEL_TEXT[obj['warning_level']]

        # 更新历史
        self.last_frame_data = detections
        self.last_timestamp = current_time

        return detections

    def _find_best_match(self, current_obj, old_objs):
        """
        自适应中心点距离匹配

        匹配阈值 = base + box对角线长度 × 0.3
        这样大目标（近车）允许更大的帧间位移
        """
        if not old_objs:
            return None

        cx, cy = self._get_center(current_obj['box'])
        box_diag = math.hypot(
            current_obj['box'][2] - current_obj['box'][0],
            current_obj['box'][3] - current_obj['box'][1],
        )
        threshold = self.match_pixel_base + box_diag * 0.3

        min_dist = float('inf')
        best_match = None

        for old_obj in old_objs:
            old_cx, old_cy = self._get_center(old_obj['box'])
            dist = math.hypot(cx - old_cx, cy - old_cy)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                best_match = old_obj

        return best_match

    @staticmethod
    def _get_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _grid_key(box):
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return (cx // 60, cy // 60)