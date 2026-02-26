"""
碰撞预警系统 (Forward Collision Warning, FCW)
=============================================
通过帧间距离差计算相对速度 → TTC → 风险分级。

v2: 增加横向车道相关性判断，避免相邻车道车辆误报。
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

                # ====== v2 新增: 横向车道相关性参数 ======
                lane_center_ratio (float): 自车道中心在画面中的水平位置比例，默认 0.5
                lane_full_width_ratio (float): 自车道完整宽度占画面宽度的比例，默认 0.30
                    → bbox 中心落在 [center - width/2, center + width/2] 内视为"同车道"
                lane_relevance_mode (str): 'hard' 硬切 / 'soft' 软衰减，默认 'soft'
                lateral_speed_threshold (float): 横向速度阈值(像素/秒)，超过则降级，默认 120.0
        """
        self.ttc_threshold = config.get('ttc_threshold', 1.5)
        self.safe_distance_time = config.get('safe_distance_time', 2.0)
        self.match_pixel_base = config.get('match_pixel_base', 80)
        self.cooldown_sec = config.get('cooldown_sec', 3.0)

        # ====== v2 新增参数 ======
        self.lane_center_ratio = config.get('lane_center_ratio', 0.5)
        self.lane_full_width_ratio = config.get('lane_full_width_ratio', 0.30)
        self.lane_relevance_mode = config.get('lane_relevance_mode', 'soft')
        self.lateral_speed_threshold = config.get('lateral_speed_threshold', 120.0)

        # 画面宽度（首帧时自动设置，或从 config 读取）
        self.frame_width = config.get('frame_width', 640)

        # 上一帧数据
        self.last_frame_data = []
        self.last_timestamp = time.time()

        # 冷却记录: { (grid_x, grid_y): last_danger_time }
        self._cooldown_map = {}

    # ------------------------------------------------------------------
    #  v2 新增: 横向车道相关性评估
    # ------------------------------------------------------------------

    def _compute_lane_relevance(self, box):
        """
        计算目标与自车道的相关性得分。

        原理:
            以画面水平中心为参考，目标 bbox 中心越偏离自车道区域，
            得分越低。同时考虑目标的宽度（近处大目标即使中心稍偏
            也可能横跨自车道）。

        Returns:
            float: 0.0 ~ 1.0, 1.0 = 完全在自车道内
        """
        x1, y1, x2, y2 = box
        bbox_cx = (x1 + x2) / 2.0
        bbox_w = x2 - x1

        # 自车道在画面中的像素范围
        lane_cx = self.frame_width * self.lane_center_ratio
        lane_half_w = self.frame_width * self.lane_full_width_ratio / 2.0

        # bbox 中心相对车道中心的偏移
        offset = abs(bbox_cx - lane_cx)

        # 考虑 bbox 自身宽度：大目标即使中心偏移也可能覆盖自车道
        # 有效偏移 = 中心偏移 - bbox半宽（但不低于0）
        effective_offset = max(0.0, offset - bbox_w / 2.0)

        if self.lane_relevance_mode == 'hard':
            # 硬切模式：bbox 任何部分与车道区域重叠则为 1.0，否则 0.0
            return 1.0 if effective_offset < lane_half_w else 0.0

        # 软衰减模式 (默认)
        if effective_offset <= lane_half_w:
            return 1.0
        else:
            # 超出车道区域后线性衰减，超出 1.5 倍车道宽度时降到 0
            overshoot = effective_offset - lane_half_w
            fade_range = lane_half_w * 1.5  # 衰减区间
            if fade_range <= 0:
                return 0.0
            return max(0.0, 1.0 - overshoot / fade_range)

    def _compute_lateral_speed(self, current_box, matched_box, time_diff):
        """
        计算帧间横向速度 (像素/秒)

        Returns:
            float: 横向速度绝对值
        """
        if time_diff <= 0:
            return 0.0
        curr_cx = (current_box[0] + current_box[2]) / 2.0
        prev_cx = (matched_box[0] + matched_box[2]) / 2.0
        return abs(curr_cx - prev_cx) / time_diff

    # ------------------------------------------------------------------

    def process(self, detections, frame_width=None):
        """
        计算相对速度、TTC，评估风险等级

        Args:
            detections (list[dict]): 含 'distance' 和 'box' 的检测列表
            frame_width (int|None): 当前帧宽度，用于动态更新横向判断参考

        Returns:
            list[dict]: 增加以下字段:
                rel_speed (float): 相对速度 m/s（正=靠近）
                ttc (float): 碰撞时间（秒）
                warning_level (int): 0/1/2
                warning_text (str): SAFE/CAUTION/DANGER
                lane_relevance (float): 车道相关性 0~1 (v2 新增, 供调试)
        """
        current_time = time.time()
        time_diff = current_time - self.last_timestamp

        # 如果提供了帧宽度，动态更新
        if frame_width is not None:
            self.frame_width = frame_width

        if time_diff < 0.001:
            for obj in detections:
                obj.setdefault('rel_speed', 0.0)
                obj.setdefault('ttc', 99.0)
                obj.setdefault('warning_level', self.LEVEL_SAFE)
                obj.setdefault('warning_text', self.LEVEL_TEXT[self.LEVEL_SAFE])
                obj.setdefault('lane_relevance', 1.0)
            return detections

        # 如果上一帧数据太老 (> 0.5s)，说明中间丢帧了，清空历史
        if time_diff > 0.5:
            self.last_frame_data = []

        for obj in detections:
            # --- 初始化基础状态 ---
            obj['rel_speed'] = 0.0
            obj['ttc'] = 99.0
            obj['warning_level'] = self.LEVEL_SAFE
            obj['raw_level'] = self.LEVEL_SAFE
            obj['streak'] = 1
            obj['warning_text'] = self.LEVEL_TEXT[self.LEVEL_SAFE]

            # ====== v2: 计算车道相关性 ======
            lane_rel = self._compute_lane_relevance(obj['box'])
            obj['lane_relevance'] = round(lane_rel, 2)

            if obj.get('distance', -1) <= 0:
                continue

            matched = self._find_best_match(obj, self.last_frame_data)

            if matched and matched.get('distance', -1) > 0:
                # 1. 计算相对速度 (EMA 平滑)
                delta_dist = matched['distance'] - obj['distance']
                raw_rel_speed = delta_dist / time_diff
                prev_speed = matched.get('rel_speed', 0.0)
                rel_speed = 0.2 * raw_rel_speed + 0.8 * prev_speed
                obj['rel_speed'] = round(rel_speed, 2)

                # ====== v2: 计算横向速度 ======
                lat_speed = self._compute_lateral_speed(
                    obj['box'], matched['box'], time_diff
                )

                # 2. 计算当前帧的原始风险等级
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

                # ====== v2 核心: 根据横向信息降级 ======
                raw_level = self._apply_lateral_downgrade(
                    raw_level, lane_rel, lat_speed
                )

                # 3. 连续帧状态机防抖 (Debounce)
                prev_raw = matched.get('raw_level', self.LEVEL_SAFE)
                streak = matched.get('streak', 0)

                if raw_level == prev_raw:
                    streak += 1
                else:
                    streak = 1

                obj['raw_level'] = raw_level
                obj['streak'] = streak

                # 4. 决定最终输出等级
                CONFIRM_FRAMES = 2
                if streak < CONFIRM_FRAMES:
                    obj['warning_level'] = matched.get('warning_level', self.LEVEL_SAFE)
                else:
                    obj['warning_level'] = raw_level

            # --- 冷却期逻辑 ---
            grid_key = self._grid_key(obj['box'])
            if obj['warning_level'] == self.LEVEL_DANGER:
                last_danger = self._cooldown_map.get(grid_key, 0)
                if (current_time - last_danger) < self.cooldown_sec:
                    obj['warning_level'] = self.LEVEL_CAUTION
                else:
                    self._cooldown_map[grid_key] = current_time

            obj['warning_text'] = self.LEVEL_TEXT[obj['warning_level']]

        # 更新历史
        self.last_frame_data = detections
        self.last_timestamp = current_time

        return detections

    def _apply_lateral_downgrade(self, raw_level, lane_relevance, lateral_speed):
        """
        v2 核心逻辑: 根据横向车道相关性和横向速度对风险等级做降级。

        规则:
            1. lane_relevance == 0 → 完全不在自车道，最高只给 SAFE
            2. lane_relevance < 0.5 → 不在自车道核心区域，DANGER 降为 CAUTION
            3. lateral_speed > 阈值 → 目标在快速横向移动（换道/经过），降一级

        这样相邻车道正常行驶的车辆即使纵向距离在缩短，也不会触发红色警报。
        """
        if raw_level == self.LEVEL_SAFE:
            return raw_level

        # 规则 1: 完全不相关（偏得太远）
        if lane_relevance <= 0.0:
            return self.LEVEL_SAFE

        # 规则 2: 弱相关（在车道边缘或相邻车道）
        if lane_relevance < 0.5:
            if raw_level == self.LEVEL_DANGER:
                return self.LEVEL_CAUTION
            # CAUTION 保持（提醒驾驶员旁边有车）

        # 规则 3: 横向速度过大 → 目标在横穿/换道，降一级
        if lateral_speed > self.lateral_speed_threshold:
            if raw_level == self.LEVEL_DANGER:
                return self.LEVEL_CAUTION
            elif raw_level == self.LEVEL_CAUTION:
                return self.LEVEL_SAFE

        return raw_level

    def _find_best_match(self, current_obj, old_objs):
        """
        自适应中心点距离匹配

        匹配阈值 = base + box对角线长度 × 0.3
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