"""
碰撞预警系统 (Forward Collision Warning, FCW)
=============================================
通过帧间距离差计算相对速度 → TTC → 风险分级。

v2: 增加横向车道相关性判断，避免相邻车道车辆误报。
v2.1: 修复中距离接近场景无法触发 CAUTION 的问题。
      - 新增基于 TTC 的 CAUTION 区间 (ttc < safe_distance_time)
      - 降低 rel_speed EMA 惯性 (0.2→0.4)
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
        x1, y1, x2, y2 = box
        bbox_cx = (x1 + x2) / 2.0
        bbox_w = x2 - x1

        lane_cx = self.frame_width * self.lane_center_ratio
        lane_half_w = self.frame_width * self.lane_full_width_ratio / 2.0

        offset = abs(bbox_cx - lane_cx)
        effective_offset = max(0.0, offset - bbox_w / 2.0)

        if self.lane_relevance_mode == 'hard':
            return 1.0 if effective_offset < lane_half_w else 0.0

        # 软衰减模式 (默认)
        if effective_offset <= lane_half_w:
            return 1.0
        else:
            overshoot = effective_offset - lane_half_w
            fade_range = lane_half_w * 1.5
            if fade_range <= 0:
                return 0.0
            return max(0.0, 1.0 - overshoot / fade_range)

    def _compute_lateral_speed(self, current_box, matched_box, time_diff):
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
                rel_speed, ttc, warning_level, warning_text, lane_relevance
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
                #    v2.1: 降低惯性 0.2→0.4，使速度更快响应变化
                delta_dist = matched['distance'] - obj['distance']
                raw_rel_speed = delta_dist / time_diff
                prev_speed = matched.get('rel_speed', 0.0)
                rel_speed = 0.4 * raw_rel_speed + 0.6 * prev_speed
                obj['rel_speed'] = round(rel_speed, 2)

                # ====== v2: 计算横向速度 ======
                lat_speed = self._compute_lateral_speed(
                    obj['box'], matched['box'], time_diff
                )

                # 2. 计算当前帧的原始风险等级
                #    v2.1: 重构分级逻辑，确保中距离接近也能触发 CAUTION
                raw_level = self._evaluate_risk(obj, rel_speed)

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
                #    v2.2: 非对称防抖 —— 升级快(2帧)、降级慢(5帧)
                #    防止速度在阈值边缘震荡时输出闪烁
                CONFIRM_UP = 2    # 升级确认帧数（SAFE→CAUTION, CAUTION→DANGER）
                CONFIRM_DOWN = 5  # 降级确认帧数（DANGER→CAUTION, CAUTION→SAFE）

                prev_out = matched.get('warning_level', self.LEVEL_SAFE)
                if raw_level > prev_out:
                    # 试图升级：需要较少帧确认
                    if streak >= CONFIRM_UP:
                        obj['warning_level'] = raw_level
                    else:
                        obj['warning_level'] = prev_out
                elif raw_level < prev_out:
                    # 试图降级：需要更多帧确认
                    if streak >= CONFIRM_DOWN:
                        obj['warning_level'] = raw_level
                    else:
                        obj['warning_level'] = prev_out
                else:
                    # 等级不变
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

    def _evaluate_risk(self, obj, rel_speed):
        """
        v2.2: 重构后的风险分级逻辑

        分级策略:
            DANGER:  TTC < ttc_threshold 且距离 < 45m
            CAUTION: 以下任一条件满足:
                     (a) TTC < safe_distance_time * 3 且距离 < 45m
                     (b) 距离 < safe_distance_time * rel_speed (原有逻辑)
                     (c) rel_speed >= 1.5 且距离 < 25m (中距离持续接近)
                     (d) 距离 < 2.0m (静止近距离)
            SAFE:    其他情况
        """
        distance = obj['distance']

        if rel_speed > 1.0:
            ttc = distance / rel_speed
            obj['ttc'] = round(ttc, 2)

            if distance < 45.0:
                # DANGER: TTC 极小
                if ttc < self.ttc_threshold:
                    return self.LEVEL_DANGER

                # CAUTION: TTC 在警告区间内
                # v2.2: 扩大为 safe_distance_time * 3，覆盖中距离接近场景
                if ttc < self.safe_distance_time * 3:
                    return self.LEVEL_CAUTION

                # CAUTION: 不满足安全跟车距离
                if distance < (rel_speed * self.safe_distance_time):
                    return self.LEVEL_CAUTION

            # CAUTION: 中距离区域有明显接近趋势
            # v2.2: 从 > 2.0 降至 >= 1.5，避免速度在 2.0 附近震荡导致闪烁
            if rel_speed >= 1.5 and distance < 25.0:
                return self.LEVEL_CAUTION

        else:
            # 速度很低或在远离，但距离极近
            if distance < 2.0:
                return self.LEVEL_CAUTION

        return self.LEVEL_SAFE

    def _apply_lateral_downgrade(self, raw_level, lane_relevance, lateral_speed):
        if raw_level == self.LEVEL_SAFE:
            return raw_level

        if lane_relevance <= 0.0:
            return self.LEVEL_SAFE

        if lane_relevance < 0.5:
            if raw_level == self.LEVEL_DANGER:
                return self.LEVEL_CAUTION

        if lateral_speed > self.lateral_speed_threshold:
            if raw_level == self.LEVEL_DANGER:
                return self.LEVEL_CAUTION
            elif raw_level == self.LEVEL_CAUTION:
                return self.LEVEL_SAFE

        return raw_level

    def _find_best_match(self, current_obj, old_objs):
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