import time
import math

class CollisionWarner:
    def __init__(self, config):
        """
        初始化碰撞预警系统
        
        Args:
            config (dict): 对应 config.yaml 的 external 部分
        """
        # 危险阈值: 碰撞时间小于此值时触发红色警报 (例如 1.5秒) 
        self.ttc_threshold = 1.5
        
        # 警告阈值: 处于跟车距离内但未快速接近时触发黄色警报 (例如 2.0秒行程) 
        self.safe_distance_time = 2.0
        
        # 记录上一帧的状态，用于计算速度
        # 格式: {'track_id': {'distance': float, 'timestamp': float, 'box': [...]}}
        # 注意: 由于大一项目通常不包含复杂的 DeepSORT 追踪，这里我们将使用简单的
        # "中心点匹配"逻辑来近似追踪同一个物体。
        self.last_frame_data = []
        self.last_timestamp = time.time()

    def process(self, detections):
        """
        计算相对速度和 TTC，评估风险等级。
        
        Args:
            detections (list): 包含 'distance' 的检测列表
            
        Returns:
            list: 原列表，但增加了 'rel_speed', 'ttc', 'warning_level' 字段
            warning_level: 0=安全(绿), 1=注意(黄), 2=危险(红)
        """
        current_time = time.time()
        time_diff = current_time - self.last_timestamp
        
        # 避免除以零或过快处理
        if time_diff < 0.001:
            return detections

        # 1. 简易追踪匹配 & 速度计算
        # 我们假设: 这一帧里的车，离上一帧里哪个框最近，它就是同一辆车。
        for obj in detections:
            # 初始化默认值
            obj['rel_speed'] = 0.0     # 相对速度 (m/s)
            obj['ttc'] = 99.0          # 碰撞时间 (秒)
            obj['warning_level'] = 0   # 安全 

            # 如果距离无效，跳过
            if obj.get('distance', -1) <= 0:
                continue

            # 寻找上一帧的匹配对象 (寻找中心点距离最近的旧对象)
            matched_old_obj = self._find_best_match(obj, self.last_frame_data)
            
            if matched_old_obj:
                # 2. 计算相对速度 V_rel 
                # V_rel = (旧距离 - 新距离) / 时间间隔
                # 正值代表正在靠近，负值代表正在远离
                delta_dist = matched_old_obj['distance'] - obj['distance']
                rel_speed = delta_dist / time_diff
                obj['rel_speed'] = rel_speed

                # 3. 计算碰撞时间 TTC 
                # TTC = 当前距离 / 相对速度
                # 只有当正在靠近 (rel_speed > 0.1) 时才计算 TTC
                if rel_speed > 0.1:
                    ttc = obj['distance'] / rel_speed
                    obj['ttc'] = ttc
                    
                    # 4. 风险分级策略 
                    if ttc < self.ttc_threshold:
                        # 红色警报: 极高风险
                        obj['warning_level'] = 2 
                    elif obj['distance'] < (rel_speed * self.safe_distance_time):
                        # 黄色警报: 车距过近 (不满足2秒法则)
                        obj['warning_level'] = 1
                else:
                    # 如果在远离或静止，且距离极近 (< 2米)，也给个黄色警告
                    if obj['distance'] < 2.0:
                        obj['warning_level'] = 1

        # 更新状态供下一帧使用
        self.last_frame_data = detections
        self.last_timestamp = current_time
        
        return detections

    def _find_best_match(self, current_obj, old_objs):
        """
        简单的中心点距离匹配算法 (Simple Euclidean Matching)
        """
        if not old_objs:
            return None
            
        cx, cy = self._get_center(current_obj['box'])
        min_dist = float('inf')
        best_match = None
        
        for old_obj in old_objs:
            old_cx, old_cy = self._get_center(old_obj['box'])
            # 计算中心点距离
            dist = math.hypot(cx - old_cx, cy - old_cy)
            
            # 只有距离比较近的才认为是同一辆车 (例如 50像素以内)
            # 这个阈值取决于车速和帧率
            if dist < 100 and dist < min_dist:
                min_dist = dist
                best_match = old_obj
                
        return best_match

    def _get_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

# ==========================================================
# Visualizer：把舱内/舱外信息画到画面上（最小可用版本）
# ==========================================================
import cv2


class Visualizer:
    """
    最小可用的可视化模块（先保证能跑通）。
    后续你们可以把它替换成更炫的 UI（数字孪生/pygame 也行）。
    """
    def __init__(self, config: dict):
        self.cfg = config or {}
        self.show_landmarks = bool(self.cfg.get("show_landmarks", False))
        self.warning_color = tuple(self.cfg.get("warning_color", [0, 0, 255]))  # BGR
        self.normal_color = tuple(self.cfg.get("normal_color", [0, 255, 0]))

    def draw_results(self, frame, face_data: dict, vehicle_data: list):
        h, w = frame.shape[:2]

        # --- A. 画舱外检测框 ---
        for obj in vehicle_data or []:
            box = obj.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            level = int(obj.get("warning_level", 0))
            color = (0, 255, 0) if level == 0 else ((0, 255, 255) if level == 1 else (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = obj.get("class_name", "obj")
            dist = obj.get("distance", None)
            ttc = obj.get("ttc", None)
            parts = [label]
            if dist is not None:
                parts.append(f"{dist}m")
            if ttc is not None:
                parts.append(f"TTC:{ttc:.1f}s")
            cv2.putText(frame, " ".join(parts), (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- B. 画舱内状态 ---
        if face_data:
            ear = face_data.get("ear", 0.0)
            mar = face_data.get("mar", 0.0)
            is_drowsy = bool(face_data.get("is_drowsy", False))
            is_yawning = bool(face_data.get("is_yawning", False))
            is_distracted = bool(face_data.get("is_distracted", False))
            is_nodding = bool(face_data.get("is_nodding", False))
            yaw = face_data.get("yaw", 0.0)
            pitch = face_data.get("pitch", 0.0)
            roll = face_data.get("roll", 0.0)

            color = self.warning_color if (is_drowsy or is_yawning or is_distracted or is_nodding) else self.normal_color

            lines = [
                f"EAR: {ear:.3f}",
                f"MAR: {mar:.3f}",
                f"Drowsy: {is_drowsy}",
                f"Yawning: {is_yawning}",
                f"Distracted: {is_distracted}",
                f"Nodding: {is_nodding}",
                f"Yaw/Pitch/Roll: {yaw:.1f}/{pitch:.1f}/{roll:.1f}",
            ]
            y0 = 25
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.show_landmarks and "landmarks" in face_data:
                for (x, y) in face_data["landmarks"]:
                    cv2.circle(frame, (int(x), int(y)), 1, self.normal_color, -1)

            if is_drowsy:
                cv2.putText(frame, "FATIGUE WARNING!", (w - 310, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_yawning:
                cv2.putText(frame, "YAWN WARNING!", (w - 260, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_distracted:
                cv2.putText(frame, "DISTRACTION WARNING!", (w - 390, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_nodding:
                cv2.putText(frame, "NODDING WARNING!", (w - 330, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)

        return frame
