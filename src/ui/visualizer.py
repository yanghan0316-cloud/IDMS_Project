import time
import math

class CollisionWarner:
    def __init__(self, config):
        self.ttc_threshold = 1.5
        self.safe_distance_time = 2.0
        self.last_frame_data = []
        self.last_timestamp = time.time()

    def process(self, detections):
        current_time = time.time()
        time_diff = current_time - self.last_timestamp

        if time_diff < 0.001:
            return detections

        for obj in detections:
            obj['rel_speed'] = 0.0
            obj['ttc'] = 99.0
            obj['warning_level'] = 0

            if obj.get('distance', -1) <= 0:
                continue

            matched_old_obj = self._find_best_match(obj, self.last_frame_data)

            if matched_old_obj:
                delta_dist = matched_old_obj['distance'] - obj['distance']
                rel_speed = delta_dist / time_diff
                obj['rel_speed'] = rel_speed

                if rel_speed > 0.1:
                    ttc = obj['distance'] / rel_speed
                    obj['ttc'] = ttc

                    if ttc < self.ttc_threshold:
                        obj['warning_level'] = 2
                    elif obj['distance'] < (rel_speed * self.safe_distance_time):
                        obj['warning_level'] = 1
                else:
                    if obj['distance'] < 2.0:
                        obj['warning_level'] = 1

        self.last_frame_data = detections
        self.last_timestamp = current_time

        return detections

    def _find_best_match(self, current_obj, old_objs):
        if not old_objs:
            return None

        cx, cy = self._get_center(current_obj['box'])
        min_dist = float('inf')
        best_match = None

        for old_obj in old_objs:
            old_cx, old_cy = self._get_center(old_obj['box'])
            dist = math.hypot(cx - old_cx, cy - old_cy)

            if dist < 100 and dist < min_dist:
                min_dist = dist
                best_match = old_obj

        return best_match

    def _get_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2


# ==========================================================
# Visualizer：把舱内/舱外信息画到画面上
# v4: 新增 PERCLOS 进度条和眨眼频率显示
# ==========================================================
import cv2


class Visualizer:
    def __init__(self, config: dict):
        self.cfg = config or {}
        self.show_landmarks = bool(self.cfg.get("show_landmarks", False))
        self.warning_color = tuple(self.cfg.get("warning_color", [0, 0, 255]))
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
            distracted_frames = face_data.get("distracted_frames", 0)
            nod_frames = face_data.get("nod_frames", 0)
            yaw_grace_cnt = face_data.get("yaw_grace_cnt", 0)

            # v4 新增
            perclos = face_data.get("perclos", 0.0)
            is_perclos_fat = bool(face_data.get("is_perclos_fatigued", False))
            blink_freq = face_data.get("blink_freq", 0.0)
            is_blink_high = bool(face_data.get("is_blink_freq_high", False))

            any_danger = (is_drowsy or is_yawning or is_distracted
                          or is_nodding or is_perclos_fat or is_blink_high)
            color = self.warning_color if any_danger else self.normal_color

            lines = [
                f"EAR: {ear:.3f}",
                f"MAR: {mar:.3f}",
                f"Drowsy: {is_drowsy}",
                f"Yawning: {is_yawning}",
                f"Distracted: {is_distracted} (f:{distracted_frames} g:{yaw_grace_cnt})",
                f"Nodding: {is_nodding} (f:{nod_frames})",
                f"Yaw/Pitch/Roll: {yaw:.1f}/{pitch:.1f}/{roll:.1f}",
                # v4 新增行
                f"PERCLOS: {perclos:.1%} {'[!]' if is_perclos_fat else ''}",
                f"Blink: {blink_freq:.1f}/min {'[!]' if is_blink_high else ''}",
            ]
            y0 = 25
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # v4: PERCLOS 进度条 (在文字下方)
            bar_y = y0 + len(lines) * 22 + 4
            bar_x, bar_w, bar_h = 10, 180, 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * min(1.0, perclos / 0.30))
            fill_c = (0, 0, 255) if perclos > 0.15 else (0, 200, 255) if perclos > 0.08 else (0, 200, 0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_c, -1)
            thresh_x = bar_x + int(bar_w * 0.15 / 0.30)
            cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 255), 1)
            cv2.putText(frame, "PERCLOS", (bar_x + bar_w + 6, bar_y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            if self.show_landmarks and "landmarks" in face_data:
                for (x, y) in face_data["landmarks"]:
                    cv2.circle(frame, (int(x), int(y)), 1, self.normal_color, -1)

            # 右上角警告（优先级排序）
            if is_drowsy:
                cv2.putText(frame, "FATIGUE WARNING!", (w - 310, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_perclos_fat:
                cv2.putText(frame, "PERCLOS WARNING!", (w - 310, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_yawning:
                cv2.putText(frame, "YAWN WARNING!", (w - 260, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_blink_high:
                cv2.putText(frame, "HIGH BLINK RATE!", (w - 310, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_distracted:
                cv2.putText(frame, "DISTRACTION WARNING!", (w - 390, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)
            elif is_nodding:
                cv2.putText(frame, "NODDING WARNING!", (w - 330, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.warning_color, 3)

        return frame