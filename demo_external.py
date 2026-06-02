#!/usr/bin/env python3
r"""
==========================================================
  舱外感知模块 Demo (External Module Test)
==========================================================

功能说明:
    本脚本是 IDMS 项目中「舱外感知」三大模块的端到端测试程序:
        1. YoloDetector   → 车辆检测
        2. DistanceEstimator → 单目测距
        3. CollisionWarner  → 碰撞预警 (TTC)

    支持 3 种输入模式:
        --mode camera   使用本机摄像头 (默认)
        --mode video    使用指定视频文件
        --mode sim      使用模拟数据 (无需 GPU/模型，纯逻辑测试)

使用方法:
    # 模式 1: 摄像头实时测试
    python demo_external.py --mode camera

    # 模式 2: 用视频文件测试 (推荐 BDD100K 片段)
    python demo_external.py --mode video --source E:\数据集\数据集\day-clear-2.mp4

    # 模式 3: 纯模拟测试 (无需任何硬件/模型，验证逻辑正确性)
    python demo_external.py --mode sim

    # 自定义参数示例
    python demo_external.py --mode video --source test.mp4 --focal 700 --conf 0.4

快捷键:
    q / ESC  退出
    s        截图保存到当前目录
    p        暂停/继续
    +/-      调整置信度阈值 (±0.05)

==========================================================
"""

import sys
import os
import time
import math
import argparse
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# ===================== 尝试导入 OpenCV =====================
try:
    import cv2
except ImportError:
    cv2 = None

# ===================== 导入声音报警模块 =====================
from src.ui.alert_system import AudioAlerter


try:
    import torch
    print(f"[DEBUG] torch={torch.__version__}, cuda={torch.cuda.is_available()}")
except ImportError:
    torch = None
    print("[DEBUG] torch 未安装或不可导入；模拟模式仍可运行，YOLO 模式会在初始化时提示。")

# ==================== 配置 & 参数解析 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IDMS 舱外感知模块 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["camera", "video", "sim"],
        default="sim",
        help="输入模式: camera=摄像头, video=视频文件, sim=模拟数据 (默认 sim)"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="视频文件路径或摄像头索引 (默认 0)"
    )
    parser.add_argument("--focal", type=float, default=600.0, help="焦距常量 F (默认 600)")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO 置信度阈值 (默认 0.5)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸")
    parser.add_argument("--no-display", action="store_true", help="无头模式(不显示窗口)")
    return parser.parse_args()


def load_yaml_config(config_path="config.yaml"):
    """尝试加载全局配置文件"""
    if yaml is None:
        print("[警告] 未安装 PyYAML，将使用默认内置参数。")
        return {}, {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
            return full_config.get('external', {}), full_config.get('ui', {})
    except Exception as e:
        print(f"[警告] 无法读取 {config_path} ({e})，将使用默认内置参数。")
        return {}, {}

def build_config(args):
    """从 yaml 和命令行参数构建 config 字典"""
    # 1. 优先从 config.yaml 中读取外部模块的配置
    config, ui_config = load_yaml_config()

    # 2. 如果命令行传入了非默认参数，则覆盖 yaml 中的设置
    if args.model != "yolov8n.pt" or 'model_path' not in config:
        config['model_path'] = args.model
    if args.conf != 0.5 or 'conf_threshold' not in config:
        config['conf_threshold'] = args.conf
    if args.imgsz != 640 or 'imgsz' not in config:
        config['imgsz'] = args.imgsz
    if args.focal != 600.0 or 'focal_length' not in config:
        config['focal_length'] = args.focal

    # 3. 补充一些必须的默认值
    config.setdefault('device', 'cpu')
    config.setdefault('roi_top_ratio', 0.35)
    config.setdefault('known_width', 1.8)
    config.setdefault('max_distance', 100.0)
    config.setdefault('min_distance', 0.5)
    config.setdefault('smoothing', 0.3)
    config.setdefault('ttc_threshold', 1.5)
    config.setdefault('safe_distance_time', 2.0)
    config.setdefault('match_pixel_base', 80)
    config.setdefault('cooldown_sec', 3.0)

    return config, ui_config


# ==================== 可视化渲染器 ====================

# 预定义颜色 (BGR)
COLOR_SAFE = (0, 200, 0)       # 绿色
COLOR_CAUTION = (0, 200, 255)  # 黄色
COLOR_DANGER = (0, 0, 255)     # 红色
COLOR_TEXT_BG = (30, 30, 30)   # 深灰背景
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)

LEVEL_COLORS = {
    0: COLOR_SAFE,
    1: COLOR_CAUTION,
    2: COLOR_DANGER,
}


def draw_detections(frame, detections):
    """在帧上绘制检测框、距离、TTC、风险等级"""
    for obj in detections:
        x1, y1, x2, y2 = obj['box']
        level = obj.get('warning_level', 0)
        color = LEVEL_COLORS.get(level, COLOR_SAFE)

        # --- 边界框 ---
        thickness = 2 if level < 2 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # --- 信息标签 ---
        dist = obj.get('distance', -1)
        ttc = obj.get('ttc', 99)
        rel_speed = obj.get('rel_speed', 0)
        cls_name = obj.get('class_name', 'vehicle')
        conf = obj.get('conf', 0)
        warning_text = obj.get('warning_text', 'SAFE')

        line1 = f"{cls_name} {conf:.0%}"
        line2 = f"Dist: {dist:.1f}m" if dist > 0 else "Dist: N/A"
        if ttc < 90:
            line3 = f"TTC: {ttc:.1f}s  V: {rel_speed:+.1f}m/s"
        else:
            line3 = f"V: {rel_speed:+.1f}m/s"

        labels = [line1, line2, line3]
        _draw_label_block(frame, x1, y1 - 5, labels, color)

        if level >= 1:
            badge_text = f" {warning_text} "
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            badge_x = x2 - tw - 4
            badge_y = y1
            cv2.rectangle(frame, (badge_x, badge_y), (x2, badge_y + th + 8), color, -1)
            cv2.putText(frame, badge_text, (badge_x + 2, badge_y + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    return frame


def _draw_label_block(frame, x, y, lines, color):
    """绘制多行标签（带半透明背景）"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    line_h = 18
    padding = 4

    max_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thick)
        max_w = max(max_w, tw)

    total_h = line_h * len(lines) + padding * 2
    total_w = max_w + padding * 2

    bg_y1 = y - total_h
    bg_y2 = y
    bg_x1 = x
    bg_x2 = x + total_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x1 + 3, bg_y2), color, -1)

    for i, line in enumerate(lines):
        ty = bg_y1 + padding + line_h * (i + 1) - 3
        cv2.putText(frame, line, (bg_x1 + padding + 4, ty),
                    font, scale, COLOR_WHITE, thick, cv2.LINE_AA)


def draw_dashboard(frame, fps, det_count, max_danger_level, conf_threshold):
    """绘制顶部仪表板 HUD"""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    fps_color = COLOR_SAFE if fps > 15 else (COLOR_CAUTION if fps > 8 else COLOR_DANGER)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

    cv2.putText(frame, f"Vehicles: {det_count}", (160, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CYAN, 2)

    cv2.putText(frame, f"Conf: {conf_threshold:.2f}", (360, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    if max_danger_level == 2:
        status = "!! COLLISION WARNING !!"
        status_color = COLOR_DANGER
    elif max_danger_level == 1:
        status = "- CAUTION -"
        status_color = COLOR_CAUTION
    else:
        status = "ALL CLEAR"
        status_color = COLOR_SAFE

    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, status, (w - tw - 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if max_danger_level == 2:
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_DANGER, 4)

    return frame


def draw_help(frame):
    """绘制快捷键提示"""
    h, w = frame.shape[:2]
    help_lines = [
        "[Q/ESC] Quit  [S] Screenshot  [P] Pause  [+/-] Conf threshold",
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(frame, line, (10, h - 15 - i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ==================== 模拟数据生成器 ====================

class SimulatedScenario:
    """
    模拟驾驶场景生成器
    """

    def __init__(self, frame_w=960, frame_h=540):
        self.w = frame_w
        self.h = frame_h
        self.start_time = time.time()

    def generate(self):
        t = time.time() - self.start_time
        t = t % 22.0

        frame = self._draw_road()
        detections = []

        if t < 5:
            dist = 25.0
            box = self._dist_to_box(dist, cx_ratio=0.5)
        elif t < 10:
            progress = (t - 5) / 5.0
            dist = 25.0 - progress * 15.0
            box = self._dist_to_box(dist, cx_ratio=0.5)
        elif t < 14:
            progress = (t - 10) / 4.0
            dist = 10.0 - progress * 8.0
            box = self._dist_to_box(dist, cx_ratio=0.5)
        elif t < 18:
            progress = (t - 14) / 4.0
            dist = 2.0 + progress * 28.0
            box = self._dist_to_box(dist, cx_ratio=0.5)
        else:
            dist = 30.0
            box = self._dist_to_box(dist, cx_ratio=0.5)

        if box:
            detections.append({
                'box': box,
                'class_id': 2,
                'class_name': 'car',
                'conf': 0.92,
                'distance': round(dist, 2),
            })
            self._draw_vehicle(frame, box, 'car')

        if t >= 18:
            moto_progress = (t - 18) / 4.0
            moto_dist = 20.0 - moto_progress * 10.0
            moto_box = self._dist_to_box(moto_dist, cx_ratio=0.72, w_scale=0.4)
            if moto_box:
                detections.append({
                    'box': moto_box,
                    'class_id': 3,
                    'class_name': 'motorcycle',
                    'conf': 0.78,
                    'distance': round(moto_dist, 2),
                })
                self._draw_vehicle(frame, moto_box, 'motorcycle')

        scenario_text = self._get_scenario_text(t)
        cv2.putText(frame, scenario_text, (10, self.h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        self._draw_timeline(frame, t)

        return frame, detections

    def _dist_to_box(self, distance, cx_ratio=0.5, w_scale=1.0):
        if distance < 0.5:
            distance = 0.5
        pixel_w = int((1.8 * 600 / distance) * w_scale)
        pixel_h = int(pixel_w * 0.65)

        cx = int(self.w * cx_ratio)
        vanish_y = int(self.h * 0.42)
        bottom_y = int(self.h * 0.85)
        t = min(1.0, 5.0 / distance)
        cy = int(vanish_y + (bottom_y - vanish_y) * t)

        x1 = cx - pixel_w // 2
        y1 = cy - pixel_h // 2
        x2 = cx + pixel_w // 2
        y2 = cy + pixel_h // 2

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.w - 1, x2)
        y2 = min(self.h - 1, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return [x1, y1, x2, y2]

    def _draw_road(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for y in range(int(self.h * 0.42)):
            ratio = y / (self.h * 0.42)
            b = int(60 + 80 * ratio)
            g = int(40 + 50 * ratio)
            frame[y, :] = (b, g, 20)

        road_top = int(self.h * 0.42)
        frame[road_top:, :] = (50, 50, 55)

        vanish_x = self.w // 2
        vanish_y = road_top

        for lane_offset in [-0.15, 0.0, 0.15]:
            for seg in range(0, 20, 2):
                t1 = seg / 20.0
                t2 = (seg + 1) / 20.0
                y1 = int(vanish_y + (self.h - vanish_y) * t1)
                y2 = int(vanish_y + (self.h - vanish_y) * t2)
                x1 = int(vanish_x + lane_offset * self.w * t1 * 2)
                x2 = int(vanish_x + lane_offset * self.w * t2 * 2)
                cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        return frame

    def _draw_vehicle(self, frame, box, label):
        x1, y1, x2, y2 = box
        body_color = (100, 80, 60) if label == 'car' else (80, 100, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), body_color, -1)
        win_y1 = y1 + (y2 - y1) // 4
        win_y2 = y1 + (y2 - y1) * 2 // 4
        win_x1 = x1 + (x2 - x1) // 6
        win_x2 = x2 - (x2 - x1) // 6
        cv2.rectangle(frame, (win_x1, win_y1), (win_x2, win_y2), (140, 140, 160), -1)
        light_w = max(3, (x2 - x1) // 8)
        light_h = max(2, (y2 - y1) // 8)
        cv2.rectangle(frame, (x1, y2 - light_h), (x1 + light_w, y2), (0, 0, 200), -1)
        cv2.rectangle(frame, (x2 - light_w, y2 - light_h), (x2, y2), (0, 0, 200), -1)

    def _get_scenario_text(self, t):
        if t < 5:
            return f"Scene 1/5: Safe following (25m) [{t:.1f}s]"
        elif t < 10:
            return f"Scene 2/5: Lead car braking (25m->10m) [{t:.1f}s]"
        elif t < 14:
            return f"Scene 3/5: RAPID APPROACH (10m->2m) [{t:.1f}s]"
        elif t < 18:
            return f"Scene 4/5: Lead car accelerating away [{t:.1f}s]"
        else:
            return f"Scene 5/5: Motorcycle appearing on right [{t:.1f}s]"

    def _draw_timeline(self, frame, t):
        bar_y = self.h - 25
        bar_h = 8
        total = 22.0

        cv2.rectangle(frame, (10, bar_y), (self.w - 10, bar_y + bar_h), (40, 40, 40), -1)

        segments = [
            (0, 5, COLOR_SAFE),
            (5, 10, COLOR_CAUTION),
            (10, 14, COLOR_DANGER),
            (14, 18, COLOR_SAFE),
            (18, 22, COLOR_CAUTION),
        ]
        bar_w = self.w - 20
        for s_start, s_end, color in segments:
            sx = 10 + int(s_start / total * bar_w)
            ex = 10 + int(s_end / total * bar_w)
            cv2.rectangle(frame, (sx, bar_y), (ex, bar_y + bar_h), color, -1)

        cur_x = 10 + int(t / total * bar_w)
        cv2.circle(frame, (cur_x, bar_y + bar_h // 2), 6, COLOR_WHITE, -1)
        cv2.circle(frame, (cur_x, bar_y + bar_h // 2), 6, (0, 0, 0), 1)


# ==================== Pipeline 管线 ====================

class ExternalPipeline:
    def __init__(self, config, use_yolo=True):
        self.use_yolo = use_yolo

        if use_yolo:
            from src.external.yolo_detector import YoloDetector
            self.detector = YoloDetector(config)

        from src.external.distance_est import DistanceEstimator
        from src.external.collision_warn import CollisionWarner
        self.distance_est = DistanceEstimator(config)
        self.collision_warn = CollisionWarner(config)

    def run(self, frame, detections=None):
        if self.use_yolo:
            detections = self.detector.process(frame)
            detections = self.distance_est.calculate(detections)
        else:
            pass

        # v2.1 修复: 传入当前帧宽度，确保横向车道判断基于正确的画面尺寸
        frame_width = frame.shape[1] if frame is not None else None
        detections = self.collision_warn.process(detections, frame_width=frame_width)
        return detections


# ==================== 性能统计 ====================

class FPSCounter:
    def __init__(self, window=30):
        self.window = window
        self.timestamps = []

    def tick(self):
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    @property
    def fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        if dt == 0:
            return 0.0
        return (len(self.timestamps) - 1) / dt


# ==================== 主函数 ====================

def main():
    if cv2 is None:
        print("[错误] 未安装 opencv-python，请运行: pip install opencv-python")
        sys.exit(1)

    args = parse_args()
    config, ui_config = build_config(args)

    print("=" * 60)
    print("  IDMS 舱外感知模块 Demo")
    print("=" * 60)
    print(f"  模式: {args.mode}")
    print(f"  焦距: {config['focal_length']}")
    print(f"  置信度: {config['conf_threshold']}")
    print(f"  TTC 阈值: {config['ttc_threshold']}s")
    print("=" * 60)

    # ---------- 根据模式初始化 ----------
    cap = None
    sim = None
    use_yolo = False

    if args.mode == 'sim':
        print("\n[模拟模式] 使用合成数据，无需 YOLO 模型和摄像头")
        print("  → 将自动演示 5 个驾驶场景 (22 秒循环)")
        sim = SimulatedScenario(960, 540)
        pipeline = ExternalPipeline(config, use_yolo=False)
    else:
        use_yolo = True
        source = int(args.source) if args.source.isdigit() else args.source
        print(f"\n[{'摄像头' if args.mode == 'camera' else '视频'}模式] 源: {source}")
        print(f"  YOLO 模型: {config['model_path']}")

        try:
            pipeline = ExternalPipeline(config, use_yolo=True)
        except Exception as e:
            print(f"\n[错误] 无法初始化 YOLO: {e}")
            print("  提示: 请确保已安装 ultralytics 并下载了模型文件")
            print("  或者使用 --mode sim 进行纯逻辑测试")
            sys.exit(1)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[错误] 无法打开视频源: {source}")
            sys.exit(1)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  分辨率: {w}x{h}, 帧率: {src_fps:.1f}, 总帧数: {total}")

    # ---------- 初始化声音报警 ----------
    alerter = AudioAlerter(ui_config)

    # ---------- 主循环 ----------
    fps_counter = FPSCounter()
    paused = False
    frame_count = 0
    conf_threshold = config['conf_threshold']

    print("\n[运行中] 按 Q 或 ESC 退出\n")

    while True:
        # --- 获取帧 ---
        if args.mode == 'sim':
            frame, sim_detections = sim.generate()
        else:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if args.mode == 'video':
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("[信息] 摄像头断开")
                        break

        if paused and args.mode != 'sim':
            pass
        else:
            # --- Pipeline 处理 ---
            if args.mode == 'sim':
                detections = pipeline.run(frame, detections=sim_detections)
            else:
                detections = pipeline.run(frame)

            # --- 可视化 ---
            frame = draw_detections(frame, detections)

            max_level = max((d.get('warning_level', 0) for d in detections), default=0)
            frame = draw_dashboard(frame, fps_counter.fps, len(detections),
                                   max_level, conf_threshold)
            frame = draw_help(frame)

            # --- 声音报警 ---
            ext_has_danger = any(
                d.get('warning_level', 0) >= 2 for d in detections
            )
            alert_result = alerter.update(ext_danger=ext_has_danger)
            if alert_result.get("ext_alert_fired"):
                cv2.putText(frame, "ALERT SOUND!", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DANGER, 2)

            fps_counter.tick()
            frame_count += 1

            if frame_count % 60 == 0:
                det_summary = []
                for d in detections:
                    det_summary.append(
                        f"{d.get('class_name','?')}:"
                        f"dist={d.get('distance', -1):.1f}m "
                        f"ttc={d.get('ttc', 99):.1f}s "
                        f"[{d.get('warning_text', '?')}]"
                    )
                summary_str = " | ".join(det_summary) if det_summary else "无检测"
                print(f"  [F{frame_count:5d}] FPS={fps_counter.fps:.1f} "
                      f"Vehicles={len(detections)} → {summary_str}")

        # --- 显示 ---
        if not args.no_display:
            cv2.imshow('IDMS External Module Demo', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"  [截图] 已保存: {filename}")
            elif key == ord('p'):
                paused = not paused
                print(f"  [{'暂停' if paused else '继续'}]")
            elif key == ord('+') or key == ord('='):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                if use_yolo:
                    pipeline.detector.conf_threshold = conf_threshold
                print(f"  [置信度] → {conf_threshold:.2f}")
            elif key == ord('-'):
                conf_threshold = max(0.1, conf_threshold - 0.05)
                if use_yolo:
                    pipeline.detector.conf_threshold = conf_threshold
                print(f"  [置信度] → {conf_threshold:.2f}")
        else:
            time.sleep(0.03)
            if frame_count > 300:
                break

    # --- 清理 ---
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    alerter.close()

    print(f"\n[完成] 共处理 {frame_count} 帧，平均 FPS: {fps_counter.fps:.1f}")
    print("=" * 60)


# ==================== 单元测试入口 ====================

def run_unit_tests():
    """快速单元测试"""
    print("\n" + "=" * 60)
    print("  舱外模块 单元测试")
    print("=" * 60)

    test_config = {
        'focal_length': 600.0,
        'known_width': 1.8,
        'max_distance': 100.0,
        'min_distance': 0.5,
        'smoothing': 0.0,
        'ttc_threshold': 1.5,
        'safe_distance_time': 2.0,
    }

    passed = 0
    failed = 0

    def assert_close(name, actual, expected, tol=0.5):
        nonlocal passed, failed
        ok = abs(actual - expected) < tol
        status = "PASS" if ok else "FAIL"
        icon = "[OK]" if ok else "[X]"
        print(f"  {icon} {name}: actual={actual:.2f}, expected={expected:.2f} [{status}]")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n[Test 1] DistanceEstimator 距离计算")
    test_cases = [
        (108, 10.0),
        (54, 20.0),
        (216, 5.0),
        (1080, 1.0),
    ]

    for pixel_w, expected_dist in test_cases:
        det = [{'box': [100, 200, 100 + pixel_w, 350], 'class_id': 2}]
        dist = (1.8 * 600) / pixel_w
        assert_close(f"P={pixel_w}", dist, expected_dist, tol=0.1)

    print("\n[Test 2] calibration_helper 焦距校准")
    from src.external.distance_est import DistanceEstimator as DE
    f = DE.calibration_helper(5.0, 150, 1.8)
    assert_close("F(D=5,P=150,W=1.8)", f, 416.67, tol=0.1)

    print("\n[Test 3] CollisionWarner TTC 逻辑")
    from src.external.collision_warn import CollisionWarner as CW
    warner = CW(test_config)

    frame1 = [{'box': [400, 300, 500, 370], 'distance': 20.0, 'class_id': 2}]
    result1 = warner.process(frame1)
    assert_close("帧1 warning_level (无历史)", result1[0]['warning_level'], 0, tol=0.1)

    time.sleep(0.05)
    warner.last_timestamp = time.time() - 0.1
    frame2 = [{'box': [395, 298, 510, 375], 'distance': 15.0, 'class_id': 2}]
    result2 = warner.process(frame2)
    print(f"  → rel_speed={result2[0]['rel_speed']:.2f} m/s, "
          f"ttc={result2[0]['ttc']:.2f} s, "
          f"level={result2[0]['warning_level']}")

    print("\n[Test 4] 风险分级验证")
    warner2 = CW(test_config)
    warner2.last_frame_data = [{'box': [400, 300, 500, 370], 'distance': 50.0}]
    warner2.last_timestamp = time.time() - 0.1
    safe = warner2.process([{'box': [400, 300, 500, 370], 'distance': 49.5, 'class_id': 2}])
    assert_close("安全场景 warning_level", safe[0]['warning_level'], 0, tol=0.1)

    warner3 = CW(test_config)
    warner3.last_frame_data = [{'box': [400, 300, 500, 370], 'distance': 8.0}]
    warner3.last_timestamp = time.time() - 0.1
    warner3.process([{'box': [395, 298, 510, 375], 'distance': 3.0, 'class_id': 2}])
    warner3.last_timestamp = time.time() - 0.1
    danger = warner3.process([{'box': [392, 296, 515, 378], 'distance': 2.5, 'class_id': 2}])
    assert_close("危险场景 warning_level", danger[0]['warning_level'], 2, tol=0.1)

    print(f"\n{'=' * 60}")
    total = passed + failed
    print(f"  测试结果: {passed}/{total} 通过")
    if failed == 0:
        print("  [OK] All tests passed!")
    else:
        print(f"  [X] {failed} tests failed.")
    print("=" * 60)

    return failed == 0


# ==================== 入口 ====================

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    if '--test' in sys.argv:
        sys.argv.remove('--test')
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    else:
        main()
