# 舱外感知 Demo 使用指南

`demo_external.py` 用来单独验证 main.py 中的舱外链路。它会完成车辆检测、单目测距、TTC 碰撞预警、风险融合、画面提示和声音报警。

## 1. 推荐运行顺序

先跑模拟模式，确认逻辑和 UI 正常：

```bash
python demo_external.py --mode sim
```

再跑摄像头：

```bash
python demo_external.py --mode camera
python demo_external.py --mode camera --source 2 --device cpu
```

最后用道路视频复测：

```bash
python demo_external.py --mode video --source E:\data\road.mp4
```

## 2. 三种输入模式

### 模拟模式

```bash
python demo_external.py --mode sim
```

特点：

- 不需要摄像头。
- 不加载 YOLO。
- 使用合成车辆框和距离，专门验证 TTC、碰撞预警、融合评分和报警逻辑。
- 画面会循环演示安全跟车、前车减速、快速接近、前车远离和右侧摩托车出现。

### 摄像头模式

```bash
python demo_external.py --mode camera
```

默认读取 `config.yaml -> system.camera_id_ext`。如需临时指定摄像头：

```bash
python demo_external.py --mode camera --source 1
```

无 GPU 时建议：

```bash
python demo_external.py --mode camera --device cpu
```

### 视频模式

```bash
python demo_external.py --mode video --source E:\data\road.mp4
```

默认按视频原始 FPS 播放，这样 TTC 计算更接近真实时间。只有做吞吐压测时才使用：

```bash
python demo_external.py --mode video --source E:\data\road.mp4 --no-pacing
```

## 3. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--config config.yaml` | 指定配置文件 |
| `--source 0` | 摄像头编号或视频路径 |
| `--device cpu` | 覆盖 YOLO 设备 |
| `--conf 0.5` | 覆盖 YOLO 置信度阈值 |
| `--model yolov8n.pt` | 覆盖模型路径 |
| `--imgsz 640` | 覆盖推理尺寸 |
| `--focal 600` | 覆盖单目测距焦距常量 |
| `--roi-top 0.35` | 覆盖 YOLO ROI 上边界 |
| `--no-audio` | 禁用声音报警 |
| `--no-display --max-frames 300` | 无窗口批量跑固定帧数 |
| `--test` | 运行舱外测距/TTC 快速测试 |

## 4. 画面信息

- 左上角：输入源、运行状态、FPS、目标数量、YOLO 置信度。
- 检测框：绿色安全，黄色注意，红色危险。
- 框标签：类别、距离、TTC。
- 右上角：融合风险等级和分值，包含 `E/I/X` 三个分值。
- 红色闪烁边框：舱外风险达到危险区间。
- 底部：快捷键提示。

## 5. 快捷键

| 按键 | 作用 |
| --- | --- |
| `q` / `ESC` | 退出 |
| `p` | 暂停 / 继续 |
| `s` | 截图 |
| `+` / `-` | 调整 YOLO 置信度 |
| `r` | 重置 TTC 跟踪与融合平滑 |

## 6. 单目测距标定

测距公式是：

```text
distance = real_width * focal_length / pixel_width
```

标定步骤：

1. 将车辆停在已知距离，例如 5 m。
2. 运行摄像头 Demo，记录车辆检测框宽度 `pixel_width`。
3. 用 `DistanceEstimator.calibration_helper(known_distance, pixel_width, real_width)` 计算焦距。
4. 把结果写入 `config.yaml -> external.focal_length`。

常用车辆宽度在 `external.class_widths` 中配置，摩托车、公交车、卡车可单独覆盖。

## 7. 常见问题

- 没有检测框：降低 `--conf`，确认 `roi_top_ratio` 没有裁掉目标，确认模型文件存在。
- 距离明显不准：重新标定 `focal_length`，并检查车辆类别宽度设置。
- TTC 一直很大：需要连续帧同一目标才能估计相对速度，刚启动的前几帧通常正常偏保守。
- 相邻车道误报：调整 `lane_center_ratio`、`lane_full_width_ratio` 或 `lane_relevance_mode`。
- 视频里 TTC 过激：不要使用 `--no-pacing`，让视频按真实 FPS 播放。

