# IDMS 智能驾驶员监控系统

IDMS 是一个面向行车安全场景的双路视觉监控系统。系统同时感知舱外道路风险和舱内驾驶员状态，并通过多模态融合给出实时风险等级、画面提示和声音报警。

## 核心功能

### 舱外环境感知

- YOLOv8 车辆检测：识别 `car`、`motorcycle`、`bus`、`truck` 等前方交通目标。
- 单目测距：基于车辆像素宽度、类别物理宽度和焦距常量估算距离。
- TTC 碰撞预警：根据帧间距离变化估计相对速度和 Time To Collision。
- 车道相关性：通过目标横向位置过滤相邻车道误报，支持软衰减和硬切模式。

### 舱内驾驶员监测

- MediaPipe FaceMesh：提取面部关键点，可选绘制网格点。
- 疲劳检测：基于 EAR、连续闭眼、PERCLOS 和眨眼频率判断疲劳风险。
- 哈欠检测：基于 MAR 和持续时间判断哈欠。
- 分心与点头：通过 SolvePnP 估计 yaw/pitch/roll，识别持续转头、低头和点头。
- 驾驶员状态协同评估：`DriverStateAssessor` 会综合多个疲劳/注意力信号，区分单一信号、互相印证和矛盾信号。

### 风险融合与报警

- `RiskFusionEngine` 将舱外风险、舱内风险和交叉项合成为 `SAFE / LOW / HIGH / CRITICAL`。
- `AudioAlerter` 支持舱外碰撞和舱内疲劳两路声音报警，并带连续帧确认与冷却机制。
- `main.py` 使用双摄像头多进程采集；两个 demo 使用单路输入，方便单模块调试。

## 项目结构

```text
main.py                     双摄像头主程序
demo_external.py            舱外独立 Demo：检测、测距、TTC、融合、报警
demo_internal.py            舱内独立 Demo：FaceMesh、疲劳/分心、融合、报警、CSV
config.yaml                 全局配置
src/external/               舱外检测、测距、碰撞预警
src/internal/               舱内 FaceMesh、疲劳、注意力、驾驶员状态评估
src/core/risk_fusion.py     多模态风险融合
src/ui/                     可视化和声音报警
tests/                      摄像头与性能辅助脚本
```

## 环境准备

建议使用 Python 3.10 或 3.11。核心依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

舱外 YOLO 模式需要项目根目录下存在 `yolov8n.pt`，本仓库当前已包含该权重文件。CUDA 不是必须项；如果没有 NVIDIA GPU，请把 `config.yaml -> external.device` 或命令行 `--device` 设置为 `cpu`。

可先运行环境检查：

```bash
python test-environment.py
```

## 快速运行

舱外模拟模式不需要摄像头和 YOLO，适合先验证逻辑：

```bash
python demo_external.py --mode sim
```

舱外摄像头模式：

```bash
python demo_external.py --mode camera
python demo_external.py --mode camera --source 2 --device cpu
```

舱外视频模式：

```bash
python demo_external.py --mode video --source E:\data\road.mp4
```

舱内摄像头模式：

```bash
python demo_internal.py
python demo_internal.py --csv logs/internal.csv
```

舱内视频模式：

```bash
python demo_internal.py --mode video --source E:\data\driver.mp4
```

双路主程序：

```bash
python main.py
```

## 常用按键

| 按键 | 作用 |
| --- | --- |
| `q` / `ESC` | 退出 |
| `p` | 暂停 / 继续 |
| `s` | 截图到 `screenshots/` |
| `r` | 重置跟踪或融合平滑状态 |
| `+` / `-` | 舱外 Demo 中调整 YOLO 置信度 |

## 配置入口

主要改 `config.yaml`：

- `system.camera_id_ext` / `system.camera_id_int`：舱外、舱内摄像头编号。
- `system.frame_width` / `system.frame_height`：采集分辨率。
- `external.model_path`、`external.device`、`external.conf_threshold`：YOLO 模型、设备和置信度。
- `external.focal_length`、`external.known_width`、`external.class_widths`：单目测距参数。
- `external.ttc_threshold`、`external.safe_distance_time`：碰撞预警阈值。
- `internal.ear_threshold`、`internal.mar_threshold`：闭眼和哈欠基础阈值。
- `internal.perclos_threshold`、`internal.blink_freq_high_threshold`：PERCLOS 和眨眼频率阈值。
- `internal.distraction_yaw_threshold_deg`、`internal.nod_pitch_threshold_deg`：分心和点头阈值。
- `fusion.level_thresholds`、`fusion.w_ext`、`fusion.w_int`、`fusion.w_cross`：融合权重与风险分级阈值。

## 调参建议

1. 先跑 `demo_external.py --mode sim`，确认测距、TTC、融合和报警链路正常。
2. 再跑 `demo_external.py --mode camera --device cpu`，根据实际画面校准 `focal_length` 和置信度。
3. 跑 `demo_internal.py --csv logs/internal.csv`，分别录制正常、闭眼、哈欠、转头、低头片段。
4. 根据 CSV 里 EAR/MAR/yaw/pitch/PERCLOS 的实际范围调整 `config.yaml`。
5. 最后运行 `main.py` 做双路联调。

## 常见问题

- 摄像头打不开：先用 `tests/check_camera.py` 或 `tests/test_camera_fps.py` 枚举摄像头编号，再修改 `config.yaml`。
- YOLO 初始化失败：检查 `ultralytics` 是否安装、`yolov8n.pt` 是否存在；无 GPU 时使用 `--device cpu`。
- 舱内没有人脸：检查光照、摄像头角度和 FaceMesh 依赖；可先关闭网格绘制提高性能。
- 声音报警不可用：`pygame.mixer` 初始化失败时会自动禁用声音，画面风险提示仍可用。
- 视频 TTC 异常：默认按视频原始 FPS 限速处理；只有做性能压测时才使用 `--no-pacing`。

## 团队成员

主创与核心开发：杨涵  
协同开发者与贡献者：何嘉乐、郑皓宇、张子凡、陈永凌
