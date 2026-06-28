# 舱内驾驶员监测 Demo 使用指南

`demo_internal.py` 用来单独验证 main.py 中的舱内链路。它会完成 FaceMesh、人脸指标计算、疲劳/哈欠/分心/点头判断、驾驶员状态协同评估、风险融合、画面提示、声音报警和 CSV 调参记录。

## 1. 推荐运行顺序

先跑摄像头实时画面：

```bash
python demo_internal.py
```

记录 CSV 做调参：

```bash
python demo_internal.py --csv logs/internal.csv
```

用离线视频复测：

```bash
python demo_internal.py --mode video --source E:\data\driver.mp4
```

如果只想跑批处理，不开窗口：

```bash
python demo_internal.py --no-display --max-frames 300 --csv logs/internal.csv
```

## 2. 画面信息

- 左上角：输入源、运行状态、FPS、人脸是否存在、PERCLOS、眨眼频率。
- 左侧指标：EAR、MAR、疲劳、哈欠、分心、点头、yaw/pitch/roll、PERCLOS、眨眼频率。
- 右上角：融合风险等级和分值。
- `I/F/A`：驾驶员总风险、疲劳可信度、注意力缺失度。
- `DriverState`：驾驶员状态协同评估标签，包含 `single`、`corroborated`、`contradicted`。
- 红色闪烁边框：舱内风险达到危险区间。

## 3. 快捷键

| 按键 | 作用 |
| --- | --- |
| `q` / `ESC` | 退出 |
| `p` | 暂停 / 继续 |
| `s` | 截图 |
| `r` | 重置融合平滑状态 |

## 4. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--config config.yaml` | 指定配置文件 |
| `--source 0` | 摄像头编号或视频路径 |
| `--csv logs/internal.csv` | 导出逐帧指标 |
| `--show-landmarks` | 强制显示 FaceMesh 网格点 |
| `--hide-landmarks` | 强制隐藏 FaceMesh 网格点 |
| `--no-audio` | 禁用声音报警 |
| `--no-display --max-frames 300` | 无窗口处理固定帧数 |
| `--no-pacing` | 视频模式不按原始 FPS 限速，仅用于性能压测 |

## 5. CSV 字段

CSV 会记录以下信息：

- 基础帧信息：`timestamp`、`frame`、`has_face`
- 疲劳指标：`ear`、`mar`、`blink`、`is_drowsy`、`is_yawning`
- 头姿指标：`yaw`、`pitch`、`roll`、`is_distracted`、`is_nodding`
- 长窗口指标：`perclos`、`is_perclos_fatigued`、`blink_freq`、`is_blink_freq_high`
- 驾驶员状态评估：`int_score`、`int_fatigue_score`、`int_attention_score`、`int_confidence_label`
- 融合结果：`fused_score`、`fused_level`、`fused_text`

这些字段可以直接用于画图、阈值标定和论文实验记录。

## 6. 指标含义

### EAR 闭眼

EAR 越低，眼睛越接近闭合。核心参数：

- `internal.ear_threshold`
- `internal.drowsy_duration_sec`
- `internal.blink_max_sec`

建议先录一段正常睁眼和闭眼视频，选择两者中间的 EAR 值作为阈值。

### MAR 哈欠

MAR 越高，嘴张得越大。核心参数：

- `internal.mar_threshold`
- `internal.yawn_duration_sec`
- `internal.enable_yawn`

戴口罩时嘴部关键点可能不稳定，可临时设置 `enable_yawn: false`。

### PERCLOS

PERCLOS 是滑动窗口内闭眼帧占比，适合捕捉长时间疲劳趋势。核心参数：

- `internal.perclos_window_sec`
- `internal.perclos_threshold`

默认阈值 `0.15` 表示窗口内超过 15% 时间处于闭眼状态。

### 眨眼频率

疲劳时眨眼频率可能升高。核心参数：

- `internal.blink_freq_window_sec`
- `internal.blink_freq_high_threshold`

默认阈值为每分钟 25 次。

### 分心与点头

分心主要看 `abs(yaw)`，低头/点头主要看 `pitch`。核心参数：

- `internal.distraction_yaw_threshold_deg`
- `internal.distraction_duration_sec`
- `internal.nod_pitch_threshold_deg`
- `internal.nod_duration_sec`

注意：不同摄像头安装角度下 pitch 正负号可能相反。第一次调参时请故意低头，看画面上的 pitch 是变大还是变小，再决定阈值方向。

## 7. 调参流程

1. 固定摄像头位置和光照。
2. 运行 `python demo_internal.py --csv logs/internal.csv`。
3. 依次录制正常、自然眨眼、闭眼 2 秒、哈欠、左右转头、低头。
4. 用 CSV 查看 EAR、MAR、yaw、pitch、PERCLOS 和眨眼频率的范围。
5. 先调 EAR/MAR，再调 yaw/pitch，最后调 PERCLOS 和眨眼频率窗口。
6. 参数稳定后再运行 `main.py` 做双路联调。

## 8. 常见问题

- 没有人脸：检查光照、相机角度、脸部是否过小或过侧。
- FPS 低：使用 `--hide-landmarks`，降低采集分辨率，或关闭不需要的检测项。
- 哈欠误报：提高 `mar_threshold` 或 `yawn_duration_sec`，口罩场景关闭 `enable_yawn`。
- 分心误报：提高 yaw 阈值或延长 `distraction_duration_sec`。
- 点头方向不对：根据实际 pitch 变化修改 `nod_pitch_threshold_deg` 的正负号。
- PERCLOS 启动阶段不报警：这是设计行为，窗口需要积累一段时间后才开始判定。
