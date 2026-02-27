# 舱内算法（MediaPipe FaceMesh）使用与调参指南

这份指南的目标：让你不用懂太多，也能把“闭眼/哈欠/分心/点头”跑起来，并且知道后面该改哪里。

> 你现在代码里所有“需要等拿到 FPS、是否戴口罩/眼镜后再改”的地方，已经用 `TODO(参数待定)` 写在注释里了。

---

## 1. 我已经帮你做了什么（交付物）

舱内模块已经完整实现并可直接集成主程序：

- **FaceMesh** 人脸关键点检测
- **EAR**（闭眼）与 **MAR**（哈欠）计算
- **疲劳状态机**（闭眼持续 / 眨眼事件 / 哈欠持续）
- **PnP 头部姿态**（yaw/pitch/roll）
- **分心/点头状态机**（abs(yaw) / pitch 持续触发）
- `demo_internal.py`：单独运行舱内模块，方便调参；支持导出 CSV

---

## 2. 怎么运行（先跑 Demo，再跑主程序）

### 2.1 安装依赖

在项目根目录：

```bash
pip install -r requirements.txt
```

### 2.2 先跑舱内独立 Demo（强烈建议）

```bash
python demo_internal.py
```

- 按 `q` 退出
- 窗口左下角会显示 FPS

> **拿到 FPS 后**，把 `config.yaml -> internal.fps` 填上，后面所有 “持续多少秒算报警” 都会更好调。

### 2.3 导出调参数据（CSV）

```bash
python demo_internal.py --csv logs/internal.csv
```

CSV 里会记录每帧的：ear/mar、yaw/pitch/roll、以及各类报警状态，后续做论文画图/阈值标定都很方便。

### 2.4 跑主程序（舱内 + 舱外 + UI）

```bash
python main.py
```

如果你本机没有 `yolov8n.pt`，舱外模块会报错；这种情况下你依然可以先用 `demo_internal.py` 完成舱内部分。

---

## 3. 代码结构：你只需要记住这几个文件

### 3.1 `src/internal/face_mesh.py`

对外接口只有一个：

```python
FaceMeshDetector.process(frame_bgr) -> dict
```

返回字段（核心）：

- `ear`, `mar`
- `blink`
- `is_drowsy`, `is_yawning`
- `yaw`, `pitch`, `roll`
- `is_distracted`, `is_nodding`

### 3.2 `src/internal/fatigue_logic.py`

疲劳/哈欠/眨眼的状态机。

你要调的阈值主要来自 `config.yaml -> internal`：
- `ear_threshold`, `mar_threshold`
- `drowsy_duration_sec`, `yawn_duration_sec`（推荐用秒）
- 如果 fps 未知，则用 `consecutive_frames_eye`, `consecutive_frames_mouth`（备用）

### 3.3 `src/internal/attention_logic.py`

分心/点头的状态机。

关键阈值：
- `distraction_yaw_threshold_deg`
- `nod_pitch_threshold_deg`

⚠️ **注意 pitch 的正负号**：不同摄像头/坐标系可能相反。

> 你要做的就是：跑 `demo_internal.py`，故意低头一次，看 `pitch` 是变大还是变小，然后决定阈值符号。

---

## 4. 调参最短方法（按这个做，不会走弯路）

### 4.1 先只调 EAR（闭眼）
1) 跑 demo
2) 正常睁眼看 `ear` 大概范围
3) 故意闭眼 2 秒，看 `ear` 最低到多少
4) 把阈值放在中间：`ear_threshold`

### 4.2 再调 MAR（哈欠）
- 正常说话时 MAR 也会抖，所以哈欠一定要靠 “持续时间” 抑制误报。
- 如果戴口罩导致嘴部关键点不稳：
  - 临时把 `enable_yawn: false`

### 4.3 最后调 yaw/pitch（分心/点头）
- 转头：`abs(yaw)` 超过阈值持续 → 分心
- 低头：`pitch` 过阈值持续 → 点头

---

## 5. 哪些参数等你们确定后再改（都已写 TODO 注释）

在 `config.yaml` 里：

- `internal.fps`  **(最重要)**
- `drowsy_duration_sec`, `yawn_duration_sec`, `blink_max_sec`
- `distraction_yaw_threshold_deg` + `distraction_duration_sec`
- `nod_pitch_threshold_deg` + `nod_duration_sec`
- 口罩场景：`enable_yawn`

---

## 6. 建议让陈永凌做的部分（最适合分出去）

1) **录数据**：正常/眨眼/闭眼/哈欠/左右看/低头，各 1–2 分钟
2) **标定阈值表**：给出不同光照、戴眼镜/口罩下的建议阈值
3) **误报清单**：什么时候误报、如何复现（你据此再加迟滞/调整持续时间）

---

如果你后面把实际 FPS、是否戴口罩/眼镜、以及你们希望的“报警需要持续多少秒”告诉我，我可以把 `config.yaml` 直接给你算成一组更稳的默认值（你就不用盲调）。
