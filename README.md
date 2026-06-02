# IDMS 智能驾驶员监控系统 (Intelligent Driver Monitoring System)

本项目是一个面向交通科学与行车安全领域的智能驾驶辅助系统（ADAS & DMS）。通过采用双目摄像头架构，系统能够同步监测外部道路状况与内部驾驶员行为，利用计算机视觉与深度学习算法，实时提供多维度的危险预警。

## 🌟 核心功能

### 1. 舱外环境感知 (External Monitoring)
* **实时车辆与目标检测**：集成 YOLOv8 (默认使用 Nano 版本 `yolov8n.pt`) 进行高帧率的前方车辆识别。
* **距离估算与碰撞预警**：基于单目视觉与预设的车辆类别宽度（如摩托车、公交车、卡车）进行动态距离估算。通过计算 TTC (Time to Collision) 结合安全跟车时间，提供前向碰撞红色预警。
* **车道相关性分析**：评估目标车辆在画面中的水平位置，并支持硬切或软衰减模式，精准筛选自车道内的威胁目标。

### 2. 舱内驾驶员监控 (Internal Monitoring)
* **面部特征点追踪**：利用 MediaPipe FaceMesh 提取高精度面部网格，支持可选的 UI 网格绘制。
* **疲劳驾驶监测**：通过计算眼睛纵横比 (EAR) 识别闭眼时长，计算嘴部纵横比 (MAR) 识别打哈欠行为，并通过 EMA 平滑系数过滤瞬时误差。
* **分心与异常姿态识别**：利用 SolvePnP 算法进行头部姿态 (Head Pose) 估计，通过偏航角 (Yaw) 判定视线偏移与分心，通过俯仰角 (Pitch) 判定低头或打瞌睡。

### 3. 系统架构与工程特性
* **多进程高并发**：系统采用 Python `multiprocessing` 构建生产者-消费者模型，双路摄像头独立进程读取，主进程负责非阻塞式 AI 推理和画面拼接，有效打破 GIL 限制并保障核心业务逻辑的高 FPS 运行。
* **视听双重警报**：内置基于 Pygame 的音频警报系统，结合画面红色闪烁提示，并在连续多帧检测到危险时触发报警，配备冷却时间机制防止声音轰炸。
* **高度可配置化**：系统的所有核心参数（包括摄像头 ID、算法阈值、焦距常量、平滑系数等）均解耦至 `config.yaml` 文件中，方便实车测试与快速调参。

## 🛠️ 环境依赖
项目基于python 3.11和torch=2.10.0+cu130进行开发, 需要有英伟达独立显卡硬件支持
项目核心依赖以下 Python 库：

* `numpy < 2.0`
* `opencv-python == 4.9.0.80`
* `mediapipe >= 0.10.0`
* `ultralytics >= 8.0.0`
* 其他工具支持：`scipy`, `pygame`, `imutils`
 
## 🚀 快速开始

1. **克隆项目并安装依赖**
   请确保你的 Python 和 torch 环境符合要求，然后执行依赖安装：
   pip install -r requirements.txt
   [注意：请务必锁定 OpenCV 的版本以防版本冲突]
 准备模型文件

2. **请将 YOLOv8 的权重文件 yolov8n.pt 放置在项目根目录下。**

3. **硬件与参数配置**

   打开全局配置文件 config.yaml，根据实际硬件情况修改摄像头 ID：
   system:
     camera_id_ext: 2      # 舱外摄像头 ID
     camera_id_int: 0      # 舱内摄像头 ID
4. **运行系统**

   external的demo部分：
   
     模式 1: 摄像头实时测试
    python demo_external.py --mode camera

     模式 2: 用视频文件测试 (推荐 BDD100K 片段)
    python demo_external.py --mode video --source E:\数据集\数据集\night-clear-1.mp4

     模式 3: 纯模拟测试 (无需任何硬件/模型，验证逻辑正确性)
    python demo_external.py --mode sim

   internal的demo部分：
    python demo_internal.py

   主程序部分（仍在优化中）：
    python main.py
  
   运行时，系统会在终端输出实时 FPS、目标检测数量与面孔识别状态。选中监控窗口并按 q 键即可安全释放资源并退出系统。



  ## 👥团队成员
  主创与核心开发：杨涵 (yanghan0316)
  协同开发者与贡献者：何嘉乐、郑皓宇、张子凡、陈永凌
