# IDMS：基于小型 RSSM 世界模型的最小风险决策系统

IDMS 面向驾驶安全研究，组合舱外目标检测、单目测距、TTC 估计、驾驶员状态评估、多模态风险融合和最小风险动作规划。当前决策层包含小型 DreamerV3-style 离散 RSSM 世界模型，并始终保留解析运动学预测与独立安全屏障。

> 当前代码是研究与辅助决策原型，不是经过功能安全认证的车辆控制器。默认 Demo 只输出建议动作，不会控制车辆。

## 1. 推荐入口：单路舱外摄像头验证 RSSM 推理链路

`demo_rssm_external.py` 默认只打开 `config.yaml -> system.camera_id_ext` 指定的舱外摄像头，不创建舱内摄像头或 FaceMesh。

```powershell
conda activate idms_env
python demo_rssm_external.py
```

指定摄像头编号或强制使用 CPU：

```powershell
python demo_rssm_external.py --mode camera --source 2
python demo_rssm_external.py --mode camera --source 2 --device cpu
```

用视频验证同一条感知与 RSSM 推理链路：

```powershell
python demo_rssm_external.py --mode video --source "E:\data\road.mp4"
python demo_rssm_external.py --mode video --source "E:\data\road.mp4" --no-display --no-audio --no-pacing --max-frames 300
```

Demo 默认严格要求 RSSM 检查点就绪。检查点、SHA-256、训练步数或特征契约不匹配时会退出，不会把运动学回退误报成“RSSM 推理链验证成功”。仅排查感知链路时可显式允许回退：

```powershell
python demo_rssm_external.py --allow-kinematic-fallback
```

该模式只表示诊断运行完成：`diagnostic_passed` 可以为 `true`，但 `rssm_validation_passed` 和兼容字段 `validation_passed` 始终为 `false`，不会把纯运动学结果标成 RSSM 验证成功。

### 如何确认 RSSM 推理链路确实生效

启动终端应显示：

```text
planner         : ready
```

每个新鲜有效帧应显示：

```text
prediction_source: rssm_hybrid
```

退出时 `logs/rssm_external_summary.json` 的关键字段类似：

```json
{
  "planner_status": "ready",
  "valid_frames": 12,
  "rssm_frames": 12,
  "fallback_valid_frames": 0,
  "max_contiguous_posterior_updates": 5,
  "max_source_time_span_sec": 1.1,
  "validation_mode": "strict_rssm_chain",
  "run_completed": true,
  "diagnostic_passed": null,
  "rssm_validation_passed": true,
  "validation_passed": true,
  "control_mode": "open_loop_advisory"
}
```

一次严格验证成功必须同时满足：进程退出码为 0、`valid_frames > 0`、`rssm_frames > 0`、`fallback_valid_frames == 0`、连续后验更新不少于 4 次，且新鲜源时间跨度不少于 0.75 秒。此时 `rssm_validation_passed` 和兼容字段 `validation_passed` 才会为 `true`。阈值可通过 `--min-posterior-updates` 和 `--min-source-span-sec` 调整，但降低阈值只适合接口诊断。

摄像头断流事件会按设计临时显示 `kinematic`，并单独计入 `sensor_loss_events`，不会伪装成有效空道路。`rssm_observe_count` 可能小于视频帧数，因为后验按照 `decision.rssm.dt_sec` 节拍更新。

这些字段只证明 RSSM 后验与混合规划推理链持续运行，不证明模型风险已经校准、预测准确或达到实车安全要求。模型质量仍需独立回放集、未来观测误差、危险召回率、误刹率和风险校准验收。

每帧建议动作写入 `logs/rssm_external_decisions.csv`。这些记录是推荐结果，不是执行器回执，不能直接作为动作条件训练数据。

## 2. “舱内失效”的准确含义

本 Demo 将舱内感知通道明确标记为：

```text
CABIN: UNAVAILABLE / INT: N/A
```

它表示舱内摄像头和驾驶员感知链路没有接入，**不表示驾驶员状态正常**。当前 schema v3 的 RSSM 只有 13 维观测，没有 `internal_perception_valid` 掩码。为了继续使用现有检查点，舱内风险、疲劳和注意力特征使用零值占位，同时 HUD、终端和汇总文件始终标注 `UNAVAILABLE`。

本模式不会临时修改融合权重，因为检查点锁定了融合特征契约。舱外危险仍由 `ext_score`、距离、TTC、运动学基线和 `SafetyShield` 独立约束。

如果以后要让世界模型区分“舱内健康且风险为零”和“舱内传感器失效”，必须增加可用性特征、升级 checkpoint schema 并重新训练，不能只修改显示文字或向现有模型伪造输入。

## 3. 系统流程

```text
舱外摄像头 / 视频 / CARLA RGB sensor
              │
              ▼
    YOLOv8 → 单目测距 → TTC/车道相关度
              │
              ▼
       RiskFusionEngine（舱内 N/A）
              │
              ▼
        13维特征 WorldState
          ┌───┴─────────────┐
          ▼                 ▼
  运动学风险预测       RSSM 随机想象 + CVaR
          └───┬─────────────┘
              ▼
  保守合并 → SafetyShield → 最低风险建议动作
```

候选动作固定为 `KEEP`、`SLOW_DOWN`、`BRAKE`、`EMERGENCY_BRAKE`。默认 `rssm_can_deescalate: false`，学习模型不能把动作降低到解析运动学方案以下。

## 4. 开放环安全边界

单摄 Demo 固定传入：

```python
applied_action = None
```

因此它可以验证观测后验、候选动作想象、CVaR 风险排序和混合规划，但不能证明“建议动作已经改变了车辆状态”。画面中的动作是 `OPEN LOOP / ADVISORY ONLY`。

绝不能把上一帧 `DecisionResult.action` 直接回灌为已执行动作。只有执行器或 CARLA 控制器确认实际施加后，才能在下一次观测中把该回执作为 `applied_action` 传入。

舱外有效空画面与摄像头断流具有不同语义：

- 新鲜帧且零检测：`external_perception_valid=True`，道路已观察但没有目标。
- 摄像头超时或断流：`external_perception_valid=False`，规划器保留最近危险状态并进入安全回退，不能把空列表解释成安全道路。

实时摄像头默认由后台线程持续采集，只向推理线程交付最新一帧，并尝试把 OpenCV 缓冲设为 1；帧的 `sequence_timestamp` 取采集完成时的单调时钟，而不是 YOLO 推理结束时间。部分相机后端可能忽略缓冲设置，因此精确时延标定仍应使用带硬件时间戳的采集链路或 CARLA 仿真时间。

## 5. 环境安装

推荐 Python 3.10 或 3.11。当前已验证 Conda 环境名为 `idms_env`。

```powershell
conda activate idms_env
python -m pip install -r requirements.txt
python test-environment.py
```

舱外 YOLO 使用 `yolov8n.pt`。没有 NVIDIA GPU 时，将 `external.device` 设为 `cpu` 或传入 `--device cpu`。YOLO 与 RSSM 设备相互独立；RSSM 可通过 `--rssm-device cpu|cuda:0` 覆盖。

## 6. RSSM 检查点

训练：

```powershell
python train_rssm.py --device auto --steps 2000 --batch 32 --seq 32 --output checkpoints/idms_rssm.pt
```

训练结束会打印 SHA-256。将其写入：

```yaml
decision:
  predictor: "hybrid"
  rssm:
    checkpoint: "checkpoints/idms_rssm.pt"
    require_checkpoint_sha256: true
    checkpoint_sha256: "<训练输出的 SHA-256>"
```

运行时还会验证 schema、网络容量、至少 1000 个优化步骤、四类动作覆盖、unknown-action 契约、融合特征契约，以及同一字节快照的文件大小和 SHA-256。完整说明见 [RSSM_GUIDE.md](RSSM_GUIDE.md)。

## 7. 其他入口

双路摄像头完整系统：

```powershell
python main.py
```

只验证舱外感知，不验证 RSSM：

```powershell
python demo_external.py --mode sim
python demo_external.py --mode camera --source 2
python demo_external.py --mode video --source "E:\data\road.mp4"
```

只验证舱内感知：

```powershell
python demo_internal.py
python demo_internal.py --mode video --source "E:\data\driver.mp4"
```

旧视频 MRM 入口仍保留：

```powershell
python demo_mrm_video.py --source "E:\data\road.mp4"
```

新项目应优先使用 `demo_rssm_external.py`，因为它具有严格 RSSM 推理链判据、显式舱内 N/A、统一源时间和验证汇总。

## 8. CARLA 接入边界

仓库尚未把 CARLA Python API 加入必装依赖，也没有声称已完成 CARLA 客户端、地图、车辆和传感器生成。当前已预留：

- `src.integration.external_source.FramePacket`
- `src.integration.external_source.ExternalFrameSource`
- `src.integration.carla_source.CarlaSensorSource`
- `demo_rssm_external.ExternalOnlyPipeline`

`CarlaSensorSource` 接收一个已由 CARLA 创建并挂载的 RGB sensor actor，模块本身不会在导入时依赖 `carla`：

```python
from src.integration.carla_source import CarlaSensorSource

# rgb_sensor 由 CARLA client/world 创建并 attach_to ego vehicle
source = CarlaSensorSource(
    rgb_sensor,
    timeout_sec=0.5,
    episode_id="Town04-run-001",
    owns_sensor=False,
)

packet = source.read()
# packet.frame: OpenCV BGR
# packet.sequence_timestamp: carla.Image.timestamp
# packet.valid=False: sensor timeout，不是空道路
```

适配器会将 CARLA BGRA 复制为 OpenCV BGR，使用容量为 1 的队列丢弃旧帧，拒绝超过 `timeout_sec` 的陈旧排队帧，并保留 frame、仿真时间和 episode reset 语义。每个 episode 的首个交付包（有效帧或 timeout 包）仅携带一次 `reset=True`，不会反复清空规划器。后续接入应遵循：

1. 使用 `carla.Image.timestamp` 作为 `sequence_timestamp`，不要使用回调完成时的宿主机时间。
2. 固定仿真步长，并让若干 tick 精确组成 RSSM 的 `dt_sec=0.25`。
3. world reload、回放 seek、重连或 episode 切换时调用 `begin_episode()` 和 `mrm_planner.reset()`。
4. 影子模式继续传 `applied_action=None`。
5. 闭环模式只把“上一观测到当前观测期间实际执行并确认的控制”作为 `applied_action`。
6. CARLA 服务端与 Python egg/wheel 版本必须一致；CARLA 依赖应作为可选环境单独安装。

建议先完成 CARLA 影子回放，再实现制动控制映射和回执，最后才进入闭环评估。

## 9. 关键配置

- `system.camera_id_ext`：默认单路 Demo 摄像头编号。
- `external.model_path/device/conf_threshold/imgsz`：YOLO。
- `external.focal_length/known_width/class_widths`：单目测距。
- `external.ttc_threshold/safe_distance_time`：FCW。
- `fusion.*`：风险融合与阈值。
- `decision.predictor`：`kinematic` 或 `hybrid`。
- `decision.rssm.*`：检查点、哈希、设备、采样数和 CVaR。
- `decision.safety_shield.*`：TTC、距离和感知失联安全下限。

单目测距依赖焦距和真实车辆宽度假设。在比较 RSSM 风险前，应先标定 `focal_length` 并验证距离/TTC 误差。

## 10. 项目结构

```text
main.py                              双路摄像头完整系统
demo_rssm_external.py                推荐：单舱外摄像头 RSSM 推理链验证
demo_external.py                     舱外感知 Demo（不含 RSSM）
demo_internal.py                     舱内感知 Demo
demo_mrm_video.py                    旧版视频 MRM Demo
train_rssm.py                        小型 RSSM 合成预训练
config.yaml                          全局配置
RSSM_GUIDE.md                        世界模型与动作回执指南
src/core/rssm_world_model.py         离散 RSSM、CVaR 想象、checkpoint
src/core/mrm_planner.py              候选动作、混合规划、安全屏障
src/integration/external_source.py   统一 FramePacket/OpenCV 帧源
src/integration/carla_source.py      CARLA RGB sensor 队列适配器
tests/                               合约与安全回归测试
```

## 11. 测试

```powershell
python -m unittest discover -v -s tests -p "test_external_rssm_demo.py"
python -m unittest discover -v -s tests -p "test_rssm_world_model.py"
python -m unittest discover -v -s tests -p "test_rssm_training.py"
python -m unittest discover -v -s tests -p "test_mrm_safety.py"
python test_mrm_planner.py
python test_risk_fusion.py
python test_driver_state.py
```

## 12. 常见问题

- `planner: fallback (...)`：检查 checkpoint 路径、SHA-256、训练步数和融合特征契约；严格 Demo 会返回非零退出码。
- 摄像头打不开：运行 `tests/check_camera.py`，确认编号后传 `--source N`。
- `No module named yaml`：在 `idms_env` 中运行 `python -m pip install -r requirements.txt`。
- CUDA 初始化失败：先传 `--device cpu`；RSSM 可继续保持 `--rssm-device cpu`。
- 视频 TTC 异常：seek/loop 后必须 reset。新 Demo 使用视频 PTS 或 FPS 回退时间，不使用推理耗时计算 TTC。
- `rssm_observe_count` 增长较慢：后验按 `dt_sec` 更新，不要求每个显示帧都更新一次；严格模式默认至少需要 4 次连续更新和 0.75 秒源时间。
- `rssm_frames=0`：本次运行没有真正使用世界模型，不能算推理链验证成功。
- `validation_passed=false`：先看 `validation_mode`。严格模式下查看 `validation_errors`；常见原因是帧数太少、视频时间跨度不足、反复 re-anchor，或有效帧发生 RSSM 回退。`fallback_diagnostic` 模式下该字段按设计始终为 `false`。

## 13. 团队成员

主创与核心开发：杨涵  
协同开发与贡献：何嘉乐、郑皓宇、张子凡、陈永凌
