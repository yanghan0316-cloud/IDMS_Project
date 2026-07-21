# Tiny DreamerV3-style RSSM 使用指南

本项目的 RSSM 是一个面向“特征级状态”的小型世界模型，用于比较候选最小风险动作；它不处理原始图像，也不直接控制车辆。

## 1. 架构与决策流程

每帧先由感知与融合模块生成 13 维 `WorldState`，包括前车存在、距离、TTC、接近速度、车道相关度、外部/驾驶员/融合风险等。`WorldStateCodec` 将其归一化后送入模型。

动作编码为 3 维：

- `target_decel`：目标减速度，按 `max_decel` 归一化；
- `response_delay_sec`：响应延迟，按 `max_delay` 归一化；
- `action_valid`：该动作是否由执行端确认。

默认小模型由以下部分组成：

```text
13维观测 -> MLP编码器(64)
上一随机状态(8个变量 × 8类) + 动作 -> GRU确定性状态(64)
                                         ├─ prior（纯想象）
观测编码 + 确定性状态 --------------------└─ posterior（状态校正）
确定性状态 + 随机状态 -> 观测重建头 / 风险头 / continue头
```

训练损失包含观测重建、风险、continue，以及分离的 dynamics/representation KL；实现了离散随机状态、straight-through 采样、uniform mixing 和 free nats 等 DreamerV3-style 机制。

在线规划会对 `KEEP`、`SLOW_DOWN`、`BRAKE`、`EMERGENCY_BRAKE` 四个候选动作进行多样本想象，使用最坏尾部的 CVaR 风险，再与运动学预测保守合并，最后按风险、驾驶员状态、舒适性和反应不足惩罚选择最低代价动作。非学习型 `SafetyShield` 始终可以抬高最低制动等级。

## 2. 训练与启用

先安装依赖：

```powershell
pip install -r requirements.txt
```

默认训练使用物理上合理的合成闭环轨迹；动作条件是模拟器确认执行的命令设定值，路面附着造成的实际减速度偏差作为环境动力学扰动：

为匹配当前主程序尚无执行器回执的情况，合成训练会把约 20% 的整段动作历史和 10% 的离散转移遮蔽为 unknown；checkpoint 会记录实际遮蔽比例、每类有效已知动作计数和动作对齐契约。保存前会拒绝 unknown 覆盖不足或任一候选动作没有有效已知样本的训练结果。合成观测中的外部风险、驾驶员风险、交叉项、融合权重和 EMA 与当前在线默认公式对齐，并在 checkpoint 中记录 `fusion_feature_contract`。

```powershell
python train_rssm.py --device auto --steps 2000 --batch 32 --seq 32 --output checkpoints/idms_rssm.pt
```

默认会读取 config.yaml，把 fusion 与 internal 的解析后特征参数写入契约；仅做默认参数隔离测试时可传 --feature-config none。 保存完成后把终端打印的 SHA-256 填入 config.yaml，再启动 hybrid 规划。

CPU 快速冒烟训练可用：

```powershell
python train_rssm.py --device cpu --steps 2 --batch 4 --seq 4 --output checkpoints/rssm_smoke.pt
```

`train_rssm.py` 会自动创建输出目录。默认模型结构固定在该脚本的 `build_model()` 中；修改维度后必须重新训练，运行配置应与 checkpoint 保持一致。

checkpoint 使用 schema v3，并锁定模型维度、归一化尺度、步长、动作字段、四候选动作表和动作对齐契约。修改网络维度、`dt_sec`、减速度或响应延迟后都必须重新训练；旧 checkpoint、容量不足以表示全部候选动作、缺少 unknown-action 元数据或缺少四类有效已知动作覆盖的手工 checkpoint 会自动回退。

融合特征契约现为 v2：除外部风险、融合权重和 EMA 外，还锁定 DriverStateAssessor 的协同、矛盾、单信号封顶、连续阈值及全部疲劳/注意力权重；配置不一致会回退到运动学基线。加载前还会拒绝空文件和超过 128 MiB 的 checkpoint，避免误放大文件耗尽内存。

## 3. `config.yaml` 关键项

推荐保留以下安全配置：

```yaml
decision:
  predictor: "hybrid"
  rssm_can_deescalate: false
  horizons_sec: [0.5, 1.0, 1.5]
  max_horizon_sec: 10.0

  safety_shield:
    enabled: true
    perception_loss_min_rank: 1
    perception_loss_brake_after_sec: 0.5

  rssm:
    enabled: true
    checkpoint: "checkpoints/idms_rssm.pt"
    require_checkpoint_sha256: true
    checkpoint_sha256: ""
    device: "cpu"
    min_training_steps: 1000
    embed_dim: 64
    deter_dim: 64
    stoch_dim: 8
    classes: 8
    hidden_dim: 64
    dt_sec: 0.25
    max_gap_sec: 1.5
    samples: 8
    cvar_alpha: 0.75
    max_decel: 8.0
    max_delay: 0.5
```

主要含义：

- `predictor: hybrid`：同时使用 RSSM 与运动学基线；建议作为默认模式。
- `rssm_can_deescalate: false`：RSSM 不得把动作降到运动学方案以下。
- `samples`：每个候选动作的随机想象条数；越大越稳健，也越耗时。实现将其限制为最多 64 条，并限制总想象为 40 个模型步，避免误配置导致内存或时延失控。
- `cvar_alpha: 0.75`：聚合最坏约 25% 样本的风险。
- `dt_sec`：RSSM 状态更新与想象步长；应接近实际遥测采样周期。
- `max_gap_sec`：时间断流的硬上限。间隔小于 `dt_sec` 时暂不更新并保留时间余量；间隔大于 `1.4 × dt_sec` 时重锚；超过 `max_gap_sec` 或时间倒退时先重置再重锚，避免猜测缺失动作历史。
- `max_decel`、`max_delay`：动作归一化上限，必须覆盖真实执行范围。`remaining_delay_sec` 可以大于单步 `dt_sec`，但不得超过 `max_delay`，并会在连续想象步中逐步扣减。
- `max_horizon_sec`：解析预测的绝对上限；混合模式还要求该 horizon 在 40 个 RSSM 步内完成。

如只需确定性基线，将 `decision.predictor` 设为 `kinematic`。若 checkpoint 缺失、格式/观测 schema、融合特征合约、动作覆盖或归一化容量不匹配，规划器会自动使用运动学结果，`model_status` 会显示 `fallback (...)`。模型一旦产生非有限值或分布计算异常会熔断并保持回退，避免逐帧重复尝试坏模型。即使 RSSM 正常，`hybrid` 也取更高风险、更短距离或更短 TTC 的保守结果；安全盾仍独立生效。

规划器默认要求 checkpoint 至少记录 1000 个优化步骤，且该下限不可通过配置降低；少量步骤只用于接口冒烟，不能被正式混合规划配置加载。训练完成后脚本会打印 SHA-256，必须复制到 checkpoint_sha256，运行时默认先校验哈希再加载。步数与哈希仍不能证明模型达到安全质量，不能替代独立验证集、危险场景召回率/误刹率验收和模型校准。

## 4. 必须遵守的数据与安全边界

`DecisionResult.action` 和当前 CSV 中的 `action` 都只是**推荐动作**，不代表车辆已经执行。当前 `logs/mrm_decisions.csv`、`logs/mrm_video_decisions.csv` 和 `logs/rssm_external_decisions.csv` 没有执行器确认，也没有严格记录“动作导致下一状态”的转移，因此**不可直接作为 RSSM 动作条件训练数据**。否则模型会错误学习“推荐制动已经改变了车辆状态”。这些 CSV 只适合回放、审计和阈值分析。

真实训练样本至少应保存：

```text
episode_id, sequence_timestamp,
obs_t(13字段),
applied_action_t(name, target_decel, response_delay_sec, remaining_delay_sec?, action_changed, valid, actuator_ack),
obs_t+1(13字段), done/is_first
```

时间对齐必须是 `(obs_t, applied_action_t, obs_t+1)`。训练序列使用 `T+1` 个观测和 `T` 个已执行动作，首个观测前的动作标记为 unknown/invalid。拒绝、超时或无法确认的命令应记录为无效动作，不能用规划器建议值补齐。

若主循环快于 `dt_sec`，引擎会保留时间余量以避免长期采样漂移，并监测被节流帧中的执行动作变化。一个模型步内出现动作切换时，下一次状态更新会保守重锚并把该转移标为 unknown，而不会用“最后一个动作”伪造整段历史。

本模块是研究/辅助决策组件，不是经过功能安全认证的车辆控制器。实车接入前应先完成离线回放、影子模式和封闭场地测试，并保留独立的执行器限幅、watchdog、人工接管与紧急制动链路。模型风险值也不应直接解释为经过标定的碰撞概率。

外部感知必须显式区分“有效帧且零检测”与“摄像头失联/帧过期”。`main.py`、`demo_rssm_external.py` 和 `CarlaSensorSource` 都保留 `external_perception_valid` 语义：失联时规划器停止更新 RSSM、保留并按接近速度外推最近一次有效外部状态，动作等级不得低于上次有效决策；无历史危险时至少 SLOW_DOWN，持续 0.5 秒后至少 BRAKE。只有新的有效空观测才能解除旧危险。集成其他传感器时必须保留这一有效性字段，不能用空列表表示失联。

## 5. 接入真实 `applied_action`

`MRMPlanner.plan()` 的 `applied_action` 表示“从上一观测转移到当前观测期间，执行器确认实际施加的动作”，不是本帧刚得到的推荐。第一帧或无法确认时传 `None`；内部会设置 `action_valid=0`。

`response_delay_sec` 只描述这个刚结束的历史转移中的动作响应延迟；它不会自动再次作用于未来预测。若当前时刻仍有尚未走完的执行器延迟，必须另传 `remaining_delay_sec`，规划器才会在未来想象中继续等待。`action_changed` 只用于消除同名动作在历史转移中的歧义，不代表未来仍剩完整响应延迟；两种延迟字段同时存在时，RSSM 后验使用 `response_delay_sec`，未来运动学与 RSSM 想象使用 `remaining_delay_sec`。

接入模式如下（执行器 API 名称需替换为实际实现）：

```python
last_applied_action = None

while running:
    now = time.monotonic()
    fusion_result, vehicle_data, face_data = read_current_observation()

    decision = mrm_planner.plan(
        fusion_result=fusion_result,
        vehicle_data=vehicle_data,
        face_data=face_data,
        timestamp=now,
        applied_action=last_applied_action,  # 上一周期已确认的实际动作
        sequence_timestamp=now,
    )

    receipt = actuator.apply(
        action=decision.action,
        target_decel=decision.target_decel,
    )
    if receipt.accepted and receipt.confirmed:
        last_applied_action = {
            "name": receipt.confirmed_action_name,
            "target_decel": receipt.confirmed_target_decel_mps2,
            "response_delay_sec": receipt.response_delay_sec,
        }
    else:
        last_applied_action = None
```

优先使用执行器回执确认的稳定动作标识、目标减速度设定值与历史响应延迟；若执行器能报告当前剩余延迟，应在每次观测时更新 `remaining_delay_sec`。实测纵向加速度属于状态遥测，不应直接替换动作目录中的命令值；除非真实训练数据和 codec 明确定义了这种语义。`sequence_timestamp` 必须单调递增；视频回放应传视频时间，发生 seek、loop、传感器重连或 episode 切换时调用 `mrm_planner.reset()`。

执行回执中的映射必须显式提供 `target_decel`（或 `decel`）；同一动作持续执行时内部只在 onset 编码一次响应延迟。若可控制执行策略，最好让命令至少保持一个 `dt_sec`，减少频繁重锚。

当前 `main.py` 没有执行器反馈，`demo_rssm_external.py` 也固定传入 `applied_action=None`；这会安全地编码成 unknown，但无法形成真实动作条件历史。接入执行器或 CARLA 闭环时，应维护上例的 `last_applied_action`，并另建执行遥测日志，不要改写推荐动作日志的语义。

## 6. 验证

运行 RSSM 合约测试与规划器回归测试：

```powershell
python -m unittest discover -v -s tests -p "test_external_rssm_demo.py"
python -m unittest discover -v -s tests -p "test_rssm_world_model.py"
python -m unittest discover -v -s tests -p "test_rssm_training.py"
python -m unittest discover -v -s tests -p "test_mrm_safety.py"
python test_mrm_planner.py
```

检查正式 checkpoint 能否加载：

```powershell
python -c "from src.core.rssm_world_model import TinyRSSM; m, metrics = TinyRSSM.load_checkpoint('checkpoints/idms_rssm.pt', 'cpu'); print(m.get_config()); print(metrics)"
```

检查当前配置的运行状态（无需打开摄像头）：

```powershell
python -c "import yaml; from src.core.mrm_planner import MRMPlanner; c=yaml.safe_load(open('config.yaml', encoding='utf-8')); d=dict(c['decision']); d['_runtime_fusion_config']=dict(c.get('fusion', {})); d['_runtime_internal_config']=dict(c.get('internal', {})); p=MRMPlanner(d); print(p.model_status)"
```

正式 checkpoint 存在且兼容时应输出 `ready`；故意改成不存在的路径时应输出 `fallback (...)`，且规划器仍可使用运动学基线。
