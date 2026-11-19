# Go2 Sim2Real 训练与真机部署方案

本文档记录 `examples/locomotion_unitree` 从当前 SynReal 训练策略走向
Unitree Go2 真机部署的实施方案。目标不是仅在训练仿真器中得到较高奖励，
而是得到一份满足以下条件的策略：

- SynReal 中稳定跟踪速度命令；
- 使用训练资产重建的 MuJoCo 中稳定运行；
- 通过 Unitree 官方 `unitree_mujoco + DDS + C++ controller` 全链路验证；
- 在具有明确安全边界的条件下部署到 Go2 真机。

## 1. 基准与当前状态

当前移植参考 Unitree RL Lab commit：

```text
4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
```

官方 Go2 策略的部署 ABI 为：

- 控制频率：50 Hz（`dt=0.005`，decimation 4）；
- actor observation：45 维；
- action：12 维关节位置增量；
- 关节顺序：FR、FL、RR、RL，每条腿依次 hip、thigh、calf；
- action 映射：`q_target = default_q + 0.25 * action`；
- 真机 PD 参数：`Kp=25`、`Kd=0.5`；
- Go2HV 扭矩-转速曲线：`X1=13.5`、`X2=30.0`、`Y1=20.2`、
  `Y2=23.4`；
- actor 推理使用 Gaussian policy 的均值，网络末端没有 `tanh`。

当前训练是在官方任务基础上为了更快学会移动而调整过的 aggressive
版本，不应再描述为逐项完全等同官方配置。主要差异包括：

| 项目 | 官方当前配置 | 当前本地配置 |
|---|---:|---:|
| 初始前向命令范围 | `[-0.1, 0.1] m/s` | `[-0.3, 0.3] m/s` |
| 站立命令比例 | `0.1` | `0.05` |
| action rate reward | `-0.1` | `-0.05` |
| joint position reward | `-0.7` | `-0.4` |
| feet slide reward | `-0.1` | `-0.05` |
| restitution | `[0.0, 0.15]` | `0.0` |
| base mass randomization | `[-1, +3] kg` | 关闭 |
| friction randomization | 64 buckets | 单个 batch 系数 |

保持质量随机化关闭是当前明确约束。后续只有在 SynReal 提供可靠的质量、
惯量刷新接口并经过验证后，才重新评估该项。

## 2. 当前已经生效的随机化

当前训练中已生效：

- 观测噪声：角速度、重力投影、关节位置、关节速度；
- reset：根节点 XY、yaw 和关节速度；
- 每 interval push：每 5 到 10 秒施加一次 XY 速度扰动，范围
  `[-0.5, 0.5] m/s`；
- command curriculum：已经可以扩展到 `vx=[-1,1] m/s`、
  `vy=[-0.4,0.4] m/s`、`yaw=[-1,1] rad/s`；
- friction：启动时从 `[0.3,1.2]` 抽取一个系数，但 512 个环境共享该值。

最后一项不是完整的摩擦域随机化。一轮 512 环境训练只覆盖一个摩擦系数，
不能替代官方的 per-environment 64 buckets。

## 3. 执行器随机化代码现状

通用 locomotion 环境已经具有以下 buffer、每 episode 采样代码和 action
delay queue：

- `stiffness_scale`；
- `damping_scale`；
- `motor_strength`；
- `joint_zero_offset`；
- `policy_delay_ticks`。

但 `locomotion_unitree` 使用了专门的 Go2HV 扭矩-转速 kernel，因此不能仅修改
YAML。当前缺口如下。

### 3.1 Kp/Kd 随机化

通用 reset 代码会修改 solver 中的 stiffness/damping 数组，但 Unitree kernel
仍使用固定 `Kp=25`、`Kd=0.5` 计算受限扭矩和等效位置目标。这会让“期望的
随机 PD 扭矩”和 solver 最终执行的扭矩不一致。

需要把每个环境、每个关节的 scale 传入专用 kernel，并在正向 PD 计算和等效
目标反解中使用完全相同的 gain：

```text
kp = base_kp[joint] * stiffness_scale[env, joint]
kd = base_kd[joint] * damping_scale[env, joint]
tau_raw = kp * (q_target - q) - kd * dq
```

### 3.2 Motor strength 随机化

`motor_strength` 表示可用电机扭矩相对额定值的比例，用来覆盖电机个体差异、
供电状态和未建模的驱动损失。例如 `0.9` 表示只有额定扭矩的 90%。

当前通用代码虽然修改了 `joint_effort_limit`，但 SynReal adapter 不消费该
limit；专用 kernel 又直接使用固定的 Go2HV `Y1/Y2`，所以该随机化目前没有
物理效果。

修复后的计算应为：

```text
speed_limit = go2hv_torque_speed_limit(dq, X1, X2, Y1, Y2)
limit = motor_strength[env, joint] * speed_limit
tau = clamp(tau_raw, -limit, limit)
q_equivalent = q + (tau + kd * dq) / max(kp, epsilon)
```

### 3.3 零位误差和控制延迟

通用实现已经具有正确的建模要素，但 Unitree override 的 `step()` 和 `_frame()`
目前绕过了它们。需要完成：

- observation 中使用带零位偏差的关节相对位置；
- position target 中使用对应的零位偏差；
- action 在进入 target 映射前通过 `_delayed_action()`；
- reset 时清空该环境的 action queue，避免跨 episode 残留。

零位偏差必须同时保持传感器读数和底层位置控制的一致含义，不能只给
observation 加噪声而不改变实际 PD 行为。

### 3.4 必须补充的测试

在开启执行器随机化前完成以下自动测试：

1. 专用 Warp kernel 与等价 Torch 公式逐元素一致；
2. `motor_strength=0.8/1.0/1.2` 时 torque-speed limit 按比例变化；
3. 非 1 的 Kp/Kd scale 下，反解出的 target 产生预期 PD 扭矩；
4. delay 为 0、1、2 tick 时，action queue 时序精确；
5. joint zero offset 同时影响 observation 和物理目标，符号一致；
6. CUDA graph 捕获前后结果一致；
7. 关闭随机化时与当前 baseline 数值一致，避免破坏已有 checkpoint。

## 4. 分阶段训练计划

随机化不在当前正在运行的进程内动态扩大。每一阶段都从上一阶段选出的
checkpoint 开启新 run，并保存独立的 `task.yaml`。

### Stage 0：完成当前 baseline

- 保持当前配置训练到约 4000 到 5000 iterations；
- 每 500 到 1000 轮做固定命令和命令扫描；
- 选择验证集最好的 checkpoint，而不是默认选择最新 checkpoint；
- 目标是先稳定步态、倒退、横移、转向和零命令站立。

### Stage 1：轻度执行器随机化

从 Stage 0 checkpoint 继续约 2500 到 3000 iterations：

```yaml
stiffness_scale: [0.9, 1.1]
damping_scale: [0.8, 1.2]
motor_strength: [0.9, 1.1]
joint_zero_offset: [-0.02, 0.02]
policy_delay_ticks: [0, 1]
push_velocity: [-0.5, 0.5]
```

原则：一次只引入已通过数值测试的随机量。前 200 到 500 轮重点观察奖励突降、
episode length、速度误差、扭矩饱和率和 action 分布；明显退化时先收窄范围，
而不是继续硬训。

### Stage 2：增强随机化

Stage 1 验收通过后继续约 2500 到 3500 iterations：

```yaml
stiffness_scale: [0.8, 1.2]
damping_scale: [0.7, 1.3]
motor_strength: [0.8, 1.2]
joint_zero_offset: [-0.04, 0.04]
policy_delay_ticks: [0, 2]
push_velocity: [-0.8, 0.8]
```

`push_velocity` 只有在恢复能力达标后才继续扩大到 `[-1.0,1.0] m/s`。观测噪声
暂时保持官方量级，不与执行器随机化同时扩大。

### Stage 3：摩擦覆盖微调

当前 12 GB GPU 上，512 环境 per-env friction 与现有 solver/model layout 组合
成本过高。候选实现按优先级排列：

1. 实现共享 topology 下的分桶材质，512 环境分配 16 到 64 个摩擦 bucket；
2. 若分桶不可行，使用 64 或 128 环境的 per-env friction，从 Stage 2
   checkpoint 微调 1000 到 2000 iterations；
3. `batch` 模式仅用于冒烟和单域训练，不作为最终摩擦鲁棒性证明。

总训练量预计约 10000 到 14000 iterations。`50000` 是 runner 上限，不是必须
跑满的目标。停止条件由跨域验证结果决定。

## 5. 随机化范围必须由真机测量校准

以上范围是首轮保守默认值。准备真机时先运行只读采集程序，不发送运动命令，
测量：

- 静止 IMU gyroscope、quaternion 漂移和噪声分布；
- 关节 q/dq 的静止噪声和不同姿态下的零位误差；
- DDS state 消息间隔、丢包、策略推理耗时和端到端 action latency；
- 不同电量、电机温度下的响应；
- 左右腿或不同关节的系统性偏差。

最终范围采用实测分布的高分位值再留适度 margin。特别是 `0-2 tick` 代表
0 到 40 ms 延迟，如果实测链路远小于该值，不应无依据继续扩大。

## 6. 部署导出必须修复

当前 `examples/locomotion/deployment.py` 按 `--robot go2` 加载的是通用
`examples/locomotion/configs/go2.yaml`，其 `Kp=80/Kd=2`、默认姿态和当前
Unitree checkpoint 不同，不能用于导出本次策略。

需要为 `locomotion_unitree` 增加 checkpoint-aware exporter：

1. 强制读取 checkpoint 同目录的 `task.yaml`；
2. 导出自包含的 `exported/policy.onnx`；
3. 生成官方 C++ controller 可读的 `params/deploy.yaml`；
4. 固化 45D observation 顺序、scale、12D action 顺序、default position、
   `Kp/Kd` 和 50 Hz；
5. 保存 checkpoint、task 和资产 hash；
6. 比较 PyTorch 与 ONNX 对同一批 observation 的输出，误差必须在浮点容差内；
7. 用录制的 `LowState` 比较 Python 和 C++ observation builder 的 45 个值。

导出目录保持官方结构：

```text
policy/
  exported/policy.onnx
  params/deploy.yaml
  manifest.json
```

## 7. 两级 sim2sim 验证

### 7.1 训练资产 MuJoCo

继续使用 `mujoco_eval.py`：它解析训练所用的同一 USD，并校验 body、joint、
inertia、collision primitive、joint limit 和 FK。该验证主要隔离 SynReal 与
MuJoCo 的动力学、约束和接触 solver 差异。

应增加批量矩阵测试：

- 命令：零速、`vx=+/-0.2,+/-0.5,+/-1.0`、横移和转向；
- 摩擦、PD gain、电机强度、零偏和 delay 的边界及组合；
- 多随机种子、至少 100 个 episode；
- 输出 tracking error、跌倒率、扭矩饱和率、足端滑动和 action 分位数。

### 7.2 官方 DDS 全链路 MuJoCo

再使用 Unitree 官方 `unitree_mujoco` 与 `unitree_rl_lab/deploy/robots/go2`
控制器。该层验证：

- ONNX Runtime 推理；
- DDS `LowState/LowCmd`；
- SDK 关节 index 和符号；
- 真实 50 Hz policy thread 与 1 kHz command publish；
- joystick command、Passive/FixStand/Velocity 状态机。

官方 MuJoCo MJCF 与训练 USD 不一定逐项相同，因此两级验证都需要，不能用
官方 DDS sim2sim 替代训练资产一致性检查。

## 8. 真机控制器安全要求

官方示例是有价值的部署骨架，但不是完整的生产安全控制器。现有代码主要只有：

- `Passive -> FixStand -> Velocity` 手柄状态机；
- LowState timeout 回 Passive；
- bad orientation 回 Passive。

真机前至少补充：

- 检测已有 LowCmd publisher 时 fail closed；官方示例当前只报警但继续运行；
- 物理急停，并保证急停不依赖策略线程；
- LowState 时间戳、CRC/有效性和 stale state 检查；
- observation、ONNX output 的 NaN/Inf 检查；
- inference deadline/loop overrun 检查，禁止落后后快速补跑多帧；
- raw action、最终 q target、每周期 target slew rate 硬限幅；
- q/dq、估算 PD torque、温度、电池电压保护；
- Velocity 状态进入时从 FixStand target 平滑过渡；
- joystick command slew limit，进入策略时命令从零开始；
- 所有保护触发时记录原因并切到可预测的低风险状态。

不能通过给现有网络输出临时追加 `tanh` 来实现安全，因为这会改变训练和部署
策略的映射。应使用与训练分布相符的 action guard、物理关节限制和 slew limit，
同时在随机化训练及 sim2sim 中验证这些 guard。

## 9. 建议验收门槛

下面是首版门槛，后续根据实测调整：

- nominal SynReal：100 个 20 秒 episode，成功率至少 99%；
- randomized SynReal：100 个 20 秒 episode，成功率至少 95%；
- MuJoCo 参数矩阵：不跌倒，平均 `vx` 绝对误差小于约 `0.12 m/s`；
- 零命令：连续站立 60 秒，无明显持续漂移或高频抖动；
- `+/-0.5 m/s` 推扰：大多数样本在约 1 秒内恢复稳定跟踪；
- ONNX 与 PyTorch 输出逐元素一致；
- C++ 与 Python observation 逐元素一致；
- 全程无 NaN/Inf、关节越限，扭矩饱和率和 action 极值有明确统计；
- DDS MuJoCo 连续运行至少 5 分钟，通过多方向命令与急停状态切换。

checkpoint 选择采用综合指标，不只看 mean reward。至少同时考虑 episode
success、速度误差、姿态、足滑、扭矩饱和、action 平滑度和跨仿真器表现。

## 10. 真机逐级放开流程

1. **离线回放**：对录制 LowState 运行 observation 和 ONNX，只记录 action，
   不发布 LowCmd。
2. **吊架测试**：四足离地，核对 12 个关节顺序、方向、default pose、PD gain、
   action limit 和急停。
3. **受约束站立**：牵引绳/保护架，零速度命令进入 FixStand，再平滑切换 Velocity。
4. **低速直行**：首次限制到 `0.1-0.2 m/s`，短时运行，检查温度、电流、抖动、
   足滑和状态延迟。
5. **逐级扩展**：通过验收后扩展到 `0.3`、`0.5 m/s`，再测试倒退、横移和转向。
6. **完整范围**：只有低速、多地面和多电量测试通过后才开放训练命令上限。

每一级失败都退回上一 checkpoint 或收窄参数，不在真机上通过反复尝试替代
仿真、全链路和安全测试。

## 11. 推荐实施顺序

- [ ] 当前 baseline 训练到 4000 到 5000，并建立批量评估报表；
- [ ] 修复 Unitree actuator randomization kernel 和 Unitree `step/_frame`；
- [ ] 增加执行器随机化 CUDA/CPU 数值与时序测试；
- [ ] 新增 checkpoint-aware Unitree ONNX/deploy exporter；
- [ ] 增加 PyTorch/ONNX/Python-C++ ABI parity 测试；
- [ ] Stage 1 轻度随机化训练与消融；
- [ ] Stage 2 增强随机化训练；
- [ ] Stage 3 friction bucket/per-env 微调；
- [ ] 训练资产 MuJoCo 参数矩阵验收；
- [ ] 官方 DDS MuJoCo 全链路验收；
- [ ] 加固真机 C++ controller；
- [ ] 按吊架、受约束站立、低速到完整范围逐级部署。

## 12. 上游参考

- Unitree RL Lab：<https://github.com/unitreerobotics/unitree_rl_lab>
- Go2 训练配置：
  <https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py>
- Go2 actuator：
  <https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py>
- Go2 deploy controller：
  <https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/go2>
- Unitree SDK2：<https://github.com/unitreerobotics/unitree_sdk2>
- Unitree MuJoCo：<https://github.com/unitreerobotics/unitree_mujoco>
