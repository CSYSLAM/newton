# MuJoCo-VBD 联合求解器设计与实施状态 v6

## 1. 目标与非目标

`SolverMuJoCoVBD` 是新增求解器。它必须在不修改既有 MuJoCo、VBD、
MJVBD 内部实现的前提下，组合其已验证的低层数值核，支持：

- 动态 reduced-coordinate articulation；
- caller-FK 驱动的运动学 articulation；
- VBD-owned 自由刚体、布料和软体；
- articulated link ↔ free rigid / cloth / soft，以及 free rigid ↔ cloth /
  soft 的统一接触与摩擦；
- DAT 约束下的无穿透候选提交。

两个产品场景是：

1. `example_vbd_dexforce_throw_rigid_into_bag.py`：手必须通过真实接触和
   摩擦抓起每个物体，逐件投进软 bag；禁止写入、恢复或跟随式搬运物体
   pose/velocity。
2. `example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py`：同一件 T-shirt
   完成两次折叠，且衣物、手、桌面和自碰撞均处于同一个 VBD/DAT 接触流程。

不把以下内容视为完成：

- VBD 独立修改每个 robot link 的 6-D pose，之后再解一次 IK；
- MJVBD 式的单向 external-FK-to-cloth 流程；
- ADMM；
- 只检查首帧、有限值或脚本发出了“闭手”命令。

## 2. 最终算法：唯一的动态 articulation 主路径

动态 articulation 的一个 substep 必须遵循以下顺序。这里的 `q` 是 MuJoCo
reduced coordinates，`X` 是 VBD-owned free rigid endpoint，`x` 是 VBD-owned
particle endpoint。

```text
accepted state (q_n, X_n, x_n)
    |
    v
MuJoCo smooth predictor
    q_n, qd_n, controls -> q_hat, qd_hat, M(q_hat), FK(q_hat), J(q_hat)
    |
    v
initialize one uncommitted VBD trial
    articulated links = FK(q_hat) contact boundaries
    free bodies/particles = VBD inertial candidates
    |
    v
for a small number of coupled sweeps:
    1. Build/refit one canonical contact graph at (q, X, x).
    2. Evaluate one normal/friction contact law for all owners.
    3. VBD produces candidate (Delta_X, Delta_x).
    4. DAT finds a common feasible alpha for each swept contact island.
    5. Commit (X, x) by that alpha; update/refit contacts.
    6. Collect the current articulated-link contact gradient/Hessian.
    7. Pull it back through J(q)^T and solve the q block:

           (M(q_hat) / dt^2 + J^T K J + limit terms) Delta_q
               = -J^T g + smooth-centering terms

    8. DAT includes the articulated-link swept motion in the same island
       alpha. Commit with MuJoCo qpos retraction:

           q <- retract_mujoco(q, alpha * Delta_q)

       Then FK(q), refit contacts, and continue.
    |
    v
post-commit physical-gap assertion and diagnostics
    |
    v
state_{n+1} = (q, X, x), with endpoint velocities and one committed
contact/friction history update
```

### 2.1 为什么不能在最后单独 IK

错误路径是：

```text
MuJoCo q_hat -> VBD independently changes link 6-D poses -> solve a new IK
```

它会丢失 `M(q)`、joint limit、关节速度、接触反力及多个接触点之间的
reduced-coordinate 耦合；一个 link pose 也未必对应唯一的关节解。

正确路径是实时的 `J^T` 回传与 MuJoCo qpos retraction。MuJoCo 不在最后
“重解一次 IK”；下一 substep 从已校正的 `q_{n+1}` 继续 smooth prediction。

### 2.2 运动学 articulation fast path

运动学 robot 不需要、也不应求接触反力导致的 `Delta_q`。它仍必须参与同一
collision/contact/DAT 图，但作为无限质量边界。

其正确状态所有权是：

```text
state_in  : previous accepted robot FK + VBD state
state_out : caller supplied target q/FK for this substep
trial     : state_in, then inject target articulated boundary
```

因此 DAT 能看见 `FK(q_n) -> FK(q_target)` 的完整扫掠。把 target FK 预先写入
`state_in` 是错误的：这会让 DAT 丢失手在本 substep 的运动，并可能从已经
穿透 cloth 的起点开始。

## 3. DAT 的正确位置与契约

DAT 是 VBD candidate-to-commit 的一部分，不是 VBD 结束后才执行的 endpoint
检查，也不是失败后整步递归二分的主算法。

每次 VBD/q candidate 都必须：

1. 从同一 accepted island state 产生候选位移；
2. 覆盖 articulation link、free rigid、particle 的起点到候选终点扫掠；
3. 计算 island 共享 `alpha in [0, 1]`；
4. 用相同 `alpha` 提交岛内全部 owner；
5. 仅在整个 coupled sweep 收敛后更新摩擦 anchor / normal multiplier。

V1 可先用“所有 active owners 共用一个 `alpha`”作为保守 fallback；这比逐个
block 独立截断正确，但会保守。后续按 swept contact graph 的 connected component
分 island 以恢复并行性。

shape margin 是接触生成和预接触稳定性的 buffer；DAT 的物理无穿透判断应以
真实 surface separation 为准，particle radius 仍属于真实几何。margin 不能被
误当作必须保持的物理净空。

post-commit gap check 的角色是 assertion/diagnostic：它发现 capacity overflow、
candidate 漏检或数值错误时回滚并显式报错，不能代替 candidate-level DAT。

## 4. 接触、摩擦与状态所有权

| 域 | 唯一状态 owner | 更新方式 |
|---|---|---|
| articulation q/qd | MuJoCo q-block | smooth predictor + `J^T` pullback + qpos retraction |
| articulated link pose | FK(q) | 不作为独立 6-D VBD unknown |
| free rigid X | VBD 6-D block | VBD candidate + DAT commit |
| cloth/soft x | VBD particle block | VBD candidate + DAT commit |
| contact geometry | canonical global buffer | 每次 accepted/candidate geometry 后重建或 refit |
| normal multiplier / friction anchor | unified coupled layer | 每个完整 accepted sweep 一次提交 |

normal、damping、friction 梯度和 Hessian 必须来自同一 canonical contact law。
q block 用 `J^T g` 和 `J^T K J` 读取 articulated endpoint 项；VBD free-body/
particle block 读取其对应项。不得让 MuJoCo native contacts 和 VBD contact law
对同一 pair 各施加一次。

## 5. 当前代码状态（截至本版本）

### 5.1 已实现并有证据

- 新目录 `newton/_src/solvers/mujoco_vbd/` 与公共
  `newton.solvers.SolverMuJoCoVBD` 已创建；不修改现有 VBD、MuJoCo、MJVBD
  源文件。
- `MuJoCoVBDOwnershipPartition` 用压缩 view 管理 owner；articulation link
  在 VBD view 中为零逆质量接触边界，free rigid/particles 为 VBD owner。
- dynamic path 已实现 smooth predictor、canonical rigid/soft contact
  Jacobian、`J^T g / J^T K J` q block、MuJoCo qpos retraction 和 FK。
- 动态 block 调度已改为 `q_hat/FK -> VBD iteration -> q pullback`，而不是
  “先 q correction 再 VBD”。单元测试记录了 VBD iteration 必先于 q pullback。
- `articulation_mode="kinematic"` 已实现为不初始化 MuJoCo smooth/q block 的
  boundary fast path；dynamic 默认仍是完整 q path。
- `VBDDATProjector` 已加入：它在每次 VBD iteration 后保存起点、对 VBD
  candidate 做共享-alpha 的保守截断、并将截断状态回写局部 VBD view。
- physical gap verifier 已改为忽略 solver shape margin，只检验真实 rigid
  surface separation 和 particle radius。
- 联合动态单元测试、endpoint gap test、swept midpoint pass-through test 已在
  CUDA 环境通过。
- bag demo 的 720-frame null-viewer 运行曾以零退出完成；其现有 `test_final()`
  检查每件物体：脚本释放、实际 hand-object contact、接触后真实上抬、bag
  deformation/recovery、以及最终落在 bag 内。

### 5.2 尚未完成，不能宣称已交付

1. **DAT 只完成 VBD-owned candidate 的局部投影。** articulation q candidate
   尚未使用同一 swept island `alpha`；当前 q 后 endpoint failure 仍会进入旧的
   whole-step rollback/bisection fallback。
2. **运动学 target contract 未接通。** 两个 demo 目前将插值后的 FK 预写入
   `state_in`。这使 cloth demo 的 DAT 看不见手的扫掠；它是当前
   `DAT exhausted` 报错的主要结构性原因。
3. **cloth demo 尚未完成验收。** 180-frame reproduction 会在
   `MuJoCo-VBD DAT exhausted ...` 失败；当前 `test_final()` 只检查 finite，
   尚未证明两次折叠成功。
4. **bag demo 当前使用 kinematic fast path。** 它证明真实接触抓取，而不是
   carry transform；但 dynamic target-actuator bag 模式仍需实现和单独验收。
5. 尚无与现有 demo 的 steady-state wall-time / contact quality 对比，不能声称
   性能或效果优于现有 MJVBD。
6. 当前 DAT 是全 owner 共享-alpha fallback，尚未实现真实 swept contact island
   分组、primitive-specific TOI 和旋转保守界。

## 6. 当前已知问题：叠衣服 DAT exhausted

复现命令：

```bash
uv run --extra examples python -u \
  newton/examples/cloth/example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py \
  --viewer null --device cuda:0 --num-frames 180 --house-visual-usd '' \
  --no-graph-capture
```

当前结果：`MuJoCo-VBD DAT exhausted its substep budget before reaching a
penetration-free endpoint`。

根因不是“应该放宽 DAT”或“恢复 MJVBD”。cloth demo 在每个 substep 先将新的
kinematic joint target/FK 写进 `state_0`，然后才调用 solver。solver 因而把新的
手位姿当作 accepted 起点，DAT 不能计算该手从上一 FK 到新 FK 的 sweep；当手/
cloth 接触持续时，局部 VBD candidate 的安全起点可能已经无效，最终触发旧的
whole-step fallback 并耗尽预算。

## 7. 下一步实施顺序与完成条件

### Phase A — 修复运动学 target ownership（当前下一步）

1. 为 `SolverMuJoCoVBD.step()` 的 kinematic 模式建立明确 target 输入：
   `state_in` 保持上一 accepted state；调用者把本 substep target q/FK 写入
   `state_out` 或专用 target scratch。
2. solver 在 workspace 的 `accepted` 保留旧 robot/VBD state，在 `trial` 注入
   target articulated q/FK；DAT 对旧 FK 到 target FK、free-body、particle
   同时求 alpha。
3. cloth/bag demo 改为不再写 `state_0` 的 target FK；只构造 `state_1` target。
4. 新增回归：快速运动 kinematic shape 穿越静态 cloth/particle 的候选被 DAT
   截断，且不再触发 whole-step exhaustion。

完成条件：上述 180-frame cloth repro 不抛异常；任何 accepted substep 都经
candidate-level DAT 提交。

### Phase B — q candidate DAT 合并

1. `ArticulationCorrector` 暴露未提交的 `Delta_q` 与 q retract candidate。
2. 将 link sweep 纳入与 VBD candidate 相同的 global-alpha fallback。
3. 同一个 alpha 同时缩放 q、free rigid、particle candidate；只在 safe commit
后写入 workspace/local views。
4. 删除 normal path 上的 `_advance_adaptive()` 递归；保留它仅为 overflow/
unsupported-pair 的明确 error diagnostic，或完全替换为 reject status。

完成条件：动态 articulation contact test 中 q、free rigid、particle 三者使用
同一 DAT alpha；无 host-side recursive substep 作为常规控制流。

### Phase C — 质量与任务验收

1. 为 T-shirt 记录初始/第一次折叠/第二次折叠的几何指标：投影面积、两次
   fold crease 的 side exchange、最终厚度和桌面稳定性。`test_final()` 必须
   验证两次折叠，不仅 finite。
2. 为 bag 保留并加强真实 hand-object contact、lift、release、inside-bag
   证据；完成 dynamic target-actuator 版本或明确 kinematic fixture 的接口
   与质量边界。
3. 使用 null viewer warmup 后分别测原 demo 与新 solver：frame time、contact
   count、DAT alpha、overflow、max penetration、grasp success、fold success。
4. 只有在质量门槛不低于基线且 warm steady-state 性能不低于基线时，才能宣称
   “效果、性能优于现有方案”。

## 8. 当前验证命令

```bash
# 动态 ownership/q/VBD 调度
uv run --extra dev -m newton.tests \
  -k test_solver_steps_q_and_vbd_owned_state_without_admm

# DAT endpoint 与 swept pass-through
uv run --extra dev -m newton.tests \
  -k 'test_endpoint_gap_verifier_detects_rigid_penetration|test_swept_probe_rejects_midpoint_rigid_pass_through'

# bag 完整任务
uv run --extra examples python -u \
  newton/examples/vbd/example_vbd_dexforce_throw_rigid_into_bag.py \
  --viewer null --device cuda:0 --num-frames 720 --house-visual-usd ''

# 当前 cloth 失败复现；Phase A 后应转为通过的回归
uv run --extra examples python -u \
  newton/examples/cloth/example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py \
  --viewer null --device cuda:0 --num-frames 180 --house-visual-usd '' \
  --no-graph-capture
```

## 9. 交付判定

求解器只有同时满足以下条件才算完成：

- dynamic 与 kinematic articulation 均遵守各自的 state ownership；
- DAT 在 VBD/q candidate commit 中执行，支持 pair 无 accepted penetration；
- bag 五件物体的真实接触抓取、上抬、投放和入袋全程通过；
- T-shirt 两次折叠的几何验收全程通过；
- 所有相关单元/示例回归通过；
- 相同质量门槛下给出对现有 demo 的 warm steady-state 性能对比，并达到目标。
