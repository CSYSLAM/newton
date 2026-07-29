# `example_mujoco_franka_vbd_cable_admm_solver.py` 全链路详解

> 本文逐层拆解 Newton 引擎中 `example_mujoco_franka_vbd_cable_admm_solver.py`（以下简称"本例"）所触发的**完整物理耦合链路**：从模型构建、IK 轨迹、碰撞检测，到 `SolverCoupledADMM` 的 ADMM 迭代、刚体‑刚体接触耦合、CUDA Graph 调度。
>
> 每个小节末尾或关键处会用 **通俗解释** 帮助理解。所有文件引用都用相对路径，可直接点击跳转。

---

## 0. 一句话总览

本例演示一台**固定基座 Franka 机械臂**（由 MuJoCo 求解器驱动，跟踪 IK 抓取‑放置轨迹）与一根**柔性/刚体负载链**（由 VBD cable 或 XPBD chain 驱动）的**多世界并行耦合仿真**。机械臂与负载的**跨 Entry 接触**由 [`SolverCoupledADMM`](newton/_src/solvers/coupled/solver_coupled_admm.py#L423) 通过 **ADMM（交替方向乘子法）迭代**协调；负载与地面等同 Entry/全局形状接触则由外部 `Contacts` 过滤后交给对应子求解器。

**通俗解释**：想象两个裁判（两个物理求解器）各自吹自己的哨子、各管一片场地，但它们管的场地有重叠区（接触面）。ADMM 是一个"调解员"：每一轮它让两个裁判各自赛跑一步，然后看接触处两边的运动对不上多少，施加一个修正力再让它们重跑，如此反复几轮直到两边在接触处"达成一致"。

### 0.1 版本和阅读边界

- `example_mujoco_franka_vbd_cable_admm_solver_v1.py` 与本文对应的正式示例拥有相同的 Python AST；`v1` 只增加了中文注释，没有改变运行逻辑。
- 文件名保留了 `vbd_cable`，但当前默认 `--payload-kind=xpbd-chain`；`vbd-cable` 是 A/B 对比模式。
- 本例的 ADMM 跨求解器约束只来自 `ContactPair("mjc", payload)` 启用的 Franka–负载接触。模型中没有跨 Entry joint，也没有 `body_particle_attachment`。
- `ADMM_SOLVER_DEEP_DIVE.md` 分析的是另一个 `example_mujoco_vbd_admm_solver.py`，其 body-particle attachment 和跨求解器关节机制是 ADMM 通用能力，不是本 Franka 示例的耦合来源。

---

## 1. 涉及的核心文件与类

| 角色 | 文件 | 关键符号 |
|------|------|----------|
| 例子本体 | [example_mujoco_franka_vbd_cable_admm_solver.py](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py) / [中文注释版](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver_v1.py) | `Example` 类 |
| 耦合基类 | [solver_coupled.py](newton/_src/solvers/coupled/solver_coupled.py) | `SolverCoupled`、`SolverEntry`、`SolverCoupled.Entry` |
| ADMM 求解器 | [solver_coupled_admm.py](newton/_src/solvers/coupled/solver_coupled_admm.py) | `SolverCoupledADMM`、`Config`、`ContactPair`、`_AdmmBuffers`、各种 `_Admm*Group` |
| ADMM GPU kernel | [admm_utils.py](newton/_src/solvers/coupled/admm_utils.py) | `u_update_quadratic_kernel`、`lambda_update_kernel`、`contact_rr_*`、`attach_rr_*` 等 |
| 碰撞管线 | [collide.py](newton/_src/sim/collide.py) | `CollisionPipeline` |
| 模型 | [model.py](newton/_src/sim/model.py) | `Model.collide`、`Model.shape_contact_pairs` |
| 接触缓冲 | [contacts.py](newton/_src/sim/contacts.py) | `Contacts` |
| 子求解器 | `newton/solvers/...` | `SolverMuJoCo`、`SolverVBD`、`SolverXPBD` |

---

## 2. 模型构建链路（`Example.__init__` → `_emit_template` → `replicate`）

### 2.1 构建顺序

[Example.__init__](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L110) 的构建流程：

1. **template builder**（[L125-130](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L125)）：建一个"单世界"模板，重力 `(0,0,-9.81)`，`rigid_gap=0.005`，并向 builder 注册 MuJoCo/VBD 需要的自定义属性。
2. **`_emit_template`**（[L238](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L238)）：往模板里放 Franka + 负载，并记录各部件的 body/joint/shape 起止区间。
3. **replicate**（[L137](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L137)）：用 `builder.replicate(template, world_count=...)` 把单世界模板复制 N 份，形成多世界模型。
4. **`_expand_world_indices`**（[L385](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L385)）：把模板内记录的"局部 body/joint/shape 下标"展开成跨世界的全局下标（`world * stride + id`）。
5. **地面**（[L139](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L139)）：`replicate` 结束后只添加一个 `world=-1` 的全局地面 `payload_ground_plane`。它与每个局部 world 的 shape 生成接触候选对，无需为每个 world 复制一份。

### 2.2 Franka 子树（`_add_franka`）

[`\_add_franka`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L279) 加载 `fr3_franka_hand.urdf`，`floating=False`（固定基座）。关节目标 PD 增益在 [`_emit_template`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L244-251) 设置：
- 前 7 个关节：`target_ke=900, target_kd=90`，`effort_limit=80`，`armature=0.05`
- 2 个手指关节：`target_ke=1000, target_kd=100`，`effort_limit=1000`
- 还为每个 Franka body 设了 `mujoco:gravcomp=1.0`（重力补偿），让 MuJoCo 抵消机械臂自身重力，减少 IK 跟踪误差。

### 2.3 负载子树（两种模式）

通过 `--payload-kind` 选择：

- **`vbd-cable`**（[`_emit_vbd_cable`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L300)）：用 `builder.add_rod` 建一根可弯曲拉伸的杆（弹性杆/线缆模型），由 `SolverVBD` 模拟。参数：`stretch_stiffness=2e5`、`bend_stiffness=0.08`、半径 `payload_radius`。
- **`xpbd-chain`**（[`_emit_xpbd_chain`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L331)，**默认**）：用 `add_link` + `add_shape_capsule` 建一串刚体胶囊，关节用 `add_joint_ball`（球关节）链式串联，由 `SolverXPBD` 模拟。

### 2.4 记录"所有权"

[`_emit_template`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L270-277) 末尾把 body/joint/shape 分成两组：

```
self.franka_bodies / franka_joints / franka_shapes   → 归 MuJoCo
self.payload_bodies / payload_joints / payload_shapes → 归 VBD/XPBD
```

**通俗解释**：这一步在告诉耦合器"机械臂这些刚体交给 MuJoCo 管，负载这些交给 VBD/XPBD 管"。所有权划分是后面 ADMM 判断"哪些接触是跨求解器接触、需要它来管"的依据。

---

## 3. IK 轨迹链路（`_build_keyframes` + `_build_ik` + `update_ik_targets`）

机械臂不靠力控，而是靠**逆运动学（IK）跟踪关键帧轨迹**，再把解出的关节角当作 MuJoCo 的位置目标。

### 3.1 关键帧（[`_build_keyframes`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L486)）

10 段位姿，每行 `[持续时间, x,y,z, qx,qy,qz,qw, 夹爪宽度]`，拼出经典抓取‑放置：接近→下降→闭合→抬起→移动→下降→松开→抬起。`key_times = cumsum(持续时间)`。

### 3.2 IK 模型（[`_build_ik`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L441)）

关键点：**IK 跑在一个"只有 Franka"的独立模型上**（注释明确：`IK runs on a Franka-only model so payload coordinates do not enter the solve`）。目标 body 是 `fr3_hand`，TCP 偏移 `z+0.107`。目标三个：
- `IKObjectivePosition`（位置）
- `IKObjectiveRotation`（姿态，固定朝下 `GRIPPER_DOWN`）
- `IKObjectiveJointLimit`（关节限位，权重 10）

IK 用解析雅可比（`IKJacobianType.ANALYTIC`），Levenberg‑Marquardt 风格，`lambda_initial=0.05`，每帧迭代 `ik_iters=24`。

因此运行时实际存在两个 `Model`：一个是包含 Franka+负载且拥有 `world_count` 个 world 的主物理模型，另一个是仅包含单台 Franka 的 IK 模型。`IKSolver(n_problems=world_count)` 在同一份 IK 模型拓扑上批量求解多个问题，不是在 IK 模型里再建 N 个 world。

### 3.3 每帧更新目标（[`update_ik_targets`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L528)）

按当前 `sim_time` 在 `key_times` 中二分定位区间，线性插值出当前目标位姿 + 夹爪宽度，再用 [`set_task_targets`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L65) kernel 写入 `ik_target_positions/ik_target_rotations/finger_pos_buf`（每个 world 一份）。

### 3.4 IK → 控制写入（[`simulate`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L557-565)）

```python
self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
wp.launch(set_gripper_q, ...)                 # 把手指目标宽度写进 IK 解
wp.copy(self.control.joint_target_q[:,:n_coords], self.ik_joint_q)  # 解→MuJoCo位置目标
```

**通俗解释**：IK 在"纯机械臂世界"里算出"要到达这个 TCP 位姿，7 个关节角该是多少"，然后把这个关节角丢给 MuJoCo 当位置目标。MuJoCo 再用 PD 力去追。负载根本不参与 IK，所以负载怎么动都不影响机械臂想去哪——但接触力会反过来影响它们俩的真实运动。

---

## 4. 子步调度与 CUDA Graph（`step` / `simulate` / `capture`）

### 4.1 帧调度

```
frame_dt = 1/60 ≈ 16.67 ms
sim_substeps = 16 (默认)
sim_dt = frame_dt / 16 ≈ 1.04 ms   每个子步
```

[`step`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L575) 每渲染帧执行一次：
1. `update_ik_targets()`：插值本帧目标。
2. 若有 graph 且设备是 CUDA：`_launch_frame_graph` 直接重放；否则 `self.simulate()` 即时跑。

### 4.2 `simulate` 主体（[L557-573](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L557)）

```python
def simulate(self):
    # 1) IK 求解 + 写控制
    self.ik_solver.step(...); set_gripper_q; wp.copy(ik→control)
    # 2) 子步循环
    for _ in range(self.sim_substeps):
        self.state_0.clear_forces()
        newton.examples.apply_coupled_viewer_forces(self, state_0)   # 鼠标交互力
        self.model.collide(state_0, self.contacts, collision_pipeline=self.collision_pipeline)
        self.solver.step(state_0, state_1, self.control, self.contacts, self.sim_dt)
        newton.eval_ik(self.model, state_1, state_1.joint_q, state_1.joint_qd)  # body 状态反算关节状态
        state_0, state_1 = state_1, state_0   # ping-pong
```

注意：任务空间 `IKSolver` 在子步循环**外**只算一次（按帧）。子步内的 `newton.eval_ik` 是另一个过程：它从子求解器输出的 `body_q/body_qd` 反算 `joint_q/joint_qd`，保证最大坐标与广义坐标在下一子步一致。它不是 `eval_fk` 的别名，也不是用来跟踪 TCP 目标的 `IKSolver`。

默认参数下，每帧调用一次 24 轮的任务空间 IK，再运行 16 个物理子步；每子步有 5 轮 ADMM，所以 MuJoCo 和 XPBD/VBD 每帧各实际执行 `16 × 5 = 80` 次 `step`。

### 4.3 CUDA Graph（[`capture`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L554) / [`_capture_frame_graph`](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L80)）

用 `wp.ScopedCapture()` 把整个 `simulate()` 录成一个 graph，之后每帧 `wp.capture_launch` 重放。前提：所有缓冲在录制前已分配（这就是为什么 `_setup_admm` 里要"eagerly allocate"内部接触缓冲，注释说明懒分配会在 graph 里留下非法指针）。

**通俗解释**：一次"碰撞+求解+FK"涉及成百上千个 GPU kernel 启动，每次启动有 CPU→GPU 的开销。CUDA Graph 把这整串录下来，之后一键回放，省掉每帧的启动开销——对 8 世界并行尤其关键。

---

## 5. 碰撞检测链路（`CollisionPipeline` / `Model.collide`）

本例有**两条碰撞管线**，理解它们的分工是理解耦合的关键。

### 5.1 用户主管线（全量碰撞）

```python
self.collision_pipeline = newton.CollisionPipeline(self.model)
self.contacts = self.collision_pipeline.contacts()
self.solver.prepare_contacts(self.contacts)
```

这条管线在 [collide.py `CollisionPipeline`](newton/_src/sim/collide.py#L710)，显式 broad phase，候选对来自 `model.shape_contact_pairs`（finalize 时由 `ModelBuilder._find_shape_contact_pairs` 构建的全量可碰撞 shape 对）。

**`Model.collide`**（[model.py L2303](newton/_src/sim/model.py#L2303)）每子步：
1. 计算 shape AABB（含 `margin+shape_gap` 扩张）→ [collide.py L165 `compute_shape_aabbs`](newton/_src/sim/collide.py#L165)
2. broad phase：`BroadPhaseExplicit` 把 `shape_contact_pairs` 拷进候选对数组
3. narrow phase（[narrow_phase.py](newton/_src/solvers/...)）：解析碰撞（球-球、胶囊-胶囊、平面-球…）+ GJK/MPR（凸-凸）+ mesh 三角化中相位 + SDF
4. 接触写进 `Contacts.rigid_contact_*`（刚-刚）和 `Contacts.soft_contact_*`（粒子-刚）
5. 可选 contact matching（帧间匹配）+ 确定性排序

**`prepare_contacts`**（[solver_coupled.py L1968](newton/_src/solvers/coupled/solver_coupled.py#L1968)）：为每个 sub‑solver entry 预分配一份**过滤后**的接触缓冲（`_ensure_entry_contact_buffer`）。真正过滤在 [`_contacts_for_entry`](newton/_src/solvers/coupled/solver_coupled.py#L2298)：用 `_filter_rigid_contacts_global_shape_ids_kernel` 按该 entry 的 `shape_flags`/`body_global_to_local` 只保留属于它的接触。这样 MuJoCo 只看到 Franka 自身和地面的接触，VBD/XPBD 只看到负载自身和地面的接触——**跨求解器接触不进任何一个子求解器**，而是留给 ADMM。

`use_mujoco_contacts=False` 的精确含义是关闭 MuJoCo/MJWarp 自身的碰撞检测，改用 Newton 生成的 `Contacts`。它不表示 MuJoCo 完全不处理接触响应：过滤后的 Franka–地面等接触会被转换到 MJWarp 接触数据中求解；只有 Franka–负载跨 Entry 接触由 ADMM 私有管线处理。

### 5.2 ADMM 私有管线（只跑跨求解器刚-刚接触）

这是耦合的灵魂。在 [`_setup_admm`](newton/_src/solvers/coupled/solver_coupled_admm.py#L1022) 里：

1. **发现跨所有者 shape 对**（[`_discover_rigid_rigid_contact_specs`](newton/_src/solvers/coupled/solver_coupled_admm.py#L2183)）：遍历 `model.shape_contact_pairs`，把每对 shape 的 body 映射到所属 entry；**只保留 `owner_a != owner_b` 的对**（即 Franka 形状 vs 负载形状）。同一 entry 内的对（如 Franka 自连杆、负载链节之间）被丢弃——那些由各自子求解器自己处理。
2. **建精确 shape 对数组**（[`_build_admm_rigid_shape_pair_array`](newton/_src/solvers/coupled/solver_coupled_admm.py#L2138)）。
3. **建专用 `CollisionPipeline`**（[L1092-1103](newton/_src/solvers/coupled/solver_coupled_admm.py#L1092)）：
   ```python
   self._admm_collision_pipeline = CollisionPipeline(
       self.model,
       broad_phase="explicit",
       shape_pairs_filtered=admm_shape_pairs,   # 只含跨 entry 对
       rigid_contact_max=rigid_contact_max,      # = 8 * 跨对数
       soft_contact_max=0 (无刚-粒时) / None,
       contact_matching=("latest"/"sticky" 或 "disabled"),
   )
   self._admm_internal_contacts = self._admm_collision_pipeline.contacts()
   ```
4. 每子步在 [`_refresh_collision_contact_groups`](newton/_src/solvers/coupled/solver_coupled_admm.py#L2825) 调 `self._admm_collision_pipeline.collide(state_in, self._admm_internal_contacts)` 重算跨求解器刚-刚接触。

**`Contacts` 缓冲结构**（[contacts.py](newton/_src/sim/contacts.py#L93)）：刚体接触字段 `rigid_contact_shape0/1`、`point0/1`（body 坐标）、`offset0/1`（摩擦锚偏移）、`normal`、`margin0/1`、`point_id`、可选 `match_index`、`force`。

**通俗解释**：接触分两类——"自家人碰自家人"（机械臂碰自己、负载碰自己）和"两家碰一起"（机械臂碰负载）。前者各回各家、各找各求解器；后者谁都不能独吞，于是 ADMM 单独开一条小碰撞管线，每步只算"两家交界处"的接触，再在迭代里协调。

---

## 6. `SolverCoupled` 基类调度骨架

[`SolverCoupled`](newton/_src/solvers/coupled/solver_coupled.py#L227) 是模板方法模式的基类。公开入口 [`step`](newton/_src/solvers/coupled/solver_coupled.py#L1947)：

```python
def step(self, state_in, state_out, control, contacts, dt):
    self._distribute_state(state_in, dt=dt)          # 把父状态拷进各 entry.state_0
    self._step_coupled(state_in, state_out, control, contacts, dt)  # ← 子类重写
    _copy_state(state_in, state_out)
    self._reconcile_state(state_out)                  # 各 entry.state_1 散射回父状态
    self._entry_output_state_valid = True
```

- **`_distribute_state`**（[L2027](newton/_src/solvers/coupled/solver_coupled.py#L2027)）：用 `_copy_state_to_entry` 把父模型的 body/particle/joint 状态按 entry 的全局→局部分映射拷进子求解器输入状态。
- **`_reconcile_state`**（[L2040](newton/_src/solvers/coupled/solver_coupled.py#L2040)）：反向，用 `_scatter_*_state_mapped` kernel 把各 entry 输出按局部→全局散射回父 `state_out`。
- **`_step_entry`**（[L2257](newton/_src/solvers/coupled/solver_coupled.py#L2257)）：单 entry 步进，处理子步（`entry.substeps`）、in‑place vs 双缓冲、接触过滤（`_contacts_for_entry`）、控制拷贝。
- **`SolverCoupled.Entry`**（[L246](newton/_src/solvers/coupled/solver_coupled.py#L246)）：用户配置 `name`、`solver`（工厂 lambda）、`bodies/joints/particles/shapes`、`substeps`、`in_place`。本例两个 entry 见下。

**通俗解释**：基类像个"调度台"：开播前把总状态分发到各分台，让分台各自演，演完再把各分台结果收回到总状态。具体怎么演（一次跑还是迭代着跑）由子类 `_step_coupled` 决定。

---

## 7. `SolverCoupledADMM` 构造与本例配置

### 7.1 本例的两个 Entry（[L152-173](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L152)）

```python
entries=[
    SolverCoupled.Entry(name="mjc", solver=lambda v: SolverMuJoCo(
        model=v, solver="newton", integrator="implicitfast",
        iterations=12, ls_iterations=25,
        use_mujoco_contacts=False,           # 关闭 MJWarp 内建碰撞检测，接收 Newton Contacts
        njmax=max(256, 64*world_count), nconmax=...),
        bodies=self.franka_bodies, joints=self.franka_joints),
    SolverCoupled.Entry(name=payload_name,  # "vbd" 或 "xpbd"
        solver=payload_solver,               # SolverVBD 或 SolverXPBD
        bodies=self.payload_bodies, joints=self.payload_joints),
],
```

**`use_mujoco_contacts=False`** 关闭的是 MuJoCo/MJWarp 内建碰撞检测，并让 MuJoCo 接收 Newton 的 `Contacts`。因此 Franka 内部或 Franka–全局地面接触仍会被转换到 MJWarp 中求解；ADMM 另外计算 Franka–负载的跨 Entry 接触力，并通过 body force 输入施加到两侧。

### 7.2 耦合配置（`SolverCoupledADMM.Config`，[L175-190](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L175)）

| 字段 | 本例默认 | 含义 |
|------|----------|------|
| `iterations` | 5 | 每子步 ADMM 迭代轮数 |
| `rho` | 200 | ADMM 罚参数 ρ |
| `gamma` | 0.001 | 近端质量缩放 γ（>0 启用 proximal term） |
| `baumgarte` | 0.5 | 位置误差修正比例 β |
| `rigid_contact_matching` | "latest" | 跨帧接触匹配模式（warm‑start） |
| `contact_matching_force_scale` | 0.9 | warm‑start λ 的缩放 |
| `contact_pairs` | `[ContactPair(source="mjc", destination=payload_name)]` | 启用 mjc↔payload 跨接触 |

`Config` 完整字段见 [solver_coupled_admm.py L579](newton/_src/solvers/coupled/solver_coupled_admm.py#L579)。

### 7.3 `__init__` 流程（[L646](newton/_src/solvers/coupled/solver_coupled_admm.py#L646)）

```
校验 config (_validate_config)
可选: _init_admm_joint_proxy_visibility   # 跨求解器关节邻居作为本地代理体
super().__init(model, entries, coupling)  # 基类建 entry、view、子求解器
_setup_admm(coupling)                      # 建所有 ADMM 缓冲/组/接触 spec/私有管线
_apply_cached_admm_joint_proxy_effective_masses
```

---

## 8. ADMM 算法核心：公式原理

这是全文最关键的部分。ADMM 求解的是"两个子系统各自最小化自己的能量，但被一组耦合约束 C(u)=0 拉到一致"。

### 8.1 增广拉格朗日 / ADMM 分裂

对每个耦合约束（接触或 attachment），令：
- `Jv` = 约束雅可比 × 当前相对速度（约束"测得"的相对速度，primal 残差）
- `u` = 局部"目标相对速度"（z 变量，ADMM 的 proximal 项）
- `λ` = 对偶变量（拉格朗日乘子，等价接触冲量/力）
- `W` = 接口权重 = `sqrt(m_a·m_b/(m_a+m_b))`（[admm_utils.py `_interface_weight` L27](newton/_src/solvers/coupled/admm_utils.py#L27)），把两侧质量折算成统一尺度
- `ρ` = 罚参数

**u 更新**（primal）：对二次耦合能量 `E(u)=(κ/2)||u−u_target||² + (d/2)||u||²`，闭式解（[`u_update_quadratic_kernel` L712](newton/_src/solvers/coupled/admm_utils.py#L712)）：

```
u^{k+1} = (ρ W² Jv + κ·u_target − W·λ^k) / (κ + d + ρ W²)
```

**λ 更新**（dual，[`lambda_update_kernel` L731](newton/_src/solvers/coupled/admm_utils.py#L731)）：

```
λ^{k+1} = λ^k + ρ·W·(u^{k+1} − Jv)
```

这正是标准 ADMM：`u` 是 primal 子问题解，`λ` 是乘子上升。`W` 让等效力按质量加权分配给两侧。

**通俗解释**：ADMM 每轮做两件事——(1) "如果当前接触相对速度是 Jv、乘子是 λ，那约束期望的相对速度 u 该是多少？"（u 更新）；(2) "实际 Jv 跟期望 u 还差多少？差多少就把乘子 λ 往那边推一格。"（λ 更新）。反复几轮，Jv→u，λ 收敛，接触力稳定。

### 8.2 接触的特殊性：非负法向 + Coulomb 摩擦

接触不是普通二次约束，而是**非互补约束**：法向力不能拉（≥0），切向受 Coulomb 锥 `||f_t|| ≤ μ·f_n`。所以接触的 u 更新用 [`contact_u_update_kernel` L811](newton/_src/solvers/coupled/admm_utils.py#L811)：

```
p = Jv − λ/(ρW)                       # 无约束解
若接触未激活 (u_min ≤ −1e8): u = p   # 自由相对滑动
否则:
    shifted = p − u_min·n             # 减去 Baumgarte 修正目标速度
    u = solve_coulomb_isotropic(μ, n, shifted) + u_min·n
```

其中 [`solve_coulomb_isotropic` L788](newton/_src/solvers/coupled/admm_utils.py#L788) 是 Daviet 风格的最大耗散投影：分离接触保持相对速度、粘附接触归零、滑动接触把切向速度沿 Coulomb 锥边界裁剪。法向 `u_n` 被钉到 `u_min`（Baumgarte 目标速度）。

### 8.3 Baumgarte 位置修正

接触会穿透，光靠速度约束会"软"。`baumgarte=β` 让法向目标速度非零，把穿透往回推。[`_contact_u_min_from_gap` L58](newton/_src/solvers/coupled/admm_utils.py#L58)：

```
gap = n·(point_a − point_b)            # 穿透量（<0 表示穿透）
若 gap < 0 (穿透): u_min = −β·gap/dt   # 正速度把两面推开
否则:             u_min = −gap/dt      # 仅消除间隙
```

`β=0.5` 意为每步修正 50% 穿透。对 attachment（关节），[`attach_rr_compute_u_target` L957](newton/_src/solvers/coupled/admm_utils.py#L957) 用 `u_target = (β/dt)·gap` 把锚点位置误差转成目标速度。

### 8.4 力的施加：`J^T W` 反推

ADMM 算出 `u, λ` 后，要把约束力施加回两侧刚体。[`contact_rr_accumulate_forces_kernel` L1286](newton/_src/solvers/coupled/admm_utils.py#L1286) 和 [`attach_rr_accumulate_forces_kernel` L1184](newton/_src/solvers/coupled/admm_utils.py#L1184) 的核心：

```
force_a = W·(λ + ρ·W·(u − Jv))     # 侧 A 受正向力
force_b = −force_a                  # 侧 B 反向
wrench = (force, cross(arm, force)) # arm = 接触点 − COM
atomic_add(body_f[ba], wrench)
```

注意 `λ + ρW(u−Jv)` 正是 ADMM 收敛后的稳定约束力（收敛时 u≈Jv，力≈W·λ）。在迭代中 `ρW(u−Jv)` 是 primal 残差罚项，把还没收敛的偏差也转成力，加速一致。`arm`（接触点到质心臂）决定力矩。

**通俗解释**：ADMM 在两边各画一个"虚拟弹簧+摩擦"把接触连起来。每轮它根据当前两边的速度差和乘子，算出该用多大力把两边拽到一致，再把这个力（含力矩）atomic 加到各自刚体的力缓冲里。两边各自带着这个外加力跑自己的物理，跑完再看差距更新乘子。

---

## 9. ADMM 单步调度：`_step_coupled` 全流程

[`_step_coupled` L2762](newton/_src/solvers/coupled/solver_coupled_admm.py#L2762) 是真正的耦合主循环。下面是完整调度栈：

```
_step_coupled(state_in, state_out, control, contacts, dt):
│
├─ 0. _refresh_collision_contact_groups(state_in)         # 重算跨求解器接触
│     ├─ _admm_collision_pipeline.collide(state_in, _admm_internal_contacts)
│     ├─ for rr_group:  snapshot_by_contact → reset → fill_from_rigid_contacts
│     ├─ for rp_group:  snapshot → reset → fill_from_soft_contacts
│     └─ for pp_group:  particle_grid.build → hashgrid → fill_from_particle_contacts
│
├─ 1. if γ>0: _refresh_admm_proximal_masks + _refresh_admm_proximal_view_overrides
│
├─ 2. 快照本步起点 n（每个 entry）:
│      body_q_n←state_0.body_q, body_qd_n←qd, body_qd_k←qd (k=迭代工作速度)
│
├─ 3. _admm_begin_step(dt)                                # 算 u_target / u_min
│      ├─ attach_*_compute_u_target  (Baumgarte 锚点目标速度)
│      └─ contact_*_compute_u_min   (Baumgarte 法向目标速度)
│
└─ 4. for k in range(iterations):                         # ADMM 主循环（默认5轮）
       │
       ├─ 4a. for entry: _prepare_admm_iteration_state     # 每轮重置到 n 起点
       │        ├─ state_0 ← n 快照（body_q/qd 复位）
       │        ├─ _notify_input_state_update
       │        ├─ if γ>0: _apply_admm_velocity_proximal_shift  # proximal 速度修正
       │        └─ body_f/particle_f ← 拷贝外部力 + 重力补偿
       │
       ├─ 4b. _accumulate_admm_forces(k, dt, refresh_jv=(k==0), init_u=(k==0))
       │        # 对每个 group: 算 Jv（用 qd_k）→ accumulate_forces 把力 splat 进 body_f
       │        ├─ attach_rr/angular/revolute/friction groups
       │        ├─ attach_rp groups
       │        ├─ contact_rr groups (compute_Jv + accumulate_forces)
       │        ├─ contact_rp groups
       │        └─ contact_pp groups (+ contact_stream normal_force)
       │
       ├─ 4c. for entry: _apply_admm_force_inputs          # body_f/particle_f → 子求解器输入
       │
       ├─ 4d. for entry: _step_entry(entry, control, contacts, dt)  # 各子求解器真跑一步
       │           └─ MuJoCo.step / VBD.step / XPBD.step
       │
       ├─ 4e. for entry: body_qd_k ← state_1.body_qd      # 收集本轮解出的速度作为下一轮 Jv 输入
       │
       └─ 4f. _update_admm_dual(k, dt)                     # 用 state_1 速度重算 Jv → 更新 u, λ
                ├─ 重算 Jv（用 state_1 qd，刚跑完的结果）
                ├─ quadratic groups: _update_admm_quadratic_dual (u, λ)
                └─ contact groups: _update_admm_contact_dual (u, λ 含 Coulomb)
```

### 9.1 关键子步骤函数作用

- **[`_refresh_collision_contact_groups` L2825](newton/_src/solvers/coupled/solver_coupled_admm.py#L2825)**：每子步开头跑一次跨求解器碰撞，把检测结果填进各 contact group 的 `body_ids/point/normal/W/friction/lambda`，并按 `rigid_contact_matching` warm‑start λ（用 `prev_contact_lambda` + `contact_matching_force_scale` 缩放 + `_rescale_lambda` 按新 W 重标定）。
- **[`_admm_begin_step` L3382](newton/_src/solvers/coupled/solver_coupled_admm.py#L3382)**：基于本步起点位姿算 `u_target`（attachment）和 `u_min`（contact）——都是 Baumgarte 位置修正目标。
- **[`_prepare_admm_iteration_state` L3605](newton/_src/solvers/coupled/solver_coupled_admm.py#L3605)**：每轮 ADMM 开头把状态复位到步起点 n（保证每轮都从同一初值迭代），拷外部力，做重力补偿。
- **[`_apply_admm_velocity_proximal_shift` L3551](newton/_src/solvers/coupled/solver_coupled_admm.py#L3551)**：γ>0 时的 proximal Newton 速度偏移，让近端项把"已知要被约束拉住"的自由度预修正。
- **[`_accumulate_admm_forces` L3742](newton/_src/solvers/coupled/solver_coupled_admm.py#L3742)**：算 `Jv` 并把 ADMM 力 splat 进 body_f/particle_f。第一轮（k=0）才算 Jv 和初始化接触 u，后续轮复用（因为输入速度 qd_k 在 4e 更新后会重算于 4f）。
- **[`_step_entry`](newton/_src/solvers/coupled/solver_coupled.py#L2257)**：调子求解器 `step(state_0, state_1, control, contacts, dt)`。注意 contacts 是过滤后的 entry‑local 接触（自家人接触），不含跨求解器接触。
- **[`_update_admm_dual` L4104](newton/_src/solvers/coupled/solver_coupled_admm.py#L4104)**：用子求解器刚跑出的 `state_1` 速度重算 Jv，再更新 `u` 和 `λ`。

### 9.2 调用栈（一子步）

```
Example.simulate
└─ self.solver.step (SolverCoupled.step)
   ├─ _distribute_state
   ├─ _step_coupled (SolverCoupledADMM 覆写)        ← ADMM 主循环
   │  ├─ _refresh_collision_contact_groups
   │  │   └─ _admm_collision_pipeline.collide
   │  ├─ _admm_begin_step
   │  └─ for k in iterations:
   │      ├─ _prepare_admm_iteration_state (×entries)
   │      ├─ _accumulate_admm_forces
   │      ├─ _apply_admm_force_inputs (×entries)
   │      ├─ _step_entry (×entries)  ─► SolverMuJoCo.step / SolverVBD.step / SolverXPBD.step
   │      └─ _update_admm_dual
   ├─ _copy_state(state_in, state_out)
   └─ _reconcile_state
```

**通俗解释**：每个 1ms 子步里，ADMM 先做一次"跨家接触检测"，然后做 5 轮"协调舞蹈"——每轮：所有家复位到子步起点→ADMM 根据当前接触算修正力→力塞给各家→各家各自跑一步物理→用跑完的速度更新接触乘子。5 轮后接触力收敛，最后把各家结果合并回总状态。所以一个子步里 MuJoCo 实际被调了 5 次（每轮一次），不是 1 次。

---

## 10. 耦合约束的三类与各自函数链路

ADMM 处理三类耦合约束，每类对应一组 group 列表和一套 kernel：

### 10.1 Attachment（跨求解器关节锚定）

来自**跨求解器关节**：模型里如果一个关节的 parent/child 分属不同 entry（本例没有，但模板支持），[`_build_admm_joint_groups` L2396](newton/_src/solvers/coupled/solver_coupled_admm.py#L2396) 把它转成 ADMM attachment：
- `BALL` 关节 → `_AdmmRigidRigidAttachmentGroup`（3 自由度位置锚，`attach_rr_*` kernel）
- `FIXED` 关节 → `_AdmmRigidRigidAngularAttachmentGroup`（3 自由度姿态锚，`attach_rr_angular_*`）
- `REVOLUTE` 关节 → `_AdmmRigidRigidAngularAttachmentGroup`（2 自由度姿态约束，留 1 转动轴，`attach_rr_revolute_angular_local_*`）
- 关节摩擦 → `_AdmmRigidRigidAngularFrictionGroup`（`joint_box_friction_u_update_kernel` 盒摩擦）

还有用户自定义的刚体‑粒子锚（[`_build_admm_body_particle_attachment_groups` L2674](newton/_src/solvers/coupled/solver_coupled_admm.py#L2674)，`add_body_particle_attachment` API）→ `_AdmmRigidParticleAttachmentGroup`。

这些用**二次能量**（κ 刚度 + d 阻尼），更新走 [`_update_admm_quadratic_dual` L3708](newton/_src/solvers/coupled/solver_coupled_admm.py#L3708)（`u_update_quadratic_kernel` + `lambda_update_kernel`）。

### 10.2 刚-刚接触（本例主路径：Franka ↔ 负载）

`_AdmmRigidRigidContactGroup`（[`_build_collision_rigid_rigid_contact_groups` L3166](newton/_src/solvers/coupled/solver_coupled_admm.py#L3166)）。kernel 链路：

```
contact_rr_compute_Jv_kernel        # 算 Jv = v_a(接触点) − v_b(接触点)
contact_rr_compute_u_min_kernel     # 算 u_min（Baumgarte 法向目标）
contact_rr_accumulate_forces_kernel # 力 splat：force_a = W(λ + ρW(u−Jv))
contact_u_update_kernel             # u 更新（Coulomb 投影）
contact_lambda_update_kernel        # λ 更新
contact_rr_fill_from_rigid_contacts_kernel  # 从 Contacts 缓冲填 group
contact_rr_snapshot/reset_kernel    # warm-start 快照与重置
```

接触点速度用 [`velocity_at_point`](newton/_src/math/spatial.py)（`v_com + ω × arm`），`arm = 接触点 − COM_world`。

### 10.3 刚-粒接触、粒-粒接触

- 刚-粒（`_AdmmRigidParticleContactGroup`，soft_contact 路径，`contact_rp_*`）
- 粒-粒（`_AdmmParticleParticleContactGroup`，`contact_pp_*` + `AdmmContactStream` + `particle_particle_contacts_hashgrid_kernel` 网格查询）

本例无粒子，这两类不激活，但代码完备。

---

## 11. 跨帧接触匹配（warm-start）流程

接触每步都重新检测，行号会变，但物理上"同一个接触点"应连续。`rigid_contact_matching` 三模式（[Config L640](newton/_src/solvers/coupled/solver_coupled_admm.py#L640)）：

- **`disabled`**：每步 λ 清零，纯冷启动。
- **`latest`**（本例默认）：用上一帧 `prev_contact_lambda` warm-start 当前 λ。匹配靠 `CollisionPipeline` 的 `ContactMatcher`（按中点距离 `contact_matching_pos_threshold` + 法向点积 `contact_matching_normal_dot_threshold`），产生 `rigid_contact_match_index` 把当前接触指回上一帧对应行。
- **`sticky`**：除 warm-start λ 外还回放上一帧接触几何（位置/法向），保持接触稳定。

填充见 [`contact_rr_fill_from_rigid_contacts_kernel` L1411](newton/_src/solvers/coupled/admm_utils.py#L1411)：若 `use_contact_matching` 且 `match_index[i]` 命中上一帧活跃行，则
```
λ_new = contact_matching_force_scale · _rescale_lambda(prev_λ, prev_W, new_W)
```
`_rescale_lambda`（[L51](newton/_src/solvers/coupled/admm_utils.py#L51)）按权重比缩放保持力量纲一致。每步开头先 `contact_rr_clear_contact_snapshot_kernel` 清快照、`contact_rr_snapshot_by_contact_kernel` 存当前 λ、`contact_rr_reset_kernel` 清 group，再 fill。

**通俗解释**：接触像"握手"，每步手的位置略变，但握的还是同一只手。如果每步都从零开始算握力（λ），会抖。warm-start 把上一步的握力按新姿势微调后当作本轮起点，收敛快得多、稳定得多。

---

## 12. 多世界并行的特殊处理

本例 `world_count=8`（默认），同模板复制 8 份。

- **下标展开**（[`_expand_world_indices` L385](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L385)）：所有 body/joint/shape 全局下标 = `world*stride + local`。
- **接触对校验**（[`_count_admm_shape_pairs_per_world` L396](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L396)）：确保每个世界 Franka↔负载接触对数量一致、无跨世界接触（跨世界接触会抛 `RuntimeError`）。`test_final` 也断言这点。
- **IK 多问题**：`IKSolver(n_problems=world_count)`，每世界独立解，目标位姿广播到所有世界。
- **MuJoCo 容量**：`njmax=max(256,64*world_count)`、`nconmax=max(64,16*world_count)`，按世界数缩放。

所有物理 world 保持在相同物理坐标附近，`viewer.set_world_offsets((1.1, 1.1, 0.0))` 只在显示时把它们排开。world 索引负责阻止局部 world 之间的碰撞，而唯一的全局地面（`world=-1`）对所有 world 可见。

---

## 13. `gamma` 近端项（proximal term）机制

`gamma=0.001`（>0）启用 ADMM 的 proximal/acceleration。核心思想：在子求解器视图里，给"参与耦合约束"的自由度临时加一份近端质量/惯性（`body_proximal_mass/inertia`），并对已知约束方向预施加速度偏移。

链路：
- [`_refresh_admm_proximal_masks` L1195](newton/_src/solvers/coupled/solver_coupled_admm.py#L1195)：标记哪些 body/粒子/关节 dof 参与约束（`mark_*` kernel + `accumulate_*_proximal_lump` kernel 累加 `γ·ρ·W²` 的等效质量）。
- [`_refresh_admm_proximal_view_overrides` L1742](newton/_src/solvers/coupled/solver_coupled_admm.py#L1742)：把 proximal 质量注入 entry view（`add_body_lumped_inertia`），并对该方向做重力补偿（`body_gravity_compensation_lumped_kernel`），避免子求解器在这些方向重复算重力。
- [`_apply_admm_velocity_proximal_shift` L3551](newton/_src/solvers/coupled/solver_coupled_admm.py#L3551)：每轮迭代前按 `qd_n`（起点）和 `qd_k`（工作）差、结合 proximal/原始质量比，给输入速度加一个预测偏移。

**通俗解释**：proximal term 像"预判"。ADMM 知道某些自由度肯定要被接触约束动一动，就提前给它们加点惯性、并按上一轮的趋势预先挪一下速度，让子求解器跑出来的结果更接近收敛解，减少迭代轮数。

---

## 14. 关键数据结构与缓冲速查

| 结构 | 位置 | 作用 |
|------|------|------|
| `SolverEntry` | [solver_coupled.py L186](newton/_src/solvers/coupled/solver_coupled.py#L186) | 单 entry 运行态：view、state_0/1、index maps、substeps |
| `_AdmmBuffers` | [L127](newton/_src/solvers/coupled/solver_coupled_admm.py#L127) | 每 entry 的 ADMM 工作缓冲：`body_q_n/qd_n/qd_k`、`body_f`、`body_effective_mass`、proximal mask/mass |
| `_AdmmRigidRigidContactGroup` | [L256](newton/_src/solvers/coupled/solver_coupled_admm.py#L256) | 刚-刚接触组：`body_ids_a/b`、`point/offset`、`normal`、`W`、`friction`、`u/lambda/Jv/u_min`、`active_count`、`prev_*` warm-start |
| `Contacts` | [contacts.py L93](newton/_src/sim/contacts.py#L93) | 全局/内部接触缓冲：`rigid_contact_*`、`soft_contact_*` |
| `AdmmContactStream` | [admm_contact_stream.py L28](newton/_src/solvers/coupled/admm_contact_stream.py#L28) | 粒-粒接触流：`normal_force/impulse`、`particle_a/b` |
| `_AdmmJointProxyMapping` | [L406](newton/_src/solvers/coupled/solver_coupled_admm.py#L406) | 跨求解器关节邻居代理体映射 |

---

## 15. 验证与测试（`test_final` / `test_post_step`）

[`test_final` L581](newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py#L581)：
1. `body_q`/`body_qd` 全有限（无 NaN/Inf）——仿真稳定。
2. 每个世界 Franka↔负载的**候选 shape pair** 数 > 0，且各世界一致——这能验证 ADMM 接触配置和模板复制，但不能证明运行时已经产生实际接触或完成抓取。
3. 若 `use_graph`，graph 非空。

---

## 16. 命令行参数与默认值速查

| 参数 | 默认 | 作用 |
|------|------|------|
| `--world-count` | 8 | 并行世界数 |
| `--substeps` | 16 | 每帧耦合子步 |
| `--admm-iterations` | 5 | 每子步 ADMM 轮数 |
| `--rho` | 200 | ADMM 罚参数 |
| `--gamma` | 0.001 | 近端质量缩放 |
| `--baumgarte` | 0.5 | 位置修正比例 |
| `--rigid-contact-matching` | latest | 跨帧匹配模式 |
| `--payload-kind` | xpbd-chain | 负载类型 |
| `--payload-segments` | 11 | 负载段数 |
| `--payload-radius` | 0.012 | 负载半径 [m] |
| `--xpbd-iterations` | 16 | XPBD 迭代 |
| `--vbd-iterations` | 8 | VBD 迭代 |
| `--mujoco-iterations` | 12 | MuJoCo 迭代 |

启动：`python -m newton.examples mujoco_franka_vbd_cable_admm_solver`

---

## 17. 全景数据流图

```
                    ┌─────────────────────────────────────────────────┐
                    │  Example.__init__                                │
                    │  template(Franka+负载) → replicate(N世界)          │
                    │  SolverCoupledADMM(entries=[mjc, payload], cfg)  │
                    │  CollisionPipeline(用户主管线) + contacts         │
                    │  IK 模型 + 关键帧                                │
                    └─────────────────────────────────────────────────┘
                                          │
   每帧 step():                            ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ update_ik_targets (插值目标) → set_task_targets kernel          │
   │ capture_launch(graph) 或 simulate():                            │
   │   ik_solver.step → set_gripper_q → copy(ik_q → control.target_q)│
   │   for substep in 16:                                            │
   │     state_0.clear_forces + viewer 力                            │
   │     model.collide(state_0, contacts, 用户管线)  ── 全量接触      │
   │     solver.step(state_0, state_1, control, contacts, dt):     │
   │       _distribute_state (父→各 entry.state_0)                  │
   │       _step_coupled (ADMM):                                     │
   │         _refresh_collision_contact_groups (ADMM私有管线,只跨家)  │
   │         _admm_begin_step (u_target/u_min)                       │
   │         for k in 5:                                             │
   │           _prepare_admm_iteration_state (复位n+重力补偿)        │
   │           _accumulate_admm_forces (Jv→力splat进body_f)          │
   │           _apply_admm_force_inputs (body_f→子求解器)            │
   │           _step_entry ×2: MuJoCo.step / VBD|XPBD.step          │
   │           body_qd_k ← state_1.qd                                │
   │           _update_admm_dual (重算Jv→u,λ 更新)                   │
   │       _reconcile_state (各entry.state_1→父state_1)             │
   │     eval_ik (body 状态→关节坐标重建)                         │
   │     state_0 ↔ state_1 ping-pong                                 │
   └──────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                     render / test_final
```

---

## 18. 总结：耦合的本质

本例展示的耦合范式可归纳为：

1. **域分解**：把整体系统按所有权拆给异构求解器（MuJoCo 管刚体关节 PD，VBD/XPBD 管柔性/链），各自在紧凑局部视图上高效求解。
2. **接触隔离**：跨域接触由专用碰撞管线单独检测，不进任何子求解器；同域接触仍由子求解器自处理。
3. **ADMM 协调**：对每个跨域约束，ADMM 用 `Jv`（测得相对速度）→ `u`（期望相对速度，含 Coulomb/Baumgarte）→ `λ`（乘子=接触力）三轮迭代，把约束力 splat 回两侧，再让子求解器重跑。`W` 按质量加权分摊，`ρ` 控收敛，`γ` 加速。
4. **warm-start 与图捕获**：跨帧接触匹配 + CUDA Graph 让多世界并行实时可行。

这套设计让"不同物理特性、不同积分器的子系统"能在一个统一时间步内**强耦合**地交互，而每个子系统内部仍用自己最擅长的算法——这是 Newton `SolverCoupledADMM` 的核心价值。

**通俗总结**：把"机械臂"和"软绳/链"这种本该用完全不同算法模拟的东西，用一个 ADMM"中间人"缝合在一个仿真步里，既能各自跑得快、又能在接触处严丝合缝地相互作用，还能批量并行、还能录成图高速回放。
