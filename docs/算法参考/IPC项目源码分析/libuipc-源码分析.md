# libuipc 源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\libuipc`

关键源码：

- `src/backends/cuda/engine/sim_engine_do_advance.cu`
- `src/backends/cuda/engine/advance_ipc.cu`
- `src/backends/cuda/engine/advance_al.cu`
- `src/backends/cuda/collision_detection/global_trajectory_filter.cu`
- `src/backends/cuda/contact_system/global_contact_manager.cu`

辅助观察目录：

- `finite_element/`
- `affine_body/`
- `contact_system/`
- `collision_detection/`
- `line_search/`
- `time_integrator/`
- `newton_tolerance/`
- `coupling_system/`

## 1. 项目定位

`libuipc` 不是“单个 IPC 求解器文件”，而是一个完整的 GPU 仿真后端框架。

从源码结构能确认它具备：

- FEM
- ABD 刚体
- 壳/杆/粒子 codim 模型
- 统一接触 tabular
- 基础 IPC 管线
- Augmented Lagrangian IPC 管线
- 动画、外力、差分仿真、耦合系统

所以理解它的关键不是“某个 barrier 公式”，而是“多个 SimSystem 如何接成一条 GPU pipeline”。

## 2. 项目架构

### 2.1 顶层系统

```text
SimEngine
-> GlobalVertexManager / SurfaceManager
-> TimeIntegratorManager
-> GlobalTrajectoryFilter
-> GlobalContactManager
-> GlobalDyTopoEffectManager
-> GlobalLinearSystem
-> LineSearcher
-> NewtonToleranceManager
-> Animator / ExternalForce / DiffSim
```

### 2.2 物理对象子系统

```text
finite_element/
affine_body/
implicit_geometry/
inter_primitive_effect_system/
coupling_system/
```

可以把它理解为：

- `finite_element/`：软体、壳、杆、粒子
- `affine_body/`：ABD 刚体及关节
- `implicit_geometry/`：半平面等解析碰撞体
- `inter_primitive_effect_system/`：缝合、点选等动态拓扑效应
- `coupling_system/`：FEM 与 ABD 等跨系统耦合

### 2.3 接触与碰撞子系统

```text
collision_detection/
-> trajectory filters
-> LBVH / stackless BVH
-> candidate detect
-> TOI filter

contact_system/
-> contact tabular
-> normal contact
-> frictional contact
-> adaptive parameter strategy
```

## 3. 算法模型

## 3.1 Basic IPC pipeline

`SimEngine::advance()` 对应的是基础 barrier IPC。

整体逻辑：

- 预测自由运动
- 生成 DCD/trajectory 候选
- 计算接触与其他动态拓扑效应的梯度/Hessian
- 解全局线性系统
- 线搜索时做 CCD/TOI/CFL 过滤
- 更新 DOF 和速度

## 3.2 Augmented Lagrangian pipeline

`SimEngine::advance_AL()` 对应 AL-IPC 管线。

相对于基础 IPC，多了：

- `GlobalActiveSetManager`
- 约束线性化
- slack 更新
- 恢复到无穿透位置
- 自适应 `mu`

因此它的接触不是直接依赖 barrier 全量候选，而是显式 active set。

## 4. 数据链路

### 4.1 场景配置链

`GlobalContactManager::do_build()` 读取：

- `contact/d_hat`
- `contact/eps_velocity`
- `cfl/enable`
- `dt`

同时从 scene 中构造：

- `contact_tabular`
- `subscene_tabular`

这意味着 `libuipc` 在工程上把“谁和谁能接触、用多大 kappa、摩擦系数多少”做成了场景级矩阵，而不是硬编码在 solver 内。

### 4.2 每帧主链

基础 IPC 管线的数据流可以概括成：

```text
Scene / config
-> VertexManager.update_attributes()
-> record_prev_positions()
-> Animator / ExternalForce
-> TimeIntegrator.predict_dof()
-> GlobalTrajectoryFilter.detect(0.0)
-> GlobalContactManager.compute_adaptive_parameters()
-> Newton loop:
   -> detect_dcd_candidates()
   -> GlobalDyTopoEffectManager.compute_dytopo_effect()
   -> GlobalLinearSystem.solve()
   -> VertexManager.collect_vertex_displacements()
   -> LineSearcher.record_start_point()
   -> GlobalTrajectoryFilter.detect(alpha)
   -> compute_energy(alpha)
   -> filter_toi(alpha)
   -> compute_cfl_condition()
   -> backtracking line search
-> TimeIntegrator.update_state()
```

### 4.3 active set 链

AL 管线下则是：

```text
trajectory filters
-> GlobalActiveSetManager.update_active_set()
-> linearize_constraints()
-> update_slack()
-> contact models consume PT/EE/PH active sets
-> line search / recovery update active-set state
```

## 5. 碰撞流程

## 5.1 `GlobalTrajectoryFilter`

它是 GPU 碰撞检测的统一门面。

### 输入

- 当前顶点位置
- 预测轨迹缩放 `alpha`
- 已注册的 filter 集合

### 输出

- 各 filter 的候选对
- 各 filter 的 TOI
- 可选摩擦候选
- 活动接触顶点标记

### 可确认行为

- `detect(alpha)`：对所有 filter 下发轨迹查询
- `filter_active()`：把候选转成 active contact 标志
- `filter_toi(alpha)`：收集所有 filter 的最小 TOI
- `record_friction_candidates()`：保存摩擦候选

## 5.2 候选类型

从目录和调用关系看，至少包含：

- simplex trajectory filter：`PT / EE`
- vertex half-plane trajectory filter：`PH`

如果启用 codim 接触，`contact_models/` 里还能处理：

- `PP / PE / PT / EE`

## 5.3 CFL

`GlobalContactManager::compute_cfl_condition()` 的逻辑很清楚：

1. 让 trajectory filter 标注活动接触顶点
2. 只统计活动接触顶点位移范数
3. 取最大位移
4. 返回 `min(0.5 * d_hat / max_disp, 1.0)`

这相当于又给了 line search 一个动力学稳定性上界。

## 6. 接触流程

## 6.1 `GlobalContactManager`

### 输入

- `scene.contact_tabular()`
- `scene.subscene_tabular()`
- `GlobalVertexManager`
- `GlobalTrajectoryFilter`

### 输出

- 设备端 `contact_tabular`
- 设备端 `contact_mask_tabular`
- 设备端 `subscene_mask_tabular`
- `vert_is_active_contact`
- `vert_disp_norms`

### 工程意义

它把“接触系数”和“场景拓扑关系”从求解器核心里抽离出来，做成了统一服务。

## 6.2 contact model 层

从目录能确认存在：

- `ipc_simplex_normal_contact`
- `ipc_simplex_frictional_contact`
- `ipc_vertex_half_plane_normal_contact`
- `ipc_vertex_half_plane_frictional_contact`
- codim 版本
- analytical barrier `pFpx` 版本

可推断：

- normal contact 负责 barrier 法向能、梯度、Hessian
- frictional contact 负责切向摩擦项
- codim / pFpx 是不同 barrier 或不同 primitive 组合的实现分支

这里的“推断”来自模块命名与标准 IPC 分层，但整体方向很清楚。

## 7. 仿真主循环

## 7.1 `SimEngine::advance()`

源码中的顺序非常明确：

```text
1. rebuild scene
2. update diff params
3. process external changes
4. record previous positions
5. animator + external force
6. predict dof
7. detect DCD candidates
8. adaptive contact parameters
9. Newton loop
10. update velocity/state
```

Newton 内部再细分为：

```text
for newton_iter:
    compute animation substep ratio
    if iter > 0: detect DCD candidates
    compute dy-topo effect gradient/Hessian
    solve global linear system
    collect vertex displacements

    alpha = 1
    detect trajectory candidates(alpha)
    E0 = compute_energy(current)
    alpha = filter_toi(alpha)
    alpha = cfl_condition(alpha)

    for line_search_iter:
        E = compute_energy(alpha)
        if converged or E <= E0:
            break
        alpha /= 2
```

## 7.2 `SimEngine::advance_AL()`

差异主线：

- 先线性化接触约束
- active set 管理 slack 和非穿透位置恢复
- 收敛更关注 `beta` 和 TOI threshold

因此 AL 路线更像“约束系统 + line search”，而基础 IPC 更像“纯 barrier 能量最小化”。

## 8. 关键函数输入输出

## 8.1 `SimEngine::do_advance()`

输入：

- 当前 `m_pipeline_type`

输出：

- 分派到 `advance()` 或 `advance_AL()`

## 8.2 `SimEngine::advance()`

输入：

- Scene config
- 全局系统注册结果
- 当前 frame 状态

输出：

- 新 frame 的 DOF、速度、接触状态、统计

## 8.3 `GlobalTrajectoryFilter::detect(alpha)`

输入：

- `alpha`
- 所有注册的轨迹 filter

输出：

- 候选接触对与后续 TOI 数据

## 8.4 `GlobalTrajectoryFilter::filter_toi(alpha)`

输入：

- 当前 `alpha`

输出：

- 所有 filter 的最小 TOI

## 8.5 `GlobalContactManager::init(world)`

输入：

- world/scene 的 contact tabular、subscene tabular

输出：

- 设备端 contact matrix
- 设备端 mask matrix
- 顶点接触状态缓冲

## 8.6 `GlobalContactManager::compute_cfl_condition()`

输入：

- 活跃接触顶点标志
- 顶点位移
- `d_hat`

输出：

- `alpha_cfl`

## 8.7 `FEMLinearSubsystem` / `ABDLinearSubsystem`

确认点：

- 两者都存在独立线性子系统和对角/预条件器模块
- 都通过 reporter 接入全局线性系统

推断：

- 它们分别负责把 FEM 与 ABD 的局部 Hessian/梯度块汇入全局系统
- `coupling_system/` 负责跨系统块装配

## 9. 输入输出视角

### 9.1 输入

- Scene graph
- geometry + material + constitution
- contact tabular / subscene tabular
- `dt`、`d_hat`、`eps_velocity`
- 可选动画、外力、差分参数

### 9.2 运行时状态

- 全局顶点位置/位移/速度
- active contact mask
- trajectory candidates
- contact coefficient tables
- 各 subsystem 的局部状态

### 9.3 输出

- 更新后的场景状态
- contact report / exporter
- active set / friction candidates
- 线搜索与牛顿统计
- 可选 diff sim 参数梯度

## 10. 实现特征总结

- 优点：最现代、最模块化、最适合大系统和多对象耦合。
- 局限：阅读门槛高，单看一个 `.cu` 很难懂全局，需要按系统看调用图。
- 最值得学的部分：
  - 如何把 IPC 做成“可插拔 GPU 后端”
  - 如何用 `GlobalTrajectoryFilter` 和 `GlobalContactManager` 解耦碰撞与接触
  - 如何在同一引擎里并存基础 IPC 和 AL-IPC 两条管线
