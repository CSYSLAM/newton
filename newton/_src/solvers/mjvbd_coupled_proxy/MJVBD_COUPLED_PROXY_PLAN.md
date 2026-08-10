# SolverMJVBDCoupledProxy 设计与实施计划

## 1. 状态

本文档记录 `SolverMJVBDCoupledProxy` 的设计方案。当前阶段仅创建独立目录和
计划文档，尚未实现、导出或注册任何新求解器。

第一阶段不修改以下现有实现：

- `SolverCoupled`
- `SolverCoupledProxy`
- `SolverCoupledADMM`
- `SolverMJVBDV2`

新实现可以复用它们已有的 ownership、coupling hook、碰撞管线和基础求解器，
但不能复制通用多 entry 调度代码后再维护一份重复版本。

## 2. 目标

新增固定 MuJoCo 与 VBD 两个物理域的专用组合求解器：

```python
SolverMJVBDCoupledProxy(
    model,
    *,
    mujoco_articulations=None,
    mujoco_joints=None,
    joint_mode="dynamic",
    coupling_mode="one_way",
    proxy_mode="staggered",
    coupling_iterations=1,
    proxy_relaxation=1.0,
    contact_mode="auto",
    mujoco_options=None,
    vbd_options=None,
    collision_options=None,
)
```

状态所有权固定为：

| 对象 | 动力学所有者 | 在 VBD 中的角色 |
| --- | --- | --- |
| 选中的 articulation、关节及 link | MuJoCo | 移动 proxy 碰撞体 |
| 非 MuJoCo 动态刚体 | VBD/AVBD | 动态刚体 |
| 软体 | VBD | 动态粒子与四面体 |
| 布料 | VBD | 动态粒子、三角形与弯曲边 |
| 静态环境 | 不积分 | 静态碰撞体 |

VBD 内部只使用一个 VBD 实例，刚体、软体和布料不再拆成不同 entry，因此
以下相互作用必须保持 VBD 原生双向耦合：

- 刚体与刚体；
- 刚体与软体；
- 刚体与布料；
- 软体与布料；
- 软体/布料自接触。

MuJoCo 与 VBD 之间支持：

- `one_way`：MuJoCo link 影响 VBD，VBD 不反馈 MuJoCo；
- `two_way`：VBD 接触 wrench 反馈给 MuJoCo，并支持固定点迭代与松弛。

第一版不考虑 ADMM、多于两个 entry、任意 solver 组合或粒子 proxy。

## 3. 语义约束

### 3.1 单向耦合

单向模式不允许执行或分配以下功能：

- coupling-force stash；
- proxy rewind；
- contact-wrench harvest；
- Aitken 或固定 relaxation；
- MuJoCo feedback buffer。

MuJoCo 可以是动态或运动学模式。动态单向模式仍推进 MuJoCo，只忽略来自
VBD 的接触反作用。

### 3.2 双向耦合

双向模式复用 `SolverCoupledProxy` 与 VBD 已有的 coupling hooks。对于
`coupling_iterations=1`，当前步产生的反馈在下一外层时间步作用于 MuJoCo；
迭代数大于一时，在同一时间步内进行固定点迭代。

第一版拒绝 `joint_mode="kinematic"` 与 `coupling_mode="two_way"` 的组合。
运动学轨迹不会被反馈力改变，允许该组合会产生误导性语义。后续如需要反力
诊断，可以单独增加只报告、不反馈的接口。

### 3.3 结果等价

优化实现不要求逐 bit 相同，但必须满足：

- 单向结果在容差内对齐 `SolverMJVBDV2`；
- 双向结果在容差内对齐相同设置的 `SolverCoupledProxy`；
- 接触种类、有效接触数量和力方向一致；
- VBD 内部刚体、软体、布料相互作用不因分派而降级。

## 4. 特征分析

构造时只进行一次场景分析，并缓存不可变的 `Features`：

```python
@dataclass(frozen=True)
class Features:
    backend: str
    mujoco_joint_count: int
    mujoco_body_count: int
    vbd_dynamic_body_count: int
    particle_count: int
    triangle_count: int
    edge_count: int
    tetrahedron_count: int
    spring_count: int
    mujoco_solve_enabled: bool
    rigid_solve_enabled: bool
    particle_solve_enabled: bool
    triangle_solve_enabled: bool
    bending_solve_enabled: bool
    tetrahedron_solve_enabled: bool
    spring_solve_enabled: bool
    two_way_feedback_enabled: bool
```

定义：

- `J`：MuJoCo-owned joint 数量；
- `R`：VBD-owned 动态刚体数量，静态 body 不计入；
- `P`：VBD-owned particle 数量；
- `Tri`、`Edge`、`Tet`、`Spring`：对应约束拓扑数量。

## 5. 后端分派矩阵

| 特征 | 后端 | 必须完全跳过 |
| --- | --- | --- |
| `J=0, R=0, P=0` | no-op | MuJoCo、VBD、碰撞 |
| `J>0, R=0, P=0` | MuJoCo-only | VBD、外部 CollisionPipeline |
| `J=0, R=0, P>0` | VBD-soft-only | MuJoCo、VBD rigid |
| `J=0, R>0, P=0` | VBD-rigid-only | MuJoCo、particle/cloth/tet |
| `J=0, R>0, P>0` | VBD-full | MuJoCo |
| `J>0, R=0, P>0`，kinematic one-way | MJVBD soft fast path | MuJoCo step、VBD rigid |
| `J>0, R>0`，kinematic one-way | MJVBD full fast path | MuJoCo step |
| `J>0` 且存在 VBD workload，dynamic one-way | fixed one-way proxy | rewind、harvest、feedback |
| `J>0` 且存在 VBD workload，dynamic two-way | fixed two-way proxy | 通用多 entry 调度 |

约束子路径必须按拓扑继续裁剪：

- 只有布料：只启动 particle、triangle、edge；
- 只有四面体软体：只启动 particle、tet；
- 只有弹簧粒子：只启动 particle、spring；
- 只有刚体：只启动 AVBD rigid；
- 刚体与任意粒子对象并存：使用完整 VBD 和完整接触路径。

`contact_mode="auto"` 根据 `R` 和 `P` 选择 soft-only 或 full collision。

## 6. 建议架构

```text
SolverMJVBDCoupledProxy
├── FeatureResolver
├── NoOpBackend
├── MuJoCoOnlyBackend
├── VBDOnlyBackend
├── MJVBDV2Backend
│   └── 复用现有 pure-VBD 与 kinematic one-way 快路径
└── FixedMuJoCoVBDProxyBackend
    ├── dynamic one-way fixed schedule
    └── dynamic two-way fixed schedule
```

优先复用：

- `mjvbd_v2.ownership.resolve_ownership`；
- `SolverMJVBDV2` 已有 pure-VBD 与 kinematic 快路径；
- `SolverCoupledProxy` 已有 coupling 数据结构、状态同步 kernel 和 hook 语义；
- `SolverVBD`/`SolverVBDSoft`；
- `SolverMuJoCo`；
- `MJVBDSoftContactPipeline`、`MJVBDV2SoftContactPipeline` 和
  `CollisionPipeline`。

不能为了“独立”而复制 `SolverCoupled`、`SolverCoupledProxy` 的通用实现。
固定 dynamic backend 可以基于 `SolverCoupledProxy` 完成构建和 coupling hook
准备，但必须使用专用 step 调度，绕过通用 entry 字典循环、全量 state
distribution、contact filtering 和重复 reconcile。

## 7. 单向固定调度

动态单向每步数据流：

```text
state_in
  |
  |-- 1. 只同步 MuJoCo 所需 joint/body/control 输入
  |-- 2. MuJoCo step
  |-- 3. 融合 kernel 同步 proxy body q/qd 到 VBD input
  |-- 4. 一次接触生成
  |-- 5. VBD step
  `-- 6. 一次性合并 MuJoCo-owned 与 VBD-owned 输出
```

该路径不能调用 harvest、rewind 或 relaxation，即使底层 solver 提供对应 hook。

运动学单向优先直接委托给 `SolverMJVBDV2` 已有的 soft/full 快路径，避免再次
实现同一套浅层 Model overlay 和碰撞调度。

## 8. 双向固定调度

```text
保存顶层输入状态

for coupling iteration:
    恢复本轮所需的 MuJoCo/VBD 输入
    将上一轮 VBD feedback 施加到 MuJoCo
    MuJoCo step
    同步 proxy body q/qd
    刷新接触
    VBD step
    harvest VBD contact wrench
    relaxation

最终只 reconcile 一次
```

优化要求：

- entry 和 proxy 引用在构造时缓存，step 中不查字典；
- 只保留一个 MuJoCo→VBD body proxy group；
- 只复制选中 joint、MuJoCo body、VBD dynamic body 和 particle 状态；
- coupling iteration 之间不重新分发完全不变的静态数据；
- collision refresh cadence 显式配置，默认每轮刷新；
- `staggered` 使用 MuJoCo end pose；
- `lagged` 保留 begin-pose/rewind 语义；
- feedback、previous feedback 和 relaxation buffer 均持久化。

## 9. 空分支与内存要求

以下条件必须由构造测试和运行 profile 同时验证：

- 无关节时不构建 `SolverMuJoCo`、MuJoCo state 或 MuJoCo view；
- 只有关节时不构建 VBD、VBD state 或外部 collision pipeline；
- 无动态 VBD 刚体时不分配 AVBD body Hessian、body-body contact buffer；
- 无粒子时不分配 particle force/Hessian、自接触和 soft-contact buffer；
- 无三角形时不启动 triangle kernel；
- 无 edge 时不启动 bending kernel；
- 无 tet 时不启动 tet kernel；
- 无 spring 时不启动 spring kernel；
- soft-only 不运行 rigid-rigid broad phase；
- rigid-only 不生成 soft contact；
- step 内没有 `.numpy()`、host synchronization 或新 Warp array 分配。

所有数组、index map、mask、contacts 和 scratch state 在构造阶段分配。目标是
支持 CUDA graph capture。

## 10. 文件规划

第一阶段计划只新增：

```text
newton/_src/solvers/mjvbd_coupled_proxy/
├── __init__.py
├── features.py
├── backends.py
├── kernels.py
├── solver.py
└── MJVBD_COUPLED_PROXY_PLAN.md

newton/mjvbd_coupled_proxy.py
newton/tests/test_solver_mjvbd_coupled_proxy.py
asv/benchmarks/simulation/bench_mjvbd_coupled_proxy.py
```

在不修改 `newton/solvers.py` 的阶段，临时公共导入路径为：

```python
from newton.mjvbd_coupled_proxy import SolverMJVBDCoupledProxy
```

性能和行为验收通过后，再单独决定是否加入 `newton.solvers` 公共导出、API
文档和 changelog。该决定不属于第一阶段。

## 11. 测试计划

全部使用 `unittest`，覆盖：

1. no-op；
2. only joints；
3. only rigid；
4. only cloth；
5. only tetrahedral soft body；
6. only spring particles；
7. rigid + cloth；
8. rigid + soft；
9. joints + cloth；
10. joints + rigid；
11. joints + rigid + cloth + soft；
12. dynamic one-way；
13. dynamic two-way；
14. kinematic one-way；
15. reset、notify-model-changed 和 BVH rebuild。

特征分派测试必须通过 mock/factory 计数证明无关 solver 没有被构建，而不只是
检查某个运行时布尔值。

数值对照：

- MuJoCo-only 对照直接 `SolverMuJoCo`；
- VBD-only 对照直接 `SolverVBD`/`SolverVBDSoft`；
- kinematic one-way 对照 `SolverMJVBDV2`；
- dynamic one-way/two-way 对照相同配置的 `SolverCoupledProxy`；
- 混合 VBD 场景验证等量反向响应、接触数和最终状态容差。

## 12. 性能验收

新增独立 ASV/CUDA benchmark，矩阵至少包括：

- only joints；
- only rigid；
- only cloth；
- joints + cloth；
- joints + rigid；
- joints + rigid + cloth/soft；
- dynamic one-way；
- dynamic two-way。

每组对比：

- 直接 `SolverMuJoCo` 或 `SolverVBD`；
- `SolverMJVBDV2`；
- 通用 `SolverCoupledProxy`；
- `SolverMJVBDCoupledProxy`。

记录：

- eager 与 CUDA graph 的 ms/step；
- kernel launch 数量；
- state copy/scatter 数量；
- 每步显存分配；
- collision、MuJoCo、VBD 和 coupling layer 分项耗时。

初始性能门槛：

- 纯 MuJoCo/VBD 路径相对直接求解器开销不超过约 5%；
- kinematic one-way 不慢于对应 MJVBDV2 快路径；
- fixed dynamic proxy 必须降低通用 Proxy 的 coupling kernel 数和 copy 数；
- 总耗时由 VBD 主导时，单独报告 coupling-layer 加速，不能只报告整体比例。

具体加速倍数在获得 CUDA baseline 后确定，不在实现前作无数据保证。

## 13. 实施顺序

1. 固定 API、错误组合和数值容差。
2. 实现 feature resolver 与 backend matrix。
3. 实现 no-op、MuJoCo-only、VBD-only。
4. 复用 MJVBDV2 pure/kinematic one-way 快路径。
5. 实现 dynamic one-way fixed schedule。
6. 实现 dynamic two-way feedback loop。
7. 完成特征矩阵和数值等价测试。
8. 建立通用 Proxy 与新求解器 CUDA baseline。
9. 根据 profiler 融合同步/scatter kernel 并消除剩余分配。
10. 性能验收后再讨论公共导出。

## 14. 完成定义

只有同时满足以下条件才视为完成：

- 所有后端分派测试通过；
- 单向与双向数值对照通过；
- VBD 内部混合对象保持双向耦合；
- 空功能分支没有对应 solver、buffer 或 kernel；
- step 无 host round-trip 和动态分配；
- CUDA benchmark 达到约定门槛；
- 现有 SolverCoupledProxy、SolverCoupledADMM 和 SolverMJVBDV2 文件未被修改。
