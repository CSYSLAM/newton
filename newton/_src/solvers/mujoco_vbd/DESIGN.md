# SolverMuJoCoVBD：独立高性能多功能求解器设计

## 0. 文档状态

本文是 `SolverMuJoCoVBD` 第一版的完整实现规格，不是概念草案或最小
可行版本。第一版只有在本文列出的数值闭环、接触类型、状态恢复、性能路径、
测试矩阵和验收门槛全部完成后才可以对外使用。

本设计的核心目标是：

> 在不依赖仓库已有 MuJoCo solver、VBD solver、`SolverCoupled`、
> `SolverCoupledProxy`、通用 `Entry` 或运行时 `ModelView` 调度的前提下，
> 在一个可独立迁移的目录中同时提供 pure VBD、pure MuJoCo、运动学直通、
> MuJoCo→VBD 单向 moving-collider 耦合和 MuJoCo↔VBD 双向耦合。

这里的“通用”限定在该独立 solver 自带的 **私有 MuJoCo + 私有 VBD/AVBD**
体系内：既能单独进行刚体、布料、tet 软体、弹簧和气动仿真，也支持任意数量
机器人 articulation、多 world 以及它们之间的单向或双向接触。它不承担把任意
两个外部 Newton solver 动态组合起来的职责。

本文只规定实现，不在本文件中实现求解器代码。

## 1. 不可退让的第一版要求

第一版必须同时满足以下要求：

1. 私有 MuJoCo、私有完整 VBD、私有轻量 VBD 和私有 contact pipeline 全部位于
   本目录，运行时不导入仓库其他 MuJoCo/VBD solver。
2. pure、passthrough、单向和双向分支在构造期静态选择；未使用的核心和 buffer
   不得构造。
3. 双向模式下 MuJoCo 与 VBD 之间存在真实的 equal-and-opposite contact feedback，
   机器人能够因 VBD 接触反力改变位置、速度和关节运动。
4. 单向模式必须独立复现 MJVBD_V2 的所有功能和数值合同，但不能运行时依赖它。
5. 单向 source 在 VBD 中必须是零逆质量 moving collider；VBD 不得改变其轨迹，
   也不得向 MuJoCo 分配或回传反力。
6. 支持 point、edge、face 三类统一 rigid-soft 接触；启用 full-surface contact
   时不能因为处于双向模式而拒绝或漏掉 edge/face feedback。
7. 支持机器人与 VBD 动态刚体的 rigid-rigid 接触反馈。
8. 双向模式每个 coupling iteration 重新计算跨 solver 接触，不能复用第一轮的过期
   narrow-phase 结果。
9. 多轮 coupling 必须从同一 substep 初始状态重算，不能把一个 substep
   错误积分多次。
10. 只有最后一轮才能提交 MuJoCo warm start、VBD contact history、DAT、Dahl
   friction、stick/slip 和 pneumatic history。
11. CUDA 路径必须支持固定拓扑的整 substep CUDA Graph。
12. CPU 路径必须具备同样的物理语义，用作调试和正确性基线。
13. deterministic 模式必须有稳定的接触排序和稳定归约路径。
14. 所有容量固定，使用 device-side count，并保留显式 overflow flag。
15. 运行时不允许依赖 MJVBD_V2、`SolverCoupledProxy` 或从它们调用状态机。

## 2. 物理合同

### 2.1 支持的机器人模式

真正的双向耦合要求 MuJoCo articulation 具有有限质量和有限控制刚度。支持：

- dynamic joint；
- torque control；
- velocity actuator；
- 有限增益 position servo；
- fixed-base 和 floating-base articulation。

hard kinematic body 等价于无限质量 source，接触反力不可能改变其公开轨迹。
它可以用于 passthrough 和单向模式；构造时若同时请求双向反馈和 hard
kinematic articulation，必须直接抛出 `ValueError`，不能静默退化成单向，也不能
在内部偷偷执行 IK。希望机器人跟随轨迹但仍响应接触时，应使用有限刚度 MuJoCo
servo。

单向模式只有一种 VBD proxy 响应：source pose/velocity 被同步到 VBD，proxy 的
solve-time inverse mass/inertia 为零。它是与 MJVBD_V2 一致的 moving collider；公开
模型中的非零质量/惯量元数据仍保留，不能通过覆盖公开 `body_mass` 实现零质量。
单向 backend 不分配 reaction、effective-mass、relaxation 或 outer-iteration 状态。

双向模式固定使用有限质量 `EFFECTIVE_MASS` proxy 和 feedback iteration。每轮 VBD
接触反力都以等大反向 wrench 回传 MuJoCo，MuJoCo 从同一 substep 初始状态重新求解，
再把新机器人状态同步到 VBD。有限质量 proxy 不允许出现在无反馈的单向模式中。

若机器人必须严格跟随目标轨迹同时表现柔顺性，轨迹应写入 MuJoCo 的有限刚度
position/velocity servo，并选择 `two_way`。不得让单向 proxy 局部运动后丢弃其状态
或反力，也不得在 VBD 求解后用隐藏 IK 覆盖物理解。

### 2.2 每个 substep 的耦合方程

令 `x0` 为本 substep 的共同初始状态，`W[k]` 为第 `k` 轮传给 MuJoCo 的
relaxed proxy wrench：

```text
M[k]     = MuJoCo(x0, control, W[k-1], dt)
proxy[k] = sync(FK(M[k]))
V[k]     = VBD(x0, proxy[k], contacts(proxy[k]), dt)
Wraw[k]  = -harvest_cross_contact_wrench(V[k])
W[k]     = relax(W[k-1], Wraw[k])
```

最终公开状态是同一轮的 `M[K]` 和 `V[K]`。最后收集到的 `W[K]` 用于 residual
报告和下一 substep warm start；它不允许触发额外的“半轮”MuJoCo step，因为那会
使公开机器人状态和 VBD 状态对应不同的边界位置。

### 2.3 力与力矩约定

所有 interface wrench 使用 world frame、以 body COM 为参考点的
`wp.spatial_vector`：

```text
wrench[0:3] = linear force [N]
wrench[3:6] = torque about COM [N m]
```

若 VBD 对软体接触点施加 `f_soft`，则机器人刚体收到：

```text
f_body   = -f_soft
tau_body = (contact_point_world - body_com_world) x f_body
```

刚体接触同理。写入 MuJoCo 的 wrench 与 VBD 侧作用严格等大反向。

## 3. 依赖边界

### 3.1 禁止依赖

`mujoco_vbd/` 下任何运行时代码都不得导入：

```text
newton._src.solvers.coupled.solver_coupled
newton._src.solvers.coupled.solver_coupled_proxy
newton._src.solvers.coupled.model_view
newton._src.solvers.coupled.proxy_utils
newton._src.solvers.mjvbd_v2
newton._src.solvers.solver_mujoco
newton._src.solvers.solver_vbd
```

也不得创建通用 `Entry`、proxy group 字典或按 solver 名称分派。生产代码的
AST/import 测试必须阻止上述依赖重新进入。

### 3.2 私有核心复制合同

第一版以当前 `mjvbd_v2_pneumati` 分支为基线，把以下实现复制进本目录并修正
相对导入：

- 私有 MuJoCo solver、kernels、constants、enums、equality 和 utils；
- 私有完整 VBD/AVBD solver 及 particle/rigid/pneumatic/coupling kernels；
- 私有轻量 `vbd_soft` solver 及其全部 kernels；
- 私有 soft contact、full contact 和 point contact pipeline；
- 私有 ownership、dispatch、pneumatic state transfer 和 reset 逻辑；
- MJVBD_V2 已保留的性能快路径。

当前 baseline core 仍有对 `coupled.interface.CouplingInterface`、
`CouplingEndpointKind` 和 `SolverCoupled` 的导入。复制时必须：

1. 把仅为类型/固定 endpoint 查询所需的 enum 和 protocol 重写到本目录
   `coupling_types.py`；
2. 将 private MuJoCo/VBD core 的这些 import 改为相对导入；
3. 删除 `CouplingInterface` 基类要求中本求解器未使用的通用 entry/group 能力；
4. 将 MJVBD_V2 中依赖 `SolverCoupled`/`SolverCoupledProxy` 的 orchestration 直接重写为
   本文各 backend 的固定调用；
5. 用 AST 测试确认私有 core 内也没有残留的 `coupled` import。

复制不是临时 vendor shim。复制完成后，新 solver 只能导入本目录中的
`mujoco.SolverMuJoCo`、`vbd.SolverVBD` 和 `vbd_soft.SolverVBD`，后续独立修改、
测试和优化。仓库其他 solver 的变化不会自动改变它的数值结果。

`PRIVATE_BASELINE.md` 必须记录：

```text
source branch and commit
copied source paths
copy date
file hashes immediately after copy
intentional differences
later manually ported fixes and their source commits
```

允许继续依赖 Newton 的基础数据模型和稳定几何层，如 `Model`、`State`、`Contacts`、
`ModelBuilder`、Warp 数学函数和通用 shape SDF evaluator；不允许依赖其他 solver
来执行 MuJoCo 或 VBD 核心步骤。

### 3.3 优化迁移规则

以下 MJVBD_V2 retained 优化属于第一版必迁项，而不是可选后续工作：

- shape-major full-surface AABB rejection；
- AABB-active edge/face compaction 和 persistent worker；
- dense rigid-side body-particle parallel reduction；
- particle-color contact membership mask；
- active self-contact record traversal；
- surface-only CUDA tile specialization；
- device-resident material selector；
- world-compatible contact capacity sizing；
- active-prefix/batch-gated dual update；
- pneumatic device state/control；
- one-stream whole-frame CUDA Graph 所需的固定 topology 支持。

优化日志中已判定无收益并回退的方案不得顺便恢复，包括：

- canonical EE pair；
- post-detection VT/EE active stream；
- rest-shape exclusion CSR；
- Morton query order；
- source-color EE row gate。

## 4. 目标目录结构

```text
newton/_src/solvers/mujoco_vbd/
├── __init__.py
├── config.py
├── ownership.py
├── dispatch.py
├── coupling_types.py
├── model_overlay.py
├── contact_routing.py
├── state.py
├── kernels.py
├── effective_mass.py
├── feedback.py
├── convergence.py
├── collision_pipeline.py
├── full_contact_pipeline.py
├── soft_contact_pipeline.py
├── point_contact_kernels.py
├── backend_mujoco.py
├── backend_vbd.py
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── pure_mujoco.py
│   ├── kinematic_passthrough.py
│   ├── pure_vbd.py
│   ├── one_way_kinematic_soft.py
│   ├── one_way_kinematic_full.py
│   ├── one_way_dynamic_soft.py
│   ├── one_way_dynamic_full.py
│   └── two_way.py
├── mujoco/
│   ├── __init__.py
│   ├── constants.py
│   ├── enums.py
│   ├── equality.py
│   ├── kernels.py
│   ├── solver_mujoco.py
│   └── utils.py
├── vbd/
│   ├── __init__.py
│   ├── particle_vbd_kernels.py
│   ├── pneumatic.py
│   ├── pneumatic_kernels.py
│   ├── rigid_vbd_kernels.py
│   ├── solver_vbd.py
│   ├── tri_mesh_collision.py
│   └── vbd_coupling_kernels.py
├── vbd_soft/
│   ├── __init__.py
│   ├── particle_vbd_kernels.py
│   ├── rigid_vbd_kernels.py
│   ├── solver_vbd.py
│   ├── tri_mesh_collision.py
│   └── vbd_coupling_kernels.py
├── solver.py
├── diagnostics.py
├── PRIVATE_BASELINE.md
├── OPTIMIZATION_LOG.md
└── DESIGN.md
```

职责如下：

- `config.py`：公开 enum、dataclass 及参数验证。
- `ownership.py`：MuJoCo/VBD/static 实体所有权和固定映射。
- `dispatch.py`：构造期 feature discovery 和静态 backend 选择。
- `coupling_types.py`：本目录私有 endpoint enum、固定 adapter protocol 和 state-transfer
  类型；不包含通用 entry/group dispatcher。
- `model_overlay.py`：构造两个浅层 model overlay，不做运行时属性代理。
- `contact_routing.py`：唯一接触所有权和 pair mask。
- `state.py`：私有 state、substep snapshot、iteration scratch、history commit。
- `kernels.py`：固定映射 copy、sync、restore、wrench scatter 和辅助 kernel。
- `effective_mass.py`：MuJoCo articulated effective mass 到 VBD proxy inertia。
- `feedback.py`：rigid 和统一 point/edge/face contact wrench harvest。
- `convergence.py`：fixed/Aitken relaxation、residual 和发散保护。
- `collision_pipeline.py`：跨 solver 专用 contact pipeline 封装。
- `backend_mujoco.py`：MuJoCo begin/restore/solve/commit 固定 adapter。
- `backend_vbd.py`：VBD begin/restore/solve/commit 固定 adapter。
- `backends/`：互斥且干净的 pure、单向和双向执行分支。
- `mujoco/`：完全私有的 MuJoCo core，不导入仓库其他 MuJoCo solver。
- `vbd/`：完全私有的完整 VBD/AVBD/pneumatic core。
- `vbd_soft/`：完全私有的软体/布料轻量 VBD core。
- `solver.py`：公开 `SolverMuJoCoVBD` 和完整 substep 调度。
- `diagnostics.py`：overflow、residual、penetration 和 timing 状态。
- `PRIVATE_BASELINE.md`：复制基线和手工同步记录。
- `OPTIMIZATION_LOG.md`：该 solver 独立的性能决策记录。

### 4.1 静态 backend 集合

第一版不是让一个大类在每个 `step()` 内判断所有模式，而是在构造期选择以下一个
backend：

| Backend | 触发拓扑/模式 | 实际构造内容 |
| --- | --- | --- |
| `pure_vbd_soft` | 无 MuJoCo joint；只有 particle/cloth/tet/spring；无 dynamic VBD rigid 和 pneumatic | 私有 `vbd_soft` + sparse/full particle contact pipeline |
| `pure_vbd_full` | 无 MuJoCo joint；存在 VBD dynamic rigid、pneumatic，或 rigid-only | 私有完整 VBD/AVBD + 构造期选定的 soft/full pipeline |
| `kinematic_passthrough` | 只有 externally prescribed articulation，无 VBD dynamics | 不构造 MuJoCo/VBD/contact |
| `pure_mujoco` | 只有 dynamic MuJoCo articulation，无 VBD dynamics | 只构造私有 MuJoCo |
| `one_way_kinematic_soft` | kinematic source + particle-only VBD，soft/point contact，immovable proxy | 不构造 MuJoCo；无 pneumatic 时用私有 `vbd_soft`，有 pneumatic 时用私有完整 VBD；均使用 sparse contact |
| `one_way_kinematic_full` | kinematic source + full contact/VBD rigid，immovable proxy | 不构造 MuJoCo；私有完整 VBD，source body solve inverse mass 为零 |
| `one_way_dynamic_soft` | MuJoCo dynamic source + particle-only VBD，单步驱动且不回传 | 私有 MuJoCo 一次 + 私有轻量 VBD 一次；不分配 full rigid/feedback/Aitken |
| `one_way_dynamic_full` | MuJoCo dynamic source + VBD dynamic rigid 或 pneumatic，单步驱动且不回传 | 私有 MuJoCo 一次 + 私有完整 VBD 一次；不分配 feedback/Aitken |
| `two_way` | dynamic/compliant MuJoCo source 与 VBD 双向 | 私有 MuJoCo + 完整 VBD + K 轮 contact-native feedback |

“干净分支”的强制含义：

- `pure_vbd_*` 不构造 MuJoCo model/data、MuJoCo state 或 coupling wrench；
- `pure_mujoco` 不构造 VBD、BVH、soft contact 或 coupling state；
- `kinematic_passthrough` 不构造任何 solver；
- immovable 单向分支不构造 effective mass、feedback、Aitken 或 transaction iteration
  scratch；
- `two_way` 才构造完整 iteration snapshot、effective mass、wrench 和 convergence
  buffer。

### 4.2 Feature discovery

`dispatch.py` 定义：

```python
class MuJoCoVBDBackendKind(Enum):
    PURE_VBD_SOFT = "pure_vbd_soft"
    PURE_VBD_FULL = "pure_vbd_full"
    KINEMATIC_PASSTHROUGH = "kinematic_passthrough"
    PURE_MUJOCO = "pure_mujoco"
    ONE_WAY_KINEMATIC_SOFT = "one_way_kinematic_soft"
    ONE_WAY_KINEMATIC_FULL = "one_way_kinematic_full"
    ONE_WAY_DYNAMIC_SOFT = "one_way_dynamic_soft"
    ONE_WAY_DYNAMIC_FULL = "one_way_dynamic_full"
    TWO_WAY = "two_way"


@dataclass(frozen=True)
class MuJoCoVBDFeatures:
    backend: MuJoCoVBDBackendKind
    coupling_mode: str
    joint_mode: str
    contact_mode: str
    vbd_core: Literal["none", "soft", "full"]
    contact_pipeline: Literal["none", "soft", "full"]
    mujoco_joint_count: int
    mujoco_body_count: int
    vbd_body_count: int
    vbd_dynamic_body_count: int
    particle_count: int
    triangle_count: int
    edge_count: int
    tetrahedron_count: int
    spring_count: int
    pneumatic_cavity_count: int
    pneumatic_face_count: int
    mujoco_solve_enabled: bool
    vbd_solve_enabled: bool
    rigid_solve_enabled: bool
    particle_solve_enabled: bool
    triangle_solve_enabled: bool
    bending_solve_enabled: bool
    tetrahedron_solve_enabled: bool
    spring_solve_enabled: bool
    pneumatic_solve_enabled: bool
    feedback_enabled: bool
    effective_mass_enabled: bool
    iteration_transaction_enabled: bool
```

```python
def discover_features(
    model: Model,
    ownership: MuJoCoVBDOwnership,
) -> MuJoCoVBDFeaturesInput:
    """Count topology and dynamics without constructing a backend."""


def select_backend_kind(
    discovered: MuJoCoVBDFeaturesInput,
    *,
    joint_mode: str,
    coupling_mode: str,
    contact_mode: str,
) -> MuJoCoVBDBackendKind:
    """Return exactly one construction-time backend or raise on contradiction."""


def build_backend(
    kind: MuJoCoVBDBackendKind,
    model: Model,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDResolvedOptions,
) -> SolverBase:
    """Construct only the modules required by the selected kind."""
```

### 4.3 Mode 选择规则

`coupling_mode="auto"` 只解析一次：

```text
no MuJoCo joints                         -> pure VBD
no VBD dynamics + kinematic joints       -> kinematic passthrough
no VBD dynamics + dynamic joints         -> pure MuJoCo
mixed + kinematic joints                 -> one_way
mixed + dynamic/compliant joints         -> two_way
```

显式 `coupling_mode="one_way"` 在 dynamic source 上按 VBD/contact feature 选择
`one_way_dynamic_soft` 或 `one_way_dynamic_full`，在
kinematic source 上根据 contact/topology 选择 soft 或 full kinematic backend；两者都必须
使用零逆质量 moving collider，并退化为独立复刻的 MJVBD_V2 路径。

显式 `coupling_mode="two_way"` 要求：

- 至少一个 MuJoCo articulation；
- 至少一个 VBD dynamic body 或 particle；
- source 不是 hard kinematic；
- 使用完整 VBD；
- feedback 与 iteration transaction 均启用。

`contact_mode="auto"` 的规则：

- pure/one-way immovable 与 MJVBD_V2 一致：存在 VBD dynamic rigid 时选 full，
  否则选 soft；
- 用户可以在 particle-only 场景显式选 soft，但此时没有 edge/face contact；
- 任何包含 VBD dynamic rigid 的场景显式选 soft 都报错。

`vbd_core` 与 `contact_pipeline` 必须独立解析。特别是 kinematic source + particle-only
pneumatic bag 应得到：

```text
backend          = one_way_kinematic_soft
vbd_core         = full
contact_pipeline = soft
```

这是 MJVBD_V2 当前 pneumatic kinematic-soft 分支的必要等价行为；不能因为 pipeline
名为 soft 就错误构造不支持 pneumatic 的 `vbd_soft` core。

## 5. 公共 API 规格

### 5.1 配置类型

`config.py` 提供以下 enum。公开 enum 使用 Python `Enum`；传入 Warp kernel 前
转换为稳定的整数常量。

```python
class MuJoCoVBDCouplingMode(Enum):
    AUTO = "auto"
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"


class MuJoCoVBDRelaxation(Enum):
    FIXED = "fixed"
    AITKEN = "aitken"


class MuJoCoVBDStaticContactOwner(Enum):
    AUTO = "auto"
    MUJOCO = "mujoco"
    VBD = "vbd"
```

完整 coupling 配置：

```python
@dataclass(frozen=True)
class MuJoCoVBDCouplingOptions:
    iterations: int = 4
    relaxation: MuJoCoVBDRelaxation | str = MuJoCoVBDRelaxation.AITKEN
    relaxation_initial: float = 0.5
    relaxation_min: float = 0.05
    relaxation_max: float = 1.0
    force_absolute_tolerance: float = 1.0e-3
    force_relative_tolerance: float = 1.0e-3
    velocity_tolerance: float = 1.0e-4
    proxy_mass_scale: float = 1.0
    proxy_mass_min: float = 1.0e-6
    proxy_mass_max: float = 1.0e6
    proxy_inertia_eigenvalue_min: float = 1.0e-8
    proxy_inertia_eigenvalue_max: float = 1.0e8
    warm_start_wrench: bool = True
    deterministic: bool = False
    fail_on_overflow: bool = True
    fail_on_nonfinite: bool = True
    static_contact_owner: MuJoCoVBDStaticContactOwner | str = MuJoCoVBDStaticContactOwner.AUTO
```

验证函数：

```python
def validate_coupling_options(
    value: MuJoCoVBDCouplingOptions | Mapping[str, object] | None,
) -> MuJoCoVBDCouplingOptions:
    """Normalize enums, reject unknown keys, and validate all finite ranges."""
```

必须验证：

- `iterations >= 1`；
- Aitken 模式建议 `iterations >= 2`，但不更改用户设置；
- relaxation 有限且位于 `[min, max]`；
- tolerance 非负；
- mass/inertia 上下界为正且有序；
- `coupling_mode="two_way"` 与 hard kinematic 不可同时出现；
- `proxy_mass_scale` 只作用于 two-way articulated effective mass，惯量按
  相同尺度缩放后再做 eigenvalue clamp；
- pure/passthrough backend 不接受用户显式传入的非默认 coupling options；
- `requires_grad=True` 时直接报错，除非两个后端和所有 feedback kernel 已具备
  完整 backward；第一版不允许静默 detach。

### 5.2 求解器构造函数

```python
class SolverMuJoCoVBD(SolverBase):
    def __init__(
        self,
        model: Model,
        *,
        mujoco_articulations: Sequence[int] | None = None,
        mujoco_joints: Sequence[int] | None = None,
        joint_mode: Literal["dynamic", "kinematic"] = "dynamic",
        coupling_mode: Literal["auto", "one_way", "two_way"] = "auto",
        contact_mode: Literal["auto", "soft", "full"] = "auto",
        coupling_options: MuJoCoVBDCouplingOptions | Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        vbd_options: Mapping[str, object] | None = None,
        collision_options: Mapping[str, object] | None = None,
    ) -> None: ...

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None: ...

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None: ...

    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None: ...

    def notify_model_changed(self, flags: ModelFlags | int) -> None: ...

    def rebuild_bvh(self, state: State) -> None: ...
```

构造顺序固定为：参数规范化 → ownership → feature discovery → backend selection →
只构造选中 backend。pure 分支即使收到了空字典以外的不适用 options 也必须报错，
防止用户以为配置已经生效。

`register_custom_attributes()` 直接调用本目录私有完整 VBD、轻量 VBD、pneumatic
和私有 MuJoCo 的注册入口，不能转调 MJVBD_V2。

`contacts` 参数的合同：

- `None`：使用所选 backend 自己持有的 contact buffer；
- pure VBD 只接受由 private VBD core 创建且布局/capacity 匹配的 contacts；
- pure MuJoCo 沿用 private MuJoCo core 的 contact 输入合同；
- mixed one-way/two-way 只接受由本 solver 的 `collision_pipeline.contacts()` 创建且
  capacity/config 匹配的 cross/VBD buffer；
- passthrough 只接受 `None`；
- mixed backend 不接受已经混入 MuJoCo-owned pair 的普通 `Contacts`，防止重复求解。

### 5.3 公开只读诊断

```python
@property
def contacts(self) -> Contacts | None: ...


@property
def diagnostics(self) -> MuJoCoVBDDiagnostics: ...


@property
def mujoco_solver(self) -> SolverMuJoCo | None: ...


@property
def vbd_solver(self) -> SolverVBD | None: ...


@property
def features(self) -> MuJoCoVBDFeatures: ...
```

在 passthrough/pure 分支中，不存在的 solver property 返回 `None`；双向诊断数组只在
`features.feedback_enabled` 时分配，否则 `diagnostics` 只包含对应分支的 contact、
overflow 和 backend timing。

诊断对象至少暴露 device arrays：

```python
@dataclass
class MuJoCoVBDDiagnostics:
    backend: MuJoCoVBDBackendKind
    residual_force_l2: wp.array | None  # float[world_count], two-way only
    residual_force_relative: wp.array | None  # float[world_count], two-way only
    residual_velocity_max: wp.array | None  # float[world_count], two-way only
    converged: wp.array | None  # bool[world_count], two-way only
    nonfinite_flag: wp.array | None  # int32[world_count]
    diverged_flag: wp.array | None  # int32[world_count], two-way only
    rigid_contact_overflow: wp.array | None  # int32[1]
    soft_contact_overflow: wp.array | None  # int32[1]
    body_particle_overflow: wp.array | None  # int32[1]
    feedback_wrench_raw: wp.array | None  # two-way only
    feedback_wrench_relaxed: wp.array | None  # two-way only
```

`None` 表示该 backend 根本没有该模块或 buffer，不表示分配了数组但当前值为零。

## 6. 所有权与静态映射

### 6.1 数据结构

`ownership.py`：

```python
@dataclass(frozen=True)
class MuJoCoVBDOwnership:
    mujoco_articulations: tuple[int, ...]
    mujoco_joints: tuple[int, ...]
    mujoco_bodies: tuple[int, ...]
    mujoco_shapes: tuple[int, ...]
    vbd_bodies: tuple[int, ...]
    vbd_particles: tuple[int, ...]
    vbd_shapes: tuple[int, ...]
    static_shapes: tuple[int, ...]
    proxy_bodies: tuple[int, ...]

    proxy_body_ids: wp.array  # int32[n_proxy]
    body_to_proxy_slot: wp.array  # int32[model.body_count], -1 when not proxy
    body_owner: wp.array  # int8[model.body_count]
    shape_owner: wp.array  # int8[model.shape_count]
    joint_owner: wp.array  # int8[model.joint_count]
```

owner 常量：

```text
OWNER_NONE    = 0
OWNER_MUJOCO  = 1
OWNER_VBD     = 2
OWNER_SHARED  = 3
```

### 6.2 解析函数

```python
def resolve_mujoco_vbd_ownership(
    model: Model,
    *,
    mujoco_articulations: Sequence[int] | None,
    mujoco_joints: Sequence[int] | None,
) -> MuJoCoVBDOwnership: ...
```

函数必须：

1. 解析 articulation 或显式 joint 集合，二者不能同时给出。
2. 验证 MuJoCo joint 是完整闭合的 joint forest。
3. 收集所有相连 MuJoCo body，包括 fixed link。
4. 把剩余动态刚体分配给 VBD。
5. 把全部粒子分配给 VBD。
6. 按 shape body 生成 MuJoCo、VBD 和 static shape 集合。
7. 建立 `body_to_proxy_slot`，保证 proxy slot 连续。
8. 验证同一动态 body 不会被两个后端提交。
9. 验证每个 world 的 source/proxy 数量和索引范围一致。
10. 拒绝 hard kinematic 的双向 articulation。

以下只在构造期执行，可以读取 host arrays；所有运行时映射必须是预分配的 device
array。

## 7. Model overlay

`model_overlay.py` 不创建通用 `ModelView`。它使用浅复制共享大型拓扑数组，只替换
ownership、solve-time effective mass、flags、joint-enable 和 collision filter 相关的
小数组。只有 mixed backend 构造 overlay；pure backend 直接使用自己的私有 core 和
原模型，passthrough 不构造 overlay。

```python
@dataclass
class MuJoCoVBDModelOverlays:
    mujoco: Model | None
    vbd: Model | None


def build_model_overlays(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
    options: MuJoCoVBDCouplingOptions,
) -> MuJoCoVBDModelOverlays: ...
```

### 7.1 MuJoCo overlay

```python
def build_mujoco_overlay(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
) -> Model: ...
```

要求：

- MuJoCo articulation 的 body/joint/shape 正常启用；
- VBD-owned 动态 body 不进入 MuJoCo integration；
- MuJoCo 只生成 `M-M` 和路由给 MuJoCo 的 `M-S` 接触；
- `M-V` pair 在 MuJoCo 中始终被过滤；
- 保留 actuator、joint limit、equality、tendon 和 robot self-contact；
- 保持全局 body/joint id 稳定，避免运行时 local/global scatter。

该 overlay 只由 `one_way_dynamic_soft`、`one_way_dynamic_full` 和 `two_way` 构造；
`pure_mujoco` 直接把原模型交给
同目录 private core。kinematic 单向分支直接消费用户提供的 `state_in` pose/velocity。

### 7.2 VBD overlay

```python
def build_vbd_overlay(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
    options: MuJoCoVBDCouplingOptions,
) -> Model: ...
```

要求：

- VBD-owned rigid body 和全部粒子正常求解；
- source body 标记 `BodyFlags.PROXY`；
- source articulation joint 在 VBD 中禁用，避免重复求解机器人 joint；
- proxy body 使用独立的 effective mass/inertia 工作数组；
- VBD 生成 `V-V`、`V-S` 和 `M-V` 接触；
- VBD 不生成 `M-M` 和由 MuJoCo 所有的 `M-S` 接触；
- proxy 输出 pose 不提交到 source-owned 公开 state；dynamic source 取 MuJoCo 输出，
  kinematic source 取 prescribed 输入；
- VBD-owned body/particle 输出只来自 VBD。

分支差异只写入 solve-time overlay：

```text
one_way immovable:
    preserve public body_mass/body_inertia
    VBD effective inv_mass/inv_inertia = 0
    proxy pose copied through unchanged

two_way effective_mass:
    VBD effective mass/inertia = current MuJoCo articulated estimate
    proxy local response participates in feedback iteration
```

## 8. 接触所有权和 pair routing

### 8.1 路由结构

`contact_routing.py`：

```python
@dataclass(frozen=True)
class MuJoCoVBDContactRouting:
    mujoco_shape_pairs: wp.array  # vec2i[n_mj_pairs]
    vbd_shape_pairs: wp.array  # vec2i[n_vbd_pairs]
    cross_shape_pairs: wp.array  # vec2i[n_cross_pairs]
    cross_shape_mask: wp.array  # uint8[shape_count]
    cross_body_mask: wp.array  # uint8[body_count]
    full_surface_shape_indices: tuple[int, ...]
```

```python
def build_contact_routing(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    *,
    collision_options: Mapping[str, object],
    static_contact_owner: MuJoCoVBDStaticContactOwner,
) -> MuJoCoVBDContactRouting: ...
```

### 8.2 固定路由规则

| Pair | Solver | two-way 是否反馈 |
| --- | --- | --- |
| MuJoCo body - MuJoCo body | MuJoCo | 否 |
| VBD body - VBD body | VBD | 否 |
| VBD particle/edge/face - VBD body | VBD | 否 |
| MuJoCo/source proxy body - VBD body | VBD | 是；one-way 明确丢弃 |
| MuJoCo/source proxy body - VBD particle/edge/face | VBD | 是；one-way 明确丢弃 |
| VBD object - static | VBD | 否 |
| MuJoCo body - static | 配置或 AUTO | 仅 MuJoCo |

`AUTO` 对 static shape 按实际 pair 分开：robot-static 由 MuJoCo，VBD-static 由
VBD。static shape 可以对两个后端可见，但同一 pair 不能进入两个 contact stream。

### 8.3 构造期检查

```python
def validate_contact_routing(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
) -> None: ...
```

必须验证：

- 没有 duplicate pair；
- 没有遗漏合法的 `M-V` pair；
- full-surface 形状具备解析 SDF 或已 provisioned mesh SDF；
- visual-only shape 没有 `COLLIDE_PARTICLES`；
- shape world 与 body world 一致；
- external filter 的非对称性保持原语义。

## 9. Runtime state 与历史事务

本节的完整 transaction runtime 只属于 `two_way`。其他 backend 使用严格裁剪的
runtime：

```text
pure_vbd_*                 VBD ping-pong + owned Contacts
pure_mujoco                MuJoCo state only
kinematic_passthrough      no private runtime
one_way_kinematic_soft     VBD particle/pneumatic state + sparse Contacts
one_way_kinematic_full     VBD state + full Contacts
one_way_dynamic_soft       MuJoCo state + VBD-soft particle state + one sync buffer
one_way_dynamic_full       MuJoCo state + VBD-full state + one sync buffer
two_way                    complete runtime below
```

### 9.1 固定 buffer

`state.py`：

```python
@dataclass
class MuJoCoVBDRuntime:
    mujoco_state_in: State
    mujoco_state_out: State
    vbd_state_in: State
    vbd_state_out: State

    substep_state_snapshot: State
    final_mujoco_state: State
    final_vbd_state: State

    proxy_qd_before: wp.array  # spatial_vector[body_count]
    proxy_mass: wp.array  # float[n_proxy]
    proxy_inertia: wp.array  # mat33[n_proxy]

    wrench_raw: wp.array  # spatial_vector[body_count]
    wrench_relaxed: wp.array  # spatial_vector[body_count]
    wrench_previous: wp.array  # spatial_vector[body_count]
    residual_current: wp.array  # spatial_vector[body_count]
    residual_previous: wp.array  # spatial_vector[body_count]

    aitken_omega: wp.array  # float[world_count]
    aitken_has_previous: wp.array  # int32[world_count]
    converged: wp.array  # bool[world_count]
    nonfinite_flag: wp.array  # int32[world_count]
```

```python
def allocate_runtime(
    model: Model,
    overlays: MuJoCoVBDModelOverlays,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDCouplingOptions,
) -> MuJoCoVBDRuntime: ...
```

所有数组在构造期分配。Graph capture 后不得 resize。

```python
def allocate_backend_runtime(
    kind: MuJoCoVBDBackendKind,
    model: Model,
    overlays: MuJoCoVBDModelOverlays,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDResolvedOptions,
) -> object:
    """Allocate only the selected backend's concrete runtime dataclass."""
```

### 9.2 历史事务接口

MuJoCo adapter：

```python
class MuJoCoCouplingBackend:
    def begin_substep(
        self,
        state_in: State,
        control: Control | None,
        dt: float,
    ) -> None:
        """Snapshot state, warm start, actuator state, sleep state and step counters."""

    def restore_iteration(self, iteration: int) -> None:
        """Restore exactly the same substep input without committing time/history."""

    def solve_iteration(
        self,
        body_wrench: wp.array[wp.spatial_vector],
        state_out: State,
        dt: float,
    ) -> None:
        """Inject cross wrench and execute one MuJoCo solve of the same interval."""

    def commit_substep(self, state_out: State) -> None:
        """Commit only the selected final iteration and advance counters once."""

    def abort_substep(self) -> None:
        """Restore snapshots after an exception or nonfinite failure."""
```

VBD adapter：

```python
class VBDCouplingBackend:
    def begin_substep(self, state_in: State, dt: float) -> None:
        """Snapshot VBD/AVBD, self-contact, rigid history, DAT and pneumatic history."""

    def restore_iteration(self, iteration: int) -> None:
        """Restore the same soft/rigid initial state and history seed."""

    def solve_iteration(
        self,
        state_out: State,
        control: Control | None,
        contacts: Contacts,
        dt: float,
    ) -> None:
        """Execute one complete VBD solve against the current proxy state."""

    def commit_substep(self, state_out: State, contacts: Contacts) -> None:
        """Commit final positions, velocities and persistent histories exactly once."""

    def abort_substep(self) -> None:
        """Restore every persistent history changed by a failed iteration."""
```

### 9.3 必须事务化的内部状态

除公开 `State` 外，至少覆盖：

- MuJoCo qacc/warm-start/constraint cache；
- MuJoCo actuator activation；
- MuJoCo `_step` 和 update interval 计数；
- MuJoCo sleeping tree bookkeeping；
- VBD `body_q_prev`、`particle_q_prev`；
- AVBD joint/contact lambda、penalty、C0；
- rigid contact matching/history；
- Dahl friction state；
- stick/slip/deadzone history；
- self-contact BVH cadence和 activity selector；
- VT/EE detector counts、minimum distance、DAT state；
- pneumatic volume、absolute pressure、volume rate、clamp flags；
- material phase device selector。

只有最后一次 `commit_substep()` 可以修改跨 substep 持久历史。

## 10. Coupling kernel 规格

`kernels.py` 中 kernel 均使用全局稳定 body id，不再做通用 entry-local 映射。

### 10.1 初始状态分发

```python
@wp.kernel
def copy_owned_state_to_backends_kernel(
    body_owner: wp.array[wp.int8],
    particle_owner: wp.array[wp.int8],
    state_body_q: wp.array[wp.transform],
    state_body_qd: wp.array[wp.spatial_vector],
    state_particle_q: wp.array[wp.vec3],
    state_particle_qd: wp.array[wp.vec3],
    mujoco_body_q: wp.array[wp.transform],
    mujoco_body_qd: wp.array[wp.spatial_vector],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
    vbd_particle_q: wp.array[wp.vec3],
    vbd_particle_qd: wp.array[wp.vec3],
): ...
```

实际实现可按 body/particle 分成两个固定 launch，避免一个巨大混合 kernel；必须通过
基准决定是否融合，不能仅凭 launch 数假设更快。

### 10.2 Proxy 同步与 rewind

```python
@wp.kernel
def sync_and_rewind_proxy_bodies_kernel(
    dt: float,
    proxy_body_ids: wp.array[wp.int32],
    source_body_q_end: wp.array[wp.transform],
    source_body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_gravity_acceleration: wp.array[wp.vec3],
    wrench_relaxed: wp.array[wp.spatial_vector],
    proxy_inv_mass: wp.array[float],
    response_mode: int,
    destination_body_q: wp.array[wp.transform],
    destination_body_qd: wp.array[wp.spatial_vector],
    destination_body_f: wp.array[wp.spatial_vector],
    destination_body_q_prev: wp.array[wp.transform],
    destination_body_q_prev_snapshot: wp.array[wp.transform],
    proxy_qd_before: wp.array[wp.spatial_vector],
): ...
```

行为：

- 复制 MuJoCo solve 后的 `body_q/body_qd`；
- 保存 `proxy_qd_before`；
- 内部 `DIRICHLET` 标记仅供 one-way sync kernel 使用，不是公开 two-way 选项；
- `EFFECTIVE_MASS` 模式按上一轮 feedback、外力和重力执行 lagged rewind，防止
  同一 wrench 在两个后端重复应用；
- 所有计算保持同一 world 和 body id。

#### 10.2.1 Substep time synchronization (implemented)

The two-way backend treats MuJoCo and VBD as two nonlinear solves of the same
physical interval, not as two consecutive integrators.  MuJoCo provides an end
pose and end velocity.  Before VBD starts, `sync_and_rewind_proxy_bodies_kernel`
reconstructs the begin-of-substep COM pose with the end velocity, installs it in
`body_q`, `body_q_prev`, and `_coupling_body_q_prev_snapshot`, then lets VBD
integrate exactly once back toward the MuJoCo end state.  The relaxed interface
wrench and gravity already included by MuJoCo are cancelled from VBD's external
force input; the rigid-body gyroscopic term is cancelled as well so angular
velocity is not advanced twice. Actuator and articulated motion remain embedded in the copied end
velocity. One-way Dirichlet synchronization uses a separate verbatim-copy
kernel and is not rewound.

### 10.3 Wrench 写入 MuJoCo

```python
@wp.kernel
def compose_mujoco_body_force_kernel(
    body_owner: wp.array[wp.int8],
    external_body_f: wp.array[wp.spatial_vector],
    coupling_body_f: wp.array[wp.spatial_vector],
    out_body_f: wp.array[wp.spatial_vector],
): ...
```

`out_body_f = external_body_f + coupling_body_f`。不得覆盖用户外力。

### 10.4 最终状态提交

```python
@wp.kernel
def reconcile_owned_state_kernel(
    body_owner: wp.array[wp.int8],
    particle_owner: wp.array[wp.int8],
    mujoco_body_q: wp.array[wp.transform],
    mujoco_body_qd: wp.array[wp.spatial_vector],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
    vbd_particle_q: wp.array[wp.vec3],
    vbd_particle_qd: wp.array[wp.vec3],
    out_body_q: wp.array[wp.transform],
    out_body_qd: wp.array[wp.spatial_vector],
    out_particle_q: wp.array[wp.vec3],
    out_particle_qd: wp.array[wp.vec3],
): ...
```

proxy body 的 VBD 输出永远不能覆盖 source-owned 输出：dynamic source 取 MuJoCo，
kinematic source 取 prescribed `state_in`。

## 11. Articulated effective mass

### 11.1 计算入口

`effective_mass.py`：

```python
class MuJoCoVBDEffectiveMass:
    def __init__(
        self,
        mujoco_solver: SolverMuJoCo,
        vbd_solver: SolverVBD,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDCouplingOptions,
    ) -> None: ...

    def update(
        self,
        mujoco_state: State,
        vbd_state: State,
    ) -> None:
        """Evaluate articulated body blocks and install them as proxy preconditioners."""
```

构造期创建：

```python
endpoint_kind  # BODY for every proxy
endpoint_index  # proxy body global ids
endpoint_local_pos  # body COM local positions
mass  # float[n_proxy]
inertia  # mat33[n_proxy]
```

每轮 MuJoCo solve 后调用现有的：

```python
SolverMuJoCo.coupling_eval_effective_mass_block(
    endpoint_kind,
    endpoint_index,
    endpoint_local_pos,
    mass,
    inertia,
)
```

### 11.2 安装到 VBD

不得每轮修改公开 `model.body_mass` 后调用完整 `notify_model_changed()`，那会重建不相关
cache。VBD adapter 增加专用入口：

```python
def set_proxy_effective_inertia(
    self,
    proxy_body_ids: wp.array[wp.int32],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
) -> None:
    """Update only proxy slots in VBD's effective inverse mass/inertia arrays."""
```

对应 kernel：

```python
@wp.kernel
def install_proxy_effective_inertia_kernel(
    proxy_body_ids: wp.array[wp.int32],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    mass_scale: float,
    mass_min: float,
    mass_max: float,
    inertia_eigenvalue_min: float,
    inertia_eigenvalue_max: float,
    out_inv_mass_effective: wp.array[float],
    out_inv_inertia_effective: wp.array[wp.mat33],
): ...
```

必须：

- 对质量做有限正数和上下界检查；
- 对 inertia 对称化并夹紧 eigenvalue；
- 非有限值设置 `nonfinite_flag`，不能传给 AVBD；
- effective mass 只是 partitioned solve 的预条件器，不能改变公开模型质量。

## 12. 专用 collision pipeline

`collision_pipeline.py`：

```python
class MuJoCoVBDCollisionPipeline:
    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        routing: MuJoCoVBDContactRouting,
        **collision_options: object,
    ) -> None: ...

    def contacts(self) -> Contacts: ...

    def collide_iteration(
        self,
        state: State,
        contacts: Contacts,
        *,
        iteration: int,
    ) -> None:
        """Refresh all V-V and M-V contacts for the current iteration state."""

    def reset(
        self,
        world_mask: wp.array[wp.bool] | None = None,
    ) -> None: ...

    def rebuild(self) -> None: ...
```

要求：

- 基于 `MJVBDV2CollisionPipeline` 的 retained full-surface AABB mask 和 active
  candidate compaction；
- 每轮清空 count 并重跑 narrow phase；
- broad phase 可以复用构造期静态 pair 拓扑，但不能复用上一轮接触点、法线和距离；
- full-surface point/edge/face 写入统一 `Contacts.soft_contact_*`；
- rigid-rigid 接触保留 contact matching；
- 只对 cross/VBD pair 生成 contact；
- fixed capacity、device counts、overflow flags；
- deterministic 模式开启稳定 contact key sort；
- CPU、CUDA contact key 集合必须一致。

### 12.1 Full-surface SDF 合同

参与 full-surface 的 mesh/convex shape 必须在 `ModelBuilder.finalize()` 前拥有真实
volume SDF。缺失时构造 pipeline 立即报错。解析 sphere、box、capsule、cylinder、
cone、ellipsoid 和 infinite plane 使用解析 SDF。

`rigid_soft_full_surface_shape_indices` 默认只包含 `M-V` 中真正可能碰软体的机器人
collision shapes，不能把 visual mesh 自动加入。

### 12.2 Velocity-aware M-V point contacts and AL history (implemented)

For each VBD particle / MuJoCo-owned shape candidate, the ordinary narrow phase
still emits actual contacts first. A second fixed-topology kernel appends a
record only when the pair is currently separated and its relative normal
velocity predicts crossing during the current `dt`:

```text
extension = min(configured_max, max(0, -dot(n, v_particle - v_shape_point) * dt))
actual_threshold <= signed_distance < actual_threshold + extension
```

Actual and speculative predicates are disjoint, so the existing maximum
capacity and one-record-per-candidate bound do not change. Static and VBD-owned
shapes retain the legacy path. The first implementation deliberately targets
point contacts used by volumetric soft bodies; edge/face full-surface contact is
unchanged and requires separate validation before extension.

The normal force for enabled M-V records is
`max(0, k * penetration + lambda_n)`. Before each VBD primal sweep, the dual is
updated with `lambda_n = max(0, lambda_n + rho_scale * material_ke * penetration)`.
The Hessian remains `k*n*n^T`, and Coulomb friction uses the same augmented
normal load. `Contacts.soft_contact_tids` is the stable candidate key: a fixed
device array persists one scalar multiplier per candidate, restores matched
active records with configurable decay, and clears disappeared candidates.
The history array is included in the outer-iteration transaction snapshot, so
only the selected final coupling round commits it.

## 13. Feedback harvest

### 13.1 当前必须补齐的功能缺口

当前 VBD proxy harvest 对 full-surface contact 存在明确限制：它读取
`Contacts.soft_contact_particle`，而 edge/face 记录在该数组中为 `-1`。因此当前
`coupling_prepare_proxy_contacts()` 会在双向 proxy + full-surface 时抛出
`NotImplementedError`。

新求解器第一版必须删除这一限制，直接消费统一记录：

```text
soft_contact_indices = (p, -1, -1)        point
soft_contact_indices = (v0, v1, -1)       edge
soft_contact_indices = (v0, v1, v2)       face
soft_contact_barycentric                  contact weights
```

### 13.2 统一 soft-contact 反力求值

为避免 VBD 解算和 feedback 采用不同 force law，应把现有 point 和 edge/face 接触
力求值抽成同一个 private Warp function：

```python
@wp.func
def eval_unified_soft_contact_force(
    dt: float,
    contact_index: int,
    corners: wp.vec3i,
    barycentric: wp.vec3,
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_penalty_k: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    friction_epsilon: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Return force on the soft feature and its world-space contact point."""
```

Warp 当前不一定支持该 Python tuple 标注形式；实现时按 Warp 支持的多返回值语法，
这里描述的是语义。

该函数由以下两处共同调用：

- VBD particle/body force-Hessian path；
- coupling feedback final-force evaluation。

### 13.3 Unified soft feedback kernel

```python
@wp.kernel
def harvest_unified_soft_proxy_wrenches_kernel(
    dt: float,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_max: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    body_to_proxy_slot: wp.array[int],
    contact_penalty_k: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    friction_epsilon: float,
    out_proxy_wrench: wp.array[wp.spatial_vector],
    out_contact_force: wp.array[wp.vec3],
): ...
```

每个 active contact：

1. 从 `indices+barycentric` 重建 soft contact point。
2. 使用和 VBD 完全相同的 point/edge/face force law 求 `f_soft`。
3. 读取 `shape_body[contact_shape]`。
4. 若 body 不是 proxy，退出。
5. 计算 `f_body=-f_soft`。
6. 计算相对 COM 的 torque。
7. 写入该 proxy 的 wrench contribution。

高性能默认路径可以 atomic add。deterministic 路径禁止跨 block 原子加法，先写
per-contact contribution，再按 `(world, proxy_slot, contact_key)` 稳定排序并 segmented
reduce。

### 13.4 Rigid-rigid feedback

```python
@wp.kernel
def harvest_rigid_proxy_wrenches_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_max: int,
    body0: wp.array[int],
    body1: wp.array[int],
    point0_world: wp.array[wp.vec3],
    point1_world: wp.array[wp.vec3],
    force_on_body1: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_proxy_slot: wp.array[int],
    out_proxy_wrench: wp.array[wp.spatial_vector],
): ...
```

只归约 `M-V` cross contact。`M-M` 不应进入 VBD contact buffer；`V-V` 不回传
MuJoCo。

### 13.5 Feedback 类

```python
class MuJoCoVBDFeedback:
    def clear(self) -> None: ...

    def harvest(
        self,
        vbd_state_in: State,
        vbd_state_out: State,
        contacts: Contacts,
        dt: float,
    ) -> wp.array[wp.spatial_vector]:
        """Return contact-native raw wrench on MuJoCo-owned proxy bodies."""

    def validate_finite(self) -> None: ...
```

不允许使用 aggregate momentum difference 作为正常 fallback；如果 contact-native
force 所需的 VBD material/penalty state 不存在，应构造或 step 时显式失败。

## 14. Relaxation 与收敛

`convergence.py`：

```python
class MuJoCoVBDConvergence:
    def begin_substep(self, warm_start: bool) -> None: ...

    def update(
        self,
        raw_wrench: wp.array[wp.spatial_vector],
        relaxed_wrench: wp.array[wp.spatial_vector],
        iteration: int,
    ) -> None: ...

    def finalize(self) -> None: ...
```

固定 relaxation：

```text
r[k] = Wraw[k] - W[k-1]
W[k] = W[k-1] + omega * r[k]
```

每 world Aitken：

```text
dr       = r[k] - r[k-1]
omega[k] = clamp(
    -omega[k-1] * dot(r[k-1], dr) / max(dot(dr, dr), eps),
    omega_min,
    omega_max,
)
W[k]     = W[k-1] + omega[k] * r[k]
```

对应 kernel：

```python
@wp.kernel
def update_relaxed_wrench_kernel(
    iteration: int,
    relaxation_mode: int,
    relaxation_initial: float,
    relaxation_min: float,
    relaxation_max: float,
    body_world: wp.array[int],
    proxy_body_ids: wp.array[int],
    wrench_raw: wp.array[wp.spatial_vector],
    wrench_previous: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
    world_dot_r_dr: wp.array[float],
    world_dot_dr_dr: wp.array[float],
    aitken_omega: wp.array[float],
    out_wrench_relaxed: wp.array[wp.spatial_vector],
    out_residual_current: wp.array[wp.spatial_vector],
): ...
```

实际 Aitken 实现分三步：per-proxy contribution、per-world reduction、per-proxy
update。deterministic 模式使用固定顺序 segmented reduction。

收敛只做诊断，不改变 CUDA Graph launch topology。固定 `iterations` 保证 eager 和
Graph 语义一致。`converged[world]` 在最后一轮由以下条件生成：

```text
||r||_2 <= abs_tol + rel_tol * max(||Wraw||_2, ||W||_2)
and proxy/source velocity mismatch <= velocity_tolerance
```

### 14.1 发散保护

```python
@wp.kernel
def detect_coupling_failure_kernel(
    residual_current: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
    wrench_raw: wp.array[wp.spatial_vector],
    divergence_ratio: float,
    nonfinite_flag: wp.array[int],
    diverged_flag: wp.array[int],
): ...
```

若发现 NaN/Inf：

- `fail_on_nonfinite=True`：aborts transaction，公开 state 不得部分更新；
- 否则将该 world feedback 清零并设置诊断 flag，但不得把非有限值送入 MuJoCo。

连续 residual 放大时，Aitken omega 夹到 `relaxation_min`；不能动态减少 iteration
数或改变 Graph topology。不能只对传入 MuJoCo 的 wrench 做单边 clamp，因为这会
破坏 equal-and-opposite 合同。需要限制极端接触力时，应修改双方共同使用的接触
材料/penalty law，或者将该 substep 判为失败并回滚。

## 15. MuJoCo backend adapter

`backend_mujoco.py`：

```python
class MuJoCoCouplingBackend:
    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        **mujoco_options: object,
    ) -> None: ...

    def begin_substep(self, state_in: State, control: Control | None, dt: float) -> None: ...
    def restore_iteration(self, iteration: int) -> None: ...
    def solve_iteration(
        self,
        external_state: State,
        coupling_wrench: wp.array[wp.spatial_vector],
        state_out: State,
        dt: float,
    ) -> None: ...
    def evaluate_effective_mass_block(
        self,
        body_ids: wp.array[int],
        out_mass: wp.array[float],
        out_inertia: wp.array[wp.mat33],
    ) -> None: ...
    def wake_from_feedback(self, world_mask: wp.array[wp.bool]) -> None: ...
    def commit_substep(self, state_out: State) -> None: ...
    def abort_substep(self) -> None: ...
```

必须修正当前 `SolverMuJoCo.step()` 每调用一次就 `_step += 1` 的行为：coupling
iteration 不能使 update interval、时间或统计计数前进多次。adapter 应提供一个
transactional iteration 入口，只有 commit 时推进一次计数。

### 15.1 MuJoCo contacts 与 sleeping

- MuJoCo 内部 `M-M/M-S` contacts 保留；
- cross `M-V` contact 不送入 MuJoCo detector；
- sleeping 只在 MuJoCo Warp GPU 后端支持；
- VBD harvest 到非零 cross wrench 后生成 world wake mask；
- 下一 coupling iteration 前调用专用 wake 入口；
- sleep bookkeeping 也必须随 iteration restore；
- CPU MuJoCo 禁止请求 sleeping，沿用后端限制。

## 16. VBD backend adapter

`backend_vbd.py`：

```python
class VBDCouplingBackend:
    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDCouplingOptions,
        **vbd_options: object,
    ) -> None: ...

    def begin_substep(self, state_in: State, dt: float) -> None: ...
    def restore_iteration(self, iteration: int) -> None: ...
    def sync_proxy_state(
        self,
        mujoco_state_out: State,
        relaxed_wrench: wp.array[wp.spatial_vector],
        dt: float,
    ) -> None: ...
    def set_proxy_effective_inertia(
        self,
        mass: wp.array[float],
        inertia: wp.array[wp.mat33],
    ) -> None: ...
    def solve_iteration(
        self,
        state_out: State,
        control: Control | None,
        contacts: Contacts,
        dt: float,
    ) -> None: ...
    def commit_substep(self, state_out: State, contacts: Contacts) -> None: ...
    def abort_substep(self) -> None: ...
```

VBD 使用完整实现而不是 soft-only 实现，只要场景包含以下任一项：

- VBD dynamic rigid body；
- effective-mass proxy body；
- pneumatic cavity；
- rigid-rigid `M-V` contact。

因此典型双向机器人场景默认走完整 `vbd/`，不能为了少一些模块错误选择
`vbd_soft/`。

## 17. Backend 执行规格

构造期只选择一次 backend。`step()` 内禁止根据 contact count、对象类型或 host 数据重新
选择分支；因此 backend 是静态执行计划，也是 CUDA Graph topology 的所有者。

### 17.1 统一 backend 协议

`backends/base.py`：

```python
class MuJoCoVBDBackend(Protocol):
    kind: MuJoCoVBDBackendKind

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None: ...

    def reset(
        self,
        state: State,
        world_mask: wp.array | None,
        flags: SolverResetFlags | None,
    ) -> None: ...

    def notify_model_changed(self, flags: ModelFlags | int) -> None: ...
    def rebuild_bvh(self, state: State) -> None: ...
    def diagnostics(self) -> MuJoCoVBDDiagnostics: ...
```

`SolverMuJoCoVBD` 只负责公共输入校验、静态 backend 构造和 API 转发。backend 自己拥有
其所需的 private core、contact pipeline、history 和 Graph；公共类不得为了接口统一而给
纯分支分配空的 feedback、proxy、Aitken 或另一套 solver state。

所有 backend 都遵守以下提交合同：

- 一个公开 `step()` 只让物理时间和内部 step counter 前进一次；
- 输出只写到 `state_out`，不得原地破坏 `state_in`；
- `contacts=None` 时使用构造期私有 contacts，显式 contacts 必须匹配该 backend 的布局；
- overflow/nonfinite 在公开状态提交前检查；
- eager 与 Graph 使用同一条 backend 路径。

### 17.2 `_PureVBDSoftBackend`

适用范围：没有 MuJoCo joint，且只有粒子、布料、tet、spring 等 soft dynamics；无 VBD
dynamic rigid body、pneumatic cavity 或 rigid-rigid contact。

构造：

```python
class _PureVBDSoftBackend:
    def __init__(self, model, contacts, vbd_options):
        self.solver = vbd_soft.SolverVBD(model, **vbd_options)
        self.contacts = contacts or self.solver.create_contacts(model)
```

执行：

```python
def step(self, state_in, state_out, control, contacts, dt):
    selected = self.contacts if contacts is None else contacts
    self.solver.step(state_in, state_out, control, selected, dt)
```

该分支只构造私有 `vbd_soft/` 和所需的 soft/self-contact pipeline；不得构造 MuJoCo、body
proxy、full VBD rigid buffers、feedback、effective mass 或 coupling transaction。其数值和
性能基线是同一 baseline commit 的 MJVBD_V2 `pure_vbd` soft-only 静态分支。

### 17.3 `_PureVBDFullBackend`

适用范围：没有 MuJoCo joint，并包含任一 VBD rigid body、pneumatic cavity、刚软耦合或
需要完整 VBD contact 的场景。必须支持：

- 纯刚体；
- 纯布料、tet、spring、软体；
- 刚体与软体混合；
- pneumatic；
- self-contact、DAT、摩擦和 multiworld。

```python
class _PureVBDFullBackend:
    def __init__(self, model, contacts, vbd_options):
        self.solver = vbd.SolverVBD(model, **vbd_options)
        self.contacts = contacts or self.solver.create_contacts(model)

    def step(self, state_in, state_out, control, contacts, dt):
        selected = self.contacts if contacts is None else contacts
        self.solver.step(state_in, state_out, control, selected, dt)
```

该分支只构造私有 `vbd/`、所选 contact pipeline 和 pneumatic 状态，不构造 MuJoCo 或
coupling 模块。它不是“先构造双向求解器再把反馈关掉”。

### 17.4 `_KinematicPassthroughBackend`

适用范围：没有 VBD dynamic DOF，MuJoCo articulation 也被配置为 hard kinematic。该分支
不进行动力学，只复制/规范化用户给定的状态，并更新由 body pose 推导的 shape pose：

```python
def step(self, state_in, state_out, control, contacts, dt):
    copy_state_kernel(state_in, state_out)
    eval_kinematic_shape_transforms_kernel(self.model, state_out)
```

不得构造私有 MuJoCo/VBD solver、contacts、proxy 或 Graph scratch。非零 body mass/inertia
保留为模型元数据，但不产生动力学响应。

### 17.5 `_PureMuJoCoBackend`

适用范围：存在动态 MuJoCo articulation，但没有任何 VBD dynamic DOF。

```python
class _PureMuJoCoBackend:
    def __init__(self, model, mujoco_options):
        self.solver = mujoco.SolverMuJoCo(model, **mujoco_options)

    def step(self, state_in, state_out, control, contacts, dt):
        self.solver.step(state_in, state_out, control, contacts, dt)
```

只构造同目录私有 `mujoco/` core。VBD/contact/feedback/effective-mass buffer 必须不存在。
此分支用于机器人、关节刚体和 MuJoCo 自身 contact 的独立仿真，并保持 private MuJoCo
core 的 sleeping、equality、actuator 和 integrator 能力。

### 17.6 `_OneWayKinematicSoftBackend`

这是独立复现 MJVBD_V2 `mjvbd_kinematic_soft` 的最快单向分支。source 机器人或移动刚体
由 `state_in` 提供 hard kinematic pose，只有 soft 粒子响应；source 永不接收反力。无
pneumatic 时构造 private `vbd_soft`，存在 pneumatic cavity 时构造 private full `vbd`；
两者都使用 soft/point contact pipeline。

```python
def step(self, state_in, state_out, control, contacts, dt):
    selected = self.contacts if contacts is None else contacts
    self.sync_kinematic_shapes(state_in, self.runtime.vbd_state_in)
    self.collision_pipeline.collide(
        self.runtime.vbd_state_in,
        selected,
    )
    self.vbd_core.step(
        self.runtime.vbd_state_in,
        self.runtime.vbd_state_out,
        control,
        selected,
        dt,
    )
    self.commit_source_and_vbd_state(
        source=state_in,
        vbd=self.runtime.vbd_state_out,
        out=state_out,
    )
```

source shape 在 VBD 中的 solve-time inverse mass 固定为零。原模型的非零 mass/inertia 不得
被覆盖，只通过单独的 effective inverse-mass view 形成不可移动 collider。该分支不得分配
MuJoCo core、finite-mass proxy、feedback、effective mass、relaxation 或 transaction restore。
仅当存在 pneumatic 时允许增加 full VBD pneumatic state；不得因此分配 VBD
dynamic-rigid contact buffer。

### 17.7 `_OneWayKinematicFullBackend`

独立复现 MJVBD_V2 `vbd_kinematic_full`。适用于 hard kinematic source 加任一 full VBD
能力，包括 VBD dynamic rigid、刚软 contact 和 full-surface contact；pneumatic 只有在
同时显式选择 full contact pipeline 时进入本分支，否则走上一节的 soft pipeline + full
VBD core 组合：

```python
def step(self, state_in, state_out, control, contacts, dt):
    selected = self.contacts if contacts is None else contacts
    self.sync_immovable_source_proxies(state_in, self.runtime.vbd_state_in)
    self.collision_pipeline.collide(self.runtime.vbd_state_in, selected)
    self.vbd_full.step(
        self.runtime.vbd_state_in,
        self.runtime.vbd_state_out,
        control,
        selected,
        dt,
    )
    self.commit_source_and_vbd_state(state_in, self.runtime.vbd_state_out, state_out)
```

source proxy 的 solve-time inverse mass/inertia 为零；source pose 由输入直接提交，VBD 只能
更新 VBD-owned DOF。该分支保留完整 contact pipeline 优化，但没有 feedback 和 outer
coupling iteration。

### 17.8 `_OneWayDynamicSoftBackend` 与 `_OneWayDynamicFullBackend`

动态机器人先由同目录 private MuJoCo core 前进一步，再作为 source 驱动 private VBD；VBD
反力被明确丢弃。这两个静态 backend 独立复现 MJVBD_V2 动态混合场景在 one-way 配置
下的 core 选择和语义：

- `_OneWayDynamicSoftBackend`：external rigid source、particle-only VBD、无 pneumatic，
  使用私有 `vbd_soft`；
- `_OneWayDynamicFullBackend`：存在 VBD dynamic rigid 或 pneumatic，使用私有 `vbd`。

full-surface contact 本身不强制选择 full core；当所有 rigid body 都由 MuJoCo source
驱动且没有 pneumatic 时，私有 `vbd_soft` 必须继续支持 MJVBD_V2 已有的 full-surface
particle/edge/face pipeline。

二者共享不含运行时分派的 `_OneWayDynamicBase._step_impl()`：

```python
def step(self, state_in, state_out, control, contacts, dt):
    selected = self.contacts if contacts is None else contacts
    self.mujoco.step(
        state_in,
        self.runtime.mujoco_state_out,
        control,
        self.runtime.mujoco_contacts,
        dt,
    )
    self.sync_immovable_source_proxies(
        self.runtime.mujoco_state_out,
        self.runtime.vbd_state_in,
    )
    self.collision_pipeline.collide(self.runtime.vbd_state_in, selected)
    self.vbd_core.step(
        self.runtime.vbd_state_in,
        self.runtime.vbd_state_out,
        control,
        selected,
        dt,
    )
    self.reconcile_one_way_state(
        self.runtime.mujoco_state_out,
        self.runtime.vbd_state_out,
        state_out,
    )
```

MuJoCo 与所选 VBD core 各自只 step 一次。cross contact 只由 VBD 处理；MuJoCo 仍处理
`M-M/M-S`，避免重复接触。这两个分支都不构造 feedback、relaxation、iteration restore
或 effective mass；soft 分支还不得构造 full VBD rigid/pneumatic buffers。因此性能路径
应与独立 MJVBD_V2 对应 core 选择等价，而不是双向路径 `iterations=1` 的别名。

### 17.9 `_TwoWayBackend`

双向 backend 才构造完整 ownership overlay、MuJoCo/VBD adapters、finite-mass/effective-mass
proxy、unified contact feedback、relaxation、transaction snapshots 和 fixed-iteration Graph。

```python
class _TwoWayBackend:
    def step(self, state_in, state_out, control, contacts, dt): ...
    def _begin_substep(self, state_in, control, dt): ...
    def _restore_iteration(self, iteration): ...
    def _record_final_iteration_state(self, iteration): ...
    def _commit_substep(self, state_out, contacts): ...
    def _abort_substep(self): ...
```

其详细时序见下一节。该 backend 固定使用 articulated effective-mass proxy，必须
harvest equal-and-opposite wrench 并反馈 MuJoCo，不能退化为 hard kinematic。

## 18. Public dispatch 与 two-way step 代码级时序

公共入口只做通用校验并静态转发：

```python
class SolverMuJoCoVBD(SolverBase):
    def step(self, state_in, state_out, control, contacts, dt):
        self._validate_step_inputs(state_in, state_out, control, contacts, dt)
        self.backend.step(state_in, state_out, control, contacts, dt)
```

`_TwoWayBackend.step()` 的主要实现应等价于：

```python
def step(self, state_in, state_out, control, contacts, dt):
    selected_contacts = self.contacts if contacts is None else contacts

    self._begin_substep(state_in, control, dt)
    try:
        for iteration in range(self.options.iterations):
            self._restore_iteration(iteration)

            self.mujoco_backend.solve_iteration(
                external_state=state_in,
                coupling_wrench=self.runtime.wrench_relaxed,
                state_out=self.runtime.mujoco_state_out,
                dt=dt,
            )

            self.vbd_backend.sync_proxy_state(
                self.runtime.mujoco_state_out,
                self.runtime.wrench_relaxed,
                dt,
            )

            self.effective_mass.update(
                self.runtime.mujoco_state_out,
                self.runtime.vbd_state_in,
            )

            self.collision_pipeline.collide_iteration(
                self.runtime.vbd_state_in,
                selected_contacts,
                iteration=iteration,
            )

            self.vbd_backend.solve_iteration(
                self.runtime.vbd_state_out,
                control,
                selected_contacts,
                dt,
            )

            self.feedback.harvest(
                self.runtime.vbd_state_in,
                self.runtime.vbd_state_out,
                selected_contacts,
                dt,
            )

            self.convergence.update(
                self.runtime.wrench_raw,
                self.runtime.wrench_relaxed,
                iteration,
            )

            self._record_final_iteration_state(iteration)

        self._commit_substep(state_out, selected_contacts)
    except Exception:
        self._abort_substep()
        raise
```

### 18.1 `_begin_substep`

```python
def _begin_substep(
    self,
    state_in: State,
    control: Control | None,
    dt: float,
) -> None: ...
```

顺序：

1. snapshot 公开输入；
2. 分发 ownership state；
3. snapshot 两个 backend 私有 history；
4. 初始化 contact generation transaction；
5. 从上一 substep wrench warm start 或清零；
6. 重置 Aitken iteration-local state；
7. 清空诊断 flag，但不清除用户可读的最终结果数组。

### 18.2 `_restore_iteration`

```python
def _restore_iteration(self, iteration: int) -> None:
    self.mujoco_backend.restore_iteration(iteration)
    self.vbd_backend.restore_iteration(iteration)
    self.collision_pipeline.restore_iteration(iteration)
```

iteration 0 也必须使用相同逻辑建立 history seed，避免第一轮和后续轮路径不同而导致
Graph/eager 差异。

### 18.3 `_commit_substep`

```python
def _commit_substep(self, state_out: State, contacts: Contacts) -> None:
    self.mujoco_backend.commit_substep(self.runtime.final_mujoco_state)
    self.vbd_backend.commit_substep(self.runtime.final_vbd_state, contacts)
    self._reconcile_public_state(state_out)
    self.convergence.finalize()
    self._validate_overflow_and_nonfinite()
```

先提交私有事务，再一次性 reconcile 公开 state。若 `fail_on_overflow`，overflow
检查必须在公开状态部分提交前完成；CUDA Graph 下把 flag 留在 device，Graph 外的显式
诊断点或测试读取并抛错。

### 18.4 `_abort_substep`

```python
def _abort_substep(self) -> None:
    self.mujoco_backend.abort_substep()
    self.vbd_backend.abort_substep()
    self.collision_pipeline.abort_substep()
    self.runtime.restore_public_snapshot()
```

失败不能留下只推进了 MuJoCo 或只推进了 VBD 的半提交状态。

## 19. Reset、模型更新和 BVH

### 19.1 Reset

公共 reset 只转发给已选 backend：

```python
def reset(self, state, world_mask=None, flags=None):
    mask = self._normalize_world_mask(world_mask)
    self.backend.reset(state, mask, flags)
```

双向 backend 的内部实现为：

```python
def reset(self, state, mask, flags=None):
    self.mujoco_backend.reset(state, mask, flags)
    self.vbd_backend.reset(state, mask, flags)
    self.collision_pipeline.reset(mask)
    self.convergence.reset(mask)
    self.feedback.reset(mask)
    self._reset_runtime_snapshots(state, mask, flags)
```

masked reset 必须同时清理：

- proxy wrench warm start；
- Aitken residual；
- contact match/history；
- VBD self-contact 与 DAT；
- pneumatic history；
- MuJoCo sleeping bookkeeping；
- overflow/nonfinite flags。

### 19.2 `notify_model_changed`

公共入口先转发给 backend。下例是双向 backend 的实现；纯分支和单向分支只通知自己实际
构造的 private core：

```python
def notify_model_changed(self, flags: ModelFlags | int) -> None:
    self.mujoco_backend.notify_model_changed(flags)
    self.vbd_backend.notify_model_changed(flags)
    if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
        self.contact_routing = build_contact_routing(...)
        self.collision_pipeline.rebuild()
    if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
        self.effective_mass.invalidate()
```

若修改会改变 capacity、shape subset、ownership 或 Graph topology，必须要求重建 solver，
不能在捕获后隐式 resize。

### 19.3 `rebuild_bvh`

```python
def rebuild_bvh(self, state: State) -> None:  # two-way backend
    self.vbd_backend.rebuild_bvh(state)
    self.collision_pipeline.rebuild_dynamic_bvhs(state)
```

纯 MuJoCo 和 kinematic passthrough 分支不暴露虚假的 VBD BVH rebuild；调用时应是明确 no-op
或根据统一 API 合同抛出能力错误，不能为了满足调用而构造空 VBD core。

## 20. CUDA Graph 与性能合同

### 20.1 Capture 前条件

构造后至少执行一次 eager warm-up，使**所选 backend 实际需要的**以下对象全部分配完成：

- MuJoCo Warp private data；
- VBD rigid/soft history；
- full-surface active candidate buffers；
- contact matching scratch；
- deterministic sort scratch；
- feedback contribution buffers；
- effective-mass buffers；
- pneumatic state；
- Aitken reduction scratch。

不适用对象必须保持未构造，不能为了统一 capture 流程预分配。例如 pure VBD 不应出现
MuJoCo data，one-way 不应出现 Aitken scratch，passthrough 不应出现 contact buffer。

捕获期间禁止：

- `.numpy()`；
- host contact-count 分支；
- resize；
- 新建 `Contacts`；
- 改 coupling iteration 数；
- 改 full-surface shape subset；
- 改 deterministic 模式。

### 20.2 固定拓扑

双向 Graph 包含：

```text
begin snapshot
K * (
    restore
    MuJoCo
    sync/rewind
    effective mass
    collision
    VBD
    harvest
    relaxation
)
commit/reconcile
```

使用单 CUDA stream。此前 MJVBD_V2 的双 stream 实验在代表场景没有收益，不恢复。

其他 backend 捕获各自最短的静态拓扑：

```text
pure_vbd_*                VBD/contact
pure_mujoco               MuJoCo
kinematic_passthrough     copy/derived-transform
one_way_kinematic_soft    sync + collision + VBD-soft + reconcile
one_way_kinematic_full    sync + collision + VBD-full + reconcile
one_way_dynamic_soft      MuJoCo + sync + collision + VBD-soft + reconcile
one_way_dynamic_full      MuJoCo + sync + collision + VBD-full + reconcile
```

纯分支和单向分支不得因为统一 Graph wrapper 而捕获空 harvest、relaxation、restore 或
effective-mass launch。

### 20.3 必须保留的优化

- shape-major full-surface AABB rejection；
- AABB-active edge/face candidate compaction；
- persistent full-surface worker；
- active-prefix dual update 的 batch gate；
- dense rigid-side soft-contact parallel reduction；
- particle-color contact membership mask；
- self-contact active-count traversal；
- surface-only tile specialization；
- device-resident material selector；
- world-compatible capacity sizing。

### 20.4 新耦合路径的融合候选

以下融合必须分别 A/B 后决定，不因文档列出就自动保留：

1. `sync_proxy + rewind`；
2. `harvest + per-proxy residual write`；
3. `relaxation + MuJoCo force scatter`；
4. final owned-state reconcile；
5. overflow flag reduction。

每项都要记录 isolated kernel 和完整 frame 数据。

## 21. CPU、CUDA、deterministic 和梯度

### 21.1 CPU

- 使用相同 ownership、contact routing、iteration 和 relaxation；
- 可以使用 Python 固定 `for` 循环；
- MuJoCo CPU 与 VBD CPU 必须在同一 world layout；
- 不支持 CUDA Graph 和 MuJoCo Warp sleeping；
- 作为 contact keys、wrench、状态和事务恢复的参考路径。

解析 primitive 的 full-surface contact 必须支持 CPU。当前 texture volume SDF 是
CUDA-only，因此 mesh/convex full-surface 在 CPU 上若没有可用的 CPU SDF evaluator，
必须在构造时明确报错；CPU/CUDA contact-key 对照使用解析形状。不能静默退化成仅
point contact。

### 21.2 CUDA

- 默认高性能路径；
- MuJoCo Warp 和 VBD 位于同一 device；
- 所有 coupling state device-resident；
- 不允许 CPU MuJoCo + CUDA VBD 每 iteration 往返传输作为默认模式。

### 21.3 Deterministic

`deterministic=True` 时：

- collision pipeline 使用稳定 contact key 排序；
- rigid/soft feedback 先写 contribution，再稳定 segmented reduce；
- Aitken dot product 使用固定树形归约；
- coupling iteration 数固定；
- 禁止依赖 atomic accumulation 顺序；
- eager 和 Graph 必须重复一致。

deterministic 路径允许较慢，但不能改变接触集合和 force law。

### 21.4 `requires_grad`

第一版构造时对 `model.requires_grad` 或 requires-grad contacts 显式报错。原因是 MuJoCo
step、VBD contact history、Aitken 和 transaction restore 尚无完整联合 backward。
这是一项明确的 solver 能力边界，不允许静默走无梯度路径。

## 22. Overflow 与错误处理

### 22.1 固定容量

至少保留：

- `rigid_contact_max`；
- `soft_contact_max`；
- rigid body contact row capacity；
- body-particle contact row capacity；
- VT row capacity；
- EE row capacity；
- full-surface active edge/face capacity；
- deterministic contribution capacity。

### 22.2 检查函数

```python
def _validate_overflow_and_nonfinite(self) -> None: ...
```

错误消息必须报告：

- 哪个 buffer overflow；
- capacity；
- device count 或 overflow maximum；
- world（如果可得）；
- 建议增加哪个构造参数。

Graph 内只设置 flag，不能 host synchronize。测试模式和显式 diagnostics polling 可以在
Graph 外读取。

## 23. 测试文件和函数级计划

### 23.1 独立性、baseline 与静态 dispatch

新建 `newton/tests/test_solver_mujoco_vbd_architecture.py`：

```python
FORBIDDEN_PRODUCTION_IMPORT_PREFIXES = (
    "newton._src.solvers.coupled",
    "newton._src.solvers.mjvbd_v2",
    "newton._src.solvers.solver_mujoco",
    "newton._src.solvers.solver_vbd",
)


def test_production_package_has_no_forbidden_solver_imports(): ...
def test_private_baseline_manifest_is_complete(): ...
def test_private_baseline_file_hashes_match_or_are_explained(): ...
def test_auto_backend_selection_matrix(): ...
def test_explicit_backend_selection_matrix(): ...
def test_reject_incompatible_mode_options(): ...
def test_backend_kind_is_immutable_after_construction(): ...
def test_pure_vbd_does_not_allocate_mujoco_or_coupling(): ...
def test_pure_mujoco_does_not_allocate_vbd_or_coupling(): ...
def test_passthrough_allocates_no_solver_core(): ...
def test_one_way_immovable_allocates_no_feedback_state(): ...
def test_two_way_allocates_complete_transaction_state(): ...
```

第一项使用 AST 遍历 `newton/_src/solvers/mujoco_vbd/**/*.py` 的 `Import` 和
`ImportFrom`，不接受懒导入、函数内导入或字符串动态导入绕过。manifest 测试读取
`PRIVATE_BASELINE.md` 中机器可解析的 source path/hash 表；发生有意分叉时必须同时更新
差异说明，避免私有副本无记录漂移。

### 23.2 纯求解能力与 MJVBD_V2 等价性

新建 `newton/tests/test_solver_mujoco_vbd_modes.py`：

```python
def test_pure_vbd_cloth(): ...
def test_pure_vbd_tet(): ...
def test_pure_vbd_spring(): ...
def test_pure_vbd_rigid(): ...
def test_pure_vbd_rigid_soft_mix(): ...
def test_pure_vbd_pneumatic(): ...
def test_pure_mujoco_articulation(): ...
def test_kinematic_passthrough(): ...


def test_pure_vbd_soft_matches_mjvbd_v2_baseline(): ...
def test_pure_vbd_full_matches_mjvbd_v2_baseline(): ...
def test_one_way_kinematic_soft_matches_mjvbd_v2_baseline(): ...
def test_one_way_kinematic_full_matches_mjvbd_v2_baseline(): ...
def test_one_way_dynamic_soft_matches_mjvbd_v2_baseline(): ...
def test_one_way_dynamic_full_matches_mjvbd_v2_baseline(): ...
```

等价性测试固定同一 baseline commit、seed、dt、iteration、capacity 和 backend，逐项对比：

- backend kind 和启用的 pipeline；
- sorted VT/EE/rigid/soft contact keys；
- active count、overflow flag 和 DAT `t`；
- 每个 substep 的 particle/body pose、velocity；
- pneumatic pressure/volume history；
- CUDA Graph launch topology 与 kernel count；
- 完整 frame GPU 时间。

数值容差按 deterministic bitwise、同 device 浮点容差和 CPU/CUDA 参考三档定义。单向分支
必须通过这些测试证明“退化为 MJVBD_V2”是行为合同，而不是类名或大致流程相似。

### 23.3 单向 moving-collider 与双向柔顺语义

```python
def test_one_way_proxy_has_zero_solve_inverse_mass_and_inertia(): ...
def test_one_way_proxy_pose_and_velocity_follow_source_exactly(): ...
def test_one_way_proxy_is_not_modified_by_vbd_contact(): ...
def test_one_way_allocates_no_feedback_or_effective_mass_state(): ...
def test_one_way_preserves_public_mass_metadata(): ...
def test_two_way_proxy_uses_articulated_effective_mass(): ...
def test_two_way_contact_wrench_is_equal_and_opposite(): ...
def test_two_way_finite_gain_servo_deviates_under_contact_load(): ...
def test_two_way_stiff_servo_tracks_target_without_hidden_ik(): ...
```

单向测试必须证明机器人是 prescribed moving boundary：VBD 接触不能改变其 state，且
reaction buffer 根本不存在。双向测试则必须证明有限质量 proxy、反力回传和 MuJoCo
重求解形成闭环。需要“接近严格轨迹但可受力”的场景使用有限但较高增益的 MuJoCo
servo；测试同时检查目标跟踪误差与接触载荷下的有限偏移，禁止在 VBD 后执行 IK 覆盖。

### 23.4 Ownership 与路由

新建 `newton/tests/test_solver_mujoco_vbd_ownership.py`：

```python
def test_closed_articulation_ownership(): ...
def test_reject_partial_joint_tree(): ...
def test_multiple_articulations(): ...
def test_floating_base_articulation(): ...
def test_reject_hard_kinematic_two_way(): ...
def test_contact_pair_routing_has_no_duplicates(): ...
def test_static_pair_auto_routing(): ...
def test_visual_shapes_do_not_enter_cross_contacts(): ...
```

### 23.5 一轮状态事务

`newton/tests/test_solver_mujoco_vbd_state.py`：

```python
def test_iteration_restore_reuses_same_initial_state(): ...
def test_mujoco_step_counter_advances_once(): ...
def test_vbd_history_commits_once(): ...
def test_dahl_history_commits_once(): ...
def test_dat_state_commits_once(): ...
def test_pneumatic_state_commits_once(): ...
def test_abort_restores_public_and_private_state(): ...
def test_masked_reset_clears_feedback_history(): ...
```

### 23.6 Feedback 正确性

`newton/tests/test_solver_mujoco_vbd_feedback.py`：

```python
def test_point_contact_equal_and_opposite_wrench(): ...
def test_edge_contact_equal_and_opposite_wrench(): ...
def test_face_contact_equal_and_opposite_wrench(): ...
def test_rigid_rigid_equal_and_opposite_wrench(): ...
def test_contact_torque_uses_body_com(): ...
def test_user_body_force_is_preserved(): ...
def test_vbd_internal_contacts_do_not_feed_mujoco(): ...
def test_mujoco_internal_contacts_are_not_duplicated_in_vbd(): ...
```

edge/face 测试必须证明当前通用 Proxy 会拒绝或漏掉的 full-surface reaction 已被新
kernel 正确回传。

### 23.7 Coupling 收敛

`newton/tests/test_solver_mujoco_vbd_coupling.py`：

```python
def test_zero_contact_matches_independent_solvers(): ...
def test_fixed_relaxation_matches_reference_iteration(): ...
def test_aitken_reduces_interface_residual(): ...
def test_dynamic_finger_deflects_under_soft_contact(): ...
def test_compliant_servo_tracks_without_contact(): ...
def test_compliant_servo_yields_under_contact(): ...
def test_nonfinite_feedback_aborts_transaction(): ...
def test_divergent_wrench_rolls_back_per_world(): ...
```

### 23.8 Contact 场景

`newton/tests/test_solver_mujoco_vbd_contacts.py`：

```python
def test_finger_against_coarse_cloth_full_surface(): ...
def test_finger_against_tet_surface(): ...
def test_robot_against_vbd_dynamic_rigid_body(): ...
def test_two_fingers_one_soft_object(): ...
def test_two_hands_one_object(): ...
def test_self_contact_and_cross_contact_coexist(): ...
def test_contact_overflow_is_reported(): ...
def test_mesh_proxy_requires_sdf(): ...
```

### 23.9 多 world、deterministic 和 Graph

`newton/tests/test_solver_mujoco_vbd_cuda.py`：

```python
def test_multiworld_contact_isolation(): ...
def test_multiworld_partial_reset(): ...
def test_cuda_graph_matches_eager(): ...
def test_cuda_graph_replays_fixed_topology(): ...
def test_deterministic_eager_repeats(): ...
def test_deterministic_graph_repeats(): ...
def test_cpu_cuda_contact_keys_match(): ...
def test_cpu_cuda_feedback_wrenches_match(): ...
def test_each_backend_graph_contains_no_unused_modules(): ...
```

### 23.10 对照测试

允许在测试中构造 `SolverCoupledProxy` 作为数值参考，但生产包不能导入它：

```python
def test_dedicated_solver_matches_generic_proxy_point_contacts(): ...
def test_dedicated_solver_matches_generic_proxy_rigid_contacts(): ...
```

full-surface 双向测试没有现成通用 Proxy 参考，应使用有限差分、直接 force-law 求值和
equal-and-opposite 检查建立独立 oracle。

## 24. 代表性示例

### 24.1 求解器能力示例

第一版必须包含以下 `Example` 类格式的独立例子：

```text
newton/examples/multiphysics/example_mujoco_vbd_modes.py
newton/examples/multiphysics/example_mujoco_vbd_kinematic_moving_collider.py
newton/examples/multiphysics/example_mujoco_vbd_two_way_gripper.py
```

`example_mujoco_vbd_modes.py` 用同一公共 solver API 展示 pure cloth、pure tet、pure VBD
rigid、pneumatic、pure MuJoCo、kinematic one-way 和 dynamic one-way；命令行模式在构造前
选择，运行中不切 backend。

`example_mujoco_vbd_kinematic_moving_collider.py` 展示 prescribed 机器人轨迹作为无限质量
moving collider：VBD 对象受接触影响，机器人轨迹保持不变，且不分配反馈状态。

双向 gripper 例子包含：

- 一个有限刚度 servo 机器人夹爪；
- 一块粗网格软体或布料；
- 指尖 full-surface SDF contact；
- 可切换 one-way/two-way 以展示机器人反作用；
- 显示 coupling residual、contact count 和最大穿透；
- `test_post_step()` 检查非有限值和 overflow；
- `test_final()` 检查机器人发生有限偏移、对象未穿透、residual 有界。

三个例子都实现 `test_final()` 或 `test_post_step()`，按仓库规范注册 README 命令并提供
320x320 JPEG。

### 24.2 MJVBD_V2 退化验收示例

第一版还必须把 `newton/examples/mjvbdv2` 中原先以 `final00` 标识的验收场景、螺母螺钉
场景和后续选定的纯 VBD/机器人操作场景重新写成以下十一个独立 demo：

| 现有行为基线 | 新的独立 demo |
| --- | --- |
| `example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_bimanual_fold_tshirt.py` |
| `example_mjvbd_v2_dexforce_w1_bimanual_plastic_bag_rod_final00.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_bimanual_plastic_bag_rod.py` |
| `example_vbd_mjvbd_v2_dexforce_recorded_plastic_inflatable_bag_pick_release_final00.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_recorded_plastic_inflatable_bag_pick_release.py` |
| `example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_recorded_soft_then_rigid_cube_into_bag.py` |
| `example_vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher_final00.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_right_hand_armadillo_into_gear_crusher.py` |
| `example_mjvbd_v2_bimanual_nut_bolt.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_bimanual_nut_bolt.py` |
| `example_mjvbd_v2_dexforce_realtime_plug_socket.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_dexforce_realtime_plug_socket.py` |
| `example_cloth_mjvbd_v2_dexforce_bimanual_place_tablecloth_waic_house.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_dexforce_bimanual_place_tablecloth_waic_house.py` |
| `example_mjvbd_v2_cloth_twist.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_cloth_twist.py` |
| `example_mjvbd_v2_dexforce_realtime_push_chair.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_dexforce_realtime_push_chair.py` |
| `example_mjvbd_v2_gear_crusher.py` | `newton/examples/mujoco_vbd/example_mujoco_vbd_gear_crusher.py` |

“重新写成独立 demo”是强制架构要求：

- 不修改或覆盖原 `examples/mjvbdv2` 文件；
- 新 demo 不得 import、继承或调用 `newton.examples.mjvbdv2` 中的 `Example`、kernel、
  solver builder 或运行时 helper；
- 新 demo 不得 import `SolverMJVBDV2`、`SolverCoupled` 或 `SolverCoupledProxy`；
- 每个文件定义自己的 `Example`、CLI、scene construction、step、render 和测试入口；
- 多个新 demo 可以共享 `newton/examples/mujoco_vbd/_shared/` 下新写的 solver-neutral
  asset/trajectory/IK helper，但不能让其中任何 helper 依赖旧 MJVBD_V2 示例包；
- 允许继续读取相同的 URDF、mesh、keyframe 和轨迹数据资产，以保证比较的是求解器而
  不是重新设计后的场景；
- 保持原例子的 topology 创建顺序、shape flags、碰撞/视觉 mesh 选择、材料、dt、
  substeps、VBD iterations、capacity、轨迹和 CUDA Graph 开关；任何必要差异必须写入
  demo 顶部的 parity 注释并进入 `OPTIMIZATION_LOG.md`。

九个包含 MuJoCo articulation proxy 的 demo 统一使用显式单向构造：

```python
solver = SolverMuJoCoVBD(
    model,
    joint_mode="kinematic",
    coupling_mode="one_way",
    contact_mode=SOURCE_CONTACT_MODE,
    vbd_options=SOURCE_VBD_OPTIONS,
    collision_options=SOURCE_COLLISION_OPTIONS,
)
```

两个无 articulation 的纯 VBD demo 显式使用 `joint_mode="dynamic"`、
`coupling_mode="auto"`，并断言 `pure_vbd_soft`；它们不得伪造空 articulation 来进入
one-way 分支。

构造后必须断言 `features.feedback_enabled is False`，并按源场景断言 backend、
`vbd_core` 和 `contact_pipeline`。其中 inflatable-bag pneumatic demo 必须明确断言：

```text
backend          = one_way_kinematic_soft
vbd_core         = full
contact_pipeline = soft
```

这十一个 demo 不是仅供人工观看。每个都必须移植且不得弱化原例子的 `test_post_step()` 和
`test_final()`，并增加统一检查：finite state、所有 overflow flag 为零、visual mesh 不参与
接触、CUDA Graph 请求后确实捕获、没有辅助隐藏手指 collider、没有 source feedback
buffer。场景特有的最终行为至少包括：

- T-shirt：实时 IK、碰撞 mesh/视觉 mesh 分离、布料状态有限；
- plastic-bag rod：双手轨迹误差有界、杆完成交接、袋子未从双手滑落；
- pneumatic plastic bag：pressure/volume/history 有限、塑性弯曲量受限且抓取产生塑性；
- soft-then-rigid cube：软体与动态刚体均完成物理释放，刚体最终落入袋内，无隐藏 pad；
- Armadillo crusher：手先接触并抬起 Armadillo，释放后 Armadillo 与齿轮接触并下落；
- nut/bolt：左手仅靠与可见手指对应的原生碰撞几何保持螺钉；右手仅由中指和螺母
  可见表面相匹配的碰撞面摩擦驱动；每次有效 stroke 至少旋紧 `60 deg`；螺纹接触
  存在、螺母不外退或离轴、最终露出螺钉端部，并且不存在隐形摩擦球或直接角速度/
  位姿驱动。
- plug/socket：完成抓取、抬升、对齐、插入、释放和撤回，插头最终由插座保持；
- tablecloth：完成双手铺放并通过最终桌面覆盖/高度断言；
- cloth twist：纯 VBD 自碰撞布料完成规定边界扭转且状态、速度有限；
- chair push：实时 IK 无轨迹缓存，椅子仅由手部接触产生满足阈值的平移和转动；
- gear crusher：规定运动齿轮与 Armadillo 发生接触并产生满足阈值的变形。

新增 `newton/tests/test_solver_mujoco_vbd_examples.py`：

```python
def test_examples_have_no_mjvbd_runtime_imports(): ...
def test_tshirt_one_way_parity(): ...
def test_plastic_bag_rod_one_way_parity(): ...
def test_pneumatic_bag_one_way_parity(): ...
def test_soft_then_rigid_cube_one_way_parity(): ...
def test_armadillo_crusher_one_way_parity(): ...
def test_bimanual_nut_bolt_one_way_parity(): ...
def test_cloth_twist_pure_vbd_parity(): ...
def test_push_chair_one_way_parity(): ...
def test_gear_crusher_pure_vbd_parity(): ...
```

测试中的 reference runner 可以运行原 MJVBD_V2 demo，但新的生产 demo 不能导入它。
reference 与新 demo 使用相同 device、seed、参数和帧数，比较 topology counts、逐 phase
轨迹采样、sorted contact keys、overflow、DAT、pneumatic history 和关键 checkpoint。
deterministic 短窗口要求 contact keys 完全一致、state 符合 23.2 的数值容差；完整长程
demo 因接触分叉不要求逐粒子 bitwise 一致，但必须同时通过相同的原始最终行为断言和
场景阈值。只有 core 等价测试与这十一个完整 demo 验收同时通过，才可声称单向模式能够
退化为 MJVBD_V2。

十一个新 demo 都必须注册独立 `python -m newton.examples ...` 命令并各自提供 320x320
JPEG；不能用一个带场景下拉框的综合 demo 代替。

## 25. Benchmark 计划

### 25.1 对比对象

每个 benchmark 使用完全相同的：

- dt/substeps；
- VBD iterations；
- coupling iterations；
- contact capacity；
- full-surface shape subset；
- MuJoCo solver/integrator；
- 初始状态和控制；
- CUDA Graph 模式。

按模式选择公平对比对象：

1. pure VBD、pure MuJoCo、passthrough 和所有 one-way 分支：同 baseline commit 的
   `SolverMJVBDV2` 对应静态分支；
2. two-way point/rigid contact：通用 `SolverCoupledProxy`；
3. two-way full-surface：没有通用 Proxy 等价路径，使用独立求值 oracle 做正确性验证，
   性能只比较专用实现自身的模块占比和优化前后版本；
4. 新 `SolverMuJoCoVBD` 的各静态 backend。

私有 core 和 MJVBD_V2 baseline 必须使用同一源提交；否则性能差异会混入上游变化，结果
无效。

### 25.2 场景矩阵

```text
bench_mujoco_vbd_pure_cloth
bench_mujoco_vbd_pure_tet
bench_mujoco_vbd_pure_rigid
bench_mujoco_vbd_pure_pneumatic
bench_mujoco_vbd_one_way_kinematic_soft
bench_mujoco_vbd_one_way_kinematic_full
bench_mujoco_vbd_one_way_dynamic_soft
bench_mujoco_vbd_one_way_dynamic_full
bench_mujoco_vbd_single_finger_cloth
bench_mujoco_vbd_two_hand_bag
bench_mujoco_vbd_tet_grasp
bench_mujoco_vbd_rigid_object_grasp
bench_mujoco_vbd_pneumatic_contact
bench_mujoco_vbd_1024_worlds
bench_mujoco_vbd_tshirt
bench_mujoco_vbd_plastic_bag_rod
bench_mujoco_vbd_pneumatic_bag
bench_mujoco_vbd_soft_then_rigid_cube
bench_mujoco_vbd_armadillo_crusher
bench_mujoco_vbd_bimanual_nut_bolt
```

### 25.3 分段计时

记录 GPU event median：

```text
snapshot/restore
MuJoCo
proxy sync/rewind
effective mass
collision total
full-surface candidate prune/compact
VBD solve
rigid feedback harvest
soft point/edge/face feedback harvest
relaxation
commit/reconcile
complete substep
complete frame
```

非耦合 backend 的不存在模块记录为 `not allocated`，不能用零耗时空 launch 伪装。每个
case 同时记录 device allocation bytes、captured kernel count 和每类固定容量，验证 clean
branch 既减少时间也减少内存。

每个结果记录 GPU、CUDA、Warp、MuJoCo Warp 版本、world count、拓扑数、接触数、
warm-up、sample 数和统计量。交互 viewer FPS 不能作为唯一证据。

## 26. 第一版验收门槛

所有门槛均为 release-blocking：

### 26.1 正确性

- 生产包对 `mjvbd_v2`、仓库 MuJoCo/VBD solver 和 `coupled` 包的 runtime import 数为零；
- `PRIVATE_BASELINE.md` 完整记录所有 private core 文件的来源、hash 和有意差异；
- pure cloth/tet/spring/rigid/刚软混合/pneumatic 与 pure MuJoCo 均可独立运行；
- 所有 one-way 对应分支在同 baseline 上通过 MJVBD_V2 contact/history/final-state 等价测试；
- 24.2 列出的十一个场景均存在新的独立 demo，且静态 import 检查确认它们不依赖
  `examples.mjvbdv2`、`SolverMJVBDV2` 或通用 coupled solver；
- 十一个 demo 的短窗口 contact/state parity 与完整长程原始行为断言全部通过；任何一个
  demo 只能启动、但未完成原任务，均视为 one-way 退化验收失败；
- 构造期 backend 静态且 clean branch 不分配未使用 solver、proxy、feedback 或 transaction；
- point/edge/face/rigid cross-contact wrench 等大反向，相对误差小于 1%；
- 同一 substep 的所有 history 只提交一次；
- 无接触时等价于独立 MuJoCo + VBD；
- 共同支持的解析形状上 CPU/CUDA sorted contact keys 一致；
- deterministic eager/Graph 重复运行一致；
- reset、masked reset、异常 abort 不留下半提交状态；
- full-surface 双向路径不再触发现有 proxy 的 `NotImplementedError`。

### 26.2 效果

- 快速手指压粗网格布料时不会从三角形内部穿过；
- 代表性抓取最大穿透不超过 `0.25 * particle_radius`，或场景定义的更严格阈值；
- 有限刚度 servo 在接触时产生可测机器人偏移；
- 2--4 轮 coupling residual 稳定下降，无持续两周期振荡；
- 摩擦方向、stick/slip 和 contact torque 正确；
- 不出现因高反馈刚度导致的非有限速度或弹飞；
- 十一个 MJVBD_V2 parity demo 分别完成折衣、双手提袋、气动袋抓放、软体/刚体入袋、
  Armadillo 入齿轮机、纯接触摩擦拧螺母、插头插接、铺桌布、扭布、推椅和齿轮挤压任务；不得用隐藏 collider、直接对象驱动或
  放宽源 `test_final()` 阈值换取通过。

### 26.3 性能

- pure 与 one-way 分支相对同 baseline MJVBD_V2 对应分支，代表性 frame 中位数不得慢
  2% 以上，device allocation 和 captured kernel count 不得因统一 API 无故增加；
- 十一个 parity demo 在相同 device、Graph、dt/substep 和渲染关闭的 benchmark 中，完整
  physics frame GPU median 均不得比对应 MJVBD_V2 demo 慢 2% 以上；任一场景超限都要
  profile、记录并修复，不能只用跨场景平均值掩盖回退；
- kinematic passthrough、pure MuJoCo 和 pure VBD 不得出现另一 solver core 的 launch；
  不得出现 MuJoCo、feedback、Aitken 或 outer iteration 成本；
- 同 iteration 数下，相比通用 `SolverCoupledProxy`，代表性整帧至少快 15%；
- 专用 orchestration（不含 MuJoCo、collision、VBD 本体）低于整帧 GPU 时间 5%；
- CUDA Graph 与 eager 数值一致，并显著降低 launch overhead；
- 1024-world batch 不出现按 world 数重复放大的 capacity；
- full-surface feedback harvest 使用 active prefix，不扫描未使用的大容量尾部；
- 新的任何 kernel fusion 都必须通过 isolated 15% 或代表性 frame 5% 的保留门槛。

如果同配置下专用 solver 达不到性能门槛，必须先记录 profile 和原因，不能以“代码更
专用”为由保留没有收益的额外实现。

### 26.4 私有优化迁移

`OPTIMIZATION_LOG.md` 对每项 MJVBD_V2 已保留优化记录：source commit、迁移文件、适用
backend、microbenchmark、frame benchmark 和 Go/No-Go。第一版至少完成本文 3.3 列出的
全部 retained optimization；下列已验证无收益方案不得在复制 core 时复活：canonical
EE、active VT/EE stream、rest exclusion CSR、Morton order、EE source-color gate。

私有副本之后的优化可以独立演进，但每笔保留规则不变：isolated kernel 至少快 15% 或
代表性整帧至少快 5%，同时接触集合、overflow、DAT 和最终状态符合该模式的等价合同。

## 27. 完成定义

第一版完成不是指类能够构造或单个抓取例子能够运行，而是：

1. private MuJoCo、full VBD、soft VBD、contact 和 pneumatic core 已复制到本目录并有
   `PRIVATE_BASELINE.md` 可追溯记录；
2. 生产代码通过 forbidden-import AST 测试，不依赖 MJVBD_V2、仓库 SolverMuJoCo、
   SolverVBD 或 SolverCoupledProxy；
3. 本文中的模块、backend 和函数均已实现；
4. pure VBD soft/full、pure MuJoCo、passthrough、两种 kinematic one-way、dynamic
   one-way 和 two-way 全部通过功能测试；
5. 单向模式通过与 MJVBD_V2 的数值、contact/history、Graph 和性能等价门槛；
6. point、edge、face、rigid 四类跨 solver feedback 全部可用；
7. 状态事务覆盖 MuJoCo、VBD、self-contact、DAT、摩擦和 pneumatic history；
8. 动态/servo 机器人受 VBD 反力影响；kinematic one-way 严格保持 prescribed 轨迹且不回传；
9. MJVBD_V2 retained optimizations 已迁移并逐项有数据，已拒绝优化未复活；
10. CPU、CUDA、deterministic、multiworld、reset、Graph 测试通过；
11. 三个能力示例和十一个 MJVBD_V2 parity 独立 demo 均完成、注册并提供截图；
12. 十一个 parity demo 不导入旧示例/求解器，并通过各自未弱化的完整任务断言；
13. benchmark 达到正确性、效果和性能门槛；
14. `uvx pre-commit run -a` 和目标 Newton tests 通过；
15. 添加 Towncrier fragment；
16. 设计、baseline、优化迁移和性能数据写入独立 solver 的决策记录。

在这些条件全部满足前，不对外导出 `SolverMuJoCoVBD`。
