# Newton 充气袋仿真：现有算法调研与函数级落地方案

> 文档状态：技术调研与实现设计稿
> 基线日期：2026-08-11
> Newton 基线：`v1.5.0`，tag commit `cca3bb8a17a3620a1343df3cf12c625e4161b317`
> 同步核对的 `main`：`98ba6dc65176d3e67d8b3b3efa94b941920e9b51`
> 目标场景：完全封闭、气体质量恒定的薄膜袋/软包装，包含大变形、褶皱、自碰撞、外力挤压，以及由机器人轨迹驱动的单向抓取
> 交付边界：先完成独立空腔袋受外力变形；随后完成机器人到袋子的单向接触。首期不实现机器人反力回传或跨求解器双向耦合。

## 0. 结论先行

建议在 Newton 中采用“薄膜壳体 + 单个气腔标量状态”的组合：

1. **薄膜**：继续使用 `SolverVBD` 的稳定 Neo-Hookean 三角膜 + 二面角弯曲。
2. **气体**：增加一个全局 `PneumaticCavity`，根据闭合表面的当前体积计算均匀腔压。
3. **耦合方式**：每个 VBD 外迭代重新计算一次体积和压力，把压力力及其正定局部切线写入 VBD 已有的 `particle_forces` / `particle_hessians`，再执行现有的按颜色顶点块下降。
4. **接触与交付顺序**：第一步使用外部粒子力或运动压板驱动袋体，并使用 VBD 内部的 vertex-triangle / edge-edge 自碰撞。第二步将机器人夹爪作为按给定轨迹更新的运动刚体，使用 Newton 1.5 的 full-surface rigid-soft contact；袋体反力不回传给机器人。
5. **默认气体模型**：低速抓取、挤压用封闭等温理想气体；快速压缩/释放用封闭绝热模型；目标体积约束只作为视觉模式或硬体积备选，不作为默认物理模型。

推荐的一期求解能量为：

\[
E(x)=E_{\text{inertia}}+E_{\text{membrane}}+E_{\text{bend}}+E_{\text{contact}}+\sum_c \Psi_c(V_c(x)).
\]

本文只处理**封闭袋子的结构变形**。腔内不离散流场，整个 cavity 每次迭代只有体积、压力和体积变化率等少量标量；不设计进出气、泄漏、阀门或开边界接口。机器人仅作为运动学驱动的碰撞几何，不属于本方案的动力学反馈环。

## 1. 问题边界与分级

### 1.1 两个目标等级

| 等级 | 目标 | 气体模型 | 推荐算法 | 典型用途 |
|---|---|---|---|---|
| L1 | 看起来像充气袋 | 指定压力或目标体积 | VBD + prescribed pressure / target volume | 动画、交互预览 |
| L2 | 挤压时压力和回弹可信 | 封闭均匀气体，满足状态方程 | VBD + isothermal/adiabatic cavity | 抓取、包装、机器人操作 |

本方案的正式落地目标是 **L2**，同时覆盖 L1。

### 1.2 两步交付边界

| 步骤 | 要完成的能力 | 输入 | 明确不做 |
|---|---|---|---|
| 第一步：独立空腔袋 | 密封袋在重力、`state.particle_f` 外力和运动压板作用下稳定变形、压缩和回弹 | 闭合三角网格、膜参数、气腔参数、外力或压板轨迹 | 机器人模型、机器人反力、跨求解器耦合 |
| 第二步：机器人单向抓取 | 按给定轨迹运动的机器人夹爪通过刚柔接触抓取、挤压和释放袋体 | 第一步能力 + 夹爪/机器人碰撞几何及其位姿、速度轨迹 | 机器人受袋体反力后的动力学响应、MuJoCo/Kamino/VBD coupled solve |

“空腔”在本文中仍指封闭的均匀气腔，而不是无气体的薄壳；第一步的气腔压力会随袋体受压体积变化而更新。

### 1.3 第一步成立的物理假设

- 袋体三角网格形成闭合、可定向的二维流形。
- 腔内声速传播相对机械运动足够快，因而每个腔体内部压力近似均匀。
- 薄膜厚度远小于袋体尺度，可用膜能 + 弯曲能表示。
- 整段仿真中气体质量恒定；初始化时可用 `pressure_scale` 做纯数值渐入，但不表示真实进气过程。
- 接触不改变网格拓扑，不发生撕裂或永久开孔。

### 1.4 本方案覆盖的封闭袋工况

| 工况 | 模型选择 | 关键验证量 |
|---|---|---|
| 缓慢挤压封闭袋 | 等温理想气体 | `p_abs * V` 近似恒定 |
| 快速压缩/释放封闭袋 | 绝热理想气体 | `p_abs * V**gamma` 近似恒定 |
| 只要求指定鼓起程度 | 指定压力或目标体积 | 平衡体积与外形 |
| 运动压板挤压 | 等温或绝热 + 体积阻尼 | 压力滞后、回弹和接触稳定性 |
| 运动学夹爪抓取 | 等温或绝热 + full-surface rigid-soft contact | 抓取、挤压、释放期间无穿层和爆炸 |
| 多个独立密闭腔 | 每腔一个 cavity | 各腔质量和状态互不串扰 |
| 多腔共享隔膜 | 同一三角面以相反符号加入两个 cavity | 隔膜载荷等于两侧压差 |

## 2. 现有算法与工业实现调研

### 2.1 方案总览

| 算法族 | 核心未知量 | 优点 | 主要缺点 | Newton 适配性 |
|---|---|---|---|---|
| 指定均匀压力 follower load | 顶点位置 | 最简单；适合标定和压力驱动 | 不自动满足气体状态方程 | 很高，可作为 P0 |
| PBD/XPBD 全局体积约束 | 顶点位置、约束乘子 | 稳、直观、实时；目标体积易控制 | 目标体积不等于真实封闭气体；全局约束并行性一般 | 高，适合作为备选硬体积模式 |
| 控制体积 + 理想气体 | 壳体位置、腔体标量状态 | 物理量清晰；只增加 O(面数) 成本 | 假设腔内均匀；全局体积带来长程耦合 | **最高，推荐** |
| 隐式壳 + 压力一致切线 | 全局位移 | 高精度、高刚度下收敛好 | 需要全局矩阵和线性求解 | 中，Style3D/新全局求解器更合适 |

### 2.2 PBD/XPBD 全局体积

全局体积约束可写成：

\[
C(x)=V(x)-V_t=0.
\]

XPBD 的标量乘子更新为：

\[
\Delta \lambda =
\frac{-C-\tilde\alpha\lambda}
{\sum_i w_i\|\nabla_i C\|^2+\tilde\alpha},
\qquad
\tilde\alpha=\frac{\alpha}{\Delta t^2}.
\]

这类方法的工业代表包括：

- NVIDIA PhysX Particle Inflatable：闭合 cloth mesh 加 `pressure`/volume target；文档明确要求 watertight、non-overlapping mesh，`pressure > 1` 表现为气球。
- Houdini Vellum Pressure：在可拉伸布料上增加一个连接全部点的全局压力/体积约束，保存原始体积并控制目标体积。

优点是参数直观、易于做“充到 1.2 倍体积”。缺点是它模拟的是目标体积约束，不是气体被压缩后自然升压。对机器人挤压，如果只使用固定目标体积，袋子通常会显得过硬或需要人为 compliance。

### 2.3 控制体积 + 理想气体

工业有限元通常把袋内空间看成一个流体腔体标量自由度，壳体负责几何体积，气体状态方程负责压力：

- Abaqus Fluid Cavity 支持 gauge/ambient pressure、理想气体、质量守恒、等温和绝热过程；其“表面几何体积 + 腔体标量压力”正是本方案参考的有限元建模方式。
- LS-DYNA 传统 airbag control-volume 方法同样采用腔内均匀压力，可作为封闭袋压力—体积耦合的工业参照。
- SOFA `SurfacePressureForceField` 支持三角/四边形表面压力、体积守恒模式和 tangent stiffness，代表开源隐式 follower-load 路线。

对封闭软袋，这是精度、稳定性和 GPU 成本之间最合适的折中，也是本方案的主路线。

### 2.4 接触算法

充气袋能否成立，往往首先取决于接触而不是气体模型。必须同时处理：

- 袋体内部两层膜的 vertex-triangle 接触；
- 褶皱处的 edge-edge 接触；
- 薄膜与夹爪/平台之间的连续表面接触；
- 大压力下的非穿透步长截断。

C-IPC 是高鲁棒、无交叉薄壳接触的参考上限；Newton VBD 已经具备三角网格 vertex-triangle、edge-edge 自碰撞以及 penetration-free truncation，适合先沿现有体系扩展，而不是更换整个接触框架。

## 3. Newton 最新代码审计

### 3.1 版本结论

- 发布基线：[Newton v1.5.0](https://github.com/newton-physics/newton/releases/tag/v1.5.0)，commit [`cca3bb8`](https://github.com/newton-physics/newton/commit/cca3bb8a17a3620a1343df3cf12c625e4161b317)。
- 同日核对 `main` commit [`98ba6dc`](https://github.com/newton-physics/newton/commit/98ba6dc65176d3e67d8b3b3efa94b941920e9b51)。该提交相对 tag 没有修改 VBD、XPBD、Style3D 或 coupled solver 的求解流程；相关 `builder/model/collide` 差异主要是刚体 mesh edge 的碰撞缓存，不影响本方案的扩展点。
- 因而函数级方案以 `v1.5.0` 可复现源码为准，同时适用于上述 `main`。

### 3.2 当前调用链

| 阶段 | 当前函数 | 与充气袋的关系 |
|---|---|---|
| 建模 | `ModelBuilder.add_cloth_mesh()` | 创建粒子、三角形、弯曲边；当前没有气腔概念 |
| 模型固化 | `ModelBuilder.finalize()` | 上传 `tri_*`、`edge_*`，构建 `soft_mesh_adjacency` |
| 状态 | `Model.state()` | 分配 `particle_q/qd/f`；`particle_f` 是现成的显式压力原型入口 |
| 外部碰撞 | `CollisionPipeline.collide()` | 生成 rigid-soft contacts；1.5 可选 full-surface contact |
| VBD 初始化 | `SolverVBD._initialize_particles()` | `forward_step()` 使用 `state_in.particle_f` 形成惯性目标 |
| VBD 外迭代 | `SolverVBD.step()` | 每步执行 `iterations` 次 rigid AVBD + particle VBD |
| 粒子迭代 | `SolverVBD._solve_particle_iteration()` | 清空内部力/Hessian，按颜色积累接触和弹性并更新顶点 |
| 薄膜能 | `solve_elasticity()` / `solve_elasticity_tile()` | 每顶点聚合相邻三角膜、弯曲边和四面体能量 |
| 速度更新 | `SolverVBD._finalize_particles()` | `update_velocity()` 根据位置差更新速度 |

关键源码：

- [`solver_vbd.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/solvers/vbd/solver_vbd.py)
- [`particle_vbd_kernels.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/solvers/vbd/particle_vbd_kernels.py)
- [`builder.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/sim/builder.py)
- [`model.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/sim/model.py)
- [`state.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/sim/state.py)
- [`collide.py`](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/sim/collide.py)

### 3.3 现有求解器的适配判断

| Newton 求解器 | 当前相关能力 | 缺口 | 结论 |
|---|---|---|---|
| `SolverVBD` | 稳定 Neo-Hookean 膜、弯曲、3×3 顶点块 Hessian、自碰撞、full-surface rigid-soft | 无气腔能量 | **主实现目标** |
| `SolverXPBD` | 距离、弯曲、四面体 FEM；tet 内有局部体积保持 | 没有闭合三角表面的全局体积约束；rigid-soft 只接受 particle contact | 可实现备选 target-volume，但不作为主壳求解器 |
| `SolverStyle3D` | 非线性 PD + PCG、各向异性布料 | 当前无压力/气腔；接入会改变全局矩阵/预条件器 | 真实编织/薄膜各向异性是后续候选 |
| `SolverCoupledProxy/ADMM` | MuJoCo/Kamino/VBD 等跨求解器耦合 | 仍是 experimental；full-surface soft contact 当前只有 standalone VBD 消费 | 机器人双向耦合需单独验证 |

### 3.4 最有价值的 1.5 代码能力：自定义频率

Newton 1.5 的 `ModelBuilder.CustomFrequency` 和 `CustomAttribute` 已能：

- 定义 `pneumatic:cavity`、`pneumatic:face` 这样的新实体频率；
- 把属性分别挂到 `Model`、`State`、`Control`；
- 声明属性引用 `triangle`、`particle`、`world` 或另一个自定义频率；
- 在 `add_builder()`、`add_world()`、`replicate()` 时自动偏移引用；
- 让 coupled `ModelView` 根据属性引用图做 custom frequency compaction。

因此首个正式版本**不需要**向 `Model.AttributeFrequency` 增加核心枚举，也不需要直接修改 `Model.state()` / `Model.control()`。这是推荐实现与旧式“大改核心数组”方案的关键区别。

## 4. 推荐数学模型

### 4.1 有向闭合表面体积

对于气腔 \(c\) 的有向三角面集合 \(F_c\)：

\[
V_c(x)=\frac{1}{6}\sum_{(i,j,k)\in F_c}s_f\,
(x_i-a_c)\cdot\left[(x_j-a_c)\times(x_k-a_c)\right],
\]

其中 \(s_f\in\{-1,+1\}\) 是该面在当前 cavity 中的方向，\(a_c\) 取该腔体某个 anchor particle 的当前位置。使用移动 anchor 而不是世界原点，可以降低袋子远离原点时 float32 的消减误差。

对闭合曲面，顶点体积梯度可以按相邻面的面积法向稳定计算：

\[
g_i=\frac{\partial V}{\partial x_i}
=\sum_{f\ni i}s_f\frac{(x_j-x_i)\times(x_k-x_i)}{6}.
\]

每个三角面对三个顶点分配相同的 \(A_f n_f/3\)，因此恒定压力的总力与总力矩在闭合表面上应接近零。

### 4.2 四种压力模式

#### A. 指定表压

\[
\Delta p=p_{\text{command}},\qquad \Psi(V)=-\Delta p\,V.
\]

用于压力扫描、标定和充气 ramp。它没有体积相关刚度，局部能量曲率为零。

#### B. 封闭等温理想气体（默认）

令参考绝对压力 \(p_0=p_{\text{amb}}+p_{0,\text{gauge}}\)，参考体积为 \(V_0\)：

\[
p_{\text{abs}}(V)=p_0\frac{V_0}{V},
\qquad
\Delta p=p_{\text{abs}}-p_{\text{amb}}.
\]

对应势能：

\[
\Psi(V)=-p_0V_0\ln V+p_{\text{amb}}V.
\]

#### C. 封闭绝热理想气体

\[
p_{\text{abs}}(V)=p_0\left(\frac{V_0}{V}\right)^\gamma,
\qquad \gamma\approx1.4\ \text{（空气）}.
\]

对应势能：

\[
\Psi(V)=\frac{p_0V_0^\gamma}{\gamma-1}V^{1-\gamma}+p_{\text{amb}}V.
\]

它比等温模型压缩更硬，适用于快速过程。

#### D. 目标体积 penalty/XPBD

Penalty 形式：

\[
\Psi(V)=\frac{k_V}{2}(V-V_t)^2.
\]

该模式适合视觉控制或与 PhysX/Vellum 对齐。若需要近似硬体积，再增加标量 XPBD/augmented-Lagrangian 投影；不建议一期把它混入默认理想气体模型。

### 4.3 压力力与局部 Hessian

对任意只依赖体积的气腔能量：

\[
f_i=-\frac{\partial \Psi}{\partial x_i}=\Delta p\,g_i.
\]

固定其他顶点时，闭合表面的体积对单个顶点是线性的，因此 VBD 的 3×3 对角块可以取：

\[
H_i^{\text{cavity}}=\Psi''(V)\,g_i g_i^T.
\]

各模式的曲率为：

| 模式 | \(\kappa=\Psi''(V)=-\mathrm d\Delta p/\mathrm dV\) |
|---|---:|
| Prescribed pressure | \(0\) |
| Isothermal | \(p_{\text{abs}}/V\) |
| Adiabatic | \(\gamma p_{\text{abs}}/V\) |
| Target-volume penalty | \(k_V\) |

在 VBD 中累积：

\[
f_i\mathrel{+}=\Delta p\,g_i,
\qquad
H_i\mathrel{+}=\kappa g_i g_i^T.
\]

这不是完整的全局一致切线。完整 Hessian 包含不同顶点之间的稠密 rank-one 耦合以及体积的交叉二阶项；VBD 保留正定的局部块，压力和法向在下一次外迭代刷新。该近似保持现有 3×3 顶点块求解和颜色并行，是性能与收敛的核心折中。

### 4.4 体积阻尼

可选的呼吸模态阻尼：

\[
\Delta p_{\text{eff}}
=\Delta p-c_V\dot V,
\qquad
\dot V\approx\frac{V^k-V^{n}}{\Delta t}.
\]

其中 \(c_V\) 的单位为 `Pa·s/m³`。局部曲率附加 \(c_V/\Delta t\)。默认设为零；只有看到明显的整体“喘振”时再开启。

### 4.5 数值保护

- 用 `min_volume_ratio * V0` 作为状态方程分母下界，但把触发次数暴露为诊断，不能静默长期 clamp。
- 对绝对压力做非负保护；最大压力只作为灾难保护，不用来代替正确时间步和材料参数。
- Builder 阶段拒绝非闭合、非流形和零体积 cavity。
- 初始表面方向由 cavity 独立的 `face_sign` 修正，不直接改写 cloth triangle winding。
- 压力从 0 平滑 ramp 到物理值，避免第一步冲击。
- 每个 VBD 外迭代更新压力；不要只在每帧或每个大步更新。

## 5. 推荐公共 API 与数据布局

### 5.1 用户侧 API

建议像 `newton.solvers.style3d` 一样暴露 `newton.solvers.vbd` helper namespace：

```python
from newton.solvers import SolverVBD, vbd

builder = newton.ModelBuilder()
vbd.register_pneumatic_attributes(builder)

handle = vbd.add_inflatable_mesh(
    builder,
    # 其余参数与 ModelBuilder.add_cloth_mesh() 一致
    pos=wp.vec3(0.0, 0.0, 1.0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0),
    vertices=vertices,
    indices=indices,
    density=0.12,
    tri_ke=2.0e5,
    tri_ka=2.0e5,
    tri_kd=2.0e2,
    edge_ke=2.0e-3,
    pneumatic=vbd.PneumaticCavityConfig(
        mode=vbd.PneumaticMode.ISOTHERMAL_IDEAL_GAS,
        ambient_pressure=101_325.0,
        initial_gauge_pressure=8_000.0,
        min_volume_ratio=0.05,
    ),
)

builder.color()
model = builder.finalize()

pipeline = newton.CollisionPipeline(
    model,
    # 第一步只有外力时可省略；运动压板或第二步夹爪抓取时启用。
    enable_rigid_soft_full_surface_contact=True,
)
contacts = pipeline.contacts()
solver = SolverVBD(
    model,
    iterations=20,
    particle_enable_self_contact=True,
    particle_self_contact_radius=thickness,
    particle_self_contact_margin=2.0 * thickness,
)
```

已有 cloth mesh 的注册方式：

```python
cavity = vbd.add_pneumatic_cavity(
    builder,
    triangle_indices=range(tri_start, tri_end),
    config=vbd.PneumaticCavityConfig(...),
)
```

### 5.2 公共类型

```python
class PneumaticMode(IntEnum):
    PRESCRIBED_PRESSURE = 0
    ISOTHERMAL_IDEAL_GAS = 1
    ADIABATIC_IDEAL_GAS = 2
    TARGET_VOLUME = 3


@dataclass(frozen=True)
class PneumaticCavityConfig:
    mode: PneumaticMode = PneumaticMode.ISOTHERMAL_IDEAL_GAS
    ambient_pressure: float = 101_325.0
    initial_gauge_pressure: float = 0.0
    gamma: float = 1.4
    target_volume_ratio: float = 1.0
    volume_stiffness: float = 0.0
    bulk_damping: float = 0.0
    min_volume_ratio: float = 0.05
    max_absolute_pressure: float = float("inf")


@dataclass(frozen=True)
class PneumaticCavityHandle:
    cavity_index: int
    face_start: int
    face_count: int
```

`add_inflatable_mesh()` 的生产签名应复制 v1.5.0 `ModelBuilder.add_cloth_mesh()` 的完整 keyword-only 参数，只额外增加 `pneumatic`；不要长期保留无类型的 `**kwargs` 公共 API。若需要压力渐入，由用户在若干步内平滑更新 `control.pneumatic.pressure_scale`，避免在 config 中隐式维护不可见的 solver 时钟。

### 5.3 Custom frequency schema

#### `pneumatic:cavity` 频率

| 属性 | Assignment | dtype | 单位/语义 |
|---|---|---|---|
| `pneumatic:cavity_mode` | MODEL | `wp.int32` | `PneumaticMode` |
| `pneumatic:cavity_world` | MODEL | `wp.int32` | `references="world"` |
| `pneumatic:cavity_rest_volume` | MODEL | `wp.float32` | m³ |
| `pneumatic:cavity_reference_abs_pressure` | MODEL | `wp.float32` | Pa |
| `pneumatic:cavity_ambient_pressure` | MODEL | `wp.float32` | Pa |
| `pneumatic:cavity_gamma` | MODEL | `wp.float32` | 无量纲 |
| `pneumatic:cavity_target_volume` | MODEL | `wp.float32` | m³ |
| `pneumatic:cavity_volume_stiffness` | MODEL | `wp.float32` | Pa/m³ |
| `pneumatic:cavity_bulk_damping` | MODEL | `wp.float32` | Pa·s/m³ |
| `pneumatic:cavity_min_volume` | MODEL | `wp.float32` | m³ |
| `pneumatic:cavity_max_abs_pressure` | MODEL | `wp.float32` | Pa |
| `pneumatic:cavity_label` | MODEL | `str` | 调试标签 |
| `pneumatic:volume` | STATE | `wp.float32` | 当前体积 m³ |
| `pneumatic:absolute_pressure` | STATE | `wp.float32` | 当前绝对压力 Pa |
| `pneumatic:volume_rate` | STATE | `wp.float32` | m³/s |
| `pneumatic:pressure_scale` | CONTROL | `wp.float32` | 0–1 ramp，默认 1 |
| `pneumatic:prescribed_gauge_pressure` | CONTROL | `wp.float32` | Pa，仅指定压力模式 |
| `pneumatic:target_volume_scale` | CONTROL | `wp.float32` | 目标体积倍率，默认 1 |

#### `pneumatic:face` 频率

| 属性 | Assignment | dtype | 引用 |
|---|---|---|---|
| `pneumatic:face_cavity` | MODEL | `wp.int32` | `references="pneumatic:cavity"` |
| `pneumatic:face_triangle` | MODEL | `wp.int32` | `references="triangle"` |
| `pneumatic:face_sign` | MODEL | `wp.int32` | `-1/+1` |

只需这两种自定义频率。CSR 邻接由 `SolverVBD` 构造时从上述行生成并保存在 solver 内部，不必把中间拓扑永久写入 `Model`。

### 5.4 Solver 内部邻接

`_build_pneumatic_adjacency()` 构造以下固定数组：

| 数组 | 长度 | 用途 |
|---|---:|---|
| `cavity_face_offsets` | `cavity_count + 1` | cavity → face membership CSR |
| `cavity_face_triangles` | `face_count` | membership → triangle |
| `cavity_face_signs` | `face_count` | membership 的方向 |
| `cavity_anchor_particles` | `cavity_count` | 平移稳定的体积参考点 |
| `particle_cavity_offsets` | `particle_count + 1` | particle → (particle,cavity) incidence group |
| `particle_cavity_ids` | `incidence_group_count` | incidence group → cavity |
| `incidence_face_offsets` | `incidence_group_count + 1` | group → 相邻 membership |
| `incidence_faces` | `3 * face_count` | 每个 cavity face 对三个顶点各出现一次 |

一个顶点可以属于多个 cavity，因此共享隔膜可自然得到两侧压差；力和 Hessian 在同一粒子线程内求和，不需要原子写。

## 6. 函数级落地设计

### 6.1 新文件：`newton/_src/solvers/vbd/pneumatic.py`

职责：公共类型、attribute 注册、builder helper、闭合性验证。

```python
def register_pneumatic_attributes(builder: ModelBuilder) -> None:
    """幂等注册 pneumatic:cavity / pneumatic:face 频率和全部属性。"""


def add_pneumatic_cavity(
    builder: ModelBuilder,
    *,
    triangle_indices: Sequence[int],
    config: PneumaticCavityConfig,
    face_signs: Sequence[int] | None = None,
    label: str | None = None,
) -> PneumaticCavityHandle:
    """验证并注册任意已有闭合三角表面。"""


def add_inflatable_mesh(
    builder: ModelBuilder,
    *,
    pneumatic: PneumaticCavityConfig,
    # 其余参数镜像 ModelBuilder.add_cloth_mesh
) -> PneumaticCavityHandle:
    """记录 particle/triangle 起点，调用 add_cloth_mesh，再注册新三角范围。"""


def _orient_cavity_faces(
    tri_indices: np.ndarray,
    cavity_triangles: np.ndarray,
) -> np.ndarray:
    """按共享边 BFS 求每个 membership 的 ±1 方向。"""


def _validate_closed_two_manifold(
    tri_indices: np.ndarray,
    cavity_triangles: np.ndarray,
    face_signs: np.ndarray,
) -> None:
    """每条无向边恰好出现两次且方向相反；检查重复面和越界索引。"""


def _compute_signed_rest_volume(
    particle_q: Sequence[Vec3],
    tri_indices: np.ndarray,
    cavity_triangles: np.ndarray,
    face_signs: np.ndarray,
) -> float:
    """用 anchor-centered tetrahedra 计算 V0；若整体为负，翻转所有 face_sign。"""
```

实现约束：

- `register_pneumatic_attributes()` 必须可以重复调用；遵循现有 `add_custom_frequency()` / `add_custom_attribute()` 幂等逻辑。
- `add_pneumatic_cavity()` 通过一次 `builder.add_custom_values()` 添加 cavity row，再为每个面添加 face row。
- `face_cavity` 和 `face_triangle` 必须正确声明 `references`，否则 `replicate()` 和 coupled compaction 会产生静默错索引。
- 不修改原始 `builder.tri_indices` winding；方向只存在于 `face_sign`。
- 默认拒绝由多个不连通闭合分量构成的一个 cavity；若未来需要，可增加 `allow_disconnected=True`，但压力相同是否合理应由用户明确决定。

### 6.2 新文件：`newton/_src/solvers/vbd/pneumatic_kernels.py`

职责：固定容量、graph-capturable 的 Warp kernels。

```python
@wp.kernel
def compute_cavity_volume_tile(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32),
    cavity_face_offsets: wp.array(dtype=wp.int32),
    cavity_face_triangles: wp.array(dtype=wp.int32),
    cavity_face_signs: wp.array(dtype=wp.int32),
    cavity_anchor_particles: wp.array(dtype=wp.int32),
    out_volume: wp.array(dtype=wp.float32),
):
    """一个 block 对应一个 cavity，线程跨步遍历 faces，块内固定顺序归约。"""


@wp.kernel
def evaluate_cavity_thermodynamics(
    dt: float,
    mode: wp.array(dtype=wp.int32),
    rest_volume: wp.array(dtype=wp.float32),
    reference_abs_pressure: wp.array(dtype=wp.float32),
    ambient_pressure: wp.array(dtype=wp.float32),
    gamma: wp.array(dtype=wp.float32),
    target_volume: wp.array(dtype=wp.float32),
    volume_stiffness: wp.array(dtype=wp.float32),
    bulk_damping: wp.array(dtype=wp.float32),
    min_volume: wp.array(dtype=wp.float32),
    max_abs_pressure: wp.array(dtype=wp.float32),
    previous_volume: wp.array(dtype=wp.float32),
    current_volume: wp.array(dtype=wp.float32),
    pressure_scale: wp.array(dtype=wp.float32),
    prescribed_gauge_pressure: wp.array(dtype=wp.float32),
    target_volume_scale: wp.array(dtype=wp.float32),
    out_absolute_pressure: wp.array(dtype=wp.float32),
    out_gauge_pressure: wp.array(dtype=wp.float32),
    out_energy_curvature: wp.array(dtype=wp.float32),
    out_volume_rate: wp.array(dtype=wp.float32),
    out_clamp_flags: wp.array(dtype=wp.int32),
):
    """计算 p、-dp/dV、dV/dt；压力 ramp 同时缩放 gauge pressure 和曲率。"""


@wp.kernel
def accumulate_cavity_force_and_hessian(
    particle_q: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32),
    particle_cavity_offsets: wp.array(dtype=wp.int32),
    particle_cavity_ids: wp.array(dtype=wp.int32),
    incidence_face_offsets: wp.array(dtype=wp.int32),
    incidence_faces: wp.array(dtype=wp.int32),
    cavity_face_triangles: wp.array(dtype=wp.int32),
    cavity_face_signs: wp.array(dtype=wp.int32),
    gauge_pressure: wp.array(dtype=wp.float32),
    energy_curvature: wp.array(dtype=wp.float32),
    particle_flags: wp.array(dtype=wp.int32),
    out_particle_force: wp.array(dtype=wp.vec3),
    out_particle_hessian: wp.array(dtype=wp.mat33),
):
    """每粒子线程遍历其 cavity groups；无原子地累加 p*g 和 kappa*g*g^T。"""


@wp.kernel
def reset_cavity_observables(
    world_mask: wp.array(dtype=wp.bool),
    cavity_world: wp.array(dtype=wp.int32),
    rest_volume: wp.array(dtype=wp.float32),
    reference_abs_pressure: wp.array(dtype=wp.float32),
    out_volume: wp.array(dtype=wp.float32),
    out_absolute_pressure: wp.array(dtype=wp.float32),
    out_volume_rate: wp.array(dtype=wp.float32),
):
    """按 world 恢复 cavity 可观测状态。"""
```

实现细节：

- `compute_cavity_volume_tile()` 不使用跨 block atomic add，保证每个 cavity 只有一个写者。
- block size 固定，例如 128；面数大于 block size 时循环处理，CUDA graph 形状不变。
- cavity 数为 0 时不创建无意义 launch。
- `accumulate_cavity_force_and_hessian()` 直接写当前 VBD iteration 已清零的内部 buffer；不要写 `state.particle_f`，避免与用户外力和显式初始化重复计算。
- 所有数组在 solver 构造阶段分配，step 中禁止 resize/CPU download。

### 6.3 修改：`newton/_src/solvers/vbd/solver_vbd.py`

#### 构造阶段

```python
def _init_pneumatic_system(self, model: Model) -> None:
    """检测 model.pneumatic 属性，验证 schema，构建邻接并分配 scratch。"""


def _build_pneumatic_adjacency(self, model: Model) -> PneumaticAdjacency:
    """在 CPU 上把 cavity/face rows 转成固定 CSR，再上传到 model.device。"""
```

在 `SolverVBD.__init__()` 中，紧接 `_init_particle_system(...)` 后调用：

```python
self._init_pneumatic_system(model)
```

自动检测规则：

- 没有 `pneumatic:cavity` count 时完全走原路径，性能和结果保持 bit-for-bit。
- 有 cavity 但缺少 face rows、状态数组或控制数组时，构造阶段抛出清晰错误。
- cavity 引用的每个三角形必须位于 solver 可见的 `ModelView` 中。

#### 每步阶段

新增：

```python
def _prepare_pneumatic_step(
    self,
    state_in: State,
    control: Control,
) -> None:
    """缓存上一时刻 volume；检查 control 数组和设备。"""
```

在 `step()` 中，control fallback 之后、VBD 外迭代之前调用一次：

```python
if control is None:
    control = self.model.control(clone_variables=False)

self._prepare_pneumatic_step(state_in, control)
self._initialize_rigid_bodies(...)
self._initialize_particles(...)
```

新增：

```python
def _evaluate_pneumatic_cavities(
    self,
    state_in: State,
    control: Control,
    dt: float,
) -> None:
    """从当前 state_in.particle_q 计算 solver scratch 中的体积、压力和局部曲率。"""


def _accumulate_pneumatic_force_and_hessian(
    self,
    state_in: State,
) -> None:
    """把 lagged cavity force/Hessian 写入已清零的 VBD 内部 buffers。"""


def _finalize_pneumatic_state(
    self,
    state_in: State,
    state_out: State,
    control: Control,
    dt: float,
) -> None:
    """从最终 particle_q 重算 cavity，并写入 state_out.pneumatic 可观测量。"""
```

修改 `_solve_particle_iteration()` 的准确插入位置：

```python
# 现有逻辑
self.particle_forces.zero_()
self.particle_hessians.zero_()

# 新逻辑：每个 VBD 外迭代刷新一次
self._evaluate_pneumatic_cavities(state_in, control, dt)
self._accumulate_pneumatic_force_and_hessian(state_in)

# 现有 color loop 不变
for color in range(len(self.model.particle_color_groups)):
    ...
```

因此 `_solve_particle_iteration()` 需要新增 `control` 参数；`step()` 中的调用改为：

```python
self._solve_particle_iteration(state_in, state_out, control, contacts, dt, iter_num)
```

选择“一次外迭代一个 pressure linearization”，而不是每个颜色组重新归约体积，原因是：

- 每个 iteration 只增加 3 个主要 kernel launch；
- 所有颜色使用同一压力线性化，结果可解释且容易确定性复现；
- 下一次 VBD iteration 会使用更新后的全部顶点重新线性化；
- 避免颜色数倍的全局归约成本。

最后一个颜色组更新完成后，iteration 内的 pressure linearization 会比最终位置早一次。为避免把滞后一轮的观测量交给下一帧，在 `step()` 末尾增加一次只读重算：

```python
self._finalize_rigid_bodies(state_in, state_out, dt)
self._finalize_particles(state_out, dt)
self._finalize_pneumatic_state(state_in, state_out, control, dt)
```

`_finalize_pneumatic_state()` 只重跑 volume reduction 和 thermodynamics，不再积累力/Hessian；它用 `_prepare_pneumatic_step()` 缓存的上一帧体积计算 `volume_rate`，并把最终 `volume`、`absolute_pressure`、`volume_rate` 写入 `state_out.pneumatic`。这样双缓冲 state 交换后，下一帧看到的是与最终 `particle_q` 一致的 cavity 状态。

#### Reset

在 `SolverVBD.reset()` 中增加：

```python
def _reset_pneumatic_state(
    self,
    state: State,
    world_mask: wp.array[wp.bool] | None,
) -> None:
    """恢复 volume/pressure/volume_rate，可在 capture 内执行。"""
```

当 `flags is None`，或 flags 包含 `PARTICLE_Q` / `PARTICLE_QD` 时重置 cavity observables；`flags=0` 时保留用户手工写入的 cavity state。

#### Determinism 与 module options

- 在 `_apply_module_options()` 覆盖的模块集合中加入 `pneumatic_kernels`。
- volume 使用单 block/cavity 固定归约，不依赖全局 atomic 的执行顺序。
- 多 cavity 对同一顶点的累加顺序由 CPU 构建邻接时按 cavity id 排序固定。

### 6.4 修改：VBD exports

#### `newton/_src/solvers/vbd/__init__.py`

导出：

```python
from .pneumatic import (
    PneumaticCavityConfig,
    PneumaticCavityHandle,
    PneumaticMode,
    add_inflatable_mesh,
    add_pneumatic_cavity,
    register_pneumatic_attributes,
)
```

#### `newton/_src/solvers/__init__.py`

仿照 `style3d` 增加 lazy module：

```python
__all__ = [..., "style3d", "vbd"]
_LAZY_IMPORTS["vbd"] = (".vbd", None)
```

用户采用：

```python
from newton.solvers import vbd
```

#### 文档 API

- 新增 `docs/api/newton_solvers_vbd.rst` 或在现有 solver API 中增加 helper namespace。
- 在 `docs/solvers/index.rst` 的 VBD capability 表中加入 pneumatic cavity。
- 运行 `python docs/generate_api.py` 更新公共符号文档。

### 6.5 不建议一期修改的文件

| 文件 | 为什么不改 |
|---|---|
| `newton/_src/sim/model.py` | custom frequency 已能表达 cavity/face |
| `newton/_src/sim/state.py` | STATE custom attributes 会由 `Model.state()` 自动创建 |
| `newton/_src/sim/control.py` | CONTROL custom attributes 会由 `Model.control()` 自动创建 |
| `newton/_src/sim/builder.py` | helper 使用现有公开 custom attribute API 即可 |
| `particle_vbd_kernels.py` 的 `solve_elasticity*` | 压力先写入现有内部 force/Hessian buffer，tile/non-tile 核心签名无需变化 |

只有 profiling 证明单独 pressure kernel launch 成为瓶颈时，二期才考虑把 cavity 邻接直接传入 `solve_elasticity()` / `solve_elasticity_tile()` 做 kernel fusion。

## 7. 两步实现路线

### 7.1 第一步：独立空腔袋受外力变形

目标是交付一个封闭空腔袋：它在重力、用户施加的 `state.particle_f`，以及可选的运动压板作用下，能够稳定地大变形、受压和回弹。该步骤不要求任何机器人模型。

实现分两小段，但它们共同构成第一步交付：

1. **P0：显式压力原型。** 在 example-local helper 中计算当前体积和压力，并把每面压力原子累加到 `state.particle_f`。它用于验证网格方向、压力量级、外力响应、压板接触和材料参数，不改 VBD 内部。
2. **P1：VBD 半隐式气腔。** 按第 5–6 节将压力在每个 VBD iteration 重新线性化，把 rank-one 3×3 局部 Hessian 与膜、弯曲和接触一起交给现有顶点块求解。这是第一步的正式实现；P0 不作为稳定性或性能结论。

P0 的调用时序为：

```python
state_in.clear_forces()
apply_external_forces(state_in, time)
apply_explicit_cavity_pressure(state_in, model, cavity, dt)
pipeline.collide(state_in, contacts)
solver.step(state_in, state_out, control, contacts, dt)
```

第一步验收：

- 对闭合袋施加局部外力或运动压板时，袋体能连续变形而无 NaN；
- 压板压缩和释放后，体积始终为正，压力随体积变化且袋体能够回弹；
- 开启自碰撞后，压扁或褶皱时不出现可见穿层；
- 正式 P1 模式在压缩测试中满足等温或绝热状态方程的预期误差。

### 7.2 第二步：机器人单向抓取

第二步复用第一步的气腔和袋体求解，只新增运动学机器人接触。机器人/夹爪的关节位姿和速度由外部轨迹、动画或控制器在每个子步先更新到当前刚体状态；碰撞管线使用该状态生成 rigid-soft contact，`SolverVBD(integrate_with_external_rigid_solver=True)` 只更新袋体。袋体对机器人产生的接触力**不**回写到机器人状态，也不调用 `SolverCoupledProxy/ADMM`。

每个子步的顺序为：

```python
advance_robot_kinematics(state_in, state_out, control, dt)
state_in.clear_forces()
pipeline.collide(state_in, contacts)
solver.step(state_in, state_out, control, contacts, dt)
```

第二步验收：

- 给定夹爪闭合、保持和张开的轨迹，袋体可被抓住、挤压和释放；
- full-surface rigid-soft contact 下夹爪与薄膜不发生可见穿透；
- 机器人轨迹相同的重复运行中，袋体结果在确定性模式容差内一致；
- 文档明确机器人为运动学输入，结果不能用于评估抓取力、执行器负载或机器人动力学稳定性。

## 8. 接触落地

### 8.1 袋体自碰撞

推荐构造：

```python
solver = SolverVBD(
    model,
    iterations=15-30,
    particle_enable_self_contact=True,
    particle_self_contact_radius=t_contact,
    particle_self_contact_margin=1.5 * t_contact,
    particle_collision_detection_interval=1,
    particle_enable_tile_solve=True,
)
```

参数是起点，不是固定答案。`particle_self_contact_radius` 表示求解器接触厚度，不必等于真实微米级薄膜厚度；实际应按最小可解析网格尺度和夹爪间隙标定。

### 8.2 第一步的外力和运动压板

第一步至少提供两种驱动方式：

- 对指定顶点/顶点集写入 `state.particle_f`，验证袋体在任意外载荷下的空腔响应；
- 使用按轨迹移动的平板刚体进行压缩，验证 pressure-volume-contact 三者共同工作。

纯外力测试可以不创建刚体接触管线。运动压板需要 `CollisionPipeline(enable_rigid_soft_full_surface_contact=True)`，且该压板应按运动学方式更新；与第二步一样，不求解或读取它从袋体获得的反力。

### 8.3 第二步的机器人单向接触

Newton 1.5 的 `CollisionPipeline(enable_rigid_soft_full_surface_contact=True)` 能生成边/面级 rigid-soft contact，standalone `SolverVBD` 可以消费。对 mesh/convex 刚体，需要按 1.5 文档提前配置可用 SDF，例如 `ShapeConfig.configure_sdf(force_sdf=True)`，并在 graph capture 前按顺序构造：

1. `CollisionPipeline`
2. `Contacts`
3. `SolverVBD`


机器人接入限制：

- 夹爪/机器人刚体必须是运动学驱动；每子步先由外部运动学更新机器人状态，然后以该当前刚体姿态调用 `CollisionPipeline.collide(state_in, contacts)`。`SolverVBD(integrate_with_external_rigid_solver=True)` 读取外部刚体状态来计算粒子接触，但不积分或改写刚体。
- 必须禁用或忽略袋体对机器人刚体的动力学积分和反馈；接触只影响粒子。
- robot mesh/convex 需要预配置 SDF；若某种机器人碰撞形状不能使用 full-surface contact，应在第二步明确拒绝或降级为可接受的粒子接触，而不是悄悄改变验收标准。
- 跨求解器 coupled view、`SolverCoupledProxy/ADMM`、机器人反力回传和抓取力控制不属于当前计划；若未来需要，应单独立项。

## 9. 测试与验收方案

### 9.1 单元测试：`newton/tests/test_vbd_pneumatic.py`

| 测试 | 方法 | 建议门槛 |
|---|---|---:|
| 四面体/立方体体积 | 与解析体积比较 | 相对误差 `< 1e-5` |
| 平移不变性 | 整体平移 `1e4 m` 后比较 | 相对误差 `< 1e-4` |
| 体积梯度 | 中心有限差分 | 相对误差 `< 1e-3` |
| 等温状态方程 | 多个 V/V0 点验证 `pV=const` | 相对误差 `< 1e-5` |
| 绝热状态方程 | 验证 `pV^gamma=const` | 相对误差 `< 1e-5` |
| Hessian 块 | 有限差分 pressure force | 相对误差 `< 5e-3` |
| 正定性 | `g g^T` 特征值 | 最小特征值 `>= -1e-6` |
| 闭合表面净力 | 累加全部 pressure forces | 相对残差 `< 1e-5` |
| 闭合表面净力矩 | 关于质心累加 | 相对残差 `< 1e-5` |
| winding 自动修正 | 随机翻面后 BFS orient | 得到正体积、边方向一致 |
| 非流形/开口网格 | 删除面或复制边 | 构造阶段必须报错 |
| 多腔共享隔膜 | 相反 `face_sign` | 合力等于两侧压差 |

### 9.2 求解回归测试

**第一步：独立空腔袋。**

1. **零表压静止**：无外力时不产生系统性膨胀。
2. **自由气球平衡**：指定膜参数和参考压力，形状收敛且无 NaN。
3. **局部外力变形**：对袋体一侧施加并撤除 `state.particle_f`，位移连续、体积为正，并在撤力后回弹。
4. **双平板压缩**：压缩到多个体积点，等温 `p_abs*V` 漂移小于 1%。
5. **等温 vs 绝热**：相同压缩量下绝热压力严格更高。
6. **压力 ramp**：第一步速度峰值相对无 ramp 显著降低。
7. **自碰撞褶皱**：压扁后不产生可见穿层，释放后可恢复。
8. **多 world**：不同 cavity 参数互不串扰。
9. **replicate**：triangle/cavity 引用全部正确偏移。
10. **CUDA graph**：capture/replay 中没有分配和 resize。
11. **deterministic**：相同设备、相同 mode 下逐位或容差内复现。
12. **tile/non-tile**：两条 VBD 路径结果在设定容差内一致。
13. **reset**：按 world mask 恢复粒子和 cavity observables。

**第二步：机器人单向抓取。**

1. **运动学姿态流**：外部运动学先更新机器人刚体状态，VBD 只更新粒子；袋体接触不会修改机器人由外部写入的刚体轨迹。
2. **夹爪闭合—保持—张开**：袋体被抓住、挤压并释放，位置、速度、体积和压力都保持有限。
3. **全表面接触**：夹爪边缘和面接触袋体时无可见穿层，soft-contact buffer 不溢出。
4. **无反力回传**：夹爪轨迹在有袋和无袋的运行中逐位一致；袋体接触不会改变机器人刚体状态。

### 9.3 示例

第一步示例 `newton/examples/cloth/example_cloth_inflatable.py` 包含：

- 一个密封袋；
- UI 切换 prescribed / isothermal / adiabatic；
- 局部外力开关和一个运动平板压缩袋体；
- 实时显示 `V/V0`、gauge pressure、clamp flags；
- `test_final()` 检查位置有限、体积为正、压力有限、没有 buffer overflow。

第二步新增 `newton/examples/vbd/example_vbd_inflatable_robot_grasp.py`：

- 复用第一步袋体场景；
- 一个由预设关节轨迹或刚体轨迹驱动的夹爪；
- 明确构造 `SolverVBD(..., integrate_with_external_rigid_solver=True)`；
- 显示夹爪轨迹、袋体体积和表压；
- `test_final()` 检查袋体稳定和夹爪终态等于给定运动学目标，不检查反力或机器人动力学。

### 9.4 性能基准

矩阵：

| 维度 | 取值 |
|---|---|
| 单袋三角形数 | 1k / 10k / 100k |
| world 数 | 1 / 64 / 1024 |
| cavity 数/world | 1 / 4 |
| VBD iterations | 5 / 10 / 20 |
| 自碰撞 | off / on |
| deterministic | off / on |
| tile solve | off / on |

记录：

- `compute_cavity_volume_tile` 时间；
- thermodynamics kernel 时间；
- pressure force/Hessian kernel 时间；
- 总 VBD step 增量；
- 显存增量；
- iteration 数增加前后的收敛残差。

一期性能目标不是先承诺固定百分比，而是设置验收闸门：1k–10k 三角面的单腔场景中，气腔三类 kernel 的总耗时应低于现有膜 + 弯曲求解耗时的 20%；若超过，进入 kernel fusion 优化。

## 10. 迭代计划与交付物

| 阶段 | 工期估计 | 交付物 | 退出条件 |
|---|---:|---|---|
| 第一步 A：显式原型 | 2–3 天 | example-local pressure kernels、局部外力和压板示例 | 可稳定受外力变形并完成平板压缩 |
| 第一步 B：气腔拓扑/API | 3–5 天 | custom frequencies、config/handle、闭合验证、replicate 测试 | 多 world/cavity 引用正确 |
| 第一步 C：VBD 气腔集成 | 5–8 天 | volume/thermo/force kernels、solver 调用、reset、graph | 外力和压板回归、等温/绝热回归通过 |
| 第二步：机器人单向抓取 | 4–7 天 | full-surface/self-contact 调参、运动学夹爪示例、无反力回传测试 | 抓取/压缩/释放无穿层和爆炸，机器人轨迹不受袋体影响 |
| 性能与文档 | 3–5 天 | tile reduction、benchmark、API 文档、两个示例 | 达到性能闸门 |

以上是一个熟悉 Newton/Warp 的工程师的净开发估计，不含真实材料标定、复杂资产修网格和大规模机器人任务调参。

## 11. 参数标定顺序

不要同时调所有参数。建议顺序：

1. 关闭压力和接触，只标定面内拉伸/剪切刚度；用单轴拉伸或悬垂测试。
2. 标定弯曲刚度，使褶皱尺度与真实薄膜接近。
3. 开启自碰撞，先在无压力压扁测试中消除穿层。
4. 使用 prescribed pressure 扫描压力-体积曲线，确认单位、winding 和接触厚度。
5. 切换 isothermal cavity，做平板压缩并验证 `p_abs V`。
6. 需要快速过程时再切 adiabatic，并标定 `gamma`/耗散。
7. 最后加入夹爪摩擦、运动学机器人轨迹和 pressure ramp；不在此路线中加入机器人双向反馈。

注意 Newton 1.5 已把 VBD 的 `kd` 语义改为绝对物理单位，不再是旧版的 stiffness-relative multiplier；迁移旧参数必须重新换算或重调。

## 12. 主要风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| 网格不闭合/方向错 | 负体积、吸瘪、压力爆炸 | Builder 阶段严格 two-manifold 验证和独立 face signs |
| 体积接近零 | 压力发散、NaN | min-volume 灾难保护 + clamp 诊断 + 更多 substeps/iterations |
| VBD 局部 Hessian忽略全局耦合 | 高压下收敛慢 | 每 iteration 刷新；必要时加 scalar AL/XPBD outer correction |
| 显式原型误判正式稳定性 | P0 需要很多 substeps | P0 只做物理/参数验证，不作为性能结论 |
| 自碰撞过滤过强 | 两层膜互穿或错误忽略接触 | 调低拓扑过滤范围；用 rest-shape exclusion 单独控制 |
| 接触厚度过大 | 袋体无法压薄 | 接触厚度按网格解析度标定，不照搬真实膜厚 |
| float32 体积消减 | 远离原点时压力抖动 | anchor-centered volume + block reduction |
| 各向同性膜不够真实 | 编织袋/吹塑膜方向响应错误 | 二期将 cavity 复用到 Style3D 或给 VBD 加各向异性膜 |
| 运动学机器人姿态与接触帧不一致 | 接触滞后一帧、夹爪穿透 | 每子步先完成外部运动学更新，再以更新后的 `state_in.body_q/body_qd` 调用 `collide` 和外部刚体 VBD 路径 |
| 将单向结果误用于机器人受力分析 | 错估抓取力或执行器负载 | 文档和示例明确不回传反力；抓取力控制或机器人动力学另行立项 |

## 13. 建议的技术决策

可以直接据此立项的决策如下：

- **主求解器**：`SolverVBD`。
- **主气体模型**：封闭等温理想气体；绝热作为同一接口的第二模式。
- **主耦合离散**：每个 VBD iteration 一次 global volume/pressure linearization，局部 rank-one Hessian。
- **数据接入**：Newton 1.5 custom frequency/custom attributes，不新增核心 `AttributeFrequency`。
- **公共 API**：`newton.solvers.vbd` helper namespace，风格对齐 `style3d`。
- **第一步接触**：VBD self-contact + 外力/运动压板；压板接触使用 standalone VBD full-surface rigid-soft。
- **第二步机器人接触**：`SolverVBD(integrate_with_external_rigid_solver=True)` + 运动学机器人刚体 + full-surface rigid-soft；不使用 coupled solver，也不回传袋体反力。
- **硬体积**：保留 target-volume penalty/XPBD 为可选模式，不替代理想气体。
- **范围边界**：只支持封闭、质量恒定的 cavity，不预留进出气或泄漏求解分支。

## 14. 参考资料

### Newton 与基础算法

- [Newton v1.5.0 release](https://github.com/newton-physics/newton/releases/tag/v1.5.0)
- [Newton SolverVBD source](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/solvers/vbd/solver_vbd.py)
- [Newton VBD particle kernels](https://github.com/newton-physics/newton/blob/v1.5.0/newton/_src/solvers/vbd/particle_vbd_kernels.py)
- [Vertex Block Descent, Chen et al. 2024](https://doi.org/10.1145/3658179)
- [Augmented Vertex Block Descent, Giles et al. 2025](https://doi.org/10.1145/3731195)
- [XPBD: Position-Based Simulation of Compliant Constrained Dynamics](https://matthias-research.github.io/pages/publications/XPBD.pdf)
- [C-IPC project](https://ipc-sim.github.io/C-IPC/)

### 工业/开源充气与气腔实现

- [PhysX 5.3.1 Particle System — Inflatables](https://nvidia-omniverse.github.io/PhysX/physx/5.3.1/docs/ParticleSystem.html)
- [PhysX ParticleInflatableDemo discussion/code](https://github.com/NVIDIA-Omniverse/PhysX/discussions/385)
- [Omniverse deformable migration note](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/deformable_migration.html)
- [Houdini Vellum Pressure](https://www.sidefx.com/docs/houdini/vellum/pressure.html)
- [SOFA SurfacePressureForceField](https://sofa-framework.github.io/doc/components/mechanicalload/surfacepressureforcefield/)
- [SOFA source repository](https://github.com/sofa-framework/sofa)
- [Abaqus Fluid Cavity](https://docs.software.vt.edu/abaqusv2025/English/?show=SIMACAEANLRefMap%2Fsimaanl-c-surfacebasedcavityover.htm)

---

最终建议：先交付“独立空腔袋受外力变形”：以 P0 验证 pressure-volume-contact 和外力/压板响应，再完成 custom-frequency + VBD 半隐式气腔集成。随后复用该袋体能力，接入按轨迹驱动的机器人夹爪，采用 VBD 已有的外部刚体单向路径完成抓取。正式实现始终保持“每个密闭腔一个压力状态”的范围，不引入气体空间离散、机器人反力回传或跨求解器双向耦合。
