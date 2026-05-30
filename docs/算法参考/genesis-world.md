# genesis-world 项目分析

**项目位置**: `C:\csy_work\CG\Engine\genesis-world`

---

## 1. 项目概述

Genesis 是一个通用物理仿真平台，支持多种物理模型（刚体、软体、流体、布料等），通过统一的场景描述和求解器架构提供端到端的仿真能力。IPC 接触处理通过集成 libuipc 实现。

### 核心特性
- **多物理耦合**: 刚体、软体 (FEM)、布料、流体 (MPM/SPH)、PBD
- **IPC 接触**: 通过 IPCCoupler 集成 libuipc 实现无穿透接触
- **ABD 刚体**: Affine Body Dynamics 刚体降维仿真
- **MuJoCo 兼容**: 支持 MJCF 场景描述
- **可微分仿真**: 支持 PyTorch 梯度计算
- **GPU 加速**: 全 GPU 流程

### 支持的求解器
- **RigidSolver**: 刚体 + ABD
- **FEMSolver**: 有限元软体/布料
- **MPMSolver**: 物质点法流体
- **SPHSolver**: 光滑粒子流体
- **PBDSolver**: 位置动力学
- **KinematicSolver**: 运动学物体

---

## 2. 项目架构

```
genesis/
├── engine/                     # 核心引擎
│   ├── scene.py               # 场景管理
│   ├── simulator.py           # 仿真器主循环
│   ├── mesh.py                # 网格数据结构
│   ├── bvh.py                 # BVH 碰撞检测
│   │
│   ├── solvers/               # 求解器
│   │   ├── rigid/             # 刚体求解器
│   │   │   ├── rigid_solver.py    # 主求解器
│   │   │   ├── abd/              # ABD 模块
│   │   │   │   ├── forward_kinematics.py
│   │   │   │   ├── forward_dynamics.py
│   │   │   │   ├── inverse_kinematics.py
│   │   │   │   └── accessor.py
│   │   │   ├── collider/         # 碰撞检测
│   │   │   └── constraint/      # 约束求解
│   │   ├── fem_solver.py         # FEM 求解器
│   │   ├── mpm_solver.py         # MPM 求解器
│   │   └── pbd_solver.py         # PBD 求解器
│   │
│   ├── couplers/              # 耦合器
│   │   ├── ipc_coupler/       # IPC 耦合器
│   │   │   ├── coupler.py     # 主耦合逻辑
│   │   │   ├── data.py        # 数据结构
│   │   │   └── utils.py       # 工具函数
│   │   ├── sap_coupler.py     # SAP 耦合器
│   │   └── legacy_coupler.py  # 旧版耦合器
│   │
│   ├── materials/             # 材料定义
│   │   ├── FEM/               # FEM 材料
│   │   │   ├── cloth.py       # 布料
│   │   │   ├── elastic.py     # 弹性体
│   │   │   └── muscle.py      # 肌肉
│   │   ├── rigid.py           # 刚体材料
│   │   ├── MPM/               # MPM 材料
│   │   └── PBD/               # PBD 材料
│   │
│   ├── entities/              # 实体定义
│   │   ├── rigid_entity/      # 刚体实体
│   │   ├── fem_entity.py      # FEM 实体
│   │   └── mpm_entity.py      # MPM 实体
│   │
│   └── states/                # 状态管理
│
├── options/                   # 配置选项
│   └── solvers.py             # 求解器配置
│
├── vis/                       # 可视化
└── utils/                     # 工具函数
```

---

## 3. IPCCoupler (IPC 耦合器)

### 3.1 耦合模式

```python
class COUPLING_TYPE(IntEnum):
    TWO_WAY_SOFT_CONSTRAINT = 0  # 双向软约束耦合
    EXTERNAL_ARTICULATION = 1    # 外部关节约束
    IPC_ONLY = 2                 # 仅 IPC 仿真
    NONE = 3                     # 无耦合
```

### 3.2 IPCCoupler 核心结构

```python
class IPCCoupler:
    def __init__(self, simulator, options: IPCCouplerOptions):
        # IPC 系统基础设施
        self._ipc_engine: Engine | None
        self._ipc_world: World | None
        self._ipc_scene: Scene

        # IPC 材料模型
        self._ipc_abd: AffineBodyConstitution      # ABD 刚体
        self._ipc_stk: StableNeoHookean            # SNH 弹性
        self._ipc_stc: SoftTransformConstraint     # 软变换约束
        self._ipc_nks: StrainLimitingBaraffWitkinShell  # 应变限制
        self._ipc_dsb: DiscreteShellBending        # 离散弯曲

        # 接触元素
        self._ipc_fem_contacts: dict[FEMEntity, ContactElement]
        self._ipc_cloth_contacts: dict[FEMEntity, ContactElement]
        self._ipc_abd_contacts: dict[RigidEntity, ContactElement]

    def build(self):
        """构建 IPC 系统"""
        self._init_ipc()
        self._setup_coupling_config()
        self._add_objects_to_ipc()
        self._finalize_ipc()
        self._init_accessors()
```

### 3.3 配置选项

```python
@dataclass
class IPCCouplerOptions:
    # 约束强度
    constraint_strength_translation: float = 100.0
    constraint_strength_rotation: float = 100.0

    # IPC 参数
    d_hat: float = 0.001          # 激活距离
    kappa: float = 1e4            # Barrier 刚度
    contact_resistance: float = 1e7

    # 求解器参数
    newton_tolerance: float = 1e-1
    newton_translation_tolerance: float = 1.0
    linear_system_tolerance: float = 1e-3
    n_linesearch_iterations: int = 8

    # CCD 参数
    ccd_method: str = "libuipc"  # 或 "hybrid"
```

---

## 4. 刚体求解器 (RigidSolver)

### 4.1 ABD (Affine Body Dynamics)

```
刚体状态表示:
- 12 DOF per body: 3 rotation axes (a1, a2, a3) + 3 translation (p)
- 顶点位置: x_i = [dot(a1, r), dot(a2, r), dot(a3, r)] + p

正交约束:
- a_i · a_j = δ_ij (保持刚体性质)

能量函数:
- E_ortho = κ * V * Σ (a_i · a_j - δ_ij)²
```

### 4.2 核心模块

```python
# 前向运动学 (forward_kinematics.py)
func_forward_kinematics_entity()  # 正向运动学
func_COM_links()                  # 质心计算
func_update_geoms()               # 更新几何体位置

# 前向动力学 (forward_dynamics.py)
func_forward_dynamics()           # 正向动力学
func_compute_mass_matrix()       # 质量矩阵
func_solve_mass()                # 求解质量矩阵

# 逆运动学 (inverse_kinematics.py)
# IK 求解用于遥操作控制
```

### 4.3 碰撞检测

```python
# collider/
# 刚体碰撞检测与响应
```

---

## 5. FEM 求解器

### 5.1 支持的模型

```python
# 材料模型
- stable_neohookean  # 稳定 Neo-Hookean
- linear             # 线性弹性
- linear_corotated   # 共旋线性
```

### 5.2 布料材料

```python
@dataclass
class Cloth(Base):
    E: float = 1e4               # 杨氏模量 (Pa)
    nu: float = 0.49             # 泊松比
    rho: float = 200.0           # 密度 (kg/m³)
    thickness: float = 0.001     # 厚度 (m)
    bending_stiffness: float | None = None  # 弯曲刚度
    friction_mu: float = 0.1     # 摩擦系数
```

---

## 6. 场景构建

### 6.1 Scene 类

```python
class Scene:
    def __init__(self, sim_options, coupler_options, viewer_options):
        self.rigid_solver = RigidSolver()
        self.fem_solver = FEMSolver()
        self.mpm_solver = MPMSolver()
        self.ipc_coupler = IPCCoupler()

    def add_entity(self, morph, material, surface):
        """添加实体到场景"""

    def build(self):
        """构建场景"""

    def step(self):
        """执行一个仿真步"""
```

### 6.2 实体类型

```python
# 刚体实体
entity = scene.add_entity(
    morph=gs.morphs.MJCF(file="robot.xml"),
    material=gs.materials.Rigid(),
)

# FEM 软体
entity = scene.add_entity(
    morph=gs.morphs.Mesh(file="soft.obj"),
    material=gs.materials.FEM.Elastic(E=1e4, nu=0.4),
)

# 布料
entity = scene.add_entity(
    morph=gs.morphs.Mesh(file="cloth.obj"),
    material=gs.materials.FEM.Cloth(
        E=1e4, nu=0.49, thickness=0.001, bending_stiffness=10.0
    ),
)
```

---

## 7. 仿真主循环

### 7.1 Simulator

```python
class Simulator:
    def step(self):
        """主仿真循环"""
        # 1. 刚体预测
        self.rigid_solver.step()

        # 2. FEM 预测
        self.fem_solver.step()

        # 3. IPC 耦合求解
        if self.ipc_coupler:
            self.ipc_coupler.advance()

        # 4. 更新状态
        self._update_states()
```

### 7.2 IPCCoupler.advance()

```python
def advance(self):
    """IPC 推进一步"""
    # 1. 从 Genesis 提取当前状态
    self._extract_rigid_state()
    self._extract_fem_state()

    # 2. 调用 libuipc 求解
    self._ipc_scene.advance()

    # 3. 写回耦合力
    self._apply_coupling_forces()

    # 4. 更新 ABD 变换
    self._update_abd_transforms()
```

---

## 8. 与 libuipc 的集成

### 8.1 核心依赖

```python
from uipc.core import Engine, World, Scene
from uipc.constitution import (
    AffineBodyConstitution,       # ABD 刚体
    StableNeoHookean,             # SNH 弹性
    SoftTransformConstraint,      # 软变换约束
    StrainLimitingBaraffWitkinShell,  # 应变限制
    DiscreteShellBending,         # 离散弯曲
)
from uipc.geometry import SimplicialComplex
```

### 8.2 数据传递

```python
# Genesis → IPC
- 顶点位置
- 速度
- 材料参数

# IPC → Genesis
- ABD 变换矩阵
- 耦合力
- 接触信息
```

---

## 9. 坐标系与变换

### 9.1 变换表示

```python
# 4x4 齐次变换矩阵
T = [
    [R11, R12, R13, tx],
    [R21, R22, R23, ty],
    [R31, R32, R33, tz],
    [0,   0,   0,   1 ],
]

# ABD link 的变换存储
self._abd_transforms_by_link: dict[RigidLink, list[np.ndarray]]
```

### 9.2 耦合力计算

```python
def update_coupling_forces(coupling_data, abd_state_feature):
    """
    从 IPC 提取耦合力并应用到 Genesis 刚体

    力的计算:
    - 基于软约束能量 E = k * ||T_aim - T_ipc||²
    - 力 = -∂E/∂x
    """
```

---

## 10. 材料参数

### 10.1 刚体材料

```python
@dataclass
class RigidMaterial:
    rho: float = 1000.0          # 密度
    friction: float = 0.5         # 摩擦系数

    # IPC 耦合参数
    coup_type: str = "two_way_soft_constraint"
    coup_links: tuple = ()        # 参与耦合的 links
    coup_friction: float = 0.5    # 耦合摩擦
```

### 10.2 FEM 材料

```python
@dataclass
class ElasticMaterial:
    E: float = 1e4               # 杨氏模量
    nu: float = 0.4              # 泊松比
    rho: float = 1000.0          # 密度
    model: str = "stable_neohookean"
```

---

## 11. 示例代码

### 11.1 基础场景

```python
import genesis as gs

gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.02),
    coupler_options=gs.options.IPCOptions(
        constraint_strength_translation=100.0,
        d_hat=0.001,
    ),
)

# 添加地面
scene.add_entity(gs.morphs.Plane())

# 添加刚体
franka = scene.add_entity(
    gs.morphs.MJCF(file="panda.xml"),
    material=gs.materials.Rigid(coup_type="two_way_soft_constraint"),
)

# 添加布料
cloth = scene.add_entity(
    gs.morphs.Mesh(file="cloth.obj"),
    material=gs.materials.FEM.Cloth(
        E=6e4, nu=0.49, thickness=0.001, bending_stiffness=10.0
    ),
)

scene.build()

for _ in range(100):
    scene.step()
```

### 11.2 遥操作示例

```python
# 参考 examples/IPC_Solver/ipc_robot_cloth_teleop.py

# IK 求解
ee_link = franka.get_link("hand")
qpos = franka.inverse_kinematics(
    link=ee_link,
    pos=target_pos,
    quat=target_quat,
)

# 控制关节
franka.control_dofs_position(qpos)

# 夹爪控制
if gripper_close:
    franka.control_dofs_position(0.0, dofs_idx_local=finger_dofs)
```

---

## 12. 与 Newton 的对比

| 特性 | Genesis | Newton |
|------|---------|--------|
| IPC 实现 | libuipc 集成 | 自实现 (Warp) |
| 刚体求解 | MuJoCo / ABD | MuJoCo |
| 布料模型 | Shell FEM | VBD |
| 可微分 | PyTorch | Warp |
| 耦合方式 | IPCCoupler | SolverCoupledProxy |

---

## 13. 参考

- GitHub: https://github.com/Genesis-Intelligence/genesis-world
- libuipc: https://github.com/spiriMirror/libuipc
- MuJoCo: https://mujoco.org/