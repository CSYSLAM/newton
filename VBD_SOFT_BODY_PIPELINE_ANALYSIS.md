# Newton VBD (Vertex Block Descent) 软体仿真管线 — 完整技术分析

> 本文档对 Newton 物理引擎中基于 VBD/AVBD 的软体仿真管线进行深度技术分析，涵盖算法原理、实现细节、代码示例和设计决策理由。

---

## 目录

1. [概述与参考文献](#1-概述与参考文献)
2. [软体创建流程](#2-软体创建流程)
3. [数据架构与数据结构](#3-数据架构与数据结构)
4. [完整仿真算法流程](#4-完整仿真算法流程)
5. [接触算法详解](#5-接触算法详解)
6. [核心算法公式与推导](#6-核心算法公式与推导)
7. [数据链路与内存模型](#7-数据链路与内存模型)
8. [架构设计与设计决策](#8-架构设计与设计决策)
9. [关键代码路径索引](#9-关键代码路径索引)
10. [优缺点分析](#10-优缺点分析)
11. [附录：配置参数与调优指南](#11-附录配置参数与调优指南)

---

## 1. 概述与参考文献

### 1.1 算法概述

Newton 使用 **VBD (Vertex Block Descent)** 算法进行软体仿真（包括布料和软组织），使用 **AVBD (Augmented VBD)** 算法进行刚体仿真。两者通过统一的异构约束系统在同一求解器中协同工作。

**VBD 的核心思想**：将全局非线性隐式欧拉优化问题分解为每个顶点独立的 3×3 局部二次子问题，通过 Gauss-Seidel 颜色分组的并行迭代来求解。每个局部子问题可以解析求解（3×3 矩阵求逆），避免了全局稀疏线性系统的组装和分解。

**AVBD 的核心思想**：将 VBD 的 per-vertex block descent 思想推广到刚体：每个刚体（6 DOF）视为一个"块"，在 6×6 的 SPD 线性系统上做 block descent，使用 penalty + augmented Lagrangian 双变量更新来驱动约束满足。

**为什么叫 "Vertex Block Descent"？**
- **Vertex Block**：每个顶点是一个独立的"块"，只求解该顶点的 3×3 局部系统
- **Descent**：每次迭代沿着局部二次近似的负梯度方向下降（等价于 Newton 步）
- 不同于传统的全局 Newton 法（需要组装和分解 N×N 稀疏矩阵），VBD 利用 Gauss-Seidel 思想，将 off-diagonal 耦合"滞后"处理

### 1.2 参考文献

- **VBD**: Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. *Vertex Block Descent*. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages. https://doi.org/10.1145/3658179
- **AVBD**: Chris Giles, Elie Diaz, and Cem Yuksel. 2025. *Augmented Vertex Block Descent*. ACM Trans. Graph. 44, 4, Article 90 (August 2025), 12 pages. https://doi.org/10.1145/3731195
- **Stable Neo-Hookean**: Breannan Smith, Fernando De Goes, and Theodore Kim. 2018. *Stable Neo-Hookean Flesh Simulation*. ACM Trans. Graph. 37, 2, Article 12 (March 2018), 15 pages.
- **Discrete Shell Bending**: Eitan Grinspun, Anil N. Hirani, Mathieu Desbrun, and Peter Schröder. 2003. *Discrete Shells*. In Proc. ACM SIGGRAPH/Eurographics Symp. Comput. Anim. (SCA '03).

---

## 2. 软体创建流程

### 2.1 方式一：程序化网格 `add_soft_grid()`

#### 2.1.1 API 调用示例

```python
import newton as nw

builder = nw.ModelBuilder()

# 创建一个 10×20×2 的六面体网格，每格 0.1m
builder.add_soft_grid(
    pos=(0.0, 0.0, 0.0),          # 世界空间位置
    rot=(0.0, 0.0, 0.0, 1.0),     # 世界空间旋转 (四元数)
    vel=(0.0, 0.0, 0.0),          # 初始速度
    dim_x=10, dim_y=20, dim_z=2,  # 各轴网格数
    cell_x=0.1, cell_y=0.1, cell_z=0.1,  # 单元格尺寸 [m]
    density=1000.0,                # 密度 [kg/m³]
    k_mu=1.0e5,                    # Lamé 第一参数 μ [Pa]
    k_lambda=1.0e5,                # Lamé 第二参数 λ [Pa]
    k_damp=0.1,                    # Rayleigh 阻尼系数
    fix_left=True,                 # 固定左边界
    add_surface_mesh_edges=True,   # 生成表面弯曲边（用于碰撞）
    edge_ke=0.0, edge_kd=0.0,     # 弯曲边刚度/阻尼（零=仅用于碰撞检测）
    particle_radius=0.05,          # 粒子碰撞半径 [m]
)

builder.color()     # 必须：Gauss-Seidel 着色
model = builder.finalize()
```

#### 2.1.2 内部实现详解

**步骤 1：创建网格顶点** (`builder.py:8337-8361`)

```python
# 伪代码
start_vertex = len(self.particle_q)
mass = cell_x * cell_y * cell_z * density  # 每个顶点质量

for z in range(dim_z + 1):
    for y in range(dim_y + 1):
        for x in range(dim_x + 1):
            v_local = (x * cell_x, y * cell_y, z * cell_z)
            v_world = quat_rotate(rot, v_local) + pos
            m = mass
            if fix_left and x == 0: m = 0.0    # 运动学粒子
            # ... 其他固定边界类似
            self.add_particle(v_world, vel, m, particle_radius)
```

**为什么每个顶点的质量是 `cell_x * cell_y * cell_z * density`？**
- 每个顶点代表其周围 Voronoi 区域的"块"质量
- 对于规则网格，每个顶点的控制体积近似等于一个 cell 的体积
- 使用 `mass=0` 来标记运动学粒子（`ParticleFlags.KINEMATIC`），求解器会自动跳过这些粒子

**步骤 2：六面体 → 四面体剖分** (`builder.py:8385-8409`)

每个六面体单元分解为 **5 个四面体**，使用交替剖分方案避免退化：

```python
# 每个 hex cell 有 8 个顶点 v0-v7
# 使用 parity = (x ^ y ^ z) & 1 交替选择剖分方案

if (x & 1) ^ (y & 1) ^ (z & 1):
    # 方案 A
    add_tet(v0, v1, v4, v3)
    add_tet(v2, v3, v6, v1)
    add_tet(v5, v4, v1, v6)
    add_tet(v7, v6, v3, v4)
    add_tet(v4, v1, v6, v3)   # 中心 tet
else:
    # 方案 B（镜像）
    add_tet(v1, v2, v5, v0)
    add_tet(v3, v0, v7, v2)
    add_tet(v4, v7, v0, v5)
    add_tet(v6, v5, v2, v7)
    add_tet(v5, v2, v7, v0)   # 中心 tet
```

**为什么交替使用两种剖分方案？**
- 单一剖分方案会在相邻 hex 的共享面上产生不匹配的对角线
- 交替方案确保相邻 hex 的共享面剖分一致，避免产生非流形边
- 中心 tet 连接四个体对角顶点，保证所有 tet 的体积为正

**步骤 3：提取表面三角形** (`builder.py:8364-8414`)

使用开放面检测（open face detection）：
```python
faces = {}  # dict: sorted_tuple(i,j,k) -> (i,j,k)

def add_face(i, j, k):
    key = tuple(sorted((i, j, k)))
    if key not in faces:
        faces[key] = (i, j, k)   # 第一次出现：添加
    else:
        del faces[key]            # 第二次出现：内部面，删除

# 每个 tet 的 4 个面调用 add_face()
# 最终 faces 中只保留表面三角形（只出现一次的面）
```

**为什么用 dict 而不是直接检测？**
- 内部面会被两个相邻 tet 各添加一次，所以出现两次
- 表面面只被一个 tet 添加，只出现一次
- O(N) 时间复杂度，比 O(N²) 的几何检测快得多

**步骤 4：生成表面弯曲边** (`builder.py:8417-8428`)

```python
if add_surface_mesh_edges:
    adj = MeshAdjacency(self.tri_indices[start_tri:end_tri])
    # 对每条边，找到其两个相邻三角形和两个对角顶点
    edge_indices = [(o0, o1, v0, v1) for each edge]
    for o1, o2, v1, v2 in edge_indices:
        self.add_edge(o1, o2, v1, v2, rest=None, edge_ke, edge_kd)
```

**为什么表面需要弯曲边？**
- 自碰撞检测需要表面三角形 + 边来检测 vertex-triangle 和 edge-edge 碰撞
- 即使 `edge_ke=0`（无弯曲刚度），边的存在也让碰撞检测器知道哪些边需要检测
- 如果 `edge_ke > 0`，这些边还会提供弯曲刚度（类似 cloth 的 bending）

### 2.2 方式二：导入四面体网格 `add_soft_mesh()`

#### 2.2.1 API 调用示例

```python
# 从 USD 文件加载
mesh = nw.TetMesh.create_from_usd("rubber_duck.usd")

builder.add_soft_mesh(
    pos=(0.0, 1.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    scale=1.0,
    vel=(0.0, 0.0, 0.0),
    mesh=mesh,
    density=1500.0,           # 覆盖 mesh.density
    k_mu=5.0e4,               # 覆盖 mesh.k_mu
    k_lambda=5.0e4,           # 覆盖 mesh.k_lambda
    k_damp=0.1,               # 覆盖 mesh.k_damp
    validate_mesh=True,       # 验证网格质量
    label="rubber_duck",
)
```

#### 2.2.2 TetMesh 数据结构

```python
# geometry/types.py:1559
class TetMesh:
    vertices: np.ndarray          # [N, 3] 顶点位置
    tet_indices: np.ndarray       # [M*4] 四面体索引 (flatten)
    density: float | None         # 密度 [kg/m³]
    k_mu: float | np.ndarray | None     # per-element 或标量
    k_lambda: float | np.ndarray | None
    k_damp: float | np.ndarray | None
    surface_triangles: np.ndarray | None  # 预计算的表面三角形

    @classmethod
    def create_from_usd(cls, path: str) -> TetMesh:
        """从 USD 文件加载四面体网格"""
```

**为什么支持 per-element 材料参数？**
- 不同区域可能需要不同的刚度（如生物组织的硬骨 vs 软骨）
- `k_mu` 可以是标量（广播到所有 tet）或数组（每个 tet 独立）
- 在 `add_soft_mesh` 中通过 `np.broadcast_to` 统一处理

#### 2.2.3 参数解析优先级

```
显式参数 > TetMesh 属性 > ModelBuilder 默认值
```

```python
# builder.py:8513-8549
if density is None:
    density = mesh.density           # TetMesh 属性
if density is None:
    density = self.default_tet_density  # Builder 默认值
```

### 2.3 方式三：布料创建 `add_cloth_grid()` / `add_cloth_mesh()`

布料使用三角形膜元素（无四面体体积约束），但共享相同的 VBD 求解器：

```python
builder.add_cloth_grid(
    pos=(0, 2, 0), rot=(0,0,0,1), vel=(0,0,0),
    dim_x=20, dim_z=20,
    cell_x=0.05, cell_z=0.05,
    density=0.5,                    # 面密度 [kg/m²]
    tri_ke=1.0e4,                   # 膜刚度
    tri_ka=5.0e4,                   # 面积刚度
    tri_kd=0.1,                     # 膜阻尼
    edge_ke=1.0e-3,                 # 弯曲刚度
    edge_kd=1.0e-2,                 # 弯曲阻尼
)
```

**布料 vs 软体的关键区别**：
- 布料只有三角形膜 + 弯曲边，没有四面体
- 布料粒子的质量计算不同（面密度 × 面积）
- 布料通常需要弯曲刚度来抵抗褶皱

### 2.4 `add_tetrahedron()` 的静息位姿计算

```python
# builder.py:7677-7709
def add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp):
    p, q, r, s = particle_q[i], particle_q[j], particle_q[k], particle_q[l]

    # 计算静息边矩阵 Dm = [q-p, r-p, s-p]
    Dm = np.array((q - p, r - p, s - p)).T  # 3×3
    volume = np.linalg.det(Dm) / 6.0

    if volume <= 0.0:
        print("inverted tetrahedral element")
    else:
        inv_Dm = np.linalg.inv(Dm)  # 存储 Dm⁻¹
        self.tet_indices.append((i, j, k, l))
        self.tet_poses.append(inv_Dm.tolist())  # 存储为 3×3 矩阵
        self.tet_materials.append((k_mu, k_lambda, k_damp))
```

**为什么存储 `Dm⁻¹` 而不是 `Dm`？**
- 变形梯度 `F = Ds · Dm⁻¹`，每次计算 F 都需要 `Dm⁻¹`
- 运行时只需一次矩阵乘法，避免每帧求逆
- `rest_volume = 1 / (det(Dm⁻¹) * 6)` 也从 `Dm⁻¹` 计算

### 2.5 约束类型总览

| 约束类型 | 能量模型 | 创建 API | 顶点数 | 每元素存储 |
|---------|---------|---------|--------|-----------|
| 四面体体积 | Stable Neo-Hookean 3D | `add_tetrahedron()` | 4 | Dm⁻¹ (3×3), k_mu, k_lambda, k_damp |
| 三角形膜 | Stable Neo-Hookean 2D | `add_triangle()` | 3 | Dm⁻¹ (2×2), area, ke, ka, kd, drag, lift |
| 弯曲边 | Discrete Shell | `add_edge()` | 4 | rest_angle, rest_length, ke, kd |
| 线性弹簧 | Linear spring | `add_spring()` | 2 | rest_length, ke, kd |

### 2.6 Gauss-Seidel 着色 (Coloring)

着色是 VBD 求解器的**强制性**前置步骤：

```python
builder.color()
# 等价于:
builder.color(include_bending=True, balance_colors=True)
```

**着色算法原理**：
- 将顶点分组，保证同一颜色组内的顶点**不共享任何约束元素**（tets, triangles, edges, springs）
- 这等价于图着色问题：顶点是图的节点，共享约束元素的顶点之间有边
- 使用贪心着色算法

**为什么着色是必需的？**
- Gauss-Seidel 要求：更新顶点 i 时，其所有邻居顶点必须已经更新（使用最新值）
- 同一颜色组内的顶点没有共享约束 → 可以完全并行更新，无需原子操作
- 不同颜色组之间串行执行，保证 Gauss-Seidel 的正确性

**着色数据存储**：
```python
model.particle_color_groups  # list[wp.array[int32]]，每组是一个颜色
model.particle_colors        # wp.array[int]，每个顶点的颜色编号
```

---

## 3. 数据架构与数据结构

### 3.1 核心数据模型 (Model)

```
Model (model.py)
├── 粒子数据 (per-particle)
│   ├── particle_q:           wp.array[wp.vec3]   # 当前位置 [m]
│   ├── particle_qd:          wp.array[wp.vec3]   # 当前速度 [m/s]
│   ├── particle_mass:        wp.array[float]     # 质量 [kg]
│   ├── particle_inv_mass:    wp.array[float]     # 逆质量 [1/kg]
│   ├── particle_radius:      wp.array[float]     # 碰撞半径 [m]
│   ├── particle_flags:       wp.array[int32]     # ACTIVE | KINEMATIC
│   ├── particle_colors:      wp.array[int]       # 颜色编号
│   └── particle_color_groups: list[wp.array]     # 颜色分组
│
├── 弹簧数据 (per-spring, 2 顶点)
│   ├── spring_indices:       wp.array[int]       # 展平顶点对 [v0,v1, v0,v1, ...]
│   ├── spring_rest_length:   wp.array[float]     # 静息长度 [m]
│   ├── spring_stiffness:     wp.array[float]     # ke
│   └── spring_damping:       wp.array[float]     # kd
│
├── 三角形 FEM 数据 (per-triangle, 3 顶点)
│   ├── tri_indices:          wp.array2d[int32]   # [M_tri, 3]
│   ├── tri_poses:            wp.array[wp.mat22]  # Dm⁻¹ (2×2)
│   ├── tri_areas:            wp.array[float]     # 静息面积 [m²]
│   └── tri_materials:        wp.array2d[float]   # [M_tri, 5]: ke,ka,kd,drag,lift
│
├── 弯曲边数据 (per-edge, 4 顶点)
│   ├── edge_indices:         wp.array2d[int32]   # [M_edge, 4]: [opp0, opp1, v0, v1]
│   ├── edge_rest_angle:      wp.array[float]     # 静息二面角 [rad]
│   ├── edge_rest_length:     wp.array[float]     # 静息边长 [m]
│   └── edge_bending_properties: wp.array2d[float] # [M_edge, 2]: ke, kd
│
├── 四面体 FEM 数据 (per-tet, 4 顶点)
│   ├── tet_indices:          wp.array2d[int32]   # [M_tet, 4]
│   ├── tet_poses:            wp.array[wp.mat33]  # Dm⁻¹ (3×3)
│   └── tet_materials:        wp.array2d[float]   # [M_tet, 3]: k_mu, k_lambda, k_damp
│
└── 接触材料 (全局)
    ├── soft_contact_ke:      float               # 自接触刚度
    ├── soft_contact_kd:      float               # 自接触阻尼
    └── soft_contact_mu:      float               # 自接触摩擦系数
```

### 3.2 VBD 求解器运行时数据 (SolverVBD)

```
SolverVBD (solver_vbd.py)
├── 粒子求解状态
│   ├── particle_q_prev:      wp.array[wp.vec3]   # 上一步位置（用于速度计算）
│   ├── inertia:              wp.array[wp.vec3]   # 惯性目标位置 x★
│   ├── particle_forces:      wp.array[wp.vec3]   # per-vertex 力累加缓冲区
│   ├── particle_hessians:    wp.array[wp.mat33]  # per-vertex Hessian 累加缓冲区
│   ├── particle_displacements: wp.array[wp.vec3] # 当前位移 Δx
│   ├── truncation_ts:        wp.array[float]     # Planar DAT 截断系数 t ∈ [0,1]
│   └── particle_conservative_bounds: wp.array[float]  # 保守位移边界
│
├── 邻接信息 (CSR 格式)
│   └── particle_adjacency: ParticleForceElementAdjacencyInfo
│       ├── v_adj_tets:       wp.array[int]       # 展平: [tet_id, v_order, ...]
│       ├── v_adj_tets_offsets: wp.array[int]     # CSR 偏移
│       ├── v_adj_faces:      wp.array[int]       # 同上
│       ├── v_adj_faces_offsets: wp.array[int]
│       ├── v_adj_edges:      wp.array[int]
│       ├── v_adj_edges_offsets: wp.array[int]
│       ├── v_adj_springs:    wp.array[int]
│       └── v_adj_springs_offsets: wp.array[int]
│
├── 自接触
│   ├── trimesh_collision_detector: TriMeshCollisionDetector
│   │   ├── bvh_tris:         wp.Bvh             # 三角形 BVH
│   │   ├── bvh_edges:        wp.Bvh             # 边 BVH
│   │   └── collision_info:   TriMeshCollisionInfo
│   │       ├── vertex_colliding_triangles:   (顶点→三角形碰撞对, CSR)
│   │       ├── triangle_colliding_vertices:  (三角形→顶点碰撞对, CSR)
│   │       └── edge_colliding_edges:         (边→边碰撞对, CSR)
│   └── 拓扑过滤列表 (CSR)
│
├── 刚体 AVBD 状态
│   ├── body_q_prev:          wp.array[wp.transform]
│   ├── body_forces:          wp.array[wp.vec3]   # 刚体线性力累加
│   ├── body_torques:         wp.array[wp.vec3]   # 刚体扭矩累加
│   ├── body_hessian_ll:      wp.array[wp.mat33]  # 6×6 Hessian: 线性-线性块
│   ├── body_hessian_al:      wp.array[wp.mat33]  # 6×6 Hessian: 角度-线性块
│   ├── body_hessian_aa:      wp.array[wp.mat33]  # 6×6 Hessian: 角度-角度块
│   ├── body_body_contact_penalty_k:  wp.array[float]   # 每接触惩罚刚度
│   ├── body_body_contact_lambda:     wp.array[wp.vec3] # 每接触 Lagrange 乘子
│   ├── body_body_contact_C0:         wp.array[wp.vec3] # 每接触 C0 稳定化参考
│   ├── body_body_contact_stick_flag: wp.array[int32]   # 粘滞标志
│   ├── body_particle_contact_penalty_k: wp.array[float] # 刚体-粒子接触 k
│   ├── joint_penalty_k:       wp.array[float]   # 每关节约束槽惩罚刚度
│   ├── joint_lambda_lin:      wp.array[wp.vec3] # 线性约束 λ
│   ├── joint_lambda_ang:      wp.array[wp.vec3] # 角度约束 λ
│   ├── joint_C0_lin:          wp.array[wp.vec3] # 线性 C0
│   └── joint_C0_ang:          wp.array[wp.vec3] # 角度 C0
```

### 3.3 CSR 邻接格式详解

`ParticleForceElementAdjacencyInfo` 是 VBD 求解器的核心数据结构，使用压缩稀疏行（CSR）格式：

```python
# 对于顶点 i，获取其第 j 个邻接三角形：
#   tri_id, vertex_order = get_vertex_adjacent_face_id_order(adjacency, i, j)
#
# 实现：
#   offset = v_adj_faces_offsets[i]
#   tri_id = v_adj_faces[offset + j * 2]       # 偶数位：元素 ID
#   v_order = v_adj_faces[offset + j * 2 + 1]  # 奇数位：顶点在元素中的局部索引

@wp.func
def get_vertex_adjacent_face_id_order(adjacency, vertex, face):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], \
           adjacency.v_adj_faces[offset + face * 2 + 1]
```

**为什么使用 CSR 格式？**
- GPU 上内存访问模式友好：同一顶点的邻接数据连续存储
- 支持变长邻接列表（不同顶点的邻接元素数不同）
- 不需要动态内存分配（在求解器构造时一次性分配）
- 偏移数组支持 O(1) 随机访问

**vertex_order 的作用**：
- 四面体：0,1,2,3 分别对应 tet 的四个顶点，用于选择正确的形函数梯度 `m_i`
- 三角形：0,1,2 对应三个顶点
- 弯曲边：0,1 是对角顶点（opposite vertices），2,3 是边端点
- 弹簧：不需要 order（弹簧只存 spring_id，不存 order）

### 3.4 邻接构建过程

邻接在求解器构造时在 CPU 上构建（`solver_vbd.py:1177`），使用两遍扫描：

```python
def _compute_particle_force_element_adjacency(self):
    with wp.ScopedDevice("cpu"):
        # 第一遍：计数
        # _count_num_adjacent_tets: 遍历所有 tet，对每个顶点的计数器 +1
        num_vertex_adjacent_tets = wp.zeros(particle_count, dtype=int32)
        wp.launch(_count_num_adjacent_tets, ...)

        # 计算 CSR 偏移：offsets[i+1] = offsets[i] + 2 * count[i]
        offsets[1:] = np.cumsum(2 * num_vertex_adjacent_tets)

        # 第二遍：填充
        # _fill_adjacent_tets: 遍历所有 tet，将 (tet_id, v_order) 写入对应位置
        wp.launch(_fill_adjacent_tets, ...)
```

**为什么在 CPU 上构建邻接？**
- 构建过程需要原子计数和 scatter 写入，GPU 上的原子操作开销大
- 邻接只需构建一次（在求解器构造时），不需要每帧更新
- CPU 上的 `np.cumsum` 比 GPU prefix sum 更简单
- 构建完成后通过 `.to(device)` 上传到 GPU

---

## 4. 完整仿真算法流程

### 4.1 顶层 step() 结构

```python
def step(state_in, state_out, control, contacts, dt):
    # ===== Phase 1: 初始化 =====
    self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid)
    self._initialize_particles(state_in, state_out, dt)

    # ===== Phase 2: 迭代求解 =====
    for iter_num in range(self.iterations):
        self._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
        self._solve_particle_iteration(state_in, state_out, contacts, dt, iter_num)

    # ===== Phase 3: 最终化 =====
    self._snapshot_rigid_contact_history(contacts)
    self._finalize_rigid_bodies(state_in, state_out, dt, ...)
    self._finalize_particles(state_out, dt)
```

**为什么先解刚体再解粒子？**
- 刚体通常更"硬"（更大的质量/惯性），先更新刚体可以提供更稳定的边界条件
- 粒子-刚体接触中，粒子侧使用刚体的当前帧位姿（`state_in.body_q`）
- 这种顺序在实践中比反向顺序更稳定

### 4.2 Phase 1: 粒子初始化

```python
def _initialize_particles(self, state_in, state_out, dt):
    # 1. 碰撞检测（在初始化前）
    if self.particle_enable_self_contact:
        self._collision_detection_penetration_free(state_in)
    else:
        self.pos_prev_collision_detection.assign(state_in.particle_q)
        self.particle_displacements.zero_()

    # 2. 前向积分
    wp.launch(kernel=forward_step, ...)
    # 对每个活动粒子：
    #   pos_prev = pos_current           # 保存当前帧起始位置
    #   vel_new = vel + (gravity + f_ext/m) * dt
    #   inertia = pos + vel_new * dt     # 惯性目标 x★
    #   displacement_init = vel_new * dt # 初始位移猜测

    # 3. 无穿透截断
    self._penetration_free_truncation(state_in.particle_q)
```

#### 4.2.1 前向积分 Kernel 详解

```python
@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.array[wp.vec3],
    pos_prev: wp.array[wp.vec3],       # output: 保存上一步位置
    pos: wp.array[wp.vec3],            # input: 当前位置
    vel: wp.array[wp.vec3],            # input: 当前速度
    inv_mass: wp.array[float],
    external_force: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    inertia_out: wp.array[wp.vec3],    # output: 惯性目标
    displacements_out: wp.array[wp.vec3], # output: 初始位移
):
    particle = wp.tid()
    pos_prev[particle] = pos[particle]

    # 运动学粒子：惯性目标 = 当前位置，位移 = 0
    if not (particle_flags[particle] & ParticleFlags.ACTIVE) or inv_mass[particle] == 0:
        inertia_out[particle] = pos[particle]
        displacements_out[particle] = wp.vec3(0.0)
        return

    # 活动粒子：显式 Euler 半步 → 惯性目标
    vel_new = vel[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt
    inertia = pos[particle] + vel_new * dt
    inertia_out[particle] = inertia
    displacements_out[particle] = vel_new * dt
```

**为什么用 `pos_prev` 保存起始位置？**
- BDF1 速度更新需要：`v_new = (x_new - x_old) / dt`
- `pos_prev` 在 forward_step 中被设为当前帧起始位置
- 在 finalize 中用于计算最终速度

**为什么惯性目标叫 "inertia"？**
- 在 BDF1 格式中，惯性项是 `m/dt² * (x - x★)`，其中 `x★ = x^n + v^n*dt + a_ext*dt²`
- `x★` 是"如果没有弹性力，粒子会到达的位置"
- 弹性力将粒子从 `x★` 拉向满足约束的位置

### 4.3 Phase 2: VBD 粒子迭代

```python
def _solve_particle_iteration(self, state_in, state_out, contacts, dt, iter_num):
    # 1. 条件性碰撞检测
    if self.particle_enable_self_contact:
        if (interval == 0 and iter_num == 0) or \
           (interval >= 1 and iter_num % interval == 0):
            self._collision_detection_penetration_free(state_in)

    # 2. 清零力/Hessian 累加缓冲区
    self.particle_forces.zero_()
    self.particle_hessians.zero_()

    # 3. 按颜色组 Gauss-Seidel 迭代
    for color in range(len(self.model.particle_color_groups)):
        # 3a. 累加 particle-body 接触力/Hessian
        if contacts is not None:
            wp.launch(accumulate_particle_body_contact_force_and_hessian, ...)

        # 3b. 累加弹簧力/Hessian
        if model.spring_count:
            wp.launch(accumulate_spring_force_and_hessian, ...)

        # 3c. 累加自接触力/Hessian
        if self.particle_enable_self_contact:
            wp.launch(accumulate_self_contact_force_and_hessian, ...)

        # 3d. 求解弹性 + 更新位移
        if self.use_particle_tile_solve:
            wp.launch(solve_elasticity_tile, ...)
        else:
            wp.launch(solve_elasticity, ...)

        # 3e. 无穿透截断
        self._penetration_free_truncation(state_in.particle_q)

    # 4. 拷贝最终位置到 state_out
    wp.copy(state_out.particle_q, state_in.particle_q)
```

#### 4.3.1 Per-Vertex 弹性求解详解

```python
@wp.kernel
def solve_elasticity(dt, particle_ids_in_color, pos_prev, pos, mass, inertia,
                     particle_flags, tri_indices, tri_poses, tri_materials, tri_areas,
                     edge_indices, edge_rest_angles, edge_rest_length, edge_bending_properties,
                     tet_indices, tet_poses, tet_materials,
                     particle_adjacency, particle_forces, particle_hessians,
                     particle_displacements):
    t_id = wp.tid()
    particle_index = particle_ids_in_color[t_id]

    # 运动学粒子：位移 = 0
    if not (particle_flags[particle_index] & ParticleFlags.ACTIVE) or mass[particle_index] == 0:
        particle_displacements[particle_index] = wp.vec3(0.0)
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # ===== 惯性力/Hessian =====
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * dt_sqr_reciprocal
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # ===== 三角形膜弹性 =====
    if tri_indices:
        for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
            tri_index, vertex_order = get_vertex_adjacent_face_id_order(...)
            if tri_materials[tri_index, 0] > 0.0 or tri_materials[tri_index, 1] > 0.0:
                f_tri, h_tri = evaluate_neo_hookean_membrane_force_hessian(...)
                f += f_tri
                h += h_tri

    # ===== 弯曲边弹性 =====
    if edge_indices:
        for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
            nei_edge_index, vertex_order = get_vertex_adjacent_edge_id_order(...)
            if edge_bending_properties[nei_edge_index, 0] > 0.0:
                f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(...)
                f += f_edge
                h += h_edge

    # ===== 四面体体积弹性 =====
    if tet_indices:
        for adj_tet_counter in range(get_vertex_num_adjacent_tets(adjacency, particle_index)):
            nei_tet_index, vertex_order = get_vertex_adjacent_tet_id_order(...)
            if tet_materials[nei_tet_index, 0] > 0.0 or tet_materials[nei_tet_index, 1] > 0.0:
                f_tet, h_tet = evaluate_volumetric_neo_hookean_force_and_hessian(...)
                f += f_tet
                h += h_tet

    # ===== 加上外部累加的力/Hessian（弹簧、接触） =====
    h += particle_hessians[particle_index]
    f += particle_forces[particle_index]

    # ===== 求解 3×3 线性系统 =====
    if abs(wp.determinant(h)) > 1e-8:
        h_inv = wp.inverse(h)
        particle_displacements[particle_index] += h_inv * f
```

**为什么每个顶点只解 3×3 系统？**
- 这是 VBD 的核心创新：将全局 N×N 系统分解为 N 个独立的 3×3 系统
- 顶点间的耦合（off-diagonal Hessian blocks）通过 Gauss-Seidel 的"最新邻居值"策略隐式处理
- 3×3 矩阵求逆是解析的（通过 `wp.inverse`），非常快
- 避免了全局稀疏矩阵的组装、存储和分解

**为什么检查 `det(h) > 1e-8`？**
- 确保 Hessian 可逆（非奇异）
- 惯性项 `m/dt² * I` 保证了 Hessian 的正定性（除非 m=0）
- 数值上极少出现奇异情况，但这是一个安全保护

#### 4.3.2 Tiled Solve 加速 (CUDA only)

```python
@wp.kernel
def solve_elasticity_tile(dt, particle_ids_in_color, ...):
    tid = wp.tid()
    block_idx = tid // TILE_SIZE  # 16 线程共享一个顶点
    thread_idx = tid % TILE_SIZE
    particle_index = particle_ids_in_color[block_idx]

    # 每个线程处理邻接列表的一个切片
    batch_counter = 0
    while batch_counter + thread_idx < num_adj_faces:
        adj_tri_counter = thread_idx + batch_counter
        batch_counter += TILE_SIZE
        # ... 计算该三角形的力/Hessian 贡献
        f += f_tri
        h += h_tri

    # Warp tile reduce：将 16 个线程的 f, h 合并
    f_tile = wp.tile(f, preserve_type=True)
    h_tile = wp.tile(h, preserve_type=True)
    f_total = wp.tile_reduce(wp.add, f_tile)[0]
    h_total = wp.tile_reduce(wp.add, h_tile)[0]

    # 只有线程 0 执行最终求解
    if thread_idx == 0:
        h_total += mass * dt_sqr_reciprocal * I + particle_hessians[particle_index]
        if abs(wp.determinant(h_total)) > 1e-8:
            h_inv = wp.inverse(h_total)
            f_total += mass * (inertia - pos) * dt_sqr_reciprocal + particle_forces[particle_index]
            particle_displacements[particle_index] += h_inv * f_total
```

**为什么需要 Tiled Solve？**
- 单个顶点的邻接元素可能很多（如 tet 网格中每个顶点可能邻接 20+ 个 tet）
- 单个线程串行处理所有邻接元素会很慢
- Tiled solve 用 16 个线程协作处理一个顶点，每个线程只处理 1/16 的邻接元素
- `wp.tile_reduce` 是 Warp 的硬件加速归约操作
- 仅在 CUDA 上可用（CPU 不支持 tile API）

### 4.4 Phase 2: AVBD 刚体迭代

```python
def _solve_rigid_body_iteration(self, state_in, state_out, control, contacts, dt):
    # 清零力/Hessian
    self.body_torques.zero_()
    self.body_forces.zero_()
    self.body_hessian_aa.zero_()
    self.body_hessian_al.zero_()
    self.body_hessian_ll.zero_()

    for color in range(len(body_color_groups)):
        color_group = body_color_groups[color]

        # 累加 body-particle 接触
        wp.launch(accumulate_body_particle_contacts_per_body, ...)

        # 累加 body-body 接触
        wp.launch(accumulate_body_body_contacts_per_body, ...)

        # 求解刚体 6×6 系统 + 关节约束
        wp.launch(solve_rigid_body, ...)

    # 更新 AVBD 双变量
    wp.launch(update_duals_body_body_contacts, ...)
    wp.launch(update_duals_body_particle_contacts, ...)
    wp.launch(update_duals_joint, ...)
```

#### 4.4.1 刚体 6×6 求解详解

```python
@wp.kernel
def solve_rigid_body(dt, body_ids_in_color, body_q, body_q_prev, body_q_rest,
                     body_mass, body_inv_mass, body_inertia, body_inertia_q, body_com,
                     adjacency, joint_*, external_forces, external_torques,
                     external_hessian_ll, external_hessian_al, external_hessian_aa,
                     body_q_new):
    tid = wp.tid()
    body_index = body_ids_in_color[tid]

    # 运动学刚体：不更新
    if body_inv_mass[body_index] == 0.0:
        body_q_new[body_index] = body_q[body_index]
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # ===== 惯性力/Hessian =====
    # 线性部分
    com_current = pos_current + rotate(rot_current, body_com_local)
    com_star = pos_star + rotate(rot_star, body_com_local)
    f_lin = mass * dt_sqr_reciprocal * (com_star - com_current)

    # 角度部分：使用四元数差 → 旋转向量
    q_delta = inverse(rot_current) * rot_star
    axis_body, angle_body = quat_to_axis_angle(q_delta)
    theta_body = axis_body * angle_body
    tau_world = rotate(rot_current, I_body * theta_body * dt_sqr_reciprocal)

    # 角度 Hessian：世界坐标系下的惯性张量
    I_world = R_cur * I_body * R_cur^T
    angular_hessian = dt_sqr_reciprocal * I_world

    # ===== 累加外部接触力/Hessian =====
    f_force = f_lin + external_forces[body_index]
    f_torque = tau_world + external_torques[body_index]
    h_ll = inertial_coeff * I + external_hessian_ll[body_index]
    h_al = external_hessian_al[body_index]
    h_aa = angular_hessian + external_hessian_aa[body_index]

    # ===== 累加关节约束力/Hessian =====
    for joint_counter in range(num_adj_joints):
        joint_idx = get_body_adjacent_joint_id(...)
        joint_force, joint_torque, joint_H_ll, joint_H_al, joint_H_aa = \
            evaluate_joint_force_hessian(...)
        f_force += joint_force
        f_torque += joint_torque
        h_ll += joint_H_ll
        h_al += joint_H_al
        h_aa += joint_H_aa

    # ===== 角度 Hessian 正则化 =====
    trA = trace(h_aa) / 3.0
    epsA = 1e-9 * (trA + 1.0)
    h_aa[0,0] += epsA; h_aa[1,1] += epsA; h_aa[2,2] += epsA

    # ===== 求解 6×6 SPD 系统 =====
    x_inc, w_world = ldlt6_solve(h_ll, h_aa, h_al, f_force, f_torque)

    # ===== 更新位姿 =====
    # 角度增量 → 四元数（小角度近似）
    half_w = w_world * 0.5
    dq_world = normalize(quat(half_w[0], half_w[1], half_w[2], 1.0))
    rot_new = normalize(dq_world * rot_current)

    # 位置更新
    com_new = com_current + x_inc
    pos_new = com_new - rotate(rot_new, body_com_local)

    body_q_new[body_index] = transform(pos_new, rot_new)
```

**为什么使用 6×6 LDLᵀ 直接求解而不是迭代求解？**
- 6×6 系统非常小，直接分解比迭代法更快
- LDLᵀ 适用于 SPD 矩阵（惯性 + 弹性 Hessian 保证 SPD）
- 展开的 LDLᵀ 代码（`ldlt6_solve`）无分支，GPU 友好
- 避免了迭代法的收敛判断和额外内存

**为什么需要角度 Hessian 正则化？**
- 自由旋转轴（如 revolute joint 的旋转轴）上的角度 Hessian 可能为零
- 零 Hessian 导致奇异系统
- 添加小的正则化项 `epsA * I` 保证可逆，同时不影响约束方向的解

**为什么使用小角度近似？**
- `_USE_SMALL_ANGLE_APPROX = True`（默认）
- 小角度近似：`exp(θ) ≈ 1 + θ/2`（四元数形式）
- 在典型时间步长（dt ≈ 1/60s）下，单步旋转角很小
- 近似避免了 `sin/cos` 计算，性能更好
- 对于大旋转，可以使用精确的 `quat_from_axis_angle`

### 4.5 Phase 3: 最终化

#### 4.5.1 粒子速度更新

```python
@wp.kernel
def update_velocity(dt, pos_prev, pos, vel):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt
```

**为什么用 BDF1 而不是显式速度更新？**
- BDF1：`v_new = (x_new - x_old) / dt`
- 这与隐式 Euler 的位移公式 `x_new = x_old + v_new * dt` 一致
- 保证速度与最终位置一致（动量守恒）

#### 4.5.2 刚体速度更新 + 粘滞死区

```python
@wp.kernel
def update_body_velocity(dt, body_q, body_com, ...,
                         body_q_prev, body_qd_out, body_qd_in, body_q_out,
                         apply_stick_deadzone, freeze_trans_eps, freeze_angular_eps):
    body_index = wp.tid()

    # 粘滞死区：如果位姿变化低于阈值，恢复到上一步
    if apply_stick_deadzone:
        if has_sticky_contacts(body_index):
            dpos = pos_current - pos_prev
            if length(dpos) < freeze_trans_eps:
                pos_current = pos_prev  # snap translation
            dangle = angle_between(rot_current, rot_prev)
            if dangle < freeze_angular_eps:
                rot_current = rot_prev  # snap rotation

    # BDF1 速度
    body_qd_out[body_index] = (pos_current - pos_prev) / dt  # + 角度速度
    body_q_prev[body_index] = body_q[body_index]  # 保存用于下一帧
    body_q_out[body_index] = transform(pos_current, rot_current)
```

**为什么需要粘滞死区（stick deadzone）？**
- 在静摩擦状态下，接触点理论上不应滑动
- 但数值误差会导致微小的"爬行"（creep）
- 死区将低于阈值的运动直接归零，防止累积漂移
- 这是 AVBD 论文中描述的 anti-creep 机制

---

## 5. 接触算法详解

### 5.1 接触类型总览

Newton VBD 处理三类接触，各有不同的检测和响应机制：

| 接触类型 | 检测器 | 响应方式 | 力作用对象 |
|---------|--------|---------|-----------|
| 粒子自接触 (vertex-triangle + edge-edge) | `TriMeshCollisionDetector` (BVH) | Penalty + friction, VBD per-vertex Hessian | 粒子 |
| 刚体-粒子接触 | `CollisionPipeline` (GJK/MPR) | Penalty + friction, AVBD k-update | 粒子 + 刚体 |
| 刚体-刚体接触 | `CollisionPipeline` (GJK/MPR) | Penalty + AL + friction + C0 stabilization | 刚体 |

### 5.2 粒子自接触检测

#### 5.2.1 检测器架构

```python
class TriMeshCollisionDetector:
    def __init__(self, model, ...):
        # 两个 BVH：三角形 BVH + 边 BVH
        self.bvh_tris = wp.Bvh(lower_bounds_tris, upper_bounds_tris)
        self.bvh_edges = wp.Bvh(lower_bounds_edges, upper_bounds_edges)

        # 碰撞缓冲区（CSR 格式，预分配）
        self.vertex_colliding_triangles   # 顶点→三角形碰撞对
        self.edge_colliding_edges         # 边→边碰撞对
```

**为什么用两个独立的 BVH？**
- 三角形 BVH 用于 vertex-triangle 碰撞查询
- 边 BVH 用于 edge-edge 碰撞查询
- 两种查询的几何体不同（三角形 vs 线段），AABB 大小不同
- 分离的 BVH 可以独立 refit/rebuild

#### 5.2.2 碰撞检测流程

```python
def _collision_detection_penetration_free(self, current_state):
    # 1. 保存碰撞前位置
    self.pos_prev_collision_detection.assign(current_state.particle_q)
    self.particle_displacements.zero_()

    # 2. BVH refit（更新 AABB 到当前位置）
    self.trimesh_collision_detector.refit(current_state.particle_q)

    # 3. 顶点-三角形碰撞检测
    self.trimesh_collision_detector.vertex_triangle_collision_detection(
        self.particle_self_contact_margin,     # 查询半径
        min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
        min_distance_filtering_ref_pos=self.particle_q_rest,
    )

    # 4. 边-边碰撞检测
    self.trimesh_collision_detector.edge_edge_collision_detection(
        self.particle_self_contact_margin,
        min_query_radius=self.particle_rest_shape_contact_exclusion_radius,
        min_distance_filtering_ref_pos=self.particle_q_rest,
    )
```

**为什么在碰撞检测前保存位置？**
- Planar DAT 需要知道"碰撞检测时的位置"作为截断的参考点
- 所有后续位移都是相对于这个参考点的
- 每次碰撞检测时重置位移为零

#### 5.2.3 BVH Refit vs Rebuild

```python
def refit(self, new_pos=None):
    """更新 AABB 并 refit BVH（不改变树结构）"""
    # 重新计算 AABB
    wp.launch(compute_tri_aabbs, ...)
    self.bvh_tris.refit()  # 只更新 bounding box，不改变树拓扑

def rebuild(self, state):
    """完全重建 BVH（改变树结构）"""
    # 重新计算 AABB + 重建树
    wp.launch(compute_tri_aabbs, ...)
    self.bvh_tris.rebuild()
```

**什么时候用 refit vs rebuild？**
- Refit：每帧调用，快（O(N)），但树质量逐渐下降
- Rebuild：当物体变形很大时调用，慢（O(N log N)），但树质量最优
- 用户可以通过 `solver.rebuild_bvh(state)` 手动触发重建

#### 5.2.4 拓扑过滤

拓扑过滤排除几何上相邻的原始体之间的碰撞检测：

```python
def build_vertex_n_ring_tris_collision_filter(n, num_vertices, edge_indices, ...):
    """
    对每个顶点 v，收集其 n-ring 邻居顶点的所有邻接三角形，
    但排除 v 自身的 1-ring 三角形。
    这些三角形被加入过滤列表，不会与 v 进行碰撞检测。
    """
    for v in range(num_vertices):
        # 获取 v 的 n-ring 邻居顶点
        ring_n_minus_1 = leq_n_ring_vertices(v, edge_indices, n-1, ...)

        # v 自身的 1-ring 三角形（需要排除）
        ring_1_tri_set = set(v 的邻接三角形)

        # 收集邻居顶点的三角形
        nei_tri_set = set()
        for w in ring_n_minus_1:
            if w != v:
                nei_tri_set.update(w 的邻接三角形)

        # 排除 1-ring
        nei_tri_set.difference_update(ring_1_tri_set)
        v_nei_tri_sets[v] = nei_tri_set
```

**为什么需要拓扑过滤？**
- 相邻的三角形共享顶点/边，它们之间的"碰撞"是正常的网格变形，不应产生接触力
- 不过滤会导致：
  - 虚假的排斥力（相邻面被推开）
  - 性能浪费（大量假阳性碰撞对）
- 默认 `threshold=2`（2-ring 过滤），即排除 2 环内的所有三角形

**为什么用 `n-ring` 而不是简单的共享顶点检测？**
- 共享顶点的三角形（1-ring）显然不应碰撞
- 但 2-ring 三角形（共享顶点的邻居的三角形）在弯曲时也可能非常接近
- `threshold=2` 表示：只检测拓扑距离 ≥ 3 的三角形对
- 更高的 threshold 过滤更多，但计算开销更大

### 5.3 粒子自接触力计算

#### 5.3.1 C2 连续接触屏障函数

```python
@wp.func
def evaluate_self_contact_force_norm(dis, collision_radius, k):
    tau = collision_radius * 0.5      # 屏障-惩罚分界点
    d_min = 1.0e-5                     # 最小距离阈值

    if tau > dis > d_min:
        # Log-barrier 区域：E ∝ -ln(d)
        # 提供指数级增长的排斥力，防止真正穿透
        k2 = tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)

    elif dis <= d_min:
        # 二次扩展：log-barrier 在 d_min 处的 Taylor 展开
        # 保持 C2 连续性，避免 d→0 时的数值爆炸
        k2 = tau * tau * k
        d_min_sq = d_min * d_min
        dEdD = k2 * (dis - 2.0 * d_min) / d_min_sq
        d2E_dDdD = k2 / d_min_sq

    else:  # dis >= tau
        # 二次惩罚区域：标准的 spring-like 力
        penetration_depth = collision_radius - dis
        dEdD = -k * penetration_depth
        d2E_dDdD = k
```

**为什么设计三段式屏障函数？**
1. **远距离 (d > tau)**：标准二次惩罚，力与穿透深度成正比，计算简单
2. **中间距 (tau > d > d_min)**：对数屏障，力随距离减小而指数增长，提供强大的穿透防护
3. **极近距 (d < d_min)**：对数屏障的二次扩展，防止 d→0 时力→∞ 的数值爆炸

**为什么需要 C2 连续性？**
- VBD 是 Newton 型方法，需要 Hessian（二阶导数）
- C2 连续保证 Hessian 在分段点处连续，避免 Newton 步的不稳定
- 在 `d = tau` 和 `d = d_min` 处，`dEdD` 和 `d2E_dDdD` 都是连续的

#### 5.3.2 顶点-三角形接触力

```python
@wp.func
def evaluate_vertex_triangle_collision_force_hessian_4_vertices(
    v, tri, pos, pos_anchor, tri_indices,
    collision_radius, collision_stiffness, collision_damping,
    friction_coefficient, friction_epsilon, dt
):
    a, b, c = pos[tri_indices[tri, 0]], pos[tri_indices[tri, 1]], pos[tri_indices[tri, 2]]
    p = pos[v]

    # 三角形上的最近点
    closest_p, bary, _feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = length(diff)
    collision_normal = diff / dis

    if 0.0 < dis < collision_radius:
        # 重心坐标权重：bs = [-α, -β, -γ, 1]
        bs = vec4(-bary[0], -bary[1], -bary[2], 1.0)

        # 法向力
        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)
        collision_force = [-dEdD * bs[i] * collision_normal for i in 0..3]
        collision_hessian = [d2E_dDdD * bs[i]^2 * outer(n, n) for i in 0..3]

        # 摩擦力
        dx_v = p - pos_anchor[v]                              # 顶点的位移
        closest_p_prev = bary[0]*a_prev + bary[1]*b_prev + bary[2]*c_prev
        dx = dx_v - (closest_p - closest_p_prev)              # 相对切向位移

        e0, e1 = orthonormal_basis(collision_normal)          # 切平面基
        T = mat32(e0, e1)                                     # 3×2 投影矩阵
        u = T^T * dx                                          # 2D 切向位移

        friction_force, friction_hessian = compute_friction(
            friction_coefficient, -dEdD, T, u, friction_epsilon * dt
        )

        # 阻尼（仅在接近时）
        for i in 0..3:
            displacement = pos_anchor[vertex_i] - pos[vertex_i]
            if dot(displacement, collision_normal * sign[i]) > 0:
                damping_hessian = (collision_damping / dt) * collision_hessian[i]
                collision_force[i] += damping_hessian * displacement
                collision_hessian[i] += damping_hessian

        # 合并法向力 + 摩擦力
        for i in 0..3:
            collision_force[i] += bs[i] * friction_force
            collision_hessian[i] += bs[i]^2 * friction_hessian
```

**为什么用重心坐标作为权重？**
- 接触力应该按重心坐标分配到三角形的三个顶点上
- 如果接触点靠近顶点 a（bary ≈ [1,0,0]），力几乎全部作用在 a 上
- 这保证了力的空间一致性和动量守恒

**为什么摩擦 Hessian 将法向力视为常数？**
```python
# 精确 IPC 摩擦 Hessian（复杂）:
# H_friction = ∂/∂x (μ * |f_normal(x)| * T * u/|u|)
# 包含 ∂f_normal/∂x 的复杂耦合项

# Newton 的简化（稳定）:
# H_friction = μ * |f_normal| * T * (f1_SF_over_x * I₂) * T^T
# 将 |f_normal| 视为常数，忽略 ∂f_normal/∂x 项
```
- 法向力对切向位移的导数（∂f_normal/∂u）在实践中很小
- 忽略该项使摩擦 Hessian 更"良性"（更接近 SPD）
- 显著提高稳定性，代价是略微降低收敛速度

#### 5.3.3 边-边接触力

```python
@wp.func
def evaluate_edge_edge_contact_2_vertices(e1, e2, pos, pos_anchor, edge_indices, ...):
    e1_v1_pos, e1_v2_pos = pos[edge_indices[e1, 2]], pos[edge_indices[e1, 3]]
    e2_v1_pos, e2_v2_pos = pos[edge_indices[e2, 2]], pos[edge_indices[e2, 3]]

    # 边-边最近点
    s, t, dis = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos,
                                            e2_v1_pos, e2_v2_pos, parallel_epsilon)
    c1 = e1_v1_pos + (e1_v2_pos - e1_v1_pos) * s
    c2 = e2_v1_pos + (e2_v2_pos - e2_v1_pos) * t
    collision_normal = (c1 - c2) / dis

    if dis < collision_radius:
        # 重心坐标权重：bs = [1-s, s, -(1-t), -t]
        bs = vec4(1.0 - s, s, -1.0 + t, -t)

        # 法向力（与顶点-三角形相同的公式）
        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)
        collision_force = [-dEdD * bs[i] * collision_normal for i in 0..3]
        collision_hessian = [d2E_dDdD * bs[i]^2 * outer(n, n) for i in 0..3]

        # 摩擦力（类似顶点-三角形）
        # ...
```

**为什么边-边使用 4 个重心坐标？**
- 两条边的 4 个端点都需要受力
- `s` 是 e1 上的参数（0=端点1, 1=端点2）
- `t` 是 e2 上的参数
- 权重符号：e1 侧为正，e2 侧为负（因为法向从 e1 指向 e2）

### 5.4 无穿透截断 (Planar DAT)

Planar DAT 是 VBD 的**核心安全机制**，保证每次 Gauss-Seidel 迭代后的位移不会导致穿透。

#### 5.4.1 算法原理

```
对每个碰撞对（vertex-triangle 或 edge-edge）：
  1. 构造分割平面 (n, d)：
     - n 指向碰撞对的第一侧（vertex 侧或 edge1 侧）
     - d 基于当前最近点的加权中点

  2. 对碰撞对的每个顶点：
     计算截断系数 t = (d - v) · n / (Δv · n)
     t = clamp(min(t * γ_r, t - γ_min), 0, 1)
     其中 γ_r = 0.85 (conservative_bound_relaxation)

  3. 对每个顶点 i：t_i = min(所有碰撞对中的 t_i)  (atomic_min)

应用截断：
  Δx_i_final = Δx_i * t_i
  x_i_new = x_i_prev + Δx_i_final
```

#### 5.4.2 分割平面构造

```python
@wp.func
def create_vertex_triangle_division_plane_closest_pt(
    v, delta_v, t1, delta_t1, t2, delta_t2, t3, delta_t3
):
    # 当前最近点
    closest_p, bary, _ = triangle_closest_point(t1, t2, t3, v)
    n_hat = v - closest_p

    if length(n_hat) < 1e-12:
        return dummy  # 已经在三角形上，无法定义平面

    n = normalize(n_hat)

    # 计算两侧沿法向的最大位移
    delta_v_n = max(-dot(n, delta_v), 0.0)    # 顶点侧（向三角形移动）
    delta_t_n = max(dot(n, delta_t1), dot(n, delta_t2), dot(n, delta_t3), 0.0)  # 三角形侧

    if delta_t_n + delta_v_n == 0.0:
        d = closest_p + 0.5 * n_hat           # 无相对运动：中点
    else:
        lambda = delta_t_n / (delta_t_n + delta_v_n)
        lambda = clamp(lambda, 0.05, 0.95)
        d = closest_p + lambda * n_hat        # 按速度比加权
```

**为什么分割平面按速度比加权？**
- 如果顶点快速向三角形移动（delta_v_n 大），平面应靠近三角形侧
- 如果三角形快速向顶点移动（delta_t_n 大），平面应靠近顶点侧
- 加权确保平面在"碰撞可能发生的位置"
- `clamp(0.05, 0.95)` 防止平面完全退到一侧

#### 5.4.3 截断系数计算

```python
@wp.func
def planar_truncation_t(v, delta_v, n, d, eps, gamma_r, gamma_min=1e-3):
    denom = dot(n, delta_v)
    if abs(denom) < eps:
        return 1.0  # 平行于平面：不截断

    t = dot(n, d - v) / denom  # 到达平面的时间

    if t < 0:
        return 1.0  # 已经过了平面：不截断

    # 保守截断：t * gamma_r 或 t - gamma_min，取更保守的
    t = clamp(min(t * gamma_r, t - gamma_min), 0.0, 1.0)
    return t
```

**为什么需要 `gamma_r` 和 `gamma_min`？**
- `gamma_r = 0.85`：将截断点提前 15%，提供安全余量
- `gamma_min = 1e-3`：即使 t 很小，也至少留出 gamma_min 的余量
- 两者结合防止因数值误差导致的微穿透

### 5.5 刚体-粒子接触

#### 5.5.1 接触检测

刚体-粒子接触由 `CollisionPipeline` 检测（`sim/collide.py`），使用 GJK/MPR 算法：

```python
model.collide(state_in, contacts)
# contacts.soft_contact_particle[i]  = 粒子索引
# contacts.soft_contact_shape[i]    = 形状索引
# contacts.soft_contact_body_pos[i] = 接触点（刚体局部坐标）
# contacts.soft_contact_normal[i]   = 接触法向
```

#### 5.5.2 接触响应

粒子侧（VBD 迭代中）：
```python
# accumulate_particle_body_contact_force_and_hessian kernel
for each contact:
    if particle_colors[particle_idx] == current_color:
        force, hessian = _eval_body_particle_contact(
            particle_idx, particle_pos, particle_prev_pos,
            contact_idx, contact_ke, contact_kd, contact_mu, ...
        )
        atomic_add(particle_forces, particle_idx, force)
        atomic_add(particle_hessians, particle_idx, hessian)
```

刚体侧（AVBD 迭代中）：
```python
# accumulate_body_particle_contacts_per_body kernel
for each body in color_group:
    for each contact on this body:
        force, torque, H_ll, H_al, H_aa = evaluate_body_particle_contact(...)
        atomic_add(body_forces, body_idx, force)
        atomic_add(body_torques, body_idx, torque)
        # ... Hessian 同理
```

**为什么接触力同时作用于粒子和刚体？**
- 牛顿第三定律：粒子受到的力 = -刚体受到的力
- 粒子侧和刚体侧分别累加，保证双向耦合
- 共享的 `body_particle_contact_penalty_k` 确保两侧使用相同的刚度

### 5.6 刚体-刚体接触

#### 5.6.1 Hard 模式（默认，Augmented Lagrangian）

```python
# compute_rigid_contact_forces kernel
C_n = thickness - dot(contact_normal, cp1_world - cp0_world)  # 穿透深度

if hard_contacts:
    lam_vec = contact_lambda[contact_idx]
    lam_n = dot(lam_vec, contact_normal)
    C0_vec = contact_C0[contact_idx]
    C0_n = dot(contact_normal, C0_vec)

    # C0 稳定化
    C_eff = C_n - avbd_alpha * C0_n

    # 有效法向力
    f_n = k * C_eff + lam_n

# 如果无穿透且无历史 λ，跳过
if C_n <= 0 and lam_n <= 0:
    return

# 计算完整的接触力（法向 + 摩擦）
force_0, torque_0, ..., force_1, torque_1, ... = evaluate_rigid_contact_from_collision(...)
```

**为什么需要 C0 稳定化？**
- 纯 penalty 方法在接触刚度有限时会有残余穿透
- C0 稳定化将约束违反从 `C` 改为 `C - α*C0`，其中 C0 是初始穿透
- 当 α=1 时，`C_stab = C - C0`，即只惩罚"新增的穿透"
- 当 α=0 时，`C_stab = C`，即惩罚全部穿透
- α=0.95 是一个折中：主要惩罚新增穿透，但也保留少量绝对穿透惩罚

#### 5.6.2 AVBD 双变量更新

```python
# update_duals_body_body_contacts kernel
for each contact:
    # 计算当前约束违反
    C_vec = compute_current_constraint_violation(...)

    # Hard 模式：更新 λ
    if hard_contacts:
        C_stab = C_vec - avbd_alpha * C0_vec
        lambda_new = k * C_stab + lambda_old
    else:
        lambda_new = lambda_old  # Soft 模式：λ 不变

    # 更新 penalty k
    if beta > 0:
        k_new = min(k_max, k_old + beta * |C_vec|)

    # 粘滞检测
    if tangential_slip < stick_motion_eps:
        stick_flag = STICK_FLAG_DEADZONE
```

**为什么需要 augmented Lagrangian（λ 累积）？**
- 纯 penalty 方法需要无限大的刚度才能精确满足约束
- AL 通过累积 λ 来补偿有限的 penalty 刚度
- λ 的更新规则 `λ_new = k*C_stab + λ_old` 等价于 Uzawa 迭代
- 在实践中，AL 可以在有限刚度下将约束违反驱动到任意小

#### 5.6.3 Contact History Warm-Start

```python
# 每帧结束时：保存接触状态
wp.launch(snapshot_body_body_contact_history, ...)
# 保存: lambda, stick_flag, penalty_k, point0/1, offset0/1, normal

# 下一帧开始时：通过 match_index 恢复
if rigid_contact_history:
    wp.launch(restore_body_body_contact_history, ...)
    # 根据 match_index 将上一帧的接触状态映射到当前帧的接触对
```

**为什么 warm-start 重要？**
- 连续帧之间的大多数接触是持久的（同一对物体持续接触）
- 复用上一帧的 λ 和 k 可以显著加速收敛
- 特别是对于静止接触（如物体放在地面上），warm-start 可以立即提供正确的接触力
- 需要 `CollisionPipeline(contact_matching="latest")` 来生成 match_index

---

## 6. 核心算法公式与推导

### 6.1 Stable Neo-Hookean 体积能量

#### 6.1.1 变形梯度

```
F = Ds · Dm⁻¹

Ds = [x₁-x₀, x₂-x₀, x₃-x₀]    # 变形后边矩阵 (3×3)
Dm = [X₁-X₀, X₂-X₀, X₃-X₀]    # 静息边矩阵 (3×3)
```

#### 6.1.2 能量密度

```
ψ(F) = (μ_NH/2)(I_C - 3) + (λ_NH/2)(J - α)²

其中:
  I_C = tr(FᵀF) = ||F||_F²
  J   = det(F)
  α   = 1 + μ_NH/λ_NH
```

**为什么使用 Smith et al. 2018 的稳定化形式？**
- 经典的 Neo-Hookean `ψ = μ/2(tr(FᵀF)-3) - μ ln(J) + λ/2 ln²(J)` 在 J→0 时能量趋于无穷（不可压缩）
- 稳定化形式用 `(J-α)²` 替代 `ln²(J)`，避免了 J→0 的奇异性
- 同时保持了小应变时的物理正确性（通过 α 参数匹配 Lamé 参数）

#### 6.1.3 Lamé → NH 参数转换

```
μ_NH   = μ_Lamé
λ_NH   = λ_Lamé + μ_Lamé
```

**为什么 λ_NH ≠ λ_Lamé？**
- NH 能量中的 (λ, μ) 符号与 Lamé 参数不同
- 匹配小应变极限（F ≈ I + ε）得到转换关系
- 详见 Smith et al. 2018, Eq. 13

#### 6.1.4 第一 Piola-Kirchhoff 应力

```
P(F) = ∂ψ/∂F = μ_NH · F + λ_NH · (J - α) · cof(F)
```

其中 `cof(F)` 是余子式矩阵（adjugate），通过直接计算获得：
```python
cof(F) = [[F22*F33 - F23*F32, F23*F31 - F21*F33, F21*F32 - F22*F31],
          [F13*F32 - F12*F33, F11*F33 - F13*F31, F12*F31 - F11*F32],
          [F12*F23 - F13*F22, F13*F21 - F11*F23, F11*F22 - F12*F21]]
```

**为什么直接计算 cof(F) 而不是 J·F⁻ᵀ？**
- 当 J ≈ 0 时（完全压扁的 tet），F⁻ᵀ 数值不稳定
- 余子式的直接计算公式只涉及乘法和减法，无条件数值稳定
- 这是 Newton 代码中的关键数值优化

#### 6.1.5 Per-Vertex Force

```
f_i = -P · m_i

其中 m_i 是形函数梯度（Dm⁻¹ 的列或列的组合）:
  m₀ = -(Dm⁻¹_col0 + Dm⁻¹_col1 + Dm⁻¹_col2)
  m₁ = Dm⁻¹_col0
  m₂ = Dm⁻¹_col1
  m₃ = Dm⁻¹_col2
```

**为什么 m₀ 是负和？**
- 形函数 N_i 满足 Σ N_i = 1，所以 N₀ = 1 - N₁ - N₂ - N₃
- ∇N₀ = -∇N₁ - ∇N₂ - ∇N₃
- ∇N_i 由 Dm⁻¹ 的列给出

#### 6.1.6 Per-Vertex Hessian (SPD 近似)

```
完整的弹性 Hessian:
  H_full = μ_NH · I₉ + λ_NH · cof_vec · cof_vecᵀ + s · ∂²J/∂F²

其中 s = λ_NH · (J - α)

在 VBD 的 per-vertex 3×3 块中:
  H_i = G_iᵀ · H_full · G_i

其中 G_i = ∂vec(F)/∂x_i 是 9×3 矩阵

关键简化: s · ∂²J/∂F² 项在 per-vertex 块中恒为零！
原因: ∂²J/∂F² 包含 Levi-Civita 张量，与 (m^a × m^a) 的缩并恒为零。

因此:
  H_i = rest_volume · G_iᵀ · (μ_NH · I₉ + λ_NH · cof_vec · cof_vecᵀ) · G_i
```

**为什么可以省略 ∂²J/∂F² 项？**
- 这是 VBD 的一个优雅的数学性质
- 在全局 Hessian 中该项非零，但在 per-vertex 3×3 对角块中恰好为零
- 省略后剩余两项是 SPD 的（by inspection：μ_NH·I 正定，外积项半正定）
- SPD 保证局部子问题有唯一解

#### 6.1.7 Rayleigh 阻尼

```
H_total = H_i * (1 + damping/dt)

f_damp = damping · H_i · Ḟ_vec / dt

其中 Ḟ 通过有限差分计算:
  Ḟ = (Ds_new - Ds_old)/dt · Dm⁻¹
```

**为什么阻尼项是 `H_i * (1 + damping/dt)`？**
- Rayleigh 阻尼假设阻尼矩阵 C = damping · K（与刚度矩阵成比例）
- 在 BDF1 格式中，有效 Hessian = M/dt² + K + C/dt = M/dt² + K(1 + damping/dt)
- 所以弹性 Hessian 乘以 (1 + damping/dt)

### 6.2 Stable Neo-Hookean 膜能量 (2D)

#### 6.2.1 变形梯度

```
F = [x₁-x₀, x₂-x₀] · Dm⁻¹    # 3×2 矩阵
Dm⁻¹ ∈ R^(2×2)  存储在 tri_poses[i] 中
```

#### 6.2.2 不变量

```
I_C  = tr(FᵀF) = ||f₀||² + ||f₁||²
J_s  = sqrt(det(FᵀF)) = sqrt(||f₀||²||f₁||² - (f₀·f₁)²)
```

**为什么 2D 用 J_s 而不是 J？**
- 3×2 矩阵没有行列式（不是方阵）
- J_s = sqrt(det(FᵀF)) 是面积比（变形后面积/静息面积）
- 在 2D 中，J_s 起到与 3D 中 J 相同的作用

#### 6.2.3 2D "余子式"向量

```
g₀ = ∂J_s/∂f₀ = (1/J_s) * (||f₁||²·f₀ - (f₀·f₁)·f₁)
g₁ = ∂J_s/∂f₁ = (1/J_s) * (||f₀||²·f₁ - (f₀·f₁)·f₀)
```

#### 6.2.4 Per-Vertex Hessian 的 PSD Clamp

```
s = λ_NH · (J_s - α)
s_clamp = max(0, s)          # PSD 保证
r = s_clamp / J_s

I_coeff = μ_NH · (df₀² + df₁²) + r · (df₀²·||f₁||² + df₁²·||f₀||² - 2·df₀·df₁·(f₀·f₁))
c1 = λ_NH - r

H = I_coeff · I₃ + c1 · dJ_dx · dJ_dxᵀ - r · w · wᵀ

其中:
  df₀ = ∂f₀/∂x_i, df₁ = ∂f₁/∂x_i
  dJ_dx = g₀·df₀ + g₁·df₁
  w = f₁·df₀ - f₀·df₁
```

**为什么需要 PSD clamp？**
- 与 3D 不同，膜的 ∂²J_s/∂F² 项在 per-vertex 块中**不**为零
- 当 s < 0 时（压缩状态），该项可能导致 Hessian 非 PSD
- `s_clamp = max(0, s)` 强制该项为半正定
- 这是一个保守近似：在压缩状态下略微低估刚度，但保证稳定性

#### 6.2.5 膜阻尼（StVK 模型）

```python
# 使用 Green 应变张量 G = (FᵀF - I)/2
G00 = 0.5 * (||f₀||² - 1)
G11 = 0.5 * (||f₁||² - 1)
G01 = 0.5 * (f₀·f₁)

# 两个独立的阻尼通道:
# Cμ  = ||G||_F     (Frobenius 范数)
# Cλ  = tr(G)       (迹)

# 阻尼力 = -kd_mu * dCμ/dt * ∂Cμ/∂x - kd_lambda * dCλ/dt * ∂Cλ/∂x
# 其中 kd_mu = μ * damping, kd_lambda = λ * damping
```

**为什么膜的阻尼用 StVK 模型而不是简单的 Rayleigh？**
- 3D tet 的 Rayleigh 阻尼直接基于弹性 Hessian（`H_damp = damping * H_elastic`）
- 膜的 Hessian 有 PSD clamp，直接用可能不准确
- StVK 阻尼独立计算两个应变度量的变化率，更物理
- 在静息态附近（G ≈ 0），阻尼自动消失

### 6.3 离散壳弯曲能量

#### 6.3.1 二面角计算

```
θ = atan2(sin(θ), cos(θ))

sin(θ) = (ê × n₁) · n₂
cos(θ) = n₁ · n₂

其中:
  ê = normalize(xₗ - xₖ)     # 边方向
  n₁ = normalize((xᵢ - xₖ) × (xₗ - xₖ))   # 三角形 1 法向
  n₂ = normalize((xⱼ - xₗ) × (xₖ - xₗ))   # 三角形 2 法向
```

#### 6.3.2 能量与力

```
E_bend = (ke/2) * (θ - θ_rest)²

∂θ/∂x = ∂sin(θ)/∂x * cos(θ) - ∂cos(θ)/∂x * sin(θ)

Force_i = -ke * (θ - θ_rest) * ∂θ/∂x_i
Hessian_i ≈ ke * ∂θ/∂x_i · ∂θ/∂x_iᵀ    (Gauss-Newton 近似)
```

**为什么使用 Gauss-Newton Hessian 近似？**
- 完整的 Hessian 包含 ∂²θ/∂x² 项，计算复杂且可能非 PSD
- Gauss-Newton 近似（忽略 ∂²θ/∂x²）总是 PSD
- 对于弯曲能量，GN 近似在实践中效果很好
- 这是标准的离散壳实现方式

### 6.4 线性弹簧

```
F_spring_v0 = +ke * (l₀ - l) / l * (x_v0 - x_v1)
F_spring_v1 = -ke * (l₀ - l) / l * (x_v0 - x_v1)

H = ke * [I - (l₀/l) * (I - d·dᵀ/l²)]

其中 d = x_v0 - x_v1, l = |d|
```

**为什么 Hessian 有这个形式？**
- 弹簧能量 E = ke/2 * (l - l₀)²
- ∂l/∂x_v0 = d/l, ∂²l/∂x_v0² = (I - d·dᵀ/l²)/l
- 代入链式法则得到上述公式
- 当 l = l₀（静息态）时，H = ke * d·dᵀ/l₀²（秩 1）

### 6.5 VBD 局部系统

```
min_Δx_i  ½ Δx_iᵀ H_i Δx_i + f_iᵀ Δx_i

H_i = m_i/dt² · I + Σ_H_elastic_i + Σ_H_contact_i + Σ_H_spring_i
f_i = m_i/dt² · (x★_i - x_i) + Σ_f_elastic_i + Σ_f_contact_i + Σ_f_spring_i

解: Δx_i = -H_i⁻¹ f_i    (3×3 矩阵求逆)
```

**为什么这个子问题是良定的？**
- 惯性项 `m/dt²·I` 保证 H_i 正定（除非 m=0，此时顶点是运动学的）
- 弹性 Hessian 是 SPD 近似的
- 接触 Hessian 是半正定的（外积形式）
- 因此 H_i 总是可逆的

### 6.6 AVBD 6×6 刚体系统

```
min_Δr  ½ Δrᵀ H Δr + bᵀ Δr

Δr = [Δx, Δθ]ᵀ         # 6D 增量
H   = [H_ll  H_alᵀ]    # 6×6 SPD
      [H_al  H_aa  ]

求解: Δr = -LDLᵀ_solve(H, b)
```

**LDLᵀ 6×6 求解器** (`ldlt6_solve`)：
- 完全展开的 6×6 LDLᵀ 分解
- 无循环、无分支（GPU 友好）
- 前向替代 → 对角求解 → 后向替代
- 约 200 行展开代码

### 6.7 摩擦模型

```python
def compute_friction(mu, normal_contact_force, T, u, eps_u):
    u_norm = length(u)
    if u_norm > 0.0:
        if u_norm > eps_u:
            f1_SF_over_x = 1.0 / u_norm       # 动摩擦
        else:
            f1_SF_over_x = (-u_norm/eps_u + 2.0) / eps_u  # 静摩擦（平滑）

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * I₂) * T^T
    else:
        force = 0, hessian = 0
```

**为什么使用 IPC 风格的平滑摩擦？**
- 标准 Coulomb 摩擦在 u=0 处不可导（静→动摩擦切换）
- IPC 的平滑近似在 |u| < eps_u 时使用二次函数过渡
- 保证 C1 连续性，使 VBD 的 Newton 型求解器收敛
- `eps_u = friction_epsilon * dt` 将速度阈值转换为位移阈值

---

## 7. 数据链路与内存模型

### 7.1 完整数据流图

```
用户代码
  │
  ├─ ModelBuilder
  │   ├─ add_particle() ─────────────────────► particle_q, particle_mass, ...
  │   ├─ add_tetrahedron() ──────────────────► tet_indices, tet_poses, tet_materials
  │   │   └─ 计算 Dm⁻¹ 和 volume
  │   ├─ add_triangle() ─────────────────────► tri_indices, tri_poses, tri_materials
  │   ├─ add_edge() ─────────────────────────► edge_indices, edge_bending_properties
  │   ├─ add_spring() ───────────────────────► spring_indices, spring_rest_length, ...
  │   ├─ color() ────────────────────────────► particle_color_groups, particle_colors
  │   └─ finalize() ─────────────────────────► Model (GPU arrays)
  │
  ├─ SolverVBD(model)
  │   ├─ 构建粒子邻接 (CSR, CPU) ────────────► particle_adjacency → GPU
  │   ├─ 构建刚体邻接 (CSR, CPU) ────────────► rigid_adjacency → GPU
  │   ├─ 初始化 AVBD 关节约束布局 ───────────► joint_constraint_start, joint_is_hard
  │   ├─ 初始化接触检测器 ───────────────────► TriMeshCollisionDetector (BVH + buffers)
  │   └─ 初始化 AVBD 接触状态缓冲区
  │
  ├─ CollisionPipeline.collide(state, contacts) ─► Contacts
  │   ├─ rigid_contact_* (刚体-刚体)
  │   └─ soft_contact_* (粒子-刚体)
  │
  └─ solver.step(state_in, state_out, control, contacts, dt)
      │
      ├─ [Phase 1: Initialize]
      │   ├─ forward_step() ──────────────► inertia[], particle_displacements[]
      │   ├─ forward_step_rigid_bodies() ──► body_inertia_q[]
      │   ├─ 碰撞检测 ────────────────────► trimesh_collision_info
      │   └─ penetration_free_truncation() ► truncation_ts[], particle_displacements[]
      │
      ├─ [Phase 2: Iterate] × N
      │   ├─ AVBD 迭代:
      │   │   ├─ accumulate_body_body_contacts ──► body_forces[], body_torques[], Hessians[]
      │   │   ├─ accumulate_body_particle_contacts ► 同上
      │   │   ├─ solve_rigid_body() ────────────► state_in.body_q[] (in-place)
      │   │   ├─ update_duals_body_body_contacts ► body_body_contact_lambda[], _penalty_k[]
      │   │   ├─ update_duals_body_particle_contacts ► body_particle_contact_penalty_k[]
      │   │   └─ update_duals_joint ────────────► joint_lambda_lin[], joint_lambda_ang[], _penalty_k[]
      │   │
      │   └─ VBD 迭代:
      │       ├─ accumulate_spring_force_and_hessian ─► particle_forces[], particle_hessians[]
      │       ├─ accumulate_self_contact_force_and_hessian ► 同上
      │       ├─ accumulate_particle_body_contact_force ► 同上
      │       ├─ solve_elasticity[_tile]() ──────────► particle_displacements[]
      │       └─ penetration_free_truncation() ──────► truncation_ts[], particle_displacements[]
      │
      └─ [Phase 3: Finalize]
          ├─ update_body_velocity() ───────► state_out.body_qd[], body_q_prev[]
          ├─ update_cable_dahl_state() ────► joint_sigma_prev[], joint_kappa_prev[]
          └─ update_velocity() ────────────► state_out.particle_qd[]
```

### 7.2 关键 Warp 数组形状参考

| 数组 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `particle_q` | `[N]` | `wp.vec3` | 粒子位置 [m] |
| `particle_qd` | `[N]` | `wp.vec3` | 粒子速度 [m/s] |
| `particle_mass` | `[N]` | `float` | 粒子质量 [kg] |
| `particle_inv_mass` | `[N]` | `float` | 逆质量 [1/kg] |
| `tet_indices` | `[M_tet, 4]` | `int32` | 四面体顶点索引 |
| `tet_poses` | `[M_tet]` | `wp.mat33` | Dm⁻¹ (3×3) |
| `tet_materials` | `[M_tet, 3]` | `float` | [k_mu, k_lambda, k_damp] |
| `tri_indices` | `[M_tri, 3]` | `int32` | 三角形顶点索引 |
| `tri_poses` | `[M_tri]` | `wp.mat22` | Dm⁻¹ (2×2) |
| `tri_materials` | `[M_tri, 5]` | `float` | [ke, ka, kd, drag, lift] |
| `edge_indices` | `[M_edge, 4]` | `int32` | [opp0, opp1, v0, v1] |
| `spring_indices` | `[2*M_spring]` | `int` | 展平顶点对 |
| `particle_color_groups` | `list[N_color]` | `wp.array[int32]` | 每颜色组的顶点索引 |
| `v_adj_tets` | `[2*Σadj_tets]` | `int` | 展平: [tet_id, v_order, ...] |
| `v_adj_tets_offsets` | `[N+1]` | `int` | CSR 偏移 |
| `inertia` | `[N]` | `wp.vec3` | 惯性目标位置 x★ |
| `particle_forces` | `[N]` | `wp.vec3` | per-vertex 力累加 |
| `particle_hessians` | `[N]` | `wp.mat33` | per-vertex Hessian 累加 |
| `particle_displacements` | `[N]` | `wp.vec3` | 当前位移 Δx |
| `truncation_ts` | `[N]` | `float` | Planar DAT 截断系数 |

### 7.3 内存占用估算

对于一个有 V 个顶点、T 个四面体、F 个三角形、E 条边的软体：

| 数据 | 大小 (bytes) | 说明 |
|------|-------------|------|
| 粒子状态 | V × 12 × 3 = 36V | q, qd, mass, inv_mass, radius, flags |
| 四面体数据 | T × (16×4 + 36 + 12) ≈ 112T | indices, Dm⁻¹, materials |
| 三角形数据 | F × (12×4 + 16 + 20) ≈ 84F | indices, Dm⁻¹, materials |
| 邻接数据 | ~8 × Σ(adj per vertex) × 4 | CSR 格式 |
| 自接触缓冲 | V × buffer_size × 8 + E × buffer_size × 8 | 预分配 |
| 求解器临时 | V × (12 + 12 + 36 + 36 + 12 + 4) ≈ 112V | forces, hessians, displacements, ... |

对于 10K 顶点、30K tet 的网格：约 10-20 MB（GPU 内存）

---

## 8. 架构设计与设计决策

### 8.1 模块分层

```
┌──────────────────────────────────────────────────────────────────┐
│                    公开 API 层 (newton/)                           │
│  newton/solvers.py      → 导出 SolverVBD                          │
│  newton/geometry.py     → 导出 TetMesh, ParticleFlags             │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────┐
│                  内部实现层 (newton/_src/)                          │
│                                                                   │
│  solvers/vbd/solver_vbd.py          → SolverVBD 主类              │
│  solvers/vbd/particle_vbd_kernels.py → 粒子 VBD GPU kernel        │
│  solvers/vbd/rigid_vbd_kernels.py    → 刚体 AVBD GPU kernel       │
│  solvers/vbd/tri_mesh_collision.py   → 三角网格自碰撞检测          │
│                                                                   │
│  sim/builder.py         → ModelBuilder (软体创建 API)              │
│  sim/model.py           → Model (数据容器)                         │
│  sim/state.py           → State (运行时状态)                       │
│  sim/contacts.py        → Contacts (接触数据)                      │
│  sim/collide.py         → CollisionPipeline (碰撞检测管道)         │
│                                                                   │
│  geometry/types.py      → TetMesh, ParticleFlags                  │
│  geometry/kernels.py    → 三角碰撞检测底层 kernel                  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 核心设计决策

#### 8.2.1 为什么选择 VBD 而不是 XPBD 或传统 FEM？

| 方面 | VBD | XPBD | 传统 FEM (Newton) |
|------|-----|------|-------------------|
| 每次迭代 | 3×3 矩阵求逆 | 标量投影 | 全局稀疏求解 |
| 收敛速度 | 快（Newton 型） | 慢（梯度型） | 最快（精确 Newton） |
| GPU 并行 | 优秀（颜色分组） | 优秀（按约束） | 差（稀疏求解器） |
| 内存 | O(N) | O(N) | O(N²) 或 O(N log N) |
| 超弹性 | 支持（NH） | 困难 | 支持 |
| 大时间步 | 稳定（隐式） | 稳定（隐式） | 稳定（隐式） |

**VBD 的核心优势**：在 GPU 上实现了接近 Newton 法的收敛速度，但避免了全局稀疏求解。

#### 8.2.2 为什么粒子-刚体耦合使用 penalty 而不是 AL？

- 粒子-刚体接触数量通常很大（每个粒子可能接触多个刚体）
- 为每个粒子-刚体接触维护 λ 状态会显著增加内存和计算开销
- Penalty 方法在粒子侧与 VBD 的 per-vertex Hessian 累加自然集成
- 通过 AVBD 的 k-update（`k_new = k_old + beta`），penalty 刚度可以自适应增长

#### 8.2.3 为什么邻接在 CPU 上构建？

- 邻接构建需要 scatter 写入和 prefix sum，GPU 上原子操作开销大
- 邻接只在求解器构造时构建一次，不需要每帧更新
- CPU 上的 NumPy `cumsum` 比 GPU prefix sum 更简单可靠
- 构建完成后通过 `.to(device)` 上传到 GPU

#### 8.2.4 为什么使用 `wp.func` 而不是 `wp.kernel` 做能量计算？

- `evaluate_volumetric_neo_hookean_force_and_hessian` 等是 `@wp.func`（device function）
- 它们被 `solve_elasticity` kernel 在每个顶点的循环中调用
- 作为 device function，它们可以内联到调用者 kernel 中
- 避免了 kernel launch 的开销（每个顶点 launch 一个 kernel 是不可行的）

#### 8.2.5 为什么碰撞检测在初始化前执行？

- 初始化步骤（前向积分）会产生初始位移
- 如果先积分再检测碰撞，初始位移可能已经导致穿透
- 先检测碰撞可以获得碰撞前的参考位置，用于 Planar DAT 截断
- 截断在初始化后立即执行，确保初始位移也不穿透

#### 8.2.6 为什么关节硬/软模式可以按 slot 控制？

- 不同类型的约束对精度要求不同
- 结构约束（如 ball joint 的线性约束）通常需要高精度 → hard 模式
- Cable 的 stretch/bend 可以容忍一些误差 → soft 模式（默认）
- 驱动/限位约束总是 soft（不需要 λ 累积）
- 通过 `vbd:joint_is_hard` custom attribute 或 `set_joint_constraint_mode()` 控制

#### 8.2.7 为什么 AVBD 的 penalty k 有衰减（gamma）？

```python
rigid_avbd_gamma = 0.999  # 每步衰减因子
k_new = k_old * gamma      # 在每步开始时应用
```

- 在持久接触中，λ 会不断累积来补偿有限的 k
- 如果 k 不衰减，k 和 λ 都会增长，导致过大的力
- gamma < 1 允许 k 逐渐减小，让 λ 承担更多的约束力
- 这是一个"信任 λ"的策略：随着时间推移，更多地依赖累积的 λ 而不是当前的 k

---

## 9. 关键代码路径索引

### 9.1 核心文件

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `newton/_src/solvers/vbd/solver_vbd.py` | ~2800+ | SolverVBD 主类、流程编排、邻接构建、碰撞检测调度 |
| `newton/_src/solvers/vbd/particle_vbd_kernels.py` | ~3400+ | 粒子 VBD kernel：前向积分、弹性求解、NH 能量、接触力、Planar DAT |
| `newton/_src/solvers/vbd/rigid_vbd_kernels.py` | ~3800+ | 刚体 AVBD kernel：LDLᵀ 求解器、关节约束、接触力、Dahl 摩擦 |
| `newton/_src/solvers/vbd/tri_mesh_collision.py` | ~450+ | 三角网格碰撞检测器、BVH 管理 |
| `newton/_src/sim/builder.py` | ~10000+ | ModelBuilder：所有 `add_*` 方法 |

### 9.2 关键函数/方法索引

**SolverVBD** (`solver_vbd.py`):
| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__()` | 196 | 求解器构造，初始化所有子系统 |
| `step()` | 1575 | 顶层仿真步进 |
| `_initialize_particles()` | 1731 | 粒子前向积分 + 碰撞检测 + 截断 |
| `_initialize_rigid_bodies()` | 1768 | 刚体前向积分 + 接触历史恢复 |
| `_solve_particle_iteration()` | 2151 | VBD 单次 Gauss-Seidel 迭代 |
| `_solve_rigid_body_iteration()` | 2339 | AVBD 单次 Gauss-Seidel 迭代 |
| `_finalize_particles()` | 2761 | BDF1 速度更新 |
| `_finalize_rigid_bodies()` | 2774 | 刚体速度 + 粘滞死区 + Dahl 状态 |
| `_collision_detection_penetration_free()` | 2832 | 自碰撞检测调度 |
| `_penetration_free_truncation()` | 1669 | Planar DAT 截断调度 |
| `_compute_particle_force_element_adjacency()` | 1177 | 粒子 CSR 邻接构建 |
| `_compute_particle_contact_filtering_list()` | 1339 | 拓扑过滤列表构建 |
| `set_joint_constraint_mode()` | 1493 | 运行时关节硬/软模式切换 |
| `rebuild_bvh()` | 2850 | BVH 重建 |

**粒子 VBD Kernels** (`particle_vbd_kernels.py`):
| 函数 | 行号 | 功能 |
|------|------|------|
| `forward_step()` | 1783 | 前向积分 kernel |
| `solve_elasticity()` | 3136 | 非 tiled VBD 求解 kernel |
| `solve_elasticity_tile()` | 2971 | Tiled VBD 求解 kernel (CUDA) |
| `evaluate_volumetric_neo_hookean_force_and_hessian()` | 335 | 3D NH 能量/Hessian |
| `evaluate_neo_hookean_membrane_force_hessian()` | 866 | 2D NH 膜能量/Hessian |
| `evaluate_dihedral_angle_based_bending_force_hessian()` | 1058 | 弯曲能量/Hessian |
| `evaluate_spring_force_and_hessian()` | 2259 | 弹簧力/Hessian |
| `evaluate_self_contact_force_norm()` | 1194 | C2 接触屏障函数 |
| `evaluate_vertex_triangle_collision_force_hessian_4_vertices()` | 1600 | 顶点-三角接触（4 顶点） |
| `evaluate_edge_edge_contact_2_vertices()` | 1372 | 边-边接触（2 边 × 2 顶点） |
| `compute_friction()` | 1749 | IPC 风格摩擦 |
| `accumulate_spring_force_and_hessian()` | 2347 | 弹簧力/Hessian 累加 |
| `accumulate_self_contact_force_and_hessian()` | 1931 | 自接触力/Hessian 累加 |
| `apply_planar_truncation_parallel_by_collision()` | 2745 | Planar DAT 截断 |
| `compute_cofactor()` | 524 | 余子式直接计算 |
| `assemble_tet_vertex_force_and_hessian()` | 155 | 组装 per-vertex 力/Hessian |

**刚体 AVBD Kernels** (`rigid_vbd_kernels.py`):
| 函数 | 行号 | 功能 |
|------|------|------|
| `ldlt6_solve()` | 77 | 6×6 LDLᵀ 直接求解器 |
| `solve_rigid_body()` | 2948 | 刚体 per-body 求解 kernel |
| `evaluate_angular_constraint_force_hessian()` | 367 | 角度约束力/Hessian |
| `compute_kappa_and_jacobian()` | 236 | 曲率向量 + Jacobian |
| `_update_dual_vec3()` | 334 | AVBD 双变量更新 |
| `compute_rigid_contact_forces()` | 2666 | 刚体接触力计算 |
| `update_duals_joint()` | 3208 | 关节双变量更新 kernel |
| `compute_projected_isotropic_friction()` | 892 | 刚体接触摩擦 |

---

## 10. 优缺点分析

### 10.1 优点

#### 1. GPU 原生并行
- 全部计算在 GPU 上执行（除初始化时的 CPU 端邻接构建）
- Gauss-Seidel 着色使每次迭代可最大化 GPU 利用率
- 对大型网格（数万顶点）有优秀扩展性
- Tiled solve 进一步利用 CUDA 的 warp 级并行

#### 2. 无条件稳定性
- 隐式欧拉格式 (BDF1) 提供大时间步稳定性
- Per-vertex 局部 Hessian 是 SPD 的，保证局部子问题可解
- Planar DAT 提供无穿透保证（比传统 penalty 方法更可靠）

#### 3. 统一框架
- 同时处理软体、布料和刚体
- 双向耦合（粒子↔刚体互动力）
- 统一的 AVBD 双变量更新框架

#### 4. 接触处理鲁棒
- C2 连续屏障函数（log + quadratic + quadratic extension）
- Planar DAT 截断防止穿透
- 支持自接触（vertex-triangle + edge-edge）
- IPC 风格平滑摩擦

#### 5. 增广 Lagrangian 收敛
- AVBD 硬约束模式具有 AL 收敛性
- Contact history warm-start 显著加速收敛
- 关节约束的硬/软模式可按 slot 控制

#### 6. 无全局线性求解
- 每个顶点只解 3×3 系统（每个刚体 6×6）
- 无需组装或分解大型稀疏矩阵
- 内存占用与顶点数线性关系

#### 7. 可微分
- 通过 Warp 的 `wp.Tape()` 支持自动微分
- 有可微分仿真示例（`example_diffsim_soft_body.py`）

#### 8. 材料灵活性
- 支持 per-element 材料（tet/triangle）
- Stable Neo-Hookean 能量处理大变形
- 多种阻尼模型（Rayleigh, StVK 膜阻尼, Dahl 摩擦）

### 10.2 缺点

#### 1. 迭代型求解器局限性
- 需要足够多的迭代次数才能收敛（典型 10-30 次）
- 高刚度比（stiffness ratio）可能导致收敛慢
- 没有全局误差估计或自适应步长

#### 2. 着色开销
- 着色必须在求解前完成
- 复杂网格需要更多颜色，降低了每次迭代的并行度
- 颜色组数量随网格复杂度增加

#### 3. 内存使用
- 自接触缓冲区需要预分配
- 缓冲区太小：遗漏碰撞 → 穿透
- 缓冲区太大：浪费 GPU 内存和带宽
- 大量 CSR 邻接数据 + AVBD 接触状态

#### 4. 参数敏感性
- AVBD 的 alpha、beta、gamma 参数需要调优
- Penalty stiffness 的种子值和上限需要合理设置
- 太小的刚度 → 约束违反大；太大的刚度 → 收敛慢或振荡

#### 5. 仅支持四面体 + 三角形膜
- 不支持六面体 FEM（虽然可用 hex→5 tets 剖分）
- 不支持非线性弯曲模型（仅 Grinspun 离散壳）
- 不支持塑性变形

#### 6. 接触限制
- 自接触仅检测三角网格
- 粒子-刚体接触使用 penalty（无 AL 选项）
- 运动学碰撞体仅支持单向耦合

#### 7. 关节限制
- 不支持 DISTANCE 关节
- 不支持 armature、friction、effort/velocity limit
- Cable 关节的 stretch/bend 默认是软约束

#### 8. CPU 性能
- Warp 的 CPU 后端不支持 tiled solve
- CPU 上 Gauss-Seidel 着色并行度有限

#### 9. 无全局收敛保证
- Gauss-Seidel 全局收敛需要 Hessian 的某些条件
- 虽然实践中对图形学应用收敛良好，但缺乏严格理论保证

### 10.3 与其他求解器对比

| 特性 | VBD (Newton) | XPBD | Semi-Implicit |
|------|--------------|------|---------------|
| 求解方式 | Per-vertex 3×3 Newton 步 | Per-constraint 标量投影 | 全局线性系统 |
| 收敛速度 | 快（Newton 型） | 慢（梯度型） | 最快（精确 Newton） |
| GPU 并行 | 优秀（颜色分组） | 优秀（按约束） | 一般（需稀疏求解器） |
| 约束处理 | 能量 + Penalty/AL | Lagrangian 乘子 | 能量 |
| 无穿透 | Planar DAT | 无特殊机制 | 无特殊机制 |
| 大变形 | 好（NH 能量） | 中 | 差（线性化假设） |
| 内存 | O(N) | O(N) | O(N²) 或 O(N log N) |
| 适合规模 | 大型 | 大型 | 中小型 |

---

## 11. 附录：配置参数与调优指南

### 11.1 SolverVBD 完整参数

```python
SolverVBD(
    model,
    # === 通用参数 ===
    iterations=10,                        # VBD/AVBD 迭代次数
    friction_epsilon=1e-2,               # 摩擦平滑阈值 [m]

    # === 粒子参数 ===
    particle_enable_self_contact=False,    # 启用自接触
    particle_self_contact_radius=0.2,     # 自接触检测半径 [m]
    particle_self_contact_margin=0.2,     # 自接触 margin [m]（应 > radius）
    particle_conservative_bound_relaxation=0.85,  # 保守边界松弛
    particle_vertex_contact_buffer_size=32,   # 每顶点碰撞缓冲大小
    particle_edge_contact_buffer_size=64,     # 每边碰撞缓冲大小
    particle_collision_detection_interval=0,  # 碰撞检测间隔
    particle_edge_parallel_epsilon=1e-5,      # 平行边检测阈值
    particle_enable_tile_solve=True,          # CUDA Tiled solve
    particle_topological_contact_filter_threshold=2,  # 拓扑过滤环数
    particle_rest_shape_contact_exclusion_radius=0.0, # 静息距离过滤

    # === 刚体 AVBD 参数 ===
    rigid_avbd_alpha=0.95,               # C0 稳定化强度 [0,1]
    rigid_avbd_beta=0.0,                 # Penalty 递增率 (0=固定k)
    rigid_avbd_gamma=0.999,              # Penalty 每步衰减
    rigid_contact_hard=True,             # 硬接触模式
    rigid_contact_history=False,         # 接触历史 warm-start
    rigid_contact_stick_motion_eps=1e-4, # 粘滞检测阈值 [m]
    rigid_contact_stick_freeze_translation_eps=1e-4,  # 平移死区 [m]
    rigid_contact_stick_freeze_angular_eps=1e-4,      # 旋转死区 [rad]
    rigid_contact_k_start=1e2,           # 接触惩罚刚度种子
    rigid_body_contact_buffer_size=64,   # 每刚体接触缓冲大小
    rigid_body_particle_contact_buffer_size=256,  # 每刚体粒子接触缓冲
    rigid_joint_linear_ke=1e5,           # 关节线性惩罚刚度上限
    rigid_joint_angular_ke=1e5,          # 关节角度惩罚刚度上限
    rigid_joint_linear_k_start=1e2,      # 线性惩罚种子
    rigid_joint_angular_k_start=1e1,     # 角度惩罚种子
    rigid_joint_linear_kd=0.0,           # 线性 Rayleigh 阻尼
    rigid_joint_angular_kd=0.0,          # 角度 Rayleigh 阻尼
)
```

### 11.2 调优建议

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 软体太"软" | k_mu/k_lambda 太小 | 增大材料刚度 |
| 软体太"硬" | k_mu/k_lambda 太大 | 减小材料刚度，或增加迭代次数 |
| 体积损失 | 迭代次数不够 | 增加 iterations (15-30) |
| 穿透 | 自接触未启用或 margin 太小 | 启用自接触，margin = 1.5-2× radius |
| 碰撞遗漏 | 缓冲太小 | 增大 vertex/edge contact buffer size |
| 关节分离 | 关节刚度不够 | 增大 rigid_joint_linear_ke，或启用 hard 模式 |
| 刚体接触穿透 | k_start 太小或 beta=0 | 增大 k_start，或设置 beta>0 |
| 性能差 | 迭代太多或网格太大 | 减少 iterations，或用 tiled solve |
| 收敛慢 | 刚度比大 | 增加 iterations，调整 AVBD alpha/beta |
| 布料太"脆" | 弯曲刚度太大 | 减小 edge_ke |
| 布料太"软" | 弯曲刚度太小 | 增大 edge_ke |
| 自接触性能差 | 过滤不够 | 增大 topological_contact_filter_threshold |
