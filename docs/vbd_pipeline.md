# Newton VBD 软体/布料算法完整链路流程

> 本文档梳理 Newton 物理引擎中 VBD (Vertex Block Descent) 算法在**粒子/软体/布料**场景下的完整仿真链路，包含公式推导、代码实现、模块职责和数据传输细节。刚体 AVBD 部分仅做简要提及。

**核心论文**: Anka He Chen, Ziheng Liu, Yin Yang, Cem Yuksel. *Vertex Block Descent*. ACM Trans. Graph. 43, 4, Article 116 (July 2024). https://doi.org/10.1145/3658179

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [模型构建阶段 (ModelBuilder)](#2-模型构建阶段-modelbuilder)
3. [图着色 (Graph Coloring)](#3-图着色-graph-coloring)
4. [Model 静态数据结构](#4-model-静态数据结构)
5. [State 动态数据结构](#5-state-动态数据结构)
6. [SolverVBD 初始化](#6-solvervbd-初始化)
7. [邻接表构建 (CSR Adjacency)](#7-邻接表构建-csr-adjacency)
8. [自碰撞检测模块](#8-自碰撞检测模块)
9. [step() 主循环详解](#9-step-主循环详解)
10. [Phase 1: 初始化 — 前向积分](#10-phase-1-初始化--前向积分)
11. [Phase 2: 迭代求解 — 核心算法](#11-phase-2-迭代求解--核心算法)
12. [弹性力/海森矩阵计算](#12-弹性力海森矩阵计算)
13. [自接触力/海森矩阵计算](#13-自接触力海森矩阵计算)
14. [弹簧力/海森矩阵计算](#14-弹簧力海森矩阵计算)
15. [粒子-刚体接触力](#15-粒子-刚体接触力)
16. [穿透自由截断 (Planar DAT)](#16-穿透自由截断-planar-dat)
17. [Phase 3: 终结 — 速度更新](#17-phase-3-终结--速度更新)
18. [完整数据流图](#18-完整数据流图)
19. [关键文件索引](#19-关键文件索引)

---

## 1. 整体架构概览

Newton VBD 仿真采用三层架构：

```
┌─────────────────────────────────────────────────┐
│  ModelBuilder (增量构建)                         │
│  add_cloth_grid / add_cloth_mesh / add_soft_grid │
│  add_soft_mesh / add_spring / color()            │
│  finalize() → Model                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Model (静态拓扑 + 材料参数, Warp device arrays) │
│  State (动态位置/速度, 由 Model.state() 创建)    │
│  Control / Contacts                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  SolverVBD (求解器实例)                          │
│  __init__: 构建邻接表 + 分配内部状态              │
│  step(): 3阶段求解 (初始化 → 迭代 → 终结)        │
└─────────────────────────────────────────────────┘
```

仿真主循环：
```python
state_in, state_out = model.state(), model.state()
control = model.control()
contacts = model.contacts()

for step in range(num_steps):
    model.collide(state_in, contacts)       # 碰撞检测
    solver.step(state_in, state_out, control, contacts, dt)
    state_in, state_out = state_out, state_in  # 交换状态
```

---

## 2. 模型构建阶段 (ModelBuilder)

**文件**: `newton/_src/sim/builder.py`

### 2.1 布料网格构建: `add_cloth_grid()` / `add_cloth_mesh()`

`add_cloth_grid()` 创建规则矩形网格，内部调用 `add_cloth_mesh()`。

`add_cloth_mesh()` 的数据流：

1. **创建粒子**: 调用 `add_particles()`，初始质量为 0（后续由三角形面积分配）
2. **创建三角形 FEM 元素**: 调用 `add_triangles()`
   - 从当前顶点位置计算 2D 参考构型逆矩阵 `inv_D`
   - 计算参考面积 `area = det(D) / 2`
   - 存储到 `tri_indices`, `tri_poses` (= inv_D), `tri_materials` = (ke, ka, kd, drag, lift), `tri_areas`
3. **质量分配**: 对每个三角形，将 `density * area / 3` 分配到 3 个顶点
4. **创建弯曲边**: 使用 `MeshAdjacency` 查找所有内部边，调用 `add_edges()`
   - `edge_indices` = [o0, o1, v0, v1]（对侧顶点 + 边顶点）
   - 弯曲刚度和阻尼
5. **可选弹簧**: 若 `add_springs=True`，沿所有网格边创建距离弹簧

### 2.2 软体网格构建: `add_soft_grid()` / `add_soft_mesh()`

`add_soft_grid()` 创建 3D 四面体网格，每个六面体分解为 5 个四面体。

`add_soft_mesh()` 的数据流：

1. **创建粒子**: 每个顶点一个粒子，初始质量为 0
2. **创建四面体 FEM 元素**: 调用 `add_tetrahedron()` per tet
   - 计算 `Dm = [q-p, r-p, s-p]`，`volume = det(Dm) / 6`
   - 存储 `tet_indices`, `tet_poses` (= inv(Dm)), `tet_materials` = (k_mu, k_lambda, k_damp)
   - 将 `density * volume / 4` 质量分配到 4 个顶点
3. **表面三角形**: 从四面体连接关系提取边界面
4. **表面弯曲边**: 可选，用于碰撞鲁棒性

### 2.3 弹簧构建: `add_spring()`

- 从当前粒子位置计算静止长度
- 存储刚度 `spring_stiffness` [N/m] 和阻尼 `spring_damping` [N·s/m]

### 2.4 finalize(): Builder → Model 转换

将 Python 列表转换为 Warp device arrays：
- `particle_inv_mass = 1.0 / particle_mass`（0 质量对应 0 逆质量，标记为运动学粒子）
- 所有拓扑数组传输到 GPU: `tri_indices`, `tri_poses`, `tri_areas`, `tri_materials`, `edge_indices`, `edge_rest_angle`, `edge_rest_length`, `edge_bending_properties`, `tet_indices`, `tet_poses`, `tet_materials`, `spring_indices`, ...
- 执行图着色（若已调用 `color()`）

---

## 3. 图着色 (Graph Coloring)

**文件**: `newton/_src/sim/graph_coloring.py`

VBD 使用 **Gauss-Seidel 迭代**，需要将粒子分为独立颜色组，同色粒子可并行更新。

### 3.1 算法流程

```
builder.color()
  → construct_particle_graph()    # 从所有力元素构建交互图
    → construct_trimesh_graph_edges()  # 三角形边 + 弯曲边
    → construct_tetmesh_graph_edges()  # 四面体边
    # 合并弹簧边
  → color_graph()                  # MCS 或 Greedy 着色
  → convert_to_color_groups()      # per-particle colors → 分组数组
  → combine_independent_particle_coloring()  # 合并独立系统
```

### 3.2 交互图构建

- **三角形**: 三角形的 3 个顶点两两连边
- **弯曲边**: 边的 4 个顶点 (o0, o1, v0, v1) 两两连边
- **四面体**: 四面体的 4 个顶点两两连边
- **弹簧**: 弹簧的 2 个顶点连边

### 3.3 着色结果

- `particle_colors: wp.array[int32]` — 每个粒子的颜色编号
- `particle_color_groups: list[wp.array[int32]]` — 每种颜色包含的粒子索引列表

同色粒子无直接力元素交互，可安全并行求解。

---

## 4. Model 静态数据结构

**文件**: `newton/_src/sim/model.py`

Model 是纯数据容器，所有数组均为 Warp device arrays。

### 4.1 粒子数据

| 属性 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `particle_q` | `wp.array[wp.vec3]` | `[particle_count]` | 初始/参考位置 [m] |
| `particle_qd` | `wp.array[wp.vec3]` | `[particle_count]` | 初始速度 [m/s] |
| `particle_mass` | `wp.array[float]` | `[particle_count]` | 质量 [kg] |
| `particle_inv_mass` | `wp.array[float]` | `[particle_count]` | 逆质量 [1/kg] |
| `particle_radius` | `wp.array[float]` | `[particle_count]` | 接触半径 [m] |
| `particle_flags` | `wp.array[int32]` | `[particle_count]` | Active/Kinematic 标志 |
| `particle_color_groups` | `list[wp.array[int32]]` | — | 着色分组 |
| `particle_colors` | `wp.array[int32]` | `[particle_count]` | 颜色编号 |

### 4.2 三角形 (布料 FEM)

| 属性 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `tri_indices` | `wp.array[int32]` | `[tri_count * 3]` | 顶点索引 |
| `tri_poses` | `wp.array[wp.mat22]` | `[tri_count]` | 参考构型逆矩阵 Dm_inv |
| `tri_areas` | `wp.array[float]` | `[tri_count]` | 参考面积 [m²] |
| `tri_materials` | `wp.array2d[float]` | `[tri_count, 5]` | [k_mu, k_lambda, k_damp, k_drag, k_lift] |

### 4.3 弯曲边

| 属性 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `edge_indices` | `wp.array[int32]` | `[edge_count * 4]` | [o0, o1, v0, v1] |
| `edge_rest_angle` | `wp.array[float]` | `[edge_count]` | 参考二面角 [rad] |
| `edge_rest_length` | `wp.array[float]` | `[edge_count]` | 参考长度 [m] |
| `edge_bending_properties` | `wp.array2d[float]` | `[edge_count, 2]` | [stiffness N·m/rad, damping N·s] |

### 4.4 四面体 (软体 FEM)

| 属性 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `tet_indices` | `wp.array[int32]` | `[tet_count * 4]` | 顶点索引 |
| `tet_poses` | `wp.array[wp.mat33]` | `[tet_count]` | 参考构型逆矩阵 Dm_inv |
| `tet_materials` | `wp.array2d[float]` | `[tet_count, 3]` | [k_mu, k_lambda, k_damp] |

### 4.5 弹簧

| 属性 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `spring_indices` | `wp.array[int32]` | `[spring_count * 2]` | 粒子对索引 |
| `spring_rest_length` | `wp.array[float]` | `[spring_count]` | 参考长度 [m] |
| `spring_stiffness` | `wp.array[float]` | `[spring_count]` | 刚度 [N/m] |
| `spring_damping` | `wp.array[float]` | `[spring_count]` | 阻尼 [N·s/m] |

---

## 5. State 动态数据结构

**文件**: `newton/_src/sim/state.py`

State 保存每个时间步的动态量，由 `Model.state()` 创建：

| 属性 | 类型 | 说明 |
|------|------|------|
| `particle_q` | `wp.array[wp.vec3]` | 粒子位置 |
| `particle_qd` | `wp.array[wp.vec3]` | 粒子速度 |
| `particle_f` | `wp.array[wp.vec3]` | 粒子外力 |
| `body_q` | `wp.array[wp.transform]` | 刚体位姿 |
| `body_qd` | `wp.array[wp.spatial_vector]` | 刚体速度 |

---

## 6. SolverVBD 初始化

**文件**: `newton/_src/solvers/vbd/solver_vbd.py`, `SolverVBD.__init__()` (line 159)

### 6.1 构造函数参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `iterations` | 10 | 每步 VBD 迭代次数 |
| `friction_epsilon` | 1e-2 | 摩擦平滑阈值 |
| `particle_enable_self_contact` | False | 启用自接触检测 |
| `particle_self_contact_radius` | 0.2 | 自接触交互距离 |
| `particle_self_contact_margin` | 0.2 | 自接触检测距离 (≥ radius) |
| `particle_conservative_bound_relaxation` | 0.85 | 穿透自由投影松弛因子 |
| `particle_enable_tile_solve` | True | GPU tile 加速 |
| `particle_topological_contact_filter_threshold` | 2 | N-ring 拓扑过滤 |

### 6.2 初始化流程

```python
SolverVBD(model, ...)
  → super().__init__(model)                    # 存储模型引用
  → _init_particle_system(model, ...)           # 粒子 VBD 状态初始化
  → _init_rigid_system(model, ...)              # 刚体 AVBD 状态初始化
```

### 6.3 `_init_particle_system()` 分配的内部状态

| 属性 | 类型 | 用途 |
|------|------|------|
| `particle_q_prev` | `wp.array[wp.vec3]` | 上一步位置（用于速度计算） |
| `inertia` | `wp.array[wp.vec3]` | 惯性目标位置（前向积分后的位置） |
| `particle_adjacency` | `ParticleForceElementAdjacencyInfo` | CSR 邻接结构 |
| `particle_q_rest` | `wp.array[wp.vec3]` | 参考构型位置 |
| `particle_forces` | `wp.array[wp.vec3]` | 每顶点累积力 |
| `particle_hessians` | `wp.array[wp.mat33]` | 每顶点 3×3 海森块 |
| `particle_displacements` | `wp.array[wp.vec3]` | 位置位移（用于截断） |
| `truncation_ts` | `wp.array[float]` | 每顶点截断缩放因子 |
| `pos_prev_collision_detection` | `wp.array[wp.vec3]` | 上次碰撞检测时的位置 |

若启用自接触，额外分配：
- `trimesh_collision_detector`: `TriMeshCollisionDetector`
- `particle_conservative_bounds`: `wp.array[float]`

---

## 7. 邻接表构建 (CSR Adjacency)

**文件**: `solver_vbd.py`, `_compute_particle_force_element_adjacency()` (line 1036)

### 7.1 ParticleForceElementAdjacencyInfo 结构

```python
@wp.struct
class ParticleForceElementAdjacencyInfo:
    v_adj_faces: wp.array[int]          # 展平的 [face_id, vertex_order] 对
    v_adj_faces_offsets: wp.array[int]  # CSR 偏移, 大小 particle_count+1

    v_adj_edges: wp.array[int]          # 展平的 [edge_id, vertex_order] 对
    v_adj_edges_offsets: wp.array[int]  # CSR 偏移

    v_adj_springs: wp.array[int]        # 展平的 spring_id
    v_adj_springs_offsets: wp.array[int]

    v_adj_tets: wp.array[int]          # 展平的 [tet_id, vertex_order] 对
    v_adj_tets_offsets: wp.array[int]
```

### 7.2 构建算法（两遍扫描）

对每种力元素（面/边/弹簧/四面体）：

1. **计数遍**: `_count_num_adjacent_*` kernel 统计每个顶点的相邻元素数
2. **前缀和**: 计算 CSR 偏移 `offsets[i+1] = offsets[i] + 2 * count[i]`
3. **填充遍**: `_fill_adjacent_*` kernel 填充展平数组，存储 `[element_id, vertex_order]`

### 7.3 vertex_order 的含义

- **三角形**: v0→0, v1→1, v2→2（三角形在 `tri_indices` 中的顺序）
- **弯曲边**: o0→0, o1→1, v0→2, v1→3（`edge_indices` 中的顺序）
- **四面体**: v0→0, v1→1, v2→2, v3→3（`tet_indices` 中的顺序）

vertex_order 用于在力/海森计算中选取当前顶点对应的偏导数方向。

### 7.4 访问接口

```python
# 获取顶点 v 的相邻面数
num = (adjacency.v_adj_faces_offsets[v+1] - adjacency.v_adj_faces_offsets[v]) >> 1

# 获取第 i 个相邻面的 (face_id, vertex_order)
offset = adjacency.v_adj_faces_offsets[v]
face_id = adjacency.v_adj_faces[offset + i * 2]
vertex_order = adjacency.v_adj_faces[offset + i * 2 + 1]
```

---

## 8. 自碰撞检测模块

**文件**: `newton/_src/solvers/vbd/tri_mesh_collision.py`

### 8.1 TriMeshCollisionInfo 结构

```python
@wp.struct
class TriMeshCollisionInfo:
    # 顶点-三角形碰撞
    vertex_colliding_triangles: wp.array[int32]       # 展平的 [vertex_id, tri_id] 对
    vertex_colliding_triangles_offsets: wp.array[int32]
    vertex_colliding_triangles_buffer_sizes: wp.array[int32]
    vertex_colliding_triangles_count: wp.array[int32]
    vertex_colliding_triangles_min_dist: wp.array[float]

    # 三角形-顶点碰撞（反向映射）
    triangle_colliding_vertices: ...

    # 边-边碰撞
    edge_colliding_edges: wp.array[int32]             # 展平的 [edge1_id, edge2_id] 对
    edge_colliding_edges_offsets: ...
    edge_colliding_edges_buffer_sizes: ...
    edge_colliding_edges_count: ...
    edge_colliding_edges_min_dist: ...
```

### 8.2 TriMeshCollisionDetector

Python 类管理 BVH 构建和碰撞检测：

| 方法 | 功能 |
|------|------|
| `refit()` | 用新位置重拟合 BVH |
| `vertex_triangle_collision_detection()` | 检测顶点-三角形自接触 |
| `edge_edge_collision_detection()` | 检测边-边自接触 |
| `set_collision_filter_list()` | 设置拓扑碰撞过滤列表 |

### 8.3 拓扑过滤

避免拓扑邻近的网格元素产生虚假自接触：

- `build_vertex_n_ring_tris_collision_filter()`: N-ring 顶点-三角形过滤
- `build_edge_n_ring_edge_collision_filter()`: N-ring 边-边过滤
- 默认 N=2，即排除 2-ring 以内的候选对

---

## 9. step() 主循环详解

**文件**: `solver_vbd.py`, `step()` (line 1338)

```python
def step(self, state_in, state_out, control, contacts, dt):
    # Phase 1: 初始化
    self._initialize_rigid_bodies(state_in, control, contacts, dt, update_rigid_history)
    self._initialize_particles(state_in, state_out, dt)

    # Phase 2: 迭代
    for iter_num in range(self.iterations):
        self._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
        self._solve_particle_iteration(state_in, state_out, contacts, dt, iter_num)

    # Phase 3: 终结
    self._finalize_rigid_bodies(state_out, dt)
    self._finalize_particles(state_out, dt)
```

---

## 10. Phase 1: 初始化 — 前向积分

**文件**: `solver_vbd.py`, `_initialize_particles()` (line 1445)

### 10.1 碰撞检测（若启用自接触）

```python
if self.particle_enable_self_contact:
    self._collision_detection_penetration_free(state_in)
```

- 重拟合 BVH
- 执行顶点-三角形碰撞检测
- 执行边-边碰撞检测
- 计算保守界 `particle_conservative_bounds`

### 10.2 前向积分: `forward_step` kernel

**文件**: `particle_vbd_kernels.py`, line 1779

```
对每个粒子 p:
  particle_q_prev[p] = pos[p]                    # 保存当前位置
  if not ACTIVE or inv_mass == 0:                 # 运动学粒子
      inertia[p] = pos[p]
      return
  vel_new = vel[p] + (gravity + external_force[p] * inv_mass[p]) * dt
  inertia[p] = pos[p] + vel_new * dt              # 惯性目标位置
  displacements[p] = vel_new * dt                  # 初始位移
```

**公式**:

$$\mathbf{v}^{n+1/2} = \mathbf{v}^n + (\mathbf{g} + \mathbf{f}_{ext} / m) \cdot \Delta t$$

$$\mathbf{x}^{inertia} = \mathbf{x}^n + \mathbf{v}^{n+1/2} \cdot \Delta t$$

### 10.3 穿透自由截断

```python
self._penetration_free_truncation(state_in.particle_q)
```

对初始位移进行截断，确保起始位置无穿透。详见 [第16节](#16-穿透自由截断-planar-dat)。

---

## 11. Phase 2: 迭代求解 — 核心算法

**文件**: `solver_vbd.py`, `_solve_particle_iteration()` (line 1704)

### 11.1 算法伪代码

```
对每次迭代 iter = 0, 1, ..., iterations-1:
    # 可选：更新碰撞检测
    if self_contact and collision_detection_interval:
        _collision_detection_penetration_free()

    # 清零力和海森
    particle_forces.zero_()
    particle_hessians.zero_()

    # 按颜色组遍历
    for color in range(num_colors):
        # 1. 累积粒子-刚体接触力/海森
        accumulate_particle_body_contact_force_and_hessian(color)

        # 2. 累积弹簧力/海森
        accumulate_spring_force_and_hessian(color)

        # 3. 累积自接触力/海森
        accumulate_self_contact_force_and_hessian(color)

        # 4. 求解弹性（每顶点 3×3 系统）
        solve_elasticity(color)  或  solve_elasticity_tile(color)

        # 5. 穿透自由截断
        _penetration_free_truncation()
```

### 11.2 VBD 核心公式

VBD 将隐式时间积分转化为逐顶点最小化问题。对每个顶点 $i$：

**总能量**:
$$E_i(\mathbf{x}_i) = \underbrace{\frac{m_i}{2\Delta t^2} \|\mathbf{x}_i - \mathbf{x}_i^{inertia}\|^2}_{\text{惯性项}} + \sum_{e \in \mathcal{N}(i)} E_e(\mathbf{x}_i)$$

其中 $\mathcal{N}(i)$ 是顶点 $i$ 的相邻力元素集合。

**牛顿步**:
$$\mathbf{H}_i \cdot \Delta\mathbf{x}_i = -\mathbf{f}_i$$

其中：
$$\mathbf{f}_i = \frac{m_i}{\Delta t^2}(\mathbf{x}_i - \mathbf{x}_i^{inertia}) + \sum_{e \in \mathcal{N}(i)} \frac{\partial E_e}{\partial \mathbf{x}_i}$$

$$\mathbf{H}_i = \frac{m_i}{\Delta t^2} \mathbf{I}_{3\times3} + \sum_{e \in \mathcal{N}(i)} \frac{\partial^2 E_e}{\partial \mathbf{x}_i^2}$$

**位置更新**:
$$\mathbf{x}_i^{new} = \mathbf{x}_i + \Delta\mathbf{x}_i = \mathbf{x}_i - \mathbf{H}_i^{-1} \mathbf{f}_i$$

### 11.3 `solve_elasticity` kernel 实现

**文件**: `particle_vbd_kernels.py`, line 3134

```python
@wp.kernel
def solve_elasticity(dt, particle_ids_in_color, pos_prev, pos, mass, inertia, ...):
    particle_index = particle_ids_in_color[tid]

    # 1. 惯性力和海森
    f = mass[p] * (inertia[p] - pos[p]) / dt²
    h = mass[p] / dt² * I₃ₓ₃

    # 2. 遍历相邻三角形 → StVK 力/海森
    for adj_tri in adjacency.faces(p):
        f_tri, h_tri = evaluate_stvk_force_hessian(...)
        f += f_tri;  h += h_tri

    # 3. 遍历相邻弯曲边 → 弯曲力/海森
    for adj_edge in adjacency.edges(p):
        f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(...)
        f += f_edge;  h += h_edge

    # 4. 遍历相邻四面体 → Neo-Hookean 力/海森
    for adj_tet in adjacency.tets(p):
        f_tet, h_tet = evaluate_volumetric_neo_hookean_force_and_hessian(...)
        f += f_tet;  h += h_tet

    # 5. 加上外部累积的力/海森（接触、弹簧等）
    h += particle_hessians[p]
    f += particle_forces[p]

    # 6. 求解 3×3 线性系统
    if |det(h)| > 1e-8:
        displacement = h⁻¹ * f
        pos[p] += displacement
```

### 11.4 Tile 加速版: `solve_elasticity_tile`

**文件**: `particle_vbd_kernels.py`, line 2969

GPU 版本使用 Warp tile API 进行协作式计算：
- 每个顶点分配 `TILE_SIZE=16` 个线程
- 力/海森的遍历通过 tile 内线程分摊
- 使用 `wp.tile_reduce()` 进行 tile 内归约求和
- 仅在 CUDA 设备上启用

---

## 12. 弹性力/海森矩阵计算

### 12.1 三角形膜弹性 (StVK)

**文件**: `particle_vbd_kernels.py`, `evaluate_stvk_force_hessian()` (line 860)

**能量密度**:
$$\psi = \mu \|\mathbf{G}\|_F^2 + \frac{\lambda}{2} (\text{tr}(\mathbf{G}))^2$$

其中 $\mathbf{G} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$ 是 Green 应变张量。

**计算步骤**:

1. **变形梯度** $\mathbf{F} = [\mathbf{x}_1 - \mathbf{x}_0, \mathbf{x}_2 - \mathbf{x}_0] \cdot \text{Dm\_inv}$
   - $\mathbf{f}_0, \mathbf{f}_1$ 为 $\mathbf{F}$ 的两列

2. **Green 应变**:
   - $G_{00} = \frac{1}{2}(\mathbf{f}_0 \cdot \mathbf{f}_0 - 1)$
   - $G_{11} = \frac{1}{2}(\mathbf{f}_1 \cdot \mathbf{f}_1 - 1)$
   - $G_{01} = \frac{1}{2}(\mathbf{f}_0 \cdot \mathbf{f}_1)$

3. **第一 Piola-Kirchhoff 应力**:
   - $\text{PK1}_{col0} = \mathbf{f}_0(2\mu G_{00} + \lambda \text{tr}(\mathbf{G})) + \mathbf{f}_1(2\mu G_{01})$
   - $\text{PK1}_{col1} = \mathbf{f}_0(2\mu G_{01}) + \mathbf{f}_1(2\mu G_{11} + \lambda \text{tr}(\mathbf{G}))$

4. **力** (链式法则):
   - $\frac{\partial \mathbf{F}}{\partial \mathbf{x}_i}$ 由 `vertex_order` 和 `Dm_inv` 确定
   - $\mathbf{f}_i = -\text{PK1} : \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i}$

5. **海森矩阵** (利用 Cauchy-Green 不变量):
   - $\mathbf{H}_i = (\frac{\partial \mathbf{F}}{\partial \mathbf{x}_i})^T \cdot \frac{\partial^2 \psi}{\partial \mathbf{F}^2} \cdot \frac{\partial \mathbf{F}}{\partial \mathbf{x}_i}$

6. **Rayleigh 阻尼**:
   - 两个约束: $C_\mu = \|\mathbf{G}\|_F$ 和 $C_\lambda = \text{tr}(\mathbf{G})$
   - 阻尼力: $-\mu \cdot k_d \cdot \dot{C}_\mu \cdot \nabla C_\mu - \lambda \cdot k_d \cdot \dot{C}_\lambda \cdot \nabla C_\lambda$
   - 阻尼海森: $\mu \cdot k_d / \Delta t \cdot \nabla C_\mu \otimes \nabla C_\mu + \lambda \cdot k_d / \Delta t \cdot \nabla C_\lambda \otimes \nabla C_\lambda$

7. **面积缩放**: `force *= area; hessian *= area`

### 12.2 四面体体积弹性 (Neo-Hookean)

**文件**: `particle_vbd_kernels.py`, `evaluate_volumetric_neo_hookean_force_and_hessian()` (line 335)

**能量密度** (Smith et al. 2018 稳定 Neo-Hookean):
$$\psi = \frac{\mu}{2}(\|\mathbf{F}\|_F^2 - 3) + \frac{\lambda}{2}(J - \alpha)^2$$

其中 $J = \det(\mathbf{F})$, $\alpha = 1 + \mu/\lambda$。

**计算步骤**:

1. **变形梯度**:
   - $\mathbf{D}_s = [\mathbf{x}_1 - \mathbf{x}_0, \mathbf{x}_2 - \mathbf{x}_0, \mathbf{x}_3 - \mathbf{x}_0]$
   - $\mathbf{F} = \mathbf{D}_s \cdot \text{Dm\_inv}$
   - $V_0 = 1 / (6 \cdot \det(\text{Dm\_inv}))$

2. **应力** (展平为 vec9):
   - $\text{cof}(\mathbf{F})$ = 余因子矩阵（数值稳定，不用 $J \cdot \mathbf{F}^{-T}$）
   - $\mathbf{P}_{vec} = V_0 \cdot (\mu \cdot \text{vec}(\mathbf{F}) + \lambda(J - \alpha) \cdot \text{vec}(\text{cof}(\mathbf{F})))$

3. **海森** (9×9):
   - $\mathbf{H} = V_0 \cdot (\mu \cdot \mathbf{I}_9 + \lambda \cdot \text{vec}(\text{cof}) \otimes \text{vec}(\text{cof}) + \text{cof\_derivative}(\mathbf{F}, \lambda(J-\alpha)))$

4. **逐顶点力和海森** (通过 $\mathbf{G}_i = \partial\text{vec}(\mathbf{F})/\partial\mathbf{x}_i$):
   - $\mathbf{f}_i = -\mathbf{G}_i^T \cdot \mathbf{P}_{vec}$
   - $\mathbf{H}_i = \mathbf{G}_i^T \cdot \mathbf{H} \cdot \mathbf{G}_i$

   其中 $\mathbf{G}_i$ 由 `Dm_inv` 的行和 `vertex_order` 确定。

5. **Rayleigh 阻尼**:
   - $\dot{\mathbf{F}} = \dot{\mathbf{D}}_s \cdot \text{Dm\_inv}$, 其中 $\dot{\mathbf{D}}_s$ 由位移差分计算
   - 阻尼力: $-\mathbf{G}_i^T \cdot (k_d \cdot \mathbf{H} \cdot \text{vec}(\dot{\mathbf{F}}))$
   - 阻尼海森: $(1 + k_d / \Delta t) \cdot \mathbf{H}_i$

### 12.3 二面角弯曲弹性

**文件**: `particle_vbd_kernels.py`, `evaluate_dihedral_angle_based_bending_force_hessian()` (line 1055)

**能量** (Grinspun et al. 2003 离散壳弯曲):
$$E = \frac{k \cdot l_0}{2} (\theta - \theta_0)^2$$

其中 $\theta$ 是二面角，$\theta_0$ 是参考二面角，$l_0$ 是边长，$k$ 是弯曲刚度。

**计算步骤**:

1. **4 个顶点**: o0, o1 (对侧), v0, v1 (边上)
2. **法向量**:
   - $\mathbf{n}_1 = (\mathbf{x}_2 - \mathbf{x}_0) \times (\mathbf{x}_3 - \mathbf{x}_0)$
   - $\mathbf{n}_2 = (\mathbf{x}_3 - \mathbf{x}_1) \times (\mathbf{x}_2 - \mathbf{x}_1)$
3. **二面角**:
   - $\sin\theta = (\hat{\mathbf{n}}_1 \times \hat{\mathbf{n}}_2) \cdot \hat{\mathbf{e}}$
   - $\cos\theta = \hat{\mathbf{n}}_1 \cdot \hat{\mathbf{n}}_2$
   - $\theta = \text{atan2}(\sin\theta, \cos\theta)$
4. **角度对位置的导数**:
   - $\frac{d\hat{\mathbf{n}}}{d\mathbf{x}} = \frac{1}{\|\mathbf{n}\|}(\mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T) \cdot \frac{d\mathbf{n}}{d\mathbf{x}}$
   - $\frac{d\theta}{d\mathbf{x}} = \frac{d\sin\theta}{d\mathbf{x}} \cos\theta - \frac{d\cos\theta}{d\mathbf{x}} \sin\theta$
5. **力和海森**:
   - $\mathbf{f} = -k \cdot l_0 \cdot (\theta - \theta_0) \cdot \frac{d\theta}{d\mathbf{x}}$
   - $\mathbf{H} = k \cdot l_0 \cdot \frac{d\theta}{d\mathbf{x}} \otimes \frac{d\theta}{d\mathbf{x}}$
6. **阻尼**:
   - $\dot{\theta} = \sum_{j=0}^{3} \frac{d\theta}{d\mathbf{x}_j} \cdot \frac{\Delta\mathbf{x}_j}{\Delta t}$
   - 阻尼力: $-k_d \cdot k \cdot l_0 \cdot \dot{\theta} \cdot \frac{d\theta}{d\mathbf{x}}$
   - 阻尼海森: $k_d \cdot k \cdot l_0 / \Delta t \cdot \frac{d\theta}{d\mathbf{x}} \otimes \frac{d\theta}{d\mathbf{x}}$

---

## 13. 自接触力/海森矩阵计算

### 13.1 接触力范数: `evaluate_self_contact_force_norm()`

**文件**: `particle_vbd_kernels.py`, line 1191

C² 连续的 log-barrier + 线性惩罚混合模型：

$$\text{给定距离 } d, \text{接触半径 } r, \text{刚度 } k:$$

- **Log-barrier 区域** ($d_{min} < d < \tau = r/2$):
  - $dE/dD = -k \tau^2 / d$
  - $d^2E/dD^2 = k \tau^2 / d^2$

- **二次扩展** ($d \leq d_{min}$, Taylor 展开保 C²):
  - $dE/dD = k \tau^2 (d - 2 d_{min}) / d_{min}^2$
  - $d^2E/dD^2 = k \tau^2 / d_{min}^2$

- **线性惩罚区域** ($d \geq \tau$):
  - $dE/dD = -k(r - d)$
  - $d^2E/dD^2 = k$

### 13.2 顶点-三角形碰撞

**文件**: `particle_vbd_kernels.py`, `evaluate_vertex_triangle_collision_force_hessian()` (line 1507)

- 计算顶点到三角形的最近点
- 使用 `evaluate_self_contact_force_norm()` 计算法向力
- 计算法向力/海森对 4 个顶点（1 顶点 + 3 三角形顶点）的梯度
- 摩擦力通过 `compute_friction()` 计算

### 13.3 边-边碰撞

**文件**: `particle_vbd_kernels.py`, `evaluate_edge_edge_contact()` (line 1239)

- 计算两条边的最近点对
- 使用 `evaluate_self_contact_force_norm()` 计算法向力
- 计算法向力/海森对 4 个顶点的梯度
- 4 个线程并行处理一个碰撞对

### 13.4 摩擦力: `compute_friction()`

**文件**: `particle_vbd_kernels.py`, line 1746

各向同性摩擦，使用 $\epsilon$ 平滑：

$$\mathbf{f}_{friction} = -\mu \cdot \|\mathbf{f}_n\| \cdot \frac{\mathbf{v}_t}{\|\mathbf{v}_t\| + \epsilon}$$

其中 $\mathbf{v}_t$ 是切向相对速度，$\epsilon$ 是 `friction_epsilon`。

### 13.5 累积 kernel: `accumulate_self_contact_force_and_hessian()`

**文件**: `particle_vbd_kernels.py`, line 1927

- 按碰撞原语并行（4 线程/原语）
- 只处理当前颜色的顶点
- 使用 `wp.atomic_add` 累积到 `particle_forces` 和 `particle_hessians`

---

## 14. 弹簧力/海森矩阵计算

**文件**: `particle_vbd_kernels.py`, `evaluate_spring_force_and_hessian()` (line 2255)

**能量**:
$$E = \frac{k}{2}(\|\mathbf{x}_0 - \mathbf{x}_1\| - l_0)^2$$

**力**:
$$\mathbf{f}_0 = k \cdot \frac{l_0 - l}{l} \cdot (\mathbf{x}_0 - \mathbf{x}_1)$$

$$\mathbf{f}_1 = -\mathbf{f}_0$$

**海森**:
$$\mathbf{H} = k \left(\mathbf{I} - \frac{l_0}{l}\left(\mathbf{I} - \frac{(\mathbf{x}_0 - \mathbf{x}_1) \otimes (\mathbf{x}_0 - \mathbf{x}_1)}{l^2}\right)\right)$$

**阻尼**:
$$\mathbf{H}_d = \mathbf{H} \cdot k_d / \Delta t$$
$$\mathbf{f}_d = \mathbf{H}_d \cdot (\mathbf{x}^{prev} - \mathbf{x})$$

**累积 kernel**: `accumulate_spring_force_and_hessian()` (line 2343)
- 按弹簧并行，使用 `wp.atomic_add` 累积到当前颜色的顶点

---

## 15. 粒子-刚体接触力

**文件**: `particle_vbd_kernels.py`, `accumulate_particle_body_contact_force_and_hessian()` (line 2898)

当场景中同时存在刚体和粒子时，VBD 处理粒子-刚体软接触：

- 从 `Contacts` 对象获取碰撞数据（法向、位置、刚体速度等）
- 使用 AVBD 自适应惩罚刚度 `body_particle_contact_penalty_k`
- 调用 `evaluate_body_particle_contact()` 计算力/海森
- 双向耦合：力同时作用于粒子和刚体
- 摩擦使用与自接触相同的 $\epsilon$ 平滑模型

---

## 16. 穿透自由截断 (Planar DAT)

**文件**: `particle_vbd_kernels.py`, `apply_planar_truncation_parallel_by_collision()` (line 2742)

VBD 使用 **Planar Divide-and-Truncate (DAT)** 方法保证迭代过程中不产生新穿透。

### 16.1 算法原理

对每个碰撞对（顶点-三角形 或 边-边），构造一个**分割平面**：

- **顶点-三角形**: 平面法向从顶点指向三角形最近点，平面位置在中间
- **边-边**: 平面法向从一条边最近点指向另一条边最近点

对每个顶点的位移 $\mathbf{d}_i$，计算其与所有分割平面的交点参数 $t$：

$$t = \frac{-\mathbf{n} \cdot (\mathbf{v} - \mathbf{p})}{\mathbf{n} \cdot \Delta\mathbf{v}}$$

取最小的 $t \in [0, 1]$ 作为截断因子 `truncation_ts[i]`。

### 16.2 截断应用: `apply_truncation_ts()` (line 2875)

```python
对每个粒子 p:
    displacement[p] *= truncation_ts[p]       # 缩放位移
    displacement[p] = clamp(displacement[p], max_displacement)  # 各向同性截断
    pos[p] = pos_prev_collision_detection[p] + displacement[p]
```

### 16.3 保守界

`compute_particle_conservative_bound` kernel 计算每个顶点的最大允许位移，基于：
- 自接触最小距离
- 相邻三角形/边的碰撞最小距离
- 乘以松弛因子 `conservative_bound_relaxation` (默认 0.85)

---

## 17. Phase 3: 终结 — 速度更新

**文件**: `particle_vbd_kernels.py`, `update_velocity()` (line 1897)

```python
@wp.kernel
def update_velocity(dt, pos_prev, pos, vel):
    particle = wp.tid()
    vel[particle] = (pos[particle] - pos_prev[particle]) / dt
```

**公式**:
$$\mathbf{v}^{n+1} = \frac{\mathbf{x}^{n+1} - \mathbf{x}^n}{\Delta t}$$

这是简单的向后差分速度更新。

---

## 18. 完整数据流图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        模型构建阶段                                   │
│                                                                      │
│  ModelBuilder                                                        │
│    .add_cloth_grid/mesh()  ──→  particles + triangles + edges       │
│    .add_soft_grid/mesh()   ──→  particles + tetrahedra + surface    │
│    .add_spring()           ──→  springs                              │
│    .color()                ──→  particle_color_groups                │
│    .finalize()             ──→  Model (Warp arrays on device)       │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        求解器初始化                                   │
│                                                                      │
│  SolverVBD(model)                                                    │
│    ├─ _compute_particle_force_element_adjacency()                    │
│    │    → ParticleForceElementAdjacencyInfo (CSR)                    │
│    │      v_adj_faces, v_adj_edges, v_adj_springs, v_adj_tets       │
│    ├─ TriMeshCollisionDetector (if self_contact)                     │
│    │    → BVH + collision filter lists                               │
│    └─ 分配内部状态:                                                   │
│         particle_q_prev, inertia, particle_forces,                   │
│         particle_hessians, particle_displacements, truncation_ts     │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     仿真主循环 (每步)                                  │
│                                                                      │
│  model.collide(state_in, contacts)  ──→  碰撞数据                    │
│  solver.step(state_in, state_out, control, contacts, dt)            │
│                                                                      │
│  ┌── Phase 1: 初始化 ──────────────────────────────────────────┐    │
│  │  自碰撞检测 (BVH refit + VT/EE detection)                   │    │
│  │  forward_step kernel:                                       │    │
│  │    v_new = v + (g + f_ext/m) * dt                           │    │
│  │    inertia = x + v_new * dt                                 │    │
│  │  穿透自由截断 (初始位移)                                     │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌── Phase 2: 迭代 (N 次) ─────────────────────────────────────┐    │
│  │  for color in colors:                                       │    │
│  │    1. 清零 particle_forces, particle_hessians               │    │
│  │    2. accumulate_particle_body_contact (if contacts)         │    │
│  │    3. accumulate_spring_force_and_hessian                    │    │
│  │    4. accumulate_self_contact_force_and_hessian (if sc)      │    │
│  │    5. solve_elasticity / solve_elasticity_tile:              │    │
│  │       per-vertex:                                           │    │
│  │         f = m/dt²*(inertia - x) + Σ ∂E_e/∂x_i + f_ext     │    │
│  │         H = m/dt²*I + Σ ∂²E_e/∂x_i² + H_ext              │    │
│  │         Δx = H⁻¹ * f                                       │    │
│  │         x += Δx                                             │    │
│  │    6. penetration_free_truncation                           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌── Phase 3: 终结 ────────────────────────────────────────────┐    │
│  │  update_velocity kernel:                                    │    │
│  │    v = (x_new - x_prev) / dt                                │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  state_in, state_out = state_out, state_in                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 19. 关键文件索引

| 模块 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| **SolverVBD 主类** | `newton/_src/solvers/vbd/solver_vbd.py` | 2401 | 求解器编排、初始化、step() 主循环 |
| **粒子 VBD kernels** | `newton/_src/solvers/vbd/particle_vbd_kernels.py` | 3462 | 力/海森计算、弹性求解、自接触、截断 |
| **刚体 AVBD kernels** | `newton/_src/solvers/vbd/rigid_vbd_kernels.py` | 3505 | 刚体求解、关节约束、AVBD 惩罚更新 |
| **三角网格碰撞** | `newton/_src/solvers/vbd/tri_mesh_collision.py` | 450 | BVH 自碰撞检测 |
| **图着色** | `newton/_src/sim/graph_coloring.py` | 530 | 粒子/刚体着色算法 |
| **Model** | `newton/_src/sim/model.py` | 1221 | 静态数据容器 |
| **ModelBuilder** | `newton/_src/sim/builder.py` | 10898 | 增量构建器 |
| **State** | `newton/_src/sim/state.py` | — | 动态状态容器 |
| **公共 API** | `newton/solvers.py` | — | SolverVBD 公开接口 |

---

## 附录 A: VBD 算法核心思想

VBD (Vertex Block Descent) 是一种**基于顶点的块坐标下降**方法，用于隐式求解弹性系统：

1. **块结构**: 每个顶点构成一个 3×3 的"块"，包含该顶点所有相邻力元素的贡献
2. **Gauss-Seidel 迭代**: 按颜色组顺序更新顶点位置，同色组内并行
3. **局部牛顿步**: 每个顶点求解一个 3×3 线性系统 $\mathbf{H}_i \Delta\mathbf{x}_i = -\mathbf{f}_i$
4. **保证收敛**: 海森矩阵正定（惯性项 $\frac{m}{\Delta t^2}\mathbf{I}$ 提供正定性保证）
5. **穿透自由**: Planar DAT 截断保证迭代过程中不产生新穿透

相比 XPBD：
- VBD 直接求解弹性能量，不需要约束投影
- VBD 的 3×3 局部系统比 XPBD 的标量约束更精确
- VBD 自然处理非线性能量（Neo-Hookean, StVK），无需线性化

## 附录 B: 约束类型汇总

| 约束类型 | 能量模型 | 力/海森函数 | 适用场景 |
|----------|----------|-------------|----------|
| 三角形膜 | StVK (Green 应变) | `evaluate_stvk_force_hessian` | 布料 |
| 四面体体积 | Neo-Hookean (Smith 2018) | `evaluate_volumetric_neo_hookean_force_and_hessian` | 软体 |
| 弯曲边 | 离散壳二面角 | `evaluate_dihedral_angle_based_bending_force_hessian` | 布料弯曲 |
| 弹簧 | 距离约束 | `evaluate_spring_force_and_hessian` | 通用连接 |
| 自接触 (VT) | C² log-barrier | `evaluate_vertex_triangle_collision_force_hessian` | 布料自碰撞 |
| 自接触 (EE) | C² log-barrier | `evaluate_edge_edge_contact` | 布料自碰撞 |
| 粒子-刚体 | AVBD 自适应惩罚 | `evaluate_body_particle_contact` | 耦合接触 |
