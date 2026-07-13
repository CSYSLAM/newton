# Newton VBD Planar DAT (Divide and Truncate) 算法详解

## 1. 算法概述

**Planar DAT**（Planar Divide and Truncate，平面分割与截断）是 Newton VBD（Verlet-Based Dynamics）求解器中的**核心无穿透保证机制**。它的作用是在每次 Gauss-Seidel 迭代后，将粒子的位移截断到一个安全范围内，确保更新后的位置不会导致网格自穿透。

### 一句话总结

> Planar DAT = 为每个碰撞对构造一个"分割平面"，然后计算每个粒子位移到达该平面的参数 t，取所有碰撞的**最小 t** 来截断位移，保证不穿透。

---

## 2. 为什么需要 Planar DAT？

VBD 使用 Gauss-Seidel 迭代求解弹性势能，每次迭代中每个粒子独立计算一个"期望位移" `delta_v`。问题在于：

1. **位移可能过大**：弹性力/自接触力可能推动粒子穿过对面的网格面
2. **并行更新冲突**：同一碰撞对涉及多个粒子，各自独立更新可能互相穿透
3. **需要无穿透保证**：软体仿真中，自穿透一旦发生就很难恢复

Planar DAT 通过在碰撞对之间插入"分割平面"，将位移截断到平面之前，从数学上保证不会穿透。

---

## 3. 数学公式

### 3.1 分割平面定义

分割平面由法线 `n` 和平面上一点 `d` 定义：

$$
\Pi: \quad \mathbf{n} \cdot (\mathbf{p} - \mathbf{d}) = 0
$$

其中：
- `n` 是单位法线，指向"安全侧"（被截断粒子应在法线正侧）
- `d` 是平面上的一个参考点

### 3.2 截断参数 t 的计算

给定粒子当前位置 `v` 和位移 `delta_v`，粒子运动轨迹为：

$$
\mathbf{p}(t) = \mathbf{v} + t \cdot \Delta\mathbf{v}, \quad t \in [0, 1]
$$

将轨迹代入平面方程，求解交点：

$$
\mathbf{n} \cdot (\mathbf{v} + t \cdot \Delta\mathbf{v} - \mathbf{d}) = 0
$$

$$
t = \frac{\mathbf{n} \cdot (\mathbf{d} - \mathbf{v})}{\mathbf{n} \cdot \Delta\mathbf{v}}
$$

### 3.3 松弛截断

直接使用交点参数 t 作为截断因子过于激进（恰好在平面上，数值误差可能导致微小穿透）。因此引入松弛因子 `gamma_r`（默认 0.85）和最小偏移 `gamma_min`（默认 1e-3）：

$$
t_{\text{truncated}} = \max\!\Big(\min\!\big(t \cdot \gamma_r,\; t - \gamma_{\min}\big),\; 0\Big)
$$

**通俗解释**：
- `t * gamma_r`：按比例缩小，留 15% 的安全余量
- `t - gamma_min`：保证绝对偏移量至少为 gamma_min
- 取两者较小值：更保守的那个生效
- `max(..., 0)`：不反向移动

### 3.4 多碰撞取最小

一个粒子可能同时参与多个碰撞，每个碰撞给出一个截断参数。最终截断因子取所有碰撞的**最小值**：

$$
t_{\text{final}} = \min_{\text{所有碰撞}} t_{\text{truncated}}^{(i)}
$$

这通过 `wp.atomic_min` 原子操作在 GPU 上并行实现。

### 3.5 最终位移

$$
\Delta\mathbf{v}_{\text{final}} = t_{\text{final}} \cdot \Delta\mathbf{v}
$$

同时还有一个最大位移限制 `max_displacement`（各向同性截断）：

$$
\Delta\mathbf{v}_{\text{final}} = \min\!\Big(|\Delta\mathbf{v}_{\text{final}}|,\; d_{\max}\Big) \cdot \frac{\Delta\mathbf{v}_{\text{final}}}{|\Delta\mathbf{v}_{\text{final}}|}
$$

其中 $d_{\max} = \text{self\_contact\_margin} \times \text{conservative\_bound\_relaxation} \times 0.5$。

---

## 4. 算法流程

```
┌─────────────────────────────────────────────────────────┐
│              VBD 一次迭代完整流程                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 碰撞检测（_collision_detection_penetration_free）     │
│     ├─ BVH refit                                       │
│     ├─ 顶点-三角形碰撞检测                               │
│     └─ 边-边碰撞检测                                    │
│     → 输出: TriMeshCollisionInfo                        │
│                                                         │
│  2. Gauss-Seidel 迭代（_solve_particle_iteration）       │
│     ├─ 累加弹性力/自接触力 → particle_forces             │
│     ├─ 累加 Hessian → particle_hessians                 │
│     └─ 求解位移 → particle_displacements                │
│                                                         │
│  3. Planar DAT 截断（_penetration_free_truncation）      │
│     ├─ truncation_ts 初始化为 1.0                       │
│     ├─ 对每个碰撞对:                                    │
│     │   ├─ 构造分割平面 (n, d)                           │
│     │   ├─ 对每个相关粒子计算 t                          │
│     │   └─ atomic_min(truncation_ts[粒子], t)           │
│     ├─ 应用截断: displacement *= truncation_ts          │
│     └─ 限制最大位移: clamp |displacement| ≤ max_disp    │
│                                                         │
│  4. 更新位置: pos_new = pos_old + displacement_final    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 详细步骤

#### Step 1: 碰撞检测

在碰撞检测阶段，系统检测两类碰撞：

| 碰撞类型 | 描述 | 涉及粒子数 |
|---------|------|-----------|
| 顶点-三角形 (VT) | 一个顶点穿透一个三角形面 | 4 (1顶点 + 3三角形顶点) |
| 边-边 (EE) | 两条边交叉/接近 | 4 (每条边2个端点) |

碰撞检测结果存储在 `TriMeshCollisionInfo` 结构体中，包含：
- `vertex_colliding_triangles`: 每个顶点碰撞的三角形列表
- `edge_colliding_edges`: 每条边碰撞的边列表

#### Step 2: 构造分割平面

**顶点-三角形碰撞**的分割平面构造：

1. 计算顶点 `v` 到三角形 `(t1, t2, t3)` 的最近点 `closest_p`
2. 法线 `n = normalize(v - closest_p)`，指向顶点侧
3. 平面位置 `d` 根据位移比例插值：
   - 计算 `delta_v_n`：顶点朝法线负方向（靠近三角形）的位移分量
   - 计算 `delta_t_n`：三角形顶点朝法线正方向（靠近顶点）的位移分量最大值
   - `lambda = delta_t_n / (delta_t_n + delta_v_n)`，clamp 到 [0.05, 0.95]
   - `d = closest_p + lambda * n_hat`（在最近点和顶点之间插值）

**边-边碰撞**的分割平面构造：

1. 计算两条边的最近点 `c1`（边1上）和 `c2`（边2上）
2. 法线 `n = normalize(c1 - c2)`，指向边1侧
3. 平面位置 `d` 类似地按位移比例插值

#### Step 3: 计算截断参数

对每个碰撞的每个相关粒子，调用 `planar_truncation_t`：

```python
# 求解: n · (v + t * delta_v - d) = 0
denom = dot(n, delta_v)
if |denom| < eps:  # 位移与平面平行，不截断
    return 1.0
t = dot(n, d - v) / denom
if t < 0:  # 交点在反方向，不截断
    return 1.0
t = clamp(min(t * gamma_r, t - gamma_min), 0, 1)
return t
```

#### Step 4: 应用截断

```python
# 对每个粒子
displacement_final = displacement * truncation_t
if |displacement_final| > max_displacement:
    displacement_final *= max_displacement / |displacement_final|
pos_new = pos_old + displacement_final
```

---

## 5. 分割平面构造的直觉理解

### 5.1 为什么平面位置要按位移比例插值？

考虑顶点-三角形碰撞：顶点在移动，三角形也在移动。分割平面应该放在"双方位移的中间"位置，这样：

- 如果顶点移动得多（`delta_v_n` 大），平面更靠近顶点 → 顶点被截断得更早
- 如果三角形移动得多（`delta_t_n` 大），平面更靠近三角形 → 三角形顶点被截断得更早

这保证了**双方都被合理约束**，不会只限制一方。

### 5.2 为什么需要 is_dummy 标记？

不是所有相关粒子的位移都会穿过分割平面。`segment_plane_intersects` 检查粒子的位移线段是否与平面相交：

- 如果不相交（`is_dummy = True`）：该粒子的位移不会导致穿透，不需要截断
- 如果相交（`is_dummy = False`）：需要计算截断参数

这避免了不必要的截断，保持仿真的灵活性。

### 5.3 robust_edge_pair_normal 的作用

当两条边平行或退化时，叉积无法给出有效法线。`robust_edge_pair_normal` 通过多级回退策略处理：

1. 优先：两条边方向的叉积
2. 回退1：边方向与中点连线的叉积
3. 回退2：边方向与坐标轴的叉积
4. 最终：使用另一个坐标轴

---

## 6. 代码对应关系

### 6.1 核心函数

| 函数 | 文件位置 | 功能 |
|------|---------|------|
| `segment_plane_intersects` | `particle_vbd_kernels.py:2476` | 判断线段是否与平面相交 |
| `create_vertex_triangle_division_plane_closest_pt` | `particle_vbd_kernels.py:2501` | 构造VT碰撞的分割平面 |
| `create_edge_edge_division_plane_closest_pt` | `particle_vbd_kernels.py:2631` | 构造EE碰撞的分割平面 |
| `robust_edge_pair_normal` | `particle_vbd_kernels.py:2562` | 鲁棒计算边对法线 |
| `planar_truncation` | `particle_vbd_kernels.py:2705` | 计算截断后的位移向量 |
| `planar_truncation_t` | `particle_vbd_kernels.py:2725` | 计算截断参数 t |
| `apply_planar_truncation_parallel_by_collision` | `particle_vbd_kernels.py:2745` | 并行处理所有碰撞的截断 |
| `apply_truncation_ts` | `particle_vbd_kernels.py:2877` | 应用截断参数到位移 |

### 6.2 求解器集成

| 方法 | 文件位置 | 功能 |
|------|---------|------|
| `_penetration_free_truncation` | `solver_vbd.py:1669` | Planar DAT 入口，调度两个 kernel |
| `_collision_detection_penetration_free` | `solver_vbd.py:2832` | 碰撞检测，为 Planar DAT 提供输入 |
| `_initialize_particles` | `solver_vbd.py:1731` | 初始化阶段调用截断 |
| `_solve_particle_iteration` | `solver_vbd.py:2168` | 每次迭代后调用截断 |

### 6.3 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `particle_conservative_bound_relaxation` | 0.85 | 松弛因子 gamma_r，越小越保守 |
| `particle_self_contact_margin` | 0.2 | 自接触检测的边距 |
| `gamma_min` | 1e-3 | 最小绝对偏移量 |
| `parallel_eps` | 1e-5 | 判断平行的阈值 |
| `max_displacement` | margin × relaxation × 0.5 | 各向同性最大位移限制 |

---

## 7. 通俗解释：用生活比喻理解 Planar DAT

### 比喻：走廊里的两个人

想象两个人在一条窄走廊里相向而行：

1. **碰撞检测**：发现两人距离太近，可能相撞
2. **构造分割平面**：在两人中间画一条"红线"（分割平面），法线指向各自一侧
3. **计算截断参数**：
   - 人A走了3步会越过红线 → t_A = 2/3（走到红线前2步的位置）
   - 人B走了2步会越过红线 → t_B = 1/2（走到红线前1步的位置）
4. **松弛截断**：不能恰好走到红线上，要留点安全距离
   - t_A_truncated = 2/3 × 0.85 ≈ 0.57（再退一点）
   - t_B_truncated = 1/2 × 0.85 ≈ 0.43（再退一点）
5. **多碰撞取最小**：如果人A同时和3个人可能相撞，取最严格的那个限制
6. **最终结果**：两人都停在红线之前，保证不会相撞

### 比喻：橡皮筋上的珠子

更贴近物理仿真的比喻：一根橡皮筋上串着很多珠子，每颗珠子都想往某个方向移动。

- **没有 Planar DAT**：珠子们各走各的，可能互相穿过 → 穿透
- **有 Planar DAT**：在可能穿过的珠子之间画"挡板"，每颗珠子最多走到挡板前 → 无穿透

---

## 8. 算法伪代码

```
function PlanarDAT(positions, displacements, collision_info, gamma_r):
    # 初始化截断因子为 1.0（不截断）
    truncation_ts = array of 1.0, size = particle_count

    # 处理边-边碰撞
    for each edge_pair (e1, e2) in collision_info.edge_collisions:
        # 获取4个端点位置和位移
        e1_v0, e1_v1 = endpoints of e1
        e2_v0, e2_v1 = endpoints of e2

        # 构造分割平面
        n, d, is_dummy = create_edge_edge_division_plane(
            pos[e1_v0], disp[e1_v0], pos[e1_v1], disp[e1_v1],
            pos[e2_v0], disp[e2_v0], pos[e2_v1], disp[e2_v1]
        )

        # 对每个相关粒子计算截断参数
        for v in [e1_v0, e1_v1, e2_v0, e2_v1]:
            if not is_dummy[v]:
                t = planar_truncation_t(pos[v], disp[v], n, d, eps, gamma_r)
                atomic_min(truncation_ts[v], t)  # 取最严格的限制

    # 处理顶点-三角形碰撞
    for each vertex_triangle (v, tri) in collision_info.vertex_triangle_collisions:
        t1, t2, t3 = vertices of tri

        # 构造分割平面
        n, d, is_dummy = create_vertex_triangle_division_plane(
            pos[v], disp[v], pos[t1], disp[t1], pos[t2], disp[t2], pos[t3], disp[t3]
        )

        # 对每个相关粒子计算截断参数
        for p in [v, t1, t2, t3]:
            if not is_dummy[p]:
                t = planar_truncation_t(pos[p], disp[p], n, d, eps, gamma_r)
                atomic_min(truncation_ts[p], t)

    # 应用截断
    for each particle i:
        displacement_final = displacements[i] * truncation_ts[i]
        if |displacement_final| > max_displacement:
            displacement_final *= max_displacement / |displacement_final|
        positions[i] += displacement_final

    return positions
```

---

## 9. 关键设计决策分析

### 9.1 为什么用 atomic_min 而不是锁？

GPU 并行计算中，一个粒子可能同时被多个碰撞对影响。`atomic_min` 是无锁的原子操作，比互斥锁高效得多。代价是截断可能**过于保守**（取了所有碰撞的最小值），但保守总比穿透好。

### 9.2 为什么分割平面位置要考虑位移？

如果平面固定在碰撞对的中间位置，当一侧粒子移动很多而另一侧几乎不动时，移动多的粒子会被过度截断。按位移比例插值平面位置，使得"谁动得多，谁被截断得多"，更加公平和高效。

### 9.3 为什么需要各向同性最大位移限制？

Planar DAT 只能防止**已检测到的碰撞**导致的穿透。对于未检测到的碰撞（BVH 遗漏、碰撞缓冲区溢出等），各向同性最大位移限制提供了一个兜底保护：每步位移不超过 `margin × relaxation × 0.5`，确保即使漏检也不会一步穿透太远。

### 9.4 gamma_r = 0.85 的含义

0.85 意味着每次截断留 15% 的安全余量。这个值是经验性的：
- 太大（如 0.99）：安全余量不足，数值误差可能导致微小穿透
- 太小（如 0.5）：过度保守，收敛变慢，仿真看起来"僵硬"
- 0.85 是一个较好的平衡点

---

## 10. 与 IPC (Incremental Potential Contact) 的对比

| 特性 | Planar DAT | IPC |
|------|-----------|-----|
| 穿透保证 | 每步迭代后保证无穿透 | 全局优化保证无穿透 |
| 计算方式 | 截断位移（后处理） | 约束优化（内嵌） |
| 并行性 | 高（atomic_min） | 低（需要全局线性系统求解） |
| 收敛性 | 可能过于保守 | 更精确 |
| 适用场景 | 实时仿真、GPU 并行 | 离线仿真、高精度 |

Planar DAT 牺牲了一定的精度（可能过度截断），换取了极高的并行效率和实时性能。

---

## 11. 数据流图

```
                    ┌──────────────────┐
                    │  碰撞检测         │
                    │  BVH + VT/EE     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ TriMeshCollision │
                    │ Info             │
                    │ ├ vertex↔tri     │
                    │ └ edge↔edge      │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ VT分割平面    │  │ EE分割平面    │  │ Gauss-Seidel │
  │ 构造         │  │ 构造         │  │ 位移计算      │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         │                 │                 │
         │    n, d, is_dummy                │
         │                 │                 │
         └────────┬────────┘                 │
                  │                          │
                  ▼                          ▼
         ┌──────────────────┐      ┌──────────────────┐
         │ planar_truncation│      │ displacements     │
         │ _t per particle  │      │ (未截断)          │
         └────────┬─────────┘      └────────┬─────────┘
                  │                          │
                  ▼                          │
         ┌──────────────────┐                │
         │ atomic_min       │                │
         │ truncation_ts[]  │                │
         └────────┬─────────┘                │
                  │                          │
                  └──────────┬───────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ apply_truncation │
                    │ displacement *= t│
                    │ clamp |d| ≤ max  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ pos_new =        │
                    │ pos + disp_final │
                    └──────────────────┘
```

---

## 12. 总结

Planar DAT 是 VBD 求解器中保证无穿透的核心机制，其核心思想可以概括为：

1. **检测**：通过 BVH 加速的碰撞检测，找出所有可能穿透的碰撞对
2. **分割**：为每个碰撞对构造一个分割平面，将空间分为"安全侧"和"危险侧"
3. **截断**：计算每个粒子位移到达分割平面的参数 t，用松弛因子缩小后截断位移
4. **聚合**：通过 atomic_min 取所有碰撞的最小截断因子
5. **兜底**：各向同性最大位移限制防止漏检碰撞导致的穿透

这种设计在保证无穿透的前提下，实现了高效的 GPU 并行计算，适合实时软体仿真场景。
