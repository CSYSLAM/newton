# Newton Style3D 源码分析

## 1. 文档目标

本文基于 `newton/_src/solvers/style3d/` 的实际实现，对 Newton 当前的 Style3D 布料求解器做一次工程向拆解。

重点不是泛泛介绍 Projective Dynamics，而是直接回答这些问题：

- 这个求解器在项目里的定位是什么。
- 模块是怎么划分的。
- 数据从 `ModelBuilder` 到 `SolverStyle3D.step()` 是怎么流动的。
- 每个关键函数的输入、输出、内部实现和伪代码是什么。
- 拉伸、弯曲、碰撞、线性求解分别如何组织。
- 整个仿真一帧的执行链路是什么。

## 2. 一句话结论

当前 Newton 的 Style3D 不是 VBD，也不是完整 FEM shell Newton 求解器，而是一个单独的、面向布料的 Projective Dynamics 风格隐式求解器。

它的核心结构是：

- 用 `cloth.py` 在建模阶段把 2D panel 空间的布料信息编码进 `ModelBuilder` 和自定义属性。
- 用 `builder.py` 预计算一份固定的 PD 稀疏矩阵。
- 在 `solver_style3d.py` 中做每步多轮非线性迭代。
- 每轮迭代里把惯性项、拉伸项、弯曲项、拖拽项、碰撞项累积到右端项 `rhs`。
- 再用 `linear_solver.py` 中的 PCG 对线性系统做近似求解。
- `collision/` 目录提供 BVH 广相、接触力和接触 Hessian 对角线。

## 3. 相关源码文件

核心目录：`newton/_src/solvers/style3d/`

### 3.1 主文件

- `solver_style3d.py`
  - 主求解器 `SolverStyle3D`
  - 负责预计算、时间步推进、非线性迭代和与碰撞系统耦合
- `cloth.py`
  - 布料建模辅助
  - 负责生成 panel 空间 rest 数据、自定义属性、边弯曲信息、缝合弹簧
- `builder.py`
  - `PDMatrixBuilder`
  - 负责预计算固定 PD 稀疏矩阵
- `kernels.py`
  - Warp 内核集合
  - 包含拉伸、弯曲、初始化、预条件器和状态更新逻辑
- `linear_solver.py`
  - `PcgSolver`
  - 负责 ELL 稀疏矩阵上的预条件共轭梯度法

### 3.2 碰撞相关

- `collision/collision.py`
  - 当前主碰撞类 `Collision`
- `collision/kernels.py`
  - 顶点-三角形、边-边、边-面解缠、刚体-粒子接触内核
- `collision/bvh.py`
  - 三角形和边的 BVH 封装
- `collision/collision_legacy.py`
  - 旧版碰撞实现，保留但已不是主路径

## 4. 项目架构

Style3D 求解器可以按四层来理解：

### 4.1 建模层

文件：`cloth.py`

职责：

- 接受 mesh 或 grid 形式的布料输入。
- 计算 panel 空间三角形 rest 数据。
- 计算边弯曲所需的 cot 权重与 rest area。
- 写入 `builder.tri_poses`、`builder.tri_areas`。
- 通过 custom attributes 把 Style3D 所需的各向异性和弯曲参数挂到 model 上。

### 4.2 预计算层

文件：`builder.py`、`solver_style3d.py::_precompute()`

职责：

- 把拉伸和弯曲约束的固定 Hessian 近似装配成稀疏矩阵。
- 生成 PD 对角项 `pd_diags`。
- 生成 ELL 格式非对角项 `pd_non_diags`。

### 4.3 每步求解层

文件：`solver_style3d.py`、`kernels.py`、`linear_solver.py`

职责：

- 计算惯性预测位置 `x_inertia`。
- 累积右端项 `rhs`。
- 构造 Jacobi 预条件器。
- 用 PCG 解近似线性系统。
- 更新位置和速度。

### 4.4 碰撞层

文件：`collision/collision.py`、`collision/kernels.py`

职责：

- 用三角形 BVH 和边 BVH 做广相。
- 在窄相里生成 VF、EE、EF 和刚体-粒子接触力。
- 输出接触力和接触 Hessian 对角线。
- 通过 Hessian 向量积接口耦合进 PCG。

## 5. 算法模型

## 5.1 时间积分模型

`SolverStyle3D` 采用隐式欧拉风格的位置更新，文档字符串里写出的主方程为：

```text
(M / dt^2 + H(x)) * dx = (M / dt^2) * (x_inertia - x) + f_int(x)
```

其中：

- `M` 是粒子质量对角矩阵。
- `x` 是当前迭代位置。
- `dx` 是本轮线性系统求出的位移增量。
- `H(x)` 是当前位置相关的 Hessian 近似。
- `x_inertia` 是显式预测出来的惯性位置。

代码里的惯性预测公式实际是：

```text
x_inertia = x_last + v_prev * dt + (gravity + f_ext / mass) * dt^2
```

## 5.2 拉伸模型

拉伸实现位于 `kernels.py::eval_stretch_kernel()`。

它不是直接使用 Newton VBD 那套 stable Neo-Hookean，而是更接近 Baraff-Witkin / PD 风格的各向异性布料拉伸模型。

核心输入：

- 当前三角形顶点位置
- 三角形面积 `face_area`
- panel 空间逆 rest 矩阵 `inv_dm`
- 各向异性刚度 `aniso_ke = (ku, kv, ks)`

核心中间量：

- `Fu`, `Fv`：三角形在 panel 的两条基方向形变梯度
- `len_Fu`, `len_Fv`：两方向拉伸长度
- `dFu_dx`, `dFv_dx`：对三个顶点的导数系数

力项结构：

- `ku * (len_Fu - 1)` 控制一个方向拉伸
- `kv * (len_Fv - 1)` 控制另一个方向拉伸
- `ks * dot(Fu, Fv)` 控制剪切

可以写成更直观的伪能量形式：

```text
E_stretch ~= area * [
    ku * (|Fu| - 1)^2
  + kv * (|Fv| - 1)^2
  + ks * dot(Fu_hat, Fv_hat)^2
]
```

这里的实现直接累积梯度力到 `rhs`，而固定 Hessian 近似则在 `builder.py` 中预装配。

## 5.3 弯曲模型

弯曲实现位于：

- 建模预处理：`cloth.py::_compute_edge_bending_data()`
- 动力学内核：`kernels.py::eval_bend_kernel()`
- 预计算矩阵：`builder.py::add_bend_constraints()`

它的结构是典型的离散铰链边弯曲模型。

关键数据：

- `edge_rest_area`
- `edge_bending_cot`
- `edge_bending_properties[:, 0]` 作为弯曲刚度来源

`edge_bending_cot` 本质上是四个 cot 权重，后续被组合为 4 个顶点的弯曲权重：

```text
w2 = cot2 + cot3
w3 = cot0 + cot1
w0 = -cot0 - cot2
w1 = -cot1 - cot3
```

再通过外积构造每条边对应的 `4 x 4` 标量 Hessian：

```text
H_bend = outer(w, w) * (edge_stiff / edge_rest_area)
```

运行时 `eval_bend_kernel()` 不再重建完整矩阵，而是直接按权重对四个顶点累积力。

## 5.4 拖拽模型

拖拽是可选项。

相关内核：

- `accumulate_dragging_pd_diag_kernel()`
- `eval_drag_force_kernel()`

模型含义：

- 用户指定一个三角形和重心坐标。
- 在该三角形的重心插值点与目标拖拽点之间接一根弹簧。
- 弹簧刚度 `drag_spring_stiff` 同时影响：
  - 右端项中的拖拽力
  - 固定 PD 矩阵对角线

## 5.5 碰撞模型

当前 `Collision` 是一个工程近似接触系统，而不是完整 IPC barrier 优化。

它包含四类接触：

- 顶点-三角形 `VF`
- 边-边 `EE`
- 边-面解缠 `EF`
- 刚体-粒子 soft contact

这些接触都输出：

- 接触力，写入 `particle_forces`
- 接触 Hessian 对角块，写入 `contact_hessian_diags`

其中：

- `VF` 和 `EE` 使用基于穿透厚度的法向 penalty 形式。
- `EF` 更像 edge-face untangling，参考了 Volino 2006 的 intersection contour minimization 思路。
- 刚体-粒子接触复用了 VBD 的 `evaluate_body_particle_contact()`。

## 6. 数据链路

## 6.1 从输入网格到模型属性

以 `add_cloth_mesh()` 为例，数据链路如下：

```text
vertices / indices / panel_verts / panel_indices
-> _compute_panel_triangles()
-> panel_inv_D, panel_areas
-> builder.add_particles()
-> builder.add_triangles(custom_attributes={style3d:tri_aniso_ke})
-> builder.tri_poses = panel_inv_D
-> builder.tri_areas = panel_areas
-> _compute_edge_bending_data()
-> builder.add_edges(custom_attributes={
     style3d:edge_rest_area,
     style3d:edge_bending_cot,
     style3d:aniso_ke,
   })
-> finalize()
-> model.style3d.*
```

这里最关键的一点是：

- `builder.add_triangles()` 先创建普通三角形约束。
- 然后 `cloth.py` 主动覆盖 `builder.tri_poses` 和 `builder.tri_areas`。

也就是说，Style3D 使用的是 panel 空间 rest 数据，不是单纯用 3D 初始几何直接做 rest pose。

## 6.2 从模型到预计算矩阵

当 `SolverStyle3D(model)` 初始化时：

```text
model.tri_indices
model.tri_poses
model.tri_areas
model.edge_indices
model.edge_bending_properties
model.style3d.tri_aniso_ke
model.style3d.edge_rest_area
model.style3d.edge_bending_cot
-> PDMatrixBuilder.add_stretch_constraints()
-> PDMatrixBuilder.add_bend_constraints()
-> finalize()
-> self.pd_diags + self.pd_non_diags
```

## 6.3 从每步输入到每步输出

每一帧 `step(state_in, state_out, control, contacts, dt)` 的核心数据流：

```text
state_in.particle_q / particle_qd / particle_f
contacts
model.gravity / masses / flags / tri data / edge data / style3d attrs
-> init_step_kernel()
-> x_prev / x_inertia / static_A_diags / dx
-> 非线性迭代
   -> rhs 初始化
   -> stretch force
   -> bend force
   -> drag force(optional)
   -> collision force(optional)
   -> Jacobi preconditioner
   -> PCG solve
   -> nonlinear_step_kernel
-> state_out.particle_q
-> update_velocity
-> state_out.particle_qd
```

## 7. 建模阶段实现方案

## 7.1 `add_cloth_mesh()`

### 输入

- `builder`
- `pos`, `rot`, `vel`
- `vertices`, `indices`
- `density`
- 可选 `panel_verts`, `panel_indices`
- 可选 `tri_aniso_ke`, `edge_aniso_ke`
- 其他阻尼、空气、弹簧、半径参数

### 输出

- 不直接返回值
- 就地修改 `builder`

### 实现步骤

```text
1. 把 3D 顶点乘 scale，再做 rot/pos 变换，得到 world-space 顶点。
2. 准备 panel 空间 2D 顶点；若未显式传入则取 vertices 的 XY。
3. 计算 panel 三角形 Dm^{-1} 和面积。
4. 过滤退化或反转三角形。
5. 调用 builder.add_particles() 创建粒子。
6. 调用 builder.add_triangles() 创建三角形，并写入 style3d:tri_aniso_ke。
7. 覆盖 builder.tri_poses 和 builder.tri_areas 为 panel rest 数据。
8. 根据 panel 面积把密度分摊到三个顶点质量。
9. 根据 mesh 邻接关系生成边、cot 权重和 rest area。
10. 调用 builder.add_edges() 写入弯曲边和 Style3D 自定义属性。
11. 若 add_springs=True，再额外生成结构弹簧。
```

### 伪代码

```python
def add_cloth_mesh(...):
    verts_3d = transform(vertices)
    panel_inv_D_all, panel_areas_all = compute_panel_triangles(panel_verts, panel_indices)
    valid_tris = filter_valid(panel_areas_all)

    builder.add_particles(verts_3d)
    builder.add_triangles(valid_tris, custom_attributes={"style3d:tri_aniso_ke": tri_aniso_ke})

    builder.tri_poses[tri_range] = panel_inv_D
    builder.tri_areas[tri_range] = panel_areas

    distribute_mass_to_vertices(density, panel_areas)

    edge_data = compute_edge_bending_data(panel_verts, panel_indices_valid, tri_indices_valid)
    builder.add_edges(..., custom_attributes=edge_data.attrs)

    if add_springs:
        add_structural_springs()
```

## 7.2 `add_cloth_grid()`

### 输入

- 网格尺寸 `dim_x`, `dim_y`
- 单元尺寸 `cell_x`, `cell_y`
- 粒子质量 `mass`
- 固定边界开关

### 输出

- 不直接返回
- 最终仍然是修改 `builder`

### 实现逻辑

- 先生成规则二维网格和三角面。
- 根据总质量和总面积反推 panel 密度：

```text
density = total_mass / total_area
```

- 再委托给 `add_cloth_mesh()`。
- 最后按 `fix_left/right/top/bottom` 把边界粒子设置为非激活，且质量清零。

## 7.3 缝合逻辑

相关函数：

- `create_mesh_sew_springs()`
- `sew_close_vertices()`

逻辑：

- 先对 edge 构建 BVH。
- 查询一定距离内的可缝合顶点对。
- 为这些顶点对添加 spring。

这提供了一种服装片段拼接的轻量方案。

## 8. 预计算阶段

## 8.1 `SolverStyle3D._precompute()`

### 输入

- `model`

### 输出

- `self.pd_diags`
- `self.pd_non_diags.num_nz`
- `self.pd_non_diags.nz_ell`

### 实现逻辑

```text
1. 检查 model.style3d 命名空间和必要属性是否存在。
2. 把 tri/edge 数据转成 Python list。
3. 创建 PDMatrixBuilder。
4. 加入 stretch constraints。
5. 加入 bend constraints。
6. finalize 为目标 device 上的 ELL 矩阵。
```

### 伪代码

```python
def _precompute(model):
    assert model.style3d.tri_aniso_ke exists
    assert model.style3d.edge_rest_area exists
    assert model.style3d.edge_bending_cot exists

    pd_builder = PDMatrixBuilder(model.particle_count)
    pd_builder.add_stretch_constraints(...)
    pd_builder.add_bend_constraints(...)
    self.pd_diags, self.pd_non_diags.num_nz, self.pd_non_diags.nz_ell = pd_builder.finalize(device)
```

## 8.2 `PDMatrixBuilder` 模块设计

这个模块的职责非常单纯：

- 不管时间步。
- 不管碰撞。
- 只负责把固定约束刚度装配成一份静态稀疏矩阵。

内部数据：

- `counts`：每个顶点已有多少邻接非零项
- `diags`：每个顶点的对角项
- `values`：每个顶点到邻居的非对角标量值
- `neighbors`：邻居编号

最终导出为 ELL：

- `num_nz`
- `nz_ell[k, row] = (column_index, value)`

## 9. 仿真流程

## 9.1 单步仿真总流程

`SolverStyle3D.step()` 的真实流程是：

```text
frame_begin()
-> init_step_kernel()
-> optional dragging diag accumulation
-> for nonlinear iterations:
     init_rhs_kernel()
     eval_stretch_kernel()
     eval_bend_kernel()
     optional eval_drag_force_kernel()
     optional collision.accumulate_contact_force()
     build Jacobi preconditioner
     linear_solver.solve()
     optional collision.linear_iteration_end()
     nonlinear_step_kernel()
     state_in.particle_q <- state_out.particle_q
-> update_velocity()
-> frame_end()
```

## 9.2 `init_step_kernel()`

### 输入

- `dt`
- `gravity`
- `particle_world`
- `f_ext`
- `v_curr`
- `x_curr`
- `pd_diags`
- `particle_masses`
- `particle_flags`

### 输出

- `x_prev`
- `x_inertia`
- `static_A_diags`
- `dx`

### 含义

- `x_prev` 保存步前位置。
- `x_inertia` 是显式预测位置。
- `static_A_diags = pd_diags + m / dt^2`。
- `dx = v_prev * dt` 作为第一轮线性求解初值。

## 9.3 `init_rhs_kernel()`

### 输入

- `dt`
- `x_curr`
- `x_inertia`
- `particle_masses`

### 输出

- `rhs`

### 公式

```text
rhs = (x_inertia - x_curr) * mass / dt^2
```

这个 `rhs` 后续再叠加所有内力和接触力。

## 9.4 拉伸与弯曲的运行时更新

这一步在每个非线性迭代里发生：

- `eval_stretch_kernel()` 对每个三角形遍历一次。
- `eval_bend_kernel()` 对每条边遍历一次。

它们都只做一件事：

- 把当前迭代位置 `x` 下的内力加到 `rhs`。

固定矩阵部分已经在 `_precompute()` 里完成，所以这里不用重新装配完整 Hessian。

## 9.5 线性求解流程

`linear_solver.solve()` 使用预条件共轭梯度法。

系统矩阵可以理解为：

```text
A = static_PD_matrix + optional_contact_hessian
```

其中：

- `static_PD_matrix` 由 `pd_non_diags + static_A_diags` 表示。
- 接触 Hessian 不显式装配完整矩阵，只通过：
  - 对角线进入 Jacobi 预条件器
  - `hessian_multiply()` 进入矩阵向量积

PCG 子流程：

```text
step1_update_r   : r = b - A x
step2_update_z   : z = M^{-1} r
step3_update_rTz : 记录内积
step4_update_p   : 更新搜索方向
step5_update_Ap  : 计算 A p
step6_update_pTAp: 记录 p^T A p
step7_update_x_r : 更新 x 和 r
```

## 9.6 非线性位置更新

`nonlinear_step_kernel()` 很简单：

```text
x_out = x_in + dx
dx = 0
```

之后：

```text
state_in.particle_q.assign(state_out.particle_q)
```

也就是说，非线性迭代直接在状态数组上原地推进当前解。

## 9.7 速度更新

`update_velocity()` 使用：

```text
v = 0.998 * (x - x_prev) / dt
```

这里的 `0.998` 是一个轻微速度衰减系数。

## 10. 碰撞流程

## 10.1 总体结构

`Collision` 的生命周期接口有四个：

- `frame_begin()`
- `accumulate_contact_force()`
- `linear_iteration_end()`
- `frame_end()`

其中当前真正有逻辑的是前两个；后两个现在还是空实现。

## 10.2 `frame_begin()`

### 输入

- `particle_q`
- `particle_qd`
- `dt`

### 输出

- 更新好的 `broad_phase_vf`
- 更新好的 `broad_phase_ee`
- 更新好的 `broad_phase_ef`

### 流程

```text
1. refit triangle BVH
2. refit edge BVH
3. triangle_vs_point -> VF 候选
4. edge_vs_edge -> EE 候选
5. aabb_vs_aabb -> EF 候选
```

注意：

- 这里主要是广相和候选生成。
- 它没有做 CCD safe step，也不是全局 barrier active set。

## 10.3 `accumulate_contact_force()`

### 输入

- `dt`
- `state_in`, `state_out`
- `contacts`
- `particle_forces`
- `particle_q_prev`
- `particle_stiff`

### 输出

- 修改 `particle_forces`
- 修改 `contact_hessian_diags`

### 真实流程

```text
contact_hessian_diags.zero_()

if VF enabled:
    handle_vertex_triangle_contacts_kernel()

if EE enabled:
    handle_edge_edge_contacts_kernel()

if EF enabled:
    solve_untangling_kernel()

eval_body_contact_kernel()
```

## 10.4 顶点-三角形 `VF`

内核：`handle_vertex_triangle_contacts_kernel()`

### 输入

- `thickness`
- `stiff_factor`
- `pos`
- `tri_indices`
- `broad_phase_vf`
- `static_diags`

### 输出

- `forces`
- `hessian_diags`

### 实现思想

```text
1. 对每个顶点读取候选三角形。
2. 计算顶点到三角形平面的法向距离 dist。
3. 计算投影点 barycentric，确保投影点在三角形内部。
4. 若 |dist| < thickness，则产生法向 penalty 力。
5. 把反作用按 barycentric 分摊到三角形三个顶点。
6. Hessian 只保留法向 outer(normal, normal) 形式的对角贡献。
```

刚度取法：

```text
stiff = stiff_factor * harmonic_mean(vertex_stiff, face_stiff)
```

## 10.5 边-边 `EE`

内核：`handle_edge_edge_contacts_kernel()`

### 输入

- `thickness`
- `stiff_factor`
- `pos`
- `edge_indices`
- `broad_phase_ee`
- `static_diags`

### 输出

- `forces`
- `hessian_diags`

### 实现思想

```text
1. 对每条边读取候选边。
2. 用 closest_point_edge_edge() 求最近点参数 s, t。
3. 若最近点落在两条边内部，且距离小于 thickness，则生成法向推离力。
4. 力按 s, t 在两边端点上线性分摊。
5. Hessian 也是法向 outer(dir, dir) 的近似。
```

此外它还做了一个局部厚度裁剪：

- 若两条边共享相邻三角形，会把允许的接触厚度限制到与局部边长相关的更小值。

## 10.6 边-面 `EF` 解缠

内核：`solve_untangling_kernel()`

这个模块更偏向 untangling，而不是普通法向接触。

思路：

- 检查边段是否穿过一个三角形平面。
- 如果交点在三角形内部，则构造一个交线梯度方向 `G`。
- 按该方向施加解缠力。

注释里给出的参考是 Volino 2006 的交线最小化方法。

## 10.7 刚体-粒子接触

内核：`eval_body_contact_kernel()`

它没有重复发明一套接触模型，而是直接调用：

```text
newton._src.solvers.vbd.rigid_vbd_kernels.evaluate_body_particle_contact()
```

也就是说，Style3D 在刚体接触上复用了 VBD 的 body-particle 接触实现。

## 10.8 碰撞模块输入输出总结

### 输入

- 当前位置 `state_in.particle_q`
- 上一步位置 `particle_q_prev`
- 刚体 contact 数据 `contacts.*`
- 固定刚度对角 `particle_stiff` 或 `static_A_diags`

### 输出

- `particle_forces`
- `contact_hessian_diags`
- 可选 `Hx = H_contact * x`

## 11. 关键函数输入输出总览

## 11.1 `solver_style3d.py`

### `SolverStyle3D.__init__(model, iterations, linear_iterations, drag_spring_stiff, enable_mouse_dragging)`

输入：

- 模型和若干求解控制参数

输出：

- 初始化后的 solver 实例

内部做的事：

- 检查 `model.style3d` 是否存在
- 初始化碰撞模块
- 初始化 PCG 求解器
- 预分配 `dx`, `rhs`, `x_prev`, `x_inertia` 等缓冲
- 调用 `_precompute()`

### `step(state_in, state_out, control, contacts, dt)`

输入：

- 当前状态、输出状态、外部 contacts、时间步长

输出：

- `state_out.particle_q`
- `state_out.particle_qd`

内部做的事：

- 初始化惯性项
- 非线性循环
- 线性求解
- 位置推进
- 速度更新

### `register_custom_attributes(builder)`

输入：

- `ModelBuilder`

输出：

- 在 builder 上声明四个 Style3D 自定义属性

属性列表：

- `style3d:tri_aniso_ke`
- `style3d:edge_rest_area`
- `style3d:edge_bending_cot`
- `style3d:aniso_ke`

## 11.2 `cloth.py`

### `_compute_panel_triangles(panel_verts, panel_indices)`

输入：

- 2D panel 顶点
- panel 三角形索引

输出：

- `inv_D`
- `areas`

内部实现：

- 计算 `D = [q-p, r-p]`
- `areas = det(D) / 2`
- 退化面用单位阵兜底
- 返回 `inv(D)` 和 area

### `_compute_edge_bending_data(panel_verts, panel_indices, tri_indices, edge_aniso_ke)`

输入：

- panel 顶点和三角形
- 3D 三角形索引
- 可选边各向异性刚度

输出：

- `edge_indices`
- `edge_ke_values`
- `edge_rest_area`
- `edge_bending_cot`
- `edge_aniso_values`

内部实现：

- 用 `MeshAdjacency` 构造共享边
- 找到两侧三角形和四个局部顶点
- 计算 cot 权重和 edge rest area
- 可选从方向角估算各向异性 edge stiffness

## 11.3 `builder.py`

### `add_stretch_constraints(...)`

输入：

- 三角形索引、rest pose、各向异性刚度、面积

输出：

- 修改内部稀疏结构

实现：

- 对每个三角形计算固定的拉伸 Hessian 权重
- 写入对角和非对角项

### `add_bend_constraints(...)`

输入：

- 边索引、边弯曲属性、rest area、cot 权重

输出：

- 修改内部稀疏结构

实现：

- 构造每条边的 4 点弯曲 Hessian
- 把对角和非对角项累积进稀疏矩阵

### `finalize(device)`

输入：

- 目标 device

输出：

- `diag`
- `num_nz`
- `nz_ell`

## 11.4 `linear_solver.py`

### `PcgSolver.solve(...)`

输入：

- 稀疏 ELL 非对角项
- 对角项
- 初始解 `x0`
- 右端项 `b`
- 预条件器 `inv_M`
- 输出解缓存 `x1`
- 迭代次数
- 可选额外 Hessian 向量积函数

输出：

- 解 `x1`

实现：

- 标准 PCG 六步法
- 支持矩阵自由 `additional_multiplier`

## 12. 仿真流程伪代码

## 12.1 完整 `step()` 伪代码

```python
def step(state_in, state_out, control, contacts, dt):
    if collision:
        collision.frame_begin(state_in.particle_q, state_in.particle_qd, dt)

    init_step_kernel(...)

    if enable_mouse_dragging:
        accumulate_dragging_pd_diag_kernel(...)

    for nl_iter in range(nonlinear_iterations):
        init_rhs_kernel(...)

        eval_stretch_kernel(...)
        eval_bend_kernel(...)

        if enable_mouse_dragging:
            eval_drag_force_kernel(...)

        if collision:
            collision.accumulate_contact_force(...)
            prepare_jacobi_preconditioner_kernel(...)
        else:
            prepare_jacobi_preconditioner_no_contact_hessian_kernel(...)

        linear_solver.solve(
            pd_non_diags,
            static_A_diags,
            dx if nl_iter == 0 else None,
            rhs,
            inv_A_diags,
            dx,
            linear_iterations,
            None if collision is None else collision.hessian_multiply,
        )

        if collision:
            collision.linear_iteration_end(dx)

        nonlinear_step_kernel(state_in.particle_q, state_out.particle_q, dx)
        state_in.particle_q.assign(state_out.particle_q)

    update_velocity(dt, x_prev, state_out.particle_q, state_out.particle_qd)

    if collision:
        collision.frame_end(state_out.particle_q, state_out.particle_qd, dt)
```

## 12.2 碰撞流程伪代码

```python
def frame_begin(particle_q, particle_qd, dt):
    refit triangle_bvh
    refit edge_bvh

    if stiff_vf > 0:
        broad_phase_vf = triangle_vs_point(...)

    if stiff_ee > 0:
        broad_phase_ee = edge_vs_edge(...)

    if stiff_ef > 0:
        broad_phase_ef = aabb_vs_aabb(...)


def accumulate_contact_force(...):
    contact_hessian_diags.zero_()

    if stiff_vf > 0:
        handle_vertex_triangle_contacts_kernel(...)

    if stiff_ee > 0:
        handle_edge_edge_contacts_kernel(...)

    if stiff_ef > 0:
        solve_untangling_kernel(...)

    eval_body_contact_kernel(...)
```

## 13. 算法输入输出总结

## 13.1 算法输入

Style3D 求解器运行时依赖以下输入：

- 位置 `particle_q`
- 速度 `particle_qd`
- 外力 `particle_f`
- 重力 `gravity`
- 质量 `particle_mass`
- 粒子激活标记 `particle_flags`
- 三角形拓扑 `tri_indices`
- panel rest 数据 `tri_poses`, `tri_areas`
- 边拓扑 `edge_indices`
- 弯曲属性 `edge_bending_properties`
- Style3D 自定义属性
  - `tri_aniso_ke`
  - `edge_rest_area`
  - `edge_bending_cot`
  - `aniso_ke`
- 碰撞输入 `contacts`

## 13.2 算法输出

每一步的最终输出是：

- `state_out.particle_q`
- `state_out.particle_qd`

中间输出包括：

- `x_inertia`
- `rhs`
- `dx`
- `contact_hessian_diags`
- `pd_diags`
- `pd_non_diags`

## 14. 这个实现的工程特点

从源码看，当前 Newton Style3D 有这些明显特征：

1. 它是独立于 VBD 的布料求解器，不共享 VBD 的局部块解框架。
2. 它大量依赖 panel 空间 rest 数据，这使它天然适合服装纸样或 panel 建模流程。
3. 它把拉伸和弯曲的固定刚度部分预计算成静态稀疏矩阵，减少每步装配成本。
4. 它采用非线性外循环加 PCG 内循环，而不是一次性精确求解。
5. 它的碰撞是工程 penalty 近似，不是完整 IPC barrier 优化。
6. 它的刚体接触直接复用了 VBD 的 body-particle 接触逻辑。
7. `linear_iteration_end()` 和 `frame_end()` 目前是空接口，说明碰撞后处理或位移限制还有扩展空间。

## 15. 与 Newton VBD 的区别

从工程实现上，Style3D 和 VBD 差异非常明确：

- VBD 以顶点块局部求解为核心。
- Style3D 以固定 PD 矩阵加 PCG 迭代为核心。
- VBD 当前 cloth 主弹性是 membrane stable Neo-Hookean。
- Style3D 当前 stretch 更接近各向异性布料拉伸模型。
- VBD 的接触体系和 ALM/历史状态更复杂。
- Style3D 的接触体系更轻量，更接近单独 cloth solver 的 penalty 处理。

## 16. 总结

如果把当前 Style3D 代码压缩成一句话，它的结构可以写成：

```text
panel-space cloth preprocessing
-> static PD matrix precompute
-> nonlinear implicit cloth iterations
-> PCG linear solve
-> lightweight collision penalty coupling
```

这条链路非常清晰，也说明了它的设计目标：

- 不是追求完整统一多物理框架。
- 而是围绕 cloth 的 stretch、bend、panel 数据和服装建模习惯，做一个相对独立、可控、可扩展的布料求解器。

后续如果继续深挖，最值得扩展的方向通常是：

- 更强的接触安全性
- 更完整的碰撞后处理
- 更细的 cloth material 参数化
- 结合 sewing、panel 和 garment workflow 的高层资产接口