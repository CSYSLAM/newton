# Newton VBD 面向机器人叠衣服优化开发计划

## 1. 文档目的

这份文档不是算法综述，而是一份可直接执行的开发计划。

目标是围绕当前仓库里的 VBD、Style3D、IPC 启发和现有 cloth 示例，逐步把当前布料系统推进到更适合“机器人叠衣服”任务的版本。

这份计划强调：

- 按工程优先级推进，而不是一次性重构。
- 每一步都落到具体文件、函数、数据结构和内核。
- 每一步都能用现有示例验证，不做纯理论改造。
- 第一阶段优先解决接触安全和可控性，而不是先追求更复杂的材料模型。

## 2. 最终目标

最终要支持的不是“布料能掉下来”，而是以下一整套机器人操作链条：

1. 机器人夹爪接近衣服。
2. 夹住衣服的一层或一片区域。
3. 提起、移动、翻折。
4. 放下时层间不明显穿透。
5. 放下后姿态稳定，不剧烈回弹或抖动。

所以真正要优化的不是单一弹性能量，而是：

- 接触安全性
- 自碰撞稳定性
- cloth-rigid 接触可控性
- 多层折叠时的摩擦和接触持续性
- 更像服装的材料方向性
- 对夹取动作友好的任务级约束

## 3. 当前系统现状判断

结合当前代码和两个示例：

- `newton/examples/cloth/example_cloth_franka_mujoco_cloth.py`
- `newton/examples/cloth/example_cloth_franka_mujoco_shirt.py`

可以把现状概括成：

1. 机器人部分已经有可用的刚体和控制链路。
2. 布料部分当前主路径仍然是 `SolverVBD`。
3. 自碰撞已经具备 BVH、VT/EE 窄相、法向势和摩擦近似，但还不是完整 IPC。
4. 无穿透主要依赖 conservative bound + truncation，而不是严格 CCD 安全步长。
5. Style3D 已经提供了 panel-space 和各向异性 stretch / cot bending 的建模前端，但它当前不是机器人叠衣服主路径。

因此最合理的路线不是直接替换成另一个求解器，而是：

- 继续保留 VBD 作为主求解骨架。
- 用 IPC 的思想升级接触安全层。
- 用 Style3D 的建模和材料表达升级 garment 模型。
- 最后补机器人抓取专用的软约束。

## 4. 总体技术路线

整体按四层推进：

### 4.1 第一层：接触安全层 IPC-lite 化

目标：

- 替代仅靠 truncation 的位移接受机制。
- 引入 active set、safe step、轻量 line search。
- 明显减少自碰撞穿透和折叠时的不稳定。

### 4.2 第二层：接触状态持久化

目标：

- 自碰撞和 cloth-rigid 接触不再每轮完全“重新开始”。
- 增加摩擦历史和接触缓存，减少 stick-slip 抖动。

### 4.3 第三层：引入 Style3D 风格服装材料模型

目标：

- 从“普通三角网布”升级到更像服装的 panel-space 材料。
- 支持各向异性 stretch 和 cotangent bending。

### 4.4 第四层：增加机器人抓取专用约束

目标：

- 不再只依赖夹爪摩擦接触来“赌”抓取稳定。
- 为叠衣服任务加入 patch-based grasp 模型。

## 5. 开发边界和原则

整个开发过程遵守以下原则：

1. 不新增外部依赖。
2. 只使用 Warp、NumPy、当前 Newton 内部模块和现有碰撞/几何工具。
3. 第一阶段不删老逻辑，只在老逻辑之外增加新路径和 fallback。
4. 优先保持当前 public API 不变。
5. 每个阶段都要先跑现有示例验证。
6. 能在内部 helper 或 kernel 层改的，不先动外部 API。
7. 不把 VBD 一次性改造成完整全局 IPC Newton 求解器。

## 6. 代码锚点总表

后续计划主要围绕以下文件推进。

### 6.1 VBD 主求解器

- `newton/_src/solvers/vbd/solver_vbd.py`

关键函数：

- `step()`
- `_initialize_particles()`
- `_solve_particle_iteration()`
- `_finalize_particles()`
- `_collision_detection_penetration_free()`
- `_penetration_free_truncation()`
- `rebuild_bvh()`

### 6.2 粒子侧 VBD 内核

- `newton/_src/solvers/vbd/particle_vbd_kernels.py`

关键函数：

- `evaluate_self_contact_force_norm()`
- `evaluate_edge_edge_contact_2_vertices()`
- `evaluate_vertex_triangle_collision_force_hessian_4_vertices()`
- `compute_friction()`
- `compute_particle_conservative_bound()`
- `apply_conservative_bound_truncation()`
- `apply_planar_truncation_parallel_by_collision()`
- `accumulate_self_contact_force_and_hessian()`
- `accumulate_particle_body_contact_force_and_hessian()`
- `solve_elasticity()`
- `solve_elasticity_tile()`
- `evaluate_neo_hookean_membrane_force_hessian()`

### 6.3 自碰撞检测器

- `newton/_src/solvers/vbd/tri_mesh_collision.py`

关键类和函数：

- `TriMeshCollisionDetector`
- `rebuild()`
- `refit()`
- `vertex_triangle_collision_detection()`
- `edge_edge_collision_detection()`

### 6.4 Style3D 可复用前端

- `newton/_src/solvers/style3d/cloth.py`
- `newton/_src/solvers/style3d/kernels.py`

关键函数：

- `add_cloth_mesh()`
- `add_cloth_grid()`
- `_compute_panel_triangles()`
- `_compute_edge_bending_data()`
- `eval_stretch_kernel()`
- `eval_bend_kernel()`

### 6.5 示例和验证场景

- `newton/examples/cloth/example_cloth_franka_mujoco_cloth.py`
- `newton/examples/cloth/example_cloth_franka_mujoco_shirt.py`

## 7. 开发阶段总览

推荐开发顺序：

1. 阶段 0：搭建基线验证和调试指标。
2. 阶段 1：做 IPC-lite 接触安全层。
3. 阶段 2：做接触状态持久化和摩擦历史。
4. 阶段 3：把 Style3D 材料模型接入 VBD。
5. 阶段 4：做机器人抓取 patch 约束。
6. 阶段 5：如有必要，再继续向更完整 IPC 推进。

## 8. 阶段 0：建立基线、日志和验收标准

这一阶段不改物理行为，只搭建后续评估基础。

### 8.1 目标

建立两套固定验证场景和一组最小监控指标，后面每次改动都看这组指标。

### 8.2 目标场景

#### 场景 A：普通布落桌

文件：`newton/examples/cloth/example_cloth_franka_mujoco_cloth.py`

目标：

- 观测 cloth-rigid 接触质量。
- 观测布落桌后是否有明显穿透和抖动。

#### 场景 B：shirt 抓取和半翻折

文件：`newton/examples/cloth/example_cloth_franka_mujoco_shirt.py`

目标：

- 观测多层布料接触稳定性。
- 观测抓取、提起、放下过程中的层间穿透和抖动。

### 8.3 要增加的调试统计量

在 `SolverVBD` 内部增加以下调试量缓存：

- `last_active_vt_count`
- `last_active_ee_count`
- `last_safe_step`
- `last_line_search_alpha`
- `last_max_penetration`
- `last_candidate_vt_count`
- `last_candidate_ee_count`

### 8.4 具体修改位置

文件：`newton/_src/solvers/vbd/solver_vbd.py`

在 `SolverVBD.__init__()` 中新增：

```python
self.last_active_vt_count = 0
self.last_active_ee_count = 0
self.last_safe_step = 1.0
self.last_line_search_alpha = 1.0
self.last_max_penetration = 0.0
self.last_candidate_vt_count = 0
self.last_candidate_ee_count = 0
```

### 8.5 示例层输出内容

在两个 example 中增加每帧或每若干帧日志：

- 当前仿真时间
- cloth-rigid 最小距离
- cloth-cloth 最小距离
- active VT 数量
- active EE 数量
- safe step alpha
- line search alpha
- 最大穿透深度
- 最大粒子位移

### 8.6 阶段 0 验收标准

这一阶段只要求：

1. 两个 example 能正常运行。
2. 日志能稳定输出。
3. 日志字段后续能用于做前后对比。

## 9. 阶段 1：IPC-lite 接触安全层

这是第一优先级，也是整个计划最值得先做的阶段。

## 9.1 目标

把当前 “碰撞检测 + 求解位移 + truncation” 路径升级成：

```text
候选碰撞检测
-> active set 构建
-> 求解位移
-> 计算 safe step
-> 轻量 line search
-> truncation fallback
```

### 9.1.1 不追求完整 IPC

本阶段不做：

- 完整 barrier Newton
- 精确 root-finding CCD
- 全局统一接触能量最小化

本阶段要做的是：

- active set
- linearized conservative advancement
- 接触驱动的位移接受

## 9.2 子阶段 1A：保留现有 detector，仅增加 active set

### 9.2.1 为什么先不改 detector

`TriMeshCollisionDetector` 已经能提供：

- `vertex_colliding_triangles`
- `edge_colliding_edges`
- `*_min_dist`
- BVH 的 `refit()` / `rebuild()`

第一版最稳的做法不是动 detector 的内部结构，而是把它继续当 broad phase + candidate provider。

### 9.2.2 新增数据结构

文件：`newton/_src/solvers/vbd/solver_vbd.py`

在 `SolverVBD.__init__()` 中新增 active set 缓冲。

#### VT active set

```python
self.active_vt_vertex = wp.full(shape=(vt_capacity,), value=-1, dtype=wp.int32, device=self.device)
self.active_vt_tri = wp.full(shape=(vt_capacity,), value=-1, dtype=wp.int32, device=self.device)
self.active_vt_bary = wp.zeros(shape=(vt_capacity,), dtype=wp.vec3, device=self.device)
self.active_vt_normal = wp.zeros(shape=(vt_capacity,), dtype=wp.vec3, device=self.device)
self.active_vt_gap = wp.zeros(shape=(vt_capacity,), dtype=wp.float32, device=self.device)
self.active_vt_pair_key = wp.zeros(shape=(vt_capacity,), dtype=wp.uint64, device=self.device)
self.active_vt_count = wp.zeros(shape=(1,), dtype=wp.int32, device=self.device)
```

#### EE active set

```python
self.active_ee_edge0 = wp.full(shape=(ee_capacity,), value=-1, dtype=wp.int32, device=self.device)
self.active_ee_edge1 = wp.full(shape=(ee_capacity,), value=-1, dtype=wp.int32, device=self.device)
self.active_ee_st = wp.zeros(shape=(ee_capacity,), dtype=wp.vec2, device=self.device)
self.active_ee_normal = wp.zeros(shape=(ee_capacity,), dtype=wp.vec3, device=self.device)
self.active_ee_gap = wp.zeros(shape=(ee_capacity,), dtype=wp.float32, device=self.device)
self.active_ee_pair_key = wp.zeros(shape=(ee_capacity,), dtype=wp.uint64, device=self.device)
self.active_ee_count = wp.zeros(shape=(1,), dtype=wp.int32, device=self.device)
```

#### 容量建议

```python
vt_capacity = model.particle_count * particle_vertex_contact_buffer_size
ee_capacity = model.edge_count * particle_edge_contact_buffer_size
```

第一版不做动态扩容，先按现有 buffer 上界分配。

### 9.2.3 新增 helper

文件：`newton/_src/solvers/vbd/solver_vbd.py`

新增函数：

- `_refresh_particle_active_contacts(self, current_state: State)`

功能：

1. 调用 `_collision_detection_penetration_free(current_state)` 更新候选。
2. 基于候选构建 active VT。
3. 基于候选构建 active EE。
4. 更新 `last_candidate_*` 和 `last_active_*`。

### 9.2.4 新增 active set 构建 kernel

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增两个 kernel：

- `build_active_vt_contacts`
- `build_active_ee_contacts`

#### `build_active_vt_contacts`

输入：

- `pos`
- `tri_indices`
- `collision_info_array`
- `activation_threshold`

输出：

- `active_vt_*`
- `active_vt_count`

内部实现要复用现有几何工具：

- `triangle_closest_point()`

逻辑：

1. 遍历 `vertex_colliding_triangles` 候选。
2. 对每个候选重新计算最近点、barycentric 和法向。
3. gap 记为 `distance - collision_margin`。
4. 只保留 `gap < activation_threshold` 的候选。
5. 生成 `pair_key = (vertex_id << 32) | tri_id`。

#### `build_active_ee_contacts`

输入：

- `pos`
- `edge_indices`
- `collision_info_array`
- `edge_edge_parallel_epsilon`
- `activation_threshold`

输出：

- `active_ee_*`
- `active_ee_count`

内部实现复用：

- `wp.closest_point_edge_edge()`

逻辑：

1. 遍历 `edge_colliding_edges` 候选。
2. 计算最近点参数 `s`、`t`。
3. 计算 gap 和法向。
4. 只保留 `gap < activation_threshold` 的候选。
5. key 取 `min/max(edge0, edge1)` 组合。

### 9.2.5 建议 activation threshold

第一版建议：

```text
activation_threshold = 0.5 * particle_self_contact_margin
```

不要一开始就设成 0，否则 active set 太稀疏，safe step 帮助有限。

## 9.3 子阶段 1B：引入 safe step

### 9.3.1 目标

在位移接受时，先计算一轮全局安全步长 `alpha_safe`，而不是直接走满步再强行 truncation。

### 9.3.2 新增 solver 内部标量缓冲

文件：`newton/_src/solvers/vbd/solver_vbd.py`

在 `__init__()` 增加：

```python
self.safe_step_alpha = wp.array([1.0], dtype=wp.float32, device=self.device)
```

### 9.3.3 新增 kernel

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增：

- `compute_vt_safe_step`
- `compute_ee_safe_step`

#### 核心算法

对每个 active contact row：

1. 已知当前 gap0。
2. 已知 proposed displacement，即 `particle_displacements`。
3. 计算接触法向上的 closing speed `vn`。
4. 若 `vn >= 0`，则该接触不限制步长。
5. 若 `vn < 0`，计算：

```text
alpha = eta * (gap0 - d_safe) / (-vn)
```

6. clamp 到 `[0, 1]`。
7. 所有 row 取全局最小值。

#### 参数建议

- `eta = 0.8 ~ 0.9`
- `d_safe = 0.2 ~ 0.5 * particle_self_contact_radius`

#### VT 中的 `vn` 计算

使用：

- 点位移 `dx_p`
- 三角形最近点位移 `dx_t = bary.x * dx_a + bary.y * dx_b + bary.z * dx_c`

再做：

```text
vn = dot(normal, dx_p - dx_t)
```

#### EE 中的 `vn` 计算

使用最近点参数 `s`, `t`：

```text
dx_c1 = lerp(dx_e0_v0, dx_e0_v1, s)
dx_c2 = lerp(dx_e1_v0, dx_e1_v1, t)
vn = dot(normal, dx_c1 - dx_c2)
```

### 9.3.4 新增 helper

文件：`newton/_src/solvers/vbd/solver_vbd.py`

新增：

- `_compute_particle_safe_step(self)`

功能：

1. `self.safe_step_alpha.fill_(1.0)`
2. launch `compute_vt_safe_step`
3. launch `compute_ee_safe_step`
4. 读取最终 alpha
5. 更新 `last_safe_step`

## 9.4 子阶段 1C：接入 line search

### 9.4.1 目标

safe step 解决“不穿透”；line search 解决“不明显过冲”。

### 9.4.2 不做 full energy Newton，只做过滤式回溯

新增 helper：

- `_line_search_particle_step(self, state_in: State)`

放在：`newton/_src/solvers/vbd/solver_vbd.py`

### 9.4.3 候选 alpha 序列

第一版建议：

```text
[1.0, 0.5, 0.25, 0.125]
```

最终接受步长：

```text
alpha = alpha_safe * alpha_backtrack
```

### 9.4.4 需要新增的能量评估 kernel

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增：

- `evaluate_active_vt_contact_energy`
- `evaluate_active_ee_contact_energy`

第一版只评估接触法向势即可，不把 body-particle 也纳入。

### 9.4.5 merit 条件

一个 alpha 被接受，当且仅当：

1. 最大 penetration 不增。
2. self-contact energy 不增。
3. 没有 NaN 或异常位移。

### 9.4.6 位移应用实现建议

不要新写复杂位置更新逻辑，优先复用现有：

- `apply_truncation_ts`
- `apply_planar_truncation_parallel_by_collision`

新的顺序改成：

```text
1. 先用 safe step alpha 做统一缩步
2. 再做 line search 确认
3. 最后保留 truncation 作为 fallback
```

## 9.5 子阶段 1D：改造 `_penetration_free_truncation()`

### 9.5.1 当前行为

当前 `_penetration_free_truncation()` 做的是：

- 如果无 self-contact，直接用 `apply_truncation_ts`
- 如果有 self-contact，先调用 `apply_planar_truncation_parallel_by_collision`
- 再调用 `apply_truncation_ts`

### 9.5.2 改造后行为

替换为：

```text
refresh active contacts
-> compute safe step
-> line search
-> truncation fallback
```

### 9.5.3 具体重构方式

在 `solver_vbd.py` 中：

新增：

- `_refresh_particle_active_contacts()`
- `_compute_particle_safe_step()`
- `_line_search_particle_step()`
- `_apply_particle_safe_step_and_fallback_truncation()`

然后让 `_penetration_free_truncation()` 成为兼容 wrapper，或者直接把老逻辑重命名成 `_fallback_particle_truncation()`。

## 9.6 子阶段 1E：改造 `_initialize_particles()` 和 `_solve_particle_iteration()`

### 9.6.1 `_initialize_particles()`

文件：`newton/_src/solvers/vbd/solver_vbd.py`

当前流程：

```text
collision detection
-> forward_step
-> penetration_free_truncation
```

改造后流程：

```text
refresh candidates + active set
-> forward_step
-> safe step + line search + truncation fallback
```

### 9.6.2 `_solve_particle_iteration()`

当前关键逻辑：

```text
if self-contact enabled and interval reached:
    _collision_detection_penetration_free()

zero force/hessian
for each color:
    body-particle contact
    springs
    self-contact
    solve_elasticity*
    penetration_free_truncation
```

改造后建议：

```text
if interval reached:
    refresh active contacts

zero force/hessian
for each color:
    body-particle contact
    springs
    self-contact using active rows
    solve_elasticity*
    safe step + line search + fallback truncation
```

### 9.6.3 自碰撞力累积改造

当前直接基于 `collision_info` 遍历候选。

第一版建议不直接改 `accumulate_self_contact_force_and_hessian()`，而是新增 active-row 版本：

- `accumulate_self_contact_force_and_hessian_from_active_set`

然后在 `_solve_particle_iteration()` 中切到新版本。

## 10. 阶段 2：接触状态持久化和摩擦历史

阶段 1 解决的是步长安全。阶段 2 解决的是接触不连续和摩擦抖动。

## 10.1 目标

把接触从“每轮重新构建的一次性候选”升级成“有生命周期的接触状态”。

## 10.2 子阶段 2A：步内 active set 持久化

### 10.2.1 做法

当前 `particle_collision_detection_interval` 已经存在，可以直接利用。

改造方式：

1. 不再让 interval 只控制 candidate refresh。
2. interval 改为同时控制 active set refresh。
3. 两次 refresh 之间，直接复用上一次 active rows。

### 10.2.2 强制刷新条件

新增条件，一旦满足就强制刷新 active set：

- `last_safe_step < 0.3`
- `last_max_penetration` 超过阈值
- 本轮最大位移超过 margin 某比例

## 10.3 子阶段 2B：跨步 active set 持久化

### 10.3.1 新增上一帧缓存

在 `SolverVBD.__init__()` 增加：

- `prev_active_vt_*`
- `prev_active_ee_*`
- `prev_active_vt_count`
- `prev_active_ee_count`

### 10.3.2 新增 helper

在 `solver_vbd.py` 新增：

- `_snapshot_particle_active_contacts()`

在每步末尾调用，把当前 active rows 拷贝到 prev。

### 10.3.3 匹配方式

第一版用 `pair_key` 匹配，不做复杂空间搜索。

匹配成功时继承：

- 上一轮摩擦历史
- 上一轮法向
- 上一轮 bary / s,t

## 10.4 子阶段 2C：摩擦历史

### 10.4.1 当前问题

`compute_friction()` 当前更接近每次重算即时摩擦，没有持久切向状态。

### 10.4.2 新增数据结构

VT 和 EE 各增一组 2D 切向历史：

```python
self.active_vt_friction_u = wp.zeros(shape=(vt_capacity,), dtype=wp.vec2, device=self.device)
self.active_ee_friction_u = wp.zeros(shape=(ee_capacity,), dtype=wp.vec2, device=self.device)
```

以及 prev 对应缓存。

### 10.4.3 新增 evaluator 版本

不要直接改底层几何函数，先新增新版本：

- `evaluate_edge_edge_contact_with_history`
- `evaluate_vertex_triangle_contact_with_history`

输入新增：

- `history_u`

输出新增：

- `updated_history_u`

### 10.4.4 调用点

修改：

- `accumulate_self_contact_force_and_hessian_from_active_set`

逻辑：

1. 取 active row 的 `history_u`
2. 计算当前切向增量 `u_increment`
3. 构造 `u_total = history_u + u_increment`
4. 根据阈值决定 stick / slip
5. 返回 `updated_history_u`
6. 写回 active row 缓冲

## 10.5 子阶段 2D：统一 contact row 数据层

### 10.5.1 目标

减少 self-contact 和 body-particle 接触在数据组织上的分裂。

### 10.5.2 先统一数据，不先统一全部 kernel

第一版新增一个内部 concept：contact row。

字段建议：

- `row_type`
- `pair_key`
- `particle_ids`
- `primitive_ids`
- `bary_or_st`
- `normal`
- `gap`
- `history_index`

第一版可以只在 solver 内部维护，不需要改 public API。

## 11. 阶段 3：把 Style3D 材料模型接进 VBD

这一阶段的重点不是再造建模前端，而是尽量复用 Style3D 已有 panel 处理逻辑。

## 11.1 目标

让 shirt 和未来 garment 不再只是“普通三角形布”，而是有：

- panel-space rest 数据
- 各向异性拉伸
- cotangent bending
- seam / sewing 结构

## 11.2 子阶段 3A：先复用 Style3D 的建模前端

### 11.2.1 为什么不自己重写 panel preprocessing

`style3d/cloth.py` 已经提供：

- `_compute_panel_triangles()`
- `_compute_edge_bending_data()`
- `add_cloth_mesh()`
- `add_cloth_grid()`

直接复用它比重写更稳。

### 11.2.2 示例层改法

在 `newton/examples/cloth/example_cloth_franka_mujoco_shirt.py` 中：

1. 在 builder 创建后调用 `SolverStyle3D.register_custom_attributes(self.scene)`。
2. 不再用 `self.scene.add_cloth_mesh()` 添加 shirt。
3. 改用 `newton.solvers.style3d.add_cloth_mesh()` 添加 shirt。
4. 求解器仍然使用 `SolverVBD`。

### 11.2.3 这样会带来的变化

模型上将拥有：

- `model.style3d.tri_aniso_ke`
- `model.style3d.edge_rest_area`
- `model.style3d.edge_bending_cot`

这些属性后面由 VBD 消费。

## 11.3 子阶段 3B：在 VBD 中新增 Style3D stretch 分支

### 11.3.1 当前膜能量入口

当前三角形膜能量主要入口：

- `evaluate_neo_hookean_membrane_force_hessian()`

### 11.3.2 新增函数

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增：

- `evaluate_style3d_stretch_force_hessian`

### 11.3.3 算法来源

直接参考：

- `newton/_src/solvers/style3d/kernels.py::eval_stretch_kernel`

### 11.3.4 输入

- `face`
- `v_order`
- `pos`
- `pos_anchor`
- `tri_indices`
- `tri_pose`
- `area`
- `tri_aniso_ke`
- `damping`
- `dt`

### 11.3.5 输出

- `force`
- `hessian`

### 11.3.6 第一版 Hessian 策略

第一版不追求严格二阶一致，优先保证 SPD 和稳定。

建议：

1. 力完全按 Style3D stretch 公式算。
2. Hessian 用 rank-1 + diagonal clamp 近似，或者直接沿用 PD builder 的局部固定权重近似。

目标是先让：

- 方向性 stretch 生效
- 求解不炸

### 11.3.7 分流入口

修改：

- `solve_elasticity()`
- `solve_elasticity_tile()`

逻辑：

```text
if model has style3d.tri_aniso_ke:
    use evaluate_style3d_stretch_force_hessian
else:
    use evaluate_neo_hookean_membrane_force_hessian
```

## 11.4 子阶段 3C：在 VBD 中新增 Style3D bend 分支

### 11.4.1 新增函数

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增：

- `evaluate_style3d_bend_force_hessian`

### 11.4.2 算法来源

参考：

- `newton/_src/solvers/style3d/kernels.py::eval_bend_kernel`

### 11.4.3 输入

- `edge_index`
- `vertex_order_on_edge`
- `pos`
- `pos_anchor`
- `edge_indices`
- `edge_rest_area`
- `edge_bending_cot`
- `edge_bending_properties`
- `dt`

### 11.4.4 输出

- `force`
- `hessian`

### 11.4.5 分流入口

同样改：

- `solve_elasticity()`
- `solve_elasticity_tile()`

逻辑：

```text
if model has style3d.edge_rest_area and style3d.edge_bending_cot:
    use Style3D bend evaluator
else:
    use current dihedral bend evaluator
```

## 11.5 子阶段 3D：sewing 支持

### 11.5.1 先不做复杂 seam 约束

先复用：

- `style3d/cloth.py::sew_close_vertices`

让 shirt 具备 seam spring。

### 11.5.2 后续扩展方向

如果未来需要更强 seam，可以再追加：

- seam patch constraint
- seam tearing / release

第一版不做。

## 12. 阶段 4：机器人抓取 patch 约束

即使接触和材料都升级，只靠夹爪摩擦接触，抓取叠衣服依然会脆弱。

所以必须引入 grasp patch 软约束。

## 12.1 目标

让机器人抓取动作从“赌摩擦”升级成“受控 patch 约束”。

## 12.2 子阶段 4A：最小 patch 选取

### 12.2.1 第一版 patch 定义

用下列二选一：

1. 三角形 + barycentric
2. 一组粒子 id + 权重

建议第一版用第二种更直接。

### 12.2.2 patch 生成方式

在 `example_cloth_franka_mujoco_shirt.py` 中：

1. 用夹爪目标点找到最近三角形。
2. 取该三角形的 3 个顶点。
3. 若不够稳，再扩展到 1-ring 邻域。
4. 权重按距离衰减。

## 12.3 子阶段 4B：新增 grasp force kernel

文件：`newton/_src/solvers/vbd/particle_vbd_kernels.py`

新增：

- `accumulate_grasp_patch_force_and_hessian`

### 输入

- patch particle ids
- patch weights
- target position or target transform
- grasp stiffness
- grasp damping
- current positions
- previous positions
- dt

### 输出

- `particle_forces`
- `particle_hessians`

### 模型

第一版直接用 weighted spring patch：

```text
p_patch = sum_i w_i * x_i
f_patch = k * (x_target - p_patch) + d * (v_target - v_patch)
```

再按权重分配到粒子。

## 12.4 子阶段 4C：接入 `_solve_particle_iteration()`

把 grasp force accumulation 插在：

- body-particle contact 之后
- spring 之后
- self-contact 之前或之后都可

推荐顺序：

```text
body-particle contact
-> spring
-> grasp patch
-> self-contact
-> elasticity solve
```

## 12.5 子阶段 4D：释放曲线

不要瞬间释放。

在 example 中把 grasp stiffness 按若干 substep 平滑衰减，例如：

```text
1.0 -> 0.7 -> 0.4 -> 0.2 -> 0.1 -> 0.0
```

## 13. 阶段 5：如有必要，再继续 IPC 化

只有当完成前四阶段后，仍然明显存在这些问题，再推进：

- 高速薄层穿透
- 强折叠时反转
- 多层接触中来回卡穿

那时再做：

1. 更精确 CCD
2. 更统一 barrier energy
3. 更完整 active set 生命周期
4. cloth-rigid 的更一致 barrier 处理

这一步不作为第一期开发目标。

## 14. 逐阶段实施顺序

推荐真实开发顺序如下。

### 顺序 1

先做阶段 0，只搭建验证和日志。

### 顺序 2

做阶段 1：

- active set
- safe step
- line search
- truncation fallback

先不改材料模型。

### 顺序 3

做阶段 2：

- 步内持久化
- 跨步缓存
- 摩擦历史

### 顺序 4

做阶段 3：

- Style3D 前端建模接入
- VBD Style3D stretch/bend 分支

### 顺序 5

做阶段 4：

- grasp patch
- release smoothing

## 15. 每一阶段的最小验收标准

## 15.1 阶段 1 验收

在 cloth 和 shirt 两个例子里：

1. 最大穿透显著下降。
2. safe step 在强折叠时能明显小于 1。
3. 没有大规模数值爆炸。
4. fallback truncation 仍能兜底。

## 15.2 阶段 2 验收

1. 折叠后接触开闭抖动减少。
2. 层间 stick-slip 变化更平滑。
3. 接触数量和摩擦响应不再每轮剧烈跳变。

## 15.3 阶段 3 验收

1. shirt 比普通 cloth 更接近服装材料表现。
2. warp/weft 方向变形差异可观测。
3. 不引入明显的新不稳定性。

## 15.4 阶段 4 验收

1. 夹取成功率明显提升。
2. 放下时不会因为突然释放而大幅弹飞。
3. 翻折动作更可控。

## 16. 第一周建议只做什么

如果时间有限，第一周只建议做这四件事：

1. 阶段 0 的日志和基线。
2. active set。
3. safe step。
4. line search + truncation fallback。

不要第一周就碰：

- Style3D 材料接入
- grasp patch
- 全局 IPC barrier

## 17. 结论

如果你最终目标是“机器人叠衣服”，最值得先做的不是换求解器，而是按下面这条路线推进：

```text
保留 VBD 主骨架
-> 用 IPC 思想升级接触安全层
-> 用 Style3D 升级服装材料前端
-> 用 grasp patch 升级机器人抓取
```

这条路线的优点是：

1. 改动可以分阶段验证。
2. 与当前示例和刚体链路最连续。
3. 风险比“全量切换到完整 IPC 或 Style3D 主求解器”低很多。
4. 每一步都有明确的工程收益。

如果后续继续扩写，本计划最适合再拆成三篇子文档：

1. 第一阶段函数级改造清单。
2. Style3D 材料接入设计文档。
3. 机器人抓取 patch 设计文档。