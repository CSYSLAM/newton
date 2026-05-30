# Newton VBD 源码分析与 IPC 启发

项目路径：`C:\csy_work\CG\Engine\newton`

核心源码：

- `newton/_src/solvers/vbd/solver_vbd.py`
- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
- `newton/_src/solvers/vbd/rigid_vbd_kernels.py`
- `newton/_src/solvers/vbd/tri_mesh_collision.py`

相关用户入口：

- `newton/solvers.py`
- `newton/examples/cloth/`
- `newton/examples/softbody/`
- `newton/examples/cable/`

## 1. 定位

当前 `newton` 里的 `SolverVBD` 不是纯布料求解器，而是一个混合求解器：

- 粒子域：VBD
- 刚体域：AVBD
- 粒子-刚体耦合：共享接触参数，双侧分别累积
- 粒子自碰撞：BVH + 近邻查询 + 力/Hessian 累积 + 位移截断

源码文档里已经明确它的三类职责：

- 粒子仿真
- 刚体仿真
- 粒子-刚体耦合

见 `solver_vbd.py` 中 `SolverVBD` 类说明。

## 2. 总体架构

可以把当前实现拆成 4 层：

### 2.1 调度层

文件：`solver_vbd.py`

职责：

- 初始化粒子和刚体状态
- 构建或刷新接触状态
- 组织每步 `iterations` 次迭代
- 维护跨步 warm-start 历史
- 回写速度和黏滞状态

### 2.2 粒子求解层

文件：`particle_vbd_kernels.py`

职责：

- 前向惯性预测
- 三角形膜、边弯曲、四面体体积弹性局部力/Hessian
- 粒子-刚体接触力/Hessian
- 粒子自碰撞力/Hessian
- 每顶点 3x3 局部块求解

### 2.3 刚体求解层

文件：`rigid_vbd_kernels.py`

职责：

- 刚体前向积分
- 关节约束和刚体接触的力/Hessian 累积
- 每刚体 6x6 局部块求解
- AVBD penalty / lambda / C0 的跨步和步内更新

### 2.4 自碰撞检测层

文件：`tri_mesh_collision.py`

职责：

- 为三角形和边构建 BVH
- refit / rebuild
- 顶点-三角形、边-边候选检测
- 管理碰撞缓存结构 `TriMeshCollisionInfo`

## 3. 主通路

`SolverVBD.step()` 的主流程非常清晰：

```text
输入:
  state_in, state_out, control, contacts, dt

1. 初始化刚体
   - 刷新 rigid-rigid / body-particle 接触状态
   - 恢复 rigid warm-start 历史
   - 计算接触 C0 / lambda 衰减 / penalty 衰减
   - 前向积分刚体到 inertial target

2. 初始化粒子
   - 若开启自碰撞，先做一次 BVH refit + 窄相检测
   - 计算 inertia target 和初始位移
   - 对初始位移做 penetration-free truncation

3. 做 iterations 次迭代
   - 先做一轮刚体 AVBD
   - 再做一轮粒子 VBD

4. 结束处理
   - 保存 rigid contact history
   - 由位置差分回写刚体/粒子速度
   - 更新 cable 的 Dahl friction 状态
```

对应实现：

- `solver_vbd.py:1551` `step()`
- `solver_vbd.py:1744` `_initialize_rigid_bodies()`
- `solver_vbd.py:1707` `_initialize_particles()`
- `solver_vbd.py:2127` `_solve_particle_iteration()`
- `solver_vbd.py:2315` `_solve_rigid_body_iteration()`
- `solver_vbd.py:2737` `_finalize_particles()`
- `solver_vbd.py:2750` `_finalize_rigid_bodies()`

## 4. 数据链路

## 4.1 粒子链路

```text
model.{tri,edge,tet,spring,particle_*}
-> _compute_particle_force_element_adjacency()
-> particle color groups
-> forward_step() 得到 inertia / displacement
-> accumulate_*_force_and_hessian()
-> solve_elasticity() / solve_elasticity_tile()
-> particle_displacements
-> penetration_free_truncation()
-> state_out.particle_q
-> update_velocity()
```

输入：

- 粒子位置 `state_in.particle_q`
- 粒子速度 `state_in.particle_qd`
- 外力 `state_in.particle_f`
- 拓扑和材料参数
- 自碰撞候选
- 粒子-刚体接触

输出：

- 更新后位置 `state_out.particle_q`
- 更新后速度 `state_out.particle_qd`
- 中间缓存 `particle_displacements / particle_forces / particle_hessians`

## 4.2 刚体链路

```text
contacts.rigid_contact_* + model joints/materials
-> build per-body contact lists
-> init_body_body_contacts_avbd() / init_body_particle_contacts()
-> step_*_C0_lambda()
-> accumulate_body_* contacts + joint terms
-> solve_rigid_body()
-> update_duals_*
-> snapshot_body_body_contact_history()
-> update_body_velocity()
```

输入：

- 刚体位姿速度
- 刚体接触集合
- 关节拓扑 / target / limit
- 关节硬软模式

输出：

- 更新后的刚体位姿速度
- 每接触 `penalty_k / lambda / C0`
- 每关节 `penalty_k / lambda / C0`

## 5. 粒子域 VBD 实现

## 5.1 邻接构建

VBD 的前提是“按顶点局部块求解”，因此必须先把每个顶点关联的力元收集好。

`_compute_particle_force_element_adjacency()` 会在 CPU 上构造 CSR 风格的邻接：

- 顶点-边
- 顶点-面
- 顶点-四面体
- 顶点-弹簧

输出结构：

- `v_adj_edges`, `v_adj_edges_offsets`
- `v_adj_faces`, `v_adj_faces_offsets`
- `v_adj_tets`, `v_adj_tets_offsets`
- `v_adj_springs`, `v_adj_springs_offsets`

这一步决定了后续每个顶点局部块会遍历哪些弹性能量项。

对应实现：

- `solver_vbd.py:1153`

## 5.2 颜色分组

当前 VBD 强依赖图着色：

- 粒子要有 `particle_color_groups`
- 刚体要有 `body_color_groups`

原因不是理论上的必须，而是当前并行实现采用“同色并行、异色顺序”的 Gauss-Seidel 风格更新，避免相邻自由度同时写同一局部状态。

这也是当前 VBD 的一个核心工程特征：

- 它不是全局线性系统
- 它也不是完全 Jacobi
- 而是 colored block Gauss-Seidel

对应检查：

- `solver_vbd.py:529`
- `solver_vbd.py:765`

## 5.3 粒子预测

`forward_step()` 做了非常直接的惯性预测：

```text
vel_new = vel + (gravity + f_ext / m) * dt
inertia = pos + vel_new * dt
displacement = vel_new * dt
```

它相当于给后续局部块一个 inertial target。

对应实现：

- `particle_vbd_kernels.py:1783`

输入：

- `dt`
- `gravity`
- `pos`
- `vel`
- `external_force`
- `inv_mass`

输出：

- `inertia_out`
- `displacements_out`

## 5.4 粒子每轮迭代

每轮 `_solve_particle_iteration()` 的结构是：

```text
如果需要:
  刷新自碰撞检测

清空 particle_forces / particle_hessians

for color in particle_color_groups:
  1. 累积粒子-刚体接触力/Hessian
  2. 累积弹簧力/Hessian
  3. 累积粒子自碰撞力/Hessian
  4. 累积三角膜/边弯曲/四面体弹性项并解每顶点 3x3 块
  5. 对更新后位移做截断，避免穿透

最后:
  拷贝到 state_out.particle_q
```

对应实现：

- `solver_vbd.py:2127`

## 5.5 每顶点块求解

`solve_elasticity()` 和 `solve_elasticity_tile()` 本质做同一件事：

- 从惯性项得到初始 `f, H`
- 对每个邻接三角形累积膜力/Hessian
- 对每个邻接边累积弯曲力/Hessian
- 对每个邻接四面体累积体积力/Hessian
- 加上接触项已预先写入的 `particle_forces / particle_hessians`
- 解一个 3x3 局部系统，更新该顶点位移

其局部形式可以写成：

```text
H_total * dx = f_total

其中
H_total = M / dt^2 + H_elastic + H_contact
f_total = M / dt^2 * (x_tilde - x) + f_elastic + f_contact
```

伪代码：

```text
for vertex i in current_color:
    f = m_i / dt^2 * (x_tilde_i - x_i)
    H = m_i / dt^2 * I

    for each adjacent triangle:
        f += f_tri_i
        H += H_tri_i

    for each adjacent edge:
        f += f_bend_i
        H += H_bend_i

    for each adjacent tet:
        f += f_tet_i
        H += H_tet_i

    f += particle_forces[i]
    H += particle_hessians[i]

    if det(H) != 0:
        dx = inv(H) * f
        displacement[i] += dx
```

对应实现：

- `particle_vbd_kernels.py:2971`
- `particle_vbd_kernels.py:3136`

## 6. 粒子自碰撞实现

## 6.1 检测结构

`TriMeshCollisionDetector` 为：

- 三角形构建一棵 BVH
- 边构建一棵 BVH
- 为每个顶点维护 vertex-triangle 冲突缓冲
- 为每条边维护 edge-edge 冲突缓冲

结果由 `TriMeshCollisionInfo` 统一打包：

- `vertex_colliding_triangles*`
- `triangle_colliding_vertices*`
- `edge_colliding_edges*`

对应实现：

- `tri_mesh_collision.py:19`
- `tri_mesh_collision.py:90`

## 6.2 检测流程

当前自碰撞通路是：

```text
particle_q
-> refit BVH
-> vertex_triangle_collision_detection()
-> edge_edge_collision_detection()
-> collision_info
-> accumulate_self_contact_force_and_hessian()
```

检测接口：

- `refit()`
- `vertex_triangle_collision_detection()`
- `edge_edge_collision_detection()`
- 需要时 `rebuild()`

对应实现：

- `tri_mesh_collision.py:283`
- `tri_mesh_collision.py:308`
- `tri_mesh_collision.py:333`
- `tri_mesh_collision.py:385`
- `solver_vbd.py:2808`

## 6.3 过滤机制

当前已经有一套相当实用的过滤：

- n-ring 拓扑过滤
- 外部提供的过滤 map
- rest-shape 距离过滤

这说明当前实现已经意识到“不是所有几何近邻都应该进入接触求解”。

对应实现：

- `solver_vbd.py:1315`

## 6.4 自碰撞力模型

当前粒子自碰撞不是典型 IPC 全局 barrier Newton，而是“局部接触力 + Hessian 近似”。

核心是 `evaluate_self_contact_force_norm()`：

- 中距离：二次 penalty
- 更近距离：切换到 `-log(d)` 风格 barrier
- 极近距离：再用二次外推保持连续性

它是一个分段模型：

```text
if tau > d > d_min:
    E ~ -k * tau^2 * log(d)
elif d <= d_min:
    用 barrier 在 d_min 处做二阶延拓
else:
    E ~ 0.5 * k * (r - d)^2
```

所以它已经吸收了一部分 IPC 的 barrier 思想，但实现方式仍然是局部接触近似，而不是全局 barrier energy 最小化。

对应实现：

- `particle_vbd_kernels.py:1194`

## 6.5 自碰撞响应

窄相支持两类 primitive：

- edge-edge
- vertex-triangle

对应函数：

- `evaluate_edge_edge_contact_2_vertices()`
- `evaluate_vertex_triangle_collision_force_hessian_4_vertices()`

它们都会输出：

- 是否发生接触
- 各参与顶点的力
- 各参与顶点的 3x3 Hessian

然后 `accumulate_self_contact_force_and_hessian()` 再按颜色把这些贡献原子加到对应顶点。

对应实现：

- `particle_vbd_kernels.py:1372`
- `particle_vbd_kernels.py:1600`
- `particle_vbd_kernels.py:1931`

## 6.6 穿透避免策略

当前 VBD 的关键安全机制不是 CCD，而是 truncation。

流程是：

1. 用当前状态做一次碰撞检测
2. 从最近碰撞距离估算 conservative bound
3. 把位移裁到允许区间内

当前实现里最终生效的是：

- 无自碰撞：直接按 `max_displacement` 截断
- 有自碰撞：先按碰撞对平面截断，再按各向同性阈值二次截断

对应实现：

- `solver_vbd.py:1645`
- `particle_vbd_kernels.py:1811`
- `particle_vbd_kernels.py:1877`
- `particle_vbd_kernels.py:2745`

这一步非常重要，因为它说明当前方案的“无穿透”主要依赖几何截断，而不是 IPC 的连续时间安全步长。

## 7. 刚体域 AVBD 实现

## 7.1 前向积分

刚体初始化阶段会先做一次显式/半隐式积分，把 `body_q` 推到 inertial target：

```text
q_new, qd_new = integrate_rigid_body(...)
body_q = q_new
body_qd = qd_new
body_inertia_q = q_new
```

对应实现：

- `rigid_vbd_kernels.py:1817`

## 7.2 接触状态

刚体接触状态不是临时量，而是持久状态机：

- `contact_penalty_k`
- `contact_lambda`
- `contact_C0`
- `contact_stick_flag`

如果开启 `rigid_contact_history`，还会跨帧保存：

- 上一帧的 `lambda`
- `penalty_k`
- contact anchors
- normal

然后通过 `match_index` 做 warm-start 恢复。

对应实现：

- `solver_vbd.py:1600`
- `solver_vbd.py:1824`
- `rigid_vbd_kernels.py:2113`
- `rigid_vbd_kernels.py:2204`

## 7.3 刚体接触初始化

`init_body_body_contacts_avbd()` 会做几件事：

- 根据两个 shape 平均材料参数
- 生成或恢复 penalty stiffness
- 若为 hard contact，恢复 lambda
- 若之前处于 sticking，还恢复接触锚点

这部分已经非常像 ALM/IPC 接触状态管理器了。

对应实现：

- `rigid_vbd_kernels.py:2113`

## 7.4 C0 稳定项

AVBD 的硬接触和硬关节都维护 `C0`：

- contact 侧：`contact_C0`
- joint 侧：`joint_C0_lin / joint_C0_ang`

其作用不是几何检测，而是把上一状态的约束偏移带入本步，构造稳定的 augmented term。

对应实现：

- `rigid_vbd_kernels.py:1980`
- `rigid_vbd_kernels.py:2245`

## 7.5 刚体每轮迭代

每轮 `_solve_rigid_body_iteration()` 做：

```text
if 没有刚体求解:
    只更新 body-particle soft contact penalty
    return

清空每体 force / torque / Hessian

for color in body_color_groups:
    1. 累积 body-particle 接触
    2. 累积 rigid-rigid 接触
    3. 累积 joint 项
    4. 解每刚体 6x6 局部块

之后:
    update_duals_body_body_contacts()
    update_duals_body_particle_contacts()
    update_duals_joint()
```

对应实现：

- `solver_vbd.py:2315`

## 7.6 局部块求解

刚体不是全局牛顿系统，而是每个刚体单独求一个 6x6 局部块：

- 平移 3 自由度
- 旋转 3 自由度

局部未知量可以看成：

```text
delta = [delta_x, delta_theta]
```

而 Hessian 分成：

- `H_ll`
- `H_al`
- `H_aa`

这和粒子顶点的 3x3 块求解是一致思路，只是刚体块更大。

对应实现入口：

- `rigid_vbd_kernels.py:2948`

## 7.7 Dual 更新

AVBD 的一个核心是“解位置块”和“更新 dual / penalty”分开做。

### body-body

`update_duals_body_body_contacts()`：

- 更新法向 `lambda_n`
- 更新切向 `lambda_t`
- 做摩擦锥截断
- 更新 `stick_flag`
- 按 penetration 增大 `penalty_k`

对应实现：

- `rigid_vbd_kernels.py:3616`

### body-particle

`update_duals_body_particle_contacts()`：

- 只有 penalty ramp
- 没有 persistent lambda

对应实现：

- `rigid_vbd_kernels.py:3743`

### joint

`update_duals_joint()`：

- 按 joint type 更新不同 slot
- drive / limit 与 structural slot 分离
- 每个 slot 有独立 penalty ceiling 和 beta ramp

对应实现：

- `rigid_vbd_kernels.py:3208`

## 8. 函数级输入输出摘要

## 8.1 `SolverVBD.step()`

输入：

- `state_in`
- `state_out`
- `control`
- `contacts`
- `dt`

输出：

- 原地更新 `state_out`
- 更新内部 persistent state

## 8.2 `forward_step()`

输入：

- 当前位置速度、质量、外力、重力、`dt`

输出：

- `inertia_out`
- `displacements_out`

## 8.3 `accumulate_self_contact_force_and_hessian()`

输入：

- 当前位置/锚点位置
- collision info
- 接触参数

输出：

- `particle_forces`
- `particle_hessians`

## 8.4 `solve_elasticity()`

输入：

- 当前颜色顶点列表
- 弹性拓扑材料
- 已累积的接触力/Hessian

输出：

- `particle_displacements`

## 8.5 `init_body_body_contacts_avbd()`

输入：

- rigid contact 集合
- shape 材料
- match history

输出：

- `contact_penalty_k`
- `contact_lambda`
- `contact_material_*`
- 必要时覆写 contact anchors

## 8.6 `update_duals_body_body_contacts()`

输入：

- 当前接触几何
- 当前/前一帧刚体位姿
- `contact_C0`
- `contact_lambda`
- `contact_penalty_k`

输出：

- 更新后的 `lambda`
- 更新后的 `penalty_k`
- `stick_flag`

## 9. 当前 VBD 的特点总结

从源码看，当前 `newton` VBD/AVBD 有这些鲜明特征：

1. 不是全局矩阵法，而是 colored local block solve。
2. 粒子域和刚体域都把问题拆成局部块，但接触状态管理分开。
3. 粒子自碰撞已经借用了 barrier 风格法向势能，但不是完整 IPC。
4. 粒子侧防穿透主要依赖 truncation，而不是 CCD safe step。
5. 刚体侧已经有比较成熟的 ALM 状态管理，包括 `lambda/C0/history/stick`。
6. 粒子-刚体接触和刚体-刚体接触已经共享了不少参数组织方式。

## 10. IPC 对当前 VBD 的启发

下面不是泛泛而谈，而是直接对照当前代码结构看“能改什么”。

## 10.1 启发一：把“截断”升级成真正的连续安全步长

当前粒子自碰撞安全性主要靠：

- 先检测
- 再估 conservative bound
- 再截断位移

这比纯 penalty 稳定，但仍然是启发式的。它没有严格回答：

- 两次检测之间会不会漏穿
- 当前下降方向的最大安全步长是多少

IPC 的直接启发是：

- 不要只做 truncation
- 为当前位移方向 `dx` 计算一个 `t_safe`
- 让顶点块更新后走 `x <- x + t_safe * dx`

可落地方案：

1. 保留当前 VBD 的局部块方向 `dx`
2. 在每个 color sweep 后，对所有潜在 VT/EE 对做一次简化 CCD
3. 取全局最小 `t_safe`
4. 用这个 `t_safe` 缩放本轮位移更新

这样做不会破坏 VBD 的块结构，但可以显著减轻：

- 高速布料折叠穿透
- 厚度较小时的漏碰
- 对 `particle_collision_detection_interval` 的敏感性

## 10.2 启发二：把粒子自接触从“局部力项”升级成“显式 active set”

当前粒子自接触数据结构是碰撞缓冲：

- 顶点有哪些碰撞三角形
- 边有哪些碰撞边

但它缺少 IPC 常见的 active set 概念：

- 哪些接触对本轮真正进入求解
- 这些接触对是否跨迭代/跨步持久
- 接触对的状态是否已稳定

IPC 的启发是：

- 给粒子自接触也做一层 active set 管理
- 把检测集和求解集分开

可落地方案：

```text
BVH 窄相候选
-> filter
-> active set build / merge / persist
-> VBD 接触力/Hessian累积
```

收益：

- 减少每轮重复扫描全部候选
- 为 warm-start、摩擦历史、接触寿命统计提供入口
- 为后续 CCD 和 line search 提供统一接口

## 10.3 启发三：把粒子接触也纳入“history + warm-start”

当前 rigid contact 已经有：

- `match_index`
- `history`
- `lambda`
- `stick_flag`
- anchor replay

而粒子自碰撞与 body-particle 接触基本没有同等级的 persistent state。

IPC 系方案的启发是：

- 接触不是每轮重新发明
- 需要稳定的 correspondence

对当前 VBD 的建议：

1. 先给粒子-刚体接触加 `match_index`
2. 再给粒子自碰撞加接触行缓存
3. 缓存内容至少包括：
   - primitive pair id
   - 最近点重心坐标
   - 法向
   - 法向接触强度或等价 penalty 状态
   - 摩擦切向历史

这会直接改善：

- 摩擦抖动
- 接触开闭震荡
- 折叠后接触对重建带来的不连续

## 10.4 启发四：把粒子接触法向从“局部固定法向”推进到“统一 barrier 微分”

当前粒子接触里已经有一句很关键的注释：

- 摩擦部分“Different from IPC, we treat the contact normal as constant”

这说明当前实现为了稳定性，主动放弃了 IPC 那种更一致的几何微分。

这不是错，但有代价：

- Hessian 更稳
- 几何一致性更弱
- 大变形或强折叠下接触方向误差更大

IPC 的启发不是立刻全量替换，而是分层推进：

1. 先保留现有 constant-normal friction
2. 只把法向 barrier 的 `dE/dd`、`d2E/dd2` 与几何距离导数做得更一致
3. 再视情况决定是否升级切向项

也就是说，先做“法向 IPC 化”，不急着做“全接触 IPC 化”。

## 10.5 启发五：把当前分段接触势能整理成统一 barrier 家族

当前 `evaluate_self_contact_force_norm()` 是：

- 外层 penalty
- 中层 log barrier
- 内层二次延拓

它其实已经很接近一个工程化 IPC barrier，但是分段逻辑仍然偏手工。

IPC 的启发是把它正规化：

- 明确 barrier 激活距离
- 明确安全距离 `d_hat`
- 明确 Hessian regularization 规则
- 明确法向力和摩擦使用同一套接触尺度

好处：

- 参数解释更统一
- 更容易跨项目对比
- 后续引入 line search / CCD 更顺

## 10.6 启发六：在 VBD 外层加一个轻量线搜索

当前粒子顶点块和刚体块都是“直接吃掉局部解”：

- 先解 `dx`
- 再截断

这意味着下降方向未必总是最合适的接受步长。

IPC 项目几乎都会有：

- filter line search
- max step size
- barrier-safe acceptance

对当前 VBD，一个轻量级版本就够了：

```text
局部块给出 dx
-> 评估接触安全性
-> 若能量/穿透变差，则缩步
-> 接受 x + t * dx
```

这样能减少：

- 过冲
- 截断后方向畸变
- penalty 很大时的锯齿振荡

## 10.7 启发七：统一 body-particle 与 self-contact 的接触后端

现在粒子域接触至少分成两套：

- 粒子自碰撞
- body-particle 接触

二者都在输出：

- 力
- Hessian
- 摩擦项

但数据结构和状态管理并不统一。

IPC 系框架，尤其 `libuipc`，给出的启发是：

- 统一 contact primitive schema
- 统一 narrow phase 输出
- 统一求解期 contact row

对 `newton` 的实际意义是：

- 可以抽象一层粒子域 contact row
- VT / EE / particle-body 都转成统一接触行
- VBD 只消费统一接触行

这样后面加：

- warm-start
- active set
- CCD
- line search

都会简单很多。

## 10.8 启发八：从“每轮重检”升级成“全局轨迹过滤 + 局部刷新”

当前粒子碰撞检测频率由：

- `particle_collision_detection_interval`

控制。

这是很直接的做法，但不够自适应。

IPC 和 `libuipc` 一类实现的启发是：

- 用 trajectory filter 或 conservative bound 管理候选寿命
- 不是每次都全量重检
- 也不是固定每 `k` 轮重检

更好的方向是：

1. 先缓存 active set
2. 根据位移幅度、BVH 盒变化、候选对距离变化决定局部刷新
3. 全局只在必要时 full refit / full narrow phase

这样会更适合大型 cloth/softbody。

## 10.9 启发九：把 VBD 看成 IPC 外层求解器，而不是 IPC 的对立面

最重要的一点是，不要把两者理解成替代关系：

- VBD 擅长局部块求解、并行颜色更新、结构简单
- IPC 擅长接触安全性、活动集管理、全局 barrier 解释

完全可以形成组合：

```text
弹性能 + 惯性能:
  继续用 VBD 局部块解

接触安全性:
  借 IPC 的 active set / CCD / line search / barrier 参数化
```

也就是说，合理路线不是“把 Newton VBD 改成完整 IPC Newton 求解器”，而是：

- 保留 VBD 作为 block solver
- 把 IPC 作为 contact safety layer

这是对当前代码侵入最小、收益最大的方向。

## 11. 针对当前代码的优先改进建议

如果按工程优先级排序，我建议是：

1. 给粒子更新增加基于 VT/EE 的 `t_safe` 计算，替代纯 truncation。
2. 给粒子自碰撞引入 active set 和 contact row 缓存。
3. 给 body-particle 和 self-contact 加粒子侧 warm-start 状态。
4. 在 VBD 外层增加轻量线搜索，而不是更新后再硬截断。
5. 统一粒子域接触后端，减少 VT/EE/body-particle 三套逻辑的分裂。

## 12. 一句话结论

当前 `newton` 的 VBD/AVBD 已经具备很强的工程骨架：

- colored block solver
- 粒子/刚体双域统一
- rigid contact 的历史状态管理
- 粒子自碰撞的 BVH 和 barrier 风格法向势能

IPC 对它最大的启发，不是“换求解器”，而是补齐 4 个短板：

- 连续时间安全步长
- active set
- 粒子侧接触历史
- 接触接受策略

一旦这 4 个点补上，当前 VBD 会从“能跑且快”明显提升到“更稳、更可控、更接近 IPC 级接触质量”。
