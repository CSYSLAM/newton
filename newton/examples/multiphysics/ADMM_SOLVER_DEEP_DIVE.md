# `example_mujoco_vbd_admm_solver.py` 完整链路深度解析

> 以 `example_mujoco_vbd_admm_solver.py` 为线索，从场景搭建到 ADMM 迭代收敛，
> 逐层拆解代码、kernel、公式，把完整链路讲透。
>
> 代码来自 `newton/examples/multiphysics/` 与 `newton/_src/solvers/coupled/`。

---

## 1. 场景长什么样

一个 3D 场景里同时跑两组"跨求解器"耦合实验：

**左半边：布料 + 铁球（body-particle 附件耦合）**

一块 11×11 的布料（VBD 模拟），左右两边缘固定。布料中心点挂一个铁球（MuJoCo 刚体）。铁球通过 `add_body_particle_attachment`（body-particle 附件标注）钉在布料中心粒子上。球往下拽布料，布料托住球。

**右半边：摆链 + 载荷（跨求解器球铰耦合）**

一根摆链（MuJoCo 管的刚体），末端用一个 `ball joint`（球铰）连接一个载荷方块（VBD 管的刚体）。摆链摆动时拖着载荷晃。

**核心难点**：铁球（MuJoCo）和布料（VBD）分属不同求解器；球铰的 parent（摆链，MuJoCo）和 child（载荷，VBD）也分属不同求解器。没有任何一个求解器能同时看到关节/附件的两端。**`SolverCoupledADMM` 的任务就是让这两个求解器在"接口"处对齐**——通过线性化 ADMM 迭代，让两边的界面相对速度收敛到满足约束。

---

## 2. 场景搭建：逐行读 `__init__`

### 2.1 时间参数

```python
self.fps = 60
self.frame_dt = 1.0 / self.fps          # 每帧 ~16.67ms
self.sim_substeps = 8                    # 每帧 8 个子步
self.sim_dt = self.frame_dt / self.sim_substeps  # 每子步 ~2.08ms
```

> 💡 为什么要子步？ADMM 在每个子步内做不动点迭代，子步越小，"线性化"假设越准确（速度变化小，雅可比 J 才近似常值），收敛越好。8 子步 × 2 次 ADMM 迭代 = 每帧 16 次子求解器步进。

### 2.2 布料（归 VBD）

```python
dim = 11
cloth_z = 2.0                          # 布料高度 z=2.0
particle_start = builder.particle_count  # 记住粒子 id 起点（此时为 0）

builder.add_cloth_grid(
    pos=wp.vec3(-0.5, -0.5, cloth_z),
    fix_left=True, fix_right=True,      # 左右边缘固定
    dim_x=dim, dim_y=dim,               # 11x11 网格
    cell_x=0.1, cell_y=0.1,             # 每格 0.1m
    mass=0.05,                           # 每粒子 0.05kg
    tri_ke=1.0e3,                        # 三角形拉伸刚度
    edge_ke=0.01,                        # 边弯曲刚度
    particle_radius=0.01,
)
```

11×11 = 121 个粒子，id 从 `particle_start` 到 `particle_start + 120`，全部归 VBD entry。

**中心粒子 id 计算**：

```python
center = dim // 2                       # center = 5
self.center_particle = particle_start + center * (dim + 1) + center
# = 0 + 5 * 12 + 5 = 65
```

布料粒子按行优先排列（每行 dim+1=12 个），第 5 行第 5 列即中心点，全局 id = 65。

### 2.3 铁球（归 MuJoCo）+ body-particle 附件

```python
ball_radius = 0.08
self.ball_body = builder.add_body(
    xform=wp.transform(p=wp.vec3(0.0, 0.0, cloth_z - ball_radius), q=wp.quat_identity()),
    mass=0.5,                            # 球质量 0.5kg
    inertia=wp.mat33(np.eye(3) * 5.0e-3),  # 各向同性惯量
)
self.ball_joint = builder.joint_count - 1   # 球的 free joint（让球能自由落体）
builder.add_shape_sphere(self.ball_body, radius=ball_radius)

# ★ 关键：注册一个 body-particle 附件标注
SolverCoupledADMM.add_body_particle_attachment(
    builder,
    self.ball_body,                     # 刚体球（归 MuJoCo）
    self.center_particle,               # 布料中心粒子（归 VBD）
    body_point=wp.vec3(0.0, 0.0, ball_radius),  # 钉在球底部（球心下方 ball_radius）
    stiffness=1.0e3,                    # 附件刚度 κ
)
```

> ⚠️ `add_body_particle_attachment` **不创建物理约束**，它只是往 Model 上写一行"自定义频率属性"（注册在 `coupling` 命名空间下）。这行标注记录：body=X、particle=Y、锚点=Z、刚度=κ。真正把它变成约束的是 `SolverCoupledADMM` 构造时扫描这些标注（见 4.3）。

`body_point=wp.vec3(0,0,ball_radius)` 的含义：附件锚点在球本体坐标系的 (0,0,0.08) 处，即球底部。球心在 z=cloth_z-ball_radius=1.92，球底部在 z=1.84，正好贴着布料中心点（z=2.0 附近，布料会被球拽下去）。

### 2.4 摆链（归 MuJoCo）

```python
link_hx = 0.28
anchor = wp.vec3(1.4, 0.0, 2.2)         # 摆链悬挂点

self.pendulum_body = builder.add_link(
    xform=wp.transform(p=anchor + wp.vec3(link_hx, 0.0, 0.0), q=wp.quat_identity()),
    mass=0.6,
    inertia=wp.mat33(np.eye(3) * 1.0e-2),
)
builder.add_shape_box(self.pendulum_body, hx=link_hx, hy=0.045, hz=0.045)
self.pendulum_joint = builder.add_joint_revolute(
    parent=-1,                          # parent 是世界（固定）
    child=self.pendulum_body,
    axis=wp.vec3(0.0, 1.0, 0.0),        # 绕 Y 轴转
    target_kd=0.5,                      # 阻尼
    parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
    child_xform=wp.transform(p=wp.vec3(-link_hx, 0.0, 0.0), q=wp.quat_identity()),
)
builder.add_articulation([self.pendulum_joint], label="pendulum")
```

摆链是一个绕 Y 轴旋转的单连杆，挂在 anchor 点。它完全归 MuJoCo entry（body 和 joint 都给 mjc）。

### 2.5 载荷（归 VBD）+ 跨求解器球铰

```python
payload_hx = 0.12
self.payload_body = builder.add_body(
    xform=wp.transform(p=anchor + wp.vec3(2.0*link_hx + payload_hx, 0.0, 0.0), q=wp.quat_identity()),
    mass=0.35,
    inertia=wp.mat33(np.eye(3) * 6.0e-3),
)
self.payload_free_joint = builder.joint_count - 1   # 载荷的 free joint（归 VBD）
builder.add_shape_box(self.payload_body, hx=payload_hx, hy=0.09, hz=0.09)

# ★ 关键：parent 是摆链（MuJoCo），child 是载荷（VBD）—— 跨求解器球铰！
builder.add_joint_ball(
    parent=self.pendulum_body,          # 归 mjc
    child=self.payload_body,            # 归 vbd
    friction=1.0,                       # 摩擦
    parent_xform=wp.transform(p=wp.vec3(link_hx, 0.0, 0.0), q=wp.quat_identity()),
    child_xform=wp.transform(p=wp.vec3(-payload_hx, 0.0, 0.0), q=wp.quat_identity()),
    collision_filter_parent=True,
)
```

这个 ball joint 的 parent body（pendulum_body）归 mjc，child body（payload_body）归 vbd。**这个关节不归任何 entry 拥有**（见 4.2 的 `_cross_solver_joint_entries` 检查），它由 ADMM 统一处理成约束。

### 2.6 关掉普通软接触

```python
self.model = builder.finalize()
# ADMM 附件已经把球绑在布上；普通软接触会跟附件打架
self.model.soft_contact_ke = 0.0
self.model.soft_contact_kd = 0.0
```

> ⚠️ 重要：本例**没有**用 `contact_pairs`（Config 里没传），所以 ADMM 不会自建碰撞管线。如果留着 `soft_contact_ke > 0`，球和布之间会产生普通软接触力，和附件约束力叠加，导致球被反复推挤。所以置零。

---

## 3. Entry 划分与所有权

```python
self.solver = SolverCoupledADMM(
    model=self.model,
    entries=[
        SolverCoupled.Entry(
            name=rigid_name,                              # "mjc"
            solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=32),
            bodies=[self.ball_body, self.pendulum_body],  # 拥有球 + 摆链
            joints=[self.ball_joint, self.pendulum_joint],# 球的 free joint + 摆链关节
        ),
        SolverCoupled.Entry(
            name="vbd",
            solver=lambda v: SolverVBD(model=v, iterations=8),
            bodies=[self.payload_body],                   # 拥有载荷
            joints=[self.payload_free_joint],             # 载荷的 free joint
            particles=list(range(self.model.particle_count)),  # 拥有所有布料粒子
        ),
    ],
    coupling=SolverCoupledADMM.Config(
        iterations=2,          # 每步 ADMM 迭代 2 次
        rho=50,                # ADMM 惩罚参数
        gamma=0.1,             # proximal term 系数
        baumgarte=0.01,        # 位置修正系数
        joint_proximal_bodies=args.joint_proximal_bodies,              # True
        joint_proximal_destination_entries=(rigid_name,),              # ("mjc",)
    ),
)
```

### 所有权表

| 实体 | 全局 id | 归属 |
| --- | --- | --- |
| ball_body | 0 | mjc |
| pendulum_body | 1 | mjc |
| payload_body | 2 | vbd |
| ball_joint (free) | 0 | mjc |
| pendulum_joint (revolute) | 1 | mjc |
| payload_free_joint | 2 | vbd |
| ball joint（球铰, parent=1, child=2） | 3 | **无人拥有**（ADMM 管） |
| 粒子 0..120 | 0..120 | vbd |

> 🔑 **跨求解器关节不归任何 entry**：`_cross_solver_joint_entries` 会检查 `self._joint_owner[joint] >= 0` 时报错——"跨求解器关节必须留给 SolverCoupledADMM，否则约束会被算两遍"（子求解器算一次，ADMM 又算一次）。所以球铰 joint 不出现在任何 Entry 的 `joints` 列表里。

### Config 参数解释

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `iterations` | 2 | 每个子步内 ADMM 迭代 2 轮 |
| `rho` | 50 | ADMM 惩罚参数 ρ，越大越急着拉拢界面 |
| `gamma` | 0.1 | proximal term 系数 γ，给参与约束的实体加虚拟质量 γ·ρ |
| `baumgarte` | 0.01 | 位置误差修正比例（每步消除 1% 穿透/漂移） |
| `joint_proximal_bodies` | True | 把跨求解器关节的邻居 body 作为局部惯量代理保留 |
| `joint_proximal_destination_entries` | ("mjc",) | 只在 mjc 视图里保留这些代理 body |

---

## 4. ADMM 构造期：发现约束、建缓冲

`SolverCoupledADMM.__init__` 调用 `super().__init__`（建视图、分配状态）后，调 `_setup_admm`。这一节讲构造期做的所有事。

### 4.1 跨求解器关节代理体可见性

`_init_admm_joint_proxy_visibility`（构造前调用）：为了让球铰两端在各自视图里都"看得见"对方，把关节邻居 body 保留为 proxy body。

对本例的球铰（parent=pendulum_body 归 mjc，child=payload_body 归 vbd）：

```python
for joint in range(model.joint_count):
    if joint_type == BALL and parent_owner != child_owner:
        # parent_name="mjc", child_name="vbd"
        if "mjc" in destination_names:   # destination_names = ("mjc",)
            add_proxy_body("mjc", "vbd", child=payload_body)  # mjc 视图保留载荷作代理
            joint_keep["mjc"].add(joint)
        if "vbd" in destination_names:   # vbd 不在 destination_names，跳过
            ...
```

因为 `joint_proximal_destination_entries=("mjc",)`，只有 mjc 视图会保留载荷 body 作为代理体（带 PROXY 标志，惯量来自 vbd 等效质量 × `joint_proximal_mass_scale`）。vbd 视图不保留摆链——它通过 ADMM 约束力"感受"摆链，不需要看见摆链 body。

> 💡 这是不对称的：mjc 需要看见载荷（才能算球铰约束对摆链的反作用），而 vbd 已经直接拥有载荷，不需要看见摆链。`joint_proximal_destination_entries` 让你控制只把代理加到需要的视图，避免无谓的体量膨胀。

### 4.2 视图构建（基类 `_build_entries`）

对每个 entry，基类做：

1. 创建 `ModelView`。
2. 非 owned 且非 proxy 的 body → `disable_body_dynamics`（逆惯量置零，子求解器推不动）。
3. proxy body → `mark_proxy_bodies`（打 PROXY 标志，保留动态）。
4. mjc 视图里：看不到粒子（粒子归 vbd，mjc 不处理 particle）；能看到载荷 body（作为 proxy）。
5. vbd 视图里：能看到所有粒子 + 载荷 body（owned）；看不到球和摆链（被 disable）。
6. `_customize_compact_view`（ADMM 覆盖）：`_disable_admm_joint_proxy_shape_collisions`——把 proxy body 的 shape 碰撞标志关掉，避免 proxy 形状参与碰撞（它只是惯量代理，不是碰撞体）。

### 4.3 发现约束（三类）

`_setup_admm` 调三个发现函数：

**A. `_build_admm_joint_groups`：扫描跨求解器关节**

```python
for joint in range(model.joint_count):
    if not joint_enabled[joint]:
        continue
    parent = joint_parent[joint]
    child = joint_child[joint]
    owner_pair = self._cross_solver_joint_entries(joint, parent, child)
    if owner_pair is None:    # 同 entry 内的关节，跳过（子求解器自己管）
        continue
    child_entry, parent_entry = owner_pair   # ("vbd", "mjc")

    if joint_type == BALL:
        # 球铰 -> 平移附件（约束三点重合）+ 角摩擦（若有 friction）
        point_items[("vbd","mjc")].append(
            (child=payload_body, child锚点, parent=pendulum_body, parent锚点,
             stiffness=joint_stiffness, damping=joint_damping))
        if friction > 0:
            angular_friction_items[("vbd","mjc")].append(...)
```

球铰被拆成：**一个平移附件**（约束 parent 锚点和 child 锚点重合，即球铰中心点）+ **一个角摩擦**（如果有 friction>0，约束相对角速度的摩擦）。

平移附件用 `Config.joint_stiffness`（默认 1e4）和 `joint_damping`（默认 0）作为 κ 和 c。

**B. `_build_admm_body_particle_attachment_groups`：扫描附件标注**

```python
count = model.custom_frequency_counts["coupling:body_particle_attachment"]  # 1
for row in range(count):
    body = body_np[row]           # ball_body = 0（归 mjc）
    particle = particle_np[row]   # center_particle = 65（归 vbd）
    body_entry = "mjc"
    particle_entry = "vbd"
    if body_entry != particle_entry:   # 跨求解器 -> 转 ADMM 附件
        grouped[("mjc","vbd")].append((body=0, point=(0,0,0.08), particle=65, κ=1e3, c=0))
```

每个 body-particle 附件变成一个 `_AdmmRigidParticleAttachmentGroup`（rigid-particle 附件 group）。

**C. `_setup_admm_contact_specs`：扫描接触对**

本例 `Config.contact_pairs` 为空，所以**不建任何接触约束**。如果传了 `ContactPair`，会自建 `CollisionPipeline` 检测碰撞，拆成 rigid-rigid / rigid-particle / particle-particle 接触 group。

### 4.4 等效质量缓冲

对每个 entry，`_setup_admm_effective_mass_buffers` 调子求解器的 `coupling_eval_effective_mass[_block]` 钩子，把 owned body/particle 的等效质量填到 `buf.body_effective_mass` / `buf.particle_effective_mass`：

```python
buf.body_effective_mass = wp.clone(model.body_mass)   # 先用模型质量初始化
# 然后调钩子覆盖（对铰接体，等效质量反映关节约束）
entry.solver.coupling_eval_effective_mass_block(
    buf.body_endpoint_kind, buf.body_endpoint_index, ...,
    buf.body_effective_mass_local, buf.body_effective_inertia_local)
# scatter 回全局数组
wp.launch(scatter_body_effective_mass_block_kernel, ...)
```

> 💡 等效质量是 ADMM 的关键输入：它决定界面权重 W（见 5.2），W 决定约束力的分配。比如球（0.5kg）和布料中心粒子（0.05kg），W 会偏向轻的一侧——约束力更多作用在轻的粒子上（因为轻的更容易被推动）。

### 4.5 proximal mask 与虚拟惯量

`gamma=0.1 > 0`，所以 `_refresh_admm_proximal_masks` 标记所有参与约束的 body/particle，并给它们叠加 `γ·ρ = 0.1×50 = 5` 的虚拟质量：

```python
# 对 body-particle 附件 group：
self._accumulate_body_point_proximal_lump(body_entry, buf, body_ids, point_body, W)
# 给 body 加 W²·γ·ρ 的点质量 lump + 对应惯量
self._accumulate_global_indices_proximal_lump(particle_ids, ..., particle_proximal_mass, ...)
# 给 particle 加 W²·γ·ρ 的质量 lump
```

然后 `_refresh_admm_proximal_view_overrides` 通过 `view.add_body_lumped_inertia` / `view.add_particle_lumped_mass` 把这些虚拟质量写到视图上。子求解器看到的 body/particle 比真实"重"一点，刚度相对降低，ADMM 更容易推动它收敛。

### 4.6 缓冲分配

每个 entry 分配 `_AdmmBuffers`：

```python
buf.body_q_n    # 步起始 body 位姿（每轮迭代回退到这）
buf.body_qd_n   # 步起始 body 速度
buf.body_qd_k   # 上一轮迭代结束的速度（算 Jv 用）
buf.body_f      # 累积约束力的缓冲
buf.body_proximal_mass       # proximal 虚拟质量
buf.body_effective_mass      # 等效质量
# particle、joint 同理
```

> 🔑 `*_n` 是"步起始快照"，`*_k` 是"迭代工作速度"。每轮 ADMM 都从 `*_n` 重置，但 `λ`（对偶变量）跨轮累积。

---

## 5. 约束的数学建模

### 5.1 统一形式：二次型耦合能量

ADMM 把每条约束（附件、关节、接触）都建模成二次型能量：

```text
E_c(u) = (κ/2)·||u − u_target||²  +  (c/2)·||u||²
```

| 符号 | 含义 | 本例取值 |
| --- | --- | --- |
| `u` | 辅助变量 = 界面协商相对速度 | 求解量 |
| `u_target` | 目标相对速度（Baumgarte 位置修正） | 通常 0，有穿透时非零 |
| `κ` | 约束刚度 | body-particle: 1e3；关节: 1e4 |
| `c` | 阻尼 | 默认 0 |

- `u_target = 0`：约束"界面相对速度为零"（附件两点速度一致）。
- `u_target ≠ 0`：Baumgarte 位置稳定——如果两点已经漂移分开，给一个非零目标速度把它们拉回。

### 5.2 界面权重 W

`compute_interface_weights_kernel` + `_interface_weight`：

```python
def _interface_weight(m_a, m_b):
    if m_a > 0 and m_b > 0:
        return sqrt((m_a * m_b) / (m_a + m_b))   # 调和平均的几何平均
    if m_a > 0:
        return sqrt(m_a)
    if m_b > 0:
        return sqrt(m_b)
    return 1.0
```

> 💡 W 像两个质量的"折中"：球 0.5kg，布粒子 0.05kg，W = sqrt(0.5×0.05/0.55) ≈ sqrt(0.0455) ≈ 0.213。W 决定约束力的"传导强度"——两侧都重则 W 大（力传导强），一侧轻则 W 小。

### 5.3 ADMM 更新公式（核心两式）

来自 `admm_utils.py:700`：

```text
         ρ·W²·Jv + κ·u_target − W·λ
u^{k+1} = ─────────────────────────────
              κ + c + ρ·W²

λ^{k+1} = λ^k + ρ·W·(u^{k+1} − Jv)
```

逐项解释：

- `Jv` = J·v = 约束雅可比 × 当前速度 = **实际界面相对速度**。每轮用子求解器算出来的速度重算。
- `u^{k+1}` = **协商速度**：综合"实际速度 Jv""目标 u_target""对偶压力 λ"，算出一个"理想界面速度"。
- `λ` = **对偶变量（拉格朗日乘子）**：跨轮累积，记录"界面还差多少没对齐"，下一轮加大推力。
- `ρ` = 50 = 惩罚参数：放大界面不匹配的代价。
- 分母 `κ + c + ρ·W²`：刚度 + 阻尼 + ADMM 惩罚的合计"刚度"。

**第二个公式**：如果协商速度 u 和实际速度 Jv 还对不上，就把差值 `ρ·W·(u−Jv)` 累加进 λ，下一轮加大力度——这是 ADMM 的反馈机制。

### 5.4 力的施加（J^T W splat）

算出 u 和 λ 后，约束力 F 作用回 body：

```text
F = W·(λ + ρ·W·(u − Jv))
```

`attach_rr_accumulate_forces_kernel` 把 F splat（泼洒）成 body 上的 spatial force：

```python
force_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
# 作用点 = body 世界位姿 × 本地锚点
point_a = transform_point(body_q_a[ba], point_a_local[i])
arm_a = point_a - transform_point(body_q_a[ba], body_com_a[ba])  # 力臂（作用点到质心）
# spatial force = (力, 力矩 = arm × force)
wp.atomic_add(body_f_a, ba, wp.spatial_vector(force_a, wp.cross(arm_a, force_a)))
# body B 受反作用力
force_b = -force_a
wp.atomic_add(body_f_b, bb, wp.spatial_vector(force_b, wp.cross(arm_b, force_b)))
```

> 💡 力矩 = 力臂 × 力：约束力作用在锚点（不是质心），所以要算力矩。`wp.atomic_add` 是 GPU 原子加——多个约束可能同时往一个 body 加力，必须原子操作避免竞争。

### 5.5 Baumgarte 位置修正

`_admm_begin_step` 在每步开始算 `u_target`：

```python
if coupling.baumgarte > 0:
    wp.launch(attach_rp_compute_u_target_kernel, ...)
    # u_target = -baumgarte/dt * (当前两点位置差)
    # 如果两点漂移开了 d，给一个 -baumgarte/dt·d 的目标速度把它们拉回
```

`baumgarte=0.01` 表示每步消除 1% 的位置误差——温和修正，避免震荡。

---

## 6. `step()` 完整调度链路

从用户代码到 kernel 的完整调用栈：

```text
Example.step()
  └─ _launch_frame_graph() 或 simulate()
       └─ Example.simulate()                          [每帧 8 子步循环]
            ├─ state_0.clear_forces()
            ├─ apply_coupled_viewer_forces()           [鼠标拖拽外力]
            ├─ # self.model.collide(...)  ← 注释掉了！本例不用普通碰撞
            ├─ solver.step(state_0, state_1, control, contacts, sim_dt)
            │    └─ SolverCoupled.step()               [基类]
            │         ├─ _distribute_state(state_in)   [父状态 → 各 entry.state_0]
            │         ├─ _step_coupled(...)            [SolverCoupledADMM 覆盖]
            │         │    └─ (见第 7 节 ADMM 迭代)
            │         ├─ _copy_state(state_in, state_out)
            │         └─ _reconcile_state(state_out)   [各 entry.state_1 → 父状态]
            ├─ eval_ik(model, state_1, joint_q, joint_qd)  [逆运动学：body_q → joint_q]
            └─ state_0, state_1 = state_1, state_0     [乒乓交换]
```

### 6.1 `_distribute_state`（分发）

把父状态按所有权拷进各 entry 的 `state_0`：

```python
for entry in self._entries.values():
    _copy_state_to_entry(state_in, entry.state_0, entry)
    # 只拷该 entry 拥有的 body/particle/joint 状态
    self._notify_input_state_update(entry, flags, dt=dt)
    # 调 entry.solver.coupling_notify_input_state_update()
    # 让子求解器刷新内部缓存（如 VBD 的 BVH）
```

### 6.2 `_reconcile_state`（调和）

各 entry 算完后，把 owned 状态 scatter 回父状态（`_scatter_body_state_mapped` 等 kernel）。**proxy body 状态不回写**——它只是镜像，真状态在源求解器那儿。

### 6.3 为什么注释掉 `collide`

```python
# ADMM 从关节和 body-particle 附件构建耦合，所以保持 state_0/contacts 空
# 不让 collide() 加多余的约束。
# self.model.collide(self.state_0, self.contacts)
```

本例没有 `contact_pairs`，ADMM 不处理碰撞接触。如果调 `model.collide`，会生成普通软接触（虽然 `soft_contact_ke=0` 已经关掉力，但接触数据本身是无谓开销）。所以注释掉。

### 6.4 `eval_ik`（逆运动学）

```python
newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
```

MuJoCo/VBD 步进后更新了 `body_q`（刚体位姿），但 `joint_q`（关节广义坐标）没更新。`eval_ik` 从 body 位姿反算关节坐标，保证两者一致（下一步分发时子求解器拿到一致的关节状态）。

---

## 7. ADMM 迭代详细流程（`_step_coupled`）

这是核心。`_step_coupled`（`solver_coupled_admm.py:2762`）：

### 7.1 步前准备

```python
def _step_coupled(self, state_in, state_out, control, contacts, dt):
    iters = int(coupling.iterations)   # 2

    # 1. 刷新碰撞接触 group（本例无 contact_pairs，跳过）
    self._refresh_collision_contact_groups(state_in)

    # 2. gamma>0：刷新 proximal mask + 视图虚拟惯量
    if coupling.gamma > 0.0:
        self._refresh_admm_proximal_masks()
        self._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)

    # 3. 快照步起始状态到 buf（每轮迭代都回退到这）
    for name, entry in self._entries.items():
        buf = self._admm_buffers[name]
        wp.copy(buf.body_q_n, entry.state_0.body_q)    # 步起始位姿
        wp.copy(buf.body_qd_n, entry.state_0.body_qd)  # 步起始速度
        wp.copy(buf.body_qd_k, entry.state_0.body_qd)  # 工作速度初始化

    # 4. 算各约束的 u_target（Baumgarte）
    self._admm_begin_step(dt)
```

> 🔑 第 3 步是 ADMM 不动点迭代的关键：**每轮都从步起始 `*_n` 重置**，不是接着上一轮结果继续。这样是在同一个 `dt` 区间上反复求解直到界面收敛。`λ` 跨轮累积（暖启动），所以越迭代越接近正确解。

### 7.2 ADMM 迭代主循环

```python
    for k in range(iters):   # k=0, 1
        # === A. 回退状态 + 施加 proximal 速度位移 ===
        for name, entry in self._entries.items():
            self._prepare_admm_iteration_state(
                entry, self._admm_buffers[name], state_in, dt,
                iteration_restart=(k > 0))

        # === B. 算 Jv + 把约束力 splat 到 body_f/particle_f ===
        self._accumulate_admm_forces(k, dt,
            refresh_jv=(k == 0),          # 第 0 轮重算 Jv
            initialize_contact_u=(k == 0))

        # === C. 把力设为各 entry 输入 ===
        for name, entry in self._entries.items():
            self._apply_admm_force_inputs(entry, self._admm_buffers[name], dt)

        # === D. 步进所有子求解器 ===
        for entry in self._entries.values():
            self._step_entry(entry, control, contacts, dt)

        # === E. 快照新速度到 *_k ===
        for name, entry in self._entries.items():
            buf = self._admm_buffers[name]
            wp.copy(buf.body_qd_k, entry.state_1.body_qd)   # 新速度

        # === F. 用新速度更新对偶变量 λ ===
        self._update_admm_dual(k, dt)
```

### 7.3 逐步详解

**A. `_prepare_admm_iteration_state`：回退 + proximal 位移**

```python
def _prepare_admm_iteration_state(self, entry, buf, state_in, dt, *, iteration_restart):
    gamma = coupling.gamma   # 0.1
    apply_proximal = gamma > 0

    # 回退到步起始
    wp.copy(entry.state_0.body_q, buf.body_q_n)
    wp.copy(entry.state_0.body_qd, buf.body_qd_n)
    # particle、joint 同理
    self._notify_input_state_update(entry, flags, dt=dt, iteration_restart=...)

    # proximal 速度位移：给参与约束的 body 一个"向 λ 方向"的初始速度推力
    if apply_proximal:
        self._apply_admm_velocity_proximal_shift(entry, buf, dt)
        # velocity_proximal_shift_body_lumped_kernel:
        #   把 proximal 虚拟质量产生的"期望速度偏移"叠加到 state_0.body_qd
        #   相当于给 ADMM 一个更好的初值，加速收敛

    # 重置 body_f = 外力（来自父状态）
    if buf.body_f is not None:
        # 从 state_in.body_f 拷外力到 buf.body_f
        wp.launch(_copy_mapped_spatial_vector, ...)
        # 若有 proximal，补偿重力（避免双计）
        if apply_proximal:
            wp.launch(body_gravity_compensation_lumped_kernel, ...)
```

> 💡 proximal 速度位移的直觉：ADMM 想让界面速度对齐。与其从零开始迭代，不如先用 proximal 虚拟质量"预判"一个偏移速度，让子求解器一上来就接近对齐状态。像考试前先看一遍答案提示，答题更快。

**B. `_accumulate_admm_forces`：算 Jv + splat 力**

对每个约束 group（本例有两组：body-particle 附件、关节球铰）：

```python
# 以 body-particle 附件为例（_admm_rp_groups）
for group in self._admm_rp_groups:
    if refresh_jv:   # 第 0 轮才算 Jv
        wp.launch(attach_rp_compute_Jv_kernel, ...)
        # Jv = body 锚点速度 − particle 速度
        #   body 锚点速度 = body 线速度 + ω × arm（arm = 锚点相对质心）
        #   particle 速度 = particle_qd
    wp.launch(attach_rp_accumulate_forces_kernel, ...)
    # F = W·(λ + ρ·W·(u − Jv))
    # splat 到 body_f（spatial force）和 particle_f（3维力）
```

> ⚠️ `refresh_jv=(k==0)`：只有第 0 轮算 Jv。后续轮用上一轮的 Jv（因为状态会变，但 ADMM 线性化假设 J 近似常值，省一次计算）。最后一轮 `_update_admm_dual` 会用新速度重算 Jv 来更新 λ。

**C. `_apply_admm_force_inputs`：设力为输入**

```python
def _apply_admm_force_inputs(self, entry, buf, dt):
    if entry.body_indices.shape[0] > 0:
        self._set_local_body_force_input(entry, buf.body_f, dt=dt)
        # 把 buf.body_f（累积的约束力）设为 entry 的 body_f 输入
    if entry.particle_indices.shape[0] > 0:
        self._set_local_particle_force_input(entry, buf.particle_f, dt=dt)
```

**D. `_step_entry`：步进子求解器**

```python
# 基类方法，调 entry.solver.step(state_0, state_1, control, contacts, dt)
# mjc.step(...)：MuJoCo 带着约束力往前算一步
# vbd.step(...)：VBD 带着约束力往前算一步
```

两个子求解器各自独立步进，但都受到了 ADMM 约束力的"引导"——这股力正是让它们在界面处对齐的推手。

**E. 快照新速度**

子求解器算完后，`state_1.body_qd` 是新速度。拷到 `buf.body_qd_k`，供下一轮算 Jv 和更新 λ 用。

**F. `_update_admm_dual`：更新对偶变量**

```python
def _update_admm_dual(self, iteration_k, dt):
    for group in self._admm_rp_groups:
        # 用新速度重算 Jv
        wp.launch(attach_rp_compute_Jv_kernel, ...,
            inputs=[..., entry.state_1.body_qd, ...])   # ← 用 state_1（新速度）
        # 更新 u 和 λ
        self._update_admm_quadratic_dual(group)
```

`_update_admm_quadratic_dual`：

```python
# u 更新（u_update_quadratic_kernel）
u^{k+1} = (ρ·W²·Jv + κ·u_target − W·λ) / (κ + c + ρ·W²)

# λ 更新（lambda_update_kernel）
λ^{k+1} = λ^k + ρ·W·(u^{k+1} − Jv)
```

> 🔑 λ 跨轮累积，所以第 1 轮（k=1）的 λ 带着第 0 轮的"记忆"——这就是 ADMM 的暖启动。如果界面还没对齐，λ 会越来越大，约束力越来越强，直到对齐。

---

## 8. 碰撞流程（本例的取舍 + ADMM 通用机制）

### 8.1 本例为什么不用 ADMM 碰撞

本例的耦合是**附件 + 关节**，不是"碰撞接触"。球和布之间是固定钉死的附件关系，不是"碰到才有力"。所以：

- Config 不传 `contact_pairs` → 不建 `CollisionPipeline`。
- `model.soft_contact_ke = 0` → 即使有碰撞数据也不产生力。
- `simulate()` 里注释掉 `model.collide()` → 不检测碰撞。

### 8.2 ADMM 的碰撞机制（通用）

如果配了 `contact_pairs`，ADMM 会：

1. **构造期**：`_setup_admm_contact_specs` 发现接触对；`_setup_admm` 自建 `CollisionPipeline`（`broad_phase="explicit"`，按 shape_pairs_filtered）；预分配 `_admm_internal_contacts`。
2. **每步**：`_refresh_collision_contact_groups` 调 `collision_pipeline.collide(state, contacts)` 检测碰撞；`contact_rr_fill_from_rigid_contacts_kernel` 把检测结果填进 ADMM 接触 group（body_ids、normal、friction、W）。
3. **迭代**：接触约束的 u 更新是**带库仑摩擦的局部投影**（`contact_u_update_kernel` → `solve_coulomb_isotropic`），不是二次型闭式解。`u_min` 是非穿透约束的下界。
4. **暖启动**：`rigid_contact_matching` 支持 `"disabled"/"latest"/"sticky"`，跨帧匹配相同接触复用 λ（dual warm-start），`contact_matching_force_scale=0.9` 控制 90% 复用。

### 8.3 为什么 ADMM 接触要替代子求解器接触

如果子求解器（如 VBD）自己也处理接触，同一对接触会被算两遍（VBD 算一次 + ADMM 算一次），力翻倍。所以用 ADMM 接触时，要么关掉子求解器接触（`use_mujoco_contacts=False`、`soft_contact_ke=0`），要么不让 `model.collide` 产生接触数据。

---

## 9. 耦合方法总结

### 9.1 本例用到的耦合方法

| 耦合类型 | 来源 | 转成的 ADMM 约束 | 数学形式 |
| --- | --- | --- | --- |
| body-particle 附件 | `add_body_particle_attachment` 标注 | `_AdmmRigidParticleAttachmentGroup` | 二次型，u=0 |
| 跨求解器球铰 | `add_joint_ball`（parent≠child entry） | 平移附件 + 角摩擦 | 二次型 + box 摩擦 |

### 9.2 ADMM 耦合的完整数据流

```text
                    步起始状态 state_0
                          │
              ┌───────────┴───────────┐
              │  快照到 buf.*_n       │
              │  算 u_target (Baumgarte)│
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │  for k in iterations: │  ← 不动点迭代
              │    A. 回退到 *_n       │
              │       + proximal 位移  │
              │    B. 算 Jv            │
              │       splat 约束力     │
              │       → body_f/particle_f
              │    C. 设力为 entry 输入 │
              │    D. 步进 mjc + vbd   │  ← 子求解器独立步进
              │    E. 快照新速度 *_k   │
              │    F. 更新 u, λ        │  ← 暖启动下一轮
              └───────────┬───────────┘
                          │
                  reconcile_state
                  → state_1
```

### 9.3 与 Proxy 耦合的对比

| 维度 | SolverCoupledProxy | SolverCoupledADMM（本例） |
| --- | --- | --- |
| 耦合媒介 | 虚拟代理体/代理粒子 | 约束方程（附件/关节/接触） |
| 信息传递 | 单向滞后（源→目标→收割→源） | 双向同时（约束力同时作用两侧） |
| 收敛性 | 依赖松弛，可能震荡 | ADMM 数学保证收敛 |
| 迭代 | 每步 1-N 次代理 pass | 每步 N 次完整子求解器步进 |
| 适合场景 | 弱耦合、单向主导 | 强两路约束（关节、附件） |
| 侵入性 | 小（只需等效质量钩子） | 大（需派生约束、维护 λ、proximal） |

### 9.4 关键设计要点

1. **跨求解器关节不归任何 entry**：留给 ADMM，避免双算。
2. **每轮回退到步起始**：不动点迭代，`λ` 暖启动累积。
3. **proximal term（γ）**：虚拟质量降低刚度，加速收敛，代价是结果稍软。
4. **等效质量驱动权重 W**：自动按两侧"分量"分配约束力。
5. **Baumgarte**：温和位置修正，防漂移。
6. **全 GPU kernel + 构造期常量**：保证 CUDA Graph 可捕获（`capture()` 注释说明"无运行时分支"）。
7. **`eval_ik`**：步进后反算关节坐标，保证 body_q 与 joint_q 一致。

### 9.5 调度关系一览

```text
SolverCoupledADMM
  ├─ 构造期
  │    ├─ _init_admm_joint_proxy_visibility   [代理 body 可见性]
  │    ├─ super().__init__ → _build_entries   [建视图、分配 state]
  │    └─ _setup_admm
  │         ├─ _setup_admm_effective_mass_buffers  [等效质量]
  │         ├─ _build_admm_joint_groups            [关节→约束]
  │         ├─ _build_admm_body_particle_attachment_groups  [附件→约束]
  │         ├─ _setup_admm_contact_specs           [接触对（本例无）]
  │         └─ _refresh_admm_proximal_masks        [proximal 虚拟质量]
  │
  └─ 每步 _step_coupled
       ├─ _refresh_collision_contact_groups        [碰撞检测（本例跳过）]
       ├─ 快照 *_n
       ├─ _admm_begin_step                         [u_target]
       └─ for k in iterations:
            ├─ _prepare_admm_iteration_state       [回退 + proximal 位移]
            ├─ _accumulate_admm_forces             [Jv + splat 力]
            ├─ _apply_admm_force_inputs            [设 entry 输入力]
            ├─ _step_entry × N                     [步进子求解器]
            ├─ 快照 *_k
            └─ _update_admm_dual                   [更新 u, λ]
```

---

## 10. 关键文件与行号索引

| 内容 | 位置 |
| --- | --- |
| 示例主文件 | `newton/examples/multiphysics/example_mujoco_vbd_admm_solver.py` |
| ADMM 求解器 | `newton/_src/solvers/coupled/solver_coupled_admm.py` |
| Config 定义 | `solver_coupled_admm.py:578` |
| `add_body_particle_attachment` | `solver_coupled_admm.py:515` |
| `_init_admm_joint_proxy_visibility` | `solver_coupled_admm.py:750` |
| `_setup_admm` | `solver_coupled_admm.py:1022` |
| `_build_admm_joint_groups` | `solver_coupled_admm.py:2396` |
| `_build_admm_body_particle_attachment_groups` | `solver_coupled_admm.py:2674` |
| `_step_coupled`（ADMM 主循环） | `solver_coupled_admm.py:2762` |
| `_admm_begin_step`（u_target） | `solver_coupled_admm.py:3382` |
| `_apply_admm_velocity_proximal_shift` | `solver_coupled_admm.py:3551` |
| `_prepare_admm_iteration_state` | `solver_coupled_admm.py:3605` |
| `_accumulate_admm_forces` | `solver_coupled_admm.py:3742` |
| `_update_admm_dual` | `solver_coupled_admm.py:4104` |
| ADMM 数学 kernel | `newton/_src/solvers/coupled/admm_utils.py` |
| u/λ 更新公式 | `admm_utils.py:700-740` |
| 界面权重 W | `admm_utils.py:26-47` |
| 力 splat kernel | `admm_utils.py:1184` |
| 基类 step/分发/调和 | `newton/_src/solvers/coupled/solver_coupled.py` |

---

## 11. 构造期完整调用栈

从 `Example.__init__` 到所有内部函数的完整调用树（带行号）：

```text
Example.__init__                                          [example:102]
  ├─ ModelBuilder().add_ground_plane() / add_cloth_grid() / add_body() / ...
  │    └─ SolverCoupledADMM.add_body_particle_attachment()       [admm:515]
  │         ├─ cls.register_custom_attributes(builder)            [admm:434]
  │         │    ├─ builder.add_custom_frequency(...)              # 注册 "coupling:body_particle_attachment"
  │         │    └─ builder.add_custom_attribute(...) × 6         # body/particle/point/stiffness/damping/enabled
  │         └─ builder.add_custom_values(**{...})                 # 写入一行标注数据
  │
  ├─ builder.finalize() -> model
  ├─ model.soft_contact_ke = 0.0                                # 关掉普通软接触
  │
  └─ SolverCoupledADMM(model, entries, coupling)                [admm:646]
       ├─ _validate_config(coupling)                            [admm:688]  校验 rho/gamma/baumgarte 等
       ├─ _init_admm_joint_proxy_visibility(model, entries, dest_entries)  [admm:750]
       │    ├─ _build_owner_map(body_count, ...)                 [base:362]  建 body/joint 归属表
       │    ├─ add_proxy_body(dst, src, body)                    # 球铰邻居 body 加入 keep 集
       │    └─ _add_admm_joint_proxy_topology_paths(...)         [admm:844]  补齐 incoming-tree 路径
       │
       ├─ super().__init__(model, entries, coupling)  == SolverCoupled.__init__  [base:293]
       │    ├─ _build_attribute_projections()                    [base:329]  属性投影规则
       │    ├─ _validate_entry_names()                           [base:355]  entry 名唯一
       │    ├─ _build_owner_map(body/particle/joint/shape)       [base:362]  四张归属表
       │    └─ _build_entries()                                  [base:389]  ★ 核心：建视图+子求解器
       │         for each entry:
       │         ├─ ModelView(model, name)                       [model_view:79]
       │         ├─ _entry_proxy_body_keep_indices(name)         [admm:894]  返回代理 body 集合
       │         │    (基类默认空；ADMM 覆盖返回 _admm_joint_proxy_body_keep)
       │         ├─ view.disable_body_dynamics(非owned非proxy)   [model_view:407]  逆惯量置零
       │         ├─ view.mark_proxy_bodies(proxy集)              [model_view:594]  打 PROXY 标志
       │         ├─ view.zero_particle_mass / disable_particles  [model_view:638/687]
       │         ├─ view.disable_joints(非owned非proxy)          [model_view:659]
       │         ├─ _apply_entry_shape_visibility(view, cfg, proxy_keep)  [base:573]
       │         ├─ _compact_entry_view_if_needed(...)           [base:761]  尝试压缩编号
       │         │    └─ _apply_compact_entry_view(...)          [base:1148]
       │         ├─ _customize_compact_view(view)                [admm:991]  ADMM 覆盖
       │         │    └─ _disable_admm_joint_proxy_shape_collisions(view)  [admm:995]
       │         │         └─ wp.launch(_disable_proxy_shape_collisions_kernel)
       │         ├─ cfg.configure_view(view)                     # 用户回调（本例无）
       │         ├─ _filter_shape_contact_pairs(view)            [base:608]
       │         ├─ _build_entry_index_maps(view, index_lists)   [base:689]  local↔global 映射
       │         ├─ solver = cfg.solver(view)                    # 实例化 SolverMuJoCo / SolverVBD
       │         ├─ _require_supports_coupling(solver)           # 校验实现 CouplingInterface
       │         └─ 存入 self._entries[name] = SolverEntry(...)
       │
       │    _after_entries_constructed()                        [admm:900]  ADMM 覆盖
       │    ├─ _refresh_admm_joint_proxy_view_maps()             [admm:908]  global->local 重映射
       │    └─ _cache_admm_joint_proxy_effective_masses()        [admm:918]  算代理体等效质量
       │
       │    for each entry: entry.state_0 = view.state(); entry.state_1 = ...
       │    _after_entry_states_created()                        [admm:994]  ADMM 覆盖
       │    └─ super()._after_entry_states_created()  -> _refresh_gravity_accelerations()  [base:1660]
       │         └─ _refresh_entry_gravity_acceleration(entry)   [base:1675]
       │              └─ entry.solver.coupling_eval_gravity_acceleration(...)  # 钩子
       │
       ├─ _setup_admm(coupling)                                 [admm:1022]  ★ ADMM 专属初始化
       │    ├─ for each entry: 分配 _AdmmBuffers (body_q_n/qd_n/qd_k/body_f/...)
       │    ├─ _entry_body_sets / _entry_particle_sets           # 建 body/particle 集合
       │    ├─ for each entry:
       │    │    ├─ _setup_admm_effective_mass_buffers(entry, buf)  [admm:1843]
       │    │    │    ├─ _setup_admm_effective_mass_endpoint_buffers(...)  [admm:1904]
       │    │    │    ├─ _populate_admm_body_effective_mass_buffer(...)    [admm:1966]
       │    │    │    │    └─ entry.solver.coupling_eval_effective_mass_block(...)  # 钩子
       │    │    │    │         wp.launch(scatter_body_effective_mass_block_kernel)
       │    │    │    └─ _populate_admm_particle_effective_mass_buffer(...) [admm:2023]
       │    │    └─ _setup_admm_body_joint_qd_proximal_map(entry, buf)     [admm:1133]
       │    │
       │    ├─ _build_admm_joint_groups(coupling)                [admm:2396]  关节->约束
       │    │    for each joint:
       │    │    └─ _cross_solver_joint_entries(joint, parent, child)  [admm:2384]
       │    │         返回 (child_entry, parent_entry) 或 None
       │    │         按 joint_type 分类到 point/angular/revolute_angular/friction items
       │    │    最后组装成 _AdmmRigidRigidAttachmentGroup 等列表
       │    │
       │    ├─ _build_admm_body_particle_attachment_groups()     [admm:2674]  附件标注->约束
       │    │    for each attachment row:
       │    │    ├─ _entry_name_for_body(body) / _entry_name_for_particle(particle)
       │    │    ├─ _compute_interface_weights(...)              [admm:1937]
       │    │    │    └─ wp.launch(compute_interface_weights_kernel)
       │    │    └─ 组装成 _AdmmRigidParticleAttachmentGroup
       │    │
       │    ├─ _setup_admm_contact_specs(coupling)               [admm:2057]  本例 contact_pairs 为空，跳过
       │    │    (若有: _discover_rigid_particle_contact_specs / _discover_rigid_rigid_contact_specs
       │    │     建 CollisionPipeline / _build_collision_*_contact_groups)
       │    │
       │    └─ if gamma > 0:
       │         ├─ _refresh_admm_proximal_masks()               [admm:1195]
       │         │    ├─ _mark_static_admm_proximal_masks()      [admm:1540]  静态约束(附件/关节)
       │         │    │    └─ _accumulate_body_point_proximal_lump / _accumulate_global_indices_proximal_lump
       │         │    │         wp.launch(accumulate_*_proximal_lump_kernel)
       │         │    ├─ _mark_dynamic_contact_admm_proximal_masks()  [admm:1587]  (本例无)
       │         │    └─ _mark_joint_qd_proximal_masks_from_bodies()  [admm:1707]
       │         │         wp.launch(accumulate_joint_qd_factor_from_body_proximal_lump_kernel)
       │         └─ _refresh_admm_proximal_view_overrides(...)   [admm:1742]
       │              ├─ view._refresh_body_inertial_properties(...)
       │              ├─ view.add_body_lumped_inertia(...)
       │              ├─ view._refresh_particle_mass_properties(...)
       │              ├─ view.add_particle_lumped_mass(...)
       │              └─ entry.solver.notify_model_changed(BODY_INERTIAL_PROPERTIES)
       │
       └─ _apply_cached_admm_joint_proxy_effective_masses()      [admm:947]
            └─ _apply_body_inertia_override(dst, body_ids, proxy_mass, proxy_inertia)  [base:1818]
                 ├─ view.set_body_inertial_properties(...)
                 └─ entry.solver.notify_model_changed(BODY_INERTIAL_PROPERTIES)
```

---

## 12. 每步完整调用栈

从 `Example.step()` 到每个 kernel 的完整调用树：

```text
Example.step()                                              [example:263]
  └─ _launch_frame_graph() 或 simulate()
       └─ Example.simulate()                                [example:251]
            for _ in range(sim_substeps):                   # 8 个子步
            ├─ state_0.clear_forces()
            ├─ apply_coupled_viewer_forces(self, state_0)   [examples/__init__.py:263]
            │    └─ viewer.apply_forces(state)              # 鼠标拖拽外力（仅 combined view）
            │
            ├─ # model.collide(...)  ← 本例注释掉
            │
            ├─ solver.step(state_0, state_1, control, contacts, sim_dt)
            │    └─ SolverCoupled.step()                    [base:1947]  基类
            │         ├─ _distribute_state(state_in, dt)    [base:2027]
            │         │    for each entry:
            │         │    ├─ _copy_state_to_entry(state_in, entry.state_0, entry)
            │         │    │    wp.launch(_scatter_*_state_mapped)  # 父->entry 拷贝
            │         │    └─ _notify_input_state_update(entry, flags, dt)  [base:2240]
            │         │         └─ entry.solver.coupling_notify_input_state_update(...)  # 钩子
            │         │
            │         ├─ _step_coupled(state_in, state_out, control, contacts, dt)  [admm:2762]  ★ ADMM 覆盖
            │         │    │
            │         │    ├─ _refresh_collision_contact_groups(state_in)  [admm:2825]  本例跳过
            │         │    │    (若有: _admm_collision_pipeline.collide(...)
            │         │    │     wp.launch(contact_rr/rp/pp_*_kernel))
            │         │    │
            │         │    ├─ if gamma > 0:
            │         │    │    ├─ _refresh_admm_proximal_masks()        [admm:1195]
            │         │    │    └─ _refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)
            │         │    │                                                              [admm:1742]
            │         │    │
            │         │    ├─ for each entry: 快照 state_0 -> buf.*_n 和 *_k
            │         │    │    wp.copy(buf.body_q_n, entry.state_0.body_q)
            │         │    │
            │         │    ├─ _admm_begin_step(dt)          [admm:3382]  算 u_target
            │         │    │    for each group:
            │         │    │    └─ wp.launch(attach_*_compute_u_target_kernel / contact_*_compute_u_min_kernel)
            │         │    │
            │         │    └─ for k in range(iterations):   # 2 轮 ADMM 迭代
            │         │         │
            │         │         ├─ A. for each entry:
            │         │         │    _prepare_admm_iteration_state(entry, buf, state_in, dt, iteration_restart=k>0)
            │         │         │                                                  [admm:3605]
            │         │         │    ├─ wp.copy(entry.state_0.body_q, buf.body_q_n)   # 回退到步起始
            │         │         │    ├─ _notify_input_state_update(entry, flags, dt, iteration_restart)
            │         │         │    ├─ if apply_proximal:
            │         │         │    │    _apply_admm_velocity_proximal_shift(entry, buf, dt)  [admm:3551]
            │         │         │    │         ├─ wp.launch(velocity_proximal_shift_body_lumped_kernel)
            │         │         │    │         ├─ wp.launch(velocity_proximal_shift_particle_lumped_kernel)
            │         │         │    │         └─ wp.launch(velocity_proximal_shift_joint_lumped_kernel)
            │         │         │    ├─ if buf.body_f: 拷外力 + 重力补偿
            │         │         │    │    wp.launch(_copy_mapped_spatial_vector)
            │         │         │    │    wp.launch(body_gravity_compensation_lumped_kernel)
            │         │         │    └─ wp.launch(particle_gravity_compensation_lumped_kernel)
            │         │         │
            │         │         ├─ B. _accumulate_admm_forces(k, dt, refresh_jv=(k==0), initialize_contact_u=(k==0))
            │         │         │                                                  [admm:3742]
            │         │         │    for each _admm_rr_groups (rigid-rigid 附件，本例无):
            │         │         │    ├─ wp.launch(attach_rr_compute_Jv_kernel)
            │         │         │    └─ wp.launch(attach_rr_accumulate_forces_kernel)  -> buf_a.body_f, buf_b.body_f
            │         │         │    for each _admm_rp_groups (body-particle 附件，本例有1组):
            │         │         │    ├─ wp.launch(attach_rp_compute_Jv_kernel)         # Jv = v_body锚点 - v_particle
            │         │         │    └─ wp.launch(attach_rp_accumulate_forces_kernel)  -> body_f, particle_f
            │         │         │    for each _admm_rr_angular_groups / revolute_angular_groups (球铰角约束):
            │         │         │    ├─ wp.launch(attach_rr_angular_*_compute_Jv_kernel)
            │         │         │    └─ wp.launch(attach_rr_angular_*_accumulate_forces_kernel)
            │         │         │    for each _admm_dynamic_*_contact_groups (接触，本例无):
            │         │         │    └─ wp.launch(contact_*_accumulate_forces_kernel)
            │         │         │
            │         │         ├─ C. for each entry:
            │         │         │    _apply_admm_force_inputs(entry, buf, dt)         [admm:3684]
            │         │         │    ├─ _set_local_body_force_input(entry, buf.body_f, dt)   [base:2151]
            │         │         │    │    ├─ _clear_body_force_input(entry)            [base:2114]  body_f.zero_()
            │         │         │    │    ├─ _copy_prefix(entry.state_0.body_f, body_f)
            │         │         │    │    └─ _notify_input_state_update(entry, BODY_F, dt)
            │         │         │    └─ _set_local_particle_force_input(entry, buf.particle_f, dt)  [base:2203]
            │         │         │
            │         │         ├─ D. for each entry:
            │         │         │    _step_entry(entry, control, contacts, dt)        [base:2257]
            │         │         │    ├─ _contacts_for_entry(entry, contacts)         [base:2298]  (本例 contacts 为空)
            │         │         │    ├─ _copy_control_to_entry(control, entry)
            │         │         │    └─ entry.solver.step(state_0, state_1, control, contacts, dt)
            │         │         │         # mjc.step(...) / vbd.step(...)  ← 子求解器独立步进
            │         │         │
            │         │         ├─ E. for each entry: 快照 state_1.qd -> buf.*_k
            │         │         │    wp.copy(buf.body_qd_k, entry.state_1.body_qd)
            │         │         │
            │         │         └─ F. _update_admm_dual(k, dt)                       [admm:4104]
            │         │              for each group:
            │         │              ├─ wp.launch(*_compute_Jv_kernel)  # 用 state_1 新速度重算 Jv
            │         │              ├─ if 二次型: _update_admm_quadratic_dual(group)  [admm:3708]
            │         │              │    ├─ wp.launch(u_update_quadratic_kernel)     # u 更新
            │         │              │    └─ wp.launch(lambda_update_kernel)          # λ += ρW(u-Jv)
            │         │              └─ if 接触: _update_admm_contact_dual(group)     [admm:3732]
            │         │                   ├─ _update_admm_contact_u(group)            [admm:3690]
            │         │                   │    └─ wp.launch(contact_u_update_kernel)  # 库仑摩擦投影
            │         │                   └─ wp.launch(contact_lambda_update_kernel)
            │         │
            │         ├─ _copy_state(state_in, state_out)
            │         └─ _reconcile_state(state_out)       [base:2040]
            │              for each entry:
            │              ├─ wp.launch(_scatter_body_state_mapped)    # entry.state_1 -> state_out.body_q
            │              ├─ wp.launch(_scatter_particle_state_mapped)
            │              └─ wp.launch(_scatter_scalar_state_mapped)  # joint_q/qd
            │
            ├─ eval_ik(model, state_1, joint_q, joint_qd)  # 逆运动学：body_q -> joint_q
            └─ state_0, state_1 = state_1, state_0         # 乒乓交换
```

---

## 13. 函数作用卡片（构造期）

每个函数的职责、参数、被谁调用、调用了谁。

### 用户层

#### `Example.__init__` `[example:102]`

- **职责**：搭建场景、划分 ownership、构造耦合求解器、分配状态、捕获 CUDA graph。
- **被调用**：`__main__` 入口。
- **调用**：`ModelBuilder` 系列、`add_body_particle_attachment`、`finalize`、`SolverCoupledADMM`、`model.state()`、`configure_coupled_view`、`eval_fk`、`self.capture`。

#### `Example.capture` `[example:242]`

- **职责**：把 `simulate()` 捕获成 CUDA graph，每帧重放省 CPU-GPU 往返。
- **调用**：`_capture_frame_graph`（`wp.ScopedCapture`）。

### ADMM 构造入口

#### `SolverCoupledADMM.__init__` `[admm:646]`

- **职责**：ADMM 求解器构造入口。校验配置、初始化代理体可见性、调基类建视图、再调 `_setup_admm` 建 ADMM 专属结构。
- **参数**：`model`、`entries`（子求解器配置）、`coupling`（`Config`）。
- **调用**：`_validate_config`、`_init_admm_joint_proxy_visibility`、`super().__init__`、`_setup_admm`、`_apply_cached_admm_joint_proxy_effective_masses`。

#### `add_body_particle_attachment` `[admm:515]` (classmethod)

- **职责**：在 builder 上注册一条 body-particle 附件标注（写自定义频率属性）。
- **参数**：`builder`、`body`、`particle`、`body_point`、`stiffness`、`damping`、`enabled`。
- **调用**：`register_custom_attributes`、`builder.add_custom_values`。

#### `register_custom_attributes` `[admm:434]` (classmethod)

- **职责**：注册 `coupling:body_particle_attachment` 自定义频率及其 6 个属性（body、particle、body_point、stiffness、damping、enabled）。
- **调用**：`builder.add_custom_frequency`、`builder.add_custom_attribute × 6`。

### 代理体可见性

#### `_init_admm_joint_proxy_visibility` `[admm:750]`

- **职责**：找出跨求解器关节（parent/child 分属不同 entry），把邻居 body 加入各视图的"保留为代理"集合。只加到 `joint_proximal_destination_entries` 指定的视图。
- **调用**：`_build_owner_map`、`_add_admm_joint_proxy_topology_paths`。
- **产出**：`_admm_joint_proxy_body_keep`、`_admm_joint_proxy_joint_keep`、`_admm_joint_proxy_mappings`。

#### `_add_admm_joint_proxy_topology_paths` `[admm:844]`

- **职责**：沿 articulation 的 incoming-tree 补齐路径--如果一个 proxy body 的父关节链上有缺失的 body，把必要的祖先 body 也加进 keep 集，确保关节可实例化。

### 基类视图构建

#### `SolverCoupled._build_entries` `[base:389]`

- **职责**：核心构造。对每个 entry 建 ModelView、禁用非 owned 实体、标记 proxy、压缩视图、实例化子求解器。
- **调用**：`ModelView`、`disable_body_dynamics`、`mark_proxy_bodies`、`_compact_entry_view_if_needed`、`_customize_compact_view`、`_build_entry_index_maps`、`cfg.solver(view)`。

#### `_build_owner_map` `[base:362]`

- **职责**：给 body/particle/joint/shape 建 `owner[id] = entry_idx` 表，校验无重复所有权。

#### `_compact_entry_view_if_needed` `[base:761]`

- **职责**：尝试把 entry 相关实体重排成连续局部编号。条件苛刻（关节父子在集合内、articulation 完整等），失败则回退到全模型布局。
- **调用**：`_ordered_world_subset`、`_compact_index_lists`、`_apply_compact_entry_view`。

### ADMM 专属构造

#### `_setup_admm` `[admm:1022]`

- **职责**：ADMM 初始化总入口。分配缓冲、算等效质量、建约束 group、设接触管线、刷 proximal mask。
- **调用**：`_setup_admm_effective_mass_buffers`、`_setup_admm_body_joint_qd_proximal_map`、`_build_admm_joint_groups`、`_build_admm_body_particle_attachment_groups`、`_setup_admm_contact_specs`、`_refresh_admm_proximal_masks`、`_refresh_admm_proximal_view_overrides`。

#### `_setup_admm_effective_mass_buffers` `[admm:1843]`

- **职责**：为每个 entry 分配等效质量缓冲，调子求解器钩子填充。
- **调用**：`_setup_admm_effective_mass_endpoint_buffers`、`_populate_admm_body_effective_mass_buffer`、`_populate_admm_particle_effective_mass_buffer`。
- **钩子**：`entry.solver.coupling_eval_effective_mass_block` / `coupling_eval_effective_mass`。

#### `_build_admm_joint_groups` `[admm:2396]`

- **职责**：遍历所有关节，把跨求解器关节（parent/child 分属不同 entry）转成 ADMM 约束 group。BALL -> 平移附件 + 角摩擦；REVOLUTE -> 平移附件 + 转角约束 + 摩擦；FIXED -> 平移附件。
- **调用**：`_cross_solver_joint_entries`、`_revolute_axis_frames_from_rows`。
- **产出**：`_admm_rr_groups`、`_admm_rr_angular_groups`、`_admm_rr_revolute_angular_groups`、`_admm_rr_angular_friction_groups`。

#### `_cross_solver_joint_entries` `[admm:2384]`

- **职责**：判断一个关节是否跨求解器（parent/child 分属不同 entry）。若是，返回 `(child_entry, parent_entry)`；若关节被某 entry 拥有则报错（避免双算）。

#### `_build_admm_body_particle_attachment_groups` `[admm:2674]`

- **职责**：扫描 `coupling:body_particle_attachment` 自定义频率的所有行，把 body/particle 分属不同 entry 的行转成 `_AdmmRigidParticleAttachmentGroup`。
- **调用**：`_entry_name_for_body`、`_entry_name_for_particle`、`_compute_interface_weights`、`_require_effective_mass`。
- **产出**：`_admm_rp_groups`。

#### `_compute_interface_weights` `[admm:1937]`

- **职责**：调 `compute_interface_weights_kernel` 算每条约束的界面权重 W（两侧等效质量的调和几何平均）。

#### `_refresh_admm_proximal_masks` `[admm:1195]`

- **职责**：清零所有 proximal mask/mass 缓冲，重新标记参与约束的实体并累加虚拟质量 lump。
- **调用**：`_mark_static_admm_proximal_masks`、`_mark_dynamic_contact_admm_proximal_masks`、`_mark_joint_qd_proximal_masks_from_bodies`。

#### `_refresh_admm_proximal_view_overrides` `[admm:1742]`

- **职责**：把 proximal 虚拟质量写到视图（`add_body_lumped_inertia`、`add_particle_lumped_mass`），并通知子求解器刷新。
- **调用**：`view._refresh_body_inertial_properties`、`view.add_body_lumped_inertia`、`entry.solver.notify_model_changed`。

#### `_apply_cached_admm_joint_proxy_effective_masses` `[admm:947]`

- **职责**：把缓存的代理体等效质量（× mass_scale）写到目标视图，让代理体有合理惯量。
- **调用**：`_apply_body_inertia_override`。

---

## 14. 函数作用卡片（每步）

### 基类步进骨架

#### `SolverCoupled.step` `[base:1947]`

- **职责**：步进骨架。分发状态 -> 子类 `_step_coupled` -> 调和状态。
- **参数**：`state_in`、`state_out`、`control`、`contacts`、`dt`。
- **调用**：`_distribute_state`、`_step_coupled`、`_copy_state`、`_reconcile_state`。

#### `_distribute_state` `[base:2027]`

- **职责**：把父状态按所有权拷进各 entry 的 `state_0`，并通知子求解器刷新内部缓存。
- **调用**：`_copy_state_to_entry`（内部 launch `_scatter_*` kernel）、`_notify_input_state_update`。
- **参数**：`state_in`、`dt`、`iteration_restart`。

#### `_reconcile_state` `[base:2040]`

- **职责**：把各 entry 的 `state_1` 按所有权 scatter 回父状态。**proxy body 状态不回写**。
- **调用**：`wp.launch(_scatter_body_state_mapped)`、`_scatter_particle_state_mapped`、`_scatter_scalar_state_mapped`。

#### `_notify_input_state_update` `[base:2240]`

- **职责**：调子求解器钩子 `coupling_notify_input_state_update`，告知"框架刚改了你的输入状态/力"，让子求解器刷新派生缓存。
- **参数**：`entry`、`flags`（StateFlags 位掩码）、`dt`、`iteration_restart`。

#### `_step_entry` `[base:2257]`

- **职责**：步进单个子求解器，支持 in_place / 单步 / 多 substep 三种模式。
- **调用**：`_contacts_for_entry`、`_copy_control_to_entry`、`entry.solver.step`。
- **本例**：每个 entry substeps=1，直接 `entry.solver.step(state_0, state_1, control, contacts, dt)`。

#### `_contacts_for_entry` `[base:2298]`

- **职责**：把父级 Contacts 按 shape 可见性过滤成 entry 局部接触缓冲。本例 contacts 为空，实际跳过。

### ADMM 步进核心

#### `_step_coupled` `[admm:2762]` (override)

- **职责**：ADMM 主循环。刷新碰撞 -> 快照步起始 -> 算 u_target -> 迭代 N 轮（回退+算力+步进+更新λ）。
- **调用**：`_refresh_collision_contact_groups`、`_refresh_admm_proximal_masks`、`_admm_begin_step`、`_prepare_admm_iteration_state`、`_accumulate_admm_forces`、`_apply_admm_force_inputs`、`_step_entry`、`_update_admm_dual`。

#### `_refresh_collision_contact_groups` `[admm:2825]`

- **职责**：若有接触 group，调内部 `CollisionPipeline.collide` 检测碰撞，把结果填进 ADMM 接触 group。本例无 `contact_pairs`，直接返回。

#### `_admm_begin_step` `[admm:3382]`

- **职责**：每步开始算各约束的 `u_target`（Baumgarte 位置修正目标）和接触的 `u_min`（非穿透下界）。
- **调用**：`wp.launch(attach_*_compute_u_target_kernel)`、`contact_*_compute_u_min_kernel`。
- **逻辑**：`baumgarte <= 0` 时 `u_target` 置零；否则 `u_target = (baumgarte/dt) · gap`（gap 是两点当前位置差）。

#### `_prepare_admm_iteration_state` `[admm:3605]`

- **职责**：每轮迭代开始，把 entry 状态回退到步起始 `*_n`，施加 proximal 速度位移，重置 `body_f/particle_f` 并拷入外力 + 重力补偿。
- **调用**：`wp.copy`（回退）、`_notify_input_state_update`、`_apply_admm_velocity_proximal_shift`、`wp.launch(_copy_mapped_spatial_vector)`、`wp.launch(body_gravity_compensation_lumped_kernel)`。
- **关键**：`iteration_restart=(k>0)` 标记是否非首轮，影响通知行为。

#### `_apply_admm_velocity_proximal_shift` `[admm:3551]`

- **职责**：用 proximal 虚拟质量给参与约束的 body/particle/joint 一个"预判速度偏移"，加速 ADMM 收敛。
- **调用**：`wp.launch(velocity_proximal_shift_body_lumped_kernel)`、`_particle`、`_joint` 三个 kernel。
- **公式**：`v_out = (m_base·v_n + m_lump·v_k) / (m_base + m_lump)`，即步起始速度与上一轮迭代速度的加权平均。

#### `_accumulate_admm_forces` `[admm:3742]`

- **职责**：对所有约束 group，算 Jv（若 `refresh_jv`）并把约束力 splat 到 `body_f/particle_f`。
- **参数**：`iteration_k`、`dt`、`refresh_jv`、`initialize_contact_u`。
- **调用**：对每组 group launch 对应的 `*_compute_Jv_kernel` 和 `*_accumulate_forces_kernel`。
- **本例**：`_admm_rp_groups`（body-particle 附件）+ 球铰的 angular group。

#### `_apply_admm_force_inputs` `[admm:3684]`

- **职责**：把 `buf.body_f/particle_f`（累积的约束力）设为 entry 的输入力。
- **调用**：`_set_local_body_force_input`、`_set_local_particle_force_input`。

#### `_set_local_body_force_input` `[base:2151]`

- **职责**：替换 entry 的 body 力输入。先清零，再拷入 buf 的力，再通知子求解器。
- **调用**：`_clear_body_force_input`（`body_f.zero_()`）、`_copy_prefix`、`_notify_input_state_update(BODY_F)`。

#### `_update_admm_dual` `[admm:4104]`

- **职责**：每轮迭代结束，用子求解器算出的新速度（`state_1.qd`）重算 Jv，更新对偶变量 `u` 和 `λ`。**λ 跨轮累积（暖启动）**。
- **调用**：`wp.launch(*_compute_Jv_kernel)`（用 `state_1`）、`_update_admm_quadratic_dual` / `_update_admm_contact_dual`。

#### `_update_admm_quadratic_dual` `[admm:3708]`

- **职责**：二次型约束（附件/关节）的 u/λ 更新。
- **调用**：`wp.launch(u_update_quadratic_kernel)`、`wp.launch(lambda_update_kernel)`。
- **公式**：`u = (ρW²Jv + κ·u_target − Wλ)/(κ+c+ρW²)`，`λ += ρW(u−Jv)`。

#### `_update_admm_contact_dual` `[admm:3732]`

- **职责**：接触约束的 u/λ 更新。u 是带库仑摩擦的局部投影。
- **调用**：`_update_admm_contact_u`（`contact_u_update_kernel` -> `solve_coulomb_isotropic`）、`contact_lambda_update_kernel`。

---

## 15. Kernel 清单与作用

所有 GPU kernel 都在 `admm_utils.py`、`proxy_utils.py`、`solver_coupled.py`、`solver_coupled_admm.py`、`model_view.py`。

### 等效质量与权重

| Kernel | 作用 |
| --- | --- |
| `compute_interface_weights_kernel` | 算每条约束的 W = sqrt(m_a·m_b/(m_a+m_b)) |
| `scatter_body_effective_mass_block_kernel` | 把钩子算出的局部等效质量 scatter 到全局缓冲 |
| `scatter_effective_mass_kernel` | 标量等效质量 scatter（particle 或无惯量 body） |
| `_coupling_eval_effective_mass*_kernel` | `CouplingInterface` 默认实现：从 inv_mass 推等效质量 |
| `_compute_body_inertia_scalar_kernel` | 算 body 惯量标量 = trace(I)/3 |

### Proximal（虚拟质量）

| Kernel | 作用 |
| --- | --- |
| `accumulate_body_point_proximal_lump_kernel` | 给 body 累加点质量 lump + 惯量 lump（附件锚点处） |
| `accumulate_body_angular_proximal_lump_kernel` | 给 body 累加角惯量 lump |
| `accumulate_global_indices_proximal_lump_kernel` | 给 particle 累加质量 lump（按全局 id 映射） |
| `accumulate_indices_proximal_lump_kernel` | 同上（按局部 id） |
| `accumulate_active_*_proximal_lump_kernel` | 动态接触的版本（按 active_count 过滤） |
| `accumulate_joint_qd_factor_from_body_proximal_lump_kernel` | 把 body proximal lump 传播到关节 DOF 因子 |
| `mark_indices_mask_kernel` / `mark_global_indices_mask_kernel` | 标记参与约束的实体（mask=1） |

### Proximal 速度位移

| Kernel | 作用 | 公式 |
| --- | --- | --- |
| `velocity_proximal_shift_body_lumped_kernel` | body 速度预判偏移 | `v = (m_base·v_n + m_lump·v_k)/(m_base+m_lump)` |
| `velocity_proximal_shift_particle_lumped_kernel` | particle 速度预判偏移 | 同上（标量质量） |
| `velocity_proximal_shift_joint_lumped_kernel` | 关节 DOF 速度偏移 | `v = (v_n + factor·v_k)/(1+factor)` |
| `body_gravity_compensation_lumped_kernel` | 补偿 proximal 部分的重力，避免双计 | 从 body_f 减去 m_lump·g |
| `particle_gravity_compensation_lumped_kernel` | particle 版重力补偿 | 同上 |

### Jv 计算（约束雅可比 × 速度）

| Kernel | 作用 | 公式 |
| --- | --- | --- |
| `attach_rp_compute_Jv_kernel` | body-particle 附件 Jv | `Jv = velocity_at_point(body_qd, arm) − particle_qd` |
| `attach_rr_compute_Jv_kernel` | body-body 附件 Jv | 两锚点速度差 |
| `attach_rr_angular_compute_Jv_kernel` | body-body 角附件 Jv | 角速度差 |
| `attach_rr_revolute_angular_local_compute_Jv_kernel` | 转角约束 Jv | 局部坐标系下角速度差 |
| `contact_rr/rp/pp_compute_Jv_kernel` | 接触 Jv | 接触点相对速度 |

### u_target / u_min（Baumgarte）

| Kernel | 作用 | 公式 |
| --- | --- | --- |
| `attach_rp_compute_u_target_kernel` | body-particle 附件目标速度 | `u_target = (baumgarte/dt)·gap` |
| `attach_rr_compute_u_target_kernel` | body-body 附件目标速度 | 同上 |
| `attach_rr_angular_*_compute_u_target_kernel` | 角附件目标 | 角度差 × baumgarte/dt |
| `contact_rr/rp/pp_compute_u_min_kernel` | 接触非穿透下界 | 基于穿透深度 |

### 力 splat（J^T W 力施加）

| Kernel | 作用 | 公式 |
| --- | --- | --- |
| `attach_rp_accumulate_forces_kernel` | body-particle 附件力 | `F = W·(λ + ρW(u−Jv))`，splat 到 body_f + particle_f |
| `attach_rr_accumulate_forces_kernel` | body-body 附件力 | 同上，splat 到两个 body_f |
| `attach_rr_angular_accumulate_forces_kernel` | 角附件力矩 | 角版本 |
| `attach_rr_revolute_angular_local_accumulate_forces_kernel` | 转角力矩 | 局部坐标版 |
| `contact_rr/rp/pp_accumulate_forces_kernel` | 接触力 | 含法向 + 切向摩擦 |

### u / λ 更新（ADMM 对偶）

| Kernel | 作用 | 公式 |
| --- | --- | --- |
| `u_update_quadratic_kernel` | 二次型 u 更新 | `u = (ρW²Jv + κ·u_target − Wλ)/(κ+c+ρW²)` |
| `lambda_update_kernel` | 二次型 λ 更新 | `λ += ρW(u−Jv)` |
| `contact_u_update_kernel` | 接触 u 更新 | 库仑摩擦投影（`solve_coulomb_isotropic`） |
| `contact_lambda_update_kernel` | 接触 λ 更新 | `λ += ρW(u−Jv)`（带 active 过滤） |
| `joint_box_friction_u_update_kernel` | 关节盒摩擦 u | 逐轴 soft-threshold |

### 状态分发/调和（基类）

| Kernel | 作用 |
| --- | --- |
| `_scatter_body_state_mapped` | entry.body_q -> 父 state.body_q（按 body_indices + global_to_local） |
| `_scatter_particle_state_mapped` | particle 版本 |
| `_scatter_scalar_state_mapped` | joint_q / joint_qd 版本 |
| `_copy_mapped_spatial_vector` | 按映射拷 spatial_vector（body_f） |
| `_copy_mapped_vec3` | 按映射拷 vec3（particle_f） |

### 视图与代理（构造期）

| Kernel | 作用 |
| --- | --- |
| `_disable_proxy_shape_collisions_kernel` | 关掉 proxy body 的 shape 碰撞标志 |
| `_clear_shape_collision_flags_kernel` | 清掉非 owned shape 的碰撞标志 |
| `_mark_visible_shape_contact_pairs_kernel` | 标记 entry 可见的接触对 |
| `_compact_visible_shape_contact_pairs_kernel` | 紧凑化可见接触对 |
| `_remap_reference_kernel` | 压缩视图时重映射引用索引 |
| `_remap_shape_body_kernel` | 重映射 shape->body 引用 |
| `model_view._zero_body_inverse_dynamics_kernel` | disable_body_dynamics：逆惯量置零 |
| `model_view._mark_body_flag_kernel` | mark_proxy_bodies：置 PROXY 标志 |
| `model_view._scale_body_mass_kernel` | scale_body_mass |
| `model_view._add_body_lumped_inertia_kernel` | add_body_lumped_inertia（proximal 用） |

---

## 16. 完整数据流总结

把所有函数串成一个数据流图：

```text
[构造期 - 一次性]
  Model + attachments + joints
    │
    ├─> _init_admm_joint_proxy_visibility  -> 代理 body keep 集
    ├─> _build_entries (基类)              -> ModelView × 2 + 子求解器实例
    ├─> _setup_admm_effective_mass_buffers -> buf.body_effective_mass (钩子填充)
    ├─> _build_admm_joint_groups           -> _admm_rr_angular_groups (球铰)
    ├─> _build_admm_body_particle_attachment_groups -> _admm_rp_groups (球-布附件)
    ├─> _refresh_admm_proximal_masks       -> buf.body_proximal_mass (虚拟质量)
    └─> _refresh_admm_proximal_view_overrides -> view 惯量覆盖

[每步 - 重复]
  state_in
    │
    ├─> _distribute_state -> entry.state_0
    │
    ├─> _step_coupled (ADMM):
    │      快照 state_0 -> buf.*_n
    │      _admm_begin_step -> u_target
    │      │
    │      └─> for k in 2 轮:
    │           _prepare_admm_iteration_state  (回退 *_n + proximal 位移 + 重置力)
    │           _accumulate_admm_forces        (Jv + F=W·(λ+ρW(u−Jv)) -> body_f/particle_f)
    │           _apply_admm_force_inputs       (buf.body_f -> entry.state_0.body_f)
    │           _step_entry × 2                (mjc.step + vbd.step)
    │           快照 state_1.qd -> buf.*_k
    │           _update_admm_dual              (重算 Jv + 更新 u/λ，λ 暖启动)
    │
    ├─> _reconcile_state -> state_out (只写 owned)
    └─> eval_ik (body_q -> joint_q)
```

> 🔑 **核心闭环**：每轮 ADMM 用 `λ`（上一轮的记忆）算约束力 -> 推子求解器 -> 看新速度是否让界面 Jv 接近 u -> 若没对齐，更新 `λ` 加大力度 -> 下一轮。`λ` 是这个闭环的"积分器"，把瞬时速度不匹配累积成持续的约束力，直到界面收敛。

