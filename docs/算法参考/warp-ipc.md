# warp-ipc 项目分析

**项目位置**: `C:\csy_work\CG\Engine\warp-ipc`

---

## 1. 项目概述

warp-ipc 是一个基于 NVIDIA Warp 框架实现的 IPC (Incremental Potential Contact) 求解器，主要用于可变形体的接触仿真。项目使用 Python + Warp 的混合架构，核心算法在 GPU 上执行。

### 核心特性
- 支持 FEM (有限元)、ABD (Affine Body Dynamics)、Shell (壳) 等多种物理模型
- IPC barrier 函数防止穿透
- GPU BVH 加速的碰撞检测
- Newton 迭代求解 + PCG 线性求解器
- 支持摩擦接触
- Codim-IPC 支持薄壳/布料仿真

---

## 2. 项目架构

```
warp_ipc/
├── core/                 # 核心数据结构和类型定义
│   ├── types.py          # Warp 向量/矩阵类型 (vec3d, mat33d, etc.)
│   └── sim_state.py      # 仿真状态管理
│
├── engine/               # IPC 仿真引擎核心
│   ├── ipc_engine.py     # 主引擎 (advance 主循环)
│   ├── energy_gradient.py # 能量/梯度计算
│   ├── solver_dispatch.py # 求解器调度
│   ├── mesh_contact.py   # mesh-mesh 接触处理
│   ├── collision.py      # 碰撞检测混合类
│   └── articulation.py   # 关节约束
│
├── solver/               # 线性求解器
│   ├── newton.py         # Newton 迭代求解器
│   ├── pcg.py            # PCG (预条件共轭梯度)
│   ├── dense.py          # 小规模稠密直接求解
│   └ cpu_newton.py       # CPU Newton (小系统优化)
│
├── collision/            # 碰撞检测
│   ├── gpu_bvh_broad_phase.py  # GPU BVH 宽相检测
│   ├── gpu_broad_phase.py      # GPU 宽相备用
│   └ collision_detection.py    # 碰撞检测辅助
│
├── kernels/              # Warp GPU 内核
│   ├── barrier.py        # IPC barrier 函数
│   ├── distance.py       # 距离计算 (PP/PE/PT/EE)
│   ├── distance_hessian.py # 距离 Hessian
│   ├── kinetic.py        # 动能/预测位置
│   ├── friction.py       # 摩擦力计算
│   ├── mesh_contact.py   # mesh 接触能量/梯度
│   ├── mesh_friction.py  # mesh 摩擦
│   ├── mesh_ccd.py       # mesh 连续碰撞检测
│   ├── constitution.py   # SNH/NH 弹性模型
│   ├── abd.py            # ABD 相关内核
│   └── ...               # 其他物理模型内核
│
├── assembly/             # 系统矩阵组装
│   └ sparse.py           # BSR 稀疏矩阵组装
│
├── io/                   # 输入输出
│   └ mesh_loader.py      # mesh 加载
│    scene_io.py          # 场景 IO
│
└── render/               # 渲染
    ├── polyscope_gui.py  # Polyscope GUI
    └ sim_renderer.py     # 仿真渲染器
```

---

## 3. IPC 算法核心流程

### 3.1 仿真主循环 (advance)

**伪代码流程**:

```
advance(frame):
    1. 保存前一帧位置: x_prev = x
    2. 速度阻尼: v *= (1 - damping)
    3. 预测位置: x_tilde = x + v*dt + g*dt^2
    4. 碰撞检测:
       - 半平面接触检测
       - Mesh-mesh 候选对生成 (BVH 宽相)
    5. 自适应 kappa 计算 (可选)
    6. Newton 求解:
       for iter in max_iter:
           a. 计算能量 E(x)
           b. 计算梯度 g = dE/dx
           c. 求解 H*dx = -g (PCG 或直接求解)
           d. CCD 过滤步长 alpha
           e. 回溯线搜索: find alpha s.t. E(x + alpha*dx) < E(x)
           f. 收敛检查:
              - 最大位移 < velocity_tol * dt
              - ABD transrate < transrate_tol * dt
              - CCD alpha >= ccd_tol
    7. 更新速度: v = (x_new - x_prev) / dt
    8. 同步 ABD mesh 位置
```

### 3.2 能量函数组成

总能量 = 动能 + 弹性势能 + Barrier 接触能 + 摩擦能 + 约束能

```
E_total = E_kinetic + E_elastic + E_barrier + E_friction + E_constraints

其中:
- E_kinetic = 0.5 * m * ||x - x_tilde||^2
- E_elastic = SNH/NH/FEM 能量 (积分形式)
- E_barrier = IPC barrier 函数 (防止穿透)
- E_friction = 摩擦耗散能 (lagged normal force)
- E_constraints = SPC/DBC/Joint 等约束能
```

---

## 4. IPC Barrier 函数

### 4.1 标准 IPC Barrier

数学定义:
```
B(D) = -κ * (D - d̂²)² * log(D / d̂²)   当 D < d̂²
B(D) = 0                                 当 D >= d̂²

其中:
- D: 平方距离 (point-point, point-edge, point-triangle, edge-edge)
- d̂: 激活距离阈值
- κ: barrier 刚度参数
```

**关键性质**:
- 当 D → 0 时, B(D) → ∞ (防止穿透)
- 当 D → d̂² 时, B(D) → 0 (连续过渡)
- 梯度和 Hessian 在激活域内光滑

### 4.2 Codim-IPC Barrier (支持薄壳)

对于有厚度 ξ 的薄壳/布料:
```
B(D) = -κ * (D - ξ² - d̂² - 2d̂ξ)² * log((D-ξ²)/(d̂²+2d̂ξ))
激活范围: ξ² < D < (ξ + d̂)²

其中 ξ = thickness (厚度参数)
```

### 4.3 Barrier 内核实现

```python
@wp.func
def barrier_function(D, d_hat_sq, kappa):
    if D >= d_hat_sq:
        return 0.0
    if D <= EPS:
        return LARGE * kappa
    diff = D - d_hat_sq
    return -kappa * diff * diff * log(D / d_hat_sq)

@wp.func
def barrier_gradient(D, d_hat_sq, kappa):
    # dB/dD = -κ * [2(D - d̂²)log(D/d̂²) + (D - d̂²)²/D]
    ...

@wp.func
def barrier_hessian(D, d_hat_sq, kappa):
    # d²B/dD²
    ...
```

---

## 5. 碰撞检测流程

### 5.1 宽相检测 (GPU BVH)

使用 Warp 的 LBVH (Linear BVH) 加速碰撞候选对生成:

```
build_candidates(positions, d_hat):
    1. 计算每个 primitive 的 AABB (扩展 d_hat)
    2. 构建 BVH (triangle, edge, vertex 各一个)
    3. BVH 查询生成候选对:
       - PT: 点-三角形候选
       - EE: 边-边候选
       - PE: 点-边候选 (codim only)
       - PP: 点-点候选 (codim only)
    4. 返回候选数组
```

**优化策略**:
- BVH refit: Newton 迭代中只更新 AABB, 不重建树结构 (~5x 加速)
- GPU-native 路径: 候选对生成完全在 GPU 上, 无 CPU 同步

### 5.2 窄相检测

计算每个候选对的精确距离:

```python
# 距离类型
PT_FACE      = 0   # 点在三角形内部
PT_EDGE_01   = 1   # 点在边 (t0-t1) 上最近
PT_EDGE_12   = 2
PT_EDGE_20   = 3
PT_VERTEX_0  = 4   # 点接近顶点 t0
PT_VERTEX_1  = 5
PT_VERTEX_2  = 6

EE_INTERIOR  = 0   # 边-边内部相交
EE_E0_START  = 1   # 边 e0 的起点最近
...
```

---

## 6. 距离计算

### 6.1 Point-Point (PP)

```
D = ||p0 - p1||²
∇D = [2(p0-p1), -2(p0-p1)]
H_D = 2*I_6x6 (block structure)
```

### 6.2 Point-Edge (PE)

```
t = clamp(dot(p-e0, e1-e0) / ||e1-e0||², 0, 1)
closest = e0 + t*(e1-e0)
D = ||p - closest||²

梯度: 对 p, e0, e1 各 3 分量
Hessian: 基于系数 (1, -(1-t), -t) 的块结构
```

### 6.3 Point-Triangle (PT)

7 种情况分类:
```
1. PT_VERTEX_0: 点接近 t0
2. PT_VERTEX_1: 点接近 t1
3. PT_VERTEX_2: 点接近 t2
4. PT_EDGE_01:  点在边 (t0,t1) 上最近
5. PT_EDGE_12:  点在边 (t1,t2) 上最近
6. PT_EDGE_20:  点在边 (t2,t0) 上最近
7. PT_FACE:     点在三角形内部投影最近
```

### 6.4 Edge-Edge (EE)

```
计算两条边的最近点参数 (s, t)
分类 9 种情况 ( interior + 4 endpoints + 4 PP degenerate)
```

---

## 7. Newton 求解器

### 7.1 主迭代流程

```python
solve(x_start, positions, energy_fn, gradient_fn, hessian_solve_fn):
    # 初始化
    x = x_start  # libuipc 用 x_prev, 不是 x_tilde

    for iter in max_iter:
        # 1. 计算梯度
        g = gradient_fn()
        neg_g = -g

        # 2. 求解线性系统
        dx = hessian_solve(neg_g)  # H*dx = -g

        # 3. 收敛检查 (step convergence)
        if max(|dx|) < velocity_tol * dt:
            step_converged = True

        # 4. CCD 过滤
        alpha = ccd_filter(x, dx, alpha_init)

        # 5. 回溯线搜索
        for ls_iter in max_ls_iter:
            x_trial = x + alpha * dx
            E_trial = energy_fn(x_trial)
            if E_trial < E_current:
                accept
            alpha *= shrink_factor

        # 6. stagnation 检测
        if actual_position_change < tol:
            stagnation_count++
            if stagnation >= max_stagnation:
                converged = True

        if converged:
            break

    return NewtonResult(iterations, converged, E_final)
```

### 7.2 线性求解器选择

```
小系统 (n_verts <= 500): DenseDirectSolver (直接求解)
大规模系统: PCG + BSR 预条件
ABD 系统: Reduced PCG (12 DOF per body)
```

---

## 8. 摩擦模型

### 8.1 Lagged Normal Force

使用前一帧的法向力:
```
f_n = -dB/dD * 2*d  (barrier 导数推导的法向力)
```

### 8.2 C1 Smooth Mollifier

平滑摩擦势能:
```
E_friction = μ * f_n * f_smooth(||tan_disp||)

其中 f_smooth 是 C1 连续的 mollifier 函数
```

---

## 9. CCD (连续碰撞检测)

### 9.1 Half-plane CCD

```
对于半平面 (P, N):
    v0 = x[sv]  # 当前位置
    v1 = v0 + alpha * dx  # 试探位置

    # 计算穿透时间
    speed = -dot(N, dv)
    t0 = dot(N, v0 - P)

    if speed > 0 and t0 > 0:
        toi = t0 / speed * (1 - eta)  # eta 是安全因子
        alpha = min(alpha, toi)
```

### 9.2 Mesh CCD

对 PT/EE/PE/PP 候选对进行连续碰撞检测:
```
计算轨迹 AABB (swept volume)
BVH 查询可能碰撞的候选对
计算精确 TOI (time of impact)
alpha = min(alpha, toi * safety_factor)
```

---

## 10. 数据结构

### 10.1 SimState (仿真状态)

```python
class SimState:
    # 位置/速度
    positions: wp.array(vec3d)     # 当前位置
    velocities: wp.array(vec3d)    # 当前速度
    x_prev: wp.array(vec3d)        # 前一帧位置
    x_tilde: wp.array(vec3d)       # 预测位置

    # 质量/约束
    masses: wp.array(float64)
    is_constrained: wp.array(int32)
    aim_positions: wp.array(vec3d)  # SPC 目标位置

    # FEM 数据
    tet_indices: wp.array(vec4i)
    tet_Dm_inv: wp.array(mat33d)
    tet_volumes: wp.array(float64)
    tet_mu, tet_lambda: wp.array(float64)

    # Surface 数据
    surf_verts: wp.array(int32)
    surf_tris: wp.array(vec3i)
    surf_edges: wp.array(vec2i)

    # Barrier 参数
    d_hat: float64
    kappa: float64
    dt: float64

    # ABD 数据
    abd_proxy_idx: wp.array(vec4i)   # 每个body的4个代理顶点
    abd_rest_positions: wp.array(vec3d)
    particle_affine: wp.array(int32)  # 顶点->body映射
```

### 10.2 候选对结构

```python
# PT 候选: (vertex_id, triangle_id)
pt_candidates: wp.array(vec2i)

# EE 候选: (edge_a_id, edge_b_id)
ee_candidates: wp.array(vec2i)

# PE 候选: (vertex_id, edge_id)
pe_candidates: wp.array(vec2i)

# PP 候选: (vertex_a_id, vertex_b_id)
pp_candidates: wp.array(vec2i)
```

---

## 11. 关键函数接口

### 11.1 IPCEngine.advance()

```python
def advance(self, animator=None) -> NewtonResult:
    """
    执行一个时间步

    输入:
        animator: 可选的动画回调 (更新约束目标)

    输出:
        NewtonResult: iterations, converged, final_energy

    内部流程:
        1. 预测位置
        2. 碰撞检测
        3. Newton 求解
        4. 速度更新
    """
```

### 11.2 NewtonSolver.solve()

```python
def solve(
    x_start: wp.array,       # 起点 (x_prev)
    positions: wp.array,     # 输出位置
    energy_fn: Callable,     # 能量函数
    gradient_fn: Callable,   # 梯度函数
    hessian_solve_fn: Callable,  # Hessian 求解函数
    filter_fn: Callable,     # CCD 过滤函数
    convergence_check_fn: Callable,  # 收敛检查
    dt: float,
) -> NewtonResult:
```

### 11.3 距离计算内核

```python
@wp.func
def pt_distance_squared(p, t0, t1, t2) -> float64:
    """点-三角形平方距离"""

@wp.func
def pt_distance_gradient(p, t0, t1, t2) -> vec12d:
    """点-三角形距离梯度 (12 DOF: 4 vertices × 3)"""

@wp.func
def pt_distance_hessian(p, t0, t1, t2) -> mat1212d:
    """点-三角形距离 Hessian"""
```

---

## 12. 自适应参数策略

### 12.1 GIPC Adaptive Kappa

```python
def _compute_adaptive_kappa(self):
    """
    基于 GIPC 策略的自适应 kappa

    公式: new_kappa = clamp(-dot(g_c, g_nc) / dot(g_c, g_c), min_kappa, max_kappa)

    其中:
    - g_c = barrier 梯度 (kappa=1)
    - g_nc = 非接触梯度 (kinetic + elastic)
    """
```

---

## 13. 性能优化技术

### 13.1 BVH Refit
- Newton 迭代中只更新 AABB, 不重建树
- ~5x 加速 (位置增量变化小)

### 13.2 GPU-native 候选生成
- 完全在 GPU 上生成候选对
- 避免 CPU-GPU 同步

### 13.3 CPU Newton (小系统)
- 对于 n_verts <= 500 的系统
- 纯 CPU 求解避免 GPU kernel launch 开销
- ~100x 加速 (小规模)

### 13.4 Cached Numpy Arrays
- 缓存静态数据避免重复 GPU→CPU 传输
- ABD rest positions, proxy indices 等

---

## 14. 与其他 IPC 实现的对比

| 特性 | warp-ipc | libuipc | IPC |
|------|----------|---------|-----|
| 语言 | Python+Warp | C++/CUDA | C++ |
| GPU支持 | ✓ | ✓ | 部分 |
| ABD支持 | ✓ | ✓ | ✗ |
| Codim-IPC | ✓ | ✓ | ✓ |
| 摩擦 | ✓ | ✓ | ✓ |
| BVH加速 | Warp LBVH | 自定义 | ✗ |

---

## 15. 参考

- LibUIPC: `src/backends/cuda/engine/sim_engine_do_advance.cu`
- GIPC Adaptive Kappa: `gipc_adaptive_parameter_strategy.cu`
- IPC Paper: "Incremental Potential Contact" (Li et al., SIGGRAPH 2020)
- Codim-IPC Paper: "Codim-IPC" (Li et al.)