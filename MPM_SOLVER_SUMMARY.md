# Newton 隐式 MPM 求解器实现方案

## 1. 概述

Newton 的 `SolverImplicitMPM` 是一个**隐式 Material Point Method (MPM)** 求解器，专为颗粒材料和弹塑性材料设计。它基于 NVIDIA Warp 框架和 `warp.fem` 有限元库实现，充分利用 GPU 并行计算能力。

该求解器特别适合处理**非常刚硬的材料**和**完全非弹性极限**，相比传统的显式 MPM，它提供**无条件的时间步长稳定性**。

### 核心特点
- **隐式求解**：无条件稳定，支持大时间步长
- **GPU 友好**：基于 Warp 框架，支持 CUDA Graph 捕获
- **多物理耦合**：支持弹塑性、摩擦接触、粘滞性、膨胀性
- **流变学求解器**：支持 Drucker-Prager 压力相关屈服、各向同性硬化/软化

---

## 2. 参考文章

### 主要参考文献

**[1] Multi-species simulation of porous sand and water mixtures**
- **作者**: Klár et al.
- **会议**: SIGGRAPH 2016
- **DOI**: [https://doi.org/10.1145/2897824.2925877](https://doi.org/10.1145/2897824.2925877)
- **说明**: 该求解器的基本算法框架参考了这篇论文，但在此基础上扩展了 GPU 友好的流变学求解器

> 注：代码中仅明确引用了这一篇文献。其他实现细节（如流变学求解、接触处理等）为项目内部开发。

---

## 3. 核心数学公式

### 3.1 粒子到网格 (P2G) 传输

#### 质量积分
```
m_i = Σ_p ρ_p * φ_i(x_p) / V_cell
```
其中 `ρ_p` 是粒子密度，`φ_i` 是基函数，`V_cell` 是单元体积。

#### 速度积分（含重力）
```
v_i = Σ_p ρ_p * (v_p + dt * g) * φ_i(x_p) / V_cell
```

#### APIC 速度预测
```
v_apic = C_p * (x_node - x_p)
```
其中 `C_p` 是粒子的速度梯度（affine momentum）。

### 3.2 应变度量

#### Hencky (对数) 应变
当 `USE_HENCKY_STRAIN_MEASURE = True` 时：
```
F_prev = U * diag(ξ) * V^T  (SVD 分解)
RlogSRt_prev = U * diag(log(ξ_1), log(ξ_2), log(ξ_3)) * U^T
```

#### 共旋 Hooke 定律（Hencky 关闭时）
```
F^{-T} = U * diag(1/ξ) * V^T
R = V * U^T
```

### 3.3 应力-应变关系（本构模型）

#### 柔度形式
```
γ = compliance / (1 + damping/dt)
```

#### Hooke 定律
```
ε = (σ * (1 + ν) - ν * trace(σ) * I) * compliance
```
其中 `ν` 是泊松比。

### 3.4 硬化定律

```
h = sinh(-hardening * log(clamp(Jp, 0.1, 1.0)))
```

硬化只影响**屈服参数**，不影响**弹性刚度**。

### 3.5 塑性变形梯度更新

```
δ_Jp = exp(p_rate * rate_factor)
Jp_new = Jp_prev * clamp(δ_Jp, 0.01, 10.0)
```

### 3.6 弹性应变更新

```
skew = 0.5 * dt * (vel_grad - vel_grad^T)
F_new = F_prev + (elastic_strain_delta + skew) @ F_prev
```

### 3.7 屈服面

#### YieldParamVec 布局（6维向量）
```
[0] p_max * sqrt(3/2)    -- 压缩屈服压力（缩放后）
[1] p_min * sqrt(3/2)    -- 拉伸屈服压力（缩放后）
[2] s_max                -- 偏应力屈服应力
[3] μ * p_max            -- 摩擦剪切极限
[4] dilatancy            -- 膨胀因子
[5] viscosity            -- 粘度
```

#### 剪切屈服应力（分段线性 Drucker-Prager）
```
μ = friction * p_max / p_max = friction
s = max(yield_stress, 0)

if r_N < p_min + 0.5*p_max:
    τ_yield = s + μ * (r_N - p_min)
elif r_N > 0.5*p_max:
    τ_yield = s + μ * (p_max - r_N)
else:
    τ_yield = s + μ * 0.5*p_max
```

### 3.7 Delassus 算子

对于每个应变节点，组装 6×6 对角块：
```
W = Σ_i (B_i^T * B_i) / (inv_mass_i * multiplicity)
```

然后进行特征分解：
- 零化剪切-散度耦合项
- 对偏应力子块进行特征分解
- 存储特征值和特征向量

### 3.8 滑动子问题（Newton 迭代）

求解切向速度 `u_T`，满足：
```
|(D + α*I)^{-1} b_T| * (1 - γ*α) = yield_stress
```

其中 `γ = dilatancy * (dyield/dN)^2 / D[0]` 耦合法向和切向。

法向分量：
```
u_N = θ * dyield/dN * |u_T|
```

### 3.9 单侧不可压缩性（Void Fraction）

```
offset = max(max_fraction * (node_vol - collider_vol) - particle_vol, 0)
```

当单元未充分压实时，此偏移量会使屈服面坍塌（消除粘聚力）。

---

## 4. 架构设计

### 4.1 文件结构

```
newton/_src/solvers/implicit_mpm/
├── __init__.py                          # 导出 SolverImplicitMPM
├── solver_implicit_mpm.py               # 主求解器类 (~107KB)
├── implicit_mpm_model.py                # 模型定义 (~25KB)
├── implicit_mpm_solver_kernels.py       # 网格传输、平流、应变更新 (~31KB)
├── solve_rheology.py                    # 流变学求解器驱动 (~69KB)
├── rheology_solver_kernels.py           # 屈服面、流动规则、Delassus (~51KB)
├── contact_solver_kernels.py            # 碰撞接触摩擦求解 (~8KB)
├── rasterized_collisions.py             # SDF 光栅化、碰撞投影 (~27KB)
└── render_grains.py                     # 高分辨率颗粒渲染 (~6KB)
```

### 4.2 类层次结构

```
SolverImplicitMPM(SolverBase)
├── ImplicitMPMModel          # 包装 newton.Model，管理粒子材料和碰撞器
├── ImplicitMPMScratchpad     # 每步的空间、场、BSR 稀疏矩阵
└── LastStepData              # 跨步的暖启动数据（应力、冲量、体变换）
```

### 4.3 函数空间（warp.fem）

三个独立的函数空间覆盖在网格上：

| 空间 | 基函数 | 用途 |
|------|--------|------|
| 速度空间 | Q1, B2, B3, pic, picN | 速度场 |
| 应变空间 | P0, P1d, Q1, Q1d, pic, picN | 应变/应力场 |
| 碰撞空间 | Q1, S2, pic, picN | 碰撞检测 |

### 4.4 网格类型

- **sparse**: Nanogrid（稀疏网格）
- **dense**: Grid3D（显式活动分区）
- **fixed**: 固定网格

---

## 5. 求解流程

### 5.1 单步求解流程 (`_step_impl`)

```
1. _particles_to_cells
   └── 分配/重建网格，将粒子分箱到单元（PIC 或 GIMP 位置）

2. _rasterize_colliders
   └── 将碰撞器 SDF、法线、速度、摩擦、粘附光栅化到网格

3. _compute_unconstrained_velocity
   └── P2G 质量 + 速度（PIC 或 APIC），应用重力，计算逆质量和自由速度

4. _build_collider_rigidity_operator
   └── 为动态（合规）碰撞器构建刚性算子

5. _build_elasticity_system
   └── 插值弹性参数，组装柔度矩阵和应变 RHS

6. _build_plasticity_system
   └── 插值屈服参数，计算单侧应变偏移，组装应变矩阵 B

7. _load_warmstart
   └── 加载上一步的应力/冲量作为初始猜测

8. _solve_rheology
   └── 调用 solve_rheology() 进行耦合隐式求解

9. _save_for_next_warmstart

10. _update_particles
    └── 平流粒子，更新应变/应力

11. _save_data
```

### 5.2 流变学求解流程 (`solve_rheology`)

```
1. 构建 Delassus 算子
   └── 对每个应变节点：
       - 组装 6×6 对角块
       - 零化剪切-散度耦合
       - 特征分解偏应力子块

2. 预处理
   └── 旋转到解耦的特征基
   └── 投影初始应力到屈服面

3. 迭代求解
   └── 选择求解器链（cg/cr/gmres → jacobi → gs/gs-soa/gs-batched）
   └── Gauss-Seidel：使用图着色实现无竞争并行
   └── Jacobi：质量分裂，适合宽模板

4. 后处理
   └── 旋转回世界空间
   └── 计算最终弹性应变和塑性应变增量

5. 接触求解
   └── 节点摩擦（nodal friction）
   └── 子网格摩擦（subgrid friction）
```

### 5.3 支持的求解器链

| 求解器 | 类型 | 特点 |
|--------|------|------|
| `cg` | 线性 | 共轭梯度，无接触 |
| `cr` | 线性 | 共轭残差，无接触 |
| `gmres` | 线性 | 广义最小残差，无接触 |
| `jacobi` | 非线性 | Jacobi 平滑器 |
| `gs` | 非线性 | Gauss-Seidel，图着色 |
| `gs-soa` | 非线性 | SoA 布局，改善内存合并 |
| `gs-batched` | 非线性 | 批量 Jacobi 质量分裂，适合 B2/B3 |

求解器链示例：`("cr", "gs")` 或 `("cg", "jacobi", "gs")`

---

## 6. 关键实现细节

### 6.1 粒子状态变量

| 变量 | 类型 | 说明 |
|------|------|------|
| `particle_elastic_strain` | `wp.mat33` | 弹性变形梯度 F |
| `particle_Jp` | `float` | 塑性变形梯度行列式 |
| `particle_stress` | `wp.mat33` | Cauchy 应力张量 [Pa] |
| `particle_qd_grad` | `wp.mat33` | APIC 速度梯度 |
| `particle_transform` | `wp.mat33` | 渲染用的整体变形梯度 |

### 6.2 材料参数

| 参数 | 单位 | 说明 |
|------|------|------|
| `young_modulus` | Pa | 杨氏模量 |
| `poisson_ratio` | - | 泊松比 |
| `damping` | s | 弹性阻尼松弛时间 |
| `friction` | - | 摩擦系数 |
| `yield_pressure` | Pa | 屈服压力 |
| `tensile_yield_ratio` | - | 拉伸屈服比 |
| `yield_stress` | Pa | 偏应力屈服应力 |
| `hardening` | - | 硬化因子 |
| `hardening_rate` | - | 硬化率 |
| `softening_rate` | - | 软化率 |
| `dilatancy` | - | 膨胀因子 |
| `viscosity` | Pa·s | 粘度 |

### 6.3 配置参数

```python
class Config:
    max_iterations: int = 250          # 流变学求解器最大迭代次数
    tolerance: float = 1.0e-4            # 流变学求解器容差
    solver: str = "auto"                  # 求解器类型
    voxel_size: float = 0.1              # 体素大小
    grid_type: str = "sparse"            # 网格类型
    transfer_scheme: str = "apic"        # 传输方案 (apic/pic)
    integration_scheme: str = "pic"    # 积分方案 (pic/gimp)
    velocity_basis: str = "Q1"           # 速度基函数
    strain_basis: str = "P0"             # 应变基函数
    collider_basis: str = "S2"           # 碰撞器基函数
```

---

## 7. 技术要点

### 7.1 隐式稳定性

该求解器使用**隐式时间积分**，这意味着：
- 无条件稳定（相对于时间步长）
- 可以处理非常刚硬的材料
- 适合完全非弹性极限

### 7.2 GPU 优化

- **CUDA Graph 捕获**：条件性使用 CUDA Graph 减少 CPU 开销
- **图着色并行**：Gauss-Seidel 使用图着色实现无竞争并行
- **SoA 布局**：`gs-soa` 使用 entry-major SoA 布局改善内存合并
- **批量处理**：`gs-batched` 合并颜色为批次，适合宽模板

### 7.3 接触处理

- **节点接触**：直接在速度节点上求解 Coulomb 摩擦
- **子网格接触**：通过碰撞器矩阵映射到速度节点
- **动态碰撞器**：支持合规（compliant）碰撞器，通过刚性算子耦合

### 7.4 暖启动（Warmstart）

- 保存上一步的应力和冲量
- 作为下一步的初始猜测
- 显著加速收敛

---

## 8. 示例场景

求解器支持多种材料和应用场景：

- **颗粒材料**：沙子、雪等
- **粘弹性材料**：流体、粘性材料
- **多材料**：不同材料参数的混合
- **切割/撕裂**：支持材料断裂
- **与刚体耦合**：与 Newton 刚体求解器耦合

---

## 9. 总结

Newton 的隐式 MPM 求解器是一个功能丰富、高度优化的 GPU 求解器，主要特点包括：

1. **隐式稳定性**：无条件稳定，支持大时间步长
2. **丰富的材料模型**：支持弹塑性、摩擦、粘滞、膨胀、硬化/软化
3. **GPU 优化**：CUDA Graph、图着色、SoA 布局、批量处理
4. **接触处理**：节点和子网格摩擦接触
5. **与刚体耦合**：支持动态碰撞器和刚性算子

该求解器参考了 Klár et al. (SIGGRAPH 2016) 的框架，但在此基础上进行了大量扩展和优化，特别是 GPU 友好的流变学求解器。
