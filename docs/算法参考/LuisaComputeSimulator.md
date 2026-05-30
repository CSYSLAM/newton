# LuisaComputeSimulator 项目分析

**项目位置**: `C:\csy_work\CG\Engine\LuisaComputeSimulator`

---

## 1. 项目概述

LuisaComputeSimulator 是一个基于 [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) 构建的高性能跨平台物理仿真器，支持软体、布料和刚体的实时仿真，使用 IPC (Incremental Potential Contact) 实现无穿透接触处理。

### 核心特性
- **多后端支持**: CUDA, DirectX 12, Vulkan, Metal, CPU
- **软体/布料/刚体仿真**: 支持 FEM 弹性模型
- **无穿透接触**: IPC barrier 函数
- **Affine Body Dynamics (ABD)**: 刚体的高效降维仿真
- **实时性能**: 88K 顶点, 174K 三角形, 3M+ 碰撞对可达 ~3 FPS (RTX 3090)
- **Python 和 C++ API**: 灵活的编程接口

### 支持的物理模型
- ✅ Soft Body / Cloth (Spring + ARAP FEM)
- ✅ Rigid Body
- ✅ Soft-Rigid Coupling
- ✅ Cloth-Soft-Rigid Coupling
- ✅ Ground Collision
- ✅ Frictional Contact
- ✅ CCD (Continuous Collision Detection)
- ✅ Fixed Point / Pinned Constraints

---

## 2. 项目架构

```
LuisaComputeSimulator/
├── Solver/                      # 核心求解器
│   ├── SimulationSolver/        # Newton 求解器
│   │   ├── newton_solver.h/cpp  # 主求解器实现
│   │   └── solver_interface.h/cpp # 求解器接口
│   │
│   ├── CollisionDetector/       # 碰撞检测
│   │   ├── lbvh.h/cpp           # 线性 BVH
│   │   ├── narrow_phase.h/cpp   # 窄相检测
│   │   ├── distance.hpp         # 距离计算
│   │   ├── cipc_kernel.hpp      # IPC/Codim-IPC 内核
│   │   ├── accd.hpp             # 连续碰撞检测
│   │   └── friction_kernel.hpp  # 摩擦内核
│   │
│   ├── Energies/                # 能量函数
│   │   ├── stretch/             # 拉伸能量
│   │   ├── bending/             # 弯曲能量
│   │   └── inertia/             # 惯性能量
│   │
│   ├── LinearSolver/            # 线性求解器
│   │   └── precond_cg.h         # 预条件 CG
│   │
│   ├── Core/                    # 核心数据类型
│   │   ├── float_n.h            # 向量类型
│   │   └── float_nxn.h          # 矩阵类型
│   │
│   ├── Initializer/             # 场景初始化
│   └── MeshOperation/           # 网格操作
│
├── Application/                 # 应用程序入口
├── PythonBindings/              # Python 绑定
├── UnitTest/                    # 单元测试
└── Resources/                   # 资源文件
```

---

## 3. IPC Barrier 函数

### 3.1 标准 IPC Barrier

```cpp
// 文件: CollisionDetector/cipc_kernel.hpp

template <typename T>
T barrier(const T d, const T dhat) {
    const T d_minus_dhat = (d - dhat);
    // b(d) = -(d-d̂)² ln(d / d̂)
    return -d_minus_dhat * d_minus_dhat * log_scalar(d / dhat);
}

// 一阶导数
template <typename T>
T barrier_first_derivative(const T d, const T dhat) {
    // b'(d) = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
    return (dhat - d) * (2.0f * log_scalar(d / dhat) - dhat / d + 1.0f);
}

// 二阶导数
template <typename T>
T barrier_second_derivative(const T d, const T dhat) {
    const T dhat_d = dhat / d;
    return (dhat_d + 2.0f) * dhat_d - 2.0f * log_scalar(d / dhat) - 3.0f;
}
```

### 3.2 Codim-IPC Barrier (支持厚度)

```cpp
// 支持薄壳/布料的厚度参数
template <typename T>
void KappaBarrier(T& R, const T& kappa, const T& D, const T& dHat, const T& xi) {
    auto x0 = xi * xi;                           // ξ²
    auto x1 = dHat * dHat + 2.0f * dHat * xi;    // d̂² + 2d̂ξ
    // B(D) = -κ(D - ξ² - d̂² - 2d̂ξ)² log((D - ξ²)/(d̂² + 2d̂ξ))
    R = -kappa * (D - x0 - x1) * (D - x0 - x1) * log_scalar((D - x0) / x1);
}

// 梯度
template <typename T>
void dKappaBarrierdD(T& R, const T& kappa, const T& D, const T& dHat, const T& xi) {
    // 自动微分生成的代码
    ...
}
```

---

## 4. Newton 求解器

### 4.1 主求解器类

```cpp
class NewtonSolver : public SolverInterface {
public:
    void physics_step_GPU();  // GPU 仿真步进
    void physics_step_CPU();  // CPU 仿真步进
    void init_solver();       // 初始化 (编译着色器)

private:
    // 编译模块
    void compile(AsyncCompiler& compiler);
    void compile_advancing(AsyncCompiler& compiler, ...);
    void compile_assembly(AsyncCompiler& compiler, ...);
    void compile_evaluate(AsyncCompiler& compiler, ...);

    // 碰撞检测
    void device_construct_lbvh(luisa::compute::Stream& stream);
    void device_broadphase_ccd(luisa::compute::Stream& stream);
    void device_narrowphase_ccd(luisa::compute::Stream& stream);
    void device_update_contact_list(...);

    // 线性求解
    void device_SpMV(luisa::compute::Stream& stream, ...);
    void line_search(luisa::compute::Device& device, ...);
};
```

### 4.2 仿真流程

```
physics_step_GPU():
    1. 预测位置 (predict_position)
       x_tilde = x + v * dt + g * dt²

    2. 碰撞检测 (collision_detection)
       a. 构建 LBVH
       b. 宽相 CCD/DCD
       c. 窄相检测
       d. 更新接触列表

    3. Newton 迭代:
       for iter in max_iter:
           a. 能量组装 (material_energy_assembly)
              - 惯性能量
              - 弹性能量 (stretch, bending)
              - 接触能量 (barrier)
           b. Hessian 矩阵组装
           c. PCG 求解
           d. CCD 线搜索
           e. 更新位置
           f. 收敛检查

    4. 更新速度 (update_velocity)
       v = (x_new - x_old) / dt
```

---

## 5. 碰撞检测

### 5.1 LBVH (Linear BVH)

```
build_lbvh():
    1. 计算每个 primitive 的 AABB
    2. 计算 Morton code
    3. 排序并构建层次结构
    4. 自底向上传播 AABB
```

### 5.2 距离计算

```cpp
// 点-边距离系数
Vec2f point_edge_distance_coeff(p, e0, e1) {
    Vec3f r = e1 - e0;
    float d = squared_norm(r);
    if (d > Epsilon) {
        float t = dot(r, p - e0) / d;
        return Vec2f(1.0f - t, t);
    }
    return Vec2f(0.5f, 0.5f);
}

// 点-三角形距离系数
Vec3f point_triangle_distance_coeff(p, t0, t1, t2) {
    Vec3f r0 = t1 - t0;
    Vec3f r1 = t2 - t0;
    // 解线性系统求重心坐标
    // c = (A^T A)^{-1} A^T (p - t0)
    ...
    return Vec3f(1 - c0 - c1, c0, c1);
}

// 边-边距离系数
Vec4f edge_edge_distance_coeff(ea0, ea1, eb0, eb1) {
    // 计算两条边的最近点参数
    ...
    return Vec4f(1 - x0, x0, 1 - x1, x1);
}
```

### 5.3 CCD (连续碰撞检测)

```cpp
// 文件: CollisionDetector/accd.hpp

// 中心化 (消除浮点误差)
template <class T, unsigned R, unsigned C>
void centerize(SMat<T, R, C>& x) {
    SVec<T, R> mov = SVec<T, R>::Zero();
    for (int k = 0; k < C; k++)
        mov += (1.0f / C) * x.col(k);
    for (int k = 0; k < C; k++)
        x.col(k) -= mov;
}

// CCD 迭代求 TOI
template <typename F, typename T, unsigned R, unsigned C>
float ccd_helper(x0, dx, u_max, square_dist_func, offset, param) {
    float toi = 0.0f;
    float eps = param.ccd_reduction * (sqrt(dist(x0)) - offset);
    float target = eps + offset;

    for (k = 0; k < ccd_max_iter; k++) {
        float d2 = distance(x0 + toi * dx);
        if (d2 < target²) break;

        float d_minus_target = (d2 - target²) / (sqrt(d2) + target);
        toi = toi + d_minus_target / u_max;
    }
    return toi;
}
```

---

## 6. 能量函数

### 6.1 拉伸能量

支持两种模型:
- **Spring**: 简单线性弹簧
- **FEM_BW98**: 有限元方法 (大变形稳定)

```cpp
// Spring 模型
E_stretch = k * ||x - x_rest||²

// FEM_BW98 模型 (Baraff-Witkin 1998)
E_stretch = μ * A * (||w_u|| - 1)² + μ * A * (||w_v|| - 1)²
```

### 6.2 弯曲能量

- **QuadraticBending**: 二次弯曲能量
- **DihedralAngle**: 二面角弯曲模型

### 6.3 惯性能量

```cpp
// 软体惯性能量
E_inertia = 0.5 * m * ||x - x_tilde||²

// ABD 惯性能量
E_abd_inertia = 0.5 * q^T M q
```

---

## 7. Affine Body Dynamics (ABD)

### 7.1 ABD 表示

```
刚体状态用 12 个 DOF 表示:
- 3 个旋转轴 (a1, a2, a3)
- 3 个平移 (p)

顶点位置: x_i = [dot(a1, r), dot(a2, r), dot(a3, r)] + p
其中 r 是局部坐标

约束:
- 正交性: a_i · a_j = δ_ij
```

### 7.2 ABD 正交性能量

```cpp
E_ortho = κ * V * Σ (a_i · a_j - δ_ij)²
```

---

## 8. 线性求解器

### 8.1 预条件共轭梯度 (PCG)

```cpp
// SpMV (稀疏矩阵-向量乘法)
// 对角块 + 非对角块

void device_SpMV(input_array, output_array):
    // 对角部分
    y[i] = diag_A[i] * x[i]

    // 非对角部分 (triplet 格式)
    for each triplet (i, j, A_ij):
        y[i] += A_ij * x[j]
        y[j] += A_ij^T * x[i]
```

---

## 9. 材料模型

### 9.1 布料材料

```cpp
struct ClothMaterial {
    ConstitutiveStretchModelCloth stretch_model;  // Spring 或 FEM_BW98
    ConstitutiveBendingModelCloth bending_model;  // QuadraticBending 或 DihedralAngle
    float thickness;        // 厚度
    float youngs_modulus;   // 杨氏模量
    float poisson_ratio;    // 泊松比
};
```

### 9.2 软体材料

```cpp
struct SoftMaterial {
    float youngs_modulus;
    float poisson_ratio;
    float density;
};
```

---

## 10. 数据结构

### 10.1 核心数据类型

```cpp
// 向量类型
using Vec2f = Var<float2>;
using Vec3f = Var<float3>;
using Vec4f = Var<float4>;

// 矩阵类型
using Mat2x2f = Var<float2x2>;
using Mat3x2f = Var<float2x3>;
using Mat2x3f = Var<float3x2>;
using Mat3x3f = Var<float3x3>;
```

### 10.2 碰撞对结构

```cpp
// PT 碰撞对: (vertex_id, triangle_id)
// EE 碰撞对: (edge_a_id, edge_b_id)
// PE 碰撞对: (vertex_id, edge_id)
// PP 碰撞对: (vertex_a_id, vertex_b_id)
```

---

## 11. LuisaCompute 框架特点

### 11.1 多后端支持

```
LuisaCompute 提供统一的前端 DSL:
- 自动编译到 CUDA, DirectX 12, Vulkan, Metal
- 运行时选择后端
- 单一代码库，多平台运行
```

### 11.2 着色器编译

```cpp
void init_solver() {
    AsyncCompiler compiler(device);
    compiler.default_option().enable_debug_info = false;
    compiler.default_option().enable_fast_math = true;

    compile(compiler);  // 编译所有着色器
    compiler.wait();    // 等待编译完成
}
```

---

## 12. Python API 使用示例

```python
import lcs_py as lcs

# 初始化求解器
solver = lcs.NewtonSolver()
solver.init_device(backend_name="cuda")  # cuda, metal, dx, vk

# 创建刚体
cube = solver.create_world_data_from_array("cube", vertices, faces)
cube.set_simulation_type(lcs.MaterialType.Rigid)
cube.set_translation(0.0, 0.34, 0.0)
cube_id = solver.register_world_data(cube)

# 创建布料
cloth = solver.create_world_data_from_file_path("cloth", "square.obj")
cloth.set_simulation_type(lcs.MaterialType.Cloth)
cloth.set_physics_material_cloth(
    thickness=0.001,
    youngs_modulus=1e6,
    poisson_ratio=0.3,
    stretch_model="FEM_BW98",
    bending_model="QuadraticBending"
)
cloth_id = solver.register_world_data(cloth)

# 配置仿真
config = solver.get_config()
config.implicit_dt = 1/60

# 初始化并运行
solver.init_solver()
for frame in range(100):
    solver.physics_step_gpu()
    solver.save_sim_result(f"output/frame_{frame}.obj")
```

---

## 13. 与其他 IPC 实现对比

| 特性 | LuisaComputeSimulator | warp-ipc | ppf-contact-solver |
|------|----------------------|----------|-------------------|
| 语言 | C++ + Luisa DSL | Python + Warp | Rust + CUDA |
| Barrier | Log (IPC) | Log | Cubic |
| 后端 | CUDA/DX/VK/Metal/CPU | CUDA | CUDA |
| ABD 支持 | ✓ | ✓ | ✗ |
| 多后端 | ✓ | ✗ | ✗ |
| Python API | ✓ | ✓ | ✓ |
| 实时性能 | ~3 FPS (88K vert) | - | 离线 |

---

## 14. 参考

- GitHub: https://github.com/ChengzhuUwU/LuisaComputeSimulator
- LuisaCompute: https://github.com/LuisaGroup/LuisaCompute
- 距离计算参考: ppf-contact-solver