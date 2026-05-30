# ppf-contact-solver 项目分析

**项目位置**: `C:\csy_work\CG\Engine\ppf-contact-solver`

---

## 1. 项目概述

ppf-contact-solver 是由日本 ZOZO 公司开发的高性能接触求解器，支持壳体(shell)、实体(solid)和杆(rod)的物理仿真。该项目发表在 SIGGRAPH Asia 2024，论文标题为 **"A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness"**。

### 核心特性
- **Cubic Barrier 函数**: 比传统 IPC 的对数 barrier 更高效
- **单精度 GPU 计算**: 无需双精度，提高 GPU 效率
- **弹性包含的动态刚度**: 自适应刚度参数
- **LBVH 加速碰撞检测**: 线性 BVH 实现宽相检测
- **支持超大规模接触**: 已验证超过 1.8 亿接触对

### 支持的物理模型
- FEM (有限元) 弹性体
- Shell (壳) 薄膜结构
- Rod (杆) 一维结构
- 刚体碰撞

---

## 2. 项目架构

```
ppf-contact-solver/
├── crates/                      # Rust 核心模块
│   ├── ppf-cts-core/            # 核心数据结构
│   ├── ppf-cts-solver/          # 求解器核心
│   │   └── src/
│   │       ├── backend.rs       # 后端接口 (Rust↔CUDA)
│   │       ├── scene.rs         # 场景构建
│   │       ├── builder.rs       # 场景构建器
│   │       ├── data.rs          # 数据结构定义
│   │       └── cpp/             # CUDA C++ 实现
│   │           ├── barrier/     # Barrier 函数实现
│   │           ├── contact/     # 接触检测
│   │           ├── solver/      # 线性求解器
│   │           ├── energy/      # 能量计算
│   │           ├── lbvh/        # 线性 BVH
│   │           ├── kernels/     # CUDA 内核
│   │           └── main/        # 主入口 (advance)
│   ├── ppf-cts-py/              # Python 绑定
│   └── ppf-cts-server/          # 服务器接口
│
├── examples/                    # Jupyter 示例
├── frontend/                    # 前端界面
├── blender_addon/               # Blender 插件
└── docs/                        # 文档
```

---

## 3. Cubic Barrier 函数 (核心创新)

### 3.1 与 IPC Barrier 对比

| 特性 | IPC Barrier | Cubic Barrier (ppf) |
|------|-------------|---------------------|
| 形式 | 对数函数 | 三次多项式 |
| 能量 | -κ(D-d̂²)² log(D/d̂²) | -2/3 * (y³)/d̂ |
| 梯度 | 复杂 (含 log) | 简单多项式 |
| Hessian | 需特殊处理 | 解析形式简单 |
| 计算效率 | 较低 | 较高 |

### 3.2 Cubic Barrier 数学定义

```cpp
// 文件: barrier/cubic.hpp

// 能量函数
__device__ float energy(float g, float ghat, float offset) {
    g -= offset;           // 偏移 (厚度)
    float y = g - ghat;    // 到激活距离的差值
    if (y < 0.0f) {
        // 激活区域内的三次势能
        return -2.0f * (y * y * y) / (3.0f * ghat);
    } else {
        return 0.0f;       // 激活区域外无能量
    }
}

// 梯度 (dB/dg)
__device__ float gradient(float g, float ghat, float offset) {
    g -= offset;
    float y = g - ghat;
    if (y < 0.0f) {
        return -2.0f * y * y / ghat;
    } else {
        return 0.0f;
    }
}

// 曲率 (Hessian 项)
__device__ float curvature(float g, float ghat, float offset) {
    g -= offset;
    if (g - ghat < 0.0f) {
        return 4.0f * (1.0f - g / ghat);
    } else {
        return 0.0f;
    }
}
```

### 3.3 激活条件

```
当 g < ghat + offset 时激活:
- g: 平方距离
- ghat: 激活距离的平方 (d̂²)
- offset: 厚度偏移 (支持 codim-IPC)
```

### 3.4 与传统 IPC 的关键差异

```
IPC Barrier:  B(D) = -κ(D-d̂²)² log(D/d̂²)
Cubic Barrier: B(g) = -2/3 * (g-d̂)³ / d̂

优势:
1. 无对数运算 → 更快的 GPU 计算
2. 梯度/Hessian 形式简单 → 线性求解更稳定
3. 三次增长足够防止穿透
```

---

## 4. 仿真主循环 (advance)

### 4.1 流程伪代码

```
advance():
    1. 构建 LBVH (宽相碰撞检测)
       - build_face_bvh()
       - build_edge_bvh()
       - build_vertex_bvh()

    2. 计算速度和动量
       - velocity = (curr_vertex - prev_vertex) / prev_dt
       - max_u = max_velocity

    3. 主循环 (Newton 迭代):
       while sim_time < frame_end_time:
           a. 预测步长 dt = min(frame_dt, remaining_time)
           b. 计算动能 (惯性项)
           c. 计算弹性势能 (FEM/Shell/Rod)
           d. 碰撞检测 → 构建接触约束
           e. 构建 Hessian 矩阵 (CSR 格式)
           f. PCG 求解线性系统
           g. CCD (连续碰撞检测) 确定 step_size
           h. 回溯线搜索
           i. 更新位置
           j. 检查收敛
           k. sim_time += step_size * dt

    4. 更新前帧位置
    5. 返回 StepResult
```

### 4.2 数据流

```
输入数据:
├── vertex (当前位置, 前一帧位置)
├── mesh (三角形, 边, 四面体)
├── prop (材料属性: mass, damping, etc.)
├── constraint (约束条件: pin, collision mesh)
└── param (仿真参数: dt, gravity, etc.)

输出数据:
├── 新的顶点位置
├── 接触记录
└── StepResult (收敛状态, PCG/CCD 成功标志)
```

---

## 5. 碰撞检测流程

### 5.1 宽相检测 (LBVH)

```
build_bvh():
    1. 计算每个 primitive 的 AABB
       - face_aabb (三角形包围盒)
       - edge_aabb (边包围盒)
       - vertex_aabb (顶点包围盒)

    2. 构建 LBVH (Linear BVH)
       - 计算 Morton code
       - 排序并构建层次结构
       - 自底向上传播 AABB

    3. 查询潜在碰撞对
       - BVH 遍历查找重叠 AABB
```

### 5.2 窄相检测 (距离计算)

```cpp
// 点-三角形距离系数
Vec3f point_triangle_distance_coeff(p, t0, t1, t2):
    // 计算重心坐标
    // 分类: 面内、边上、顶点

// 点-边距离系数
Vec2f point_edge_distance_coeff(p, e0, e1):
    // 参数化 t = dot(p-e0, e1-e0) / ||e1-e0||²
    // clamp(t, 0, 1)

// 边-边距离系数
Vec4f edge_edge_distance_coeff(ea0, ea1, eb0, eb1):
    // 计算两条边的最近点参数
    // 分类: 内部相交、端点退化
```

### 5.3 CCD (连续碰撞检测)

```cpp
// 文件: contact/accd.hpp

float point_triangle_ccd(p0, p1, t00, t01, t02, t10, t11, t12, offset):
    // 计算轨迹
    dp = p1 - p0
    dt0 = t10 - t00
    dt1 = t11 - t01
    dt2 = t12 - t02

    // 中心化 (消除浮点误差)
    centerize(x0)
    centerize(dx)

    // 迭代求 TOI (time of impact)
    for k in ccd_max_iter:
        d2 = distance(x0 + toi * dx)
        if d2 < target²:
            break
        toi = toi + d_minus_target / u_max

    return toi
```

---

## 6. 能量函数

### 6.1 总能量组成

```
E_total = E_kinetic + E_elastic + E_barrier + E_constraints

其中:
- E_kinetic: 动能 (惯性项)
- E_elastic: 弹性势能 (FEM/Shell/Rod)
- E_barrier: Cubic barrier 接触能
- E_constraints: 约束能 (pin, collision mesh)
```

### 6.2 FEM 弹性模型

支持的材料模型:
- Neo-Hookean (NH)
- Stabilized Neo-Hookean (SNH)
- ARAP (As-Rigid-As-Possible)

---

## 7. 线性求解器

### 7.1 PCG 求解器

```cpp
// 文件: solver/solver.hpp

bool solve(DynCSRMat A, FixedCSRMat B, Vec<Mat3x3f> C,
           Vec<float> b, float tol, unsigned max_iter,
           Vec<float> x, unsigned &iter, float &resid):
    // 混合矩阵结构:
    // - A: 动态部分 (接触 Hessian)
    // - B: 固定部分 (弹性 Hessian)
    // - C: 对角块

    // PCG 迭代
    // 预条件: 对角块 Jacobi
```

### 7.2 矩阵结构

```
Hessian = H_fixed + H_dynamic

H_fixed:
- 预计算的弹性 Hessian (CSR 格式)
- 固定拓扑，只需组装一次

H_dynamic:
- 动态的接触 Hessian
- 每步重新构建
- 稀疏结构变化
```

---

## 8. 距离计算详解

### 8.1 点-三角形距离

7 种分类情况:

```
PT_FACE:     点在三角形内部投影
PT_EDGE_01:  点在边 (t0,t1) 上最近
PT_EDGE_12:  点在边 (t1,t2) 上最近
PT_EDGE_20:  点在边 (t2,t0) 上最近
PT_VERTEX_0: 点接近顶点 t0
PT_VERTEX_1: 点接近顶点 t1
PT_VERTEX_2: 点接近顶点 t2
```

### 8.2 边-边距离

```
EE_INTERIOR:  两边内部最近点
EE_E0_START:  边 ea 的起点最近
EE_E0_END:    边 ea 的终点最近
EE_E1_START:  边 eb 的起点最近
EE_E1_END:    边 eb 的终点最近
EE_PP_A0_B0:  点点 (ea0, eb0)
EE_PP_A0_B1:  点点 (ea0, eb1)
EE_PP_A1_B0:  点点 (ea1, eb0)
EE_PP_A1_B1:  点点 (ea1, eb1)
```

---

## 9. 核心数据结构

### 9.1 DataSet (Rust)

```rust
struct DataSet {
    vertex: VertexData,           // 顶点位置/速度
    mesh: MeshData,               // 网格拓扑
    prop: PropertyData,           // 材料属性
    constraint: ConstraintData,   // 约束条件
    param_arrays: ParamArrays,    // 参数数组
}
```

### 9.2 ParamSet (CUDA)

```cpp
struct ParamSet {
    float dt;
    float prev_dt;
    Vec3f gravity;
    float ccd_max_iter;
    float ccd_reduction;
    float line_search_max_t;
    // ... 更多参数
};
```

### 9.3 CSR 矩阵

```cpp
struct DynCSRMat {
    unsigned *row_ptr;    // 行指针
    unsigned *col_idx;    // 列索引
    Mat3x3f *values;      // 3x3 块值
    unsigned nnz;
};

struct FixedCSRMat {
    unsigned *row_ptr;
    unsigned *col_idx;
    Mat3x3f *values;
    unsigned nnz;
    TransposeTable *transpose;  // 用于对称矩阵操作
};
```

---

## 10. 自适应刚度策略

### 10.1 弹性包含的动态刚度

```
关键创新: Barrier 刚度与弹性刚度耦合

传统方法: κ 是固定参数或手动调整
ppf 方法: κ 自动包含弹性刚度的影响

好处:
1. 避免参数调优
2. 自动平衡接触和弹性
3. 更稳定的收敛
```

---

## 11. Strain Limiting (应变限制)

```
三角形应变限制:
- 防止三角形过度拉伸
- 严格的应变上限 (如 1%)
- 保证几何稳定性
```

---

## 12. 性能特点

### 12.1 单精度优化

```
所有计算使用 float32:
- GPU 单精度性能更高
- 减少内存带宽需求
- 数值稳定性通过中心化技术保证
```

### 12.2 大规模支持

```
验证场景:
- 超过 1.8 亿接触对
- 支持 10 连续运行无失败
- GitHub Actions 压力测试
```

---

## 13. 与其他 IPC 实现对比

| 特性 | ppf-contact-solver | warp-ipc | libuipc |
|------|-------------------|----------|---------|
| 语言 | Rust + CUDA | Python + Warp | C++ + CUDA |
| Barrier | Cubic | Logarithm | Logarithm |
| 精度 | float32 | float64 | float64 |
| 求解器 | PCG | PCG/Dense | PCG |
| 自适应刚度 | ✓ (创新) | ✓ (GIPC) | ✓ |
| 摩擦 | ✓ | ✓ | ✓ |
| Blender 插件 | ✓ | ✗ | ✗ |

---

## 14. 参考论文

**A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness**
- 发表于 ACM Transactions on Graphics (TOG) Vol.43, No.6
- SIGGRAPH Asia 2024
- 作者: Ryoichi Ando (ZOZO, Inc.)

### 论文核心贡献

1. **Cubic Barrier 函数**: 替代传统对数 barrier
2. **弹性包含的动态刚度**: 自动刚度调节策略
3. **高效的 GPU 实现**: 单精度、大规模并行
4. **严格的应变限制**: 几何稳定性保证

---

## 15. 使用示例

### Python API 调用

```python
from ppf_contact_solver import Scene, run

# 创建场景
scene = Scene()
scene.add_mesh("cloth.obj", material="shell")
scene.add_mesh("body.obj", material="solid")

# 设置参数
scene.set_param(dt=0.01, gravity=(0, 0, -9.81))

# 运行仿真
result = run(scene, frames=100)
```

---

## 16. 参考

- GitHub: https://github.com/st-tech/ppf-contact-solver
- Paper: https://dl.acm.org/doi/abs/10.1145/3687908
- Docker: ghcr.io/st-tech/ppf-contact-solver-compiled:latest