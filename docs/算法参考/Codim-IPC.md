# Codim-IPC 项目分析

**项目位置**: `C:\csy_work\CG\Engine\IPC\Codim-IPC`

---

## 1. 项目概述

Codim-IPC (C-IPC) 是 SIGGRAPH 2021 论文 **"Codimensional Incremental Potential Contact"** 的开源实现。它扩展了标准 IPC 以支持薄壳 (Shell) 和杆 (Rod) 等低维结构的接触处理。

### 论文信息
- **标题**: Codimensional Incremental Potential Contact
- **作者**: Minchen Li, Danny M. Kaufman, Chenfanfu Jiang
- **发表**: SIGGRAPH 2021
- **网址**: https://ipc-sim.github.io/C-IPC/

### 核心特性
- **薄壳支持**: 2D 薄壳 (三角形网格) 的 IPC 接触
- **杆支持**: 1D 杆 (线段) 的 IPC 接触
- **Codim 表示**: 减少 DOF，提高仿真效率
- **统一框架**: 3D/2D/1D 统一处理

---

## 2. 项目架构

```
Codim-IPC/
├── Library/                   # 核心库
│   ├── FEM/                   # FEM 模块
│   │   ├── IPC.h             # IPC 接触处理
│   │   ├── SHELL.h           # 壳模型
│   │   ├── FRICTION.h        # 摩擦模型
│   │   └── TimeStepper/      # 时间步进器
│   │       ├── IMPLICIT_EULER.h
│   │       └── TIME_STEPPER.h
│   ├── Math/                  # 数学模块
│   │   ├── BARRIER.h         # Barrier 函数
│   │   ├── Distance/         # 距离计算
│   │   │   ├── DISTANCE_TYPE.h
│   │   │   ├── POINT_POINT.h
│   │   │   ├── POINT_EDGE.h
│   │   │   ├── POINT_TRIANGLE.h
│   │   │   └── EDGE_EDGE.h
│   │   └── CCD.h             # 连续碰撞检测
│   ├── Physics/               # 物理模型
│   │   ├── NEOHOOKEAN.h
│   │   ├── FIXED_COROTATED.h
│   │   └── ...
│   └── Utils/                 # 工具函数
│
├── Projects/                  # 示例项目
│   ├── FEMShell/              # 壳仿真示例
│   └── FEM/                   # FEM 示例
│
└── Python/                    # Python 绑定
```

---

## 3. Codim-IPC 核心概念

### 3.1 维度分类

```cpp
// 3D 体积网格 (标准 IPC)
// 四面体: 4 个顶点
// 接触类型: PP, PE, PT, EE

// 2D 薄壳 (Codim-IPC 扩展)
// 三角形: 3 个顶点
// 接触类型: PP, PE, PT (退化)
// 厚度参数 ξ

// 1D 杆 (Codim-IPC 扩展)
// 线段: 2 个顶点
// 接触类型: PP, PE (退化)
// 厚度参数 ξ
```

### 3.2 厚度参数

```cpp
// Codim-IPC 引入厚度参数 ξ
// Barrier 激活范围: ξ² < D < (ξ + d̂)²

// 当 ξ = 0 时，退化为标准 IPC
// 当 ξ > 0 时，支持薄壳/杆的接触

template <class T, int dim, bool shell = false>
void Compute_Constraint_Set(...) {
    T dHat = std::sqrt(dHat2) + thickness;
    dHat2 = dHat * dHat;
    // ...
}
```

---

## 4. 接触约束计算

### 4.1 约束集构建

```cpp
// 文件: Library/FEM/IPC.h

template <class T, int dim, bool shell = false, bool elasticIPC = false>
void Compute_Constraint_Set(
    MESH_NODE<T, dim>& X,
    MESH_NODE_ATTR<T, dim>& nodeAttr,
    const std::vector<int>& boundaryNode,      // 边界顶点
    const std::vector<VECTOR<int, 2>>& boundaryEdge,  // 边界边
    const std::vector<VECTOR<int, 3>>& boundaryTri,   // 边界三角形
    // ...
    T dHat2, T thickness, bool getPTEE,
    std::vector<VECTOR<int, dim + 1>>& constraintSet,  // 约束集
    std::vector<VECTOR<int, 2>>& cs_PTEE,
    std::vector<VECTOR<T, 2>>& stencilInfo)
{
    // 1. 构建空间哈希
    SPATIAL_HASH<T, dim> sh;
    sh.Build(X, boundaryNode, boundaryEdge, boundaryTri, 1.0);

    // 2. 查询候选对
    // 对于每个边界顶点，查询邻近的边/三角形

    // 3. 计算距离并分类
    // PP, PE, PT, EE
}
```

### 4.2 距离类型分类

```cpp
// 2D 情况 (薄壳)
enum DistanceType2D {
    PP,  // Point-Point
    PE   // Point-Edge
};

// 3D 情况 (体积)
enum DistanceType3D {
    PP,  // Point-Point
    PE,  // Point-Edge
    PT,  // Point-Triangle
    EE   // Edge-Edge
};
```

---

## 5. 时间步进器

### 5.1 隐式欧拉

```cpp
// 文件: Library/FEM/TimeStepper/IMPLICIT_EULER.h

template <class T, int dim, bool shell = false, bool elasticIPC = false>
int Advance_One_Step_IE(
    MESH_ELEM<dim>& Elem,
    VECTOR_STORAGE<T, dim + 1>& DBC,  // Dirichlet 边界条件
    const VECTOR<T, dim>& gravity, T h,  // 重力, 时间步长
    T NewtonTol, bool withCollision,
    T dHat2, VECTOR<T, 3>& kappaVec,
    T mu, T epsv2,
    bool staticSolve, bool withShapeMatching,
    // ...
)
{
    // 1. 构建能量函数
    ENERGY<T, dim> energy;
    energy.Add(std::make_shared<ELASTICITY_ENERGY<T,dim>>());  // 弹性
    energy.Add(std::make_shared<INERTIA_ENERGY<T,dim>>());     // 惯性
    energy.Add(std::make_shared<IPC_ENERGY<T,dim,elasticIPC>>());  // 接触

    // 2. 预测位置
    // X_tilde = X + h * v + h² * g

    // 3. Newton 迭代
    // while (!converged):
    //     compute_gradient_and_hessian()
    //     solve_linear_system()
    //     line_search()
    //     update_position()

    // 4. 更新速度
    // v = (X_new - X_old) / h
}
```

---

## 6. 能量函数

### 6.1 弹性能量

```cpp
// Neo-Hookean
NEOHOOKEAN.h

// Fixed Corotated
FIXED_COROTATED.h

// Linear Corotated
LINEAR_COROTATED.h

// STVK Hencky
STVK_HENCKY.h

// Symmetric Dirichlet
SYMMETRIC_DIRICHLET.h
```

### 6.2 接触能量

```cpp
// IPC barrier (Codim 版本)
// 支持厚度参数

template <class T, int dim>
class IPC_ENERGY {
    // Barrier 函数
    // b(D) = -(D - d̂²)² log(D / d̂²)  (标准 IPC)
    // b(D) = -(D - ξ² - d̂² - 2d̂ξ)² log((D - ξ²)/(d̂² + 2d̂ξ))  (Codim IPC)
};
```

---

## 7. 摩擦模型

```cpp
// 文件: Library/FEM/FRICTION.h

template <class T, int dim>
class FRICTION {
    // 摩擦能量
    // E_friction = μ * f_n * f_smooth(||tan_disp||)

    // Lagged normal force
    // f_n = -dB/dD * 2*d
};
```

---

## 8. 与标准 IPC 的对比

| 特性 | IPC (2020) | Codim-IPC (2021) |
|------|-----------|------------------|
| 维度 | 3D | 3D/2D/1D |
| 网格 | 四面体 | 四面体/三角形/线段 |
| 接触类型 | PP, PE, PT, EE | PP, PE, PT, EE (退化) |
| 厚度 | ✗ | ✓ (ξ) |
| 壳/杆 | ✗ | ✓ |
| Barrier | 标准 | Codim 版本 |

---

## 9. 参考

- GitHub: https://github.com/ipc-sim/Codim-IPC
- Paper: https://ipc-sim.github.io/C-IPC/
- 论文: "Codimensional Incremental Potential Contact" (SIGGRAPH 2021)