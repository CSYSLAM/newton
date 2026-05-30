# IPC 项目分析 (原版参考实现)

**项目位置**: `C:\csy_work\CG\Engine\IPC\IPC`

---

## 1. 项目概述

这是 SIGGRAPH 2020 论文 **"Incremental Potential Contact: Intersection- and Inversion-free Large Deformation Dynamics"** 的开源参考实现。IPC 是一个基于优化时间积分和 barrier 函数的接触处理方法，能够保证无穿透和无反转的大变形动力学仿真。

### 论文信息
- **标题**: Incremental Potential Contact
- **作者**: Minchen Li, Danny M. Kaufman, Chenfanfu Jiang
- **发表**: SIGGRAPH 2020
- **网址**: https://ipc-sim.github.io/

### 核心特性
- **无穿透**: Barrier 函数保证接触无穿透
- **无反转**: 弹性模型保证四面体无反转
- **大规模**: 支持大规模网格仿真
- **自动化**: 无需手动调整接触参数

---

## 2. 项目架构

```
IPC/
├── src/                        # 源代码
│   ├── main.cpp               # 主程序入口
│   ├── Mesh.cpp/hpp           # 网格数据结构
│   ├── Config.cpp/hpp         # 配置管理
│   ├── AnimScripter.cpp/hpp   # 动画脚本
│   │
│   ├── TimeStepper/           # 时间步进器
│   │   ├── Optimizer.cpp/hpp  # Newton 优化器
│   │   └── Energy.cpp/hpp     # 能量计算框架
│   │
│   ├── Energy/                # 能量函数
│   │   └── Physics_Elasticity/
│   │       ├── NeoHookeanEnergy.cpp/hpp   # Neo-Hookean
│   │       └── FixedCoRotEnergy.cpp/hpp   # 固定共旋
│   │
│   ├── CollisionObject/       # 碰撞对象
│   │
│   ├── LinSysSolver/          # 线性系统求解器
│   │
│   ├── Utils/                 # 工具模块
│   │   ├── BarrierFunctions.hpp   # Barrier 函数
│   │   ├── CCDUtils.cpp/hpp       # 连续碰撞检测
│   │   ├── SpatialHash.hpp        # 空间哈希
│   │   ├── AutoFlipSVD.hpp        # SVD 计算
│   │   └ Types.hpp               # 类型定义
│   │
│   └── Projects/              # 子项目
│
├── cmake/                     # CMake 配置
├── input/                     # 输入数据
├── tests/                     # 单元测试
├── tools/                     # 工具脚本
└── wiki/                      # Wiki 图片
```

---

## 3. IPC Barrier 函数 (原版定义)

### 3.1 三种 Barrier 形式

```cpp
// 文件: Utils/BarrierFunctions.hpp

// C0 clamped log barrier
inline void b_C0(double d, double dHat, double& b) {
    b = -log(d / dHat);  // 简单负对数
}

// C1 clamped log barrier (一阶连续)
inline void b_C1(double d, double dHat, double& b) {
    b = (d - dHat) * log(d / dHat);
}

// C2 clamped log barrier (二阶连续) - 默认使用
inline void b_C2(double d, double dHat, double& b) {
    b = -(d - dHat) * (d - dHat) * log(d / dHat);
}
```

### 3.2 Barrier 梯度

```cpp
// C2 barrier 梯度
inline void g_bC2(double d1, double dHat1, double& g) {
    double t2 = d1 - dHat1;
    g = t2 * log(d1 / dHat1) * -2.0 - (t2 * t2) / d1;
}

// C2 barrier Hessian
inline void H_bC2(double d1, double dHat1, double& H) {
    double t2 = d1 - dHat1;
    H = (log(d1 / dHat1) * -2.0 - t2 * 4.0 / d1) + 1.0 / (d1 * d1) * (t2 * t2);
}
```

### 3.3 Barrier 函数选择

```cpp
// 通过编译宏选择 barrier 类型
#define BARRIER_FUNC_TYPE 2  // 默认使用 C2

inline void compute_b(double d, double dHat, double& b) {
    if constexpr (BARRIER_FUNC_TYPE == 0) {
        b_C0(d, dHat, b);
    } else if constexpr (BARRIER_FUNC_TYPE == 1) {
        b_C1(d, dHat, b);
    } else if constexpr (BARRIER_FUNC_TYPE == 2) {
        b_C2(d, dHat, b);  // 默认
    }
}
```

### 3.4 与其他实现的对比

| Barrier 形式 | IPC 原版 | warp-ipc | ppf-contact-solver |
|-------------|---------|----------|-------------------|
| 默认类型 | C2 (-log) | Log (平方距离) | Cubic |
| C0 连续性 | ✓ | - | - |
| C1 连续性 | ✓ (可选) | - | - |
| C2 连续性 | ✓ (默认) | ✓ | ✓ |

---

## 4. Optimizer (Newton 求解器)

### 4.1 核心类结构

```cpp
template <int dim>
class Optimizer {
protected:
    const Mesh<dim>& data0;                   // 初始网格
    const std::vector<Energy<dim>*>& energyTerms;  // 能量项列表
    const std::vector<double>& energyParams;  // 能量权重

    // 线性求解器
    LinSysSolver* linSysSolver;

    // IPC 参数
    double kappa;       // Barrier 刚度
    double dHat;        // 激活距离
    double dHatEps;     // Barrier 容差

    // 状态变量
    Eigen::VectorXd gradient;    // 能量梯度
    Eigen::VectorXd searchDir;   // 搜索方向
    double lastEnergyVal;        // 上次能量值

    // 动力学信息
    Eigen::VectorXd velocity;    // 速度
    Eigen::MatrixXd xTilta;      // 预测位置
    double dt;                   // 时间步长

public:
    // 主求解函数
    virtual int solve(int maxIter = 100);

    // 预计算
    virtual void precompute(void);
    virtual void updatePrecondMtrAndFactorize(void);
};
```

### 4.2 Newton 求解流程

```
solve(maxIter):
    1. 初始化:
       - 构建预条件矩阵
       - 因子化线性系统

    2. Newton 迭代:
       for iter in range(maxIter):
           a. 计算能量 E(x)
           b. 计算梯度 g = ∂E/∂x
           c. 计算 Hessian H = ∂²E/∂x²
           d. 求解线性系统: H * Δx = -g
           e. 线搜索确定步长 α
              - CCD 过滤步长
              - 回溯线搜索
           f. 更新位置: x = x + α * Δx
           g. 检查收敛:
              - ||g||₂ < tol
              - ||Δx||₂ < tol

    3. 返回迭代次数
```

---

## 5. 能量函数框架

### 5.1 Energy 基类

```cpp
template <int dim>
class Energy {
public:
    // 核心接口
    virtual void computeEnergyVal(const Mesh<dim>& data, ...);
    virtual void computeGradient(const Mesh<dim>& data, ...);
    virtual void computeHessian(const Mesh<dim>& data, ...);

    // 基于 SVD 的能量计算框架
    virtual void compute_E(const Vec<dim>& singularValues, double u, double lambda, double& E);
    virtual void compute_dE_div_dsigma(const Vec<dim>& singularValues, ...);
    virtual void compute_d2E_div_dsigma2(const Vec<dim>& singularValues, ...);
    virtual void compute_BLeftCoef(const Vec<dim>& singularValues, ...);
};
```

### 5.2 Neo-Hookean 弹性模型

```cpp
template <int dim>
class NeoHookeanEnergy : public Energy<dim> {
public:
    // 能量公式 (基于奇异值)
    void compute_E(const Vec<dim>& sigma, double mu, double lambda, double& E) {
        // Neo-Hookean 能量:
        // E = μ/2 * (Σσᵢ² - dim) + λ/2 * (J - 1)²
        // 其中 J = σ₁ * σ₂ * σ₃ (体积变化)
        ...
    }

    // 梯度和 Hessian 基于 PK (Piola-Kirchhoff) 应力
    void computeGradientByPK(...);
    void computeHessianByPK(...);
};
```

### 5.3 固定共旋弹性

```cpp
template <int dim>
class FixedCoRotEnergy : public Energy<dim> {
    // 固定共旋模型 (适用于小变形)
    // E = μ * Σ(σᵢ - 1)² + λ/2 * (J - 1)²
};
```

---

## 6. 网格数据结构

### 6.1 Mesh 类

```cpp
template <int dim>
class Mesh {
public:
    // 顶点数据
    Eigen::MatrixXd V;   // 顶点位置 (nV x dim)

    // 元素数据
    Eigen::MatrixXi F;   // 表面三角形 (nF x 3)
    Eigen::MatrixXi T;   // 四面体 (nT x 4)

    // 材料参数
    std::vector<double> restVol;  // 静止体积
    std::vector<double> mu;      // 泊松比
    std::vector<double> lambda;  // 杨氏模量导出的参数

    // 状态检查
    bool checkInversion(void);   // 检查四面体反转
};
```

---

## 7. 碰撞检测

### 7.1 空间哈希

```cpp
// 空间哈希加速碰撞检测
class SpatialHash {
    // 将空间划分为网格单元
    // 通过哈希查找邻近三角形/顶点
};
```

### 7.2 CCD (连续碰撞检测)

```cpp
// 文件: Utils/CCDUtils.cpp

// 使用 CCD-Wrapper 库
// 包括:
// - Etienne Vouga's CTCD
// - Tight-Inclusion CCD

// 返回 TOI (time of impact)
double compute_ccd(...);
```

---

## 8. 线性系统求解器

### 8.1 支持的求解器

```
1. CHOLMOD (SuiteSparse) - 推荐，最快
2. Eigen LDLT - 备用
3. AMGCL - 大规模问题
```

### 8.2 LinSysSolver 接口

```cpp
template <typename IndexType, typename ValueType>
class LinSysSolver {
public:
    virtual void setPattern(const Matrix& A);
    virtual void analyzePattern(void);
    virtual void factorize(void);
    virtual void solve(const Vector& b, Vector& x);
};
```

---

## 9. 动力学时间积分

### 9.1 增量势能形式

```
时间积分采用隐式 Euler:

E_total(x) = E_inertia(x) + E_elastic(x) + E_barrier(x)

其中:
- E_inertia = 1/2 * ||x - x̃||²_M (惯性势能)
- E_elastic = Σ E_tet(x) (弹性势能)
- E_barrier = Σ κ * b(d(x)) (接触势能)

x̃ = x_prev + v * dt + g * dt² (预测位置)
```

### 9.2 时间步进流程

```cpp
void setTime(double duration, double dt) {
    // 设置时间步长
    this->dt = dt;
    dtSq = dt * dt;

    // 计算重力作用
    gravityDtSq = gravity * dtSq;
}

void step() {
    // 1. 预测位置
    xTilta = V_prev + velocity * dt + gravity * dtSq;

    // 2. Newton 求解最小化 E(x)
    solve();

    // 3. 更新速度
    velocity = (V_new - V_prev) / dt;
}
```

---

## 10. 接触约束类型

### 10.1 MMCVID (Mesh-Mesh Contact Vertex-ID)

```cpp
// 接触对类型枚举
enum MMCVIDType {
    PP,   // Point-Point
    PE,   // Point-Edge
    PT,   // Point-Triangle
    EE    // Edge-Edge
};

struct MMCVID {
    int vI, eI, fI;  // 顶点、边、面索引
    MMCVIDType type; // 接触类型
};
```

### 10.2 活动集管理

```cpp
// 活动接触对集合
std::vector<std::vector<MMCVID>> MMActiveSet;
std::vector<std::vector<MMCVID>> MMActiveSet_next;

// 拉格朗日乘子
std::vector<Eigen::VectorXd> MMLambda_lastH;
```

---

## 11. 运行示例

### 11.1 构建和运行

```bash
# 构建
python build.py
# 或
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4

# 运行示例
./IPC input/tutorials/1-uv_ball.json
```

### 11.2 输入配置 (JSON)

```json
{
    "input": "ball.obj",
    "output": "output/ball",

    // 材料参数
    "material": {
        "E": 1e4,
        "nu": 0.4,
        "rho": 1000
    },

    // IPC 参数
    "ipc": {
        "dHat": 1e-3,
        "kappa": 1e4,
        "friction": 0.5
    },

    // 时间步进
    "dt": 0.01,
    "duration": 10.0
}
```

---

## 12. 依赖关系

### 12.1 必需依赖
- libigl (几何处理)
- OSQP (QP 求解)
- TBB (并行)
- spdlog (日志)
- CCD-Wrapper (CCD)

### 12.2 可选依赖
- SuiteSparse/CHOLMOD (高性能线性求解)
- AMGCL (大规模线性求解)
- Gurobi (QP 替代)

---

## 13. 与其他 IPC 实现对比

| 特性 | IPC 原版 | warp-ipc | libuipc |
|------|---------|----------|---------|
| 语言 | C++ | Python+Warp | C++/CUDA |
| GPU | ✗ | ✓ | ✓ |
| Barrier | C2 Log | Log | Log |
| 线性求解器 | CHOLMOD | PCG | PCG |
| 弹性模型 | NH/FCR | SNH/NH | SNH |
| CCD | Tight-Inclusion | 自实现 | 自实现 |

---

## 14. 参考

- GitHub: https://github.com/ipc-sim/IPC
- Paper: https://ipc-sim.github.io/
- Wiki: https://github.com/ipc-sim/IPC/wiki