# libuipc 项目分析

**项目位置**: `C:\csy_work\CG\Engine\IPC\libuipc`

---

## 1. 项目概述

libuipc 是一个跨平台的现代 C++20 IPC (Incremental Potential Contact) 库，提供 GPU 加速的增量势接触框架，支持刚体、软体、布料和绳索的动力学仿真。

### 核心特性
- **GPU 并行**: 全 GPU 流程，CUDA 后端
- **多物理耦合**: 刚体、软体、布料、绳索统一接触
- **无穿透接触**: IPC barrier 函数 + 摩擦模型
- **可微分仿真**: 支持反向传播 (Diff-Sim)
- **Python API**: `pip install pyuipc`
- **跨平台**: Linux/Windows

### 论文引用
- **StiffGIPC (SIGGRAPH 2025)**: "Advancing GPU IPC for Stiff Affine-Deformable Simulation"
- **AL-IPC (2025)**: "Augmented Lagrangian IPC"

---

## 2. 项目架构

```
libuipc/
├── src/                       # C++ 源码
│   ├── backends/              # 后端实现
│   │   └── cuda/              # CUDA 后端
│   │       ├── engine/        # 仿真引擎
│   │       │   ├── advance_ipc.cu      # IPC 主循环
│   │       │   ├── advance_al.cu       # AL (Augmented Lagrangian) 主循环
│   │       │   └── sim_engine_do_advance.cu  # 调度器
│   │       ├── contact_system/         # 接触系统
│   │       ├── collision_detection/    # 碰撞检测
│   │       ├── linear_system/          # 线性系统求解
│   │       ├── line_search/            # 线搜索
│   │       ├── time_integrator/        # 时间积分
│   │       └── ...
│   ├── constitution/          # 材料模型
│   ├── core/                  # 核心框架
│   ├── geometry/              # 几何模块
│   └── io/                    # 输入输出
│
├── include/uipc/              # 公共头文件
│   ├── constitution/          # 材料模型 API
│   ├── core/                  # 核心 API
│   ├── geometry/              # 几何 API
│   └── backend/               # 后端接口
│
├── python/                    # Python 绑定 (pybind11)
├── apps/                      # 示例应用
└── docs/                      # 文档
```

---

## 3. IPC 主循环 (advance_ipc.cu)

### 3.1 核心流程伪代码

```cpp
void SimEngine::advance() {
    // 1. 重建场景
    rebuild_scene();

    // 2. 记录摩擦候选 (lagged friction)
    record_friction_candidates();

    // 3. 预测运动: x_tilde = x + v * dt
    predict_dof();

    // 4. 自适应参数计算 (adaptive kappa)
    detect_dcd_candidates();
    compute_adaptive_kappa();

    // 5. Newton 迭代
    for (newton_iter = 0; newton_iter < max_iter; ++newton_iter) {
        // 5.1 构建碰撞对
        detect_dcd_candidates();

        // 5.2 计算动态拓扑效应 (接触 + 约束)
        compute_dytopo_effect();

        // 5.3 求解全局线性系统: H * dx = -g
        solve_global_linear_system();

        // 5.4 收集顶点位移
        collect_vertex_displacements();

        // 5.5 线搜索
        for (line_search_iter = 0; line_search_iter < max_ls_iter; ++ls_iter) {
            // CCD 过滤
            alpha = filter_toi(alpha);

            // CFL 条件
            alpha = cfl_condition(alpha);

            // 计算试探能量 E = E(x + alpha * dx)
            E = compute_energy(alpha);

            // 能量下降检查
            if (E <= E0) break;
            alpha /= 2;
        }

        // 5.6 收敛检查
        if (converged && newton_iter >= min_iter) break;
    }

    // 6. 更新速度: v = (x - x_0) / dt
    update_velocity();
}
```

### 3.2 关键模块

```cpp
// 碰撞检测
m_global_trajectory_filter->detect(alpha);    // 宽相检测
m_global_trajectory_filter->filter_active(); // 窄相过滤
m_global_trajectory_filter->filter_toi(alpha);  // CCD TOI

// 接触系统
m_global_contact_manager->compute_adaptive_parameters();  // 自适应 kappa
m_global_contact_manager->compute_cfl_condition();         // CFL 条件

// 线性系统
m_global_linear_system->solve();  // PCG 求解

// 线搜索
m_line_searcher->compute_energy(true);  // 初始能量
m_line_searcher->compute_energy(false);  // 试探能量
```

---

## 4. 材料模型 (Constitution)

### 4.1 刚体 (Affine Body)

```cpp
// ABD 刚体
AffineBodyConstitution

// 关节
AffineBodyRevoluteJoint       // 旋转关节
AffineBodyPrismaticJoint      // 平移关节
AffineBodySphericalJoint      // 球形关节
AffineBodyFixedJoint          // 固定关节

// 关节约束
ExternalArticulationConstraint  // 外部关节约束
```

### 4.2 弹性体

```cpp
// 稳定 Neo-Hookean
StableNeoHookean

// ARAP (As-Rigid-As-Possible)
ARAP

// 有限元材料参数
ElasticModuli       // 弹性模量
ElasticModuli2D     // 2D 弹性模量
```

### 4.3 壳/布料

```cpp
// Baraff-Witkin 壳模型
BaraffWitkinShell
StrainLimitingBaraffWitkinShell  // 带应变限制

// 离散壳弯曲
DiscreteShellBending

// Neo-Hookean 壳
NeoHookeanShell
```

### 4.4 杆

```cpp
// 杆
AffineBodyRod

// Kirchhoff 杆弯曲
KirchhoffRodBending
```

### 4.5 约束

```cpp
// 软位置约束
SoftPositionConstraint

// 软变换约束
SoftTransformConstraint

// 缝合约束
SoftVertexStitch
SoftVertexEdgeStitch
SoftVertexTriangleStitch

// 弹簧
HookeanSpring
```

---

## 5. 碰撞检测系统

### 5.1 宽相检测

```cpp
// LBVH (Linear Bounding Volume Hierarchy)
// 基于 Morton code 的层次结构
```

### 5.2 窄相检测

```cpp
// 距离类型
enum class DistanceType {
    PP,  // Point-Point
    PE,  // Point-Edge
    PT,  // Point-Triangle
    EE   // Edge-Edge
};

// 距离计算
// 参考: ppf-contact-solver distance.hpp
```

### 5.3 CCD (连续碰撞检测)

```cpp
// CCD 过滤步长
Float filter_toi(Float alpha) {
    // 计算 TOI (time of impact)
    // 返回安全的步长 alpha
}
```

---

## 6. 接触系统

### 6.1 Barrier 函数

```cpp
// IPC barrier (对数形式)
// b(d) = -(d - d̂)² log(d / d̂)

// Codim-IPC barrier (支持厚度)
// B(D) = -κ(D - ξ² - d̂² - 2d̂ξ)² log((D - ξ²)/(d̂² + 2d̂ξ))
```

### 6.2 摩擦模型

```cpp
// 摩擦候选记录 (lagged)
record_friction_candidates();

// 摩擦计算
// 基于 lagged normal force
```

---

## 7. 线性系统求解

### 7.1 系统结构

```
H = H_inertia + H_elastic + H_contact + H_constraints

其中:
- H_inertia: 质量矩阵 (对角)
- H_elastic: 弹性 Hessian
- H_contact: 接触 Hessian (barrier)
- H_constraints: 约束 Hessian
```

### 7.2 PCG 求解

```cpp
// 预条件共轭梯度
// 预条件器: 对角 Jacobi

void GlobalLinearSystem::solve() {
    // 1. 组装系统矩阵
    // 2. PCG 迭代
    // 3. 返回解 dx
}
```

---

## 8. 自适应参数

### 8.1 Adaptive Kappa (GIPC)

```cpp
void compute_adaptive_kappa() {
    // 公式: new_kappa = clamp(-dot(g_c, g_nc) / dot(g_c, g_c), min_kappa, max_kappa)
    // 其中:
    // - g_c = barrier 梯度 (kappa=1)
    // - g_nc = 非接触梯度
}
```

---

## 9. Python API

### 9.1 基本使用

```python
import pyuipc as uipc

# 创建场景
scene = uipc.Scene()
world = uipc.World(scene)

# 添加几何体
mesh = uipc.geometry.SimplicialComplex(vertices, triangles)
scene.objects().create("object", mesh)

# 配置材料
constitution = uipc.constitution.StableNeoHookean()
scene.constitution_tabular().create(constitution)

# 运行仿真
engine = uipc.Engine(world)
engine.advance()  # 推进一帧
```

### 9.2 Genesis 集成

```python
# Genesis 使用 libuipc 作为 IPC 后端
from genesis.engine.couplers import IPCCoupler

coupler = IPCCoupler(simulator, options)
coupler.build()

# 每帧调用
simulator.step()  # 内部调用 libuipc 的 advance()
```

---

## 10. 与 Genesis 的集成

### 10.1 IPCCoupler 架构

```
Genesis Scene
    ├── RigidSolver (刚体求解)
    ├── FEMSolver (FEM 求解)
    └── IPCCoupler (IPC 耦合)
        └── libuipc Engine
            ├── ABD (刚体)
            ├── FEM (软体)
            └── Shell (布料)
```

### 10.2 数据流

```
Genesis → libuipc:
    - 顶点位置
    - 速度
    - 材料参数
    - 约束条件

libuipc → Genesis:
    - ABD 变换
    - 耦合力
    - 接触信息
```

---

## 11. 性能特点

### 11.1 GPU 优化
- 全 GPU 流程 (CUDA)
- 并行碰撞检测
- 并行线性求解

### 11.2 大规模支持
- 支持百万级接触对
- 支持大规模网格

---

## 12. 参考

- GitHub: https://github.com/spiriMirror/libuipc
- 文档: https://spirimirror.github.io/libuipc-doc/
- 示例: https://github.com/spiriMirror/libuipc-samples/
- 论文: StiffGIPC (SIGGRAPH 2025)