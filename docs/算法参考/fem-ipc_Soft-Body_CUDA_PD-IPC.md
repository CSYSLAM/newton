# fem-ipc 项目分析

**项目位置**: `C:\csy_work\CG\Engine\IPC\fem-ipc`

---

## 1. 项目概述

fem-ipc 是一个专注于 FEM (有限元) 弹性体的 IPC 接触仿真项目，使用 xmake 构建系统。

### 核心特性
- FEM 弹性体仿真
- IPC 接触处理
- xmake 构建系统

---

## 2. 项目架构

```
fem-ipc/
├── fem-ipc/           # 核心库
│   ├── src/           # 源码
│   └── ...
├── apps/              # 示例应用
├── models/            # 模型文件
├── xmake.lua          # xmake 配置
└── CMakeLists.txt     # CMake 配置
```

---

## 3. 核心模块

### 3.1 FEM 弹性模型

```cpp
// 支持的弹性模型:
// - Neo-Hookean
// - Fixed Corotated
// - Linear Elastic
```

### 3.2 IPC 接触

```cpp
// Barrier 函数
// 碰撞检测
// 摩擦模型
```

---

## 4. 参考

- 项目: fem-ipc (IPC 目录下)

---

# Soft-Body-Simulation-CUDA 项目分析

**项目位置**: `C:\csy_work\CG\Engine\IPC\Soft-Body-Simulation-CUDA`

---

## 1. 项目概述

Soft-Body-Simulation-CUDA 是一个基于 CUDA 的软体仿真项目，使用 IPC 接触处理。

### 核心特性
- CUDA GPU 加速
- 软体碰撞和接触
- 大规模并行计算

---

## 2. 项目架构

```
Soft-Body-Simulation-CUDA/
├── src/               # CUDA 源码
├── tests/             # 测试
├── assets/            # 资源
├── vis.ipynb          # Jupyter 可视化
└── README.md
```

---

## 3. 核心模块

### 3.1 CUDA 内核

```cpp
// GPU 并行计算
// - 顶点更新
// - 碰撞检测
// - 接触响应
```

### 3.2 可视化

```python
# vis.ipynb
# Jupyter Notebook 用于结果可视化
```

---

## 4. 参考

- 项目: Soft-Body-Simulation-CUDA (IPC 目录下)

---

# PD-IPC-ArmadilloDemo 项目分析

**项目位置**: `C:\csy_work\CG\Engine\IPC\PD-IPC-ArmadilloDemo`

---

## 1. 项目概述

PD-IPC-ArmadilloDemo 是一个基于 Position-Based Dynamics (PBD) 与 IPC 结合的演示项目，使用 Armadillo 模型展示软体仿真。

### 核心特性
- Position-Based Dynamics (PBD)
- IPC 接触约束
- Armadillo 模型演示
- Qt 界面

---

## 2. 项目架构

```
PD-IPC-ArmadilloDemo/
├── SimFramework.h/cpp     # 主框架
├── SimFramework.ui          # Qt UI
├── Scene/                 # 场景管理
├── Simulator/             # 仿真器
├── Model/                 # 模型
├── Shader/                # 着色器
├── Ui/                    # UI 组件
├── Resources/             # 资源
└── main.cpp
```

---

## 3. 核心模块

### 3.1 PBD 求解器

```cpp
// Position-Based Dynamics
// 基于位置的约束求解

class Simulator {
    // 约束投影
    void project_constraints();

    // IPC 接触约束
    void project_ipc_constraints();
};
```

### 3.2 Armadillo 模型

```cpp
// Stanford Armadillo 模型
// 用于演示软体仿真
```

---

## 4. 参考

- 项目: PD-IPC-ArmadilloDemo (IPC 目录下)