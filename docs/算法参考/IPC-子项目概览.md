# IPC 子项目概览

**项目位置**: `C:\csy_work\CG\Engine\IPC\`

IPC 目录包含多个 IPC 相关的实现项目，以下是简要分析。

---

## 1. Codim-IPC (C-IPC)

**位置**: `C:\csy_work\CG\Engine\IPC\Codim-IPC\`

### 论文信息
- **标题**: Codimensional Incremental Potential Contact (C-IPC)
- **作者**: Minchen Li, Danny M. Kaufman, Chenfanfu Jiang
- **发表**: SIGGRAPH 2021
- **网址**: https://ipc-sim.github.io/C-IPC/

### 核心特性
- 支持薄壳 (Shell) 和杆 (Rod) 的 IPC 接触
- Codimensional 表示减少 DOF 数量
- 支持布料、头发、线等 thin structures

### 目录结构
```
Codim-IPC/
├── Library/           # 核心库
├── Projects/          # 示例项目
│   └── FEMShell/      # FEM 壳仿真
├── Python/            # Python 绑定
├── Externals/         # 外部依赖
└── Documents/         # 文档
```

### 关键概念
```
Codimensional IPC 扩展了标准 IPC:
- 标准 IPC: 3D 体积网格 (四面体)
- C-IPC: 2D 薄壳 (三角形) + 1D 杆 (线段)

距离函数:
- PP (Point-Point)
- PE (Point-Edge)
- PT (Point-Triangle)
- EE (Edge-Edge)

厚度参数 ξ:
- Barrier 在 D > ξ² 时激活
- 支持不同厚度的物体接触
```

---

## 2. libuipc

**位置**: `C:\csy_work\CG\Engine\IPC\libuipc\`

### 项目简介
libuipc 是一个跨平台的现代 C++20 IPC 库，支持 GPU 并行仿真，提供 C++ 和 Python API。

### 核心特性
- **GPU 加速**: 全 GPU 并行流程
- **多物理耦合**: 刚体、软体、布料、绳索
- **无穿透接触**: IPC barrier 函数
- **可微分仿真**: 支持反向传播
- **Python API**: `pip install pyuipc`

### 目录结构
```
libuipc/
├── src/               # C++ 源码
│   ├── core/          # 核心框架
│   ├── geometry/      # 几何模块
│   ├── constitution/  # 材料模型
│   ├── contact/       # 接触系统
│   └── solver/        # 求解器
├── python/            # Python 绑定
├── include/           # 头文件
├── apps/              # 应用程序
└── docs/              # 文档
```

### 材料模型
```cpp
// Affine Body (刚体)
AffineBodyConstitution

// 弹性体
StableNeoHookean

// 布料/壳
StrainLimitingBaraffWitkinShell
DiscreteShellBending

// 约束
SoftTransformConstraint
ExternalArticulationConstraint
AffineBodyRevoluteJoint
AffineBodyPrismaticJoint
```

### Genesis 集成
libuipc 被 Genesis 物理仿真平台集成使用:
- Genesis 的 IPCCoupler 封装 libuipc
- 支持刚体-FEM-布料耦合仿真

---

## 3. fem-ipc

**位置**: `C:\csy_work\CG\Engine\IPC\fem-ipc\`

### 项目简介
专注于 FEM (有限元) 弹性体的 IPC 接触仿真。

### 核心特性
- FEM 弹性模型实现
- 四面体网格接触处理
- 与 IPC 原版相似的架构

---

## 4. Soft-Body-Simulation-CUDA

**位置**: `C:\csy_work\GPU/Engine/IPC\Soft-Body-Simulation-CUDA\`

### 项目简介
CUDA 实现的软体仿真，使用 IPC 接触处理。

### 核心特性
- CUDA GPU 加速
- 软体碰撞和接触
- 大规模并行计算

---

## 5. PD-IPC-ArmadilloDemo

**位置**: `C:\csy_work\CG\Engine\IPC\PD-IPC-ArmadilloDemo\`

### 项目简介
PD (Position-Based Dynamics) 与 IPC 结合的演示项目。

### 核心特性
- Position-Based Dynamics
- Armadillo 模型演示
- IPC 接触约束

---

## 各项目对比

| 项目 | 语言 | GPU | 壳/布料 | 刚体 | 可微分 |
|------|------|-----|---------|------|--------|
| IPC 原版 | C++ | ✗ | ✗ | ✗ | ✗ |
| Codim-IPC | C++ | ✗ | ✓ | ✗ | ✗ |
| libuipc | C++20 | ✓ | ✓ | ✓ (ABD) | ✓ |
| fem-ipc | C++ | 部分 | ✗ | ✗ | ✗ |
| Soft-Body-CUDA | CUDA | ✓ | ✗ | ✗ | ✗ |
| PD-IPC | C++ | 部分 | ✗ | ✗ | ✗ |

---

## 参考论文

1. **IPC (2020)**: "Incremental Potential Contact" - SIGGRAPH 2020
2. **C-IPC (2021)**: "Codimensional IPC" - SIGGRAPH 2021
3. **StiffGIPC (2025)**: "StiffGIPC: Advancing GPU IPC" - SIGGRAPH 2025
4. **AL-IPC (2025)**: "Augmented Lagrangian IPC" - 2025