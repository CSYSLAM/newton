# IPC 项目源码分析索引

本目录基于以下两类信息整理：

- `docs/算法参考/` 现有参考文档
- `C:\csy_work\CG\Engine\IPC\` 下各项目的 `README`、入口文件、求解器、碰撞检测和能量模块源码

目标不是复写论文，而是把“代码里真正怎么跑”拆成统一视角：

- 算法模型
- 仿真主循环
- 碰撞流程
- 数据链路
- 模块设计
- 关键函数输入输出
- 关键实现思路与伪代码

## 项目列表

- [IPC-原版-源码分析.md](./IPC-原版-源码分析.md)
- [Codim-IPC-源码分析.md](./Codim-IPC-源码分析.md)
- [fem-ipc-源码分析.md](./fem-ipc-源码分析.md)
- [libuipc-源码分析.md](./libuipc-源码分析.md)
- [PD-IPC-ArmadilloDemo-源码分析.md](./PD-IPC-ArmadilloDemo-源码分析.md)
- [Soft-Body-Simulation-CUDA-源码分析.md](./Soft-Body-Simulation-CUDA-源码分析.md)

## 统一观察框架

虽然 6 个项目实现风格差异很大，但都能抽象成一条共用主链：

```text
场景/网格/材料/边界条件输入
-> 状态初始化(X, V, 质量, 拓扑, 接触参数)
-> 预测位置(X_tilde)
-> 宽相碰撞候选生成
-> 窄相距离/CCD/TOI
-> 接触能量或接触约束组装
-> 求解增量(牛顿、局部-全局、Jacobi/PCG/Cholesky)
-> 线搜索/步长过滤
-> 更新位置与速度
-> 输出网格、日志、统计
```

## 快速对比

| 项目 | 主语言 | 核心求解 | 接触表示 | 宽相 | 典型对象 |
| --- | --- | --- | --- | --- | --- |
| IPC 原版 | C++ | Newton + 线搜索 | barrier / 约束 / QP | Spatial Hash | 四面体弹性体 |
| Codim-IPC | Python + C++ 模板库 | IE/SIE + 增量势能 | Codim barrier | Spatial Hash | 壳、杆、粒子、体 |
| fem-ipc | C++ | LDLT Newton | barrier | 暴力枚举 | 小规模 FEM |
| libuipc | C++20 + CUDA | GPU Newton 管线 | barrier / AL active set | GPU trajectory filters | FEM、ABD、壳、杆 |
| PD-IPC-ArmadilloDemo | C++ + CUDA | Local-Global + Jacobi/Chebyshev | penalty/IPC 混合 | Patch BVH | 体 + 布 |
| Soft-Body-Simulation-CUDA | C++ + CUDA | 双精度 IPC / 单精度 PD | barrier | BVH + CCD | 软体教学框架 |

## 阅读建议

- 如果你想理解“论文 IPC 最原始的 CPU 实现”，先看 `IPC-原版-源码分析.md`
- 如果你关心“布料/杆/缝合/厚度”，看 `Codim-IPC-源码分析.md`
- 如果你想看“最小可读的简化版实现”，看 `fem-ipc-源码分析.md`
- 如果你关心“现代 GPU 大系统架构”，看 `libuipc-源码分析.md`
- 如果你想看“PD 与 IPC 混合”的工程实现，看 `PD-IPC-ArmadilloDemo-源码分析.md`
- 如果你想看“教学型 CUDA 软体框架里的 IPC 分支”，看 `Soft-Body-Simulation-CUDA-源码分析.md`

## 说明

- 文档中的“确认”部分来自已读源码。
- 文档中的“推断”部分来自模块命名、调用关系和常见 IPC 结构，已显式标注。
- 这里更偏重工程实现链路，不追求论文公式的完全展开。
