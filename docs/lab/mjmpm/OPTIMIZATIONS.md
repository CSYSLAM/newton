# MJMPM 优化索引

本页汇总当前分支已保留的 MJMPM 性能改动。各项测量使用的场景和配置不同，不能将表中的百分比相加，也不能推断为任意场景的加速比。功能、稳定性修复和实验性选项另见对应的 changelog 条目。

| 路径 | 已保留的改动 | 详细记录 |
| --- | --- | --- |
| CUDA Graph 与容量 | 固定容量的接触基和可重建稀疏网格支持 Graph 回放；未捕获步骤确定单元粒子容量，回放后可检查溢出。 | [容量条目](../../../changelog/+mjmpm-capture-capacity.changed.md)、[实施记录](README.md) |
| 碰撞与任务编排 | 刚体接触按 world 并行；剔除远距离 MPM 碰撞体和 MuJoCo 已负责的形状对；按（节点，碰撞体）并行栅格化，并把独立 MPM 准备放到第二条 CUDA 流。 | [性能条目](../../../changelog/+mjmpm-performance.changed.md)、[栅格化条目](../../../changelog/+mjmpm-parallel-rasterization.changed.md)、[米锅测量](RICE_COOKER_PERF.md) |
| MPM 网格准备 | 从粒子分箱直接构造 PIC 碰撞体邻接；P0 等单节点应变基直接积分质量对角线；小型关节系统使用专用 CUDA 三角代入路径。 | [邻接](../../../changelog/+mjmpm-pic-adjacency-6f4e82a1.changed.md)、[P0](../../../changelog/+mjmpm-p0-strain-mass.changed.md)、[三角代入](../../../changelog/+mjmpm-local-substitution-5d9e27b1.changed.md)、[米锅测量](RICE_COOKER_PERF.md) |
| MPM 迭代 | 残差停滞时可提前结束；小应变色块分散到更多 CUDA 块，大色块保持连续映射。面包连接矩阵改用粒子邻接 gather。 | [停滞退出](../../../changelog/+mjmpm-stall-exit.added.md)、[颜色映射](../../../changelog/+mjmpm-gs-color-mapping.changed.md)、[面包矩阵](../../../changelog/+mjmpm-bread-solve.changed.md)、[面包测量](BREAD_CUT_PERF.md) |
| 刚体接触 | 跨步热启动、预计算关节连杆接触响应、按局部状态做顺序 GS 扫描、CUDA cooperative 扫描，以及带数值检查和回退的接触空间 Delassus 求解。Jacobi 是可选路径，夹持场景默认仍用 GS。 | [热启动](../../../changelog/+mjmpm-rigid-warm-start.added.md)、[连杆响应](../../../changelog/+mjmpm-link-responses.changed.md)、[局部扫描](../../../changelog/+mjmpm-local-rigid-sweeps.changed.md)、[cooperative 扫描](../../../changelog/+mjmpm-cooperative-sweeps-c84f9a21.changed.md)、[接触空间](../../../changelog/+mjmpm-contact-space.changed.md)、[米锅测量](RICE_COOKER_PERF.md) |
| 刚体与材料耦合 | 小型刚体碰撞体直接装配稀疏耦合块；符合条件的同一步 MuJoCo 反馈重算复用位置和速度导数；扩大融合刚体迁移率反馈的适用规模。接触空间的第二次求解复用未变的接触几何与 Delassus 矩阵。 | [稀疏装配](../../../changelog/+mjmpm-rigid-coupling-assembly.changed.md)、[导数复用](../../../changelog/+mjmpm-force-resolve.changed.md)、[融合反馈](../../../changelog/+mjmpm-bread-solve.changed.md)、[接触复用](TEAPOT_PERF.md) |
| 耦合收敛与稳定性 | 可选同一步迭代使用 Aitken 松弛与界面残差提前退出；默认 constrained 模式让材料求解看到刚体接触支撑的当前迁移率。这里的目标主要是数值行为，而非通用加速。 | [同一步耦合](../../../changelog/+mjmpm-same-step.added.md)、[constrained 模式](../../../changelog/+mjmpm-constrained-material-coupling.added.md) |
| 示例与显示 | W1 抓取示例调整默认求解配置并减少逐帧读回；米锅复用静态显示属性；面包减少绘制批次和实例属性上传；GL 窗口事件改为非阻塞轮询。 | [W1 配置](../../../changelog/+mjmpm-w1-speed.changed.md)、[米锅测量](RICE_COOKER_PERF.md)、[面包测量](BREAD_CUT_PERF.md)、[事件轮询](../../../changelog/+viewer-event-polling.fixed.md) |

倒水示例的接触复用、完整回放和帧率数据见 [倒水性能记录](TEAPOT_PERF.md)。[米锅性能记录](RICE_COOKER_PERF.md)第 2.8 节保存了更早、已回退的接触准备复用实验；它与当前仅在接触空间路径复用矩阵的实现不是同一个改动。

未保留的候选方案与失败原因分别记在[米锅性能记录](RICE_COOKER_PERF.md)和[面包性能记录](BREAD_CUT_PERF.md)。Changelog 记录发布层面的变化，性能日志保留场景、对照配置和测量边界。

三例联合剖面与本轮未保留的通用求解器原型见[跨场景优化调查](GENERAL_OPTIMIZATION_REVIEW.md)。

## Mesh SDF collision culling (2026-10-26)

See [edge group culling](EDGE_GROUP_CULLING.md) for the solver-level algorithm, measured teapot speedup, and contact equivalence check.
