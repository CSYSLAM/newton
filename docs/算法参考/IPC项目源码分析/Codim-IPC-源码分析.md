# Codim-IPC 源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\Codim-IPC`

关键源码：

- `Python/Drivers/SimulationBase.py`
- `Python/Drivers/FEMDiscreteShellBase.py`
- `Library/FEM/Shell/IMPLICIT_EULER.h`
- `Library/FEM/IPC.h`

## 1. 项目定位

`Codim-IPC` 的核心特征是把 IPC 从“3D 体网格接触”扩展到：

- 2D 壳/布料
- 1D 杆/头发
- 0D 离散粒子
- 它们和 3D 体对象之间的 codimensional 接触

工程形态不是单体可执行程序，而是：

- Python 驱动脚本负责场景搭建和参数调度
- `JGSL`/`Library` 中的 C++ 模板库负责真正的计算

也就是说，Python 是“场景 DSL”，C++ 才是“求解器内核”。

## 2. 项目架构

```text
Projects/FEMShell/*.py
-> Python/Drivers/FEMDiscreteShellBase.py
   -> add_shell / add_rod / add_particle / initialize_*
   -> advance_one_time_step()
      -> FEM.DiscreteShell.Advance_One_Step_*
         -> Library/FEM/Shell/IMPLICIT_EULER.h
            -> 预测位置 Xtilde
            -> 表面 primitive 提取
            -> Compute_Constraint_Set()
            -> Compute_IncPotential()
            -> Solve_Direct / AMGCL / 线搜索
```

从 Python 层看，外部接口非常稳定：

- `initialize()`
- `initialize_rod()`
- `initialize_OIPC()`
- `initialize_EIPC()`
- `advance_one_time_step()`
- `run()`

从 C++ 层看，核心模板函数族很多，但主线集中在：

- `Advance_One_Step_IE_*`
- `Advance_One_Step_SIE_*`
- `Line_Search(...)`
- `Compute_Constraint_Set(...)`
- `Compute_IncPotential(...)`

## 3. 算法模型

### 3.1 对象层

从 `FEMDiscreteShellBase.py` 能看出它支持四类对象：

- shell：三角网格中面
- rod：线段链
- particle：离散点
- volumetric object：四面体体

### 3.2 势能层

从接口和命名可以确认以下能量族：

- 薄壳膜能 / hinge bending
- 体弹性能
- rod stretching / rod bending
- stitch 缝合能
- IPC 或 OIPC/EIPC 接触势
- 摩擦势
- DBC / body force / gravity

### 3.3 OIPC 与 EIPC

Python 层有两条初始化路径：

- `initialize_EIPC(E, nu, thickness, h)`
- `initialize_OIPC(thickness, offset, stiffMult=1)`

可确认的结论：

- `EIPC` 更接近把弹性和 IPC 紧耦合
- `OIPC` 明确暴露 `offset`，用于 codim 接触厚度偏移

更直白地说，`Codim-IPC` 不只是“碰撞库”，而是“壳/杆物理 + codim 接触”的整套求解器。

## 4. 数据链路

### 4.1 场景搭建链路

```text
Python 场景脚本
-> add_shell / add_garment / add_rod / add_particle / add_object
-> X / Elem / rod / tet / stitchInfo / compNodeRange
-> initialize_*()
-> nodeAttr / elemAttr / massMatrix / bodyForce / elasticity / kappa / dHat2
```

### 4.2 时间步链路

`SimulationBase.advance_one_frame()` 会把一帧拆成多个 `dt` 子步：

```text
frame_dt
-> 若剩余时间较大，拆成多个 current_dt
-> advance_one_time_step(current_dt)
-> TIMER_FLUSH
```

### 4.3 单步核心链路

`FEMDiscreteShellBase.advance_one_time_step()` 的代码主线是：

```text
更新法向流 / DBC / 序列目标 / rest-shape 追踪
-> 选择求解器族:
   Advance_One_Step_IE_Hinge_EIPC
   Advance_One_Step_IE_Hinge
   Advance_One_Step_SIE_Hinge
   Advance_One_Step_IE_Flow
-> 更新时间 t
-> 按需更新缩放、速度清零
```

这意味着 Python 层本身不求解，只负责“把所有状态塞进内核函数”。

## 5. 碰撞流程

### 5.1 `Compute_Constraint_Set(...)`

`Library/FEM/IPC.h` 是碰撞主入口。

函数输入：

- `X`：当前节点坐标
- `boundaryNode / boundaryEdge / boundaryTri`
- `particle / rod`
- `NNExclusion`：缝合或邻近排除集
- `dHat2`、`thickness`
- `DBCb`

函数输出：

- `constraintSet`
- `cs_PTEE`
- `stencilInfo`

### 5.2 宽相

源码中默认启用了 `USE_SH_CCS`，说明宽相依赖：

- `SPATIAL_HASH<T, dim>`

它会对：

- point 查询 triangle
- edge 查询 edge
- codim point 查询 edge

做局部近邻筛选。

### 5.3 窄相与接触分类

3D 分支里点三角部分有 7 类：

- 点点到三角顶点
- 点边到三角边
- 点三角面内投影

边边部分则统一成 `MMCVID` 类似的 4 点 stencil。

此外还有 codim 补充分支：

- rod/particle 对 edge 的 `PP/PE`

### 5.4 厚度处理

源码里先做：

```text
dHat = sqrt(dHat2) + thickness
dHat2 = dHat * dHat
```

这说明 codim 厚度不是额外后处理，而是直接吸收到接触激活半径里。

## 6. 仿真与求解流程

### 6.1 `Advance_One_Step_IE_Discrete_Shell(...)`

这是最关键的模板函数之一。能确认的流程如下：

1. 若不是静态求解，记录 `Xn` 并构造 `Xtilde`
2. 提取表面 primitive：
   - `boundaryNode`
   - `boundaryEdge`
   - `boundaryTri`
   - rod / particle 附加 primitive
3. 构造邻接排除集 `NNExclusion`
4. 设置 DBC mask 和位移
5. 后续进入增量势能、线性系统和线搜索

### 6.2 `Line_Search(...)`

`IMPLICIT_EULER.h` 里的 `Line_Search` 清楚展示了两层过滤：

1. 先检查增量势能可行性，必要时减半 `alpha`
2. 若开启碰撞，再执行 `Compute_Intersection_Free_StepSize(...)`
3. 重新计算势能
4. 若碰撞后最小距离 `<= 0`，继续回退
5. 若能量不下降，继续回退

### 6.3 主伪代码

```text
advance_one_time_step(dt):
    update scripted DBC / loading / optional normal flow

    choose kernel family:
        IE_Hinge_EIPC / IE_Hinge / SIE_Hinge / IE_Flow

kernel Advance_One_Step_*:
    if dynamic:
        Xn = X
        Xtilde = X + h * v + h^2 * M^{-1} b

    build boundary primitives
    build exclusion sets
    build contact constraints via spatial hash

    repeat until Newton/PN convergence:
        assemble incremental potential
        assemble linear system
        solve for search direction
        line search with feasibility + CCD + energy decrease
        update X

    update velocities / states
```

## 7. 关键函数输入输出

## 7.1 `SimulationBase.run()`

输入：

- `frame_num`
- `frame_dt`
- 子类实现的 `advance_one_time_step()`

输出：

- 逐帧输出 `.obj`

作用：

- 统一所有示例脚本的帧循环逻辑

## 7.2 `FEMDiscreteShellBase.initialize(...)`

输入：

- 密度、`E`、`nu`、`thickness`
- 当前 `X`、`Elem`、`segs`

输出：

- `X0`
- `nodeAttr`
- `massMatrix`
- `bodyForce`
- `elemAttr`
- `elasticity`
- 更新后的 `dHat2`、`kappa`

## 7.3 `FEMDiscreteShellBase.initialize_OIPC(...)`

输入：

- `thickness`
- `offset`
- `stiffMult`

输出：

- 更新 `dHat2`
- 更新 `kappa`
- 设置 `self.thickness = offset`

作用：

- 用于 codim 有厚度接触

## 7.4 `FEMDiscreteShellBase.advance_one_time_step(dt)`

输入：

- 当前状态与 `dt`

输出：

- 更新 `self.X`
- 增加 `PNIterCount`
- 更新时间 `self.t`

实现特点：

- 只做调度，不做数值核心
- 根据 `elasticIPC`、`split`、`flow` 选择不同求解内核

## 7.5 `Compute_Constraint_Set(...)`

输入：

- 当前几何 primitive 集
- `dHat2`
- `thickness`
- `NNExclusion`

输出：

- 接触 stencil 集
- 可选 `PTEE` 集
- `stencilInfo`

## 7.6 `Line_Search(...)`

输入：

- 当前 `sol`
- `Xprev`
- `constraintSet`
- `dHat2`
- 摩擦、DBC、stitch、rod、tet 等所有系统状态

输出：

- 更新 `X`
- 回写 `alpha`
- 回写 `feasibleAlpha`
- 回写新的 `Eprev`

## 8. 输入输出视角

### 8.1 输入

- 三角网格、杆、粒子、四面体体
- 材料：密度、杨氏模量、泊松比、厚度
- 接触参数：`dHat2`、`kappa`、`mu`、`epsv2`
- DBC、序列驱动、缝合信息

### 8.2 运行时状态

- `X / X0 / X_stage`
- `Elem / segs / rod / tet`
- `nodeAttr / elemAttr / tetAttr`
- `massMatrix / bodyForce`
- `edge2tri / edgeStencil / edgeInfo`
- `stitchInfo / stitchRatio`

### 8.3 输出

- `shell*.obj`
- 可选 `seg*.obj`
- 可选 `rod*.obj`
- 可选体对象表面导出

## 9. 实现特征总结

- 优点：壳、杆、粒子、体和缝合都在同一求解框架内，codim 支持最完整。
- 局限：模板层和 Python 封装层相隔较远，第一次读会觉得跳跃。
- 最值得学的部分：
  - Python 场景层如何把复杂对象统一装配成内核输入
  - `Compute_Constraint_Set` 如何把不同维度 primitive 统一进接触集合
  - `Line_Search` 如何把可行性、CCD、能量下降和摩擦整合起来
