# IPC 原版源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\IPC`

核心入口与关键源码：

- `src/main.cpp`
- `src/TimeStepper/Optimizer.cpp`
- `src/CollisionObject/SelfCollisionHandler.cpp`
- `src/CollisionObject/CollisionConstraints.cpp`
- `src/Energy/*`

## 1. 项目定位

这是 SIGGRAPH 2020 IPC 的参考实现，整体风格是：

- CPU 主导
- 稀疏 Hessian + 直接线性求解
- barrier 接触 + CCD 步长过滤
- 兼容三种接触求解模式：`IP`、`QP`、`SQP`

从源码看，真正的核心不在 `main.cpp`，而在 `Optimizer<dim>`：

- `main.cpp` 负责配置、网格构造、可视化、逐步推进
- `Optimizer.cpp` 负责时间积分、能量/梯度/Hessian、线性求解、线搜索
- `SelfCollisionHandler.cpp` 负责自碰撞候选集、约束值、梯度、Hessian、摩擦和 CCD

## 2. 项目架构

```text
main.cpp
-> Config / Mesh / Energy 初始化
-> Optimizer::solve()
   -> AnimScripter 更新 DBC/NBC/运动脚本
   -> fullyImplicit() 或 fullyImplicit_IP()
      -> computeXTilta()
      -> computeConstraintSets()
      -> computeEnergyVal()
      -> computeGradient()
      -> computePrecondMtr()
      -> linSysSolver->solve()
      -> lineSearch()
      -> 更新 V / V_prev / velocity / acceleration
```

模块职责可以压缩成下面几类：

| 模块 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `Config` | 解析脚本和参数 | 输入脚本/命令行 | 求解参数、时间步、碰撞体配置 |
| `Mesh` | 保存几何和材料状态 | 顶点、四面体、面片 | `V/F/T`、材料参数、邻接 |
| `Energy` | 弹性能量 | `Mesh`、材料参数 | 能量、梯度、Hessian |
| `SelfCollisionHandler` | 自碰撞与摩擦 | 表面顶点/边/面 | 活动集、CCD 步长、接触导数 |
| `CollisionObject/*` | 外部碰撞体 | mesh 与碰撞体 | 约束值与接触导数 |
| `LinSysSolver/*` | 稀疏线性系统 | Hessian、梯度 | 搜索方向 |
| `Optimizer` | 主时间积分器 | 全部状态 | 新位置、新速度、统计 |

## 3. 算法模型

源码中的主目标函数是隐式时间积分下的总势能最小化：

```text
E_total(x)
= E_inertia(x; x_tilde)
+ E_elastic(x)
+ E_contact_barrier(x)
+ E_friction(x)
+ E_DBC/NBC(x)
```

其中：

- `x_tilde` 是预测位置，由上一帧位置、速度、重力和积分格式给出
- `E_elastic` 由 `NeoHookeanEnergy` 或 `FixedCoRotEnergy` 给出
- `E_contact_barrier` 由 barrier 函数和当前活动接触集给出
- `E_friction` 由接触切向基、上一轮法向力和摩擦参数给出

源码里默认 barrier 是 `C2` 的对数势垒：

```text
b(d) = -(d - dHat)^2 log(d / dHat)
```

这点在 `src/Utils/BarrierFunctions.hpp` 很明确。

## 4. 数据链路

### 4.1 初始化链路

```text
配置脚本
-> Config
-> Mesh / collisionObjects / meshCollisionObjects / selfCollision flags
-> Energy terms
-> Optimizer 构造
-> setTime()
-> computeXTilta()
-> 初始 SpatialHash
```

### 4.2 每时间步链路

```text
AnimScripter.stepAnimScript()
-> 必要时更新边界条件与脚本驱动
-> updatePrecondMtrAndFactorize()
-> fullyImplicit[_IP]()
-> 更新 result.V
-> 回写 velocity / acceleration / V_prev
-> computeXTilta() 准备下一步
```

### 4.3 接触数据链

```text
表面顶点/边/面
-> SpatialHash 查询
-> SelfCollisionHandler::computeConstraintSet()
-> MMCVID 活动集
-> evaluateConstraints()
-> compute_b()
-> augment gradient / hessian / friction
-> CCD / line search 使用同一批候选或其子集
```

## 5. 碰撞流程

### 5.1 候选生成

`SelfCollisionHandler::computeConstraintSet()` 做的事情很清楚：

- 点对三角形：分类成 `PP / PE / PT`
- 边对边：分类成 `PP / PE / EE / nearly parallel EE`
- 可选地生成 `PTEE` 集合供 CCD 使用

它先用 `SpatialHash` 做近邻查询，再做精确距离分类。

### 5.2 接触类型

编码使用 `MMCVID`，本质上就是一个 4 元 primitive stencil：

- `(-v-1, t0, t1, t2)` 表示点三角形
- `(e0a, e0b, e1a, e1b)` 表示边边
- 某些位置为 `-1` 时表示退化成点点或点边

### 5.3 CCD 与步长过滤

源码里有多种 CCD 路径：

- `largestFeasibleStepSize()`
- `largestFeasibleStepSize_CCD()`
- `largestFeasibleStepSize_CCD_TightInclusion()`
- `largestFeasibleStepSize_CCD_exact()`

这说明原版实现把“能量下降线搜索”和“无交叉步长限制”分成两层：

1. 先用 CCD 给出物理可行步长上界
2. 再用回溯线搜索保证能量下降

### 5.4 摩擦

摩擦数据链在 `SelfCollisionHandler.cpp` 中：

- `computeDistCoordAndTanBasis()`
- `computeFrictionEnergy()`
- `augmentFrictionGradient()`
- `augmentFrictionHessian()`

这说明法向接触和切向摩擦不是两个独立求解器，而是被统一写回总能量导数中。

## 6. 仿真主循环

### 6.1 `Optimizer::solve()` 的角色

`solve(int maxIter)` 在工程上不是“单次牛顿迭代”，而是“若干时间步推进器入口”：

- 先让 `AnimScripter` 推进运动脚本
- 再进入 `fullyImplicit()` 或 `fullyImplicit_IP()`
- 最后更新时间积分状态

### 6.2 `solve_oneStep()` 的角色

`solve_oneStep()` 是单轮牛顿子步骤：

- 组装 Hessian
- 数值分解
- 求解搜索方向
- 做 line search
- 更新 `result`

### 6.3 伪代码

```text
for each frame step:
    update scripted motion / DBC / NBC
    recompute and factorize proxy/Hessian if needed

    while not converged:
        build or refresh contact active sets
        E  = computeEnergyVal(x)
        g  = computeGradient(x)
        H  = computePrecondMtr(x)
        dx = solve(H, -g)

        alpha_ccd = CCD_filter(x, dx)
        alpha_ls  = backtracking_line_search(x, dx, alpha_ccd)
        x = x + alpha_ls * dx

    velocity, acceleration = update from x and previous state
    x_tilde = predict_next_position()
```

## 7. 关键函数输入输出

## 7.1 `main.cpp::proceedOptimization()`

输入：

- 当前 `optimizer`
- 当前 `iterNum`
- 配置中的容差表 `tol`

输出：

- 调一次 `optimizer->solve(1)`
- 更新当前显示、输出、日志

作用：

- 把“一个 frame 的推进”包装成用户可见的仿真 step

## 7.2 `Optimizer::solve(int maxIter)`

输入：

- `maxIter`
- 当前 `result`、`velocity`、`xTilta`
- `animConfig`

输出：

- 返回状态码：收敛 / 达到上限 / 失败
- 更新 `result.V`、`velocity`、`acceleration`、`globalIterNum`

实现要点：

- 先执行脚本驱动
- 再调用 `fullyImplicit()` 或 `fullyImplicit_IP()`
- 最后按 `BE` 或 `Newmark` 更新速度与加速度

## 7.3 `Optimizer::solve_oneStep()`

输入：

- 当前梯度、Hessian 模式、活动集、求解器

输出：

- `searchDir`
- 更新后的 `result`
- 是否在 line search 中提前停止

实现要点：

- `computePrecondMtr(result, false, linSysSolver)`
- `linSysSolver->factorize()`
- `linSysSolver->solve(-gradient, searchDir)`
- `lineSearch(stepSize)`

## 7.4 `Optimizer::computeEnergyVal(const Mesh&, ...)`

输入：

- 当前网格状态 `data`
- `xTilta`
- 活动接触集

输出：

- 标量总能量 `energyVal`

实现组成：

- 各弹性能量项
- 惯性项 `||x - x_tilde||_M^2 / 2`
- NBC 势能
- 外部接触 / mesh 接触 / 自碰撞 barrier
- 自碰撞摩擦能

## 7.5 `SelfCollisionHandler::computeConstraintSet(...)`

输入：

- `mesh`
- `SpatialHash`
- `dHat`
- 是否同时返回 `PTEE`

输出：

- `constraintSet`
- `paraEEMMCVIDSet`
- `paraEEeIeJSet`
- `cs_PTEE`

实现要点：

- 对所有表面顶点做点三角查询
- 对所有表面边做边边查询
- 根据距离类型把 primitive 归一成统一 stencil
- 特判近平行边边

## 7.6 `SelfCollisionHandler::largestFeasibleStepSize_CCD_exact(...)`

输入：

- 当前位置
- 搜索方向
- CCD 方法枚举

输出：

- 不产生交叉的最大步长

作用：

- 给 line search 一个物理可行上界

## 8. 输入输出视角

### 8.1 主要输入

- 体网格/表面网格
- 材料参数：`E`、`nu`、密度
- 时间积分参数：`dt`、总时长、阻尼
- 接触参数：`dHat`、摩擦、CCD 方法
- 边界条件：DBC / NBC / 运动脚本

### 8.2 运行时核心状态

- `result.V`
- `result.V_prev`
- `velocity`
- `acceleration`
- `xTilta`
- `activeSet` / `MMActiveSet`
- `MMLambda_lastH` / `MMTanBasis`

### 8.3 主要输出

- 每帧 `.obj`
- `status<n>`
- `iterStats.txt`
- `sysE/sysL/sysM`
- PNG/GIF/日志

## 9. 实现特征总结

- 优点：结构完整，论文概念到工程实现的对应关系最清楚。
- 局限：CPU 路线，代码分支很多，`IP/QP/SQP`、摩擦和多 CCD 路径交织，阅读成本高。
- 最值得学的部分：
  - `Optimizer` 如何把“动力学 + 接触 + 线性求解 + 线搜索”统一起来
  - `SelfCollisionHandler` 如何把不同 primitive 的接触统一成一个活动集体系
  - barrier、摩擦、CCD 三者如何共享同一套 stencil 数据
