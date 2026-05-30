# Soft-Body-Simulation-CUDA 源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\Soft-Body-Simulation-CUDA`

关键源码：

- `src/simulation/simulationContext.cpp`
- `src/simulation/simulationContext.cu`
- `src/simulation/solver/IPC/ipc.cu`
- `src/collision/ccd.cu`
- `src/collision/broadphase.cu`

## 1. 项目定位

这是一个教学型 CUDA 软体框架，但它内部其实有两条求解路线：

- `float32`：Projective Dynamics
- `float64`：Incremental Potential Contact

这点在 `simulationContext.cu` 里是明确写死的：

- `double -> IPCSolver`
- `float -> PdSolver`

所以这个项目要区分两层理解：

1. 作为框架，它支持多种 solver
2. 作为 IPC 实现，核心在 `src/simulation/solver/IPC/ipc.cu`

## 2. 项目架构

```text
Context(context.json)
-> SimulationCUDAContext
   -> DataLoader / FixedBodyData / CollisionDetection
   -> choose solver by precision
      -> IPCSolver (double)
      -> PdSolver  (float)
-> Update()
   -> solver->Update(data, params)
   -> CollisionDetection PrepareRenderData()
```

碰撞部分独立成模块：

```text
collision/
-> bvh.*
-> broadphase.cu
-> narrowphase.cu
-> intersections.cu
-> ccd.cu
```

## 3. 算法模型

对 `double` 路径，源码清楚表明它是标准 IPC 结构：

- `computeXTilde`
- barrier energy
- `SearchDirection()`
- line search
- collision detection `UpdateQueries(...)`
- `updateVel`

从 `ipc.cu` 的调用可推断总能量至少包括：

- 惯性项
- 弹性项
- barrier 接触项
- DBC 相关项

真正的数值细节封装在 `energy` 对象里：

- `energy.Val(...)`
- `energy.GradientHessian(...)`
- `energy.InitStepSize(...)`

## 4. 数据链路

### 4.1 初始化链路

`SimulationCUDAContext::Impl<Scalar>::Init(...)` 做了这些事：

1. 读取 `context.json`
2. 加载软体 mesh、材料参数、DBC
3. 创建 `CollisionDetection`
4. 分配 solver data
5. 根据精度选择 solver

可确认的求解参数包括：

- `dt`
- `kappa`
- `tolerance`
- `maxIterations`
- `dhat`
- `gravity`
- `damp`
- `muN`、`muT`

### 4.2 每帧链路

`SimulationCUDAContext::Update()`：

```text
UI 参数
-> CopyUIToParams()
-> solver->Update(data, params)
-> CollisionDetection::UpdateX()
-> CollisionDetection::PrepareRenderData()
```

因此 UI、仿真和渲染是弱耦合的：

- UI 只更新 `SolverParams`
- solver 只更新 `SolverData`
- 碰撞模块负责候选查询和调试渲染

## 5. 碰撞流程

## 5.1 BVH 宽相

从 `broadphase.cu` 可确认：

- 支持 `BuildBVHTreeCCD(...)`
- 支持普通 `BuildBVHTree(...)`
- 然后 `DetectCollisionCandidates(...)`

这说明它会对：

- 当前几何
- 或 swept 几何

构造 BVH，再做候选检测。

## 5.2 `CollisionDetection`

`ccd.cu` 中 `CollisionDetection<Scalar>` 负责：

- 管理 queries 缓冲
- 管理 BVH
- 保存 `mpX`、`mpP`
- 准备调试可视化数据

从接口看，它更像“碰撞检测服务对象”而不是单纯的 kernel 集合。

## 5.3 query 级碰撞

从 `Query` 调试视图可见，它支持：

- `EE`
- 点三角等其他 query 类型
- `toi`
- `normal`
- 距离类型标签

说明窄相最终会把候选对写成统一 query 数组。

## 5.4 CCD

`intersections.cu` 里有：

- `ccdTriangleIntersectionTest`
- `ccdCollisionTest`

而 `narrowphase.cu` 会把 `toi` 写入 query。

因此其 CCD 路线是：

```text
BVH 候选
-> narrowphase per-query
-> query.toi / normal
-> solver line search or post-collision handling use it
```

## 6. IPC 求解流程

## 6.1 `IPCSolver::Update(...)`

逻辑非常简单：

- 若之前失败，直接返回
- 否则执行 `SolverStep`
- 若失败则标记 `failed`

## 6.2 `IPCSolver::SolverStep(...)`

源码主线：

1. `SolverPrepare()` 生成 fixed mask
2. 保存 `x_n`
3. `computeXTilde`
4. `pCollisionDetection->UpdateQueries(...)`
5. `energy.UpdateKappa(...)`
6. `E_last = energy.Val(...)`
7. `SearchDirection(...)`
8. while 未满足终止条件：
   - `computeXMinusAP`
   - `alpha = energy.InitStepSize(...)`
   - 回溯线搜索直到 `E <= E_last`
   - 接受 `xTmp -> X`
   - 更新 queries
   - 重算 `E_last`
   - 重算搜索方向
9. `updateVel(...)`

## 6.3 终止条件

`EndCondition(h, tolerance)` 直接检查：

```text
||p||_inf / h < tolerance
```

其中 `p` 是线性系统解出的搜索方向。

## 6.4 线性求解

`IPCSolver` 支持多种线性求解器：

- `CholeskySpImmedSolver`
- `CGSolver`
- `PCGJacobiSolver`
- `JacobiSolver`

`SetLinearSolver()` 允许 UI 动态切换。

## 6.5 固定自由度消元

`DOFElimination(...)` 分两步：

- `DOFEliminationHessKernel`
- `DOFEliminationGradKernel`

即：

- 固定点相关 Hessian 行列改成单位块
- 固定点梯度清零

这是很标准的“硬约束消元”写法。

## 7. 主伪代码

```text
Update():
    load UI params
    solver.Update()
    refresh collision debug data

IPCSolver::SolverStep():
    build fixed mask
    x_n = x
    x_tilde = x + h * v * damp
    update collision queries on current x
    update contact kappa
    E_last = energy(x)
    p = solve(H(x), g(x))

    while not converged:
        alpha = init_step_size_from_collision()
        while energy(x - alpha p) > E_last:
            alpha /= 2
        x = x - alpha p
        update collision queries
        E_last = energy(x)
        p = solve(H(x), g(x))

    v = (x - x_n) / h
```

## 8. 关键函数输入输出

## 8.1 `SimulationCUDAContext::Update()`

输入：

- UI 参数
- 当前 `SolverData`

输出：

- 更新后的 solver 状态
- 调试渲染数据

## 8.2 `IPCSolver::SolverPrepare(...)`

输入：

- `DBCIdx`
- `numDBC`

输出：

- `d_isFixed`

## 8.3 `IPCSolver::SolverStep(...)`

输入：

- `SolverData`
- `SolverParams`

输出：

- 更新后的 `X`
- 更新后的 `V`
- 返回是否成功

## 8.4 `IPCSolver::SearchDirection(...)`

输入：

- 当前 `X`
- 求解参数

输出：

- `p`

实现：

- `energy.GradientHessian(...)`
- `DOFElimination(...)`
- `currLinearSolver->Solve(...)`

## 8.5 `CollisionDetection::PrepareRenderData()`

输入：

- 当前 queries
- 当前 `X`
- 当前 `p`

输出：

- 用于 GUI/BVH/query 可视化的缓存

## 8.6 `CollisionDetection::GetSQDisplay(...)`

输入：

- query id
- `X`

输出：

- 单个 query 的几何可视化

作用：

- 非求解核心，但很适合调试 IPC query 分类和法向

## 9. 输入输出视角

### 9.1 输入

- `context.json`
- soft body mesh / tet / face
- 材料参数：`mass`、`mu`、`lambda`
- DBC
- IPC 参数：`dhat`、`kappa`、`tol`

### 9.2 运行时状态

- `X / XTilde / X0`
- `V`
- `gradient / hessian / p`
- fixed mask
- collision queries / normals / toi

### 9.3 输出

- 更新后的顶点位置和速度
- GUI/BVH/query 调试信息
- solver 性能统计

## 10. 实现特征总结

- 优点：框架和求解器边界清楚，适合学习“如何把 IPC 接进一个 CUDA 软体框架”。
- 局限：教学项目属性明显，很多工程细节为演示服务，不是最极致的工业实现。
- 最值得学的部分：
  - double/float 双路径如何共存
  - `SolverData` / `SolverParams` / `CollisionDetection` 三者如何解耦
  - IPC 线搜索如何和 query-based CCD 模块配合
