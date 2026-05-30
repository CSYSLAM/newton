# PD-IPC-ArmadilloDemo 源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\PD-IPC-ArmadilloDemo`

关键源码：

- `Simulator/PD-IPC/PdIpcSimulator.h`
- `Simulator/PD-IPC/PdIpcSimulator.cpp`
- `Simulator/PD-IPC/PdIpcGpuFunc.cuh`
- `Simulator/Bvhs/AipcBvhs.*`

## 1. 项目定位

这是一个典型的“PD + IPC 混合工程实现”：

- 约束部分走 projective / local-global
- 碰撞部分走 patch BVH + CCD + 接触刚度项
- 求解内核大量放在 CUDA host wrapper + kernel 中

它不是论文式纯粹 IPC，而是把接触插入到 PD 迭代中。

## 2. 项目架构

```text
PdIpcSimulator
-> models (volumetric / cloth / static models)
-> flatten geometry and constraints
-> gpuGeneralInit()
-> gpuSolverInit()
-> gpuCollisionInit()
-> doTime()
   -> advance()
      -> predictiveHost()
      -> LocalGlobalIteration()
      -> gpuCcdCulling()
      -> getImpactOfTimeHost()
      -> clampDirectionHost()
      -> setCollisionConstraintsHost()
      -> updateVelocityHost()
```

模块层级：

| 模块 | 作用 |
| --- | --- |
| `PdIpcModel/*` | 各模型把本地几何/约束展平到全局数组 |
| `PdIpcSimulator` | 全局状态、主循环、GPU 资源 |
| `PdIpcGpuFunc.*` | CUDA host API 与 kernels |
| `AipcBvhs` | patch 级 BVH |
| `TetConstraint` | 体约束辅助 |

## 3. 算法模型

从 `ConstraintEnergyInfo` 可以确认本项目的总能量构成至少包含：

- `inertialEnergy`
- `volumetricStrainEnergy`
- `clothStrainEnergy`
- `clothBendingEnergy`
- `positionEnergy`
- `collisionEnergy`

这说明它把 PD 常见的局部约束能和 IPC/接触能混在一个 outer iteration 中迭代处理。

## 4. 数据链路

### 4.1 初始化链路

`PdIpcSimulator::initialization()` 会做这些事：

1. 遍历动态模型：
   - volumetric model
   - cloth model
2. 收集：
   - `pointsList`
   - `volumetricElementList`
   - 质量/逆质量
   - 边界点/边/面
   - cloth 邻接和 bending/strain 数据
   - position constraints
3. 遍历静态模型追加边界 primitive
4. flatten 成统一矩阵/数组：
   - `points`
   - `simElements`
   - `boundaryPoints`
   - `boundaryEdges`
   - `boundaryFaces`
5. 构造重力、速度、稀疏 Hessian 结构
6. 调用：
   - `gpuGeneralInit()`
   - `gpuSolverInit()`
   - `gpuCollisionInit()`

### 4.2 运行时状态

头文件里能看到非常完整的状态展开：

- `devPoints / devOldPoints / devDirection`
- `devVelocity / devNextVelocity`
- `devInitRhs / devSysRhs / devCollisionRhs`
- `devSysDiagonal / devElasticsDiagonal / devCollisionDiagonal`
- `devBoundaryFaces / devBoundaryEdges`
- patch/BVH/碰撞对列表

这意味着项目把“几何状态”、“线性系统状态”和“碰撞状态”分别维护在独立的 GPU 数组上。

## 5. 碰撞流程

### 5.1 patch BVH

从 `gpuCollisionInit()`、`gpuCcdCulling()` 和 `PdIpcGpuFunc.cuh` 可以看出宽相流程是：

```text
boundary faces
-> collision patches
-> patch bbox
-> BVH node bbox
-> patch-patch overlap pairs
-> potential CCD primitive pairs list
```

### 5.2 `gpuCcdCulling(...)`

这一步按源码顺序执行：

1. `updateCcdPatchBboxesHost`
2. `updateBvhsBboxesHost`
3. `findPotentialCollisionsHost`
4. `getCcdPotentialICPairsListHost`

可以理解为：

- 先把每个 patch 的 swept AABB 算出来
- 再把 patch 装进 BVH
- 再查询 patch-patch 重叠
- 最后展开成真实 primitive 级碰撞对

### 5.3 TOI

`advance()` 中：

- 第一次 outer iteration 后调用 `gpuCcdCulling(devOldPoints, devDirection)`
- 然后 `getImpactOfTimeHost(...)`
- 再 `clampDirectionHost(...)`

这说明它的步长控制是：

```text
先得到无碰撞方向
-> 做 CCD 估计 TOI
-> 把方向按 TOI 裁剪
-> 再基于裁剪后的状态构造碰撞约束
```

### 5.4 接触刚度项

`setCollisionConstraintsHost(...)` 会往：

- `devCollisionDiagonal`
- `devCollisionRhs`

写入碰撞贡献。

这表明接触没有单独线性求解，而是被折叠进下一轮 local-global 的系统右端和对角项中。

## 6. 仿真流程

## 6.1 `doTime(frame)`

目前实现很直接：

- `subStep = 1`
- `dt = timeStep / subStep`
- 调 `advance(frame, dt)`

因此现版本的“substep 框架”已经留好，但默认只跑 1 次。

## 6.2 `advance(frame, dt)`

源码主线：

1. 保存 `devX0` 和 `devOldPoints`
2. `predictiveHost()` 计算预测状态和惯性项
3. outer iteration 最多 30 次
4. 每次 outer iteration：
   - `LocalGlobalIteration(frame, dt)`
   - `computeDirectionHost()`
   - 首轮做 `gpuCcdCulling()`
   - `getImpactOfTimeHost()`
   - `clampDirectionHost()`
   - 清空碰撞对角和 RHS
   - `setCollisionConstraintsHost()`
   - 判断方向残差
   - 更新 `devOldPoints`
5. `updateVelocityHost()`

## 6.3 `LocalGlobalIteration(frame, dt)`

这一步做的是 PD 风格 local-global：

1. 系统对角 = 弹性对角 + 惯性对角 + 可选碰撞对角
2. `updateSuperJacobiOperatorsHost(...)`
3. 做 10 次 local-global：
   - `doLocalProjection(...)`
   - `chebyshevJacobiR2SolverHost(...)`

### 6.4 `doLocalProjection(...)`

局部阶段会把不同约束贡献依次写进 `devSysRhs`：

- volumetric strain
- cloth strain + bending
- collision rhs
- position constraints

所以它很像：

```text
local project all constraints
-> accumulate target rhs
-> global linear solve
```

## 7. 主伪代码

```text
initialize():
    flatten models and constraints
    allocate GPU arrays
    build solver structures
    build collision patches/BVH support

advance(dt):
    X0 = X
    Xold = X
    predictiveHost()  // free motion

    for outer_iter in [0, max_outer):
        LocalGlobalIteration()
        direction = X - Xold

        if first outer iter:
            gpuCcdCulling(Xold, direction)

        toi = getImpactOfTime()
        clampDirection(toi)

        clear collision diagonal/rhs
        setCollisionConstraints()

        if ||direction||^2 small and outer_iter > 0:
            break

        Xold = X

    updateVelocity()
```

## 8. 关键函数输入输出

## 8.1 `PdIpcSimulator::initialization()`

输入：

- 模型列表
- 静态模型列表

输出：

- 全局点、单元、边界 primitive
- GPU 缓冲区
- 稀疏系统结构

## 8.2 `PdIpcSimulator::LocalGlobalIteration(frame, dt)`

输入：

- 当前 `devPoints`
- 各种刚度与系统对角
- 是否存在 active collision list

输出：

- 更新后的 `devPoints`

## 8.3 `PdIpcSimulator::doLocalProjection(frame, dt, energy)`

输入：

- 当前几何
- 体/布/位置/碰撞约束数据

输出：

- `devSysRhs`
- 若干分项能量

## 8.4 `predictiveHost(...)`

输入：

- `X`
- `V`
- `dt`
- 重力、鼠标力、罚力、逆质量

输出：

- 预测位置
- 初始 RHS
- 惯性对角

## 8.5 `gpuCcdCulling(devX, devDir)`

输入：

- 当前点位置
- 当前方向

输出：

- `devPotentialCcdICPairsList`
- `hostPotentialCcdICPairsNum`

## 8.6 `setCollisionConstraintsHost(...)`

输入：

- `dHat`
- `kappa`
- `dt`
- 潜在 CCD 对
- 当前点坐标、质量

输出：

- `devCollisionDiagonal`
- `devCollisionRhs`

## 8.7 `updateVelocityHost(...)`

输入：

- `X`
- `X0`
- `dt`

输出：

- 新速度 `V`

## 9. 输入输出视角

### 9.1 输入

- volumetric model
- cloth model
- static model
- 位置约束
- `dHat`、`kappa`、阻尼、时间步

### 9.2 运行时状态

- 全局点位置/速度
- strain / bending / position / collision rhs
- patch 级 BVH 数据
- potential CCD pair 列表

### 9.3 输出

- 更新后的 cloth/tet 顶点
- 对齐后的渲染 buffer
- 统计计数：Jacobi、LG、outer iter、碰撞数

## 10. 实现特征总结

- 优点：非常典型地展示了“PD 约束投影 + IPC 风格碰撞裁剪/约束”的混合思路。
- 局限：状态数组非常多，工程风格偏 demo，需要边读边画数据图。
- 最值得学的部分：
  - 如何把碰撞项写入 local-global 系统
  - patch BVH 如何服务于 cloth/soft-body 的 CCD
  - 为什么 hybrid solver 在工程上常比纯牛顿更容易稳定落地
