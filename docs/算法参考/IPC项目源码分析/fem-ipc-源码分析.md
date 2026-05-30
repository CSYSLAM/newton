# fem-ipc 源码分析

项目路径：`C:\csy_work\CG\Engine\IPC\fem-ipc`

关键源码：

- `fem-ipc/simulator/simulator.h`
- `fem-ipc/simulator/simulator.cpp`
- `fem-ipc/energy/barrier_energy.cpp`
- `fem-ipc/geometry/*`

## 1. 项目定位

`fem-ipc` 是一个非常适合读源码的简化版 IPC：

- 只聚焦 FEM 软体
- 没有大规模工程框架
- 碰撞集直接暴力枚举
- Newton + LDLT 非常直接

它的价值不是性能，而是“把 IPC 最小系统写得很直白”。

## 2. 项目架构

```text
apps/main.cpp
-> Simulator
   -> initialize()
   -> step()
      -> buildCollisions()
      -> totalEnergyGradient()
      -> totalEnergyHessian()
      -> LDLT solve
      -> computeStepSize()
      -> line search
      -> update velocity
```

核心模块：

| 模块 | 作用 |
| --- | --- |
| `TetMesh` | 四面体网格与表面/边界提取 |
| `StableNeoHookeanEnergy` | 体弹性能 |
| `BarrierEnergy` | 碰撞 barrier、梯度、Hessian、ACC D |
| `Inertia / GravityPotential` | 惯性和重力 |
| `MovingDBCEnergy` | 移动边界能量 |
| `Simulator` | 主时间步进器 |

## 3. 算法模型

总能量在 `Simulator::totalEnergyValue()` 中非常清楚：

```text
E
= E_inertia
+ dt^2 * E_gravity
+ dt^2 * E_barrier
+ E_moving_dbc
+ dt^2 * E_stable_neo_hookean
```

这里的结构非常接近标准隐式 Euler IPC。

## 4. 数据链路

### 4.1 初始化链路

```text
TetMesh
-> computeMassAndRestVolume()
-> 创建 GravityPotential / Inertia / StableNeoHookeanEnergy / BarrierEnergy / MovingDBCEnergy
-> velocity = 0
-> 初始化 ceiling 位置和运动目标
```

### 4.2 单步链路

`Simulator::step()` 的数据链非常短：

```text
x_tilde = V + dt * velocity
-> buildCollisions()
-> while true:
   dir   = computeSearchDirection(x_tilde)
   alpha = barrier->computeStepSize(...)
   line search on alpha
   update mesh.V
-> velocity = (V - x_n) / dt
```

## 5. 碰撞流程

### 5.1 候选构造

`BarrierEnergy::buildCollisions()` 做了三类事：

- `vertex-face`
- `edge-edge`
- `vertex-floor` / `vertex-ceil`

而且是双重循环暴力枚举，没有空间哈希或 BVH。

这也是 README 里明确提到的性能瓶颈。

### 5.2 距离模型

所用几何函数来自：

- `geometry/distance.h`
- `geometry/vertex_face_collision.h`
- `geometry/edge_edge_collision.h`
- `geometry/accd.h`

支持：

- 点三角距离与导数
- 边边距离、mollifier 与导数
- ACCD 步长求解

### 5.3 barrier 形式

`barrier_energy.cpp` 中 barrier 和导数就是标准 IPC 对数形式：

```text
b(d)   = -(d-dhat)^2 log(d/dhat)
b'(d)  = (dhat-d)(2log(d/dhat) - dhat/d + 1)
b''(d) = (dhat/d + 2)(dhat/d) - 2log(d/dhat) - 3
```

厚度处理则使用：

```text
barrier_sq(dist_squared, dhat, dmin)
```

其中 `dmin` 会被折算进激活阈值。

## 6. 仿真流程

### 6.1 `Simulator::step()`

流程可以直接照源码写成伪代码：

```text
x_tilde = x + dt * v
x_n = x
update moving ceiling position

buildCollisions(x)

while true:
    dir = solve(H(x), -g(x))

    if ||dir||_inf <= tol and DBC not satisfied:
        increase movingDBC stiffness

    E_last = totalEnergyValue(x_tilde)
    alpha  = barrier.computeStepSize(x, dir)

    while true:
        x_trial = x_0 + alpha * dir
        E_new = totalEnergyValue(x_tilde)
        if E_new > E_last:
            alpha /= 2
        else:
            break

    if ||dir||_inf < tol and DBC satisfied:
        break

v = (x - x_n) / dt
```

### 6.2 搜索方向

`computeSearchDirection()` 非常简单：

1. `totalEnergyHessian(x_tilde)`
2. `totalEnergyGradient(x_tilde)`
3. `Eigen::SimplicialLDLT`
4. 解 `dir = -H^{-1} g`

### 6.3 Hessian 组装

Hessian 直接由 triplets 构造：

- inertia
- barrier
- moving DBC
- 每个 tet 的 stable neo-hookean Hessian

没有分块系统，也没有接触活动集管理器。

## 7. 关键函数输入输出

## 7.1 `Simulator::initialize()`

输入：

- `mesh`

输出：

- `mesh.masses`
- `gravityPotential`
- `inertia`
- `neoHookean`
- `barrier`
- `movingDBC`
- `velocity`
- ceiling 初始状态

## 7.2 `Simulator::step()`

输入：

- 当前 `mesh.V`
- `velocity`
- `dt`

输出：

- 更新后的 `mesh.V`
- 更新后的 `velocity`

## 7.3 `Simulator::computeSearchDirection(x_tilde)`

输入：

- `x_tilde`
- 当前 `mesh.V`

输出：

- 一维向量 `dir`

实现：

- 组装梯度和 Hessian
- 直接 LDLT 分解求解

## 7.4 `BarrierEnergy::buildCollisions(...)`

输入：

- 顶点、面、边、`dhat`
- floor / ceil 位置

输出：

- `vertexFaceCollisions`
- `edgeCollisions`
- `floorCollisions`
- `DBC_Collisions`

## 7.5 `BarrierEnergy::computeStepSize(...)`

输入：

- 当前位置 `V`
- 搜索方向 `searchDir`
- floor / ceil 高度

输出：

- `toi`

实现：

- 先检查 floor/ceil
- 再对所有 `VF/EE` 候选做 ACCD
- 返回最小可行时间

## 7.6 `BarrierEnergy::gradient(...) / hessian(...)`

输入：

- 当前 `V`
- 已构好的 collision list

输出：

- 总接触梯度 / Hessian

实现：

- `EE` 使用距离与 mollifier 的链式法则
- `VF` 使用点三角距离导数
- floor/ceil 视为一维 barrier

## 8. 输入输出视角

### 8.1 输入

- 四面体网格
- `E`、`nu`
- `dt`、`tol`
- `dhat`、`kappa`
- floor / moving ceiling

### 8.2 运行时状态

- `mesh.V`
- `velocity`
- `x_tilde`
- 4 类 collision list

### 8.3 输出

- 新位置
- 新速度
- 接触能量、梯度、Hessian 隐式写入求解流程

## 9. 实现特征总结

- 优点：最容易读懂的“小 IPC”之一。
- 局限：碰撞集暴力枚举，几何规模一大就很慢。
- 最值得学的部分：
  - barrier 的值/梯度/Hessian 如何直接写成一个最小可运行系统
  - ACCD 如何和回溯线搜索配合
  - 总能量装配在简化系统中的写法
