# XPBD FEM + DAT：第一版实现与验收记录

初始日期：2026-09-06；整理日期：2026-09-07。FEM 是 `SolverXPBD` 的显式实验分支，默认关闭。
本次未修改 VBD / MJVBD_V2。**算法实测见第 11 节，拉取远程后的集成验证见第 12 节**：demo 默认使用全局 XPBD，
10 substeps × 2 非线性轮 × 5 PCG；本轮保留等价算子缓存，未保留 multigrid 实验。
整体形变接近 VBD，但逐点轨迹精度未等同；不能把线性残差改善称为完整物理精度达标。
第 1–10 节记录历史配置和实验，不代表当前 demo 默认配置或性能。

修改区清理：实验输出 `outputs/xpbd_physx_alignment/` 的 300 个文件已移入回收站，
不纳入代码；关键结果与 No-Go 原因保留在本记录。历史输出标签不再指向工作区现存文件。
保留 `compare_cloth_twist_solvers.py` 用于性能/形变复测，
`analyze_cloth_twist_results.py` 用于独立最近特征和逐点误差诊断；删除重复的 benchmark 脚本。
复测命令见第 11 节；先运行对照脚本生成新的 NPZ，再运行诊断。

## 1. 范围与 PhysX 对齐程度

参考本地 PhysX 5.6.1，版本 `5.6.1.51c1f783`，仓库 HEAD `ed6e5ca24`。
采用其 **FEM deformable surface 的 TGS 路径**，不是旧 NvCloth 距离弹簧模型。

| PhysX 来源 | 本实现 | 已对齐的内容 |
| --- | --- | --- |
| `FEMClothUtil.cuh:437` / `membraneEnergySolvePerTriangle` | `fem_kernels.membrane_update` | 局部二维坐标，极分解 ARAP，更新后重算面积约束 |
| `FEMClothUtil.cuh:543` / `bendingEnergySolvePerTrianglePair` | `fem_kernels.solve_bending` | signed dihedral，角度 wrap，±π/2 clamp，XPBD compliance |
| `PxgFEMCloth.cpp:190,421` | `fem_cooking.categorize` | 降序无向边 key 贪心分类；每个膜三角形只求解一次，每个 hinge 一次 |
| `PxgFEMClothCore.cpp:2550–3030` | `cook_copy_chains` / `PairBatch` | first-free coloring、最多 8 组合分区、前向 copy/remap、末端引用数 CSR |
| `FEMClothUtil.cuh:662` | `solve_pair_partition` / `average_pair_copies` | shared pair 内两次膜约束再弯曲；有效逆质量乘副本数；末端相对位移平均 |
| `PxgFEMClothCore.cpp:1335,1360,1687` | `FEMSurface.step` / `_solve_shell` | 布料 TGS 顺序、非共享单三角→共享 pair→非共享 pair、接触外部增量延迟应用 |
| `FEMCloth.cu:947,1046` | `membrane_damping` / `bending_damping` | 每个 TGS slice 后独立速度阻尼；去除平动/刚性转动分量，原子求和 |

**已补齐所选纯布料 TGS 分支的核心材料、分区与阶段流程，不是完整 PhysX 后端复刻。**
本轮实现了此前遗漏的 pair/copy/remap/reference-count。Warp 使用每分区一次 launch，
没有移植 CUDA 单 block 尾部分区融合；没有移植 PhysX BV32、PGS multiplier 路径、
tet、attachment 或刚柔耦合。接触按任务要求替换为 VT/EE + DAT，并非 PhysX 原接触内核。
没有运行原生 PhysX 同场景逐步对照；方程和调度测试通过不代表全 SDK 轨迹一致。
copy storage 使用独立 validity bit，避免用 inverse mass=0 同时表示固定点和未初始化。

第一版只支持纯三角布料。刚体、shape、tet、spring、requires-grad、deterministic
模式显式拒绝，不能借本 demo 宣称已完成 PhysX soft body 或刚柔耦合。
原 XPBD 路径没有被这些限制收窄，不设置 `particle_fem=True` 就保持原行为。

## 2. 材料和时间语义

设三角形 rest area 为 A，厚度为 h_s，时间片长度为 h：

- `tri_ke = μ h_s`，ARAP compliance `1 / (2 μ h_s A h²)`。
- `tri_ka = λ h_s`，面积 compliance `1 / (λ h_s A h²)`。
- `edge_ke = 1 / kInv`，弯曲 compliance `kInv / h²`。
- 采用 PhysX TGS 的零 elastic multiplier 分支，每个时间片一轮约束。
- 此模式的 `iterations` 是每次 `step(dt)` 内的 temporal slices 数量，
  **不是同一 dt 下重复普通 XPBD sweep**。默认 demo：10 个外部 substeps × 12 slices。
- `tri_kd`、`edge_kd` 是 PhysX 独立材料速度阻尼率 [1/s]，因子 `clamp(kd*h,0,1)`。
  不能照搬 VBD 的阻尼数值。旧记录称 PhysX 没有 material damping 是错误的：
  它不在弹性方程函数内，而在后面的 `solve_velocity → applyDamping` 阶段。
  aerodynamic drag/lift 仍显式拒绝。
- 本分支接触响应使用 `soft_contact_ke`、`soft_contact_mu`，不使用 `soft_contact_kd`、
  restitution、控制激活或 legacy relaxation 系数。尝试加入官方法向势垒及阻尼的实验
  没有通过质量验收，已撤出；不能宣称已经完整对齐 VBD 接触能量。

## 3. 管线与函数边界

```text
SolverXPBD.step
  FEMSurface.step
    copy input positions / velocities
    _detect: 固定本次检测基点 X
      primitive_bounds
      triangle / edge LBVH rebuild（外部调度 rebuild 后，此处改为 refit）
      detect_vt / detect_ee → active prefix + device count
    repeat iterations times, h = dt / iterations
      save slice previous positions
      predict → _commit(DAT)
      contact_response(predicted state) → 暂存外部增量
      singles → shared pairs → nonshared pairs → _commit(DAT)
      apply_contact_delta(暂存增量) → _commit(DAT)
      update_velocity(accepted positions, slice previous positions, h)
      membrane/bending damping → atomic sum → apply velocity delta
    copy accepted state to output
```

弹性 workspace 是 proposal，不是已接受状态；完成内部 sweep 后统一 DAT 提交。
预测、接触和固定边界动画都不能绕过提交。这不是在原 XPBD 最后随意夹一次位置。
`step()` 内不分配数组、不读取 GPU count 到 CPU；固定容量、固定 worker 数和
原位 BVH rebuild 可录入 CUDA Graph。CPU 可执行同一实现。

## 4. Broad phase / narrow phase

- 三角形和所有唯一表面边各有一棵 BVH，包含边界边。
- 当前实现使用 Warp LBVH，不声称移植了 PhysX BV32 CUDA 内核。
- 未调用 `solver.rebuild_bvh(state)` 时，每次 step 原位 rebuild。
  调用该方法后，由调用者安排后续 rebuild，step 仍更新 bounds、refit 并完整检测。
  demo 每帧 rebuild 一次、10 个 substeps 分别 refit/detect；没有跳过 substep 碰撞检测。
- VT 排除 incident vertex；EE 排除共享端点，且只输出 `edge_a < edge_b`。
- 限定同一 world。不使用 rest-near exclusion，因此不会因距离过滤删掉安全候选。
- 紧凑 contact stream 保存四顶点、种类、normal、四个基点 signed distance、gap。
- float 最近点提供候选轴，计算两个 primitive 的 support interval。
  **只有 support gap（距离下界）大于 query radius 加舍入余量才剔除**，
  不用可能不准确的最近点距离上界剔除。
- 不能可靠分隔或 gap 很小时，使用 double 最近特征重算，覆盖近并行/退化边界候选。
  双精度后仍不能分隔则报错并停止提交，不给一个任意法向继续穿过去。
- float 快路径可能保留半径外的保守额外候选；它们不会自动产生接触力。
  响应阶段使用当前几何重新求最近点，距离小于物理 radius 才响应。
- 默认容量 `128 × particle_count`。device-side 原子计数，超过容量置 sticky overflow，
  所有线程都以 `min(count, capacity)` 访问，失败后停止提交位置。

FP64 refinement + FP32 support cache 是实测后的选择，不是严格的精确算术谓词库。
舍入保护是工程数值余量，不能包装成任意尺度、任意退化输入的形式化证明。

## 5. DAT 与接触响应

DAT 参考 [Divide and Truncate 原论文](https://arxiv.org/html/2604.15513v1)
以及 Newton 官方 `upstream/main` 的 `particle_vbd_kernels.py`：
参考 revision `6d2ece7c314248960a249ebc46b0dafbb5a89232`，不是 MJVBD_V2 私有实现。

- 按两侧最大 approaching displacement 计算 division ratio，限制在 `[0.05, 0.95]`。
- 检测基点、normal 和 signed distance 可以缓存；division ratio 和每顶点 t 每次重算。
- 每个 pair 对四个顶点做 atomic min，适用于一次 FEM proposal 同时改变多个顶点。
- 位移总预算仍相对 detection base X，不能每次提交清零预算。
  每次截断的是 `candidate - accepted`，不再重复缩小 `accepted - X`。
  分隔平面必须同时分隔 X 和已接受状态，其可行区间由两者 support interval 的交集给出。
  在交集中按 division ratio 选平面，再对本次增量求 t。交集耗尽则拒绝该增量。
  最后的累计球形范数限制朝 X 收缩；因为 X 也在本轮所有半空间内，这一步保持可行。
  此修正避免重新计算 ratio 时撤销已接受位置；无增量的提交保持原位置。
- `gamma = 0.85`，累计位移范数上限 `0.5 × gamma × query_radius`。
- 即使候选未被检测到，两 primitive 的最大接近量也小于 query radius。
- 另外预留浮点位置写回余量，避免很小间隙下的舍入把两侧合并。
- 固定边界通过 `particle_kinematic_targets` 输入，也受 DAT 限制。
  不允许用直接改 state 的瞬移来冒充安全运动。

接触法向仍采用 compliant quadratic distance constraint，只在 `0 < d < radius` 启用响应。
切向根据当前时间片滑移应用正则化 Coulomb 摩擦；法向和切向使用当前最近点重算。
这是对 VBD 近接触距离能量/摩擦思路的 XPBD 改写，**不是完整 OGC、ALM 或 VBD Hessian 求解**。
接触 proposal 使用逐顶点 Jacobi 平均；多接触数不均匀时不严格保持动量。
这不是官方 VBD 的完整二次/对数/二次延拓势垒。补齐势垒的 Newton 和闭式 proximal
实验已测，但前者形变质量下降，后者完整场景复跑失败，因此都没有混入最终实现。
密集接触收敛与数值间隙问题仍未解决，见第 9 节。

DAT 保证依赖初始无交叉、候选覆盖和正确数值分隔等前提。当前不包含独立的
tet/triangle inversion constraint；不能宣称自动修复已有自交或所有拓扑退化。

## 6. 用户接口和 demo

```python
solver = newton.solvers.SolverXPBD(
    model,
    particle_fem=True,
    iterations=12,
    particle_self_contact_radius=0.002,
    particle_self_contact_margin=0.0035,
)
```

容量不足可显式设置 `particle_self_contact_max`。失败状态是 sticky，不自动丢失候选。
调用 `solver.validate_particle_contacts()` 会同步检查状态并独立查询 edge–triangle
交叉（包括共面重叠）。该审计不使用接触 rows，也不是 CCD；拓扑 incident pair 被排除。
它会有 CPU 同步开销，因此在帧外验收，不在 timed graph 内调用。

```powershell
uv run --extra examples -m newton.examples cloth_twist_xpbd
uv run --extra examples -m newton.examples cloth_twist_xpbd --viewer null --num-frames 600
uv run --extra examples -m newton.examples cloth_twist_xpbd --no-self-contact
```

独立例子 `newton/examples/cloth/example_cloth_twist_xpbd.py` 使用原 cloth_twist 的
2500 顶点 / 4802 三角形网格、相反边界旋转、60 FPS 和 10 秒动画。
无 VBD solver、无隐藏碰撞代理。边界计时单独 kernel，避免同一 kernel 读写时间的 race。
默认每 60 帧审计；benchmark 每帧审计。验收还要求边界误差 ≤ 20 μm，不能通过
冻结边界来获得无交叉结果。README 截图为本例真实第 600 帧，320×320 JPEG。

## 7. 初版历史记录（以下不是本轮默认版本的性能）

环境：RTX 5060 Ti 8 GiB，Warp `1.17.0.dev20260807`，CUDA 12.9 / driver 13.3。
完整 600 帧、固定 graph，前 10 帧不计时；CUDA event 测 GPU，wall 等待 event 完成。
不包含初始化、编译、渲染、独立几何审计。以下是不同实现运行的实测，不是等轨迹精度比较。

| 实现 | GPU ms/frame | wall ms/frame | 结果 |
| --- | ---: | ---: | --- |
| 最初全 float 最近点/分隔 | — | — | 约第 103 帧出现 4 处交叉，No-Go |
| double detector + double DAT，refit | 68.087 | 68.184 | 600 帧审计通过，开销过高 |
| double detector + float support cache，refit | 62.741 | 62.829 | 通过 |
| double detector + support cache，rebuild | 56.609 | 56.703 | 通过 |
| 混合精度检测 + support cache + rebuild | 19.490 | 19.570 | 600 帧逐帧审计通过，status=0 |
| 同一 FEM、关闭接触和 DAT | 4.691 | 4.765 | 仅测弹性 baseline，不承诺无自交 |

最后一轮混合精度运行：第 600 帧 active pairs 为 28737，最后 100 帧平均 GPU
24.044 ms/frame；全程边界误差打印为 0.000000 m。不是所有阶段都稳定 60 FPS。
主要剩余开销在自接触，不把 4.691 ms 的无接触结果当作可用成品性能。

测试命令：

```powershell
uv run --extra dev -m unittest newton.tests.test_xpbd_fem
uv run --extra dev -m unittest newton.tests.test_solver_xpbd newton.tests.test_solver_xpbd_tetrahedra
```

- 新增 9 项测试通过：NumPy SVD membrane 公式、有限差分弯曲梯度、DAT 累计预算/
  切向运动、overflow 拒绝、分区/legacy gate、物理时间与 graph replay、独立交叉/
  共面审计、6 VT + 9 EE 完整覆盖/近重合 graph、安全范围内的摩擦响应。
- 原 XPBD 66 项刚体/关节/tet 等回归测试通过。
- Ruff 和 `git diff --check` 通过；没有执行整个仓库全部测试。

这证明初版 cloth 场景可运行，但不能证明形变效果良好或完整 PhysX 算法迁移。
本轮修正与新增测试见下节，不能把本节 19.49 ms 当成本轮默认版本耗时。

## 8. 本轮问题修正与纯 VBD 对照

对照使用未修改的 `newton/examples/cloth/example_cloth_twist.py`，不是 MJVBD_V2。
同一网格、密度、锚点旋转、10 外部 substeps、600 帧；VBD 原例是 4 普通迭代。
原 VBD 使用 stable Neo-Hookean，而 PhysX cloth 使用 corotational ARAP + 面积约束。
因此相同 Lamé 数值只是在小应变下有可比意义，不是大变形相同本构的精度对照。

### 实际问题

1. 上版只做独立 triangle / hinge GS，遗漏 PhysX 的 shared pair、copy/remap、有效质量和平均。
2. 错把 soft-body 阶段顺序当成 cloth 顺序，且遗漏了独立速度阻尼。
3. 原 VBD hinge 能量系数为 `edge_ke * edge_rest_length`。FEM 输入的是 cooked 系数；
   demo 直接复制 `0.001` 会把弯曲硬度放大约几十倍。现在 demo 显式乘 rest length，
   **求解器本身不隐式乘长度**，保留 PhysX cooked hinge 语义。
4. DAT 重新分配平面后再次截断累计位移，会拉回已接受位置，放大速度抖动。
   已改成上述公共可行平面 + 增量截断，并新增零增量幂等回归。

### 材料换算

PhysX `youngs=E, poissons=nu, thickness=t` 对应：

```text
mu = E / (2*(1+nu))
lambda = E*nu / ((1+nu)*(1-2*nu))
tri_ke = mu*t; tri_ka = lambda*t
Ldual = 2*(A0+A1)/(3*Lhinge)
edge_ke = bendingStiffness * t^3 * Lhinge/Ldual
```

上式用于正常可压缩材料 `0 <= nu < 0.5`；输入层是 cooked Lamé / hinge 系数，
不是完整 PhysX material API。不同相邻材料时 PhysX 对 thickness、bendingStiffness、
bendingDamping 分别取两侧平均后 cooking。当前 demo 为均匀材料。

### 实测

GPU/wall 定义和环境同第 7 节。表中的面积指标是
`sqrt(mean((A/Arest - 1)^2))`，**不是相对参考解误差或收敛残差**。
contact 原子累加非确定性，不承诺不同运行的逐点轨迹完全一致。

| 版本 | GPU ms/frame | wall ms/frame | 第 600 帧面积变化 RMS | 备注 |
| --- | ---: | ---: | ---: | --- |
| 初版重新测量 | 19.260 | 19.336 | 0.2668 | 未补齐 cooking/damping；旧 DAT |
| 原始纯 VBD | 25.902 | 26.085 | 0.1048 | 原例 4 iterations，不改参数 |
| 修正后 4 slices | 23.492 | 23.561 | 0.2357 | 匹配 hinge，阻尼 600/s，增量 DAT |
| 修正后 12 slices | 40.533 | 40.620 | 0.1392 | 同上，整体形变改善但更慢 |
| 8 slices，膜刚度 ×4 | 31.809 | 31.882 | 0.2349 | 未改善，No-Go，不进入默认 |

最终默认加入 DAT 平面区间耗尽保护后重新完整验收：GPU **40.760**、wall **40.835**
ms/frame；第 600 帧面积比范围 **0.392912–1.570925**。600 帧逐帧几何审计通过，
status=0，锚点误差打印为 0.000000 m。README 图片已用这一版实际最终帧更新。
12 slices 的效果明显优于旧版，但不是“与 VBD 同精度而更快”；不能做这一结论。

默认 demo 选 12 slices、阻尼 600/s 和正确的 hinge 换算，以效果优先。
`--iterations 4` 可测快速版本；`--damping 0` 可检查无阻尼基线。
600/s 是本场景调试值，不是 PhysX 官方默认值，也没有写死到 SolverXPBD 内。

### 复现与测试范围

```powershell
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver vbd --label vbd_original
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver xpbd --label xpbd12 --iterations 12 --match-vbd-hinge --damping 600 --bending-damping 600
uv run --extra dev -m unittest newton.tests.test_xpbd_fem newton.tests.test_solver_xpbd newton.tests.test_solver_xpbd_tetrahedra
```

对照脚本保存 JSON、NPZ 和每 100 帧真实截图到 `outputs/xpbd_physx_alignment/`，
VBD 分支禁止迭代/材料覆盖，避免误称“原例”。不把截图、审计、编译计入物理耗时。
旧的参数探索 JSON 没有记录全部参数，以上表和命令为准；新脚本补齐了参数元数据。
尚未对 VBD 全程运行同一独立几何审计，因此不宣称 VBD 全程无交叉。

新增测试共 14 项（覆盖 CPU/CUDA）：原 9 项加 DAT 零增量幂等、cooking 覆盖/
前向链不变量、共享 pair 非单位 rest frame 的 NumPy 顺序参考、超过 8 colors 的
质量缩放/复制/回写平均参考、阻尼保留刚体运动/固定点/动量检查。
14 项新增测试与 66 项原 XPBD 刚体/关节/tet 回归合计 **80 项通过**（54.591 秒）。
修改文件 Ruff 检查和格式检查通过，`git diff --check` 通过；对照脚本另做了 20 帧
输出/元数据 smoke test。未执行全仓库全部测试，未验收任意网格无自交，
未完成原生 PhysX 同场景二进制运行对照；这些不能被上述单测替代。

## 9. “快于 VBD 且形变质量基本一致”修复轮：未达标（历史 TGS 版本）

### 目标和对照口径

仍使用未修改的官方 `example_cloth_twist.py`：2500 顶点、4802 三角形，600 帧、
10 秒完整扭转，VBD 为 10 substeps × 4 iterations，XPBD 默认 10 × 12 temporal slices。
没有降网格分辨率、减转角、关闭自碰撞、放宽 overflow 或边界运动验收。
FEM 的 ARAP 与 VBD 的 stable Neo-Hookean 本构不同，相同 Lamé 数值不能直接称为相同精度。

最终帧报告以下四类指标，不能只看 FPS 或只看平均面积：

- 面积变化 RMS `sqrt(mean((A/Arest-1)^2))` 与主伸长变化 RMS。
- 面积最小/最大值、最大主伸长、最大速度，防止局部塌缩被平均数掩盖。
- 同编号粒子的最终位置 RMSE / p95。两者本构和接触离散不同，此指标是轨迹差异，
  不是对解析真值的误差；不能由它单独判定物理正确性。
- 每帧独立 edge–triangle 交叉审计、sticky status、驱动边界误差 <= 20 μm。
  候选中几何距离另用独立 NumPy float64 最近特征计算，不能把 cached support gap 当真实间隙。

GPU 与 wall 均为完整物理 Graph，前 10 帧不计，初始化、编译、截图和同步几何审计在计时外。
GPU benchmark 顺序执行；一次临时 `blocklinear12_kd01` 实验与 GPU 测试编译重叠，
其 35.48 ms 不能用于性能结论。逐阶段大量插入 event 的 profile 明显改变总耗时，
也不把该 profile 的阶段百分比当无扰动时间分布。

### 查明的问题

1. 默认 120 个小时间片，每片三次 DAT。一次显式 launch 统计为每帧约 3630 个 kernel、
   360 次 DAT 和 120 次接触响应。这里 XPBD 的标量约束便宜，并不代表整条自接触管线便宜。
2. 密集扭转时约五万条候选，参与顶点的接触计数可到数百。单对修正再按顶点接触数平均，
   削弱撑开布层的效果；不同顶点分别平均还不严格保持动量。
3. 原实现只有二次接触势，遗漏官方 VBD 的内层对数势及极小距离延拓。
   直接加入该势的梯度/曲率，单步 Newton 在很小间隙处仍给出很小的步长；
   单对闭式 proximal 虽通过独立双精度二分公式测试，整场景仍可能卡住边界。
4. 官方 VBD 默认 topological threshold=2，当前 FEM 仅排除 incident pair。
   试过仅过滤一环响应、保留全部 DAT 候选，仍没有解决局部塌缩，不将它设成隐藏默认过滤。
5. 最后帧实际最近特征距离存在纳米量级间隙。补齐势垒后的两次 Newton 实验中，
   `< 1 μm` 的候选对分别为 93、164，原 VBD 对照为 0；
   最小距离分别约 11.3 nm、8.9 nm，VBD 对照约 2.52 μm。
   这说明“没有查到交叉”不等于布层间距、折叠形态已经正确。

### 保留与撤出

保留的修改限于：

- `SolverXPBD.rebuild_bvh(state)`：显式外部 rebuild 调度；demo 每帧 rebuild，
  substep 仍完整更新 bounds/refit/detect。未使用者仍每 step rebuild。
- DAT 的 `atomic_min(1)` 无效写入跳过；`t=0` 真正 bitwise no-op；
  accepted 到 candidate 的端点插值及累计预算 clamp 使用 double 中间值、最后只转换一次。
- 比较脚本保存失败帧 NPZ；独立最近特征与轨迹差异诊断；可显式测试 substeps/margin。
  减少外部 substeps 的试验保持原例 `1/600 s` 目标时间滞后，不借少转一点获得更好结果。

材料、copy/remap、三阶段 DAT、默认 10 × 12 与阻尼 600/s 没有改成失败实验的配置。
**最终接触响应恢复原二次约束；这意味着完整 VBD 接触能量对齐仍是未完成项。**
日志记录势垒公式问题，不保留导致形变退步的势垒 Newton、闭式 proximal 或额外模式开关。

以下均为试验 No-Go，不是当前代码提供的模式：

| 实验 / 输出标签 | 结果 | 撤出原因 |
| --- | --- | --- |
| `barrier8_onecommit_debug` | 第 224 帧 3 处交叉 | 合并为一次 DAT 未通过数值安全审计 |
| `blocklinear12_area4` | 500 帧后边界失败 | 3×3 接触聚合与增面积刚度没有通过完整运动 |
| `scalarbarrier12_all` | 26.52 ms，但最大主伸长约 29.7 | 即使无交叉，形变严重错误 |
| `blockarap8` | 28.98 ms，面积 RMS 0.1498 | 未同时达标，且更改了 PhysX 标量 ARAP 更新 |
| `masssplit8` | 33.55 ms，面积 RMS 0.1667 | 接触副本质量改写未带来足够精度/性能 |
| `scalarprox8_responsefilter` | 32.70 ms，最小面积比约 0.0102 | 过滤局部响应仍有严重局部塌缩 |
| `exactprox12` | 单次 600 帧通过，39.42 ms | 闭式公式正确不代表整场景可靠 |
| `xpbd_repaired12` | 闭式接触修订版第 591 帧边界失败 | 撤出 proximal；不能只报上一次成功 |
| `exactprox_4x20` | 第 572 帧边界失败 | 4 外层检测、80 总时间片、margin 0.006 没有完成运动 |
| `final_newton12` / `_repeat` | 35.88 / 36.81 ms，位置 RMSE 68.5 / 62.1 mm | 势垒 Newton 比旧版约 52.7 mm 更差，撤出 |

Warp 条件 Graph 内尝试调用 LBVH rebuild 还遇到不支持 memory allocation 的报错。
该方案已删除，没有留下 CUDA 版本相关的运行时条件分支。

### 实测结果与判定

| 版本 | GPU ms/frame | wall ms/frame | 面积变化 RMS | 主伸长变化 RMS |
| --- | ---: | ---: | ---: | ---: |
| 本轮原 VBD 复测 `vbd_recheck` | 25.636 | 25.760 | 0.10522 | 0.17234 |
| 上轮 XPBD `xpbd_incremental12` | 40.533 | 40.620 | 0.13923 | 0.20097 |
| 保留原二次响应 + 本轮数值/BVH 修改 `retained_quadratic12` | 37.641 | 37.705 | 0.14166 | 0.20136 |

最后一行 600 帧逐帧审计通过，但最小面积比约 0.1997、最大约 1.9818，
不是局部质量已解决。整体形变指标与旧 XPBD 接近，仍明显偏离 VBD；
不声称这笔修改提高了精度，也不将约 7% 的时间下降包装为达成目标。
直接运行默认 demo 的第二次完整 600 帧验收为 GPU **36.562**、wall **36.627** ms/frame，
面积比范围 **0.398799–1.761582**；逐帧 status=0、没有检出非 incident 交叉、边界误差通过。
README 的 320×320 JPEG 已用这次真实最终帧更新。两次运行存在原子累加导致的轨迹差异，
因此不承诺局部最坏形变一致，也不把单次最小面积当稳健界限。
`retained_quadratic12` 相对本轮 VBD 的位置 RMSE **57.19 mm**、p95 **110.58 mm**，
其中 88 条候选真实距离 <1 μm，最小约 9.54 nm；仍是未达标的实验版。

**验收结论：No-Go。没有达到快于 VBD，更没有同时达到基本一致的形变质量。**
下一步需要改进接触与弹性的联合收敛、降低密集接触下的串行阶段/全局遍历次数，
而不是继续减少时间片、增刚度或放松 DAT。具体联合方法尚未验证，不承诺预期倍率。

复现与诊断：

```powershell
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver vbd --label vbd_recheck
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver xpbd --label retained_quadratic12 --iterations 12 --match-vbd-hinge --damping 600 --bending-damping 600
uv run --extra examples python scripts/analyze_cloth_twist_results.py outputs/xpbd_physx_alignment/retained_quadratic12.npz --reference outputs/xpbd_physx_alignment/vbd_recheck.npz
```

历史实验标签记录的是当时实现，不能仅凭同名 CLI 参数在当前已清理代码中重现。
当时的失败快照、JSON、NPZ 和真实截图保存在 `outputs/xpbd_physx_alignment/`；后续清理见本文开头。

清理实验后的 16 项 FEM 测试（CPU/CUDA）和原 XPBD 66 项测试合计 **82 项通过**。
本轮净新增两个回归：`t=0` 提交 bitwise no-op，外部 rebuild/refit 与每步 rebuild 的
VT/EE 完整 keys 一致。原先 84 项包含现已撤出的势垒/闭式求解实验测试，
不能将 84 项当最终代码的测试数量。未运行全仓库测试，未提交或推送这些改动。

## 10. 全局 XPBD 扩展：性能达标，整体形变接近，逐点轨迹仍有差异

本轮按用户允许其他 XPBD 优化方法的要求，新增 `particle_fem_solver="global"`。
它仍由 XPBD 的柔顺约束方程导出，不调用 VBD 求解器或任何 VBD 私有 kernel。
**它不是 PhysX 原流程的等价实现**。原 PhysX 方程、copy/remap 与 temporal slicing
保留在独立的 `"tgs"` 路径；`SolverXPBD` 的非 FEM 默认路径不变。
仅 `cloth_twist_xpbd` demo 默认改为全局版本。

### 为什么不再单纯减少 temporal slices

局部标量约束便宜，但密集接触需要大量串行传播与 DAT 遍历。
本轮改成每 substep 一次预测、两次非线性全局约束求解，各四次 PCG。
弹性、弯曲、法向接触和摩擦在同一线性系统中耦合，不能再按顶点接触数平均削弱反力。
每 substep 三次 DAT（预测 + 两次全局增量），不是每个局部约束都遍历候选流。
外层依然 10 substeps/frame，未降分辨率、减少旋转、关闭自接触或放宽验收。

参考方向包括 [MGPBD](https://arxiv.org/html/2505.13390v1) 的全局 XPBD 系统与
primal residual 问题；本实现**没有实现或移植该文的 AMG**。
材料对照来自仓库官方 `solvers/vbd/particle_vbd_kernels.py`，不来自 MJVBD_V2。

### 方程与实现

设 `F=[f0,f1]`，`a=f0·f0, b=f1·f1, c=f0·f1, J=|f0×f1|`，
`s=sqrt(a+b+2J)`。使用三个零静态值的膜约束：

```text
C0 = (a-b)/s
C1 = 2c/s
C2 = J-1
C0²+C1² = a+b-2J
E = A/2 * [mu*(C0²+C1²) + (mu+lambda)*C2²]
```

这与官方 VBD 的 stable Neo-Hookean 膜能量相差常数，不是此前的 ARAP 本构。
不能只用 `sqrt(a+b-2J)` 一个偏应变约束：其静态点退化，GN 近似会漏掉独立剪切方向。

消去全局柔顺约束 KKT 中的乘子增量，保留 DAT 后的 primal residual：

```text
(M + G^T alpha_tilde^-1 G) dx
    = -M*(x-xhat) - G^T alpha_tilde^-1 C
alpha_tilde = alpha/dt²
```

实际算子再加入法向几何刚度、冻结的隐式阻尼和接触切线块。
没有省略 `x-xhat`，也没有把 DAT 截断后的位置当成新的惯性目标。
`fem_global.py` 的 `build_elastic/build_contacts` 线性化约束；
`initialize_system` 聚合 RHS 和完整 3×3 block-Jacobi 预条件器；
`multiply_rows/gather_product` 实现矩阵自由乘积；PCG 固定轮数，无步内 CPU readback。
静态 CSR 在构造时生成；接触 active prefix、device count 与顶点 CSR 每次线性化更新。
容量固定，原 overflow/nonseparated/nonfinite 标志继续阻止提交。

膜的三个应变 Jacobian 都在面内，受拉时还必须补法向几何刚度：

```text
r = ((mu+lambda)*(J-1)-mu)/J
Hnormal = [[mu+r*b, -r*c], [-r*c, mu+r*a]]
```

取该 2×2 块的非负特征值，加入两个零约束值的切线行。
它们不增加弹性力或改变静态能量，只补几何曲率与相应阻尼；受压负曲率投影为零。
静态点该块为零。缺少该块的版本有约 0.82 m/s 动态速度峰值；补齐后约 0.49 m/s。
膜/弯曲阻尼为隐式刚度比例阻尼，单位是时间 [s]；与 TGS 的独立阻尼率 [1/s] 不通用。

接触使用官方 VBD 同形的二次外层/对数内层/极小距离二次延拓势；
闭合法向阻尼和 Coulomb 摩擦使用冻结切线近似，不宣称完整接触 Hessian 完全相同。
所有非 incident 候选响应保留，没有新增隐藏一环过滤。
候选由原保守 BVH 管线生成；两次非线性求解重新计算当前距离、权重和响应。

DAT 后检查真实弹性能、接触法向势、惯性项与冻结耗散二次项。
能量上升最多进行四次折半，仍失败则恢复该次求解前状态。
折半端点都在同一 DAT 平面可行区内；候选缺失保护的位移预算仍为 `0.425*margin`。
这是当前场景的数值安全策略，不是对任意退化网格的精确 CCD 保证。

### 完整轨迹实测

RTX 5060 Ti / Warp 1.17.0.dev20260807；2500 点、4802 三角形，600 帧。
计时仍是完整物理 Graph，不含初始化、渲染与计时区间外的几何审计。
下面所有 XPBD 成功运行均逐帧通过独立交叉审计、sticky status 和驱动边界误差检查。

| 版本 / 结果标签 | GPU ms/frame | wall ms/frame | 末帧面积变化 RMS | 主伸长变化 RMS |
| --- | ---: | ---: | ---: | ---: |
| 原 VBD `vbd_global_reference` | 25.661 | 25.783 | 0.10493 | 0.17132 |
| 全局 XPBD 2×6 PCG `globalgeo2cg6` | 25.179 | 25.243 | 0.10404 | 0.16411 |
| 全局 XPBD 2×4 PCG `globalgeo2cg4` | 21.237 | 21.302 | 0.10156 | 0.16253 |
| 公共 API 复测 `global_public2cg4` | 21.389 | 21.453 | 0.10183 | 0.16265 |
| 直接运行新默认 demo | 21.257 | 21.315 | — | — |

相对原 VBD，2×4 版本完整物理 wall time 减少约 **16.8%–17.4%**。
公共 API 复测末帧面积比范围 `0.73057–1.49186`，VBD 为 `0.76026–1.47363`；
最大主伸长 `2.1241` vs `2.2293`；动态速度峰值 `0.48722` vs `0.48807 m/s`。
末帧动态动能 `0.003326` vs `0.003505`。动能统计排除固定边界，
避免 XPBD 给运动边界重建速度而原 VBD 记录零速度造成假差异。
但第 100 帧动态动能 `0.00624` vs `0.01150`，早期动态响应仍不同，不能只展示末帧。

独立 NumPy float64 最近特征诊断：公共 API 复测最小间距 **2.79 µm**，
没有 `<1 µm` 的候选；本次 VBD 最小 **0.348 µm**，有 1 个 `<1 µm` 候选。
该统计包含全部非 incident 近邻；不是平均厚度或穿透量，也不能推出任意场景无自交。
公共 API 复测与 VBD 的末帧逐点位置 RMSE **43.08 mm**，p95 **75.16 mm**。
两次 VBD 自身运行的末帧 RMSE 约 **19.70 mm**，不能把 XPBD 的全部差异归因于非确定性。

结论：**本场景性能达标，整体形变指标与观感已接近，严格逐点轨迹精度未等同**。
不能声称已证明所有场景同精度更快；目前只支持既有 FEM 门控下的纯三角布料，
不新增 tet、机器人刚体耦合、requires-grad 或 deterministic 的支持。

### 撤出的实验

| 标签 | 结果与 No-Go 原因 |
| --- | --- |
| `blocknh4` | 局部双约束非零静态预应力，面积 RMS 0.703，严重抖动 |
| `zerorest_nh4` | 局部双约束虽 21.70 ms，但面积 RMS 0.235，质量不足；删除局部 block 模式 |
| `global3cg8` | 单偏应变约束，33.33 ms、过大动能，未保留 |
| `globalblock3cg4` | 尚无能量接受保护/几何刚度，20.86 ms 但面积 RMS 0.147，未保留 |
| `globalmerit3cg4` | 第 381 帧驱动边界验收失败 |
| `globalmerit2cg6` | 24.36 ms、面积 RMS 0.107，但动态速度峰值约 0.82 m/s，需补几何刚度 |
| `globaltopo2cg8` | 一环响应过滤，第 142 帧驱动边界失败；过滤已删除 |
| `global8x2cg8` | 减到 8 外层 substeps，第 191 帧边界失败；默认仍为 10 |

历史输出中的早期全局实验 contact metadata 尚显示 quadratic/未启用 damping，
但当时全局代码已使用 barrier 和 kd=0.1；这些旧元数据有误，以本节实验说明为准。
新结果 `globalgeo*` 与 `global_public*` 的 contact metadata 已修正。
历史标签是当时实现，撤出的代码不作为现在可用的隐藏开关。

### 使用与回归

```powershell
# 新 demo 默认：10 substeps，2 次非线性求解，每次 4 PCG，global 材料阻尼
uv run --extra examples -m newton.examples cloth_twist_xpbd
# 原 PhysX-equation 路径：12 temporal slices，阻尼率 600/s
uv run --extra examples -m newton.examples cloth_twist_xpbd --fem-solver tgs

uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver vbd --label my_vbd
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver xpbd --label my_global --global-xpbd --iterations 2 --cg-iterations 4 --damping 0.0002 --bending-damping 0.01 --match-vbd-hinge
uv run --extra examples python scripts/analyze_cloth_twist_results.py outputs/xpbd_physx_alignment/my_global.npz --reference outputs/xpbd_physx_alignment/my_vbd.npz
uv run --extra examples -m unittest newton.tests.test_xpbd_global newton.tests.test_xpbd_fem newton.tests.test_solver_xpbd newton.tests.test_solver_xpbd_tetrahedra
```

原 82 项回归通过；新增全局测试覆盖 CPU/CUDA 稠密组装算子/PCG 直接解对照，
DAT 截断后的 primal residual、固定点、自由平移及关闭自接触、选项校验、
膜能量有限差分梯度、法向几何曲率有限差分/压缩 PSD，以及 overflow 拒绝全部全局位移。
先完整运行 87 项通过，再补充 overflow 回归后全局 6 项通过（合计 88 项）。
未运行全仓库回归，也未提交或推送。README 的 320×320 截图更新为新默认实际最终帧。
公共 `python -m newton.examples cloth_twist_xpbd` 入口的 global/TGS 两种模式
另各通过 20 帧 `--viewer null --test` smoke test；修改文件 Ruff lint/format 检查通过。

## 11. 2026-09-07：两层 PCG 筛选与等价 element cache

本轮以用户已暂存的第 10 节版本为基线。**没有执行 stage、commit 或 push，没有改变暂存区。**
只保留全局 XPBD 的等价算子优化、测试、demo 默认值和本记录；没有增加 multigrid 模式或公共调参开关。

### 保留实现

- 三角形 5 个标量切线行合并为三个 3×3 块 `K11/K12/K22`，作用于
  `(dx1-dx0, dx2-dx0)`。利用每行 `g0=-g1-g2`，精确表示原 6×6 相对位移算子，
  **不是低秩近似，也不是跨帧使用旧刚度**。每次非线性线性化后重新构建。
- 同时缓存每个 element 的顶点 RHS 和对角块。`initialize_system` 与每轮 PCG
  都通过 element→vertex CSR gather，避免三角形的 5 条约束在每个顶点重复读取/计算。
- 弯曲、膜阻尼、法向几何刚度、完整 3×3 block-Jacobi、固定点、惯性 primal residual、
  活跃接触和摩擦仍保留。没有改接触候选、检测频率、材料、DAT 或能量回溯。
- GPU dot/energy 改为 block 内归约后 FP64 全局累加；全局 kernel 使用 64-thread block，
  改善本场景 2500 顶点的并行占用。依旧是固定 Graph，无步内 CPU readback。
- 原逐行乘积保留为私有数值诊断参考，不进入运行时 PCG。新增缓存容量在构造期固定，
  按三角形/hinge 数线性增长，不组装全局稀疏矩阵。
- demo 使用 **10 substeps × 2 nonlinear solves × 5 PCG**，把部分节省的时间用于线性收敛。
  `--linear-iterations 4` 可用；`SolverXPBD` 公共线性迭代默认值仍为 4，
  原非 FEM 和 `tgs` 路径不改，未扩大纯三角布料的支持范围。

### 完整 600 帧性能与形变

同机 RTX 5060 Ti / Warp 1.17.0.dev20260807；同网格、轨迹、材料、10 substeps。
完整物理 Graph，前 10 帧不计，初始化/编译/渲染/独立几何审计不计。
所有 GPU 计时顺序执行；基线由暂存源码独立加载，未替换工作文件。

| 实现 / 本机结果标签 | GPU ms/frame | wall ms/frame | 末帧面积变化 RMS | 主伸长变化 RMS |
| --- | ---: | ---: | ---: | ---: |
| 暂存原版 `baseline` | 20.652 | 20.691 | 0.10191 | 0.16223 |
| 暂存原版复测 `baseline_repeat` | 20.750 | 20.792 | 0.10411 | 0.16645 |
| 缓存 4 PCG `retained_elements2cg4` | 20.100 | 20.144 | 0.10385 | 0.16337 |
| 缓存 5 PCG `assembled_elements64_2cg5` | 20.374 | 20.416 | 0.10263 | 0.16584 |
| 缓存 5 PCG 复测 `retained_elements2cg5_repeat` | 20.332 | 20.373 | 0.10276 | 0.16587 |

缓存 4 轮相对两次原版 wall 减少 **2.6%–3.1%**；5 轮约 **1.3%–2.0%**。
这是小幅端到端收益，不是大幅加速结论，也没有证明跨 GPU/跨场景收益。
不使用旧日期的 VBD 耗时计算本轮新提速比。

以上完整轨迹逐帧通过 sticky status、规定边界误差和独立非 incident edge–triangle 交叉审计。
缓存 4 轮末帧面积比 `0.70481–1.51026`、动态速度最大 `0.48761 m/s`；
5 轮复测为 `0.70398–1.50865`、`0.49772 m/s`。
独立 float64 最近特征诊断：4 轮最小距离 **0.208 µm**、1 个 `<1 µm` 候选；
5 轮复测 **7.571 µm**、0 个 `<1 µm` 候选。
没有交叉不代表最小间距有可靠统一下界，不能据此承诺任意场景无穿透。
二者相对本轮第一份原版末帧位置 RMSE 分别 **20.34 / 24.99 mm**；
代数等价不意味着原子累加与舍入顺序改变后，600 帧非线性接触轨迹 bitwise 相同。

### 同输入线性精度检查

取原版第 100、300、600 帧位置/速度，构造一次非线性线性化，固定全部梯度、
接触、RHS 和预条件器，分别运行逐行/缓存乘积。另用 NumPy/SciPy float64
独立组装同一带固定点的稀疏系统并直接求解（仅离线诊断，不作为求解器依赖）。
直接解相对残差为 `8.9e-15 / 3.2e-14 / 3.4e-13`；不是把尚未收敛的 64 PCG 当真值。

| 帧 | 4 PCG 真残差 `||b-Ax||/||b||` | 5 PCG 真残差 | 4→5 的相对 A-norm 解误差 |
| --- | ---: | ---: | ---: |
| 100 | 0.42689 | 0.39536 | 0.54644 → 0.49547 |
| 300 | 0.32390 | 0.24594 | 0.62680 → 0.58854 |
| 600 | 0.35494 | 0.29627 | 0.94687 → 0.94191 |

同轮数下缓存/逐行 A-norm 误差之差小于 `3e-8`，GPU RHS 对独立组装相对误差小于 `6e-6`。
5 轮在这三个冻结样本上确实改善线性收敛，但**后期低频误差仍很大**。
冻结的一次线性修正精度不等于整条带 DAT/回溯的非线性轨迹精度；
本轮没有证明与充分收敛解或 VBD 全部精度等同，视觉效果仍由用户验收。

### 未保留的方向

参考 [MGPBD 论文](https://arxiv.org/html/2505.13390v1) 的 AMG-PCG 思路，
本轮实际筛选的是 **primal 两层 additive Galerkin 预条件**，不是论文 dual-space AMG 的完整复现：
按连通块/rest-space 聚合为 16 个平移 cluster，48 自由度粗层，精确构建 `PᵀAP`，
含接触与固定点，GPU 小矩阵 Cholesky，细层仍为 block-Jacobi。
小系统 Galerkin/粗层解 CPU、CUDA 对照通过，但不代表完整场景受益。

| 试验 | wall ms/frame | 未保留原因 |
| --- | ---: | --- |
| 两层 + 每轮 3 PCG `mg2cg3` | 21.126 | 不快于原版；简单平移基未给出足够的收敛收益 |
| 两层 + 每轮 2 PCG `mg2cg2` | 19.708 | 动态/线性收敛检查不足，未证明速度收益不以质量为代价 |
| 全局细层稀疏 block 装配 + 6 PCG `blocks2cg6` | 29.071 | 每次非线性装配成本过高 |
| 单次非线性 × 8 PCG `global1cg8` | 18.761 | 后半段反复约 0.893 m/s 速度峰值，不保留减非线性轮方案 |

不能据此说 multigrid 对 XPBD 无用；只能判定**这版简单平移粗基不适合设为默认**。
更强粗空间、接触自适应层次和构建复用仍需独立设计和端到端验证，本轮没有加入这些未验证模式。

### 回归与复测

89 项相关测试通过，覆盖 CPU/CUDA、原 XPBD 刚体/关节/tet 和 FEM/DAT。
新增缓存乘积与稠密组装对照（含接触），65 顶点的非平面受拉网格测试涵盖
非零 hinge/阻尼、固定点、完整 block-Jacobi、RHS 和不满 64-thread block 的点积归约。
新默认 demo 直接入口另完成 600 帧 `--viewer null --test`。
未运行全仓库全部 demo，不作“绝无 bug”或通用无穿透保证。

本机本轮原始 JSON/NPZ/截图和临时试验存放于 `.git/xpbd_mg_experiments/`，不进入修改区或提交。
正式复测继续使用现有脚本，不增加散落的 benchmark 脚本：

```powershell
uv run --extra examples -m newton.examples cloth_twist_xpbd
uv run --extra examples -m newton.examples cloth_twist_xpbd --linear-iterations 4
uv run --extra examples -m newton.examples cloth_twist_xpbd --fem-solver tgs
uv run --extra examples python scripts/compare_cloth_twist_solvers.py --solver xpbd --label cache5 --global-xpbd --iterations 2 --cg-iterations 5 --damping 0.0002 --bending-damping 0.01 --match-vbd-hinge
uv run --extra examples python scripts/analyze_cloth_twist_results.py outputs/xpbd_physx_alignment/cache5.npz
uv run --extra examples -m unittest newton.tests.test_xpbd_global newton.tests.test_xpbd_fem newton.tests.test_solver_xpbd newton.tests.test_solver_xpbd_tetrahedra
```

## 12. 2026-09-07：同步远程 FAST_MJVBDV2 后的提交前验证

按用户要求先保存本地全部暂存内容，再将分支从 `d2fb3197` 快进到远程 `9746291e`，
合入 64 笔更新。三方恢复本地改动，解决 `solver_xpbd.py` 的 import/property 冲突：
保留远程 restitution kernel、刚体速度增量更新及兼容属性，同时保留 FEM/DAT API 和独立分支。
README 自动合并，未回退远程 VBD/MJVBDV2 或其他模块的更新。

依照新 lockfile 使用 Warp **1.17.0 正式版**、MuJoCo/MuJoCo-Warp **3.12**。
同一新增 FEM 构造 API 测试，在远程原始 SolverXPBD 上因不支持 `particle_fem` 失败，
在集成版本通过。相关四个测试模块共 **115 项通过**（239.35 s，包含新环境首次编译），
含远程新增 restitution 等回归。默认 global demo 600 帧、原 TGS demo 20 帧
`--viewer null --test` 均通过。没有以并行测试期间的耗时作为性能数据；
第 11 节数字属于拉取前环境，本次同步只做功能验证，未重新做完整性能对照。

执行 `uvx pre-commit run -a`：全仓库 Ruff 的 **737 条既有错误**在恢复本地改动前后都存在，
集中于无关 multiphysics 示例（中文标点、局部 import、未使用变量等），没有混入清理。
其他全仓库 hooks 通过；本次全部提交文件的 `pre-commit run --files ...` 通过。
本轮应只提交原 15 个任务文件及本记录，不包含临时实验文件或无关远程代码格式化。
