# 全局 barrier-free 求解试验

日期：2026-09-10。基线：FAST_MJVBDV2 / 49bb6220。

## 决策

**最终决定：No-Go，按用户要求回退，仅保留本日志。**

- 全局软接触实验完成 50×50 / 600 帧，平均约 74.176 ms/frame；原 VBD 基线约
  11.716 ms/frame，超过允许的 +20% 上限（约 14.059 ms/frame），性能未达标。
- 接受运动通过数值几何路径检查，但不构成严格无穿透证明；摩擦滞后离散仍与 VBD 不同，
  有限 Newton 也未达到严格力残差收敛，不能宣称精度等价。
- 撤出本次未跟踪的实验求解器、demo、测试、性能/绘图脚本及生成的数据、图像和轨迹。
  生产求解器没有改动，不需要回退任何已跟踪代码。未 stage、未提交。
- 下文保留历史尝试、测量值及失败原因。文中脚本和数据文件名是历史记录，
  回退后不再作为工作区内可运行脚本或可访问的验收附件。

目标是通用求解器的收敛质量和整帧性能，不要求保留 VBD 或 DAT。
试验独立于生产路径；目前没有修改 surface-fast、生产 DAT 或 demo 默认值。
不能在安全替代方案未实现、未验证时直接删除 DAT。

参考：[Robust and Efficient Penetration-Free Elastodynamics without Barriers](https://arxiv.org/html/2512.12151v3)。
重点是分开维护优化状态和安全状态，以带符号线性接触约束做 AL 全局求解。
原文仍使用 CCD 发现路径中的接触并构造安全输出；无 barrier 不等于无安全检查。
用 DAT 替代它属于混合算法，不能直接继承原文的保证。

## 已完成的最小实现

- `scripts/barrier_free_contact_experiment.py`：冻结接触行的 AL 目标、梯度、完整跨顶点 Hessian-vector product、slack 和 multiplier/decay 更新。
- 固定容量、device-side count 和独立 sticky overflow 检查；overflow 必须使本次求解失败，不能把截断的接触集当成安全集合。
- 固定自由度从方向和梯度中消元，但仍参与接触间隙计算。
- slack 在一次 Newton 方向及其 line search 内固定；不能每个试探步都更新它后仍使用原 Hessian。
- slack-positive 接触梯度可为零，但仍保留衰减的曲率，不能简化为立即删行。
- 矩阵乘是正确性参考，不是已选定的高性能最终实现；仍需与 BSR 装配比较。

`scripts/test_barrier_free_contact_experiment.py` 的四项测试在 CPU 和 CUDA 上通过：

1. 独立 NumPy 稠密参考与有限差分检查目标、梯度、Hessian（含非对角顶点块）。
2. 非激活行曲率及乘子、衰减更新。
3. 惯性项加接触项的预条件 CG 对照稠密直接解。
4. CUDA Graph 重放时改变活动数量，空接触、填充行屏蔽及 sticky overflow。

CPU 有限差分测试初版失败，是 `.numpy()` 的共享视图导致两个历史梯度指向同一内存；改成显式复制后通过，未调整数学公式或误差阈值。

运行：

```powershell
uv run --no-sync python scripts/test_barrier_free_contact_experiment.py
```

这不是完整模拟器，也不是性能测量。此前 `test_mjvbd_contact_gap_split.py` 检验的是原有柔性接触势的 proximal 分裂，不是本次论文算法，不能混用其通过结果。

## 第二阶段：全局布料目标与线性系统（已实现并测试）

新增 `scripts/barrier_free_cloth_experiment.py`、对应 unittest 和
`scripts/benchmark_barrier_free_cloth_experiment.py`，仍为独立诊断，不是可安全步进的求解器。

- 原始 stable-NH 膜能、二面角弯曲能、膜/弯曲阻尼及惯性目标；梯度对照原 VBD force kernel 和有限差分。
- 膜能采用 3×2 形变梯度空间的完整六维谱投影，保留跨顶点块。
  拉伸、压缩、旋转/剪切状态对照 NumPy 数值 Hessian 的 eigendecomposition。
  弯曲和阻尼仍用 Gauss–Newton 曲率，不称为完整精确 Newton Hessian。
- block-Jacobi 预条件 PCG、真实目标的 Armijo 回溯和固定自由度消元。
- 膜能/应力与能量求和使用双精度评估，位置、矩阵和 PCG 保持单精度。
  仅将单精度能量转换为 double 再累加不能消除相消误差，因此转换发生在几何运算前。
- 退化几何、非有限量、非下降方向和过大的 PCG residual 明确拒绝，不默默接受。
- 禁止嵌套 element/block 循环自动展开，首次 CUDA 编译由约 208 秒缩短到约 7 秒。
  这是编译行为观察，不算运行时加速收益。
- PCG 明确设置绝对容差；原本仅设置相对容差会被 Warp 默认绝对容差截住。

五项布料测试均在 CPU/CUDA 通过，另四项 AL 接触测试继续独立保留。
布料测试覆盖能量/力匹配、谱投影、全局矩阵正定和对称、目标下降，以及 inactive massive particle 固定/退化拒绝。

### 实测结果与不通过项

4 粒子、2 三角形补丁：12 轮后目标从 1.08689775 J 降为 0.02061684 J，
梯度范数从 114.369 N 降到 CPU 1.672e-4 N / GPU 1.660e-4 N。
满足测试的相对下降要求，但**没有达到指定的 1e-4 N 绝对停止阈值**，不能称为绝对容差收敛。
曾有不同编译路径 CPU 10 轮达到 9.78e-5 N 的结果；不能把该结果泛化为稳定的 GPU 10 轮收敛。
最初简单 PSD 近似在双精度能量评估下跑 80 轮仍未通过相对残差 1% 的检查，已用谱投影替换，不保留开关。

更大网格的配置、逐轮记录和三次 wall time 在 `global_cloth_grid16.json`、`global_cloth_grid32.json`。
测试是独立的、无接触增量目标；不是 T-shirt 完整 substep，更不是整帧。

- 289 粒子、512 三角形：一次测试装配加一次矩阵乘 Graph 约 0.140 ms；
  20 轮预算诊断求解 wall time 约 79–174 ms，出现 line-search failure，末端残差约 0.0038 N。
- 1089 粒子、2048 三角形：首次测试 Newton 第 3 轮 PCG 残差
  0.0076858 N / RHS 0.0763508 N，略超 10% 的不精确 Newton 上限，被明确拒绝。
  复跑三次没有触发该拒绝，但均不能据此宣称达到目标容差；一次 20 轮末端残差为 0.00823 N。
  复跑的 Graph 约 0.139 ms，wall time 约 110–205 ms。
  原子累加及阈值附近的数值差异会影响停止情况，当前可靠性仍需改进。

这些 wall time 包含 CPU 回读、重复 PCG 临时分配及线搜索装配，不能作为最终全 GPU 求解器成本。
与此同时，首轮 PCG 达 180/200 次是确实存在的算法瓶颈，不能全部归因于 Python 开销。
当前结论：**保留数学基线用于下一步改进，但不是可替换 surface-fast 的性能通过版本。**

下一优先级是更强的全局预条件器/粗空间和近收敛数值处理，再消除重复装配与分配。
不可用放宽误差阈值或添加速度衰减把失败包装成通过。

运行：

```powershell
uv run --no-sync python scripts/test_barrier_free_cloth_experiment.py
uv run --no-sync python scripts/benchmark_barrier_free_cloth_experiment.py --grid 16 --iterations 20 --output docs/lab/mjvbd_surface_convergence_2026-09-10/global_cloth_grid16.json
uv run --no-sync python scripts/benchmark_barrier_free_cloth_experiment.py --grid 32 --iterations 20 --output docs/lab/mjvbd_surface_convergence_2026-09-10/global_cloth_grid32.json
```

## 第三阶段：优先确认效果，平面接触驱动（2026-09-10）

本阶段不做性能 Go/No-Go。`FrozenContactRows` 已接入全局目标、梯度、Hessian-vector product
与对角预条件器，slack 在 Newton 方向/线搜索期间冻结，每轮重新准备。
新增 `validate_barrier_free_contact_effects.py`，是**只支持一个水平平面的效果验证器**，
不是通用生产 solver，也不是 T-shirt demo 的替代品。

### 数值修正，而不是放宽验收

单精度位置版本在尚未接触时就会在约 2.13e-4 N 停滞，超过 2e-4 N 的验收阈值。
位置、几何目标和二面角计算改为 double，矩阵及 PCG 保持 float。
同一布料补丁现在 CPU/GPU 均在 4 轮后达到约 5e-6 N，确实通过 1e-4 N 绝对阈值。
五项布料测试补上了 `converged=True` 断言，没有仅凭相对下降宣称收敛。
为效果验证，实验 PCG 预算从 200 提高到 2000；此前 JSON 性能报告属于旧预算/旧精度版本，不可套用于当前版本。

### 接触与摩擦语义

- 平面半空间是凸集，两个可行端点之间的整条直线路径可解析证明可行；不是只查步末距离。
- 优化迭代可进入平面，安全状态单独维护。每步最终输出还必须通过非穿透、KKT 力残差、间隙和互补条件检查。
- 失败时不更新物理位置和速度；失败迭代不能作为仿真结果输出。
- 法向接触为 AL 硬约束，不是旧 VBD 软接触势，因此不直接横比两者能量数值。
- 摩擦按[论文附录 A](https://arxiv.org/html/2512.12151v3#A1)的半隐式平滑摩擦思路：前一时间步的法向载荷、切空间固定到当前步。
  当前只实现水平面切空间；不是完整 VT/EE 摩擦。
- 未加速度衰减、位置冻结、末端时间开关或虚构接触几何。

### 真实动态测试

25 粒子、32 三角形的倾斜布片，0.25 m 边长、0.03 kg，总计 0.01 s/step。
相同材料、初始状态，分别运行摩擦 0.4 和 0；另外测试一次显式标注的向上体力脉冲。
体力脉冲在第 60–69 步施加净向上加速度 20 m/s²，用于检查脱离接触；不是解算器隐藏驱动。

- 摩擦 0.4：60 步完成。末端 RMS 速度约 2.4e-8 m/s。
- 摩擦 0：60 步完成。末端 RMS 速度 0.299982 m/s，保持初始约 0.30 m/s 的水平滑动。
  因而不能把有摩擦时静止归因于统一速度衰减。
- 抬升/再次落地：120 步完成。第 80 步离接触平面的最小间隙约 0.256 m；
  第 120 步再次稳定，RMS 速度约 1.97e-8 m/s。
- 三条轨迹的接受状态都满足平面非穿透。接触态保持约 0.10 mm 的安全分离量；
  粒子物理半径为 2 mm，视觉表面与接触中心要区分。
- 所有接受步满足 KKT 力残差 <= 2e-4 N、约束间隙违例 <= 1e-7 m、互补残差 <= 1e-7 J。

新增三项测试：整段平面可行性、摩擦梯度/曲率有限差分、失败步不发布优化状态。
连同原九项，共十二项通过；摩擦与原核心均有 CPU/CUDA 覆盖，失败发布测试在 CPU 上执行。

### 本地查看

- `plane_effects.gif`：慢放动画，标题时间是仿真时间，不是实测运行速度。
- `plane_effects_snapshots.png`：下落/贴地/抬升/再次落地的几何快照，已人工视图检查。
- `plane_effects_curves.png`：速度、间隙和 KKT 残差曲线，已视图检查。
- `plane_effects/`、`plane_no_friction/`、`plane_release/`：完整 JSON 与 NPZ 轨迹。

```powershell
uv run --no-sync python scripts/validate_barrier_free_contact_effects.py --frames 60 --output docs/lab/mjvbd_surface_convergence_2026-09-10/plane_effects
uv run --no-sync python scripts/validate_barrier_free_contact_effects.py --frames 60 --friction 0 --output docs/lab/mjvbd_surface_convergence_2026-09-10/plane_no_friction
uv run --no-sync python scripts/validate_barrier_free_contact_effects.py --frames 120 --release-frame 60 --output docs/lab/mjvbd_surface_convergence_2026-09-10/plane_release
uv run --no-sync python scripts/plot_barrier_free_contact_effects.py --root docs/lab/mjvbd_surface_convergence_2026-09-10
uv run --no-sync python scripts/test_barrier_free_effects.py
```

这只确认了较简单的平面接触效果；布片几何接近平面，不是复杂褶皱验收。
**尚不能证明自碰撞安全、密集叠层收敛或 T-shirt 末端不抖。**
下一阶段必须做 VT/EE 路径检测和活动约束更新，再验证层间接触和真实 T-shirt。
生产 surface-fast、DAT、材质和 demo 默认值仍未改动。

## 尚未实现，禁止据此宣称完整验收通过

- 通用 VT/EE 与机器人接触驱动；下述 twist 初版仅支持实验性法向自接触。
- 通用 VT/EE 摩擦的梯度与 merit function；旧 frozen-friction 能量图不能直接用于新算法 line search。
- 弯曲完整二阶项及其投影；目前为 GN 近似。
- GPU swept broad phase、完整 VT/EE 安全推进与论文碰撞时间筛选；下述 CPU 数值路径检查不是其完整实现。
- 持久接触标识、约束筛选/淘汰、容量恢复和 moving-boundary 语义。
- 原文硬间隙约束与旧软接触模型的差异评估；不能把刚度模型变化叫作求解精度提升。
- 非平面摩擦路径及完整动量/能量测试。

## 集成与验收顺序

先完成弹性全局系统，在相同输入状态上检查目标下降、力残差和约束残差。
再接触检测与安全路径，测试高速、多层、边边擦碰、释放、初始近接触和 overflow。
安全检查不得仅检查步末距离，必须覆盖运动路径；优化迭代的穿透不能泄漏为输出状态。

随后使用真实 T-shirt 轨迹测试整帧，普通 30 sweep 只作同状态参考而不是精确解。
同时检查 plastic bag 和 cloth twist，以各自原方案作参考。
保持 substep、材质、轨迹、几何不变，不加轨迹结束后的速度衰减或冻结。

记录完整 Graph GPU 时间和端到端 wall time（排除编译但不排除碰撞、装配、PCG、线搜索），
以及自交、接触间隙、速度尾部、形变和能量残差。
30 sweep 参考是软接触时，另报告接触模型差异，不能只比较两个不同目标的能量数值。
只有这些通过后才考虑替换生产路径或移除 DAT；当前没有整帧收益结论。

## 第四阶段：独立 cloth twist 实验例子

入口：`scripts/example_barrier_free_cloth_twist.py`。保留在研究脚本目录，
不注册为成熟求解器，不修改原 cloth_twist、surface-fast 或生产 DAT。

```powershell
uv run --no-sync python scripts/example_barrier_free_cloth_twist.py
uv run --no-sync python scripts/example_barrier_free_cloth_twist.py --viewer null --num-frames 120 --test --output docs/lab/mjvbd_surface_convergence_2026-09-10/cloth_twist
```

- 尺寸 1.5 × 1 m，沿两端施加相反方向的 π/3 rad/s 转动，10 秒后停止边界转动。
  密度、膜/弯曲刚度和物理阻尼采用原 cloth_twist 配置，重力为零。
- 初版为 12 × 12 顶点规则网格，不是原 50 × 50 USD 网格；不能据此宣称同分辨率效果或速度一致。
- 全局增量势能 Newton-PCG + 法向 VT/EE AL 约束；没有额外速度衰减或收尾冻结。
  **自接触摩擦尚未实现**，因此仍不是原 twist 的完整替代。
- `barrier_free_mesh_guard.py` 使用 swept AABB 和双精度距离保守推进检查整段运动；
  VT 排除所属三角形，EE 排除共享顶点。不是精确谓词 CCD，不宣称理论上无条件无穿透。
- `barrier_free_twist_contact.py` 保留活动对和法向乘子，用 AL 罚系数延续改善约束收敛。
  尚未实现论文碰撞时间接触筛选及完整衰减生命周期；大量 CPU 回读和重新装配只用于正确性原型。
- 每个接受步检查路径安全、KKT 力残差 ≤ 2e-4 N、线性约束违反 ≤ 1e-7 m、
  互补残差 ≤ 1e-7 J。失败时明确报错并保留最后接受状态；**报停不是物理收敛**。
- 接触 offset、slack、乘子和间隙累积改为双精度，避免大坐标相消吞掉小间隙；
  Jacobian、矩阵和 PCG 仍为单精度。

阶段性测试：无接触前 30 帧（0.5 秒）通过；独立“顶点穿过三角形”法向响应测试通过，
将顶点求解到接触间隙，而非仅拒绝更新。路径检查的穿越、刚体平移、退化/边界距离三项通过。
早期固定罚系数 80 外迭代在 1.5183 秒失败：力残差 2.08e-5 N，
但间隙违反 3.24e-7 m、互补残差 1.83e-6 J 超标，未接受失败状态。
这说明单接触通过不代表密集 twist 已通过；不通过放宽验收阈值掩盖问题。

随后自适应罚系数版本完成 120 帧（2 秒），末步力残差 2.4811e-5 N，
间隙违反 2.7614e-8 m、互补残差 9.8371e-8 J；结果位于 `cloth_twist/`。
只覆盖这段低分辨率运动，不是完整 10 秒扭转或原 2500 顶点场景验收。

### 性能门槛收紧后的改动

用户要求端到端耗时不超过现有基线的 1.2 倍。当前实验 **尚未达标，不可替换生产路径**。

1. PCG 工作区改为持久分配，CUDA 使用条件 Graph 在设备端检查残差，
   不再每五轮回读 CPU。接触对象或 penalty 变化时重录算子，避免重放旧约束。
   相同线性容差和最大轮数，未用固定少量迭代代替收敛。
2. 线搜索只计算弹性目标，不重复 SVD/Hessian 装配；弯曲梯度每个 hinge 计算四次后复用，
   不再在 Hessian 双重循环内重复计算。
3. swept AABB 从 CPU 二次复杂度枚举改为 GPU BVH 查询，活动计数在设备端维护，
   候选缓冲容量固定，overflow 明确报错；保守膨胀覆盖 float32 包围盒舍入。
   同一组候选同时用于路径验证和新接触提取，不再重复生成。
   CPU 枚举保留为独立正确性参考；窄相路径验证仍在 CPU，这是剩余瓶颈之一。

实测（RTX 5060 Ti，Warp 1.17.0，排除初始化/预热；未启用 profiler）：

| 范围 | 方案 | wall time |
| --- | --- | --- |
| 169 顶点孤立增量势能，三次重复中位数 | 原诊断 PCG | 57.34 ms/solve |
| 同一输入、容差 | 持久 PCG + GPU 条件循环 | 30.34 ms/solve |
| 12×12 twist，早期第 2–11 帧，10 substeps | 同网格 MJVBDV2 surface-fast / 3 sweeps | 2.369 ms/frame |
| 同一窗口 | 全局求解，GPU PCG，仍用 CPU broad phase | 151.56 ms/frame |
| 同一窗口 | 再加入 GPU BVH 和候选复用 | 58.63 ms/frame |

最后一项比上一项降低约 61.3%，但仍约为 VBD 的 24.7 倍，远超 1.2 倍限制。
不能用相对低效原型的加速来冒充对生产基线达标。这里两种接触模型不同，
只用于量化性能差距，不是物理精度等价结论，也不是完整 twist 后段性能。
GPU event 数据包含 CPU 发射空洞，字段明确命名 `gpu_timeline_ms_including_host_gaps`，
不能把它称为纯 kernel 耗时。

测速入口 `scripts/benchmark_barrier_free_twist.py`，同一程序创建相同网格、材料、边界和 substeps，
分别测 `--backend global` / `--backend vbd`。JSON 和单独 cProfile 结果均在本目录。
带 profiler 的数字不参与上述比较。

另尝试解析消去 slack 的 reduced AL，但用 `cloth_twist/trajectory.npz` 的末帧冷启动时，
reduced 和 frozen 两者均在同一 VT 路径检查报停（pair 66,30,31,42），未证明端到端收益。
该冷启动速度由相邻显示帧差分指定，不是原运行的精确 substep checkpoint，
不能据此判定原连续轨迹回归。reduced 实现及其新增测试文件已撤除，不增加默认求解分支。

最终补测 50×50 同构规则网格、相同第 2–11 帧窗口：VBD 中位数 **3.550 ms/frame**，
全局实验 **317.14 ms/frame**。门槛应为 4.260 ms/frame，因此仍明确 No-Go。
该网格与原 USD 分辨率一致，但不是逐字节相同的 USD 三角剖分；双方测速使用同一程序构建的同一网格。
结果见 `twist50_vbd_baseline.json` 和 `twist50_global_gpu_bvh.json`。

执行优化后连续 12×12 twist 120 帧再次通过，末步 KKT 力残差 2.2132e-5 N，
间隙违反 2.4526e-8 m、互补残差 7.4080e-8 J。轨迹、JSON 和快照见 `cloth_twist_gpu/`。
21 项实验回归测试通过（包括 GPU BVH 对独立 CPU 枚举的覆盖、Graph refit 重放、
overflow 拒绝、能量专用路径一致性、PCG 在接触行/罚系数变化后的显式线性残差）。
该连续测试录入时 BVH 从零边界构建；其后仅将初始 BVH 构建改为初始实际几何，
最终构建形式经过单元测试与 50×50 早期帧测试，尚未再次跑完整 120 帧。

结论：保留独立实验中已验证的执行优化，不发布为满足性能要求的求解器，
生产 demo、MJVBDV2 默认选项、DAT 和摩擦均未修改。后续需要解决完整 GPU 驱动和
多接触非线性/约束系统的收敛成本，而不是继续调低精度或把报停包装成成功。

### 继续优化：设备端 Newton、因子化算子与融合 PCG

以下为后续实验结果，更新上文历史进度；**仍未达到不超过 VBD 基线 1.2 倍的门槛**。
改动全部位于独立实验脚本，不改变生产 MJVBDV2、默认 demo、DAT 或摩擦。

- 窄相 VT/EE 双精度距离和保守路径推进迁到 GPU；只回读临近接触与失败信息。
  固定容量、device count、overflow 拒绝不变。CPU 几何检查保留为独立测试参考。
  这不是精确谓词 CCD，也不是论文碰撞时间筛选的完整实现，不能据此宣称无条件无穿透。
- 将完整 Newton / Armijo / PCG 嵌套循环录成条件 Graph；同容量接触行复用分配。
  首版发现线搜索耗尽语义与 CPU 不一致，导致 1.655 秒报停；已修正为返回最后接受的
  **优化迭代**供 AL 更新，物理状态仍必须通过最终路径和 KKT 检查才发布，并添加回归测试。
- 膜 Hessian 使用 F 空间 6×6 因子化，弯曲使用角度梯度的秩一乘法，保留跨顶点耦合。
  3×2 分解、stretch 模态正定投影和二面角梯度使用解析表达式；未删除能量项。
  能量、范数和方向内积使用分块归约，避免全网格竞争单个地址。
- 使用 inexact Newton：远离收敛时线性相对容差 0.03，接近时 0.005；最终非线性
  力、间隙、互补条件和 Armijo 条件不变。不是将线性系统当作精确求解。
- 用最近 8 个 Newton 方向生成 Galerkin 初值；每次重新计算当前 H 的乘积，
  不复用旧 Hessian。退化的小系统回退零初值，最终仍由 PCG/非线性验收决定。
- 初值改为惯性预测位置；只改变优化起点，不直接接受预测运动，完整路径仍从原物理位置检查。
- 融合 PCG 的向量更新、块 Jacobi 预条件和归约；退出后重新计算显式 `b-Hx`。
  添加多块归约、变 RHS/非零初值的 Graph 重放、零 RHS 和负曲率拒绝测试。
- 完整测试发现零 hinge 数量触发 Warp 空切片错误，已修复；同时处理零 triangle 切片。

同一 RTX 5060 Ti、50×50 生成网格、10 substeps、预热后第 2–11 帧、无 profiler：

| 累积方案 | wall 中位数 ms/frame |
| --- | ---: |
| VBD surface-fast 基线 | 3.550 |
| 此轮起点（GPU BVH，CPU 窄相） | 317.143 |
| GPU 窄相路径 | 112.054 |
| 完整设备 Newton 循环 | 61.225 |
| 弯曲低秩算子 | 53.109 |
| 8 方向 Galerkin 初值 | 42.934 |
| 单元线程调整与解析切线 | 35.612 |
| 分块归约与惯性预测初值 | 29.872 |
| 融合 PCG，显式线性残差验收 | 27.427 |

门槛仍为 **4.260 ms/frame**，最后一项约为基线 **7.73 倍**，不可声称性能达标。
JSON 为 `twist50_*.json`，GPU event 字段仍包含 host gaps。
以上仅是早期帧测速，不是完整扭转场景的平均性能，也不证明与 VBD 精度等价。

撤销的尝试：

- Galerkin 初值从 8 扩至 16 个方向：40.238 ms，对应 8 方向解析切线版 35.612 ms，
  当前 H 投影开销超过节省的 PCG，恢复为 8，不保留可选分支。
- 每轮 Newton 装配弹性块 CSR 并复用：30.971 ms；改转置 ELL 布局为 30.873 ms，
  均慢于融合 PCG 因子化算子的 27.427 ms。新增稀疏实现已删除，保留 JSON No-Go 记录。
  不能仅凭单次 matvec 更便宜就判断完整 Newton 更快，装配成本及舍入后的迭代数也需计入。

当前完成的连续效果回归为 **12×12、120 帧、2 秒**：

- 惯性初值版：力残差 2.14553e-5 N，间隙违反 2.96611e-8 m，互补残差 9.86611e-8 J。
- 融合 PCG 版：力残差 2.23503e-5 N，间隙违反 2.49637e-8 m，互补残差 8.57586e-8 J。
  对应 `cloth_twist_inertial_seed/` 和 `cloth_twist_fused_pcg/`。
- 25 项实验单元测试通过；不等于完整生产测试或完整 10 秒、2500 顶点验收。
  通用自接触摩擦仍缺失，不能替换原 demo。

随后完成 **50×50、120 帧、2 秒**融合 PCG 连续回归：末步 370 个活动约束，
KKT 力残差 3.68449e-5 N、间隙违反 5.25789e-8 m、互补残差 9.78995e-8 J；
见 `cloth_twist50_fused_pcg/`。这段测试通过但接触后仍很慢，不能外推早期 27.427 ms。

进一步将最终 Newton 迭代的纯弹性/惯性梯度保存，用于独立 KKT 检查，避免完整重装配。
该缓存不含 AL 力，且不被线搜索试探位置覆盖；不是使用上一物理步的旧梯度。
早期帧中位数降至 **26.056 ms**，见 `twist50_cached_kkt.json`，仍远未达标。

AL 罚系数跨 substep 直接 warm-start 首版为 **No-Go**：
`cloth_twist50_warm_al/` 在 1.4933 秒报停，间隙违反仅 1.916e-10 m，
但罚系数增长至 1.038e7，力残差 1.606e-3 N。原 continuation 仅判断相对停滞，
会在约束已达标后继续放大微小数值误差。修正为仅在间隙/互补误差仍超容差时允许加罚。
持续接触保留上一接受步罚系数，释放后恢复初始值；失败步不更新持久状态。
这一修正仍需连续场景复验；不放宽原最终力/间隙/互补阈值。

修正后的 warm AL 完成 50×50 / 120 帧：末步力残差 1.91457e-5 N、间隙违反
1.11399e-8 m、互补残差 5.91163e-8 J。与逐步重置罚系数版的全轨迹位置 RMS 差
1.44835e-6 m，最大 3.38237e-5 m；这只是两实验算法之间的差，不是对 VBD 的精度结论。
累计 step 时间约 76.7 秒（包含第一帧准备、不含渲染），末 30 帧中位数约 1989.8 ms。
早期帧速度不能代表接触阶段！

### 接触条件数与 CPU 行构建的进一步实验

重新评估此前因不精确冷启动而未采用的 reduced AL，改为在**精确速度和乘子检查点**上比较。
新的 trajectory 记录 final_v、sim_time、活动 key、乘子和 warm penalty；测速器拒绝使用
仅有显示帧差分速度的旧检查点。重启不恢复 Krylov 历史，并先完整预热一帧，两方案一致。

`barrier_free_reduced_contact.py` 在能量/线搜索中解析消去 slack：
`((max(lambda-rho*c,0))^2-lambda^2)/(2*rho)`；梯度为
`-max(lambda-rho*c,0)*J`，广义 Hessian 仅对当前激活行保留 `rho*J^T*J`。
这是修改 AL 内部求解，不是删去物理接触约束；最终 KKT/路径标准不变。
`FrozenContactRows` 保留作独立参考与其原有单元测试，twist 实验只使用 reduced 路径。

在同一 `cloth_twist50_contact_checkpoint/trajectory.npz` 输入、预热后一段 3 帧比较：

| 实验方案 | 中位 wall ms/frame |
| --- | ---: |
| frozen slack + guarded warm AL | 2349.918 |
| reduced AL | 1131.846 |
| reduced AL + 批量接触行构建 | 707.572 |
| 再加 10 顶点接触块 Jacobi（No-Go） | 833.988 |

这些是短检查点测试，不是完整 120 帧均值，GPU timeline 仍包含 host gaps。
小块预条件器保留块内耦合但未解决跨块接触条件数，不能因 PCG 次数下降就宣称提速。

完整 50×50 / 120 帧：reduced AL 累计 step 约 39.1 秒，末步力残差 3.55588e-5 N、
间隙违反 2.75338e-9 m、互补残差 1.45896e-8 J。全轨迹相对 frozen warm AL 的
位置 RMS 差 1.49605e-6 m、最大差 2.90243e-5 m。

批量行构建使用 NumPy 数组操作计算所有闭合 VT/EE 特征候选，不近似距离、不跳过端点。
独立 scalar 参考保留，新增随机/退化/平行特征、仿射间隙和零距离拒绝测试。
完整 120 帧累计 step 约 24.3 秒，末步力残差 2.07384e-5 N，
间隙违反 1.11662e-8 m，互补残差 5.92129e-8 J。
末帧仍需 18240 次 PCG、82 次 Newton 更新和 21 次 AL solve，约 591 ms；
因此仍不能替换原 VBD，性能要求远未达到。
这些累计时间来自单次运行，存在系统运行波动，不作为严谨多次重复加速比。

### 接触空间 Schur 预条件器与后续 No-Go

`barrier_free_contact_schur.py` 使用 Woodbury 公式构造
`(D_elastic + B^T B)^-1`，其中 `B=sqrt(rho*active_decay)*J`。
弹性只在预条件器中取 3×3 块对角，实际 PCG 算子仍保留完整膜、弯曲及接触耦合。
接触空间矩阵为 `I+B*D_elastic^-1*B^T`，双精度分块 Cholesky 和逆矩阵均在 GPU。
工作区限于最多 512 行；超过时仍求解全部接触，只退回块对角预条件器，不截断约束。
不支持梯度反传的这个实验模块显式禁用了 backward 编译，避免不必要的共享内存警告。

为减少重分解，仅在接触索引/激活状态/罚系数变化、Jacobian 明显变化或弹性块对角
相对变化超过 20% 时更新预条件器；只缓存预条件器，真实 Hessian 和显式线性残差不缓存。
新增 NumPy 稠密解对照，覆盖跨两个 Cholesky tile、卸载和不变状态的缓存重放。
早先 10 顶点小块预条件器新增实现已删除。

同一接触检查点的 3 帧中位耗时：

| 方案 | ms/frame | 说明 |
| --- | ---: | --- |
| 全量重建 Schur | 699.960 | PCG 约 2700–3400 次，但重分解昂贵 |
| 按变化重建 | 660.604 | 仍有串行矩阵向量内积成本 |
| 分块并行 Schur 乘法 | 520.468 | 继续保留显式 `b-Hx` 检查 |
| 32 行容量分块复用 | 347.397 | device count 表示前缀，减少重新分配和 Graph 录制 |
| 同输入 VBD surface-fast | 9.342 | 完整接触模型不同，仍不是精度等价比较 |

接触窗口相应 1.2 倍门槛为 **11.210 ms/frame**，依然明确不达标。
全量重建 Schur、缓存 Schur及容量复用版均完成过 50×50 / 120 帧；
容量复用版末步力残差 2.12455e-5 N、间隙违反 1.40412e-8 m、互补残差 2.72539e-8 J，
累计 step 约 17.28 秒，见 `cloth_twist50_bucketed_schur/`。
复用是 grow-only 的 32 行容量分块，不是已经固定了整个仿真外层 Graph 的 topology。

后续尝试与处理：

- 以内外层残差调整 AL 内部容差：迭代数基本未降，计时小幅变化不足以证明收益，撤销策略。
  Newton 容差改为 device 标量以允许重放时更新，只有一种执行路径，无 demo 配置开关。
- 上一接受步的惯性修正量作为下一步优化初值：早期帧约 21 ms，但接触检查点约 365 ms，
  完整 120 帧 17.00 秒，对照 17.28 秒，优势不明确；撤销预测器，仅保留原惯性预测初值。
- 基于 8 个历史方向的 balanced 两级 PCG 预条件：早期帧约 74 ms，PCG 下降不足以
  抵消每次预条件的双投影开销；新增模块已删除，不保留开关。
- 每次只做 1 次 primal Newton 就更新 AL 乘子：检查点在 2.0017 秒触发 swept overflow，
  未发布失败状态。改为 2 次在检查点有约 295 ms 的短期收益，但完整场景在 1.490 秒
  力残差升至 36.6 N、耗尽 240 外迭代；同样 No-Go，已恢复完整内层收敛求解。

### 可行路径停滞修正与大步长测试

将每帧 substeps 临时改为 1 的实验，在 1.4167 秒出现“线性 KKT 已收敛但路径反复拒绝”：
旧代码始终检查 `本物理步起点 -> candidate`，路径失败时也始终在旧点线性化，无法前进。
这比参考论文的分段可行路径要求更严格，也容易停滞。
参考 [论文 §4.1](https://arxiv.org/html/2512.12151v3#S4.SS1)，改为显式维护可行内部状态：
失败时尝试推进经过全量路径验证的安全前缀，随后重新线性化；保留未截断的优化状态。
报告中的 `safe` 现在指**逐段验证的连续折线路径**，不是承诺物理步起终点的直线弦也安全。
未改用论文的累计碰撞时间提前停止，最终依然必须达到原 KKT、间隙和互补阈值。
路径检查仍为数值保守推进，不是精确谓词 CCD 或完整论文保证。

不能直接把“第一个返回的失败 pair”的时间当作全局最早碰撞时间：safe_prefix 会再次检查
整段前缀，并纳入检查中遇到的其他阻塞 pair。新增更早阻塞 pair 和逐段路径回归测试。
`last_path` 保留最后一个 substep 的验证节点，仅用于检查，不作为额外物理状态输出。

修正后 1 substep 的 120 帧虽然通过，但全轨迹相对 10 substeps 的位置 RMS 差约
**34.94 mm**、最大差约 **102.78 mm**，累计 step 29.15 秒，且更慢：明确 No-Go。
默认仍为 10 substeps，没有通过减少积分次数宣称性能达标。
10 substeps 的可行路径版再次完成 120 帧，末步力残差 3.30014e-5 N、间隙违反
1.40290e-8 m、互补残差 2.71903e-8 J，见 `cloth_twist50_feasible_path/`。
当前 30 项实验测试通过；这些不是生产全仓回归，也不代表 10 秒完整 twist 已验收。

当前结论仍是：实验改进显著，但**性能/完整模型要求未达标**。全部生产文件未修改、未 stage、未提交。
另一项需要用户明确的模型差异是：VBD 使用有限刚度软接触，而本实验为硬间隙、且缺少通用
自接触摩擦。不能把两者改成同一种模型的工作伪装成单纯容差调整或已完成的精度对照。

### 用户确认：改用 VBD 软接触，独立保留几何防交叉检查

用户明确选择「对齐 VBD 软接触，仍须防自穿透」。独立实验 demo 改用
`barrier_free_soft_contact.py`；旧 AL 模块仅作为已有实验/单元测试参考，没有修改生产求解器。

实际检查了当前 `mjvbd_v2/vbd/particle_vbd_kernels.py` 的
`evaluate_self_contact_force_norm`、`damp_collision`、`compute_friction`，而不是假定接触为普通弹簧。
令作用半径 `r=0.002 m`、`tau=r/2`、`k=1000`，法向势能为：

- `d >= r`：零。
- `tau <= d < r`：`k*(d-r)^2/2`。
- `1e-5 < d < tau`：`k*tau^2*(1/2-log(d/tau))`。
- 更小距离：在 `d=1e-5` 处作二次 Taylor 延拓，保持力和曲率连续。

几何检查另用 `1e-6 m` 最小距离；BVH 候选半径仍为 `0.0035 m`。压缩接触作用厚度不再被当作
硬约束违反，但每段实际接受路径仍经过 VT/EE 保守推进检查，候选溢出/初始不可行/不收敛都拒绝发布。
这是数值防交叉检查，不宣称 exact-predicate CCD 的形式保证。没有靠放宽几何检查获得计时收益。

摩擦采用 VBD 的平滑 Coulomb 势能，`mu=0.2`、`epsilon=0.01*dt`。**载荷、切平面、重心权重在
物理子步起点冻结**，接触阻尼 `kd=0.1` 使用同一滞后接触集合；法向最近点在每次 Newton/线搜索重新计算。
这与 VBD 每 color 重算接触力并不完全相同，尤其新接触第一子步的摩擦/阻尼激活不同。
因此不能把「法向公式一致」写成「完整接触离散及轨迹完全一致」。未新增 demo 模式开关。

第一版冻结法向最近点的外层迭代在 1.4067 秒不收敛。暂时关闭摩擦/阻尼的诊断版仍在 1.7500 秒失败；
关闭它们不是可接受版本，已恢复。改成 GPU 每次求值更新法向几何后完成 50×50 / 120 帧。
独立测试覆盖 CPU/GPU 闭合最近点特征、法向各分段能量导数、完整接触+摩擦有限差分梯度、耦合 PSD 曲率，
以及单顶点压入三角面的场景：允许 `d<0.002`，但不允许穿过表面，逐段复查接受路径。

同一个 2 秒检查点（完整预热一帧，然后 3 帧）的中位 wall time：

| 软接触实验 | ms/frame | 说明 |
| --- | ---: | --- |
| 当前距离 + CPU 行准备 | 592.35 | `twist50_contact_soft_nonlinear.json` |
| GPU 行准备 + 容量复用 | 368.65 | `twist50_contact_soft_gpu_rows.json` |
| 批量接触键合并、消除无效重解 | 200.51 左右 | `twist50_contact_soft_vector_keys.json`，精确值以 JSON 为准 |
| 精确平滑摩擦 Hessian | 124.47 左右 | `twist50_contact_soft_exact_friction.json`，精确值以 JSON 为准 |

候选增加后重新计算完整梯度，只有新增候选确实产生未平衡力才重解；不是跳过新增接触。
新 Newton 摩擦 Hessian 是同一平滑势能的精确半正定曲率，取消了滑动方向不必要的正曲率。
**相同能量/力/收敛阈值不变**；不同于直接削减摩擦或使用额外速度衰减。
3 帧仅为短检查点计时，不替代长轨迹验收；VBD 同检查点约 9.342 ms，20% 上限约 11.210 ms，仍远未达标。

长测记录：

- 保守摩擦曲率版：在 4.1450 秒拒绝，残差 `2.3771e-4 N`，并未完成 600 帧。
- 精确摩擦曲率版：到 7.0333 秒拒绝，残差 `1.0062e-3 N`，`line_search_failed`；
  见 `cloth_twist50_soft_exact_friction/`。不能仅凭完成前 120 帧宣称完整通过。

随后修复数值一致性：弹性力中的 pose 系数求和、面积乘法、弯曲梯度不再过早取单精度；
弹性和新软接触的梯度在 FP64 累加后再转 FP32 交给 PCG，Hessian 和线性向量仍为 FP32。
只有双精度力累加仍会在 7.0333 秒检查点停滞（残差约 `3.4549e-4 N`）。
总能量约 74 J，此时直接比较两个总能量会淹没很小的下降量；改为逐单元/逐接触先求差再归约，
仍使用原 Armijo 不等式，没有放宽力阈值。对不提供逐项能量的旧测试 term，保留其总能量比较。

逐项能量差版通过该失败检查点后续 4 帧（包含预热帧），测量中位约 `421.265 ms/frame`，
见 `twist50_soft_relative_energy.json`。这与 2 秒检查点不是同一状态，不能把两组时间横向当成增减幅。
该严格求解版最终在 8.3717 秒拒绝，残差 `3.47595e-4 N`，未完成 600 帧。

### 后续精度诊断与有限 Newton 接触流

进一步将 VT/EE 内部最近点的行列式改成叉积平方，避免近乎平行时点积差消失。
新增夹角 `1e-8` 的 EE 内部最近点测试。修正后的严格版在 8.2167 秒停止，
残差约 `5.95e-4 N`；因此这些数值修正不能等同于完成长轨迹验收。
另记录独立的 `position_precision` 停止条件：全局 PCG 候选更新不超过
`16*eps(float32)*radius`（约 3.815 nm），同时保留原线性残差有效性检查。
该条件不是力残差收敛，报告保留实际力残差，不能混称。

单步精度需要区分接触集合。原 VBD 对照默认 topology threshold=2，实验只排除 incident pairs。
曾尝试简单一环邻接过滤，但它不等价于官方 EE 双向交集过滤，已撤掉。
早期估算的“642 个活动 pair 被过滤”只是简单邻接掩码估计，不是官方 CSR 的精确计数。
在 7.0333 秒实验检查点的一次 VBD 单步回放中，VT `[230,280,281,330]` 的有符号面距
从 `+0.214612 mm` 变成 `-0.075954 mm`，交点在三角形内部；不能为了复现这个过滤而
删除实验几何检查。这也不代表原始生产 demo 整段轨迹都发生同一交叉。

`twist50_matching_response_quality_7s.json` 加入 topology=1 的 VBD **精度诊断对照**。
相对严格全局参考，该 VBD 单步 RMS 差约 15.37 µm，Newton 2 步约 0.099 µm，均通过该步路径检查。
摩擦滞后/边界离散仍不同；性能基线继续使用原 VBD 配置，不用修改后的拓扑偷换基线。
还需补全 VBD row overflow 记录和长轨迹质量评价，不能仅靠这一帧宣布效果相同。

当前独立 demo 使用有限 Newton，每块 2 次；**不要求每物理子步达到 `2e-4 N` 力收敛**。
接受条件变为：有限值、无候选/接触流 overflow、逐段几何安全、完整接触集合下增量势能
不高于初始可行状态、刷新候选造成的梯度变化范数不超过 `2e-4 N`。未通过则继续块迭代，
最多 240 块后拒绝发布。报告标记 `inexact_newton_energy_decrease`，不是严格 Newton 收敛。
这是一项明确的求解停止语义变更，不归类成“只优化 kernel、完全不影响精度”。

`cloth_twist50_stream_inexact2/` 首次完成 50×50 / 600 帧 / 10 秒。
去掉首帧编译后平均 86.667 ms/frame，中位 88.447 ms/frame，末步力残差约 0.275 N。
相对严格版共有前 493 帧，轨迹 RMS 差约 5.024 mm，末个公共帧 RMS 约 13.171 mm。
因此仅说明这版完整运行且接受路径通过检查，不说明已达到严格版精度。
原 VBD 同场景 600 帧计时：`twist50_vbd_full600.json`，去掉预热帧平均 11.716 ms/frame，
中位 12.877 ms/frame；按平均时间的 +20% 门槛为 14.059 ms/frame。**性能依然不达标。**

### 设备接触流和有覆盖界限的候选缓存

接触流固定最大容量、device-side count，采集/接受条件用 CUDA Graph；无需每子步下载、排序、
合并再上传 CPU 接触键。行求值和矩阵乘改为 4096 persistent workers 遍历 active prefix。
数量收缩时清除旧能量；溢出不能通过复用缓存变成“有效”。新增流采集、新接触力复核、
超过 4096 行、数量归零、溢出和软压缩防穿透测试。

原分项 profile `stream_stage_profile.json`：一次候选+几何检查约 0.664 ms，粗矩阵构造约
0.474 ms，完整梯度/曲率组装约 0.180 ms；这是隔离 Graph 时间，不能直接相加当整帧时间。
有限两步 Newton 中暂不使用 aggregate 预条件器：同状态 10 帧对照，中位 68.793 ms
（aggregate）对 61.091 ms（block diagonal）。严格参考实验仍可用 aggregate。
移除的只是预条件器开销，细层目标函数/Hessian/线性有效性检查不变。

惯性预测点及仅用于接触集合的包围查询只执行 broad phase；每个实际接受运动段仍做完整
窄相几何检查。`collect_near=False` 的几何路径不再重复生成 CPU 不使用的 endpoint near 列表。

候选缓存使用额外 `skin=0.002 m`，不改变接触半径或接触能量。重建时查询半径
`0.0035 + skin`，缓存旧 start/end；只有每个顶点的新 start/end 相对各自参考点的偏移
均小于 `0.49*skin` 才复用。两个 primitive 的相对偏移至多 `0.98*skin`，因此当前路径中
距离低于 0.0035 m 的 pair 必在旧 swept AABB 的扩张查询中。超界则设备端 conditional
Graph 刷新。缓存只覆盖 broad phase，最近点、接触力和接受路径检查仍使用当前位置。
额外候选可能增加工作；固定容量溢出始终拒绝，不能截断接受。

同一个 2 秒检查点，预热 1 帧、测量 10 帧：

| 版本 | 中位 wall ms/frame | 记录 |
| --- | ---: | --- |
| active-prefix + aggregate | 68.793 | `twist50_stream_aggregate10.json` |
| block diagonal | 61.091 | `twist50_stream_diagonal_diagnostic.json` |
| 额外 skin 候选缓存 + 查询拆分 | 52.881 | `twist50_stream_skin_cache.json` |

这是局部计时，不替代后半程稳定性/精度验收；不能与 600 帧全程平均时间混比。
当前生产求解器文件未修改，未 stage、未提交。

缓存版完整测试 `cloth_twist50_stream_skin_cache/` 完成 600 帧 / 10 秒，所有接受路径均经过检查。
排除首帧后平均 **74.176 ms/frame**、中位 **75.199 ms/frame**、p95 **124.161 ms/frame**。
对前一接触流全程平均 86.667 ms，下降约 14.4%，但仍显著超过原 VBD 的 +20% 性能门槛。
与严格版共有的前 493 帧 RMS 差约 **0.759 mm**，最大差 **26.561 mm**，末个公共帧
RMS 差 **0.552 mm**。这是单次轨迹对照，不能把相对上次的巨大变化全归因于缓存“提升精度”；
接触原子累加顺序和预条件器也改变了有限迭代轨迹。末步力残差 **1.319 N**，仍不能报严格收敛。

随后试验 8 列历史方向批量 Hessian 乘法：单项 recycled-guess 从约 0.191 ms 到 0.141 ms，
但 2 秒检查点 10 帧整体无明显收益（`twist50_stream_batch_history.json`），且普通 matvec
隔离时间变差。批量/逐列乘法一致性测试通过，但性能证据不足，已撤销批处理实现及其专用测试，
没有留下一个额外运行模式。保留当前更简单的逐列乘法。

最终复查：41 项独立实验单元测试通过，所改实验脚本 Ruff 检查通过；生产 tracked diff 和
staged diff 均为空。这不是全仓 demo 回归或生产可用性认证，性能与完整质量验收仍未通过。
