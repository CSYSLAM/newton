# 高精度耦合求解器（SolverCoupledHighAccuracy）纯 Warp 移植计划

分支：`coupled-high-accuracy`（基于 newton main `03b36585`）。仅本地研究使用，不开源。

## 1. 目标

1. 把 engineRTS 中的 `style3d_sim`（Python 适配层 + 原生 CUDA `_core`）完整移植进 Newton，
   作为独立求解器 `newton.solvers.SolverCoupledHighAccuracy`。
2. 求解器核心全部用 Warp 重写，不再依赖任何原生扩展（`_core.so`、pybind11、CMake、cub/thrust）。
3. 在 `newton/examples/coupled_high_accuracy/` 复现 engineRTS 中基于 style3d_sim 的全部 example 与 demo，
   demo 原有的 `validate.py` 检查全部通过。

### 验收标准

| 项目 | 标准 |
| --- | --- |
| 功能 | engineRTS 905 个测试移植后在 Warp 后端全部通过，不加载 `_core` |
| 一致性 | 与原生后端逐步对照：同一场景、同一输入，状态差异在浮点容差内（不要求逐位一致，求和顺序不同） |
| 示例 | 全部 example 可运行；demo 的 `validate.py` 全部通过，Blender 渲染效果与原版一致 |
| 性能 | 主要场景每帧耗时不超过原生的 1.3 倍（目标），以 engineRTS dev1118 并发 PCG 为基准 |
| 独立性 | 卸载 style3d_sim 后 Newton 仍能完整运行求解器、测试和示例 |

## 2. 原生实现结构（移植对象）

`engineRTS/reconstruction/csrc`，约 1.4 万行：

| 部分 | 规模 | 内容 |
| --- | --- | --- |
| `native/source_solver.cpp` | 3.5k 行 | 主机端 `Solver`：`initialize` 预计算（模板稀疏结构、BVH、弹簧/布料/四面体/仿射块/线缆/颗粒/流体预处理）和 `step` 调度 |
| `kernels/*.cu` + `*.cuh` | 约 8.8k 行，176 个 kernel | 各物理项的梯度、Hessian、矩阵无关切线积、PCG、接触、BVH |
| `native/bvh_library.cu`、`library/*` | 约 0.4k 行 | LBVH 构建、cub 排序/规约、thrust 填充 |
| `revisions/*_23fb797.cu` | 0.6k 行 | SDK 23fb797 版本的 PCG（按 world 分别求 CG 系数）、四面体、颗粒网格 |

对外接口很小：`Solver.initialize(model) / step(in, out, control, contact, dt, stream) /
reserve_rigid_contact_capacity / reset_joint_drive_state / reset_cable_penalty_state / copy_cable_penalties /
config / launch_counts`，数据结构 `Model`（150 个字段）、`State`、`Control`、`Contact`、`Config`、`Span`，
以及 5 个枚举。适配层 `newton_solver.py` 通过 `Span(ptr, size)` 把 Warp 数组的设备指针交给原生代码。

### 每步算法（`Solver::step`）

刚体以仿射体表示：每个刚体 4 个代理粒子（12 自由度），刚性由 SO(3) 增广拉格朗日约束保持；
布料、软体、线缆、颗粒、流体也都是粒子。整个系统是一个统一的隐式欧拉问题：

1. 惯性预测 `start`：`target = q + dt·v + dt²·g`，记录 `previous = q`；刷新仿射体惯量非对角块；
   初始化距离约束、SO(3)、关节弹簧/点线约束的乘子。
2. 非线性迭代 `nonlinear_iterations` 次，每次：
   - 组装右端项 `rhs`：惯性项 + 外力 + 流体、颗粒、风阻、刚体接触（软/硬）、滚动摩擦、全表面接触、粒子接触、
     自碰撞、刚体弹簧、距离约束、关节弹簧、点线约束、关节驱动、电枢、角度限位、线缆、SO(3)、弹簧、膜、四面体、弯曲。
   - 组装 3×3 对角块 `blocks` 和 ELL 格式的非对角块（模板 + 膜/弹簧/刚体弹簧/点线 Hessian），
     再叠加预条件器专用项，求块逆 `inverse`；可选 12×12 仿射块预条件器 `affine_build`。
   - PCG `linear_iterations` 次：每个 world 单独计算 CG 系数；矩阵向量积 = ELL 显式部分 +
     各物理项的矩阵无关切线积（`*_tangent`，原子加到 `contact_product`）。
   - `q += solution`；更新硬接触、距离约束、SO(3)、关节乘子（按 primal 迭代次数推进）。
3. 输出接触力；`velocity = (q - previous) / dt`；更新关节角度分支。

## 3. 实现方式

### 3.1 同接口替换 `_core`

在 Newton 中用纯 Warp 实现一个与原生 `_core` **接口完全相同**的模块 `core/`：

- `Model / State / Control / Contact / Config / Span` 和枚举用 Python 实现，字段名、默认值与原生一致；
  `Span(ptr, size)` 在 Warp 侧用 `wp.array(ptr=..., dtype, shape, device)` 零拷贝包装成数组。
- `Solver` 用 Python 调度 Warp kernel，`initialize` 的预计算用 numpy/Warp 实现，`step` 全部是 Warp launch，
  可被 CUDA graph 捕获。
- 适配层（原 `newton_solver.py` 等）原样迁入，只把 `from ._core import ...` 换成可选后端：
  `backend="warp"`（默认）或 `backend="native"`（仅开发期对照，导入 engineRTS 的 `style3d_sim._core`）。

这样同一场景、同一个适配层，可以随时切换两个后端逐步对比，定位到具体物理项。

### 3.2 kernel 移植约定

- 每个 `kernels/*.cu` 对应一个 Python 模块（`kernels/cloth.py` 等），`*.cuh` 的设备函数改为 `@wp.func`。
- 按功能精确翻译，保持原生的数值公式、迭代顺序和分支条件；只调整 Warp 无法表达的实现细节
  （cub 排序/规约 → `wp.utils.radix_sort_pairs`、`array_scan`、原子规约；warp shuffle → tile 或原子）。
- 原生中“为兼容旧 SDK 而保留”的分支（PD_NEWTON、SIX_SPRINGS、构成律刚性、罚函数关节、
  `reference_linear_schedule` 等）也一并移植，因为测试覆盖了它们；无测试覆盖的分支在代码中标注。
- 遇到暂未移植的功能，`initialize` 直接抛出 `NotImplementedError`（同原生的 `reject_model`），
  不会静默得到错误结果。

### 3.3 对照与测试

- 对照工具 `scripts/coupled_high_accuracy/compare_backends.py`：同一场景分别用两个后端运行，
  逐步比较 `particle_q / particle_qd / body_q / body_qd` 的最大差异；可把迭代次数设为 1 来对照单次组装和单次 CG。
- engineRTS 的 905 个 pytest 测试移入 `newton/tests/coupled_high_accuracy/`，保持 pytest 形式
  （该目录不是 Python 包，不会被 Newton 自带的 unittest 发现机制收集），用 `uv run pytest` 运行。
  `import style3d_sim as s3d` 改为导入 Newton 中的同名模块。
- 测试按后端参数化：功能移植完成之前只在原生后端运行，完成后切到 Warp 后端。

### 3.4 Newton 侧布局

```
newton/_src/solvers/coupled_high_accuracy/
    __init__.py                          # 公开 API：SolverCoupledHighAccuracy、ModelBuilder、Config、枚举等
    solver_coupled_high_accuracy.py      # 适配层（原 newton_solver.py）
    builder.py                           # 原 newton_model_builder.py（ModelBuilder 子类）
    cable_material.py  rigid_body_dynamics.py  joint_drive_effort.py
    granular.py  fluid.py  particle_surface.py  assets.py
    core/
        api.py                           # Model/State/Control/Contact/Config/Span/枚举
        solver.py                        # Solver：initialize + step 调度
        native.py                        # 可选原生后端加载（开发期对照）
        kernels/                         # 每个 .cu 一个模块
newton/tests/coupled_high_accuracy/      # 移植的 pytest 测试
newton/examples/coupled_high_accuracy/   # 移植的示例与 demo
```

公开名称：`newton.solvers.SolverCoupledHighAccuracy`、`newton.solvers.coupled_high_accuracy`。
原名 `NewtonSolver` 作为别名保留，方便直接迁移 engineRTS 代码。

## 4. 分阶段计划

每个阶段有必须达到的验收门槛，达到后再进入下一阶段。

| 阶段 | 内容 | 验收门槛 | 预计 |
| --- | --- | --- | --- |
| P0 基础设施 | 目录与公开 API；适配层迁入并接原生后端；修复 main 的接口变化（rod、`use_coord_layout_targets`、`SolverVBD` 参数）；测试迁入；对照工具 | 原生后端下 905 个测试全部通过 | 0.5～1 天 |
| P1 最小闭环 | `api.py`、`Solver` 骨架、惯性预测、Newton 迭代、ELL 组装、块预条件 PCG（按 world 系数）、`step.cu`、`springs.cu`、`drag.cu`、速度恢复 | 粒子自由落体、弹簧、风阻场景与原生一致 | 2 天 |
| P2 可变形体 | `cloth.cu`（各向异性膜、弯曲及阻尼）、`tetrahedra.cu`（含 23fb797 版本）、粒子质量非对角耦合 | 布料、膜收敛、非流形弯曲、软体测试通过 | 1.5 天 |
| P3 仿射刚体与关节 | `affine.cu`、`affine_block.cu`、`joints.cu`（刚体弹簧、距离约束、点线、增广拉格朗日关节、驱动、电枢、限位、角度分支）、reset 与 world 掩码 | 关节、MuJoCo 语义、驱动反馈、限位与 reset、旋转刚体测试通过 | 3 天 |
| P4 碰撞 | LBVH、粒子-形状软接触、全表面接触、刚体接触（软/硬、历史、滑动切向）、滚动摩擦、自碰撞 | 接触正确性、法向响应、滚动摩擦、硬接触、自碰撞测试通过 | 4 天 |
| P5 线缆 | `cables.cu`、`cable_evaluate.cu`、`cable_tangents.cu`、罚函数爬升、Dahl 材料 | 全部 cable 测试通过 | 1.5 天 |
| P6 颗粒与流体 | `granular.cu`（含网格、粘附、动态键）、`fluid.cu`、粒子表面 | 颗粒、流体、粒子表面测试通过 | 1.5 天 |
| P7 性能 | 融合 PCG、多 stream 并发切线积、整步 CUDA graph、tile 规约；与原生的基准对比 | 主要场景每帧 ≤ 原生 1.3 倍 | 3 天 |
| P8 示例复现 | 见第 5 节 | demo `validate.py` 全部通过，渲染一致 | 3 天 |

合计约 3～4 周连续工作。风险最大：P3 关节、P4 接触、P7 性能。

## 5. 示例与 demo

| 依赖阶段 | engineRTS 来源 |
| --- | --- |
| P1～P3 | dzhanibekov、spin、ball、basic_joints、joints、body_spring、distance_constraint、body_point_line、chain_heavy_ball、physical_pendulum、rigidbody_integrators、basic_shapes |
| P2 + P4 | cloth_hanging、cloth_balls、ball_cloth、cloth_style3d、cloth_particles、plastic_bag、ball_brick_wall |
| P3 + P4 机器人 | robot_allegro_hand、robot_ur10、robot_panda_hydro、robot_h1、robot_g1、robot_anymal_d、robot_cartpole、robot_microduck、cloth_h1、cloth_h1_free、robot_cloth_trampoline、robot_policy、robot_cloth_policy |
| P5 | `examples/cable/*` |
| P6 | granular_beaf、granular_bread_three_pieces、snow、fluid |
| 综合 demo | g1_jelly_drop、piper_bag_transport、w1_courier_transport、w1_pharmacy_carton、w1_pharmacy_variants、w1_snack_basket |

每个示例注册到 `newton.examples` 并带 `test_final` 检查；demo 的检查脚本与 Blender 渲染脚本一并迁入；
资产引用本地 engineRTS 的 `assets/`（仅本地研究）。

## 6. 进度记录

| 日期 | 阶段 | 结果 |
| --- | --- | --- |
| 2026-11-18 | 依赖 | newton venv 按锁文件同步（mujoco 3.17.0、warp 1.20.0.dev20261117），`example_mujoco_flex` 正常运行 |
| 2026-11-18 | 摸底 | style3d_sim 原样在 newton main 上运行：905 个测试中 843 通过，61 个失败均为 main 接口变化 |
| 2026-11-18 | P0 | 适配层迁入 `newton/_src/solvers/coupled_high_accuracy/`，公开 `newton.solvers.SolverCoupledHighAccuracy`；`core/` 按 `NEWTON_CHA_BACKEND` 选择后端（Warp 后端尚未实现，暂默认 `native`）。39 个测试文件迁入 `newton/tests/coupled_high_accuracy/`：原生后端 699 通过、4 跳过（等线缆示例迁入） |
| 2026-11-18 | P0 | 适配 main 的接口变化：rod joint 由 4 个 DOF 改为 6 个（剪切 x/y、拉伸 z、弯曲 x/y、扭转 z），线缆材料按 `ROD_MATERIAL_DOFS` 取值；`JointType.CABLE` → `ROD`；测试改用 `newton.Rod`、`add_joint_rod`、`target_q`、`CollisionPipeline`。与 Newton 1.7 `SolverVBD` 对照的两组测试改为只检查解析期望值（main 的 VBD 已更换接触与罚函数模型） |
| 2026-11-18 | P0 | 暂缓迁入（依赖 example/demo，P8 一并迁入）：beef_surface、motor_telemetry(_plot)、nonmanifold_bending、particle_surface、trampoline_mesh、ur10_joint_scan、locomotion_g1、robot_policy_target_layout、w1_courier_grasp_motion、w1_mimic_compat、w1_sdf_filter。不迁入（engineRTS 构建/登录/Windows 专用）：authentication、local_credentials、platform_sources、standalone_selection、w1_courier_acceptance、windows_upstream_comparison、render_provenance、cable_profile_evidence |
| 2026-11-18 | P1 | Warp core：`core/api.py`（由原生头文件生成的同名字段与默认值，`Span` 零拷贝包装）、`core/solver.py`（initialize 与 step 调度：惯性预测、右端项、块对角/ELL 组装、按 world 系数的块预条件 PCG，含融合与参考两种调度）、kernels `step/pcg/springs/drag`。对照工具 `scripts/coupled_high_accuracy/compare_backends.py`：自由粒子与原生逐位一致；弹簧 Newton 位置差 4e-9；弹簧 PD 位置差 3.6e-7，与原生自身两次运行的差异完全相同（原子加求和顺序）。Warp 后端 175 个测试通过，其余均为后续阶段功能，`initialize` 明确报未移植 |
| 2026-11-18 | P2 | 布料（各向异性膜力与 Gauss-Newton Hessian、余切弯曲及阻尼的预条件与切线）、四面体（力、预条件、切线；激活量）、风阻。主机端稀疏模板 `core/templates.py` 按原生 `std::map` 顺序做 float32 累加。对照：单步误差在最后一位（位置差 ≤ 5e-13），30 步后布料/四面体位置差约 1e-6（原生自身两次运行约 1e-7）。Warp 后端 190 个测试通过 |
| 2026-11-18 | P3 | 仿射刚体（SO(3) 增广拉格朗日、六弹簧、ABD 正交、共旋构成律）、12×12 仿射块预条件器（局部 Cholesky）、惯性模板（L0/L1/L2/L4/L5/L6 组合与按步刷新）、刚体弹簧、距离约束、点线罚函数、关节点线增广拉格朗日、关节驱动（旋转/平移，角度展开）、电枢、角度限位、驱动状态 reset。对照：单刚体 6 种配置、三连杆摆 8 种配置（含 PD、块预条件、罚函数关节、平移驱动、电枢、限位）30 步位置差 1e-7～1e-5；PD 摆在 CG 收敛后单步差 1 ULP。Warp 后端 348 个测试通过，其余为 P4 及以后功能 |
| 2026-11-18 | P4a | 粒子-形状软接触（静态形状与刚体上的形状，力、标量对角、各向异性块修正、非对角切线积、接触力输出）、刚体惩罚接触（含平滑摩擦 epsilon 模型）、滚动/扭转摩擦。对照工具新增 `--replay`（每步从原生状态出发只走一步）：布料落地、粒子撞刚体、箱子落地/叠放、滚动摩擦等场景单步误差 ≤ 1.2e-7。平滑摩擦场景 50 步后有一次分支翻转导致毫米级差异，回放证实每步均与原生一致。Warp 后端 447 通过，251 个失败全部为未移植功能 |
| 2026-11-18 | P4b | 硬接触（增广拉格朗日罚函数与对偶、Coulomb 锥投影、历史匹配热启动、精确滑动切向、接触块 12×12 交叉项、容量预留与快照）。箱子落地场景 5 种配置 60 步位置差 ≤ 2.5e-6。Warp 后端 480 通过，218 个失败全部为未移植功能 |
| 2026-11-18 | P4c | 全表面软接触（三角面重心点对静态/刚体形状，法向与摩擦两条雅可比列表，力、刚度缩放、预条件块、切线积，按牛顿迭代缓存切线供 CG 复用）。布料落地（全表面）、布料搭在刚体箱子上（含 PD、滚动摩擦、硬接触）回放单步误差 ≤ 2e-6；布料搭箱场景对舍入敏感，原生自身两次运行即差 5e-5。修正：对照场景脚本的两处替换没有落到预期位置，重新核对后补齐。Warp 后端 485 通过，213 个失败全部为未移植功能（自碰撞、颗粒、线缆、流体） |
| 2026-11-18 | P4d | 自碰撞：Karras LBVH（Morton 码、稳定基数排序、基数树、逃逸链接）、顶点-面/粒子-粒子/边-边/边-面候选查询（每图元 31 个，原生遍历顺序）、罚函数力与 Hessian 累加、全切线与对角切线。差异：Warp 无内存栅栏，refit 改为按节点高度逐层合并（结果与原生的原子计数 refit 相同）；无 FMA，两处补偿乘积差改用 float64 计算。布料自折叠、粒子堆场景回放单步误差 ≤ 4e-7，确认自碰撞接触确实发生。P4 完成：Warp 后端 559 通过，139 个失败全部为颗粒、线缆、流体 |
| 2026-11-18 | P5 | 线缆：惯量加权极分解（float64 缩放 Newton 迭代）与自旋映射、拉伸/剪切/弯曲/扭转应变及雅可比（含反平行区的输运导数）、每次牛顿迭代预计算的 8 槽雅可比、力/切线/预条件块、罚函数爬升与 reset/copy、12×12 仿射块中的线缆项、关节所有线缆的调度。Warp 后端 cable 测试 118 个全过；对照：CG 收敛（200 次迭代）时单步差 3.6e-7，默认 50 次时该刚性杆 CG 未收敛、舍入放大到 1e-5。Warp 后端 609 通过，89 个失败全部为颗粒、流体 |
| 2026-11-18 | P6 | 流体：空间哈希网格（流体粒子 + 静态边界采样）、64 邻居行缓存与超出时的二次遍历、密度约束力/对角/无矩阵切线、粘性、按世界的步长限制器；单步差 ≤1.5e-8（含超 64 邻居路径）。颗粒：颗粒接触（罚函数法向 + 正则化库仑摩擦，32 槽缓存）、作者键（内聚）与动态键（粘附）的弹塑性/断裂、提交与成键（贪心最近邻，排除作者键 CSR）；单步差 ≤3e-8（含多世界、粘附、PD）。粒子表面是纯 Python/Warp 的宿主工具，无需移植。Warp 后端全部测试 698 通过、5 跳过，求解器功能移植完成 |
| 2026-11-18 | P7a | 整步 CUDA graph 捕获：一步有 600–1500 次小核启动，Python 启动开销（约 15 µs/次）使 Warp 慢 5–10×。现按全部输入指针/标量、dt、配置、流以及求解器自有数组地址做键缓存 graph（首步即时执行做校验与懒分配，第二步捕获，之后重放；调用方自己在捕获时直接录入；`NEWTON_CHA_GRAPH=0` 关闭）。`benchmark_backends.py`：Warp/原生耗时比 cloth 1.03、tetrahedra 0.71、cloth_self 0.91、cloth_on_box 1.44、boxes_on_ground 1.09、cable 0.86、fluid_drop 1.07、granular_heap 1.13。测试 698 通过，对照结果不变 |
| 2026-11-18 | P7b | 表面接触核按原生做法限制网格为 SM 数 ×256 并 grid-stride；缓存切线乘积改为定长循环（7 项列表留在寄存器）。cloth_on_box 由 1.44× 降到 1.30–1.37×，其余场景 0.7–1.14×。尝试过但无收益而撤回：软接触乘积限 32 块（Warp 侧更慢）、刚性接触切线 8×8 展开（代码膨胀更慢）。剩余差距主要在刚性接触切线（约 9 µs/次 vs 原生 5.4 µs）。nsys 对原生只记录到约 1/5 的核，比较时应以每次平均耗时为准 |
| 2026-11-19 | P7c | 刚性接触切线：每次牛顿迭代缓存一次雅可比列表与刚度（扁平标量数组；结构体内向量动态下标会落到 384 B 局部栈），CG 内只做乘积；法向与摩擦两部分合并为每目标一次原子加（同一刚体的接触都打在 4 个代理粒子上，瓶颈在原子竞争）。所有核模块关闭 backward 生成（求解器不求导，编译时间减半）。基准改为交替多次取最小值（GPU 与其他任务共享）。15 个场景 Warp/原生 = 0.68–1.20（cloth_on_box 1.05、boxes_on_ground 0.99、granular_heap 1.20、pendulum 1.17），达到 ≤1.3× 目标；测试 698 通过，对照仍为 ULP 级 |

