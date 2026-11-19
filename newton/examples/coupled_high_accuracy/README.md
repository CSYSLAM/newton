# 高精度耦合求解器示例

engineRTS 中 `style3d_sim` 的示例与 demo 在 Newton 里的复现，全部运行在 `SolverCoupledHighAccuracy`（默认 Warp 后端）上。

## 目录

| 位置 | 内容 |
| --- | --- |
| `example_cha_*.py` | engineRTS `examples/` 与 `examples/cable/` 的示例，加 `cha_` 前缀以免与 Newton 自带示例重名 |
| `cable_utils.py`、`_cable_viewer.py`、`cable_examples.json` | 线缆示例共用的求解器配置与可视化 |
| `snow_interaction.py`、`microduck_assets.py`、`trampoline/`、`locomotion*/` | 示例依赖的辅助模块与训练脚本 |
| `demos/` | engineRTS `demos/`：`demo/` 下为主入口，其余目录为各场景的运动、几何、导出、渲染与验证 |
| `assets/` | engineRTS `assets/` 中受版本管理的资源（W1、果冻、面包、陀螺、塑料袋等）；Microduck 模型由 `microduck_assets.py` 按需下载 |

示例自动注册到 `newton.examples`：

```bash
python -m newton.examples cha_ball
python -m newton.examples cha_cable_twist --viewer null --test
```

demo 直接运行主入口：

```bash
python newton/examples/coupled_high_accuracy/demos/demo/demo_spin.py --viewer null --num-frames 100
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_snack_basket.py --viewer=false --output build/snack
```


## 迁移方式

`scripts/coupled_high_accuracy/port_example.py` 做机械迁移：`style3d_sim` 导入改为 `newton.solvers.coupled_high_accuracy`，
`NewtonSolver` 改为 `SolverCoupledHighAccuracy`，示例辅助模块改为包内导入，并删去 Newton main 已移除的
`newton.use_coord_layout_targets`（关节位置目标总是与 `joint_q` 同布局）。其余按 Newton main 的 API 手工调整：

- 线缆：全部改用 Newton main 的 `newton.Rod` + `ModelBuilder.add_rod(rod=...)`，质量按 Newton 的 rod 模型（圆柱段）。
  静态验证类示例（`bend_stiffness`、`bend_twist_analytic`、`torsion_material_mapping`）原版每子步只做 1 次牛顿迭代、1–5 次
  CG，停在离平衡点约 8 倍、且随刚体质量变化的截断不动点上（原版能过 `bend_stiffness` 是偶然）；现在用 4 次牛顿、50 次
  CG，结果收敛到与质量无关的平衡，三例均通过（后两例原版不通过）。
- 全表面网格接触：Newton main 对网格报告接触带内所有点-面、面-点、边-边记录，由求解器用 `_soft_contact_force_enabled`
  选出承担力的记录（其余仅供防穿透）。求解器每步按 Newton 的规则筛选并压缩后再交给核心（`mesh_contact_filter.py`），
  否则会把同一接触片的多重表示（约 9 成）都当作独立罚接触。
- `demos/utils/sdf_candidate_filter.py`：Newton main 的 SDF 软接触核改为工厂生成并自带候选压缩，过滤器在此退化为直接调用上游。
- 只在 Windows 源码构建下使用的 `run_source.py`、`source_runtime_guard.py`、`source_scene_runtime.py` 未迁入。
- `example_franka_cable_ik_pick_place`、`example_mujoco_franka_vbd_cable_admm_solver` 使用 Newton 实验性耦合求解器（MuJoCo + VBD/XPBD），
  与本求解器无关，未迁入。

## 验证

每个示例与 engineRTS 原版在各自环境中以 `--viewer null --test`（默认帧数）运行，要求结论一致（同为通过，或同为相同的失败）。
部分原版示例在默认帧数下本身不满足其 `test_final`（如 `basic_joints`、`basic_shapes`、`body_spring`），或未实现 `test_final`
（`body_point_line`、`joints`）；它们在 Newton 中的结论与原版相同。

性能：robot_cloth_policy 每帧约为原版的 2.2 倍，其中约 80% 来自 Newton main 的网格接触检测（每帧 GPU 196 ms，
Newton 1.7 为 42 ms；新算法对网格穷举特征对，同一场景从 4.6k 条记录变为约 20 万条），求解器部分为原版的 1.14 倍。
Newton main 中所有与粒子碰撞的网格形状在开启全表面接触时都走这条路径，没有退回旧 SDF 最小化的开关；要降低这部分
开销需改场景（例如用凸体/基本形状做碰撞，或对网格只保留逐粒子接触）。逐例结果记录在仓库根目录的 `COUPLED_HIGH_ACCURACY_PLAN.md`。
