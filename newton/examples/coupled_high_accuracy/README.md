# 高精度耦合求解器示例

engineRTS 中 `style3d_sim` 的示例与 demo 在 Newton 里的复现，全部运行在 `SolverCoupledHighAccuracy`（默认 Warp 后端）上。

## 目录

| 位置 | 内容 |
| --- | --- |
| `example_cha_*.py` | engineRTS `examples/` 与 `examples/cable/` 的示例，加 `cha_` 前缀以免与 Newton 自带示例重名 |
| `cable_utils.py`、`_cable_viewer.py`、`cable_examples.json` | 线缆示例共用的求解器配置与可视化 |
| `rod_compat.py` | Newton 1.7 的 `add_rod(positions=...)`、`add_rod_graph()` 与 `newton.utils` 线缆工具；保留原场景参数与胶囊质量 |
| `snow_interaction.py`、`microduck_assets.py`、`trampoline/`、`locomotion*/` | 示例依赖的辅助模块与训练脚本 |
| `demos/` | engineRTS `demos/`：`demo/` 下为主入口，其余目录为各场景的运动、几何、导出、渲染与验证 |
| `assets` | 指向本地 engineRTS `assets/` 的软链接（不提交，仅本地研究） |

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

首次使用先建立资源链接：

```bash
ln -s ~/code/agent/engineRTS/assets newton/examples/coupled_high_accuracy/assets
```

## 迁移方式

`scripts/coupled_high_accuracy/port_example.py` 做机械迁移：`style3d_sim` 导入改为 `newton.solvers.coupled_high_accuracy`，
`NewtonSolver` 改为 `SolverCoupledHighAccuracy`，示例辅助模块改为包内导入，并删去 Newton main 已移除的
`newton.use_coord_layout_targets`（关节位置目标总是与 `joint_q` 同布局）。其余按 Newton main 的 API 手工调整：

- 线缆：Newton main 删除了旧的 rod 接口，且 rod 刚体只计圆柱段质量。`rod_compat` 调用新旧版本共用的链/图组装，
  并恢复整段胶囊（含两端半球）的质量与惯量，使悬臂挠度与原版一致。
- `demos/utils/sdf_candidate_filter.py`：Newton main 的 SDF 软接触核改为工厂生成并自带候选压缩，过滤器在此退化为直接调用上游。
- 只在 Windows 源码构建下使用的 `run_source.py`、`source_runtime_guard.py`、`source_scene_runtime.py` 未迁入。
- `example_franka_cable_ik_pick_place`、`example_mujoco_franka_vbd_cable_admm_solver` 使用 Newton 实验性耦合求解器（MuJoCo + VBD/XPBD），
  与本求解器无关，未迁入。

## 验证

每个示例与 engineRTS 原版在各自环境中以 `--viewer null --test`（默认帧数）运行，要求结论一致（同为通过，或同为相同的失败）。
部分原版示例在默认帧数下本身不满足其 `test_final`（如 `basic_joints`、`basic_shapes`、`body_spring`），或未实现 `test_final`
（`body_point_line`、`joints`）；它们在 Newton 中的结论与原版相同。逐例结果记录在仓库根目录的 `COUPLED_HIGH_ACCURACY_PLAN.md`。
