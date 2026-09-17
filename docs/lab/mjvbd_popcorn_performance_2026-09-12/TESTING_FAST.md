# Popcorn CUDA 优化：已接入当前求解器

2026-09-13：已将验收组合迁入 MJVBD V2 和 IK 实现，移除实验入口及全局方法替换。

## 运行

直接运行原来的例子即可，默认已启用：

```powershell
uv run --no-sync python newton/examples/mjvbdv2/example_mjvbd_v2_popcorn.py
```

完整 1080p 端到端验收（30 帧预热＋2850 帧计时）：

```powershell
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/validate_default.py
```

旧的 `run_popcorn_fast.py` 和 probe 安装器已移出工作区，不再使用。

## 实现边界

- 接触、DAT、表面融合、刚体 worker 调度和 Ritz 预条件器位于 MJVBD V2 内部。
- IK 的解析行融合和精确固定点跳过位于 IK 模块。
- 不替换 `wp.launch`、数组方法或求解器类方法；缓存/图分支状态由实例持有。
- IK 仍使用普通 Warp kernel factory，为不同目标列表生成类型化参数包装；
  数学函数为静态代码，不再在运行时分析、改写原方法的 AST。
- `vbd_options["enable_cuda_fast_path"]` 和同名 collision option 控制求解/接触优化；
  `IKSolver(enable_cuda_fast_path=True)` 控制 IK 优化。默认均为关闭，只有本例显式启用。
- 这是运行时执行策略，不是物理资产属性，因此不新增 USD schema。
- CPU、可微、确定性以及不满足具体优化条件的路径保留原实现。
  接触非空或邻接容量溢出时，表面融合退回普通 sweep。
- 改动模型属性后须通知求解器并重新捕获 CUDA Graph；更新 rest/filter 数据也须通知，
  不能在复用旧图时静默修改几何过滤规则。

物体仍为 160，8 substeps、8 local sweeps、8 PCG iterations、每 substep 两次全局修正。
没有改杯子/铲子几何、抓取控制、材料摩擦、轨迹、物理断言或渲染分辨率。

## 验证与限制

- 实验组合整理前：27.47 FPS；隔离入口复测 27.80 FPS。
- 集成后第二次完整运行：36.793996 ms/frame，27.178347 FPS；
  22/29 颗保留（75.86%），原物理验收通过，同状态渲染像素差为零。
- 第一次完整运行在 22.683 秒触发纸杯滑落断言；没有放宽断言或修改抓取参数。
  因此不宣称每次都稳定成功，也不将一次通过视为消除了接触轨迹分叉。
- 清理后的第三次运行在 31.833 秒再次触发滑落（30.2 mm）。三次完整流程尝试
  是一次通过、两次失败；当前只能称核心迁移完成，不能称场景稳定性验收通过。
- 并行归约及不同编译单元可能产生浮点差异；不能承诺长时间轨迹逐位一致。
- 30 项针对性回归通过，涵盖新路径、原 DAT、耦合平移、接触优化和 popcorn 诊断；
  IK 又补测了位置＋旋转＋关节限位组合，逐位状态对比及 CPU/可微回退通过。
  这不是所有 demo 全部回归通过的声明。

新增长期保留测试：

```powershell
uv run --no-sync python -m unittest newton.tests.test_mjvbd_v2_cuda_fast newton.tests.test_mjvbd_v2_cuda_search newton.tests.test_ik_cuda_fast
```

32 个已迁移的实验脚本备份在：
`E:\csy_work\CG\Engine\newton_cleanup_archive\popcorn_integrated_probes_2026-09-13`。
此前未采用的 69 个试验仍在 `popcorn_rejected_2026-09-13`，均可恢复。
未提交、未推送。
