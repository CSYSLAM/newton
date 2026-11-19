# Demo 目录

主入口统一放在 [`demo/`](demo/)；运动、几何、配置、导出、渲染和验证文件按场景放在各自目录。
通用工具集中在 [`utils/`](utils/README.md)，模型资源按场景归档在仓库的 `assets/` 下。

```text
demos/
├── demo/                       # 所有 demo 主入口
├── g1_jelly_drop/               # G1、果冻几何、导出与渲染
├── piper_bag_transport/         # Piper 运动、袋子、遥测与渲染
├── w1_courier_transport/        # W1 快递袋的运动、导出与验证
├── w1_pharmacy_carton/          # 药盒材料、运动与参考渲染
├── w1_pharmacy_variants/        # 四种药盒的几何、对比与验证
├── piper_package_transport/     # 多个包装场景共用的几何和材质依赖
├── reference_package_transport/ # 包装场景共用的参考渲染
├── w1_shopping_bag/             # 药盒场景复用的夹爪与运动依赖
├── w1_snack_basket/             # W1 左手提购物篮、右手放薯片袋
├── utils/                      # 通用渲染、编码、接触和运行工具
```

从仓库根目录直接运行任一主入口，默认开窗并开始仿真：

| 主入口（位于 `demos/demo/`） | 场景依赖 / 资源 | 用途 |
| --- | --- | --- |
| `demo_spin.py` | `assets/spin/` | 陀螺旋转 |
| `demo_dzhanibekov.py` | `assets/dzhanibekov/` | 中间轴翻转 |
| `demo_g1_jelly_drop.py` | `demos/g1_jelly_drop/` | G1 落到果冻堆 |
| `demo_piper_bag_transport.py` | `demos/piper_bag_transport/` | Piper 装袋和搬运 |
| `demo_w1_courier_transport.py` | `demos/w1_courier_transport/` | W1 双臂夹快递袋 |
| `demo_w1_pharmacy_carton.py` | `demos/w1_pharmacy_carton/` | W1 药盒参考场景 |
| `demo_w1_pharmacy_variants.py` | `demos/w1_pharmacy_variants/` | 四种药盒、四级变形对比 |
| `demo_w1_snack_basket.py` | `demos/w1_snack_basket/` | W1 提购物篮、放入可变形薯片袋 |

```bash
python newton/examples/coupled_high_accuracy/demos/demo/demo_spin.py
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_courier_transport.py
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_pharmacy_variants.py
```

需要无窗口运行时显式指定参数；药盒系列的 `--render` 也会自动使用无窗口录制：

```bash
python newton/examples/coupled_high_accuracy/demos/demo/demo_spin.py --viewer null --num-frames 100
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_courier_transport.py --viewer=false --output build/my-courier
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_pharmacy_carton.py --no-viewer --output build/my-cartons
python newton/examples/coupled_high_accuracy/demos/demo/demo_w1_pharmacy_variants.py --no-viewer --output build/my-pharmacy
```

辅助命令在对应场景目录下运行，例如：

```bash
python newton/examples/coupled_high_accuracy/demos/w1_courier_transport/validate_w1.py build/my-courier
python newton/examples/coupled_high_accuracy/demos/w1_courier_transport/make_video.py --output build/my-courier
```

Piper 和 G1 的导出、遥测、渲染等输出仍归属于各自场景目录的 `outputs/`；W1 的输出由 `--output` 指定。
以上命令在 Newton 仓库根目录运行。各场景的详细命令见其 README。资源通过上级目录的 `assets` 链接读取 engineRTS 的
`assets/`（见 `../README.md`）。
历史验收记录保留生成时的路径和哈希，新导出会记录新的入口路径。
