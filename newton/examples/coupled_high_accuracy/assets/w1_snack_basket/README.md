# W1 购物篮装薯片场景资产

由 `demos/demo/demo_w1_snack_basket.py` 和 `demos/w1_snack_basket/` 使用，来源为用户提供的 `~/下载/assets`。

- `robot/`：W1 V030 simready 版（DexForce），原样拷贝自 `~/下载/assets/w1-030`。
  与 `assets/w1/robot/w1-030` 运动学相同，惯量已修正、视觉网格重新着色（`visual_recoloured/`）。
  `robot.urdf.before-simready-20261019` 是修正前的版本，仅作对照保留。
- `snack/`：零食 snack004（Extwin，CC BY-NC 4.0），原样拷贝；来源与校验见其中的 `README.md`、`provenance.json`。
- `basket/`：购物篮 shopping basket001（Extwin，CC BY-NC 4.0）；来源见其中的 `README.md`、`provenance.json`。

## 购物篮的格式转换

原始的三个文本 USD 层共 201 MB，其中 `link_default_0.usda` 为 120 MB，超过 GitHub 单文件 100 MB 上限。
三个层已用 `pxr.Sdf.Layer.Export` 逐层无损转为二进制 `.usdc`（共 59 MB），入口层的两个子层路径和 `link_default_0` 对材质层的子层路径相应改为 `.usdc`，
层结构、prim、属性、关系和材质均不变。转换后用 `pxr.Usd.Stage` 打开两份组合舞台，逐 prim、逐属性比较取值，
33 个 prim 无任何差异。`basket/README.md` 和 `provenance.json` 中的原始文件校验对应转换前的 `.usda`：

| 原始文件 | SHA-256 |
| --- | --- |
| `model_basket_7.usda` | `2bb241c6f9394864fd6208cd5a784e10cb8570f8743122ab353cd73011f0873c` |
| `resource/material.usda` | `891fe8784a1ed4881d03358f0069e3c7585d1844f20706c8534a48a29bc737b9` |
| `resource/v_0/link_default_0.usda` | `c9732bfcc6473b4451a79a702cac030b1a57d49609ade35394d11950cbe4bd59` |

| 转换后文件 | SHA-256 |
| --- | --- |
| `model_basket_7.usdc` | `c14467bd7afaeecc615b8764f6b3ba7deb4ed310bdb203b38a5a6bf56decbee3` |
| `resource/material.usdc` | `f44e9ee144026ffaaf7abd0df51f7853792034899ca341fab405a973e06be4be` |
| `resource/v_0/link_default_0.usdc` | `c7f2290578daa95fdaa39ecd83f230e5d200414d4b1d34da9027f86534060c3f` |

许可：两件 Extwin 模型均为 CC BY-NC 4.0（署名、非商业），使用时须保留署名和许可文件 `LICENSE`。
