# Kamino-only 预粘接压扁纸盒

这是一个没有机器人的 Newton/Kamino 原型：四块刚性纸板通过 4 个竖直转轴（压痕线）首尾粘接，形成闭环四杆机构；第 4 个转轴作为 loop-closure bilateral constraint 由 Kamino 求解。相对的两面各带一个顶封口页和一个底封口页，共 4 个刚性封口页。流程从地面上的平铺态开始，由动态基座控制器模拟夹具把纸盒抬起、转正、撑开后再放回地面；底部两页不设置驱动，只靠重力和地面接触自动闭合，顶部两页在本阶段保持打开。

本次结果在 Newton `1.5.0.dev0`（2026-08-04 main 快照）、Warp `1.16.0` 的 CPU 后端上实测生成；代码只调用 Newton/Warp 的公开 API。

> 当前版本包含四个侧壁和四个刚性顶/底封口页，并已开启 Kamino primitive 碰撞检测；纸板柔性、塑性压痕、机器人本体与真实接触撑开器还没有加入。搬运基座由外部 wrench 轨迹模拟夹具，底部封口页没有主动驱动。

## 为什么不是精确 0°

完全压平的四杆机构处于死点/奇异位形，没有方向信息，数值求解器无法凭空决定向哪一侧撑开。例子默认从 12° 开始，等价于吸盘或手指先制造了一点缝隙。可用 `--initial-angle-deg` 调整。

## 运行

在 Newton 源码仓库中安装 examples 依赖后运行：

```bash
uv run --extra examples python /path/to/kamino_carton_demo/carton_kamino.py
```

也可以只跑物理、不生成 GIF：

```bash
uv run --extra examples python /path/to/kamino_carton_demo/carton_kamino.py --no-gif
```

实时打开 Newton OpenGL 窗口：

```bash
uv run --extra examples python /path/to/kamino_carton_demo/carton_kamino.py --live
```

实时模式默认按 24 FPS 播放“平铺 → 抬起 → 转正 → 撑开 → 放回地面 → 底部封口页落地闭合”的慢速流程，`--duration` 秒后停在最终状态，关闭窗口即可退出。加上 `--loop` 可循环播放：

```bash
uv run --extra examples python /path/to/kamino_carton_demo/carton_kamino.py --live --loop
```

没有可用 GPU 时可显式使用 CPU：

```bash
uv run --extra examples python /path/to/kamino_carton_demo/carton_kamino.py --live --device cpu
```

可用 `--paused` 启动后暂停；窗口中的播放/单步控件可控制模拟。

输出位于 `output/`：

- `carton_kamino.gif`：效果动画
- `carton_final.png`：最终方形状态静帧
- `metrics.csv`：目标/实测开口角与闭环缝隙
- `summary.json`：模型与最终误差摘要

## 模型边界

- 求解器只有 `newton.solvers.SolverKamino`，没有 MuJoCo、XPBD、VBD 或 coupled solver。
- 侧壁和封口页当前都按刚体处理；“纸”的物性集中在转轴阻尼、驱动顺应性与极薄几何。
- 平铺态的纸板层之间过滤了重复自碰撞，纸板与地面仍由 Kamino primitive 碰撞检测处理；铰链父子形状也通过关节过滤，避免重复接触。
- 动态基座上的外部 wrench 只代表夹具搬运整箱；底部封口页的铰链是无驱动的，重力使其向下转动，地面接触/水平止挡防止它穿过地面。
- 侧壁撑开使用隐式 PD 目标代表撑开器；顶部封口页暂时保持打开，下一步可加入机器人、夹具和真实接触力传递。
- 达到 90° 后侧壁驱动仍保持目标，相当于撑开夹具或暂时锁止。仅靠四个粘接侧壁，真实纸盒在没有底部锁扣/夹具时不会稳定保持正方形。
