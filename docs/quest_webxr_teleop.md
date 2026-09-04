# Quest 3S WebXR 遥操

以下命令都在 Newton 仓库根目录执行：

```bash
cd /home/oem/code/repos/newton
```

页面右上角的“隐藏面板”可在进入沉浸式遥操前后使用。隐藏后，场景信息和操作说明不再遮挡实时画面，只保留右上角的小型“显示面板”按钮；再次点击即可恢复完整面板。该开关不暂停物理仿真、遥操输入或轨迹录制。

七个场景都支持机器人第一人称。进入 WebXR 后，点击“切换到机器人第一人称”或按左手柄 X，可把双眼放到 W1 的眼睛位置；之后头部左右转动驱动 `NECK1`，抬头和低头驱动 `NECK2`，视角随之变化。再次按 X 返回观察模式，观察模式下左摇杆可以转动视角。切换模式或按 B 时会重新建立头显与 Newton 相机的中立朝向。

## 插头/插座场景

先连接 Quest 3S，并在头显里允许 USB 调试。然后执行：

```bash
./scripts/start_quest_webxr_teleop.sh
```

脚本默认使用 `cuda:0` 做物理仿真，并在完成 CUDA 首帧预热后自动打开 Quest Browser。戴上头显，在页面中点击“进入沉浸式遥操”。

如果提示 Quest 未授权，戴上头显确认 USB 调试授权后，再执行一次启动命令。

退出 Quest 沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_teleop.sh
```

修改场景 Python 代码后，使用分阶段重载：

```bash
./scripts/reload_quest_webxr_plug_socket_teleop.sh
```

## 双手推椅场景

启动：

```bash
./scripts/start_quest_webxr_chair_teleop.sh
```

等待启动完成，然后在 Quest Browser 中点击“进入沉浸式遥操”。左右 Grip 分别移动 W1 左右手臂，左右 Trigger 分别控制对应手指。

退出 Quest 沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_chair_teleop.sh
```

修改代码后使用：

```bash
./scripts/reload_quest_webxr_chair_teleop.sh
```

## 充气塑料袋场景

启动：

```bash
./scripts/start_quest_webxr_bag_teleop.sh
```

右 Grip 控制 W1 右臂，右 Trigger 控制完整右手手指。Quest 会实时更新塑料袋的 216 个物理顶点，因此看到的是实际充气、接触和塑性变形，不是刚体替代物。

退出 Quest 沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_bag_teleop.sh
```

修改代码后使用：

```bash
./scripts/reload_quest_webxr_bag_teleop.sh
```

启用录制后，袋子轨迹文件会额外保存逐帧粒子位置、速度、体积和绝对压力，所以文件增长速度高于插头和椅子场景。

## 软方块、硬方块依次入袋场景

启动：

```bash
./scripts/start_quest_webxr_soft_rigid_bag_teleop.sh
```

右 Grip 控制 W1 右臂，右 Trigger 控制完整右手的十个关节，左摇杆水平/竖直方向转动观察视角。该场景进入 WebXR 时会沿 Newton 相机当前朝向拉近视点；按 B 会以当前头部位姿重新对齐相机，并清零摇杆产生的视角偏转。软方块和硬方块统一使用软方块场景的抓取手型；硬方块仍使用刚体接触，袋子和软方块仍使用实时软体物理。袋子按双面网格渲染，从袋内和背面观察也不会因背面剔除而缺失。

退出 Quest 沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_soft_rigid_bag_teleop.sh
```

修改代码后使用：

```bash
./scripts/reload_quest_webxr_soft_rigid_bag_teleop.sh
```

端口为 `8768`。启用录制后，JSONL 同时保存 W1 关节、袋子粒子、软方块粒子以及硬方块位姿和速度。

## 双手叠 T 恤场景

启动或从任意其他遥操场景安全切换：

```bash
./scripts/start_quest_webxr_tshirt_teleop.sh
```

左右 Grip 分别移动 W1 左右手臂，左右 Trigger 独立控制对应手指捏取 T 恤。手指不会随 Trigger 瞬间跳到目标角度：每个关节在未接触时最多移动 180°/s，检测到对应手与布料接触后降为 60°/s，张开方向也使用同一限速，兼顾响应速度并减少穿模。默认桌面观察相机向前拉近 1.8 m、抬高 0.35 m 并额外向下俯视 8°，左摇杆可以继续转动观察视角。Quest 显示完整 W1、物理桌面和 6436 个实时布料顶点；T 恤采用双面渲染，翻折后从正反两面都能完整看到。按 B 以当前头部位姿重新对齐相机，按右摇杆原地复位 W1 和完整 T 恤。

点击页面中的“切换到机器人第一人称”或按左手柄 X，可以把双眼切到 W1 的 `eyes` 位置。切换或按 B 时的当前头姿作为中立方向；之后真实头部左右转动驱动 `NECK1`，抬头和低头驱动 `NECK2`，双眼视角同时自然变化。头部平移保留为头显的小范围 6DoF 观察，滚转不映射到机器人，因为 W1 头部只有偏航和俯仰两个关节。再次按 X 或页面按钮返回桌面观察模式。

退出沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_tshirt_teleop.sh
```

开发过程中修改了 T 恤 WebXR Python 代码后，不要用 `stop` 再 `start`，也不要设置 `NEWTON_WEBXR_TERMINATE=1`。使用分阶段重载：

```bash
./scripts/reload_quest_webxr_tshirt_teleop.sh
```

该脚本先建立一个只提交极小心跳 kernel 的 CUDA guard，再依次确认手柄与录制停止、Quest 退出沉浸、旧场景停泊、协作式 shutdown、旧 PID 完全退出，最后等待冷却并启动新代码。整个过程不移除 ADB 映射、不使用 `SIGKILL`，任一步骤未确认都会中止。CUDA guard 会在本次开发会话中继续运行，避免场景切换时最后一个 CUDA 上下文退出；它会占用少量显存和电力，并在电脑重启时结束。

这条路径针对当前驱动的电源/连接状态切换问题降低风险，但无法绕过 NVIDIA 驱动自身的 Completion Timeout 缺陷，因此不能承诺绝对不会硬锁。若 guard 无法启动、驱动预检失败或协作式 shutdown 超时，脚本会保留现状并要求查看日志，此时正常重启仍是最保守的恢复方式。

端口为 `8769`。原场景的约 271 MiB 房屋 USD 只是渲染背景，不参与物理；为避免 Quest WebGL 资源耗尽，头显版本不传输整栋房屋，只保留任务所需的桌面、W1 和 T 恤。启用录制后，JSONL 保存双手目标与抓握、W1 关节以及逐帧 T 恤粒子位置和速度。

## 双手螺母螺栓场景

启动或从其他遥操场景安全切换：

```bash
./scripts/start_quest_webxr_nut_bolt_teleop.sh
```

场景保留 `example_mjvbd_v2_bimanual_nut_bolt.py` 的完整 W1、动态 M20 螺母/螺栓、预旋入状态和真实 SDF 螺纹接触。头显版本在机器人前方增加了真实刚体桌面，桌面顶面与螺母接触包络从第一帧接触，未抓稳时预旋入组件也不会直接掉到地面；详细螺纹 SDF 与桌面之间已过滤，避免和螺母凸包产生重复承托力。左右 Grip 分别移动对应手根；左右 Trigger 从张开手型缓慢闭合到原例子的任务手型。手指无接触时限制为 90°/s，发生刚体接触后降为 30°/s。原例子的接触设计也保持不变：左手负责夹持螺栓，右手用中指末端沿螺母侧面作切向扫动。

观察模式下左摇杆转动视角；点击“切换到机器人第一人称”或按左手柄 X 可切到 W1 眼睛位置，并用头部转动控制两个颈部关节。按右摇杆可原地复位 W1、螺母和螺栓。

退出沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_nut_bolt_teleop.sh
```

修改 Python 代码后需要加载新版本时使用分阶段重载：

```bash
./scripts/reload_quest_webxr_nut_bolt_teleop.sh
```

该场景使用端口 `8770`，并在通用 WebXR 缓存目录持久保存螺母和螺栓的 SDF。首次缺少外部 IsaacGymEnvs 网格或 SDF 缓存时启动会明显更久；后续启动会复用缓存。启用录制后，JSONL 保存双手目标、抓握、W1 关节、颈部目标以及螺母/螺栓的逐帧位姿和速度。

## 双手抓无纺布袋场景

启动或从其他遥操场景安全切换：

```bash
./scripts/start_quest_webxr_nonwoven_bag_teleop.sh
```

场景使用正常站立、双手张开的完整 W1，袋子位于机器人前方 0.85 m 高的有限桌面上。左右 Grip 分别移动对应手臂，左右 Trigger 独立闭合对应手指；手指无接触时限制为 90°/s，接触袋子后降为 30°/s，以减少穿模。首次按住任一 Grip 或 Trigger 后，会解除袋子用于保持初始直立造型的世界坐标恢复力，使袋子可以被抓起、转动和搬运；物理复位会恢复袋子初态与定型状态。

Quest 渲染完整 W1、桌面与桌腿，以及实时更新的双面无纺布袋。观察模式下左摇杆转动视角；点击“切换到机器人第一人称”或按左手柄 X 可切到 W1 眼睛位置。按右摇杆可原地复位 W1 和袋子。

退出沉浸模式并进入安全待机：

```bash
./scripts/stop_quest_webxr_nonwoven_bag_teleop.sh
```

修改 Python 代码后需要加载新版本时使用分阶段重载：

```bash
./scripts/reload_quest_webxr_nonwoven_bag_teleop.sh
```

该场景使用端口 `8771`。默认不录制；按右手柄 A 后，JSONL 保存双手目标、抓握、W1 关节、颈部目标以及袋子全部粒子的逐帧位置和速度。

## 轨迹录制

七个遥操场景默认都不自动录制轨迹。需要录制时，按右手柄 A 键开始；再次按 A 键可暂停或继续。也可以在启动命令末尾添加 `--record-on-connect`，显式恢复连接手柄后自动开始录制的行为。

未按 A 键且未传入 `--record-on-connect` 时，不会写入逐帧轨迹数据。此设置减少的是本机 JSONL 写盘；WebXR 实时画面、控制状态和袋子网格更新仍会正常传输。

## Quest Browser 标签页复用

七个启动脚本使用同一个浏览器应用标识。重复启动同一场景或切换场景时，Quest Browser 会在原有 Newton 标签页中导航，不再为每次启动创建新的标签页和渲染进程。

旧版脚本已经创建的标签页没有这个标识，无法被新版脚本自动接管。首次使用新版脚本前，在 Quest Browser 中手动关闭已有的 `127.0.0.1:8765`、`127.0.0.1:8766`、`127.0.0.1:8767`、`127.0.0.1:8768`、`127.0.0.1:8769`、`127.0.0.1:8770` 和 `127.0.0.1:8771` 标签页；如果此前已经提示 WebGL 不可用，可以关闭全部旧 Newton 标签页并重启一次 Quest Browser。此操作不需要停止或销毁 Newton CUDA 进程。

## 安全切换遥操场景

最简单的方式是直接执行目标场景的启动脚本。例如从软硬方块入袋切换到充气塑料袋，只需：

```bash
./scripts/start_quest_webxr_bag_teleop.sh
```

启动器会先禁用当前场景的手柄输入、暂停录制并退出其沉浸模式，但保持当前 Newton 进程持续提交物理帧。目标场景完成 CUDA 首帧预热后，启动器才自动停泊旧场景并让 Quest Browser 的同一个标签页进入新地址。切换期间不会销毁 CUDA 上下文，也不会移除任何 ADB 端口映射。

也可以先运行当前场景的关闭脚本，确认进入安全待机，再运行目标启动脚本；结果相同。如果新场景启动失败，旧场景会保持安全待机，不会被自动停泊或终止。修复启动错误后，重新运行目标启动脚本即可重试接棒。

已经加载过的场景进程会保留 CUDA 上下文和显存，再次切回时会直接恢复。当前驱动问题解决前，不要同时手工启动多个未由这些脚本管理的 CUDA 场景。

## 为什么关闭后进程仍然存在

当前机器的 NVIDIA 595.71.05 驱动曾在 CUDA 工作突然停止后的设备电源或连接状态切换阶段触发整机硬锁。默认关闭脚本因此采用安全待机：禁用手柄输入、暂停录制并退出 Quest WebXR，但继续提交稳定物理帧，同时保留 ADB 映射、CUDA 进程和上下文。再次执行同一场景的启动脚本会恢复原进程，不会重新初始化 CUDA。

因此，安全 `stop` 后再 `start` 不会加载期间修改的 Python 场景代码。启动器会比较运行进程和场景源码的时间戳；如果发现源码更新，会明确列出尚未加载的文件。七个场景都可使用各节列出的分阶段重载脚本，在 CUDA guard 保持设备活跃的情况下替换旧进程；正常重启仍是出现驱动异常时最保守的恢复方式。启动器每次都会用唯一查询参数刷新 Quest Browser 中复用的 Newton 标签页，因此新的 HTML/JavaScript 不需要手动清缓存。

安全待机会继续占用 GPU 和电力。只有在另一个场景已经稳定运行后，目标启动脚本才会将旧场景降为不提交物理帧的停泊状态，使 GPU 总体上不会在切换窗口突然空闲。

不要使用 `kill -9`。在 BIOS 和 NVIDIA 驱动更新完成前，也不要显式终止 CUDA 遥操进程。

驱动问题解决后，如确实需要销毁进程，可显式执行以下高风险命令：

```bash
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_chair_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_bag_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_soft_rigid_bag_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_tshirt_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_nut_bolt_teleop.sh
NEWTON_WEBXR_TERMINATE=1 ./scripts/stop_quest_webxr_nonwoven_bag_teleop.sh
```

不要绕过这些启动脚本手工恢复多个场景；启动脚本会自动串行完成安全接棒。

## 异常重启后清理

电脑异常重启后，也先执行一次退出命令来清除未完成运行标记，再重新启动：

```bash
./scripts/stop_quest_webxr_teleop.sh
./scripts/start_quest_webxr_teleop.sh
```

椅子场景对应为：

```bash
./scripts/stop_quest_webxr_chair_teleop.sh
./scripts/start_quest_webxr_chair_teleop.sh
```

充气塑料袋场景对应为：

```bash
./scripts/stop_quest_webxr_bag_teleop.sh
./scripts/start_quest_webxr_bag_teleop.sh
```

软方块、硬方块入袋场景对应为：

```bash
./scripts/stop_quest_webxr_soft_rigid_bag_teleop.sh
./scripts/start_quest_webxr_soft_rigid_bag_teleop.sh
```

双手叠 T 恤场景对应为：

```bash
./scripts/stop_quest_webxr_tshirt_teleop.sh
./scripts/start_quest_webxr_tshirt_teleop.sh
```

双手螺母螺栓场景对应为：

```bash
./scripts/stop_quest_webxr_nut_bolt_teleop.sh
./scripts/start_quest_webxr_nut_bolt_teleop.sh
```

双手抓无纺布袋场景对应为：

```bash
./scripts/stop_quest_webxr_nonwoven_bag_teleop.sh
./scripts/start_quest_webxr_nonwoven_bag_teleop.sh
```

## 检查运行状态和日志

```bash
systemctl --user status newton-quest-webxr.service --no-pager
curl --fail http://127.0.0.1:8765/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-teleop/latest.log"
```

椅子场景：

```bash
systemctl --user status newton-quest-webxr-chair.service --no-pager
curl --fail http://127.0.0.1:8766/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-chair-teleop/latest.log"
```

充气塑料袋场景：

```bash
systemctl --user status newton-quest-webxr-bag.service --no-pager
curl --fail http://127.0.0.1:8767/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-bag-teleop/latest.log"
```

软方块、硬方块入袋场景：

```bash
systemctl --user status newton-quest-webxr-soft-rigid-bag.service --no-pager
curl --fail http://127.0.0.1:8768/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-soft-rigid-bag-teleop/latest.log"
```

双手叠 T 恤场景：

```bash
systemctl --user status newton-quest-webxr-tshirt.service --no-pager
curl --fail http://127.0.0.1:8769/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-tshirt-teleop/latest.log"
```

双手螺母螺栓场景：

```bash
systemctl --user status newton-quest-webxr-nut-bolt.service --no-pager
curl --fail http://127.0.0.1:8770/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-nut-bolt-teleop/latest.log"
```

双手抓无纺布袋场景：

```bash
systemctl --user status newton-quest-webxr-nonwoven-bag.service --no-pager
curl --fail http://127.0.0.1:8771/healthz
tail -f "${XDG_STATE_HOME:-$HOME/.local/state}/newton-webxr-nonwoven-bag-teleop/latest.log"
```

安全待机后，`healthz` 中应显示 `teleoperationActive=false`、`simulationActive=true` 和 `operationMode=standby`。新场景接棒后，旧场景应显示 `simulationActive=false` 和 `operationMode=parked`；当前活动场景应显示两个 active 字段均为 `true`。

## 可选配置

默认使用不创建本机 OpenGL 窗口的 `ViewerNull`，画面直接在 Quest 中双眼渲染。仅在明确需要桌面调试窗口时使用：

```bash
NEWTON_WEBXR_VIEWER=gl ./scripts/start_quest_webxr_teleop.sh
```

当前机器曾发生过显示/计算负载下的整机卡死，因此日常遥操不要启用这个选项。
