# 三工作区 Director 设计方案

本文总结 `example_cloth_dexforce_three_workcell_director.py` 当前采用的设计。

目标场景是一个房间级机器人仿真：W1 在房间内移动，到达某个工作区后执行该工作区的局部任务，任务结束后离开并前往下一个工作区。视觉上可以是一镜到底的大场景，但昂贵的物理仿真只在当前激活的工作区运行。

## 当前范围

当前 demo 包含三个工作区：

- `example_cloth_dexforce_bimanual_fold_tshirt_v2.py`
- `example_cloth_dexforce_bimanual_grasp_cloth_v3.py`
- `example_cloth_dexforce_bimanual_grasp_cube.py`

房间级 director 负责显示：

- 大地面。
- 三张放在不同房间坐标的桌子。
- 每个工作区桌面上的静态预览物体。
- 用于房间导航展示的 W1 机器人。

当 W1 到达某个工作区后，director 会加载并运行该工作区原始 demo。任务时间结束后，该局部任务会被卸载或缓存，然后 director 切回轻量房间场景。

## 两层仿真模型

当前设计分成两层：

- 房间层。
- 工作区层。

## 房间层

房间层是轻量层，包含地面、桌子、静态预览物体和一个导航用 W1。

导航阶段的 W1 以 floating-base URDF 加载。房间导航时不运行机器人动力学求解器，而是直接设置 W1 根 free joint 的位姿，然后调用 FK：

```python
self.room_model.joint_q.assign(root_q)
newton.eval_fk(self.room_model, self.room_model.joint_q, self.room_model.joint_qd, self.room_state)
```

这样可以保证 W1 的底座和全身一起移动，同时避免导航阶段产生求解器开销。

房间层主要用于连续相机、空间关系和工作区预览。它不是一个包含所有工作区完整物理对象的大型全局物理场景。

## 工作区层

每个工作区仍然是一个独立 Newton example。director 只有在 W1 到达对应工作区时，才 import 并实例化该工作区模块：

```python
mod = importlib.import_module(workcell.module_path)
self.active_task = mod.Example(self.viewer, task_args)
```

工作区 example 自己拥有完整的局部内容：

- model
- state
- contacts
- solver
- robot setup
- cloth / rigid object
- task logic

工作区激活时，director 只 step 当前这个任务：

```python
if self.active_task is not None:
    self.active_task.step()
```

这样可以保留原始 demo 的行为，同时避免把所有任务逻辑硬塞进一个巨大的 model。

## 加载策略

启动时只创建房间预览场景。

工作区任务按需加载，流程如下：

1. W1 在房间中通过直接设置 root pose 移动。
2. W1 进入某个工作区的触发半径。
3. director 从 viewer 中清掉房间 model。
4. 实例化目标工作区 example。
5. 该工作区的求解器运行一段任务时间。
6. director 停止并退出该工作区。
7. 恢复房间 model。
8. W1 前往下一个工作区。

默认情况下，离开工作区后 inactive task 会被释放，并调用 `gc.collect()`。这样可以控制内存和显存占用。

也可以使用 `--cache-tasks`。开启后，已经创建过的工作区 example 会保留下来，下一次进入时复用。这样调试更快，但会占用更多内存。

## 数据管理

大场景最容易混乱的不是怎么摆物体，而是不同层级的数据生命周期不清楚。当前方案需要把数据分成几类管理，避免所有数据都堆在一个全局对象里。

## 1. 房间配置数据

房间配置数据描述“有哪些工作区、工作区在哪里、怎么切换”，属于 director 层。

典型数据包括：

- `WorkcellSpec.name`
- `WorkcellSpec.module_path`
- `WorkcellSpec.room_pos`
- `travel_time`
- `task_time`
- `trigger_radius`
- 相机参数
- 工作区静态预览几何

这些数据应该常驻内存，因为它们很小，而且用于调度整个 timeline。

房间配置数据不应该包含某个工作区的完整 solver state、cloth particles、MPM particles、contacts 或机器人控制轨迹。那些是局部任务数据。

## 2. 房间运行态数据

房间运行态数据描述“当前大房间跑到哪里了”。

典型数据包括：

- `sim_time`
- 当前 timeline segment
- W1 当前导航位置 `w1_pos`
- active workcell index
- 当前 room model / room state
- task start / task end events

这部分数据由 director 持有。它只负责全局调度和视觉连续性，不负责保存每个工作区的内部物理细节。

导航 W1 的状态也属于房间运行态。它通过 root free joint pose + FK 更新，不是工作区机器人动力学状态。进入某个工作区后，active task 会创建自己的 W1 robot model/state，两者是不同层的数据。

## 3. 工作区配置数据

工作区配置数据保存在各自的 example 模块里。

例如叠衣服工作区的数据包括：

- 衣服 asset、scale、初始 pose
- 桌子尺寸和位置
- W1 URDF 加载参数
- 手指姿态参数
- 抓取/折叠轨迹
- contact material 参数
- solver 参数

这些数据不应该上移到 director。director 只需要知道该工作区的 module path 和 room position。这样每个工作区可以独立调试，也可以使用不同 solver。

## 4. 工作区运行态数据

工作区运行态数据是最重的数据，生命周期应该最短。

典型数据包括：

- local `model`
- local `state_0` / `state_1`
- local `control`
- solver object
- contacts buffer
- cloth particle state
- rigid body state
- MPM particle state
- cable state
- soft body state
- IK solver state
- task 内部计时器和阶段状态

这些数据只在该工作区 active 时存在。离开工作区后有两种策略：

- 默认策略：释放 active task，并调用 `gc.collect()`，降低内存和显存占用。
- 调试策略：使用 `--cache-tasks` 保留 task object，下次进入时复用，减少加载时间。

对于大场景，默认应该优先使用释放策略。只有在某个工作区需要频繁反复调试时，才启用缓存。

## 5. 静态资产数据

静态资产包括 URDF、USD、mesh、纹理等。这些数据通常来自磁盘：

- W1 URDF
- shirt USD
- cube mesh
- table geometry
- cable / softbody / MPM 相关 asset

资产数据可以被多个工作区引用，但不等于多个工作区的运行态共享。推荐做法是：

- director 只加载轻量预览需要的资产。
- active workcell 自己加载本任务需要的完整资产。
- 如果某些资产加载很慢，可以加一个只读 asset cache。
- asset cache 只缓存 mesh/路径/解析结果，不缓存 solver state。

这样可以避免“为了复用资产，把整个工作区 model 都常驻”的问题。

## 6. Snapshot / Restore 数据

如果希望离开工作区后下次回来还能接着上次状态继续，就不能只靠重新实例化 example，需要引入 snapshot/restore。

snapshot 应该保存最小必要状态，而不是把整个 Python object 直接序列化。

一个工作区 snapshot 可以包含：

- 当前 task time / phase
- robot joint q / qd / target
- rigid body q / qd
- cloth particle q / qd
- MPM particle q / v / material state
- cable control points / velocities
- softbody particle state
- 需要恢复的随机种子
- 必要的 task-specific 状态

restore 时先重新创建该工作区 example，然后把 snapshot 写回 local state。

推荐接口：

```python
class WorkcellRuntime:
    def snapshot(self) -> dict: ...
    def restore(self, snapshot: dict): ...
```

这样 director 可以只管理 snapshot 字典，而不需要理解每个 solver 的内部数据结构。

## 7. 跨工作区物体交接数据

如果未来需要“机器人从一个工作区拿着物体走到另一个工作区”，需要额外的跨工作区状态交接层。

当前 demo 没有做真实跨工作区物理连续交接。当前切换是：

- 房间层负责 W1 导航和视觉连续。
- active workcell 负责局部物理。
- 离开 workcell 后局部物理状态释放或缓存。

如果要做物体交接，需要定义一个 portable object state，例如：

```python
{
    "object_id": "plate_001",
    "type": "rigid",
    "pose": ...,
    "velocity": ...,
    "attached_to": "w1_right_hand",
}
```

进入下一个工作区时，该工作区根据 portable state 创建对应局部对象，或者把对象 attach 到机器人手上。

对于 cloth、MPM、softbody 这种高维状态，跨工作区交接成本更高，通常需要：

- 降维成一个代表性 pose / bounding state；
- 或者完整保存 particle state；
- 或者在下一个工作区重新生成一个近似状态。

这部分不应该隐式发生，需要显式的数据协议。

## 8. 日志、轨迹和传感器数据

日志和训练数据不要直接塞进 model/state。建议单独建立 episode 级数据结构。

推荐层级：

```text
episode_id/
  room_timeline.json
  workcells/
    fold_tshirt_v2/
      config.json
      events.jsonl
      robot_trajectory.npz
      cloth_state_samples.npz
      camera/
    grasp_cloth_v3/
      config.json
      events.jsonl
      robot_trajectory.npz
    grasp_cube/
      config.json
      events.jsonl
      rigid_state_samples.npz
```

房间层日志记录：

- W1 在房间中的导航轨迹。
- 工作区进入/退出事件。
- 相机轨迹。
- 每个工作区的开始时间和结束时间。

工作区层日志记录：

- 局部 robot joint / TCP 轨迹。
- 局部物体状态。
- contact / force 信息。
- solver 统计。
- 传感器图像或点云。

这样后处理时可以按 episode 聚合，也可以只分析某个局部任务。

## 9. 推荐的数据所有权原则

建议采用以下原则：

- director 拥有 timeline、房间坐标、active workcell、全局事件。
- room model 拥有轻量静态预览和导航 W1。
- workcell example 拥有局部 model、solver、state 和任务逻辑。
- asset cache 只拥有只读资产，不拥有物理运行态。
- snapshot store 只拥有可恢复状态，不拥有 solver object。
- logger 拥有 episode 数据，不参与求解。

这样不同数据不会互相污染，场景变大后也能保持可控。

## 大场景为什么不容易崩

核心原则是：

同一时间只激活一个重型物理任务。

房间 model 很轻，因为它主要包含静态几何和一个 kinematic 方式移动的 W1。它不会同时包含所有 cloth、soft body、MPM、cable、液体、机器人任务等完整仿真对象。

稳定性来自这些边界：

- 未激活工作区只是静态视觉预览。
- 未激活工作区的 solver 不 step。
- 未激活工作区的 CPU/GPU 状态可以释放。
- 当前 active solver 只看到它自己的局部工作区 model。
- director 不把所有任务合并成一个单体大 model。

这样可以避免常见的大场景问题：一个全局场景同时分配并求解所有子系统，导致显存爆炸、接触数量过大、求解器耦合复杂、调试困难。

## 多求解器共存

推荐的多求解器设计不是把所有 solver 同时塞进一个 model，而是让每个工作区保持自包含。

例如：

- 叠衣服工作区可以使用 MJVBD 或 MuJoCo 相关 cloth/robot setup。
- 抓方块工作区可以使用刚体求解器。
- cable 工作区可以使用 cable 相关模型和求解器设置。
- softbody 工作区可以使用 soft body solver。
- MPM 工作区可以使用 MPM solver。

director 不关心具体工作区内部使用哪种 solver。它只把每个任务当成一个具有 `step()` 和 `render()` 行为的 black-box example。

运行时，只有当前工作区的 solver 存在并被 step。不同工作区可以使用不同 solver，因为它们不会被强行放进同一个 Newton model 里同时求解。

这是房间级多求解器仿真的实用方案：

- 房间层使用同一条 timeline。
- W1 导航抽象一致。
- 每个工作区有独立局部 model。
- 同一时间只运行一个物理 backend。

如果未来某个单独工作区内部确实需要两个 solver 做耦合，这个耦合应该在该工作区 example 内部实现，而不是放到房间 director 层。

## 相机策略

导航阶段，相机跟随房间中的 W1。

当 W1 到达工作区后，相机切换到工作区视角。当前三个 cloth 工作区使用统一的本地相机参数：

```python
DEMO_CAMERA_POS = wp.vec3(1.15, -2.10, 1.65)
DEMO_CAMERA_PITCH = -16.0
DEMO_CAMERA_YAW = 112.0
```

在 `--dry-run-tasks` 模式下，这个相机位置会加上工作区的 room position，用来观察房间预览中的对应桌子。真正进入任务仿真后，task example 使用自己的局部坐标系。

## 时间线

当前时间线如下：

```text
home -> 工作区 1 -> 任务 1
工作区 1 -> 工作区 2 -> 任务 2
工作区 2 -> 工作区 3 -> 任务 3
工作区 3 -> 工作区 1
```

demo 默认自动循环。使用 `--no-loop` 可以只跑一轮，回到第一个工作区后停止。

## 当前命令

运行完整 director：

```bash
python -m newton.examples cloth_dexforce_three_workcell_director
```

只看房间导航和工作区切换，不加载重型任务：

```bash
python -m newton.examples cloth_dexforce_three_workcell_director --dry-run-tasks
```

只跑一轮：

```bash
python -m newton.examples cloth_dexforce_three_workcell_director --no-loop
```

离开工作区后缓存 task example：

```bash
python -m newton.examples cloth_dexforce_three_workcell_director --cache-tasks
```

## 扩展方式

新增一个工作区时，推荐流程：

1. 先把局部任务写成一个普通 Newton `Example`。
2. 保持任务自包含：model、solver、state、contacts、robot setup 和任务逻辑都留在该模块内。
3. 在 director 中新增一个 `WorkcellSpec`。
4. 在房间场景里新增轻量静态预览。
5. 调整 travel time、task time、trigger radius 和 camera。

不要把所有局部任务对象和求解器直接加到房间 model 中。房间 model 是调度层和可视化层，不是所有工作区的全局物理 model。

## 设计取舍

当前方案优先保证鲁棒性、可调试性和内存可控，而不是追求所有工作区物理状态完全连续。

优点：

- 大房间保持轻量。
- 同一时间只运行一个昂贵 solver。
- 不同工作区可以使用不同 solver。
- 可以复用现有 example，改动少。
- inactive task 可以卸载，内存和显存可控。

限制：

- 物理仿真只在当前 active workcell 内局部连续。
- 工作区状态默认不会自动保存，除非使用 `--cache-tasks` 或实现显式 snapshot/restore。
- 房间预览和 active workcell 之间是场景切换，不是一个单体物理世界。
- 如果未来需要把物体从一个工作区搬到另一个工作区，需要额外实现状态交接层。

## 后续建议

如果要做生产级房间大场景，建议抽象一个 workcell runtime 接口：

```python
class WorkcellRuntime:
    def load(self): ...
    def unload(self): ...
    def step(self): ...
    def render(self): ...
    def snapshot(self): ...
    def restore(self, snapshot): ...
```

这样可以把任务加载、卸载、缓存、状态保存和状态恢复显式化，同时保留当前“局部求解器 + director 调度”的设计。
