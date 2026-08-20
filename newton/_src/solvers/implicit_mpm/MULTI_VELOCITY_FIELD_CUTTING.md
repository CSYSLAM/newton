# MPM 多速度场切割方案

## 文档状态

- 记录日期：2026-08-18
- 状态：实验性实现，尚未提交
- 对应示例：`newton/examples/mpm/example_mpm_suspended_sheet_scissors.py`
- 主要实现目录：`newton/_src/solvers/implicit_mpm/`

本文记录当前工作区中 MPM 布料切割方案的实际实现状态。它不是通用拓扑切割系统，而是通过预设材料分区、局部多速度场和碰撞接触激活，实现粒子不消失的渐进式切割效果。

## 目标

当前方案需要满足以下目标：

1. 剪刀必须作为真实的 MPM 粒子碰撞体参与切割触发。
2. 切口处不删除粒子，不修改粒子的 `ACTIVE` 标记，也不把渲染半径设为零。
3. 只有两片刀刃同时接触并相向闭合时才触发切割。
4. 切开后两侧粒子即使落在同一空间网格中，也不再交换速度。
5. 未切区域仍保持连续，不应因为预先划分了两个速度场而自动裂开。
6. 默认 MPM 配置不启用该功能，不影响现有示例的计算路径和结果。

## 核心思路

布料粒子在初始化时按照预设切割曲线划分到两个速度场：

```text
预设切割曲线上侧粒子 -> velocity field 1
预设切割曲线下侧粒子 -> velocity field 0
```

两个速度场使用空间重合但相互独立的 FEM 环境和网格自由度。初始状态下，这些重合节点会执行质量加权的动量耦合，因此整块布料仍表现为连续材料。

```text
粒子预分区
    ↓
分别 P2G 到 field 0 / field 1
    ↓
查找两套场中空间重合的网格节点
    ↓
未激活裂纹：合并节点速度，保持材料连续
    ↓
剪刀双刃接触并相向闭合
    ↓
将接触粒子的 separation 永久置为正值
    ↓
裂纹影响节点跳过速度合并
    ↓
切口两侧独立运动并在重力下分开
```

因此，切割不是通过移除材料实现的，而是通过局部取消两套速度场之间的运动学约束实现的。

## 求解器接口和数据

### `Config.velocity_field_count`

`SolverImplicitMPM.Config` 新增 `velocity_field_count`：

- 默认值为 `1`，保持原有单速度场 MPM 路径。
- 大于 `1` 时启用多速度场路径。
- 当前剪刀示例使用 `2`。

当值为 `1` 时，不分配多速度场粒子数组、节点映射和 HashGrid，也不启动分离激活及速度场耦合内核。

### `particle_velocity_field`

每个粒子的速度场编号，类型为 `wp.array[wp.int32]`。编号必须位于：

```text
[0, velocity_field_count)
```

它是材料分区标签，不会随着粒子运动自动变化。剪刀示例根据粒子的静止位置和整条预设曲线一次性完成赋值。

### `particle_velocity_field_separation`

每个粒子的局部分离激活值，类型为 `wp.array[float]`：

- `0`：该粒子影响到的重合网格节点继续耦合。
- 大于 `0`：该粒子影响到的节点允许不同速度场独立运动。

碰撞触发后当前实现会将值设为 `1`。该值具有持久性，不会因剪刀离开而自动恢复；只有显式覆写数组或重新创建求解器才会重新粘合。

### `set_velocity_field_separation_colliders()`

该方法选择可以触发局部分离的 MPM collider，并配置：

- `contact_margin`：粒子中心到碰撞表面的附加接触距离。
- `minimum_contact_count`：同一粒子需要同时接触的选中 collider 数量。
- `minimum_closing_speed`：选中表面沿相对法线方向的最小闭合速度。

传入 `None` 或空序列可关闭碰撞触发。重新调用 `setup_collider()` 会清空选择，因为 collider 索引可能发生变化。

## 多速度场网格实现

### 独立 FEM 环境

每个速度场对应一个 FEM environment。粒子只向自己所属的 environment 传递质量、动量、应变和碰撞信息。

为了让各速度场拥有可对应的空间节点，多速度场模式会：

1. 在所有 environment 中激活匹配的网格单元。
2. 为每套 environment 构建空间重合的速度节点。
3. 使用 Warp `HashGrid` 将其他 field 的节点映射到 field 0 的同位置节点槽位。

若某个 field 缺失对应节点，当前节点组会被当作已分离，避免错误地把不完整节点组合并。

### 节点耦合

未分离节点使用质量加权速度：

```text
merged_velocity = sum(field_mass * field_velocity) / sum(field_mass)
```

然后把结果写回所有重合 field。只要节点组中的任一节点受到正的 separation 影响，该组便跳过合并。

耦合目前在以下位置执行：

1. P2G 和自由速度计算完成后。
2. 隐式 rheology/contact 求解开始前。
3. Gauss-Seidel 非线性迭代的每个内部求解步之后。
4. 整个隐式求解完成后、G2P 之前。

这几次投影用于避免未切区域在隐式求解过程中产生相对速度。

## 剪刀接触激活

每个时间步在 collider 栅格化之后、P2G 自由速度计算之前执行接触激活。

对每个尚未分离的动态活动粒子：

1. 查询所有被选为 cutter 的 collider SDF。
2. 判断 `sdf <= contact_margin`。
3. 将 collider 局部法线和速度转换到世界坐标。
4. 对刚体 collider 加上质心线速度和接触点角速度 `omega × r`。
5. 统计同时接触的 cutter collider 数量。
6. 若启用闭合速度条件，检查至少一对接触面的相对法线速度是否达到阈值。
7. 条件满足后将该粒子的 separation 置为 `1`。

以下粒子不会触发切割：

- 已经分离的粒子。
- `ACTIVE` 标记关闭的粒子。
- 零质量固定粒子。

当前 `minimum_contact_count` 统计的是 collider 数量，并未明确要求 collider 来自不同刚体。剪刀示例主要依靠相反法线和闭合速度条件排除同一刀片相邻分段的误触发。

## 剪刀示例配置

### 布料分区

示例通过曲线：

```text
envelope(p) = 16 * p^2 * (1 - p)^2
```

生成切割中心线。静止位置位于曲线上侧的粒子分配到 field 1，下侧粒子分配到 field 0。

这意味着当前切口路径是预先指定的。剪刀负责沿该路径逐步激活裂纹，但不能临时改变方向后在任意位置生成新的拓扑分支。

### 刀刃几何

- 完整弯曲刀片 mesh 仅用于渲染。
- 每片刀刃沿弯曲边缘使用 8 个窄 box 近似真实碰撞边。
- 窄 box 启用粒子碰撞，但关闭 shape-to-shape collision。
- 手柄、圆环和铰链仅用于渲染。

当前剪刀采用缩小后的几何尺寸：

```text
刀片长度 = 135 mm
刀片根部高度 = 14 mm
刀片尖端高度 = 2 mm
刀刃最大曲率偏移 = 4.5 mm
刀片半厚度 = 1.25 mm
手柄直杆长度 = 95 mm
手柄圆环主半径 = 21 mm
手柄圆环管半径 = 4 mm
铰链半径 = 9 mm
```

相较原始几何，刀片长度缩短 25%，手柄部分缩小约 10%。刀刃曲率与长度按相同比例缩放，所以约 5.98 度的啮合半角保持不变。

没有直接使用完整刀片 mesh 作为粒子 collider，是因为粒子容易被夹在两个快速闭合的薄 mesh 之间，之前会导致严重穿透和数值爆炸。当前窄边 collider 是稳定性和几何真实性之间的折中。

### 切割触发参数

当前示例参数为：

```text
velocity_field_count = 2
contact_margin = 0.55 * min(sheet_spacing)
minimum_contact_count = 2
minimum_closing_speed = 1.0e-3 m/s
```

这使整体平移、单片刀刃擦过、剪刀张开和刀片远离粒子时不会主动激活裂纹。

### 剪刀轨迹

轨迹不再简单按照刀尖做直线插值，而是计算两条弯曲刀刃的实际交点：

1. 根据刀片曲线求当前张角对应的交点。
2. 张角从打开状态先运动到刀刃刚好啮合的位置。
3. 继续闭合时才让实际切割交点沿曲线路径推进。
4. 剪刀 yaw 使用曲线切线方向。
5. 每次咬合后重新张开并进入下一次咬合。

默认使用 8 次咬合、18 度打开半角和 4 度闭合半角。弯曲刀刃的啮合半角约为 5.98 度。

## 不再使用的旧方案

旧示例包含 `_advance_cut()`：根据预设的 `cut_front` 逐行清除粒子的 `ACTIVE` 标记，并把渲染半径设为零。这会表现为一排或两排粒子消失，而不是材料从粒子之间断开。

当前实现已经移除：

- `_advance_cut()`。
- `particle_flags` 的切割修改。
- 切口粒子渲染半径归零。
- 由时间轨迹直接推进整条删除带的逻辑。

测试会验证切割前后 `particle_flags` 完全一致。

## 默认路径兼容性

现有 MPM 示例默认使用：

```text
velocity_field_count = 1
```

此时：

- 不创建 `particle_velocity_field`。
- 不创建 `particle_velocity_field_separation`。
- 不复制 FEM environment。
- 不构建重合节点 HashGrid。
- 不执行接触分离 kernel。
- 不执行额外的多场速度耦合投影。

因此普通 MPM demo 仍然使用原有求解路径。除了少量 Python 条件判断，默认模式没有额外 GPU 工作。

## 当前约束

多速度场模式当前存在以下硬限制：

1. 只支持单 world 模型。
2. `grid_padding` 必须为 `0`。
3. 只支持 Gauss-Seidel rheology solver。
4. CPU 只能使用 dense 或 fixed grid。
5. sparse 多速度场要求 CUDA。
6. sparse 模式不支持可重建网格，`max_active_cell_count` 必须为 `-1`。
7. 不支持裂纹自动分叉或任意方向扩展。
8. 不支持裂纹表面之间的 field-to-field 自接触和摩擦；分离后的两侧理论上仍可能互相穿过。
9. separation 当前为持久二值激活，没有损伤演化、部分断裂或自动愈合。

## 性能影响

启用两个速度场后，主要额外成本包括：

- 两套 FEM environment 的网格自由度和临时字段。
- 每步构建重合节点位置及 HashGrid 映射。
- separation 从粒子到节点的栅格化。
- 自由速度阶段的节点耦合。
- 每个 Gauss-Seidel 内部迭代中的节点耦合。
- cutter SDF 对所有动态粒子的接触扫描。

因此该实现的目标是获得可用的切割行为，不是提升普通 MPM 的性能。默认单速度场模式会跳过这些成本。

## 测试覆盖

`newton/tests/test_implicit_mpm.py` 当前新增三组测试：

1. `test_velocity_field_local_separation`
   - 验证 separation 为零时两场速度耦合。
   - 验证 separation 激活后两场保留不同速度。
   - 验证不会改变粒子活动标记。
2. `test_velocity_field_collider_separation`
   - 验证选中 collider 的接触只激活邻近动态粒子。
   - 验证分离状态具有持久性。
3. `test_velocity_field_collider_contact_count`
   - 验证可以要求两个 collider 同时接触并相向闭合。
   - 验证单 collider 接触和远处粒子不会触发。

上述单元测试覆盖 CPU 和 CUDA 支持的基础设备配置。implicit MPM 测试集此前结果为 153 项通过、1 项跳过。

示例本身还检查：

- 每帧粒子和剪刀状态有限。
- 粒子范围没有数值爆炸。
- settle 和 approach 阶段不能提前激活切割。
- 两端固定粒子保持不动。
- 切割不能修改粒子活动标记。
- 接触激活需要横跨至少 85% 的布料宽度。

## 当前已知问题

### 1. 末段剪刀穿模

上一版尝试加入“最后一剪保持闭合、完整抽出后再张开”的轨迹，但该修改出现问题，现已回退。

当前末段恢复为：

```text
hinge_x = cut_exit_x + 0.16 * depart_alpha
hinge_z += 0.10 * depart_alpha
angle = open_angle
```

剪刀在离场阶段始终保持打开，后端几何与布料边缘之间没有稳健余量，因此末段仍可能出现刀片或手柄与粒子视觉穿模。当前代码不包含对此问题的修复。

### 2. 示例最终接触数量阈值存在离散波动

当前 `test_final()` 要求至少 15% 的切割候选粒子直接激活 separation。默认 420 帧测试曾出现：

```text
实际直接接触粒子数 = 38
要求数量 = 45
接触横向跨度 = 0.640 m
```

当时切割横向跨度已经满足要求，但直接接触粒子数量未达到固定比例。说明粒子接触数量受离散位置影响，当前最终断言尚不完全稳定。该阈值在回退后保持原值，没有通过放宽测试掩盖问题。

### 3. 切割路径仍是预设的

两套 field 在初始化时按照完整曲线提前分区。接触只控制何时解除耦合，不决定裂纹将向哪里扩展。若要支持任意剪刀轨迹，需要动态生成材料分区、增加更多局部 field，或采用粒子邻域级裂纹图。

### 4. 裂纹面没有自接触

当前 field 分离后不再交换速度，但也没有额外求解两侧裂纹面的接触与摩擦。布料大幅折叠时，切口两侧可能互相穿过。

## 当前修改文件

```text
changelog/+mpm-multifield-cutting-c3a7f2d1.added.md
newton/_src/solvers/implicit_mpm/MULTI_VELOCITY_FIELD_CUTTING.md
newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py
newton/_src/solvers/implicit_mpm/rasterized_collisions.py
newton/_src/solvers/implicit_mpm/solve_rheology.py
newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
newton/examples/mpm/example_mpm_suspended_sheet_scissors.py
newton/tests/test_implicit_mpm.py
```

本文档只描述当前实现，不代表上述接口已经稳定或适合作为最终公开 API。
