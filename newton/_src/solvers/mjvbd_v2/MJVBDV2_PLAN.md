# MJVBDV2 设计与实施计划

## 1. 目标

新增独立的 `SolverMJVBDV2`，保留现有 `SolverMJVBD` 及其私有
MuJoCo/VBD 快照不变。

MJVBDV2 的状态所有权如下：

| 对象 | 状态与动力学所有者 | 在 VBD 中的角色 | 耦合方向 |
| --- | --- | --- | --- |
| MuJoCo 关节树及其 link | MuJoCo | 零逆质量运动碰撞体 | 关节树到其他对象，单向 |
| 非关节自由刚体 | VBD/AVBD | 动态刚体 | 与 VBD 对象双向 |
| 软体 | VBD | 动态粒子/四面体 | 与 VBD 对象双向 |
| 布料 | VBD | 动态粒子/三角形 | 与 VBD 对象双向 |
| 静态环境 | 不积分 | 静态碰撞体 | 对动态对象施加约束 |

MuJoCo 只负责选定关节树的约化坐标动力学、运动学和 link 状态，不负责
自由刚体、软体、布料或场景接触。MuJoCo 必须保留关节树 link 的质量和
惯量，因为关节动力学仍依赖这些数据。

MJVBDV2 必须同时支持：

- 动态关节树：MuJoCo 根据控制、质量、惯量和关节约束推进
  `joint_q/joint_qd`。
- 运动学关节树：根据外部关节目标或 `BodyFlags.KINEMATIC` 更新 link
  位姿和速度，不接收 VBD 碰撞反作用。
- VBD 内部刚体、软体和布料之间的双向耦合。
- MuJoCo link 与任意 VBD 对象之间的严格单向碰撞耦合。

## 2. 不可修改的基准

以下两个文件是只读基准，不允许为适配 MJVBDV2 而修改、删减、重命名或
加入 V2 分支：

- `newton/examples/vbd/example_vbd_dexforce_throw_rigid_into_bag.py`
- `newton/examples/cloth/example_cloth_dexforce_bimanual_fold_tshirt_waic_house.py`

实施前记录两个文件的 Git blob/hash 和工作区内容摘要；实现结束后再次
核对，确保这两个文件没有因 V2 工作产生新差异。它们已有的用户改动也必须
原样保留。

V2 另写两个独立示例：

- `newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_throw_rigid_into_bag.py`
- `newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house.py`

新示例可以复用只读基准中的场景资产、材料数值和轨迹数据，但不能通过修改
或在原文件中加入条件分支来实现。

## 3. 算法基线

### 3.1 VBD 基线

MJVBDV2 的刚体、软体和布料算法以实现时的公共
`newton/_src/solvers/vbd` 为基线，而不是直接以
`newton/_src/solvers/mjvbd/vbd` 为基线。

原因：

- 公共 `SolverVBD` 是完整、当前的 VBD/AVBD 行为来源。
- MJVBD 的 VBD 是固定提交的私有快照，并包含面向衣物场景的专用修改。
- 当前两份 `solver_vbd.py` 已存在数百行差异，不能默认这些差异仅影响性能
  或与完整刚体仿真完全等价。
- box-bag 验收依赖完整的自由刚体 AVBD、刚体-刚体和刚体-粒子双向接触，
  不允许因沿用 MJVBD 的单向软体路径而丢失刚体响应。

实施时将公共 VBD 源复制到 MJVBDV2 私有目录，然后只在 V2 副本中加入
body ownership 和耦合所需修改。

### 3.2 可选择移植的 MJVBD 优化

MJVBD 中下列优化可以参考，但必须逐项移植、逐项做数值等价和性能测试：

- 空 VT/EE 自接触时跳过自接触及全数组截断。
- surface/volumetric color 分区。
- 不包含四面体工作的 surface tile kernel。
- uniform active/material flag 快速路径。
- 只对实际有效 soft contact 数量启动 material、force 和 dual kernel。
- CUDA graph capture 下的保守容量路径。

未经 A/B 验证，不把这些优化直接整体复制到 V2。任何优化只要造成 box-bag
刚体结果或叠衣服结果退化，就保留公共 VBD 原实现。

### 3.3 MuJoCo 基线

MuJoCo 部分从现有 MJVBD 的私有 MuJoCo 快照复制到 V2 目录，仅在 V2
副本中增加关节树选择、compact mapping 和只同步选中关节/link 的能力。

现有 `newton/_src/solvers/mjvbd` 下的任何 Python 文件均不修改。

## 4. 包结构

计划新增：

```text
newton/_src/solvers/mjvbd_v2/
├── __init__.py
├── solver_mjvbd_v2.py
├── ownership.py
├── state_sync.py
├── collision_pipeline.py
├── MJVBDV2_PLAN.md
├── mujoco/
│   └── MuJoCo 私有快照副本
└── vbd/
    └── 公共 VBD 的私有副本
```

公开入口为：

```python
from newton.solvers import SolverMJVBDV2
```

`SolverMJVBD` 的导出、构造参数和运行行为保持不变。

## 5. 所有权划分

建议构造接口：

```python
SolverMJVBDV2(
    model,
    *,
    mujoco_articulations=None,
    mujoco_joints=None,
    joint_mode="dynamic",
    vbd_options=None,
    mujoco_options=None,
    collision_options=None,
)
```

选择优先级：

1. 显式 `mujoco_articulations`。
2. 显式 `mujoco_joints`。
3. 默认根据 `model.joint_articulation >= 0` 推断关节机构。

不能简单地把所有 `FREE` joint 交给 MuJoCo，因为非关节自由刚体也可能通过
FREE joint 表示，但它们必须由 VBD/AVBD 推进。

构造阶段必须验证：

- 每个动态 body 只有一个动力学所有者。
- MuJoCo joint 的父子 link 对 MuJoCo 可见。
- 选中的 joint 构成完整、闭合的关节树或完整 articulation。
- 不允许一个结构 joint 跨越 MuJoCo-owned 和 VBD-owned body。
- VBD 视图中禁用全部 MuJoCo-owned joint。
- MuJoCo 视图中不包含 VBD-owned 自由刚体、软体或布料。

## 6. 单个子步的数据流

采用无一帧接触延迟的顺序：

```text
输入 state_in
  |
  |-- 1. MuJoCo 推进选中的关节树
  |      q_joint[n] -> q_joint[n+1]
  |      link_q[n]  -> link_q[n+1]
  |
  |-- 2. 构造 VBD scratch input
  |      MuJoCo link 使用 n+1 位姿和速度
  |      VBD 自由刚体使用 n 位姿和速度
  |      粒子使用 n 位姿和速度
  |
  |-- 3. 在该混合状态上重新生成碰撞
  |
  |-- 4. VBD/AVBD 联合求解
  |      MuJoCo link: 零逆质量，不允许被推动
  |      自由刚体: 完整 AVBD
  |      软体/布料: 完整 VBD
  |
  `-- 5. 合并 state_out
         joint/link <- MuJoCo
         自由刚体/粒子 <- VBD
```

VBD 当前可能在刚体求解期间原地修改输入 `body_q`。V2 必须使用内部 scratch
state 或等价的隔离机制，不能让 VBD 意外覆盖调用者的 `state_in` 或
MuJoCo-owned link。

需要同时保存 MuJoCo link 的 `n` 和 `n+1` 位姿，供摩擦速度、CCD、接触历史
和 reset/rebaseline 使用。机器人瞬移、IK 重定位和 reset 后必须重建相应历史。

## 7. 接触所有权与两条运行路径

MuJoCo 场景接触关闭，VBD 是唯一接触求解器，避免重复接触力：

```python
disable_contacts=True
```

根据场景拓扑使用两条路径。

### 7.1 Soft-only fast path

适用于叠衣服场景：没有 VBD-owned 动态刚体，只有 MuJoCo link、静态环境和
粒子对象。

- 采用 MuJoCo 新 link 位姿后的稀疏 particle-shape contact pass。
- VBD 使用 external-rigid/零刚体求解快速路径。
- 布料约束和自接触仍来自完整公共 VBD 基线。
- 不分配或遍历没有意义的 AVBD 刚体求解数据。

该路径用于保证叠衣服效果和性能不比现有 MJVBD 基准退化。

### 7.2 Full VBD path

适用于 box-bag 及包含 VBD-owned 动态刚体的场景。

- 使用完整 CollisionPipeline。
- 生成 rigid-rigid、particle-shape 和需要时的 full-surface soft contact。
- VBD 以内部刚体模式运行。
- MuJoCo link 在 VBD 视图中零逆质量，但保留当前/上一帧速度信息。
- VBD-owned 自由刚体保持完整质量、惯量和 AVBD 状态。

必须支持：

- MuJoCo link 对自由刚体：单向。
- MuJoCo link 对软体/布料：单向。
- 自由刚体对自由刚体：双向。
- 自由刚体对软体/布料：双向。
- 软体/布料之间：使用 VBD 自接触/表面接触能力双向求解。

可以预过滤 MuJoCo-link/MuJoCo-link 以及两端都没有 VBD 动态自由度的接触，
但过滤前后必须做接触结果和性能对比。

## 8. 新叠衣服验收示例

新示例保持只读基准的以下内容一致：

- 同一 Dexforce W1 和 T-shirt 资产。
- 同一桌面、坐标变换和相机。
- 同一两阶段折叠轨迹和缓存 IK 目标。
- 同一帧率、substep 数、VBD iteration 数。
- 同一布料材料、摩擦切换、自接触半径和 buffer 设置。
- 同一 CUDA graph 开关和测试命令参数。

差别仅限于使用 `SolverMJVBDV2` 和 V2 所需的独立状态/碰撞管线。

效果验收：

- 全程粒子、link 和关节状态有限，无 NaN/Inf。
- 两次抓取、移动、放置和释放均完成。
- 无明显爆炸、穿桌、布料丢失或错误粘手。
- 在 approach、first-place、first-release、second-place、final 五个关键帧记录
  粒子位置和几何指标。
- 相对原 demo 基准，关键帧粒子位置 RMS/P95、衣物包围盒、桌面以下粒子数、
  自接触穿透和最终折叠区域覆盖率不得超出预先记录的容差。
- 初始建议容差：位置 RMS 不超过 5 mm、P95 不超过 15 mm，最终包围盒各轴
  不恶化超过 5%，桌面以下异常粒子数不增加。若基准自身波动更大，先用至少
  三次基准运行确定稳定容差，不能通过放宽到失去视觉等价意义来通过测试。
- 保存相同关键帧截图，进行人工视觉复核后再验收。

性能验收：

- 同一 GPU、同一 Warp/Newton 构建、同一参数、graph capture 开启。
- 预热至少 60 帧，正式运行至少三次完整流程。
- 分别记录 median、P95 GPU frame time、完整流程 wall time 和峰值显存。
- IK 预缓存和首次 kernel 编译时间单独记录，不混入稳态物理性能。
- V2 median GPU frame time 不高于基准的 1.05 倍，P95 和完整 wall time不高于
  1.10 倍，峰值显存不高于 1.10 倍。
- 若未达到性能门槛，优先恢复/移植经过数值验证的 MJVBD soft-contact 和
  surface tile 快速路径，不允许通过减少 substep、iteration、接触或自接触
  来换取性能。

## 9. 新 box-bag 验收示例

新示例保持只读基准的桌子、软包拓扑、包口固定方式、五种刚体、机器人资产、
相机和总体抓取顺序，但必须改成真实接触抓取。

### 9.1 真实抓取的硬性约束

禁止：

- 抓取期间直接写被抓刚体的 `body_q/body_qd`。
- `_carry_body`、attach/weld、临时固定关节或刚体跟随 TCP。
- 把物体瞬移到手中或袋口。
- 释放时人为写入物体位姿或投掷速度。
- 为等待中的物体每个 substep 强制恢复桌面初始位姿。

允许：

- 在仿真开始前设置五个刚体的初始桌面位姿。
- 通过机器人五指的关节轨迹闭合手掌。
- 使用手指/手掌与刚体间的 VBD 接触、摩擦和材料参数完成抓取。
- 通过手臂运动产生物体速度，打开五指后自然释放。

五个刚体从第一仿真帧开始均为 VBD-owned 动态刚体。机器人手部 link 为
MuJoCo-owned 运动碰撞体。

### 9.2 行为验收

对五个物体逐个验证以下状态机：

```text
on_table -> contacted_by_hand -> lifted -> carried -> released -> inside_bag
```

判定要求：

- `contacted_by_hand`：至少一个时间窗口内有多个手指/手掌 contact，不接受
  仅 TCP 距离判定。
- `lifted`：物体离开桌面达到规定高度，且该阶段没有脚本位姿写入。
- `carried`：物体在连续若干帧中由真实接触保持在手内。
- `released`：五指打开后手-物体接触消失，物体随后由自身动力学运动。
- `inside_bag`：最终质心和形状范围位于软包内部，不能只以释放过为成功。

最终硬性条件：

- 五个物体全部依次完成上述状态机。
- 最终 `inside == 5`。
- 五个刚体之间在包内由 VBD 双向碰撞，不能互相穿透。
- 刚体撞击软包时软包产生可见变形，并对刚体产生反作用。
- 沿用基准的最低形变要求：首个重物冲击相对基线的下沉量大于 15 mm。
- 沿用基准的恢复要求：恢复量大于首个最大下沉量的 50%。
- soft-contact 和 per-body contact buffer 无溢出。
- 机器人手与刚体、刚体与桌面、刚体与软包无持续明显穿模。
- 全部 body、particle 和 joint 状态有限。

新增诊断数据：

- 每个物体的手指接触数量及持续时间。
- 每个物体的桌面离地时间、释放时间和入袋时间。
- 手-物体、物体-桌面、物体-软包的最大穿透深度。
- 每个刚体的最终位姿和速度。
- 软包首个冲击的 baseline/minimum/recovery 高度。
- rigid 和 soft contact 峰值及 buffer 使用率。

## 10. 自动化测试

新增 focused `unittest`，不修改原 demo：

1. ownership 划分和非法跨所有权 joint 验证。
2. MuJoCo compact view 不包含 VBD 自由刚体。
3. VBD view 中 MuJoCo link 零逆质量且 MuJoCo joint 被禁用。
4. 动态关节树在有/无 VBD 对象时轨迹一致，证明无反作用。
5. 运动学 link 在当前子步推动 VBD 刚体，证明无一帧延迟。
6. 两个 VBD 自由刚体碰撞后双方状态改变。
7. VBD 刚体撞软体后双方状态改变。
8. reset、IK teleport 和 history rebaseline。
9. soft-only fast path 与公共 VBD/MJVBD 基准的数值对比。
10. full VBD path 的 contact buffer 和 graph capture 测试。
11. 新叠衣服示例完整 `test_final()`。
12. 新 box-bag 示例完整 `test_final()`，要求真实抓取状态机和 `inside == 5`。

测试使用 `unittest`，并按 Newton 约定通过 `uv run` 执行。

## 11. 实施阶段

### 阶段 A：锁定基准

- 记录两个只读 demo 的内容摘要、默认运行命令、关键帧和性能数据。
- 建立独立 benchmark harness；不向原 demo 注入测试代码。

### 阶段 B：建立 V2 隔离副本

- 新建 `mjvbd_v2` 包。
- 复制当前公共 VBD。
- 复制 MJVBD 私有 MuJoCo。
- 添加公开导出，但不改 `SolverMJVBD`。

### 阶段 C：所有权与状态同步

- 实现 articulation/joint/body 集合推导和验证。
- 实现 MuJoCo/VBD model view。
- 实现 scratch state 和按所有权 merge。
- 完成 dynamic/kinematic joint 单元测试。

### 阶段 D：接触与求解

- 实现 post-MuJoCo collision generation。
- 实现 soft-only fast path。
- 实现 full VBD path。
- 验证单向 link coupling 和 VBD 内部双向 coupling。

### 阶段 E：叠衣服 V2 示例

- 复制场景逻辑到新文件。
- 接入 V2 soft-only fast path。
- 完成效果、性能和 graph capture 验收。

### 阶段 F：box-bag V2 示例

- 新建真实五指接触抓取轨迹。
- 禁止任何物体 attach/teleport。
- 调整手指接触材料、抓取姿态和时间，但不降低物理 substep/contact 要求。
- 完成五物体逐个抓取、释放、入袋和包变形验收。

### 阶段 G：优化和交付

- 只移植通过 A/B 验证的 MJVBD 优化。
- 运行 focused tests、两个完整示例和 lint。
- 注册两个新示例及截图。
- 再次确认两个只读基准文件未被修改。

## 12. 完成定义

只有同时满足以下条件才算 MJVBDV2 完成：

- 原 `SolverMJVBD` 和两个只读 demo 没有被 V2 工作改动。
- MuJoCo 只拥有选中关节树，VBD 拥有全部非关节刚体、软体和布料。
- MuJoCo link 对 VBD 对象严格单向耦合。
- VBD-owned 对象之间双向耦合。
- 新叠衣服示例效果达标且性能不超过规定退化阈值。
- 新 box-bag 示例不用任何 attach/teleport，真实逐个抓取并使五个刚体最终都在
  可变形软包内。
- 所有 focused unittest、完整示例测试、lint 和 graph capture 验证通过。
