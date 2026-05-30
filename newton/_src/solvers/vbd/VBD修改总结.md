# VBD 修改总结

本文总结当前工作区内，相比初版 VBD 的实际修改内容、修改原因、带来的改善、潜在风险，以及各改动对应的代码锚点。重点覆盖 `newton/_src/solvers/vbd/` 内核与调度逻辑，同时补充为了让新路径真正可用而修改的 example 与回归测试。

## 1. 初版 VBD 的基线状态

以这轮修改前的 VBD 为基线，粒子布料部分可以概括为以下特征：

- 三角形膜能量只有一条主路径，即 `evaluate_neo_hookean_membrane_force_hessian()`，属于稳定 2D Neo-Hookean shell membrane。
- 自碰撞检测依赖 `TriMeshCollisionDetector` 的当前帧候选结果，求解阶段只消费“这一轮检测发现的接触”。
- 碰撞安全性主要依赖局部 `planar truncation`，没有全局安全步长（safe-step）或回溯线搜索来约束当前轮 `particle_displacements`。
- 接触缺少跨检测轮次的“持久化”机制，上一轮仍然很接近但本轮 detector 暂时没报出的对偶，会直接丢失。
- VBD 不识别 Style3D 材料模型。即便 cloth 是按 Style3D 方式 author 的，VBD 内部也不会自动切换到 Style3D stretch。

从机器人叠衣服这个目标看，上述基线有两个主要短板：

- 自碰撞接触的“进入/退出”太离散，容易在剧烈折叠、夹持翻折时出现漏接触、抖动或过大的试探位移。
- 膜材料仍然是各向同性 Neo-Hookean，不适合表达经纬向差异、剪切方向性和 Style3D 流程里常见的布料参数化方式。

## 2. 当前修改总览

当前实际落在代码里的改动，可以分成五类：

1. 自碰撞“活跃接触集”与调试指标。
2. 自碰撞全局 safe-step 与固定轮数回溯线搜索，并接回原有 truncation 管线。
3. 自碰撞持久接触缓存与回灌。
4. VBD 内部接入 Style3D stretch 膜模型，替换 Style3D-authored cloth 上原来的 Neo-Hookean 膜分支。
5. 为了让新路径可验证而新增的 example authoring 与回归测试。

另外有一个重要的“设计决策”也需要记录：

- 期间曾尝试过自碰撞 friction history 持久化，但因为明显拖慢速度，已经回退，不属于当前代码状态。

## 3. 逐项详细说明

## 3.1 自碰撞活跃接触集

### 修改位置

- `newton/_src/solvers/vbd/solver_vbd.py`
  - 粒子系统初始化里新增了活跃 VT/EE 缓冲区、上轮活跃接触缓存、safe-step/line-search 标量状态。
  - 代码锚点：`last_safe_step`、`last_active_vt_count`、`last_active_ee_count`、`active_vt_*`、`active_ee_*`、`previous_active_*`。
- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
  - 新增 `build_active_vt_contacts()`。
  - 新增 `build_active_ee_contacts()`。

### 初版行为

- 初版 VBD 在自碰撞求解时，更多是“检测到什么就处理什么”，没有显式构建一个供后续全局安全分析复用的活跃接触集。
- 因此 safe-step、持久接触、调试统计这几类能力都缺一个统一的中间表示。

### 当前行为

- `_collision_detection_penetration_free()` 在 detector 完成 VT/EE 查询后，会先把候选中距离小于 `particle_active_contact_distance` 的接触对压缩成活跃接触集。
- VT 活跃集记录：`vertex`、`tri`、`bary`、`normal`、`distance`。
- EE 活跃集记录：`edge0`、`edge1`、`st`、`normal`、`distance`。
- solver 还会在非 CUDA capture 场景下，把活跃接触数量同步到 `last_active_vt_count` 和 `last_active_ee_count`，用于调试和测试断言。

### 为什么这么改

- safe-step 需要知道“当前最危险的接触是谁、法向间距是多少、位移沿法向会推进多少”，这要求有一个结构化的活跃接触列表。
- 持久化接触需要拿“上一轮仍然有效的接触对”回灌到本轮活跃集，也需要统一的数据结构。
- 机器人叠衣服场景里布料多层折叠、自接触频繁，单靠 detector 当前一帧结果不够稳。

### 改善了什么

- 给后续 global safe-step、line search、persistent contacts 提供了统一输入。
- 能更精确地区分“只是 detector 候选”与“真正进入危险近邻区”的接触。
- 便于做行为级回归测试，而不是只能看最终 cloth 是否爆炸。

### 风险与代价

- 需要额外的 GPU/CPU 缓冲区，内存开销上升。
- 如果 buffer 预分配不足，极端接触密集场景下可能发生截断，表现为活跃接触集不完整。
- 活跃接触的阈值选择会影响灵敏度，阈值过大增加开销，过小则可能错过需要提前约束的接触。

## 3.2 全局 safe-step 与回溯线搜索

### 修改位置

- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
  - `compute_vt_safe_step()`
  - `compute_ee_safe_step()`
  - `update_line_search_alpha()`
  - `seed_truncation_ts_from_line_search()`
- `newton/_src/solvers/vbd/solver_vbd.py`
  - `_compute_particle_safe_step()`
  - `_run_particle_contact_line_search()`
  - `_penetration_free_truncation()`

### 初版行为

- 初版 VBD 的 penetration-free 逻辑更偏局部处理：先得到 `particle_displacements`，再用平面截断把危险位移压回安全范围。
- 这种方式在“单个局部接触”上是有效的，但对于多处接触同时竞争位移预算时，缺少一个全局的步长约束。

### 当前行为

- 先基于活跃 VT/EE 集估计一个全局 `safe_step_alpha`。
- safe-step 的核心思想是：如果某个接触在当前位移方向上继续沿法向逼近，则根据当前间距 `distance`、安全距离 `safe_distance`、法向相对位移 `vn` 计算允许比例。
- 多个接触通过 `atomic_min` 竞争，最终得到全局最小可接受比例。
- 之后运行固定轮数的回溯线搜索：
  - 如果候选比例已经足够接近 1（大于接受阈值），就接受当前比例。
  - 否则把 `line_search_alpha` 按 `shrink_factor` 收缩，再重复评估。
- 线搜索得到的 `accepted_alpha` 不是取代原 truncation，而是先通过 `seed_truncation_ts_from_line_search()` 写入 `truncation_ts`，然后再交给原来的 `apply_planar_truncation_parallel_by_collision()` 做最后局部修正。

### 为什么这么改

- 初版纯局部 truncation 的问题，是它没有在“位移真正应用之前”就把全局最危险的推进方向收住。
- 对折叠布料来说，多个接触经常同时出现，先做一个保守但便宜的全局 safe-step，更适合作为第一道闸门。
- 保留原 planar truncation 作为第二道防线，可以降低直接替换旧逻辑的风险。
- 固定轮数、设备侧标量的实现方式也兼顾了 CUDA Graph capture 兼容性。

### 改善了什么

- 在自碰撞高发场景下，粒子位移不再一上来就满量更新，而是先被全局接触几何约束过滤。
- 对“刚好要穿过去”的位移更敏感，减少穿透和大幅度抖动。
- 与旧的局部 truncation 组合后，更接近“两层防线”：
  - 第一层：全局 safe-step/line-search。
  - 第二层：局部按碰撞平面截断。

### 风险与代价

- safe-step 本质上是保守策略，过于保守时会降低每轮有效位移，表现为 cloth 变“慢”或“僵”。
- 线搜索轮数、接受阈值、收缩因子需要调参，参数不合适会导致：
  - 接受过快，保护不足。
  - 收缩过多，收敛变慢。
- 该 safe-step 仍然是局部线性近似，不是完整 IPC 意义上的非线性 barrier line search，所以不能把它误认为严格的全局无穿透证明。

## 3.3 接触持久化缓存与回灌

### 修改位置

- `newton/_src/solvers/vbd/solver_vbd.py`
  - `_append_persistent_particle_contacts()`
  - `_snapshot_active_particle_contacts()`
  - `_collision_detection_penetration_free()` 中插入了 persistent 回灌和 snapshot。
- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
  - `append_persistent_vt_contacts()`
  - `append_persistent_ee_contacts()`
  - `has_active_vt_contact()`
  - `has_active_ee_contact()`

### 初版行为

- 某个接触对只要这轮 detector 没报出来，就会从求解视野里完全消失。
- 这在布料多层接近接触、法向轻微摆动、BVH refit 结果轻微波动时，会导致接触“闪烁”。

### 当前行为

- 每次完成活跃接触构建后，solver 会把活跃 VT/EE 对 snapshot 到 `previous_active_*`。
- 下一轮碰撞检测之后，会先把“上一轮活跃接触”拿出来做一次纯几何复验：
  - 如果接触对本轮依然小于 `particle_persistent_contact_distance`，且当前活跃集中还没有它，就重新追加进去。
- 这相当于给接触增加一个短期记忆。

### 为什么这么改

- detector 的候选结果天然会有离散波动，尤其在 cloth 轻微滑动、褶皱贴近但尚未深穿时，漏掉一轮就可能让 solver 突然放松约束。
- 对机器人叠衣服来说，这种“一会儿有接触一会儿没接触”的现象会直接伤害褶皱稳定性和夹持区域附近的接触连续性。

### 改善了什么

- 接触连续性更好，减少闪烁式丢接触。
- 与 safe-step 联合使用时，活跃接触集更稳定，回溯线搜索也更有意义。
- 对多层布料接近但未完全分离的状态更友好。

### 风险与代价

- 持久化阈值过大时，会把实际上已经不该继续约束的接触“黏”太久，导致不必要的保守。
- 需要额外线性扫描检查活跃集中是否已存在对应 pair，接触很多时会增大一些开销。
- 当前持久化是几何重验证，不包含更复杂的历史切向状态，因此它解决的是“接触存在性连续”，不是“摩擦历史连续”。

## 3.4 Style3D stretch 膜模型接入 VBD

### 修改位置

- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
  - 保留原 `evaluate_neo_hookean_membrane_force_hessian()`。
  - 新增 `evaluate_style3d_stretch_force_hessian()`。
  - `solve_elasticity_tile()` 和 `solve_elasticity()` 增加 `tri_material_models`、`tri_style3d_aniso_ke` 输入，并在三角形层面分支选用膜模型。
- `newton/_src/solvers/vbd/solver_vbd.py`
  - 新增 `_init_particle_membrane_materials()`。
  - 在 solver 初始化时读取 `model.style3d.tri_aniso_ke`。
  - 在两处 elasticity kernel launch 把材料模型数组传入。

### 初版行为

- 所有 cloth triangle 一律走稳定 Neo-Hookean 膜模型。
- 这条路径对于普通布料是稳定的，但本质是各向同性的，无法直接表达 Style3D 材料里的 `ku / kv / ks` 方向性。

### 当前行为

- SolverVBD 初始化时会检查 `model` 是否带 `style3d.tri_aniso_ke`。
- 如果某个三角形的 `tri_aniso_ke` 任一分量大于 0，则把该三角形标记为 Style3D 膜模型。
- 在 `solve_elasticity_tile()` 和 `solve_elasticity()` 内部：
  - Style3D 三角形走 `evaluate_style3d_stretch_force_hessian()`。
  - 其余三角形继续走 `evaluate_neo_hookean_membrane_force_hessian()`。
- `evaluate_style3d_stretch_force_hessian()` 的力模型来源于 Style3D stretch：
  - 沿 `Fu` 的拉伸项 `ku * (|Fu| - 1)`。
  - 沿 `Fv` 的拉伸项 `kv * (|Fv| - 1)`。
  - `Fu` 与 `Fv` 的剪切耦合项 `ks * dot(Fu_dir, Fv_dir)`。
- Hessian 不是完整二阶导，而是一个按 `dFu_dx`、`dFv_dx` 组合出来的局部 PSD 近似，再叠加统一阻尼项。

### 为什么这么改

- 用户目标不是“把 Style3D 参数近似映射到旧膜模型”，而是“在 VBD 里直接用 Style3D stretch 材料”。
- 对服装折叠、拉拽、翻折等操作，经纬向刚度差异很关键。仅靠各向同性 Neo-Hookean 很难调出接近真实织物的方向性。
- 保留逐三角形分支而不是全局切换，可以做到：
  - 老场景不受影响。
  - 新场景只要 author Style3D 属性即可自动切过去。

### 改善了什么

- VBD 现在可以直接消费 Style3D cloth builder 写入的 `tri_aniso_ke`。
- 允许在经向、纬向、剪切三个方向上独立调膜响应，更接近服装材料实际调参方式。
- 不需要把运行时求解器切到 `SolverStyle3D`，依然保留 VBD 的整体求解框架和已有自碰撞改造。

### 风险与代价

- 当前 Hessian 是 PSD 近似，不是严格完整二阶导。优点是更稳，缺点是局部曲率信息被简化，可能影响收敛速度与真实刚度感。
- `tri_material_models` 的判定逻辑是“任一分量大于 0 就走 Style3D 分支”，这很直接，但也要求 authoring 端不要误填半残缺参数。
- 混合场景中若部分 triangle 走 Style3D、部分走 NH，材料过渡区可能出现响应不连续，需要作者自己保证材质 authoring 一致。
- 当前接入的是 Style3D stretch 膜项，不等价于完整 Style3D solver 的所有行为；不能把“VBD + stretch 分支”理解成“完整复刻 Style3D”。

## 3.5 `example_cloth_franka_mujoco_cloth.py` 的 authoring 改造

### 修改位置

- `newton/examples/cloth/example_cloth_franka_mujoco_cloth.py`

### 初版行为

- 示例里的 cloth 仍用 `scene.add_cloth_grid(...)` 和普通 VBD cloth 参数构造。
- 即便 VBD 内部已经支持 Style3D stretch，这个例子也不会自动命中那条路径。

### 当前行为

- 示例先注册 `SolverStyle3D.register_custom_attributes()`。
- 然后用 `style3d.add_cloth_grid(...)` author cloth。
- 运行时求解器仍然是 `SolverVBD`。

### 为什么这么改

- 只有 builder 端真正把 `style3d:tri_aniso_ke` 写进 model，VBD 才能识别到该三角形应走 Style3D 分支。
- 这个例子最接近机器人叠衣服目标，所以优先让它吃到新的材料模型。

### 改善了什么

- 机器人 cloth 场景与 VBD 新材料路径打通。
- 后续调 cloth 折叠效果时，可以直接从 `tri_aniso_ke` 入手，而不是继续围绕各向同性膜参数打补丁。

### 风险与代价

- 场景参数语义发生变化。对 Style3D-authored cloth 来说，重点参数从 `tri_ke` 转为 `tri_aniso_ke`。
- 如果用户误以为这里仍然是旧 VBD 膜模型，可能按错参数方向调试。

## 3.6 `example_cloth_twist.py` 的 authoring 改造

### 修改位置

- `newton/examples/cloth/example_cloth_twist.py`

### 初版行为

- twist example 使用普通 `scene.add_cloth_mesh(...)`，只能走初版 VBD 膜模型。

### 当前行为

- 改为注册 Style3D custom attributes。
- 改为调用 `style3d.add_cloth_mesh(...)`。
- 显式设置 `tri_aniso_ke`、`edge_aniso_ke`、`tri_kd`、`edge_kd`。
- 由于 `square_cloth.usd` 的几何点位于 `XZ` 平面，而 Style3D panel 需要一个 2D 正面积 rest-space，示例中额外把 `panel_verts` 显式设为顶点的 `ZX` 坐标，避免 panel 空间里三角形面积全为负或退化。

### 为什么这么改

- twist example 是最适合观察膜方向性和大变形自碰撞稳定性的 smoke case。
- 让它走 Style3D stretch 分支，可以快速验证 VBD 材料分派与 example authoring 是否真的打通。

### 改善了什么

- 现在这个例子不只是“能跑 VBD”，而是“能跑带 Style3D stretch 的 VBD”。
- 便于后续专门观察 twist 工况下的各向异性响应。

### 风险与代价

- 这里的 `panel_verts` 是按当前几何平面显式指定的，如果未来换 mesh 资源，不能假设 `ZX` 仍然正确，可能需要重新选择 panel 空间。
- 这个例子目前只是把 authoring 接通了，不代表参数已经是最终物理标定值。

## 3.7 回归测试补强

### 修改位置

- `newton/tests/test_solver_vbd.py`

### 新增测试

- `_self_contact_safe_step_tracks_active_contacts()`
  - 验证 self-contact 场景下确实建立了活跃接触，且 `last_safe_step` 被压到 1 以下。
- `_persistent_self_contacts_reactivate_from_previous_active_set()`
  - 验证上一轮活跃接触在 detector 清空后，可以通过 persistent 逻辑重新回到活跃集。
- `_style3d_cloth_uses_style3d_membrane_in_vbd()`
  - 验证 Style3D-authored cloth 进入 SolverVBD 后，`tri_material_models` 会被正确标记，`tri_style3d_aniso_ke` 会被正确读入，且一步求解后粒子状态是有限值。

### 为什么这么改

- 这些改动都不是单纯数值微调，而是控制流和材料分派层面的结构变化，必须有行为级测试兜底。
- 尤其是 self-contact 与 Style3D stretch 两类路径，肉眼看 example 跑起来不等于逻辑正确。

### 改善了什么

- 降低后续重构时把 safe-step、persistent contacts、Style3D branch 悄悄破坏的风险。
- 给 CI 或本地回归提供了更窄、更快、更有针对性的检查点。

### 风险与代价

- 测试仍是窄回归，不是完整的 cloth quality benchmark。
- 当前测试能证明“路径被走到”和“基础行为成立”，但不能完全证明长期稳定性和视觉效果最优。

## 4. 当前代码相对初版 VBD 的核心收益

从结果导向看，这轮改动带来四个最核心的提升：

### 4.1 自碰撞稳定性更强

- 活跃接触集 + 全局 safe-step + 原局部 truncation 的组合，使得 cloth 在高接触密度场景下不再只靠最后一步局部修补。
- 这对于折叠、翻面、夹持时的近距离多接触更重要。

### 4.2 接触连续性更强

- 持久接触缓存避免了“上一轮还压着，这一轮 detector 没报出就突然放开”的闪烁现象。

### 4.3 材料表达能力提升

- VBD 不再局限于各向同性膜模型，已经能识别并使用 Style3D 的三向刚度参数。

### 4.4 机器人折衣场景更贴近真实需求

- 机器人 cloth example 已接上 Style3D authoring。
- 后续优化重点可以从“防爆炸”进一步转向“材料方向性、折痕保真、夹持传力”。

## 5. 当前版本的主要风险

当前版本虽然比初版更接近目标，但还存在下面这些明确风险：

### 5.1 safe-step 偏保守

- 全局 safe-step 不是免费的。越保守，单步位移越小，视觉上越容易显得慢。
- 如果后续 cloth 太“拖”，优先排查的就是这组阈值与 line-search 参数。

### 5.2 Style3D stretch 只接入了膜项

- 现在的材料升级重点在 stretch branch，不是把完整 Style3D 求解器逐项搬进 VBD。
- 因而它提升了膜响应表达，但不等价于“VBD 已经完整变成 Style3D”。

### 5.3 Style3D Hessian 是近似的

- 为了兼顾稳定性和 VBD 的 per-vertex solve 结构，这里用了 PSD 近似 Hessian。
- 对视觉稳定性通常是利好，但在材料“硬度感”和收敛速度上仍可能与完整二阶模型有差异。

### 5.4 接触持久化只解决存在性，不解决完整摩擦历史

- 当前持久化的重点是 pair 级几何重用，不包含切向 stick/slip 历史。
- 这也是为什么之前 friction history 路径虽然方向合理，但因为速度回退而未保留在当前版本里。

## 6. 当前没有保留在代码中的尝试

需要特别说明，下面这个方向在本轮中被尝试过，但已经回退，不属于当前代码：

- 自碰撞 friction history 持久化。

回退原因很直接：

- 它确实有机会改善接触切向连续性。
- 但在当前实现形态下带来了明显速度下降，不适合作为当前基线继续保留。

这意味着当前版本的策略是：

- 先把“法向安全性”和“接触存在性连续”做扎实。
- 摩擦历史连续性留到后续以更低成本的方式再设计。

## 7. 建议如何理解这轮改动

如果用一句话概括当前版本，相比初版 VBD，它不是简单地“加了几个参数”，而是把粒子 cloth 的核心路径从：

- 各向同性膜模型 + 单轮 detector + 纯局部截断

推进到了：

- 可选 Style3D 各向异性膜模型 + 活跃接触集 + 全局 safe-step/回溯线搜索 + 持久接触回灌 + 原局部截断兜底

这套结构更适合继续往机器人叠衣服目标推进，因为它同时覆盖了两个关键问题：

- 几何安全性与接触连续性。
- 材料方向性表达能力。

## 8. 本文对应的主要代码锚点

- `newton/_src/solvers/vbd/solver_vbd.py`
  - `_init_particle_membrane_materials()`
  - `_collision_detection_penetration_free()`
  - `_append_persistent_particle_contacts()`
  - `_compute_particle_safe_step()`
  - `_run_particle_contact_line_search()`
  - `_penetration_free_truncation()`
- `newton/_src/solvers/vbd/particle_vbd_kernels.py`
  - `evaluate_neo_hookean_membrane_force_hessian()`
  - `evaluate_style3d_stretch_force_hessian()`
  - `build_active_vt_contacts()`
  - `build_active_ee_contacts()`
  - `append_persistent_vt_contacts()`
  - `append_persistent_ee_contacts()`
  - `compute_vt_safe_step()`
  - `compute_ee_safe_step()`
  - `update_line_search_alpha()`
  - `seed_truncation_ts_from_line_search()`
- `newton/examples/cloth/example_cloth_franka_mujoco_cloth.py`
- `newton/examples/cloth/example_cloth_twist.py`
- `newton/tests/test_solver_vbd.py`
