# Newton 多求解器耦合设计：从零看懂

> 这份文档带你从"为什么需要耦合"开始，一步步看懂 `newton/examples/multiphysics/` 里的耦合设计。
> 代码来自 `newton/_src/solvers/coupled/`（公开入口 `newton.solvers.experimental.coupled`）。
> 尽量用比喻和带注释的代码讲，遇到术语都会先解释。

---

## 0. 先搞清楚：什么是"耦合"，为什么要它？

想象一个场景：一个刚体铁球砸下来，下面是一块布（布料模拟），布下面是一堆沙子（颗粒模拟）。

- **刚体**（铁球）：MuJoCo 或 Kamino 擅长算，它们处理刚体碰撞、关节、铰接体很在行。
- **布料/软体**：VBD 或 XPBD 擅长算，它们会算形变、弹性、弯曲。
- **沙子/流体**：MPM 擅长算，它用网格-粒子混合方法处理大变形材料。

问题是：**没有一个求解器能同时把这三件事都算好**。如果你用 VBD 去算刚体，铰接关节会很别扭；用 MuJoCo 去算布料，它根本不懂形变。

**耦合（coupling）就是让这几个各有所长的求解器在同一场景里合作**：铁球砸到布上，布要"感受到"铁球的重量和冲击然后凹陷，铁球也要"感受到"布的支撑而被托住；布压到沙子，沙子要被推开，布要被沙子顶住。

这和"各算各的然后拼起来"不一样--**两个求解器要互相施加力，互相影响**，这才是两路耦合（two-way coupling）。

Newton 的耦合框架就是干这件事的，它提供了两套算法（Proxy 和 ADMM），让你能把任意两个求解器拼起来。

---

## 1. 整体架构：三个关键角色

在讲算法之前，先认识三个贯穿始终的角色。理解了它们，后面的算法就是"怎么用这三个角色搭积木"。

### 角色 1：共享的 `Model` -- 大家共用同一本"户口本"

Newton 不给每个求解器单独建一个模型。**整个场景只有一个 `Model`**，里面所有的 body（刚体）、particle（粒子/布料顶点/软体顶点）、joint（关节）、shape（碰撞形状）都有一个**全局唯一的 id**。

> 💡 **比喻**：像一栋合租公寓，只有一张户口本，写着所有房间和所有室友。每个室友（求解器）只"认领"其中几个房间，但房间号是全楼统一的。

代码里你看长这样（来自 `mujoco_vbd_coupled_solver.py`）：

```python
builder = newton.ModelBuilder()
builder.add_ground_plane()

# 记住刚体 id 的起始号
rigid_body_start = builder.body_count      # 比如 0
self._emit_rigid_bodies(builder)           # 加几个刚体盒子，id 是 0,1,2
self._emit_articulated_chain(builder)      # 加一个铰接摆链，id 是 3,4,5
rigid_body_end = builder.body_count        # 6

self._emit_cloth(builder)                  # 加布料（产生 particle，不产生 body）
self._emit_soft_bodies(builder)            # 加软体（也是 particle）

model = builder.finalize()                 # 整个场景就这一个 model
```

注意：body 和 particle 是**不同的东西**。刚体是 body（有位姿、惯量、6 自由度），布料顶点是 particle（只有位置+速度+质量，3 自由度）。一个场景里可以同时有 body 和 particle。

### 角色 2：`ModelView` -- 给每个求解器戴一副"滤镜眼镜"

虽然只有一个 Model，但每个求解器不能看到全部--否则 VBD 会去管刚体，MuJoCo 会去管布料，乱套了。

`ModelView` 就是给每个求解器定制的"视图"：**底层还是同一个 Model，但戴上眼镜后，不属于自己的东西看起来是"静止的/没有质量的"，属于别人的东西被屏蔽掉**。

> 💡 **比喻**：ModelView 像一副 AR 眼镜。透过 VBD 的眼镜看，刚体盒子还在那儿（能看到形状去碰撞），但标记成"这是别人的，你别去推动它"；透过 MuJoCo 的眼镜看，布料顶点根本看不见（因为 MuJoCo 不处理 particle）。

关键性质：**ModelView 永远不修改父 Model**。它用"copy-on-write"（写时复制）--你第一次要改某个数组时，它克隆一份到自己手里再改，父 Model 的原件纹丝不动。代码在 `model_view.py`：

```python
class ModelView:                              # ← 注意：没有 (Model)，不是继承
    def __init__(self, parent: Model, name: str):
        object.__setattr__(self, "_parent", parent)   # 持有父 Model 的引用（不是复制）
        object.__setattr__(self, "_overrides", {})    # 空的覆盖字典，一开始什么都没有

    def __getattr__(self, name):
        # 读属性：先看我自己有没有覆盖，没有就透传到父 Model
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return self._count_limited_attribute(name, overrides[name])
        return self._count_limited_attribute(name, getattr(parent, name))  # 直接返回父的数组

    def __setattr__(self, name, value):
        # 写属性：存到我的 _overrides 里，父 Model 不受影响
        ...
        object.__getattribute__(self, "_overrides")[name] = value
```

#### ModelView 和 Model 到底是什么关系（容易搞混，单独说清楚）

初学者常把 ModelView 理解成"继承 Model、复制了一部分数据来管"，这两个理解都要纠正：

##### 1. 不是继承，是组合（持有引用）

`class ModelView` 后面**没有 `(Model)`**，它不是 Model 的子类。它是独立类，内部 `_parent` 字段**指向**父 Model（像指针，不是拷贝）。这是 has-a（持有）关系，不是 is-a（继承）关系。

##### 2. 默认零复制，读取直接透传

构造完一个 ModelView，`_overrides` 是空的，**什么都没复制**。读任何属性时直接返回父 Model 的同一个数组对象（同一个 GPU 指针）。所以 `view.body_mass` 和 `model.body_mass` 在没被覆盖时是**同一个数组**，零拷贝，1000 个 body 的质量数组视图不占额外显存。

##### 3. 只在"写"时，复制被改的那一个数组

只有当你真要修改某个属性时，才克隆那**一个**数组。看 `_cow_array`（`model_view.py:340`）：

```python
def _cow_array(self, name: str) -> wp.array:
    parent = object.__getattribute__(self, "_parent")
    overrides = object.__getattribute__(self, "_overrides")
    array = overrides.get(name)
    if array is None:                       # 第一次改这个数组
        array = wp.clone(getattr(parent, name))   # 只克隆这一个！
        overrides[name] = array
    return array                            # 以后再改，返回已克隆的这份
```

具体例子，调用 `view.disable_body_dynamics([3,4,5])` 时：

```text
调用前：
  model.body_inv_mass = [a, a, a, a, a, a, a, a]   ← 父 Model 的原数组
  view._overrides = {}                               ← 空

调用 disable_body_dynamics([3,4,5])：
  _cow_array("body_inv_mass") 触发克隆
  model.body_inv_mass = [a, a, a, a, a, a, a, a]   ← 原数组没动！
  view._overrides["body_inv_mass"] = [a, a, a, 0, 0, 0, a, a]  ← 克隆版，3,4,5 置零

之后读 view.body_inv_mass -> 返回克隆版 [a,a,a,0,0,0,a,a]
而读 model.body_inv_mass -> 还是原版 [a,a,a,a,a,a,a,a]
```

**只有 `body_inv_mass` 这一个数组被复制了**，`body_mass`、`particle_q` 等其他属性还是共享父 Model 的原数组。

##### 4. "只管理一部分数据"的准确理解

不是"管理一部分数据"，而是**"让求解器看到一个被裁剪/静音过的视图"**。两种手段：

- **count 截断**：视图声明 `body_count = 6`（父有 10 个），读 `view.body_mass` 时自动切片到前 6 个（切片本身也不复制）。
- **标志位/置零**：`disable_body_dynamics` 把某些 body 逆质量置零、`mark_proxy_bodies` 打 PROXY 标志--都在克隆的那一个数组上改。

> 🎬 **账本比喻**：把 Model 想象成一本厚厚的账本。ModelView 不是账本的复印件，而是**一个助理 + 一沓便利贴**：助理拿着账本本身的引用，平时你问"body 3 的质量"，他直接翻原账本念给你（透传，零复制）；只有当你说"把 body 3,4,5 标记冻结"时，他才把"质量"那一页复印一份，在复印件上把 3,4,5 涂掉贴上便利贴，下次问质量他念复印件，问别的（位置等）还是翻原账本。老板（父 Model）的原账本从来没被改过，别的助理（别的 ModelView）各有各的便利贴互不干扰。
>
> 好处：场景里可能同时有 4 个求解器（mjc/vbd/xpbd/mpm），若每个都复制一份 Model，显存翻 4 倍；用 ModelView，共享的 99% 数据只存一份，每个视图只多花几个被改数组的显存。

ModelView 还提供一堆便捷方法来"调整视图"，举几个常用的（都在 `model_view.py`）：

```python
view.disable_body_dynamics(body_indices)   # 让这些 body 在本视图里"动不了"
# 原理：把它们的逆质量/逆惯量置零。子求解器看到逆质量=0，
#       就知道"这个 body 无限重，推不动"，于是不动它。
#       但质量/惯量本身保留，作为元数据供查询。

view.zero_particle_mass(particle_indices)  # 让这些粒子在本视图里"没有质量"
view.disable_particles(particle_indices)   # 顺便把 active 标志清掉
view.mark_proxy_bodies(body_indices)       # 给这些 body 打上 PROXY 标记
view.scale_body_mass(body_indices, 0.5)    # 把这些 body 的质量缩放 0.5
```

> 🔑 **关键**：这些操作都只影响"这个视图"，换个视图看，body 的质量还是原来的值。

### 角色 3：`CouplingInterface` -- 求解器与框架之间的"插座规格"

框架要指挥各种求解器（VBD/XPBD/MPM/MuJoCo），但框架不能去读每个求解器的内部代码。怎么办？定一套"插座规格"（接口契约），求解器实现这套插座，框架就能插上去用。

`CouplingInterface` 就是一个 mixin（混入类），求解器继承它，按需重写里面的"钩子方法"（hook）。框架调用钩子，求解器响应。看 `interface.py` 里的几个核心钩子：

```python
class CouplingInterface:

    def coupling_eval_effective_mass(self, endpoint_kind, endpoint_index, ...):
        """问求解器：这些端点（body 或 particle）的'等效质量'是多少？
        
        '等效质量'就是：通过关节连起来的铰接体，从某个端点推它，
        它'感觉起来'有多重。就像推门--推门把手（离铰链远）很轻松，
        推门轴附近（离铰链近）很费劲，虽然门的总质量没变，
        但'等效质量'不一样。框架需要知道这个，才能给代理体分配合适的虚拟惯量。
        """
        # 默认实现：直接用 body_inv_mass 取倒数，简单情况够用
        ...

    def coupling_eval_gravity_acceleration(self, out_body_acc, out_particle_acc):
        """问求解器：你内部会给这些实体施加多大的重力加速度？
        
        框架问这个是为了'抵消'--如果求解器内部已经加了重力，
        框架就不能再加一次，否则重力被算了两遍。
        """
        ...

    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global, out_body_f, ...):
        """告诉求解器：把你刚算出来的代理体动量变化，转换成力交给我。
        
        '收割'（harvest）就是：目标求解器把代理体推动了 Δv，
        那么 F = m·Δv / dt 就是代理体受到的力，也就是源求解器该受到的反馈力。
        """
        # 默认实现：F = m·Δv, τ = I·Δω / dt（力 = 质量×速度变化/时间）
        ...

    def coupling_rewind_proxy_body(self, body_local_to_proxy_global, state, coupling_forces, ...):
        """告诉求解器：把你之前施加的滞后反馈力、重力、外力都'撤回'，
        因为这些力等会儿目标求解器自己会再算一遍，不撤就会重复计算。"""
        ...
```

> 💡 **比喻**：CouplingInterface 像是快递柜的标准化接口。不管你内部是顺丰还是京东，只要你的包裹符合柜子的尺寸规格（实现钩子），柜子（框架）就能收发你的包裹（交换力/质量/状态）。求解器**不需要**知道别的求解器长什么样，只要实现这套插座就行。

---

## 2. 框架的骨架：`SolverCoupled` 基类

两套耦合算法（Proxy 和 ADMM）有大量共性工作：划分所有权、建视图、分发状态、收集结果、过滤接触……这些全抽到基类 `SolverCoupled` 里。两个子类只覆盖一个方法 `_step_coupled`（"具体怎么一步步耦合"），其余都复用。

### 2.1 Entry：声明"谁拥有什么"

你用 `Entry` 来告诉框架：有个求解器，它叫某某，它拥有这些 body/particle/joint。

```python
# 来自 solver_coupled.py，简化展示
class SolverCoupled:
    @dataclass(frozen=True)
    class Entry:
        name: str                                    # 求解器名字，比如 "mjc"、"vbd"
        solver: Callable[[ModelView], SolverBase]    # 工厂函数：传入视图，返回求解器实例
        bodies: Sequence[int] = ()                   # 拥有哪些 body（全局 id）
        particles: Sequence[int] = ()                # 拥有哪些 particle
        joints: Sequence[int] = ()                   # 拥有哪些 joint
        shapes: Sequence[int] = ()                   # 拥有哪些 shape
        configure_view: Callable | None = None       # 可选：额外调整视图的回调
        substeps: int = 1                            # 这个求解器每步内部走几个子步
        in_place: bool = False                       # 是否原地步进（state_0 当 state_1 用）
```

> 注意 `solver` 是个**工厂闭包**，不是求解器实例。因为框架要先建好 `ModelView`，再把视图传给求解器构造函数。所以你写 `lambda v: SolverVBD(model=v, iterations=10)`，框架会在合适的时机调用它。

### 2.2 所有权校验：一个实体只能有一个主人

`_build_owner_map` 给每种实体建一张"归属表"：

```python
def _build_owner_map(self, count, owned_by_entry):
    owner = [-1] * count    # 初始都是"无主"
    for entry_idx, indices in enumerate(owned_by_entry):
        for raw_index in indices:
            index = int(raw_index)
            if owner[index] != -1:
                raise ValueError(f"Index {index} is owned by more than one coupled solver entry")
            owner[index] = entry_idx    # 标记：这个实体归第 entry_idx 个求解器
    return owner
```

比如 body 0,1,2,3,4,5 归 "mjc"，body 6,7,8 归 "vbd"，那 `body_owner = [0,0,0,0,0,0,1,1,1]`。

### 2.3 建视图：给每个 entry "戴眼镜"

`_build_entries` 是最核心的构造逻辑。对每个 entry，它要做这几件事（我按顺序讲）：

```python
def _build_entries(self):
    for idx, cfg in enumerate(self._entry_configs):
        # 1. 创建这个 entry 的视图
        view = ModelView(model, cfg.name)

        # 2. 决定哪些"别人的 body"要作为"代理体"保留在本视图里
        #    （由子类决定，Proxy 和 ADMM 有不同策略）
        proxy_body_keep = self._entry_proxy_body_keep_indices(cfg.name)

        # 3. 对"既不属于我、又不是代理"的 body：让它在本视图里静止
        body_dynamics_disabled = [
            i for i, owner in enumerate(self._body_owner)
            if owner != idx and i not in proxy_body_keep
        ]
        if body_dynamics_disabled:
            view.disable_body_dynamics(...)    # 置零逆惯量，让它"推不动"

        # 4. 对代理 body：打上 PROXY 标记，保留动态
        if proxy_body_keep:
            view.mark_proxy_bodies(...)

        # 5. particle 同理（disable 或 mark_proxy）
        # 6. joint 同理（disable_joints）

        # 7. 尝试压缩视图（让编号变紧凑，见 2.5）
        index_lists = self._compact_entry_view_if_needed(...)

        # 8. 调用用户自定义的视图调整回调
        if cfg.configure_view is not None:
            cfg.configure_view(view)

        # 9. 用这个视图实例化求解器
        solver = cfg.solver(view)    # 比如 SolverVBD(model=view, iterations=10)
        _require_supports_coupling(solver)   # 必须实现了 CouplingInterface
```

> 💡 **第 3 步是精髓**：为什么要把别人的 body "静止"？因为 VBD 视图里也会看到刚体盒子（为了碰撞），但 VBD 不应该去推动刚体。把逆惯量置零后，VBD 算 `F = m·a` 时 `1/m = 0`，加速度就是 0，刚体纹丝不动。但它的形状、质量（作为元数据）还在，碰撞检测照样能用。这就是"看得见摸不着"。

### 2.4 状态分发与调和：父状态 ↔ 各 entry 状态

每个 entry 有自己的 `state_0`（输入）和 `state_1`（输出）。框架在每个 step 开头把父状态"分发"给各 entry，结尾再把各 entry 结果"调和"回父状态。

```python
def step(self, state_in, state_out, control, contacts, dt):
    self._distribute_state(state_in, dt=dt)        # 父状态 -> 各 entry.state_0
    self._step_coupled(...)                         # 子类算法：真正干活的
    _copy_state(state_in, state_out)
    self._reconcile_state(state_out)                # 各 entry.state_1 -> 父状态（只写 owned 的）
```

`_reconcile_state` 只把 entry **拥有**的实体写回父状态，代理体的状态不写回（因为代理体只是镜像，真正的状态在源求解器那儿）：

```python
def _reconcile_state(self, state_out):
    for entry in self._entries.values():
        # 只把 entry 拥有的 body 状态 scatter 回父状态
        if entry.body_indices.shape[0] > 0:
            wp.launch(
                _scatter_body_state_mapped,
                dim=entry.body_indices.shape[0],
                inputs=[entry.body_indices,         # 这个 entry 拥有哪些全局 body id
                        entry.body_global_to_local, # 全局 id -> 本 entry 局部 id
                        entry.state_1.body_q,       # 本 entry 算出来的 body 位姿
                        entry.state_1.body_qd,      # 本 entry 算出来的 body 速度
                        state_out.body_q,           # 写回父状态
                        state_out.body_qd],
                device=self.model.device,
            )
        # particle、joint 同理
```

> 💡 **比喻**：像一个项目经理（框架）把一个大任务（父状态）拆给几个工程师（entry），每人发一份和自己负责部分相关的资料（分发）；工程师各自算完，把结果交回来，项目经理把各人的结果拼回总报告（调和）。代理体就像工程师手里的"参考模型"，算完不用上交，因为真品在别人那儿。

### 2.5 视图压缩（可选优化）

如果 entry 只拥有少数 body，让子求解器看着全部 body 编号会很浪费（比如 VBD 要遍历所有 body，哪怕大部分是静止的）。`_compact_entry_view_if_needed` 会尝试把 entry 相关的实体重排成**从 0 开始的连续编号**，让子求解器只看到一个小模型。

```
全局编号:  body 0,1,2,3,4,5 (mjc), 6,7,8 (vbd)
                        ↓ 压缩 vbd 视图
vbd 视图:  body 0,1,2  (对应全局 6,7,8)
```

这个压缩条件比较苛刻（关节的父子都得在集合里、铰接体完整、多 world 布局齐整等），满足不了就回退到"不压缩，只是把别人的 body 静止"，照样能跑，只是慢一点。

---

## 3. 算法一：`SolverCoupledProxy`（代理体耦合）

这是更容易理解的一种耦合，核心思想非常直观。

### 3.1 核心思想：用"替身"让两个求解器对话

> 💡 **比喻**：假设你是导演（源求解器，管刚体），要拍一场"铁球砸布"的戏，但布料的动作由另一个摄影师（目标求解器，管布料）负责。你没法直接把手伸进他的镜头去推布。怎么办？你做一个**和铁球一样大的泡沫球**（代理体），递给摄影师说"这个泡沫球的位置和速度我每帧告诉你，你拿它去和布料算碰撞，算完告诉我泡沫球被布'顶'了多狠，我再把这股力加到真铁球上"。

- 泡沫球就是**代理体（proxy body）**，它存在于目标求解器（布料）的视图里。
- 泡沫球的质量不是真铁球的质量，而是导演告诉它的"等效质量"（可能还缩放一下，`mass_scale`）。
- 每帧：导演把真铁球的位姿/速度同步给泡沫球 → 摄影师拿泡沫球和布算碰撞，泡沫球被布顶了 → 摄影师报告"泡沫球速度变了 Δv" → 导演换算成力 `F = m·Δv/dt`，加到真铁球上 → 真铁球下一步就被这股力影响。

这就是**滞后冲量耦合（lagged impulse coupling）**：力的传递隔着一步，像回合制游戏。

### 3.2 配置长什么样

来自 `mujoco_vbd_coupled_solver.py`，我加了详细注释：

```python
self.solver = SolverCoupledProxy(
    model=self.model,
    entries=[
        # ---- 刚体求解器 entry ----
        SolverCoupledProxy.Entry(
            name="mjc",                              # 叫 "mjc"
            solver=lambda v: SolverMuJoCo(model=v,   # 用 MuJoCo，v 是框架建好的视图
                                          use_mujoco_contacts=False, njmax=200),
            bodies=[int(i) for i in rigid_body_indices.numpy()],  # 拥有刚体 body
            joints=list(range(self.model.joint_count)),           # 拥有关节
        ),
        # ---- 柔体求解器 entry ----
        SolverCoupledProxy.Entry(
            name="vbd",                              # 叫 "vbd"
            solver=lambda v: SolverVBD(model=v, iterations=10, ...),
            bodies=[int(i) for i in vbd_body_indices.numpy()],    # 拥有非刚体的 body（本例为空）
            particles=list(range(self.model.particle_count)),     # 拥有所有 particle（布料+软体）
        ),
    ],
    coupling=SolverCoupledProxy.Config(
        proxies=[
            SolverCoupledProxy.Proxy(
                source="mjc",                        # 源：刚体求解器（真铁球在那）
                destination="vbd",                   # 目标：柔体求解器（泡沫球放这儿）
                bodies=[int(i) for i in rigid_body_indices.numpy()],  # 把哪些 body 镜像成代理体
                mass_scale=args.mass_scale,          # 代理体虚拟质量 = 源等效质量 × 这个值
                mode=args.coupling_mode,             # "lagged" 或 "staggered"
                collision_pipeline=lambda model: newton.examples.create_collision_pipeline(model, self.args),
                collide_interval=1,                  # 每步都重新检测碰撞
            )
        ],
        iterations=args.proxy_iterations,            # 每步迭代几次（1 次就是纯滞后）
    ),
)
```

### 3.3 代理体的虚拟惯量从哪来

代理体不能随便给个质量，它得"像"真铁球。`_apply_proxy_effective_masses` 做这件事：

```python
def _apply_proxy_body_effective_masses(self):
    for proxy in self._proxy_mappings:
        src = self._entries[proxy.src_name]    # 源求解器（mjc）
        dst = self._entries[proxy.dst_name]    # 目标求解器（vbd）

        # 问源求解器：这些 body 的"等效质量"和"等效惯量张量"是多少？
        # 对铰接体来说，这反映的是"通过关节传到这个 body 上的有效惯量"，
        # 不是孤立 body 的惯量。这一点很关键。
        masses, inertias = self._eval_effective_body_inertial_properties(src, proxy.src_ids)

        # 乘以 mass_scale，写到目标视图的代理体上
        proxy_masses = [proxy.mass_scale * m for m in masses]
        proxy_inertias = [I * proxy.mass_scale for I in inertias]
        self._apply_body_inertia_override(dst, proxy.proxy_ids_local, proxy_masses, proxy_inertias)
```

> 💡 **为什么用"等效质量"而不是"孤立质量"？** 想象铰接摆链：第一个连杆被关节固定在天花板上，你推它，它感觉"很重"，因为整个链的惯量都通过关节传过来。如果代理体只填这个连杆自己的孤立质量，布料推代理体时代理体会飞走，但真连杆纹丝不动--这就失真了。用等效质量，代理体被推的反应和真连杆一致。

### 3.4 一个代理 pass 的完整流程

`_step_proxy` 是单次代理耦合的核心，我逐步拆解（对应 `solver_coupled_proxy.py:1302`）：

```
┌─ 1. STASH：保存上一轮的反馈力（供松弛混合用）
│      coupling_forces → coupling_forces_previous
│
├─ 2. 把反馈力注入源求解器
│      源求解器.body_f += 外力 + 上一轮收割的 coupling_forces
│      （coupling_forces 就是上一次目标"顶"代理体的力，现在加回真铁球）
│
├─ 3. 步进源求解器（MuJoCo 往前走一步）
│      mjc.step(...) → 真铁球有了新位姿、新速度
│
├─ 4. 同步：把真铁球的位姿/速度抄给泡沫球（代理体）
│      sync_proxy_states_kernel: dst.body_q[proxy] = src.body_q[real]
│      记录 proxy_qd_before（泡沫球被算之前的速度）
│
├─ 5. REWIND（仅 lagged 模式）：撤回之前施加的力，避免重复
│      dst.solver.coupling_rewind_proxy_body(...)
│      减去：上一轮的耦合反馈力 + 重力 + 外力
│      （因为目标求解器等会儿会自己重新算这些力）
│
├─ 6. 准备碰撞接触
│      如果有 collision_pipeline：检测代理体和布料的碰撞
│      过滤掉"代理体 vs 代理体"和"代理体 vs 静态"的接触
│      （这些是虚拟对象之间的，不该反馈回源）
│
├─ 7. 步进目标求解器（VBD 往前走一步）
│      vbd.step(...) → 泡沫球被布顶了，速度变了
│
├─ 8. HARVEST 收割：把泡沫球的速度变化换算成力
│      dst.solver.coupling_harvest_proxy_wrenches(...)
│      F = m·Δv / dt,  τ = I·Δω / dt
│      Δv = 泡沫球被算之后的速度 - proxy_qd_before
│      写入 coupling_forces
│
└─ 9. BLEND 混合（可选松弛）
       coupling_forces = ω·新力 + (1-ω)·旧力
       （防止反馈力来回震荡，类似低通滤波）
```

> 💡 **第 5 步 REWIND 最绕，再解释一下**：lagged 模式下，上一轮我们往源求解器加了反馈力 `F_old`，源求解器算出真铁球的新状态时已经把 `F_old` 算进去了。然后我们把真铁球状态同步给泡沫球--这时候泡沫球的速度里**已经包含了 `F_old` 的影响**。如果目标求解器（VBD）再带着这个速度去和布料算碰撞，`F_old` 的影响就会被算两遍。所以要先"撤回" `F_old`、重力、外力，让泡沫球以一个"干净"的状态进入目标求解。

### 3.5 lagged vs staggered：两种同步策略

```python
class _ProxyMode(IntEnum):
    LAGGED = 0      # 同步源"起始"位姿 + "终止"速度
    STAGGERED = 1   # 同步源"终止"位姿 + "终止"速度
```

- **lagged**（默认）：泡沫球拿到的位姿是**这一步开始时**真铁球的位置，速度是**这一步结束时**真铁球的速度。位姿用旧的（滞后），需要 rewind 避免重复。更稳定，是默认选择。
- **staggered**：泡沫球拿到的是**这一步结束时**真铁球的位姿和速度（都是最新的）。直接衔接，不 rewind，清零 `coupling_forces`。更"新"但可能不稳定。

> 💡 **比喻**：lagged 像是"我告诉你一秒前我在哪、现在我的速度是多少，你算一下"；staggered 像是"我告诉你现在我在哪、速度多少，你接着算"。前者信息旧一点但更稳，后者新一点但容易抖。

### 3.6 Aitken 自适应松弛

如果直接用收割的反馈力，两路耦合容易震荡（A 推 B，B 反弹推 A 更狠，A 再推 B 更更狠……）。松弛就是"别用满新的反馈力，掺一点旧的"：

```
固定松弛：f = ω·f_new + (1-ω)·f_old    # ω=0.5 就是一半新一半旧
```

Aitken 松弛更进一步：**根据最近两轮反馈力的变化趋势，自动算一个最优 ω**。如果连续两轮反馈力差很多（震荡剧烈），ω 调小一点（保守）；如果变化平缓，ω 调大（加速收敛）。代码在 `_blend_proxy_feedback`，数学是经典的 Aitken Δ² 方法：

```python
# _update_aitken_relaxation_kernel 的核心：
# ω_new = clamp(-ω_old · (r_old · Δr) / (Δr · Δr), min, max)
# 其中 r 是残差（本轮反馈 - 上轮反馈），Δr = r - r_prev
```

---

## 4. 算法二：`SolverCoupledADMM`（ADMM 耦合）

ADMM 比 Proxy 更"数学"，但收敛性更好，能处理更强的两路约束。

### 4.1 核心思想：让两个求解器在"界面"上谈判

> 💡 **比喻**：两个公司（求解器）在交界处（界面，比如刚体和布的接触面）做生意，各自内部账本（物理状态）自己管，但交界处的"价格"（界面相对速度）得谈拢。Proxy 是回合制传话（我说什么你听什么，滞后），ADMM 是**面对面谈多轮，每轮都修正，直到双方在界面上达成一致**。

具体说，ADMM 把"界面相对速度应该满足约束"这件事写成优化问题，然后**交替方向**地更新：

- 每个子求解器各自往前算一步（各自最优）
- 看看界面上两边对不上多少（残差）
- 用对偶变量 λ 去"推"两边，让它们下一轮更接近
- 重复若干轮，直到界面基本一致

### 4.2 约束从哪来：三个来源

ADMM 不需要你手动指定"谁耦合谁"，它从 Model 元数据里**自动发现**跨求解器的约束：

#### 来源 1：跨求解器的关节

如果一个关节的 parent body 和 child body 分属不同 entry（比如刚体铰接链的一个关节，parent 在 mjc，child 在 vbd），这个关节就变成 ADMM 约束。`_build_admm_joint_groups` 遍历所有关节：

```python
for joint in range(model.joint_count):
    parent = joint_parent[joint]
    child = joint_child[joint]
    owner_pair = self._cross_solver_joint_entries(joint, parent, child)
    if owner_pair is None:    # parent 和 child 在同一个 entry，跳过
        continue
    # parent 和 child 分属不同 entry → 这个关节转成 ADMM 附件约束
    if joint_type == BALL:
        point_items[...].append((child, 位置, parent, 位置, stiffness, damping))
    elif joint_type == REVOLUTE:
        point_items[...].append(...)      # 平移约束
        revolute_angular_items[...].append(...)  # 转角约束
```

> 注意：跨求解器的关节**不能**被任何 entry 拥有（`_cross_solver_joint_entries` 会检查），否则约束会被算两遍（子求解器算一次，ADMM 又算一次）。它留给 ADMM 统一处理。

#### 来源 2：body-particle 附件标注

你可以显式声明"这个刚体点钉在这个粒子上"，用 `add_body_particle_attachment`：

```python
# 在 builder 上注册（来自 mujoco_vbd_admm_solver.py）
SolverCoupledADMM.add_body_particle_attachment(
    builder,
    body=self.ball_body,             # 刚体球
    particle=self.center_particle,   # 布料中心点
    body_point=wp.vec3(0.0, 0.0, ball_radius),  # 钉在球底部
    stiffness=1.0e3,                 # 附件刚度
)
```

这会在 Model 上注册一组自定义属性（`coupling:body_particle_attachment_*`）。构造 ADMM 求解器时，`_build_admm_body_particle_attachment_groups` 找出 body 和 particle 分属不同 entry 的行，转成 ADMM 附件。

#### 来源 3：碰撞接触

通过 `ContactPair` 启用：

```python
SolverCoupledADMM.Config(
    contact_pairs=[
        SolverCoupledADMM.ContactPair(source="mjc", destination="vbd"),
    ],
    ...
)
```

框架会自建一个 `CollisionPipeline`，检测两个 entry 之间的碰撞，按所有权拆成 rigid-rigid / rigid-particle / particle-particle 三类接触，每条接触变成一个带摩擦的 ADMM 约束。

> ⚠️ **重要**：用 ADMM 接触时，**别再让子求解器自己处理接触**。所以 ADMM 示例里经常看到 `model.soft_contact_ke = 0.0` 或注释掉 `model.collide(...)`，避免接触力被算两遍。

### 4.3 ADMM 的数学（别怕，就两个公式）

对一条二次型约束（比如附件），耦合能量是：

```
E_c(u) = (κ/2)·||u - u_target||² + (c/2)·||u||²
```

其中 `u` 是界面相对速度，`u_target` 是目标相对速度（Baumgarte 位置修正用），`κ` 是刚度，`c` 是阻尼。线性化后，每轮 ADMM 更新（来自 `admm_utils.py:700`）：

```
         ρ·W²·Jv + κ·u_target − W·λ
u^{k+1} = ─────────────────────────────
            κ + c + ρ·W²

λ^{k+1} = λ^k + ρ·W·(u^{k+1} − Jv)
```

别被符号吓到，逐个解释：

| 符号 | 含义 | 比喻 |
|---|---|---|
| `u` | 辅助变量（界面相对速度的"协商值"） | 两人谈判桌上的"提议价格" |
| `Jv` | 约束雅可比 × 当前速度 = 实际界面相对速度 | 各自公司"当前报价" |
| `λ` | 对偶变量（拉格朗日乘子），跨轮累积 | 谈判筹码，每轮根据差距调整 |
| `ρ` | ADMM 惩罚参数（`Config.rho`） | 谈判激烈程度，越大越急着拉拢 |
| `W` | 界面权重（由两侧等效质量调和） | 谁分量重，谁的话事权大 |
| `κ` | 约束刚度 | 合同违约金，越大越不能偏离 |

**第一个公式**（u 更新）：综合"当前实际速度 Jv""目标 u_target""对偶压力 λ"，算出一个"协商速度 u"。
**第二个公式**（λ 更新）：如果协商速度 u 和实际速度 Jv 还对不上，就调整筹码 λ，下一轮加大力度。

代码里就是两个 kernel：

```python
@wp.kernel
def u_update_quadratic_kernel(kappa, damping, W, rho, lambda_k, Jv, u_target, u_out):
    i = wp.tid()
    W_i = W[i]
    W2 = W_i * W_i
    denom = kappa[i] + damping[i] + rho * W2
    u_out[i] = (rho * W2 * Jv[i] + kappa[i] * u_target[i] - W_i * lambda_k[i]) / denom

@wp.kernel
def lambda_update_kernel(rho, W, u, Jv, lambda_inout):
    i = wp.tid()
    lambda_inout[i] = lambda_inout[i] + rho * W[i] * (u[i] - Jv[i])   # λ += ρW(u - Jv)
```

### 4.4 力怎么施加到 body 上

算出 `u` 和 `λ` 后，要把约束力 splat（泼洒）到相关 body 上。`attach_rr_accumulate_forces_kernel`：

```python
@wp.kernel
def attach_rr_accumulate_forces_kernel(body_a, point_a, body_b, point_b, ..., body_f_a, body_f_b):
    i = wp.tid()
    ba = body_a[i]    # 约束连着的 body A
    bb = body_b[i]    # 约束连着的 body B
    W_i = W[i]

    # 约束力 = W·(λ + ρ·W·(u - Jv))
    force_a = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))

    # 作用在 body A 上的 spatial force = (力, 力矩)
    # 力矩 = arm × force（arm 是作用点到质心的力臂）
    point_a_world = transform_point(body_q_a[ba], point_a[i])
    arm_a = point_a_world - transform_point(body_q_a[ba], body_com_a[ba])
    wp.atomic_add(body_f_a, ba, wp.spatial_vector(force_a, wp.cross(arm_a, force_a)))

    # body B 受反作用力
    force_b = -force_a
    ...
    wp.atomic_add(body_f_b, bb, wp.spatial_vector(force_b, wp.cross(arm_b, force_b)))
```

> 💡 `wp.atomic_add` 是 GPU 上的原子加--多个约束可能同时往同一个 body 上加力，必须原子操作避免竞争。

### 4.5 一个 ADMM step 的完整流程

`_step_coupled`（`solver_coupled_admm.py:2762`）：

```python
def _step_coupled(self, state_in, state_out, control, contacts, dt):
    iters = int(coupling.iterations)

    # 1. 检测碰撞接触，填充动态接触 group
    self._refresh_collision_contact_groups(state_in)
    # 若有 proximal term (gamma>0)：刷新虚拟质量覆盖
    if coupling.gamma > 0.0:
        self._refresh_admm_proximal_masks()
        self._refresh_admm_proximal_view_overrides(...)

    # 2. 快照步起始状态到 buf（每轮迭代都要回到这个起点）
    for name, entry in self._entries.items():
        buf = self._admm_buffers[name]
        wp.copy(buf.body_q_n, entry.state_0.body_q)    # 记住步起始位姿
        wp.copy(buf.body_qd_n, entry.state_0.body_qd)  # 记住步起始速度
        wp.copy(buf.body_qd_k, entry.state_0.body_qd)  # 迭代用的工作速度

    # 3. 算各约束的 u_target（Baumgarte 位置修正目标）
    self._admm_begin_step(dt)

    # 4. ADMM 迭代
    for k in range(iters):
        # 4a. 每轮把状态回退到步起始（在同一个时间区间上做不动点迭代）
        for name, entry in self._entries.items():
            self._prepare_admm_iteration_state(entry, buf, state_in, dt, iteration_restart=k>0)

        # 4b. 算 Jv，把约束力 splat 到 body_f/particle_f
        self._accumulate_admm_forces(k, dt, refresh_jv=(k==0), initialize_contact_u=(k==0))

        # 4c. 把力设为各 entry 的输入
        for name, entry in self._entries.items():
            self._apply_admm_force_inputs(entry, buf, dt)

        # 4d. 步进所有子求解器（每个都往前算一步）
        for entry in self._entries.values():
            self._step_entry(entry, control, contacts, dt)

        # 4e. 快照新速度到 buf.*_k
        for name, entry in self._entries.items():
            wp.copy(buf.body_qd_k, entry.state_1.body_qd)

        # 4f. 用新速度更新对偶变量 λ（暖启动下一轮）
        self._update_admm_dual(k, dt)
```

> 💡 **第 4a 步是关键**：每轮迭代都回到步起始状态重算，**不是**接着上一轮的结果继续往前。这是 ADMM 的不动点迭代--在同一个 `dt` 区间上反复算，直到界面速度收敛。`λ` 是跨轮累积的（暖启动），所以越迭代越接近正确答案。

### 4.6 proximal term（γ）：改善收敛的"虚拟质量"

ADMM 收敛可能慢，特别是当子求解器很"硬"（刚度大）时。`Config.gamma > 0` 引入**近端项**：在参与约束的 body 上临时加一点虚拟质量（`γ·ρ`），让子求解器"软"一点，更容易被 ADMM 推动到一致。`_refresh_admm_proximal_masks` 标记哪些 body/particle 参与约束，然后通过 `add_body_lumped_inertia` 给它们加虚拟惯量。

> 💡 **比喻**：两个倔脾气的谈判方（硬求解器），谁也不肯让步，谈判卡住。proximal term 像是给双方都灌点镇静剂（加虚拟质量，降低刚度），让它们更容易妥协，谈判（迭代）就顺畅了。代价是结果稍微"软"一点，但通常可接受。

### 4.7 跨求解器关节的"代理体"

问题：ball 关节的 parent 在 mjc、child 在 vbd，但 vbd 视图里要能"看见" parent body 才能算附件约束。`_init_admm_joint_proxy_visibility` 把关节邻居 body 保留为 proxy body：

```python
for joint in range(model.joint_count):
    parent = joint_parent[joint]
    child = joint_child[joint]
    if parent_owner != child_owner:    # 跨求解器关节
        # 在 parent 的 entry 视图里保留 child 作为 proxy body
        body_keep[parent_name].add(child)
        # 在 child 的 entry 视图里保留 parent 作为 proxy body
        body_keep[child_name].add(parent)
```

这样两个视图都能看见关节两端，ADMM 附件约束才能施加。proxy body 的惯量来自源等效质量 × `joint_proximal_mass_scale`。

---

## 5. 接触处理：三种模式别搞混

接触是耦合里最容易混淆的部分，梳理一下：

### 模式 A：父级接触（Proxy 耦合常用）

调用方在 `simulate()` 里自己跑碰撞，把接触传给 `solver.step()`：

```python
def simulate(self):
    self.model.collide(self.state_0, self.contacts)    # 框架外检测
    self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
```

框架用 `_contacts_for_entry` 把父级接触按 shape 可见性**过滤**给每个 entry（只保留该 entry 能看见的 shape 的接触）。

### 模式 B：Proxy 专属碰撞管线

`SolverCoupledProxy.Proxy.collision_pipeline` 让你给某个代理方向定制碰撞检测，而不是用父级接触：

```python
Proxy(
    source="mjc", destination="vbd",
    bodies=...,
    collision_pipeline=lambda view: newton.examples.create_collision_pipeline(view, args),
    collide_interval=1,
)
```

`collide_interval` 控制多久重新检测一次（检测有成本，但代理体每步都在动，间隔太大会穿透）。

### 模式 C：ADMM 内部碰撞

`SolverCoupledADMM` 自建 `CollisionPipeline`，接触由 ADMM 接触约束处理，**不**走子求解器自己的接触。所以 ADMM 示例里：

```python
# 来自 mujoco_vbd_admm_solver.py
# ADMM 从关节和 body-particle 附件构建耦合，所以这里保持 contacts 为空，
# 不让 collide() 加多余的约束。
# self.model.collide(self.state_0, self.contacts)   ← 注释掉了
self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
```

---

## 6. 实战：逐行走一遍 `mujoco_vbd_coupled_solver`

把前面学的串起来，走一遍这个示例的 `__init__` 和 `simulate`。

### 6.1 构造阶段

```python
class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8                                    # 每帧 8 个子步
        self.sim_dt = self.frame_dt / self.sim_substeps          # 每个子步 dt
        self.use_coupled = getattr(args, "solver", "coupled") == "coupled"

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 2.0e4
        builder.add_ground_plane()

        # 记住刚体 id 范围
        rigid_body_start = builder.body_count
        self._emit_rigid_bodies(builder)      # 3 个盒子
        self._emit_articulated_chain(builder) # 3 连杆摆链
        rigid_body_end = builder.body_count

        self._emit_cloth(builder)             # 30×30 布料
        self._emit_soft_bodies(builder)       # 3 个软体
        builder.color()
        self.model = builder.finalize()

        # 接触参数
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_mu = 0.5

        vbd_kwargs = {"iterations": 10, "particle_enable_self_contact": True, ...}

        if self.use_coupled:
            # 刚体 id 列表
            rigid_body_indices = wp.array(list(range(rigid_body_start, rigid_body_end)), dtype=int)
            # vbd 拥有的 body（本例中是空集，因为所有 body 都是刚体）
            vbd_body_indices = wp.array(
                [i for i in range(self.model.body_count) if i < rigid_body_start or i >= rigid_body_end],
                dtype=int,
            )

            self.solver = SolverCoupledProxy(
                model=self.model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mjc",
                        solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=200),
                        bodies=[int(i) for i in rigid_body_indices.numpy()],
                        joints=list(range(self.model.joint_count)),   # 关节全归 mjc
                    ),
                    SolverCoupledProxy.Entry(
                        name="vbd",
                        solver=lambda v: SolverVBD(model=v, **vbd_kwargs),
                        bodies=[int(i) for i in vbd_body_indices.numpy()],
                        particles=list(range(self.model.particle_count)),  # 粒子全归 vbd
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc",               # 刚体当源
                            destination="vbd",           # vbd 里放代理体
                            bodies=[int(i) for i in rigid_body_indices.numpy()],
                            mass_scale=args.mass_scale,
                            mode=args.coupling_mode,
                            collision_pipeline=lambda model: newton.examples.create_collision_pipeline(model, self.args),
                            collide_interval=1,
                        )
                    ],
                    iterations=args.proxy_iterations,
                ),
            )
        else:
            # 不耦合：纯 VBD 基线对比
            self.solver = SolverVBD(model=self.model, **vbd_kwargs)

        # 状态
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.contacts = self.model.contacts()
        self.control = self.model.control()

        newton.examples.configure_coupled_view(self, args)    # 配置渲染视图
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.capture()    # CUDA graph 捕获
```

### 6.2 步进阶段

```python
    def simulate(self):
        # 每帧开始：检测碰撞（父级接触，模式 A）
        self.model.collide(self.state_0, self.contacts)
        # 8 个子步
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()                            # 清力
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)  # 鼠标拖拽等外力
            # ↓ 这里 SolverCoupledProxy.step 会：
            #   1. distribute_state: 父状态 -> mjc.state_0, vbd.state_0
            #   2. _step_coupled (proxy 算法):
            #      - stash 上一轮反馈
            #      - 给 mjc 注入反馈力，步进 mjc
            #      - 同步刚体位姿到 vbd 的代理体
            #      - rewind（lagged 模式）
            #      - 用 collision_pipeline 检测代理体 vs 布料/软体接触
            #      - 步进 vbd（代理体被布顶）
            #      - 收割代理体动量变化 -> coupling_forces
            #      - blend（松弛）
            #   3. reconcile_state: mjc.state_1, vbd.state_1 -> 父状态（只写各自的）
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0   # 乒乓交换

    def step(self):
        # 优先用 CUDA graph，捕不到就裸跑
        if not _launch_frame_graph(self.model, self.graph):
            self.simulate()
        self.sim_time += self.frame_dt
```

---

## 7. 示例对照表

| 示例 | 源 → 目标 | 算法 | 耦合什么 | 亮点 |
|---|---|---|---|---|
| `mujoco_vbd_coupled_solver` | MuJoCo → VBD | Proxy body | 刚体盒+摆链 → 布料+软体 | 可切 lagged/staggered，有 collision_pipeline |
| `mujoco_xpbd_coupled_solver` | MuJoCo → XPBD | Proxy body | 刚体+8连杆链 → 布料 | 动量收割反馈刚体 |
| `xpbd_vbd_coupled_solver` | VBD → XPBD | Proxy particle | VBD 布料 → XPBD 粒子床 | `configure_view` 把 XPBD 设成"只碰不弹"（strip 掉弹性拓扑） |
| `mujoco_mpm_coupled_solver` | MuJoCo → MPM | Proxy body | 刚体 → 沙床 | `collision_pipeline=lambda: None`（MPM 自己处理 collider） |
| `vbd_mpm_coupled_solver` | VBD → MPM | Proxy particle | VBD 软体表面 → MPM 网格 collider | 自定义子类，`setup_collider` 注册变形网格 |
| `xpbd_mpm_coupled_solver` | XPBD → MPM | Proxy particle | XPBD 粒子 → MPM 立方体 | 代理粒子参与 P2G/G2P 动量交换但不贡献应力 |
| `mujoco_vbd_admm_solver` | MuJoCo → VBD | ADMM | ball-布料附件 + 跨求解器球铰 | 同时演示附件标注和关节两种约束来源 |
| `admm_contact_solver` | SemiImplicit → XPBD | ADMM | particle-shape 接触 | ADMM 内部碰撞 + 摩擦接触 |
| `mujoco_franka_vbd_cable_admm_solver` | MuJoCo → VBD | ADMM | 关节 + 接触 | 机器人-线缆复杂场景 |

### 怎么选 Proxy 还是 ADMM？

- **耦合比较"弱"或单向主导**（刚体砸柔体、刚体落沙堆）：Proxy 够用，实现简单，对求解器侵入小。
- **需要强两路约束**（刚体和柔体用关节连在一起、刚体被钉在布上）：ADMM 收敛更好。
- **要精确控制界面摩擦接触**：ADMM 的接触约束更严谨（带库仑摩擦投影）。
- **求解器不支持 `coupling_eval_effective_mass`**：Proxy 可能跑不了（需要等效质量装代理惯量），ADMM 也有类似需求但有 fallback。

---

## 8. CUDA Graph 支持：为什么框架这么"啰嗦"

所有示例都用 `wp.ScopedCapture` 把 `simulate()` 捕获成 CUDA graph，每帧 `wp.capture_launch(graph)` 重放，省去 CPU-GPU 往返。为此框架必须保证：**每次 `simulate()` 执行的 GPU 操作序列完全一样**，不能有运行时分支。

这就是为什么框架里有那么多"看起来多余"的设计：

- **所有计数（attachment 数、proxy 数、接触容量）都是构造期常量**--运行时不增减约束。
- **接触缓冲、ADMM 内部 contacts 在构造时就预分配**--避免捕获时 lazy 分配留下野指针（`solver_coupled_admm.py:1121` 有注释说明）。
- **`prepare_contacts` 预建 entry 局部接触缓冲**。
- **`coupling_supports_inertial_property_refresh`** 区分哪些求解器能在图内刷新惯量，不能的走另一条路。

---

## 9. 一张图总结

```text
                    用户代码
                       │
                       ▼
            solver.step(state_in, state_out, ...)
                       │
        ┌──────────────┴──────────────┐
        │  SolverCoupled (基类)        │
        │  1. _distribute_state        │  父状态 → 各 entry.state_0
        │  2. _step_coupled (子类实现) │  ← 真正的耦合算法
        │  3. _reconcile_state         │  各 entry.state_1 → 父状态
        └──────────────┬──────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
   SolverCoupledProxy        SolverCoupledADMM
   (代理体 + 滞后冲量)        (线性化 ADMM)
           │                       │
           │                       │
   ┌───────┴───────┐       ┌───────┴───────┐
   │ 每轮:          │       │ 每轮:          │
   │ 1.注入反馈力   │       │ 1.回退到步起始 │
   │ 2.步进 source  │       │ 2.算 Jv+splat力│
   │ 3.同步代理体   │       │ 3.步进所有子解 │
   │ 4.rewind       │       │ 4.更新对偶 λ   │
   │ 5.步进 dest    │       └───────────────┘
   │ 6.收割动量     │
   │ 7.松弛混合     │
   └───────────────┘
           │
           ▼
    通过 CouplingInterface 钩子与子求解器通信:
      coupling_eval_effective_mass   (问等效质量)
      coupling_eval_gravity_accel    (问重力)
      coupling_rewind_proxy_*        (撤回力)
      coupling_harvest_proxy_*       (收割动量)
      coupling_notify_input_state_*  (通知状态变了)
```

---

## 10. 关键文件索引

| 文件 | 干什么的 |
|---|---|
| `newton/_src/solvers/coupled/__init__.py` | 公开导出入口 |
| `solver_coupled.py` | 基类：所有权、视图、压缩、状态分发/调和、子步进、接触过滤 |
| `model_view.py` | `ModelView`：COW 覆盖层 + 实体修改 API（disable/scale/mark_proxy） |
| `interface.py` | `CouplingInterface`：钩子契约 + 通用默认实现 |
| `solver_coupled_proxy.py` | 代理体耦合算法 |
| `solver_coupled_admm.py` | ADMM 耦合算法 |
| `proxy_utils.py` | 代理同步/rewind/收割的 GPU kernel |
| `admm_utils.py` | ADMM 的 Jv/u/λ 更新、力累加、proximal、摩擦的 GPU kernel |
| `admm_contact_stream.py` | ADMM 粒子-粒子接触流 |

---

## 附：术语速查

- **ownership（所有权）**：每个 body/particle/joint 归且仅归一个求解器管。
- **proxy body/particle（代理体）**：源求解器实体在目标求解器视图里的"替身"，有虚拟惯量，参与目标求解，动量变化被收割。
- **effective mass（等效质量）**：通过关节连起来的铰接体，从某个端点"推动"时感觉到的质量，反映关节约束的惯量耦合。
- **lagged（滞后）**：用上一步的信息算这一步的反馈，回合制。
- **staggered（交错）**：用当前步最新的位姿衔接，更紧但可能不稳。
- **harvest（收割）**：把目标求解器对代理体的动量变化换算成力，反馈给源。
- **rewind（撤回）**：目标求解前先把之前施加的力撤掉，避免重复计算。
- **ADMM（交替方向乘子法）**：把优化问题拆成子问题交替求解，用对偶变量推动收敛。
- **proximal term（近端项）**：给参与约束的实体加虚拟质量，降低刚度，改善 ADMM 收敛。
- **Baumgarte**：位置误差修正，用目标速度消除穿透/漂移。
- **ModelView（模型视图）**：父 Model 的 COW 覆盖层，给每个求解器呈现定制视图。
- **Entry（条目）**：一个子求解器 + 它拥有的实体 + 步进策略。
