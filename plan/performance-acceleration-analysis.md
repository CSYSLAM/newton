---
name: performance-acceleration-analysis
description: Newton physics engine performance acceleration analysis and priority ranking
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ed88e56-69a2-47d3-8aef-fc4cd7034f50
---

# Newton 性能加速分析 (2026-05-18)

## 优先级排序

1. **碰撞加速 (Broadphase/Narrowphase)** — 收益益⭐⭐⭐⭐⭐，难度中 — 碰撞占 step 时间 30-70%，是最大瓶颈
2. **Kernel Launch 融合** — 收益收益⭐⭐⭐⭐，难度低 — 每步 15-30+ 次 wp.launch，launch overhead 累积 15-30%
3. **消除每帧 GPU 分配** — 收益收益⭐⭐⭐⭐，难度低 — step() 内大量 wp.zeros/wp.clone 每帧触发 GPU alloc
4. **算法加速（求解器收敛）** — 收益⭐⭐⭐，难度高 — XPBD/VBD 迭代次数多但算法已较优
5. **访存优化（SoA/Coalescing）** — 收益收益⭐⭐⭐，难度中 — 已做 SoA 存储，contact 数据仍有优化空间
6. **GPU 核心利用率** — 收益⭐⭐，难度中 — Warp 已做 tile/shared memory，进一步需深入 CUDA

## 碰撞加速具体方向

- Broadphase: 引入 GPU LBVH (Morton code 线性 BVH) 替换 SAP，构建 O(N)、查询 O(log N)
- Narrowphase GJK/MPR: 对 convex mesh 预计算 BVH 加速 support mapping，增加 early-out
- Mesh-Mesh SDF: Brent's method 极重，考虑 GPU SDF voxelization 预计算 + 3D texture 采样

## 关键架构信息

- 碰撞管线: AABB → Broadphase (NxN/SAP/Explicit) → Narrowphase (GJK/MPR/解析/SDF) → Contacts
- Broadphase 三种模式: NxN O(N²), SAP O(N log N), Explicit (预计算对)
- Narrowphase: 解析碰撞(12种原始对), GJK/MPR(凸-凸), BVH midphase(mesh-凸), SDF(mesh-mesh)
- 碰撞与求解器集成: Contacts buffer 被各 solver 消费

---

## P2/P3 实施记录：Kernel Launch 融合 + 消除每帧 GPU 分配 (2026-05-22)

### 总览

对碰撞管线、三个求解器（XPBD/Semi-implicit/VBD）和 Viewer 子系统实施了 P2（kernel launch 融合）和 P3（消除每帧 GPU 分配）优化。共消除约 25+ 处每帧 GPU 分配，融合 1 处 counter zero kernel launch。

### P2: Kernel Launch 融合

#### 融合 narrow-phase counter zeroing 进 compute_shape_aabbs

**文件**: `newton/_src/sim/collide.py`

**改动**: `compute_shape_aabbs` kernel 的 thread 0 原本已融合了 contact counter zeroing + generation bump + broad_phase_pair_count zeroing。现在额外融合了 narrow-phase counter zeroing（`self.narrow_phase._counter_array`），消除了 narrow phase 中单独的 `_counter_array.zero_()` 调用。

```python
# compute_shape_aabbs kernel 中 thread 0:
if shape_id == 0:
    for c in range(num_contact_counters):
        contact_counters[c] = 0
    # generation bump ...
    broad_phase_pair_count[0] = 0
    for c in range(num_narrow_phase_counters):  # 新增
        narrow_phase_counters[c] = 0            # 新增
```

**文件**: `newton/_src/geometry/narrow_phase.py`

**改动**: `launch_custom_write()` 新增 `skip_counter_zero: bool = False` 参数。当 `True` 时跳过 `_counter_array.zero_()`，因为调用方已在 `compute_shape_aabbs` 中完成。

```python
if not skip_counter_zero:
    self._counter_array.zero_()
```

**收益**: 消除 1 次 kernel launch per frame（`_counter_array.zero_()` 是一个 dim=1 的 kernel）。

#### 无法融合的 kernel 对

**`solve_body_contact_positions` + `accumulate_weighted_contact_impulse` (XPBD)**: 两者 dim 相同（`rigid_contact_max`），但存在数据依赖——前者通过 `atomic_add` 写入 `rigid_contact_inv_weight`，后者读取它。在同一个 kernel 内，跨 warp 的 atomic_add 结果不保证对所有线程可见，因此不能安全融合。

### P3: 消除每帧 GPU 分配

#### 核心模式

1. **Sentinel array 模式**: 在 `__init__` 中预分配空数组，`launch()` 中复用，避免每帧 `wp.empty(0, ...)`
2. **Lazy persistent buffer 模式**: `__init__` 中设 `None`，首次使用时分配，后续帧检查大小是否匹配，匹配则 `zero_()` 或 `wp.copy()` 复用
3. **`wp.copy()` 替代 `wp.clone()`**: `wp.copy(dst, src)` 拷贝到已有缓冲区；`wp.clone(src)` 每帧分配新缓冲区
4. **Grow-only cache 模式**: Viewer 中按 name 缓存数组，仅当需求大小超过缓存时重新分配
5. **Per-name cache 字典**: `log_lines`/`log_arrows`/`log_capsules` 等方法中按 name 缓存，避免同 name 重复分配

#### 各子系统详细改动

##### 1. XPBD Solver (`newton/_src/solvers/xpbd/solver_xpbd.py`)

消除 7-8 处每帧分配，是改动最大的子系统。

`__init__` 新增 12 个持久缓冲区属性：
```python
self._rigid_contact_inv_weight = None
self._contact_impulse = None
self._contact_impulse_iter = None
self._body_deltas = None
self._particle_deltas = None
self._body_f_tmp = None
self._spring_constraint_lambdas = None
self._edge_constraint_lambdas = None
self._particle_q_init_buf = None
self._particle_qd_init_buf = None
self._body_q_init_buf = None
self._body_qd_init_buf = None
```

`step()` 中的替换：

| 原代码 | 新代码 | 说明 |
|--------|--------|------|
| `wp.clone(state_in.particle_q)` | `wp.copy(self._particle_q_init_buf, state_in.particle_q)` | 避免每帧分配 |
| `wp.clone(state_in.particle_qd)` | `wp.copy(self._particle_qd_init_buf, state_in.particle_qd)` | 同上 |
| `wp.zeros(model.body_count, ...)` (inv_weight) | lazy persistent + `zero_()` | 首帧分配，后续帧 zero 复用 |
| `wp.zeros(cap, ...)` (contact_impulse ×2) | lazy persistent + `zero_()` | 同上 |
| `wp.empty_like(state_out.particle_qd)` (particle_deltas) | lazy persistent `wp.empty(n, ...)` | 首帧分配，后续帧复用 |
| `wp.empty_like(state_out.body_qd)` (body_deltas) | lazy persistent `wp.empty(n, ...)` | 同上 |
| `wp.clone(state_in.body_f)` | `wp.copy(self._body_f_tmp, state_in.body_f)` | 避免每帧分配 |
| `wp.empty_like(model.spring_rest_length)` | lazy persistent `wp.empty(n, ...)` | 同上 |
| `wp.empty_like(model.edge_rest_angle)` | lazy persistent `wp.empty(n, ...)` | 同上 |

Lazy persistent buffer 的典型模式：
```python
if self._buf is None or self._buf.shape[0] != required_size:
    self._buf = wp.zeros(required_size, dtype=..., device=...)
else:
    self._buf.zero_()
```

##### 2. Semi-implicit Solver (`newton/_src/solvers/semi_implicit/solver_semi_implicit.py`)

消除 1 处每帧分配。

`__init__` 新增：
```python
self._body_f_work = None
```

`step()` 中替换：
```python
# 原: body_f_work = wp.clone(state_in.body_f)
# 新:
if self._body_f_work is None or self._body_f_work.shape[0] != model.body_count:
    self._body_f_work = wp.clone(state_in.body_f)
else:
    wp.copy(self._body_f_work, state_in.body_f)
body_f_work = self._body_f_work
```

##### 3. VBD Solver (`newton/_src/solvers/vbd/solver_vbd.py`)

消除 1 处每帧分配，与 semi-implicit 相同的模式。

##### 4. Viewer (`newton/_src/viewer/viewer.py`)

消除 4 处每帧分配（粒子 compaction 相关）。

`__init__` 新增：
```python
self._particle_mask = None
self._particle_offsets = None
self._particle_points_out = None
self._particle_radii_out = None
```

粒子 compaction 中的替换：

| 原代码 | 新代码 |
|--------|--------|
| `wp.zeros(n, dtype=wp.int32, ...)` (mask) | lazy persistent + `zero_()` |
| `wp.empty(n, dtype=wp.int32, ...)` (offsets) | lazy persistent |
| `wp.empty(active_count, dtype=wp.vec3, ...)` (points_out) | `self._particle_points_out[:active_count]` (slice 预分配缓冲区) |
| `wp.empty(active_count, dtype=wp.float32, ...)` (radii_out) | `self._particle_radii_out[:active_count]` (同上) |

关键：`points_out` 和 `radii_out` 的大小是 `active_count`（每帧可能不同），使用预分配最大尺寸缓冲区 + slice 避免每帧分配。

##### 5. Viewer GL (`newton/_src/viewer/viewer_gl.py`)

消除约 8 处每帧分配（颜色展开 + 胶囊 cap 数组）。

`__init__` 新增 per-name 缓存字典：
```python
self._color_expand_cache: dict[str, wp.array] = {}
self._cap_xforms_cache: dict[str, wp.array] = {}
self._cap_scales_cache: dict[str, wp.array] = {}
self._cap_colors_cache: dict[str, wp.array] = {}
self._cap_materials_cache: dict[str, wp.array] = {}
```

**log_lines 颜色展开**：
```python
# 原: colors = wp.zeros(num_lines, dtype=wp.vec3, device=self.device)
# 新: grow-only cache
cache_key = f"lines_{name}"
cached = self._color_expand_cache.get(cache_key)
if cached is None or cached.shape[0] < num_lines:
    cached = wp.zeros(num_lines, dtype=wp.vec3, device=self.device)
    self._color_expand_cache[cache_key] = cached
colors = cached[:num_lines]
colors.fill_(color_vec)
```

**log_arrows 颜色展开**：同 log_lines 模式，cache_key = `f"arrows_{name}"`。

**log_capsules cap 数组**：per-name grow-only cache，用于 `cap_xforms`/`cap_scales`/`cap_colors`/`cap_materials`。

##### 6. Collision Pipeline (`newton/_src/sim/collide.py`)

融合 narrow-phase counter zeroing（见 P2 部分）。无额外 per-frame 分配消除（collide.py 的分配已在 `__init__` 中完成）。

##### 7. Narrow Phase (`newton/_src/geometry/narrow_phase.py`)

消除 2 处每帧分配 + 新增 `skip_counter_zero` 参数。

`__init__` 新增 sentinel 数组：
```python
self._empty_int_array = wp.zeros(1, dtype=wp.int32, device=device)
self._empty_sdf_index = wp.full(max(num_shapes, 1), -1, dtype=wp.int32, device=device)
self._empty_texture_sdf = wp.zeros(0, dtype=TextureSDFData, device=device)
```

替换：
| 原代码 | 新代码 |
|--------|--------|
| `wp.full(shape_types.shape[0], -1, ...)` (shape_sdf_index) | `self._empty_sdf_index` |
| `wp.zeros(0, dtype=TextureSDFData, ...)` (texture_sdf_data) | `self._empty_texture_sdf` |

##### 8. SAP Broadphase (`newton/_src/geometry/broad_phase_sap.py`)

消除 2 处每帧分配。

`__init__` 新增 sentinel 数组：
```python
self._empty_gap = wp.empty(0, dtype=wp.float32, device=device)
self._empty_filter = wp.empty(0, dtype=wp.vec2i, device=device)
```

替换：
| 原代码 | 新代码 |
|--------|--------|
| `wp.empty(0, dtype=wp.float32, ...)` (shape_gap) | `self._empty_gap` |
| `wp.empty(0, dtype=wp.vec2i, ...)` (filter_pairs) | `self._empty_filter` |

##### 9. BVH Broadphase (`newton/_src/geometry/broad_phase_bvh.py`)

消除 1 处每帧分配。

`__init__` 新增：
```python
self._empty_filter = wp.empty(0, dtype=wp.vec2i, device=device)
```

替换 `wp.empty(0, dtype=wp.vec2i, ...)` → `self._empty_filter`。

### 未优化的子系统及原因

| 子系统 | 每帧分配数 | 未优化原因 |
|--------|-----------|-----------|
| MuJoCo solver CPU 路径 | 5+ | 仅 `use_mujoco_cpu=True` 触发；`wp.array([numpy_data])` 本质是 CPU→GPU 传输，不可避免 |
| MPM solver | 5 | BSR matrix 结构每帧可能变化；`wp.clone` 在 `state_in is state_out` 时做 double-buffering，改动风险高 |
| `solve_rheology` | 1 (`wp.clone`) | 嵌入 MPM 内部，需理解完整数据流 |
| `viewer_file.py` | 1 (`wp.clone`) | `playback()` 引用克隆数组，需 double-buffering 更复杂 |

### 测试验证

所有核心测试通过（2026-05-22）：
- Collision pipeline: 106 tests OK
- GJK: 4 tests OK
- Broadphase: 18 tests OK
- Viewer (log_shapes, particle_flags, geometry_batching): 9 tests OK
- XPBD solver: 24 tests OK
- VBD solver: 4 tests OK

### 汇总

| 子系统 | 消除的每帧分配数 | 优化方法 |
|--------|----------------|---------|
| XPBD solver | 7-8 | lazy persistent + `wp.copy()` 替代 `wp.clone()` |
| Semi-implicit solver | 1 | 持久 `_body_f_work` + `wp.copy()` |
| VBD solver | 1 | 持久 `_body_f_work` + `wp.copy()` |
| Viewer (viewer.py) | 4 | 持久粒子压缩缓冲区 + slice |
| Viewer (viewer_gl.py) | ~8 | per-name grow-only cache |
| Collide (collide.py) | 1 kernel launch | counter zero 融合进 `compute_shape_aabbs` |
| Narrow phase | 2 | sentinel 数组 + `skip_counter_zero` |
| SAP broadphase | 2 | sentinel 空数组 |
| BVH broadphase | 1 | sentinel 空数组 |
| **合计** | **~27** | |
