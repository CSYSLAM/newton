# Phase 1: LBVH Broadphase 完整实现

## 目标

新增 `BroadPhaseBvh` 类，基于 Warp 内置 `wp.Bvh`（Morton code LBVH）实现 broadphase，为大型场景提供比 SAP 更好的 GPU 利用率。

## 新建文件

### `newton/_src/geometry/broad_phase_bvh.py`

#### 类：BroadPhaseBvh

```python
class BroadPhaseBvh:
    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        max_pairs_per_shape: int = 64,
        device: Devicelike | None = None,
    ) -> None:
```

**初始化流程：**
1. 调用 `precompute_world_map()` 获取 `world_index_map`, `world_slice_ends`
2. 分配 BVH lower/upper bounds 持久数组 (`_bvh_lower`, `_bvh_upper`)
3. 首帧时构建 `wp.Bvh`，后续帧用 `refit()`

**核心方法：**

```python
def launch(
    self,
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],
    collision_group: wp.array[int],
    shape_world: wp.array[int],
    num_shapes: int,
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],
    max_candidate_pair: int,
    filter_pairs: wp.array[wp.vec2i],
    num_filter_pairs: int,
) -> None:
```

**launch() 签名与 NxN/SAP 完全一致**，确保 duck-type 兼容。

#### BVH 更新策略

- **首帧**：`self._bvh = wp.Bvh(self._bvh_lower, self._bvh_upper)` — O(N log N) 构建
- **后续帧**：`self._bvh.refit()` — O(N)，仅更新 bounds，保留树拓扑
- **周期性 rebuild**：可选，当 refit 质量下降时触发（当前未实现）

#### 核心 Kernel

```python
@wp.kernel(enable_backward=False)
def _bvh_broadphase_kernel(
    bvh_id: wp.uint64,
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],
    collision_group: wp.array[int],
    shape_world: wp.array[int],
    world_index_map: wp.array[int],
    world_slice_ends: wp.array[int],
    num_regular_worlds: int,
    filter_pairs: wp.array[wp.vec2i],
    num_filter_pairs: int,
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],
    max_candidate_pair: int,
):
```

**Kernel 逻辑：**
1. 每个 shape 一个线程 (`shape_id = wp.tid()`)
2. 用 `wp.bvh_query_aabb(bvh_id, lower, upper)` 查询重叠
3. `wp.bvh_query_next(query, hit_index)` 遍历命中
4. 强制 canonical ordering: `if hit_index <= shape_id: continue`（保证 i < j）
5. 复用 `broad_phase_common.py` 的过滤逻辑：
   - `test_world_and_group_pair()` — 世界和碰撞组过滤
   - `is_pair_excluded()` — filter_pairs 排除
6. `write_pair()` — 原子写入候选对

#### 辅助函数

复用 `broad_phase_common.py` 中的：
- `precompute_world_map()` — 预计算世界索引映射
- `test_world_and_group_pair()` — 世界/碰撞组过滤
- `is_pair_excluded()` — 排除对过滤
- `write_pair()` — 原子写入候选对

## 修改文件

### `newton/_src/geometry/__init__.py`

```python
from .broad_phase_bvh import BroadPhaseBvh
```

### `newton/geometry.py`

```python
from newton._src.geometry.broad_phase_bvh import BroadPhaseBvh
```

### `newton/_src/sim/collide.py`

1. `BROAD_PHASE_MODES` 添加 `"bvh"`
2. `__init__` 添加 `"bvh"` 分支：
   ```python
   elif broad_phase == "bvh":
       self._broad_phase = BroadPhaseBvh(shape_world, shape_flags, ...)
   ```
3. `collide()` 添加 `isinstance(BroadPhaseBvh)` 分支 — BVH refit + query
4. `_infer_broad_phase_mode_from_instance()` 添加识别逻辑

## 测试

在 `newton/tests/test_broad_phase.py` 中添加：
- `test_bvh_broadphase()` — 单世界，与 NxN 结果交叉验证
- `test_bvh_broadphase_multiple_worlds()` — 多世界过滤
- `test_bvh_broadphase_with_shape_flags()` — Shape flags 过滤
- `test_bvh_broadphase_consistency_with_nxn()` — NxN vs BVH 输出一致性
- `test_bvh_broadphase_refit()` — 多帧 refit 正确性

在 `newton/tests/test_collision_pipeline.py` 中添加：
- `test_collision_pipeline_bvh()` — 端到端集成测试