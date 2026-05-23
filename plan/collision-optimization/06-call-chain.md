# 邻接数组贯穿调用链的完整路径

## 调用链总览

邻接数组 (`vertex_adj_offsets`, `vertex_adj_vertices`) 和 per-shape 数组 (`shape_adj_offset`, `shape_vertex_count`) 需要从 kernel 参数一直传递到 `support_map()` 函数。以下是完整的调用链：

```
wp.launch (kernel 入口)
  │
  ├─ [路径 A: GJK/MPR 直接碰撞]
  │  narrow_phase_kernel_gjk_mpr
  │    → create_find_contacts() → find_contacts()
  │      → create_compute_gjk_mpr_contacts() → compute_gjk_mpr_contacts()
  │        → SupportMapDataProvider(shape_adj_offset[shape_a], shape_vertex_count[shape_a])
  │        → solve_convex_single_contact / solve_convex_multi_contact
  │          → solve_mpr / solve_closest_distance (GJK)
  │            → solve_mpr_core / solve_closest_distance_core
  │              → geometric_center / minkowski_support
  │                → support_map_b → support_func (= support_map)
  │                  → _support_map_convex_hill_climb
  │
  ├─ [路径 B: Mesh-Convex 碰撞 (midphase)]
  │  narrow_phase_find_mesh_triangle_overlaps_kernel
  │    → mesh_vs_convex_midphase
  │      → compute_tight_aabb_from_support (for non-convex shapes)
  │        → support_map → _support_map_convex_hill_climb
  │
  ├─ [路径 C: Mesh triangle contacts 处理]
  │  narrow_phase_process_mesh_triangle_contacts_kernel
  │    → compute_gjk_mpr_contacts (同路径 A)
  │
  └─ [路径 D: Global contact reduction]
     mesh_triangle_contacts_to_reducer_kernel
       → compute_gjk_mpr_contacts (同路径 A)
```

## 函数签名变更详情

### Level 1: Kernel 入口

#### `narrow_phase_kernel_gjk_mpr` (narrow_phase.py:628)

**新增参数**（在 output 参数之前）：
```python
shape_adj_offset: wp.array[int],
shape_vertex_count: wp.array[int],
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `narrow_phase_find_mesh_triangle_overlaps_kernel` (narrow_phase.py:796)

**新增参数**（在 output 参数 `triangle_pairs` 之前）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `narrow_phase_process_mesh_triangle_contacts_kernel` (narrow_phase.py:915)

**新增参数**：
```python
shape_adj_offset: wp.array[int],
shape_vertex_count: wp.array[int],
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `mesh_triangle_contacts_to_reducer_kernel` (contact_reduction_global.py:1479)

**新增参数**：
```python
shape_adj_offset: wp.array[int],
shape_vertex_count: wp.array[int],
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `_update_geom_poses_and_compute_aabbs` (kamino unified.py:325)

**新增参数**：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 2: 碰撞核心函数

#### `compute_gjk_mpr_contacts` (collision_core.py:297, factory 内)

**新增参数**（在 `writer_data` 之后，`sort_sub_key` 之前）：
```python
shape_adj_offset: wp.array[int],
shape_vertex_count: wp.array[int],
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
sort_sub_key: int = 0,  # 移到最后，有默认值
```

**内部构造 SupportMapDataProvider：**
```python
data_provider_a = SupportMapDataProvider()
data_provider_a.shape_adj_offset = shape_adj_offset[shape_a]
data_provider_a.shape_vertex_count = shape_vertex_count[shape_a]
data_provider_a.prev_best_vertex = -1
```

#### `find_contacts` (collision_core.py, factory 内)

**新增参数**（在 `writer_data` 之后）：
```python
shape_adj_offset: wp.array[int],
shape_vertex_count: wp.array[int],
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `mesh_vs_convex_midphase` (collision_core.py:932)

**新增参数**（在 `triangle_pairs_count` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `compute_tight_aabb_from_support` (collision_core.py:418)

**新增参数**（在 `data_provider` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 3: 求解器函数

#### `solve_convex_multi_contact` (collision_convex.py:35)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `solve_convex_single_contact` (collision_convex.py:150)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 4: MPR/GJK 核心

#### `solve_mpr_core` (mpr.py:255)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `solve_mpr` (mpr.py:476)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `solve_closest_distance_core` (simplex_solver.py:312)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `solve_closest_distance` (simplex_solver.py:497)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 5: MPR 辅助函数

#### `minkowski_support` (mpr.py:113)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `geometric_center` (mpr.py:164)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `support_map_b` (mpr.py, factory 内)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 6: Manifold 构建

#### `build_manifold` (multicontact.py:782)

**新增参数**（在 `data_provider_b` 之后）：
```python
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

### Level 7: Support Mapping

#### `support_map` (support_function.py:186)

**新增参数**（在 `direction` 之后）：
```python
data_provider: SupportMapDataProvider,
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

#### `support_map_lean` (support_function.py:405)

**新增参数**（同上）：
```python
data_provider: SupportMapDataProvider,
vertex_adj_offsets: wp.array[int],
vertex_adj_vertices: wp.array[int],
```

## 参数传递规则

1. **全局数组** (`vertex_adj_offsets`, `vertex_adj_vertices`) 作为 kernel 参数传入，逐层传递
2. **Per-shape 数组** (`shape_adj_offset`, `shape_vertex_count`) 作为 kernel 参数传入，在 `compute_gjk_mpr_contacts` 中按 `shape_id` 索引提取标量值，封装到 `SupportMapDataProvider`
3. **SupportMapDataProvider** 包含标量值，在 kernel 内构造，逐层传递
4. **两个 data provider**：`data_provider_a`（shape A）和 `data_provider_b`（shape B）分别构造
5. **_empty_int_array**：当邻接数据为 None 时，使用 `wp.zeros(1, dtype=wp.int32)` 占位

## Kernel 参数顺序规则

**Warp 的 `wp.launch` 规则**：`inputs` 列表按顺序匹配 kernel 的 input 参数，`outputs` 列表按顺序匹配 output 参数。**output 参数之后的所有参数也被视为 output。**

因此，新增的邻接数组参数必须放在 **output 参数之前**，否则 Warp 会错误匹配类型。

例如 `narrow_phase_find_mesh_triangle_overlaps_kernel` 的参数顺序：
```python
# inputs (按顺序):
..., total_num_threads: int,
vertex_adj_offsets: wp.array[int],    # ← 新增，在 output 之前
vertex_adj_vertices: wp.array[int],   # ← 新增，在 output 之前
# outputs:
triangle_pairs: wp.array[wp.vec3i],
triangle_pairs_count: wp.array[int],
```