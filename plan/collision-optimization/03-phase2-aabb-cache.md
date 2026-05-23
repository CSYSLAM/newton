# Phase 2: AABB 缓存 (mesh_vs_convex midphase)

## 目标

消除 narrowphase 中 `mesh_vs_convex_midphase()` 对 convex mesh 的 O(N_vertices) AABB 计算，改用预计算的 local AABB 变换。

## 现状

- `collide.py` 的 `compute_shape_aabbs` 已为 CONVEX_MESH 预计算 local AABB（存在 `shape_collision_aabb_lower/upper`）
- 但 `collision_core.py` 的 `mesh_vs_convex_midphase()` 仍调用 `compute_tight_aabb_from_support()` 遍历所有顶点计算 mesh local space 下的 AABB

## 实现方案

### `newton/_src/geometry/collision_core.py`

修改 `mesh_vs_convex_midphase()` 接受 `shape_collision_aabb_lower/upper` 参数。

**快速路径**：对 CONVEX_MESH 类型，用预计算 local AABB + 变换矩阵计算 mesh local space AABB，替代 `compute_tight_aabb_from_support()`。

**Fallback**：对非 CONVEX_MESH 类型（如普通 MESH、HFIELD），保留 tight AABB 路径，调用 `compute_tight_aabb_from_support()`。

### 关键代码逻辑

```python
# mesh_vs_convex_midphase 中
if shape_type[non_mesh_shape] == GeoType.CONVEX_MESH:
    # 快速路径：使用预计算 AABB
    aabb_lower = shape_collision_aabb_lower[non_mesh_shape]
    aabb_upper = shape_collision_aabb_upper[non_mesh_shape]
    # 变换到 mesh local space
    ...
else:
    # Fallback：tight AABB
    data_provider = SupportMapDataProvider()
    data_provider.shape_adj_offset = -1
    data_provider.shape_vertex_count = 0
    data_provider.prev_best_vertex = -1
    aabb_lower, aabb_upper = compute_tight_aabb_from_support(
        shape_data, orientation, center_pos, data_provider,
        vertex_adj_offsets, vertex_adj_vertices,
    )
```

## 测试

1. 正确性：现有 mesh-vs-convex 测试结果不变
2. 包含性：快速路径 AABB 是 tight AABB 的超集（不会漏检）