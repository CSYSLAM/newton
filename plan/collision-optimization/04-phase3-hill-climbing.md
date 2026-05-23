# Phase 3: Hill-climbing Support Mapping 完整实现

## 目标

将 convex mesh 的 support mapping 从 O(N_vertices) 降至 O(k)（k 为 hill-climb 步数，最多 16 步）。

## 方案选择

| 方案 | 复杂度 | 优势 | 劣势 |
|------|--------|------|------|
| Hill-climbing + warm-start | O(k) amortized | 简单，无需额外数据结构 | 依赖 mesh 邻接信息 |
| Per-mesh vertex BVH | O(log N) | 稳定，无局部最优问题 | 需为每个 convex mesh 维护 BVH |

**选择：Hill-climbing，BVH 作为 fallback。**

## 核心算法

### `_support_map_convex_hill_climb` (support_function.py:109-183)

```python
@wp.func
def _support_map_convex_hill_climb(
    mesh_ptr: wp.uint64,
    mesh_scale: wp.vec3,
    direction: wp.vec3,
    prev_best_vertex: int,
    vertex_adj_offsets: wp.array[int],
    vertex_adj_vertices: wp.array[int],
    shape_adj_offset: int,
    shape_vertex_count: int,
) -> tuple[wp.vec3, int]:
```

**算法流程：**

1. **预缩放方向**：`scaled_dir = cw_mul(direction, mesh_scale)` — 利用 `dot(scale*v, d) == dot(v, scale*d)` 避免每次缩放顶点

2. **Warm start 分支**（`prev_best_vertex >= 0 and shape_adj_offset >= 0`）：
   - 从 `prev_best_vertex` 开始
   - 最多 16 步 hill-climb
   - 每步：遍历当前顶点的所有邻接顶点（CSR 查找）
   - 如果某个邻居的 dot product 更大，移动到该邻居
   - 如果没有改善（`improved == 0`），停止

3. **Cold start 分支**（`prev_best_vertex < 0`）：
   - 暴力 O(N) 扫描所有顶点

4. **CSR 查找模式**：
   ```python
   adj_start = vertex_adj_offsets[shape_adj_offset + current]
   adj_end = vertex_adj_offsets[shape_adj_offset + current + 1]
   for k in range(adj_start, adj_end):
       neighbor = vertex_adj_vertices[k] - shape_adj_offset
   ```
   - `shape_adj_offset + current`：将局部顶点索引映射到全局 CSR 数组
   - `vertex_adj_vertices[k] - shape_adj_offset`：将全局邻居索引转回局部索引

### `support_map` (support_function.py:186-400)

```python
@wp.func
def support_map(
    geom: GenericShapeData,
    direction: wp.vec3,
    data_provider: SupportMapDataProvider,
    vertex_adj_offsets: wp.array[int],
    vertex_adj_vertices: wp.array[int],
) -> wp.vec3:
```

**CONVEX_MESH 分支逻辑：**
```python
if data_provider.shape_vertex_count > 0 and data_provider.shape_adj_offset >= 0:
    # Hill-climbing 路径
    result, _best_idx = _support_map_convex_hill_climb(...)
else:
    # Fallback: brute-force O(N) scan
    ...
```

### `support_map_lean` (support_function.py:405+)

与 `support_map` 相同的 hill-climb 逻辑，但只支持 CONVEX_MESH、BOX、SPHERE 三种类型（用于性能敏感路径）。

## 邻接数组贯穿调用链

所有 GJK/MPR 路径上的函数都需要接收并传递 `vertex_adj_offsets` 和 `vertex_adj_vertices`。详见 [06-call-chain.md](06-call-chain.md)。

## SupportMapDataProvider 构造模式

在 kernel 中，当 shape_id 已知时构造：

```python
data_provider_a = SupportMapDataProvider()
data_provider_a.shape_adj_offset = shape_adj_offset[shape_a]
data_provider_a.shape_vertex_count = shape_vertex_count[shape_a]
data_provider_a.prev_best_vertex = -1  # Cold start
```

**对于非 convex mesh 形状**，使用 sentinel 值禁用 hill-climbing：
```python
data_provider = SupportMapDataProvider()
data_provider.shape_adj_offset = -1
data_provider.shape_vertex_count = 0
data_provider.prev_best_vertex = -1
```

## 测试

1. 单元测试：icosphere 各方向 support 结果与暴力法一致
2. 收敛测试：所有 GJK/MPR 测试用例通过（106/106 collision pipeline tests）
3. 性能对比：`convex_mesh_benchmark` demo（162 顶点 icosphere，309 FPS）