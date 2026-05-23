# 数据结构详解

## 1. CSR 邻接格式 (Compressed Sparse Row)

### 存储

| 数组 | 类型 | 说明 |
|------|------|------|
| `vertex_adj_offsets` | `wp.array[int]` | 长度 = total_convex_mesh_vertices + num_convex_meshes + 1 |
| `vertex_adj_vertices` | `wp.array[int]` | 长度 = 2 × total_edges（每条边双向存储） |

### 格式说明

所有 convex mesh 的邻接数据拼接在同一个 CSR 数组中，通过 `shape_adj_offset` 索引到对应 mesh 的起始位置。

```
vertex_adj_offsets:  [0, 3, 7, ...]     # 全局偏移
vertex_adj_vertices: [1, 2, 5, 0, 3, ...] # 全局邻居索引

mesh_0 (shape_adj_offset=0, shape_vertex_count=V0):
  vertex 0 的邻居: vertex_adj_vertices[0:3]   → [1, 2, 5]
  vertex 1 的邻居: vertex_adj_vertices[3:7]   → [0, 3, 4, 6]
  ...

mesh_1 (shape_adj_offset=V0+1, shape_vertex_count=V1):
  vertex 0 的邻居: vertex_adj_offsets[V0+1] ~ vertex_adj_offsets[V0+2]
  ...
```

### 查找模式

```python
# 在 kernel 中，已知 shape_id 和局部顶点索引 local_v
offset = shape_adj_offset[shape_id]  # 该 mesh 在 CSR 中的起始偏移
adj_start = vertex_adj_offsets[offset + local_v]
adj_end = vertex_adj_offsets[offset + local_v + 1]
for k in range(adj_start, adj_end):
    neighbor_global = vertex_adj_vertices[k]
    neighbor_local = neighbor_global - offset  # 转回局部索引
```

### 构建 (builder.py)

```python
# 在 Builder.finalize() 中
# 1. 遍历所有 CONVEX_MESH shape
# 2. 对每个 mesh，从 wp.Mesh 的 edges 提取顶点邻接关系
# 3. 构建 CSR 格式，拼接所有 mesh 的邻接数据
# 4. 设置 shape_adj_offset[i] = 该 mesh 在 CSR 中的起始偏移
# 5. 设置 shape_vertex_count[i] = 该 mesh 的顶点数
```

## 2. SupportMapDataProvider struct

```python
@wp.struct
class SupportMapDataProvider:
    shape_adj_offset: int    # 该 shape 在 CSR 中的起始偏移，-1 表示无邻接数据
    shape_vertex_count: int  # 该 shape 的顶点数，0 表示非 convex mesh
    prev_best_vertex: int    # 上次 hill-climb 的最佳顶点（warm-start），-1 表示 cold start
```

**重要**：Warp struct 的 int 字段默认值为 0，不是 -1。必须显式设置 `shape_adj_offset = -1` 以禁用 hill-climbing。

**为什么用 struct 而不是直接传数组？**
- Warp kernel 参数数量有限
- struct 封装了 per-shape 的标量信息（从数组中按 shape_id 索引得到）
- 全局数组 `vertex_adj_offsets/vertex_adj_vertices` 仍需作为 kernel 参数传递

## 3. Model 新字段

```python
class Model:
    # CSR 邻接数据
    vertex_adj_offsets: wp.array[int] | None    # 全局 CSR offset 数组
    vertex_adj_vertices: wp.array[int] | None   # 全局 CSR neighbor 数组

    # Per-shape 索引
    shape_adj_offset: wp.array[int] | None      # shape_id → CSR 起始偏移
    shape_vertex_count: wp.array[int] | None    # shape_id → 顶点数

    # 统计
    total_convex_mesh_vertices: int              # 所有 convex mesh 总顶点数
```

## 4. GenericShapeData (未修改)

```python
@wp.struct
class GenericShapeData:
    shape_type: int      # GeoType 枚举
    scale: wp.vec3       # 尺寸参数
    auxiliary: wp.vec3   # 辅助参数
```

Hill-climbing 不修改此 struct，邻接信息通过 `SupportMapDataProvider` 传递。

## 5. _empty_int_array 模式

当邻接数据为 None（场景中无 convex mesh）时，使用空数组占位：

```python
_empty_int_array = wp.zeros(1, dtype=wp.int32)
```

在 `wp.launch` 中：
```python
vertex_adj_offsets=vertex_adj_offsets if vertex_adj_offsets is not None else _empty_int_array,
```

**为什么需要？** Warp kernel 参数不能为 None，必须传一个数组。空数组保证 kernel 不会越界访问（因为 `shape_adj_offset = -1` 会跳过 hill-climb 路径）。