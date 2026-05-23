# Warp 开发踩坑记录

## 1. Warp struct 默认值陷阱

**问题**：Warp `@wp.struct` 的 int 字段默认值为 0，不是 -1。

```python
@wp.struct
class SupportMapDataProvider:
    shape_adj_offset: int    # 默认值 = 0，不是 -1！
    shape_vertex_count: int  # 默认值 = 0
    prev_best_vertex: int    # 默认值 = 0，不是 -1！
```

**后果**：如果 `shape_adj_offset = 0`，hill-climb 会错误地认为邻接数据可用，尝试访问 CSR 数组索引 0，导致错误结果或越界。

**修复**：所有 `SupportMapDataProvider()` 实例必须显式设置：
```python
data_provider = SupportMapDataProvider()
data_provider.shape_adj_offset = -1   # 禁用 hill-climb
data_provider.shape_vertex_count = 0
data_provider.prev_best_vertex = -1   # Cold start
```

## 2. Warp 不允许在动态循环中修改常量

**问题**：以下代码在 Warp 中报错 "mutating a constant inside a dynamic loop"：
```python
improved = False
for _step in range(16):
    improved = True  # Error!
    if not improved:  # Error!
        break
```

**原因**：Warp 将 `False/True` 视为编译期常量，不允许在动态循环中修改。

**修复**：使用 `int(0)/int(1)` 代替 `False/True`：
```python
improved = int(0)
for _step in range(16):
    improved = int(1)
    if improved == 0:
        break
```

## 3. Python 常量不能在 `@wp.func` 中使用

**问题**：`MAX_HILL_CLIMB_STEPS = 16` 定义在模块级别，但在 `@wp.func` 中 `range(MAX_HILL_CLIMB_STEPS)` 报错 undefined。

**原因**：Warp `@wp.func` 编译为 CUDA 代码，无法访问 Python 模块级常量。

**修复**：使用字面量 `range(16)`。

## 4. Warp 不允许在 kernel 中创建数组

**问题**：`wp.array[int]()` 在 kernel 中无效。

**原因**：Warp kernel 运行在 GPU 上，不能动态分配内存。

**修复**：所有数组必须作为 kernel 参数传入。对于可选数组，使用 `_empty_int_array = wp.zeros(1, dtype=wp.int32)` 占位。

## 5. wp.launch 参数顺序匹配规则

**问题**：`wp.launch` 的 `inputs` 和 `outputs` 按位置匹配 kernel 参数。output 参数之后的所有参数也被视为 output。

**踩坑案例**：`narrow_phase_find_mesh_triangle_overlaps_kernel` 中，`vertex_adj_offsets` 和 `vertex_adj_vertices` 被放在 `triangle_pairs` 之后，导致 Warp 将它们视为 output 参数。`_empty_int_array`（int32）被匹配到 `triangle_pairs`（vec3i），报 dtype mismatch 错误。

**修复**：将邻接数组参数移到 output 参数之前。

## 6. compute_gjk_mpr_contacts 参数顺序

**问题**：在 `mesh_triangle_contacts_to_reducer_kernel` 中调用 `compute_gjk_mpr_contacts` 时，`sort_sub_key`（`(tri_idx << 1) | 1`）被放在邻接数组之前，但函数签名中 `sort_sub_key` 是最后一个参数（有默认值）。

**后果**：Warp 按位置匹配参数类型，`int32` 被匹配到 `shape_adj_offset: wp.array[int]`，报 overload 找不到错误。

**修复**：将 `sort_sub_key` 移到最后，邻接数组放在 `writer_data` 之后：
```python
compute_gjk_mpr_contacts(
    shape_data_a, shape_data_b, quat_a, quat_b,
    pos_a, pos_b, gap_sum, shape_a, shape_b,
    margin_offset_a, margin_offset_b, reducer_data,
    shape_adj_offset, shape_vertex_count,       # ← 邻接数组
    vertex_adj_offsets, vertex_adj_vertices,     # ← 邻接数组
    (tri_idx << 1) | 1,                          # ← sort_sub_key 在最后
)
```

## 7. Warp struct 不能包含数组字段

**问题**：最初尝试在 `SupportMapDataProvider` 中包含 `vertex_adj_offsets: wp.array[int]` 字段，导致 CUDA 编译错误。

**原因**：Warp struct 只能包含标量类型和固定大小的向量/矩阵，不能包含动态大小的数组。

**修复**：将全局数组作为 kernel 参数传递，struct 只包含标量索引字段（`shape_adj_offset`, `shape_vertex_count`, `prev_best_vertex`）。

## 8. GJK simplex_solver 重写导致类型冲突

**问题**：之前的会话完全重写了 `simplex_solver.py`，使用 `simplex_a/b/c/d` 单独变量代替原始的 `Mat83f` 矩阵存储。导致 Warp 类型冲突：`v0` 先被赋值为 `Vert` struct，后被赋值为 `wp.vec3`。

**修复**：通过 `git checkout HEAD -- simplex_solver.py` 恢复原始实现，然后在原始代码上正确添加邻接数组参数。

**教训**：不要重写已有实现，而是在原有代码上增量修改。