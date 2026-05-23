# Demo 与测试清单

## Demo

### Phase 1: BVH Broadphase

| Demo | 命令 | 说明 |
|------|------|------|
| `broad_phase_comparison` | `python -m newton.examples broad_phase_comparison --broad-phase bvh` | 球体掉落，可视化对比 NxN/SAP/BVH |
| `broad_phase_benchmark` | `python -m newton.examples broad_phase_benchmark --compare-all` | 三种模式 FPS 基准对比 |
| `balls_in_box` | `python -m newton.examples balls_in_box --broad-phase bvh` | 300 个球掉入盒子，密集碰撞场景 |

### Phase 3: Convex Mesh Hill-climbing

| Demo | 命令 | 说明 |
|------|------|------|
| `convex_mesh_benchmark` | `python -m newton.examples convex_mesh_benchmark` | 64 个 icosphere 碰撞基准，打印 FPS 和邻接数据信息 |
| `convex_mesh_demo` | `python -m newton.examples convex_mesh_demo` | 1000 个 icosphere 可视化掉落 |

## 测试

### Broadphase 测试

```bash
uv run --extra dev -m newton.tests -k test_bvh
```

包含：
- `test_bvh_broadphase` — 单世界正确性
- `test_bvh_broadphase_multiple_worlds` — 多世界过滤
- `test_bvh_broadphase_with_shape_flags` — Shape flags 过滤
- `test_bvh_broadphase_consistency_with_nxn` — NxN vs BVH 一致性

### Collision Pipeline 测试

```bash
uv run --extra dev -m newton.tests -k test_collision_pipeline
```

106 个测试全部通过，覆盖：
- 原始碰撞管线测试
- BVH broadphase 集成测试
- Mesh-convex 碰撞（含 hill-climbing）
- 多世界碰撞
- 各种 shape pair 组合

### GJK 测试

```bash
uv run --extra dev -m unittest newton.tests.test_gjk
```

4 个测试全部通过：
- `test_spheres_distance` — 分离球体距离
- `test_spheres_touching` — 接触球体距离
- `test_sphere_sphere_overlapping` — 重叠球体碰撞
- `test_box_box_separated` — 分离盒子距离和法线

### 全量测试

```bash
uv run --extra dev -m newton.tests
```

3255 个测试，24 failures + 36 errors — 均为预存问题（actuators、robot examples、USD import、sensors），与碰撞优化无关。

## 验证结果

| 测试 | 结果 |
|------|------|
| Collision pipeline (106 tests) | ✅ 全部通过 |
| GJK (4 tests) | ✅ 全部通过 |
| BVH broadphase (4 tests) | ✅ 全部通过 |
| Convex mesh benchmark | ✅ 309 FPS (64 icospheres, 162 vertices each) |
| BVH vs NxN contact count | ✅ 一致 (9 contacts) |
| Convex mesh + BVH collision | ✅ 正常 (adjacency: 1304 offsets, 7680 edges) |