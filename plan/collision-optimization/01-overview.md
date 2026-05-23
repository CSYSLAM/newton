# 总览：架构、数据流、修改文件清单

## 碰撞管线架构

```
AABB compute → Broadphase (NxN/SAP/BVH/Explicit) → Narrowphase → Contacts
                                                    ↓
                                              GJK/MPR (凸-凸)
                                              解析碰撞 (12种原始对)
                                              BVH midphase (mesh-凸)
                                              SDF (mesh-mesh)
```

## 三个 Phase 的修改范围

### Phase 1: LBVH Broadphase (commit `541fa97a`)

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `newton/_src/geometry/broad_phase_bvh.py` | **新建** | BroadPhaseBvh 类 + BVH kernel |
| `newton/_src/geometry/__init__.py` | 修改 | 添加 BroadPhaseBvh 导出 |
| `newton/geometry.py` | 修改 | 添加 BroadPhaseBvh 到 public API |
| `newton/_src/sim/collide.py` | 修改 | 添加 "bvh" broadphase 模式 |
| `newton/tests/test_broad_phase.py` | 修改 | 添加 BVH 测试 |
| `newton/tests/test_collision_pipeline.py` | 修改 | 添加 BVH 集成测试 |
| `newton/examples/contacts/example_broad_phase_benchmark.py` | **新建** | BVH 基准对比 |
| `newton/examples/contacts/example_broad_phase_comparison.py` | **新建** | BVH 可视化对比 |
| `newton/examples/contacts/example_balls_in_box.py` | **新建** | 密集球体碰撞 |

### Phase 2: AABB 缓存 (commit `541fa97a`)

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `newton/_src/geometry/collision_core.py` | 修改 | mesh_vs_convex_midphase 使用预计算 AABB |

### Phase 3: Hill-climbing Support Mapping (未提交)

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `newton/_src/geometry/support_function.py` | 修改 | SupportMapDataProvider struct、hill-climb func、support_map/support_map_lean 签名 |
| `newton/_src/geometry/simplex_solver.py` | 修改 | solve_closest_distance_core/solve_closest_distance 添加邻接数组参数 |
| `newton/_src/geometry/mpr.py` | 修改 | minkowski_support/geometric_center/solve_mpr_core/solve_mpr 添加邻接数组参数 |
| `newton/_src/geometry/collision_convex.py` | 修改 | solve_convex_multi_contact/solve_convex_single_contact 添加邻接数组参数 |
| `newton/_src/geometry/multicontact.py` | 修改 | build_manifold 添加邻接数组参数 |
| `newton/_src/geometry/collision_core.py` | 修改 | compute_gjk_mpr_contacts/find_contacts/mesh_vs_convex_midphase/compute_tight_aabb_from_support 添加邻接数组参数 |
| `newton/_src/geometry/narrow_phase.py` | 修改 | 三个 kernel 签名 + launch_custom_write 方法 |
| `newton/_src/geometry/contact_reduction_global.py` | 修改 | mesh_triangle_contacts_to_reducer_kernel 签名 |
| `newton/_src/sim/builder.py` | 修改 | finalize() 中计算 CSR 邻接数据 |
| `newton/_src/sim/model.py` | 修改 | 5 个新字段 |
| `newton/_src/sim/collide.py` | 修改 | compute_shape_aabbs kernel + CollisionPipeline 传递邻接数组 |
| `newton/_src/solvers/kamino/_src/core/geometry.py` | 修改 | GeometriesModel 4 个新字段 |
| `newton/_src/solvers/kamino/_src/core/model.py` | 修改 | 传递邻接数组到 GeometriesModel |
| `newton/_src/solvers/kamino/_src/geometry/unified.py` | 修改 | kernel 签名 + launch 传递 |
| `newton/tests/test_gjk.py` | 修改 | 更新 kernel 签名 |
| `newton/examples/contacts/example_convex_mesh_benchmark.py` | **新建** | Convex mesh 碰撞基准 |
| `newton/examples/contacts/example_convex_mesh_demo.py` | **新建** | Convex mesh 可视化 demo |

## 数据流总览

```
Builder.finalize()
  ├─ 从 mesh.edges 构建 CSR 邻接数据
  ├─ 设置 Model.vertex_adj_offsets / vertex_adj_vertices
  ├─ 设置 Model.shape_adj_offset / shape_vertex_count
  └─ 设置 Model.total_convex_mesh_vertices

CollisionPipeline.collide()
  ├─ compute_shape_aabbs kernel ← 传入邻接数组
  │   └─ compute_tight_aabb_from_support() ← SupportMapDataProvider + 邻接数组
  │       └─ support_map() ← hill-climb 或 brute-force
  ├─ Broadphase.launch() ← BVH refit + query
  └─ NarrowPhase.launch_custom_write() ← 传入邻接数组
      ├─ narrow_phase_kernel_gjk_mpr ← find_contacts ← compute_gjk_mpr_contacts
      │   └─ SupportMapDataProvider(shape_adj_offset[shape_a], shape_vertex_count[shape_a])
      │   └─ solve_convex_single/multi_contact ← solve_mpr/solve_gjk
      │       └─ minkowski_support / geometric_center ← support_map ← hill-climb
      └─ narrow_phase_process_mesh_triangle_contacts_kernel ← compute_gjk_mpr_contacts
          └─ (同上)
```