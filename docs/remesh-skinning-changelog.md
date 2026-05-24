# Surface Remesh + Volume Skinning (PhysX CPU Exact)

## Overview

This change adds PhysX CPU-exact surface remeshing and volume skinning/embedding to Newton's native `_voxel_tet` module. The algorithm and data flow match PhysX 4.1's `ExtRemesher`, `ExtDeformableSkinning`, and `GuAABBTree` implementations, enabling the full proxy pipeline:

```
render mesh → remesh (proxy surface) → tetrahedralize → volume embedding → deform playback
```

The purpose is to allow Newton's soft body simulation to run on a low-resolution proxy tet mesh while preserving the ability to render the original high-resolution surface mesh with accurate deformation playback via barycentric skinning.

---

## Architecture

### 1. Surface Remesher (`nt_remesher.h` / `nt_remesher.cpp`)

Ported from PhysX `ExtRemesher.h/cpp`. Pipeline steps:

1. **Voxelize** — place input surface triangles into a voxel grid at the given resolution
2. **Marching cubes** — extract isosurface using `nt_marching_cubes_table.h` (256-entry lookup table, direct port from `ExtMarchingCubesTable.h`)
3. **Remove duplicate vertices** — hash-based deduplication
4. **Prune internal surfaces** — flood-fill from exterior to remove disconnected interior shells (island detection via Union-Find in `nt_union_find`)
5. **Project onto input surface** — push remeshed vertices back toward the nearest point on the original surface
6. **Optional vertex map** — create mapping from output vertex indices to nearest input vertex indices
7. **Compute normals** — per-vertex normal estimation from face adjacency

The remesher also uses `nt_box_triangle.h` (AABB-triangle SAT intersection, ported from `GuIntersectionTriangleBoxRef.h`) for voxelization — determining which grid cells each triangle overlaps.

### 2. Volume Skinning / Embedding (`nt_skinning.h` / `nt_skinning.cpp`)

Ported from PhysX `ExtDeformableSkinning.cpp` `initializeInterpolatedVertices`. For each render vertex:

1. **Build BVH** over tetrahedra (`TinyTetBvh::constructFromTetrahedra`)
2. **Create traversal controller** (`ClosestDistanceToTetmeshTraversalController`)
3. **Set query point** (render vertex position)
4. **Traverse BVH** — controller prunes nodes via `Bounds3::sqrDistance(point) <= bestDistance²`
5. **Visit leaf** — compute closest point on tetrahedron via `closestPtPointTetrahedronWithInsideCheck`
6. **Compute barycentric** — `computeBarycentricExact` on the closest tet, yielding 4-component barycentric weights

The `deform()` method then recovers deformed render positions:
```
deformedRenderVert[i] = tetVert[i0] * w0 + tetVert[i1] * w1 + tetVert[i2] * w2 + tetVert[i3] * w3
```

### 3. BVH (`nt_physx_aabb_tree.h` / `nt_physx_aabb_tree.cpp`)

Ported from PhysX `GuAABBTree.h/cpp`, `GuAABBTreeNode.h`, `GuAABBTreeBounds.h`, `GuAABBTreeBuildStats.h`.

- **BvhNode** — bit-packed `mData` encoding matching PhysX:
  - Internal: `mData = posChildIndex << 1` (bit 0 = 0)
  - Leaf: `mData = (nbPrims << 1) | 1 | (primIndex << 5)` (bit 0 = 1)
- **Build** — recursive median-split with variance-based axis selection, using NodeAllocator slab-based allocation
- **Traverse** — template method `traverse<TraversalController>(controller)` with stack-based iteration, calling `controller.shouldVisitNode(bounds)` for pruning and `controller.visitLeaf(primitiveIndex)` for leaf evaluation

### 4. Closest-Point & Barycentric (`nt_physx_distance_tetrahedron.h`)

Ported from:
- `GuDistancePointTetrahedron.h` — `closestPtPointTetrahedron`, `closestPtPointTetrahedronWithInsideCheck`
- `GuDistancePointTriangle.h` — `closestPtPointTriangle2`
- `PxMathUtils.h` — `PxComputeBarycentric` → `computeBarycentricExact`

Key detail: `closestPtPointTetrahedronWithInsideCheck` first computes barycentric coordinates via `computeBarycentricExact`, then checks whether all barycentric components are within tolerance (inside check). This matches PhysX's `PxComputeBarycentric → check bary` pattern.

### 5. Traversal Controller (`nt_physx_tetmesh_traversal.h`)

Ported from `GuAABBTree.h` `ClosestDistanceToTetmeshTraversalController`. Maintains:
- Query point, best hit result (tetId, closestPoint, distanceSquared)
- `shouldVisitNode(bounds)` — prune if `pointToAABBSqrDistance > bestDistance²`
- `visitLeaf(tetId)` — compute closest point on tetrahedron, update best if closer

### 6. Supporting Changes (`nt_types.h`)

Added to `Bounds3`:
- `contains(const Vec3& p)` — point-in-AABB test (used by BVH pruning)
- `sqrDistance(const Vec3& p)` — squared distance from point to AABB (used by traversal controller for pruning)

---

## File List

### New C++ files (header-only unless noted)

| File | Type | Ported from | Purpose |
|------|------|-------------|---------|
| `nt_remesher.h` | Header | `ExtRemesher.h` | Surface remesher class declaration |
| `nt_remesher.cpp` | Source | `ExtRemesher.cpp` | Surface remesher implementation |
| `nt_skinning.h` | Header | `ExtDeformableSkinning.cpp` | VolumeSkinning class + SkinWeight struct |
| `nt_skinning.cpp` | Source | `ExtDeformableSkinning.cpp` | BVH + traversal + barycentric embedding |
| `nt_physx_aabb_tree.h` | Header | `GuAABBTree.h`, `GuAABBTreeNode.h`, `GuAABBTreeBounds.h`, `GuAABBTreeBuildStats.h` | BvhNode, TinyTetBvh build + traverse |
| `nt_physx_aabb_tree.cpp` | Source | `GuAABBTree.cpp` | BVH build implementation |
| `nt_physx_tetmesh_traversal.h` | Header | `GuAABBTree.h` (ClosestDistanceToTetmeshTraversalController) | BVH traversal controller for closest-point queries |
| `nt_physx_distance_tetrahedron.h` | Header | `GuDistancePointTetrahedron.h`, `GuDistancePointTriangle.h`, `PxMathUtils.h` | closestPtPointTriangle2, closestPtPointTetrahedron, closestPtPointTetrahedronWithInsideCheck, computeBarycentricExact |
| `nt_box_triangle.h` | Header | `GuIntersectionTriangleBoxRef.h` | AABB-triangle SAT intersection (Tomas Akenine-Moller) |
| `nt_marching_cubes_table.h` | Header | `ExtMarchingCubesTable.h` | 256-entry marching cubes edge/face lookup tables |

### Modified files

| File | Change |
|------|--------|
| `nt_types.h` | Added `Bounds3::contains()` and `Bounds3::sqrDistance()` |
| `nt_pybind.cpp` | Added pybind11 bindings for `remesh_surface`, `compute_volume_embedding`, `deform_with_embedding`, `_skin_build_info`, `_debug_dump_bvh`; added `num_inner_tets`/`num_border_tets` to voxel debug stats |
| `CMakeLists.txt` | Added `nt_remesher.cpp`, `nt_skinning.cpp`, `nt_physx_aabb_tree.cpp` to build |
| `_voxel_tet.cp310-win_amd64.pyd` | Rebuilt binary with all new modules |
| `_voxelize_native.py` | Added `remesh_surface_native()`, `compute_volume_embedding_native()`, `deform_with_embedding_native()` Python wrappers |
| `tetgen.py` | Added `remesh_surface()`, `compute_volume_embedding()`, `deform_with_embedding()` public Python API functions |

### New test file

| File | Contents |
|------|----------|
| `test_remesh_skinning.py` | 10 tests: 5 remesh (cube basic, sphere, bounds, resolution scaling), 4 skinning (basic embedding, corner, multiple points, identity deform, translation deform), 1 integration (full pipeline) |

---

## Python API

### `newton._src.geometry.tetgen.remesh_surface()`

```python
remeshed_verts, remeshed_faces = remesh_surface(vertices, faces, resolution=100, verbose=False)
```

- Input: surface mesh `(N, 3)` float32 vertices, `(M, 3)` int32 faces
- Output: remeshed `(K, 3)` float32 vertices, `(L, 3)` uint32 faces
- `resolution`: controls voxel grid density (higher = finer proxy)

### `newton._src.geometry.tetgen.compute_volume_embedding()`

```python
tet_idx, bary_weights = compute_volume_embedding(tet_verts, tet_indices, render_verts, verbose=False)
```

- Input: tet mesh `(N, 3)` float32 vertices, `(M, 4)` int32 indices, render `(K, 3)` float32 vertices
- Output: `(K,)` int32 tet index per render vertex, `(K, 4)` float32 barycentric weights

### `newton._src.geometry.tetgen.deform_with_embedding()`

```python
deformed_render_verts = deform_with_embedding(deformed_tet_verts, tet_indices, skin_tet_idx, skin_weights)
```

- Input: deformed tet vertices `(N, 3)`, tet indices `(M, 4)`, skin data `(K,)` + `(K, 4)`
- Output: deformed render vertices `(K, 3)` float32

### `_voxel_tet._skin_build_info()`

Returns a dict describing build-time capabilities:
- `available_backends`: tuple of supported embedding implementations
- `default_backend`: currently `"physx_exact_cpu"`
- `bvh_impl`, `closest_point_impl`, `traversal_impl`, `barycentric_impl`: PhysX port provenance
- `cuda_enabled`: false (no CUDA implementation yet)

---

## Algorithmic Notes

### Inside Check Ordering

The `closestPtPointTetrahedronWithInsideCheck` function follows the PhysX pattern:

1. **First**: compute barycentric via `computeBarycentricExact` (exact 4x4 determinant solve)
2. **Then**: check if all barycentric components are within tolerance → point is inside the tet
3. **If inside**: closest point is the query point itself
4. **If outside**: fall back to closest point on tet boundary (face/edge/vertex regions)

This ordering matters because the inside check requires the barycentric coordinates, which are computed first. The alternative approach (geometric closest-point first, then barycentric) would not detect inside points correctly.

### BVH Traversal Pruning

The traversal controller uses `Bounds3::sqrDistance(queryPoint)` to prune AABB nodes. The pruning condition is:

```
sqrDistance(query, nodeBounds) <= bestDistanceSquared
```

This is the PhysX `GuAABBTreeQuery.h` pattern — a node is visited only if its AABB is closer than the current best hit. This reduces the per-query complexity from O(numTets) brute-force to O(log(numTets)) average case.

### Embedding Backend Enum

`VolumeEmbeddingImpl` has two values:
- `PhysXExactCpu` — BVH + traversal + PhysX-exact closest-point + barycentric
- `LegacyCustom` — currently falls through to `PhysXExactCpu` (placeholder for future alternative)

A previous `PhysXCudaCanonical` enum value was removed during this work since it had no real implementation.

---

## Build & Test

```bash
# Build native module
python scripts/build_voxel_tet.py --install

# Run tests
python -m unittest newton.tests.test_remesh_skinning -v
python -m unittest newton.tests.test_voxel_tet -v
```

All 10 remesh/skinning tests and 18 voxel tet tests pass.

---

## PhysX Source Provenance

All C++ code is ported from PhysX 4.1 internal modules. The original source files were used only as reference during porting and have been removed from the repo (they are not committed). The port replaces all PhysX types (`PxVec3`, `PxBounds3`, `PxU32`, etc.) with standalone equivalents (`Vec3`, `Bounds3`, `Pu32`, etc.) defined in `nt_types.h`.

Ported modules:
- `ExtRemesher.h/cpp` → `nt_remesher.h/cpp`
- `ExtMarchingCubesTable.h` → `nt_marching_cubes_table.h`
- `ExtDeformableSkinning.cpp` → `nt_skinning.cpp`
- `GuAABBTree.h/cpp` → `nt_physx_aabb_tree.h/cpp`
- `GuAABBTreeNode.h` → `BvhNode` struct in `nt_physx_aabb_tree.h`
- `GuAABBTreeBounds.h` → `AabbTreeBounds` class in `nt_physx_aabb_tree.h`
- `GuAABBTreeBuildStats.h` → `BuildStats` struct in `nt_physx_aabb_tree.h`
- `GuAABBTreeQuery.h` → `TinyTetBvh::traverse<TraversalController>` in `nt_physx_aabb_tree.h`
- `GuDistancePointTetrahedron.h` → `nt_physx_distance_tetrahedron.h`
- `GuDistancePointTriangle.h` → `closestPtPointTriangle2` in `nt_physx_distance_tetrahedron.h`
- `PxMathUtils.h` → `computeBarycentricExact` in `nt_physx_distance_tetrahedron.h`
- `GuIntersectionTriangleBoxRef.h` → `intersectTriangleBox` in `nt_box_triangle.h`