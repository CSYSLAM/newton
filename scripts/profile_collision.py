"""Profile per-frame launch count and GPU allocation overhead in Newton collision pipeline."""

import time
import numpy as np
import warp as wp

wp.init()
wp.set_device("cuda:0")

import newton

# ── Build a medium-sized scene ──
builder = newton.ModelBuilder()
builder.add_shape_plane()

# 100 spheres - enough to stress broadphase + narrowphase
for i in range(10):
    for j in range(10):
        b = builder.add_body(xform=wp.transform(wp.vec3(i * 0.6 - 3.0, j * 0.6 - 3.0, 2.0)))
        builder.add_shape_sphere(body=b, radius=0.2)

model = builder.finalize()
print(f"Model: {model.shape_count} shapes, {model.body_count} bodies")

# ── Create collision pipelines ──
pipeline_nxn = newton.CollisionPipeline(model, broad_phase="nxn")
pipeline_sap = newton.CollisionPipeline(model, broad_phase="sap")
pipeline_bvh = newton.CollisionPipeline(model, broad_phase="bvh")

state = model.state()
contacts_nxn = pipeline_nxn.contacts()
contacts_sap = pipeline_sap.contacts()
contacts_bvh = pipeline_bvh.contacts()

# ── Warmup ──
for _ in range(5):
    pipeline_nxn.collide(state, contacts_nxn)
wp.synchronize()

# ── Count wp.launch calls by monkey-patching ──
original_launch = wp.launch
launch_log = []


def counting_launch(kernel, dim, inputs=None, outputs=None, **kwargs):
    launch_log.append((kernel.__name__ if hasattr(kernel, "__name__") else str(kernel), dim))
    if outputs is not None:
        return original_launch(kernel, dim, inputs=inputs, outputs=outputs, **kwargs)
    else:
        return original_launch(kernel, dim, inputs=inputs, **kwargs)


wp.launch = counting_launch

# ── Profile NxN ──
launch_log.clear()
t0 = time.perf_counter()
for _ in range(100):
    pipeline_nxn.collide(state, contacts_nxn)
wp.synchronize()
t_nxn = (time.perf_counter() - t0) / 100
nxn_launches = list(launch_log)

# ── Profile SAP ──
launch_log.clear()
t0 = time.perf_counter()
for _ in range(100):
    pipeline_sap.collide(state, contacts_sap)
wp.synchronize()
t_sap = (time.perf_counter() - t0) / 100
sap_launches = list(launch_log)

# ── Profile BVH ──
launch_log.clear()
t0 = time.perf_counter()
for _ in range(100):
    pipeline_bvh.collide(state, contacts_bvh)
wp.synchronize()
t_bvh = (time.perf_counter() - t0) / 100
bvh_launches = list(launch_log)

wp.launch = original_launch

# ── Report ──
print("\n" + "=" * 70)
print("SPHERE SCENE PROFILE (100 spheres)")
print("=" * 70)

for name, launches, t in [("NxN", nxn_launches, t_nxn), ("SAP", sap_launches, t_sap), ("BVH", bvh_launches, t_bvh)]:
    print(f"\n--- {name} broadphase ---")
    print(f"Total time per frame: {t * 1000:.3f} ms")
    print(f"Total wp.launch calls: {len(launches)}")
    print(f"Kernel breakdown:")
    kernel_counts = {}
    for kname, dim in launches:
        kernel_counts[kname] = kernel_counts.get(kname, 0) + 1
    for kname, count in sorted(kernel_counts.items(), key=lambda x: -x[1]):
        print(f"  {kname}: {count}x")

# ── Profile with convex meshes ──
print("\n" + "=" * 70)
print("CONVEX MESH SCENE PROFILE (64 icospheres)")
print("=" * 70)

# Build icosphere mesh
phi = (1.0 + np.sqrt(5.0)) / 2.0
vertices = [
    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
]
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
for _ in range(2):
    new_faces = []
    edge_midpoints = {}
    for face in faces:
        midpoints = []
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
            if edge not in edge_midpoints:
                v1 = np.array(vertices[edge[0]])
                v2 = np.array(vertices[edge[1]])
                mid = (v1 + v2) / 2.0
                mid = mid / np.linalg.norm(mid) * np.sqrt(phi * phi + 1)
                edge_midpoints[edge] = len(vertices)
                vertices.append(mid.tolist())
            midpoints.append(edge_midpoints[edge])
        v0, v1, v2 = face
        m0, m1, m2 = midpoints
        new_faces.extend([[v0, m0, m2], [v1, m1, m0], [v2, m2, m1], [m0, m1, m2]])
    faces = new_faces
vertices = np.array(vertices)
norms = np.linalg.norm(vertices, axis=1, keepdims=True)
vertices = vertices / norms * 0.3
indices = np.array(faces, dtype=np.int32).flatten()
mesh = newton.Mesh(vertices.astype(np.float32), indices)

builder2 = newton.ModelBuilder()
builder2.add_shape_plane()
for i in range(8):
    for j in range(8):
        b = builder2.add_body(xform=wp.transform(wp.vec3(i * 0.8 - 3.0, j * 0.8 - 3.0, 2.0)))
        builder2.add_shape_convex_hull(body=b, mesh=mesh, scale=wp.vec3(1.0, 1.0, 1.0))
model2 = builder2.finalize()
print(f"Model: {model2.shape_count} shapes, {model2.body_count} bodies")
if model2.vertex_adj_offsets is not None:
    print(f"Adjacency: {model2.vertex_adj_offsets.shape[0]} offsets, {model2.vertex_adj_vertices.shape[0]} edges")

pipeline2_bvh = newton.CollisionPipeline(model2, broad_phase="bvh")
pipeline2_nxn = newton.CollisionPipeline(model2, broad_phase="nxn")
state2 = model2.state()
contacts2_bvh = pipeline2_bvh.contacts()
contacts2_nxn = pipeline2_nxn.contacts()

# Warmup
for _ in range(5):
    pipeline2_bvh.collide(state2, contacts2_bvh)
wp.synchronize()

# Profile BVH with convex meshes
wp.launch = counting_launch
launch_log.clear()
t0 = time.perf_counter()
for _ in range(50):
    pipeline2_bvh.collide(state2, contacts2_bvh)
wp.synchronize()
t_conv_bvh = (time.perf_counter() - t0) / 50
conv_bvh_launches = list(launch_log)

# Profile NxN with convex meshes
launch_log.clear()
t0 = time.perf_counter()
for _ in range(50):
    pipeline2_nxn.collide(state2, contacts2_nxn)
wp.synchronize()
t_conv_nxn = (time.perf_counter() - t0) / 50
conv_nxn_launches = list(launch_log)

wp.launch = original_launch

for name, launches, t in [("BVH", conv_bvh_launches, t_conv_bvh), ("NxN", conv_nxn_launches, t_conv_nxn)]:
    print(f"\n--- Convex mesh + {name} ---")
    print(f"Total time per frame: {t * 1000:.3f} ms")
    print(f"Total wp.launch calls: {len(launches)}")
    print(f"Kernel breakdown:")
    kernel_counts = {}
    for kname, dim in launches:
        kernel_counts[kname] = kernel_counts.get(kname, 0) + 1
    for kname, count in sorted(kernel_counts.items(), key=lambda x: -x[1]):
        print(f"  {kname}: {count}x")

print("\nDone.")
