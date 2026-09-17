# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Filter table-guard triangles on GPU, retaining the original exact CPU SAT."""

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2 import example_mjvbd_v2_popcorn as scene


@wp.kernel(enable_backward=False)
def mesh_candidates(
    poses: wp.array[wp.transform],
    bodies: wp.array[int],
    centers: wp.array[wp.vec3d],
    extents: wp.array[wp.vec3d],
    lower: wp.vec3d,
    upper: wp.vec3d,
    active: wp.array[int],
):
    mesh = wp.tid()
    pose = poses[bodies[mesh]]
    rotation = wp.mat33d(wp.quat_to_matrix(wp.transform_get_rotation(pose)))
    center = rotation * centers[mesh] + wp.vec3d(wp.transform_get_translation(pose))
    extent = wp.vec3d(0.0)
    for row in range(3):
        for column in range(3):
            extent[row] += wp.abs(rotation[row, column]) * extents[mesh][column]
    overlaps = True
    for axis in range(3):
        if (
            center[axis] + extent[axis] + wp.float64(1e-6) < lower[axis]
            or center[axis] - extent[axis] - wp.float64(1e-6) > upper[axis]
        ):
            overlaps = False
    active[mesh] = int(overlaps)


@wp.kernel(enable_backward=False)
def triangle_candidates(
    poses: wp.array[wp.transform],
    bodies: wp.array[int],
    owners: wp.array[int],
    vertices: wp.array[wp.vec3d],
    triangles: wp.array[wp.vec3i],
    active: wp.array[int],
    lower: wp.vec3d,
    upper: wp.vec3d,
    count: wp.array[int],
    candidates: wp.array[int],
):
    triangle = wp.tid()
    mesh = owners[triangle]
    if active[mesh] == 0:
        return
    pose = poses[bodies[mesh]]
    rotation = wp.mat33d(wp.quat_to_matrix(wp.transform_get_rotation(pose)))
    translation = wp.vec3d(wp.transform_get_translation(pose))
    ids = triangles[triangle]
    a = rotation * vertices[ids[0]] + translation
    b = rotation * vertices[ids[1]] + translation
    c = rotation * vertices[ids[2]] + translation
    for axis in range(3):
        # Only broad phase: round outward, then retain the original CPU SAT.
        if wp.max(a[axis], wp.max(b[axis], c[axis])) + wp.float64(1e-9) < lower[axis]:
            return
        if wp.min(a[axis], wp.min(b[axis], c[axis])) - wp.float64(1e-9) > upper[axis]:
            return
    slot = wp.atomic_add(count, 0, 1)
    candidates[slot] = triangle


class TableGuardProbe:
    def __init__(self, example):
        self.example = example
        vertices, triangles, owners = [], [], []
        offset = 0
        for mesh, (_, points, indices) in enumerate(example.table_guard_meshes):
            vertices.append(points)
            triangles.append(indices + offset)
            owners.extend([mesh] * len(indices))
            offset += len(points)
        self.vertices = np.concatenate(vertices)
        self.triangles = np.concatenate(triangles)
        self.owners = np.asarray(owners)
        device = example.model.device
        self.points_gpu = wp.array(self.vertices, dtype=wp.vec3d, device=device)
        self.triangles_gpu = wp.array(self.triangles, dtype=wp.vec3i, device=device)
        self.owners_gpu = wp.array(self.owners, dtype=int, device=device)
        self.bodies = wp.array(example.table_guard_bodies, dtype=int, device=device)
        self.centers = wp.array(example.table_guard_centers, dtype=wp.vec3d, device=device)
        self.extents = wp.array(example.table_guard_extents, dtype=wp.vec3d, device=device)
        self.active = wp.zeros(len(vertices), dtype=int, device=device)
        self.count = wp.zeros(1, dtype=int, device=device)
        self.candidates = wp.empty(len(self.triangles), dtype=int, device=device)
        self.host_candidates = wp.empty(len(self.triangles), dtype=int, device="cpu", pinned=True)
        self.lower = wp.vec3d(scene.TABLE_FRONT - 0.001, -0.761, scene.TABLE - 0.051)
        self.upper = wp.vec3d(1.501 + scene.WORKSPACE_X, 0.761, scene.TABLE + 0.001)

    def __call__(self, joint_q):
        example = self.example
        example.table_guard_q.assign(joint_q[: example.robot_coords])
        newton.eval_fk(example.ik_model, example.table_guard_q, example.ik_model.joint_qd, example.table_guard_state)
        poses_gpu = example.table_guard_state.body_q
        wp.launch(
            mesh_candidates,
            dim=self.active.size,
            inputs=[poses_gpu, self.bodies, self.centers, self.extents, self.lower, self.upper, self.active],
        )
        self.count.zero_()
        wp.launch(
            triangle_candidates,
            dim=len(self.triangles),
            inputs=[
                poses_gpu,
                self.bodies,
                self.owners_gpu,
                self.points_gpu,
                self.triangles_gpu,
                self.active,
                self.lower,
                self.upper,
                self.count,
                self.candidates,
            ],
        )
        count = int(self.count.numpy()[0])
        if count == 0:
            return
        wp.copy(self.host_candidates, self.candidates, count=count)
        wp.synchronize_device(example.model.device)
        ids = self.host_candidates.numpy()[:count]
        poses = poses_gpu.numpy()
        for mesh in np.unique(self.owners[ids]):
            body = example.table_guard_bodies[mesh]
            selected = ids[self.owners[ids] == mesh]
            points = self.vertices[self.triangles[selected]].reshape(-1, 3)
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:]))).reshape(3, 3)
            world = points @ rotation.T + poses[body, :3]
            indices = np.arange(len(world)).reshape(-1, 3)
            if scene._mesh_intersects_table(world, indices):
                raise RuntimeError(
                    f"Robot/table clearance rejected for {example.ik_model.body_label[body]} at {example.sim_time:.3f}s"
                )
