# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental frozen-geometry caching for planar particle truncation."""

from functools import lru_cache

import warp as wp


@wp.struct
class _TruncationGeometry:
    ee_base: wp.array[wp.vec3]
    ee_hat: wp.array[wp.vec3]
    ee_normal: wp.array[wp.vec4]
    vt_base: wp.array[wp.vec3]
    vt_hat: wp.array[wp.vec3]
    vt_normal: wp.array[wp.vec4]


@lru_cache(maxsize=2)
def _make_kernels(k):
    """Specialize collision-info types and math functions for one private backend."""

    @wp.kernel
    def prepare(
        pos: wp.array[wp.vec3],
        triangles: wp.array2d[wp.int32],
        edges: wp.array2d[wp.int32],
        info_array: wp.array[k.TriMeshCollisionInfo],
        active: wp.array[wp.int32],
        cache: _TruncationGeometry,
    ):
        tid = wp.tid()
        if active[0] == 0:
            return
        info = info_array[0]
        prim = tid // 4
        lane = tid % 4
        if prim < info.edge_colliding_edges_buffer_sizes.shape[0]:
            count = k.get_edge_colliding_edges_count(info, prim)
            offset = info.edge_colliding_edges_offsets[prim]
            index = lane
            while index < count:
                slot = offset + index
                other = info.edge_colliding_edges[2 * slot + 1]
                if other >= 0:
                    p0 = pos[edges[prim, 2]]
                    p1 = pos[edges[prim, 3]]
                    p2 = pos[edges[other, 2]]
                    p3 = pos[edges[other, 3]]
                    st = wp.closest_point_edge_edge(p0, p1, p2, p3, 1.0e-6)
                    c1 = p0 + (p1 - p0) * st[0]
                    c2 = p2 + (p3 - p2) * st[1]
                    nhat = c1 - c2
                    n = wp.vec3(0.0)
                    flag = float(1.0)
                    if wp.length(nhat) < 1.0e-12:
                        n = k.robust_edge_pair_normal(p0, p1, p2, p3)
                        c2 = c1 * 0.5 + c2 * 0.5
                        flag = 0.0
                    else:
                        n = wp.normalize(nhat)
                    cache.ee_base[slot] = c2
                    cache.ee_hat[slot] = nhat
                    cache.ee_normal[slot] = wp.vec4(n[0], n[1], n[2], flag)
                index += 4
        if prim < info.vertex_colliding_triangles_buffer_sizes.shape[0]:
            count = k.get_vertex_colliding_triangles_count(info, prim)
            offset = info.vertex_colliding_triangles_offsets[prim]
            index = lane
            while index < count:
                slot = offset + index
                tri = info.vertex_colliding_triangles[2 * slot + 1]
                if tri >= 0:
                    p = pos[prim]
                    p1 = pos[triangles[tri, 0]]
                    p2 = pos[triangles[tri, 1]]
                    p3 = pos[triangles[tri, 2]]
                    c, _bary, _feature = k.triangle_closest_point(p1, p2, p3, p)
                    nhat = p - c
                    n = wp.vec3(0.0)
                    flag = float(1.0)
                    if wp.length(nhat) < 1.0e-12:
                        c = p
                        flag = 0.0
                    else:
                        n = wp.normalize(nhat)
                    cache.vt_base[slot] = c
                    cache.vt_hat[slot] = nhat
                    cache.vt_normal[slot] = wp.vec4(n[0], n[1], n[2], flag)
                index += 4

    @wp.func
    def plane(
        p0: wp.vec3,
        d0: wp.vec3,
        p1: wp.vec3,
        d1: wp.vec3,
        p2: wp.vec3,
        d2: wp.vec3,
        p3: wp.vec3,
        d3: wp.vec3,
        base: wp.vec3,
        hat: wp.vec3,
        normal: wp.vec4,
        ee: bool,
    ):
        n = wp.vec3(normal[0], normal[1], normal[2])
        if normal[3] == 0.0:
            return wp.vector(False, False, False, False, length=4, dtype=wp.bool), n, base
        a = float(0.0)
        b = float(0.0)
        if ee:
            a = wp.max(wp.vec3(-wp.dot(n, d0), -wp.dot(n, d1), 0.0))
            b = wp.max(wp.vec3(wp.dot(n, d2), wp.dot(n, d3), 0.0))
        else:
            a = wp.max(-wp.dot(n, d0), 0.0)
            b = wp.max(wp.vec4(wp.dot(n, d1), wp.dot(n, d2), wp.dot(n, d3), 0.0))
        origin = base + 0.5 * hat
        if a + b != 0.0:
            blend = wp.clamp(b / (a + b), 0.05, 0.95)
            origin = base + blend * hat
        eps_far = float(1.0e-8)
        if ee:
            eps_far = 1.0e-6
        dummy0 = a == 0.0 or not k.segment_plane_intersects(p0, d0, n, origin, 1.0e-6, -1.0e-8, eps_far, False)
        first_motion = b
        if ee:
            first_motion = a
        dummy1 = first_motion == 0.0 or not k.segment_plane_intersects(
            p1, d1, n, origin, 1.0e-6, -1.0e-8, eps_far, False
        )
        dummy2 = b == 0.0 or not k.segment_plane_intersects(p2, d2, n, origin, 1.0e-6, -1.0e-8, eps_far, False)
        dummy3 = b == 0.0 or not k.segment_plane_intersects(p3, d3, n, origin, 1.0e-6, -1.0e-8, eps_far, False)
        return wp.vector(dummy0, dummy1, dummy2, dummy3, length=4, dtype=wp.bool), n, origin

    @wp.kernel
    def truncate(
        pos: wp.array[wp.vec3],
        delta: wp.array[wp.vec3],
        triangles: wp.array2d[wp.int32],
        edges: wp.array2d[wp.int32],
        info_array: wp.array[k.TriMeshCollisionInfo],
        eps: float,
        gamma: float,
        active: wp.array[wp.int32],
        cache: _TruncationGeometry,
        ts: wp.array[float],
    ):
        tid = wp.tid()
        if active[0] == 0:
            return
        info = info_array[0]
        prim = tid // 4
        lane = tid % 4
        if prim < info.edge_colliding_edges_buffer_sizes.shape[0]:
            count = k.get_edge_colliding_edges_count(info, prim)
            offset = info.edge_colliding_edges_offsets[prim]
            index = lane
            while index < count:
                slot = offset + index
                other = info.edge_colliding_edges[2 * slot + 1]
                if other >= 0:
                    i0 = edges[prim, 2]
                    i1 = edges[prim, 3]
                    i2 = edges[other, 2]
                    i3 = edges[other, 3]
                    p0 = pos[i0]
                    p1 = pos[i1]
                    p2 = pos[i2]
                    p3 = pos[i3]
                    d0 = delta[i0]
                    d1 = delta[i1]
                    d2 = delta[i2]
                    d3 = delta[i3]
                    dummy, n, d = plane(
                        p0,
                        d0,
                        p1,
                        d1,
                        p2,
                        d2,
                        p3,
                        d3,
                        cache.ee_base[slot],
                        cache.ee_hat[slot],
                        cache.ee_normal[slot],
                        True,
                    )
                    if not dummy[0]:
                        wp.atomic_min(ts, i0, k.planar_truncation_t(p0, d0, n, d, eps, gamma))
                    if not dummy[1]:
                        wp.atomic_min(ts, i1, k.planar_truncation_t(p1, d1, n, d, eps, gamma))
                    if not dummy[2]:
                        wp.atomic_min(ts, i2, k.planar_truncation_t(p2, d2, n, d, eps, gamma))
                    if not dummy[3]:
                        wp.atomic_min(ts, i3, k.planar_truncation_t(p3, d3, n, d, eps, gamma))
                index += 4
        if prim < info.vertex_colliding_triangles_buffer_sizes.shape[0]:
            count = k.get_vertex_colliding_triangles_count(info, prim)
            offset = info.vertex_colliding_triangles_offsets[prim]
            index = lane
            while index < count:
                slot = offset + index
                tri = info.vertex_colliding_triangles[2 * slot + 1]
                if tri >= 0:
                    i0 = prim
                    i1 = triangles[tri, 0]
                    i2 = triangles[tri, 1]
                    i3 = triangles[tri, 2]
                    p0 = pos[i0]
                    p1 = pos[i1]
                    p2 = pos[i2]
                    p3 = pos[i3]
                    d0 = delta[i0]
                    d1 = delta[i1]
                    d2 = delta[i2]
                    d3 = delta[i3]
                    dummy, n, d = plane(
                        p0,
                        d0,
                        p1,
                        d1,
                        p2,
                        d2,
                        p3,
                        d3,
                        cache.vt_base[slot],
                        cache.vt_hat[slot],
                        cache.vt_normal[slot],
                        False,
                    )
                    if not dummy[0]:
                        wp.atomic_min(ts, i0, k.planar_truncation_t(p0, d0, n, d, eps, gamma))
                    if not dummy[1]:
                        wp.atomic_min(ts, i1, k.planar_truncation_t(p1, d1, n, d, eps, gamma))
                    if not dummy[2]:
                        wp.atomic_min(ts, i2, k.planar_truncation_t(p2, d2, n, d, eps, gamma))
                    if not dummy[3]:
                        wp.atomic_min(ts, i3, k.planar_truncation_t(p3, d3, n, d, eps, gamma))
                index += 4

    return prepare, truncate


@wp.kernel
def _finish_truncation(
    ids: wp.array[wp.int32],
    count: int,
    pos: wp.array[wp.vec3],
    delta: wp.array[wp.vec3],
    ts: wp.array[float],
    limit: float,
    active: wp.array[wp.int32],
    all_particles: int,
    out: wp.array[wp.vec3],
):
    tid = wp.tid()
    p = tid
    t = float(1.0)
    if active[0] != 0 or all_particles != 0:
        t = ts[p]
    else:
        if tid >= count:
            return
        p = ids[tid]
    d = delta[p] * t
    length = wp.length(d)
    if length > limit:
        d = d * limit / length
    delta[p] = d
    if out:
        out[p] = pos[p] + d
    ts[p] = 1.0


class ParticleTruncationCache:
    """Cache DAT's fixed geometry, not displacement-dependent division planes.

    The solver must rebuild this after every collision detection, including
    refreshes inside a substep. CPU kernels are supported for reference tests;
    solver integration is currently opt-in and limited to CUDA execution.
    """

    _MAX_BYTES = 256 * 1024 * 1024

    def __init__(self, solver, kernels):
        self.geometry = _TruncationGeometry()
        self._prepare, self._truncate = _make_kernels(kernels)
        self._active = getattr(solver, "has_active_self_contact", None)
        if self._active is None:
            self._active = wp.ones(1, dtype=wp.int32, device=solver.device)
        self._capacities = (0, 0)
        self.allocated_bytes = 0
        self._ensure_capacity(solver)

    def _ensure_capacity(self, solver):
        detector = solver.trimesh_collision_detector
        capacities = (
            detector.edge_colliding_edges.size // 2,
            detector.vertex_colliding_triangles.size // 2,
        )
        if all(required <= capacity for required, capacity in zip(capacities, self._capacities, strict=True)):
            return
        if solver.device.is_capturing:
            raise RuntimeError("Grow the particle truncation cache outside CUDA capture before replaying the scene")
        allocated_bytes = 40 * sum(capacities)
        if allocated_bytes > self._MAX_BYTES:
            raise ValueError("The experimental particle truncation cache exceeds its 256 MiB memory limit")
        for prefix, count in zip(("ee", "vt"), capacities, strict=True):
            setattr(self.geometry, prefix + "_base", wp.empty(count, dtype=wp.vec3, device=solver.device))
            setattr(self.geometry, prefix + "_hat", wp.empty(count, dtype=wp.vec3, device=solver.device))
            setattr(self.geometry, prefix + "_normal", wp.empty(count, dtype=wp.vec4, device=solver.device))
        self._capacities = capacities
        self.allocated_bytes = allocated_bytes

    def rebuild(self, solver):
        """Refresh fixed geometry and reset factors for the new collision snapshot."""
        self._ensure_capacity(solver)
        wp.launch(
            self._prepare,
            dim=solver.particle_self_contact_evaluation_kernel_launch_size,
            inputs=[
                solver.pos_prev_collision_detection,
                solver.model.tri_indices,
                solver.model.edge_indices,
                solver.trimesh_collision_info,
                self._active,
                self.geometry,
            ],
            device=solver.device,
        )
        solver.truncation_ts.fill_(1.0)

    def apply(self, solver, particle_q_out, selected_particles):
        """Apply ordinary DAT while resetting consumed factors for the next color."""
        wp.launch(
            self._truncate,
            dim=solver.particle_self_contact_evaluation_kernel_launch_size,
            inputs=[
                solver.pos_prev_collision_detection,
                solver.particle_displacements,
                solver.model.tri_indices,
                solver.model.edge_indices,
                solver.trimesh_collision_info,
                solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
                solver.particle_conservative_bound_relaxation,
                self._active,
                self.geometry,
            ],
            outputs=[solver.truncation_ts],
            device=solver.device,
        )
        ids = solver.model.particle_color_groups[0] if selected_particles is None else selected_particles
        wp.launch(
            _finish_truncation,
            dim=solver.model.particle_count,
            inputs=[
                ids,
                ids.size,
                solver.pos_prev_collision_detection,
                solver.particle_displacements,
                solver.truncation_ts,
                solver.particle_self_contact_margin * solver.particle_conservative_bound_relaxation * 0.5,
                self._active,
                int(selected_particles is None),
            ],
            outputs=[particle_q_out],
            device=solver.device,
        )
