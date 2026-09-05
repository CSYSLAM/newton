# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental surface solve with cached, fixed substep damping anchors."""

from functools import lru_cache

import warp as wp

from ...geometry import ParticleFlags
from ...utils.mesh import (
    MeshAdjacencyData,
    get_vertex_adjacent_edge_id_order,
    get_vertex_adjacent_face_id_order,
    get_vertex_num_adjacent_edges,
    get_vertex_num_adjacent_faces,
)


@wp.func
def _angle(pos: wp.array[wp.vec3], indices: wp.array2d[wp.int32], edge: int) -> wp.vec2:
    if indices[edge, 0] < 0 or indices[edge, 1] < 0:
        return wp.vec2(0.0)
    x0 = pos[indices[edge, 0]]
    x1 = pos[indices[edge, 1]]
    x2 = pos[indices[edge, 2]]
    x3 = pos[indices[edge, 3]]
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)
    if n1_length < 1.0e-6 or n2_length < 1.0e-6 or e_length < 1.0e-6:
        return wp.vec2(0.0)
    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length
    theta = wp.atan2(wp.dot(wp.cross(n1_hat, n2_hat), e_hat), wp.dot(n1_hat, n2_hat))
    return wp.vec2(theta, 1.0)


@wp.kernel
def _prepare_anchor_angles(pos: wp.array[wp.vec3], indices: wp.array2d[wp.int32], anchor_angles: wp.array[wp.vec2]):
    edge = wp.tid()
    anchor_angles[edge] = _angle(pos, indices, edge)


@wp.func
def _evaluate_bending(
    edge: int,
    order: int,
    pos: wp.array[wp.vec3],
    indices: wp.array2d[wp.int32],
    rest_angles: wp.array[float],
    rest_lengths: wp.array[float],
    stiffness: float,
    damping: float,
    dt: float,
    anchor: wp.vec2,
) -> tuple[wp.vec3, wp.mat33]:
    if indices[edge, 0] < 0 or indices[edge, 1] < 0:
        return wp.vec3(0.0), wp.mat33(0.0)
    x0 = pos[indices[edge, 0]]
    x1 = pos[indices[edge, 1]]
    x2 = pos[indices[edge, 2]]
    x3 = pos[indices[edge, 3]]
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    e = x3 - x2
    n1 = wp.cross(x02, x03)
    n2 = wp.cross(x13, x12)
    n1_length = wp.length(n1)
    n2_length = wp.length(n2)
    e_length = wp.length(e)
    if n1_length < 1.0e-6 or n2_length < 1.0e-6 or e_length < 1.0e-6:
        return wp.vec3(0.0), wp.mat33(0.0)
    n1_hat = n1 / n1_length
    n2_hat = n2 / n2_length
    e_hat = e / e_length
    theta = wp.atan2(wp.dot(wp.cross(n1_hat, n2_hat), e_hat), wp.dot(n1_hat, n2_hat))

    # Contract the angle derivative for just this vertex, without four matrix chains.
    c1 = -e_length * float(order == 0) + wp.dot(x03, e_hat) * float(order == 2) - wp.dot(x02, e_hat) * float(order == 3)
    c2 = -e_length * float(order == 1) + wp.dot(x13, e_hat) * float(order == 2) - wp.dot(x12, e_hat) * float(order == 3)
    gradient = (c1 / n1_length) * n1_hat + (c2 / n2_length) * n2_hat
    k = stiffness * rest_lengths[edge]
    force = -(k * (theta - rest_angles[edge])) * gradient
    hessian = k * wp.outer(gradient, gradient)
    if damping > 0.0 and anchor[1] > 0.0:
        delta = theta - anchor[0]
        if delta > 3.141592653589793:
            delta -= 6.283185307179586
        elif delta < -3.141592653589793:
            delta += 6.283185307179586
        inv_dt = 1.0 / dt
        force += -damping * rest_lengths[edge] * (delta * inv_dt) * gradient
        hessian += damping * rest_lengths[edge] * inv_dt * wp.outer(gradient, gradient)
    return force, hessian


@lru_cache(maxsize=2)
def make_surface_kernel(evaluate_membrane):
    """Specialize the cached surface path for either private backend's membrane function."""

    @wp.kernel(enable_backward=False)
    def solve_surface_cached(
        dt: float,
        particle_ids: wp.array[wp.int32],
        pos_prev: wp.array[wp.vec3],
        pos: wp.array[wp.vec3],
        mass: wp.array[float],
        inertia: wp.array[wp.vec3],
        flags: wp.array[wp.int32],
        tri_indices: wp.array2d[wp.int32],
        tri_poses: wp.array[wp.mat22],
        tri_materials: wp.array2d[float],
        tri_areas: wp.array[float],
        edge_indices: wp.array2d[wp.int32],
        rest_angles: wp.array[float],
        rest_lengths: wp.array[float],
        bending: wp.array2d[float],
        adjacency: MeshAdjacencyData,
        forces: wp.array[wp.vec3],
        hessians: wp.array[wp.mat33],
        skip_active: int,
        skip_material: int,
        relaxation: float,
        anchor_angles: wp.array[wp.vec2],
        displacement: wp.array[wp.vec3],
    ):
        tid = wp.tid()
        lane = tid % 16
        particle = particle_ids[tid // 16]
        if skip_active == 0 and (not flags[particle] & ParticleFlags.ACTIVE or mass[particle] == 0.0):
            if lane == 0:
                displacement[particle] = wp.vec3(0.0)
            return
        force = wp.vec3(0.0)
        hessian = wp.mat33(0.0)
        index = lane
        count = get_vertex_num_adjacent_faces(adjacency, particle)
        while index < count:
            tri, order = get_vertex_adjacent_face_id_order(adjacency, particle, index)
            if skip_material == 1 or tri_materials[tri, 0] > 0.0 or tri_materials[tri, 1] > 0.0:
                f, h = evaluate_membrane(
                    tri,
                    order,
                    pos,
                    pos_prev,
                    tri_indices,
                    tri_poses[tri],
                    tri_areas[tri],
                    tri_materials[tri, 0],
                    tri_materials[tri, 1],
                    tri_materials[tri, 2],
                    dt,
                )
                force += f
                hessian += h
            index += 16
        index = lane
        count = get_vertex_num_adjacent_edges(adjacency, particle)
        while index < count:
            edge, order = get_vertex_adjacent_edge_id_order(adjacency, particle, index)
            if skip_material == 1 or bending[edge, 0] > 0.0:
                f, h = _evaluate_bending(
                    edge,
                    order,
                    pos,
                    edge_indices,
                    rest_angles,
                    rest_lengths,
                    bending[edge, 0],
                    bending[edge, 1],
                    dt,
                    anchor_angles[edge],
                )
                force += f
                hessian += h
            index += 16
        f_total = wp.tile_reduce(wp.add, wp.tile(force, preserve_type=True))[0]
        h_total = wp.tile_reduce(wp.add, wp.tile(hessian, preserve_type=True))[0]
        if lane == 0:
            inv_dt_sq = 1.0 / (dt * dt)
            h_total += mass[particle] * inv_dt_sq * wp.identity(n=3, dtype=float) + hessians[particle]
            if wp.abs(wp.determinant(h_total)) > 1.0e-8:
                f_total += mass[particle] * (inertia[particle] - pos[particle]) * inv_dt_sq + forces[particle]
                delta = wp.inverse(h_total) * f_total
                if relaxation != 1.0 and wp.ddot(hessians[particle], hessians[particle]) == 0.0:
                    delta *= relaxation
                displacement[particle] += delta

    return solve_surface_cached
