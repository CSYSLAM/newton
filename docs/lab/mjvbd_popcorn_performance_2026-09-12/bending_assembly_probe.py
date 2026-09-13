# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse each bending edge's four angle gradients across its six cross blocks."""

import warp as wp

from newton._src.solvers.mjvbd_v2.particle_multilevel import _angle_derivative, _normalized_vector_derivative


@wp.func
def bending_gradients(
    edge: int,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[float],
    stiffness: float,
    damping: float,
    dt: float,
):
    """Evaluate one 3-by-3 block of the projected bending Hessian."""
    if edge_indices[edge, 0] < 0 or edge_indices[edge, 1] < 0:
        return wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)

    x0 = pos[edge_indices[edge, 0]]
    x1 = pos[edge_indices[edge, 1]]
    x2 = pos[edge_indices[edge, 2]]
    x3 = pos[edge_indices[edge, 3]]
    x02 = x2 - x0
    x03 = x3 - x0
    x13 = x3 - x1
    x12 = x2 - x1
    edge_vector = x3 - x2
    normal_0_raw = wp.cross(x02, x03)
    normal_1_raw = wp.cross(x13, x12)
    normal_0_length = wp.length(normal_0_raw)
    normal_1_length = wp.length(normal_1_raw)
    edge_length = wp.length(edge_vector)
    if normal_0_length < 1.0e-6 or normal_1_length < 1.0e-6 or edge_length < 1.0e-6:
        return wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)

    normal_0 = normal_0_raw / normal_0_length
    normal_1 = normal_1_raw / normal_1_length
    edge_direction = edge_vector / edge_length
    sine = wp.dot(wp.cross(normal_0, normal_1), edge_direction)
    cosine = wp.dot(normal_0, normal_1)
    skew_edge = wp.skew(edge_vector)
    skew_x03 = wp.skew(x03)
    skew_x02 = wp.skew(x02)
    skew_x13 = wp.skew(x13)
    skew_x12 = wp.skew(x12)
    skew_normal_0 = wp.skew(normal_0)
    skew_normal_1 = wp.skew(normal_1)

    dnormal_0_dx0 = _normalized_vector_derivative(normal_0_length, normal_0, skew_edge)
    dnormal_1_dx0 = wp.mat33(0.0)
    dnormal_0_dx1 = wp.mat33(0.0)
    dnormal_1_dx1 = _normalized_vector_derivative(normal_1_length, normal_1, -skew_edge)
    dnormal_0_dx2 = _normalized_vector_derivative(normal_0_length, normal_0, -skew_x03)
    dnormal_1_dx2 = _normalized_vector_derivative(normal_1_length, normal_1, skew_x13)
    dnormal_0_dx3 = _normalized_vector_derivative(normal_0_length, normal_0, skew_x02)
    dnormal_1_dx3 = _normalized_vector_derivative(normal_1_length, normal_1, -skew_x12)

    gradient_0 = _angle_derivative(
        normal_0,
        normal_1,
        edge_direction,
        dnormal_0_dx0,
        dnormal_1_dx0,
        sine,
        cosine,
        skew_normal_0,
        skew_normal_1,
    )
    gradient_1 = _angle_derivative(
        normal_0,
        normal_1,
        edge_direction,
        dnormal_0_dx1,
        dnormal_1_dx1,
        sine,
        cosine,
        skew_normal_0,
        skew_normal_1,
    )
    gradient_2 = _angle_derivative(
        normal_0,
        normal_1,
        edge_direction,
        dnormal_0_dx2,
        dnormal_1_dx2,
        sine,
        cosine,
        skew_normal_0,
        skew_normal_1,
    )
    gradient_3 = _angle_derivative(
        normal_0,
        normal_1,
        edge_direction,
        dnormal_0_dx3,
        dnormal_1_dx3,
        sine,
        cosine,
        skew_normal_0,
        skew_normal_1,
    )
    return gradient_0, gradient_1, gradient_2, gradient_3


@wp.kernel(enable_backward=False)
def assemble_bending(
    dt: float,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    edge_slots: wp.array[wp.int32],
    coarse_blocks: wp.array[wp.mat33],
):
    edge = wp.tid()
    stiffness = edge_bending_properties[edge, 0]
    damping = edge_bending_properties[edge, 1]
    if stiffness <= 0.0:
        return
    g0, g1, g2, g3 = bending_gradients(edge, pos, edge_indices, edge_rest_length, stiffness, damping, dt)
    coefficient = edge_rest_length[edge] * (stiffness + damping / dt)
    for local_pair in range(6):
        row = int(0)
        column = local_pair + 1
        if local_pair >= 3 and local_pair < 5:
            row = 1
            column = local_pair - 1
        elif local_pair == 5:
            row = 2
            column = 3
        row_slot = edge_slots[edge * 16 + row * 4 + column]
        column_slot = edge_slots[edge * 16 + column * 4 + row]
        if row_slot >= 0 and column_slot >= 0:
            row_gradient = float(row == 0) * g0 + float(row == 1) * g1 + float(row == 2) * g2 + float(row == 3) * g3
            column_gradient = (
                float(column == 0) * g0 + float(column == 1) * g1 + float(column == 2) * g2 + float(column == 3) * g3
            )
            block = coefficient * wp.outer(row_gradient, column_gradient)
            if row_slot == column_slot:
                wp.atomic_add(coarse_blocks, row_slot, block + wp.transpose(block))
            else:
                wp.atomic_add(coarse_blocks, row_slot, block)
                wp.atomic_add(coarse_blocks, column_slot, wp.transpose(block))
