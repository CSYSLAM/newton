# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-level particle correction helpers private to MJVBDV2."""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
import warp as wp

from ...geometry import ParticleFlags

__all__ = ["ParticleMultilevelCorrection"]

_COARSE_PCG_BLOCK_DIM = 256
_AUTO_MIN_ACTIVE_PARTICLES = 1024
_AUTO_MIN_CLUSTER_COUNT = 128
_AUTO_MAX_CLUSTER_COUNT = 1500

_RUNTIME_STATUS_NONFINITE_RHS = 1
_RUNTIME_STATUS_NONFINITE_SOLUTION = 2
_RUNTIME_STATUS_RESIDUAL_NOT_REDUCED = 4
_RUNTIME_STATUS_EXCESSIVE_RADIUS_CLAMP = 8
_RUNTIME_STATUS_NONPOSITIVE_CURVATURE = 16

ParticleMultilevelMode = Literal["off", "on", "auto"]
ParticleMultilevelOperator = Literal["graph", "galerkin"]


def _normalize_multilevel_mode(enabled: bool | Literal["auto"]) -> ParticleMultilevelMode:
    """Normalize the backward-compatible multilevel option."""
    if enabled is False:
        return "off"
    if enabled is True:
        return "on"
    if enabled == "auto":
        return "auto"
    raise ValueError(f"particle_enable_multilevel_correction must be False, True, or 'auto', got {enabled!r}")


def _normalize_multilevel_operator(operator: str) -> ParticleMultilevelOperator:
    """Validate the coarse operator independently of the device eligibility gate."""
    if operator in ("graph", "galerkin"):
        return operator
    raise ValueError(f"particle_multilevel_operator must be 'graph' or 'galerkin', got {operator!r}")


def _automatic_rejection_reason(model, correction: ParticleMultilevelCorrection) -> str | None:
    """Return why the conservative automatic policy rejected a hierarchy."""
    if model.tet_count > 0:
        return "tetrahedra_present"
    if model.world_count > 1:
        return "multiple_worlds"
    if correction.active_particle_count < _AUTO_MIN_ACTIVE_PARTICLES:
        return "too_few_active_particles"
    if correction.cluster_count < _AUTO_MIN_CLUSTER_COUNT:
        return "too_few_clusters"
    if correction.cluster_count > _AUTO_MAX_CLUSTER_COUNT:
        return "too_many_clusters"
    return None


_vec6f = wp.types.vector(length=6, dtype=wp.float32)
_mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
_vec9f = wp.types.vector(length=9, dtype=wp.float32)
_mat96f = wp.types.matrix(shape=(9, 6), dtype=wp.float32)


@wp.func
def _rigid_basis_displacement(value: _vec6f, offset: wp.vec3):
    translation = wp.vec3(value[0], value[1], value[2])
    rotation = wp.vec3(value[3], value[4], value[5])
    return translation + wp.cross(rotation, offset)


@wp.func
def _cluster_basis_displacement(value: _vec6f, offset: wp.vec3, rigid: bool):
    if rigid:
        return _rigid_basis_displacement(value, offset)
    return wp.vec3(value[0], value[1], value[2])


@wp.func
def _rigid_basis_force(force: wp.vec3, offset: wp.vec3):
    torque = wp.cross(offset, force)
    return _vec6f(force[0], force[1], force[2], torque[0], torque[1], torque[2])


@wp.func
def _cluster_basis_force(force: wp.vec3, offset: wp.vec3, rigid: bool):
    if rigid:
        return _rigid_basis_force(force, offset)
    return _vec6f(force[0], force[1], force[2], 0.0, 0.0, 0.0)


@wp.func
def _rigid_basis_column(column: int, offset: wp.vec3):
    if column == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if column == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    if column == 2:
        return wp.vec3(0.0, 0.0, 1.0)
    if column == 3:
        return wp.vec3(0.0, -offset[2], offset[1])
    if column == 4:
        return wp.vec3(offset[2], 0.0, -offset[0])
    return wp.vec3(-offset[1], offset[0], 0.0)


@wp.func
def _cluster_basis_column(column: int, offset: wp.vec3, rigid: bool):
    if rigid or column < 3:
        return _rigid_basis_column(column, offset)
    return wp.vec3(0.0)


@wp.func
def _dot6(left: _vec6f, right: _vec6f):
    result = float(0.0)
    for component in range(6):
        result += left[component] * right[component]
    return result


@wp.func
def _dot9(left: _vec9f, right: _vec9f):
    result = float(0.0)
    for component in range(9):
        result += left[component] * right[component]
    return result


@wp.func
def _flatten_mat33_columns(value: wp.mat33):
    return _vec9f(
        value[0, 0],
        value[1, 0],
        value[2, 0],
        value[0, 1],
        value[1, 1],
        value[2, 1],
        value[0, 2],
        value[1, 2],
        value[2, 2],
    )


@wp.func
def _tet_cluster_deformation_basis(
    tet: int,
    cluster: int,
    component: int,
    tet_indices: wp.array2d[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    rest_pose_inverse: wp.mat33,
):
    if cluster < 0:
        return _vec9f(0.0)
    delta_0 = wp.vec3(0.0)
    delta_1 = wp.vec3(0.0)
    delta_2 = wp.vec3(0.0)
    delta_3 = wp.vec3(0.0)
    particle_0 = tet_indices[tet, 0]
    particle_1 = tet_indices[tet, 1]
    particle_2 = tet_indices[tet, 2]
    particle_3 = tet_indices[tet, 3]
    centroid = cluster_centroids[cluster]
    if fine_to_coarse[particle_0] == cluster:
        delta_0 = _rigid_basis_column(component, particle_q[particle_0] - centroid)
    if fine_to_coarse[particle_1] == cluster:
        delta_1 = _rigid_basis_column(component, particle_q[particle_1] - centroid)
    if fine_to_coarse[particle_2] == cluster:
        delta_2 = _rigid_basis_column(component, particle_q[particle_2] - centroid)
    if fine_to_coarse[particle_3] == cluster:
        delta_3 = _rigid_basis_column(component, particle_q[particle_3] - centroid)
    delta_ds = wp.matrix_from_cols(delta_1 - delta_0, delta_2 - delta_0, delta_3 - delta_0)
    return _flatten_mat33_columns(delta_ds * rest_pose_inverse)


@wp.func
def _tet_cluster_deformation_basis_matrix(
    tet: int,
    cluster: int,
    tet_indices: wp.array2d[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    rest_pose_inverse: wp.mat33,
):
    basis = _mat96f(0.0)
    for column in range(6):
        column_value = _tet_cluster_deformation_basis(
            tet,
            cluster,
            column,
            tet_indices,
            fine_to_coarse,
            cluster_centroids,
            particle_q,
            rest_pose_inverse,
        )
        for row in range(9):
            basis[row, column] = column_value[row]
    return basis


@wp.func
def _tet_metric_basis_matrix(deformation_gradient: wp.mat33, deformation_basis: _mat96f):
    """Return d(F^T F)/dq in symmetric-Voigt order for one cluster."""
    metric_basis = _mat66f(0.0)
    f0 = wp.vec3(deformation_gradient[0, 0], deformation_gradient[1, 0], deformation_gradient[2, 0])
    f1 = wp.vec3(deformation_gradient[0, 1], deformation_gradient[1, 1], deformation_gradient[2, 1])
    f2 = wp.vec3(deformation_gradient[0, 2], deformation_gradient[1, 2], deformation_gradient[2, 2])
    for column in range(6):
        df0 = wp.vec3(deformation_basis[0, column], deformation_basis[1, column], deformation_basis[2, column])
        df1 = wp.vec3(deformation_basis[3, column], deformation_basis[4, column], deformation_basis[5, column])
        df2 = wp.vec3(deformation_basis[6, column], deformation_basis[7, column], deformation_basis[8, column])
        metric_basis[0, column] = 2.0 * wp.dot(f0, df0)
        metric_basis[1, column] = wp.dot(df0, f1) + wp.dot(f0, df1)
        metric_basis[2, column] = wp.dot(df0, f2) + wp.dot(f0, df2)
        metric_basis[3, column] = 2.0 * wp.dot(f1, df1)
        metric_basis[4, column] = wp.dot(df1, f2) + wp.dot(f1, df2)
        metric_basis[5, column] = 2.0 * wp.dot(f2, df2)
    return metric_basis


@wp.func
def _factor_cholesky6(matrix: _mat66f):
    factor = _mat66f(0.0)
    scale = float(1.0)
    for component in range(6):
        scale = wp.max(scale, wp.abs(matrix[component, component]))
    regularization = 1.0e-9 * scale
    for row in range(6):
        for column in range(row + 1):
            value = matrix[row, column]
            for inner in range(column):
                value -= factor[row, inner] * factor[column, inner]
            if row == column:
                factor[row, column] = wp.sqrt(wp.max(value, regularization))
            else:
                factor[row, column] = value / factor[column, column]
    return factor


@wp.func
def _solve_cholesky6(factor: _mat66f, rhs: _vec6f):
    forward = _vec6f(0.0)
    for row in range(6):
        value = rhs[row]
        for column in range(row):
            value -= factor[row, column] * forward[column]
        forward[row] = value / factor[row, row]
    solution = _vec6f(0.0)
    for reverse_row in range(6):
        row = 5 - reverse_row
        value = forward[row]
        for column in range(row + 1, 6):
            value -= factor[column, row] * solution[column]
        solution[row] = value / factor[row, row]
    return solution


@wp.func
def _membrane_hessian_block(
    face: int,
    row_order: int,
    column_order: int,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
    dt: float,
):
    """Evaluate one 3-by-3 block of the projected membrane Hessian."""
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]
    x0 = pos[v0]
    x01 = pos[v1] - x0
    x02 = pos[v2] - x0

    dm_inv_00 = tri_pose[0, 0]
    dm_inv_01 = tri_pose[0, 1]
    dm_inv_10 = tri_pose[1, 0]
    dm_inv_11 = tri_pose[1, 1]
    f0 = x01 * dm_inv_00 + x02 * dm_inv_10
    f1 = x01 * dm_inv_01 + x02 * dm_inv_11
    f0_dot_f0 = wp.dot(f0, f0)
    f1_dot_f1 = wp.dot(f1, f1)
    f0_dot_f1 = wp.dot(f0, f1)
    surface_jacobian_sq = wp.max(f0_dot_f0 * f1_dot_f1 - f0_dot_f1 * f0_dot_f1, 1.0e-20)
    surface_jacobian = wp.sqrt(surface_jacobian_sq)
    inverse_jacobian = 1.0 / surface_jacobian

    lambda_nh = lmbd + mu
    lambda_safe = wp.sign(lambda_nh) * wp.max(wp.abs(lambda_nh), 1.0e-6)
    alpha = 1.0 + mu / lambda_safe
    g0 = inverse_jacobian * (f1_dot_f1 * f0 - f0_dot_f1 * f1)
    g1 = inverse_jacobian * (f0_dot_f0 * f1 - f0_dot_f1 * f0)
    stress_scale = lambda_nh * (surface_jacobian - alpha)
    clamped_scale = wp.max(0.0, stress_scale)
    ratio = clamped_scale * inverse_jacobian
    projected_scale = lambda_nh - ratio

    row_mask_0 = float(row_order == 0)
    row_mask_1 = float(row_order == 1)
    row_mask_2 = float(row_order == 2)
    column_mask_0 = float(column_order == 0)
    column_mask_1 = float(column_order == 1)
    column_mask_2 = float(column_order == 2)
    row_df0 = dm_inv_00 * (row_mask_1 - row_mask_0) + dm_inv_10 * (row_mask_2 - row_mask_0)
    row_df1 = dm_inv_01 * (row_mask_1 - row_mask_0) + dm_inv_11 * (row_mask_2 - row_mask_0)
    column_df0 = dm_inv_00 * (column_mask_1 - column_mask_0) + dm_inv_10 * (column_mask_2 - column_mask_0)
    column_df1 = dm_inv_01 * (column_mask_1 - column_mask_0) + dm_inv_11 * (column_mask_2 - column_mask_0)

    row_dj = g0 * row_df0 + g1 * row_df1
    column_dj = g0 * column_df0 + g1 * column_df1
    row_cross = f1 * row_df0 - f0 * row_df1
    column_cross = f1 * column_df0 - f0 * column_df1
    identity_scale = mu * (row_df0 * column_df0 + row_df1 * column_df1)
    identity_scale += ratio * (
        row_df0 * column_df0 * f1_dot_f1
        + row_df1 * column_df1 * f0_dot_f0
        - (row_df0 * column_df1 + row_df1 * column_df0) * f0_dot_f1
    )
    hessian = (
        identity_scale * wp.identity(n=3, dtype=float)
        + projected_scale * wp.outer(row_dj, column_dj)
        # D_i.T @ D_j has outer(w_j, w_i); reversing these preserves the
        # diagonal but can make the assembled element matrix indefinite.
        - ratio * wp.outer(column_cross, row_cross)
    )

    if damping > 0.0:
        row_dc00 = 2.0 * row_df0 * f0
        row_dc01 = row_df0 * f1 + row_df1 * f0
        row_dc11 = 2.0 * row_df1 * f1
        column_dc00 = 2.0 * column_df0 * f0
        column_dc01 = column_df0 * f1 + column_df1 * f0
        column_dc11 = 2.0 * column_df1 * f1
        hessian += (damping / dt) * (
            wp.outer(row_dc00, column_dc00) + 2.0 * wp.outer(row_dc01, column_dc01) + wp.outer(row_dc11, column_dc11)
        )
    return area * hessian


@wp.func
def _normalized_vector_derivative(
    vector_length: float,
    normalized_vector: wp.vec3,
    vector_derivative: wp.mat33,
):
    projection = wp.identity(n=3, dtype=float) - wp.outer(normalized_vector, normalized_vector)
    return (1.0 / vector_length) * projection * vector_derivative


@wp.func
def _angle_derivative(
    normal_0: wp.vec3,
    normal_1: wp.vec3,
    edge_direction: wp.vec3,
    normal_0_derivative: wp.mat33,
    normal_1_derivative: wp.mat33,
    sine: float,
    cosine: float,
    skew_normal_0: wp.mat33,
    skew_normal_1: wp.mat33,
):
    sine_derivative = (
        wp.transpose(skew_normal_0 * normal_1_derivative - skew_normal_1 * normal_0_derivative) * edge_direction
    )
    cosine_derivative = wp.transpose(normal_0_derivative) * normal_1 + wp.transpose(normal_1_derivative) * normal_0
    return sine_derivative * cosine - cosine_derivative * sine


@wp.func
def _bending_hessian_block(
    edge: int,
    row_order: int,
    column_order: int,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[float],
    stiffness: float,
    damping: float,
    dt: float,
):
    """Evaluate one 3-by-3 block of the projected bending Hessian."""
    if edge_indices[edge, 0] < 0 or edge_indices[edge, 1] < 0:
        return wp.mat33(0.0)

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
        return wp.mat33(0.0)

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
    row_gradient = (
        float(row_order == 0) * gradient_0
        + float(row_order == 1) * gradient_1
        + float(row_order == 2) * gradient_2
        + float(row_order == 3) * gradient_3
    )
    column_gradient = (
        float(column_order == 0) * gradient_0
        + float(column_order == 1) * gradient_1
        + float(column_order == 2) * gradient_2
        + float(column_order == 3) * gradient_3
    )
    coefficient = edge_rest_length[edge] * (stiffness + damping / dt)
    return coefficient * wp.outer(row_gradient, column_gradient)


@wp.kernel(enable_backward=False)
def _restrict_particle_corrections(
    cluster_particle_offsets: wp.array[wp.int32],
    cluster_particles: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    local_correction: wp.array[wp.vec3],
    local_hessian: wp.array[wp.mat33],
    contact_hessian: wp.array[wp.mat33],
    dt: float,
    coarse_rhs: wp.array[wp.vec3],
    coarse_mass: wp.array[float],
    coarse_stiffness: wp.array[float],
    coarse_contact_stiffness: wp.array[float],
):
    cluster = wp.tid()
    force_sum = wp.vec3(0.0)
    mass_sum = float(0.0)
    stiffness_sum = float(0.0)
    contact_stiffness_sum = float(0.0)
    inv_dt_sq = 1.0 / (dt * dt)
    for slot in range(cluster_particle_offsets[cluster], cluster_particle_offsets[cluster + 1]):
        particle = cluster_particles[slot]
        mass = particle_mass[particle]
        if mass > 0.0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0:
            hessian = local_hessian[particle]
            contact = contact_hessian[particle]
            force_sum += hessian * local_correction[particle]
            mass_sum += mass
            mean_diagonal = (hessian[0, 0] + hessian[1, 1] + hessian[2, 2]) / 3.0
            contact_mean_diagonal = (contact[0, 0] + contact[1, 1] + contact[2, 2]) / 3.0
            stiffness_sum += wp.max(mean_diagonal - mass * inv_dt_sq - contact_mean_diagonal, 0.0)
            contact_stiffness_sum += wp.max(contact_mean_diagonal, 0.0)
    coarse_rhs[cluster] = force_sum
    coarse_mass[cluster] = mass_sum
    coarse_stiffness[cluster] = stiffness_sum
    coarse_contact_stiffness[cluster] = contact_stiffness_sum


@wp.kernel(enable_backward=False)
def _restrict_energy_galerkin(
    cluster_particle_offsets: wp.array[wp.int32],
    cluster_particles: wp.array[wp.int32],
    local_correction: wp.array[wp.vec3],
    local_hessian: wp.array[wp.mat33],
    diagonal_slots: wp.array[wp.int32],
    coarse_rhs: wp.array[wp.vec3],
    coarse_blocks: wp.array[wp.mat33],
):
    cluster = wp.tid()
    force_sum = wp.vec3(0.0)
    hessian_sum = wp.mat33(0.0)
    for slot in range(cluster_particle_offsets[cluster], cluster_particle_offsets[cluster + 1]):
        particle = cluster_particles[slot]
        hessian = local_hessian[particle]
        force_sum += hessian * local_correction[particle]
        hessian_sum += hessian
    coarse_rhs[cluster] = force_sum
    coarse_blocks[diagonal_slots[cluster]] = hessian_sum


@wp.kernel(enable_backward=False)
def _assemble_triangle_energy_galerkin(
    dt: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    triangle_slots: wp.array[wp.int32],
    coarse_blocks: wp.array[wp.mat33],
):
    pair = wp.tid()
    face = pair // 3
    local_pair = pair - face * 3
    row = wp.int32(0)
    column = local_pair + 1
    if local_pair == 2:
        row = 1
        column = 2
    row_slot = triangle_slots[face * 9 + row * 3 + column]
    column_slot = triangle_slots[face * 9 + column * 3 + row]
    if row_slot < 0 or column_slot < 0 or (tri_materials[face, 0] <= 0.0 and tri_materials[face, 1] <= 0.0):
        return
    block = _membrane_hessian_block(
        face,
        row,
        column,
        pos,
        tri_indices,
        tri_poses[face],
        tri_areas[face],
        tri_materials[face, 0],
        tri_materials[face, 1],
        tri_materials[face, 2],
        dt,
    )
    if row_slot == column_slot:
        wp.atomic_add(coarse_blocks, row_slot, block + wp.transpose(block))
    else:
        wp.atomic_add(coarse_blocks, row_slot, block)
        wp.atomic_add(coarse_blocks, column_slot, wp.transpose(block))


@wp.kernel(enable_backward=False)
def _assemble_bending_energy_galerkin(
    dt: float,
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    edge_slots: wp.array[wp.int32],
    coarse_blocks: wp.array[wp.mat33],
):
    pair = wp.tid()
    edge = pair // 6
    local_pair = pair - edge * 6
    row = wp.int32(0)
    column = local_pair + 1
    if local_pair >= 3 and local_pair < 5:
        row = 1
        column = local_pair - 1
    elif local_pair == 5:
        row = 2
        column = 3
    row_slot = edge_slots[edge * 16 + row * 4 + column]
    column_slot = edge_slots[edge * 16 + column * 4 + row]
    if row_slot < 0 or column_slot < 0 or edge_bending_properties[edge, 0] <= 0.0:
        return
    block = _bending_hessian_block(
        edge,
        row,
        column,
        pos,
        edge_indices,
        edge_rest_length,
        edge_bending_properties[edge, 0],
        edge_bending_properties[edge, 1],
        dt,
    )
    if row_slot == column_slot:
        wp.atomic_add(coarse_blocks, row_slot, block + wp.transpose(block))
    else:
        wp.atomic_add(coarse_blocks, row_slot, block)
        wp.atomic_add(coarse_blocks, column_slot, wp.transpose(block))


@wp.kernel(enable_backward=False)
def _assemble_spring_energy_galerkin(
    dt: float,
    pos: wp.array[wp.vec3],
    spring_indices: wp.array[wp.int32],
    spring_rest_length: wp.array[float],
    spring_stiffness: wp.array[float],
    spring_damping: wp.array[float],
    spring_slots: wp.array[wp.int32],
    coarse_blocks: wp.array[wp.mat33],
):
    spring = wp.tid()
    row_slot = spring_slots[spring * 4 + 1]
    column_slot = spring_slots[spring * 4 + 2]
    if row_slot < 0 or column_slot < 0:
        return
    particle_0 = spring_indices[spring * 2]
    particle_1 = spring_indices[spring * 2 + 1]
    difference = pos[particle_0] - pos[particle_1]
    length = wp.max(wp.length(difference), 1.0e-8)
    rest_length = spring_rest_length[spring]
    identity = wp.identity(n=3, dtype=float)
    structural = identity - (rest_length / length) * (identity - wp.outer(difference, difference) / (length * length))
    direction = difference / length
    diagonal_hessian = spring_stiffness[spring] * structural
    diagonal_hessian += (spring_damping[spring] / dt) * wp.outer(direction, direction)
    block = -diagonal_hessian
    if row_slot == column_slot:
        wp.atomic_add(coarse_blocks, row_slot, block + wp.transpose(block))
    else:
        wp.atomic_add(coarse_blocks, row_slot, block)
        wp.atomic_add(coarse_blocks, column_slot, wp.transpose(block))


@wp.func
def _coarse_edge_weight(
    cluster: wp.int32,
    neighbor: wp.int32,
    edge_multiplicity: wp.int32,
    coarse_incident_edges: wp.array[wp.int32],
    coarse_stiffness: wp.array[float],
    coupling: float,
):
    cluster_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
    neighbor_edges = wp.max(float(coarse_incident_edges[neighbor]), 1.0)
    cluster_weight = coarse_stiffness[cluster] / cluster_edges
    neighbor_weight = coarse_stiffness[neighbor] / neighbor_edges
    return 0.5 * coupling * float(edge_multiplicity) * (cluster_weight + neighbor_weight)


@wp.kernel(enable_backward=False)
def _solve_coarse_pcg_persistent(
    coarse_count: int,
    coarse_neighbor_offsets: wp.array[wp.int32],
    coarse_neighbors: wp.array[wp.int32],
    coarse_neighbor_multiplicity: wp.array[wp.int32],
    coarse_incident_edges: wp.array[wp.int32],
    coarse_anchor_edges: wp.array[wp.int32],
    rhs: wp.array[wp.vec3],
    coarse_mass: wp.array[float],
    coarse_stiffness: wp.array[float],
    coarse_contact_stiffness: wp.array[float],
    dt: float,
    coupling: float,
    iterations: int,
    validate_residual: bool,
    minimum_residual_reduction: float,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    preconditioned_residual: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    diagonal: wp.array[float],
    runtime_status: wp.array[wp.int32],
    runtime_metrics: wp.array[float],
    runtime_counters: wp.array[wp.int32],
):
    """Solve the coarse system in one block to avoid per-PCG-step launches."""
    lane = wp.tid()
    stride = wp.block_dim()
    inv_dt_sq = 1.0 / (dt * dt)
    rz_local = float(0.0)
    initial_residual_norm_sq_local = float(0.0)
    nonfinite_rhs_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        begin = coarse_neighbor_offsets[cluster]
        end = coarse_neighbor_offsets[cluster + 1]
        diagonal_value = coarse_mass[cluster] * inv_dt_sq + coarse_contact_stiffness[cluster]
        incident_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
        diagonal_value += coupling * float(coarse_anchor_edges[cluster]) * coarse_stiffness[cluster] / incident_edges
        for slot in range(begin, end):
            diagonal_value += _coarse_edge_weight(
                cluster,
                coarse_neighbors[slot],
                coarse_neighbor_multiplicity[slot],
                coarse_incident_edges,
                coarse_stiffness,
                coupling,
            )
        diagonal_value = wp.max(diagonal_value, 1.0e-12)
        r = rhs[cluster]
        z = r / diagonal_value
        solution[cluster] = wp.vec3(0.0)
        residual[cluster] = r
        preconditioned_residual[cluster] = z
        direction[cluster] = z
        diagonal[cluster] = diagonal_value
        rz_local += wp.dot(r, z)
        initial_residual_norm_sq_local += wp.dot(r, r)
        if not (wp.isfinite(r[0]) and wp.isfinite(r[1]) and wp.isfinite(r[2])):
            nonfinite_rhs_local += 1.0
        cluster += stride

    rz = wp.tile_sum(wp.tile(rz_local))[0]
    initial_residual_norm_sq = wp.tile_sum(wp.tile(initial_residual_norm_sq_local))[0]
    nonfinite_rhs_count = wp.tile_sum(wp.tile(nonfinite_rhs_local))[0]
    if lane == 0:
        runtime_metrics[4] = rz
    for _iteration in range(iterations):
        direction_product_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            value = (coarse_mass[cluster] * inv_dt_sq + coarse_contact_stiffness[cluster]) * direction[cluster]
            incident_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
            anchor_weight = coupling * float(coarse_anchor_edges[cluster]) * coarse_stiffness[cluster] / incident_edges
            value += anchor_weight * direction[cluster]
            begin = coarse_neighbor_offsets[cluster]
            end = coarse_neighbor_offsets[cluster + 1]
            for slot in range(begin, end):
                neighbor = coarse_neighbors[slot]
                weight = _coarse_edge_weight(
                    cluster,
                    neighbor,
                    coarse_neighbor_multiplicity[slot],
                    coarse_incident_edges,
                    coarse_stiffness,
                    coupling,
                )
                value += weight * (direction[cluster] - direction[neighbor])
            product[cluster] = value
            direction_product_local += wp.dot(direction[cluster], value)
            cluster += stride

        direction_product = wp.tile_sum(wp.tile(direction_product_local))[0]
        alpha = float(0.0)
        if direction_product > 1.0e-20:
            alpha = rz / direction_product

        new_rz_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            solution[cluster] += alpha * direction[cluster]
            r = residual[cluster] - alpha * product[cluster]
            z = r / diagonal[cluster]
            residual[cluster] = r
            preconditioned_residual[cluster] = z
            new_rz_local += wp.dot(r, z)
            cluster += stride

        new_rz = wp.tile_sum(wp.tile(new_rz_local))[0]
        if lane == 0:
            runtime_metrics[5 + _iteration] = new_rz
        beta = float(0.0)
        if rz > 1.0e-20:
            beta = new_rz / rz
        rz = new_rz

        direction_norm_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            direction[cluster] = preconditioned_residual[cluster] + beta * direction[cluster]
            direction_norm_local += wp.dot(direction[cluster], direction[cluster])
            cluster += stride
        # This reduction is also the block-wide barrier before neighbors read
        # the direction values written above.
        direction_norm = wp.tile_sum(wp.tile(direction_norm_local))[0]
        if direction_norm <= 1.0e-30:
            rz = 0.0

    final_residual_norm_sq_local = float(0.0)
    nonfinite_solution_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        final_residual = residual[cluster]
        coarse_solution = solution[cluster]
        final_residual_norm_sq_local += wp.dot(final_residual, final_residual)
        if not (
            wp.isfinite(final_residual[0])
            and wp.isfinite(final_residual[1])
            and wp.isfinite(final_residual[2])
            and wp.isfinite(coarse_solution[0])
            and wp.isfinite(coarse_solution[1])
            and wp.isfinite(coarse_solution[2])
        ):
            nonfinite_solution_local += 1.0
        cluster += stride

    final_residual_norm_sq = wp.tile_sum(wp.tile(final_residual_norm_sq_local))[0]
    nonfinite_solution_count = wp.tile_sum(wp.tile(nonfinite_solution_local))[0]
    if lane == 0:
        status = wp.int32(0)
        if nonfinite_rhs_count > 0.0 or not wp.isfinite(initial_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_RHS)
        if nonfinite_solution_count > 0.0 or not wp.isfinite(final_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_SOLUTION)
        if validate_residual and initial_residual_norm_sq > 1.0e-30 and wp.isfinite(final_residual_norm_sq):
            residual_reduction = 1.0 - final_residual_norm_sq / initial_residual_norm_sq
            if residual_reduction < minimum_residual_reduction:
                status = status | wp.int32(_RUNTIME_STATUS_RESIDUAL_NOT_REDUCED)
        runtime_status[0] = status
        runtime_metrics[0] = initial_residual_norm_sq
        runtime_metrics[1] = final_residual_norm_sq
        runtime_metrics[2] = 0.0
        runtime_metrics[3] = 0.0
        runtime_counters[0] = 0
        runtime_counters[1] = 0


@wp.kernel(enable_backward=False)
def _compute_cluster_centroids(
    cluster_particle_offsets: wp.array[wp.int32],
    cluster_particles: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    cluster_centroids: wp.array[wp.vec3],
):
    cluster = wp.tid()
    weighted_position = wp.vec3(0.0)
    mass_sum = float(0.0)
    for slot in range(cluster_particle_offsets[cluster], cluster_particle_offsets[cluster + 1]):
        particle = cluster_particles[slot]
        mass = particle_mass[particle]
        weighted_position += mass * particle_q[particle]
        mass_sum += mass
    if mass_sum > 0.0:
        cluster_centroids[cluster] = weighted_position / mass_sum
    else:
        cluster_centroids[cluster] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _restrict_rigid_particle_corrections(
    cluster_particle_offsets: wp.array[wp.int32],
    cluster_particles: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    cluster_is_rigid: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    local_correction: wp.array[wp.vec3],
    local_hessian: wp.array[wp.mat33],
    contact_hessian: wp.array[wp.mat33],
    dt: float,
    coarse_rhs: wp.array[_vec6f],
    coarse_local_block: wp.array[_mat66f],
    coarse_stiffness: wp.array[float],
):
    cluster = wp.tid()
    centroid = cluster_centroids[cluster]
    rigid = cluster_is_rigid[cluster] != 0
    rhs = _vec6f(0.0)
    local_block = _mat66f(0.0)
    stiffness_sum = float(0.0)
    inv_dt_sq = 1.0 / (dt * dt)
    for slot in range(cluster_particle_offsets[cluster], cluster_particle_offsets[cluster + 1]):
        particle = cluster_particles[slot]
        mass = particle_mass[particle]
        if mass > 0.0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0:
            hessian = local_hessian[particle]
            contact = contact_hessian[particle]
            force = hessian * local_correction[particle]
            offset = particle_q[particle] - centroid
            rhs += _cluster_basis_force(force, offset, rigid)

            base_hessian = contact + mass * inv_dt_sq * wp.identity(n=3, dtype=float)
            for row in range(6):
                row_basis = _cluster_basis_column(row, offset, rigid)
                for column in range(6):
                    column_basis = _cluster_basis_column(column, offset, rigid)
                    local_block[row, column] += wp.dot(row_basis, base_hessian * column_basis)

            mean_diagonal = (hessian[0, 0] + hessian[1, 1] + hessian[2, 2]) / 3.0
            contact_mean_diagonal = (contact[0, 0] + contact[1, 1] + contact[2, 2]) / 3.0
            stiffness_sum += wp.max(mean_diagonal - mass * inv_dt_sq - contact_mean_diagonal, 0.0)
    coarse_rhs[cluster] = rhs
    coarse_local_block[cluster] = local_block
    coarse_stiffness[cluster] = stiffness_sum


@wp.func
def _rigid_edge_weight(
    cluster: int,
    neighbor: int,
    coarse_incident_edges: wp.array[wp.int32],
    coarse_stiffness: wp.array[float],
    coupling: float,
):
    cluster_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
    cluster_weight = coarse_stiffness[cluster] / cluster_edges
    if neighbor >= 0:
        neighbor_edges = wp.max(float(coarse_incident_edges[neighbor]), 1.0)
        neighbor_weight = coarse_stiffness[neighbor] / neighbor_edges
        return 0.5 * coupling * (cluster_weight + neighbor_weight)
    return coupling * cluster_weight


@wp.kernel(enable_backward=False)
def _assemble_rigid_coarse_edge_blocks(
    coarse_group_source_clusters: wp.array[wp.int32],
    coarse_group_target_clusters: wp.array[wp.int32],
    coarse_group_edge_offsets: wp.array[wp.int32],
    coarse_edge_source_particles: wp.array[wp.int32],
    coarse_edge_target_particles: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    cluster_is_rigid: wp.array[wp.int32],
    coarse_incident_edges: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    coarse_stiffness: wp.array[float],
    coupling: float,
    coarse_edge_self_blocks: wp.array[_mat66f],
    coarse_edge_cross_blocks: wp.array[_mat66f],
):
    group = wp.tid()
    source_cluster = coarse_group_source_clusters[group]
    target_cluster = coarse_group_target_clusters[group]
    weight = _rigid_edge_weight(
        source_cluster,
        target_cluster,
        coarse_incident_edges,
        coarse_stiffness,
        coupling,
    )
    self_block = _mat66f(0.0)
    cross_block = _mat66f(0.0)
    source_centroid = cluster_centroids[source_cluster]
    source_rigid = cluster_is_rigid[source_cluster] != 0
    for slot in range(coarse_group_edge_offsets[group], coarse_group_edge_offsets[group + 1]):
        source_particle = coarse_edge_source_particles[slot]
        target_particle = coarse_edge_target_particles[slot]
        source_offset = particle_q[source_particle] - source_centroid
        target_offset = wp.vec3(0.0)
        target_rigid = False
        if target_cluster >= 0:
            target_offset = particle_q[target_particle] - cluster_centroids[target_cluster]
            target_rigid = cluster_is_rigid[target_cluster] != 0
        for row in range(6):
            source_row = _cluster_basis_column(row, source_offset, source_rigid)
            for column in range(6):
                source_column = _cluster_basis_column(column, source_offset, source_rigid)
                self_block[row, column] += weight * wp.dot(source_row, source_column)
                if target_cluster >= 0:
                    target_column = _cluster_basis_column(column, target_offset, target_rigid)
                    cross_block[row, column] -= weight * wp.dot(source_row, target_column)
    coarse_edge_self_blocks[group] = self_block
    coarse_edge_cross_blocks[group] = cross_block


@wp.kernel(enable_backward=False)
def _assemble_rigid_coarse_tet_blocks(
    coarse_group_source_clusters: wp.array[wp.int32],
    coarse_group_target_clusters: wp.array[wp.int32],
    coarse_group_tet_offsets: wp.array[wp.int32],
    coarse_group_tets: wp.array[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    tet_indices: wp.array2d[wp.int32],
    tet_poses: wp.array[wp.mat33],
    tet_materials: wp.array2d[float],
    dt: float,
    coarse_tet_blocks: wp.array[_mat66f],
):
    group = wp.tid()
    source_cluster = coarse_group_source_clusters[group]
    target_cluster = coarse_group_target_clusters[group]
    block = _mat66f(0.0)
    for slot in range(coarse_group_tet_offsets[group], coarse_group_tet_offsets[group + 1]):
        tet = coarse_group_tets[slot]
        rest_pose_inverse = tet_poses[tet]
        particle_0 = tet_indices[tet, 0]
        particle_1 = tet_indices[tet, 1]
        particle_2 = tet_indices[tet, 2]
        particle_3 = tet_indices[tet, 3]
        deformation_gradient = (
            wp.matrix_from_cols(
                particle_q[particle_1] - particle_q[particle_0],
                particle_q[particle_2] - particle_q[particle_0],
                particle_q[particle_3] - particle_q[particle_0],
            )
            * rest_pose_inverse
        )
        column_0 = wp.vec3(
            deformation_gradient[0, 0],
            deformation_gradient[1, 0],
            deformation_gradient[2, 0],
        )
        column_1 = wp.vec3(
            deformation_gradient[0, 1],
            deformation_gradient[1, 1],
            deformation_gradient[2, 1],
        )
        column_2 = wp.vec3(
            deformation_gradient[0, 2],
            deformation_gradient[1, 2],
            deformation_gradient[2, 2],
        )
        cofactor = wp.matrix_from_cols(
            wp.cross(column_1, column_2),
            wp.cross(column_2, column_0),
            wp.cross(column_0, column_1),
        )
        cofactor_vector = _flatten_mat33_columns(cofactor)
        rest_volume = 1.0 / (wp.determinant(rest_pose_inverse) * 6.0)
        mu = tet_materials[tet, 0]
        lmbd = tet_materials[tet, 1] + mu
        source_basis = _tet_cluster_deformation_basis_matrix(
            tet,
            source_cluster,
            tet_indices,
            fine_to_coarse,
            cluster_centroids,
            particle_q,
            rest_pose_inverse,
        )
        target_basis = source_basis
        if target_cluster != source_cluster:
            target_basis = _tet_cluster_deformation_basis_matrix(
                tet,
                target_cluster,
                tet_indices,
                fine_to_coarse,
                cluster_centroids,
                particle_q,
                rest_pose_inverse,
            )
        source_basis_transpose = wp.transpose(source_basis)
        source_cofactor_projection = source_basis_transpose * cofactor_vector
        target_basis_transpose = wp.transpose(target_basis)
        target_cofactor_projection = target_basis_transpose * cofactor_vector
        block += rest_volume * (
            mu * (source_basis_transpose * target_basis)
            + lmbd * wp.outer(source_cofactor_projection, target_cofactor_projection)
        )
        damping = tet_materials[tet, 2]
        # Avoid paying for the metric projection when its spectral scale is
        # below one part per million of the elastic block.  This bounds the
        # omitted coarse Hessian term while keeping effectively-undamped tet
        # models (which commonly use tiny nonzero sentinels) on the fast path.
        if damping / dt > 1.0e-6 * wp.max(mu, lmbd):
            source_metric_basis = _tet_metric_basis_matrix(deformation_gradient, source_basis)
            target_metric_basis = source_metric_basis
            if target_cluster != source_cluster:
                target_metric_basis = _tet_metric_basis_matrix(deformation_gradient, target_basis)
            weighted_target_metric_basis = _mat66f(0.0)
            # The off-diagonal metric components occur twice in C:C.
            for row in range(6):
                weight = 1.0
                if row == 1 or row == 2 or row == 4:
                    weight = 2.0
                for column in range(6):
                    weighted_target_metric_basis[row, column] = weight * target_metric_basis[row, column]
            block += rest_volume * damping / dt * (wp.transpose(source_metric_basis) * weighted_target_metric_basis)
    coarse_tet_blocks[group] = block


@wp.kernel(enable_backward=False)
def _solve_rigid_coarse_pcg_persistent(
    coarse_count: int,
    coarse_row_offsets: wp.array[wp.int32],
    coarse_group_target_clusters: wp.array[wp.int32],
    coarse_edge_self_blocks: wp.array[_mat66f],
    coarse_edge_cross_blocks: wp.array[_mat66f],
    coarse_tet_row_offsets: wp.array[wp.int32],
    coarse_tet_group_target_clusters: wp.array[wp.int32],
    coarse_tet_blocks: wp.array[_mat66f],
    rhs: wp.array[_vec6f],
    local_block: wp.array[_mat66f],
    iterations: int,
    solution: wp.array[_vec6f],
    residual: wp.array[_vec6f],
    preconditioned_residual: wp.array[_vec6f],
    direction: wp.array[_vec6f],
    product: wp.array[_vec6f],
    diagonal: wp.array[_mat66f],
    residual_ratio: wp.array[float],
):
    """Solve the six-DOF cluster system in one persistent CUDA block."""
    lane = wp.tid()
    stride = wp.block_dim()
    rz_local = float(0.0)
    initial_residual_norm_sq_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        diagonal_block = local_block[cluster]
        for group in range(coarse_row_offsets[cluster], coarse_row_offsets[cluster + 1]):
            diagonal_block += coarse_edge_self_blocks[group]
        for group in range(coarse_tet_row_offsets[cluster], coarse_tet_row_offsets[cluster + 1]):
            if coarse_tet_group_target_clusters[group] == cluster:
                diagonal_block += coarse_tet_blocks[group]
        factor = _factor_cholesky6(diagonal_block)

        r = rhs[cluster]
        z = _solve_cholesky6(factor, r)
        solution[cluster] = _vec6f(0.0)
        residual[cluster] = r
        preconditioned_residual[cluster] = z
        direction[cluster] = z
        diagonal[cluster] = factor
        rz_local += _dot6(r, z)
        initial_residual_norm_sq_local += _dot6(r, r)
        cluster += stride

    rz = wp.tile_sum(wp.tile(rz_local))[0]
    initial_residual_norm_sq = wp.tile_sum(wp.tile(initial_residual_norm_sq_local))[0]
    for _iteration in range(iterations):
        direction_product_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            cluster_direction = direction[cluster]
            value = local_block[cluster] * cluster_direction
            for group in range(coarse_row_offsets[cluster], coarse_row_offsets[cluster + 1]):
                value += coarse_edge_self_blocks[group] * cluster_direction
                neighbor = coarse_group_target_clusters[group]
                if neighbor >= 0:
                    value += coarse_edge_cross_blocks[group] * direction[neighbor]
            for group in range(coarse_tet_row_offsets[cluster], coarse_tet_row_offsets[cluster + 1]):
                neighbor = coarse_tet_group_target_clusters[group]
                if neighbor >= 0:
                    value += coarse_tet_blocks[group] * direction[neighbor]
            product[cluster] = value
            direction_product_local += _dot6(cluster_direction, value)
            cluster += stride

        direction_product = wp.tile_sum(wp.tile(direction_product_local))[0]
        alpha = float(0.0)
        if direction_product > 1.0e-20:
            alpha = rz / direction_product

        new_rz_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            x = solution[cluster] + alpha * direction[cluster]
            r = residual[cluster] - alpha * product[cluster]
            z = _solve_cholesky6(diagonal[cluster], r)
            solution[cluster] = x
            residual[cluster] = r
            preconditioned_residual[cluster] = z
            new_rz_local += _dot6(r, z)
            cluster += stride

        new_rz = wp.tile_sum(wp.tile(new_rz_local))[0]
        beta = float(0.0)
        if rz > 1.0e-20:
            beta = new_rz / rz
        rz = new_rz

        direction_norm_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            next_direction = preconditioned_residual[cluster] + beta * direction[cluster]
            direction[cluster] = next_direction
            direction_norm_local += _dot6(next_direction, next_direction)
            cluster += stride
        direction_norm = wp.tile_sum(wp.tile(direction_norm_local))[0]
        if direction_norm <= 1.0e-30:
            rz = 0.0

    final_residual_norm_sq_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        final_residual_norm_sq_local += _dot6(residual[cluster], residual[cluster])
        cluster += stride
    final_residual_norm_sq = wp.tile_sum(wp.tile(final_residual_norm_sq_local))[0]

    if lane == 0:
        if initial_residual_norm_sq > 1.0e-20 and final_residual_norm_sq >= 0.0:
            residual_ratio[0] = wp.sqrt(final_residual_norm_sq / initial_residual_norm_sq)
        elif initial_residual_norm_sq <= 1.0e-20:
            residual_ratio[0] = 0.0
        else:
            residual_ratio[0] = 1.0e30


@wp.kernel(enable_backward=False)
def _prolong_rigid_coarse_corrections(
    active_particles: wp.array[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    cluster_centroids: wp.array[wp.vec3],
    cluster_is_rigid: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    coarse_correction: wp.array[_vec6f],
    relaxation: float,
    max_radius_fraction: float,
    particle_displacements: wp.array[wp.vec3],
):
    particle = active_particles[wp.tid()]
    cluster = fine_to_coarse[particle]
    offset = particle_q[particle] - cluster_centroids[cluster]
    correction = relaxation * _cluster_basis_displacement(
        coarse_correction[cluster],
        offset,
        cluster_is_rigid[cluster] != 0,
    )
    correction_length = wp.length(correction)
    max_length = max_radius_fraction * particle_radius[particle]
    if correction_length > max_length:
        if max_length > 0.0:
            correction *= max_length / correction_length
        else:
            correction = wp.vec3(0.0)
    particle_displacements[particle] += correction


@wp.kernel(enable_backward=False)
def _solve_energy_galerkin_pcg_persistent(
    coarse_count: int,
    matrix_offsets: wp.array[wp.int32],
    matrix_columns: wp.array[wp.int32],
    diagonal_slots: wp.array[wp.int32],
    matrix_blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    iterations: int,
    validate_residual: bool,
    minimum_residual_reduction: float,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    preconditioned_residual: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    inverse_diagonal: wp.array[wp.mat33],
    runtime_status: wp.array[wp.int32],
    runtime_metrics: wp.array[float],
    runtime_counters: wp.array[wp.int32],
):
    """Solve the energy-projected block system in one CUDA block."""
    lane = wp.tid()
    stride = wp.block_dim()
    identity = wp.identity(n=3, dtype=float)
    rz_local = float(0.0)
    initial_residual_norm_sq_local = float(0.0)
    nonfinite_rhs_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        diagonal = matrix_blocks[diagonal_slots[cluster]]
        diagonal = 0.5 * (diagonal + wp.transpose(diagonal))
        diagonal_inverse = wp.inverse(diagonal + 1.0e-9 * identity)
        r = rhs[cluster]
        z = diagonal_inverse * r
        solution[cluster] = wp.vec3(0.0)
        residual[cluster] = r
        preconditioned_residual[cluster] = z
        direction[cluster] = z
        inverse_diagonal[cluster] = diagonal_inverse
        rz_local += wp.dot(r, z)
        initial_residual_norm_sq_local += wp.dot(r, r)
        if not (wp.isfinite(r[0]) and wp.isfinite(r[1]) and wp.isfinite(r[2])):
            nonfinite_rhs_local += 1.0
        cluster += stride

    rz = wp.tile_sum(wp.tile(rz_local))[0]
    initial_residual_norm_sq = wp.tile_sum(wp.tile(initial_residual_norm_sq_local))[0]
    nonfinite_rhs_count = wp.tile_sum(wp.tile(nonfinite_rhs_local))[0]
    curvature_failed = initial_residual_norm_sq > 1.0e-30 and rz <= 0.0
    if lane == 0:
        runtime_metrics[4] = rz
    for iteration in range(iterations):
        direction_product_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            value = wp.vec3(0.0)
            for slot in range(matrix_offsets[cluster], matrix_offsets[cluster + 1]):
                value += matrix_blocks[slot] * direction[matrix_columns[slot]]
            product[cluster] = value
            direction_product_local += wp.dot(direction[cluster], value)
            cluster += stride

        direction_product = wp.tile_sum(wp.tile(direction_product_local))[0]
        if rz > 1.0e-20 and direction_product <= 0.0:
            curvature_failed = True
        alpha = float(0.0)
        if not curvature_failed and direction_product > 1.0e-20:
            alpha = rz / direction_product

        new_rz_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            solution[cluster] += alpha * direction[cluster]
            r = residual[cluster] - alpha * product[cluster]
            z = inverse_diagonal[cluster] * r
            residual[cluster] = r
            preconditioned_residual[cluster] = z
            new_rz_local += wp.dot(r, z)
            cluster += stride

        new_rz = wp.tile_sum(wp.tile(new_rz_local))[0]
        if new_rz < 0.0:
            curvature_failed = True
        if lane == 0:
            runtime_metrics[5 + iteration] = new_rz
        beta = float(0.0)
        if rz > 1.0e-20:
            beta = new_rz / rz
        rz = new_rz

        direction_norm_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            direction[cluster] = preconditioned_residual[cluster] + beta * direction[cluster]
            direction_norm_local += wp.dot(direction[cluster], direction[cluster])
            cluster += stride
        direction_norm = wp.tile_sum(wp.tile(direction_norm_local))[0]
        if direction_norm <= 1.0e-30:
            rz = 0.0

    final_residual_norm_sq_local = float(0.0)
    nonfinite_solution_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        final_residual = residual[cluster]
        coarse_solution = solution[cluster]
        final_residual_norm_sq_local += wp.dot(final_residual, final_residual)
        if not (
            wp.isfinite(final_residual[0])
            and wp.isfinite(final_residual[1])
            and wp.isfinite(final_residual[2])
            and wp.isfinite(coarse_solution[0])
            and wp.isfinite(coarse_solution[1])
            and wp.isfinite(coarse_solution[2])
        ):
            nonfinite_solution_local += 1.0
        cluster += stride

    final_residual_norm_sq = wp.tile_sum(wp.tile(final_residual_norm_sq_local))[0]
    nonfinite_solution_count = wp.tile_sum(wp.tile(nonfinite_solution_local))[0]
    if lane == 0:
        status = wp.int32(0)
        if curvature_failed:
            status = status | wp.int32(_RUNTIME_STATUS_NONPOSITIVE_CURVATURE)
        if nonfinite_rhs_count > 0.0 or not wp.isfinite(initial_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_RHS)
        if nonfinite_solution_count > 0.0 or not wp.isfinite(final_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_SOLUTION)
        if validate_residual and initial_residual_norm_sq > 1.0e-30 and wp.isfinite(final_residual_norm_sq):
            residual_reduction = 1.0 - final_residual_norm_sq / initial_residual_norm_sq
            if residual_reduction < minimum_residual_reduction:
                status = status | wp.int32(_RUNTIME_STATUS_RESIDUAL_NOT_REDUCED)
        runtime_status[0] = status
        runtime_metrics[0] = initial_residual_norm_sq
        runtime_metrics[1] = final_residual_norm_sq
        runtime_metrics[2] = 0.0
        runtime_metrics[3] = 0.0
        runtime_counters[0] = 0
        runtime_counters[1] = 0


@wp.kernel(enable_backward=False)
def _prepare_prolonged_corrections(
    active_particles: wp.array[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    particle_radius: wp.array[float],
    coarse_correction: wp.array[wp.vec3],
    relaxation: float,
    max_radius_fraction: float,
    runtime_counters: wp.array[wp.int32],
    fine_correction: wp.array[wp.vec3],
):
    particle = active_particles[wp.tid()]
    cluster = fine_to_coarse[particle]
    correction = relaxation * coarse_correction[cluster]
    if not (wp.isfinite(correction[0]) and wp.isfinite(correction[1]) and wp.isfinite(correction[2])):
        wp.atomic_add(runtime_counters, 1, 1)
        fine_correction[particle] = wp.vec3(0.0)
        return
    correction_length = wp.length(correction)
    max_length = max_radius_fraction * particle_radius[particle]
    if correction_length > max_length:
        wp.atomic_add(runtime_counters, 0, 1)
        if max_length > 0.0:
            correction *= max_length / correction_length
        else:
            correction = wp.vec3(0.0)
    fine_correction[particle] = correction


@wp.kernel(enable_backward=False)
def _commit_prolonged_corrections(
    active_particles: wp.array[wp.int32],
    active_particle_count: int,
    fine_correction: wp.array[wp.vec3],
    max_clamp_fraction: float,
    runtime_counters: wp.array[wp.int32],
    runtime_status: wp.array[wp.int32],
    runtime_metrics: wp.array[float],
    particle_displacements: wp.array[wp.vec3],
):
    active_index = wp.tid()
    status = runtime_status[0]
    clamp_fraction = float(runtime_counters[0]) / float(active_particle_count)
    if runtime_counters[1] > 0:
        status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_SOLUTION)
    if clamp_fraction > max_clamp_fraction:
        status = status | wp.int32(_RUNTIME_STATUS_EXCESSIVE_RADIUS_CLAMP)

    if active_index == 0:
        runtime_status[0] = status
        runtime_metrics[2] = clamp_fraction
        runtime_metrics[3] = float(status == 0)

    if status == 0:
        particle = active_particles[active_index]
        particle_displacements[particle] += fine_correction[particle]


def _particle_topology_edges(model) -> np.ndarray:
    """Return unique undirected particle edges without crossing worlds."""
    edge_chunks: list[np.ndarray] = []
    if model.tri_count:
        triangles = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        edge_chunks.extend(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        )
    if model.tet_count:
        tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
        edge_chunks.extend(tetrahedra[:, pair] for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    if model.spring_count:
        edge_chunks.append(np.asarray(model.spring_indices.numpy(), dtype=np.int32).reshape((-1, 2)))
    if not edge_chunks:
        return np.empty((0, 2), dtype=np.int32)

    edges = np.concatenate(edge_chunks, axis=0)
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if model.particle_world is not None:
        particle_world = np.asarray(model.particle_world.numpy(), dtype=np.int32)
        edges = edges[particle_world[edges[:, 0]] == particle_world[edges[:, 1]]]
    return np.unique(edges, axis=0)


def _build_energy_galerkin_structure(
    model,
    fine_to_coarse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the block-CSR pattern and element-to-block lookup tables."""
    cluster_count = int(fine_to_coarse.max(initial=-1)) + 1
    row_columns: list[set[int]] = [{cluster} for cluster in range(cluster_count)]

    triangles = (
        np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        if model.tri_count
        else np.empty((0, 3), dtype=np.int32)
    )
    bending_edges = (
        np.asarray(model.edge_indices.numpy(), dtype=np.int32).reshape((-1, 4))
        if model.edge_count
        else np.empty((0, 4), dtype=np.int32)
    )
    springs = (
        np.asarray(model.spring_indices.numpy(), dtype=np.int32).reshape((-1, 2))
        if model.spring_count
        else np.empty((0, 2), dtype=np.int32)
    )

    def add_element_pattern(vertices: np.ndarray) -> None:
        clusters = [int(fine_to_coarse[particle]) for particle in vertices if particle >= 0]
        for row in clusters:
            if row >= 0:
                row_columns[row].update(column for column in clusters if column >= 0)

    for element_vertices in triangles:
        add_element_pattern(element_vertices)
    for element_vertices in bending_edges:
        if np.all(element_vertices >= 0):
            add_element_pattern(element_vertices)
    for element_vertices in springs:
        add_element_pattern(element_vertices)

    offsets = np.zeros(cluster_count + 1, dtype=np.int32)
    columns: list[int] = []
    entry_lookup: dict[tuple[int, int], int] = {}
    for row, entries in enumerate(row_columns):
        for column in sorted(entries):
            entry_lookup[row, column] = len(columns)
            columns.append(column)
        offsets[row + 1] = len(columns)
    columns_array = np.asarray(columns, dtype=np.int32)
    diagonal_slots = np.asarray([entry_lookup[cluster, cluster] for cluster in range(cluster_count)], dtype=np.int32)

    def element_slots(elements: np.ndarray, *, require_all_vertices: bool = False) -> np.ndarray:
        width = elements.shape[1]
        slots = np.full(elements.shape[0] * width * width, -1, dtype=np.int32)
        for element, vertices in enumerate(elements):
            if require_all_vertices and np.any(vertices < 0):
                continue
            for local_row, particle_row in enumerate(vertices):
                if particle_row < 0:
                    continue
                coarse_row = int(fine_to_coarse[particle_row])
                if coarse_row < 0:
                    continue
                for local_column, particle_column in enumerate(vertices):
                    if particle_column < 0:
                        continue
                    coarse_column = int(fine_to_coarse[particle_column])
                    if coarse_column >= 0:
                        slots[element * width * width + local_row * width + local_column] = entry_lookup[
                            coarse_row, coarse_column
                        ]
        return slots

    return (
        offsets,
        columns_array,
        diagonal_slots,
        element_slots(triangles),
        element_slots(bending_edges, require_all_vertices=True),
        element_slots(springs),
    )


def _build_clusters(
    model, target_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    particle_count = model.particle_count
    edges = _particle_topology_edges(model)
    neighbors: list[list[int]] = [[] for _ in range(particle_count)]
    for particle_a, particle_b in edges:
        neighbors[int(particle_a)].append(int(particle_b))
        neighbors[int(particle_b)].append(int(particle_a))

    masses = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
    flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
    movable = (masses > 0.0) & ((flags & int(ParticleFlags.ACTIVE)) != 0)
    movable &= np.fromiter((bool(particle_neighbors) for particle_neighbors in neighbors), dtype=bool)
    tet_particle = np.zeros(particle_count, dtype=bool)
    if model.tet_count:
        tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
        tet_particle[np.unique(tetrahedra)] = True
    fine_to_coarse = np.full(particle_count, -1, dtype=np.int32)
    clusters: list[list[int]] = []

    for seed in np.flatnonzero(movable):
        if fine_to_coarse[seed] >= 0:
            continue
        cluster_index = len(clusters)
        cluster: list[int] = []
        frontier = deque([int(seed)])
        fine_to_coarse[seed] = cluster_index
        while frontier and len(cluster) < target_size:
            particle = frontier.popleft()
            cluster.append(particle)
            for neighbor in neighbors[particle]:
                if movable[neighbor] and tet_particle[neighbor] == tet_particle[seed] and fine_to_coarse[neighbor] < 0:
                    fine_to_coarse[neighbor] = cluster_index
                    frontier.append(neighbor)
        # Return overflowed frontier vertices to the unassigned pool. They
        # become seeds of adjacent clusters in the outer loop.
        while frontier:
            fine_to_coarse[frontier.popleft()] = -1
        clusters.append(cluster)

    cluster_offsets = np.zeros(len(clusters) + 1, dtype=np.int32)
    if clusters:
        cluster_offsets[1:] = np.cumsum([len(cluster) for cluster in clusters], dtype=np.int32)
    cluster_particles = (
        np.concatenate([np.asarray(cluster, dtype=np.int32) for cluster in clusters])
        if clusters
        else np.empty(0, dtype=np.int32)
    )

    coarse_edges: dict[tuple[int, int], int] = {}
    coarse_incident_edges = np.zeros(len(clusters), dtype=np.int32)
    coarse_anchor_edges = np.zeros(len(clusters), dtype=np.int32)
    for particle_a, particle_b in edges:
        cluster_a = int(fine_to_coarse[particle_a])
        cluster_b = int(fine_to_coarse[particle_b])
        if cluster_a >= 0:
            coarse_incident_edges[cluster_a] += 1
        if cluster_b >= 0:
            coarse_incident_edges[cluster_b] += 1
        if cluster_a < 0 <= cluster_b:
            coarse_anchor_edges[cluster_b] += 1
            continue
        if cluster_b < 0 <= cluster_a:
            coarse_anchor_edges[cluster_a] += 1
            continue
        if cluster_a < 0 or cluster_b < 0 or cluster_a == cluster_b:
            continue
        coarse_edge = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
        coarse_edges[coarse_edge] = coarse_edges.get(coarse_edge, 0) + 1
    coarse_neighbors: list[list[tuple[int, int]]] = [[] for _ in clusters]
    for (cluster_a, cluster_b), multiplicity in sorted(coarse_edges.items()):
        coarse_neighbors[cluster_a].append((cluster_b, multiplicity))
        coarse_neighbors[cluster_b].append((cluster_a, multiplicity))
    coarse_neighbor_offsets = np.zeros(len(clusters) + 1, dtype=np.int32)
    if clusters:
        coarse_neighbor_offsets[1:] = np.cumsum(
            [len(cluster_neighbors) for cluster_neighbors in coarse_neighbors],
            dtype=np.int32,
        )
    coarse_neighbor_indices = (
        np.concatenate(
            [
                np.asarray([neighbor for neighbor, _multiplicity in entries], dtype=np.int32)
                for entries in coarse_neighbors
            ]
        )
        if any(coarse_neighbors)
        else np.empty(0, dtype=np.int32)
    )
    coarse_neighbor_multiplicity = (
        np.concatenate(
            [
                np.asarray([multiplicity for _neighbor, multiplicity in entries], dtype=np.int32)
                for entries in coarse_neighbors
            ]
        )
        if any(coarse_neighbors)
        else np.empty(0, dtype=np.int32)
    )
    return (
        fine_to_coarse,
        cluster_offsets,
        cluster_particles,
        coarse_neighbor_offsets,
        coarse_neighbor_indices,
        coarse_neighbor_multiplicity,
        coarse_incident_edges,
        coarse_anchor_edges,
    )


def _build_rigid_cluster_edges(
    model,
    fine_to_coarse: np.ndarray,
    cluster_count: int,
    cluster_is_rigid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group directed fine edges by source and target coarse clusters."""
    grouped_edges: list[dict[int, list[tuple[int, int]]]] = [{} for _ in range(cluster_count)]
    for particle_a_value, particle_b_value in _particle_topology_edges(model):
        particle_a = int(particle_a_value)
        particle_b = int(particle_b_value)
        cluster_a = int(fine_to_coarse[particle_a])
        cluster_b = int(fine_to_coarse[particle_b])
        if cluster_a < 0 and cluster_b < 0:
            continue
        if cluster_a == cluster_b:
            continue
        if cluster_a >= 0 and not cluster_is_rigid[cluster_a] and (cluster_b < 0 or not cluster_is_rigid[cluster_b]):
            grouped_edges[cluster_a].setdefault(cluster_b, []).append((particle_a, particle_b))
        if cluster_b >= 0 and not cluster_is_rigid[cluster_b] and (cluster_a < 0 or not cluster_is_rigid[cluster_a]):
            grouped_edges[cluster_b].setdefault(cluster_a, []).append((particle_b, particle_a))

    row_offsets = np.zeros(cluster_count + 1, dtype=np.int32)
    if cluster_count:
        row_offsets[1:] = np.cumsum([len(entries) for entries in grouped_edges], dtype=np.int32)
    source_clusters: list[int] = []
    target_clusters: list[int] = []
    edge_offsets = [0]
    flat_edges: list[tuple[int, int]] = []
    for source_cluster, entries in enumerate(grouped_edges):
        for target_cluster in sorted(entries):
            edges = entries[target_cluster]
            source_clusters.append(source_cluster)
            target_clusters.append(target_cluster)
            flat_edges.extend(edges)
            edge_offsets.append(len(flat_edges))
    if flat_edges:
        source_particles, target_particles = np.asarray(flat_edges, dtype=np.int32).T
    else:
        source_particles = np.empty(0, dtype=np.int32)
        target_particles = np.empty(0, dtype=np.int32)
    return (
        row_offsets,
        np.asarray(source_clusters, dtype=np.int32),
        np.asarray(target_clusters, dtype=np.int32),
        np.asarray(edge_offsets, dtype=np.int32),
        source_particles,
        target_particles,
    )


def _build_rigid_cluster_tets(
    model,
    fine_to_coarse: np.ndarray,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group tetrahedra by directed coarse block row and column."""
    grouped_tets: list[dict[int, list[int]]] = [{} for _ in range(cluster_count)]
    tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
    for tet, particles in enumerate(tetrahedra):
        clusters = np.unique(fine_to_coarse[particles])
        for source_cluster_value in clusters:
            source_cluster = int(source_cluster_value)
            if source_cluster < 0:
                continue
            for target_cluster in clusters:
                grouped_tets[source_cluster].setdefault(int(target_cluster), []).append(tet)

    row_offsets = np.zeros(cluster_count + 1, dtype=np.int32)
    if cluster_count:
        row_offsets[1:] = np.cumsum([len(entries) for entries in grouped_tets], dtype=np.int32)
    source_clusters: list[int] = []
    target_clusters: list[int] = []
    tet_offsets = [0]
    flat_tets: list[int] = []
    for source_cluster, entries in enumerate(grouped_tets):
        for target_cluster in sorted(entries):
            source_clusters.append(source_cluster)
            target_clusters.append(target_cluster)
            flat_tets.extend(entries[target_cluster])
            tet_offsets.append(len(flat_tets))
    return (
        row_offsets,
        np.asarray(source_clusters, dtype=np.int32),
        np.asarray(target_clusters, dtype=np.int32),
        np.asarray(tet_offsets, dtype=np.int32),
        np.asarray(flat_tets, dtype=np.int32),
    )


class ParticleMultilevelCorrection:
    """Fixed-topology two-level propagation correction for particle VBD."""

    def __init__(
        self,
        model,
        *,
        operator: ParticleMultilevelOperator = "graph",
        cluster_size: int,
        coarse_iterations: int,
        coupling: float,
        relaxation: float,
        max_radius_fraction: float,
        minimum_residual_reduction: float | None,
        max_clamp_fraction: float,
    ):
        operator = _normalize_multilevel_operator(operator)
        if cluster_size < 2:
            raise ValueError(f"particle multilevel cluster_size must be at least 2, got {cluster_size}")
        if coarse_iterations < 1:
            raise ValueError(f"particle multilevel coarse_iterations must be at least 1, got {coarse_iterations}")
        if coupling < 0.0:
            raise ValueError(f"particle multilevel coupling must be nonnegative, got {coupling}")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f"particle multilevel relaxation must be in (0, 1], got {relaxation}")
        if max_radius_fraction <= 0.0:
            raise ValueError(f"particle multilevel max_radius_fraction must be positive, got {max_radius_fraction}")
        if minimum_residual_reduction is not None and not 0.0 <= minimum_residual_reduction < 1.0:
            raise ValueError(
                f"particle multilevel minimum_residual_reduction must be in [0, 1), got {minimum_residual_reduction}"
            )
        if not 0.0 <= max_clamp_fraction <= 1.0:
            raise ValueError(f"particle multilevel max_clamp_fraction must be in [0, 1], got {max_clamp_fraction}")

        (
            fine_to_coarse,
            cluster_offsets,
            cluster_particles,
            coarse_neighbor_offsets,
            coarse_neighbors,
            coarse_neighbor_multiplicity,
            coarse_incident_edges,
            coarse_anchor_edges,
        ) = _build_clusters(model, cluster_size)
        self.cluster_count = int(cluster_offsets.size - 1)
        self.active_particle_count = int(cluster_particles.size)
        self.operator = operator
        self.coarse_iterations = coarse_iterations
        self.coupling = coupling
        self.relaxation = relaxation
        self.max_radius_fraction = max_radius_fraction
        self.validate_residual = minimum_residual_reduction is not None
        self.minimum_residual_reduction = 0.0 if minimum_residual_reduction is None else minimum_residual_reduction
        self.max_clamp_fraction = max_clamp_fraction
        tet_particle = np.zeros(model.particle_count, dtype=bool)
        if model.tet_count:
            tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
            tet_particle[np.unique(tetrahedra)] = True
        cluster_is_rigid = np.zeros(self.cluster_count, dtype=np.int32)
        for cluster in range(self.cluster_count):
            particles = cluster_particles[cluster_offsets[cluster] : cluster_offsets[cluster + 1]]
            if particles.size:
                cluster_is_rigid[cluster] = int(tet_particle[particles[0]])
                if np.any(tet_particle[particles] != bool(cluster_is_rigid[cluster])):
                    raise RuntimeError("particle multilevel cluster mixes surface and tetrahedral particles")
        self.use_rigid_basis = bool(np.any(cluster_is_rigid))
        self.cluster_is_rigid = wp.array(cluster_is_rigid, dtype=wp.int32, device=model.device)
        self.fine_to_coarse = wp.array(fine_to_coarse, dtype=wp.int32, device=model.device)
        self.active_particles = wp.array(cluster_particles, dtype=wp.int32, device=model.device)
        self.cluster_particle_offsets = wp.array(cluster_offsets, dtype=wp.int32, device=model.device)
        self.cluster_particles = wp.array(cluster_particles, dtype=wp.int32, device=model.device)
        self.coarse_neighbor_offsets = wp.array(coarse_neighbor_offsets, dtype=wp.int32, device=model.device)
        self.coarse_neighbors = wp.array(coarse_neighbors, dtype=wp.int32, device=model.device)
        self.coarse_neighbor_multiplicity = wp.array(
            coarse_neighbor_multiplicity,
            dtype=wp.int32,
            device=model.device,
        )
        self.coarse_incident_edges = wp.array(coarse_incident_edges, dtype=wp.int32, device=model.device)
        self.coarse_anchor_edges = wp.array(coarse_anchor_edges, dtype=wp.int32, device=model.device)
        if self.operator == "galerkin" and not self.use_rigid_basis:
            (
                coarse_matrix_offsets,
                coarse_matrix_columns,
                coarse_diagonal_slots,
                triangle_coarse_slots,
                edge_coarse_slots,
                spring_coarse_slots,
            ) = _build_energy_galerkin_structure(model, fine_to_coarse)
        else:
            coarse_matrix_offsets = np.empty(0, dtype=np.int32)
            coarse_matrix_columns = np.empty(0, dtype=np.int32)
            coarse_diagonal_slots = np.empty(0, dtype=np.int32)
            triangle_coarse_slots = np.empty(0, dtype=np.int32)
            edge_coarse_slots = np.empty(0, dtype=np.int32)
            spring_coarse_slots = np.empty(0, dtype=np.int32)
        self.coarse_matrix_offsets = wp.array(coarse_matrix_offsets, dtype=wp.int32, device=model.device)
        self.coarse_matrix_columns = wp.array(coarse_matrix_columns, dtype=wp.int32, device=model.device)
        self.coarse_diagonal_slots = wp.array(coarse_diagonal_slots, dtype=wp.int32, device=model.device)
        self.triangle_coarse_slots = wp.array(triangle_coarse_slots, dtype=wp.int32, device=model.device)
        self.edge_coarse_slots = wp.array(edge_coarse_slots, dtype=wp.int32, device=model.device)
        self.spring_coarse_slots = wp.array(spring_coarse_slots, dtype=wp.int32, device=model.device)
        self.coarse_matrix_blocks = wp.zeros(coarse_matrix_columns.size, dtype=wp.mat33, device=model.device)
        self.local_correction = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.local_hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.contact_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.contact_hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.coarse_rhs = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_mass = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_stiffness = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_contact_stiffness = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_solution = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_residual = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_preconditioned_residual = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_direction = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_product = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_diagonal = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_inverse_diagonal = wp.zeros(self.cluster_count, dtype=wp.mat33, device=model.device)
        self.fine_correction = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.runtime_status = wp.zeros(1, dtype=wp.int32, device=model.device)
        # Initial/final Euclidean residual squared, radius-clamp fraction,
        # acceptance, then the initial and per-iteration preconditioned residuals.
        self.runtime_metrics = wp.zeros(5 + coarse_iterations, dtype=float, device=model.device)
        # Radius-clamp count and non-finite fine-correction count.
        self.runtime_counters = wp.zeros(2, dtype=wp.int32, device=model.device)
        self.coarse_residual_ratio = wp.zeros(1, dtype=float, device=model.device)

        if self.use_rigid_basis:
            (
                coarse_row_offsets,
                coarse_group_source_clusters,
                coarse_group_target_clusters,
                coarse_group_edge_offsets,
                coarse_edge_source_particles,
                coarse_edge_target_particles,
            ) = _build_rigid_cluster_edges(model, fine_to_coarse, self.cluster_count, cluster_is_rigid)
            (
                coarse_tet_row_offsets,
                coarse_tet_group_source_clusters,
                coarse_tet_group_target_clusters,
                coarse_tet_group_offsets,
                coarse_group_tets,
            ) = _build_rigid_cluster_tets(model, fine_to_coarse, self.cluster_count)
            self.coarse_group_count = int(coarse_group_source_clusters.size)
            self.coarse_tet_group_count = int(coarse_tet_group_source_clusters.size)
            self.cluster_centroids = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
            self.coarse_row_offsets = wp.array(coarse_row_offsets, dtype=wp.int32, device=model.device)
            self.coarse_group_source_clusters = wp.array(
                coarse_group_source_clusters,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_group_target_clusters = wp.array(
                coarse_group_target_clusters,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_group_edge_offsets = wp.array(
                coarse_group_edge_offsets,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_edge_source_particles = wp.array(
                coarse_edge_source_particles,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_edge_target_particles = wp.array(
                coarse_edge_target_particles,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_edge_self_blocks = wp.zeros(self.coarse_group_count, dtype=_mat66f, device=model.device)
            self.coarse_edge_cross_blocks = wp.zeros(self.coarse_group_count, dtype=_mat66f, device=model.device)
            self.coarse_tet_row_offsets = wp.array(coarse_tet_row_offsets, dtype=wp.int32, device=model.device)
            self.coarse_tet_group_source_clusters = wp.array(
                coarse_tet_group_source_clusters,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_tet_group_target_clusters = wp.array(
                coarse_tet_group_target_clusters,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_tet_group_offsets = wp.array(
                coarse_tet_group_offsets,
                dtype=wp.int32,
                device=model.device,
            )
            self.coarse_group_tets = wp.array(coarse_group_tets, dtype=wp.int32, device=model.device)
            self.coarse_tet_blocks = wp.zeros(self.coarse_tet_group_count, dtype=_mat66f, device=model.device)
            self.rigid_coarse_rhs = wp.zeros(self.cluster_count, dtype=_vec6f, device=model.device)
            self.rigid_coarse_local_block = wp.zeros(self.cluster_count, dtype=_mat66f, device=model.device)
            self.rigid_coarse_solution = wp.zeros(self.cluster_count, dtype=_vec6f, device=model.device)
            self.rigid_coarse_residual = wp.zeros(self.cluster_count, dtype=_vec6f, device=model.device)
            self.rigid_coarse_preconditioned_residual = wp.zeros(
                self.cluster_count,
                dtype=_vec6f,
                device=model.device,
            )
            self.rigid_coarse_direction = wp.zeros(self.cluster_count, dtype=_vec6f, device=model.device)
            self.rigid_coarse_product = wp.zeros(self.cluster_count, dtype=_vec6f, device=model.device)
            self.rigid_coarse_diagonal = wp.zeros(self.cluster_count, dtype=_mat66f, device=model.device)
        else:
            self.coarse_group_count = 0
            self.coarse_tet_group_count = 0
            self.cluster_centroids = None
            self.coarse_row_offsets = None
            self.coarse_group_source_clusters = None
            self.coarse_group_target_clusters = None
            self.coarse_group_edge_offsets = None
            self.coarse_edge_source_particles = None
            self.coarse_edge_target_particles = None
            self.coarse_edge_self_blocks = None
            self.coarse_edge_cross_blocks = None
            self.coarse_tet_row_offsets = None
            self.coarse_tet_group_source_clusters = None
            self.coarse_tet_group_target_clusters = None
            self.coarse_tet_group_offsets = None
            self.coarse_group_tets = None
            self.coarse_tet_blocks = None
            self.rigid_coarse_rhs = None
            self.rigid_coarse_local_block = None
            self.rigid_coarse_solution = None
            self.rigid_coarse_residual = None
            self.rigid_coarse_preconditioned_residual = None
            self.rigid_coarse_direction = None
            self.rigid_coarse_product = None
            self.rigid_coarse_diagonal = None

    @property
    def enabled(self) -> bool:
        return self.cluster_count > 1 and self.active_particle_count > 0

    def restrict_and_prolong(
        self,
        model,
        particle_q: wp.array[wp.vec3],
        particle_displacements: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Restrict the local system, solve its coarse approximation, and prolong."""
        if not self.enabled:
            return
        # Keep the validated mixed/tet six-DOF path independent of the optional
        # three-DOF surface operator; do not project tet updates as translations.
        if self.use_rigid_basis:
            self._restrict_and_prolong_rigid(model, particle_q, particle_displacements, dt)
            return
        if self.operator == "galerkin":
            self.coarse_matrix_blocks.zero_()
            wp.launch(
                _restrict_energy_galerkin,
                dim=self.cluster_count,
                inputs=[
                    self.cluster_particle_offsets,
                    self.cluster_particles,
                    self.local_correction,
                    self.local_hessians,
                    self.coarse_diagonal_slots,
                ],
                outputs=[self.coarse_rhs, self.coarse_matrix_blocks],
                device=model.device,
            )
            if model.tri_count:
                wp.launch(
                    _assemble_triangle_energy_galerkin,
                    dim=model.tri_count * 3,
                    inputs=[
                        dt,
                        particle_q,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_materials,
                        model.tri_areas,
                        self.triangle_coarse_slots,
                    ],
                    outputs=[self.coarse_matrix_blocks],
                    device=model.device,
                )
            if model.edge_count:
                wp.launch(
                    _assemble_bending_energy_galerkin,
                    dim=model.edge_count * 6,
                    inputs=[
                        dt,
                        particle_q,
                        model.edge_indices,
                        model.edge_rest_length,
                        model.edge_bending_properties,
                        self.edge_coarse_slots,
                    ],
                    outputs=[self.coarse_matrix_blocks],
                    device=model.device,
                )
            if model.spring_count:
                wp.launch(
                    _assemble_spring_energy_galerkin,
                    dim=model.spring_count,
                    inputs=[
                        dt,
                        particle_q,
                        model.spring_indices,
                        model.spring_rest_length,
                        model.spring_stiffness,
                        model.spring_damping,
                        self.spring_coarse_slots,
                    ],
                    outputs=[self.coarse_matrix_blocks],
                    device=model.device,
                )
            wp.launch(
                _solve_energy_galerkin_pcg_persistent,
                dim=_COARSE_PCG_BLOCK_DIM,
                block_dim=_COARSE_PCG_BLOCK_DIM,
                inputs=[
                    self.cluster_count,
                    self.coarse_matrix_offsets,
                    self.coarse_matrix_columns,
                    self.coarse_diagonal_slots,
                    self.coarse_matrix_blocks,
                    self.coarse_rhs,
                    self.coarse_iterations,
                    self.validate_residual,
                    self.minimum_residual_reduction,
                ],
                outputs=[
                    self.coarse_solution,
                    self.coarse_residual,
                    self.coarse_preconditioned_residual,
                    self.coarse_direction,
                    self.coarse_product,
                    self.coarse_inverse_diagonal,
                    self.runtime_status,
                    self.runtime_metrics,
                    self.runtime_counters,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                _restrict_particle_corrections,
                dim=self.cluster_count,
                inputs=[
                    self.cluster_particle_offsets,
                    self.cluster_particles,
                    model.particle_mass,
                    model.particle_flags,
                    self.local_correction,
                    self.local_hessians,
                    self.contact_hessians,
                    dt,
                ],
                outputs=[
                    self.coarse_rhs,
                    self.coarse_mass,
                    self.coarse_stiffness,
                    self.coarse_contact_stiffness,
                ],
                device=model.device,
            )
            wp.launch(
                _solve_coarse_pcg_persistent,
                dim=_COARSE_PCG_BLOCK_DIM,
                block_dim=_COARSE_PCG_BLOCK_DIM,
                inputs=[
                    self.cluster_count,
                    self.coarse_neighbor_offsets,
                    self.coarse_neighbors,
                    self.coarse_neighbor_multiplicity,
                    self.coarse_incident_edges,
                    self.coarse_anchor_edges,
                    self.coarse_rhs,
                    self.coarse_mass,
                    self.coarse_stiffness,
                    self.coarse_contact_stiffness,
                    dt,
                    self.coupling,
                    self.coarse_iterations,
                    self.validate_residual,
                    self.minimum_residual_reduction,
                ],
                outputs=[
                    self.coarse_solution,
                    self.coarse_residual,
                    self.coarse_preconditioned_residual,
                    self.coarse_direction,
                    self.coarse_product,
                    self.coarse_diagonal,
                    self.runtime_status,
                    self.runtime_metrics,
                    self.runtime_counters,
                ],
                device=model.device,
            )
        wp.launch(
            _prepare_prolonged_corrections,
            dim=self.active_particle_count,
            inputs=[
                self.active_particles,
                self.fine_to_coarse,
                model.particle_radius,
                self.coarse_solution,
                self.relaxation,
                self.max_radius_fraction,
                self.runtime_counters,
            ],
            outputs=[self.fine_correction],
            device=model.device,
        )
        wp.launch(
            _commit_prolonged_corrections,
            dim=self.active_particle_count,
            inputs=[
                self.active_particles,
                self.active_particle_count,
                self.fine_correction,
                self.max_clamp_fraction,
                self.runtime_counters,
                self.runtime_status,
                self.runtime_metrics,
            ],
            outputs=[particle_displacements],
            device=model.device,
        )

    def _restrict_and_prolong_rigid(
        self,
        model,
        particle_q: wp.array[wp.vec3],
        particle_displacements: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Apply the six-DOF rigid-cluster correction used by volumetric meshes."""
        wp.launch(
            _compute_cluster_centroids,
            dim=self.cluster_count,
            inputs=[
                self.cluster_particle_offsets,
                self.cluster_particles,
                particle_q,
                model.particle_mass,
            ],
            outputs=[self.cluster_centroids],
            device=model.device,
        )
        wp.launch(
            _restrict_rigid_particle_corrections,
            dim=self.cluster_count,
            inputs=[
                self.cluster_particle_offsets,
                self.cluster_particles,
                self.cluster_centroids,
                self.cluster_is_rigid,
                particle_q,
                model.particle_mass,
                model.particle_flags,
                self.local_correction,
                self.local_hessians,
                self.contact_hessians,
                dt,
            ],
            outputs=[
                self.rigid_coarse_rhs,
                self.rigid_coarse_local_block,
                self.coarse_stiffness,
            ],
            device=model.device,
        )
        wp.launch(
            _assemble_rigid_coarse_edge_blocks,
            dim=self.coarse_group_count,
            inputs=[
                self.coarse_group_source_clusters,
                self.coarse_group_target_clusters,
                self.coarse_group_edge_offsets,
                self.coarse_edge_source_particles,
                self.coarse_edge_target_particles,
                self.cluster_centroids,
                self.cluster_is_rigid,
                self.coarse_incident_edges,
                particle_q,
                self.coarse_stiffness,
                self.coupling,
            ],
            outputs=[self.coarse_edge_self_blocks, self.coarse_edge_cross_blocks],
            device=model.device,
        )
        wp.launch(
            _assemble_rigid_coarse_tet_blocks,
            dim=self.coarse_tet_group_count,
            inputs=[
                self.coarse_tet_group_source_clusters,
                self.coarse_tet_group_target_clusters,
                self.coarse_tet_group_offsets,
                self.coarse_group_tets,
                self.fine_to_coarse,
                self.cluster_centroids,
                particle_q,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                dt,
            ],
            outputs=[self.coarse_tet_blocks],
            device=model.device,
        )
        wp.launch(
            _solve_rigid_coarse_pcg_persistent,
            dim=_COARSE_PCG_BLOCK_DIM,
            block_dim=_COARSE_PCG_BLOCK_DIM,
            inputs=[
                self.cluster_count,
                self.coarse_row_offsets,
                self.coarse_group_target_clusters,
                self.coarse_edge_self_blocks,
                self.coarse_edge_cross_blocks,
                self.coarse_tet_row_offsets,
                self.coarse_tet_group_target_clusters,
                self.coarse_tet_blocks,
                self.rigid_coarse_rhs,
                self.rigid_coarse_local_block,
                self.coarse_iterations,
            ],
            outputs=[
                self.rigid_coarse_solution,
                self.rigid_coarse_residual,
                self.rigid_coarse_preconditioned_residual,
                self.rigid_coarse_direction,
                self.rigid_coarse_product,
                self.rigid_coarse_diagonal,
                self.coarse_residual_ratio,
            ],
            device=model.device,
        )
        wp.launch(
            _prolong_rigid_coarse_corrections,
            dim=self.active_particle_count,
            inputs=[
                self.active_particles,
                self.fine_to_coarse,
                self.cluster_centroids,
                self.cluster_is_rigid,
                particle_q,
                model.particle_radius,
                self.rigid_coarse_solution,
                self.relaxation,
                self.max_radius_fraction,
            ],
            outputs=[particle_displacements],
            device=model.device,
        )
