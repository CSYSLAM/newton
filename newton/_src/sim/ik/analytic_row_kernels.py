# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic objective rows for instance-owned CUDA batching."""

import warp as wp

from . import ik_objectives as objectives


@wp.struct
class pos_residuals_Inputs:
    body_q: wp.array2d[wp.transform]
    target_pos: wp.array[wp.vec3]
    link_index: int
    link_offset: wp.vec3
    start_idx: int
    weight: float
    problem_idx_map: wp.array[wp.int32]
    residuals: wp.array2d[wp.float32]


@wp.func
def _pos_residuals_at(
    body_q: wp.array2d[wp.transform],
    target_pos: wp.array[wp.vec3],
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array[wp.int32],
    residuals: wp.array2d[wp.float32],
    _batch: int,
    _column: int,
):
    row = _batch
    base = problem_idx_map[row]
    body_tf = body_q[row, link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)
    error = target_pos[base] - ee_pos
    residuals[row, start_idx + 0] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]
    residuals[row, start_idx + 2] = weight * error[2]


@wp.struct
class pos_jac_analytic_Inputs:
    link_index: int
    link_offset: wp.vec3
    affects_dof: wp.array[wp.uint8]
    body_q: wp.array2d[wp.transform]
    joint_S_s: wp.array2d[wp.spatial_vector]
    start_idx: int
    n_dofs: int
    weight: float
    jacobian: wp.array3d[wp.float32]


@wp.func
def _pos_jac_analytic_at(
    link_index: int,
    link_offset: wp.vec3,
    affects_dof: wp.array[wp.uint8],
    body_q: wp.array2d[wp.transform],
    joint_S_s: wp.array2d[wp.spatial_vector],
    start_idx: int,
    n_dofs: int,
    weight: float,
    jacobian: wp.array3d[wp.float32],
    _batch: int,
    _column: int,
):
    problem_idx, dof_idx = (_batch, _column)
    if affects_dof[dof_idx] == 0:
        return
    body_tf = body_q[problem_idx, link_index]
    rot_w = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    pos_w = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    ee_pos_world = pos_w + wp.quat_rotate(rot_w, link_offset)
    S = joint_S_s[problem_idx, dof_idx]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])
    v_ee = v_orig + wp.cross(omega, ee_pos_world)
    jacobian[problem_idx, start_idx + 0, dof_idx] = -weight * v_ee[0]
    jacobian[problem_idx, start_idx + 1, dof_idx] = -weight * v_ee[1]
    jacobian[problem_idx, start_idx + 2, dof_idx] = -weight * v_ee[2]


@wp.struct
class rot_residuals_Inputs:
    body_q: wp.array2d[wp.transform]
    target_rot: wp.array[wp.vec4]
    link_index: int
    link_offset_rotation: wp.quat
    canonicalize_quat_err: wp.bool
    start_idx: int
    weight: float
    problem_idx_map: wp.array[wp.int32]
    residuals: wp.array2d[wp.float32]


@wp.func
def _rot_residuals_at(
    body_q: wp.array2d[wp.transform],
    target_rot: wp.array[wp.vec4],
    link_index: int,
    link_offset_rotation: wp.quat,
    canonicalize_quat_err: wp.bool,
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array[wp.int32],
    residuals: wp.array2d[wp.float32],
    _batch: int,
    _column: int,
):
    row = _batch
    base = problem_idx_map[row]
    body_tf = body_q[row, link_index]
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    actual_rot = body_rot * link_offset_rotation
    target_quat_vec = target_rot[base]
    target_quat = wp.quat(target_quat_vec[0], target_quat_vec[1], target_quat_vec[2], target_quat_vec[3])
    q_err = actual_rot * wp.quat_inverse(target_quat)
    if canonicalize_quat_err and wp.dot(actual_rot, target_quat) < 0.0:
        q_err = -q_err
    v_norm = wp.sqrt(q_err[0] * q_err[0] + q_err[1] * q_err[1] + q_err[2] * q_err[2])
    angle = 2.0 * wp.atan2(v_norm, q_err[3])
    eps = float(1e-08)
    axis_angle = wp.vec3(0.0, 0.0, 0.0)
    if v_norm > eps:
        axis = wp.vec3(q_err[0] / v_norm, q_err[1] / v_norm, q_err[2] / v_norm)
        axis_angle = axis * angle
    else:
        axis_angle = wp.vec3(2.0 * q_err[0], 2.0 * q_err[1], 2.0 * q_err[2])
    residuals[row, start_idx + 0] = weight * axis_angle[0]
    residuals[row, start_idx + 1] = weight * axis_angle[1]
    residuals[row, start_idx + 2] = weight * axis_angle[2]


@wp.struct
class rot_jac_analytic_Inputs:
    affects_dof: wp.array[wp.uint8]
    joint_S_s: wp.array2d[wp.spatial_vector]
    start_idx: int
    n_dofs: int
    weight: float
    jacobian: wp.array3d[wp.float32]


@wp.func
def _rot_jac_analytic_at(
    affects_dof: wp.array[wp.uint8],
    joint_S_s: wp.array2d[wp.spatial_vector],
    start_idx: int,
    n_dofs: int,
    weight: float,
    jacobian: wp.array3d[wp.float32],
    _batch: int,
    _column: int,
):
    problem_idx, dof_idx = (_batch, _column)
    if affects_dof[dof_idx] == 0:
        return
    S = joint_S_s[problem_idx, dof_idx]
    omega = wp.vec3(S[3], S[4], S[5])
    jacobian[problem_idx, start_idx + 0, dof_idx] = weight * omega[0]
    jacobian[problem_idx, start_idx + 1, dof_idx] = weight * omega[1]
    jacobian[problem_idx, start_idx + 2, dof_idx] = weight * omega[2]


@wp.struct
class limit_residuals_Inputs:
    joint_q: wp.array2d[wp.float32]
    joint_limit_lower: wp.array[wp.float32]
    joint_limit_upper: wp.array[wp.float32]
    dof_to_coord: wp.array[wp.int32]
    n_dofs: int
    weight: float
    start_idx: int
    residuals: wp.array2d[wp.float32]


@wp.func
def _limit_residuals_at(
    joint_q: wp.array2d[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    dof_to_coord: wp.array[wp.int32],
    n_dofs: int,
    weight: float,
    start_idx: int,
    residuals: wp.array2d[wp.float32],
    _batch: int,
    _column: int,
):
    problem, dof_idx = (_batch, _column)
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    if upper - lower > 990000.0:
        return
    viol = wp.max(0.0, q - upper) + wp.max(0.0, lower - q)
    residuals[problem, start_idx + dof_idx] = weight * viol


@wp.struct
class limit_jac_analytic_Inputs:
    joint_q: wp.array2d[wp.float32]
    joint_limit_lower: wp.array[wp.float32]
    joint_limit_upper: wp.array[wp.float32]
    dof_to_coord: wp.array[wp.int32]
    n_dofs: int
    start_idx: int
    weight: float
    jacobian: wp.array3d[wp.float32]


@wp.func
def _limit_jac_analytic_at(
    joint_q: wp.array2d[wp.float32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    dof_to_coord: wp.array[wp.int32],
    n_dofs: int,
    start_idx: int,
    weight: float,
    jacobian: wp.array3d[wp.float32],
    _batch: int,
    _column: int,
):
    problem, dof_idx = (_batch, _column)
    coord_idx = dof_to_coord[dof_idx]
    if coord_idx < 0:
        return
    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]
    if upper - lower > 990000.0:
        return
    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight
    jacobian[problem, start_idx + dof_idx, dof_idx] = grad


LAYOUTS = {
    objectives._pos_residuals: (
        _pos_residuals_at,
        pos_residuals_Inputs,
        ("body_q", "target_pos", "link_index", "link_offset", "start_idx", "weight", "problem_idx_map", "residuals"),
    ),
    objectives._pos_jac_analytic: (
        _pos_jac_analytic_at,
        pos_jac_analytic_Inputs,
        (
            "link_index",
            "link_offset",
            "affects_dof",
            "body_q",
            "joint_S_s",
            "start_idx",
            "n_dofs",
            "weight",
            "jacobian",
        ),
    ),
    objectives._rot_residuals: (
        _rot_residuals_at,
        rot_residuals_Inputs,
        (
            "body_q",
            "target_rot",
            "link_index",
            "link_offset_rotation",
            "canonicalize_quat_err",
            "start_idx",
            "weight",
            "problem_idx_map",
            "residuals",
        ),
    ),
    objectives._rot_jac_analytic: (
        _rot_jac_analytic_at,
        rot_jac_analytic_Inputs,
        ("affects_dof", "joint_S_s", "start_idx", "n_dofs", "weight", "jacobian"),
    ),
    objectives._limit_residuals: (
        _limit_residuals_at,
        limit_residuals_Inputs,
        (
            "joint_q",
            "joint_limit_lower",
            "joint_limit_upper",
            "dof_to_coord",
            "n_dofs",
            "weight",
            "start_idx",
            "residuals",
        ),
    ),
    objectives._limit_jac_analytic: (
        _limit_jac_analytic_at,
        limit_jac_analytic_Inputs,
        (
            "joint_q",
            "joint_limit_lower",
            "joint_limit_upper",
            "dof_to_coord",
            "n_dofs",
            "start_idx",
            "weight",
            "jacobian",
        ),
    ),
}
