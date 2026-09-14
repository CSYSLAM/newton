# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Damped least-squares IK used by the carton-frame feedback controller."""

from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _target_frame(approach: tuple[float, float, float], up_hint: tuple[float, float, float]) -> np.ndarray:
    """Build a right-handed site frame from forward and approximate up axes."""
    forward = np.asarray(approach, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up_hint, dtype=float)
    up -= np.dot(up, forward) * forward
    up /= np.linalg.norm(up)
    lateral = np.cross(up, forward)
    return np.column_stack((forward, lateral, up))


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the world-frame logarithm of a 3x3 rotation matrix."""
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    vector = quaternion[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1.0e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, abs(quaternion[0]))
    if quaternion[0] < 0.0:
        vector = -vector
    return angle * vector / vector_norm


def solve_arm(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    side: str,
    target_position: tuple[float, float, float],
    approach: tuple[float, float, float],
    up_hint: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_joint_step: float | None = None,
    allow_partial: bool = False,
) -> np.ndarray:
    """Solve one six-axis arm pose with damped least-squares differential IK."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}/contact")
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}/{name}") for name in ARM_JOINTS]
    dof_ids = model.jnt_dofadr[joint_ids]
    qpos_ids = model.jnt_qposadr[joint_ids]
    limits = model.jnt_range[joint_ids].copy()
    if max_joint_step is not None:
        limits[:, 0] = np.maximum(limits[:, 0], qpos[qpos_ids] - max_joint_step)
        limits[:, 1] = np.minimum(limits[:, 1], qpos[qpos_ids] + max_joint_step)
    target_position_array = np.asarray(target_position)
    target_rotation = _target_frame(approach, up_hint)
    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))

    for iteration in range(320):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        position_error = target_position_array - data.site_xpos[site_id]
        current_rotation = data.site_xmat[site_id].reshape(3, 3)
        rotation_error = _rotation_vector(target_rotation @ current_rotation.T)
        orientation_weight = 0.08 if allow_partial or iteration < 100 else 0.003
        error = np.concatenate((position_error, orientation_weight * rotation_error))
        rotation_tolerance = 0.08 if allow_partial or iteration < 100 else 3.2
        if np.linalg.norm(position_error) < 0.0004 and np.linalg.norm(rotation_error) < rotation_tolerance:
            break

        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        jacobian = np.vstack((jacobian_position[:, dof_ids], orientation_weight * jacobian_rotation[:, dof_ids]))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 2.0e-4 * np.eye(6), error)
        active = np.ones(6, dtype=bool)
        for _ in range(6):
            for index, (_joint_id, qpos_id) in enumerate(zip(joint_ids, qpos_ids, strict=True)):
                lower, upper = limits[index]
                if (data.qpos[qpos_id] <= lower + 1.0e-5 and step[index] < 0.0) or (
                    data.qpos[qpos_id] >= upper - 1.0e-5 and step[index] > 0.0
                ):
                    active[index] = False
            reduced = jacobian[:, active]
            step[:] = 0.0
            step[active] = reduced.T @ np.linalg.solve(reduced @ reduced.T + 2.0e-4 * np.eye(6), error)
        step *= min(1.0, 0.12 / max(np.linalg.norm(step), 1.0e-12))
        data.qpos[qpos_ids] += step
        for index, (joint_id, qpos_id) in enumerate(zip(joint_ids, qpos_ids, strict=True)):
            if model.jnt_limited[joint_id]:
                data.qpos[qpos_id] = np.clip(data.qpos[qpos_id], *limits[index])
    else:
        if np.linalg.norm(position_error) > 0.04 and not allow_partial:
            raise RuntimeError(
                f"IK failed for {side} arm at {target_position}: "
                f"position error={np.linalg.norm(position_error):.4g}, "
                f"rotation error={np.linalg.norm(rotation_error):.4g}; joints={data.qpos[qpos_ids]}"
            )

    result = qpos.copy()
    result[qpos_ids] = data.qpos[qpos_ids]
    return result
