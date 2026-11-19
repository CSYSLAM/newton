# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Instantaneous scalar-joint actuator effort using Newton/MuJoCo semantics."""

import warp as wp


@wp.func
def _point_velocity(pose: wp.transform, twist: wp.spatial_vector, center: wp.vec3, point: wp.vec3):
    world_center = wp.transform_point(pose, center)
    return wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), point - world_center)


@wp.func
def _unwrap_near(value: float, reference: float):
    """Return the 2*pi-equivalent of ``value`` nearest ``reference``."""
    turns = (reference - value) / (2.0 * wp.pi)
    nearest_turn = wp.sign(turns) * wp.floor(wp.abs(turns) + 0.5)
    return value + (2.0 * wp.pi) * nearest_turn


@wp.kernel
def evaluate_joint_drive_effort(
    body_q: wp.array[wp.transform],
    previous_body_q: wp.array[wp.transform],
    previous_angles: wp.array[wp.float32],
    angle_initialized: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    affine_body_ids: wp.array[wp.int32],
    drive_types: wp.array[wp.int32],
    drive_bodies: wp.array[wp.vec2i],
    parent_origins: wp.array[wp.vec3],
    child_origins: wp.array[wp.vec3],
    parent_axis_points: wp.array[wp.vec3],
    parent_references: wp.array[wp.vec3],
    child_references: wp.array[wp.vec3],
    dof_indices: wp.array[wp.int32],
    target_indices: wp.array[wp.int32],
    stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    target_modes: wp.array[wp.int32],
    effort_limits: wp.array[wp.float32],
    feedforward: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    use_previous_pose: int,
    advance_angle_state: int,
    effort: wp.array[wp.float32],
    coordinate_velocity: wp.array[wp.float32],
):
    drive = wp.tid()
    bodies = drive_bodies[drive]
    child = affine_body_ids[bodies[1]]
    child_pose = body_q[child]
    child_twist = body_qd[child]
    child_origin = wp.transform_point(child_pose, child_origins[drive])
    child_reference = wp.transform_point(child_pose, child_references[drive])
    child_velocity = _point_velocity(child_pose, child_twist, body_com[child], child_origin)

    parent_origin = parent_origins[drive]
    parent_axis_point = parent_axis_points[drive]
    parent_reference = parent_references[drive]
    parent_velocity = wp.vec3(0.0)
    parent_omega = wp.vec3(0.0)
    if bodies[0] >= 0:
        parent = affine_body_ids[bodies[0]]
        parent_pose = body_q[parent]
        parent_twist = body_qd[parent]
        parent_origin = wp.transform_point(parent_pose, parent_origin)
        parent_axis_point = wp.transform_point(parent_pose, parent_axis_point)
        parent_reference = wp.transform_point(parent_pose, parent_reference)
        parent_velocity = _point_velocity(parent_pose, parent_twist, body_com[parent], parent_origin)
        parent_omega = wp.spatial_bottom(parent_twist)

    axis_vector = parent_axis_point - parent_origin
    axis_length = wp.length(axis_vector)
    if axis_length <= 1.0e-8:
        return
    axis = axis_vector / axis_length
    displacement = child_origin - parent_origin
    q = wp.dot(displacement, axis)
    qd = wp.dot(child_velocity - parent_velocity, axis) + wp.dot(displacement, wp.cross(parent_omega, axis))
    dof = dof_indices[drive]
    if drive_types[drive] == 1:
        parent_arm = parent_reference - parent_origin
        child_arm = child_reference - child_origin
        parent_arm = parent_arm - wp.dot(parent_arm, axis) * axis
        child_arm = child_arm - wp.dot(child_arm, axis) * axis
        if wp.length(parent_arm) <= 1.0e-8 or wp.length(child_arm) <= 1.0e-8:
            return
        q = wp.atan2(wp.dot(axis, wp.cross(parent_arm, child_arm)), wp.dot(parent_arm, child_arm))
        # A requested multi-turn target must not redefine the physical angle branch.
        if angle_initialized[drive] != 0:
            q = _unwrap_near(q, previous_angles[drive])
        elif use_previous_pose != 0:
            # A supplied prior pose can seed the angle branch without replacing
            # the current physical twist by a finite-difference velocity.
            previous_child_pose = previous_body_q[child]
            previous_child_arm = wp.transform_vector(
                previous_child_pose, child_references[drive] - child_origins[drive]
            )
            previous_parent_arm = parent_references[drive] - parent_origins[drive]
            previous_axis = parent_axis_points[drive] - parent_origins[drive]
            if bodies[0] >= 0:
                previous_parent_pose = previous_body_q[affine_body_ids[bodies[0]]]
                previous_parent_arm = wp.transform_vector(previous_parent_pose, previous_parent_arm)
                previous_axis = wp.transform_vector(previous_parent_pose, previous_axis)
            previous_axis = wp.normalize(previous_axis)
            previous_parent_arm -= wp.dot(previous_parent_arm, previous_axis) * previous_axis
            previous_child_arm -= wp.dot(previous_child_arm, previous_axis) * previous_axis
            previous_q = wp.atan2(
                wp.dot(previous_axis, wp.cross(previous_parent_arm, previous_child_arm)),
                wp.dot(previous_parent_arm, previous_child_arm),
            )
            q = _unwrap_near(q, previous_q)
        if advance_angle_state != 0:
            previous_angles[drive] = q
            angle_initialized[drive] = 1
        qd = wp.dot(wp.spatial_bottom(child_twist) - parent_omega, axis)

    mode = target_modes[dof]
    actuator = wp.float32(0.0)
    if mode == 1 or mode == 3:
        actuator = stiffness[dof] * (target_q[target_indices[drive]] - q)
    if mode == 1:
        actuator -= damping[dof] * qd
    elif mode == 2 or mode == 3:
        actuator += damping[dof] * (target_qd[dof] - qd)
    # MuJoCo clips the scalar joint's total actuator effort. Applied joint_f
    # and passive damping are separate forces and do not consume this limit.
    limit = effort_limits[dof]
    actuator = wp.clamp(actuator, -limit, limit)
    wp.atomic_add(effort, dof, feedforward[dof] + actuator)
    coordinate_velocity[drive] = qd
