"""Capture-safe actuator and contact aggregation kernels."""

from __future__ import annotations

import warp as wp


@wp.kernel
def update_unitree_joint_targets(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    desired_targets: wp.array2d[wp.float32],
    policy_q: wp.array[wp.int64],
    policy_qd: wp.array[wp.int64],
    policy_targets: wp.array[wp.int64],
    action_count: int,
    q_per_env: int,
    qd_per_env: int,
    target_per_env: int,
    kp: float,
    kd: float,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    control_targets: wp.array[wp.float32],
    applied_torque: wp.array2d[wp.float32],
) -> None:
    index = wp.tid()
    env = index // action_count
    action = index - env * action_count
    q = joint_q[env * q_per_env + wp.int32(policy_q[action])]
    qd = joint_qd[env * qd_per_env + wp.int32(policy_qd[action])]
    desired = kp * (desired_targets[env, action] - q) - kd * qd
    limit = y2
    if qd * desired > 0.0:
        limit = y1
    speed = wp.abs(qd)
    if speed >= x1:
        limit = limit * wp.clamp((x2 - speed) / (x2 - x1), 0.0, 1.0)
    torque = wp.clamp(desired, -limit, limit)
    control_targets[env * target_per_env + wp.int32(policy_targets[action])] = q + (torque + kd * qd) / kp
    applied_torque[env, action] = torque


@wp.kernel
def clear_policy_contacts(
    foot_contact_any: wp.array[wp.int32],
    first_foot_contact: wp.array[wp.int32],
    undesired_contact: wp.array[wp.int32],
    base_contact: wp.array[wp.int32],
) -> None:
    index = wp.tid()
    if index < foot_contact_any.shape[0]:
        foot_contact_any[index] = 0
        first_foot_contact[index] = 0
    if index < undesired_contact.shape[0]:
        undesired_contact[index] = 0
    if index < base_contact.shape[0]:
        base_contact[index] = 0


@wp.kernel
def clear_substep_contacts(foot_contact: wp.array[wp.int32]) -> None:
    foot_contact[wp.tid()] = 0


@wp.func
def _world_point(
    point: wp.vec3,
    body: int,
    body_q: wp.array[wp.transform],
) -> wp.vec3:
    result = point
    if body >= 0:
        result = wp.transform_point(body_q[body], point)
    return result


@wp.func
def _classify_shape(
    shape: int,
    shape_to_foot: wp.array[wp.int32],
    shape_to_undesired: wp.array[wp.int32],
    shape_to_base_env: wp.array[wp.int32],
    foot_contact: wp.array[wp.int32],
    foot_contact_any: wp.array[wp.int32],
    undesired_contact: wp.array[wp.int32],
    base_contact: wp.array[wp.int32],
):
    if shape >= 0 and shape < shape_to_foot.shape[0]:
        foot = shape_to_foot[shape]
        if foot >= 0:
            wp.atomic_max(foot_contact, foot, 1)
            wp.atomic_max(foot_contact_any, foot, 1)
        else:
            undesired = shape_to_undesired[shape]
            if undesired >= 0:
                wp.atomic_max(undesired_contact, undesired, 1)
        env = shape_to_base_env[shape]
        if env >= 0:
            wp.atomic_max(base_contact, env, 1)


@wp.kernel
def aggregate_rigid_contacts(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[wp.float32],
    margin1: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    shape_to_foot: wp.array[wp.int32],
    shape_to_undesired: wp.array[wp.int32],
    shape_to_base_env: wp.array[wp.int32],
    foot_contact: wp.array[wp.int32],
    foot_contact_any: wp.array[wp.int32],
    undesired_contact: wp.array[wp.int32],
    base_contact: wp.array[wp.int32],
) -> None:
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    first = shape0[contact]
    second = shape1[contact]
    if first < 0 or second < 0:
        return
    body0 = shape_body[first]
    body1 = shape_body[second]
    world0 = _world_point(point0[contact], body0, body_q)
    world1 = _world_point(point1[contact], body1, body_q)
    separation = wp.dot(world1 - world0, normal[contact])
    if separation >= margin0[contact] + margin1[contact]:
        return
    _classify_shape(
        first,
        shape_to_foot,
        shape_to_undesired,
        shape_to_base_env,
        foot_contact,
        foot_contact_any,
        undesired_contact,
        base_contact,
    )
    _classify_shape(
        second,
        shape_to_foot,
        shape_to_undesired,
        shape_to_base_env,
        foot_contact,
        foot_contact_any,
        undesired_contact,
        base_contact,
    )


@wp.kernel
def update_contact_timers(
    foot_contact: wp.array[wp.int32],
    dt: float,
    previous_foot_contact: wp.array[wp.int32],
    first_foot_contact: wp.array[wp.int32],
    current_air_time: wp.array[wp.float32],
    current_contact_time: wp.array[wp.float32],
    last_air_time: wp.array[wp.float32],
    last_contact_time: wp.array[wp.float32],
) -> None:
    foot = wp.tid()
    contact = foot_contact[foot] != 0
    previous = previous_foot_contact[foot] != 0
    if contact:
        if not previous:
            last_air_time[foot] = current_air_time[foot]
            first_foot_contact[foot] = 1
            current_contact_time[foot] = dt
        else:
            current_contact_time[foot] = current_contact_time[foot] + dt
        current_air_time[foot] = 0.0
    else:
        if previous:
            last_contact_time[foot] = current_contact_time[foot]
            current_air_time[foot] = dt
        else:
            current_air_time[foot] = current_air_time[foot] + dt
        current_contact_time[foot] = 0.0
    previous_foot_contact[foot] = foot_contact[foot]
