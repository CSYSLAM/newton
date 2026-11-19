"""Capture-safe Warp kernels used by the locomotion environment."""

from __future__ import annotations

import warp as wp


@wp.kernel
def update_limited_joint_targets(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    desired_targets: wp.array2d[wp.float32],
    base_stiffness: wp.array[wp.float32],
    base_damping: wp.array[wp.float32],
    base_effort: wp.array[wp.float32],
    stiffness_scale: wp.array2d[wp.float32],
    damping_scale: wp.array2d[wp.float32],
    motor_strength: wp.array2d[wp.float32],
    policy_q: wp.array[wp.int64],
    policy_qd: wp.array[wp.int64],
    policy_targets: wp.array[wp.int64],
    action_count: int,
    q_per_env: int,
    qd_per_env: int,
    target_per_env: int,
    control_targets: wp.array[wp.float32],
) -> None:
    index = wp.tid()
    env = index // action_count
    action = index - env * action_count

    q = joint_q[env * q_per_env + wp.int32(policy_q[action])]
    qd = joint_qd[env * qd_per_env + wp.int32(policy_qd[action])]
    kp = base_stiffness[action] * stiffness_scale[env, action]
    kd = base_damping[action] * damping_scale[env, action]
    limit = base_effort[action] * motor_strength[env, action]
    torque = wp.clamp(kp * (desired_targets[env, action] - q) - kd * qd, -limit, limit)
    limited_target = q + (torque + kd * qd) / wp.max(kp, 1.0e-6)
    control_targets[env * target_per_env + wp.int32(policy_targets[action])] = limited_target
