"""Unitree RL Lab G1 rewards with explicit geometric-contact substitutes."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from .config import RewardConfig
from .math_ops import yaw_rotate_inverse


@dataclass
class RewardInputs:
    commands: torch.Tensor
    root_position: torch.Tensor
    root_quaternion: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    root_linear_velocity_body: torch.Tensor
    root_angular_velocity_body: torch.Tensor
    projected_gravity: torch.Tensor
    joint_position: torch.Tensor
    default_joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    previous_joint_velocity: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
    estimated_torque: torch.Tensor
    soft_joint_lower: torch.Tensor
    soft_joint_upper: torch.Tensor
    foot_position_world: torch.Tensor
    foot_linear_velocity_world: torch.Tensor
    foot_contact: torch.Tensor
    recent_foot_contact: torch.Tensor
    recent_undesired_contact: torch.Tensor
    foot_duty_imbalance: torch.Tensor
    repeated_touchdown: torch.Tensor
    episode_steps: torch.Tensor
    terminated: torch.Tensor


def compute_reward(
    inputs: RewardInputs,
    config: RewardConfig,
    policy_dt: float,
    arm_indices: torch.Tensor,
    waist_indices: torch.Tensor,
    leg_deviation_indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted per-step rewards, including IsaacLab's dt scaling."""

    yaw_linear_velocity = yaw_rotate_inverse(inputs.root_quaternion, inputs.root_linear_velocity_world)
    linear_error = torch.sum(torch.square(inputs.commands[:, :2] - yaw_linear_velocity[:, :2]), dim=-1)
    angular_error = torch.square(inputs.commands[:, 2] - inputs.root_angular_velocity_body[:, 2])
    joint_error = inputs.joint_position - inputs.default_joint_position
    joint_acceleration = (inputs.joint_velocity - inputs.previous_joint_velocity) / policy_dt
    lower_violation = (inputs.soft_joint_lower - inputs.joint_position).clamp_min(0.0)
    upper_violation = (inputs.joint_position - inputs.soft_joint_upper).clamp_min(0.0)

    phase = (inputs.episode_steps.float() * policy_dt / config.gait_period).remainder(1.0)
    offsets = torch.tensor(
        config.gait_offsets,
        dtype=phase.dtype,
        device=phase.device,
    )
    stance = (phase.unsqueeze(-1) + offsets).remainder(1.0) < config.gait_stance_ratio
    moving = torch.linalg.vector_norm(inputs.commands, dim=-1) > 0.1
    gait = torch.sum(stance == inputs.foot_contact, dim=-1).float() * moving

    foot_xy_speed = torch.linalg.vector_norm(inputs.foot_linear_velocity_world[..., :2], dim=-1)
    clearance_error = torch.square(inputs.foot_position_world[..., 2] - config.foot_clearance_target)
    clearance_motion = torch.tanh(config.foot_clearance_tanh_multiplier * foot_xy_speed)
    if config.supported_swing_clearance:
        opposite_contact = inputs.foot_contact.flip(dims=(1,))
        supported_swing = (~inputs.foot_contact) & opposite_contact
        height_score = torch.exp(-clearance_error / (config.foot_clearance_std * config.foot_clearance_std))
        clearance = torch.sum(height_score * clearance_motion * supported_swing.float(), dim=-1) * moving
    else:
        clearance = torch.exp(-torch.sum(clearance_error * clearance_motion, dim=-1) / config.foot_clearance_std)

    raw = {
        "track_lin_vel_xy": torch.exp(-linear_error / (config.tracking_std * config.tracking_std)),
        "track_ang_vel_z": torch.exp(-angular_error / (config.tracking_std * config.tracking_std)),
        "alive": (~inputs.terminated).float(),
        "lin_vel_z_l2": torch.square(inputs.root_linear_velocity_body[:, 2]),
        "ang_vel_xy_l2": torch.sum(torch.square(inputs.root_angular_velocity_body[:, :2]), dim=-1),
        "joint_vel_l2": torch.sum(torch.square(inputs.joint_velocity), dim=-1),
        "joint_acc_l2": torch.sum(torch.square(joint_acceleration), dim=-1),
        "action_rate_l2": torch.sum(torch.square(inputs.actions - inputs.previous_actions), dim=-1),
        "joint_pos_limits": torch.sum(lower_violation + upper_violation, dim=-1),
        "energy": torch.sum(
            torch.abs(inputs.joint_velocity) * torch.abs(inputs.estimated_torque),
            dim=-1,
        ),
        "joint_deviation_arms": torch.sum(torch.abs(joint_error.index_select(1, arm_indices)), dim=-1),
        "joint_deviation_waists": torch.sum(torch.abs(joint_error.index_select(1, waist_indices)), dim=-1),
        "joint_deviation_legs": torch.sum(torch.abs(joint_error.index_select(1, leg_deviation_indices)), dim=-1),
        "flat_orientation": torch.sum(torch.square(inputs.projected_gravity[:, :2]), dim=-1),
        "base_height": torch.square(inputs.root_position[:, 2] - config.base_height_target),
        "gait": gait,
        "feet_slide": torch.sum(foot_xy_speed * inputs.recent_foot_contact.float(), dim=-1),
        "feet_clearance": clearance,
        "undesired_contacts": torch.sum(inputs.recent_undesired_contact, dim=-1).float(),
        "action_l2": torch.sum(torch.square(inputs.actions), dim=-1),
        "arm_action_l2": torch.sum(torch.square(inputs.actions.index_select(1, arm_indices)), dim=-1),
        "lateral_vel_error_l2": torch.square(inputs.commands[:, 1] - yaw_linear_velocity[:, 1]),
        "yaw_rate_error_l2": angular_error,
        "foot_duty_imbalance": inputs.foot_duty_imbalance,
        "repeated_touchdown": inputs.repeated_touchdown,
    }
    names = {item.name for item in fields(config)}
    weighted = {name: value * float(getattr(config, name)) * policy_dt for name, value in raw.items() if name in names}
    total = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
    return total, weighted
