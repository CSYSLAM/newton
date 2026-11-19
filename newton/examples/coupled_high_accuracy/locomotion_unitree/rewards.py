"""Reward terms matching Unitree RL Lab's active Go2 configuration."""

from __future__ import annotations

import torch


def unitree_go2_reward(
    *,
    commands: torch.Tensor,
    linear_velocity_body: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    gravity_body: torch.Tensor,
    joint_position_error: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_acceleration: torch.Tensor,
    joint_torque: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    soft_joint_lower: torch.Tensor,
    soft_joint_upper: torch.Tensor,
    joint_position: torch.Tensor,
    first_foot_contact: torch.Tensor,
    last_air_time: torch.Tensor,
    last_contact_time: torch.Tensor,
    foot_contact: torch.Tensor,
    foot_velocity_xy: torch.Tensor,
    undesired_contact: torch.Tensor,
    weights: dict[str, float],
    step_dt: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    command_norm = torch.linalg.vector_norm(commands, dim=-1)
    moving = torch.logical_or(
        command_norm > 0.0,
        torch.linalg.vector_norm(linear_velocity_body[:, :2], dim=-1) > 0.3,
    )
    joint_position_norm = torch.linalg.vector_norm(joint_position_error, dim=-1)
    position_limit = (
        (soft_joint_lower - joint_position).clamp_min(0.0) + (joint_position - soft_joint_upper).clamp_min(0.0)
    ).sum(-1)
    raw = {
        "track_lin_vel_xy": torch.exp(-(commands[:, :2] - linear_velocity_body[:, :2]).square().sum(-1) / 0.25),
        "track_ang_vel_z": torch.exp(-(commands[:, 2] - angular_velocity_body[:, 2]).square() / 0.25),
        "lin_vel_z": linear_velocity_body[:, 2].square(),
        "ang_vel_xy": angular_velocity_body[:, :2].square().sum(-1),
        "joint_vel": joint_velocity.square().sum(-1),
        "joint_acc": joint_acceleration.square().sum(-1),
        "joint_torques": joint_torque.square().sum(-1),
        "action_rate": (actions - previous_actions).square().sum(-1),
        "joint_pos_limits": position_limit,
        "energy": (joint_velocity.abs() * joint_torque.abs()).sum(-1),
        "flat_orientation": gravity_body[:, :2].square().sum(-1),
        "joint_pos": torch.where(moving, joint_position_norm, 5.0 * joint_position_norm),
        "feet_air_time": ((last_air_time - 0.5) * first_foot_contact.float()).sum(-1) * (command_norm > 0.1),
        "air_time_variance": torch.var(last_air_time.clamp_max(0.5), dim=1)
        + torch.var(last_contact_time.clamp_max(0.5), dim=1),
        "feet_slide": (torch.linalg.vector_norm(foot_velocity_xy, dim=-1) * foot_contact.float()).sum(-1),
        "undesired_contacts": undesired_contact.float().sum(-1),
    }
    contributions = {name: value * float(weights[name]) * step_dt for name, value in raw.items()}
    return sum(contributions.values()), contributions
