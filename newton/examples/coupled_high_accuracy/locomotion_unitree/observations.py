"""Unitree RL Lab Go2 actor observations."""

from __future__ import annotations

import torch


def _uniform_noise(
    value: torch.Tensor,
    amplitude: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if amplitude == 0.0:
        return value
    return value + amplitude * (2.0 * torch.rand(value.shape, device=value.device, generator=generator) - 1.0)


def actor_observation(
    *,
    angular_velocity_body: torch.Tensor,
    gravity_body: torch.Tensor,
    commands: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    last_action: torch.Tensor,
    default_joint_position: torch.Tensor,
    corrupt: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    angular = angular_velocity_body
    gravity = gravity_body
    position = joint_position - default_joint_position
    velocity = joint_velocity
    if corrupt:
        angular = _uniform_noise(angular, 0.2, generator)
        gravity = _uniform_noise(gravity, 0.05, generator)
        position = _uniform_noise(position, 0.01, generator)
        velocity = _uniform_noise(velocity, 1.5, generator)
    return torch.cat(
        (
            0.2 * angular,
            gravity,
            commands,
            position,
            0.05 * velocity,
            last_action,
        ),
        dim=-1,
    ).clamp(-100.0, 100.0)


def critic_observation(
    *,
    linear_velocity_body: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    gravity_body: torch.Tensor,
    commands: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_effort: torch.Tensor,
    last_action: torch.Tensor,
    default_joint_position: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            linear_velocity_body,
            0.2 * angular_velocity_body,
            gravity_body,
            commands,
            joint_position - default_joint_position,
            0.05 * joint_velocity,
            0.01 * joint_effort,
            last_action,
        ),
        dim=-1,
    ).clamp(-100.0, 100.0)
