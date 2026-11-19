"""Observation terms for the independent G1 environment."""

from __future__ import annotations

import torch

from .config import ObservationNoiseConfig
from .math_ops import quat_rotate_inverse


def _uniform_noise(
    value: torch.Tensor,
    magnitude: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if magnitude == 0.0:
        return value
    noise = torch.rand(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )
    return value + (2.0 * noise - 1.0) * magnitude


def root_terms(
    root_quaternion: torch.Tensor,
    root_velocity_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quaternion = torch.nn.functional.normalize(root_quaternion, dim=-1)
    linear_velocity = quat_rotate_inverse(quaternion, root_velocity_world[:, :3])
    angular_velocity = quat_rotate_inverse(quaternion, root_velocity_world[:, 3:])
    gravity = torch.zeros_like(linear_velocity)
    gravity[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(quaternion, gravity)
    return linear_velocity, angular_velocity, projected_gravity


def policy_terms(
    angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    joint_position: torch.Tensor,
    default_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    actions: torch.Tensor,
    noise: ObservationNoiseConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    if noise.enabled:
        angular_velocity = _uniform_noise(angular_velocity, noise.base_ang_vel, generator)
        projected_gravity = _uniform_noise(projected_gravity, noise.projected_gravity, generator)
        joint_position = _uniform_noise(joint_position, noise.joint_pos, generator)
        joint_velocity = _uniform_noise(joint_velocity, noise.joint_vel, generator)
    return (
        angular_velocity * 0.2,
        projected_gravity,
        commands,
        joint_position - default_joint_position,
        joint_velocity * 0.05,
        actions,
    )


def critic_terms(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    joint_position: torch.Tensor,
    default_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        linear_velocity,
        angular_velocity * 0.2,
        projected_gravity,
        commands,
        joint_position - default_joint_position,
        joint_velocity * 0.05,
        actions,
    )
