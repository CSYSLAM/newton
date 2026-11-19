"""Deployment-exact actor observations and history buffering."""

from __future__ import annotations

import torch

from .policy_spec import PolicySpec


class ObservationHistory:
    """IsaacLab-compatible term-major history, oldest sample first per term."""

    def __init__(self, num_envs: int, spec: PolicySpec, device: torch.device) -> None:
        self.spec = spec
        self.buffer = torch.zeros(
            num_envs,
            spec.history_length,
            spec.frame_observation_dim,
            device=device,
        )

    def reset(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        self.buffer[env_ids] = frame[env_ids].unsqueeze(1)

    def append(self, frame: torch.Tensor) -> torch.Tensor:
        if self.spec.history_length > 1:
            self.buffer[:, :-1] = self.buffer[:, 1:].clone()
        self.buffer[:, -1] = frame
        return self.value()

    def value(self) -> torch.Tensor:
        # IsaacLab gives each observation term its own circular buffer, flattens
        # its time dimension, then concatenates terms.  G1 deployment uses the
        # same order unless deploy.yaml explicitly opts into gym history.
        widths = (3, 3, 3, self.spec.action_dim, self.spec.action_dim, self.spec.action_dim)
        values = []
        start = 0
        for width in widths:
            values.append(self.buffer[:, :, start : start + width].flatten(1))
            start += width
        return torch.cat(values, dim=-1)


def actor_frame(
    spec: PolicySpec,
    angular_velocity_body: torch.Tensor,
    gravity_body: torch.Tensor,
    commands: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
    default_joint_position: torch.Tensor,
    *,
    noise: dict[str, float] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build one 45D Go2 or 96D G1 frame in deployment term order."""

    values = {
        "ang_vel": angular_velocity_body * spec.ang_vel_scale,
        "gravity": gravity_body,
        "joint_pos": joint_position - default_joint_position,
        "joint_vel": joint_velocity * spec.joint_vel_scale,
    }
    if noise:
        for name, tensor in values.items():
            amplitude = float(noise.get(name, 0.0))
            if amplitude:
                tensor = tensor + amplitude * (
                    2.0
                    * torch.rand(
                        tensor.shape,
                        device=tensor.device,
                        generator=generator,
                    )
                    - 1.0
                )
                values[name] = tensor
    frame = torch.cat(
        (
            values["ang_vel"],
            values["gravity"],
            commands,
            values["joint_pos"],
            values["joint_vel"],
            previous_action,
        ),
        dim=-1,
    ).clamp(-100.0, 100.0)
    if frame.shape[-1] != spec.frame_observation_dim:
        raise RuntimeError(f"observation ABI mismatch: {frame.shape[-1]} != {spec.frame_observation_dim}")
    return frame
