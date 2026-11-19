"""Per-environment actuator, sensor, latency, and push randomization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .policy_spec import PolicySpec


def uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low, high = bounds
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


@dataclass
class EpisodeRandomization:
    stiffness_scale: torch.Tensor
    damping_scale: torch.Tensor
    motor_strength: torch.Tensor
    joint_zero_offset: torch.Tensor
    delay_ticks: torch.Tensor
    next_push_step: torch.Tensor

    @classmethod
    def create(
        cls,
        spec: PolicySpec,
        num_envs: int,
        device: torch.device,
        generator: torch.Generator,
        enabled: bool,
    ) -> EpisodeRandomization:
        ones = torch.ones((num_envs, spec.action_dim), device=device)
        zeros = torch.zeros((num_envs, spec.action_dim), device=device)
        if enabled:
            stiffness = uniform(ones.shape, spec.randomization.stiffness_scale, device, generator)
            damping = uniform(ones.shape, spec.randomization.damping_scale, device, generator)
            strength = uniform(ones.shape, spec.randomization.motor_strength, device, generator)
            zero = uniform(zeros.shape, spec.randomization.joint_zero_offset, device, generator)
            delay = torch.randint(
                spec.randomization.policy_delay_ticks[0],
                spec.randomization.policy_delay_ticks[1] + 1,
                (num_envs,),
                device=device,
                generator=generator,
            )
        else:
            stiffness, damping, strength, zero = ones, ones.clone(), ones.clone(), zeros
            delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        next_push = torch.zeros(num_envs, dtype=torch.long, device=device)
        result = cls(stiffness, damping, strength, zero, delay, next_push)
        result.resample_push(spec, torch.arange(num_envs, device=device), generator)
        return result

    def resample_push(
        self,
        spec: PolicySpec,
        env_ids: torch.Tensor,
        generator: torch.Generator,
    ) -> None:
        seconds = uniform(
            (len(env_ids),),
            spec.randomization.push_interval_s,
            env_ids.device,
            generator,
        )
        self.next_push_step[env_ids] = (seconds / spec.policy_dt).long()
