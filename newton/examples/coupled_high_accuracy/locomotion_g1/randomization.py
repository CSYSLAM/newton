"""Per-environment actuator, calibration, latency, and push randomization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import RandomizationConfig


def uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low, high = bounds
    if high < low:
        raise ValueError(f"invalid randomization range {bounds}")
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


def delayed_action(queue: torch.Tensor, delay_ticks: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Insert raw actions and select each environment's delayed control sample."""

    queue[:, 1:] = queue[:, :-1].clone()
    queue[:, 0] = actions
    rows = torch.arange(queue.shape[0], device=queue.device)
    return queue[rows, delay_ticks]


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
        config: RandomizationConfig,
        num_envs: int,
        num_actions: int,
        policy_dt: float,
        device: torch.device,
        generator: torch.Generator,
    ) -> EpisodeRandomization:
        result = cls(
            stiffness_scale=torch.ones((num_envs, num_actions), device=device),
            damping_scale=torch.ones((num_envs, num_actions), device=device),
            motor_strength=torch.ones((num_envs, num_actions), device=device),
            joint_zero_offset=torch.zeros((num_envs, num_actions), device=device),
            delay_ticks=torch.zeros(num_envs, dtype=torch.long, device=device),
            next_push_step=torch.zeros(num_envs, dtype=torch.long, device=device),
        )
        result.resample(
            config,
            torch.arange(num_envs, device=device),
            policy_dt,
            generator,
        )
        return result

    def resample(
        self,
        config: RandomizationConfig,
        env_ids: torch.Tensor,
        policy_dt: float,
        generator: torch.Generator,
    ) -> None:
        if env_ids.numel() == 0:
            return
        shape = (env_ids.numel(), self.stiffness_scale.shape[1])
        if config.actuator_enabled:
            positive_ranges = (
                config.stiffness_scale,
                config.damping_scale,
                config.motor_strength,
            )
            if any(low <= 0.0 or high < low for low, high in positive_ranges):
                raise ValueError("actuator scale ranges must be positive and ordered")
            self.stiffness_scale[env_ids] = uniform(shape, config.stiffness_scale, env_ids.device, generator)
            self.damping_scale[env_ids] = uniform(shape, config.damping_scale, env_ids.device, generator)
            self.motor_strength[env_ids] = uniform(shape, config.motor_strength, env_ids.device, generator)
            self.joint_zero_offset[env_ids] = uniform(shape, config.joint_zero_offset, env_ids.device, generator)
            low, high = config.policy_delay_ticks
            if low < 0 or high < low:
                raise ValueError("policy_delay_ticks must be non-negative and ordered")
            self.delay_ticks[env_ids] = torch.randint(
                low,
                high + 1,
                (env_ids.numel(),),
                device=env_ids.device,
                generator=generator,
            )
        else:
            self.stiffness_scale[env_ids] = 1.0
            self.damping_scale[env_ids] = 1.0
            self.motor_strength[env_ids] = 1.0
            self.joint_zero_offset[env_ids] = 0.0
            self.delay_ticks[env_ids] = 0
        self.resample_push(config, env_ids, policy_dt, generator, current_step=0)

    def resample_push(
        self,
        config: RandomizationConfig,
        env_ids: torch.Tensor,
        policy_dt: float,
        generator: torch.Generator,
        *,
        current_step: int | torch.Tensor,
    ) -> None:
        if not config.push_enabled:
            self.next_push_step[env_ids] = torch.iinfo(torch.long).max
            return
        if config.push_interval_s[0] <= 0.0:
            raise ValueError("push intervals must be positive")
        seconds = uniform((env_ids.numel(),), config.push_interval_s, env_ids.device, generator)
        interval = (seconds / policy_dt).long().clamp_min(1)
        self.next_push_step[env_ids] = interval + current_step
