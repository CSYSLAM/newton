"""GPU reward and termination functions for flat-ground locomotion."""

from __future__ import annotations

import torch


def locomotion_reward(
    *,
    robot: str,
    commands: torch.Tensor,
    linear_velocity_body: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    gravity_body: torch.Tensor,
    joint_position_error: torch.Tensor,
    joint_velocity: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    root_height: torch.Tensor,
    base_height_target: float,
    base_height_sigma: float,
    base_height_reward_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    linear_error = (commands[:, :2] - linear_velocity_body[:, :2]).square().sum(-1)
    yaw_error = (commands[:, 2] - angular_velocity_body[:, 2]).square()
    linear_tracking_sigma = 0.25
    track_linear = torch.exp(-linear_error / linear_tracking_sigma)
    track_yaw = torch.exp(-yaw_error / 0.25)
    upright = (-gravity_body[:, 2]).clamp(-1.0, 1.0)
    vertical = linear_velocity_body[:, 2].square()
    roll_pitch_rate = angular_velocity_body[:, :2].square().sum(-1)
    action_rate = (actions - previous_actions).square().sum(-1)
    action_magnitude = actions.square().sum(-1)
    joint_motion = joint_velocity.square().sum(-1)
    posture = joint_position_error.square().sum(-1)
    posture_deviation = joint_position_error.abs().sum(-1)
    base_height_error = (root_height - base_height_target).square()
    height_error = (root_height - base_height_target) / base_height_sigma
    height_reward = torch.exp(-height_error.square())

    if robot == "go2":
        # Keep locomotion progress and posture costs additive. Multiplying the
        # tracking reward by a stability gate makes the first imperfect steps
        # unrewarding and leaves standing as an easy local optimum.
        reward = (
            3.0 * track_linear
            + 1.0 * track_yaw
            + 0.5 * upright.clamp(0.0, 1.0)
            + base_height_reward_scale * height_reward
            - 1.0 * vertical
            - 0.005 * action_rate
            - 0.1 * posture_deviation
            - 50.0 * base_height_reward_scale * base_height_error
        )
    else:
        reward = (
            3.0 * track_linear
            + 1.0 * track_yaw
            + upright
            - 0.2 * vertical
            - 0.1 * roll_pitch_rate
            - 0.005 * action_rate
            - 1.0e-7 * joint_motion
            - 0.01 * posture
        )
    terms = {
        "track_linear": track_linear,
        "track_yaw": track_yaw,
        "upright": upright,
        "height": height_reward,
        "base_height_error": base_height_error,
        "action_rate": action_rate,
        "action_magnitude": action_magnitude,
    }
    return reward, terms


def terminated(
    root_quaternion: torch.Tensor,
    gravity_body: torch.Tensor,
    roll_limit_degrees: float | None,
    pitch_limit_degrees: float | None,
    low_height_confirmed: torch.Tensor,
    finite: torch.Tensor,
) -> torch.Tensor:
    angle_failure = torch.zeros_like(finite)
    if roll_limit_degrees is not None and pitch_limit_degrees is not None:
        x, y, z, w = root_quaternion.unbind(-1)
        roll = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x.square() + y.square()),
        )
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        angle_failure = (torch.abs(torch.rad2deg(roll)) > roll_limit_degrees) | (
            torch.abs(torch.rad2deg(pitch)) > pitch_limit_degrees
        )
    else:
        angle_failure = (-gravity_body[:, 2]) < 0.45
    return angle_failure | low_height_confirmed | ~finite
