#!/usr/bin/env python3
"""Play a G1 locomotion checkpoint or inspect the zero-action model."""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path

import torch

if __package__ in (None, ""):
    from locomotion_g1.config import (
        G1EnvConfig,
        ObservationNoiseConfig,
        make_effort_bootstrap_config,
        make_ppo_config,
    )
    from locomotion_g1.env import SynRealG1LocomotionEnv
else:
    from .config import (
        G1EnvConfig,
        ObservationNoiseConfig,
        make_effort_bootstrap_config,
        make_ppo_config,
    )
    from .env import SynRealG1LocomotionEnv


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=Path)
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="rollout duration; use 0 for a viewer that runs until manually closed",
    )
    parser.add_argument("--cmd-x", type=float, default=0.3)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable-noise", action="store_true")
    parser.add_argument(
        "--effort-bootstrap",
        action="store_true",
        help="reproduce the nominal effort-limited bootstrap control configuration",
    )
    parser.add_argument(
        "--enable-periodic-push",
        action="store_true",
        help="enable the five-second random velocity pushes for robustness testing",
    )
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    return parser


def make_fixed_command_config(config: G1EnvConfig, x: float, y: float, yaw: float) -> G1EnvConfig:
    """Keep evaluation commands unchanged across timed resampling and resets."""

    return replace(
        config,
        commands=replace(
            config.commands,
            initial_lin_vel_x=(x, x),
            initial_lin_vel_y=(y, y),
            initial_ang_vel_z=(yaw, yaw),
            standing_fraction=0.0,
            curriculum_enabled=False,
        ),
    )


def make_play_config(
    config: G1EnvConfig,
    x: float,
    y: float,
    yaw: float,
    *,
    effort_bootstrap: bool = False,
) -> G1EnvConfig:
    """Apply the checkpoint control preset before fixing evaluation commands."""

    if effort_bootstrap:
        config = make_effort_bootstrap_config(config)
    return make_fixed_command_config(config, x, y, yaw)


def main() -> None:
    args = create_parser().parse_args()
    if args.num_envs <= 0 or args.seconds < 0.0:
        raise SystemExit("--num-envs must be positive and --seconds must be non-negative")
    if args.headless and args.seconds == 0.0:
        raise SystemExit("--seconds 0 requires a viewer")
    viewer = None
    if not args.headless:
        import newton

        viewer = newton.viewer.ViewerGL(vsync=True)
    config = make_play_config(
        G1EnvConfig(
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            asset_path=args.asset,
            enable_self_collisions=not args.disable_self_collisions,
            use_cuda_graph=not args.disable_cuda_graph,
            periodic_push_enabled=args.enable_periodic_push,
            observation_noise=ObservationNoiseConfig(enabled=args.enable_noise),
        ),
        args.cmd_x,
        args.cmd_y,
        args.cmd_yaw,
        effort_bootstrap=args.effort_bootstrap,
    )
    env = SynRealG1LocomotionEnv(config, viewer=viewer)
    policy = None
    if args.checkpoint is not None:
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as error:
            env.close()
            raise SystemExit("RSL-RL is not installed. Run `uv sync --extra locomotion`.") from error
        runner = OnPolicyRunner(env, make_ppo_config(1), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint.expanduser().resolve()))
        policy = runner.get_inference_policy(device=args.device)

    steps = itertools.count() if args.seconds == 0.0 else range(round(args.seconds / env.step_dt))
    observation, _ = env.get_observations()
    try:
        for _ in steps:
            start = time.perf_counter()
            with torch.inference_mode():
                actions = (
                    torch.zeros((env.num_envs, env.num_actions), device=env.device)
                    if policy is None
                    else policy(observation)
                )
                observation, _, _, _ = env.step(actions)
            if viewer is not None:
                delay = env.step_dt - (time.perf_counter() - start)
                if delay > 0.0:
                    time.sleep(delay)
                if not viewer.is_running():
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
