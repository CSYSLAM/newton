#!/usr/bin/env python3
"""Play a Unitree Go2 checkpoint in a single visible SynReal world."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    from locomotion_unitree.config import load_config
    from locomotion_unitree.env import UnitreeGo2Env
    from locomotion_unitree.train import rsl_rl_config
else:
    from .config import load_config
    from .env import UnitreeGo2Env
    from .train import rsl_rl_config


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=0, help="0 runs until closed")
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.frames < 0:
        raise SystemExit("--frames must be non-negative")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as error:
        raise SystemExit("install locomotion dependencies with `uv sync --extra locomotion`") from error
    import newton

    spec, _ = load_config()
    viewer = newton.viewer.ViewerGL(width=1280, height=800, vsync=True)
    env = UnitreeGo2Env(
        num_envs=1,
        device=args.device,
        seed=42,
        randomize=not args.no_randomization,
        auto_reset=False,
        viewer=viewer,
        use_cuda_graph=not args.disable_cuda_graph,
        command_vx=args.command_vx,
        friction_randomization="fixed",
    )
    runner = OnPolicyRunner(env, rsl_rl_config(), log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    env.command_ranges["lin_vel_y"] = [args.command_vy, args.command_vy]
    env.command_ranges["ang_vel_z"] = [args.command_yaw, args.command_yaw]
    observation = env.reset()
    velocity_samples: list[float] = []
    try:
        frame = 0
        while args.frames == 0 or frame < args.frames:
            if not viewer.is_running():
                break
            started = time.perf_counter()
            with torch.no_grad():
                actions = policy(observation)
            observation, _, done, _ = env.step(actions)
            _, linear, _, _, _, _ = env._state()
            vx = float(linear[0, 0])
            velocity_samples.append(vx)
            if frame % 100 == 0:
                print(
                    f"frame={frame:4d} command_vx={args.command_vx:+.2f} "
                    f"actual_vx={vx:+.3f} m/s "
                    f"error={vx - args.command_vx:+.3f} done={bool(done[0])}",
                    flush=True,
                )
            if bool(done[0]):
                observation = env.reset()
            frame += 1
            delay = spec.policy_dt - (time.perf_counter() - started)
            if delay > 0.0:
                time.sleep(delay)
    finally:
        if velocity_samples:
            window = velocity_samples[-min(200, len(velocity_samples)) :]
            print(
                f"final actual_vx={velocity_samples[-1]:+.3f} m/s, "
                f"last_{len(window)}_mean_vx={sum(window) / len(window):+.3f} m/s, "
                f"command_vx={args.command_vx:+.3f} m/s",
                flush=True,
            )
        env.close()


if __name__ == "__main__":
    main()
