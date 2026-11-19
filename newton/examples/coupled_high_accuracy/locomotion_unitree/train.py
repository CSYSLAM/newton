#!/usr/bin/env python3
"""Train Unitree RL Lab's Go2 velocity task with SynReal/Newton."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

if __package__ in (None, ""):
    from locomotion_unitree.config import CONFIG_PATH, load_config
    from locomotion_unitree.env import UnitreeGo2Env
else:
    from .config import CONFIG_PATH, load_config
    from .env import UnitreeGo2Env


def rsl_rl_config() -> dict:
    spec, _ = load_config()
    ppo = spec.ppo
    return {
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "experiment_name": "synreal_unitree_go2",
        "empirical_normalization": False,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": float(ppo["init_noise_std"]),
            "noise_std_type": "log",
            "actor_hidden_dims": list(ppo["actor_hidden_dims"]),
            "critic_hidden_dims": list(ppo["critic_hidden_dims"]),
            "activation": str(ppo["activation"]),
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": float(ppo["clip_param"]),
            "entropy_coef": float(ppo["entropy_coef"]),
            "num_learning_epochs": int(ppo["num_learning_epochs"]),
            "num_mini_batches": int(ppo["num_mini_batches"]),
            "learning_rate": float(ppo["learning_rate"]),
            "schedule": str(ppo["schedule"]),
            "gamma": float(ppo["gamma"]),
            "lam": float(ppo["lam"]),
            "desired_kl": float(ppo["desired_kl"]),
            "max_grad_norm": 1.0,
        },
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--command-vx", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument(
        "--friction-randomization",
        choices=("per-env", "batch", "fixed"),
        default="batch",
    )
    parser.add_argument("--enable-self-collisions", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    spec, task = load_config()
    cfg = rsl_rl_config()
    if args.iterations is not None:
        cfg["max_iterations"] = args.iterations
    summary = {
        "source": f"Unitree RL Lab {task['source_commit']}",
        "mass_randomization": False,
        "actor_observation_dim": spec.observation_dim,
        "critic_observation_dim": 60,
        "action_dim": spec.action_dim,
        "num_envs": args.num_envs,
        "friction_randomization": args.friction_randomization,
        "self_collisions": args.enable_self_collisions,
        "policy_dt": spec.policy_dt,
        "rsl_rl": cfg,
    }
    print(json.dumps(summary, indent=2))
    if not args.run:
        print("configuration valid; pass --run to start PPO")
        return

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as error:
        raise SystemExit("install locomotion dependencies with `uv sync --extra locomotion`") from error

    viewer = None
    if args.visualize:
        import newton

        viewer = newton.viewer.ViewerGL(vsync=True)
    env = UnitreeGo2Env(
        num_envs=args.num_envs,
        device=args.device,
        asset=args.asset,
        seed=args.seed,
        randomize=not args.no_randomization,
        viewer=viewer,
        use_cuda_graph=not args.disable_cuda_graph,
        command_vx=args.command_vx,
        friction_randomization=args.friction_randomization,
        enable_self_collisions=args.enable_self_collisions,
    )
    log_dir = None
    if not args.disable_logs:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = args.log_root / cfg["experiment_name"] / run_name
        log_dir.mkdir(parents=True, exist_ok=False)
        (log_dir / "task.yaml").write_bytes(CONFIG_PATH.read_bytes())
        (log_dir / "agent.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    runner = OnPolicyRunner(
        env,
        cfg,
        log_dir=str(log_dir) if log_dir is not None else None,
        device=args.device,
    )
    if args.disable_logs:
        runner.disable_logs = True
    if args.resume:
        runner.load(str(args.resume))
    try:
        runner.learn(
            num_learning_iterations=cfg["max_iterations"],
            init_at_random_ep_len=False,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
