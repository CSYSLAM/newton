#!/usr/bin/env python3
"""Train an original phase-free Go2 or G1 policy with RSL-RL.

The default invocation performs configuration validation only.  Actual PPO
rollouts require the explicit ``--run`` flag, making accidental long training
jobs impossible during smoke validation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import yaml

if __package__ in (None, ""):
    from locomotion.bounded_actor_critic import install_bounded_actor_critic
    from locomotion.env import SynRealLocomotionEnv
    from locomotion.policy_spec import load_policy_spec
else:
    from .bounded_actor_critic import install_bounded_actor_critic
    from .env import SynRealLocomotionEnv
    from .policy_spec import load_policy_spec


def rsl_rl_config(robot: str) -> dict:
    spec = load_policy_spec(robot)
    ppo = spec.ppo
    bounded_actions = bool(ppo.get("bounded_actions", False))
    desired_kl = ppo.get("desired_kl")
    return {
        "num_steps_per_env": int(ppo["num_steps_per_env"]),
        "max_iterations": int(ppo["max_iterations"]),
        "save_interval": int(ppo["save_interval"]),
        "experiment_name": f"synreal_{spec.name}_locomotion",
        "empirical_normalization": False,
        "logger": "tensorboard",
        "policy": {
            "class_name": "BoundedActorCritic" if bounded_actions else "ActorCritic",
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
            "schedule": str(ppo.get("schedule", "adaptive")),
            "gamma": float(ppo["gamma"]),
            "lam": float(ppo["lam"]),
            "desired_kl": float(desired_kl) if desired_kl is not None else None,
            "max_grad_norm": 1.0,
        },
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go2", "g1"), required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--command-vx", type=float)
    parser.add_argument(
        "--reset-joint-curriculum",
        nargs="+",
        metavar="ITERATION:MAGNITUDE",
        help="reset joint perturbation stages, for example 0:0.01 2000:0.015",
    )
    parser.add_argument("--noise-std", type=float, help="override policy exploration std")
    parser.add_argument("--learning-rate", type=float, help="override PPO learning rate")
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="load resume weights without stale optimizer momentum",
    )
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument(
        "--init-at-random-ep-len",
        action="store_true",
        help="randomize initial timeout counters (disabled by default)",
    )
    parser.add_argument("--visualize", action="store_true", help="show all training worlds in ViewerGL")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="run simulation kernels without Warp CUDA Graph replay",
    )
    parser.add_argument("--disable-logs", action="store_true", help="disable checkpoints and TensorBoard")
    parser.add_argument("--run", action="store_true", help="start PPO; omitted means validate only")
    return parser


def parse_reset_joint_curriculum(values: list[str] | None) -> tuple[tuple[int, float], ...]:
    if not values:
        return ()
    stages: list[tuple[int, float]] = []
    for value in values:
        try:
            iteration_text, magnitude_text = value.split(":", maxsplit=1)
            iteration = int(iteration_text)
            magnitude = float(magnitude_text)
        except ValueError as error:
            raise SystemExit(f"invalid reset curriculum stage {value!r}; expected ITERATION:MAGNITUDE") from error
        if iteration < 0 or not math.isfinite(magnitude) or magnitude < 0.0:
            raise SystemExit(f"invalid reset curriculum stage {value!r}")
        stages.append((iteration, magnitude))
    if stages[0][0] != 0 or any(current[0] >= following[0] for current, following in zip(stages, stages[1:])):
        raise SystemExit("reset curriculum must start at 0 with increasing iterations")
    return tuple(stages)


def apply_runner_overrides(runner, *, noise_std: float | None, learning_rate: float | None) -> None:
    if noise_std is not None:
        policy = runner.alg.policy
        with torch.no_grad():
            if policy.noise_std_type == "log":
                policy.log_std.fill_(math.log(noise_std))
            else:
                policy.std.fill_(noise_std)
    if learning_rate is not None:
        runner.alg.learning_rate = learning_rate
        for parameter_group in runner.alg.optimizer.param_groups:
            parameter_group["lr"] = learning_rate


@torch.no_grad()
def stagger_resumed_episodes(runner, env) -> None:
    """Warm resumed environments to physically consistent episode ages."""

    env.episode_length_buf.copy_(torch.randint_like(env.episode_length_buf, high=env.max_episode_length))
    observation, _ = env.get_observations()
    policy = runner.get_inference_policy(device=env.device)
    print("staggering resumed episodes without PPO updates", flush=True)
    for _ in range(env.max_episode_length):
        actions = policy(observation)
        observation, _, _, _ = env.step(actions.to(env.device))
    runner.train_mode()


def main() -> None:
    args = create_parser().parse_args()
    reset_joint_curriculum = parse_reset_joint_curriculum(args.reset_joint_curriculum)
    spec = load_policy_spec(args.robot)
    cfg = rsl_rl_config(args.robot)
    if args.iterations is not None:
        cfg["max_iterations"] = args.iterations
    if args.reset_optimizer and args.resume is None:
        raise SystemExit("--reset-optimizer requires --resume")
    for name, value in (("noise std", args.noise_std), ("learning rate", args.learning_rate)):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise SystemExit(f"{name} must be a positive finite value")
    if args.noise_std is not None:
        cfg["policy"]["init_noise_std"] = args.noise_std
    if args.learning_rate is not None:
        cfg["algorithm"]["learning_rate"] = args.learning_rate
    summary = {
        "robot": spec.name,
        "abi_sha256": spec.abi_sha256,
        "observation_dim": spec.observation_dim,
        "frame_observation_dim": spec.frame_observation_dim,
        "history_length": spec.history_length,
        "action_dim": spec.action_dim,
        "policy_dt": spec.policy_dt,
        "cuda_graph": not args.disable_cuda_graph,
        "reset_joint_position_curriculum": reset_joint_curriculum,
        "rsl_rl": cfg,
    }
    print(json.dumps(summary, indent=2))
    if not args.run:
        print("configuration valid; pass --run to start PPO")
        return

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as error:
        raise SystemExit(
            "RSL-RL is not installed in this environment. Install the "
            "locomotion dependencies with `uv sync --extra locomotion`."
        ) from error
    install_bounded_actor_critic()
    num_envs = args.num_envs or (2048 if args.robot == "go2" else 512)
    viewer = None
    if args.visualize:
        import newton

        viewer = newton.viewer.ViewerGL(vsync=True)
    env = SynRealLocomotionEnv(
        args.robot,
        num_envs=num_envs,
        device=args.device,
        asset=args.asset,
        seed=args.seed,
        randomize=not args.no_randomization,
        viewer=viewer,
        use_cuda_graph=not args.disable_cuda_graph,
        command_vx=args.command_vx,
        reset_joint_position_curriculum=reset_joint_curriculum,
    )
    log_dir = None
    if not args.disable_logs:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = args.log_root / cfg["experiment_name"] / run_name
        log_dir.mkdir(parents=True, exist_ok=False)
        (log_dir / "policy_spec.yaml").write_bytes(spec.abi_bytes())
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
        runner.load(str(args.resume), load_optimizer=not args.reset_optimizer)
    apply_runner_overrides(
        runner,
        noise_std=args.noise_std,
        learning_rate=args.learning_rate,
    )
    if args.resume:
        stagger_resumed_episodes(runner, env)
    try:
        runner.learn(
            num_learning_iterations=cfg["max_iterations"],
            init_at_random_ep_len=args.init_at_random_ep_len,
        )
        print("training complete", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
