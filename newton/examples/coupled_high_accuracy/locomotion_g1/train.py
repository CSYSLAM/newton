#!/usr/bin/env python3
"""Train Unitree RL Lab's G1 velocity task with the SynReal environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    from locomotion_g1.config import (
        G1EnvConfig,
        make_effort_bootstrap_config,
        make_ppo_config,
        make_robustness_config,
        make_straight_symmetry_config,
        make_velocity_bootstrap_config,
    )
    from locomotion_g1.env import SynRealG1LocomotionEnv
else:
    from .config import (
        G1EnvConfig,
        make_effort_bootstrap_config,
        make_ppo_config,
        make_robustness_config,
        make_straight_symmetry_config,
        make_velocity_bootstrap_config,
    )
    from .env import SynRealG1LocomotionEnv


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actor_only(policy: torch.nn.Module, checkpoint: Path) -> int:
    """Warm-start the actor without restoring critic, exploration, or optimizer state."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_state = state.get("model_state_dict", state.get("model", state))
    actor_state = {
        name.removeprefix("actor."): value for name, value in model_state.items() if name.startswith("actor.")
    }
    if not actor_state:
        raise ValueError(f"checkpoint contains no actor parameters: {checkpoint}")
    policy.actor.load_state_dict(actor_state, strict=True)
    return len(actor_state)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="start PPO training")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="construct the environment and run this many zero-action steps",
    )
    parser.add_argument(
        "--smoke-action-std",
        type=float,
        default=0.0,
        help="standard deviation of Gaussian actions used by smoke mode",
    )
    parser.add_argument(
        "--smoke-report-interval",
        type=int,
        default=1,
        help="print one smoke record every N steps",
    )
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset", type=Path)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--actor-init",
        type=Path,
        help="initialize only actor weights; critic, exploration, and optimizer stay fresh",
    )
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--velocity-bootstrap",
        action="store_true",
        help="enable the forward-only preset for escaping the standing optimum",
    )
    preset.add_argument(
        "--effort-bootstrap",
        action="store_true",
        help="adapt a walking actor to nominal effort limits before randomization",
    )
    preset.add_argument(
        "--straight-symmetry-stage",
        type=int,
        choices=(1, 2),
        help="enable staged phase-free straightness and bilateral-balance training",
    )
    preset.add_argument(
        "--robustness-stage",
        type=int,
        choices=(1, 2, 3),
        help="enable staged actuator, latency, and push robustness training",
    )
    parser.add_argument("--log-root", type=Path, default=Path("runs"))
    parser.add_argument("--disable-logs", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--mesh-approximation",
        choices=("convex_hull", "bounding_box"),
        default="convex_hull",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if (
        args.num_envs <= 0
        or args.iterations <= 0
        or args.smoke_steps < 0
        or args.smoke_action_std < 0.0
        or args.smoke_report_interval <= 0
    ):
        raise SystemExit("environment, iteration, and smoke-step counts are invalid")
    actor_only_preset = (
        args.velocity_bootstrap or args.effort_bootstrap or args.straight_symmetry_stage or args.robustness_stage
    )
    if actor_only_preset and args.actor_init is None:
        raise SystemExit("the selected training preset requires --actor-init")
    actor_checkpoint = None
    if args.actor_init is not None:
        actor_checkpoint = args.actor_init.expanduser().resolve()
        if not actor_checkpoint.is_file():
            raise SystemExit(f"actor checkpoint does not exist: {actor_checkpoint}")
    environment_config = G1EnvConfig(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        asset_path=args.asset,
        enable_self_collisions=not args.disable_self_collisions,
        mesh_approximation=args.mesh_approximation,
        use_cuda_graph=not args.disable_cuda_graph,
    )
    if args.velocity_bootstrap:
        environment_config = make_velocity_bootstrap_config(environment_config)
    elif args.effort_bootstrap:
        environment_config = make_effort_bootstrap_config(environment_config)
    elif args.straight_symmetry_stage is not None:
        environment_config = make_straight_symmetry_config(environment_config, args.straight_symmetry_stage)
    elif args.robustness_stage is not None:
        environment_config = make_robustness_config(environment_config, args.robustness_stage)
    if args.velocity_bootstrap:
        experiment_name = "synreal_g1_velocity_bootstrap"
        training_preset = "velocity_bootstrap_v1"
        init_noise_std = 0.5
    elif args.effort_bootstrap:
        experiment_name = "synreal_g1_effort_bootstrap"
        training_preset = "effort_bootstrap_v1"
        init_noise_std = 0.1
    elif args.straight_symmetry_stage is not None:
        experiment_name = f"synreal_g1_straight_symmetry_s{args.straight_symmetry_stage}"
        training_preset = f"straight_symmetry_v1_stage{args.straight_symmetry_stage}"
        init_noise_std = 0.3
    elif args.robustness_stage is not None:
        experiment_name = f"synreal_g1_robustness_s{args.robustness_stage}"
        training_preset = f"robustness_v1_stage{args.robustness_stage}"
        init_noise_std = 0.2
    else:
        experiment_name = "synreal_g1_locomotion"
        training_preset = "official_parity"
        init_noise_std = 1.0
    ppo_config = make_ppo_config(
        args.iterations,
        experiment_name=experiment_name,
        init_noise_std=init_noise_std,
    )
    resume_checkpoint = args.resume.expanduser().resolve() if args.resume is not None else None
    source_checkpoint = actor_checkpoint or resume_checkpoint
    initialization_summary = {
        "mode": (
            "actor_only" if actor_checkpoint is not None else "resume" if resume_checkpoint is not None else "fresh"
        ),
        "checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
        "checkpoint_sha256": (_sha256(source_checkpoint) if source_checkpoint is not None else None),
        "critic": "fresh" if resume_checkpoint is None else "restored",
        "optimizer": "fresh" if resume_checkpoint is None else "restored",
        "exploration_std": ppo_config["policy"]["init_noise_std"],
    }
    summary = {
        "environment": "synreal_g1_locomotion",
        "training_preset": training_preset,
        "initialization": initialization_summary,
        "policy_observation_dim": SynRealG1LocomotionEnv.num_obs,
        "critic_observation_dim": SynRealG1LocomotionEnv.num_privileged_obs,
        "action_dim": SynRealG1LocomotionEnv.num_actions,
        "contact_semantics": "debounced_geometry_not_force_threshold",
        "friction_semantics": (
            "single_coulomb_coefficient_64_bucket_correlated_per_env"
            if environment_config.randomization.correlate_friction_per_env
            else "single_coulomb_coefficient_64_bucket_per_shape"
        ),
        "actuator_randomization": (
            "per_joint_per_environment_per_episode" if environment_config.randomization.actuator_enabled else "disabled"
        ),
        "effort_limit_enforced_in_control": (environment_config.randomization.actuator_enabled),
        "raw_action_clipping": False,
        "mass_randomization": False,
        "com_randomization": False,
        "environment_config": asdict(environment_config),
        "rsl_rl": ppo_config,
    }
    print(json.dumps(summary, indent=2, default=_json_default))
    if not args.run and args.smoke_steps == 0:
        print("configuration valid; pass --run or --smoke-steps N")
        return

    viewer = None
    if args.visualize:
        import newton

        viewer = newton.viewer.ViewerGL(vsync=True)
    env = SynRealG1LocomotionEnv(environment_config, viewer=viewer)
    summary.update(
        {
            "resolved_asset": str(env.asset_path),
            "asset_source": env.asset_source,
            "asset_sha256": _sha256(env.asset_path),
        }
    )
    print(
        json.dumps(
            {
                "resolved_asset": summary["resolved_asset"],
                "asset_source": summary["asset_source"],
                "asset_sha256": summary["asset_sha256"],
                "contact_capacity": int(env.contacts.rigid_contact_max),
                "self_collisions": environment_config.enable_self_collisions,
                "friction": env.friction_summary,
            },
            indent=2,
        )
    )
    if args.smoke_steps:
        try:
            for step in range(args.smoke_steps):
                actions = args.smoke_action_std * torch.randn(
                    (env.num_envs, env.num_actions),
                    device=env.device,
                    generator=env.generator,
                )
                observation, reward, done, extras = env.step(actions)
                should_report = (
                    step == 0
                    or step + 1 == args.smoke_steps
                    or (step + 1) % args.smoke_report_interval == 0
                    or bool(torch.any(done))
                )
                if not should_report:
                    continue
                state = env._current_state()
                metrics = extras["metrics"]
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "observation_finite": bool(torch.isfinite(observation).all()),
                            "mean_reward": float(reward.mean()),
                            "done_count": int(done.sum()),
                            "contact_overflow": bool(metrics["contact_overflow"]),
                            "mean_root_height": float(state["root_position"][:, 2].mean()),
                            "left_foot_contact_rate": float(metrics["geometry_contact/left"]),
                            "right_foot_contact_rate": float(metrics["geometry_contact/right"]),
                            "mean_undesired_contact_bodies": float(metrics["geometry_contact/undesired_bodies"]),
                            "gpu_max_memory_mb": round(
                                torch.cuda.max_memory_allocated(env.device) / (1024 * 1024),
                                1,
                            ),
                        }
                    ),
                    flush=True,
                )
        finally:
            env.close()
        return

    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError as error:
        env.close()
        raise SystemExit("RSL-RL is not installed. Run `uv sync --extra locomotion`.") from error

    log_dir = None
    if not args.disable_logs:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = args.log_root / ppo_config["experiment_name"] / stamp
        log_dir.mkdir(parents=True, exist_ok=False)
        (log_dir / "environment.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
    runner = OnPolicyRunner(
        env,
        ppo_config,
        log_dir=str(log_dir) if log_dir is not None else None,
        device=args.device,
    )
    if args.disable_logs:
        runner.disable_logs = True
    if args.resume is not None:
        runner.load(str(args.resume.expanduser().resolve()))
    elif actor_checkpoint is not None:
        loaded_tensors = load_actor_only(runner.alg.policy, actor_checkpoint)
        print(
            json.dumps(
                {
                    "actor_initialization": str(actor_checkpoint),
                    "actor_tensors_loaded": loaded_tensors,
                    "critic_optimizer_and_exploration_restored": False,
                },
                indent=2,
            ),
            flush=True,
        )
    try:
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
