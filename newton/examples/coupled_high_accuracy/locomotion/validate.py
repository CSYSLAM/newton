#!/usr/bin/env python3
"""Fast ABI, export, and optional SynReal dynamics smoke validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

if __package__ in (None, ""):
    from locomotion.deployment import build_actor, deploy_config
    from locomotion.env import SynRealLocomotionEnv
    from locomotion.observations import ObservationHistory, actor_frame
    from locomotion.policy_spec import load_policy_spec
else:
    from .deployment import build_actor, deploy_config
    from .env import SynRealLocomotionEnv
    from .observations import ObservationHistory, actor_frame
    from .policy_spec import load_policy_spec


def validate_contract(robot: str) -> dict:
    spec = load_policy_spec(robot)
    batch = 3
    device = torch.device("cpu")
    zeros3 = torch.zeros((batch, 3), device=device)
    q = spec.tensor("default_joint_pos", device).expand(batch, -1)
    action = torch.zeros((batch, spec.action_dim), device=device)
    frame = actor_frame(spec, zeros3, zeros3, zeros3, q, action, action, q[0])
    history = ObservationHistory(batch, spec, device)
    ids = torch.arange(batch)
    history.reset(ids, frame)
    obs = history.value()
    actor = build_actor(spec)
    output = actor(obs)
    deployment = deploy_config(spec)
    serialized = yaml.safe_load(yaml.safe_dump(deployment, sort_keys=False))
    terms = serialized["observations"]
    exported_dim = sum(
        len(term["scale"]) * int(term["history_length"]) for name, term in terms.items() if name != "use_gym_history"
    )
    if obs.shape != (batch, spec.observation_dim):
        raise AssertionError(f"invalid observation shape {obs.shape}")
    if output.shape != (batch, spec.action_dim):
        raise AssertionError(f"invalid action shape {output.shape}")
    if exported_dim != spec.observation_dim:
        raise AssertionError(f"deploy.yaml dimension {exported_dim} != {spec.observation_dim}")
    if deployment["joint_ids_map"] != list(spec.joint_ids_map):
        raise AssertionError("joint_ids_map changed during deployment export")
    return {
        "robot": robot,
        "status": "pass",
        "abi_sha256": spec.abi_sha256,
        "observation_dim": spec.observation_dim,
        "action_dim": spec.action_dim,
        "history_length": spec.history_length,
        "policy_dt": spec.policy_dt,
        "actor_output_finite": bool(torch.isfinite(output).all()),
        "deploy_yaml_observation_dim": exported_dim,
    }


def validate_physics(robot: str, steps: int, asset: Path | None) -> dict:
    env = SynRealLocomotionEnv(
        robot,
        num_envs=1,
        asset=asset,
        randomize=False,
        auto_reset=False,
    )
    observations, _ = env.get_observations()
    done = torch.zeros(1, dtype=torch.bool, device=env.device)
    reward = torch.zeros(1, device=env.device)
    for _ in range(steps):
        observations, reward, done, _ = env.step(torch.zeros((1, env.num_actions), device=env.device))
        if not torch.isfinite(observations).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("non-finite state during physics smoke")
        if bool(done.item()):
            break
    return {
        "robot": robot,
        "status": "pass" if not bool(done.item()) else "terminated",
        "steps_requested": steps,
        "steps_completed": int(env.episode_length_buf.item()),
        "final_reward": float(reward.item()),
        "observation_finite": bool(torch.isfinite(observations).all()),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go2", "g1", "all"), default="all")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    robots = ("go2", "g1") if args.robot == "all" else (args.robot,)
    report = {
        "contracts": [validate_contract(robot) for robot in robots],
        "physics": [validate_physics(robot, args.steps, args.asset) for robot in robots] if args.physics else [],
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
