#!/usr/bin/env python3
"""Run a short rollout and report Unitree Go2 task signals."""

from __future__ import annotations

import argparse
import json

import torch

if __package__ in (None, ""):
    from locomotion_unitree.env import UnitreeGo2Env
else:
    from .env import UnitreeGo2Env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--friction-randomization",
        choices=("per-env", "batch", "fixed"),
        default="per-env",
    )
    parser.add_argument("--enable-self-collisions", action="store_true")
    args = parser.parse_args()

    env = UnitreeGo2Env(
        num_envs=args.num_envs,
        device=args.device,
        randomize=args.randomize,
        use_cuda_graph=not args.disable_cuda_graph,
        friction_randomization=args.friction_randomization,
        enable_self_collisions=args.enable_self_collisions,
    )
    contact_sum = torch.zeros(4, device=env.device)
    base_contact_steps = 0
    undesired_sum = 0.0
    done_sum = 0
    reward_sum = 0.0
    try:
        actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        for _ in range(args.steps):
            _, reward, done, _ = env.step(actions)
            contact_sum += env.foot_contact_any.float().mean(0)
            base_contact_steps += int(torch.any(env.base_contact))
            undesired_sum += float(env.undesired_contact.float().sum(1).mean())
            done_sum += int(done.sum())
            reward_sum += float(reward.mean())
        observation, extras = env.get_observations()
        result = {
            "actor_shape": list(observation.shape),
            "critic_shape": list(extras["observations"]["critic"].shape),
            "foot_contact_fraction": (contact_sum / args.steps).cpu().tolist(),
            "base_contact_step_fraction": base_contact_steps / args.steps,
            "undesired_contacts_per_env_step": undesired_sum / args.steps,
            "done_count": done_sum,
            "mean_reward": reward_sum / args.steps,
            "root_height_mean": float(env.body_q_0[:, env.root_body, 2].mean()),
            "friction_min": float(env.friction.min()),
            "friction_max": float(env.friction.max()),
            "solver_shared_world_count": int(env.solver._shared_world_count),
            "finite": bool(torch.isfinite(observation).all()),
        }
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
