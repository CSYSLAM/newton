# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained G1 trampoline-jumping checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
import warp as wp
from env import TrampolineEnv
from train import (
    DEFAULT_CHECKPOINT,
    ActorCritic,
    ObservationNormalizer,
    load_checkpoint,
)

import newton.examples


class Evaluation:
    def __init__(self, viewer, args) -> None:
        self.viewer = viewer
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}\n"
                "Train one with: uv run python "
                "examples/python/trampoline/train.py"
            )

        self.env = TrampolineEnv(
            num_envs=1,
            device=str(wp.get_device()),
            viewer=viewer,
            episode_length=args.episode_length,
            seed=args.seed,
            sim_substeps=args.sim_substeps,
            contact_stiffness=args.contact_stiffness,
            contact_damping=args.contact_damping,
            shape_friction=args.shape_friction,
            soft_contact_friction=args.soft_contact_friction,
            particle_radius=args.particle_radius,
            foot_contact_margin=args.foot_contact_margin,
            spawn_height_offset=args.spawn_height_offset,
        )
        self.frame_dt = self.env.frame_dt
        self.sim_time = 0.0
        self.model = ActorCritic(
            self.env.observation_dim,
            self.env.action_dim,
            self.env.policy_action_dim,
            self.env.policy_action_indices,
        ).to(self.env.torch_device)
        self.normalizer = ObservationNormalizer(self.env.observation_dim, self.env.torch_device)
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.normalizer)
        self.model.eval()
        self.observations = self.env.reset()
        self.frame_count = 0
        self.total_reward = 0.0
        self.termination_count = 0
        self.completed_episodes = 0
        self.longest_episode = 0
        self.height_sum = 0.0
        self.upright_sum = 0.0
        self.center_sum = 0.0
        self.last_reward = 0.0
        self.last_height = 0.0
        self.last_jump_height = 0.0
        self.last_desired_jump_height = 0.0
        self.peak_jump_height = 0.0
        self.jump_count = 0
        self.sustained_jump_count = 0
        self.above_target_frames = 0
        self.last_upright = 0.0
        print(f"Loaded checkpoint update {checkpoint['update']}: {checkpoint_path}")
        print(
            "Evaluation contact settings: "
            f"substeps={args.sim_substeps}, "
            f"ke={args.contact_stiffness:g}, kd={args.contact_damping:g}, "
            f"particle_radius={args.particle_radius:g}, "
            f"foot_margin={args.foot_contact_margin:g}, "
            f"spawn_offset={args.spawn_height_offset:g}"
        )
        if hasattr(viewer, "renderer"):
            viewer.register_ui_callback(self.render_ui, position="side")

    @torch.no_grad()
    def step(self) -> None:
        normalized = self.normalizer.normalize(self.observations)
        actions = self.model.deterministic(normalized)
        self.observations, rewards, dones, info = self.env.step(actions)
        self.sim_time = self.env.sim_time
        self.frame_count += 1
        self.last_reward = float(rewards[0])
        self.last_height = float(info["base_height"][0])
        self.last_jump_height = float(info["jump_height"][0])
        self.last_desired_jump_height = float(info["desired_jump_height"][0])
        self.peak_jump_height = max(self.peak_jump_height, float(info["jump_peak_height"][0]))
        self.jump_count += int(info["jump_completed"][0])
        self.sustained_jump_count += int(info["sustained_jump_completed"][0])
        self.above_target_frames += int(info["above_jump_target"][0])
        self.last_upright = float(info["upright"][0])
        self.total_reward += self.last_reward
        self.height_sum += self.last_height
        self.upright_sum += self.last_upright
        self.center_sum += float(info["center_distance"][0])
        if bool(dones[0]):
            self.completed_episodes += 1
            self.longest_episode = max(self.longest_episode, int(info["episode_length"][0]))
            if bool(info["terminated"][0]):
                self.termination_count += 1

    def render(self) -> None:
        self.env.render()

    def render_ui(self, imgui) -> None:
        imgui.text(f"Reward: {self.last_reward:.3f}")
        imgui.text(f"Base height: {self.last_height:.3f} m")
        imgui.text(f"Jump height: {self.last_jump_height:.3f} m")
        imgui.text(f"Target height: {self.last_desired_jump_height:.3f} m")
        imgui.text(f"Peak jump: {self.peak_jump_height:.3f} m")
        imgui.text(f"Completed 10 cm jumps: {self.jump_count}")
        imgui.text(f"Sustained jumps: {self.sustained_jump_count}")
        imgui.text(f"Upright: {self.last_upright:.3f}")
        imgui.text(f"Falls: {self.termination_count}")

    def test_final(self) -> None:
        frames = max(self.frame_count, 1)
        current_episode = int(self.env.episode_steps[0])
        longest = max(self.longest_episode, current_episode)
        print(
            "G1 trampoline evaluation:\n"
            f"  simulated time: {self.frame_count * self.frame_dt:.2f} s\n"
            f"  mean reward: {self.total_reward / frames:.4f}\n"
            f"  mean base height: {self.height_sum / frames:.4f} m\n"
            f"  peak jump height: {self.peak_jump_height:.4f} m\n"
            f"  completed 10 cm jumps: {self.jump_count}\n"
            f"  sustained jumps after first bounce: {self.sustained_jump_count}\n"
            f"  time above 10 cm: {self.above_target_frames / frames:.1%}\n"
            f"  mean upright score: {self.upright_sum / frames:.4f}\n"
            f"  mean center distance: {self.center_sum / frames:.4f} m\n"
            f"  longest episode: {longest * self.frame_dt:.2f} s\n"
            f"  falls: {self.termination_count}"
        )
        if not torch.isfinite(self.observations).all():
            raise ValueError("Policy evaluation produced non-finite observations")


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="trained checkpoint produced by train.py",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=3600,
        help="maximum episode length in 50 Hz policy steps",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=8,
        help="physics substeps per 50 Hz policy step",
    )
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=1.0e3)
    parser.add_argument("--shape-friction", type=float, default=0.75)
    parser.add_argument("--soft-contact-friction", type=float, default=0.6)
    parser.add_argument("--particle-radius", type=float, default=0.025)
    parser.add_argument(
        "--foot-contact-margin",
        type=float,
        default=0.005,
        help="extra soft-contact thickness on the two foot collision boxes",
    )
    parser.add_argument(
        "--spawn-height-offset",
        type=float,
        default=0.06,
        help="raise G1 so its feet start above the trampoline contact layer",
    )
    parser.set_defaults(num_frames=1800)
    return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())
    newton.examples.run(Evaluation(viewer, args), args)
