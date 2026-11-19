# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Train a Unitree G1 policy to jump continuously on a cloth trampoline."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from env import JUMP_TARGET_HEIGHT, TrampolineEnv
from torch import nn
from torch.distributions import Normal

import newton.utils

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "g1_trampoline_jump.pt"


def checkpoint_base_path(path: Path) -> Path:
    """Return the stable latest-checkpoint path for a versioned path."""
    prefix, marker, update = path.stem.rpartition("_update_")
    if marker and prefix and update.isdigit():
        return path.with_name(prefix + path.suffix)
    return path


def checkpoint_path_for_update(path: Path, update: int) -> Path:
    """Return an immutable checkpoint path containing the update number."""
    if update <= 0:
        raise ValueError("checkpoint update must be positive")
    base = checkpoint_base_path(path)
    return base.with_name(f"{base.stem}_update_{update:06d}{base.suffix}")


def best_checkpoint_path(path: Path) -> Path:
    """Return the stable best-policy path for a checkpoint series."""
    base = checkpoint_base_path(path)
    return base.with_name(f"{base.stem}_best{base.suffix}")


class ObservationNormalizer:
    """Streaming per-feature mean and variance on the policy device."""

    def __init__(self, size: int, device: torch.device) -> None:
        self.mean = torch.zeros(size, device=device)
        self.var = torch.ones(size, device=device)
        self.count = torch.tensor(1.0e-4, device=device)

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        batch_mean = observations.mean(dim=0)
        batch_var = observations.var(dim=0, unbiased=False)
        batch_count = observations.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        first_moment = self.var * self.count
        second_moment = batch_var * batch_count
        correction = delta.square() * self.count * batch_count / total
        self.mean = new_mean
        self.var = (first_moment + second_moment + correction) / total
        self.count = total

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return ((observations - self.mean) / torch.sqrt(self.var + 1.0e-6)).clamp(-10.0, 10.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.count.copy_(state["count"].to(self.count.device))


def initialize_layer(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs for continuous joint targets."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        policy_action_dim: int,
        policy_action_indices: torch.Tensor,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.policy_action_dim = policy_action_dim
        self.register_buffer("policy_action_indices", policy_action_indices.to(dtype=torch.long))
        self.actor = nn.Sequential(
            initialize_layer(nn.Linear(observation_dim, 256)),
            nn.ELU(),
            initialize_layer(nn.Linear(256, 128)),
            nn.ELU(),
            initialize_layer(nn.Linear(128, 128)),
            nn.ELU(),
            initialize_layer(nn.Linear(128, policy_action_dim), gain=0.01),
        )
        self.critic = nn.Sequential(
            initialize_layer(nn.Linear(observation_dim, 256)),
            nn.ELU(),
            initialize_layer(nn.Linear(256, 128)),
            nn.ELU(),
            initialize_layer(nn.Linear(128, 128)),
            nn.ELU(),
            initialize_layer(nn.Linear(128, 1), gain=1.0),
        )
        self.phase_action_sin = nn.Parameter(torch.zeros(policy_action_dim))
        self.phase_action_cos = nn.Parameter(torch.zeros(policy_action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        policy_actions = (
            self.actor(observations)
            + observations[:, -2:-1] * self.phase_action_sin
            + observations[:, -1:] * self.phase_action_cos
        )
        return policy_actions.index_select(-1, self.policy_action_indices)

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.action_mean(observations)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        values = self.critic(observations).squeeze(-1)
        return actions, log_prob, values

    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        values = self.critic(observations).squeeze(-1)
        return log_prob, entropy, values

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_mean(observations).clamp(-1.0, 1.0)


@torch.no_grad()
def clamp_action_std(model: ActorCritic, min_action_std: float, max_action_std: float) -> None:
    model.log_std.clamp_(
        min=math.log(min_action_std),
        max=math.log(max_action_std),
    )


@torch.no_grad()
def set_action_std(model: ActorCritic, action_std: float) -> None:
    model.log_std.fill_(math.log(action_std))


@torch.no_grad()
def reset_critic(model: ActorCritic) -> None:
    """Reset the value function when fine-tuning on a different reward."""
    linear_layers = [layer for layer in model.critic if isinstance(layer, nn.Linear)]
    for index, layer in enumerate(linear_layers):
        gain = 1.0 if index == len(linear_layers) - 1 else math.sqrt(2.0)
        initialize_layer(layer, gain=gain)


def default_pretrained_policy_path() -> Path:
    asset_path = Path(newton.utils.download_asset("unitree_g1"))
    return asset_path / "rl_policies" / "mjw_g1_29DOF.onnx"


def initialize_actor_from_onnx(model: ActorCritic, path: Path) -> None:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise RuntimeError("Loading the pretrained G1 actor requires the optional 'onnx' package") from error

    graph = onnx.load(path, load_external_data=False).graph
    initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in graph.initializer}
    layer_names = ("actor.0", "actor.2", "actor.4", "actor.6")
    layers = (model.actor[0], model.actor[2], model.actor[4], model.actor[6])
    with torch.no_grad():
        for name, layer in zip(layer_names, layers):
            for parameter_name, parameter in (
                ("weight", layer.weight),
                ("bias", layer.bias),
            ):
                initializer_name = f"{name}.{parameter_name}"
                if initializer_name not in initializers:
                    raise ValueError(f"Pretrained policy is missing {initializer_name}: {path}")
                value = torch.from_numpy(initializers[initializer_name].copy()).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if value.shape != parameter.shape:
                    can_extend_observations = (
                        initializer_name == "actor.0.weight"
                        and value.shape[0] == parameter.shape[0]
                        and parameter.shape[1] - value.shape[1] in (2, 4)
                    )
                    if not can_extend_observations:
                        raise ValueError(
                            f"Pretrained {initializer_name} has shape "
                            f"{tuple(value.shape)}, expected {tuple(parameter.shape)}"
                        )
                    parameter.zero_()
                    parameter[:, : value.shape[1]].copy_(value)
                else:
                    parameter.copy_(value)


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    normalizer: ObservationNormalizer,
    optimizer: torch.optim.Optimizer,
    update: int,
    args: argparse.Namespace,
    best_return: float,
    best_jump_score: float = -math.inf,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    training_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "best_return": best_return,
            "best_jump_score": best_jump_score,
            "observation_dim": model.observation_dim,
            "action_dim": model.action_dim,
            "policy_action_dim": model.policy_action_dim,
            "training_args": training_args,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    normalizer: ObservationNormalizer,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    checkpoint_observation_dim = int(checkpoint["observation_dim"])
    added_observations = model.observation_dim - checkpoint_observation_dim
    migrate_observations = added_observations in (2, 4)
    if checkpoint_observation_dim != model.observation_dim and not migrate_observations:
        raise ValueError("Checkpoint observation dimension does not match the environment")
    if checkpoint["action_dim"] != model.action_dim:
        raise ValueError("Checkpoint action dimension does not match the environment")
    if checkpoint.get("policy_action_dim", model.action_dim) != model.policy_action_dim:
        raise ValueError("Checkpoint pretrained-policy action dimension does not match the environment")
    model_state = checkpoint["model"]
    normalizer_state = checkpoint["normalizer"]
    missing_model_keys = set(model.state_dict()) - set(model_state)
    if optimizer is not None and missing_model_keys:
        raise ValueError(
            "A checkpoint with an older policy architecture must be loaded with "
            "--resume-from so optimizer state can be reset"
        )
    if migrate_observations:
        current_state = model.state_dict()
        for name in ("actor.0.weight", "critic.0.weight"):
            expanded_weight = torch.zeros_like(current_state[name])
            expanded_weight[:, :checkpoint_observation_dim].copy_(model_state[name])
            model_state[name] = expanded_weight
        normalizer_state = {
            **normalizer_state,
            "mean": torch.cat(
                (
                    normalizer_state["mean"],
                    normalizer.mean[-added_observations:].clone(),
                )
            ),
            "var": torch.cat(
                (
                    normalizer_state["var"],
                    normalizer.var[-added_observations:].clone(),
                )
            ),
        }
        if optimizer is not None:
            raise ValueError(
                "A checkpoint with older observations must be loaded with --resume-from so optimizer state can be reset"
            )
        print(f"Migrated checkpoint observations: {checkpoint_observation_dim} -> {model.observation_dim}")
    merged_model_state = model.state_dict()
    merged_model_state.update(model_state)
    model.load_state_dict(merged_model_state)
    normalizer.load_state_dict(normalizer_state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO to make Unitree G1 jump continuously on the trampoline.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument(
        "--phase-learning-rate",
        type=float,
        default=1.0e-3,
        help="learning rate for the compact sinusoidal action residual",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=1.0e-4)
    parser.add_argument("--min-action-std", type=float, default=0.03)
    parser.add_argument("--max-action-std", type=float, default=0.12)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="stop PPO epochs early when approximate KL exceeds this value",
    )
    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="linearly anneal learning rate to 10%% of its initial value",
    )
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="output path for latest, versioned, and best checkpoints",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="exactly resume the output checkpoint, including optimizer state",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="fine-tune actor and normalizer from a checkpoint with fresh optimizer state",
    )
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=0.12,
        help="exploration standard deviation used with --resume-from",
    )
    parser.add_argument(
        "--pretrained-policy",
        type=Path,
        help="G1 ONNX actor; defaults to the Newton unitree_g1 asset",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="randomly initialize the actor instead of loading the G1 ONNX policy",
    )
    parser.add_argument(
        "--normalize-observations",
        action="store_true",
        help="train a running observation normalizer (off for ONNX compatibility)",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_envs",
        "updates",
        "horizon",
        "epochs",
        "minibatches",
        "episode_length",
        "save_interval",
        "log_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.horizon * args.num_envs < args.minibatches:
        raise ValueError("rollout sample count must be at least the minibatch count")
    if args.entropy_coef < 0.0:
        raise ValueError("--entropy-coef must be non-negative")
    if args.learning_rate <= 0.0 or args.phase_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if args.target_kl <= 0.0:
        raise ValueError("--target-kl must be positive")
    if not 0.0 < args.min_action_std <= args.max_action_std:
        raise ValueError("--min-action-std must be positive and no greater than --max-action-std")
    if args.scratch and args.pretrained_policy is not None:
        raise ValueError("--scratch and --pretrained-policy cannot be used together")
    if args.resume and args.resume_from is not None:
        raise ValueError("--resume and --resume-from cannot be used together")
    if (args.resume or args.resume_from is not None) and args.pretrained_policy is not None:
        raise ValueError("checkpoint loading cannot be combined with --pretrained-policy")
    if not args.min_action_std <= args.resume_action_std <= args.max_action_std:
        raise ValueError("--resume-action-std must be between --min-action-std and --max-action-std")


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    torch.manual_seed(args.seed)
    env = TrampolineEnv(
        num_envs=args.num_envs,
        device=args.device,
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
    model = ActorCritic(
        env.observation_dim,
        env.action_dim,
        env.policy_action_dim,
        env.policy_action_indices,
    ).to(env.torch_device)
    normalizer = ObservationNormalizer(env.observation_dim, env.torch_device)
    phase_parameters = [model.phase_action_sin, model.phase_action_cos]
    base_parameters = [
        *model.actor.parameters(),
        *model.critic.parameters(),
        model.log_std,
    ]
    optimizer = torch.optim.Adam(
        [
            {
                "params": base_parameters,
                "lr": args.learning_rate,
                "initial_lr": args.learning_rate,
            },
            {
                "params": phase_parameters,
                "lr": args.phase_learning_rate,
                "initial_lr": args.phase_learning_rate,
            },
        ],
        eps=1.0e-5,
    )

    start_update = 1
    best_return = -math.inf
    best_jump_score = -math.inf
    resume_path = args.resume_from if args.resume_from is not None else args.checkpoint
    if args.resume or args.resume_from is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        fine_tuning = args.resume_from is not None
        checkpoint = load_checkpoint(
            resume_path,
            model,
            normalizer,
            None if fine_tuning else optimizer,
        )
        checkpoint_update = int(checkpoint["update"])
        resumed_action_std = float(model.log_std.detach().exp().mean())
        if fine_tuning:
            set_action_std(model, args.resume_action_std)
            reset_critic(model)
        else:
            best_return = float(checkpoint.get("best_return", -math.inf))
            best_jump_score = float(checkpoint.get("best_jump_score", -math.inf))
            clamp_action_std(model, args.min_action_std, args.max_action_std)
        start_update = checkpoint_update + 1
        if args.updates < start_update:
            raise ValueError(f"--updates must be at least {start_update} when resuming from update {checkpoint_update}")
        print(f"Resuming from update {checkpoint_update}: {resume_path}")
        print(
            f"{'Reset' if fine_tuning else 'Loaded'} action std: "
            f"{resumed_action_std:.4f} -> "
            f"{float(model.log_std.detach().exp().mean()):.4f}"
        )
        if fine_tuning:
            print("Reset critic for the jumping reward")
    elif not args.scratch:
        pretrained_policy = (
            args.pretrained_policy.expanduser().resolve()
            if args.pretrained_policy is not None
            else default_pretrained_policy_path()
        )
        if not pretrained_policy.is_file():
            raise FileNotFoundError(f"Pretrained G1 policy does not exist: {pretrained_policy}")
        initialize_actor_from_onnx(model, pretrained_policy)
        args.pretrained_policy = pretrained_policy
        print(f"Initialized actor from pretrained G1 policy: {pretrained_policy}")

    num_envs = env.num_envs
    horizon = args.horizon
    batch_size = horizon * num_envs
    minibatch_size = max(1, batch_size // args.minibatches)
    observations_buffer = torch.zeros((horizon, num_envs, env.observation_dim), device=env.torch_device)
    actions_buffer = torch.zeros((horizon, num_envs, env.action_dim), device=env.torch_device)
    log_probs_buffer = torch.zeros((horizon, num_envs), device=env.torch_device)
    rewards_buffer = torch.zeros_like(log_probs_buffer)
    dones_buffer = torch.zeros_like(log_probs_buffer)
    values_buffer = torch.zeros_like(log_probs_buffer)

    observations = env.reset()
    if args.normalize_observations:
        normalizer.update(observations)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    start_time = time.perf_counter()

    try:
        for update in range(start_update, args.updates + 1):
            if args.anneal_lr:
                training_updates = max(args.updates - start_update + 1, 1)
                progress = (update - start_update) / training_updates
                current_lr = args.learning_rate * (1.0 - 0.9 * progress)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = parameter_group["initial_lr"] * (1.0 - 0.9 * progress)
            else:
                current_lr = args.learning_rate
            rollout_completed_jumps = 0
            rollout_sustained_jumps = 0
            rollout_terminations = 0
            rollout_above_target = 0
            rollout_jump_height = 0.0
            for step in range(horizon):
                if args.normalize_observations:
                    normalizer.update(observations)
                normalized = normalizer.normalize(observations)
                with torch.no_grad():
                    actions, log_probs, values = model.act(normalized)
                next_observations, rewards, dones, info = env.step(actions)

                observations_buffer[step] = normalized
                actions_buffer[step] = actions
                log_probs_buffer[step] = log_probs
                rewards_buffer[step] = rewards
                dones_buffer[step] = dones.float()
                values_buffer[step] = values
                observations = next_observations
                rollout_completed_jumps += int(info["jump_completed"].sum())
                rollout_sustained_jumps += int(info["sustained_jump_completed"].sum())
                rollout_terminations += int(info["terminated"].sum())
                rollout_above_target += int(info["above_jump_target"].sum())
                rollout_jump_height += float(info["jump_height"].clamp_min(0.0).sum())

                finished = info["episode_length"] > 0
                if torch.any(finished):
                    completed_returns.extend(info["episode_return"][finished].cpu().tolist())
                    completed_lengths.extend(info["episode_length"][finished].cpu().tolist())

            with torch.no_grad():
                next_value = model.critic(normalizer.normalize(observations)).squeeze(-1)
                advantages = torch.zeros_like(rewards_buffer)
                last_advantage = torch.zeros(num_envs, device=env.torch_device)
                for step in reversed(range(horizon)):
                    next_nonterminal = 1.0 - dones_buffer[step]
                    next_values = next_value if step == horizon - 1 else values_buffer[step + 1]
                    delta = rewards_buffer[step] + args.gamma * next_values * next_nonterminal - values_buffer[step]
                    last_advantage = delta + args.gamma * args.gae_lambda * next_nonterminal * last_advantage
                    advantages[step] = last_advantage
                returns = advantages + values_buffer

            batch_observations = observations_buffer.flatten(0, 1)
            batch_actions = actions_buffer.flatten(0, 1)
            batch_log_probs = log_probs_buffer.flatten()
            batch_advantages = advantages.flatten()
            batch_returns = returns.flatten()
            batch_values = values_buffer.flatten()
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std(unbiased=False) + 1.0e-8
            )

            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0
            stop_for_kl = False
            for _ in range(args.epochs):
                permutation = torch.randperm(batch_size, device=env.torch_device)
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    new_log_probs, entropy, new_values = model.evaluate(
                        batch_observations[indices], batch_actions[indices]
                    )
                    ratio = (new_log_probs - batch_log_probs[indices]).exp()
                    log_ratio = new_log_probs - batch_log_probs[indices]
                    approx_kl_value = float(((ratio - 1.0) - log_ratio).mean().detach())
                    if approx_kl_value > args.target_kl:
                        stop_for_kl = True
                        break
                    unclipped = -batch_advantages[indices] * ratio
                    clipped = -batch_advantages[indices] * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    policy_loss = torch.maximum(unclipped, clipped).mean()
                    value_prediction_clipped = batch_values[indices] + (new_values - batch_values[indices]).clamp(
                        -args.clip_coef, args.clip_coef
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (new_values - batch_returns[indices]).square(),
                            (value_prediction_clipped - batch_returns[indices]).square(),
                        ).mean()
                    )
                    entropy_loss = entropy.mean()
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_parameters = [
                        *model.actor.parameters(),
                        model.phase_action_sin,
                        model.phase_action_cos,
                        model.log_std,
                    ]
                    nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm)
                    nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                    optimizer.step()
                    clamp_action_std(model, args.min_action_std, args.max_action_std)
                    policy_loss_value = float(policy_loss.detach())
                    value_loss_value = float(value_loss.detach())
                    entropy_value = float(entropy_loss.detach())
                if stop_for_kl:
                    break

            recent_return = (
                sum(completed_returns[-20:]) / min(20, len(completed_returns))
                if completed_returns
                else float(rewards_buffer.mean() * args.episode_length)
            )
            mean_jump_height = rollout_jump_height / batch_size
            above_target_fraction = rollout_above_target / batch_size
            jump_rate = rollout_completed_jumps / num_envs / (horizon * env.frame_dt)
            sustained_jump_rate = rollout_sustained_jumps / num_envs / (horizon * env.frame_dt)
            termination_rate = rollout_terminations / num_envs / (horizon * env.frame_dt)
            jump_score = (
                sustained_jump_rate
                - termination_rate
                + 0.01 * above_target_fraction
                + 0.001 * mean_jump_height / JUMP_TARGET_HEIGHT
            )
            is_best = jump_score > best_jump_score
            best_return = max(best_return, recent_return)
            best_jump_score = max(best_jump_score, jump_score)
            if update % args.log_interval == 0:
                elapsed = time.perf_counter() - start_time
                completed_updates = update - start_update + 1
                steps_per_second = completed_updates * batch_size / max(elapsed, 1.0e-6)
                recent_length = (
                    sum(completed_lengths[-20:]) / min(20, len(completed_lengths))
                    if completed_lengths
                    else float(env.episode_steps.float().mean())
                )
                action_std = float(model.log_std.detach().exp().mean())
                print(
                    f"update={update:5d} return={recent_return:8.2f} "
                    f"length={recent_length:7.1f} policy={policy_loss_value:7.4f} "
                    f"value={value_loss_value:8.4f} entropy={entropy_value:7.3f} "
                    f"jumps/s={jump_rate:5.2f} sustained={sustained_jump_rate:5.2f} "
                    f"falls/s={termination_rate:5.2f} "
                    f"height={mean_jump_height:5.3f} "
                    f"above10cm={above_target_fraction:5.1%} "
                    f"kl={approx_kl_value:7.5f} lr={current_lr:.2e} "
                    f"std={action_std:6.3f} steps/s={steps_per_second:7.1f}"
                )
            if is_best:
                best_path = best_checkpoint_path(args.checkpoint)
                save_checkpoint(
                    best_path,
                    model,
                    normalizer,
                    optimizer,
                    update,
                    args,
                    best_return,
                    best_jump_score,
                )
                print(f"Updated best checkpoint: {best_path} (jump score {best_jump_score:.4f})")
            if update % args.save_interval == 0 or update == args.updates:
                latest_checkpoint = checkpoint_base_path(args.checkpoint)
                update_checkpoint = checkpoint_path_for_update(latest_checkpoint, update)
                save_checkpoint(
                    update_checkpoint,
                    model,
                    normalizer,
                    optimizer,
                    update,
                    args,
                    best_return,
                    best_jump_score,
                )
                save_checkpoint(
                    latest_checkpoint,
                    model,
                    normalizer,
                    optimizer,
                    update,
                    args,
                    best_return,
                    best_jump_score,
                )
                print(f"Saved checkpoint: {update_checkpoint}")
                print(f"Updated latest checkpoint: {latest_checkpoint}")
    finally:
        env.close()
    return checkpoint_base_path(args.checkpoint)


if __name__ == "__main__":
    train(create_parser().parse_args())
