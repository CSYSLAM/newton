"""Single-source policy ABI shared by training, sim2sim, and deployment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
OBSERVATION_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


def _tuple(values: Any, cast: type = float) -> tuple:
    return tuple(cast(value) for value in values)


@dataclass(frozen=True)
class RandomizationSpec:
    static_friction: tuple[float, float]
    dynamic_friction: tuple[float, float]
    restitution: tuple[float, float]
    base_mass_add: tuple[float, float]
    link_mass_scale: tuple[float, float]
    base_com: tuple[float, float]
    stiffness_scale: tuple[float, float]
    damping_scale: tuple[float, float]
    motor_strength: tuple[float, float]
    joint_zero_offset: tuple[float, float]
    observation_noise: dict[str, float]
    policy_delay_ticks: tuple[int, int]
    push_interval_s: tuple[float, float]
    push_velocity: tuple[float, float]
    reset_xy: tuple[float, float]
    reset_joint_position: tuple[float, float]


@dataclass(frozen=True)
class PolicySpec:
    """Validated locomotion contract in policy joint order."""

    name: str
    asset_source: str
    asset_name: str
    asset_folder: str
    asset_relative_path: str
    asset_commit: str
    sdk_joint_names: tuple[str, ...]
    joint_ids_map: tuple[int, ...]
    default_joint_pos: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    torque_limits: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    action_scale: tuple[float, ...]
    action_clip: float
    joint_target_limits: tuple[tuple[float, float], ...] | None
    command_ranges: dict[str, tuple[float, float]]
    standing_probability: float
    ang_vel_scale: float
    joint_vel_scale: float
    history_length: int
    dt: float
    decimation: int
    episode_length_s: float
    initial_height: float
    fall_height: float
    fall_grace_s: float
    fall_confirm_s: float
    base_height_target: float
    base_height_sigma: float
    base_height_reward_scale: float
    termination_roll_degrees: float | None
    termination_pitch_degrees: float | None
    contact_stiffness: float
    contact_damping: float
    contact_friction: float
    solver_iterations: int
    linear_iterations: int
    use_gym_history: bool
    randomization: RandomizationSpec
    ppo: dict[str, Any]

    @property
    def action_dim(self) -> int:
        return len(self.default_joint_pos)

    @property
    def frame_observation_dim(self) -> int:
        return 9 + 3 * self.action_dim

    @property
    def observation_dim(self) -> int:
        return self.frame_observation_dim * self.history_length

    @property
    def policy_dt(self) -> float:
        return self.dt * self.decimation

    @property
    def max_episode_length(self) -> int:
        return round(self.episode_length_s / self.policy_dt)

    @property
    def policy_joint_names(self) -> tuple[str, ...]:
        return tuple(self.sdk_joint_names[index] for index in self.joint_ids_map)

    @property
    def abi_sha256(self) -> str:
        return hashlib.sha256(self.abi_bytes()).hexdigest()

    def abi_bytes(self) -> bytes:
        return yaml.safe_dump(self.as_dict(), sort_keys=True).encode()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sdk_joint_names": list(self.sdk_joint_names),
            "joint_ids_map": list(self.joint_ids_map),
            "policy_joint_names": list(self.policy_joint_names),
            "default_joint_pos": list(self.default_joint_pos),
            "stiffness": list(self.stiffness),
            "damping": list(self.damping),
            "torque_limits": list(self.torque_limits),
            "velocity_limits": list(self.velocity_limits),
            "action_scale": list(self.action_scale),
            "action_clip": self.action_clip,
            "joint_target_limits": (
                [list(bounds) for bounds in self.joint_target_limits] if self.joint_target_limits is not None else None
            ),
            "command_ranges": self.command_ranges,
            "standing_probability": self.standing_probability,
            "ang_vel_scale": self.ang_vel_scale,
            "joint_vel_scale": self.joint_vel_scale,
            "history_length": self.history_length,
            "use_gym_history": self.use_gym_history,
            "dt": self.dt,
            "decimation": self.decimation,
            "episode_length_s": self.episode_length_s,
            "fall_height": self.fall_height,
            "fall_grace_s": self.fall_grace_s,
            "fall_confirm_s": self.fall_confirm_s,
            "base_height_target": self.base_height_target,
            "base_height_sigma": self.base_height_sigma,
            "base_height_reward_scale": self.base_height_reward_scale,
            "observation_terms": list(OBSERVATION_TERMS),
        }

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(getattr(self, name), device=device, dtype=torch.float32)


def _pair(data: dict[str, Any], key: str, cast: type = float) -> tuple:
    values = data[key]
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{key} must be an ordered two-element range")
    return _tuple(values, cast)


def _validate(spec: PolicySpec) -> None:
    size = spec.action_dim
    vector_fields = (
        "joint_ids_map",
        "stiffness",
        "damping",
        "torque_limits",
        "velocity_limits",
        "action_scale",
    )
    for field in vector_fields:
        if len(getattr(spec, field)) != size:
            raise ValueError(f"{spec.name}.{field} must contain {size} values")
    if len(spec.sdk_joint_names) != size:
        raise ValueError(f"{spec.name}.sdk_joint_names must contain {size} values")
    if sorted(spec.joint_ids_map) != list(range(size)):
        raise ValueError(f"{spec.name}.joint_ids_map must be a permutation")
    if len(set(spec.sdk_joint_names)) != size:
        raise ValueError(f"{spec.name} contains duplicate SDK joint names")
    if (
        min(
            spec.dt,
            spec.decimation,
            spec.history_length,
            spec.solver_iterations,
            spec.linear_iterations,
        )
        <= 0
    ):
        raise ValueError("time steps, history, and solver iterations must be positive")
    if abs(spec.policy_dt - 0.02) > 1.0e-9:
        raise ValueError(f"{spec.name} deployment ABI must run at 50 Hz")
    if spec.asset_source not in {"git", "newton"}:
        raise ValueError(f"unsupported asset source: {spec.asset_source}")
    if spec.asset_source == "newton" and not spec.asset_name:
        raise ValueError("Newton assets require asset.name")
    if spec.asset_source == "git" and (not spec.asset_folder or not spec.asset_commit):
        raise ValueError("Git assets require asset.folder and asset.commit")
    if spec.action_clip <= 0:
        raise ValueError("action_clip must be positive")
    if (
        min(
            spec.fall_grace_s,
            spec.fall_confirm_s,
            spec.base_height_sigma,
        )
        < 0
    ):
        raise ValueError("fall and startup timing values must be non-negative")
    if spec.base_height_target <= 0 or spec.base_height_reward_scale < 0:
        raise ValueError("base height target must be positive and reward scale non-negative")
    if spec.joint_target_limits is not None:
        if len(spec.joint_target_limits) != size:
            raise ValueError(f"{spec.name}.joint_target_limits must contain {size} pairs")
        for name, position, bounds in zip(spec.policy_joint_names, spec.default_joint_pos, spec.joint_target_limits):
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"invalid joint target limits for {name}")
            if not bounds[0] < position < bounds[1]:
                raise ValueError(f"default position for {name} must lie inside target limits")
    if not 0.0 <= spec.standing_probability <= 1.0:
        raise ValueError("standing_probability must be in [0, 1]")
    if set(spec.command_ranges) != {"lin_vel_x", "lin_vel_y", "ang_vel_z"}:
        raise ValueError("command_ranges must define vx, vy, and yaw")
    for field in ("stiffness", "damping", "torque_limits", "velocity_limits", "action_scale"):
        if any(value <= 0 for value in getattr(spec, field)):
            raise ValueError(f"{spec.name}.{field} values must be positive")


def load_policy_spec(robot: str | Path) -> PolicySpec:
    """Load ``go2``/``g1`` or an explicit YAML policy contract."""

    path = Path(robot)
    if not path.suffix:
        path = CONFIG_DIR / f"{path.name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    is_go2 = str(data["name"]) == "go2"
    asset = data["asset"]
    randomization = data["randomization"]
    sdk_joint_names = _tuple(data["sdk_joint_names"], str)
    joint_ids_map = _tuple(data["joint_ids_map"], int)
    policy_joint_names = tuple(sdk_joint_names[index] for index in joint_ids_map)
    target_limits = data.get("joint_target_limits")
    if target_limits is not None:
        target_limits = tuple(_tuple(target_limits[name]) for name in policy_joint_names)
    spec = PolicySpec(
        name=str(data["name"]),
        asset_source=str(asset.get("source", "git")),
        asset_name=str(asset.get("name", "")),
        asset_folder=str(asset.get("folder", "")),
        asset_relative_path=str(asset["relative_path"]),
        asset_commit=str(asset.get("commit", "")),
        sdk_joint_names=sdk_joint_names,
        joint_ids_map=joint_ids_map,
        default_joint_pos=_tuple(data["default_joint_pos"]),
        stiffness=_tuple(data["stiffness"]),
        damping=_tuple(data["damping"]),
        torque_limits=_tuple(data["torque_limits"]),
        velocity_limits=_tuple(data["velocity_limits"]),
        action_scale=_tuple(data["action_scale"]),
        action_clip=float(data.get("action_clip", 100.0)),
        joint_target_limits=target_limits,
        command_ranges={key: _tuple(value) for key, value in data["command_ranges"].items()},
        standing_probability=float(data.get("standing_probability", 0.1)),
        ang_vel_scale=float(data["observations"]["ang_vel_scale"]),
        joint_vel_scale=float(data["observations"]["joint_vel_scale"]),
        history_length=int(data["observations"]["history_length"]),
        dt=float(data["simulation"]["dt"]),
        decimation=int(data["simulation"]["decimation"]),
        episode_length_s=float(data["simulation"]["episode_length_s"]),
        initial_height=float(data["simulation"]["initial_height"]),
        fall_height=float(data["simulation"]["fall_height"]),
        fall_grace_s=float(data["simulation"].get("fall_grace_s", 0.6 if is_go2 else 0.0)),
        fall_confirm_s=float(data["simulation"].get("fall_confirm_s", 0.12 if is_go2 else 0.0)),
        base_height_target=float(
            data["simulation"].get("base_height_target", 0.28 if is_go2 else data["simulation"]["initial_height"])
        ),
        base_height_sigma=float(data["simulation"].get("base_height_sigma", 0.05 if is_go2 else 1.0)),
        base_height_reward_scale=float(data["simulation"].get("base_height_reward_scale", 1.0 if is_go2 else 0.0)),
        termination_roll_degrees=(
            float(data["simulation"]["termination_roll_degrees"])
            if "termination_roll_degrees" in data["simulation"]
            else None
        ),
        termination_pitch_degrees=(
            float(data["simulation"]["termination_pitch_degrees"])
            if "termination_pitch_degrees" in data["simulation"]
            else None
        ),
        contact_stiffness=float(data["simulation"]["contact_stiffness"]),
        contact_damping=float(data["simulation"]["contact_damping"]),
        contact_friction=float(data["simulation"]["contact_friction"]),
        solver_iterations=int(data["simulation"]["solver_iterations"]),
        linear_iterations=int(data["simulation"].get("linear_iterations", 10)),
        use_gym_history=bool(data["observations"].get("use_gym_history", False)),
        randomization=RandomizationSpec(
            static_friction=_pair(randomization, "static_friction"),
            dynamic_friction=_pair(randomization, "dynamic_friction"),
            restitution=_pair(randomization, "restitution"),
            base_mass_add=_pair(randomization, "base_mass_add"),
            link_mass_scale=_pair(randomization, "link_mass_scale"),
            base_com=_pair(randomization, "base_com"),
            stiffness_scale=_pair(randomization, "stiffness_scale"),
            damping_scale=_pair(randomization, "damping_scale"),
            motor_strength=_pair(randomization, "motor_strength"),
            joint_zero_offset=_pair(randomization, "joint_zero_offset"),
            observation_noise={key: float(value) for key, value in randomization["observation_noise"].items()},
            policy_delay_ticks=_pair(randomization, "policy_delay_ticks", int),
            push_interval_s=_pair(randomization, "push_interval_s"),
            push_velocity=_pair(randomization, "push_velocity"),
            reset_xy=_pair(randomization, "reset_xy") if "reset_xy" in randomization else (-0.05, 0.05),
            reset_joint_position=_pair(randomization, "reset_joint_position")
            if "reset_joint_position" in randomization
            else (-0.02, 0.02),
        ),
        ppo=dict(data["ppo"]),
    )
    _validate(spec)
    return spec
