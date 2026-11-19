"""Configuration matching Unitree RL Lab's 29-DoF G1 velocity task."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_JOINT_POS = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3,
    0.25,
    0.0,
    0.97,
    0.15,
    0.0,
    0.0,
    0.3,
    -0.25,
    0.0,
    0.97,
    -0.15,
    0.0,
    0.0,
)


def _joint_parameter(name: str) -> tuple[float, float, float, float]:
    """Return kp, kd, effort limit, and monitored velocity limit."""

    if "hip_pitch" in name or "hip_yaw" in name:
        return 100.0, 2.0, 88.0, 32.0
    if "hip_roll" in name:
        return 100.0, 2.0, 139.0, 20.0
    if "knee" in name:
        return 150.0, 4.0, 139.0, 20.0
    if name == "waist_yaw_joint":
        return 200.0, 5.0, 88.0, 32.0
    if "wrist_pitch" in name or "wrist_yaw" in name:
        return 40.0, 1.0, 5.0, 22.0
    if "ankle" in name:
        return 40.0, 2.0, 25.0, 37.0
    if name in {"waist_roll_joint", "waist_pitch_joint"}:
        return 40.0, 5.0, 25.0, 37.0
    return 40.0, 1.0, 25.0, 37.0


JOINT_KP = tuple(_joint_parameter(name)[0] for name in JOINT_NAMES)
JOINT_KD = tuple(_joint_parameter(name)[1] for name in JOINT_NAMES)
JOINT_EFFORT_LIMIT = tuple(_joint_parameter(name)[2] for name in JOINT_NAMES)
JOINT_VELOCITY_LIMIT = tuple(_joint_parameter(name)[3] for name in JOINT_NAMES)

ARM_JOINTS = tuple(
    index for index, name in enumerate(JOINT_NAMES) if "shoulder" in name or "elbow" in name or "wrist" in name
)
WAIST_JOINTS = tuple(index for index, name in enumerate(JOINT_NAMES) if name.startswith("waist"))
LEG_DEVIATION_JOINTS = tuple(index for index, name in enumerate(JOINT_NAMES) if "hip_roll" in name or "hip_yaw" in name)

POLICY_TERM_DIMS = (3, 3, 3, 29, 29, 29)
CRITIC_TERM_DIMS = (3, 3, 3, 3, 29, 29, 29)
HISTORY_LENGTH = 5
POLICY_OBSERVATION_DIM = sum(POLICY_TERM_DIMS) * HISTORY_LENGTH
CRITIC_OBSERVATION_DIM = sum(CRITIC_TERM_DIMS) * HISTORY_LENGTH


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 20.0
    gravity: float = -9.81
    contact_stiffness: float = 5e4
    contact_damping: float = 500.0
    contact_friction_gain: float = 1.0e3
    ground_friction: float = 1.0
    joint_armature_approximation: float = 0.01
    constraint_alpha: float = 200.0
    joint_constraint_primal_iterations: int = 2
    nonlinear_iterations: int = 30
    linear_iterations: int = 10

    @property
    def policy_dt(self) -> float:
        return self.dt * self.decimation


@dataclass(frozen=True)
class ContactConfig:
    debounce_substeps: int = 2
    history_substeps: int = 3
    maximum_separation: float = 1.0e-4
    duty_window_s: float = 0.8


@dataclass(frozen=True)
class RandomizationConfig:
    friction_range: tuple[float, float] = (0.3, 1.0)
    friction_buckets: int = 64
    correlate_friction_per_env: bool = False
    actuator_enabled: bool = False
    stiffness_scale: tuple[float, float] = (1.0, 1.0)
    damping_scale: tuple[float, float] = (1.0, 1.0)
    motor_strength: tuple[float, float] = (1.0, 1.0)
    joint_zero_offset: tuple[float, float] = (0.0, 0.0)
    policy_delay_ticks: tuple[int, int] = (0, 0)
    push_enabled: bool = False
    push_interval_s: tuple[float, float] = (10.0, 15.0)
    push_velocity: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class CommandConfig:
    initial_lin_vel_x: tuple[float, float] = (-0.1, 0.1)
    initial_lin_vel_y: tuple[float, float] = (-0.1, 0.1)
    initial_ang_vel_z: tuple[float, float] = (-0.1, 0.1)
    limit_lin_vel_x: tuple[float, float] = (-0.5, 1.0)
    limit_lin_vel_y: tuple[float, float] = (-0.3, 0.3)
    limit_ang_vel_z: tuple[float, float] = (-0.2, 0.2)
    resampling_time_s: float = 10.0
    standing_fraction: float = 0.02
    curriculum_increment: float = 0.1
    curriculum_success_rate: float = 0.8
    curriculum_enabled: bool = True
    sampling_mode: str = "uniform"
    slow_fraction: float = 0.0
    turn_fraction: float = 0.0
    lateral_fraction: float = 0.0
    slow_lin_vel_x: tuple[float, float] = (0.0, 0.0)
    turn_lin_vel_x: tuple[float, float] = (0.0, 0.0)
    turn_ang_vel_z: tuple[float, float] = (0.0, 0.0)
    lateral_lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lateral_lin_vel_y: tuple[float, float] = (0.0, 0.0)
    mirror_sampled_commands: bool = False


@dataclass(frozen=True)
class ObservationNoiseConfig:
    enabled: bool = True
    base_ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5


@dataclass(frozen=True)
class RewardConfig:
    track_lin_vel_xy: float = 1.0
    track_ang_vel_z: float = 0.5
    alive: float = 0.15
    lin_vel_z_l2: float = -2.0
    ang_vel_xy_l2: float = -0.05
    joint_vel_l2: float = -0.001
    joint_acc_l2: float = -2.5e-7
    action_rate_l2: float = -0.05
    joint_pos_limits: float = -5.0
    energy: float = -2.0e-5
    joint_deviation_arms: float = -0.1
    joint_deviation_waists: float = -1.0
    joint_deviation_legs: float = -1.0
    flat_orientation: float = -5.0
    base_height: float = -10.0
    gait: float = 0.5
    feet_slide: float = -0.2
    feet_clearance: float = 1.0
    undesired_contacts: float = -1.0
    action_l2: float = 0.0
    arm_action_l2: float = 0.0
    lateral_vel_error_l2: float = 0.0
    yaw_rate_error_l2: float = 0.0
    foot_duty_imbalance: float = 0.0
    repeated_touchdown: float = 0.0
    tracking_std: float = 0.5
    base_height_target: float = 0.78
    foot_clearance_target: float = 0.1
    foot_clearance_std: float = 0.05
    foot_clearance_tanh_multiplier: float = 2.0
    gait_period: float = 0.8
    gait_stance_ratio: float = 0.55
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    supported_swing_clearance: bool = False


@dataclass(frozen=True)
class G1EnvConfig:
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42
    asset_path: Path | None = None
    enable_self_collisions: bool = True
    mesh_approximation: str = "convex_hull"
    action_scale: float = 0.25
    initial_base_height: float = 0.8
    auto_reset: bool = True
    use_cuda_graph: bool = True
    periodic_push_enabled: bool = True
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    contacts: ContactConfig = field(default_factory=ContactConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)


def make_velocity_bootstrap_config(config: G1EnvConfig) -> G1EnvConfig:
    """Return the forward-only preset used to escape the standing optimum."""

    return replace(
        config,
        periodic_push_enabled=False,
        commands=replace(
            config.commands,
            initial_lin_vel_x=(0.35, 0.65),
            initial_lin_vel_y=(0.0, 0.0),
            initial_ang_vel_z=(0.0, 0.0),
            standing_fraction=0.0,
            curriculum_enabled=False,
        ),
        rewards=replace(
            config.rewards,
            track_lin_vel_xy=4.0,
            alive=0.05,
            joint_deviation_waists=-0.5,
            joint_deviation_legs=-0.5,
            base_height=-5.0,
            gait=0.0,
            action_l2=-0.001,
            arm_action_l2=-0.003,
            supported_swing_clearance=True,
        ),
    )


def make_straight_symmetry_config(config: G1EnvConfig, stage: int = 1) -> G1EnvConfig:
    """Return a phase-free preset for straight and bilaterally balanced walking."""

    if stage not in (1, 2):
        raise ValueError("straight-symmetry stage must be 1 or 2")
    command_overrides = {
        "initial_lin_vel_x": (0.25, 0.70),
        "initial_lin_vel_y": (0.0, 0.0),
        "initial_ang_vel_z": (0.0, 0.0),
        "standing_fraction": 0.05,
        "curriculum_enabled": False,
        "sampling_mode": "straight_symmetry",
        "slow_fraction": 0.10,
        "slow_lin_vel_x": (0.05, 0.25),
        "mirror_sampled_commands": True,
    }
    if stage == 2:
        command_overrides.update(
            {
                "initial_lin_vel_x": (0.20, 0.75),
                "standing_fraction": 0.10,
                "slow_fraction": 0.15,
                "turn_fraction": 0.10,
                "lateral_fraction": 0.05,
                "turn_lin_vel_x": (0.20, 0.60),
                "turn_ang_vel_z": (-0.15, 0.15),
                "lateral_lin_vel_x": (0.20, 0.60),
                "lateral_lin_vel_y": (-0.08, 0.08),
            }
        )
    return replace(
        config,
        periodic_push_enabled=False,
        randomization=replace(
            config.randomization,
            correlate_friction_per_env=True,
        ),
        commands=replace(config.commands, **command_overrides),
        rewards=replace(
            config.rewards,
            track_lin_vel_xy=4.0,
            track_ang_vel_z=1.5,
            alive=0.05,
            joint_deviation_waists=-0.5,
            joint_deviation_legs=-0.5,
            base_height=-5.0,
            gait=0.0,
            feet_clearance=0.5,
            action_l2=-0.001,
            arm_action_l2=-0.003,
            lateral_vel_error_l2=-0.5,
            yaw_rate_error_l2=-0.5,
            foot_duty_imbalance=-0.25,
            repeated_touchdown=-0.5,
            supported_swing_clearance=True,
        ),
    )


def make_robustness_config(config: G1EnvConfig, stage: int = 1) -> G1EnvConfig:
    """Add staged actuator, latency, and push variation to the straight preset."""

    if stage not in (1, 2, 3):
        raise ValueError("robustness stage must be 1, 2, or 3")
    ranges = {
        1: {
            "stiffness_scale": (0.95, 1.05),
            "damping_scale": (0.90, 1.10),
            "motor_strength": (0.95, 1.05),
            "joint_zero_offset": (-0.01, 0.01),
            "policy_delay_ticks": (0, 1),
            "push_enabled": False,
        },
        2: {
            "stiffness_scale": (0.90, 1.10),
            "damping_scale": (0.85, 1.15),
            "motor_strength": (0.90, 1.10),
            "joint_zero_offset": (-0.015, 0.015),
            "policy_delay_ticks": (0, 2),
            "push_enabled": False,
        },
        3: {
            "stiffness_scale": (0.80, 1.20),
            "damping_scale": (0.80, 1.20),
            "motor_strength": (0.85, 1.15),
            "joint_zero_offset": (-0.02, 0.02),
            "policy_delay_ticks": (0, 2),
            "push_enabled": True,
            "push_interval_s": (8.0, 12.0),
            "push_velocity": (-0.20, 0.20),
        },
    }
    straight = make_straight_symmetry_config(config, stage=2)
    return replace(
        straight,
        randomization=replace(
            straight.randomization,
            actuator_enabled=True,
            **ranges[stage],
        ),
    )


def make_effort_bootstrap_config(config: G1EnvConfig) -> G1EnvConfig:
    """Adapt the walking actor to the deployment effort envelope before randomization."""

    bootstrap = make_velocity_bootstrap_config(config)
    return replace(
        bootstrap,
        randomization=replace(
            bootstrap.randomization,
            correlate_friction_per_env=True,
            actuator_enabled=True,
            stiffness_scale=(1.0, 1.0),
            damping_scale=(1.0, 1.0),
            motor_strength=(1.0, 1.0),
            joint_zero_offset=(0.0, 0.0),
            policy_delay_ticks=(0, 0),
            push_enabled=False,
        ),
    )


def make_ppo_config(
    max_iterations: int = 50_000,
    *,
    experiment_name: str = "synreal_g1_locomotion",
    init_noise_std: float = 1.0,
) -> dict:
    """Return the RSL-RL settings used by Unitree RL Lab locomotion."""

    return {
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "save_interval": 100,
        "experiment_name": experiment_name,
        "empirical_normalization": False,
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": init_noise_std,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
