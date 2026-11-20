"""Independent vectorized Unitree G1 locomotion environment for SynReal."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import warp as wp

import newton
import newton.utils
from newton import JointTargetMode
from newton.solvers import coupled_high_accuracy as cha

from .config import (
    ARM_JOINTS,
    CRITIC_OBSERVATION_DIM,
    CRITIC_TERM_DIMS,
    DEFAULT_JOINT_POS,
    HISTORY_LENGTH,
    JOINT_EFFORT_LIMIT,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    JOINT_VELOCITY_LIMIT,
    LEG_DEVIATION_JOINTS,
    POLICY_OBSERVATION_DIM,
    POLICY_TERM_DIMS,
    WAIST_JOINTS,
    CommandConfig,
    G1EnvConfig,
)
from .contact import GeometryContactTracker
from .math_ops import FootContactDutyWindow, TermHistory, yaw_quaternion
from .observations import critic_terms, policy_terms, root_terms
from .randomization import EpisodeRandomization, delayed_action, uniform
from .rewards import RewardInputs, compute_reward

DEFAULT_UNITREE_RL_LAB_ASSET = Path(__file__).resolve().parent / "assets" / "unitree_g1_29dof" / "g1_29dof_rev_1_0.usd"


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def resolve_g1_asset(configured_path: Path | None) -> tuple[Path, str]:
    """Resolve the local derived USD package, with a portable URDF fallback."""

    candidates: list[tuple[Path, str]] = []
    if configured_path is not None:
        candidates.append((configured_path.expanduser(), "configured"))
    environment_path = os.environ.get("G1_ASSET_PATH")
    if environment_path:
        candidates.append((Path(environment_path).expanduser(), "G1_ASSET_PATH"))
    candidates.append((DEFAULT_UNITREE_RL_LAB_ASSET, "repository_local_derived_usd"))
    for path, source in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved, source
    if configured_path is not None:
        raise FileNotFoundError(configured_path)

    asset_root = newton.utils.download_asset("unitree_g1")
    fallback = asset_root / "urdf" / "g1_29dof_rev_1_0.urdf"
    if not fallback.is_file():
        raise FileNotFoundError("Could not resolve the Unitree RL Lab G1 asset or Newton's G1 URDF")
    return fallback, "newton_urdf_fallback"


def _map_named_indices(labels: list[str], names: tuple[str, ...], kind: str) -> list[int]:
    by_name = {_basename(label): index for index, label in enumerate(labels)}
    missing = sorted(set(names) - by_name.keys())
    if missing:
        raise RuntimeError(f"G1 asset is missing {kind}: {missing}")
    result = [by_name[name] for name in names]
    if len(set(result)) != len(names):
        raise RuntimeError(f"G1 {kind} mapping is not one-to-one")
    return result


def _randomize_robot_friction(
    builder: Any,
    *,
    num_envs: int,
    shapes_per_env: int,
    friction_range: tuple[float, float],
    num_buckets: int,
    seed: int,
    correlate_per_env: bool,
) -> dict[str, float | bool]:
    if num_buckets <= 0:
        raise ValueError("friction_buckets must be positive")
    low, high = friction_range
    if low < 0.0 or high < low:
        raise ValueError("friction_range must be non-negative and ordered")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    buckets = low + (high - low) * torch.rand(num_buckets, generator=generator)
    bucket_shape = (num_envs, 1) if correlate_per_env else (num_envs, shapes_per_env)
    bucket_ids = torch.randint(num_buckets, bucket_shape, generator=generator)
    if correlate_per_env:
        bucket_ids = bucket_ids.expand(-1, shapes_per_env)
    coefficients = buckets[bucket_ids]
    robot_shape_count = num_envs * shapes_per_env
    builder.shape_material_mu[:robot_shape_count] = coefficients.flatten().tolist()
    return {
        "minimum": float(coefficients.min()),
        "mean": float(coefficients.mean()),
        "maximum": float(coefficients.max()),
        "correlated_per_env": correlate_per_env,
    }


def _sample_range(
    count: int,
    bounds: tuple[float, float],
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    low, high = bounds
    if high < low:
        raise ValueError(f"invalid sampling range {bounds}")
    return low + (high - low) * torch.rand(count, device=device, generator=generator)


def sample_velocity_commands(
    config: CommandConfig,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample either upstream-uniform or explicitly mirrored staged commands."""

    if count < 0:
        raise ValueError("command count must be non-negative")
    if count == 0:
        return torch.zeros((0, 3), device=device)
    if config.sampling_mode == "uniform":
        commands = torch.stack(
            [
                _sample_range(count, bounds, device=device, generator=generator)
                for bounds in (
                    config.initial_lin_vel_x,
                    config.initial_lin_vel_y,
                    config.initial_ang_vel_z,
                )
            ],
            dim=-1,
        )
        standing = torch.rand(count, device=device, generator=generator)
        commands[standing <= config.standing_fraction] = 0.0
        return commands
    if config.sampling_mode != "straight_symmetry":
        raise ValueError(f"unsupported command sampling mode {config.sampling_mode!r}")

    fractions = (
        config.standing_fraction,
        config.slow_fraction,
        config.turn_fraction,
        config.lateral_fraction,
    )
    if any(value < 0.0 for value in fractions) or sum(fractions) > 1.0:
        raise ValueError("straight-symmetry command fractions must sum to at most one")
    sample_count = (count + 1) // 2 if config.mirror_sampled_commands else count
    commands = torch.zeros((sample_count, 3), device=device)
    draw = torch.rand(sample_count, device=device, generator=generator)
    standing_end = config.standing_fraction
    slow_end = standing_end + config.slow_fraction
    turn_end = slow_end + config.turn_fraction
    lateral_end = turn_end + config.lateral_fraction
    masks = {
        "standing": draw < standing_end,
        "slow": (draw >= standing_end) & (draw < slow_end),
        "turn": (draw >= slow_end) & (draw < turn_end),
        "lateral": (draw >= turn_end) & (draw < lateral_end),
        "straight": draw >= lateral_end,
    }
    mode_ranges = {
        "slow": (config.slow_lin_vel_x, None, None),
        "turn": (config.turn_lin_vel_x, None, config.turn_ang_vel_z),
        "lateral": (
            config.lateral_lin_vel_x,
            config.lateral_lin_vel_y,
            None,
        ),
        "straight": (config.initial_lin_vel_x, None, None),
    }
    for mode, (x_range, y_range, yaw_range) in mode_ranges.items():
        mask = masks[mode]
        mode_count = int(mask.sum())
        commands[mask, 0] = _sample_range(mode_count, x_range, device=device, generator=generator)
        if y_range is not None:
            commands[mask, 1] = _sample_range(mode_count, y_range, device=device, generator=generator)
        if yaw_range is not None:
            commands[mask, 2] = _sample_range(mode_count, yaw_range, device=device, generator=generator)
    if not config.mirror_sampled_commands:
        return commands
    mirrored = commands.clone()
    mirrored[:, 1:] *= -1.0
    return torch.cat((commands, mirrored), dim=0)[:count]


def build_g1_scene(config: G1EnvConfig) -> tuple[Any, dict[str, Any]]:
    """Build replicated G1 worlds without using another locomotion environment."""
    simulation = config.simulation
    robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
    robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=simulation.joint_armature_approximation,
        limit_ke=1.0e3,
        limit_kd=10.0,
        friction=1.0e-5,
    )
    robot.default_shape_cfg.ke = simulation.contact_stiffness
    robot.default_shape_cfg.kd = simulation.contact_damping
    robot.default_shape_cfg.kf = simulation.contact_friction_gain
    robot.default_shape_cfg.mu = simulation.ground_friction

    asset_path, asset_source = resolve_g1_asset(config.asset_path)
    import_options = {
        "floating": True,
        "collapse_fixed_joints": True,
        "enable_self_collisions": config.enable_self_collisions,
    }
    if asset_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        robot.add_usd(
            str(asset_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, config.initial_base_height)),
            joint_ordering="dfs",
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
            **import_options,
        )
    elif asset_path.suffix.lower() == ".urdf":
        robot.default_shape_cfg.density = 0.0
        robot.add_urdf(
            str(asset_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, config.initial_base_height)),
            **import_options,
        )
    else:
        raise ValueError(f"Unsupported G1 asset type: {asset_path}")

    joint_indices = _map_named_indices(robot.joint_label, JOINT_NAMES, "joints")
    q_indices = [int(robot.joint_q_start[index]) for index in joint_indices]
    qd_indices = [int(robot.joint_qd_start[index]) for index in joint_indices]
    if len(set(q_indices)) != 29 or len(set(qd_indices)) != 29:
        raise RuntimeError("Expected 29 independent G1 joint coordinates")

    robot.joint_q[:3] = [0.0, 0.0, config.initial_base_height]
    robot.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    for slot, (q_index, qd_index) in enumerate(zip(q_indices, qd_indices, strict=True)):
        default = DEFAULT_JOINT_POS[slot]
        robot.joint_q[q_index] = default
        robot.joint_target_q[q_index] = default
        robot.joint_target_ke[qd_index] = JOINT_KP[slot]
        robot.joint_target_kd[qd_index] = JOINT_KD[slot]
        robot.joint_effort_limit[qd_index] = JOINT_EFFORT_LIMIT[slot]
        robot.joint_armature[qd_index] = simulation.joint_armature_approximation
        robot.joint_target_mode[qd_index] = int(JointTargetMode.POSITION)

    for shape in range(len(robot.shape_material_mu)):
        robot.shape_material_ke[shape] = simulation.contact_stiffness
        robot.shape_material_kd[shape] = simulation.contact_damping
        robot.shape_material_kf[shape] = simulation.contact_friction_gain
        robot.shape_material_mu[shape] = simulation.ground_friction
    if config.mesh_approximation:
        robot.approximate_meshes(config.mesh_approximation)

    body_indices = _map_named_indices(
        robot.body_label,
        ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"),
        "bodies",
    )
    root_body, left_foot_body, right_foot_body = body_indices
    body_names = tuple(_basename(label) for label in robot.body_label)
    undesired_body_indices = tuple(index for index, name in enumerate(body_names) if "ankle" not in name)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.ke = simulation.contact_stiffness
    builder.default_shape_cfg.kd = simulation.contact_damping
    builder.default_shape_cfg.kf = simulation.contact_friction_gain
    builder.default_shape_cfg.mu = simulation.ground_friction
    builder.replicate(robot, config.num_envs)
    friction_summary = _randomize_robot_friction(
        builder,
        num_envs=config.num_envs,
        shapes_per_env=len(robot.shape_material_mu),
        friction_range=config.randomization.friction_range,
        num_buckets=config.randomization.friction_buckets,
        seed=config.seed,
        correlate_per_env=config.randomization.correlate_friction_per_env,
    )
    ground_config = newton.ModelBuilder.ShapeConfig(
        ke=simulation.contact_stiffness,
        kd=simulation.contact_damping,
        kf=simulation.contact_friction_gain,
        mu=simulation.ground_friction,
    )
    builder.add_ground_plane(cfg=ground_config)
    return builder, {
        "asset_path": asset_path,
        "asset_source": asset_source,
        "q_indices": tuple(q_indices),
        "qd_indices": tuple(qd_indices),
        "root_body": root_body,
        "foot_bodies": (left_foot_body, right_foot_body),
        "undesired_bodies": undesired_body_indices,
        "bodies_per_env": len(robot.body_label),
        "q_per_env": len(robot.joint_q),
        "qd_per_env": len(robot.joint_qd),
        "targets_per_env": len(robot.joint_target_q),
        "friction_summary": friction_summary,
    }


class SynRealG1LocomotionEnv:
    """RSL-RL vector environment for Unitree G1 velocity tracking."""

    num_actions = 29
    num_obs = POLICY_OBSERVATION_DIM
    num_privileged_obs = CRITIC_OBSERVATION_DIM

    def __init__(
        self,
        config: G1EnvConfig | None = None,
        *,
        viewer: Any | None = None,
    ) -> None:
        self.cfg = config or G1EnvConfig()
        if self.cfg.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("SynRealG1LocomotionEnv requires CUDA")
        if self.cfg.simulation.decimation <= 0:
            raise ValueError("simulation decimation must be positive")

        self.num_envs = self.cfg.num_envs
        self.device = torch.device(self.cfg.device)
        torch.cuda.set_device(self.device)
        self.wp_device = wp.get_device(self.cfg.device)
        if not self.wp_device.is_cuda:
            raise RuntimeError("SynRealG1LocomotionEnv requires a CUDA Warp device")
        self.stream = wp.get_stream(self.wp_device)
        self.generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        self.step_dt = self.cfg.simulation.policy_dt
        self.physics_dt = self.cfg.simulation.dt
        self.max_episode_length = round(self.cfg.simulation.episode_length_s / self.step_dt)
        self.sim_time = 0.0
        self.common_step_counter = 0
        self.viewer = viewer

        builder, layout = build_g1_scene(self.cfg)
        self.asset_path = layout["asset_path"]
        self.asset_source = layout["asset_source"]
        self.model = builder.finalize(device=self.wp_device)
        self.friction_summary = layout["friction_summary"]
        self.model.set_gravity((0.0, 0.0, self.cfg.simulation.gravity))
        self.solver = cha.SolverCoupledHighAccuracy(
            self.model,
            joint_drive_armature=self.cfg.simulation.joint_armature_approximation,
        )
        self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SIX_SPRINGS
        self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.constraint_alpha = self.cfg.simulation.constraint_alpha
        self.solver.config.joint_constraint_primal_iterations = self.cfg.simulation.joint_constraint_primal_iterations
        self.solver.config.nonlinear_iterations = self.cfg.simulation.nonlinear_iterations
        self.solver.config.linear_iterations = self.cfg.simulation.linear_iterations

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_target_q)
        wp.copy(self.control.joint_target_qd, self.model.joint_target_qd)
        wp.copy(self.control.joint_f, self.model.joint_f)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
        )
        self.state_1.assign(self.state_0)
        self.solver.sync_affine_body_state(self.state_0)
        self.solver.sync_affine_body_state(self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.q_per_env = layout["q_per_env"]
        self.qd_per_env = layout["qd_per_env"]
        self.targets_per_env = layout["targets_per_env"]
        self.bodies_per_env = layout["bodies_per_env"]
        expected_sizes = (
            (len(self.state_0.joint_q), self.q_per_env, "joint_q"),
            (len(self.state_0.joint_qd), self.qd_per_env, "joint_qd"),
            (len(self.state_0.body_q), self.bodies_per_env, "body_q"),
            (len(self.control.joint_target_q), self.targets_per_env, "targets"),
        )
        for total, per_env, name in expected_sizes:
            if total != self.num_envs * per_env:
                raise RuntimeError(f"Replicated {name} is not world-contiguous")

        self.q_indices = torch.tensor(layout["q_indices"], dtype=torch.long, device=self.device)
        self.qd_indices = torch.tensor(layout["qd_indices"], dtype=torch.long, device=self.device)
        self.target_indices = self.q_indices
        self.root_body = int(layout["root_body"])
        self.foot_bodies = torch.tensor(layout["foot_bodies"], dtype=torch.long, device=self.device)
        self.arm_indices = torch.tensor(ARM_JOINTS, dtype=torch.long, device=self.device)
        self.waist_indices = torch.tensor(WAIST_JOINTS, dtype=torch.long, device=self.device)
        self.leg_deviation_indices = torch.tensor(LEG_DEVIATION_JOINTS, dtype=torch.long, device=self.device)
        self._bind_torch_views()

        self.initial_joint_q = self.joint_q_0.clone()
        self.initial_joint_qd = self.joint_qd_0.clone()
        self.default_control_targets = self.control_targets.clone()
        self.default_joint_position = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device).expand(
            self.num_envs, -1
        )
        self.base_kp = torch.tensor(JOINT_KP, device=self.device)
        self.base_kd = torch.tensor(JOINT_KD, device=self.device)
        self.base_effort_limit = torch.tensor(JOINT_EFFORT_LIMIT, device=self.device)
        self.kp = self.base_kp.expand(self.num_envs, -1).clone()
        self.kd = self.base_kd.expand(self.num_envs, -1).clone()
        self.effort_limit = self.base_effort_limit.expand(self.num_envs, -1).clone()
        self.velocity_limit = torch.tensor(JOINT_VELOCITY_LIMIT, device=self.device).expand(self.num_envs, -1)
        joint_lower = (
            wp.to_torch(self.model.joint_limit_lower)
            .view(self.num_envs, self.qd_per_env)
            .index_select(1, self.qd_indices)
        )
        joint_upper = (
            wp.to_torch(self.model.joint_limit_upper)
            .view(self.num_envs, self.qd_per_env)
            .index_select(1, self.qd_indices)
        )
        midpoint = 0.5 * (joint_lower + joint_upper)
        half_range = 0.45 * (joint_upper - joint_lower)
        self.soft_joint_lower = midpoint - half_range
        self.soft_joint_upper = midpoint + half_range

        contact_cfg = self.cfg.contacts
        self.contact_tracker = GeometryContactTracker(
            self.model,
            self.contacts,
            num_envs=self.num_envs,
            bodies_per_env=self.bodies_per_env,
            foot_body_indices=layout["foot_bodies"],
            undesired_body_indices=layout["undesired_bodies"],
            debounce_substeps=contact_cfg.debounce_substeps,
            history_substeps=contact_cfg.history_substeps,
            maximum_separation=contact_cfg.maximum_separation,
            dt=self.physics_dt,
            device=self.wp_device,
            stream=self.stream,
        )
        duty_window_steps = round(contact_cfg.duty_window_s / self.step_dt)
        self.foot_contact_duty = FootContactDutyWindow(self.num_envs, duty_window_steps, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.applied_actions = torch.zeros_like(self.actions)
        self.desired_targets = torch.zeros_like(self.actions)
        delay_high = self.cfg.randomization.policy_delay_ticks[1]
        if delay_high < 0:
            raise ValueError("maximum policy delay must be non-negative")
        self.action_queue = torch.zeros((self.num_envs, delay_high + 1, self.num_actions), device=self.device)
        self.episode_randomization = EpisodeRandomization.create(
            self.cfg.randomization,
            self.num_envs,
            self.num_actions,
            self.step_dt,
            self.device,
            self.generator,
        )
        self._bind_actuator_arrays()
        self.previous_joint_velocity = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        command_cfg = self.cfg.commands
        self.command_ranges = {
            "x": list(command_cfg.initial_lin_vel_x),
            "y": list(command_cfg.initial_lin_vel_y),
            "yaw": list(command_cfg.initial_ang_vel_z),
        }
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_track_reward = torch.zeros(self.num_envs, device=self.device)
        self.episode_sample_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_forward_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_lateral_velocity_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_lateral_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_yaw_rate_error_abs_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_lateral_displacement = torch.zeros(self.num_envs, device=self.device)
        self.episode_yaw_error_integral = torch.zeros(self.num_envs, device=self.device)
        self.episode_flight_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_single_support_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_left_air_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_right_air_steps = torch.zeros(self.num_envs, device=self.device)
        self.episode_action_over_three = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_abs_action = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_abs_arm_action = torch.zeros(self.num_envs, device=self.device)
        self.episode_torque_saturation = torch.zeros(self.num_envs, device=self.device)
        self.episode_push_count = torch.zeros(self.num_envs, device=self.device)
        self.last_push_velocity = torch.zeros((self.num_envs, 2), device=self.device)
        self.episode_touchdowns = torch.zeros(self.num_envs, device=self.device)
        self.episode_alternating_touchdowns = torch.zeros(self.num_envs, device=self.device)
        self.episode_repeated_touchdowns = torch.zeros(self.num_envs, device=self.device)
        self.previous_policy_foot_contact = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.last_touchdown_foot = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timeout_buf = torch.zeros_like(self.reset_buf)
        self.policy_history = TermHistory(
            self.num_envs,
            POLICY_TERM_DIMS,
            HISTORY_LENGTH,
            device=self.device,
        )
        self.critic_history = TermHistory(
            self.num_envs,
            CRITIC_TERM_DIMS,
            HISTORY_LENGTH,
            device=self.device,
        )
        self._actor_observation = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self._critic_observation = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device)
        self.extras: dict[str, Any] = {}
        self.simulation_graph = None

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            if self.num_envs > 1:
                self.viewer.set_world_offsets((2.5, 2.5, 0.0))
            self.viewer.set_camera(pos=wp.vec3(3.2, -4.2, 2.2), pitch=-10.0, yaw=52.0)

        self.reset()
        if self.cfg.use_cuda_graph:
            self.capture_simulation_graph()
            self.reset()

    def _bind_torch_views(self) -> None:
        def body_view(array: Any) -> torch.Tensor:
            return wp.to_torch(array).view(self.num_envs, self.bodies_per_env, -1)

        self.joint_q_0 = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self.q_per_env)
        self.joint_q_1 = wp.to_torch(self.state_1.joint_q).view(self.num_envs, self.q_per_env)
        self.joint_qd_0 = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, self.qd_per_env)
        self.joint_qd_1 = wp.to_torch(self.state_1.joint_qd).view(self.num_envs, self.qd_per_env)
        self.body_q_0 = body_view(self.state_0.body_q)
        self.body_q_1 = body_view(self.state_1.body_q)
        self.body_qd_0 = body_view(self.state_0.body_qd)
        self.body_qd_1 = body_view(self.state_1.body_qd)
        self.control_targets = wp.to_torch(self.control.joint_target_q).view(self.num_envs, self.targets_per_env)
        self.model_stiffness = wp.to_torch(self.model.joint_target_ke).view(self.num_envs, self.qd_per_env)
        self.model_damping = wp.to_torch(self.model.joint_target_kd).view(self.num_envs, self.qd_per_env)
        self.model_effort = wp.to_torch(self.model.joint_effort_limit).view(self.num_envs, self.qd_per_env)

    def _bind_actuator_arrays(self) -> None:
        randomization = self.episode_randomization
        self._desired_targets_wp = wp.from_torch(self.desired_targets)
        self._base_kp_wp = wp.from_torch(self.base_kp)
        self._base_kd_wp = wp.from_torch(self.base_kd)
        self._base_effort_limit_wp = wp.from_torch(self.base_effort_limit)
        self._stiffness_scale_wp = wp.from_torch(randomization.stiffness_scale)
        self._damping_scale_wp = wp.from_torch(randomization.damping_scale)
        self._motor_strength_wp = wp.from_torch(randomization.motor_strength)
        self._q_indices_wp = wp.from_torch(self.q_indices)
        self._qd_indices_wp = wp.from_torch(self.qd_indices)
        self._target_indices_wp = wp.from_torch(self.target_indices)

    def _current_state(self) -> dict[str, torch.Tensor]:
        root_pose = self.body_q_0[:, self.root_body]
        root_velocity_world = self.body_qd_0[:, self.root_body]
        root_quaternion = torch.nn.functional.normalize(root_pose[:, 3:7], dim=-1)
        linear_velocity, angular_velocity, projected_gravity = root_terms(root_quaternion, root_velocity_world)
        return {
            "root_position": root_pose[:, :3],
            "root_quaternion": root_quaternion,
            "root_velocity_world": root_velocity_world,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "projected_gravity": projected_gravity,
            "joint_position": self.joint_q_0.index_select(1, self.q_indices),
            "joint_velocity": self.joint_qd_0.index_select(1, self.qd_indices),
            "foot_position": self.body_q_0.index_select(1, self.foot_bodies)[..., :3],
            "foot_velocity": self.body_qd_0.index_select(1, self.foot_bodies)[..., :3],
        }

    def _observation_terms(
        self, state: dict[str, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        randomized_default = self._randomized_default_joint_position()
        policy = policy_terms(
            state["angular_velocity"],
            state["projected_gravity"],
            self.commands,
            state["joint_position"],
            randomized_default,
            state["joint_velocity"],
            self.actions,
            self.cfg.observation_noise,
            self.generator,
        )
        critic = critic_terms(
            state["linear_velocity"],
            state["angular_velocity"],
            state["projected_gravity"],
            self.commands,
            state["joint_position"],
            randomized_default,
            state["joint_velocity"],
            self.actions,
        )
        return policy, critic

    def _update_observations(self, reset_ids: torch.Tensor | None = None) -> None:
        policy, critic = self._observation_terms(self._current_state())
        self._actor_observation = self.policy_history.update(policy, reset_ids)
        self._critic_observation = self.critic_history.update(critic, reset_ids)
        if self._actor_observation.shape[1] != self.num_obs:
            raise RuntimeError("Policy observation ABI changed unexpectedly")
        if self._critic_observation.shape[1] != self.num_privileged_obs:
            raise RuntimeError("Critic observation ABI changed unexpectedly")

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._actor_observation, {"observations": {"critic": self._critic_observation}}

    def get_privileged_observations(self) -> torch.Tensor:
        return self._critic_observation

    def _uniform(self, count: int, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        if count == 0:
            return
        command_config = self.cfg.commands
        if command_config.sampling_mode == "uniform":
            command_config = replace(
                command_config,
                initial_lin_vel_x=tuple(self.command_ranges["x"]),
                initial_lin_vel_y=tuple(self.command_ranges["y"]),
                initial_ang_vel_z=tuple(self.command_ranges["yaw"]),
            )
        self.commands[env_ids] = sample_velocity_commands(
            command_config,
            count,
            device=self.device,
            generator=self.generator,
        )
        self.command_steps[env_ids] = 0

    def _randomized_default_joint_position(self) -> torch.Tensor:
        return self.default_joint_position + self.episode_randomization.joint_zero_offset

    def _apply_episode_randomization(self, env_ids: torch.Tensor) -> None:
        randomization = self.episode_randomization
        randomization.resample(
            self.cfg.randomization,
            env_ids,
            self.step_dt,
            self.generator,
        )
        self.kp[env_ids] = self.base_kp * randomization.stiffness_scale[env_ids]
        self.kd[env_ids] = self.base_kd * randomization.damping_scale[env_ids]
        self.effort_limit[env_ids] = self.base_effort_limit * randomization.motor_strength[env_ids]
        self.model_stiffness[env_ids[:, None], self.qd_indices] = self.kp[env_ids]
        self.model_damping[env_ids[:, None], self.qd_indices] = self.kd[env_ids]
        self.model_effort[env_ids[:, None], self.qd_indices] = self.effort_limit[env_ids]

    def _delayed_action(self, actions: torch.Tensor) -> torch.Tensor:
        return delayed_action(self.action_queue, self.episode_randomization.delay_ticks, actions)

    def _launch_limited_joint_targets(self) -> None:
        from .simulation_kernels import update_limited_joint_targets

        wp.launch(
            update_limited_joint_targets,
            dim=self.num_envs * self.num_actions,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._desired_targets_wp,
                self._base_kp_wp,
                self._base_kd_wp,
                self._base_effort_limit_wp,
                self._stiffness_scale_wp,
                self._damping_scale_wp,
                self._motor_strength_wp,
                self._q_indices_wp,
                self._qd_indices_wp,
                self._target_indices_wp,
                self.num_actions,
                self.q_per_env,
                self.qd_per_env,
                self.targets_per_env,
            ],
            outputs=[self.control.joint_target_q],
            device=self.wp_device,
            stream=self.stream,
        )

    def _sync_kinematics(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()
        with wp.ScopedStream(self.stream):
            for state in (self.state_0, self.state_1):
                newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
                self.solver.sync_affine_body_state(state)
        wp.synchronize_stream(self.stream)

    @torch.no_grad()
    def _reset_state(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        self._apply_episode_randomization(env_ids)
        randomized_default = self._randomized_default_joint_position()
        reset_q = self.initial_joint_q[env_ids].clone()
        reset_qd = self.initial_joint_qd[env_ids].clone()
        reset_q[:, 0] += self._uniform(count, -0.5, 0.5)
        reset_q[:, 1] += self._uniform(count, -0.5, 0.5)
        reset_q[:, 3:7] = yaw_quaternion(self._uniform(count, -math.pi, math.pi))
        reset_q[:, self.q_indices] = self.default_joint_position[env_ids]
        reset_qd.zero_()
        reset_qd[:, self.qd_indices] = self._uniform(count * 29, -1.0, 1.0).view(count, 29)
        self.joint_q_0[env_ids] = reset_q
        self.joint_q_1[env_ids] = reset_q
        self.joint_qd_0[env_ids] = reset_qd
        self.joint_qd_1[env_ids] = reset_qd
        self.control_targets[env_ids] = self.default_control_targets[env_ids]
        self.control_targets[env_ids[:, None], self.target_indices] = randomized_default[env_ids]
        self.desired_targets[env_ids] = randomized_default[env_ids]
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.applied_actions[env_ids] = 0.0
        self.action_queue[env_ids] = 0.0
        self.previous_joint_velocity[env_ids] = reset_qd.index_select(1, self.qd_indices)
        self.episode_length_buf[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.episode_track_reward[env_ids] = 0.0
        self.episode_sample_steps[env_ids] = 0
        self.episode_forward_velocity_sum[env_ids] = 0.0
        self.episode_velocity_error_sum[env_ids] = 0.0
        self.episode_lateral_velocity_abs_sum[env_ids] = 0.0
        self.episode_lateral_velocity_sum[env_ids] = 0.0
        self.episode_yaw_rate_error_abs_sum[env_ids] = 0.0
        self.episode_lateral_displacement[env_ids] = 0.0
        self.episode_yaw_error_integral[env_ids] = 0.0
        self.episode_flight_steps[env_ids] = 0.0
        self.episode_single_support_steps[env_ids] = 0.0
        self.episode_left_air_steps[env_ids] = 0.0
        self.episode_right_air_steps[env_ids] = 0.0
        self.episode_action_over_three[env_ids] = 0.0
        self.episode_max_abs_action[env_ids] = 0.0
        self.episode_max_abs_arm_action[env_ids] = 0.0
        self.episode_torque_saturation[env_ids] = 0.0
        self.episode_push_count[env_ids] = 0.0
        self.last_push_velocity[env_ids] = 0.0
        self.episode_touchdowns[env_ids] = 0.0
        self.episode_alternating_touchdowns[env_ids] = 0.0
        self.episode_repeated_touchdowns[env_ids] = 0.0
        self.previous_policy_foot_contact[env_ids] = False
        self.last_touchdown_foot[env_ids] = -1
        self.reset_buf[env_ids] = False
        self.timeout_buf[env_ids] = False
        self.contact_tracker.reset(env_ids)
        self.foot_contact_duty.reset(env_ids)
        self._resample_commands(env_ids)
        self._sync_kinematics()

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._reset_state(env_ids)
        policy, critic = self._observation_terms(self._current_state())
        self._actor_observation = self.policy_history.fill(policy, env_ids)
        self._critic_observation = self.critic_history.fill(critic, env_ids)
        return self._actor_observation

    def _simulate_uncaptured(self) -> None:
        for _ in range(self.cfg.simulation.decimation):
            if self.cfg.randomization.actuator_enabled:
                self._launch_limited_joint_targets()
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.contact_tracker.update(self.state_0.body_q)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.physics_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture_simulation_graph(self) -> None:
        """Capture one 50 Hz policy interval on the SynReal CUDA stream."""

        if self.cfg.simulation.decimation % 2 != 0:
            raise RuntimeError("CUDA graph capture currently requires even decimation")
        self.simulation_graph = None
        torch.cuda.current_stream(self.device).synchronize()
        wp.synchronize_stream(self.stream)
        with (
            wp.ScopedStream(self.stream),
            wp.ScopedCapture(stream=self.stream) as capture,
        ):
            self._simulate_uncaptured()
        self.simulation_graph = capture.graph

    def _simulate(self) -> None:
        if self.simulation_graph is None:
            self._simulate_uncaptured()
        else:
            wp.capture_launch(self.simulation_graph, stream=self.stream)

    def _update_curriculum(self, done: torch.Tensor) -> None:
        if not self.cfg.commands.curriculum_enabled:
            return
        if self.common_step_counter % self.max_episode_length != 0:
            return
        if not torch.any(done):
            return
        mean_rate = self.episode_track_reward[done].mean() / self.cfg.simulation.episode_length_s
        if float(mean_rate) <= self.cfg.commands.curriculum_success_rate:
            return
        increment = self.cfg.commands.curriculum_increment
        limits = {
            "x": self.cfg.commands.limit_lin_vel_x,
            "y": self.cfg.commands.limit_lin_vel_y,
        }
        for key in ("x", "y"):
            low, high = self.command_ranges[key]
            limit_low, limit_high = limits[key]
            self.command_ranges[key] = [
                max(limit_low, low - increment),
                min(limit_high, high + increment),
            ]

    def _update_commands(self, done_ids: torch.Tensor) -> None:
        self.command_steps += 1
        if self.cfg.auto_reset:
            self.command_steps[done_ids] = 0
        interval = round(self.cfg.commands.resampling_time_s / self.step_dt)
        due = torch.nonzero(self.command_steps >= interval, as_tuple=False).squeeze(-1)
        self._resample_commands(due)

    def _apply_random_push(self) -> None:
        config = self.cfg.randomization
        self.last_push_velocity.zero_()
        if not config.push_enabled:
            return
        randomization = self.episode_randomization
        due = self.episode_length_buf >= randomization.next_push_step
        if not torch.any(due):
            return
        env_ids = torch.nonzero(due, as_tuple=False).squeeze(-1)
        velocity = uniform(
            (env_ids.numel(), 2),
            config.push_velocity,
            self.device,
            self.generator,
        )
        self.joint_qd_0[env_ids, :2] += velocity
        self.joint_qd_1[env_ids, :2] = self.joint_qd_0[env_ids, :2]
        self.last_push_velocity[env_ids] = velocity
        self.episode_push_count[env_ids] += 1.0
        randomization.resample_push(
            config,
            env_ids,
            self.step_dt,
            self.generator,
            current_step=self.episode_length_buf[env_ids],
        )
        self._sync_kinematics()

    def _apply_periodic_push(self) -> None:
        if not self.cfg.periodic_push_enabled:
            return
        interval = round(5.0 / self.step_dt)
        if self.common_step_counter == 0 or self.common_step_counter % interval != 0:
            return
        root_velocity = self.joint_qd_0[:, :6]
        root_velocity[:, 0] = self._uniform(self.num_envs, -0.5, 0.5)
        root_velocity[:, 1] = self._uniform(self.num_envs, -0.5, 0.5)
        self.joint_qd_1[:, :6] = root_velocity
        self._sync_kinematics()

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"expected actions {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}")
        actions = actions.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(actions).all():
            actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)
        applied_actions = self._delayed_action(actions)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self.applied_actions.copy_(applied_actions)
        randomized_default = self._randomized_default_joint_position()
        targets = randomized_default + self.cfg.action_scale * applied_actions
        self.desired_targets.copy_(targets)
        self.control_targets.copy_(self.default_control_targets)
        self.control_targets[:, self.target_indices] = targets
        self._apply_random_push()

        torch.cuda.current_stream(self.device).synchronize()
        with wp.ScopedStream(self.stream):
            self._simulate()
            newton.eval_ik(
                self.model,
                self.state_0,
                self.state_0.joint_q,
                self.state_0.joint_qd,
            )
        wp.synchronize_stream(self.stream)
        self.sim_time += self.step_dt
        self.common_step_counter += 1
        self.episode_length_buf += 1

        state = self._current_state()
        finite = (
            torch.isfinite(self.joint_q_0).all(dim=-1)
            & torch.isfinite(self.joint_qd_0).all(dim=-1)
            & torch.isfinite(self.body_q_0).all(dim=(1, 2))
            & torch.isfinite(self.body_qd_0).all(dim=(1, 2))
        )
        tilt = torch.acos((-state["projected_gravity"][:, 2]).clamp(-1.0, 1.0))
        self.reset_buf = (state["root_position"][:, 2] < 0.2) | (tilt > 0.8) | ~finite
        self.timeout_buf = self.episode_length_buf >= self.max_episode_length
        done = self.reset_buf | self.timeout_buf
        raw_torque = self.kp * (targets - state["joint_position"]) - self.kd * state["joint_velocity"]
        estimated_torque = torch.clamp(
            raw_torque,
            min=-self.effort_limit,
            max=self.effort_limit,
        )
        foot_contact = self.contact_tracker.foot_contact
        contact_count = foot_contact.sum(dim=-1)
        touchdowns = foot_contact & ~self.previous_policy_foot_contact
        touchdown_count = touchdowns.sum(dim=-1)
        touchdown_foot = touchdowns.to(torch.int64).argmax(dim=-1)
        single_touchdown = touchdown_count == 1
        repeated_touchdown = (
            single_touchdown & (self.last_touchdown_foot >= 0) & (touchdown_foot == self.last_touchdown_foot)
        )
        alternating = single_touchdown & (self.last_touchdown_foot >= 0) & (touchdown_foot != self.last_touchdown_foot)
        foot_duty = self.foot_contact_duty.update(foot_contact)
        foot_duty_imbalance = torch.abs(foot_duty[:, 0] - foot_duty[:, 1])
        self.episode_touchdowns += single_touchdown.float()
        self.episode_alternating_touchdowns += alternating.float()
        self.episode_repeated_touchdowns += repeated_touchdown.float()
        self.last_touchdown_foot[single_touchdown] = touchdown_foot[single_touchdown]
        self.previous_policy_foot_contact.copy_(foot_contact)
        reward, reward_terms = compute_reward(
            RewardInputs(
                commands=self.commands,
                root_position=state["root_position"],
                root_quaternion=state["root_quaternion"],
                root_linear_velocity_world=state["root_velocity_world"][:, :3],
                root_linear_velocity_body=state["linear_velocity"],
                root_angular_velocity_body=state["angular_velocity"],
                projected_gravity=state["projected_gravity"],
                joint_position=state["joint_position"],
                default_joint_position=randomized_default,
                joint_velocity=state["joint_velocity"],
                previous_joint_velocity=self.previous_joint_velocity,
                actions=self.actions,
                previous_actions=self.previous_actions,
                estimated_torque=estimated_torque,
                soft_joint_lower=self.soft_joint_lower,
                soft_joint_upper=self.soft_joint_upper,
                foot_position_world=state["foot_position"],
                foot_linear_velocity_world=state["foot_velocity"],
                foot_contact=self.contact_tracker.foot_contact,
                recent_foot_contact=self.contact_tracker.recent_foot_contact,
                recent_undesired_contact=(self.contact_tracker.recent_undesired_contact),
                foot_duty_imbalance=foot_duty_imbalance,
                repeated_touchdown=repeated_touchdown.float(),
                episode_steps=self.episode_length_buf,
                terminated=self.reset_buf,
            ),
            self.cfg.rewards,
            self.step_dt,
            self.arm_indices,
            self.waist_indices,
            self.leg_deviation_indices,
        )
        reward = torch.nan_to_num(reward, nan=-10.0, posinf=10.0, neginf=-10.0)
        self.previous_joint_velocity.copy_(state["joint_velocity"])
        self.episode_return += reward
        self.episode_track_reward += reward_terms["track_lin_vel_xy"]
        self.episode_sample_steps += 1
        self.episode_forward_velocity_sum += state["linear_velocity"][:, 0]
        self.episode_velocity_error_sum += torch.abs(self.commands[:, 0] - state["linear_velocity"][:, 0])
        lateral_velocity_error = state["linear_velocity"][:, 1] - self.commands[:, 1]
        yaw_rate_error = state["angular_velocity"][:, 2] - self.commands[:, 2]
        self.episode_lateral_velocity_sum += state["linear_velocity"][:, 1]
        self.episode_lateral_velocity_abs_sum += torch.abs(lateral_velocity_error)
        self.episode_yaw_rate_error_abs_sum += torch.abs(yaw_rate_error)
        self.episode_lateral_displacement += lateral_velocity_error * self.step_dt
        self.episode_yaw_error_integral += yaw_rate_error * self.step_dt
        self.episode_flight_steps += (contact_count == 0).float()
        self.episode_single_support_steps += (contact_count == 1).float()
        self.episode_left_air_steps += (~foot_contact[:, 0]).float()
        self.episode_right_air_steps += (~foot_contact[:, 1]).float()
        self.episode_action_over_three += (self.actions.abs() > 3.0).float().mean(dim=-1)
        self.episode_max_abs_action = torch.maximum(self.episode_max_abs_action, self.actions.abs().max(dim=-1).values)
        self.episode_max_abs_arm_action = torch.maximum(
            self.episode_max_abs_arm_action,
            self.actions.index_select(1, self.arm_indices).abs().max(dim=-1).values,
        )
        self.episode_torque_saturation += (estimated_torque.abs() >= (self.effort_limit - 1.0e-5)).float().mean(dim=-1)

        self._update_curriculum(done)
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        extras: dict[str, Any] = {
            "time_outs": self.timeout_buf.clone(),
            "terminated": self.reset_buf.clone(),
            "observations": {},
            "reward_terms": {name: value.mean() for name, value in reward_terms.items()},
            "metrics": {
                "geometry_contact/left": (self.contact_tracker.foot_contact[:, 0].float().mean()),
                "geometry_contact/right": (self.contact_tracker.foot_contact[:, 1].float().mean()),
                "geometry_contact/undesired_bodies": (
                    self.contact_tracker.recent_undesired_contact.float().sum(dim=-1).mean()
                ),
                "contact_overflow": self.contact_tracker.overflow.float(),
                "joint_velocity_limit_fraction": ((state["joint_velocity"].abs() > self.velocity_limit).float().mean()),
                "locomotion/body_vx": state["linear_velocity"][:, 0].mean(),
                "locomotion/vx_abs_error": torch.abs(self.commands[:, 0] - state["linear_velocity"][:, 0]).mean(),
                "locomotion/body_vy": state["linear_velocity"][:, 1].mean(),
                "locomotion/vy_abs_error": torch.abs(lateral_velocity_error).mean(),
                "locomotion/yaw_rate_abs_error": torch.abs(yaw_rate_error).mean(),
                "locomotion/flight_fraction": (contact_count == 0).float().mean(),
                "locomotion/single_support_fraction": (contact_count == 1).float().mean(),
                "gait/left_contact_duty": foot_duty[:, 0].mean(),
                "gait/right_contact_duty": foot_duty[:, 1].mean(),
                "gait/contact_duty_imbalance": foot_duty_imbalance.mean(),
                "gait/repeated_touchdown_fraction": (
                    repeated_touchdown.float().sum() / single_touchdown.float().sum().clamp_min(1.0)
                ),
                "action/mean_abs": self.actions.abs().mean(),
                "action/max_abs": self.actions.abs().max(),
                "action/over_3_fraction": (self.actions.abs() > 3.0).float().mean(),
                "action/arm_over_3_fraction": (self.actions.index_select(1, self.arm_indices).abs() > 3.0)
                .float()
                .mean(),
                "action/delay_delta_mean_abs": (self.actions - self.applied_actions).abs().mean(),
                "torque/saturation_fraction": (estimated_torque.abs() >= (self.effort_limit - 1.0e-5)).float().mean(),
                "randomization/stiffness_scale_mean": (self.episode_randomization.stiffness_scale.mean()),
                "randomization/damping_scale_mean": (self.episode_randomization.damping_scale.mean()),
                "randomization/motor_strength_mean": (self.episode_randomization.motor_strength.mean()),
                "randomization/joint_zero_offset_abs_mean": (self.episode_randomization.joint_zero_offset.abs().mean()),
                "randomization/delay_ticks_mean": (self.episode_randomization.delay_ticks.float().mean()),
                "push/applied_fraction": (self.last_push_velocity.abs().sum(dim=-1) > 0.0).float().mean(),
                "push/velocity_abs_mean": self.last_push_velocity.abs().mean(),
            },
        }
        if done_ids.numel() > 0:
            sample_steps = self.episode_sample_steps[done_ids].clamp_min(1).float()
            touchdowns = self.episode_touchdowns[done_ids]
            extras["episode"] = {
                "return": self.episode_return[done_ids].mean(),
                "length": self.episode_length_buf[done_ids].float().mean(),
                "track_lin_vel_rate": (
                    self.episode_track_reward[done_ids].mean() / self.cfg.simulation.episode_length_s
                ),
                "mean_body_vx": (self.episode_forward_velocity_sum[done_ids] / sample_steps).mean(),
                "mean_vx_abs_error": (self.episode_velocity_error_sum[done_ids] / sample_steps).mean(),
                "mean_body_vy": (self.episode_lateral_velocity_sum[done_ids] / sample_steps).mean(),
                "mean_vy_abs_error": (self.episode_lateral_velocity_abs_sum[done_ids] / sample_steps).mean(),
                "mean_yaw_rate_abs_error": (self.episode_yaw_rate_error_abs_sum[done_ids] / sample_steps).mean(),
                "forward_displacement": (self.episode_forward_velocity_sum[done_ids] * self.step_dt).mean(),
                "lateral_error_displacement": self.episode_lateral_displacement[done_ids].mean(),
                "integrated_yaw_error": self.episode_yaw_error_integral[done_ids].mean(),
                "absolute_yaw_error_per_forward_meter": (
                    self.episode_yaw_error_integral[done_ids].abs()
                    / (self.episode_forward_velocity_sum[done_ids].abs() * self.step_dt).clamp_min(0.1)
                ).mean(),
                "flight_fraction": (self.episode_flight_steps[done_ids] / sample_steps).mean(),
                "single_support_fraction": (self.episode_single_support_steps[done_ids] / sample_steps).mean(),
                "left_air_fraction": (self.episode_left_air_steps[done_ids] / sample_steps).mean(),
                "right_air_fraction": (self.episode_right_air_steps[done_ids] / sample_steps).mean(),
                "foot_air_fraction_imbalance": torch.abs(
                    self.episode_left_air_steps[done_ids] - self.episode_right_air_steps[done_ids]
                )
                .div(sample_steps)
                .mean(),
                "alternating_touchdown_fraction": (
                    self.episode_alternating_touchdowns[done_ids] / touchdowns.clamp_min(1.0)
                ).mean(),
                "repeated_touchdown_fraction": (
                    self.episode_repeated_touchdowns[done_ids] / touchdowns.clamp_min(1.0)
                ).mean(),
                "action_over_3_fraction": (self.episode_action_over_three[done_ids] / sample_steps).mean(),
                "max_abs_action": self.episode_max_abs_action[done_ids].mean(),
                "max_abs_arm_action": self.episode_max_abs_arm_action[done_ids].mean(),
                "torque_saturation_fraction": (self.episode_torque_saturation[done_ids] / sample_steps).mean(),
                "push_count": self.episode_push_count[done_ids].mean(),
                "stiffness_scale_mean": self.episode_randomization.stiffness_scale[done_ids].mean(),
                "damping_scale_mean": self.episode_randomization.damping_scale[done_ids].mean(),
                "motor_strength_mean": self.episode_randomization.motor_strength[done_ids].mean(),
                "joint_zero_offset_abs_mean": (self.episode_randomization.joint_zero_offset[done_ids].abs().mean()),
                "delay_ticks_mean": self.episode_randomization.delay_ticks[done_ids].float().mean(),
            }
        if self.cfg.auto_reset and done_ids.numel() > 0:
            self._reset_state(done_ids)
        self._update_commands(done_ids)
        self._apply_periodic_push()
        self._update_observations(done_ids if self.cfg.auto_reset else None)
        extras["observations"] = {"critic": self._critic_observation}
        self.extras = extras
        self.render()
        return self._actor_observation, reward, done, extras

    @torch.no_grad()
    def set_command(self, x: float, y: float, yaw: float) -> None:
        """Set a deterministic command for evaluation."""

        self.commands[:, 0] = x
        self.commands[:, 1] = y
        self.commands[:, 2] = yaw
        self.command_steps.zero_()

    def render(self) -> None:
        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
