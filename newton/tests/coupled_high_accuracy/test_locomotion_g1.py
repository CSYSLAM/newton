"""Pure-Torch contract tests for the independent G1 locomotion task."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Training-only dependency, not part of Newton's environment.
torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "coupled_high_accuracy"))

from locomotion_g1.config import (
    CRITIC_OBSERVATION_DIM,
    DEFAULT_JOINT_POS,
    JOINT_EFFORT_LIMIT,
    JOINT_KD,
    JOINT_KP,
    JOINT_NAMES,
    POLICY_OBSERVATION_DIM,
    G1EnvConfig,
    RewardConfig,
    make_effort_bootstrap_config,
    make_robustness_config,
    make_straight_symmetry_config,
    make_velocity_bootstrap_config,
)
from locomotion_g1.env import (
    SynRealG1LocomotionEnv,
    _randomize_robot_friction,
    sample_velocity_commands,
)
from locomotion_g1.math_ops import FootContactDutyWindow, TermHistory
from locomotion_g1.mujoco_eval import TermHistory as MuJoCoTermHistory
from locomotion_g1.mujoco_eval import load_run_settings, resolve_training_asset
from locomotion_g1.observations import (
    critic_terms,
    policy_terms,
    root_terms,
)
from locomotion_g1.play import (
    create_parser as create_play_parser,
)
from locomotion_g1.play import (
    make_fixed_command_config,
    make_play_config,
)
from locomotion_g1.randomization import EpisodeRandomization, delayed_action
from locomotion_g1.rewards import RewardInputs, compute_reward
from locomotion_g1.train import load_actor_only


def test_g1_contract_dimensions_and_actuators() -> None:
    assert len(JOINT_NAMES) == 29
    assert len(set(JOINT_NAMES)) == 29
    assert len(DEFAULT_JOINT_POS) == 29
    assert POLICY_OBSERVATION_DIM == 480
    assert CRITIC_OBSERVATION_DIM == 495
    config = G1EnvConfig()
    assert config.simulation.contact_stiffness == 50000.0
    assert config.simulation.contact_damping == 500.0
    assert config.simulation.contact_friction_gain == 1000.0
    assert config.simulation.ground_friction == 1.0
    assert config.simulation.constraint_alpha == 200.0
    assert config.simulation.joint_constraint_primal_iterations == 2
    assert config.simulation.nonlinear_iterations == 30
    assert config.simulation.linear_iterations == 10
    assert config.randomization.friction_range == (0.3, 1.0)
    assert config.randomization.friction_buckets == 64
    assert not hasattr(config.randomization, "mass_range")
    knee = JOINT_NAMES.index("left_knee_joint")
    wrist_yaw = JOINT_NAMES.index("right_wrist_yaw_joint")
    waist_yaw = JOINT_NAMES.index("waist_yaw_joint")
    assert (JOINT_KP[knee], JOINT_KD[knee], JOINT_EFFORT_LIMIT[knee]) == (
        150.0,
        4.0,
        139.0,
    )
    assert JOINT_EFFORT_LIMIT[wrist_yaw] == 5.0
    assert (JOINT_KP[waist_yaw], JOINT_KD[waist_yaw]) == (200.0, 5.0)


def test_velocity_bootstrap_preset_is_explicit_and_preserves_action_abi() -> None:
    original = G1EnvConfig()
    bootstrap = make_velocity_bootstrap_config(original)

    assert original.commands.initial_lin_vel_x == (-0.1, 0.1)
    assert bootstrap.commands.initial_lin_vel_x == (0.35, 0.65)
    assert bootstrap.commands.initial_lin_vel_y == (0.0, 0.0)
    assert bootstrap.commands.initial_ang_vel_z == (0.0, 0.0)
    assert bootstrap.commands.standing_fraction == 0.0
    assert not bootstrap.commands.curriculum_enabled
    assert not bootstrap.periodic_push_enabled
    assert bootstrap.rewards.track_lin_vel_xy == 4.0
    assert bootstrap.rewards.gait == 0.0
    assert bootstrap.rewards.supported_swing_clearance
    assert POLICY_OBSERVATION_DIM == 480


def test_straight_symmetry_presets_are_staged_and_preserve_action_abi() -> None:
    stage1 = make_straight_symmetry_config(G1EnvConfig(), stage=1)
    stage2 = make_straight_symmetry_config(G1EnvConfig(), stage=2)

    assert stage1.commands.initial_lin_vel_x == (0.25, 0.70)
    assert stage1.commands.slow_fraction == 0.10
    assert stage1.commands.turn_fraction == 0.0
    assert stage1.commands.standing_fraction == 0.05
    assert stage2.commands.initial_lin_vel_x == (0.20, 0.75)
    assert stage2.commands.slow_fraction == 0.15
    assert stage2.commands.turn_fraction == 0.10
    assert stage2.commands.lateral_fraction == 0.05
    assert stage2.commands.standing_fraction == 0.10
    assert stage1.commands.mirror_sampled_commands
    assert stage1.randomization.correlate_friction_per_env
    assert not stage1.periodic_push_enabled
    assert stage1.rewards.track_ang_vel_z == 1.5
    assert stage1.rewards.gait == 0.0
    assert stage1.rewards.foot_duty_imbalance == -0.25
    assert POLICY_OBSERVATION_DIM == 480


def test_robustness_presets_expand_actuator_latency_and_push_ranges() -> None:
    stage1 = make_robustness_config(G1EnvConfig(), stage=1)
    stage2 = make_robustness_config(G1EnvConfig(), stage=2)
    stage3 = make_robustness_config(G1EnvConfig(), stage=3)

    assert stage1.commands.sampling_mode == "straight_symmetry"
    assert stage1.randomization.actuator_enabled
    assert stage1.randomization.stiffness_scale == (0.95, 1.05)
    assert stage1.randomization.policy_delay_ticks == (0, 1)
    assert not stage1.randomization.push_enabled
    assert stage2.randomization.motor_strength == (0.90, 1.10)
    assert stage2.randomization.joint_zero_offset == (-0.015, 0.015)
    assert stage2.randomization.policy_delay_ticks == (0, 2)
    assert stage3.randomization.stiffness_scale == (0.80, 1.20)
    assert stage3.randomization.push_enabled
    assert stage3.randomization.push_velocity == (-0.20, 0.20)
    assert not hasattr(stage3.randomization, "mass_range")
    assert POLICY_OBSERVATION_DIM == 480


def test_effort_bootstrap_enables_only_nominal_actuator_envelope() -> None:
    config = make_effort_bootstrap_config(G1EnvConfig())

    assert config.randomization.actuator_enabled
    assert config.randomization.stiffness_scale == (1.0, 1.0)
    assert config.randomization.damping_scale == (1.0, 1.0)
    assert config.randomization.motor_strength == (1.0, 1.0)
    assert config.randomization.joint_zero_offset == (0.0, 0.0)
    assert config.randomization.policy_delay_ticks == (0, 0)
    assert not config.randomization.push_enabled
    assert config.commands.initial_lin_vel_x == (0.35, 0.65)
    assert config.commands.standing_fraction == 0.0
    assert not config.periodic_push_enabled


def test_episode_randomization_samples_requested_ranges() -> None:
    config = make_robustness_config(G1EnvConfig(), stage=2).randomization
    randomization = EpisodeRandomization.create(
        config,
        num_envs=32,
        num_actions=29,
        policy_dt=0.02,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.all(randomization.stiffness_scale >= 0.90)
    assert torch.all(randomization.stiffness_scale <= 1.10)
    assert torch.all(randomization.motor_strength >= 0.90)
    assert torch.all(randomization.motor_strength <= 1.10)
    assert torch.all(randomization.joint_zero_offset >= -0.015)
    assert torch.all(randomization.joint_zero_offset <= 0.015)
    assert torch.all((randomization.delay_ticks >= 0) & (randomization.delay_ticks <= 2))


def test_g1_action_delay_sequence_for_zero_one_and_two_ticks() -> None:
    queue = torch.zeros((3, 3, 1))
    delay_ticks = torch.tensor([0, 1, 2])

    first = delayed_action(queue, delay_ticks, torch.full((3, 1), 1.0))
    second = delayed_action(queue, delay_ticks, torch.full((3, 1), 2.0))
    third = delayed_action(queue, delay_ticks, torch.full((3, 1), 3.0))

    assert torch.equal(first[:, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(second[:, 0], torch.tensor([2.0, 1.0, 0.0]))
    assert torch.equal(third[:, 0], torch.tensor([3.0, 2.0, 1.0]))


def test_random_push_updates_due_roots_and_reschedules_each_environment() -> None:
    config = make_robustness_config(G1EnvConfig(), stage=3).randomization
    env = object.__new__(SynRealG1LocomotionEnv)
    env.cfg = SimpleNamespace(randomization=config)
    env.device = torch.device("cpu")
    env.generator = torch.Generator().manual_seed(9)
    env.step_dt = 0.02
    env.episode_length_buf = torch.tensor([0, 10, 20])
    env.joint_qd_0 = torch.zeros((3, 35))
    env.joint_qd_1 = torch.zeros((3, 35))
    env.last_push_velocity = torch.zeros((3, 2))
    env.episode_push_count = torch.zeros(3)
    env.episode_randomization = EpisodeRandomization.create(
        config,
        num_envs=3,
        num_actions=29,
        policy_dt=env.step_dt,
        device=env.device,
        generator=env.generator,
    )
    env.episode_randomization.next_push_step[:] = torch.tensor([0, 11, 20])
    env._sync_kinematics = lambda: None

    env._apply_random_push()

    due = torch.tensor([0, 2])
    assert torch.equal(env.joint_qd_0[:, :2], env.joint_qd_1[:, :2])
    assert torch.equal(env.joint_qd_0[due, :2], env.last_push_velocity[due])
    assert torch.count_nonzero(env.joint_qd_0[due, :2]) > 0
    assert torch.equal(env.joint_qd_0[1, :2], torch.zeros(2))
    assert torch.equal(env.episode_push_count, torch.tensor([1.0, 0.0, 1.0]))
    assert torch.all(env.episode_randomization.next_push_step[due] > env.episode_length_buf[due])


def test_joint_zero_offset_centers_observation_and_control_default() -> None:
    env = object.__new__(SynRealG1LocomotionEnv)
    offset = torch.linspace(-0.02, 0.02, 29).unsqueeze(0)
    env.default_joint_position = torch.tensor(DEFAULT_JOINT_POS).unsqueeze(0)
    env.episode_randomization = SimpleNamespace(joint_zero_offset=offset)

    randomized_default = env._randomized_default_joint_position()
    assert torch.equal(randomized_default, env.default_joint_position + offset)


def test_straight_symmetry_command_sampling_pairs_lateral_and_yaw_signs() -> None:
    config = make_straight_symmetry_config(G1EnvConfig(), stage=2).commands
    commands = sample_velocity_commands(
        config,
        512,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    first, mirrored = commands.chunk(2)
    assert torch.equal(first[:, 0], mirrored[:, 0])
    assert torch.equal(first[:, 1], -mirrored[:, 1])
    assert torch.equal(first[:, 2], -mirrored[:, 2])
    assert torch.all((commands[:, 0] >= 0.0) & (commands[:, 0] <= 0.75))


def test_straight_symmetry_friction_is_correlated_within_each_environment() -> None:
    class Builder:
        shape_material_mu = [0.0] * 12

    builder = Builder()
    summary = _randomize_robot_friction(
        builder,
        num_envs=3,
        shapes_per_env=4,
        friction_range=(0.3, 1.0),
        num_buckets=64,
        seed=42,
        correlate_per_env=True,
    )
    coefficients = torch.tensor(builder.shape_material_mu).view(3, 4)

    assert torch.equal(coefficients, coefficients[:, :1].expand_as(coefficients))
    assert summary["correlated_per_env"] is True


def test_foot_contact_duty_window_uses_only_available_samples() -> None:
    window = FootContactDutyWindow(2, 4, device=torch.device("cpu"))
    first = window.update(torch.tensor([[True, False], [False, True]]))
    second = window.update(torch.tensor([[True, False], [True, False]]))

    assert torch.equal(first, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(second, torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
    window.reset(torch.tensor([0]))
    third = window.update(torch.tensor([[False, True], [True, False]]))
    assert torch.equal(third[0], torch.tensor([0.0, 1.0]))


def test_play_disables_periodic_pushes_by_default() -> None:
    parser = create_play_parser()
    assert not parser.parse_args([]).enable_periodic_push
    assert not parser.parse_args([]).effort_bootstrap
    assert parser.parse_args(["--enable-periodic-push"]).enable_periodic_push
    assert parser.parse_args(["--effort-bootstrap"]).effort_bootstrap
    assert parser.parse_args(["--seconds", "0"]).seconds == 0.0


def test_play_command_remains_fixed_when_environment_resamples() -> None:
    config = make_fixed_command_config(G1EnvConfig(), 0.5, -0.1, 0.2)
    commands = sample_velocity_commands(
        config.commands,
        64,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(commands, torch.tensor([[0.5, -0.1, 0.2]]).expand(64, -1))
    assert config.commands.standing_fraction == 0.0
    assert not config.commands.curriculum_enabled


def test_play_effort_bootstrap_reproduces_nominal_limiter_without_pushes() -> None:
    config = make_play_config(G1EnvConfig(), 0.3, 0.0, 0.0, effort_bootstrap=True)

    assert config.randomization.actuator_enabled
    assert config.randomization.stiffness_scale == (1.0, 1.0)
    assert config.randomization.damping_scale == (1.0, 1.0)
    assert config.randomization.motor_strength == (1.0, 1.0)
    assert config.randomization.joint_zero_offset == (0.0, 0.0)
    assert config.randomization.policy_delay_ticks == (0, 0)
    assert not config.randomization.push_enabled
    assert not config.periodic_push_enabled
    assert config.commands.initial_lin_vel_x == (0.3, 0.3)


def test_actor_only_initialization_does_not_replace_critic_or_std(tmp_path: Path) -> None:
    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = torch.nn.Linear(3, 2)
            self.critic = torch.nn.Linear(4, 1)
            self.std = torch.nn.Parameter(torch.full((2,), 0.5))

    source = Policy()
    target = Policy()
    with torch.no_grad():
        source.actor.weight.fill_(2.0)
        source.actor.bias.fill_(3.0)
    critic_before = {name: value.clone() for name, value in target.critic.state_dict().items()}
    std_before = target.std.detach().clone()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)

    loaded = load_actor_only(target, checkpoint)

    assert loaded == 2
    assert torch.equal(target.actor.weight, source.actor.weight)
    assert torch.equal(target.actor.bias, source.actor.bias)
    for name, value in target.critic.state_dict().items():
        assert torch.equal(value, critic_before[name])
    assert torch.equal(target.std, std_before)


def test_term_history_is_oldest_to_newest_and_term_major() -> None:
    history = TermHistory(1, (1, 2), 3, device=torch.device("cpu"))
    history.fill((torch.tensor([[0.0]]), torch.tensor([[10.0, 20.0]])), torch.tensor([0]))
    history.update((torch.tensor([[1.0]]), torch.tensor([[11.0, 21.0]])))
    output = history.update((torch.tensor([[2.0]]), torch.tensor([[12.0, 22.0]])))
    assert output.tolist() == [[0.0, 1.0, 2.0, 10.0, 20.0, 11.0, 21.0, 12.0, 22.0]]


def test_mujoco_history_matches_training_layout() -> None:
    import numpy as np

    history = MuJoCoTermHistory((1, 2), 3)
    history.fill((np.asarray([0.0]), np.asarray([10.0, 20.0])))
    history.update((np.asarray([1.0]), np.asarray([11.0, 21.0])))
    output = history.update((np.asarray([2.0]), np.asarray([12.0, 22.0])))
    assert output.tolist() == [0.0, 1.0, 2.0, 10.0, 20.0, 11.0, 21.0, 12.0, 22.0]


def test_mujoco_refines_integration_without_changing_policy_period(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    snapshot = {
        "environment": "synreal_g1_locomotion",
        "policy_observation_dim": POLICY_OBSERVATION_DIM,
        "action_dim": len(JOINT_NAMES),
        "environment_config": {
            "simulation": {
                "dt": 0.005,
                "decimation": 4,
                "gravity": -9.81,
                "contact_stiffness": 50000.0,
                "contact_damping": 500.0,
                "contact_friction_gain": 1000.0,
                "ground_friction": 1.0,
                "joint_armature_approximation": 0.01,
                "nonlinear_iterations": 30,
            },
            "randomization": {"friction_range": [0.3, 1.0]},
            "action_scale": 0.25,
            "initial_base_height": 0.8,
            "mesh_approximation": "convex_hull",
            "enable_self_collisions": True,
        },
        "rsl_rl": {"policy": {"actor_hidden_dims": [512, 256, 128], "activation": "elu"}},
    }
    (tmp_path / "environment.json").write_text(json.dumps(snapshot), encoding="utf-8")

    settings = load_run_settings(checkpoint, robot_friction=None)

    assert settings.training_dt == 0.005
    assert settings.training_decimation == 4
    assert settings.dt == 0.0025
    assert settings.decimation == 8
    assert settings.policy_dt == settings.training_dt * settings.training_decimation


def test_mujoco_requires_the_asset_recorded_by_training(tmp_path: Path) -> None:
    import hashlib

    import pytest

    asset = tmp_path / "g1.usd"
    asset.write_bytes(b"training asset")
    snapshot = {
        "resolved_asset": str(asset),
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "environment_config": {"asset_path": None},
    }
    resolved, source, verified = resolve_training_asset(snapshot, None)
    assert resolved == asset.resolve()
    assert source == "training_resolved_asset"
    assert verified

    asset.write_bytes(b"modified asset")
    with pytest.raises(ValueError, match="differs from the training snapshot"):
        resolve_training_asset(snapshot, None)


def test_observation_terms_match_unitree_scaling() -> None:
    from locomotion_g1.config import ObservationNoiseConfig

    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    root_velocity = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    linear, angular, gravity = root_terms(quaternion, root_velocity)
    joint_position = torch.tensor(DEFAULT_JOINT_POS).unsqueeze(0)
    joint_velocity = torch.ones((1, 29)) * 2.0
    actions = torch.ones((1, 29)) * 0.25
    command = torch.tensor([[0.3, -0.1, 0.2]])
    policy = policy_terms(
        angular,
        gravity,
        command,
        joint_position,
        joint_position,
        joint_velocity,
        actions,
        ObservationNoiseConfig(enabled=False),
        torch.Generator().manual_seed(1),
    )
    critic = critic_terms(
        linear,
        angular,
        gravity,
        command,
        joint_position,
        joint_position,
        joint_velocity,
        actions,
    )
    assert torch.allclose(policy[0], torch.tensor([[0.8, 1.0, 1.2]]))
    assert torch.allclose(policy[1], torch.tensor([[0.0, 0.0, -1.0]]))
    assert torch.count_nonzero(policy[3]) == 0
    assert torch.allclose(policy[4], torch.full((1, 29), 0.1))
    assert torch.equal(critic[0], linear)
    assert sum(term.shape[1] for term in policy) == 96
    assert sum(term.shape[1] for term in critic) == 99


def test_reward_matches_dt_scaled_upstream_terms() -> None:
    batch = 1
    zeros_29 = torch.zeros((batch, 29))
    default = torch.tensor(DEFAULT_JOINT_POS).unsqueeze(0)
    command = torch.tensor([[0.2, 0.0, 0.0]])
    reward, terms = compute_reward(
        RewardInputs(
            commands=command,
            root_position=torch.tensor([[0.0, 0.0, 0.78]]),
            root_quaternion=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            root_linear_velocity_world=torch.tensor([[0.2, 0.0, 0.0]]),
            root_linear_velocity_body=torch.tensor([[0.2, 0.0, 0.0]]),
            root_angular_velocity_body=torch.zeros((batch, 3)),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            joint_position=default,
            default_joint_position=default,
            joint_velocity=zeros_29,
            previous_joint_velocity=zeros_29,
            actions=zeros_29,
            previous_actions=zeros_29,
            estimated_torque=zeros_29,
            soft_joint_lower=torch.full((batch, 29), -10.0),
            soft_joint_upper=torch.full((batch, 29), 10.0),
            foot_position_world=torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]]),
            foot_linear_velocity_world=torch.zeros((batch, 2, 3)),
            foot_contact=torch.tensor([[True, False]]),
            recent_foot_contact=torch.tensor([[True, False]]),
            recent_undesired_contact=torch.zeros((batch, 25), dtype=torch.bool),
            foot_duty_imbalance=torch.zeros(batch),
            repeated_touchdown=torch.zeros(batch),
            episode_steps=torch.tensor([8]),
            terminated=torch.tensor([False]),
        ),
        RewardConfig(),
        0.02,
        torch.arange(15, 29),
        torch.arange(12, 15),
        torch.tensor([1, 2, 7, 8]),
    )
    assert torch.allclose(terms["track_lin_vel_xy"], torch.tensor([0.02]))
    assert torch.allclose(terms["gait"], torch.tensor([0.02]))
    assert torch.allclose(reward, torch.tensor([0.073]), atol=1.0e-6)


def test_straight_symmetry_reward_penalizes_drift_and_foot_imbalance() -> None:
    batch = 1
    zeros_29 = torch.zeros((batch, 29))
    default = torch.tensor(DEFAULT_JOINT_POS).unsqueeze(0)
    _reward, terms = compute_reward(
        RewardInputs(
            commands=torch.tensor([[0.2, 0.0, 0.0]]),
            root_position=torch.tensor([[0.0, 0.0, 0.78]]),
            root_quaternion=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            root_linear_velocity_world=torch.tensor([[0.2, 0.1, 0.0]]),
            root_linear_velocity_body=torch.tensor([[0.2, 0.1, 0.0]]),
            root_angular_velocity_body=torch.tensor([[0.0, 0.0, 0.2]]),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            joint_position=default,
            default_joint_position=default,
            joint_velocity=zeros_29,
            previous_joint_velocity=zeros_29,
            actions=zeros_29,
            previous_actions=zeros_29,
            estimated_torque=zeros_29,
            soft_joint_lower=torch.full((batch, 29), -10.0),
            soft_joint_upper=torch.full((batch, 29), 10.0),
            foot_position_world=torch.zeros((batch, 2, 3)),
            foot_linear_velocity_world=torch.zeros((batch, 2, 3)),
            foot_contact=torch.tensor([[True, False]]),
            recent_foot_contact=torch.tensor([[True, False]]),
            recent_undesired_contact=torch.zeros((batch, 25), dtype=torch.bool),
            foot_duty_imbalance=torch.tensor([0.25]),
            repeated_touchdown=torch.tensor([1.0]),
            episode_steps=torch.tensor([8]),
            terminated=torch.tensor([False]),
        ),
        make_straight_symmetry_config(G1EnvConfig(), stage=1).rewards,
        0.02,
        torch.arange(15, 29),
        torch.arange(12, 15),
        torch.tensor([1, 2, 7, 8]),
    )

    assert torch.allclose(terms["lateral_vel_error_l2"], torch.tensor([-0.0001]))
    assert torch.allclose(terms["yaw_rate_error_l2"], torch.tensor([-0.0004]))
    assert torch.allclose(terms["foot_duty_imbalance"], torch.tensor([-0.00125]))
    assert torch.allclose(terms["repeated_touchdown"], torch.tensor([-0.01]))


def test_bootstrap_clearance_requires_a_moving_swing_foot_with_support() -> None:
    batch = 1
    zeros_29 = torch.zeros((batch, 29))
    default = torch.tensor(DEFAULT_JOINT_POS).unsqueeze(0)

    def clearance(speed: float, contacts: tuple[bool, bool]) -> torch.Tensor:
        _reward, terms = compute_reward(
            RewardInputs(
                commands=torch.tensor([[0.5, 0.0, 0.0]]),
                root_position=torch.tensor([[0.0, 0.0, 0.78]]),
                root_quaternion=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                root_linear_velocity_world=torch.zeros((batch, 3)),
                root_linear_velocity_body=torch.zeros((batch, 3)),
                root_angular_velocity_body=torch.zeros((batch, 3)),
                projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
                joint_position=default,
                default_joint_position=default,
                joint_velocity=zeros_29,
                previous_joint_velocity=zeros_29,
                actions=zeros_29,
                previous_actions=zeros_29,
                estimated_torque=zeros_29,
                soft_joint_lower=torch.full((batch, 29), -10.0),
                soft_joint_upper=torch.full((batch, 29), 10.0),
                foot_position_world=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]]),
                foot_linear_velocity_world=torch.tensor([[[0.0, 0.0, 0.0], [speed, 0.0, 0.0]]]),
                foot_contact=torch.tensor([contacts]),
                recent_foot_contact=torch.tensor([contacts]),
                recent_undesired_contact=torch.zeros((batch, 25), dtype=torch.bool),
                foot_duty_imbalance=torch.zeros(batch),
                repeated_touchdown=torch.zeros(batch),
                episode_steps=torch.tensor([1]),
                terminated=torch.tensor([False]),
            ),
            make_velocity_bootstrap_config(G1EnvConfig()).rewards,
            0.02,
            torch.arange(15, 29),
            torch.arange(12, 15),
            torch.tensor([1, 2, 7, 8]),
        )
        return terms["feet_clearance"]

    assert torch.equal(clearance(0.0, (True, False)), torch.zeros(1))
    assert clearance(1.0, (True, False)).item() > 0.019
    assert torch.equal(clearance(1.0, (False, False)), torch.zeros(1))
