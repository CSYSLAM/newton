"""UR10 worlds must retain their own joint channels and change scan stages smoothly."""

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

from newton.examples.coupled_high_accuracy.example_cha_robot_ur10 import (
    Example,
    joint_scan_trajectory,
    update_joint_target_trajectory_kernel,
)


def initial_poses():
    return np.arange(8 * 6).reshape(8, 6) * 0.03 - 0.7


def test_stages_change_joint_order_without_swapping_target_channels():
    initial = initial_poses()
    trajectory = joint_scan_trajectory(initial, np.full(6, -2.0), np.full(6, 2.0))
    np.testing.assert_array_equal(trajectory[0], initial.astype(np.float32))
    np.testing.assert_array_equal(trajectory[-1], trajectory[0])
    steps = (len(trajectory) - 1) // 6
    for stage in range(6):
        positions = trajectory[stage * steps + steps // 4]
        for world in range(8):
            moved = np.flatnonzero(np.abs(positions[world] - initial[world]) > 1e-5)
            np.testing.assert_array_equal(moved, [(stage + world) % 6])


@pytest.mark.parametrize("amplitude,speed", [(5, 10), (30, 30), (60, 45)])
def test_scan_respects_joint_limits_and_peak_target_speed(amplitude, speed):
    initial = np.array([[0.99, -0.98, 0], [0.7, -0.7, 0.5]])
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    rate = 50.0
    trajectory = joint_scan_trajectory(initial, lower, upper, rate, amplitude, speed)
    assert np.all(trajectory >= lower - 1e-7) and np.all(trajectory <= upper + 1e-7)
    assert np.abs(trajectory - initial).max() <= np.deg2rad(amplitude) + 1e-7
    peak_speed = np.rad2deg(np.abs(np.diff(trajectory, axis=0) * rate).max())
    assert peak_speed <= speed + 0.01


def test_each_stage_and_the_loop_join_at_rest():
    trajectory = joint_scan_trajectory(initial_poses(), np.full(6, -2.0), np.full(6, 2.0))
    steps = (len(trajectory) - 1) // 6
    velocity = np.diff(trajectory, axis=0) * 50
    for stage in range(6):
        np.testing.assert_array_equal(trajectory[stage * steps], trajectory[0])
        assert np.rad2deg(np.abs(velocity[stage * steps]).max()) < 0.02
        assert np.rad2deg(np.abs(velocity[(stage + 1) * steps - 1]).max()) < 0.02
    acceleration = np.diff(velocity, axis=0) * 50
    assert np.rad2deg(np.abs(acceleration).max()) < 100


@pytest.mark.parametrize("wrap", [False, True])
def test_interpolation_preserves_world_joint_identity_and_velocity_units(wrap):
    rate = 50.0
    trajectory = joint_scan_trajectory(initial_poses(), np.full(6, -2.0), np.full(6, 2.0))
    count, worlds, joints = trajectory.shape
    start = (
        np.full(worlds, count - 1 - 0.1, dtype=np.float32) if wrap else np.linspace(0, 0.7, worlds, dtype=np.float32)
    )
    dt = 0.2
    target, velocity = [wp.zeros((worlds, 1, joints), dtype=float, device="cpu") for _ in range(2)]
    clock = wp.array(start, dtype=float, device="cpu")
    wp.launch(
        update_joint_target_trajectory_kernel,
        dim=worlds,
        inputs=[wp.array(trajectory, dtype=float, device="cpu"), clock, dt, rate],
        outputs=[target, velocity],
        device="cpu",
    )
    time = (start + np.float32(dt)) % (count - 1)
    step = time.astype(int)
    fraction = (time - step)[:, None]
    a, b = trajectory[step, np.arange(worlds)], trajectory[step + 1, np.arange(worlds)]
    np.testing.assert_allclose(target.numpy()[:, 0], a + (b - a) * fraction, atol=2e-7)
    np.testing.assert_allclose(velocity.numpy()[:, 0], (b - a) * rate, atol=2e-7)
    np.testing.assert_allclose(clock.numpy(), time, atol=2e-7)


@pytest.mark.parametrize(
    "kwargs", [{"amplitude_degrees": -1}, {"speed_degrees": 0}, {"sample_rate": np.inf}, {"speed_degrees": np.nan}]
)
def test_invalid_motion_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        joint_scan_trajectory(np.zeros((2, 6)), np.full(6, -1), np.full(6, 1), **kwargs)


def test_invalid_pose_or_limits_are_rejected():
    with pytest.raises(ValueError, match="initial pose"):
        joint_scan_trajectory(np.full((2, 6), 2), np.full(6, -1), np.full(6, 1))
    with pytest.raises(ValueError, match="limit vectors"):
        joint_scan_trajectory(np.zeros((2, 6)), np.full(6, np.nan), np.full(6, 1))


@pytest.mark.parametrize("fault", [None, "nonfinite", "linear", "angular"])
def test_motion_check_rejects_the_original_finite_but_explosive_state(fault):
    poses = np.array([[0, 0, 1.2, 0, 0, 0, 1]], dtype=np.float32)
    velocity = np.zeros((1, 6), dtype=np.float32)
    if fault == "nonfinite":
        poses[0, 0] = np.nan
    elif fault == "linear":
        velocity[0, 0] = 32.18
    elif fault == "angular":
        velocity[0, 3] = 782.39
    example = Example.__new__(Example)
    example.state_0 = SimpleNamespace(
        body_q=SimpleNamespace(numpy=lambda: poses), body_qd=SimpleNamespace(numpy=lambda: velocity)
    )
    if fault is None:
        example.test_post_step()
    else:
        with pytest.raises(ValueError):
            example.test_post_step()
