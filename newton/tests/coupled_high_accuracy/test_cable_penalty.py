"""VBD-compatible adaptive cable penalties and selected-world cache resets."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.cable_material import register_cable_attributes
from newton.solvers import coupled_high_accuracy as s3d

# Rod DOFs of each material family in Newton main: stretch z, shear x/y, bend x/y, twist z.
ROD_MATERIAL_SLOTS = ((2,), (0, 1), (3, 4), (5,))


cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native cable penalties require CUDA")
FRAMEWORKS = [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON]


def _scene(*, worlds=1, globals=False, builder_type=s3d.ModelBuilder):
    builder = builder_type(gravity=wp.vec3(0.0))
    register_cable_attributes(builder)
    bodies = []

    def cable_pair(offset):
        for index in range(2):
            position = wp.vec3(offset, float(index), 0.0)
            body = builder.add_link(
                xform=wp.transform(position, wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                lock_inertia=True,
                is_kinematic=True,
            )
            builder.add_joint_rod(
                parent=-1,
                child=body,
                parent_xform=wp.transform(position, wp.quat_identity()),
                stretch_stiffness=15.0,
                shear_stiffness=25.0,
                bend_stiffness=30.0,
                twist_stiffness=40.0,
                stretch_damping=0.3,
                shear_damping=0.4,
                bend_damping=0.5,
                twist_damping=0.6,
            )
            bodies.append(body)

    for world in range(worlds):
        builder.begin_world()
        cable_pair(float(world))
        builder.end_world()
    if globals:
        cable_pair(float(worlds))
    builder.color()
    return builder.finalize(device="cuda:0"), bodies


def _solver(model, framework):
    solver = s3d.NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 3
    solver.config.linear_iterations = 2
    solver.config.cable_linear_beta = 100.0
    solver.config.cable_angular_beta = 20.0
    solver.config.cable_penalty_gamma = 0.5
    solver.config.cable_linear_k_start = 10.0
    solver.config.cable_angular_k_start = 5.0
    return solver


def _deform(state, bodies):
    poses = state.body_q.numpy()
    for index, body in enumerate(bodies):
        poses[body, :3] += np.asarray([0.03, 0.04, 0.02], dtype=np.float32)
        axis = wp.vec3(1.0, 0.0, 0.0) if index % 2 == 0 else wp.vec3(0.0, 0.0, 1.0)
        poses[body, 3:] = np.asarray(wp.quat_from_axis_angle(axis, 0.2))
    state.body_q.assign(poses)


def _rod_materials(model, solver, values):
    """Joint target stiffnesses with one (stretch, shear, bend, twist) value per material family on every cable."""
    materials = model.joint_target_ke.numpy().copy()
    for start in model.joint_qd_start.numpy()[solver._cable_joints.numpy()]:
        for slot, value in zip(ROD_MATERIAL_SLOTS, values):
            materials[start + np.asarray(slot)] = value
    return materials


def _penalties(solver):
    values = wp.empty(solver._cable_count, dtype=wp.vec4, device=solver.device)
    solver._native_solver.copy_cable_penalties(
        s3d.Span(ptr=values.ptr, size=len(values)),
        wp.get_stream(solver.device),
    )
    return values.numpy()


@cuda
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_cable_penalty_matches_prescribed_strain_recurrence(framework):
    # The engineRTS original also compared against Newton 1.7's SolverVBD penalty ramp; Newton main no longer
    # exposes per-family AVBD betas, so only the analytic recurrence is checked here.
    model, bodies = _scene()
    solver = _solver(model, framework)
    state_in, state_out = model.state(), model.state()
    _deform(state_in, bodies)
    solver.sync_affine_body_state(state_in)
    caps = np.asarray([[15.0, 25.0, 30.0, 40.0]] * 2)
    floors = np.asarray([[10.0, 10.0, 5.0, 5.0]] * 2)
    # A prescribed anchor makes strain constant, separating penalty scheduling
    # from the convergence and integration behavior of either primal solver.
    increment = np.asarray([[2.0, 5.0, 40.0 * np.tan(0.1), 0.0], [2.0, 5.0, 0.0, 4.0]])
    expected = floors.copy()
    materials = model.joint_target_ke.numpy().copy()
    damping = model.joint_target_kd.numpy().copy()
    for _ in range(3):
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
        expected = np.minimum(caps, np.maximum(floors, 0.5 * expected) + 3 * increment)
        current = _penalties(solver)
        np.testing.assert_allclose(current, expected, atol=2.0e-5, rtol=1.0e-5)
        state_in, state_out = state_out, state_in
    np.testing.assert_array_equal(model.joint_target_ke.numpy(), materials)
    np.testing.assert_array_equal(model.joint_target_kd.numpy(), damping)


@cuda
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_disabled_cable_decays_without_strain_ramping(framework):
    model, bodies = _scene()
    solver = _solver(model, framework)
    state_in, state_out = model.state(), model.state()
    _deform(state_in, bodies)
    solver.sync_affine_body_state(state_in)
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    before = _penalties(solver)
    model.joint_enabled.assign(np.zeros(int(model.joint_count), dtype=bool))
    # Enabled flags are live inputs. A notification would rebuild the cache
    # and hide whether disabled joints receive their one per-step decay.
    solver.step(state_out, state_in, model.control(), None, 1.0e-3)
    floors = np.asarray([[10.0, 10.0, 5.0, 5.0]] * 2)
    np.testing.assert_allclose(_penalties(solver), np.maximum(floors, 0.5 * before), atol=2.0e-5)


@cuda
@pytest.mark.parametrize("linear_beta,angular_beta", [(0.0, 0.0), (0.0, 20.0), (100.0, 0.0)])
def test_zero_beta_uses_authored_caps_independently_per_slot_family(linear_beta, angular_beta):
    model, bodies = _scene()
    solver = _solver(model, s3d.NonlinearSolver.NEWTON)
    solver.config.cable_linear_beta = linear_beta
    solver.config.cable_angular_beta = angular_beta
    solver.config.nonlinear_iterations = 0
    state_in, state_out = model.state(), model.state()
    _deform(state_in, bodies)
    solver.sync_affine_body_state(state_in)
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    expected = [15.0, 25.0, 30.0, 40.0]
    if linear_beta:
        expected[:2] = [10.0, 10.0]
    if angular_beta:
        expected[2:] = [5.0, 5.0]
    np.testing.assert_array_equal(_penalties(solver), [expected, expected])


@cuda
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("selected_slot", [0, 2])
def test_selected_penalty_reset_is_world_isolated_and_graph_safe(captured, selected_slot):
    model, bodies = _scene(worlds=2, globals=True)
    solver = _solver(model, s3d.NonlinearSolver.NEWTON)
    state_in, state_out = model.state(), model.state()
    _deform(state_in, bodies)
    solver.sync_affine_body_state(state_in)
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    before = _penalties(solver)
    assert np.all(before[:, 0] > 10.0)
    mask_values = np.zeros(3, dtype=bool)
    mask_values[selected_slot] = True
    mask = wp.array(mask_values, dtype=wp.bool, device=model.device)
    selected = model.joint_world.numpy()[solver._cable_joints.numpy()] == (selected_slot if selected_slot < 2 else -1)
    if captured:
        # Replaying must read the mask's live device contents, rather than a host snapshot.
        solver.reset(state_out, mask, flags=0)
        solver.step(state_out, state_in, model.control(), None, 1.0e-3)
        before = _penalties(solver)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state_in, mask, flags=0)
        alternate = np.zeros(3, dtype=bool)
        alternate[1] = True
        mask.assign(alternate)
        wp.capture_launch(capture.graph)
        selected = model.joint_world.numpy()[solver._cable_joints.numpy()] == 1
    else:
        solver.reset(state_out, mask, flags=0)
    current = _penalties(solver)
    np.testing.assert_array_equal(current[selected], [[10.0, 10.0, 5.0, 5.0]] * int(np.sum(selected)))
    np.testing.assert_array_equal(current[~selected], before[~selected])


@cuda
def test_material_notification_reinitializes_penalty_at_new_capped_seed():
    model, bodies = _scene()
    solver = _solver(model, s3d.NonlinearSolver.NEWTON)
    state_in, state_out = model.state(), model.state()
    _deform(state_in, bodies)
    solver.sync_affine_body_state(state_in)
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    assert np.max(_penalties(solver)[:, 0]) == 15.0
    model.joint_target_ke.assign(_rod_materials(model, solver, (3.0, 7.0, 2.0, 4.0)))
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    np.testing.assert_array_equal(_penalties(solver), [[3.0, 7.0, 2.0, 4.0]] * 2)
    solver.reset(state_out, flags=0)
    np.testing.assert_array_equal(_penalties(solver), [[3.0, 7.0, 2.0, 4.0]] * 2)


@cuda
@pytest.mark.parametrize(
    "name,value",
    [
        ("cable_linear_beta", -1.0),
        ("cable_angular_beta", float("nan")),
        ("cable_penalty_gamma", 1.1),
        ("cable_linear_k_start", -1.0),
        ("cable_angular_k_start", float("inf")),
    ],
)
def test_invalid_cable_penalty_config_is_rejected(name, value):
    model, _ = _scene()
    solver = _solver(model, s3d.NonlinearSolver.NEWTON)
    setattr(solver.config, name, value)
    with pytest.raises(ValueError, match="cable.*(gamma|ramps|seeds)"):
        _penalties(solver)


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, (0.0, 0.0, 0.999, 100.0, 10.0)),
        ({"rigid_avbd_beta": 7.0}, (7.0, 7.0, 0.999, 100.0, 10.0)),
        (
            {
                "rigid_avbd_beta": 7.0,
                "rigid_avbd_linear_beta": 0.0,
                "rigid_avbd_angular_beta": 3.0,
                "rigid_avbd_gamma": 0.5,
                "rigid_joint_linear_k_start": 11.0,
                "rigid_joint_angular_k_start": 2.0,
            },
            (0.0, 3.0, 0.5, 11.0, 2.0),
        ),
        (
            {"rigid_avbd_beta": 7.0, "rigid_avbd_linear_beta": None, "rigid_avbd_angular_beta": None},
            (7.0, 7.0, 0.999, 100.0, 10.0),
        ),
    ],
)
def test_example_helper_preserves_vbd_penalty_option_precedence(monkeypatch, options, expected):
    path = Path(__file__).resolve().parents[2] / "examples" / "coupled_high_accuracy" / "cable_utils.py"
    spec = importlib.util.spec_from_file_location("_penalty_example_helper", path)
    helper = importlib.util.module_from_spec(spec)
    # Dataclasses resolves its defining module while the helper is executing.
    import sys

    monkeypatch.setitem(sys.modules, spec.name, helper)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(
        helper.s3d, "SolverCoupledHighAccuracy", lambda model: SimpleNamespace(config=SimpleNamespace())
    )
    solver = helper.create_cable_solver(argparse.Namespace(solver="synreal"), object(), **options)
    names = (
        "cable_linear_beta",
        "cable_angular_beta",
        "cable_penalty_gamma",
        "cable_linear_k_start",
        "cable_angular_k_start",
    )
    assert tuple(getattr(solver.config, name) for name in names) == expected
