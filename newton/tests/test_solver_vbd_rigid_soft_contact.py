# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for VBD full-surface rigid-soft contact and rigid-soft coupling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverObservableFlags
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

devices = get_test_devices()
cuda_devices = get_cuda_test_devices()

_SPHERE_RADIUS = 0.0065


def _add_sheet(builder, dim_x, dim_y, origin, tri_ke=1.0e6):
    """A stiff paper-like sheet with 5 mm cells lying flat at ``origin``."""
    builder.add_cloth_grid(
        pos=wp.vec3(*origin),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=dim_x,
        dim_y=dim_y,
        cell_x=0.005,
        cell_y=0.005,
        mass=0.24 * 0.005 * 0.005,
        tri_ke=tri_ke,
        tri_ka=tri_ke,
        tri_kd=0.01,
        edge_ke=20.0,
        edge_kd=0.002,
        particle_radius=0.001,
    )


def _set_soft_material(model):
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0
    model.soft_contact_mu = 0.9


def _flat_press_force(device, full_surface):
    """Normal force [N] of a static box pressing a whole pinned 6 cm x 6 cm sheet 0.5 mm deep."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _add_sheet(builder, 12, 12, (-0.03, -0.03, 0.0))
    # Box top 0.5 mm into the sheet's contact radius (particle radius 1 mm, no shape margin).
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.0105), wp.quat_identity()),
        hx=0.05,
        hy=0.05,
        hz=0.01,
        cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=0.0, mu=0.0),
    )
    builder.color()
    model = builder.finalize(device=device)
    _set_soft_material(model)
    model.particle_inv_mass.zero_()
    model.particle_mass.zero_()
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.003, enable_rigid_soft_full_surface_contact=full_surface
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=1)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    state_0, state_1 = model.state(), model.state()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0, observables=observables)
    count = int(contacts.soft_contact_count.numpy()[0])
    rows = observables.contact_f.numpy()[contacts.rigid_contact_max : contacts.rigid_contact_max + count]
    return float(rows[:, 2].sum())


def test_full_surface_flat_contact_carries_its_area(test, device):
    """Flat full-surface contact is as stiff as the pressed area, not as the number of records.

    ``ke`` is the stiffness of one vertex's share of the surface (one 5 mm cell on this grid), so
    pressing all 144 cells 0.5 mm deep takes about ``144 * ke * 0.5 mm``. Every vertex, edge and
    triangle of the sheet reaches that depth; counted at full stiffness, their records would
    carry the same contact about six times over.
    """
    expected = -144 * 1.0e4 * 0.0005
    full = _flat_press_force(device, full_surface=True)
    test.assertAlmostEqual(full / expected, 1.0, delta=0.07)
    # Per-vertex contact counts each boundary vertex as a whole cell.
    vertex = _flat_press_force(device, full_surface=False)
    test.assertAlmostEqual(vertex / expected, 169.0 / 144.0, delta=0.02)


def _rolling_sphere_speeds(device, pinned, translation_correction=False, seconds=0.75):
    """Speed history [m/s] of a sphere launched rolling at 0.1 m/s across a stiff sheet."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _add_sheet(builder, 40, 12, (-0.10, -0.03, 0.001))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0023 + _SPHERE_RADIUS + 0.0003 + 0.0002), wp.quat_identity())
    )
    builder.add_shape_sphere(
        body,
        radius=_SPHERE_RADIUS,
        cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=0.02, mu=0.45, density=90.0, margin=0.0003),
    )
    builder.color()
    model = builder.finalize(device=device)
    _set_soft_material(model)
    if pinned:
        model.particle_inv_mass.zero_()
        model.particle_mass.zero_()
    pipeline = newton.CollisionPipeline(
        model, soft_contact_gap=0.003, enable_rigid_soft_full_surface_contact=True, contact_matching="latest"
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model,
        iterations=8,
        friction_epsilon=1.0e-4,
        rigid_contact_history=True,
        particle_enable_translation_correction=translation_correction,
    )
    state_0, state_1 = model.state(), model.state()
    qd = state_0.body_qd.numpy()
    qd[body] = (0.1, 0.0, 0.0, 0.0, 0.1 / _SPHERE_RADIUS, 0.0)
    state_0.body_qd.assign(qd)
    control = model.control()
    dt = 1.0 / 480.0
    speeds = []
    for _ in range(int(seconds / dt)):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        speeds.append(float(np.linalg.norm(state_0.body_qd.numpy()[body, :3])))
    return np.asarray(speeds)


def test_full_surface_rolling_contact_is_conservative(test, device):
    """A sphere rolling over a fixed triangulated sheet neither speeds up nor gets kicked.

    Each edge and triangle record samples the shape at the deepest point of its feature; held where
    detection found it, that point lags behind a rolling body and pushes it forward every step.
    """
    speeds = _rolling_sphere_speeds(device, pinned=True)[48:]
    test.assertLess(float(np.max(np.abs(np.diff(speeds)))), 0.003)
    test.assertLess(float(speeds.max()), 0.102)
    test.assertGreater(float(speeds.min()), 0.09)


def test_translation_correction_carries_resting_body(test, device):
    """A light body resting on a stiff sheet moves with the sheet's coarse translation.

    The sheet lies on the ground, so its translation correction keeps settling it; a sphere rolling
    on it must follow those corrections rather than be struck by them.
    """
    speeds = _rolling_sphere_speeds(device, pinned=False, translation_correction=True)[48:]
    test.assertLess(float(np.max(np.abs(np.diff(speeds)))), 0.003)
    test.assertLess(float(speeds.max()), 0.102)
    test.assertGreater(float(speeds.min()), 0.09)


def _hanging_strip(device, yield_angle):
    """A strip pinned along one short edge, folding down under gravity."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=8,
        dim_y=2,
        cell_x=0.01,
        cell_y=0.01,
        mass=1.0e-4,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=0.0,
        edge_ke=1.0e-3,
        edge_kd=0.0,
        particle_radius=0.001,
        fix_left=True,
    )
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=8, particle_bending_yield_angle=yield_angle)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return model, solver, state_0


def test_plastic_bending_sets_folds(test, device):
    """Hinges bent past the yield angle keep the excess as permanent set; reset restores them."""
    model, elastic, _ = _hanging_strip(device, None)
    test.assertIs(elastic.edge_rest_angle, model.edge_rest_angle)

    yield_angle = 0.05
    model, solver, state = _hanging_strip(device, yield_angle)
    rest = model.edge_rest_angle.numpy()
    plastic = solver.edge_rest_angle.numpy()
    np.testing.assert_array_equal(rest, 0.0)
    set_angle = np.abs(plastic - rest)
    test.assertGreater(float(set_angle.max()), 0.2, "the strip folded past yield without taking a set")

    # Each hinge sits within the yield angle of its rest angle once the excess has set.
    pos = state.particle_q.numpy()
    edges = model.edge_indices.numpy()
    for e, (o0, o1, v0, v1) in enumerate(edges):
        if o0 < 0 or o1 < 0:
            continue
        n1 = np.cross(pos[v0] - pos[o0], pos[v1] - pos[o0])
        n2 = np.cross(pos[v1] - pos[o1], pos[v0] - pos[o1])
        axis = (pos[v1] - pos[v0]) / np.linalg.norm(pos[v1] - pos[v0])
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        theta = np.arctan2(np.dot(np.cross(n1, n2), axis), np.dot(n1, n2))
        test.assertLessEqual(abs(theta - plastic[e]), yield_angle + 1.0e-3)

    solver.reset(model.state(), flags=newton.StateFlags.PARTICLE_Q)
    np.testing.assert_array_equal(solver.edge_rest_angle.numpy(), rest)


class TestSolverVBDRigidSoftContact(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDRigidSoftContact,
    "test_full_surface_flat_contact_carries_its_area",
    test_full_surface_flat_contact_carries_its_area,
    devices=devices,
)
add_function_test(
    TestSolverVBDRigidSoftContact,
    "test_full_surface_rolling_contact_is_conservative",
    test_full_surface_rolling_contact_is_conservative,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBDRigidSoftContact,
    "test_translation_correction_carries_resting_body",
    test_translation_correction_carries_resting_body,
    devices=cuda_devices,
)
add_function_test(
    TestSolverVBDRigidSoftContact,
    "test_plastic_bending_sets_folds",
    test_plastic_bending_sets_folds,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
