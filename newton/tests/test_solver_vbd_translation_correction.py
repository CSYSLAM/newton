# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VBD translation coarse correction."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()


def _tube_mesh(radius=0.03, height=0.1, segments=16, rings=6):
    """Open cylindrical tube along +z, centered at the origin."""
    points, faces = [], []
    for j in range(rings + 1):
        z = height * j / rings - 0.5 * height
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    for j in range(rings):
        for i in range(segments):
            a, b = j * segments + i, j * segments + (i + 1) % segments
            faces.extend((a, b, b + segments, a, b + segments, a + segments))
    return points, faces


def _add_paper_tube(builder, pos):
    points, faces = _tube_mesh()
    builder.add_cloth_mesh(
        pos=wp.vec3(*pos),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(),
        vertices=points,
        indices=faces,
        density=0.24,
        tri_ke=5.0e4,
        tri_ka=5.0e4,
        tri_kd=0.01,
        # Stiff enough in bending that a two-sided pinch grips instead of flattening the tube.
        edge_ke=10.0,
        particle_radius=0.001,
    )


def _add_patch(builder, pos, dim=6):
    builder.add_cloth_grid(
        pos=wp.vec3(*pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=dim,
        dim_y=dim,
        cell_x=0.01,
        cell_y=0.01,
        mass=1.0e-4,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=0.1,
        particle_radius=0.001,
    )


def _run(model, solver, frames, substeps, contacts=None, pipeline=None, drive=None):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    dt = 1.0 / (60.0 * substeps)
    for step in range(frames * substeps):
        if drive is not None:
            drive(state_0, (step + 1) * dt)
        if pipeline is not None:
            pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0


def test_translation_components(test, device):
    """Separate soft bodies get separate components; a lone particle is left to the sweeps."""
    builder = newton.ModelBuilder()
    _add_patch(builder, (0.0, 0.0, 1.0), dim=2)
    first = builder.particle_count
    _add_patch(builder, (1.0, 0.0, 1.0), dim=2)
    second = builder.particle_count
    builder.add_particle(pos=(2.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0e-3)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=2, particle_enable_translation_correction=True)

    component = solver._translation_particle_component.numpy()
    test.assertEqual(solver._translation_component_count, 2)
    test.assertEqual(len(set(component[:first])), 1)
    test.assertEqual(len(set(component[first:second])), 1)
    test.assertNotEqual(component[0], component[first])
    test.assertEqual(component[second], -1)


def test_translation_correction_free_fall_unchanged(test, device):
    """Without contacts the correction only removes error the sweeps already converge, so free fall matches."""
    results = []
    for enabled in (False, True):
        builder = newton.ModelBuilder()
        _add_patch(builder, (0.0, 0.0, 1.0))
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(model, iterations=4, particle_enable_translation_correction=enabled)
        results.append(_run(model, solver, frames=10, substeps=4).particle_q.numpy())
    np.testing.assert_allclose(results[0], results[1], atol=1.0e-5)
    expected_drop = 0.5 * 9.81 * (10.0 / 60.0) ** 2
    test.assertAlmostEqual(float(1.0 - results[1][:, 2].mean()), expected_drop, delta=0.01)


def _pinched_tube_drops(device, enabled):
    """A light, stiff paper tube held by two kinematic plates against gravity.

    Returns the center drop after 1/6 s and after 1/3 s [m].
    """
    builder = newton.ModelBuilder()
    _add_paper_tube(builder, (0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.2e5, kd=10.0, mu=1.0)
    cfg.configure_sdf(force_sdf=True)
    plates = []
    for side in (-1.0, 1.0):
        # Each plate overlaps the wall by 1 mm.
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, side * 0.039, 0.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.02, hy=0.01, hz=0.02, cfg=cfg)
        builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        plates.append(body)
    builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0
    model.soft_contact_mu = 0.9
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.003, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(
        model, iterations=8, friction_epsilon=1.0e-4, particle_enable_translation_correction=enabled
    )
    start = model.particle_q.numpy()[:, 2].mean()
    state = _run(model, solver, frames=10, substeps=8, contacts=contacts, pipeline=pipeline)
    first = float(start - state.particle_q.numpy()[:, 2].mean())
    # Continue from the same solver and state for another 1/6 s.
    state_1 = model.state()
    control = model.control()
    for _ in range(80):
        pipeline.collide(state, contacts)
        solver.step(state, state_1, control, contacts, 1.0 / 480.0)
        state, state_1 = state_1, state
    return first, float(start - state.particle_q.numpy()[:, 2].mean())


def test_translation_correction_holds_pinched_shell(test, device):
    """Friction at a few gripped vertices must carry the whole shell, not let it slide through the grip."""
    # The tube settles into the grip while the contact forms, then hangs still.
    settle, held = _pinched_tube_drops(device, enabled=True)
    test.assertLess(settle, 0.05, f"pinched tube dropped {settle * 1000:.1f} mm while the grip formed")
    test.assertLess(held - settle, 0.001, f"pinched tube kept slipping: {(held - settle) * 1000:.1f} mm in 1/6 s")
    # Vertex sweeps alone propagate the grip too slowly: the tube slides through the plates.
    _, slipped = _pinched_tube_drops(device, enabled=False)
    test.assertGreater(slipped, 0.3, f"pinched tube dropped only {slipped * 1000:.1f} mm without the correction")


def test_translation_correction_skips_pinned_component(test, device):
    """A component with a pinned particle cannot translate as a whole, so the correction leaves it alone."""
    results = []
    for enabled in (False, True):
        builder = newton.ModelBuilder()
        _add_patch(builder, (0.0, 0.0, 1.0))
        builder.particle_mass[0] = 0.0
        builder.particle_flags[0] = builder.particle_flags[0] & ~int(newton.ParticleFlags.ACTIVE)
        builder.color()
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverVBD(model, iterations=4, particle_enable_translation_correction=enabled)
        results.append(_run(model, solver, frames=10, substeps=4).particle_q.numpy())
    np.testing.assert_allclose(results[0], results[1], atol=1.0e-6)


class TestSolverVBDTranslationCorrection(unittest.TestCase):
    pass


add_function_test(
    TestSolverVBDTranslationCorrection, "test_translation_components", test_translation_components, devices=devices
)
add_function_test(
    TestSolverVBDTranslationCorrection,
    "test_translation_correction_free_fall_unchanged",
    test_translation_correction_free_fall_unchanged,
    devices=devices,
)
add_function_test(
    TestSolverVBDTranslationCorrection,
    "test_translation_correction_holds_pinched_shell",
    test_translation_correction_holds_pinched_shell,
    devices=devices,
)
add_function_test(
    TestSolverVBDTranslationCorrection,
    "test_translation_correction_skips_pinned_component",
    test_translation_correction_skips_pinned_component,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
