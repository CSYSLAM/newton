# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check contact motion invariance and the zero-slip friction tangent."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as full_particle
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as full_rigid
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as soft_particle
from newton._src.solvers.mjvbd_v2.vbd_soft import rigid_vbd_kernels as soft_rigid


def _make_self_contact_probe(module):
    @wp.kernel
    def probe(
        pos: wp.array[wp.vec3],
        anchor: wp.array[wp.vec3],
        triangles: wp.array2d[wp.int32],
        edges: wp.array2d[wp.int32],
        edge_contact: bool,
        damping: float,
        friction: float,
        forces: wp.array2d[wp.vec3],
    ):
        i = wp.tid()
        if edge_contact:
            for j in range(4):
                f, _h = module.evaluate_edge_edge_contact(
                    4 * i + j,
                    j,
                    2 * i,
                    2 * i + 1,
                    pos,
                    anchor,
                    edges,
                    0.002,
                    300000.0,
                    damping,
                    friction,
                    0.01,
                    1.0 / 600.0,
                    1.0e-5,
                )
                forces[i, j] = f
            _valid, f0, f1, _h0, _h1 = module.evaluate_edge_edge_contact_2_vertices(
                2 * i,
                2 * i + 1,
                pos,
                anchor,
                edges,
                0.002,
                300000.0,
                damping,
                friction,
                0.01,
                1.0 / 600.0,
                1.0e-5,
            )
            forces[i, 4] = f0
            forces[i, 5] = f1
        else:
            for j in range(4):
                f, _h = module.evaluate_vertex_triangle_collision_force_hessian(
                    4 * i + 3,
                    j,
                    i,
                    pos,
                    anchor,
                    triangles,
                    0.002,
                    300000.0,
                    damping,
                    friction,
                    0.01,
                    1.0 / 600.0,
                )
                forces[i, j] = f
            _valid, f0, f1, f2, f3, _h0, _h1, _h2, _h3 = (
                module.evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                    4 * i + 3,
                    i,
                    pos,
                    anchor,
                    triangles,
                    0.002,
                    300000.0,
                    damping,
                    friction,
                    0.01,
                    1.0 / 600.0,
                )
            )
            forces[i, 4] = f0
            forces[i, 5] = f1
            forces[i, 6] = f2
            forces[i, 7] = f3

    return probe


def _make_friction_probe(particle, rigid):
    @wp.kernel
    def probe(
        slip: wp.array[wp.vec3],
        mu: float,
        load: float,
        epsilon: float,
        force: wp.array2d[wp.vec3],
        tangent: wp.array2d[wp.mat33],
    ):
        i = wp.tid()
        basis = particle.mat32(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        f, h = particle.compute_friction(mu, load, basis, wp.vec2(slip[i][0], slip[i][1]), epsilon)
        force[i, 0] = f
        tangent[i, 0] = h
        f, h = rigid.compute_projected_isotropic_friction(mu, load, wp.vec3(0.0, 0.0, 1.0), slip[i], epsilon)
        force[i, 1] = f
        tangent[i, 1] = h

    return probe


def _make_body_contact_probe(module):
    @wp.kernel
    def probe(
        pos: wp.array[wp.vec3],
        anchor: wp.array[wp.vec3],
        body: wp.array[wp.transform],
        body_prev: wp.array[wp.transform],
        velocity: wp.array[wp.spatial_vector],
        com: wp.array[wp.vec3],
        radius: wp.array[float],
        indices: wp.array[wp.int32],
        local_point: wp.array[wp.vec3],
        local_velocity: wp.array[wp.vec3],
        normal: wp.array[wp.vec3],
        margin: wp.array[float],
        damping: float,
        friction: float,
        forces: wp.array[wp.vec3],
    ):
        i = wp.tid()
        if i == 0:
            f, _h = module._eval_body_particle_contact(
                0,
                pos[0],
                anchor[0],
                0,
                300000.0,
                damping,
                friction,
                0.01,
                radius,
                indices,
                body,
                body_prev,
                velocity,
                com,
                indices,
                local_point,
                local_velocity,
                normal,
                margin,
                1.0 / 600.0,
            )
            forces[i] = f
        else:
            corners = wp.vec3i(0, -1, -1)
            bary = wp.vec3(1.0, 0.0, 0.0)
            if i == 2:
                corners = wp.vec3i(0, 1, -1)
                bary = wp.vec3(0.3, 0.7, 0.0)
            elif i == 3:
                corners = wp.vec3i(0, 1, 2)
                bary = wp.vec3(0.2, 0.3, 0.5)
            f, _h, _bx = module._eval_soft_ef_contact(
                0,
                corners,
                bary,
                pos,
                anchor,
                radius,
                300000.0,
                damping,
                friction,
                0.01,
                indices,
                body,
                body_prev,
                velocity,
                com,
                indices,
                local_point,
                local_velocity,
                normal,
                margin,
                1.0 / 600.0,
            )
            forces[i] = f

    return probe


_VARIANTS = (
    ("full", _make_self_contact_probe(full_particle), _make_friction_probe(full_particle, full_rigid)),
    ("soft", _make_self_contact_probe(soft_particle), _make_friction_probe(soft_particle, soft_rigid)),
)
_BODY_VARIANTS = tuple(
    (name, _make_body_contact_probe(module))
    for name, module in (
        ("full", full_rigid),
        ("soft", soft_rigid),
        ("full_particle", full_particle),
        ("soft_particle", soft_particle),
    )
)


class TestMJVBDV2ContactInvariants(unittest.TestCase):
    def test_body_contact_relative_slip(self):
        """Preserve friction from particle slip, body rotation and conveyor motion."""
        pos = np.array([[2.125, 1.25, 3.0009765625]] * 3, dtype=np.float32)
        point = np.array([0.153, 0.231, 0], dtype=np.float32)
        dt = 1.0 / 600.0
        particle_delta = np.array([1.0 / 8192, 0, 0], dtype=np.float32)
        for device in wp.get_devices():
            for name, probe in _BODY_VARIANTS:
                for mode in ("particle", "rotation", "conveyor", "velocity"):
                    with self.subTest(device=str(device), backend=name, mode=mode):

                        def array(values, dtype=wp.vec3, device=device):
                            return wp.array(values, dtype=dtype, device=device)

                        previous_rotation = np.array([0, 0, 0, 1], dtype=np.float32)
                        if mode == "rotation":
                            previous_rotation = np.array([0, 0, np.sin(0.01), np.cos(0.01)], dtype=np.float32)
                        belt = np.array([0.03, 0, 0] if mode == "conveyor" else [0, 0, 0], dtype=np.float32)
                        velocity = np.array([0.015, 0.02, 0, 0, 0, 0.02], dtype=np.float32)
                        body_prev = array([wp.transform([2, 1, 3], wp.quat(*previous_rotation))], wp.transform)
                        if mode == "velocity":
                            body_prev = wp.empty(0, dtype=wp.transform, device=device)
                        inputs = [
                            array(pos),
                            array(pos - particle_delta),
                            array([wp.transform([2, 1, 3], wp.quat_identity())], wp.transform),
                            body_prev,
                            array([velocity], wp.spatial_vector),
                            array([[0, 0, 0]]),
                            array([0.002] * 3, float),
                            array([0], wp.int32),
                            array([point]),
                            array([belt]),
                            array([[0, 0, 1]]),
                            array([0], float),
                        ]
                        force = wp.zeros(4, dtype=wp.vec3, device=device)
                        wp.launch(probe, 4, inputs=[*inputs, 0.0, 0.0], outputs=[force], device=device)
                        reference = force.numpy()
                        # Evaluate the displacement and radial friction law independently in float64.
                        q = previous_rotation.astype(np.float64)
                        p = point.astype(np.float64)
                        rotated_previous = p + 2 * np.cross(q[:3], np.cross(q[:3], p) + q[3] * p)
                        body_delta = p - rotated_previous + dt * belt.astype(np.float64)
                        if mode == "velocity":
                            v = velocity.astype(np.float64)
                            body_delta = dt * (v[:3] + np.cross(v[3:], p))
                        slip = particle_delta.astype(np.float64) - body_delta
                        norm = np.linalg.norm(slip)
                        epsilon = 0.01 * dt
                        factor = 1 / norm if norm > epsilon else (2 - norm / epsilon) / epsilon
                        expected = reference.astype(np.float64) - 0.4 * reference[:, 2:3] * factor * slip
                        wp.launch(probe, 4, inputs=[*inputs, 0.0, 0.4], outputs=[force], device=device)
                        np.testing.assert_allclose(force.numpy(), expected, rtol=2e-5, atol=2e-3)

    def test_body_contact_common_motion(self):
        """Keep point, edge and face contact forces invariant under common motion."""
        pos = np.array(
            [[2.125, 1.25, 3.0009765625], [2.375, 1.5, 3.0009765625], [2.5, 1.125, 3.0009765625]], dtype=np.float32
        )
        translation = np.array([2.0, 1.0, 3.0], dtype=np.float32)
        for device in wp.get_devices():
            for name, probe in _BODY_VARIANTS:
                for motion in ((0, 0, 0), (0.125, -0.25, 0.5), (4, -8, 16)):
                    with self.subTest(device=str(device), backend=name, motion=motion):

                        def array(values, dtype=wp.vec3, device=device):
                            return wp.array(values, dtype=dtype, device=device)

                        delta = np.asarray(motion, dtype=np.float32)
                        inputs = [
                            array(pos),
                            array(pos - delta),
                            array([wp.transform(translation, wp.quat_identity())], wp.transform),
                            array([wp.transform(translation - delta, wp.quat_identity())], wp.transform),
                            wp.zeros(1, dtype=wp.spatial_vector, device=device),
                            array([[0, 0, 0]]),
                            array([0.002] * 3, float),
                            array([0], wp.int32),
                            array([[0.153, 0.231, 0]]),
                            array([[0, 0, 0]]),
                            array([[0, 0, 1]]),
                            array([0], float),
                        ]
                        force = wp.zeros(4, dtype=wp.vec3, device=device)
                        wp.launch(probe, 4, inputs=[*inputs, 0.0, 0.0], outputs=[force], device=device)
                        reference = force.numpy()
                        self.assertGreater(np.linalg.norm(reference), 1.0)
                        wp.launch(probe, 4, inputs=[*inputs, 100.0, 0.4], outputs=[force], device=device)
                        np.testing.assert_allclose(force.numpy(), reference, rtol=0, atol=1e-5)

    def test_self_contact_common_motion(self):
        """Produce no dissipative force at rest or under common translation."""
        count = 32
        triangles = np.arange(4 * count, dtype=np.int32).reshape(-1, 4)[:, :3]
        edges = np.full((2 * count, 4), -1, dtype=np.int32)
        edges[:, 2:] = np.arange(4 * count, dtype=np.int32).reshape(-1, 2)
        for device in wp.get_devices():
            for name, probe, _ in _VARIANTS:
                for edge_contact in (False, True):
                    rng = np.random.default_rng(42)
                    positions = []
                    for _ in range(count):
                        origin = rng.integers(8000, 24000, size=3)
                        if edge_contact:
                            local = np.array([[0, 0, 0], [801, 391, 0], [30, 430, 8], [640, -100, 8]])
                            local[:, :2] += rng.integers(-50, 50, size=(4, 2))
                        else:
                            local = np.array([[0, 0, 0], [809, 51, 0], [81, 711, 0], [241, 279, 8]])
                        positions.extend((origin + local) / 8192.0)
                    pos = np.asarray(positions, dtype=np.float32)
                    inputs = [
                        wp.array(pos, dtype=wp.vec3, device=device),
                        None,
                        wp.array(triangles, dtype=wp.int32, device=device),
                        wp.array(edges, dtype=wp.int32, device=device),
                        edge_contact,
                    ]
                    force = wp.zeros((count, 8), dtype=wp.vec3, device=device)
                    for motion in ((0.0, 0.0, 0.0), (0.125, -0.25, 0.5), (4.0, -8.0, 16.0)):
                        with self.subTest(device=str(device), backend=name, edge=edge_contact, motion=motion):
                            anchor = pos - np.asarray(motion, dtype=np.float32)
                            np.testing.assert_array_equal(pos - anchor, np.broadcast_to(motion, pos.shape))
                            inputs[1] = wp.array(anchor, dtype=wp.vec3, device=device)
                            wp.launch(probe, count, inputs=[*inputs, 0.0, 0.0], outputs=[force], device=device)
                            normal = force.numpy()
                            self.assertGreater(np.linalg.norm(normal), 1.0)
                            for damping, friction in ((100.0, 0.0), (0.0, 0.4)):
                                wp.launch(
                                    probe, count, inputs=[*inputs, damping, friction], outputs=[force], device=device
                                )
                                np.testing.assert_allclose(force.numpy(), normal, rtol=0.0, atol=1.0e-5)

    def test_zero_slip_friction_tangent(self):
        """Match the force derivative at zero slip and retain the nonzero-slip law."""
        step = 1.0e-7
        slips = np.array(
            [
                [0, 0, 0],
                [step, 0, 0],
                [-step, 0, 0],
                [0, step, 0],
                [0, -step, 0],
                [0.0002, 0.0003, 0],
                [0.003, 0.004, 0],
                [0, 0, 0.02],
            ],
            dtype=np.float32,
        )
        for device in wp.get_devices():
            for name, _, probe in _VARIANTS:
                for mu, load in ((0.4, 12.0), (0.0, 12.0), (0.4, 0.0)):
                    with self.subTest(device=str(device), backend=name, mu=mu, load=load):
                        force = wp.zeros((len(slips), 2), dtype=wp.vec3, device=device)
                        tangent = wp.zeros((len(slips), 2), dtype=wp.mat33, device=device)
                        wp.launch(
                            probe,
                            len(slips),
                            inputs=[wp.array(slips, dtype=wp.vec3, device=device), mu, load, 0.001],
                            outputs=[force, tangent],
                            device=device,
                        )
                        f, h = force.numpy(), tangent.numpy()
                        expected = np.diag([2 * mu * load / 0.001, 2 * mu * load / 0.001, 0])
                        for variant in range(2):
                            np.testing.assert_array_equal(f[0, variant], np.zeros(3))
                            np.testing.assert_allclose(h[0, variant], expected, rtol=2e-6, atol=1e-6)
                            np.testing.assert_allclose(h[7, variant], expected, rtol=2e-6, atol=1e-6)
                            derivative = np.column_stack(
                                (
                                    -(f[1, variant] - f[2, variant]) / (2 * step),
                                    -(f[3, variant] - f[4, variant]) / (2 * step),
                                    np.zeros(3),
                                )
                            )
                            np.testing.assert_allclose(h[0, variant], derivative, rtol=1e-4, atol=1e-6)
                            for i in (5, 6):
                                u = slips[i].astype(np.float64)
                                norm = np.linalg.norm(u)
                                scale = mu * load * (1 / norm if norm > 0.001 else (2 - norm / 0.001) / 0.001)
                                np.testing.assert_allclose(f[i, variant], -scale * u, rtol=2e-6, atol=1e-6)

    def test_unsmoothed_zero_slip_is_finite(self):
        """Preserve the zero-slip fallback when friction smoothing is disabled."""
        for device in wp.get_devices():
            for name, _, probe in _VARIANTS:
                with self.subTest(device=str(device), backend=name):
                    force = wp.zeros((1, 2), dtype=wp.vec3, device=device)
                    tangent = wp.zeros((1, 2), dtype=wp.mat33, device=device)
                    wp.launch(
                        probe,
                        1,
                        inputs=[wp.zeros(1, dtype=wp.vec3, device=device), 0.4, 12.0, 0.0],
                        outputs=[force, tangent],
                        device=device,
                    )
                    np.testing.assert_array_equal(force.numpy(), np.zeros((1, 2, 3)))
                    np.testing.assert_array_equal(tangent.numpy(), np.zeros((1, 2, 3, 3)))


if __name__ == "__main__":
    unittest.main()
