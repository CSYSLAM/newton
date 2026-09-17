# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the PR #4180 rigid-soft DAT port."""

import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.rigid_soft_dat import _apply_particles
from newton._src.solvers.mjvbd_v2.rigid_soft_dat_kernels import (
    apply_body_truncation_ts,
    apply_rigid_soft_truncation,
    planar_truncation_t,
)
from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import accelerate_particle_iteration_chebyshev_guarded
from newton.solvers import SolverMJVBDV2


@wp.kernel
def _plane_probe(out: wp.array[float]):
    n = wp.vec3(0.0, 0.0, 1.0)
    out[0] = planar_truncation_t(wp.vec3(0.0, 0.0, 0.01), wp.vec3(0.0, 0.0, -0.02), n, wp.vec3(), 0.85)
    out[1] = planar_truncation_t(wp.vec3(0.0, 0.0, -0.01), wp.vec3(0.0, 0.0, -0.02), n, wp.vec3(), 0.85)
    out[2] = planar_truncation_t(wp.vec3(0.0, 0.0, -0.01), wp.vec3(0.0, 0.0, 0.001), n, wp.vec3(), 0.85)


class TestRigidSoftDAT(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Coupled translation requires CUDA")
    def test_dat_rejects_coupled_translation(self):
        """Keep coupled body-particle corrections available only without DAT."""
        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder()
            link = builder.add_link(is_kinematic=True)
            joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0, 1, 0))
            builder.add_articulation([joint])
            body = builder.add_body()
            builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
            builder.add_cloth_grid(
                pos=wp.vec3(0, 0, 0.1),
                rot=wp.quat_identity(),
                vel=wp.vec3(),
                dim_x=4,
                dim_y=4,
                cell_x=0.01,
                cell_y=0.01,
                mass=0.001,
            )
            builder.color(include_bending=True)
            # Only the free body participates in the coupled rigid solve.
            builder.body_color_groups = [np.array([body], dtype=np.int32)]
            model = builder.finalize()
            options = {
                "rigid_contact_hard": False,
                "particle_enable_multilevel_correction": True,
                "particle_multilevel_operator": "galerkin",
                "particle_enable_coupled_translation": True,
            }

            def construct(dat):
                return SolverMJVBDV2(
                    model,
                    mujoco_articulations=(0,),
                    joint_mode="kinematic",
                    contact_mode="full",
                    vbd_options={**options, "rigid_soft_enable_dat": dat},
                    collision_options={"soft_contact_margin": 0.004},
                )

            self.assertTrue(construct(False).vbd_solver.particle_enable_coupled_translation)
            with self.assertRaisesRegex(ValueError, "Rigid-soft DAT.*coupled translation"):
                construct(True)

    @unittest.skipUnless(wp.is_cuda_available(), "Multilevel correction requires CUDA")
    def test_multilevel_contact_preserves_support_plane(self):
        """Keep coarse corrections above a support in eager and captured steps."""
        for mixed, operator, capture in product((False, True), ("graph", "galerkin"), (False, True)):
            with self.subTest(mixed=mixed, operator=operator, capture=capture), wp.ScopedDevice("cuda:0"):
                builder = newton.ModelBuilder()
                link = builder.add_link(is_kinematic=True)
                joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0, 1, 0))
                builder.add_articulation([joint])
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity()),
                    hx=0.2,
                    hy=0.2,
                    hz=0.025,
                )
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.02, -0.02, 0.006),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0, 0, -0.2),
                    dim_x=4,
                    dim_y=4,
                    cell_x=0.01,
                    cell_y=0.01,
                    mass=0.001,
                    tri_ke=5e4,
                    tri_ka=5e4,
                    edge_ke=1.0,
                    particle_radius=0.001,
                )
                if mixed:
                    builder.add_soft_grid(
                        pos=wp.vec3(0.05, 0, 0.006),
                        rot=wp.quat_identity(),
                        vel=wp.vec3(0, 0, -0.2),
                        dim_x=2,
                        dim_y=2,
                        dim_z=2,
                        cell_x=0.01,
                        cell_y=0.01,
                        cell_z=0.01,
                        density=100,
                        k_mu=1e4,
                        k_lambda=1e4,
                        k_damp=0.1,
                        particle_radius=0.001,
                    )
                builder.color(include_bending=True)
                model = builder.finalize()
                solver = SolverMJVBDV2(
                    model,
                    mujoco_articulations=(0,),
                    joint_mode="kinematic",
                    contact_mode="full",
                    vbd_options={
                        "iterations": 6,
                        "rigid_soft_enable_dat": True,
                        "particle_enable_multilevel_correction": True,
                        "particle_multilevel_operator": operator,
                        "particle_multilevel_cluster_size": 4,
                        "particle_multilevel_coarse_iterations": 12,
                        "particle_multilevel_checkpoints": (3, 5),
                        "particle_multilevel_relaxation": 0.8,
                        "particle_multilevel_max_radius_fraction": 2.0,
                    },
                    collision_options={"soft_contact_margin": 0.004, "soft_contact_max": 2048},
                )
                state, output, control = model.state(), model.state(), model.control()
                self.assertIsNotNone(solver.vbd_solver.particle_multilevel)
                # Warm both state buffers before capturing the two-step cycle.
                solver.step(state, output, control, None, 1 / 600)
                solver.step(output, state, control, None, 1 / 600)
                if capture:
                    with wp.ScopedCapture() as captured:
                        solver.step(state, output, control, None, 1 / 600)
                        solver.step(output, state, control, None, 1 / 600)
                for _ in range(45):
                    if capture:
                        wp.capture_launch(captured.graph)
                    else:
                        solver.step(state, output, control, None, 1 / 600)
                        solver.step(output, state, control, None, 1 / 600)
                    q = state.particle_q.numpy()
                    self.assertTrue(np.isfinite(q).all())
                    self.assertGreaterEqual(float(q[:, 2].min()), -1e-6)
                self.assertLess(float(q[:25, 2].mean()), 0.003)

    def test_dat_clipping_excludes_further_chebyshev_extrapolation(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                reference = wp.zeros(2, dtype=wp.vec3)
                q = wp.zeros(2, dtype=wp.vec3)
                excluded = wp.zeros(2, dtype=int)
                wp.launch(
                    _apply_particles,
                    2,
                    [
                        reference,
                        wp.array(((0.002, 0, 0), (0.0001, 0, 0)), dtype=wp.vec3),
                        wp.array((0.5, 1.0), dtype=float),
                        0.001,
                        q,
                        excluded,
                    ],
                )
                np.testing.assert_array_equal(excluded.numpy(), (1, 0))
                correction = wp.zeros(2, dtype=wp.vec3)
                wp.launch(
                    accelerate_particle_iteration_chebyshev_guarded,
                    2,
                    [q, reference, wp.full(2, 0.001), wp.ones(2, dtype=int), excluded, 1.5, 1.0, correction],
                )
                np.testing.assert_array_equal(correction.numpy()[0], (0.0, 0.0, 0.0))
                self.assertGreater(float(correction.numpy()[1, 0]), 0.0)

    def test_wrong_side_recovers_but_cannot_deepen(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                out = wp.zeros(3)
                wp.launch(_plane_probe, 1, [out])
                np.testing.assert_allclose(out.numpy(), (0.425, 0.0, 1.0), atol=1.0e-6)

    def test_joint_plane_truncates_both_sides_beyond_worker_grid(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                particle_t, body_t = wp.ones(1), wp.ones(1)
                count = 4097
                indices = np.full((count, 3), -1, dtype=np.int32)
                indices[-1] = (0, -1, -1)
                wp.launch(
                    apply_rigid_soft_truncation,
                    4096,
                    [
                        wp.array([count], dtype=int),
                        wp.array(indices, dtype=wp.vec3i),
                        wp.zeros(count, dtype=int),
                        wp.zeros(count, dtype=wp.vec3),
                        wp.array(np.tile((0, 0, 1), (count, 1)), dtype=wp.vec3),
                        wp.array(np.tile((1, 0, 0), (count, 1)), dtype=wp.vec3),
                        wp.array([0], dtype=int),
                        wp.array([[0, 0, 0.01]], dtype=wp.vec3),
                        wp.array([[0, 0, -0.02]], dtype=wp.vec3),
                        wp.array([wp.transform_identity()], dtype=wp.transform),
                        wp.array([wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity())], dtype=wp.transform),
                        wp.array([[0, 0, 0]], dtype=wp.vec3),
                        0.85,
                        False,
                        particle_t,
                        body_t,
                    ],
                )
                pt, bt = float(particle_t.numpy()[0]), float(body_t.numpy()[0])
                self.assertLess(pt, 1.0)
                self.assertLess(bt, 1.0)
                self.assertGreater(0.01 - 0.02 * pt, 0.02 * bt)

    def test_kinematic_pose_is_preserved(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                target = wp.transform(wp.vec3(0, 0, 1), wp.quat_identity())
                q = wp.array([target, target], dtype=wp.transform)
                wp.launch(
                    apply_body_truncation_ts,
                    2,
                    [
                        wp.array([wp.transform_identity()] * 2, dtype=wp.transform),
                        wp.array([2, 1], dtype=int),
                        wp.zeros(2, dtype=wp.vec3),
                        wp.zeros(2),
                        wp.ones(2),
                        wp.full(2, 0.01),
                        q,
                    ],
                )
                self.assertAlmostEqual(float(q.numpy()[0, 2]), 1.0)
                self.assertAlmostEqual(float(q.numpy()[1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
