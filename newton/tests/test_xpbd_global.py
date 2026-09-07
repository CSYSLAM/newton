# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent small-system checks for experimental global XPBD constraints."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd import fem_global
from newton.tests.test_xpbd_fem import _layers


class TestGlobalXPBD(unittest.TestCase):
    def test_public_options(self):
        """Reject unsupported solvers and invalid linear iteration counts."""
        model = _layers("cpu")
        for kwargs in (
            {"particle_fem_solver": "unknown"},
            {"particle_fem_solver": "global"},
            {"particle_fem": True, "particle_fem_linear_iterations": 7},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                newton.solvers.SolverXPBD(model, **kwargs)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                newton.solvers.SolverXPBD(
                    model, particle_fem=True, particle_fem_solver="global", particle_fem_linear_iterations=value
                )

    def test_free_translation_without_contact(self):
        """Zero contact radius must bypass detection and preserve a free translation."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            # Exactly coincident layers are allowed when self-contact is disabled.
            model = _layers(device, gap=0.0)
            solver = newton.solvers.SolverXPBD(model, particle_fem=True, particle_fem_solver="global")
            state, out = model.state(), model.state()
            speed = np.tile(np.array([0.02, -0.01, 0.03], dtype=np.float32), (model.particle_count, 1))
            initial = state.particle_q.numpy()
            state.particle_qd.assign(speed)
            solver.step(state, out, None, None, 0.001)
            np.testing.assert_allclose(out.particle_q.numpy(), initial + 0.001 * speed, atol=1e-8)
            np.testing.assert_allclose(out.particle_qd.numpy(), speed, atol=1e-5)
            self.assertEqual(int(solver._fem.count.numpy()[0]), 0)
            self.assertEqual(int(solver._fem.status.numpy()[0]), 0)

    def test_overflow_rejects_global_motion(self):
        """Global elasticity and line search must not bypass a failed DAT transaction."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            model = _layers(device, gap=0.0005)
            solver = newton.solvers.SolverXPBD(
                model,
                particle_fem=True,
                particle_fem_solver="global",
                iterations=2,
                particle_self_contact_radius=0.001,
                particle_self_contact_margin=0.01,
                particle_self_contact_max=1,
            )
            state, out = model.state(), model.state()
            initial = state.particle_q.numpy()
            state.particle_qd.fill_(wp.vec3(0.0, 0.0, 0.1))
            solver.step(state, out, None, None, 0.001)
            self.assertNotEqual(int(solver._fem.status.numpy()[0]) & 1, 0)
            np.testing.assert_array_equal(out.particle_q.numpy(), initial)
            with self.assertRaisesRegex(RuntimeError, "overflow"):
                solver.validate_particle_contacts()

    def test_normal_geometric_stiffness(self):
        """Match tensile normal curvature; discard only negative compressive curvature."""
        model = _layers("cpu", gap=0.2)
        fem = newton.solvers.SolverXPBD(model, particle_fem=True, iterations=1)._fem
        g = fem_global.GlobalConstraints(fem)
        faces, poses = model.tri_indices.numpy(), model.tri_poses.numpy()
        areas, materials = model.tri_areas.numpy(), model.tri_materials.numpy()
        rest = model.particle_q.numpy().astype(float)
        dt = 0.001
        for scale in (1.0, 1.3, 0.8):
            p = rest.copy()
            p[:, :2] *= scale
            fem.pos.assign(p)
            fem.prev.assign(p)
            fem._launch(
                fem_global.build_elastic,
                max(model.tri_count, model.edge_count),
                [
                    fem.pos,
                    fem.prev,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_areas,
                    model.tri_materials,
                    model.edge_indices,
                    model.edge_rest_angle,
                    model.edge_bending_properties,
                    dt,
                    g.gradients,
                    g.weights,
                    g.bias,
                ],
            )
            gradients, weights = g.gradients.numpy(), g.weights.numpy()
            for t, ids in enumerate(faces):
                direction = np.array([0.3, -0.4, 0.7])
                curvature = 0.0
                for row in range(5 * t + 3, 5 * t + 5):
                    self.assertGreaterEqual(weights[row], 0.0)
                    derivative = sum(direction[j] * gradients[4 * row + j, 2] for j in range(3))
                    curvature += weights[row] * derivative**2
                if scale <= 1.0:
                    self.assertLess(abs(curvature), 1e-10)
                    continue

                def energy(offset, p=p, ids=ids, direction=direction, t=t):
                    tri = p[ids].copy()
                    tri[:, 2] += offset * direction
                    f = np.stack((tri[1] - tri[0], tri[2] - tri[0]), axis=1) @ poses[t]
                    jac = np.linalg.norm(np.cross(f[:, 0], f[:, 1]))
                    mu, lam = materials[t, :2]
                    return 0.5 * dt**2 * areas[t] * (mu * (np.sum(f * f) - 2 * jac) + (mu + lam) * (jac - 1) ** 2)

                eps = 1e-5
                expected = (energy(eps) - 2 * energy(0) + energy(-eps)) / eps**2
                np.testing.assert_allclose(curvature, expected, rtol=2e-5, atol=1e-10)

    def test_operator_and_pcg(self):
        """Match assembled matrix products and a dense direct solve including fixed vertices."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for contact in (False, True):
                model = _layers(device, gap=0.0005 if contact else 0.02)
                flags = model.particle_flags.numpy()
                flags[0] = 0
                model.particle_flags.assign(flags)
                model.soft_contact_kd = 0.1
                model.soft_contact_mu = 0.2
                solver = newton.solvers.SolverXPBD(
                    model,
                    particle_fem=True,
                    iterations=1,
                    particle_self_contact_radius=0.001,
                    particle_self_contact_margin=0.01,
                )
                fem = solver._fem
                global_solver = fem_global.GlobalConstraints(fem, 32)
                fem.global_constraints = global_solver
                state, out = model.state(), model.state()
                p = state.particle_q.numpy()
                p[1, 0] += 0.001
                p[2, 1] -= 0.001
                state.particle_q.assign(p)
                velocity = np.zeros_like(p)
                if contact:
                    # Prediction crosses the other layer; DAT must truncate it.
                    velocity[3:, 2] = -2.0
                state.particle_qd.assign(velocity)
                fem.targets.assign(p)
                solver.step(state, out, None, None, 0.001)
                g = global_solver
                ids, grads, weights, bias = g.ids.numpy(), g.gradients.numpy(), g.weights.numpy(), g.bias.numpy()
                n = model.particle_count
                mass = model.particle_mass.numpy()
                matrix = np.diag(np.repeat(mass, 3).astype(float))
                primal_residual = g.backup.numpy() - g.target.numpy()
                if contact:
                    self.assertGreater(np.linalg.norm(primal_residual), 1e-4)
                rhs = (-mass[:, None] * primal_residual).flatten().astype(float)
                for row, v in enumerate(ids):
                    gradient = np.zeros((n, 3))
                    for j, i in enumerate(v):
                        gradient[i] += grads[4 * row + j]
                    gradient = gradient.flatten()
                    matrix += weights[row] * np.outer(gradient, gradient)
                    rhs -= bias[row] * gradient
                count = int(g.active_count.numpy()[0])
                for slot in g.active_slots.numpy()[:count]:
                    jacobian = np.zeros((3, 3 * n))
                    for i, b in zip(fem.pairs.numpy()[slot], g.bary.numpy()[slot], strict=True):
                        jacobian[:, 3 * i : 3 * i + 3] += np.eye(3) * b
                    matrix += jacobian.T @ g.matrices.numpy()[slot] @ jacobian
                    rhs -= jacobian.T @ g.contact_bias.numpy()[slot]
                free = np.repeat(flags != 0, 3)
                expected = np.zeros(3 * n)
                expected[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs[free])
                np.testing.assert_allclose(g.solution.numpy().flatten(), expected, rtol=2e-4, atol=2e-8)
                rng = np.random.default_rng(15)
                direction = rng.normal(size=3 * n).astype(np.float32)
                direction[~free] = 0
                g.direction.assign(direction.reshape(-1, 3))
                g.dots.zero_()
                fem._launch(
                    fem_global.multiply_rows,
                    g.workers,
                    [
                        g.direction,
                        g.ids,
                        g.gradients,
                        g.weights,
                        g.active_count,
                        g.active_slots,
                        fem.pairs,
                        g.bary,
                        g.matrices,
                        g.workers,
                        g.values,
                        g.contact_values,
                    ],
                )
                fem._launch(
                    fem_global.gather_product,
                    n,
                    [
                        g.direction,
                        model.particle_inv_mass,
                        model.particle_flags,
                        g.offsets,
                        g.links,
                        g.gradients,
                        g.values,
                        g.contact_offsets,
                        g.contact_links,
                        g.bary,
                        g.contact_values,
                        g.product,
                        g.dots,
                    ],
                )
                product = matrix @ direction
                product[~free] = 0
                np.testing.assert_allclose(g.product.numpy().flatten(), product, rtol=3e-5, atol=1e-7)
                g.dots.zero_()
                g._multiply()
                np.testing.assert_allclose(g.product.numpy().flatten(), product, rtol=3e-5, atol=1e-7)
                np.testing.assert_allclose(g.dots.numpy()[2], direction @ product, rtol=3e-5, atol=1e-7)

    def test_cached_hinges_and_partial_blocks(self):
        """Condensation must retain bending, damping, anchors and partial-block reductions."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_cloth_grid(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=12,
                dim_y=4,
                cell_x=0.05,
                cell_y=0.04,
                mass=0.02,
                tri_ke=100.0,
                tri_ka=200.0,
                tri_kd=0.003,
                edge_ke=0.5,
                edge_kd=0.01,
                fix_left=True,
            )
            model = builder.finalize(device=device)
            n = model.particle_count
            self.assertEqual(n, 65)  # One extra particle beyond a full 64-thread block.
            solver = newton.solvers.SolverXPBD(
                model,
                particle_fem=True,
                particle_fem_solver="global",
                iterations=1,
            )
            g = solver._fem.global_constraints
            state, out = model.state(), model.state()
            rest = state.particle_q.numpy()
            p = rest.copy()
            p[:, 0] *= 1.1
            p[:, 2] = 0.02 * np.sin(12.0 * p[:, 0]) * np.cos(8.0 * p[:, 1])
            state.particle_q.assign(p)
            state.particle_qd.assign(np.random.default_rng(27).normal(0.0, 0.02, (n, 3)))
            solver.particle_kinematic_targets.assign(p)
            solver.step(state, out, None, None, 0.001)
            self.assertGreater(np.max(g.weights.numpy()[5 * model.tri_count :]), 0.0)
            jac = np.zeros((len(g.ids), 3 * n))
            gradients = g.gradients.numpy()
            for row, ids in enumerate(g.ids.numpy()):
                for j, i in enumerate(ids):
                    jac[row, 3 * i : 3 * i + 3] += gradients[4 * row + j]
            inv_mass = model.particle_inv_mass.numpy()
            active = (inv_mass > 0) & (model.particle_flags.numpy() != 0)
            mass = np.zeros(n)
            mass[active] = 1.0 / inv_mass[active].astype(float)
            matrix = np.diag(np.repeat(mass, 3)) + jac.T @ (g.weights.numpy()[:, None] * jac)
            rhs = -mass[:, None] * (g.backup.numpy() - g.target.numpy())
            rhs -= (jac.T @ g.bias.numpy()).reshape(n, 3)
            rhs[~active] = 0
            preconditioner = g.diagonal.numpy()
            for i in np.flatnonzero(active):
                np.testing.assert_allclose(
                    preconditioner[i],
                    np.linalg.inv(matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]),
                    rtol=2e-5,
                    atol=1e-7,
                )
            # Rebuild RHS without changing the linearization or accepted DAT state.
            g.dots.zero_()
            g._launch(
                fem_global.initialize_system,
                n,
                [
                    g.backup,
                    g.target,
                    model.particle_inv_mass,
                    model.particle_flags,
                    g.element_offsets,
                    g.element_links,
                    g.element_diagonal,
                    g.element_bias,
                    g.contact_offsets,
                    g.contact_links,
                    g.bary,
                    g.matrices,
                    g.contact_bias,
                    g.diagonal,
                    g.r,
                    g.direction,
                    g.solution,
                    g.dots,
                ],
            )
            np.testing.assert_allclose(g.r.numpy(), rhs, rtol=3e-5, atol=1e-9)
            expected_dot = np.einsum("ni,nij,nj->", rhs, preconditioner, rhs)
            np.testing.assert_allclose(g.dots.numpy()[0], expected_dot, rtol=3e-5, atol=1e-12)
            direction = np.random.default_rng(43).normal(size=(n, 3)).astype(np.float32)
            direction[~active] = 0
            g.direction.assign(direction)
            g.dots.zero_()
            g._multiply()
            product = (matrix @ direction.ravel()).reshape(n, 3)
            product[~active] = 0
            np.testing.assert_allclose(g.product.numpy(), product, rtol=3e-5, atol=1e-7)
            np.testing.assert_allclose(g.dots.numpy()[2], np.sum(direction * product), rtol=3e-5)

    def test_membrane_gradient(self):
        """Check constraint forces against finite differences of stable Neo-Hookean energy."""
        model = _layers("cpu", gap=0.2)
        solver = newton.solvers.SolverXPBD(model, particle_fem=True, iterations=1)
        fem = solver._fem
        g = fem_global.GlobalConstraints(fem)
        p = model.particle_q.numpy().astype(float)
        p += np.random.default_rng(145).normal(scale=0.01, size=p.shape)
        fem.pos.assign(p)
        fem.prev.assign(p)
        dt = 0.001
        fem._launch(
            fem_global.build_elastic,
            max(model.tri_count, model.edge_count),
            [
                fem.pos,
                fem.prev,
                model.tri_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_bending_properties,
                dt,
                g.gradients,
                g.weights,
                g.bias,
            ],
        )
        faces, rest, area, materials = (
            model.tri_indices.numpy(),
            model.tri_poses.numpy(),
            model.tri_areas.numpy(),
            model.tri_materials.numpy(),
        )

        def energy(x):
            total = 0.0
            for ids, q, a, m in zip(faces, rest, area, materials, strict=True):
                f = np.stack((x[ids[1]] - x[ids[0]], x[ids[2]] - x[ids[0]]), axis=1) @ q
                singular = np.linalg.svd(f, compute_uv=False)
                jac = np.prod(singular)
                total += 0.5 * a * (m[0] * (sum(singular**2) - 2 * jac) + (m[0] + m[1]) * (jac - 1) ** 2)
            return dt * dt * total

        gradient = np.zeros_like(p)
        for row, ids in enumerate(g.ids.numpy()[: 5 * model.tri_count]):
            for j, i in enumerate(ids):
                gradient[i] += g.bias.numpy()[row] * g.gradients.numpy()[4 * row + j]
        reference = np.zeros_like(p)
        for i in range(len(p)):
            for axis in range(3):
                step = np.zeros_like(p)
                step[i, axis] = 1e-6
                reference[i, axis] = (energy(p + step) - energy(p - step)) / 2e-6
        np.testing.assert_allclose(gradient, reference, rtol=2e-5, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
