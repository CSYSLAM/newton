# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check PhysX-equation FEM independently of VBD and audit DAT transactions."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd import fem_contacts, fem_kernels
from newton._src.solvers.xpbd.fem_cooking import categorize, cook_copy_chains


@wp.kernel
def evaluate_membrane(p: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid() * 3
    a, b, c = fem_kernels.membrane_update(
        p[i],
        p[i + 1],
        p[i + 2],
        1.0,
        2.0,
        0.5,
        wp.mat22(1.0, 0.0, 0.0, 1.0),
        0.5,
        100.0,
        200.0,
        0.01,
    )
    out[i] = a
    out[i + 1] = b
    out[i + 2] = c


def _reference_membrane(p, q=None, area=0.5, w=None, mu=100.0, lam=200.0, dt=0.01):
    """Evaluate PhysX's two sequential constraints using an independent NumPy SVD."""
    x01, x02 = p[1] - p[0], p[2] - p[0]
    axis0 = x01 / np.linalg.norm(x01)
    axis1 = np.cross(np.cross(x01, x02), axis0)
    axis1 /= np.linalg.norm(axis1)
    basis = np.array([axis0, axis1])
    edges = np.stack((basis @ x01, basis @ x02), axis=1)
    q = np.eye(2) if q is None else q
    u, _, vh = np.linalg.svd(edges @ q)
    residual = edges @ q - u @ vh
    c = np.linalg.norm(residual)
    w = np.array([1.0, 2.0, 0.5]) if w is None else np.asarray(w)
    d = np.zeros((3, 2))
    if c > 1.0e-14:
        derivative = residual @ q.T
        g = np.stack((-derivative.sum(axis=1), derivative[:, 0], derivative[:, 1])) / c
        alpha = 1.0 / (2 * mu * area * dt**2)
        dl = -c / (np.sum(w * np.sum(g * g, axis=1)) + alpha)
        d = w[:, None] * dl * g
    edges += np.stack((d[1] - d[0], d[2] - d[0]), axis=1)
    a, b = edges[:, 0], edges[:, 1]
    c = np.linalg.det(edges) / (2 * area) - 1
    g1, g2 = np.array([b[1], -b[0]]) / (2 * area), np.array([-a[1], a[0]]) / (2 * area)
    g = np.stack((-g1 - g2, g1, g2))
    alpha = 1.0 / (lam * area * dt**2)
    d += w[:, None] * (-c / (np.sum(w * np.sum(g * g, axis=1)) + alpha)) * g
    return p + d @ basis


def _layers(device, gap=0.002):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for z in (0.0, gap):
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, z),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=[wp.vec3(-0.1, -0.1, 0.0), wp.vec3(0.1, -0.1, 0.0), wp.vec3(0.0, 0.1, 0.0)],
            indices=[0, 1, 2],
            vel=wp.vec3(0.0),
            density=1.0,
            tri_ke=100.0,
            tri_ka=100.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
        )
    return builder.finalize(device=device)


class TestXPBDFEM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])

    def test_physx_membrane_equations(self):
        """Match independently evaluated ARAP and sequential area updates."""
        rng = np.random.default_rng(182)
        p = np.tile(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (32, 1, 1))
        p += rng.normal(scale=0.2, size=p.shape)
        expected = np.stack([_reference_membrane(tri) for tri in p]).reshape(-1, 3)
        for device in self.devices:
            with self.subTest(device=device):
                inputs = wp.array(p.reshape(-1, 3), dtype=wp.vec3, device=device)
                output = wp.zeros_like(inputs)
                wp.launch(evaluate_membrane, 32, inputs=[inputs, output], device=device)
                np.testing.assert_allclose(output.numpy(), expected, atol=3.0e-6)

    def test_zero_dat_factor_is_bitwise_noop(self):
        """Never recompose or move a point rejected by DAT."""
        for device in self.devices:
            p = np.array([[0.01337921, 0.08143278, -0.00231797]], dtype=np.float32)
            accepted = wp.array(p, dtype=wp.vec3, device=device)
            base = wp.array(p + 0.002, dtype=wp.vec3, device=device)
            proposal = wp.array(p + 1.0, dtype=wp.vec3, device=device)
            wp.launch(
                fem_contacts.commit,
                1,
                inputs=[
                    base,
                    proposal,
                    wp.zeros(1, device=device),
                    0.001,
                    wp.zeros(1, dtype=int, device=device),
                    accepted,
                ],
                device=device,
            )
            np.testing.assert_array_equal(accepted.numpy(), p)

    def test_external_rebuild_preserves_detection(self):
        """Preserve complete VT/EE membership when refitting between external rebuilds."""
        for device in self.devices:
            model = _layers(device)
            options = {
                "particle_fem": True,
                "particle_self_contact_radius": 0.001,
                "particle_self_contact_margin": 0.01,
            }
            ref = newton.solvers.SolverXPBD(model, **options)
            trial = newton.solvers.SolverXPBD(model, **options)
            trial.rebuild_bvh(model.state())
            p = model.particle_q.numpy()

            def keys(fem):
                n = int(fem.count.numpy()[0])
                return sorted(tuple(row) for row in np.column_stack((fem.kinds.numpy()[:n], fem.pairs.numpy()[:n])))

            for theta in (0.0, 0.7, 1.4):
                c, s = np.cos(theta), np.sin(theta)
                moved = p @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]) + np.array([0.02 * theta, 0, 0])
                for solver in (ref, trial):
                    solver._fem.pos.assign(moved)
                    solver._fem._detect()
                self.assertEqual(keys(ref._fem), keys(trial._fem))
            with self.assertRaisesRegex(ValueError, "particle_fem"):
                newton.solvers.SolverXPBD(model).rebuild_bvh(model.state())

    def test_dat_cumulative_motion_and_sliding(self):
        """Prevent two layers crossing while allowing tangential motion inside the query bound."""
        for device in self.devices:
            with self.subTest(device=device):
                model = _layers(device)
                solver = newton.solvers.SolverXPBD(
                    model, particle_fem=True, particle_self_contact_radius=0.001, particle_self_contact_margin=0.01
                )
                fem = solver._fem
                fem._detect()
                initial = model.particle_q.numpy()
                proposal = initial.copy()
                proposal[:3, 2] += 0.03
                proposal[3:, 2] -= 0.03
                fem.candidate.assign(proposal)
                fem._commit()
                result = fem.pos.numpy()
                self.assertGreater(result[3:, 2].min() - result[:3, 2].max(), 0.0)
                # A fresh proposal still shares the original detection budget.
                proposal[:, 0] += 0.1
                fem.candidate.assign(proposal)
                fem._commit()
                self.assertLessEqual(np.linalg.norm(fem.pos.numpy() - initial, axis=1).max(), 0.004251)
                slide = initial.copy()
                slide[:, 0] += 0.001
                fem.candidate.assign(slide)
                fem._commit()
                np.testing.assert_allclose(fem.pos.numpy(), slide, atol=1.0e-7)
                fem.validate()

    def test_dat_does_not_retract_accepted_state(self):
        """Recomputing the division ratio must not undo an accepted correction."""
        for device in self.devices:
            model = _layers(device)
            fem = newton.solvers.SolverXPBD(
                model, particle_fem=True, particle_self_contact_radius=0.001, particle_self_contact_margin=0.01
            )._fem
            fem._detect()
            proposal = fem.pos.numpy()
            proposal[:3, 2] += 0.0015
            fem.candidate.assign(proposal)
            fem._commit()
            accepted = fem.pos.numpy().copy()
            # Zero incremental motion changes the division ratio from asymmetric
            # to 1/2. The old cumulative truncation incorrectly retracted A.
            fem.candidate.assign(accepted)
            for _ in range(4):
                fem._commit()
            np.testing.assert_allclose(fem.pos.numpy(), accepted, rtol=0.0, atol=1.0e-9)
            fem.validate()

    def test_physx_cooking_coverage_and_copy_chains(self):
        """Membranes/hinges occur once; remaps are disjoint forward chains."""
        faces = np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4], [5, 6, 7]])
        hinges = np.array([[0, 3, 1, 2], [1, 4, 2, 3]])
        shared, other, singles, pair_faces = categorize(faces, hinges)
        self.assertCountEqual([*singles, *pair_faces[shared].flatten()], range(len(faces)))
        self.assertCountEqual([*shared, *other], range(len(hinges)))
        # More than eight conflicting colors forces real copies/averaging.
        elements = np.array([[0, 3 * i + 1, 3 * i + 2, 3 * i + 3] for i in range(19)])
        order, ends, remap, offsets = cook_copy_chains(elements, 58)
        self.assertEqual(len(ends), 8)
        self.assertEqual(offsets[1], 3)
        self.assertEqual(len(np.unique(remap)), len(remap))
        phase = np.empty(19, dtype=int)
        start = 0
        for p, end in enumerate(ends):
            phase[start:end] = p
            start = end
        for corner_slot, dest in enumerate(remap):
            source_vertex = elements[order[corner_slot % 19], corner_slot // 19]
            if dest < 76:
                self.assertGreater(phase[dest % 19], phase[corner_slot % 19])
                self.assertEqual(source_vertex, elements[order[dest % 19], dest // 19])
            else:
                self.assertLessEqual(offsets[source_vertex], dest - 76)
                self.assertLess(dest - 76, offsets[source_vertex + 1])
        self.assertEqual(sum(remap >= 76), offsets[-1])

    def test_shared_pair_membrane_order(self):
        """Check cooked nonidentity rest frames against two sequential NumPy solves."""
        for device in self.devices:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_cloth_mesh(
                pos=wp.vec3(0),
                rot=wp.quat_identity(),
                scale=1.0,
                vertices=[wp.vec3(0, 0, 0), wp.vec3(2, 0, 0), wp.vec3(0.3, 1, 0), wp.vec3(1.7, 1.4, 0)],
                indices=[0, 1, 2, 2, 1, 3],
                vel=wp.vec3(0),
                density=1.0,
                tri_ke=100.0,
                tri_ka=200.0,
                tri_kd=0.0,
                edge_ke=0.0,
                edge_kd=0.0,
            )
            model = builder.finalize(device=device)
            fem = newton.solvers.SolverXPBD(model, particle_fem=True)._fem
            p = model.particle_q.numpy().astype(np.float64)
            p += np.random.default_rng(172).normal(scale=0.15, size=p.shape)
            batch = fem.shared_pairs
            self.assertEqual(len(batch.ids), 1)
            self.assertEqual(len(fem.nonshared_pairs.ids), 0)
            w = model.particle_inv_mass.numpy()
            q, areas = batch.poses.numpy(), batch.areas.numpy()
            a, b, c, d = batch.vertices.numpy()[0]
            expected = p.copy()
            for k, tip in enumerate((a, b)):
                ids = [c, d, tip]
                expected[ids] = _reference_membrane(expected[ids], q[k], areas[k], w[ids])
            fem.candidate.assign(p)
            fem._solve_shell(0.01)
            np.testing.assert_allclose(fem.candidate.numpy(), expected, rtol=0.0, atol=2.0e-6)

    def test_physx_damping_rigid_motion_and_fixed_vertices(self):
        """Material damping must preserve rigid motion and fixed vertex velocities."""
        for device in self.devices:
            model = _layers(device)
            materials = model.tri_materials.numpy()
            materials[:, 2] = 60.0
            model.tri_materials.assign(materials)
            p = model.particle_q.numpy()
            velocity = np.cross(np.array([0.0, 0.0, 2.0]), p) + np.array([0.1, 0.2, 0.3])
            vel = wp.array(velocity, dtype=wp.vec3, device=device)
            delta = wp.zeros_like(vel)
            inputs = [
                model.tri_indices,
                model.tri_materials,
                model.particle_q,
                vel,
                model.particle_inv_mass,
                model.particle_flags,
                0.01,
                delta,
            ]
            wp.launch(fem_kernels.membrane_damping, model.tri_count, inputs=inputs, device=device)
            np.testing.assert_allclose(delta.numpy(), 0.0, atol=1.0e-7)
            velocity[0, 0] += 1.0
            vel.assign(velocity)
            delta.zero_()
            wp.launch(fem_kernels.membrane_damping, model.tri_count, inputs=inputs, device=device)
            dv = delta.numpy()
            self.assertLess(dv[0, 0], 0.0)
            np.testing.assert_allclose((model.particle_mass.numpy()[:, None] * dv).sum(axis=0), 0, atol=1.0e-7)
            flags = model.particle_flags.numpy()
            flags[0] = 0
            model.particle_flags.assign(flags)
            delta.zero_()
            wp.launch(fem_kernels.membrane_damping, model.tri_count, inputs=inputs, device=device)
            np.testing.assert_array_equal(delta.numpy()[0], 0.0)

    def test_combined_partition_execution(self):
        """Exercise >8 colors and mass-scaled copy chains against a host reference."""
        theta = np.arange(19) * 2 * np.pi / 19
        vertices = np.concatenate((np.zeros((1, 3)), np.stack((np.cos(theta), np.sin(theta), np.zeros(19)), axis=1)))
        faces = [[0, i + 1, (i + 1) % 19 + 1] for i in range(19)]
        for device in self.devices:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            builder.add_cloth_mesh(
                pos=wp.vec3(0),
                rot=wp.quat_identity(),
                scale=1.0,
                vertices=[wp.vec3(v) for v in vertices],
                indices=np.asarray(faces).flatten(),
                vel=wp.vec3(0),
                density=1.0,
                tri_ke=100.0,
                tri_ka=200.0,
                tri_kd=0.0,
                edge_ke=0.0,
                edge_kd=0.0,
            )
            model = builder.finalize(device=device)
            fem = newton.solvers.SolverXPBD(model, particle_fem=True)._fem
            p = model.particle_q.numpy().astype(np.float64)
            p += np.random.default_rng(102).normal(scale=0.02, size=p.shape)
            expected = p.copy()
            for group in fem.membrane_groups:
                for tri in group.numpy():
                    ids = model.tri_indices.numpy()[tri]
                    expected[ids] = _reference_membrane(
                        expected[ids],
                        model.tri_poses.numpy()[tri],
                        model.tri_areas.numpy()[tri],
                        model.particle_inv_mass.numpy()[ids],
                    )
            batch = fem.shared_pairs
            count = len(batch.ids)
            offsets = batch.offsets.numpy()
            self.assertGreater(offsets[1], 1)
            remap = batch.remap.numpy()
            copies = {}
            for k, v in enumerate(batch.vertices.numpy()):
                x = np.array([copies.get(k + j * count, expected[v[j]]) for j in range(4)])
                w = model.particle_inv_mass.numpy()[v] * (offsets[v + 1] - offsets[v])
                for j in range(2):
                    ids = [2, 3, j]
                    x[ids] = _reference_membrane(
                        x[ids], batch.poses.numpy()[2 * k + j], batch.areas.numpy()[2 * k + j], w[ids]
                    )
                for j in range(4):
                    copies[remap[k + j * count]] = x[j].copy()
            for i in range(model.particle_count):
                if offsets[i + 1] > offsets[i]:
                    expected[i] = np.mean([copies[4 * count + j] for j in range(offsets[i], offsets[i + 1])], axis=0)
            fem.candidate.assign(p)
            fem._solve_shell(0.01)
            np.testing.assert_allclose(fem.candidate.numpy(), expected, rtol=0.0, atol=3.0e-6)

    def test_overflow_rejects_step(self):
        """Keep positions unchanged when candidate capacity is exhausted."""
        model = _layers(self.devices[-1])
        solver = newton.solvers.SolverXPBD(
            model,
            particle_fem=True,
            particle_self_contact_radius=0.001,
            particle_self_contact_margin=0.01,
            particle_self_contact_max=1,
        )
        a, b = model.state(), model.state()
        a.particle_qd.fill_(wp.vec3(0.0, 0.0, 10.0))
        solver.step(a, b, None, None, 0.01)
        np.testing.assert_array_equal(a.particle_q.numpy(), b.particle_q.numpy())
        with self.assertRaisesRegex(RuntimeError, "status=1"):
            solver.validate_particle_contacts()

    def test_partition_and_legacy_gate(self):
        """Check conflict-free element partitions and preserve the default XPBD path."""
        model = _layers("cpu")
        self.assertIsNone(newton.solvers.SolverXPBD(model)._fem)
        fem = newton.solvers.SolverXPBD(model, particle_fem=True)._fem
        faces = model.tri_indices.numpy()
        for group in fem.membrane_groups:
            vertices = faces[group.numpy()].flatten()
            self.assertEqual(len(vertices), len(np.unique(vertices)))
        with self.assertRaises(ValueError):
            newton.solvers.SolverXPBD(model, particle_self_contact_radius=0.001)

    def test_predict_and_graph_replay(self):
        """Advance physical time consistently and replay CUDA graphs without host readback."""
        for device in self.devices:
            model = _layers(device, gap=0.01)
            solver = newton.solvers.SolverXPBD(model, particle_fem=True, iterations=4)
            a, b = model.state(), model.state()
            a.particle_qd.fill_(wp.vec3(0.02, 0.0, 0.0))
            initial = a.particle_q.numpy().copy()
            if model.device.is_cuda:
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(a, b, None, None, 0.01)
                    solver.step(b, a, None, None, 0.01)
                for _ in range(3):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(3):
                    solver.step(a, b, None, None, 0.01)
                    solver.step(b, a, None, None, 0.01)
            np.testing.assert_allclose(a.particle_q.numpy(), initial + np.array([0.0012, 0, 0]), atol=2.0e-6)

    def test_independent_crossing_audit(self):
        """Detect a segment through a face even without a preexisting VT/EE row."""
        device = self.devices[-1]
        pos = wp.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=wp.vec3, device=device)
        triangles = wp.array([[0, 1, 2]], dtype=int, device=device)
        edges = wp.array([[3, 4]], dtype=wp.vec2i, device=device)
        lo = wp.array([[-1, -1, 0]], dtype=wp.vec3, device=device)
        hi = wp.array([[1, 1, 0]], dtype=wp.vec3, device=device)
        bvh = wp.Bvh(lo, hi)
        count = wp.zeros(1, dtype=int, device=device)
        wp.launch(
            fem_contacts.count_surface_crossings,
            1,
            inputs=[
                bvh.id,
                pos,
                triangles,
                edges,
                wp.zeros(5, dtype=int, device=device),
                count,
                wp.empty(1, dtype=wp.vec2i, device=device),
            ],
            device=device,
        )
        self.assertEqual(int(count.numpy()[0]), 1)
        # Coplanar overlaps must not silently pass the nonparallel predicate.
        coplanar = pos.numpy().copy()
        coplanar[3:] = [[-0.1, 0, 0], [0.1, 0, 0]]
        pos.assign(coplanar)
        count.zero_()
        wp.launch(
            fem_contacts.count_surface_crossings,
            1,
            inputs=[
                bvh.id,
                pos,
                triangles,
                edges,
                wp.zeros(5, dtype=int, device=device),
                count,
                wp.empty(1, dtype=wp.vec2i, device=device),
            ],
            device=device,
        )
        self.assertEqual(int(count.numpy()[0]), 1)

    def test_physx_bending_equations(self):
        """Match the signed-angle update against a finite-difference constraint gradient."""
        p = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.2], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        def angle(x):
            n0 = np.cross(x[2] - x[0], x[3] - x[0])
            n1 = np.cross(x[3] - x[1], x[2] - x[1])
            n0 /= np.linalg.norm(n0)
            n1 /= np.linalg.norm(n1)
            axis = (x[3] - x[2]) / np.linalg.norm(x[3] - x[2])
            return np.arctan2(np.dot(np.cross(n0, n1), axis), np.dot(n0, n1))

        gradient = np.zeros_like(p)
        for i in range(4):
            for axis in range(3):
                offset = np.zeros_like(p)
                offset[i, axis] = 1.0e-6
                gradient[i, axis] = (angle(p + offset) - angle(p - offset)) / 2.0e-6
        c = np.clip(angle(p) - 0.1, -np.pi / 2, np.pi / 2)
        expected = p - c * gradient / (np.sum(gradient**2) + 1 / (0.2 * 0.01**2))
        for device in self.devices:
            pos = wp.array(p, dtype=wp.vec3, device=device)
            wp.launch(
                fem_kernels.solve_bending,
                1,
                inputs=[
                    wp.array([0], dtype=int, device=device),
                    wp.array([[0, 1, 2, 3]], dtype=int, device=device),
                    wp.array([0.1], dtype=float, device=device),
                    wp.array([[0.2, 0.0]], dtype=float, device=device),
                    wp.ones(4, device=device),
                    wp.full(4, int(newton.ParticleFlags.ACTIVE), dtype=int, device=device),
                    0.01,
                    pos,
                ],
                device=device,
            )
            np.testing.assert_allclose(pos.numpy(), expected, atol=1.0e-7)

    def test_candidate_coverage_and_near_coincident_graph(self):
        """Keep all VT/EE pairs and truncate near-coincident layers under graph replay."""
        for device in self.devices:
            model = _layers(device, gap=1.0e-7)
            fem = newton.solvers.SolverXPBD(
                model, particle_fem=True, particle_self_contact_radius=0.001, particle_self_contact_margin=0.4
            )._fem
            fem._detect()
            self.assertEqual(int(fem.count.numpy()[0]), 15)  # 6 VT + 9 unique EE.
            kinds = fem.kinds.numpy()[:15]
            self.assertEqual(int(np.sum(kinds == 0)), 6)
            self.assertEqual(int(fem.status.numpy()[0]), 0)
            proposal = model.particle_q.numpy().copy()
            proposal[:3, 2] += 0.01
            proposal[3:, 2] -= 0.01
            fem.candidate.assign(proposal)
            if model.device.is_cuda:
                with wp.ScopedCapture(device=model.device) as capture:
                    fem._detect()
                    fem._commit()
                for _ in range(3):
                    wp.capture_launch(capture.graph)
            else:
                for _ in range(3):
                    fem._detect()
                    fem._commit()
            result = fem.pos.numpy()
            self.assertGreater(result[3:, 2].min() - result[:3, 2].max(), 0.0)
            fem.validate()

    def test_friction_and_contact_support(self):
        """Oppose tangential slip only inside the physical contact radius."""
        for device in self.devices:
            p = np.array([[0.0, 0.0, 0.0005], [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            anchor = p.copy()
            anchor[0, 0] -= 0.0001
            pos = wp.array(p, dtype=wp.vec3, device=device)
            delta = wp.zeros(4, dtype=wp.vec3, device=device)
            weights = wp.zeros(4, device=device)
            inputs = [
                pos,
                wp.array(anchor, dtype=wp.vec3, device=device),
                wp.ones(4, device=device),
                wp.array([int(newton.ParticleFlags.ACTIVE), 0, 0, 0], dtype=int, device=device),
                wp.array([1], dtype=int, device=device),
                wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=device),
                wp.array([0], dtype=int, device=device),
                0.001,
                1.0e6,
                0.5,
                0.01,
                1,
                delta,
                weights,
            ]
            wp.launch(fem_contacts.contact_response, 1, inputs=inputs, device=device)
            self.assertGreater(delta.numpy()[0, 2], 0.0)
            self.assertLess(delta.numpy()[0, 0], 0.0)
            np.testing.assert_array_equal(delta.numpy()[1:], 0.0)
            p[0, 2] = 0.002
            pos.assign(p)
            delta.zero_()
            weights.zero_()
            wp.launch(fem_contacts.contact_response, 1, inputs=inputs, device=device)
            np.testing.assert_array_equal(delta.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
