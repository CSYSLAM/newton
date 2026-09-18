# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise bread-cube surface contact on CPU without starting the robot scene."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD
from newton._src.solvers.mjvbd_v2.vbd.tri_mesh_collision import TriMeshCollisionDetector
from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing


class TestBreadCubeContact(unittest.TestCase):
    def test_compressed_cube_lands_and_separates_from_cloth(self):
        """Release a pinched cube onto cloth and separate it without a trailing vertex."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(-0.09, -0.09, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=10,
            dim_y=10,
            cell_x=0.018,
            cell_y=0.018,
            mass=packing.BAG_SURFACE_DENSITY * 0.018**2,
            fix_left=True,
            fix_right=True,
            fix_top=True,
            fix_bottom=True,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=0.4,
            edge_ke=60,
            edge_kd=2,
            particle_radius=0.0012,
        )
        start, first_face = builder.particle_count, builder.tri_count
        packing.add_soft_cube(builder, position=(0, 0, 0.08))
        end = builder.particle_count
        faces = np.asarray(builder.tri_indices[first_face:])
        builder.color(include_bending=True)
        model = builder.finalize(device="cpu")
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 2e5, 10.0, 0.6
        solver = SolverVBD(
            model,
            iterations=12,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.0012,
            particle_self_contact_margin=0.003,
            particle_rest_shape_contact_exclusion_radius=0.0015,
            particle_external_vertex_contact_filtering_map=packing.soft_cube_contact_filter(
                start, end, faces, model.tri_count
            ),
            friction_epsilon=0.001,
        )
        state, output = model.state(), model.state()
        rest = model.particle_q.numpy()
        points = rest.copy()
        points[start:, 0] *= (2 * packing.SOFT_CUBE_OPENING - 0.001403) / packing.SOFT_CUBE_SIZE
        state.particle_q.assign(points)
        tets = model.tet_indices.numpy()
        rest_volumes = np.linalg.det(rest[tets[:, 1:]] - rest[tets[:, :1]])
        control = model.control()
        for step in range(225):
            if step == 180:
                landed = state.particle_q.numpy()
                self.assertLess(float(landed[start:, 2].min()), 0.005)
                velocity = state.particle_qd.numpy()
                velocity[start:] = (0.2, 0, 1.0)
                state.particle_qd.assign(velocity)
            solver.step(state, output, control, None, 1 / 360)
            state, output = output, state
            points = state.particle_q.numpy()
            ratios = np.linalg.det(points[tets[:, 1:]] - points[tets[:, :1]]) / rest_volumes
            self.assertTrue(np.isfinite(points).all())
            self.assertGreater(float(ratios.min()), 0.2)
            self.assertLess(float(np.linalg.norm(np.ptp(points[start:], axis=0))), 0.15)
        self.assertGreater(float(points[start:, 2].min() - points[:start, 2].max()), 0.01)

    def test_buried_nodes_do_not_contact_cloth(self):
        """Remove internal-node contacts while retaining the exposed cube boundary."""
        builder = newton.ModelBuilder()
        packing.add_soft_cube(builder, position=(0, 0, 0))
        end = builder.particle_count
        faces = np.asarray(builder.tri_indices).copy()
        # Put a sheet near both an internal node and exposed boundary nodes.
        # This mimics a deeply compressed contact and tests its topology filter.
        for p in ((-0.1, -0.1, 0.0005), (0.1, -0.1, 0.0005), (0, 0.1, 0.0005)):
            builder.add_particle(wp.vec3(*p), wp.vec3(), 0)
        builder.add_triangle(end, end + 1, end + 2)
        model = builder.finalize(device="cpu")
        filters = packing.soft_cube_contact_filter(0, end, faces, model.tri_count)
        self.assertEqual(len(filters), 27)
        self.assertEqual(end, 125)
        np.testing.assert_allclose(model.particle_mass.numpy().sum(), 0.12005, rtol=1e-5)
        interior = np.asarray(list(filters), dtype=int)
        for filtered in (False, True):
            detector = TriMeshCollisionDetector(
                model,
                external_vertex_triangle_filtering_map=filters if filtered else None,
                topological_contact_filter_threshold=2,
            )
            detector.vertex_triangle_collision_detection(0.0012)
            counts = detector.vertex_colliding_triangles_count.numpy()
            if filtered:
                self.assertEqual(int(counts[interior].sum()), 0)
            else:
                self.assertGreater(int(counts[interior].sum()), 0)
            self.assertGreater(int(counts[np.unique(faces)].sum()), 0)


if __name__ == "__main__":
    unittest.main()
