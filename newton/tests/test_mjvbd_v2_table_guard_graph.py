# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify table-guard graph replay follows every new joint command."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import TABLE, TABLE_FRONT, WORKSPACE_X, _mesh_intersects_table
from newton.examples.mjvbdv2.support.table_clearance import TableClearanceGuard


class TestTableGuardGraph(unittest.TestCase):
    def test_moving_command(self):
        """Reject penetrating commands after safe commands on CPU and CUDA."""
        mesh = newton.Mesh.create_box(0.05, 0.05, 0.05)
        points = np.asarray(mesh.vertices, dtype=np.float64)
        indices = np.asarray(mesh.indices).reshape(-1, 3)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            builder = newton.ModelBuilder()
            body = builder.add_link()
            builder.add_joint_free(child=body)
            builder.add_articulation([0])
            model = builder.finalize(device=device)
            example = SimpleNamespace(
                model=model,
                ik_model=model,
                robot_coords=model.joint_coord_count,
                table_guard_q=wp.clone(model.joint_q),
                table_guard_state=model.state(),
                table_guard_meshes=[(0, points, indices)],
                table_guard_bodies=np.array([0]),
                table_guard_centers=np.zeros((1, 3)),
                table_guard_extents=np.full((1, 3), 0.05),
                sim_time=0.0,
            )
            guard = TableClearanceGuard(
                example,
                (TABLE_FRONT - 0.001, -0.761, TABLE - 0.051),
                (1.501 + WORKSPACE_X, 0.761, TABLE + 0.001),
                _mesh_intersects_table,
            )
            for on_device in (False, True):
                for height, intersects in (
                    (TABLE + 0.3, False),
                    (TABLE - 0.01, True),
                    (TABLE + 0.3, False),
                    (TABLE - 0.01, True),
                ):
                    command = model.joint_q.numpy().copy()
                    command[:3] = ((TABLE_FRONT + 1.5 + WORKSPACE_X) / 2, 0, height)
                    command[3:7] = (0, 0, 0, 1)
                    argument = wp.array(command, device=device) if on_device else command
                    if intersects:
                        with self.assertRaisesRegex(RuntimeError, "Robot/table clearance rejected"):
                            guard(argument)
                    else:
                        guard(argument)
            self.assertEqual(guard.graph is not None, model.device.is_cuda)


if __name__ == "__main__":
    unittest.main()
