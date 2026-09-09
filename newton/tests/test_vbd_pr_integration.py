# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise compatibility boundaries between the locally integrated VBD PRs."""

import unittest
from unittest.mock import patch

import newton
from newton.tests.test_solver_vbd import _run_sphere_drop
from newton.tests.test_solver_vbd_joint_friction import _build_actuated_mimic_model, _build_single_dof_model


class TestVBDPRIntegration(unittest.TestCase):
    def test_sparse_rejects_joint_friction(self):
        """Reject friction rather than silently omitting it in sparse assembly."""
        model = _build_single_dof_model("cpu", newton.JointType.REVOLUTE, 1.0)
        with self.assertRaisesRegex(ValueError, "joint friction"):
            newton.solvers.SolverVBD(model, rigid_compliant_alm=True, rigid_articulation_solve="block_sparse_joints")

    def test_sparse_rejects_mimic(self):
        """Reject mimic relationships that the sparse solve does not assemble."""
        model = _build_actuated_mimic_model("cpu", 0.0)
        model.joint_friction.zero_()
        with self.assertRaisesRegex(ValueError, "mimic"):
            newton.solvers.SolverVBD(model, rigid_compliant_alm=True, rigid_articulation_solve="block_sparse_joints")

    def test_sparse_friction_notification(self):
        """Reject unsupported friction added after sparse solver construction."""
        model = _build_single_dof_model("cpu", newton.JointType.REVOLUTE, 0.0)
        solver = newton.solvers.SolverVBD(
            model, rigid_compliant_alm=True, rigid_articulation_solve="block_sparse_joints"
        )
        model.joint_friction.fill_(1.0)
        with self.assertRaisesRegex(ValueError, "joint friction"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_sparse_dat_sphere_drop(self):
        """Keep a fast sparse-solved sphere separated from cloth using DAT."""
        solver_type = newton.solvers.SolverVBD

        def make_solver(model, **kwargs):
            return solver_type(
                model, **kwargs, rigid_compliant_alm=True, rigid_articulation_solve="block_sparse_joints"
            )

        with patch.object(newton.solvers, "SolverVBD", side_effect=make_solver):
            penetration, height = _run_sphere_drop("cpu", True, frames=60)
        self.assertLess(penetration, 1.0e-4)
        self.assertGreater(height, -0.5)


if __name__ == "__main__":
    unittest.main()
