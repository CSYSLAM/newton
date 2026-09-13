# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare opted-in analytic LM batching with independent ordinary solvers."""

import unittest

import numpy as np
import warp as wp

import newton.ik as ik
from newton.tests.test_ik import _build_two_link_planar


class TestIKCudaFastFallback(unittest.TestCase):
    def test_cpu_fallback(self):
        """Retain the ordinary optimizer when CUDA scheduling is requested on CPU."""
        model = _build_two_link_planar("cpu", requires_grad=False)
        target = wp.array([[1.3, 0.5, 0]], dtype=wp.vec3, device=model.device)
        solver = ik.IKSolver(
            model,
            n_problems=1,
            objectives=[ik.IKObjectivePosition(1, wp.vec3(0.5, 0, 0), target)],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            parallel_objectives=False,
            enable_cuda_fast_path=True,
        )
        self.assertIsNone(solver._impl._cuda_fast)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_differentiable_fallback(self):
        """Avoid disabling gradients when analytic objectives use a differentiable model."""
        model = _build_two_link_planar("cuda:0", requires_grad=True)
        target = wp.array([[1.3, 0.5, 0]], dtype=wp.vec3, device=model.device)
        solver = ik.IKSolver(
            model,
            n_problems=1,
            objectives=[ik.IKObjectivePosition(1, wp.vec3(0.5, 0, 0), target)],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            parallel_objectives=False,
            enable_cuda_fast_path=True,
        )
        self.assertIsNone(solver._impl._cuda_fast)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestIKCudaFast(unittest.TestCase):
    def test_exact_state_and_instance_isolation(self):
        """Preserve all LM state for reachable/unreachable targets and odd iteration tails."""
        model = _build_two_link_planar("cuda:0", requires_grad=False)
        fields = ("residuals", "lambda_values", "costs", "accept_flags", "dq_dof", "joint_q_proposed")
        for batch in (1, 4):
            target = wp.array(np.tile([1.3, 0.5, 0.0], (batch, 1)), dtype=wp.vec3, device=model.device)
            rotation_target = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (batch, 1)), dtype=wp.vec4, device=model.device)
            solvers = []
            for fast in (False, True):
                objectives = [
                    ik.IKObjectivePosition(1, wp.vec3(0.5, 0, 0), target),
                    ik.IKObjectiveRotation(1, wp.quat_identity(), rotation_target, weight=0.3),
                    ik.IKObjectiveJointLimit(
                        wp.full(2, -2.0, device=model.device), wp.full(2, 2.0, device=model.device), weight=30
                    ),
                ]
                solvers.append(
                    ik.IKSolver(
                        model,
                        n_problems=batch,
                        objectives=objectives,
                        jacobian_mode=ik.IKJacobianType.ANALYTIC,
                        parallel_objectives=False,
                        enable_cuda_fast_path=fast,
                    )._impl
                )
            self.assertIsNone(solvers[0]._cuda_fast)
            self.assertIsNotNone(solvers[1]._cuda_fast)
            coordinates = [wp.zeros((batch, model.joint_coord_count), device=model.device) for _ in solvers]
            for solver, q in zip(solvers, coordinates, strict=True):
                solver.step(q, q, 32, 1.0)
            for iterations in (8, 12, 31, 32, 80):
                for step_size in (0.0, 1.0):
                    graphs = []
                    for solver, q in zip(solvers, coordinates, strict=True):
                        with wp.ScopedCapture(device=model.device) as capture:
                            solver.step(q, q, iterations, step_size)
                        graphs.append(capture.graph)
                    for position in ([1.3, 0.5, 0], [5.0, -3.0, 0], [1.5, 0.0, 0.0]):
                        target.assign(np.tile(position, (batch, 1)).astype(np.float32))
                        for q, graph in zip(coordinates, graphs, strict=True):
                            q.assign(np.tile([0.1, -0.2], (batch, 1)).astype(np.float32))
                            wp.capture_launch(graph)
                        np.testing.assert_array_equal(
                            coordinates[0].numpy().view(np.uint32), coordinates[1].numpy().view(np.uint32)
                        )
                        for field in fields:
                            np.testing.assert_array_equal(
                                getattr(solvers[0], field).numpy().view(np.uint32),
                                getattr(solvers[1], field).numpy().view(np.uint32),
                                err_msg=field,
                            )


if __name__ == "__main__":
    unittest.main()
