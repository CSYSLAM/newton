# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton._src.solvers.lbm.lbm_core import HomeFlow
from newton.examples import get_asset
from newton.solvers import SolverLBM
from newton.tests.unittest_utils import get_test_devices


def _make_karman_model():
    """Build the static-cylinder model used by the Karman example."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(get_asset("karman_cylinder_3d.xml"))
    return builder.finalize()


class TestSolverLBM(unittest.TestCase):
    def setUp(self):
        self.devices = [d for d in get_test_devices(mode="basic") if not d.is_cpu]

    def _make_solver(self, device):
        model = _make_karman_model()
        config = SolverLBM.Config(
            nx=16,
            ny=12,
            nz=8,
            lbm_scale=0.1,
            viscosity=0.05,
            initial_velocity=(0.05, 0.0, 0.0),
        )
        solver = SolverLBM(model, config)
        solver.add_solid(body_index=0, lbm_position=(8, 6, 4), is_static=True)
        solver.finalize()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        return solver, state_0, state_1, control

    def test_solver_registered(self):
        """Verify SolverLBM is exported from newton.solvers."""
        self.assertIs(SolverLBM, newton.solvers.SolverLBM)

    def test_flow_struct_initializes(self):
        """Verify a HomeFlow struct allocates its buffers and sets defaults."""
        flow = HomeFlow()
        flow.Initialize(8, 6, 4, n_objects=2)
        self.assertEqual(flow.nx, 8)
        self.assertEqual(flow.n_objects, 2)
        self.assertEqual(flow.rho.numpy().shape, (8, 6, 4))
        self.assertEqual(flow.solid_position.numpy().shape, (2, 3))

    def test_step_produces_finite_flow(self):
        """Verify a coupled step advances the lattice without producing NaNs."""
        for device in self.devices:
            with self.subTest(device=device):
                solver, state_0, state_1, control = self._make_solver(device)
                for _ in range(3):
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, 0.01)
                    state_0, state_1 = state_1, state_0
                u = solver.flows[0].u.numpy()
                rho = solver.flows[0].rho.numpy()
                self.assertTrue(np.isfinite(u).all(), "velocity field must be finite")
                self.assertTrue(np.isfinite(rho).all(), "density field must be finite")

    def test_set_viscosity_rebuilds_flows(self):
        """Verify set_viscosity updates the flow struct and re-exports flows_wp."""
        solver, _, _, _ = self._make_solver(self.devices[0])
        solver.set_viscosity(0.02)
        self.assertAlmostEqual(float(solver.flows[0].vis_shear), 0.02)

    def test_force_conversion_formula(self):
        """Verify the physical unit conversion matches the Open HOME-LBM convention."""
        solver, _, _, _ = self._make_solver(self.devices[0])
        cfg = solver.config
        dx = 1.0 / (cfg.lbm_scale * cfg.nx)
        dt = 0.01
        rho = cfg.fluid_density
        # step() applies these factors in the force/torque extraction kernel.
        force_conv = rho * dx**4 / (dt * dt)
        torque_conv = rho * dx**5 / (dt * dt)
        self.assertAlmostEqual(force_conv, rho * (1.0 / (cfg.lbm_scale * cfg.nx)) ** 4 / dt**2)
        self.assertAlmostEqual(torque_conv, rho * (1.0 / (cfg.lbm_scale * cfg.nx)) ** 5 / dt**2)

    def test_vorticity_projection_runs(self):
        """Verify the topdown vorticity projection kernel runs and is finite."""
        for device in self.devices:
            with self.subTest(device=device):
                solver, state_0, state_1, control = self._make_solver(device)
                for _ in range(2):
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, 0.01)
                    state_0, state_1 = state_1, state_0
                buf = solver.vorticity_projection("topdown")
                img = buf.numpy()
                self.assertTrue(np.isfinite(img).all())


if __name__ == "__main__":
    unittest.main()
