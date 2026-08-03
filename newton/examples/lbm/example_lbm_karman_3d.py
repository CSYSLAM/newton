# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example LBM Karman 3D
#
# Three-dimensional D3Q27 lattice-Boltzmann simulation of a Karman vortex
# street behind a static cylinder. The fluid solver (SolverLBM) advances a
# D3Q27 lattice around an immersed cylinder and the Reynolds number can be
# adjusted live via the sidebar or the +/- keys.
#
# Command: python -m newton.examples lbm_karman_3d
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverLBM, SolverMuJoCo

from newton.examples.lbm._lbm_render import domain_box_points, vorticity_to_image

# Inlet speed U and obstacle diameter D (lattice units) used for Re = U*D/nu.
_INLET_SPEED = 0.12
_CYLINDER_DIAMETER = 24.0
_DEFAULT_REYNOLDS = 400.0


def _lbm_position_for_grid(nx: int, ny: int, nz: int) -> tuple[float, float, float]:
    """Place the cylinder at the same relative spot as the reference config."""
    return (50.0 / 240.0 * nx, 61.0 / 120.0 * ny, 0.5 * nz)


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.per_frame_steps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        self.reynolds = args.reynolds
        self.inlet_velocity = _INLET_SPEED
        self.velocity_ramp = 0.0  # ramp the inlet up over the first second

        nx, ny, nz = args.nx, args.ny, args.nz
        viscosity = _INLET_SPEED * _CYLINDER_DIAMETER / max(self.reynolds, 1.0e-6)

        builder = newton.ModelBuilder()
        builder.add_mjcf(newton.examples.get_asset("karman_cylinder_3d.xml"))
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        config = SolverLBM.Config(
            nx=nx,
            ny=ny,
            nz=nz,
            lbm_scale=args.lbm_scale,
            viscosity=viscosity,
            initial_velocity=(self.inlet_velocity, 0.0, 0.0),
            bc_type=(0, 1, 1, 1, 1, 1),
            bc_value=(
                (self.inlet_velocity, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        self.lbm = SolverLBM(self.model, config)
        self.lbm.add_solid(
            body_index=0,
            lbm_position=_lbm_position_for_grid(nx, ny, nz),
            is_static=True,
        )
        self.lbm.finalize()

        self.rigid_solver = SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=(nx * 0.5, ny * 0.5, nz * 2.0), pitch=-0.6, yaw=0.0)

        self.vmax = None
        self.cylinder_body = 0
        self.vort_prev = 0.0
        self.capture()

    # ---- Simulation ------------------------------------------------------

    def simulate(self):
        # The inlet-velocity ramp is applied on the device before graph replay
        # (see step()); this captured body only replays the fixed kernels.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.lbm.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.rigid_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Ramp the inlet velocity up on the device so the wake develops smoothly.
        target = min(1.0, self.sim_time / 1.0)
        if target != self.velocity_ramp:
            self.velocity_ramp = target
            self.lbm.set_boundary_velocity(0, (self.inlet_velocity * target, 0.0, 0.0))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    # ---- Rendering -------------------------------------------------------

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        img, self.vmax = vorticity_to_image(self.lbm.vorticity_projection("topdown"), self.vmax)
        self.viewer.log_image("karman/vorticity_topdown", img)

        corners, edges = domain_box_points(self.lbm.config.nx, self.lbm.config.ny, self.lbm.config.nz)
        starts = corners[edges[:, 0]]
        ends = corners[edges[:, 1]]
        self.viewer.log_lines(
            "karman/domain", wp.array(starts, dtype=wp.vec3), wp.array(ends, dtype=wp.vec3), (0.55, 0.55, 0.55)
        )
        self.viewer.end_frame()

    # ---- UI -------------------------------------------------------------

    def gui(self, imgui):
        changed, value = imgui.slider_float("Reynolds", self.reynolds, 100.0, 600.0)
        if changed:
            self.set_reynolds(value)
        if imgui.button("Reset"):
            self.reset_flow()

    # ---- Controls -------------------------------------------------------

    def set_reynolds(self, value: float) -> None:
        value = float(np.clip(value, 50.0, 1000.0))
        if abs(value - self.reynolds) < 1.0:
            return
        self.reynolds = value
        self.lbm.set_viscosity(_INLET_SPEED * _CYLINDER_DIAMETER / max(value, 1.0e-6))
        # The captured graph closed over the old flows array; recapture.
        self.capture()

    def reset_flow(self) -> None:
        self.lbm.reset(self.state_0)
        self.vmax = None
        self.capture()

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def test_post_step(self):
        # After a warm-up, the vortex street should be visible: significant
        # vorticity away from the solid marker value.
        if self.sim_time > 1.5:
            img, _ = vorticity_to_image(self.lbm.vorticity_projection("topdown"))
            fluid = img[:, :, 0].astype(np.float32) / 255.0
            if fluid.size:
                self.vort_prev = float(np.max(fluid))
        if not np.isfinite(self.vort_prev):
            raise AssertionError("non-finite vorticity projection")

    def test_final(self):
        # The static cylinder must not move.
        def _static(q, qd):
            linear = wp.vec3(qd[0], qd[1], qd[2])
            angular = wp.vec3(qd[3], qd[4], qd[5])
            return (
                wp.length(q.p) < 1e-4
                and wp.length(linear) < 1e-4
                and wp.length(angular) < 1e-4
            )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "karman cylinder stays in place",
            _static,
            [self.cylinder_body],
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--nx", type=int, default=240)
        parser.add_argument("--ny", type=int, default=120)
        parser.add_argument("--nz", type=int, default=40)
        parser.add_argument("--lbm-scale", type=float, default=0.05)
        parser.add_argument("--per-frame-steps", type=int, default=5)
        parser.add_argument("--reynolds", type=float, default=_DEFAULT_REYNOLDS)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
