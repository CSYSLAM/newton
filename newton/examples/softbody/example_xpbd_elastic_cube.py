# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Elastic Cube Drop
#
# This simulation demonstrates an elastic cube (tetrahedral grid) falling
# onto the ground using the XPBD solver.
#
# Note: XPBD has experimental support for soft bodies. The VBD solver is
# recommended for production use. The soft_body_relaxation parameter
# controls the effective stiffness (lower = stiffer).
#
# Command: python -m newton.examples xpbd_elastic_cube
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Elastic cube: 8x8x8 cells, 0.08m per cell = 0.64m cube
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=8,
            dim_y=8,
            dim_z=8,
            cell_x=0.08,
            cell_y=0.08,
            cell_z=0.08,
            density=30.0,
            k_mu=5.0e3,
            k_lambda=5.0e3,
            k_damp=1.0e-2,
            particle_radius=0.02,
        )

        # Note: builder.color() is NOT needed for XPBD (VBD-only)

        self.model = builder.finalize()

        # Contact parameters (high kd = heavy damping to minimize bounce)
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 5.0e3
        self.model.soft_contact_mu = 1.0

        # XPBD solver
        # soft_body_relaxation is the FEM constraint compliance (alpha).
        # Internally: alpha = relaxation / dt² * inv_rest_volume.
        # With dt ≈ 0.00167 and rest volume ≈ 8.5e-5, a value of 5e-5 gives
        # effective stiffness comparable to k_mu=1e4 in the VBD solver.
        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=self.iterations,
            soft_body_relaxation=1e-3,
            soft_contact_relaxation=1e-3,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.04,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-2.0, -2.0, -0.5)
        p_upper = wp.vec3(2.0, 2.0, 3.5)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
