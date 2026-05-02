# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Elastic Cubes Drop
#
# This simulation demonstrates multiple elastic cubes falling from different
# heights and colliding with the ground using the VBD solver.
# Each cube exhibits realistic Neo-Hookean elastic deformation.
#
# Command: uv run -m newton.examples elastic_cubes_drop
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
        self.iterations = 15
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        # Create 8 elastic cubes at different positions
        # Arrange in a 2x2x2 grid pattern with varying heights
        positions = [
            wp.vec3(-0.5, -0.5, 1.0),   # bottom-left-front, lower
            wp.vec3(0.5, -0.5, 2.0),    # bottom-right-front, higher
            wp.vec3(-0.5, 0.5, 1.5),    # bottom-left-back, medium
            wp.vec3(0.5, 0.5, 2.5),     # bottom-right-back, highest
            wp.vec3(-0.5, -0.5, 3.0),   # top-left-front
            wp.vec3(0.5, -0.5, 3.5),    # top-right-front
            wp.vec3(-0.5, 0.5, 4.0),    # top-left-back
            wp.vec3(0.5, 0.5, 4.5),     # top-right-back
        ]

        for pos in positions:
            builder.add_soft_grid(
                pos=pos,
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=4,  # Smaller cube (4x4x4 cells)
                dim_y=4,
                dim_z=4,
                cell_x=0.1,
                cell_y=0.1,
                cell_z=0.1,
                density=100.0,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=1e-5,
                particle_radius=0.02,
            )

        # Color the mesh for VBD solver (required!)
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

        # Create VBD solver with self-contact enabled for cube-cube collisions
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.04,
            particle_self_contact_margin=0.06,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Use CollisionPipeline for better collision handling
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.05,
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

            # Apply gravity
            self.viewer.apply_forces(self.state_0)

            # Collision detection
            self.collision_pipeline.collide(self.state_0, self.contacts)

            # VBD step
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Verify particles are within a reasonable volume
        p_lower = wp.vec3(-3.0, -3.0, -0.5)
        p_upper = wp.vec3(3.0, 3.0, 6.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume after simulation",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)