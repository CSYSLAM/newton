# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Elastic Cube Drop
#
# This simulation demonstrates an elastic cube (tetrahedral grid) falling
# from a height and colliding with the ground using the VBD solver.
# The cube exhibits realistic Neo-Hookean elastic deformation and bouncing.
#
# Note: The XPBD solver has only experimental support for soft bodies.
# The VBD solver is recommended for volumetric soft body simulations.
#
# Command: uv run -m newton.examples elastic_cube_drop
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

        # Create an elastic cube (6x6x6 cells, 0.1m per cell = 0.6m cube)
        # Position it 2.0m above the ground
        # Use parameters similar to example_softbody_gift
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=6,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,  # Lower density like gift example
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=1e-5,
            particle_radius=0.02,
        )

        # Color the mesh for VBD solver (required!)
        builder.color()

        self.model = builder.finalize()

        # Contact parameters (use values from example_softbody_gift)
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

        # Create VBD solver with self-contact enabled
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

            # Collision detection using CollisionPipeline
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
        # Verify the cube has settled within a reasonable volume
        # Initial position: 2.0m high, 0.6m cube size
        # After falling and bouncing, it should be near ground level
        p_lower = wp.vec3(-2.0, -2.0, -0.5)
        p_upper = wp.vec3(2.0, 2.0, 3.0)
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