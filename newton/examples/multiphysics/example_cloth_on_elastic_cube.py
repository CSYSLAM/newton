# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dropping on Elastic Cube
#
# This simulation demonstrates a cloth sheet falling onto an elastic cube
# (tetrahedral grid). Both the cloth and the soft body use the VBD solver.
# The cube exhibits realistic Neo-Hookean elastic deformation as the cloth
# lands on it.
#
# Command: uv run -m newton.examples cloth_on_elastic_cube
#
###########################################################################

import numpy as np
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

        # Add elastic cube (tetrahedral grid) at ground level
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.3),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=6,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1000.0,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-4,
            particle_radius=0.02,
        )

        # Add cloth grid above the elastic cube
        builder.add_cloth_grid(
            pos=wp.vec3(-0.8, -0.8, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            fix_left=False,
            fix_right=False,
            dim_x=32,
            dim_y=32,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.001,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=1e-5,
            edge_ke=0.1,
            edge_kd=1e-3,
            particle_radius=0.03,
        )

        # Color the mesh for VBD solver (required!)
        builder.color()

        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1e-4
        self.model.soft_contact_mu = 0.8

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

        # Position camera
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(3.0, -3.0, 2.5), pitch=-25.0, yaw=45.0)

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
        # Test that bounding box size is reasonable (not exploding)
        particle_q = self.state_0.particle_q.numpy()
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)

        # Check bbox size is reasonable
        assert bbox_size < 10.0, f"Bounding box exploded: size={bbox_size:.2f}"

        # Check no excessive penetration
        assert min_pos[2] > -0.5, f"Excessive penetration: z_min={min_pos[2]:.4f}"

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