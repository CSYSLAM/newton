# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Elastic Cubes Stack Drop
#
# This simulation demonstrates a configurable number of elastic cubes
# arranged in layers (4x4 per layer) at different heights, falling and
# colliding with the ground using the VBD solver.
#
# Command: uv run -m newton.examples elastic_cubes_stack_drop
#
###########################################################################

import argparse

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

        num_cubes = args.num_cubes

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        cubes_per_layer = 16  # 4x4
        num_layers = (num_cubes + cubes_per_layer - 1) // cubes_per_layer

        cube_spacing = 0.6  # horizontal spacing between cube centers
        layer_height_gap = 1.5  # vertical gap between layers

        cube_id = 0
        for layer in range(num_layers):
            for row in range(4):
                for col in range(4):
                    if cube_id >= num_cubes:
                        break
                    x = (col - 1.5) * cube_spacing
                    y = (row - 1.5) * cube_spacing
                    z = (layer + 1) * layer_height_gap
                    builder.add_soft_grid(
                        pos=wp.vec3(x, y, z),
                        rot=wp.quat_identity(),
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=4,
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
                    cube_id += 1
                if cube_id >= num_cubes:
                    break
            if cube_id >= num_cubes:
                break

        builder.color()

        self.model = builder.finalize()

        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 1e-5
        self.model.soft_contact_mu = 1.0

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

            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        num_layers = max(1, (self.model.particle_count() // (4 * 4 * 4 * 5)) + 1)
        p_lower = wp.vec3(-3.0, -3.0, -0.5)
        p_upper = wp.vec3(3.0, 3.0, float(num_layers * 2 + 2))
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
        parser = newton.examples.create_parser()
        parser.add_argument("--num-cubes", type=int, default=64, help="Number of elastic cubes (default: 32)")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
