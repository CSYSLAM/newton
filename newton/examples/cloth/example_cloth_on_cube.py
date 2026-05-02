# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sim Cloth on Cube
#
# This simulation demonstrates a cloth falling onto a rigid body cube
# placed on the ground. The cloth drapes over the cube and settles,
# showing cloth-rigid body interaction with the VBD solver.
#
# Command: python -m newton.examples cloth_on_cube
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10
        self.sim_time = 0.0

        self.viewer = viewer

        # Cube half-size [m] — small cube centered under the cloth
        cube_half = 0.1

        builder = newton.ModelBuilder()

        # Ground with stiff contact so the cube doesn't sink in
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = 1.0e5
        ground_cfg.kd = 1.0e0
        ground_cfg.mu = 0.5
        builder.add_ground_plane(cfg=ground_cfg)

        # Static cube — infinite mass so the cloth drapes over it without moving it
        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = 0.0
        cube_cfg.has_particle_collision = True
        body_cube = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, cube_half),
                q=wp.quat_identity(),
            ),
            mass=0.0,
            label="cube",
        )
        builder.add_shape_box(
            body_cube,
            hx=cube_half,
            hy=cube_half,
            hz=cube_half,
            cfg=cube_cfg,
        )

        # Cloth grid centered above the cube (pos is the corner, so offset by -half_size)
        cloth_size = 1.6  # [m]
        dim = 64
        cell = cloth_size / dim
        builder.add_cloth_grid(
            pos=wp.vec3(-cloth_size / 2, -cloth_size / 2, 1.5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=cell,
            cell_y=cell,
            mass=0.001,
            fix_left=False,
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=8.0e1,
            tri_ka=8.0e1,
            tri_kd=5.0e-1,
            edge_ke=5.0e-2,
            edge_kd=5.0e-1,
            particle_radius=0.015,
        )

        builder.color(include_bending=True)
        self.model = builder.finalize()

        self.model.soft_contact_ke = 5.0e3
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.8

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.005,
            particle_self_contact_margin=0.01,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Use explicit CollisionPipeline with margin for reliable particle-rigid contact
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=0.01,
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
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > 0.0,
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cube is above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
