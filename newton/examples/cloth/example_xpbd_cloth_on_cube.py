# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example XPBD Cloth on Cube
#
# This simulation demonstrates a cloth falling onto a rigid cube using
# the XPBD solver. The cloth uses spring-based constraints and bending
# resistance to drape naturally over the cube edges.
#
# Command: python -m newton.examples xpbd_cloth_on_cube
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
        self.iterations = 15
        self.sim_time = 0.0

        self.viewer = viewer

        cube_half = 0.2

        builder = newton.ModelBuilder()

        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = 1.0e5
        ground_cfg.kd = 1.0e0
        builder.add_ground_plane(cfg=ground_cfg)

        # Static cube
        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = 0.0
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

        # Cloth with spring-based XPBD constraints
        builder.add_cloth_grid(
            pos=wp.vec3(-1.6, -1.6, 3.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=32,
            dim_y=32,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            add_springs=True,
            spring_ke=5.0e1,
            spring_kd=5.0e0,
            edge_ke=1.0e-1,
            edge_kd=0.0,
            particle_radius=0.05,
        )

        self.model = builder.finalize()

        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 5.0e0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverXPBD(
            model=self.model,
            iterations=self.iterations,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

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
            self.model.collide(self.state_0, self.contacts)
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


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
