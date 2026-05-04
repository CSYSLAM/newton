# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cup and cube drop example.

Loads a cup mesh from USD as a kinematic rigid body (floating in air),
then drops a cube above the cup so it falls into the cup.

Command: python -m newton.examples cup_drop
"""

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = newton.ModelBuilder()

        cup_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e6,
            kd=1.0e3,
            mu=0.5,
            density=0.0,
            is_hydroelastic=True,
            kh=1.0e7,
        )

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            ke=1.0e5,
            kd=1.0e2,
            mu=0.5,
            density=1000.0,
        )

        usd_stage = Usd.Stage.Open("newton/_src/usd/cup_new.usdc")
        cup_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/柱体"))
        cup_mesh.build_sdf(
            max_resolution=128,
            narrow_band_range=(-0.05, 0.05),
            margin=0.01,
        )

        cup_pos = wp.vec3(0.0, 0.0, 2.0)
        body_cup = builder.add_body(
            xform=wp.transform(p=cup_pos, q=wp.quat_identity()),
            is_kinematic=True,
            label="cup",
        )
        builder.add_shape_mesh(body_cup, mesh=cup_mesh, cfg=cup_cfg, label="cup_mesh")

        cube_half_size = 0.3
        cube_pos = wp.vec3(0.0, 0.0, 4.5)
        body_cube = builder.add_body(
            xform=wp.transform(p=cube_pos, q=wp.quat_identity()),
            label="cube",
        )
        builder.add_shape_box(
            body_cube,
            hx=cube_half_size,
            hy=cube_half_size,
            hz=cube_half_size,
            cfg=cube_cfg,
            label="cube_box",
        )

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=20)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -6.0, 13.0),
            pitch=-65.0,
            yaw=135.0,
        )

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
