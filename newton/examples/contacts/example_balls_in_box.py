# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body spheres falling into a box.

Spawns a large number of small rigid-body spheres above a box container
and lets them fall under gravity, colliding with each other and the box
walls.  Demonstrates dense rigid-body contact handling with the selected
broad phase method.

Command: python -m newton.examples balls_in_box
With BVH: python -m newton.examples balls_in_box --broad-phase bvh
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

BOX_HALF = 3.0
SPHERE_RADIUS = 0.15
DEFAULT_NUM_SPHERES = 300


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.num_spheres = args.num_spheres
        self.broad_phase = args.broad_phase

        builder = newton.ModelBuilder()

        # Ground plane
        builder.add_shape_plane()

        # Box walls (static, body=-1)
        wall_thickness = 0.2
        wall_half = BOX_HALF
        ht = wall_thickness / 2.0

        # +X wall
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(wall_half + ht, 0.0, wall_half)),
            hx=ht, hy=wall_half, hz=wall_half,
        )
        # -X wall
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-wall_half - ht, 0.0, wall_half)),
            hx=ht, hy=wall_half, hz=wall_half,
        )
        # +Y wall
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, wall_half + ht, wall_half)),
            hx=wall_half, hy=ht, hz=wall_half,
        )
        # -Y wall
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, -wall_half - ht, wall_half)),
            hx=wall_half, hy=ht, hz=wall_half,
        )

        # Spawn spheres in a grid above the box
        rng = np.random.default_rng(42)
        spacing = SPHERE_RADIUS * 2.5
        side = int(np.ceil(np.sqrt(self.num_spheres)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= self.num_spheres:
                    break
                x = (i - side / 2.0) * spacing + rng.uniform(-0.02, 0.02)
                y = (j - side / 2.0) * spacing + rng.uniform(-0.02, 0.02)
                z = SPHERE_RADIUS + (count // (side * side)) * spacing + rng.uniform(0.0, 0.05)
                b = builder.add_body(
                    xform=wp.transform(wp.vec3(x, y, z)),
                )
                builder.add_shape_sphere(body=b, radius=SPHERE_RADIUS)
                count += 1

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=self.broad_phase,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.solver = newton.solvers.SolverXPBD(self.model)

        self.viewer.set_model(self.model)

        cam_dist = BOX_HALF * 2.0
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.6),
            pitch=-25.0,
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
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        """Verify all spheres stayed inside the box."""
        body_q = self.state_0.body_q.numpy()
        for i in range(self.model.body_count):
            pos = body_q[i, :3]
            assert pos[2] > -0.1, f"Body {i} fell through ground at z={pos[2]:.4f}"
            assert abs(pos[0]) < BOX_HALF + SPHERE_RADIUS, f"Body {i} escaped box at x={pos[0]:.4f}"
            assert abs(pos[1]) < BOX_HALF + SPHERE_RADIUS, f"Body {i} escaped box at y={pos[1]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="nxn")
        parser.add_argument(
            "--num-spheres",
            type=int,
            default=DEFAULT_NUM_SPHERES,
            help="Number of spheres to spawn.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
