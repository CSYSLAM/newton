# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Broad phase collision detection comparison example.

Drops a grid of spheres onto a ground plane using the selected broad
phase method (NxN, SAP, or BVH).  BVH uses Warp's GPU-accelerated
Linear BVH construction for O(N log N) performance.

Command: python -m newton.examples broad_phase_comparison
With BVH: python -m newton.examples broad_phase_comparison --broad-phase bvh
"""

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_SPHERES = 400
CUBE_HALF = 0.5


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
        builder.add_shape_plane()

        # Build a grid of spheres above the ground plane
        side = int(np.ceil(np.sqrt(self.num_spheres)))
        spacing = 1.2 * CUBE_HALF * 2
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx >= self.num_spheres:
                    break
                x = (i - side / 2.0) * spacing
                y = (j - side / 2.0) * spacing
                z = 2.0 + (i % 3) * 0.5
                b = builder.add_body(
                    xform=wp.transform(wp.vec3(x, y, z)),
                )
                builder.add_shape_sphere(body=b, radius=CUBE_HALF)

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

        cam_dist = side * spacing * 0.6
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.4),
            pitch=-15.0,
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
        """Verify all spheres settled above the ground plane."""
        body_q = self.state_0.body_q.numpy()
        for i in range(self.model.body_count):
            assert body_q[i, 2] > -0.1, f"Body {i} fell through ground at z={body_q[i, 2]:.4f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="bvh")
        parser.add_argument(
            "--num-spheres",
            type=int,
            default=DEFAULT_NUM_SPHERES,
            help="Number of spheres to simulate.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)