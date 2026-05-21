# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convex mesh collision visualization demo.

Shows multiple convex mesh shapes (icospheres) falling from random positions
onto an infinite ground plane. Demonstrates the collision acceleration improvements.

Command: python -m newton.examples convex_mesh_demo
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

SPHERE_RADIUS = 0.12
DEFAULT_NUM_SHAPES = 1000
SPAWN_HEIGHT = 5.0
CUBE_SIZE = 10  # 10x10x10 cube arrangement


def create_icosphere_mesh(radius=0.25, subdivisions=2):
    """Create an icosphere mesh for convex mesh testing."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}

        for face in faces:
            midpoints = []
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                if edge not in edge_midpoints:
                    v1 = np.array(vertices[edge[0]])
                    v2 = np.array(vertices[edge[1]])
                    mid = (v1 + v2) / 2.0
                    mid = mid / np.linalg.norm(mid) * np.sqrt(phi * phi + 1)
                    edge_midpoints[edge] = len(vertices)
                    vertices.append(mid.tolist())
                midpoints.append(edge_midpoints[edge])

            v0, v1, v2 = face
            m0, m1, m2 = midpoints
            new_faces.extend([
                [v0, m0, m2], [v1, m1, m0], [v2, m2, m1], [m0, m1, m2]
            ])
        faces = new_faces

    vertices = np.array(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms * radius

    indices = np.array(faces, dtype=np.int32).flatten()

    return newton.Mesh(vertices.astype(np.float32), indices)


class Example:
    """Convex mesh collision visualization demo."""

    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.num_shapes = args.num_shapes
        self.broad_phase = args.broad_phase

        builder = newton.ModelBuilder()

        # Infinite ground plane for collision
        builder.add_shape_plane()

        # Add a large visible ground box for visualization
        ground_size = 20.0
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.5)),
            hx=ground_size, hy=ground_size, hz=0.5,
        )

        # Create icosphere mesh
        mesh = create_icosphere_mesh(radius=SPHERE_RADIUS, subdivisions=2)
        print(f"Icosphere mesh: {len(mesh.vertices)} vertices per shape")

        # Spawn convex mesh shapes in a cube arrangement above ground
        rng = np.random.default_rng(42)
        spacing = SPHERE_RADIUS * 2.5

        # Calculate cube dimensions to fit num_shapes
        side = int(np.ceil(self.num_shapes ** (1/3)))
        count = 0
        for ix in range(side):
            for iy in range(side):
                for iz in range(side):
                    if count >= self.num_shapes:
                        break
                    # Position in a cube above ground
                    x = (ix - side/2.0) * spacing + rng.uniform(-0.02, 0.02)
                    y = (iy - side/2.0) * spacing + rng.uniform(-0.02, 0.02)
                    z = SPAWN_HEIGHT + iz * spacing + rng.uniform(-0.02, 0.02)

                    # Random orientation
                    quat = wp.quat_from_axis_angle(
                        wp.normalize(wp.vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))),
                        rng.uniform(0, 2 * np.pi)
                    )

                    b = builder.add_body(
                        xform=wp.transform(wp.vec3(x, y, z), quat),
                    )
                    builder.add_shape_convex_hull(
                        body=b,
                        mesh=mesh,
                        scale=wp.vec3(1.0, 1.0, 1.0),
                    )
                    count += 1

        self.model = builder.finalize()

        # Print collision acceleration info
        print(f"Total shapes: {self.num_shapes}")
        if hasattr(self.model, 'vertex_adj_offsets') and self.model.vertex_adj_offsets is not None:
            print(f"Vertex adjacency data: {self.model.vertex_adj_offsets.shape[0]} offsets")
            print(f"Total convex mesh vertices: {self.model.total_convex_mesh_vertices}")
        print(f"Broad phase: {self.broad_phase}")

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

        # Camera setup - view from left side to see falling cube
        self.viewer.set_camera(
            pos=wp.vec3(-15.0, 5.0, 10.0),
            pitch=-25.0,
            yaw=-60.0,
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

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="bvh")
        parser.add_argument(
            "--num-shapes",
            type=int,
            default=DEFAULT_NUM_SHAPES,
            help="Number of convex mesh shapes to spawn.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
