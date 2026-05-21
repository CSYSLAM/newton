# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convex mesh collision benchmark.

Creates a scene with convex mesh shapes (icospheres) and measures collision
detection performance. This validates the collision acceleration improvements:
- Phase 2: AABB cache for mesh_vs_convex midphase (active)
- Phase 3: Hill-climbing for convex mesh support mapping (infrastructure ready)

Command: python -m newton.examples convex_mesh_benchmark
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_SHAPES = 64
DEFAULT_WARMUP = 20
DEFAULT_BENCH_FRAMES = 100


def create_icosphere_mesh(radius=0.3, subdivisions=2):
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

    # Use Newton's Mesh type
    return newton.Mesh(vertices.astype(np.float32), indices)


class Example:
    """Convex mesh collision benchmark."""

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.num_shapes = args.num_shapes
        self.broad_phase = args.broad_phase

        builder = newton.ModelBuilder()
        builder.add_shape_plane()

        # Create icosphere mesh
        mesh = create_icosphere_mesh(radius=0.3, subdivisions=2)
        print(f"Icosphere mesh: {len(mesh.vertices)} vertices")

        # Add convex mesh shapes
        side = int(np.ceil(np.sqrt(self.num_shapes)))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= self.num_shapes:
                    break
                x = (i - side / 2.0) * 0.8
                y = (j - side / 2.0) * 0.8
                z = 0.5 + (i + j) * 0.1
                b = builder.add_body(xform=wp.transform(wp.vec3(x, y, z)))
                builder.add_shape_convex_hull(body=b, mesh=mesh, scale=wp.vec3(1.0, 1.0, 1.0))
                count += 1

        self.model = builder.finalize()

        # Print adjacency data info (Phase 3 infrastructure)
        if hasattr(self.model, 'vertex_adj_offsets') and self.model.vertex_adj_offsets is not None:
            print(f"Vertex adjacency: {self.model.vertex_adj_offsets.shape[0]} offsets, "
                  f"{self.model.vertex_adj_vertices.shape[0]} edges")
            print(f"Total convex mesh vertices: {self.model.total_convex_mesh_vertices}")

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

        cam_dist = side * 0.6
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.5),
            pitch=-20.0,
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
            self.collision_pipeline.collide(self.state_0, self.contacts)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shapes", type=int, default=DEFAULT_NUM_SHAPES)
    parser.add_argument("--broad-phase", type=str, default="bvh", choices=["nxn", "sap", "bvh"])
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--bench-frames", type=int, default=DEFAULT_BENCH_FRAMES)
    parser.add_argument("--headless", action="store_true", help="Run benchmark without viewer")
    args = parser.parse_args()

    wp.init()
    device = wp.get_device()

    print("\n" + "=" * 60)
    print("Convex Mesh Collision Benchmark")
    print("=" * 60)
    print(f"Shapes: {args.num_shapes}")
    print(f"Broad phase: {args.broad_phase}")
    print(f"Device: {device}")

    # Create example without viewer for benchmarking
    builder = newton.ModelBuilder()
    builder.add_shape_plane()

    mesh = create_icosphere_mesh(radius=0.3, subdivisions=2)
    print(f"Icosphere vertices: {len(mesh.vertices)}")

    side = int(np.ceil(np.sqrt(args.num_shapes)))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= args.num_shapes:
                break
            x = (i - side / 2.0) * 0.8
            y = (j - side / 2.0) * 0.8
            z = 0.5 + (i + j) * 0.1
            b = builder.add_body(xform=wp.transform(wp.vec3(x, y, z)))
            builder.add_shape_convex_hull(body=b, mesh=mesh, scale=wp.vec3(1.0, 1.0, 1.0))
            count += 1

    model = builder.finalize()

    # Print adjacency info
    if hasattr(model, 'vertex_adj_offsets') and model.vertex_adj_offsets is not None:
        print(f"Vertex adjacency: {model.vertex_adj_offsets.shape[0]} offsets")
        print(f"Total convex mesh vertices: {model.total_convex_mesh_vertices}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = newton.CollisionPipeline(model, broad_phase=args.broad_phase)
    contacts = collision_pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model)

    sim_dt = 1.0 / 1000.0
    sim_substeps = 10

    # Capture graph
    if device.is_cuda:
        with wp.ScopedCapture() as capture:
            for _ in range(sim_substeps):
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
        graph = capture.graph
    else:
        graph = None

    # Warmup
    print(f"\nWarming up ({args.warmup} frames)...")
    for _ in range(args.warmup):
        if graph:
            wp.capture_launch(graph)
        else:
            for _ in range(sim_substeps):
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

    wp.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.bench_frames} frames)...")
    start_time = time.perf_counter()

    for _ in range(args.bench_frames):
        if graph:
            wp.capture_launch(graph)
        else:
            for _ in range(sim_substeps):
                collision_pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

    wp.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_frame_time = total_time / args.bench_frames * 1000
    fps = args.bench_frames / total_time

    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg frame time: {avg_frame_time:.2f}ms")
    print(f"  FPS: {fps:.1f}")
