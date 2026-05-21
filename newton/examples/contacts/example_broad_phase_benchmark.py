# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Broad phase benchmark: NxN vs SAP vs BVH on a dense sphere pile.

Creates a large pile of overlapping rigid-body spheres and measures
per-frame FPS for each broad phase method.

BVH uses Warp's GPU-accelerated Linear BVH with O(N log N) tree
construction and O(N) refit.  Without CUDA graph capture, BVH and
NxN perform similarly; with CUDA graphs, NxN benefits more from
kernel-launch amortization because all its operations are pure
``wp.launch()`` calls, while BVH's ``refit()`` and ``wp.copy()``
go through a Python-to-C bridge that has higher overhead during
graph replay.

Command: python -m newton.examples broad_phase_benchmark
Compare all: python -m newton.examples broad_phase_benchmark --compare-all
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
import newton.examples

SPHERE_RADIUS = 0.3
SPHERE_SPACING = 0.62
DEFAULT_NUM_SPHERES = 500
DEFAULT_WARMUP = 20
DEFAULT_BENCH_FRAMES = 200


class Example:
    """Dense sphere pile benchmark comparing broad phase methods."""

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

        # Dense sphere pile: grid with slight vertical offset per row
        # so many spheres overlap -> stress broad phase
        side = int(np.ceil(np.sqrt(self.num_spheres)))
        count = 0
        for layer in range(5):
            for i in range(side):
                for j in range(side):
                    if count >= self.num_spheres:
                        break
                    x = (i - side / 2.0) * SPHERE_SPACING
                    y = (j - side / 2.0) * SPHERE_SPACING
                    z = SPHERE_RADIUS + layer * SPHERE_SPACING * 0.9
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

        cam_dist = side * SPHERE_SPACING * 0.8
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
        parser.add_argument("--num-spheres", type=int, default=DEFAULT_NUM_SPHERES)
        return parser


def _run_benchmark(broad_phase: str, num_spheres: int, warmup: int, bench_frames: int):
    """Run one broad phase mode and return (avg_ms, min_ms, max_ms, fps)."""
    parser = Example.create_parser()
    args = newton.examples.default_args(parser)
    args.broad_phase = broad_phase
    args.num_spheres = num_spheres
    args.viewer = "null"
    args.test = False

    viewer = newton.viewer.ViewerNull(num_frames=warmup + bench_frames)
    ex = Example(viewer, args)

    # Warmup
    for _ in range(warmup):
        ex.step()

    device = ex.model.device
    wp.synchronize_device(device)

    times = []
    for _ in range(bench_frames):
        t0 = time.perf_counter()
        ex.step()
        wp.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))
    fps = 1000.0 / avg_ms

    viewer.close()
    return avg_ms, min_ms, max_ms, fps


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--bench-frames", type=int, default=DEFAULT_BENCH_FRAMES)
    parser.add_argument("--compare-all", action="store_true", default=False)
    viewer, args = newton.examples.init(parser)

    if args.compare_all:
        # Benchmark all three modes and print comparison
        num = args.num_spheres
        warmup = args.warmup
        frames = args.bench_frames
        print(f"=== Broad Phase Benchmark ===")
        print(f"Spheres: {num}  Warmup: {warmup}  Frames: {frames}")
        print()

        results = {}
        for mode in ("nxn", "sap", "bvh"):
            print(f"  {mode} ...", end=" ", flush=True)
            avg, lo, hi, fps = _run_benchmark(mode, num, warmup, frames)
            results[mode] = (avg, lo, hi, fps)
            print(f"{fps:.1f} FPS  (avg {avg:.2f}ms)")

        print()
        print("--- Comparison ---")
        nxn_avg = results["nxn"][0]
        for mode, (avg, lo, hi, fps) in results.items():
            ratio = nxn_avg / avg
            print(f"  {mode:8s}: {fps:7.1f} FPS  avg={avg:7.2f}ms  ({ratio:.2f}x vs NxN)")

        viewer.close()
    else:
        # Normal mode: run single broad phase with viewer
        example = Example(viewer, args)
        newton.examples.run(example, args)