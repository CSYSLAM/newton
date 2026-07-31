# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MPM Lite plastic wheel compressed between moving plates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
from newton.examples.mpm.mpm_lite_common import add_particles, boundary_band_nodes
from newton.solvers import SolverMPMLite


class Example:
    """Compress a plastic toroidal wheel using MPM Lite."""

    def __init__(self, viewer, options):
        self.viewer = viewer
        self.sim_dt = options.dt
        self.sim_substeps = options.substeps
        self.sim_time = 0.0
        self._initial_render = True
        self.press_start = 0.51
        self.press_stop = 0.21
        self.press_speed = 0.5
        self.floor_height = 0.03
        self.domain_extent = np.asarray(options.grid_size, dtype=np.float32) * options.voxel_size

        builder = newton.ModelBuilder()
        SolverMPMLite.register_custom_attributes(builder)
        self._add_wheel(builder, options)
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, options.gravity))
        self.solver = SolverMPMLite(
            self.model,
            SolverMPMLite.Config(
                grid_size=tuple(options.grid_size),
                voxel_size=options.voxel_size,
                solver_type="lite_implicit",
                max_iterations=options.max_iterations,
                density=options.density,
                young_modulus=options.young_modulus,
                poisson_ratio=options.poisson_ratio,
                yield_stress=options.yield_stress,
            ),
        )
        boundary = boundary_band_nodes(tuple(options.grid_size), width=5)
        self.solver.paint_boundary(boundary, np.ones(len(boundary), dtype=np.int32))

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(1.40, -1.40, 0.90), pitch=-16.0, yaw=110.0)

    def _add_wheel(self, builder, options) -> None:
        mesh = trimesh.load(Path(__file__).with_name("assets") / "mpm_lite" / "wheel.obj", force="mesh")
        voxels = mesh.voxelized(pitch=options.voxel_size).fill().points
        volume = len(voxels) * options.voxel_size**3
        particle_count = int(options.particles_per_cell * volume / options.voxel_size**3)
        rng = np.random.RandomState(0)
        points = voxels[rng.choice(len(voxels), particle_count, replace=True)]
        points += rng.uniform(-0.5 * options.voxel_size, 0.5 * options.voxel_size, points.shape)
        add_particles(
            builder,
            np.asarray(points, dtype=np.float32) + np.asarray((0.5, 0.5, 0.02), dtype=np.float32),
            particle_volume=volume / particle_count,
            density=options.density,
            particle_radius=0.25 * options.voxel_size,
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            press_z = max(self.press_stop, self.press_start - self.press_speed * self.sim_time)
            self.solver.paint_halfspace_boundary(
                np.array([[0.0, 0.0, press_z], [0.0, 0.0, self.floor_height]], dtype=np.float32),
                np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                np.array(
                    [[0.0, 0.0, -self.press_speed if press_z > self.press_stop else 0.0], [0.0, 0.0, 0.0]],
                    dtype=np.float32,
                ),
                np.array([1, 1], dtype=np.int32),
            )
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self) -> None:
        if self._initial_render:
            self._initial_render = False
            return
        self.simulate()

    def test_final(self) -> None:
        """Verify that wheel particles remain inside the MPM grid."""
        positions = self.state_0.particle_q.numpy()
        if not np.isfinite(positions).all() or np.any(positions < 0.0) or np.any(positions > self.domain_extent):
            raise ValueError("MPM Lite wheel particles left the simulation grid.")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--substeps", type=int, default=1)
        parser.add_argument("--grid-size", type=int, nargs=3, default=(355, 355, 355))
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.005)
        parser.add_argument("--dt", type=float, default=1.0e-3)
        parser.add_argument("--max-iterations", type=int, default=10)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--young-modulus", type=float, default=1.0e8)
        parser.add_argument("--poisson-ratio", type=float, default=0.3)
        parser.add_argument("--yield-stress", type=float, default=1.9e5)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--particles-per-cell", type=float, default=24.0)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
