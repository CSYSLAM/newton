# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Suspend a thin volumetric MPM sheet between two fixed edges.

The sheet is exactly two particles thick. Two particle rows at each end are
kinematically fixed while gravity pulls the free middle region downward into a
hammock-like shape.

Run from the repository root::

    uv run --extra examples -m newton.examples mpm_suspended_sheet
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

SHEET_COLOR = wp.vec3(0.12, 0.42, 0.85)
ANCHOR_COLOR = wp.vec3(1.0, 0.48, 0.08)


class Example:
    """Thin volumetric MPM sheet sagging between two fixed edges."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = float(args.fps)
        self.sim_substeps = int(args.substeps)
        if self.fps <= 0.0 or self.sim_substeps <= 0:
            raise ValueError("FPS and substeps must be positive")
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sheet_length = float(args.sheet_length)

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        particle_data = self._emit_sheet(builder, args)

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity(args.gravity)

        self.fixed_indices_np = particle_data["fixed_indices"]
        self.center_indices_np = particle_data["center_indices"]
        self.fixed_indices = wp.array(self.fixed_indices_np, dtype=wp.int32, device=self.model.device)
        self.model.particle_mass[self.fixed_indices].fill_(0.0)

        material = self.model.mpm
        material.young_modulus.fill_(float(args.young_modulus))
        material.poisson_ratio.fill_(float(args.poisson_ratio))
        material.damping.fill_(float(args.damping))
        material.yield_pressure.fill_(1.0e15)
        material.tensile_yield_ratio.fill_(1.0)
        material.yield_stress.fill_(0.0)
        material.hardening.fill_(0.0)
        material.dilatancy.fill_(0.0)
        material.viscosity.fill_(0.0)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.mpm.particle_Jp.fill_(1.0)
        self.state_1.mpm.particle_Jp.fill_(1.0)

        initial_positions = self.state_0.particle_q.numpy()
        self.fixed_positions_initial = initial_positions[self.fixed_indices_np].copy()
        self.center_height_initial = float(np.mean(initial_positions[self.center_indices_np, 2]))

        config = SolverImplicitMPM.Config()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.warmstart_mode = "particles"
        self.solver = SolverImplicitMPM(self.model, config=config)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=SHEET_COLOR,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.particle_colors[self.fixed_indices].fill_(ANCHOR_COLOR)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.85, -1.00, 1.36), pitch=-7.5, yaw=122.0)

        print(
            f"[newton] MPM suspended sheet: particles={self.model.particle_count}, "
            f"fixed={self.fixed_indices.shape[0]}, thickness_layers=2, "
            f"thickness={float(args.sheet_thickness):.3f} m"
        )

    @staticmethod
    def _emit_sheet(builder: newton.ModelBuilder, args) -> dict[str, np.ndarray]:
        width = float(args.sheet_width)
        length = float(args.sheet_length)
        thickness = float(args.sheet_thickness)
        voxel_size = float(args.voxel_size)
        particles_per_cell = int(args.particles_per_cell)
        anchor_rows = int(args.anchor_rows)
        density = float(args.density)
        anchor_height = float(args.anchor_height)
        initial_sag = float(args.initial_sag)
        initial_ripple = float(args.initial_ripple)

        if min(width, length, thickness, voxel_size, density) <= 0.0 or particles_per_cell <= 0:
            raise ValueError("sheet dimensions, voxel size, density, and particles per cell must be positive")
        if anchor_rows <= 0:
            raise ValueError("anchor rows must be positive")
        if initial_sag < 0.0 or initial_ripple < 0.0:
            raise ValueError("initial sag and ripple must be non-negative")

        target_spacing = voxel_size / particles_per_cell
        resolution = np.array(
            [
                max(int(np.ceil(width / target_spacing)), 1),
                max(int(np.ceil(length / target_spacing)), 1),
                1,
            ],
            dtype=np.int32,
        )
        if 2 * anchor_rows >= resolution[1] + 1:
            raise ValueError("anchor rows must leave free particle rows between the fixed ends")
        spacing = np.array([width, length, thickness]) / resolution

        x = np.linspace(-0.5 * width, 0.5 * width, int(resolution[0]) + 1)
        y = np.linspace(0.0, length, int(resolution[1]) + 1)
        z = np.linspace(-0.5 * thickness, 0.5 * thickness, int(resolution[2]) + 1)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

        normalized_y = grid_y / length
        center_envelope = 4.0 * normalized_y * (1.0 - normalized_y)
        ripple_phase = 3.0 * np.pi * (grid_x / width + 0.5)
        ripple = initial_ripple * np.sin(ripple_phase) * center_envelope
        positions = np.column_stack(
            (
                grid_x.ravel(),
                grid_y.ravel(),
                (anchor_height + grid_z - initial_sag * center_envelope + ripple).ravel(),
            )
        )

        cell_volume = float(np.prod(spacing))
        particle_radius = 0.55 * float(np.min(spacing))
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=[cell_volume * density] * len(positions),
            radius=[particle_radius] * len(positions),
        )

        anchor_width = (anchor_rows - 0.5) * spacing[1]
        fixed_indices = np.flatnonzero(
            (positions[:, 1] <= anchor_width) | (positions[:, 1] >= length - anchor_width)
        ).astype(np.int32)
        center_indices = np.flatnonzero(np.abs(positions[:, 1] - 0.5 * length) <= 0.51 * spacing[1]).astype(np.int32)
        if len(center_indices) == 0:
            raise ValueError("the center of the sheet contains no particles")
        return {
            "fixed_indices": fixed_indices,
            "center_indices": center_indices,
        }

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self.viewer.show_particles = False
        self.viewer.log_state(self.state_0)
        self.viewer.show_particles = show_particles
        self.viewer.log_points(
            name="/suspended_sheet",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not show_particles,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Verify the suspended sheet remains finite after every frame."""
        positions = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(positions)):
            raise ValueError("suspended-sheet particle positions are not finite")

    def test_final(self) -> None:
        """Verify both fixed edges stay fixed and the center sags."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("suspended-sheet state is not finite")
        if np.ptp(self.fixed_positions_initial[:, 1]) < 0.9 * self.sheet_length:
            raise ValueError("fixed particles do not span both ends of the sheet")
        if np.max(np.abs(positions[self.fixed_indices_np] - self.fixed_positions_initial)) > 1.0e-5:
            raise ValueError("kinematic edge particles moved")
        if np.linalg.norm(np.ptp(positions, axis=0)) > 5.0:
            raise ValueError("suspended-sheet particles became unbounded")
        if self.sim_time >= 1.0:
            center_height = float(np.mean(positions[self.center_indices_np, 2]))
            center_drop = self.center_height_initial - center_height
            if center_drop < 0.06:
                raise ValueError(
                    "the sheet center did not fall into a suspended configuration "
                    f"(drop={center_drop:.3f} m, expected at least 0.060 m)"
                )

    @staticmethod
    def create_parser():
        """Create command-line arguments for the suspended-sheet demo."""
        parser = newton.examples.create_parser()
        parser.description = "Suspend a thin volumetric MPM sheet between two fixed edges."
        parser.set_defaults(num_frames=360)
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, -9.81))
        parser.add_argument("--sheet-width", type=float, default=0.70, help="Sheet width [m].")
        parser.add_argument("--sheet-length", type=float, default=0.75, help="Distance between fixed ends [m].")
        parser.add_argument("--sheet-thickness", type=float, default=0.01, help="Two-layer sheet thickness [m].")
        parser.add_argument("--anchor-height", type=float, default=1.20, help="Fixed-edge height [m].")
        parser.add_argument("--anchor-rows", type=int, default=2, help="Particle rows fixed at each end.")
        parser.add_argument("--initial-sag", type=float, default=0.005, help="Initial center sag [m].")
        parser.add_argument("--initial-ripple", type=float, default=0.004, help="Initial cross-sheet ripple [m].")
        parser.add_argument("--particles-per-cell", type=int, default=2)
        parser.add_argument("--density", type=float, default=100.0, help="Effective sheet density [kg/m³].")
        parser.add_argument("--young-modulus", "-ym", type=float, default=5.0e4, help="Young's modulus [Pa].")
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.30)
        parser.add_argument("--damping", type=float, default=0.05, help="Elastic damping relaxation time [s].")
        parser.add_argument("--air-drag", type=float, default=1.5, help="Background numerical drag.")
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.02)
        parser.add_argument("--grid-type", choices=("sparse", "dense", "fixed"), default="sparse")
        parser.add_argument("--strain-basis", choices=("P0", "P1d", "Q1", "Q1d"), default="P1d")
        parser.add_argument("--max-iterations", "-it", type=int, default=150)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        return parser


def main():
    """Run the MPM suspended-sheet demo."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
