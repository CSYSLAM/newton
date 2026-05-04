# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Water pouring into a square cup using MPM.

A square cup made of four static box walls and a ground-plane floor is
filled with viscous fluid particles that pour in from above, demonstrating
viscoplastic material behavior with box colliders.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        SolverImplicitMPM.register_custom_attributes(builder)

        self.build_cup(builder, args)

        Example.emit_particles(builder, args)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.ground_friction))

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        self.model.mpm.viscosity.fill_(args.viscosity)
        self.model.mpm.tensile_yield_ratio.fill_(args.tensile_yield_ratio)
        self.model.mpm.friction.fill_(args.particle_friction)

        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = args.voxel_size
        mpm_options.tolerance = args.tolerance
        mpm_options.max_iterations = args.max_iterations
        mpm_options.strain_basis = args.strain_basis
        mpm_options.collider_basis = args.collider_basis

        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.show_particles = True
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera"):
            cup_w = args.cup_width
            cup_h = args.cup_height
            cup_base = args.cup_base
            self.viewer.set_camera(
                pos=wp.vec3(cup_w * 2.0, -cup_w * 2.0, cup_base + cup_h + cup_w * 2.0),
                pitch=-50.0,
                yaw=155.0,
            )

    @staticmethod
    def build_cup(builder, args):
        half_w = args.cup_width / 2.0
        wall_t = args.wall_thickness
        cup_h = args.cup_height
        cup_base = args.cup_base
        mu = args.wall_friction
        cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0)

        wall_lo_z = cup_base - wall_t
        wall_hi_z = cup_base + cup_h
        wall_center_z = (wall_lo_z + wall_hi_z) / 2.0
        wall_hz = (wall_hi_z - wall_lo_z) / 2.0

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(half_w + wall_t / 2.0, 0.0, wall_center_z), wp.quat_identity()),
            hx=wall_t / 2.0,
            hy=half_w + wall_t,
            hz=wall_hz,
            cfg=cfg,
            color=wp.vec3(0.8, 0.85, 0.9),
            label="wall_x+",
        )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-(half_w + wall_t / 2.0), 0.0, wall_center_z), wp.quat_identity()),
            hx=wall_t / 2.0,
            hy=half_w + wall_t,
            hz=wall_hz,
            cfg=cfg,
            color=wp.vec3(0.8, 0.85, 0.9),
            label="wall_x-",
        )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, half_w + wall_t / 2.0, wall_center_z), wp.quat_identity()),
            hx=half_w + wall_t,
            hy=wall_t / 2.0,
            hz=wall_hz,
            cfg=cfg,
            color=wp.vec3(0.8, 0.85, 0.9),
            label="wall_y+",
        )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, -(half_w + wall_t / 2.0), wall_center_z), wp.quat_identity()),
            hx=half_w + wall_t,
            hy=wall_t / 2.0,
            hz=wall_hz,
            cfg=cfg,
            color=wp.vec3(0.8, 0.85, 0.9),
            label="wall_y-",
        )

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, cup_base), wp.quat_identity()),
            hx=half_w + wall_t,
            hy=half_w + wall_t,
            hz=wall_t / 2.0,
            cfg=cfg,
            color=wp.vec3(0.75, 0.8, 0.85),
            label="cup_bottom",
        )

    @staticmethod
    def emit_particles(builder, args):
        voxel_size = args.voxel_size
        density = args.density
        particles_per_cell = 3.0

        half_w = args.cup_width / 2.0
        margin = voxel_size

        cup_top_z = args.cup_base + args.cup_height
        emit_lo = np.array([-half_w + margin, -half_w + margin, cup_top_z + 0.02])
        emit_hi = np.array([half_w - margin, half_w - margin, cup_top_z + args.pour_height])

        particle_res = np.array(
            np.ceil(particles_per_cell * (emit_hi - emit_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (emit_hi - emit_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = np.max(cell_size) * 0.5
        mass = cell_volume * density

        dim_x = particle_res[0] + 1
        dim_y = particle_res[1] + 1
        dim_z = particle_res[2] + 1

        px = np.arange(dim_x) * cell_size[0]
        py = np.arange(dim_y) * cell_size[1]
        pz = np.arange(dim_z) * cell_size[2]
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        jitter = 2.0 * np.max(cell_size)
        rng = np.random.default_rng(42)
        points += (rng.random(points.shape) - 0.5) * jitter

        points += emit_lo

        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[mass] * points.shape[0],
            radius=[radius] * points.shape[0],
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        voxel_size = self.solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )
        positions = self.state_0.particle_q.numpy()
        min_z = np.min(positions[:, 2])
        if min_z > 0.5:
            raise ValueError("Particles did not settle into the cup")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        parser.add_argument("--cup-width", type=float, default=0.15, help="Inner width of the square cup [m]")
        parser.add_argument("--cup-height", type=float, default=0.15, help="Wall height of the cup [m]")
        parser.add_argument("--cup-base", type=float, default=0.2, help="Height of cup bottom above ground [m]")
        parser.add_argument("--wall-thickness", type=float, default=0.01, help="Wall thickness [m]")
        parser.add_argument("--pour-height", type=float, default=0.08, help="Height of water column above cup [m]")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
        parser.add_argument("--fps", type=float, default=240.0)
        parser.add_argument("--substeps", type=int, default=2)

        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--viscosity", type=float, default=5.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=1.0)
        parser.add_argument("--particle-friction", "-mu", type=float, default=0.0)
        parser.add_argument("--wall-friction", type=float, default=0.0)
        parser.add_argument("--ground-friction", type=float, default=0.5)

        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.015)
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="S2")

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
