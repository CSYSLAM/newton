# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Water pouring into a round cup using MPM.

A cup mesh loaded from USD is filled with viscous fluid particles
that pour in from above, demonstrating viscoplastic material behavior
with mesh colliders.
"""

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
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
            cup_scale = args.cup_scale
            cup_base = args.cup_base
            cup_h = 2.0 * cup_scale
            self.viewer.set_camera(
                pos=wp.vec3(cup_scale * 4.0, -cup_scale * 4.0, cup_base + cup_h + cup_scale * 4.0),
                pitch=-50.0,
                yaw=155.0,
            )

    @staticmethod
    def build_cup(builder, args):
        cup_base = args.cup_base
        mu = args.wall_friction
        cup_scale = args.cup_scale

        usd_stage = Usd.Stage.Open("newton/_src/usd/cup_new.usdc")
        cup_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/柱体"))

        cup_center_z = cup_base + cup_scale
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, cup_center_z), wp.quat_identity()),
            mesh=cup_mesh,
            scale=(cup_scale, cup_scale, cup_scale),
            cfg=newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0),
            color=wp.vec3(0.7, 0.85, 0.95),
        )

    @staticmethod
    def emit_particles(builder, args):
        voxel_size = args.voxel_size
        density = args.density
        particles_per_cell = 3.0

        cup_scale = args.cup_scale
        inner_r = cup_scale * 0.9
        margin = voxel_size

        cup_base = args.cup_base
        cup_top_z = cup_base + 2.0 * cup_scale
        emit_r = inner_r - margin
        emit_lo_z = cup_top_z + 0.02
        emit_hi_z = cup_top_z + args.pour_height

        n_r = int(np.ceil(particles_per_cell * emit_r / voxel_size))
        n_theta = int(np.ceil(particles_per_cell * 2.0 * np.pi * emit_r / voxel_size))
        n_z = int(np.ceil(particles_per_cell * (emit_hi_z - emit_lo_z) / voxel_size))

        r_vals = np.linspace(0, emit_r, n_r + 1)
        z_vals = np.linspace(emit_lo_z, emit_hi_z, n_z + 1)
        theta_vals = np.linspace(0, 2.0 * np.pi, n_theta, endpoint=False)

        points = []
        for r in r_vals:
            for t in theta_vals:
                for z in z_vals:
                    x = r * np.cos(t)
                    y = r * np.sin(t)
                    if x * x + y * y <= emit_r * emit_r:
                        points.append([x, y, z])

        if not points:
            return

        points = np.array(points)
        cell_volume = (emit_r / max(n_r, 1)) * (2.0 * np.pi * emit_r / max(n_theta, 1)) * ((emit_hi_z - emit_lo_z) / max(n_z, 1))
        radius = voxel_size / particles_per_cell * 0.5
        mass = cell_volume * density

        jitter = radius
        rng = np.random.default_rng(42)
        points += (rng.random(points.shape) - 0.5) * jitter

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

        parser.add_argument("--cup-scale", type=float, default=0.05, help="Scale factor for the USD cup mesh")
        parser.add_argument("--cup-base", type=float, default=0.25, help="Height of cup bottom above ground [m]")
        parser.add_argument("--pour-height", type=float, default=0.06, help="Height of water column above cup [m]")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -10])
        parser.add_argument("--fps", type=float, default=240.0)
        parser.add_argument("--substeps", type=int, default=2)

        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--viscosity", type=float, default=1.0)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=1.0)
        parser.add_argument("--particle-friction", "-mu", type=float, default=0.0)
        parser.add_argument("--wall-friction", type=float, default=0.0)
        parser.add_argument("--ground-friction", type=float, default=0.5)

        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.01)
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--collider-basis", "-cb", type=str, default="S2")

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
