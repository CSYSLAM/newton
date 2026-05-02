# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM Elastic Cube Drop
#
# This simulation demonstrates an elastic cube falling from a height
# and colliding with the ground using the implicit MPM solver.
# The cube exhibits realistic elastic deformation and bouncing behavior.
#
# MPM (Material Point Method) is particularly well-suited for very stiff
# materials and offers unconditional stability with respect to the time step.
#
# Command: uv run -m newton.examples mpm_elastic_cube_drop
#
###########################################################################

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    """Elastic cube drop using the implicit MPM solver."""

    def __init__(self, viewer, args):
        # Setup simulation parameters
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        builder = newton.ModelBuilder()

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # Add ground plane
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.friction))

        # Emit elastic cube particles
        self.emit_cube_particles(builder, args)

        self.model = builder.finalize()
        self.model.set_gravity(args.gravity)

        # Copy CLI arguments to MPM options
        mpm_options = SolverImplicitMPM.Config()
        for key in vars(args):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(args, key))

        # Create states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize material properties
        self.init_materials(args)

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, mpm_options)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        # Position camera to see the cube at initial height
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(5.0, 5.0, 5.0), pitch=-30.0, yaw=-160.0)

        self.capture()

    def emit_cube_particles(self, builder: newton.ModelBuilder, args):
        """Emit a cube of particles."""
        density = args.density
        voxel_size = args.voxel_size

        # Cube dimensions
        cube_size = args.cube_size
        cube_center = np.array(args.cube_center)

        # Particles per cell dimension
        particles_per_cell_dim = 2
        spacing = voxel_size / particles_per_cell_dim

        # Number of particles along each dimension
        num_particles_dim = int(np.ceil(cube_size / spacing))

        # Actual cell size to fit exactly
        cell_size = cube_size / num_particles_dim
        cell_volume = cell_size ** 3

        # Particle radius and mass
        radius = cell_size * 0.5
        mass = cell_volume * density

        # Create particle grid centered at cube_center
        particle_lo = cube_center - cube_size * 0.5
        particle_hi = cube_center + cube_size * 0.5

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=num_particles_dim + 1,
            dim_y=num_particles_dim + 1,
            dim_z=num_particles_dim + 1,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            mass=mass,
            jitter=0.5 * radius,
            radius_mean=radius,
        )

        print(f"Created {num_particles_dim + 1}^3 = {(num_particles_dim + 1)**3} particles")

    def init_materials(self, args):
        """Initialize per-particle material properties for elastic behavior."""
        model = self.model

        # Elastic material properties
        # High Young's modulus for stiff elastic material
        model.mpm.young_modulus.fill_(args.young_modulus)
        model.mpm.poisson_ratio.fill_(args.poisson_ratio)
        model.mpm.damping.fill_(args.damping)

        # Friction for ground contact
        model.mpm.friction.fill_(args.friction)

        # Set high yield pressure to prevent plastic deformation (compression)
        model.mpm.yield_pressure.fill_(args.yield_pressure)

        # CRITICAL: Set tensile_yield_ratio to 1.0 to prevent tensile fracture
        # This keeps particles together as a cohesive solid
        model.mpm.tensile_yield_ratio.fill_(args.tensile_yield_ratio)

        # No additional yield stress
        model.mpm.yield_stress.fill_(args.yield_stress)

        # No hardening or dilatancy for pure elastic behavior
        model.mpm.hardening.fill_(args.hardening)
        model.mpm.dilatancy.fill_(args.dilatancy)

        # Initialize plastic deformation gradient determinant (Jp = 1 means no plastic strain)
        self.state_0.mpm.particle_Jp.fill_(1.0)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                warnings.warn("Sim substeps must be even for graph capture of MPM step", stacklevel=2)
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
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

    def test_final(self):
        voxel_size = self.solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

        # Test that particles are within reasonable bounds after simulation
        p_lower = wp.vec3(-2.0, -2.0, -voxel_size)
        p_upper = wp.vec3(2.0, 2.0, 5.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume after simulation",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration
        parser.add_argument("--cube-size", type=float, default=0.5, help="Size of the cube in meters")
        parser.add_argument("--cube-center", type=float, nargs=3, default=[0.0, 0.0, 2.0], help="Center position of the cube")
        parser.add_argument("--gravity", type=float, nargs=3, default=[0, 0, -9.81])
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2)

        # Material properties (elastic)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--young-modulus", "-ym", type=float, default=5.0e6, help="Young's modulus in Pa (stiffness)")
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.45, help="Poisson's ratio (0.45 for nearly incompressible)")
        parser.add_argument("--friction", "-mu", type=float, default=0.5)
        parser.add_argument("--damping", type=float, default=0.001, help="Damping for energy dissipation")

        # Yield parameters (set high for elastic behavior)
        parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e15, help="High value prevents plastic compression")
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=1.0, help="Set to 1.0 to prevent tensile fracture, keeps particles cohesive")
        parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
        parser.add_argument("--hardening", type=float, default=0.0)
        parser.add_argument("--dilatancy", type=float, default=0.0)

        # Solver parameters
        parser.add_argument(
            "--solver",
            "-s",
            type=str,
            default="gauss-seidel",
            choices=["gauss-seidel", "jacobi", "cg", "cg+jacobi", "cg+gauss-seidel"],
        )
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--max-iterations", "-it", type=int, default=250)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.05)
        parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
        parser.add_argument("--transfer-scheme", "-ts", type=str, default="apic", choices=["apic", "pic"])
        parser.add_argument("--integration-scheme", "-is", type=str, default="pic", choices=["pic", "gimp"])

        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)