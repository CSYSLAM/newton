# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Roll a particle snowball down a snowy slope, collecting fresh snow through contact adhesion.

Run: uv run python -m newton.examples cha_snow
More particles: uv run python -m newton.examples cha_snow --resolution 24
Headless: uv run python -m newton.examples cha_snow --viewer null --num-frames 360 --test
Interaction: right-click and drag the snowball; release to let it roll freely.

The seed and snow bed use the native granular solver. New bonds carry real snow
particles with their original mass; no particle or rigid-body radius is enlarged.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.coupled_high_accuracy.granular import _initial_bonds
from newton.examples.coupled_high_accuracy.snow_interaction import SnowballPicking
from newton.solvers import coupled_high_accuracy as cha

FRAME_DT = 1.0 / 60.0
SNOW_DENSITY = 250.0
SEED_RADIUS = 0.36
BED_HEIGHT = 0.16
SEED_CENTER = np.array([-2.5, 0.0, BED_HEIGHT + SEED_RADIUS + 0.025], dtype=np.float32)


def particle_layout(resolution: int) -> tuple[np.ndarray, float, int]:
    """Put a compact seed above a porous snow bed in slope-local coordinates."""
    if resolution < 6:
        raise ValueError("resolution must be at least 6")
    spacing = 1.28 / resolution
    radius = 0.43 * spacing
    axis = np.arange(-SEED_RADIUS, SEED_RADIUS + 0.5 * spacing, spacing)
    seed = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    seed = seed[np.linalg.norm(seed, axis=1) <= SEED_RADIUS - radius]
    seed -= seed.mean(axis=0)
    x, y = np.arange(-3.1, 3.25, spacing), np.arange(-0.85, 0.86, spacing)
    z = np.arange(radius + 0.002, BED_HEIGHT, spacing)
    bed = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(42)
    bed[:, :2] += rng.uniform(-0.015 * spacing, 0.015 * spacing, (len(bed), 2))
    return np.concatenate((seed + SEED_CENTER, bed)).astype(np.float32), spacing, len(seed)


class Example:
    def __init__(self, viewer, args):
        if args.iterations < 1 or args.linear_iterations < 1 or (args.substeps is not None and args.substeps < 1):
            raise ValueError("iterations, linear_iterations, and substeps must be positive")
        for name in ("sphere_mass", "seed_stiffness"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(args.slope) or not 0.0 <= args.slope <= 30.0:
            raise ValueError("slope must be in [0, 30] degrees")
        if not math.isfinite(args.speed) or args.speed < 0.0:
            raise ValueError("speed must be finite and non-negative")
        local, self.spacing, self.seed_count = particle_layout(args.resolution)
        self.slope_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(args.slope))
        self.rotation = np.asarray(wp.quat_to_matrix(self.slope_rotation)).reshape(3, 3)
        positions = local @ self.rotation.T
        velocities = np.zeros_like(positions)
        offsets = local[: self.seed_count] - SEED_CENTER
        seed_velocity = np.array([args.speed, 0.0, 0.0]) + np.cross([0.0, args.speed / SEED_RADIUS, 0.0], offsets)
        velocities[: self.seed_count] = seed_velocity @ self.rotation.T
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("The native snowball example requires a CUDA device (--device cuda:0).")
        self.viewer, self.sim_time, self.frame_dt = viewer, 0.0, FRAME_DT
        self.sim_substeps = args.substeps if args.substeps is not None else math.ceil(6 * args.resolution / 16)
        self.sim_dt, self.radius = self.frame_dt / self.sim_substeps, 0.43 * self.spacing
        self.granular_count = len(positions)
        self.initial_positions = positions.copy()
        self.initial_seed_center = positions[: self.seed_count].mean(axis=0)
        self.adhesion = args.cohesion
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        masses = np.full(len(positions), SNOW_DENSITY * self.spacing**3, dtype=np.float32)
        masses[: self.seed_count] = args.sphere_mass / self.seed_count
        builder.add_particles(pos=positions, vel=velocities, mass=masses, radius=np.full(len(positions), self.radius))
        cfg = builder.default_shape_cfg.copy()
        cfg.ke, cfg.kd, cfg.kf, cfg.mu = 2.0e5, 60.0, 100.0, 0.8

        def add_box(center, half_size, color):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*(np.asarray(center) @ self.rotation.T)), self.slope_rotation),
                hx=half_size[0],
                hy=half_size[1],
                hz=half_size[2],
                cfg=cfg,
                color=color,
            )

        add_box((0, 0, -0.1), (3.8, 1.15, 0.1), (0.12, 0.18, 0.25))
        for side in (-1.0, 1.0):
            add_box((0, side * 1.19, 0.1), (3.8, 0.04, 0.2), (0.21, 0.29, 0.36))
        add_box((3.84, 0, 0.28), (0.04, 1.23, 0.38), (0.21, 0.29, 0.36))
        add_box((-3.84, 0, 0.08), (0.04, 1.23, 0.18), (0.21, 0.29, 0.36))
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd = cfg.ke, cfg.kd
        self.model.soft_contact_kf, self.model.soft_contact_mu = cfg.kf, cfg.mu
        # Only the compact seed starts bonded. Bed grains gain bonds as actual contact develops.
        bonds, _ = _initial_bonds(
            positions[: self.seed_count],
            np.zeros(self.seed_count, dtype=int),
            np.arange(self.seed_count, dtype=np.int32),
            1.5 * self.spacing,
        )
        cha.configure_granular(
            self.model,
            np.arange(self.granular_count, dtype=np.int32),
            contact_stiffness=20000.0,
            friction=args.friction,
            friction_damping=30.0,
            cohesion=args.seed_stiffness,
            yield_strain=0.08,
            max_compaction=0.16,
            break_strain=0.45,
            bond_indices=bonds,
            adhesion=args.cohesion,
            bond_formation_gap=0.07 * self.spacing,
            bond_formation_speed=3.0,
            max_dynamic_bonds=12,
        )
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.nonlinear_iterations, self.solver.config.linear_iterations = (
            args.iterations,
            args.linear_iterations,
        )
        self.solver.config.enable_collision_hessian_non_diagonal = True
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.granular_positions, self.granular_velocities = self.state_0.particle_q, self.state_0.particle_qd
        colors = np.tile([0.94, 0.97, 1.0], (self.granular_count, 1)).astype(np.float32)
        # The cool seed distinguishes the original snowball from the white snow it collects.
        colors[: self.seed_count] = [0.64, 0.81, 0.96]
        self.colors = wp.array(colors, dtype=wp.vec3, device=device)
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=5 * self.granular_count)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.picking = SnowballPicking(self.model, self.seed_count)
        self.picking.attach(self.viewer)
        self.viewer.show_particles, self.viewer.show_ground = False, False
        self.viewer.set_camera(pos=wp.vec3(5.8, -8.2, 5.5), pitch=-29.0, yaw=124.0)
        self.stream, self.graph = wp.get_stream(device), None
        if not args.no_capture:
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph
        print(
            f"Rolling snowball: {self.seed_count} seed + {self.granular_count - self.seed_count:,} bed particles, "
            f"slope={args.slope:g} degrees, adhesion={args.cohesion:g} N/m, substeps={self.sim_substeps}",
            flush=True,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.picking._apply_picking_force(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points("snow", self.granular_positions, self.radius, self.colors)
        self.viewer.end_frame()

    def snowball_stats(self):
        """Read-only geometric diagnostics; particle positions, mass, and radius are never changed."""
        points = self.granular_positions.numpy()
        center = points[: self.seed_count].mean(axis=0)
        local = points @ self.rotation
        distances = np.linalg.norm(points[self.seed_count :] - center, axis=1)
        carried = (distances < SEED_RADIUS + 3.0 * self.spacing) & (
            local[self.seed_count :, 2] > BED_HEIGHT + self.radius
        )
        travel = float(((center - self.initial_seed_center) @ self.rotation)[0])
        return dict(center=center, carried=int(carried.sum()), travel=travel)

    def test_final(self):
        positions, velocities = self.granular_positions.numpy(), self.granular_velocities.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Snowball produced a non-finite state")
        local = positions @ self.rotation
        if local[:, 2].min() < -0.2 or np.abs(local[:, :2]).max() > 15.0:
            raise ValueError("Snow particles escaped the slope")
        speed = float(np.linalg.norm(velocities, axis=1).max())
        if speed > 25.0:
            raise ValueError(f"Snowball became unstable: maximum speed {speed:.3f} m/s")
        stats = self.snowball_stats()
        if self.sim_time >= 2.0 and stats["travel"] < 0.3:
            raise ValueError("The snowball did not roll down the slope")
        if self.sim_time >= 2.0 and self.adhesion > 0.0 and stats["carried"] < 3:
            raise ValueError("The snowball did not collect fresh snow")
        print(
            f"Snowball checks passed: travel={stats['travel']:.2f} m, lifted snow={stats['carried']}, speed={speed:.3f}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(num_frames=360)
        parser.add_argument("--resolution", type=int, default=16, help="Particles per 1.28 m (>=6).")
        parser.add_argument("--substeps", type=int, default=None, help="Default: ceil(6 * resolution / 16).")
        parser.add_argument("--iterations", type=int, default=6)
        parser.add_argument("--linear-iterations", type=int, default=24)
        parser.add_argument(
            "--cohesion", type=float, default=500.0, help="Fresh contact bond stiffness in N/m; zero disables pickup."
        )
        parser.add_argument(
            "--seed-stiffness", type=float, default=10000.0, help="Initial compact seed bond stiffness in N/m."
        )
        parser.add_argument("--friction", type=float, default=0.2, help="Grain contact friction coefficient.")
        parser.add_argument(
            "--sphere-mass", type=float, default=100.0, help="Total initial particle snowball mass in kg."
        )
        parser.add_argument("--slope", type=float, default=20.0, help="Downhill slope angle in degrees.")
        parser.add_argument("--speed", type=float, default=1.0, help="Initial rolling speed in m/s.")
        parser.add_argument("--no-capture", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    try:
        newton.examples.run(Example(viewer, args), args)
    finally:
        viewer.close()
