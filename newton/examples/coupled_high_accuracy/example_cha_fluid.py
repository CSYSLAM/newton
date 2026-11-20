# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Native particle-fluid dam break with five freely floating textured rubber ducks.

Run: uv run python -m newton.examples cha_fluid
Surface: uv run python -m newton.examples cha_fluid --build-surface
Large: uv run python -m newton.examples cha_fluid --resolution 48
Headless: uv run python -m newton.examples cha_fluid --viewer null --num-frames 180 --test
Interaction: right-click and drag any duck; release to let it float freely.

Density pressure, viscosity, and duck/wall contacts share the native Newton/PCG solve.
See docs/particle_fluid.md for material parameters and static boundary quadrature.
"""

import copy
import itertools
import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
from newton.solvers import coupled_high_accuracy as cha

TANK_LOWER = (0.0, -0.56, 0.0)
TANK_UPPER = (3.36, 0.56, 2.24)
REST_DENSITY = 1000.0
FRAME_DT = 1.0 / 60.0
DUCK_ASSET = "models/rubber_duck/rubber_duck.usd"
DUCK_SCALE = 4.8
DUCK_DENSITY = 300.0
DUCK_POSITIONS = ((0.22, -0.21), (0.56, -0.21), (0.90, -0.21), (0.39, 0.21), (0.73, 0.21))


@wp.kernel
def diagnostic_density(
    grid: wp.uint64,
    samples: wp.array[wp.vec3],
    volumes: wp.array[float],
    h: float,
    ratios: wp.array[float],
):
    # Read-only validation; all simulation forces and position updates are native.
    i = wp.tid()
    value = float(0.0)
    for j in wp.hash_grid_query(grid, samples[i], h):
        r = samples[i] - samples[j]
        s = wp.max(1.0 - wp.dot(r, r) / (h * h), 0.0)
        value += volumes[j] * (315.0 / (64.0 * wp.pi)) * s * s * s / (h * h * h)
    ratios[i] = value


@wp.kernel
def color_particles(velocities: wp.array[wp.vec3], colors: wp.array[wp.vec3]):
    i = wp.tid()
    speed = wp.clamp(wp.length(velocities[i]) / 4.0, 0.0, 1.0)
    colors[i] = wp.lerp(wp.vec3(0.025, 0.20, 0.65), wp.vec3(0.30, 0.90, 1.0), speed)


def particle_layout(resolution: int) -> tuple[np.ndarray, float]:
    """Change particle count while keeping the initial water volume approximately constant."""
    if resolution < 8:
        raise ValueError("resolution must be at least 8")
    spacing = 1.12 / resolution
    nx, ny, nz = resolution, 3 * resolution // 4, 3 * resolution // 2
    cells = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1)
    positions = (cells.reshape(-1, 3).astype(np.float32) + 0.5) * spacing
    positions[:, 1] -= 0.5 * ny * spacing
    # Break exact column symmetry without changing neighbor counts or overlapping the walls.
    rng = np.random.default_rng(42)
    positions += rng.uniform(-0.01 * spacing, 0.01 * spacing, positions.shape).astype(np.float32)
    return positions, spacing


def rest_volume(spacing):
    """Calibrate particle mass against the discrete Poly6 lattice density."""
    h = 2.0 * spacing
    offsets = np.asarray(list(itertools.product(range(-2, 3), repeat=3))) * spacing
    weights = 315.0 / (64.0 * math.pi) * np.maximum(1.0 - np.sum(offsets**2, axis=1) / h**2, 0.0) ** 3
    return h**3 / weights.sum()


def boundary_samples(spacing):
    """Fill two layers outside the tank without duplicating samples at edges or corners."""
    lower, upper = np.asarray(TANK_LOWER), np.asarray(TANK_UPPER)
    counts = np.rint((upper - lower) / spacing).astype(int)
    axes = [(np.arange(-2, n + 2) + 0.5) * spacing + low for n, low in zip(counts, lower, strict=True)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    outside = np.any((points < lower) | (points > upper), axis=1)
    return points[outside].astype(np.float32)


def add_ducks(builder, water_height):
    """Resolve the textured ducks with several fluid particles across their submerged hulls."""
    duck_asset = cha.get_asset(DUCK_ASSET)
    visual = newton.usd.get_mesh(
        duck_asset, root_path="/World/simple_xhy/Body/visual", load_normals=True, load_uvs=True, compute_inertia=False
    )
    source = newton.usd.get_mesh(duck_asset, root_path="/World/simple_xhy/Body/collision", compute_inertia=False)
    source.maxhullvert = 0
    hull = source.compute_convex_hull()
    collision = newton.Mesh(hull.vertices, hull.indices, maxhullvert=0)
    center = 0.5 * (collision.vertices.min(axis=0) + collision.vertices.max(axis=0))
    local_vertices = DUCK_SCALE * (collision.vertices - center)
    mesh_xform = wp.transform(wp.vec3(*(-DUCK_SCALE * center)), wp.quat_identity())
    collision_cfg = builder.default_shape_cfg.copy()
    # A light shell and a small bottom weight model a self-righting floating toy.
    # Both masses are physical: contact forces determine the pose without an upright controller.
    collision_cfg.density = 50.0
    collision_cfg.ke = 2.0e5
    collision_cfg.kd = 20.0
    collision_cfg.mu = 0.0
    collision_cfg.kf = 0.0
    collision_cfg.is_visible = False
    visual_cfg = builder.default_shape_cfg.copy()
    visual_cfg.density = 0.0
    visual_cfg.has_shape_collision = False
    visual_cfg.has_particle_collision = False
    ballast_mass = (DUCK_DENSITY - collision_cfg.density) * collision.mass * DUCK_SCALE**3
    ballast_center = wp.vec3(0.0, 0.0, float(local_vertices[:, 2].min() + 0.025))
    ballast_inertia = wp.mat33(np.eye(3) * (0.4 * ballast_mass * 0.025**2))
    bodies = []
    for i, (x, y) in enumerate(DUCK_POSITIONS):
        position = wp.vec3(x, y, water_height - float(local_vertices[:, 2].min()) + 0.01)
        body = builder.add_body(
            xform=wp.transform(position, wp.quat_identity()),
            mass=ballast_mass,
            com=ballast_center,
            inertia=ballast_inertia,
            label=f"duck_{i}",
        )
        builder.add_shape_convex_hull(
            body,
            mesh=collision,
            scale=wp.vec3(DUCK_SCALE),
            xform=mesh_xform,
            cfg=collision_cfg,
            label=f"duck_{i}_collision",
        )
        builder.add_shape_mesh(
            body,
            mesh=visual,
            scale=wp.vec3(DUCK_SCALE),
            xform=mesh_xform,
            cfg=visual_cfg,
            color=(1.0, 1.0, 1.0),
            label=f"duck_{i}_visual",
        )
        bodies.append(body)
    return bodies, local_vertices


def configure_duck_picking(viewer, model, duck_bodies):
    picking = getattr(viewer, "picking", None)
    if picking is None:
        return
    visible = (model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE)) != 0
    selected = np.flatnonzero(visible & np.isin(model.shape_body.numpy(), duck_bodies))
    # The default raycast includes invisible tank planes, which occlude every duck from outside.
    # Give picking its own shape view while retaining the solver's body IDs and force buffers.
    pick_model = copy.copy(model)
    pick_model.shape_count = len(selected)
    for name in ("shape_body", "shape_transform", "shape_type", "shape_scale", "shape_source_ptr", "shape_world"):
        source = getattr(model, name)
        setattr(pick_model, name, wp.array(source.numpy()[selected], dtype=source.dtype, device=model.device))
    picking.model = pick_model
    viewer.picking_enabled = True


class Example:
    def __init__(self, viewer, args):
        if args.build_surface and args.viewer == "rtx":
            raise ValueError("--build-surface requires a viewer with dynamic mesh topology, such as gl or usd")
        if args.iterations < 1 or args.linear_iterations < 1 or (args.substeps is not None and args.substeps < 1):
            raise ValueError("iterations, linear_iterations, and substeps must be positive")
        positions, self.spacing = particle_layout(args.resolution)
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("The native fluid example requires a CUDA device (--device cuda:0).")
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = FRAME_DT
        self.sim_substeps = args.substeps if args.substeps is not None else math.ceil(2 * args.resolution / 32)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.radius = 0.45 * self.spacing
        self.volume = float(rest_volume(self.spacing))
        self.fluid_count = len(positions)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=positions,
            vel=np.zeros_like(positions),
            mass=np.full(len(positions), REST_DENSITY * self.volume),
            radius=np.full(len(positions), self.radius),
        )
        wall_cfg = builder.default_shape_cfg.copy()
        wall_cfg.ke = 2.0e5
        wall_cfg.kd = 20.0
        wall_cfg.kf = 0.0
        wall_cfg.mu = 0.0
        # Inward plane normals confine particles through native soft contacts, including the tank ceiling.
        for axis in range(3):
            for sign, bound in ((1.0, TANK_LOWER[axis]), (-1.0, TANK_UPPER[axis])):
                normal = np.zeros(3)
                normal[axis] = sign
                position = np.zeros(3)
                position[axis] = bound
                # Give the antiparallel ceiling normal an explicit rotation axis.
                rotation = (
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
                    if axis == 2 and sign < 0
                    else (wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*normal)))
                )
                builder.add_shape_plane(
                    xform=wp.transform(wp.vec3(*position), rotation), width=0.0, length=0.0, cfg=wall_cfg
                )
        self.duck_bodies, self.duck_vertices = add_ducks(builder, positions[:, 2].max() + 0.5 * self.spacing)
        self.model = builder.finalize(device=device)
        boundaries = boundary_samples(self.spacing)
        cha.configure_fluid(
            self.model,
            np.arange(len(positions), dtype=np.int32),
            smoothing_radius=2.0 * self.spacing,
            bulk_modulus=args.bulk_modulus,
            viscosity=args.viscosity,
            boundary_positions=boundaries,
            boundary_volumes=np.full(len(boundaries), self.volume),
        )
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.nonlinear_iterations = args.iterations
        self.solver.config.linear_iterations = args.linear_iterations
        self.solver.config.enable_collision_hessian_non_diagonal = True
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # The solver appends affine coordinates; only the original particles represent water.
        self.fluid_positions = self.state_0.particle_q[: self.fluid_count]
        self.fluid_velocities = self.state_0.particle_qd[: self.fluid_count]
        self.surface = (
            cha.ParticleSurface(self.spacing, self.fluid_positions, voxel_size=0.5 * self.spacing)
            if args.build_surface
            else None
        )
        self.colors = wp.empty(self.fluid_count, dtype=wp.vec3, device=device)
        self.initial_positions = positions.copy()
        self.stream = wp.get_stream(device)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_max=(6 + len(self.duck_bodies)) * len(positions)
        )
        self.contacts = self.collision_pipeline.contacts()
        corners = np.asarray(list(itertools.product(*zip(TANK_LOWER, TANK_UPPER, strict=True))), dtype=np.float32)
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(corners[i] != corners[j]) == 1]
        self.tank_starts = wp.array(corners[[i for i, _ in edges]], dtype=wp.vec3, device=device)
        self.tank_ends = wp.array(corners[[j for _, j in edges]], dtype=wp.vec3, device=device)
        self.viewer.set_model(self.model)
        configure_duck_picking(self.viewer, self.model, self.duck_bodies)
        self.viewer.show_particles = False
        self.viewer.show_static = False
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(4.8, -5.2, 3.3), pitch=-20.0, yaw=122.0)
        self.graph = None
        if not args.no_capture:
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph
        print(
            f"Native fluid: {self.fluid_count:,} particles, 5 ducks, {len(boundaries):,} boundary samples, "
            f"substeps={self.sim_substeps}, Newton/PCG={args.iterations}/{args.linear_iterations}, device={device}"
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            # Fixed buffer identities keep graph replay correct with odd substep counts.
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        mesh = self.surface.rebuild(self.fluid_positions) if self.surface is not None else None
        if mesh is None:
            wp.launch(
                color_particles,
                dim=self.fluid_count,
                inputs=[self.fluid_velocities],
                outputs=[self.colors],
                device=self.model.device,
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if mesh is not None:
            cha.log_surface_mesh(self.viewer, "fluid_surface", mesh, color=(0.025, 0.20, 0.65), roughness=0.18)
        else:
            self.viewer.log_points("fluid", self.fluid_positions, self.radius, self.colors)
        self.viewer.log_lines("tank", self.tank_starts, self.tank_ends, (0.45, 0.62, 0.72))
        self.viewer.end_frame()

    def test_final(self):
        positions = self.fluid_positions.numpy()
        velocities = self.fluid_velocities.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Fluid produced a non-finite state")
        # Penalty contacts allow small penetration; positions are never clamped after the native solve.
        if np.any(positions < np.asarray(TANK_LOWER) - 0.1 * self.spacing) or np.any(
            positions > np.asarray(TANK_UPPER) + 0.1 * self.spacing
        ):
            raise ValueError("Fluid particles escaped the tank")
        maximum_speed = float(np.linalg.norm(velocities, axis=1).max())
        if maximum_speed > 15.0:
            raise ValueError(f"Fluid became unstable: maximum speed {maximum_speed:.3f} m/s")
        if self.sim_time >= 1.0:
            if positions[:, 0].mean() < self.initial_positions[:, 0].mean() + 0.25:
                raise ValueError("The dam break did not spread along the tank")
            if np.quantile(positions[:, 2], 0.95) > np.quantile(self.initial_positions[:, 2], 0.95) - 0.15:
                raise ValueError("The initial water column did not collapse")
        samples = wp.array(
            np.concatenate((positions, self.model.synreal_fluid_boundary_positions.numpy())),
            dtype=wp.vec3,
            device=self.model.device,
        )
        volumes = wp.full(len(samples), self.volume, dtype=float, device=self.model.device)
        ratios = wp.empty(len(positions), dtype=float, device=self.model.device)
        grid = wp.HashGrid(64, 32, 64, device=self.model.device)
        grid.build(samples, 2.0 * self.spacing)
        wp.launch(
            diagnostic_density,
            dim=len(positions),
            inputs=[grid.id, samples, volumes, 2.0 * self.spacing],
            outputs=[ratios],
            device=self.model.device,
        )
        density = ratios.numpy()
        if not np.isfinite(density).all() or np.quantile(density, 0.99) > 1.35:
            raise ValueError("Native fluid retained excessive density compression")
        poses = self.state_0.body_q.numpy()[self.duck_bodies]
        body_velocities = self.state_0.body_qd.numpy()[self.duck_bodies]
        if not np.isfinite(poses).all() or not np.isfinite(body_velocities).all():
            raise ValueError("A duck produced a non-finite state")
        duck_radius = np.linalg.norm(self.duck_vertices, axis=1).max()
        for i, pose in enumerate(poses):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            vertices = self.duck_vertices @ rotation.T + pose[:3]
            if np.any(vertices < np.asarray(TANK_LOWER) - self.spacing) or np.any(
                vertices > np.asarray(TANK_UPPER) + self.spacing
            ):
                raise ValueError(f"Duck {i} escaped the tank")
            if self.sim_time >= 4.0:
                # Compare against nearby water: the breaking wave is not a level plane.
                nearby = np.linalg.norm(positions[:, :2] - pose[:2], axis=1) < 2.0 * duck_radius
                if np.count_nonzero(nearby) < 8:
                    raise ValueError(f"Duck {i} has no water beneath it")
                surface = np.quantile(positions[nearby, 2], 0.95) + 0.5 * self.spacing
                bottom, top = vertices[:, 2].min(), vertices[:, 2].max()
                if bottom <= TANK_LOWER[2] + 0.1 * self.spacing:
                    raise ValueError(f"Duck {i} is resting on the tank floor")
                if top < surface - 2.0 * self.spacing or bottom > surface + 2.0 * self.spacing:
                    raise ValueError(f"Duck {i} is not floating near the local water surface")
        print(
            f"Native fluid checks passed: max speed={maximum_speed:.3f} m/s, density p99={np.quantile(density, 0.99):.3f}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(num_frames=600)
        parser.add_argument("--resolution", type=int, default=32, help="Initial block X resolution (>=8).")
        parser.add_argument("--substeps", type=int, default=None, help="Default: ceil(2 * resolution / 32).")
        parser.add_argument("--iterations", type=int, default=5, help="Native Newton iterations per substep.")
        parser.add_argument("--linear-iterations", type=int, default=20, help="PCG iterations per Newton iteration.")
        parser.add_argument("--bulk-modulus", type=float, default=200000.0, help="Fluid bulk modulus in Pa.")
        parser.add_argument("--viscosity", type=float, default=0.1, help="Dynamic viscosity in Pa*s.")
        parser.add_argument(
            "--build-surface", action="store_true", help="Render a closed, reconstructed fluid surface mesh."
        )
        parser.add_argument("--no-capture", action="store_true", help="Disable CUDA graph capture for debugging.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    try:
        newton.examples.run(Example(viewer, args), args)
    except Exception:
        # The runner closes the viewer on success; USD viewers cannot be closed twice.
        viewer.close()
        raise
