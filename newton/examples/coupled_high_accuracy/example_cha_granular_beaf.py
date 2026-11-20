# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut a cohesive granular beef block with a manually dragged rigid knife.

Run: uv run python -m newton.examples cha_granular_beaf
Surface: uv run python -m newton.examples cha_granular_beaf --build-surface
Headless: uv run python -m newton.examples cha_granular_beaf --viewer null --num-frames 360 --test
Control: right-drag the knife to cut; release to return it above the beef.

The blade separates particles through ordinary rigid contacts; tensile strain
irreversibly breaks native granular bonds. This is a coarse cohesive material
demonstration, without an anatomical muscle or sharp-crack continuum model.
"""

import copy
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.coupled_high_accuracy.particle_surface import connected_components
from newton.solvers import coupled_high_accuracy as cha

FRAME_DT = 1.0 / 60.0
INITIAL_KNIFE_HEIGHT = 0.45
BEEF_DENSITY = 1050.0
KNIFE_MASS = 8.0
KNIFE_HEIGHT = 0.34
KNIFE_LENGTH = 0.70
KNIFE_PUSH_DISTANCE = 0.10
LEAN_DARK = wp.constant(wp.vec3(0.20, 0.018, 0.028))
LEAN_LIGHT = wp.constant(wp.vec3(0.36, 0.046, 0.059))
FAT_COLOR = wp.constant(wp.vec3(0.74, 0.67, 0.54))


@wp.func
def beef_material_coordinates(rest: wp.vec3):
    # Warping several noise scales produces irregular connective tissue rather than parallel fish-like stripes.
    coarse = rest * 11.0
    warp = wp.vec3(
        wp.noise(wp.uint32(17), coarse),
        wp.noise(wp.uint32(31), coarse + wp.vec3(4.3, 1.7, 8.1)),
        wp.noise(wp.uint32(47), coarse + wp.vec3(7.9, 3.4, 2.6)),
    )
    point = rest + 0.025 * warp
    vein = wp.noise(wp.uint32(59), point * 16.0) + 0.18 * wp.noise(wp.uint32(71), point * 38.0)
    branching = wp.noise(wp.uint32(83), point * 35.0) + 0.16 * wp.noise(wp.uint32(97), point * 71.0)
    region = wp.noise(wp.uint32(101), point * 9.0 + wp.vec3(1.2, 4.6, 7.3))
    thickness = 0.008 + 0.020 * wp.smoothstep(-0.3, 0.3, region)
    main_fat = 1.0 - wp.smoothstep(thickness * 0.35, thickness, wp.abs(vein))
    fine_fat = 1.0 - wp.smoothstep(0.003, 0.012, wp.abs(branching))
    # Interrupt the fine branches to keep some lean regions instead of enclosing every noise cell in fat.
    fine_fat *= wp.smoothstep(0.05, 0.3, region)
    fat = wp.max(main_fat, 0.55 * fine_fat)
    # A strip of subcutaneous fat follows one flank of the steak rather than covering all of its perimeter.
    center_y = 0.006 * wp.sin(9.0 * rest[0] + 0.4)
    half_width = 0.138 * (1.0 + 0.07 * wp.sin(11.0 * rest[0] + 0.6))
    footprint = (rest[0] / 0.36) ** 2.0 + ((rest[1] - center_y) / half_width) ** 2.0
    cap_start = 0.84 + 0.05 * wp.noise(wp.uint32(113), point * 23.0)
    fat_cap = wp.smoothstep(cap_start, cap_start + 0.10, footprint) * wp.smoothstep(0.0, 0.04, -rest[1])
    fat = wp.max(fat, fat_cap)
    grain = wp.noise(wp.uint32(127), wp.vec3(point[0] * 80.0, point[1] * 320.0, point[2] * 180.0))
    tone = wp.clamp(0.48 + 0.70 * wp.noise(wp.uint32(139), point * 18.0) + 0.06 * grain, 0.0, 1.0)
    return wp.vec2(fat, tone)


@wp.kernel
def color_beef_particles(rest_positions: wp.array[wp.vec3], colors: wp.array[wp.vec3]):
    i = wp.tid()
    material = beef_material_coordinates(rest_positions[i])
    lean = wp.lerp(LEAN_DARK, LEAN_LIGHT, material[1])
    fat = FAT_COLOR * (0.85 + 0.15 * material[1])
    colors[i] = wp.lerp(lean, fat, material[0])


def beef_texture():
    """Encode fat coverage and muscle variation in a compact, smoothly sampled color lookup."""
    fat = np.linspace(0.0, 1.0, 256)[None, :, None]
    tone = np.linspace(0.0, 1.0, 128)[:, None, None]
    lean = np.asarray(LEAN_DARK) + tone * (np.asarray(LEAN_LIGHT) - np.asarray(LEAN_DARK))
    fat_color = np.asarray(FAT_COLOR) * (0.85 + 0.15 * tone)
    albedo = lean + fat * (fat_color - lean)
    return np.round(albedo * 255.0).astype(np.uint8)


@wp.kernel
def surface_uvs(rest_positions: wp.array[wp.vec3], uvs: wp.array[wp.vec2]):
    i = wp.tid()
    # Transport volumetric marbling with each piece, including newly exposed cut faces.
    material = beef_material_coordinates(rest_positions[i])
    # Images have a top-left origin; texture V starts at the bottom of the uploaded image.
    uvs[i] = wp.vec2((0.5 + 255.0 * material[0]) / 256.0, (0.5 + 127.0 * (1.0 - material[1])) / 128.0)


class BeefSurfaceMaterial:
    def __init__(self):
        self.texture = beef_texture()

    def build_uvs(self, mesh):
        if mesh.rest_positions is None:
            raise ValueError("Beef surface materials require interpolated rest positions")
        uvs = wp.empty(len(mesh.vertices), dtype=wp.vec2, device=mesh.vertices.device)
        wp.launch(surface_uvs, dim=len(uvs), inputs=[mesh.rest_positions], outputs=[uvs], device=mesh.vertices.device)
        return uvs

    def log_mesh(self, viewer, mesh, uvs=None):
        if uvs is None:
            uvs = self.build_uvs(mesh)
        cha.log_surface_mesh(
            viewer,
            "beef_surface",
            mesh,
            uvs=uvs,
            texture=self.texture,
            color=(1.0, 1.0, 1.0),
            roughness=0.68,
            texture_lookup=True,
        )


def blade_mesh(spacing: float) -> newton.Mesh:
    """Resolve the bevel inside the lowest particle radius so cutting needs no slot in the board."""
    half_width = 0.35 * spacing
    section = np.array(
        [
            [0.0, 0.0],
            [half_width, 0.12 * spacing],
            [half_width, KNIFE_HEIGHT],
            [-half_width, KNIFE_HEIGHT],
            [-half_width, 0.12 * spacing],
        ],
        dtype=np.float32,
    )
    vertices = np.array(
        [[x, y, z] for y in (-0.5 * KNIFE_LENGTH, 0.5 * KNIFE_LENGTH) for x, z in section],
        dtype=np.float32,
    )
    faces = []
    # The section winds toward -Y. Reverse the far cap and wrap each side outward.
    for i in range(1, 4):
        faces.extend(((0, i, i + 1), (5, i + 6, i + 5)))
    for i in range(5):
        j = (i + 1) % 5
        faces.extend(((i, i + 5, j + 5), (i, j + 5, j)))
    return newton.Mesh(
        vertices,
        np.asarray(faces, dtype=np.int32).ravel(),
        compute_inertia=False,
        maxhullvert=0,
        roughness=0.25,
        metallic=0.8,
    )


def add_knife(builder, spacing: float) -> int:
    """Add a freely simulated cutting tool; all imposed motion is supplied as force and torque."""
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, INITIAL_KNIFE_HEIGHT), wp.quat_identity()),
        mass=KNIFE_MASS,
        com=wp.vec3(0.0),
        inertia=wp.mat33(np.diag([0.32, 0.18, 0.22])),
        lock_inertia=True,
        label="knife",
    )
    collision = builder.default_shape_cfg.copy()
    collision.density, collision.ke, collision.kd = 0.0, 2.0e5, 40.0
    collision.mu, collision.kf = 0.1, 20.0
    # The zero-thickness edge is intentional; particle radii provide the contact surface offset.
    collision.margin = 0.0
    builder.add_shape_mesh(body, mesh=blade_mesh(spacing), cfg=collision, color=(0.68, 0.75, 0.82), label="knife_blade")
    visual = collision.copy()
    visual.has_shape_collision, visual.has_particle_collision = False, False
    builder.add_shape_box(
        body,
        xform=wp.transform(wp.vec3(0.0, 0.48, KNIFE_HEIGHT - 0.04), wp.quat_identity()),
        hx=0.032,
        hy=0.15,
        hz=0.035,
        cfg=visual,
        color=(0.16, 0.075, 0.045),
        label="knife_handle",
    )
    return body


@wp.func
def knife_target(time: float, initial_height: float, final_height: float):
    height, speed = initial_height, 0.0
    if time >= 0.5 and time < 2.5:
        u = (time - 0.5) / 2.0
        height = initial_height + (final_height - initial_height) * u * u * (3.0 - 2.0 * u)
        speed = (final_height - initial_height) * 3.0 * u * (1.0 - u)
    elif time >= 2.5 and time < 3.8:
        height = final_height
    elif time >= 3.8 and time < 4.8:
        u = time - 3.8
        height = final_height + (initial_height - final_height) * u * u * (3.0 - 2.0 * u)
        speed = (initial_height - final_height) * 6.0 * u * (1.0 - u)
    return wp.vec2(height, speed)


@wp.func
def knife_push_target(time: float):
    u = wp.clamp(time - 2.8, 0.0, 1.0)
    distance = KNIFE_PUSH_DISTANCE * u * u * (3.0 - 2.0 * u)
    speed = KNIFE_PUSH_DISTANCE * 6.0 * u * (1.0 - u)
    return wp.vec2(distance, speed)


@wp.func
def knife_tilt_sample(index: int, seed: int):
    random_state = wp.rand_init(seed, index)
    return wp.vec3(
        0.6 * wp.randf(random_state, -1.0, 1.0),
        wp.randf(random_state, -1.0, 1.0),
        0.1 * wp.randf(random_state, -1.0, 1.0),
    )


@wp.func
def knife_orientation_target(time: float, tilt_degrees: float, seed: int):
    # Smooth seeded targets make the cutting plane wander without injecting frame-rate-dependent torque noise.
    phase = wp.max(time, 0.0) / 0.7
    index = int(wp.floor(phase))
    blend = phase - float(index)
    blend = blend * blend * (3.0 - 2.0 * blend)
    ramp = wp.min(phase, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    first, second = knife_tilt_sample(index, seed), knife_tilt_sample(index + 1, seed)
    angles = (first + blend * (second - first)) * (tilt_degrees * 0.017453292519943295 * ramp)
    # Keep yaw small so the knife nudges the cut face without scraping far along it.
    roll = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angles[0])
    lean = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angles[1])
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angles[2])
    return yaw * lean * roll


@wp.kernel
def drive_knife(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    clock: wp.array[float],
    dt: float,
    body: int,
    mass: float,
    initial_height: float,
    final_height: float,
    enabled: wp.array[int],
    pick_body: wp.array[int],
    tilt_degrees: float,
    tilt_seed: int,
):
    pose, velocity = body_q[body], body_qd[body]
    position, rotation = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
    linear, angular = wp.spatial_top(velocity), wp.spatial_bottom(velocity)
    target = wp.vec2(initial_height, 0.0)
    push = wp.vec2(0.0, 0.0)
    if enabled[0] != 0:
        target = knife_target(clock[0] + dt, initial_height, final_height)
        push = knife_push_target(clock[0] + dt)
    force = wp.vec3(0.0, 0.0, 9.81 * mass)
    if pick_body[0] != body:
        # Backward PD gains remain well behaved as substeps change; contact reaction can still deflect the tool.
        omega = 80.0
        denominator = 1.0 + 2.0 * omega * dt + omega * omega * dt * dt
        error = wp.vec3(push[0], 0.0, target[0]) - position - dt * linear
        target_velocity = wp.vec3(push[1], 0.0, target[1])
        acceleration = (omega * omega * error + 2.0 * omega * (target_velocity - linear)) / denominator
        drive = mass * acceleration
        magnitude = wp.length(drive)
        if magnitude > 4500.0:
            drive *= 4500.0 / magnitude
        force += drive
    # Retain orientation assistance while picked so a screen-space drag can operate the cutting edge.
    target_rotation = knife_orientation_target(clock[0] + dt, tilt_degrees, tilt_seed)
    # Level the edge near the solid board while preserving the random lean of the cutting plane.
    edge = wp.quat_rotate(target_rotation, wp.vec3(0.0, 1.0, 0.0))
    edge_level = wp.quat_between_vectors(edge, wp.vec3(0.0, 1.0, 0.0))
    near_board = 1.0 - wp.clamp((target[0] - final_height) / 0.12, 0.0, 1.0)
    near_board = near_board * near_board * (3.0 - 2.0 * near_board)
    target_rotation = wp.quat_slerp(wp.quat_identity(), edge_level, near_board) * target_rotation
    rotation_delta = target_rotation * wp.quat_inverse(rotation)
    sign = 1.0
    if rotation_delta[3] < 0.0:
        sign = -1.0
    error_rotation = 2.0 * sign * wp.vec3(rotation_delta[0], rotation_delta[1], rotation_delta[2])
    torque = (2000.0 * (error_rotation - dt * angular) - 70.0 * angular) / (1.0 + 400.0 * dt)
    magnitude_torque = wp.length(torque)
    if magnitude_torque > 600.0:
        torque *= 600.0 / magnitude_torque
    body_f[body] = body_f[body] + wp.spatial_vector(force, torque)


@wp.kernel
def advance_clock(clock: wp.array[float], dt: float):
    clock[0] += dt


@wp.kernel
def observe_fracture(
    positions: wp.array[wp.vec3],
    pairs: wp.array[wp.vec2i],
    rest: wp.array[float],
    break_strain: float,
    broken: wp.array[int],
):
    # Read-only diagnostics mirror accepted-step fracture with plastic compaction disabled.
    i = wp.tid()
    pair = pairs[i]
    if wp.length(positions[pair[0]] - positions[pair[1]]) > rest[i] * (1.0 + break_strain):
        broken[i] = 1


def particle_layout(resolution):
    """Use even x columns so the blade starts between grains, with all bonds intact."""
    if resolution < 12 or resolution % 2:
        raise ValueError("resolution must be an even integer of at least 12")
    spacing = 0.72 / resolution
    nx, ny, nz = resolution, round(0.3 / spacing), round(0.06 / spacing)
    axes = [(np.arange(n) + 0.5 - 0.5 * n) * spacing for n in (nx, ny)]
    z = np.arange(nz) * spacing + 0.43 * spacing + 0.002
    points = np.stack(np.meshgrid(*axes, z, indexing="ij"), axis=-1).reshape(-1, 3)
    # An asymmetric oval reads as a whole steak while preserving the isotropic bond lattice.
    x, y = points[:, 0], points[:, 1]
    center_y = 0.006 * np.sin(9.0 * x + 0.4)
    half_width = 0.138 * (1.0 + 0.07 * np.sin(11.0 * x + 0.6))
    footprint = (x / 0.36) ** 2 + ((y - center_y) / half_width) ** 2
    return points[footprint < 1.0].astype(np.float32), spacing


def configure_knife_picking(viewer, model, body):
    picking = getattr(viewer, "picking", None)
    if picking is None:
        return wp.array([-1], dtype=int, device=model.device)
    selected = np.flatnonzero(model.shape_body.numpy() == body)
    pick_model = copy.copy(model)
    pick_model.shape_count = len(selected)
    # Restrict raycasts to the knife so the board cannot intercept its drag gesture.
    for name in ("shape_body", "shape_transform", "shape_type", "shape_scale", "shape_source_ptr", "shape_world"):
        source = getattr(model, name)
        setattr(pick_model, name, wp.array(source.numpy()[selected], dtype=source.dtype, device=model.device))
    picking.model = pick_model
    viewer.picking_enabled = True
    return picking.pick_body


class Example:
    def __init__(self, viewer, args):
        if args.build_surface and args.viewer == "rtx":
            raise ValueError("--build-surface requires a viewer with dynamic mesh topology, such as gl or usd")
        if min(args.substeps, args.iterations, args.linear_iterations) < 1:
            raise ValueError("substeps and solver iterations must be positive")
        for name in ("cohesion", "break_strain"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(args.knife_tilt) or not 0.0 <= args.knife_tilt <= 3.0:
            raise ValueError("knife_tilt must be finite and between 0 and 3 degrees")
        points, self.spacing = particle_layout(args.resolution)
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("The native granular beef example requires a CUDA device (--device cuda:0).")
        self.viewer, self.sim_time = viewer, 0.0
        self.frame_dt, self.sim_substeps = FRAME_DT, args.substeps
        self.sim_dt, self.radius = FRAME_DT / args.substeps, 0.43 * self.spacing
        self.granular_count, self.initial_positions = len(points), points.copy()
        self.break_strain, self.auto_cut = args.break_strain, not args.no_cut
        self.knife_tilt, self.knife_seed = args.knife_tilt, args.seed
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_particles(
            pos=points,
            vel=np.zeros_like(points),
            mass=np.full(len(points), BEEF_DENSITY * self.spacing**3),
            radius=np.full(len(points), self.radius),
        )
        cfg = builder.default_shape_cfg.copy()
        cfg.ke, cfg.kd, cfg.kf, cfg.mu = 2.0e5, 50.0, 80.0, 0.45
        self.board_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.045), wp.quat_identity()),
            hx=0.68,
            hy=0.45,
            hz=0.045,
            cfg=cfg,
            color=(0.42, 0.235, 0.105),
            label="cutting_board",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.14), wp.quat_identity()),
            hx=0.95,
            hy=0.75,
            hz=0.04,
            cfg=cfg,
            color=(0.13, 0.17, 0.20),
        )
        self.knife_body = add_knife(builder, self.spacing)
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke, self.model.soft_contact_kd = cfg.ke, cfg.kd
        self.model.soft_contact_kf, self.model.soft_contact_mu = 15.0, 0.15
        cha.configure_granular(
            self.model,
            np.arange(len(points), dtype=np.int32),
            contact_stiffness=20000.0,
            friction=0.3,
            friction_damping=20.0,
            cohesion=args.cohesion,
            bond_damping=args.bond_damping,
            max_compaction=0.0,
            break_strain=args.break_strain,
            bond_radius=1.5 * self.spacing,
            adhesion=0.0,
        )
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.nonlinear_iterations = args.iterations
        self.solver.config.linear_iterations = args.linear_iterations
        self.solver.config.enable_collision_hessian_non_diagonal = True
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        # Affine rigid coordinates appended by the adapter are not visible beef particles.
        self.granular_positions = self.state_0.particle_q[: self.granular_count]
        self.granular_velocities = self.state_0.particle_qd[: self.granular_count]
        rest_positions = wp.array(points, dtype=wp.vec3, device=device)
        self.colors = wp.empty(len(points), dtype=wp.vec3, device=device)
        wp.launch(color_beef_particles, dim=len(points), inputs=[rest_positions], outputs=[self.colors], device=device)
        self.clock = wp.zeros(1, dtype=float, device=device)
        self.drive_enabled = wp.array([int(self.auto_cut)], dtype=int, device=device)
        self.bonds = self.model.synreal_granular_bond_indices
        self.broken = wp.zeros(len(self.bonds), dtype=int, device=device)
        self.surface = (
            cha.ParticleSurface(
                self.spacing,
                rest_positions,
                bonds=self.bonds,
                broken=self.broken,
                interpolate_rest_positions=True,
            )
            if args.build_surface
            else None
        )
        self.surface_material = BeefSurfaceMaterial() if self.surface is not None else None
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=6 * self.granular_count)
        self.contacts = self.collision_pipeline.contacts()
        viewer.set_model(self.model)
        self.pick_body = configure_knife_picking(viewer, self.model, self.knife_body)
        viewer.show_particles, viewer.show_ground = False, False
        viewer.set_camera(pos=wp.vec3(1.7, -2.2, 1.5), pitch=-26.0, yaw=126.0)
        self.stream, self.graph = wp.get_stream(device), None
        if not args.no_capture:
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph
        print(
            f"Granular beef: {self.granular_count:,} particles, {len(self.bonds):,} bonds, "
            f"rigid knife, auto cut={self.auto_cut}, substeps={self.sim_substeps}",
            flush=True,
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                drive_knife,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.clock,
                    self.sim_dt,
                    self.knife_body,
                    KNIFE_MASS,
                    INITIAL_KNIFE_HEIGHT,
                    0.02 * self.spacing,
                    self.drive_enabled,
                    self.pick_body,
                    self.knife_tilt,
                    self.knife_seed,
                ],
                device=self.model.device,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, None, self.contacts, self.sim_dt)
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
            wp.launch(
                observe_fracture,
                dim=len(self.bonds),
                inputs=[
                    self.granular_positions,
                    self.bonds,
                    self.model.synreal_granular_bond_rest_lengths,
                    self.break_strain,
                    self.broken,
                ],
                device=self.model.device,
            )
            wp.launch(advance_clock, dim=1, inputs=[self.clock, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        mesh = self.surface.rebuild(self.granular_positions) if self.surface is not None else None
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if mesh is not None:
            self.surface_material.log_mesh(self.viewer, mesh)
        else:
            self.viewer.log_points("beef", self.granular_positions, self.radius, self.colors)
        self.viewer.end_frame()

    def cut_stats(self):
        """Count connected material pieces from observed, irreversible bond fractures."""
        bonds, broken = self.bonds.numpy(), self.broken.numpy().astype(bool)
        labels = connected_components(self.granular_count, bonds[~broken])
        sizes = np.sort(np.unique(labels, return_counts=True)[1])[::-1]
        cross = self.initial_positions[bonds[:, 0], 0] * self.initial_positions[bonds[:, 1], 0] < 0.0
        points = self.granular_positions.numpy()
        left = self.initial_positions[:, 0] < 0.0
        gap = float(points[~left, 0].min() - points[left, 0].max() - 2.0 * self.radius)
        return dict(broken=int(broken.sum()), cross_remaining=int((cross & ~broken).sum()), pieces=sizes, gap=gap)

    def test_final(self):
        points, velocities = self.granular_positions.numpy(), self.granular_velocities.numpy()
        if not np.isfinite(points).all() or not np.isfinite(velocities).all():
            raise ValueError("Beef produced a non-finite state")
        if points[:, 2].min() < -0.15 or np.abs(points).max() > 2.0 or np.linalg.norm(velocities, axis=1).max() > 10.0:
            raise ValueError("Beef particles escaped or became unstable")
        stats = self.cut_stats()
        if self.sim_time >= 5.0 and self.auto_cut:
            if stats["cross_remaining"] or len(stats["pieces"]) < 2 or stats["pieces"][1] < 0.4 * self.granular_count:
                raise ValueError(f"The knife did not produce two connected beef pieces: {stats}")
        print(f"Beef checks passed: {stats}", flush=True)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        # Skip the automatic cutting trajectory while retaining mouse forces and the release-to-home controller.
        parser.set_defaults(num_frames=360, no_cut=True)
        parser.add_argument(
            "--resolution", type=int, default=64, help="Even particle count across 0.72 m; at least 12."
        )
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--iterations", type=int, default=5)
        parser.add_argument("--linear-iterations", type=int, default=40)
        parser.add_argument("--cohesion", type=float, default=2000.0, help="Initial bond stiffness in N/m.")
        parser.add_argument("--bond-damping", type=float, default=5.0, help="Internal viscous bond damping in N*s/m.")
        parser.add_argument("--break-strain", type=float, default=0.15, help="Irreversible tensile fracture threshold.")
        parser.add_argument(
            "--knife-tilt", type=float, default=2.0, help="Smooth random knife lean in degrees, from 0 to 3."
        )
        parser.add_argument(
            "--seed", type=int, default=17, help="Random seed for the knife's smooth orientation changes."
        )
        parser.add_argument(
            "--no-cut", action="store_true", help="Skip automatic cutting (default); drag the knife manually."
        )
        parser.add_argument(
            "--build-surface", action="store_true", help="Render a closed, reconstructed beef surface mesh."
        )
        parser.add_argument("--no-capture", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    try:
        newton.examples.run(Example(viewer, args), args)
    except Exception:
        # The runner closes the viewer on success; USD viewers cannot be closed twice.
        viewer.close()
        raise
