# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Piper loads five apples, swings their bag horizontally then vertically, and returns it to the platform.

Run from the repository root:
    uv run python demos/demo/demo_piper_bag_transport.py
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.usd
from newton import JointTargetMode, ShapeFlags
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d

DEMO_DIRECTORY = Path(__file__).resolve().parents[1] / "piper_bag_transport"

if __package__:
    from ..piper_bag_transport.bag_bending import rebuild_bag_bending
    from ..piper_bag_transport.bag_pattern import create_pattern
    from ..piper_bag_transport.motor_telemetry import MotorTelemetry
    from ..piper_bag_transport.piper_motion import PiperKinematics, loading_drop_rotation
else:
    sys.path.insert(0, str(DEMO_DIRECTORY))
    from bag_bending import rebuild_bag_bending
    from bag_pattern import create_pattern
    from motor_telemetry import MotorTelemetry
    from piper_motion import PiperKinematics, loading_drop_rotation


# Local width/depth resolution below the low mouth; shoulders and handles extend above it.
BAG_CELLS = (20, 10, 21)
CELL_SIZE = 0.01
BAG_BOTTOM = 0.033
BAG_CENTER = (0.0, 0.03)
BAG_SHOULDER_HEIGHT = 0.02
# The folded side gusset is sealed into each double-layer handle at its top.
HANDLE_CELLS = (5, 6)
HANDLE_RISE = 0.06
# Refined edge midpoints are 5 mm apart; the smaller radius keeps neighboring particles separated.
PARTICLE_RADIUS = 0.0015
BAG_COLOR = (1.0, 0.985, 0.90)
APPLE_MASS = 0.2
APPLE_COUNT = 5
APPLE_SCALE = 0.58
APPLE_COLLISION_MARGIN = 0.003
APPLE_FRICTION = 0.2
# Dimensionless contact rolling resistance; the torque limit is mu_r * normal_load * contact_radius.
FLOOR_ROLLING_FRICTION = 0.03
# Normal-axis spin needs its own contact torque; rolling resistance only damps the two tangent axes.
FLOOR_TORSIONAL_FRICTION = 0.005
FLOOR_FRICTION = 0.3
GRIPPER_FRICTION = 1.2
# The film must slide off an opened, inclined finger instead of remaining hung on its contact surface.
CLOTH_CONTACT_FRICTION = 0.6
CONTACT_FRICTION_GAIN = 1.0e3
# At 480 Hz, overly stiff or critically damped multi-point contacts suppress even the first rebound.
FLOOR_CONTACT_STIFFNESS = 3.0e3
FLOOR_CONTACT_DAMPING = 3.0
FLOOR_SIZE = (4.4, 4.4, 0.04)
FLOOR_CENTER = (0.0, -0.2, -0.02)
PLATFORM_SIZE = (0.70, 0.58, 0.03)
PLATFORM_CENTER = (0.0, -0.01, 0.015)
REPO_ROOT = Path(__file__).resolve().parents[2]
# Resolve asset-repository paths through the shared cache only when loading the scene.
APPLE_ASSET = "models/apple/apple.usd"
PIPER_ASSET = "robots/piper/piper_with_texture.xml"
# Seat the mounting face below the floor by 0.4 mm to avoid coplanar light leaks at the contact patch.
PIPER_BASE = (0.0, -0.52, -0.0004)
PIPER_ARM_KP = 3500.0
PIPER_ARM_KD = 35.0
PIPER_FINGER_KP = 6000.0
PIPER_FINGER_KD = 20.0
PIPER_BAG_FINGER_KP = 20000.0
PIPER_BAG_FINGER_KD = 40.0
PIPER_BAG_GAIN_RAMP = 0.5
GRIP_POINT_LOCAL = (0.0, 0.0, 0.115)
FINGER_TIP_LOCAL_Z = 0.1358
# These reachable poses seed IK; the complete loading and pickup plan supplies the actual motor targets.
PIPER_GRASP_Q = (0.0000105, 1.5619050, -1.0543934, 0.0000249, -0.4203075, 1.5707734, 0.012, -0.012)
PIPER_LIFT_Q = (0.0, 1.5954217, -1.8541491, 0.0, 0.3458371, 1.5707963, 0.0, 0.0)
APPLE_LOADING_START = 0.0
APPLE_LOADING_END = 0.45
HANDLE_OPENING = 0.007
GRASP_START = APPLE_LOADING_END
GRASP_REFINE_TIME = 0.30
GRASP_DURATION = 0.4
LIFT_START = GRASP_START + 0.9
LIFT_DURATION = 2.0
SWING_START = LIFT_START + LIFT_DURATION + 0.5
SWING_PERIOD = 1.35
SWING_CYCLES = 3
SWING_AMPLITUDE = 0.37
SWING_RAMP = 0.9
HORIZONTAL_START = SWING_START
HORIZONTAL_MOTION_PROFILE = "coordinated_cartesian"
HORIZONTAL_PERIOD = SWING_PERIOD
HORIZONTAL_CYCLES = SWING_CYCLES
HORIZONTAL_END = HORIZONTAL_START + HORIZONTAL_PERIOD * HORIZONTAL_CYCLES
HORIZONTAL_LATERAL_AMPLITUDE = 0.19
HORIZONTAL_DEPTH_ARC = 0.012
HORIZONTAL_HEIGHT_ARC = 0.009
HORIZONTAL_WRIST_PITCH = np.deg2rad(4.0)
HORIZONTAL_WAYPOINT_HZ = 100
VERTICAL_START = HORIZONTAL_END + 0.5
VERTICAL_MOTION_PROFILE = "asymmetric_jolt"
VERTICAL_PERIOD = 1.0
VERTICAL_CYCLES = 3
VERTICAL_END = VERTICAL_START + VERTICAL_PERIOD * VERTICAL_CYCLES
# The short downstroke briefly accelerates faster than gravity, unloading the apples without changing physics.
VERTICAL_CENTER_HEIGHT = 0.49
BOUNCE_AMPLITUDE = 0.14
VERTICAL_RISE_DURATION = 0.32
VERTICAL_DROP_DURATION = 0.24
# Shoulder and elbow drive the vertical path; the two joints immediately before the gripper hold their entry angles.
VERTICAL_ACTIVE_JOINTS = (1, 2)
VERTICAL_FIXED_JOINTS = (4, 5)
VERTICAL_WAYPOINT_HZ = 100
BOUNCE_FREQUENCY = 1.0 / VERTICAL_PERIOD
SWING_END = VERTICAL_END
AIRBORNE_RELEASE = False
PLACE_START = SWING_END
PLACE_DURATION = 4.0
PLACE_SETTLE_DURATION = 1.0
PLACE_GRIP_POINT = (0.0, -0.05, 0.29)
RELEASE_START = PLACE_START + PLACE_DURATION + PLACE_SETTLE_DURATION
RELEASE_DURATION = 0.4
RELEASE_OPENING = 0.035
RETRACT_DELAY = RELEASE_DURATION + 0.25
RETRACT_CENTER_DURATION = 0.35
RETRACT_CENTER_OFFSET = (0.025, 0.0, 0.0)
RETRACT_UNHOOK_DURATION = 0.5
RETRACT_UNHOOK_OFFSET = (0.025, -0.10, 0.06)
RETRACT_OFFSET = (0.025, -0.22, -0.07)
RETRACT_DURATION = RETRACT_CENTER_DURATION + 1.4
RETRACT_END = RELEASE_START + RETRACT_DELAY + RETRACT_DURATION
DURATION_SECONDS = float(np.ceil(RETRACT_END + 2.5))
DEFAULT_NUM_FRAMES = round(DURATION_SECONDS * 60)
# Computed by the same IK planner as the loading waypoints; retained for export metadata.
PIPER_RETRACT_Q = (-0.336610517, 0.775613584, -0.502096181, 0.699540327, 0.460359842, -0.819201129, 0.035, -0.035)


@wp.func
def smooth_motion(t: float):
    t = wp.clamp(t, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


@wp.kernel
def drive_finger_gains(
    time: wp.array[float],
    finger_dofs: wp.array[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
):
    # Ramp the existing finger gains when closing on the loaded handle.
    blend = smooth_motion((time[0] - APPLE_LOADING_END) / PIPER_BAG_GAIN_RAMP)
    dof = finger_dofs[wp.tid()]
    joint_target_ke[dof] = PIPER_FINGER_KP + blend * (PIPER_BAG_FINGER_KP - PIPER_FINGER_KP)
    joint_target_kd[dof] = PIPER_FINGER_KD + blend * (PIPER_BAG_FINGER_KD - PIPER_FINGER_KD)


@wp.kernel
def drive_piper(
    time: wp.array[float],
    dt: float,
    target_indices: wp.array[int],
    waypoint_times: wp.array[float],
    waypoint_q: wp.array2d[float],
    joint_target_q: wp.array[float],
):
    t = time[0]
    segment = int(0)
    for index in range(waypoint_times.shape[0] - 1):
        if t >= waypoint_times[index]:
            segment = index
    following = wp.min(segment + 1, waypoint_times.shape[0] - 1)
    duration = wp.max(waypoint_times[following] - waypoint_times[segment], 1.0e-6)
    blend = smooth_motion((t - waypoint_times[segment]) / duration)
    for joint in range(8):
        joint_target_q[target_indices[joint]] = waypoint_q[segment, joint] + blend * (
            waypoint_q[following, joint] - waypoint_q[segment, joint]
        )
        if (
            (t >= HORIZONTAL_START and t <= HORIZONTAL_END)
            or (t >= VERTICAL_START and t <= VERTICAL_END)
            or (t >= RELEASE_START + RETRACT_DELAY and t <= RETRACT_END)
        ):
            # Central tangents keep the dense IK path smooth instead of stopping at every sampled pose.
            path_start = float(VERTICAL_START)
            path_end = float(VERTICAL_END)
            if t <= HORIZONTAL_END:
                path_start = HORIZONTAL_START
                path_end = HORIZONTAL_END
            if t >= RELEASE_START + RETRACT_DELAY:
                path_start = RELEASE_START + RETRACT_DELAY
                path_end = RETRACT_END
            previous = wp.max(segment - 1, 0)
            after = wp.min(following + 1, waypoint_times.shape[0] - 1)
            start_rate = float(0.0)
            end_rate = float(0.0)
            if waypoint_times[segment] > path_start + 1.0e-5:
                start_rate = (waypoint_q[following, joint] - waypoint_q[previous, joint]) / (
                    waypoint_times[following] - waypoint_times[previous]
                )
            if waypoint_times[following] < path_end - 1.0e-5:
                end_rate = (waypoint_q[after, joint] - waypoint_q[segment, joint]) / (
                    waypoint_times[after] - waypoint_times[segment]
                )
            u = wp.clamp((t - waypoint_times[segment]) / duration, 0.0, 1.0)
            u2, u3 = u * u, u * u * u
            joint_target_q[target_indices[joint]] = (
                (2.0 * u3 - 3.0 * u2 + 1.0) * waypoint_q[segment, joint]
                + (u3 - 2.0 * u2 + u) * duration * start_rate
                + (-2.0 * u3 + 3.0 * u2) * waypoint_q[following, joint]
                + (u3 - u2) * duration * end_rate
            )
        if (t >= VERTICAL_START and t <= VERTICAL_END) and (joint == 4 or joint == 5):
            # Copy the held target exactly; float32 Hermite weights can otherwise introduce one-ULP jitter.
            joint_target_q[target_indices[joint]] = waypoint_q[segment, joint]
    time[0] = t + dt


def create_motion_plan(positions=None):
    """Drive only physical arm motors; the bag and apples remain free dynamic objects."""
    from scipy.spatial.transform import Rotation

    arm = PiperKinematics(s3d.get_asset(PIPER_ASSET), PIPER_BASE)
    seed = np.asarray(PIPER_GRASP_Q)
    times, poses = [], []

    def pose(time, position, rotation, opening, local=GRIP_POINT_LOCAL):
        nonlocal seed
        seed = arm.solve(position, rotation, seed, opening=opening, grip_point_local=local)
        times.append(time)
        poses.append(seed.copy())

    grasp_rotation = loading_drop_rotation(downward_tilt=0.0)
    lift_rotation = arm.fk(PIPER_GRASP_Q, GRIP_POINT_LOCAL)[:3, :3]
    bag_points = create_bag_mesh()[0]
    grip = bag_points[bag_grip_vertices(bag_points)].mean(axis=0)
    # The open lower jaw supports one handle by contact while apples pass through the unobstructed center.
    pose(0.0, grip, grasp_rotation, HANDLE_OPENING)
    pose(GRASP_REFINE_TIME, grip, grasp_rotation, HANDLE_OPENING)
    pose(GRASP_START, grip, grasp_rotation, HANDLE_OPENING)
    pose(GRASP_START + GRASP_DURATION, grip, grasp_rotation, 0.0)
    pose(LIFT_START, grip, grasp_rotation, 0.0)
    lift = np.array((0.0, -0.05, 0.51))
    pose(LIFT_START + LIFT_DURATION, lift, lift_rotation, 0.0, GRIP_POINT_LOCAL)
    horizontal_duration = HORIZONTAL_END - HORIZONTAL_START
    for time in np.linspace(HORIZONTAL_START, HORIZONTAL_END, round(horizontal_duration * HORIZONTAL_WAYPOINT_HZ) + 1):
        elapsed = time - HORIZONTAL_START
        ramp = np.clip(np.array((elapsed, horizontal_duration - elapsed)) / SWING_RAMP, 0.0, 1.0)
        envelope = np.prod(ramp**3 * (10.0 + ramp * (-15.0 + 6.0 * ramp)))
        lateral = envelope * np.sin(2.0 * np.pi * elapsed / HORIZONTAL_PERIOD)
        # A shallow Cartesian arc recruits the shoulder and elbow while the wrist follows the direction of travel.
        offset = (
            -HORIZONTAL_LATERAL_AMPLITUDE * lateral,
            -HORIZONTAL_DEPTH_ARC * lateral**2,
            HORIZONTAL_HEIGHT_ARC * lateral**2,
        )
        yaw = Rotation.from_euler("z", SWING_AMPLITUDE * lateral).as_matrix()
        pitch = Rotation.from_euler("x", -HORIZONTAL_WRIST_PITCH * lateral).as_matrix()
        pose(time, lift + offset, yaw @ pitch @ lift_rotation, 0.0, GRIP_POINT_LOCAL)
    pose(VERTICAL_START, lift, lift_rotation, 0.0, GRIP_POINT_LOCAL)
    start_time, start_height = VERTICAL_START, lift[2]
    recovery_duration = VERTICAL_PERIOD - VERTICAL_RISE_DURATION - VERTICAL_DROP_DURATION
    strokes = (
        (VERTICAL_RISE_DURATION, VERTICAL_CENTER_HEIGHT + BOUNCE_AMPLITUDE),
        (VERTICAL_DROP_DURATION, VERTICAL_CENTER_HEIGHT - BOUNCE_AMPLITUDE),
        (recovery_duration, lift[2]),
    )
    for _ in range(VERTICAL_CYCLES):
        for duration, destination in strokes:
            steps = round(duration * VERTICAL_WAYPOINT_HZ)
            for step in range(1, steps + 1):
                u = step / steps
                blend = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
                height = start_height + blend * (destination - start_height)
                # Solve at the held ribbon while allowing orientation to follow the shoulder and elbow naturally.
                seed = arm.solve_position((lift[0], lift[1], height), seed, VERTICAL_ACTIVE_JOINTS, GRIP_POINT_LOCAL)
                times.append(start_time + duration * u)
                poses.append(seed.copy())
            start_time, start_height = start_time + duration, destination
    pre_release_rotation = loading_drop_rotation()
    release_rotation = pre_release_rotation @ Rotation.from_rotvec((0.0, 0.0, -0.5 * np.pi)).as_matrix()
    release_point = np.asarray(PLACE_GRIP_POINT)
    centered = release_point + RETRACT_CENTER_OFFSET
    unhooked = release_point + RETRACT_UNHOOK_OFFSET
    withdrawn = release_point + RETRACT_OFFSET
    pose(PLACE_START + 0.8, lift, lift_rotation, 0.0, GRIP_POINT_LOCAL)
    # Lower before tilting, preserving the previously validated wrist branch and platform support sequence.
    pose(PLACE_START + 2.5, (0.0, -0.05, 0.40), pre_release_rotation, 0.0, GRIP_POINT_LOCAL)
    pose(PLACE_START + PLACE_DURATION, release_point, pre_release_rotation, 0.0, GRIP_POINT_LOCAL)
    pose(RELEASE_START, release_point, pre_release_rotation, 0.0, GRIP_POINT_LOCAL)
    pose(RELEASE_START + 0.15, release_point, pre_release_rotation, 0.025, GRIP_POINT_LOCAL)
    pose(RELEASE_START + RELEASE_DURATION, release_point, release_rotation, RELEASE_OPENING, GRIP_POINT_LOCAL)
    pose(RELEASE_START + RETRACT_DELAY, release_point, release_rotation, RELEASE_OPENING, GRIP_POINT_LOCAL)
    # Recenter the open gap, then extract along the inclined fingers before lowering the already cleared jaws.
    start_time, start_point = RELEASE_START + RETRACT_DELAY, release_point
    retreat_duration = RETRACT_DURATION - RETRACT_CENTER_DURATION - RETRACT_UNHOOK_DURATION
    for duration, destination in (
        (RETRACT_CENTER_DURATION, centered),
        (RETRACT_UNHOOK_DURATION, unhooked),
        (retreat_duration, withdrawn),
    ):
        for step in range(1, round(duration * 40) + 1):
            u = step / round(duration * 40)
            blend = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
            pose(
                start_time + duration * u,
                start_point + blend * (destination - start_point),
                release_rotation,
                RELEASE_OPENING,
                GRIP_POINT_LOCAL,
            )
        start_time, start_point = start_time + duration, destination
    pose(DURATION_SECONDS, withdrawn, release_rotation, RELEASE_OPENING, GRIP_POINT_LOCAL)
    return np.asarray(times, dtype=np.float32), np.asarray(poses, dtype=np.float32)


def add_piper(builder, initial_q=PIPER_GRASP_Q):
    """Import the fixed-base arm, retaining physical joints and mesh contacts rather than cloth attachments."""
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 1.0
    builder.default_shape_cfg.kf = CONTACT_FRICTION_GAIN
    builder.default_shape_cfg.mu = GRIPPER_FRICTION
    builder.add_mjcf(
        s3d.get_asset(PIPER_ASSET),
        xform=wp.transform(wp.vec3(*PIPER_BASE), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)),
        floating=False,
        enable_self_collisions=False,
        parse_visuals=True,
        parse_visuals_as_colliders=False,
        force_show_colliders=False,
        collapse_fixed_joints=True,
        parse_sites=False,
        parse_mujoco_options=False,
    )
    joint_ids = [
        next(i for i, label in enumerate(builder.joint_label) if label.endswith(f"/joint{j}")) for j in range(1, 9)
    ]
    for index, joint in enumerate(joint_ids):
        coord = builder.joint_q_start[joint]
        dof = builder.joint_qd_start[joint]
        target = coord
        builder.joint_q[coord] = initial_q[index]
        builder.joint_target_q[target] = initial_q[index]
        builder.joint_target_mode[dof] = int(JointTargetMode.POSITION_VELOCITY)
        # Let the carried load perturb the arm while retaining the native fingers' grip stiffness.
        builder.joint_target_ke[dof] = PIPER_ARM_KP if index < 6 else PIPER_FINGER_KP
        builder.joint_target_kd[dof] = PIPER_ARM_KD if index < 6 else PIPER_FINGER_KD
    for shape, source in enumerate(builder.shape_source):
        if not builder.shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES:
            continue
        # Mesh-backed full-surface cloth contacts require an SDF, including the imported finger meshes.
        hull = source.compute_convex_hull()
        hull.build_sdf(max_resolution=64, narrow_band_range=(-0.02, 0.02), cache_dir=REPO_ROOT / "build" / "piper-sdf")
        builder.shape_source[shape] = hull
        builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    wrist = next(i for i, label in enumerate(builder.body_label) if label.endswith("/link6"))
    return joint_ids, wrist


def bag_to_world(points):
    """Orient the handle openings toward the fingers so withdrawal can clear the complete ribbon."""
    points = np.asarray(points, dtype=np.float32)
    return np.stack(
        (-points[..., 1] + BAG_CENTER[0], points[..., 0] + BAG_CENTER[1], points[..., 2] + BAG_BOTTOM), axis=-1
    )


@lru_cache(maxsize=1)
def create_bag_mesh():
    """Open a flat-cut, gusseted T-shirt bag with welded bottom and handle heat seals."""
    points, faces, panel_points, panel_faces = create_pattern(
        BAG_CELLS,
        CELL_SIZE,
        BAG_SHOULDER_HEIGHT,
        HANDLE_CELLS[0],
        HANDLE_RISE,
    )
    return bag_to_world(points), faces, panel_points, panel_faces


def bag_grip_vertices(points):
    """Track the near handle crown seated between the original Piper fingers."""
    handle_y = BAG_CENTER[1] - (BAG_CELLS[0] - HANDLE_CELLS[0]) * CELL_SIZE * 0.5
    candidates = (np.abs(points[:, 0] - BAG_CENTER[0]) <= 0.5 * CELL_SIZE + 1.0e-6) & (
        np.abs(points[:, 1] - handle_y) <= CELL_SIZE + 1.0e-6
    )
    return np.flatnonzero(candidates & (points[:, 2] >= points[candidates, 2].max() - 0.001))


def load_apple_meshes():
    """Retain the original textured apple and its complete collision hull."""
    apple_asset = Path(s3d.get_asset(APPLE_ASSET))
    visual = newton.usd.get_mesh(
        apple_asset,
        root_path="/World/Apple/visual",
        load_normals=True,
        load_uvs=True,
        compute_inertia=False,
    )
    source = newton.usd.get_mesh(apple_asset, root_path="/World/Apple/collision", compute_inertia=False)
    # A simplified 64-vertex hull cuts inside this asset by more than its 3 mm contact margin.
    source.maxhullvert = 0
    hull = source.compute_convex_hull()
    collision = newton.Mesh(hull.vertices, hull.indices, maxhullvert=0)
    collision.build_sdf(max_resolution=64, narrow_band_range=(-0.02, 0.02), cache_dir=REPO_ROOT / "build" / "apple-sdf")
    center = (collision.vertices.min(axis=0) + collision.vertices.max(axis=0)) * 0.5
    return visual, collision, center


def apple_positions(half_extents):
    """Stagger the original apples within the mouth, with space between their full collision hulls."""
    half_extents = np.asarray(half_extents)
    clearance = half_extents[2] + APPLE_COLLISION_MARGIN + PARTICLE_RADIUS
    low = BAG_BOTTOM + BAG_CELLS[2] * CELL_SIZE + clearance + 0.010
    spacing = 2 * (half_extents[2] + APPLE_COLLISION_MARGIN) + 0.008
    # The lowest apple must also clear the raised handle ribbon; higher apples can use more of the wide mouth.
    offsets = ((-0.010, -0.011), (0.011, 0.016), (-0.008, 0.022), (0.011, -0.009), (-0.003, 0.001))
    return np.asarray(
        [(BAG_CENTER[0] + x, BAG_CENTER[1] + y, low + row * spacing) for row, (x, y) in enumerate(offsets)],
        dtype=np.float32,
    )


def add_environment(builder):
    """Use the same finite solid stage for collision, the live viewer, and USD rendering."""
    cfg = builder.default_shape_cfg.copy()
    cfg.ke = FLOOR_CONTACT_STIFFNESS
    # Keep impact damping low enough for a small rebound before rolling resistance settles the apple.
    cfg.kd = FLOOR_CONTACT_DAMPING
    cfg.kf = CONTACT_FRICTION_GAIN
    cfg.mu = FLOOR_FRICTION
    cfg.mu_rolling = FLOOR_ROLLING_FRICTION
    cfg.mu_torsional = FLOOR_TORSIONAL_FRICTION
    for label, size, center, color in (
        ("scene_floor", FLOOR_SIZE, FLOOR_CENTER, (0.48, 0.50, 0.53)),
        ("bag_platform", PLATFORM_SIZE, PLATFORM_CENTER, (0.22, 0.24, 0.27)),
    ):
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
            hx=size[0] * 0.5,
            hy=size[1] * 0.5,
            hz=size[2] * 0.5,
            cfg=cfg,
            color=color,
            label=label,
        )


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 2 or args.substeps % 2:
            raise ValueError("--substeps must be a positive even number for ping-pong CUDA graph replay")
        if args.iterations < 1 or args.linear_iterations < 1:
            raise ValueError("Solver iteration counts must be positive")
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.graph = None
        self.telemetry = None
        self.handle_plan_updated = False
        self.handle_grasp_refined = False

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        visual_mesh, collision_mesh, mesh_center = load_apple_meshes()
        self.apple_collision_vertices = (collision_mesh.vertices - mesh_center) * APPLE_SCALE
        half_extents = 0.5 * np.ptp(self.apple_collision_vertices, axis=0)
        self.initial_apple_positions = apple_positions(half_extents)
        self.plan_times, self.plan_q = create_motion_plan()
        robot_joints, self.wrist_body = add_piper(builder, self.plan_q[0])
        self.robot_body_count = builder.body_count
        points, indices, panel_points, panel_indices = create_bag_mesh()
        self.bag_panel_points = panel_points
        self.bag_panel_indices = panel_indices
        edges, counts = np.unique(
            np.sort(indices[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0, return_counts=True
        )
        self.mouth_vertices = np.unique(edges[counts == 1])
        self.bag_particle_count = len(points)
        self.grip_vertices = bag_grip_vertices(points)
        edge_start = len(builder.edge_indices)
        particle_start = builder.particle_count
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=points.tolist(),
            indices=indices.ravel().tolist(),
            panel_verts=panel_points.tolist(),
            panel_indices=panel_indices.ravel().tolist(),
            density=0.1,
            particle_radius=PARTICLE_RADIUS,
            # Resist in-plane extension while allowing fine, low-energy bending of the plastic film.
            tri_aniso_ke=wp.vec3(5.0e3, 5.0e3, 2.0e3),
            tri_ka=0.0e3,
            tri_kd=0.0e-4,
            edge_aniso_ke=wp.vec3(1.0e-6),
            edge_kd=0.0e-6,
            validate_mesh=True,
            label="plastic_bag",
        )
        self.bag_bending = rebuild_bag_bending(
            builder,
            indices,
            panel_points,
            panel_indices,
            particle_start=particle_start,
            edge_start=edge_start,
            edge_aniso_ke=wp.vec3(1.0e-6),
            edge_kd=0.0e-6,
        )
        builder.default_shape_cfg.kd = 0.1
        builder.default_shape_cfg.ke = 1.0e2
        builder.default_shape_cfg.kf = CONTACT_FRICTION_GAIN
        builder.default_shape_cfg.mu = APPLE_FRICTION
        collision_cfg = builder.default_shape_cfg.copy()
        collision_cfg.margin = APPLE_COLLISION_MARGIN
        collision_cfg.mu_rolling = 0.0
        collision_cfg.mu_torsional = 0.0
        collision_cfg.density = APPLE_MASS / (collision_mesh.mass * APPLE_SCALE**3)
        collision_cfg.is_visible = False
        visual_cfg = builder.default_shape_cfg.copy()
        visual_cfg.density = 0.0
        visual_cfg.has_shape_collision = False
        visual_cfg.has_particle_collision = False
        mesh_xform = wp.transform(wp.vec3(*(-APPLE_SCALE * mesh_center)), wp.quat_identity())
        self.bag_bodies = []
        for i, position in enumerate(self.initial_apple_positions):
            label = f"apple_{i}"
            body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label=label)
            builder.add_shape_convex_hull(
                body,
                mesh=collision_mesh,
                scale=wp.vec3(APPLE_SCALE),
                xform=mesh_xform,
                cfg=collision_cfg,
                label=f"{label}_collision",
            )
            builder.add_shape_mesh(
                body,
                mesh=visual_mesh,
                scale=wp.vec3(APPLE_SCALE),
                xform=mesh_xform,
                cfg=visual_cfg,
                color=(1.0, 1.0, 1.0),
                label=f"{label}_visual",
            )
            self.bag_bodies.append(body)
        add_environment(builder)
        self.model = builder.finalize()
        self.gripper_bodies = [
            i for i, label in enumerate(self.model.body_label) if label.endswith(("/link7", "/link8"))
        ]
        self.gripper_shapes = np.flatnonzero(
            np.isin(self.model.shape_body.numpy(), self.gripper_bodies)
            & ((self.model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_PARTICLES)) != 0)
        )
        # Generate ABD coordinates from the first loading waypoint rather than the MJCF zero-angle pose.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_kf = CONTACT_FRICTION_GAIN
        # Cloth/shape contact uses the smaller coefficient, retaining grip while apples can slide on the film.
        self.model.soft_contact_mu = CLOTH_CONTACT_FRICTION

        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        # Retain the PD Hessian bound on the light, valence-two boundary vertices introduced by refinement.
        self.solver.config.nonlinear_solver = s3d.NonlinearSolver.PD_NEWTON
        self.solver.config.inertia_mode = s3d.InertiaMode.NONE
        self.solver.config.joint_constraint_primal_iterations = 1
        self.solver.config.nonlinear_iterations = args.iterations
        self.solver.config.linear_iterations = args.linear_iterations
        # Keep cloth and rigid contact blocks coupled when the bag supports the stacked load.
        self.solver.config.enable_collision_hessian_non_diagonal = True
        self.solver.config.self_collision_radius = PARTICLE_RADIUS
        self.solver.config.enable_self_collision = True

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.motion_time = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.motor_targets = wp.array(self.model.joint_target_q_start.numpy()[robot_joints], dtype=wp.int32)
        self.finger_dofs = wp.array(self.model.joint_qd_start.numpy()[robot_joints[-2:]], dtype=wp.int32)
        self.waypoint_times = wp.array(self.plan_times, dtype=wp.float32)
        self.waypoint_q = wp.array(self.plan_q, dtype=wp.float32)
        # Face and edge contacts close the gaps between particles in this single-layer shell.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            # Dense convex-mesh stacks emit many more contacts than the default primitive-shape estimate.
            rigid_contact_max=max(4096, 512 * (APPLE_COUNT + self.robot_body_count)),
            soft_contact_margin=0.006,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = True
        self.viewer.log_mesh(
            "/model/triangles",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            color=BAG_COLOR,
            roughness=0.3,
            backface_culling=False,
        )
        self.viewer.set_camera(pos=wp.vec3(0.95, 0.75, 0.85), pitch=-24.0, yaw=230.0)
        if not args.no_telemetry:
            self.telemetry = MotorTelemetry(self, robot_joints, args.telemetry_csv)
        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for substep in range(self.sim_substeps):
            wp.launch(
                drive_finger_gains,
                dim=len(self.finger_dofs),
                inputs=[self.motion_time, self.finger_dofs],
                outputs=[self.model.joint_target_ke, self.model.joint_target_kd],
                device=self.model.device,
            )
            wp.launch(
                drive_piper,
                dim=1,
                inputs=[self.motion_time, self.sim_dt, self.motor_targets, self.waypoint_times, self.waypoint_q],
                outputs=[self.control.joint_target_q],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.telemetry is not None:
                self.telemetry.capture_substep(substep)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_handle_approach(self, refine_grasp=False):
        """Aim at the settled material crown after loading, without attaching or repositioning the bag."""
        arm = PiperKinematics(s3d.get_asset(PIPER_ASSET), PIPER_BASE)
        cloth = self.state_0.particle_q.numpy()[: self.bag_particle_count]
        crown = cloth[self.grip_vertices].mean(axis=0)
        rotation = loading_drop_rotation(downward_tilt=0.0)
        targets = [
            (GRASP_START, crown, HANDLE_OPENING),
            (GRASP_START + GRASP_DURATION, crown, 0.0),
            (LIFT_START, crown, 0.0),
        ]
        for time, point, opening in targets:
            index = int(np.argmin(np.abs(self.plan_times - time)))
            self.plan_q[index] = arm.solve(
                point, rotation, self.plan_q[index], opening=opening, grip_point_local=GRIP_POINT_LOCAL
            )
        self.waypoint_q.assign(self.plan_q)
        self.handle_plan_updated = True
        if refine_grasp:
            self.handle_grasp_refined = True

    def step(self):
        if self.sim_time >= GRASP_REFINE_TIME and not self.handle_grasp_refined:
            self.update_handle_approach(refine_grasp=True)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        if self.telemetry is not None:
            self.telemetry.record_frame()

    def finish_telemetry(self):
        if self.telemetry is not None:
            self.telemetry.finish()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def current_grip_point(self, body_poses):
        """Track the actual jaws, including asymmetric finger compliance under the carried load."""
        wrist = wp.transform(*body_poses[self.wrist_body])
        # Native finger origins sit at fingertip depth; move their midpoint back to the seated handle depth.
        offset = wp.vec3(0.0, 0.0, GRIP_POINT_LOCAL[2] - FINGER_TIP_LOCAL_Z)
        return body_poses[self.gripper_bodies, :3].mean(axis=0) + np.asarray(wp.transform_vector(wrist, offset))

    def test_post_step(self):
        # The live example runner otherwise checks only the final state and misses failed loading stages.
        self.test_final()

    def test_final(self):
        expected_bodies = APPLE_COUNT + self.robot_body_count
        if self.model.body_count != expected_bodies or len(self.bag_bodies) != APPLE_COUNT:
            raise AssertionError("Scene body counts do not match the arm and five apples")
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            if not np.isfinite(getattr(self.state_0, name).numpy()).all():
                raise AssertionError(f"Bag-and-apples {name} became non-finite")
        cloth = self.state_0.particle_q.numpy()[: self.bag_particle_count]
        if np.any(self.model.particle_mass.numpy()[: self.bag_particle_count] <= 0.0):
            raise AssertionError("The bag must not contain fixed particles")
        body_poses = self.state_0.body_q.numpy()
        wrist = wp.transform(*body_poses[self.wrist_body])
        carrying = LIFT_START <= self.sim_time < RELEASE_START
        if (
            carrying
            and np.linalg.norm(cloth[self.grip_vertices].mean(axis=0) - self.current_grip_point(body_poses)) > 0.025
        ):
            raise AssertionError("The gripper lost the bag handle")
        inside = body_poses[self.bag_bodies, :3]
        if carrying and (np.any(inside < cloth.min(axis=0) - 0.04) or np.any(inside > cloth.max(axis=0) + 0.04)):
            raise AssertionError("A carried apple escaped the bag")
        if APPLE_LOADING_END + 0.4 <= self.sim_time < LIFT_START:
            tolerance = float(np.linalg.norm(self.apple_collision_vertices, axis=1).max())
            if np.any(inside < cloth.min(axis=0) - tolerance) or np.any(inside > cloth.max(axis=0) + tolerance):
                raise AssertionError("A falling apple did not enter the bag")
        # Apples rotate irregular hulls; support must use their actual lowest transformed collision vertex.
        lowest_points = []
        for pose in body_poses[self.bag_bodies]:
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:])), dtype=np.float32).reshape(3, 3)
            lowest_points.append(float((self.apple_collision_vertices @ rotation.T)[:, 2].min() + pose[2]))
        platform_top = PLATFORM_CENTER[2] + PLATFORM_SIZE[2] * 0.5
        # A swinging corner may dip below an arbitrary height while still clearing the platform.
        if (
            LIFT_START + LIFT_DURATION <= self.sim_time < PLACE_START
            and cloth[:, 2].min() - PARTICLE_RADIUS <= platform_top
        ):
            raise AssertionError("The bag is no longer suspended above the platform")
        if cloth[:, 2].min() < -0.005:
            raise AssertionError("The bag fell through the floor")
        if np.min(lowest_points) < -0.005:
            raise AssertionError("An apple fell through the floor")
        if RELEASE_START - 0.5 * PLACE_SETTLE_DURATION <= self.sim_time <= RELEASE_START:
            if cloth[:, 2].min() - PARTICLE_RADIUS > platform_top + 0.01:
                raise AssertionError("The bag must rest on the platform before the gripper opens")
            if np.min(lowest_points) - APPLE_COLLISION_MARGIN > platform_top + 2.0 * PARTICLE_RADIUS + 0.01:
                raise AssertionError("The platform must support the loaded bag before release")
        if self.sim_time >= RETRACT_END + 0.25:
            if np.linalg.norm(cloth[self.grip_vertices].mean(axis=0) - self.current_grip_point(body_poses)) <= 0.05:
                raise AssertionError("The released handle has not cleared the open gripper")
            count = min(int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_max)
            if np.isin(self.contacts.soft_contact_shape.numpy()[:count], self.gripper_shapes).any():
                raise AssertionError("The opened jaws still support the released bag")
        if self.sim_time >= RETRACT_END + 1.0:
            count = min(int(self.contacts.soft_contact_count.numpy()[0]), self.contacts.soft_contact_max)
            finger_contact = np.isin(self.contacts.soft_contact_shape.numpy()[:count], self.gripper_shapes).any()
            if finger_contact:
                raise AssertionError("The open gripper still supports the bag")
            if cloth[:, 2].min() > 0.04:
                raise AssertionError("The released bag did not land on the stage")
        if self.sim_time >= DURATION_SECONDS - 0.25:
            radius = float(np.linalg.norm(self.apple_collision_vertices, axis=1).max())
            outside = np.any((inside < cloth.min(axis=0) - radius) | (inside > cloth.max(axis=0) + radius), axis=1)
            if np.any(outside):
                raise AssertionError("A placed apple escaped the bag")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument("--substeps", type=int, default=4, help="Even number of physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--linear-iterations", type=int, default=80)
        parser.add_argument(
            "--telemetry-csv",
            type=Path,
            default=DEMO_DIRECTORY / "outputs" / "motor_telemetry.csv",
            help="Record actual motor commands and native effort feedback at every frame.",
        )
        parser.add_argument("--no-telemetry", action="store_true", help="Disable motor CSV recording.")
        return parser


if __name__ == "__main__":
    wp.config.kernel_cache_dir = str(REPO_ROOT / "build" / "warp-cloth-apples")
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.finish_telemetry()
