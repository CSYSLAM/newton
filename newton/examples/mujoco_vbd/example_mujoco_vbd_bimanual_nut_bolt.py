# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Standalone MuJoCo/VBD acceptance demo.

This file owns its complete scene construction and trajectory implementation;
it does not import another example or a scene-specific helper module.
"""

from __future__ import annotations

"Turn a pre-threaded nut with the full Dexforce W1 in MJVBDV2.\n\nThe recorded hand targets drive the full W1 directly through realtime IK. Its\nleft hand physically holds a dynamic horizontal M20 bolt against gravity. The\nright middle fingertip presses onto the upper lateral face of the pre-threaded\nnut, follows a tangential arc, retracts, returns, and repeats. Only the middle\nfinger can collide with the nut, so its rotation is driven by contact friction\nrather than a prescribed nut trajectory.\n\nRun from the repository root::\n\n    uv run --extra examples -m newton.examples         mjvbd_v2_bimanual_nut_bolt --viewer gl\n"
import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

import newton
import newton.examples
import newton.ik as ik

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
ROBOT_URDF = ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"
HAND_KEYFRAME_PATH = ASSET_ROOT / "vbd_mjvbd_v2" / "mjvbd_v2_bimanual_nut_bolt_last_keyframe.json"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
ASSEMBLY = "m20_loose"
SDF_CACHE_DIR = Path(tempfile.gettempdir()) / "newton_sdf_cache"
FPS = 60
SIM_SUBSTEPS = 8
VBD_ITERATIONS = 40
INITIAL_BOLT_TIP_PROTRUSION = 0.00075
PRETHREAD_REVOLUTIONS = 4.9714285714
WORKSPACE_HEIGHT_OFFSET = 0.28
ASSEMBLY_SCALE = 1.75
ASSEMBLY_AXIS = np.array((1.0, 0.0, 0.0), dtype=np.float32)
ASSEMBLY_ORIGIN = np.array((-0.08, 0.0, 0.56 + WORKSPACE_HEIGHT_OFFSET), dtype=np.float32)
ASSEMBLY_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
NUT_START_OFFSET = 0.041 * ASSEMBLY_SCALE
M20_THREAD_PITCH = 0.0025 * ASSEMBLY_SCALE
BOLT_HEAD_END = 0.02
BOLT_HEAD_RADIUS = 0.015
BOLT_TIP_LOCAL_Z = 0.065
BOLT_DENSITY = 8000.0
NUT_OUTER_FACE_LOCAL_Z = 0.036
CONTACT_MARGIN = 0.0002
LEFT_HAND_CONTACT_KE = 2000000.0
RIGHT_HAND_CONTACT_KE = 100000.0
THREAD_CONTACT_KE = 300000.0
THREAD_CONTACT_KD = 300.0
THREAD_CONTACT_GAP = 0.0002
CONTACT_KD = 10.0
LEFT_HAND_CONTACT_KD = 5000.0
LEFT_HAND_CONTACT_MU = 100.0
RIGHT_HAND_CONTACT_MU = 2.0
RIGHT_MIDDLE_CONTACT_MU = 10.0
THREAD_CONTACT_MU = 1.0
NUT_ANGULAR_DAMPING = 0.03
RIGID_CONTACT_MAX = 32768
RIGID_BODY_CONTACT_BUFFER_SIZE = 8192
LEFT_GRIP_JOINT_OFFSETS_DEGREES = {
    "HAND_THUMB1": 0.6,
    "INDEX_PIP": 10.0,
    "MIDDLE_PIP": 4.0,
    "RING_PIP": 5.0,
    "PINKY_PIP": 5.0,
}
RIGHT_HAND_JOINT_OFFSETS_DEGREES = {"MIDDLE_PIP": -24.0}
RIGHT_STROKE_COUNT = 3
RIGHT_SETTLE_FRAMES = 8
RIGHT_APPROACH_FRAMES = 24
RIGHT_LOWER_FRAMES = 10
RIGHT_CONTACT_DWELL_FRAMES = 8
RIGHT_CONTACT_SETTLE_FRAMES = 18
RIGHT_STROKE_FRAMES = 60
RIGHT_LIFT_FRAMES = 8
RIGHT_RETURN_FRAMES = 12
RIGHT_INDEX_PIP_STROKE_DEGREES = 110.0
RIGHT_SIDE_STROKE_END_ROOT = np.array((0.0854, -0.1647, 0.6368 + WORKSPACE_HEIGHT_OFFSET), dtype=np.float32)
RIGHT_SIDE_STROKE_START_ROOT = np.array((0.0854, -0.22, 0.6368 + WORKSPACE_HEIGHT_OFFSET), dtype=np.float32)
RIGHT_STROKE_START_Y_OFFSETS = (0.0, 0.0, 0.0)
RIGHT_SIDE_STROKE_ARC_HEIGHT = 0.0088
RIGHT_SIDE_END_CLEARANCE = 0.004
RIGHT_SIDE_RETRACT = np.array((0.0, 0.0, 0.025), dtype=np.float32)
RIGHT_ADAPTIVE_CONTACT_CLEARANCE = 0.0
MIN_STROKE_ROTATION_DEGREES = 60.0
MAX_BOLT_GRIP_TRANSLATION = 0.02
MAX_BOLT_GRIP_ROTATION_DEGREES = 30.0
ROBOT_BASE_POSITION = wp.vec3(0.0, -0.28, -0.18)
ROBOT_BASE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
INITIAL_IK_ITERATIONS = 400
RUNTIME_IK_ITERATIONS = 60
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
J7_TO_HAND_BASE_ROTATIONS = {"LEFT": wp.quat(-0.5, 0.5, 0.5, 0.5), "RIGHT": wp.quat(0.5, -0.5, 0.5, 0.5)}
CAMERA_POSITION = wp.vec3(0.38, 0.67, 1.13)
CAMERA_PITCH = -20.7
CAMERA_YAW = -110.7
HAND_JOINT_DEGREES = {
    "HAND_THUMB1": 6.0,
    "HAND_THUMB2": 90.0,
    "HAND_INDEX": 41.0,
    "INDEX_PIP": 24.0,
    "HAND_MIDDLE": 57.0,
    "MIDDLE_PIP": 0.0,
    "HAND_RING": 48.0,
    "RING_PIP": 15.0,
    "HAND_PINKY": 24.0,
    "PINKY_PIP": 26.0,
}


def _quat_multiply(a: wp.quat, b: wp.quat) -> wp.quat:
    """Multiply host-side quaternions without launching a Warp kernel."""
    ax, ay, az, aw = map(float, a)
    bx, by, bz, bw = map(float, b)
    return wp.quat(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotate(rotation: wp.quat, vector: np.ndarray) -> np.ndarray:
    """Rotate one NumPy vector with a host-side Warp quaternion."""
    return np.asarray(wp.quat_rotate(rotation, wp.vec3(*vector)), dtype=np.float32)


def _smoothstep(value: float) -> float:
    """Interpolate a normalized scalar with zero endpoint velocities."""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _quat_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert one xyzw quaternion to a NumPy rotation matrix."""
    x, y, z, w = map(float, rotation)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _signed_twist_angle(rotation: np.ndarray, reference: np.ndarray, axis: np.ndarray) -> float:
    """Return the signed twist of one world rotation relative to another."""
    current = wp.quat(*rotation)
    initial = wp.quat(*reference)
    delta = np.asarray(tuple(_quat_multiply(current, wp.quat_inverse(initial))), dtype=np.float64)
    if delta[3] < 0.0:
        delta = -delta
    return 2.0 * math.atan2(float(delta[:3] @ axis), float(delta[3]))


def _load_hand_keyframe(path: Path) -> dict[str, dict]:
    """Load the two target hand poses used to initialize the full robot."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    keyframe = payload.get("keyframe")
    hands = keyframe.get("hands") if isinstance(keyframe, dict) else None
    if not isinstance(hands, dict):
        raise ValueError(f"Invalid bimanual nut/bolt keyframe: {path}")
    result = {}
    for side in ("LEFT", "RIGHT"):
        hand = hands.get(side)
        if not isinstance(hand, dict):
            raise ValueError(f"Missing {side} hand in keyframe: {path}")
        pose = hand.get("target_root_world")
        joints = hand.get("target_finger_joints_degrees")
        if not isinstance(pose, dict) or not isinstance(joints, dict):
            raise ValueError(f"Invalid {side} hand target in keyframe: {path}")
        position = pose.get("position_m")
        rotation = pose.get("quaternion_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Invalid {side} hand position in keyframe: {path}")
        if not isinstance(rotation, list) or len(rotation) != 4:
            raise ValueError(f"Invalid {side} hand rotation in keyframe: {path}")
        missing_joints = set(HAND_JOINT_DEGREES) - joints.keys()
        if missing_joints:
            raise ValueError(f"Missing {side} hand joints in keyframe: {sorted(missing_joints)}")
        result[side] = {
            "position": np.asarray(position, dtype=np.float32),
            "rotation": wp.normalize(wp.quat(*rotation)),
            "joint_degrees": {suffix: float(joints[suffix]) for suffix in HAND_JOINT_DEGREES},
        }
    return result


def _hexagonalize_bolt_head(vertices: np.ndarray) -> np.ndarray:
    """Replace the M20 asset's cylindrical head with a regular hexagonal head."""
    vertices = vertices.copy()
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    outer_head = (vertices[:, 2] <= BOLT_HEAD_END + 1e-06) & (radii >= 0.95 * BOLT_HEAD_RADIUS)
    angles = np.arctan2(vertices[outer_head, 1], vertices[outer_head, 0])
    sector_width = math.pi / 3.0
    sector_angle = np.remainder(angles, sector_width) - 0.5 * sector_width
    apothem = BOLT_HEAD_RADIUS * math.cos(math.pi / 6.0)
    boundary_radius = apothem / np.cos(sector_angle)
    vertices[outer_head, :2] *= (boundary_radius / radii[outer_head])[:, None]
    return vertices


def _build_nut_contact_envelope_mesh(thread_mesh: newton.Mesh) -> newton.Mesh:
    """Build a convex hex envelope for stable hand contact."""
    source_vertices = np.asarray(thread_mesh.vertices, dtype=np.float32)
    lower_z = float(source_vertices[:, 2].min())
    upper_z = float(source_vertices[:, 2].max())
    segment_count = 6
    angles = np.arange(segment_count, dtype=np.float32) * (math.pi / 3.0) + math.pi / 6.0
    circumradius = 0.015 / math.cos(math.pi / 6.0)
    outer_xy = circumradius * np.column_stack((np.cos(angles), np.sin(angles)))
    vertices = np.vstack(
        (
            np.column_stack((outer_xy, np.full(segment_count, lower_z))),
            np.column_stack((outer_xy, np.full(segment_count, upper_z))),
        )
    ).astype(np.float32)
    faces: list[tuple[int, int, int]] = []
    for index in range(segment_count):
        next_index = (index + 1) % segment_count
        bottom_i, bottom_j = (index, next_index)
        top_i, top_j = (segment_count + index, segment_count + next_index)
        faces.extend(
            (
                (bottom_i, bottom_j, top_j),
                (bottom_i, top_j, top_i),
                (segment_count, top_i, top_j),
                (0, bottom_j, bottom_i),
            )
        )
    return newton.Mesh(vertices, np.asarray(faces, dtype=np.int32).reshape(-1))


def _load_centered_sdf_mesh(path: Path, *, hexagonal_head: bool = False) -> tuple[newton.Mesh, np.ndarray]:
    """Load, center, and cook one threaded collision mesh."""
    mesh_data = trimesh.load(path, force="mesh")
    vertices = np.asarray(mesh_data.vertices, dtype=np.float32)
    if hexagonal_head:
        vertices = _hexagonalize_bolt_head(vertices)
    indices = np.asarray(mesh_data.faces, dtype=np.int32).reshape(-1)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = 0.5 * (lower + upper)
    mesh = newton.Mesh(vertices - center, indices)
    mesh.build_sdf(max_resolution=512, narrow_band_range=(-0.005, 0.005), margin=0.005, cache_dir=SDF_CACHE_DIR)
    return (mesh, center)


@wp.kernel
def _interpolate_joint_q(q_start: wp.array[float], q_end: wp.array[float], alpha: float, q_out: wp.array[float]):
    """Interpolate all kinematic hand coordinates over one display frame."""
    coordinate = wp.tid()
    q_out[coordinate] = q_start[coordinate] * (1.0 - alpha) + q_end[coordinate] * alpha


@wp.kernel
def _update_joint_velocity(
    q_start: wp.array[float],
    q_end: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    qd_out: wp.array[float],
):
    """Compute hand velocities from consecutive frame targets."""
    joint = wp.tid()
    q_begin = joint_q_start[joint]
    q_end_index = joint_q_start[joint + 1]
    qd_begin = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + 1]
    if joint_type[joint] == newton.JointType.FREE:
        qd_out[qd_begin + 0] = (q_end[q_begin + 0] - q_start[q_begin + 0]) * inv_dt
        qd_out[qd_begin + 1] = (q_end[q_begin + 1] - q_start[q_begin + 1]) * inv_dt
        qd_out[qd_begin + 2] = (q_end[q_begin + 2] - q_start[q_begin + 2]) * inv_dt
        rotation_delta = wp.normalize(
            wp.quat(q_end[q_begin + 3], q_end[q_begin + 4], q_end[q_begin + 5], q_end[q_begin + 6])
            * wp.quat_inverse(
                wp.quat(q_start[q_begin + 3], q_start[q_begin + 4], q_start[q_begin + 5], q_start[q_begin + 6])
            )
        )
        axis, angle = wp.quat_to_axis_angle(rotation_delta)
        qd_out[qd_begin + 3] = axis[0] * angle * inv_dt
        qd_out[qd_begin + 4] = axis[1] * angle * inv_dt
        qd_out[qd_begin + 5] = axis[2] * angle * inv_dt
    else:
        for coordinate in range(qd_end - qd_begin):
            if q_begin + coordinate < q_end_index:
                qd_out[qd_begin + coordinate] = (q_end[q_begin + coordinate] - q_start[q_begin + coordinate]) * inv_dt


@wp.kernel
def _apply_nut_angular_damping(
    body_qd: wp.array[wp.spatial_vector], body_f: wp.array[wp.spatial_vector], nut_body: int, damping: float
):
    """Apply a physical viscous torque that removes free nut spin."""
    angular_velocity = wp.spatial_bottom(body_qd[nut_body])
    wp.atomic_add(body_f, nut_body, wp.spatial_vector(wp.vec3(0.0), -damping * angular_velocity))


@wp.kernel
def _copy_robot_joint_q(source: wp.array[float], target: wp.array[float]):
    """Copy the full-W1 IK coordinates into the scene target."""
    coordinate = wp.tid()
    target[coordinate] = source[coordinate]


@wp.kernel
def _set_indexed_joint_q(indices: wp.array[int], values: wp.array[float], target: wp.array[float]):
    """Overwrite a small set of full-W1 joint coordinates."""
    index = wp.tid()
    target[indices[index]] = values[index]


@wp.kernel
def _lock_ik_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    """Restore non-arm coordinates after one IK solve."""
    index = wp.tid()
    q[0, indices[index]] = values[index]


class Example:
    """Hold a bolt with the left hand and tighten its nut with the right."""

    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    ROBOT_HAND_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")

    @staticmethod
    def create_parser():
        """Create command-line arguments for the bimanual nut/bolt scene."""
        parser = newton.examples.create_parser()
        parser.add_argument("--robot-urdf", type=Path, default=ROBOT_URDF)
        parser.add_argument("--ik-iterations", type=int, default=RUNTIME_IK_ITERATIONS)
        parser.add_argument("--robot-base-x", type=float, default=float(ROBOT_BASE_POSITION[0]))
        parser.add_argument("--robot-base-y", type=float, default=float(ROBOT_BASE_POSITION[1]))
        parser.add_argument("--robot-base-z", type=float, default=float(ROBOT_BASE_POSITION[2]))
        parser.add_argument("--robot-base-qx", type=float, default=float(ROBOT_BASE_ROTATION[0]))
        parser.add_argument("--robot-base-qy", type=float, default=float(ROBOT_BASE_ROTATION[1]))
        parser.add_argument("--robot-base-qz", type=float, default=float(ROBOT_BASE_ROTATION[2]))
        parser.add_argument("--robot-base-qw", type=float, default=float(ROBOT_BASE_ROTATION[3]))
        parser.add_argument(
            "--initial-keyframe",
            type=Path,
            default=HAND_KEYFRAME_PATH,
            help="Bimanual hand keyframe used to initialize the full robot.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture one complete MuJoCo/VBD display frame on CUDA.",
        )
        parser.set_defaults(num_frames=350)
        return parser

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self.test_mode = bool(args.test)
        self.robot_base_position = wp.vec3(args.robot_base_x, args.robot_base_y, args.robot_base_z)
        self.robot_base_rotation = wp.normalize(
            wp.quat(args.robot_base_qx, args.robot_base_qy, args.robot_base_qz, args.robot_base_qw)
        )
        if args.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        self.ik_iterations = int(args.ik_iterations)
        robot_path = Path(args.robot_urdf).expanduser().resolve()
        keyframe_path = Path(args.initial_keyframe).expanduser().resolve()
        for description, path in (("Dexforce W1 URDF", robot_path), ("Bimanual hand keyframe", keyframe_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")
        self.robot_urdf = robot_path
        self.initial_hand_poses = _load_hand_keyframe(keyframe_path)
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        bolt_mesh, bolt_center = _load_centered_sdf_mesh(
            asset_path / f"factory_bolt_{ASSEMBLY}.obj", hexagonal_head=True
        )
        nut_thread_mesh, nut_center = _load_centered_sdf_mesh(asset_path / f"factory_nut_{ASSEMBLY}_subdiv_3x.obj")
        nut_hand_mesh = _build_nut_contact_envelope_mesh(nut_thread_mesh)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = RIGHT_HAND_CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = RIGHT_HAND_CONTACT_MU
        builder.default_shape_cfg.margin = CONTACT_MARGIN
        builder.default_shape_cfg.gap = 0.0
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        newton.solvers.SolverMuJoCoVBD.register_custom_attributes(builder)
        robot_articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(self.robot_base_position, self.robot_base_rotation),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = list(range(robot_articulation_start, builder.articulation_count))
        if not self.robot_articulations:
            raise RuntimeError("Dexforce W1 URDF did not create an articulation")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.hand_shapes: dict[str, list[int]] = {"LEFT": [], "RIGHT": []}
        self.robot_non_hand_shapes: list[int] = []
        self.robot_visual_shapes: list[int] = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            side = "LEFT" if "left" in label else "RIGHT" if "right" in label else None
            is_hand = side is not None and any(keyword in label for keyword in self.ROBOT_HAND_KEYWORDS)
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if not is_collider:
                self.robot_visual_shapes.append(shape)
            if is_hand and is_collider:
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                builder.shape_material_ke[shape] = LEFT_HAND_CONTACT_KE if side == "LEFT" else RIGHT_HAND_CONTACT_KE
                builder.shape_material_kd[shape] = LEFT_HAND_CONTACT_KD if side == "LEFT" else CONTACT_KD
                builder.shape_material_mu[shape] = LEFT_HAND_CONTACT_MU if side == "LEFT" else RIGHT_HAND_CONTACT_MU
                self.hand_shapes[side].append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
                self.robot_non_hand_shapes.append(shape)
        if not self.hand_shapes["LEFT"] or not self.hand_shapes["RIGHT"]:
            raise RuntimeError("The full W1 URDF did not produce hand collision shapes")
        thread_cfg = newton.ModelBuilder.ShapeConfig(
            density=BOLT_DENSITY,
            ke=THREAD_CONTACT_KE,
            kd=THREAD_CONTACT_KD,
            mu=THREAD_CONTACT_MU,
            gap=THREAD_CONTACT_GAP,
            margin=0.0,
        )
        bolt_rotation = ASSEMBLY_ROTATION
        bolt_position = ASSEMBLY_ORIGIN + _rotate(bolt_rotation, bolt_center * ASSEMBLY_SCALE)
        self.bolt_body = builder.add_body(
            xform=wp.transform(wp.vec3(*bolt_position), bolt_rotation), label="left_hand_bolt"
        )
        self.bolt_shape = builder.add_shape_mesh(
            self.bolt_body,
            mesh=bolt_mesh,
            scale=(ASSEMBLY_SCALE,) * 3,
            cfg=thread_cfg,
            color=(0.33, 0.38, 0.45),
            label="threaded_bolt_sdf",
        )
        nut_local_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 8.0)
        nut_entry_rotation = _quat_multiply(ASSEMBLY_ROTATION, nut_local_rotation)
        prethread_angle = math.radians(15.0)
        prethread_rotation = wp.quat_from_axis_angle(wp.vec3(*ASSEMBLY_AXIS), prethread_angle)
        nut_rotation = _quat_multiply(prethread_rotation, nut_entry_rotation)
        nut_origin = ASSEMBLY_ORIGIN + ASSEMBLY_AXIS * (NUT_START_OFFSET - PRETHREAD_REVOLUTIONS * M20_THREAD_PITCH)
        self.nut_origin = nut_origin.copy()
        nut_position = nut_origin + _rotate(nut_rotation, nut_center * ASSEMBLY_SCALE)
        self.nut_body = builder.add_body(
            xform=wp.transform(wp.vec3(*nut_position), nut_rotation), label="right_hand_nut"
        )
        nut_hand_cfg = newton.ModelBuilder.ShapeConfig(
            density=8000.0, ke=RIGHT_HAND_CONTACT_KE, kd=CONTACT_KD, mu=RIGHT_HAND_CONTACT_MU, gap=0.0, margin=0.0
        )
        self.nut_shape = builder.add_shape_convex_hull(
            self.nut_body,
            mesh=nut_hand_mesh,
            scale=(ASSEMBLY_SCALE,) * 3,
            cfg=nut_hand_cfg,
            color=(0.88, 0.46, 0.1),
            label="nut_hand_contact_envelope",
        )
        nut_thread_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=THREAD_CONTACT_KE,
            kd=THREAD_CONTACT_KD,
            mu=THREAD_CONTACT_MU,
            gap=THREAD_CONTACT_GAP,
            margin=0.0,
        )
        self.nut_thread_shape = builder.add_shape_mesh(
            self.nut_body,
            mesh=nut_thread_mesh,
            scale=(ASSEMBLY_SCALE,) * 3,
            cfg=nut_thread_cfg,
            color=(0.88, 0.46, 0.1),
            label="threaded_nut_visual_and_contact_sdf",
        )
        self._thread_bolt_center_host = bolt_center * ASSEMBLY_SCALE
        self._thread_nut_center_host = nut_center * ASSEMBLY_SCALE
        self._initial_nut_axis_origin = nut_origin.copy()
        thread_initial_relative_rotation = _quat_multiply(wp.quat_inverse(bolt_rotation), nut_rotation)
        self._thread_initial_relative_rotation_host = np.asarray(
            tuple(thread_initial_relative_rotation), dtype=np.float32
        )
        self.thread_initial_axial_offset = float(np.dot(nut_origin - ASSEMBLY_ORIGIN, ASSEMBLY_AXIS))
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible = int(newton.ShapeFlags.VISIBLE)
        builder.shape_flags[self.bolt_shape] |= collide_shapes
        builder.shape_flags[self.nut_shape] |= collide_shapes
        builder.shape_flags[self.nut_shape] &= ~visible
        builder.shape_flags[self.nut_thread_shape] |= visible
        self.left_grip_shapes = list(self.hand_shapes["LEFT"])
        for shape in (*self.hand_shapes["LEFT"], *self.hand_shapes["RIGHT"], self.nut_thread_shape):
            builder.shape_flags[shape] |= collide_shapes
        for shape in self.hand_shapes["LEFT"]:
            builder.add_shape_collision_filter_pair(shape, self.nut_shape)
            builder.add_shape_collision_filter_pair(shape, self.nut_thread_shape)
        for shape in self.hand_shapes["RIGHT"]:
            builder.add_shape_collision_filter_pair(shape, self.bolt_shape)
            builder.add_shape_collision_filter_pair(shape, self.nut_thread_shape)
        builder.add_shape_collision_filter_pair(self.bolt_shape, self.nut_shape)
        self.right_middle_shapes = [
            shape
            for shape in self.hand_shapes["RIGHT"]
            if builder.body_label[int(builder.shape_body[shape])].endswith("/right_middle_dist")
        ]
        right_middle_shape_set = set(self.right_middle_shapes)
        for shape in self.right_middle_shapes:
            builder.shape_material_mu[shape] = RIGHT_MIDDLE_CONTACT_MU
        for shape in self.hand_shapes["RIGHT"]:
            if shape not in right_middle_shape_set:
                builder.add_shape_collision_filter_pair(shape, self.nut_shape)
        builder.color()
        self.model = builder.finalize(requires_grad=False)
        self.model.rigid_contact_max = RIGID_CONTACT_MAX
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[self.hand_shapes["LEFT"]] = LEFT_HAND_CONTACT_KE
        shape_ke[self.hand_shapes["RIGHT"]] = RIGHT_HAND_CONTACT_KE
        shape_kd[self.hand_shapes["LEFT"]] = LEFT_HAND_CONTACT_KD
        shape_kd[self.hand_shapes["RIGHT"]] = CONTACT_KD
        shape_mu[self.hand_shapes["LEFT"]] = LEFT_HAND_CONTACT_MU
        shape_mu[self.hand_shapes["RIGHT"]] = RIGHT_HAND_CONTACT_MU
        shape_mu[self.right_middle_shapes] = RIGHT_MIDDLE_CONTACT_MU
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self._initialize_hand_coordinates(self.model, record_indices=True)
        self._build_robot_ik()
        self._initialize_robot_pose()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.solver = newton.solvers.SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0,
                "rigid_contact_stick_freeze_translation_eps": 0.001,
                "rigid_contact_stick_freeze_angular_eps": 0.001,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "rigid_contact_max": RIGID_CONTACT_MAX,
                "include_static_kinematic_pairs": False,
            },
            coupling_mode="one_way",
        )
        if self.solver.features.backend.value != "one_way_kinematic_full":
            raise RuntimeError(
                f"The bimanual nut/bolt scene requires vbd_kinematic_full, got {self.solver.features.backend.value}"
            )
        self.contacts = self.solver.contacts
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.hand_target_q = wp.clone(self.model.joint_q)
        right_index_pip = self.hand_joint_q_indices["RIGHT"]["INDEX_PIP"]
        self.right_index_pip_indices = wp.array([right_index_pip], dtype=wp.int32, device=self.model.device)
        self.right_index_pip_values = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.initial_right_index_pip = float(self.model.joint_q.numpy()[right_index_pip])
        self.current_hand_root_targets = self._initial_hand_root_targets()
        self.initial_bolt_transform = self.state_0.body_q.numpy()[self.bolt_body].copy()
        self.right_other_shapes = [
            shape for shape in self.hand_shapes["RIGHT"] if shape not in set(self.right_middle_shapes)
        ]
        self._build_right_stroke_trajectory()
        self.maximum_thread_contact_count = 0
        self.maximum_left_hand_bolt_contact_count = 0
        self.last_left_hand_bolt_contact_count = 0
        self.maximum_other_right_hand_nut_contact_count = 0
        self.stroke_middle_contact_counts = [0] * RIGHT_STROKE_COUNT
        self.stroke_middle_contact_frame_counts = [0] * RIGHT_STROKE_COUNT
        self.stroke_rotation_starts = [None] * RIGHT_STROKE_COUNT
        self.stroke_rotation_ends = [None] * RIGHT_STROKE_COUNT
        self.accumulated_nut_rotation = 0.0
        self._last_nut_twist_angle = 0.0
        self.maximum_bolt_displacement = 0.0
        self.maximum_bolt_rotation = 0.0
        self.maximum_nut_radial_offset = 0.0
        self.maximum_nut_outward_motion = 0.0
        self.minimum_nut_inward_motion = 0.0
        self.bolt_tip_protrusion = 0.0
        self.maximum_robot_hand_position_error = 0.0
        self.maximum_robot_hand_angle_error = 0.0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=CAMERA_POSITION, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 38.0
        self.use_graph = bool(args.graph_capture) and self.model.device.is_cuda
        self.graph = None
        self.capture()

    def _initialize_hand_coordinates(self, model: newton.Model, *, record_indices: bool = False) -> None:
        """Apply the recorded finger poses directly to the full W1."""
        joint_q = model.joint_q.numpy()
        joint_q_start = model.joint_q_start.numpy()
        if record_indices:
            self.hand_joint_q_indices: dict[str, dict[str, int]] = {"LEFT": {}, "RIGHT": {}}
        for side in ("LEFT", "RIGHT"):
            for suffix, degrees in self.initial_hand_poses[side]["joint_degrees"].items():
                offsets = LEFT_GRIP_JOINT_OFFSETS_DEGREES if side == "LEFT" else RIGHT_HAND_JOINT_OFFSETS_DEGREES
                target_degrees = degrees + offsets.get(suffix, 0.0)
                label = f"{side}_{suffix}"
                joint = next((index for index, name in enumerate(model.joint_label) if name.endswith("/" + label)))
                q_index = int(joint_q_start[joint])
                if record_indices:
                    self.hand_joint_q_indices[side][suffix] = q_index
                joint_q[q_index] = math.radians(target_degrees)
        model.joint_q.assign(joint_q)

    def _initial_hand_root_targets(self) -> dict[str, wp.transform]:
        """Return the two recorded hand-root targets."""
        return {
            side: wp.transform(wp.vec3(*pose["position"]), pose["rotation"])
            for side, pose in self.initial_hand_poses.items()
        }

    def _build_robot_ik(self) -> None:
        """Build the full-W1 arm IK used to place both robot hands."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(self.robot_base_position, self.robot_base_rotation),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = builder.finalize(device=self.model.device)
        self._initialize_hand_coordinates(self.ik_model)
        if self.ik_model.joint_coord_count > self.model.joint_coord_count:
            raise RuntimeError("Full-W1 IK coordinates do not fit the simulation model")
        left_body = self._body_index(self.ik_model.body_label, "left_j7")
        right_body = self._body_index(self.ik_model.body_label, "right_j7")
        self.robot_j7_bodies = {
            "LEFT": self._body_index(self.model.body_label, "left_j7"),
            "RIGHT": self._body_index(self.model.body_label, "right_j7"),
        }
        initial_targets = self._initial_hand_root_targets()
        left_target = self._root_to_tcp("LEFT", initial_targets["LEFT"])
        right_target = self._root_to_tcp("RIGHT", initial_targets["RIGHT"])
        self.left_position_objective = ik.IKObjectivePosition(
            left_body,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(left_target)], dtype=wp.vec3, device=self.model.device),
        )
        self.left_rotation_objective = ik.IKObjectiveRotation(
            left_body,
            wp.quat_identity(),
            wp.array(
                [self._quaternion_vector(wp.transform_get_rotation(left_target))],
                dtype=wp.vec4,
                device=self.model.device,
            ),
        )
        self.right_position_objective = ik.IKObjectivePosition(
            right_body,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(right_target)], dtype=wp.vec3, device=self.model.device),
        )
        self.right_rotation_objective = ik.IKObjectiveRotation(
            right_body,
            wp.quat_identity(),
            wp.array(
                [self._quaternion_vector(wp.transform_get_rotation(right_target))],
                dtype=wp.vec4,
                device=self.model.device,
            ),
        )
        lower, upper = self._robot_joint_limits()
        limit_objective = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.model.device),
            wp.array(upper, dtype=wp.float32, device=self.model.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[
                self.left_position_objective,
                self.left_rotation_objective,
                self.right_position_objective,
                self.right_rotation_objective,
                limit_objective,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_lock_indices, self.ik_lock_values = self._locked_robot_q()

    def _initialize_robot_pose(self) -> None:
        """Solve both arms to the lowered initial hand-root targets."""
        targets = self._initial_hand_root_targets()
        self._set_robot_ik_targets(targets)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        self._lock_robot_ik()
        joint_q = self.model.joint_q.numpy()
        joint_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.zero_()

    def _update_robot_target(self, targets: dict[str, wp.transform]) -> None:
        """Solve the full W1 arms for the two hand-root targets."""
        self._set_robot_ik_targets(targets)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._lock_robot_ik()
        wp.launch(
            _copy_robot_joint_q,
            self.ik_model.joint_coord_count,
            [self.ik_q[0], self.hand_target_q],
            device=self.model.device,
        )

    def _set_robot_ik_targets(self, root_targets: dict[str, wp.transform]) -> None:
        """Convert hand-root targets into full-W1 wrist targets."""
        left = self._root_to_tcp("LEFT", root_targets["LEFT"])
        right = self._root_to_tcp("RIGHT", root_targets["RIGHT"])
        self.left_position_objective.set_target_position(0, wp.transform_get_translation(left))
        self.left_rotation_objective.set_target_rotation(0, self._quaternion_vector(wp.transform_get_rotation(left)))
        self.right_position_objective.set_target_position(0, wp.transform_get_translation(right))
        self.right_rotation_objective.set_target_rotation(0, self._quaternion_vector(wp.transform_get_rotation(right)))

    def _lock_robot_ik(self) -> None:
        """Keep every full-W1 coordinate except the two arms fixed."""
        wp.launch(
            _lock_ik_q,
            self.ik_lock_indices.shape[0],
            [self.ik_q, self.ik_lock_indices, self.ik_lock_values],
            device=self.model.device,
        )

    def _root_to_tcp(self, side: str, root_transform: wp.transform) -> wp.transform:
        """Convert one hand-root pose to its W1 wrist TCP."""
        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = _quat_multiply(hand_rotation, wp.quat_inverse(J7_TO_HAND_BASE_ROTATIONS[side]))
        target_offset = TCP_OFFSET - J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _robot_hand_root(self, side: str, body_pose: np.ndarray) -> wp.transform:
        """Return the robot hand-root transform from its J7 pose."""
        wrist_position = wp.vec3(*body_pose[:3])
        wrist_rotation = wp.quat(*body_pose[3:7])
        return wp.transform(
            wrist_position + wp.quat_rotate(wrist_rotation, J7_TO_HAND_BASE_OFFSET),
            _quat_multiply(wrist_rotation, J7_TO_HAND_BASE_ROTATIONS[side]),
        )

    @staticmethod
    def _transform_error(actual: wp.transform, target: np.ndarray) -> tuple[float, float]:
        """Return translation [m] and shortest-angle [rad] errors."""
        actual_position = np.asarray(wp.transform_get_translation(actual), dtype=np.float64)
        position_error = float(np.linalg.norm(actual_position - target[:3]))
        actual_rotation = np.asarray(wp.transform_get_rotation(actual), dtype=np.float64)
        target_rotation = np.asarray(target[3:7], dtype=np.float64)
        actual_rotation /= max(float(np.linalg.norm(actual_rotation)), 1e-08)
        target_rotation /= max(float(np.linalg.norm(target_rotation)), 1e-08)
        cosine = float(np.clip(abs(actual_rotation @ target_rotation), 0.0, 1.0))
        return (position_error, 2.0 * math.acos(cosine))

    def _robot_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Lock non-arm IK degrees of freedom at their authored values."""
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        qd_start = self.ik_model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for joint, label in enumerate(self.ik_model.joint_label):
            if label not in controlled:
                lower[int(qd_start[joint])] = q[int(q_start[joint])] - 0.0001
                upper[int(qd_start[joint])] = q[int(q_start[joint])] + 0.0001
        return (lower, upper)

    def _locked_robot_q(self) -> tuple[wp.array[int], wp.array[float]]:
        """Return non-arm coordinates restored after every IK solve."""
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices = [
            int(q_start[joint]) for joint, label in enumerate(self.ik_model.joint_label) if label not in controlled
        ]
        return (
            wp.array(indices, dtype=wp.int32, device=self.model.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.model.device),
        )

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        """Return a body index from its unprefixed asset name."""
        return next((index for index, label in enumerate(labels) if label.endswith("/" + name)))

    @staticmethod
    def _quaternion_vector(value: wp.quat) -> wp.vec4:
        """Convert a quaternion to the IK rotation target type."""
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    def _build_right_stroke_trajectory(self) -> None:
        """Build tangential arcs over the nut's upper lateral surface."""
        initial_root = self.initial_hand_poses["RIGHT"]["position"].copy()
        stroke_end = RIGHT_SIDE_STROKE_END_ROOT.copy()
        stroke_starts = [
            RIGHT_SIDE_STROKE_START_ROOT + np.array((0.0, offset, 0.0), dtype=np.float32)
            for offset in RIGHT_STROKE_START_Y_OFFSETS
        ]
        stroke_start = stroke_starts[0]
        end_retracted = stroke_end + RIGHT_SIDE_RETRACT
        start_retracted = stroke_start + RIGHT_SIDE_RETRACT
        frame = 0
        self.right_root_waypoints = [(frame, initial_root)]
        frame += RIGHT_SETTLE_FRAMES
        self.right_root_waypoints.append((frame, initial_root))
        frame += RIGHT_APPROACH_FRAMES
        self.right_root_waypoints.append((frame, start_retracted))
        frame += RIGHT_LOWER_FRAMES
        self.right_root_waypoints.append((frame, stroke_start))
        frame += RIGHT_CONTACT_DWELL_FRAMES
        self.right_root_waypoints.append((frame, stroke_start))
        frame += RIGHT_CONTACT_SETTLE_FRAMES
        self.right_root_waypoints.append((frame, stroke_start))
        self.stroke_frame_ranges = []
        self.stroke_contact_waypoint_indices = [[] for _ in range(RIGHT_STROKE_COUNT)]
        self.stroke_contact_waypoint_indices[0].extend(((2, True), (3, False), (4, False)))
        self.stroke_contact_update_frames = {RIGHT_SETTLE_FRAMES + RIGHT_APPROACH_FRAMES: 0}
        for stroke in range(RIGHT_STROKE_COUNT):
            self.stroke_contact_waypoint_indices[stroke].append((len(self.right_root_waypoints) - 1, False))
            stroke_frame_start = frame
            frame += RIGHT_STROKE_FRAMES
            self.right_root_waypoints.append((frame, stroke_end))
            self.stroke_contact_waypoint_indices[stroke].append((len(self.right_root_waypoints) - 1, False))
            self.stroke_frame_ranges.append((stroke_frame_start, frame))
            frame += RIGHT_LIFT_FRAMES
            self.right_root_waypoints.append((frame, end_retracted))
            self.stroke_contact_waypoint_indices[stroke].append((len(self.right_root_waypoints) - 1, True))
            if stroke + 1 == RIGHT_STROKE_COUNT:
                break
            next_stroke_start = stroke_starts[stroke + 1]
            next_start_retracted = next_stroke_start + RIGHT_SIDE_RETRACT
            frame += RIGHT_RETURN_FRAMES
            self.right_root_waypoints.append((frame, next_start_retracted))
            self.stroke_contact_update_frames[frame] = stroke + 1
            self.stroke_contact_waypoint_indices[stroke + 1].append((len(self.right_root_waypoints) - 1, True))
            frame += RIGHT_LOWER_FRAMES
            self.right_root_waypoints.append((frame, next_stroke_start))
            self.stroke_contact_waypoint_indices[stroke + 1].append((len(self.right_root_waypoints) - 1, False))
            frame += RIGHT_CONTACT_DWELL_FRAMES
            self.right_root_waypoints.append((frame, next_stroke_start))
            self.stroke_contact_waypoint_indices[stroke + 1].append((len(self.right_root_waypoints) - 1, False))
        self.right_motion_end_frame = frame

    def _update_stroke_contact_pose(self, stroke: int) -> None:
        """Move one future stroke with the nut axis, without following its twist."""
        nut_transform = self.state_0.body_q.numpy()[self.nut_body]
        rotation = _quat_rotation_matrix(nut_transform[3:7])
        nut_axis_origin = nut_transform[:3] - rotation @ self._thread_nut_center_host
        face_shift = nut_axis_origin - self._initial_nut_axis_origin
        face_shift[0] += RIGHT_ADAPTIVE_CONTACT_CLEARANCE
        for waypoint, _is_retracted in self.stroke_contact_waypoint_indices[stroke]:
            frame, position = self.right_root_waypoints[waypoint]
            position = position.copy()
            position += face_shift
            self.right_root_waypoints[waypoint] = (frame, position)

    def _right_root_target(self, frame: int) -> np.ndarray:
        """Interpolate the scripted right-hand root at one display frame."""
        for (frame_0, position_0), (frame_1, position_1) in zip(
            self.right_root_waypoints[:-1], self.right_root_waypoints[1:], strict=True
        ):
            if frame <= frame_1:
                alpha = _smoothstep((frame - frame_0) / (frame_1 - frame_0))
                target = position_0 * (1.0 - alpha) + position_1 * alpha
                for stroke_start, stroke_end in self.stroke_frame_ranges:
                    if stroke_start <= frame <= stroke_end:
                        stroke_phase = (frame - stroke_start) / (stroke_end - stroke_start)
                        stroke_alpha = _smoothstep(stroke_phase)
                        target[2] += RIGHT_SIDE_STROKE_ARC_HEIGHT * math.sin(math.pi * stroke_alpha)
                        target[2] += RIGHT_SIDE_END_CLEARANCE * _smoothstep((stroke_phase - 0.84) / 0.16)
                        break
                return target
        return self.right_root_waypoints[-1][1].copy()

    def _update_hand_target(self) -> None:
        """Advance the middle-finger stroke while preserving the left grip."""
        targets = self._initial_hand_root_targets()
        right_position = self._right_root_target(self.frame_index)
        targets["RIGHT"] = wp.transform(right_position, self.initial_hand_poses["RIGHT"]["rotation"])
        self.current_hand_root_targets = targets
        self._update_robot_target(targets)
        closure = _smoothstep((self.frame_index - RIGHT_SETTLE_FRAMES) / RIGHT_APPROACH_FRAMES)
        target_pip = (
            self.initial_right_index_pip * (1.0 - closure) + math.radians(RIGHT_INDEX_PIP_STROKE_DEGREES) * closure
        )
        self.right_index_pip_values.assign([target_pip])
        wp.launch(
            _set_indexed_joint_q,
            dim=1,
            inputs=[self.right_index_pip_indices, self.right_index_pip_values],
            outputs=[self.hand_target_q],
            device=self.model.device,
        )

    def capture(self) -> None:
        """Capture one complete MJVBDV2 display frame on CUDA."""
        if not self.use_graph:
            return
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.simulate()
        self.graph = capture.graph
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)

    def simulate(self) -> None:
        """Advance one frame while interpolating the kinematic hands."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.hand_target_q)
        for substep in range(SIM_SUBSTEPS):
            alpha = (substep + 1) / SIM_SUBSTEPS
            wp.launch(
                _interpolate_joint_q,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_q_start, self.frame_q_end, alpha],
                outputs=[self.state_0.joint_q],
                device=self.model.device,
            )
            wp.launch(
                _update_joint_velocity,
                dim=self.model.joint_count,
                inputs=[
                    self.frame_q_start,
                    self.frame_q_end,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.frame_dt,
                ],
                outputs=[self.state_0.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            wp.launch(
                _apply_nut_angular_damping,
                dim=1,
                inputs=[self.state_0.body_qd, self.state_0.body_f, self.nut_body, NUT_ANGULAR_DAMPING],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = (self.state_1, self.state_0)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def step(self) -> None:
        """Advance the physical grip and repeated middle-finger strokes."""
        next_stroke = self.stroke_contact_update_frames.get(self.frame_index)
        if next_stroke is not None:
            self._update_stroke_contact_pose(next_stroke)
        self._update_hand_target()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frame_index += 1
        self.sim_time += self.frame_dt
        if self.test_mode:
            self._track_test_state()

    def _track_test_state(self) -> None:
        """Track stroke contacts, nut rotation, and left-grip stability."""
        count = min(int(self.contacts.rigid_contact_count.numpy()[0]), self.contacts.rigid_contact_max)
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        thread_contacts = np.count_nonzero(
            (shape0 == self.bolt_shape) & (shape1 == self.nut_thread_shape)
            | (shape0 == self.nut_thread_shape) & (shape1 == self.bolt_shape)
        )
        self.maximum_thread_contact_count = max(self.maximum_thread_contact_count, int(thread_contacts))
        left_hand_shapes = np.asarray(self.left_grip_shapes, dtype=np.int32)
        left_bolt_contacts = np.count_nonzero(
            (shape1 == self.bolt_shape) & np.isin(shape0, left_hand_shapes)
            | (shape0 == self.bolt_shape) & np.isin(shape1, left_hand_shapes)
        )
        self.maximum_left_hand_bolt_contact_count = max(
            self.maximum_left_hand_bolt_contact_count, int(left_bolt_contacts)
        )
        self.last_left_hand_bolt_contact_count = int(left_bolt_contacts)
        right_middle_shapes = np.asarray(self.right_middle_shapes, dtype=np.int32)
        middle_nut_contacts = np.count_nonzero(
            (shape1 == self.nut_shape) & np.isin(shape0, right_middle_shapes)
            | (shape0 == self.nut_shape) & np.isin(shape1, right_middle_shapes)
        )
        right_other_shapes = np.asarray(self.right_other_shapes, dtype=np.int32)
        other_nut_contacts = np.count_nonzero(
            (shape1 == self.nut_shape) & np.isin(shape0, right_other_shapes)
            | (shape0 == self.nut_shape) & np.isin(shape1, right_other_shapes)
        )
        self.maximum_other_right_hand_nut_contact_count = max(
            self.maximum_other_right_hand_nut_contact_count, int(other_nut_contacts)
        )
        simulated_frame = self.frame_index - 1
        for stroke, (frame_start, frame_end) in enumerate(self.stroke_frame_ranges):
            if frame_start <= simulated_frame < frame_end:
                self.stroke_middle_contact_counts[stroke] = max(
                    self.stroke_middle_contact_counts[stroke], int(middle_nut_contacts)
                )
                if middle_nut_contacts > 0:
                    self.stroke_middle_contact_frame_counts[stroke] += 1
        body_q = self.state_0.body_q.numpy()
        for side in ("LEFT", "RIGHT"):
            target = self.current_hand_root_targets[side]
            target_array = np.concatenate(
                (
                    np.asarray(wp.transform_get_translation(target), dtype=np.float32),
                    np.asarray(wp.transform_get_rotation(target), dtype=np.float32),
                )
            )
            position_error, angle_error = self._transform_error(
                self._robot_hand_root(side, body_q[self.robot_j7_bodies[side]]), target_array
            )
            self.maximum_robot_hand_position_error = max(self.maximum_robot_hand_position_error, position_error)
            self.maximum_robot_hand_angle_error = max(self.maximum_robot_hand_angle_error, angle_error)
        bolt_rotation_host = wp.quat(*body_q[self.bolt_body, 3:7])
        nut_rotation_host = wp.quat(*body_q[self.nut_body, 3:7])
        relative_rotation = np.asarray(
            tuple(_quat_multiply(wp.quat_inverse(bolt_rotation_host), nut_rotation_host)), dtype=np.float32
        )
        twist_angle = _signed_twist_angle(
            relative_rotation, self._thread_initial_relative_rotation_host, np.array((0.0, 0.0, 1.0), dtype=np.float32)
        )
        twist_step = (twist_angle - self._last_nut_twist_angle + math.pi) % (2.0 * math.pi) - math.pi
        self.accumulated_nut_rotation += twist_step
        self._last_nut_twist_angle = twist_angle
        for stroke, (frame_start, frame_end) in enumerate(self.stroke_frame_ranges):
            if simulated_frame == frame_start:
                self.stroke_rotation_starts[stroke] = self.accumulated_nut_rotation
            if simulated_frame == frame_end - 1:
                self.stroke_rotation_ends[stroke] = self.accumulated_nut_rotation
        bolt_transform = body_q[self.bolt_body]
        bolt_displacement = float(np.linalg.norm(bolt_transform[:3] - self.initial_bolt_transform[:3]))
        bolt_rotation_dot = min(abs(float(bolt_transform[3:7] @ self.initial_bolt_transform[3:7])), 1.0)
        bolt_rotation = 2.0 * math.acos(bolt_rotation_dot)
        self.maximum_bolt_displacement = max(self.maximum_bolt_displacement, bolt_displacement)
        self.maximum_bolt_rotation = max(self.maximum_bolt_rotation, bolt_rotation)
        bolt_rotation_matrix = _quat_rotation_matrix(bolt_transform[3:7])
        nut_rotation_matrix = _quat_rotation_matrix(body_q[self.nut_body, 3:7])
        bolt_axis_origin = bolt_transform[:3] - bolt_rotation_matrix @ self._thread_bolt_center_host
        nut_axis_origin = body_q[self.nut_body, :3] - nut_rotation_matrix @ self._thread_nut_center_host
        thread_axis = bolt_rotation_matrix @ np.array((0.0, 0.0, 1.0), dtype=np.float32)
        nut_bolt_delta = nut_axis_origin - bolt_axis_origin
        axial_separation = float(nut_bolt_delta @ thread_axis)
        nut_radial_offset = float(np.linalg.norm(nut_bolt_delta - thread_axis * axial_separation))
        self.maximum_nut_radial_offset = max(self.maximum_nut_radial_offset, nut_radial_offset)
        axial_motion = axial_separation - self.thread_initial_axial_offset
        self.maximum_nut_outward_motion = max(self.maximum_nut_outward_motion, axial_motion)
        self.minimum_nut_inward_motion = min(self.minimum_nut_inward_motion, axial_motion)
        bolt_tip = bolt_axis_origin + thread_axis * (BOLT_TIP_LOCAL_Z * ASSEMBLY_SCALE)
        nut_outer_face = nut_axis_origin + thread_axis * (NUT_OUTER_FACE_LOCAL_Z * ASSEMBLY_SCALE)
        self.bolt_tip_protrusion = float((bolt_tip - nut_outer_face) @ thread_axis)

    def render(self) -> None:
        """Render both hands, the threaded pair, and current contacts."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify lateral friction strokes and physical left-hand retention."""
        if self.solver.features.backend.value != "one_way_kinematic_full":
            raise ValueError(f"Unexpected solver backend: {self.solver.features.backend.value}")
        body_q = self.state_0.body_q.numpy()
        joint_q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(joint_q)):
            raise ValueError("Bimanual nut/bolt state is not finite")
        if self.frame_index < self.right_motion_end_frame:
            raise ValueError(
                f"Example stopped before all strokes completed: {self.frame_index} < {self.right_motion_end_frame}"
            )
        expected_prethread_advance = -PRETHREAD_REVOLUTIONS * M20_THREAD_PITCH
        entry_origin = ASSEMBLY_ORIGIN + ASSEMBLY_AXIS * NUT_START_OFFSET
        actual_prethread_advance = float((self.nut_origin - entry_origin) @ ASSEMBLY_AXIS)
        if not math.isclose(actual_prethread_advance, expected_prethread_advance, abs_tol=1e-07):
            raise ValueError(
                f"Unexpected initial thread advance: {actual_prethread_advance:.6f} m, expected {expected_prethread_advance:.6f} m"
            )
        initial_bolt_tip = ASSEMBLY_ORIGIN + ASSEMBLY_AXIS * (BOLT_TIP_LOCAL_Z * ASSEMBLY_SCALE)
        initial_nut_outer_face = self.nut_origin + ASSEMBLY_AXIS * (NUT_OUTER_FACE_LOCAL_Z * ASSEMBLY_SCALE)
        initial_flush_error = float((initial_nut_outer_face - initial_bolt_tip) @ ASSEMBLY_AXIS)
        if not math.isclose(initial_flush_error, -INITIAL_BOLT_TIP_PROTRUSION, abs_tol=1e-07):
            raise ValueError(
                f"Unexpected initial bolt-tip protrusion: {-initial_flush_error:.6f} m, expected {INITIAL_BOLT_TIP_PROTRUSION:.6f} m"
            )
        if self.maximum_nut_radial_offset > 0.002:
            raise ValueError(f"Nut left the thread axis: {self.maximum_nut_radial_offset:.6f} m")
        if self.maximum_nut_outward_motion > 0.002:
            raise ValueError(f"Nut backed out along the thread axis: {self.maximum_nut_outward_motion:.6f} m")
        if self.maximum_thread_contact_count <= 0:
            raise ValueError("No physical nut/bolt thread contact was observed")
        if self.maximum_left_hand_bolt_contact_count <= 0:
            raise ValueError("No left-hand/bolt contact was observed")
        if self.last_left_hand_bolt_contact_count <= 0:
            raise ValueError("The left hand was no longer contacting the bolt at the end")
        if any(count <= 0 for count in self.stroke_middle_contact_counts):
            raise ValueError(f"A middle-finger stroke missed the nut: {self.stroke_middle_contact_counts}")
        minimum_contact_frames = RIGHT_STROKE_FRAMES - 5
        if any(count < minimum_contact_frames for count in self.stroke_middle_contact_frame_counts):
            raise ValueError(
                f"A middle-finger stroke did not maintain face contact: {self.stroke_middle_contact_frame_counts} < {minimum_contact_frames}"
            )
        if self.maximum_other_right_hand_nut_contact_count != 0:
            raise ValueError(
                f"A right-hand shape other than the middle finger contacted the nut: {self.maximum_other_right_hand_nut_contact_count}"
            )
        if len(self.right_middle_shapes) != 1:
            raise ValueError(f"Expected one native right-middle collider, got {self.right_middle_shapes}")
        hand_shape_gaps = self.model.shape_gap.numpy()[self.hand_shapes["LEFT"] + self.hand_shapes["RIGHT"]]
        if np.any(hand_shape_gaps != 0.0):
            raise ValueError(f"Robot hand contact gaps must be zero, got {hand_shape_gaps}")
        tightening_stroke_rotations = [
            start - end
            for start, end in zip(self.stroke_rotation_starts, self.stroke_rotation_ends, strict=True)
            if start is not None and end is not None
        ]
        if len(tightening_stroke_rotations) != RIGHT_STROKE_COUNT or any(
            rotation < math.radians(MIN_STROKE_ROTATION_DEGREES) for rotation in tightening_stroke_rotations
        ):
            raise ValueError(
                f"A lateral friction stroke rotated the nut less than 60 degrees: {[math.degrees(rotation) for rotation in tightening_stroke_rotations]}"
            )
        if -self.accumulated_nut_rotation < math.radians(RIGHT_STROKE_COUNT * MIN_STROKE_ROTATION_DEGREES):
            raise ValueError(
                f"Full-angle friction strokes did not accumulate enough tightening: {math.degrees(-self.accumulated_nut_rotation):.3f} tightening degrees"
            )
        if self.bolt_tip_protrusion <= 0.0001:
            raise ValueError(f"Tightening did not expose the bolt tip: {self.bolt_tip_protrusion:.6f} m")
        expected_right = np.concatenate(
            (
                self.right_root_waypoints[-1][1],
                np.asarray(self.initial_hand_poses["RIGHT"]["rotation"], dtype=np.float32),
            )
        )
        right_position_error, right_angle_error = self._transform_error(
            self._robot_hand_root("RIGHT", body_q[self.robot_j7_bodies["RIGHT"]]), expected_right
        )
        if right_position_error > 0.002:
            raise ValueError(f"Right hand did not finish retracted above the nut: {right_position_error:.6f} m")
        if right_angle_error > math.radians(1.0):
            raise ValueError(
                f"Right hand did not retain the recorded stroke orientation: {math.degrees(right_angle_error):.3f} degrees"
            )
        if self.maximum_bolt_displacement > MAX_BOLT_GRIP_TRANSLATION or self.maximum_bolt_rotation > math.radians(
            MAX_BOLT_GRIP_ROTATION_DEGREES
        ):
            raise ValueError(
                f"Left hand did not retain the bolt: translation={self.maximum_bolt_displacement:.6f} m, rotation={math.degrees(self.maximum_bolt_rotation):.3f} degrees"
            )
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        shape_flags = self.model.shape_flags.numpy()
        if not shape_flags[self.nut_shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            raise ValueError("The nut contact envelope must participate in fingertip contact")
        if shape_flags[self.nut_shape] & int(newton.ShapeFlags.VISIBLE):
            raise ValueError("The simplified nut contact envelope must not replace the detailed visual mesh")
        if not shape_flags[self.nut_thread_shape] & int(newton.ShapeFlags.VISIBLE):
            raise ValueError("The detailed threaded nut mesh must remain visible")
        visual_shape_flags = self.model.shape_flags.numpy()[self.robot_visual_shapes]
        if np.any(visual_shape_flags & collision_mask):
            raise ValueError("Full-W1 visual shapes must not participate in contact")
        non_hand_shape_flags = self.model.shape_flags.numpy()[self.robot_non_hand_shapes]
        if np.any(non_hand_shape_flags & collision_mask):
            raise ValueError("Full-W1 shapes outside the hands must not participate in contact")
        if self.maximum_robot_hand_position_error > 0.002 or self.maximum_robot_hand_angle_error > math.radians(1.0):
            raise ValueError(
                f"Full-W1 hands did not follow their IK targets: translation={self.maximum_robot_hand_position_error:.6f} m, rotation={math.degrees(self.maximum_robot_hand_angle_error):.3f} degrees"
            )
        rigid_count = int(self.contacts.rigid_contact_count.numpy()[0])
        if rigid_count >= RIGID_CONTACT_MAX:
            raise ValueError(f"Rigid-contact capacity exhausted: {rigid_count} >= {RIGID_CONTACT_MAX}")


def main():
    """Run the standalone bimanual nut-and-bolt demo."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
