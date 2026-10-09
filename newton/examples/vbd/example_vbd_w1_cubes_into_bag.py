# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD W1 Cubes Into Bag
#
# A kinematic Dexforce W1 replays a recorded right-hand trajectory: it picks
# a tetrahedral soft cube from a table and drops it into a pinned cloth box
# bag, then picks a rigid cube and drops it into the same bag. SolverVBD
# integrates the bag, the soft cube, and the rigid cube together; the robot
# links are prescribed colliders driven by per-frame IK. Only collision meshes
# imported from the W1 URDF touch the objects.
#
# Command: python -m newton.examples vbd_w1_cubes_into_bag
#
###########################################################################

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
ROBOT_URDF = ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"
SOFT_GRASP_KEYFRAME = ASSET_ROOT / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
RIGID_GRASP_KEYFRAME = ASSET_ROOT / "vbd_mjvbd_v2" / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"

FPS = 60
SIM_SUBSTEPS = 8
IK_ITERATIONS = 8
INITIAL_IK_ITERATIONS = 240

# World frame of the recorded scene.
BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
BASE_ROT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)

TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]

CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
SOFT_CUBE_CENTRE = wp.vec3(-0.14931439, -2.76669516, TABLE_TOP_Z + CUBE_HALF_EXTENTS[2] + 0.001)
SOFT_CUBE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
SOFT_CUBE_DIMS = (6, 4, 6)
SOFT_CUBE_DENSITY = 100.0
SOFT_CUBE_K_MU = 3.0e5
SOFT_CUBE_K_LAMBDA = 1.0e6
SOFT_CUBE_K_DAMP = 15.0
SOFT_CUBE_PARTICLE_RADIUS = 0.0025

# The recorded rigid grasp used a cube at the soft cube's position; the
# sequential scene places it 11 cm further along -x.
RIGID_CUBE_OFFSET = wp.vec3(-0.11, 0.0, 0.0)
RIGID_CUBE_CENTRE = SOFT_CUBE_CENTRE + RIGID_CUBE_OFFSET
RIGID_CUBE_DENSITY = 1500.0
RIGID_CUBE_MARGIN = 0.0015

BAG_WIDTH = 0.20
BAG_DEPTH = 0.16
BAG_HEIGHT = 0.24
BAG_POS = wp.vec3(0.24068561, -2.79869516, 0.93122798)
BAG_RESOLUTION = 20
BAG_PARTICLE_RADIUS = 0.003
BAG_DENSITY = 0.08
BAG_TRI_KE = 1.5e2
BAG_TRI_KA = 1.5e2
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 0.5
BAG_EDGE_KD = 1.5e-5

# (ke [N/m], kd, mu) contact settings for each stage of the recorded task.
SOFT_FREE_CONTACT = (5.0e3, 5.0e-2, 0.25)
SOFT_GRASP_CONTACT = (1.5e4, 0.2, 6.0)
SOFT_GRASP_HAND_FRICTION = 40.0
RIGID_GRASP_CONTACT = (3.0e3, 1.0, 3.0e3)
RIGID_RELEASE_CONTACT = (5.0e3, 0.0, 0.0)
SOFT_CONTACT_GAP = 0.003
SOFT_CONTACT_MAX = 4096
RIGID_CONTACT_MAX = 2048

# Fixed mount from the W1 URDF: right_j7 -> right_ee -> right_hand_base.
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
RIGHT_J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
RIGHT_J7_TO_HAND_BASE_ROTATION = wp.quat(0.5, -0.5, 0.5, 0.5)

HAND_JOINTS = (
    "RIGHT_HAND_THUMB1",
    "RIGHT_HAND_THUMB2",
    "RIGHT_HAND_INDEX",
    "RIGHT_INDEX_PIP",
    "RIGHT_HAND_MIDDLE",
    "RIGHT_MIDDLE_PIP",
    "RIGHT_HAND_RING",
    "RIGHT_RING_PIP",
    "RIGHT_HAND_PINKY",
    "RIGHT_PINKY_PIP",
)
IDLE_JOINTS = {name: 90.0 if name == "RIGHT_HAND_THUMB2" else 0.0 for name in HAND_JOINTS}
OPEN_JOINTS = dict.fromkeys(HAND_JOINTS, 0.0)
# Recorded hand pose at the soft-cube pick point before the final closure,
# which is also the rigid pick's pre-grasp finger pose.
APPROACH_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
APPROACH_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}

LEFT_ARM = tuple(f"LEFT_J{i}" for i in range(1, 8))
RIGHT_ARM = tuple(f"RIGHT_J{i}" for i in range(1, 8))
HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")

SOFT_CUBE_COLOR = (0.2, 0.8, 0.2)
BAG_COLOR = (1.0, 1.0, 1.0)
RIGID_CUBE_COLOR = (0.90, 0.32, 0.18)
CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
CAMERA_PITCH = -18.0
CAMERA_YAW = 126.0


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _set_indexed(q: wp.array2d[float], indices: wp.array[wp.int32], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


_vec_hand = wp.types.vector(length=len(HAND_JOINTS), dtype=float)


@wp.kernel
def _write_frame_inputs(
    tcp: wp.transform,
    fingers: _vec_hand,
    tcp_target: wp.array[wp.transform],
    finger_q: wp.array[float],
):
    """Store this frame's sampled right-hand targets in persistent device buffers."""
    tcp_target[0] = tcp
    for i in range(finger_q.shape[0]):
        finger_q[i] = fingers[i]


@wp.kernel
def _unpack_tcp_target(
    tcp_target: wp.array[wp.transform],
    position_target: wp.array[wp.vec3],
    rotation_target: wp.array[wp.vec4],
):
    """Feed the stored right TCP target to its IK position and rotation objectives."""
    tf = tcp_target[0]
    q = wp.transform_get_rotation(tf)
    position_target[0] = wp.transform_get_translation(tf)
    rotation_target[0] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def _hold_pinned(
    indices: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    q0: wp.array[wp.vec3],
    q1: wp.array[wp.vec3],
):
    i = wp.tid()
    q0[indices[i]] = positions[i]
    q1[indices[i]] = positions[i]


def _generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
    """Generate a merged five-face box mesh with an open top."""
    cell_x = 2.0 * half_x / resolution
    cell_y = 2.0 * half_y / resolution
    cell_z = height / resolution
    vertex_map = {}
    vertices = []
    indices = []

    def vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    def quad(v00, v10, v01, v11):
        indices.extend((v00, v10, v01, v10, v11, v01))

    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            y0 = -half_y + j * cell_y
            quad(
                vertex(x0, y0, 0.0),
                vertex(x0 + cell_x, y0, 0.0),
                vertex(x0, y0 + cell_y, 0.0),
                vertex(x0 + cell_x, y0 + cell_y, 0.0),
            )
    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            x1 = x0 + cell_x
            y0 = -half_y + i * cell_y
            y1 = y0 + cell_y
            z0 = j * cell_z
            z1 = z0 + cell_z
            quad(vertex(x0, -half_y, z0), vertex(x1, -half_y, z0), vertex(x0, -half_y, z1), vertex(x1, -half_y, z1))
            quad(vertex(x1, half_y, z0), vertex(x0, half_y, z0), vertex(x1, half_y, z1), vertex(x0, half_y, z1))
            quad(vertex(-half_x, y1, z0), vertex(-half_x, y0, z0), vertex(-half_x, y1, z1), vertex(-half_x, y0, z1))
            quad(vertex(half_x, y0, z0), vertex(half_x, y1, z0), vertex(half_x, y0, z1), vertex(half_x, y1, z1))
    return np.asarray(vertices, dtype=np.float32), indices


def _load_keyframe(path: Path) -> tuple[wp.transform, dict[str, float], dict]:
    """Load a recorded right-hand root pose and finger angles [deg]."""
    if not path.is_file():
        raise FileNotFoundError(f"Recorded hand keyframe not found: {path}")
    keyframe = json.loads(path.read_text(encoding="utf-8"))["keyframe"]
    root = keyframe["target_root_pose"]
    joints = {name: float(value) for name, value in keyframe["target_finger_joints_degrees"].items()}
    missing = set(HAND_JOINTS) - joints.keys()
    if missing:
        raise ValueError(f"Keyframe {path} is missing finger joints {sorted(missing)}")
    return wp.transform(wp.vec3(*root["position_m"]), wp.quat(*root["quaternion_xyzw"])), joints, keyframe


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float) -> wp.transform:
    pa, pb = wp.transform_get_translation(a), wp.transform_get_translation(b)
    return wp.transform(
        pa + (pb - pa) * alpha, wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), alpha)
    )


def _offset(
    transform: wp.transform, dz: float = 0.0, x: float | None = None, y: float | None = None, z: float | None = None
):
    p = wp.transform_get_translation(transform)
    return wp.transform(
        wp.vec3(p[0] if x is None else x, p[1] if y is None else y, (p[2] if z is None else z) + dz),
        wp.transform_get_rotation(transform),
    )


def _root_to_tcp(root: wp.transform) -> wp.transform:
    """Convert a recorded ``right_hand_base`` pose to the W1 right-arm IK target."""
    wrist_rotation = wp.transform_get_rotation(root) * wp.quat_inverse(RIGHT_J7_TO_HAND_BASE_ROTATION)
    target = wp.transform_get_translation(root) + wp.quat_rotate(
        wrist_rotation, TCP_OFFSET - RIGHT_J7_TO_HAND_BASE_OFFSET
    )
    return wp.transform(target, wrist_rotation)


def build_trajectory(soft_grasp_joints, rigid_approach, rigid_grasp_joints):
    """Build the recorded ``(duration, root_a, root_b, joints_a, joints_b, phase)`` segments."""
    soft_approach = APPROACH_ROOT
    release_z = float(BAG_POS[2]) + BAG_HEIGHT + 0.06

    def bag_hover(approach, cube_centre):
        offset = wp.transform_get_translation(approach) - cube_centre
        return _offset(
            approach,
            x=float(BAG_POS[0]) + float(offset[0]),
            y=float(BAG_POS[1]) + float(offset[1]),
            z=release_z + float(offset[2]),
        )

    soft_hover = bag_hover(soft_approach, SOFT_CUBE_CENTRE)
    rigid_hover = bag_hover(rigid_approach, RIGID_CUBE_CENTRE)
    soft_lift = _offset(soft_approach, 0.07)
    rigid_lift = _offset(rigid_approach, 0.10)
    rigid_transport = _offset(rigid_hover, 0.05)
    soft_retreat = _offset(soft_hover, 0.10)
    rigid_retreat = _offset(rigid_hover, 0.12)
    soft, rigid, pre = soft_grasp_joints, rigid_grasp_joints, APPROACH_JOINTS
    return (
        (0.50, soft_approach, soft_approach, IDLE_JOINTS, IDLE_JOINTS, "soft_wait"),
        (1.50, soft_approach, soft_approach, IDLE_JOINTS, pre, "soft_prepare"),
        (0.50, soft_approach, soft_approach, pre, pre, "soft_prepare"),
        (1.80, soft_approach, soft_approach, pre, soft, "soft_grasp"),
        (0.60, soft_approach, soft_approach, soft, soft, "soft_carry"),
        (1.20, soft_approach, soft_lift, soft, soft, "soft_carry"),
        (7.00, soft_lift, soft_hover, soft, soft, "soft_carry"),
        (0.40, soft_hover, soft_hover, soft, soft, "soft_carry"),
        (0.25, soft_hover, soft_hover, soft, OPEN_JOINTS, "soft_release"),
        (0.90, soft_hover, soft_hover, OPEN_JOINTS, OPEN_JOINTS, "soft_release"),
        (1.00, soft_hover, soft_retreat, OPEN_JOINTS, OPEN_JOINTS, "soft_release"),
        (1.50, soft_retreat, rigid_approach, OPEN_JOINTS, IDLE_JOINTS, "rigid_move"),
        (0.50, rigid_approach, rigid_approach, IDLE_JOINTS, IDLE_JOINTS, "rigid_prepare"),
        (1.50, rigid_approach, rigid_approach, IDLE_JOINTS, pre, "rigid_prepare"),
        (0.50, rigid_approach, rigid_approach, pre, pre, "rigid_prepare"),
        (0.45, rigid_approach, rigid_approach, pre, rigid, "rigid_grasp"),
        (0.30, rigid_approach, rigid_approach, rigid, rigid, "rigid_carry"),
        (0.75, rigid_approach, rigid_lift, rigid, rigid, "rigid_carry"),
        (5.00, rigid_lift, rigid_transport, rigid, rigid, "rigid_carry"),
        (1.20, rigid_transport, rigid_hover, rigid, rigid, "rigid_carry"),
        (0.50, rigid_hover, rigid_hover, rigid, rigid, "rigid_carry"),
        (0.80, rigid_hover, rigid_hover, rigid, OPEN_JOINTS, "rigid_release"),
        (1.50, rigid_hover, rigid_hover, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
        (1.00, rigid_hover, rigid_retreat, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
    )


def sample_trajectory(segments, time_s: float):
    """Return the hand root, finger angles [deg], and phase at ``time_s`` [s]."""
    for duration, root_a, root_b, joints_a, joints_b, phase in segments:
        if time_s <= duration:
            alpha = _smoothstep(time_s / duration)
            root = _lerp_transform(root_a, root_b, alpha)
            joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in HAND_JOINTS}
            return root, joints, phase
        time_s -= duration
    _, _, root, _, joints, phase = segments[-1]
    return root, joints, phase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0

        _, soft_grasp_joints, _ = _load_keyframe(Path(args.soft_grasp_keyframe))
        rigid_root, rigid_grasp_joints, rigid_keyframe = _load_keyframe(Path(args.rigid_grasp_keyframe))
        if "rigid_cube_pose" not in rigid_keyframe:
            raise ValueError("The rigid grasp keyframe must record the cube pose")
        rigid_approach = wp.transform(
            wp.transform_get_translation(rigid_root) + RIGID_CUBE_OFFSET, wp.transform_get_rotation(rigid_root)
        )
        self.segments = build_trajectory(soft_grasp_joints, rigid_approach, rigid_grasp_joints)
        self.script_duration = sum(segment[0] for segment in self.segments)

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Pin the bag's top rim.
        flags = self.model.particle_flags.numpy()
        flags[self.bag_top_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        self.bag_pinned = wp.array(self.bag_top_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_positions = wp.array(
            self.model.particle_q.numpy()[self.bag_top_indices], dtype=wp.vec3, device=self.device
        )

        self._build_ik()
        self._initialize_pose()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            include_static_kinematic_pairs=False,
            soft_contact_gap=SOFT_CONTACT_GAP,
            enable_rigid_soft_full_surface_contact=True,
            contact_matching="latest",
            soft_contact_max=SOFT_CONTACT_MAX,
            rigid_contact_max=RIGID_CONTACT_MAX,
        )
        self.contacts = self.collision_pipeline.contacts()
        contact_radius = max(BAG_PARTICLE_RADIUS, SOFT_CUBE_PARTICLE_RADIUS)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=4096,
            rigid_body_particle_contact_buffer_size=4096,
            particle_enable_self_contact=True,
            # Interaction distance plus extra detection reach (query radius = 2 * radius).
            particle_self_contact_margin=contact_radius,
            particle_self_contact_gap=contact_radius,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=256,
            # One self-contact detection per step: the bag moves slowly enough that
            # re-detecting after the inertial prediction changes nothing visible.
            collision_frequency_type={
                newton.solvers.SolverBase.CollisionSlot.SOFT_SELF_CONTACT: newton.solvers.SolverBase.CollisionFrequencyType.PRE_INIT,
            },
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.03,
        )

        triangles = self.model.tri_indices.numpy().reshape((-1, 3))

        def triangle_subset(begin, end):
            mask = np.all((triangles >= begin) & (triangles < end), axis=1)
            return wp.array(triangles[mask].reshape(-1).astype(np.int32), dtype=wp.int32, device=self.device)

        self.bag_triangles = triangle_subset(self.bag_particle_start, self.bag_particle_end)
        self.soft_cube_triangles = triangle_subset(self.soft_cube_particle_start, self.soft_cube_particle_end)

        self.hand_shape_collision = True
        self.hand_particle_collision = True
        self.contact_phase = None
        self.object_released = np.zeros(2, dtype=bool)
        self._apply_contact_phase("soft_wait")

        # Soft-contact material values are kernel arguments, so each setting
        # gets its own captured frame graph. Shape materials and flags are
        # device arrays and update inside a graph.
        self.use_graph = self.device.is_cuda and not args.no_cuda_graph
        self.graphs = {}

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)

    # Scene -------------------------------------------------------------------

    def _build_scene(self):
        if not ROBOT_URDF.is_file():
            raise FileNotFoundError(f"Dexforce W1 URDF not found: {ROBOT_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = 2.0e5
        builder.default_shape_cfg.kd = 1.0e-4
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.configure_sdf(force_sdf=True)

        builder.add_urdf(
            str(ROBOT_URDF),
            xform=wp.transform(BASE_POS, BASE_ROT),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        # Every finger joint, including the URDF's mimic followers, is commanded
        # directly from the keyframes, so the solver's mimic projection has
        # nothing to do on these kinematic links.
        for joint in range(self.robot_joint_end):
            builder.joint_mimic_joint[joint] = -1

        table_cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9)
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POS, BASE_ROT),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="table",
        )
        builder.add_ground_plane(height=float(BASE_POS[2]), label="ground")

        bag_vertices, bag_indices = _generate_box_bag(0.5 * BAG_WIDTH, 0.5 * BAG_DEPTH, BAG_HEIGHT, BAG_RESOLUTION)
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=BAG_POS,
            rot=BASE_ROT,
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=BAG_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
            label="box_bag",
        )
        self.bag_particle_end = builder.particle_count
        top = np.flatnonzero(np.abs(bag_vertices[:, 2] - BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = (top + self.bag_particle_start).astype(np.int32)

        half = wp.vec3(*CUBE_HALF_EXTENTS)
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=SOFT_CUBE_CENTRE - wp.quat_rotate(SOFT_CUBE_ROTATION, half),
            rot=SOFT_CUBE_ROTATION,
            vel=wp.vec3(),
            dim_x=SOFT_CUBE_DIMS[0],
            dim_y=SOFT_CUBE_DIMS[1],
            dim_z=SOFT_CUBE_DIMS[2],
            cell_x=2.0 * CUBE_HALF_EXTENTS[0] / SOFT_CUBE_DIMS[0],
            cell_y=2.0 * CUBE_HALF_EXTENTS[1] / SOFT_CUBE_DIMS[1],
            cell_z=2.0 * CUBE_HALF_EXTENTS[2] / SOFT_CUBE_DIMS[2],
            density=SOFT_CUBE_DENSITY,
            k_mu=SOFT_CUBE_K_MU,
            k_lambda=SOFT_CUBE_K_LAMBDA,
            k_damp=SOFT_CUBE_K_DAMP,
            particle_radius=SOFT_CUBE_PARTICLE_RADIUS,
            label="soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count

        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=RIGID_CUBE_DENSITY,
            ke=RIGID_GRASP_CONTACT[0],
            kd=RIGID_GRASP_CONTACT[1],
            mu=RIGID_GRASP_CONTACT[2],
            margin=RIGID_CUBE_MARGIN,
        )
        rigid_cfg.configure_sdf(force_sdf=True)
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(RIGID_CUBE_CENTRE, wp.quat_identity()), label="rigid_cube"
        )
        self.rigid_cube_shape = builder.add_shape_box(
            self.rigid_cube_body,
            hx=CUBE_HALF_EXTENTS[0],
            hy=CUBE_HALF_EXTENTS[1],
            hz=CUBE_HALF_EXTENTS[2],
            cfg=rigid_cfg,
            color=RIGID_CUBE_COLOR,
            label="rigid_cube_shape",
        )

        # Only the URDF hand colliders touch objects; the rest of the robot is visual.
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        mask = collide_shapes | collide_particles
        self.hand_shapes, self.right_hand_shapes, self.robot_visual_shapes = [], [], []
        for shape in range(robot_shape_end):
            is_collider = bool(builder.shape_flags[shape] & mask)
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            is_hand = any(side in label for side in ("left", "right")) and any(
                word in label for word in HAND_CONTACT_KEYWORDS
            )
            if not is_collider:
                self.robot_visual_shapes.append(shape)
            if is_hand and is_collider:
                self.hand_shapes.append(shape)
                if "right" in label:
                    self.right_hand_shapes.append(shape)
                builder.shape_flags[shape] |= mask
            else:
                builder.shape_flags[shape] &= ~mask
        for shape in range(robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= mask

        builder.color(include_bending=True)
        # The rigid cube is the only body VBD integrates and it has no joints, so
        # one body color is a valid ordering; the kinematic robot links are never
        # solved. This halves the per-iteration rigid launches.
        builder.body_color_groups = [np.arange(builder.body_count, dtype=np.int32)]
        self.model = builder.finalize(requires_grad=False)
        shape_mu = self.model.shape_material_mu.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_mu[self.hand_shapes] = SOFT_GRASP_HAND_FRICTION
        shape_kd[self.hand_shapes] = SOFT_GRASP_CONTACT[1]
        shape_ke[self.right_hand_shapes] = SOFT_GRASP_CONTACT[0]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_ke.assign(shape_ke)

    # Robot control -------------------------------------------------------------

    def _joint_index(self, name):
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels, name):
        return next(i for i, label in enumerate(labels) if label.endswith("/" + name))

    def _tcp(self, state, body):
        tf = wp.transform(*state.body_q.numpy()[body])
        rot = wp.transform_get_rotation(tf)
        return wp.transform(wp.transform_get_translation(tf) + wp.quat_rotate(rot, TCP_OFFSET), rot)

    def _build_ik(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.add_urdf(
            str(ROBOT_URDF),
            xform=wp.transform(BASE_POS, BASE_ROT),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = builder.finalize(device=self.device)
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        self.left_home = self._tcp(state, self._body_index(self.model.body_label, "left_j7"))
        right_home = self._tcp(state, self._body_index(self.model.body_label, "right_j7"))

        def position(body_name, target):
            return ik.IKObjectivePosition(
                self._body_index(self.ik_model.body_label, body_name),
                TCP_OFFSET,
                wp.array([wp.transform_get_translation(target)], dtype=wp.vec3, device=self.device),
            )

        def rotation(body_name, target):
            q = wp.transform_get_rotation(target)
            return ik.IKObjectiveRotation(
                self._body_index(self.ik_model.body_label, body_name),
                wp.quat_identity(),
                wp.array([wp.vec4(q[0], q[1], q[2], q[3])], dtype=wp.vec4, device=self.device),
            )

        self.left_obj, self.left_rot = position("left_j7", self.left_home), rotation("left_j7", self.left_home)
        self.right_obj, self.right_rot = position("right_j7", right_home), rotation("right_j7", right_home)

        # Only the two arms move in IK; the mask gives every other coordinate an
        # exactly-zero update, so the arm joints keep their real limits.
        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        dofs = self.ik_model.joint_dof_count
        dof_mask = np.zeros(dofs, dtype=bool)
        controlled = {*LEFT_ARM, *RIGHT_ARM}
        locked = []
        for joint in range(self.ik_model.joint_count):
            if self.model.joint_label[joint].rsplit("/", 1)[-1] in controlled:
                dof_mask[qd_start[joint] : qd_start[joint + 1]] = True
            else:
                locked.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        limits = ik.IKObjectiveJointLimit(
            wp.array(self.ik_model.joint_limit_lower.numpy()[:dofs], dtype=wp.float32, device=self.device),
            wp.array(self.ik_model.joint_limit_upper.numpy()[:dofs], dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            joint_dof_mask=wp.array(dof_mask, dtype=wp.bool, device=self.device),
        )
        self.lock_indices = wp.array(locked, dtype=wp.int32, device=self.device)
        self.lock_values = wp.array(q[locked], dtype=float, device=self.device)
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.hand_indices = wp.array(
            [int(q_start[self._joint_index(name)]) for name in HAND_JOINTS], dtype=wp.int32, device=self.device
        )
        # Per-frame IK runs inside the captured frame graph; the host only
        # writes the sampled targets into these buffers.
        self.finger_q = wp.zeros(len(HAND_JOINTS), dtype=float, device=self.device)
        self.tcp_target = wp.zeros(1, dtype=wp.transform, device=self.device)
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)

    def _set_arm_targets(self, right: wp.transform):
        for obj, rot, target in (
            (self.left_obj, self.left_rot, self.left_home),
            (self.right_obj, self.right_rot, right),
        ):
            q = wp.transform_get_rotation(target)
            obj.set_target_position(0, wp.transform_get_translation(target))
            rot.set_target_rotation(0, wp.vec4(q[0], q[1], q[2], q[3]))

    def _solve_ik(self, iterations):
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        wp.launch(_set_indexed, self.lock_indices.shape[0], [self.ik_q, self.lock_indices, self.lock_values])

    def _initialize_pose(self):
        root, joints, _ = sample_trajectory(self.segments, 0.0)
        self._set_arm_targets(_root_to_tcp(root))
        self._solve_ik(INITIAL_IK_ITERATIONS)
        q = self.model.joint_q.numpy()
        q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        q[self.hand_indices.numpy()] = np.radians([joints[name] for name in HAND_JOINTS])
        self.model.joint_q.assign(q)
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(q)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        wp.copy(self.frame_q_end, self.model.joint_q)

    def _prepare_frame(self):
        """Sample this frame's targets on the host and apply contact-phase changes."""
        script_time = (self.frame_index + 1) * self.frame_dt * self.args.trajectory_time_scale
        root, joints, phase = sample_trajectory(self.segments, script_time)
        fingers = _vec_hand(*np.radians([joints[name] for name in HAND_JOINTS]))
        wp.launch(_write_frame_inputs, 1, [_root_to_tcp(root), fingers, self.tcp_target, self.finger_q])
        if phase != self.contact_phase:
            self._apply_contact_phase(phase)
        if phase == "soft_release":
            self.object_released[0] = True
        elif phase == "rigid_release":
            self.object_released[1] = True

    # Contact phases ------------------------------------------------------------

    def _set_shape_material(self, shapes, ke=None, kd=None, mu=None, margin=None):
        for array, value in (
            (self.model.shape_material_ke, ke),
            (self.model.shape_material_kd, kd),
            (self.model.shape_material_mu, mu),
            (self.model.shape_margin, margin),
        ):
            if value is not None:
                values = array.numpy()
                values[shapes] = value
                array.assign(values)

    def _set_hand_flag(self, flag, enabled):
        flags = self.model.shape_flags.numpy()
        if enabled:
            flags[self.right_hand_shapes] |= int(flag)
        else:
            flags[self.right_hand_shapes] &= ~int(flag)
        self.model.shape_flags.assign(flags)

    def _apply_contact_phase(self, phase):
        """Switch the hand and object contact settings for one recorded stage."""
        hand_shapes = phase != "rigid_move"
        if hand_shapes != self.hand_shape_collision:
            self._set_hand_flag(newton.ShapeFlags.COLLIDE_SHAPES, hand_shapes)
            self.hand_shape_collision = hand_shapes
        hand_particles = phase in {"soft_prepare", "soft_grasp", "soft_carry", "soft_release"}
        if hand_particles != self.hand_particle_collision:
            self._set_hand_flag(newton.ShapeFlags.COLLIDE_PARTICLES, hand_particles)
            self.hand_particle_collision = hand_particles

        rigid_shapes = [*self.right_hand_shapes, self.rigid_cube_shape]
        if phase in {"soft_prepare", "soft_grasp", "soft_carry"}:
            self._set_shape_material(self.right_hand_shapes, *SOFT_GRASP_CONTACT[:2], SOFT_GRASP_HAND_FRICTION)
            soft = SOFT_GRASP_CONTACT
        elif phase in {"rigid_prepare", "rigid_grasp", "rigid_carry"}:
            self._set_shape_material(rigid_shapes, *RIGID_GRASP_CONTACT, RIGID_CUBE_MARGIN)
            soft = SOFT_FREE_CONTACT
        elif phase == "rigid_release":
            self._set_shape_material(rigid_shapes, *RIGID_RELEASE_CONTACT, RIGID_CUBE_MARGIN)
            soft = RIGID_RELEASE_CONTACT
        else:  # soft_wait, soft_release, rigid_move
            self._set_shape_material(self.right_hand_shapes, mu=0.0)
            soft = SOFT_FREE_CONTACT
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = soft
        self.contact_phase = phase

    # Simulation ----------------------------------------------------------------

    def _solve_frame_ik(self):
        """Solve this frame's arm IK on the device and assemble the frame's joint targets."""
        wp.launch(
            _unpack_tcp_target,
            1,
            [self.tcp_target, self.right_obj.target_positions, self.right_rot.target_rotations],
        )
        self._solve_ik(IK_ITERATIONS)
        wp.copy(self.frame_q_start, self.frame_q_end)
        wp.copy(self.frame_q_end, self.ik_q.flatten(), count=self.ik_model.joint_coord_count)
        wp.launch(
            _set_indexed,
            self.hand_indices.shape[0],
            [self.frame_q_end.reshape((1, -1)), self.hand_indices, self.finger_q],
        )

    def _simulate_frame(self):
        self._solve_frame_ik()
        self._simulate_substeps()

    def _simulate_substeps(self):
        for substep in range(self.sim_substeps):
            wp.launch(
                _hold_pinned,
                self.bag_pinned.shape[0],
                [self.bag_pinned, self.bag_pinned_positions, self.state_0.particle_q, self.state_1.particle_q],
            )
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.ik_model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
            )
            wp.launch(
                _joint_velocity,
                self.ik_model.joint_dof_count,
                [self.frame_q_start, self.frame_q_end, 1.0 / self.frame_dt, self.state_0.joint_qd],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._prepare_frame()
        if not self.use_graph:
            self._simulate_frame()
        else:
            key = (self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu)
            graph = self.graphs.get(key)
            if graph is None:
                # Warm up uncaptured once so lazily sized buffers exist, then capture.
                self._simulate_frame()
                saved = [self.model.state(), self.model.state()]
                saved[0].assign(self.state_0)
                saved[1].assign(self.state_1)
                saved_ik = [wp.clone(a) for a in (self.ik_q, self.frame_q_start, self.frame_q_end)]
                with wp.ScopedCapture() as capture:
                    self._simulate_frame()
                self.state_0.assign(saved[0])
                self.state_1.assign(saved[1])
                for array, value in zip((self.ik_q, self.frame_q_start, self.frame_q_end), saved_ik, strict=True):
                    array.assign(value)
                self.graphs[key] = capture.graph
            else:
                wp.capture_launch(graph)
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/box_bag", self.state_0.particle_q, self.bag_triangles, backface_culling=False, color=BAG_COLOR
        )
        self.viewer.log_mesh(
            "/soft_cube",
            self.state_0.particle_q,
            self.soft_cube_triangles,
            backface_culling=False,
            color=SOFT_CUBE_COLOR,
        )
        self.viewer.end_frame()

    # Validation ----------------------------------------------------------------

    @staticmethod
    def _scene_xy(points):
        """Express world points in the bag's frame (rotated with the robot base)."""
        relative = np.asarray(points, dtype=np.float64) - np.asarray(BAG_POS, dtype=np.float64)
        # BASE_ROT is +90 degrees about z: scene x = world y, scene y = -world x.
        return np.stack((relative[..., 1], -relative[..., 0], relative[..., 2]), axis=-1)

    def test_post_step(self):
        for array in (self.state_0.particle_q, self.state_0.body_q):
            if not np.all(np.isfinite(array.numpy())):
                raise ValueError("Non-finite simulation state")

    def test_final(self):
        self.test_post_step()
        mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if np.any(self.model.shape_flags.numpy()[self.robot_visual_shapes] & mask):
            raise ValueError("Robot visual shapes must remain non-colliding")
        if int(self.model.body_flags.numpy()[self.rigid_cube_body]) & int(newton.BodyFlags.KINEMATIC):
            raise ValueError("The rigid cube must remain dynamic")
        script_frames = int(np.ceil(self.script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        if self.frame_index < script_frames:
            return
        if not np.all(self.object_released):
            raise ValueError(f"Not every object was released: {self.object_released.tolist()}")

        particles = self.state_0.particle_q.numpy()
        bag = self._scene_xy(particles[self.bag_particle_start : self.bag_particle_end])
        soft = self._scene_xy(particles[self.soft_cube_particle_start : self.soft_cube_particle_end])
        rigid = self._scene_xy(self.state_0.body_q.numpy()[self.rigid_cube_body, :3])
        half = np.array((0.5 * BAG_WIDTH, 0.5 * BAG_DEPTH))
        bag_floor = float(bag[:, 2].min())
        top = TABLE_TOP_Z - float(BAG_POS[2])
        soft_inside = (
            np.all(soft[:, :2].min(axis=0) > -half - 0.02)
            and np.all(soft[:, :2].max(axis=0) < half + 0.02)
            and soft[:, 2].min() > bag_floor - 0.08
            and soft[:, 2].max() < top + 0.08
        )
        if not soft_inside:
            raise ValueError(f"The soft cube did not settle in the bag: centre {soft.mean(axis=0)} (bag frame)")
        rigid_inside = (
            np.all(np.abs(rigid[:2]) < half + np.array(CUBE_HALF_EXTENTS[:2]))
            and bag_floor - CUBE_HALF_EXTENTS[2] < rigid[2] < top + 0.08
        )
        if not rigid_inside:
            raise ValueError(f"The rigid cube did not settle in the bag: {rigid} (bag frame)")
        print(
            f"[W1CubesIntoBag] PASS: soft cube centre {np.round(soft.mean(axis=0), 3).tolist()}, "
            f"rigid cube {np.round(rigid, 3).tolist()} in the bag frame (bag floor z={bag_floor:.3f} m).",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1900)
        parser.add_argument("--vbd-iterations", type=int, default=12, help="VBD iterations per substep.")
        parser.add_argument("--trajectory-time-scale", type=float, default=1.0)
        parser.add_argument("--soft-grasp-keyframe", default=str(SOFT_GRASP_KEYFRAME))
        parser.add_argument("--rigid-grasp-keyframe", default=str(RIGID_GRASP_KEYFRAME))
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
