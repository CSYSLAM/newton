# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Insert a rigid plug with realtime Dexforce W1 IK and MJVBDV2.

The right hand approaches, pinches, raises, and inserts a rigid plug into a
static socket, then opens and withdraws to test whether the socket alone retains
the plug. Newton IK solves the arm every displayed frame; no baked joint
trajectory is replayed. The standalone plug moves only through gravity and
physical contact with the hand, table, and socket. The scene deliberately has
no cable, gravity cancellation, pose locking, hidden attachment force,
collision switching, or direct plug state correction.

Run, from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_dexforce_realtime_plug_socket
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBDV2

FPS = 60
DEFAULT_NUM_FRAMES = 660
DEFAULT_SUBSTEPS = 8
DEFAULT_VBD_ITERATIONS = 16
DEFAULT_IK_ITERATIONS = 32
INITIAL_IK_ITERATIONS = 300

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_ROBOT_URDF = ASSET_ROOT / "W1-hand-obj" / "DexforceW1V021_visual_collision.urdf"
DEFAULT_PLUG_MESH = ASSET_ROOT / "waic_plug_socket" / "plug0630.obj"
DEFAULT_SOCKET_MESH = ASSET_ROOT / "waic_plug_socket" / "socket0624.obj"

ROBOT_BASE_POSITION = wp.vec3(-0.1, -0.5, 0.0)
ROBOT_BASE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi)

SOCKET_POSITION = wp.vec3(0.0, 0.0, 1.0)
SOCKET_ROTATION = wp.quat_identity()
PLUG_REST_POSITION = wp.vec3(0.05, 0.0, 0.876)
PLUG_INITIAL_POSITION = PLUG_REST_POSITION
PLUG_FORWARD_POSITION = wp.vec3(0.05, 0.0, 1.0)
PLUG_INSERTED_POSITION = wp.vec3(-0.01, 0.0, 1.0)
PLUG_TO_GRIP = wp.vec3(0.015, 0.01, 0.0)
HAND_CARRY_CORRECTION = wp.vec3(-0.00055, 0.00206, 0.00009)
HAND_STANDBY_POSITION = wp.vec3(0.18, -0.10, 1.10)
HAND_TOP_OFFSET = wp.vec3(0.0, 0.0, 0.14)
HAND_TARGET_ROTATION = wp.quat(-0.000413, 0.952521, -0.304302, -0.010154)
HAND_INSERT_ROTATION = wp.quat(-0.012441, 0.974554, -0.220568, -0.037934)

TABLE_POSITION = wp.vec3(0.05, 0.13, 0.835)
TABLE_HALF_EXTENTS = wp.vec3(0.28, 0.18, 0.01)
PLUG_MASS = 0.2
PLUG_DENSITY = 1408.43
PLUG_COLOR = (0.10, 0.16, 0.20)
SOCKET_COLOR = (0.68, 0.72, 0.76)
TABLE_COLOR = (0.42, 0.49, 0.56)
GROUND_COLOR = (0.22, 0.25, 0.29)

SETTLE_SECONDS = 0.5
APPROACH_SECONDS = 1.4
DESCEND_SECONDS = 1.0
GRASP_SECONDS = 0.8
POST_GRASP_HOLD_SECONDS = 0.4
RAISE_SECONDS = 1.2
ALIGN_SECONDS = 0.6
INSERT_SECONDS = 1.5
INSERT_SETTLE_SECONDS = 0.3
RELEASE_SECONDS = 0.8
RETRACT_SECONDS = 1.2
HOLD_SECONDS = 1.0

CONTACT_KE = 2.0e5
CONTACT_KD = 0.5
CONTACT_MARGIN = 2.0e-4
CONTACT_GAP = 5.0e-4
HAND_FRICTION = 4.0
PLUG_FRICTION = 2.0
SOCKET_FRICTION = 0.05
TABLE_FRICTION = 0.8
GROUND_FRICTION = 0.8
PLUG_SDF_RESOLUTION = 256
PLUG_SDF_BAND = 0.005
RIGID_CONTACT_MAX = 16384
RIGID_BODY_CONTACT_BUFFER_SIZE = 768
RIGID_CONTACT_ALPHA = 0.0

HAND_VHACD_MAX_HULLS = 4
HAND_VHACD_RESOLUTION = 50_000
HAND_VHACD_VOLUME_ERROR = 4.0
HAND_VHACD_MAX_RECURSION = 6
HAND_VHACD_MAX_VERTICES = 32

RIGHT_THUMB_TIP_OFFSET = wp.vec3(0.0006, 0.0867, 0.0)
RIGHT_INDEX_TIP_OFFSET = wp.vec3(0.0, 0.0492, 0.0086)

CAMERA_POSITION = wp.vec3(0.50, -0.52, 1.15)
CAMERA_PITCH = -12.0
CAMERA_YAW = 137.0


@wp.kernel
def _copy_indexed_q(
    source: wp.array[float],
    source_indices: wp.array[wp.int32],
    destination_indices: wp.array[wp.int32],
    destination: wp.array[float],
):
    """Copy matching IK coordinates into the floating-base scene model."""
    index = wp.tid()
    destination[destination_indices[index]] = source[source_indices[index]]


@wp.kernel
def _lock_ik_q(
    q: wp.array2d[float],
    indices: wp.array[wp.int32],
    values: wp.array[float],
):
    """Restore every non-arm coordinate after an IK iteration batch."""
    index = wp.tid()
    q[0, indices[index]] = values[index]


@wp.kernel
def _write_indexed_lerp(
    indices: wp.array[wp.int32],
    values_start: wp.array[float],
    values_end: wp.array[float],
    alpha: float,
    q: wp.array[float],
):
    """Interpolate an authored hand pose into generalized coordinates."""
    index = wp.tid()
    q[indices[index]] = values_start[index] * (1.0 - alpha) + values_end[index] * alpha


@wp.kernel
def _write_free_root_pose(
    root_q_start: int,
    position: wp.vec3,
    rotation: wp.quat,
    joint_q: wp.array[float],
):
    """Write one floating-base transform to generalized coordinates."""
    if wp.tid() == 0:
        joint_q[root_q_start + 0] = position[0]
        joint_q[root_q_start + 1] = position[1]
        joint_q[root_q_start + 2] = position[2]
        joint_q[root_q_start + 3] = rotation[0]
        joint_q[root_q_start + 4] = rotation[1]
        joint_q[root_q_start + 5] = rotation[2]
        joint_q[root_q_start + 6] = rotation[3]


@wp.kernel
def _interpolate_indexed_q(
    indices: wp.array[wp.int32],
    q_start: wp.array[float],
    q_end: wp.array[float],
    alpha: float,
    q_out: wp.array[float],
):
    """Interpolate only coordinates owned by the kinematic robot."""
    coordinate = indices[wp.tid()]
    q_out[coordinate] = q_start[coordinate] * (1.0 - alpha) + q_end[coordinate] * alpha


@wp.kernel
def _update_indexed_joint_velocity(
    joint_indices: wp.array[wp.int32],
    q_start: wp.array[float],
    q_end: wp.array[float],
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    inv_dt: float,
    qd_out: wp.array[float],
):
    """Compute velocity only for joints owned by the kinematic robot."""
    joint = joint_indices[wp.tid()]
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
                wp.quat(
                    q_start[q_begin + 3],
                    q_start[q_begin + 4],
                    q_start[q_begin + 5],
                    q_start[q_begin + 6],
                )
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


class Example:
    """Run realtime right-arm IK against a physical VBD plug and socket."""

    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    RIGHT_CONTACT_BODY_KEYWORDS = ("right_thumb", "right_index")
    INITIAL_POSTURE: ClassVar[dict[str, float]] = {
        "ANKLE": math.radians(30.0),
        "KNEE": math.radians(-60.0),
        "BUTTOCK": math.radians(30.0),
        "LEFT_J1": math.radians(14.0),
        "LEFT_J2": math.radians(-75.0),
        "LEFT_J3": 0.0,
        "LEFT_J4": math.radians(-30.0),
        "LEFT_J5": 0.0,
        "LEFT_J6": 0.0,
        "LEFT_J7": 0.0,
        "NECK2": math.radians(-30.0),
    }
    OPEN_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "RIGHT_HAND_INDEX": 0.2,
        "RIGHT_INDEX_PIP": 0.2,
        "RIGHT_HAND_THUMB1": 0.2,
        "RIGHT_HAND_THUMB2": 1.4,
    }
    GRASP_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "RIGHT_HAND_INDEX": 0.47,
        "RIGHT_INDEX_PIP": 0.47,
        "RIGHT_HAND_THUMB1": 0.35,
        "RIGHT_HAND_THUMB2": 1.4,
    }

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = int(args.substeps)
        self.vbd_iterations = int(args.vbd_iterations)
        self.ik_iterations = int(args.ik_iterations)
        if self.sim_substeps < 1:
            raise ValueError("--substeps must be at least 1")
        if self.vbd_iterations < 1:
            raise ValueError("--vbd-iterations must be at least 1")
        if self.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        if args.graph_capture and self.sim_substeps % 2 != 0:
            raise ValueError("CUDA graph capture requires an even --substeps value")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.phase = "settle"

        self.robot_urdf = Path(args.robot_urdf).expanduser().resolve()
        self.plug_mesh_path = Path(args.plug_mesh).expanduser().resolve()
        self.socket_mesh_path = Path(args.socket_mesh).expanduser().resolve()
        assets = (
            ("Dexforce W1 URDF", self.robot_urdf),
            ("WAIC plug mesh", self.plug_mesh_path),
            ("WAIC socket mesh", self.socket_mesh_path),
        )
        for label, path in assets:
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self._build_scene()
        self.device = self.model.device
        # This narrow, high-friction insertion was authored against the serial
        # FK reduction order.  CUDA level-parallel FK is equivalent up to
        # roundoff, but those last-bit pose differences select a different
        # rigid-contact branch during the pinch and make the plug miss the
        # socket.  Keep only the contact-driving scene FK deterministic; the IK
        # model remains on the parallel CUDA path.
        self.model._fk_articulation_level_start = None
        self._build_robot_coordinate_sets()
        self._build_hand_pose_control()
        self._build_ik()
        self._initialize_pose()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": self.vbd_iterations,
                "rigid_avbd_contact_alpha": RIGID_CONTACT_ALPHA,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0,
                "rigid_contact_stick_freeze_translation_eps": 0.0,
                "rigid_contact_stick_freeze_angular_eps": 0.0,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "friction_epsilon": 1.0e-4,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "rigid_contact_max": RIGID_CONTACT_MAX,
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend != "vbd_kinematic_full":
            raise RuntimeError(
                "Rigid plug/socket insertion requires the vbd_kinematic_full backend, "
                f"got {self.solver.features.backend}"
            )

        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        wp.copy(self.frame_q_start, self.model.joint_q)
        wp.copy(self.frame_q_end, self.model.joint_q)
        self.initial_plug_position = self._plug_world_position(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POSITION, CAMERA_PITCH, CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 38.0

        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.graph = None

    def _build_scene(self) -> None:
        """Build the robot, dynamic plug, socket, and support table."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = CONTACT_GAP
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = HAND_FRICTION
        builder.default_shape_cfg.margin = CONTACT_MARGIN
        SolverMJVBDV2.register_custom_attributes(builder)

        robot_articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(robot_articulation_start, builder.articulation_count))
        if len(self.robot_articulations) != 1:
            raise RuntimeError(f"Expected one W1 articulation, got {self.robot_articulations}")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self._set_builder_posture(builder)
        self._decompose_hand_collision_meshes(builder)
        self.robot_shape_end = builder.shape_count
        self._configure_robot_collision_flags(builder)

        plug_mesh = newton.Mesh.create_from_file(str(self.plug_mesh_path), compute_inertia=True, is_solid=True)
        plug_mesh.build_sdf(
            narrow_band_range=(-PLUG_SDF_BAND, PLUG_SDF_BAND),
            max_resolution=PLUG_SDF_RESOLUTION,
            margin=CONTACT_MARGIN + CONTACT_GAP,
        )
        plug_cfg = newton.ModelBuilder.ShapeConfig(
            density=PLUG_DENSITY,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=PLUG_FRICTION,
            margin=CONTACT_MARGIN,
            gap=CONTACT_GAP,
            is_solid=True,
        )
        self.plug_body = builder.add_body(
            xform=wp.transform(PLUG_INITIAL_POSITION, wp.quat_identity()),
            label="waic_plug_body",
        )
        self.plug_shape = builder.add_shape_mesh(
            self.plug_body,
            xform=wp.transform_identity(),
            mesh=plug_mesh,
            cfg=plug_cfg,
            color=PLUG_COLOR,
            label="waic_plug",
        )
        self._set_body_mass(builder, self.plug_body, PLUG_MASS)

        socket_mesh = newton.Mesh.create_from_file(
            str(self.socket_mesh_path),
            compute_inertia=False,
            is_solid=False,
        )
        socket_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=SOCKET_FRICTION,
            margin=CONTACT_MARGIN,
            gap=CONTACT_GAP,
            is_solid=False,
        )
        self.socket_shape = builder.add_shape_mesh(
            -1,
            xform=wp.transform(SOCKET_POSITION, SOCKET_ROTATION),
            mesh=socket_mesh,
            cfg=socket_cfg,
            color=SOCKET_COLOR,
            label="waic_socket",
        )

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=TABLE_FRICTION,
            margin=CONTACT_MARGIN,
            gap=CONTACT_GAP,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POSITION, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="plug_socket_table",
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=GROUND_FRICTION,
            margin=0.0,
        )
        builder.add_ground_plane(height=0.0, cfg=ground_cfg, color=GROUND_COLOR, label="plug_socket_ground")

        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.root_q_start = self._robot_root_q_start()

    def _set_builder_posture(self, builder: newton.ModelBuilder) -> None:
        """Apply the source scene's fixed W1 posture."""
        for joint_name, value in self.INITIAL_POSTURE.items():
            self._set_builder_joint_coordinate(builder, joint_name, value)
        for joint_name, value in self.OPEN_HAND_JOINTS.items():
            self._set_builder_joint_coordinate(builder, joint_name, value)

    def _build_robot_coordinate_sets(self) -> None:
        """Cache coordinates and joints owned exclusively by the robot proxy."""
        articulation_start = self.model.articulation_start.numpy()
        articulation_end = self.model.articulation_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        joints: list[int] = []
        coordinates: list[int] = []
        for articulation in self.robot_articulations:
            for joint in range(int(articulation_start[articulation]), int(articulation_end[articulation])):
                joints.append(joint)
                coordinates.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        if not joints or not coordinates:
            raise RuntimeError("The W1 proxy did not provide any kinematic coordinates")
        self.robot_joint_indices = wp.array(joints, dtype=wp.int32, device=self.device)
        self.robot_q_indices = wp.array(coordinates, dtype=wp.int32, device=self.device)

    def _build_hand_pose_control(self) -> None:
        """Cache the four authored thumb-index pinch coordinates."""
        q_start = self.model.joint_q_start.numpy()
        indices: list[int] = []
        open_values: list[float] = []
        grasp_values: list[float] = []
        for name, open_value in self.OPEN_HAND_JOINTS.items():
            joint = next(
                (index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name)),
                None,
            )
            if joint is None:
                raise ValueError(f"W1 joint is missing: {name}")
            if int(q_start[joint + 1] - q_start[joint]) != 1:
                raise ValueError(f"W1 hand joint must have one coordinate: {name}")
            indices.append(int(q_start[joint]))
            open_values.append(float(open_value))
            grasp_values.append(float(self.GRASP_HAND_JOINTS[name]))
        self.hand_q_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self.hand_q_open = wp.array(open_values, dtype=wp.float32, device=self.device)
        self.hand_q_grasp = wp.array(grasp_values, dtype=wp.float32, device=self.device)

    def _configure_robot_collision_flags(self, builder: newton.ModelBuilder) -> None:
        """Keep only the physical right palm, thumb, index, and middle colliders."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collision_mask = collide_shapes | collide_particles
        self.robot_contact_shapes: list[int] = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            body_label = "" if body < 0 else builder.body_label[body].lower()
            is_contact_body = body >= 0 and any(keyword in body_label for keyword in self.RIGHT_CONTACT_BODY_KEYWORDS)
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if is_contact_body and is_collider:
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
                self.robot_contact_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        if not self.robot_contact_shapes:
            raise RuntimeError("The W1 asset did not produce right-hand collision shapes")

    def _decompose_hand_collision_meshes(self, builder: newton.ModelBuilder) -> None:
        """Replace right-hand collision meshes with bounded convex parts."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        source_shapes: list[int] = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            if body < 0 or not (builder.shape_flags[shape] & collide_shapes):
                continue
            body_label = builder.body_label[body].lower()
            if (
                any(keyword in body_label for keyword in self.RIGHT_CONTACT_BODY_KEYWORDS)
                and builder.shape_type[shape] == newton.GeoType.MESH
            ):
                source_shapes.append(shape)
        if not source_shapes:
            raise RuntimeError("The W1 asset did not provide right-hand collision meshes for V-HACD")

        remeshed = builder.approximate_meshes(
            method="vhacd",
            shape_indices=source_shapes,
            raise_on_failure=True,
            keep_visual_shapes=False,
            maxConvexHulls=HAND_VHACD_MAX_HULLS,
            resolution=HAND_VHACD_RESOLUTION,
            minimumVolumePercentErrorAllowed=HAND_VHACD_VOLUME_ERROR,
            maxRecursionDepth=HAND_VHACD_MAX_RECURSION,
            maxNumVerticesPerCH=HAND_VHACD_MAX_VERTICES,
            asyncACD=False,
        )
        if len(remeshed) != len(source_shapes):
            raise RuntimeError(f"V-HACD converted {len(remeshed)} of {len(source_shapes)} hand collision meshes")

    @staticmethod
    def _set_body_mass(builder: newton.ModelBuilder, body: int, target_mass: float) -> None:
        """Scale imported inertia while preserving the mesh-derived center of mass."""
        imported_mass = float(builder.body_mass[body])
        if imported_mass <= 0.0:
            raise ValueError("The plug mesh produced zero mass")
        scale = target_mass / imported_mass
        inertia = np.asarray(builder.body_inertia[body], dtype=np.float64).reshape(3, 3) * scale
        builder.body_mass[body] = float(target_mass)
        builder.body_inertia[body] = wp.mat33(*[float(value) for value in inertia.reshape(-1)])

    def _build_ik(self) -> None:
        """Build a fixed-base W1 used only for per-frame right-arm IK."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self._set_builder_posture(builder)
        self.ik_model = builder.finalize(device=self.device)
        hand_body = self._body_index(self.ik_model.body_label, "right_hand_base")
        thumb_body = self._body_index(self.ik_model.body_label, "right_thumb_dist")
        index_body = self._body_index(self.ik_model.body_label, "right_index_dist")

        ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
        body_q = ik_state.body_q.numpy()
        hand_tf = wp.transform(*body_q[hand_body])
        thumb_tip = wp.transform_point(wp.transform(*body_q[thumb_body]), RIGHT_THUMB_TIP_OFFSET)
        index_tip = wp.transform_point(wp.transform(*body_q[index_body]), RIGHT_INDEX_TIP_OFFSET)
        fingertip_midpoint = 0.5 * (thumb_tip + index_tip)
        hand_offset = wp.transform_point(wp.transform_inverse(hand_tf), fingertip_midpoint)

        self.position_objective = ik.IKObjectivePosition(
            hand_body,
            hand_offset,
            wp.array([HAND_STANDBY_POSITION], dtype=wp.vec3, device=self.device),
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            hand_body,
            wp.quat_identity(),
            wp.array([self._quat_vector(HAND_TARGET_ROTATION)], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._ik_joint_limits()
        joint_limit_objective = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, joint_limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_lock_indices, self.ik_lock_values = self._ik_locked_q()
        source, destination = self._joint_coordinate_mapping()
        self.ik_source_indices = wp.array(source, dtype=wp.int32, device=self.device)
        self.scene_destination_indices = wp.array(destination, dtype=wp.int32, device=self.device)

    def _initialize_pose(self) -> None:
        """Solve the standby pose before allocating simulation states."""
        self._set_ik_target(HAND_STANDBY_POSITION)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        self._restore_locked_ik_q()
        self._copy_ik_to_scene(self.model.joint_q)
        self._write_hand_pose(0.0, self.model.joint_q)
        self._write_root_pose(self.model.joint_q)
        self.model.joint_qd.zero_()

    def _prepare_frame(self) -> None:
        """Evaluate the procedural controller and solve the right arm."""
        target, grasp_alpha, self.phase = self._sample_controller(self.sim_time)
        self._set_ik_target(target, self._sample_hand_rotation(self.sim_time))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._restore_locked_ik_q()

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        self._copy_ik_to_scene(self.frame_q_end)
        self._write_hand_pose(grasp_alpha, self.frame_q_end)
        self._write_root_pose(self.frame_q_end)

    def _simulate_substeps(self) -> None:
        """Advance prescribed robot proxies and the VBD-owned rigid plug."""
        for substep in range(self.sim_substeps):
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_indexed_q,
                self.robot_q_indices.shape[0],
                [self.robot_q_indices, self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _update_indexed_joint_velocity,
                self.robot_joint_indices.shape[0],
                [
                    self.robot_joint_indices,
                    self.frame_q_start,
                    self.frame_q_end,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.frame_dt,
                    self.state_0.joint_qd,
                ],
                device=self.device,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_simulation_graph(self) -> None:
        """Capture physics only; realtime IK remains outside the graph."""
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        state_0_ref = self.state_0
        state_1_ref = self.state_1
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.state_0 = state_0_ref
        self.state_1 = state_1_ref
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on {self.device}")

    def step(self) -> None:
        """Solve realtime IK, then simulate physical plug motion."""
        self._prepare_frame()
        if self.graph is None:
            self._simulate_substeps()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the W1, plug, socket, table, and ground."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Check that realtime coupled state remains finite."""
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Realtime plug/socket scene contains a non-finite body pose")
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise ValueError("Realtime plug/socket scene contains a non-finite joint coordinate")

    def test_final(self) -> None:
        """Verify realtime IK dispatch and, for long runs, physical insertion."""
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        if hasattr(self, "cached_joint_targets"):
            raise ValueError("Realtime plug/socket unexpectedly created a trajectory cache")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("Realtime IK returned a non-finite coordinate")

        retention_observation = int(
            (
                SETTLE_SECONDS
                + APPROACH_SECONDS
                + DESCEND_SECONDS
                + GRASP_SECONDS
                + POST_GRASP_HOLD_SECONDS
                + RAISE_SECONDS
                + ALIGN_SECONDS
                + INSERT_SECONDS
                + INSERT_SETTLE_SECONDS
                + RELEASE_SECONDS
                + RETRACT_SECONDS
            )
            * FPS
        )
        if self.frame_index < retention_observation:
            return
        plug_pose = np.asarray(self.state_0.body_q.numpy()[self.plug_body], dtype=np.float32)
        plug_position = plug_pose[:3]
        if not (-0.02 < plug_position[0] < 0.01):
            raise ValueError(f"Physical contact did not insert the plug: position={plug_position}")
        if abs(float(plug_position[1])) > 0.015 or abs(float(plug_position[2]) - 1.0) > 0.02:
            raise ValueError(f"The plug missed the socket axis: position={plug_position}")
        if abs(float(plug_pose[6])) < math.cos(math.radians(10.0)):
            raise ValueError(f"The plug entered the socket at excessive tilt: pose={plug_pose}")

    def _sample_controller(self, time_seconds: float) -> tuple[wp.vec3, float, str]:
        """Return continuous fingertip and pinch targets without replay."""
        hand_top = PLUG_REST_POSITION + HAND_TOP_OFFSET
        hand_grip = PLUG_REST_POSITION + PLUG_TO_GRIP
        hand_forward = PLUG_FORWARD_POSITION + PLUG_TO_GRIP
        hand_aligned = hand_forward + HAND_CARRY_CORRECTION
        hand_inserted = PLUG_INSERTED_POSITION + PLUG_TO_GRIP + HAND_CARRY_CORRECTION

        if time_seconds < SETTLE_SECONDS:
            return HAND_STANDBY_POSITION, 0.0, "settle"

        time_seconds -= SETTLE_SECONDS
        if time_seconds < APPROACH_SECONDS:
            alpha = self._smoothstep(time_seconds / APPROACH_SECONDS)
            return self._lerp_vec3(HAND_STANDBY_POSITION, hand_top, alpha), 0.0, "approach"

        time_seconds -= APPROACH_SECONDS
        if time_seconds < DESCEND_SECONDS:
            alpha = self._smoothstep(time_seconds / DESCEND_SECONDS)
            return self._lerp_vec3(hand_top, hand_grip, alpha), 0.0, "descend"

        time_seconds -= DESCEND_SECONDS
        if time_seconds < GRASP_SECONDS:
            return hand_grip, self._smoothstep(time_seconds / GRASP_SECONDS), "grasp"

        time_seconds -= GRASP_SECONDS
        if time_seconds < POST_GRASP_HOLD_SECONDS:
            return hand_grip, 1.0, "hold_grasp"

        time_seconds -= POST_GRASP_HOLD_SECONDS
        if time_seconds < RAISE_SECONDS:
            alpha = self._smoothstep(time_seconds / RAISE_SECONDS)
            return self._lerp_vec3(hand_grip, hand_forward, alpha), 1.0, "raise"

        time_seconds -= RAISE_SECONDS
        if time_seconds < ALIGN_SECONDS:
            alpha = self._smoothstep(time_seconds / ALIGN_SECONDS)
            return self._lerp_vec3(hand_forward, hand_aligned, alpha), 1.0, "align"

        time_seconds -= ALIGN_SECONDS
        if time_seconds < INSERT_SECONDS:
            alpha = self._smoothstep(time_seconds / INSERT_SECONDS)
            return self._lerp_vec3(hand_aligned, hand_inserted, alpha), 1.0, "insert"

        time_seconds -= INSERT_SECONDS
        if time_seconds < INSERT_SETTLE_SECONDS:
            return hand_inserted, 1.0, "settle_inserted"

        time_seconds -= INSERT_SETTLE_SECONDS
        if time_seconds < RELEASE_SECONDS:
            alpha = self._smoothstep(time_seconds / RELEASE_SECONDS)
            return hand_inserted, 1.0 - alpha, "release"

        time_seconds -= RELEASE_SECONDS
        if time_seconds < RETRACT_SECONDS:
            alpha = self._smoothstep(time_seconds / RETRACT_SECONDS)
            return self._lerp_vec3(hand_inserted, HAND_STANDBY_POSITION, alpha), 0.0, "retract"

        _ = min(time_seconds - RETRACT_SECONDS, HOLD_SECONDS)
        return HAND_STANDBY_POSITION, 0.0, "observe"

    def _sample_hand_rotation(self, time_seconds: float) -> wp.quat:
        """Rotate the wrist into axial alignment before contacting the socket."""
        align_start = (
            SETTLE_SECONDS
            + APPROACH_SECONDS
            + DESCEND_SECONDS
            + GRASP_SECONDS
            + POST_GRASP_HOLD_SECONDS
            + RAISE_SECONDS
        )
        if time_seconds <= align_start:
            return HAND_TARGET_ROTATION
        if time_seconds < align_start + ALIGN_SECONDS:
            alpha = self._smoothstep((time_seconds - align_start) / ALIGN_SECONDS)
            return wp.quat_slerp(HAND_TARGET_ROTATION, HAND_INSERT_ROTATION, alpha)
        return HAND_INSERT_ROTATION

    def _set_ik_target(self, position: wp.vec3, rotation: wp.quat = HAND_TARGET_ROTATION) -> None:
        self.position_objective.set_target_position(0, position)
        self.rotation_objective.set_target_rotation(0, self._quat_vector(rotation))

    def _restore_locked_ik_q(self) -> None:
        wp.launch(
            _lock_ik_q,
            self.ik_lock_indices.shape[0],
            [self.ik_q, self.ik_lock_indices, self.ik_lock_values],
            device=self.device,
        )

    def _copy_ik_to_scene(self, destination: wp.array[float]) -> None:
        wp.launch(
            _copy_indexed_q,
            self.ik_source_indices.shape[0],
            [self.ik_q[0], self.ik_source_indices, self.scene_destination_indices, destination],
            device=self.device,
        )

    def _write_hand_pose(self, grasp_alpha: float, destination: wp.array[float]) -> None:
        wp.launch(
            _write_indexed_lerp,
            self.hand_q_indices.shape[0],
            [self.hand_q_indices, self.hand_q_open, self.hand_q_grasp, grasp_alpha, destination],
            device=self.device,
        )

    def _write_root_pose(self, destination: wp.array[float]) -> None:
        wp.launch(
            _write_free_root_pose,
            1,
            [self.root_q_start, ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION, destination],
            device=self.device,
        )

    def _ik_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        qd_start = self.ik_model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in self.RIGHT_ARM}
        for joint, label in enumerate(self.ik_model.joint_label):
            if label in controlled:
                continue
            qd_begin = int(qd_start[joint])
            qd_end = int(qd_start[joint + 1])
            q_begin = int(q_start[joint])
            for offset in range(qd_end - qd_begin):
                lower[qd_begin + offset] = q[q_begin + offset] - 1.0e-4
                upper[qd_begin + offset] = q[q_begin + offset] + 1.0e-4
        return lower, upper

    def _ik_locked_q(self) -> tuple[wp.array, wp.array]:
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in self.RIGHT_ARM}
        indices: list[int] = []
        for joint, label in enumerate(self.ik_model.joint_label):
            if label in controlled:
                continue
            indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.device),
        )

    def _joint_coordinate_mapping(self) -> tuple[list[int], list[int]]:
        ik_q_start = self.ik_model.joint_q_start.numpy()
        scene_q_start = self.model.joint_q_start.numpy()
        scene_joints = {label: joint for joint, label in enumerate(self.model.joint_label)}
        source: list[int] = []
        destination: list[int] = []
        for ik_joint, label in enumerate(self.ik_model.joint_label):
            scene_joint = scene_joints.get(label)
            if scene_joint is None:
                continue
            ik_count = int(ik_q_start[ik_joint + 1] - ik_q_start[ik_joint])
            scene_count = int(scene_q_start[scene_joint + 1] - scene_q_start[scene_joint])
            if ik_count != scene_count:
                raise ValueError(f"Coordinate count differs for {label}: IK={ik_count}, scene={scene_count}")
            for offset in range(ik_count):
                source.append(int(ik_q_start[ik_joint]) + offset)
                destination.append(int(scene_q_start[scene_joint]) + offset)
        if not source:
            raise RuntimeError("No W1 joint coordinates could be mapped from IK to the scene")
        return source, destination

    def _robot_root_q_start(self) -> int:
        joint_type = self.model.joint_type.numpy()
        joint_child = self.model.joint_child.numpy()
        q_start = self.model.joint_q_start.numpy()
        for joint in range(self.model.joint_count):
            if int(joint_type[joint]) == int(newton.JointType.FREE) and int(joint_child[joint]) < self.robot_body_end:
                return int(q_start[joint])
        raise RuntimeError("The floating W1 root joint was not found")

    def _plug_world_position(self, state: newton.State) -> np.ndarray:
        position = state.body_q.numpy()[self.plug_body, :3]
        return np.asarray(position, dtype=np.float32)

    @staticmethod
    def _set_builder_joint_coordinate(builder: newton.ModelBuilder, name: str, value: float) -> None:
        joint = next(
            (index for index, label in enumerate(builder.joint_label) if label.endswith("/" + name)),
            None,
        )
        if joint is None:
            raise ValueError(f"Joint is missing: {name}")
        builder.joint_q[int(builder.joint_q_start[joint])] = float(value)

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    @staticmethod
    def _quat_vector(value: wp.quat) -> wp.vec4:
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def _lerp_vec3(a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
        return wp.vec3(
            float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
            float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
            float(a[2]) * (1.0 - alpha) + float(b[2]) * alpha,
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def create_parser():
        """Create command-line options for the realtime plug/socket scene."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
        parser.add_argument("--plug-mesh", type=Path, default=DEFAULT_PLUG_MESH)
        parser.add_argument("--socket-mesh", type=Path, default=DEFAULT_SOCKET_MESH)
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=DEFAULT_IK_ITERATIONS,
            help="Realtime right-arm IK iterations per displayed frame.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture only fixed-topology physics substeps; IK remains realtime.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
