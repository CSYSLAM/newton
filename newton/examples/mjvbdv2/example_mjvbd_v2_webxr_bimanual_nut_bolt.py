# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the full-W1 bimanual nut-and-bolt scene from Quest.

The original scene's pre-threaded dynamic M20 nut and bolt, detailed SDF
thread contact, and task-specific W1 hand collision filters are preserved.
Each Quest grip clutches the corresponding hand root, each trigger moves that
hand between an open pose and the recorded task pose, and rigid contact slows
the finger motion to reduce tunnelling through the threaded parts.

Use the guarded launcher from the repository root::

    ./scripts/start_quest_webxr_nut_bolt_teleop.sh
"""

from __future__ import annotations

import argparse
import signal
import time
import weakref
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from . import example_mjvbd_v2_bimanual_nut_bolt as nut_bolt
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    Pose,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)

DEFAULT_STALE_SECONDS = 0.25
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3
HANDS = ("left", "right")
OBSERVER_VIEW_MODE = "observer"
FIRST_PERSON_VIEW_MODE = "robot-first-person"
WEBXR_CAMERA_DOLLY_METERS = 0.20
WEBXR_CAMERA_HEIGHT_METERS = 0.05
WEBXR_CAMERA_PITCH_OFFSET_DEGREES = -5.0
HEAD_YAW_LIMITS = (-0.5 * np.pi, 0.5 * np.pi)
HEAD_PITCH_LIMITS = (-0.25 * np.pi, np.deg2rad(25.0))
HEAD_MAX_SPEED_RADIANS_S = np.deg2rad(50.0)
MAX_FINGER_SPEED_DEG_S = 90.0
MAX_FINGER_CONTACT_SPEED_DEG_S = 30.0
EYES_POSITION_IN_NECK2 = np.array((0.091, -0.051, 0.0), dtype=np.float32)
ROBOT_FORWARD = np.array((1.0, 0.0, 0.0), dtype=np.float32)
ROBOT_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)
# The M20 nut contact envelope bottoms out at z=0.809948 m.  A 0.052 mm
# initial overlap keeps the pre-threaded pair supported from the first step.
TABLE_TOP_Z = float(nut_bolt.ASSEMBLY_ORIGIN[2] - 0.030)
TABLE_HALF_EXTENTS = (0.38, 0.25, 0.03)
TABLE_POSITION = np.array(
    (float(nut_bolt.ASSEMBLY_ORIGIN[0] + 0.05), 0.20, TABLE_TOP_Z - TABLE_HALF_EXTENTS[2]),
    dtype=np.float32,
)
TABLE_COLOR = (0.30, 0.36, 0.44)
TABLE_CONTACT_KE = 3.0e5
TABLE_CONTACT_KD = 3.0e2
TABLE_CONTACT_MU = 0.8
WORKSPACE_LOWER_OFFSET = np.array((-0.65, -0.65, -0.40), dtype=np.float32)
WORKSPACE_UPPER_OFFSET = np.array((0.65, 0.65, 0.60), dtype=np.float32)
OPEN_HAND_JOINT_DEGREES = {
    "HAND_THUMB1": 0.0,
    "HAND_THUMB2": 90.0,
    "HAND_INDEX": 0.0,
    "INDEX_PIP": 0.0,
    "HAND_MIDDLE": 0.0,
    "MIDDLE_PIP": 0.0,
    "HAND_RING": 0.0,
    "RING_PIP": 0.0,
    "HAND_PINKY": 0.0,
    "PINKY_PIP": 0.0,
}


@wp.kernel
def _write_hand_target(
    open_q: wp.array[float],
    grasp_q: wp.array[float],
    grasp: float,
    desired_q: wp.array[float],
):
    """Interpolate one hand's desired joint coordinates from its trigger."""
    index = wp.tid()
    desired_q[index] = open_q[index] * (1.0 - grasp) + grasp_q[index] * grasp


@wp.kernel
def _limit_hand_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    desired_finger_q: wp.array[float],
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    hand_shape_mask: wp.array[int],
    free_max_step: float,
    contact_max_step: float,
    target_q: wp.array[float],
):
    """Limit one hand's joint step and slow it further during rigid contact."""
    finger = wp.tid()
    active_contact_count = wp.min(rigid_contact_count[0], rigid_contact_shape0.shape[0])
    active_contact_count = wp.min(active_contact_count, rigid_contact_shape1.shape[0])
    hand_contact = bool(False)
    for contact in range(active_contact_count):
        shape0 = rigid_contact_shape0[contact]
        shape1 = rigid_contact_shape1[contact]
        if shape0 >= 0 and shape0 < hand_shape_mask.shape[0] and hand_shape_mask[shape0] != 0:
            hand_contact = True
        if shape1 >= 0 and shape1 < hand_shape_mask.shape[0] and hand_shape_mask[shape1] != 0:
            hand_contact = True

    max_step = free_max_step
    if hand_contact:
        max_step = contact_max_step
    q_index = finger_q_indices[finger]
    delta = wp.clamp(desired_finger_q[finger] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


@wp.kernel
def _write_neck_pose(
    indices: wp.array[int],
    yaw: float,
    pitch: float,
    joint_q: wp.array[float],
):
    """Write the W1 neck yaw and pitch coordinates."""
    index = wp.tid()
    if index == 0:
        joint_q[indices[index]] = yaw
    else:
        joint_q[indices[index]] = pitch


def head_pose_to_neck_targets(head_pose: Pose | None) -> tuple[float, float]:
    """Map a relative WebXR head pose to W1 neck yaw and pitch offsets."""
    if head_pose is None:
        return 0.0, 0.0
    x, y, z, w = (float(value) for value in head_pose.orientation)
    forward_x = -2.0 * (x * z + y * w)
    forward_y = 2.0 * (x * w - y * z)
    forward_z = 2.0 * (x * x + y * y) - 1.0
    yaw = np.arctan2(-forward_x, -forward_z)
    pitch = np.arctan2(forward_y, np.hypot(forward_x, forward_z))
    return (
        float(np.clip(yaw, *HEAD_YAW_LIMITS)),
        float(np.clip(pitch, *HEAD_PITCH_LIMITS)),
    )


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(nut_bolt.Example):
    """Drive both W1 hands against the physical threaded pair with Quest."""

    reset_in_place = True

    def _add_scene_support(self, builder: newton.ModelBuilder) -> None:
        """Place the pre-threaded pair on a physical tabletop."""
        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=TABLE_CONTACT_KE,
            kd=TABLE_CONTACT_KD,
            mu=TABLE_CONTACT_MU,
            margin=0.0,
            gap=0.0,
        )
        self.table_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*TABLE_POSITION), wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="webxr_nut_bolt_table",
        )
        # Use the convex nut envelope for tabletop contact.  The detailed
        # nut SDF remains dedicated to the physical bolt threads.
        builder.add_shape_collision_filter_pair(self.table_shape, self.nut_thread_shape)
        self._static_boxes = [
            {
                "role": "table",
                "position": TABLE_POSITION.tolist(),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [2.0 * float(value) for value in TABLE_HALF_EXTENTS],
                "color": [float(value) for value in TABLE_COLOR],
            }
        ]

    def __init__(self, viewer, args):
        if args.graph_capture:
            raise ValueError("WebXR nut/bolt teleoperation requires --no-graph-capture")
        stale_seconds = float(args.xr_stale_seconds)
        if not np.isfinite(stale_seconds) or stale_seconds <= 0.0:
            raise ValueError("xr_stale_seconds must be finite and greater than zero")
        self.args = args
        self.xr_stale_seconds = stale_seconds
        self.xr_state = LatestXRFrame()
        self.retargeters = {
            hand: RelativePoseRetargeter(
                translation_scale=float(args.xr_translation_scale),
                max_translation=float(args.xr_max_translation),
            )
            for hand in HANDS
        }
        self._last_input_stream: str | None = None
        self._last_input_sequence: int | None = None
        self._record_button_pressed = False
        self._reset_button_pressed = False
        self._has_seen_controller = False
        self.episode_index = 0
        self.episode_frame = 0
        self.last_reset_source: str | None = None
        self.phase = "waiting_for_quest"
        self.exit_requested = False
        self.teleoperation_active = True
        self.simulation_active = True
        self.view_mode = OBSERVER_VIEW_MODE
        self._startup_started_at = time.perf_counter()

        nut_bolt.SDF_CACHE_DIR = Path(args.sdf_cache_dir).expanduser().resolve()
        super().__init__(viewer, args)
        self.device = self.model.device

        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._initial_hand_target_q = wp.clone(self.hand_target_q)
        self._initialize_neck_control()
        self._initialize_teleop_targets()
        self._build_independent_hand_control()
        self._robot_q_host_indices = self._robot_coordinate_indices()
        self._robot_body_ids = tuple(range(self.robot_body_end))
        self._robot_segments = self._build_robot_segments()

        trajectory_path = self._trajectory_path(args.trajectory_output)
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            trajectory_path,
            {
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": nut_bolt.SIM_SUBSTEPS,
                "physicsSolver": "SolverMJVBDV2",
                "robotUrdf": str(self.robot_urdf),
                "assembly": nut_bolt.ASSEMBLY,
                "sdfCacheDirectory": str(nut_bolt.SDF_CACHE_DIR),
                "robotJointLabels": list(self.model.joint_label),
                "robotCoordinateIndices": list(self._robot_q_host_indices),
                "headCoordinateIndices": self._neck_q_indices_host.tolist(),
                "boltBody": self.bolt_body,
                "nutBody": self.nut_body,
            },
            flush_every=int(args.record_flush_every),
        )
        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR W1/nut/bolt geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR W1/nut/bolt geometry packing complete")
        self.webxr_server = WebXRServer(
            self.xr_state,
            host=args.webxr_host,
            port=args.webxr_port,
            geometry_payload=geometry_payload,
            require_simulation_ready=args.webxr_server,
        )
        self._startup_sync_pending = bool(args.webxr_server)
        self._resource_finalizer = weakref.finalize(
            self,
            _close_resources,
            self.webxr_server,
            self.trajectory_recorder,
        )

        if args.webxr_server:
            self.webxr_server.start()
            print(
                f"Quest WebXR nut/bolt server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/nut/bolt geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
                "Reset scene: web page button or right thumbstick press\n"
                f"Trajectory: {self.trajectory_recorder.path}",
                flush=True,
            )

    def _trace_startup(self, phase: str) -> None:
        elapsed = time.perf_counter() - self._startup_started_at
        print(f"[startup +{elapsed:8.3f}s] {phase}", flush=True)

    @staticmethod
    def _vec3_array(value: wp.vec3) -> np.ndarray:
        return np.asarray([float(value[index]) for index in range(3)], dtype=np.float32)

    @staticmethod
    def _quat_array(value: wp.quat) -> np.ndarray:
        return np.asarray([float(value[index]) for index in range(4)], dtype=np.float32)

    @staticmethod
    def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = np.asarray(quaternion[:3], dtype=np.float32)
        value = np.asarray(vector, dtype=np.float32)
        twice_cross = 2.0 * np.cross(xyz, value)
        return value + float(quaternion[3]) * twice_cross + np.cross(xyz, twice_cross)

    @staticmethod
    def _trajectory_path(path_value: Path | None) -> Path:
        if path_value is not None:
            return Path(path_value).expanduser()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return Path("recordings") / f"webxr_bimanual_nut_bolt_{timestamp}.jsonl"

    def _initialize_neck_control(self) -> None:
        neck_joints = [self._joint_index(name) for name in ("NECK1", "NECK2")]
        joint_q_start = self.model.joint_q_start.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_limit_lower = self.model.joint_limit_lower.numpy()
        joint_limit_upper = self.model.joint_limit_upper.numpy()
        self._neck_q_indices_host = np.asarray(
            [int(joint_q_start[joint]) for joint in neck_joints],
            dtype=np.int32,
        )
        self._neck_q_indices = wp.array(self._neck_q_indices_host, dtype=wp.int32, device=self.device)
        self._neck_neutral = np.asarray(joint_q[self._neck_q_indices_host], dtype=np.float32)
        self._neck_lower = np.asarray(
            [joint_limit_lower[int(joint_qd_start[joint])] for joint in neck_joints],
            dtype=np.float32,
        )
        self._neck_upper = np.asarray(
            [joint_limit_upper[int(joint_qd_start[joint])] for joint in neck_joints],
            dtype=np.float32,
        )
        self._neck_targets = self._neck_neutral.copy()
        self._desired_neck_targets = self._neck_neutral.copy()
        self._head_body_ids = tuple(self._body_index(self.model.body_label, name) for name in ("neck1", "neck2"))
        self._eye_body = self._head_body_ids[-1]
        base_orientation = self._quat_array(self.robot_base_rotation)
        self._robot_forward = self._rotate_vector(base_orientation, ROBOT_FORWARD)
        self._robot_up = self._rotate_vector(base_orientation, ROBOT_UP)

    def _initialize_teleop_targets(self) -> None:
        initial_targets = self._initial_hand_root_targets()
        self._teleop_positions = {
            hand: self._vec3_array(wp.transform_get_translation(initial_targets[hand.upper()])) for hand in HANDS
        }
        self._teleop_orientations = {
            hand: self._quat_array(wp.transform_get_rotation(initial_targets[hand.upper()])) for hand in HANDS
        }
        self._initial_teleop_positions = {hand: value.copy() for hand, value in self._teleop_positions.items()}
        self._initial_teleop_orientations = {hand: value.copy() for hand, value in self._teleop_orientations.items()}
        self._teleop_grasps = dict.fromkeys(HANDS, 1.0)
        assembly_origin = np.asarray(nut_bolt.ASSEMBLY_ORIGIN, dtype=np.float32)
        self._target_position_min = assembly_origin + WORKSPACE_LOWER_OFFSET
        self._target_position_max = assembly_origin + WORKSPACE_UPPER_OFFSET
        for position in self._teleop_positions.values():
            self._target_position_min = np.minimum(self._target_position_min, position - 0.10)
            self._target_position_max = np.maximum(self._target_position_max, position + 0.10)

    def _build_independent_hand_control(self) -> None:
        initial_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
        self._hand_indices_by_side = {}
        self._hand_open_by_side = {}
        self._hand_grasp_by_side = {}
        self._desired_hand_q_by_side = {}
        self._hand_shape_mask_by_side = {}
        for hand in HANDS:
            side = hand.upper()
            indices = np.asarray(
                [self.hand_joint_q_indices[side][suffix] for suffix in OPEN_HAND_JOINT_DEGREES],
                dtype=np.int32,
            )
            open_q = np.radians([OPEN_HAND_JOINT_DEGREES[suffix] for suffix in OPEN_HAND_JOINT_DEGREES]).astype(
                np.float32
            )
            self._hand_indices_by_side[hand] = wp.array(indices, dtype=wp.int32, device=self.device)
            self._hand_open_by_side[hand] = wp.array(open_q, dtype=wp.float32, device=self.device)
            self._hand_grasp_by_side[hand] = wp.array(initial_q[indices], dtype=wp.float32, device=self.device)
            self._desired_hand_q_by_side[hand] = wp.zeros(len(indices), dtype=wp.float32, device=self.device)
            shape_mask = np.zeros(self.model.shape_count, dtype=np.int32)
            shape_mask[self.hand_shapes[side]] = 1
            self._hand_shape_mask_by_side[hand] = wp.array(shape_mask, dtype=wp.int32, device=self.device)

        self._max_finger_step = float(np.radians(MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self._max_finger_contact_step = float(np.radians(MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt)

    def _joint_index(self, name: str) -> int:
        return next(index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    def _robot_coordinate_indices(self) -> tuple[int, ...]:
        articulation_start = self.model.articulation_start.numpy()
        articulation_end = self.model.articulation_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        coordinates: list[int] = []
        for articulation in self.robot_articulations:
            for joint in range(int(articulation_start[articulation]), int(articulation_end[articulation])):
                coordinates.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        if not coordinates:
            raise RuntimeError("The nut/bolt scene did not provide W1 coordinates")
        return tuple(coordinates)

    def _build_robot_segments(self) -> tuple[tuple[int, int], ...]:
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        segments: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for parent, child in zip(parents, children, strict=True):
            pair = (int(parent), int(child))
            if pair[0] < 0 or pair[1] < 0 or pair[0] >= self.robot_body_end or pair[1] >= self.robot_body_end:
                continue
            if pair not in seen:
                seen.add(pair)
                segments.append(pair)
        return tuple(segments)

    @staticmethod
    def _mesh_vertex_normals(mesh: newton.Mesh) -> np.ndarray:
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        supplied = mesh.normals
        if supplied is not None:
            normals = np.asarray(supplied, dtype=np.float32).reshape(-1, 3).copy()
            if normals.shape == vertices.shape and np.all(np.isfinite(normals)):
                lengths = np.linalg.norm(normals, axis=1)
                if np.all(lengths > 1.0e-8):
                    normals /= lengths[:, None]
                    return normals

        indices = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
        normals = np.zeros_like(vertices)
        triangles = vertices[indices]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        for corner in range(3):
            np.add.at(normals, indices[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1.0e-8
        normals[valid] /= lengths[valid, None]
        normals[~valid] = (0.0, 0.0, 1.0)
        return normals

    def _build_webxr_geometry(self) -> bytes:
        """Pack the complete visible W1 and detailed threaded meshes."""
        shape_bodies = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy()
        shape_types = self.model.shape_type.numpy()
        shape_transforms = self.model.shape_transform.numpy()
        shape_scales = self.model.shape_scale.numpy()
        shape_colors = self.model.shape_color.numpy()
        visible_flag = int(newton.ShapeFlags.VISIBLE)
        mesh_ids: dict[int, int] = {}
        meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        shapes: list[dict[str, object]] = []
        role_by_shape = {
            **dict.fromkeys(self.robot_visual_shapes, "robot"),
            self.bolt_shape: "bolt",
            self.nut_thread_shape: "nut",
        }

        for shape, role in role_by_shape.items():
            body = int(shape_bodies[shape])
            source = self.model.shape_source[shape]
            if (
                body < 0
                or not int(shape_flags[shape]) & visible_flag
                or int(shape_types[shape]) != int(newton.GeoType.MESH)
                or not isinstance(source, newton.Mesh)
            ):
                continue
            source_id = id(source)
            mesh_index = mesh_ids.get(source_id)
            if mesh_index is None:
                mesh_index = len(meshes)
                mesh_ids[source_id] = mesh_index
                meshes.append(
                    (
                        np.asarray(source.vertices, dtype=np.float32),
                        self._mesh_vertex_normals(source),
                        np.asarray(source.indices, dtype=np.uint32),
                    )
                )
            transform = shape_transforms[shape]
            shapes.append(
                {
                    "body": body,
                    "role": role,
                    "mesh": mesh_index,
                    "position": [float(value) for value in transform[:3]],
                    "orientation": [float(value) for value in transform[3:7]],
                    "scale": [float(value) for value in shape_scales[shape]],
                    "color": [float(value) for value in shape_colors[shape]],
                }
            )

        roles = {str(shape["role"]) for shape in shapes}
        if not {"robot", "bolt", "nut"}.issubset(roles):
            raise RuntimeError(f"WebXR nut/bolt scene is missing required mesh roles: {sorted(roles)}")
        return pack_scene_geometry(meshes, shapes)

    def _viewer_camera_state(self) -> dict[str, list[float]]:
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "get_front") and hasattr(camera, "get_up"):
            position = camera.pos
            front = camera.get_front()
            up = camera.get_up()
        else:
            position = nut_bolt.CAMERA_POSITION
            pitch = np.deg2rad(nut_bolt.CAMERA_PITCH)
            yaw = np.deg2rad(nut_bolt.CAMERA_YAW)
            front = np.array(
                (np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)),
                dtype=np.float32,
            )
            right = np.cross(front, (0.0, 0.0, 1.0))
            right /= np.linalg.norm(right)
            up = np.cross(right, front)
        return {
            "position": [float(value) for value in position],
            "front": [float(value) for value in front],
            "up": [float(value) for value in up],
        }

    def _first_person_camera_state(self, body_q: np.ndarray) -> dict[str, list[float]]:
        neck_pose = body_q[self._eye_body]
        eye_position = neck_pose[:3] + self._rotate_vector(neck_pose[3:7], EYES_POSITION_IN_NECK2)
        return {
            "position": eye_position.tolist(),
            "front": self._robot_forward.tolist(),
            "up": self._robot_up.tolist(),
        }

    def _prepare_frame(self) -> None:
        """Retarget the newest Quest controller poses into both W1 hand roots."""
        frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds) if self.teleoperation_active else None
        controllers = {} if frame is None else frame.controllers
        if not self.teleoperation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._desired_neck_targets = self._neck_neutral.copy()
        elif frame is not None:
            if frame.view_mode != self.view_mode:
                self.view_mode = frame.view_mode
                for retargeter in self.retargeters.values():
                    retargeter.reset()
                print(f"Quest view mode changed to {self.view_mode}", flush=True)
            if self.view_mode == FIRST_PERSON_VIEW_MODE:
                neck_offsets = np.asarray(head_pose_to_neck_targets(frame.head_pose), dtype=np.float32)
                self._desired_neck_targets = np.clip(
                    self._neck_neutral + neck_offsets,
                    self._neck_lower,
                    self._neck_upper,
                )
            else:
                self._desired_neck_targets = self._neck_neutral.copy()
        elif self.view_mode == FIRST_PERSON_VIEW_MODE:
            self._desired_neck_targets = self._neck_neutral.copy()

        if controllers and not self._has_seen_controller:
            self._has_seen_controller = True
            if self.args.record_on_connect:
                self.trajectory_recorder.start()
        if frame is not None and "right" in controllers:
            if self._process_controller_buttons(frame.stream_id, frame.sequence, controllers["right"]):
                self.reset_physics(source="quest-controller")

        clutched: list[str] = []
        for hand in HANDS:
            controller = controllers.get(hand)
            if controller is None:
                self.retargeters[hand].reset()
                if self.teleoperation_active and self._has_seen_controller:
                    self._teleop_grasps[hand] = 0.0
                continue
            target = self.retargeters[hand].update(
                controller.pose,
                clutch=controller.clutch,
                robot_position=self._teleop_positions[hand],
                robot_orientation=self._teleop_orientations[hand],
                source_to_robot_rotation=(
                    IDENTITY_ROTATION if frame is not None and frame.controller_space == "newton-world" else None
                ),
            )
            if target is not None:
                self._teleop_positions[hand] = np.clip(
                    target.position,
                    self._target_position_min,
                    self._target_position_max,
                )
                self._teleop_orientations[hand] = target.orientation
            self._teleop_grasps[hand] = controller.trigger_value
            if controller.clutch:
                clutched.append(hand)

        if not self.teleoperation_active:
            self.phase = "teleoperation_standby"
        elif not controllers:
            self.phase = "waiting_for_quest" if not self._has_seen_controller else "quest_input_stale"
        elif clutched:
            self.phase = "quest_" + "_".join(clutched) + "_clutched"
        else:
            self.phase = "quest_idle"

    def _process_controller_buttons(self, stream_id: str, sequence: int, controller) -> bool:
        new_stream = stream_id != self._last_input_stream
        new_frame = new_stream or sequence != self._last_input_sequence
        if not new_frame:
            return False

        record_pressed = controller.is_button_pressed(QUEST_A_BUTTON_INDEX)
        reset_pressed = controller.is_button_pressed(QUEST_THUMBSTICK_BUTTON_INDEX)
        reset_requested = not new_stream and reset_pressed and not self._reset_button_pressed
        if not new_stream and record_pressed and not self._record_button_pressed:
            recording = self.trajectory_recorder.toggle()
            print(f"Quest trajectory recording {'resumed' if recording else 'paused'}", flush=True)

        self._record_button_pressed = record_pressed
        self._reset_button_pressed = reset_pressed
        self._last_input_stream = stream_id
        self._last_input_sequence = sequence
        return reset_requested

    def _update_hand_target(self) -> None:
        """Solve both live hand roots and rate-limit the independent finger targets."""
        self._prepare_frame()
        targets = {
            hand.upper(): wp.transform(
                wp.vec3(*[float(value) for value in self._teleop_positions[hand]]),
                wp.quat(*[float(value) for value in self._teleop_orientations[hand]]),
            )
            for hand in HANDS
        }
        self.current_hand_root_targets = targets
        self._update_robot_target(targets)
        for hand in HANDS:
            indices = self._hand_indices_by_side[hand]
            wp.launch(
                _write_hand_target,
                indices.shape[0],
                [
                    self._hand_open_by_side[hand],
                    self._hand_grasp_by_side[hand],
                    float(self._teleop_grasps[hand]),
                    self._desired_hand_q_by_side[hand],
                ],
                device=self.device,
            )
            wp.launch(
                _limit_hand_target_step,
                indices.shape[0],
                [
                    self.state_0.joint_q,
                    indices,
                    self._desired_hand_q_by_side[hand],
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self._hand_shape_mask_by_side[hand],
                    self._max_finger_step,
                    self._max_finger_contact_step,
                    self.hand_target_q,
                ],
                device=self.device,
            )

        maximum_neck_step = HEAD_MAX_SPEED_RADIANS_S * self.frame_dt
        self._neck_targets += np.clip(
            self._desired_neck_targets - self._neck_targets,
            -maximum_neck_step,
            maximum_neck_step,
        )
        wp.launch(
            _write_neck_pose,
            self._neck_q_indices.shape[0],
            [
                self._neck_q_indices,
                float(self._neck_targets[0]),
                float(self._neck_targets[1]),
                self.hand_target_q,
            ],
            device=self.device,
        )

    def _consume_webxr_reset(self) -> None:
        if self.xr_state.consume_reset() is not None:
            self.reset_physics(source="webxr")

    def _consume_teleoperation_mode(self) -> None:
        requested_mode = self.xr_state.consume_teleoperation_mode()
        if requested_mode is None:
            return
        request_id, teleoperation_active, simulation_active = requested_mode
        self.teleoperation_active = teleoperation_active
        self.simulation_active = simulation_active
        for retargeter in self.retargeters.values():
            retargeter.reset()
        self._record_button_pressed = False
        self._reset_button_pressed = False
        if teleoperation_active:
            self.phase = "teleoperation_resumed"
            print(f"Nut/bolt teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._desired_neck_targets = self._neck_neutral.copy()
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Nut/bolt controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.view_mode = OBSERVER_VIEW_MODE
            self._desired_neck_targets = self._neck_neutral.copy()
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(
                f"Nut/bolt teleoperation parked without destroying CUDA state (request {request_id})",
                flush=True,
            )

    def reset_physics(self, *, source: str) -> None:
        """Restore W1 and the pre-threaded pair without rebuilding CUDA resources."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.hand_target_q, self._initial_hand_target_q)
        wp.copy(self.ik_q, self._initial_ik_q)
        self._teleop_positions = {hand: value.copy() for hand, value in self._initial_teleop_positions.items()}
        self._teleop_orientations = {hand: value.copy() for hand, value in self._initial_teleop_orientations.items()}
        self._teleop_grasps = dict.fromkeys(HANDS, 1.0)
        self._neck_targets = self._neck_neutral.copy()
        self._desired_neck_targets = self._neck_neutral.copy()
        self.current_hand_root_targets = self._initial_hand_root_targets()
        for retargeter in self.retargeters.values():
            retargeter.reset()
        self.phase = "scene_reset"
        self.episode_index += 1
        self.episode_frame = 0
        self.last_reset_source = source
        self.trajectory_recorder.append_event(
            {
                "event": "scene-reset",
                "episode": self.episode_index,
                "frame": self.frame_index,
                "simulationTimeSeconds": self.sim_time,
                "source": source,
            }
        )
        if self.webxr_server.running and self.xr_state.status().clients > 0:
            self._publish_scene_state()
        print(f"Nut/bolt scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA nut/bolt simulation step started")
        self._update_hand_target()
        self.simulate()
        self.frame_index += 1
        self.sim_time += self.frame_dt
        if self._startup_sync_pending:
            self._trace_startup("first CUDA nut/bolt step complete; device synchronization started")
            wp.synchronize_device(self.device)
            self.webxr_server.mark_simulation_ready()
            self._startup_sync_pending = False
            self._trace_startup("device synchronization complete; Quest client may connect")

        self.episode_frame += 1
        should_record = self.trajectory_recorder.recording
        should_publish = self.teleoperation_active and self.webxr_server.running and self.xr_state.status().clients > 0
        if not (should_record or should_publish):
            return

        body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32)
        target_poses = [
            [
                *[float(value) for value in self._teleop_positions[hand]],
                *[float(value) for value in self._teleop_orientations[hand]],
            ]
            for hand in HANDS
        ]
        if should_record:
            joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
            body_qd = np.asarray(self.state_0.body_qd.numpy(), dtype=np.float32)
            input_frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds)
            self.trajectory_recorder.append(
                {
                    "frame": self.frame_index,
                    "episode": self.episode_index,
                    "episodeFrame": self.episode_frame,
                    "simulationTimeSeconds": self.sim_time,
                    "xrStreamId": None if input_frame is None else input_frame.stream_id,
                    "xrSequence": None if input_frame is None else input_frame.sequence,
                    "xrClientTimeMs": None if input_frame is None else input_frame.client_time_ms,
                    "xrControllerSpace": None if input_frame is None else input_frame.controller_space,
                    "viewMode": self.view_mode,
                    "headPose": (
                        None
                        if input_frame is None or input_frame.head_pose is None
                        else [
                            *input_frame.head_pose.position.tolist(),
                            *input_frame.head_pose.orientation.tolist(),
                        ]
                    ),
                    "neckJointTargets": self._neck_targets.tolist(),
                    "phase": self.phase,
                    "targetPoses": {hand: target_poses[index] for index, hand in enumerate(HANDS)},
                    "grasps": dict(self._teleop_grasps),
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "boltPose": body_q[self.bolt_body].tolist(),
                    "boltVelocity": body_qd[self.bolt_body].tolist(),
                    "nutPose": body_q[self.nut_body].tolist(),
                    "nutVelocity": body_qd[self.nut_body].tolist(),
                }
            )
        if should_publish:
            self._publish_scene_state(body_q, target_poses)

    def _publish_scene_state(
        self,
        body_q: np.ndarray | None = None,
        target_poses: list[list[float]] | None = None,
    ) -> None:
        if body_q is None:
            body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32)
        if target_poses is None:
            target_poses = [
                [
                    *[float(value) for value in self._teleop_positions[hand]],
                    *[float(value) for value in self._teleop_orientations[hand]],
                ]
                for hand in HANDS
            ]
        body_ids = (*self._robot_body_ids, self.bolt_body, self.nut_body)
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "bimanual-nut-bolt",
                "sceneInfo": {
                    "kind": "bimanual-nut-bolt",
                    "title": "双手螺母螺栓遥操作",
                    "description": "Quest 双眼显示完整 W1、承托桌面、动态 M20 螺栓、螺母和实时物理螺纹接触。",
                    "controls": [
                        ["左右 Grip", "按住并移动对应机器人手臂"],
                        ["左右 Trigger", "缓慢闭合对应任务手型"],
                        ["左摇杆", "观察模式下转动视角"],
                        ["X / 视角按钮", "切换观察模式与机器人第一人称"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位 W1、螺母和螺栓"],
                    ],
                },
                "frame": self.frame_index,
                "episode": self.episode_index,
                "episodeFrame": self.episode_frame,
                "lastResetSource": self.last_reset_source,
                "simulationTimeSeconds": self.sim_time,
                "phase": self.phase,
                "recording": self.trajectory_recorder.recording,
                "recordedFrames": self.trajectory_recorder.sample_count,
                "grasps": dict(self._teleop_grasps),
                "targetPoses": target_poses,
                "boltPose": body_q[self.bolt_body].tolist(),
                "nutPose": body_q[self.nut_body].tolist(),
                "camera": self._viewer_camera_state(),
                "firstPersonCamera": self._first_person_camera_state(body_q),
                "firstPersonHiddenBodies": list(self._head_body_ids),
                "viewMode": self.view_mode,
                "neckJointTargets": self._neck_targets.tolist(),
                "viewControls": {
                    "leftThumbstickRotate": True,
                    "cameraDollyMeters": WEBXR_CAMERA_DOLLY_METERS,
                    "cameraHeightMeters": WEBXR_CAMERA_HEIGHT_METERS,
                    "cameraPitchOffsetDegrees": WEBXR_CAMERA_PITCH_OFFSET_DEGREES,
                    "firstPersonEnabled": True,
                },
                "staticBoxes": self._static_boxes,
                "bodyPoses": [[body, *body_q[body].tolist()] for body in body_ids],
                "robotBodies": [[body, *body_q[body].tolist()] for body in self._robot_body_ids],
                "robotSegments": self._robot_segments,
            }
        )

    def render(self) -> None:
        self._consume_teleoperation_mode()
        if self.simulation_active:
            if self.teleoperation_active:
                self._consume_webxr_reset()
            super().render()
        else:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
        if self.xr_state.consume_shutdown() is not None:
            self.exit_requested = True

    def close(self) -> None:
        if self._resource_finalizer.alive:
            print("[shutdown] Nut/bolt WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] Nut/bolt WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        """Verify finite teleoperation state and the intended full MJVBDV2 backend."""
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("WebXR nut/bolt scene contains a non-finite body pose")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR bimanual nut/bolt IK returned a non-finite coordinate")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--robot-urdf", type=Path, default=nut_bolt.ROBOT_URDF)
        parser.add_argument("--ik-iterations", type=int, default=nut_bolt.RUNTIME_IK_ITERATIONS)
        parser.add_argument("--robot-base-x", type=float, default=float(nut_bolt.ROBOT_BASE_POSITION[0]))
        parser.add_argument("--robot-base-y", type=float, default=float(nut_bolt.ROBOT_BASE_POSITION[1]))
        parser.add_argument("--robot-base-z", type=float, default=float(nut_bolt.ROBOT_BASE_POSITION[2]))
        parser.add_argument("--robot-base-qx", type=float, default=float(nut_bolt.ROBOT_BASE_ROTATION[0]))
        parser.add_argument("--robot-base-qy", type=float, default=float(nut_bolt.ROBOT_BASE_ROTATION[1]))
        parser.add_argument("--robot-base-qz", type=float, default=float(nut_bolt.ROBOT_BASE_ROTATION[2]))
        parser.add_argument("--robot-base-qw", type=float, default=float(nut_bolt.ROBOT_BASE_ROTATION[3]))
        parser.add_argument(
            "--initial-keyframe",
            type=Path,
            default=nut_bolt.HAND_KEYFRAME_PATH,
            help="Bimanual hand keyframe used to initialize the full robot.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Capture one complete MJVBDV2 display frame on CUDA; unsupported for live WebXR input.",
        )
        parser.add_argument(
            "--sdf-cache-dir",
            type=Path,
            default=nut_bolt.SDF_CACHE_DIR,
            help="Persistent directory for the cooked M20 nut and bolt SDF meshes.",
        )
        parser.add_argument(
            "--webxr-server",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Serve the Quest browser client and accept bimanual controller frames.",
        )
        parser.add_argument(
            "--webxr-host",
            default="127.0.0.1",
            help="WebXR listen address; use 127.0.0.1 with adb reverse or 0.0.0.0 for LAN access.",
        )
        parser.add_argument("--webxr-port", type=int, default=8770)
        parser.add_argument(
            "--xr-translation-scale",
            type=float,
            default=1.0,
            help="Scale applied to clutched Quest controller translation.",
        )
        parser.add_argument(
            "--xr-max-translation",
            type=float,
            default=0.60,
            help="Maximum translation from one clutch anchor in metres.",
        )
        parser.add_argument(
            "--xr-stale-seconds",
            type=float,
            default=DEFAULT_STALE_SECONDS,
            help="Release clutch anchors after this interval without a Quest frame.",
        )
        parser.add_argument(
            "--trajectory-output",
            type=Path,
            default=None,
            help="JSONL trajectory path; defaults to a timestamped file under recordings/.",
        )
        parser.add_argument(
            "--record-on-connect",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Automatically start trajectory recording when the first controller frame arrives.",
        )
        parser.add_argument(
            "--record-flush-every",
            type=int,
            default=60,
            help="Flush the JSONL trajectory after this many recorded simulation frames.",
        )
        parser.set_defaults(num_frames=350)
        return parser


if __name__ == "__main__":
    exit_signal_requested = [False]
    active_example: list[Example] = []

    def _request_cooperative_exit(_signal_number, _frame) -> None:
        exit_signal_requested[0] = True
        if active_example:
            active_example[0].exit_requested = True

    signal.signal(signal.SIGINT, _request_cooperative_exit)
    signal.signal(signal.SIGTERM, _request_cooperative_exit)
    parser = Example.create_parser()
    print("[startup] Newton nut/bolt viewer and simulation device initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton nut/bolt viewer and simulation device initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
