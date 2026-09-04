# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the raised-table Dexforce W1 nonwoven-bag scene from Quest.

Both Quest grips clutch the corresponding W1 hand target and both triggers
independently close the fingers. The complete robot, finite table, and live
two-sided nonwoven bag are rendered in WebXR. The bag's initial world-space
forming force is released when manipulation starts so that it can be lifted.

Use the guarded launcher from the repository root::

    ./scripts/start_quest_webxr_nonwoven_bag_teleop.sh
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
import newton.ik as ik

from . import example_mjvbd_v2_nonwoven_bag_table_drop as bag_scene
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)
from ._webxr_w1_head import (
    FIRST_PERSON_VIEW_MODE,
    OBSERVER_VIEW_MODE,
    W1HeadController,
    serialize_head_pose,
)

DEFAULT_STALE_SECONDS = 0.25
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3
HANDS = ("left", "right")

WEBXR_CAMERA_DOLLY_METERS = 1.0
WEBXR_CAMERA_HEIGHT_METERS = 0.10
WEBXR_CAMERA_PITCH_OFFSET_DEGREES = -4.0
MAX_FINGER_SPEED_DEG_S = 90.0
MAX_FINGER_CONTACT_SPEED_DEG_S = 30.0
IK_ITERATIONS = 24
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

WORKSPACE_LOWER_OFFSET = np.array((-0.70, -0.90, -0.45), dtype=np.float32)
WORKSPACE_UPPER_OFFSET = np.array((0.70, 0.90, 0.75), dtype=np.float32)

GRASP_HAND_JOINTS = {
    "HAND_THUMB2": 0.5 * np.pi,
    "HAND_THUMB1": 0.4141,
    "HAND_INDEX": 0.6211,
    "INDEX_PIP": 0.9938,
    "HAND_MIDDLE": 0.6211,
    "MIDDLE_PIP": 0.9938,
    "HAND_RING": 0.6211,
    "RING_PIP": 0.9938,
    "HAND_PINKY": 0.6211,
    "PINKY_PIP": 0.9938,
}


@wp.kernel
def _interpolate_q(
    q0: wp.array[float],
    q1: wp.array[float],
    alpha: float,
    out: wp.array[float],
):
    coordinate = wp.tid()
    out[coordinate] = q0[coordinate] * (1.0 - alpha) + q1[coordinate] * alpha


@wp.kernel
def _joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    inverse_dt: float,
    out: wp.array[float],
):
    coordinate = wp.tid()
    out[coordinate] = (q1[coordinate] - q0[coordinate]) * inverse_dt


@wp.kernel
def _lock_q(
    q: wp.array2d[float],
    indices: wp.array[int],
    values: wp.array[float],
):
    index = wp.tid()
    q[0, indices[index]] = values[index]


@wp.kernel
def _copy_joint_q(src: wp.array[float], dst: wp.array[float]):
    coordinate = wp.tid()
    dst[coordinate] = src[coordinate]


@wp.kernel
def _write_ik_target_poses(
    left: wp.transform,
    right: wp.transform,
    target_poses: wp.array[wp.transform],
):
    if wp.tid() == 0:
        target_poses[0] = left
        target_poses[1] = right


@wp.kernel
def _unpack_ik_target_poses(
    target_poses: wp.array[wp.transform],
    left_positions: wp.array[wp.vec3],
    left_rotations: wp.array[wp.vec4],
    right_positions: wp.array[wp.vec3],
    right_rotations: wp.array[wp.vec4],
):
    if wp.tid() == 0:
        left = target_poses[0]
        left_positions[0] = wp.transform_get_translation(left)
        left_rotation = wp.transform_get_rotation(left)
        left_rotations[0] = wp.vec4(left_rotation[0], left_rotation[1], left_rotation[2], left_rotation[3])
        right = target_poses[1]
        right_positions[0] = wp.transform_get_translation(right)
        right_rotation = wp.transform_get_rotation(right)
        right_rotations[0] = wp.vec4(right_rotation[0], right_rotation[1], right_rotation[2], right_rotation[3])


@wp.kernel
def _write_hand_target(
    open_q: wp.array[float],
    grasp_q: wp.array[float],
    grasp: float,
    desired_q: wp.array[float],
):
    finger = wp.tid()
    desired_q[finger] = open_q[finger] * (1.0 - grasp) + grasp_q[finger] * grasp


@wp.kernel
def _limit_hand_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    desired_finger_q: wp.array[float],
    soft_contact_count: wp.array[int],
    soft_contact_shape: wp.array[int],
    hand_shape_mask: wp.array[int],
    free_max_step: float,
    contact_max_step: float,
    target_q: wp.array[float],
):
    """Limit finger motion and slow both directions while that hand touches the bag."""
    finger = wp.tid()
    active_contact_count = wp.min(soft_contact_count[0], soft_contact_shape.shape[0])
    hand_contact = bool(False)
    for contact in range(active_contact_count):
        shape = soft_contact_shape[contact]
        if shape >= 0 and shape < hand_shape_mask.shape[0] and hand_shape_mask[shape] != 0:
            hand_contact = True

    maximum_step = free_max_step
    if hand_contact:
        maximum_step = contact_max_step
    coordinate = finger_q_indices[finger]
    delta = wp.clamp(desired_finger_q[finger] - current_q[coordinate], -maximum_step, maximum_step)
    target_q[coordinate] = current_q[coordinate] + delta


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(bag_scene.Example):
    """Drive both W1 hands to grasp the physical nonwoven bag from Quest."""

    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_SUFFIXES = tuple(bag_scene.Example.OPEN_HAND_JOINTS)
    reset_in_place = True

    def __init__(self, viewer, args):
        if args.graph_capture:
            raise ValueError("WebXR nonwoven-bag teleoperation requires --no-graph-capture")
        if args.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        stale_seconds = float(args.xr_stale_seconds)
        if not np.isfinite(stale_seconds) or stale_seconds <= 0.0:
            raise ValueError("xr_stale_seconds must be finite and greater than zero")

        self.args = args
        self.xr_stale_seconds = stale_seconds
        self.ik_iterations = int(args.ik_iterations)
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
        self._bag_released_from_rest = False
        self.episode_index = 0
        self.episode_frame = 0
        self.last_reset_source: str | None = None
        self.phase = "waiting_for_quest"
        self.exit_requested = False
        self.teleoperation_active = True
        self.simulation_active = True
        self.view_mode = OBSERVER_VIEW_MODE
        self._startup_started_at = time.perf_counter()

        super().__init__(viewer, args)

        self.sim_substeps = bag_scene.SIM_SUBSTEPS
        self.frame_index = 0
        self.contacts = self.solver.contacts
        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self._build_ik()
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.frame_q_start = wp.clone(self.state_0.joint_q)
        self.frame_q_end = wp.clone(self.state_0.joint_q)
        self.lock_indices, self.lock_values = self._locked_q()
        self.ik_target_poses = wp.array(
            [self.left_home, self.right_home],
            dtype=wp.transform,
            device=self.device,
        )
        self._unpack_runtime_ik_target_poses()
        self._head_controller = W1HeadController(self.model, self.device, bag_scene.ROBOT_BASE_ROTATION)
        self._initialize_teleop_targets()
        self._build_independent_hand_control()

        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._robot_q_host_indices = self._robot_coordinate_indices()
        self._robot_body_ids = tuple(range(self.robot_body_end))
        self._robot_segments = self._build_robot_segments()
        self._bag_local_indices = np.asarray(self.bag_triangle_indices, dtype=np.uint32)
        if (
            self._bag_local_indices.size == 0
            or int(self._bag_local_indices.min()) < 0
            or int(self._bag_local_indices.max()) >= self.bag_particle_count
        ):
            raise RuntimeError("Nonwoven-bag triangle topology does not match its particle range")
        self._static_boxes = self._build_static_boxes()

        trajectory_path = self._trajectory_path(args.trajectory_output)
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            trajectory_path,
            {
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": self.sim_substeps,
                "physicsSolver": "SolverMJVBDV2",
                "robotUrdf": str(self.robot_urdf),
                "robotJointLabels": list(self.model.joint_label),
                "robotCoordinateIndices": list(self._robot_q_host_indices),
                "headCoordinateIndices": self._head_controller.coordinate_indices_host.tolist(),
                "bagObj": str(args.bag_obj),
                "bagParticleStart": self.bag_particle_start,
                "bagParticleCount": self.bag_particle_count,
                "bagTriangleIndices": self._bag_local_indices.reshape(-1).tolist(),
                "restShapeReleasedOnManipulation": True,
            },
            flush_every=int(args.record_flush_every),
        )

        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR W1/nonwoven-bag geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR W1/nonwoven-bag geometry packing complete")
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
                f"Quest WebXR nonwoven-bag server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/nonwoven-bag geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
                f"Deforming bag: {self.bag_particle_count} vertices\n"
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
    def _trajectory_path(path_value: Path | None) -> Path:
        if path_value is not None:
            return Path(path_value).expanduser()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return Path("recordings") / f"webxr_bimanual_nonwoven_bag_{timestamp}.jsonl"

    @staticmethod
    def _body_index(labels, name: str) -> int:
        matches = [index for index, label in enumerate(labels) if label == name or label.endswith("/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one W1 {name} body, found {len(matches)}")
        return matches[0]

    def _joint_index(self, name: str) -> int:
        matches = [
            index for index, label in enumerate(self.model.joint_label) if label == name or label.endswith("/" + name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one W1 {name} joint, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _v4(quaternion: wp.quat) -> wp.vec4:
        return wp.vec4(*(float(quaternion[index]) for index in range(4)))

    def _tcp(self, state, body: int) -> wp.transform:
        body_pose = wp.transform(*state.body_q.numpy()[body])
        rotation = wp.transform_get_rotation(body_pose)
        return wp.transform(
            wp.transform_get_translation(body_pose) + wp.quat_rotate(rotation, TCP_OFFSET),
            rotation,
        )

    def _build_ik(self) -> None:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(bag_scene.ROBOT_BASE_POSITION, bag_scene.ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self._set_builder_posture(builder)
        self.ik_model = builder.finalize(device=self.device)
        if tuple(self.ik_model.joint_label) != tuple(self.model.joint_label):
            raise RuntimeError("W1 IK and physical models have different joint layouts")

        left = self._body_index(self.ik_model.body_label, "left_j7")
        right = self._body_index(self.ik_model.body_label, "right_j7")
        self.left_obj = ik.IKObjectivePosition(
            left,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.right_home)], dtype=wp.vec3, device=self.device),
        )
        self.right_rot = ik.IKObjectiveRotation(
            right,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.right_home))], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._joint_limits()
        joint_limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.model.joint_q.numpy()
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        controlled = {*self.LEFT_ARM, *self.RIGHT_ARM}
        for joint, label in enumerate(self.model.joint_label):
            if label.rsplit("/", maxsplit=1)[-1] in controlled:
                continue
            q_count = int(q_starts[joint + 1] - q_starts[joint])
            qd_count = int(qd_starts[joint + 1] - qd_starts[joint])
            for local_index in range(min(q_count, qd_count)):
                q_index = int(q_starts[joint]) + local_index
                qd_index = int(qd_starts[joint]) + local_index
                lower[qd_index] = q[q_index] - 1.0e-4
                upper[qd_index] = q[q_index] + 1.0e-4
        return lower, upper

    def _locked_q(self) -> tuple[wp.array, wp.array]:
        q = self.model.joint_q.numpy()
        q_starts = self.model.joint_q_start.numpy()
        controlled = {*self.LEFT_ARM, *self.RIGHT_ARM}
        indices: list[int] = []
        for joint, label in enumerate(self.model.joint_label):
            if label.rsplit("/", maxsplit=1)[-1] in controlled:
                continue
            indices.extend(range(int(q_starts[joint]), int(q_starts[joint + 1])))
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.device),
        )

    def _initialize_teleop_targets(self) -> None:
        self._teleop_positions = {
            "left": self._vec3_array(wp.transform_get_translation(self.left_home)),
            "right": self._vec3_array(wp.transform_get_translation(self.right_home)),
        }
        self._teleop_orientations = {
            "left": self._quat_array(wp.transform_get_rotation(self.left_home)),
            "right": self._quat_array(wp.transform_get_rotation(self.right_home)),
        }
        self._initial_teleop_positions = {hand: value.copy() for hand, value in self._teleop_positions.items()}
        self._initial_teleop_orientations = {hand: value.copy() for hand, value in self._teleop_orientations.items()}
        self._teleop_grasps = dict.fromkeys(HANDS, 0.0)
        table_center = self._vec3_array(bag_scene.TABLE_CENTER)
        self._target_position_min = table_center + WORKSPACE_LOWER_OFFSET
        self._target_position_max = table_center + WORKSPACE_UPPER_OFFSET
        for position in self._teleop_positions.values():
            self._target_position_min = np.minimum(self._target_position_min, position - 0.10)
            self._target_position_max = np.maximum(self._target_position_max, position + 0.10)

    def _build_independent_hand_control(self) -> None:
        q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
        q_starts = self.model.joint_q_start.numpy()
        shape_body = np.asarray(self.model.shape_body.numpy(), dtype=np.int32)
        self._hand_indices_by_side = {}
        self._hand_open_by_side = {}
        self._hand_grasp_by_side = {}
        self._desired_hand_q_by_side = {}
        self._hand_shape_mask_by_side = {}
        for hand in HANDS:
            side = hand.upper()
            indices = np.asarray(
                [int(q_starts[self._joint_index(f"{side}_{suffix}")]) for suffix in self.HAND_SUFFIXES],
                dtype=np.int32,
            )
            self._hand_indices_by_side[hand] = wp.array(indices, dtype=wp.int32, device=self.device)
            self._hand_open_by_side[hand] = wp.array(q[indices], dtype=wp.float32, device=self.device)
            self._hand_grasp_by_side[hand] = wp.array(
                [GRASP_HAND_JOINTS[suffix] for suffix in self.HAND_SUFFIXES],
                dtype=wp.float32,
                device=self.device,
            )
            self._desired_hand_q_by_side[hand] = wp.zeros(len(indices), dtype=wp.float32, device=self.device)

            hand_shapes = [
                shape
                for shape in self.hand_particle_shapes
                if hand in self.model.body_label[int(shape_body[shape])].lower()
            ]
            if not hand_shapes:
                raise RuntimeError(f"Dexforce W1 did not provide particle colliders for the {hand} hand")
            shape_mask = np.zeros(self.model.shape_count, dtype=np.int32)
            shape_mask[hand_shapes] = 1
            self._hand_shape_mask_by_side[hand] = wp.array(shape_mask, dtype=wp.int32, device=self.device)

        self._max_finger_step = float(np.radians(MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self._max_finger_contact_step = float(np.radians(MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt)

    def _robot_coordinate_indices(self) -> tuple[int, ...]:
        articulation_start = self.model.articulation_start.numpy()
        articulation_end = self.model.articulation_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        coordinates: list[int] = []
        for articulation in self.robot_articulations:
            for joint in range(int(articulation_start[articulation]), int(articulation_end[articulation])):
                coordinates.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        if not coordinates:
            raise RuntimeError("The nonwoven-bag scene did not provide W1 coordinates")
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

    def _build_static_boxes(self) -> list[dict[str, object]]:
        boxes: list[dict[str, object]] = [
            {
                "role": "table",
                "position": self._vec3_array(bag_scene.TABLE_CENTER).tolist(),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [2.0 * float(value) for value in bag_scene.TABLE_HALF_EXTENTS],
                "color": [float(value) for value in bag_scene.TABLE_COLOR],
            }
        ]
        offset_x = bag_scene.TABLE_HALF_EXTENTS[0] - 2.0 * bag_scene.TABLE_LEG_HALF_EXTENTS[0]
        offset_y = bag_scene.TABLE_HALF_EXTENTS[1] - 2.0 * bag_scene.TABLE_LEG_HALF_EXTENTS[1]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                boxes.append(
                    {
                        "role": "table",
                        "position": [
                            float(bag_scene.TABLE_CENTER[0]) + x_sign * offset_x,
                            float(bag_scene.TABLE_CENTER[1]) + y_sign * offset_y,
                            bag_scene.TABLE_LEG_HALF_EXTENTS[2],
                        ],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                        "scale": [2.0 * float(value) for value in bag_scene.TABLE_LEG_HALF_EXTENTS],
                        "color": [float(value) for value in bag_scene.TABLE_COLOR],
                    }
                )
        return boxes

    @staticmethod
    def _vertex_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        normals = np.zeros_like(vertices)
        triangle_vertices = vertices[triangles]
        face_normals = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1.0e-8
        normals[valid] /= lengths[valid, None]
        normals[~valid] = (0.0, 0.0, 1.0)
        return normals

    @classmethod
    def _mesh_vertex_normals(cls, mesh: newton.Mesh) -> np.ndarray:
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        if mesh.normals is not None:
            normals = np.asarray(mesh.normals, dtype=np.float32).reshape(-1, 3).copy()
            if normals.shape == vertices.shape and np.all(np.isfinite(normals)):
                lengths = np.linalg.norm(normals, axis=1)
                if np.all(lengths > 1.0e-8):
                    normals /= lengths[:, None]
                    return normals
        return cls._vertex_normals(vertices, np.asarray(mesh.indices, dtype=np.int64))

    def _build_webxr_geometry(self) -> bytes:
        """Pack complete W1 visuals and the live, double-sided bag topology."""
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

        for shape in range(self.robot_shape_end):
            body = int(shape_bodies[shape])
            source = self.model.shape_source[shape]
            if (
                body < 0
                or body >= self.robot_body_end
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
                    "role": "robot",
                    "mesh": mesh_index,
                    "position": [float(value) for value in transform[:3]],
                    "orientation": [float(value) for value in transform[3:7]],
                    "scale": [float(value) for value in shape_scales[shape]],
                    "color": [float(value) for value in shape_colors[shape]],
                }
            )

        bag_positions = self._bag_positions()
        self._bag_webxr_mesh_index = len(meshes)
        meshes.append(
            (
                bag_positions,
                self._vertex_normals(bag_positions, self._bag_local_indices),
                self._bag_local_indices,
            )
        )
        shapes.append(
            {
                "body": -1,
                "role": "bag",
                "mesh": self._bag_webxr_mesh_index,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
                "color": [float(value) for value in bag_scene.BAG_COLOR],
                "doubleSided": True,
            }
        )
        roles = {str(shape["role"]) for shape in shapes}
        if not {"robot", "bag"}.issubset(roles):
            raise RuntimeError(f"WebXR nonwoven-bag scene is missing required mesh roles: {sorted(roles)}")
        return pack_scene_geometry(meshes, shapes)

    def _bag_positions(self) -> np.ndarray:
        particle_q = np.asarray(self.state_0.particle_q.numpy(), dtype=np.float32)
        return particle_q[self.bag_particle_start : self.bag_particle_start + self.bag_particle_count]

    def _viewer_camera_state(self) -> dict[str, list[float]]:
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "get_front") and hasattr(camera, "get_up"):
            position = camera.pos
            front = camera.get_front()
            up = camera.get_up()
        else:
            position = bag_scene.CAMERA_POSITION
            pitch = np.deg2rad(bag_scene.CAMERA_PITCH)
            yaw = np.deg2rad(bag_scene.CAMERA_YAW)
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

    def _prepare_frame(self) -> None:
        """Retarget the newest Quest controller poses into both W1 TCPs."""
        frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds) if self.teleoperation_active else None
        controllers = {} if frame is None else frame.controllers
        if not self.teleoperation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
        elif frame is not None:
            if frame.view_mode != self.view_mode:
                self.view_mode = frame.view_mode
                for retargeter in self.retargeters.values():
                    retargeter.reset()
                print(f"Quest view mode changed to {self.view_mode}", flush=True)
            self._head_controller.set_desired_pose(self.view_mode, frame.head_pose)
        elif self.view_mode == FIRST_PERSON_VIEW_MODE:
            self._head_controller.set_desired_pose(self.view_mode, None)

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
                if self.teleoperation_active:
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
            if controller.clutch or controller.trigger_value > 0.05:
                self._bag_released_from_rest = True

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

    def _set_runtime_ik_target_poses(self, left: wp.transform, right: wp.transform) -> None:
        wp.launch(_write_ik_target_poses, 1, [left, right, self.ik_target_poses], device=self.device)

    def _unpack_runtime_ik_target_poses(self) -> None:
        wp.launch(
            _unpack_ik_target_poses,
            1,
            [
                self.ik_target_poses,
                self.left_obj.target_positions,
                self.left_rot.target_rotations,
                self.right_obj.target_positions,
                self.right_rot.target_rotations,
            ],
            device=self.device,
        )

    def _solve_runtime_ik_frame(self) -> None:
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        self._unpack_runtime_ik_target_poses()
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        wp.launch(
            _lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.device,
        )
        wp.launch(
            _copy_joint_q,
            self.model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.device,
        )
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
                    self.frame_q_start,
                    indices,
                    self._desired_hand_q_by_side[hand],
                    self.contacts.soft_contact_count,
                    self.contacts.soft_contact_shape,
                    self._hand_shape_mask_by_side[hand],
                    self._max_finger_step,
                    self._max_finger_contact_step,
                    self.frame_q_end,
                ],
                device=self.device,
            )
        self._head_controller.write_targets(self.frame_q_end, self.frame_dt)

    def _simulate_substeps(self) -> None:
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                bag_scene._apply_particle_drag,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.model.particle_mass, bag_scene.AIR_DRAG_RATE],
                outputs=[self.state_0.particle_f],
                device=self.device,
            )
            if not self._bag_released_from_rest:
                wp.launch(
                    bag_scene._apply_rest_shape_elasticity,
                    dim=self.model.particle_count,
                    inputs=[
                        self.state_0.particle_q,
                        self.state_0.particle_qd,
                        self.model.particle_mass,
                        self.rest_shape_targets,
                        bag_scene.REST_SHAPE_STIFFNESS,
                        bag_scene.REST_SHAPE_DAMPING,
                    ],
                    outputs=[self.state_0.particle_f],
                    device=self.device,
                )
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_1.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity,
                self.model.joint_dof_count,
                [self.state_0.joint_q, self.state_1.joint_q, 1.0 / self.sim_dt, self.state_1.joint_qd],
                device=self.device,
            )
            newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self) -> None:
        self._prepare_frame()
        targets = {
            hand: wp.transform(
                wp.vec3(*[float(value) for value in self._teleop_positions[hand]]),
                wp.quat(*[float(value) for value in self._teleop_orientations[hand]]),
            )
            for hand in HANDS
        }
        self._set_runtime_ik_target_poses(targets["left"], targets["right"])
        self._solve_runtime_ik_frame()
        self._simulate_substeps()
        self.sim_time += self.frame_dt
        self.frame_index += 1

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
            print(f"Nonwoven-bag teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Nonwoven-bag controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(
                f"Nonwoven-bag teleoperation parked without destroying CUDA state (request {request_id})",
                flush=True,
            )

    def reset_physics(self, *, source: str) -> None:
        """Restore W1, open hands, and the complete bag without rebuilding CUDA resources."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.ik_q, self._initial_ik_q)
        self._teleop_positions = {hand: value.copy() for hand, value in self._initial_teleop_positions.items()}
        self._teleop_orientations = {hand: value.copy() for hand, value in self._initial_teleop_orientations.items()}
        self._teleop_grasps = dict.fromkeys(HANDS, 0.0)
        self._bag_released_from_rest = False
        self._head_controller.reset()
        self._set_runtime_ik_target_poses(self.left_home, self.right_home)
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
        print(f"Nonwoven-bag scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA nonwoven-bag simulation step started")
        self.simulate()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA nonwoven-bag step complete; device synchronization started")
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
        bag_positions = self._bag_positions()
        target_poses = self._target_pose_values()
        if should_record:
            joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
            particle_qd = np.asarray(self.state_0.particle_qd.numpy(), dtype=np.float32)
            bag_slice = slice(self.bag_particle_start, self.bag_particle_start + self.bag_particle_count)
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
                    "headPose": None if input_frame is None else serialize_head_pose(input_frame.head_pose),
                    "neckJointTargets": self._head_controller.targets.tolist(),
                    "phase": self.phase,
                    "targetPoses": {hand: target_poses[index] for index, hand in enumerate(HANDS)},
                    "grasps": dict(self._teleop_grasps),
                    "restShapeReleased": self._bag_released_from_rest,
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "bagParticleQ": bag_positions.reshape(-1).tolist(),
                    "bagParticleQd": particle_qd[bag_slice].reshape(-1).tolist(),
                }
            )
        if should_publish:
            self._publish_scene_state(body_q, bag_positions, target_poses)

    def _target_pose_values(self) -> list[list[float]]:
        return [
            [
                *[float(value) for value in self._teleop_positions[hand]],
                *[float(value) for value in self._teleop_orientations[hand]],
            ]
            for hand in HANDS
        ]

    def _publish_scene_state(
        self,
        body_q: np.ndarray | None = None,
        bag_positions: np.ndarray | None = None,
        target_poses: list[list[float]] | None = None,
    ) -> None:
        if body_q is None:
            body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32)
        if bag_positions is None:
            bag_positions = self._bag_positions()
        if target_poses is None:
            target_poses = self._target_pose_values()
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "bimanual-nonwoven-bag",
                "sceneInfo": {
                    "kind": "bimanual-nonwoven-bag",
                    "title": "无纺布袋双手抓取遥操作",
                    "description": "Quest 双眼显示完整 W1、高桌面和实时双面无纺布袋; 可直接抓住袋身或提手。",
                    "controls": [
                        ["左右 Grip", "按住并移动对应机器人手臂"],
                        ["左右 Trigger", "渐进闭合对应手指抓袋子"],
                        ["左摇杆", "观察模式下转动视角"],
                        ["X / 视角按钮", "切换观察模式与机器人第一人称"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位 W1、桌面袋子和定型状态"],
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
                "restShapeReleased": self._bag_released_from_rest,
                "camera": self._viewer_camera_state(),
                "firstPersonCamera": self._head_controller.camera_state(body_q),
                "firstPersonHiddenBodies": list(self._head_controller.hidden_body_ids),
                "viewMode": self.view_mode,
                "neckJointTargets": self._head_controller.targets.tolist(),
                "viewControls": {
                    "leftThumbstickRotate": True,
                    "cameraDollyMeters": WEBXR_CAMERA_DOLLY_METERS,
                    "cameraHeightMeters": WEBXR_CAMERA_HEIGHT_METERS,
                    "cameraPitchOffsetDegrees": WEBXR_CAMERA_PITCH_OFFSET_DEGREES,
                    "firstPersonEnabled": True,
                },
                "staticBoxes": self._static_boxes,
                "deformableMeshes": [
                    {
                        "mesh": self._bag_webxr_mesh_index,
                        "positions": bag_positions.reshape(-1).tolist(),
                    }
                ],
                "bodyPoses": [[body, *body_q[body].tolist()] for body in self._robot_body_ids],
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
            print("[shutdown] Nonwoven-bag WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] Nonwoven-bag WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        """Verify finite live IK and nonwoven-bag state without requiring a completed drop."""
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("WebXR nonwoven-bag scene contains a non-finite W1 body pose")
        if not np.all(np.isfinite(self.state_0.particle_q.numpy())):
            raise ValueError("WebXR nonwoven-bag scene contains a non-finite particle position")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR bimanual nonwoven-bag IK returned a non-finite coordinate")

    @staticmethod
    def create_parser():
        parser = bag_scene.Example.create_parser()
        parser.set_defaults(graph_capture=False)
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=IK_ITERATIONS,
            help="Realtime IK iterations for each displayed bimanual target.",
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
        parser.add_argument("--webxr-port", type=int, default=8771)
        parser.add_argument(
            "--xr-translation-scale",
            type=float,
            default=1.0,
            help="Scale applied to clutched Quest controller translation.",
        )
        parser.add_argument(
            "--xr-max-translation",
            type=float,
            default=0.80,
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
        parser.set_defaults(num_frames=600)
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
    print("[startup] Newton nonwoven-bag viewer and simulation device initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton nonwoven-bag viewer and simulation device initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
