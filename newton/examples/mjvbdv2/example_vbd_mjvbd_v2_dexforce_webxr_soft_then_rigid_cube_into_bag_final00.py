# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the full-W1 soft/rigid-cube bag scene from Quest.

The right Quest grip clutches the W1 right wrist and the right trigger drives
all ten right-hand finger coordinates through the recorded soft-cube grasp
pose.  The same finger pose is used for both cubes, while the soft cube, rigid
cube, and box bag all remain physical and are streamed from live Newton state.

Use the guarded USB workflow from the repository root::

    ./scripts/start_quest_webxr_soft_rigid_bag_teleop.sh
    ./scripts/stop_quest_webxr_soft_rigid_bag_teleop.sh
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

from . import example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00 as cube_scene
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)

soft0 = cube_scene.soft0
hand_reference = cube_scene.hand_reference

DEFAULT_STALE_SECONDS = 0.25
TARGET_POSITION_MIN = np.array((-0.75, -3.45, 0.85), dtype=np.float32)
TARGET_POSITION_MAX = np.array((0.35, -2.15, 1.85), dtype=np.float32)
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3
RELEASE_TRIGGER_THRESHOLD = 0.05
MAX_FINGER_SPEED_DEG_S = 90.0
WEBXR_CAMERA_DOLLY_METERS = 1.8
BAG_COLOR = cube_scene.BOX_BAG_COLOR
SOFT_CUBE_COLOR = cube_scene.SOFT_CUBE_COLOR
RIGID_CUBE_COLOR = tuple(float(value) for value in cube_scene.rigid_reference.CUBE_COLORS[0])


@wp.kernel
def _limit_finger_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    desired_finger_q: wp.array[float],
    max_step: float,
    target_q: wp.array[float],
):
    finger = wp.tid()
    q_index = finger_q_indices[finger]
    delta = wp.clamp(desired_finger_q[finger] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(cube_scene.Example):
    """Drive the W1 right hand in the mixed soft/rigid bag scene."""

    reset_in_place = True

    def __init__(self, viewer, args):
        stale_seconds = float(args.xr_stale_seconds)
        if not np.isfinite(stale_seconds) or stale_seconds <= 0.0:
            raise ValueError("xr_stale_seconds must be finite and greater than zero")
        self.xr_stale_seconds = stale_seconds
        self.xr_state = LatestXRFrame()
        self.retargeter = RelativePoseRetargeter(
            translation_scale=float(args.xr_translation_scale),
            max_translation=float(args.xr_max_translation),
        )
        self._last_input_stream: str | None = None
        self._last_input_sequence: int | None = None
        self._record_button_pressed = False
        self._reset_button_pressed = False
        self._has_seen_controller = False
        self.episode_index = 0
        self.episode_frame = 0
        self.last_reset_source: str | None = None
        self.exit_requested = False
        self.teleoperation_active = True
        self.simulation_active = True
        self._startup_started_at = time.perf_counter()

        super().__init__(viewer, args)

        initial_tcp = self._tcp(self.state_0, self.right_body)
        self._initial_target_position = self._vec3_array(wp.transform_get_translation(initial_tcp))
        self._initial_target_orientation = self._quat_array(wp.transform_get_rotation(initial_tcp))
        self._teleop_position = self._initial_target_position.copy()
        self._teleop_orientation = self._initial_target_orientation.copy()
        self._teleop_grasp = 0.0
        self._grasp_profile = "soft"
        self._open_finger_q = np.asarray(self.hand_start.numpy(), dtype=np.float32)
        self._soft_grasp_finger_q = np.asarray(self.hand_grasp.numpy(), dtype=np.float32)
        self._max_finger_step = float(np.radians(MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self._initial_hand_shape_margin = self.model.shape_margin.numpy()[self.right_hand_shapes].copy()
        self._teleop_contact_key: bool | None = None
        self._apply_teleop_contact_material(grasping=False)

        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._robot_q_host_indices = self._robot_coordinate_indices()
        self._robot_body_ids = tuple(range(self.robot_body_end))
        self._dynamic_body_ids = (*self._robot_body_ids, self.rigid_cube_body)
        self._robot_segments = self._build_robot_segments()
        self._bag_local_indices = (
            np.asarray(self.bag_render_triangle_indices.numpy(), dtype=np.int64) - self.bag_particle_start
        ).astype(np.uint32)
        self._soft_cube_local_indices = (
            np.asarray(self.soft_cube_render_triangle_indices.numpy(), dtype=np.int64) - self.soft_cube_particle_start
        ).astype(np.uint32)
        self._update_object_centers()
        table_position = self._world_vec(soft0.TABLE_POS)
        self._static_boxes = [
            {
                "position": self._vec3_array(table_position).tolist(),
                "orientation": self._quat_array(self.base_rot).tolist(),
                "scale": [2.0 * float(value) for value in soft0.TABLE_HALF_EXTENTS],
                "color": [float(value) for value in soft0.TABLE_COLOR],
            }
        ]

        trajectory_path = self._trajectory_path(args.trajectory_output)
        rigid_half_extents = hand_reference.sequential_base.recorder.CUBE_HALF_EXTENTS
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            trajectory_path,
            {
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": self.sim_substeps,
                "physicsSolver": "SolverMJVBDV2",
                "robotUrdf": str(self.urdf_path),
                "robotJointLabels": list(self.model.joint_label),
                "robotCoordinateIndices": list(self._robot_q_host_indices),
                "bagParticleStart": self.bag_particle_start,
                "bagParticleCount": self.bag_particle_end - self.bag_particle_start,
                "bagTriangleIndices": self._bag_local_indices.reshape(-1).tolist(),
                "softCubeParticleStart": self.soft_cube_particle_start,
                "softCubeParticleCount": self.soft_cube_particle_end - self.soft_cube_particle_start,
                "softCubeTriangleIndices": self._soft_cube_local_indices.reshape(-1).tolist(),
                "rigidCubeBody": self.rigid_cube_body,
                "rigidCubeHalfExtents": [float(value) for value in rigid_half_extents],
                "graspProfileSelection": "soft-cube-pose-for-both-objects",
            },
            flush_every=int(args.record_flush_every),
        )
        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR W1/soft-rigid-bag geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR W1/soft-rigid-bag geometry packing complete")
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
                f"Quest WebXR soft/rigid-cube bag server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/cubes/bag geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
                f"Deforming bag: {self.bag_particle_end - self.bag_particle_start} vertices\n"
                f"Deforming soft cube: {self.soft_cube_particle_end - self.soft_cube_particle_start} vertices\n"
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
        return Path("recordings") / f"webxr_soft_rigid_cube_into_bag_{timestamp}.jsonl"

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
        supplied = mesh.normals
        if supplied is not None:
            normals = np.asarray(supplied, dtype=np.float32).reshape(-1, 3).copy()
            if normals.shape == vertices.shape and np.all(np.isfinite(normals)):
                lengths = np.linalg.norm(normals, axis=1)
                if np.all(lengths > 1.0e-8):
                    normals /= lengths[:, None]
                    return normals
        return cls._vertex_normals(vertices, np.asarray(mesh.indices, dtype=np.int64))

    @staticmethod
    def _box_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a sharp-normal unit box scaled by Newton half extents."""
        vertices = np.asarray(
            [
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, 1),
                (-1, 1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (1, 1, 1),
                (1, -1, 1),
                (-1, -1, -1),
                (1, -1, -1),
                (1, -1, 1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, 1, 1),
                (1, 1, -1),
                (-1, -1, -1),
                (-1, 1, -1),
                (1, 1, -1),
                (1, -1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            dtype=np.float32,
        )
        normals = np.repeat(
            np.asarray(
                [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
                dtype=np.float32,
            ),
            4,
            axis=0,
        )
        indices = np.asarray(
            [
                0,
                1,
                2,
                0,
                2,
                3,
                4,
                5,
                6,
                4,
                6,
                7,
                8,
                9,
                10,
                8,
                10,
                11,
                12,
                13,
                14,
                12,
                14,
                15,
                16,
                17,
                18,
                16,
                18,
                19,
                20,
                21,
                22,
                20,
                22,
                23,
            ],
            dtype=np.uint32,
        )
        return vertices, normals, indices

    def _build_webxr_geometry(self) -> bytes:
        """Pack full W1 visuals and all three manipulated objects."""
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

        for shape in self.robot_visual_shapes:
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

        particle_q = np.asarray(self.state_0.particle_q.numpy(), dtype=np.float32)
        bag_positions = particle_q[self.bag_particle_start : self.bag_particle_end]
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
                "doubleSided": True,
                "mesh": self._bag_webxr_mesh_index,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
                "color": list(BAG_COLOR),
            }
        )

        soft_cube_positions = particle_q[self.soft_cube_particle_start : self.soft_cube_particle_end]
        self._soft_cube_webxr_mesh_index = len(meshes)
        meshes.append(
            (
                soft_cube_positions,
                self._vertex_normals(soft_cube_positions, self._soft_cube_local_indices),
                self._soft_cube_local_indices,
            )
        )
        shapes.append(
            {
                "body": -1,
                "role": "soft-cube",
                "mesh": self._soft_cube_webxr_mesh_index,
                "position": [0.0, 0.0, 0.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
                "color": list(SOFT_CUBE_COLOR),
            }
        )

        rigid_mesh_index = len(meshes)
        meshes.append(self._box_mesh())
        rigid_transform = shape_transforms[self.rigid_cube_shape]
        shapes.append(
            {
                "body": self.rigid_cube_body,
                "role": "rigid-cube",
                "mesh": rigid_mesh_index,
                "position": [float(value) for value in rigid_transform[:3]],
                "orientation": [float(value) for value in rigid_transform[3:7]],
                "scale": [float(value) for value in shape_scales[self.rigid_cube_shape]],
                "color": list(RIGID_CUBE_COLOR),
            }
        )

        roles = {str(shape["role"]) for shape in shapes}
        required_roles = {"robot", "bag", "soft-cube", "rigid-cube"}
        if not required_roles.issubset(roles):
            raise RuntimeError(f"WebXR soft/rigid bag scene is missing mesh roles: {sorted(required_roles - roles)}")
        return pack_scene_geometry(meshes, shapes)

    def _robot_coordinate_indices(self) -> tuple[int, ...]:
        articulation_start = self.model.articulation_start.numpy()
        articulation_end = self.model.articulation_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        coordinates: list[int] = []
        for articulation in self.robot_articulations:
            for joint in range(int(articulation_start[articulation]), int(articulation_end[articulation])):
                coordinates.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        if not coordinates:
            raise RuntimeError("The W1 soft/rigid bag scene did not provide robot coordinates")
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

    def _viewer_camera_state(self) -> dict[str, list[float]]:
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "get_front") and hasattr(camera, "get_up"):
            position = camera.pos
            front = camera.get_front()
            up = camera.get_up()
        else:
            position = soft0.CAMERA_POS
            pitch = np.deg2rad(soft0.CAMERA_PITCH)
            yaw = np.deg2rad(soft0.CAMERA_YAW)
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

    def _tcp_to_root(self, tcp: wp.transform) -> wp.transform:
        wrist_position = wp.transform_get_translation(tcp)
        wrist_rotation = wp.transform_get_rotation(tcp)
        target_offset = soft0.TCP_OFFSET - cube_scene.recorded_soft.RIGHT_J7_TO_HAND_BASE_OFFSET
        root_position = wrist_position - wp.quat_rotate(wrist_rotation, target_offset)
        root_rotation = self._quat_mul(
            wrist_rotation,
            cube_scene.recorded_soft.RIGHT_J7_TO_HAND_BASE_ROTATION,
        )
        return wp.transform(root_position, root_rotation)

    def _restore_hand_shape_margins(self) -> None:
        margins = self.model.shape_margin.numpy()
        margins[self.right_hand_shapes] = self._initial_hand_shape_margin
        margins[self.rigid_cube_shape] = hand_reference.RIGID_CUBE_MARGIN
        self.model.shape_margin.assign(margins)

    def _apply_teleop_contact_material(self, *, grasping: bool) -> None:
        key = bool(grasping)
        if self._teleop_contact_key == key:
            return
        self._set_hand_shape_collision(True)
        self._set_hand_particle_collision(True)
        self._restore_hand_shape_margins()

        if not grasping:
            self._set_shape_material(
                [*self.right_hand_shapes, self.rigid_cube_shape],
                hand_reference.RIGID_RELEASE_CONTACT[0],
                hand_reference.RIGID_RELEASE_CONTACT[1],
                hand_reference.RIGID_RELEASE_CONTACT[2],
            )
            soft_material = hand_reference.SOFT_FREE_CONTACT
            soft_material_index = cube_scene._SOFT_MATERIAL_FREE
        else:
            self._set_shape_material(
                self.right_hand_shapes,
                hand_reference.SOFT_GRASP_CONTACT[0],
                hand_reference.SOFT_GRASP_CONTACT[1],
                soft0.GRASP_FRICTION,
            )
            self._set_shape_material(
                [self.rigid_cube_shape],
                hand_reference.RIGID_GRASP_CONTACT[0],
                hand_reference.RIGID_GRASP_CONTACT[1],
                hand_reference.RIGID_GRASP_CONTACT[2],
            )
            soft_material = hand_reference.SOFT_GRASP_CONTACT
            soft_material_index = cube_scene._SOFT_MATERIAL_GRASP

        self.model.soft_contact_ke = soft_material[0]
        self.model.soft_contact_kd = soft_material[1]
        self.model.soft_contact_mu = soft_material[2]
        self.soft_contact_material_index.fill_(soft_material_index)
        self.contact_phase = f"teleop_soft_pose_{'grasp' if grasping else 'release'}"
        self._teleop_contact_key = key

    def _update_object_centers(
        self,
        particle_q: np.ndarray | None = None,
        body_q: np.ndarray | None = None,
    ) -> None:
        if particle_q is None:
            particle_q = np.asarray(self.state_0.particle_q.numpy(), dtype=np.float32)
        if body_q is None:
            body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32)
        self._soft_cube_center = np.mean(
            particle_q[self.soft_cube_particle_start : self.soft_cube_particle_end],
            axis=0,
        ).astype(np.float32)
        self._rigid_cube_center = np.asarray(body_q[self.rigid_cube_body, :3], dtype=np.float32).copy()

    def _prepare_frame(self) -> None:
        """Retarget the newest right Quest controller pose into W1 IK."""
        frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds) if self.teleoperation_active else None
        controller = None if frame is None else frame.controllers.get("right")
        if controller is None:
            self.retargeter.reset()
            self._record_button_pressed = False
            self._reset_button_pressed = False
            if self.teleoperation_active:
                self.phase = "waiting_for_quest" if not self._has_seen_controller else "quest_input_stale"
            else:
                self.phase = "teleoperation_standby"
        else:
            if self._process_controller_buttons(frame.stream_id, frame.sequence, controller):
                self.reset_physics(source="quest-controller")
            target = self.retargeter.update(
                controller.pose,
                clutch=controller.clutch,
                robot_position=self._teleop_position,
                robot_orientation=self._teleop_orientation,
                source_to_robot_rotation=IDENTITY_ROTATION if frame.controller_space == "newton-world" else None,
            )
            if target is not None:
                self._teleop_position = np.clip(target.position, TARGET_POSITION_MIN, TARGET_POSITION_MAX)
                self._teleop_orientation = target.orientation
            self._teleop_grasp = controller.trigger_value
            self.phase = f"quest_{'clutched' if controller.clutch else 'idle'}_soft_pose"

        self._apply_teleop_contact_material(grasping=self._teleop_grasp > RELEASE_TRIGGER_THRESHOLD)
        target_tcp = wp.transform(
            wp.vec3(*[float(value) for value in self._teleop_position]),
            wp.quat(*[float(value) for value in self._teleop_orientation]),
        )
        self.current_target_root = self._tcp_to_root(target_tcp)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(target_tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(target_tcp)))

        desired_finger_q = self._open_finger_q + self._teleop_grasp * (self._soft_grasp_finger_q - self._open_finger_q)
        self.desired_finger_q.assign(desired_finger_q)
        self._solve_ik_and_assemble_joint_targets()
        wp.launch(
            _limit_finger_target_step,
            self.hand_indices.shape[0],
            [
                self.frame_q_start,
                self.hand_indices,
                self.desired_finger_q,
                self._max_finger_step,
                self.frame_q_end,
            ],
            device=self.device,
        )

    def _process_controller_buttons(self, stream_id: str, sequence: int, controller) -> bool:
        new_stream = stream_id != self._last_input_stream
        new_frame = new_stream or sequence != self._last_input_sequence
        if not new_frame:
            return False
        if not self._has_seen_controller:
            self._has_seen_controller = True
            if self.args.record_on_connect:
                self.trajectory_recorder.start()

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
        self.retargeter.reset()
        self._record_button_pressed = False
        self._reset_button_pressed = False
        if teleoperation_active:
            self.phase = "teleoperation_resumed"
            print(f"Soft/rigid-cube bag teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Soft/rigid-cube bag controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(
                f"Soft/rigid-cube bag teleoperation parked without destroying CUDA state (request {request_id})",
                flush=True,
            )

    def reset_physics(self, *, source: str) -> None:
        """Restore W1, both cubes, and the deformable bag in place."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.ik_q, self._initial_ik_q)
        self.desired_finger_q.assign(self._open_finger_q)
        self._teleop_position = self._initial_target_position.copy()
        self._teleop_orientation = self._initial_target_orientation.copy()
        self._teleop_grasp = 0.0
        self._grasp_profile = "soft"
        self._teleop_contact_key = None
        self._apply_teleop_contact_material(grasping=False)
        self.retargeter.reset()
        self.maximum_soft_contact_count.zero_()
        self.maximum_body_particle_contact_count.zero_()
        self.object_released.fill(False)
        self.previous_grip = 0.0
        self.release_contact_material_applied = False
        self.current_target_root = self._tcp_to_root(
            wp.transform(
                wp.vec3(*[float(value) for value in self._initial_target_position]),
                wp.quat(*[float(value) for value in self._initial_target_orientation]),
            )
        )
        self._update_object_centers()
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
        print(f"Soft/rigid-cube bag scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first soft/rigid-cube bag simulation step started")
        super().step()
        if self._startup_sync_pending:
            self._trace_startup("first soft/rigid-cube bag step complete; device synchronization started")
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
        particle_q = np.asarray(self.state_0.particle_q.numpy(), dtype=np.float32)
        bag_positions = particle_q[self.bag_particle_start : self.bag_particle_end]
        soft_cube_positions = particle_q[self.soft_cube_particle_start : self.soft_cube_particle_end]
        self._update_object_centers(particle_q, body_q)
        target_pose = [*self._teleop_position.tolist(), *self._teleop_orientation.tolist()]
        rigid_cube_pose = [float(value) for value in body_q[self.rigid_cube_body]]
        if should_record:
            joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
            particle_qd = np.asarray(self.state_0.particle_qd.numpy(), dtype=np.float32)
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
                    "phase": self.phase,
                    "graspProfile": self._grasp_profile,
                    "targetPose": target_pose,
                    "grasp": float(self._teleop_grasp),
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "bagParticleQ": bag_positions.reshape(-1).tolist(),
                    "bagParticleQd": particle_qd[self.bag_particle_start : self.bag_particle_end].reshape(-1).tolist(),
                    "softCubeParticleQ": soft_cube_positions.reshape(-1).tolist(),
                    "softCubeParticleQd": particle_qd[self.soft_cube_particle_start : self.soft_cube_particle_end]
                    .reshape(-1)
                    .tolist(),
                    "rigidCubePose": rigid_cube_pose,
                    "rigidCubeVelocity": body_qd[self.rigid_cube_body].reshape(-1).tolist(),
                }
            )
        if should_publish:
            self._publish_scene_state(
                body_q,
                bag_positions,
                soft_cube_positions,
                target_pose,
                rigid_cube_pose,
            )

    def _publish_scene_state(
        self,
        body_q: np.ndarray,
        bag_positions: np.ndarray,
        soft_cube_positions: np.ndarray,
        target_pose: list[float],
        rigid_cube_pose: list[float],
    ) -> None:
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "soft-rigid-cubes-into-bag",
                "sceneInfo": {
                    "kind": "soft-rigid-cubes-into-bag",
                    "title": "软方块与硬方块入袋遥操作",
                    "description": (
                        "Quest 双眼显示完整 W1、软方块、硬方块和实时变形袋子。两个方块统一使用软方块的抓取手型。"
                    ),
                    "controls": [
                        ["右 Grip", "按住并移动机器人右手"],
                        ["右 Trigger", "用软方块手型抓取任一方块"],
                        ["左摇杆", "转动视角"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位 W1、两个方块和袋子"],
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
                "grasp": float(self._teleop_grasp),
                "graspProfile": self._grasp_profile,
                "targetPose": target_pose,
                "targetPoses": [target_pose],
                "rigidCubePose": rigid_cube_pose,
                "softCubeCenter": self._soft_cube_center.tolist(),
                "camera": self._viewer_camera_state(),
                "viewControls": {
                    "leftThumbstickRotate": True,
                    "cameraDollyMeters": WEBXR_CAMERA_DOLLY_METERS,
                },
                "staticBoxes": self._static_boxes,
                "deformableMeshes": [
                    {
                        "mesh": self._bag_webxr_mesh_index,
                        "positions": bag_positions.reshape(-1).tolist(),
                    },
                    {
                        "mesh": self._soft_cube_webxr_mesh_index,
                        "positions": soft_cube_positions.reshape(-1).tolist(),
                    },
                ],
                "bodyPoses": [[body, *[float(value) for value in body_q[body]]] for body in self._dynamic_body_ids],
                "robotBodies": [[body, *[float(value) for value in body_q[body]]] for body in self._robot_body_ids],
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
            print("[shutdown] Soft/rigid-cube bag WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] Soft/rigid-cube bag WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        particle_q = self.state_0.particle_q.numpy()
        bag_positions = particle_q[self.bag_particle_start : self.bag_particle_end]
        soft_cube_positions = particle_q[self.soft_cube_particle_start : self.soft_cube_particle_end]
        if not np.all(np.isfinite(bag_positions)):
            raise ValueError("WebXR mixed-cube scene contains a non-finite bag position")
        if not np.all(np.isfinite(soft_cube_positions)):
            raise ValueError("WebXR mixed-cube scene contains a non-finite soft-cube position")
        if not np.all(np.isfinite(self.state_0.body_q.numpy()[self.rigid_cube_body])):
            raise ValueError("WebXR mixed-cube scene contains a non-finite rigid-cube pose")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR mixed-cube IK returned a non-finite coordinate")

    @staticmethod
    def create_parser():
        parser = cube_scene.Example.create_parser()
        parser.set_defaults(graph_capture=False)
        parser.add_argument(
            "--webxr-server",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Serve the Quest browser client and accept right-controller frames.",
        )
        parser.add_argument(
            "--webxr-host",
            default="127.0.0.1",
            help="WebXR listen address; use 127.0.0.1 with adb reverse or 0.0.0.0 for LAN access.",
        )
        parser.add_argument("--webxr-port", type=int, default=8768)
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
            help="Release the clutch anchor after this interval without a Quest frame.",
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
    print("[startup] Newton soft/rigid-cube bag viewer and device initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton soft/rigid-cube bag viewer and device initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
