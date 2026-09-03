# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the realtime Dexforce chair scene from a Quest headset.

Newton keeps the full W1 and the free 8 kg WAIC chair in the existing
MJVBDV2 scene. Each Quest grip clutches the corresponding arm, each trigger
controls that hand's fingers, A pauses or resumes recording, B realigns the
stereo scene, and the right thumbstick resets physics in place.

Use the guarded USB workflow from the repository root::

    ./scripts/start_quest_webxr_chair_teleop.sh

Leave immersive mode and park the CUDA simulation without destroying its
context::

    ./scripts/stop_quest_webxr_chair_teleop.sh
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

from . import example_mjvbd_v2_dexforce_realtime_push_chair as push_chair
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)

FPS = push_chair.FPS
DEFAULT_STALE_SECONDS = 0.25
TARGET_POSITION_MIN = np.array((-0.10, -0.70, 0.35), dtype=np.float32)
TARGET_POSITION_MAX = np.array((1.00, 0.70, 1.40), dtype=np.float32)
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3
HANDS = ("left", "right")


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(push_chair.Example):
    """Drive both W1 arms against the physical chair with Quest controllers."""

    reset_in_place = True

    def __init__(self, viewer, args):
        stale_seconds = float(args.xr_stale_seconds)
        if not np.isfinite(stale_seconds) or stale_seconds <= 0.0:
            raise ValueError("xr_stale_seconds must be finite and greater than zero")
        self.xr_stale_seconds = stale_seconds
        self.xr_state = LatestXRFrame()
        self.retargeters = {
            hand: RelativePoseRetargeter(
                translation_scale=float(args.xr_translation_scale),
                max_translation=float(args.xr_max_translation),
            )
            for hand in HANDS
        }
        self._teleop_positions = {
            "left": self._vec3_array(push_chair.LEFT_TCP_STANDBY),
            "right": self._vec3_array(push_chair.RIGHT_TCP_STANDBY),
        }
        self._teleop_orientations = {
            "left": self._quat_array(push_chair.LEFT_TCP_ROTATION),
            "right": self._quat_array(push_chair.RIGHT_TCP_ROTATION),
        }
        self._teleop_grasps = dict.fromkeys(HANDS, 0.0)
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

        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._robot_q_host_indices = self._robot_coordinate_indices()
        trajectory_path = self._trajectory_path(args.trajectory_output)
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            trajectory_path,
            {
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": self.sim_substeps,
                "physicsSolver": "SolverMJVBDV2",
                "robotUrdf": str(self.robot_urdf),
                "chairUrdf": str(self.chair_urdf),
                "robotJointLabels": list(self.model.joint_label),
                "robotCoordinateIndices": list(self._robot_q_host_indices),
            },
            flush_every=int(args.record_flush_every),
        )
        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR W1/chair geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR W1/chair geometry packing complete")
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
        self._robot_body_ids = tuple(range(self.robot_body_end))
        self._robot_segments = self._build_robot_segments()

        if args.webxr_server:
            self.webxr_server.start()
            print(
                f"Quest WebXR chair server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/chair geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
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
        return Path("recordings") / f"webxr_push_chair_{timestamp}.jsonl"

    def _initialize_pose(self) -> None:
        """Start with a stationary base and both open hands near the chair."""
        self._set_ik_targets(
            push_chair.LEFT_TCP_STANDBY,
            push_chair.RIGHT_TCP_STANDBY,
            push_chair.LEFT_TCP_ROTATION,
            push_chair.RIGHT_TCP_ROTATION,
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=push_chair.INITIAL_IK_ITERATIONS)
        self._restore_locked_ik_q()
        self._copy_ik_to_scene(self.model.joint_q)
        self._write_hand_pose(0.0, self.model.joint_q)
        self._write_root_pose(push_chair.ROBOT_BASE_GRASP, self.model.joint_q)
        self.model.joint_qd.zero_()

    def _build_hand_pose_control(self) -> None:
        """Keep independent scalar grasp arrays for the two Quest triggers."""
        q_start = self.model.joint_q_start.numpy()
        index_groups: dict[str, list[int]] = {}
        open_groups: dict[str, list[float]] = {}
        grasp_groups: dict[str, list[float]] = {}
        all_indices: list[int] = []
        all_open: list[float] = []
        all_grasp: list[float] = []
        for hand in HANDS:
            side = hand.upper()
            indices: list[int] = []
            open_values: list[float] = []
            grasp_values: list[float] = []
            for suffix, open_value in self.OPEN_HAND_JOINTS.items():
                name = f"{side}_{suffix}"
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
                grasp_values.append(float(self.GRASP_HAND_JOINTS[suffix]))
            index_groups[hand] = indices
            open_groups[hand] = open_values
            grasp_groups[hand] = grasp_values
            all_indices.extend(indices)
            all_open.extend(open_values)
            all_grasp.extend(grasp_values)

        self.hand_q_indices_by_side = {
            hand: wp.array(index_groups[hand], dtype=wp.int32, device=self.device) for hand in HANDS
        }
        self.hand_q_open_by_side = {
            hand: wp.array(open_groups[hand], dtype=wp.float32, device=self.device) for hand in HANDS
        }
        self.hand_q_grasp_by_side = {
            hand: wp.array(grasp_groups[hand], dtype=wp.float32, device=self.device) for hand in HANDS
        }
        self.hand_q_indices = wp.array(all_indices, dtype=wp.int32, device=self.device)
        self.hand_q_open = wp.array(all_open, dtype=wp.float32, device=self.device)
        self.hand_q_grasp = wp.array(all_grasp, dtype=wp.float32, device=self.device)

    def _write_hand_pose(self, grasp_alpha: float, destination: wp.array[float]) -> None:
        self._write_independent_hand_pose(grasp_alpha, grasp_alpha, destination)

    def _write_independent_hand_pose(
        self,
        left_grasp: float,
        right_grasp: float,
        destination: wp.array[float],
    ) -> None:
        for hand, grasp_alpha in (("left", left_grasp), ("right", right_grasp)):
            indices = self.hand_q_indices_by_side[hand]
            wp.launch(
                push_chair._write_indexed_lerp,
                indices.shape[0],
                [
                    indices,
                    self.hand_q_open_by_side[hand],
                    self.hand_q_grasp_by_side[hand],
                    float(grasp_alpha),
                    destination,
                ],
                device=self.device,
            )

    def _robot_coordinate_indices(self) -> tuple[int, ...]:
        articulation_start = self.model.articulation_start.numpy()
        articulation_end = self.model.articulation_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        coordinates: list[int] = []
        for articulation in self.robot_articulations:
            for joint in range(int(articulation_start[articulation]), int(articulation_end[articulation])):
                coordinates.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        if not coordinates:
            raise RuntimeError("The W1 scene did not provide robot coordinates")
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
        """Pack full visible W1 and exact chair meshes for Quest rendering."""
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

        for shape in range(self.model.shape_count):
            body = int(shape_bodies[shape])
            role = "robot" if 0 <= body < self.robot_body_end else "chair" if body == self.chair_body else None
            source = self.model.shape_source[shape]
            if (
                role is None
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
        if not {"robot", "chair"}.issubset(roles):
            raise RuntimeError(f"WebXR chair scene is missing required mesh roles: {sorted(roles)}")
        return pack_scene_geometry(meshes, shapes)

    def _viewer_camera_state(self) -> dict[str, list[float]]:
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "get_front") and hasattr(camera, "get_up"):
            position = camera.pos
            front = camera.get_front()
            up = camera.get_up()
        else:
            position = push_chair.CAMERA_POSITION
            pitch = np.deg2rad(push_chair.CAMERA_PITCH)
            yaw = np.deg2rad(push_chair.CAMERA_YAW)
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
        """Retarget both newest Quest controller poses into the fixed W1 base."""
        frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds) if self.teleoperation_active else None
        controllers = {} if frame is None else frame.controllers
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
                self._teleop_positions[hand] = np.clip(target.position, TARGET_POSITION_MIN, TARGET_POSITION_MAX)
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

        self._set_ik_targets(
            wp.vec3(*[float(value) for value in self._teleop_positions["left"]]),
            wp.vec3(*[float(value) for value in self._teleop_positions["right"]]),
            wp.quat(*[float(value) for value in self._teleop_orientations["left"]]),
            wp.quat(*[float(value) for value in self._teleop_orientations["right"]]),
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._restore_locked_ik_q()

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        self._copy_ik_to_scene(self.frame_q_end)
        self._write_independent_hand_pose(
            self._teleop_grasps["left"],
            self._teleop_grasps["right"],
            self.frame_q_end,
        )
        self._write_root_pose(push_chair.ROBOT_BASE_GRASP, self.frame_q_end)

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
            print(f"Chair teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Chair controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(f"Chair teleoperation parked without destroying CUDA state (request {request_id})", flush=True)

    def reset_physics(self, *, source: str) -> None:
        """Restore the initial W1/chair physics state without rebuilding CUDA."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.ik_q, self._initial_ik_q)
        self._teleop_positions = {
            "left": self._vec3_array(push_chair.LEFT_TCP_STANDBY),
            "right": self._vec3_array(push_chair.RIGHT_TCP_STANDBY),
        }
        self._teleop_orientations = {
            "left": self._quat_array(push_chair.LEFT_TCP_ROTATION),
            "right": self._quat_array(push_chair.RIGHT_TCP_ROTATION),
        }
        self._teleop_grasps = dict.fromkeys(HANDS, 0.0)
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
        print(f"Chair scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA chair simulation step started")
        super().step()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA chair step complete; device synchronization started")
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
        chair_pose = [float(value) for value in body_q[self.chair_body]]
        target_poses = [
            [
                *[float(value) for value in self._teleop_positions[hand]],
                *[float(value) for value in self._teleop_orientations[hand]],
            ]
            for hand in HANDS
        ]
        if should_record:
            joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
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
                    "targetPoses": {hand: target_poses[index] for index, hand in enumerate(HANDS)},
                    "grasps": dict(self._teleop_grasps),
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "chairPose": chair_pose,
                }
            )
        if should_publish:
            self._publish_scene_state(body_q, chair_pose, target_poses)

    def _publish_scene_state(
        self,
        body_q: np.ndarray,
        chair_pose: list[float],
        target_poses: list[list[float]],
    ) -> None:
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "push-chair",
                "sceneInfo": {
                    "kind": "push-chair",
                    "title": "双手推椅遥操作",
                    "description": "Quest 双眼显示完整 W1 与真实椅子网格, 左右手柄分别控制机器人双臂和手指。",
                    "controls": [
                        ["左右 Grip", "按住并移动对应机器人手臂"],
                        ["左右 Trigger", "控制对应机器人手指抓握"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位 W1 和椅子"],
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
                "chairPose": chair_pose,
                "camera": self._viewer_camera_state(),
                "bodyPoses": [
                    [body, *[float(value) for value in body_q[body]]]
                    for body in (*self._robot_body_ids, self.chair_body)
                ],
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
            print("[shutdown] Chair WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] Chair WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("WebXR chair scene contains a non-finite body pose")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR bimanual IK returned a non-finite coordinate")

    @staticmethod
    def create_parser():
        parser = push_chair.Example.create_parser()
        parser.set_defaults(graph_capture=False)
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
        parser.add_argument("--webxr-port", type=int, default=8766)
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
    print("[startup] Newton chair viewer and CUDA initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton chair viewer and CUDA initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
