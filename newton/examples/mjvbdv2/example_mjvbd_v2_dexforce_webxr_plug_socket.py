# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the realtime Dexforce plug/socket scene from a Quest headset.

The Quest Browser supplies the right-controller pose while Newton keeps the
existing 60 FPS MJVBDV2 contact simulation. Hold the right grip button to
clutch relative wrist motion, use the trigger to pinch, press A to pause or
resume trajectory recording, press the right thumbstick to reset the physical
scene in place, and press B to align the full W1 stereo scene with the live
Newton desktop camera at the current headset pose.

The one-command USB workflow avoids certificate setup::

    ./scripts/start_quest_webxr_teleop.sh

Use ``./scripts/stop_quest_webxr_teleop.sh`` to make the browser leave immersive
mode before stopping Newton. The launcher defaults to ``ViewerNull`` so CUDA
physics cannot stall the desktop OpenGL display; set
``NEWTON_WEBXR_VIEWER=gl`` only when a local debug window is explicitly needed.
The manual Quest URL is ``http://127.0.0.1:8765/``.
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

from . import example_mjvbd_v2_dexforce_realtime_plug_socket as plug_socket
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)

FPS = plug_socket.FPS
DEFAULT_STALE_SECONDS = 0.25
TARGET_POSITION_MIN = np.array((-0.28, -0.38, 0.78), dtype=np.float32)
TARGET_POSITION_MAX = np.array((0.38, 0.28, 1.38), dtype=np.float32)
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(plug_socket.Example):
    """Drive the physical plug insertion scene from a Quest right controller."""

    reset_in_place = True

    def __init__(self, viewer, args):
        self._startup_started_at = time.perf_counter()
        stale_seconds = float(args.xr_stale_seconds)
        if not np.isfinite(stale_seconds) or stale_seconds <= 0.0:
            raise ValueError("xr_stale_seconds must be finite and greater than zero")
        self.xr_stale_seconds = stale_seconds
        self.xr_state = LatestXRFrame()
        self.retargeter = RelativePoseRetargeter(
            translation_scale=float(args.xr_translation_scale),
            max_translation=float(args.xr_max_translation),
        )
        self._teleop_position = np.array(
            [float(component) for component in plug_socket.HAND_STANDBY_POSITION],
            dtype=np.float32,
        )
        self._teleop_orientation = np.array(
            [float(plug_socket.HAND_TARGET_ROTATION[index]) for index in range(4)],
            dtype=np.float32,
        )
        self._teleop_grasp = 0.0
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

        super().__init__(viewer, args)

        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._robot_q_host_indices = tuple(int(index) for index in self.robot_q_indices.numpy())
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
            },
            flush_every=int(args.record_flush_every),
        )
        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR geometry packing complete")
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
                f"Quest WebXR server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/plug/socket geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
                "Reset scene: Newton Reset button, web page button, or right thumbstick press\n"
                f"Trajectory: {self.trajectory_recorder.path}",
                flush=True,
            )

    def _trace_startup(self, phase: str) -> None:
        elapsed = time.perf_counter() - self._startup_started_at
        print(f"[startup +{elapsed:8.3f}s] {phase}", flush=True)

    @staticmethod
    def _trajectory_path(path_value: Path | None) -> Path:
        if path_value is not None:
            return Path(path_value).expanduser()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return Path("recordings") / f"webxr_plug_socket_{timestamp}.jsonl"

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
        """Return finite unit vertex normals for a Newton mesh."""
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)
        supplied = mesh.normals
        if supplied is not None:
            normals = np.asarray(supplied, dtype=np.float32).reshape(-1, 3).copy()
            if normals.shape == vertices.shape and np.all(np.isfinite(normals)):
                lengths = np.linalg.norm(normals, axis=1)
                valid = lengths > 1.0e-8
                if np.all(valid):
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
        """Pack the visible W1, plug, and socket meshes for Quest rendering."""
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

        shape_roles = [
            *((shape, "robot") for shape in range(self.robot_shape_end)),
            (self.plug_shape, "plug"),
            (self.socket_shape, "socket"),
        ]
        for shape, role in shape_roles:
            body = int(shape_bodies[shape])
            source = self.model.shape_source[shape]
            if (
                not int(shape_flags[shape]) & visible_flag
                or int(shape_types[shape]) != int(newton.GeoType.MESH)
                or not isinstance(source, newton.Mesh)
            ):
                continue
            if role == "robot" and (body < 0 or body >= self.robot_body_end):
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
        if not {"robot", "plug", "socket"}.issubset(roles):
            raise RuntimeError(f"WebXR scene is missing required mesh roles: {sorted(roles)}")
        return pack_scene_geometry(meshes, shapes)

    def _viewer_camera_state(self) -> dict[str, list[float]]:
        """Return the live desktop camera basis for Quest eye-space alignment."""
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "get_front") and hasattr(camera, "get_up"):
            position = camera.pos
            front = camera.get_front()
            up = camera.get_up()
        else:
            position = plug_socket.CAMERA_POSITION
            pitch = np.deg2rad(plug_socket.CAMERA_PITCH)
            yaw = np.deg2rad(plug_socket.CAMERA_YAW)
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
        """Consume only the newest Quest pose and solve the existing W1 IK."""
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
            self.phase = "quest_clutched" if controller.clutch else "quest_idle"

        self._set_ik_target(
            wp.vec3(*[float(value) for value in self._teleop_position]),
            wp.quat(*[float(value) for value in self._teleop_orientation]),
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._restore_locked_ik_q()

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        self._copy_ik_to_scene(self.frame_q_end)
        self._write_hand_pose(self._teleop_grasp, self.frame_q_end)
        self._write_root_pose(self.frame_q_end)

    def _process_controller_buttons(self, stream_id: str, sequence: int, controller) -> bool:
        """Handle record/reset edges once per newly received Quest frame."""
        new_stream = stream_id != self._last_input_stream
        new_frame = new_stream or sequence != self._last_input_sequence
        if not new_frame:
            return False

        record_pressed = controller.is_button_pressed(QUEST_A_BUTTON_INDEX)
        reset_pressed = controller.is_button_pressed(QUEST_THUMBSTICK_BUTTON_INDEX)
        reset_requested = not new_stream and reset_pressed and not self._reset_button_pressed
        if not self._has_seen_controller:
            self._has_seen_controller = True
            if self.args.record_on_connect:
                self.trajectory_recorder.start()
        elif not new_stream and record_pressed and not self._record_button_pressed:
            recording = self.trajectory_recorder.toggle()
            print(f"Quest trajectory recording {'resumed' if recording else 'paused'}", flush=True)

        self._record_button_pressed = record_pressed
        self._reset_button_pressed = reset_pressed
        self._last_input_stream = stream_id
        self._last_input_sequence = sequence
        return reset_requested

    def _consume_webxr_reset(self) -> None:
        """Apply a browser reset request on Newton's simulation thread."""
        if self.xr_state.consume_reset() is not None:
            self.reset_physics(source="webxr")

    def _consume_teleoperation_mode(self) -> None:
        """Park or resume without tearing down the current CUDA context."""
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
            print(f"Teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Teleoperation controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(
                f"Teleoperation parked without destroying CUDA state (request {request_id})",
                flush=True,
            )

    def reset_physics(self, *, source: str) -> None:
        """Restore the initial physics state without rebuilding WebXR resources."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.ik_q, self._initial_ik_q)

        self._teleop_position = np.array(
            [float(component) for component in plug_socket.HAND_STANDBY_POSITION],
            dtype=np.float32,
        )
        self._teleop_orientation = np.array(
            [float(plug_socket.HAND_TARGET_ROTATION[index]) for index in range(4)],
            dtype=np.float32,
        )
        self._teleop_grasp = 0.0
        self.retargeter.reset()
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
        print(f"Physical scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        """Advance physics, then publish and optionally record the resulting state."""
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA simulation step started")
        super().step()
        if self._startup_sync_pending:
            self._trace_startup("first CUDA simulation step complete; device synchronization started")
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
        plug_pose = [float(value) for value in body_q[self.plug_body]]
        target_pose = [
            *[float(value) for value in self._teleop_position],
            *[float(value) for value in self._teleop_orientation],
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
                    "targetPose": target_pose,
                    "grasp": float(self._teleop_grasp),
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "plugPose": plug_pose,
                }
            )

        if should_publish:
            self._publish_scene_state(body_q=body_q, plug_pose=plug_pose, target_pose=target_pose)

    def _publish_scene_state(
        self,
        *,
        body_q: np.ndarray | None = None,
        plug_pose: list[float] | None = None,
        target_pose: list[float] | None = None,
    ) -> None:
        """Publish the current state, including an immediate paused reset."""
        if body_q is None:
            body_q = np.asarray(self.state_0.body_q.numpy(), dtype=np.float32)
        if plug_pose is None:
            plug_pose = [float(value) for value in body_q[self.plug_body]]
        if target_pose is None:
            target_pose = [
                *[float(value) for value in self._teleop_position],
                *[float(value) for value in self._teleop_orientation],
            ]
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "plug-socket",
                "sceneInfo": {
                    "kind": "plug-socket",
                    "title": "插头遥操作",
                    "description": "Quest 双眼显示完整 W1、插头和插座, 右手柄控制机器人右手完成抓取与插接。",
                    "controls": [
                        ["右 Grip", "按住并移动机器人右手"],
                        ["右 Trigger", "控制拇指和食指捏合"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位物理场景"],
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
                "targetPose": target_pose,
                "plugPose": plug_pose,
                "camera": self._viewer_camera_state(),
                "bodyPoses": [
                    [body, *[float(value) for value in body_q[body]]]
                    for body in (*self._robot_body_ids, self.plug_body)
                ],
                "robotBodies": [[body, *[float(value) for value in body_q[body]]] for body in self._robot_body_ids],
                "robotSegments": self._robot_segments,
            }
        )

    def render(self) -> None:
        """Consume control requests while paused and finish the current frame."""
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
        """Flush trajectory data and stop the WebXR worker."""
        if self._resource_finalizer.alive:
            print("[shutdown] WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        """Verify the idle teleoperation path without requiring a real headset."""
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("WebXR plug/socket scene contains a non-finite body pose")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR realtime IK returned a non-finite coordinate")

    @staticmethod
    def create_parser():
        """Create command-line options for Quest teleoperation and recording."""
        parser = plug_socket.Example.create_parser()
        parser.set_defaults(graph_capture=False)
        parser.add_argument(
            "--webxr-server",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Serve the Quest browser client and accept controller frames.",
        )
        parser.add_argument(
            "--webxr-host",
            default="127.0.0.1",
            help="WebXR listen address; use 127.0.0.1 with adb reverse or 0.0.0.0 for LAN access.",
        )
        parser.add_argument("--webxr-port", type=int, default=8765)
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
            help="Automatically start trajectory recording when the first right-controller frame arrives.",
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
    print("[startup] Newton viewer and CUDA initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton viewer and CUDA initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
