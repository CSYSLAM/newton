# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate the full-W1 plastic inflatable-bag scene from Quest.

The right Quest grip clutches the W1 right wrist, the right trigger drives
all right-hand finger coordinates between the source scene's open and
recorded grasp poses, A toggles trajectory recording, B realigns the stereo
view, X switches to the W1 eye camera with head tracking, and the right
thumbstick resets the physical scene in place. The Quest renderer receives all
216 deforming bag vertices rather than a rigid proxy.

Use the guarded USB workflow from the repository root::

    ./scripts/start_quest_webxr_bag_teleop.sh
    ./scripts/stop_quest_webxr_bag_teleop.sh
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

from . import example_vbd_mjvbd_v2_dexforce_recorded_plastic_inflatable_bag_pick_release_final00 as bag_scene
from ._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    RelativePoseRetargeter,
    WebXRServer,
    pack_scene_geometry,
)
from ._webxr_w1_head import FIRST_PERSON_VIEW_MODE, OBSERVER_VIEW_MODE, W1HeadController, serialize_head_pose

robot_reference = bag_scene.robot_reference
bag_recorder = robot_reference.hand_reference.recorder
hand_recorder = bag_recorder.hand_recorder

FPS = robot_reference.FPS
DEFAULT_STALE_SECONDS = 0.25
TARGET_POSITION_MIN = np.array((-0.75, -3.45, 0.85), dtype=np.float32)
TARGET_POSITION_MAX = np.array((0.35, -2.15, 1.85), dtype=np.float32)
IDENTITY_ROTATION = np.eye(3, dtype=np.float32)
BAG_COLOR = (0.86, 0.68, 0.34)
QUEST_A_BUTTON_INDEX = 4
QUEST_THUMBSTICK_BUTTON_INDEX = 3
RELEASE_TRIGGER_THRESHOLD = 0.05


def _close_resources(server: WebXRServer, recorder: JsonlTrajectoryRecorder) -> None:
    recorder.close()
    server.stop()


class Example(bag_scene.Example):
    """Drive the W1 right hand against the deformable pneumatic bag."""

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
        self.view_mode = OBSERVER_VIEW_MODE
        self._startup_started_at = time.perf_counter()

        super().__init__(viewer, args)

        initial_tcp = self._tcp(self.state_0, self.right_body)
        self._initial_target_position = self._vec3_array(wp.transform_get_translation(initial_tcp))
        self._initial_target_orientation = self._quat_array(wp.transform_get_rotation(initial_tcp))
        self._teleop_position = self._initial_target_position.copy()
        self._teleop_orientation = self._initial_target_orientation.copy()
        self._teleop_grasp = 0.0
        self._open_finger_q = np.asarray(self.hand_open.numpy(), dtype=np.float32)
        self._grasp_finger_q = np.asarray(self.hand_grasp.numpy(), dtype=np.float32)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik_q = wp.clone(self.ik_q)
        self._head_controller = W1HeadController(self.model, self.device, self.base_rot)
        self._initial_edge_rest_angle = wp.clone(self.authored_edge_rest_angle)
        self._initial_edge_bending_properties = wp.clone(self.authored_edge_bending_properties)
        self._robot_q_host_indices = self._robot_coordinate_indices()
        self._robot_body_ids = tuple(range(self.robot_body_end))
        self._robot_segments = self._build_robot_segments()
        self._bag_local_indices = (
            np.asarray(self.bag_triangle_indices.numpy(), dtype=np.int64) - self.bag_particle_start
        ).astype(np.uint32)
        self._static_boxes = [
            {
                "position": self._vec3_array(hand_recorder.TABLE_POS).tolist(),
                "orientation": self._quat_array(hand_recorder.TABLE_ROTATION).tolist(),
                "scale": [2.0 * float(value) for value in hand_recorder.TABLE_HALF_EXTENTS],
                "color": [0.35, 0.42, 0.48],
            }
        ]

        trajectory_path = self._trajectory_path(args.trajectory_output)
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            trajectory_path,
            {
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": self.sim_substeps,
                "physicsSolver": "SolverMJVBDV2",
                "robotUrdf": str(self.urdf_path),
                "robotJointLabels": list(self.model.joint_label),
                "robotCoordinateIndices": list(self._robot_q_host_indices),
                "headCoordinateIndices": self._head_controller.coordinate_indices_host.tolist(),
                "bagParticleStart": self.bag_particle_start,
                "bagParticleCount": self.bag_particle_end - self.bag_particle_start,
                "bagTriangleIndices": self._bag_local_indices.reshape(-1).tolist(),
                "plasticityEnabled": self.plasticity_enabled,
                "pneumaticMode": self.pneumatic_mode_name,
            },
            flush_every=int(args.record_flush_every),
        )
        geometry_payload = None
        if args.webxr_server:
            self._trace_startup("WebXR W1/bag geometry packing started")
            geometry_payload = self._build_webxr_geometry()
            self._trace_startup("WebXR W1/bag geometry packing complete")
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
                f"Quest WebXR plastic-bag server: http://{args.webxr_host}:{args.webxr_port}/\n"
                f"ADB USB route: adb reverse tcp:{args.webxr_port} tcp:{args.webxr_port}\n"
                f"W1/bag geometry: {len(geometry_payload) / (1024 * 1024):.1f} MiB\n"
                f"Deforming bag: {self.bag_particle_end - self.bag_particle_start} vertices\n"
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
        return Path("recordings") / f"webxr_plastic_inflatable_bag_{timestamp}.jsonl"

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

    def _build_webxr_geometry(self) -> bytes:
        """Pack complete W1 visuals plus the bag's initial mesh topology."""
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

        bag_positions = np.asarray(
            self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end],
            dtype=np.float32,
        )
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
                "color": list(BAG_COLOR),
            }
        )
        roles = {str(shape["role"]) for shape in shapes}
        if not {"robot", "bag"}.issubset(roles):
            raise RuntimeError(f"WebXR bag scene is missing required mesh roles: {sorted(roles)}")
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
            raise RuntimeError("The W1 bag scene did not provide robot coordinates")
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
            position = robot_reference.CAMERA_POS
            pitch = np.deg2rad(robot_reference.CAMERA_PITCH)
            yaw = np.deg2rad(robot_reference.CAMERA_YAW)
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

    def _set_hand_material(self, *, released: bool) -> None:
        if self.release_material_applied == released:
            return
        if released:
            friction_value = robot_reference.hand_reference.RELEASE_FRICTION
            material_index = robot_reference._SOFT_MATERIAL_RELEASE
        else:
            friction_value = bag_recorder.CONTACT_MU
            material_index = robot_reference._SOFT_MATERIAL_GRASP
        friction = self.model.shape_material_mu.numpy()
        friction[self.right_hand_shapes] = friction_value
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_mu = friction_value
        self.soft_contact_material_index.fill_(material_index)
        self.release_material_applied = released

    def _prepare_frame(self) -> None:
        """Retarget the newest right Quest controller pose into W1 IK."""
        frame = self.xr_state.snapshot(max_age_seconds=self.xr_stale_seconds) if self.teleoperation_active else None
        controller = None if frame is None else frame.controllers.get("right")
        if not self.teleoperation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
        elif frame is not None:
            if frame.view_mode != self.view_mode:
                self.view_mode = frame.view_mode
                self.retargeter.reset()
                print(f"Quest view mode changed to {self.view_mode}", flush=True)
            self._head_controller.set_desired_pose(self.view_mode, frame.head_pose)
        elif self.view_mode == FIRST_PERSON_VIEW_MODE:
            self._head_controller.set_desired_pose(self.view_mode, None)
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

        self.active_phase_name = self.phase
        self._set_hand_material(released=self._teleop_grasp <= RELEASE_TRIGGER_THRESHOLD)
        target_tcp = wp.transform(
            wp.vec3(*[float(value) for value in self._teleop_position]),
            wp.quat(*[float(value) for value in self._teleop_orientation]),
        )
        self.current_target_root = self._tcp_to_root(target_tcp)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(target_tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(target_tcp)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=robot_reference.RUNTIME_IK_ITERATIONS)
        wp.launch(
            robot_reference._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.launch(
            robot_reference._copy_joint_q,
            self.model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.device,
        )
        desired_finger_q = self._open_finger_q + self._teleop_grasp * (self._grasp_finger_q - self._open_finger_q)
        self.desired_finger_q.assign(desired_finger_q)
        wp.launch(
            robot_reference._limit_right_finger_target_step,
            self.hand_indices.shape[0],
            [
                self.frame_q_start,
                self.hand_indices,
                self.desired_finger_q,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.right_hand_shape_mask,
                self.max_finger_step,
                self.max_finger_contact_step,
                self.frame_q_end,
            ],
            device=self.device,
        )
        self._head_controller.write_targets(self.frame_q_end, self.frame_dt)

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
            print(f"Plastic-bag teleoperation resumed in existing process (request {request_id})", flush=True)
        elif simulation_active:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
            self.phase = "teleoperation_standby"
            self.trajectory_recorder.pause()
            print(
                f"Plastic-bag controls disarmed while CUDA simulation remains active (request {request_id})",
                flush=True,
            )
        else:
            self.view_mode = OBSERVER_VIEW_MODE
            self._head_controller.set_desired_pose(self.view_mode, None)
            self.phase = "teleoperation_parked"
            self.trajectory_recorder.pause()
            print(
                f"Plastic-bag teleoperation parked without destroying CUDA state (request {request_id})",
                flush=True,
            )

    def reset_physics(self, *, source: str) -> None:
        """Restore W1, pneumatic bag, plasticity, and contact material in place."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.model.edge_rest_angle, self._initial_edge_rest_angle)
        wp.copy(self.model.edge_bending_properties, self._initial_edge_bending_properties)
        wp.copy(self.frame_q_start, self._initial_state.joint_q)
        wp.copy(self.frame_q_end, self._initial_state.joint_q)
        wp.copy(self.ik_q, self._initial_ik_q)
        self.desired_finger_q.assign(self._open_finger_q)
        self._teleop_position = self._initial_target_position.copy()
        self._teleop_orientation = self._initial_target_orientation.copy()
        self._teleop_grasp = 0.0
        self._head_controller.reset()
        self._set_hand_material(released=False)
        self.retargeter.reset()
        self.maximum_soft_contact_count.zero_()
        self.maximum_body_particle_contact_count.zero_()
        self.minimum_volume_ratio = 1.0
        self.maximum_pressure = bag_recorder.BAG_REFERENCE_ABSOLUTE_PRESSURE
        self.maximum_root_position_error = 0.0
        self.maximum_root_angle_error = 0.0
        self.lifted_bag_center_z = None
        self.active_phase_index = -1
        self.current_target_root = self._tcp_to_root(
            wp.transform(
                wp.vec3(*[float(value) for value in self._initial_target_position]),
                wp.quat(*[float(value) for value in self._initial_target_orientation]),
            )
        )
        self.phase = "scene_reset"
        self.active_phase_name = self.phase
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
        print(f"Plastic-bag scene reset in place (episode {self.episode_index}, source={source})", flush=True)

    def step(self) -> None:
        self._consume_teleoperation_mode()
        if not self.simulation_active:
            return
        if self.teleoperation_active:
            self._consume_webxr_reset()
        if self._startup_sync_pending:
            self._trace_startup("first plastic-bag simulation step started")
        super().step()
        if self._startup_sync_pending:
            self._trace_startup("first plastic-bag step complete; device synchronization started")
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
        bag_positions = np.asarray(
            self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end],
            dtype=np.float32,
        )
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        target_pose = [*self._teleop_position.tolist(), *self._teleop_orientation.tolist()]
        if should_record:
            joint_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float32)
            bag_velocities = np.asarray(
                self.state_0.particle_qd.numpy()[self.bag_particle_start : self.bag_particle_end],
                dtype=np.float32,
            )
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
                    "headPose": serialize_head_pose(None if input_frame is None else input_frame.head_pose),
                    "neckJointTargets": self._head_controller.targets.tolist(),
                    "phase": self.phase,
                    "targetPose": target_pose,
                    "grasp": float(self._teleop_grasp),
                    "robotJointQ": [float(joint_q[index]) for index in self._robot_q_host_indices],
                    "bagParticleQ": bag_positions.reshape(-1).tolist(),
                    "bagParticleQd": bag_velocities.reshape(-1).tolist(),
                    "bagVolume": volume,
                    "bagAbsolutePressure": pressure,
                }
            )
        if should_publish:
            self._publish_scene_state(body_q, bag_positions, target_pose, volume, pressure)

    def _publish_scene_state(
        self,
        body_q: np.ndarray,
        bag_positions: np.ndarray,
        target_pose: list[float],
        volume: float,
        pressure: float,
    ) -> None:
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "plastic-inflatable-bag",
                "sceneInfo": {
                    "kind": "plastic-inflatable-bag",
                    "title": "充气塑料袋遥操作",
                    "description": "Quest 双眼显示完整 W1、桌面和实时变形袋子。也可切换机器人眼睛第一人称。",
                    "controls": [
                        ["右 Grip", "按住并移动机器人右手"],
                        ["右 Trigger", "控制右手全部手指抓握"],
                        ["左摇杆", "观察模式下转动视角"],
                        ["X / 视角按钮", "切换观察模式与机器人第一人称"],
                        ["A", "开始 / 暂停 / 继续轨迹录制"],
                        ["B", "用当前头部位姿重新对齐 Newton 相机"],
                        ["右摇杆按下", "原地复位 W1、袋子和塑性状态"],
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
                "targetPoses": [target_pose],
                "bagVolume": volume,
                "bagAbsolutePressure": pressure,
                "camera": self._viewer_camera_state(),
                "firstPersonCamera": self._head_controller.camera_state(body_q),
                "firstPersonHiddenBodies": list(self._head_controller.hidden_body_ids),
                "viewMode": self.view_mode,
                "neckJointTargets": self._head_controller.targets.tolist(),
                "viewControls": {
                    "leftThumbstickRotate": True,
                    "firstPersonEnabled": True,
                },
                "staticBoxes": self._static_boxes,
                "deformableMeshes": [
                    {
                        "mesh": self._bag_webxr_mesh_index,
                        "positions": bag_positions.reshape(-1).tolist(),
                    }
                ],
                "bodyPoses": [[body, *[float(value) for value in body_q[body]]] for body in self._robot_body_ids],
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
            print("[shutdown] Plastic-bag WebXR and trajectory cleanup started", flush=True)
            self._resource_finalizer()
            print("[shutdown] Plastic-bag WebXR and trajectory cleanup complete", flush=True)

    def test_final(self) -> None:
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected MJVBDV2 backend: {self.solver.features.backend}")
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        if not np.all(np.isfinite(positions)):
            raise ValueError("WebXR plastic-bag scene contains a non-finite particle position")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("WebXR plastic-bag IK returned a non-finite coordinate")
        plastic_offset = self.model.edge_rest_angle.numpy() - self._initial_edge_rest_angle.numpy()
        maximum_offset = float(np.max(np.abs(plastic_offset), initial=0.0))
        if not np.isfinite(maximum_offset) or maximum_offset > self.plastic_max_angle + 1.0e-5:
            raise ValueError(f"WebXR plastic-bag curvature is invalid: {maximum_offset}")

    @staticmethod
    def create_parser():
        parser = bag_scene.Example.create_parser()
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
        parser.add_argument("--webxr-port", type=int, default=8767)
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
    print("[startup] Newton plastic-bag viewer and simulation device initialization started", flush=True)
    viewer, args = newton.examples.init(parser)
    print("[startup] Newton plastic-bag viewer and simulation device initialization complete", flush=True)
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
