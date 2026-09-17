# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate W1 paper-bag packing with Quest controllers or optical hands.

Run ``./scripts/start_quest_webxr_w1_bag_packing_teleop.sh``. Each Grip
clutches its arm and each Trigger closes its parallel gripper. Experimental
optical mode follows both wrists and maps thumb/index spacing to jaw opening.
Paper and snacks remain dynamic; the automatic packing trajectory is disabled.
"""

import argparse
import signal
import time
import weakref
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from . import example_mjvbd_v2_w1_bag_packing as scene
from ._webxr_gripper_input import GripperInput
from ._webxr_parallel_gripper import ParallelGripperRetargeter
from ._webxr_teleop import JsonlTrajectoryRecorder, LatestXRFrame, Pose, WebXRServer, pack_scene_geometry
from ._webxr_w1_head import OBSERVER_VIEW_MODE, W1HeadController

HANDS = ("left", "right")
WORKSPACE_LOWER = np.array((0.12, -0.72, scene.TABLE_Z + 0.015))
WORKSPACE_UPPER = np.array((0.95, 0.72, 1.80))


def _normals(vertices, indices):
    triangles = np.asarray(indices).reshape(-1, 3)
    points = vertices[triangles]
    faces = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], faces)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    return normals


def _close(server, recorder):
    try:
        recorder.close()
    finally:
        server.stop()


class Example(scene.Example):
    """Drive both V030 arms and grippers while retaining physical bag/snack contact."""

    reset_in_place = True
    _initial_bag_yaw = np.pi / 2
    _initial_gripper_openings = (scene.OPEN, scene.SNACK_OPENINGS["can"])

    def __init__(self, viewer, args):
        if args.ik_iterations < 1 or args.record_flush_every < 1:
            raise ValueError("IK iterations and recording flush interval must be positive")
        for name in ("xr_stale_seconds", "xr_translation_scale", "xr_max_translation", "arm_speed", "gripper_speed"):
            if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        args.no_cuda_graph = args.no_cuda_graph or not args.graph_capture
        super().__init__(viewer, args)
        self.xr_state = LatestXRFrame()
        self.teleoperation_active = self.simulation_active = True
        self.exit_requested = False
        self.view_mode = OBSERVER_VIEW_MODE
        self.episode_index = self.episode_frame = 0
        self.last_reset_source = None
        self._buttons = self._record_request = None
        self._seen_input = False
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_ik = wp.clone(self.ik_q)
        self.finger_indices = {
            hand: np.array([self.coords[f"{hand.upper()}_FINGER{i}_JOINT"] for i in (1, 2)]) for hand in HANDS
        }
        self.arm_indices = np.array([self.coords[f"{hand.upper()}_J{i}"] for hand in HANDS for i in range(1, 8)])
        self._arm_velocity = np.minimum(args.arm_speed, self.ik_model.joint_velocity_limit.numpy()[self.arm_indices])
        self.head_control = W1HeadController(
            self.model,
            self.model.device,
            wp.quat_identity(),
            body_names=("head_yaw_j1_link", "head_pitch_j2_link"),
            eye_position=(0.10, 0, 0),
        )
        self.inputs = {}
        self._right_grasp_body = None
        bodies, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        for hand, body in zip(HANDS, self.ee, strict=True):
            mapper = ParallelGripperRetargeter(scene.ASSET, side=hand)
            minimum = scene.SUPPORT_OPENING if hand == "left" else scene.SNACK_OPENINGS["can"]
            mapper.lower = np.maximum(mapper.lower, minimum)
            mapper.upper = np.minimum(mapper.upper, scene.OPEN)
            mapper.reset(q[self.finger_indices[hand]])
            self.inputs[hand] = GripperInput(
                hand,
                mapper,
                self._tcp_pose(bodies[body]),
                q[self.finger_indices[hand]],
                translation_scale=args.xr_translation_scale,
                max_translation=args.xr_max_translation,
            )
        output = args.trajectory_output or Path("recordings") / f"webxr_w1_bag_packing_{time.time_ns()}.jsonl"
        self.trajectory_recorder = JsonlTrajectoryRecorder(
            output,
            {
                "scene": "w1-bag-packing",
                "robotUrdf": str(scene.ASSET),
                "frameDtSeconds": self.frame_dt,
                "simulationSubsteps": args.substeps,
                "robotCoordinateIndices": list(range(self.robot_coords)),
                "robotJointLabels": list(self.ik_model.joint_label),
                "bagTriangleIndices": self.faces.reshape(-1).tolist(),
                "snackBodies": self.objects,
            },
            flush_every=args.record_flush_every,
        )
        self._bag_meshes = []
        self._static_boxes = []
        geometry = self._build_webxr_geometry() if args.webxr_server else None
        self.webxr_server = WebXRServer(
            self.xr_state,
            host=args.webxr_host,
            port=args.webxr_port,
            geometry_payload=geometry,
            require_simulation_ready=True,
        )
        self._resource_finalizer = weakref.finalize(self, _close, self.webxr_server, self.trajectory_recorder)
        self._publish_scene_state(bodies)
        if args.webxr_server:
            self.webxr_server.start()
            print(f"Quest W1 packing: http://{args.webxr_host}:{args.webxr_port}/", flush=True)

    @staticmethod
    def _tcp_pose(body) -> Pose:
        transform = wp.transform(*body)
        return Pose(np.asarray(wp.transform_point(transform, scene.TCP)), np.asarray(body[3:7]).copy())

    def _hold_inputs(self) -> None:
        bodies, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        for hand, body in zip(HANDS, self.ee, strict=True):
            self.inputs[hand].hold(self._tcp_pose(bodies[body]), q[self.finger_indices[hand]])

    def _consume_controls(self) -> bool:
        if self.xr_state.consume_shutdown() is not None:
            self.exit_requested = True
        request = self.xr_state.consume_teleoperation_mode()
        if request is not None:
            _, self.teleoperation_active, self.simulation_active = request
            self._hold_inputs()
            self._buttons = None
            if not self.teleoperation_active:
                self.trajectory_recorder.pause()
        if self.teleoperation_active and self.xr_state.consume_reset() is not None:
            self.reset_physics(source="webxr")
        return self.simulation_active and not self.exit_requested

    def _prepare_frame(self):
        frame = (
            self.xr_state.snapshot(max_age_seconds=self.args.xr_stale_seconds) if self.teleoperation_active else None
        )
        if frame is not None and frame.visibility_state != "visible":
            frame = None
        if frame is not None:
            if frame.view_mode != self.view_mode:
                self._hold_inputs()
                self.view_mode = frame.view_mode
            request = (frame.stream_id, frame.recording_request)
            if request != self._record_request:
                if frame.recording_request:
                    self.trajectory_recorder.toggle()
                self._record_request = request
            controller = frame.controllers.get("right")
            if controller is not None:
                buttons = (frame.stream_id, controller.is_button_pressed(4), controller.is_button_pressed(3))
                if self._buttons is not None and self._buttons[0] == buttons[0]:
                    if buttons[1] and not self._buttons[1]:
                        self.trajectory_recorder.toggle()
                    if buttons[2] and not self._buttons[2]:
                        self.reset_physics(source="quest-controller")
                        self._buttons = buttons
                        return None
                self._buttons = buttons
            else:
                self._buttons = None
            if (frame.controllers or frame.hands) and not self._seen_input:
                self._seen_input = True
                if self.args.record_on_connect:
                    self.trajectory_recorder.start()
        else:
            self._buttons = None
        self.head_control.set_desired_pose(self.view_mode, None if frame is None else frame.head_pose)
        bodies, q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
        for hand, body in zip(HANDS, self.ee, strict=True):
            control = self.inputs[hand]
            control.update(frame, self._tcp_pose(bodies[body]), q[self.finger_indices[hand]])
            control.position = np.clip(control.position, WORKSPACE_LOWER, WORKSPACE_UPPER)
        if (
            frame is not None
            and (
                (frame.input_mode == "controllers" and "right" in frame.controllers)
                or (frame.input_mode == "hands" and self.inputs["right"].status == "tracking")
            )
            and frame.visibility_state == "visible"
        ):
            self._update_right_grasp_limit(bodies)
        return frame

    def _update_right_grasp_limit(self, bodies) -> None:
        """Latch the nearby snack's grasp opening until the right hand releases it."""
        control = self.inputs["right"]
        closure = control.mapper.closure(control.jaws)
        if closure <= 0.05:
            self._right_grasp_body = None
        elif self._right_grasp_body is None:
            tcp = self._tcp_pose(bodies[self.ee[1]]).position
            distances = np.linalg.norm(bodies[self.objects, :3] - tcp, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 0.12:
                self._right_grasp_body = self.objects[nearest]
        kind = "can" if self._right_grasp_body is None else self.kinds[self.objects.index(self._right_grasp_body)]
        control.mapper.lower[:] = scene.SNACK_OPENINGS[kind]
        control.jaws = control.mapper.coordinates(closure)

    def _solve_teleop_ik(self) -> None:
        previous = self.ik_q.numpy()[0]
        for index, hand in enumerate(HANDS):
            control = self.inputs[hand]
            self.positions[index].set_target_position(0, wp.vec3(*control.position))
            self.rotations[index].set_target_rotation(0, wp.vec4(*control.orientation))
            self.elbows[index].weight = 0.005
        if any(control.retargeter.active for control in self.inputs.values()):
            self._step_ik(self.args.ik_iterations)
        solved = self.ik_q.numpy()[0]
        if not np.isfinite(solved).all():
            solved = previous.copy()
        q = previous.copy()
        indices = self.arm_indices
        q[indices] += np.clip(
            solved[indices] - previous[indices], -self._arm_velocity * self.frame_dt, self._arm_velocity * self.frame_dt
        )
        for hand, control in self.inputs.items():
            indices = self.finger_indices[hand]
            step = self.args.gripper_speed * self.frame_dt
            q[indices] += np.clip(control.jaws - previous[indices], -step, step)
        self.frame_start.assign(previous)
        self.frame_end.assign(np.clip(q, self.lower, self.upper))
        self.head_control.write_targets(self.frame_end, self.frame_dt)
        wp.copy(self.ik_q.flatten(), self.frame_end)

    def step(self) -> None:
        if not self._consume_controls():
            return
        frame = self._prepare_frame()
        self._solve_teleop_ik()
        self._advance_physics()
        self.frame += 1
        self.episode_frame += 1
        self.sim_time = self.frame * self.frame_dt
        # This read also completes the first GPU frame before advertising readiness.
        bodies = self.state_0.body_q.numpy()
        self.webxr_server.mark_simulation_ready()
        if self.trajectory_recorder.recording:
            self.trajectory_recorder.append(
                {
                    "frame": self.frame,
                    "episode": self.episode_index,
                    "episodeFrame": self.episode_frame,
                    "simulationTimeSeconds": self.sim_time,
                    "inputMode": self.inputs["left"].mode,
                    "xrStreamId": None if frame is None else frame.stream_id,
                    "xrSequence": None if frame is None else frame.sequence,
                    "xrControllerSpace": None if frame is None else frame.controller_space,
                    "headPose": None
                    if frame is None or frame.head_pose is None
                    else [*frame.head_pose.position.tolist(), *frame.head_pose.orientation.tolist()],
                    "xrControllers": {}
                    if frame is None
                    else {
                        side: {
                            "pose": [*control.pose.position.tolist(), *control.pose.orientation.tolist()],
                            "clutch": control.clutch,
                            "trigger": control.trigger_value,
                        }
                        for side, control in frame.controllers.items()
                    },
                    "xrHands": {}
                    if frame is None
                    else {
                        side: {
                            "pose": [*hand.pose.position.tolist(), *hand.pose.orientation.tolist()],
                            "joints": hand.joints.tolist(),
                            "activation": hand.activation,
                            "enabled": hand.enabled,
                        }
                        for side, hand in frame.hands.items()
                    },
                    "targetPoses": self._target_poses(),
                    "gripperJointTargets": {side: control.jaws.tolist() for side, control in self.inputs.items()},
                    "robotJointQ": self.state_0.joint_q.numpy()[: self.robot_coords].tolist(),
                    "bodyPoses": bodies.tolist(),
                    "bagParticleQ": self.state_0.particle_q.numpy().tolist(),
                    "bagParticleQd": self.state_0.particle_qd.numpy().tolist(),
                }
            )
        if self.webxr_server.running:
            self._publish_scene_state(bodies)

    def reset_physics(self, *, source: str) -> None:
        """Restore robot, paper and snacks together without reallocating the solver."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.ik_q, self._initial_ik)
        wp.copy(self.frame_start, self._initial_ik.flatten())
        wp.copy(self.frame_end, self.frame_start)
        self._hold_inputs()
        self._right_grasp_body = None
        self.inputs["right"].mapper.lower[:] = scene.SNACK_OPENINGS["can"]
        self.head_control.reset()
        self.episode_index += 1
        self.episode_frame = 0
        self.last_reset_source = source
        self.trajectory_recorder.append_event(
            {"event": "scene-reset", "episode": self.episode_index, "frame": self.frame, "source": source}
        )
        self._publish_scene_state(self.state_0.body_q.numpy())

    def _target_poses(self):
        return {
            side: [*control.position.tolist(), *control.orientation.tolist()] for side, control in self.inputs.items()
        }

    def _build_webxr_geometry(self) -> bytes:
        meshes, shapes, mesh_ids = [], [], {}
        body_ids = self.model.shape_body.numpy()
        flags, types = self.model.shape_flags.numpy(), self.model.shape_type.numpy()
        transforms, scales = self.model.shape_transform.numpy(), self.model.shape_scale.numpy()
        colors = self.model.shape_color.numpy()
        for index, shape_source in enumerate(self.model.shape_source):
            if not int(flags[index]) & int(newton.ShapeFlags.VISIBLE):
                continue
            body, transform = int(body_ids[index]), transforms[index]
            role = "snack" if body in self.objects else "robot"
            source = shape_source
            scale = scales[index]
            mesh_key = ("mesh", id(source))
            # Plain snack primitives need triangle meshes for the Quest client.
            if body >= 0 and int(types[index]) == int(newton.GeoType.BOX):
                source = newton.Mesh.create_box(*scale, compute_inertia=False)
                mesh_key, scale = ("primitive", index), np.ones(3)
            elif body >= 0 and int(types[index]) == int(newton.GeoType.CYLINDER):
                source = newton.Mesh.create_cylinder(
                    radius=float(scale[0]),
                    half_height=float(scale[1]),
                    barrel_radius=float(scale[2]),
                    up_axis=newton.Axis.Z,
                    compute_inertia=False,
                )
                mesh_key, scale = ("primitive", index), np.ones(3)
            if isinstance(source, newton.Mesh):
                if mesh_key not in mesh_ids:
                    vertices = np.asarray(source.vertices, dtype=np.float32)
                    mesh_ids[mesh_key] = len(meshes)
                    meshes.append((vertices, _normals(vertices, source.indices), source.indices))
                shapes.append(
                    {
                        "body": body,
                        "role": role,
                        "mesh": mesh_ids[mesh_key],
                        "position": transform[:3].tolist(),
                        "orientation": transform[3:].tolist(),
                        "scale": scale.tolist(),
                        "color": colors[index].tolist(),
                    }
                )
            elif body < 0 and int(types[index]) == int(newton.GeoType.BOX):
                self._static_boxes.append(
                    {
                        "role": "table",
                        "position": transform[:3].tolist(),
                        "orientation": transform[3:].tolist(),
                        "scale": (2 * scales[index]).tolist(),
                        "color": colors[index].tolist(),
                    }
                )
        vertices = self.state_0.particle_q.numpy()
        for indices, color in (
            (self.faces[: self.paper_faces], (0.66, 0.46, 0.25)),
            (self.faces[self.paper_faces :], (0.42, 0.32, 0.16)),
        ):
            mesh = len(meshes)
            meshes.append((vertices, _normals(vertices, indices), indices))
            self._bag_meshes.append(mesh)
            shapes.append(
                {
                    "body": -1,
                    "role": "bag",
                    "mesh": mesh,
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                    "color": color,
                    "doubleSided": True,
                }
            )
        return pack_scene_geometry(meshes, shapes)

    def _publish_scene_state(self, bodies) -> None:
        positions = self.state_0.particle_q.numpy().reshape(-1).tolist()
        self.webxr_server.publish_scene(
            {
                "type": "scene-state",
                "version": 1,
                "sceneKind": "w1-bag-packing",
                "sceneInfo": {
                    "kind": "w1-bag-packing",
                    "title": "W1 纸袋装零食遥操作",
                    "description": "双手控制 V030 二指夹: 扶起纸袋、夹取零食并放入袋中。",
                    "controls": [
                        ["左右 Grip", "按住移动对应手臂"],
                        ["左右 Trigger", "控制对应夹爪闭合"],
                        ["裸手模式 (实验)", "手腕控制双臂, 拇指和食指间距控制夹爪"],
                        ["看向面板", "暂停手势跟随; 离开面板后重新对齐"],
                        ["X / 视角按钮", "切换机器人第一人称"],
                        ["A / 录制按钮", "开始或暂停录制"],
                        ["右摇杆按下 / 复位按钮", "复位机器人、纸袋和零食"],
                    ],
                },
                "frame": self.frame,
                "episode": self.episode_index,
                "episodeFrame": self.episode_frame,
                "lastResetSource": self.last_reset_source,
                "simulationTimeSeconds": self.sim_time,
                "phase": " / ".join(f"{side}:{control.status}" for side, control in self.inputs.items()),
                "recording": self.trajectory_recorder.recording,
                "recordedFrames": self.trajectory_recorder.sample_count,
                "handTrackingEnabled": True,
                "handTrackingSides": list(HANDS),
                "inputMode": self.inputs["left"].mode,
                "handTrackingState": {side: control.status for side, control in self.inputs.items()},
                "handTrackingActivation": {
                    side: None if control.token is None else control.token[1] for side, control in self.inputs.items()
                },
                "targetPoses": list(self._target_poses().values()),
                "grasps": {side: control.mapper.closure(control.jaws) for side, control in self.inputs.items()},
                "camera": {"position": [1.90, -1.9, 1.85], "front": [-0.7, 0.67, -0.25], "up": [0, 0, 1]},
                "firstPersonCamera": self.head_control.camera_state(bodies),
                "firstPersonHiddenBodies": list(self.head_control.hidden_body_ids),
                "viewMode": self.view_mode,
                "neckJointTargets": self.head_control.targets.tolist(),
                "viewControls": {"leftThumbstickRotate": True, "firstPersonEnabled": True},
                "bodyPoses": [[i, *pose.tolist()] for i, pose in enumerate(bodies)],
                "staticBoxes": self._static_boxes,
                "deformableMeshes": [{"mesh": mesh, "positions": positions} for mesh in self._bag_meshes],
            }
        )

    def render(self) -> None:
        if self._consume_controls():
            super().render()
        else:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()

    def close(self) -> None:
        self._resource_finalizer()

    def test_post_step(self) -> None:
        """Check finite physical state and bounded joints without requiring scripted packing."""
        for values in (self.state_0.body_q, self.state_0.particle_q, self.ik_q):
            if not np.isfinite(values.numpy()).all():
                raise AssertionError("Nonfinite WebXR packing state")
        q = self.ik_q.numpy()[0]
        if np.any(q < self.lower - 1e-5) or np.any(q > self.upper + 1e-5):
            raise AssertionError("Teleoperation exceeded joint limits")
        if np.any(self.model.particle_inv_mass.numpy() <= 0):
            raise AssertionError("Paper must remain unpinned")
        if np.any(self.model.body_flags.numpy()[self.objects] & int(newton.BodyFlags.KINEMATIC)):
            raise AssertionError("Snacks must remain dynamic")

    def test_final(self) -> None:
        """Validate a user-driven run without imposing the automatic trajectory."""
        self.test_post_step()

    @staticmethod
    def create_parser():
        parser = scene.Example.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument("--graph-capture", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--ik-iterations", type=int, default=24)
        parser.add_argument("--arm-speed", type=float, default=2.0, help="Maximum arm joint speed [rad/s].")
        parser.add_argument("--gripper-speed", type=float, default=0.08, help="Maximum speed of each jaw [m/s].")
        parser.add_argument("--webxr-server", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--webxr-host", default="127.0.0.1")
        parser.add_argument("--webxr-port", type=int, default=8773)
        parser.add_argument("--xr-stale-seconds", type=float, default=0.25)
        parser.add_argument("--xr-translation-scale", type=float, default=1.0)
        parser.add_argument("--xr-max-translation", type=float, default=0.8)
        parser.add_argument("--trajectory-output", type=Path, default=None)
        parser.add_argument("--record-on-connect", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--record-flush-every", type=int, default=60)
        return parser


if __name__ == "__main__":
    active = []
    exit_pending = [False]

    def _request_exit(_signum, _frame):
        exit_pending[0] = True
        for example in active:
            example.exit_requested = True

    signal.signal(signal.SIGINT, _request_exit)
    signal.signal(signal.SIGTERM, _request_exit)
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    active.append(example)
    example.exit_requested = exit_pending[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
