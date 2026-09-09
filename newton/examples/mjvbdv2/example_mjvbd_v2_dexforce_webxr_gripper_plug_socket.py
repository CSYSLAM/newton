# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate plug insertion with the W1 Pikka parallel gripper from Dexsim MR 1269.

Download the pinned model with ``uv run scripts/download_quest_w1_gripper.py``.
Start Quest with ``./scripts/start_quest_webxr_gripper_plug_socket_teleop.sh``.
The right Grip clutches wrist motion and Trigger closes the two jaws. In
experimental optical mode, the wrist follows automatically and thumb/index
separation controls opening. Looking toward the panel holds wrist and jaw
targets; looking away re-anchors the wrist before resuming. Missing hand
tracking also holds targets. Existing recording, reset and view shortcuts apply.
"""

import signal
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ik

from . import example_mjvbd_v2_dexforce_realtime_plug_socket as scene
from . import example_mjvbd_v2_dexforce_webxr_plug_socket as teleop
from ._webxr_parallel_gripper import DEFAULT_GRIPPER_URDF, ParallelGripperRetargeter
from ._webxr_w1_single_hand import _limit_finger_targets


class Example(teleop.Example):
    """Use the MR's two prismatic fingers and authored TCP in the plug scene."""

    # Tool +Z points down; the jaws close across the plug's Y dimension.
    TARGET_ROTATION = wp.quat(0.0, 1.0, 0.0, 0.0)
    RIGHT_CONTACT_BODY_KEYWORDS = ("right_finger1", "right_finger2")
    INITIAL_POSTURE: ClassVar[dict[str, float]] = {
        **scene.Example.INITIAL_POSTURE,
        # Seed the authored TCP's standby pose to avoid an elbow-branch change at startup.
        "RIGHT_J1": -0.82637965,
        "RIGHT_J2": -1.10197379,
        "RIGHT_J3": 0.28462542,
        "RIGHT_J4": 1.35057311,
        "RIGHT_J5": 1.04604870,
        "RIGHT_J6": -0.14626876,
        "RIGHT_J7": 1.07239250,
    }

    def __init__(self, viewer, args):
        if not Path(args.robot_urdf).is_file():
            raise FileNotFoundError("Download the model first: uv run scripts/download_quest_w1_gripper.py")
        if not np.isfinite(args.gripper_speed) or args.gripper_speed <= 0:
            raise ValueError("--gripper-speed must be finite and positive")
        self._gripper = ParallelGripperRetargeter(
            Path(args.robot_urdf), closed_span=args.pinch_closed_span, open_span=args.pinch_open_span
        )
        self.OPEN_HAND_JOINTS = dict(zip(self._gripper.joint_names, self._gripper.upper, strict=True))
        self.GRASP_HAND_JOINTS = dict(zip(self._gripper.joint_names, self._gripper.lower, strict=True))
        super().__init__(viewer, args)

    def _decompose_hand_collision_meshes(self, builder: newton.ModelBuilder) -> None:
        """Use the MR's authored convex jaw parts without repeating V-HACD."""
        colliders = [
            shape
            for shape in range(self.robot_shape_end)
            if int(builder.shape_body[shape]) >= 0
            and any(
                name in builder.body_label[int(builder.shape_body[shape])].lower()
                for name in self.RIGHT_CONTACT_BODY_KEYWORDS
            )
            and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ]
        if len(colliders) < 4:
            raise ValueError("Expected the MR's two convex collision parts per jaw")

    def _build_ik(self) -> None:
        """Target the authored right_gripper_tcp frame instead of a five-finger midpoint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(scene.ROBOT_BASE_POSITION, scene.ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self._set_builder_posture(builder)
        self.ik_model = builder.finalize(device=self.device)
        tcp_body = self._body_index(self.ik_model.body_label, "right_gripper_tcp")
        self.position_objective = ik.IKObjectivePosition(
            tcp_body, wp.vec3(0.0), wp.array([scene.HAND_STANDBY_POSITION], dtype=wp.vec3, device=self.device)
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            tcp_body,
            wp.quat_identity(),
            wp.array([self._quat_vector(self.TARGET_ROTATION)], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._ik_joint_limits()
        limit_objective = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_lock_indices, self.ik_lock_values = self._ik_locked_q()
        source, destination = self._joint_coordinate_mapping()
        self.ik_source_indices = wp.array(source, dtype=wp.int32, device=self.device)
        self.scene_destination_indices = wp.array(destination, dtype=wp.int32, device=self.device)

    def _ik_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Keep right-arm DOFs free regardless of the imported URDF robot name."""
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        starts = self.ik_model.joint_q_start.numpy()
        dofs = self.ik_model.joint_qd_start.numpy()
        for joint, label in enumerate(self.ik_model.joint_label):
            if label.rsplit("/", 1)[-1] in self.RIGHT_ARM:
                continue
            for offset in range(int(dofs[joint + 1] - dofs[joint])):
                lower[int(dofs[joint]) + offset] = q[int(starts[joint]) + offset] - 1.0e-4
                upper[int(dofs[joint]) + offset] = q[int(starts[joint]) + offset] + 1.0e-4
        return lower, upper

    def _ik_locked_q(self) -> tuple[wp.array, wp.array]:
        """Lock the body, left arm and gripper while solving right-arm IK."""
        q = self.ik_model.joint_q.numpy()
        starts = self.ik_model.joint_q_start.numpy()
        indices = [
            index
            for joint, label in enumerate(self.ik_model.joint_label)
            if label.rsplit("/", 1)[-1] not in self.RIGHT_ARM
            for index in range(int(starts[joint]), int(starts[joint + 1]))
        ]
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(q[indices], dtype=wp.float32, device=self.device),
        )

    def _set_ik_target(self, position: wp.vec3, rotation: wp.quat | None = None) -> None:
        super()._set_ik_target(position, self.TARGET_ROTATION if rotation is None else rotation)

    def _init_optical_hand(self, urdf_path: Path) -> None:
        self._configure_optical_hand(self._gripper, self.hand_q_indices.numpy().tolist())

    def _optical_grasp_fraction(self, desired: np.ndarray) -> float:
        return self._gripper.closure(desired)

    def _write_gripper_coordinates(self, desired: np.ndarray, destination: wp.array[float]) -> None:
        """Rate-limit each linear jaw identically in controller and optical modes."""
        self._optical_command.assign(desired)
        wp.launch(
            _limit_finger_targets,
            2,
            [
                self.state_0.joint_q,
                self.hand_q_indices,
                self._optical_command,
                float(self.args.gripper_speed * self.frame_dt),
                destination,
            ],
            device=self.device,
        )

    def _write_hand_pose(self, grasp_alpha: float, destination: wp.array[float]) -> None:
        if not hasattr(self, "state_0"):
            super()._write_hand_pose(grasp_alpha, destination)
        else:
            self._write_gripper_coordinates(self._gripper.coordinates(grasp_alpha), destination)

    def _write_optical_fingers(self, destination: wp.array[float]) -> None:
        self._write_gripper_coordinates(self._optical_target, destination)

    def _scene_info(self) -> dict:
        info = super()._scene_info()
        info.update(
            kind="plug-socket-gripper",
            title="W1 二指夹插头遥操",
            description="MR 1269 的 Pikka 二指夹。支持手柄和裸手遥操。",
        )
        info["controls"][0] = ["裸手模式", "右手手腕自动跟随。拇指与食指间距控制二指夹开合"]
        info["controls"][2] = ["右 Trigger", "按下闭合二指夹。松开张开"]
        return info

    @staticmethod
    def _trajectory_path(path_value: Path | None) -> Path:
        if path_value is not None:
            return Path(path_value).expanduser()
        return Path("recordings") / f"webxr_gripper_plug_socket_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def test_final(self) -> None:
        """Check finite robot state and jaw limits after a test-mode run."""
        super().test_final()
        q = self.state_0.joint_q.numpy()[self.hand_q_indices.numpy()]
        if np.any(q < self._gripper.lower - 1.0e-4) or np.any(q > self._gripper.upper + 1.0e-4):
            raise ValueError("Parallel-gripper coordinates exceeded their URDF limits")
        if not np.isclose(q[0], q[1], atol=1.0e-4):
            raise ValueError("Parallel-gripper mimic coordinates diverged")

    @staticmethod
    def create_parser():
        parser = teleop.Example.create_parser()
        parser.description = __doc__
        parser.set_defaults(robot_urdf=DEFAULT_GRIPPER_URDF, webxr_port=8772)
        parser.add_argument("--gripper-speed", type=float, default=0.08, help="Maximum speed of each jaw [m/s].")
        parser.add_argument(
            "--pinch-closed-span", type=float, default=0.015, help="Thumb/index span for closed jaws [m]."
        )
        parser.add_argument("--pinch-open-span", type=float, default=0.10, help="Thumb/index span for open jaws [m].")
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
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    active_example.append(example)
    example.exit_requested = exit_signal_requested[0]
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
