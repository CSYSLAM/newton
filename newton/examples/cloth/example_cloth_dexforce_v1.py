# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Dexforce Grasp Ball
#
# Loads the DexforceW1V021 URDF and closes the right hand around a small
# rigid ball. The motion is intentionally scripted so the URDF loading,
# contact setup, and hand joint targets are easy to inspect and modify.
#
# Command: python -m newton.examples cloth_dexforce_grasp_ball
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCo


@wp.kernel
def interpolate_joint_targets(
    pose_a: wp.array[wp.float32],
    pose_b: wp.array[wp.float32],
    alpha: float,
    joint_target_pos: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_pos[i] = pose_a[i] * (1.0 - alpha) + pose_b[i] * alpha


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.ball_radius = args.ball_radius
        self.ball_initial_height = 1.365
        self.max_ball_height = self.ball_initial_height

        builder = newton.ModelBuilder(gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.armature = 0.02
        builder.default_joint_cfg.target_ke = 450.0
        builder.default_joint_cfg.target_kd = 45.0
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 8.0e2
        builder.default_shape_cfg.mu = 1.6
        builder.default_shape_cfg.margin = 0.002
        builder.default_shape_cfg.gap = 0.001

        urdf_path = Path(__file__).with_name("DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

        self._configure_robot(builder)
        self._add_ball_scene(builder)

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.pose_open, self.pose_grasp, self.pose_lift = self._build_pose_targets()
        wp.copy(self.control.joint_target_pos, self.pose_open)

        self.solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=2048,
            nconmax=2048,
            iterations=25,
            ls_iterations=25,
            use_mujoco_contacts=False,
            impratio=20.0,
            cone="elliptic",
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.65, -1.55, 1.65), -22.0, 24.0)

        self.capture()

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 450.0
            builder.joint_target_kd[i] = 45.0
            builder.joint_effort_limit[i] = 150.0
            builder.joint_armature[i] = 0.02

        for joint_name in self.RIGHT_HAND_JOINTS:
            joint_idx = self._joint_index(builder, joint_name)
            dof_idx = builder.joint_qd_start[joint_idx]
            builder.joint_target_ke[dof_idx] = 900.0
            builder.joint_target_kd[dof_idx] = 70.0
            builder.joint_effort_limit[dof_idx] = 35.0
            builder.joint_armature[dof_idx] = 0.005

    def _add_ball_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 2.0e4
        table_cfg.kd = 8.0e2
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, -0.975, 1.305), wp.quat_identity()),
            hx=0.16,
            hy=0.12,
            hz=0.012,
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
        )

        ball_cfg = newton.ModelBuilder.ShapeConfig()
        ball_cfg.density = 180.0
        ball_cfg.ke = 2.0e4
        ball_cfg.kd = 8.0e2
        ball_cfg.mu = 2.0
        ball_cfg.margin = 0.001

        self.ball_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -0.955, self.ball_initial_height), wp.quat_identity()),
            label="grasp_ball",
        )
        builder.add_shape_sphere(
            self.ball_body,
            radius=self.ball_radius,
            cfg=ball_cfg,
            color=(0.95, 0.38, 0.16),
        )

        builder.add_ground_plane()

    def _build_pose_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        pose_open = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        pose_open[:40] = np.asarray(self.model.joint_q.numpy()[:40], dtype=np.float32)
        pose_grasp = pose_open.copy()
        pose_lift = pose_open.copy()

        right_hand_targets = {
            "RIGHT_HAND_THUMB2": 1.05,
            "RIGHT_HAND_THUMB1": 0.58,
            "RIGHT_HAND_INDEX": 0.88,
            "RIGHT_INDEX_PIP": 1.12,
            "RIGHT_HAND_MIDDLE": 0.95,
            "RIGHT_MIDDLE_PIP": 1.18,
            "RIGHT_HAND_RING": 0.92,
            "RIGHT_RING_PIP": 1.12,
            "RIGHT_HAND_PINKY": 0.84,
            "RIGHT_PINKY_PIP": 1.02,
        }
        for joint_name, target in right_hand_targets.items():
            pose_grasp[self._joint_target_index(joint_name)] = target

        pose_lift[:] = pose_grasp

        return (
            wp.array(pose_open, dtype=wp.float32, device=self.model.device),
            wp.array(pose_grasp, dtype=wp.float32, device=self.model.device),
            wp.array(pose_lift, dtype=wp.float32, device=self.model.device),
        )

    def _joint_index(self, builder: newton.ModelBuilder, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(suffix))

    def _joint_target_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        joint_idx = next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))
        return int(self.model.joint_qd_start.numpy()[joint_idx])

    def set_joint_targets(self) -> None:
        if self.sim_time < 1.0:
            pose_a = self.pose_open
            pose_b = self.pose_open
            alpha = 0.0
        elif self.sim_time < 2.2:
            pose_a = self.pose_open
            pose_b = self.pose_grasp
            alpha = (self.sim_time - 1.0) / 1.2
        elif self.sim_time < 6.0:
            pose_a = self.pose_grasp
            pose_b = self.pose_grasp
            alpha = 0.0
        else:
            pose_a = self.pose_lift
            pose_b = self.pose_lift
            alpha = 0.0

        wp.launch(
            interpolate_joint_targets,
            dim=self.model.joint_dof_count,
            inputs=[pose_a, pose_b, float(np.clip(alpha, 0.0, 1.0))],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.set_joint_targets()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        ball_z = float(self.state_0.body_q.numpy()[self.ball_body, 2])
        self.max_ball_height = max(self.max_ball_height, ball_z)

    def test_final(self) -> None:
        ball_pos = self.state_0.body_q.numpy()[self.ball_body, :3]
        if not np.all(np.isfinite(ball_pos)):
            raise ValueError(f"Ball position is not finite: {ball_pos}")
        if not (-0.35 < ball_pos[0] < 0.35 and -1.2 < ball_pos[1] < -0.65 and 1.1 < ball_pos[2] < 1.6):
            raise ValueError(f"Ball ended outside the expected grasp volume: {ball_pos}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=360)
        parser.add_argument(
            "--ball-radius",
            type=float,
            default=0.038,
            help="Radius of the grasped ball [m].",
        )
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported URDF self-collisions. This is slower but useful for debugging.",
        )
        return parser

    RIGHT_HAND_JOINTS = (
        "RIGHT_HAND_THUMB2",
        "RIGHT_HAND_THUMB1",
        "RIGHT_HAND_INDEX",
        "RIGHT_INDEX_PIP",
        "RIGHT_HAND_MIDDLE",
        "RIGHT_MIDDLE_PIP",
        "RIGHT_HAND_RING",
        "RIGHT_RING_PIP",
        "RIGHT_HAND_PINKY",
        "RIGHT_PINKY_PIP",
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
