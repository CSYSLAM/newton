# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Rice Cooker Open
#
# Loads Dexforce W1 and a hinged rice cooker URDF. W1's right hand follows a
# scripted IK trajectory that pushes the lid handle open by contact.
#
# Command: python -m newton.examples cloth_ricecooker_open
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton import JointTargetMode
from newton.examples.cloth.example_cloth_dexforce_bimanual_grasp_cloth import lock_joint_q_kernel
from newton.solvers import SolverMuJoCo


@wp.kernel
def copy_ik_to_joint_targets_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    joint_target_pos: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_pos[i] = ik_joint_q[0, i]


@wp.kernel
def set_indexed_joint_targets_kernel(
    target_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    push_values: wp.array[wp.float32],
    alpha: float,
    joint_target_pos: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_pos[target_indices[i]] = open_values[i] * (1.0 - alpha) + push_values[i] * alpha


ASSET_DIR = Path(__file__).resolve().parent
W1_URDF = ASSET_DIR / "DexforceW1V021" / "DexforceW1V021.urdf"
RICECOOKER_URDF = ASSET_DIR / "ricecooker" / "my_robot.urdf"

FPS = 60
SIM_SUBSTEPS = 10

TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.36, 0.46, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

COOKER_POS = wp.vec3(0.55, 0.0, TABLE_TOP_Z + 0.002)
COOKER_HANDLE_LOCAL_POS = wp.vec3(-0.34, 0.0, 0.065)

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(0.45, -0.44, 1.48),
    wp.quat(0.0237, 0.6998, 0.7126, -0.0439),
)
RIGHT_CONTACT_TF = wp.transform(
    wp.vec3(0.40, -0.12, 1.435),
    wp.quat(0.0237, 0.6998, 0.7126, -0.0439),
)
RIGHT_OPEN_TF = wp.transform(
    wp.vec3(0.61, -0.10, 1.68),
    wp.quat(0.0237, 0.6998, 0.7126, -0.0439),
)
RIGHT_RETREAT_TF = wp.transform(
    wp.vec3(0.63, -0.34, 1.62),
    wp.quat(0.0237, 0.6998, 0.7126, -0.0439),
)


class Example:
    def __init__(self, viewer, args):
        if not W1_URDF.exists():
            raise FileNotFoundError(f"Dexforce W1 asset not found: {W1_URDF}")
        if not RICECOOKER_URDF.exists():
            raise FileNotFoundError(f"Rice cooker asset not found: {RICECOOKER_URDF}")

        self.viewer = viewer
        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0

        builder = newton.ModelBuilder(gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.armature = 0.02
        builder.default_joint_cfg.target_ke = 650.0
        builder.default_joint_cfg.target_kd = 65.0
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 8.0e2
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.margin = 0.002
        builder.default_shape_cfg.gap = 0.001

        builder.add_urdf(
            str(W1_URDF),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._disable_robot_gravity(builder)
        self._configure_w1(builder)
        self._add_table(builder)

        builder.add_urdf(
            str(RICECOOKER_URDF),
            xform=wp.transform(COOKER_POS, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,
            mesh_maxhullvert=64,
        )
        self.cooker_lid_body = self._builder_body_index(builder, "RiceCooker/top")
        self.cooker_hinge_joint = self._builder_joint_index(builder, "RiceCooker/joint")
        self._configure_ricecooker(builder)
        self._add_lid_handle(builder)
        builder.add_ground_plane()

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET
        self.left_home_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_home_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_tf = self.left_home_tf
        self.right_tf = self.right_home_tf

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.hand_q_indices, self.hand_open, self.hand_push = self._build_hand_targets()
        self.motion_segments = self._build_motion_segments()
        self.setup_ik()

        self.cooker_hinge_q_start = int(self.model.joint_q_start.numpy()[self._joint_index("RiceCooker/joint")])
        self.max_lid_angle = 0.0

        self.solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=8192,
            nconmax=8192,
            iterations=30,
            ls_iterations=30,
            use_mujoco_contacts=False,
            impratio=25.0,
            cone="elliptic",
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.00, -1.15, 1.55), -20.0, 45.0)

    def _configure_w1(self, builder: newton.ModelBuilder) -> None:
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 650.0
            builder.joint_target_kd[i] = 65.0
            builder.joint_target_mode[i] = int(JointTargetMode.POSITION)
            builder.joint_effort_limit[i] = 180.0
            builder.joint_armature[i] = 0.02

        for joint_name in (*self.LEFT_HAND_JOINTS, *self.RIGHT_HAND_JOINTS):
            joint_idx = self._builder_joint_index(builder, f"DexforceW1V021/{joint_name}")
            dof_idx = int(builder.joint_qd_start[joint_idx])
            builder.joint_target_ke[dof_idx] = 950.0
            builder.joint_target_kd[dof_idx] = 75.0
            builder.joint_effort_limit[dof_idx] = 45.0
            builder.joint_armature[dof_idx] = 0.005

    def _configure_ricecooker(self, builder: newton.ModelBuilder) -> None:
        dof_idx = int(builder.joint_qd_start[self.cooker_hinge_joint])
        builder.joint_target_pos[dof_idx] = 0.0
        builder.joint_target_ke[dof_idx] = 0.0
        builder.joint_target_kd[dof_idx] = 0.0
        builder.joint_target_mode[dof_idx] = int(JointTargetMode.NONE)
        builder.joint_armature[dof_idx] = 0.01
        builder.joint_effort_limit[dof_idx] = 10.0

    def _disable_robot_gravity(self, builder: newton.ModelBuilder) -> None:
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(len(builder.body_label)):
            gravcomp_body.values[body_idx] = 1.0

        gravcomp_joint = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_joint.values is None:
            gravcomp_joint.values = {}
        for dof_idx in range(builder.joint_dof_count):
            gravcomp_joint.values[dof_idx] = True

    def _add_table(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=8.0e2, mu=1.0, density=0.0)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
        )

    def _add_lid_handle(self, builder: newton.ModelBuilder) -> None:
        handle_cfg = newton.ModelBuilder.ShapeConfig(
            density=250.0,
            ke=2.0e4,
            kd=8.0e2,
            mu=1.8,
            margin=0.001,
            gap=0.0005,
        )
        builder.add_shape_box(
            body=self.cooker_lid_body,
            xform=wp.transform(COOKER_HANDLE_LOCAL_POS, wp.quat_identity()),
            hx=0.025,
            hy=0.075,
            hz=0.016,
            cfg=handle_cfg,
            color=(0.10, 0.10, 0.10),
            label="ricecooker_lid_handle",
        )

    def setup_ik(self) -> None:
        left_tcp = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_tcp = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        self.left_pos_obj = ik.IKObjectivePosition(
            link_index=self.left_ee_index,
            link_offset=self.left_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(left_tcp)], dtype=wp.vec3),
        )
        self.left_rot_obj = ik.IKObjectiveRotation(
            link_index=self.left_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(left_tcp))], dtype=wp.vec4),
        )
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=self.right_ee_index,
            link_offset=self.right_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(right_tcp)], dtype=wp.vec3),
        )
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=self.right_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(right_tcp))], dtype=wp.vec4),
        )

        lower, upper = self._joint_limits_with_locked_dofs()
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=self.model.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=self.model.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[
                self.left_pos_obj,
                self.left_rot_obj,
                self.right_pos_obj,
                self.right_rot_obj,
                self.joint_limits_obj,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _build_motion_segments(self) -> tuple[
        tuple[float, wp.transform, wp.transform, wp.transform, wp.transform, float, float], ...
    ]:
        return (
            (0.8, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (1.8, self.left_home_tf, self.left_home_tf, self.right_home_tf, RIGHT_APPROACH_TF, 0.0, 0.0),
            (1.2, self.left_home_tf, self.left_home_tf, RIGHT_APPROACH_TF, RIGHT_CONTACT_TF, 0.0, 1.0),
            (2.4, self.left_home_tf, self.left_home_tf, RIGHT_CONTACT_TF, RIGHT_OPEN_TF, 1.0, 1.0),
            (1.2, self.left_home_tf, self.left_home_tf, RIGHT_OPEN_TF, RIGHT_RETREAT_TF, 1.0, 0.0),
            (1.5, self.left_home_tf, self.left_home_tf, RIGHT_RETREAT_TF, self.right_home_tf, 0.0, 0.0),
            (1.0, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
        )

    def _sample_script(self) -> tuple[wp.transform, wp.transform, float]:
        remaining = self.sim_time
        for duration, left_start, left_end, right_start, right_end, grasp_start, grasp_end in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / max(duration, 1.0e-6), 0.0, 1.0))
                left_tf = self._interpolate_transform(left_start, left_end, alpha)
                right_tf = self._interpolate_transform(right_start, right_end, alpha)
                grasp_alpha = grasp_start * (1.0 - alpha) + grasp_end * alpha
                return left_tf, right_tf, grasp_alpha
            remaining -= duration

        _, _, left_end, _, right_end, _, grasp_end = self.motion_segments[-1]
        return left_end, right_end, grasp_end

    def _update_control_targets(self) -> None:
        self.left_tf, self.right_tf, grasp_alpha = self._sample_script()

        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        wp.launch(
            copy_ik_to_joint_targets_kernel,
            dim=self.model.joint_dof_count,
            inputs=[self.ik_joint_q],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )
        wp.launch(
            set_indexed_joint_targets_kernel,
            dim=self.hand_q_indices.shape[0],
            inputs=[self.hand_q_indices, self.hand_open, self.hand_push, grasp_alpha],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self._update_control_targets()

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self.max_lid_angle = max(self.max_lid_angle, float(self.state_0.joint_q.numpy()[self.cooker_hinge_q_start]))

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint coordinates are not finite")

    def test_final(self) -> None:
        self.test_post_step()
        if self.max_lid_angle < 0.05:
            raise ValueError(f"Rice cooker lid did not open enough; max angle={self.max_lid_angle:.4f} rad")

    def _build_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        q_indices = []
        open_values = []
        push_values = []
        targets = {
            "HAND_THUMB2": 0.84,
            "HAND_THUMB1": 0.46,
            "HAND_INDEX": 0.70,
            "INDEX_PIP": 0.90,
            "HAND_MIDDLE": 0.60,
            "MIDDLE_PIP": 0.70,
            "HAND_RING": 0.55,
            "RING_PIP": 0.65,
            "HAND_PINKY": 0.50,
            "PINKY_PIP": 0.60,
        }

        for side in ("LEFT", "RIGHT"):
            for suffix in self.HAND_JOINT_SUFFIXES:
                joint_idx = self._joint_index(f"DexforceW1V021/{side}_{suffix}")
                q_idx = int(q_start[joint_idx])
                dof_idx = int(self.model.joint_qd_start.numpy()[joint_idx])
                open_value = float(q_home[q_idx])
                q_indices.append(dof_idx)
                open_values.append(open_value)
                push_values.append(targets.get(suffix, open_value))

        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(push_values, dtype=wp.float32, device=self.model.device),
        )

    def _joint_limits_with_locked_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.model.joint_limit_lower.numpy().copy()
        upper = self.model.joint_limit_upper.numpy().copy()
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        controlled = self._controlled_joint_labels()

        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            dof_idx = int(qd_start[joint_idx])
            lower[dof_idx] = q_home[q_idx] - 1.0e-4
            upper[dof_idx] = q_home[q_idx] + 1.0e-4

        return lower, upper

    def _build_locked_joint_arrays(self) -> tuple[wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = self._controlled_joint_labels()
        locked_q_indices = []
        locked_q_values = []

        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            locked_q_indices.append(q_idx)
            locked_q_values.append(float(q_home[q_idx]))

        return (
            wp.array(locked_q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(locked_q_values, dtype=wp.float32, device=self.model.device),
        )

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _interpolate_transform(self, tf_a: wp.transform, tf_b: wp.transform, alpha: float) -> wp.transform:
        pos_a = self._vec3_to_np(wp.transform_get_translation(tf_a))
        pos_b = self._vec3_to_np(wp.transform_get_translation(tf_b))
        quat_a = self._quat_to_np(wp.transform_get_rotation(tf_a))
        quat_b = self._quat_to_np(wp.transform_get_rotation(tf_b))
        pos = pos_a * (1.0 - alpha) + pos_b * alpha
        quat = self._slerp_quat_xyzw(quat_a, quat_b, alpha)
        return wp.transform(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), wp.quat(*quat.tolist()))

    def _slerp_quat_xyzw(self, quat_a: np.ndarray, quat_b: np.ndarray, alpha: float) -> np.ndarray:
        qa = self._normalize_quat(quat_a)
        qb = self._normalize_quat(quat_b)
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_quat(qa * (1.0 - alpha) + qb * alpha)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        scale_a = np.sin(theta_0 - theta) / sin_theta_0
        scale_b = np.sin(theta) / sin_theta_0
        return qa * scale_a + qb * scale_b

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _builder_body_index(self, builder: newton.ModelBuilder, body_name: str) -> int:
        return next(i for i, label in enumerate(builder.body_label) if label.endswith(body_name))

    def _joint_index(self, joint_name: str) -> int:
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(joint_name))

    def _builder_joint_index(self, builder: newton.ModelBuilder, joint_name: str) -> int:
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(joint_name))

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)}

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=620)
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported W1 self-collisions. This is slower but useful for debugging.",
        )
        return parser

    LEFT_ARM_JOINTS = (
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
    )
    RIGHT_ARM_JOINTS = (
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    )
    LEFT_HAND_JOINTS = (
        "LEFT_HAND_THUMB2",
        "LEFT_HAND_THUMB1",
        "LEFT_HAND_INDEX",
        "LEFT_INDEX_PIP",
        "LEFT_HAND_MIDDLE",
        "LEFT_MIDDLE_PIP",
        "LEFT_HAND_RING",
        "LEFT_RING_PIP",
        "LEFT_HAND_PINKY",
        "LEFT_PINKY_PIP",
    )
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
    HAND_JOINT_SUFFIXES = (
        "HAND_THUMB2",
        "HAND_THUMB1",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
