# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual Place Tablecloth
#
# Starts with a small tablecloth on the right side of the table. Dexforce W1
# grasps two corners, lifts until the cloth hangs, backs away, rotates, walks
# to the table center, turns toward the table, then lowers and feeds the cloth
# forward so it lays across the tabletop.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_place_tablecloth
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.mjvbd


@wp.kernel
def lock_joint_q_kernel(
    joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[0, locked_q_indices[i]] = locked_q_values[i]


@wp.kernel
def copy_ik_to_joint_q_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = ik_joint_q[0, i]


@wp.kernel
def set_indexed_joint_q_kernel(
    q_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    grasp_values: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[q_indices[i]] = open_values[i] * (1.0 - alpha) + grasp_values[i] * alpha


@wp.kernel
def interpolate_joint_positions_kernel(
    joint_q_start: wp.array[wp.float32],
    joint_q_end: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = joint_q_start[i] * (1.0 - alpha) + joint_q_end[i] * alpha


@wp.kernel
def update_joint_velocity_kernel(
    joint_q_prev: wp.array[wp.float32],
    joint_q_next: wp.array[wp.float32],
    inv_dt: float,
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    joint_qd[i] = (joint_q_next[i] - joint_q_prev[i]) * inv_dt


@wp.func
def smoothstep(u: float) -> float:
    x = wp.clamp(u, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.kernel
def set_free_root_motion_kernel(
    root_q_start: int,
    root_q0: wp.array[wp.float32],
    base_time: float,
    retreat_distance: float,
    retreat_time: float,
    turn_out_radians: float,
    turn_out_time: float,
    center_shift_y: float,
    center_move_time: float,
    turn_in_radians: float,
    turn_in_time: float,
    approach_distance: float,
    approach_time: float,
    joint_q_out: wp.array[wp.float32],
):
    t0 = retreat_time
    t1 = t0 + turn_out_time
    t2 = t1 + center_move_time
    t3 = t2 + turn_in_time

    retreat_u = smoothstep(base_time / wp.max(retreat_time, 1.0e-6))
    turn_out_u = smoothstep((base_time - t0) / wp.max(turn_out_time, 1.0e-6))
    center_u = smoothstep((base_time - t1) / wp.max(center_move_time, 1.0e-6))
    turn_in_u = smoothstep((base_time - t2) / wp.max(turn_in_time, 1.0e-6))
    approach_u = smoothstep((base_time - t3) / wp.max(approach_time, 1.0e-6))

    yaw = turn_out_radians * turn_out_u + turn_in_radians * turn_in_u
    q_yaw = wp.quat(0.0, 0.0, wp.sin(0.5 * yaw), wp.cos(0.5 * yaw))
    q0 = wp.quat(root_q0[3], root_q0[4], root_q0[5], root_q0[6])
    q = q_yaw * q0

    joint_q_out[root_q_start + 0] = root_q0[0] - retreat_distance * retreat_u + approach_distance * approach_u
    joint_q_out[root_q_start + 1] = root_q0[1] + center_shift_y * center_u
    joint_q_out[root_q_start + 2] = root_q0[2]
    joint_q_out[root_q_start + 3] = q[0]
    joint_q_out[root_q_start + 4] = q[1]
    joint_q_out[root_q_start + 5] = q[2]
    joint_q_out[root_q_start + 6] = q[3]


@wp.kernel
def update_joint_velocity_from_positions_kernel(
    joint_q_prev: wp.array[wp.float32],
    joint_q_next: wp.array[wp.float32],
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    inv_dt: float,
    joint_qd: wp.array[wp.float32],
):
    joint_idx = wp.tid()
    q_start = joint_q_start[joint_idx]
    qd_start = joint_qd_start[joint_idx]
    q_end = joint_q_start[joint_idx + 1]
    qd_end = joint_qd_start[joint_idx + 1]
    jtype = joint_type[joint_idx]

    if jtype == newton.JointType.FREE or jtype == newton.JointType.DISTANCE:
        joint_qd[qd_start + 0] = (joint_q_next[q_start + 0] - joint_q_prev[q_start + 0]) * inv_dt
        joint_qd[qd_start + 1] = (joint_q_next[q_start + 1] - joint_q_prev[q_start + 1]) * inv_dt
        joint_qd[qd_start + 2] = (joint_q_next[q_start + 2] - joint_q_prev[q_start + 2]) * inv_dt

        q_prev = wp.quat(
            joint_q_prev[q_start + 3],
            joint_q_prev[q_start + 4],
            joint_q_prev[q_start + 5],
            joint_q_prev[q_start + 6],
        )
        q_next = wp.quat(
            joint_q_next[q_start + 3],
            joint_q_next[q_start + 4],
            joint_q_next[q_start + 5],
            joint_q_next[q_start + 6],
        )
        q_delta = wp.normalize(q_next * wp.quat_inverse(q_prev))
        axis, angle = wp.quat_to_axis_angle(q_delta)
        joint_qd[qd_start + 3] = axis[0] * angle * inv_dt
        joint_qd[qd_start + 4] = axis[1] * angle * inv_dt
        joint_qd[qd_start + 5] = axis[2] * angle * inv_dt
    elif jtype == newton.JointType.BALL:
        q_prev = wp.quat(
            joint_q_prev[q_start + 0],
            joint_q_prev[q_start + 1],
            joint_q_prev[q_start + 2],
            joint_q_prev[q_start + 3],
        )
        q_next = wp.quat(
            joint_q_next[q_start + 0],
            joint_q_next[q_start + 1],
            joint_q_next[q_start + 2],
            joint_q_next[q_start + 3],
        )
        q_delta = wp.normalize(q_next * wp.quat_inverse(q_prev))
        axis, angle = wp.quat_to_axis_angle(q_delta)
        joint_qd[qd_start + 0] = axis[0] * angle * inv_dt
        joint_qd[qd_start + 1] = axis[1] * angle * inv_dt
        joint_qd[qd_start + 2] = axis[2] * angle * inv_dt
    else:
        for i in range(qd_end - qd_start):
            if q_start + i < q_end:
                joint_qd[qd_start + i] = (joint_q_next[q_start + i] - joint_q_prev[q_start + i]) * inv_dt


TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.02)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

TABLECLOTH_START_Y = -0.58
CLOTH_DIM_X = 18
CLOTH_DIM_Y = 18
CLOTH_CELL_X = 0.020
CLOTH_CELL_Y = 0.020
CLOTH_HALF_X = 0.5 * CLOTH_DIM_X * CLOTH_CELL_X
CLOTH_HALF_Y = 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y
CLOTH_CENTER = wp.vec3(float(TABLE_POS[0]) - 0.12, TABLECLOTH_START_Y, TABLE_TOP_Z + 0.018)
CLOTH_POS = wp.vec3(
    float(CLOTH_CENTER[0]) - 0.5 * CLOTH_DIM_X * CLOTH_CELL_X,
    float(CLOTH_CENTER[1]) - 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y,
    float(CLOTH_CENTER[2]),
)
CLOTH_COLOR = (0.78, 0.12, 0.10)
CLOTH_COLLISION_RADIUS = 0.010
SOFT_CONTACT_MARGIN = 0.020
SELF_CONTACT_RADIUS = 0.010
SELF_CONTACT_MARGIN = 0.012
SOLVER_ITERATIONS = 24
ROBOT_CONTACT_KE = 5.0e4
ROBOT_CONTACT_KD = 1.0e-4
ROBOT_CONTACT_MU = 1.5
HAND_CONTACT_KE = 3.0e5
HAND_CONTACT_KD = 1.0e-4
HAND_CONTACT_MU = 2.2
HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")
TABLE_CONTACT_MU_NORMAL = 1.2
TABLE_CONTACT_MU_LOW = 0.12

RETREAT_DISTANCE = 0.48
RETREAT_TIME = 1.4
TURN_OUT_DEGREES = 90.0
TURN_OUT_TIME = 1.4
CENTER_SHIFT_Y = -TABLECLOTH_START_Y
CENTER_MOVE_TIME = 2.2
TURN_IN_DEGREES = -90.0
TURN_IN_TIME = 1.4
APPROACH_DISTANCE = 0.40
PRE_LAYDOWN_TIME = 1.2
LAYDOWN_TIME = 1.8
APPROACH_TIME = PRE_LAYDOWN_TIME + LAYDOWN_TIME
RELEASE_TIME = 0.8
POST_RELEASE_LIFT_TIME = 1.0
POST_RELEASE_HOLD_TIME = 0.8
FINAL_HOLD_TIME = 0.8
HOME_HOLD_TIME = 0.5
APPROACH_TIME_HANDS = 1.4
GRASP_TIME = 1.4
CLOSE_TIME = 1.8
LIFT_TIME = 2.0
HANG_SETTLE_TIME = 0.6
ROOT_MOTION_START_TIME = HOME_HOLD_TIME + APPROACH_TIME_HANDS + GRASP_TIME + CLOSE_TIME + LIFT_TIME + HANG_SETTLE_TIME
ROOT_MOTION_DURATION = RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME + APPROACH_TIME
ROOT_PRE_APPROACH_TIME = RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME
SCRIPT_DURATION = ROOT_MOTION_START_TIME + ROOT_MOTION_DURATION + RELEASE_TIME + POST_RELEASE_HOLD_TIME + POST_RELEASE_LIFT_TIME
DEFAULT_NUM_FRAMES = int(math.ceil((SCRIPT_DURATION + FINAL_HOLD_TIME) * 60.0))

CAMERA_POS = wp.vec3(1.10, -2.35, 1.70)
CAMERA_PITCH = -16.0
CAMERA_YAW = 105.0

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

RIGHT_TCP_ROT = wp.quat(0.0245, 0.6878, 0.7139, -0.1294)
RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(float(CLOTH_CENTER[0]) + CLOTH_HALF_X - 0.06, TABLECLOTH_START_Y - CLOTH_HALF_Y, 1.30),
    RIGHT_TCP_ROT,
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(float(CLOTH_CENTER[0]) + CLOTH_HALF_X - 0.012, TABLECLOTH_START_Y - CLOTH_HALF_Y + 0.012, TABLE_TOP_Z + 0.055),
    RIGHT_TCP_ROT,
)
GRASP_HEIGHT_OFFSET = 0.015
LIFT_HEIGHT = 0.24
LAYDOWN_HEIGHT = 0.015
LAYDOWN_FORWARD = 0.18



class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 12
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0

        self.particle_radius = CLOTH_COLLISION_RADIUS
        self.soft_contact_margin = SOFT_CONTACT_MARGIN
        self.particle_self_contact_radius = SELF_CONTACT_RADIUS
        self.particle_self_contact_margin = SELF_CONTACT_MARGIN
        self.self_contact_bvh_rebuild_interval_frames = 30

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        urdf_path = Path("E:/csy_work/CG/assets/DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, TABLECLOTH_START_Y, 0.0), wp.quat_identity()),
            floating=True,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)
        self.table_shape_start = builder.shape_count
        self._add_table_scene(builder)
        self.table_shape_end = builder.shape_count
        self.cloth_start = builder.particle_count
        self._add_cloth(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._configure_particle_contacts()
        self.root_joint_index = self._root_free_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint_index])
        self.root_qd_start = int(self.model.joint_qd_start.numpy()[self.root_joint_index])
        root_q0_np = self.model.joint_q.numpy()[self.root_q_start : self.root_q_start + 7].copy()
        self.root_q0_np = root_q0_np
        self.root_q0 = wp.array(root_q0_np, dtype=wp.float32, device=self.model.device)
        self.root_motion_start_time = ROOT_MOTION_START_TIME

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET

        self.left_home_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_home_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.right_approach_tf = RIGHT_APPROACH_TF
        self.left_approach_tf = self._mirror_about_tablecloth_center_y(RIGHT_APPROACH_TF)
        self.right_grasp_tf = self._offset_transform(RIGHT_GRASP_TF, wp.vec3(0.0, 0.0, GRASP_HEIGHT_OFFSET))
        self.left_grasp_tf = self._mirror_about_tablecloth_center_y(self.right_grasp_tf)
        self.left_lift_tf = self._offset_transform(self.left_grasp_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        self.right_lift_tf = self._offset_transform(self.right_grasp_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        self.left_pre_laydown_tf = self._offset_transform(
            self.left_lift_tf,
            wp.vec3(0.7 * LAYDOWN_FORWARD, 0.0, 0.0),
        )
        self.right_pre_laydown_tf = self._offset_transform(
            self.right_lift_tf,
            wp.vec3(0.7 * LAYDOWN_FORWARD, 0.0, 0.0),
        )
        self.left_laydown_tf = self._offset_transform(
            self.left_lift_tf,
            wp.vec3(LAYDOWN_FORWARD, 0.0, -LIFT_HEIGHT + LAYDOWN_HEIGHT),
        )
        self.right_laydown_tf = self._offset_transform(
            self.right_lift_tf,
            wp.vec3(LAYDOWN_FORWARD, 0.0, -LIFT_HEIGHT + LAYDOWN_HEIGHT),
        )
        self.left_post_release_tf = self._offset_transform(
            self.left_laydown_tf,
            wp.vec3(0.0, 0.0, 0.16),
        )
        self.right_post_release_tf = self._offset_transform(
            self.right_laydown_tf,
            wp.vec3(0.0, 0.0, 0.16),
        )
        self.left_tf = self.left_home_tf
        self.right_tf = self.right_home_tf

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.hand_q_indices, self.hand_open, self.hand_grasp = self._build_hand_targets()
        self.setup_ik()
        self.motion_segments = self._build_motion_segments()

        self.model.soft_contact_ke = 1.0e6
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 2.0
        self._configure_robot_contacts()

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=self.soft_contact_margin,
            iterations=SOLVER_ITERATIONS,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.03,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=96,
            particle_edge_contact_buffer_size=128,
            particle_collision_detection_interval=-1,
        )

        self.initial_cloth_height = float(np.max(self.state_0.particle_q.numpy()[:, 2]))
        self.max_cloth_height = self.initial_cloth_height

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self._report_pose(force=True)

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        for joint_idx, joint_type in enumerate(builder.joint_type):
            dof_begin = builder.joint_qd_start[joint_idx]
            dof_end = (
                builder.joint_qd_start[joint_idx + 1]
                if joint_idx + 1 < len(builder.joint_qd_start)
                else builder.joint_dof_count
            )
            q_begin = builder.joint_q_start[joint_idx]
            q_end = (
                builder.joint_q_start[joint_idx + 1]
                if joint_idx + 1 < len(builder.joint_q_start)
                else builder.joint_coord_count
            )
            is_free_root = joint_type == newton.JointType.FREE and builder.joint_parent[joint_idx] == -1

            for local_dof, dof_idx in enumerate(range(dof_begin, dof_end)):
                q_idx = q_begin + local_dof
                if q_idx < q_end:
                    builder.joint_target_pos[dof_idx] = builder.joint_q[q_idx]
                builder.joint_target_ke[dof_idx] = 0.0 if is_free_root else 650.0
                builder.joint_target_kd[dof_idx] = 0.0 if is_free_root else 65.0
                builder.joint_effort_limit[dof_idx] = 0.0 if is_free_root else 180.0
                builder.joint_armature[dof_idx] = 0.0 if is_free_root else 0.02

        for joint_name in (*self.LEFT_HAND_JOINTS, *self.RIGHT_HAND_JOINTS):
            joint_idx = self._builder_joint_index(builder, joint_name)
            dof_idx = builder.joint_qd_start[joint_idx]
            builder.joint_target_ke[dof_idx] = 950.0
            builder.joint_target_kd[dof_idx] = 75.0
            builder.joint_effort_limit[dof_idx] = 45.0
            builder.joint_armature[dof_idx] = 0.005

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
        )
        builder.add_ground_plane()

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=CLOTH_POS,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_X,
            cell_y=CLOTH_CELL_Y,
            mass=0.002,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-5,
            edge_ke=1.0,
            edge_kd=0.05,
            particle_radius=self.particle_radius,
            label="bimanual_grasp_cloth",
        )

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _configure_robot_contacts(self) -> None:
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_body = self.model.shape_body.numpy()

        for shape_idx in range(self.robot_shape_end):
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue
            link_name = self.model.body_label[body_idx].lower()
            shape_ke[shape_idx] = ROBOT_CONTACT_KE
            shape_kd[shape_idx] = ROBOT_CONTACT_KD
            shape_mu[shape_idx] = ROBOT_CONTACT_MU
            if any(keyword in link_name for keyword in HAND_CONTACT_KEYWORDS):
                shape_ke[shape_idx] = HAND_CONTACT_KE
                shape_kd[shape_idx] = HAND_CONTACT_KD
                shape_mu[shape_idx] = HAND_CONTACT_MU

        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

    def _set_table_contact_mu(self, table_mu: float) -> None:
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[self.table_shape_start : self.table_shape_end] = table_mu
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
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

    def _build_motion_segments(self) -> tuple[tuple[float, wp.transform, wp.transform, wp.transform, wp.transform, float, float], ...]:
        return (
            (
                HOME_HOLD_TIME,
                self.left_home_tf,
                self.left_home_tf,
                self.right_home_tf,
                self.right_home_tf,
                0.0,
                0.0,
            ),
            (
                APPROACH_TIME_HANDS,
                self.left_home_tf,
                self.left_approach_tf,
                self.right_home_tf,
                self.right_approach_tf,
                0.0,
                0.0,
            ),
            (
                GRASP_TIME,
                self.left_approach_tf,
                self.left_grasp_tf,
                self.right_approach_tf,
                self.right_grasp_tf,
                0.0,
                0.0,
            ),
            (
                CLOSE_TIME,
                self.left_grasp_tf,
                self.left_grasp_tf,
                self.right_grasp_tf,
                self.right_grasp_tf,
                0.0,
                1.00,
            ),
            (
                LIFT_TIME,
                self.left_grasp_tf,
                self.left_lift_tf,
                self.right_grasp_tf,
                self.right_lift_tf,
                0.99,
                1.0,
            ),
            (
                HANG_SETTLE_TIME,
                self.left_lift_tf,
                self.left_lift_tf,
                self.right_lift_tf,
                self.right_lift_tf,
                1.0,
                1.0,
            ),
            (
                ROOT_PRE_APPROACH_TIME,
                self.left_lift_tf,
                self.left_lift_tf,
                self.right_lift_tf,
                self.right_lift_tf,
                1.0,
                1.0,
            ),
            (
                PRE_LAYDOWN_TIME,
                self.left_lift_tf,
                self.left_pre_laydown_tf,
                self.right_lift_tf,
                self.right_pre_laydown_tf,
                1.0,
                1.0,
            ),
            (
                LAYDOWN_TIME,
                self.left_pre_laydown_tf,
                self.left_laydown_tf,
                self.right_pre_laydown_tf,
                self.right_laydown_tf,
                1.0,
                1.0,
            ),
            (
                RELEASE_TIME,
                self.left_laydown_tf,
                self.left_laydown_tf,
                self.right_laydown_tf,
                self.right_laydown_tf,
                1.0,
                0.0,
            ),
            (
                POST_RELEASE_HOLD_TIME,
                self.left_laydown_tf,
                self.left_laydown_tf,
                self.right_laydown_tf,
                self.right_laydown_tf,
                0.0,
                0.0,
            ),
            (
                POST_RELEASE_LIFT_TIME,
                self.left_laydown_tf,
                self.left_post_release_tf,
                self.right_laydown_tf,
                self.right_post_release_tf,
                0.0,
                0.0,
            ),
        )

    def _build_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        q_indices = []
        open_values = []
        grasp_values = []
        targets = {
            "HAND_THUMB2": 0.84,
            "HAND_THUMB1": 0.46,
            "HAND_INDEX": 0.70,
            "INDEX_PIP": 0.90,
        }

        for side in ("LEFT", "RIGHT"):
            for suffix, target in targets.items():
                joint_idx = self._joint_index(f"{side}_{suffix}")
                q_idx = int(q_start[joint_idx])
                q_indices.append(q_idx)
                open_values.append(float(q_home[q_idx]))
                grasp_values.append(target)

        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(grasp_values, dtype=wp.float32, device=self.model.device),
        )

    def _grasp_alpha(self) -> float:
        close_start_time = HOME_HOLD_TIME + APPROACH_TIME_HANDS + GRASP_TIME
        t = self.sim_time + self.frame_dt - close_start_time
        if t <= 0.0:
            return 0.0
        if t < CLOSE_TIME:
            u = self._smoothstep(t / CLOSE_TIME)
            return float(u)
        return 1.0

    def _smoothstep(self, u: float) -> float:
        x = float(np.clip(u, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

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
            q_begin = int(q_start[joint_idx])
            q_end = int(q_start[joint_idx + 1])
            dof_begin = int(qd_start[joint_idx])
            dof_end = int(qd_start[joint_idx + 1])
            for q_idx, dof_idx in zip(range(q_begin, q_end), range(dof_begin, dof_end), strict=False):
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
            for q_idx in range(int(q_start[joint_idx]), int(q_start[joint_idx + 1])):
                locked_q_indices.append(q_idx)
                locked_q_values.append(float(q_home[q_idx]))

        return (
            wp.array(locked_q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(locked_q_values, dtype=wp.float32, device=self.model.device),
        )

    def _base_motion_time(self, query_time: float) -> float:
        return max(float(query_time) - self.root_motion_start_time, 0.0)

    def _motion_phase(self) -> str:
        base_time = self.sim_time - self.root_motion_start_time
        if base_time < 0.0:
            return "grasp_lift_hang"
        if base_time < RETREAT_TIME:
            return "retreat"
        if base_time < RETREAT_TIME + TURN_OUT_TIME:
            return "turn_out"
        if base_time < RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME:
            return "move_to_center"
        if base_time < ROOT_MOTION_DURATION:
            return "turn_to_table"
        if base_time < ROOT_MOTION_DURATION + PRE_LAYDOWN_TIME:
            return "pre_laydown"
        if base_time < ROOT_MOTION_DURATION + PRE_LAYDOWN_TIME + LAYDOWN_TIME:
            return "lay_down"
        if base_time < ROOT_MOTION_DURATION + PRE_LAYDOWN_TIME + LAYDOWN_TIME + RELEASE_TIME:
            return "release_hold"
        if base_time < ROOT_MOTION_DURATION + PRE_LAYDOWN_TIME + LAYDOWN_TIME + RELEASE_TIME + POST_RELEASE_HOLD_TIME:
            return "post_release_hold"
        return "post_release_lift"
    def _refresh_self_contact_bvh(self) -> None:
        if self.frame_index > 0 and self.frame_index % self.self_contact_bvh_rebuild_interval_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

    def _sample_script(self, query_time: float) -> tuple[wp.transform, wp.transform, float]:
        remaining = query_time
        for duration, left_start, left_end, right_start, right_end, grasp_start, grasp_end in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                left_tf = self._interpolate_transform(left_start, left_end, alpha)
                right_tf = self._interpolate_transform(right_start, right_end, alpha)
                grasp_alpha = grasp_start * (1.0 - alpha) + grasp_end * alpha
                return left_tf, right_tf, grasp_alpha
            remaining -= duration

        _, _, left_end, _, right_end, _, grasp_end = self.motion_segments[-1]
        return left_end, right_end, grasp_end

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        self.left_tf, self.right_tf, grasp_alpha = self._sample_script(self.sim_time + self.frame_dt)

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
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.hand_q_indices.shape[0],
            inputs=[self.hand_q_indices, self.hand_open, self.hand_grasp, grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        phase = self._motion_phase()
        if phase in ("pre_laydown", "lay_down"):
            self._set_table_contact_mu(TABLE_CONTACT_MU_LOW)
        elif phase in ("release_hold", "post_release_hold", "post_release_lift"):
            self._set_table_contact_mu(TABLE_CONTACT_MU_NORMAL)
        else:
            self._set_table_contact_mu(TABLE_CONTACT_MU_NORMAL)

        wp.launch(
            set_free_root_motion_kernel,
            dim=1,
            inputs=[
                self.root_q_start,
                self.root_q0,
                self._base_motion_time(self.sim_time + self.frame_dt),
                RETREAT_DISTANCE,
                RETREAT_TIME,
                math.radians(TURN_OUT_DEGREES),
                TURN_OUT_TIME,
                CENTER_SHIFT_Y,
                CENTER_MOVE_TIME,
                math.radians(TURN_IN_DEGREES),
                TURN_IN_TIME,
                APPROACH_DISTANCE,
                APPROACH_TIME,
            ],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self._refresh_self_contact_bvh()
        self._prepare_frame_targets()

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_1.joint_q],
                device=self.model.device,
            )
            wp.launch(
                set_free_root_motion_kernel,
                dim=1,
                inputs=[
                    self.root_q_start,
                    self.root_q0,
                    self._base_motion_time(self.sim_time + (substep + 1) * self.sim_dt),
                    RETREAT_DISTANCE,
                    RETREAT_TIME,
                    math.radians(TURN_OUT_DEGREES),
                    TURN_OUT_TIME,
                    CENTER_SHIFT_Y,
                    CENTER_MOVE_TIME,
                    math.radians(TURN_IN_DEGREES),
                    TURN_IN_TIME,
                    APPROACH_DISTANCE,
                    APPROACH_TIME,
                ],
                outputs=[self.state_1.joint_q],
                device=self.model.device,
            )
            wp.launch(
                update_joint_velocity_from_positions_kernel,
                dim=len(self.model.joint_label),
                inputs=[
                    self.state_0.joint_q,
                    self.state_1.joint_q,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.sim_dt,
                ],
                outputs=[self.state_1.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._track_cloth_height()

    def step(self) -> None:
        self.simulate()
        self._report_pose()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bimanual_grasp_cloth",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=CLOTH_COLOR,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        self._track_cloth_height()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Cloth particle positions are not finite")

    def _track_cloth_height(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        self.max_cloth_height = max(self.max_cloth_height, float(np.max(particle_q[:, 2])))

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _root_motion_transform(self, query_time: float) -> wp.transform:
        base_time = self._base_motion_time(query_time)

        def smooth(u: float) -> float:
            x = float(np.clip(u, 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        t0 = RETREAT_TIME
        t1 = t0 + TURN_OUT_TIME
        t2 = t1 + CENTER_MOVE_TIME
        t3 = t2 + TURN_IN_TIME
        retreat_u = smooth(base_time / max(RETREAT_TIME, 1.0e-6))
        turn_out_u = smooth((base_time - t0) / max(TURN_OUT_TIME, 1.0e-6))
        center_u = smooth((base_time - t1) / max(CENTER_MOVE_TIME, 1.0e-6))
        turn_in_u = smooth((base_time - t2) / max(TURN_IN_TIME, 1.0e-6))
        approach_u = smooth((base_time - t3) / max(APPROACH_TIME, 1.0e-6))

        yaw = math.radians(TURN_OUT_DEGREES) * turn_out_u + math.radians(TURN_IN_DEGREES) * turn_in_u
        q_yaw = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        q0 = wp.quat(
            float(self.root_q0_np[3]),
            float(self.root_q0_np[4]),
            float(self.root_q0_np[5]),
            float(self.root_q0_np[6]),
        )
        q = q_yaw * q0
        p = wp.vec3(
            float(self.root_q0_np[0]) - RETREAT_DISTANCE * retreat_u + APPROACH_DISTANCE * approach_u,
            float(self.root_q0_np[1]) + CENTER_SHIFT_Y * center_u,
            float(self.root_q0_np[2]),
        )
        return wp.transform(p, q)

    def _target_to_current_root_world(self, target_tf: wp.transform) -> wp.transform:
        root_initial_pos = wp.vec3(float(self.root_q0_np[0]), float(self.root_q0_np[1]), float(self.root_q0_np[2]))
        root_initial_rot = wp.quat(
            float(self.root_q0_np[3]),
            float(self.root_q0_np[4]),
            float(self.root_q0_np[5]),
            float(self.root_q0_np[6]),
        )
        root_current = self._root_motion_transform(self.sim_time)
        root_current_pos = wp.transform_get_translation(root_current)
        root_current_rot = wp.transform_get_rotation(root_current)
        target_pos = wp.transform_get_translation(target_tf)
        target_rot = wp.transform_get_rotation(target_tf)

        local_pos = wp.quat_rotate_inv(root_initial_rot, target_pos - root_initial_pos)
        local_rot = wp.quat_inverse(root_initial_rot) * target_rot
        world_pos = root_current_pos + wp.quat_rotate(root_current_rot, local_pos)
        world_rot = root_current_rot * local_rot
        return wp.transform(world_pos, world_rot)

    def _mirror_y_transform(self, tf: wp.transform) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        quat = wp.transform_get_rotation(tf)
        return wp.transform(
            wp.vec3(float(pos[0]), -float(pos[1]), float(pos[2])),
            wp.quat(-float(quat[0]), float(quat[1]), -float(quat[2]), float(quat[3])),
        )

    def _mirror_about_tablecloth_center_y(self, tf: wp.transform) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        quat = wp.transform_get_rotation(tf)
        mirrored_y = 2.0 * TABLECLOTH_START_Y - float(pos[1])
        return wp.transform(
            wp.vec3(float(pos[0]), mirrored_y, float(pos[2])),
            wp.quat(-float(quat[0]), float(quat[1]), -float(quat[2]), float(quat[3])),
        )

    def _offset_transform(self, tf: wp.transform, offset: wp.vec3) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        return wp.transform(
            wp.vec3(float(pos[0]) + float(offset[0]), float(pos[1]) + float(offset[1]), float(pos[2]) + float(offset[2])),
            wp.transform_get_rotation(tf),
        )

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

    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return

        left_actual_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        left_target_tf = self._target_to_current_root_world(self.left_tf)
        right_target_tf = self._target_to_current_root_world(self.right_tf)

        left_target_pos = self._vec3_to_np(wp.transform_get_translation(left_target_tf))
        left_actual_pos = self._vec3_to_np(wp.transform_get_translation(left_actual_tf))
        right_target_pos = self._vec3_to_np(wp.transform_get_translation(right_target_tf))
        right_actual_pos = self._vec3_to_np(wp.transform_get_translation(right_actual_tf))

        left_target_rot = self._quat_to_np(wp.transform_get_rotation(left_target_tf))
        left_actual_rot = self._quat_to_np(wp.transform_get_rotation(left_actual_tf))
        right_target_rot = self._quat_to_np(wp.transform_get_rotation(right_target_tf))
        right_actual_rot = self._quat_to_np(wp.transform_get_rotation(right_actual_tf))

        left_pos_err = float(np.linalg.norm(left_target_pos - left_actual_pos))
        right_pos_err = float(np.linalg.norm(right_target_pos - right_actual_pos))
        left_rot_err = self._quat_angle_error_deg(left_target_rot, left_actual_rot)
        right_rot_err = self._quat_angle_error_deg(right_target_rot, right_actual_rot)

        print(
            f"[{self.sim_time:7.3f}s] "
            f"L target={self._format_pose(left_target_pos, left_target_rot)} "
            f"actual={self._format_pose(left_actual_pos, left_actual_rot)} "
            f"pos_err={left_pos_err:.5f} m rot_err={left_rot_err:.3f} deg | "
            f"R target={self._format_pose(right_target_pos, right_target_rot)} "
            f"actual={self._format_pose(right_actual_pos, right_actual_rot)} "
            f"pos_err={right_pos_err:.5f} m rot_err={right_rot_err:.3f} deg"
        )
        self.last_print_time = self.sim_time

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _builder_joint_index(self, builder: newton.ModelBuilder, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(suffix))

    def _joint_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))

    def _root_free_joint_index(self) -> int:
        joint_type = self.model.joint_type.numpy()
        joint_parent = self.model.joint_parent.numpy()
        for i, (jtype, parent) in enumerate(zip(joint_type, joint_parent, strict=False)):
            if int(parent) == -1 and int(jtype) == int(newton.JointType.FREE):
                return i
        raise ValueError('Expected the Dexforce W1 root to be imported as a FREE joint')

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)}

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _quat_angle_error_deg(self, quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        dot = abs(float(np.dot(quat_a, quat_b)))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    def _format_xyz(self, xyz: np.ndarray) -> str:
        return f"[{xyz[0]: .4f}, {xyz[1]: .4f}, {xyz[2]: .4f}]"

    def _format_xyzw(self, xyzw: np.ndarray) -> str:
        return f"[{xyzw[0]: .4f}, {xyzw[1]: .4f}, {xyzw[2]: .4f}, {xyzw[3]: .4f}]"

    def _format_pose(self, xyz: np.ndarray, xyzw: np.ndarray) -> str:
        return f"pos={self._format_xyz(xyz)} quat_xyzw={self._format_xyzw(xyzw)}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument(
            "--print-interval",
            type=float,
            default=3.0,
            help="Seconds between TCP position reports. Use 0.0 to print every frame.",
        )
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported URDF self-collisions while building the Dexforce model.",
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


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
