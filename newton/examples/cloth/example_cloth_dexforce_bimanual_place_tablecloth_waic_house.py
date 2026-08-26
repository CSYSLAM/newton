# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 bimanual tablecloth placement in the WAIC house.

The robot's fixed arm and free-base motion are solved once in an *IK-only*
model, then replayed as cached targets in the main MJVBD model. The optional
WAIC house USD is rendering-only; the aligned table box is the sole tablecloth
particle-contact collider.

Run, from the repository root::

    uv run --extra examples -m newton.examples cloth_dexforce_bimanual_place_tablecloth_waic_house
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBD

# The captured trajectory was authored at this table, then rigidly transformed
# into the aligned WAIC dining-table coordinate system below.
OLD_ROBOT_BASE_POS = wp.vec3(0.0, -0.58, 0.0)
OLD_TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.02)
TABLE_COLOR = (0.35, 0.42, 0.48)
TABLE_COLLIDER_Z_OFFSET = 0.018
WAIC_ROBOT_BASE_POS = wp.vec3(3.36697769, 1.27712452, -0.00377202)
WAIC_ROBOT_BASE_QUAT = wp.quat_identity()
CAMERA_POS = wp.vec3(6.13405228, -0.21738398, 1.36522913)
CAMERA_PITCH, CAMERA_YAW = -11.2, 142.4
DEFAULT_HOUSE_USD = (
    "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/"
    "House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
)

TABLE_TOP_Z = float(OLD_TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
CLOTH_DIM_X = 18
CLOTH_DIM_Y = 18
CLOTH_CELL_X = 0.020
CLOTH_CELL_Y = 0.020
CLOTH_CENTER = wp.vec3(float(OLD_TABLE_POS[0]) - 0.12, float(OLD_ROBOT_BASE_POS[1]), TABLE_TOP_Z + 0.018)
CLOTH_POS = wp.vec3(
    float(CLOTH_CENTER[0]) - 0.5 * CLOTH_DIM_X * CLOTH_CELL_X,
    float(CLOTH_CENTER[1]) - 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y,
    float(CLOTH_CENTER[2]),
)
CLOTH_COLOR = (0.78, 0.12, 0.10)
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

RADIUS = 0.010
SOFT_MARGIN = 0.020
SELF_RADIUS = 0.010
SELF_MARGIN = 0.012
VBD_ITERATIONS = 24
IK_ITERATIONS = 24
LEGACY_SOFT_CONTACT_KE = 1.0e6
LEGACY_SOFT_CONTACT_KD = 1.0e-6
ROBOT_CONTACT_KE = 5.0e4
ROBOT_CONTACT_KD = 1.0e-4
ROBOT_CONTACT_MU = 1.5
HAND_CONTACT_KE = 3.0e5
HAND_CONTACT_KD = 1.0e-4
HAND_CONTACT_MU = 2.2
TABLE_CONTACT_MU = 1.2
TABLE_CONTACT_MU_LOW = 0.12
HAND_RELEASE_CONTACT_MU = 0.08
HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")

RETREAT_DISTANCE = 0.48
RETREAT_TIME = 1.4
TURN_OUT_TIME = 1.4
CENTER_SHIFT_Y = 0.58
CENTER_MOVE_TIME = 2.2
TURN_IN_TIME = 1.4
APPROACH_DISTANCE = 0.40
PRE_LAYDOWN_TIME = 1.2
LAYDOWN_TIME = 1.8
RELEASE_TIME = 0.8
POST_RELEASE_HOLD_TIME = 0.8
POST_RELEASE_LIFT_TIME = 1.0
HOME_HOLD_TIME = 0.5
APPROACH_TIME_HANDS = 1.4
GRASP_TIME = 1.4
CLOSE_TIME = 1.8
LIFT_TIME = 2.0
HANG_SETTLE_TIME = 0.6
ROOT_MOTION_START_TIME = HOME_HOLD_TIME + APPROACH_TIME_HANDS + GRASP_TIME + CLOSE_TIME + LIFT_TIME + HANG_SETTLE_TIME
ROOT_PRE_APPROACH_TIME = RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME
ROOT_MOTION_DURATION = ROOT_PRE_APPROACH_TIME + PRE_LAYDOWN_TIME + LAYDOWN_TIME

# Canonical wrist TCP poses. The whole scripted scene is transformed into the
# WAIC world from ``OLD_ROBOT_BASE_POS``.
P = wp.vec3
Q = wp.quat
RIGHT_TCP_ROT = Q(0.0245, 0.6878, 0.7139, -0.1294)
RIGHT_APPROACH = wp.transform(P(0.60, -0.76, 1.30), RIGHT_TCP_ROT)
RIGHT_GRASP = wp.transform(P(0.648, -0.748, TABLE_TOP_Z + 0.070), RIGHT_TCP_ROT)


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _joint_velocity_from_positions(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
    joint = wp.tid()
    q_begin, q_end = joint_q_start[joint], joint_q_start[joint + 1]
    qd_begin, qd_end = joint_qd_start[joint], joint_qd_start[joint + 1]
    if joint_type[joint] == newton.JointType.FREE:
        out[qd_begin + 0] = (q1[q_begin + 0] - q0[q_begin + 0]) * inv_dt
        out[qd_begin + 1] = (q1[q_begin + 1] - q0[q_begin + 1]) * inv_dt
        out[qd_begin + 2] = (q1[q_begin + 2] - q0[q_begin + 2]) * inv_dt
        q_delta = wp.normalize(
            wp.quat(q1[q_begin + 3], q1[q_begin + 4], q1[q_begin + 5], q1[q_begin + 6])
            * wp.quat_inverse(wp.quat(q0[q_begin + 3], q0[q_begin + 4], q0[q_begin + 5], q0[q_begin + 6]))
        )
        axis, angle = wp.quat_to_axis_angle(q_delta)
        out[qd_begin + 3] = axis[0] * angle * inv_dt
        out[qd_begin + 4] = axis[1] * angle * inv_dt
        out[qd_begin + 5] = axis[2] * angle * inv_dt
    else:
        for i in range(qd_end - qd_begin):
            if q_begin + i < q_end:
                out[qd_begin + i] = (q1[q_begin + i] - q0[q_begin + i]) * inv_dt


@wp.kernel
def _lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _copy_joint_q(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _load_cached_joint_q(
    cached_q: wp.array2d[float], frame_counter: wp.array[wp.int32], max_frame_index: int, dst: wp.array[float]
):
    i = wp.tid()
    dst[i] = cached_q[wp.min(frame_counter[0], max_frame_index), i]


@wp.kernel
def _advance_frame_counter(frame_counter: wp.array[wp.int32], max_frame_index: int):
    if wp.tid() == 0:
        frame_counter[0] = wp.min(frame_counter[0] + 1, max_frame_index)


@wp.kernel
def _set_hand_friction(shape_mu: wp.array[float], hand_shapes: wp.array[int], hand_mu: float):
    i = wp.tid()
    shape_mu[hand_shapes[i]] = hand_mu


@wp.kernel
def _set_one_shape_friction(shape_mu: wp.array[float], shape_index: int, mu: float):
    if wp.tid() == 0:
        shape_mu[shape_index] = mu


class Example:
    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_SUFFIXES = (
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

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd

        self._build_scene()
        self.device = self.model.device
        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.solver = SolverMJVBD(
            self.model,
            rigid_mode="external",
            soft_contact_margin=SOFT_MARGIN,
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": SELF_RADIUS,
                "particle_self_contact_margin": SELF_MARGIN,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
                "particle_vertex_contact_buffer_size": 16,
                "particle_edge_contact_buffer_size": 20,
                "rigid_body_particle_contact_buffer_size": 256,
                "particle_collision_detection_interval": -1,
            },
        )
        self.contacts = getattr(self.solver, "contacts", None)
        if self.contacts is None:
            self.contacts = self.solver.create_contacts()

        self._build_ik()
        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.left_home = self._tcp(self.ik_state, self.ik_left_body)
        self.right_home = self._tcp(self.ik_state, self.ik_right_body)
        self.segments = self._segments()
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._hand_q()
        self.hand_shape_indices = self._hand_shape_indices()
        self._set_materials(2.0, HAND_CONTACT_MU, TABLE_CONTACT_MU)
        self.ik_q_indices, self.main_q_indices = self._joint_coordinate_mapping()
        self._build_joint_target_cache()
        self.graph_frame_index = wp.array([1], dtype=wp.int32, device=self.device)

        self._attach_house_usd()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.graph = None
        self.graphs = {}
        self.capture()

    # --- scene ----------------------------------------------------------
    def _robot_urdf(self) -> Path:
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = Path(__file__).resolve().parents[3] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"
        if path.is_file():
            return path
        raise FileNotFoundError("Dexforce W1 URDF is unavailable; pass --robot-urdf PATH.")

    def _build_scene(self):
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0
        SolverMJVBD.register_custom_attributes(builder)
        self.robot_shape_end = builder.shape_count
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=True,
            enable_self_collisions=bool(self.args.enable_self_collisions),
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._add_table(builder)
        self._add_tablecloth(builder)
        self._configure_flags(builder)
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        if self.model.edge_rest_angle is not None:
            self.model.edge_rest_angle.zero_()
        self._configure_contact_materials()

    def _add_table(self, builder):
        old_table = self._world_vec(OLD_TABLE_POS)
        table_position = wp.vec3(
            float(old_table[0]),
            float(old_table[1]),
            float(old_table[2]) + float(self.args.table_collider_z_offset),
        )
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=5.0e5, kd=1.0e-6, mu=1.2, is_visible=bool(self.args.show_physics_table)
        )
        self.table_shape_index = builder.shape_count
        builder.add_shape_box(
            -1,
            xform=wp.transform(table_position, self.base_rot),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=cfg,
            color=TABLE_COLOR,
            label="waic_physics_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="waic_ground")

    def _add_tablecloth(self, builder):
        builder.add_cloth_grid(
            pos=self._world_vec(CLOTH_POS),
            rot=self.base_rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_X,
            cell_y=CLOTH_CELL_Y,
            mass=0.003,
            tri_ke=7.0e2,
            tri_ka=7.0e2,
            tri_kd=5.0e-5,
            edge_ke=0.35,
            edge_kd=0.12,
            particle_radius=RADIUS,
            label="waic_bimanual_place_tablecloth",
        )

    def _configure_flags(self, builder):
        cp, cs = int(newton.ShapeFlags.COLLIDE_PARTICLES), int(newton.ShapeFlags.COLLIDE_SHAPES)
        # Match the captured fold scene: every robot collider can touch cloth,
        # but the rigid-only collision path remains disabled in external-FK mode.
        for i in range(self.robot_shape_end):
            builder.shape_flags[i] &= ~cs
            builder.shape_flags[i] |= cp
        for i in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[i] |= cp

    # --- independent IK model -----------------------------------------
    def _build_ik(self):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        b.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(OLD_ROBOT_BASE_POS, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = b.finalize(device=self.model.device)
        self.ik_left_body = self._body_index(self.ik_model.body_label, "left_j7")
        self.ik_right_body = self._body_index(self.ik_model.body_label, "right_j7")
        self.ik_state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, self.ik_state)
        self.left_obj = ik.IKObjectivePosition(
            self.ik_left_body,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(self._tcp(self.ik_state, self.ik_left_body))],
                dtype=wp.vec3,
                device=self.device,
            ),
        )
        self.left_rot = ik.IKObjectiveRotation(
            self.ik_left_body,
            wp.quat_identity(),
            wp.array(
                [self._v4(wp.transform_get_rotation(self._tcp(self.ik_state, self.ik_left_body)))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        self.right_obj = ik.IKObjectivePosition(
            self.ik_right_body,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(self._tcp(self.ik_state, self.ik_right_body))],
                dtype=wp.vec3,
                device=self.device,
            ),
        )
        self.right_rot = ik.IKObjectiveRotation(
            self.ik_right_body,
            wp.quat_identity(),
            wp.array(
                [self._v4(wp.transform_get_rotation(self._tcp(self.ik_state, self.ik_right_body)))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        lower, upper = self._joint_limits()
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # --- motion / simulation ------------------------------------------
    def _segments(self):
        left_approach = self._mirror_y(RIGHT_APPROACH)
        right_grasp = self._offset(RIGHT_GRASP, wp.vec3(0.0, 0.0, 0.015))
        left_grasp = self._mirror_y(right_grasp)
        left_lift, right_lift = (
            self._offset(left_grasp, wp.vec3(0.0, 0.0, LIFT_TIME * 0.12)),
            self._offset(right_grasp, wp.vec3(0.0, 0.0, LIFT_TIME * 0.12)),
        )
        left_pre, right_pre = (
            self._offset(left_lift, wp.vec3(0.098, 0.0, 0.0)),
            self._offset(right_lift, wp.vec3(0.098, 0.0, 0.0)),
        )
        left_lay, right_lay = (
            self._offset(left_lift, wp.vec3(0.14, 0.0, -0.225)),
            self._offset(right_lift, wp.vec3(0.14, 0.0, -0.225)),
        )
        left_release, right_release = (
            self._offset(left_lay, wp.vec3(0.0, 0.0, 0.16)),
            self._offset(right_lay, wp.vec3(0.0, 0.0, 0.16)),
        )
        return (
            (HOME_HOLD_TIME, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0),
            (APPROACH_TIME_HANDS, self.left_home, left_approach, self.right_home, RIGHT_APPROACH, 0.0, 0.0),
            (GRASP_TIME, left_approach, left_grasp, RIGHT_APPROACH, right_grasp, 0.0, 0.0),
            (CLOSE_TIME, left_grasp, left_grasp, right_grasp, right_grasp, 0.0, 1.0),
            (LIFT_TIME, left_grasp, left_lift, right_grasp, right_lift, 1.0, 1.0),
            (
                HANG_SETTLE_TIME + RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME,
                left_lift,
                left_lift,
                right_lift,
                right_lift,
                1.0,
                1.0,
            ),
            (PRE_LAYDOWN_TIME, left_lift, left_pre, right_lift, right_pre, 1.0, 1.0),
            (LAYDOWN_TIME, left_pre, left_lay, right_pre, right_lay, 1.0, 1.0),
            (RELEASE_TIME, left_lay, left_lay, right_lay, right_lay, 1.0, 0.0),
            (POST_RELEASE_HOLD_TIME, left_lay, left_lay, right_lay, right_lay, 0.0, 0.0),
            (POST_RELEASE_LIFT_TIME, left_lay, left_release, right_lay, right_release, 0.0, 0.0),
        )

    def _materials_for_script_time(self, script_time, grip):
        root_time = script_time - ROOT_MOTION_START_TIME
        if ROOT_PRE_APPROACH_TIME <= root_time < ROOT_MOTION_DURATION:
            return 2.0, HAND_CONTACT_MU, TABLE_CONTACT_MU_LOW
        if (
            ROOT_MOTION_DURATION
            <= root_time
            < ROOT_MOTION_DURATION + RELEASE_TIME + POST_RELEASE_HOLD_TIME + POST_RELEASE_LIFT_TIME
        ):
            return 2.0, HAND_RELEASE_CONTACT_MU, TABLE_CONTACT_MU
        return 2.0, HAND_CONTACT_MU if grip > 0.15 else ROBOT_CONTACT_MU, TABLE_CONTACT_MU

    def _build_joint_target_cache(self):
        """Solve the fixed scripted motion once, before simulation starts."""
        script_duration = sum(segment[0] for segment in self.segments)
        script_frames = int(np.ceil(script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        self.cached_joint_target_frame_count = max(int(self.args.num_frames), script_frames)

        initial_q = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        cache = np.repeat(initial_q[None, :], self.cached_joint_target_frame_count + 1, axis=0)
        hand_indices = self.hand_indices.numpy()
        hand_open = self.hand_open.numpy()
        hand_grasp = self.hand_grasp.numpy()
        self.cached_materials = [self._materials_for_script_time(0.0, 0.0)]

        for frame_index in range(1, self.cached_joint_target_frame_count + 1):
            script_time = frame_index * self.frame_dt * self.args.trajectory_time_scale
            left, right, grip = self._sample(script_time)
            self.left_obj.set_target_position(0, wp.transform_get_translation(left))
            self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left)))
            self.right_obj.set_target_position(0, wp.transform_get_translation(right))
            self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right)))
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=IK_ITERATIONS)
            wp.launch(
                _lock_q,
                self.lock_indices.shape[0],
                [self.ik_q, self.lock_indices, self.lock_values],
                device=self.device,
            )
            cache[frame_index, self.main_q_indices] = self.ik_q.numpy()[0, self.ik_q_indices]
            cache[frame_index, self.root_q_start : self.root_q_start + 7] = self._root_q(script_time)
            cache[frame_index, hand_indices] = hand_open * (1.0 - grip) + hand_grasp * grip
            self.cached_materials.append(self._materials_for_script_time(script_time, grip))

        self.cached_joint_targets = wp.array(cache, dtype=wp.float32, device=self.device)
        self.cached_materials = tuple(self.cached_materials)
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))

    def _prepare_cached_frame(self):
        """Load one baked target for the uncaptured CPU execution path."""
        cache_index = min(self.frame_index + 1, self.cached_joint_target_frame_count)
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.cached_joint_targets[cache_index])
        self._set_materials(*self.cached_materials[cache_index])

    def _simulate_substeps(self):
        for step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            alpha = (step + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_1.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity_from_positions,
                len(self.model.joint_label),
                [
                    self.state_0.joint_q,
                    self.state_1.joint_q,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.sim_dt,
                    self.state_1.joint_qd,
                ],
                device=self.device,
            )
            newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        self._prepare_cached_frame()
        self._simulate_substeps()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self):
        if not self.use_graph:
            self.simulate()
            return

        cache_index = min(self.frame_index + 1, self.cached_joint_target_frame_count)
        self.graph = self.graphs[self.cached_materials[cache_index]]
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _capture_graph(self, materials):
        """Capture one material variant of a complete scripted display frame."""
        self.model.soft_contact_mu = materials[0]
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)

        with wp.ScopedCapture() as capture:
            wp.launch(
                _copy_joint_q,
                self.model.joint_coord_count,
                [self.state_0.joint_q, self.frame_q_start],
                device=self.device,
            )
            wp.launch(
                _load_cached_joint_q,
                self.model.joint_coord_count,
                [
                    self.cached_joint_targets,
                    self.graph_frame_index,
                    self.cached_joint_target_frame_count,
                    self.frame_q_end,
                ],
                device=self.device,
            )
            wp.launch(
                _set_hand_friction,
                self.hand_shape_indices.shape[0],
                [self.model.shape_material_mu, self.hand_shape_indices, materials[1]],
                device=self.device,
            )
            wp.launch(
                _set_one_shape_friction,
                1,
                [self.model.shape_material_mu, self.table_shape_index, materials[2]],
                device=self.device,
            )
            self._simulate_substeps()
            wp.launch(
                _advance_frame_counter,
                1,
                [self.graph_frame_index, self.cached_joint_target_frame_count],
                device=self.device,
            )

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        return capture.graph

    def capture(self):
        """Capture every fixed-trajectory material variant on CUDA."""
        self.graph = None
        self.graphs = {}
        if not self.use_graph:
            return

        for materials in dict.fromkeys(self.cached_materials):
            self.graphs[materials] = self._capture_graph(materials)
        self.graph_frame_index.fill_(1)
        self.graph = self.graphs[self.cached_materials[1]]

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/waic_bimanual_place_tablecloth",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=CLOTH_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self):
        if not np.all(np.isfinite(self.state_0.particle_q.numpy())):
            raise ValueError("Tablecloth particle positions are not finite")

    # --- helpers --------------------------------------------------------
    def _set_materials(self, cloth_mu, robot_mu, table_mu):
        self.model.soft_contact_mu = cloth_mu
        mu = self.model.shape_material_mu.numpy()
        mu[self.hand_shape_indices.numpy()] = robot_mu
        mu[self.table_shape_index] = table_mu
        self.model.shape_material_mu.assign(mu)

    def _configure_contact_materials(self):
        """Translate the reference VBD contact damping to MJVBD's current units."""
        self.model.soft_contact_ke = LEGACY_SOFT_CONTACT_KE
        # The old fork used ``kd * ke`` as the normal damping coefficient;
        # current MJVBD stores that coefficient directly. Keep the particle
        # side positive, then solve each shape-side value for the same mixed
        # coefficient at contact initialization.
        self.model.soft_contact_kd = LEGACY_SOFT_CONTACT_KD * LEGACY_SOFT_CONTACT_KE
        ke = self.model.shape_material_ke.numpy()
        kd = self.model.shape_material_kd.numpy()
        mu = self.model.shape_material_mu.numpy()
        shape_body = self.model.shape_body.numpy()

        for shape in range(self.model.shape_count):
            shape_ke = float(ke[shape])
            shape_kd = LEGACY_SOFT_CONTACT_KD
            shape_mu = float(mu[shape])
            if shape < self.robot_shape_end:
                body = int(shape_body[shape])
                label = self.model.body_label[body].lower() if body >= 0 else ""
                if any(keyword in label for keyword in HAND_CONTACT_KEYWORDS):
                    shape_ke, shape_kd, shape_mu = HAND_CONTACT_KE, HAND_CONTACT_KD, HAND_CONTACT_MU
                else:
                    shape_ke, shape_kd, shape_mu = ROBOT_CONTACT_KE, ROBOT_CONTACT_KD, ROBOT_CONTACT_MU

            ke[shape] = shape_ke
            mu[shape] = shape_mu
            legacy_mixed_kd = 0.5 * (LEGACY_SOFT_CONTACT_KD + shape_kd)
            legacy_mixed_ke = 0.5 * (LEGACY_SOFT_CONTACT_KE + shape_ke)
            kd[shape] = 2.0 * legacy_mixed_kd * legacy_mixed_ke - self.model.soft_contact_kd

        self.model.shape_material_ke.assign(ke)
        self.model.shape_material_kd.assign(kd)
        self.model.shape_material_mu.assign(mu)

    def _hand_shape_indices(self):
        shape_body = self.model.shape_body.numpy()
        indices = []
        for shape in range(self.robot_shape_end):
            body = int(shape_body[shape])
            if body >= 0 and any(keyword in self.model.body_label[body].lower() for keyword in HAND_CONTACT_KEYWORDS):
                indices.append(shape)
        return wp.array(indices, dtype=wp.int32, device=self.device)

    def _joint_limits(self):
        lo, hi = self.ik_model.joint_limit_lower.numpy().copy(), self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        qs = self.ik_model.joint_q_start.numpy()
        qds = self.ik_model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{n}" for n in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for j, label in enumerate(self.ik_model.joint_label):
            if label not in controlled:
                lo[int(qds[j])] = q[int(qs[j])] - 1.0e-4
                hi[int(qds[j])] = q[int(qs[j])] + 1.0e-4
        return lo, hi

    def _locked_q(self):
        q, qs = self.ik_model.joint_q.numpy(), self.ik_model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{n}" for n in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        ids = [int(qs[j]) for j, label in enumerate(self.ik_model.joint_label) if label not in controlled]
        return wp.array(ids, dtype=wp.int32, device=self.device), wp.array(
            [q[i] for i in ids], dtype=wp.float32, device=self.device
        )

    def _joint_coordinate_mapping(self):
        """Map static-IK joint coordinates onto the floating main model."""
        ik_start = self.ik_model.joint_q_start.numpy()
        main_start = self.model.joint_q_start.numpy()
        ik_labels = {label: index for index, label in enumerate(self.ik_model.joint_label)}
        ik_indices, main_indices = [], []

        for main_joint, label in enumerate(self.model.joint_label):
            ik_joint = ik_labels.get(label)
            if ik_joint is None:
                continue
            ik_range = range(int(ik_start[ik_joint]), int(ik_start[ik_joint + 1]))
            main_range = range(int(main_start[main_joint]), int(main_start[main_joint + 1]))
            if len(ik_range) != len(main_range):
                raise ValueError(f"Joint coordinate mismatch for {label}")
            ik_indices.extend(ik_range)
            main_indices.extend(main_range)

        return np.asarray(ik_indices, dtype=np.int32), np.asarray(main_indices, dtype=np.int32)

    def _hand_q(self):
        q, qs = self.model.joint_q.numpy(), self.model.joint_q_start.numpy()
        ids = []
        open_q = []
        grasp = []
        targets = {"HAND_THUMB2": 0.84, "HAND_THUMB1": 0.46, "HAND_INDEX": 0.70, "INDEX_PIP": 0.90}
        for side in ("LEFT", "RIGHT"):
            for suffix in self.HAND_SUFFIXES:
                j = self._joint_index(f"{side}_{suffix}")
                i = int(qs[j])
                ids.append(i)
                open_q.append(q[i])
                grasp.append(targets.get(suffix, q[i]))
        return (
            wp.array(ids, dtype=wp.int32, device=self.device),
            wp.array(open_q, dtype=wp.float32, device=self.device),
            wp.array(grasp, dtype=wp.float32, device=self.device),
        )

    def _joint_index(self, name):
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels, name):
        return next(i for i, label in enumerate(labels) if label.endswith("/" + name))

    def _tcp(self, state, body):
        tf = wp.transform(*state.body_q.numpy()[body])
        return wp.transform(
            wp.transform_get_translation(tf) + wp.quat_rotate(wp.transform_get_rotation(tf), TCP_OFFSET),
            wp.transform_get_rotation(tf),
        )

    def _sample(self, time):
        t = time
        for d, la, lb, ra, rb, ga, gb in self.segments:
            if t <= d:
                a = float(np.clip(t / d, 0.0, 1.0))
                return self._lerp_tf(la, lb, a), self._lerp_tf(ra, rb, a), ga * (1 - a) + gb * a
            t -= d
        _, _, lb, _, rb, _, g = self.segments[-1]
        return lb, rb, g

    def _root_joint_index(self):
        types = self.model.joint_type.numpy()
        parents = self.model.joint_parent.numpy()
        for index, (joint_type, parent) in enumerate(zip(types, parents, strict=False)):
            if int(joint_type) == int(newton.JointType.FREE) and int(parent) == -1:
                return index
        raise ValueError("Dexforce W1 must be imported with a free root joint")

    def _root_q(self, script_time):
        """Return the WAIC free-root pose for a canonical script time."""
        t = max(float(script_time) - ROOT_MOTION_START_TIME, 0.0)

        def smooth(value):
            return value * value * (3.0 - 2.0 * value)

        retreat = smooth(np.clip(t / RETREAT_TIME, 0.0, 1.0))
        turn_out = smooth(np.clip((t - RETREAT_TIME) / TURN_OUT_TIME, 0.0, 1.0))
        center = smooth(np.clip((t - RETREAT_TIME - TURN_OUT_TIME) / CENTER_MOVE_TIME, 0.0, 1.0))
        turn_in = smooth(np.clip((t - RETREAT_TIME - TURN_OUT_TIME - CENTER_MOVE_TIME) / TURN_IN_TIME, 0.0, 1.0))
        approach = smooth(np.clip((t - ROOT_PRE_APPROACH_TIME) / (PRE_LAYDOWN_TIME + LAYDOWN_TIME), 0.0, 1.0))
        old_position = wp.vec3(
            -RETREAT_DISTANCE * retreat + APPROACH_DISTANCE * approach,
            float(OLD_ROBOT_BASE_POS[1]) + CENTER_SHIFT_Y * center,
            0.0,
        )
        yaw = 0.5 * np.pi * turn_out - 0.5 * np.pi * turn_in
        local_rotation = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        position = self._world_vec(old_position)
        rotation = self._quat_mul(self.base_rot, local_rotation)
        return np.array([*position, *rotation], dtype=np.float32)

    @staticmethod
    def _offset(tf, offset):
        pos = wp.transform_get_translation(tf)
        return wp.transform(pos + offset, wp.transform_get_rotation(tf))

    @staticmethod
    def _mirror_y(tf):
        pos, rot = wp.transform_get_translation(tf), wp.transform_get_rotation(tf)
        return wp.transform(
            wp.vec3(float(pos[0]), 2.0 * float(OLD_ROBOT_BASE_POS[1]) - float(pos[1]), float(pos[2])),
            wp.quat(-float(rot[0]), float(rot[1]), -float(rot[2]), float(rot[3])),
        )

    def _world_vec(self, v):
        # All source poses are expressed relative to the old free-base anchor.
        r = wp.quat_rotate(self.base_rot, v - OLD_ROBOT_BASE_POS)
        return wp.vec3(
            float(r[0]) + float(self.base_pos[0]),
            float(r[1]) + float(self.base_pos[1]),
            float(r[2]) + float(self.base_pos[2]),
        )

    def _world_tf(self, tf):
        return wp.transform(
            self._world_vec(wp.transform_get_translation(tf)),
            self._quat_mul(self.base_rot, wp.transform_get_rotation(tf)),
        )

    @staticmethod
    def _normal_quat(q):
        a = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        a /= max(np.linalg.norm(a), 1.0e-8)
        return wp.quat(*a)

    @staticmethod
    def _quat_mul(a, b):
        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _v4(q):
        return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @staticmethod
    def _lerp_tf(a, b, t):
        pa = np.array(wp.transform_get_translation(a))
        pb = np.array(wp.transform_get_translation(b))
        qa = np.array(wp.transform_get_rotation(a))
        qb = np.array(wp.transform_get_rotation(b))
        if np.dot(qa, qb) < 0:
            qb = -qb
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
        if dot > 0.9995:
            q = qa * (1.0 - t) + qb * t
            q /= np.linalg.norm(q)
        else:
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)
            q = qa * (np.sin((1.0 - t) * theta) / sin_theta) + qb * (np.sin(t * theta) / sin_theta)
        p = pa * (1 - t) + pb * t
        return wp.transform(wp.vec3(*p), wp.quat(*q))

    def _attach_house_usd(self):
        if not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.isfile(self.house_visual_usd):
            print(f"WAIC house USD not found; continuing without it: {self.house_visual_usd}")
            return
        prim = self.viewer.stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1600)
        parser.add_argument(
            "--robot-urdf", default=None, help="Optional Dexforce W1 URDF; defaults to the ignored tablecloth asset."
        )
        parser.add_argument(
            "--house-visual-usd", default=DEFAULT_HOUSE_USD, help="Optional WAIC USD reference; it is visual-only."
        )
        parser.add_argument("--show-physics-table", action="store_true")
        parser.add_argument(
            "--table-collider-z-offset",
            type=float,
            default=TABLE_COLLIDER_Z_OFFSET,
            help="Raise the collision-only WAIC table box [m] without moving the visual table or cloth spawn.",
        )
        parser.add_argument("--enable-self-collisions", action="store_true")
        parser.add_argument(
            "--trajectory-time-scale",
            type=float,
            default=1.0,
            help="Scale the reference 60 Hz motion; 1.0 preserves the original grasp timing.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture fixed-target replay and physics as CUDA graphs.",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(WAIC_ROBOT_BASE_QUAT[3]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
