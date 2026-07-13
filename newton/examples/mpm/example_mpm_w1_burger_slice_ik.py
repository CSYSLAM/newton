# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice IK
#
# Interactive trajectory-tuning variant. The knife stands vertically on the
# table as a separate kinematic body; the W1 (hands-only) robot starts in its
# URDF home pose with all five right-hand fingers already closed (pinching).
# Drag the right-hand TCP gizmo to drive the arm toward the knife; the IK
# solver follows in real time. Use this to place waypoints, then transfer
# them back into the scripted example's motion_segments.
#
# The meat is an MPM particle body; the robot, table, ground and knife share
# one Newton model and are read back by MPM as kinematic colliders.
#
# Command: python -m newton.examples mpm_w1_burger_slice_ik
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverImplicitMPM
from newton.examples.mpm.example_mpm_w1_burger_slice import (
    URDF_PATH,
    TABLE_POS,
    TABLE_HALF_EXTENTS,
    TABLE_TOP_Z,
    TABLE_COLOR,
    MEAT_LO,
    MEAT_HI,
    MEAT_COLOR,
    CUT_COLOR,
    TCP_OFFSET,
    BLADE_HX,
    BLADE_HY,
    BLADE_HZ,
    BLADE_COLOR,
    LEFT_HOLD_TF,
    lock_joint_q_kernel,
    copy_ik_to_joint_q_kernel,
    interpolate_joint_positions_kernel,
    update_joint_velocity_kernel,
    set_indexed_joint_q_kernel,
    compute_deformation_colors,
)


# The knife stands vertically on the table as its own kinematic body
# (body=-1, density 0). The blade is thin in x, long in y, tall in z so it
# stands upright. Placed in front of the robot, beside the meat.
KNIFE_POS = wp.vec3(0.50, -0.30, TABLE_TOP_Z + BLADE_HZ)
KNIFE_COLOR = (0.80, 0.80, 0.85)


class Example:
    """Interactive IK trajectory tuner for the W1 burger slice.

    The knife stands on the table; the robot starts at home with five fingers
    closed. Drag the right TCP gizmo to move the arm.
    """

    LEFT_ARM_JOINTS = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM_JOINTS = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
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

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0
        self.voxel_size = float(args.voxel_size)

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        SolverImplicitMPM.register_custom_attributes(builder)

        builder.add_urdf(
            URDF_PATH,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)
        self._add_table(builder)
        self._add_knife(builder)
        self._emit_meat(builder, args)
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._configure_particle_contacts()

        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_offset = TCP_OFFSET
        self.left_ee_offset = TCP_OFFSET

        # Gizmo target starts at the robot's current home TCP (no scripted motion).
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_tf = LEFT_HOLD_TF
        # Five fingers stay closed for the whole session.
        self.grasp_alpha = 1.0

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.substep_joint_q_prev = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp = self._build_right_hand_targets()
        self.setup_ik()

        self._init_mpm_materials(args)

        mpm_options = SolverImplicitMPM.Config()
        for key in vars(args):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(args, key))
        mpm_options.collider_velocity_mode = "forward"
        self.solver = SolverImplicitMPM(self.model, mpm_options)
        self.solver.setup_collider(
            model=self.model,
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self.particle_colors = wp.empty(self.model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.particle_colors.fill_(MEAT_COLOR)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.55, -1.55, 1.55), -25.0, 35.0)
        self.viewer.show_particles = True
        self.show_deformation = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        print(
            f"[newton] W1 burger slice IK: particles={self.model.particle_count}, "
            f"bodies={self.model.body_count}, shapes={self.model.shape_count}"
        )
        print("[newton] Robot at home, five fingers closed. Drag the right TCP gizmo to move the arm.")

    # ------------------------------------------------------------------ setup
    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 650.0
            builder.joint_target_kd[i] = 65.0
            builder.joint_effort_limit[i] = 180.0
            builder.joint_armature[i] = 0.02

    def _add_table(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2
        table_cfg.density = 0.0
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="table",
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

    def _add_knife(self, builder: newton.ModelBuilder) -> None:
        """Stand the knife vertically on the table as a separate kinematic body."""
        knife_cfg = newton.ModelBuilder.ShapeConfig()
        knife_cfg.ke = 1.0e6
        knife_cfg.kd = 1.0e-6
        knife_cfg.mu = 0.2
        knife_cfg.density = 0.0
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(KNIFE_POS, wp.quat_identity()),
            hx=BLADE_HX,
            hy=BLADE_HY,
            hz=BLADE_HZ,
            cfg=knife_cfg,
            color=KNIFE_COLOR,
            label="knife",
        )

    def _emit_meat(self, builder: newton.ModelBuilder, args) -> None:
        density = float(args.density)
        ppc = int(args.particles_per_cell)
        lo, hi = MEAT_LO, MEAT_HI
        res = np.ceil(ppc * (hi - lo) / self.voxel_size).astype(int)
        cell_size = (hi - lo) / res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * density
        builder.add_particle_grid(
            pos=wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=radius,
            radius_mean=radius,
        )

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        self.model.shape_flags = wp.array(
            flags, dtype=self.model.shape_flags.dtype, device=self.model.device
        )

    def _init_mpm_materials(self, args) -> None:
        m = self.model.mpm
        m.young_modulus.fill_(float(args.young_modulus))
        m.poisson_ratio.fill_(float(args.poisson_ratio))
        m.damping.fill_(float(args.damping))
        m.friction.fill_(float(args.friction))
        m.yield_pressure.fill_(float(args.yield_pressure))
        m.tensile_yield_ratio.fill_(float(args.tensile_yield_ratio))
        m.yield_stress.fill_(float(args.yield_stress))
        m.hardening.fill_(float(args.hardening))
        m.dilatancy.fill_(float(args.dilatancy))
        self.state_0.mpm.particle_Jp.fill_(1.0)

    # ------------------------------------------------------------------ IK
    def setup_ik(self) -> None:
        right_tcp = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        left_tcp = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)

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
                self.right_pos_obj,
                self.right_rot_obj,
                self.left_pos_obj,
                self.left_rot_obj,
                self.joint_limits_obj,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    # ------------------------------------------------------------------ step
    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)

        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))

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
        # Fingers stay closed (grasp_alpha = 1.0) throughout.
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp, self.grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self._prepare_frame_targets()
        for substep in range(self.sim_substeps):
            wp.copy(self.substep_joint_q_prev, self.state_0.joint_q)
            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_0.joint_q],
                device=self.model.device,
            )
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.substep_joint_q_prev, self.state_0.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_0.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
            self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self) -> None:
        self.simulate()
        self._report_pose()

    # ----------------------------------------------------------------- render
    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_left_tcp_target",
                self.left_tf,
                snap_to=self._current_tcp_transform(self.left_ee_index, self.left_ee_offset),
            )
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        if self.show_deformation:
            wp.launch(
                compute_deformation_colors,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.mpm.particle_Jp,
                    self.particle_colors,
                    MEAT_COLOR,
                    CUT_COLOR,
                    12.0,
                ],
                device=self.model.device,
            )
        else:
            self.particle_colors.fill_(MEAT_COLOR)
        self.viewer.log_points(
            "/meat",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_deformation = imgui.checkbox("Show Deformation", self.show_deformation)

    # ------------------------------------------------------------------ tests
    def test_post_step(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("meat particle positions are not finite")

    def test_final(self):
        q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(q)):
            raise ValueError("meat particle positions are not finite")
        if np.min(q[:, 2]) < TABLE_TOP_Z - 0.05:
            raise ValueError(f"meat fell through the table: z_min={np.min(q[:, 2]):.4f}")
        if np.linalg.norm(q.max(axis=0) - q.min(axis=0)) > 5.0:
            raise ValueError("meat exploded")

    # ----------------------------------------------------------- helpers
    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _joint_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))

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

    def _controlled_joint_labels(self) -> set[str]:
        return {
            f"DexforceW1V021/{name}"
            for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS, *self.RIGHT_HAND_JOINTS)
        }

    def _build_right_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        grasp_fraction = {
            "RIGHT_HAND_THUMB2": 0.54,
            "RIGHT_HAND_THUMB1": 0.53,
            "RIGHT_HAND_INDEX": 0.54,
            "RIGHT_INDEX_PIP": 0.43,
            "RIGHT_HAND_MIDDLE": 0.54,
            "RIGHT_MIDDLE_PIP": 0.43,
            "RIGHT_HAND_RING": 0.54,
            "RIGHT_RING_PIP": 0.43,
            "RIGHT_HAND_PINKY": 0.54,
            "RIGHT_PINKY_PIP": 0.43,
        }
        q_indices = []
        open_values = []
        grasp_values = []
        for joint_name in self.RIGHT_HAND_JOINTS:
            joint_idx = self._joint_index(joint_name)
            q_idx = int(q_start[joint_idx])
            upper = float(self.model.joint_limit_upper.numpy()[joint_idx])
            q_indices.append(q_idx)
            open_values.append(float(q_home[q_idx]))
            grasp_values.append(upper * grasp_fraction[joint_name])
        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(grasp_values, dtype=wp.float32, device=self.model.device),
        )

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    def _quat_angle_error_deg(self, quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        dot = abs(float(np.dot(quat_a, quat_b)))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))

    def _format_xyz(self, xyz: np.ndarray) -> str:
        return f"[{xyz[0]: .4f}, {xyz[1]: .4f}, {xyz[2]: .4f}]"

    def _format_xyzw(self, xyzw: np.ndarray) -> str:
        return f"[{xyzw[0]: .4f}, {xyzw[1]: .4f}, {xyzw[2]: .4f}, {xyzw[3]: .4f}]"

    def _format_pose(self, xyz: np.ndarray, xyzw: np.ndarray) -> str:
        return f"pos={self._format_xyz(xyz)} quat_xyzw={self._format_xyzw(xyzw)}"

    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return

        left_actual_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        left_target_pos = self._vec3_to_np(wp.transform_get_translation(self.left_tf))
        left_actual_pos = self._vec3_to_np(wp.transform_get_translation(left_actual_tf))
        right_target_pos = self._vec3_to_np(wp.transform_get_translation(self.right_tf))
        right_actual_pos = self._vec3_to_np(wp.transform_get_translation(right_actual_tf))

        left_target_rot = self._quat_to_np(wp.transform_get_rotation(self.left_tf))
        left_actual_rot = self._quat_to_np(wp.transform_get_rotation(left_actual_tf))
        right_target_rot = self._quat_to_np(wp.transform_get_rotation(self.right_tf))
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
            f"pos_err={right_pos_err:.5f} m rot_err={right_rot_err:.3f} deg | "
            f"grasp={self.grasp_alpha:.2f}"
        )
        self.last_print_time = self.sim_time

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=0)

        parser.add_argument(
            "--print-interval",
            type=float,
            default=3.0,
            help="Seconds between TCP pose reports. Use 0.0 to print every frame.",
        )
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.022)
        parser.add_argument("--particles-per-cell", "-ppc", type=int, default=2)
        parser.add_argument("--density", type=float, default=1000.0)

        parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.45)
        parser.add_argument("--friction", "-mu", type=float, default=0.35)
        parser.add_argument("--damping", type=float, default=0.1)
        parser.add_argument("--yield-pressure", "-yp", type=float, default=1.0e5)
        parser.add_argument("--tensile-yield-ratio", "-tyr", type=float, default=0.3)
        parser.add_argument("--yield-stress", "-ys", type=float, default=0.0)
        parser.add_argument("--hardening", type=float, default=0.0)
        parser.add_argument("--dilatancy", type=float, default=0.0)

        parser.add_argument("--grid-type", "-gt", type=str, default="sparse", choices=["sparse", "fixed", "dense"])
        parser.add_argument("--strain-basis", "-sb", type=str, default="P0")
        parser.add_argument("--max-iterations", "-it", type=int, default=150)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported URDF self-collisions while building the Dexforce model.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
