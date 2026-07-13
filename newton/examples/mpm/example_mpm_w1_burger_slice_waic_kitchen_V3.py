# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice in the WAIC kitchen scene
#
# Reuses the W1 burger slicing setup, but moves the robot to the W1 pose
# currently arranged in the WAIC Blender kitchen and aligns the slicing
# table/meat/knife/TCP-script rig to the Blender worktop height. The selected
# Blender kitchen counter, table and frying pan are loaded from USD as
# visual-only background; the real physics table remains a simple Newton box.
#
# Command: python -m newton.examples mpm_w1_burger_slice_waic_kitchen
#
###########################################################################

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverImplicitMPM


# ---------------------------------------------------------------------------
# Kernels (same pattern as the cloth dexforce IK examples)
# ---------------------------------------------------------------------------
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


@wp.kernel
def set_indexed_joint_q_kernel(
    q_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    grasp_values: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    """Blend hand joints between open and grasp poses by ``alpha``."""
    i = wp.tid()
    joint_q_out[q_indices[i]] = open_values[i] * (1.0 - alpha) + grasp_values[i] * alpha


@wp.kernel
def compute_deformation_colors(
    Jp: wp.array[float],
    colors: wp.array[wp.vec3],
    base_color: wp.vec3,
    cut_color: wp.vec3,
    dev_scale: float,
):
    """Recolor particles by accumulated plastic deformation (|Jp - 1|)."""
    i = wp.tid()
    dev = wp.abs(Jp[i] - 1.0)
    t = wp.clamp(dev * dev_scale, 0.0, 1.0)
    colors[i] = base_color * (1.0 - t) + cut_color * t


@wp.kernel
def set_body_transform_kernel(
    body_q: wp.array[wp.transform],
    body_index: int,
    xform: wp.transform,
):
    """Overwrite a single body's transform (for kinematic bodies)."""
    body_q[body_index] = xform


@wp.kernel
def set_body_velocity_kernel(
    body_qd: wp.array[wp.spatial_vector],
    body_index: int,
    vel: wp.vec3,
):
    """Set a kinematic body's linear velocity (angular velocity left zero)."""
    body_qd[body_index] = wp.spatial_vector(vel, wp.vec3(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
URDF_PATH = Path(__file__).resolve().parents[1] / "cloth" / "DexforceW1V021" / "DexforceW1V021.urdf"

OLD_ROBOT_BASE_POS = wp.vec3(0.0, 0.0, 0.0)
WAIC_ROBOT_BASE_POS = wp.vec3(5.156154155731201, 0.6696404814720154, -0.0037720240652561188)
WAIC_ROBOT_BASE_QUAT = wp.quat_identity()
WAIC_WORKTOP_Z = 0.9000003337860107
WAIC_SELECTED_VISUAL_USD = r"E:\csy_work\CG\assets\WAIC\house_background\waic_kitchen_counter_table_pan.usd"

TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

PAN_CENTER_WORLD = wp.vec3(5.827057838439941, 0.331002801656723, 0.8562719821929932)
PAN_RADIUS = 0.1125
PAN_DISK_HALF_HEIGHT = 0.0125
PAN_HANDLE_LOCAL_POS = wp.vec3(-0.2115001678466797, 0.0, 0.0050)
PAN_HANDLE_LENGTH = 0.215
PAN_HANDLE_RADIUS = 0.012
PAN_HANDLE_COLOR = (0.1, 0.85, 0.95)
PAN_DISK_COLOR = (0.95, 0.35, 0.08)

# Meat block: a slab centered on the table at MEAT_CENTER_X, in front of the
# robot. The size (length x, width y, height z) is configurable via CLI args
# --meat-length / --meat-width / --meat-height. Defaults below are overridden
# in __init__ from args.
MEAT_CENTER_X = 0.63
MEAT_LENGTH = 0.14  # x
MEAT_WIDTH = 0.065  # y
MEAT_HEIGHT = 0.045  # z
MEAT_COLOR = wp.vec3(0.55, 0.27, 0.15)
CUT_COLOR = wp.vec3(0.90, 0.15, 0.10)

# Knife: thin in x (cutting edge), long in y (blade length), tall in z.
# Attached to the right wrist (right_j7) at the TCP offset so the blade tip
# coincides with the IK target.
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
BLADE_HX = 0.003
BLADE_HY = 0.10
BLADE_HZ = 0.06
BLADE_COLOR = (0.80, 0.80, 0.85)

SOFT_CONTACT_MARGIN = 0.015

# The knife stands vertically on the table as a separate kinematic shape
# (body=-1), matching example_mpm_w1_burger_slice_ik. During the scripted
# sequence the robot grasps it; from the grasp point onward the knife's
# shape_transform is updated each frame to follow the right-hand TCP so it
# appears held by the hand.
KNIFE_POS = wp.vec3(0.50, -0.30, TABLE_TOP_Z + BLADE_HZ)
KNIFE_REST_WORLD_POS = wp.vec3(5.63, 0.58, WAIC_WORKTOP_Z + BLADE_HZ + 0.005)
KNIFE_COLOR = (0.80, 0.80, 0.85)

# Right-arm key poses. The wrist orientation is fixed (recorded from the IK
# example); positions are computed in __init__ from the meat size so the cut
# always spans the full meat width and reaches the right depth.
RIGHT_WRIST_QUAT = wp.quat(-0.0711, 0.7045, 0.7017, -0.0789)

# Fixed waypoints (independent of meat size):
# 1. Grasp the knife (hand open -> closed at the knife handle)
RIGHT_HOME_TF = wp.transform(wp.vec3(0.4941, -0.4039, 1.3822), RIGHT_WRIST_QUAT)
# 2. Lift the knife after grasping
RIGHT_LIFT_TF = wp.transform(wp.vec3(0.4941, -0.3997, 1.4564), RIGHT_WRIST_QUAT)
# Final hold pose: higher than the lift, knife held in the air after cutting.
RIGHT_HOLD_TF = wp.transform(wp.vec3(0.55, -0.10, 1.55), RIGHT_WRIST_QUAT)


def _rt(x: float, y: float, z: float) -> wp.transform:
    """Right TCP target at (x, y, z) with the fixed cutting wrist orientation."""
    return wp.transform(wp.vec3(x, y, z), RIGHT_WRIST_QUAT)


# Left hand parks here for the whole sequence (recorded from IK example).
LEFT_HOLD_TF = wp.transform(
    wp.vec3(0.3508, 0.7376, 1.3503),
    wp.quat(-0.1019, 0.7012, -0.6981, 0.1033),
)


class Example:
    """W1 (hands-only) holding a knife, slicing an MPM meat block on a table."""

    BODY_JOINTS = ("BUTTOCK",)
    LEFT_ARM_JOINTS = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM_JOINTS = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    # All five fingers of the right hand — driven between open and grasp poses
    # so the hand pinches the knife handle.
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
        self.waic_robot_base_pos = wp.vec3(
            float(args.waic_robot_base_x),
            float(args.waic_robot_base_y),
            float(args.waic_robot_base_z),
        )
        self.scene_rotation = self._normalize_wp_quat(
            wp.quat(
                float(args.waic_robot_base_qx),
                float(args.waic_robot_base_qy),
                float(args.waic_robot_base_qz),
                float(args.waic_robot_base_qw),
            )
        )
        self.waic_visual_usd = str(args.waic_visual_usd)
        self.hide_waic_visual = bool(args.hide_waic_visual)
        self.show_physics_table = bool(args.show_physics_table)
        self.waic_worktop_z = float(args.waic_worktop_z)
        base_transformed_table_top_z = float(self.waic_robot_base_pos[2]) + TABLE_TOP_Z
        self.rig_z_correction = self.waic_worktop_z - base_transformed_table_top_z
        self.waic_table_top_z = self.waic_worktop_z
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.voxel_size = float(args.voxel_size)

        # Meat geometry (configurable via CLI). The slab is centred at
        # the simplified pan disk, spanning [-w/2, +w/2] in y and sitting on
        # the pan top in world space.
        self.meat_length = float(args.meat_length)
        self.meat_width = float(args.meat_width)
        self.meat_height = float(args.meat_height)
        self.pan_center = wp.vec3(
            float(args.pan_center_x),
            float(args.pan_center_y),
            float(args.pan_center_z),
        )
        self.pan_top_z = float(self.pan_center[2]) + PAN_DISK_HALF_HEIGHT
        self.pan_handle_grasp_tf = wp.transform(
            self._pan_handle_target_pos(0.050),
            self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT),
        )
        self.pan_handle_approach_tf = wp.transform(
            self._pan_handle_target_pos(0.180),
            self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT),
        )
        knife_grasp_pos = wp.vec3(
            float(KNIFE_REST_WORLD_POS[0]),
            float(KNIFE_REST_WORLD_POS[1]),
            float(KNIFE_REST_WORLD_POS[2]) + BLADE_HZ,
        )
        self.knife_rest_tf = wp.transform(KNIFE_REST_WORLD_POS, self.scene_rotation)
        self.knife_grasp_tf = wp.transform(knife_grasp_pos, self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT))
        self.knife_lift_tf = wp.transform(
            wp.vec3(float(knife_grasp_pos[0]), float(knife_grasp_pos[1]), float(knife_grasp_pos[2]) + 0.22),
            self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT),
        )
        self.meat_lo = np.array([
            float(self.pan_center[0]) - 0.5 * self.meat_length,
            float(self.pan_center[1]) - 0.5 * self.meat_width,
            self.pan_top_z + 0.012,
        ])
        self.meat_hi = np.array([
            float(self.pan_center[0]) + 0.5 * self.meat_length,
            float(self.pan_center[1]) + 0.5 * self.meat_width,
            self.pan_top_z + 0.012 + self.meat_height,
        ])

        # Compute the meat-dependent waypoints. The blade bottom is
        # 2*BLADE_HZ below the TCP. Keep it above the pan collision proxy, and
        # use a moderate, slow centered stroke so the motion reads like a real
        # slicing pass without dragging the MPM meat too far sideways.
        cut_z = self.pan_top_z + 0.006 + 2.0 * BLADE_HZ
        cut_stroke = min(0.018, 0.35 * self.meat_width)
        cut_y_lo = float(self.pan_center[1]) - cut_stroke
        cut_y_hi = float(self.pan_center[1]) + cut_stroke
        cut_x = float(self.pan_center[0])
        self.right_above_tf = wp.transform(wp.vec3(cut_x, cut_y_lo, self.pan_top_z + 0.28), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT))
        self.right_descend_tf = wp.transform(wp.vec3(cut_x, cut_y_lo, cut_z), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT))
        self.right_cut_end_tf = wp.transform(wp.vec3(cut_x, cut_y_hi, cut_z), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT))

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        SolverImplicitMPM.register_custom_attributes(builder)

        builder.add_urdf(
            URDF_PATH,
            xform=wp.transform(self.waic_robot_base_pos, self.scene_rotation),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)
        self._add_table(builder)
        self._add_simplified_pan(builder)
        self.knife_shape = self._add_knife(builder)
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

        self.right_tf = self.knife_grasp_tf
        self.left_tf = self._waic_transform(LEFT_HOLD_TF)
        self._knife_prev_pos = wp.transform_get_translation(self.knife_rest_tf)

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.substep_joint_q_prev = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp = self._build_right_hand_targets()
        self.setup_ik()
        self.motion_segments = self._build_motion_segments()
        self.knife_alpha = 0.0

        self._init_mpm_materials(args)

        mpm_options = SolverImplicitMPM.Config()
        for key in vars(args):
            if hasattr(mpm_options, key):
                setattr(mpm_options, key, getattr(args, key))
        mpm_options.collider_velocity_mode = "forward"
        self.solver = SolverImplicitMPM(self.model, mpm_options)
        # Re-setup colliders treating all bodies as kinematic (zero mass): the
        # robot is driven by IK/eval_fk, not by the MPM rigid-body solve, so
        # colliders (table, ground, blade) must not participate as dynamic
        # bodies. Without this the blade's body mass makes the solver treat it
        # as a compliant collider and contacts are silently ignored.
        self.solver.setup_collider(
            model=self.model,
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        self.particle_colors = wp.empty(self.model.particle_count, dtype=wp.vec3, device=self.model.device)
        self.particle_colors.fill_(MEAT_COLOR)

        self.viewer.set_model(self.model)
        CAMERA_POS = wp.vec3(4.10, -1.25, 1.85)
        CAMERA_PITCH = -17.0
        CAMERA_YAW = 62.0
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.viewer.show_particles = True
        self.show_deformation = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        print(
            f"[newton] W1 burger slice: particles={self.model.particle_count}, "
            f"bodies={self.model.body_count}, shapes={self.model.shape_count}"
        )

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
        table_cfg.is_visible = self.show_physics_table
        table_cfg.has_shape_collision = False
        table_cfg.has_particle_collision = False
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self._waic_vec3(TABLE_POS), self.scene_rotation),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR if self.show_physics_table else (0.02, 0.025, 0.03),
            label="waic_hidden_physics_table",
        )
        builder.add_ground_plane(
            height=float(self.waic_robot_base_pos[2]),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5),
            label="waic_kitchen_ground",
        )
        self._add_waic_visual_model(builder)

    def _add_simplified_pan(self, builder: newton.ModelBuilder) -> None:
        pan_cfg = newton.ModelBuilder.ShapeConfig()
        pan_cfg.ke = 8.0e5
        pan_cfg.kd = 1.0e-6
        pan_cfg.mu = 0.9
        pan_cfg.density = 0.0
        pan_cfg.has_particle_collision = True

        self.pan_body = builder.add_body(
            xform=wp.transform(self.pan_center, self.scene_rotation),
            label="waic_simplified_pan_body",
        )
        builder.add_shape_cylinder(
            body=self.pan_body,
            xform=wp.transform_identity(),
            radius=PAN_RADIUS,
            half_height=PAN_DISK_HALF_HEIGHT,
            cfg=pan_cfg,
            color=PAN_DISK_COLOR,
            label="waic_pan_disk",
        )
        builder.add_shape_cylinder(
            body=self.pan_body,
            xform=wp.transform(
                PAN_HANDLE_LOCAL_POS,
                wp.quat(0.0, 0.7071067690849304, 0.0, 0.7071067690849304),
            ),
            radius=PAN_HANDLE_RADIUS,
            half_height=0.5 * PAN_HANDLE_LENGTH,
            cfg=pan_cfg,
            color=PAN_HANDLE_COLOR,
            label="waic_pan_handle",
        )

    def _add_knife(self, builder: newton.ModelBuilder) -> int:
        """Stand the knife vertically on the table as a separate kinematic body.

        The knife is its own body (not body=-1) so that MPM, which bakes
        shape_transform into the collider mesh at setup time and only moves
        colliders via body_q, can follow it once grasped. We drive this
        body's body_q from _update_knife_transform.
        """
        self.knife_body = builder.add_body(
            xform=self.knife_rest_tf,
            label="knife_body",
        )
        knife_cfg = newton.ModelBuilder.ShapeConfig()
        knife_cfg.ke = 1.0e6
        knife_cfg.kd = 1.0e-6
        knife_cfg.mu = 0.02
        knife_cfg.density = 0.0
        return builder.add_shape_box(
            body=self.knife_body,
            xform=wp.transform_identity(),
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
        lo, hi = self.meat_lo, self.meat_hi
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
            jitter=0.15 * radius,
            radius_mean=radius,
        )

    def _configure_particle_contacts(self) -> None:
        # Only table, ground and blade collide with particles; robot body
        # shapes (arms, hands) are excluded so they don't disturb the meat.
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

    def _build_motion_segments(self):
        # (duration, right_start, right_end, hand_start, hand_end, knife_start, knife_end)
        # hand alpha: 0 = hand open, 1 = fully pinching.
        # knife alpha: 0 = knife at rest, 1 = knife follows the right TCP.
        # Sequence recorded from example_mpm_w1_burger_slice_ik:
        #   1. approach the knife and close the fingers to grasp
        #   2. lift the knife
        #   3. move to above the meat
        #   4. lower the blade onto the meat
        #   5. slice across the meat
        #   6. lift, put the knife back down, release it
        #   7. move to the pan handle and close the fingers on it
        knife_rest_tf = self.knife_grasp_tf
        knife_lift_tf = self.knife_lift_tf
        return (
            (1.2, knife_rest_tf, knife_rest_tf, 0.0, 1.0, 0.0, 1.0),
            (0.8, knife_rest_tf, knife_lift_tf, 1.0, 1.0, 1.0, 1.0),
            (1.2, knife_lift_tf, self.right_above_tf, 1.0, 1.0, 1.0, 1.0),
            (0.8, self.right_above_tf, self.right_descend_tf, 1.0, 1.0, 1.0, 1.0),
            (3.2, self.right_descend_tf, self.right_cut_end_tf, 1.0, 1.0, 1.0, 1.0),
            (0.8, self.right_cut_end_tf, self.right_above_tf, 1.0, 1.0, 1.0, 1.0),
            (0.8, self.right_above_tf, knife_lift_tf, 1.0, 1.0, 1.0, 1.0),
            (0.8, knife_lift_tf, knife_rest_tf, 1.0, 1.0, 1.0, 1.0),
            (0.5, knife_rest_tf, knife_rest_tf, 1.0, 1.0, 1.0, 1.0),
            (0.6, knife_rest_tf, knife_rest_tf, 1.0, 0.0, 1.0, 0.0),
            (0.4, knife_rest_tf, knife_rest_tf, 0.0, 0.0, 0.0, 0.0),
            (0.8, knife_rest_tf, self.pan_handle_approach_tf, 0.0, 0.0, 0.0, 0.0),
            (0.8, self.pan_handle_approach_tf, self.pan_handle_grasp_tf, 0.0, 0.0, 0.0, 0.0),
            (0.8, self.pan_handle_grasp_tf, self.pan_handle_grasp_tf, 0.0, 1.0, 0.0, 0.0),
            (2.0, self.pan_handle_grasp_tf, self.pan_handle_grasp_tf, 1.0, 1.0, 0.0, 0.0),
        )

    def _sample_script(self, t: float) -> tuple[wp.transform, float, float]:
        """Return (right TCP transform, hand alpha, knife alpha) at time ``t``."""
        remaining = t
        for duration, start, end, hand_start, hand_end, knife_start, knife_end in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                right_tf = self._interpolate_transform(start, end, alpha)
                hand_alpha = hand_start * (1.0 - alpha) + hand_end * alpha
                knife_alpha = knife_start * (1.0 - alpha) + knife_end * alpha
                return right_tf, hand_alpha, knife_alpha
            remaining -= duration
        _, _, right_end, _, hand_end, _, knife_end = self.motion_segments[-1]
        return right_end, hand_end, knife_end

    # ------------------------------------------------------------------ step
    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        self.right_tf, grasp_alpha, self.knife_alpha = self._sample_script(self.sim_time + self.frame_dt)

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
        # Blend the five fingers between open and grasp poses for this frame.
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp, grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def _update_knife_transform(self, grasp_alpha: float) -> None:
        """Move the knife to follow the right-hand TCP once grasped.

        Before the grasp the knife rests at KNIFE_POS on the table. Once the
        fingers close (grasp_alpha > 0.5) the knife is parented to the right
        TCP: its tip sits at the TCP position and its orientation stays
        vertical. We blend between the rest pose and the held pose across the
        grasp so the knife lifts smoothly off the table.

        The knife is a kinematic body, so updating its body_q moves the MPM
        collider (collider poses are read from body_q each step). The
        collider_velocity_mode is "forward", so body_qd must also be set so
        MPM sees a non-zero blade velocity while cutting.
        """
        tcp_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        tcp_pos = wp.transform_get_translation(tcp_tf)
        # Knife tip aligns with the TCP; centre is BLADE_HZ below along world -z.
        held_pos = wp.vec3(float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2]) - BLADE_HZ)
        held_tf = wp.transform(held_pos, self.scene_rotation)

        a = float(np.clip((grasp_alpha - 0.5) * 2.0, 0.0, 1.0))
        rest_tf = self.knife_rest_tf
        knife_tf = self._interpolate_transform(rest_tf, held_tf, a)

        # Linear velocity from the previous knife pose (for forward mode).
        prev_pos = self._knife_prev_pos
        vel = (wp.transform_get_translation(knife_tf) - prev_pos) * (1.0 / self.sim_dt)
        self._knife_prev_pos = wp.transform_get_translation(knife_tf)

        wp.launch(
            set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.knife_body, knife_tf],
            device=self.model.device,
        )
        wp.launch(
            set_body_velocity_kernel,
            dim=1,
            inputs=[self.state_0.body_qd, self.knife_body, vel],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self._prepare_frame_targets()
        _, _, knife_alpha = self._sample_script(self.sim_time + self.frame_dt)
        for substep in range(self.sim_substeps):
            # store current joint_q as the substep start for velocity differencing
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
            # Once the hand is closing, drive the knife to follow the TCP.
            self._update_knife_transform(knife_alpha)
            self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
            self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self) -> None:
        self.simulate()

    # ----------------------------------------------------------------- render
    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
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
        if np.min(q[:, 2]) < self.waic_table_top_z - 0.05:
            raise ValueError(f"meat fell through the table: z_min={np.min(q[:, 2]):.4f}")
        if np.linalg.norm(q.max(axis=0) - q.min(axis=0)) > 5.0:
            raise ValueError("meat exploded")

    # ----------------------------------------------------------- helpers
    def _pan_handle_target_pos(self, lift: float) -> wp.vec3:
        return wp.vec3(
            float(self.pan_center[0]) + float(PAN_HANDLE_LOCAL_POS[0]),
            float(self.pan_center[1]) + float(PAN_HANDLE_LOCAL_POS[1]),
            self.pan_top_z + lift,
        )

    def _waic_vec3(self, value: wp.vec3) -> wp.vec3:
        local = wp.vec3(
            float(value[0]) - float(OLD_ROBOT_BASE_POS[0]),
            float(value[1]) - float(OLD_ROBOT_BASE_POS[1]),
            float(value[2]) - float(OLD_ROBOT_BASE_POS[2]),
        )
        rotated = self._quat_rotate_vec3(self.scene_rotation, local)
        return wp.vec3(
            float(self.waic_robot_base_pos[0]) + float(rotated[0]),
            float(self.waic_robot_base_pos[1]) + float(rotated[1]),
            float(self.waic_robot_base_pos[2]) + float(rotated[2]) + float(self.rig_z_correction),
        )

    def _waic_transform(self, tf: wp.transform) -> wp.transform:
        return wp.transform(
            self._waic_vec3(wp.transform_get_translation(tf)),
            self._quat_multiply(self.scene_rotation, wp.transform_get_rotation(tf)),
        )

    @staticmethod
    def _normalize_wp_quat(q: wp.quat) -> wp.quat:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        length = (x * x + y * y + z * z + w * w) ** 0.5
        if length == 0.0:
            return wp.quat_identity()
        return wp.quat(x / length, y / length, z / length, w / length)

    @staticmethod
    def _quat_multiply(a: wp.quat, b: wp.quat) -> wp.quat:
        ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _quat_rotate_vec3(q: wp.quat, v: wp.vec3) -> wp.vec3:
        qv = wp.quat(float(v[0]), float(v[1]), float(v[2]), 0.0)
        qc = wp.quat(-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))
        rotated = Example._quat_multiply(Example._quat_multiply(q, qv), qc)
        return wp.vec3(float(rotated[0]), float(rotated[1]), float(rotated[2]))

    def _add_waic_visual_model(self, builder: newton.ModelBuilder) -> None:
        if self.hide_waic_visual or not self.waic_visual_usd:
            return
        if not os.path.exists(self.waic_visual_usd):
            print(f"WAIC selected visual USD not found, skipping: {self.waic_visual_usd}")
            return

        stage = Usd.Stage.Open(self.waic_visual_usd)
        if stage is None:
            print(f"WAIC selected visual USD failed to open, skipping: {self.waic_visual_usd}")
            return

        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.density = 0.0
        visual_cfg.collision_group = 0
        visual_cfg.has_shape_collision = False
        visual_cfg.has_particle_collision = False
        visual_cfg.is_visible = True

        xform_cache = UsdGeom.XformCache()
        mesh_count = 0
        tri_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            usd_mesh = UsdGeom.Mesh(prim)
            vertices, indices = self._usd_mesh_to_newton_arrays(usd_mesh, xform_cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=-1,
                mesh=mesh,
                xform=wp.transform_identity(),
                cfg=visual_cfg,
                color=self._semantic_visual_color(str(prim.GetPath())),
                label=f"waic_kitchen_visual_{mesh_count:03d}",
            )
            mesh_count += 1
            tri_count += int(len(indices) // 3)
        print(f"Loaded WAIC kitchen visual: {mesh_count} meshes, {tri_count} triangles from {self.waic_visual_usd}")

    @staticmethod
    def _usd_mesh_to_newton_arrays(usd_mesh: UsdGeom.Mesh, xform_cache: UsdGeom.XformCache):
        points = usd_mesh.GetPointsAttr().Get()
        face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)

        world_from_local = xform_cache.GetLocalToWorldTransform(usd_mesh.GetPrim())
        vertices = np.empty((len(points), 3), dtype=np.float32)
        for i, point in enumerate(points):
            p = world_from_local.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            vertices[i] = (float(p[0]), float(p[1]), float(p[2]))

        triangles: list[int] = []
        cursor = 0
        for count in face_counts:
            count = int(count)
            if count >= 3:
                face = face_indices[cursor : cursor + count]
                for i in range(1, count - 1):
                    triangles.extend((int(face[0]), int(face[i]), int(face[i + 1])))
            cursor += count

        return vertices, np.asarray(triangles, dtype=np.int32)

    @staticmethod
    def _semantic_visual_color(prim_path: str) -> tuple[float, float, float]:
        text = prim_path.lower()
        if "frying_pan" in text or "pan_" in text:
            if "rivet" in text or "rim" in text or "ring" in text or "hole" in text:
                return (0.58, 0.57, 0.54)
            return (0.025, 0.026, 0.028)
        if "table" in text:
            return (0.52, 0.37, 0.23)
        if "kitchen" in text or "furniture" in text:
            return (0.70, 0.68, 0.62)
        return (0.62, 0.60, 0.56)

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

    def _slerp_quat_xyzw(self, qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
        qa = self._normalize_quat(qa)
        qb = self._normalize_quat(qb)
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
        return qa * (np.sin(theta_0 - theta) / sin_theta_0) + qb * (np.sin(theta) / sin_theta_0)

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _builder_body_index(self, builder: newton.ModelBuilder, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(builder.body_label) if label.endswith(suffix))

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
            for name in (*self.BODY_JOINTS, *self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS, *self.RIGHT_HAND_JOINTS)
        }

    def _build_right_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        """Build open/grasp joint values for all five right-hand fingers.

        Grasp values are set to roughly 2/3 of each joint's upper limit so the
        fingers curl around the knife handle without fully bottoming out.
        """
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        # Grasp target as a fraction of the joint's upper limit (lower is 0).
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

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=510)

        parser.add_argument("--voxel-size", "-dx", type=float, default=0.016)
        parser.add_argument("--particles-per-cell", "-ppc", type=int, default=2)
        parser.add_argument("--density", type=float, default=1000.0)

        # Meat block dimensions [m]: length (x), width (y, cut direction), height (z).
        parser.add_argument("--meat-length", type=float, default=MEAT_LENGTH, help="Meat block length [m] (x).")
        parser.add_argument("--meat-width", type=float, default=MEAT_WIDTH, help="Meat block width [m] (y, cut direction).")
        parser.add_argument("--meat-height", type=float, default=MEAT_HEIGHT, help="Meat block height [m] (z).")

        parser.add_argument("--young-modulus", "-ym", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.45)
        parser.add_argument("--friction", "-mu", type=float, default=0.25)
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
        parser.add_argument("--waic-robot-base-x", type=float, default=float(WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(WAIC_ROBOT_BASE_QUAT[3]))
        parser.add_argument("--pan-center-x", type=float, default=float(PAN_CENTER_WORLD[0]))
        parser.add_argument("--pan-center-y", type=float, default=float(PAN_CENTER_WORLD[1]))
        parser.add_argument("--pan-center-z", type=float, default=float(PAN_CENTER_WORLD[2]))
        parser.add_argument(
            "--waic-worktop-z",
            type=float,
            default=WAIC_WORKTOP_Z,
            help="World Z of the visual worktop/frying-pan support surface.",
        )
        parser.add_argument(
            "--waic-visual-usd",
            type=str,
            default=WAIC_SELECTED_VISUAL_USD,
            help="USD containing the selected WAIC kitchen counter/table visual assets.",
        )
        parser.add_argument(
            "--hide-waic-visual",
            action="store_true",
            help="Skip loading the selected WAIC kitchen visual meshes.",
        )
        parser.add_argument(
            "--show-physics-table",
            action="store_true",
            help="Render the Newton physics table box visibly for alignment/debugging.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
