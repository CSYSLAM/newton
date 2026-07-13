# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice in the WAIC full-house kitchen scene V6
#
# Standalone version: contains the V4 burger slicing setup, the full-house
# visual background, the right-turn scripted carry path, and the V5 final
# lowered-table pan placement. It intentionally does not import the intermediate
# V4/house/path/V5 example files.
#
# Command: python -m newton.examples mpm_w1_burger_slice_waic_kitchen_V6
#
###########################################################################

from __future__ import annotations

import math
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
def copy_joint_q_to_ik_kernel(
    joint_q: wp.array[wp.float32],
    ik_joint_q: wp.array2d[wp.float32],
):
    i = wp.tid()
    ik_joint_q[0, i] = joint_q[i]


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
def set_indexed_ik_joint_q_kernel(
    q_indices: wp.array[wp.int32],
    start_values: wp.array[wp.float32],
    end_values: wp.array[wp.float32],
    alpha: float,
    ik_joint_q: wp.array2d[wp.float32],
):
    i = wp.tid()
    ik_joint_q[0, q_indices[i]] = start_values[i] * (1.0 - alpha) + end_values[i] * alpha


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


@wp.kernel
def set_transformed_body_poses_kernel(
    body_q_start: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    global_tf: wp.transform,
    inv_dt: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    body_index = body_indices[i]
    start_tf = body_q_start[body_index]
    prev_tf = body_q[body_index]
    global_rot = wp.transform_get_rotation(global_tf)
    new_tf = wp.transform(
        wp.transform_point(global_tf, wp.transform_get_translation(start_tf)),
        global_rot * wp.transform_get_rotation(start_tf),
    )
    body_q[body_index] = new_tf
    vel = (wp.transform_get_translation(new_tf) - wp.transform_get_translation(prev_tf)) * inv_dt
    body_qd[body_index] = wp.spatial_vector(vel, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def set_transformed_shape_transforms_kernel(
    shape_transform_start: wp.array[wp.transform],
    shape_indices: wp.array[wp.int32],
    global_tf: wp.transform,
    shape_transform: wp.array[wp.transform],
):
    i = wp.tid()
    shape_index = shape_indices[i]
    start_tf = shape_transform_start[shape_index]
    global_rot = wp.transform_get_rotation(global_tf)
    shape_transform[shape_index] = wp.transform(
        wp.transform_point(global_tf, wp.transform_get_translation(start_tf)),
        global_rot * wp.transform_get_rotation(start_tf),
    )


@wp.kernel
def apply_body_transform_kernel(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[wp.int32],
    global_tf: wp.transform,
):
    i = wp.tid()
    body_index = body_indices[i]
    body_tf = body_q[body_index]
    global_rot = wp.transform_get_rotation(global_tf)
    body_q[body_index] = wp.transform(
        wp.transform_point(global_tf, wp.transform_get_translation(body_tf)),
        global_rot * wp.transform_get_rotation(body_tf),
    )


@wp.kernel
def apply_particle_delta_transform_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    delta_tf: wp.transform,
    inv_dt: float,
):
    i = wp.tid()
    old_pos = particle_q[i]
    new_pos = wp.transform_point(delta_tf, old_pos)
    particle_q[i] = new_pos
    particle_qd[i] = (new_pos - old_pos) * inv_dt


# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
URDF_PATH = (
    Path(__file__).resolve().parents[1] / "cloth" / "DexforceW1V021" / "DexforceW1V021.urdf"
)
HOUSE_VISUAL_USD = (
    r"E:\csy_work\CG\assets\WAIC\house_background"
    r"\House5_Simple2_visual.usd"
)
PAN_VISUAL_USD = r"E:\csy_work\CG\assets\WAIC\house_background\waic_frying_pan.usd"
WAIC_SELECTED_VISUAL_USD = HOUSE_VISUAL_USD

OLD_ROBOT_BASE_POS = wp.vec3(0.0, 0.0, 0.0)
WAIC_ROBOT_BASE_POS = wp.vec3(5.156154155731201, 0.6696404814720154, -0.0037720240652561188)
WAIC_ROBOT_BASE_QUAT = wp.quat_identity()
WAIC_WORKTOP_Z = 0.9000003337860107

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
PAN_HANDLE_GRASP_TF = wp.transform(
    wp.vec3(5.5512, 0.3046, 0.8566),
    wp.quat(0.6680, -0.0625, 0.7413, 0.0168),
)
PAN_HANDLE_CARRY_TF = wp.transform(
    wp.vec3(5.5512, 0.3046, 1.0166),
    wp.quat(0.6421, 0.1946, 0.6927, -0.2647),
)
PAN_HANDLE_APPROACH_LIFT = 0.12
PAN_HANDLE_LIFT_HEIGHT = 0.16
PAN_CARRY_BACK_DISTANCE = 0.28
PAN_CARRY_RIGHT_SHIFT = -0.10
PAN_CARRY_TURN_DEGREES = -180.0
PAN_CARRY_TIME = 10.0
PAN_PLACE_TIME = 3.0
PAN_GRASP_SETTLE_TIME = 1.0
PAN_PLACE_SETTLE_TIME = 1.0
PAN_HAND_CLEAR_TIME = 2.5
PAN_CARRY_GRIP_RESTORE_TIME = 1.0
PAN_HAND_GRASP_ALPHA = 1.0
PAN_HAND_CLEAR_SIDE_OFFSET = 0.28
PAN_HAND_CLEAR_LIFT = 0.02
PAN_HIGH_FRICTION = 2.5
REAR_TABLE_HALF_EXTENTS = (0.34, 0.28, 0.025)

RIGHT_TURN_DEGREES = -90.0
ROOT_TARGETS = (
    wp.vec3(4.8, -0.85, 0.29),
    wp.vec3(3.0, -0.85, 0.29),
    wp.vec3(2.8, 2.22, 0.29),
)

V6_TABLE_TOP_Z = 0.9000
V6_TABLE_NEAR_EDGE_X = 3.45926
V6_TABLE_CENTER_Y = 1.881144
V6_TABLE_BOX_HALF_EXTENTS = (0.58, 0.50, 0.025)
V6_TABLE_BOX_CENTER = wp.vec3(
    V6_TABLE_NEAR_EDGE_X + V6_TABLE_BOX_HALF_EXTENTS[0],
    V6_TABLE_CENTER_Y,
    V6_TABLE_TOP_Z - V6_TABLE_BOX_HALF_EXTENTS[2],
)
PAN_TABLE_EDGE_CLEARANCE = 0.060
PAN_APPROACH_CLEARANCE_Z = 0.030
V6_MEAT_PAN_GAP = 0.003
V6_KNIFE_PAN_CLEARANCE = -0.008
V6_PAN_PLACE_CENTER = wp.vec3(
    V6_TABLE_NEAR_EDGE_X + PAN_RADIUS + PAN_TABLE_EDGE_CLEARANCE,
    V6_TABLE_CENTER_Y,
    V6_TABLE_TOP_Z + PAN_DISK_HALF_HEIGHT,
)

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


# Left hand parks here for the whole sequence. Kept close to the body so it
# does not clip surrounding kitchen geometry while the base turns/moves along
# the scripted carry path. Mirrored to the right arm's home Y magnitude.
LEFT_HOLD_TF = wp.transform(
    wp.vec3(0.20, 0.28, 1.20),
    wp.quat(-0.1019, 0.7012, -0.6981, 0.1033),
)


class Example:
    """W1 (hands-only) holding a knife, slicing an MPM meat block on a table."""

    BODY_JOINTS = ("ANKLE", "KNEE", "BUTTOCK")
    TORSO_REACH_JOINT_TARGETS = {
        "ANKLE": 0.60,
        "KNEE": -0.90,
        "BUTTOCK": 0.80,
    }
    TORSO_REACH_RAMP_TIME = 0.6
    TORSO_RETURN_TIME = 1.0
    TORSO_GRASP_ALPHA = 0.25
    TORSO_CUT_ALPHA = 0.70
    TORSO_POST_PLACE_ALPHA = 0.55
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
        self.turn_time = float(args.path_turn_time)
        self.move_time = float(args.path_move_time)
        self.path_duration = 4.0 * self.turn_time + 3.0 * self.move_time
        self.path_start_root_tf: wp.transform | None = None
        self.gizmo_carry_tf = wp.transform_identity()
        self.post_forward_time = float(args.post_forward_time)
        self.post_lower_time = float(args.post_lower_time)
        self.post_release_time = float(args.post_release_time)
        self.post_clear_time = float(args.post_clear_time)
        self.pose_print_interval = float(args.pose_print_interval)
        self.last_pose_print_time = -1.0e9
        self.post_mode = False
        self.post_start_time = 0.0
        self.post_start_right_tf = wp.transform_identity()
        self.post_start_pan_tf = wp.transform_identity()
        self.post_restore_right_tf = wp.transform_identity()
        self.post_right_local_pos = wp.vec3(0.0, 0.0, 0.0)
        self.post_right_local_rot = wp.quat_identity()
        self.post_approach_tf = wp.transform_identity()
        self.post_place_tf = wp.transform_identity()
        self.post_place_right_tf = wp.transform_identity()
        self.post_clear_tf = wp.transform_identity()
        self.post_prev_pan_tf = wp.transform_identity()
        self.post_current_hand_alpha = 1.0
        self.post_current_elapsed = 0.0
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
        self.pan_visual_usd = str(args.pan_visual_usd)
        self.pan_visual_axis_mode = str(args.pan_visual_axis_mode)
        self.pan_visual_offset = wp.vec3(
            float(args.pan_visual_offset_x),
            float(args.pan_visual_offset_y),
            float(args.pan_visual_offset_z),
        )
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
        self.pan_handle_grasp_tf = PAN_HANDLE_GRASP_TF
        self.pan_handle_approach_tf = wp.transform(
            wp.transform_get_translation(self.pan_handle_grasp_tf) + wp.vec3(0.0, 0.0, PAN_HANDLE_APPROACH_LIFT),
            wp.transform_get_rotation(self.pan_handle_grasp_tf),
        )
        self.pan_handle_lift_tf = PAN_HANDLE_CARRY_TF
        self.pan_rest_tf = wp.transform(self.pan_center, self.scene_rotation)
        self.pan_lift_tf = wp.transform(
            self.pan_center + wp.vec3(0.0, 0.0, PAN_HANDLE_LIFT_HEIGHT),
            self.scene_rotation,
        )
        self.rigid_carry_end_tf = self._global_carry_transform(1.0)
        self.pan_place_start_tf = self._compose_transform(self.rigid_carry_end_tf, self.pan_lift_tf)
        self.rear_table_top_z = self.waic_worktop_z
        self.pan_place_drop = max(
            float(wp.transform_get_translation(self.pan_place_start_tf)[2])
            - (self.rear_table_top_z + PAN_DISK_HALF_HEIGHT),
            0.0,
        )
        self.pan_place_end_tf = wp.transform(
            wp.transform_get_translation(self.pan_place_start_tf) - wp.vec3(0.0, 0.0, self.pan_place_drop),
            wp.transform_get_rotation(self.pan_place_start_tf),
        )
        self.rear_table_pos, self.rear_table_half_extents = self._compute_rear_table_proxy()
        self.pan_handle_local_pos, self.pan_handle_local_rot = self._relative_transform(
            self.pan_rest_tf,
            self.pan_handle_grasp_tf,
        )
        self.pan_handle_placed_pre_carry_tf, self.pan_handle_clear_pre_carry_tf = self._build_post_place_hand_targets()
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
        self.meat_lo = np.array(
            [
                float(self.pan_center[0]) - 0.5 * self.meat_length,
                float(self.pan_center[1]) - 0.5 * self.meat_width,
                self.pan_top_z + float(args.v6_meat_pan_gap),
            ]
        )
        self.meat_hi = np.array(
            [
                float(self.pan_center[0]) + 0.5 * self.meat_length,
                float(self.pan_center[1]) + 0.5 * self.meat_width,
                self.pan_top_z + float(args.v6_meat_pan_gap) + self.meat_height,
            ]
        )

        # Compute the meat-dependent waypoints. The blade bottom is
        # 2*BLADE_HZ below the TCP. Keep it just above the pan collision proxy
        # while dipping below the lowest meat particle centers so the cut
        # reaches the final layer without visually entering the pan.
        cut_z = self.pan_top_z + float(args.v6_knife_pan_clearance) + 2.0 * BLADE_HZ
        cut_stroke = min(0.018, 0.35 * self.meat_width)
        cut_y_lo = float(self.pan_center[1]) - cut_stroke
        cut_y_hi = float(self.pan_center[1]) + cut_stroke
        cut_x = float(self.pan_center[0])
        self.right_above_tf = wp.transform(
            wp.vec3(cut_x, cut_y_lo, self.pan_top_z + 0.28), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT)
        )
        self.right_descend_tf = wp.transform(
            wp.vec3(cut_x, cut_y_lo, cut_z), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT)
        )
        self.right_cut_end_tf = wp.transform(
            wp.vec3(cut_x, cut_y_hi, cut_z), self._quat_multiply(self.scene_rotation, RIGHT_WRIST_QUAT)
        )

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
        self._add_rear_table(builder)
        self._add_simplified_pan(builder)
        self._add_waic_visual_model(builder)
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
        self._pan_prev_pos = wp.transform_get_translation(self.pan_rest_tf)

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.substep_joint_q_prev = wp.zeros_like(self.model.joint_q)
        self.rigid_robot_body_indices = self._build_rigid_robot_body_indices()
        self.rigid_place_body_indices = self._build_rigid_place_body_indices()
        self.rigid_pan_body_indices = wp.array([self.pan_body], dtype=wp.int32, device=self.model.device)
        self.rigid_carry_shape_indices = self._build_rigid_carry_shape_indices()
        self.rigid_carry_body_q_start = wp.empty_like(self.state_0.body_q)
        self.rigid_carry_shape_transform_start = wp.empty_like(self.model.shape_transform)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        (
            self.torso_q_indices,
            self.torso_dof_indices,
            self.torso_q_home,
            self.torso_q_reach,
        ) = self._build_torso_reach_targets()
        self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp = self._build_right_hand_targets()
        self.setup_ik()
        self.motion_segments = self._build_motion_segments()
        self.knife_sequence_duration = sum(segment[0] for segment in self.motion_segments[:11])
        self.rigid_carry_start_time = sum(segment[0] for segment in self.motion_segments)
        self.rigid_carry_prev_object_tf = wp.transform_identity()
        self.rigid_carry_prev_alpha = 0.0
        self.rigid_place_prev_alpha = 0.0
        self.rigid_carry_initialized = False
        self.knife_alpha = 0.0
        self.pan_alpha = 0.0
        self.pan_high_friction_enabled = False
        self.pan_material_ids: wp.array | None = None

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
        self.pan_material_ids = self._build_pan_material_ids()

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

    def _compute_rear_table_proxy(self) -> tuple[wp.vec3, tuple[float, float, float]]:
        pan_place_pos = wp.transform_get_translation(self.pan_place_end_tf)
        hx, hy, hz = REAR_TABLE_HALF_EXTENTS
        top_z = self.rear_table_top_z

        table_bounds = self._visual_table_bounds()
        if table_bounds is not None:
            table_min, table_max = table_bounds
            top_z = min(top_z, float(table_max[2]))
            right_room = float(table_max[0]) - float(pan_place_pos[0])
            if right_room > PAN_RADIUS:
                hx = min(hx, right_room)

            y_room = min(
                float(pan_place_pos[1]) - float(table_min[1]),
                float(table_max[1]) - float(pan_place_pos[1]),
            )
            if y_room > PAN_RADIUS:
                hy = min(hy, y_room)

        pos = wp.vec3(float(pan_place_pos[0]), float(pan_place_pos[1]), top_z - hz)
        return pos, (hx, hy, hz)

    def _visual_table_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.waic_visual_usd or not os.path.exists(self.waic_visual_usd):
            return None
        stage = Usd.Stage.Open(self.waic_visual_usd)
        if stage is None:
            return None

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True)
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath()).lower()
            if not prim.IsA(UsdGeom.Mesh) or "table" not in prim_path or self._is_pan_visual(prim_path):
                continue
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bounds_min = aligned.GetMin()
            bounds_max = aligned.GetMax()
            mins.append(np.array([bounds_min[0], bounds_min[1], bounds_min[2]], dtype=np.float32))
            maxs.append(np.array([bounds_max[0], bounds_max[1], bounds_max[2]], dtype=np.float32))

        if not mins:
            return None
        return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)

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

    def _add_rear_table(self, builder: newton.ModelBuilder) -> None:
        collision_cfg = newton.ModelBuilder.ShapeConfig()
        collision_cfg.ke = 8.0e5
        collision_cfg.kd = 1.0e-6
        collision_cfg.mu = 1.2
        collision_cfg.density = 0.0
        collision_cfg.has_shape_collision = False
        collision_cfg.has_particle_collision = True
        collision_cfg.is_visible = False
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(V6_TABLE_BOX_CENTER, wp.quat_identity()),
            hx=V6_TABLE_BOX_HALF_EXTENTS[0],
            hy=V6_TABLE_BOX_HALF_EXTENTS[1],
            hz=V6_TABLE_BOX_HALF_EXTENTS[2],
            cfg=collision_cfg,
            color=(0.20, 0.24, 0.28),
            label="waic_v6_destination_hidden_table_box",
        )

    def _add_simplified_pan(self, builder: newton.ModelBuilder) -> None:
        pan_cfg = newton.ModelBuilder.ShapeConfig()
        pan_cfg.ke = 8.0e5
        pan_cfg.kd = 1.0e-6
        pan_cfg.mu = 0.9
        pan_cfg.density = 0.0
        pan_cfg.has_particle_collision = True
        pan_cfg.is_visible = False

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
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

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
        self.ik_limit_lower_base = lower
        self.ik_limit_upper_base = upper
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
        # (duration, right_start, right_end, hand_start, hand_end, knife_start, knife_end, pan_start, pan_end)
        # hand alpha: 0 = hand open, 1 = fully pinching.
        # knife alpha: 0 = knife at rest, 1 = knife follows the right TCP.
        # pan alpha: 0 = pan at rest, 1 = pan lifted with the grasped handle.
        # Sequence recorded from example_mpm_w1_burger_slice_ik:
        #   1. approach the knife and close the fingers to grasp
        #   2. lift the knife
        #   3. move to above the meat
        #   4. lower the blade onto the meat
        #   5. slice across the meat
        #   6. lift, put the knife back down, release it
        #   7. move to the pan handle, close the fingers on it, then lift
        knife_rest_tf = self.knife_grasp_tf
        knife_lift_tf = self.knife_lift_tf
        return (
            (1.2, knife_rest_tf, knife_rest_tf, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            (0.8, knife_rest_tf, knife_lift_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (1.2, knife_lift_tf, self.right_above_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.8, self.right_above_tf, self.right_descend_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (3.2, self.right_descend_tf, self.right_cut_end_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.8, self.right_cut_end_tf, self.right_above_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.8, self.right_above_tf, knife_lift_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.8, knife_lift_tf, knife_rest_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.5, knife_rest_tf, knife_rest_tf, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
            (0.6, knife_rest_tf, knife_rest_tf, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (0.4, knife_rest_tf, knife_rest_tf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.2, knife_rest_tf, self.pan_handle_approach_tf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, self.pan_handle_approach_tf, self.pan_handle_grasp_tf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.9, self.pan_handle_grasp_tf, self.pan_handle_grasp_tf, 0.0, PAN_HAND_GRASP_ALPHA, 0.0, 0.0, 0.0, 0.0),
            (
                PAN_GRASP_SETTLE_TIME,
                self.pan_handle_grasp_tf,
                self.pan_handle_grasp_tf,
                PAN_HAND_GRASP_ALPHA,
                PAN_HAND_GRASP_ALPHA,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                3.0,
                self.pan_handle_grasp_tf,
                self.pan_handle_lift_tf,
                PAN_HAND_GRASP_ALPHA,
                PAN_HAND_GRASP_ALPHA,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            (
                0.6,
                self.pan_handle_lift_tf,
                self.pan_handle_lift_tf,
                PAN_HAND_GRASP_ALPHA,
                PAN_HAND_GRASP_ALPHA,
                0.0,
                0.0,
                1.0,
                1.0,
            ),
        )

    def _sample_script(self, t: float) -> tuple[wp.transform, float, float, float]:
        """Return (right TCP transform, hand alpha, knife alpha, pan alpha) at time ``t``."""
        remaining = t
        for (
            duration,
            start,
            end,
            hand_start,
            hand_end,
            knife_start,
            knife_end,
            pan_start,
            pan_end,
        ) in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                right_tf = self._interpolate_transform(start, end, alpha)
                hand_alpha = hand_start * (1.0 - alpha) + hand_end * alpha
                knife_alpha = knife_start * (1.0 - alpha) + knife_end * alpha
                pan_alpha = pan_start * (1.0 - alpha) + pan_end * alpha
                return right_tf, hand_alpha, knife_alpha, pan_alpha
            remaining -= duration
        _, _, right_end, _, hand_end, _, knife_end, _, pan_end = self.motion_segments[-1]
        return right_end, hand_end, knife_end, pan_end

    def _sample_post_place_script(self, t: float) -> tuple[wp.transform, float] | None:
        post_t = float(t) - (self.rigid_carry_start_time + PAN_CARRY_TIME + PAN_PLACE_TIME)
        if post_t < 0.0:
            return None

        if post_t <= PAN_PLACE_SETTLE_TIME:
            return self.pan_handle_placed_pre_carry_tf, PAN_HAND_GRASP_ALPHA

        clear_t = post_t - PAN_PLACE_SETTLE_TIME
        if clear_t <= PAN_HAND_CLEAR_TIME:
            alpha = self._smoothstep(clear_t / PAN_HAND_CLEAR_TIME)
            right_tf = self._interpolate_transform(
                self.pan_handle_placed_pre_carry_tf,
                self.pan_handle_clear_pre_carry_tf,
                alpha,
            )
            return right_tf, PAN_HAND_GRASP_ALPHA * (1.0 - alpha)

        return self.pan_handle_clear_pre_carry_tf, 0.0

    # ------------------------------------------------------------------ step
    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        wp.launch(
            copy_joint_q_to_ik_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.state_0.joint_q],
            outputs=[self.ik_joint_q],
            device=self.model.device,
        )
        target_time = self.sim_time + self.frame_dt
        torso_alpha = self._scripted_torso_reach_alpha(target_time)
        self.right_tf, grasp_alpha, self.knife_alpha, self.pan_alpha = self._sample_script(target_time)
        post_place_target = self._sample_post_place_script(target_time)
        if post_place_target is not None:
            self.right_tf, grasp_alpha = post_place_target
            self.knife_alpha = 0.0
            self.pan_alpha = 1.0
            torso_alpha = 0.0

        self._seed_torso_reach_posture(torso_alpha)
        self._set_torso_ik_constraints(torso_alpha)

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

    def _scripted_torso_reach_alpha(self, target_time: float) -> float:
        if target_time <= 0.0:
            return 0.0

        grasp_alpha = self.TORSO_GRASP_ALPHA * self._smooth_time_window(
            target_time,
            start=0.0,
            end=sum(segment[0] for segment in self.motion_segments[:2]),
            ramp_in=0.5,
            ramp_out=0.5,
        )
        cut_alpha = self.TORSO_CUT_ALPHA * self._smooth_time_window(
            target_time,
            start=sum(segment[0] for segment in self.motion_segments[:3]),
            end=sum(segment[0] for segment in self.motion_segments[:6]),
            ramp_in=0.7,
            ramp_out=0.5,
        )
        place_alpha = self.TORSO_GRASP_ALPHA * self._smooth_time_window(
            target_time,
            start=sum(segment[0] for segment in self.motion_segments[:6]),
            end=sum(segment[0] for segment in self.motion_segments[:10]),
            ramp_in=0.5,
            ramp_out=0.5,
        )
        return max(grasp_alpha, cut_alpha, place_alpha)

    def _smooth_time_window(self, t: float, *, start: float, end: float, ramp_in: float, ramp_out: float) -> float:
        if t <= start or t >= end:
            return 0.0
        if ramp_in > 0.0 and t < start + ramp_in:
            return self._smoothstep((t - start) / ramp_in)
        if ramp_out > 0.0 and t > end - ramp_out:
            return 1.0 - self._smoothstep((t - (end - ramp_out)) / ramp_out)
        return 1.0

    def _post_place_torso_reach_alpha(self, elapsed: float) -> float:
        release_end = (
            PAN_CARRY_GRIP_RESTORE_TIME + self.post_forward_time + self.post_lower_time + self.post_release_time
        )
        clear_end = release_end + self.post_clear_time
        return self.TORSO_POST_PLACE_ALPHA * self._smooth_time_window(
            elapsed,
            start=0.0,
            end=clear_end,
            ramp_in=0.6,
            ramp_out=self.post_clear_time,
        )

    def _seed_torso_reach_posture(self, alpha: float) -> None:
        wp.launch(
            set_indexed_ik_joint_q_kernel,
            dim=self.torso_q_indices.shape[0],
            inputs=[
                self.torso_q_indices,
                self.torso_q_home,
                self.torso_q_reach,
                float(np.clip(alpha, 0.0, 1.0)),
            ],
            outputs=[self.ik_joint_q],
            device=self.model.device,
        )

    def _set_torso_ik_constraints(self, alpha: float) -> None:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        lower = self.ik_limit_lower_base.copy()
        upper = self.ik_limit_upper_base.copy()
        target = self.torso_q_home_np * (1.0 - alpha) + self.torso_q_reach_np * alpha
        lower[self.torso_dof_indices_np] = target - 1.0e-4
        upper[self.torso_dof_indices_np] = target + 1.0e-4
        self.joint_limits_obj.joint_limit_lower = wp.array(lower, dtype=wp.float32, device=self.model.device)
        self.joint_limits_obj.joint_limit_upper = wp.array(upper, dtype=wp.float32, device=self.model.device)

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

    def _update_pan_transform(self, pan_alpha: float) -> None:
        """Lift the simplified pan body after the right hand grasps the handle."""
        self._update_pan_friction(pan_alpha)

        if pan_alpha > 0.0:
            pan_tf = self._pan_transform_from_handle(self.right_tf)
        else:
            pan_tf = self.pan_rest_tf

        prev_pos = self._pan_prev_pos
        vel = (wp.transform_get_translation(pan_tf) - prev_pos) * (1.0 / self.sim_dt)
        self._pan_prev_pos = wp.transform_get_translation(pan_tf)

        wp.launch(
            set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.pan_body, pan_tf],
            device=self.model.device,
        )
        wp.launch(
            set_body_velocity_kernel,
            dim=1,
            inputs=[self.state_0.body_qd, self.pan_body, vel],
            device=self.model.device,
        )

    def _update_pan_friction(self, pan_alpha: float) -> None:
        if self.pan_high_friction_enabled or pan_alpha < 0.999 or self.pan_material_ids is None:
            return

        collider = self.solver._mpm_model.collider
        friction = collider.material_friction.numpy()
        friction[self.pan_material_ids.numpy()] = PAN_HIGH_FRICTION
        collider.material_friction = wp.array(friction, dtype=wp.float32, device=self.model.device)
        self.pan_high_friction_enabled = True

    def simulate(self) -> None:
        self._prepare_frame_targets()
        knife_alpha = self.knife_alpha
        pan_alpha = self.pan_alpha
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
            self._update_pan_transform(pan_alpha)
            carry_active = self._apply_rigid_carry_transform(self.sim_time + (substep + 1) * self.sim_dt)
            if not carry_active:
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

    def _apply_rigid_carry_transform(self, query_time: float) -> bool:
        object_tf, carry_tf, robot_tf, carry_alpha, place_alpha = self._rigid_carry_transform(query_time)
        if carry_alpha <= 0.0:
            return False

        if not self.rigid_carry_initialized:
            wp.copy(self.rigid_carry_body_q_start, self.state_0.body_q)
            wp.copy(self.rigid_carry_shape_transform_start, self.model.shape_transform)
            self.rigid_carry_initialized = True

        post_place_active = place_alpha >= 0.999
        if post_place_active:
            wp.launch(
                apply_body_transform_kernel,
                dim=self.rigid_robot_body_indices.shape[0],
                inputs=[self.state_0.body_q, self.rigid_robot_body_indices, robot_tf],
                device=self.model.device,
            )
        else:
            wp.launch(
                set_transformed_body_poses_kernel,
                dim=self.rigid_robot_body_indices.shape[0],
                inputs=[
                    self.rigid_carry_body_q_start,
                    self.rigid_robot_body_indices,
                    carry_tf,
                    1.0 / self.sim_dt,
                ],
                outputs=[self.state_0.body_q, self.state_0.body_qd],
                device=self.model.device,
            )
            if place_alpha > 0.0:
                wp.launch(
                    set_transformed_body_poses_kernel,
                    dim=self.rigid_place_body_indices.shape[0],
                    inputs=[
                        self.rigid_carry_body_q_start,
                        self.rigid_place_body_indices,
                        object_tf,
                        1.0 / self.sim_dt,
                    ],
                    outputs=[self.state_0.body_q, self.state_0.body_qd],
                    device=self.model.device,
                )

        wp.launch(
            set_transformed_body_poses_kernel,
            dim=self.rigid_pan_body_indices.shape[0],
            inputs=[
                self.rigid_carry_body_q_start,
                self.rigid_pan_body_indices,
                object_tf,
                1.0 / self.sim_dt,
            ],
            outputs=[self.state_0.body_q, self.state_0.body_qd],
            device=self.model.device,
        )
        if self.rigid_carry_shape_indices.shape[0] > 0:
            wp.launch(
                set_transformed_shape_transforms_kernel,
                dim=self.rigid_carry_shape_indices.shape[0],
                inputs=[self.rigid_carry_shape_transform_start, self.rigid_carry_shape_indices, robot_tf],
                outputs=[self.model.shape_transform],
                device=self.model.device,
            )

        if carry_alpha > self.rigid_carry_prev_alpha + 1.0e-6 or place_alpha > self.rigid_place_prev_alpha + 1.0e-6:
            delta_tf = self._compose_transform(object_tf, self._transform_inverse(self.rigid_carry_prev_object_tf))
            wp.launch(
                apply_particle_delta_transform_kernel,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_q, self.state_0.particle_qd, delta_tf, 1.0 / self.sim_dt],
                device=self.model.device,
            )

        self.rigid_carry_prev_object_tf = object_tf
        self.rigid_carry_prev_alpha = carry_alpha
        self.rigid_place_prev_alpha = place_alpha
        return True

    def _rigid_carry_transform(
        self, query_time: float
    ) -> tuple[wp.transform, wp.transform, wp.transform, float, float]:
        carry_t = max(float(query_time) - self.rigid_carry_start_time, 0.0)
        carry_alpha = self._smoothstep(carry_t / PAN_CARRY_TIME)
        place_alpha = self._smoothstep((carry_t - PAN_CARRY_TIME) / PAN_PLACE_TIME)
        if carry_alpha <= 0.0:
            return (
                wp.transform_identity(),
                wp.transform_identity(),
                wp.transform_identity(),
                0.0,
                0.0,
            )

        carry_tf = self._global_carry_transform(carry_alpha)
        if place_alpha >= 0.999:
            carry_tf = self.rigid_carry_end_tf
            place_alpha = 1.0

        place_tf = wp.transform(wp.vec3(0.0, 0.0, -self.pan_place_drop * place_alpha), wp.quat_identity())
        object_tf = self._compose_transform(place_tf, carry_tf)
        return object_tf, carry_tf, carry_tf, carry_alpha, place_alpha

    def _global_carry_transform(self, carry_alpha: float) -> wp.transform:
        yaw = math.radians(PAN_CARRY_TURN_DEGREES) * float(carry_alpha)
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        pivot = self.waic_robot_base_pos
        offset = wp.vec3(
            -PAN_CARRY_BACK_DISTANCE * float(carry_alpha),
            PAN_CARRY_RIGHT_SHIFT * float(carry_alpha),
            0.0,
        )
        pos = pivot + offset - self._quat_rotate_vec3(yaw_rot, pivot)
        return wp.transform(pos, yaw_rot)

    def _build_rigid_robot_body_indices(self) -> wp.array:
        body_indices = [i for i in range(self.model.body_count) if i not in {self.pan_body, self.knife_body}]
        return wp.array(body_indices, dtype=wp.int32, device=self.model.device)

    def _build_rigid_place_body_indices(self) -> wp.array:
        body_indices = [i for i, label in enumerate(self.model.body_label) if "right" in label.lower()]
        return wp.array(body_indices, dtype=wp.int32, device=self.model.device)

    def _build_rigid_carry_shape_indices(self) -> wp.array:
        shape_body = self.model.shape_body.numpy()
        shape_indices = [
            i for i, body_index in enumerate(shape_body) if i < self.robot_shape_end and int(body_index) == -1
        ]
        return wp.array(shape_indices, dtype=wp.int32, device=self.model.device)

    def _relative_transform(self, parent_tf: wp.transform, child_tf: wp.transform) -> tuple[wp.vec3, wp.quat]:
        parent_pos = wp.transform_get_translation(parent_tf)
        parent_rot = wp.transform_get_rotation(parent_tf)
        child_pos = wp.transform_get_translation(child_tf)
        child_rot = wp.transform_get_rotation(child_tf)
        parent_rot_inv = self._quat_inverse(parent_rot)
        local_pos = self._quat_rotate_vec3(parent_rot_inv, child_pos - parent_pos)
        local_rot = self._quat_multiply(parent_rot_inv, child_rot)
        return local_pos, local_rot

    def _handle_transform_from_pan(self, pan_tf: wp.transform) -> wp.transform:
        pan_pos = wp.transform_get_translation(pan_tf)
        pan_rot = wp.transform_get_rotation(pan_tf)
        handle_pos = pan_pos + self._quat_rotate_vec3(pan_rot, self.pan_handle_local_pos)
        handle_rot = self._quat_multiply(pan_rot, self.pan_handle_local_rot)
        return wp.transform(handle_pos, handle_rot)

    def _pan_transform_from_handle(self, handle_tf: wp.transform) -> wp.transform:
        handle_pos = wp.transform_get_translation(handle_tf)
        handle_rot = wp.transform_get_rotation(handle_tf)
        pan_rot = self._quat_multiply(handle_rot, self._quat_inverse(self.pan_handle_local_rot))
        pan_pos = handle_pos - self._quat_rotate_vec3(pan_rot, self.pan_handle_local_pos)
        return wp.transform(pan_pos, pan_rot)

    def _build_post_place_hand_targets(self) -> tuple[wp.transform, wp.transform]:
        placed_handle_tf = self._handle_transform_from_pan(self.pan_place_end_tf)
        placed_handle_pos = wp.transform_get_translation(placed_handle_tf)
        robot_right_world = self._quat_rotate_vec3(
            wp.transform_get_rotation(self.rigid_carry_end_tf),
            wp.vec3(0.0, -1.0, 0.0),
        )
        clear_pos = (
            placed_handle_pos
            + robot_right_world * PAN_HAND_CLEAR_SIDE_OFFSET
            + wp.vec3(
                0.0,
                0.0,
                PAN_HAND_CLEAR_LIFT,
            )
        )
        clear_tf = wp.transform(clear_pos, wp.transform_get_rotation(placed_handle_tf))
        inv_carry = self._transform_inverse(self.rigid_carry_end_tf)
        return self._compose_transform(inv_carry, placed_handle_tf), self._compose_transform(inv_carry, clear_tf)

    def _offset_pan_transform(self, pan_tf: wp.transform, offset: wp.vec3, yaw_degrees: float) -> wp.transform:
        yaw = math.radians(yaw_degrees)
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(
            wp.transform_get_translation(pan_tf) + offset,
            self._quat_multiply(yaw_rot, wp.transform_get_rotation(pan_tf)),
        )

    def _compose_transform(self, tf_a: wp.transform, tf_b: wp.transform) -> wp.transform:
        rot_a = wp.transform_get_rotation(tf_a)
        return wp.transform(
            wp.transform_point(tf_a, wp.transform_get_translation(tf_b)),
            self._quat_multiply(rot_a, wp.transform_get_rotation(tf_b)),
        )

    def _transform_inverse(self, tf: wp.transform) -> wp.transform:
        rot_inv = self._quat_inverse(wp.transform_get_rotation(tf))
        pos_inv = self._quat_rotate_vec3(rot_inv, -wp.transform_get_translation(tf))
        return wp.transform(pos_inv, rot_inv)

    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _build_pan_material_ids(self) -> wp.array | None:
        collider = self.solver._mpm_model.collider
        body_indices = collider.collider_body_index.numpy()
        face_material_ids = collider.face_material_index.numpy()

        material_ids: list[int] = []
        face_offset = 0
        for collider_id, body_index in enumerate(body_indices):
            mesh = self.solver._mpm_model._collider_meshes[collider_id]
            face_count = int(mesh.indices.shape[0] // 3)
            if int(body_index) == int(self.pan_body):
                material_ids.extend(np.unique(face_material_ids[face_offset : face_offset + face_count]).astype(int))
            face_offset += face_count

        if not material_ids:
            print("[newton] Warning: pan collider material ids not found; pan friction will not be raised.")
            return None
        return wp.array(sorted(set(material_ids)), dtype=wp.int32, device=self.model.device)

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
    def _quat_inverse(q: wp.quat) -> wp.quat:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        length_sq = x * x + y * y + z * z + w * w
        if length_sq == 0.0:
            return wp.quat_identity()
        inv = 1.0 / length_sq
        return wp.quat(-x * inv, -y * inv, -z * inv, w * inv)

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
            prim_path = str(prim.GetPath())
            usd_mesh = UsdGeom.Mesh(prim)
            vertices, indices = self._usd_mesh_to_newton_arrays(usd_mesh, xform_cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue
            body = -1
            if self._is_pan_visual(prim_path):
                body = self.pan_body
                vertices = self._world_vertices_to_pan_body(vertices)
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=body,
                mesh=mesh,
                xform=wp.transform_identity(),
                cfg=visual_cfg,
                color=self._semantic_visual_color(prim_path),
                label=f"waic_kitchen_visual_{mesh_count:03d}",
            )
            mesh_count += 1
            tri_count += int(len(indices) // 3)
        print(f"Loaded WAIC full-house visual: {mesh_count} meshes, {tri_count} triangles from {self.waic_visual_usd}")
        self._add_pan_visual_model(builder, visual_cfg)

    def _add_pan_visual_model(self, builder: newton.ModelBuilder, visual_cfg: newton.ModelBuilder.ShapeConfig) -> None:
        if not self.pan_visual_usd:
            return
        if not os.path.exists(self.pan_visual_usd):
            print(f"WAIC pan visual USD not found, skipping: {self.pan_visual_usd}")
            return

        stage = Usd.Stage.Open(self.pan_visual_usd)
        if stage is None:
            print(f"WAIC pan visual USD failed to open, skipping: {self.pan_visual_usd}")
            return

        xform_cache = UsdGeom.XformCache()
        mesh_count = 0
        tri_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            prim_path = str(prim.GetPath())
            vertices, indices = self._usd_mesh_to_newton_arrays(UsdGeom.Mesh(prim), xform_cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue

            vertices = self._pan_visual_vertices_to_newton_world(vertices)
            vertices = self._world_vertices_to_pan_body(vertices)
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=self.pan_body,
                mesh=mesh,
                xform=wp.transform_identity(),
                cfg=visual_cfg,
                color=self._semantic_visual_color(prim_path),
                label=f"waic_v6_pan_visual_{mesh_count:03d}",
            )
            mesh_count += 1
            tri_count += int(len(indices) // 3)

        print(
            f"Loaded WAIC pan visual: {mesh_count} meshes, {tri_count} triangles from {self.pan_visual_usd}, "
            f"axis_mode={self.pan_visual_axis_mode}, offset="
            f"({float(self.pan_visual_offset[0]):.4f}, {float(self.pan_visual_offset[1]):.4f}, "
            f"{float(self.pan_visual_offset[2]):.4f})"
        )

    def _pan_visual_vertices_to_newton_world(self, vertices: np.ndarray) -> np.ndarray:
        if self.pan_visual_axis_mode == "blender_usd":
            converted = np.empty_like(vertices)
            converted[:, 0] = vertices[:, 0]
            converted[:, 1] = vertices[:, 2]
            converted[:, 2] = -vertices[:, 1]
        elif self.pan_visual_axis_mode == "identity":
            converted = vertices.copy()
        else:
            raise ValueError(f"Unknown pan visual axis mode: {self.pan_visual_axis_mode}")

        converted[:, 0] += float(self.pan_visual_offset[0])
        converted[:, 1] += float(self.pan_visual_offset[1])
        converted[:, 2] += float(self.pan_visual_offset[2])
        return converted

    def _world_vertices_to_pan_body(self, vertices: np.ndarray) -> np.ndarray:
        local_vertices = np.empty_like(vertices)
        inv_pan_rot = wp.quat(
            -float(self.scene_rotation[0]),
            -float(self.scene_rotation[1]),
            -float(self.scene_rotation[2]),
            float(self.scene_rotation[3]),
        )
        for i, vertex in enumerate(vertices):
            world_offset = wp.vec3(
                float(vertex[0]) - float(self.pan_center[0]),
                float(vertex[1]) - float(self.pan_center[1]),
                float(vertex[2]) - float(self.pan_center[2]),
            )
            local = self._quat_rotate_vec3(inv_pan_rot, world_offset)
            local_vertices[i] = (float(local[0]), float(local[1]), float(local[2]))
        return local_vertices

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
        for raw_face_count in face_counts:
            face_vertex_count = int(raw_face_count)
            if face_vertex_count >= 3:
                face = face_indices[cursor : cursor + face_vertex_count]
                for i in range(1, face_vertex_count - 1):
                    triangles.extend((int(face[0]), int(face[i]), int(face[i + 1])))
            cursor += face_vertex_count

        return vertices, np.asarray(triangles, dtype=np.int32)

    @staticmethod
    def _semantic_visual_color(prim_path: str) -> tuple[float, float, float]:
        text = prim_path.lower()
        if Example._is_pan_visual(prim_path):
            if "rivet" in text or "rim" in text or "ring" in text or "hole" in text:
                return (0.58, 0.57, 0.54)
            return (0.025, 0.026, 0.028)
        if "table" in text:
            return (0.52, 0.37, 0.23)
        if "kitchen" in text or "furniture" in text:
            return (0.70, 0.68, 0.62)
        return (0.62, 0.60, 0.56)

    @staticmethod
    def _is_pan_visual(prim_path: str) -> bool:
        text = prim_path.lower()
        return "frying_pan" in text or "pan_" in text

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

    def _build_torso_reach_targets(self) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()

        q_indices = []
        dof_indices = []
        home_values = []
        reach_values = []
        for joint_name, target in self.TORSO_REACH_JOINT_TARGETS.items():
            joint_idx = self._joint_index(joint_name)
            q_idx = int(q_start[joint_idx])
            dof_idx = int(qd_start[joint_idx])
            q_indices.append(q_idx)
            dof_indices.append(dof_idx)
            home_values.append(float(q_home[q_idx]))
            reach_values.append(float(np.clip(target, lower[dof_idx], upper[dof_idx])))

        self.torso_dof_indices_np = np.asarray(dof_indices, dtype=np.int32)
        self.torso_q_home_np = np.asarray(home_values, dtype=np.float32)
        self.torso_q_reach_np = np.asarray(reach_values, dtype=np.float32)

        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(dof_indices, dtype=wp.int32, device=self.model.device),
            wp.array(home_values, dtype=wp.float32, device=self.model.device),
            wp.array(reach_values, dtype=wp.float32, device=self.model.device),
        )

    def _controlled_joint_labels(self) -> set[str]:
        return {
            f"DexforceW1V021/{name}"
            for name in (*self.BODY_JOINTS, *self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS, *self.RIGHT_HAND_JOINTS)
        }

    def _build_right_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        """Build open/grasp joint values for all five right-hand fingers.

        Grasp values are set below each joint's upper limit so the fingers curl
        around the pan handle without bottoming out.
        """
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        # Grasp target as a fraction of the joint's upper limit (lower is 0).
        grasp_fraction = {
            "RIGHT_HAND_THUMB2": 0.66,
            "RIGHT_HAND_THUMB1": 0.62,
            "RIGHT_HAND_INDEX": 0.68,
            "RIGHT_INDEX_PIP": 0.58,
            "RIGHT_HAND_MIDDLE": 0.68,
            "RIGHT_MIDDLE_PIP": 0.58,
            "RIGHT_HAND_RING": 0.66,
            "RIGHT_RING_PIP": 0.56,
            "RIGHT_HAND_PINKY": 0.62,
            "RIGHT_PINKY_PIP": 0.52,
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

    def step(self) -> None:
        path_end_time = self.rigid_carry_start_time + self.path_duration
        if not self.post_mode and self.sim_time < path_end_time:
            self.simulate()
            return

        if not self.post_mode:
            self._enter_post_mode()

        self._step_post_place_script()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_post_pose()

    def _global_carry_transform(self, carry_alpha: float) -> wp.transform:
        # Used by __init__ before the dynamic root path exists. The actual
        # carry path is sampled in _rigid_carry_transform once simulation runs.
        yaw = math.radians(4.0 * RIGHT_TURN_DEGREES) * float(carry_alpha)
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), yaw_rot)

    def _sample_post_place_script(self, t: float):
        return None

    def _rigid_carry_transform(
        self, query_time: float
    ) -> tuple[wp.transform, wp.transform, wp.transform, float, float]:
        path_t = max(float(query_time) - self.rigid_carry_start_time, 0.0)
        if path_t <= 0.0:
            return (
                wp.transform_identity(),
                wp.transform_identity(),
                wp.transform_identity(),
                0.0,
                0.0,
            )

        if self.path_start_root_tf is None:
            self.path_start_root_tf = self._current_robot_root_transform()

        path_alpha = float(np.clip(path_t / max(self.path_duration, 1.0e-6), 0.0, 1.0))
        target_root_tf = self._sample_root_path(path_t)
        carry_tf = self._compose_transform(target_root_tf, self._transform_inverse(self.path_start_root_tf))
        return carry_tf, carry_tf, carry_tf, path_alpha, 0.0

    def _sample_root_path(self, path_t: float) -> wp.transform:
        start_tf = self.path_start_root_tf or self._current_robot_root_transform()
        start_pos = wp.transform_get_translation(start_tf)
        start_rot = wp.transform_get_rotation(start_tf)

        waypoints = (start_pos, *ROOT_TARGETS)
        segment = 0
        local_t = float(path_t)
        yaw_turns = 0

        while segment < 7:
            duration = self.turn_time if segment % 2 == 0 else self.move_time
            if local_t <= duration:
                break
            local_t -= duration
            if segment % 2 == 0:
                yaw_turns += 1
            segment += 1

        segment = min(segment, 6)
        duration = self.turn_time if segment % 2 == 0 else self.move_time
        u = self._smoothstep(local_t / max(duration, 1.0e-6))

        if segment % 2 == 0:
            yaw = math.radians(RIGHT_TURN_DEGREES) * (yaw_turns + u)
            pos_index = min(yaw_turns, len(waypoints) - 1)
            pos = waypoints[pos_index]
        else:
            yaw = math.radians(RIGHT_TURN_DEGREES) * yaw_turns
            move_index = min(yaw_turns, len(waypoints) - 1)
            pos = self._lerp_vec3(waypoints[move_index - 1], waypoints[move_index], u)

        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(pos, self._quat_multiply(yaw_rot, start_rot))

    def _enter_post_mode(self) -> None:
        self.post_mode = True
        self.post_start_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        path_start_tf = self.path_start_root_tf or root_tf
        self.gizmo_carry_tf = self._compose_transform(root_tf, self._transform_inverse(path_start_tf))

        self.post_start_right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.post_start_pan_tf = self._current_pan_transform()
        self.post_right_local_pos = self.pan_handle_local_pos
        self.post_right_local_rot = self.pan_handle_local_rot
        self.post_restore_right_tf = self._right_tf_from_pan(self.post_start_pan_tf)
        start_pan_pos = wp.transform_get_translation(self.post_start_pan_tf)
        approach_z = max(float(start_pan_pos[2]), float(V6_PAN_PLACE_CENTER[2]) + PAN_APPROACH_CLEARANCE_Z)
        self.post_approach_tf = wp.transform(
            wp.vec3(float(V6_PAN_PLACE_CENTER[0]), float(V6_PAN_PLACE_CENTER[1]), approach_z),
            self.scene_rotation,
        )
        self.post_place_tf = wp.transform(V6_PAN_PLACE_CENTER, self.scene_rotation)
        self.post_place_right_tf = self._right_tf_from_pan(self.post_place_tf)
        self.post_clear_tf = wp.transform(
            wp.transform_get_translation(self.post_place_right_tf) + wp.vec3(0.0, -0.34, 0.10),
            wp.transform_get_rotation(self.post_place_right_tf),
        )
        self.post_prev_pan_tf = self.post_start_pan_tf
        self.left_hold_tf_for_gizmo = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.post_current_hand_alpha = 1.0

        print(
            "[newton] V6 post-place script started: move pan forward to lowered table, "
            "level it down, clear right hand."
        )

    def _step_post_place_script(self) -> None:
        elapsed = max(self.sim_time + self.frame_dt - self.post_start_time, 0.0)
        self.post_current_elapsed = elapsed
        right_tf, pan_tf, hand_alpha = self._sample_v6_post_place_script(elapsed)
        self.post_current_hand_alpha = hand_alpha
        self._solve_scripted_right_ik(right_tf, hand_alpha)
        restore_t = PAN_CARRY_GRIP_RESTORE_TIME
        release_end = restore_t + self.post_forward_time + self.post_lower_time + self.post_release_time
        clear_end = release_end + self.post_clear_time
        if restore_t < elapsed <= release_end:
            pan_tf = self._pan_tf_from_current_right_hand()
        elif release_end < elapsed <= clear_end:
            pan_tf = self.post_prev_pan_tf
        self._set_pan_and_meat_pose(pan_tf)

    def _sample_v6_post_place_script(self, elapsed: float) -> tuple[wp.transform, wp.transform, float]:
        t_restore = PAN_CARRY_GRIP_RESTORE_TIME
        t0 = t_restore + self.post_forward_time
        t1 = t0 + self.post_lower_time
        t2 = t1 + self.post_release_time
        t3 = t2 + self.post_clear_time

        if elapsed <= t_restore:
            u = self._smoothstep(elapsed / max(t_restore, 1.0e-6))
            return (
                self._interpolate_transform(self.post_start_right_tf, self.post_restore_right_tf, u),
                self.post_start_pan_tf,
                1.0,
            )

        if elapsed <= t0:
            u = self._smoothstep((elapsed - t_restore) / max(self.post_forward_time, 1.0e-6))
            pan_tf = self._interpolate_transform(self.post_start_pan_tf, self.post_approach_tf, u)
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0
        if elapsed <= t1:
            u = self._smoothstep((elapsed - t0) / max(self.post_lower_time, 1.0e-6))
            pan_tf = self._interpolate_transform(self.post_approach_tf, self.post_place_tf, u)
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0
        if elapsed <= t2:
            u = self._smoothstep((elapsed - t1) / max(self.post_release_time, 1.0e-6))
            pan_tf = self.post_place_tf
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0 - u
        if elapsed <= t3:
            u = self._smoothstep((elapsed - t2) / max(self.post_clear_time, 1.0e-6))
            return (
                self._interpolate_transform(self.post_place_right_tf, self.post_clear_tf, u),
                self.post_place_tf,
                0.0,
            )

        return self.post_clear_tf, self.post_place_tf, 0.0

    def _right_tf_from_pan(self, pan_tf: wp.transform) -> wp.transform:
        return self._compose_transform(
            pan_tf,
            wp.transform(self.post_right_local_pos, self.post_right_local_rot),
        )

    def _pan_tf_from_current_right_hand(self) -> wp.transform:
        actual_right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        return self._compose_transform(
            actual_right_tf,
            self._transform_inverse(wp.transform(self.post_right_local_pos, self.post_right_local_rot)),
        )

    def _solve_scripted_right_ik(self, world_right_tf: wp.transform, hand_alpha: float) -> None:
        inv_carry_tf = self._transform_inverse(self.gizmo_carry_tf)
        self.right_tf = self._compose_transform(inv_carry_tf, world_right_tf)
        self.left_tf = self._compose_transform(inv_carry_tf, self.left_hold_tf_for_gizmo)
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        wp.launch(
            copy_joint_q_to_ik_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.state_0.joint_q],
            outputs=[self.ik_joint_q],
            device=self.model.device,
        )
        torso_alpha = self._post_place_torso_reach_alpha(self.post_current_elapsed)
        self._seed_torso_reach_posture(torso_alpha)
        self._set_torso_ik_constraints(torso_alpha)

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
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[
                self.right_hand_q_indices,
                self.right_hand_open,
                self.right_hand_grasp,
                float(hand_alpha),
            ],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

        wp.copy(self.state_0.joint_q, self.frame_joint_q_end)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._reapply_carried_robot_pose()

    def _reapply_carried_robot_pose(self) -> None:
        wp.launch(
            apply_body_transform_kernel,
            dim=self.rigid_robot_body_indices.shape[0],
            inputs=[self.state_0.body_q, self.rigid_robot_body_indices, self.gizmo_carry_tf],
            device=self.model.device,
        )
        if self.rigid_carry_shape_indices.shape[0] > 0:
            wp.launch(
                set_transformed_shape_transforms_kernel,
                dim=self.rigid_carry_shape_indices.shape[0],
                inputs=[
                    self.rigid_carry_shape_transform_start,
                    self.rigid_carry_shape_indices,
                    self.gizmo_carry_tf,
                ],
                outputs=[self.model.shape_transform],
                device=self.model.device,
            )

    def _set_pan_and_meat_pose(self, pan_tf: wp.transform) -> None:
        delta_tf = self._compose_transform(pan_tf, self._transform_inverse(self.post_prev_pan_tf))
        wp.launch(
            set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.pan_body, pan_tf],
            device=self.model.device,
        )
        wp.launch(
            set_body_velocity_kernel,
            dim=1,
            inputs=[self.state_0.body_qd, self.pan_body, wp.vec3(0.0, 0.0, 0.0)],
            device=self.model.device,
        )
        wp.launch(
            apply_particle_delta_transform_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                delta_tf,
                1.0 / self.frame_dt,
            ],
            device=self.model.device,
        )
        self.post_prev_pan_tf = pan_tf

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/meat",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()
        wp.synchronize()

    def _report_post_pose(self) -> None:
        if self.pose_print_interval > 0.0 and self.sim_time - self.last_pose_print_time < self.pose_print_interval:
            return
        self.last_pose_print_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        root_pos = self._vec3_to_np(wp.transform_get_translation(root_tf))
        pan_pos = self._vec3_to_np(wp.transform_get_translation(self._current_pan_transform()))
        right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        right_pos = self._vec3_to_np(wp.transform_get_translation(right_tf))
        right_rot = self._quat_to_np(wp.transform_get_rotation(right_tf))
        print(
            "[waic kitchen V6] "
            f"root_pos=({root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}) "
            f"pan_pos=({pan_pos[0]:.6f}, {pan_pos[1]:.6f}, {pan_pos[2]:.6f}) "
            f"right_actual_pos=({right_pos[0]:.6f}, {right_pos[1]:.6f}, {right_pos[2]:.6f}) "
            f"right_actual_quat_xyzw=({right_rot[0]:.6f}, {right_rot[1]:.6f}, {right_rot[2]:.6f}, {right_rot[3]:.6f}) "
            f"hand_alpha={self.post_current_hand_alpha:.3f}"
        )

    def _current_robot_root_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[int(self.rigid_robot_body_indices.numpy()[0])])

    def _current_pan_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[self.pan_body])

    @staticmethod
    def _lerp_vec3(a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
        return wp.vec3(
            float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
            float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
            float(a[2]) * (1.0 - alpha) + float(b[2]) * alpha,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100000, paused=False)

        parser.add_argument("--voxel-size", "-dx", type=float, default=0.016)
        parser.add_argument("--particles-per-cell", "-ppc", type=int, default=3)
        parser.add_argument("--density", type=float, default=1000.0)

        # Meat block dimensions [m]: length (x), width (y, cut direction), height (z).
        parser.add_argument("--meat-length", type=float, default=MEAT_LENGTH, help="Meat block length [m] (x).")
        parser.add_argument(
            "--meat-width", type=float, default=MEAT_WIDTH, help="Meat block width [m] (y, cut direction)."
        )
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
            default=HOUSE_VISUAL_USD,
            help="Complete WAIC house USD visual background. Visual only; no collision.",
        )
        parser.add_argument(
            "--pan-visual-usd",
            type=str,
            default=PAN_VISUAL_USD,
            help="Frying-pan USD visual mesh attached to the simplified pan body.",
        )
        parser.add_argument(
            "--pan-visual-axis-mode",
            choices=("blender_usd", "identity"),
            default="blender_usd",
            help="Axis conversion for the pan USD. blender_usd maps USD (x,y,z) to Newton/Blender (x,z,-y).",
        )
        parser.add_argument("--pan-visual-offset-x", type=float, default=0.0)
        parser.add_argument("--pan-visual-offset-y", type=float, default=0.0)
        parser.add_argument("--pan-visual-offset-z", type=float, default=0.0)
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
        parser.add_argument(
            "--path-turn-time",
            type=float,
            default=2.0,
            help="Seconds for each 90-degree right turn in the scripted carry path.",
        )
        parser.add_argument(
            "--path-move-time",
            type=float,
            default=3.0,
            help="Seconds for each forward move between root target points.",
        )
        parser.add_argument(
            "--pose-print-interval",
            type=float,
            default=0.25,
            help="Seconds between printed V6 post-place pose diagnostics. Use 0.0 to print every frame.",
        )
        parser.add_argument("--post-forward-time", type=float, default=3.0)
        parser.add_argument("--post-lower-time", type=float, default=2.0)
        parser.add_argument("--post-release-time", type=float, default=0.7)
        parser.add_argument("--post-clear-time", type=float, default=1.5)
        parser.add_argument(
            "--v6-meat-pan-gap",
            type=float,
            default=V6_MEAT_PAN_GAP,
            help="Initial vertical gap between the pan collision top and the MPM meat particles in V6.",
        )
        parser.add_argument(
            "--v6-knife-pan-clearance",
            type=float,
            default=V6_KNIFE_PAN_CLEARANCE,
            help="Vertical clearance between the knife bottom and pan collision top at the deepest cut.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
