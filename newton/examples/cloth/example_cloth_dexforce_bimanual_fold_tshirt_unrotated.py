# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual Fold T-Shirt Unrotated
#
# Replays captured bimanual wrist TCP targets on the unrotated T-shirt scene.
# The source IK gizmo scene remains available as
# ``cloth_dexforce_bimanual_ik_tshirt_unrotated`` for trajectory collection.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_fold_tshirt_unrotated
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.cloth.example_cloth_dexforce_bimanual_grasp_cloth import (
    copy_ik_to_joint_q_kernel,
    lock_joint_q_kernel,
    set_indexed_joint_q_kernel,
)
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_cloth import (
    Example as DexforceBimanualIKClothExample,
)
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_tshirt_unrotated import (
    SHIRT_ASSET,
    SHIRT_COLOR,
    SHIRT_COLLISION_RADIUS,
    SHIRT_DENSITY,
    SHIRT_EDGE_KD,
    SHIRT_EDGE_KE,
    SHIRT_POS,
    SHIRT_PRIM_PATH,
    SHIRT_ROT,
    SHIRT_SCALE,
    SHIRT_SELF_CONTACT_MARGIN,
    SHIRT_SELF_CONTACT_RADIUS,
    SHIRT_SOFT_CONTACT_MARGIN,
    Example as DexforceBimanualIKTShirtUnrotatedExample,
)


LEFT_OPEN_TF = wp.transform(
    wp.vec3(0.3287, 0.4218, 1.3640),
    wp.quat(0.1959, 0.6952, -0.6604, -0.2057),
)
RIGHT_OPEN_TF = wp.transform(
    wp.vec3(0.4089, -0.5070, 1.3645),
    wp.quat(-0.1256, 0.6959, 0.6956, -0.1270),
)
LEFT_GRASP_TF = wp.transform(
    wp.vec3(0.2508, 0.1233, 1.2460),
    wp.quat(-0.3621, -0.6058, 0.6085, 0.3629),
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(0.3103, -0.2936, 1.2498),
    wp.quat(-0.3511, 0.6407, 0.5911, -0.3418),
)
LEFT_PLACE_TF = wp.transform(
    wp.vec3(0.4628, 0.1421, 1.3087),
    wp.quat(-0.3619, -0.6062, 0.6086, 0.3622),
)
RIGHT_PLACE_TF = wp.transform(
    wp.vec3(0.4448, -0.2853, 1.3050),
    wp.quat(-0.3454, 0.6472, 0.5930, -0.3319),
)
LEFT_RELEASE_TF = wp.transform(
    wp.vec3(0.6834, 0.1128, 1.2695),
    wp.quat(0.1210, 0.7092, -0.6798, -0.1425),
)
RIGHT_RELEASE_TF = wp.transform(
    wp.vec3(0.6681, -0.3053, 1.2971),
    wp.quat(-0.0700, 0.6805, 0.7255, -0.0753),
)
LEFT_SECOND_FOLD_TF = wp.transform(
    wp.vec3(0.5083, 0.5281, 1.3588),
    wp.quat(0.0837, 0.7052, -0.6993, -0.0814),
)
RIGHT_SECOND_GRASP_TF = wp.transform(
    wp.vec3(0.5284, -0.3398, 1.2308),
    wp.quat(0.0346, 0.7038, 0.7087, 0.0347),
)
RIGHT_SECOND_LIFT_TF = wp.transform(
    wp.vec3(0.5292, -0.2702, 1.4038),
    wp.quat(0.0352, 0.7038, 0.7087, 0.0344),
)
RIGHT_SECOND_PLACE_TF = wp.transform(
    wp.vec3(0.5289, 0.3200, 1.3086),
    wp.quat(0.0352, 0.7038, 0.7087, 0.0345),
)

HOME_HOLD_TIME = 0.8
APPROACH_TIME = 2.0
GRASP_TIME = 1.4
GRASP_CLOSE_TIME = 1.6
GRASP_LIFT_TIME = 1.4
FOLD_TRAVEL_TIME = 2.4
FOLD_PLACE_TIME = 1.4
RELEASE_TIME = 1.0
LIFT_TIME = 1.2
HOLD_TIME = 1.0
SECOND_APPROACH_TIME = 1.6
SECOND_DESCEND_TIME = 1.0
SECOND_GRASP_CLOSE_TIME = 1.4
SECOND_LIFT_TIME = 1.2
SECOND_FOLD_TRAVEL_TIME = 2.4
SECOND_PLACE_TIME = 1.0
SECOND_RELEASE_TIME = 1.0
SECOND_RELEASE_LIFT_TIME = 1.2
SECOND_FINAL_HOLD_TIME = 1.0

GRASP_LIFT_HEIGHT = 0.15
GRASP_HEIGHT_OFFSET = 0.0
FOLD_TRAVEL_HEIGHT = float(wp.transform_get_translation(LEFT_PLACE_TF)[2]) + 0.15
FOLD_PLACE_HEIGHT_OFFSET = 0.0
LIFT_HEIGHT = 0.20
SECOND_APPROACH_HEIGHT = 0.08
SECOND_GRASP_HEIGHT_OFFSET = 0.016
SECOND_FOLD_TRAVEL_HEIGHT = float(wp.transform_get_translation(RIGHT_SECOND_LIFT_TF)[2])
SECOND_GRASP_ALPHA = 1.18
RELEASE_CRACK_ALPHA = 0.75
SECOND_RELEASE_CRACK_ALPHA = 0.35
SECOND_RELEASE_LIFT_HEIGHT = 0.20
FIRST_RELEASE_START_TIME = (
    HOME_HOLD_TIME
    + APPROACH_TIME
    + GRASP_TIME
    + GRASP_CLOSE_TIME
    + GRASP_LIFT_TIME
    + FOLD_TRAVEL_TIME
    + FOLD_PLACE_TIME
)
FIRST_RELEASE_END_TIME = FIRST_RELEASE_START_TIME + RELEASE_TIME + LIFT_TIME
FIRST_FOLD_END_TIME = FIRST_RELEASE_END_TIME + HOLD_TIME
SECOND_RELEASE_START_TIME = (
    FIRST_FOLD_END_TIME
    + SECOND_APPROACH_TIME
    + SECOND_DESCEND_TIME
    + SECOND_GRASP_CLOSE_TIME
    + SECOND_LIFT_TIME
    + SECOND_FOLD_TRAVEL_TIME
    + SECOND_PLACE_TIME
)
SECOND_RELEASE_END_TIME = SECOND_RELEASE_START_TIME + SECOND_RELEASE_TIME + SECOND_RELEASE_LIFT_TIME
SCRIPT_END_TIME = (
    SECOND_RELEASE_END_TIME
    + SECOND_FINAL_HOLD_TIME
)

GRASP_CLOTH_SOFT_CONTACT_MU = 1.20
GRASP_ROBOT_CLOTH_CONTACT_MU = 1.20
GRASP_TABLE_CLOTH_CONTACT_MU = 0.12
SECOND_GRASP_TABLE_CLOTH_CONTACT_MU = 0.04
RELEASE_CLOTH_SOFT_CONTACT_MU = 0.50
RELEASE_ZERO_CLOTH_SOFT_CONTACT_MU = 0.0
RELEASE_ROBOT_CLOTH_CONTACT_MU = 0.25
RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU = 0.0
RELEASE_TABLE_CLOTH_CONTACT_MU = 0.5

TSHIRT_SIM_SUBSTEPS = 24
TSHIRT_SOLVER_ITERATIONS = 40
TSHIRT_SOFT_CONTACT_KE = 3.0e5
TSHIRT_SOFT_CONTACT_KD = 5.0e-2
TSHIRT_ROBOT_CONTACT_KE = 3.0e5
TSHIRT_TRI_KE = 2.0e4
TSHIRT_TRI_KA = 2.0e4
TSHIRT_TRI_KD = 5.0e-5


class Example(DexforceBimanualIKTShirtUnrotatedExample):
    def __init__(self, viewer, args):
        self.trajectory_time_scale = float(args.trajectory_time_scale)
        self.grasp_height_offset = float(args.grasp_height_offset)
        self.grasp_lift_height = float(args.grasp_lift_height)
        self.fold_travel_height = float(args.fold_travel_height)
        self.fold_place_height_offset = float(args.fold_place_height_offset)
        super().__init__(viewer, args)
        self.left_home_tf = self.left_tf
        self.right_home_tf = self.right_tf
        self.motion_segments = self._build_motion_segments()
        self._robot_particle_contacts_enabled = True

        self.sim_substeps = TSHIRT_SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver.iterations = TSHIRT_SOLVER_ITERATIONS
        self.model.soft_contact_ke = TSHIRT_SOFT_CONTACT_KE
        self.model.soft_contact_kd = TSHIRT_SOFT_CONTACT_KD
        self._set_robot_contact_stiffness(TSHIRT_ROBOT_CONTACT_KE)
        self._set_tshirt_contact_materials(
            RELEASE_CLOTH_SOFT_CONTACT_MU,
            RELEASE_ROBOT_CLOTH_CONTACT_MU,
            RELEASE_TABLE_CLOTH_CONTACT_MU,
        )

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        self.particle_radius = SHIRT_COLLISION_RADIUS
        self.soft_contact_margin = SHIRT_SOFT_CONTACT_MARGIN
        self.particle_self_contact_radius = SHIRT_SELF_CONTACT_RADIUS
        self.particle_self_contact_margin = SHIRT_SELF_CONTACT_MARGIN

        usd_stage = Usd.Stage.Open(newton.examples.get_asset(SHIRT_ASSET))
        usd_prim = usd_stage.GetPrimAtPath(SHIRT_PRIM_PATH)
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = self._center_shirt_vertices(shirt_mesh.vertices)

        builder.add_cloth_mesh(
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
            indices=shirt_mesh.indices,
            rot=SHIRT_ROT,
            pos=SHIRT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=SHIRT_DENSITY,
            scale=1.0,
            tri_ke=TSHIRT_TRI_KE,
            tri_ka=TSHIRT_TRI_KA,
            tri_kd=TSHIRT_TRI_KD,
            edge_ke=SHIRT_EDGE_KE,
            edge_kd=SHIRT_EDGE_KD,
            particle_radius=self.particle_radius,
        )

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        script_time = (self.sim_time + self.frame_dt) * self.trajectory_time_scale
        self.left_tf, self.right_tf, grasp_alpha = self._sample_script(script_time)

        second_release = self._is_second_release_no_friction_time(script_time)
        self._set_robot_particle_contacts_enabled(not second_release)

        if second_release:
            self._set_tshirt_contact_materials(
                RELEASE_ZERO_CLOTH_SOFT_CONTACT_MU,
                RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU,
                RELEASE_TABLE_CLOTH_CONTACT_MU,
            )
        elif self._is_first_release_no_friction_time(script_time):
            self._set_tshirt_contact_materials(
                RELEASE_CLOTH_SOFT_CONTACT_MU,
                RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU,
                RELEASE_TABLE_CLOTH_CONTACT_MU,
            )
        elif grasp_alpha > 0.15:
            table_mu = (
                SECOND_GRASP_TABLE_CLOTH_CONTACT_MU
                if script_time > FIRST_FOLD_END_TIME
                else GRASP_TABLE_CLOTH_CONTACT_MU
            )
            self._set_tshirt_contact_materials(
                GRASP_CLOTH_SOFT_CONTACT_MU,
                GRASP_ROBOT_CLOTH_CONTACT_MU,
                table_mu,
            )
        else:
            self._set_tshirt_contact_materials(
                RELEASE_CLOTH_SOFT_CONTACT_MU,
                RELEASE_ROBOT_CLOTH_CONTACT_MU,
                RELEASE_TABLE_CLOTH_CONTACT_MU,
            )

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
            inputs=[self.hand_q_indices, self.hand_open, self.hand_closed, grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def _build_motion_segments(self) -> tuple[tuple[float, wp.transform, wp.transform, wp.transform, wp.transform, float, float], ...]:
        left_approach_tf = LEFT_OPEN_TF
        right_approach_tf = RIGHT_OPEN_TF
        left_grasp_tf = self._offset_transform(LEFT_GRASP_TF, wp.vec3(0.0, 0.0, self.grasp_height_offset))
        right_grasp_tf = self._offset_transform(RIGHT_GRASP_TF, wp.vec3(0.0, 0.0, self.grasp_height_offset))
        left_grasp_lift_tf = self._offset_transform(left_grasp_tf, wp.vec3(0.0, 0.0, self.grasp_lift_height))
        right_grasp_lift_tf = self._offset_transform(right_grasp_tf, wp.vec3(0.0, 0.0, self.grasp_lift_height))
        left_place_lift_tf = self._with_height(LEFT_PLACE_TF, self.fold_travel_height)
        right_place_lift_tf = self._with_height(RIGHT_PLACE_TF, self.fold_travel_height)
        left_place_tf = self._offset_transform(LEFT_RELEASE_TF, wp.vec3(0.0, 0.0, self.fold_place_height_offset))
        right_place_tf = self._offset_transform(RIGHT_RELEASE_TF, wp.vec3(0.0, 0.0, self.fold_place_height_offset))
        left_release_tf = self._offset_transform(left_place_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        right_release_tf = self._offset_transform(right_place_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        right_second_grasp_tf = self._offset_transform(
            RIGHT_SECOND_GRASP_TF, wp.vec3(0.0, 0.0, SECOND_GRASP_HEIGHT_OFFSET)
        )
        right_second_approach_tf = self._offset_transform(
            right_second_grasp_tf, wp.vec3(0.0, 0.0, SECOND_APPROACH_HEIGHT)
        )
        right_second_place_lift_tf = self._with_height(RIGHT_SECOND_PLACE_TF, SECOND_FOLD_TRAVEL_HEIGHT)
        right_second_release_tf = self._offset_transform(
            RIGHT_SECOND_PLACE_TF, wp.vec3(0.0, 0.0, SECOND_RELEASE_LIFT_HEIGHT)
        )

        return (
            (HOME_HOLD_TIME, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (
                APPROACH_TIME,
                self.left_home_tf,
                left_approach_tf,
                self.right_home_tf,
                right_approach_tf,
                0.0,
                0.0,
            ),
            (
                GRASP_TIME,
                left_approach_tf,
                left_grasp_tf,
                right_approach_tf,
                right_grasp_tf,
                0.0,
                0.0,
            ),
            (GRASP_CLOSE_TIME, left_grasp_tf, left_grasp_tf, right_grasp_tf, right_grasp_tf, 0.0, 1.0),
            (
                GRASP_LIFT_TIME,
                left_grasp_tf,
                left_grasp_lift_tf,
                right_grasp_tf,
                right_grasp_lift_tf,
                1.0,
                1.0,
            ),
            (
                FOLD_TRAVEL_TIME,
                left_grasp_lift_tf,
                left_place_lift_tf,
                right_grasp_lift_tf,
                right_place_lift_tf,
                1.0,
                1.0,
            ),
            (
                FOLD_PLACE_TIME,
                left_place_lift_tf,
                left_place_tf,
                right_place_lift_tf,
                right_place_tf,
                1.0,
                1.0,
            ),
            (RELEASE_TIME, left_place_tf, left_place_tf, right_place_tf, right_place_tf, 1.0, RELEASE_CRACK_ALPHA),
            (
                LIFT_TIME,
                left_place_tf,
                left_release_tf,
                right_place_tf,
                right_release_tf,
                RELEASE_CRACK_ALPHA,
                0.0,
            ),
            (HOLD_TIME, left_release_tf, left_release_tf, right_release_tf, right_release_tf, 0.0, 0.0),
            (
                SECOND_APPROACH_TIME,
                left_release_tf,
                LEFT_SECOND_FOLD_TF,
                right_release_tf,
                right_second_approach_tf,
                0.0,
                0.0,
            ),
            (
                SECOND_DESCEND_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                right_second_approach_tf,
                right_second_grasp_tf,
                0.0,
                0.0,
            ),
            (
                SECOND_GRASP_CLOSE_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                right_second_grasp_tf,
                right_second_grasp_tf,
                0.0,
                SECOND_GRASP_ALPHA,
            ),
            (
                SECOND_LIFT_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                right_second_grasp_tf,
                RIGHT_SECOND_LIFT_TF,
                SECOND_GRASP_ALPHA,
                SECOND_GRASP_ALPHA,
            ),
            (
                SECOND_FOLD_TRAVEL_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                RIGHT_SECOND_LIFT_TF,
                right_second_place_lift_tf,
                SECOND_GRASP_ALPHA,
                SECOND_GRASP_ALPHA,
            ),
            (
                SECOND_PLACE_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                right_second_place_lift_tf,
                RIGHT_SECOND_PLACE_TF,
                SECOND_GRASP_ALPHA,
                SECOND_GRASP_ALPHA,
            ),
            (
                SECOND_RELEASE_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                RIGHT_SECOND_PLACE_TF,
                RIGHT_SECOND_PLACE_TF,
                SECOND_GRASP_ALPHA,
                SECOND_RELEASE_CRACK_ALPHA,
            ),
            (
                SECOND_RELEASE_LIFT_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                RIGHT_SECOND_PLACE_TF,
                right_second_release_tf,
                SECOND_RELEASE_CRACK_ALPHA,
                SECOND_RELEASE_CRACK_ALPHA,
            ),
            (
                SECOND_FINAL_HOLD_TIME,
                LEFT_SECOND_FOLD_TF,
                LEFT_SECOND_FOLD_TF,
                right_second_release_tf,
                right_second_release_tf,
                SECOND_RELEASE_CRACK_ALPHA,
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

    def _sample_script(self, query_time: float) -> tuple[wp.transform, wp.transform, float]:
        remaining = query_time
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

    def _is_first_release_no_friction_time(self, script_time: float) -> bool:
        return FIRST_RELEASE_START_TIME <= script_time <= FIRST_RELEASE_END_TIME

    def _is_second_release_no_friction_time(self, script_time: float) -> bool:
        return SECOND_RELEASE_START_TIME <= script_time <= SECOND_RELEASE_END_TIME

    def _set_robot_particle_contacts_enabled(self, enabled: bool) -> None:
        if self._robot_particle_contacts_enabled == enabled:
            return

        flags = self.model.shape_flags.numpy()
        if enabled:
            flags[: self.robot_shape_end] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        else:
            flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)
        self._robot_particle_contacts_enabled = enabled

    def _set_tshirt_contact_materials(self, cloth_mu: float, robot_mu: float, table_mu: float) -> None:
        self.model.soft_contact_mu = cloth_mu
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[: self.robot_shape_end] = robot_mu
        shape_mu[self.robot_shape_end :] = table_mu
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.device)

    def _set_robot_contact_stiffness(self, robot_ke: float) -> None:
        shape_ke = self.model.shape_material_ke.numpy()
        shape_ke[: self.robot_shape_end] = robot_ke
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.device)

    def _offset_transform(self, tf: wp.transform, offset: wp.vec3) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        return wp.transform(
            wp.vec3(float(pos[0]) + float(offset[0]), float(pos[1]) + float(offset[1]), float(pos[2]) + float(offset[2])),
            wp.transform_get_rotation(tf),
        )

    def _with_height(self, tf: wp.transform, height: float) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        return wp.transform(wp.vec3(float(pos[0]), float(pos[1]), height), wp.transform_get_rotation(tf))

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

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        show_triangles = getattr(self.viewer, "show_triangles", True)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        self.viewer.log_state(self.state_0)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = show_triangles
        self.viewer.log_mesh(
            "/fold_tshirt_unrotated",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not show_triangles,
            backface_culling=False,
            color=SHIRT_COLOR,
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = DexforceBimanualIKClothExample.create_parser()
        default_time_scale = 1.0
        parser.set_defaults(num_frames=int(np.ceil(SCRIPT_END_TIME * 60.0 / default_time_scale)) + 120)
        parser.add_argument(
            "--trajectory-time-scale",
            type=float,
            default=default_time_scale,
            help="Multiplier applied to captured trajectory time; values >1 replay faster.",
        )
        parser.add_argument(
            "--grasp-height-offset",
            type=float,
            default=GRASP_HEIGHT_OFFSET,
            help="Vertical offset [m] applied at the shirt grasp pose.",
        )
        parser.add_argument(
            "--grasp-lift-height",
            type=float,
            default=GRASP_LIFT_HEIGHT,
            help="Height [m] to lift from the grasp pose before folding.",
        )
        parser.add_argument(
            "--fold-travel-height",
            type=float,
            default=FOLD_TRAVEL_HEIGHT,
            help="Absolute TCP z height [m] used while translating toward the fold pose.",
        )
        parser.add_argument(
            "--fold-place-height-offset",
            type=float,
            default=FOLD_PLACE_HEIGHT_OFFSET,
            help="Vertical offset [m] applied at the fold placement pose.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
