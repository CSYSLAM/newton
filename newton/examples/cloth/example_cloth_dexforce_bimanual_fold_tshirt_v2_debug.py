# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual Fold T-Shirt
#
# Replays captured bimanual wrist TCP targets on the rotated T-shirt scene.
# The source IK gizmo scene remains available as
# ``cloth_dexforce_bimanual_ik_tshirt`` for trajectory collection.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_fold_tshirt
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
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_tshirt import (
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
    Example as DexforceBimanualIKTShirtExample,
)


LEFT_APPROACH_TF = wp.transform(
    wp.vec3(0.4117, 0.5145, 1.3642),
    wp.quat(0.1099, 0.6989, -0.6980, -0.1107),
)
RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(0.4053, -0.5598, 1.3642),
    wp.quat(-0.0658, 0.7043, 0.7037, -0.0668),
)
LEFT_GRASP_TF = wp.transform(
    wp.vec3(0.3565, 0.1627, 1.2406),
    wp.quat(0.1103, 0.6987, -0.6982, -0.1105),
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(0.3519, -0.1623, 1.2405),
    wp.quat(-0.0733, 0.7031, 0.7037, -0.0717),
)
LEFT_LIFT_TF = wp.transform(
    wp.vec3(0.3709, 0.1679, 1.3359),
    wp.quat(0.1105, 0.6985, -0.6981, -0.1119),
)
RIGHT_LIFT_TF = wp.transform(
    wp.vec3(0.3764, -0.1638, 1.3433),
    wp.quat(-0.0738, 0.7030, 0.7037, -0.0721),
)
LEFT_FOLD_TRAVEL_TF = wp.transform(
    wp.vec3(0.5542, 0.1679, 1.3376),
    wp.quat(0.1093, 0.6986, -0.6984, -0.1107),
)
RIGHT_FOLD_TRAVEL_TF = wp.transform(
    wp.vec3(0.6038, -0.1635, 1.3444),
    wp.quat(-0.0738, 0.7030, 0.7036, -0.0719),
)
LEFT_PLACE_TF = wp.transform(
    wp.vec3(0.6871, 0.1563, 1.2571),
    wp.quat(0.0195, 0.7068, -0.7067, -0.0247),
)
RIGHT_PLACE_TF = wp.transform(
    wp.vec3(0.6965, -0.1642, 1.2722),
    wp.quat(0.0237, 0.6998, 0.7126, -0.0439),
)
LEFT_RELEASE_TF = wp.transform(
    wp.vec3(0.6973, 0.1596, 1.4336),
    wp.quat(-0.0203, -0.7068, 0.7068, 0.0239),
)
RIGHT_RELEASE_TF = wp.transform(
    wp.vec3(0.6974, -0.1641, 1.4101),
    wp.quat(0.0238, 0.6998, 0.7126, -0.0441),
)

HOME_HOLD_TIME = 0.8
APPROACH_TIME = 2.0
DESCEND_TIME = 1.2
GRASP_CLOSE_TIME = 1.4
LIFT_TIME = 1.4
FOLD_TRAVEL_TIME = 2.4
FOLD_PLACE_TIME = 1.4
RELEASE_TIME = 1.0
RELEASE_LIFT_TIME = 1.2
HOLD_TIME = 1.0

RELEASE_CRACK_ALPHA = 0.75
RELEASE_START_TIME = (
    HOME_HOLD_TIME
    + APPROACH_TIME
    + DESCEND_TIME
    + GRASP_CLOSE_TIME
    + LIFT_TIME
    + FOLD_TRAVEL_TIME
    + FOLD_PLACE_TIME
)
RELEASE_END_TIME = RELEASE_START_TIME + RELEASE_TIME + RELEASE_LIFT_TIME
SCRIPT_END_TIME = (
    RELEASE_END_TIME
    + HOLD_TIME
)
POST_FOLD_IK_START_TIME = RELEASE_END_TIME
POST_FOLD_IK_EXTRA_SECONDS = 60.0

GRASP_CLOTH_SOFT_CONTACT_MU = 1.20
GRASP_ROBOT_CLOTH_CONTACT_MU = 1.20
GRASP_TABLE_CLOTH_CONTACT_MU = 0.12
RELEASE_CLOTH_SOFT_CONTACT_MU = 0.50
RELEASE_ZERO_CLOTH_SOFT_CONTACT_MU = 0.0
RELEASE_ROBOT_CLOTH_CONTACT_MU = 0.25
RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU = 0.0
RELEASE_TABLE_CLOTH_CONTACT_MU = 0.5

TSHIRT_SIM_SUBSTEPS = 10
TSHIRT_SOLVER_ITERATIONS = 20
TSHIRT_SOFT_CONTACT_KE = 3.0e5
TSHIRT_SOFT_CONTACT_KD = 5.0e-2
TSHIRT_ROBOT_CONTACT_KE = 3.0e5
TSHIRT_TRI_KE = 1.5e3
TSHIRT_TRI_KA = 1.5e3
TSHIRT_TRI_KD = 1.0e-5
TSHIRT_EDGE_KE = 1.2
TSHIRT_EDGE_KD = 0.1


class Example(DexforceBimanualIKTShirtExample):
    def __init__(self, viewer, args):
        self.trajectory_time_scale = float(args.trajectory_time_scale)
        self.post_fold_ik_active = False
        super().__init__(viewer, args)
        self.left_home_tf = self.left_tf
        self.right_home_tf = self.right_tf
        self.motion_segments = self._build_motion_segments()

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

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

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
            edge_ke=TSHIRT_EDGE_KE,
            edge_kd=TSHIRT_EDGE_KD,
            particle_radius=self.particle_radius,
        )

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        script_time = (self.sim_time + self.frame_dt) * self.trajectory_time_scale

        if script_time >= POST_FOLD_IK_START_TIME:
            if not self.post_fold_ik_active:
                self.left_tf, self.right_tf, _ = self._sample_script(POST_FOLD_IK_START_TIME)
                self.post_fold_ik_active = True
            grasp_alpha = 1.0
            self._set_tshirt_contact_materials(
                RELEASE_ZERO_CLOTH_SOFT_CONTACT_MU,
                RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU,
                RELEASE_TABLE_CLOTH_CONTACT_MU,
            )
        else:
            self.left_tf, self.right_tf, grasp_alpha = self._sample_script(script_time)

            if self._is_release_no_friction_time(script_time):
                self._set_tshirt_contact_materials(
                    RELEASE_ZERO_CLOTH_SOFT_CONTACT_MU,
                    RELEASE_ZERO_ROBOT_CLOTH_CONTACT_MU,
                    RELEASE_TABLE_CLOTH_CONTACT_MU,
                )
            elif grasp_alpha > 0.15:
                self._set_tshirt_contact_materials(
                    GRASP_CLOTH_SOFT_CONTACT_MU,
                    GRASP_ROBOT_CLOTH_CONTACT_MU,
                    GRASP_TABLE_CLOTH_CONTACT_MU,
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
            inputs=[self.hand_q_indices, self.hand_open, self.hand_grasp, grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def _build_motion_segments(self) -> tuple[tuple[float, wp.transform, wp.transform, wp.transform, wp.transform, float, float], ...]:
        return (
            (HOME_HOLD_TIME, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (
                APPROACH_TIME,
                self.left_home_tf,
                LEFT_APPROACH_TF,
                self.right_home_tf,
                RIGHT_APPROACH_TF,
                0.0,
                0.0,
            ),
            (
                DESCEND_TIME,
                LEFT_APPROACH_TF,
                LEFT_GRASP_TF,
                RIGHT_APPROACH_TF,
                RIGHT_GRASP_TF,
                0.0,
                0.0,
            ),
            (GRASP_CLOSE_TIME, LEFT_GRASP_TF, LEFT_GRASP_TF, RIGHT_GRASP_TF, RIGHT_GRASP_TF, 0.0, 1.0),
            (
                LIFT_TIME,
                LEFT_GRASP_TF,
                LEFT_LIFT_TF,
                RIGHT_GRASP_TF,
                RIGHT_LIFT_TF,
                1.0,
                1.0,
            ),
            (
                FOLD_TRAVEL_TIME,
                LEFT_LIFT_TF,
                LEFT_FOLD_TRAVEL_TF,
                RIGHT_LIFT_TF,
                RIGHT_FOLD_TRAVEL_TF,
                1.0,
                1.0,
            ),
            (
                FOLD_PLACE_TIME,
                LEFT_FOLD_TRAVEL_TF,
                LEFT_PLACE_TF,
                RIGHT_FOLD_TRAVEL_TF,
                RIGHT_PLACE_TF,
                1.0,
                1.0,
            ),
            (RELEASE_TIME, LEFT_PLACE_TF, LEFT_PLACE_TF, RIGHT_PLACE_TF, RIGHT_PLACE_TF, 1.0, RELEASE_CRACK_ALPHA),
            (
                RELEASE_LIFT_TIME,
                LEFT_PLACE_TF,
                LEFT_RELEASE_TF,
                RIGHT_PLACE_TF,
                RIGHT_RELEASE_TF,
                RELEASE_CRACK_ALPHA,
                0.0,
            ),
            (HOLD_TIME, LEFT_RELEASE_TF, LEFT_RELEASE_TF, RIGHT_RELEASE_TF, RIGHT_RELEASE_TF, 0.0, 0.0),
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

    def _is_release_no_friction_time(self, script_time: float) -> bool:
        return RELEASE_START_TIME <= script_time <= RELEASE_END_TIME

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
        if self.post_fold_ik_active and hasattr(self.viewer, "log_gizmo"):
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
        show_triangles = getattr(self.viewer, "show_triangles", True)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        self.viewer.log_state(self.state_0)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = show_triangles
        self.viewer.log_mesh(
            "/fold_tshirt",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not show_triangles,
            backface_culling=False,
            color=SHIRT_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Cloth particle positions are not finite")

    @staticmethod
    def create_parser():
        parser = DexforceBimanualIKTShirtExample.create_parser()
        default_time_scale = 4.0
        script_frames = int(np.ceil(POST_FOLD_IK_START_TIME * 60.0 / default_time_scale))
        parser.set_defaults(num_frames=script_frames + int(POST_FOLD_IK_EXTRA_SECONDS * 60.0))
        parser.add_argument(
            "--trajectory-time-scale",
            type=float,
            default=default_time_scale,
            help="Multiplier applied to captured trajectory time; values >1 replay faster.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
