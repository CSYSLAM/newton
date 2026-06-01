# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka MJVBD Style3D IK
#
# Same table + Style3D garment scene as the interactive Franka cloth example,
# but the Franka arm is driven directly from a TCP gizmo like
# ``example_ik_franka.py``. The arm is visible, solves IK every frame, and
# reports both target and solved TCP poses in scene units [cm], while robot
# shapes are marked non-colliding so cloth simulates independently.
#
# Command: python -m newton.examples.cloth.example_cloth_franka_mjvbd_style3d_ik
#          python -m newton.examples cloth_franka_mjvbd_style3d_ik
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.mjvbd
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_target: float,
):
    joint_targets[0] = ik_solution[0, 0]
    joint_targets[1] = ik_solution[0, 1]
    joint_targets[2] = ik_solution[0, 2]
    joint_targets[3] = ik_solution[0, 3]
    joint_targets[4] = ik_solution[0, 4]
    joint_targets[5] = ik_solution[0, 5]
    joint_targets[6] = ik_solution[0, 6]
    joint_targets[7] = gripper_target
    joint_targets[8] = gripper_target


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
def interpolate_joint_positions_kernel(
    joint_q_start: wp.array[wp.float32],
    joint_q_end: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = joint_q_start[i] * (1.0 - alpha) + joint_q_end[i] * alpha


GARMENT_USD_NAME = "Women_Sweatshirt"
GARMENT_SCALE = 75.0
GARMENT_DENSITY = 0.02
GARMENT_POS = wp.vec3(0.0, -50.0, 30.0)
GARMENT_ROT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)

GRIPPER_OPEN = 3.2
DEFAULT_GRIPPER_WIDTH = GRIPPER_OPEN

CLOTH_COLOR = (0.72, 0.72, 0.76)
TABLE_HX_CM = 40.0
TABLE_HY_CM = 40.0
TABLE_HZ_CM = 1.0
TABLE_POS_CM = wp.vec3(0.0, -50.0, 2.0)
TABLE_COLOR = wp.vec3(0.16, 0.55, 0.78)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.viz_scale = 0.01

        self.gripper_width = float(getattr(args, "gripper_width", DEFAULT_GRIPPER_WIDTH))
        self.pose_report_interval = 0.1
        self.last_pose_report_time = -1.0
        self.last_reported_pose = None

        self.cloth_particle_radius = 0.35
        self.cloth_body_contact_margin = 0.4
        self.particle_self_contact_radius = 0.1
        self.particle_self_contact_margin = 0.1

        self.soft_contact_ke = 1.0e4
        self.soft_contact_kd = 1.0e-2
        self.self_contact_friction = 0.25
        self.table_contact_ke = 5.0e4
        self.table_contact_kd = 1.0e-3
        self.table_contact_mu = 1.5

        self.tri_ke = 1.0e3
        self.tri_ka = 1.0e3
        self.tri_kd = 1.0e-5
        self.bending_ke = 1.0
        self.bending_kd = 0.1

        self.scene = ModelBuilder(gravity=-981.0)

        franka = ModelBuilder()
        self.create_articulation(franka)
        self.robot_shape_start = self.scene.shape_count
        self.scene.add_world(franka)
        self.robot_shape_end = self.scene.shape_count

        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(TABLE_POS_CM, wp.quat_identity()),
            hx=TABLE_HX_CM,
            hy=TABLE_HY_CM,
            hz=TABLE_HZ_CM,
        )

        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(str(asset_path / "garments" / f"{GARMENT_USD_NAME}.usd"))
        usd_prim = usd_stage.GetPrimAtPath(f"/Root/{GARMENT_USD_NAME}/Root_Garment")
        garment_mesh = newton.usd.get_mesh(usd_prim)
        garment_vertices = self._center_garment_vertices(garment_mesh.vertices)

        self.scene.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in garment_vertices],
            indices=garment_mesh.indices,
            rot=GARMENT_ROT,
            pos=GARMENT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=GARMENT_DENSITY,
            scale=GARMENT_SCALE,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

        self.scene.color(include_bending=True)
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        flags = self.model.shape_flags.numpy()
        robot_no_collision = ~(int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES))
        flags[self.robot_shape_start : self.robot_shape_end] &= robot_no_collision
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(TABLE_POS_CM[0]) * self.viz_scale,
                        float(TABLE_POS_CM[1]) * self.viz_scale,
                        float(TABLE_POS_CM[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            TABLE_HX_CM * self.viz_scale,
            TABLE_HY_CM * self.viz_scale,
            TABLE_HZ_CM * self.viz_scale,
        )
        self.table_viz_color = wp.array([TABLE_COLOR], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.table_contact_ke
        shape_kd[...] = self.table_contact_kd
        shape_mu[...] = self.table_contact_mu
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.viz_state = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=self.cloth_body_contact_margin,
            iterations=self.iterations,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.setup_ik()
        self.ee_tf = self._tcp_transform_to_viz(self._current_tcp_transform_cm())
        self._report_pose(force=True)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.72, 0.62, 1.28), -41.0, -60.0)

    def _center_garment_vertices(self, vertices: np.ndarray) -> np.ndarray:
        vertices_np = np.asarray(vertices, dtype=np.float32)
        bbox_min = vertices_np.min(axis=0)
        bbox_max = vertices_np.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        return vertices_np - bbox_center

    def create_articulation(self, builder: ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=100.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:7] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.7854]
        builder.joint_q[7:9] = [self.gripper_width, self.gripper_width]
        builder.joint_target_pos[:9] = [*builder.joint_q[:9]]
        builder.joint_target_ke[:9] = [4000.0] * 7 + [12000.0, 12000.0]
        builder.joint_target_kd[:9] = [400.0] * 7 + [1200.0, 1200.0]
        builder.joint_effort_limit[:7] = [300.0] * 7
        builder.joint_effort_limit[7:9] = [2000.0, 2000.0]
        builder.joint_armature[:7] = [0.2] * 7
        builder.joint_armature[7:9] = [0.5, 0.5]

        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.vec3(0.0, 0.0, 22.0)

    def setup_ik(self) -> None:
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        tcp_tf = self._current_tcp_transform_cm()
        tcp_pos = wp.transform_get_translation(tcp_tf)
        tcp_rot = wp.transform_get_rotation(tcp_tf)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=self.endeffector_offset,
            target_positions=wp.array([tcp_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(tcp_rot[0], tcp_rot[1], tcp_rot[2], tcp_rot[3])], dtype=wp.vec4),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _current_tcp_transform_cm(self) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[self.endeffector_id])
        return self._tcp_from_body_transform(body_tf)

    def _tcp_from_body_transform(self, body_tf: wp.transform) -> wp.transform:
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, self.endeffector_offset)
        return wp.transform(tcp_pos, body_rot)

    def _tcp_transform_to_viz(self, tcp_tf_cm: wp.transform) -> wp.transform:
        pos_cm = wp.transform_get_translation(tcp_tf_cm)
        rot = wp.transform_get_rotation(tcp_tf_cm)
        pos_m = wp.vec3(float(pos_cm[0]) * self.viz_scale, float(pos_cm[1]) * self.viz_scale, float(pos_cm[2]) * self.viz_scale)
        return wp.transform(pos_m, rot)

    def _push_targets_from_gizmo(self) -> None:
        pos_m = wp.transform_get_translation(self.ee_tf)
        pos_cm = wp.vec3(
            float(pos_m[0]) / self.viz_scale,
            float(pos_m[1]) / self.viz_scale,
            max(float(pos_m[2]) / self.viz_scale, 6.0),
        )
        rot = wp.transform_get_rotation(self.ee_tf)

        self.pos_obj.set_target_position(0, pos_cm)
        self.rot_obj.set_target_rotation(0, wp.vec4(rot[0], rot[1], rot[2], rot[3]))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=1,
            inputs=[self.ik_joint_q, self.control.joint_target_pos, self.gripper_width],
            device=self.model.device,
        )

    def _prepare_frame_inputs(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        self._push_targets_from_gizmo()
        wp.copy(self.frame_joint_q_end, self.control.joint_target_pos)

    def _simulate_substeps(self) -> None:
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.copy(self.state_1.particle_q, self.state_0.particle_q)

            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_1.joint_q],
                device=self.model.device,
            )
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.state_0.joint_q, self.state_1.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_1.joint_qd],
                device=self.model.device,
            )
            eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _pose_list_from_transform(self, tf: wp.transform, pos_scale: float = 1.0) -> list[float]:
        pos = wp.transform_get_translation(tf)
        rot = wp.transform_get_rotation(tf)
        return [
            float(pos[0]) * pos_scale,
            float(pos[1]) * pos_scale,
            float(pos[2]) * pos_scale,
            float(rot[0]),
            float(rot[1]),
            float(rot[2]),
            float(rot[3]),
        ]

    def _report_pose(self, force: bool = False) -> None:
        target_pose = np.array(self._pose_list_from_transform(self.ee_tf, pos_scale=1.0 / self.viz_scale), dtype=np.float64)
        actual_pose = np.array(self._pose_list_from_transform(self._current_tcp_transform_cm()), dtype=np.float64)
        report_pose = np.concatenate([target_pose, actual_pose])

        if not force:
            if self.sim_time - self.last_pose_report_time < self.pose_report_interval:
                return
            if self.last_reported_pose is not None and np.max(np.abs(report_pose - self.last_reported_pose)) < 0.05:
                return

        self.last_pose_report_time = self.sim_time
        self.last_reported_pose = report_pose
        target_str = ", ".join(f"{v:.4f}" for v in target_pose)
        actual_str = ", ".join(f"{v:.4f}" for v in actual_pose)
        print(f"target_tcp_cm = [{target_str}]")
        print(f"actual_tcp_cm = [{actual_str}]")

    def step(self) -> None:
        self._prepare_frame_inputs()
        self._simulate_substeps()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self) -> None:
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_mesh(
            "/model/triangles",
            self.viz_state.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
            color=CLOTH_COLOR,
        )
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )

        tcp_snap_tf = self._tcp_transform_to_viz(self._current_tcp_transform_cm())
        self.viewer.log_gizmo("target_tcp", self.ee_tf, snap_to=tcp_snap_tf)
        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

        self._report_pose()

    def test_final(self) -> None:
        p_lower = wp.vec3(-46.0, -96.0, -5.0)
        p_upper = wp.vec3(46.0, 8.0, 60.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--gripper-width", type=float, default=DEFAULT_GRIPPER_WIDTH, help="Franka finger opening in scene units [cm].")
        parser.set_defaults(num_frames=20000)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)