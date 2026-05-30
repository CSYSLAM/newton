# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka MuJoCo
#
# This simulation reuses the shirt/table/Franka setup from
# ``example_cloth_franka.py`` but drives the robot with ``SolverMuJoCo``
# instead of the Featherstone-based kinematic update. The cloth is still
# simulated with VBD in the same shared model, without any coupled solver
# wrapper.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples cloth_franka_mujoco
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverMuJoCo, SolverVBD


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


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viz_scale = 0.01

        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        self.particle_self_contact_radius = 0.2
        self.particle_self_contact_margin = 0.2

        self.soft_contact_ke = 1.0e4
        self.soft_contact_kd = 1.0e-2

        self.robot_contact_ke = 5.0e4
        self.robot_contact_kd = 1.0e-3
        self.robot_contact_mu = 1.5
        self.self_contact_friction = 0.25

        self.tri_ke = 1.0e3
        self.tri_ka = 1.0e3
        self.tri_kd = 1.0e-5

        self.bending_ke = 1.0
        self.bending_kd = 0.1

        self.scene = ModelBuilder(gravity=-981.0)

        franka = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(franka)
        self.create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count

        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 1
        self.table_pos_cm = wp.vec3(0.0, -50.0, 2.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(self.table_pos_cm, wp.quat_identity()),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)

        self.scene.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in shirt_mesh.vertices],
            indices=shirt_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 70.0, 30.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.02,
            scale=1.0,
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
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu
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

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverMuJoCo(self.model, use_mujoco_contacts=False, njmax=768, nconmax=768)
        self.cloth_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
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

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

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

        gripper_open = 3.2
        gripper_close = 0.4
        builder.joint_q[7:9] = [gripper_open, gripper_open]
        builder.joint_target_pos[:9] = [*builder.joint_q[:9]]
        builder.joint_target_ke[:9] = [4000.0] * 7 + [12000.0, 12000.0]
        builder.joint_target_kd[:9] = [400.0] * 7 + [1200.0, 1200.0]
        builder.joint_effort_limit[:7] = [300.0] * 7
        builder.joint_effort_limit[7:9] = [2000.0, 2000.0]
        builder.joint_armature[:7] = [0.2] * 7
        builder.joint_armature[7:9] = [0.5, 0.5]

        self.robot_key_poses = np.array(
            [
                [4.0, 30.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_open],
                [2.0, 30.0, -58.0,  7.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_open],
                [2.0, 30.0, -58.0,  7.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_close],
                [2.0, 25.0, -60.0, 13.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_close],
                [2.0, 12.0, -60.0, 23.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_close],
                [3.0, -6.0, -60.0, 23.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_close],
                [3.0, -6.0, -60.0, 10.0, 0.8536, -0.3536, 0.3536, -0.1464, gripper_open],
            ],
            dtype=np.float32,
        )
        self.robot_targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.vec3(0.0, 0.0, 22.0)

    def setup_ik(self) -> None:
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=self.endeffector_offset,
            target_positions=wp.array([wp.vec3(*self.robot_targets[0][:3].tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.robot_targets[0][3:7].tolist())], dtype=wp.vec4),
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
        self.current_gripper_target = float(self.robot_targets[0][-1])

    def interpolated_target(self) -> np.ndarray:
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return self.robot_targets[-1]

        interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        t_start = float(self.robot_key_poses_time[interval - 1]) if interval > 0 else 0.0
        t_end = float(self.robot_key_poses_time[interval])
        alpha = float(np.clip((self.sim_time - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        target_cur = self.robot_targets[interval]
        target_prev = self.robot_targets[interval - 1] if interval > 0 else target_cur
        return (1.0 - alpha) * target_prev + alpha * target_cur

    def set_joint_targets(self) -> None:
        target = self.interpolated_target()
        self.pos_obj.set_target_position(0, wp.vec3(*target[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target[3:7].tolist()))
        self.current_gripper_target = float(target[-1])
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=1,
            inputs=[self.ik_joint_q, self.control.joint_target_pos, self.current_gripper_target],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self.cloth_solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # External-rigid VBD expects state_0 to hold the previous rigid pose
            # and state_1 to hold the current rigid pose for this substep.
            self.state_1.assign(self.state_0)
            self.set_joint_targets()
            wp.copy(self.state_1.joint_q, self.control.joint_target_pos)
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.state_0.joint_q, self.state_1.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_1.joint_qd],
                device=self.model.device,
            )
            eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.collision_pipeline.collide(self.state_1, self.contacts)
            self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def step(self) -> None:
        self.simulate()

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
        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )
        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self) -> None:
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=3850)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)