# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Body Franka MJVBD Tet Grasp
#
# Demonstrates a Franka Panda robot grasping a simple tetrahedral soft block
# from a table. The motion follows a compact open -> first close -> lift
# sequence so the grasp timing is easy to inspect and modify.
#
# Command: python -m newton.examples.softbody.example_softbody_franka_mjvbd_tet_grasp
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.mjvbd
import newton.utils
from newton import ModelBuilder, eval_fk


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


class Example:
    def __init__(self, viewer, args=None):
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.enable_franka = True if args is None else getattr(args, "enable_franka", True) and not getattr(args, "disable_franka", False)

        self.particle_radius = 0.006
        self.soft_body_contact_margin = 0.012
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.006

        self.soft_contact_ke = 2.5e6
        self.soft_contact_kd = 1.0e-7
        self.self_contact_friction = 0.6

        self.table_top_z = 0.2
        self.block_dim_x = 3
        self.block_dim_y = 2
        self.block_dim_z = 4
        self.block_cell_x = 0.025
        self.block_cell_y = 0.022
        self.block_cell_z = 0.025
        self.block_size = np.array(
            [
                self.block_dim_x * self.block_cell_x,
                self.block_dim_y * self.block_cell_y,
                self.block_dim_z * self.block_cell_z,
            ],
            dtype=np.float32,
        )
        self.block_origin = wp.vec3(
            -0.5 * float(self.block_size[0]),
            -0.5 - 0.5 * float(self.block_size[1]),
            self.table_top_z + 0.004,
        )
        self.initial_block_top_z = float(self.block_origin[2]) + float(self.block_size[2])
        self.max_particle_height = self.initial_block_top_z
        self.expected_lift_height = self.initial_block_top_z + 0.012

        self.scene = ModelBuilder(gravity=-9.81)
        self.viewer = viewer

        franka = ModelBuilder()
        newton.mjvbd.SolverMJVBD.register_custom_attributes(franka)
        self.create_articulation(franka)
        self.scene.add_world(franka)

        table_hx = 0.4
        table_hy = 0.4
        table_hz = 0.1
        table_pos = wp.vec3(0.0, -0.5, 0.1)
        self.scene.add_shape_box(
            -1,
            wp.transform(table_pos, wp.quat_identity()),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
        )

        self.scene.add_soft_grid(
            pos=self.block_origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self.block_dim_x,
            dim_y=self.block_dim_y,
            dim_z=self.block_dim_z,
            cell_x=self.block_cell_x,
            cell_y=self.block_cell_y,
            cell_z=self.block_cell_z,
            density=200.0,
            k_mu=8.0e5,
            k_lambda=8.0e5,
            k_damp=5.0e-6,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0e-6,
            edge_ke=4.0,
            edge_kd=5.0e-3,
            particle_radius=self.particle_radius,
        )

        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        self.model.shape_material_ke.fill_(self.soft_contact_ke)
        self.model.shape_material_kd.fill_(self.soft_contact_kd)
        self.model.shape_material_mu.fill_(1.75)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.control = self.model.control()
        wp.copy(
            self.control.joint_target_pos[: self.model.joint_coord_count],
            self.model.joint_q[: self.model.joint_coord_count],
        )

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=32,
            soft_contact_margin=self.soft_body_contact_margin,
            step_rigid_bodies=True,
            rigid_njmax=768,
            rigid_nconmax=768,
            iterations=self.iterations,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )

        self.set_up_ik()
        self.build_joint_target_cache()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.capture()

    def set_up_ik(self):
        self.n_coords = self.model.joint_coord_count
        self.n_dofs = self.model.joint_dof_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.zeros(1, dtype=float)
        self.target_joint_q = wp.zeros(self.n_coords, dtype=float)

        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=wp.vec3(0.0, 0.0, 0.22),
            target_positions=wp.array([target_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([target_rot], dtype=wp.vec4),
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
        self.current_gripper_target = float(self.targets[0][-1])

    def capture(self):
        if wp.get_device().is_cuda:
            self.set_joint_targets()
            self.solver.reset_joint_target_frame()
            with wp.ScopedCapture() as capture:
                if self.enable_franka:
                    self.solver.load_joint_targets_from_counter(self.control)
                self._simulate_substeps()
                if self.enable_franka:
                    self.solver.advance_joint_target_frame()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        gripper_open = 1.0
        gripper_close = 0.52

        finger_open = gripper_open * 0.04
        builder.joint_q[:9] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0, finger_open, finger_open]
        builder.joint_target_pos[:9] = [*builder.joint_q[:9]]
        builder.joint_target_ke[:9] = [650.0] * 7 + [1200.0, 1200.0]
        builder.joint_target_kd[:9] = [100.0] * 7 + [180.0, 180.0]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [40.0, 40.0]
        builder.joint_armature[:7] = [0.1] * 7
        builder.joint_armature[7:9] = [0.5, 0.5]

        grasp_x = 0.0
        grasp_y = -0.5
        approach_z = self.initial_block_top_z + 0.10
        close_z = self.initial_block_top_z - 0.01
        lift_z = self.initial_block_top_z + 0.12

        self.robot_key_poses = np.array(
            [
                [2.0, grasp_x, grasp_y, approach_z, 1.0, 0.0, 0.0, 0.0, gripper_open],
                [1.0, grasp_x, grasp_y, close_z, 1.0, 0.0, 0.0, 0.0, gripper_open],
                [1.0, grasp_x, grasp_y, close_z, 1.0, 0.0, 0.0, 0.0, gripper_close],
                [1.5, grasp_x, grasp_y, lift_z, 1.0, 0.0, 0.0, 0.0, gripper_close],
                [1.5, grasp_x + 0.08, grasp_y + 0.04, lift_z, 1.0, 0.0, 0.0, 0.0, gripper_close],
                [1.0, grasp_x + 0.08, grasp_y + 0.04, close_z + 0.04, 1.0, 0.0, 0.0, 0.0, gripper_close],
                [1.0, grasp_x + 0.08, grasp_y + 0.04, close_z + 0.04, 1.0, 0.0, 0.0, 0.0, gripper_open],
                [1.5, grasp_x + 0.08, grasp_y + 0.04, approach_z, 1.0, 0.0, 0.0, 0.0, gripper_open],
            ],
            dtype=np.float32,
        )

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3

    def interpolated_target(self, query_time: float | None = None) -> np.ndarray:
        sample_time = self.sim_time if query_time is None else query_time

        if sample_time >= self.robot_key_poses_time[-1]:
            return self.targets[-1]

        current_interval = int(np.searchsorted(self.robot_key_poses_time, sample_time))
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        alpha = float(np.clip((sample_time - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else target_cur
        return (1.0 - alpha) * target_prev + alpha * target_cur

    def set_joint_targets(self, query_time: float | None = None):
        sample_time = self.sim_time if query_time is None else query_time
        if self.solver.load_joint_targets(self.control, sample_time, self.fps):
            return

        self._set_joint_targets_via_ik(query_time)

    def _set_joint_targets_via_ik(self, query_time: float | None = None):
        target_interp = self.interpolated_target(query_time)

        self.pos_obj.set_target_position(0, wp.vec3(*target_interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_interp[3:7].tolist()))

        finger_pos = float(target_interp[-1]) * 0.04
        self.finger_pos_buf.fill_(finger_pos)

        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.model.device,
        )
        wp.copy(self.target_joint_q, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)
        wp.copy(self.control.joint_target_pos[: self.n_coords], self.target_joint_q)

    def build_joint_target_cache(self):
        cache_frame_count = int(np.ceil(self.robot_key_poses_time[-1] * self.fps))
        cache = np.zeros((cache_frame_count + 1, self.n_coords), dtype=np.float32)

        for frame_index in range(cache_frame_count + 1):
            query_time = min(frame_index * self.frame_dt, float(self.robot_key_poses_time[-1]))
            self._set_joint_targets_via_ik(query_time)
            wp.synchronize_device()
            cache[frame_index] = self.target_joint_q.numpy()[: self.n_coords]

        self.solver.set_joint_target_cache(wp.array(cache, dtype=wp.float32, device=self.model.device), cache_frame_count)
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))
        wp.copy(self.control.joint_target_pos[: self.n_coords], self.model.joint_q[: self.n_coords])

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.set_joint_targets()
            self.simulate()

    def _simulate_substeps(self):
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        self._simulate_substeps()

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        max_height = float(np.max(self.state_0.particle_q.numpy()[:, 2]))
        self.max_particle_height = max(self.max_particle_height, max_height)

    def test_final(self):
        p_lower = wp.vec3(-0.5, -1.0, -0.05)
        p_upper = wp.vec3(0.5, 0.1, 0.7)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 3.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.0,
        )
        if self.max_particle_height <= self.expected_lift_height:
            raise ValueError(
                f"Expected the tet block to lift above {self.expected_lift_height:.3f} m, got {self.max_particle_height:.3f} m"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)