# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo-VBD Cloth Franka Coupled Solver
#
# A Franka Panda arm tracks IK pose targets with MuJoCo while a shirt mesh is
# simulated with VBD. SolverCoupledProxy maps the robot links into VBD proxy
# bodies so cloth-hand contacts are resolved on the cloth side and the
# resulting reaction impulses are fed back to the robot.
#
# The shirt asset and grasp trajectory are adapted from ``example_cloth_franka``
# but the rigid/deformable interaction now runs through the generic coupled
# solver path used by the multiphysics demos.
#
# Command: python -m newton.examples mujoco_vbd_cloth_franka_coupled_solver
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
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_value: float,
):
    joint_targets[0] = ik_solution[0, 0]
    joint_targets[1] = ik_solution[0, 1]
    joint_targets[2] = ik_solution[0, 2]
    joint_targets[3] = ik_solution[0, 3]
    joint_targets[4] = ik_solution[0, 4]
    joint_targets[5] = ik_solution[0, 5]
    joint_targets[6] = ik_solution[0, 6]
    joint_targets[7] = gripper_value
    joint_targets[8] = gripper_value


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0

        self.vbd_iterations = int(args.vbd_iterations)
        self.proxy_iterations = int(args.proxy_iterations)

        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.014
        self.particle_self_contact_radius = 0.002
        self.particle_self_contact_margin = 0.002

        self.soft_contact_ke = 1.0e4
        self.soft_contact_kd = 1.0e-2
        self.soft_contact_mu = 0.25

        self.robot_contact_ke = 5.0e4
        self.robot_contact_kd = 1.0e-3
        self.robot_contact_mu = 1.5
        self.gripper_contact_mu = 5.0 

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_ground_plane()

        self.robot_body_start = builder.body_count
        self._add_franka(builder)
        self.robot_body_end = builder.body_count
        self.robot_body_indices = list(range(self.robot_body_start, self.robot_body_end))
        self.proxy_body_indices = [self.ee_body, self.left_finger_body, self.right_finger_body]

        builder.add_shape_box(
            -1,
            wp.transform((0.0, -0.5, 0.02), wp.quat_identity()),
            hx=0.4,
            hy=0.4,
            hz=0.01,
        )

        self.cloth_particle_start = builder.particle_count
        self._add_shirt(builder)
        self.cloth_particle_end = builder.particle_count
        self.cloth_particle_indices = list(range(self.cloth_particle_start, self.cloth_particle_end))

        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.model.edge_rest_angle.zero_()
        self._configure_materials()

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="robot",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        use_mujoco_contacts=False,
                        njmax=256,
                        nconmax=256,
                    ),
                    bodies=self.robot_body_indices,
                    joints=list(range(self.model.joint_count)),
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(
                        model=v,
                        iterations=self.vbd_iterations,
                        friction_epsilon=1.0e-3,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=self.particle_self_contact_radius,
                        particle_self_contact_margin=self.particle_self_contact_margin,
                        particle_topological_contact_filter_threshold=1,
                        particle_rest_shape_contact_exclusion_radius=0.005,
                        particle_vertex_contact_buffer_size=16,
                        particle_edge_contact_buffer_size=20,
                        particle_collision_detection_interval=-1,
                    ),
                    particles=self.cloth_particle_indices,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="robot",
                        destination="vbd",
                        bodies=self.proxy_body_indices,
                        mass_scale=args.mass_scale,
                        mode=args.coupling_mode,
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model,
                            args,
                            soft_contact_margin=self.cloth_body_contact_margin,
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=self.proxy_iterations,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        self._setup_ik()
        self._select_grasp_particles()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.55, 0.35, 0.95), -38.0, -34.0)

    def _find_label_index(self, labels: list[str], name: str) -> int:
        for index, label in enumerate(labels):
            short = label.rsplit("/", 1)[-1] if "/" in label else label
            if short == name:
                return index
        raise ValueError(f"Could not find body label {name!r}")

    def _add_franka(self, builder: newton.ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, 0.0), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )

        init_q = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.7854]
        gripper_open = 0.020
        builder.joint_q[:9] = [*init_q, gripper_open, gripper_open]
        builder.joint_target_pos[:9] = [*init_q, gripper_open, gripper_open]
        builder.joint_target_ke[:9] = [4000.0] * 7 + [12000.0, 12000.0]
        builder.joint_target_kd[:9] = [400.0] * 7 + [1200.0, 1200.0]
        builder.joint_effort_limit[:7] = [300.0] * 7
        builder.joint_effort_limit[7:9] = [2000.0, 2000.0]
        builder.joint_armature[:7] = [0.2] * 7
        builder.joint_armature[7:9] = [0.5] * 2

        self.ee_body = self._find_label_index(builder.body_label, "fr3_link7")
        self.left_finger_body = self._find_label_index(builder.body_label, "fr3_leftfinger")
        self.right_finger_body = self._find_label_index(builder.body_label, "fr3_rightfinger")
        self.ee_offset = wp.vec3(0.0, 0.0, 0.22)

        gripper_closed = 0.004
        grasp_quat = [0.8536, -0.3536, 0.3536, -0.1464]
        self.robot_key_poses = np.array(
            [
                [1.5, 0.33, -0.60, 0.34, *grasp_quat, gripper_open],
                [1.5, 0.33, -0.57, 0.13, *grasp_quat, gripper_open],
                [1.2, 0.33, -0.54, 0.11, *grasp_quat, gripper_open],
                [1.2, 0.33, -0.57, 0.11, *grasp_quat, gripper_closed],
                [1.0, 0.30, -0.58, 0.18, *grasp_quat, gripper_closed],
                [2.0, 0.26, -0.60, 0.20, *grasp_quat, gripper_closed],
                [2.0, 0.12, -0.60, 0.33, *grasp_quat, gripper_closed],
                [1.5, -0.05, -0.60, 0.34, *grasp_quat, gripper_closed],
                [2.0, -0.05, -0.60, 0.34, *grasp_quat, gripper_closed],
            ],
            dtype=np.float32,
        )
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.robot_targets = self.robot_key_poses[:, 1:]

    def _add_shirt(self, builder: newton.ModelBuilder) -> None:
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.70, 0.30),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            scale=0.01,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(v) for v in shirt_mesh.vertices],
            indices=shirt_mesh.indices,
            density=0.02,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.5e-6,
            edge_ke=5.0,
            edge_kd=1.0e-2,
            particle_radius=self.cloth_particle_radius,
        )

    def _configure_materials(self) -> None:
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.soft_contact_mu

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_body = self.model.shape_body.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

        for shape_index, body_index in enumerate(shape_body):
            if body_index in (self.left_finger_body, self.right_finger_body):
                shape_mu[shape_index] = self.gripper_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke,
            dtype=self.model.shape_material_ke.dtype,
            device=self.model.device,
        )
        self.model.shape_material_kd = wp.array(
            shape_kd,
            dtype=self.model.shape_material_kd.dtype,
            device=self.model.device,
        )
        self.model.shape_material_mu = wp.array(
            shape_mu,
            dtype=self.model.shape_material_mu.dtype,
            device=self.model.device,
        )

    def _setup_ik(self) -> None:
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_body,
            link_offset=self.ee_offset,
            target_positions=wp.array([wp.vec3(*self.robot_targets[0][:3].tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_body,
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

    def _interpolated_target(self) -> np.ndarray:
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return self.robot_targets[-1]

        interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        t_start = float(self.robot_key_poses_time[interval - 1]) if interval > 0 else 0.0
        t_end = float(self.robot_key_poses_time[interval])
        alpha = float(np.clip((self.sim_time - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        target_cur = self.robot_targets[interval]
        target_prev = self.robot_targets[interval - 1] if interval > 0 else target_cur
        return (1.0 - alpha) * target_prev + alpha * target_cur

    def _select_grasp_particles(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        grasp_probe = np.array([0.31, -0.60, 0.22], dtype=np.float32)
        particle_q = particle_q[self.cloth_particle_start : self.cloth_particle_end]
        distance = np.linalg.norm(particle_q - grasp_probe, axis=1)
        local_indices = np.where(distance < 0.03)[0]
        if len(local_indices) == 0:
            local_indices = np.argsort(distance)[:8]

        self.grasp_particle_indices = self.cloth_particle_start + local_indices
        grasp_q = self.state_0.particle_q.numpy()[self.grasp_particle_indices]
        self.initial_grasp_max_height = float(np.max(grasp_q[:, 2]))

    def set_joint_targets(self) -> None:
        target = self._interpolated_target()
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
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        proxy_contacts = self.solver.get_proxy_contacts("robot", "vbd")
        if proxy_contacts is not None:
            self.viewer.log_contacts(proxy_contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"
        assert np.isfinite(body_q).all(), "Body transforms contain NaN or inf values"

        cloth_q = particle_q[self.cloth_particle_start : self.cloth_particle_end]
        min_pos = np.min(cloth_q, axis=0)
        max_pos = np.max(cloth_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 3.0, f"Shirt bounding box exploded: size={bbox_size:.3f}"
        assert min_pos[2] > -0.05, f"Shirt penetrated too far below ground: z_min={min_pos[2]:.4f}"

        grasp_q = particle_q[self.grasp_particle_indices]
        grasp_lift = float(np.max(grasp_q[:, 2]) - self.initial_grasp_max_height)
        assert grasp_lift > 0.04, f"Tracked grasp region did not lift enough: lift={grasp_lift:.4f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--coupling-mode",
            help="Proxy body state transfer mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--vbd-iterations",
            help="Number of VBD iterations per substep",
            type=int,
            default=5,
        )
        parser.add_argument(
            "--substeps",
            help="Simulation substeps per rendered frame",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--mass-scale",
            help="Scale factor for the hand proxy mass/inertia on the VBD side",
            type=float,
            default=0.25,
        )
        parser.set_defaults(num_frames=780)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)