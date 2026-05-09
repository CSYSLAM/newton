# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Softbody Gift Pickup with Robot Arm (XPBD Version)
#
# A Franka Panda robot picks up a soft body gift from a table,
# lifts it to mid-air, and releases it for free fall.
# Uses XPBD solver for soft body and cloth.
#
# Gift structure: 4 stacked soft body cubes with 2 cloth ribbons
# Note: XPBD does not support closed cloth loops (non-manifold topology),
# so we use long cloth strips instead of wrapped ribbons.
#
# Command: uv run -m newton.examples softbody_gift_pickup_xpbd
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverXPBD


# =============================================================================
# Geometry Helpers (from example_softbody_gift.py)
# =============================================================================


def cloth_loop_around_box(
    hx=1.6,
    hz=2.0,
    width=0.25,
    center_y=0.0,
    nu=120,
    nv=6,
):
    """
    Vertical closed cloth loop wrapped around a cuboid.
    Loop lies in X-Z plane, strap width is along Y.
    Z is up.
    """
    verts = []
    faces = []

    P = 4.0 * (hx + hz)

    for i in range(nu):
        s = (i / nu) * P

        if s < 2 * hx:
            x = -hx + s
            z = -hz
        elif s < 2 * hx + 2 * hz:
            x = hx
            z = -hz + (s - 2 * hx)
        elif s < 4 * hx + 2 * hz:
            x = hx - (s - (2 * hx + 2 * hz))
            z = hz
        else:
            x = -hx
            z = hz - (s - (4 * hx + 2 * hz))

        for j in range(nv):
            v = (j / (nv - 1) - 0.5) * width
            y = center_y + v
            verts.append([x, y, z])

    def idx(i, j):
        return (i % nu) * nv + j

    for i in range(nu):
        for j in range(nv - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i, j + 1)])
            faces.append([idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])

    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


PYRAMID_TET_INDICES = np.array(
    [
        [0, 1, 3, 9],
        [1, 4, 3, 13],
        [1, 3, 9, 13],
        [3, 9, 13, 12],
        [1, 9, 10, 13],
        [1, 2, 4, 10],
        [2, 5, 4, 14],
        [2, 4, 10, 14],
        [4, 10, 14, 13],
        [2, 10, 11, 14],
        [3, 4, 6, 12],
        [4, 7, 6, 16],
        [4, 6, 12, 16],
        [6, 12, 16, 15],
        [4, 12, 13, 16],
        [4, 5, 7, 13],
        [5, 8, 7, 17],
        [5, 7, 13, 17],
        [7, 13, 17, 16],
        [5, 13, 14, 17],
    ],
    dtype=np.int32,
)

PYRAMID_PARTICLES = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (2.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (2.0, 2.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (0.0, 2.0, 1.0),
    (1.0, 2.0, 1.0),
    (2.0, 2.0, 1.0),
]


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def compute_joint_qd(
    target_q: wp.array[float],
    current_q: wp.array[float],
    out_qd: wp.array[float],
    inv_frame_dt: float,
):
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt


class Example:
    def __init__(self, viewer, args=None):
        self.sim_substeps = 10
        self.iterations = 20
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.scene = ModelBuilder(gravity=-9.81)
        self.viewer = viewer

        # =========================================================================
        # Gift and table geometry (meter scale)
        # =========================================================================
        # Scale factor: original example uses ~2 unit cubes, we scale to meters
        gift_scale = 0.01  # 1cm per unit

        # Table: 0.3m x 0.3m x 0.05m
        table_hx = 0.15
        table_hy = 0.15
        table_hz = 0.025
        table_pos = wp.vec3(0.0, -0.5, table_hz)
        table_top_z = table_pos[2] + table_hz

        # Gift position
        gift_center_x = table_pos[0]
        gift_center_y = table_pos[1]

        # Add table
        self.scene.add_shape_box(
            -1, wp.transform(table_pos, wp.quat_identity()),
            hx=table_hx, hy=table_hy, hz=table_hz,
        )

        # Generate cloth geometry for straps
        strap1_verts, strap1_faces = cloth_loop_around_box(hx=1.01, hz=2.02, width=0.6)
        strap2_verts, strap2_faces = cloth_loop_around_box(hx=1.015, hz=2.025, width=0.6)

        # Stack spacing
        spacing = 1.01 * gift_scale
        drop_height = table_top_z + 0.005  # 5mm above table

        # Add 4 stacked soft body blocks
        for i in range(4):
            self.scene.add_soft_mesh(
                pos=(gift_center_x, gift_center_y, drop_height + i * spacing),
                rot=wp.quat_identity(),
                scale=gift_scale,
                vel=(0.0, 0.0, 0.0),
                vertices=PYRAMID_PARTICLES,
                indices=PYRAMID_TET_INDICES.flatten().tolist(),
                density=100,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=1e-5,
                particle_radius=0.002,
            )

        # Gift top z for IK targets
        # 4 blocks, each ~2 units tall, scaled by gift_scale
        gift_height = 4 * 2.0 * gift_scale
        gift_top_z = drop_height + gift_height

        # Add cloth ribbons using cloth_mesh with closed loops (XPBD with springs)
        # Two perpendicular straps wrapping around the gift
        strap_scale = gift_scale

        # First strap - wraps around X direction
        strap1_verts, strap1_faces = cloth_loop_around_box(
            hx=1.01 * gift_scale / strap_scale,
            hz=2.02 * gift_scale / strap_scale,
            width=0.6,
            center_y=gift_center_y,
            nu=40,
            nv=3,
        )
        self.scene.add_cloth_mesh(
            pos=wp.vec3(gift_center_x, 0.0, drop_height),
            rot=wp.quat_identity(),
            scale=strap_scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=strap1_verts.tolist(),
            indices=strap1_faces.flatten().tolist(),
            density=0.05,
            tri_ke=1.0e2,
            tri_ka=1.0e1,
            tri_kd=1.0e0,
            edge_ke=1.0e0,
            edge_kd=1.0e-1,
            add_springs=True,
            spring_ke=1.0e3,
            spring_kd=1.0e1,
            particle_radius=0.001,
        )

        # Second strap - wraps around Y direction (perpendicular)
        strap2_verts, strap2_faces = cloth_loop_around_box(
            hx=1.015 * gift_scale / strap_scale,
            hz=2.025 * gift_scale / strap_scale,
            width=0.6,
            center_y=gift_center_x,  # Use X as Y offset for perpendicular strap
            nu=40,
            nv=3,
        )
        # Rotate the second strap 90 degrees around Z to make it perpendicular
        rot_90 = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
        self.scene.add_cloth_mesh(
            pos=wp.vec3(0.0, gift_center_y, drop_height),
            rot=rot_90,
            scale=strap_scale,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=strap2_verts.tolist(),
            indices=strap2_faces.flatten().tolist(),
            density=0.05,
            tri_ke=1.0e2,
            tri_ka=1.0e1,
            tri_kd=1.0e0,
            edge_ke=1.0e0,
            edge_kd=1.0e-1,
            add_springs=True,
            spring_ke=1.0e3,
            spring_kd=1.0e1,
            particle_radius=0.001,
        )

        # =========================================================================
        # Add Franka Panda robot arm (scale=1.0, meter units)
        # =========================================================================
        franka = ModelBuilder()
        try:
            asset_path = newton.utils.download_asset("franka_emika_panda")
        except Exception:
            asset_path = "C:/csy_work/CG/Engine/newton/Cache/newton-assets_unitree_h1_4589a7d5/franka_emika_panda"

        # Robot base: positioned to reach the gift
        robot_base_x = -0.2
        robot_base_y = -0.5
        robot_base_z = 0.0

        franka.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((robot_base_x, robot_base_y, robot_base_z), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # Initial joint configuration - arm reaching forward
        franka.joint_q[:6] = [0.0, -0.3, 0.0, -1.5, 0.0, 1.8]

        # Store endeffector ID (link7 for Franka)
        self.endeffector_id = franka.body_count - 3
        self.scene.add_world(franka)

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Contact parameters for XPBD (lower values for cloth stability)
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 5.0e0
        self.model.soft_contact_mu = 1.0

        # XPBD solver - use larger relaxation for stability
        self.solver = SolverXPBD(
            model=self.model,
            iterations=self.iterations,
            soft_body_relaxation=1e-2,
            soft_contact_relaxation=1e-2,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model, soft_contact_margin=0.04
        )
        self.contacts = self.collision_pipeline.contacts()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.3, 0.3, 0.5), -25.0, -35.0)

        # =========================================================================
        # IK setup
        # =========================================================================
        gripper_open = 1.0
        gripper_close = 0.5

        # Keyframe sequence: approach, grab, lift, hold, release
        self.robot_key_poses = np.array([
            [2.0, gift_center_x, gift_center_y, gift_top_z + 0.08, 1, 0.0, 0.0, 0.0, gripper_open],
            [2.0, gift_center_x, gift_center_y, gift_top_z + 0.01, 1, 0.0, 0.0, 0.0, gripper_open],
            [1.0, gift_center_x, gift_center_y, gift_top_z + 0.01, 1, 0.0, 0.0, 0.0, gripper_close],
            [2.0, gift_center_x, gift_center_y, gift_top_z + 0.15, 1, 0.0, 0.0, 0.0, gripper_close],
            [3.0, gift_center_x, gift_center_y, gift_top_z + 0.15, 1, 0.0, 0.0, 0.0, gripper_close],
            [1.0, gift_center_x, gift_center_y, gift_top_z + 0.15, 1, 0.0, 0.0, 0.0, gripper_open],
            [2.0, gift_center_x, gift_center_y, gift_top_z + 0.10, 1, 0.0, 0.0, 0.0, gripper_open],
        ], dtype=np.float32)

        self.targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        # IK solver
        state = self.model.state()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

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
            model=self.model, n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1, jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def update_ik_targets(self):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return
        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        alpha = float(np.clip((self.sim_time - t_start) / (t_end - t_start), 0.0, 1.0))
        target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else target_cur
        target_interp = (1.0 - alpha) * target_prev + alpha * target_cur
        self.pos_obj.set_target_position(0, wp.vec3(*target_interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_interp[3:7].tolist()))
        finger_pos = float(target_interp[-1]) * 0.04
        self.finger_pos_buf.fill_(finger_pos)

    def step(self):
        self.update_ik_targets()
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.simulate()

    def simulate(self):
        # IK solve
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(set_gripper_q, dim=1, inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1])
        wp.copy(self.target_joint_q, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)
        wp.launch(compute_joint_qd, dim=self.n_dofs,
                  inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt])

        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            # Robot step
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0

            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # Collision and soft body step
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.5, -1.0, -0.05)
        p_upper = wp.vec3(0.5, 0.0, 0.5)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1000)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
