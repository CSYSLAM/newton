# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka FR3 VBD Pick and Place
#
# A fixed-base FR3 grasps a rigid cube using its URDF finger colliders.
# IK generates joint targets only; SolverVBD integrates the entire scene.
# The cube is held by contact friction, without attachment constraints.
#
# Command: python -m newton.examples robot_franka_vbd
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.table_height = 0.2
        self.cube_half_extent = 0.02
        self.pick_position = np.array([0.45, 0.0, self.table_height + self.cube_half_extent])
        self.place_position = self.pick_position + np.array([0.0, 0.16, 0.0])
        self.max_cube_height = float(self.pick_position[2])
        self.checked_hold = False

        builder = newton.ModelBuilder()
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=100.0, mu=1.0)
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
        # VBD does not enforce URDF mimic constraints: drive both fingers explicitly.
        builder.constraint_mimic_enabled[:] = [False] * len(builder.constraint_mimic_enabled)
        builder.joint_q[:] = [0.0, -0.4, 0.0, -1.6, 0.0, 1.2, 0.7, 0.04, 0.04]
        builder.joint_target_ke[:] = [5000.0] * 7 + [1000.0] * 2
        builder.joint_target_kd[:] = [100.0] * 7 + [10.0] * 2
        self.arm_coord_count = builder.joint_coord_count
        # Fixed-joint collapse removes massless URDF frames, which would otherwise
        # act as static anchors in a maximal-coordinate solver. The hand TCP is
        # 0.107 + 0.1034 m along link7 Z, with a -pi/4 rotation about that axis.
        self.tool_body = next(i for i, label in enumerate(builder.body_label) if label.endswith("/fr3_link7"))
        self._build_trajectory(builder.finalize())
        builder.joint_q[:] = self.joint_waypoints[0].tolist()
        builder.joint_target_q[:] = builder.joint_q

        builder.add_ground_plane()
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.5, 0.08, self.table_height / 2), wp.quat_identity()),
            hx=0.25,
            hy=0.3,
            hz=self.table_height / 2,
            color=(0.35, 0.42, 0.5),
            label="table",
        )
        self.cube_body = builder.add_body(
            xform=wp.transform(wp.vec3(*self.pick_position), wp.quat_identity()), label="cube"
        )
        builder.add_shape_box(
            self.cube_body,
            hx=self.cube_half_extent,
            hy=self.cube_half_extent,
            hz=self.cube_half_extent,
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, kd=100.0, mu=1.0),
            color=(1.0, 0.35, 0.06),
        )
        builder.color()
        self.model = builder.finalize()
        # VBD captures structural rest poses at construction.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=30,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e7,
            rigid_joint_angular_ke=1.0e7,
            rigid_body_contact_buffer_size=128,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.target_q = self.control.joint_target_q.numpy()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.35, -1.4, 1.1), pitch=-25.0, yaw=130.0)

        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _build_trajectory(self, model):
        """Generate joint waypoints with IK before running any dynamics."""
        pick = self.pick_position
        place = self.place_position
        above = np.array([0.0, 0.0, 0.18])
        # Time [s], TCP position [m], and each finger's opening [m].
        keyframes = [
            (0.0, pick + above, 0.04),
            (1.0, pick + above, 0.04),
            (2.5, pick, 0.04),
            (3.5, pick, 0.015),
            (4.0, pick, 0.015),
            (5.5, pick + above, 0.015),
            (6.5, pick + above, 0.015),
            (8.0, place + above, 0.015),
            (9.5, place, 0.015),
            (10.5, place, 0.04),
            (12.0, place + above, 0.04),
            (13.0, place + above, 0.04),
        ]
        position = ik.IKObjectivePosition(
            link_index=self.tool_body,
            link_offset=wp.vec3(0.0, 0.0, 0.2104),
            target_positions=wp.array([wp.vec3(*keyframes[0][1])], dtype=wp.vec3),
        )
        rotation = ik.IKObjectiveRotation(
            link_index=self.tool_body,
            link_offset_rotation=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi / 4),
            target_rotations=wp.array([wp.vec4(1.0, 0.0, 0.0, 0.0)], dtype=wp.vec4),
        )
        solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[position, rotation, ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper)],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        q = wp.clone(model.joint_q).reshape((1, self.arm_coord_count))
        self.key_times = np.array([frame[0] for frame in keyframes])
        self.joint_waypoints = []
        for _, target, opening in keyframes:
            position.set_target_position(0, wp.vec3(*target))
            solver.step(q, q, iterations=100)
            joints = q.numpy()[0].copy()
            joints[-2:] = opening
            self.joint_waypoints.append(joints)
        self.joint_waypoints = np.array(self.joint_waypoints)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

    def step(self):
        t = min(self.sim_time, self.key_times[-1])
        index = min(int(np.searchsorted(self.key_times, t, side="right")), len(self.key_times) - 1)
        alpha = np.clip((t - self.key_times[index - 1]) / (self.key_times[index] - self.key_times[index - 1]), 0, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        self.target_q[: self.arm_coord_count] = (1.0 - alpha) * self.joint_waypoints[
            index - 1
        ] + alpha * self.joint_waypoints[index]
        self.control.joint_target_q.assign(self.target_q)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify finite states and sustained cube elevation during the hold."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite rigid body state"
        cube = poses[self.cube_body, :3]
        self.max_cube_height = max(self.max_cube_height, float(cube[2]))
        if 5.8 <= self.sim_time <= 6.4:
            assert cube[2] > self.pick_position[2] + 0.12, f"Cube was not held above the table: {cube}"
            self.checked_hold = True

    def test_final(self):
        """Verify the cube was lifted, transported, and released onto the table."""
        assert self.sim_time >= 12.5, "Run at least 750 frames to test the full pick-and-place sequence"
        assert self.checked_hold, "The elevated hold phase was not verified"
        cube = self.state_0.body_q.numpy()[self.cube_body, :3]
        assert np.linalg.norm(cube - self.place_position) < 0.035, f"Cube missed the placement target: {cube}"
        hand = self.state_0.body_q.numpy()[self.tool_body]
        tcp = wp.transform_point(wp.transform(*hand), wp.vec3(0.0, 0.0, 0.2104))
        assert tcp[2] > cube[2] + 0.12, "Gripper did not retract after release"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=780)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
