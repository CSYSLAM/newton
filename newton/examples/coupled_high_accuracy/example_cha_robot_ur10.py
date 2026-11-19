# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot UR10
#
# Shows how to set up a simulation of a UR10 robot arm
# from a USD file using newton.ModelBuilder.add_usd() and
# applies a bounded, smooth joint scan with a different joint order per world.
#
# Command: python -m newton.examples cha_robot_ur10 --solver synreal --world-count 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.selection import ArticulationView
from newton.solvers import coupled_high_accuracy as s3d


def joint_scan_trajectory(initial, lower, upper, sample_rate=50.0, amplitude_degrees=30.0, speed_degrees=30.0):
    """Move one joint per world at a time, starting and finishing each excursion at rest."""
    initial = np.asarray(initial, dtype=np.float64)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if initial.ndim != 2 or 0 in initial.shape or not np.isfinite(initial).all():
        raise ValueError("Initial joint positions must be a finite world-by-joint array")
    if (
        lower.shape != initial.shape[1:]
        or upper.shape != lower.shape
        or np.any(lower >= upper)
        or np.isnan(lower).any()
        or np.isnan(upper).any()
    ):
        raise ValueError("Joint limit vectors must match the joint count and have positive ranges")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("The initial pose is outside the joint limits")
    if not all(np.isfinite(v) and v > 0 for v in (sample_rate, amplitude_degrees, speed_degrees)):
        raise ValueError("Trajectory sample rate, amplitude, and speed must be finite and positive")
    amplitude = np.minimum(np.deg2rad(amplitude_degrees), np.minimum(initial - lower, upper - initial))
    # sin(phase)^3 has zero velocity and acceleration at both ends. Its maximum
    # slope is 2/sqrt(3), giving this duration bound for the requested angular speed.
    duration = (4 * np.pi / np.sqrt(3)) * np.deg2rad(amplitude_degrees) / np.deg2rad(speed_degrees)
    steps = max(2, int(np.ceil(duration * sample_rate)))
    wave = np.sin(2 * np.pi * np.arange(steps) / steps) ** 3
    worlds, joints = initial.shape
    trajectory = np.broadcast_to(initial, (steps * joints + 1, worlds, joints)).copy()
    for stage in range(joints):
        for world in range(worlds):
            # Reorder motion stages, not joint channels: every target retains its own initial pose and limits.
            joint = (stage + world) % joints
            trajectory[stage * steps : (stage + 1) * steps, world, joint] += amplitude[world, joint] * wave
    return trajectory.astype(np.float32)


@wp.kernel
def update_joint_target_trajectory_kernel(
    joint_target_trajectory: wp.array3d[wp.float32],
    time: wp.array[wp.float32],
    dt: wp.float32,
    sample_rate: wp.float32,
    # output
    joint_target: wp.array3d[wp.float32],
    joint_target_velocity: wp.array3d[wp.float32],
):
    world_idx = wp.tid()
    t = time[world_idx]
    t = wp.mod(t + dt, float(joint_target_trajectory.shape[0] - 1))
    step = int(t)
    time[world_idx] = t

    num_dofs = joint_target.shape[2]
    for dof in range(num_dofs):
        joint_target[world_idx, 0, dof] = wp.lerp(
            joint_target_trajectory[step, world_idx, dof],
            joint_target_trajectory[step + 1, world_idx, dof],
            wp.frac(t),
        )
        joint_target_velocity[world_idx, 0, dof] = sample_rate * (
            joint_target_trajectory[step + 1, world_idx, dof] - joint_target_trajectory[step, world_idx, dof]
        )


class Example:
    def __init__(self, viewer, args):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = args.solver

        self.viewer = viewer

        self.device = wp.get_device()

        ur10 = newton.ModelBuilder()
        if self.solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        # create a pedestal
        ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0, 0, height / 2)), half_height=height / 2, radius=0.08)

        for i in range(len(ur10.joint_target_ke)):
            # Resist gravity without relying on a large position error to generate holding torque.
            ur10.joint_target_ke[i] = 2000
            ur10.joint_target_kd[i] = 100
            ur10.joint_target_mode[i] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder()
        builder.replicate(ur10, self.world_count, spacing=(2, 2, 0))

        # Keep the starting pose clear of the pedestal instead of folding each arm through its full range.
        home = np.deg2rad([0, -90, 90, -90, -90, 0])
        if ur10.joint_coord_count != len(home) or ur10.joint_dof_count != len(home):
            raise ValueError("The UR10 joint scan requires six scalar joints")
        rng = np.random.default_rng(42)
        joint_q = home + rng.uniform(-np.deg2rad(5), np.deg2rad(5), (self.world_count, len(home)))
        builder.joint_q = joint_q.ravel().tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.model,
        )

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,
            )
        else:
            self.solver = s3d.SolverCoupledHighAccuracy(
                self.model,
                joint_drive_armature=0.1,
            )
            self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
            self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
            self.solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
            self.solver.config.inertia_mode = s3d.InertiaMode.MIDPOINT_CAYLEY
            self.solver.config.constraint_alpha = 200.0
            self.solver.config.joint_constraint_primal_iterations = 3
            self.solver.config.nonlinear_iterations = 5
            self.solver.config.linear_iterations = 50

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.articulation_view = ArticulationView(
            self.model, "*ur10*", exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE]
        )
        assert self.articulation_view.count == self.world_count, (
            "Number of worlds must match the number of articulations"
        )
        self.trajectory_sample_rate = float(self.fps)

        dof_lower = self.articulation_view.get_attribute("joint_limit_lower", self.model)[0, 0].numpy()
        dof_upper = self.articulation_view.get_attribute("joint_limit_upper", self.model)[0, 0].numpy()
        joint_q = self.articulation_view.get_attribute("joint_q", self.state_0).numpy().squeeze(axis=1)
        joint_target_trajectory = joint_scan_trajectory(
            joint_q,
            dof_lower,
            dof_upper,
            sample_rate=self.trajectory_sample_rate,
            amplitude_degrees=args.motion_amplitude,
            speed_degrees=args.max_joint_speed,
        )

        self.joint_target_trajectory = wp.array(joint_target_trajectory, dtype=wp.float32, device=self.device)
        self.time_step = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)

        self.ctrl = self.articulation_view.get_attribute("joint_target_q", self.control)
        self.ctrl_velocity = self.articulation_view.get_attribute("joint_target_qd", self.control)
        self.ctrl.assign(joint_q[:, None, :])
        self.articulation_view.set_attribute("joint_target_q", self.control, self.ctrl)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                update_joint_target_trajectory_kernel,
                dim=self.world_count,
                inputs=[
                    self.joint_target_trajectory,
                    self.time_step,
                    self.sim_dt * self.trajectory_sample_rate,
                    self.trajectory_sample_rate,
                ],
                outputs=[self.ctrl, self.ctrl_velocity],
                device=self.device,
            )
            self.articulation_view.set_attribute("joint_target_q", self.control, self.ctrl)
            self.articulation_view.set_attribute("joint_target_qd", self.control, self.ctrl_velocity)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            raise ValueError("UR10 body state contains non-finite values")
        if np.linalg.norm(body_qd[:, :3], axis=1).max() > 10.0:
            raise ValueError("UR10 body speed exceeds 10 m/s")
        if np.linalg.norm(body_qd[:, 3:], axis=1).max() > 30.0:
            raise ValueError("UR10 body angular speed exceeds 30 rad/s")

    def test_final(self):
        self.test_post_step()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=128)
        parser.add_argument("--motion-amplitude", type=float, default=30.0, help="Joint scan amplitude in degrees")
        parser.add_argument(
            "--max-joint-speed", type=float, default=30.0, help="Maximum target speed in degrees per second"
        )
        parser.add_argument(
            "--solver",
            type=str,
            default="synreal",
            choices=["mujoco", "synreal"],
            help="Solver type: synreal (default) or mujoco.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    try:
        newton.examples.run(Example(viewer, args), args)
    except BaseException:
        viewer.close()
        raise
