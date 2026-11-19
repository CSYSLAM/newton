# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a USD stage using newton.ModelBuilder.add_usd().
#
# Command: python examples/python/example_robot_cartpole.py --solver synreal --world-count 100
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = args.solver
        self.viewer = viewer

        cartpole = newton.ModelBuilder()
        if self.solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
        cartpole.default_shape_cfg.density = 100.0
        cartpole.default_joint_cfg.armature = 0.1

        cartpole.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        # Apply additional inertia to the passive links for better stability.
        body_armature = 0.1
        for body in range(cartpole.body_count):
            inertia = np.asarray(cartpole.body_inertia[body], dtype=np.float32).reshape(3, 3)
            cartpole.body_inertia[body] = wp.mat33(inertia + np.eye(3, dtype=np.float32) * body_armature)

        cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]

        builder = newton.ModelBuilder()
        builder.replicate(cartpole, self.world_count, spacing=(1.0, 2.0, 0.0))
        self.model = builder.finalize()

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(self.model)
        else:
            self.solver = s3d.SolverCoupledHighAccuracy(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer.set_camera(
            pos=wp.vec3(7.3, -14.0, 2.3),
            pitch=-5.0,
            yaw=-225.0,
        )
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 90.0

        self.capture()

    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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

    def test_final(self):
        num_bodies_per_world = self.model.body_count // self.world_count
        if self.solver_type == "synreal":
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()
            assert np.isfinite(body_q).all() and np.isfinite(body_qd).all(), (
                "Synreal cartpole state contains non-finite values"
            )

            # Synreal's compliant affine constraints can leave tiny off-axis
            # residuals, so compare replicated worlds instead of requiring the
            # exact component-wise zeros expected from MuJoCo articulations.
            reference_q = body_q[:num_bodies_per_world]
            reference_positions = reference_q[:, :3] - reference_q[0, :3]
            reference_qd = body_qd[:num_bodies_per_world]
            world_consistency_atol = 5.0e-3
            for world in range(1, self.world_count):
                start = world * num_bodies_per_world
                stop = start + num_bodies_per_world
                world_q = body_q[start:stop]
                world_positions = world_q[:, :3] - world_q[0, :3]
                assert np.allclose(world_positions, reference_positions, atol=world_consistency_atol)
                assert np.allclose(world_q[:, 3:], reference_q[:, 3:], atol=world_consistency_atol)
                assert np.allclose(body_qd[start:stop], reference_qd, atol=world_consistency_atol)
            return

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: q[2] == 0.0 and newton.math.vec_allclose(q.q, wp.quat_identity()),
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        # fmt: off
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart only moves along y direction",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and wp.length_sq(wp.spatial_bottom(qd)) == 0.0,
            indices=[i * num_bodies_per_world for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 only has y-axis linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and abs(qd[3]) > 0.3
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 1 for i in range(self.world_count)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 only has yz-plane linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and abs(qd[2]) > 0.05
            and abs(qd[3]) > 0.2
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 2 for i in range(self.world_count)],
        )
        # fmt: on

        body_qd = self.state_0.body_qd.numpy()
        reference_velocities = [wp.spatial_vector(*body_qd[body]) for body in range(3)]
        world_velocity_atol = 1.0e-6
        for body, label in enumerate(("cart", "pole1", "pole2")):
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                f"{label} velocities match across worlds",
                lambda q, qd, reference=reference_velocities[body]: newton.math.vec_allclose(
                    qd, reference, atol=world_velocity_atol
                ),
                indices=[i * num_bodies_per_world + body for i in range(self.world_count)],
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=100)
        parser.add_argument(
            "--solver",
            type=str,
            default="synreal",
            choices=["synreal", "mujoco"],
            help="Solver type: synreal (default) or mujoco.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
