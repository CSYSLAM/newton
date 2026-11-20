# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Spinning Gyroscope
#
# Demonstrates a gyroscope spinning upright under gravity.
#
# Command: python demos/demo/demo_spin.py
# XPBD: python demos/demo/demo_spin.py --solver xpbd
# MuJoCo: python demos/demo/demo_spin.py --solver mujoco
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as cha

DEFAULT_GYROSCOPE_ASSET = Path(__file__).resolve().parents[2] / "assets/spin/Gyroscope.usd"


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "vbd"

        builder_type = cha.ModelBuilder if self.solver_type == "synreal" else newton.ModelBuilder
        builder = builder_type(up_axis=newton.Axis.Z)
        # Both contact shapes must be frictionless because solvers combine their coefficients.
        builder.default_shape_cfg.mu = 0.0
        builder.default_shape_cfg.mu_torsional = 0.0
        builder.default_shape_cfg.mu_rolling = 0.0

        self.body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            label="gyroscope",
        )
        self.free_joint = len(builder.joint_type) - 1

        gyroscope_path = Path(getattr(args, "gyroscope_asset", DEFAULT_GYROSCOPE_ASSET)).expanduser().resolve()
        if not gyroscope_path.is_file():
            raise FileNotFoundError(
                f"Gyroscope mesh is missing: {gyroscope_path}. Restore bundled assets or use --gyroscope-asset."
            )
        gyroscope_mesh = newton.Mesh.create_from_usd(gyroscope_path, root_path="/root/Gyroscope")
        cfg_gyroscope = builder.default_shape_cfg.copy()
        cfg_gyroscope.has_shape_collision = True
        # The asset uses real-world dimensions; scale it to match the meter-scale demo.
        gyroscope_scale = wp.vec3(30.0, 30.0, 30.0)
        builder.add_shape_mesh(
            self.body,
            mesh=gyroscope_mesh,
            scale=gyroscope_scale,
            cfg=cfg_gyroscope,
            label="gyroscope",
        )

        builder.body_qd[self.body] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
        builder.joint_qd = np.array(builder.body_qd).flatten().tolist()

        if self.solver_type == "vbd":
            builder.color()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=4)
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        elif self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=False,
                solver="newton",
                integrator="implicitfast",
                iterations=10,
            )
            self.contacts = None
        elif self.solver_type == "synreal":
            self.solver = cha.SolverCoupledHighAccuracy(model=self.model)
            self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.CO_ROTATED
            self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
            self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
            self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
            self.solver.config.joint_constraint_primal_iterations = 4
            self.solver.config.nonlinear_iterations = 30
            self.solver.config.linear_iterations = 10
        else:
            raise ValueError(
                f"Unknown solver type: {self.solver_type}. Choose from 'vbd', 'xpbd', 'mujoco', or 'synreal'."
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        if self.solver_type == "mujoco":
            self.collision_pipeline = None
            self.contacts = None
        else:
            # This scene has rigid shapes only; affine proxy particles are not extra collision samples.
            self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=0)
            self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q[self.body]).all() or not np.isfinite(body_qd[self.body]).all():
            raise ValueError("Gyroscope state contains non-finite values")
        if body_q[self.body][2] < -0.05:
            raise ValueError("Gyroscope fell through the ground plane")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--gyroscope-asset",
        type=Path,
        default=DEFAULT_GYROSCOPE_ASSET,
        help="USD mesh with a /root/Gyroscope prim; defaults to the bundled procedural demonstration asset.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="synreal",
        choices=["vbd", "xpbd", "mujoco", "synreal"],
        help="Solver type: synreal (default), vbd, xpbd, or mujoco.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
