# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Dzhanibekov
#
# Demonstrates the Dzhanibekov effect on a free rigid body loaded from
# assets/dzhanibekov/Nut.usd.
#
# Command: python demos/demo/demo_dzhanibekov.py
# XPBD: python demos/demo/demo_dzhanibekov.py --solver xpbd
# MuJoCo: python demos/demo/demo_dzhanibekov.py --solver mujoco
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

DEFAULT_NUT_ASSET = Path(__file__).resolve().parents[2] / "assets/dzhanibekov/Nut.usd"


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver if hasattr(args, "solver") and args.solver else "vbd"
        self.min_spin_axis_alignment = 1.0

        builder_type = s3d.ModelBuilder if self.solver_type == "synreal" else newton.ModelBuilder
        builder = builder_type(up_axis=newton.Axis.Z)

        initial_orientation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)
        initial_rotation = np.array(wp.quat_to_matrix(initial_orientation), dtype=np.float32).reshape(3, 3)
        self.body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), initial_orientation),
            label="dzhanibekov_nut",
        )
        self.free_joint = len(builder.joint_type) - 1

        nut_path = Path(getattr(args, "nut_asset", DEFAULT_NUT_ASSET)).expanduser().resolve()
        if not nut_path.is_file():
            raise FileNotFoundError(f"Nut mesh is missing: {nut_path}. Restore the bundled assets or use --nut-asset.")
        nut_mesh = newton.Mesh.create_from_usd(nut_path, root_path="/root/nut")
        cfg_nut = builder.default_shape_cfg.copy()
        cfg_nut.has_shape_collision = False
        # The asset uses real-world dimensions; scale it to match the meter-scale demo.
        nut_scale = wp.vec3(30.0, 30.0, 30.0)
        builder.add_shape_mesh(
            self.body,
            mesh=nut_mesh,
            scale=nut_scale,
            cfg=cfg_nut,
            label="nut",
        )

        # Derive the spin direction from the asset so it remains correct if the mesh is
        # reoriented. eigh returns principal axes ordered by increasing moment.
        _, principal_axes = np.linalg.eigh(np.asarray(nut_mesh.inertia).reshape(3, 3))
        perturbation_axis = principal_axes[:, 0]
        self.local_spin_axis = principal_axes[:, 1].astype(np.float32)
        self.initial_spin_axis_world = initial_rotation @ self.local_spin_axis
        # Remove the mesh's tiny asymmetry-induced tilt to keep the initial rotation horizontal.
        self.initial_spin_axis_world[2] = 0.0
        self.initial_spin_axis_world /= np.linalg.norm(self.initial_spin_axis_world)

        perturbation_axis_world = initial_rotation @ perturbation_axis
        perturbation_axis_world[2] = 0.0
        perturbation_axis_world -= (
            np.dot(perturbation_axis_world, self.initial_spin_axis_world) * self.initial_spin_axis_world
        )
        perturbation_axis_world /= np.linalg.norm(perturbation_axis_world)
        initial_angular_speed = getattr(args, "init_w", 50.0)
        if not np.isfinite(initial_angular_speed) or initial_angular_speed < 0.0:
            raise ValueError("--init-w must be finite and non-negative")
        angular_velocity_world = perturbation_axis_world + initial_angular_speed * self.initial_spin_axis_world

        # Exact intermediate-axis spin is an unstable equilibrium. Use a fixed
        # perturbation so every rigidity model demonstrates the same instability.
        builder.body_qd[self.body] = wp.spatial_vector(0.0, 0.0, 0.0, *angular_velocity_world)
        builder.joint_qd = np.array(builder.body_qd).flatten().tolist()

        if self.solver_type == "vbd":
            builder.color()

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, 0.0))

        if self.solver_type == "vbd":
            self.solver = newton.solvers.SolverVBD(self.model, iterations=4)
            self.contacts = self.model.contacts()
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
            self.contacts = self.model.contacts()
        elif self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                disable_contacts=True,
                solver="newton",
                integrator="implicitfast",
                iterations=10,
            )
            self.contacts = None
        elif self.solver_type == "synreal":
            self.solver = s3d.SolverCoupledHighAccuracy(model=self.model)
            self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.CO_ROTATED
            self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
            self.solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
            self.solver.config.inertia_mode = s3d.InertiaMode.MIDPOINT_CAYLEY
            self.solver.config.joint_constraint_primal_iterations = 2
            self.solver.config.nonlinear_iterations = 20
            self.solver.config.linear_iterations = 10
            # The nut disables collision and the ground is visual context; this free-body run has no contacts.
            self.contacts = None
        else:
            raise ValueError(
                f"Unknown solver type: {self.solver_type}. Choose from 'vbd', 'xpbd', 'mujoco', or 'synreal'."
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.body_com = self.model.body_com.numpy()[self.body].copy()
        self.initial_com = (
            self.initial_body_q[self.body][:3]
            + np.array(wp.quat_to_matrix(wp.quat(*self.initial_body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)
            @ self.body_com
        )

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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        body_q = self.state_0.body_q.numpy()
        spin_axis_world = (
            np.array(wp.quat_to_matrix(wp.quat(*body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)
            @ self.local_spin_axis
        )
        self.min_spin_axis_alignment = min(
            self.min_spin_axis_alignment,
            float(np.dot(spin_axis_world, self.initial_spin_axis_world)),
        )

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q[self.body]).all() or not np.isfinite(body_qd[self.body]).all():
            raise ValueError("Dzhanibekov rigid body state contains non-finite values")
        final_com = (
            body_q[self.body][:3]
            + np.array(wp.quat_to_matrix(wp.quat(*body_q[self.body][3:7])), dtype=np.float32).reshape(3, 3)
            @ self.body_com
        )
        if np.linalg.norm(final_com - self.initial_com) >= 1.0e-3:
            raise ValueError("Free rigid body developed linear drift")
        if self.min_spin_axis_alignment >= 0.0:
            raise ValueError("Dzhanibekov flip did not occur")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--nut-asset",
        type=Path,
        default=DEFAULT_NUT_ASSET,
        help="USD nut mesh with a /root/nut prim; defaults to the bundled procedural demonstration asset.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="synreal",
        choices=["vbd", "xpbd", "mujoco", "synreal"],
        help="Solver type: synreal (default), vbd, xpbd, or mujoco.",
    )
    parser.add_argument(
        "--init-w",
        type=float,
        default=50.0,
        metavar="RAD_PER_SEC",
        help="Initial angular speed around the intermediate principal axis in rad/s.",
    )
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
