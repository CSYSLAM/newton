# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sim Cloth Hanging
#
# This simulation demonstrates a simple cloth hanging behavior. A planar cloth
# mesh is fixed on one side and hangs under gravity, colliding with the ground.
#
# Command: python -m newton.examples cloth_hanging (--solver [semi_implicit, style3d, xpbd, vbd])
#
###########################################################################


import warp as wp

import newton
import newton.examples
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        self.args = args

        # setup simulation parameters first
        self.solver_type = args.solver

        self.sim_height = args.height
        self.sim_width = args.width
        self.sim_time = 0.0

        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        if self.solver_type == "semi_implicit":
            self.sim_substeps = 32
        elif self.solver_type in ("style3d", "synreal", "reconstructed"):
            self.sim_substeps = 2
        else:
            self.sim_substeps = 10

        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        if self.solver_type in ("style3d", "synreal", "reconstructed"):
            builder = newton.ModelBuilder()
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        else:
            builder = newton.ModelBuilder()

        if self.solver_type == "semi_implicit":
            ground_cfg = builder.default_shape_cfg.copy()
            ground_cfg.ke = 1.0e2
            ground_cfg.kd = 5.0e1
            builder.add_ground_plane(cfg=ground_cfg)
        else:
            builder.add_ground_plane()

        # common cloth properties
        common_params = {
            "pos": wp.vec3(0.0, 0.0, 4.0),
            "rot": wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            "vel": wp.vec3(0.0, 0.0, 0.0),
            "dim_x": self.sim_width,
            "dim_y": self.sim_height,
            "cell_x": 0.1,
            "cell_y": 0.1,
            "mass": 0.1,
            "fix_left": True,
            "edge_ke": 1.0e1,
            "edge_kd": 0.0,
            "particle_radius": 0.05,
        }

        solver_params = {}
        if self.solver_type == "semi_implicit":
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e1,
            }

        elif self.solver_type in ("style3d", "synreal", "reconstructed"):
            common_params.pop("edge_ke")
            solver_params = {
                "tri_aniso_ke": wp.vec3(1.0e4, 1.0e4, 1.0e3),
                "edge_aniso_ke": wp.vec3(2.0e-6, 1.0e-6, 5.0e-6),
            }

        elif self.solver_type == "xpbd":
            solver_params = {
                "add_springs": True,
                "spring_ke": 1.0e3,
                "spring_kd": 1.0e0,
            }

        else:  # self.solver_type == "vbd"
            solver_params = {
                "tri_ke": 1.0e3,
                "tri_ka": 1.0e3,
                "tri_kd": 1.0e2,
            }

        if self.solver_type in ("style3d", "synreal", "reconstructed"):
            style3d.add_cloth_grid(builder, **common_params, **solver_params)
        else:
            builder.add_cloth_grid(**common_params, **solver_params)

        if self.solver_type == "vbd":
            builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e2 if self.solver_type in ("style3d", "vbd") else 1.0e0
        self.model.soft_contact_mu = 1.0

        if self.solver_type == "semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(model=self.model)
        elif self.solver_type == "reconstructed":
            from reconstruction.solver import ReconstructedSolver

            self.solver = ReconstructedSolver(self.model)
        elif self.solver_type == "synreal":
            from newton.solvers import coupled_high_accuracy as cha

            self.solver = cha.SolverCoupledHighAccuracy(self.model)
            self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
            self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
            self.solver.config.joint_constraint_primal_iterations = 1
            self.solver.config.nonlinear_iterations = 5
            self.solver.config.linear_iterations = 50
        elif self.solver_type == "style3d":
            self.solver = newton.solvers.SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
            )
        elif self.solver_type == "xpbd":
            self.solver = newton.solvers.SolverXPBD(
                model=self.model,
                iterations=self.iterations,
            )
        else:  # self.solver_type == "vbd"
            self.solver = newton.solvers.SolverVBD(
                model=self.model,
                iterations=self.iterations,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.02,
                particle_self_contact_margin=0.03,
            )

        if self.solver_type in ("synreal", "reconstructed") and getattr(args, "self_collision", False):
            # Folding on the ground brings layers together, which exercises the self-collision module.
            self.solver.config.enable_self_collision = True
            self.solver.config.self_collision_radius = args.self_collision_radius

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        self.stream = wp.get_stream(self.model.device)
        self.capture()

    def capture(self):
        if (
            wp.get_device().is_cuda
            and self.solver_type != "reconstructed"
            and not getattr(self.args, "no_capture", False)
        ):
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        if self.solver_type not in ("style3d", "synreal", "reconstructed"):
            newton.examples.test_particle_state(
                self.state_0,
                "particles are above the ground",
                lambda q, qd: q[2] > 0.0,
            )

        min_x = -float(self.sim_width) * 0.11
        p_lower = wp.vec3(min_x, -4.0, -1.8)
        p_upper = wp.vec3(0.1, 7.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--solver",
            help="Type of solver",
            type=str,
            choices=["semi_implicit", "style3d", "xpbd", "vbd", "synreal", "reconstructed"],
            default="synreal",
        )
        parser.add_argument("--no-capture", action="store_true", help="Run direct steps for numerical comparisons.")
        parser.add_argument("--self-collision", action="store_true", help="Enable native cloth self-collision.")
        parser.add_argument(
            "--self-collision-radius",
            type=float,
            default=0.01,
            help="Self-collision radius [m] when --self-collision is set.",
        )
        parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
        parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
