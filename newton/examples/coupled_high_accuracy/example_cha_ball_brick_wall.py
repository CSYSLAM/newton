"""Launch a heavy sphere into a freestanding wall of individual rigid bricks.

Run: uv run python -m newton.examples cha_ball_brick_wall
Compare: uv run python -m newton.examples cha_ball_brick_wall --solver xpbd
Larger wall: --columns 40 --rows 24
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def observe_ball_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    ball: int,
    observed: wp.array[int],
):
    index = wp.tid()
    if index < count[0]:
        a, b = shape_body[shape0[index]], shape_body[shape1[index]]
        if (a == ball and b >= 0 and b != ball) or (b == ball and a >= 0 and a != ball):
            wp.atomic_add(observed, 0, 1)


class Example:
    def __init__(self, viewer, args):
        for name in ("columns", "rows", "substeps"):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ball_speed", "ball_mass", "ball_radius"):
            value = getattr(args, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # An even count keeps input/output buffer identities unchanged across CUDA graph replays.
        self.sim_substeps = args.substeps + args.substeps % 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.graph = None
        self.args = args

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.ke = 2.0e5
        builder.default_shape_cfg.kd = 1.0e3
        builder.default_shape_cfg.mu = 0.65
        builder.default_shape_cfg.restitution = 0.05
        builder.default_shape_cfg.density = 600.0
        builder.add_ground_plane(color=(0.16, 0.19, 0.23))
        width, depth, height = 0.40, 0.28, 0.20
        gap = 0.002
        wall_width = args.columns * (width + gap)
        wall_height = args.rows * (height + gap)
        for row in range(args.rows):
            # Alternating half bricks keep straight wall ends while staggering the vertical seams.
            lengths = (
                [width] * args.columns if row % 2 == 0 else [width / 2] + [width] * (args.columns - 1) + [width / 2]
            )
            x = -0.5 * (sum(lengths) + gap * (len(lengths) - 1))
            for column, length in enumerate(lengths):
                position = wp.vec3(x + length / 2, 0, height / 2 + row * (height + gap) + gap)
                body = builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=f"brick_{row}_{column}")
                shade = ((row * 13 + column * 7) % 11) / 10.0
                builder.add_shape_box(
                    body,
                    hx=length / 2,
                    hy=depth / 2,
                    hz=height / 2,
                    color=(0.48 + 0.16 * shade, 0.17 + 0.08 * shade, 0.08 + 0.05 * shade),
                )
                x += length + gap
        self.brick_count = builder.body_count

        distance = max(2.0, 4.0 * args.ball_radius)
        target_height = max(args.ball_radius + 0.02, wall_height * 0.55)
        # Compensate gravity so the sphere reaches the wall near the requested initial impact height.
        travel = (distance - depth / 2 - args.ball_radius) / args.ball_speed
        self.impact_time = travel
        self.ball = builder.add_body(
            xform=wp.transform(wp.vec3(0, -distance, target_height), wp.quat_identity()),
            label="projectile",
        )
        ball_cfg = builder.default_shape_cfg.copy()
        ball_cfg.density = args.ball_mass / (4.0 / 3.0 * math.pi * args.ball_radius**3)
        builder.add_shape_sphere(self.ball, radius=args.ball_radius, cfg=ball_cfg, color=(0.08, 0.38, 0.8))
        builder.body_qd[self.ball] = wp.spatial_vector(0, args.ball_speed, 0.5 * 9.81 * travel, 0, 0, 0)

        self.model = builder.finalize()
        if args.solver == "synreal":
            from newton.solvers import coupled_high_accuracy as cha

            self.solver = cha.SolverCoupledHighAccuracy(self.model)
            self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
            self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
            self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
            self.solver.config.nonlinear_iterations = 5
            self.solver.config.linear_iterations = 50
        else:
            self.solver = newton.solvers.SolverXPBD(self.model, iterations=20)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.initial_bricks = self.state_0.body_q.numpy()[: self.brick_count, :3].copy()
        self.control = self.model.control()
        # Sweep-and-prune avoids testing every pair as the user increases the wall dimensions.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=max(4096, self.brick_count * 32),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.ball_contacts = wp.zeros(1, dtype=int, device=self.model.device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(wall_width * 0.8 + 2, -wall_width * 1.1 - 2, wall_height + 2), -18, 125)
        print(f"Brick wall: {self.brick_count} bricks, solver={args.solver}, substeps={self.sim_substeps}", flush=True)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            wp.launch(
                observe_ball_contacts,
                dim=self.contacts.rigid_contact_max,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contacts.rigid_contact_shape0,
                    self.contacts.rigid_contact_shape1,
                    self.model.shape_body,
                    self.ball,
                    self.ball_contacts,
                ],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        elif self.model.device.is_cuda:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        poses, velocities = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise AssertionError("Nonfinite rigid body state")
        if poses[:, 2].min() < -0.1:
            raise AssertionError("A rigid body fell through the ground")
        if self.sim_time > self.impact_time + 0.25:
            if self.ball_contacts.numpy()[0] == 0:
                raise AssertionError("The ball did not contact the wall")
            displacement = np.linalg.norm(poses[: self.brick_count, :3] - self.initial_bricks, axis=1)
            if displacement.max() < 0.05:
                raise AssertionError("No brick moved significantly after impact")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=360)
        parser.add_argument("--solver", choices=("synreal", "xpbd"), default="synreal")
        parser.add_argument("--columns", type=int, default=20, help="Full brick widths per row.")
        parser.add_argument("--rows", type=int, default=12, help="Number of brick courses.")
        parser.add_argument("--ball-speed", type=float, default=16.0, help="Launch speed toward the wall (m/s).")
        parser.add_argument("--ball-mass", type=float, default=200.0, help="Projectile mass (kg).")
        parser.add_argument("--ball-radius", type=float, default=0.32, help="Projectile radius (m).")
        parser.add_argument(
            "--substeps", type=int, default=4, help="Physics steps per frame; rounded up to an even count."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
