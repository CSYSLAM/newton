# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Push a cohesionless sand pile with a hinged paddle driven through MuJoCo.

    uv run --extra examples -m newton.examples mjmpm_push_sand
    uv run --extra examples -m newton.examples mjmpm_push_sand --articulated-response --substeps 4

A gravity-compensated gantry slides a paddle along x on a prismatic drive.
The paddle hangs from a sprung revolute hinge, so the sand load both slows
the gantry and swings the paddle back. The sand settles, is pushed, held
under load, and left behind when the paddle retracts; the pushed material
keeps its new shape because it flows plastically instead of springing back.
"""

from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJMPM

# The blade bottom stays two MPM cells above the ground. Sand trapped in a
# thinner gap between the jointed blade and the static ground gets
# contradictory collider constraints and is ejected.
GANTRY_HEIGHT = 0.19
# At least two MPM cells thick so grains cannot tunnel into the blade.
PADDLE_HALF = np.array((0.02, 0.15, 0.055))
HINGE_DROP = 0.04
SAND_SIZE = np.array((0.16, 0.14, 0.12))
SAND_SPACING = 0.01
SAND_DENSITY = 1600.0
START_X = -0.14
PUSH = 0.16
# (time [s], gantry travel [m]) keyframes: settle, push, hold, retract, rest.
SCHEDULE = ((0.0, 0.0), (1.2, 0.0), (3.2, PUSH), (4.0, PUSH), (5.0, 0.0), (5.8, 0.0))
SETTLED = (1.0, 1.2)
HOLD = (3.4, 4.0)
# The pushed sand decelerates into slow creep under the held drive force.
CREEP = (3.7, 4.0)
REST = (5.4, 5.8)


def smoothstep(time):
    """Gantry travel [m] at ``time`` [s] with quintic easing between keyframes."""
    for (t0, x0), (t1, x1) in pairwise(SCHEDULE):
        if time <= t1:
            s = float(np.clip((time - t0) / (t1 - t0), 0.0, 1.0))
            s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
            return x0 + (x1 - x0) * s
    return SCHEDULE[-1][1]


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 2 or args.substeps % 2:
            raise ValueError("Use an even substep count >= 2 so captured frames end in the same state buffer")
        self.viewer, self.args = viewer, args
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / args.substeps
        self.sim_time = 0.0
        self.frame = 0

        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        builder.add_ground_plane()
        rail = wp.transform(wp.vec3(START_X, 0.0, GANTRY_HEIGHT), wp.quat_identity())
        gantry = builder.add_link(
            xform=rail,
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="gantry",
        )
        builder.add_shape_box(
            gantry,
            hx=0.03,
            hy=0.03,
            hz=0.02,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
            color=(0.35, 0.38, 0.42),
        )
        slide = builder.add_joint_prismatic(
            parent=-1,
            child=gantry,
            parent_xform=rail,
            axis=(1.0, 0.0, 0.0),
            target_ke=args.drive_kp,
            target_kd=args.drive_kd,
            effort_limit=args.drive_effort,
        )
        paddle_center = GANTRY_HEIGHT - HINGE_DROP - PADDLE_HALF[2]
        paddle = builder.add_link(
            xform=wp.transform(wp.vec3(START_X, 0.0, paddle_center), wp.quat_identity()),
            mass=0.6,
            inertia=wp.mat33(np.diag((0.0025, 0.0007, 0.0024))),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="paddle",
        )
        builder.add_shape_box(
            paddle,
            hx=PADDLE_HALF[0],
            hy=PADDLE_HALF[1],
            hz=PADDLE_HALF[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5),
            color=(0.2, 0.45, 0.8),
        )
        hinge = builder.add_joint_revolute(
            parent=gantry,
            child=paddle,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -HINGE_DROP), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, PADDLE_HALF[2]), wp.quat_identity()),
            target_ke=args.hinge_kp,
            target_kd=args.hinge_kd,
        )
        builder.add_articulation([slide, hinge])

        cells = np.round(SAND_SIZE / SAND_SPACING).astype(int)
        radius = 0.5 * SAND_SPACING
        self.sand_begin = builder.particle_count
        builder.add_particle_grid(
            pos=wp.vec3(START_X + 0.06 + radius, -0.5 * SAND_SIZE[1] + radius, radius),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(cells[0]),
            dim_y=int(cells[1]),
            dim_z=int(cells[2]),
            cell_x=SAND_SPACING,
            cell_y=SAND_SPACING,
            cell_z=SAND_SPACING,
            mass=SAND_DENSITY * SAND_SPACING**3,
            jitter=0.25 * SAND_SPACING,
            radius_mean=radius,
            custom_attributes={"mpm:friction": args.sand_friction},
        )
        self.sand_end = builder.particle_count
        self.model = builder.finalize()
        self.paddle = paddle
        self.slide_dof = 0
        self.hinge_dof = 1

        self.solver = SolverMJMPM(
            self.model,
            coupling=SolverMJMPM.Coupling(args.coupling),
            relaxation=args.proxy_relaxation,
            mass_scale=args.proxy_mass_scale,
            coupling_iterations=args.coupling_iterations,
            articulated_response=args.articulated_response,
            mpm_options={
                "grid_type": "fixed",
                "voxel_size": args.voxel_size,
                "grid_padding": 8,
                "max_active_cell_count": 2048,
                # Eight particles per cell compact under the paddle; reserve room
                # because captured frames cannot grow the contact basis.
                "collider_basis": "pic32",
                "max_iterations": args.iterations,
                "solver": args.mpm_solver,
            },
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.graph = None

        self.settled_speed = 0.0
        self.peak_lag = 0.0
        self.peak_swing = 0.0
        self.hold_q = []
        self.hold_speed = 0.0
        self.rest_speed = 0.0
        self.pushed_center = None
        self.initial_center = self._sand().mean(axis=0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.3, -0.42, 0.26), pitch=-24.0, yaw=122.0)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    def _sand(self):
        return self.state_0.particle_q.numpy()[self.sand_begin : self.sand_end]

    def _simulate(self):
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        target = smoothstep(self.sim_time + self.frame_dt)
        self.control.joint_target_q.assign(np.array((target, 0.0), dtype=np.float32))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            # The first frame also sizes the MPM capacities that captured frames reuse.
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph:
                saved = [self.model.state(), self.model.state()]
                saved[0].assign(self.state_0)
                saved[1].assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(saved[0])
                self.state_1.assign(saved[1])
                self.graph = capture.graph
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        self._record(target)

    def _record(self, target):
        q = self.state_0.joint_q.numpy()
        # Bulk grain speed; single surface grains keep tumbling down the slope.
        grains = self.state_0.particle_qd.numpy()[self.sand_begin : self.sand_end]
        speed = float(np.percentile(np.linalg.norm(grains, axis=1), 99.0))
        if SETTLED[0] <= self.sim_time <= SETTLED[1]:
            self.settled_speed = max(self.settled_speed, speed)
        if SCHEDULE[1][0] <= self.sim_time <= SCHEDULE[3][0]:
            self.peak_lag = max(self.peak_lag, target - float(q[self.slide_dof]))
            self.peak_swing = max(self.peak_swing, abs(float(q[self.hinge_dof])))
        if HOLD[0] <= self.sim_time <= HOLD[1]:
            self.hold_q.append((self.sim_time, *q[: self.hinge_dof + 1]))
        if CREEP[0] <= self.sim_time <= CREEP[1]:
            self.hold_speed = max(self.hold_speed, speed)
        if abs(self.sim_time - SCHEDULE[3][0]) < 0.5 * self.frame_dt:
            self.pushed_center = self._sand().mean(axis=0)
        if REST[0] <= self.sim_time <= REST[1]:
            self.rest_speed = max(self.rest_speed, speed)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        q = self.state_0.joint_q.numpy()
        imgui.text(f"Time: {self.sim_time:.2f} s")
        imgui.text(f"Gantry target / travel: {smoothstep(self.sim_time) * 1e3:.1f} / {q[0] * 1e3:.1f} mm")
        imgui.text(f"Paddle swing: {np.rad2deg(q[1]):.2f} deg")

    def test_post_step(self):
        for array in (self.state_0.body_q, self.state_0.joint_q, self.state_0.particle_q, self.state_0.particle_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")
        sand = self._sand()
        if sand[:, 2].min() < -0.01:
            raise AssertionError(f"Sand sank through the ground: {sand[:, 2].min()}")

    def test_final(self):
        self.test_post_step()
        self.solver.check_capacity_status()
        if self.sim_time < SCHEDULE[-1][0] - 1.0e-6:
            raise AssertionError(f"Run at least {round(SCHEDULE[-1][0] * 60)} frames")
        # The pile is at rest before the paddle moves.
        if self.settled_speed > 0.01:
            raise AssertionError(f"The sand did not settle before the push: {self.settled_speed} m/s")
        moved = self.pushed_center - self.initial_center
        if self.args.coupling == "two_way":
            # The sand resists the push: the gantry lags its target and the paddle swings back.
            if self.peak_lag < 0.002:
                raise AssertionError(f"The sand did not resist the gantry: lag {self.peak_lag * 1e3:.2f} mm")
            if self.peak_swing < np.deg2rad(0.5):
                raise AssertionError(f"The sand load did not swing the paddle: {np.rad2deg(self.peak_swing):.2f} deg")
            if self.peak_swing > np.deg2rad(30.0):
                raise AssertionError(f"The paddle swung over: {np.rad2deg(self.peak_swing):.2f} deg")
            # Under the held drive force the sand keeps yielding, so the paddle
            # creeps forward; it must not oscillate about that creep.
            hold = np.asarray(self.hold_q)
            trend = np.polynomial.polynomial.polyfit(hold[:, 0], hold[:, 1:], 1)
            residual = np.abs(hold[:, 1:] - np.polynomial.polynomial.polyval(hold[:, 0], trend).T).max(axis=0)
            if residual[0] > 0.001 or residual[1] > np.deg2rad(0.5):
                raise AssertionError(
                    f"The held paddle oscillates: {residual[0] * 1e3:.2f} mm, {np.rad2deg(residual[1]):.2f} deg"
                )
        if moved[0] < 0.01:
            raise AssertionError(f"The paddle did not displace the sand: {moved[0] * 1e3:.1f} mm")
        if self.hold_speed > 0.05:
            raise AssertionError(f"The held sand keeps flowing fast: {self.hold_speed} m/s")
        # Plastic flow: the material stays where it was pushed after the paddle
        # leaves; only the unsupported face slumps back. Elastic material would
        # return to its initial shape.
        final = self._sand().mean(axis=0)
        if final[0] - self.initial_center[0] < 0.5 * moved[0]:
            raise AssertionError(
                f"The sand sprang back after release: {(final[0] - self.initial_center[0]) * 1e3:.1f} of "
                f"{moved[0] * 1e3:.1f} mm"
            )
        if self.rest_speed > 0.01:
            raise AssertionError(f"The released sand keeps flowing: {self.rest_speed} m/s")
        print(
            f"[PushSand] PASS: lag {self.peak_lag * 1e3:.1f} mm, swing {np.rad2deg(self.peak_swing):.2f} deg, "
            f"sand moved {moved[0] * 1e3:.1f} mm and kept {(final[0] - self.initial_center[0]) * 1e3:.1f} mm; "
            f"bulk speed settled {self.settled_speed * 1e3:.1f}, creep {self.hold_speed * 1e3:.1f}, "
            f"rest {self.rest_speed * 1e3:.1f} mm/s.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=round(SCHEDULE[-1][0] * 60))
        parser.add_argument("--coupling", choices=("two_way", "one_way"), default="two_way")
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=50)
        parser.add_argument("--mpm-solver", choices=("gs", "gs-soa", "jacobi"), default="gs")
        parser.add_argument("--voxel-size", type=float, default=0.02, help="MPM grid cell size [m].")
        parser.add_argument("--sand-friction", type=float, default=0.6, help="Drucker-Prager friction coefficient.")
        parser.add_argument("--drive-kp", type=float, default=3000.0, help="Gantry drive stiffness [N/m].")
        parser.add_argument("--drive-kd", type=float, default=150.0, help="Gantry drive damping [N*s/m].")
        parser.add_argument("--drive-effort", type=float, default=150.0, help="Gantry force limit [N].")
        parser.add_argument("--hinge-kp", type=float, default=40.0, help="Paddle hinge stiffness [N*m/rad].")
        parser.add_argument("--hinge-kd", type=float, default=0.5, help="Paddle hinge damping [N*m*s/rad].")
        parser.add_argument("--proxy-relaxation", type=float, default=0.5)
        parser.add_argument(
            "--proxy-mass-scale",
            type=float,
            default=4.0,
            help="Scale of the effective link inertia given to MPM proxies.",
        )
        parser.add_argument("--coupling-iterations", type=int, default=1)
        parser.add_argument("--articulated-response", action="store_true")
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
