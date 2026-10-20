# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp, lift, and release an MPM ball with a five-finger DexForce W1 hand.

    uv run --extra examples -m newton.examples mjmpm_hand_grasp
    uv run --extra examples -m newton.examples mjmpm_hand_grasp --material plastic

A palm-down W1 left hand hangs from a gravity-compensated vertical slide.
MuJoCo integrates the slide and the fingers, whose distal joints follow the
proximal ones through mimic equality constraints. The MPM environment owns
the ball. The force-limited finger and thumb drives are commanded fully
shut, so the ball's reaction stops them. The hand lowers onto the ball,
closes, lifts, holds, lowers, and opens. The elastic ball recovers its shape
after release; the ``plastic`` ball keeps the finger dents.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJMPM

HAND_URDF = Path(__file__).resolve().parent / "assets/w1_hand/left_hand.urdf"
TABLE_TOP = 0.10
BALL_RADIUS = 0.03
# A soft ball resting on a table is flattened at the contact; the flat base
# also keeps the particle sphere from rocking on its faceted bottom.
BALL_FLAT = 0.003
PARTICLE_SPACING = 0.004
# Ball center in the hand frame at the grasp pose (+z is the palm normal, +y
# points along the fingers): 2 mm from the palm hollow, where the four fingers
# reach the ball at about 70% closure and the thumb at about 70%.
GRASP_IN_HAND = np.array((0.005, 0.095, 0.045))
APPROACH = 0.05
LIFT = 0.08
FINGERS = ("INDEX", "MIDDLE", "RING", "PINKY")
# (time [s], slide offset above the grasp pose [m], finger closure, thumb closure)
SCHEDULE = (
    (0.0, APPROACH, 0.0, 0.0),
    (1.5, APPROACH, 0.0, 0.0),
    (2.3, 0.0, 0.0, 0.0),
    # Close slowly: fast fingers flick the light ball out before the thumb arrives.
    (3.8, 0.0, 1.0, 0.5),
    (4.2, 0.0, 1.0, 1.0),
    (4.6, 0.0, 1.0, 1.0),
    (5.6, LIFT, 1.0, 1.0),
    (6.6, LIFT, 1.0, 1.0),
    (7.6, 0.0, 1.0, 1.0),
    (8.2, 0.0, 0.0, 0.0),
    (9.0, APPROACH, 0.0, 0.0),
    (10.0, APPROACH, 0.0, 0.0),
)
SETTLED = (1.3, 1.5)
HOLD = (5.8, 6.6)
RELEASED = (9.6, 10.0)


def schedule(time):
    """Slide offset [m] and finger/thumb closure fractions at ``time`` [s]."""
    for a, b in pairwise(SCHEDULE):
        if time <= b[0]:
            s = float(np.clip((time - a[0]) / (b[0] - a[0]), 0.0, 1.0))
            s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
            return tuple(va + (vb - va) * s for va, vb in zip(a[1:], b[1:], strict=True))
    return SCHEDULE[-1][1:]


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
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * TABLE_TOP), wp.quat_identity()),
            hx=0.15,
            hy=0.15,
            hz=0.5 * TABLE_TOP,
            cfg=table_cfg,
            color=(0.57, 0.48, 0.36),
            label="table",
        )

        # Palm down: the hand frame is turned half a revolution about the finger axis.
        hand_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi)
        ball_center = np.array((0.0, 0.0, TABLE_TOP + BALL_RADIUS - BALL_FLAT))
        grasp_origin = ball_center - np.asarray(wp.quat_rotate(hand_rotation, wp.vec3(*GRASP_IN_HAND)))
        self.grasp_height = float(grasp_origin[2])
        rail = wp.transform(wp.vec3(*grasp_origin), wp.quat_identity())
        mount = builder.add_link(
            xform=rail,
            mass=0.5,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="wrist_mount",
        )
        slide = builder.add_joint_prismatic(
            parent=-1,
            child=mount,
            parent_xform=rail,
            axis=(0.0, 0.0, 1.0),
            target_ke=args.slide_kp,
            target_kd=args.slide_kd,
            effort_limit=args.slide_effort,
        )
        builder.add_articulation([slide])
        joint_begin = builder.joint_count
        body_begin = builder.body_count
        builder.add_urdf(
            str(HAND_URDF),
            xform=wp.transform(wp.vec3(0.0), hand_rotation),
            parent_body=mount,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            joint_damping_mode="passive",
        )
        self.hand_bodies = list(range(body_begin, builder.body_count))
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in [mount, *self.hand_bodies]:
            gravcomp.values[body] = 1.0
        hand_cfg_mu = args.finger_mu
        for shape in range(builder.shape_count):
            if builder.shape_body[shape] in self.hand_bodies:
                builder.shape_material_mu[shape] = hand_cfg_mu

        names = {label.rsplit("/", 1)[-1]: joint for joint, label in enumerate(builder.joint_label)}
        self.slide_dof = builder.joint_qd_start[slide]
        self.finger_dofs = [builder.joint_qd_start[names[f"LEFT_HAND_{finger}"]] for finger in FINGERS]
        self.thumb_dofs = [builder.joint_qd_start[names[f"LEFT_HAND_THUMB{k}"]] for k in (2, 1)]
        self.pip_dofs = [builder.joint_qd_start[names[f"LEFT_{finger}_PIP"]] for finger in FINGERS]
        upper = np.asarray(builder.joint_limit_upper)
        self.finger_upper = upper[self.finger_dofs]
        self.thumb_upper = upper[self.thumb_dofs]
        for dof in self.finger_dofs + self.thumb_dofs:
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
            builder.joint_target_ke[dof] = args.finger_kp
            builder.joint_target_kd[dof] = args.finger_kd
            builder.joint_effort_limit[dof] = args.finger_effort
        for dof in self.pip_dofs:
            # Distal joints follow their mimic constraint; the URDF's zero
            # effort is not a valid MuJoCo actuator force range.
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.NONE)
            builder.joint_target_ke[dof] = 0.0
            builder.joint_target_kd[dof] = 0.0
            builder.joint_effort_limit[dof] = args.finger_effort
        del joint_begin

        self.ball_begin = builder.particle_count
        h = PARTICLE_SPACING
        axis = np.arange(-BALL_RADIUS + 0.5 * h, BALL_RADIUS, h)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = (np.linalg.norm(grid, axis=1) <= BALL_RADIUS - 0.5 * h) & (grid[:, 2] >= BALL_FLAT - BALL_RADIUS)
        points = grid[inside]
        points += ball_center + np.array((0.0, 0.0, BALL_FLAT - BALL_RADIUS + 0.5 * h - points[:, 2].min()))
        if args.material == "elastic":
            material = {"mpm:young_modulus": 4.0e4, "mpm:poisson_ratio": 0.3, "mpm:yield_stress": 1.0e6}
        else:
            material = {"mpm:young_modulus": 4.0e4, "mpm:poisson_ratio": 0.3, "mpm:yield_stress": args.yield_stress}
        material["mpm:tensile_yield_ratio"] = 1.0
        # Foam and dough are viscoelastic; the relaxation time lets the ball come to rest.
        material["mpm:damping"] = args.ball_damping
        count = points.shape[0]
        builder.add_particles(
            pos=[wp.vec3(*p) for p in points],
            vel=[wp.vec3(0.0)] * count,
            mass=[args.ball_density * h**3] * count,
            radius=[0.5 * h] * count,
            custom_attributes={key: [value] * count for key, value in material.items()},
        )
        self.ball_end = builder.particle_count
        self.model = builder.finalize()

        self.solver = SolverMJMPM(
            self.model,
            coupling=SolverMJMPM.Coupling(args.coupling),
            relaxation=args.proxy_relaxation,
            mass_scale=args.proxy_mass_scale,
            coupling_iterations=args.coupling_iterations,
            articulated_response=args.articulated_response,
            rigid_contact_hard=False,
            mpm_options={
                "grid_type": "fixed",
                "voxel_size": args.voxel_size,
                "grid_padding": 12,
                "max_active_cell_count": 4096,
                "strain_basis": "P1d",
                "max_iterations": args.iterations,
                "solver": "gs",
                "air_drag": 0.0,
            },
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.target_q = self.model.joint_q.numpy().astype(np.float32)
        self.graph = None
        self.initial_ball = self._ball().copy()
        self.initial_extent = np.ptp(self.initial_ball, axis=0) + PARTICLE_SPACING
        self.records = []
        self.hold_impulse = np.zeros(self.model.body_count)
        self.settled_ball = None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.32, -0.32, 0.3), pitch=-24.0, yaw=135.0)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    def _ball(self):
        return self.state_0.particle_q.numpy()[self.ball_begin : self.ball_end]

    def _simulate(self):
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        offset, fingers, thumb = schedule(self.sim_time + self.frame_dt)
        q = self.target_q
        q[self.slide_dof] = offset
        q[self.finger_dofs] = fingers * self.finger_upper
        q[self.thumb_dofs] = thumb * self.thumb_upper
        self.control.joint_target_q.assign(q)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
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
        self._record(offset, fingers, thumb)
        if self.settled_ball is None and self.sim_time >= SETTLED[1]:
            self.settled_ball = self._ball().copy()
        if HOLD[0] <= self.sim_time <= HOLD[1]:
            # Material contact impulse on each hand link in the last substep.
            impulse = self.solver.environment.material_impulses.numpy()
            self.hold_impulse = np.maximum(self.hold_impulse, np.linalg.norm(impulse[:, :3], axis=1))

    def _record(self, offset, fingers, thumb):
        q = self.state_0.joint_q.numpy()
        ball = self._ball()
        velocity = self.state_0.particle_qd.numpy()[self.ball_begin : self.ball_end]
        speed = np.linalg.norm(velocity, axis=1)
        self.records.append(
            {
                "time": self.sim_time,
                "slide": float(q[self.slide_dof]),
                "slide_target": offset,
                "fingers": q[self.finger_dofs].copy(),
                "thumb": q[self.thumb_dofs].copy(),
                "pip": q[self.pip_dofs].copy(),
                "finger_command": fingers,
                "ball_bottom": float(ball[:, 2].min()) - 0.5 * PARTICLE_SPACING,
                "ball_center": ball.mean(axis=0),
                "ball_extent": np.ptp(ball, axis=0) + PARTICLE_SPACING,
                "speed": float(np.percentile(speed, 99.0)),
                "bulk_speed": float(np.linalg.norm(velocity.mean(axis=0))),
                "peak_speed": float(speed.max()),
            }
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        r = self.records[-1] if self.records else None
        imgui.text(f"Time: {self.sim_time:.2f} s")
        if r is not None:
            imgui.text(f"Slide: {r['slide'] * 1e3:.1f} mm, ball lift {(r['ball_bottom'] - TABLE_TOP) * 1e3:.1f} mm")
            imgui.text(f"Fingers: {np.round(np.rad2deg(r['fingers']), 1).tolist()} deg")

    def residual_deformation(self):
        """RMS particle offset [m] from the settled ball after the best rigid alignment."""
        reference = self.settled_ball if self.settled_ball is not None else self.initial_ball
        initial = reference - reference.mean(axis=0)
        current = self._ball() - self._ball().mean(axis=0)
        u, _, vt = np.linalg.svd(current.T @ initial)
        rotation = u @ np.diag((1.0, 1.0, np.linalg.det(u @ vt))) @ vt
        return float(np.sqrt(np.mean(np.sum((current - initial @ rotation.T) ** 2, axis=1))))

    def window(self, span):
        return [r for r in self.records if span[0] <= r["time"] <= span[1]]

    def test_post_step(self):
        for array in (self.state_0.joint_q, self.state_0.body_q, self.state_0.particle_q, self.state_0.particle_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")
        if self.records and self.records[-1]["ball_bottom"] < TABLE_TOP - 0.004:
            raise AssertionError(f"The ball sank into the table: {self.records[-1]['ball_bottom'] - TABLE_TOP}")

    def test_final(self):
        self.test_post_step()
        self.solver.check_capacity_status()
        if self.sim_time < SCHEDULE[-1][0] - 1.0e-6:
            raise AssertionError(f"Run at least {round(SCHEDULE[-1][0] * 60)} frames")
        settled = max(r["bulk_speed"] for r in self.window(SETTLED))
        if settled > 0.005:
            raise AssertionError(f"The ball did not settle before the approach: {settled} m/s")
        if self.args.coupling != "two_way":
            print(f"[HandGrasp] one-way run finished: {self.summary()}", flush=True)
            return
        hold = self.window(HOLD)
        fingers = np.array([r["fingers"] for r in hold])
        # The thumb and at least two fingers carry the ball through material contact.
        names = [label.rsplit("/", 1)[-1] for label in self.model.body_label]
        loaded = [names[b] for b in self.hand_bodies if self.hold_impulse[b] > 1.0e-6]
        if (
            not any("thumb" in n for n in loaded)
            or sum(f in n for n in loaded for f in ("index", "middle", "ring", "pinky")) < 2
        ):
            raise AssertionError(f"The ball is not held by the thumb and two fingers: {loaded}")
        # Contact stops the force-limited fingers short of their joint limits.
        if np.min(fingers.mean(axis=0) / self.finger_upper) > 0.95:
            raise AssertionError(f"No finger stopped on the ball: {np.rad2deg(fingers.mean(axis=0))} deg")
        lift = min(r["ball_bottom"] for r in hold) - TABLE_TOP
        if lift < 0.9 * LIFT:
            raise AssertionError(f"The ball was not lifted: {lift * 1e3:.1f} mm")
        drift = float(np.ptp(np.array([r["ball_center"] for r in hold]), axis=0).max())
        if drift > 0.002:
            raise AssertionError(f"The held ball slipped: {drift * 1e3:.2f} mm")
        pulse = np.rad2deg(np.ptp(fingers, axis=0))
        if np.any(pulse > 1.0):
            raise AssertionError(f"The held grip pulses: {pulse} deg")
        released = self.records[-1]
        if abs(released["ball_bottom"] - TABLE_TOP) > 0.003:
            raise AssertionError(f"The released ball is not on the table: {released['ball_bottom'] - TABLE_TOP}")
        residual = self.residual_deformation()
        print(
            f"[HandGrasp] PASS: {self.summary()}, loaded links {loaded}, residual deformation {residual * 1e3:.2f} mm",
            flush=True,
        )

    def summary(self):
        hold = self.window(HOLD)
        lift = min(r["ball_bottom"] for r in hold) - TABLE_TOP if hold else float("nan")
        fingers = np.array([r["fingers"] for r in hold]) if hold else np.zeros((1, 4))
        return (
            f"settled {max(r['bulk_speed'] for r in self.window(SETTLED)) * 1e3:.1f} mm/s, "
            f"min lift {lift * 1e3:.1f} mm, finger angles {np.round(np.rad2deg(fingers.mean(axis=0)), 1).tolist()} deg, "
            f"finger p2p {np.round(np.rad2deg(np.ptp(fingers, axis=0)), 2).tolist()} deg, "
            f"peak grain {max(r['peak_speed'] for r in self.records):.2f} m/s"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=round(SCHEDULE[-1][0] * 60))
        parser.add_argument("--material", choices=("elastic", "plastic"), default="elastic")
        parser.add_argument("--coupling", choices=("two_way", "one_way"), default="two_way")
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=50)
        parser.add_argument("--voxel-size", type=float, default=0.0075, help="MPM grid cell size [m].")
        parser.add_argument("--ball-density", type=float, default=300.0, help="Ball density [kg/m^3].")
        parser.add_argument("--yield-stress", type=float, default=1.0e3, help="Yield stress of the plastic ball [Pa].")
        parser.add_argument("--ball-damping", type=float, default=0.005, help="Ball viscous relaxation time [s].")
        parser.add_argument("--finger-kp", type=float, default=2.0, help="Finger drive stiffness [N*m/rad].")
        parser.add_argument("--finger-kd", type=float, default=0.02, help="Finger drive damping [N*m*s/rad].")
        parser.add_argument("--finger-effort", type=float, default=0.3, help="Finger joint torque limit [N*m].")
        parser.add_argument("--finger-mu", type=float, default=1.0, help="Hand friction coefficient.")
        parser.add_argument("--slide-kp", type=float, default=2000.0, help="Slide drive stiffness [N/m].")
        parser.add_argument("--slide-kd", type=float, default=100.0, help="Slide drive damping [N*s/m].")
        parser.add_argument("--slide-effort", type=float, default=50.0, help="Slide force limit [N].")
        parser.add_argument("--proxy-relaxation", type=float, default=0.5)
        parser.add_argument("--proxy-mass-scale", type=float, default=4.0)
        parser.add_argument("--coupling-iterations", type=int, default=1)
        parser.add_argument("--articulated-response", action="store_true")
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
