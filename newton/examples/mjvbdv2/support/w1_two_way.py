# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared dynamic W1 V030 scene driven through two-way MJVBDV2 coupling.

MuJoCo integrates the robot from PD joint targets produced by per-frame IK, so
VBD contact wrenches and MuJoCo contacts with static shapes can stop the arms
and fingers instead of being overridden by prescribed joint positions.
Subclasses add task objects with :meth:`W1TwoWayScene.build_task` and return
per-gripper TCP targets and the finger command from :meth:`W1TwoWayScene.plan`.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBDV2

ASSET = Path(__file__).resolve().parents[1] / "assets/w1_v030/w1-030/robot.urdf"
TABLE_Z = 0.86
TCP = wp.vec3(0.0, 0.0, 0.125)
FINGER_OPEN = 0.045
FINGER_SHUT = 0.0
SIDES = ("left", "right")
# Gripper +Z points forward; its -X (the camera side) points upward.
HORIZONTAL_GRIPPER = wp.vec4(0.0, float(np.sqrt(0.5)), 0.0, float(np.sqrt(0.5)))
# Gripper +Z points down; its camera side faces away from the robot.
DOWNWARD_GRIPPER = wp.vec4(0.0, 1.0, 0.0, 0.0)


@wp.kernel
def _interpolate_targets(
    start: wp.array[float],
    end: wp.array[float],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    fraction: float,
    inv_frame_dt: float,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    j = wp.tid()
    if q_start[j + 1] > q_start[j]:
        i = q_start[j]
        target_q[i] = wp.lerp(start[i], end[i], fraction)
        target_qd[qd_start[j]] = (end[i] - start[i]) * inv_frame_dt


def smoothstep(points, time):
    """Interpolate ``(time, value...)`` keyframes with a quintic ease."""
    for a, b in pairwise(points):
        if time <= b[0]:
            s = float(np.clip((time - a[0]) / (b[0] - a[0]), 0.0, 1.0))
            s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
            return tuple((1.0 - s) * np.asarray(va) + s * np.asarray(vb) for va, vb in zip(a[1:], b[1:], strict=True))
    return tuple(np.asarray(v) for v in points[-1][1:])


class W1TwoWayScene:
    """Dynamic W1 V030 with gravity compensation, PD drives, and per-frame IK."""

    camera_pos = wp.vec3(2.05, -2.0, 1.95)
    camera_pitch = -18.0
    camera_yaw = 137.0
    table_z = TABLE_Z
    gripper_rotation = HORIZONTAL_GRIPPER
    home_offset = np.array((-0.13, 0.0, 0.10))

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if args.substeps < 2 or args.substeps % 2 or args.iterations < 1:
            raise ValueError("Use an even substep count >= 2 and a positive iteration count")
        self.frame = 0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / args.substeps
        self.peak_free_tcp_error = 0.0

        builder = newton.ModelBuilder()
        SolverMJVBDV2.register_custom_attributes(builder)
        builder.rigid_gap = 0.002
        builder.add_urdf(str(ASSET), floating=False, enable_self_collisions=False, collapse_fixed_joints=False)
        self.robot_joints = builder.joint_count
        self.robot_bodies = builder.body_count
        self.coords = {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[j]
            for j, name in enumerate(builder.joint_label)
            if builder.joint_type[j] != newton.JointType.FIXED
        }
        self.ee = [next(i for i, n in enumerate(builder.body_label) if n.endswith(f"/{s}_gripper_base")) for s in SIDES]
        self.finger_coords = [
            [self.coords[f"{side}_FINGER{finger}_JOINT"] for finger in (1, 2)] for side in ("LEFT", "RIGHT")
        ]
        for side, sign in (("LEFT", -1), ("RIGHT", 1)):
            builder.joint_q[self.coords[f"{side}_J2"]] = sign * 0.9
            builder.joint_q[self.coords[f"{side}_J4"]] = sign * 1.0
            for coord in self.finger_coords[SIDES.index(side.lower())]:
                builder.joint_q[coord] = FINGER_OPEN
        # MuJoCo cancels link weight passively, like a gravity-compensated
        # controller, so the PD drives only correct tracking error.
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in range(self.robot_bodies):
            gravcomp.values[body] = 1.0
        for shape in range(builder.shape_count):
            builder.shape_material_mu[shape] = 1.0
            builder.shape_material_ke[shape] = 2.0e4
            builder.shape_material_kd[shape] = 100.0

        self.ik_model = builder.finalize()
        self.home = self.initial_targets() + self.home_offset
        self._build_ik()
        self._solve_ik(self.home, FINGER_OPEN, iterations=150)
        builder.joint_q[:] = self.ik_q.numpy()[0].tolist()

        self.surface_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.8)
        builder.add_ground_plane(cfg=self.surface_cfg)
        self.add_box(builder, "worktop", (0.46, 0.0, self.table_z - 0.025), (0.30, 0.82, 0.025), (0.57, 0.48, 0.36))
        for x in (0.21, 0.71):
            for y in (-0.74, 0.74):
                half_height = (self.table_z - 0.05) / 2
                self.add_box(builder, "table_leg", (x, y, half_height), (0.025, 0.025, half_height), (0.22, 0.25, 0.29))
        self.build_task(builder)
        builder.color()
        self.model = builder.finalize()
        self._configure_drives()
        self.configure_model(self.model)

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        self.frame_start = wp.clone(self.ik_model.joint_q)
        self.frame_end = wp.clone(self.ik_model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="dynamic",
            contact_mode="full",
            coupling=args.coupling,
            vbd_options={
                "iterations": args.iterations,
                "friction_epsilon": 1.0e-4,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
                **self.vbd_options(),
            },
            collision_options={
                "include_static_kinematic_pairs": False,
                "broad_phase": "nxn",
                **self.collision_options(),
            },
        )
        self.graph = None
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=self.camera_pos, pitch=self.camera_pitch, yaw=self.camera_yaw)

    # Hooks -----------------------------------------------------------------

    def initial_targets(self) -> np.ndarray:
        """Return the two TCP positions [m] the home pose hovers behind."""
        raise NotImplementedError

    def build_task(self, builder: newton.ModelBuilder) -> None:
        """Add task objects after the robot, ground, and worktop."""

    def plan(self, time: float) -> tuple[np.ndarray, float]:
        """Return ``(tcp_targets[2, 3], finger_target)`` at ``time`` [s]."""
        raise NotImplementedError

    def tracks_freely(self, time: float) -> bool:
        """Whether the TCPs should reach their targets (no intended contact)."""
        del time
        return True

    def configure_model(self, model: newton.Model) -> None:
        """Adjust model-level contact settings before the solver is built."""

    def vbd_options(self) -> dict:
        return {}

    def collision_options(self) -> dict:
        return {}

    # Scene helpers -----------------------------------------------------------

    def add_box(self, builder, label, position, half, color, *, body=-1, cfg=None):
        return builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()) if body < 0 else wp.transform_identity(),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=cfg or self.surface_cfg,
            color=color,
            label=label,
        )

    def tcp_positions(self, body_q: np.ndarray) -> np.ndarray:
        return np.array([np.asarray(wp.transform_point(wp.transform(*body_q[body]), TCP)) for body in self.ee])

    def finger_q(self) -> np.ndarray:
        """Finger-1 opening [m] of the left and right grippers."""
        q = self.state_0.joint_q.numpy()
        return np.array([q[coords[0]] for coords in self.finger_coords])

    # Robot control -----------------------------------------------------------

    def _configure_drives(self):
        """Use URDF effort limits, stiff arm drives, and geared-finger armature.

        Finger links weigh 25 g. Without reflected gear inertia their contact
        feedback, which reaches MuJoCo one substep later, is not stable.
        """
        mode = self.model.joint_target_mode.numpy().copy()
        ke = self.model.joint_target_ke.numpy().copy()
        kd = self.model.joint_target_kd.numpy().copy()
        armature = self.model.joint_armature.numpy().copy()
        qd_start = self.model.joint_qd_start.numpy()
        for joint, label in enumerate(self.model.joint_label[: self.robot_joints]):
            begin, end = int(qd_start[joint]), int(qd_start[joint + 1])
            if begin == end:
                continue
            if "FINGER" in label.rsplit("/", 1)[-1]:
                ke[begin:end], kd[begin:end] = self.args.finger_kp, self.args.finger_kd
                armature[begin:end] = np.maximum(armature[begin:end], self.args.finger_armature)
            else:
                ke[begin:end], kd[begin:end] = self.args.arm_kp, self.args.arm_kd
            mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.model.joint_armature.assign(armature)

    def _build_ik(self):
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.positions = [
            ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*p)], dtype=wp.vec3))
            for body, p in zip(self.ee, self.home, strict=True)
        ]
        objectives = list(self.positions) + [
            ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([self.gripper_rotation], dtype=wp.vec4))
            for body in self.ee
        ]
        objectives.append(
            ik.IKObjectiveJointLimit(self.ik_model.joint_limit_lower, self.ik_model.joint_limit_upper, weight=10.0)
        )
        mask = np.zeros(self.ik_model.joint_dof_count, dtype=bool)
        for side in ("LEFT", "RIGHT"):
            for j in range(1, 8):
                mask[self.coords[f"{side}_J{j}"]] = True
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            1,
            objectives,
            joint_dof_mask=wp.array(mask, dtype=wp.bool),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            lambda_initial=0.03,
        )
        self.lower = self.ik_model.joint_limit_lower.numpy()
        self.upper = self.ik_model.joint_limit_upper.numpy()

    def _solve_ik(self, targets, finger_target, *, iterations=12):
        for objective, p in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*p))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        q = np.clip(self.ik_q.numpy()[0], self.lower, self.upper)
        for coords in self.finger_coords:
            q[coords] = finger_target
        self.ik_q.assign(q.reshape(1, -1))
        return q

    def _simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                _interpolate_targets,
                self.robot_joints,
                [
                    self.frame_start,
                    self.frame_end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (substep + 1) / self.args.substeps,
                    1.0 / self.frame_dt,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        targets, finger_target = self.plan(self.sim_time + self.frame_dt)
        start = self.ik_q.numpy()[0].copy()
        end = self._solve_ik(targets, finger_target)
        self.frame_start.assign(start)
        self.frame_end.assign(end)
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph:
                saved_0, saved_1 = self.model.state(), self.model.state()
                saved_0.assign(self.state_0)
                saved_1.assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(saved_0)
                self.state_1.assign(saved_1)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        self.targets = targets
        if self.tracks_freely(self.sim_time):
            tcp = self.tcp_positions(self.state_0.body_q.numpy())
            self.peak_free_tcp_error = max(
                self.peak_free_tcp_error, float(np.max(np.linalg.norm(tcp - targets, axis=1)))
            )
        self.after_step()

    def after_step(self):
        """Record task metrics after each frame."""

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def check_finite(self):
        for array in (self.state_0.body_q, self.state_0.body_qd, self.state_0.particle_q):
            if array is not None and not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")

    @staticmethod
    def add_arguments(parser, *, num_frames):
        parser.set_defaults(num_frames=num_frames)
        parser.add_argument(
            "--coupling",
            choices=("two_way", "one_way"),
            default="two_way",
            help="MuJoCo/VBD feedback. one_way lets targets drive the robot through obstacles.",
        )
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=12)
        parser.add_argument("--arm-kp", type=float, default=12000.0, help="Arm drive stiffness [N*m/rad].")
        parser.add_argument("--arm-kd", type=float, default=350.0, help="Arm drive damping [N*m*s/rad].")
        parser.add_argument("--finger-kp", type=float, default=1000.0, help="Finger drive stiffness [N/m].")
        parser.add_argument("--finger-kd", type=float, default=20.0, help="Finger drive damping [N*s/m].")
        parser.add_argument(
            "--finger-armature", type=float, default=0.3, help="Reflected finger gear inertia [kg] for stable feedback."
        )
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser
