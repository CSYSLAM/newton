# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Use the V030 W1 parallel grippers to place two blocks into adjacent bins.

    uv run --extra examples -m newton.examples mjvbd_v2_w1_pick_place

The fixed-base robot approaches horizontally with its wrist cameras above the
grippers. Both blocks remain dynamic: only contact and friction lift them.
Use --robot-only to inspect the unmodified URDF at its original zero pose.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBDV2

ASSET = Path(__file__).resolve().parent / "assets/w1_v030/w1-030/robot.urdf"
TABLE_Z = 0.86
HALF_SIZE = np.array((0.024, 0.024, 0.045))
TCP = wp.vec3(0.0, 0.0, 0.125)
OPEN = 0.045
CLOSED = 0.0225
SIDES = ("left", "right")


@wp.kernel
def _prescribe(
    start: wp.array[float],
    end: wp.array[float],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    fraction: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    j = wp.tid()
    if q_start[j + 1] > q_start[j]:
        i = q_start[j]
        q[i] = wp.lerp(start[i], end[i], fraction)
        qd[qd_start[j]] = (end[i] - start[i]) * 60.0


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.frame = 0
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_dt = self.frame_dt / args.substeps
        self.pick = np.array(((0.43, 0.25, TABLE_Z + HALF_SIZE[2]), (0.43, -0.25, TABLE_Z + HALF_SIZE[2])))
        self.bins = np.array(((0.34, 0.50, TABLE_Z), (0.34, -0.50, TABLE_Z)))
        self.home = self.pick + np.array((-0.13, 0, 0.10))
        self.peak_z = self.pick[:, 2].copy()
        self.peak_ik_error = 0.0
        self.peak_joint_speed = 0.0

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.add_urdf(
            str(args.robot_urdf),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        if args.robot_only:
            builder.add_ground_plane()
            self.model = builder.finalize()
            self.state_0 = self.model.state()
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=wp.vec3(2.05, -2.0, 1.95), pitch=-18.0, yaw=137.0)
            return
        self.robot_joints = builder.joint_count
        self.robot_coords = len(builder.joint_q)
        self.robot_bodies = builder.body_count
        self.coords = {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[j]
            for j, name in enumerate(builder.joint_label)
            if builder.joint_type[j] != newton.JointType.FIXED
        }
        self.ee = [next(i for i, n in enumerate(builder.body_label) if n.endswith(f"/{s}_gripper_base")) for s in SIDES]
        for side, sign in (("LEFT", -1), ("RIGHT", 1)):
            builder.joint_q[self.coords[f"{side}_J2"]] = sign * 0.9
            builder.joint_q[self.coords[f"{side}_J4"]] = sign * 1.0
            for finger in (1, 2):
                builder.joint_q[self.coords[f"{side}_FINGER{finger}_JOINT"]] = OPEN
        for i in range(self.robot_bodies):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        for i in range(builder.shape_count):
            builder.shape_material_mu[i] = 1.0
            builder.shape_material_ke[i] = 2.0e4
            builder.shape_material_kd[i] = 100.0

        self.ik_model = builder.finalize()
        self._build_ik()
        self._solve_ik(self.home, OPEN, iterations=150)
        builder.joint_q[:] = self.ik_q.numpy()[0].tolist()

        cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.8)

        def box(label, position, half, color):
            return builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
                color=color,
                label=label,
            )

        builder.add_ground_plane(cfg=cfg)
        box("worktop", (0.46, 0.0, TABLE_Z - 0.025), (0.30, 0.82, 0.025), (0.57, 0.48, 0.36))
        for x in (0.21, 0.71):
            for y in (-0.74, 0.74):
                box("table_leg", (x, y, (TABLE_Z - 0.05) / 2), (0.025, 0.025, (TABLE_Z - 0.05) / 2), (0.22, 0.25, 0.29))
        colors = ((0.18, 0.55, 0.78), (0.88, 0.39, 0.15))
        for side, center, color in zip(SIDES, self.bins, colors, strict=True):
            box(f"{side}_bin_floor", center + np.array((0, 0, 0.009)), (0.11, 0.13, 0.009), color)
            for sign in (-1, 1):
                box(f"{side}_bin_end", center + np.array((sign * 0.105, 0, 0.065)), (0.005, 0.13, 0.065), color)
                box(f"{side}_bin_side", center + np.array((0, sign * 0.125, 0.065)), (0.10, 0.005, 0.065), color)
        self.objects = []
        for side, position, color in zip(SIDES, self.pick, colors, strict=True):
            body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label=f"{side}_block")
            builder.add_shape_box(
                body,
                hx=HALF_SIZE[0],
                hy=HALF_SIZE[1],
                hz=HALF_SIZE[2],
                color=color,
                cfg=newton.ModelBuilder.ShapeConfig(density=450.0, mu=1.0, ke=2.0e4, kd=100.0),
            )
            self.objects.append(body)
        builder.color()
        self.model = builder.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.frame_start = wp.clone(self.ik_model.joint_q)
        self.frame_end = wp.clone(self.ik_model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": args.iterations,
                "friction_epsilon": 1.0e-4,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
            },
            collision_options={"include_static_kinematic_pairs": False, "broad_phase": "nxn"},
        )
        self.graph = None
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.05, -2.0, 1.95), pitch=-18.0, yaw=137.0)

    def _build_ik(self):
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.positions = [
            ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*p)], dtype=wp.vec3))
            for body, p in zip(self.ee, self.home, strict=True)
        ]
        # Gripper +Z points forward; its -X (the camera side) points upward.
        rotation = wp.vec4(0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5))
        objectives = list(self.positions) + [
            ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([rotation], dtype=wp.vec4)) for body in self.ee
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

    def _solve_ik(self, targets, opening, *, iterations=12):
        for objective, p in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*p))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        q = np.clip(self.ik_q.numpy()[0], self.lower, self.upper)
        for side in ("LEFT", "RIGHT"):
            for finger in (1, 2):
                q[self.coords[f"{side}_FINGER{finger}_JOINT"]] = opening
        self.ik_q.assign(q.reshape(1, -1))
        return q

    def _plan(self, time):
        lift = self.pick + np.array((0, 0, 0.25))
        carry = self.bins + np.array((0, 0, HALF_SIZE[2] + 0.25))
        # Release above the rim so the horizontal palm clears the near wall.
        drop = self.bins + np.array((0, 0, 0.13 + HALF_SIZE[2] + 0.012))
        approach = self.pick + np.array((-0.13, 0, 0))
        retreat = carry + np.array((-0.10, 0, 0))
        points = [
            (0.0, self.home, OPEN),
            (0.5, self.home, OPEN),
            (1.5, approach, OPEN),
            (2.8, self.pick, OPEN),
            (3.8, self.pick, CLOSED),
            (5.5, lift, CLOSED),
            (7.5, carry, CLOSED),
            (8.5, drop, CLOSED),
            (9.3, drop, OPEN),
            (10.5, carry, OPEN),
            (12.0, retreat, OPEN),
        ]
        for (a, pa, ga), (b, pb, gb) in pairwise(points):
            if time <= b:
                t = float(np.clip((time - a) / (b - a), 0, 1))
                t = t * t * t * (10 + t * (-15 + 6 * t))
                return (1 - t) * pa + t * pb, (1 - t) * ga + t * gb
        return retreat, OPEN

    def _simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                _prescribe,
                self.robot_joints,
                [
                    self.frame_start,
                    self.frame_end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (substep + 1) / self.args.substeps,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                ],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.args.robot_only:
            self.frame += 1
            self.sim_time = self.frame * self.frame_dt
            return
        targets, opening = self._plan(self.sim_time + self.frame_dt)
        start = self.ik_q.numpy()[0]
        end = self._solve_ik(targets, opening)
        self.peak_joint_speed = max(self.peak_joint_speed, float(np.max(np.abs(end - start)) / self.frame_dt))
        self.frame_start.assign(start)
        self.frame_end.assign(end)
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph and self.args.substeps % 2 == 0:
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
        body_q = self.state_0.body_q.numpy()
        self.peak_z = np.maximum(self.peak_z, body_q[self.objects, 2])
        for body, target in zip(self.ee, targets, strict=True):
            tcp = np.asarray(wp.transform_point(wp.transform(*body_q[body]), TCP))
            self.peak_ik_error = max(self.peak_ik_error, float(np.linalg.norm(tcp - target)))
        if self.frame % 60 == 0:
            print(
                f"[W1PickPlace] t={self.sim_time:.1f} blocks={body_q[self.objects, :3].round(4).tolist()} "
                f"peak_IK={self.peak_ik_error:.5f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Reject invalid states, escaped blocks and excessive IK error."""
        for array in (self.state_0.body_q, self.state_0.body_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite rigid-body state")
        if self.args.robot_only:
            return
        if np.min(self.state_0.body_q.numpy()[self.objects, 2]) < TABLE_Z - 0.01:
            raise AssertionError("A block fell through or off the table")
        if self.peak_ik_error > 0.01:
            raise AssertionError(f"TCP error exceeded 1 cm: {self.peak_ik_error}")
        q = self.state_0.joint_q.numpy()[: self.robot_coords]
        if np.any(q < self.lower - 1.0e-5) or np.any(q > self.upper + 1.0e-5):
            raise AssertionError("Robot exceeded the source joint position limits")

    def test_final(self):
        """Require both physical lifts and released blocks settled inside their bins."""
        self.test_post_step()
        if self.args.robot_only:
            if np.any(self.state_0.joint_q.numpy() != self.model.joint_q.numpy()):
                raise AssertionError("Robot preview must preserve the original URDF zero pose")
            return
        if self.frame < 840:
            raise AssertionError("Run at least 840 frames to validate both placements")
        if np.any(self.peak_z < self.pick[:, 2] + 0.18):
            raise AssertionError(f"Both blocks must be lifted at least 18 cm: {self.peak_z}")
        poses = self.state_0.body_q.numpy()[self.objects]
        for pose, center in zip(poses, self.bins, strict=True):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            extent = np.abs(rotation) @ HALF_SIZE
            if np.any(np.abs(pose[:2] - center[:2]) + extent[:2] > (0.10, 0.12)):
                raise AssertionError(f"Block not fully inside its bin: {pose}")
            if abs(pose[2] - extent[2] - TABLE_Z - 0.018) > 0.006:
                raise AssertionError(f"Block not resting on bin floor: {pose}")
        if np.max(np.linalg.norm(self.state_0.body_qd.numpy()[self.objects], axis=1)) > 0.05:
            raise AssertionError("Released blocks have not settled")
        if np.any(self.model.body_flags.numpy()[self.objects] & int(newton.BodyFlags.KINEMATIC)):
            raise AssertionError("Blocks must remain dynamic")
        if self.peak_joint_speed > 4.145:
            raise AssertionError(f"Arm velocity exceeded source limit: {self.peak_joint_speed}")
        q = self.state_0.joint_q.numpy()
        for side in ("LEFT", "RIGHT"):
            if abs(q[self.coords[f"{side}_FINGER1_JOINT"]] - OPEN) > 1.0e-4:
                raise AssertionError("Both grippers must be open after release")
        print("[W1PickPlace] PASS: both blocks lifted, released and settled in their bins.", flush=True)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        parser.add_argument("--robot-urdf", type=Path, default=ASSET)
        parser.add_argument(
            "--robot-only", action="store_true", help="Inspect the original URDF zero pose without the task"
        )
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=12)
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
