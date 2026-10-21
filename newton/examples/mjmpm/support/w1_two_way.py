# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared dynamic W1 V030 scene driven through two-way MJMPM coupling.

MuJoCo integrates the robot from PD joint targets produced by per-frame IK, so
MPM contact wrenches and MuJoCo contacts with static shapes can stop the arms
and fingers instead of being overridden by prescribed joint positions.
Subclasses add task objects with :meth:`W1MJMPMScene.build_task` and return
per-gripper TCP targets and the finger command from :meth:`W1MJMPMScene.plan`.
"""

import argparse
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJMPM

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


@wp.kernel
def _write_frame_inputs(
    left: wp.vec3,
    right: wp.vec3,
    left_rotation: wp.vec4,
    right_rotation: wp.vec4,
    left_finger: float,
    right_finger: float,
    tcp_targets: wp.array[wp.vec3],
    rotation_targets: wp.array[wp.vec4],
    finger_target: wp.array[float],
):
    """Store this frame's sampled targets in persistent device buffers."""
    tcp_targets[0] = left
    tcp_targets[1] = right
    rotation_targets[0] = left_rotation
    rotation_targets[1] = right_rotation
    finger_target[0] = left_finger
    finger_target[1] = right_finger


@wp.kernel
def _unpack_tcp_targets(
    tcp_targets: wp.array[wp.vec3],
    rotation_targets: wp.array[wp.vec4],
    left_objective: wp.array[wp.vec3],
    right_objective: wp.array[wp.vec3],
    left_rotation_objective: wp.array[wp.vec4],
    right_rotation_objective: wp.array[wp.vec4],
):
    """Feed the stored TCP targets to the IK position and rotation objectives."""
    left_objective[0] = tcp_targets[0]
    right_objective[0] = tcp_targets[1]
    left_rotation_objective[0] = rotation_targets[0]
    right_rotation_objective[0] = rotation_targets[1]


@wp.kernel
def _finish_ik_frame(
    ik_q: wp.array2d[float],
    lower: wp.array[float],
    upper: wp.array[float],
    finger_coords: wp.array[int],
    finger_target: wp.array[float],
    frame_end: wp.array[float],
):
    """Clamp the IK solution to joint limits, apply the finger commands, and store the frame target.

    ``finger_coords`` lists the two finger coordinates of the left, then the right gripper.
    """
    i = wp.tid()
    value = wp.clamp(ik_q[0, i], lower[i], upper[i])
    for k in range(finger_coords.shape[0]):
        if finger_coords[k] == i:
            value = finger_target[k // 2]
    ik_q[0, i] = value
    frame_end[i] = value


def smoothstep(points, time):
    """Interpolate ``(time, value...)`` keyframes with a quintic ease."""
    for a, b in pairwise(points):
        if time <= b[0]:
            s = float(np.clip((time - a[0]) / (b[0] - a[0]), 0.0, 1.0))
            s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
            return tuple((1.0 - s) * np.asarray(va) + s * np.asarray(vb) for va, vb in zip(a[1:], b[1:], strict=True))
    return tuple(np.asarray(v) for v in points[-1][1:])


class W1MJMPMScene:
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

        # IK keeps the URDF's fixed links so the gripper-base end effectors
        # exist. The simulated robot collapses them into their parents: the
        # movable-joint coordinates are identical, shapes keep their poses,
        # and MuJoCo's per-level tree passes run 13 instead of 22 levels deep.
        ik_builder = newton.ModelBuilder()
        # IK only needs the joint tree; avoid loading all visual/collision
        # meshes twice. The simulated robot below retains every source shape.
        ik_urdf = ET.parse(ASSET)
        for link in ik_urdf.getroot().findall("link"):
            for shape in [*link.findall("visual"), *link.findall("collision")]:
                link.remove(shape)
        ik_builder.add_urdf(
            ET.tostring(ik_urdf.getroot(), encoding="unicode"),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        builder.rigid_gap = 0.002
        builder.add_urdf(str(ASSET), floating=False, enable_self_collisions=False, collapse_fixed_joints=True)
        self.robot_joints = builder.joint_count
        self.robot_bodies = builder.body_count
        self.coords = self._movable_coordinates(builder)
        if self._movable_coordinates(ik_builder) != self.coords:
            raise RuntimeError("Collapsing fixed W1 joints changed the joint-coordinate layout")
        self.ee = [
            next(i for i, n in enumerate(ik_builder.body_label) if n.endswith(f"/{s}_gripper_base")) for s in SIDES
        ]
        self.finger_coords = [
            [self.coords[f"{side}_FINGER{finger}_JOINT"] for finger in (1, 2)] for side in ("LEFT", "RIGHT")
        ]
        for target in (builder, ik_builder):
            for side, sign in (("LEFT", -1), ("RIGHT", 1)):
                target.joint_q[self.coords[f"{side}_J2"]] = sign * 0.9
                target.joint_q[self.coords[f"{side}_J4"]] = sign * 1.0
                for coord in self.finger_coords[SIDES.index(side.lower())]:
                    target.joint_q[coord] = FINGER_OPEN
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

        self.ik_model = ik_builder.finalize()
        self._map_tcp_to_scene(builder)
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
        # Per-frame IK runs inside the captured frame graph; the host only
        # writes the sampled targets into these buffers.
        self.frame_end = wp.clone(self.ik_q.flatten())
        self.frame_start = wp.clone(self.frame_end)
        self.tcp_targets = wp.zeros(2, dtype=wp.vec3, device=self.model.device)
        self.rotation_targets = wp.zeros(2, dtype=wp.vec4, device=self.model.device)
        self.finger_target = wp.zeros(2, dtype=float, device=self.model.device)
        self.finger_coord_array = wp.array(
            [coord for coords in self.finger_coords for coord in coords], dtype=int, device=self.model.device
        )
        self.solver = SolverMJMPM(
            self.model,
            coupling=SolverMJMPM.Coupling(args.coupling),
            relaxation=args.proxy_relaxation,
            relaxation_mode=args.relaxation_mode,
            mass_scale=args.proxy_mass_scale,
            coupling_iterations=args.coupling_iterations,
            coupling_tolerance=args.coupling_tolerance,
            articulated_response=args.articulated_response and args.coupling == "two_way",
            rigid_iterations=args.rigid_iterations,
            rigid_contact_hard=False,
            # Held-block friction converges over steps instead of restarting
            # each step; cold-started sweeps let the block slip and turn in the
            # grip at coarse substeps.
            rigid_warm_start=True,
            rigid_solver=args.rigid_solver,
            rigid_material_coupling=args.rigid_material_coupling,
            mpm_options={
                "grid_type": "fixed",
                "voxel_size": args.voxel_size,
                "grid_padding": 20,
                "max_active_cell_count": 512,
                "strain_basis": args.strain_basis,
                "collider_basis": "pic48",
                "max_iterations": args.iterations,
                "stall_tolerance": args.stall_tolerance,
                "solver": args.mpm_solver,
                "air_drag": 0.0,
            }
            | self.mpm_option_overrides(),
        )
        self.graph = None
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=self.camera_pos, pitch=self.camera_pitch, yaw=self.camera_yaw)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    # Hooks -----------------------------------------------------------------

    def initial_targets(self) -> np.ndarray:
        """Return the two TCP positions [m] the home pose hovers behind."""
        raise NotImplementedError

    def build_task(self, builder: newton.ModelBuilder) -> None:
        """Add task objects after the robot, ground, and worktop."""

    def plan(self, time: float) -> tuple[np.ndarray, float | np.ndarray]:
        """Return ``(tcp_targets[2, 3], finger_target)`` at ``time`` [s].

        ``finger_target`` is one opening [m] for both grippers or one per gripper.
        """
        raise NotImplementedError

    def gripper_rotations(self, time: float) -> tuple[wp.vec4, wp.vec4]:
        """Return the left and right gripper orientation targets ``(x, y, z, w)`` at ``time`` [s]."""
        del time
        return self.gripper_rotation, self.gripper_rotation

    def mpm_option_overrides(self) -> dict:
        """Return MPM options that replace the scene defaults."""
        return {}

    def tracks_freely(self, time: float) -> bool:
        """Whether the TCPs should reach their targets (no intended contact)."""
        del time
        return True

    def configure_model(self, model: newton.Model) -> None:
        """Adjust model-level contact settings before the solver is built."""

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
        """World TCP positions of the simulated grippers."""
        return np.array(
            [
                np.asarray(wp.transform_point(wp.transform(*body_q[body]), offset))
                for body, offset in zip(self.scene_ee, self.scene_tcp, strict=True)
            ]
        )

    @staticmethod
    def _movable_coordinates(builder: newton.ModelBuilder) -> dict[str, int]:
        return {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[j]
            for j, name in enumerate(builder.joint_label)
            if builder.joint_type[j] != newton.JointType.FIXED
        }

    def _map_tcp_to_scene(self, scene: newton.ModelBuilder) -> None:
        """Express each TCP in the simulated link that its gripper base collapses into."""
        model = self.ik_model
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        body_q = state.body_q.numpy()
        joint_type = model.joint_type.numpy()
        parent_joint = {int(child): joint for joint, child in enumerate(model.joint_child.numpy())}
        joint_parent = model.joint_parent.numpy()
        self.scene_ee, self.scene_tcp = [], []
        for gripper in self.ee:
            body = gripper
            while int(joint_type[parent_joint[body]]) == int(newton.JointType.FIXED):
                body = int(joint_parent[parent_joint[body]])
            label = model.body_label[body]
            self.scene_ee.append(scene.body_label.index(label))
            tcp_world = wp.transform_point(wp.transform(*body_q[gripper]), TCP)
            self.scene_tcp.append(wp.transform_point(wp.transform_inverse(wp.transform(*body_q[body])), tcp_world))

    def finger_q(self) -> np.ndarray:
        """Finger-1 opening [m] of the left and right grippers."""
        return self.finger_pairs()[:, 0]

    def finger_pairs(self) -> np.ndarray:
        """Opening [m] of both fingers of the left and right grippers, shape ``(2, 2)``."""
        q = self.state_0.joint_q.numpy()
        return np.array([q[coords] for coords in self.finger_coords])

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
        damping = self.model.joint_damping.numpy().copy()
        qd_start = self.model.joint_qd_start.numpy()
        for joint, label in enumerate(self.model.joint_label[: self.robot_joints]):
            begin, end = int(qd_start[joint]), int(qd_start[joint + 1])
            if begin == end:
                continue
            if "FINGER" in label.rsplit("/", 1)[-1]:
                ke[begin:end], kd[begin:end] = self.args.finger_kp, self.args.finger_kd
                armature[begin:end] = np.maximum(armature[begin:end], self.args.finger_armature)
                damping[begin:end] = np.maximum(damping[begin:end], self.args.finger_damping)
            else:
                ke[begin:end], kd[begin:end] = self.args.arm_kp, self.args.arm_kd
            mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.model.joint_armature.assign(armature)
        self.model.joint_damping.assign(damping)

    def _build_ik(self):
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.positions = [
            ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*p)], dtype=wp.vec3))
            for body, p in zip(self.ee, self.home, strict=True)
        ]
        self.rotations = [
            ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([self.gripper_rotation], dtype=wp.vec4))
            for body in self.ee
        ]
        objectives = list(self.positions) + list(self.rotations)
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

    def _solve_frame_ik(self):
        """Solve this frame's IK target on the device; the previous target starts the interval."""
        wp.copy(self.frame_start, self.frame_end)
        wp.launch(
            _unpack_tcp_targets,
            1,
            [
                self.tcp_targets,
                self.rotation_targets,
                self.positions[0].target_positions,
                self.positions[1].target_positions,
                self.rotations[0].target_rotations,
                self.rotations[1].target_rotations,
            ],
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.args.ik_iterations)
        wp.launch(
            _finish_ik_frame,
            self.frame_end.shape[0],
            [
                self.ik_q,
                self.ik_model.joint_limit_lower,
                self.ik_model.joint_limit_upper,
                self.finger_coord_array,
                self.finger_target,
                self.frame_end,
            ],
        )

    def _simulate_frame(self):
        self._solve_frame_ik()
        self._simulate()

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
        fingers = np.broadcast_to(np.asarray(finger_target, dtype=float), (2,))
        left_rotation, right_rotation = self.gripper_rotations(self.sim_time + self.frame_dt)
        wp.launch(
            _write_frame_inputs,
            1,
            [
                wp.vec3(*targets[0]),
                wp.vec3(*targets[1]),
                wp.vec4(*left_rotation),
                wp.vec4(*right_rotation),
                float(fingers[0]),
                float(fingers[1]),
                self.tcp_targets,
                self.rotation_targets,
                self.finger_target,
            ],
        )
        if self.graph is None:
            self._simulate_frame()
            if self.model.device.is_cuda and not self.args.no_cuda_graph:
                saved = [self.model.state(), self.model.state()]
                saved[0].assign(self.state_0)
                saved[1].assign(self.state_1)
                saved_ik = [wp.clone(a) for a in (self.ik_q, self.frame_start, self.frame_end)]
                with wp.ScopedCapture() as capture:
                    self._simulate_frame()
                self.state_0.assign(saved[0])
                self.state_1.assign(saved[1])
                for array, backup in zip((self.ik_q, self.frame_start, self.frame_end), saved_ik, strict=True):
                    array.assign(backup)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        self.targets = targets
        # Reading back state synchronizes the device, so track only in tests.
        if self.args.test and self.tracks_freely(self.sim_time):
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

    def render_ui(self, imgui):
        imgui.text("Dynamic W1: MuJoCo joint drives")
        imgui.text(f"MPM feedback: {self.args.coupling}")
        imgui.text(f"Time: {self.sim_time:.2f} s")
        fingers = self.finger_q() * 1000.0
        imgui.text(f"Left / right finger travel: {fingers[0]:.1f} / {fingers[1]:.1f} mm")

    def check_finite(self):
        for array in (
            self.state_0.body_q,
            self.state_0.body_qd,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0.particle_q,
            self.state_0.particle_qd,
            self.state_0.mpm.particle_elastic_strain,
            self.state_0.mpm.particle_stress,
            self.state_0.mpm.particle_Jp,
        ):
            if array is not None and not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")

    @staticmethod
    def add_arguments(parser, *, num_frames):
        parser.set_defaults(num_frames=num_frames)
        parser.add_argument(
            "--coupling",
            choices=("two_way", "one_way"),
            default="two_way",
            help="MuJoCo/MPM feedback. one_way lets targets drive the robot through obstacles.",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=4,
            help="Physics substeps per frame; finer steps stabilize delayed robot-object contact feedback.",
        )
        # Unconverged Jacobi solves can eject squeezed material at coarse substeps.
        parser.add_argument("--iterations", type=int, default=60, help="Maximum MPM solver iterations per substep.")
        parser.add_argument("--mpm-solver", choices=("gs", "jacobi", "cr"), default="jacobi")
        # Resting material keeps Jacobi updates just above the tolerance without
        # changing the state; contact phases stay well above this and use every iteration.
        parser.add_argument(
            "--stall-tolerance",
            type=float,
            default=2.0e-3,
            help="MPM residual below which a solve that stopped improving ends early (0 disables).",
        )
        parser.add_argument("--rigid-iterations", type=int, default=20, help="Rigid-block contact sweeps per substep.")
        parser.add_argument(
            "--rigid-solver",
            choices=("gauss_seidel", "jacobi"),
            default="gauss_seidel",
            help="Rigid-block contact solve: sequential sweeps or parallel sweeps with mass splitting.",
        )
        parser.add_argument(
            "--rigid-material-coupling",
            choices=("implicit", "lagged", "constrained"),
            default="constrained",
            help="Contact-aware material response (default), independent implicit response, or delayed material reaction.",
        )
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=4,
            help="IK iterations per frame, each started from the previous frame's solution.",
        )
        # Squeezed on the coarse grid, up to about 40 particles share a cell; a fixed
        # particle-basis capacity keeps captured frames within it.
        parser.add_argument("--strain-basis", choices=("P0", "P1d", "Q1d", "pic48"), default="pic48")
        parser.add_argument("--arm-kp", type=float, default=12000.0, help="Arm drive stiffness [N*m/rad].")
        parser.add_argument("--arm-kd", type=float, default=350.0, help="Arm drive damping [N*m*s/rad].")
        parser.add_argument("--finger-kp", type=float, default=1000.0, help="Finger drive stiffness [N/m].")
        parser.add_argument("--finger-kd", type=float, default=20.0, help="Finger drive damping [N*s/m].")
        parser.add_argument(
            "--finger-armature", type=float, default=0.3, help="Reflected finger gear inertia [kg] for stable feedback."
        )
        parser.add_argument(
            "--finger-damping",
            type=float,
            default=50.0,
            help="Passive finger joint damping [N*s/m]; unlike drive damping it acts while the effort limit clips.",
        )
        parser.add_argument("--proxy-relaxation", type=float, default=0.5, help="Two-way feedback relaxation.")
        parser.add_argument(
            "--relaxation-mode", choices=("fixed", "aitken"), default="fixed", help="Two-way feedback relaxation mode."
        )
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=1,
            help="MuJoCo/MPM passes per substep; more passes converge contact feedback within the substep.",
        )
        parser.add_argument(
            "--coupling-tolerance",
            type=float,
            default=0.0,
            help="Interface velocity residual [m/s, rad/s] that ends passes early without a CUDA graph.",
        )
        parser.add_argument(
            "--proxy-mass-scale",
            type=float,
            default=4.0,
            help="Scale of the MuJoCo effective inertia given to the MPM link proxies.",
        )
        parser.add_argument(
            "--articulated-response",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Move the robot links through the MuJoCo link mobility in MPM contacts instead of proxy inertia "
            "(two-way coupling only).",
        )
        parser.add_argument("--voxel-size", type=float, default=0.04, help="MPM grid cell size [m].")
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser
