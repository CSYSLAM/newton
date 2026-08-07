#!/usr/bin/env python3
"""Kamino-only demo: pick up and assemble a pre-glued carton.

The carton wall is a closed four-bar linkage.  Four rigid paperboard panels are
connected by four revolute crease joints; the fourth joint closes the kinematic
loop.  Two opposite walls each have a rigid top and bottom closure flap on a
revolute hinge.  An implicit-PD hinge drive approximates the external opener;
the top flap hinges only hold their open pose in this assembly stage.
The first panel is a kinematic carrier for the scripted pick-and-place motion;
the bottom closure flaps have no actuator and close only through gravity and
contact with the ground.  There is no robot, no MuJoCo solver, and no coupled
solver in this example.

The exactly-flat state of a four-bar is a singular configuration, so the demo
starts at ``--initial-angle-deg`` (12 degrees by default).  This represents the
small clearance that a real suction cup or finger creates before opening.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton


PANEL_COLORS = (
    (0.92, 0.60, 0.25),
    (0.96, 0.75, 0.39),
    (0.78, 0.42, 0.18),
    (0.88, 0.54, 0.23),
)

# Use two opposite walls for the top and bottom closure pages.  Body order is
# top/bottom per wall.  Bottom pages are passive; top pages remain held open for
# this assembly stage.
FLAP_PANEL_INDICES = (0, 2)
BOTTOM_FLAP_INDICES = (1, 3)
TOP_FLAP_INDICES = (0, 2)
FLAT_BASE_ANGLE = 0.5 * math.pi


def smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def smoothstep01_derivative(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 6.0 * value * (1.0 - value)


def quat_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_inertia(mass: float, side: float, thickness: float, height: float) -> wp.mat33:
    ixx = mass * (thickness * thickness + height * height) / 12.0
    iyy = mass * (side * side + height * height) / 12.0
    izz = mass * (side * side + thickness * thickness) / 12.0
    return wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


def make_pose(center: np.ndarray, yaw: float) -> wp.transform:
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.float32(yaw))
    return wp.transformf(wp.vec3(*center.astype(np.float32)), rotation)


def make_transform(translation: np.ndarray, rotation: wp.quat) -> wp.transform:
    return wp.transformf(wp.vec3(*translation.astype(np.float32)), rotation)


def local_joint_frame(body_pose: wp.transform, corner: np.ndarray, rotation=None) -> wp.transform:
    if rotation is None:
        rotation = wp.quat_identity(dtype=wp.float32)
    world_joint = wp.transformf(
        wp.vec3(*corner.astype(np.float32)),
        rotation,
    )
    return wp.transform_multiply(wp.transform_inverse(body_pose), world_joint)


def make_flap_pose(hinge_center: np.ndarray, yaw: float, fold_angle: float, depth: float) -> wp.transform:
    panel_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.float32(yaw))
    fold_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.float32(fold_angle))
    rotation = wp.mul(panel_rotation, fold_rotation)

    radial = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offset = 0.5 * depth * (math.cos(fold_angle) * radial + math.sin(fold_angle) * vertical)
    center = hinge_center + offset
    return wp.transformf(wp.vec3(*center.astype(np.float32)), rotation)


def assembly_timeline(args: argparse.Namespace) -> tuple[float, float, float, float, float, float]:
    flat_end = args.flat_duration
    lift_end = flat_end + args.lift_duration
    turn_end = lift_end + args.turn_duration
    open_start = turn_end + args.open_delay
    open_end = open_start + args.open_duration
    place_end = open_end + args.place_duration
    return flat_end, lift_end, turn_end, open_start, open_end, place_end


def assembly_base_pose(args: argparse.Namespace, time_s: float) -> wp.transform:
    ground_height = args.thickness
    flat_z = ground_height + 0.5 * args.thickness
    landing_z = 0.5 * args.height + ground_height
    carry_z = args.carry_height
    flat_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), FLAT_BASE_ANGLE)
    upright_rotation = wp.quat_identity(dtype=wp.float32)
    flat_end, lift_end, turn_end, _open_start, open_end, _place_end = assembly_timeline(args)

    if time_s <= flat_end:
        return make_transform(np.array([0.0, 0.0, flat_z]), flat_rotation)
    if time_s <= lift_end:
        blend = smoothstep01((time_s - flat_end) / args.lift_duration)
        z = flat_z + (carry_z - flat_z) * blend
        return make_transform(np.array([0.0, 0.0, z]), flat_rotation)
    if time_s <= turn_end:
        blend = smoothstep01((time_s - lift_end) / args.turn_duration)
        rotation = wp.quat_slerp(flat_rotation, upright_rotation, blend)
        return make_transform(np.array([0.0, 0.0, carry_z]), rotation)
    if time_s <= open_end:
        return make_transform(np.array([0.0, 0.0, carry_z]), upright_rotation)

    blend = smoothstep01((time_s - open_end) / args.place_duration)
    z = carry_z + (landing_z - carry_z) * blend
    return make_transform(np.array([0.0, 0.0, z]), upright_rotation)


def assembly_base_velocity(args: argparse.Namespace, time_s: float) -> wp.spatial_vector:
    ground_height = args.thickness
    flat_z = ground_height + 0.5 * args.thickness
    landing_z = 0.5 * args.height + ground_height
    carry_z = args.carry_height
    flat_end, lift_end, turn_end, _open_start, open_end, _place_end = assembly_timeline(args)

    if flat_end < time_s < lift_end:
        derivative = smoothstep01_derivative((time_s - flat_end) / args.lift_duration)
        vz = (carry_z - flat_z) * derivative / args.lift_duration
        return wp.spatial_vectorf(0.0, 0.0, vz, 0.0, 0.0, 0.0)
    if lift_end < time_s < turn_end:
        derivative = smoothstep01_derivative((time_s - lift_end) / args.turn_duration)
        wx = -FLAT_BASE_ANGLE * derivative / args.turn_duration
        return wp.spatial_vectorf(0.0, 0.0, 0.0, wx, 0.0, 0.0)
    if open_end < time_s < open_end + args.place_duration:
        derivative = smoothstep01_derivative((time_s - open_end) / args.place_duration)
        vz = (landing_z - carry_z) * derivative / args.place_duration
        return wp.spatial_vectorf(0.0, 0.0, vz, 0.0, 0.0, 0.0)
    return wp.spatial_vectorf(0.0)


def opening_blend(args: argparse.Namespace, time_s: float) -> tuple[float, float]:
    _flat_end, _lift_end, _turn_end, open_start, open_end, _place_end = assembly_timeline(args)
    phase = (time_s - open_start) / args.open_duration
    blend = smoothstep01(phase)
    blend_velocity = (
        smoothstep01_derivative(phase) / args.open_duration
        if 0.0 < phase < 1.0
        else 0.0
    )
    return blend, blend_velocity


class CartonSimulation:
    def __init__(
        self,
        *,
        side: float,
        height: float,
        thickness: float,
        initial_angle: float,
        sim_dt: float,
    ) -> None:
        self.side = side
        self.height = height
        self.thickness = thickness
        self.initial_angle = initial_angle
        self.target_delta = 0.5 * math.pi - initial_angle
        self.sim_dt = sim_dt

        builder = newton.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=wp.vec3(0.0, 0.0, -9.81),
        )
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0
        builder.begin_world(label="pre_glued_carton")

        # A slightly-open parallelogram: at 0 degrees it becomes the singular,
        # perfectly flattened two-layer configuration.
        s = side
        a = initial_angle
        corners_xy = np.array(
            [
                [0.0, 0.0],
                [s, 0.0],
                [s + s * math.cos(a), s * math.sin(a)],
                [s * math.cos(a), s * math.sin(a)],
            ],
            dtype=np.float64,
        )
        corners_xy -= corners_xy.mean(axis=0, keepdims=True)
        corners = np.column_stack((corners_xy, np.full(4, 0.5 * height)))
        next_corners = np.roll(corners, -1, axis=0)
        centers = 0.5 * (corners + next_corners)
        # Author the mechanism in the first panel's local frame.  The free
        # base can then carry the entire carton through the pick-and-place path.
        base_center = centers[0].copy()
        corners -= base_center
        centers -= base_center
        yaws = (0.0, a, math.pi, a + math.pi)

        mass = 0.08
        inertia = box_inertia(mass, side, thickness, height)
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=0.0,
            gap=0.0,
            has_shape_collision=True,
            density=0.0,
        )

        self.body_ids: list[int] = []
        self.body_poses: list[wp.transform] = []
        self.flap_body_ids: list[int] = []
        self.flap_body_poses: list[wp.transform] = []
        shape_ids: list[int] = []
        bottom_flap_shape_ids: list[int] = []
        for index, (center, yaw, color) in enumerate(zip(centers, yaws, PANEL_COLORS, strict=True)):
            pose = make_pose(center, yaw)
            body = builder.add_link(
                label=f"panel_{index}",
                mass=mass,
                inertia=inertia,
                xform=pose,
                lock_inertia=True,
            )
            shape = builder.add_shape_box(
                label=f"paperboard_{index}",
                body=body,
                hx=0.5 * side,
                hy=0.5 * thickness,
                hz=0.5 * height,
                cfg=shape_cfg,
                color=wp.vec3(*color),
            )
            self.body_ids.append(body)
            self.body_poses.append(pose)
            shape_ids.append(shape)

        # Add one top and one bottom closure flap to two opposite walls.  Each
        # flap reaches the center of the carton without overlapping its mate.
        flap_depth = 0.5 * side - thickness
        flap_mass = 0.5 * mass
        flap_inertia = box_inertia(flap_mass, side, flap_depth, thickness)
        bottom_fold_angle = math.radians(78.0)
        flap_specs = (
            ("top", 0.5 * height, 0.5 * math.pi),
            # Start above the bottom crease.  Gravity brings the page through
            # the horizontal closing position, where the ground catches it.
            # A small inward lean avoids the exactly-vertical unstable pose.
            ("bottom", -0.5 * height, bottom_fold_angle),
        )
        for panel_index in FLAP_PANEL_INDICES:
            center = centers[panel_index]
            yaw = yaws[panel_index]
            for flap_label, hinge_z, fold_angle in flap_specs:
                hinge_center = center.copy()
                hinge_center[2] = hinge_z
                pose = make_flap_pose(hinge_center, yaw, fold_angle, flap_depth)
                body = builder.add_link(
                    label=f"{flap_label}_flap_{panel_index}",
                    mass=flap_mass,
                    inertia=flap_inertia,
                    xform=pose,
                    lock_inertia=True,
                )
                shape = builder.add_shape_box(
                    label=f"{flap_label}_paperboard_{panel_index}",
                    body=body,
                    hx=0.5 * side,
                    hy=0.5 * flap_depth,
                    hz=0.5 * thickness,
                    cfg=shape_cfg,
                    color=wp.vec3(*PANEL_COLORS[panel_index]),
                )
                self.flap_body_ids.append(body)
                self.flap_body_poses.append(pose)
                shape_ids.append(shape)
                if flap_label == "bottom":
                    bottom_flap_shape_ids.append(shape)

        # Primitive broad-phase does not support plane shapes, so use a large
        # static box whose top surface is the assembly table at z=0.
        ground_shape = builder.add_shape_box(
            label="assembly_ground",
            body=-1,
            hx=5.0,
            hy=5.0,
            hz=0.5,
            xform=wp.transformf(0.0, 0.0, self.thickness - 0.5, 0.0, 0.0, 0.0, 1.0),
            cfg=shape_cfg,
        )
        # The folded carton is a bonded, layered stack.  Filter duplicate
        # paper-to-paper contacts in that authored state while retaining every
        # paper-to-ground pair needed for the passive bottom closure.
        for shape_a, shape_b in itertools.combinations(shape_ids, 2):
            builder.add_shape_collision_filter_pair(shape_a, shape_b)
        for shape_id in shape_ids:
            if shape_id not in bottom_flap_shape_ids:
                builder.add_shape_collision_filter_pair(shape_id, ground_shape)

        # Panel 0 is the kinematic carrier used only for the scripted lift,
        # reorientation, and placement of the carton.
        base_joint = builder.add_joint_free(
            label="carrier_free_base",
            parent=-1,
            child=self.body_ids[0],
            parent_xform=wp.transform_identity(dtype=wp.float32),
            child_xform=wp.transform_identity(dtype=wp.float32),
            collision_filter_parent=False,
        )
        self.base_joint_id = base_joint

        passive_axis = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Z,
            actuator_mode=newton.JointTargetMode.NONE,
            damping=0.012,
            armature=2.0e-4,
            limit_lower=-math.pi,
            limit_upper=math.pi,
            limit_ke=20.0,
            limit_kd=0.2,
        )
        driven_axis = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.Z,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=0.75,
            target_kd=0.085,
            damping=0.006,
            armature=5.0e-4,
            effort_limit=1.5,
            velocity_limit=4.0,
            limit_lower=-0.08,
            limit_upper=self.target_delta + 0.08,
            limit_ke=30.0,
            limit_kd=0.3,
        )
        flap_axis = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.X,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=0.55,
            target_kd=0.06,
            damping=0.006,
            armature=5.0e-4,
            effort_limit=0.8,
            velocity_limit=8.0,
            limit_lower=-math.pi,
            limit_upper=math.pi,
            limit_ke=30.0,
            limit_kd=0.3,
        )
        passive_flap_axis = newton.ModelBuilder.JointDofConfig(
            axis=newton.Axis.X,
            actuator_mode=newton.JointTargetMode.NONE,
            damping=0.018,
            armature=5.0e-4,
            # The lower stop is the horizontal page-on-floor pose relative to
            # the slightly inward-leaning authored pose.  It keeps the
            # discrete contact solve from tunnelling through the table.
            limit_lower=-bottom_fold_angle,
            limit_upper=math.pi - bottom_fold_angle,
            limit_ke=30.0,
            limit_kd=0.3,
        )

        articulation_joints: list[int] = [base_joint]
        self.hinge_ids: list[int] = []
        for index in range(3):
            parent_index = index
            child_index = (index + 1) % 4
            corner_index = (index + 1) % 4
            hinge = builder.add_joint_revolute(
                label=f"glued_crease_{index}",
                parent=self.body_ids[parent_index],
                child=self.body_ids[child_index],
                parent_xform=local_joint_frame(self.body_poses[parent_index], corners[corner_index]),
                child_xform=local_joint_frame(self.body_poses[child_index], corners[corner_index]),
                axis=driven_axis if index == 0 else passive_axis,
                collision_filter_parent=True,
            )
            self.hinge_ids.append(hinge)
            articulation_joints.append(hinge)

        self.flap_hinge_ids: list[int] = []
        flap_index = 0
        for panel_index in FLAP_PANEL_INDICES:
            for flap_label, hinge_z, _fold_angle in flap_specs:
                hinge_center = centers[panel_index].copy()
                hinge_center[2] = hinge_z
                joint_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaws[panel_index])
                hinge = builder.add_joint_revolute(
                    label=f"closure_flap_hinge_{flap_index}",
                    parent=self.body_ids[panel_index],
                    child=self.flap_body_ids[flap_index],
                    parent_xform=local_joint_frame(
                        self.body_poses[panel_index], hinge_center, rotation=joint_rotation
                    ),
                    child_xform=local_joint_frame(
                        self.flap_body_poses[flap_index], hinge_center, rotation=joint_rotation
                    ),
                    axis=flap_axis if flap_label == "top" else passive_flap_axis,
                    collision_filter_parent=True,
                )
                self.flap_hinge_ids.append(hinge)
                articulation_joints.append(hinge)
                flap_index += 1

        # Hinge 3 remains outside the articulation tree and is the loop-closing
        # bilateral constraint solved by Kamino.
        index = 3
        parent_index = index
        child_index = (index + 1) % 4
        corner_index = (index + 1) % 4
        hinge = builder.add_joint_revolute(
            label=f"glued_crease_{index}",
            parent=self.body_ids[parent_index],
            child=self.body_ids[child_index],
            parent_xform=local_joint_frame(self.body_poses[parent_index], corners[corner_index]),
            child_xform=local_joint_frame(self.body_poses[child_index], corners[corner_index]),
            axis=passive_axis,
            collision_filter_parent=True,
        )
        self.hinge_ids.append(hinge)

        builder.add_articulation(articulation_joints, label="carton_open_chain_tree")
        builder.end_world()

        self.model = builder.finalize(skip_validation_joints=True)
        config = newton.solvers.SolverKamino.Config.from_model(self.model)
        config.use_collision_detector = True
        config.use_fk_solver = True
        config.collision_detector.pipeline = "primitive"
        config.collision_detector.max_contacts = 128
        config.dynamics.preconditioning = True
        config.padmm.primal_tolerance = 1.0e-5
        config.padmm.dual_tolerance = 1.0e-5
        config.padmm.compl_tolerance = 1.0e-5
        config.padmm.max_iterations = 160
        config.padmm.rho_0 = 0.1
        config.padmm.use_acceleration = True
        config.padmm.warmstart_mode = "containers"

        self.solver = newton.solvers.SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # One compile/warm-start step, followed by a clean reset to the authored
        # flattened pose.
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.solver.reset(self.state_0)
        self.drive_q_index = int(self.model.joint_target_q_start.numpy()[self.hinge_ids[0]])
        self.drive_qd_index = int(self.model.joint_qd_start.numpy()[self.hinge_ids[0]])
        self.top_flap_drive_indices = [
            (
                int(self.model.joint_target_q_start.numpy()[hinge]),
                int(self.model.joint_qd_start.numpy()[hinge]),
                0.0,
            )
            for flap_index, hinge in enumerate(self.flap_hinge_ids)
            if flap_index % 2 == 0
        ]
        self.base_q = wp.zeros(shape=(1,), dtype=wp.transformf, device=self.model.device)
        self.base_u = wp.zeros(shape=(1,), dtype=wp.spatial_vectorf, device=self.model.device)
        self.carrier_target_pose = self.body_poses[0]
        self.carrier_target_velocity = wp.spatial_vectorf(0.0)
        self.carton_mass = float(np.sum(self.model.body_mass.numpy()))

    def set_opening_target(
        self,
        target: float,
        target_velocity: float,
    ) -> None:
        self.control.joint_target_q[self.drive_q_index : self.drive_q_index + 1].fill_(target)
        self.control.joint_target_qd[self.drive_qd_index : self.drive_qd_index + 1].fill_(target_velocity)
        for q_index, qd_index, open_angle in self.top_flap_drive_indices:
            self.control.joint_target_q[q_index : q_index + 1].fill_(open_angle)
            self.control.joint_target_qd[qd_index : qd_index + 1].fill_(0.0)

    def set_base_pose(self, pose: wp.transform, velocity: wp.spatial_vector) -> None:
        self.base_q.assign([pose])
        self.base_u.assign([velocity])
        reset_config = newton.solvers.SolverKamino.ResetConfig(
            body_poses=newton.solvers.SolverKamino.ResetConfig.Preserve(),
            body_velocities=newton.solvers.SolverKamino.ResetConfig.Preserve(),
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q),
            base_velocity=newton.solvers.SolverKamino.ResetConfig.FromBaseU(self.base_u),
        )
        self.solver.reset(self.state_0, config=reset_config)

    def set_carrier_target(self, pose: wp.transform, velocity: wp.spatial_vector) -> None:
        self.carrier_target_pose = pose
        self.carrier_target_velocity = velocity

    def apply_carrier_wrench(self) -> None:
        target = np.asarray(self.carrier_target_pose, dtype=np.float64)
        current = self.state_0.body_q.numpy()[self.body_ids[0]].astype(np.float64)
        current_velocity = self.state_0.body_qd.numpy()[self.body_ids[0]].astype(np.float64)
        target_velocity = np.asarray(self.carrier_target_velocity, dtype=np.float64)

        position_error = target[:3] - current[:3]
        rotation_error = quat_to_matrix(target[3:7]) @ quat_to_matrix(current[3:7]).T
        trace = float(np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0))
        angle = math.acos(trace)
        if angle < 1.0e-6:
            rotation_vector = np.zeros(3, dtype=np.float64)
        else:
            axis = np.array(
                [
                    rotation_error[2, 1] - rotation_error[1, 2],
                    rotation_error[0, 2] - rotation_error[2, 0],
                    rotation_error[1, 0] - rotation_error[0, 1],
                ],
                dtype=np.float64,
            )
            rotation_vector = 0.5 * angle / math.sin(angle) * axis

        force = (
            520.0 * position_error
            + 34.0 * (target_velocity[:3] - current_velocity[:3])
            + self.carton_mass * np.array([0.0, 0.0, 9.81])
        )
        torque = 8.0 * rotation_vector + 0.5 * (target_velocity[3:] - current_velocity[3:])
        body_f = self.state_0.body_f.numpy()
        body_f[self.body_ids[0]] = np.concatenate((force, torque)).astype(np.float32)
        self.state_0.body_f.assign(body_f)

    def step(self) -> None:
        self.state_0.clear_forces()
        self.apply_carrier_wrench()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.update_contacts(self.contacts, self.state_0)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> None:
        """Reset the carton to its authored, flat-on-ground pose."""
        self.solver.reset(self.state_0)

    def panel_poses(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.body_ids].astype(np.float64, copy=True)

    def render_poses(self) -> np.ndarray:
        body_ids = self.body_ids + self.flap_body_ids
        return self.state_0.body_q.numpy()[body_ids].astype(np.float64, copy=True)


def panel_endpoints(poses: np.ndarray, side: float) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    for pose in poses:
        center = pose[:3]
        rotation = quat_to_matrix(pose[3:7])
        half_edge = rotation @ np.array([0.5 * side, 0.0, 0.0])
        starts.append(center - half_edge)
        ends.append(center + half_edge)
    return np.asarray(starts), np.asarray(ends)


def opening_angle_deg(poses: np.ndarray) -> float:
    d0 = quat_to_matrix(poses[0, 3:7])[:, 0]
    d1 = quat_to_matrix(poses[1, 3:7])[:, 0]
    dot = float(np.clip(np.dot(d0[:2], d1[:2]), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def closure_gap(poses: np.ndarray, side: float) -> float:
    starts, ends = panel_endpoints(poses, side)
    gaps = [np.linalg.norm(ends[i] - starts[(i + 1) % 4]) for i in range(4)]
    return float(max(gaps))


def set_opening_drive(sim: CartonSimulation, args: argparse.Namespace, time_s: float) -> None:
    blend, blend_velocity = opening_blend(args, time_s)
    sim.set_opening_target(sim.target_delta * blend, sim.target_delta * blend_velocity)


def run_simulation(args: argparse.Namespace) -> tuple[list[np.ndarray], list[dict[str, float]], dict[str, float]]:
    sim = CartonSimulation(
        side=args.side,
        height=args.height,
        thickness=args.thickness,
        initial_angle=math.radians(args.initial_angle_deg),
        sim_dt=args.sim_dt,
    )
    initial_base_pose = assembly_base_pose(args, 0.0)
    initial_base_velocity = assembly_base_velocity(args, 0.0)
    sim.set_base_pose(initial_base_pose, initial_base_velocity)
    sim.set_carrier_target(initial_base_pose, initial_base_velocity)

    output_fps = args.output_fps
    sample_stride = max(1, round((1.0 / output_fps) / args.sim_dt))
    step_count = int(round(args.duration / args.sim_dt))
    poses_history: list[np.ndarray] = [sim.render_poses()]
    metrics: list[dict[str, float]] = []

    for step in range(step_count + 1):
        time_s = step * args.sim_dt
        sim.set_carrier_target(assembly_base_pose(args, time_s), assembly_base_velocity(args, time_s))
        blend, _blend_velocity = opening_blend(args, time_s)
        set_opening_drive(sim, args, time_s)

        if step > 0:
            sim.step()

        if step % sample_stride == 0 or step == step_count:
            poses = sim.panel_poses()
            render_poses = sim.render_poses()
            poses_history.append(render_poses)
            metrics.append(
                {
                    "time_s": time_s,
                    "target_opening_deg": args.initial_angle_deg
                    + math.degrees(sim.target_delta * blend),
                    "measured_opening_deg": opening_angle_deg(poses),
                    "closure_gap_mm": 1000.0 * closure_gap(poses, args.side),
                }
            )

    final = metrics[-1]
    summary = {
        "solver": "newton.solvers.SolverKamino",
        "newton_version": getattr(newton, "__version__", "unknown"),
        "warp_version": getattr(wp, "__version__", "unknown"),
        "device": str(wp.get_device()),
        "robot_present": False,
        "panel_count": 4,
        "closure_flap_count": 4,
        "bottom_flaps_passive": True,
        "assembly_sequence": "flat_lift_turn_open_place_settle",
        "collision_detection": True,
        "closed_loop_hinge_count": 4,
        "initial_angle_deg": args.initial_angle_deg,
        "final_measured_angle_deg": final["measured_opening_deg"],
        "final_closure_gap_mm": final["closure_gap_mm"],
        "side_m": args.side,
        "height_m": args.height,
        "thickness_m": args.thickness,
        "sim_dt_s": args.sim_dt,
        "duration_s": args.duration,
        "carry_height_m": args.carry_height,
    }
    return poses_history[1:], metrics, summary


def run_realtime(args: argparse.Namespace) -> None:
    """Run the carton in the interactive OpenGL viewer."""
    if args.device:
        wp.set_device(args.device)

    sim = CartonSimulation(
        side=args.side,
        height=args.height,
        thickness=args.thickness,
        initial_angle=math.radians(args.initial_angle_deg),
        sim_dt=args.sim_dt,
    )
    initial_base_pose = assembly_base_pose(args, 0.0)
    initial_base_velocity = assembly_base_velocity(args, 0.0)
    sim.set_base_pose(initial_base_pose, initial_base_velocity)
    sim.set_carrier_target(initial_base_pose, initial_base_velocity)

    viewer = newton.viewer.ViewerGL(headless=False, paused=args.paused)
    viewer.set_model(sim.model)
    viewer.vsync = True
    viewer.camera.fov = 45.0
    viewer.set_camera(
        pos=wp.vec3(0.57, -2.49, 0.05),
        pitch=-3.9,
        yaw=98.8,
    )

    step_count = int(round(args.duration / args.sim_dt))
    sample_stride = max(1, round((1.0 / args.output_fps) / args.sim_dt))
    step = 0
    next_frame_time = time.perf_counter()
    frame_period = 1.0 / args.output_fps

    def render_frame(sim_time: float) -> None:
        viewer.begin_frame(sim_time)
        viewer.log_state(sim.state_0)
        if hasattr(viewer, "log_contacts"):
            viewer.log_contacts(sim.contacts, sim.state_1)
        viewer.end_frame()

    try:
        while viewer.is_running():
            if step >= step_count:
                if not args.loop:
                    # Keep the final pose visible until the user closes the window.
                    render_frame(step_count * args.sim_dt)
                    time.sleep(frame_period)
                    continue
                else:
                    sim.reset()
                    sim.set_base_pose(initial_base_pose, initial_base_velocity)
                    sim.set_carrier_target(initial_base_pose, initial_base_velocity)
                    step = 0
                    next_frame_time = time.perf_counter()

            if step < step_count and viewer.should_step():
                for _ in range(sample_stride):
                    if step >= step_count:
                        break
                    time_s = step * args.sim_dt
                    sim.set_carrier_target(assembly_base_pose(args, time_s), assembly_base_velocity(args, time_s))
                    set_opening_drive(sim, args, time_s)
                    sim.step()
                    step += 1

            sim_time = min(step, step_count) * args.sim_dt
            render_frame(sim_time)

            next_frame_time += frame_period
            remaining = next_frame_time - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
    finally:
        viewer.close()


def cuboid_vertices(pose: np.ndarray, dimensions: tuple[float, float, float]) -> np.ndarray:
    side, depth, height = dimensions
    local = np.array(
        [
            [x, y, z]
            for x in (-0.5 * side, 0.5 * side)
            for y in (-0.5 * depth, 0.5 * depth)
            for z in (-0.5 * height, 0.5 * height)
        ],
        dtype=np.float64,
    )
    rotation = quat_to_matrix(pose[3:7])
    return local @ rotation.T + pose[:3]


def render_gif(
    poses_history: list[np.ndarray],
    metrics: list[dict[str, float]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    face_indices = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    frames: list[Image.Image] = []
    radius = 1.55 * args.side
    flap_depth = 0.5 * args.side - args.thickness

    for frame_index, poses in enumerate(poses_history):
        fig = plt.figure(figsize=(7.2, 6.2), dpi=110, facecolor="#f7f3ea")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="3d", facecolor="#f7f3ea")

        for body_index, pose in enumerate(poses):
            is_flap = body_index >= 4
            dimensions = (
                (args.side, flap_depth, args.thickness)
                if is_flap
                else (args.side, args.thickness, args.height)
            )
            verts = cuboid_vertices(pose, dimensions)
            faces = [[verts[index] for index in face] for face in face_indices]
            flap_group = (body_index - 4) // 2
            color_index = FLAP_PANEL_INDICES[flap_group] if is_flap else body_index
            poly = Poly3DCollection(
                faces,
                facecolor=(*PANEL_COLORS[color_index], 0.86),
                edgecolor="#5a371f",
                linewidth=1.15,
            )
            ax.add_collection3d(poly)

        # Hinge/crease lines make the glued closed loop visually explicit.
        starts, ends = panel_endpoints(poses[:4], args.side)
        hinge_points = 0.5 * (ends + np.roll(starts, -1, axis=0))
        for point in hinge_points:
            ax.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [0.0, args.height],
                color="#2c2017",
                linewidth=2.0,
                alpha=0.85,
            )

        metric = metrics[min(frame_index, len(metrics) - 1)]
        ax.text2D(
            0.04,
            0.95,
            "Kamino-only · pre-glued carton with closure flaps",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
            color="#33251d",
        )
        ax.text2D(
            0.04,
            0.89,
            f"t = {metric['time_s']:.2f} s    opening = {metric['measured_opening_deg']:.1f}°",
            transform=ax.transAxes,
            fontsize=10,
            color="#5a4233",
        )
        state_label = (
            "FLAT-FOLDED"
            if metric["measured_opening_deg"] < 22.0
            else "OPENING"
            if metric["measured_opening_deg"] < 87.0
            else "SQUARE / HELD"
        )
        ax.text2D(
            0.04,
            0.83,
            state_label,
            transform=ax.transAxes,
            fontsize=10,
            color="#a1501f",
            weight="bold",
        )

        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-0.6 * args.side, 1.6 * args.height)
        ax.set_box_aspect((2.0 * radius, 2.0 * radius, 2.2 * args.height))
        ax.view_init(elev=26.0, azim=-58.0)
        ax.set_axis_off()
        fig.tight_layout(pad=0.25)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy()))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(round(1000.0 / args.output_fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    frames[-1].save(output_path.with_name("carton_final.png"))


def write_metrics(metrics: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--side", type=float, default=0.24, help="Panel width / final cube side [m].")
    parser.add_argument("--height", type=float, default=0.24, help="Carton wall height [m].")
    parser.add_argument("--thickness", type=float, default=0.004, help="Paperboard thickness [m].")
    parser.add_argument("--initial-angle-deg", type=float, default=12.0)
    parser.add_argument("--sim-dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--flat-duration", type=float, default=1.0, help="Time resting flat on the ground [s].")
    parser.add_argument("--lift-duration", type=float, default=1.0, help="Time to lift the flat carton [s].")
    parser.add_argument("--turn-duration", type=float, default=1.2, help="Time to rotate the carton upright [s].")
    parser.add_argument("--open-delay", type=float, default=0.25, help="Pause after standing the carton before opening [s].")
    parser.add_argument("--open-duration", type=float, default=2.0, help="Time to open the four-bar walls [s].")
    parser.add_argument("--place-duration", type=float, default=1.5, help="Time to put the carton back on the ground [s].")
    parser.add_argument("--carry-height", type=float, default=0.34, help="Height of the scripted carrier pose [m].")
    parser.add_argument("--output-fps", type=int, default=24)
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cuda:0 or cpu.")
    parser.add_argument("--live", action="store_true", help="Open a real-time OpenGL simulation window.")
    parser.add_argument("--loop", action="store_true", help="Restart the simulation after it reaches --duration.")
    parser.add_argument("--paused", action="store_true", help="Start the real-time viewer paused.")
    parser.add_argument("--no-gif", action="store_true", help="Run physics and metrics only.")
    args = parser.parse_args()
    if args.duration is None:
        args.duration = assembly_timeline(args)[-1] + 1.5
    return args


def main() -> None:
    args = parse_args()
    if args.live:
        run_realtime(args)
        return

    if args.device:
        wp.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses_history, metrics, summary = run_simulation(args)

    write_metrics(metrics, args.output_dir / "metrics.csv")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    if not args.no_gif:
        render_gif(poses_history, metrics, args, args.output_dir / "carton_kamino.gif")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
