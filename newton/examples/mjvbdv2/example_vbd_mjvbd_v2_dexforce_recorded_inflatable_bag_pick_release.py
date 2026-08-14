# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay the recorded inflatable-bag grasp with the full Dexforce W1.

The right arm samples the isolated-hand example's hand-root trajectory at the
same simulation time and only then converts that pose to the full-W1 wrist TCP.
This preserves every commanded hand-root waypoint and interpolation exactly;
IK supplies the corresponding full-arm joint motion. The finger targets,
contact-aware closing speed, pneumatic bag, table, and release material also
match the isolated-hand example.

CUDA devices capture the warmed physics substeps by default. Pass
``--no-graph-capture`` to use direct kernel launches instead.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_dexforce_recorded_inflatable_bag_pick_release --viewer gl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release as hand_reference,
)
from newton.solvers import SolverMJVBDV2

FPS = 60
INITIAL_IK_ITERATIONS = 240
RUNTIME_IK_ITERATIONS = 24
END_EFFECTOR_POSITION_TOLERANCE = 5.0e-4
END_EFFECTOR_ANGLE_TOLERANCE_DEG = 0.25

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
RIGHT_J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
RIGHT_J7_TO_HAND_BASE_ROTATION = wp.quat(0.5, -0.5, 0.5, 0.5)

WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
CAMERA_POS = wp.vec3(0.37, -3.15, 1.57)
CAMERA_FOV = 45.0
CAMERA_PITCH = -23.3
CAMERA_YAW = 150.4
DEFAULT_HOUSE_USD = (
    "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/"
    "House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
)

_SOFT_MATERIAL_GRASP = 0
_SOFT_MATERIAL_RELEASE = 1


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    joint_coord = wp.tid()
    out[joint_coord] = q0[joint_coord] * (1.0 - alpha) + q1[joint_coord] * alpha


@wp.kernel
def _joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    joint_dof = wp.tid()
    out[joint_dof] = (q1[joint_dof] - q0[joint_dof]) * inv_dt


@wp.kernel
def _lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    index = wp.tid()
    q[0, indices[index]] = values[index]


@wp.kernel
def _accumulate_contact_diagnostics(
    soft_contact_count: wp.array[int],
    body_particle_contact_overflow: wp.array[int],
    maximum_soft_contact_count: wp.array[int],
    maximum_body_particle_contact_count: wp.array[int],
):
    if wp.tid() == 0:
        maximum_soft_contact_count[0] = wp.max(maximum_soft_contact_count[0], soft_contact_count[0])
        maximum_body_particle_contact_count[0] = wp.max(
            maximum_body_particle_contact_count[0], body_particle_contact_overflow[0]
        )


@wp.kernel
def _copy_joint_q(source: wp.array[float], target: wp.array[float]):
    joint_coord = wp.tid()
    target[joint_coord] = source[joint_coord]


@wp.kernel
def _limit_right_finger_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    desired_finger_q: wp.array[float],
    soft_contact_count: wp.array[int],
    soft_contact_shape: wp.array[int],
    right_hand_shape_mask: wp.array[int],
    free_max_step: float,
    contact_max_step: float,
    target_q: wp.array[float],
):
    finger = wp.tid()
    active_contact_count = wp.min(soft_contact_count[0], soft_contact_shape.shape[0])
    hand_contact = bool(False)
    for contact in range(active_contact_count):
        shape = soft_contact_shape[contact]
        if shape >= 0 and shape < right_hand_shape_mask.shape[0] and right_hand_shape_mask[shape] != 0:
            hand_contact = True

    max_step = free_max_step
    if hand_contact:
        max_step = contact_max_step
    q_index = finger_q_indices[finger]
    delta = wp.clamp(desired_finger_q[finger] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


class Example:
    """Track the isolated-hand pneumatic-bag trajectory with the full W1."""

    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_SUFFIXES = (
        "HAND_THUMB2",
        "HAND_THUMB1",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )
    HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")

    _copy_transform = staticmethod(hand_reference.Example._copy_transform)
    _transform_duration = staticmethod(hand_reference.Example._transform_duration)
    _interpolate_transform = staticmethod(hand_reference.Example._interpolate_transform)

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = hand_reference.recorder.SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd
        self.recorded_grasp = hand_reference.Example._load_recorded_grasp(args.grasp_keyframe)
        self.contact_phase = None

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options=self._solver_vbd_options(),
            collision_options=self._solver_collision_options(),
        )
        self.contacts = self.solver.contacts
        self.maximum_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.maximum_body_particle_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)

        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self._build_ik()
        self.phases = hand_reference.Example._build_phases(self)
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._right_hand_q()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self._build_joint_target_cache()

        self._raise_pressure_limit()
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)

        right_hand_shape_mask = np.zeros(self.model.shape_count, dtype=np.int32)
        right_hand_shape_mask[self.right_hand_shapes] = 1
        self.right_hand_shape_mask = wp.array(
            right_hand_shape_mask,
            dtype=wp.int32,
            device=self.device,
        )
        self.desired_finger_q = wp.zeros(
            self.hand_indices.shape[0],
            dtype=wp.float32,
            device=self.device,
        )
        self.max_finger_step = float(np.radians(hand_reference.recorder.MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self.max_finger_contact_step = float(
            np.radians(hand_reference.recorder.MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt
        )

        contact_materials = np.asarray(
            (
                (
                    hand_reference.recorder.CONTACT_KE,
                    hand_reference.recorder.CONTACT_KD,
                    hand_reference.recorder.CONTACT_MU,
                ),
                (
                    hand_reference.recorder.CONTACT_KE,
                    hand_reference.recorder.CONTACT_KD,
                    hand_reference.RELEASE_FRICTION,
                ),
            ),
            dtype=np.float32,
        )
        self.soft_contact_materials = wp.array(
            contact_materials,
            dtype=wp.vec3,
            device=self.device,
        )
        self.soft_contact_material_index = wp.full(
            1,
            _SOFT_MATERIAL_GRASP,
            dtype=wp.int32,
            device=self.device,
        )
        self.solver.vbd_solver.set_soft_contact_material_source(
            self.soft_contact_materials,
            self.soft_contact_material_index,
        )

        self.graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.release_material_applied = False
        self.active_phase_index = -1
        self.active_phase_name = "initialization"
        self.current_target_root = self._copy_transform(hand_reference.recorder.INITIAL_HAND_ROOT)
        self.initial_bag_center_z = self._bag_center_z()
        self.lifted_bag_center_z: float | None = None
        self.minimum_volume_ratio = 1.0
        self.maximum_pressure = hand_reference.recorder.BAG_REFERENCE_ABSOLUTE_PRESSURE
        self.maximum_root_position_error = 0.0
        self.maximum_root_angle_error = 0.0
        self.script_duration = sum(phase.duration for phase in self.phases)

        self._attach_house_usd()
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV
        self._validate_mount_mapping()
        self._validate_initial_pose()

    def _solver_vbd_options(self):
        """Match the isolated pneumatic-bag solver configuration."""

        return {
            "iterations": hand_reference.recorder.VBD_ITERATIONS,
            "rigid_body_contact_buffer_size": hand_reference.recorder.RIGID_BODY_CONTACT_BUFFER_SIZE,
            "rigid_body_particle_contact_buffer_size": (
                hand_reference.recorder.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE
            ),
            "particle_enable_self_contact": False,
        }

    def _solver_collision_options(self):
        """Match the isolated pneumatic-bag collision pipeline."""

        return {
            "broad_phase": "nxn",
            "soft_contact_margin": hand_reference.recorder.SOFT_CONTACT_MARGIN,
            "enable_rigid_soft_full_surface_contact": True,
        }

    def _robot_urdf(self) -> Path:
        """Return the configured full Dexforce W1 URDF."""

        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = Path(__file__).resolve().parents[3] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"
        if path.is_file():
            return path
        raise FileNotFoundError("Dexforce W1 URDF is unavailable; pass --robot-urdf PATH.")

    def _build_scene(self):
        """Build the full W1 around the original table and pneumatic bag."""

        recorder = hand_reference.recorder
        hand_recorder = recorder.hand_recorder
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = recorder.CONTACT_KE
        builder.default_shape_cfg.kd = recorder.CONTACT_KD
        builder.default_shape_cfg.mu = recorder.CONTACT_MU
        builder.default_shape_cfg.margin = recorder.SHAPE_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(articulation_start, builder.articulation_count))
        if not self.robot_articulations:
            raise RuntimeError("Dexforce W1 URDF did not create an articulation.")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=recorder.CONTACT_KE,
            kd=recorder.CONTACT_KD,
            mu=0.9,
            margin=recorder.SHAPE_CONTACT_MARGIN,
            is_visible=bool(self.args.show_physics_table),
        )
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(hand_recorder.TABLE_POS, hand_recorder.TABLE_ROTATION),
            hx=hand_recorder.TABLE_HALF_EXTENTS[0],
            hy=hand_recorder.TABLE_HALF_EXTENTS[1],
            hz=hand_recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="full_w1_inflatable_bag_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="full_w1_inflatable_bag_ground")

        bag_mesh = recorder._load_chip_bag_mesh()
        bag_rest_volume = recorder._scaled_mesh_volume(bag_mesh, recorder.BAG_SCALE)
        self.pneumatic_mode_name = self.args.pneumatic_mode
        self.pneumatic_config = recorder._make_pneumatic_config(
            self.pneumatic_mode_name,
            bag_rest_volume,
        )
        self.bag_particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=recorder.BAG_CENTER,
            rot=recorder.BAG_ROTATION,
            scale=recorder.BAG_SCALE,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=recorder.BAG_DENSITY,
            tri_ke=recorder.BAG_TRI_KE,
            tri_ka=recorder.BAG_TRI_KA,
            tri_kd=recorder.BAG_TRI_KD,
            edge_ke=recorder.BAG_EDGE_KE,
            edge_kd=recorder.BAG_EDGE_KD,
            particle_radius=recorder.BAG_PARTICLE_RADIUS,
            validate_mesh=True,
            label="full_w1_graspable_sealed_chip_bag",
            config=self.pneumatic_config,
        )
        self.bag_particle_end = builder.particle_count
        bag_triangle_indices = np.asarray(bag_mesh.indices, dtype=np.int32) + self.bag_particle_start

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        self.hand_shapes = []
        self.right_hand_shapes = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            right_hand_shape = "right" in label and any(keyword in label for keyword in self.HAND_CONTACT_KEYWORDS)
            if right_hand_shape:
                self.hand_shapes.append(shape)
                self.right_hand_shapes.append(shape)
                builder.shape_flags[shape] |= collide_shapes | collide_particles
            else:
                builder.shape_flags[shape] &= ~(collide_shapes | collide_particles)
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = recorder.CONTACT_KE
        self.model.soft_contact_kd = recorder.CONTACT_KD
        self.model.soft_contact_mu = recorder.CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[self.right_hand_shapes] = recorder.CONTACT_MU
        shape_ke[self.right_hand_shapes] = recorder.CONTACT_KE
        shape_kd[self.right_hand_shapes] = recorder.CONTACT_KD
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)

        self.bag_triangle_indices = wp.array(
            bag_triangle_indices,
            dtype=wp.int32,
            device=self.model.device,
        )

    def _build_ik(self):
        """Build the independent full-W1 arm IK model."""

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = builder.finalize(device=self.model.device)
        left = self._body_index(self.ik_model.body_label, "left_j7")
        right = self._body_index(self.ik_model.body_label, "right_j7")
        self.left_obj = ik.IKObjectivePosition(
            left,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.right_home)], dtype=wp.vec3, device=self.device),
        )
        self.right_rot = ik.IKObjectiveRotation(
            right,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.right_home))], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._joint_limits()
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _root_to_tcp(self, root_transform: wp.transform) -> wp.transform:
        """Convert an isolated hand-root pose to the full-W1 wrist TCP."""

        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = self._quat_mul(
            hand_rotation,
            wp.quat_inverse(RIGHT_J7_TO_HAND_BASE_ROTATION),
        )
        target_offset = TCP_OFFSET - RIGHT_J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _tcp_to_root(self, tcp_transform: wp.transform) -> wp.transform:
        """Recover the isolated hand-root pose from a full-W1 wrist TCP."""

        tcp_position = wp.transform_get_translation(tcp_transform)
        wrist_rotation = wp.transform_get_rotation(tcp_transform)
        root_position = tcp_position + wp.quat_rotate(
            wrist_rotation,
            RIGHT_J7_TO_HAND_BASE_OFFSET - TCP_OFFSET,
        )
        root_rotation = self._quat_mul(
            wrist_rotation,
            RIGHT_J7_TO_HAND_BASE_ROTATION,
        )
        return wp.transform(root_position, root_rotation)

    def _sample_hand_trajectory(self, time_s: float):
        """Sample the original root, finger target, and phase without rebuilding it."""

        return hand_reference.Example._sample(self, time_s)

    def _right_hand_q(self):
        """Return the canonical open and recorded grasp finger configurations."""

        q_start = self.model.joint_q_start.numpy()
        indices = []
        open_q = []
        grasp_q = []
        for suffix in self.HAND_SUFFIXES:
            name = f"RIGHT_{suffix}"
            joint = self._joint_index(name)
            indices.append(int(q_start[joint]))
            open_q.append(np.radians(hand_reference.OPEN_JOINTS[name]))
            grasp_q.append(np.radians(self.recorded_grasp.joints_degrees[name]))
        self.hand_start = wp.array(open_q, dtype=wp.float32, device=self.device)
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(open_q, dtype=wp.float32, device=self.device),
            wp.array(grasp_q, dtype=wp.float32, device=self.device),
        )

    def _build_joint_target_cache(self):
        """Initialize W1 at the isolated example's exact first hand pose."""

        initial_root = hand_reference.recorder.INITIAL_HAND_ROOT
        initial_tcp = self._root_to_tcp(initial_root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(initial_tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(initial_tcp)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        wp.launch(
            _lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        initial_q = self.model.joint_q.numpy()
        initial_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        initial_q[self.hand_indices.numpy()] = self.hand_start.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _raise_pressure_limit(self):
        """Preserve target-volume response throughout the physical grasp."""

        pressure_limit = self.model.pneumatic.max_absolute_pressure.numpy()
        pressure_limit[self.cavity.cavity_index] = hand_reference.DEMO_MAX_ABSOLUTE_PRESSURE
        self.model.pneumatic.max_absolute_pressure.assign(pressure_limit)

    def _validate_mount_mapping(self):
        """Verify every phase maps to TCP and back without changing its root pose."""

        for phase in self.phases:
            for root in (phase.root_start, phase.root_end):
                recovered = self._tcp_to_root(self._root_to_tcp(root))
                position_error, angle_error = self._transform_error(recovered, root)
                if position_error > 1.0e-6 or angle_error > np.radians(1.0e-4):
                    raise ValueError(
                        "The W1 wrist mount transform changes the isolated hand-root trajectory: "
                        f"position error={position_error:.9f} m, angle error={np.degrees(angle_error):.9f}°."
                    )

    def _validate_initial_pose(self):
        """Verify the initial arm, fingers, and cavity match the isolated scene."""

        actual_root = self._actual_hand_root()
        target_root = hand_reference.recorder.INITIAL_HAND_ROOT
        position_error, angle_error = self._transform_error(actual_root, target_root)
        if position_error > END_EFFECTOR_POSITION_TOLERANCE:
            raise ValueError(f"Initial W1 hand-root position error is {position_error:.6f} m.")
        if angle_error > np.radians(END_EFFECTOR_ANGLE_TOLERANCE_DEG):
            raise ValueError(f"Initial W1 hand-root angle error is {np.degrees(angle_error):.6f}°.")

        joint_q = self.state_0.joint_q.numpy()
        for suffix, q_index in zip(self.HAND_SUFFIXES, self.hand_indices.numpy(), strict=True):
            name = f"RIGHT_{suffix}"
            actual_degrees = float(np.degrees(joint_q[q_index]))
            expected_degrees = hand_reference.OPEN_JOINTS[name]
            if abs(actual_degrees - expected_degrees) > 1.0e-4:
                raise ValueError(f"Initial joint {name} is {actual_degrees:.6f}°, expected {expected_degrees:.6f}°.")

        cavity_index = self.cavity.cavity_index
        initial_volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        if abs(initial_volume / self.cavity.rest_volume - 1.0) > 1.0e-4:
            raise ValueError("Initial inflatable-bag volume does not match its authored rest volume.")

    def _actual_hand_root(self) -> wp.transform:
        """Return the full robot's current ``right_hand_base`` world pose."""

        body_transform = wp.transform(*self.state_0.body_q.numpy()[self.right_body])
        wrist_position = wp.transform_get_translation(body_transform)
        wrist_rotation = wp.transform_get_rotation(body_transform)
        root_position = wrist_position + wp.quat_rotate(
            wrist_rotation,
            RIGHT_J7_TO_HAND_BASE_OFFSET,
        )
        root_rotation = self._quat_mul(
            wrist_rotation,
            RIGHT_J7_TO_HAND_BASE_ROTATION,
        )
        return wp.transform(root_position, root_rotation)

    @staticmethod
    def _transform_error(actual: wp.transform, target: wp.transform) -> tuple[float, float]:
        """Return translation [m] and shortest-angle [rad] transform errors."""

        actual_position = np.asarray(wp.transform_get_translation(actual), dtype=np.float64)
        target_position = np.asarray(wp.transform_get_translation(target), dtype=np.float64)
        position_error = float(np.linalg.norm(actual_position - target_position))
        actual_rotation = np.asarray(wp.transform_get_rotation(actual), dtype=np.float64)
        target_rotation = np.asarray(wp.transform_get_rotation(target), dtype=np.float64)
        actual_rotation /= max(float(np.linalg.norm(actual_rotation)), 1.0e-12)
        target_rotation /= max(float(np.linalg.norm(target_rotation)), 1.0e-12)
        cosine = float(np.clip(abs(np.dot(actual_rotation, target_rotation)), 0.0, 1.0))
        return position_error, 2.0 * float(np.arccos(cosine))

    def _bag_center_z(self) -> float:
        """Return the current inflatable-bag center height [m]."""

        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        return float(np.mean(positions[:, 2]))

    def _contact_counts(self) -> tuple[int, int]:
        """Return current right-hand-to-bag and total rigid-soft contacts."""

        total = int(self.contacts.soft_contact_count.numpy()[0])
        shape_indices = self.contacts.soft_contact_shape.numpy()
        active = min(total, shape_indices.shape[0])
        shape_mask = self.right_hand_shape_mask.numpy()
        active_shapes = shape_indices[:active]
        valid = (active_shapes >= 0) & (active_shapes < shape_mask.shape[0])
        safe_shapes = np.clip(active_shapes, 0, shape_mask.shape[0] - 1)
        hand_contacts = int(np.count_nonzero(valid & (shape_mask[safe_shapes] != 0)))
        return hand_contacts, total

    def _apply_release_material(self):
        """Use the isolated example's release friction once opening begins."""

        if self.release_material_applied:
            return
        friction = self.model.shape_material_mu.numpy()
        friction[self.right_hand_shapes] = hand_reference.RELEASE_FRICTION
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_mu = hand_reference.RELEASE_FRICTION
        self.soft_contact_material_index.fill_(_SOFT_MATERIAL_RELEASE)
        self.release_material_applied = True

    def _enter_phase(self, phase_index: int):
        """Apply one-time diagnostics and material changes at phase boundaries."""

        if phase_index == self.active_phase_index:
            return
        self.active_phase_index = phase_index
        phase = self.phases[phase_index]
        self.active_phase_name = phase.name
        if phase.name == "release":
            self.lifted_bag_center_z = self._bag_center_z()
        if phase.release:
            self._apply_release_material()

    def _prepare_frame(self):
        """Solve the arm target sampled directly from the isolated trajectory."""

        script_time = self.sim_time * self.args.trajectory_time_scale
        root, finger_joints, phase_index = self._sample_hand_trajectory(script_time)
        self.current_target_root = root
        self._enter_phase(phase_index)

        tcp = self._root_to_tcp(root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(tcp)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=RUNTIME_IK_ITERATIONS)
        wp.launch(
            _lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.launch(
            _copy_joint_q,
            self.model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.device,
        )
        self.desired_finger_q.assign([np.radians(finger_joints[f"RIGHT_{suffix}"]) for suffix in self.HAND_SUFFIXES])
        wp.launch(
            _limit_right_finger_target_step,
            self.hand_indices.shape[0],
            [
                self.frame_q_start,
                self.hand_indices,
                self.desired_finger_q,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.right_hand_shape_mask,
                self.max_finger_step,
                self.max_finger_contact_step,
                self.frame_q_end,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance one full-W1 frame with the isolated scene's substep count."""

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity,
                self.ik_model.joint_dof_count,
                [self.frame_q_start, self.frame_q_end, 1.0 / self.frame_dt, self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(
                _accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            if self.sim_substeps % 2 != 0 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_simulation_graph(self):
        """Capture the warmed, fixed physics substeps as one CUDA graph."""

        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}.")

    def step(self):
        """Advance one frame using the exact source target and warmed graph."""

        self._prepare_frame()
        if self.graph is None:
            self._simulate_substeps()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        """Render the complete W1 and the pneumatic bag surface."""

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/inflatable_bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify finite pneumatic state and accurate hand-root tracking."""

        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        joint_q = self.state_0.joint_q.numpy()
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        volume_ratio = volume / self.cavity.rest_volume
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(joint_q))
        assert np.isfinite(volume_ratio) and volume_ratio > 0.70
        assert np.isfinite(pressure) and 0.0 < pressure <= hand_reference.DEMO_MAX_ABSOLUTE_PRESSURE + 1.0
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, volume_ratio)
        self.maximum_pressure = max(self.maximum_pressure, pressure)

        position_error, angle_error = self._transform_error(
            self._actual_hand_root(),
            self.current_target_root,
        )
        self.maximum_root_position_error = max(self.maximum_root_position_error, position_error)
        self.maximum_root_angle_error = max(self.maximum_root_angle_error, angle_error)
        assert position_error <= END_EFFECTOR_POSITION_TOLERANCE, (
            f"W1 hand-root position error is {position_error:.6f} m."
        )
        assert angle_error <= np.radians(END_EFFECTOR_ANGLE_TOLERANCE_DEG), (
            f"W1 hand-root angle error is {np.degrees(angle_error):.6f}°."
        )
        if self.active_phase_name == "validate_initial":
            hand_contacts, _ = self._contact_counts()
            assert hand_contacts == 0, f"Initial W1 hand pose unexpectedly has {hand_contacts} bag contacts."

    def test_final(self):
        """Verify trajectory fidelity, physical lift, release, and bag volume."""

        assert not any("physical_pad" in label for label in self.model.shape_label)
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        if self.use_graph:
            assert self.graph is not None, "The warmed CUDA physics graph was not captured."

        script_time = self.sim_time * self.args.trajectory_time_scale
        if script_time + self.frame_dt * self.args.trajectory_time_scale < self.script_duration:
            return
        assert self.lifted_bag_center_z is not None, "The scripted lift phase did not complete."
        assert self.lifted_bag_center_z > self.initial_bag_center_z + 0.010, (
            f"The bag was not lifted: initial z={self.initial_bag_center_z:.6f}, "
            f"lifted z={self.lifted_bag_center_z:.6f}."
        )
        final_bag_center_z = self._bag_center_z()
        assert final_bag_center_z < self.lifted_bag_center_z - 0.008, (
            f"The released bag did not fall: lifted z={self.lifted_bag_center_z:.6f}, final z={final_bag_center_z:.6f}."
        )
        assert self.minimum_volume_ratio > 0.85, (
            f"The grasp compressed the target-volume bag excessively: minimum ratio={self.minimum_volume_ratio:.6f}."
        )

        joint_q = self.state_0.joint_q.numpy()
        maximum_open_error = max(
            abs(float(np.degrees(joint_q[q_index])) - hand_reference.OPEN_JOINTS[f"RIGHT_{suffix}"])
            for suffix, q_index in zip(self.HAND_SUFFIXES, self.hand_indices.numpy(), strict=True)
        )
        assert maximum_open_error < 2.0, f"The hand did not reopen fully: error={maximum_open_error:.3f}°."

    def _joint_limits(self):
        """Lock non-arm coordinates in the IK objective limits."""

        lower = self.model.joint_limit_lower.numpy().copy()
        upper = self.model.joint_limit_upper.numpy().copy()
        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for joint, label in enumerate(self.model.joint_label[: self.ik_model.joint_count]):
            if label not in controlled:
                lower[int(qd_start[joint])] = q[int(q_start[joint])] - 1.0e-4
                upper[int(qd_start[joint])] = q[int(q_start[joint])] + 1.0e-4
        return lower[: self.ik_model.joint_dof_count], upper[: self.ik_model.joint_dof_count]

    def _locked_q(self):
        """Return non-arm coordinates held fixed after every IK solve."""

        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices = [
            int(q_start[joint])
            for joint, label in enumerate(self.model.joint_label[: self.ik_model.joint_count])
            if label not in controlled
        ]
        return wp.array(indices, dtype=wp.int32, device=self.device), wp.array(
            [q[index] for index in indices], dtype=wp.float32, device=self.device
        )

    def _joint_index(self, name: str) -> int:
        """Return a model joint index from its unprefixed asset name."""

        return next(index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        """Return a body index from its unprefixed asset name."""

        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    def _tcp(self, state: newton.State, body: int) -> wp.transform:
        """Return one wrist TCP world transform."""

        body_transform = wp.transform(*state.body_q.numpy()[body])
        body_rotation = wp.transform_get_rotation(body_transform)
        return wp.transform(
            wp.transform_get_translation(body_transform) + wp.quat_rotate(body_rotation, TCP_OFFSET),
            body_rotation,
        )

    def _attach_house_usd(self):
        """Attach the visual-only WAIC house background when available."""

        if not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.isfile(self.house_visual_usd):
            print(f"WAIC house USD not found; continuing without it: {self.house_visual_usd}")
            return
        prim = self.viewer.stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    @staticmethod
    def _normal_quat(value: wp.quat) -> wp.quat:
        """Return a normalized quaternion."""

        array = np.asarray([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
        array /= max(float(np.linalg.norm(array)), 1.0e-8)
        return wp.quat(*array)

    @staticmethod
    def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
        """Multiply two quaternions without relying on expression overloads."""

        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _v4(value: wp.quat) -> wp.vec4:
        """Convert a quaternion to the IK rotation target type."""

        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def create_parser():
        """Create full-W1 options plus the canonical pneumatic-bag inputs."""

        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720, paused=False)
        parser.add_argument("--robot-urdf", default=None, help="Optional Dexforce W1 URDF path.")
        parser.add_argument(
            "--house-visual-usd",
            default=DEFAULT_HOUSE_USD,
            help="Optional WAIC house USD reference; it is visual-only.",
        )
        parser.add_argument(
            "--show-physics-table",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render the physical table collider.",
        )
        parser.add_argument("--trajectory-time-scale", type=float, default=1.0)
        parser.add_argument(
            "--pneumatic-mode",
            choices=tuple(hand_reference.recorder.PNEUMATIC_MODES),
            default="target-volume",
            help="Pressure law for the sealed bag (default: %(default)s).",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics substeps as one CUDA graph.",
        )
        parser.add_argument(
            "--grasp-keyframe",
            default=str(hand_reference.DEFAULT_GRASP_KEYFRAME),
            help="Inflatable-bag grasp keyframe used by the isolated-hand trajectory.",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(WAIC_ROBOT_BASE_QUAT[3]))
        return parser


def main():
    """Run the full-W1 recorded inflatable-bag pick-and-release demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
