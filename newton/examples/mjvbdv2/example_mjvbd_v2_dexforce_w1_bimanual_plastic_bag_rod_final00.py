# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Transfer a loaded supermarket bag from a rod to the full Dexforce W1.

This example preserves the standalone-hand root and finger targets from
``example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff.py``. The
saved left and right hand-root poses are converted to the corresponding W1
wrist targets, then both arms are solved with realtime IK before every
displayed frame. The invisible rod still releases the bag at frame 80, and
the hands still begin their bounded forward-and-upward carry at frame 120.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_dexforce_w1_bimanual_plastic_bag_rod_final00 --viewer gl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2 import (
    example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff as hand_reference,
)
from newton.solvers import SolverMJVBDV2

ROBOT_URDF = hand_reference.ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"
ROBOT_BASE_POSITION = wp.vec3(0.0, -0.28, -0.18)
ROBOT_BASE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)

INITIAL_IK_ITERATIONS = 400
RUNTIME_IK_ITERATIONS = 40
END_EFFECTOR_POSITION_TOLERANCE = 5.0e-4
END_EFFECTOR_ANGLE_TOLERANCE_DEG = 0.25
SOFT_CONTACT_MAX = 32768

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
J7_TO_HAND_BASE_ROTATIONS = {
    "LEFT": wp.quat(-0.5, 0.5, 0.5, 0.5),
    "RIGHT": wp.quat(0.5, -0.5, 0.5, 0.5),
}

CAMERA_POSITION = wp.vec3(1.15, -1.40, 1.02)
CAMERA_PITCH = -12.0
CAMERA_YAW = 132.0


@wp.kernel
def _copy_joint_q(source: wp.array[float], target: wp.array[float]):
    """Copy the full-W1 coordinates into the scene target."""
    coordinate = wp.tid()
    target[coordinate] = source[coordinate]


@wp.kernel
def _set_indexed_joint_q(
    indices: wp.array[int],
    values: wp.array[float],
    target: wp.array[float],
):
    """Write the saved finger coordinates into the scene target."""
    index = wp.tid()
    target[indices[index]] = values[index]


@wp.kernel
def _lock_ik_q(
    q: wp.array2d[float],
    indices: wp.array[int],
    values: wp.array[float],
):
    """Restore non-arm coordinates after one IK solve."""
    index = wp.tid()
    q[0, indices[index]] = values[index]


class Example:
    """Track the rod-handoff hand targets with both arms of the full W1."""

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

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / hand_reference.FPS
        self.sim_dt = self.frame_dt / hand_reference.SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self.support_phase = "rod"
        self.lift_height = 0.0
        self.forward_distance = 0.0
        self.maximum_root_position_error = 0.0
        self.maximum_root_angle_error = 0.0

        self.base_position = wp.vec3(args.robot_base_x, args.robot_base_y, args.robot_base_z)
        self.base_rotation = self._normalize_quaternion(
            wp.quat(args.robot_base_qx, args.robot_base_qy, args.robot_base_qz, args.robot_base_qw)
        )
        if args.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        self.ik_iterations = int(args.ik_iterations)

        bag_mesh_path = Path(args.bag_mesh).expanduser().resolve()
        robot_urdf = Path(args.robot_urdf).expanduser().resolve()
        hand_pose_path = Path(args.hand_pose).expanduser().resolve()
        for description, path in (
            ("Plastic bag mesh", bag_mesh_path),
            ("Dexforce W1 URDF", robot_urdf),
            ("Bimanual hand pose", hand_pose_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")
        self.robot_urdf = robot_urdf
        self.hand_poses = hand_reference._load_hand_pose(hand_pose_path)
        self.initial_root_targets = {
            side: self._copy_transform(self.hand_poses[side]["transform"]) for side in ("LEFT", "RIGHT")
        }
        self.current_root_targets = dict(self.initial_root_targets)

        self._build_scene(bag_mesh_path)
        self.device = self.model.device
        self._build_ik()
        self._initialize_robot_pose()

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
            vbd_options={
                "iterations": hand_reference.VBD_ITERATIONS,
                "friction_epsilon": 1.0e-4,
                "rigid_body_contact_buffer_size": 2048,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": hand_reference.BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": hand_reference.SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": 48,
                "particle_edge_contact_buffer_size": 96,
                "particle_collision_detection_interval": -1,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": hand_reference.SELF_CONTACT_MARGIN,
                "rigid_body_particle_contact_buffer_size": (hand_reference.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE),
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": hand_reference.SOFT_CONTACT_MARGIN,
                "soft_contact_max": SOFT_CONTACT_MAX,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": (
                    self.rod_shape_index,
                    *self.hand_particle_shapes,
                    self.ground_shape_index,
                ),
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend != "vbd_kinematic_full":
            raise RuntimeError(
                f"The full-W1 rod handoff requires vbd_kinematic_full, got {self.solver.features.backend}"
            )

        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self._collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        shape_flags = self.model.shape_flags.numpy()
        self._rod_active_shape_flag = int(shape_flags[self.rod_shape_index])
        self._hand_active_shape_flags = shape_flags[self.hand_particle_shapes].copy()
        self._set_rod_support_phase()

        self.render_indices = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(CAMERA_POSITION, CAMERA_PITCH, CAMERA_YAW)

        self.graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self._validate_initial_pose()

    def _build_scene(self, bag_mesh_path: Path):
        """Build the full robot around the original rod, bag, and balls."""
        bag_mesh = newton.Mesh.create_from_file(str(bag_mesh_path), compute_inertia=False, is_solid=False)
        vertices = np.asarray(bag_mesh.vertices, dtype=np.float32)
        indices = np.asarray(bag_mesh.indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected bag vertices with shape (n, 3), got {vertices.shape}")
        if indices.ndim != 1 or indices.size % 3 != 0:
            raise ValueError(f"Expected a flat triangle index buffer, got {indices.shape}")

        max_abs_x = float(np.abs(vertices[:, 0]).max())
        max_z = float(vertices[:, 2].max())
        handle_local = np.flatnonzero((np.abs(vertices[:, 0]) > 0.7 * max_abs_x) & (vertices[:, 2] > 0.65 * max_z))
        self.left_handle_indices = handle_local[vertices[handle_local, 0] < 0.0]
        self.right_handle_indices = handle_local[vertices[handle_local, 0] > 0.0]
        if self.left_handle_indices.size == 0 or self.right_handle_indices.size == 0:
            raise ValueError("The bag mesh must contain both left and right handles")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = hand_reference.HAND_CONTACT_KE
        builder.default_shape_cfg.kd = hand_reference.HAND_CONTACT_KD
        builder.default_shape_cfg.mu = hand_reference.HAND_CONTACT_MU
        builder.default_shape_cfg.margin = hand_reference.HAND_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(self.base_position, self.base_rotation),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(articulation_start, builder.articulation_count))
        if not self.robot_articulations:
            raise RuntimeError("Dexforce W1 URDF did not create an articulation")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collision_mask = collide_shapes | collide_particles
        self.hand_particle_shapes = []
        self.robot_visual_shapes = []
        for shape in range(self.robot_shape_end):
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if not is_collider:
                self.robot_visual_shapes.append(shape)
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            is_hand = ("left" in label or "right" in label) and any(
                keyword in label for keyword in self.HAND_CONTACT_KEYWORDS
            )
            if is_hand and is_collider:
                builder.shape_flags[shape] |= collide_particles
                builder.shape_flags[shape] &= ~collide_shapes
                self.hand_particle_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        if not self.hand_particle_shapes:
            raise RuntimeError("The full W1 URDF did not produce hand particle-collision shapes")

        builder.add_cloth_mesh(
            pos=wp.vec3(*hand_reference.BAG_POSITION),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            density=hand_reference.BAG_AREAL_DENSITY,
            tri_ke=hand_reference.BAG_TRI_KE,
            tri_ka=hand_reference.BAG_TRI_KA,
            tri_kd=hand_reference.BAG_TRI_KD,
            edge_ke=hand_reference.BAG_EDGE_KE,
            edge_kd=hand_reference.BAG_EDGE_KD,
            particle_radius=hand_reference.BAG_PARTICLE_RADIUS,
        )

        self.rod_position = hand_reference.BAG_POSITION + np.array(
            (0.0, 0.0, hand_reference.HANDLE_HOLE_CENTER_Z), dtype=np.float32
        )
        rod_cfg = newton.ModelBuilder.ShapeConfig(
            ke=hand_reference.ROD_CONTACT_KE,
            kd=hand_reference.ROD_CONTACT_KD,
            mu=0.8,
            margin=hand_reference.ROD_CONTACT_MARGIN,
        )
        rod_cfg.configure_sdf(force_sdf=True)
        self.rod_shape_index = builder.add_shape_capsule(
            -1,
            xform=wp.transform(
                wp.vec3(*self.rod_position),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi),
            ),
            radius=hand_reference.ROD_RADIUS,
            half_height=hand_reference.ROD_HALF_LENGTH,
            cfg=rod_cfg,
            color=hand_reference.ROD_COLOR,
            label="handle support rod",
        )
        builder.shape_flags[self.rod_shape_index] &= ~int(newton.ShapeFlags.VISIBLE)

        ball_cfg = newton.ModelBuilder.ShapeConfig(
            density=hand_reference.BALL_DENSITY,
            ke=hand_reference.BALL_CONTACT_KE,
            kd=hand_reference.BALL_CONTACT_KD,
            mu=hand_reference.BALL_FRICTION,
            margin=hand_reference.BALL_CONTACT_MARGIN,
        )
        for ball_index, (local_position, color) in enumerate(
            zip(hand_reference.BALL_LOCAL_POSITIONS, hand_reference.BALL_COLORS, strict=True)
        ):
            position = hand_reference.BAG_POSITION + np.asarray(local_position, dtype=np.float32)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                label=f"bag ball {ball_index}",
            )
            builder.body_qd[body] = wp.spatial_vector(
                0.0, 0.0, -hand_reference.BALL_INITIAL_DOWNWARD_SPEED, 0.0, 0.0, 0.0
            )
            builder.add_shape_sphere(
                body,
                radius=hand_reference.BALL_RADIUS,
                cfg=ball_cfg,
                color=color,
                label=f"bag ball {ball_index} shape",
            )

        self.ground_shape_index = builder.add_ground_plane(height=float(self.base_position[2]))
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = hand_reference.SOFT_CONTACT_KE
        self.model.soft_contact_kd = hand_reference.SOFT_CONTACT_KD
        self.model.soft_contact_mu = hand_reference.SOFT_CONTACT_FRICTION

        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.grip_body_indices = np.asarray(
            (
                self._body_index(self.model.body_label, "left_middle_dist"),
                self._body_index(self.model.body_label, "right_middle_dist"),
            ),
            dtype=np.int32,
        )

    def _build_ik(self):
        """Build an independent full-W1 model for realtime bimanual IK."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(self.base_position, self.base_rotation),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = builder.finalize(device=self.device)
        left_body = self._body_index(self.ik_model.body_label, "left_j7")
        right_body = self._body_index(self.ik_model.body_label, "right_j7")
        left_target = self._root_to_tcp("LEFT", self.initial_root_targets["LEFT"])
        right_target = self._root_to_tcp("RIGHT", self.initial_root_targets["RIGHT"])

        self.left_position_objective = ik.IKObjectivePosition(
            left_body,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(left_target)],
                dtype=wp.vec3,
                device=self.device,
            ),
        )
        self.left_rotation_objective = ik.IKObjectiveRotation(
            left_body,
            wp.quat_identity(),
            wp.array(
                [self._quaternion_vector(wp.transform_get_rotation(left_target))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        self.right_position_objective = ik.IKObjectivePosition(
            right_body,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(right_target)],
                dtype=wp.vec3,
                device=self.device,
            ),
        )
        self.right_rotation_objective = ik.IKObjectiveRotation(
            right_body,
            wp.quat_identity(),
            wp.array(
                [self._quaternion_vector(wp.transform_get_rotation(right_target))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        lower, upper = self._joint_limits()
        limit_objective = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[
                self.left_position_objective,
                self.left_rotation_objective,
                self.right_position_objective,
                self.right_rotation_objective,
                limit_objective,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.lock_indices, self.lock_values = self._locked_q()
        self.finger_indices, self.finger_values = self._finger_q()
        self.finger_indices_host = self.finger_indices.numpy()
        self.finger_values_host = self.finger_values.numpy()

    def _initialize_robot_pose(self):
        """Solve the saved root poses before constructing simulation states."""
        self._set_ik_targets(self.initial_root_targets)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        wp.launch(
            _lock_ik_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.device,
        )

        joint_q = self.model.joint_q.numpy()
        joint_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        joint_q[self.finger_indices.numpy()] = self.finger_values.numpy()
        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.zero_()

    def _set_rod_support_phase(self):
        """Enable rod support while disabling physical hand contact."""
        shape_flags = self.model.shape_flags.numpy()
        shape_flags[self.rod_shape_index] = self._rod_active_shape_flag
        for shape, active_flags in zip(
            self.hand_particle_shapes,
            self._hand_active_shape_flags,
            strict=True,
        ):
            shape_flags[shape] = int(active_flags) & ~self._collision_mask
        self.model.shape_flags.assign(shape_flags)
        self.support_phase = "rod"

    def _handoff_to_hands(self):
        """Disable rod contact and enable the full robot's hand colliders."""
        shape_flags = self.model.shape_flags.numpy()
        shape_flags[self.rod_shape_index] = self._rod_active_shape_flag & ~self._collision_mask
        for shape, active_flags in zip(
            self.hand_particle_shapes,
            self._hand_active_shape_flags,
            strict=True,
        ):
            shape_flags[shape] = active_flags
        self.model.shape_flags.assign(shape_flags)
        self.support_phase = "hands"

    def _update_root_targets(self):
        """Apply the source example's bounded carry offset to both roots."""
        if self.frame_index >= hand_reference.HAND_LIFT_START_FRAME:
            lift_frames = self.frame_index - hand_reference.HAND_LIFT_START_FRAME + 1
            self.lift_height = min(
                lift_frames * hand_reference.HAND_LIFT_SPEED * self.frame_dt,
                hand_reference.HAND_LIFT_HEIGHT,
            )
            self.forward_distance = min(
                lift_frames * hand_reference.HAND_FORWARD_SPEED * self.frame_dt,
                hand_reference.HAND_FORWARD_DISTANCE,
            )

        targets = {}
        for side, initial in self.initial_root_targets.items():
            position = wp.transform_get_translation(initial)
            targets[side] = wp.transform(
                wp.vec3(
                    float(position[0]),
                    float(position[1]) - self.forward_distance,
                    float(position[2]) + self.lift_height,
                ),
                wp.transform_get_rotation(initial),
            )
        self.current_root_targets = targets

    def _prepare_frame(self):
        """Solve both arm targets while preserving the saved finger pose."""
        self._update_root_targets()
        self._set_ik_targets(self.current_root_targets)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        wp.launch(
            _lock_ik_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.device,
        )

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        wp.launch(
            _copy_joint_q,
            self.ik_model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.device,
        )
        wp.launch(
            _set_indexed_joint_q,
            self.finger_indices.shape[0],
            [self.finger_indices, self.finger_values, self.frame_q_end],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance one display frame with the source MJVBDV2 settings."""
        for substep in range(hand_reference.SIM_SUBSTEPS):
            alpha = (substep + 1) / hand_reference.SIM_SUBSTEPS
            wp.launch(
                hand_reference._interpolate_joint_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                hand_reference._update_joint_velocity,
                self.model.joint_count,
                [
                    self.frame_q_start,
                    self.frame_q_end,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.frame_dt,
                    self.state_0.joint_qd,
                ],
                device=self.device,
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
            wp.launch(
                hand_reference._apply_particle_drag,
                self.model.particle_count,
                [
                    self.state_0.particle_qd,
                    self.model.particle_mass,
                    hand_reference.AIR_DRAG_RATE,
                    self.state_0.particle_f,
                ],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_simulation_graph(self):
        """Capture the warmed physics substeps while keeping IK realtime."""
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
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}")

    def step(self):
        """Solve realtime IK and advance one rod-handoff frame."""
        if self.frame_index == hand_reference.ROD_RELEASE_FRAME:
            self._handoff_to_hands()
        self._prepare_frame()
        if self.graph is None:
            self._simulate_substeps()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self):
        """Render the full W1 and the source transparent-red bag."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/plastic_bag",
            self.state_0.particle_q,
            self.render_indices,
            backface_culling=False,
            color=hand_reference.BAG_COLOR,
            roughness=0.65,
            metallic=0.0,
            opacity=hand_reference.BAG_OPACITY,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify finite state and exact tracking of both source hand roots."""
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(particle_q))
        assert np.all(np.isfinite(body_q))
        assert np.all(np.isfinite(joint_q))
        assert np.allclose(
            joint_q[self.finger_indices_host],
            self.finger_values_host,
            rtol=0.0,
            atol=1.0e-6,
        )
        for side, body in (("LEFT", self.left_body), ("RIGHT", self.right_body)):
            position_error, angle_error = self._transform_error(
                self._actual_hand_root(side, body),
                self.current_root_targets[side],
            )
            self.maximum_root_position_error = max(self.maximum_root_position_error, position_error)
            self.maximum_root_angle_error = max(self.maximum_root_angle_error, angle_error)
            assert position_error <= END_EFFECTOR_POSITION_TOLERANCE, (
                f"{side} W1 hand-root position error is {position_error:.6f} m"
            )
            assert angle_error <= np.radians(END_EFFECTOR_ANGLE_TOLERANCE_DEG), (
                f"{side} W1 hand-root angle error is {np.degrees(angle_error):.6f} degrees"
            )

    def test_final(self):
        """Verify realtime IK, handoff flags, and stable loaded-bag state."""
        assert self.solver.features.backend == "vbd_kinematic_full"
        assert not hasattr(self, "cached_joint_targets")
        assert np.all(np.isfinite(self.ik_q.numpy()))
        visual_flags = self.model.shape_flags.numpy()[self.robot_visual_shapes]
        assert np.all((visual_flags & self._collision_mask) == 0), "Robot visual shapes must remain non-colliding"
        if self.use_graph and self.frame_index > 0:
            assert self.graph is not None

        if self.frame_index <= hand_reference.ROD_RELEASE_FRAME:
            return
        assert self.support_phase == "hands"
        rod_flags = int(self.model.shape_flags.numpy()[self.rod_shape_index])
        assert rod_flags & self._collision_mask == 0
        assert rod_flags & int(newton.ShapeFlags.VISIBLE) == 0
        hand_flags = self.model.shape_flags.numpy()[self.hand_particle_shapes]
        assert np.all((hand_flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0)

        soft_contact_count = int(self.solver.contacts.soft_contact_count.numpy()[0])
        if soft_contact_count >= SOFT_CONTACT_MAX:
            raise ValueError(f"Soft-contact capacity exhausted: {soft_contact_count} >= {SOFT_CONTACT_MAX}")
        body_contact_overflow = int(self.solver.vbd_solver.body_particle_contact_overflow_max.numpy()[0])
        if body_contact_overflow > 0:
            raise ValueError(
                "Per-body particle-contact capacity exhausted: "
                f"{body_contact_overflow} > {hand_reference.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
            )

        if self.frame_index <= hand_reference.HAND_LIFT_START_FRAME + 30:
            return
        if self.lift_height <= 0.0 or self.forward_distance <= 0.0:
            raise ValueError("The scripted bimanual carry did not start")
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        grip_positions = body_q[self.grip_body_indices, :3]
        left_distance = np.linalg.norm(particle_q[self.left_handle_indices] - grip_positions[0], axis=1).min()
        right_distance = np.linalg.norm(particle_q[self.right_handle_indices] - grip_positions[1], axis=1).min()
        if left_distance > 0.12 or right_distance > 0.12:
            raise ValueError(
                "Plastic bag slipped away from a hand: "
                f"handle distances are {left_distance:.6g} m and {right_distance:.6g} m"
            )

    def _set_ik_targets(self, root_targets: dict[str, wp.transform]):
        """Convert both isolated roots into W1 wrist objectives."""
        left = self._root_to_tcp("LEFT", root_targets["LEFT"])
        right = self._root_to_tcp("RIGHT", root_targets["RIGHT"])
        self.left_position_objective.set_target_position(0, wp.transform_get_translation(left))
        self.left_rotation_objective.set_target_rotation(0, self._quaternion_vector(wp.transform_get_rotation(left)))
        self.right_position_objective.set_target_position(0, wp.transform_get_translation(right))
        self.right_rotation_objective.set_target_rotation(0, self._quaternion_vector(wp.transform_get_rotation(right)))

    def _root_to_tcp(self, side: str, root_transform: wp.transform) -> wp.transform:
        """Convert one standalone hand-root pose to its W1 wrist TCP."""
        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = self._quaternion_multiply(
            hand_rotation,
            wp.quat_inverse(J7_TO_HAND_BASE_ROTATIONS[side]),
        )
        target_offset = TCP_OFFSET - J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _actual_hand_root(self, side: str, body: int) -> wp.transform:
        """Return one full robot hand-base world transform."""
        body_transform = wp.transform(*self.state_0.body_q.numpy()[body])
        wrist_position = wp.transform_get_translation(body_transform)
        wrist_rotation = wp.transform_get_rotation(body_transform)
        return wp.transform(
            wrist_position + wp.quat_rotate(wrist_rotation, J7_TO_HAND_BASE_OFFSET),
            self._quaternion_multiply(wrist_rotation, J7_TO_HAND_BASE_ROTATIONS[side]),
        )

    def _validate_initial_pose(self):
        """Ensure the complete W1 starts at the exact saved hand poses."""
        for side, body in (("LEFT", self.left_body), ("RIGHT", self.right_body)):
            position_error, angle_error = self._transform_error(
                self._actual_hand_root(side, body),
                self.initial_root_targets[side],
            )
            if position_error > END_EFFECTOR_POSITION_TOLERANCE:
                raise ValueError(f"Initial {side} hand-root position error is {position_error:.6f} m")
            if angle_error > np.radians(END_EFFECTOR_ANGLE_TOLERANCE_DEG):
                raise ValueError(f"Initial {side} hand-root angle error is {np.degrees(angle_error):.6f} degrees")

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Lock every non-arm IK degree of freedom at its authored value."""
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        qd_start = self.ik_model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for joint, label in enumerate(self.ik_model.joint_label):
            if label not in controlled:
                lower[int(qd_start[joint])] = q[int(q_start[joint])] - 1.0e-4
                upper[int(qd_start[joint])] = q[int(q_start[joint])] + 1.0e-4
        return lower, upper

    def _locked_q(self) -> tuple[wp.array[int], wp.array[float]]:
        """Return non-arm coordinates restored after every IK solve."""
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices = [
            int(q_start[joint]) for joint, label in enumerate(self.ik_model.joint_label) if label not in controlled
        ]
        return wp.array(indices, dtype=wp.int32, device=self.device), wp.array(
            [q[index] for index in indices], dtype=wp.float32, device=self.device
        )

    def _finger_q(self) -> tuple[wp.array[int], wp.array[float]]:
        """Return the unchanged saved finger configuration for both hands."""
        q_start = self.model.joint_q_start.numpy()
        indices = []
        values = []
        for side in ("LEFT", "RIGHT"):
            saved = self.hand_poses[side]["joint_degrees"]
            for suffix in self.HAND_SUFFIXES:
                joint = self._joint_index(f"{side}_{suffix}")
                indices.append(int(q_start[joint]))
                values.append(np.radians(saved[suffix]))
        return wp.array(indices, dtype=wp.int32, device=self.device), wp.array(
            values, dtype=wp.float32, device=self.device
        )

    def _joint_index(self, name: str) -> int:
        """Return a model joint index from its unprefixed asset name."""
        return next(index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        """Return a body index from its unprefixed asset name."""
        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    @staticmethod
    def _copy_transform(value: wp.transform) -> wp.transform:
        """Return an independent host transform value."""
        position = wp.transform_get_translation(value)
        rotation = wp.transform_get_rotation(value)
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    @staticmethod
    def _transform_error(actual: wp.transform, target: wp.transform) -> tuple[float, float]:
        """Return translation [m] and shortest-angle [rad] errors."""
        actual_position = np.asarray(wp.transform_get_translation(actual), dtype=np.float64)
        target_position = np.asarray(wp.transform_get_translation(target), dtype=np.float64)
        position_error = float(np.linalg.norm(actual_position - target_position))
        actual_rotation = np.asarray(wp.transform_get_rotation(actual), dtype=np.float64)
        target_rotation = np.asarray(wp.transform_get_rotation(target), dtype=np.float64)
        actual_rotation /= max(float(np.linalg.norm(actual_rotation)), 1.0e-8)
        target_rotation /= max(float(np.linalg.norm(target_rotation)), 1.0e-8)
        cosine = float(np.clip(abs(np.dot(actual_rotation, target_rotation)), 0.0, 1.0))
        return position_error, 2.0 * float(np.arccos(cosine))

    @staticmethod
    def _normalize_quaternion(value: wp.quat) -> wp.quat:
        """Return a normalized quaternion."""
        array = np.asarray([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
        array /= max(float(np.linalg.norm(array)), 1.0e-8)
        return wp.quat(*array)

    @staticmethod
    def _quaternion_multiply(a: wp.quat, b: wp.quat) -> wp.quat:
        """Multiply two host quaternion values."""
        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _quaternion_vector(value: wp.quat) -> wp.vec4:
        """Convert a quaternion to the IK rotation target type."""
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def create_parser():
        """Create full-W1 realtime-IK rod-handoff options."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        parser.add_argument("--bag-mesh", type=Path, default=hand_reference.BAG_MESH_PATH)
        parser.add_argument("--robot-urdf", type=Path, default=ROBOT_URDF)
        parser.add_argument("--hand-pose", type=Path, default=hand_reference.HAND_POSE_PATH)
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=RUNTIME_IK_ITERATIONS,
            help="Realtime bimanual IK iterations for each displayed frame.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed MJVBDV2 physics substeps on CUDA.",
        )
        parser.add_argument("--robot-base-x", type=float, default=float(ROBOT_BASE_POSITION[0]))
        parser.add_argument("--robot-base-y", type=float, default=float(ROBOT_BASE_POSITION[1]))
        parser.add_argument("--robot-base-z", type=float, default=float(ROBOT_BASE_POSITION[2]))
        parser.add_argument("--robot-base-qx", type=float, default=float(ROBOT_BASE_ROTATION[0]))
        parser.add_argument("--robot-base-qy", type=float, default=float(ROBOT_BASE_ROTATION[1]))
        parser.add_argument("--robot-base-qz", type=float, default=float(ROBOT_BASE_ROTATION[2]))
        parser.add_argument("--robot-base-qw", type=float, default=float(ROBOT_BASE_ROTATION[3]))
        return parser


def main():
    """Run the full-W1 realtime-IK plastic-bag rod handoff."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
