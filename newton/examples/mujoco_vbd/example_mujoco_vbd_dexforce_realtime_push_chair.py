# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Push a dynamic WAIC chair with realtime Dexforce W1 IK and MuJoCo/VBD.

The scene is intentionally trajectory-free.  A smooth procedural controller
opens both hands for approach, closes them around the chair back, lifts the
chair, carries it forward, places it, then opens and withdraws both hands.
Both arm targets are solved by Newton IK every displayed frame.  The resulting
articulated pose is injected as an immovable one-way proxy; the 8 kg chair
remains a free VBD rigid body and is moved only by hand contact.

Run, from the repository root::

    uv run --extra examples -m newton.examples \
        mujoco_vbd_dexforce_realtime_push_chair
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCoVBD

FPS = 60
DEFAULT_NUM_FRAMES = 900
DEFAULT_SUBSTEPS = 4
DEFAULT_VBD_ITERATIONS = 6
DEFAULT_IK_ITERATIONS = 32
INITIAL_IK_ITERATIONS = 300

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_ROBOT_URDF = ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"
DEFAULT_CHAIR_URDF = ASSET_ROOT / "waic_house9_chair" / "chair.urdf"

ROBOT_BASE_START = wp.vec3(-0.70, 0.0, 0.0)
ROBOT_BASE_GRASP = wp.vec3(0.0, 0.0, 0.0)
ROBOT_BASE_FORWARD = wp.vec3(0.3630, 0.0, 0.0)
ROBOT_BASE_CARRY_END = wp.vec3(0.5825, -0.1129, 0.0)
ROBOT_BASE_RETREAT = wp.vec3(0.2043, 0.2337, 0.0)
ROBOT_BASE_ROTATION = wp.quat_identity()
ROBOT_BASE_TURN_ONE_YAW = math.radians(-21.667)
ROBOT_BASE_TURN_TWO_YAW = math.radians(-31.667)
ROBOT_BASE_CARRY_END_YAW = math.radians(-42.5)

CHAIR_POSITION = wp.vec3(0.7155, 0.0146, 0.0)
CHAIR_ROTATION = wp.quat(0.0, 0.0, -0.7571, 0.6533)
CHAIR_MASS = 8.0
CHAIR_COLOR = (0.72, 0.43, 0.22)
GROUND_COLOR = (0.22, 0.25, 0.29)

# Target positions are expressed in the moving W1 base frame.  The chair's
# back arc faces -X, so the palms arrive at its two upper corners together.
LEFT_TCP_STANDBY = wp.vec3(0.4564, 0.3098, 0.9047)
RIGHT_TCP_STANDBY = wp.vec3(0.4564, -0.3098, 0.9047)
LEFT_TCP_CONTACT = wp.vec3(0.6041, 0.2612, 0.7063)
RIGHT_TCP_CONTACT = wp.vec3(0.6041, -0.2612, 0.7063)
LEFT_TCP_LIFT = wp.vec3(0.6042, 0.2612, 0.7710)
RIGHT_TCP_LIFT = wp.vec3(0.6042, -0.2612, 0.7710)
LEFT_TCP_PLACE = wp.vec3(0.5999, 0.2596, 0.7114)
RIGHT_TCP_PLACE = wp.vec3(0.5999, -0.2596, 0.7114)
LEFT_TCP_RETREAT = wp.vec3(0.4312, 0.2881, 0.8371)
RIGHT_TCP_RETREAT = wp.vec3(0.4312, -0.2881, 0.8371)
LEFT_TCP_ROTATION = wp.quat(0.60942, 0.30651, 0.72839, -0.06413)
RIGHT_TCP_ROTATION = wp.quat(0.60942, -0.30651, 0.72839, 0.06413)
LEFT_TCP_RETREAT_ROTATION = wp.quat(0.04206, 0.00173, 0.98846, 0.14554)
RIGHT_TCP_RETREAT_ROTATION = wp.quat(0.04206, -0.00173, 0.98846, -0.14553)
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

SETTLE_SECONDS = 0.5
APPROACH_SECONDS = 2.0
GRASP_SECONDS = 1.0
LIFT_SECONDS = 1.0
CARRY_FORWARD_SECONDS = 1.2
CARRY_TURN_ONE_SECONDS = 0.8
CARRY_TURN_TWO_SECONDS = 0.5
CARRY_FINAL_SECONDS = 1.5
CARRY_SECONDS = CARRY_FORWARD_SECONDS + CARRY_TURN_ONE_SECONDS + CARRY_TURN_TWO_SECONDS + CARRY_FINAL_SECONDS
LOWER_SECONDS = 1.0
PLACEMENT_SETTLE_SECONDS = 0.5
PLACEMENT_GRASP_ALPHA = 0.57 / 0.73
RELEASE_SECONDS = 1.2933
BASE_RETREAT_SECONDS = 1.0
ARM_RETREAT_SECONDS = 2.3975
HOLD_SECONDS = 0.5

CONTACT_KE = 5.0e5
CONTACT_KD = 1.0e-7
HAND_FRICTION = 1.0
CHAIR_FRICTION = 1.0
GROUND_FRICTION = 1.5
RIGID_CONTACT_MAX = 8192
RIGID_BODY_CONTACT_BUFFER_SIZE = 1280
RIGID_CONTACT_ALPHA = 0.95

HAND_VHACD_MAX_HULLS = 2
HAND_VHACD_RESOLUTION = 50_000
HAND_VHACD_VOLUME_ERROR = 4.0
HAND_VHACD_MAX_RECURSION = 6
HAND_VHACD_MAX_VERTICES = 32

CAMERA_POSITION = wp.vec3(2.15, -2.75, 1.55)
CAMERA_PITCH = -13.0
CAMERA_YAW = 132.0


def _quat_from_yaw(yaw: float) -> wp.quat:
    """Construct a Z-up root rotation from a scalar yaw."""
    half_yaw = 0.5 * float(yaw)
    return wp.quat(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


@wp.kernel
def _copy_indexed_q(
    source: wp.array[float],
    source_indices: wp.array[wp.int32],
    destination_indices: wp.array[wp.int32],
    destination: wp.array[float],
):
    """Copy matching IK coordinates into the floating-base scene model."""
    index = wp.tid()
    destination[destination_indices[index]] = source[source_indices[index]]


@wp.kernel
def _lock_ik_q(
    q: wp.array2d[float],
    indices: wp.array[wp.int32],
    values: wp.array[float],
):
    """Restore every non-arm coordinate after an IK iteration batch."""
    index = wp.tid()
    q[0, indices[index]] = values[index]


@wp.kernel
def _write_indexed_lerp(
    indices: wp.array[wp.int32],
    values_start: wp.array[float],
    values_end: wp.array[float],
    alpha: float,
    q: wp.array[float],
):
    """Interpolate an authored hand pose into generalized coordinates."""
    index = wp.tid()
    q[indices[index]] = values_start[index] * (1.0 - alpha) + values_end[index] * alpha


@wp.kernel
def _write_free_root_pose(
    root_q_start: int,
    position: wp.vec3,
    rotation: wp.quat,
    joint_q: wp.array[float],
):
    """Write one floating-base transform to generalized coordinates."""
    if wp.tid() == 0:
        joint_q[root_q_start + 0] = position[0]
        joint_q[root_q_start + 1] = position[1]
        joint_q[root_q_start + 2] = position[2]
        joint_q[root_q_start + 3] = rotation[0]
        joint_q[root_q_start + 4] = rotation[1]
        joint_q[root_q_start + 5] = rotation[2]
        joint_q[root_q_start + 6] = rotation[3]


@wp.kernel
def _interpolate_joint_q(
    q_start: wp.array[float],
    q_end: wp.array[float],
    alpha: float,
    q_out: wp.array[float],
):
    """Interpolate one frame of prescribed generalized coordinates."""
    coordinate = wp.tid()
    q_out[coordinate] = q_start[coordinate] * (1.0 - alpha) + q_end[coordinate] * alpha


@wp.kernel
def _update_joint_velocity(
    q_start: wp.array[float],
    q_end: wp.array[float],
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    inv_dt: float,
    qd_out: wp.array[float],
):
    """Compute prescribed rigid velocity, including a free root."""
    joint = wp.tid()
    q_begin = joint_q_start[joint]
    q_end_index = joint_q_start[joint + 1]
    qd_begin = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + 1]
    if joint_type[joint] == newton.JointType.FREE:
        qd_out[qd_begin + 0] = (q_end[q_begin + 0] - q_start[q_begin + 0]) * inv_dt
        qd_out[qd_begin + 1] = (q_end[q_begin + 1] - q_start[q_begin + 1]) * inv_dt
        qd_out[qd_begin + 2] = (q_end[q_begin + 2] - q_start[q_begin + 2]) * inv_dt
        rotation_delta = wp.normalize(
            wp.quat(q_end[q_begin + 3], q_end[q_begin + 4], q_end[q_begin + 5], q_end[q_begin + 6])
            * wp.quat_inverse(
                wp.quat(
                    q_start[q_begin + 3],
                    q_start[q_begin + 4],
                    q_start[q_begin + 5],
                    q_start[q_begin + 6],
                )
            )
        )
        axis, angle = wp.quat_to_axis_angle(rotation_delta)
        qd_out[qd_begin + 3] = axis[0] * angle * inv_dt
        qd_out[qd_begin + 4] = axis[1] * angle * inv_dt
        qd_out[qd_begin + 5] = axis[2] * angle * inv_dt
    else:
        for coordinate in range(qd_end - qd_begin):
            if q_begin + coordinate < q_end_index:
                qd_out[qd_begin + coordinate] = (q_end[q_begin + coordinate] - q_start[q_begin + coordinate]) * inv_dt


class Example:
    """Run trajectory-free realtime IK against a dynamic VBD chair."""

    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_BODY_KEYWORDS = ("j7", "thumb", "index", "middle", "ring", "pinky")
    CROUCH_JOINTS: ClassVar[dict[str, float]] = {
        "ANKLE": math.radians(55.0),
        "KNEE": math.radians(-110.0),
        "BUTTOCK": math.radians(70.0),
    }
    # The hands approach open, then interpolate to GRASP_HAND_JOINTS only after
    # both wrists reach the chair back.
    OPEN_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "HAND_THUMB2": 0.5 * math.pi,
        "HAND_THUMB1": 0.0,
        "HAND_INDEX": 0.0,
        "INDEX_PIP": 0.0,
        "HAND_MIDDLE": 0.0,
        "MIDDLE_PIP": 0.0,
        "HAND_RING": 0.0,
        "RING_PIP": 0.0,
        "HAND_PINKY": 0.0,
        "PINKY_PIP": 0.0,
    }
    GRASP_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "HAND_THUMB2": 0.5 * math.pi,
        "HAND_THUMB1": 0.4141,
        "HAND_INDEX": 0.6211,
        "INDEX_PIP": 0.9938,
        "HAND_MIDDLE": 0.6211,
        "MIDDLE_PIP": 0.9938,
        "HAND_RING": 0.6211,
        "RING_PIP": 0.9938,
        "HAND_PINKY": 0.6211,
        "PINKY_PIP": 0.9938,
    }

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = int(args.substeps)
        self.vbd_iterations = int(args.vbd_iterations)
        self.ik_iterations = int(args.ik_iterations)
        if self.sim_substeps < 1:
            raise ValueError("--substeps must be at least 1")
        if self.vbd_iterations < 1:
            raise ValueError("--vbd-iterations must be at least 1")
        if self.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.phase = "settle"

        self.robot_urdf = Path(args.robot_urdf).expanduser().resolve()
        self.chair_urdf = Path(args.chair_urdf).expanduser().resolve()
        for label, path in (("Dexforce W1 URDF", self.robot_urdf), ("WAIC chair URDF", self.chair_urdf)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self._build_scene()
        self.device = self.model.device
        self._build_hand_pose_control()
        self._build_ik()
        self._initialize_pose()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            coupling_mode="one_way",
            contact_mode="full",
            vbd_options={
                "iterations": self.vbd_iterations,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                # The moving hand contacts are regenerated every substep and
                # intentionally carry no trajectory-dependent history.  The
                # stabilized alpha lets sustained kinematic motion push the
                # chair without turning the initial multi-hull contact into
                # an impact impulse.
                "rigid_avbd_contact_alpha": RIGID_CONTACT_ALPHA,
                "friction_epsilon": 1.0e-4,
            },
            collision_options={
                "broad_phase": "nxn",
                "rigid_contact_max": RIGID_CONTACT_MAX,
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend.value != "one_way_kinematic_full":
            raise RuntimeError(
                f"Realtime chair pushing requires one_way_kinematic_full, got {self.solver.features.backend.value}"
            )

        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        wp.copy(self.frame_q_start, self.model.joint_q)
        wp.copy(self.frame_q_end, self.model.joint_q)
        self.initial_chair_position = self.state_0.body_q.numpy()[self.chair_body, :3].copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POSITION, CAMERA_PITCH, CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 42.0

        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.graph = None

    def _build_scene(self):
        """Build one kinematic W1 articulation and one free rigid chair."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = HAND_FRICTION
        builder.default_shape_cfg.margin = 0.0
        SolverMuJoCoVBD.register_custom_attributes(builder)

        robot_articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(ROBOT_BASE_START, ROBOT_BASE_ROTATION),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(robot_articulation_start, builder.articulation_count))
        if len(self.robot_articulations) != 1:
            raise RuntimeError(f"Expected one W1 articulation, got {self.robot_articulations}")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self._set_builder_posture(builder)
        self._decompose_hand_collision_meshes(builder)
        self.robot_shape_end = builder.shape_count
        self._configure_robot_collision_flags(builder)

        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CHAIR_FRICTION
        builder.default_shape_cfg.margin = 0.0
        chair_body_start = builder.body_count
        chair_shape_start = builder.shape_count
        builder.add_urdf(
            str(self.chair_urdf),
            xform=wp.transform(CHAIR_POSITION, CHAIR_ROTATION),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        if builder.body_count != chair_body_start + 1:
            raise RuntimeError("The WAIC chair asset must contain exactly one rigid link")
        self.chair_body = chair_body_start
        self._set_chair_mass(builder, self.chair_body, CHAIR_MASS)
        collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(chair_shape_start, builder.shape_count):
            builder.shape_color[shape] = CHAIR_COLOR
            if builder.shape_flags[shape] & (collision_flag | particle_flag):
                builder.shape_flags[shape] |= collision_flag
                builder.shape_flags[shape] &= ~particle_flag

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=GROUND_FRICTION,
            margin=0.0,
        )
        builder.add_ground_plane(
            height=0.0,
            cfg=ground_cfg,
            color=GROUND_COLOR,
            label="waic_push_chair_ground",
        )
        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.root_q_start = self._robot_root_q_start()
        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")

    def _set_builder_posture(self, builder: newton.ModelBuilder) -> None:
        """Author the crouched chair-pushing posture without a trajectory."""
        for joint_name, value in self.CROUCH_JOINTS.items():
            self._set_builder_joint_coordinate(builder, joint_name, value)
        for side in ("LEFT", "RIGHT"):
            for suffix, value in self.OPEN_HAND_JOINTS.items():
                self._set_builder_joint_coordinate(builder, f"{side}_{suffix}", value)

    def _build_hand_pose_control(self) -> None:
        """Cache scalar hand coordinates for the procedural grasp."""
        q_start = self.model.joint_q_start.numpy()
        indices: list[int] = []
        open_values: list[float] = []
        grasp_values: list[float] = []
        for side in ("LEFT", "RIGHT"):
            for suffix, open_value in self.OPEN_HAND_JOINTS.items():
                name = f"{side}_{suffix}"
                joint = next(
                    (index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name)),
                    None,
                )
                if joint is None:
                    raise ValueError(f"W1 joint is missing: {name}")
                if int(q_start[joint + 1] - q_start[joint]) != 1:
                    raise ValueError(f"W1 hand joint must have one coordinate: {name}")
                indices.append(int(q_start[joint]))
                open_values.append(float(open_value))
                grasp_values.append(float(self.GRASP_HAND_JOINTS[suffix]))
        self.hand_q_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self.hand_q_open = wp.array(open_values, dtype=wp.float32, device=self.device)
        self.hand_q_grasp = wp.array(grasp_values, dtype=wp.float32, device=self.device)

    def _configure_robot_collision_flags(self, builder: newton.ModelBuilder) -> None:
        """Keep only distal hand geometry active against the chair."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collision_mask = collide_shapes | collide_particles
        self.robot_contact_shapes: list[int] = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            body_label = "" if body < 0 else builder.body_label[body].lower()
            is_hand = (
                body >= 0
                and ("left_" in body_label or "right_" in body_label)
                and any(keyword in body_label for keyword in self.HAND_BODY_KEYWORDS)
            )
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if is_hand and is_collider:
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
                self.robot_contact_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        if not self.robot_contact_shapes:
            raise RuntimeError("The W1 asset did not produce any distal hand collision shapes")

    def _decompose_hand_collision_meshes(self, builder: newton.ModelBuilder) -> None:
        """Replace active hand collision meshes with bounded V-HACD parts.

        The URDF's separate visual meshes remain untouched.  Convex collision
        parts use the GJK/MPR narrow phase and therefore avoid expanding each
        nearby hand/chair pair into thousands of triangle candidates.
        """
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        source_shapes: list[int] = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            if body < 0 or not (builder.shape_flags[shape] & collide_shapes):
                continue
            body_label = builder.body_label[body].lower()
            is_hand = ("left_" in body_label or "right_" in body_label) and any(
                keyword in body_label for keyword in self.HAND_BODY_KEYWORDS
            )
            if is_hand and builder.shape_type[shape] == newton.GeoType.MESH:
                source_shapes.append(shape)

        if not source_shapes:
            raise RuntimeError("The W1 asset did not provide hand collision meshes for V-HACD")

        remeshed = builder.approximate_meshes(
            method="vhacd",
            shape_indices=source_shapes,
            raise_on_failure=True,
            keep_visual_shapes=False,
            maxConvexHulls=HAND_VHACD_MAX_HULLS,
            resolution=HAND_VHACD_RESOLUTION,
            minimumVolumePercentErrorAllowed=HAND_VHACD_VOLUME_ERROR,
            maxRecursionDepth=HAND_VHACD_MAX_RECURSION,
            maxNumVerticesPerCH=HAND_VHACD_MAX_VERTICES,
            asyncACD=False,
        )
        if len(remeshed) != len(source_shapes):
            raise RuntimeError(f"V-HACD converted {len(remeshed)} of {len(source_shapes)} W1 hand collision meshes")
        self.hand_collision_source_count = len(source_shapes)

    @staticmethod
    def _set_chair_mass(builder: newton.ModelBuilder, body: int, target_mass: float) -> None:
        """Scale the imported collision inertia to the authored 8 kg mass."""
        imported_mass = float(builder.body_mass[body])
        if imported_mass <= 0.0:
            raise ValueError("The chair collision asset produced zero mass")
        scale = target_mass / imported_mass
        inertia = np.asarray(builder.body_inertia[body], dtype=np.float64).reshape(3, 3) * scale
        builder.body_mass[body] = float(target_mass)
        builder.body_inertia[body] = wp.mat33(*[float(value) for value in inertia.reshape(-1)])

    def _build_ik(self):
        """Build a fixed-base W1 used only for per-frame realtime IK."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform_identity(),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self._set_builder_posture(builder)
        self.ik_model = builder.finalize(device=self.device)
        left_body = self._body_index(self.ik_model.body_label, "left_j7")
        right_body = self._body_index(self.ik_model.body_label, "right_j7")

        self.left_position_objective = ik.IKObjectivePosition(
            left_body,
            TCP_OFFSET,
            wp.array([LEFT_TCP_STANDBY], dtype=wp.vec3, device=self.device),
        )
        self.left_rotation_objective = ik.IKObjectiveRotation(
            left_body,
            wp.quat_identity(),
            wp.array([self._quat_vector(LEFT_TCP_ROTATION)], dtype=wp.vec4, device=self.device),
        )
        self.right_position_objective = ik.IKObjectivePosition(
            right_body,
            TCP_OFFSET,
            wp.array([RIGHT_TCP_STANDBY], dtype=wp.vec3, device=self.device),
        )
        self.right_rotation_objective = ik.IKObjectiveRotation(
            right_body,
            wp.quat_identity(),
            wp.array([self._quat_vector(RIGHT_TCP_ROTATION)], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._ik_joint_limits()
        joint_limit_objective = ik.IKObjectiveJointLimit(
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
                joint_limit_objective,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_lock_indices, self.ik_lock_values = self._ik_locked_q()
        source, destination = self._joint_coordinate_mapping()
        self.ik_source_indices = wp.array(source, dtype=wp.int32, device=self.device)
        self.scene_destination_indices = wp.array(destination, dtype=wp.int32, device=self.device)

    def _initialize_pose(self):
        """Solve the first realtime target before allocating simulation states."""
        self._set_ik_targets(LEFT_TCP_STANDBY, RIGHT_TCP_STANDBY)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        self._restore_locked_ik_q()
        self._copy_ik_to_scene(self.model.joint_q)
        self._write_hand_pose(0.0, self.model.joint_q)
        self._write_root_pose(ROBOT_BASE_START, self.model.joint_q)
        self.model.joint_qd.zero_()

    def _prepare_frame(self):
        """Evaluate the controller and solve both arms for this frame."""
        root_position, root_rotation, left_target, right_target, grasp_alpha, self.phase = self._sample_controller(
            self.sim_time
        )
        left_rotation, right_rotation = self._sample_tcp_rotations(self.sim_time)
        self._set_ik_targets(left_target, right_target, left_rotation, right_rotation)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._restore_locked_ik_q()

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        self._copy_ik_to_scene(self.frame_q_end)
        self._write_hand_pose(grasp_alpha, self.frame_q_end)
        self._write_root_pose(root_position, self.frame_q_end, root_rotation)

    def _simulate_substeps(self):
        """Advance one displayed frame with prescribed robot FK and VBD chair dynamics."""
        for substep in range(self.sim_substeps):
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_joint_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _update_joint_velocity,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_simulation_graph(self):
        """Capture physics only; realtime IK remains outside the graph."""
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
            raise RuntimeError(f"CUDA graph capture failed on {self.device}")

    def step(self):
        """Solve realtime IK, then simulate the dynamic chair."""
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
        """Render the full W1, dynamic chair, and ground."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Check that the realtime coupled state stays finite."""
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Realtime chair scene contains a non-finite body pose")
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise ValueError("Realtime chair scene contains a non-finite joint coordinate")

    def test_final(self):
        """Verify realtime IK dispatch and, for long runs, chair motion."""
        if self.solver.features.backend.value != "one_way_kinematic_full":
            raise ValueError(f"Unexpected SolverMuJoCoVBD backend: {self.solver.features.backend.value}")
        if hasattr(self, "cached_joint_targets"):
            raise ValueError("Realtime chair pushing unexpectedly created a trajectory cache")
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("Realtime IK returned a non-finite coordinate")
        carry_observation_frame = int(
            (SETTLE_SECONDS + APPROACH_SECONDS + GRASP_SECONDS + LIFT_SECONDS + 0.5 * CARRY_SECONDS) * FPS
        )
        if self.frame_index < carry_observation_frame:
            return
        chair_pose = self.state_0.body_q.numpy()[self.chair_body]
        chair_position = chair_pose[:3]
        displacement = chair_position - self.initial_chair_position
        horizontal_displacement = float(np.linalg.norm(displacement[:2]))
        if horizontal_displacement < 0.02:
            raise ValueError(f"The realtime hand contact did not move the chair: dxy={horizontal_displacement:.6f} m")
        if horizontal_displacement > 1.5 or abs(float(displacement[2])) > 0.25:
            raise ValueError(f"The realtime hand contact destabilized the chair: displacement={displacement}")
        qx, qy = float(chair_pose[3]), float(chair_pose[4])
        chair_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        if chair_up_z < 0.5:
            raise ValueError(f"The chair tipped over after realtime placement: up_z={chair_up_z:.6f}")

    def _sample_controller(self, time_seconds: float) -> tuple[wp.vec3, wp.quat, wp.vec3, wp.vec3, float, str]:
        """Return continuous root, TCP, and grasp targets without replay."""
        if time_seconds < SETTLE_SECONDS:
            return ROBOT_BASE_START, ROBOT_BASE_ROTATION, LEFT_TCP_STANDBY, RIGHT_TCP_STANDBY, 0.0, "settle"

        time_seconds -= SETTLE_SECONDS
        if time_seconds < APPROACH_SECONDS:
            alpha = self._smoothstep(time_seconds / APPROACH_SECONDS)
            return (
                self._lerp_vec3(ROBOT_BASE_START, ROBOT_BASE_GRASP, alpha),
                ROBOT_BASE_ROTATION,
                self._lerp_vec3(LEFT_TCP_STANDBY, LEFT_TCP_CONTACT, alpha),
                self._lerp_vec3(RIGHT_TCP_STANDBY, RIGHT_TCP_CONTACT, alpha),
                0.0,
                "approach",
            )

        time_seconds -= APPROACH_SECONDS
        if time_seconds < GRASP_SECONDS:
            grasp_alpha = self._smoothstep(time_seconds / GRASP_SECONDS)
            return (
                ROBOT_BASE_GRASP,
                ROBOT_BASE_ROTATION,
                LEFT_TCP_CONTACT,
                RIGHT_TCP_CONTACT,
                grasp_alpha,
                "grasp",
            )

        time_seconds -= GRASP_SECONDS
        if time_seconds < LIFT_SECONDS:
            alpha = self._smoothstep(time_seconds / LIFT_SECONDS)
            return (
                ROBOT_BASE_GRASP,
                ROBOT_BASE_ROTATION,
                self._lerp_vec3(LEFT_TCP_CONTACT, LEFT_TCP_LIFT, alpha),
                self._lerp_vec3(RIGHT_TCP_CONTACT, RIGHT_TCP_LIFT, alpha),
                1.0,
                "lift",
            )

        time_seconds -= LIFT_SECONDS
        if time_seconds < CARRY_FORWARD_SECONDS:
            alpha = self._smoothstep(time_seconds / CARRY_FORWARD_SECONDS)
            return (
                self._lerp_vec3(ROBOT_BASE_GRASP, ROBOT_BASE_FORWARD, alpha),
                ROBOT_BASE_ROTATION,
                LEFT_TCP_LIFT,
                RIGHT_TCP_LIFT,
                1.0,
                "carry_forward",
            )

        time_seconds -= CARRY_FORWARD_SECONDS
        if time_seconds < CARRY_TURN_ONE_SECONDS:
            alpha = self._smoothstep(time_seconds / CARRY_TURN_ONE_SECONDS)
            return (
                ROBOT_BASE_FORWARD,
                _quat_from_yaw(ROBOT_BASE_TURN_ONE_YAW * alpha),
                LEFT_TCP_LIFT,
                RIGHT_TCP_LIFT,
                1.0,
                "carry_turn_one",
            )

        time_seconds -= CARRY_TURN_ONE_SECONDS
        if time_seconds < CARRY_TURN_TWO_SECONDS:
            alpha = self._smoothstep(time_seconds / CARRY_TURN_TWO_SECONDS)
            yaw = ROBOT_BASE_TURN_ONE_YAW * (1.0 - alpha) + ROBOT_BASE_TURN_TWO_YAW * alpha
            return (
                ROBOT_BASE_FORWARD,
                _quat_from_yaw(yaw),
                LEFT_TCP_LIFT,
                RIGHT_TCP_LIFT,
                1.0,
                "carry_turn_two",
            )

        time_seconds -= CARRY_TURN_TWO_SECONDS
        if time_seconds < CARRY_FINAL_SECONDS:
            alpha = self._smoothstep(time_seconds / CARRY_FINAL_SECONDS)
            yaw = ROBOT_BASE_TURN_TWO_YAW * (1.0 - alpha) + ROBOT_BASE_CARRY_END_YAW * alpha
            return (
                self._lerp_vec3(ROBOT_BASE_FORWARD, ROBOT_BASE_CARRY_END, alpha),
                _quat_from_yaw(yaw),
                LEFT_TCP_LIFT,
                RIGHT_TCP_LIFT,
                1.0,
                "carry_final",
            )

        time_seconds -= CARRY_FINAL_SECONDS
        carry_end_rotation = _quat_from_yaw(ROBOT_BASE_CARRY_END_YAW)
        if time_seconds < LOWER_SECONDS:
            alpha = self._smoothstep(time_seconds / LOWER_SECONDS)
            return (
                ROBOT_BASE_CARRY_END,
                carry_end_rotation,
                self._lerp_vec3(LEFT_TCP_LIFT, LEFT_TCP_PLACE, alpha),
                self._lerp_vec3(RIGHT_TCP_LIFT, RIGHT_TCP_PLACE, alpha),
                1.0 * (1.0 - alpha) + PLACEMENT_GRASP_ALPHA * alpha,
                "lower",
            )

        time_seconds -= LOWER_SECONDS
        if time_seconds < PLACEMENT_SETTLE_SECONDS:
            return (
                ROBOT_BASE_CARRY_END,
                carry_end_rotation,
                LEFT_TCP_PLACE,
                RIGHT_TCP_PLACE,
                PLACEMENT_GRASP_ALPHA,
                "settle_placement",
            )

        time_seconds -= PLACEMENT_SETTLE_SECONDS
        if time_seconds < RELEASE_SECONDS:
            alpha = self._smoothstep(time_seconds / RELEASE_SECONDS)
            return (
                ROBOT_BASE_CARRY_END,
                carry_end_rotation,
                LEFT_TCP_PLACE,
                RIGHT_TCP_PLACE,
                PLACEMENT_GRASP_ALPHA * (1.0 - alpha),
                "release",
            )

        time_seconds -= RELEASE_SECONDS
        if time_seconds < BASE_RETREAT_SECONDS:
            alpha = self._smoothstep(time_seconds / BASE_RETREAT_SECONDS)
            return (
                self._lerp_vec3(ROBOT_BASE_CARRY_END, ROBOT_BASE_RETREAT, alpha),
                carry_end_rotation,
                LEFT_TCP_PLACE,
                RIGHT_TCP_PLACE,
                0.0,
                "retreat_base",
            )

        time_seconds -= BASE_RETREAT_SECONDS
        if time_seconds < ARM_RETREAT_SECONDS:
            alpha = self._smoothstep(time_seconds / ARM_RETREAT_SECONDS)
            return (
                ROBOT_BASE_RETREAT,
                carry_end_rotation,
                self._lerp_vec3(LEFT_TCP_PLACE, LEFT_TCP_RETREAT, alpha),
                self._lerp_vec3(RIGHT_TCP_PLACE, RIGHT_TCP_RETREAT, alpha),
                0.0,
                "retreat_arms",
            )

        time_seconds -= ARM_RETREAT_SECONDS
        _ = min(time_seconds, HOLD_SECONDS)
        return ROBOT_BASE_RETREAT, carry_end_rotation, LEFT_TCP_RETREAT, RIGHT_TCP_RETREAT, 0.0, "hold"

    def _sample_tcp_rotations(self, time_seconds: float) -> tuple[wp.quat, wp.quat]:
        arm_retreat_start = (
            SETTLE_SECONDS
            + APPROACH_SECONDS
            + GRASP_SECONDS
            + LIFT_SECONDS
            + CARRY_SECONDS
            + LOWER_SECONDS
            + PLACEMENT_SETTLE_SECONDS
            + RELEASE_SECONDS
            + BASE_RETREAT_SECONDS
        )
        if time_seconds <= arm_retreat_start:
            return LEFT_TCP_ROTATION, RIGHT_TCP_ROTATION
        alpha = self._smoothstep((time_seconds - arm_retreat_start) / ARM_RETREAT_SECONDS)
        return (
            self._lerp_quat(LEFT_TCP_ROTATION, LEFT_TCP_RETREAT_ROTATION, alpha),
            self._lerp_quat(RIGHT_TCP_ROTATION, RIGHT_TCP_RETREAT_ROTATION, alpha),
        )

    def _set_ik_targets(
        self,
        left: wp.vec3,
        right: wp.vec3,
        left_rotation: wp.quat = LEFT_TCP_ROTATION,
        right_rotation: wp.quat = RIGHT_TCP_ROTATION,
    ) -> None:
        self.left_position_objective.set_target_position(0, left)
        self.left_rotation_objective.set_target_rotation(0, self._quat_vector(left_rotation))
        self.right_position_objective.set_target_position(0, right)
        self.right_rotation_objective.set_target_rotation(0, self._quat_vector(right_rotation))

    def _restore_locked_ik_q(self) -> None:
        wp.launch(
            _lock_ik_q,
            self.ik_lock_indices.shape[0],
            [self.ik_q, self.ik_lock_indices, self.ik_lock_values],
            device=self.device,
        )

    def _copy_ik_to_scene(self, destination: wp.array[float]) -> None:
        wp.launch(
            _copy_indexed_q,
            self.ik_source_indices.shape[0],
            [self.ik_q[0], self.ik_source_indices, self.scene_destination_indices, destination],
            device=self.device,
        )

    def _write_hand_pose(self, grasp_alpha: float, destination: wp.array[float]) -> None:
        wp.launch(
            _write_indexed_lerp,
            self.hand_q_indices.shape[0],
            [self.hand_q_indices, self.hand_q_open, self.hand_q_grasp, grasp_alpha, destination],
            device=self.device,
        )

    def _write_root_pose(
        self,
        position: wp.vec3,
        destination: wp.array[float],
        rotation: wp.quat = ROBOT_BASE_ROTATION,
    ) -> None:
        wp.launch(
            _write_free_root_pose,
            1,
            [self.root_q_start, position, rotation, destination],
            device=self.device,
        )

    def _ik_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.ik_model.joint_limit_lower.numpy().copy()
        upper = self.ik_model.joint_limit_upper.numpy().copy()
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        qd_start = self.ik_model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for joint, label in enumerate(self.ik_model.joint_label):
            if label in controlled:
                continue
            qd_begin = int(qd_start[joint])
            qd_end = int(qd_start[joint + 1])
            q_begin = int(q_start[joint])
            for offset in range(qd_end - qd_begin):
                lower[qd_begin + offset] = q[q_begin + offset] - 1.0e-4
                upper[qd_begin + offset] = q[q_begin + offset] + 1.0e-4
        return lower, upper

    def _ik_locked_q(self) -> tuple[wp.array, wp.array]:
        q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices: list[int] = []
        for joint, label in enumerate(self.ik_model.joint_label):
            if label in controlled:
                continue
            indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.device),
        )

    def _joint_coordinate_mapping(self) -> tuple[list[int], list[int]]:
        ik_q_start = self.ik_model.joint_q_start.numpy()
        scene_q_start = self.model.joint_q_start.numpy()
        scene_joints = {label: joint for joint, label in enumerate(self.model.joint_label)}
        source: list[int] = []
        destination: list[int] = []
        for ik_joint, label in enumerate(self.ik_model.joint_label):
            scene_joint = scene_joints.get(label)
            if scene_joint is None:
                continue
            ik_count = int(ik_q_start[ik_joint + 1] - ik_q_start[ik_joint])
            scene_count = int(scene_q_start[scene_joint + 1] - scene_q_start[scene_joint])
            if ik_count != scene_count:
                raise ValueError(f"Coordinate count differs for {label}: IK={ik_count}, scene={scene_count}")
            for offset in range(ik_count):
                source.append(int(ik_q_start[ik_joint]) + offset)
                destination.append(int(scene_q_start[scene_joint]) + offset)
        if not source:
            raise RuntimeError("No W1 joint coordinates could be mapped from IK to the scene")
        return source, destination

    def _robot_root_q_start(self) -> int:
        joint_type = self.model.joint_type.numpy()
        joint_child = self.model.joint_child.numpy()
        q_start = self.model.joint_q_start.numpy()
        for joint in range(self.model.joint_count):
            if int(joint_type[joint]) == int(newton.JointType.FREE) and int(joint_child[joint]) < self.robot_body_end:
                return int(q_start[joint])
        raise RuntimeError("The floating W1 root joint was not found")

    @staticmethod
    def _set_builder_joint_coordinate(builder: newton.ModelBuilder, name: str, value: float) -> None:
        joint = next(
            (index for index, label in enumerate(builder.joint_label) if label.endswith("/" + name)),
            None,
        )
        if joint is None:
            raise ValueError(f"W1 joint is missing: {name}")
        builder.joint_q[int(builder.joint_q_start[joint])] = float(value)

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    @staticmethod
    def _quat_vector(value: wp.quat) -> wp.vec4:
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def _lerp_vec3(a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
        return wp.vec3(
            float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
            float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
            float(a[2]) * (1.0 - alpha) + float(b[2]) * alpha,
        )

    @staticmethod
    def _lerp_quat(a: wp.quat, b: wp.quat, alpha: float) -> wp.quat:
        values = np.asarray(
            [
                float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
                float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
                float(a[2]) * (1.0 - alpha) + float(b[2]) * alpha,
                float(a[3]) * (1.0 - alpha) + float(b[3]) * alpha,
            ],
            dtype=np.float64,
        )
        values /= np.linalg.norm(values)
        return wp.quat(float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def create_parser():
        """Create command-line options for the realtime push-chair scene."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
        parser.add_argument("--chair-urdf", type=Path, default=DEFAULT_CHAIR_URDF)
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=DEFAULT_IK_ITERATIONS,
            help="Realtime bimanual IK iterations per displayed frame.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture only the fixed-topology physics substeps; IK remains realtime.",
        )
        return parser


def main():
    """Run the realtime chair-pushing example."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
