# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD W1 Plug Socket
#
# A kinematic Dexforce W1 approaches, pinches, raises, and inserts a rigid
# plug into a static socket, then opens and withdraws to test whether the
# socket alone retains the plug. Newton IK solves the right arm every frame
# inside the captured frame graph; no joint trajectory is replayed. SolverVBD
# moves the plug only through gravity and contact with the hand, table, and
# socket: there is no attachment force, pose locking, or collision switching.
#
# Command: python -m newton.examples vbd_w1_plug_socket
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik

FPS = 60
DEFAULT_NUM_FRAMES = 780
DEFAULT_SUBSTEPS = 8
DEFAULT_VBD_ITERATIONS = 16
DEFAULT_IK_ITERATIONS = 8
INITIAL_IK_ITERATIONS = 300

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
ROBOT_URDF = ASSET_ROOT / "W1-hand-obj" / "DexforceW1V021_visual_collision.urdf"
PLUG_MESH = ASSET_ROOT / "waic_plug_socket" / "plug0630.obj"
SOCKET_MESH = ASSET_ROOT / "waic_plug_socket" / "socket0624.obj"

ROBOT_BASE_POSITION = wp.vec3(-0.1, -0.5, 0.0)
ROBOT_BASE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi)

SOCKET_POSITION = wp.vec3(0.0, 0.0, 1.0)
PLUG_REST_POSITION = wp.vec3(0.05, 0.0, 0.876)
PLUG_FORWARD_POSITION = wp.vec3(0.05, 0.0, 1.0)
PLUG_INSERTED_POSITION = wp.vec3(-0.01, 0.0, 1.0)
PLUG_TO_GRIP = wp.vec3(0.015, 0.01, 0.0)
HAND_CARRY_CORRECTION = wp.vec3(-0.00055, 0.00206, 0.00009)
HAND_STANDBY_POSITION = wp.vec3(0.18, -0.10, 1.10)
HAND_TOP_OFFSET = wp.vec3(0.0, 0.0, 0.14)
HAND_TARGET_ROTATION = wp.quat(-0.000413, 0.952521, -0.304302, -0.010154)
# Compensate the carried plug's pitch before advancing along the socket axis.
# This changes only the wrist target; the plug remains contact-driven.
HAND_INSERT_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.045) * wp.quat(
    -0.012441, 0.974554, -0.220568, -0.037934
)

TABLE_POSITION = wp.vec3(0.05, 0.13, 0.835)
TABLE_HALF_EXTENTS = (0.28, 0.18, 0.01)
PLUG_MASS = 0.2
PLUG_DENSITY = 1408.43
PLUG_COLOR = (0.10, 0.16, 0.20)
SOCKET_COLOR = (0.68, 0.72, 0.76)
TABLE_COLOR = (0.42, 0.49, 0.56)
GROUND_COLOR = (0.22, 0.25, 0.29)

# Phase durations [s] of the procedural controller.
PHASES = (
    ("settle", 0.5),
    ("approach", 1.4),
    ("descend", 1.0),
    ("grasp", 0.8),
    ("hold_grasp", 0.4),
    ("raise", 1.2),
    ("align", 0.6),
    ("insert", 2.5),
    ("settle_inserted", 0.3),
    ("release", 0.8),
    ("retract", 1.2),
    ("observe", 1.0),
)

CONTACT_KE = 2.0e5
CONTACT_KD = 0.5
CONTACT_MARGIN = 2.0e-4
CONTACT_GAP = 5.0e-4
HAND_FRICTION = 4.0
PLUG_FRICTION = 2.0
SOCKET_FRICTION = 0.05
TABLE_FRICTION = 0.8
GROUND_FRICTION = 0.8
PLUG_SDF_RESOLUTION = 256
PLUG_SDF_BAND = 0.005
RIGID_CONTACT_MAX = 16384
RIGID_BODY_CONTACT_BUFFER_SIZE = 768


RIGHT_THUMB_TIP_OFFSET = wp.vec3(0.0006, 0.0867, 0.0)
RIGHT_INDEX_TIP_OFFSET = wp.vec3(0.0, 0.0492, 0.0086)
RIGHT_ARM = tuple(f"RIGHT_J{i}" for i in range(1, 8))
RIGHT_CONTACT_BODY_KEYWORDS = ("right_thumb", "right_index")

INITIAL_POSTURE = {
    "ANKLE": math.radians(30.0),
    "KNEE": math.radians(-60.0),
    "BUTTOCK": math.radians(30.0),
    "LEFT_J1": math.radians(14.0),
    "LEFT_J2": math.radians(-75.0),
    "LEFT_J3": 0.0,
    "LEFT_J4": math.radians(-30.0),
    "LEFT_J5": 0.0,
    "LEFT_J6": 0.0,
    "LEFT_J7": 0.0,
    "NECK2": math.radians(-30.0),
}
OPEN_HAND_JOINTS = {
    "RIGHT_HAND_INDEX": 0.2,
    "RIGHT_INDEX_PIP": 0.2,
    "RIGHT_HAND_THUMB1": 0.2,
    "RIGHT_HAND_THUMB2": 1.4,
}
GRASP_HAND_JOINTS = {
    "RIGHT_HAND_INDEX": 0.47,
    "RIGHT_INDEX_PIP": 0.47,
    "RIGHT_HAND_THUMB1": 0.35,
    "RIGHT_HAND_THUMB2": 1.4,
}

CAMERA_POSITION = wp.vec3(0.50, -0.52, 1.15)
CAMERA_PITCH = -12.0
CAMERA_YAW = 137.0


@wp.kernel
def _write_frame_inputs(
    position: wp.vec3,
    rotation: wp.quat,
    grasp: float,
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.quat],
    grasp_alpha: wp.array[float],
):
    """Store this frame's sampled controller targets in persistent device buffers."""
    target_position[0] = position
    target_rotation[0] = rotation
    grasp_alpha[0] = grasp


@wp.kernel
def _unpack_ik_targets(
    target_position: wp.array[wp.vec3],
    target_rotation: wp.array[wp.quat],
    position_objective: wp.array[wp.vec3],
    rotation_objective: wp.array[wp.vec4],
):
    """Feed the stored targets to the IK position and rotation objectives."""
    q = target_rotation[0]
    position_objective[0] = target_position[0]
    rotation_objective[0] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def _set_indexed(q: wp.array2d[float], indices: wp.array[wp.int32], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _copy_indexed(
    source: wp.array2d[float],
    source_indices: wp.array[wp.int32],
    destination_indices: wp.array[wp.int32],
    destination: wp.array[float],
):
    """Copy IK coordinates into the matching scene coordinates."""
    i = wp.tid()
    destination[destination_indices[i]] = source[0, source_indices[i]]


@wp.kernel
def _write_hand_pose(
    indices: wp.array[wp.int32],
    open_q: wp.array[float],
    grasp_q: wp.array[float],
    grasp_alpha: wp.array[float],
    q: wp.array[float],
):
    """Blend the authored open and pinch finger poses."""
    i = wp.tid()
    alpha = grasp_alpha[0]
    q[indices[i]] = open_q[i] * (1.0 - alpha) + grasp_q[i] * alpha


@wp.kernel
def _interpolate_robot(
    indices: wp.array[wp.int32],
    q_start: wp.array[float],
    q_end: wp.array[float],
    alpha: float,
    inv_frame_dt: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    """Interpolate the robot's one-DOF coordinates within a frame and set their velocity."""
    i = indices[wp.tid()]
    q[i] = q_start[i] * (1.0 - alpha) + q_end[i] * alpha
    qd[i] = (q_end[i] - q_start[i]) * inv_frame_dt


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
    return a * (1.0 - alpha) + b * alpha


def _phase_start(name: str) -> float:
    start = 0.0
    for phase, duration in PHASES:
        if phase == name:
            return start
        start += duration
    raise KeyError(name)


def sample_controller(time: float) -> tuple[wp.vec3, wp.quat, float, str]:
    """Return the fingertip-midpoint target, wrist rotation, pinch amount, and phase."""
    top = PLUG_REST_POSITION + HAND_TOP_OFFSET
    grip = PLUG_REST_POSITION + PLUG_TO_GRIP
    forward = PLUG_FORWARD_POSITION + PLUG_TO_GRIP
    aligned = forward + HAND_CARRY_CORRECTION
    inserted = PLUG_INSERTED_POSITION + PLUG_TO_GRIP + HAND_CARRY_CORRECTION
    motion = {
        "settle": (HAND_STANDBY_POSITION, HAND_STANDBY_POSITION, 0.0, 0.0),
        "approach": (HAND_STANDBY_POSITION, top, 0.0, 0.0),
        "descend": (top, grip, 0.0, 0.0),
        "grasp": (grip, grip, 0.0, 1.0),
        "hold_grasp": (grip, grip, 1.0, 1.0),
        "raise": (grip, forward, 1.0, 1.0),
        "align": (forward, aligned, 1.0, 1.0),
        "insert": (aligned, inserted, 1.0, 1.0),
        "settle_inserted": (inserted, inserted, 1.0, 1.0),
        "release": (inserted, inserted, 1.0, 0.0),
        "retract": (inserted, HAND_STANDBY_POSITION, 0.0, 0.0),
        "observe": (HAND_STANDBY_POSITION, HAND_STANDBY_POSITION, 0.0, 0.0),
    }
    # Rotate the wrist into axial alignment before the plug reaches the socket.
    align_start = _phase_start("align")
    align = _smoothstep((time - align_start) / dict(PHASES)["align"])
    rotation = wp.quat_slerp(HAND_TARGET_ROTATION, HAND_INSERT_ROTATION, align)

    phase, alpha = PHASES[-1][0], 1.0
    for name, duration in PHASES:
        if time < duration:
            phase, alpha = name, _smoothstep(time / duration)
            break
        time -= duration
    a, b, grasp_a, grasp_b = motion[phase]
    return _lerp(a, b, alpha), rotation, grasp_a * (1.0 - alpha) + grasp_b * alpha, phase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = args.substeps
        if self.sim_substeps < 2 or self.sim_substeps % 2:
            raise ValueError("--substeps must be even so the captured state swap returns to its start")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.phase = "settle"
        for label, path in (("W1 URDF", ROBOT_URDF), ("plug mesh", PLUG_MESH), ("socket mesh", SOCKET_MESH)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self._build_scene()
        self.device = self.model.device
        self._build_ik()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._initialize_pose()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            contact_matching="latest",
            rigid_contact_max=RIGID_CONTACT_MAX,
            include_static_kinematic_pairs=False,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=RIGID_BODY_CONTACT_BUFFER_SIZE,
            friction_epsilon=1.0e-4,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POSITION, CAMERA_PITCH, CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 38.0

        self.use_graph = self.device.is_cuda and not args.no_cuda_graph
        self.graph = None

    # Scene -------------------------------------------------------------------

    def _build_scene(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = CONTACT_GAP
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = HAND_FRICTION
        builder.default_shape_cfg.margin = CONTACT_MARGIN

        builder.add_urdf(
            str(ROBOT_URDF),
            xform=wp.transform(ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        for joint in range(self.robot_joint_end):
            # The pinch commands every finger coordinate that touches the plug,
            # so the solver's mimic projection has nothing to do on kinematic links.
            builder.joint_mimic_joint[joint] = -1
        for name, value in {**INITIAL_POSTURE, **OPEN_HAND_JOINTS}.items():
            builder.joint_q[builder.joint_q_start[self._joint_index(builder.joint_label, name)]] = value
        self._configure_hand_colliders(builder)

        plug_mesh = newton.Mesh.create_from_file(str(PLUG_MESH), compute_inertia=True, is_solid=True)
        plug_mesh.build_sdf(
            narrow_band_range=(-PLUG_SDF_BAND, PLUG_SDF_BAND),
            max_resolution=PLUG_SDF_RESOLUTION,
            margin=CONTACT_MARGIN + CONTACT_GAP,
        )
        contact = {"ke": CONTACT_KE, "kd": CONTACT_KD, "margin": CONTACT_MARGIN, "gap": CONTACT_GAP}
        self.plug_body = builder.add_body(xform=wp.transform(PLUG_REST_POSITION, wp.quat_identity()), label="plug")
        self.plug_shape = builder.add_shape_mesh(
            self.plug_body,
            mesh=plug_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(density=PLUG_DENSITY, mu=PLUG_FRICTION, is_solid=True, **contact),
            color=PLUG_COLOR,
            label="plug",
        )
        # Keep the mesh-derived centre of mass; scale inertia to the real plug mass.
        scale = PLUG_MASS / float(builder.body_mass[self.plug_body])
        inertia = np.asarray(builder.body_inertia[self.plug_body], dtype=np.float64).reshape(3, 3) * scale
        builder.body_mass[self.plug_body] = PLUG_MASS
        builder.body_inertia[self.plug_body] = wp.mat33(*inertia.reshape(-1).tolist())

        builder.add_shape_mesh(
            -1,
            xform=wp.transform(SOCKET_POSITION, wp.quat_identity()),
            mesh=newton.Mesh.create_from_file(str(SOCKET_MESH), compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=SOCKET_FRICTION, is_solid=False, **contact),
            color=SOCKET_COLOR,
            label="socket",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POSITION, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=TABLE_FRICTION, **contact),
            color=TABLE_COLOR,
            label="table",
        )
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=CONTACT_KE, kd=CONTACT_KD, mu=GROUND_FRICTION, margin=0.0),
            color=GROUND_COLOR,
            label="ground",
        )

        # The plug is the only body VBD integrates and it has no joints, so one
        # body color is a valid ordering; the kinematic robot links are never solved.
        builder.body_color_groups = [np.arange(builder.body_count, dtype=np.int32)]
        self.model = builder.finalize(requires_grad=False)

    def _configure_hand_colliders(self, builder):
        """Keep only the right thumb and index colliders, as convex hulls."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        mask = collide_shapes | int(newton.ShapeFlags.COLLIDE_PARTICLES)

        def is_contact_body(shape):
            body = int(builder.shape_body[shape])
            return body >= 0 and any(k in builder.body_label[body].lower() for k in RIGHT_CONTACT_BODY_KEYWORDS)

        meshes = [
            shape
            for shape in range(builder.shape_count)
            if is_contact_body(shape)
            and builder.shape_flags[shape] & collide_shapes
            and builder.shape_type[shape] == newton.GeoType.MESH
        ]
        if not meshes:
            raise RuntimeError("The W1 asset did not provide right-hand collision meshes")
        # Each thumb and index link is nearly convex, so one hull per link keeps
        # the pinch pads while giving the narrow phase cheap convex colliders.
        builder.approximate_meshes(method="convex_hull", shape_indices=meshes, keep_visual_shapes=False)
        self.robot_shape_end = builder.shape_count
        self.hand_shapes = []
        for shape in range(builder.shape_count):
            if is_contact_body(shape) and builder.shape_flags[shape] & mask:
                builder.shape_flags[shape] = (builder.shape_flags[shape] | collide_shapes) & ~int(
                    newton.ShapeFlags.COLLIDE_PARTICLES
                )
                self.hand_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~mask
        if not self.hand_shapes:
            raise RuntimeError("The W1 asset did not produce right-hand collision shapes")

    # Robot control -------------------------------------------------------------

    @staticmethod
    def _joint_index(labels, name):
        return next(i for i, label in enumerate(labels) if label.endswith("/" + name))

    def _build_ik(self):
        """Build an uncollapsed fixed-base W1 for per-frame right-arm IK."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_urdf(
            str(ROBOT_URDF),
            xform=wp.transform(ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        for name, value in {**INITIAL_POSTURE, **OPEN_HAND_JOINTS}.items():
            builder.joint_q[builder.joint_q_start[self._joint_index(builder.joint_label, name)]] = value
        self.ik_model = builder.finalize(device=self.device)
        labels = self.ik_model.body_label
        hand = next(i for i, label in enumerate(labels) if label.endswith("/right_hand_base"))
        thumb = next(i for i, label in enumerate(labels) if label.endswith("/right_thumb_dist"))
        index = next(i for i, label in enumerate(labels) if label.endswith("/right_index_dist"))

        # The IK target is the open pinch's fingertip midpoint, fixed in the palm frame.
        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        body_q = state.body_q.numpy()
        midpoint = 0.5 * (
            wp.transform_point(wp.transform(*body_q[thumb]), RIGHT_THUMB_TIP_OFFSET)
            + wp.transform_point(wp.transform(*body_q[index]), RIGHT_INDEX_TIP_OFFSET)
        )
        offset = wp.transform_point(wp.transform_inverse(wp.transform(*body_q[hand])), midpoint)
        self.position_objective = ik.IKObjectivePosition(
            hand, offset, wp.array([HAND_STANDBY_POSITION], dtype=wp.vec3, device=self.device)
        )
        q = HAND_TARGET_ROTATION
        self.rotation_objective = ik.IKObjectiveRotation(
            hand, wp.quat_identity(), wp.array([wp.vec4(q[0], q[1], q[2], q[3])], dtype=wp.vec4, device=self.device)
        )

        # Only the right arm moves in IK; the mask gives every other coordinate an
        # exactly-zero update, so the arm joints keep their real limits.
        ik_q = self.ik_model.joint_q.numpy()
        q_start = self.ik_model.joint_q_start.numpy()
        qd_start = self.ik_model.joint_qd_start.numpy()
        dofs = self.ik_model.joint_dof_count
        dof_mask = np.zeros(dofs, dtype=bool)
        locked = []
        for joint, label in enumerate(self.ik_model.joint_label):
            if label.rsplit("/", 1)[-1] in RIGHT_ARM:
                dof_mask[qd_start[joint] : qd_start[joint + 1]] = True
            else:
                locked.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
        limits = ik.IKObjectiveJointLimit(
            wp.array(self.ik_model.joint_limit_lower.numpy()[:dofs], dtype=wp.float32, device=self.device),
            wp.array(self.ik_model.joint_limit_upper.numpy()[:dofs], dtype=wp.float32, device=self.device),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            joint_dof_mask=wp.array(dof_mask, dtype=wp.bool, device=self.device),
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.lock_indices = wp.array(locked, dtype=wp.int32, device=self.device)
        self.lock_values = wp.array(ik_q[locked], dtype=float, device=self.device)

        # The simulated robot collapses fixed joints; movable joints keep their labels.
        scene_q_start = self.model.joint_q_start.numpy()
        scene_joints = {label: j for j, label in enumerate(self.model.joint_label[: self.robot_joint_end])}
        source, destination = [], []
        for joint, label in enumerate(self.ik_model.joint_label):
            scene_joint = scene_joints.get(label)
            count = int(q_start[joint + 1] - q_start[joint])
            if scene_joint is None or count == 0:
                continue
            if int(scene_q_start[scene_joint + 1] - scene_q_start[scene_joint]) != count:
                raise RuntimeError(f"Coordinate count differs for {label}")
            source.extend(range(int(q_start[joint]), int(q_start[joint]) + count))
            destination.extend(range(int(scene_q_start[scene_joint]), int(scene_q_start[scene_joint]) + count))
        self.ik_source = wp.array(source, dtype=wp.int32, device=self.device)
        self.scene_destination = wp.array(destination, dtype=wp.int32, device=self.device)
        robot_q_end = int(scene_q_start[self.robot_joint_end])
        if robot_q_end != int(self.model.joint_qd_start.numpy()[self.robot_joint_end]):
            raise RuntimeError("Expected one-DOF robot joints")
        self.robot_q_indices = wp.array(np.arange(robot_q_end), dtype=wp.int32, device=self.device)

        hand_q = [int(scene_q_start[self._joint_index(self.model.joint_label, name)]) for name in OPEN_HAND_JOINTS]
        self.hand_q_indices = wp.array(hand_q, dtype=wp.int32, device=self.device)
        self.hand_q_open = wp.array(list(OPEN_HAND_JOINTS.values()), dtype=float, device=self.device)
        self.hand_q_grasp = wp.array([GRASP_HAND_JOINTS[n] for n in OPEN_HAND_JOINTS], dtype=float, device=self.device)

        # Per-frame IK runs inside the captured frame graph; the host only
        # writes the sampled targets into these buffers.
        self.target_position = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.target_rotation = wp.zeros(1, dtype=wp.quat, device=self.device)
        self.grasp_alpha = wp.zeros(1, dtype=float, device=self.device)

    def _initialize_pose(self):
        """Solve the standby pose before the first frame."""
        self.position_objective.set_target_position(0, HAND_STANDBY_POSITION)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        wp.launch(_set_indexed, self.lock_indices.shape[0], [self.ik_q, self.lock_indices, self.lock_values])
        wp.launch(
            _copy_indexed,
            self.ik_source.shape[0],
            [self.ik_q, self.ik_source, self.scene_destination, self.model.joint_q],
        )
        self.model.joint_qd.zero_()
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(self.model.joint_q)
            state.joint_qd.zero_()
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.frame_q_start = wp.clone(self.model.joint_q)
        self.frame_q_end = wp.clone(self.model.joint_q)

    def _write_frame_inputs(self):
        position, rotation, grasp, self.phase = sample_controller(self.sim_time)
        wp.launch(
            _write_frame_inputs,
            1,
            [position, rotation, grasp, self.target_position, self.target_rotation, self.grasp_alpha],
        )

    # Simulation ----------------------------------------------------------------

    def _simulate_frame(self):
        wp.launch(
            _unpack_ik_targets,
            1,
            [
                self.target_position,
                self.target_rotation,
                self.position_objective.target_positions,
                self.rotation_objective.target_rotations,
            ],
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.args.ik_iterations)
        wp.launch(_set_indexed, self.lock_indices.shape[0], [self.ik_q, self.lock_indices, self.lock_values])
        wp.copy(self.frame_q_start, self.frame_q_end)
        wp.launch(
            _copy_indexed,
            self.ik_source.shape[0],
            [self.ik_q, self.ik_source, self.scene_destination, self.frame_q_end],
        )
        wp.launch(
            _write_hand_pose,
            self.hand_q_indices.shape[0],
            [self.hand_q_indices, self.hand_q_open, self.hand_q_grasp, self.grasp_alpha, self.frame_q_end],
        )
        for substep in range(self.sim_substeps):
            wp.launch(
                _interpolate_robot,
                self.robot_q_indices.shape[0],
                [
                    self.robot_q_indices,
                    self.frame_q_start,
                    self.frame_q_end,
                    (substep + 1) / self.sim_substeps,
                    1.0 / self.frame_dt,
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
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._write_frame_inputs()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            # The first frame runs uncaptured so lazily sized buffers exist; the
            # capture then records IK, the finger command, and every substep.
            self._simulate_frame()
            if self.use_graph:
                with wp.ScopedCapture() as capture:
                    self._simulate_frame()
                self.graph = capture.graph
        self.frame_index += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("The plug/socket scene contains a non-finite body pose")

    def test_final(self):
        if not np.all(np.isfinite(self.ik_q.numpy())):
            raise ValueError("IK returned a non-finite coordinate")
        if self.sim_time < _phase_start("observe"):
            return
        pose = self.state_0.body_q.numpy()[self.plug_body]
        position = pose[:3]
        if not -0.02 < position[0] < 0.01:
            raise ValueError(f"Contact did not insert the plug: position={position}")
        if abs(position[1]) > 0.015 or abs(position[2] - 1.0) > 0.02:
            raise ValueError(f"The plug missed the socket axis: position={position}")
        if abs(pose[6]) < math.cos(math.radians(10.0)):
            raise ValueError(f"The plug entered the socket at excessive tilt: pose={pose}")
        print(f"[W1PlugSocket] PASS: plug retained in the socket at {np.round(position, 4).tolist()}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        parser.add_argument("--ik-iterations", type=int, default=DEFAULT_IK_ITERATIONS)
        parser.add_argument("--no-cuda-graph", action="store_true", help="Run frames without CUDA graph capture.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
