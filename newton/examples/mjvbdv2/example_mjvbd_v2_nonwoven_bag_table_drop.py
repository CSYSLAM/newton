# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulate a moving Dexforce W1 beside a nonwoven bag on a table.

The Z-up ``nonwoven_5k.obj`` garment starts at a contact-safe clearance above
a visible static table and settles under gravity. A complete kinematic W1
starts in its normal standing pose with both hands open, then slowly cycles
both arms to exercise the same prescribed-motion path used by teleoperation.
Formed-shape elasticity models the nonwoven material's structural memory.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_nonwoven_bag_table_drop --viewer gl
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
from newton.solvers import SolverMJVBDV2

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
BAG_OBJ_PATH = ASSET_ROOT / "style3d_probe" / "bag" / "nonwoven_5k" / "nonwoven_5k.obj"
DEFAULT_ROBOT_URDF = ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"

FPS = 60
SIM_SUBSTEPS = 10
VBD_ITERATIONS = 30

ROBOT_BASE_POSITION = wp.vec3(0.0, 0.0, 0.0)
ROBOT_BASE_ROTATION = wp.quat_identity()
ROBOT_MOTION_SETTLE_SECONDS = 0.5
ROBOT_MOTION_PERIOD_SECONDS = 4.0
BAG_CENTER_X = 0.6041
BAG_YAW = 0.5 * math.pi

BAG_SCALE = 1.0
BAG_TABLE_CLEARANCE = 0.004
BAG_AREAL_DENSITY = 0.01
BAG_PARTICLE_RADIUS = 0.0015
BAG_TRI_KE = 1.0e10
BAG_TRI_KA = 1.0e10
BAG_TRI_KD = 10.0
BAG_EDGE_KE = 1.0e5
BAG_EDGE_KD = 20.0
BAG_COLOR = (0.86, 0.50, 0.16)
BAG_OPACITY = 1.0
AIR_DRAG_RATE = 6.0
REST_SHAPE_STIFFNESS = 1.0e4
REST_SHAPE_DAMPING = 200.0
REST_SHAPE_TARGET_CLEARANCE = 0.003

TABLE_TOP_Z = 0.85
TABLE_COLOR = (0.34, 0.20, 0.10)
TABLE_CENTER = wp.vec3(0.72, 0.0, 0.82)
TABLE_HALF_EXTENTS = (0.48, 0.45, 0.03)
TABLE_LEG_HALF_EXTENTS = (0.035, 0.035, 0.395)
TABLE_CONTACT_MARGIN = 0.003
TABLE_CONTACT_KE = 4.0e8
TABLE_CONTACT_KD = 100.0
TABLE_FRICTION = 0.8
GROUND_COLOR = (0.16, 0.18, 0.21)

CAMERA_POSITION = wp.vec3(2.00, -2.45, 1.70)
CAMERA_PITCH = -12.0
CAMERA_YAW = 132.0
CAMERA_FOV = 44.0

HAND_CONTACT_KE = 2.0e6
HAND_CONTACT_KD = 200.0
HAND_CONTACT_FRICTION = 0.8
HAND_CONTACT_MARGIN = 0.004

SOFT_CONTACT_KE = 2.0e6
SOFT_CONTACT_KD = 200.0
SOFT_CONTACT_FRICTION = 0.35
SOFT_CONTACT_MARGIN = 0.008
SOFT_CONTACT_MAX = 65536
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
SELF_CONTACT_MARGIN = 0.003


def _prepare_bag_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load continuous simulation topology and authored render normals."""
    points = []
    normals = []
    face_corners = []
    with path.open(encoding="utf-8", errors="strict") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            if line.startswith("v "):
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"Invalid OBJ vertex at {path}:{line_number}")
                points.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("vn "):
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"Invalid OBJ normal at {path}:{line_number}")
                normals.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    raise ValueError(f"OBJ face must be triangular at {path}:{line_number}")
                triangle_corners = []
                for corner in corners:
                    corner_indices = corner.split("/")
                    vertex_index = int(corner_indices[0])
                    if vertex_index == 0:
                        raise ValueError(f"OBJ vertex indices are one-based at {path}:{line_number}")
                    vertex_index = vertex_index - 1 if vertex_index > 0 else len(points) + vertex_index
                    if len(corner_indices) < 3 or not corner_indices[2]:
                        raise ValueError(f"OBJ face is missing a normal index at {path}:{line_number}")
                    normal_index = int(corner_indices[2])
                    normal_index = normal_index - 1 if normal_index > 0 else len(normals) + normal_index
                    triangle_corners.append((vertex_index, normal_index))
                face_corners.append(triangle_corners)

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    simulation_indices = np.asarray(
        [[vertex_index for vertex_index, _ in triangle] for triangle in face_corners], dtype=np.int32
    ).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"OBJ mesh points have an invalid shape in {path}: {points.shape}")
    if simulation_indices.ndim != 1 or simulation_indices.size % 3 != 0:
        raise ValueError(f"OBJ mesh must be fully triangulated: {path}")
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) == 0:
        raise ValueError(f"OBJ normals have an invalid shape in {path}: {normals.shape}")

    points[:, :2] -= 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
    points[:, 2] -= points[:, 2].min()
    points *= scale

    render_vertex_lookup: dict[tuple[int, int], int] = {}
    render_particle_indices = []
    render_normals = []
    render_indices = []
    for triangle in face_corners:
        for corner in triangle:
            render_vertex = render_vertex_lookup.get(corner)
            if render_vertex is None:
                render_vertex = len(render_particle_indices)
                render_vertex_lookup[corner] = render_vertex
                render_particle_indices.append(corner[0])
                render_normals.append(normals[corner[1]])
            render_indices.append(render_vertex)

    return (
        points,
        simulation_indices,
        np.asarray(render_particle_indices, dtype=np.int32),
        np.asarray(render_indices, dtype=np.int32),
        np.asarray(render_normals, dtype=np.float32),
    )


def _rotate_z(values: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a point or normal array about the Z axis."""
    result = values.copy()
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result[:, 0] = cosine * values[:, 0] - sine * values[:, 1]
    result[:, 1] = sine * values[:, 0] + cosine * values[:, 1]
    return result


@wp.kernel
def _apply_particle_drag(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    drag_rate: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional air drag to all cloth particles."""
    particle = wp.tid()
    particle_f[particle] = particle_f[particle] - drag_rate * particle_mass[particle] * particle_qd[particle]


@wp.kernel
def _apply_rest_shape_elasticity(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    target_q: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional elasticity toward the bag's formed shape."""
    particle = wp.tid()
    restoring_acceleration = stiffness * (target_q[particle] - particle_q[particle])
    damping_acceleration = damping * particle_qd[particle]
    particle_f[particle] = particle_f[particle] + particle_mass[particle] * (
        restoring_acceleration - damping_acceleration
    )


@wp.kernel
def _gather_render_positions(
    particle_q: wp.array[wp.vec3],
    render_particle_indices: wp.array[int],
    render_q: wp.array[wp.vec3],
):
    """Expand simulation positions across authored OBJ normal seams."""
    render_vertex = wp.tid()
    render_q[render_vertex] = particle_q[render_particle_indices[render_vertex]]


@wp.kernel
def _write_indexed_lerp(
    indices: wp.array[wp.int32],
    values_start: wp.array[float],
    values_end: wp.array[float],
    alpha: float,
    q: wp.array[float],
):
    """Interpolate selected robot coordinates into a frame target."""
    index = wp.tid()
    q[indices[index]] = values_start[index] * (1.0 - alpha) + values_end[index] * alpha


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
    inverse_dt: float,
    qd_out: wp.array[float],
):
    """Compute the velocity of each prescribed robot joint."""
    joint = wp.tid()
    q_begin = joint_q_start[joint]
    q_end_index = joint_q_start[joint + 1]
    qd_begin = joint_qd_start[joint]
    qd_end = joint_qd_start[joint + 1]
    if joint_type[joint] == newton.JointType.FREE:
        qd_out[qd_begin + 0] = (q_end[q_begin + 0] - q_start[q_begin + 0]) * inverse_dt
        qd_out[qd_begin + 1] = (q_end[q_begin + 1] - q_start[q_begin + 1]) * inverse_dt
        qd_out[qd_begin + 2] = (q_end[q_begin + 2] - q_start[q_begin + 2]) * inverse_dt
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
        qd_out[qd_begin + 3] = axis[0] * angle * inverse_dt
        qd_out[qd_begin + 4] = axis[1] * angle * inverse_dt
        qd_out[qd_begin + 5] = axis[2] * angle * inverse_dt
    else:
        for coordinate in range(qd_end - qd_begin):
            if q_begin + coordinate < q_end_index:
                qd_out[qd_begin + coordinate] = (
                    q_end[q_begin + coordinate] - q_start[q_begin + coordinate]
                ) * inverse_dt


class Example:
    """Simulate a formed nonwoven bag while a standing W1 moves both arms."""

    HAND_BODY_KEYWORDS = ("j7", "thumb", "index", "middle", "ring", "pinky")
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
    ARM_MOTION_OFFSETS: ClassVar[dict[str, float]] = {
        "LEFT_J2": -0.20,
        "LEFT_J4": -0.35,
        "LEFT_J6": 0.18,
        "RIGHT_J2": 0.20,
        "RIGHT_J4": 0.35,
        "RIGHT_J6": -0.18,
    }

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.bag_initial_bottom_z = TABLE_TOP_Z + BAG_TABLE_CLEARANCE

        self.robot_urdf = Path(args.robot_urdf).expanduser().resolve()
        bag_obj = Path(args.bag_obj).expanduser().resolve()
        for description, path in (("Dexforce W1 URDF", self.robot_urdf), ("Nonwoven bag OBJ", bag_obj)):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")

        bag_vertices, bag_indices, render_particle_indices, render_indices, render_normals = _prepare_bag_mesh(
            bag_obj, args.bag_scale
        )
        # Face the broad side and both handles toward the standing W1.
        bag_vertices = _rotate_z(bag_vertices, BAG_YAW)
        render_normals = _rotate_z(render_normals, BAG_YAW)
        self.initial_particle_positions = bag_vertices + np.array(
            (BAG_CENTER_X, 0.0, self.bag_initial_bottom_z), dtype=np.float32
        )
        self.rest_shape_target_positions = bag_vertices + np.array(
            (BAG_CENTER_X, 0.0, TABLE_TOP_Z + REST_SHAPE_TARGET_CLEARANCE), dtype=np.float32
        )
        self.initial_bag_height = float(np.ptp(bag_vertices[:, 2]))
        self._build_bag_topology(bag_vertices, bag_indices)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        SolverMJVBDV2.register_custom_attributes(builder)
        self._add_robot(builder)

        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=wp.vec3(BAG_CENTER_X, 0.0, self.bag_initial_bottom_z),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices,
            indices=bag_indices,
            density=BAG_AREAL_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
        )
        self.bag_particle_count = builder.particle_count - self.bag_particle_start
        self._add_table(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.device = self.model.device
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_FRICTION
        self.rest_shape_targets = wp.array(
            self.rest_shape_target_positions,
            dtype=wp.vec3,
            device=self.device,
        )

        self.open_q_indices, self.open_q_values = self._open_hand_coordinates(self.model)
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
                "iterations": VBD_ITERATIONS,
                "friction_epsilon": 1.0e-4,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": 48,
                "particle_edge_contact_buffer_size": 96,
                "particle_collision_detection_interval": -1,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": SELF_CONTACT_MARGIN,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "soft_contact_max": SOFT_CONTACT_MAX,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": (
                    self.table_shape_index,
                    *self.hand_particle_shapes,
                    self.ground_shape_index,
                ),
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend != "vbd_kinematic_full":
            raise RuntimeError(
                f"The W1 nonwoven-bag scene requires vbd_kinematic_full, got {self.solver.features.backend}"
            )

        motion_indices, motion_start, motion_end = self._arm_motion_coordinates(self.model)
        self.motion_q_indices_host = motion_indices.copy()
        self.motion_q_start_host = motion_start.copy()
        self.motion_q_indices = wp.array(motion_indices, dtype=wp.int32, device=self.device)
        self.motion_q_start = wp.array(motion_start, dtype=wp.float32, device=self.device)
        self.motion_q_end = wp.array(motion_end, dtype=wp.float32, device=self.device)
        self.motion_body_indices_host = np.asarray(
            [self._body_index(self.model.body_label, "left_j7"), self._body_index(self.model.body_label, "right_j7")],
            dtype=np.int32,
        )
        self.motion_body_start_host = self.state_0.body_q.numpy()[self.motion_body_indices_host, :3].copy()
        self.frame_q_start = wp.clone(self.state_0.joint_q)
        self.frame_q_end = wp.clone(self.state_0.joint_q)
        self.motion_phase = "settle"
        self.max_motion_alpha = 0.0

        self.render_particle_indices = wp.array(
            render_particle_indices + self.bag_particle_start,
            dtype=wp.int32,
            device=self.device,
        )
        self.render_positions = wp.zeros(len(render_particle_indices), dtype=wp.vec3, device=self.device)
        self.render_indices = wp.array(render_indices, dtype=wp.int32, device=self.device)
        self.render_normals = wp.array(render_normals, dtype=wp.vec3, device=self.device)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=CAMERA_POSITION, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV

        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.capture()

    def _build_bag_topology(self, vertices: np.ndarray, indices: np.ndarray) -> None:
        """Cache the local bag edges used by the final stability check."""
        self.bag_triangle_indices = indices.reshape(-1, 3)
        self.bag_edges = np.unique(
            np.sort(
                np.concatenate(
                    (
                        self.bag_triangle_indices[:, (0, 1)],
                        self.bag_triangle_indices[:, (1, 2)],
                        self.bag_triangle_indices[:, (2, 0)],
                    )
                ),
                axis=1,
            ),
            axis=0,
        )
        self.rest_edge_lengths = np.linalg.norm(vertices[self.bag_edges[:, 1]] - vertices[self.bag_edges[:, 0]], axis=1)

    def _add_robot(self, builder: newton.ModelBuilder) -> None:
        """Add the complete W1 and retain only hand-to-cloth collision."""
        builder.default_shape_cfg.ke = HAND_CONTACT_KE
        builder.default_shape_cfg.kd = HAND_CONTACT_KD
        builder.default_shape_cfg.mu = HAND_CONTACT_FRICTION
        builder.default_shape_cfg.margin = HAND_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(ROBOT_BASE_POSITION, ROBOT_BASE_ROTATION),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(articulation_start, builder.articulation_count))
        if len(self.robot_articulations) != 1:
            raise RuntimeError(f"Expected one W1 articulation, got {self.robot_articulations}")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self._set_builder_posture(builder)

        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collision_mask = collide_particles | collide_shapes
        self.hand_particle_shapes: list[int] = []
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
                builder.shape_flags[shape] |= collide_particles
                builder.shape_flags[shape] &= ~collide_shapes
                self.hand_particle_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        if not self.hand_particle_shapes:
            raise RuntimeError("The full W1 URDF did not produce hand particle-collision shapes")

    def _add_table(self, builder: newton.ModelBuilder) -> None:
        """Add a finite tabletop, four legs, and a floor for W1."""
        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=TABLE_CONTACT_KE,
            kd=TABLE_CONTACT_KD,
            mu=TABLE_FRICTION,
            margin=TABLE_CONTACT_MARGIN,
        )
        self.table_shape_index = builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_CENTER, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="nonwoven_bag_tabletop",
        )
        leg_offset_x = TABLE_HALF_EXTENTS[0] - 2.0 * TABLE_LEG_HALF_EXTENTS[0]
        leg_offset_y = TABLE_HALF_EXTENTS[1] - 2.0 * TABLE_LEG_HALF_EXTENTS[1]
        for x_sign in (-1.0, 1.0):
            for y_sign in (-1.0, 1.0):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(
                        wp.vec3(
                            float(TABLE_CENTER[0]) + x_sign * leg_offset_x,
                            y_sign * leg_offset_y,
                            TABLE_LEG_HALF_EXTENTS[2],
                        ),
                        wp.quat_identity(),
                    ),
                    hx=TABLE_LEG_HALF_EXTENTS[0],
                    hy=TABLE_LEG_HALF_EXTENTS[1],
                    hz=TABLE_LEG_HALF_EXTENTS[2],
                    cfg=table_cfg,
                    color=TABLE_COLOR,
                    label="nonwoven_bag_table_leg",
                )
        self.ground_shape_index = builder.add_ground_plane(
            height=0.0,
            color=GROUND_COLOR,
            label="nonwoven_bag_ground",
        )

    def _set_builder_posture(self, builder: newton.ModelBuilder) -> None:
        """Keep the default standing body pose and explicitly open both hands."""
        for side in ("LEFT", "RIGHT"):
            for suffix, value in self.OPEN_HAND_JOINTS.items():
                self._set_builder_joint_coordinate(builder, f"{side}_{suffix}", value)

    def _open_hand_coordinates(self, model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
        """Collect the expected open-finger coordinates for validation."""
        q_start = model.joint_q_start.numpy()
        indices: list[int] = []
        values: list[float] = []
        for side in ("LEFT", "RIGHT"):
            for suffix, value in self.OPEN_HAND_JOINTS.items():
                joint_name = f"{side}_{suffix}"
                joint = next(
                    (index for index, label in enumerate(model.joint_label) if label.endswith("/" + joint_name)),
                    None,
                )
                if joint is None:
                    raise ValueError(f"W1 joint is missing: {joint_name}")
                indices.append(int(q_start[joint]))
                values.append(float(value))
        return np.asarray(indices, dtype=np.int32), np.asarray(values, dtype=np.float32)

    def _arm_motion_coordinates(self, model: newton.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collect a slow, symmetric arm-motion target within the URDF limits."""
        joint_q = model.joint_q.numpy()
        q_start = model.joint_q_start.numpy()
        limit_lower = model.joint_limit_lower.numpy()
        limit_upper = model.joint_limit_upper.numpy()
        qd_start = model.joint_qd_start.numpy()
        indices: list[int] = []
        values_start: list[float] = []
        values_end: list[float] = []
        for name, offset in self.ARM_MOTION_OFFSETS.items():
            joint = next(
                (index for index, label in enumerate(model.joint_label) if label.endswith("/" + name)),
                None,
            )
            if joint is None:
                raise ValueError(f"W1 joint is missing: {name}")
            coordinate = int(q_start[joint])
            dof = int(qd_start[joint])
            start = float(joint_q[coordinate])
            end = start + float(offset)
            if not float(limit_lower[dof]) <= end <= float(limit_upper[dof]):
                raise ValueError(f"W1 motion target for {name} is outside its joint limits: {end:.6g} rad")
            indices.append(coordinate)
            values_start.append(start)
            values_end.append(end)
        return (
            np.asarray(indices, dtype=np.int32),
            np.asarray(values_start, dtype=np.float32),
            np.asarray(values_end, dtype=np.float32),
        )

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
        """Find one W1 body by its URDF-local name."""
        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    def capture(self) -> None:
        """Capture one complete simulation frame on CUDA."""
        self.graph = None
        if not self.use_graph:
            return

        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.graph = capture.graph
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)

    def _prepare_robot_motion(self) -> None:
        """Prepare the next low-speed arm target outside the captured graph."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        if self.sim_time < ROBOT_MOTION_SETTLE_SECONDS:
            motion_alpha = 0.0
            self.motion_phase = "settle"
        else:
            motion_time = self.sim_time - ROBOT_MOTION_SETTLE_SECONDS
            phase = 2.0 * math.pi * motion_time / ROBOT_MOTION_PERIOD_SECONDS
            motion_alpha = 0.5 - 0.5 * math.cos(phase)
            self.motion_phase = "arm_cycle"
        self.max_motion_alpha = max(self.max_motion_alpha, motion_alpha)
        wp.launch(
            _write_indexed_lerp,
            dim=self.motion_q_indices.shape[0],
            inputs=[
                self.motion_q_indices,
                self.motion_q_start,
                self.motion_q_end,
                motion_alpha,
                self.frame_q_end,
            ],
            device=self.device,
        )

    def _simulate_substeps(self) -> None:
        """Advance the moving kinematic W1 and physical nonwoven bag."""
        for substep in range(SIM_SUBSTEPS):
            alpha = (substep + 1) / SIM_SUBSTEPS
            wp.launch(
                _interpolate_joint_q,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _update_joint_velocity,
                dim=self.model.joint_count,
                inputs=[
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
                _apply_particle_drag,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.model.particle_mass, AIR_DRAG_RATE],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            wp.launch(
                _apply_rest_shape_elasticity,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.model.particle_mass,
                    self.rest_shape_targets,
                    REST_SHAPE_STIFFNESS,
                    REST_SHAPE_DAMPING,
                ],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self) -> None:
        """Prepare arm motion and advance one rendered physics frame."""
        self._prepare_robot_motion()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate_substeps()

    def step(self) -> None:
        """Advance the staged W1 table-drop scene by one frame."""
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the complete W1, nonwoven bag, table, and floor."""
        wp.launch(
            _gather_render_positions,
            dim=len(self.render_particle_indices),
            inputs=[self.state_0.particle_q, self.render_particle_indices],
            outputs=[self.render_positions],
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/nonwoven_bag",
            self.render_positions,
            self.render_indices,
            normals=self.render_normals,
            backface_culling=False,
            color=BAG_COLOR,
            roughness=0.78,
            metallic=0.0,
            opacity=BAG_OPACITY,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify the standing, open-hand W1 pose and a stable bag landing."""
        if self.solver.features.backend != "vbd_kinematic_full":
            raise ValueError(f"Unexpected solver backend: {self.solver.features.backend}")
        if len(self.robot_articulations) != 1 or not self.hand_particle_shapes:
            raise ValueError("The scene is missing the full W1 hand-contact articulation")

        joint_q = self.state_0.joint_q.numpy()
        open_hand_error = float(np.max(np.abs(joint_q[self.open_q_indices] - self.open_q_values)))
        if open_hand_error > 1.0e-4:
            raise ValueError(f"The W1 fingers left the open pose: error={open_hand_error:.6g} rad")
        arm_motion = float(np.max(np.abs(joint_q[self.motion_q_indices_host] - self.motion_q_start_host)))
        if self.max_motion_alpha < 0.5 or arm_motion < 0.1:
            raise ValueError(
                f"The W1 arm-motion path was not exercised: alpha={self.max_motion_alpha:.6g}, "
                f"motion={arm_motion:.6g} rad"
            )
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("The W1 body pose is not finite")
        wrist_motion = float(
            np.max(
                np.linalg.norm(
                    body_q[self.motion_body_indices_host, :3] - self.motion_body_start_host,
                    axis=1,
                )
            )
        )
        if wrist_motion < 0.02:
            raise ValueError(f"The W1 FK did not move either rendered wrist: motion={wrist_motion:.6g} m")

        soft_contact_count = int(self.solver.contacts.soft_contact_count.numpy()[0])
        if soft_contact_count >= SOFT_CONTACT_MAX:
            raise ValueError(f"Soft-contact capacity exhausted: {soft_contact_count} >= {SOFT_CONTACT_MAX}")

        bag_slice = slice(self.bag_particle_start, self.bag_particle_start + self.bag_particle_count)
        particle_q = self.state_0.particle_q.numpy()[bag_slice]
        particle_qd = self.state_0.particle_qd.numpy()[bag_slice]
        if not np.all(np.isfinite(particle_q)) or not np.all(np.isfinite(particle_qd)):
            raise ValueError("Nonwoven table-drop state is not finite")
        minimum_z = float(particle_q[:, 2].min())
        if minimum_z < TABLE_TOP_Z - 0.01:
            raise ValueError(f"The nonwoven bag penetrated the table: z={minimum_z:.6g} m")
        if minimum_z > TABLE_TOP_Z + SOFT_CONTACT_MARGIN + BAG_PARTICLE_RADIUS:
            raise ValueError(f"The nonwoven bag did not settle onto the table: z={minimum_z:.6g} m")

        edge_lengths = np.linalg.norm(particle_q[self.bag_edges[:, 1]] - particle_q[self.bag_edges[:, 0]], axis=1)
        edge_stretch_95 = float(np.quantile(edge_lengths / self.rest_edge_lengths, 0.95))
        if edge_stretch_95 > 1.15:
            raise ValueError(f"Nonwoven bag 95th-percentile edge stretch reached {edge_stretch_95:.6g}")

        final_height_ratio = float(np.ptp(particle_q[:, 2])) / self.initial_bag_height
        if final_height_ratio < 0.75:
            raise ValueError(f"Nonwoven bag retained only {final_height_ratio:.2%} of its initial height")

        rest_centered = self.initial_particle_positions - self.initial_particle_positions.mean(axis=0)
        final_centered = particle_q - particle_q.mean(axis=0)
        covariance = rest_centered.T @ final_centered
        left_vectors, _, right_vectors = np.linalg.svd(covariance)
        rotation = right_vectors.T @ left_vectors.T
        if np.linalg.det(rotation) < 0.0:
            right_vectors[-1] *= -1.0
            rotation = right_vectors.T @ left_vectors.T
        shape_rms = float(np.sqrt(np.mean(np.sum((rest_centered @ rotation - final_centered) ** 2, axis=1))))
        if shape_rms > 0.06:
            raise ValueError(f"Nonwoven bag shape RMS deviation reached {shape_rms:.6g} m")

    @staticmethod
    def create_parser():
        """Create command-line options for the staged W1 bag scene."""
        parser = newton.examples.create_parser()
        parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
        parser.add_argument("--bag-obj", type=Path, default=BAG_OBJ_PATH)
        parser.add_argument("--bag-scale", type=float, default=BAG_SCALE)
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture one complete MJVBDV2 display frame on CUDA.",
        )
        parser.set_defaults(num_frames=180)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
