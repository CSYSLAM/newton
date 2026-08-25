# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pick up a soft Armadillo and release it above counter-rotating gears.

The scene embeds a tuned W1 right-hand grasp and builds the DexSim Armadillo,
table, and roller geometry directly. The Armadillo initially rests on the
table. The hand approaches, closes through physical contact, lifts the
Armadillo, carries it over the rollers to the right of the table, and opens.
No attachment or particle teleport is used.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher --viewer gl
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJVBDV2

FPS = 60
SIM_SUBSTEPS = 10
VBD_ITERATIONS = 10
DEFAULT_NUM_FRAMES = 1500

RIGHT_HAND_URDF = Path(__file__).resolve().parents[3] / "assets" / "W1_right_hand" / "DexforceW1_right_hand.urdf"
ARMADILLO_ASSET = Path(__file__).resolve().parents[3] / "assets" / "gear_crusher_assets" / "Armadilo_15K.1.vtk"

TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
TABLE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
GROUND_HEIGHT = -0.00377202

ARMADILLO_CENTER_X = -0.14931439
ARMADILLO_CENTER_Y = -2.76669516
CRUSHER_SCENE_SCALE = 0.16
ARMADILLO_SCALE = 0.20
ARMADILLO_INITIAL_CLEARANCE = 0.06
ARMADILLO_DENSITY = 1000.0
ARMADILLO_K_MU = 5.0e5
ARMADILLO_K_LAMBDA = 5.0e6
ARMADILLO_K_DAMP = 1.0e-7
ARMADILLO_PARTICLE_RADIUS = 0.005

ROLLER_INNER_RADIUS = 0.36 * CRUSHER_SCENE_SCALE
ROLLER_OUTER_RADIUS = 0.40 * CRUSHER_SCENE_SCALE
ROLLER_LENGTH = 1.60 * CRUSHER_SCENE_SCALE
ROLLER_TEETH = 16
ROLLER_GAP = 0.08 * CRUSHER_SCENE_SCALE
ROLLER_SEPARATION = 2.0 * ROLLER_OUTER_RADIUS + ROLLER_GAP
# TABLE_ROTATION swaps the box's local X/Y extents in world space.
TABLE_WORLD_MAX_X = float(TABLE_POS[0]) + TABLE_HALF_EXTENTS[1]
CRUSHER_CENTER_X = TABLE_WORLD_MAX_X + 0.03 + 0.5 * ROLLER_LENGTH
CRUSHER_CENTER_Y = float(TABLE_POS[1])
CRUSHER_CENTER_Z = TABLE_TOP_Z - ROLLER_OUTER_RADIUS
ROLLER_ANGULAR_SPEED = 1.0

CRUSHER_CONTACT_KE = 1.0e6
CRUSHER_CONTACT_KD = 1.0e-7
CRUSHER_CONTACT_MU = 0.2
HAND_CONTACT_KE = 5.0e6
HAND_CONTACT_MU = 200.0
HAND_APPROACH_FRICTION = 200.0
HAND_GRASP_FRICTION = 200.0
HAND_GRASP_SOFT_CONTACT_MU = CRUSHER_CONTACT_MU
HAND_RELEASE_FRICTION = 0.0
SOFT_CONTACT_MARGIN = 0.01
RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 16384

MAX_FINGER_SPEED_DEG_S = 45.0
HAND_TRANSLATION_SPEED = 0.15
HAND_GRASP_TRANSLATION_SPEED = 0.08
HAND_LIFT_SPEED = 0.04
HAND_TRANSFER_SPEED = 0.04
MINIMUM_ROOT_PHASE_DURATION = 0.25

INITIAL_HOLD_DURATION = 1.0
PREGRASP_HOLD_DURATION = 0.50
GRASP_SETTLE_DURATION = 0.75
TRANSFER_HOLD_DURATION = 0.30
DROP_SETTLE_DURATION = 3.00
SCRIPTED_LIFT_HEIGHT = 0.13

CAMERA_POS = wp.vec3(0.95, -4.05, 1.72)
CAMERA_PITCH = -15.0
CAMERA_YAW = 128.0
CAMERA_FOV = 46.0

GEAR_COLOR = (0.50, 0.53, 0.58)
ARMADILLO_COLOR = (0.20, 0.78, 0.42)
TABLE_COLOR = (0.35, 0.42, 0.48)
GROUND_COLOR = (0.18, 0.20, 0.23)

HAND_JOINTS = (
    "RIGHT_HAND_THUMB1",
    "RIGHT_HAND_THUMB2",
    "RIGHT_HAND_INDEX",
    "RIGHT_INDEX_PIP",
    "RIGHT_HAND_MIDDLE",
    "RIGHT_MIDDLE_PIP",
    "RIGHT_HAND_RING",
    "RIGHT_RING_PIP",
    "RIGHT_HAND_PINKY",
    "RIGHT_PINKY_PIP",
)

OPEN_JOINTS = {
    "RIGHT_HAND_THUMB1": 0.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 0.0,
    "RIGHT_INDEX_PIP": 0.0,
    "RIGHT_HAND_MIDDLE": 0.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 0.0,
    "RIGHT_RING_PIP": 0.0,
    "RIGHT_HAND_PINKY": 0.0,
    "RIGHT_PINKY_PIP": 0.0,
}

# Embedded zero-offset finger closure.
GRASP_JOINTS = {
    "RIGHT_HAND_THUMB1": 22.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 32.0,
    "RIGHT_INDEX_PIP": 46.0,
    "RIGHT_HAND_MIDDLE": 31.0,
    "RIGHT_MIDDLE_PIP": 50.0,
    "RIGHT_HAND_RING": 32.0,
    "RIGHT_RING_PIP": 50.0,
    "RIGHT_HAND_PINKY": 33.0,
    "RIGHT_PINKY_PIP": 55.0,
}

APPROACH_HAND_ROOT = wp.transform(
    wp.vec3(-0.13015694916248322, -2.8182713985443115, 1.3436577320098877),
    wp.quat(0.002444127108901739, 0.9206019043922424, -0.3872908055782318, -0.049920178949832916),
)
GRASP_HAND_ROOT = wp.transform(
    # The 130 mm lift is applied later as a separate slow phase.
    wp.vec3(-0.13015694916248322, -2.8182713985443115, 1.3389617204666138),
    wp.quat(0.002444127108901739, 0.9206019043922424, -0.3872908055782318, -0.049920178949832916),
)
INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.13406670093536377, -2.9498887062072754, 1.6027559041976929),
    wp.quat(0.006665660534054041, 0.9210913777351379, -0.38724106550216675, -0.03988214209675789),
)

_CONTACT_MATERIAL_APPROACH = 0
_CONTACT_MATERIAL_GRASP = 1
_CONTACT_MATERIAL_RELEASE = 2


@dataclass(frozen=True)
class _Phase:
    """Describe one autonomous hand-motion phase."""

    name: str
    duration: float
    root_start: wp.transform
    root_end: wp.transform
    finger_start_degrees: dict[str, float]
    finger_end_degrees: dict[str, float]


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    """Interpolate generalized coordinates for one physics substep."""
    index = wp.tid()
    out[index] = q0[index] * (1.0 - alpha) + q1[index] * alpha


@wp.kernel
def _joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
    """Compute hand joint velocities from the displayed-frame target."""
    joint = wp.tid()
    q_begin, q_end = joint_q_start[joint], joint_q_start[joint + 1]
    qd_begin, qd_end = joint_qd_start[joint], joint_qd_start[joint + 1]
    if joint_type[joint] == newton.JointType.FREE:
        out[qd_begin + 0] = (q1[q_begin + 0] - q0[q_begin + 0]) * inv_dt
        out[qd_begin + 1] = (q1[q_begin + 1] - q0[q_begin + 1]) * inv_dt
        out[qd_begin + 2] = (q1[q_begin + 2] - q0[q_begin + 2]) * inv_dt
        q_delta = wp.normalize(
            wp.quat(q1[q_begin + 3], q1[q_begin + 4], q1[q_begin + 5], q1[q_begin + 6])
            * wp.quat_inverse(wp.quat(q0[q_begin + 3], q0[q_begin + 4], q0[q_begin + 5], q0[q_begin + 6]))
        )
        axis, angle = wp.quat_to_axis_angle(q_delta)
        out[qd_begin + 3] = axis[0] * angle * inv_dt
        out[qd_begin + 4] = axis[1] * angle * inv_dt
        out[qd_begin + 5] = axis[2] * angle * inv_dt
    else:
        for dof in range(qd_end - qd_begin):
            if q_begin + dof < q_end:
                out[qd_begin + dof] = (q1[q_begin + dof] - q0[q_begin + dof]) * inv_dt


@wp.kernel
def _prescribe_crusher_bodies(
    left_body: int,
    right_body: int,
    crusher_time: wp.array[float],
    angular_speed: float,
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Set the two externally prescribed counter-rotating rollers."""
    time = crusher_time[0]
    axis = wp.vec3(1.0, 0.0, 0.0)
    zero = wp.vec3(0.0, 0.0, 0.0)
    left_q = wp.transform(
        wp.vec3(CRUSHER_CENTER_X, CRUSHER_CENTER_Y - 0.5 * ROLLER_SEPARATION, CRUSHER_CENTER_Z),
        wp.quat_from_axis_angle(axis, -angular_speed * time),
    )
    right_q = wp.transform(
        wp.vec3(CRUSHER_CENTER_X, CRUSHER_CENTER_Y + 0.5 * ROLLER_SEPARATION, CRUSHER_CENTER_Z),
        wp.quat_from_axis_angle(axis, angular_speed * time),
    )
    left_qd = wp.spatial_vector(zero, axis * -angular_speed)
    right_qd = wp.spatial_vector(zero, axis * angular_speed)

    body_q_in[left_body] = left_q
    body_q_in[right_body] = right_q
    body_qd_in[left_body] = left_qd
    body_qd_in[right_body] = right_qd
    body_q_out[left_body] = left_q
    body_q_out[right_body] = right_q
    body_qd_out[left_body] = left_qd
    body_qd_out[right_body] = right_qd


@wp.kernel
def _advance_crusher_time(crusher_time: wp.array[float], dt: float):
    """Advance the device-resident roller clock."""
    crusher_time[0] = crusher_time[0] + dt


@wp.kernel
def _set_hand_shape_friction(shape_mu: wp.array[float], hand_shape_end: int, friction: float):
    """Set friction on one contiguous hand-shape range."""
    shape = wp.tid()
    if shape < hand_shape_end:
        shape_mu[shape] = friction


@wp.kernel
def _mark_interaction_contacts(
    contact_count: wp.array[int],
    contact_shape: wp.array[int],
    hand_shape_end: int,
    left_gear_shape: int,
    right_gear_shape: int,
    observed: wp.array[int],
):
    """Record whether hand and roller contacts occurred during a test."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape = contact_shape[contact]
    if shape >= 0 and shape < hand_shape_end:
        wp.atomic_max(observed, 0, 1)
    if shape == left_gear_shape or shape == right_gear_shape:
        wp.atomic_max(observed, 1, 1)


def _copy_transform(transform: wp.transform) -> wp.transform:
    """Copy one Warp transform value."""
    return wp.transform(
        wp.vec3(*wp.transform_get_translation(transform)),
        wp.quat(*wp.transform_get_rotation(transform)),
    )


def _translated_transform(transform: wp.transform, offset: wp.vec3) -> wp.transform:
    """Translate a hand pose without changing its orientation."""
    return wp.transform(
        wp.transform_get_translation(transform) + offset,
        wp.transform_get_rotation(transform),
    )


def _interpolate_transform(start: wp.transform, end: wp.transform, alpha: float) -> wp.transform:
    """Interpolate translation and the shortest normalized quaternion path."""
    start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
    end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
    start_rotation = np.asarray(wp.transform_get_rotation(start), dtype=np.float64)
    end_rotation = np.asarray(wp.transform_get_rotation(end), dtype=np.float64)
    if np.dot(start_rotation, end_rotation) < 0.0:
        end_rotation = -end_rotation
    rotation = start_rotation * (1.0 - alpha) + end_rotation * alpha
    rotation /= max(float(np.linalg.norm(rotation)), 1.0e-8)
    position = start_position * (1.0 - alpha) + end_position * alpha
    return wp.transform(wp.vec3(*position), wp.quat(*rotation))


def _transform_duration(
    start: wp.transform,
    end: wp.transform,
    translation_speed: float = HAND_TRANSLATION_SPEED,
) -> float:
    """Choose a hand-motion duration from its translation distance."""
    start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
    end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
    return max(
        MINIMUM_ROOT_PHASE_DURATION,
        float(np.linalg.norm(end_position - start_position)) / translation_speed,
    )


def _create_gear_cylinder_mesh() -> newton.Mesh:
    """Create the DexSim-style extruded 16-tooth roller along the X axis."""
    profile: list[tuple[float, float]] = []
    tooth_angle = 2.0 * math.pi / ROLLER_TEETH
    for tooth in range(ROLLER_TEETH):
        base = tooth * tooth_angle
        profile.extend(
            (
                (base, ROLLER_INNER_RADIUS),
                (base + tooth_angle * 0.08, ROLLER_INNER_RADIUS),
                (base + tooth_angle * 0.15, ROLLER_OUTER_RADIUS),
                (base + tooth_angle * 0.85, ROLLER_OUTER_RADIUS),
                (base + tooth_angle * 0.92, ROLLER_INNER_RADIUS),
            )
        )

    half_length = 0.5 * ROLLER_LENGTH
    vertices = []
    for x in (-half_length, half_length):
        vertices.extend((x, radius * math.cos(angle), radius * math.sin(angle)) for angle, radius in profile)

    profile_count = len(profile)
    triangles = []
    for index in range(profile_count):
        next_index = (index + 1) % profile_count
        triangles.append((index, next_index, profile_count + next_index))
        triangles.append((index, profile_count + next_index, profile_count + index))

    left_center = len(vertices)
    vertices.append((-half_length, 0.0, 0.0))
    right_center = len(vertices)
    vertices.append((half_length, 0.0, 0.0))
    for index in range(profile_count):
        next_index = (index + 1) % profile_count
        triangles.append((left_center, next_index, index))
        triangles.append((right_center, profile_count + index, profile_count + next_index))

    mesh = newton.Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(triangles, dtype=np.int32).reshape(-1),
        compute_inertia=False,
    )
    mesh.color = GEAR_COLOR
    mesh.roughness = 0.38
    mesh.metallic = 0.92
    return mesh


def _load_vtk_tet_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the supplied ASCII legacy-VTK tetrahedral mesh."""
    if not path.is_file():
        raise FileNotFoundError(f"Armadillo asset is missing: {path}")

    tokens = path.read_text(encoding="utf-8").split()
    try:
        points_at = tokens.index("POINTS")
        point_count = int(tokens[points_at + 1])
        points_start = points_at + 3
        points_end = points_start + 3 * point_count
        vertices = np.asarray(tokens[points_start:points_end], dtype=np.float32).reshape(-1, 3)

        cells_at = tokens.index("CELLS", points_end)
        cell_count = int(tokens[cells_at + 1])
        cursor = cells_at + 3
        tetrahedra: list[list[int]] = []
        for _ in range(cell_count):
            arity = int(tokens[cursor])
            cursor += 1
            cell = [int(index) for index in tokens[cursor : cursor + arity]]
            cursor += arity
            if arity == 4:
                tetrahedra.append(cell)
    except (ValueError, IndexError) as error:
        raise ValueError(f"Invalid legacy VTK tetrahedral mesh: {path}") from error

    if len(vertices) != point_count or not tetrahedra:
        raise ValueError(f"VTK mesh has incomplete points or no tetrahedra: {path}")
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


class Example:
    """Physically carry a soft Armadillo from a table to the crusher."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = int(args.substeps)
        self.vbd_iterations = int(args.vbd_iterations)
        if self.sim_substeps < 1:
            raise ValueError("--substeps must be at least 1")
        if self.vbd_iterations < 1:
            raise ValueError("--vbd-iterations must be at least 1")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.rotation_speed = float(args.rotation_speed)

        self._build_scene(Path(args.asset_path).expanduser())
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": self.vbd_iterations,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": ARMADILLO_PARTICLE_RADIUS,
                "particle_self_contact_margin": 0.0075,
                "particle_conservative_bound_relaxation": 0.85,
                "particle_collision_detection_interval": 5,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.02,
                "particle_enable_tile_solve": True,
                "particle_vertex_contact_buffer_size": 32,
                "particle_edge_contact_buffer_size": 64,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )
        self.soft_contact_materials = wp.array(
            np.asarray(
                (
                    (CRUSHER_CONTACT_KE, CRUSHER_CONTACT_KD, CRUSHER_CONTACT_MU),
                    (CRUSHER_CONTACT_KE, CRUSHER_CONTACT_KD, HAND_GRASP_SOFT_CONTACT_MU),
                    (CRUSHER_CONTACT_KE, CRUSHER_CONTACT_KD, CRUSHER_CONTACT_MU),
                ),
                dtype=np.float32,
            ),
            dtype=wp.vec3,
            device=self.device,
        )
        self.soft_contact_material_index = wp.full(
            1,
            _CONTACT_MATERIAL_APPROACH,
            dtype=wp.int32,
            device=self.device,
        )
        self.solver.vbd_solver.set_soft_contact_material_source(
            self.soft_contact_materials,
            self.soft_contact_material_index,
        )
        self.active_contact_material = _CONTACT_MATERIAL_APPROACH

        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.script_target_q = self.model.joint_q.numpy().copy()

        self.phases = self._build_phases()
        self.script_duration = sum(phase.duration for phase in self.phases)
        self.active_phase_index = -1
        self.active_phase_name = "initialization"
        self.grasp_armadillo_center: np.ndarray | None = None
        self.lifted_armadillo_center: np.ndarray | None = None
        self.release_armadillo_center: np.ndarray | None = None
        self._set_hand_target(self.phases[0].root_start, OPEN_JOINTS)
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)

        self.crusher_time = wp.zeros(1, dtype=float, device=self.device)
        self.interaction_observed = wp.zeros(2, dtype=int, device=self.device)
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = CAMERA_FOV

        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.graph = None

    def _build_scene(self, asset_path: Path):
        """Build the hand, support table, rollers, and tetrahedral Armadillo."""
        if not RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {RIGHT_HAND_URDF}")
        vertices, tetrahedra = _load_vtk_tet_mesh(asset_path)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_particle_radius = ARMADILLO_PARTICLE_RADIUS
        builder.default_shape_cfg.ke = HAND_CONTACT_KE
        builder.default_shape_cfg.kd = CRUSHER_CONTACT_KD
        builder.default_shape_cfg.mu = HAND_CONTACT_MU
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(RIGHT_HAND_URDF),
            xform=GRASP_HAND_ROOT,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.hand_articulations = tuple(range(articulation_start, builder.articulation_count))
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=3.0e5,
            kd=1.0e-4,
            mu=0.9,
            margin=0.0,
            is_visible=True,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POS, TABLE_ROTATION),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="armadillo_pick_table",
        )

        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CRUSHER_CONTACT_KE,
            kd=CRUSHER_CONTACT_KD,
            mu=CRUSHER_CONTACT_MU,
            restitution=0.0,
            margin=0.0,
        )
        gear_cfg.configure_sdf(force_sdf=True)
        gear_mesh = _create_gear_cylinder_mesh()
        gear_mass = 25.0
        i_axis = 0.5 * gear_mass * ROLLER_OUTER_RADIUS**2
        i_transverse = gear_mass * (3.0 * ROLLER_OUTER_RADIUS**2 + ROLLER_LENGTH**2) / 12.0
        gear_inertia = wp.mat33(i_axis, 0.0, 0.0, 0.0, i_transverse, 0.0, 0.0, 0.0, i_transverse)
        self.left_gear_body = builder.add_link(
            xform=wp.transform(
                wp.vec3(CRUSHER_CENTER_X, CRUSHER_CENTER_Y - 0.5 * ROLLER_SEPARATION, CRUSHER_CENTER_Z),
                wp.quat_identity(),
            ),
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="left_crusher_gear",
        )
        self.right_gear_body = builder.add_link(
            xform=wp.transform(
                wp.vec3(CRUSHER_CENTER_X, CRUSHER_CENTER_Y + 0.5 * ROLLER_SEPARATION, CRUSHER_CENTER_Z),
                wp.quat_identity(),
            ),
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="right_crusher_gear",
        )
        self.left_gear_shape = builder.add_shape_mesh(
            self.left_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="left_crusher_gear_mesh",
        )
        self.right_gear_shape = builder.add_shape_mesh(
            self.right_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="right_crusher_gear_mesh",
        )
        builder.add_shape_collision_filter_pair(self.left_gear_shape, self.right_gear_shape)

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CRUSHER_CONTACT_KE,
            kd=CRUSHER_CONTACT_KD,
            mu=CRUSHER_CONTACT_MU,
        )
        builder.add_ground_plane(
            height=GROUND_HEIGHT,
            cfg=ground_cfg,
            color=GROUND_COLOR,
            label="crusher_ground",
        )

        bounds_min = np.min(vertices, axis=0)
        bounds_max = np.max(vertices, axis=0)
        local_center_x = 0.5 * float(bounds_min[0] + bounds_max[0])
        local_center_y = 0.5 * float(bounds_min[1] + bounds_max[1])
        armadillo_position = wp.vec3(
            ARMADILLO_CENTER_X + ARMADILLO_SCALE * local_center_y,
            ARMADILLO_CENTER_Y - ARMADILLO_SCALE * local_center_x,
            TABLE_TOP_Z + ARMADILLO_INITIAL_CLEARANCE - ARMADILLO_SCALE * float(bounds_min[2]),
        )
        armadillo_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi)
        self.armadillo_particle_start = builder.particle_count
        builder.add_soft_mesh(
            pos=armadillo_position,
            rot=armadillo_rotation,
            scale=ARMADILLO_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=tetrahedra.reshape(-1).tolist(),
            density=ARMADILLO_DENSITY,
            k_mu=ARMADILLO_K_MU,
            k_lambda=ARMADILLO_K_LAMBDA,
            k_damp=ARMADILLO_K_DAMP,
            particle_radius=ARMADILLO_PARTICLE_RADIUS,
            validate_mesh=True,
            label="graspable_armadillo",
        )
        self.armadillo_particle_end = builder.particle_count

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = CRUSHER_CONTACT_KE
        self.model.soft_contact_kd = CRUSHER_CONTACT_KD
        self.model.soft_contact_mu = CRUSHER_CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[: self.hand_shape_end] = HAND_APPROACH_FRICTION
        self.model.shape_material_mu.assign(shape_mu)
        self.armadillo_surface_indices = self.model.tri_indices.flatten()

    def _root_joint_index(self) -> int:
        """Return the imported floating-hand root joint."""
        joint_types = self.model.joint_type.numpy()
        parents = self.model.joint_parent.numpy()
        for index, (joint_type, parent) in enumerate(zip(joint_types, parents, strict=True)):
            if int(joint_type) == int(newton.JointType.FREE) and int(parent) == -1:
                return index
        raise RuntimeError("Right-hand URDF must import with a free root joint")

    def _hand_joint_indices(self) -> dict[str, int]:
        """Map embedded finger targets to generalized-coordinate indices."""
        labels = self.model.joint_label
        starts = self.model.joint_q_start.numpy()
        indices = {}
        for name in HAND_JOINTS:
            joint = next(index for index, label in enumerate(labels) if label.endswith("/" + name))
            indices[name] = int(starts[joint])
        return indices

    def _set_hand_target(self, root: wp.transform, joints_degrees: dict[str, float]):
        """Update the next root and finger targets without moving particles."""
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        self.script_target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, q_index in self.hand_joint_indices.items():
            self.script_target_q[q_index] = np.radians(joints_degrees[name])
        self.manual_target_q.assign(self.script_target_q)

    def _set_initial_hand_pose(self):
        """Initialize both state buffers at the scripted open-hand pose."""
        self.state_0.joint_q.assign(self.manual_target_q)
        self.state_1.joint_q.assign(self.manual_target_q)
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        newton.eval_fk(
            self.model,
            self.state_1.joint_q,
            self.state_1.joint_qd,
            self.state_1,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def _build_phases(self) -> tuple[_Phase, ...]:
        """Build settle, grasp, carry, release, and drop phases."""
        approach_root = _copy_transform(APPROACH_HAND_ROOT)
        grasp_root = _copy_transform(GRASP_HAND_ROOT)
        initial_root = _copy_transform(INITIAL_HAND_ROOT)
        lifted_root = _translated_transform(grasp_root, wp.vec3(0.0, 0.0, SCRIPTED_LIFT_HEIGHT))
        transfer_offset = wp.vec3(
            CRUSHER_CENTER_X - ARMADILLO_CENTER_X,
            CRUSHER_CENTER_Y - ARMADILLO_CENTER_Y,
            0.0,
        )
        transfer_root = _translated_transform(lifted_root, transfer_offset)
        maximum_finger_delta = max(abs(GRASP_JOINTS[name] - OPEN_JOINTS[name]) for name in HAND_JOINTS)
        finger_duration = maximum_finger_delta / MAX_FINGER_SPEED_DEG_S
        grasp_duration = max(
            finger_duration,
            _transform_duration(approach_root, grasp_root, HAND_GRASP_TRANSLATION_SPEED),
        )
        return (
            _Phase("initial_hold", INITIAL_HOLD_DURATION, initial_root, initial_root, OPEN_JOINTS, OPEN_JOINTS),
            _Phase(
                "approach",
                _transform_duration(initial_root, approach_root),
                initial_root,
                approach_root,
                OPEN_JOINTS,
                OPEN_JOINTS,
            ),
            _Phase(
                "pregrasp_hold",
                PREGRASP_HOLD_DURATION,
                approach_root,
                approach_root,
                OPEN_JOINTS,
                OPEN_JOINTS,
            ),
            _Phase("close", grasp_duration, approach_root, grasp_root, OPEN_JOINTS, GRASP_JOINTS),
            _Phase(
                "grasp_settle",
                GRASP_SETTLE_DURATION,
                grasp_root,
                grasp_root,
                GRASP_JOINTS,
                GRASP_JOINTS,
            ),
            _Phase(
                "lift",
                _transform_duration(grasp_root, lifted_root, HAND_LIFT_SPEED),
                grasp_root,
                lifted_root,
                GRASP_JOINTS,
                GRASP_JOINTS,
            ),
            _Phase(
                "transfer",
                _transform_duration(lifted_root, transfer_root, HAND_TRANSFER_SPEED),
                lifted_root,
                transfer_root,
                GRASP_JOINTS,
                GRASP_JOINTS,
            ),
            _Phase(
                "transfer_hold",
                TRANSFER_HOLD_DURATION,
                transfer_root,
                transfer_root,
                GRASP_JOINTS,
                GRASP_JOINTS,
            ),
            _Phase("release", finger_duration, transfer_root, transfer_root, GRASP_JOINTS, OPEN_JOINTS),
            _Phase("drop_settle", DROP_SETTLE_DURATION, transfer_root, transfer_root, OPEN_JOINTS, OPEN_JOINTS),
        )

    def _sample(self, time_s: float) -> tuple[wp.transform, dict[str, float], int]:
        """Sample the scripted hand root and finger target."""
        for phase_index, phase in enumerate(self.phases):
            if time_s <= phase.duration:
                alpha = float(np.clip(time_s / phase.duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                finger_target = {
                    name: phase.finger_start_degrees[name] * (1.0 - alpha) + phase.finger_end_degrees[name] * alpha
                    for name in HAND_JOINTS
                }
                return (
                    _interpolate_transform(phase.root_start, phase.root_end, alpha),
                    finger_target,
                    phase_index,
                )
            time_s -= phase.duration
        final_index = len(self.phases) - 1
        final_phase = self.phases[final_index]
        return _copy_transform(final_phase.root_end), final_phase.finger_end_degrees, final_index

    def _armadillo_center(self) -> np.ndarray:
        """Return the current soft-body center position [m]."""
        positions = self.state_0.particle_q.numpy()[self.armadillo_particle_start : self.armadillo_particle_end]
        return np.mean(positions, axis=0)

    def _set_contact_material(self, material_index: int, shape_friction: float, soft_friction: float):
        """Switch hand friction while preserving CUDA graph compatibility."""
        if material_index == self.active_contact_material:
            return
        wp.launch(
            _set_hand_shape_friction,
            dim=self.hand_shape_end,
            inputs=[self.model.shape_material_mu, self.hand_shape_end, shape_friction],
            device=self.device,
        )
        self.model.soft_contact_mu = soft_friction
        self.soft_contact_material_index.fill_(material_index)
        self.active_contact_material = material_index

    def _enter_phase(self, phase_index: int):
        """Apply one-time diagnostics and material changes at phase boundaries."""
        if phase_index == self.active_phase_index:
            return
        self.active_phase_index = phase_index
        phase = self.phases[phase_index]
        self.active_phase_name = phase.name
        # Increase friction as closure begins so physical contact establishes
        # force closure before the lift starts.
        if phase.name == "close":
            self._set_contact_material(
                _CONTACT_MATERIAL_GRASP,
                HAND_GRASP_FRICTION,
                HAND_GRASP_SOFT_CONTACT_MU,
            )
        if phase.name == "lift":
            self.grasp_armadillo_center = self._armadillo_center()
        if phase.name == "transfer":
            self.lifted_armadillo_center = self._armadillo_center()
        if phase.name == "release":
            self.release_armadillo_center = self._armadillo_center()
            self._set_contact_material(
                _CONTACT_MATERIAL_RELEASE,
                HAND_RELEASE_FRICTION,
                CRUSHER_CONTACT_MU,
            )

    def _prepare_physics_frame(self):
        """Prepare graph-compatible scripted hand targets."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)

    def _simulate_substeps(self):
        """Advance the fixed hand, roller, and VBD substep sequence."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity,
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
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            wp.launch(
                _prescribe_crusher_bodies,
                dim=1,
                inputs=[
                    self.left_gear_body,
                    self.right_gear_body,
                    self.crusher_time,
                    self.rotation_speed,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            contacts = self.solver.contacts
            wp.launch(
                _mark_interaction_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_shape,
                    self.hand_shape_end,
                    self.left_gear_shape,
                    self.right_gear_shape,
                    self.interaction_observed,
                ],
                device=self.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                _advance_crusher_time,
                dim=1,
                inputs=[self.crusher_time, self.sim_dt],
                device=self.device,
            )

    def _advance_physics_frame(self):
        """Prepare targets and advance one graph-capturable frame."""
        self._prepare_physics_frame()
        self._simulate_substeps()

    def _capture_simulation_graph(self):
        """Capture one frame while restoring capture-time physical state."""
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        crusher_time_backup = wp.clone(self.crusher_time)
        interaction_backup = wp.clone(self.interaction_observed)

        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._advance_physics_frame()

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        wp.copy(self.crusher_time, crusher_time_backup)
        wp.copy(self.interaction_observed, interaction_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}")

    def step(self):
        """Advance the scripted physical pick, carry, and release sequence."""
        root, joints, phase_index = self._sample(self.sim_time)
        self._enter_phase(phase_index)
        self._set_hand_target(root, joints)
        if self.graph is None:
            self._advance_physics_frame()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        """Render the hand, table, rollers, and green Armadillo surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/armadillo_gear_crusher/armadillo",
            self.state_0.particle_q,
            self.armadillo_surface_indices,
            backface_culling=True,
            color=ARMADILLO_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite state and the completed physical handoff to the gears."""
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Armadillo particle state contains non-finite values")
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise ValueError("Right-hand joint state contains non-finite values")
        if self.use_graph and self.frame_index > 1 and self.graph is None:
            raise ValueError("CUDA graph capture was requested but not created")
        if self.sim_time + self.frame_dt < self.script_duration:
            return

        observed = self.interaction_observed.numpy()
        if int(observed[0]) == 0:
            raise ValueError("The right hand never contacted the Armadillo")
        if int(observed[1]) == 0:
            raise ValueError("The released Armadillo never contacted the crusher gears")
        if (
            self.grasp_armadillo_center is None
            or self.lifted_armadillo_center is None
            or self.release_armadillo_center is None
        ):
            raise ValueError("The scripted lift or release phase did not complete")
        if self.lifted_armadillo_center[2] < self.grasp_armadillo_center[2] + 0.05:
            raise ValueError("The hand did not lift the Armadillo from the table")
        if abs(float(self.release_armadillo_center[0]) - CRUSHER_CENTER_X) > 0.5 * ROLLER_LENGTH:
            raise ValueError("The hand did not carry the Armadillo over the right-side gears")
        final_center = self._armadillo_center()
        if final_center[2] > self.release_armadillo_center[2] - 0.03:
            raise ValueError("The Armadillo did not fall after the hand opened")

    @staticmethod
    def create_parser():
        """Create command-line arguments for the autonomous carrying demo."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES, paused=False)
        parser.add_argument(
            "--asset-path",
            type=Path,
            default=ARMADILLO_ASSET,
            help="Armadillo legacy-VTK asset path.",
        )
        parser.add_argument(
            "--rotation-speed",
            type=float,
            default=ROLLER_ANGULAR_SPEED,
            help="Magnitude of each roller's counter-rotation speed [rad/s].",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=SIM_SUBSTEPS,
            help="Physics substeps per displayed frame.",
        )
        parser.add_argument(
            "--vbd-iterations",
            type=int,
            default=VBD_ITERATIONS,
            help="VBD iterations per physics substep.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics frame as one CUDA graph.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
