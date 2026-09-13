# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scoop rigid popcorn into a deformable paper cup with the full W1.

Experimental contact-driven scene: the robot follows IK targets, while the
cup, scoop and popcorn remain dynamic. No attachments or particle animation
are used. The cup is an experimental elastoplastic shell, not calibrated paper.
Run ``uv run --extra examples -m newton.examples mjvbd_v2_popcorn``.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2.support.paper_shell_material import update_paper_hinges
from newton.examples.mjvbdv2.support.table_clearance import TableClearanceGuard
from newton.solvers import SolverMJVBDV2

ASSETS = Path(__file__).resolve().parents[3] / "assets"
POPCORN_PROPS = Path(__file__).resolve().parent / "assets/popcorn/props.json"
ROBOT = ASSETS / "DexforceW1V021/DexforceW1V021.urdf"
TABLE = 0.84
TABLE_FRONT = 0.57
WORKSPACE_X = 0.06
CUP_HEIGHT = 0.14
CUP_RADIUS_SCALE = 0.8
PAPER_MEMBRANE_STIFFNESS = 50000.0
PAPER_BENDING_STIFFNESS = 1.0
PAPER_YIELD_ANGLE = 0.015  # Hinge rotation [rad], experimental paper crease onset.
CUP_MATERIAL_COLORS = ((0.88, 0.82, 0.68), (0.48, 0.33, 0.23), (0.92, 0.86, 0.73))
CUP_LIFT = 0.11
CUP = np.array((0.56 + WORKSPACE_X, 0.27, TABLE + CUP_HEIGHT / 2 + 0.001), dtype=np.float32)
SCOOP_FLOOR_Z = -0.048
BOWL_REAR = 0.135
BOWL_FRONT = 0.280
HANDLE_RADIUS = 0.014
CARRY_PITCH = -0.20
SCOOP_ENTRY_PITCH = math.radians(5)
POUR_PITCH = math.radians(40)
POUR_YAW = math.pi / 3
RECEIVING_CUP = np.array((0.70, 0.0, CUP[2] + CUP_LIFT - 0.02))
CUP_GRIP_FORCE = np.array((4.0, 1.0, 1.0, 1.0, 1.0))  # Thumb, then four fingers [N].
# Extra distal-finger travel follows the smaller radius of the indented wall.
# Force feedback, rate limits and the unchanged URDF limits still apply.
CUP_GRIP_MAX_OFFSET = np.radians((30, 12, 12, 16, 20))
PITCH_BEGIN, PITCH_END = 10.5, 11.25
# Pronate around the round shaft: palm above the handle, fingers curling below.
RIGHT_HAND_ROLL = math.radians(-121)
TOOL_FEEDBACK_MAX_ANGLE = math.radians(10)
RIGHT_GRIP_YAW = math.pi / 6
ROBOT_X = 0.25
TOOL_PLAN_ORIGIN = np.array((0.49 + WORKSPACE_X, -0.30, 1.03), dtype=np.float32)
MACHINE_PLAN = np.array((0.80 + WORKSPACE_X, -0.34, TABLE + 0.06), dtype=np.float32)
# Leave room between the lowered utensil and the torso for the elbow-down
# approach. The machine and tool path move together with this station frame.
HANDLE = np.array((0.70, -0.28, 1.03), dtype=np.float32)
SCOOP_YAW = 0.0
STATION_ROTATION = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), SCOOP_YAW)
STATION_FRAME = wp.transform(
    wp.vec3(*HANDLE) - wp.quat_rotate(STATION_ROTATION, wp.vec3(*TOOL_PLAN_ORIGIN)), STATION_ROTATION
)


def station_point(point):
    """Map the machine's working coordinates to the robot's right-hand bay."""
    return np.asarray(wp.transform_point(STATION_FRAME, wp.vec3(*point)))


MACHINE = station_point(MACHINE_PLAN)
TCP = wp.vec3(-0.066, 0, 0)
APPROACH = (np.array((-0.04, 0.0, 0.0)), np.zeros(3))
TRAY_CONTACT_KE = 4.0e4  # ALM penalty ceiling [N/m].
GRAIN_CONTACT_KE = 1.0e4
RIGID_CONTACT_BETA = 5.0e5  # Penalty growth per penetration [N/m^2].
GRASP_CENTERS = ((0.00290154, 0.10364690, 0.05120512), (-0.0005498, 0.0825728, 0.0354889))
LEFT_GRIP_TILT = math.radians(-9.89896)
LEFT_GRIP_TILT_Z = math.radians(-31.90938)
LEFT_THUMB_APPROACH = 0.0
# Index/middle/ring/pinky MCP,PIP, followed by the two thumb coordinates [deg].
GRASP_ANGLES = (
    (18.64723, 12.15433, 48.02730, 3.36554, 64.48925, 1.55911, 70.86599, 2.60215, 7.05563, 45.40551),
    (71.9066, 46.0249, 71.1363, 61.4614, 75.0, 75.8950, 74.9999, 82.8914, 20.5, 89.9997),
)


def grasp_joint_angle(side, name):
    """Map the unchanged W1 joint names to the fitted exterior grasp [radians]."""
    if name.endswith("THUMB1"):
        index = 8
    elif name.endswith("THUMB2"):
        index = 9
    else:
        digit = next(j for j, token in enumerate(("INDEX", "MIDDLE", "RING", "PINKY")) if token in name)
        index = 2 * digit + int(name.endswith("PIP"))
    return math.radians(GRASP_ANGLES[side][index])


def _right_hand_rotation():
    """Place the unchanged hand in an overhand grip around the scoop shaft."""
    return (
        STATION_ROTATION
        * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), RIGHT_HAND_ROLL)
        * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -RIGHT_GRIP_YAW)
    )


def _scoop_floor_height_offset(vertices, pitch, *, observed_rotation=None):
    """Raise the wrist target to preserve the real blade's minimum floor clearance."""
    points = np.asarray(vertices)
    rotated_z = -math.sin(pitch) * points[:, 0] + math.cos(pitch) * points[:, 2]
    if observed_rotation is not None:
        rotation = np.asarray(wp.quat_to_matrix(wp.quat(*observed_rotation))).reshape(3, 3)
        rotated_z = points @ rotation[2]
    return float(points[:, 2].min() - rotated_z.min())


def _popcorn_spawn_position(index):
    """Stack extra grains behind the initially held scoop, not inside its pan."""
    if index < 128:
        column, row, layer = index % 8, (index // 8) % 8, index // 64
    else:
        extra = index - 128
        column, row, layer = 3 + extra % 5, (extra // 5) % 8, 2 + extra // 40
    return station_point(
        (
            MACHINE_PLAN[0] - 0.07 + column * 0.020,
            MACHINE_PLAN[1] - 0.0805 + row * 0.023,
            MACHINE_PLAN[2] + 0.045 + layer * 0.022,
        )
    )


def _bounds_overlap_table(lower, upper, *, clearance=0.001):
    """Conservatively reject arm geometry bounds touching the solid tabletop."""
    table_lower = np.array((TABLE_FRONT, -0.76, TABLE - 0.05)) - clearance
    table_upper = np.array((1.50 + WORKSPACE_X, 0.76, TABLE)) + clearance
    return bool(np.all(upper >= table_lower) and np.all(lower <= table_upper))


def _mesh_intersects_table(points, indices, *, clearance=0.001):
    """Test mesh triangles against the expanded tabletop using separating axes."""
    lower = np.array((TABLE_FRONT, -0.76, TABLE - 0.05)) - clearance
    upper = np.array((1.50 + WORKSPACE_X, 0.76, TABLE)) + clearance
    triangles = points[indices]
    candidates = np.all(triangles.max(axis=1) >= lower, axis=1)
    candidates &= np.all(triangles.min(axis=1) <= upper, axis=1)
    triangles = triangles[candidates] - (lower + upper) / 2
    if not len(triangles):
        return False
    half = (upper - lower) / 2
    edges = np.roll(triangles, -1, axis=1) - triangles
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    for edge in range(3):
        for box_axis in np.eye(3):
            axes.append(np.cross(edges[:, edge], box_axis))
    overlaps = np.ones(len(triangles), dtype=bool)
    for axis in axes:
        projections = np.einsum("nvi,ni->nv", triangles, axis)
        radius = np.abs(axis) @ half
        overlaps &= (projections.min(axis=1) <= radius) & (projections.max(axis=1) >= -radius)
    return bool(np.any(overlaps))


def _grasp_force_targets(forces, ready):
    """Establish light opposing contact before applying the full grip preload."""
    target = CUP_GRIP_FORCE.copy()
    if not ready:
        target[0] = min(target[0], 0.2 + 4.0 * max(0.0, float(np.min(forces[1:]))))
    return target


def _lift_speed_factor(forces):
    """Slow the robot's lift when a digit loses its contact preload."""
    if not np.isfinite(forces).all():
        return 0.0
    weakest = float(np.min(forces / CUP_GRIP_FORCE))
    return float(np.clip((weakest - 0.15) / 0.35, 0.0, 1.0))


def _all_grasp_digits_in_contact(forces):
    """Require positive measured contact on each of the five physical digits."""
    return bool(np.isfinite(forces).all() and np.all(np.asarray(forces) >= 0.02))


def _grasp_offset_update(offset, forces, ready):
    """Integrate bounded pressure feedback for the robot's five grasp joints."""
    target = _grasp_force_targets(forces, ready)
    # Hold the joint pose within a load band instead of opening the grip as
    # soon as carried load raises contact pressure above the closing target.
    # Once lifted, normal load from falling grains must not command an opening
    # of the supporting fingers. Retain a finite overload-release threshold.
    upper = np.maximum(2 * target, 4.0) if ready else 2 * target
    if ready and np.any(forces < 0.5 * target):
        # A weak digit transfers load onto its neighbors. Opening those
        # neighbors independently creates positive feedback and rolls the cup
        # out. Recover opposition first, while still releasing hard overloads.
        upper = np.maximum(4 * target, 8.0)
    error = np.where(forces < target, target - forces, np.minimum(0.0, upper - forces))
    velocity = np.clip(0.05 * error, -0.06, 0.06)
    # The opposed thumb needs more closing travel than the wrapped fingers.
    # Original URDF limits are additionally enforced on the final commands.
    return np.clip(offset + velocity / 60.0, math.radians(-8), CUP_GRIP_MAX_OFFSET)


def _grasp_finger_commands(mcp, pip, offset):
    """Use the distal joint's closing travel when the knuckle reaches its limit."""
    requested = mcp + offset
    overflow = max(0.0, requested - math.radians(75))
    return np.clip(requested, 0, math.radians(75)), np.clip(pip + 2 * overflow, 0, math.radians(120))


def popcorn_mesh():
    """Build one irregular convex kernel with matching render/collision geometry."""
    sphere = newton.Mesh.create_sphere(radius=1, num_latitudes=6, num_longitudes=10)
    cloud = np.concatenate(
        [
            np.asarray(sphere.vertices) * radius + np.asarray(center)
            for center, radius in (
                ((0.002, 0.001, 0), 0.0065),
                ((-0.004, -0.001, 0.003), 0.005),
                ((-0.001, 0.004, 0.004), 0.005),
                ((0.003, -0.004, -0.003), 0.0045),
            )
        ]
    )
    return newton.Mesh(cloud, indices=[]).compute_convex_hull()


def cup_mesh(segments=40, rings=12):
    """Build a tapered paper shell with a rolled lip and a subdivided bottom."""
    points, faces = [], []
    for j in range(rings + 1):
        z = CUP_HEIGHT * j / rings
        radius = CUP_RADIUS_SCALE * (0.030 + 0.012 * j / rings)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), z - CUP_HEIGHT / 2))
    for j in range(rings):
        for i in range(segments):
            a, b = j * segments + i, j * segments + (i + 1) % segments
            faces.extend(((a, b, b + segments), (a, b + segments, a + segments)))
    previous = 0
    for base_radius in (0.0225, 0.015, 0.0075):
        radius = base_radius * CUP_RADIUS_SCALE
        current = len(points)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), -CUP_HEIGHT / 2))
        for i in range(segments):
            a, b = previous + i, previous + (i + 1) % segments
            c, d = current + i, current + (i + 1) % segments
            faces.extend(((a, d, b), (a, c, d)))
        previous = current
    center = len(points)
    points.append((0, 0, -CUP_HEIGHT / 2))
    for i in range(segments):
        faces.append((previous + i, center, previous + (i + 1) % segments))
    # Fold the actual paper sheet inward around a 1.4 mm lip radius. Leave
    # the inner edge open: closing it onto the wall makes a non-manifold seam.
    # Membrane/bending forces of this visible roll reinforce the opening;
    # there is no rigid rim collider or kinematic constraint.
    previous = rings * segments
    for fold in range(1, 6):
        theta = fold * math.pi / 3
        radius = 0.042 * CUP_RADIUS_SCALE - 0.0014 + 0.0014 * math.cos(theta)
        z = CUP_HEIGHT / 2 + 0.0014 * math.sin(theta)
        current = len(points)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))
        for i in range(segments):
            a, b = previous + i, previous + (i + 1) % segments
            c, d = current + i, current + (i + 1) % segments
            faces.extend(((a, b, d), (a, d, c)))
        previous = current
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def scoop_bowl_panels(sections=12):
    """Make a rolled sheet-metal trough from closed, matching convex panels.

    The 1.2 mm aluminum sheet curves continuously across the 108 mm opening.
    Each panel is both visible and collidable; there is no invisible box bowl.
    """
    angles = np.linspace(-1.30, 1.30, sections + 1)
    radius, thickness = 0.056, 0.0012
    profiles = []
    for angle in angles:
        profiles.append(
            [(r * math.sin(angle), SCOOP_FLOOR_Z + radius - r * math.cos(angle)) for r in (radius, radius + thickness)]
        )
    panels = []
    for a, b in pairwise(profiles):
        vertices = [(x, y, z) for x in (BOWL_REAR, BOWL_FRONT) for y, z in (*a, *b)]
        panels.append(newton.Mesh(np.asarray(vertices, dtype=np.float32), indices=[]).compute_convex_hull())
    # The rear closes the trough; the front is its thin pouring/scooping lip.
    rear = [(x, y, z) for x in (BOWL_REAR - thickness, BOWL_REAR) for profile in profiles for y, z in profile]
    panels.append(newton.Mesh(np.asarray(rear, dtype=np.float32), indices=[]).compute_convex_hull())
    return panels


def _inside_scoop(points):
    """Classify grain centers against the actual piecewise-planar inner trough."""
    angles = np.linspace(-1.30, 1.30, 13)
    wall_y = 0.056 * np.sin(angles)
    wall_z = SCOOP_FLOOR_Z + 0.056 * (1 - np.cos(angles))
    bottom = np.interp(points[:, 1], wall_y, wall_z)
    return (
        (points[:, 0] > BOWL_REAR)
        & (points[:, 0] < BOWL_FRONT)
        & (np.abs(points[:, 1]) < wall_y[-1])
        & (points[:, 2] > bottom)
        & (points[:, 2] < wall_z[-1])
    )


def _cup_rim_indices(vertices, faces):
    """Find the sheet's open boundary, including the inside edge of a rolled lip."""
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1])
    return boundary[np.argsort(np.arctan2(vertices[boundary, 1], vertices[boundary, 0]))]


def _inside_cup(points, vertices, faces, rim_indices):
    """Count grain centers using the deformed shell, capped only for measurement.

    The virtual rim cap is never added to the collision or render model. Ray
    parity replaces the inaccurate rigid best-fit cylinder containment metric.
    Points exactly on the surface are not suitable acceptance samples.
    """
    rim = vertices[rim_indices]
    cap = np.stack((rim, np.roll(rim, -1, axis=0), np.broadcast_to(rim.mean(axis=0), rim.shape)), axis=1)
    triangles = np.concatenate((vertices[faces], cap)).astype(np.float64)
    direction = np.array((0.8713, 0.3371, 0.3579))
    edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    p = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    valid = np.abs(det) > 1e-12
    inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = np.asarray(points)[:, None, :] - triangles[None, :, 0]
    u = np.einsum("nfi,fi->nf", s, p) * inverse
    q = np.cross(s, edge1)
    v = np.einsum("nfi,i->nf", q, direction) * inverse
    t = np.einsum("nfi,fi->nf", q, edge2) * inverse
    hits = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    return np.count_nonzero(hits, axis=1) % 2 == 1


@wp.kernel
def prescribe(
    begin: wp.array[float],
    end: wp.array[float],
    starts: wp.array[int],
    dofs: wp.array[int],
    fraction: float,
    inverse_dt: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    j = wp.tid()
    if starts[j + 1] > starts[j]:
        i = starts[j]
        q[i] = wp.lerp(begin[i], end[i], fraction)
        qd[dofs[j]] = (end[i] - begin[i]) * inverse_dt


@wp.kernel
def lock_joints(indices: wp.array[int], values: wp.array[float], q: wp.array2d[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def measure_grasp_pressure(
    count: wp.array[int],
    shapes: wp.array[int],
    corners: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    body_points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    stiffness: wp.array[float],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    digit: wp.array[int],
    body_q: wp.array[wp.transform],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    force: wp.array[float],
):
    """Read the solver's normal penalty law; do not modify physical state."""
    i = wp.tid()
    if i >= count[0]:
        return
    shape = shapes[i]
    group = digit[shape]
    if group < 0:
        return
    ids, weights = corners[i], barycentric[i]
    point, radius = wp.vec3(), float(0.0)
    for j in range(3):
        if ids[j] >= 0:
            point += weights[j] * particle_q[ids[j]]
            radius = wp.max(radius, particle_radius[ids[j]])
    surface = wp.transform_point(body_q[shape_body[shape]], body_points[i])
    penetration = wp.max(0.0, radius + shape_margin[shape] - wp.dot(normals[i], point - surface))
    wp.atomic_add(force, group, stiffness[i] * penetration)


@wp.kernel
def _copy_elbow_reference(
    poses: wp.array[wp.transform],
    left: int,
    right: int,
    left_target: wp.array[wp.vec3],
    right_target: wp.array[wp.vec3],
):
    left_target[0] = wp.transform_get_translation(poses[left])
    right_target[0] = wp.transform_get_translation(poses[right])


class _ContinuousIK(ik.IKSolver):
    """Regularize elbow displacement in this single-robot trajectory controller."""

    def __init__(self, model, *, elbows, objectives, **kwargs):
        self._reference_model = model
        self._reference_state = model.state()
        self._elbows = elbows
        self._elbow_targets = [wp.zeros(1, dtype=wp.vec3, device=model.device) for _ in elbows]
        continuity = [
            ik.IKObjectivePosition(body, wp.vec3(), target, weight=0.15)
            for body, target in zip(elbows, self._elbow_targets, strict=True)
        ]
        super().__init__(model, objectives=[*objectives, *continuity], **kwargs)

    def step(self, joint_q_in, joint_q_out, iterations=50, step_size=1.0):
        # Freeze the reference for each IK block, not each Newton iteration.
        # This penalizes redundant arm reconfiguration without filtering the
        # commanded wrist pose or altering any dynamic object's state.
        model = self._reference_model
        newton.eval_fk(model, joint_q_in[0], model.joint_qd, self._reference_state)
        wp.launch(
            _copy_elbow_reference,
            1,
            [self._reference_state.body_q, *self._elbows, *self._elbow_targets],
            device=model.device,
        )
        super().step(joint_q_in, joint_q_out, iterations=iterations, step_size=step_size)


class _WristTargetError(RuntimeError):
    """Identify a failed task-space target independently of backend errors."""

    def __init__(self, message, *, side):
        super().__init__(message)
        self.side = side


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 1 or args.substeps % 2 or args.popcorn_count < 1 or args.popcorn_count > 192:
            raise ValueError("Require positive even substeps and 1..192 popcorn bodies")
        self.viewer, self.args = viewer, args
        if hasattr(viewer, "cache_static_appearance"):
            viewer.cache_static_appearance = True
        self.sim_time, self.dt = 0.0, 1 / (60 * args.substeps)
        self.grasp_wait = 0.0
        self.lift_wait = 0.0
        self.grasp_ready_time = 0.0
        self.grasp_ready = False
        self.grasp_contact_samples = 0
        self.five_finger_contact_samples = 0
        self.phase = "approach"
        self.max_lift = self.max_scoop_lift = self.max_cup_deformation = 0.0
        self.max_inside = 0
        self.retained_samples = 0
        self.last_retention_time = -1.0
        self.max_scoop_count = 0
        self.loaded_lift_count = 0
        self.payload_eligible = None
        self.lifted_payload = np.zeros(args.popcorn_count, dtype=bool)
        self.graph = None
        self.tool_grasp = None
        self.tool_orientation_correction = wp.quat_identity()
        self.cup_grasp = None
        self.cup_pickup_frame = None
        print("Preparing W1 hand collision geometry and the popcorn scene...", flush=True)
        self._build()
        print("Planning the bimanual grasp and delivery IK trajectory...", flush=True)
        self._ik()
        self._gpu_table_guard = TableClearanceGuard(
            self,
            (TABLE_FRONT - 0.001, -0.761, TABLE - 0.051),
            (1.501 + WORKSPACE_X, 0.761, TABLE + 0.001),
            _mesh_intersects_table,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self._calibrate_tool_grasp(require_lift=False)
        self.begin, self.end = wp.clone(self.model.joint_q), wp.clone(self.model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_preset="surface-fast",
            vbd_options={
                "iterations": args.vbd_iterations,
                "rigid_body_contact_buffer_size": 2048,
                "rigid_body_particle_contact_buffer_size": 4096,
                "rigid_contact_history": True,
                "rigid_contact_hard": False,
                "rigid_avbd_contact_beta": RIGID_CONTACT_BETA,
                "friction_epsilon": 1.0e-4,
                "particle_enable_self_contact": True,
                # Cache fixed DAT geometry, not its displacement-dependent planes.
                "particle_enable_truncation_cache": True,
                "particle_self_contact_radius": 0.001,
                "particle_self_contact_margin": 0.0025,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.008,
                # The stiff, light shell needs a global translation solve to
                # transmit contact friction through the cup within a substep.
                "particle_enable_multilevel_correction": True,
                "particle_multilevel_operator": "galerkin",
                "particle_multilevel_cluster_size": 400,
                "particle_multilevel_coarse_iterations": 8,
                "particle_enable_coupled_translation": True,
                "particle_enable_surface_cache": False,
                "particle_multilevel_relaxation": 1.0,
                "particle_multilevel_max_radius_fraction": 0.25,
                "particle_multilevel_checkpoints": tuple(i for i in (2, 4) if i <= args.vbd_iterations),
            },
            collision_options={
                "broad_phase": "sap",
                "contact_matching": "latest",
                "rigid_contact_max": 32768,
                "soft_contact_max": 65536,
                "soft_contact_margin": 0.003,
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": self.hand_shapes + self.popcorn_shapes,
            },
        )
        digits = np.full(self.model.shape_count, -1, dtype=np.int32)
        shape_bodies = self.model.shape_body.numpy()
        for shape in self.hand_shapes:
            label = self.model.body_label[shape_bodies[shape]]
            for digit, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
                if f"/left_{name}" in label:
                    digits[shape] = digit
        self.grasp_shape_digit = wp.array(digits, dtype=int, device=self.model.device)
        self.grasp_normal_force = wp.zeros(5, dtype=float, device=self.model.device)
        self.grasp_force_filtered = np.zeros(5)
        self.grasp_joint_offset = np.zeros(5)
        viewer.set_model(self.model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.set_camera(wp.vec3(1.75, 1.65, 1.62), -17.0, -127.0)
        if isinstance(viewer, newton.viewer.ViewerGL):
            viewer.renderer.exposure = 1.25
            viewer.renderer.shadow_extents = 3.0
            viewer.renderer.sky_upper = (0.16, 0.19, 0.24)
            viewer.renderer.ambient_sky = (0.65, 0.68, 0.72)

    def _build(self):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        SolverMJVBDV2.register_custom_attributes(builder)
        # Preserve the 65 kN/m mixed hand/paper contact while giving lightweight
        # kernels a less stiff paper contact; the solver averages the pair.
        builder.default_shape_cfg.ke = 1.2e5
        builder.default_shape_cfg.kd = 10.0
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.margin = 0.0005
        builder.default_shape_cfg.gap = 0.002
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        builder.add_urdf(
            str(ROBOT),
            xform=wp.transform(wp.vec3(ROBOT_X, 0, 0), wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        self.robot_joints, self.robot_coords = builder.joint_count, builder.joint_coord_count
        self.wrists = [
            next(i for i, label in enumerate(builder.body_label) if label.endswith(f"/{side}_j7"))
            for side in ("left", "right")
        ]
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        self.fingers = [[], []]
        self.finger_names = [[], []]
        for j, label in enumerate(builder.joint_label):
            name = label.rsplit("/", 1)[-1]
            coordinate = builder.joint_q_start[j]
            if name in ("ANKLE", "KNEE", "BUTTOCK"):
                builder.joint_q[coordinate] = math.radians({"ANKLE": 45, "KNEE": -90, "BUTTOCK": 45}[name])
            for side, prefix in enumerate(("LEFT_", "RIGHT_")):
                if name.startswith(prefix) and ("HAND_" in name or "_PIP" in name):
                    self.fingers[side].append(coordinate)
                    self.finger_names[side].append(name[len(prefix) :])
                    builder.joint_q[coordinate] = math.pi / 2 if side and name.endswith("THUMB2") else 0.0
        mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        self.hand_shapes = []
        for shape in range(builder.shape_count):
            body = builder.shape_body[shape]
            name = builder.body_label[body].lower() if body >= 0 else ""
            if any(word in name for word in ("j7", "hand", "thumb", "index", "middle", "ring", "pinky")):
                if builder.shape_flags[shape] & mask:
                    self.hand_shapes.append(shape)
        # Use the same closed collision decomposition as the conveyor grasp demo.
        # The URDF visual shapes stay separate; no fingertip spheres are added.
        builder.approximate_meshes(
            method="vhacd",
            shape_indices=self.hand_shapes,
            raise_on_failure=True,
            keep_visual_shapes=False,
            maxConvexHulls=8,
            resolution=200000,
            minimumVolumePercentErrorAllowed=1.0,
            maxRecursionDepth=8,
            maxNumVerticesPerCH=64,
            asyncACD=False,
        )
        self.hand_shapes = [
            s
            for s in range(builder.shape_count)
            if builder.shape_body[s] >= 0
            and builder.shape_flags[s] & mask
            and any(
                word in builder.body_label[builder.shape_body[s]].lower()
                for word in ("j7", "hand", "thumb", "index", "middle", "ring", "pinky")
            )
        ]
        self.ik_model = builder.finalize()
        self.ik_model.body_label = list(self.ik_model.body_label)
        self.ik_model.joint_label = list(self.ik_model.joint_label)
        # Include visual and collision geometry: the prescribed arm cannot be
        # pushed out of a table by contact forces. Screen commands before FK is
        # applied to the physical moving boundary, not after penetration occurs.
        self.table_guard_meshes = []
        for s, mesh in enumerate(builder.shape_source):
            body = builder.shape_body[s]
            if body < 0 or mesh is None or not hasattr(mesh, "vertices"):
                continue
            name = builder.body_label[body].lower()
            if not any(token in name for token in ("left_", "right_")):
                continue
            transform = builder.shape_transform[s]
            rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            points = (np.asarray(mesh.vertices) * np.asarray(builder.shape_scale[s])) @ rotation.T
            points += np.asarray(wp.transform_get_translation(transform))
            self.table_guard_meshes.append((body, points, np.asarray(mesh.indices).reshape(-1, 3)))
        self.table_guard_q = wp.clone(self.ik_model.joint_q)
        self.table_guard_state = self.ik_model.state()
        self.table_guard_bodies = np.array([body for body, _, _ in self.table_guard_meshes])
        lower = np.array([points.min(axis=0) for _, points, _ in self.table_guard_meshes])
        upper = np.array([points.max(axis=0) for _, points, _ in self.table_guard_meshes])
        self.table_guard_centers = (lower + upper) * 0.5
        self.table_guard_extents = (upper - lower) * 0.5
        # Sub-gram grains start with a well-conditioned ALM penalty;
        # the solver ramps it toward these ceilings only as penetration requires.
        cfg = newton.ModelBuilder.ShapeConfig(ke=TRAY_CONTACT_KE, kd=0.2, mu=0.7, margin=0.0005, density=700)
        visual = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)

        def box(pos, half, color, *, body=-1, config=cfg, label="", opacity=1.0):
            return builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=config,
                color=color,
                label=label,
                opacity=opacity,
            )

        builder.add_ground_plane(color=(0.18, 0.20, 0.23))
        table_back = 1.50 + WORKSPACE_X
        box(
            ((TABLE_FRONT + table_back) / 2, 0, TABLE - 0.025),
            ((table_back - TABLE_FRONT) / 2, 0.76, 0.025),
            (0.80, 0.78, 0.71),
            label="worktop",
        )
        leg_half_height = (TABLE - 0.05) / 2
        for x in (TABLE_FRONT + 0.07, 1.43 + WORKSPACE_X):
            for y in (-0.56, 0.56):
                box((x, y, leg_half_height), (0.023, 0.023, leg_half_height), (0.25, 0.28, 0.31))
        # Open-front red steel popcorn warmer: floor and walls are real collisions.
        machine_shape_start = builder.shape_count
        steel, red = (0.64, 0.67, 0.70), (0.65, 0.055, 0.035)
        box((0.81, -0.37, TABLE + 0.011), (0.22, 0.18, 0.011), red, label="machine_base")
        box((0.81, -0.37, TABLE + 0.025), (0.20, 0.16, 0.004), steel, label="popcorn_tray")
        box((0.945, -0.37, TABLE + 0.075), (0.005, 0.16, 0.045), steel, label="tray_rear_baffle")
        for y in (-0.54, -0.20):
            box((0.81, y, TABLE + 0.07), (0.21, 0.007, 0.045), steel)
            box(
                (0.81, y, TABLE + 0.295),
                (0.20, 0.002, 0.18),
                (0.72, 0.88, 0.91),
                label="machine_side_glass",
                opacity=0.18,
            )
            for x in (1.01,):
                box((x, y, TABLE + 0.26), (0.011, 0.011, 0.25), red)
        box((1.02, -0.37, TABLE + 0.25), (0.008, 0.17, 0.22), steel)
        box((0.81, -0.37, TABLE + 0.52), (0.235, 0.195, 0.035), red, label="machine_canopy")
        for y in (-0.52, -0.22):
            box((0.80, y, TABLE + 0.478), (0.16, 0.008, 0.004), (1.0, 0.78, 0.35), config=visual)
        # A kettle under the canopy leaves an unobstructed scooping bay below.
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.87, -0.37, TABLE + 0.37), wp.quat_identity()),
            radius=0.075,
            half_height=0.047,
            cfg=visual,
            color=steel,
        )
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(wp.vec3(0.87, -0.37, TABLE + 0.42), wp.quat_identity()),
            radius=0.079,
            half_height=0.005,
            cfg=visual,
            color=steel,
        )
        box((0.87, -0.37, TABLE + 0.435), (0.023, 0.009, 0.01), (0.12, 0.12, 0.12), config=visual)
        box((0.96, -0.37, TABLE + 0.395), (0.055, 0.012, 0.012), (0.12, 0.12, 0.12), config=visual)
        machine_shift = MACHINE_PLAN - np.array((0.81, -0.37, TABLE))
        for shape in range(machine_shape_start, builder.shape_count):
            pose = builder.shape_transform[shape]
            position = np.asarray(wp.transform_get_translation(pose)).copy()
            position[0] = 0.81 + 0.75 * (position[0] - 0.81)
            if builder.shape_type[shape] == newton.GeoType.BOX:
                scale = builder.shape_scale[shape]
                builder.shape_scale[shape] = wp.vec3(0.75 * scale[0], scale[1], scale[2])
            builder.shape_transform[shape] = wp.transform(
                wp.vec3(*(position + machine_shift)), wp.transform_get_rotation(pose)
            )
        riser_half_height = float(MACHINE[2] - TABLE) / 2
        box(
            (MACHINE_PLAN[0], MACHINE_PLAN[1], TABLE + riser_half_height),
            (0.165, 0.18, riser_half_height),
            red,
            label="machine_riser",
        )
        # Longitudinal cylindrical grip and rolled aluminum scoop.
        for shape in range(machine_shape_start, builder.shape_count):
            builder.shape_transform[shape] = wp.transform_multiply(STATION_FRAME, builder.shape_transform[shape])
        self.scoop_body = builder.add_body(xform=wp.transform(wp.vec3(*HANDLE), STATION_ROTATION), label="scoop")
        grip_cfg = cfg.copy()
        grip_cfg.density = 350
        grip_cfg.ke = 4e4
        grip_cfg.kd = 10
        grip_cfg.mu = 1.2
        builder.add_shape_cylinder(
            self.scoop_body,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
            radius=HANDLE_RADIUS,
            half_height=0.055,
            cfg=grip_cfg,
            color=(0.10, 0.12, 0.13),
            label="round_scoop_handle",
        )
        metal = cfg.copy()
        metal.density = 2700
        builder.add_shape_cylinder(
            self.scoop_body,
            xform=wp.transform(wp.vec3(0.095, 0, -0.010), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -1.326)),
            radius=0.005,
            half_height=0.04123,
            cfg=metal,
            color=steel,
        )
        panels = scoop_bowl_panels()
        self.scoop_vertices = np.concatenate([np.asarray(panel.vertices) for panel in panels])
        for index, panel in enumerate(panels):
            builder.add_shape_convex_hull(
                self.scoop_body, mesh=panel, cfg=metal, color=steel, label=f"scoop_aluminum_panel_{index}"
            )
        # Blender-authored display meshes never contribute mass or contacts.
        # The pan/grip and enclosure collision envelopes remain unchanged.
        with POPCORN_PROPS.open(encoding="utf-8") as stream:
            prop_assets = json.load(stream)
        if prop_assets["version"] != 1 or prop_assets["units"] != "m":
            raise ValueError("Unsupported popcorn asset format")
        for shape in range(machine_shape_start, builder.shape_count):
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
        cup_asset = None
        prop_visuals = []
        for asset in prop_assets["meshes"]:
            if asset["group"] == "cup":
                cup_asset = asset
                continue
            if not asset.get("display", True):
                continue
            vertices = np.asarray(asset["render_vertices"], dtype=np.float32)
            normals = np.asarray(asset["render_normals"], dtype=np.float32)
            if asset["group"] == "machine":
                rotation = np.asarray(wp.quat_to_matrix(STATION_ROTATION)).reshape(3, 3)
                vertices = vertices @ rotation.T + MACHINE
                normals = normals @ rotation.T
                body = -1
            else:
                body = self.scoop_body
            prop_visuals.append(
                (asset, body, newton.Mesh(vertices, np.arange(len(vertices), dtype=np.int32), normals=normals))
            )
        self.popcorn_bodies = []
        self.popcorn_shapes = []
        rng = np.random.default_rng(37)
        corn_cfg = newton.ModelBuilder.ShapeConfig(ke=GRAIN_CONTACT_KE, kd=0.02, mu=0.45, density=90, margin=0.0003)
        corn_cfg.configure_sdf(force_sdf=True)
        grain_mesh = popcorn_mesh()
        for i in range(self.args.popcorn_count):
            center = wp.vec3(*_popcorn_spawn_position(i))
            body = builder.add_body(
                xform=wp.transform(
                    center,
                    STATION_ROTATION * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(rng.uniform(-math.pi, math.pi))),
                ),
                label=f"popcorn_{i}",
            )
            self.popcorn_bodies.append(body)
            color = (1.0, float(rng.uniform(0.80, 0.94)), float(rng.uniform(0.43, 0.65)))
            self.popcorn_shapes.append(builder.add_shape_convex_hull(body, mesh=grain_mesh, cfg=corn_cfg, color=color))
        points, faces = cup_mesh()
        if cup_asset is None or not np.allclose(cup_asset["vertices"], points, atol=1e-7, rtol=0):
            raise ValueError("Blender cup must preserve the physical shell vertex correspondence")
        if not np.array_equal(cup_asset["faces"], faces):
            raise ValueError("Blender cup topology must match the deformable shell")
        self.cup_faces = faces
        # A 0.24 kg/m^2 shell gives approximately a 7 g paper cup.
        colors = [CUP_MATERIAL_COLORS[index] for index in cup_asset["material_indices"]]
        builder.add_cloth_mesh(
            pos=wp.vec3(*CUP),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=points.tolist(),
            indices=faces.reshape(-1).tolist(),
            density=0.24,
            tri_ke=PAPER_MEMBRANE_STIFFNESS,
            tri_ka=PAPER_MEMBRANE_STIFFNESS,
            tri_kd=0.01,
            edge_ke=PAPER_BENDING_STIFFNESS,
            edge_kd=0.002,
            particle_radius=0.001,
            color=np.asarray(colors, dtype=np.float32),
            label="paper_cup",
        )
        builder.color()
        # Joint-only coloring places free grains together. Interleaving the
        # initial packing improves contact propagation; later dynamic contacts
        # may still join a color, so this is not exact dynamic graph coloring.
        grain_ids = np.asarray(self.popcorn_bodies, dtype=np.int32)
        # Prescribed robot bodies are never solved by AVBD. Exclude their
        # empty solve colors, not their collision shapes or moving boundaries.
        groups = [np.asarray([self.scoop_body], dtype=np.int32)]
        index = np.arange(len(grain_ids))
        # Keep adjacent grains alternating. Simultaneous soft-contact pairs
        # use the solver's majorizer; this does not filter any collisions.
        grain_colors = index % 2
        groups.extend(grain_ids[grain_colors == color] for color in range(2))
        builder.body_color_groups = [group for group in groups if len(group)]
        self.model = builder.finalize()
        self.prop_render_data = [
            (
                asset["name"],
                body,
                mesh,
                wp.array([asset["color"]], dtype=wp.vec3, device=self.model.device),
                wp.array([(asset["roughness"], asset["metallic"], 0, 0)], dtype=wp.vec4, device=self.model.device),
                wp.array([asset["opacity"]], dtype=float, device=self.model.device),
            )
            for asset, body, mesh in prop_visuals
        ]
        self.prop_identity = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.model.device)
        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_mu = 0.9
        self.rest_cup = self.model.particle_q.numpy().copy()
        self.rim_indices = _cup_rim_indices(points, faces)
        self.base_indices = np.flatnonzero(np.isclose(points[:, 2], -CUP_HEIGHT / 2))
        self.reference_angles = wp.clone(self.model.edge_rest_angle)
        self.plastic_angles = wp.zeros_like(self.reference_angles)
        self.accumulated_angles = wp.zeros_like(self.reference_angles)
        self.cup_triangles = wp.array(faces.reshape(-1), dtype=int, device=self.model.device)
        self.cup_inner_triangles = wp.array(faces[:, ::-1].reshape(-1).copy(), dtype=int, device=self.model.device)
        self.cup_sides = [
            wp.array(
                faces[np.asarray(cup_asset["material_indices"]) == index].reshape(-1),
                dtype=int,
                device=self.model.device,
            )
            for index in range(len(CUP_MATERIAL_COLORS))
        ]

    def _ik(self):
        # Hand-base coordinates, fitted against the actual URDF surfaces.
        # Left hand: thumb up, palm toward -Y, fingers toward +X. Right hand:
        # palm above the round shaft, with the fingers curling underneath.
        hand_rotations = [
            wp.quat_from_matrix(wp.mat33(0, 1, 0, 0, 0, -1, -1, 0, 0))
            * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -LEFT_GRIP_TILT)
            * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -LEFT_GRIP_TILT_Z),
            _right_hand_rotation(),
        ]
        grasp_centers = [wp.vec3(*center) for center in GRASP_CENTERS]
        self.rotations, self.offsets, self.grips = [], [], []
        for side in range(2):
            mount = (
                wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -math.pi / 2)
                * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi if side == 0 else 0.0)
                * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2)
            )
            self.rotations.append(hand_rotations[side] * wp.quat_inverse(mount))
            self.offsets.append(-np.asarray(wp.quat_rotate(hand_rotations[side], grasp_centers[side])))
            self.grips.append(
                np.asarray([grasp_joint_angle(side, name) for name in self.finger_names[side]], dtype=np.float32)
            )
        q = self.ik_model.joint_q.numpy()
        lower, upper = self.ik_model.joint_limit_lower.numpy(), self.ik_model.joint_limit_upper.numpy()
        starts, dofs = self.ik_model.joint_q_start.numpy(), self.ik_model.joint_qd_start.numpy()
        locked = []
        active_dofs = np.zeros(self.ik_model.joint_dof_count, dtype=bool)
        arms = {f"{side}_J{i}" for side in ("LEFT", "RIGHT") for i in range(1, 8)}
        # Reach with the arms. Do not solve an unreachable tray target by
        # leaning the torso backward; the workstation is within arm reach.
        controlled = arms
        for j, label in enumerate(self.ik_model.joint_label):
            name = label.rsplit("/", 1)[-1]
            if name in controlled:
                active_dofs[dofs[j] : dofs[j + 1]] = True
            if name not in controlled and starts[j + 1] > starts[j]:
                locked.append(starts[j])
                lower[dofs[j]], upper[dofs[j]] = q[starts[j]] - 1e-5, q[starts[j]] + 1e-5
        self.lock_indices = wp.array(locked, dtype=int, device=self.model.device)
        self.lock_values = wp.array(q[locked], dtype=float, device=self.model.device)
        self.position_goals, self.rotation_goals = [], []
        for side, center in enumerate((CUP, HANDLE)):
            target = center + self.offsets[side] + APPROACH[side]
            self.position_goals.append(
                ik.IKObjectivePosition(self.wrists[side], TCP, wp.array([target], dtype=wp.vec3))
            )
            self.rotation_goals.append(
                ik.IKObjectiveRotation(
                    self.wrists[side], wp.quat_identity(), wp.array([wp.vec4(*self.rotations[side])], dtype=wp.vec4)
                )
            )
        limits = ik.IKObjectiveJointLimit(wp.array(lower), wp.array(upper), weight=30)
        elbows = [
            next(i for i, name in enumerate(self.ik_model.body_label) if name.endswith(f"/{side}_j4"))
            for side in ("left", "right")
        ]
        elbow_goals = [
            ik.IKObjectivePosition(
                elbow,
                wp.vec3(),
                wp.array([wp.vec3(0.30, 0.32 if side == 0 else -0.32, 1.03)], dtype=wp.vec3),
                weight=0.02,
            )
            for side, elbow in enumerate(elbows)
        ]
        self.ik_solver = _ContinuousIK(
            self.ik_model,
            elbows=elbows,
            n_problems=1,
            objectives=self.position_goals + self.rotation_goals + elbow_goals + [limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            joint_dof_mask=wp.array(active_dofs, dtype=wp.bool, device=self.model.device),
            compact_dof_mask=self.model.device.is_cuda,
            parallel_objectives=False,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_state = self.ik_model.state()
        self.max_ik_position_error = 0.0
        targets = [center + self.offsets[side] + APPROACH[side] for side, center in enumerate((CUP, HANDLE))]
        arm_coords = [
            int(starts[j]) for j, name in enumerate(self.ik_model.joint_label) if name.rsplit("/", 1)[-1] in arms
        ]
        initial_q = self.ik_q.numpy().copy()
        # Original-limit URDF IK seed for the coordinated overhand workstation.
        # This is only a robot planning seed; the complete path is checked below.
        initial_q[0, arm_coords] = (
            0.0144163,
            -1.7067342,
            -0.7542595,
            -1.3887618,
            -0.2611011,
            -0.7830787,
            -0.9704825,
            -0.9692214,
            -1.0738721,
            0.5207778,
            1.5360833,
            2.4136404,
            -0.6009967,
            -1.0037693,
        )
        self.ik_q.assign(initial_q)
        search_graph = None
        self.ik_graph = None
        if self.model.device.is_cuda:
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=1)
            self.ik_q.assign(initial_q)
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_q, self.ik_q, iterations=80)
            search_graph = capture.graph
            with wp.ScopedCapture() as capture:
                self.ik_solver.step(self.ik_q, self.ik_q, iterations=32)
                wp.launch(lock_joints, len(locked), [self.lock_indices, self.lock_values, self.ik_q])
            self.ik_graph = capture.graph

        def solve_search(repeats=1):
            for _ in range(repeats):
                if search_graph is None:
                    self.ik_solver.step(self.ik_q, self.ik_q, iterations=80)
                else:
                    wp.capture_launch(search_graph)

        rng = np.random.default_rng(23)
        arm_dofs = [int(dofs[j]) for j, name in enumerate(self.ik_model.joint_label) if name.rsplit("/", 1)[-1] in arms]
        # Pronated grasps can require forearm rotations beyond +/- 90 degrees.
        # Search the original URDF range, not an artificial narrow seed range.
        arm_seeds = rng.uniform(lower[arm_dofs], upper[arm_dofs], (35, len(arm_coords)))
        best_score, best_q = (True, float("inf")), None
        best_failed_targets = []
        # Plan with the same initial, fixed tool-in-hand transform used at runtime.
        # No later rebaselining is allowed to conceal tool slip during pickup.
        estimated_tool = wp.transform(wp.vec3(*HANDLE), STATION_ROTATION)
        estimated_wrist = wp.transform(
            wp.vec3(*(HANDLE + self.offsets[1])) - wp.quat_rotate(self.rotations[1], TCP),
            self.rotations[1],
        )
        estimated_grasp = (
            wp.transform_multiply(wp.transform_inverse(estimated_tool), estimated_wrist),
            estimated_tool,
            HANDLE.copy(),
            0.0,
        )
        estimated_cup_center = self.rest_cup.astype(np.float64).mean(axis=0) + np.array((0.0, 0.0, CUP_LIFT))
        estimated_cup_wrist = wp.transform(
            wp.vec3(*(CUP + np.array((0, 0, CUP_LIFT)) + self.offsets[0])) - wp.quat_rotate(self.rotations[0], TCP),
            self.rotations[0],
        )
        estimated_cup_grasp = (
            estimated_cup_wrist,
            estimated_cup_center,
            CUP + np.array((0, 0, CUP_LIFT)),
        )
        # Select a feasible elbow-down IK branch once, without altering joint limits.
        # Try the URDF-screened branch first. It still must pass the complete
        # preflight; retain randomized original-limit seeds as a fallback.
        for seed in range(36):
            self.sim_time = 0.0
            self.tool_grasp = None
            self.cup_grasp = None
            self._set_wrist_targets()
            candidate = initial_q.copy()
            if seed:
                candidate[0, arm_coords] = arm_seeds[seed - 1]
            self.ik_q.assign(candidate)
            if seed >= 24:
                # A redundant arm can reach both endpoints on different IK
                # branches. Seed from the pour pose and track backward before
                # validating the complete forward path with the same limits.
                for time in np.linspace(19.0, 0.0, 153):
                    self.sim_time = float(time)
                    self.tool_grasp = estimated_grasp
                    self.cup_grasp = estimated_cup_grasp if time >= 5.0 else None
                    self._set_wrist_targets()
                    solve_search(5 if time == 19.0 else 1)
                self.sim_time = 0.0
                self.tool_grasp = None
                self.cup_grasp = None
                self._set_wrist_targets()
            solve_search(5)
            wp.launch(lock_joints, len(locked), [self.lock_indices, self.lock_values, self.ik_q])
            newton.eval_fk(self.ik_model, self.ik_q[0], self.ik_model.joint_qd, self.ik_state)
            poses = self.ik_state.body_q.numpy()
            error = 0.0
            for side, wrist in enumerate(self.wrists):
                actual = np.asarray(wp.transform_point(wp.transform(*poses[wrist]), TCP))
                error += float(np.linalg.norm(actual - targets[side]))
                error += 0.1 * (1 - abs(float(np.dot(poses[wrist, 3:], np.asarray(self.rotations[side])))))
            score = 100000 * error + float(np.sum(poses[elbows, 2]))
            if error > 0.005:
                print(f"IK seed {seed}: reject initial_error={error:.6f}", flush=True)
                continue
            grasp_q = self.ik_q.numpy().copy()
            path_error = 0.0
            worst_target = None
            failed_targets = []
            for time in np.linspace(0.0, 27.0, 217):
                self.sim_time = float(time)
                self.tool_grasp = estimated_grasp
                self.cup_grasp = estimated_cup_grasp if time >= 5.0 else None
                path_targets, _ = self._set_wrist_targets()
                solve_search()
                # Coarse planning samples are farther apart than control frames.
                # As in runtime IK, refresh the elbow reference and refine an
                # unconverged target before declaring it unreachable.
                for refinement in range(4):
                    newton.eval_fk(self.ik_model, self.ik_q[0], self.ik_model.joint_qd, self.ik_state)
                    path_poses = self.ik_state.body_q.numpy()
                    converged = True
                    for side, (target, rotation) in enumerate(path_targets):
                        pose = path_poses[self.wrists[side]]
                        actual = np.asarray(wp.transform_point(wp.transform(*pose), TCP))
                        converged &= float(np.linalg.norm(actual - target)) <= 0.004
                        converged &= abs(float(np.dot(pose[3:], np.asarray(rotation)))) >= math.cos(math.radians(2.5))
                    if converged or refinement == 3:
                        break
                    solve_search()
                for side, (target, rotation) in enumerate(path_targets):
                    pose = path_poses[self.wrists[side]]
                    actual = np.asarray(wp.transform_point(wp.transform(*pose), TCP))
                    position_error = float(np.linalg.norm(actual - target))
                    rotation_dot = abs(float(np.dot(pose[3:], np.asarray(rotation))))
                    rotation_error = 2 * math.acos(np.clip(rotation_dot, 0.0, 1.0))
                    if position_error > 0.004 or rotation_error > math.radians(5):
                        failed_targets.append(
                            (float(time), side, round(position_error * 1000, 1), round(math.degrees(rotation_error), 1))
                        )
                    if position_error > path_error:
                        worst_target = (float(time), side, np.round(target, 3).tolist())
                    path_error = max(path_error, position_error)
                    path_error = max(path_error, 0.1 * (1 - abs(float(np.dot(pose[3:], np.asarray(rotation))))))
            score += 10000 * path_error
            print(
                f"IK seed {seed}: initial_error={error:.6f}, path_error={path_error:.6f}, worst={worst_target}",
                flush=True,
            )
            # Any feasible full path outranks a lower-error starting pose that
            # later crosses a joint limit. Never trade reachability for posture.
            rank = (bool(failed_targets), score)
            if np.isfinite(score) and rank < best_score:
                best_score, best_q = rank, grasp_q
                best_failed_targets = failed_targets
            if not failed_targets:
                break
        print("Selected IK path errors over 4 mm:", best_failed_targets, flush=True)
        self.planned_ik_failures = tuple(best_failed_targets)
        if best_q is None or best_failed_targets:
            raise RuntimeError(f"No full bimanual IK path reaches all targets: {self.planned_ik_failures}")
        self.sim_time = 0.0
        self.tool_grasp = None
        self.cup_grasp = None
        self._set_wrist_targets()
        self.ik_q.assign(best_q)
        self._check_ik(
            [
                (center + self.offsets[side] + APPROACH[side], self.rotations[side])
                for side, center in enumerate((CUP, HANDLE))
            ]
        )
        wp.copy(self.model.joint_q, self.ik_q[0], count=self.robot_coords)
        initial = self.model.joint_q.numpy()
        for side in range(2):
            for index, name, angle in zip(self.fingers[side], self.finger_names[side], self.grips[side], strict=True):
                joint = next(j for j in range(self.robot_joints) if starts[j] == index and starts[j + 1] > index)
                dof = int(dofs[joint])
                asset_lower = float(self.ik_model.joint_limit_lower.numpy()[dof])
                asset_upper = float(self.ik_model.joint_limit_upper.numpy()[dof])
                if not asset_lower - 1e-6 <= angle <= asset_upper + 1e-6:
                    raise ValueError(f"Grasp angle for {name} exceeds the unchanged URDF joint limits")
                initial[index] = self._finger_angle(side, name, angle, 0.0)
        self.model.joint_q.assign(initial)

    @staticmethod
    def _finger_angle(side, name, angle, closure):
        # Oppose the thumb before the approach. Rotating it from the open pose
        # beside the cup sweeps its distal link through the wall midway through
        # closure, even though both endpoint poses are clear.
        if side == 0 and "THUMB" in name:
            # Keep the thumb outside the shell's finite contact thickness while
            # the fingers approach. Engage both sides of the pinch together.
            opposed = angle if name.endswith("THUMB2") else math.radians(LEFT_THUMB_APPROACH)
            pinch = float(np.clip((closure - 0.88) / 0.12, 0.0, 1.0))
            return opposed + pinch * (angle - opposed)
        if "THUMB" in name:
            # Establish opposition gradually, together with the four fingers.
            # The initial just-touching thumb pose cannot resist shaft torque.
            return angle + (math.radians(2.0) * float(closure) if name.endswith("THUMB1") else 0.0)
        if side:
            # Start with a physical grasp of the empty dynamic utensil.
            # Finish the power-grasp squeeze before moving the pan into the tray.
            return angle + (math.radians(1.2) * float(closure) if name.endswith("PIP") else 0.0)
        return float(closure) * angle

    def _check_ik(self, targets):
        """Reject unreachable wrist targets before driving a moving collider."""
        newton.eval_fk(self.ik_model, self.ik_q[0], self.ik_model.joint_qd, self.ik_state)
        poses = self.ik_state.body_q.numpy()
        for side, (target, rotation) in enumerate(targets):
            pose = poses[self.wrists[side]]
            actual = np.asarray(wp.transform_point(wp.transform(*pose), TCP))
            error = float(np.linalg.norm(actual - target))
            dot = abs(float(np.dot(pose[3:], np.asarray(rotation))))
            angle = 2 * math.acos(np.clip(dot, 0.0, 1.0))
            self.max_ik_position_error = max(self.max_ik_position_error, error)
            if not np.isfinite(error + angle) or error > 0.005 or angle > math.radians(5):
                raise _WristTargetError(
                    f"{'Left' if side == 0 else 'Right'} wrist target unreachable at {self.sim_time:.3f}s: "
                    f"position error {error * 1000:.1f} mm, rotation error {math.degrees(angle):.1f} deg",
                    side=side,
                )

    @property
    def trajectory_time(self):
        """Pause robot transport, not physical simulation, until the grip settles."""
        return self.sim_time - getattr(self, "grasp_wait", 0.0) - getattr(self, "tool_wait", 0.0)

    def _targets(self):
        # Positions describe the desired grasped-object frame, never set its state.
        # Right-hand keyframes use the station's planning frame. Rotate the
        # entire path with the machine, not just the initial utensil pose.
        HANDLE = TOOL_PLAN_ORIGIN
        MACHINE = MACHINE_PLAN
        lifted_cup = CUP + np.array((0, 0, CUP_LIFT))
        receiving_cup = RECEIVING_CUP
        keyframes = [
            (0.0, CUP + APPROACH[0], HANDLE + APPROACH[1], 0.0, 0.0, "approach"),
            (1.0, CUP, HANDLE, 0.0, 0.0, "close"),
            (2.5, CUP, HANDLE, 1.0, 0.0, "settle_grasp"),
            (3.5, CUP, HANDLE, 1.0, 0.0, "lift"),
            (5.0, lifted_cup, HANDLE, 1.0, 0.0, "reach_machine"),
            (6.5, lifted_cup, np.array((HANDLE[0], MACHINE[1], TABLE + 0.18)), 1.0, 0.0, "approach_tray"),
            # Lower the lip on the empty tray apron before advancing. Lowering above
            # the first grain row traps grains under the bowl and tips it upward.
            (7.5, lifted_cup, np.array((MACHINE[0] - 0.395, MACHINE[1], TABLE + 0.18)), 1.0, 0.0, "lower_scoop"),
            (
                8.5,
                lifted_cup,
                # Account for the real 1.2 mm sheet and both contact margins;
                # do not command the bottom surface through the tray floor.
                np.array((MACHINE[0] - 0.395, MACHINE[1], MACHINE[2] + 0.0315 - SCOOP_FLOOR_Z)),
                1.0,
                0.0,
                "scoop",
            ),
            (
                10.5,
                lifted_cup,
                np.array((MACHINE[0] - 0.21, MACHINE[1], MACHINE[2] + 0.0315 - SCOOP_FLOOR_Z)),
                1.0,
                0.0,
                "raise_scoop",
            ),
            (11.25, lifted_cup, np.array((MACHINE[0] - 0.21, MACHINE[1], TABLE + 0.20)), 1.0, 0.0, "retract_scoop"),
            # Pull the whole bowl past the front glass edge before yawing.
            (12.5, lifted_cup, np.array((MACHINE[0] - 0.445, MACHINE[1], TABLE + 0.21)), 1.0, 0.0, "transfer"),
            (13.5, lifted_cup, np.array((MACHINE[0] - 0.445, MACHINE[1], TABLE + 0.25)), 1.0, 0.0, "cross_above_cup"),
            (15.0, receiving_cup, np.array((MACHINE[0] - 0.32, MACHINE[1], TABLE + 0.25)), 1.0, 0.0, "align_cup"),
            (16.5, receiving_cup, np.array((MACHINE[0] - 0.32, MACHINE[1], TABLE + 0.25)), 1.0, 0.0, "pour"),
            (19.0, receiving_cup, np.array((MACHINE[0] - 0.32, MACHINE[1], TABLE + 0.25)), 1.0, 1.0, "drain"),
            (21.0, receiving_cup, np.array((MACHINE[0] - 0.32, MACHINE[1], TABLE + 0.25)), 1.0, 1.0, "withdraw"),
            (27.0, receiving_cup, np.array((MACHINE[0] - 0.36, MACHINE[1], TABLE + 0.27)), 1.0, 0.0, "complete"),
            (25.0, receiving_cup, HANDLE, 1.0, 0.0, "complete"),
        ]
        t = self.trajectory_time
        for a, b in pairwise(keyframes):
            if t <= b[0]:
                u = np.clip((t - a[0]) / (b[0] - a[0]), 0, 1)
                u = u * u * (3 - 2 * u)
                self.phase = a[5]
                values = [(1 - u) * np.asarray(a[i]) + u * np.asarray(b[i]) for i in range(1, 5)]
                values[1] = station_point(values[1])
                return tuple(values)
        self.phase = "complete"
        return keyframes[-1][1], station_point(keyframes[-1][2]), *keyframes[-1][3:5]

    def _set_wrist_targets(self, *, update_feedback=True):
        left, right, closure, pour_fraction = self._targets()
        t = self.trajectory_time
        if hasattr(self, "state_0") and t <= 1.0:
            # Visual grasp servo: follow the observed cup only while approaching.
            # Freeze before closing, so contact does not make the wrist chase it.
            current = self._read_state("particle_q").astype(np.float64)
            rest_center, current_center = self.rest_cup.mean(axis=0, dtype=np.float64), current.mean(axis=0)
            # Keep the intended upright grasp orientation. Following the cup's
            # contact-induced tilt here creates an unwanted positive feedback.
            center = current_center + CUP - rest_center
            self.cup_pickup_frame = wp.transform(wp.vec3(*center), wp.quat_identity())
        turn = np.clip((t - 13.5) / 1.5, 0, 1)
        turn = turn * turn * (3 - 2 * turn)
        # Approach the receiving cup from its right, not from behind it where
        # the long handle would be trapped between the cup and the torso.
        yaw = SCOOP_YAW + turn * (POUR_YAW - SCOOP_YAW)
        carry_fraction = np.clip((t - PITCH_BEGIN) / (PITCH_END - PITCH_BEGIN), 0, 1)
        carry_blend = carry_fraction * carry_fraction * (3 - 2 * carry_fraction)
        entry = float(np.clip((t - 6.5) / 1.0, 0, 1))
        entry = entry * entry * (3 - 2 * entry)
        entry_pitch = SCOOP_ENTRY_PITCH * entry
        pitch = (1 - carry_blend) * entry_pitch + carry_blend * CARRY_PITCH
        # Tilt on the clear apron first, then lower the physical front edge.
        # Compensating its geometry avoids commanding a tilted blade into the tray.
        right = right.copy()
        if entry > 0 and carry_blend < 1:
            observed_rotation = None
            if hasattr(self, "state_0") and self.tool_grasp is not None:
                observed_rotation = self._read_state("body_q")[self.scoop_body, 3:]
            right[2] += (1 - carry_blend) * _scoop_floor_height_offset(
                self.scoop_vertices, entry_pitch, observed_rotation=observed_rotation
            )
        # Pour through the open front lip by lifting the handle. Do not roll
        # around the shaft: that reverses the wrist and spills over a side wall.
        pitch = (1 - pour_fraction) * pitch + pour_fraction * POUR_PITCH
        if t >= 13.5:
            # Position the physical pouring edge, not the handle, over the rim.
            # Use the rest-pose estimate during initial IK branch selection.
            rim = RECEIVING_CUP + np.array((0, 0, CUP_HEIGHT / 2))
            if self.cup_grasp is not None:
                receiving_center = RECEIVING_CUP
                rim_offset = self.rest_cup[self.rim_indices].mean(axis=0) - self.rest_cup.mean(axis=0)
                rim = self.cup_grasp[1] + receiving_center - self.cup_grasp[2] + rim_offset
            if hasattr(self, "state_0") and t >= 15.0:
                observed_rim = self._read_state("particle_q")[self.rim_indices].mean(axis=0)
                tracking = np.clip((t - 15.0) / 1.5, 0.0, 1.0)
                tracking = tracking * tracking * (3 - 2 * tracking)
                rim = (1 - tracking) * rim + tracking * observed_rim
            u = np.clip((t - 13.5) / 3.0, 0, 1)
            u = u * u * (3 - 2 * u)
            pour_position = self._pour_tool_position(rim, float(pitch), float(yaw))
            if hasattr(self, "state_0"):
                observed_rotation = wp.quat(*self._read_state("body_q")[self.scoop_body, 3:])
                measured_position = self._pour_tool_position(
                    rim, float(pitch), float(yaw), observed_rotation=observed_rotation
                )
                correction = getattr(self, "outlet_correction", np.zeros(3))
                if update_feedback:
                    desired = measured_position - pour_position
                    desired *= min(1.0, 0.05 / max(float(np.linalg.norm(desired)), 1e-12))
                    delta = desired - correction
                    # Limit only robot visual-servo motion to 30 mm/s. A noisy
                    # tool angle must not cause an impulsive moving boundary.
                    delta *= min(1.0, (0.03 / 60.0) / max(float(np.linalg.norm(delta)), 1e-12))
                    correction = correction + delta
                    self.outlet_correction = correction
                pour_position = pour_position + correction
            right = (1 - u) * right + u * pour_position
            if t >= 21.0:
                if hasattr(self, "state_0") and not hasattr(self, "withdraw_origin"):
                    self.withdraw_origin = self._read_state("body_q")[self.scoop_body, :3].copy()
                right, pitch, yaw = self._withdraw_tool_pose(rim, t, start=getattr(self, "withdraw_origin", None))
        targets = []
        for side, center in enumerate((left, right)):
            rotation_delta = wp.quat_identity()
            if side:
                rotation_delta = (
                    wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw))
                    * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(pitch))
                    * wp.quat_inverse(STATION_ROTATION)
                )
            rotation = rotation_delta * self.rotations[side]
            target = center + np.asarray(wp.quat_rotate(rotation_delta, wp.vec3(*self.offsets[side])))
            if not side and self.cup_pickup_frame is not None and self.cup_grasp is None:
                pickup_rotation = wp.transform_get_rotation(self.cup_pickup_frame)
                target = np.asarray(wp.transform_point(self.cup_pickup_frame, wp.vec3(*(target - CUP))))
                rotation = pickup_rotation * rotation
            if not side and self.cup_grasp is not None:
                initial_wrist, _, reference_center = self.cup_grasp
                # Preserve the established grasp while translating the wrist.
                # An automatic leveling rotation can roll the tapered cup out
                # of a shallow friction grasp, even when its tilt is harmless.
                wrist_position = wp.transform_get_translation(initial_wrist) + wp.vec3(*(center - reference_center))
                rotation = wp.transform_get_rotation(initial_wrist)
                target = np.asarray(wrist_position + wp.quat_rotate(rotation, TCP))
            if side and self.tool_grasp is not None:
                relative, initial_pose, reference_center, observed_time = self.tool_grasp
                u = np.clip((t - observed_time) / 1.5, 0, 1)
                u = u * u * (3 - 2 * u)
                # Blend from the observed grasp pose to a level scoop. Only the
                # robot target changes: the utensil remains an unconstrained body.
                tool_rotation = rotation_delta * wp.quat_slerp(
                    wp.transform_get_rotation(initial_pose), STATION_ROTATION, float(u)
                )
                if hasattr(self, "state_0"):
                    # Visual attitude servo: compensate a small angular slip by
                    # moving the wrist. Never overwrite the dynamic tool pose.
                    if update_feedback:
                        actual = self._read_state("body_q")[self.scoop_body, 3:]
                        error = tool_rotation * wp.quat_inverse(wp.quat(*actual))
                        increment = wp.quat_slerp(wp.quat_identity(), error, 4.0 / 60.0)
                        correction = wp.normalize(increment * self.tool_orientation_correction)
                        angle = 2 * math.acos(min(abs(float(correction[3])), 1.0))
                        if angle > TOOL_FEEDBACK_MAX_ANGLE:
                            correction = wp.quat_slerp(wp.quat_identity(), correction, TOOL_FEEDBACK_MAX_ANGLE / angle)
                        self.tool_orientation_correction = correction
                    tool_rotation = self.tool_orientation_correction * tool_rotation
                shift = (1 - u) * (np.asarray(wp.transform_get_translation(initial_pose)) - reference_center)
                desired_tool = wp.transform(wp.vec3(*(center + shift)), tool_rotation)
                wrist_pose = wp.transform_multiply(desired_tool, relative)
                rotation = wp.transform_get_rotation(wrist_pose)
                target = np.asarray(wp.transform_point(wrist_pose, TCP))
            self.position_goals[side].set_target_position(0, wp.vec3(*target))
            self.rotation_goals[side].set_target_rotation(0, wp.vec4(*rotation))
            targets.append((target, rotation))
        return targets, closure

    @staticmethod
    def _withdraw_tool_pose(rim, time, *, start=None):
        """Clear the cup before returning the empty scoop to the right-hand bay."""
        if start is None:
            start = Example._pour_tool_position(rim, POUR_PITCH, POUR_YAW)
        # Withdraw laterally before leveling the bowl; a large upward arc
        # drives the pronated wrist into its high-workspace joint limits.
        clear = start + np.array((0.0, -0.12, 0.02))
        if time <= 22.0:
            u = float(np.clip(time - 21.0, 0.0, 1.0))
            u = u * u * (3 - 2 * u)
            return (1 - u) * start + u * clear, POUR_PITCH, POUR_YAW
        u = float(np.clip((time - 22.0) / 5.0, 0.0, 1.0))
        u = u * u * (3 - 2 * u)
        # Keep the entire 280 mm bowl in front of the machine while leveling.
        # Returning to HANDLE at this height sweeps it through the side glass.
        end = HANDLE + np.array((-0.16, 0, 0.12))
        return (1 - u) * clear + u * end, (1 - u) * POUR_PITCH, (1 - u) * POUR_YAW + u * SCOOP_YAW

    @staticmethod
    def _pour_tool_position(rim, pitch, yaw=0.0, *, observed_rotation=None):
        """Keep the open front lip above the cup while pitching the handle up."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch)
        if observed_rotation is not None:
            rotation = observed_rotation
        outlet = wp.vec3(BOWL_FRONT, 0, SCOOP_FLOOR_Z)
        # This is a robot aiming target only; grains and scoop remain dynamic.
        fraction = np.clip((pitch - CARRY_PITCH) / (POUR_PITCH - CARRY_PITCH), 0.0, 1.0)
        # Approach above the holding hand, then bring the open lip closer as
        # the handle rises. This avoids folding the elbow tightly by the torso.
        height = (1 - fraction) * 0.077 + fraction * 0.025
        # Aim upstream: grains retain forward velocity after leaving the lip.
        # Only the robot target is adjusted; no grain position/velocity is set.
        aim = np.asarray(rim) + np.array((-0.02 * fraction * math.cos(yaw), -0.02 * fraction * math.sin(yaw), height))
        return aim - np.asarray(wp.quat_rotate(rotation, outlet))

    def _calibrate_tool_grasp(self, require_lift=True):
        poses = self._read_state("body_q")
        selected = poses[[self.scoop_body, self.wrists[1]]]
        if not np.isfinite(selected).all() or (require_lift and selected[0, 2] < HANDLE[2] + 0.06):
            raise AssertionError("The scoop must remain lifted before calibrating the held-tool trajectory")
        tool, wrist = (wp.transform(*pose) for pose in selected)
        self.tool_grasp = (
            wp.transform_multiply(wp.transform_inverse(tool), wrist),
            tool,
            np.asarray(self._targets()[1]),
            self.trajectory_time,
        )
        self.initial_grip_center = np.asarray(
            wp.transform_get_translation(wp.transform_multiply(wp.transform_inverse(wrist), tool))
        )

    def _calibrate_cup_grasp(self):
        """Validate the lift and record the wrist frame for translation-only transport."""
        q = self._read_state("particle_q").astype(np.float64)
        center = q.mean(axis=0)
        axis = q[self.rim_indices].mean(axis=0) - q[self.base_indices].mean(axis=0)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        lift = center[2] - self.rest_cup.mean(axis=0)[2]
        if not np.isfinite(q).all() or lift < 0.08 or axis[2] < 0.5:
            raise AssertionError(
                f"Cup grasp failed before transport: lift={lift * 1000:.1f} mm, upright cosine={axis[2]:.4f}"
            )
        self.cup_grasp = (
            wp.transform(*self._read_state("body_q")[self.wrists[0]]),
            center,
            np.asarray(self._targets()[0]),
        )

    @staticmethod
    def _grip_center_in_wrist(wrist_in_tool):
        """Measure shaft-center slip independently of rotation about that shaft."""
        return np.asarray(wp.transform_get_translation(wp.transform_inverse(wrist_in_tool)))

    def _read_state(self, name):
        """Share read-only positions within one controller phase, never across physics."""
        cache = getattr(self, "_controller_snapshot", None)
        if cache is None:
            return getattr(self.state_0, name).numpy()
        if name not in cache:
            values = getattr(self.state_0, name).numpy()
            values.flags.writeable = False
            cache[name] = values
        return cache[name]

    def step(self):
        self._controller_snapshot = {}
        if self.cup_grasp is None and self.trajectory_time >= 5.0 - 1e-8:
            self._calibrate_cup_grasp()
        if self.cup_grasp is not None:
            initial_wrist, initial_center = self.cup_grasp[:2]
            current_wrist = wp.transform(*self._read_state("body_q")[self.wrists[0]])
            current_center = self._read_state("particle_q").mean(axis=0)
            initial_local = wp.transform_point(wp.transform_inverse(initial_wrist), wp.vec3(*initial_center))
            current_local = wp.transform_point(wp.transform_inverse(current_wrist), wp.vec3(*current_center))
            slip = float(wp.length(current_local - initial_local))
            self.cup_grip_slip = slip
            if not math.isfinite(slip) or slip > 0.030:
                raise AssertionError(f"Cup slipped out of the grasp at {self.sim_time:.3f}s: {slip * 1000:.1f} mm")
        if self.tool_grasp is not None:
            # Keep the observed grasp frame fixed. Updating it from ongoing
            # slip makes the wrist chase the slipping shaft on every frame.
            poses = self._read_state("body_q")
            actual = wp.transform_multiply(
                wp.transform_inverse(wp.transform(*poses[self.scoop_body])),
                wp.transform(*poses[self.wrists[1]]),
            )
            relative = self.tool_grasp[0]
            grip_slip = np.linalg.norm(self._grip_center_in_wrist(actual) - self.initial_grip_center)
            if not math.isfinite(grip_slip) or grip_slip > 0.01:
                raise AssertionError(
                    f"The round grip lost the scoop at {self.sim_time:.3f}s: "
                    f"shaft-center slip from initial grasp {grip_slip * 1000:.1f} mm; "
                    f"actual={np.asarray(actual)}, reference={np.asarray(relative)}"
                )
        _, _, closure, _ = self._targets()
        self._update_grasp_pressure(closure)
        self._wait_for_grasp()
        previous_correction = self.tool_orientation_correction
        targets, closure = self._set_wrist_targets(update_feedback=not getattr(self, "_tool_ik_holding", False))
        try:
            self._solve_wrist_ik_with_feedback(targets, previous_correction=previous_correction)
        except _WristTargetError as error:
            if error.side != 1 or self.trajectory_time < 5.0:
                raise
            self._recover_tool_ik(previous_correction, error)
        else:
            self._tool_ik_holding = False
        wp.copy(self.begin, self.state_0.joint_q)
        end = self.state_0.joint_q.numpy()
        end[: self.robot_coords] = self.ik_q.numpy().reshape(-1)
        for side in range(2):
            for index, name, angle in zip(self.fingers[side], self.finger_names[side], self.grips[side], strict=True):
                end[index] = self._finger_angle(side, name, angle, closure)
                if side == 0 and name.endswith("THUMB1"):
                    end[index] = np.clip(end[index] + self.grasp_joint_offset[0], 0, math.radians(50))
                elif side == 0 and "THUMB" not in name:
                    digit = next(i for i, token in enumerate(("INDEX", "MIDDLE", "RING", "PINKY")) if token in name)
                    mcp, pip = _grasp_finger_commands(
                        math.radians(GRASP_ANGLES[0][2 * digit]) * closure,
                        math.radians(GRASP_ANGLES[0][2 * digit + 1]) * closure,
                        self.grasp_joint_offset[digit + 1],
                    )
                    end[index] = pip if name.endswith("PIP") else mcp
        self.end.assign(end)
        self._check_robot_table(end)
        self._controller_snapshot = None
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda and self.args.substeps % 2 == 0:
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.sim_time += 1 / 60
        measure_interval = 6 if self.args.test and self.sim_time >= 16.0 else 30
        if int(round(self.sim_time * 60)) % measure_interval == 0:
            self._measure()

    def _recover_tool_ik(self, previous_correction, error):
        """Hold the path clock and gently unwind feedback before executing IK."""
        old_wait = getattr(self, "tool_wait", 0.0)
        angle = 2 * math.acos(min(abs(float(previous_correction[3])), 1.0))
        if old_wait + 1 / 60 > 2.0 or angle < 1e-5:
            raise error
        self.tool_wait = old_wait + 1 / 60
        # Revisit the preceding path time, not an unreachable future target.
        # Physics still advances, and accepted feedback changes by <=15 deg/s.
        fraction = max(0.0, 1.0 - math.radians(0.25) / angle)
        self.tool_orientation_correction = wp.quat_slerp(wp.quat_identity(), previous_correction, fraction)
        try:
            targets, _ = self._set_wrist_targets(update_feedback=False)
            self._solve_wrist_ik(targets)
            self._tool_ik_holding = False
        except _WristTargetError as retry_error:
            if retry_error.side != 1:
                self.tool_wait = old_wait
                self.tool_orientation_correction = previous_correction
                raise
            # Do not execute an unreachable IK iterate. Hold the robot at its
            # actual current coordinates while feedback unwinds over subsequent
            # physical frames. The bounded path pause remains visible in reports.
            held = self.state_0.joint_q.numpy()[: self.robot_coords]
            self.ik_q.assign(held.reshape(1, -1))
            self._tool_ik_holding = True

    def _check_robot_table(self, joint_q):
        if hasattr(self, "_gpu_table_guard"):
            return self._gpu_table_guard(joint_q)
        """Reject a colliding final arm/finger command without moving the robot."""
        self.table_guard_q.assign(joint_q[: self.robot_coords])
        newton.eval_fk(self.ik_model, self.table_guard_q, self.ik_model.joint_qd, self.table_guard_state)
        poses = self.table_guard_state.body_q.numpy()
        rotations = {
            int(body): np.asarray(wp.quat_to_matrix(wp.quat(*poses[body, 3:]))).reshape(3, 3)
            for body in np.unique(self.table_guard_bodies)
        }
        matrices = np.array([rotations[int(body)] for body in self.table_guard_bodies])
        centers = np.einsum("nij,nj->ni", matrices, self.table_guard_centers) + poses[self.table_guard_bodies, :3]
        extents = np.einsum("nij,nj->ni", np.abs(matrices), self.table_guard_extents)
        # Enclose the transformed local AABB. Round outward: this broad phase
        # may over-select meshes, but must never omit the precise old test.
        lower, upper = centers - extents - 1e-6, centers + extents + 1e-6
        table_lower = np.array((TABLE_FRONT, -0.76, TABLE - 0.05)) - 0.001
        table_upper = np.array((1.50 + WORKSPACE_X, 0.76, TABLE)) + 0.001
        candidates = np.all(upper >= table_lower, axis=1) & np.all(lower <= table_upper, axis=1)
        for index in np.flatnonzero(candidates):
            body, points, indices = self.table_guard_meshes[index]
            world = points @ rotations[body].T + poses[body, :3]
            if _bounds_overlap_table(world.min(axis=0), world.max(axis=0)) and _mesh_intersects_table(world, indices):
                raise RuntimeError(
                    f"Robot/table clearance rejected for {self.ik_model.body_label[body]} "
                    f"at {self.sim_time:.3f}s: bounds={world.min(axis=0)}..{world.max(axis=0)}; "
                    "replan the arm/workstation, do not disable collisions"
                )

    def _solve_wrist_ik_with_feedback(self, targets, *, previous_correction=None):
        """Backtrack unexecuted attitude feedback, never dynamic prop state.

        Backtrack only this frame's increment, never abruptly unwind an
        established grasp correction. If the path remains unreachable with
        the previous correction, retain the failure.
        """
        previous_q = self.ik_q.numpy().copy()
        correction = self.tool_orientation_correction
        if previous_correction is None:
            previous_correction = wp.quat_identity()
        for fraction in (1.0, 0.5, 0.25, 0.0):
            if fraction != 1.0:
                self.ik_q.assign(previous_q)
                self.tool_orientation_correction = wp.quat_slerp(previous_correction, correction, fraction)
                targets, _ = self._set_wrist_targets(update_feedback=False)
            try:
                self._solve_wrist_ik(targets)
            except RuntimeError:
                if fraction == 0.0:
                    self.ik_q.assign(previous_q)
                    self.tool_orientation_correction = previous_correction
                    raise
            else:
                return

    def _solve_wrist_ik(self, targets):
        """Refine a robot command and reject it before writing physical state."""
        for attempt in range(5):
            if self.ik_graph is None:
                self.ik_solver.step(self.ik_q, self.ik_q, iterations=32)
                wp.launch(lock_joints, len(self.lock_indices), [self.lock_indices, self.lock_values, self.ik_q])
            else:
                wp.capture_launch(self.ik_graph)
            try:
                self._check_ik(targets)
            except RuntimeError:
                if attempt == 4:
                    raise
            else:
                break

    def _update_grasp_pressure(self, closure):
        """A bounded tactile finger servo, not an object constraint or pose drive."""
        contacts = self.solver.contacts
        if closure < 0.88 or contacts is None:
            return
        self.grasp_normal_force.zero_()
        wp.launch(
            measure_grasp_pressure,
            contacts.soft_contact_max,
            [
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                self.solver.vbd_solver.body_particle_contact_penalty_k,
                self.model.shape_body,
                self.model.shape_margin,
                self.grasp_shape_digit,
                self.state_0.body_q,
                self.state_0.particle_q,
                self.model.particle_radius,
                self.grasp_normal_force,
            ],
            device=self.model.device,
        )
        measured = self.grasp_normal_force.numpy()
        if not np.isfinite(measured).all():
            raise AssertionError("Nonfinite grasp contact pressure")
        self.grasp_force_filtered += 0.25 * (measured - self.grasp_force_filtered)
        if self.grasp_ready and self.trajectory_time >= 5.0:
            self.grasp_contact_samples += 1
            self.five_finger_contact_samples += int(_all_grasp_digits_in_contact(self.grasp_force_filtered))
        self.grasp_joint_offset = _grasp_offset_update(
            self.grasp_joint_offset, self.grasp_force_filtered, self.grasp_ready
        )

    def _wait_for_grasp(self):
        """Require sustained opposing contacts before lifting a free cup."""
        if self.grasp_ready:
            if 3.5 <= self.trajectory_time < 5.0:
                delay = (1.0 - _lift_speed_factor(self.grasp_force_filtered)) / 60.0
                self.grasp_wait += delay
                self.lift_wait = getattr(self, "lift_wait", 0.0) + delay
                if self.lift_wait > 8.0:
                    raise AssertionError("Five-finger contact was not maintained during lift")
            return
        if self.trajectory_time < 3.5 - 1e-8:
            return
        forces = self.grasp_force_filtered
        supported = bool(
            np.isfinite(forces).all() and forces[0] >= 1.0 and np.all(forces[1:] >= 0.2) and np.sum(forces[1:]) >= 1.0
        )
        self.grasp_ready_time = self.grasp_ready_time + 1 / 60 if supported else 0.0
        self.grasp_ready = self.grasp_ready_time >= 0.3
        if not self.grasp_ready:
            self.grasp_wait += 1 / 60
            if self.grasp_wait > 8.0:
                raise AssertionError("Cup grasp did not establish opposing contact forces before lift")

    def _simulate(self):
        for i in range(self.args.substeps):
            wp.launch(
                prescribe,
                self.robot_joints,
                [
                    self.begin,
                    self.end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (i + 1) / self.args.substeps,
                    60.0,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                update_paper_hinges,
                self.model.edge_count,
                [
                    self.state_0.particle_q,
                    self.model.edge_indices,
                    self.reference_angles,
                    self.plastic_angles,
                    self.accumulated_angles,
                    self.model.edge_rest_angle,
                    PAPER_YIELD_ANGLE,
                    2.0,
                ],
            )

    def _measure(self):
        q = self._read_state("particle_q")
        bodies = self._read_state("body_q")
        if not np.isfinite(q).all() or not np.isfinite(bodies).all():
            raise AssertionError("Nonfinite popcorn scene state")
        center = q.mean(axis=0)
        rim = q[self.rim_indices]
        axis = rim.mean(axis=0) - q[self.base_indices].mean(axis=0)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        singular = np.linalg.svd(rim - rim.mean(axis=0), compute_uv=False)
        self.rim_min_radius = float(singular[1] * math.sqrt(2 / len(rim)))
        self.cup_upright = float(axis[2])
        self.current_cup_lift = float(center[2] - self.rest_cup.mean(axis=0)[2])
        self.max_lift = max(self.max_lift, float(center[2] - self.rest_cup.mean(axis=0)[2]))
        self.max_scoop_lift = max(self.max_scoop_lift, float(bodies[self.scoop_body, 2] - HANDLE[2]))
        # Remove rigid translation/rotation before measuring shell distortion.
        rest = self.rest_cup - self.rest_cup.mean(axis=0)
        current = q - center
        u, _, vt = np.linalg.svd(rest.T @ current)
        u[:, -1] *= np.linalg.det(u @ vt)
        rotation = u @ vt
        self.max_cup_deformation = max(
            self.max_cup_deformation, float(np.sqrt(np.mean(np.sum((rest @ rotation - current) ** 2, axis=1))))
        )
        inside = _inside_cup(bodies[self.popcorn_bodies, :3], q, self.cup_faces, self.rim_indices)
        self.current_inside = int(np.count_nonzero(inside))
        self.max_inside = max(self.max_inside, self.current_inside)
        scoop = bodies[self.scoop_body]
        self.grip_slip = 0.0
        if hasattr(self, "initial_grip_center"):
            wrist_in_tool = wp.transform_multiply(
                wp.transform_inverse(wp.transform(*scoop)), wp.transform(*bodies[self.wrists[1]])
            )
            self.grip_slip = float(np.linalg.norm(self._grip_center_in_wrist(wrist_in_tool) - self.initial_grip_center))
        scoop_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*scoop[3:]))).reshape(3, 3)
        scoop_local = (bodies[self.popcorn_bodies, :3] - scoop[:3]) @ scoop_rotation
        scoop_mask = _inside_scoop(scoop_local)
        self.scoop_count = int(np.count_nonzero(scoop_mask))
        self.max_scoop_count = max(self.max_scoop_count, self.scoop_count)
        bowl_height = float((self.scoop_vertices @ scoop_rotation.T + scoop[:3])[:, 2].min())
        self.scoop_support_clearance = bowl_height - TABLE
        if self.trajectory_time >= 10.5 and bowl_height > MACHINE[2] + 0.039:
            self.loaded_lift_count = max(self.loaded_lift_count, self.scoop_count)
            if self.payload_eligible is not None:
                self.lifted_payload |= scoop_mask & self.payload_eligible
        self.delivered_inside = int(np.count_nonzero(inside & self.lifted_payload))
        if self.trajectory_time >= 18.0 and self.sim_time > self.last_retention_time + 0.49:
            self.retained_samples = self.retained_samples + 1 if self.delivered_inside >= 8 else 0
            self.last_retention_time = self.sim_time
        station_rotation = np.asarray(wp.quat_to_matrix(STATION_ROTATION)).reshape(3, 3)
        grains = (bodies[self.popcorn_bodies, :3] - MACHINE) @ station_rotation
        machine_mask = (
            (np.abs(grains[:, 0]) < 0.21) & (np.abs(grains[:, 1]) < 0.17) & (grains[:, 2] > 0.0) & (grains[:, 2] < 0.25)
        )
        self.grains_in_machine = int(np.count_nonzero(machine_mask))
        if self.payload_eligible is None and self.trajectory_time >= 7.49:
            self.payload_eligible = machine_mask.copy()
        self.max_grain_speed = float(
            np.linalg.norm(self.state_0.body_qd.numpy()[self.popcorn_bodies, :3], axis=1).max()
        )
        if not math.isfinite(self.max_grain_speed) or self.max_grain_speed > 20.0:
            raise AssertionError(f"Unstable grain motion: {self.max_grain_speed:.3g} m/s")
        if self.args.test:
            contacts = self.solver.contacts
            obstacles = []
            if contacts is not None:
                count = min(int(contacts.rigid_contact_count.numpy()[0]), len(contacts.rigid_contact_shape0))
                s0, s1 = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
                shape_bodies = self.model.shape_body.numpy()
                for first, second in zip(s0, s1, strict=True):
                    if first < 0 or second < 0:
                        continue
                    if shape_bodies[first] == self.scoop_body and shape_bodies[second] < 0:
                        obstacles.append(self.model.shape_label[second])
                    if shape_bodies[second] == self.scoop_body and shape_bodies[first] < 0:
                        obstacles.append(self.model.shape_label[first])
                    for tool_shape, other_shape in ((first, second), (second, first)):
                        other_body = shape_bodies[other_shape]
                        if shape_bodies[tool_shape] == self.scoop_body and 0 <= other_body < self.scoop_body:
                            name = self.model.body_label[other_body]
                            if other_shape not in self.hand_shapes or "/right_" not in name:
                                obstacles.append(name)
            print(
                f"{self.sim_time:.1f}s {self.phase}: cup={center.round(3)}, scoop={bodies[self.scoop_body, :3].round(3)}, "
                f"inside={int(inside.sum())}, in_scoop={self.scoop_count}, rim_radius_mm={1000 * self.rim_min_radius:.1f}, "
                f"tray={self.grains_in_machine}, grain_speed={self.max_grain_speed:.2f}, "
                f"scoop_q={scoop[3:].round(3)}, grip_slip_mm={self.grip_slip * 1000:.1f}, "
                f"finger_force_N={self.grasp_force_filtered.round(3)}, "
                f"finger_adjust_deg={np.degrees(self.grasp_joint_offset).round(2)}, "
                f"rim={rim.mean(axis=0).round(3)}, obstacles={sorted(set(obstacles))}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        persistent = hasattr(self.viewer, "log_mesh_group") and getattr(self, "_persistent_render", True)
        if not hasattr(self, "_uploaded_props"):
            self._uploaded_props = set()
        for name, body, mesh, colors, materials, opacities in self.prop_render_data:
            if persistent:
                first = name not in self._uploaded_props
                if first:
                    self.viewer.log_geo(
                        "/prop_mesh/" + name, newton.GeoType.MESH, (1.0, 1.0, 1.0), 0.0, True, geo_src=mesh, hidden=True
                    )
                    self._uploaded_props.add(name)
                if first or body >= 0:
                    self.viewer.log_instances(
                        "/props/" + name,
                        "/prop_mesh/" + name,
                        self.prop_identity if body < 0 else self.state_0.body_q[body : body + 1],
                        None,
                        colors if first else None,
                        materials if first else None,
                        opacities=opacities if first else None,
                    )
                continue
            self.viewer.log_shapes(
                "/props/" + name,
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                self.prop_identity if body < 0 else self.state_0.body_q[body : body + 1],
                colors=colors,
                materials=materials,
                geo_src=mesh,
                opacities=opacities,
            )
        if persistent:
            parts = [
                (f"/cup/{name}", indices, color, 0.88, 0.0)
                for name, indices, color in zip(
                    ("paper", "print", "lap_seam"), self.cup_sides, CUP_MATERIAL_COLORS, strict=True
                )
            ]
            parts.append(("/cup/inside", self.cup_inner_triangles, (0.88, 0.82, 0.68), 0.88, 0.0))
            self.viewer.log_mesh_group("/cup", self.state_0.particle_q, parts)
            self.viewer.end_frame()
            return
        for name, indices, color in zip(
            ("paper", "print", "lap_seam"), self.cup_sides, CUP_MATERIAL_COLORS, strict=True
        ):
            self.viewer.log_mesh(
                f"/cup/{name}",
                self.state_0.particle_q,
                indices,
                color=color,
                roughness=0.88,
                metallic=0,
                backface_culling=True,
            )
        # Print only the exterior; the inside is unprinted paper. Opposite
        # winding selects the other side of the same physical shell.
        self.viewer.log_mesh(
            "/cup/inside",
            self.state_0.particle_q,
            self.cup_inner_triangles,
            color=(0.88, 0.82, 0.68),
            roughness=0.88,
            metallic=0,
            backface_culling=True,
        )
        self.viewer.end_frame()

    def test_final(self):
        self._measure()
        report = {
            "time": self.sim_time,
            "trajectory_time": self.trajectory_time,
            "grasp_wait_s": self.grasp_wait,
            "tool_recovery_wait_s": getattr(self, "tool_wait", 0.0),
            "five_finger_contact_fraction": self.five_finger_contact_samples / max(self.grasp_contact_samples, 1),
            "finger_contact_force_N": self.grasp_force_filtered.tolist(),
            "cup_lift_m": self.max_lift,
            "scoop_lift_m": self.max_scoop_lift,
            "scoop_clearance_m": self.scoop_support_clearance,
            "cup_distortion_rms_m": self.max_cup_deformation,
            "cup_max_plastic_hinge_rotation_rad": float(np.max(np.abs(self.plastic_angles.numpy()))),
            "popcorn_inside": self.max_inside,
            "popcorn_retained": self.current_inside,
            "delivered_popcorn_retained": self.delivered_inside,
            "retained_fraction_of_lifted": self.delivered_inside / max(self.loaded_lift_count, 1),
            "retained_checkpoints": self.retained_samples,
            "max_grains_in_scoop": self.max_scoop_count,
            "grains_lifted_in_scoop": self.loaded_lift_count,
            "max_ik_position_error_m": self.max_ik_position_error,
            "rim_min_radius_m": self.rim_min_radius,
            "cup_upright_cosine": self.cup_upright,
            "grains_in_scoop": self.scoop_count,
            "grip_slip_m": self.grip_slip,
        }
        print(json.dumps(report), flush=True)
        if self.trajectory_time >= 5.0 - 1e-8 and (self.current_cup_lift < 0.08 or self.scoop_support_clearance < 0.05):
            raise AssertionError("The cup must be lifted and the initially held scoop must remain clear of the table")
        if self.trajectory_time >= 12.0 - 1e-8 and self.loaded_lift_count < 3:
            raise AssertionError("The scoop must lift at least three grains clear of the tray")
        if self.trajectory_time >= 20.0:
            if (
                self.five_finger_contact_samples < 0.95 * max(self.grasp_contact_samples, 1)
                or self.current_cup_lift < 0.08
                or self.scoop_support_clearance < 0.05
                or self.current_inside < 8
                or self.delivered_inside < 8
                or self.delivered_inside < 0.7 * self.loaded_lift_count
                or self.retained_samples < 3
                or self.loaded_lift_count < 12
                or self.rim_min_radius < 0.028
                or self.cup_upright < 0.94
                or self.max_cup_deformation > 0.012
            ):
                raise AssertionError(
                    "Delivery quality failed: require >=12 lifted, >=8 and >=70% retained, "
                    "cup upright cosine >=0.94, rim radius >=28 mm and distortion RMS <=12 mm"
                )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=2880)
        parser.add_argument("--popcorn-count", type=int, default=160)
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--vbd-iterations", type=int, default=8)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
