# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example VBD W1 Popcorn
#
# A Dexforce W1 stands at a popcorn warmer with a deformable paper cup and a
# rigid aluminum scoop on the worktop. SolverVBD integrates the paper shell,
# the scoop, and every popcorn kernel together; the robot links are kinematic
# colliders driven by per-frame IK. The left hand grasps the paper cup with
# a five-finger wrap, lifts it, and holds it out over the worktop; the right
# hand picks the scoop up from the worktop with a power grasp around its round
# grip, slides it along the warmer tray into the heap, lifts out a load of
# popcorn, and pours it into the cup the left hand is holding.
#
# Command: python -m newton.examples vbd_w1_popcorn
#
###########################################################################

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
from newton.solvers import SolverObservableFlags

ASSETS = Path(__file__).resolve().parents[3] / "assets"
ROBOT_URDF = ASSETS / "DexforceW1V021" / "DexforceW1V021.urdf"
POPCORN_PROPS = ASSETS / "popcorn" / "props.json"

FPS = 60
DEFAULT_SUBSTEPS = 6
DEFAULT_VBD_ITERATIONS = 8
DEFAULT_POPCORN_COUNT = 320

# Workstation layout [m]. The warmer, its tray, and the display meshes in
# props.json share the station frame, a translation of the planning layout.
ROBOT_X = 0.25
TABLE = 0.84
TABLE_FRONT = 0.57
TABLE_BACK = 1.56
TABLE_HALF_Y = 0.76
MACHINE_PLAN = np.array((0.86, -0.34, TABLE + 0.06))
STATION_OFFSET = np.array((0.15, 0.02, 0.0))
MACHINE = MACHINE_PLAN + STATION_OFFSET

# Scoop body frame: the grip axis is +x; the bowl spans BOWL_REAR..BOWL_FRONT.
SCOOP_FLOOR_Z = -0.048
BOWL_REAR = 0.135
BOWL_FRONT = 0.280
HANDLE_RADIUS = 0.014
# Resting on the worktop on the robot's right, where the pronated right-hand
# grasp is reachable; the bowl ends just short of the warmer's front face and
# the grip overhangs the table edge.
SCOOP_REST = np.array((0.55, -0.20, TABLE + 0.05))

# Paper cup: a tapered 0.24 kg/m^2 shell (about 7 g) with a rolled lip.
CUP_HEIGHT = 0.14
CUP_RADIUS_SCALE = 0.8
CUP = np.array((0.62, 0.27, TABLE + CUP_HEIGHT / 2 + 0.001))
# Cup-stock paperboard. The membrane takes the plane-stress Lame parameters of
# E*t, and each hinge the discrete-shell stiffness of the plate's flexural
# rigidity D = E t^3 / (12 (1 - nu^2)) over its own two triangles.
PAPER_YOUNG_MODULUS = 3.5e9  # [Pa]
PAPER_THICKNESS = 0.35e-3  # [m]
PAPER_POISSON = 0.3
PAPER_MEMBRANE_DAMPING = 0.01
PAPER_BENDING_DAMPING = 0.002
# Paperboard creases once bent past ~1 % strain: at 0.35 mm that is a curvature
# of ~60 1/m, about 0.15 rad across one hinge of this mesh.
PAPER_YIELD_ANGLE = 0.15
PAPER_DENSITY = 0.24  # [kg/m^2]
CUP_MATERIAL_COLORS = ((0.88, 0.82, 0.68), (0.48, 0.33, 0.23), (0.92, 0.86, 0.73))

TRAY_CONTACT_KE = 4.0e4
GRAIN_CONTACT_KE = 1.0e4
# Hand contact stiffness [N/m]; VBD averages it with the paper and kernel stiffness.
HAND_CONTACT_KE = 1.2e5
# Friction of the hands' elastomer finger and palm pads.
HAND_FRICTION = 1.0
HAND_CONTACT_KEYWORDS = ("j7", "hand", "thumb", "index", "middle", "ring", "pinky")

# W1 posture: crouched legs; both arms start from elbow-down IK seeds for
# their grasps [rad].
LEG_POSTURE = {"ANKLE": math.radians(45.0), "KNEE": math.radians(-90.0), "BUTTOCK": math.radians(45.0)}
LEFT_ARM = tuple(f"LEFT_J{i}" for i in range(1, 8))
RIGHT_ARM = tuple(f"RIGHT_J{i}" for i in range(1, 8))
ARM_POSTURE = {
    **dict(zip(LEFT_ARM, (0.0144, -1.7067, -0.7543, -1.3888, -0.2611, -0.7831, -0.9705), strict=True)),
    **dict(zip(RIGHT_ARM, (-0.9692, -1.0739, 0.5208, 1.5361, 2.4136, -0.6010, -1.0038), strict=True)),
}
IK_ITERATIONS = 8
INITIAL_IK_ITERATIONS = 300

# Left-hand cup grasp, fitted against the W1 hand surfaces: thumb up, palm
# toward -y, fingers toward +x. LEFT_GRASP_CENTER is the cup axis point in the
# hand-base frame; TCP is the wrist (left_j7) point that IK places.
TCP = wp.vec3(-0.066, 0.0, 0.0)
LEFT_GRASP_CENTER = wp.vec3(0.00290154, 0.10364690, 0.05120512)
LEFT_GRIP_TILT = math.radians(-9.89896)
LEFT_GRIP_TILT_Z = math.radians(-31.90938)
# Index/middle/ring/pinky MCP and PIP, then THUMB1 and THUMB2 [deg].
LEFT_GRASP_DEG = (18.64723, 12.15433, 48.02730, 3.36554, 64.48925, 1.55911, 70.86599, 2.60215, 7.05563, 45.40551)
# The cup is round, so the fitted grasp may be turned about the cup axis; the
# turn sets which way the forearm leaves the cup [rad].
LEFT_GRASP_YAW = math.radians(-30.0)
CUP_APPROACH = np.array((-0.04 * math.cos(LEFT_GRASP_YAW), -0.04 * math.sin(LEFT_GRASP_YAW), 0.0))
CUP_LIFT = 0.11
# The left hand holds the cup out next to the right hand, where the scoop can
# tip its lip over the rim within reach, with the base clear of the worktop.
RECEIVING_CUP = np.array((0.66, -0.05, CUP[2] + 0.03))

# Right-hand power grasp around the scoop's round grip, fitted against the W1
# hand surfaces: palm above the shaft, fingers curling underneath, rolled about
# the shaft axis and yawed across it. RIGHT_GRASP_CENTER is the shaft axis
# point in the hand-base frame.
RIGHT_GRASP_CENTER = wp.vec3(-0.0005498, 0.0825728, 0.0354889)
RIGHT_HAND_ROLL = math.radians(-121.0)
RIGHT_GRIP_YAW = math.pi / 6.0
RIGHT_GRASP_DEG = (71.9066, 46.0249, 71.1363, 61.4614, 75.0, 75.8950, 74.9999, 82.8914, 20.5, 89.9997)
SCOOP_APPROACH = np.array((0.0, 0.0, 0.08))

# Force-guided grasp acquisition, then hold, as on grippers with fingertip
# force sensing: once a hand has closed to its fitted pre-grasp, each digit
# closes or opens until its measured normal contact force has stayed within
# [target, overload] for a short dwell, then latches. After the acquisition
# window the whole hand holds its joint angles until it is told to release, so
# contact noise and impacts from the load never move the fingers. Digits are
# thumb, index, middle, ring, pinky.
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
GRIP_FORCE_TARGET = (
    # A firm hold on the paper cup, enough to dent and ovalize the thin wall [N].
    np.array((25.0, 6.0, 6.0, 6.0, 6.0)),
    np.array((4.0, 3.0, 3.0, 3.0, 3.0)),  # Right hand on the scoop grip [N].
)
GRIP_FORCE_FILTER = 0.3  # Exponential smoothing of the measured force per frame.
GRIP_SEEK_RATE = 0.3  # Finger speed while seeking the force band [rad/s].
GRIP_OVERLOAD = (4.0, 8.0)  # The band's upper edge: max(4 * target, 8 N).
GRIP_DWELL = 0.1  # Time in the band before a digit latches [s].
GRIP_ACQUIRE_TIME = 1.5  # Acquisition window after the pre-grasp closes [s].
GRIP_OFFSET_RANGE = (-0.3, 0.8)  # Bounds of the closure relative to the pre-grasp [rad].
SCOOP_LIFT = 0.10

# (time [s], cup-center offset, left closure, scoop-grip offset, right closure)
# keyframes. Offsets are from each object's grasp pose; closure is in [0, 1].
KEYFRAMES = (
    (0.0, CUP_APPROACH, 0.0, SCOOP_APPROACH, 0.0),
    (1.0, CUP_APPROACH, 0.0, SCOOP_APPROACH, 0.0),
    (2.0, np.zeros(3), 0.0, np.zeros(3), 0.0),
    (3.0, np.zeros(3), 1.0, np.zeros(3), 1.0),
    (3.5, np.zeros(3), 1.0, np.zeros(3), 1.0),
    (5.0, np.array((0.0, 0.0, CUP_LIFT)), 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
    (13.0, np.array((0.0, 0.0, CUP_LIFT)), 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
    # Lower the cup to the holding height first, then slide it across: carrying
    # it across higher drives the left arm's J2 into its limit.
    (14.0, np.array((0.0, 0.0, RECEIVING_CUP[2] - CUP[2])), 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
    (15.0, RECEIVING_CUP - CUP, 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
    (15.5, RECEIVING_CUP - CUP, 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
    (29.5, RECEIVING_CUP - CUP, 1.0, np.array((0.0, 0.0, SCOOP_LIFT)), 1.0),
)
# The scoop settles on the worktop first; its pose is read once at this time,
# just before the right hand descends, and anchors the grasp target.
SCOOP_PERCEPTION_TIME = 1.0

# Once the scoop is held, its pose in the hand is read once at this time and
# the right arm then steers the scoop itself along SCOOP_PATH.
SCOOP_CALIBRATION_TIME = 5.5
# Grip height that puts the bowl's outer bottom about 1.3 mm above the tray
# floor: the 1.2 mm sheet plus both contact margins stay clear of the floor.
TRAY_FLOOR = MACHINE[2] + 0.029
SCOOP_FLOOR_GRIP_Z = TRAY_FLOOR + 0.0025 - SCOOP_FLOOR_Z
SCOOP_CARRY_PITCH = -0.20  # Nose up [rad], so the load stays in the bowl.
# (time [s], grip-center position, pitch about the scoop's +y [rad]) keyframes,
# heading along +x into the warmer. The first entry is the observed held pose.
SCOOP_PATH = (
    (7.0, np.array((MACHINE[0] - 0.395, MACHINE[1], TRAY_FLOOR + 0.12)), 0.0),
    (8.0, np.array((MACHINE[0] - 0.395, MACHINE[1], SCOOP_FLOOR_GRIP_Z)), 0.0),
    (10.0, np.array((MACHINE[0] - 0.21, MACHINE[1], SCOOP_FLOOR_GRIP_Z)), 0.0),
    (11.0, np.array((MACHINE[0] - 0.21, MACHINE[1], SCOOP_FLOOR_GRIP_Z + 0.10)), SCOOP_CARRY_PITCH),
    (12.0, np.array((MACHINE[0] - 0.45, MACHINE[1], SCOOP_FLOOR_GRIP_Z + 0.12)), SCOOP_CARRY_PITCH),
    (13.0, np.array((MACHINE[0] - 0.45, MACHINE[1], SCOOP_FLOOR_GRIP_Z + 0.12)), SCOOP_CARRY_PITCH),
)

# The pour steers the bowl's front lip, the edge the popcorn leaves from.
SCOOP_LIP = wp.vec3(BOWL_FRONT, 0.0, SCOOP_FLOOR_Z)
POUR_YAW = math.pi / 3.0  # Swing toward the cup so the lip crosses its rim from the right.
POUR_PITCH = math.radians(40.0)  # Nose down, past the kernels' ~24 deg sliding angle.
# Aim offset of the lip from the rim center, along and across the pour heading
# [m], calibrated from the measured stream: kernels keep moving forward as they
# leave the lip and the curved trough releases them slightly to one side.
POUR_AIM_UPSTREAM = 0.026
POUR_AIM_ACROSS = 0.013
# The scoop waits clear of the cup until it has arrived; the rim is read once
# then, to aim the lip.
POUR_SWING_START = 16.5
POUR_PERCEPTION_TIME = 16.5
# (time [s], lip offset from the aim point, yaw [rad], pitch [rad]) keyframes.
# The tilt is slow and the lip low over the rim, so the rolled trough funnels
# a narrow stream into the cup rather than dumping the load at once.
POUR_PATH = (
    (18.5, np.array((0.0, 0.0, 0.06)), POUR_YAW, SCOOP_CARRY_PITCH),
    (21.5, np.array((0.0, 0.0, 0.015)), POUR_YAW, POUR_PITCH),
    (23.5, np.array((0.0, 0.0, 0.015)), POUR_YAW, POUR_PITCH),
    # Withdraw straight up while levelling, clear of the rim, then move aside.
    (24.5, np.array((0.0, 0.0, 0.10)), POUR_YAW, SCOOP_CARRY_PITCH),
    (25.5, np.array((0.0, -0.12, 0.10)), POUR_YAW, SCOOP_CARRY_PITCH),
    # Hold the end pose so the scene is checked at rest.
    (29.5, np.array((0.0, -0.12, 0.10)), POUR_YAW, SCOOP_CARRY_PITCH),
)

CAMERA_POS = wp.vec3(1.75, 1.65, 1.62)
CAMERA_PITCH = -17.0
CAMERA_YAW = -127.0


# One popcorn kernel: four overlapping spheres (center [m], radius [m]).
POPCORN_LOBES = (
    ((0.002, 0.001, 0.0), 0.0065),
    ((-0.004, -0.001, 0.003), 0.005),
    ((-0.001, 0.004, 0.004), 0.005),
    ((0.003, -0.004, -0.003), 0.0045),
)
POPCORN_DENSITY = 90.0
POPCORN_FRICTION = 0.45


def popcorn_lobe_density():
    """Density for the overlapping lobes that gives each kernel its convex-hull mass."""
    sphere = newton.Mesh.create_sphere(radius=1, num_latitudes=12, num_longitudes=24)
    cloud = np.concatenate([np.asarray(sphere.vertices) * r + np.asarray(c) for c, r in POPCORN_LOBES])
    hull = newton.Mesh(cloud, indices=[]).compute_convex_hull()
    triangles = np.asarray(hull.vertices)[np.asarray(hull.indices).reshape(-1, 3)]
    hull_volume = abs(np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6.0
    lobe_volume = sum(4.0 / 3.0 * math.pi * r**3 for _, r in POPCORN_LOBES)
    return float(POPCORN_DENSITY * hull_volume / lobe_volume)


def paper_membrane_lame():
    """Plane-stress Lame parameters (mu, lambda) of the paper sheet [N/m]."""
    stiffness = PAPER_YOUNG_MODULUS * PAPER_THICKNESS
    mu = stiffness / (2.0 * (1.0 + PAPER_POISSON))
    lam = stiffness * PAPER_POISSON / (1.0 - PAPER_POISSON**2)
    return mu, lam


def paper_hinge_stiffness(points, edge):
    """Bending stiffness of one hinge [N/rad] for VBD's per-length dihedral energy.

    A plate of flexural rigidity D bent by an angle theta across a hinge of length L, spread over
    the hinge's width h = (A0 + A1) / (3 L), stores D theta^2 L / (2 h) (Grinspun et al. 2003);
    VBD stores k L theta^2 / 2, so k = 3 D L / (A0 + A1).
    """
    o0, o1, v0, v1 = edge
    flexural_rigidity = PAPER_YOUNG_MODULUS * PAPER_THICKNESS**3 / (12.0 * (1.0 - PAPER_POISSON**2))
    a, b = np.asarray(points[v0]), np.asarray(points[v1])
    length = float(np.linalg.norm(b - a))
    area = sum(
        0.5 * float(np.linalg.norm(np.cross(a - np.asarray(points[o]), b - np.asarray(points[o])))) for o in (o0, o1)
    )
    return 3.0 * flexural_rigidity * length / area


def popcorn_display_mesh(seed=11):
    """A puffed kernel's surface: the boundary of its collision lobes, with shallow puffs.

    Rays from the cluster center leave the lobe union at the farthest lobe they cross, which also
    bridges the creases between lobes; the puffs stay within 0.25 mm of that surface.
    """
    sphere = newton.Mesh.create_sphere(radius=1.0, num_latitudes=24, num_longitudes=48)
    directions = np.asarray(sphere.vertices, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    centers = np.asarray([c for c, _ in POPCORN_LOBES], dtype=np.float64)
    radii = np.asarray([r for _, r in POPCORN_LOBES], dtype=np.float64)
    origin = centers.mean(axis=0)
    offsets = centers - origin
    along = directions @ offsets.T
    across2 = np.sum(offsets**2, axis=1)[None, :] - along**2
    exit_distance = np.where(across2 < radii**2, along + np.sqrt(np.maximum(radii**2 - across2, 0.0)), 0.0)
    distance = exit_distance.max(axis=1)
    rng = np.random.default_rng(seed)
    puffs = rng.normal(size=(14, 3))
    puffs /= np.linalg.norm(puffs, axis=1, keepdims=True)
    distance += 0.00025 * np.exp(-(1.0 - directions @ puffs.T) / 0.04).sum(axis=1).clip(max=1.0)
    vertices = (origin + directions * distance[:, None]).astype(np.float32)
    return newton.Mesh(vertices, np.asarray(sphere.indices, dtype=np.int32))


def cup_mesh(segments=40, rings=12):
    """A tapered paper shell with a subdivided bottom and a rolled lip.

    The lip folds the sheet inward around a 1.4 mm radius and stays open: its
    membrane and bending stiffness reinforce the opening, with no rigid rim.
    """
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
    """A rolled 1.2 mm aluminum trough as closed convex panels, plus its rear wall."""
    angles = np.linspace(-1.30, 1.30, sections + 1)
    radius, thickness = 0.056, 0.0012
    profiles = [
        [(r * math.sin(angle), SCOOP_FLOOR_Z + radius - r * math.cos(angle)) for r in (radius, radius + thickness)]
        for angle in angles
    ]
    panels = []
    for a, b in pairwise(profiles):
        vertices = [(x, y, z) for x in (BOWL_REAR, BOWL_FRONT) for y, z in (*a, *b)]
        panels.append(newton.Mesh(np.asarray(vertices, dtype=np.float32), indices=[]).compute_convex_hull())
    rear = [(x, y, z) for x in (BOWL_REAR - thickness, BOWL_REAR) for profile in profiles for y, z in profile]
    panels.append(newton.Mesh(np.asarray(rear, dtype=np.float32), indices=[]).compute_convex_hull())
    return panels


def popcorn_spawn_grid(count):
    """Stack kernels in a stepped heap on the tray floor; return positions and grid parity.

    The tray is open at the front (-x), so each higher layer starts one column
    further back, near the heap's natural slope; nothing topples over the lip.
    The last column keeps every lobe clear of the rear baffle.
    """
    columns, rows = 9, 13
    cells = [
        (column, row, layer) for layer in range(columns) for row in range(rows) for column in range(layer, columns)
    ]
    if count > len(cells):
        raise ValueError(f"The tray holds at most {len(cells)} kernels")
    positions = [
        MACHINE + np.array((-0.09 + 0.020 * c, -0.138 + 0.023 * r, 0.045 + 0.022 * l)) for c, r, l in cells[:count]
    ]
    parity = [(c + r + l) % 2 for c, r, l in cells[:count]]
    return np.asarray(positions), np.asarray(parity)


def _smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def sample_keyframes(time):
    """Return (cup offset, left closure, scoop offset, right closure) at ``time``."""
    for (t0, *a), (t1, *b) in pairwise(KEYFRAMES):
        if time <= t1:
            u = _smoothstep((time - t0) / (t1 - t0))
            return tuple((1.0 - u) * np.asarray(x) + u * np.asarray(y) for x, y in zip(a, b, strict=True))
    return tuple(np.asarray(x) for x in KEYFRAMES[-1][1:])


def _hand_mount(side):
    """Fixed URDF mount from the j7 wrist link to the hand base (0 = left, 1 = right)."""
    return (
        wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -math.pi / 2)
        * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi if side == 0 else 0.0)
        * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi / 2)
    )


def left_wrist_pose():
    """Wrist rotation and the TCP offset from the cup axis for the fitted left grasp."""
    hand = (
        wp.quat_from_axis_angle(wp.vec3(0, 0, 1), LEFT_GRASP_YAW)
        * wp.quat_from_matrix(wp.mat33(0, 1, 0, 0, 0, -1, -1, 0, 0))
        * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -LEFT_GRIP_TILT)
        * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -LEFT_GRIP_TILT_Z)
    )
    return hand * wp.quat_inverse(_hand_mount(0)), -np.asarray(wp.quat_rotate(hand, LEFT_GRASP_CENTER))


def right_wrist_pose(heading):
    """Wrist rotation and the TCP offset from the shaft axis for the fitted right grasp.

    ``heading`` is the scoop's yaw about +z; the hand stays level about the shaft.
    """
    hand = (
        wp.quat_from_axis_angle(wp.vec3(0, 0, 1), heading)
        * wp.quat_from_axis_angle(wp.vec3(1, 0, 0), RIGHT_HAND_ROLL)
        * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -RIGHT_GRIP_YAW)
    )
    return hand * wp.quat_inverse(_hand_mount(1)), -np.asarray(wp.quat_rotate(hand, RIGHT_GRASP_CENTER))


def _digit_index(name):
    """Index into the fitted grasp angles for a hand joint name without its side prefix."""
    if name.endswith("THUMB1"):
        return 8
    if name.endswith("THUMB2"):
        return 9
    digit = next(i for i, token in enumerate(("INDEX", "MIDDLE", "RING", "PINKY")) if token in name)
    return 2 * digit + int(name.endswith("PIP"))


def left_finger_command(name, closure):
    """Command for one left-hand joint at a closure in [0, 1]."""
    index = _digit_index(name)
    angle = math.radians(LEFT_GRASP_DEG[index])
    if index == 9:
        # The thumb is opposed before the approach and stays there.
        return angle
    if index == 8:
        # Swing the thumb in last, so it meets the wall as the fingers wrap.
        return min(max((closure - 0.88) / 0.12, 0.0), 1.0) * angle
    return closure * angle


def right_finger_command(name, closure):
    """Command for one right-hand joint at a closure in [0, 1]."""
    index = _digit_index(name)
    angle = math.radians(RIGHT_GRASP_DEG[index])
    if index == 9:
        # The thumb is opposed across the palm before the approach.
        return angle
    return closure * angle


def grip_offset_share(name):
    """Digit index and share of the grip offset for a hand joint, or (-1, 0) if uncontrolled.

    The thumb closes about its flexion joint; a finger closes at its knuckle with
    the middle joint following at half the rate.
    """
    index = _digit_index(name)
    if index == 9:
        return -1, 0.0
    if index == 8:
        return 0, 1.0
    return 1 + index // 2, (0.5 if name.endswith("PIP") else 1.0)


class GuardedGrip:
    """Force-guided grasp acquisition for one hand's digits, then a held grasp."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)
        self.overload = np.maximum(GRIP_OVERLOAD[0] * self.target, GRIP_OVERLOAD[1])
        self.reset()

    def reset(self):
        self.offset = np.zeros(len(self.target))
        self.latched = np.zeros(len(self.target), dtype=bool)
        self.time_in_band = np.zeros(len(self.target))
        self.time_acquiring = 0.0

    def update(self, force):
        """Advance one frame from the filtered digit forces [N]; return the closure offsets [rad]."""
        dt = 1.0 / FPS
        self.time_acquiring += dt
        if self.time_acquiring > GRIP_ACQUIRE_TIME:
            return self.offset
        low = force < self.target
        high = force > self.overload
        self.time_in_band = np.where(~low & ~high, self.time_in_band + dt, 0.0)
        self.latched |= self.time_in_band >= GRIP_DWELL
        rate = np.where(low, GRIP_SEEK_RATE, np.where(high, -GRIP_SEEK_RATE, 0.0))
        self.offset = np.where(self.latched, self.offset, np.clip(self.offset + rate * dt, *GRIP_OFFSET_RANGE))
        return self.offset


@wp.kernel
def _write_wrist_targets(left: wp.transform, right: wp.transform, target: wp.array[wp.transform]):
    """Store this frame's sampled wrist targets in a persistent device buffer."""
    target[0] = left
    target[1] = right


@wp.kernel
def _unpack_wrist_targets(
    target: wp.array[wp.transform],
    left_position: wp.array[wp.vec3],
    left_rotation: wp.array[wp.vec4],
    right_position: wp.array[wp.vec3],
    right_rotation: wp.array[wp.vec4],
):
    """Feed both stored wrist targets to their IK position and rotation objectives."""
    for side in range(2):
        tf = target[side]
        q = wp.transform_get_rotation(tf)
        if side == 0:
            left_position[0] = wp.transform_get_translation(tf)
            left_rotation[0] = wp.vec4(q[0], q[1], q[2], q[3])
        else:
            right_position[0] = wp.transform_get_translation(tf)
            right_rotation[0] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def _accumulate_digit_forces(
    contact_f: wp.array[wp.spatial_vector],
    rigid_count: wp.array[wp.int32],
    rigid_max: int,
    rigid_shape0: wp.array[wp.int32],
    rigid_shape1: wp.array[wp.int32],
    rigid_normal: wp.array[wp.vec3],
    soft_count: wp.array[wp.int32],
    soft_shape: wp.array[wp.int32],
    soft_normal: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_digit: wp.array[wp.int32],
    held_body: int,
    digit_force: wp.array[float],
):
    """Sum the normal contact force [N] each hand digit applies to the object it holds.

    Rigid records count only against ``held_body`` (the scoop); soft records are the
    paper cup, the only soft body. Popcorn striking a finger is not grip force.
    """
    i = wp.tid()
    if i < rigid_max:
        if i >= rigid_count[0]:
            return
        shape0 = rigid_shape0[i]
        shape1 = rigid_shape1[i]
        if shape0 < 0 or shape1 < 0:
            return
        normal_force = wp.abs(wp.dot(wp.spatial_top(contact_f[i]), rigid_normal[i]))
        if shape_digit[shape0] >= 0 and shape_body[shape1] == held_body:
            wp.atomic_add(digit_force, shape_digit[shape0], normal_force)
        if shape_digit[shape1] >= 0 and shape_body[shape0] == held_body:
            wp.atomic_add(digit_force, shape_digit[shape1], normal_force)
    else:
        j = i - rigid_max
        if j >= soft_count[0]:
            return
        shape = soft_shape[j]
        if shape >= 0:
            digit = shape_digit[shape]
            if digit >= 0:
                wp.atomic_add(digit_force, digit, wp.abs(wp.dot(wp.spatial_top(contact_f[i]), soft_normal[j])))


@wp.kernel
def _set_indexed(q: wp.array2d[float], indices: wp.array[wp.int32], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _assemble_frame_target(
    ik_q: wp.array2d[float],
    finger_indices: wp.array[wp.int32],
    finger_values: wp.array[float],
    q_start: wp.array[float],
    q_end: wp.array[float],
):
    """Shift the frame interval and write the IK arm solution plus finger commands."""
    i = wp.tid()
    q_start[i] = q_end[i]
    q_end[i] = ik_q[0, i]
    for k in range(finger_indices.shape[0]):
        if finger_indices[k] == i:
            q_end[i] = finger_values[k]


@wp.kernel
def _interpolate_robot(
    q_start: wp.array[float],
    q_end: wp.array[float],
    alpha: float,
    inv_frame_dt: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    """Interpolate the robot's one-DOF coordinates within a frame and set their velocity."""
    i = wp.tid()
    q[i] = q_start[i] * (1.0 - alpha) + q_end[i] * alpha
    qd[i] = (q_end[i] - q_start[i]) * inv_frame_dt


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 2 or args.substeps % 2:
            raise ValueError("--substeps must be even so the captured state swap returns to its start")
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._build_ik()
        self._build_grip_sensing()
        self._initialize_pose()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching="latest",
            rigid_contact_max=32768,
            soft_contact_max=16384,
            soft_contact_gap=0.003,
            include_static_kinematic_pairs=False,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.vbd_iterations,
            rigid_body_contact_buffer_size=2048,
            rigid_body_particle_contact_buffer_size=4096,
            rigid_contact_history=True,
            friction_epsilon=1.0e-4,
            particle_enable_self_contact=True,
            # 1 mm interaction distance plus 1.5 mm extra detection reach.
            particle_self_contact_margin=0.001,
            particle_self_contact_gap=0.0015,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.008,
            # The 7 g paper cup is light and stiff: vertex sweeps alone propagate
            # the table's support and the fingers' grip through its wall too
            # slowly, so it crumples or slides through the hand. The translation
            # correction moves each soft body as a whole within every iteration.
            particle_enable_translation_correction=True,
            # Paper creases where the grip or the load bends it past yield.
            particle_bending_yield_angle=PAPER_YIELD_ANGLE,
        )
        # Contact forces drive the force-limited grip; they are exported on each
        # frame's last substep and reduced per digit inside the frame graph.
        self.observables = self.solver.observables({SolverObservableFlags.CONTACT_F})

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.use_graph = self.device.is_cuda and not args.no_cuda_graph
        self.graph = None

    # Scene -------------------------------------------------------------------

    def _build_scene(self):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.default_shape_cfg.ke = 1.2e5
        builder.default_shape_cfg.kd = 10.0
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.margin = 0.0005
        builder.default_shape_cfg.gap = 0.002
        builder.default_shape_cfg.configure_sdf(force_sdf=True)

        self._add_robot(builder)
        # The robot-only model drives IK; its coordinates match the scene's.
        self.ik_model = builder.finalize()
        self._add_workstation(builder)
        self._add_scoop(builder)
        self._add_popcorn(builder)
        self._add_cup(builder)

        builder.color(include_bending=True)
        # VBD solves only the scoop and the kernels; the robot links are
        # kinematic. Kernels alternate by spawn-grid parity, so packed
        # neighbors start in different colors.
        grains = np.asarray(self.popcorn_bodies, dtype=np.int32)
        builder.body_color_groups = [
            np.asarray([self.scoop_body], dtype=np.int32),
            *(grains[self.popcorn_parity == color] for color in range(2)),
        ]
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_mu = 0.9
        self._load_prop_visuals()

    def _add_robot(self, builder):
        builder.add_urdf(
            str(ROBOT_URDF),
            xform=wp.transform(wp.vec3(ROBOT_X, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        for joint, label in enumerate(builder.joint_label[: self.robot_joint_end]):
            name = label.rsplit("/", 1)[-1]
            value = {**LEG_POSTURE, **ARM_POSTURE}.get(name)
            if value is not None:
                builder.joint_q[builder.joint_q_start[joint]] = value
            # Finger joints are commanded directly, so the solver's mimic
            # projection has nothing to do on these kinematic links.
            builder.joint_mimic_joint[joint] = -1
        self.robot_coords = builder.joint_coord_count
        if self.robot_coords != builder.joint_dof_count:
            raise RuntimeError("Expected one-DOF robot joints")
        # (coordinate, side, joint name without its side prefix) for every finger joint.
        self.fingers = []
        for joint, label in enumerate(builder.joint_label[: self.robot_joint_end]):
            name = label.rsplit("/", 1)[-1]
            for side, prefix in enumerate(("LEFT_", "RIGHT_")):
                if name.startswith(prefix) and ("HAND_" in name or "_PIP" in name):
                    self.fingers.append((builder.joint_q_start[joint], side, name[len(prefix) :]))

        # Only the hands touch objects, each link as one closed convex hull.
        mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        hand = [
            shape
            for shape in range(builder.shape_count)
            if builder.shape_flags[shape] & mask
            and any(word in builder.body_label[builder.shape_body[shape]].lower() for word in HAND_CONTACT_KEYWORDS)
        ]
        for shape in range(builder.shape_count):
            if shape not in hand:
                builder.shape_flags[shape] &= ~mask
        builder.approximate_meshes(method="convex_hull", shape_indices=hand, keep_visual_shapes=False)
        self.hand_shapes = [
            shape
            for shape in range(builder.shape_count)
            if builder.shape_body[shape] >= 0
            and builder.shape_body[shape] < self.robot_body_end
            and builder.shape_flags[shape] & mask
        ]
        for shape in self.hand_shapes:
            builder.shape_material_ke[shape] = HAND_CONTACT_KE
            builder.shape_material_mu[shape] = HAND_FRICTION

    def _add_workstation(self, builder):
        """Worktop and warmer collision geometry; the warmer renders from props.json."""
        tray_cfg = newton.ModelBuilder.ShapeConfig(ke=TRAY_CONTACT_KE, kd=0.2, mu=0.7, margin=0.0005)
        visual = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)

        def box(center, half, color, config=tray_cfg, label=""):
            return builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=config,
                color=color,
                label=label,
            )

        builder.add_ground_plane(color=(0.18, 0.20, 0.23))
        box(
            ((TABLE_FRONT + TABLE_BACK) / 2, 0.0, TABLE - 0.025),
            ((TABLE_BACK - TABLE_FRONT) / 2, TABLE_HALF_Y, 0.025),
            (0.80, 0.78, 0.71),
            label="worktop",
        )
        for x in (TABLE_FRONT + 0.07, TABLE_BACK - 0.07):
            for y in (-0.56, 0.56):
                box((x, y, (TABLE - 0.05) / 2), (0.023, 0.023, (TABLE - 0.05) / 2), (0.25, 0.28, 0.31), visual)

        # Warmer: riser, tray, rear baffle, side walls and glass, posts, back
        # panel, and canopy.
        # Coordinates are relative to the tray center MACHINE; the props.json
        # meshes draw the detailed cabinet over these collision envelopes.
        red, steel = (0.65, 0.055, 0.035), (0.64, 0.67, 0.70)
        self.machine_shapes_start = builder.shape_count
        x0, y0, top = MACHINE[0], MACHINE[1], MACHINE[2]
        box((x0, y0, (TABLE + top) / 2), (0.165, 0.18, (top - TABLE) / 2), red, label="machine_riser")
        box((x0, y0, top + 0.011), (0.165, 0.18, 0.011), red, label="machine_base")
        box((x0, y0, top + 0.025), (0.15, 0.16, 0.004), steel, label="popcorn_tray")
        box((x0 + 0.101, y0, top + 0.075), (0.004, 0.16, 0.045), steel, label="tray_rear_baffle")
        for side in (-1.0, 1.0):
            box((x0, y0 + side * 0.17, top + 0.07), (0.158, 0.007, 0.045), steel, label="tray_side_wall")
            box((x0, y0 + side * 0.17, top + 0.295), (0.15, 0.002, 0.18), steel, label="machine_side_glass")
            box((x0 + 0.15, y0 + side * 0.17, top + 0.26), (0.008, 0.011, 0.25), red, label="machine_post")
        box((x0 + 0.158, y0, top + 0.25), (0.006, 0.17, 0.22), steel, label="machine_back")
        box((x0, y0, top + 0.52), (0.176, 0.195, 0.035), red, label="machine_canopy")
        self.machine_shapes_end = builder.shape_count

    def _add_scoop(self, builder):
        """A dynamic scoop: phenolic grip, steel shank, and rolled aluminum bowl."""
        metal = newton.ModelBuilder.ShapeConfig(ke=TRAY_CONTACT_KE, kd=0.2, mu=0.7, margin=0.0005, density=2700)
        grip = newton.ModelBuilder.ShapeConfig(ke=4.0e4, kd=10.0, mu=1.2, margin=0.0005, density=350)
        self.scoop_body = builder.add_body(xform=wp.transform(wp.vec3(*SCOOP_REST), wp.quat_identity()), label="scoop")
        self.scoop_shapes_start = builder.shape_count
        builder.add_shape_cylinder(
            self.scoop_body,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
            radius=HANDLE_RADIUS,
            half_height=0.055,
            cfg=grip,
            label="scoop_grip",
        )
        builder.add_shape_cylinder(
            self.scoop_body,
            xform=wp.transform(wp.vec3(0.095, 0, -0.010), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -1.326)),
            radius=0.005,
            half_height=0.04123,
            cfg=metal,
            label="scoop_shank",
        )
        for index, panel in enumerate(scoop_bowl_panels()):
            # Full-surface cup contact samples each panel's SDF; resolve the
            # 1.2 mm sheet with about three voxels across its thickness.
            panel.build_sdf(target_voxel_size=0.0004, margin=0.0025)
            builder.add_shape_convex_hull(self.scoop_body, mesh=panel, cfg=metal, label=f"scoop_panel_{index}")
        self.scoop_shapes_end = builder.shape_count

    def _add_popcorn(self, builder):
        """Each kernel is a rigid cluster of four spheres: analytic contacts, one point per lobe pair."""
        # The lobes are collision geometry only; each kernel is drawn as one puffed surface.
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=GRAIN_CONTACT_KE,
            kd=0.02,
            mu=POPCORN_FRICTION,
            density=popcorn_lobe_density(),
            margin=0.0003,
            is_visible=False,
        )
        rng = np.random.default_rng(37)
        positions, self.popcorn_parity = popcorn_spawn_grid(self.args.popcorn_count)
        self.popcorn_bodies = []
        self.popcorn_colors = []
        for index, position in enumerate(positions):
            spin = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(rng.uniform(-math.pi, math.pi)))
            body = builder.add_body(xform=wp.transform(wp.vec3(*position), spin), label=f"popcorn_{index}")
            self.popcorn_colors.append((1.0, float(rng.uniform(0.86, 0.95)), float(rng.uniform(0.62, 0.78))))
            for center, radius in POPCORN_LOBES:
                builder.add_shape_sphere(
                    body, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), radius=radius, cfg=cfg
                )
            self.popcorn_bodies.append(body)

    def _add_cup(self, builder):
        points, faces = cup_mesh()
        with POPCORN_PROPS.open(encoding="utf-8") as stream:
            self.props = json.load(stream)
        cup_asset = next(asset for asset in self.props["meshes"] if asset["group"] == "cup")
        if not np.allclose(cup_asset["vertices"], points, atol=1e-7) or not np.array_equal(cup_asset["faces"], faces):
            raise ValueError("The props.json cup must match the simulated shell")
        self.cup_particle_start = builder.particle_count
        edge_start = len(builder.edge_indices)
        mu, lam = paper_membrane_lame()
        builder.add_cloth_mesh(
            pos=wp.vec3(*CUP),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=points.tolist(),
            indices=faces.reshape(-1).tolist(),
            density=PAPER_DENSITY,
            tri_ke=mu,
            tri_ka=lam,
            tri_kd=PAPER_MEMBRANE_DAMPING,
            edge_kd=PAPER_BENDING_DAMPING,
            particle_radius=0.001,
            label="paper_cup",
        )
        self.cup_particle_end = builder.particle_count
        cup_points = points + CUP.astype(np.float32)
        for e in range(edge_start, len(builder.edge_indices)):
            edge = np.asarray(builder.edge_indices[e]) - self.cup_particle_start
            if edge[0] >= 0 and edge[1] >= 0:
                builder.edge_bending_properties[e] = (paper_hinge_stiffness(cup_points, edge), PAPER_BENDING_DAMPING)
        self.cup_faces = faces
        # Open boundary (the rolled lip's inner edge) and flat base, for the grasp checks.
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.cup_rim = np.unique(unique[counts == 1])
        self.cup_base = np.flatnonzero(np.isclose(points[:, 2], -CUP_HEIGHT / 2))
        material = np.asarray(cup_asset["material_indices"])
        self.cup_parts = [faces[material == index].reshape(-1) for index in range(len(CUP_MATERIAL_COLORS))]

    def _load_prop_visuals(self):
        """Blender display meshes for the warmer and scoop; they never collide or add mass."""
        flags = self.model.shape_flags.numpy()
        for start, end in (
            (self.machine_shapes_start, self.machine_shapes_end),
            (self.scoop_shapes_start, self.scoop_shapes_end),
        ):
            flags[start:end] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags.assign(flags)
        device = self.model.device
        self.prop_render = []
        for asset in self.props["meshes"]:
            if asset["group"] == "cup" or not asset.get("display", True):
                continue
            vertices = np.asarray(asset["render_vertices"], dtype=np.float32)
            body = -1
            if asset["group"] == "machine":
                vertices = vertices + MACHINE.astype(np.float32)
            else:
                body = self.scoop_body
            mesh = newton.Mesh(
                vertices,
                np.arange(len(vertices), dtype=np.int32),
                normals=np.asarray(asset["render_normals"], dtype=np.float32),
            )
            self.prop_render.append(
                (
                    asset["name"],
                    body,
                    mesh,
                    wp.array([asset["color"]], dtype=wp.vec3, device=device),
                    wp.array([(asset["roughness"], asset["metallic"], 0, 0)], dtype=wp.vec4, device=device),
                    wp.array([asset["opacity"]], dtype=float, device=device),
                )
            )
        self.prop_identity = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.popcorn_mesh = popcorn_display_mesh()
        self.popcorn_color_array = wp.array(self.popcorn_colors, dtype=wp.vec3, device=device)
        self.popcorn_material = wp.array(
            [(0.75, 0.0, 0.0, 0.0)] * len(self.popcorn_bodies), dtype=wp.vec4, device=device
        )
        first = self.popcorn_bodies[0] if self.popcorn_bodies else 0
        if self.popcorn_bodies != list(range(first, first + len(self.popcorn_bodies))):
            raise ValueError("Popcorn bodies must be contiguous to draw them from one pose slice")
        self.popcorn_body_slice = slice(first, first + len(self.popcorn_bodies))
        self.cup_part_arrays = [wp.array(part, dtype=int, device=device) for part in self.cup_parts]
        self.cup_inside = wp.array(self.cup_faces[:, ::-1].reshape(-1).copy(), dtype=int, device=device)

    # Robot control -------------------------------------------------------------

    def _build_ik(self):
        """Two-arm IK for the cup and scoop grasps; every other coordinate is masked out."""
        model = self.ik_model
        device = model.device
        self.wrists = [
            next(i for i, label in enumerate(model.body_label) if label.endswith(f"/{side}_j7"))
            for side in ("left", "right")
        ]
        self.left_rotation, self.left_offset = left_wrist_pose()
        self.position_objectives, self.rotation_objectives = [], []
        for wrist in self.wrists:
            self.position_objectives.append(
                ik.IKObjectivePosition(wrist, TCP, wp.zeros(1, dtype=wp.vec3, device=device))
            )
            self.rotation_objectives.append(
                ik.IKObjectiveRotation(
                    wrist, wp.quat_identity(), wp.array([wp.vec4(0, 0, 0, 1)], dtype=wp.vec4, device=device)
                )
            )
        qd_start = model.joint_qd_start.numpy()
        mask = np.zeros(model.joint_dof_count, dtype=bool)
        for joint, label in enumerate(model.joint_label):
            if label.rsplit("/", 1)[-1] in (*LEFT_ARM, *RIGHT_ARM):
                mask[qd_start[joint] : qd_start[joint + 1]] = True
        limits = ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=30.0)
        self.ik_solver = ik.IKSolver(
            model,
            n_problems=1,
            objectives=[*self.position_objectives, *self.rotation_objectives, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            joint_dof_mask=wp.array(mask, dtype=wp.bool, device=device),
        )
        self.ik_q = wp.clone(model.joint_q).reshape((1, -1))
        locked = np.flatnonzero(~mask).astype(np.int32)
        self.lock_indices = wp.array(locked, dtype=wp.int32, device=device)
        self.lock_values = wp.array(model.joint_q.numpy()[locked], dtype=float, device=device)

        # Per-frame IK runs inside the captured frame graph; the host only
        # writes the sampled wrist targets and finger commands.
        self.wrist_targets = wp.zeros(2, dtype=wp.transform, device=device)
        self.finger_indices = wp.array([coord for coord, _, _ in self.fingers], dtype=wp.int32, device=device)
        self.finger_values = wp.zeros(len(self.fingers), dtype=float, device=device)
        self.finger_host = wp.zeros(len(self.fingers), dtype=float, device="cpu", pinned=device.is_cuda)
        # Until the scoop is observed, plan against its nominal resting pose.
        self.scoop_grip = wp.transform(wp.vec3(*SCOOP_REST), wp.quat_identity())
        self.scoop_observed = False
        # Wrist pose in the scoop frame and the held scoop pose, once calibrated.
        self.scoop_to_wrist = None
        self.scoop_held = None
        # Until the rim is observed, aim at its nominal position over the receiving spot.
        self.rim_center = RECEIVING_CUP + np.array((0.0, 0.0, 0.5 * CUP_HEIGHT))
        self.rim_observed = False
        # Kernels in the scoop as the pour begins, to judge the delivery.
        self.scoop_load = None

    def _build_grip_sensing(self):
        """Map hand shapes to digits and set up the grip controller state."""
        shape_body = self.model.shape_body.numpy()
        shape_digit = np.full(self.model.shape_count, -1, dtype=np.int32)
        for shape in self.hand_shapes:
            label = self.model.body_label[shape_body[shape]].lower()
            for side, prefix in enumerate(("/left_", "/right_")):
                for digit, name in enumerate(DIGITS):
                    if prefix + name in label:
                        shape_digit[shape] = 5 * side + digit
        device = self.model.device
        self.shape_digit = wp.array(shape_digit, dtype=wp.int32, device=device)
        self.digit_force = wp.zeros(10, dtype=float, device=device)
        self.grip_force = np.zeros((2, 5))
        self.grip_offset = np.zeros((2, 5))
        self.grips = [GuardedGrip(target) for target in GRIP_FORCE_TARGET]
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        self.finger_limits = np.array([(lower[coord], upper[coord]) for coord, _, _ in self.fingers])

    def _update_grip(self, closures):
        """Advance each hand's force-limited grip from the last frame's measured digit forces."""
        measured = self.digit_force.numpy().reshape(2, 5)
        self.grip_force += GRIP_FORCE_FILTER * (measured - self.grip_force)
        for side, grip in enumerate(self.grips):
            if closures[side] >= 0.999:
                self.grip_offset[side] = grip.update(self.grip_force[side])
            else:
                grip.reset()
                self.grip_offset[side] = 0.0

    def _observe_scoop(self):
        """Read the settled scoop pose once; the grasp target follows its grip and heading."""
        pose = wp.transform(*self.state_0.body_q.numpy()[self.scoop_body])
        self.scoop_grip = pose
        self.scoop_observed = True

    def _calibrate_scoop_in_hand(self):
        """Read the held scoop and wrist poses once; afterwards the arm steers the scoop itself."""
        body_q = self.state_0.body_q.numpy()
        scoop = wp.transform(*body_q[self.scoop_body])
        wrist = wp.transform(*body_q[self.wrists[1]])
        self.scoop_to_wrist = wp.transform_multiply(wp.transform_inverse(scoop), wrist)
        self.scoop_held = scoop

    @staticmethod
    def _scoop_pose_from_lip(lip, yaw, pitch):
        """Scoop pose that puts the front lip at ``lip`` with the given heading and pitch."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(yaw)) * wp.quat_from_axis_angle(
            wp.vec3(0, 1, 0), float(pitch)
        )
        return wp.transform(wp.vec3(*(np.asarray(lip) - np.asarray(wp.quat_rotate(rotation, SCOOP_LIP)))), rotation)

    def _pour_aim(self, yaw):
        """Lip aim point over the observed rim, upstream along the pour heading."""
        heading = np.array((math.cos(yaw), math.sin(yaw), 0.0))
        across = np.array((-math.sin(yaw), math.cos(yaw), 0.0))
        return self.rim_center - POUR_AIM_UPSTREAM * heading + POUR_AIM_ACROSS * across

    def _scoop_target(self, time):
        """Desired scoop pose: from the held pose along SCOOP_PATH, heading +x, then the pour."""
        if time > SCOOP_PATH[-1][0]:
            t0, grip, pitch0 = SCOOP_PATH[-1]
            lip0 = np.asarray(
                wp.transform_point(
                    wp.transform(wp.vec3(*grip), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch0)), SCOOP_LIP
                )
            )
            keys = [(t0, lip0, 0.0, pitch0), (POUR_SWING_START, lip0, 0.0, pitch0)]
            keys += [(t, self._pour_aim(yaw) + offset, yaw, pitch) for t, offset, yaw, pitch in POUR_PATH]
            for (ta, la, ya, pa), (tb, lb, yb, pb) in pairwise(keys):
                if time <= tb:
                    u = _smoothstep((time - ta) / (tb - ta))
                    return self._scoop_pose_from_lip(
                        (1.0 - u) * la + u * lb, (1.0 - u) * ya + u * yb, (1.0 - u) * pa + u * pb
                    )
            _, lip, yaw, pitch = keys[-1]
            return self._scoop_pose_from_lip(lip, yaw, pitch)
        keys = [(SCOOP_CALIBRATION_TIME, self.scoop_held)]
        for t, position, pitch in SCOOP_PATH:
            keys.append((t, wp.transform(wp.vec3(*position), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch))))
        for (t0, a), (t1, b) in pairwise(keys):
            if time <= t1:
                u = _smoothstep((time - t0) / (t1 - t0))
                pa, pb = np.asarray(wp.transform_get_translation(a)), np.asarray(wp.transform_get_translation(b))
                rotation = wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), u)
                return wp.transform(wp.vec3(*((1.0 - u) * pa + u * pb)), rotation)
        return keys[-1][1]

    def _write_frame_inputs(self, time):
        """Sample the keyframes and upload this frame's wrist targets and finger commands."""
        if not self.scoop_observed and time >= SCOOP_PERCEPTION_TIME:
            self._observe_scoop()
        if self.scoop_to_wrist is None and time >= SCOOP_CALIBRATION_TIME:
            self._calibrate_scoop_in_hand()
        if not self.rim_observed and time >= POUR_PERCEPTION_TIME:
            cup = self.state_0.particle_q.numpy()[self.cup_particle_start : self.cup_particle_end]
            self.rim_center = cup[self.cup_rim].mean(axis=0)
            self.rim_observed = True
            body_q = self.state_0.body_q.numpy()
            scoop = wp.transform(*body_q[self.scoop_body])
            self.scoop_load = int(self._in_scoop_bowl(body_q[self.popcorn_bodies, :3], scoop).sum())
        cup_offset, self.left_closure, scoop_offset, self.right_closure = sample_keyframes(time)
        left_tcp = CUP + cup_offset + self.left_offset
        if self.scoop_to_wrist is None:
            axis = wp.quat_rotate(wp.transform_get_rotation(self.scoop_grip), wp.vec3(1.0, 0.0, 0.0))
            right_rotation, right_offset = right_wrist_pose(math.atan2(float(axis[1]), float(axis[0])))
            right_tcp = np.asarray(wp.transform_get_translation(self.scoop_grip)) + scoop_offset + right_offset
        else:
            wrist = wp.transform_multiply(self._scoop_target(time), self.scoop_to_wrist)
            right_rotation = wp.transform_get_rotation(wrist)
            right_tcp = np.asarray(wp.transform_point(wrist, TCP))
        closures = (self.left_closure, self.right_closure)
        commands = (left_finger_command, right_finger_command)
        self._update_grip(closures)
        values = []
        for (_, side, name), (lower, upper) in zip(self.fingers, self.finger_limits, strict=True):
            digit, share = grip_offset_share(name)
            value = commands[side](name, closures[side])
            if digit >= 0:
                value += share * self.grip_offset[side, digit]
            values.append(min(max(value, lower), upper))
        self.finger_host.numpy()[:] = values
        wp.copy(self.finger_values, self.finger_host)
        wp.launch(
            _write_wrist_targets,
            1,
            [
                wp.transform(wp.vec3(*left_tcp), self.left_rotation),
                wp.transform(wp.vec3(*right_tcp), right_rotation),
                self.wrist_targets,
            ],
        )

    def _solve_ik(self, iterations):
        wp.launch(
            _unpack_wrist_targets,
            1,
            [
                self.wrist_targets,
                self.position_objectives[0].target_positions,
                self.rotation_objectives[0].target_rotations,
                self.position_objectives[1].target_positions,
                self.rotation_objectives[1].target_rotations,
            ],
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        wp.launch(_set_indexed, self.lock_indices.shape[0], [self.ik_q, self.lock_indices, self.lock_values])

    def _initialize_pose(self):
        """Solve the approach pose before the first frame."""
        self.frame_q_start = wp.clone(self.model.joint_q[: self.robot_coords])
        self.frame_q_end = wp.clone(self.frame_q_start)
        self._write_frame_inputs(0.0)
        self._solve_ik(INITIAL_IK_ITERATIONS)
        wp.launch(
            _assemble_frame_target,
            self.robot_coords,
            [self.ik_q, self.finger_indices, self.finger_values, self.frame_q_start, self.frame_q_end],
        )
        wp.copy(self.frame_q_start, self.frame_q_end)
        wp.copy(self.model.joint_q, self.frame_q_end, count=self.robot_coords)
        for state in (self.state_0, self.state_1):
            state.joint_q.assign(self.model.joint_q)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    # Simulation ----------------------------------------------------------------

    def _simulate(self):
        self._solve_ik(IK_ITERATIONS)
        wp.launch(
            _assemble_frame_target,
            self.robot_coords,
            [self.ik_q, self.finger_indices, self.finger_values, self.frame_q_start, self.frame_q_end],
        )
        for substep in range(self.sim_substeps):
            wp.launch(
                _interpolate_robot,
                self.robot_coords,
                [
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
            last = substep == self.sim_substeps - 1
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
                observables=self.observables if last else None,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.digit_force.zero_()
        contacts = self.contacts
        wp.launch(
            _accumulate_digit_forces,
            contacts.rigid_contact_max + contacts.soft_contact_max,
            [
                self.observables.contact_f,
                contacts.rigid_contact_count,
                contacts.rigid_contact_max,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_normal,
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                contacts.soft_contact_normal,
                self.model.shape_body,
                self.shape_digit,
                self.scoop_body,
                self.digit_force,
            ],
        )

    def step(self):
        self._write_frame_inputs(self.sim_time + self.frame_dt)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            # The first frame runs uncaptured so lazily sized buffers exist;
            # the capture records IK, the frame target, and every substep.
            self._simulate()
            if self.use_graph:
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.graph = capture.graph
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, body, mesh, colors, materials, opacities in self.prop_render:
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
        if self.popcorn_bodies:
            self.viewer.log_shapes(
                "/popcorn",
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                self.state_0.body_q[self.popcorn_body_slice],
                colors=self.popcorn_color_array,
                materials=self.popcorn_material,
                geo_src=self.popcorn_mesh,
            )
        # Print only the exterior; the reversed winding draws the plain inside.
        for name, indices, color in zip(
            ("paper", "print", "lap_seam"), self.cup_part_arrays, CUP_MATERIAL_COLORS, strict=True
        ):
            self.viewer.log_mesh(
                f"/cup/{name}", self.state_0.particle_q, indices, color=color, roughness=0.88, metallic=0.0
            )
        self.viewer.log_mesh(
            "/cup/inside",
            self.state_0.particle_q,
            self.cup_inside,
            color=CUP_MATERIAL_COLORS[0],
            roughness=0.88,
            metallic=0.0,
        )
        self.viewer.end_frame()

    # Checks ------------------------------------------------------------------

    def _in_cup(self, points, cup):
        """Points inside the deformed paper shell, capped at its rim for measurement only.

        Counts crossings of a fixed ray with the shell and a virtual rim cap (odd = inside); the
        cap is never part of the collision or render model.
        """
        points = np.asarray(points, dtype=np.float64)
        inside = np.zeros(len(points), dtype=bool)
        candidates = np.all((points >= cup.min(axis=0)) & (points <= cup.max(axis=0)), axis=1)
        if not np.any(candidates):
            return inside
        rim = cup[self.cup_rim]
        rim = rim[np.argsort(np.arctan2(rim[:, 1] - rim[:, 1].mean(), rim[:, 0] - rim[:, 0].mean()))]
        cap = np.stack((rim, np.roll(rim, -1, axis=0), np.broadcast_to(rim.mean(axis=0), rim.shape)), axis=1)
        triangles = np.concatenate((cup[self.cup_faces], cap)).astype(np.float64)
        direction = np.array((0.8713, 0.3371, 0.3579))
        edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        p = np.cross(direction, edge2)
        det = np.einsum("ij,ij->i", edge1, p)
        valid = np.abs(det) > 1.0e-12
        inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
        offset = points[candidates, None, :] - triangles[None, :, 0]
        u = np.einsum("nfi,fi->nf", offset, p) * inverse
        q = np.cross(offset, edge1)
        v = np.einsum("nfi,i->nf", q, direction) * inverse
        t = np.einsum("nfi,fi->nf", q, edge2) * inverse
        hits = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1.0e-9)
        inside[candidates] = np.count_nonzero(hits, axis=1) % 2 == 1
        return inside

    @staticmethod
    def _in_scoop_bowl(points, scoop):
        """Kernel centers inside the scoop's rolled trough, in the scoop frame."""
        local = np.array([np.asarray(wp.transform_point(wp.transform_inverse(scoop), wp.vec3(*p))) for p in points])
        angles = np.linspace(-1.30, 1.30, 13)
        wall_y = 0.056 * np.sin(angles)
        wall_z = SCOOP_FLOOR_Z + 0.056 * (1.0 - np.cos(angles))
        bottom = np.interp(local[:, 1], wall_y, wall_z)
        return (
            (local[:, 0] > BOWL_REAR)
            & (local[:, 0] < BOWL_FRONT)
            & (np.abs(local[:, 1]) < wall_y[-1])
            & (local[:, 2] > bottom - 0.002)
            & (local[:, 2] < wall_z[-1] + 0.02)
        )

    def test_final(self):
        bodies = self.state_0.body_q.numpy()
        particles = self.state_0.particle_q.numpy()
        if not (np.isfinite(bodies).all() and np.isfinite(particles).all()):
            raise ValueError("The popcorn scene contains a non-finite state")
        grains = bodies[self.popcorn_bodies, :3]
        rel = grains - MACHINE
        in_tray = (np.abs(rel[:, 0]) < 0.16) & (np.abs(rel[:, 1]) < 0.17) & (rel[:, 2] > 0.0)
        cup = particles[self.cup_particle_start : self.cup_particle_end]
        in_bowl = self._in_scoop_bowl(grains, wp.transform(*bodies[self.scoop_body]))
        in_cup = self._in_cup(grains, cup)
        lost = int(np.count_nonzero(~in_tray & ~in_bowl & ~in_cup))
        scoop_height = float(bodies[self.scoop_body, 2])
        if scoop_height < TABLE:
            raise ValueError(f"The scoop fell through the worktop: {bodies[self.scoop_body, :3]}")

        rim, base = self.cup_rim, self.cup_base
        axis = cup[rim].mean(axis=0) - cup[base].mean(axis=0)
        upright = float(axis[2] / np.linalg.norm(axis))
        rim_radius = float(
            np.linalg.svd(cup[rim] - cup[rim].mean(axis=0), compute_uv=False)[1] * math.sqrt(2 / len(rim))
        )
        placement = float(np.linalg.norm(cup.mean(axis=0)[:2] - RECEIVING_CUP[:2]))
        cup_speed = float(
            np.linalg.norm(
                self.state_0.particle_qd.numpy()[self.cup_particle_start : self.cup_particle_end], axis=1
            ).max()
        )
        if self.sim_time >= POUR_PATH[-2][0]:
            if placement > 0.02:
                raise ValueError(f"The cup is not held at the receiving spot: {placement * 1000:.1f} mm away")
            clearance = float(cup[:, 2].min() - TABLE)
            if clearance < 0.01:
                raise ValueError(
                    f"The left hand is not holding the cup up: base {clearance * 1000:.1f} mm over the worktop"
                )
            # After the final hold the held cup and its load must be at rest.
            if cup_speed > 0.05:
                raise ValueError(f"The cup is not at rest after the task: paper moving at {cup_speed:.2f} m/s")
            if upright < 0.94 or rim_radius < 0.028:
                raise ValueError(
                    f"The grasp tipped or crushed the cup: upright {upright:.3f}, rim {rim_radius * 1000:.1f} mm"
                )
            # Only the right hand's grip can raise the scoop off the worktop.
            if scoop_height < SCOOP_REST[2] + 0.08:
                raise ValueError(f"The right hand did not lift the scoop: grip at {scoop_height:.3f} m")
            # Pouring from the 108 mm scoop into the 66 mm cup spills a few kernels
            # off the rim; most of the load must arrive.
            delivered = int(in_cup.sum())
            if delivered < 15 or delivered < 0.6 * self.scoop_load:
                raise ValueError(f"Only {delivered} of the {self.scoop_load} scooped kernels reached the cup")
        print(
            f"[W1Popcorn] PASS: {int(in_cup.sum())} of {self.scoop_load} scooped kernels poured into the cup (upright {upright:.3f}, "
            f"rim radius {rim_radius * 1000:.1f} mm, at rest at {cup_speed * 1000:.1f} mm/s); "
            f"{int(in_bowl.sum())} left in the scoop, {int(in_tray.sum())} in the tray, {lost} lost"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=int(KEYFRAMES[-1][0] * FPS))
        parser.add_argument("--popcorn-count", type=int, default=DEFAULT_POPCORN_COUNT)
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        parser.add_argument("--no-cuda-graph", action="store_true", help="Run frames without CUDA graph capture.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
