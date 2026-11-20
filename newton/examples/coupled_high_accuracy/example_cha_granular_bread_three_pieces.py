# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""A dynamic W1 robot cuts both ends off a cohesive granular loaf, leaving three pieces.

Run: .venv/bin/python -m newton.examples cha_granular_bread_three_pieces
Headless: .venv/bin/python -m newton.examples cha_granular_bread_three_pieces --viewer null --test

The scene, robot, knife, board and motion timeline follow Newton's MJMPM example
``example_mjmpm_w1_bread_three_pieces.py``. The W1 robot comes from ``assets/w1`` and the
board/knife visuals from ``assets/bread_cut``, both resolved relative to this repository.
The material model differs: the loaf is a Style3D Sim granular body whose bonds break
irreversibly under tension when the blade wedges the material apart. The knife is
mounted rigidly on the right wrist, and the whole arm is integrated by the same
solver, so material loads on the blade feed back into the PD-driven joints.
Because bonds only break in tension, the collision blade is thickened and beveled
to the grain resolution (see ``BLADE_MIN_HALF_WIDTH``); the rendered knife is unchanged.
"""

import json
import math
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ik
from newton._src.solvers.coupled_high_accuracy.particle_surface import connected_components
from newton.solvers import coupled_high_accuracy as cha

# Resolved from the repository location so the example runs from any working directory.
# Local link to the engineRTS asset directory (see README.md).
REPO_ASSETS = Path(__file__).resolve().parent / "assets"

FRAME_DT = 1.0 / 60.0
TABLE_Z = 0.86
BOARD_TOP = TABLE_Z + 0.018
LOAF_CENTER = np.array((0.54, -0.24, BOARD_TOP))
CUT_Y = -0.275
SECOND_CUT_Y = -0.205
# Three-piece variant: the handle is moved forward so the stroke uses the straight blade segment.
HANDLE = np.array((0.39 + 0.0225, -0.43, BOARD_TOP + 0.13))
LEFT_PARK = np.array((0.40, 0.24, TABLE_Z + 0.18))
SPACING = 0.005
BREAD_DENSITY = 280.0
# Minimum half-thickness of the collision blade. Bonds only break in tension, so the blade must wedge the
# grain rows on both sides of the cut apart by more than the break strain. The reference's 1 mm edge is far
# thinner than a 5 mm grain and would slide between rows with ~5 % bond strain. Half a spacing pushes the
# rows ~9 mm apart, which also breaks the bottom-row diagonal bonds after sawing shears them along the blade.
BLADE_MIN_HALF_WIDTH = 0.5 * SPACING
# Height of the V-shaped edge, kept inside one grain radius so the lowest row is wedged aside at full depth.
BLADE_BEVEL = 0.3 * SPACING
# The reference stroke stops its edge 2 mm above the board, level with the centres of the settled bottom grain
# row, which a wedge then cannot push aside. The collision edge reaches 2 mm lower, to the board surface; the
# rendered knife keeps the reference edge, and the blade does not collide with the board.
BLADE_EDGE_DROP = 0.002
TOTAL_FRAMES = 1200

TCP = wp.vec3(0.0, 0.0, 0.125)
FINGER_OPEN = 0.045
HANDLE_HALF_WIDTH = 0.009
# Includes the W1 visual pad overhang, flush rivets and drive tracking margin.
HANDLE_GRIP = HANDLE_HALF_WIDTH + 0.0018
# Gripper +Z points forward; its -X (the camera side) points upward.
HORIZONTAL_GRIPPER = wp.vec4(0.0, float(np.sqrt(0.5)), 0.0, float(np.sqrt(0.5)))
SIDES = ("left", "right")

# Stroke shape shared by both cuts, from the reference three-piece example.
CUT_DEPTH = 0.073
CUT_SEPARATION = 0.010
SAW_FORWARD = 0.008
SAW_BACKWARD = -0.005
WITHDRAW_HEIGHT = 0.133


BLADE_OUTLINE = (
    # (x, z, half_width) of the reference blade; the first six points form the cutting edge and curved tip.
    (0.060, -0.055, 0.0005),
    (0.190, -0.055, 0.0005),
    (0.205, -0.052, 0.0005),
    (0.218, -0.045, 0.0005),
    (0.230, -0.032, 0.0005),
    (0.240, -0.015, 0.0005),
    (0.225, 0.005, 0.0045),
    (0.200, 0.014, 0.0060),
    (0.060, 0.015, 0.0060),
)
CUTTING_EDGE = 6


def blade_mesh(min_half_width=0.0, bevel=0.0, edge_drop=0.0):
    """Closed convex wedge with the reference outline, curved tip and 12 mm spine.

    ``min_half_width`` thickens the thin edge region to the grain resolution. The edge keeps its reference
    position: its flanks reach ``min_half_width`` only ``bevel`` inside the outline, so the V-shaped edge
    wedges grains sideways instead of pressing the bottom row onto the board with a flat underside.
    ``edge_drop`` lowers the cutting edge and tip below the reference outline.
    """
    from scipy.spatial import ConvexHull

    outline = np.array([(x, z - (edge_drop if i < CUTTING_EDGE else 0.0)) for i, (x, z, _) in enumerate(BLADE_OUTLINE)])
    centroid = outline.mean(axis=0)
    points = []
    for i, (_, _, width) in enumerate(BLADE_OUTLINE):
        x, z = outline[i]
        width = max(width, min_half_width)
        if i < CUTTING_EDGE and bevel > 0.0:
            points.append((x, 0.0, z))
            # Move the flanks inward along the outline's vertex normal, approximated toward the centroid.
            inward = centroid - outline[i]
            x, z = outline[i] + bevel * inward / np.linalg.norm(inward)
        points.extend(((x, -width, z), (x, width, z)))
    vertices = np.array(points, dtype=np.float32)
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    # Orient every triangle outward; Qhull does not guarantee a consistent winding.
    center = vertices.mean(axis=0)
    for face in faces:
        a, b, c = vertices[face]
        if np.dot(np.cross(b - a, c - a), a - center) < 0.0:
            face[[1, 2]] = face[[2, 1]]
    return vertices, faces.astype(np.int32)


def loaf_contains(points):
    """Rounded loaf with a flat underside, in loaf-local metres."""
    x, y, z = np.asarray(points).T
    end = np.sqrt(np.clip(1.0 - (np.maximum(np.abs(y) - 0.060, 0.0) / 0.040) ** 2, 0.0, 1.0))
    return (z >= 0.002) & ((x / 0.050) ** 2 + ((z - 0.027) / 0.038) ** 2 < end**2)


def loaf_particles():
    local = np.stack(
        np.meshgrid(
            np.arange(-0.0475, 0.05, SPACING),
            np.arange(-0.0925, 0.095, SPACING),
            np.arange(0.0045, 0.066, SPACING),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return (local[loaf_contains(local)] + LOAF_CENTER).astype(np.float32)


def bread_colors(rest):
    colors = np.tile((0.89, 0.78, 0.55), (len(rest), 1))
    local = rest - LOAF_CENTER
    radial = np.sqrt((local[:, 0] / 0.050) ** 2 + ((local[:, 2] - 0.027) / 0.038) ** 2)
    # A crust several samples thick avoids alternating pale lattice stripes; colour never refers to the cuts.
    outside = (radial > 0.72) | (np.abs(local[:, 1]) > 0.079) | (local[:, 2] < 0.012)
    colors[outside] = (0.49, 0.24, 0.07)
    return colors * np.random.default_rng(57).uniform(0.94, 1.04, (len(colors), 1))


def keyframes():
    """Return ``(time, tcp_targets[2, 3], finger_openings[2])`` keyframes of both strokes."""
    home = np.array((LEFT_PARK, HANDLE))
    open_hands = (FINGER_OPEN, FINGER_OPEN)
    holding = (FINGER_OPEN, HANDLE_GRIP)
    release = np.array((0.0, 0.004, 0.0))
    rise = np.array((0.0, 0.0, WITHDRAW_HEIGHT))

    def stroke(cut_y):
        above = HANDLE.copy()
        above[1] = cut_y
        cut = above - np.array((0.0, 0.0, CUT_DEPTH))
        # A small sawing stroke keeps the blade from only compressing the loaf.
        return above, cut + (SAW_FORWARD, 0.0, 0.035), cut + (SAW_BACKWARD, 0.0, 0.015), cut

    def pose(right):
        return np.array((LEFT_PARK, right))

    above, saw_in, saw_out, cut = stroke(CUT_Y)
    # The first cut pushes the end slice outward (-Y); the second pushes the other end outward (+Y).
    separated = cut - (0.0, CUT_SEPARATION, 0.0)
    above_2, saw_in_2, saw_out_2, cut_2 = stroke(SECOND_CUT_Y)
    separated_2 = cut_2 + (0.0, CUT_SEPARATION, 0.0)
    return [
        (0.0, home, open_hands),
        (1.3, home, open_hands),
        (2.2, home, holding),
        (2.7, pose(above), holding),
        (4.5, pose(saw_in), holding),
        (5.7, pose(saw_out), holding),
        (6.8, pose(cut), holding),
        (7.2, pose(cut), holding),
        (8.2, pose(separated), holding),
        # Withdraw into the opened kerf instead of dragging the pushed slice.
        (8.6, pose(separated + release + (0.0, 0.0, 0.030)), holding),
        (9.6, pose(separated + release + rise), holding),
        (10.3, pose(separated + release + rise), holding),
        (11.6, pose(above_2), holding),
        (13.4, pose(saw_in_2), holding),
        (14.6, pose(saw_out_2), holding),
        (15.7, pose(cut_2), holding),
        (16.1, pose(cut_2), holding),
        (17.1, pose(separated_2), holding),
        (17.5, pose(separated_2 - release + (0.0, 0.0, 0.030)), holding),
        (18.5, pose(separated_2 - release + rise), holding),
        (20.0, pose(separated_2 - release + rise), holding),
    ]


def sample_keyframes(frames, time):
    """Interpolate keyframes with a quintic ease, holding the last value."""
    for a, b in pairwise(frames):
        if time <= b[0]:
            s = float(np.clip((time - a[0]) / (b[0] - a[0]), 0.0, 1.0))
            s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
            return tuple((1.0 - s) * np.asarray(va) + s * np.asarray(vb) for va, vb in zip(a[1:], b[1:], strict=True))
    return tuple(np.asarray(v) for v in frames[-1][1:])


@wp.kernel
def interpolate_joint_targets(
    start: wp.array[float],
    end: wp.array[float],
    fraction: float,
    inv_frame_dt: float,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    # The fixed-base W1 has only scalar joints, so coordinate and DOF indices coincide.
    i = wp.tid()
    target_q[i] = wp.lerp(start[i], end[i], fraction)
    target_qd[i] = (end[i] - start[i]) * inv_frame_dt


@wp.kernel
def compensate_gravity(body_mass: wp.array[float], gravity: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    # Like the reference's MuJoCo gravcomp: link weight is cancelled passively so drives only correct tracking error.
    i = wp.tid()
    body_f[i] = body_f[i] + wp.spatial_vector(-body_mass[i] * gravity, wp.vec3(0.0))


@wp.kernel
def observe_fracture(
    positions: wp.array[wp.vec3],
    pairs: wp.array[wp.vec2i],
    rest: wp.array[float],
    break_strain: float,
    broken: wp.array[int],
):
    # Read-only diagnostics mirror accepted-step fracture with plastic compaction disabled.
    i = wp.tid()
    pair = pairs[i]
    if wp.length(positions[pair[0]] - positions[pair[1]]) > rest[i] * (1.0 + break_strain):
        broken[i] = 1


def regularize_planar_inertia(model, relative=0.02):
    """Thicken planar link inertias so every body has a nonsingular second moment.

    Style3D Sim's affine body proxies use the second moment ``tr(I) / 2 - I``. Several W1 URDF links have
    principal moments with ``I3 = I1 + I2`` (an infinitely thin mass sheet), which makes it singular. Adding
    ``delta`` to every principal moment raises each second-moment eigenvalue by ``delta / 2``.
    """
    inertia = model.body_inertia.numpy()
    changed = 0
    for i, tensor in enumerate(inertia):
        moments = np.linalg.eigvalsh(tensor)
        second = 0.5 * moments.sum() - moments
        if moments.max() > 0.0 and second.min() < relative * second.max():
            inertia[i] = tensor + np.eye(3) * 2.0 * relative * second.max()
            changed += 1
    if changed:
        model.body_inertia.assign(inertia)
        model.body_inv_inertia.assign(np.linalg.inv(inertia))
    return changed


def count_detached_samples(positions, labels, spacing):
    """Samples outside the largest spatially connected patch of each piece."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as graph_components
    from scipy.spatial import cKDTree

    detached = 0
    for label in np.unique(labels):
        points = positions[labels == label]
        pairs = cKDTree(points).query_pairs(1.8 * spacing, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        _, patch = graph_components(graph, directed=False)
        detached += len(points) - int(np.bincount(patch).max())
    return detached


class AuthoredVisuals:
    """Draw the reference's authored board and knife meshes; the loaf itself is drawn from particles."""

    def __init__(self, viewer, assets, device, tool_body, tool_local):
        from PIL import Image

        self.viewer, self.tool_body, self.tool_local = viewer, tool_body, tool_local
        path = assets / "bread_cut"
        self.parts = []
        with np.load(path / "visuals.npz", allow_pickle=False) as archive:
            manifest = [part for part in json.loads(str(archive["manifest"])) if part["parent"] != "bread"]
            for part in manifest:
                key, texture = part["key"], part["texture"]
                self.parts.append(
                    dict(
                        name="/bread_cut/" + key,
                        parent=part["parent"],
                        points=wp.array(archive[key + "_points"], dtype=wp.vec3, device=device),
                        indices=wp.array(archive[key + "_indices"].astype(np.int32), dtype=int, device=device),
                        normals=wp.array(archive[key + "_normals"].astype(np.float32), dtype=wp.vec3, device=device),
                        uvs=wp.array(archive[key + "_uvs"], dtype=wp.vec2, device=device),
                        texture=np.array(Image.open(path / texture).convert("RGB")) if texture else None,
                        color=wp.full(
                            1, wp.vec3(*((1.0, 1.0, 1.0) if texture else part["color"])), dtype=wp.vec3, device=device
                        ),
                        material=wp.full(
                            1,
                            wp.vec4(part["roughness"], part["metallic"], 0.0, float(bool(texture))),
                            dtype=wp.vec4,
                            device=device,
                        ),
                        roughness=part["roughness"],
                        metallic=part["metallic"],
                        tint=tuple(part["color"]),
                    )
                )
        self.scale = wp.full(1, wp.vec3(1.0), dtype=wp.vec3, device=device)
        self.board_pose = wp.array(
            [wp.transform(wp.vec3(*LOAF_CENTER), wp.quat_identity())], dtype=wp.transform, device=device
        )
        self.tool_pose = wp.empty(1, dtype=wp.transform, device=device)
        self.logged = False

    def render(self, state):
        pose = wp.mul(wp.transform(*state.body_q.numpy()[self.tool_body]), self.tool_local)
        self.tool_pose.assign([pose])
        for part in self.parts:
            if not self.logged:
                self.viewer.log_mesh(
                    part["name"],
                    part["points"],
                    part["indices"],
                    normals=part["normals"],
                    uvs=part["uvs"],
                    texture=part["texture"],
                    color=part["tint"],
                    roughness=part["roughness"],
                    metallic=part["metallic"],
                    backface_culling=False,
                    hidden=True,
                )
            if not self.logged or part["parent"] == "knife":
                self.viewer.log_instances(
                    part["name"] + "/instance",
                    part["name"],
                    self.tool_pose if part["parent"] == "knife" else self.board_pose,
                    self.scale,
                    part["color"],
                    part["material"],
                )
        self.logged = True


class Example:
    camera_pos = wp.vec3(0.90, -0.50, 1.18)
    camera_pitch = -30.5
    camera_yaw = 144.16

    def __init__(self, viewer, args):
        if args.substeps < 1 or args.iterations < 1 or args.linear_iterations < 1:
            raise ValueError("substeps and solver iterations must be positive")
        for name in ("cohesion", "break_strain", "contact_stiffness"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise RuntimeError("The granular bread example requires a CUDA device (--device cuda:0).")
        self.viewer, self.args = viewer, args
        assets = Path(args.assets)
        urdf = assets / "w1/robot/w1-030/robot.urdf"
        if not urdf.is_file():
            raise FileNotFoundError(f"W1 robot not found at {urdf}; pass --assets <engineRTS>/assets")
        self.frame, self.sim_time = 0, 0.0
        self.frame_dt, self.sim_substeps = FRAME_DT, args.substeps
        self.sim_dt = FRAME_DT / args.substeps
        self.keyframes = keyframes()

        # IK keeps the URDF's fixed links so the gripper-base end effectors exist. The simulated robot
        # collapses them into their parents; movable coordinates and shape poses are unchanged.
        ik_builder = newton.ModelBuilder()
        ik_urdf = ET.parse(urdf)
        for link in ik_urdf.getroot().findall("link"):
            for shape in [*link.findall("visual"), *link.findall("collision")]:
                link.remove(shape)
        ik_builder.add_urdf(
            ET.tostring(ik_urdf.getroot(), encoding="unicode"),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.002
        builder.add_urdf(str(urdf), floating=False, enable_self_collisions=False, collapse_fixed_joints=True)
        self.robot_joints, self.robot_bodies = builder.joint_count, builder.body_count
        self.coords = self._movable_coordinates(builder)
        if self._movable_coordinates(ik_builder) != self.coords:
            raise RuntimeError("Collapsing fixed W1 joints changed the joint-coordinate layout")
        if builder.joint_coord_count != builder.joint_dof_count:
            raise RuntimeError("The W1 joint-target interpolation assumes scalar joints only")
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
        robot_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=1.0)
        for shape in range(builder.shape_count):
            builder.shape_material_mu[shape] = robot_cfg.mu
            builder.shape_material_ke[shape] = robot_cfg.ke
            builder.shape_material_kd[shape] = robot_cfg.kd

        self.ik_model = ik_builder.finalize(device=device)
        self._map_tcp_to_scene(builder)
        self._build_ik(device)
        self._solve_ik(*sample_keyframes(self.keyframes, 0.0), iterations=150)
        builder.joint_q[:] = self.ik_q.numpy()[0].tolist()

        surface_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.8)
        builder.add_ground_plane(cfg=surface_cfg)
        self._add_box(
            builder, "worktop", (0.46, 0.0, TABLE_Z - 0.025), (0.30, 0.82, 0.025), (0.57, 0.48, 0.36), surface_cfg
        )
        for x in (0.21, 0.71):
            for y in (-0.74, 0.74):
                half_height = (TABLE_Z - 0.05) / 2
                self._add_box(
                    builder,
                    "table_leg",
                    (x, y, half_height),
                    (0.025, 0.025, half_height),
                    (0.22, 0.25, 0.29),
                    surface_cfg,
                )
        visible = args.visual_style == "particles"
        board_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.8, is_visible=visible)
        self.board_shape = self._add_box(
            builder,
            "chopping_board",
            (0.54, -0.24, TABLE_Z + 0.009),
            (0.16, 0.16, 0.009),
            (0.45, 0.27, 0.12),
            board_cfg,
        )
        self._add_knife(builder, visible)

        self.rest = loaf_particles()
        self.bread_count = len(self.rest)
        self.radius = 0.43 * SPACING
        first_particle = builder.particle_count
        builder.add_particles(
            pos=self.rest,
            vel=np.zeros_like(self.rest),
            mass=np.full(self.bread_count, BREAD_DENSITY * SPACING**3),
            radius=np.full(self.bread_count, self.radius),
        )
        self.model = builder.finalize(device=device)
        regularize_planar_inertia(self.model)
        # The solver builds its body proxies from the model's body poses, so they must match the initial joint pose.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self._configure_drives()
        self.model.soft_contact_ke, self.model.soft_contact_kd = args.contact_stiffness, 2.0
        self.model.soft_contact_kf, self.model.soft_contact_mu = 1.0, args.board_friction
        cha.configure_granular(
            self.model,
            np.arange(first_particle, first_particle + self.bread_count, dtype=np.int32),
            contact_stiffness=args.grain_stiffness,
            friction=0.4,
            friction_damping=0.5,
            cohesion=args.cohesion,
            bond_damping=args.bond_damping,
            max_compaction=0.0,
            break_strain=args.break_strain,
            bond_radius=1.5 * SPACING,
            adhesion=0.0,
        )
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.joint_constraint_primal_iterations = 2
        self.solver.config.nonlinear_iterations = args.iterations
        self.solver.config.linear_iterations = args.linear_iterations
        self.solver.config.enable_collision_hessian_non_diagonal = True

        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        # Affine rigid coordinates appended by the adapter are not visible bread particles.
        self.bread_q = self.state_0.particle_q[first_particle : first_particle + self.bread_count]
        self.bread_qd = self.state_0.particle_qd[first_particle : first_particle + self.bread_count]
        self.first_particle = first_particle

        self.frame_end = wp.clone(self.ik_q.flatten())
        self.frame_start = wp.clone(self.frame_end)
        gravity = self.model.gravity.numpy() if hasattr(self.model.gravity, "numpy") else self.model.gravity
        self.gravity = wp.vec3(*np.asarray(gravity, dtype=float).reshape(-1, 3)[0])
        self.bonds = self.model.synreal_granular_bond_indices
        self.broken = wp.zeros(len(self.bonds), dtype=int, device=device)
        self.bond_pairs = self.bonds.numpy() - first_particle
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_max=8 * self.bread_count)
        self.contacts = self.collision_pipeline.contacts()

        self.colors = wp.array(bread_colors(self.rest), dtype=wp.vec3, device=device)
        self.visuals = None
        if args.visual_style == "realistic":
            self.visuals = AuthoredVisuals(viewer, assets, device, self.knife, self.knife_local)
        viewer.set_model(self.model)
        viewer.show_particles = False
        viewer.set_camera(pos=self.camera_pos, pitch=self.camera_pitch, yaw=self.camera_yaw)

        self.piece_history, self.observed_piece_counts, self.peak_speed = [], set(), 0.0
        self.stream, self.graph = wp.get_stream(device), None
        if not args.no_capture:
            # Capture one frame of substeps; host IK only refreshes the frame endpoint buffers between launches.
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph
        print(
            f"[GranularBread] {self.bread_count:,} grains, {len(self.bonds):,} bonds; "
            f"{self.model.body_count} bodies; substeps={self.sim_substeps}",
            flush=True,
        )

    # Scene construction ------------------------------------------------------

    @staticmethod
    def _movable_coordinates(builder):
        return {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[j]
            for j, name in enumerate(builder.joint_label)
            if builder.joint_type[j] != newton.JointType.FIXED
        }

    @staticmethod
    def _add_box(builder, label, position, half, color, cfg):
        return builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            hx=half[0],
            hy=half[1],
            hz=half[2],
            cfg=cfg,
            color=color,
            label=label,
        )

    def _map_tcp_to_scene(self, scene):
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
            self.scene_ee.append(scene.body_label.index(model.body_label[body]))
            tcp_world = wp.transform_point(wp.transform(*body_q[gripper]), TCP)
            self.scene_tcp.append(wp.transform_point(wp.transform_inverse(wp.transform(*body_q[body])), tcp_world))
        self._ik_body_q = body_q

    def _add_knife(self, builder, visible):
        """Mount the knife rigidly on the right wrist link so its inertia and contact loads act on the arm."""
        self.knife = self.scene_ee[1]
        poses = self._ik_body_q
        link = self.ik_model.body_label.index(builder.body_label[self.knife])
        rotation = wp.mul(wp.quat_inverse(wp.quat(*poses[link, 3:])), wp.quat(*poses[self.ee[1], 3:]))
        # With the gripper at its target orientation, the knife frame is aligned with the world:
        # +X along the blade, +Y across its thickness and +Z up.
        rotation = wp.mul(rotation, wp.quat_inverse(wp.quat(*HORIZONTAL_GRIPPER)))
        self.knife_local = wp.transform(self.scene_tcp[1], rotation)
        robot_shapes = builder.shape_count
        handle_cfg = newton.ModelBuilder.ShapeConfig(density=700.0, ke=1.0e5, kd=100.0, mu=2.0, is_visible=visible)
        builder.add_shape_box(
            self.knife,
            xform=self.knife_local,
            hx=0.048,
            hy=HANDLE_HALF_WIDTH,
            hz=0.012,
            cfg=handle_cfg,
            color=(0.08, 0.05, 0.03),
            label="knife_handle",
        )
        blade_cfg = newton.ModelBuilder.ShapeConfig(density=2800.0, ke=1.0e5, kd=100.0, mu=0.03, is_visible=visible)
        # Particle radii supply the contact offset, so no extra margin is added.
        blade_cfg.margin = 0.0
        vertices, faces = blade_mesh(BLADE_MIN_HALF_WIDTH, bevel=BLADE_BEVEL, edge_drop=BLADE_EDGE_DROP)
        builder.add_shape_mesh(
            self.knife,
            xform=self.knife_local,
            mesh=newton.Mesh(vertices, faces.ravel(), maxhullvert=0),
            cfg=blade_cfg,
            color=(0.75, 0.78, 0.81),
            label="knife_blade",
        )
        self.knife_shapes = (robot_shapes, robot_shapes + 1)
        builder.add_shape_collision_filter_pair(self.knife_shapes[1], self.board_shape)
        for tool in self.knife_shapes:
            for robot in range(robot_shapes):
                if builder.shape_body[robot] >= 0:
                    builder.add_shape_collision_filter_pair(tool, robot)

    def _configure_drives(self):
        """Stiff arm drives and geared-finger armature, as in the reference W1 scene."""
        args = self.args
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
                ke[begin:end], kd[begin:end] = args.finger_kp, args.finger_kd
                armature[begin:end] = np.maximum(armature[begin:end], args.finger_armature)
                damping[begin:end] = np.maximum(damping[begin:end], args.finger_damping)
            else:
                ke[begin:end], kd[begin:end] = args.arm_kp, args.arm_kd
            mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.model.joint_armature.assign(armature)
        self.model.joint_damping.assign(damping)

    # Robot control -------------------------------------------------------------

    def _build_ik(self, device):
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        home = sample_keyframes(self.keyframes, 0.0)[0]
        self.positions = [
            ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*p)], dtype=wp.vec3, device=device))
            for body, p in zip(self.ee, home, strict=True)
        ]
        rotations = [
            ik.IKObjectiveRotation(
                body, wp.quat_identity(), wp.array([HORIZONTAL_GRIPPER], dtype=wp.vec4, device=device)
            )
            for body in self.ee
        ]
        limits = ik.IKObjectiveJointLimit(self.ik_model.joint_limit_lower, self.ik_model.joint_limit_upper, weight=10.0)
        mask = np.zeros(self.ik_model.joint_dof_count, dtype=bool)
        for side in ("LEFT", "RIGHT"):
            for j in range(1, 8):
                mask[self.coords[f"{side}_J{j}"]] = True
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            1,
            [*self.positions, *rotations, limits],
            joint_dof_mask=wp.array(mask, dtype=wp.bool, device=device),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            lambda_initial=0.03,
        )
        self.lower = self.ik_model.joint_limit_lower.numpy()
        self.upper = self.ik_model.joint_limit_upper.numpy()

    def _solve_ik(self, targets, fingers, *, iterations):
        for objective, p in zip(self.positions, targets, strict=True):
            objective.set_target_position(0, wp.vec3(*p))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        q = np.clip(self.ik_q.numpy()[0], self.lower, self.upper)
        for coords, opening in zip(self.finger_coords, np.broadcast_to(fingers, (2,)), strict=True):
            q[coords] = opening
        self.ik_q.assign(q.reshape(1, -1))
        return q

    # Simulation ----------------------------------------------------------------

    def simulate(self):
        for substep in range(self.sim_substeps):
            wp.launch(
                interpolate_joint_targets,
                dim=len(self.frame_end),
                inputs=[
                    self.frame_start,
                    self.frame_end,
                    (substep + 1) / self.sim_substeps,
                    1.0 / self.frame_dt,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                compensate_gravity,
                dim=self.robot_bodies,
                inputs=[self.model.body_mass, self.gravity, self.state_0.body_f],
                device=self.model.device,
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # Copy instead of swapping so the captured graph and the bread views keep fixed buffers.
            self.state_0.assign(self.state_1)
            wp.launch(
                observe_fracture,
                dim=len(self.bonds),
                inputs=[
                    self.state_0.particle_q,
                    self.bonds,
                    self.model.synreal_granular_bond_rest_lengths,
                    self.args.break_strain,
                    self.broken,
                ],
                device=self.model.device,
            )

    def step(self):
        targets, fingers = sample_keyframes(self.keyframes, self.sim_time + self.frame_dt)
        wp.copy(self.frame_start, self.frame_end)
        self._solve_ik(targets, fingers, iterations=self.args.ik_iterations)
        wp.copy(self.frame_end, self.ik_q.flatten())
        if self.graph is not None:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        if self.frame % 30 == 0:
            self._report()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.visuals is not None:
            self.visuals.render(self.state_0)
        self.viewer.log_points("/bread/material", self.bread_q, 0.57 * SPACING, self.colors)
        self.viewer.end_frame()

    # Diagnostics ---------------------------------------------------------------

    def pieces(self):
        """Label connected material pieces from irreversibly broken bonds."""
        intact = self.bond_pairs[self.broken.numpy() == 0]
        labels = connected_components(self.bread_count, intact)
        _, labels = np.unique(labels, return_inverse=True)
        return labels, np.bincount(labels)

    def knife_pose(self):
        return wp.mul(wp.transform(*self.state_0.body_q.numpy()[self.knife]), self.knife_local)

    def _report(self):
        q, qd = self.bread_q.numpy(), self.bread_qd.numpy()
        if not (np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(self.state_0.body_q.numpy()).all()):
            raise RuntimeError(f"Non-finite state at t={self.sim_time:.2f}s")
        speed = float(np.linalg.norm(qd, axis=1).max())
        self.peak_speed = max(self.peak_speed, speed)
        _, sizes = self.pieces()
        self.observed_piece_counts.add(int((sizes >= 0.1 * self.bread_count).sum()))
        openings = []
        for cut_y in (CUT_Y, SECOND_CUT_Y):
            near = np.abs(self.rest[:, 1] - cut_y) < 0.015
            sides = np.where(self.rest[:, 1] < cut_y, -1.0, 1.0)
            openings.append(float(np.mean((q[near, 1] - self.rest[near, 1]) * sides[near]) * 2))
        knife = self.knife_pose()
        broken = self.broken.numpy().astype(bool)
        self.piece_history.append((self.sim_time, len(sizes), *openings))
        print(
            f"[GranularBread] t={self.sim_time:5.2f}s knife_z={knife[2] - BOARD_TOP:+.3f}m "
            f"pieces={sorted(sizes.tolist(), reverse=True)[:5]} total={len(sizes)} "
            f"opening={openings[0] * 1000:.1f}/{openings[1] * 1000:.1f}mm "
            f"broken={int(broken.sum())} peak_v={speed:.2f}m/s",
            flush=True,
        )

    def test_final(self):
        q = self.bread_q.numpy()
        if not np.isfinite(q).all():
            raise ValueError("Bread produced a non-finite state")
        if q[:, 2].min() < BOARD_TOP - 0.006:
            raise ValueError("Bread penetrated the chopping board")
        labels, sizes = self.pieces()
        large = np.sort(sizes)[::-1]
        detached = count_detached_samples(q, labels, SPACING)
        loose = int(sizes[sizes < 0.1 * self.bread_count].sum())
        stats = dict(
            pieces=large[:5].tolist(), loose_grains=loose, detached=detached, peak_speed=round(self.peak_speed, 3)
        )
        if self.frame >= TOTAL_FRAMES:
            if (
                (large[:3] < 0.1 * self.bread_count).any()
                or len(large) < 3
                or large[3:].sum() > 0.01 * self.bread_count
            ):
                raise ValueError(f"Expected three coherent bread pieces: {stats}")
            if detached:
                raise ValueError(f"A bread piece shed grains: {stats}")
            if self.observed_piece_counts != {1, 2, 3}:
                raise ValueError(f"Missing sequential cutting stages: {sorted(self.observed_piece_counts)}")
        print(f"[GranularBread] checks passed: {stats}", flush=True)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.set_defaults(num_frames=TOTAL_FRAMES)
        parser.add_argument("--assets", default=str(REPO_ASSETS), help="engineRTS asset directory.")
        parser.add_argument("--visual-style", choices=("realistic", "particles"), default="realistic")
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=6)
        parser.add_argument("--linear-iterations", type=int, default=40)
        parser.add_argument("--ik-iterations", type=int, default=4)
        parser.add_argument("--cohesion", type=float, default=60.0, help="Initial bond stiffness in N/m.")
        parser.add_argument("--bond-damping", type=float, default=0.02, help="Axial bond damping in N*s/m.")
        parser.add_argument("--break-strain", type=float, default=0.25, help="Irreversible tensile fracture threshold.")
        parser.add_argument("--grain-stiffness", type=float, default=400.0, help="Grain contact stiffness in N/m.")
        parser.add_argument("--contact-stiffness", type=float, default=2000.0, help="Grain-shape contact stiffness.")
        parser.add_argument("--board-friction", type=float, default=0.6)
        parser.add_argument("--arm-kp", type=float, default=12000.0, help="Arm drive stiffness [N*m/rad].")
        parser.add_argument("--arm-kd", type=float, default=350.0, help="Arm drive damping [N*m*s/rad].")
        parser.add_argument("--finger-kp", type=float, default=2500.0, help="Finger drive stiffness [N/m].")
        parser.add_argument("--finger-kd", type=float, default=20.0, help="Finger drive damping [N*s/m].")
        parser.add_argument("--finger-armature", type=float, default=0.3, help="Reflected finger gear inertia [kg].")
        parser.add_argument("--finger-damping", type=float, default=50.0, help="Passive finger damping [N*s/m].")
        parser.add_argument("--no-capture", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    try:
        newton.examples.run(Example(viewer, args), args)
    except Exception:
        # The runner closes the viewer on success; USD viewers cannot be closed twice.
        viewer.close()
        raise
