# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Crouch at a worktable, turn up a bag with the left hand, then pack snacks.

uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing

Record once with --record (headless by default), then play with --replay --loop.
Both options accept an optional recording directory.
Use --robot-setback to tune the distance from the fixed worktable [m].
Only the robot is prescribed. Paper and snacks move through contact, with no
attachments, pose resets, or fixed bag vertices. Asset scale is estimated from
the user's video; snacks use untextured primitive rendering.
"""

import json
import math
from itertools import pairwise, product
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_pick_place import ASSET, OPEN, SIDES, TCP, _prescribe
from newton.solvers import SolverMJVBDV2

ASSETS = Path(__file__).resolve().parents[3] / "assets/w1_paper_bag"
TABLE_Z = 0.92
TABLE_CENTER = np.array((0.66, -0.01, TABLE_Z - 0.022))
TABLE_HALF = np.array((0.30, 0.66, 0.022))
DEPTH, WIDTH, HEIGHT = 0.13, 0.32, 0.25
BAG_Y = 0.0
SUPPORT_OPENING = 0.0035
IDLE_OPENING = 0.015
IDLE_PITCH = math.radians(100)
PACK_START = 24.0
PACK_DURATION = 12.0


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t**3 * (10 + t * (-15 + 6 * t))


def bag_frame(progress):
    """Return the desired tipping arc; it supplies robot targets only."""
    angle = -math.pi / 2 * (1 - progress)
    rotation = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle)))).reshape(3, 3)
    position = np.array((0.60 - 0.10 * progress, BAG_Y, TABLE_Z + 0.002 + (DEPTH / 2 + 0.014) * abs(math.sin(angle))))
    return position, rotation


def fit_bag(rest, current):
    """Fit the moving paper frame, excluding the flexible handles."""
    a, b = rest.mean(0), current.mean(0)
    u, _, vt = np.linalg.svd((rest - a).T @ (current - b))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = (u @ correction @ vt).T
    position = b - rotation @ a
    error = np.linalg.norm(current - (rest @ rotation.T + position), axis=1)
    return position, rotation, float(np.sqrt(np.mean(error**2)))


def snack_extent(kind, rotation, half):
    """Return the actual primitive's half-extents in the moving bag frame."""
    if kind == "can":
        return half[0] * np.linalg.norm(rotation[:, :2], axis=1) + half[2] * np.abs(rotation[:, 2])
    return np.abs(rotation) @ half


def lip_deviation(positions, chains):
    """Measure broad-panel lip bowing independently of rigid tipping motion [m]."""
    maximum = 0.0
    for chain in chains:
        points = positions[chain]
        axis = points[-1] - points[0]
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        delta = points - points[0]
        offset = delta - np.outer(delta @ axis, axis)
        maximum = max(maximum, float(np.linalg.norm(offset, axis=1).max()))
    return maximum


def triangles_overlap_box(vertices, indices, lower, upper):
    """Detect triangle/box overlap, including crossings with no vertex inside."""
    center = (np.asarray(lower) + upper) / 2
    half = (np.asarray(upper) - lower) / 2
    triangles = vertices[indices] - center
    candidate = np.all(triangles.min(axis=1) < half, axis=1) & np.all(triangles.max(axis=1) > -half, axis=1)
    triangles = triangles[candidate]
    if not len(triangles):
        return False
    edges = np.roll(triangles, -1, axis=1) - triangles
    axes = [np.cross(edges[:, 0], edges[:, 1])]
    for edge in range(3):
        for axis in np.eye(3):
            axes.append(np.cross(edges[:, edge], axis))
    overlap = np.ones(len(triangles), dtype=bool)
    for axis in axes:
        projections = np.einsum("nij,nj->ni", triangles, axis)
        radius = np.abs(axis) @ half
        overlap &= (projections.min(axis=1) <= radius) & (projections.max(axis=1) >= -radius)
    return bool(overlap.any())


def count_handle_crossings(positions, triangles, edges):
    """Count free handle edges passing through the paper's triangle interiors."""
    a, b, c = np.moveaxis(positions[triangles], 1, 0)
    ab, ac = b - a, c - a
    count = 0
    for batch in np.array_split(edges, 16):
        origin = positions[batch[:, 0], None]
        direction = (positions[batch[:, 1]] - positions[batch[:, 0]])[:, None]
        cross = np.cross(direction, ac)
        determinant = np.sum(ab * cross, axis=-1)
        valid = np.abs(determinant) > 1e-12
        inverse = np.where(valid, 1 / np.where(valid, determinant, 1), 0)
        offset = origin - a
        u = inverse * np.sum(offset * cross, axis=-1)
        side = np.cross(offset, ab)
        v = inverse * np.sum(direction * side, axis=-1)
        t = inverse * np.sum(ac * side, axis=-1)
        count += np.count_nonzero(valid & (u > 1e-5) & (v > 1e-5) & (u + v < 1 - 1e-5) & (t > 1e-5) & (t < 1 - 1e-5))
    return count


class Example:
    def __init__(self, viewer, args, *, render_only=False):
        self.viewer, self.args = viewer, args
        if args.substeps < 1 or args.iterations < 1 or args.snacks not in (1, 2):
            raise ValueError("Use positive solver settings and one or two snacks")
        if not math.isfinite(args.robot_setback) or args.robot_setback < 0:
            raise ValueError("Robot setback must be finite and nonnegative")
        self.frame, self.sim_time = 0, 0.0
        self.tipping_height_offset = 0.0
        self.pack_x = 0.5
        self.support_target = None
        self.support_yaw = 0.0
        self.frame_dt, self.sim_dt = 1 / 60, 1 / (60 * args.substeps)
        self.home = np.array(((0.32, 0.28, TABLE_Z + 0.14), (0.32, -0.34, TABLE_Z + 0.14)))
        self.grips = np.array(((-0.060, WIDTH / 2, HEIGHT - 0.013), (0, -WIDTH / 2, HEIGHT - 0.013)))
        self.pick = np.array(((0.43, -0.36, TABLE_Z + 0.060), (0.43, -0.25, TABLE_Z + 0.060)))[: args.snacks]
        self.kinds = ("can", "carton")[: args.snacks]
        self.half = np.array(((0.0325, 0.0325, 0.059), (0.024, 0.030, 0.059)))[: args.snacks]
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.add_urdf(
            str(ASSET),
            xform=wp.transform(wp.vec3(-args.robot_setback, 0, 0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        self.robot_joints, self.robot_coords = builder.joint_count, len(builder.joint_q)
        self.coords = {
            name.rsplit("/", 1)[-1]: builder.joint_q_start[j]
            for j, name in enumerate(builder.joint_label)
            if builder.joint_type[j] != newton.JointType.FIXED
        }
        # Equal lower links fold symmetrically, lowering the torso without lean.
        for name, value in (("ANKLE", 25.0), ("KNEE", -50.0), ("BUTTOCK", 25.0)):
            builder.joint_q[self.coords[name]] = math.radians(value)
        self.ee = [
            next(i for i, name in enumerate(builder.body_label) if name.endswith(f"/{s}_gripper_base")) for s in SIDES
        ]
        for side, sign in (("LEFT", -1), ("RIGHT", 1)):
            builder.joint_q[self.coords[f"{side}_J2"]] = sign * 0.9
            builder.joint_q[self.coords[f"{side}_J4"]] = sign * 1.0
            for finger in (1, 2):
                builder.joint_q[self.coords[f"{side}_FINGER{finger}_JOINT"]] = OPEN
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        full_shapes = []
        for i in range(builder.shape_count):
            builder.shape_material_mu[i] = 1.2
            builder.shape_material_ke[i], builder.shape_material_kd[i] = 4e4, 80.0
            label = builder.body_label[builder.shape_body[i]].lower()
            source = builder.shape_source[i]
            if isinstance(source, newton.Mesh):
                source.texture = None
            if "finger" in label or label.endswith(("/link7", "/link8")):
                builder.shape_margin[i] = 0.0015
                if builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                    source = builder.shape_source[i]
                    if source is not None and not render_only:
                        source.build_sdf(target_voxel_size=0.001)
                    full_shapes.append(i)
        if not render_only:
            self.ik_model = builder.finalize()
            self.table_geometries = []
            for shape, source in enumerate(builder.shape_source):
                if source is None:
                    continue
                transform = np.asarray(builder.shape_transform[shape])
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*transform[3:]))).reshape(3, 3)
                vertices = np.asarray(source.vertices) * np.asarray(builder.shape_scale[shape])
                vertices = vertices @ rotation.T + transform[:3]
                bounds = np.array(list(product(*zip(vertices.min(axis=0), vertices.max(axis=0), strict=True))))
                self.table_geometries.append(
                    (builder.shape_body[shape], vertices, np.asarray(source.indices).reshape(-1, 3), bounds)
                )
            self._build_ik()
            self._solve_ik(self.home, (OPEN, IDLE_OPENING), (IDLE_PITCH, IDLE_PITCH), iterations=200)
            builder.joint_q[:] = self.ik_q.numpy()[0].tolist()
            self.head = next(i for i, name in enumerate(builder.body_label) if name.endswith("/head_pitch_j2_link"))
            neutral = self.ik_model.state()
            newton.eval_fk(self.ik_model, self.ik_q.flatten(), neutral.joint_qd, neutral)
            self.head_origin = neutral.body_q.numpy()[self.head, :3].copy()
            self.neck = [self.coords["NECK1"], self.coords["NECK2"]]
            self.head_yaw_range = np.array((0.0, 0.0))

        cfg = builder.ShapeConfig(ke=4e4, kd=80, mu=0.65)
        builder.add_ground_plane(cfg=builder.ShapeConfig(has_particle_collision=False), color=(0.30, 0.32, 0.34))
        table = builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*TABLE_CENTER), wp.quat_identity()),
            hx=TABLE_HALF[0],
            hy=TABLE_HALF[1],
            hz=TABLE_HALF[2],
            cfg=cfg,
            color=(0.52, 0.55, 0.56),
            label="Video worktable",
        )
        full_shapes.append(table)
        for x in (0.40, 0.91):
            for y in (-0.59, 0.57):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(x, y, (TABLE_Z - 0.044) / 2), wp.quat_identity()),
                    hx=0.022,
                    hy=0.022,
                    hz=(TABLE_Z - 0.044) / 2,
                    cfg=builder.ShapeConfig(has_particle_collision=False),
                    color=(0.22, 0.24, 0.25),
                    label="Table leg",
                )
        with np.load(ASSETS / "bag.npz") as data:
            self.rest, self.faces = data["vertices"], data["faces"]
            self.paper_count, self.paper_faces = int(data["paper_count"]), int(data["paper_faces"])
            self.bottom, self.rim = data["bottom"], data["rim"]
        self.lip_chains = []
        for side in (-1, 1):
            chain = self.rim[side * self.rest[self.rim, 0] > DEPTH / 2 - 0.005]
            self.lip_chains.append(chain[np.argsort(self.rest[chain, 1])])
        self.peak_lip_deviation = 0.0
        handle_faces = self.faces[self.paper_faces :]
        edges = np.concatenate([handle_faces[:, [0, 1]], handle_faces[:, [1, 2]], handle_faces[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.free_handle_edges = edges[np.all(edges >= self.paper_count, axis=1)]
        position, _ = bag_frame(0)
        builder.add_cloth_mesh(
            pos=wp.vec3(*position),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -math.pi / 2),
            scale=1,
            vel=wp.vec3(),
            vertices=self.rest.tolist(),
            indices=self.faces.ravel().tolist(),
            density=0.18,
            tri_ke=5e4,
            tri_ka=5e4,
            tri_kd=0.2,
            edge_ke=30.0,
            edge_kd=1.0,
            particle_radius=0.0012,
        )
        for i, edge in enumerate(builder.edge_indices):
            if max(edge) >= self.paper_count:
                builder.edge_bending_properties[i] = (0.03, 0.001)
                continue
            points = self.rest[np.asarray(edge)[np.asarray(edge) >= 0]]
            hinge = self.rest[np.asarray(edge)[2:]]
            corner = np.all(np.abs(hinge[:, 1]) > WIDTH / 2 - 0.001) and np.all(np.abs(hinge[:, 0]) > DEPTH / 2 - 0.003)
            gusset = np.all(np.abs(hinge[:, 0]) < 1e-6) and np.all(np.abs(hinge[:, 1]) > WIDTH / 2 - 0.02)
            bottom_fold = np.all(np.abs(hinge[:, 2]) < 1e-6) or np.all(np.abs(hinge[:, 2] - HEIGHT / 9) < 1e-6)
            if points[:, 2].min() > HEIGHT - 0.04:
                builder.edge_bending_properties[i] = (120.0, 2.0)
            elif corner or gusset or bottom_fold:
                builder.edge_bending_properties[i] = (8.0, 0.3)
        # The folded top hem has bonded double plies, including the grasp area.
        for i, face in enumerate(self.faces[: self.paper_faces]):
            if self.rest[face, 2].min() > HEIGHT - 0.032:
                ke, ka, kd, drag, lift = builder.tri_materials[i]
                builder.tri_materials[i] = (2 * ke, 2 * ka, 2 * kd, drag, lift)
                for v in face:
                    builder.particle_mass[v] += 0.18 * builder.tri_areas[i] / 3
        self.objects = []
        snack_info = json.loads((ASSETS / "snacks.json").read_text())
        with np.load(ASSETS / "snacks.npz") as meshes:
            for kind, point, half in zip(self.kinds, self.pick, self.half, strict=True):
                body = builder.add_body(xform=wp.transform(wp.vec3(*point), wp.quat_identity()), label=kind)
                contact = builder.ShapeConfig(density=280, ke=4e4, kd=80, mu=1.2)
                if kind == "can":
                    shape = builder.add_shape_cylinder(body, radius=half[0], half_height=half[2], cfg=contact)
                else:
                    shape = builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=contact)
                full_shapes.append(shape)
                # Retain hidden packaging geometry for existing recording signatures.
                for part in snack_info[kind]:
                    key = part["key"]
                    builder.add_shape_mesh(
                        body,
                        xform=wp.transform(wp.vec3(), wp.quat(0, 0, 1, 0)),
                        mesh=newton.Mesh(meshes[key + "_vertices"], meshes[key + "_faces"].ravel()),
                        cfg=builder.ShapeConfig(
                            density=0, has_shape_collision=False, has_particle_collision=False, is_visible=False
                        ),
                        color=tuple(part["color"]),
                        label=key,
                    )
                self.objects.append(body)
        if not render_only:
            builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 2e5, 10.0, 0.6
        self.state_0 = self.model.state()
        if not render_only:
            self._build_simulation(full_shapes)
        self.graph = None
        self.peak_ik_error, self.peak_joint_speed = 0.0, 0.0
        self.peak_z = self.pick[:, 2].copy()
        self.packed = [False] * args.snacks
        self.loaded = [False] * args.snacks
        self.viewer.set_model(self.model)
        self.viewer.show_particles, self.viewer.show_triangles = False, True
        self.viewer.set_camera(pos=wp.vec3(1.90, -1.9, 1.85), pitch=-19, yaw=137)

    @classmethod
    def create_render_scene(cls, viewer, args):
        """Build matching geometry without IK, SDFs, or physics solvers."""
        return cls(viewer, args, render_only=True)

    def _build_simulation(self, full_shapes):
        self.state_1, self.control = self.model.state(), self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.frame_start, self.frame_end = wp.clone(self.ik_model.joint_q), wp.clone(self.ik_model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": self.args.iterations,
                "rigid_soft_enable_dat": True,
                "particle_enable_multilevel_correction": True,
                "particle_multilevel_operator": "galerkin",
                "particle_multilevel_cluster_size": 8,
                "particle_multilevel_coarse_iterations": 32,
                "particle_multilevel_checkpoints": tuple(
                    sorted(
                        {
                            max(1, self.args.iterations // 3),
                            max(1, 2 * self.args.iterations // 3),
                            max(1, self.args.iterations - 2),
                        }
                    )
                ),
                "particle_multilevel_relaxation": 0.8,
                "particle_multilevel_max_radius_fraction": 2.0,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.0012,
                "particle_self_contact_margin": 0.003,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": 0.0015,
                "particle_collision_detection_interval": 0,
                "particle_chebyshev_spectral_radius": 0.9,
                "particle_chebyshev_warmup_iterations": 2,
                "particle_chebyshev_polish_iterations": 3,
                "particle_chebyshev_contact_rings": 1,
                "friction_epsilon": 0.001,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": 8192,
            },
            collision_options={
                "include_static_kinematic_pairs": False,
                "broad_phase": "nxn",
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": full_shapes,
                "soft_contact_margin": 0.004,
                "soft_contact_max": 131072,
            },
        )

    def _build_ik(self):
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.positions = [
            ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*point)], dtype=wp.vec3))
            for body, point in zip(self.ee, self.home, strict=True)
        ]
        rotation = wp.vec4(0, math.sqrt(0.5), 0, math.sqrt(0.5))
        self.rotations = [
            ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([rotation], dtype=wp.vec4), weight=0.3)
            for body in self.ee
        ]
        self.elbows = []
        self.shoulders = []
        for side, sign in (("l", 1), ("r", -1)):
            self.shoulders.append(
                next(
                    i
                    for i, name in enumerate(self.ik_model.body_label)
                    if name.endswith(f"/shoulder_pitch_{side}_j1_link")
                )
            )
            body = next(
                i for i, name in enumerate(self.ik_model.body_label) if name.endswith(f"/elbow_yaw_{side}_j4_link")
            )
            # Keep a gentle elbow preference even while turning the wrist;
            # removing it lets the redundant arm drift into another posture.
            self.elbows.append(
                ik.IKObjectivePosition(
                    body,
                    wp.vec3(),
                    wp.array([wp.vec3(0.05 - self.args.robot_setback, sign * 0.30, TABLE_Z + 0.14)], dtype=wp.vec3),
                    weight=0.03,
                )
            )
        self.lower, self.upper = self.ik_model.joint_limit_lower.numpy(), self.ik_model.joint_limit_upper.numpy()
        self.arm_coords = np.array([self.coords[f"{side}_J{j}"] for side in ("LEFT", "RIGHT") for j in range(1, 8)])
        # Preserve room at each arm joint's stops, including during wrist turns.
        self.motion_lower, self.motion_upper = self.lower.copy(), self.upper.copy()
        self.motion_lower[self.arm_coords] += math.radians(10)
        self.motion_upper[self.arm_coords] -= math.radians(10)
        objectives = (
            self.positions
            + self.elbows
            + self.rotations
            + [ik.IKObjectiveJointLimit(wp.array(self.motion_lower), wp.array(self.motion_upper), weight=10)]
        )
        mask = np.zeros(self.ik_model.joint_dof_count, dtype=bool)
        for side in ("LEFT", "RIGHT"):
            for j in range(1, 8):
                mask[self.coords[f"{side}_J{j}"]] = True
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            1,
            objectives,
            joint_dof_mask=wp.array(mask, dtype=wp.bool),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            lambda_initial=0.1,
            lambda_min=0.01,
        )
        self.ik_graph = None

    def _solve_ik(self, targets, openings, angles, *, iterations=24):
        for side, (position, rotation, point, angle) in enumerate(
            zip(self.positions, self.rotations, targets, angles, strict=True)
        ):
            position.set_target_position(0, wp.vec3(*point))
            orientation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle))
            if side == 0:
                yaw = self.support_yaw * smooth((self.sim_time - 17.0) / 1.5)
                orientation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * orientation
            rotation.set_target_rotation(0, wp.vec4(*orientation))
        previous = self.ik_q.numpy()[0]
        if self.ik_model.device.is_cuda and not self.args.no_cuda_graph and iterations == 24:
            if self.ik_graph is None:
                with wp.ScopedCapture() as capture:
                    self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
                self.ik_graph = capture.graph
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=iterations)
        q = np.clip(self.ik_q.numpy()[0], self.motion_lower, self.motion_upper)
        if iterations <= 24:
            delta = q - previous
            fraction = min(1.0, 4.0 * self.frame_dt / max(float(np.max(np.abs(delta))), 1e-9))
            q = previous + fraction * delta
        for side, opening in zip(("LEFT", "RIGHT"), openings, strict=True):
            for finger in (1, 2):
                q[self.coords[f"{side}_FINGER{finger}_JOINT"]] = opening
        if iterations <= 24:
            point = self._gaze_target(targets, self.sim_time)
            direction = point - self.head_origin
            desired = np.array(
                (
                    math.atan2(direction[1], direction[0]),
                    math.atan2(-direction[2], np.linalg.norm(direction[:2])),
                )
            )
            desired = np.clip(desired, np.radians((-55, -25)), np.radians((55, 40)))
            desired = np.clip(desired, self.lower[self.neck], self.upper[self.neck])
            blend = 1 - math.exp(-self.frame_dt / 0.3)
            q[self.neck] = previous[self.neck] + np.clip(
                blend * (desired - previous[self.neck]), -0.7 * self.frame_dt, 0.7 * self.frame_dt
            )
            self.head_yaw_range[0] = min(self.head_yaw_range[0], q[self.neck[0]])
            self.head_yaw_range[1] = max(self.head_yaw_range[1], q[self.neck[0]])
        self.ik_q.assign(q.reshape(1, -1))
        return q

    @staticmethod
    def _gaze_target(targets, t):
        packing = smooth((t - (PACK_START - 1.0)) / 2.0)
        return (1 - packing) * targets[0] + packing * targets[1]

    def _plan(self, t):
        progress = smooth((t - 5.5) / 4.5)
        position, rotation = bag_frame(progress)
        grip = self.grips @ rotation.T + position
        grip[:, 2] += self.tipping_height_offset
        angles = np.full(2, math.radians(60) + math.radians(90) * progress)
        # Grip nearer the finger root on the upper edge of the laid bag. This
        # permits the full tip-up rotation without ending in a vertical wrist.
        grip[0] += 0.025 * np.array((math.sin(angles[0]), 0, math.cos(angles[0])))
        grip[0, 0] += 0.075 * progress
        grip[0, 2] += 0.015 * (1 - progress)
        targets = grip.copy()
        openings = np.full(2, OPEN - (OPEN - SUPPORT_OPENING) * smooth(t - 4.5))
        if t < 4.5:
            approach = grip + np.array((-0.10, 0, 0))
            a = smooth((t - 1.0) / 2)
            targets = (1 - a) * self.home + a * approach
            angles[0] = (1 - a) * IDLE_PITCH + a * math.pi / 3
            if t > 3:
                a = smooth((t - 3) / 1.5)
                targets = (1 - a) * approach + a * grip
        if t > 10.5:
            # Clear the inner jaw before moving sideways, then translate while
            # turning the wrist to avoid the down-facing arm's joint limits.
            raised = grip[0] + np.array((0, 0, 0.08))
            outside = raised + np.array((-0.04, 0.13, 0))
            turned = outside + np.array((0.09, 0, -0.06))
            support = (
                self.support_target if self.support_target is not None else np.array((0.55, 0.18, TABLE_Z + 0.267))
            )
            waypoints = [
                (10.5, grip[0], SUPPORT_OPENING, math.radians(150)),
                (11.2, grip[0], OPEN, math.radians(150)),
                (12.2, raised, OPEN, math.radians(105)),
                (13.2, outside, OPEN, math.radians(105)),
                (17.0, turned, OPEN, math.radians(95)),
                (18.5, support + np.array((0, 0, 0.10)), OPEN, math.radians(95)),
                (19.7, support, OPEN, math.radians(95)),
                (20.5, support, SUPPORT_OPENING, math.radians(95)),
            ]
            targets[0], openings[0], angles[0] = waypoints[-1][1:]
            for (a, pa, ga, ra), (b, pb, gb, rb) in pairwise(waypoints):
                if t <= b:
                    alpha = smooth((t - a) / (b - a))
                    targets[0] = (1 - alpha) * pa + alpha * pb
                    openings[0] = (1 - alpha) * ga + alpha * gb
                    angles[0] = (1 - alpha) * ra + alpha * rb
                    break
        # The left hand turns the bag up; the right hand waits clear of the rim.
        targets[1], openings[1], angles[1] = self.home[1], IDLE_OPENING, IDLE_PITCH
        if t >= PACK_START:
            item = min(int((t - PACK_START) / PACK_DURATION), len(self.objects) - 1)
            phase = t - PACK_START - item * PACK_DURATION
            # Grasp above the centre so the base enters the mouth before the
            # fingers release, while the wrist remains clear of the paper.
            pick = self.pick[item] + np.array((0, 0, 0.035))
            high = TABLE_Z + HEIGHT + 0.14
            above_pick = np.array((pick[0], pick[1], high))
            approach = pick + np.array((0, 0, 0.08))
            # Clear the back wall's inward fold before lowering into the mouth.
            carry = np.array((self.pack_x, BAG_Y - 0.045 + item * 0.075, high))
            # Release close to the mouth while keeping the horizontal finger
            # bodies above the rim when they open.
            drop = carry - np.array((0, 0, 0.07))
            start = self.home[1] if item == 0 else np.array((self.pack_x - 0.10, BAG_Y - 0.045, high))
            closed = 0.030 if item == 0 else 0.028
            waypoints = [
                (0, start, IDLE_OPENING if item == 0 else OPEN, IDLE_PITCH if item == 0 else math.pi / 2),
                (2.8, approach if item == 0 else above_pick, OPEN, math.pi / 2),
                (4.0, approach, OPEN, math.pi / 2),
                (5.4, pick, OPEN, math.pi / 2),
                (6.0, pick, closed, math.pi / 2),
                (7.2, np.array((pick[0], pick[1], high)), closed, math.pi / 2),
                (8.6, carry, closed, math.pi / 2),
                (9.3, drop, closed, math.pi / 2),
                (10.0, drop, OPEN, math.pi / 2),
                (11.0, carry, OPEN, math.pi / 2),
                (12.0, carry + np.array((-0.10, 0, 0)), OPEN, math.pi / 2),
            ]
            targets[1], openings[1], angles[1] = waypoints[-1][1:]
            for (a, pa, ga, ra), (b, pb, gb, rb) in pairwise(waypoints):
                if phase <= b:
                    alpha = smooth((phase - a) / (b - a))
                    targets[1] = (1 - alpha) * pa + alpha * pb
                    openings[1] = (1 - alpha) * ga + alpha * gb
                    angles[1] = (1 - alpha) * ra + alpha * rb
                    break
        return targets, openings, angles

    def _simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                _prescribe,
                self.robot_joints,
                [
                    self.frame_start,
                    self.frame_end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (substep + 1) / self.args.substeps,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= 17.0 and self.support_target is None:
            # Regrasp the settled rim once; following contacted vertices would
            # feed paper deformation back into the commanded hand motion.
            self.support_target = self.bag_position + self.bag_rotation @ np.array((0, WIDTH / 2, HEIGHT + 0.005))
            self.support_yaw = math.atan2(self.bag_rotation[1, 0], self.bag_rotation[0, 0])
        if PACK_START - 0.5 <= self.sim_time < PACK_START:
            # Leave room for the upright snack to settle toward the bag bottom
            # when its weight reduces the supported shell's residual tilt.
            self.pack_x = float((self.bag_position + self.bag_rotation @ np.array((0, 0, HEIGHT * 0.35)))[0])
        if 4.5 <= self.sim_time <= 10.5:
            gap = float(self.state_0.particle_q.numpy()[self.bottom, 2].min()) - TABLE_Z
            correction = float(np.clip(0.25 * (0.0015 - gap), -0.0004, 0.0004))
            # Once the wrists rise clear, allow enough correction to keep the pivot on the table.
            minimum = -0.033 * smooth((self.sim_time - 5.5) / 1.5)
            self.tipping_height_offset = float(np.clip(self.tipping_height_offset + correction, minimum, 0.01))
        targets, openings, angles = self._plan(self.sim_time + self.frame_dt)
        start = self.ik_q.numpy()[0]
        end = self._solve_ik(targets, openings, angles)
        self.peak_joint_speed = max(self.peak_joint_speed, float(np.max(np.abs(end - start)) / self.frame_dt))
        self.frame_start.assign(start)
        self.frame_end.assign(end)
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph and self.args.substeps % 2 == 0:
                saved_a, saved_b = self.model.state(), self.model.state()
                saved_a.assign(self.state_0)
                saved_b.assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(saved_a)
                self.state_1.assign(saved_b)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame / 60
        bodies, q = self.state_0.body_q.numpy(), self.state_0.particle_q.numpy()
        self.bag_position, self.bag_rotation, self.shape_error = fit_bag(
            self.rest[: self.paper_count], q[: self.paper_count]
        )
        self.peak_lip_deviation = max(self.peak_lip_deviation, lip_deviation(q, self.lip_chains))
        self.peak_z = np.maximum(self.peak_z, bodies[self.objects, 2])
        for body, target in zip(self.ee, targets, strict=True):
            tcp = np.asarray(wp.transform_point(wp.transform(*bodies[body]), TCP))
            self.peak_ik_error = max(self.peak_ik_error, float(np.linalg.norm(tcp - target)))
        for i, body in enumerate(self.objects):
            local = self.bag_rotation.T @ (bodies[body, :3] - self.bag_position)
            orientation = np.asarray(wp.quat_to_matrix(wp.quat(*bodies[body, 3:]))).reshape(3, 3)
            extent = snack_extent(self.kinds[i], self.bag_rotation.T @ orientation, self.half[i])
            self.loaded[i] = bool(
                np.all(np.abs(local[:2]) + extent[:2] < (DEPTH / 2 + 0.003, WIDTH / 2 + 0.003))
                and local[2] - extent[2] > -0.007
                and local[2] + extent[2] < HEIGHT
            )
            if self.sim_time > PACK_START + i * PACK_DURATION + 8.5:
                self.packed[i] |= self.loaded[i]
        if self.frame % 60 == 0:
            print(
                f"[W1Bag] t={self.sim_time:.1f} up={self.bag_rotation[2, 2]:.3f} shape={self.shape_error:.4f} "
                f"IK={self.peak_ik_error:.4f} loaded={self.loaded} objects={bodies[self.objects, :3].round(3).tolist()}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Reject unstable paper, unreachable targets, or lost snacks."""
        for field in (self.state_0.particle_q, self.state_0.body_q):
            if not np.isfinite(field.numpy()).all():
                raise AssertionError("Nonfinite simulation state")
        bodies = self.state_0.body_q.numpy()
        if self.sim_time > 3:
            left_drop = bodies[self.shoulders[0], 2] - bodies[self.elbows[0].link_index, 2]
            if left_drop < 0.05:
                raise AssertionError("The left elbow rose into the horizontal, outstretched posture")
            hands = np.array([wp.transform_point(wp.transform(*bodies[body]), TCP) for body in self.ee])
            direction = self._gaze_target(hands, self.sim_time) - bodies[self.head, :3]
            forward = np.asarray(wp.quat_rotate(wp.quat(*bodies[self.head, 3:]), wp.vec3(1, 0, 0)))
            if np.dot(forward, direction) / np.linalg.norm(direction) < math.cos(math.radians(35)):
                raise AssertionError("The head turned away from the operating hand")
        lower = TABLE_CENTER - TABLE_HALF
        upper = TABLE_CENTER + TABLE_HALF + np.array((0, 0, 0.003))
        for body, vertices, indices, bounds in self.table_geometries:
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*bodies[body, 3:]))).reshape(3, 3)
            world_bounds = bounds @ rotation.T + bodies[body, :3]
            if np.any(world_bounds.min(axis=0) >= upper) or np.any(world_bounds.max(axis=0) <= lower):
                continue
            world = vertices @ rotation.T + bodies[body, :3]
            if triangles_overlap_box(world, indices, lower, upper):
                raise AssertionError(f"Robot mesh entered the table's 3 mm clearance: {self.model.body_label[body]}")
        if self.sim_time < PACK_START:
            right_tcp = np.asarray(wp.transform_point(wp.transform(*bodies[self.ee[1]]), TCP))
            if np.linalg.norm(right_tcp - self.home[1]) > 0.01:
                raise AssertionError("The right hand must wait clear during single-hand tipping")
            joints = self.state_0.joint_q.numpy()
            for finger in (1, 2):
                if abs(joints[self.coords[f"RIGHT_FINGER{finger}_JOINT"]] - IDLE_OPENING) > 1e-4:
                    raise AssertionError("The right gripper must stay relaxed during single-hand tipping")
        else:
            left_forward = wp.quat_rotate(wp.quat(*bodies[self.ee[0], 3:]), wp.vec3(0, 0, 1))
            if abs(left_forward[2]) > 0.35:
                raise AssertionError("The left gripper must support the rim from the side during packing")
        if self.frame % 10 == 0:
            crossings = count_handle_crossings(
                self.state_0.particle_q.numpy(), self.faces[: self.paper_faces], self.free_handle_edges
            )
            if crossings:
                raise AssertionError(f"Handle edges intersect the paper wall: {crossings}")
        # One-sided support may bow the rim; reject the previous large panel collapse.
        if self.peak_lip_deviation > 0.03:
            raise AssertionError(f"Broad paper panel lip bowed too far: {self.peak_lip_deviation:.4f} m")
        if self.peak_ik_error > 0.04:
            raise AssertionError(f"TCP tracking error exceeded 4 cm: {self.peak_ik_error:.4f} m")
        if 5.5 <= self.sim_time <= 11:
            gap = float(self.state_0.particle_q.numpy()[self.bottom, 2].min()) - TABLE_Z
            if gap > 0.01:
                raise AssertionError(f"Bag bottom left the table during tipping: {gap:.4f} m")
        if self.sim_time >= 11 and self.bag_rotation[2, 2] < math.cos(math.radians(30)):
            raise AssertionError("The left hand must turn the bag mouth upward before packing")
        if self.peak_joint_speed > 4.145:
            raise AssertionError("Arm motion exceeded the source joint velocity limit")
        joints = self.state_0.joint_q.numpy()[: self.robot_coords]
        if np.any(joints < self.lower - 1e-5) or np.any(joints > self.upper + 1e-5):
            raise AssertionError("Arm motion exceeded the source joint position limits")
        arms = self.arm_coords
        margin = np.minimum(joints[arms] - self.lower[arms], self.upper[arms] - joints[arms])
        if np.min(margin) < math.radians(10) - 1e-5:
            raise AssertionError("Arm joints must stay at least 10 degrees away from their stops")
        shape_limit = 0.04 if self.sim_time < 11 else 0.025
        if self.shape_error > shape_limit:
            raise AssertionError(f"Paper body collapsed: {self.shape_error:.4f} m")

    def test_final(self):
        """Require physical tipping, snack pickup, release and full containment."""
        self.test_post_step()
        if self.sim_time < PACK_START + len(self.objects) * PACK_DURATION + 1:
            raise AssertionError("Run the complete stand-up and packing sequence")
        if not all(self.loaded) or not all(self.packed):
            raise AssertionError(f"Snacks not retained inside the paper bag: {self.loaded}")
        if self.head_yaw_range[0] > -0.3 or self.head_yaw_range[1] < 0.2:
            raise AssertionError("The head must follow both operating hands")
        if np.any(self.peak_z < self.pick[:, 2] + 0.20):
            raise AssertionError("The right gripper must physically lift every snack")
        joint_q = self.state_0.joint_q.numpy()
        for finger in (1, 2):
            if abs(joint_q[self.coords[f"RIGHT_FINGER{finger}_JOINT"]] - OPEN) > 1e-4:
                raise AssertionError("The right hand must release the snacks")
        if np.max(np.linalg.norm(self.state_0.body_qd.numpy()[self.objects], axis=1)) > 0.08:
            raise AssertionError("Released snacks have not settled")
        if np.any(self.model.particle_inv_mass.numpy() <= 0):
            raise AssertionError("Paper bag must remain unpinned")
        if np.any(self.model.body_flags.numpy()[self.objects] & int(newton.BodyFlags.KINEMATIC)):
            raise AssertionError("Snacks must remain dynamic")
        print("[W1Bag] PASS: bag stood up; snacks grasped, released and retained.", flush=True)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=3000)
        parser.add_argument("--snacks", type=int, choices=(1, 2), default=2)
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=12)
        parser.add_argument(
            "--robot-setback",
            type=float,
            default=0.03,
            help="Move the robot away from the table along -X [m] (default: 0.03).",
        )
        parser.add_argument("--no-cuda-graph", action="store_true")
        from newton.examples.mjvbdv2.support.w1_bag_recording import DEFAULT_RECORDING  # noqa: PLC0415

        modes = parser.add_mutually_exclusive_group()
        modes.add_argument(
            "--record",
            nargs="?",
            const=str(DEFAULT_RECORDING),
            metavar="DIRECTORY",
            help="Record every simulation frame to a new directory (defaults to a null viewer).",
        )
        modes.add_argument(
            "--replay",
            nargs="?",
            const=str(DEFAULT_RECORDING),
            metavar="DIRECTORY",
            help="Play saved states without IK or physics; use the recorded scene settings.",
        )
        parser.add_argument("--loop", action="store_true", help="Loop the recording during replay.")
        parser.add_argument("--start-frame", type=int, default=0, help="First recorded frame to replay.")
        parser.add_argument("--unthrottled", action="store_true", help="Replay without the 60 FPS limit.")
        return parser


if __name__ == "__main__":
    from newton.examples.mjvbdv2.support.w1_bag_recording import run_recording, run_replay

    parser = Example.create_parser()
    mode = parser.parse_args()
    if mode.record:
        parser.set_defaults(viewer="null")
    if mode.replay:
        parser.set_defaults(num_frames=2**31 - 1)
    viewer, args = newton.examples.init(parser)
    if args.record:
        run_recording(Example, viewer, args)
    elif args.replay:
        run_replay(Example, viewer, args)
    else:
        newton.examples.run(Example(viewer, args), args)
