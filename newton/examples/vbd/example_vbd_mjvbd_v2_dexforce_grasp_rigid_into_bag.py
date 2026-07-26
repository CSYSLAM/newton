# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 moves rigid objects from a table into a suspended soft bag.

Five dynamic rigid shapes start on the table. Physical five-finger contact picks them up one at a
time, moves them over a suspended open-topped cloth bag, and releases them.
The bag matches the material and resolution of
example_vbd_soft_rigid_mix_contact.py and has a pinned top rim.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBDV2

FPS = 60
SIM_SUBSTEPS = 5
VBD_ITERATIONS = 15
IK_ITERATIONS = 24

TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

# Start with the box: it is the repeatable heavy-impact case for this example.
SHAPE_NAMES = ("box", "mesh", "cone", "sphere", "cylinder")
SHAPE_SIZE = 0.027
SHAPE_MARGIN = 0.005
SHAPE_DENSITY = 2500.0
SHAPE_POSITIONS = (
    wp.vec3(0.34, -0.22, TABLE_TOP_Z + SHAPE_SIZE + SHAPE_MARGIN),
    wp.vec3(0.42, -0.22, TABLE_TOP_Z + 0.575 * SHAPE_SIZE + SHAPE_MARGIN),
    wp.vec3(0.50, -0.22, TABLE_TOP_Z + SHAPE_SIZE + SHAPE_MARGIN),
    wp.vec3(0.58, -0.22, TABLE_TOP_Z + SHAPE_SIZE + SHAPE_MARGIN),
    wp.vec3(0.64, -0.22, TABLE_TOP_Z + 0.5 * SHAPE_SIZE + SHAPE_MARGIN),
)
SHAPE_COLORS = (
    (0.83, 0.56, 0.24),
    (0.90, 0.32, 0.18),
    (0.20, 0.58, 0.90),
    (0.25, 0.76, 0.38),
    (0.66, 0.35, 0.86),
)

# Match example_vbd_soft_rigid_mix_contact.py.
BAG_WIDTH = 0.12
BAG_DEPTH = 0.07
BAG_HEIGHT = 0.24
BAG_TABLE_GAP = 0.06
BAG_SETTLE_FRAMES = 120
BAG_PREP_FRAMES = BAG_SETTLE_FRAMES
BAG_READY_BASE_Z = TABLE_TOP_Z - BAG_HEIGHT
BAG_POS = wp.vec3(
    0.448,
    -(TABLE_HALF_EXTENTS[1] + BAG_TABLE_GAP + 0.5 * BAG_DEPTH),
    BAG_READY_BASE_Z,
)
BAG_RESOLUTION = 20
BAG_PARTICLE_RADIUS = 0.003
BAG_DENSITY = 0.08
BAG_TRI_KE = 1.5e2
BAG_TRI_KA = 1.5e2
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 0.5
BAG_EDGE_KD = 1.5e-5
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
SOFT_CONTACT_MARGIN = 0.01
SOFT_CONTACT_KE = 5.0e3
SOFT_CONTACT_KD = 5.0e-2
SOFT_CONTACT_MU = 0.25
GRASP_CONTACT_KE = 1.5e4
DEFAULT_IMPACT_HEIGHT = 0.15
FIRST_IMPACT_RECOVERY_TIME = 0.75
GRASP_TCP_X_OFFSET = 0.03
BAG_TCP_X_OFFSET = 0.036
BAG_TCP_Y_OFFSET = -0.077
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
RIGHT_GRASP_ROT = wp.quat(-0.0733, 0.7031, 0.7037, -0.0717)

WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
CAMERA_PITCH = -18.0
CAMERA_YAW = 126.0
DEFAULT_HOUSE_USD = (
    "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/"
    "House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
)
MESH_CONTACT_KD = 1.0


def _generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
    """Generate a merged five-face box mesh with an open top."""
    cell_x = 2.0 * half_x / resolution
    cell_y = 2.0 * half_y / resolution
    cell_z = height / resolution
    vertex_map = {}
    vertices = []
    indices = []

    def vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    def quad(v00, v10, v01, v11):
        indices.extend((v00, v10, v01, v10, v11, v01))

    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            x1 = x0 + cell_x
            y0 = -half_y + j * cell_y
            y1 = y0 + cell_y
            quad(vertex(x0, y0, 0.0), vertex(x1, y0, 0.0), vertex(x0, y1, 0.0), vertex(x1, y1, 0.0))

    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            x1 = x0 + cell_x
            y0 = -half_y + i * cell_y
            y1 = y0 + cell_y
            z0 = j * cell_z
            z1 = z0 + cell_z
            quad(vertex(x0, -half_y, z0), vertex(x1, -half_y, z0), vertex(x0, -half_y, z1), vertex(x1, -half_y, z1))
            quad(vertex(x1, half_y, z0), vertex(x0, half_y, z0), vertex(x1, half_y, z1), vertex(x0, half_y, z1))
            quad(vertex(-half_x, y1, z0), vertex(-half_x, y0, z0), vertex(-half_x, y1, z1), vertex(-half_x, y0, z1))
            quad(vertex(half_x, y0, z0), vertex(half_x, y1, z0), vertex(half_x, y0, z1), vertex(half_x, y1, z1))

    return np.asarray(vertices, dtype=np.float32), indices


def _load_bear_mesh(target_size: float):
    stage = Usd.Stage.Open(os.path.join(newton.examples.get_asset_directory(), "bear.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
    points = np.asarray(geom.GetPointsAttr().Get(), dtype=np.float32).copy()
    indices = np.asarray(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    points -= 0.5 * (points.max(axis=0) + points.min(axis=0))
    points *= (2.0 * target_size) / (points.max(axis=0) - points.min(axis=0)).max()
    return points, indices


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _lift_pinned_vertices(
    pinned_indices: wp.array[int],
    original_positions: wp.array[wp.vec3],
    dz: float,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    i = wp.tid()
    particle = pinned_indices[i]
    position = original_positions[i]
    lifted = wp.vec3(position[0], position[1], position[2] + dz)
    pos_0[particle] = lifted
    pos_1[particle] = lifted


@wp.kernel
def _measure_bag_min_z(
    particle_start: int,
    particle_end: int,
    particle_q: wp.array[wp.vec3],
    minimum_z: wp.array[float],
):
    if wp.tid() != 0:
        return

    z = float(1.0e6)
    for particle in range(particle_start, particle_end):
        z = wp.min(z, particle_q[particle][2])
    minimum_z[0] = z


@wp.kernel
def _accumulate_bag_min_z(
    particle_start: int,
    particle_end: int,
    particle_q: wp.array[wp.vec3],
    minimum_z: wp.array[float],
):
    if wp.tid() != 0:
        return

    z = float(1.0e6)
    for particle in range(particle_start, particle_end):
        z = wp.min(z, particle_q[particle][2])
    minimum_z[0] = wp.min(minimum_z[0], z)


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


class Example:
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
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        if args.impact_height <= 0.0:
            raise ValueError(f"--impact-height must be > 0, got {args.impact_height}")
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        flags = self.model.particle_flags.numpy()
        flags[self.bag_top_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        particle_q = self.state_0.particle_q.numpy()
        self.bag_pinned_indices = wp.array(self.bag_top_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_original = wp.array(
            particle_q[self.bag_top_indices].copy(),
            dtype=wp.vec3,
            device=self.device,
        )

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": 2.0 * BAG_PARTICLE_RADIUS,
                "particle_topological_contact_filter_threshold": 3,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )
        self.contacts = self.solver.contacts
        self.maximum_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.maximum_body_particle_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.first_impact_baseline_z = wp.zeros(1, dtype=float, device=self.device)
        self.first_impact_min_z = wp.zeros(1, dtype=float, device=self.device)
        self.first_impact_recovery_z = wp.zeros(1, dtype=float, device=self.device)
        self._capture_first_impact_baseline = False
        self._track_first_impact = False
        self._first_impact_substeps = 0
        self._first_impact_recovery_substeps = int(np.ceil(FIRST_IMPACT_RECOVERY_TIME / self.sim_dt))
        self.object_released = np.zeros(len(self.object_bodies), dtype=bool)
        self.previous_grip = 0.0

        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self._build_ik()
        self.segments = self._segments()
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._right_hand_q()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self._build_joint_target_cache()

        self._attach_house_usd()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)

    def _robot_urdf(self) -> Path:
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = (
            Path(__file__).parents[1]
            / "multiphysics"
            / "newton_cloth_dexforce_place_tablecloth"
            / "DexforceW1V021"
            / "DexforceW1V021.urdf"
        )
        if path.is_file():
            return path
        raise FileNotFoundError("Dexforce W1 URDF is unavailable; pass --robot-urdf PATH.")

    def _build_scene(self):
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = 2.0e5
        builder.default_shape_cfg.kd = 1.0e-4
        builder.default_shape_cfg.mu = 1.0
        SolverMJVBDV2.register_custom_attributes(builder)
        robot_articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(robot_articulation_start, builder.articulation_count))
        if not self.robot_articulations:
            raise RuntimeError("Dexforce URDF did not create an articulation")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=3.0e5,
            kd=1.0e-4,
            mu=0.9,
            is_visible=bool(self.args.show_physics_table),
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self._world_vec(TABLE_POS), self.base_rot),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="pick_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="pick_ground")

        bag_vertices, bag_indices = _generate_box_bag(
            0.5 * BAG_WIDTH,
            0.5 * BAG_DEPTH,
            BAG_HEIGHT,
            BAG_RESOLUTION,
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=self._world_vec(BAG_POS),
            rot=self.base_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=BAG_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
            label="suspended_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        top = np.flatnonzero(np.abs(bag_vertices[:, 2] - BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = top.astype(np.int32) + self.bag_particle_start

        cfg = newton.ModelBuilder.ShapeConfig(
            density=SHAPE_DENSITY,
            ke=SOFT_CONTACT_KE,
            kd=SOFT_CONTACT_KD,
            mu=SOFT_CONTACT_MU,
            margin=SHAPE_MARGIN,
        )
        # Full-surface contact needs a volume SDF for the bear mesh; primitives
        # use their analytic SDFs.
        cfg.configure_sdf(force_sdf=True)
        cfg.has_particle_collision = True
        mesh_cfg = newton.ModelBuilder.ShapeConfig(
            density=SHAPE_DENSITY,
            ke=SOFT_CONTACT_KE,
            kd=MESH_CONTACT_KD,
            mu=SOFT_CONTACT_MU,
            margin=SHAPE_MARGIN,
        )
        mesh_cfg.configure_sdf(force_sdf=True)
        mesh_cfg.has_particle_collision = True
        bear_points, bear_indices = _load_bear_mesh(SHAPE_SIZE)
        bear_mesh = newton.Mesh(bear_points, bear_indices)
        self.object_bodies = []
        self.object_shapes = []
        for name, position, color in zip(SHAPE_NAMES, SHAPE_POSITIONS, SHAPE_COLORS, strict=True):
            body = builder.add_body(
                xform=wp.transform(self._world_vec(position), self.base_rot),
                label=f"pick_{name}",
            )
            shape = builder.shape_count
            if name == "mesh":
                builder.add_shape_mesh(body, mesh=bear_mesh, cfg=mesh_cfg, color=color, label=f"pick_{name}_shape")
            elif name == "cone":
                builder.add_shape_cone(
                    body,
                    radius=SHAPE_SIZE,
                    half_height=SHAPE_SIZE,
                    cfg=cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            elif name == "sphere":
                builder.add_shape_sphere(body, radius=SHAPE_SIZE, cfg=cfg, color=color, label=f"pick_{name}_shape")
            elif name == "box":
                builder.add_shape_box(
                    body,
                    hx=SHAPE_SIZE,
                    hy=SHAPE_SIZE,
                    hz=SHAPE_SIZE,
                    cfg=cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            elif name == "cylinder":
                builder.add_shape_cylinder(
                    body,
                    radius=SHAPE_SIZE,
                    half_height=0.5 * SHAPE_SIZE,
                    cfg=cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            self.object_bodies.append(body)
            self.object_shapes.append(shape)

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.right_hand_shapes = []
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            right_hand = "right" in label and any(word in label for word in self.HAND_CONTACT_KEYWORDS)
            if right_hand:
                self.right_hand_shapes.append(shape)
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
            else:
                builder.shape_flags[shape] &= ~(collide_shapes | collide_particles)
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_mu[self.right_hand_shapes] = 12.0
        shape_mu[self.object_shapes] = 8.0
        shape_ke[self.object_shapes] = GRASP_CONTACT_KE
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)

    def _build_ik(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
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
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.model.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.model.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.right_home)], dtype=wp.vec3, device=self.model.device),
        )
        self.right_rot = ik.IKObjectiveRotation(
            right,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.right_home))], dtype=wp.vec4, device=self.model.device),
        )
        lower, upper = self._joint_limits()
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.model.device),
            wp.array(upper, dtype=wp.float32, device=self.model.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _object_grasp_tf(self, object_index: int):
        position = SHAPE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]),
            ),
            RIGHT_GRASP_ROT,
        )

    def _object_approach_tf(self, object_index: int):
        position = SHAPE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + 0.18,
            ),
            RIGHT_GRASP_ROT,
        )

    def _object_lift_tf(self, object_index: int):
        position = SHAPE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + 0.22,
            ),
            RIGHT_GRASP_ROT,
        )

    def _segments(self):
        world = self._world_tf
        bag_prep_duration = BAG_PREP_FRAMES * self.frame_dt * self.args.trajectory_time_scale
        segments = [
            (bag_prep_duration + 0.35, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0, -1)
        ]
        right_start = self.right_home
        bag_transport = world(
            wp.transform(
                wp.vec3(
                    float(BAG_POS[0]) + BAG_TCP_X_OFFSET + GRASP_TCP_X_OFFSET,
                    float(BAG_POS[1]) + BAG_TCP_Y_OFFSET,
                    TABLE_TOP_Z + max(self.args.impact_height, 0.40),
                ),
                RIGHT_GRASP_ROT,
            )
        )
        bag_hover = world(
            wp.transform(
                wp.vec3(
                    float(BAG_POS[0]) + BAG_TCP_X_OFFSET + GRASP_TCP_X_OFFSET,
                    float(BAG_POS[1]) + BAG_TCP_Y_OFFSET,
                    TABLE_TOP_Z + self.args.impact_height,
                ),
                RIGHT_GRASP_ROT,
            )
        )
        retreat = world(
            wp.transform(
                wp.vec3(
                    float(BAG_POS[0]) + BAG_TCP_X_OFFSET + GRASP_TCP_X_OFFSET,
                    float(BAG_POS[1]) + BAG_TCP_Y_OFFSET,
                    TABLE_TOP_Z + self.args.impact_height + 0.10,
                ),
                RIGHT_GRASP_ROT,
            )
        )
        for object_index in range(len(self.object_bodies)):
            approach = world(self._object_approach_tf(object_index))
            grasp = world(self._object_grasp_tf(object_index))
            lift = world(self._object_lift_tf(object_index))
            segments.extend(
                (
                    (0.80, self.left_home, self.left_home, right_start, approach, 0.0, 0.0, object_index),
                    (1.00, self.left_home, self.left_home, approach, grasp, 0.0, 0.0, object_index),
                    (1.00, self.left_home, self.left_home, grasp, grasp, 0.0, 1.0, object_index),
                    (1.20, self.left_home, self.left_home, grasp, lift, 1.0, 1.0, object_index),
                    (2.00, self.left_home, self.left_home, lift, bag_transport, 1.0, 1.0, object_index),
                    (1.00, self.left_home, self.left_home, bag_transport, bag_hover, 1.0, 1.0, object_index),
                    (0.40, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, object_index),
                    (0.50, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 0.0, object_index),
                    (1.00, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, object_index),
                )
            )
            right_start = retreat
        segments.append((1.2, self.left_home, self.left_home, right_start, right_start, 0.0, 0.0, -1))
        return tuple(segments)

    def _build_joint_target_cache(self):
        script_duration = sum(segment[0] for segment in self.segments)
        script_frames = int(np.ceil(script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        self.cached_frame_count = max(int(self.args.num_frames), script_frames + FPS)
        initial_q = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        cache = np.repeat(initial_q[None, :], self.cached_frame_count + 1, axis=0)
        grips = np.zeros(self.cached_frame_count + 1, dtype=np.float32)
        objects = np.full(self.cached_frame_count + 1, -1, dtype=np.int32)
        hand_indices = self.hand_indices.numpy()
        hand_open = self.hand_open.numpy()
        hand_grasp = self.hand_grasp.numpy()

        for frame in range(1, self.cached_frame_count + 1):
            script_time = frame * self.frame_dt * self.args.trajectory_time_scale
            left, right, grip, object_index = self._sample(script_time)
            self.left_obj.set_target_position(0, wp.transform_get_translation(left))
            self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left)))
            self.right_obj.set_target_position(0, wp.transform_get_translation(right))
            self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right)))
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=IK_ITERATIONS)
            wp.launch(
                _lock_q,
                self.lock_indices.shape[0],
                [self.ik_q, self.lock_indices, self.lock_values],
                device=self.model.device,
            )
            cache[frame, : self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
            cache[frame, hand_indices] = hand_open * (1.0 - grip) + hand_grasp * grip
            grips[frame] = grip
            objects[frame] = object_index

        self.cached_joint_targets = wp.array(cache, dtype=wp.float32, device=self.model.device)
        self.cached_grips = grips
        self.cached_objects = objects

    def _prepare_frame(self):
        cache_index = min(self.frame_index + 1, self.cached_frame_count)
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.cached_joint_targets[cache_index])
        grip = float(self.cached_grips[cache_index])
        script_object = int(self.cached_objects[cache_index])
        if self.previous_grip > 1.0e-4 and grip <= 1.0e-4 and script_object >= 0:
            self.object_released[script_object] = True
            if script_object == 0:
                self._capture_first_impact_baseline = True
        self.previous_grip = grip

    def simulate(self):
        self._prepare_frame()
        bag_dz = 0.0
        if self._capture_first_impact_baseline:
            wp.launch(
                _measure_bag_min_z,
                1,
                [
                    self.bag_particle_start,
                    self.bag_particle_end,
                    self.state_0.particle_q,
                    self.first_impact_baseline_z,
                ],
                device=self.device,
            )
            wp.copy(self.first_impact_min_z, self.first_impact_baseline_z)
            self._capture_first_impact_baseline = False
            self._track_first_impact = True
            self._first_impact_substeps = 0
        for substep in range(self.sim_substeps):
            wp.launch(
                _lift_pinned_vertices,
                self.bag_pinned_indices.shape[0],
                [
                    self.bag_pinned_indices,
                    self.bag_pinned_original,
                    bag_dz,
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.ik_model.joint_coord_count,
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
            if self._track_first_impact:
                wp.launch(
                    _accumulate_bag_min_z,
                    1,
                    [
                        self.bag_particle_start,
                        self.bag_particle_end,
                        self.state_1.particle_q,
                        self.first_impact_min_z,
                    ],
                    device=self.device,
                )
                self._first_impact_substeps += 1
                if self._first_impact_substeps == self._first_impact_recovery_substeps:
                    wp.launch(
                        _measure_bag_min_z,
                        1,
                        [
                            self.bag_particle_start,
                            self.bag_particle_end,
                            self.state_1.particle_q,
                            self.first_impact_recovery_z,
                        ],
                        device=self.device,
                    )
                    self._track_first_impact = False
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        assert np.all(np.isfinite(bag_q)), "Bag particle positions contain non-finite values"
        bag_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in bag_q])
        bag_min_z = float(bag_scene_q[:, 2].min())
        body_q = self.state_0.body_q.numpy()[self.object_bodies]
        assert np.all(np.isfinite(body_q)), "Rigid object states contain non-finite values"

        script_frames = int(
            np.ceil(sum(segment[0] for segment in self.segments) / (self.frame_dt * self.args.trajectory_time_scale))
        )
        if self.frame_index < script_frames:
            return

        released = self.object_released
        assert np.all(released == 1), f"Not all rigid objects were released: {released.tolist()}"
        maximum_soft_contacts = int(self.maximum_soft_contact_count.numpy()[0])
        assert maximum_soft_contacts <= self.contacts.soft_contact_max, (
            f"Soft-contact buffer overflowed: {maximum_soft_contacts} > {self.contacts.soft_contact_max}"
        )
        maximum_body_particle_contacts = int(self.maximum_body_particle_contact_count.numpy()[0])
        assert maximum_body_particle_contacts <= RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE, (
            "Per-body soft-contact buffer overflowed: "
            f"{maximum_body_particle_contacts} > {RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
        )
        baseline_z = float(self.first_impact_baseline_z.numpy()[0])
        impact_min_z = float(self.first_impact_min_z.numpy()[0])
        recovery_z = float(self.first_impact_recovery_z.numpy()[0])
        impact_drop = baseline_z - impact_min_z
        assert impact_drop > 0.015, f"Heavy box did not visibly deform the bag: drop={impact_drop:.4f} m"
        assert recovery_z - impact_min_z > 0.5 * impact_drop, (
            "Bag did not recover enough after the heavy-box impact: "
            f"baseline={baseline_z:.4f}, peak={impact_min_z:.4f}, recovery={recovery_z:.4f}"
        )
        inside = 0
        positions = []
        for transform in body_q:
            position = self._scene_vec(wp.vec3(*transform[:3]))
            positions.append(tuple(float(value) for value in position))
            if (
                abs(float(position[0]) - float(BAG_POS[0])) < 0.5 * BAG_WIDTH + SHAPE_SIZE
                and abs(float(position[1]) - float(BAG_POS[1])) < 0.5 * BAG_DEPTH + SHAPE_SIZE
                and bag_min_z - SHAPE_SIZE < float(position[2]) < TABLE_TOP_Z + 0.08
            ):
                inside += 1
        assert inside == len(self.object_bodies), (
            f"Only {inside}/{len(self.object_bodies)} rigid objects settled in the bag; positions={positions}"
        )

    def _joint_limits(self):
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

    def _right_hand_q(self):
        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        indices = []
        open_q = []
        grasp_q = []
        targets = {
            "HAND_THUMB2": 0.82,
            "HAND_THUMB1": 0.52,
            "HAND_INDEX": 0.62,
            "INDEX_PIP": 0.82,
            "HAND_MIDDLE": 0.50,
            "MIDDLE_PIP": 0.82,
            "HAND_RING": 0.48,
            "RING_PIP": 0.82,
            "HAND_PINKY": 0.52,
            "PINKY_PIP": 0.82,
        }
        for suffix in self.HAND_SUFFIXES:
            joint = self._joint_index(f"RIGHT_{suffix}")
            index = int(q_start[joint])
            indices.append(index)
            open_q.append(q[index])
            grasp_q.append(targets.get(suffix, q[index]))
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(open_q, dtype=wp.float32, device=self.device),
            wp.array(grasp_q, dtype=wp.float32, device=self.device),
        )

    def _joint_index(self, name):
        return next(index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels, name):
        return next(index for index, label in enumerate(labels) if label.endswith("/" + name))

    def _tcp(self, state, body):
        body_tf = wp.transform(*state.body_q.numpy()[body])
        body_rot = wp.transform_get_rotation(body_tf)
        return wp.transform(
            wp.transform_get_translation(body_tf) + wp.quat_rotate(body_rot, TCP_OFFSET),
            body_rot,
        )

    def _sample(self, time):
        for duration, left_a, left_b, right_a, right_b, grip_a, grip_b, object_index in self.segments:
            if time <= duration:
                alpha = float(np.clip(time / duration, 0.0, 1.0))
                return (
                    self._lerp_tf(left_a, left_b, alpha),
                    self._lerp_tf(right_a, right_b, alpha),
                    grip_a * (1.0 - alpha) + grip_b * alpha,
                    object_index,
                )
            time -= duration
        _, _, left, _, right, _, grip, object_index = self.segments[-1]
        return left, right, grip, object_index

    def _world_vec(self, value):
        rotated = wp.quat_rotate(self.base_rot, value)
        return wp.vec3(
            float(rotated[0]) + float(self.base_pos[0]),
            float(rotated[1]) + float(self.base_pos[1]),
            float(rotated[2]) + float(self.base_pos[2]),
        )

    def _scene_vec(self, value):
        relative = wp.vec3(
            float(value[0]) - float(self.base_pos[0]),
            float(value[1]) - float(self.base_pos[1]),
            float(value[2]) - float(self.base_pos[2]),
        )
        return wp.quat_rotate(wp.quat_inverse(self.base_rot), relative)

    def _world_tf(self, transform):
        return wp.transform(
            self._world_vec(wp.transform_get_translation(transform)),
            self._quat_mul(self.base_rot, wp.transform_get_rotation(transform)),
        )

    def _attach_house_usd(self):
        if not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.isfile(self.house_visual_usd):
            print(f"WAIC house USD not found; continuing without it: {self.house_visual_usd}")
            return
        prim = self.viewer.stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    @staticmethod
    def _normal_quat(value):
        array = np.asarray([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
        array /= max(np.linalg.norm(array), 1.0e-8)
        return wp.quat(*array)

    @staticmethod
    def _quat_mul(a, b):
        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _v4(value):
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def _lerp_tf(a, b, alpha):
        pa = np.asarray(wp.transform_get_translation(a))
        pb = np.asarray(wp.transform_get_translation(b))
        qa = np.asarray(wp.transform_get_rotation(a))
        qb = np.asarray(wp.transform_get_rotation(b))
        if np.dot(qa, qb) < 0.0:
            qb = -qb
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
        if dot > 0.9995:
            quat = qa * (1.0 - alpha) + qb * alpha
            quat /= np.linalg.norm(quat)
        else:
            theta = np.arccos(dot)
            quat = qa * (np.sin((1.0 - alpha) * theta) / np.sin(theta)) + qb * (np.sin(alpha * theta) / np.sin(theta))
        position = pa * (1.0 - alpha) + pb * alpha
        return wp.transform(wp.vec3(*position), wp.quat(*quat))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # The physical five-finger trajectory is deliberately slower than the
        # reference demo's kinematically attached object motion.
        # keep one additional second for the fifth object to settle in the bag.
        parser.set_defaults(num_frames=1500)
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
            help="Render the physics table collider.",
        )
        parser.add_argument("--trajectory-time-scale", type=float, default=2.0)
        parser.add_argument(
            "--impact-height",
            type=float,
            default=DEFAULT_IMPACT_HEIGHT,
            help="Vertical release height above the bag rim [m].",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(WAIC_ROBOT_BASE_QUAT[3]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
