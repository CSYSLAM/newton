# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 moves one volumetric soft cube from a table into a suspended soft bag.

One volumetric soft cube starts on the table. Physical five-finger contact picks it up, moves it over
a suspended open-topped cloth bag, and releases it into the bag.
The bag matches the material and resolution of
example_vbd_soft_rigid_mix_contact.py and has a pinned top rim.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMJVBDV2

FPS = 60
SIM_SUBSTEPS = 5
VBD_ITERATIONS = 24
IK_ITERATIONS = 24

TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

# Narrow the pinch direction so the thumb can oppose the four fingers on one
# side without requiring the hand to squeeze through a full cube.
CUBE_COUNT = 1
CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
CUBE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
CUBE_MARGIN = 0.001
CUBE_POSITIONS = (wp.vec3(0.48, -0.20, TABLE_TOP_Z + CUBE_HALF_EXTENTS[2] + CUBE_MARGIN),)
SOFT_CUBE_DIMS = (6, 4, 6)
SOFT_CUBE_DENSITY = 300.0
SOFT_CUBE_K_MU = 3.0e5
SOFT_CUBE_K_LAMBDA = 1.0e6
SOFT_CUBE_K_DAMP = 15.0
SOFT_CUBE_PARTICLE_RADIUS = 0.0025
SOFT_CUBE_TRANSPORT_DURATION = 14.0
SOFT_CUBE_RELEASE_OPEN_DURATION = 0.25
SOFT_CUBE_RELEASE_SETTLE_DURATION = 0.90
GRASP_PRE_CLOSE_HOLD_DURATION = 0.35
GRASP_CLOSE_DURATION = 1.80

# Match example_vbd_soft_rigid_mix_contact.py.
BAG_WIDTH = 0.20
BAG_DEPTH = 0.16
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
SOFT_CONTACT_MARGIN = 0.003
SOFT_CONTACT_KE = 5.0e3
SOFT_CONTACT_KD = 5.0e-2
SOFT_CONTACT_MU = 0.25
GRASP_CONTACT_KE = 1.5e4
GRASP_CONTACT_KD = 0.2
GRASP_FRICTION = 40
GRASP_SOFT_CONTACT_MU = 6.0
RELEASE_CONTACT_KE = 5.0e3
RELEASE_CONTACT_KD = 0.0
RELEASE_FRICTION = 0.0
DEFAULT_IMPACT_HEIGHT = 0.16
MIN_BAG_RELEASE_HEIGHT = 0.16
DEFAULT_RECORDED_GRASP_Z_OFFSET = 0.0
GRASP_TCP_X_OFFSET = -0.03
GRASP_TCP_Z_OFFSET = 0.02
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
DEFAULT_RECORDED_HAND_POSE = Path(__file__).resolve().parents[3] / "vbd_mjvbd_v2_hand_pose.json"


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
        self.recorded_hand_pose = self._load_recorded_hand_pose(args.recorded_hand_pose)
        self.recorded_grasp_tf = self._recorded_grasp_tf()

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
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": max(BAG_PARTICLE_RADIUS, SOFT_CUBE_PARTICLE_RADIUS),
                "particle_self_contact_margin": 2.0 * max(BAG_PARTICLE_RADIUS, SOFT_CUBE_PARTICLE_RADIUS),
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
        # Keep MJVBD-v2's hand-particle candidate pairs allocated, but gate
        # their runtime participation until the recorded grasp point.
        self.hand_particle_collision_enabled = True
        self._set_hand_particle_collision(False)
        self.object_released = np.zeros(CUBE_COUNT, dtype=bool)
        self.previous_grip = 0.0
        self.release_contact_material_applied = False

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

    def _load_recorded_hand_pose(self, path_value):
        """Load the saved right-hand TCP and finger pose when available."""

        if not path_value:
            return None
        path = Path(path_value).expanduser()
        if not path.is_file():
            print(f"Recorded hand pose not found; using the built-in grasp pose: {path}")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pose = payload["pose"]
            right = pose["hands"]["RIGHT"]
            tcp = right["target_tcp_world"]
            position = np.asarray(tcp["position_m"], dtype=np.float32)
            quaternion = np.asarray(tcp["quaternion_xyzw"], dtype=np.float32)
            joints = {name: float(value) for name, value in right["target_joints_radians"].items()}
        except (OSError, KeyError, TypeError, ValueError) as error:
            print(f"Could not load recorded hand pose {path}: {error}; using the built-in grasp pose.")
            return None
        if position.shape != (3,) or quaternion.shape != (4,):
            print(f"Recorded hand pose has invalid TCP data: {path}; using the built-in grasp pose.")
            return None
        return {
            "tcp_position_world": wp.vec3(*position),
            "tcp_rotation_world": self._normal_quat(wp.quat(*quaternion)),
            "joints": joints,
        }

    def _recorded_grasp_tf(self):
        if self.recorded_hand_pose is None:
            return None
        base_inverse = wp.quat_inverse(self.base_rot)
        world_position = self.recorded_hand_pose["tcp_position_world"]
        local_position = wp.quat_rotate(base_inverse, world_position - self.base_pos)
        local_position += wp.vec3(0.0, 0.0, self.args.recorded_grasp_z_offset)
        local_rotation = self._normal_quat(self._quat_mul(base_inverse, self.recorded_hand_pose["tcp_rotation_world"]))
        return wp.transform(local_position, local_rotation)

    def _offset_recorded_grasp_tf(self, dz: float):
        position = wp.transform_get_translation(self.recorded_grasp_tf) + wp.vec3(0.0, 0.0, dz)
        rotation = wp.transform_get_rotation(self.recorded_grasp_tf)
        return wp.transform(position, rotation)

    def _build_scene(self):
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = 2.0e5
        builder.default_shape_cfg.kd = 1.0e-4
        builder.default_shape_cfg.mu = 1.0
        # The soft-contact path samples SDFs for all rigid hand meshes.
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
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
        self.robot_urdf_shape_end = builder.shape_count
        # The URDF fingertip meshes are too sparse for a stable rigid-cube pinch;
        # these invisible pads remain kinematic hand geometry and carry no object state.
        finger_pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=GRASP_CONTACT_KE,
            kd=GRASP_CONTACT_KD,
            mu=GRASP_FRICTION,
            is_visible=False,
        )
        finger_pad_cfg.configure_sdf(force_sdf=True)
        for body_name, half_extents, pad_xform in (
            (
                "right_thumb_dist",
                (0.018, 0.004, 0.008),
                wp.transform(
                    wp.vec3(-0.0548988, 0.0529312, 0.0141373),
                    wp.quat(0.2773950, -0.8700770, 0.0411925, 0.4053648),
                ),
            ),
            (
                "right_index_dist",
                (0.018, 0.004, 0.008),
                wp.transform(
                    wp.vec3(-0.0388362, 0.0073618, -0.0145817),
                    wp.quat(0.0581093, -0.6604385, 0.7410694, 0.1061119),
                ),
            ),
            (
                "right_middle_dist",
                (0.010, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0004232, 0.0163224, 0.0208014),
                    wp.quat(0.0982183, -0.9514985, 0.2900473, 0.0295879),
                ),
            ),
            (
                "right_ring_dist",
                (0.010, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0013989, 0.0131392, 0.0240774),
                    wp.quat(0.0912622, -0.9611189, 0.2605852, -0.0040627),
                ),
            ),
            (
                "right_pinky_dist",
                (0.010, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0025118, 0.0185027, 0.0180359),
                    wp.quat(0.0746882, -0.9661100, 0.2468487, -0.0108671),
                ),
            ),
        ):
            body = self._body_index(builder.body_label, body_name)
            builder.add_shape_box(
                body,
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=finger_pad_cfg,
                xform=pad_xform,
                label=f"{body_name}_physical_pad",
            )
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

        # Use the same narrow footprint as the earlier rigid cube, but model it as
        # a volumetric tetrahedral body so the hand and bag interact with particles.
        soft_position = CUBE_POSITIONS[0]
        soft_half_extents = wp.vec3(*CUBE_HALF_EXTENTS)
        soft_origin = soft_position - wp.quat_rotate(CUBE_ROTATION, soft_half_extents)
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=self._world_vec(soft_origin),
            rot=self._quat_mul(self.base_rot, CUBE_ROTATION),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=SOFT_CUBE_DIMS[0],
            dim_y=SOFT_CUBE_DIMS[1],
            dim_z=SOFT_CUBE_DIMS[2],
            cell_x=2.0 * CUBE_HALF_EXTENTS[0] / SOFT_CUBE_DIMS[0],
            cell_y=2.0 * CUBE_HALF_EXTENTS[1] / SOFT_CUBE_DIMS[1],
            cell_z=2.0 * CUBE_HALF_EXTENTS[2] / SOFT_CUBE_DIMS[2],
            density=SOFT_CUBE_DENSITY,
            k_mu=SOFT_CUBE_K_MU,
            k_lambda=SOFT_CUBE_K_LAMBDA,
            k_damp=SOFT_CUBE_K_DAMP,
            particle_radius=SOFT_CUBE_PARTICLE_RADIUS,
            label="pick_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.hand_shapes = []
        self.right_hand_shapes = []
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        # Keep the asset colliders and small pads as physical boundaries.  The
        # pads remain rigid-only helpers; the soft body must interact with the
        # actual URDF finger surfaces instead of an enclosing box.
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            hand_shape = any(side in label for side in ("left", "right")) and any(
                word in label for word in self.HAND_CONTACT_KEYWORDS
            )
            if hand_shape:
                if shape < self.robot_urdf_shape_end:
                    self.hand_shapes.append(shape)
                    if "right" in label:
                        self.right_hand_shapes.append(shape)
                    builder.shape_flags[shape] |= collide_shapes | collide_particles
                else:
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
        self.model.soft_contact_mu = GRASP_SOFT_CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[self.hand_shapes] = GRASP_FRICTION
        shape_kd[self.hand_shapes] = GRASP_CONTACT_KD
        shape_ke[self.right_hand_shapes] = GRASP_CONTACT_KE
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_ke.assign(shape_ke)

    def _set_hand_particle_collision(self, enabled: bool):
        """Enable or disable runtime hand-to-particle contacts."""
        if enabled == self.hand_particle_collision_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_collision_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[self.hand_shapes] |= particle_collision_flag
        else:
            flags[self.hand_shapes] &= ~particle_collision_flag
        self.model.shape_flags.assign(flags)
        self.hand_particle_collision_enabled = enabled

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
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self.recorded_grasp_tf
        position = CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + GRASP_TCP_Z_OFFSET,
            ),
            RIGHT_GRASP_ROT,
        )

    def _object_approach_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.18)
        position = CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + 0.18,
            ),
            RIGHT_GRASP_ROT,
        )

    def _object_lift_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.10)
        position = CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + 0.10,
            ),
            RIGHT_GRASP_ROT,
        )

    def _segments(self):
        world = self._world_tf
        if self.recorded_grasp_tf is None:
            grasp_rotation = RIGHT_GRASP_ROT
        else:
            grasp_rotation = wp.transform_get_rotation(self.recorded_grasp_tf)
        if self.recorded_grasp_tf is not None:
            tcp_object_offset = wp.transform_get_translation(self.recorded_grasp_tf) - CUBE_POSITIONS[0]
        else:
            tcp_object_offset = wp.vec3(GRASP_TCP_X_OFFSET, 0.0, GRASP_TCP_Z_OFFSET)
        bag_target_x = float(BAG_POS[0]) + float(tcp_object_offset[0])
        bag_target_y = float(BAG_POS[1]) + float(tcp_object_offset[1])
        bag_release_height = max(self.args.impact_height, MIN_BAG_RELEASE_HEIGHT)
        bag_prep_duration = BAG_PREP_FRAMES * self.frame_dt * self.args.trajectory_time_scale
        segments = [
            (bag_prep_duration + 0.35, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0, -1)
        ]
        right_start = self.right_home
        hand_collision_enable_time = None
        bag_transport = world(
            wp.transform(
                wp.vec3(
                    bag_target_x,
                    bag_target_y,
                    TABLE_TOP_Z + bag_release_height,
                ),
                grasp_rotation,
            )
        )
        bag_hover = world(
            wp.transform(
                wp.vec3(
                    bag_target_x,
                    bag_target_y,
                    TABLE_TOP_Z + bag_release_height,
                ),
                grasp_rotation,
            )
        )
        retreat = world(
            wp.transform(
                wp.vec3(
                    bag_target_x,
                    bag_target_y,
                    TABLE_TOP_Z + bag_release_height + 0.10,
                ),
                grasp_rotation,
            )
        )
        for object_index in range(CUBE_COUNT):
            approach = world(self._object_approach_tf(object_index))
            grasp = world(self._object_grasp_tf(object_index))
            lift = world(self._object_lift_tf(object_index))
            # Descend with the asset's default wrist pose and default open
            # fingers.  Introduce the recorded wrist pose only after the TCP
            # reaches the grasp point, so the approach cannot press the soft cube.
            open_rotation = wp.transform_get_rotation(self.right_home)
            open_approach = wp.transform(wp.transform_get_translation(approach), open_rotation)
            open_grasp = wp.transform(wp.transform_get_translation(grasp), open_rotation)
            if object_index == 0:
                # Enable contact at the end of the wrist-pose transition,
                # before the separate open-hand settling interval.
                hand_collision_enable_time = sum(segment[0] for segment in segments) + 0.80 + 1.20 + 0.20 + 0.35
            segments.extend(
                (
                    (0.80, self.left_home, self.left_home, right_start, open_approach, 0.0, 0.0, object_index),
                    # Descend to the target TCP with the default wrist pose and
                    # default open fingers.
                    (
                        1.20,
                        self.left_home,
                        self.left_home,
                        open_approach,
                        open_grasp,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    # Stop at the target before changing the wrist pose.
                    (
                        0.20,
                        self.left_home,
                        self.left_home,
                        open_grasp,
                        open_grasp,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    # Switch to the recorded wrist pose with the fingers still
                    # open and hand-particle collision still disabled.
                    (
                        0.35,
                        self.left_home,
                        self.left_home,
                        open_grasp,
                        grasp,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    (
                        GRASP_PRE_CLOSE_HOLD_DURATION,
                        self.left_home,
                        self.left_home,
                        grasp,
                        grasp,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    (
                        GRASP_CLOSE_DURATION,
                        self.left_home,
                        self.left_home,
                        grasp,
                        grasp,
                        0.0,
                        1.0,
                        object_index,
                    ),
                    (0.60, self.left_home, self.left_home, grasp, grasp, 1.0, 1.0, object_index),
                    (2.00, self.left_home, self.left_home, grasp, lift, 1.0, 1.0, object_index),
                    (0.60, self.left_home, self.left_home, lift, lift, 1.0, 1.0, object_index),
                    (
                        SOFT_CUBE_TRANSPORT_DURATION,
                        self.left_home,
                        self.left_home,
                        lift,
                        bag_transport,
                        1.0,
                        1.0,
                        object_index,
                    ),
                    # The transport already ends at the release height.  Do
                    # not lower a closed hand to the bag rim, which makes the
                    # soft cube slide down the fingers instead of dropping.
                    (0.20, self.left_home, self.left_home, bag_transport, bag_hover, 1.0, 1.0, object_index),
                    (0.40, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, object_index),
                    # Stop over the opening, open all five fingers quickly, then
                    # keep the hand still so the cube drops vertically.
                    (
                        SOFT_CUBE_RELEASE_OPEN_DURATION,
                        self.left_home,
                        self.left_home,
                        bag_hover,
                        bag_hover,
                        1.0,
                        0.0,
                        object_index,
                    ),
                    (
                        SOFT_CUBE_RELEASE_SETTLE_DURATION,
                        self.left_home,
                        self.left_home,
                        bag_hover,
                        bag_hover,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    (1.00, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, object_index),
                )
            )
            segments.append((1.20, self.left_home, self.left_home, retreat, self.right_home, 0.0, 0.0, -1))
            right_start = self.right_home
        self.hand_collision_enable_time = float(hand_collision_enable_time or 0.0)
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
        hand_collision = np.zeros(self.cached_frame_count + 1, dtype=bool)
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
            # Interpolate to the recorded five-finger pose only during closure.
            cache[frame, hand_indices] = hand_open * (1.0 - grip) + hand_grasp * grip
            grips[frame] = grip
            objects[frame] = object_index
            hand_collision[frame] = script_time >= self.hand_collision_enable_time

        self.cached_joint_targets = wp.array(cache, dtype=wp.float32, device=self.model.device)
        self.cached_grips = grips
        self.cached_objects = objects
        self.cached_hand_collision = hand_collision

    def _prepare_frame(self):
        cache_index = min(self.frame_index + 1, self.cached_frame_count)
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.cached_joint_targets[cache_index])
        grip = float(self.cached_grips[cache_index])
        script_object = int(self.cached_objects[cache_index])
        if self.cached_hand_collision[cache_index] and not self.hand_particle_collision_enabled:
            # Let the open hand settle against the soft body before applying
            # any finger closure, so a small recorded-pose overlap is resolved
            # over several physical frames instead of one closing impulse.
            self._set_hand_particle_collision(True)
        if self.previous_grip > 1.0e-4 and grip <= 1.0e-4 and script_object >= 0:
            self.object_released[script_object] = True
        if self.previous_grip > 0.99 and grip < 0.99 and not self.release_contact_material_applied:
            shape_mu = self.model.shape_material_mu.numpy()
            shape_ke = self.model.shape_material_ke.numpy()
            shape_kd = self.model.shape_material_kd.numpy()
            shape_mu[self.right_hand_shapes] = RELEASE_FRICTION
            shape_ke[self.right_hand_shapes] = RELEASE_CONTACT_KE
            shape_kd[self.right_hand_shapes] = RELEASE_CONTACT_KD
            self.model.shape_material_mu.assign(shape_mu)
            self.model.shape_material_ke.assign(shape_ke)
            self.model.shape_material_kd.assign(shape_kd)
            # The cube is a particle body, so its release friction comes from
            # the global soft-contact material rather than a shape material.
            self.model.soft_contact_ke = RELEASE_CONTACT_KE
            self.model.soft_contact_kd = RELEASE_CONTACT_KD
            self.model.soft_contact_mu = RELEASE_FRICTION
            self.release_contact_material_applied = True
        self.previous_grip = grip

    def simulate(self):
        self._prepare_frame()
        bag_dz = 0.0
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
        """Verify that the soft cube is finite, released, and inside the bag."""

        bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        assert np.all(np.isfinite(bag_q)), "Bag particle positions contain non-finite values"
        bag_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in bag_q])
        bag_min_z = float(bag_scene_q[:, 2].min())
        soft_q = self.state_0.particle_q.numpy()[self.soft_cube_particle_start : self.soft_cube_particle_end]
        assert np.all(np.isfinite(soft_q)), "Soft cube particle positions contain non-finite values"
        soft_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in soft_q])

        script_frames = int(
            np.ceil(sum(segment[0] for segment in self.segments) / (self.frame_dt * self.args.trajectory_time_scale))
        )
        if self.frame_index < script_frames:
            return

        released = self.object_released
        assert np.all(released == 1), f"Soft cube was not released: {released.tolist()}"
        maximum_soft_contacts = int(self.maximum_soft_contact_count.numpy()[0])
        assert maximum_soft_contacts <= self.contacts.soft_contact_max, (
            f"Soft-contact buffer overflowed: {maximum_soft_contacts} > {self.contacts.soft_contact_max}"
        )
        maximum_body_particle_contacts = int(self.maximum_body_particle_contact_count.numpy()[0])
        assert maximum_body_particle_contacts <= RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE, (
            "Per-body soft-contact buffer overflowed: "
            f"{maximum_body_particle_contacts} > {RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
        )
        soft_min = soft_scene_q.min(axis=0)
        soft_max = soft_scene_q.max(axis=0)
        soft_center = soft_scene_q.mean(axis=0)
        bag_center = np.asarray((float(BAG_POS[0]), float(BAG_POS[1])))
        bag_half_extents = np.asarray((0.5 * BAG_WIDTH, 0.5 * BAG_DEPTH))
        soft_inside = (
            np.all(soft_min[:2] > bag_center - bag_half_extents - 0.02)
            and np.all(soft_max[:2] < bag_center + bag_half_extents + 0.02)
            and soft_min[2] > bag_min_z - 0.08
            and soft_max[2] < TABLE_TOP_Z + 0.08
        )
        assert soft_inside, (
            f"Soft cube did not settle in the bag; center={tuple(soft_center)} bounds=({soft_min}, {soft_max})"
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
        targets = self.recorded_hand_pose["joints"] if self.recorded_hand_pose is not None else {}
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
                # Zero the velocity at every waypoint.  This removes the
                # tangential impulse when horizontal transport starts and ends.
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
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
        # Leave enough physical time for contact to settle after each placement.
        parser.set_defaults(num_frames=1500)
        parser.add_argument("--robot-urdf", default=None, help="Optional Dexforce W1 URDF path.")
        parser.add_argument(
            "--house-visual-usd",
            default=DEFAULT_HOUSE_USD,
            help="Optional WAIC house USD reference; it is visual-only.",
        )
        parser.add_argument(
            "--recorded-hand-pose",
            default=str(DEFAULT_RECORDED_HAND_POSE),
            help="Saved right-hand TCP/finger pose used for the grasp, when present.",
        )
        parser.add_argument(
            "--recorded-grasp-z-offset",
            type=float,
            default=DEFAULT_RECORDED_GRASP_Z_OFFSET,
            help="Additional recorded grasp TCP height correction [m].",
        )
        parser.add_argument(
            "--show-physics-table",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render the physics table collider.",
        )
        parser.add_argument("--trajectory-time-scale", type=float, default=1.0)
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
