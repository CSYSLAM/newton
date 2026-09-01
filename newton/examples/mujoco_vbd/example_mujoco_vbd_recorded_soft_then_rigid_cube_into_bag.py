# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415
"""Standalone MuJoCo/VBD acceptance demo.

This file owns its complete scene construction and trajectory implementation;
it does not import another example or a scene-specific helper module.
"""

from __future__ import annotations


class _LocalModule:
    """Expose one flattened source block through its original module API."""

    def __init__(self, prefix: str, **modules):
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "modules", modules)

    def __getattr__(self, name):
        if name in self.modules:
            return self.modules[name]
        return globals()[self.prefix + name]

    def __setattr__(self, name, value):
        globals()[self.prefix + name] = value


recorded_soft = _LocalModule("_m2_", soft0=_LocalModule("_m1_"))
sequential_base = _LocalModule("_m7_", recorder=_LocalModule("_m5_"))
import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCoVBD

_m0_FPS = 60
_m0_SIM_SUBSTEPS = 5
_m0_VBD_ITERATIONS = 24
_m0_IK_ITERATIONS = 24
_m0_TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
_m0_TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
_m0_TABLE_TOP_Z = float(_m0_TABLE_POS[2]) + _m0_TABLE_HALF_EXTENTS[2]
_m0_TABLE_COLOR = (0.35, 0.42, 0.48)
_m0_CUBE_COUNT = 1
_m0_CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
_m0_CUBE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
_m0_CUBE_MARGIN = 0.001
_m0_CUBE_DENSITY = 1500.0
_m0_CUBE_POSITIONS = (wp.vec3(0.48, -0.2, _m0_TABLE_TOP_Z + _m0_CUBE_HALF_EXTENTS[2] + _m0_CUBE_MARGIN),)
_m0_CUBE_COLORS = ((0.9, 0.32, 0.18),)
_m0_BAG_WIDTH = 0.2
_m0_BAG_DEPTH = 0.16
_m0_BAG_HEIGHT = 0.24
_m0_BAG_TABLE_GAP = 0.06
_m0_BAG_SETTLE_FRAMES = 120
_m0_BAG_PREP_FRAMES = _m0_BAG_SETTLE_FRAMES
_m0_BAG_READY_BASE_Z = _m0_TABLE_TOP_Z - _m0_BAG_HEIGHT
_m0_BAG_POS = wp.vec3(
    0.448, -(_m0_TABLE_HALF_EXTENTS[1] + _m0_BAG_TABLE_GAP + 0.5 * _m0_BAG_DEPTH), _m0_BAG_READY_BASE_Z
)
_m0_BAG_RESOLUTION = 20
_m0_BAG_PARTICLE_RADIUS = 0.003
_m0_BAG_DENSITY = 0.08
_m0_BAG_TRI_KE = 150.0
_m0_BAG_TRI_KA = 150.0
_m0_BAG_TRI_KD = 0.5
_m0_BAG_EDGE_KE = 0.5
_m0_BAG_EDGE_KD = 1.5e-05
_m0_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
_m0_SOFT_CONTACT_MARGIN = 0.01
_m0_SOFT_CONTACT_KE = 5000.0
_m0_SOFT_CONTACT_KD = 0.05
_m0_SOFT_CONTACT_MU = 0.25
_m0_GRASP_CONTACT_KE = 30000.0
_m0_GRASP_CONTACT_KD = 0.5
_m0_GRASP_FRICTION = 25
_m0_RELEASE_CONTACT_KE = 5000.0
_m0_RELEASE_CONTACT_KD = 0.0
_m0_RELEASE_FRICTION = 0.0
_m0_DEFAULT_IMPACT_HEIGHT = 0.15
_m0_DEFAULT_RECORDED_GRASP_Z_OFFSET = 0.0
_m0_GRASP_TCP_X_OFFSET = -0.03
_m0_GRASP_TCP_Z_OFFSET = 0.02
_m0_BAG_TCP_X_OFFSET = 0.06
_m0_BAG_TCP_Y_OFFSET = -0.077
_m0_TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
_m0_RIGHT_GRASP_ROT = wp.quat(-0.0733, 0.7031, 0.7037, -0.0717)
_m0_WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
_m0_WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m0_CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
_m0_CAMERA_PITCH = -18.0
_m0_CAMERA_YAW = 126.0
_m0_DEFAULT_HOUSE_USD = "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
_m0_DEFAULT_RECORDED_HAND_POSE = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_mjvbd_v2_hand_pose.json"
)


def _m0__generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
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
    return (np.asarray(vertices, dtype=np.float32), indices)


@wp.kernel
def _m0__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m0__joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _m0__lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _m0__lift_pinned_vertices(
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
def _m0__accumulate_contact_diagnostics(
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


class _m0_Example:
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
        self.frame_dt = 1.0 / _m0_FPS
        self.sim_substeps = _m0_SIM_SUBSTEPS
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
        self.bag_pinned_original = wp.array(particle_q[self.bag_top_indices].copy(), dtype=wp.vec3, device=self.device)
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m0_VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": _m0_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": _m0_BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": 2.0 * _m0_BAG_PARTICLE_RADIUS,
                "particle_topological_contact_filter_threshold": 3,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": _m0_SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )
        self.contacts = self.solver.contacts
        self.maximum_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.maximum_body_particle_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.object_released = np.zeros(_m0_CUBE_COUNT, dtype=bool)
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
            self.viewer.set_camera(_m0_CAMERA_POS, _m0_CAMERA_PITCH, _m0_CAMERA_YAW)

    def _robot_urdf(self) -> Path:
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = Path(__file__).resolve().parents[3] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"
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
        builder.default_shape_cfg.ke = 200000.0
        builder.default_shape_cfg.kd = 0.0001
        builder.default_shape_cfg.mu = 1.0
        SolverMuJoCoVBD.register_custom_attributes(builder)
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
        finger_pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_m0_GRASP_CONTACT_KE, kd=_m0_GRASP_CONTACT_KD, mu=_m0_GRASP_FRICTION, is_visible=False
        )
        for body_name, half_extents, pad_xform in (
            (
                "right_thumb_dist",
                (0.03, 0.006, 0.015),
                wp.transform(
                    wp.vec3(-0.0548988, 0.0529312, 0.0141373), wp.quat(0.277395, -0.870077, 0.0411925, 0.4053648)
                ),
            ),
            (
                "right_index_dist",
                (0.03, 0.006, 0.015),
                wp.transform(
                    wp.vec3(-0.0388362, 0.0073618, -0.0145817), wp.quat(0.0581093, -0.6604385, 0.7410694, 0.1061119)
                ),
            ),
            (
                "right_middle_dist",
                (0.014, 0.006, 0.012),
                wp.transform(
                    wp.vec3(-0.0004232, 0.0163224, 0.0208014), wp.quat(0.0982183, -0.9514985, 0.2900473, 0.0295879)
                ),
            ),
            (
                "right_ring_dist",
                (0.014, 0.006, 0.012),
                wp.transform(
                    wp.vec3(-0.0013989, 0.0131392, 0.0240774), wp.quat(0.0912622, -0.9611189, 0.2605852, -0.0040627)
                ),
            ),
            (
                "right_pinky_dist",
                (0.014, 0.006, 0.012),
                wp.transform(
                    wp.vec3(-0.0025118, 0.0185027, 0.0180359), wp.quat(0.0746882, -0.96611, 0.2468487, -0.0108671)
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
            ke=300000.0, kd=0.0001, mu=0.9, is_visible=bool(self.args.show_physics_table)
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self._world_vec(_m0_TABLE_POS), self.base_rot),
            hx=_m0_TABLE_HALF_EXTENTS[0],
            hy=_m0_TABLE_HALF_EXTENTS[1],
            hz=_m0_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=_m0_TABLE_COLOR,
            label="pick_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="pick_ground")
        bag_vertices, bag_indices = _m0__generate_box_bag(
            0.5 * _m0_BAG_WIDTH, 0.5 * _m0_BAG_DEPTH, _m0_BAG_HEIGHT, _m0_BAG_RESOLUTION
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=self._world_vec(_m0_BAG_POS),
            rot=self.base_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=_m0_BAG_DENSITY,
            tri_ke=_m0_BAG_TRI_KE,
            tri_ka=_m0_BAG_TRI_KA,
            tri_kd=_m0_BAG_TRI_KD,
            edge_ke=_m0_BAG_EDGE_KE,
            edge_kd=_m0_BAG_EDGE_KD,
            particle_radius=_m0_BAG_PARTICLE_RADIUS,
            label="suspended_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        top = np.flatnonzero(np.abs(bag_vertices[:, 2] - _m0_BAG_HEIGHT) < 1e-05)
        self.bag_top_indices = top.astype(np.int32) + self.bag_particle_start
        cfg = newton.ModelBuilder.ShapeConfig(
            density=_m0_CUBE_DENSITY,
            ke=_m0_SOFT_CONTACT_KE,
            kd=_m0_SOFT_CONTACT_KD,
            mu=_m0_SOFT_CONTACT_MU,
            margin=_m0_CUBE_MARGIN,
        )
        cfg.configure_sdf(force_sdf=True)
        cfg.has_particle_collision = True
        self.object_bodies = []
        self.object_shapes = []
        for object_index, (position, color) in enumerate(zip(_m0_CUBE_POSITIONS, _m0_CUBE_COLORS, strict=True)):
            body = builder.add_body(
                xform=wp.transform(self._world_vec(position), self.base_rot), label=f"pick_cube_{object_index}"
            )
            shape = builder.shape_count
            builder.add_shape_box(
                body,
                hx=_m0_CUBE_HALF_EXTENTS[0],
                hy=_m0_CUBE_HALF_EXTENTS[1],
                hz=_m0_CUBE_HALF_EXTENTS[2],
                cfg=cfg,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), _m0_CUBE_ROTATION),
                color=color,
                label=f"pick_cube_{object_index}_shape",
            )
            self.object_bodies.append(body)
            self.object_shapes.append(shape)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.hand_shapes = []
        self.right_hand_shapes = []
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            hand_shape = any(side in label for side in ("left", "right")) and any(
                word in label for word in self.HAND_CONTACT_KEYWORDS
            )
            if hand_shape:
                self.hand_shapes.append(shape)
                if "right" in label:
                    self.right_hand_shapes.append(shape)
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
            else:
                builder.shape_flags[shape] &= ~(collide_shapes | collide_particles)
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m0_SOFT_CONTACT_KE
        self.model.soft_contact_kd = _m0_SOFT_CONTACT_KD
        self.model.soft_contact_mu = _m0_SOFT_CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[self.hand_shapes] = _m0_GRASP_FRICTION
        shape_mu[self.object_shapes] = _m0_GRASP_FRICTION
        shape_kd[self.hand_shapes] = _m0_GRASP_CONTACT_KD
        shape_kd[self.object_shapes] = _m0_GRASP_CONTACT_KD
        shape_ke[self.right_hand_shapes] = _m0_GRASP_CONTACT_KE
        shape_ke[self.object_shapes] = _m0_GRASP_CONTACT_KE
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_kd.assign(shape_kd)
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
            _m0_TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.model.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.model.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            _m0_TCP_OFFSET,
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
        position = _m0_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + _m0_GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + _m0_GRASP_TCP_Z_OFFSET,
            ),
            _m0_RIGHT_GRASP_ROT,
        )

    def _object_approach_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.18)
        position = _m0_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(float(position[0]) + _m0_GRASP_TCP_X_OFFSET, float(position[1]), float(position[2]) + 0.18),
            _m0_RIGHT_GRASP_ROT,
        )

    def _object_lift_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.1)
        position = _m0_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(float(position[0]) + _m0_GRASP_TCP_X_OFFSET, float(position[1]), float(position[2]) + 0.1),
            _m0_RIGHT_GRASP_ROT,
        )

    def _segments(self):
        world = self._world_tf
        if self.recorded_grasp_tf is None:
            grasp_rotation = _m0_RIGHT_GRASP_ROT
        else:
            grasp_rotation = wp.transform_get_rotation(self.recorded_grasp_tf)
        bag_prep_duration = _m0_BAG_PREP_FRAMES * self.frame_dt * self.args.trajectory_time_scale
        segments = [
            (bag_prep_duration + 0.35, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0, -1)
        ]
        right_start = self.right_home
        bag_transport = world(
            wp.transform(
                wp.vec3(
                    float(_m0_BAG_POS[0]) + _m0_BAG_TCP_X_OFFSET + _m0_GRASP_TCP_X_OFFSET,
                    float(_m0_BAG_POS[1]) + _m0_BAG_TCP_Y_OFFSET,
                    _m0_TABLE_TOP_Z + max(self.args.impact_height, 0.2),
                ),
                grasp_rotation,
            )
        )
        bag_hover = world(
            wp.transform(
                wp.vec3(
                    float(_m0_BAG_POS[0]) + _m0_BAG_TCP_X_OFFSET + _m0_GRASP_TCP_X_OFFSET,
                    float(_m0_BAG_POS[1]) + _m0_BAG_TCP_Y_OFFSET,
                    _m0_TABLE_TOP_Z + self.args.impact_height,
                ),
                grasp_rotation,
            )
        )
        retreat = world(
            wp.transform(
                wp.vec3(
                    float(_m0_BAG_POS[0]) + _m0_BAG_TCP_X_OFFSET + _m0_GRASP_TCP_X_OFFSET,
                    float(_m0_BAG_POS[1]) + _m0_BAG_TCP_Y_OFFSET,
                    _m0_TABLE_TOP_Z + self.args.impact_height - 0.05,
                ),
                grasp_rotation,
            )
        )
        for object_index in range(len(self.object_bodies)):
            approach = world(self._object_approach_tf(object_index))
            grasp = world(self._object_grasp_tf(object_index))
            lift = world(self._object_lift_tf(object_index))
            segments.extend(
                (
                    (0.8, self.left_home, self.left_home, right_start, approach, 0.0, 0.0, object_index),
                    (1.8, self.left_home, self.left_home, approach, grasp, 0.0, 1.0, object_index),
                    (1.0, self.left_home, self.left_home, grasp, grasp, 1.0, 1.0, object_index),
                    (2.0, self.left_home, self.left_home, grasp, lift, 1.0, 1.0, object_index),
                    (0.6, self.left_home, self.left_home, lift, lift, 1.0, 1.0, object_index),
                    (5.0, self.left_home, self.left_home, lift, bag_transport, 1.0, 1.0, object_index),
                    (1.5, self.left_home, self.left_home, bag_transport, bag_hover, 1.0, 1.0, object_index),
                    (0.4, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, object_index),
                    (0.8, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 0.0, object_index),
                    (0.2, self.left_home, self.left_home, bag_hover, bag_hover, 0.0, 0.0, object_index),
                    (1.0, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, object_index),
                )
            )
            segments.append((1.2, self.left_home, self.left_home, retreat, self.right_home, 0.0, 0.0, -1))
            right_start = self.right_home
        segments.append((1.2, self.left_home, self.left_home, right_start, right_start, 0.0, 0.0, -1))
        return tuple(segments)

    def _build_joint_target_cache(self):
        script_duration = sum(segment[0] for segment in self.segments)
        script_frames = int(np.ceil(script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        self.cached_frame_count = max(int(self.args.num_frames), script_frames + _m0_FPS)
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
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m0_IK_ITERATIONS)
            wp.launch(
                _m0__lock_q,
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
        if self.previous_grip > 0.0001 and grip <= 0.0001 and (script_object >= 0):
            self.object_released[script_object] = True
        if self.previous_grip > 0.99 and grip < 0.99 and (not self.release_contact_material_applied):
            shape_mu = self.model.shape_material_mu.numpy()
            shape_ke = self.model.shape_material_ke.numpy()
            shape_kd = self.model.shape_material_kd.numpy()
            shape_mu[self.right_hand_shapes] = _m0_RELEASE_FRICTION
            shape_mu[self.object_shapes] = _m0_RELEASE_FRICTION
            shape_ke[self.right_hand_shapes] = _m0_RELEASE_CONTACT_KE
            shape_ke[self.object_shapes] = _m0_RELEASE_CONTACT_KE
            shape_kd[self.right_hand_shapes] = _m0_RELEASE_CONTACT_KD
            shape_kd[self.object_shapes] = _m0_RELEASE_CONTACT_KD
            self.model.shape_material_mu.assign(shape_mu)
            self.model.shape_material_ke.assign(shape_ke)
            self.model.shape_material_kd.assign(shape_kd)
            self.release_contact_material_applied = True
        self.previous_grip = grip

    def simulate(self):
        self._prepare_frame()
        bag_dz = 0.0
        for substep in range(self.sim_substeps):
            wp.launch(
                _m0__lift_pinned_vertices,
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
                _m0__interpolate_q,
                self.ik_model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m0__joint_velocity,
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
                _m0__accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            self.state_0, self.state_1 = (self.state_1, self.state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the red cube is finite, released, and inside the bag."""
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
        assert maximum_body_particle_contacts <= _m0_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE, (
            f"Per-body soft-contact buffer overflowed: {maximum_body_particle_contacts} > {_m0_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
        )
        inside = 0
        positions = []
        for transform in body_q:
            position = self._scene_vec(wp.vec3(*transform[:3]))
            positions.append(tuple(float(value) for value in position))
            if (
                abs(float(position[0]) - float(_m0_BAG_POS[0])) < 0.5 * _m0_BAG_WIDTH + _m0_CUBE_HALF_EXTENTS[0]
                and abs(float(position[1]) - float(_m0_BAG_POS[1])) < 0.5 * _m0_BAG_DEPTH + _m0_CUBE_HALF_EXTENTS[1]
                and (bag_min_z - _m0_CUBE_HALF_EXTENTS[2] < float(position[2]) < _m0_TABLE_TOP_Z + 0.08)
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
                lower[int(qd_start[joint])] = q[int(q_start[joint])] - 0.0001
                upper[int(qd_start[joint])] = q[int(q_start[joint])] + 0.0001
        return (lower[: self.ik_model.joint_dof_count], upper[: self.ik_model.joint_dof_count])

    def _locked_q(self):
        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices = [
            int(q_start[joint])
            for joint, label in enumerate(self.model.joint_label[: self.ik_model.joint_count])
            if label not in controlled
        ]
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.device),
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
        return next((index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name)))

    @staticmethod
    def _body_index(labels, name):
        return next((index for index, label in enumerate(labels) if label.endswith("/" + name)))

    def _tcp(self, state, body):
        body_tf = wp.transform(*state.body_q.numpy()[body])
        body_rot = wp.transform_get_rotation(body_tf)
        return wp.transform(wp.transform_get_translation(body_tf) + wp.quat_rotate(body_rot, _m0_TCP_OFFSET), body_rot)

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
        return (left, right, grip, object_index)

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
        array /= max(np.linalg.norm(array), 1e-08)
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
        parser.set_defaults(num_frames=1500)
        parser.add_argument("--robot-urdf", default=None, help="Optional Dexforce W1 URDF path.")
        parser.add_argument(
            "--house-visual-usd",
            default=_m0_DEFAULT_HOUSE_USD,
            help="Optional WAIC house USD reference; it is visual-only.",
        )
        parser.add_argument(
            "--recorded-hand-pose",
            default=str(_m0_DEFAULT_RECORDED_HAND_POSE),
            help="Saved right-hand TCP/finger pose used for the grasp, when present.",
        )
        parser.add_argument(
            "--recorded-grasp-z-offset",
            type=float,
            default=_m0_DEFAULT_RECORDED_GRASP_Z_OFFSET,
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
            default=_m0_DEFAULT_IMPACT_HEIGHT,
            help="Vertical release height above the bag rim [m].",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(_m0_WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(_m0_WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(_m0_WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(_m0_WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(_m0_WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(_m0_WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(_m0_WAIC_ROBOT_BASE_QUAT[3]))
        return parser


import argparse
import json
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCoVBD

_m1_FPS = 60
_m1_SIM_SUBSTEPS = 5
_m1_VBD_ITERATIONS = 24
_m1_IK_ITERATIONS = 24
_m1_TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
_m1_TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
_m1_TABLE_TOP_Z = float(_m1_TABLE_POS[2]) + _m1_TABLE_HALF_EXTENTS[2]
_m1_TABLE_COLOR = (0.35, 0.42, 0.48)
_m1_CUBE_COUNT = 1
_m1_CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
_m1_CUBE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
_m1_CUBE_MARGIN = 0.001
_m1_CUBE_POSITIONS = (wp.vec3(0.48, -0.2, _m1_TABLE_TOP_Z + _m1_CUBE_HALF_EXTENTS[2] + _m1_CUBE_MARGIN),)
_m1_SOFT_CUBE_DIMS = (6, 4, 6)
_m1_SOFT_CUBE_DENSITY = 300.0
_m1_SOFT_CUBE_K_MU = 300000.0
_m1_SOFT_CUBE_K_LAMBDA = 1000000.0
_m1_SOFT_CUBE_K_DAMP = 15.0
_m1_SOFT_CUBE_PARTICLE_RADIUS = 0.0025
_m1_SOFT_CUBE_TRANSPORT_DURATION = 14.0
_m1_SOFT_CUBE_RELEASE_OPEN_DURATION = 0.25
_m1_SOFT_CUBE_RELEASE_SETTLE_DURATION = 0.9
_m1_GRASP_PRE_CLOSE_HOLD_DURATION = 0.35
_m1_GRASP_CLOSE_DURATION = 1.8
_m1_BAG_WIDTH = 0.2
_m1_BAG_DEPTH = 0.16
_m1_BAG_HEIGHT = 0.24
_m1_BAG_TABLE_GAP = 0.06
_m1_BAG_SETTLE_FRAMES = 120
_m1_BAG_PREP_FRAMES = _m1_BAG_SETTLE_FRAMES
_m1_BAG_READY_BASE_Z = _m1_TABLE_TOP_Z - _m1_BAG_HEIGHT
_m1_BAG_POS = wp.vec3(
    0.448, -(_m1_TABLE_HALF_EXTENTS[1] + _m1_BAG_TABLE_GAP + 0.5 * _m1_BAG_DEPTH), _m1_BAG_READY_BASE_Z
)
_m1_BAG_RESOLUTION = 20
_m1_BAG_PARTICLE_RADIUS = 0.003
_m1_BAG_DENSITY = 0.08
_m1_BAG_TRI_KE = 150.0
_m1_BAG_TRI_KA = 150.0
_m1_BAG_TRI_KD = 0.5
_m1_BAG_EDGE_KE = 0.5
_m1_BAG_EDGE_KD = 1.5e-05
_m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
_m1_SOFT_CONTACT_MARGIN = 0.003
_m1_SOFT_CONTACT_KE = 5000.0
_m1_SOFT_CONTACT_KD = 0.05
_m1_SOFT_CONTACT_MU = 0.25
_m1_GRASP_CONTACT_KE = 15000.0
_m1_GRASP_CONTACT_KD = 0.2
_m1_GRASP_FRICTION = 40
_m1_GRASP_SOFT_CONTACT_MU = 6.0
_m1_RELEASE_CONTACT_KE = 5000.0
_m1_RELEASE_CONTACT_KD = 0.0
_m1_RELEASE_FRICTION = 0.0
_m1_DEFAULT_IMPACT_HEIGHT = 0.16
_m1_MIN_BAG_RELEASE_HEIGHT = 0.16
_m1_DEFAULT_RECORDED_GRASP_Z_OFFSET = 0.0
_m1_GRASP_TCP_X_OFFSET = -0.03
_m1_GRASP_TCP_Z_OFFSET = 0.02
_m1_TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
_m1_RIGHT_GRASP_ROT = wp.quat(-0.0733, 0.7031, 0.7037, -0.0717)
_m1_WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
_m1_WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m1_CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
_m1_CAMERA_PITCH = -18.0
_m1_CAMERA_YAW = 126.0
_m1_DEFAULT_HOUSE_USD = "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
_m1_DEFAULT_RECORDED_HAND_POSE = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_mjvbd_v2_hand_pose.json"
)


def _m1__generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
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
    return (np.asarray(vertices, dtype=np.float32), indices)


@wp.kernel
def _m1__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m1__joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _m1__lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _m1__lift_pinned_vertices(
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
def _m1__accumulate_contact_diagnostics(
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


class _m1_Example:
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
        self.frame_dt = 1.0 / _m1_FPS
        self.sim_substeps = _m1_SIM_SUBSTEPS
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
        self.bag_pinned_original = wp.array(particle_q[self.bag_top_indices].copy(), dtype=wp.vec3, device=self.device)
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options=self._solver_vbd_options(),
            collision_options=self._solver_collision_options(),
            coupling_mode="one_way",
        )
        self.contacts = self.solver.contacts
        self.maximum_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.maximum_body_particle_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.hand_particle_collision_enabled = True
        self._set_hand_particle_collision(False)
        self.object_released = np.zeros(_m1_CUBE_COUNT, dtype=bool)
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
            self.viewer.set_camera(_m1_CAMERA_POS, _m1_CAMERA_PITCH, _m1_CAMERA_YAW)

    def _robot_urdf(self) -> Path:
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = Path(__file__).resolve().parents[3] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"
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

    def _finger_pad_specs(self):
        """Return the invisible rigid-contact pads attached to the right fingertips."""
        return (
            (
                "right_thumb_dist",
                (0.018, 0.004, 0.008),
                wp.transform(
                    wp.vec3(-0.0548988, 0.0529312, 0.0141373), wp.quat(0.277395, -0.870077, 0.0411925, 0.4053648)
                ),
            ),
            (
                "right_index_dist",
                (0.018, 0.004, 0.008),
                wp.transform(
                    wp.vec3(-0.0388362, 0.0073618, -0.0145817), wp.quat(0.0581093, -0.6604385, 0.7410694, 0.1061119)
                ),
            ),
            (
                "right_middle_dist",
                (0.01, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0004232, 0.0163224, 0.0208014), wp.quat(0.0982183, -0.9514985, 0.2900473, 0.0295879)
                ),
            ),
            (
                "right_ring_dist",
                (0.01, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0013989, 0.0131392, 0.0240774), wp.quat(0.0912622, -0.9611189, 0.2605852, -0.0040627)
                ),
            ),
            (
                "right_pinky_dist",
                (0.01, 0.004, 0.007),
                wp.transform(
                    wp.vec3(-0.0025118, 0.0185027, 0.0180359), wp.quat(0.0746882, -0.96611, 0.2468487, -0.0108671)
                ),
            ),
        )

    def _add_additional_scene_objects(self, builder):
        """Add optional objects before finalizing the coupled scene."""

    def _add_additional_finger_pads(self, builder):
        """Add optional stage-specific finger pads before shape filtering."""

    def _solver_vbd_options(self):
        """Return the VBD options for the coupled scene."""
        return {
            "iterations": _m1_VBD_ITERATIONS,
            "rigid_body_contact_buffer_size": 4096,
            "rigid_body_particle_contact_buffer_size": _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": max(_m1_BAG_PARTICLE_RADIUS, _m1_SOFT_CUBE_PARTICLE_RADIUS),
            "particle_self_contact_margin": 2.0 * max(_m1_BAG_PARTICLE_RADIUS, _m1_SOFT_CUBE_PARTICLE_RADIUS),
            "particle_topological_contact_filter_threshold": 3,
        }

    def _solver_collision_options(self):
        """Return the collision-pipeline options for the coupled scene."""
        return {
            "broad_phase": "nxn",
            "soft_contact_margin": _m1_SOFT_CONTACT_MARGIN,
            "enable_rigid_soft_full_surface_contact": True,
        }

    def _build_scene(self):
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = 200000.0
        builder.default_shape_cfg.kd = 0.0001
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
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
        finger_pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_m1_GRASP_CONTACT_KE, kd=_m1_GRASP_CONTACT_KD, mu=_m1_GRASP_FRICTION, is_visible=False
        )
        finger_pad_cfg.configure_sdf(force_sdf=True)
        self.finger_pad_shapes = []
        for body_name, half_extents, pad_xform in self._finger_pad_specs():
            body = self._body_index(builder.body_label, body_name)
            self.finger_pad_shapes.append(
                builder.add_shape_box(
                    body,
                    hx=half_extents[0],
                    hy=half_extents[1],
                    hz=half_extents[2],
                    cfg=finger_pad_cfg,
                    xform=pad_xform,
                    label=f"{body_name}_physical_pad",
                )
            )
        self._add_additional_finger_pads(builder)
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=300000.0, kd=0.0001, mu=0.9, is_visible=bool(self.args.show_physics_table)
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self._world_vec(_m1_TABLE_POS), self.base_rot),
            hx=_m1_TABLE_HALF_EXTENTS[0],
            hy=_m1_TABLE_HALF_EXTENTS[1],
            hz=_m1_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=_m1_TABLE_COLOR,
            label="pick_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="pick_ground")
        bag_vertices, bag_indices = _m1__generate_box_bag(
            0.5 * _m1_BAG_WIDTH, 0.5 * _m1_BAG_DEPTH, _m1_BAG_HEIGHT, _m1_BAG_RESOLUTION
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=self._world_vec(_m1_BAG_POS),
            rot=self.base_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=_m1_BAG_DENSITY,
            tri_ke=_m1_BAG_TRI_KE,
            tri_ka=_m1_BAG_TRI_KA,
            tri_kd=_m1_BAG_TRI_KD,
            edge_ke=_m1_BAG_EDGE_KE,
            edge_kd=_m1_BAG_EDGE_KD,
            particle_radius=_m1_BAG_PARTICLE_RADIUS,
            label="suspended_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        top = np.flatnonzero(np.abs(bag_vertices[:, 2] - _m1_BAG_HEIGHT) < 1e-05)
        self.bag_top_indices = top.astype(np.int32) + self.bag_particle_start
        soft_position = _m1_CUBE_POSITIONS[0]
        soft_half_extents = wp.vec3(*_m1_CUBE_HALF_EXTENTS)
        soft_origin = soft_position - wp.quat_rotate(_m1_CUBE_ROTATION, soft_half_extents)
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=self._world_vec(soft_origin),
            rot=self._quat_mul(self.base_rot, _m1_CUBE_ROTATION),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=_m1_SOFT_CUBE_DIMS[0],
            dim_y=_m1_SOFT_CUBE_DIMS[1],
            dim_z=_m1_SOFT_CUBE_DIMS[2],
            cell_x=2.0 * _m1_CUBE_HALF_EXTENTS[0] / _m1_SOFT_CUBE_DIMS[0],
            cell_y=2.0 * _m1_CUBE_HALF_EXTENTS[1] / _m1_SOFT_CUBE_DIMS[1],
            cell_z=2.0 * _m1_CUBE_HALF_EXTENTS[2] / _m1_SOFT_CUBE_DIMS[2],
            density=_m1_SOFT_CUBE_DENSITY,
            k_mu=_m1_SOFT_CUBE_K_MU,
            k_lambda=_m1_SOFT_CUBE_K_LAMBDA,
            k_damp=_m1_SOFT_CUBE_K_DAMP,
            particle_radius=_m1_SOFT_CUBE_PARTICLE_RADIUS,
            label="pick_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count
        self._add_additional_scene_objects(builder)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.hand_shapes = []
        self.right_hand_shapes = []
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collision_mask = collide_shapes | collide_particles
        self.robot_visual_shapes = []
        for shape in range(self.robot_shape_end):
            is_urdf_shape = shape < self.robot_urdf_shape_end
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if is_urdf_shape and (not is_collider):
                self.robot_visual_shapes.append(shape)
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            hand_shape = any(side in label for side in ("left", "right")) and any(
                word in label for word in self.HAND_CONTACT_KEYWORDS
            )
            if hand_shape and is_collider:
                if is_urdf_shape:
                    self.hand_shapes.append(shape)
                    if "right" in label:
                        self.right_hand_shapes.append(shape)
                    builder.shape_flags[shape] |= collide_shapes | collide_particles
                else:
                    builder.shape_flags[shape] |= collide_shapes
                    builder.shape_flags[shape] &= ~collide_particles
            else:
                builder.shape_flags[shape] &= ~collision_mask
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m1_SOFT_CONTACT_KE
        self.model.soft_contact_kd = _m1_SOFT_CONTACT_KD
        self.model.soft_contact_mu = _m1_GRASP_SOFT_CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[self.hand_shapes] = _m1_GRASP_FRICTION
        shape_kd[self.hand_shapes] = _m1_GRASP_CONTACT_KD
        shape_ke[self.right_hand_shapes] = _m1_GRASP_CONTACT_KE
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
            _m1_TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.model.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.model.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            _m1_TCP_OFFSET,
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
        position = _m1_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(
                float(position[0]) + _m1_GRASP_TCP_X_OFFSET,
                float(position[1]),
                float(position[2]) + _m1_GRASP_TCP_Z_OFFSET,
            ),
            _m1_RIGHT_GRASP_ROT,
        )

    def _object_approach_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.18)
        position = _m1_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(float(position[0]) + _m1_GRASP_TCP_X_OFFSET, float(position[1]), float(position[2]) + 0.18),
            _m1_RIGHT_GRASP_ROT,
        )

    def _object_lift_tf(self, object_index: int):
        if object_index == 0 and self.recorded_grasp_tf is not None:
            return self._offset_recorded_grasp_tf(0.1)
        position = _m1_CUBE_POSITIONS[object_index]
        return wp.transform(
            wp.vec3(float(position[0]) + _m1_GRASP_TCP_X_OFFSET, float(position[1]), float(position[2]) + 0.1),
            _m1_RIGHT_GRASP_ROT,
        )

    def _segments(self):
        world = self._world_tf
        if self.recorded_grasp_tf is None:
            grasp_rotation = _m1_RIGHT_GRASP_ROT
        else:
            grasp_rotation = wp.transform_get_rotation(self.recorded_grasp_tf)
        if self.recorded_grasp_tf is not None:
            tcp_object_offset = wp.transform_get_translation(self.recorded_grasp_tf) - _m1_CUBE_POSITIONS[0]
        else:
            tcp_object_offset = wp.vec3(_m1_GRASP_TCP_X_OFFSET, 0.0, _m1_GRASP_TCP_Z_OFFSET)
        bag_target_x = float(_m1_BAG_POS[0]) + float(tcp_object_offset[0])
        bag_target_y = float(_m1_BAG_POS[1]) + float(tcp_object_offset[1])
        bag_release_height = max(self.args.impact_height, _m1_MIN_BAG_RELEASE_HEIGHT)
        bag_prep_duration = _m1_BAG_PREP_FRAMES * self.frame_dt * self.args.trajectory_time_scale
        segments = [
            (bag_prep_duration + 0.35, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0, -1)
        ]
        right_start = self.right_home
        hand_collision_enable_time = None
        bag_transport = world(
            wp.transform(wp.vec3(bag_target_x, bag_target_y, _m1_TABLE_TOP_Z + bag_release_height), grasp_rotation)
        )
        bag_hover = world(
            wp.transform(wp.vec3(bag_target_x, bag_target_y, _m1_TABLE_TOP_Z + bag_release_height), grasp_rotation)
        )
        retreat = world(
            wp.transform(
                wp.vec3(bag_target_x, bag_target_y, _m1_TABLE_TOP_Z + bag_release_height + 0.1), grasp_rotation
            )
        )
        for object_index in range(_m1_CUBE_COUNT):
            approach = world(self._object_approach_tf(object_index))
            grasp = world(self._object_grasp_tf(object_index))
            lift = world(self._object_lift_tf(object_index))
            open_rotation = wp.transform_get_rotation(self.right_home)
            open_approach = wp.transform(wp.transform_get_translation(approach), open_rotation)
            open_grasp = wp.transform(wp.transform_get_translation(grasp), open_rotation)
            if object_index == 0:
                hand_collision_enable_time = sum(segment[0] for segment in segments) + 0.8 + 1.2 + 0.2 + 0.35
            segments.extend(
                (
                    (0.8, self.left_home, self.left_home, right_start, open_approach, 0.0, 0.0, object_index),
                    (1.2, self.left_home, self.left_home, open_approach, open_grasp, 0.0, 0.0, object_index),
                    (0.2, self.left_home, self.left_home, open_grasp, open_grasp, 0.0, 0.0, object_index),
                    (0.35, self.left_home, self.left_home, open_grasp, grasp, 0.0, 0.0, object_index),
                    (
                        _m1_GRASP_PRE_CLOSE_HOLD_DURATION,
                        self.left_home,
                        self.left_home,
                        grasp,
                        grasp,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    (_m1_GRASP_CLOSE_DURATION, self.left_home, self.left_home, grasp, grasp, 0.0, 1.0, object_index),
                    (0.6, self.left_home, self.left_home, grasp, grasp, 1.0, 1.0, object_index),
                    (2.0, self.left_home, self.left_home, grasp, lift, 1.0, 1.0, object_index),
                    (0.6, self.left_home, self.left_home, lift, lift, 1.0, 1.0, object_index),
                    (
                        _m1_SOFT_CUBE_TRANSPORT_DURATION,
                        self.left_home,
                        self.left_home,
                        lift,
                        bag_transport,
                        1.0,
                        1.0,
                        object_index,
                    ),
                    (0.2, self.left_home, self.left_home, bag_transport, bag_hover, 1.0, 1.0, object_index),
                    (0.4, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, object_index),
                    (
                        _m1_SOFT_CUBE_RELEASE_OPEN_DURATION,
                        self.left_home,
                        self.left_home,
                        bag_hover,
                        bag_hover,
                        1.0,
                        0.0,
                        object_index,
                    ),
                    (
                        _m1_SOFT_CUBE_RELEASE_SETTLE_DURATION,
                        self.left_home,
                        self.left_home,
                        bag_hover,
                        bag_hover,
                        0.0,
                        0.0,
                        object_index,
                    ),
                    (1.0, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, object_index),
                )
            )
            segments.append((1.2, self.left_home, self.left_home, retreat, self.right_home, 0.0, 0.0, -1))
            right_start = self.right_home
        self.hand_collision_enable_time = float(hand_collision_enable_time or 0.0)
        segments.append((1.2, self.left_home, self.left_home, right_start, right_start, 0.0, 0.0, -1))
        return tuple(segments)

    def _build_joint_target_cache(self):
        script_duration = sum(segment[0] for segment in self.segments)
        script_frames = int(np.ceil(script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        self.cached_frame_count = max(int(self.args.num_frames), script_frames + _m1_FPS)
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
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m1_IK_ITERATIONS)
            wp.launch(
                _m1__lock_q,
                self.lock_indices.shape[0],
                [self.ik_q, self.lock_indices, self.lock_values],
                device=self.model.device,
            )
            cache[frame, : self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
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
        if self.cached_hand_collision[cache_index] and (not self.hand_particle_collision_enabled):
            self._set_hand_particle_collision(True)
        if self.previous_grip > 0.0001 and grip <= 0.0001 and (script_object >= 0):
            self.object_released[script_object] = True
        if self.previous_grip > 0.99 and grip < 0.99 and (not self.release_contact_material_applied):
            shape_mu = self.model.shape_material_mu.numpy()
            shape_ke = self.model.shape_material_ke.numpy()
            shape_kd = self.model.shape_material_kd.numpy()
            shape_mu[self.right_hand_shapes] = _m1_RELEASE_FRICTION
            shape_ke[self.right_hand_shapes] = _m1_RELEASE_CONTACT_KE
            shape_kd[self.right_hand_shapes] = _m1_RELEASE_CONTACT_KD
            self.model.shape_material_mu.assign(shape_mu)
            self.model.shape_material_ke.assign(shape_ke)
            self.model.shape_material_kd.assign(shape_kd)
            self.model.soft_contact_ke = _m1_RELEASE_CONTACT_KE
            self.model.soft_contact_kd = _m1_RELEASE_CONTACT_KD
            self.model.soft_contact_mu = _m1_RELEASE_FRICTION
            self.release_contact_material_applied = True
        self.previous_grip = grip

    def _simulate_substeps(self):
        """Advance one display frame after its joint targets are prepared."""
        bag_dz = 0.0
        for substep in range(self.sim_substeps):
            wp.launch(
                _m1__lift_pinned_vertices,
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
                _m1__interpolate_q,
                self.ik_model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m1__joint_velocity,
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
                _m1__accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            self.state_0, self.state_1 = (self.state_1, self.state_0)

    def simulate(self):
        self._prepare_frame()
        self._simulate_substeps()

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
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        visual_flags = self.model.shape_flags.numpy()[self.robot_visual_shapes]
        assert np.all(visual_flags & collision_mask == 0), "Robot visual shapes must remain non-colliding"
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
        assert maximum_body_particle_contacts <= _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE, (
            f"Per-body soft-contact buffer overflowed: {maximum_body_particle_contacts} > {_m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE}"
        )
        soft_min = soft_scene_q.min(axis=0)
        soft_max = soft_scene_q.max(axis=0)
        soft_center = soft_scene_q.mean(axis=0)
        bag_center = np.asarray((float(_m1_BAG_POS[0]), float(_m1_BAG_POS[1])))
        bag_half_extents = np.asarray((0.5 * _m1_BAG_WIDTH, 0.5 * _m1_BAG_DEPTH))
        soft_inside = (
            np.all(soft_min[:2] > bag_center - bag_half_extents - 0.02)
            and np.all(soft_max[:2] < bag_center + bag_half_extents + 0.02)
            and (soft_min[2] > bag_min_z - 0.08)
            and (soft_max[2] < _m1_TABLE_TOP_Z + 0.08)
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
                lower[int(qd_start[joint])] = q[int(q_start[joint])] - 0.0001
                upper[int(qd_start[joint])] = q[int(q_start[joint])] + 0.0001
        return (lower[: self.ik_model.joint_dof_count], upper[: self.ik_model.joint_dof_count])

    def _locked_q(self):
        q = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        indices = [
            int(q_start[joint])
            for joint, label in enumerate(self.model.joint_label[: self.ik_model.joint_count])
            if label not in controlled
        ]
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array([q[index] for index in indices], dtype=wp.float32, device=self.device),
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
        return next((index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name)))

    @staticmethod
    def _body_index(labels, name):
        return next((index for index, label in enumerate(labels) if label.endswith("/" + name)))

    def _tcp(self, state, body):
        body_tf = wp.transform(*state.body_q.numpy()[body])
        body_rot = wp.transform_get_rotation(body_tf)
        return wp.transform(wp.transform_get_translation(body_tf) + wp.quat_rotate(body_rot, _m1_TCP_OFFSET), body_rot)

    def _sample(self, time):
        for duration, left_a, left_b, right_a, right_b, grip_a, grip_b, object_index in self.segments:
            if time <= duration:
                alpha = float(np.clip(time / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                return (
                    self._lerp_tf(left_a, left_b, alpha),
                    self._lerp_tf(right_a, right_b, alpha),
                    grip_a * (1.0 - alpha) + grip_b * alpha,
                    object_index,
                )
            time -= duration
        _, _, left, _, right, _, grip, object_index = self.segments[-1]
        return (left, right, grip, object_index)

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
        array /= max(np.linalg.norm(array), 1e-08)
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
        parser.set_defaults(num_frames=1500)
        parser.add_argument("--robot-urdf", default=None, help="Optional Dexforce W1 URDF path.")
        parser.add_argument(
            "--house-visual-usd",
            default=_m1_DEFAULT_HOUSE_USD,
            help="Optional WAIC house USD reference; it is visual-only.",
        )
        parser.add_argument(
            "--recorded-hand-pose",
            default=str(_m1_DEFAULT_RECORDED_HAND_POSE),
            help="Saved right-hand TCP/finger pose used for the grasp, when present.",
        )
        parser.add_argument(
            "--recorded-grasp-z-offset",
            type=float,
            default=_m1_DEFAULT_RECORDED_GRASP_Z_OFFSET,
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
            default=_m1_DEFAULT_IMPACT_HEIGHT,
            help="Vertical release height above the bag rim [m].",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(_m1_WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(_m1_WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(_m1_WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(_m1_WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(_m1_WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(_m1_WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(_m1_WAIC_ROBOT_BASE_QUAT[3]))
        return parser


import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_m1_SOFT_CUBE_DENSITY = 100.0
_m1_SOFT_CUBE_DIMS = (6, 4, 6)
_m2_DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
)
_m2_INITIAL_IK_ITERATIONS = 240
_m2_RIGHT_J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
_m2_RIGHT_J7_TO_HAND_BASE_ROTATION = wp.quat(0.5, -0.5, 0.5, 0.5)
_m2_APPROACH_ROOT_POSITION = (-0.16214203834533691, -2.838686943054199, 1.3409454822540283)
_m2_APPROACH_ROOT_ROTATION = (0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569)
_m2_APPROACH_JOINTS_DEGREES = {
    "HAND_THUMB2": 90.0,
    "HAND_THUMB1": 6.0,
    "HAND_INDEX": 41.0,
    "INDEX_PIP": 24.0,
    "HAND_MIDDLE": 57.0,
    "MIDDLE_PIP": 0.0,
    "HAND_RING": 48.0,
    "RING_PIP": 15.0,
    "HAND_PINKY": 24.0,
    "PINKY_PIP": 26.0,
}
_m2_START_HOLD_DURATION = 0.5
_m2_START_TO_APPROACH_DURATION = 1.5
_m2_APPROACH_HOLD_DURATION = 0.5


class _m2_Example(_m1_Example):
    """Use recorded approach and grasp poses in the physical bag-placement demo."""

    def __init__(self, viewer, args):
        self.approach_root_world = wp.transform(
            wp.vec3(*_m2_APPROACH_ROOT_POSITION), wp.quat(*_m2_APPROACH_ROOT_ROTATION)
        )
        self.approach_hand_q = dict(_m2_APPROACH_JOINTS_DEGREES)
        self.grasp_root_world, self.grasp_hand_q = self._load_grasp_keyframe(args.recorded_grasp_keyframe)
        super().__init__(viewer, args)

    @staticmethod
    def _load_grasp_keyframe(path_value: str):
        """Load the current physical grasp pose written by the hand recorder."""
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Recorded grasp keyframe not found: {path}. Run the right-hand recorder and click Record keyframe."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload["keyframe"]
        root = keyframe["target_root_pose"]
        position = root["position_m"]
        rotation = root["quaternion_xyzw"]
        joints = keyframe["target_finger_joints_degrees"]
        if len(position) != 3 or len(rotation) != 4:
            raise ValueError(f"Invalid root pose in recorded grasp keyframe: {path}")
        grasp_joints = {name.removeprefix("RIGHT_"): float(value) for name, value in joints.items()}
        return (wp.transform(wp.vec3(*position), wp.quat(*rotation)), grasp_joints)

    def _load_recorded_hand_pose(self, path_value):
        """Disable the legacy one-pose loader from the base example."""
        return None

    def _root_to_tcp(self, root_transform):
        """Convert an isolated ``right_hand_base`` pose to the full-W1 IK target."""
        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = self._quat_mul(hand_rotation, wp.quat_inverse(_m2_RIGHT_J7_TO_HAND_BASE_ROTATION))
        target_offset = _m1_TCP_OFFSET - _m2_RIGHT_J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _right_hand_q(self):
        """Return the recorded target-point and closure finger configurations."""
        q_start = self.model.joint_q_start.numpy()
        indices = []
        start_q = []
        approach_q = []
        grasp_q = []
        for suffix in self.HAND_SUFFIXES:
            joint = self._joint_index(f"RIGHT_{suffix}")
            index = int(q_start[joint])
            indices.append(index)
            start_q.append(np.radians(90.0 if suffix == "HAND_THUMB2" else 0.0))
            approach_q.append(np.radians(self.approach_hand_q[suffix]))
            grasp_q.append(np.radians(self.grasp_hand_q[suffix]))
        self.hand_start = wp.array(start_q, dtype=wp.float32, device=self.device)
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(approach_q, dtype=wp.float32, device=self.device),
            wp.array(grasp_q, dtype=wp.float32, device=self.device),
        )

    def _set_hand_particle_collision(self, enabled: bool):
        """Match the isolated hand's URDF-only soft-contact geometry."""
        if enabled == self.hand_particle_collision_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        pad_shapes = set(getattr(self, "finger_pad_shapes", ()))
        urdf_hand_shapes = [shape for shape in self.right_hand_shapes if shape not in pad_shapes]
        if enabled:
            flags[urdf_hand_shapes] |= particle_flag
            self.model.soft_contact_ke = _m1_GRASP_CONTACT_KE
            self.model.soft_contact_kd = _m1_GRASP_CONTACT_KD
            self.model.soft_contact_mu = _m1_GRASP_SOFT_CONTACT_MU
        else:
            flags[urdf_hand_shapes] &= ~particle_flag
            self.model.soft_contact_ke = _m1_SOFT_CONTACT_KE
            self.model.soft_contact_kd = _m1_SOFT_CONTACT_KD
            self.model.soft_contact_mu = _m1_SOFT_CONTACT_MU
        if pad_shapes:
            flags[list(pad_shapes)] &= ~particle_flag
        self.model.shape_flags.assign(flags)
        self.hand_particle_collision_enabled = enabled

    def _build_joint_target_cache(self):
        """Initialize W1 at the recorded hand pose before caching the script."""
        approach = self._root_to_tcp(self.approach_root_world)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(approach))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(approach)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m2_INITIAL_IK_ITERATIONS)
        wp.launch(
            _m1__lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )
        initial_q = self.model.joint_q.numpy()
        initial_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        initial_q[self.hand_indices.numpy()] = self.hand_start.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        super()._build_joint_target_cache()
        cache = self.cached_joint_targets.numpy()
        hand_indices = self.hand_indices.numpy()
        start_q = self.hand_start.numpy()
        approach_q = self.hand_open.numpy()
        transition_end = _m2_START_HOLD_DURATION + _m2_START_TO_APPROACH_DURATION
        for frame in range(self.cached_frame_count + 1):
            script_time = frame * self.frame_dt * self.args.trajectory_time_scale
            if script_time <= _m2_START_HOLD_DURATION:
                cache[frame, hand_indices] = start_q
            elif script_time <= transition_end:
                alpha = (script_time - _m2_START_HOLD_DURATION) / _m2_START_TO_APPROACH_DURATION
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                cache[frame, hand_indices] = start_q * (1.0 - alpha) + approach_q * alpha
            else:
                break
        self.cached_joint_targets.assign(cache)

    def _segments(self):
        """Build the isolated-hand trajectory from its recorded initial pose."""
        approach = self._root_to_tcp(self.approach_root_world)
        approach_root_position = wp.transform_get_translation(self.approach_root_world)
        cube_position = self._world_vec(_m1_CUBE_POSITIONS[0])
        root_cube_offset = approach_root_position - cube_position
        bag_position = self._world_vec(_m1_BAG_POS)
        bag_hover_root = wp.transform(
            wp.vec3(
                float(bag_position[0]) + float(root_cube_offset[0]),
                float(bag_position[1]) + float(root_cube_offset[1]),
                float(bag_position[2]) + _m1_BAG_HEIGHT + 0.06 + float(root_cube_offset[2]),
            ),
            wp.transform_get_rotation(self.approach_root_world),
        )
        bag_hover = self._root_to_tcp(bag_hover_root)
        lift = wp.transform(
            wp.transform_get_translation(approach) + wp.vec3(0.0, 0.0, 0.07), wp.transform_get_rotation(approach)
        )
        retreat = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.1), wp.transform_get_rotation(bag_hover)
        )
        self.hand_collision_enable_time = 0.5
        return (
            (_m2_START_HOLD_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (_m2_START_TO_APPROACH_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (_m2_APPROACH_HOLD_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (1.8, self.left_home, self.left_home, approach, approach, 0.0, 1.0, 0),
            (0.6, self.left_home, self.left_home, approach, approach, 1.0, 1.0, 0),
            (1.2, self.left_home, self.left_home, approach, lift, 1.0, 1.0, 0),
            (7.0, self.left_home, self.left_home, lift, bag_hover, 1.0, 1.0, 0),
            (0.4, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, 0),
            (_m1_SOFT_CUBE_RELEASE_OPEN_DURATION, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 0.0, 0),
            (_m1_SOFT_CUBE_RELEASE_SETTLE_DURATION, self.left_home, self.left_home, bag_hover, bag_hover, 0.0, 0.0, 0),
            (1.0, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, -1),
        )

    @staticmethod
    def create_parser():
        """Create parser options for the recorded two-pose grasp demo."""
        parser = _m1_Example.create_parser()
        parser.add_argument(
            "--recorded-grasp-keyframe",
            default=str(_m2_DEFAULT_GRASP_KEYFRAME),
            help="Latest keyframe JSON from example_vbd_mjvbd_v2_right_hand_soft_cube_recorder.py.",
        )
        return parser


def _m2_main():
    """Run the recorded two-pose soft-cube bag-placement demo."""
    parser = _m2_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m2_Example(viewer, args), args)


import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m3_FPS = 60
_m3_SIM_SUBSTEPS = 8
_m3_VBD_ITERATIONS = 40
_m3_RIGHT_HAND_URDF = Path(__file__).resolve().parents[3] / "assets" / "W1_right_hand" / "DexforceW1_right_hand.urdf"
_m3_HAND_HOME = wp.transform(
    wp.vec3(-0.15679353, -2.8874836, 1.3789376), wp.quat(-0.31233013, 0.67216527, 0.32775849, -0.58584785)
)
_m3_TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
_m3_TABLE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m3_TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
_m3_TABLE_TOP_Z = float(_m3_TABLE_POS[2]) + _m3_TABLE_HALF_EXTENTS[2]
_m3_CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
_m3_CUBE_CENTRE = wp.vec3(-0.14931439, -2.76669516, _m3_TABLE_TOP_Z + _m3_CUBE_HALF_EXTENTS[2] + 0.001)
_m3_CUBE_DENSITY = 1500.0
_m3_CONTACT_MARGIN = 0.0015
_m3_CONTACT_KE = 3000.0
_m3_CONTACT_KD = 1.0
_m3_CONTACT_MU = 3000.0
_m3_RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
_m3_POSITION_LIMIT_MM = 500.0
_m3_CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
_m3_CAMERA_PITCH = -18.0
_m3_CAMERA_YAW = 126.0
_m3_HAND_JOINTS = (
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
_m3_INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
_m3_INITIAL_HAND_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}


@wp.kernel
def _m3__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m3__joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
    joint = wp.tid()
    q_begin, q_end = (joint_q_start[joint], joint_q_start[joint + 1])
    qd_begin, qd_end = (joint_qd_start[joint], joint_qd_start[joint + 1])
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
        for i in range(qd_end - qd_begin):
            if q_begin + i < q_end:
                out[qd_begin + i] = (q1[q_begin + i] - q0[q_begin + i]) * inv_dt


class _m3_Example:
    """Tune a mesh-only physical grasp of one dynamic rigid cube."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / _m3_FPS
        self.sim_dt = self.frame_dt / _m3_SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self._root = None
        self._status_var = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._last_target_signature: tuple[float, ...] | None = None
        self._initial_keyframe = self._load_initial_keyframe()
        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m3_VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0005,
                "rigid_contact_stick_freeze_translation_eps": 0.0002,
                "rigid_contact_stick_freeze_angular_eps": 0.0002,
                "rigid_body_contact_buffer_size": _m3_RIGID_BODY_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
            },
            collision_options={"broad_phase": "nxn", "contact_matching": "latest"},
            coupling_mode="one_way",
        )
        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m3_INITIAL_HAND_ROOT)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(_m3_INITIAL_HAND_JOINTS)
        self._restore_initial_controls()
        self.joint_limits = self._joint_limits()
        self.target_transform = self._copy_transform(self.gizmo_transform)
        self._refresh_target()
        self._set_initial_hand_pose()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(_m3_CAMERA_POS, _m3_CAMERA_PITCH, _m3_CAMERA_YAW)

    def _build_scene(self):
        if not _m3_RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {_m3_RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m3_CONTACT_KE
        builder.default_shape_cfg.kd = _m3_CONTACT_KD
        builder.default_shape_cfg.mu = _m3_CONTACT_MU
        builder.default_shape_cfg.margin = _m3_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m3_RIGHT_HAND_URDF),
            xform=_m3_HAND_HOME,
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
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=300000.0, kd=0.0001, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(_m3_TABLE_POS, _m3_TABLE_ROTATION),
            hx=_m3_TABLE_HALF_EXTENTS[0],
            hy=_m3_TABLE_HALF_EXTENTS[1],
            hz=_m3_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="hand_tuning_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="hand_tuning_ground")
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m3_CUBE_DENSITY, ke=_m3_CONTACT_KE, kd=_m3_CONTACT_KD, mu=_m3_CONTACT_MU, margin=_m3_CONTACT_MARGIN
        )
        cube_cfg.configure_sdf(force_sdf=True)
        self.cube_body = builder.add_body(
            xform=wp.transform(_m3_CUBE_CENTRE, wp.quat_identity()), label="tunable_rigid_cube"
        )
        self.cube_shape = builder.shape_count
        builder.add_shape_box(
            self.cube_body,
            hx=_m3_CUBE_HALF_EXTENTS[0],
            hy=_m3_CUBE_HALF_EXTENTS[1],
            hz=_m3_CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            color=(0.9, 0.32, 0.18),
            label="tunable_rigid_cube_shape",
        )
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes
        builder.color()
        self.model = builder.finalize(requires_grad=False)

    def _root_joint_index(self):
        types = self.model.joint_type.numpy()
        parents = self.model.joint_parent.numpy()
        for index, (joint_type, parent) in enumerate(zip(types, parents, strict=True)):
            if int(joint_type) == int(newton.JointType.FREE) and int(parent) == -1:
                return index
        raise RuntimeError("Right-hand URDF must import with a free root joint")

    def _hand_joint_indices(self):
        labels = self.model.joint_label
        starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        indices = {}
        self.hand_joint_limit_indices = {}
        for name in _m3_HAND_JOINTS:
            joint = next((index for index, label in enumerate(labels) if label.endswith("/" + name)))
            indices[name] = int(starts[joint])
            self.hand_joint_limit_indices[name] = int(dof_starts[joint])
        return indices

    def _joint_limits(self):
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        return {
            name: tuple(
                sorted(
                    (
                        float(np.degrees(lower[self.hand_joint_limit_indices[name]])),
                        float(np.degrees(upper[self.hand_joint_limit_indices[name]])),
                    )
                )
            )
            for name in _m3_HAND_JOINTS
        }

    @staticmethod
    def _copy_transform(transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    def _load_initial_keyframe(self):
        path = Path(self.args.keyframe_output).expanduser()
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload.get("keyframe")
        return keyframe if isinstance(keyframe, dict) else None

    def _restore_initial_controls(self):
        """Restore the gizmo, relative offsets, and finger targets of a keyframe."""
        keyframe = self._initial_keyframe
        if keyframe is None:
            return
        gizmo = keyframe.get("gizmo_world")
        if isinstance(gizmo, dict):
            position = gizmo.get("position_m")
            rotation = gizmo.get("quaternion_xyzw")
            if (
                isinstance(position, list)
                and len(position) == 3
                and isinstance(rotation, list)
                and (len(rotation) == 4)
            ):
                self.gizmo_transform = wp.transform(wp.vec3(*position), wp.quat(*rotation))
        position_offset = keyframe.get("position_offset_mm")
        if isinstance(position_offset, list) and len(position_offset) == 3:
            self.position_mm = np.asarray(position_offset, dtype=np.float32)
        rotation_offset = keyframe.get("rotation_offset_deg")
        if isinstance(rotation_offset, list) and len(rotation_offset) == 3:
            self.rotation_deg = np.asarray(rotation_offset, dtype=np.float32)
        joints = keyframe.get("target_finger_joints_degrees")
        if isinstance(joints, dict):
            for name in _m3_HAND_JOINTS:
                if name in joints:
                    self.joint_degrees[name] = float(joints[name])

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

    def _offset_transform(self):
        base_position = np.asarray(wp.transform_get_translation(self.gizmo_transform), dtype=np.float32)
        position = base_position + self.position_mm * 0.001
        rx, ry, rz = np.radians(self.rotation_deg)
        rotation = self._quat_mul(
            self._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        return wp.transform(
            wp.vec3(*position), self._quat_mul(rotation, wp.transform_get_rotation(self.gizmo_transform))
        )

    def _refresh_target(self):
        self.target_transform = self._offset_transform()
        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(self.target_transform)
        rotation = wp.transform_get_rotation(self.target_transform)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(self.joint_degrees[name])
        self.manual_target_q.assign(target_q)
        self._last_target_signature = self._target_signature()

    def _set_initial_hand_pose(self):
        """Initialize the physical hand at the configured grasp keyframe."""
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

    def _target_signature(self):
        return tuple(
            [*wp.transform_get_translation(self.gizmo_transform), *wp.transform_get_rotation(self.gizmo_transform)]
            + self.position_mm.tolist()
            + self.rotation_deg.tolist()
            + [self.joint_degrees[name] for name in _m3_HAND_JOINTS]
        )

    def step_once(self):
        """Advance one real-time physical frame toward the current hand target."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(_m3_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m3_SIM_SUBSTEPS
            wp.launch(
                _m3__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m3__joint_velocity,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = (self.state_1, self.state_0)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _reset_physics(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m3_INITIAL_HAND_ROOT)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(_m3_INITIAL_HAND_JOINTS)
        self._restore_initial_controls()
        self.sim_time = 0.0
        self.frame_index = 0
        self._trajectory_frames.clear()
        self._refresh_target()
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)

    def _contact_counts(self) -> tuple[int, int]:
        """Return current hand-cube and total rigid contact counts."""
        contacts = self.solver.contacts
        total = int(contacts.rigid_contact_count.numpy()[0])
        shape_0 = contacts.rigid_contact_shape0.numpy()
        shape_1 = contacts.rigid_contact_shape1.numpy()
        active = min(total, shape_0.shape[0])
        shape_0 = shape_0[:active]
        shape_1 = shape_1[:active]
        hand_cube = np.count_nonzero(
            (shape_0 == self.cube_shape) & (shape_1 >= 0) & (shape_1 < self.hand_shape_end)
            | (shape_1 == self.cube_shape) & (shape_0 >= 0) & (shape_0 < self.hand_shape_end)
        )
        return (int(hand_cube), total)

    def _transform_dict(self, transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return {
            "position_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in rotation],
        }

    def _capture_frame(self):
        current_q = self.state_0.joint_q.numpy()
        root_q = current_q[self.root_q_start : self.root_q_start + 7]
        cube_q = self.state_0.body_q.numpy()[self.cube_body]
        cube_qd = self.state_0.body_qd.numpy()[self.cube_body]
        hand_cube_contacts, total_rigid_contacts = self._contact_counts()
        return {
            "frame": self.frame_index,
            "time_s": self.sim_time,
            "gizmo_world": self._transform_dict(self.gizmo_transform),
            "position_offset_mm": self.position_mm.tolist(),
            "rotation_offset_deg": self.rotation_deg.tolist(),
            "target_root_pose": self._transform_dict(self.target_transform),
            "target_finger_joints_degrees": dict(self.joint_degrees),
            "root_pose": {
                "position_m": [float(value) for value in root_q[:3]],
                "quaternion_xyzw": [float(value) for value in root_q[3:]],
            },
            "finger_joints_radians": {name: float(current_q[index]) for name, index in self.hand_joint_indices.items()},
            "finger_joints_degrees": {
                name: float(np.degrees(current_q[index])) for name, index in self.hand_joint_indices.items()
            },
            "rigid_cube_pose": {
                "position_m": [float(value) for value in cube_q[:3]],
                "quaternion_xyzw": [float(value) for value in cube_q[3:]],
            },
            "rigid_cube_twist": [float(value) for value in cube_qd],
            "hand_cube_contact_count": hand_cube_contacts,
            "total_rigid_contact_count": total_rigid_contacts,
        }

    def _store_trajectory_frame(self):
        frame = self._capture_frame()
        if self._trajectory_frames and self._trajectory_frames[-1]["frame"] == frame["frame"]:
            self._trajectory_frames[-1] = frame
        else:
            self._trajectory_frames.append(frame)

    @staticmethod
    def _write_json(path_value: str, payload: dict[str, Any]):
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_pose(self):
        path = self._write_json(
            self.args.pose_output, {"format": "newton_w1_right_hand_rigid_cube_pose_v1", "pose": self._capture_frame()}
        )
        self._set_status(f"Saved pose: {path}")

    def save_trajectory(self):
        path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_rigid_cube_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        self._set_status(f"Saved trajectory: {path}")

    def render(self):
        if self._target_signature() != self._last_target_signature:
            self._refresh_target()
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("right_hand_target", self.gizmo_transform)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _set_status(self, message: str):
        if self._status_var is not None:
            self._status_var.set(message)

    def _on_control_changed(self, variables):
        for name, variable in variables["joints"].items():
            self.joint_degrees[name] = float(variable.get())
        self.position_mm = np.asarray([float(variable.get()) for variable in variables["position"]], dtype=np.float32)
        self.rotation_deg = np.asarray([float(variable.get()) for variable in variables["rotation"]], dtype=np.float32)
        self._refresh_target()
        self.render()

    def _make_scale(self, parent, row: int, label: str, variable, minimum: float, maximum: float, command):
        import tkinter as tk

        self._ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        value_label = self._ttk.Label(parent, textvariable=variable._display_var, width=8, anchor="e")
        value_label.grid(row=row, column=1, sticky="e", padx=3)
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=1.0,
            orient="horizontal",
            showvalue=False,
            length=440,
            highlightthickness=0,
            command=command,
        )
        scale.grid(row=row, column=2, sticky="ew", padx=4)

        def update_display(*_):
            variable._display_var.set(f"{float(variable.get()):.1f}")

        variable.trace_add("write", update_display)
        update_display()

    def _build_controls(self, root):
        import tkinter as tk

        frame = self._ttk.Frame(root, padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 0))
        frame.columnconfigure(2, weight=1)
        variables = {"joints": {}, "position": [], "rotation": []}
        joints = self._ttk.LabelFrame(frame, text="RIGHT finger joint angles (degrees)", padding=5)
        joints.grid(row=0, column=0, columnspan=3, sticky="nsew")
        joints.columnconfigure(2, weight=1)
        for row, name in enumerate(_m3_HAND_JOINTS):
            variable = tk.DoubleVar(value=self.joint_degrees[name])
            variable._display_var = tk.StringVar()
            variables["joints"][name] = variable
            lower, upper = self.joint_limits[name]
            self._make_scale(
                joints,
                row,
                name,
                variable,
                lower,
                upper,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        root_box = self._ttk.LabelFrame(frame, text="Whole-hand target offset / rotation relative to gizmo", padding=5)
        root_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root_box.columnconfigure(2, weight=1)
        for index, label in enumerate(("Position X (mm)", "Position Y (mm)", "Position Z (mm)")):
            variable = tk.DoubleVar(value=float(self.position_mm[index]))
            variable._display_var = tk.StringVar()
            variables["position"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -_m3_POSITION_LIMIT_MM,
                _m3_POSITION_LIMIT_MM,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        for index, label in enumerate(("Rotation X (deg)", "Rotation Y (deg)", "Rotation Z (deg)"), start=3):
            variable = tk.DoubleVar(value=float(self.rotation_deg[index - 3]))
            variable._display_var = tk.StringVar()
            variables["rotation"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -180.0,
                180.0,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        return variables

    def run_recorder(self):
        if self.args.recorder_no_gui:
            self.render()
            self.viewer.close()
            return
        import tkinter as tk
        from tkinter import ttk

        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()
        self._ttk = ttk
        root = tk.Tk()
        self._root = root
        root.title("MJVBD-v2 W1 right-hand rigid-cube recorder")
        root.geometry("710x650")
        root.minsize(660, 630)
        self._build_controls(root)
        buttons = ttk.Frame(root, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Record keyframe", command=self._record_keyframe_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset physics", command=self._reset_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save pose JSON", command=self.save_pose).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save trajectory", command=self.save_trajectory).pack(side="left", padx=3)
        self._status_var = tk.StringVar(
            value="Realtime rigid physics running; adjust the hand, then record a stable grasp keyframe."
        )
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.step_once()
            self.render()
            root.after(max(1, int(1000.0 / _m3_FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _record_keyframe_from_ui(self):
        self._store_trajectory_frame()
        keyframe_path = self._write_json(
            self.args.keyframe_output,
            {"format": "newton_w1_right_hand_rigid_cube_keyframe_v1", "keyframe": self._trajectory_frames[-1]},
        )
        trajectory_path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_rigid_cube_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        hand_cube_contacts, total_rigid_contacts = self._contact_counts()
        self._set_status(
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; hand-cube contacts: {hand_cube_contacts}, total rigid contacts: {total_rigid_contacts}. Saved: {keyframe_path}, {trajectory_path}"
        )

    def _reset_from_ui(self):
        self._reset_physics()
        self._set_status("Reset the hand and rigid cube to their initial states.")
        self.render()

    def test_final(self):
        """Verify that one physical step keeps the hand and rigid cube finite."""
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_qd.numpy()))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        parser.add_argument(
            "--pose-output",
            default=str(
                Path(__file__).resolve().parents[3]
                / "assets"
                / "vbd_mjvbd_v2"
                / "vbd_w1_right_hand_rigid_cube_pose.json"
            ),
        )
        parser.add_argument(
            "--trajectory-output",
            default=str(
                Path(__file__).resolve().parents[3]
                / "assets"
                / "vbd_mjvbd_v2"
                / "vbd_w1_right_hand_rigid_cube_trajectory.json"
            ),
        )
        parser.add_argument(
            "--keyframe-output",
            default=str(
                Path(__file__).resolve().parents[3]
                / "assets"
                / "vbd_mjvbd_v2"
                / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"
            ),
        )
        parser.add_argument("--recorder-no-gui", action="store_true")
        return parser


def _m3_main():
    """Launch the interactive right-hand rigid-cube recorder."""
    parser = _m3_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = _m3_Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m4_DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"
)
_m4_BAG_WIDTH = 0.2
_m4_BAG_DEPTH = 0.16
_m4_BAG_HEIGHT = 0.24
_m4_BAG_POS = wp.vec3(0.24068561, -2.79869516, 0.93122798)
_m4_BAG_RESOLUTION = 20
_m4_BAG_PARTICLE_RADIUS = 0.003
_m4_BAG_DENSITY = 0.08
_m4_BAG_TRI_KE = 150.0
_m4_BAG_TRI_KA = 150.0
_m4_BAG_TRI_KD = 0.5
_m4_BAG_EDGE_KE = 0.5
_m4_BAG_EDGE_KD = 1.5e-05
_m4_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
_m4_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE = 128
_m4_PARTICLE_EDGE_CONTACT_BUFFER_SIZE = 256
_m4_SOFT_CONTACT_MARGIN = 0.01
_m4_SOFT_CONTACT_KE = 5000.0
_m4_SOFT_CONTACT_KD = 0.05
_m4_SOFT_CONTACT_MU = 0.25
_m4_RELEASE_CONTACT_KE = 5000.0
_m4_RELEASE_CONTACT_KD = 0.0
_m4_RELEASE_FRICTION = 0.0
_m4_GRASP_CLOSE_DURATION = 0.45
_m4_GRASP_SETTLE_DURATION = 0.3
_m4_LIFT_DURATION = 0.75
_m4_OPEN_JOINTS = dict.fromkeys(_m3_HAND_JOINTS, 0.0)
_m4_START_JOINTS = dict(_m4_OPEN_JOINTS)
_m4_START_JOINTS["RIGHT_HAND_THUMB2"] = 90.0


def _m4__generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
    """Generate the five faces of an open-topped cloth box."""
    cell_x = 2.0 * half_x / resolution
    cell_y = 2.0 * half_y / resolution
    cell_z = height / resolution
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []

    def vertex(x: float, y: float, z: float):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    def quad(v00: int, v10: int, v01: int, v11: int):
        indices.extend((v00, v10, v01, v10, v11, v01))

    for i in range(resolution):
        for j in range(resolution):
            x0, y0 = (-half_x + i * cell_x, -half_y + j * cell_y)
            x1, y1 = (x0 + cell_x, y0 + cell_y)
            quad(vertex(x0, y0, 0.0), vertex(x1, y0, 0.0), vertex(x0, y1, 0.0), vertex(x1, y1, 0.0))
    for i in range(resolution):
        for j in range(resolution):
            x0, x1 = (-half_x + i * cell_x, -half_x + (i + 1) * cell_x)
            y0, y1 = (-half_y + i * cell_y, -half_y + (i + 1) * cell_y)
            z0, z1 = (j * cell_z, (j + 1) * cell_z)
            quad(vertex(x0, -half_y, z0), vertex(x1, -half_y, z0), vertex(x0, -half_y, z1), vertex(x1, -half_y, z1))
            quad(vertex(x1, half_y, z0), vertex(x0, half_y, z0), vertex(x1, half_y, z1), vertex(x0, half_y, z1))
            quad(vertex(-half_x, y1, z0), vertex(-half_x, y0, z0), vertex(-half_x, y1, z1), vertex(-half_x, y0, z1))
            quad(vertex(half_x, y0, z0), vertex(half_x, y1, z0), vertex(half_x, y0, z1), vertex(half_x, y1, z1))
    return (np.asarray(vertices, dtype=np.float32), indices)


@wp.kernel
def _m4__pin_bag_particles(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    i = wp.tid()
    particle = pinned_indices[i]
    pos_0[particle] = original_positions[i]
    pos_1[particle] = original_positions[i]


@wp.kernel
def _m4__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m4__joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
    joint = wp.tid()
    q_begin, q_end = (joint_q_start[joint], joint_q_start[joint + 1])
    qd_begin, qd_end = (joint_qd_start[joint], joint_qd_start[joint + 1])
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
        for i in range(qd_end - qd_begin):
            if q_begin + i < q_end:
                out[qd_begin + i] = (q1[q_begin + i] - q0[q_begin + i]) * inv_dt


class _m4_Example(_m3_Example):
    """Run the recorded mesh-only rigid-cube grasp and bag placement."""

    def __init__(self, viewer, args):
        self.grasp_root, self.grasp_joints, self.recorded_cube_position = self._load_grasp_keyframe(args.grasp_keyframe)
        super().__init__(viewer, args)
        self._initialize_bag_pin()
        self._create_solver()
        self.release_contact_material_applied = False
        self._set_hand_target(self.grasp_root, _m4_START_JOINTS)
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)
        self.segments = self._build_segments()
        self.script_duration = sum(segment[0] for segment in self.segments)

    @staticmethod
    def _load_grasp_keyframe(path_value: str):
        """Load the recorded hand target and settled rigid-cube position."""
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recorded rigid-cube grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload.get("keyframe")
        if not isinstance(keyframe, dict):
            raise ValueError(f"Missing keyframe object in recorded grasp: {path}")
        root = keyframe.get("target_root_pose")
        joints = keyframe.get("target_finger_joints_degrees")
        cube = keyframe.get("rigid_cube_pose")
        if not isinstance(root, dict) or not isinstance(joints, dict) or (not isinstance(cube, dict)):
            raise ValueError(f"Incomplete rigid-cube grasp keyframe: {path}")
        position = root.get("position_m")
        rotation = root.get("quaternion_xyzw")
        cube_position = cube.get("position_m")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Invalid root position in recorded grasp: {path}")
        if not isinstance(rotation, list) or len(rotation) != 4:
            raise ValueError(f"Invalid root rotation in recorded grasp: {path}")
        if not isinstance(cube_position, list) or len(cube_position) != 3:
            raise ValueError(f"Invalid cube position in recorded grasp: {path}")
        missing_joints = set(_m3_HAND_JOINTS) - joints.keys()
        if missing_joints:
            raise ValueError(f"Missing hand joints in recorded grasp {path}: {sorted(missing_joints)}")
        return (
            wp.transform(wp.vec3(*position), wp.quat(*rotation)),
            {name: float(joints[name]) for name in _m3_HAND_JOINTS},
            wp.vec3(*cube_position),
        )

    def _build_scene(self):
        """Build the recorder's rigid scene plus the pinned soft bag."""
        if not _m3_RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {_m3_RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m3_CONTACT_KE
        builder.default_shape_cfg.kd = _m3_CONTACT_KD
        builder.default_shape_cfg.mu = _m3_CONTACT_MU
        builder.default_shape_cfg.margin = _m3_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m3_RIGHT_HAND_URDF),
            xform=_m3_HAND_HOME,
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
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=300000.0, kd=0.0001, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(_m3_TABLE_POS, _m3_TABLE_ROTATION),
            hx=_m3_TABLE_HALF_EXTENTS[0],
            hy=_m3_TABLE_HALF_EXTENTS[1],
            hz=_m3_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="recorded_rigid_cube_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="recorded_rigid_cube_ground")
        bag_vertices, bag_indices = _m4__generate_box_bag(
            0.5 * _m4_BAG_WIDTH, 0.5 * _m4_BAG_DEPTH, _m4_BAG_HEIGHT, _m4_BAG_RESOLUTION
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=_m4_BAG_POS,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=_m4_BAG_DENSITY,
            tri_ke=_m4_BAG_TRI_KE,
            tri_ka=_m4_BAG_TRI_KA,
            tri_kd=_m4_BAG_TRI_KD,
            edge_ke=_m4_BAG_EDGE_KE,
            edge_kd=_m4_BAG_EDGE_KD,
            particle_radius=_m4_BAG_PARTICLE_RADIUS,
            label="recorded_rigid_cube_soft_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - _m4_BAG_HEIGHT) < 1e-05)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m3_CUBE_DENSITY, ke=_m3_CONTACT_KE, kd=_m3_CONTACT_KD, mu=_m3_CONTACT_MU, margin=_m3_CONTACT_MARGIN
        )
        cube_cfg.configure_sdf(force_sdf=True)
        self.cube_body = builder.add_body(
            xform=wp.transform(_m3_CUBE_CENTRE, wp.quat_identity()), label="recorded_rigid_cube"
        )
        self.cube_shape = builder.shape_count
        builder.add_shape_box(
            self.cube_body,
            hx=_m3_CUBE_HALF_EXTENTS[0],
            hy=_m3_CUBE_HALF_EXTENTS[1],
            hz=_m3_CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            color=(0.9, 0.32, 0.18),
            label="recorded_rigid_cube_shape",
        )
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m4_SOFT_CONTACT_KE
        self.model.soft_contact_kd = _m4_SOFT_CONTACT_KD
        self.model.soft_contact_mu = _m4_SOFT_CONTACT_MU

    def _initialize_bag_pin(self):
        """Pin the open bag rim at its initial world positions."""
        flags = self.model.particle_flags.numpy()
        flags[self.bag_top_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        particle_q = self.state_0.particle_q.numpy()
        self.bag_pinned_indices = wp.array(self.bag_top_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_original = wp.array(particle_q[self.bag_top_indices].copy(), dtype=wp.vec3, device=self.device)

    def _create_solver(self):
        """Create VBD with the recorder's rigid settings and bag contacts."""
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m3_VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0005,
                "rigid_contact_stick_freeze_translation_eps": 0.0002,
                "rigid_contact_stick_freeze_angular_eps": 0.0002,
                "rigid_body_contact_buffer_size": _m3_RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": _m4_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": _m4_BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": 2.0 * _m4_BAG_PARTICLE_RADIUS,
                "particle_vertex_contact_buffer_size": _m4_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": _m4_PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "soft_contact_margin": _m4_SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )

    def _set_hand_target(self, root: wp.transform, joints: dict[str, float]):
        """Set the floating-hand root and finger target for the next frame."""
        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(joints[name])
        self.manual_target_q.assign(target_q)

    def _build_segments(self):
        """Build closure, lift, transport, release, and retreat phases."""
        grasp_position = wp.transform_get_translation(self.grasp_root)
        grasp_rotation = wp.transform_get_rotation(self.grasp_root)
        root_cube_offset = grasp_position - self.recorded_cube_position
        lift = wp.transform(grasp_position + wp.vec3(0.0, 0.0, 0.1), grasp_rotation)
        release_cube_position = wp.vec3(
            float(_m4_BAG_POS[0]), float(_m4_BAG_POS[1]), float(_m4_BAG_POS[2]) + _m4_BAG_HEIGHT + 0.06
        )
        bag_hover = wp.transform(release_cube_position + root_cube_offset, grasp_rotation)
        transport = wp.transform(wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.05), grasp_rotation)
        retreat = wp.transform(wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.12), grasp_rotation)
        approach_joints = _m3_INITIAL_HAND_JOINTS
        segments = (
            (0.5, self.grasp_root, self.grasp_root, _m4_START_JOINTS, _m4_START_JOINTS),
            (1.5, self.grasp_root, self.grasp_root, _m4_START_JOINTS, approach_joints),
            (0.5, self.grasp_root, self.grasp_root, approach_joints, approach_joints),
            (_m4_GRASP_CLOSE_DURATION, self.grasp_root, self.grasp_root, approach_joints, self.grasp_joints),
            (_m4_GRASP_SETTLE_DURATION, self.grasp_root, self.grasp_root, self.grasp_joints, self.grasp_joints),
            (_m4_LIFT_DURATION, self.grasp_root, lift, self.grasp_joints, self.grasp_joints),
            (5.0, lift, transport, self.grasp_joints, self.grasp_joints),
            (1.2, transport, bag_hover, self.grasp_joints, self.grasp_joints),
            (0.5, bag_hover, bag_hover, self.grasp_joints, self.grasp_joints),
            (0.8, bag_hover, bag_hover, self.grasp_joints, _m4_OPEN_JOINTS),
            (1.5, bag_hover, bag_hover, _m4_OPEN_JOINTS, _m4_OPEN_JOINTS),
            (1.0, bag_hover, retreat, _m4_OPEN_JOINTS, _m4_OPEN_JOINTS),
        )
        self.release_start_time = sum(segment[0] for segment in segments[:-3])
        return segments

    def _apply_release_contact_material(self):
        """Use the rigid reference's low-friction material during release."""
        if self.release_contact_material_applied:
            return
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[: self.hand_shape_end] = _m4_RELEASE_CONTACT_KE
        shape_kd[: self.hand_shape_end] = _m4_RELEASE_CONTACT_KD
        shape_mu[: self.hand_shape_end] = _m4_RELEASE_FRICTION
        shape_ke[self.cube_shape] = _m4_RELEASE_CONTACT_KE
        shape_kd[self.cube_shape] = _m4_RELEASE_CONTACT_KD
        shape_mu[self.cube_shape] = _m4_RELEASE_FRICTION
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self.model.soft_contact_ke = _m4_RELEASE_CONTACT_KE
        self.model.soft_contact_kd = _m4_RELEASE_CONTACT_KD
        self.model.soft_contact_mu = _m4_RELEASE_FRICTION
        self.release_contact_material_applied = True

    def _restore_grasp_contact_material(self):
        """Restore the recorder's hand-cube material after a reset."""
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[: self.hand_shape_end] = _m3_CONTACT_KE
        shape_kd[: self.hand_shape_end] = _m3_CONTACT_KD
        shape_mu[: self.hand_shape_end] = _m3_CONTACT_MU
        shape_ke[self.cube_shape] = _m3_CONTACT_KE
        shape_kd[self.cube_shape] = _m3_CONTACT_KD
        shape_mu[self.cube_shape] = _m3_CONTACT_MU
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self.model.soft_contact_ke = _m4_SOFT_CONTACT_KE
        self.model.soft_contact_kd = _m4_SOFT_CONTACT_KD
        self.model.soft_contact_mu = _m4_SOFT_CONTACT_MU
        self.release_contact_material_applied = False

    def _sample(self, time_s: float):
        """Interpolate the recorded root and joint targets at a script time."""
        for duration, root_a, root_b, joints_a, joints_b in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in _m3_HAND_JOINTS}
                return (root, joints)
            time_s -= duration
        _, _, root, _, joints = self.segments[-1]
        return (root, joints)

    @staticmethod
    def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float):
        """Linearly interpolate position and normalize the quaternion."""
        position_a = np.asarray(wp.transform_get_translation(a), dtype=np.float32)
        position_b = np.asarray(wp.transform_get_translation(b), dtype=np.float32)
        rotation_a = np.asarray(wp.transform_get_rotation(a), dtype=np.float32)
        rotation_b = np.asarray(wp.transform_get_rotation(b), dtype=np.float32)
        if np.dot(rotation_a, rotation_b) < 0.0:
            rotation_b = -rotation_b
        rotation = rotation_a * (1.0 - alpha) + rotation_b * alpha
        rotation /= max(np.linalg.norm(rotation), 1e-08)
        return wp.transform(wp.vec3(*position_a * (1.0 - alpha) + position_b * alpha), wp.quat(*rotation))

    def step_once(self):
        """Advance one frame while keeping the bag rim pinned."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(_m3_SIM_SUBSTEPS):
            wp.launch(
                _m4__pin_bag_particles,
                self.bag_pinned_indices.shape[0],
                [self.bag_pinned_indices, self.bag_pinned_original, self.state_0.particle_q, self.state_1.particle_q],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m3_SIM_SUBSTEPS
            wp.launch(
                _m4__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m4__joint_velocity,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = (self.state_1, self.state_0)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _reset_physics(self):
        super()._reset_physics()
        self._initialize_bag_pin()
        self._restore_grasp_contact_material()
        self._set_hand_target(self.grasp_root, _m4_START_JOINTS)
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)

    def step(self):
        """Advance the autonomous recorded grasp by one physical frame."""
        root, joints = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        if self.sim_time >= self.release_start_time:
            self._apply_release_contact_material()
        self.step_once()

    def render(self):
        """Render the hand, rigid cube, and soft bag without the recorder UI."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite dynamic rigid-cube and soft-bag states."""
        body_flags = int(self.model.body_flags.numpy()[self.cube_body])
        assert not body_flags & int(newton.BodyFlags.KINEMATIC), "The rigid cube must remain dynamic"
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_qd.numpy()))
        bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        assert np.all(np.isfinite(bag_q))
        bag_height = float(bag_q[:, 2].max() - bag_q[:, 2].min())
        assert bag_height < 0.5, f"Bag stretched excessively: height={bag_height:.3f} m"

    @staticmethod
    def create_parser():
        """Create parser options for the recorded rigid-cube placement."""
        parser = _m3_Example.create_parser()
        parser.set_defaults(num_frames=875, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(_m4_DEFAULT_GRASP_KEYFRAME),
            help="Rigid-cube grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def _m4_main():
    """Run the right-hand recorded rigid-cube bag-placement example."""
    parser = _m4_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m4_Example(viewer, args), args)


import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m5_FPS = 60
_m5_SIM_SUBSTEPS = 5
_m5_VBD_ITERATIONS = 24
_m5_RIGHT_HAND_URDF = Path(__file__).resolve().parents[3] / "assets" / "W1_right_hand" / "DexforceW1_right_hand.urdf"
_m5_HAND_HOME = wp.transform(
    wp.vec3(-0.15679353, -2.8874836, 1.3789376), wp.quat(-0.31233013, 0.67216527, 0.32775849, -0.58584785)
)
_m5_TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
_m5_TABLE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m5_TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
_m5_TABLE_TOP_Z = float(_m5_TABLE_POS[2]) + _m5_TABLE_HALF_EXTENTS[2]
_m5_CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
_m5_CUBE_CENTRE = wp.vec3(-0.14931439, -2.76669516, _m5_TABLE_TOP_Z + _m5_CUBE_HALF_EXTENTS[2] + 0.001)
_m5_CUBE_DIMS = (10, 6, 10)
_m5_CUBE_DENSITY = 300.0
_m5_CUBE_K_MU = 1000000.0
_m5_CUBE_K_LAMBDA = 3000000.0
_m5_CUBE_K_DAMP = 20.0
_m5_CUBE_PARTICLE_RADIUS = 0.0025
_m5_CUBE_SELF_CONTACT_RADIUS = 0.003
_m5_CUBE_SELF_CONTACT_MARGIN = 2.0 * _m5_CUBE_SELF_CONTACT_RADIUS
_m5_BAG_WIDTH = 0.2
_m5_BAG_DEPTH = 0.16
_m5_BAG_HEIGHT = 0.24
_m5_BAG_POS = wp.vec3(0.24068561, -2.79869516, 0.93122798)
_m5_BAG_RESOLUTION = 20
_m5_BAG_PARTICLE_RADIUS = 0.003
_m5_BAG_DENSITY = 0.08
_m5_BAG_TRI_KE = 150.0
_m5_BAG_TRI_KA = 150.0
_m5_BAG_TRI_KD = 0.5
_m5_BAG_EDGE_KE = 0.5
_m5_BAG_EDGE_KD = 1.5e-05
_m5_CONTACT_KE = 30000.0
_m5_CONTACT_KD = 0.5
_m5_CONTACT_MU = 50.0
_m5_CONTACT_MARGIN = 0.003
_m5_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
_m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE = 128
_m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE = 256
_m5_POSITION_LIMIT_MM = 500.0
_m5_CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
_m5_CAMERA_PITCH = -18.0
_m5_CAMERA_YAW = 126.0
_m5_HAND_JOINTS = (
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
_m5_RECORDED_JOINT_DEGREES = {
    "RIGHT_HAND_THUMB1": 12.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}


def _m5__generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
    """Generate the five faces of an open-topped cloth box."""
    cell_x = 2.0 * half_x / resolution
    cell_y = 2.0 * half_y / resolution
    cell_z = height / resolution
    vertex_map: dict[tuple[float, float, float], int] = {}
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []

    def vertex(x: float, y: float, z: float):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    def quad(v00: int, v10: int, v01: int, v11: int):
        indices.extend((v00, v10, v01, v10, v11, v01))

    for i in range(resolution):
        for j in range(resolution):
            x0, y0 = (-half_x + i * cell_x, -half_y + j * cell_y)
            x1, y1 = (x0 + cell_x, y0 + cell_y)
            quad(vertex(x0, y0, 0.0), vertex(x1, y0, 0.0), vertex(x0, y1, 0.0), vertex(x1, y1, 0.0))
    for i in range(resolution):
        for j in range(resolution):
            x0, x1 = (-half_x + i * cell_x, -half_x + (i + 1) * cell_x)
            y0, y1 = (-half_y + i * cell_y, -half_y + (i + 1) * cell_y)
            z0, z1 = (j * cell_z, (j + 1) * cell_z)
            quad(vertex(x0, -half_y, z0), vertex(x1, -half_y, z0), vertex(x0, -half_y, z1), vertex(x1, -half_y, z1))
            quad(vertex(x1, half_y, z0), vertex(x0, half_y, z0), vertex(x1, half_y, z1), vertex(x0, half_y, z1))
            quad(vertex(-half_x, y1, z0), vertex(-half_x, y0, z0), vertex(-half_x, y1, z1), vertex(-half_x, y0, z1))
            quad(vertex(half_x, y0, z0), vertex(half_x, y1, z0), vertex(half_x, y0, z1), vertex(half_x, y1, z1))
    return (np.asarray(vertices, dtype=np.float32), indices)


@wp.kernel
def _m5__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m5__joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
    joint = wp.tid()
    q_begin, q_end = (joint_q_start[joint], joint_q_start[joint + 1])
    qd_begin, qd_end = (joint_qd_start[joint], joint_qd_start[joint + 1])
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
        for i in range(qd_end - qd_begin):
            if q_begin + i < q_end:
                out[qd_begin + i] = (q1[q_begin + i] - q0[q_begin + i]) * inv_dt


@wp.kernel
def _m5__pin_bag_particles(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    i = wp.tid()
    particle = pinned_indices[i]
    pos_0[particle] = original_positions[i]
    pos_1[particle] = original_positions[i]


class _m5_Example:
    """Interactive right-hand physical recorder for a soft-cube grasp."""

    RIGID_BODY_CONTACT_BUFFER_SIZE = 64

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.include_bag = getattr(self, "include_bag", False)
        self.particle_self_contact_enabled = getattr(self, "particle_self_contact_enabled", True)
        self.frame_dt = 1.0 / _m5_FPS
        self.sim_dt = self.frame_dt / _m5_SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self._root = None
        self._status_var = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._last_target_signature: tuple[float, ...] | None = None
        self._initial_keyframe = self._load_initial_keyframe()
        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        if self.include_bag:
            self._initialize_bag_pin()
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m5_VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": self.RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": _m5_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": self.particle_self_contact_enabled,
                "particle_self_contact_radius": _m5_CUBE_SELF_CONTACT_RADIUS,
                "particle_self_contact_margin": _m5_CUBE_SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": _m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": _m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": _m5_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )
        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m5_HAND_HOME)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(_m5_RECORDED_JOINT_DEGREES)
        self._restore_initial_controls()
        self.joint_limits = self._joint_limits()
        self.target_transform = self._copy_transform(_m5_HAND_HOME)
        self._refresh_target()
        self._set_initial_hand_pose()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(_m5_CAMERA_POS, _m5_CAMERA_PITCH, _m5_CAMERA_YAW)

    def _build_scene(self):
        if not _m5_RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {_m5_RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m5_CONTACT_KE
        builder.default_shape_cfg.kd = _m5_CONTACT_KD
        builder.default_shape_cfg.mu = _m5_CONTACT_MU
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m5_RIGHT_HAND_URDF),
            xform=_m5_HAND_HOME,
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
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=300000.0, kd=0.0001, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(_m5_TABLE_POS, _m5_TABLE_ROTATION),
            hx=_m5_TABLE_HALF_EXTENTS[0],
            hy=_m5_TABLE_HALF_EXTENTS[1],
            hz=_m5_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="hand_tuning_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="hand_tuning_ground")
        if self.include_bag:
            bag_vertices, bag_indices = _m5__generate_box_bag(
                0.5 * _m5_BAG_WIDTH, 0.5 * _m5_BAG_DEPTH, _m5_BAG_HEIGHT, _m5_BAG_RESOLUTION
            )
            self.bag_particle_start = builder.particle_count
            builder.add_cloth_mesh(
                pos=_m5_BAG_POS,
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=bag_vertices.tolist(),
                indices=bag_indices,
                density=_m5_BAG_DENSITY,
                tri_ke=_m5_BAG_TRI_KE,
                tri_ka=_m5_BAG_TRI_KA,
                tri_kd=_m5_BAG_TRI_KD,
                edge_ke=_m5_BAG_EDGE_KE,
                edge_kd=_m5_BAG_EDGE_KD,
                particle_radius=_m5_BAG_PARTICLE_RADIUS,
                label="open_soft_box_bag",
            )
            self.bag_particle_end = builder.particle_count
            bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - _m5_BAG_HEIGHT) < 1e-05)
            self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start
        cube_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        cube_origin = _m5_CUBE_CENTRE - wp.quat_rotate(cube_rotation, wp.vec3(*_m5_CUBE_HALF_EXTENTS))
        self.cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=cube_origin,
            rot=cube_rotation,
            vel=wp.vec3(),
            dim_x=_m5_CUBE_DIMS[0],
            dim_y=_m5_CUBE_DIMS[1],
            dim_z=_m5_CUBE_DIMS[2],
            cell_x=2.0 * _m5_CUBE_HALF_EXTENTS[0] / _m5_CUBE_DIMS[0],
            cell_y=2.0 * _m5_CUBE_HALF_EXTENTS[1] / _m5_CUBE_DIMS[1],
            cell_z=2.0 * _m5_CUBE_HALF_EXTENTS[2] / _m5_CUBE_DIMS[2],
            density=_m5_CUBE_DENSITY,
            k_mu=_m5_CUBE_K_MU,
            k_lambda=_m5_CUBE_K_LAMBDA,
            k_damp=_m5_CUBE_K_DAMP,
            particle_radius=_m5_CUBE_PARTICLE_RADIUS,
            label="tunable_soft_cube",
        )
        self.cube_particle_end = builder.particle_count
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(self.hand_shape_end):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        for shape in range(self.hand_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m5_CONTACT_KE
        self.model.soft_contact_kd = _m5_CONTACT_KD
        self.model.soft_contact_mu = _m5_CONTACT_MU

    def _initialize_bag_pin(self):
        """Pin the open bag rim at its recorded world positions."""
        pinned_indices = self.bag_top_indices
        flags = self.model.particle_flags.numpy()
        flags[pinned_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        particle_q = self.state_0.particle_q.numpy()
        self.bag_pinned_indices = wp.array(pinned_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_original = wp.array(particle_q[pinned_indices].copy(), dtype=wp.vec3, device=self.device)

    def _root_joint_index(self):
        types = self.model.joint_type.numpy()
        parents = self.model.joint_parent.numpy()
        for index, (joint_type, parent) in enumerate(zip(types, parents, strict=True)):
            if int(joint_type) == int(newton.JointType.FREE) and int(parent) == -1:
                return index
        raise RuntimeError("Right-hand URDF must import with a free root joint")

    def _hand_joint_indices(self):
        labels = self.model.joint_label
        starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        indices = {}
        self.hand_joint_limit_indices = {}
        for name in _m5_HAND_JOINTS:
            joint = next((index for index, label in enumerate(labels) if label.endswith("/" + name)))
            indices[name] = int(starts[joint])
            self.hand_joint_limit_indices[name] = int(dof_starts[joint])
        return indices

    def _joint_limits(self):
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        return {
            name: tuple(
                sorted(
                    (
                        float(np.degrees(lower[self.hand_joint_limit_indices[name]])),
                        float(np.degrees(upper[self.hand_joint_limit_indices[name]])),
                    )
                )
            )
            for name in _m5_HAND_JOINTS
        }

    def _default_joint_degrees(self):
        q = self.model.joint_q.numpy()
        return {name: float(np.degrees(q[index])) for name, index in self.hand_joint_indices.items()}

    @staticmethod
    def _copy_transform(transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    def _load_initial_keyframe(self):
        path = Path(self.args.keyframe_output).expanduser()
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload.get("keyframe")
        return keyframe if isinstance(keyframe, dict) else None

    def _restore_initial_controls(self):
        """Restore the gizmo, relative offsets, and finger targets of a keyframe."""
        keyframe = self._initial_keyframe
        if keyframe is None:
            return
        gizmo = keyframe.get("gizmo_world")
        if isinstance(gizmo, dict):
            position = gizmo.get("position_m")
            rotation = gizmo.get("quaternion_xyzw")
            if (
                isinstance(position, list)
                and len(position) == 3
                and isinstance(rotation, list)
                and (len(rotation) == 4)
            ):
                self.gizmo_transform = wp.transform(wp.vec3(*position), wp.quat(*rotation))
        position_offset = keyframe.get("position_offset_mm")
        if isinstance(position_offset, list) and len(position_offset) == 3:
            self.position_mm = np.asarray(position_offset, dtype=np.float32)
        rotation_offset = keyframe.get("rotation_offset_deg")
        if isinstance(rotation_offset, list) and len(rotation_offset) == 3:
            self.rotation_deg = np.asarray(rotation_offset, dtype=np.float32)
        joints = keyframe.get("target_finger_joints_degrees")
        if isinstance(joints, dict):
            for name in _m5_HAND_JOINTS:
                if name in joints:
                    self.joint_degrees[name] = float(joints[name])

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

    def _offset_transform(self):
        base_position = np.asarray(wp.transform_get_translation(self.gizmo_transform), dtype=np.float32)
        position = base_position + self.position_mm * 0.001
        rx, ry, rz = np.radians(self.rotation_deg)
        rotation = self._quat_mul(
            self._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        return wp.transform(
            wp.vec3(*position), self._quat_mul(rotation, wp.transform_get_rotation(self.gizmo_transform))
        )

    def _refresh_target(self):
        self.target_transform = self._offset_transform()
        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(self.target_transform)
        rotation = wp.transform_get_rotation(self.target_transform)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(self.joint_degrees[name])
        self.manual_target_q.assign(target_q)
        self._last_target_signature = self._target_signature()

    def _set_initial_hand_pose(self):
        """Initialize the physical hand at the recorded grasp keyframe."""
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

    def _target_signature(self):
        return tuple(
            [*wp.transform_get_translation(self.gizmo_transform), *wp.transform_get_rotation(self.gizmo_transform)]
            + self.position_mm.tolist()
            + self.rotation_deg.tolist()
            + [self.joint_degrees[name] for name in _m5_HAND_JOINTS]
        )

    def step_once(self):
        """Advance one real-time physical frame toward the current hand target."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(_m5_SIM_SUBSTEPS):
            if self.include_bag:
                wp.launch(
                    _m5__pin_bag_particles,
                    self.bag_pinned_indices.shape[0],
                    [
                        self.bag_pinned_indices,
                        self.bag_pinned_original,
                        self.state_0.particle_q,
                        self.state_1.particle_q,
                    ],
                    device=self.device,
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m5_SIM_SUBSTEPS
            wp.launch(
                _m5__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m5__joint_velocity,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = (self.state_1, self.state_0)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _reset_physics(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        if self.include_bag:
            self._initialize_bag_pin()
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m5_HAND_HOME)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(_m5_RECORDED_JOINT_DEGREES)
        self._restore_initial_controls()
        self.sim_time = 0.0
        self.frame_index = 0
        self._trajectory_frames.clear()
        self._refresh_target()
        self._set_initial_hand_pose()

    def _transform_dict(self, transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return {
            "position_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in rotation],
        }

    def _capture_frame(self):
        current_q = self.state_0.joint_q.numpy()
        root_q = current_q[self.root_q_start : self.root_q_start + 7]
        return {
            "frame": self.frame_index,
            "time_s": self.sim_time,
            "gizmo_world": self._transform_dict(self.gizmo_transform),
            "position_offset_mm": self.position_mm.tolist(),
            "rotation_offset_deg": self.rotation_deg.tolist(),
            "target_root_pose": self._transform_dict(self.target_transform),
            "target_finger_joints_degrees": dict(self.joint_degrees),
            "root_pose": {
                "position_m": [float(value) for value in root_q[:3]],
                "quaternion_xyzw": [float(value) for value in root_q[3:]],
            },
            "finger_joints_radians": {name: float(current_q[index]) for name, index in self.hand_joint_indices.items()},
            "finger_joints_degrees": {
                name: float(np.degrees(current_q[index])) for name, index in self.hand_joint_indices.items()
            },
            "soft_cube_particles": int(self.cube_particle_end - self.cube_particle_start),
        }

    def _store_trajectory_frame(self):
        frame = self._capture_frame()
        if self._trajectory_frames and self._trajectory_frames[-1]["frame"] == frame["frame"]:
            self._trajectory_frames[-1] = frame
        else:
            self._trajectory_frames.append(frame)

    @staticmethod
    def _write_json(path_value: str, payload: dict[str, Any]):
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_pose(self):
        path = self._write_json(
            self.args.pose_output, {"format": "newton_w1_right_hand_pose_v1", "pose": self._capture_frame()}
        )
        self._set_status(f"Saved pose: {path}")

    def save_trajectory(self):
        path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        self._set_status(f"Saved trajectory: {path}")

    def render(self):
        if self._target_signature() != self._last_target_signature:
            self._refresh_target()
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("right_hand_target", self.gizmo_transform)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _set_status(self, message: str):
        if self._status_var is not None:
            self._status_var.set(message)

    def _on_control_changed(self, variables):
        for name, variable in variables["joints"].items():
            self.joint_degrees[name] = float(variable.get())
        self.position_mm = np.asarray([float(variable.get()) for variable in variables["position"]], dtype=np.float32)
        self.rotation_deg = np.asarray([float(variable.get()) for variable in variables["rotation"]], dtype=np.float32)
        self._refresh_target()
        self.render()

    def _make_scale(self, parent, row: int, label: str, variable, minimum: float, maximum: float, command):
        import tkinter as tk

        self._ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        value_label = self._ttk.Label(parent, textvariable=variable._display_var, width=8, anchor="e")
        value_label.grid(row=row, column=1, sticky="e", padx=3)
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=1.0,
            orient="horizontal",
            showvalue=False,
            length=440,
            highlightthickness=0,
            command=command,
        )
        scale.grid(row=row, column=2, sticky="ew", padx=4)

        def update_display(*_):
            variable._display_var.set(f"{float(variable.get()):.1f}")

        variable.trace_add("write", update_display)
        update_display()

    def _build_controls(self, root):
        import tkinter as tk

        frame = self._ttk.Frame(root, padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 0))
        frame.columnconfigure(2, weight=1)
        variables = {"joints": {}, "position": [], "rotation": []}
        joints = self._ttk.LabelFrame(frame, text="RIGHT finger joint angles (degrees)", padding=5)
        joints.grid(row=0, column=0, columnspan=3, sticky="nsew")
        joints.columnconfigure(2, weight=1)
        for row, name in enumerate(_m5_HAND_JOINTS):
            variable = tk.DoubleVar(value=self.joint_degrees[name])
            variable._display_var = tk.StringVar()
            variables["joints"][name] = variable
            lower, upper = self.joint_limits[name]
            self._make_scale(
                joints,
                row,
                name,
                variable,
                lower,
                upper,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        root_box = self._ttk.LabelFrame(frame, text="Whole-hand target offset / rotation relative to gizmo", padding=5)
        root_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root_box.columnconfigure(2, weight=1)
        for index, label in enumerate(("Position X (mm)", "Position Y (mm)", "Position Z (mm)")):
            variable = tk.DoubleVar(value=float(self.position_mm[index]))
            variable._display_var = tk.StringVar()
            variables["position"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -_m5_POSITION_LIMIT_MM,
                _m5_POSITION_LIMIT_MM,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        for index, label in enumerate(("Rotation X (deg)", "Rotation Y (deg)", "Rotation Z (deg)"), start=3):
            variable = tk.DoubleVar(value=float(self.rotation_deg[index - 3]))
            variable._display_var = tk.StringVar()
            variables["rotation"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -180.0,
                180.0,
                lambda _value, variables=variables: self._on_control_changed(variables),
            )
        return variables

    def run_recorder(self):
        if self.args.recorder_no_gui:
            self.render()
            self.viewer.close()
            return
        import tkinter as tk
        from tkinter import ttk

        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()
        self._ttk = ttk
        root = tk.Tk()
        self._root = root
        root.title("MJVBD-v2 W1 right-hand soft-cube recorder")
        root.geometry("710x650")
        root.minsize(660, 630)
        self._build_controls(root)
        buttons = ttk.Frame(root, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Record keyframe", command=self._record_keyframe_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset physics", command=self._reset_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save pose JSON", command=self.save_pose).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save trajectory", command=self.save_trajectory).pack(side="left", padx=3)
        self._status_var = tk.StringVar(
            value="Realtime physics running; move the gizmo or sliders, then record a keyframe."
        )
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.step_once()
            self.render()
            root.after(max(1, int(1000.0 / _m5_FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _record_keyframe_from_ui(self):
        self._store_trajectory_frame()
        keyframe_path = self._write_json(
            self.args.keyframe_output,
            {"format": "newton_w1_right_hand_keyframe_v1", "keyframe": self._trajectory_frames[-1]},
        )
        trajectory_path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        contact_count = int(self.solver.contacts.soft_contact_count.numpy()[0])
        self._set_status(
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; solved hand/soft contacts: {contact_count}. Saved: {keyframe_path}, {trajectory_path}"
        )

    def _reset_from_ui(self):
        self._reset_physics()
        self._set_status("Reset hand and soft cube to the recorded initial keyframe.")
        self.render()

    def test_final(self):
        """Verify that one manual physical step keeps all states finite."""
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        parser.add_argument(
            "--pose-output",
            default=str(
                Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_pose.json"
            ),
        )
        parser.add_argument(
            "--trajectory-output",
            default=str(
                Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_trajectory.json"
            ),
        )
        parser.add_argument(
            "--keyframe-output",
            default=str(
                Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
            ),
        )
        parser.add_argument("--recorder-no-gui", action="store_true")
        return parser


def _m5_main():
    """Launch the interactive right-hand soft-cube recorder."""
    parser = _m5_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = _m5_Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m6_DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
)
_m6_APPROACH_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
_m6_APPROACH_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}
_m6_OPEN_JOINTS = dict.fromkeys(_m5_HAND_JOINTS, 0.0)
_m6_START_JOINTS = dict(_m6_OPEN_JOINTS)
_m6_START_JOINTS["RIGHT_HAND_THUMB2"] = 90.0
_m5_CUBE_DENSITY = 100.0
_m5_CUBE_DIMS = _m1_SOFT_CUBE_DIMS
_m5_CUBE_K_MU = _m1_SOFT_CUBE_K_MU
_m5_CUBE_K_LAMBDA = _m1_SOFT_CUBE_K_LAMBDA
_m5_CUBE_K_DAMP = _m1_SOFT_CUBE_K_DAMP
_m5_CUBE_PARTICLE_RADIUS = _m1_SOFT_CUBE_PARTICLE_RADIUS
_m5_CUBE_SELF_CONTACT_RADIUS = 0.003
_m5_CUBE_SELF_CONTACT_MARGIN = 0.006
_m5_CONTACT_KE = _m1_GRASP_CONTACT_KE
_m5_CONTACT_KD = _m1_GRASP_CONTACT_KD
_m5_CONTACT_MU = _m1_GRASP_FRICTION
_m5_CONTACT_MARGIN = _m1_SOFT_CONTACT_MARGIN
_m5_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE
_m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE = 32
_m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE = 64
_m5_BAG_RESOLUTION = _m1_BAG_RESOLUTION
_m5_BAG_PARTICLE_RADIUS = _m1_BAG_PARTICLE_RADIUS
_m5_BAG_DENSITY = _m1_BAG_DENSITY
_m5_BAG_TRI_KE = 50000.0
_m5_BAG_TRI_KA = 50000.0
_m5_BAG_TRI_KD = 50.0
_m5_BAG_EDGE_KE = 25.0
_m5_BAG_EDGE_KD = 0.25


class _m6_Example(_m5_Example):
    """Run the recorded right-hand grasp and physical bag placement."""

    def __init__(self, viewer, args):
        self.include_bag = True
        self.particle_self_contact_enabled = True
        self.grasp_root, self.grasp_joints = self._load_grasp_keyframe(args.grasp_keyframe)
        self.release_friction_applied = False
        self.hand_soft_contact_enabled = True
        super().__init__(viewer, args)
        self._set_hand_target(_m6_APPROACH_ROOT, _m6_START_JOINTS)
        self._set_initial_hand_pose()
        self._create_reference_solver()
        self._set_hand_soft_contact(False)
        self.segments = self._build_segments()
        self.script_duration = sum(segment[0] for segment in self.segments)

    def _create_reference_solver(self):
        """Match the full-W1 cube-bag self-contact configuration."""
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m1_VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": max(_m1_BAG_PARTICLE_RADIUS, _m1_SOFT_CUBE_PARTICLE_RADIUS),
                "particle_self_contact_margin": 2.0 * max(_m1_BAG_PARTICLE_RADIUS, _m1_SOFT_CUBE_PARTICLE_RADIUS),
                "particle_topological_contact_filter_threshold": 3,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": _m1_SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )

    @staticmethod
    def _load_grasp_keyframe(path_value: str):
        """Load the most recently recorded physical grasp keyframe."""
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recorded grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload["keyframe"]
        root = keyframe["target_root_pose"]
        position = root["position_m"]
        rotation = root["quaternion_xyzw"]
        joints = keyframe["target_finger_joints_degrees"]
        if len(position) != 3 or len(rotation) != 4:
            raise ValueError(f"Invalid root pose in recorded grasp keyframe: {path}")
        return (
            wp.transform(wp.vec3(*position), wp.quat(*rotation)),
            {name: float(value) for name, value in joints.items()},
        )

    def _set_hand_target(self, root: wp.transform, joints: dict[str, float]):
        """Set the next kinematic root and five-finger target without moving particles."""
        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(joints[name])
        self.manual_target_q.assign(target_q)

    def _build_segments(self):
        """Build approach, closure, transport, release, and retreat phases."""
        grasp_position = wp.transform_get_translation(_m6_APPROACH_ROOT)
        cube_position = _m5_CUBE_CENTRE
        root_cube_offset = grasp_position - cube_position
        cube_release_height = float(_m5_BAG_POS[2]) + _m5_BAG_HEIGHT + 0.06
        bag_hover = wp.transform(
            wp.vec3(
                float(_m5_BAG_POS[0]) + float(root_cube_offset[0]),
                float(_m5_BAG_POS[1]) + float(root_cube_offset[1]),
                cube_release_height + float(root_cube_offset[2]),
            ),
            wp.transform_get_rotation(_m6_APPROACH_ROOT),
        )
        lift = wp.transform(grasp_position + wp.vec3(0.0, 0.0, 0.07), wp.transform_get_rotation(_m6_APPROACH_ROOT))
        retreat = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(_m6_APPROACH_ROOT),
        )
        return (
            (0.5, _m6_APPROACH_ROOT, _m6_APPROACH_ROOT, _m6_START_JOINTS, _m6_START_JOINTS, False),
            (1.5, _m6_APPROACH_ROOT, _m6_APPROACH_ROOT, _m6_START_JOINTS, _m6_APPROACH_JOINTS, False),
            (0.5, _m6_APPROACH_ROOT, _m6_APPROACH_ROOT, _m6_APPROACH_JOINTS, _m6_APPROACH_JOINTS, False),
            (1.8, _m6_APPROACH_ROOT, _m6_APPROACH_ROOT, _m6_APPROACH_JOINTS, self.grasp_joints, False),
            (0.6, _m6_APPROACH_ROOT, _m6_APPROACH_ROOT, self.grasp_joints, self.grasp_joints, False),
            (1.2, _m6_APPROACH_ROOT, lift, self.grasp_joints, self.grasp_joints, False),
            (7.0, lift, bag_hover, self.grasp_joints, self.grasp_joints, False),
            (0.4, bag_hover, bag_hover, self.grasp_joints, self.grasp_joints, False),
            (0.25, bag_hover, bag_hover, self.grasp_joints, _m6_OPEN_JOINTS, True),
            (0.9, bag_hover, bag_hover, _m6_OPEN_JOINTS, _m6_OPEN_JOINTS, True),
            (1.0, bag_hover, retreat, _m6_OPEN_JOINTS, _m6_OPEN_JOINTS, True),
        )

    def _sample(self, time_s: float):
        """Interpolate the recorded root and joint targets at a script time."""
        for duration, root_a, root_b, joints_a, joints_b, release in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in _m5_HAND_JOINTS}
                return (root, joints, release)
            time_s -= duration
        _, _, root, _, joints, release = self.segments[-1]
        return (root, joints, release)

    @staticmethod
    def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float):
        position_a = np.asarray(wp.transform_get_translation(a), dtype=np.float32)
        position_b = np.asarray(wp.transform_get_translation(b), dtype=np.float32)
        rotation_a = np.asarray(wp.transform_get_rotation(a), dtype=np.float32)
        rotation_b = np.asarray(wp.transform_get_rotation(b), dtype=np.float32)
        if np.dot(rotation_a, rotation_b) < 0.0:
            rotation_b = -rotation_b
        rotation = rotation_a * (1.0 - alpha) + rotation_b * alpha
        rotation /= max(np.linalg.norm(rotation), 1e-08)
        return wp.transform(wp.vec3(*position_a * (1.0 - alpha) + position_b * alpha), wp.quat(*rotation))

    def _apply_release_friction(self):
        """Match the full-W1 release material after opening the hand."""
        if self.release_friction_applied:
            return
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[: self.hand_shape_end] = _m1_RELEASE_CONTACT_KE
        shape_kd[: self.hand_shape_end] = _m1_RELEASE_CONTACT_KD
        shape_mu[: self.hand_shape_end] = _m1_RELEASE_FRICTION
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self.model.soft_contact_ke = _m1_RELEASE_CONTACT_KE
        self.model.soft_contact_kd = _m1_RELEASE_CONTACT_KD
        self.model.soft_contact_mu = _m1_RELEASE_FRICTION
        self.release_friction_applied = True

    def _set_hand_soft_contact(self, enabled: bool):
        """Match the reference contact material before and during the pinch."""
        if enabled == self.hand_soft_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[: self.hand_shape_end] |= particle_flag
            self.model.soft_contact_ke = _m1_GRASP_CONTACT_KE
            self.model.soft_contact_kd = _m1_GRASP_CONTACT_KD
            self.model.soft_contact_mu = _m1_GRASP_SOFT_CONTACT_MU
        else:
            flags[: self.hand_shape_end] &= ~particle_flag
            self.model.soft_contact_ke = _m1_SOFT_CONTACT_KE
            self.model.soft_contact_kd = _m1_SOFT_CONTACT_KD
            self.model.soft_contact_mu = _m1_SOFT_CONTACT_MU
        self.model.shape_flags.assign(flags)
        self.hand_soft_contact_enabled = enabled

    def step(self):
        """Advance the autonomous recorded-pose trajectory by one physical frame."""
        root, joints, release = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        if self.sim_time >= 0.5:
            self._set_hand_soft_contact(True)
        if release:
            self._apply_release_friction()
        self.step_once()

    def render(self):
        """Render the physical hand, cube, and bag without the tuning gizmo."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite hand, soft-cube, and bag particle states."""
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        if self.frame_index >= 30:
            bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
            bag_height = float(bag_q[:, 2].max() - bag_q[:, 2].min())
            assert bag_height < 0.4, f"Bag stretched before release: height={bag_height:.3f} m"
        if self.sim_time >= self.script_duration:
            cube_q = self.state_0.particle_q.numpy()[self.cube_particle_start : self.cube_particle_end]
            cube_centre = cube_q.mean(axis=0)
            inside = (
                abs(float(cube_centre[0]) - float(_m5_BAG_POS[0])) < 0.5 * _m5_BAG_WIDTH
                and abs(float(cube_centre[1]) - float(_m5_BAG_POS[1])) < 0.5 * _m5_BAG_DEPTH
                and (float(_m5_BAG_POS[2]) - 0.02 < float(cube_centre[2]) < float(_m5_BAG_POS[2]) + _m5_BAG_HEIGHT)
            )
            assert inside, f"Soft cube missed the bag: centre={tuple(float(value) for value in cube_centre)}"

    @staticmethod
    def create_parser():
        """Create parser options for the right-hand recorded grasp demo."""
        parser = _m5_Example.create_parser()
        parser.set_defaults(num_frames=1020, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(_m6_DEFAULT_GRASP_KEYFRAME),
            help="Latest grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def _m6_main():
    """Run the right-hand-only physical soft-cube bag-placement example."""
    parser = _m6_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m6_Example(viewer, args), args)


import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m7_DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
)
_m7_FREE_SOFT_CONTACT = (_m1_SOFT_CONTACT_KE, _m1_SOFT_CONTACT_KD, _m1_SOFT_CONTACT_MU)
_m7_GRASP_SOFT_CONTACT = (_m1_GRASP_CONTACT_KE, _m1_GRASP_CONTACT_KD, _m1_GRASP_SOFT_CONTACT_MU)
_m7_RIGID_GRASP_CONTACT = (_m0_GRASP_CONTACT_KE, _m0_GRASP_CONTACT_KD, _m0_GRASP_FRICTION)
_m7_RIGID_RELEASE_CONTACT = (_m0_RELEASE_CONTACT_KE, _m0_RELEASE_CONTACT_KD, _m0_RELEASE_FRICTION)
_m5_CUBE_DENSITY = _m1_SOFT_CUBE_DENSITY
_m5_CUBE_DIMS = _m1_SOFT_CUBE_DIMS
_m5_CUBE_K_MU = _m1_SOFT_CUBE_K_MU
_m5_CUBE_K_LAMBDA = _m1_SOFT_CUBE_K_LAMBDA
_m5_CUBE_K_DAMP = _m1_SOFT_CUBE_K_DAMP
_m5_CUBE_PARTICLE_RADIUS = _m1_SOFT_CUBE_PARTICLE_RADIUS
_m5_CUBE_SELF_CONTACT_RADIUS = max(_m1_BAG_PARTICLE_RADIUS, _m1_SOFT_CUBE_PARTICLE_RADIUS)
_m5_CUBE_SELF_CONTACT_MARGIN = 2.0 * _m5_CUBE_SELF_CONTACT_RADIUS
_m5_CONTACT_KE = _m7_GRASP_SOFT_CONTACT[0]
_m5_CONTACT_KD = _m7_GRASP_SOFT_CONTACT[1]
_m5_CONTACT_MU = _m1_GRASP_FRICTION
_m5_CONTACT_MARGIN = _m1_SOFT_CONTACT_MARGIN
_m5_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE
_m5_BAG_RESOLUTION = _m1_BAG_RESOLUTION
_m5_BAG_PARTICLE_RADIUS = _m1_BAG_PARTICLE_RADIUS
_m5_BAG_DENSITY = _m1_BAG_DENSITY
_m5_BAG_TRI_KE = _m1_BAG_TRI_KE
_m5_BAG_TRI_KA = _m1_BAG_TRI_KA
_m5_BAG_TRI_KD = _m1_BAG_TRI_KD
_m5_BAG_EDGE_KE = _m1_BAG_EDGE_KE
_m5_BAG_EDGE_KD = _m1_BAG_EDGE_KD
_m7_APPROACH_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
_m7_APPROACH_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}
_m7_OPEN_JOINTS = dict.fromkeys(_m5_HAND_JOINTS, 0.0)
_m7_RIGID_GRASP_JOINTS = dict(_m5_RECORDED_JOINT_DEGREES)
_m7_RIGID_CUBE_OFFSET = wp.vec3(-0.11, 0.0, 0.0)
_m7_RIGID_CUBE_CENTRE = _m5_CUBE_CENTRE + _m7_RIGID_CUBE_OFFSET
_m7_RIGID_GRASP_ROOT = wp.transform(
    wp.vec3(-0.2589742839336395, -2.834425926208496, 1.3404709100723267),
    wp.quat(0.0508713573217392, 0.9490510821342468, -0.3089625239372253, 0.03544386848807335),
)
_m7_RIGID_CUBE_DENSITY = _m0_CUBE_DENSITY
_m7_RIGID_CUBE_MARGIN = _m0_CUBE_MARGIN
_m7_RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
_m7_RIGID_FINGER_PADS = (
    (
        "right_thumb_dist",
        (0.03, 0.006, 0.015),
        wp.transform(wp.vec3(-0.0548988, 0.0529312, 0.0141373), wp.quat(0.277395, -0.870077, 0.0411925, 0.4053648)),
    ),
    (
        "right_index_dist",
        (0.03, 0.006, 0.015),
        wp.transform(wp.vec3(-0.0388362, 0.0073618, -0.0145817), wp.quat(0.0581093, -0.6604385, 0.7410694, 0.1061119)),
    ),
    (
        "right_middle_dist",
        (0.014, 0.006, 0.012),
        wp.transform(wp.vec3(-0.0004232, 0.0163224, 0.0208014), wp.quat(0.0982183, -0.9514985, 0.2900473, 0.0295879)),
    ),
    (
        "right_ring_dist",
        (0.014, 0.006, 0.012),
        wp.transform(wp.vec3(-0.0013989, 0.0131392, 0.0240774), wp.quat(0.0912622, -0.9611189, 0.2605852, -0.0040627)),
    ),
    (
        "right_pinky_dist",
        (0.014, 0.006, 0.012),
        wp.transform(wp.vec3(-0.0025118, 0.0185027, 0.0180359), wp.quat(0.0746882, -0.96611, 0.2468487, -0.0108671)),
    ),
)


class _m7_Example(_m5_Example):
    """Run two physical five-finger pick-and-place operations with one hand."""

    def __init__(self, viewer, args):
        self.include_bag = True
        self.particle_self_contact_enabled = True
        self.grasp_joints = self._load_grasp_joints(args.grasp_keyframe)
        self.hand_soft_contact_enabled = True
        self.soft_release_applied = False
        self.rigid_release_applied = False
        self.rigid_grasp_material_applied = False
        super().__init__(viewer, args)
        self._set_hand_target(_m7_APPROACH_ROOT, _m7_APPROACH_JOINTS)
        self._set_initial_hand_pose()
        self._set_hand_soft_contact(False)
        self._set_hand_shape_friction(0.0)
        self._create_solver()
        self.segments = self._build_segments()
        self.script_duration = sum(segment[0] for segment in self.segments)

    @staticmethod
    def _load_grasp_joints(path_value: str) -> dict[str, float]:
        """Load the recorded five-finger closure target."""
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recorded grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        joints = payload["keyframe"]["target_finger_joints_degrees"]
        return {name: float(value) for name, value in joints.items()}

    def _set_hand_target(self, root: wp.transform, joints: dict[str, float]):
        """Set the floating-hand root and finger target for the next frame."""
        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(joints[name])
        self.manual_target_q.assign(target_q)

    def _create_solver(self):
        """Allocate contact buffers sized for the hand's many rigid colliders."""
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m5_VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": _m7_RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": _m5_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": self.particle_self_contact_enabled,
                "particle_self_contact_radius": _m5_CUBE_SELF_CONTACT_RADIUS,
                "particle_self_contact_margin": _m5_CUBE_SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": _m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": _m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": _m5_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )

    def _build_scene(self):
        """Build the recorded soft-cube scene plus one dynamic rigid cube."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m5_CONTACT_KE
        builder.default_shape_cfg.kd = _m5_CONTACT_KD
        builder.default_shape_cfg.mu = _m5_CONTACT_MU
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m5_RIGHT_HAND_URDF),
            xform=_m5_HAND_HOME,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.hand_articulations = tuple(range(articulation_start, builder.articulation_count))
        self.hand_soft_shape_end = builder.shape_count
        finger_pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=_m7_RIGID_GRASP_CONTACT[0],
            kd=_m7_RIGID_GRASP_CONTACT[1],
            mu=_m7_RIGID_GRASP_CONTACT[2],
            is_visible=False,
        )
        self.rigid_finger_pad_start = builder.shape_count
        for body_name, half_extents, pad_xform in _m7_RIGID_FINGER_PADS:
            body = next((index for index, label in enumerate(builder.body_label) if label.endswith(body_name)))
            builder.add_shape_box(
                body,
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=finger_pad_cfg,
                xform=pad_xform,
                label=f"{body_name}_physical_pad",
            )
        self.rigid_finger_pad_end = builder.shape_count
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=300000.0, kd=0.0001, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(_m5_TABLE_POS, _m5_TABLE_ROTATION),
            hx=_m5_TABLE_HALF_EXTENTS[0],
            hy=_m5_TABLE_HALF_EXTENTS[1],
            hz=_m5_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="two_pick_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="two_pick_ground")
        bag_vertices, bag_indices = _m5__generate_box_bag(
            0.5 * _m5_BAG_WIDTH, 0.5 * _m5_BAG_DEPTH, _m5_BAG_HEIGHT, _m5_BAG_RESOLUTION
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=_m5_BAG_POS,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=_m5_BAG_DENSITY,
            tri_ke=_m5_BAG_TRI_KE,
            tri_ka=_m5_BAG_TRI_KA,
            tri_kd=_m5_BAG_TRI_KD,
            edge_ke=_m5_BAG_EDGE_KE,
            edge_kd=_m5_BAG_EDGE_KD,
            particle_radius=_m5_BAG_PARTICLE_RADIUS,
            label="two_pick_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - _m5_BAG_HEIGHT) < 1e-05)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start
        cube_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        cube_origin = _m5_CUBE_CENTRE - wp.quat_rotate(cube_rotation, wp.vec3(*_m5_CUBE_HALF_EXTENTS))
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=cube_origin,
            rot=cube_rotation,
            vel=wp.vec3(),
            dim_x=_m5_CUBE_DIMS[0],
            dim_y=_m5_CUBE_DIMS[1],
            dim_z=_m5_CUBE_DIMS[2],
            cell_x=2.0 * _m5_CUBE_HALF_EXTENTS[0] / _m5_CUBE_DIMS[0],
            cell_y=2.0 * _m5_CUBE_HALF_EXTENTS[1] / _m5_CUBE_DIMS[1],
            cell_z=2.0 * _m5_CUBE_HALF_EXTENTS[2] / _m5_CUBE_DIMS[2],
            density=_m5_CUBE_DENSITY,
            k_mu=_m5_CUBE_K_MU,
            k_lambda=_m5_CUBE_K_LAMBDA,
            k_damp=_m5_CUBE_K_DAMP,
            particle_radius=_m5_CUBE_PARTICLE_RADIUS,
            label="first_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count
        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m7_RIGID_CUBE_DENSITY,
            ke=_m5_CONTACT_KE,
            kd=_m5_CONTACT_KD,
            mu=_m5_CONTACT_MU,
            margin=_m7_RIGID_CUBE_MARGIN,
        )
        rigid_cfg.configure_sdf(force_sdf=True)
        rigid_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(_m7_RIGID_CUBE_CENTRE, cube_rotation), label="second_rigid_cube"
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=_m5_CUBE_HALF_EXTENTS[0],
            hy=_m5_CUBE_HALF_EXTENTS[1],
            hz=_m5_CUBE_HALF_EXTENTS[2],
            cfg=rigid_cfg,
            color=(0.9, 0.32, 0.18),
            label="second_rigid_cube_shape",
        )
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        for shape in range(self.rigid_finger_pad_start, self.rigid_finger_pad_end):
            builder.shape_flags[shape] &= ~collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m5_CONTACT_KE
        self.model.soft_contact_kd = _m5_CONTACT_KD
        self.model.soft_contact_mu = _m5_CONTACT_MU

    def _set_hand_soft_contact(self, enabled: bool):
        """Toggle only hand-to-soft-particle contact for the first pick."""
        if enabled == self.hand_soft_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[: self.hand_soft_shape_end] |= particle_flag
            self.model.soft_contact_ke = _m7_GRASP_SOFT_CONTACT[0]
            self.model.soft_contact_kd = _m7_GRASP_SOFT_CONTACT[1]
            self.model.soft_contact_mu = _m7_GRASP_SOFT_CONTACT[2]
        else:
            flags[: self.hand_soft_shape_end] &= ~particle_flag
            self.model.soft_contact_ke = _m7_FREE_SOFT_CONTACT[0]
            self.model.soft_contact_kd = _m7_FREE_SOFT_CONTACT[1]
            self.model.soft_contact_mu = _m7_FREE_SOFT_CONTACT[2]
        self.model.shape_flags.assign(flags)
        self.hand_soft_contact_enabled = enabled

    def _set_hand_shape_friction(self, friction: float):
        """Set rigid-contact friction for the floating hand shapes."""
        values = self.model.shape_material_mu.numpy()
        values[: self.hand_shape_end] = friction
        self.model.shape_material_mu.assign(values)

    def _apply_soft_release_material(self):
        """Match the recorded soft-cube release contact and friction."""
        if self.soft_release_applied:
            return
        self._set_hand_shape_friction(0.0)
        self.model.soft_contact_ke = _m7_FREE_SOFT_CONTACT[0]
        self.model.soft_contact_kd = _m7_FREE_SOFT_CONTACT[1]
        self.model.soft_contact_mu = _m7_FREE_SOFT_CONTACT[2]
        self.soft_release_applied = True

    def _apply_rigid_grasp_material(self):
        """Match the rigid-cube reference material during the second pick."""
        if self.rigid_grasp_material_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[: self.hand_shape_end] = _m7_RIGID_GRASP_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = _m7_RIGID_GRASP_CONTACT[2]
        shape_ke[: self.hand_shape_end] = _m7_RIGID_GRASP_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = _m7_RIGID_GRASP_CONTACT[0]
        shape_kd[: self.hand_shape_end] = _m7_RIGID_GRASP_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = _m7_RIGID_GRASP_CONTACT[1]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.rigid_grasp_material_applied = True

    def _apply_rigid_release_material(self):
        """Match the rigid reference release material after opening the hand."""
        if self.rigid_release_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[: self.hand_shape_end] = _m7_RIGID_RELEASE_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = _m7_RIGID_RELEASE_CONTACT[2]
        shape_ke[: self.hand_shape_end] = _m7_RIGID_RELEASE_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = _m7_RIGID_RELEASE_CONTACT[0]
        shape_kd[: self.hand_shape_end] = _m7_RIGID_RELEASE_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = _m7_RIGID_RELEASE_CONTACT[1]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.rigid_release_applied = True

    def _build_segments(self):
        """Build two recorded five-finger pick-and-place sequences."""
        soft_approach = _m7_APPROACH_ROOT
        rigid_approach = _m7_RIGID_GRASP_ROOT
        rigid_pregrasp = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.18),
            wp.transform_get_rotation(rigid_approach),
        )
        soft_root_cube_offset = wp.transform_get_translation(soft_approach) - _m5_CUBE_CENTRE
        rigid_root_cube_offset = wp.transform_get_translation(rigid_approach) - _m7_RIGID_CUBE_CENTRE
        cube_release_height = float(_m5_BAG_POS[2]) + _m5_BAG_HEIGHT + 0.06
        soft_bag_hover = wp.transform(
            wp.vec3(
                float(_m5_BAG_POS[0]) + float(soft_root_cube_offset[0]),
                float(_m5_BAG_POS[1]) + float(soft_root_cube_offset[1]),
                cube_release_height + float(soft_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_bag_hover = wp.transform(
            wp.vec3(
                float(_m5_BAG_POS[0]) + float(rigid_root_cube_offset[0]),
                float(_m5_BAG_POS[1]) + float(rigid_root_cube_offset[1]),
                cube_release_height + float(rigid_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(rigid_approach),
        )
        soft_lift = wp.transform(
            wp.transform_get_translation(soft_approach) + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_lift = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(rigid_approach),
        )
        rigid_transport = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.05),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        soft_retreat = wp.transform(
            wp.transform_get_translation(soft_bag_hover) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(soft_bag_hover),
        )
        rigid_retreat = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        return (
            (0.5, soft_approach, soft_approach, _m7_APPROACH_JOINTS, _m7_APPROACH_JOINTS, "soft_wait"),
            (1.8, soft_approach, soft_approach, _m7_APPROACH_JOINTS, self.grasp_joints, "soft_grasp"),
            (0.6, soft_approach, soft_approach, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (1.2, soft_approach, soft_lift, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (7.0, soft_lift, soft_bag_hover, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (0.4, soft_bag_hover, soft_bag_hover, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (
                _m1_SOFT_CUBE_RELEASE_OPEN_DURATION,
                soft_bag_hover,
                soft_bag_hover,
                self.grasp_joints,
                _m7_OPEN_JOINTS,
                "soft_release",
            ),
            (
                _m1_SOFT_CUBE_RELEASE_SETTLE_DURATION,
                soft_bag_hover,
                soft_bag_hover,
                _m7_OPEN_JOINTS,
                _m7_OPEN_JOINTS,
                "soft_release",
            ),
            (1.0, soft_bag_hover, soft_retreat, _m7_OPEN_JOINTS, _m7_OPEN_JOINTS, "soft_release"),
            (0.8, soft_retreat, rigid_pregrasp, _m7_OPEN_JOINTS, _m7_OPEN_JOINTS, "rigid_approach"),
            (1.8, rigid_pregrasp, rigid_approach, _m7_OPEN_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_grasp"),
            (1.0, rigid_approach, rigid_approach, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (2.0, rigid_approach, rigid_lift, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.6, rigid_lift, rigid_lift, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (5.0, rigid_lift, rigid_transport, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (1.5, rigid_transport, rigid_bag_hover, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.4, rigid_bag_hover, rigid_bag_hover, _m7_RIGID_GRASP_JOINTS, _m7_RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.8, rigid_bag_hover, rigid_bag_hover, _m7_RIGID_GRASP_JOINTS, _m7_OPEN_JOINTS, "rigid_release"),
            (0.2, rigid_bag_hover, rigid_bag_hover, _m7_OPEN_JOINTS, _m7_OPEN_JOINTS, "rigid_release"),
            (1.0, rigid_bag_hover, rigid_retreat, _m7_OPEN_JOINTS, _m7_OPEN_JOINTS, "rigid_release"),
        )

    def _sample(self, time_s: float):
        """Interpolate one hand target from the two-pick script."""
        for duration, root_a, root_b, joints_a, joints_b, phase in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in _m5_HAND_JOINTS}
                return (root, joints, phase)
            time_s -= duration
        _, _, root, _, joints, phase = self.segments[-1]
        return (root, joints, phase)

    @staticmethod
    def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float):
        """Linearly interpolate position and normalize the interpolated quaternion."""
        position_a = np.asarray(wp.transform_get_translation(a), dtype=np.float32)
        position_b = np.asarray(wp.transform_get_translation(b), dtype=np.float32)
        rotation_a = np.asarray(wp.transform_get_rotation(a), dtype=np.float32)
        rotation_b = np.asarray(wp.transform_get_rotation(b), dtype=np.float32)
        if np.dot(rotation_a, rotation_b) < 0.0:
            rotation_b = -rotation_b
        rotation = rotation_a * (1.0 - alpha) + rotation_b * alpha
        rotation /= max(np.linalg.norm(rotation), 1e-08)
        return wp.transform(wp.vec3(*position_a * (1.0 - alpha) + position_b * alpha), wp.quat(*rotation))

    def step(self):
        """Advance the two physical pick-and-place operations by one frame."""
        root, joints, phase = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        soft_grasp = phase in {"soft_grasp", "soft_carry"}
        self._set_hand_soft_contact(soft_grasp or phase == "soft_release")
        if soft_grasp:
            self._set_hand_shape_friction(_m5_CONTACT_MU)
        if phase in {"rigid_approach", "rigid_grasp", "rigid_carry", "rigid_release"}:
            self._apply_rigid_grasp_material()
        if phase == "soft_release":
            self._apply_soft_release_material()
        if phase == "rigid_release":
            self._apply_rigid_release_material()
        self.step_once()

    def render(self):
        """Render the hand, two cubes, and soft bag without the recorder controls."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite physical state after the sequential placements."""
        body_flags = int(self.model.body_flags.numpy()[self.rigid_cube_body])
        assert not body_flags & int(newton.BodyFlags.KINEMATIC), (
            "The rigid cube must be dynamic when the solver is constructed"
        )
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))

    @staticmethod
    def create_parser():
        """Create parser options for the sequential soft-and-rigid pick demo."""
        parser = _m5_Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(_m7_DEFAULT_GRASP_KEYFRAME),
            help="Latest grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def _m7_main():
    """Run the right-hand sequential soft-and-rigid pick demo."""
    parser = _m7_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m7_Example(viewer, args), args)


from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m8_DEFAULT_RIGID_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"
)
_m8_HAND_JOINTS = _m5_HAND_JOINTS
_m8_IDLE_JOINTS = dict.fromkeys(_m8_HAND_JOINTS, 0.0)
_m8_IDLE_JOINTS["RIGHT_HAND_THUMB2"] = 90.0
_m8_OPEN_JOINTS = dict.fromkeys(_m8_HAND_JOINTS, 0.0)
_m8_SOFT_GRASP_CONTACT = (_m1_GRASP_CONTACT_KE, _m1_GRASP_CONTACT_KD, _m1_GRASP_SOFT_CONTACT_MU)
_m8_SOFT_FREE_CONTACT = (_m1_SOFT_CONTACT_KE, _m1_SOFT_CONTACT_KD, _m1_SOFT_CONTACT_MU)
_m8_RIGID_GRASP_CONTACT = (_m3_CONTACT_KE, _m3_CONTACT_KD, _m3_CONTACT_MU)
_m8_RIGID_RELEASE_CONTACT = (_m4_RELEASE_CONTACT_KE, _m4_RELEASE_CONTACT_KD, _m4_RELEASE_FRICTION)
_m8_SOFT_CUBE_CENTRE = _m5_CUBE_CENTRE
_m8_SOFT_CUBE_DIMS = _m5_CUBE_DIMS
_m8_SOFT_CUBE_DENSITY = _m5_CUBE_DENSITY
_m8_SOFT_CUBE_K_MU = _m5_CUBE_K_MU
_m8_SOFT_CUBE_K_LAMBDA = _m5_CUBE_K_LAMBDA
_m8_SOFT_CUBE_K_DAMP = _m5_CUBE_K_DAMP
_m8_SOFT_CUBE_PARTICLE_RADIUS = _m5_CUBE_PARTICLE_RADIUS
_m8_RIGID_CUBE_CENTRE = _m7_RIGID_CUBE_CENTRE
_m8_RIGID_CUBE_DENSITY = _m3_CUBE_DENSITY
_m8_RIGID_CUBE_MARGIN = _m3_CONTACT_MARGIN
_m8_BAG_RESOLUTION = _m5_BAG_RESOLUTION
_m8_BAG_PARTICLE_RADIUS = _m5_BAG_PARTICLE_RADIUS
_m8_BAG_DENSITY = _m5_BAG_DENSITY
_m8_BAG_TRI_KE = _m5_BAG_TRI_KE
_m8_BAG_TRI_KA = _m5_BAG_TRI_KA
_m8_BAG_TRI_KD = _m5_BAG_TRI_KD
_m8_BAG_EDGE_KE = _m5_BAG_EDGE_KE
_m8_BAG_EDGE_KD = _m5_BAG_EDGE_KD
_m5_SIM_SUBSTEPS = max(_m5_SIM_SUBSTEPS, _m3_SIM_SUBSTEPS)
_m5_VBD_ITERATIONS = max(_m5_VBD_ITERATIONS, _m3_VBD_ITERATIONS)


class _m8_Example(_m7_Example):
    """Run the recorded soft-cube pick followed by the recorded rigid pick."""

    def __init__(self, viewer, args):
        rigid_root, self.rigid_grasp_joints, _ = _m4_Example._load_grasp_keyframe(args.rigid_grasp_keyframe)
        rigid_root_position = wp.transform_get_translation(rigid_root)
        self.rigid_grasp_root = wp.transform(
            rigid_root_position + _m8_RIGID_CUBE_CENTRE - _m3_CUBE_CENTRE, wp.transform_get_rotation(rigid_root)
        )
        super().__init__(viewer, args)
        self.hand_rigid_contact_enabled = True
        self._set_hand_target(_m6_APPROACH_ROOT, _m8_IDLE_JOINTS)
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)

    def _create_solver(self):
        """Create one solver for soft-soft, rigid-soft, and rigid-rigid contact."""
        contact_radius = max(_m8_BAG_PARTICLE_RADIUS, _m8_SOFT_CUBE_PARTICLE_RADIUS)
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": _m3_VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0005,
                "rigid_contact_stick_freeze_translation_eps": 0.0002,
                "rigid_contact_stick_freeze_angular_eps": 0.0002,
                "rigid_body_contact_buffer_size": _m3_RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": contact_radius,
                "particle_self_contact_margin": 2.0 * contact_radius,
                "particle_vertex_contact_buffer_size": _m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": _m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "soft_contact_margin": _m1_SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )

    def _build_scene(self):
        """Build one URDF hand, two cubes, and a pinned soft bag without pads."""
        recorder = sequential_base.recorder
        if not recorder.RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {recorder.RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m8_SOFT_GRASP_CONTACT[0]
        builder.default_shape_cfg.kd = _m8_SOFT_GRASP_CONTACT[1]
        builder.default_shape_cfg.mu = _m1_GRASP_FRICTION
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(recorder.RIGHT_HAND_URDF),
            xform=recorder.HAND_HOME,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.hand_articulations = tuple(range(articulation_start, builder.articulation_count))
        self.hand_soft_shape_end = builder.shape_count
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=300000.0, kd=0.0001, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(recorder.TABLE_POS, recorder.TABLE_ROTATION),
            hx=recorder.TABLE_HALF_EXTENTS[0],
            hy=recorder.TABLE_HALF_EXTENTS[1],
            hz=recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="recorded_two_pick_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="recorded_two_pick_ground")
        bag_vertices, bag_indices = recorder._generate_box_bag(
            0.5 * recorder.BAG_WIDTH, 0.5 * recorder.BAG_DEPTH, recorder.BAG_HEIGHT, _m8_BAG_RESOLUTION
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=recorder.BAG_POS,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=_m8_BAG_DENSITY,
            tri_ke=_m8_BAG_TRI_KE,
            tri_ka=_m8_BAG_TRI_KA,
            tri_kd=_m8_BAG_TRI_KD,
            edge_ke=_m8_BAG_EDGE_KE,
            edge_kd=_m8_BAG_EDGE_KD,
            particle_radius=_m8_BAG_PARTICLE_RADIUS,
            label="recorded_two_pick_soft_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - recorder.BAG_HEIGHT) < 1e-05)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start
        soft_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        soft_origin = _m8_SOFT_CUBE_CENTRE - wp.quat_rotate(soft_rotation, wp.vec3(*recorder.CUBE_HALF_EXTENTS))
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=soft_origin,
            rot=soft_rotation,
            vel=wp.vec3(),
            dim_x=_m8_SOFT_CUBE_DIMS[0],
            dim_y=_m8_SOFT_CUBE_DIMS[1],
            dim_z=_m8_SOFT_CUBE_DIMS[2],
            cell_x=2.0 * recorder.CUBE_HALF_EXTENTS[0] / _m8_SOFT_CUBE_DIMS[0],
            cell_y=2.0 * recorder.CUBE_HALF_EXTENTS[1] / _m8_SOFT_CUBE_DIMS[1],
            cell_z=2.0 * recorder.CUBE_HALF_EXTENTS[2] / _m8_SOFT_CUBE_DIMS[2],
            density=_m8_SOFT_CUBE_DENSITY,
            k_mu=_m8_SOFT_CUBE_K_MU,
            k_lambda=_m8_SOFT_CUBE_K_LAMBDA,
            k_damp=_m8_SOFT_CUBE_K_DAMP,
            particle_radius=_m8_SOFT_CUBE_PARTICLE_RADIUS,
            label="first_recorded_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count
        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m8_RIGID_CUBE_DENSITY,
            ke=_m8_RIGID_GRASP_CONTACT[0],
            kd=_m8_RIGID_GRASP_CONTACT[1],
            mu=_m8_RIGID_GRASP_CONTACT[2],
            margin=_m8_RIGID_CUBE_MARGIN,
        )
        rigid_cfg.configure_sdf(force_sdf=True)
        rigid_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(_m8_RIGID_CUBE_CENTRE, wp.quat_identity()), label="second_recorded_rigid_cube"
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=recorder.CUBE_HALF_EXTENTS[0],
            hy=recorder.CUBE_HALF_EXTENTS[1],
            hz=recorder.CUBE_HALF_EXTENTS[2],
            cfg=rigid_cfg,
            color=(0.9, 0.32, 0.18),
            label="second_recorded_rigid_cube_shape",
        )
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m8_SOFT_FREE_CONTACT[0]
        self.model.soft_contact_kd = _m8_SOFT_FREE_CONTACT[1]
        self.model.soft_contact_mu = _m8_SOFT_FREE_CONTACT[2]

    def _set_hand_rigid_contact(self, enabled: bool):
        """Gate hand-to-rigid contact while moving between the two objects."""
        if enabled == self.hand_rigid_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        rigid_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        if enabled:
            flags[: self.hand_shape_end] |= rigid_flag
        else:
            flags[: self.hand_shape_end] &= ~rigid_flag
        self.model.shape_flags.assign(flags)
        self.hand_rigid_contact_enabled = enabled

    def _apply_rigid_grasp_material(self):
        """Restore the standalone rigid recorder's mesh-contact material."""
        if self.rigid_grasp_material_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_margin = self.model.shape_margin.numpy()
        shape_mu[: self.hand_shape_end] = _m8_RIGID_GRASP_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = _m8_RIGID_GRASP_CONTACT[2]
        shape_ke[: self.hand_shape_end] = _m8_RIGID_GRASP_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = _m8_RIGID_GRASP_CONTACT[0]
        shape_kd[: self.hand_shape_end] = _m8_RIGID_GRASP_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = _m8_RIGID_GRASP_CONTACT[1]
        shape_margin[: self.hand_shape_end] = _m8_RIGID_CUBE_MARGIN
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_margin.assign(shape_margin)
        self.model.soft_contact_ke = _m4_SOFT_CONTACT_KE
        self.model.soft_contact_kd = _m4_SOFT_CONTACT_KD
        self.model.soft_contact_mu = _m4_SOFT_CONTACT_MU
        self.rigid_grasp_material_applied = True

    def _apply_rigid_release_material(self):
        """Remove mesh and rigid-soft friction for the second release."""
        if self.rigid_release_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[: self.hand_shape_end] = _m8_RIGID_RELEASE_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = _m8_RIGID_RELEASE_CONTACT[2]
        shape_ke[: self.hand_shape_end] = _m8_RIGID_RELEASE_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = _m8_RIGID_RELEASE_CONTACT[0]
        shape_kd[: self.hand_shape_end] = _m8_RIGID_RELEASE_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = _m8_RIGID_RELEASE_CONTACT[1]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.soft_contact_ke = _m8_RIGID_RELEASE_CONTACT[0]
        self.model.soft_contact_kd = _m8_RIGID_RELEASE_CONTACT[1]
        self.model.soft_contact_mu = _m8_RIGID_RELEASE_CONTACT[2]
        self.rigid_release_applied = True

    def _build_segments(self):
        """Build two idle-to-pre-grasp-to-recorded-grasp trajectories."""
        recorder = sequential_base.recorder
        soft_approach = _m6_APPROACH_ROOT
        rigid_approach = self.rigid_grasp_root
        soft_root_cube_offset = wp.transform_get_translation(soft_approach) - _m8_SOFT_CUBE_CENTRE
        rigid_root_cube_offset = wp.transform_get_translation(rigid_approach) - _m8_RIGID_CUBE_CENTRE
        release_height = float(recorder.BAG_POS[2]) + recorder.BAG_HEIGHT + 0.06
        soft_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(soft_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(soft_root_cube_offset[1]),
                release_height + float(soft_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(rigid_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(rigid_root_cube_offset[1]),
                release_height + float(rigid_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(rigid_approach),
        )
        soft_lift = wp.transform(
            wp.transform_get_translation(soft_approach) + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_lift = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(rigid_approach),
        )
        rigid_transport = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.05),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        soft_retreat = wp.transform(
            wp.transform_get_translation(soft_bag_hover) + wp.vec3(0.0, 0.0, 0.1),
            wp.transform_get_rotation(soft_bag_hover),
        )
        rigid_retreat = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.12),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        soft_grasp = self.grasp_joints
        rigid_grasp = self.rigid_grasp_joints
        rigid_pregrasp = _m3_INITIAL_HAND_JOINTS
        return (
            (0.5, soft_approach, soft_approach, _m8_IDLE_JOINTS, _m8_IDLE_JOINTS, "soft_wait"),
            (1.5, soft_approach, soft_approach, _m8_IDLE_JOINTS, _m6_APPROACH_JOINTS, "soft_prepare"),
            (0.5, soft_approach, soft_approach, _m6_APPROACH_JOINTS, _m6_APPROACH_JOINTS, "soft_prepare"),
            (1.8, soft_approach, soft_approach, _m6_APPROACH_JOINTS, soft_grasp, "soft_grasp"),
            (0.6, soft_approach, soft_approach, soft_grasp, soft_grasp, "soft_carry"),
            (1.2, soft_approach, soft_lift, soft_grasp, soft_grasp, "soft_carry"),
            (7.0, soft_lift, soft_bag_hover, soft_grasp, soft_grasp, "soft_carry"),
            (0.4, soft_bag_hover, soft_bag_hover, soft_grasp, soft_grasp, "soft_carry"),
            (0.25, soft_bag_hover, soft_bag_hover, soft_grasp, _m8_OPEN_JOINTS, "soft_release"),
            (0.9, soft_bag_hover, soft_bag_hover, _m8_OPEN_JOINTS, _m8_OPEN_JOINTS, "soft_release"),
            (1.0, soft_bag_hover, soft_retreat, _m8_OPEN_JOINTS, _m8_OPEN_JOINTS, "soft_release"),
            (1.5, soft_retreat, rigid_approach, _m8_OPEN_JOINTS, _m8_IDLE_JOINTS, "rigid_move"),
            (0.5, rigid_approach, rigid_approach, _m8_IDLE_JOINTS, _m8_IDLE_JOINTS, "rigid_prepare"),
            (1.5, rigid_approach, rigid_approach, _m8_IDLE_JOINTS, rigid_pregrasp, "rigid_prepare"),
            (0.5, rigid_approach, rigid_approach, rigid_pregrasp, rigid_pregrasp, "rigid_prepare"),
            (0.45, rigid_approach, rigid_approach, rigid_pregrasp, rigid_grasp, "rigid_grasp"),
            (0.3, rigid_approach, rigid_approach, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.75, rigid_approach, rigid_lift, rigid_grasp, rigid_grasp, "rigid_carry"),
            (5.0, rigid_lift, rigid_transport, rigid_grasp, rigid_grasp, "rigid_carry"),
            (1.2, rigid_transport, rigid_bag_hover, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.5, rigid_bag_hover, rigid_bag_hover, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.8, rigid_bag_hover, rigid_bag_hover, rigid_grasp, _m8_OPEN_JOINTS, "rigid_release"),
            (1.5, rigid_bag_hover, rigid_bag_hover, _m8_OPEN_JOINTS, _m8_OPEN_JOINTS, "rigid_release"),
            (1.0, rigid_bag_hover, rigid_retreat, _m8_OPEN_JOINTS, _m8_OPEN_JOINTS, "rigid_release"),
        )

    def _sample(self, time_s: float):
        """Sample the canonical shared hand-root and finger trajectory."""
        return _m8__sample_recorded_trajectory(self.segments, time_s)

    def step(self):
        """Advance the two recorded physical picks by one frame."""
        root, joints, phase = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        soft_contact = phase in {"soft_prepare", "soft_grasp", "soft_carry", "soft_release"}
        self._set_hand_soft_contact(soft_contact)
        self._set_hand_rigid_contact(phase != "rigid_move")
        if phase in {"soft_prepare", "soft_grasp", "soft_carry"}:
            self._set_hand_shape_friction(_m1_GRASP_FRICTION)
        if phase in {"rigid_prepare", "rigid_grasp", "rigid_carry", "rigid_release"}:
            self._apply_rigid_grasp_material()
        if phase == "soft_release":
            self._apply_soft_release_material()
        if phase == "rigid_release":
            self._apply_rigid_release_material()
        self.step_once()

    def test_final(self):
        """Verify finite mixed-body state and the absence of added fingertip pads."""
        super().test_final()
        assert not any("physical_pad" in label for label in self.model.shape_label)

    @staticmethod
    def create_parser():
        """Create parser options for both recorded grasp keyframes."""
        parser = _m7_Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--rigid-grasp-keyframe",
            default=str(_m8_DEFAULT_RIGID_GRASP_KEYFRAME),
            help="Rigid-cube keyframe generated by the standalone right-hand recorder.",
        )
        return parser


def _m8__build_recorded_trajectory(
    soft_grasp_joints: dict[str, float], rigid_grasp_root: wp.transform, rigid_grasp_joints: dict[str, float]
):
    """Build the canonical hand-root and finger-joint trajectory."""
    inputs = SimpleNamespace(
        grasp_joints=soft_grasp_joints, rigid_grasp_root=rigid_grasp_root, rigid_grasp_joints=rigid_grasp_joints
    )
    return _m8_Example._build_segments(inputs)


def _m8__sample_recorded_trajectory(segments, time_s: float):
    """Sample the canonical hand-root and finger-joint trajectory."""
    for duration, root_a, root_b, joints_a, joints_b, phase in segments:
        if time_s <= duration:
            alpha = float(np.clip(time_s / duration, 0.0, 1.0))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            root = _m8_Example._lerp_transform(root_a, root_b, alpha)
            joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in _m8_HAND_JOINTS}
            return (root, joints, phase)
        time_s -= duration
    _, _, root, _, joints, phase = segments[-1]
    return (root, joints, phase)


def _m8_main():
    """Run the mesh-only sequential soft-then-rigid placement example."""
    parser = _m8_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m8_Example(viewer, args), args)


"Run the recorded soft-then-rigid placement trajectory on the full W1.\n\nThe right arm tracks the exact hand-root and finger-joint trajectory from\n``example_vbd_mjvbd_v2_right_hand_recorded_soft_then_rigid_cube_into_bag``.\nEach hand-root target is converted to the W1 right-arm TCP and solved with IK.\nBoth cubes remain physical, and only collision meshes imported from the W1\nURDF participate in either grasp; no auxiliary fingertip pads are created.\n\nRun from the repository root::\n\n    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag --viewer gl\n"
import argparse

import numpy as np
import warp as wp

import newton
import newton.examples

soft0 = recorded_soft.soft0
RIGID_CUBE_POSITION = wp.vec3(
    float(soft0.CUBE_POSITIONS[0][0]), float(soft0.CUBE_POSITIONS[0][1]) + 0.11, float(soft0.CUBE_POSITIONS[0][2])
)
soft0.SIM_SUBSTEPS = 8
soft0.VBD_ITERATIONS = 12
soft0.BAG_RESOLUTION = _m8_BAG_RESOLUTION
soft0.BAG_PARTICLE_RADIUS = _m8_BAG_PARTICLE_RADIUS
soft0.BAG_DENSITY = _m8_BAG_DENSITY
soft0.BAG_TRI_KE = _m8_BAG_TRI_KE
soft0.BAG_TRI_KA = _m8_BAG_TRI_KA
soft0.BAG_TRI_KD = _m8_BAG_TRI_KD
soft0.BAG_EDGE_KE = _m8_BAG_EDGE_KE
soft0.BAG_EDGE_KD = _m8_BAG_EDGE_KD
_SOFT_MATERIAL_FREE = 0
_SOFT_MATERIAL_GRASP = 1
_SOFT_MATERIAL_RIGID_RELEASE = 2
_SOFT_CONTACT_MATERIALS = (
    tuple(float(value) for value in _m8_SOFT_FREE_CONTACT),
    tuple(float(value) for value in _m8_SOFT_GRASP_CONTACT),
    tuple(float(value) for value in _m8_RIGID_RELEASE_CONTACT),
)
SOFT_CUBE_COLOR = (0.2, 0.8, 0.2)
BOX_BAG_COLOR = (1.0, 1.0, 1.0)
SOFT_CONTACT_MAX = 4096
RIGID_CONTACT_MAX = 2048


@wp.kernel
def _copy_joint_q_prefix(source: wp.array[float], target: wp.array[float]):
    joint_coord = wp.tid()
    target[joint_coord] = source[joint_coord]


class Example(_m2_Example):
    """Track the canonical two-pick trajectory with the full Dexforce W1."""

    def __init__(self, viewer, args):
        rigid_root, self.rigid_grasp_joints, _ = _m4_Example._load_grasp_keyframe(args.rigid_grasp_keyframe)
        rigid_root_position = wp.transform_get_translation(rigid_root)
        self.rigid_grasp_root_world = wp.transform(
            rigid_root_position + _m8_RIGID_CUBE_CENTRE - _m3_CUBE_CENTRE, wp.transform_get_rotation(rigid_root)
        )
        self.contact_phase = None
        self.hand_shape_collision_enabled = True
        super().__init__(viewer, args)
        self.frame_q_end_2d = self.frame_q_end.reshape((1, -1))
        self.desired_finger_q = wp.zeros(self.hand_indices.shape[0], dtype=wp.float32, device=self.model.device)
        triangles = self.model.tri_indices.numpy().reshape((-1, 3))
        bag_mask = np.all((triangles >= self.bag_particle_start) & (triangles < self.bag_particle_end), axis=1)
        soft_cube_mask = np.all(
            (triangles >= self.soft_cube_particle_start) & (triangles < self.soft_cube_particle_end), axis=1
        )
        self.bag_render_triangle_indices = wp.array(triangles[bag_mask].reshape(-1), dtype=int, device=self.device)
        self.soft_cube_render_triangle_indices = wp.array(
            triangles[soft_cube_mask].reshape(-1), dtype=int, device=self.device
        )
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        self.graph = None
        self.ik_graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        if self.use_graph and self.sim_substeps % 2 != 0:
            raise ValueError("--graph-capture requires an even simulation substep count")
        if self.use_graph:
            self._capture_ik_graph()
        self.soft_contact_materials = wp.array(
            np.asarray(_SOFT_CONTACT_MATERIALS, dtype=np.float32), dtype=wp.vec3, device=self.device
        )
        self.soft_contact_material_index = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.solver.vbd_solver.set_soft_contact_material_source(
            self.soft_contact_materials, self.soft_contact_material_index
        )
        self.object_released = np.zeros(2, dtype=bool)
        self._apply_contact_phase("soft_wait")

    def _finger_pad_specs(self):
        """Disable the base example's auxiliary fingertip pads."""
        return ()

    def _add_additional_finger_pads(self, builder):
        """Keep the full-W1 scene limited to URDF collision meshes."""

    def _solver_vbd_options(self):
        """Combine the canonical particle and rigid-contact solver settings."""
        options = super()._solver_vbd_options()
        options.update(
            iterations=soft0.VBD_ITERATIONS,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_contact_stick_motion_eps=0.0005,
            rigid_contact_stick_freeze_translation_eps=0.0002,
            rigid_contact_stick_freeze_angular_eps=0.0002,
            rigid_body_contact_buffer_size=_m3_RIGID_BODY_CONTACT_BUFFER_SIZE,
            particle_vertex_contact_buffer_size=_m5_PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
            particle_edge_contact_buffer_size=_m5_PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
            particle_collision_detection_interval=0,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.03,
        )
        return options

    def _solver_collision_options(self):
        """Enable contact matching required by rigid-contact history."""
        options = super()._solver_collision_options()
        options.update(
            contact_matching="latest", soft_contact_max=SOFT_CONTACT_MAX, rigid_contact_max=RIGID_CONTACT_MAX
        )
        return options

    def _add_additional_scene_objects(self, builder):
        """Add the second dynamic cube with the canonical rigid material."""
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m8_RIGID_CUBE_DENSITY,
            ke=_m8_RIGID_GRASP_CONTACT[0],
            kd=_m8_RIGID_GRASP_CONTACT[1],
            mu=_m8_RIGID_GRASP_CONTACT[2],
            margin=_m8_RIGID_CUBE_MARGIN,
        )
        cube_cfg.configure_sdf(force_sdf=True)
        cube_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(self._world_vec(RIGID_CUBE_POSITION), self.base_rot), label="pick_recorded_rigid_cube"
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=_m5_CUBE_HALF_EXTENTS[0],
            hy=_m5_CUBE_HALF_EXTENTS[1],
            hz=_m5_CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            xform=wp.transform(wp.vec3(), _m0_CUBE_ROTATION),
            color=_m0_CUBE_COLORS[0],
            label="pick_recorded_rigid_cube_shape",
        )

    def _segments(self):
        """Convert the canonical hand-root segments to right-arm TCP segments."""
        soft_grasp_joints = {f"RIGHT_{suffix}": float(self.grasp_hand_q[suffix]) for suffix in self.HAND_SUFFIXES}
        self.hand_trajectory_segments = _m8__build_recorded_trajectory(
            soft_grasp_joints, self.rigid_grasp_root_world, self.rigid_grasp_joints
        )
        arm_segments = []
        for duration, root_a, root_b, _joints_a, _joints_b, phase in self.hand_trajectory_segments:
            right_a = self._root_to_tcp(root_a)
            right_b = self._root_to_tcp(root_b)
            if phase.startswith("soft_"):
                object_index = 0
            elif phase != "rigid_move":
                object_index = 1
            else:
                object_index = -1
            arm_segments.append((duration, self.left_home, self.left_home, right_a, right_b, 0.0, 0.0, object_index))
        return tuple(arm_segments)

    def _build_joint_target_cache(self):
        """Initialize the robot at the canonical idle hand and first root pose."""
        first_root = self.hand_trajectory_segments[0][1]
        approach = self._root_to_tcp(first_root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(approach))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(approach)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m2_INITIAL_IK_ITERATIONS)
        wp.launch(
            soft0._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )
        initial_q = self.model.joint_q.numpy()
        initial_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        initial_q[self.hand_indices.numpy()] = self.hand_start.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _set_shape_material(self, shapes, ke, kd, mu, margin=None):
        """Set effective contact material for selected model shapes."""
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[shapes] = mu
        shape_ke[shapes] = ke
        shape_kd[shapes] = kd
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        if margin is not None:
            shape_margin = self.model.shape_margin.numpy()
            shape_margin[shapes] = margin
            self.model.shape_margin.assign(shape_margin)

    def _set_shape_friction(self, shapes, friction):
        """Set friction for selected model shapes."""
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[shapes] = friction
        self.model.shape_material_mu.assign(shape_mu)

    def _set_hand_shape_collision(self, enabled):
        """Gate right-hand rigid contact during the inter-object move."""
        if enabled == self.hand_shape_collision_enabled:
            return
        flags = self.model.shape_flags.numpy()
        shape_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        if enabled:
            flags[self.right_hand_shapes] |= shape_flag
        else:
            flags[self.right_hand_shapes] &= ~shape_flag
        self.model.shape_flags.assign(flags)
        self.hand_shape_collision_enabled = enabled

    def _apply_contact_phase(self, phase):
        """Apply the canonical stage-specific contact configuration."""
        soft_active = phase in {"soft_prepare", "soft_grasp", "soft_carry"}
        rigid_active = phase in {"rigid_prepare", "rigid_grasp", "rigid_carry"}
        soft_material_index = _SOFT_MATERIAL_FREE
        self._set_hand_shape_collision(phase != "rigid_move")
        if soft_active:
            soft_material_index = _SOFT_MATERIAL_GRASP
            self._set_hand_particle_collision(True)
            self._set_shape_material(
                self.right_hand_shapes, _m8_SOFT_GRASP_CONTACT[0], _m8_SOFT_GRASP_CONTACT[1], soft0.GRASP_FRICTION
            )
            self.model.soft_contact_ke = _m8_SOFT_GRASP_CONTACT[0]
            self.model.soft_contact_kd = _m8_SOFT_GRASP_CONTACT[1]
            self.model.soft_contact_mu = _m8_SOFT_GRASP_CONTACT[2]
        elif phase == "soft_release":
            self._set_hand_particle_collision(True)
            self._set_shape_friction(self.right_hand_shapes, 0.0)
            self.model.soft_contact_ke = _m8_SOFT_FREE_CONTACT[0]
            self.model.soft_contact_kd = _m8_SOFT_FREE_CONTACT[1]
            self.model.soft_contact_mu = _m8_SOFT_FREE_CONTACT[2]
        elif rigid_active:
            self._set_hand_particle_collision(False)
            rigid_shapes = [*self.right_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                _m8_RIGID_GRASP_CONTACT[0],
                _m8_RIGID_GRASP_CONTACT[1],
                _m8_RIGID_GRASP_CONTACT[2],
                _m8_RIGID_CUBE_MARGIN,
            )
            self.model.soft_contact_ke = _m4_SOFT_CONTACT_KE
            self.model.soft_contact_kd = _m4_SOFT_CONTACT_KD
            self.model.soft_contact_mu = _m4_SOFT_CONTACT_MU
        elif phase == "rigid_release":
            soft_material_index = _SOFT_MATERIAL_RIGID_RELEASE
            self._set_hand_particle_collision(False)
            rigid_shapes = [*self.right_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                _m8_RIGID_RELEASE_CONTACT[0],
                _m8_RIGID_RELEASE_CONTACT[1],
                _m8_RIGID_RELEASE_CONTACT[2],
                _m8_RIGID_CUBE_MARGIN,
            )
            self.model.soft_contact_ke = _m8_RIGID_RELEASE_CONTACT[0]
            self.model.soft_contact_kd = _m8_RIGID_RELEASE_CONTACT[1]
            self.model.soft_contact_mu = _m8_RIGID_RELEASE_CONTACT[2]
        else:
            self._set_hand_particle_collision(False)
            self._set_shape_friction(self.right_hand_shapes, 0.0)
            self.model.soft_contact_ke = _m8_SOFT_FREE_CONTACT[0]
            self.model.soft_contact_kd = _m8_SOFT_FREE_CONTACT[1]
            self.model.soft_contact_mu = _m8_SOFT_FREE_CONTACT[2]
        self.soft_contact_material_index.fill_(soft_material_index)
        self.contact_phase = phase

    def _capture_simulation_graph(self):
        """Capture the warmed physics frame as one reusable CUDA graph."""
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph

    def _solve_ik_and_assemble_joint_targets(self):
        """Solve IK and assemble the complete joint target on the GPU."""
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=soft0.IK_ITERATIONS)
        wp.launch(
            soft0._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.state_0.joint_q)
        wp.launch(
            _copy_joint_q_prefix,
            self.ik_model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.model.device,
        )
        wp.launch(
            soft0._lock_q,
            self.hand_indices.shape[0],
            [self.frame_q_end_2d, self.hand_indices, self.desired_finger_q],
            device=self.model.device,
        )

    def _capture_ik_graph(self):
        """Capture IK and GPU joint-target assembly as one reusable graph."""
        ik_q_backup = wp.clone(self.ik_q)
        frame_q_start_backup = wp.clone(self.frame_q_start)
        frame_q_end_backup = wp.clone(self.frame_q_end)
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._solve_ik_and_assemble_joint_targets()
        self.ik_q.assign(ik_q_backup)
        self.frame_q_start.assign(frame_q_start_backup)
        self.frame_q_end.assign(frame_q_end_backup)
        self.ik_graph = capture.graph
        if self.ik_graph is None:
            raise RuntimeError(f"IK CUDA graph capture failed on device {self.device}.")

    def _prepare_frame(self):
        """Solve arm IK and copy the canonical finger target for one frame."""
        script_time = (self.frame_index + 1) * self.frame_dt * self.args.trajectory_time_scale
        left, right, _grip, _script_object = self._sample(script_time)
        _root, finger_joints, phase = _m8__sample_recorded_trajectory(self.hand_trajectory_segments, script_time)
        self.left_obj.set_target_position(0, wp.transform_get_translation(left))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(right))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right)))
        self.desired_finger_q.assign([np.radians(finger_joints[f"RIGHT_{suffix}"]) for suffix in self.HAND_SUFFIXES])
        if self.ik_graph is None:
            self._solve_ik_and_assemble_joint_targets()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.ik_graph)
        if phase != self.contact_phase:
            self._apply_contact_phase(phase)
        if phase == "soft_release":
            self.object_released[0] = True
        elif phase == "rigid_release":
            self.object_released[1] = True

    def step(self):
        """Advance with one warmed CUDA graph when available."""
        self._prepare_frame()
        if self.graph is None:
            self._simulate_substeps()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        """Render the bag and soft cube as separately colored solid meshes."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/box_bag",
            self.state_0.particle_q,
            self.bag_render_triangle_indices,
            backface_culling=False,
            color=BOX_BAG_COLOR,
        )
        self.viewer.log_mesh(
            "/soft_cube",
            self.state_0.particle_q,
            self.soft_cube_render_triangle_indices,
            backface_culling=False,
            color=SOFT_CUBE_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite objects, physical releases, placement, and no pads."""
        assert not any("physical_pad" in label for label in self.model.shape_label)
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        visual_flags = self.model.shape_flags.numpy()[self.robot_visual_shapes]
        assert np.all(visual_flags & collision_mask == 0), "Robot visual shapes must remain non-colliding"
        body_flags = int(self.model.body_flags.numpy()[self.rigid_cube_body])
        assert not body_flags & int(newton.BodyFlags.KINEMATIC), (
            "The rigid cube must be dynamic when the solver is constructed"
        )
        assert np.all(np.isfinite(self.state_0.body_q.numpy()[self.rigid_cube_body])), (
            "Rigid cube state contains non-finite values"
        )
        script_frames = int(
            np.ceil(sum(segment[0] for segment in self.segments) / (self.frame_dt * self.args.trajectory_time_scale))
        )
        if self.frame_index < script_frames:
            return
        super().test_final()
        rigid_world_position = wp.vec3(*self.state_0.body_q.numpy()[self.rigid_cube_body, :3])
        rigid_position = self._scene_vec(rigid_world_position)
        bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        bag_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in bag_q])
        bag_min_z = float(bag_scene_q[:, 2].min())
        half_extents = _m5_CUBE_HALF_EXTENTS
        rigid_inside = (
            abs(float(rigid_position[0]) - float(soft0.BAG_POS[0])) < 0.5 * soft0.BAG_WIDTH + half_extents[0]
            and abs(float(rigid_position[1]) - float(soft0.BAG_POS[1])) < 0.5 * soft0.BAG_DEPTH + half_extents[1]
            and (bag_min_z - half_extents[2] < float(rigid_position[2]) < soft0.TABLE_TOP_Z + 0.08)
        )
        assert rigid_inside, f"Rigid cube did not settle in the bag; position={tuple(rigid_position)}"

    @staticmethod
    def create_parser():
        """Create parser options for both canonical grasp keyframes."""
        parser = _m2_Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics frame as one CUDA graph.",
        )
        parser.add_argument(
            "--rigid-grasp-keyframe",
            default=str(_m8_DEFAULT_RIGID_GRASP_KEYFRAME),
            help="Rigid-cube keyframe used by the canonical isolated-hand trajectory.",
        )
        return parser


def main():
    """Run the full-W1 canonical soft-then-rigid placement trajectory."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
