# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactively tune a physical W1 right-hand soft-cube grasp.

The scene contains only a floating, kinematic W1 right hand and one volumetric
VBD soft cube. The hand's visible URDF collision meshes are SDF contact
boundaries for the soft cube; no particle pose is attached to or copied from
the hand.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_soft_cube_recorder --viewer gl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJVBDV2

FPS = 60
SIM_SUBSTEPS = 5
VBD_ITERATIONS = 24

RIGHT_HAND_URDF = Path(
    "E:/csy_work/CG/assets/W1-hand-obj/W1-hands-blender/urdf/W1_right_hand/DexforceW1_right_hand.urdf"
)
# Keep the existing grasp scene's world layout after removing the robot.
HAND_HOME = wp.transform(
    wp.vec3(-0.15679353, -2.88748360, 1.37893760),
    wp.quat(-0.31233013, 0.67216527, 0.32775849, -0.58584785),
)
TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
TABLE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
CUBE_CENTRE = wp.vec3(-0.14931439, -2.76669516, TABLE_TOP_Z + CUBE_HALF_EXTENTS[2] + 0.001)
CUBE_DIMS = (10, 6, 10)
CUBE_DENSITY = 300.0
CUBE_K_MU = 1.0e6
CUBE_K_LAMBDA = 3.0e6
CUBE_K_DAMP = 20.0
CUBE_PARTICLE_RADIUS = 0.0025
CUBE_SELF_CONTACT_RADIUS = 0.003
CUBE_SELF_CONTACT_MARGIN = 2.0 * CUBE_SELF_CONTACT_RADIUS

BAG_WIDTH = 0.20
BAG_DEPTH = 0.16
BAG_HEIGHT = 0.24
BAG_POS = wp.vec3(0.24068561, -2.79869516, 0.93122798)
BAG_RESOLUTION = 20
BAG_PARTICLE_RADIUS = 0.003
BAG_DENSITY = 0.08
BAG_TRI_KE = 1.5e2
BAG_TRI_KA = 1.5e2
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 0.5
BAG_EDGE_KD = 1.5e-5

CONTACT_KE = 3.0e4
CONTACT_KD = 0.5
CONTACT_MU = 50.0
CONTACT_MARGIN = 0.003
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
PARTICLE_VERTEX_CONTACT_BUFFER_SIZE = 128
PARTICLE_EDGE_CONTACT_BUFFER_SIZE = 256
POSITION_LIMIT_MM = 500.0
CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
CAMERA_PITCH = -18.0
CAMERA_YAW = 126.0

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

RECORDED_JOINT_DEGREES = {
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


def _generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
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
            x0, y0 = -half_x + i * cell_x, -half_y + j * cell_y
            x1, y1 = x0 + cell_x, y0 + cell_y
            quad(vertex(x0, y0, 0.0), vertex(x1, y0, 0.0), vertex(x0, y1, 0.0), vertex(x1, y1, 0.0))
    for i in range(resolution):
        for j in range(resolution):
            x0, x1 = -half_x + i * cell_x, -half_x + (i + 1) * cell_x
            y0, y1 = -half_y + i * cell_y, -half_y + (i + 1) * cell_y
            z0, z1 = j * cell_z, (j + 1) * cell_z
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
def _joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    inv_dt: float,
    out: wp.array[float],
):
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
        for i in range(qd_end - qd_begin):
            if q_begin + i < q_end:
                out[qd_begin + i] = (q1[q_begin + i] - q0[q_begin + i]) * inv_dt


@wp.kernel
def _pin_bag_particles(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    i = wp.tid()
    particle = pinned_indices[i]
    pos_0[particle] = original_positions[i]
    pos_1[particle] = original_positions[i]


class Example:
    """Interactive right-hand physical recorder for a soft-cube grasp."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.include_bag = getattr(self, "include_bag", False)
        self.particle_self_contact_enabled = getattr(self, "particle_self_contact_enabled", True)
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
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
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": self.particle_self_contact_enabled,
                "particle_self_contact_radius": CUBE_SELF_CONTACT_RADIUS,
                "particle_self_contact_margin": CUBE_SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )
        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(HAND_HOME)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(RECORDED_JOINT_DEGREES)
        self._restore_initial_controls()
        self.joint_limits = self._joint_limits()
        self.target_transform = self._copy_transform(HAND_HOME)
        self._refresh_target()
        self._set_initial_hand_pose()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)

    def _build_scene(self):
        if not RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CONTACT_MU
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(RIGHT_HAND_URDF),
            xform=HAND_HOME,
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

        table_cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POS, TABLE_ROTATION),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="hand_tuning_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="hand_tuning_ground")

        if self.include_bag:
            bag_vertices, bag_indices = _generate_box_bag(
                0.5 * BAG_WIDTH,
                0.5 * BAG_DEPTH,
                BAG_HEIGHT,
                BAG_RESOLUTION,
            )
            self.bag_particle_start = builder.particle_count
            builder.add_cloth_mesh(
                pos=BAG_POS,
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(),
                vertices=bag_vertices.tolist(),
                indices=bag_indices,
                density=BAG_DENSITY,
                tri_ke=BAG_TRI_KE,
                tri_ka=BAG_TRI_KA,
                tri_kd=BAG_TRI_KD,
                edge_ke=BAG_EDGE_KE,
                edge_kd=BAG_EDGE_KD,
                particle_radius=BAG_PARTICLE_RADIUS,
                label="open_soft_box_bag",
            )
            self.bag_particle_end = builder.particle_count
            bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - BAG_HEIGHT) < 1.0e-5)
            self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start

        cube_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        cube_origin = CUBE_CENTRE - wp.quat_rotate(cube_rotation, wp.vec3(*CUBE_HALF_EXTENTS))
        self.cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=cube_origin,
            rot=cube_rotation,
            vel=wp.vec3(),
            dim_x=CUBE_DIMS[0],
            dim_y=CUBE_DIMS[1],
            dim_z=CUBE_DIMS[2],
            cell_x=2.0 * CUBE_HALF_EXTENTS[0] / CUBE_DIMS[0],
            cell_y=2.0 * CUBE_HALF_EXTENTS[1] / CUBE_DIMS[1],
            cell_z=2.0 * CUBE_HALF_EXTENTS[2] / CUBE_DIMS[2],
            density=CUBE_DENSITY,
            k_mu=CUBE_K_MU,
            k_lambda=CUBE_K_LAMBDA,
            k_damp=CUBE_K_DAMP,
            particle_radius=CUBE_PARTICLE_RADIUS,
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
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KD
        self.model.soft_contact_mu = CONTACT_MU

    def _initialize_bag_pin(self):
        """Pin the open bag rim at its recorded world positions."""

        pinned_indices = self.bag_top_indices
        flags = self.model.particle_flags.numpy()
        flags[pinned_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        particle_q = self.state_0.particle_q.numpy()
        self.bag_pinned_indices = wp.array(pinned_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_original = wp.array(
            particle_q[pinned_indices].copy(), dtype=wp.vec3, device=self.device
        )

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
        for name in HAND_JOINTS:
            joint = next(index for index, label in enumerate(labels) if label.endswith("/" + name))
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
            for name in HAND_JOINTS
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
            if isinstance(position, list) and len(position) == 3 and isinstance(rotation, list) and len(rotation) == 4:
                self.gizmo_transform = wp.transform(wp.vec3(*position), wp.quat(*rotation))
        position_offset = keyframe.get("position_offset_mm")
        if isinstance(position_offset, list) and len(position_offset) == 3:
            self.position_mm = np.asarray(position_offset, dtype=np.float32)
        rotation_offset = keyframe.get("rotation_offset_deg")
        if isinstance(rotation_offset, list) and len(rotation_offset) == 3:
            self.rotation_deg = np.asarray(rotation_offset, dtype=np.float32)
        joints = keyframe.get("target_finger_joints_degrees")
        if isinstance(joints, dict):
            for name in HAND_JOINTS:
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
        position = base_position + self.position_mm * 1.0e-3
        rx, ry, rz = np.radians(self.rotation_deg)
        rotation = self._quat_mul(
            self._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        return wp.transform(wp.vec3(*position), self._quat_mul(rotation, wp.transform_get_rotation(self.gizmo_transform)))

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
            + [self.joint_degrees[name] for name in HAND_JOINTS]
        )

    def step_once(self):
        """Advance one real-time physical frame toward the current hand target."""

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(SIM_SUBSTEPS):
            if self.include_bag:
                wp.launch(
                    _pin_bag_particles,
                    self.bag_pinned_indices.shape[0],
                    [self.bag_pinned_indices, self.bag_pinned_original, self.state_0.particle_q, self.state_1.particle_q],
                    device=self.device,
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / SIM_SUBSTEPS
            wp.launch(_interpolate_q, self.model.joint_coord_count, [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q], device=self.device)
            wp.launch(
                _joint_velocity,
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
            self.state_0, self.state_1 = self.state_1, self.state_0
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
        self.gizmo_transform = self._copy_transform(HAND_HOME)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(RECORDED_JOINT_DEGREES)
        self._restore_initial_controls()
        self.sim_time = 0.0
        self.frame_index = 0
        self._trajectory_frames.clear()
        self._refresh_target()
        self._set_initial_hand_pose()

    def _transform_dict(self, transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return {"position_m": [float(value) for value in position], "quaternion_xyzw": [float(value) for value in rotation]}

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
            "finger_joints_degrees": {name: float(np.degrees(current_q[index])) for name, index in self.hand_joint_indices.items()},
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
        path = self._write_json(self.args.pose_output, {"format": "newton_w1_right_hand_pose_v1", "pose": self._capture_frame()})
        self._set_status(f"Saved pose: {path}")

    def save_trajectory(self):
        path = self._write_json(
            self.args.trajectory_output,
            {"format": "newton_w1_right_hand_trajectory_v1", "frame_dt_s": self.frame_dt, "frames": self._trajectory_frames},
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
        import tkinter as tk  # noqa: PLC0415

        self._ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        value_label = self._ttk.Label(parent, textvariable=variable._display_var, width=8, anchor="e")
        value_label.grid(row=row, column=1, sticky="e", padx=3)
        scale = tk.Scale(parent, variable=variable, from_=minimum, to=maximum, resolution=1.0, orient="horizontal", showvalue=False, length=440, highlightthickness=0, command=command)
        scale.grid(row=row, column=2, sticky="ew", padx=4)

        def update_display(*_):
            variable._display_var.set(f"{float(variable.get()):.1f}")

        variable.trace_add("write", update_display)
        update_display()

    def _build_controls(self, root):
        import tkinter as tk  # noqa: PLC0415

        frame = self._ttk.Frame(root, padding=8)
        frame.pack(fill="x", padx=8, pady=(8, 0))
        frame.columnconfigure(2, weight=1)
        variables = {"joints": {}, "position": [], "rotation": []}
        joints = self._ttk.LabelFrame(frame, text="RIGHT finger joint angles (degrees)", padding=5)
        joints.grid(row=0, column=0, columnspan=3, sticky="nsew")
        joints.columnconfigure(2, weight=1)
        for row, name in enumerate(HAND_JOINTS):
            variable = tk.DoubleVar(value=self.joint_degrees[name])
            variable._display_var = tk.StringVar()
            variables["joints"][name] = variable
            lower, upper = self.joint_limits[name]
            self._make_scale(joints, row, name, variable, lower, upper, lambda _value, variables=variables: self._on_control_changed(variables))
        root_box = self._ttk.LabelFrame(frame, text="Whole-hand target offset / rotation relative to gizmo", padding=5)
        root_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root_box.columnconfigure(2, weight=1)
        for index, label in enumerate(("Position X (mm)", "Position Y (mm)", "Position Z (mm)")):
            variable = tk.DoubleVar(value=float(self.position_mm[index]))
            variable._display_var = tk.StringVar()
            variables["position"].append(variable)
            self._make_scale(root_box, index, label, variable, -POSITION_LIMIT_MM, POSITION_LIMIT_MM, lambda _value, variables=variables: self._on_control_changed(variables))
        for index, label in enumerate(("Rotation X (deg)", "Rotation Y (deg)", "Rotation Z (deg)"), start=3):
            variable = tk.DoubleVar(value=float(self.rotation_deg[index - 3]))
            variable._display_var = tk.StringVar()
            variables["rotation"].append(variable)
            self._make_scale(root_box, index, label, variable, -180.0, 180.0, lambda _value, variables=variables: self._on_control_changed(variables))
        return variables

    def run_recorder(self):
        if self.args.recorder_no_gui:
            self.render()
            self.viewer.close()
            return
        import tkinter as tk  # noqa: PLC0415
        from tkinter import ttk  # noqa: PLC0415

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
        self._status_var = tk.StringVar(value="Realtime physics running; move the gizmo or sliders, then record a keyframe.")
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))

        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.step_once()
            self.render()
            root.after(max(1, int(1000.0 / FPS)), pump_viewer)

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
            {"format": "newton_w1_right_hand_trajectory_v1", "frame_dt_s": self.frame_dt, "frames": self._trajectory_frames},
        )
        contact_count = int(self.solver.contacts.soft_contact_count.numpy()[0])
        self._set_status(
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; "
            f"solved hand/soft contacts: {contact_count}. Saved: {keyframe_path}, {trajectory_path}"
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
        parser.add_argument("--pose-output", default="vbd_w1_right_hand_pose.json")
        parser.add_argument("--trajectory-output", default="vbd_w1_right_hand_trajectory.json")
        parser.add_argument("--keyframe-output", default="vbd_w1_right_hand_last_keyframe.json")
        parser.add_argument("--recorder-no-gui", action="store_true")
        return parser


def main():
    """Launch the interactive right-hand soft-cube recorder."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


if __name__ == "__main__":
    main()
