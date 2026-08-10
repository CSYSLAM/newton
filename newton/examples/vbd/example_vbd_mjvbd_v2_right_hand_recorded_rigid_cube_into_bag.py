# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Place a rigid cube into a soft bag with a recorded right-hand grasp.

The floating W1 right hand closes from the recorder's approach pose to a
recorded rigid-cube keyframe, then lifts, transports, and physically releases
the dynamic cube above an open soft bag. No object pose is attached to or
copied from the hand.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_recorded_rigid_cube_into_bag --viewer gl
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_right_hand_rigid_cube_recorder as recorder
from newton.solvers import SolverMJVBDV2

DEFAULT_GRASP_KEYFRAME = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"

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

RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
PARTICLE_VERTEX_CONTACT_BUFFER_SIZE = 128
PARTICLE_EDGE_CONTACT_BUFFER_SIZE = 256
SOFT_CONTACT_MARGIN = 0.01
SOFT_CONTACT_KE = 5.0e3
SOFT_CONTACT_KD = 5.0e-2
SOFT_CONTACT_MU = 0.25
RELEASE_CONTACT_KE = 5.0e3
RELEASE_CONTACT_KD = 0.0
RELEASE_FRICTION = 0.0

GRASP_CLOSE_DURATION = 0.45
GRASP_SETTLE_DURATION = 0.30
LIFT_DURATION = 0.75

OPEN_JOINTS = dict.fromkeys(recorder.HAND_JOINTS, 0.0)
START_JOINTS = dict(OPEN_JOINTS)
START_JOINTS["RIGHT_HAND_THUMB2"] = 90.0


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


class Example(recorder.Example):
    """Run the recorded mesh-only rigid-cube grasp and bag placement."""

    def __init__(self, viewer, args):
        self.grasp_root, self.grasp_joints, self.recorded_cube_position = self._load_grasp_keyframe(args.grasp_keyframe)
        super().__init__(viewer, args)
        self._initialize_bag_pin()
        self._create_solver()
        self.release_contact_material_applied = False
        self._set_hand_target(self.grasp_root, START_JOINTS)
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
        if not isinstance(root, dict) or not isinstance(joints, dict) or not isinstance(cube, dict):
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
        missing_joints = set(recorder.HAND_JOINTS) - joints.keys()
        if missing_joints:
            raise ValueError(f"Missing hand joints in recorded grasp {path}: {sorted(missing_joints)}")
        return (
            wp.transform(wp.vec3(*position), wp.quat(*rotation)),
            {name: float(joints[name]) for name in recorder.HAND_JOINTS},
            wp.vec3(*cube_position),
        )

    def _build_scene(self):
        """Build the recorder's rigid scene plus the pinned soft bag."""

        if not recorder.RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {recorder.RIGHT_HAND_URDF}")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = recorder.CONTACT_KE
        builder.default_shape_cfg.kd = recorder.CONTACT_KD
        builder.default_shape_cfg.mu = recorder.CONTACT_MU
        builder.default_shape_cfg.margin = recorder.CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

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
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(recorder.TABLE_POS, recorder.TABLE_ROTATION),
            hx=recorder.TABLE_HALF_EXTENTS[0],
            hy=recorder.TABLE_HALF_EXTENTS[1],
            hz=recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="recorded_rigid_cube_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="recorded_rigid_cube_ground")

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
            label="recorded_rigid_cube_soft_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=recorder.CUBE_DENSITY,
            ke=recorder.CONTACT_KE,
            kd=recorder.CONTACT_KD,
            mu=recorder.CONTACT_MU,
            margin=recorder.CONTACT_MARGIN,
        )
        cube_cfg.configure_sdf(force_sdf=True)
        self.cube_body = builder.add_body(
            xform=wp.transform(recorder.CUBE_CENTRE, wp.quat_identity()),
            label="recorded_rigid_cube",
        )
        self.cube_shape = builder.shape_count
        builder.add_shape_box(
            self.cube_body,
            hx=recorder.CUBE_HALF_EXTENTS[0],
            hy=recorder.CUBE_HALF_EXTENTS[1],
            hz=recorder.CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            color=(0.90, 0.32, 0.18),
            label="recorded_rigid_cube_shape",
        )

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_MU

    def _initialize_bag_pin(self):
        """Pin the open bag rim at its initial world positions."""

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

    def _create_solver(self):
        """Create VBD with the recorder's rigid settings and bag contacts."""

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": recorder.VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 5.0e-4,
                "rigid_contact_stick_freeze_translation_eps": 2.0e-4,
                "rigid_contact_stick_freeze_angular_eps": 2.0e-4,
                "rigid_body_contact_buffer_size": recorder.RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": 2.0 * BAG_PARTICLE_RADIUS,
                "particle_vertex_contact_buffer_size": PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
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
        lift = wp.transform(grasp_position + wp.vec3(0.0, 0.0, 0.10), grasp_rotation)
        release_cube_position = wp.vec3(
            float(BAG_POS[0]),
            float(BAG_POS[1]),
            float(BAG_POS[2]) + BAG_HEIGHT + 0.06,
        )
        bag_hover = wp.transform(release_cube_position + root_cube_offset, grasp_rotation)
        transport = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.05),
            grasp_rotation,
        )
        retreat = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.12),
            grasp_rotation,
        )
        approach_joints = recorder.INITIAL_HAND_JOINTS
        segments = (
            (0.50, self.grasp_root, self.grasp_root, START_JOINTS, START_JOINTS),
            (1.50, self.grasp_root, self.grasp_root, START_JOINTS, approach_joints),
            (0.50, self.grasp_root, self.grasp_root, approach_joints, approach_joints),
            (GRASP_CLOSE_DURATION, self.grasp_root, self.grasp_root, approach_joints, self.grasp_joints),
            (GRASP_SETTLE_DURATION, self.grasp_root, self.grasp_root, self.grasp_joints, self.grasp_joints),
            (LIFT_DURATION, self.grasp_root, lift, self.grasp_joints, self.grasp_joints),
            (5.00, lift, transport, self.grasp_joints, self.grasp_joints),
            (1.20, transport, bag_hover, self.grasp_joints, self.grasp_joints),
            (0.50, bag_hover, bag_hover, self.grasp_joints, self.grasp_joints),
            (0.80, bag_hover, bag_hover, self.grasp_joints, OPEN_JOINTS),
            (1.50, bag_hover, bag_hover, OPEN_JOINTS, OPEN_JOINTS),
            (1.00, bag_hover, retreat, OPEN_JOINTS, OPEN_JOINTS),
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
        shape_ke[: self.hand_shape_end] = RELEASE_CONTACT_KE
        shape_kd[: self.hand_shape_end] = RELEASE_CONTACT_KD
        shape_mu[: self.hand_shape_end] = RELEASE_FRICTION
        shape_ke[self.cube_shape] = RELEASE_CONTACT_KE
        shape_kd[self.cube_shape] = RELEASE_CONTACT_KD
        shape_mu[self.cube_shape] = RELEASE_FRICTION
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self.model.soft_contact_ke = RELEASE_CONTACT_KE
        self.model.soft_contact_kd = RELEASE_CONTACT_KD
        self.model.soft_contact_mu = RELEASE_FRICTION
        self.release_contact_material_applied = True

    def _restore_grasp_contact_material(self):
        """Restore the recorder's hand-cube material after a reset."""

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[: self.hand_shape_end] = recorder.CONTACT_KE
        shape_kd[: self.hand_shape_end] = recorder.CONTACT_KD
        shape_mu[: self.hand_shape_end] = recorder.CONTACT_MU
        shape_ke[self.cube_shape] = recorder.CONTACT_KE
        shape_kd[self.cube_shape] = recorder.CONTACT_KD
        shape_mu[self.cube_shape] = recorder.CONTACT_MU
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_material_mu.assign(shape_mu)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_MU
        self.release_contact_material_applied = False

    def _sample(self, time_s: float):
        """Interpolate the recorded root and joint targets at a script time."""

        for duration, root_a, root_b, joints_a, joints_b in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {
                    name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in recorder.HAND_JOINTS
                }
                return root, joints
            time_s -= duration
        _, _, root, _, joints = self.segments[-1]
        return root, joints

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
        rotation /= max(np.linalg.norm(rotation), 1.0e-8)
        return wp.transform(
            wp.vec3(*(position_a * (1.0 - alpha) + position_b * alpha)),
            wp.quat(*rotation),
        )

    def step_once(self):
        """Advance one frame while keeping the bag rim pinned."""

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(recorder.SIM_SUBSTEPS):
            wp.launch(
                _pin_bag_particles,
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
            alpha = (substep + 1) / recorder.SIM_SUBSTEPS
            wp.launch(
                _interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
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
        super()._reset_physics()
        self._initialize_bag_pin()
        self._restore_grasp_contact_material()
        self._set_hand_target(self.grasp_root, START_JOINTS)
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
        assert bag_height < 0.50, f"Bag stretched excessively: height={bag_height:.3f} m"

    @staticmethod
    def create_parser():
        """Create parser options for the recorded rigid-cube placement."""

        parser = recorder.Example.create_parser()
        parser.set_defaults(num_frames=875, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(DEFAULT_GRASP_KEYFRAME),
            help="Rigid-cube grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def main():
    """Run the right-hand recorded rigid-cube bag-placement example."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
