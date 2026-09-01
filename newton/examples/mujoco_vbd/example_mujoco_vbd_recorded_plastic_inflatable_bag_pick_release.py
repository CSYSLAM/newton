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


hand_reference = _LocalModule(
    "_m4_",
    recorder=_LocalModule("_m1_", hand_recorder=_LocalModule("_m0_")),
)
import json
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

_m0_FPS = 60
_m0_SIM_SUBSTEPS = 8
_m0_VBD_ITERATIONS = 40
_m0_RIGHT_HAND_URDF = Path(__file__).resolve().parents[3] / "assets" / "W1_right_hand" / "DexforceW1_right_hand.urdf"
_m0_HAND_HOME = wp.transform(
    wp.vec3(-0.15679353, -2.8874836, 1.3789376), wp.quat(-0.31233013, 0.67216527, 0.32775849, -0.58584785)
)
_m0_TABLE_POS = wp.vec3(-0.34931439, -2.69669516, 1.14622798)
_m0_TABLE_ROTATION = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m0_TABLE_HALF_EXTENTS = (0.32, 0.45, 0.025)
_m0_TABLE_TOP_Z = float(_m0_TABLE_POS[2]) + _m0_TABLE_HALF_EXTENTS[2]
_m0_CUBE_HALF_EXTENTS = (0.027, 0.012, 0.027)
_m0_CUBE_CENTRE = wp.vec3(-0.14931439, -2.76669516, _m0_TABLE_TOP_Z + _m0_CUBE_HALF_EXTENTS[2] + 0.001)
_m0_CUBE_DENSITY = 1500.0
_m0_CONTACT_MARGIN = 0.0015
_m0_CONTACT_KE = 3000.0
_m0_CONTACT_KD = 1.0
_m0_CONTACT_MU = 3000.0
_m0_RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
_m0_POSITION_LIMIT_MM = 500.0
_m0_CAMERA_POS = wp.vec3(2.15, -5.78, 1.94)
_m0_CAMERA_PITCH = -18.0
_m0_CAMERA_YAW = 126.0
_m0_HAND_JOINTS = (
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
_m0_INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
_m0_INITIAL_HAND_JOINTS = {
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
def _m0__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _m0__joint_velocity(
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


class _m0_Example:
    """Tune a mesh-only physical grasp of one dynamic rigid cube."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / _m0_FPS
        self.sim_dt = self.frame_dt / _m0_SIM_SUBSTEPS
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
                "iterations": _m0_VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 0.0005,
                "rigid_contact_stick_freeze_translation_eps": 0.0002,
                "rigid_contact_stick_freeze_angular_eps": 0.0002,
                "rigid_body_contact_buffer_size": _m0_RIGID_BODY_CONTACT_BUFFER_SIZE,
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
        self.gizmo_transform = self._copy_transform(_m0_INITIAL_HAND_ROOT)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(_m0_INITIAL_HAND_JOINTS)
        self._restore_initial_controls()
        self.joint_limits = self._joint_limits()
        self.target_transform = self._copy_transform(self.gizmo_transform)
        self._refresh_target()
        self._set_initial_hand_pose()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(_m0_CAMERA_POS, _m0_CAMERA_PITCH, _m0_CAMERA_YAW)

    def _build_scene(self):
        if not _m0_RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {_m0_RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = _m0_CONTACT_KE
        builder.default_shape_cfg.kd = _m0_CONTACT_KD
        builder.default_shape_cfg.mu = _m0_CONTACT_MU
        builder.default_shape_cfg.margin = _m0_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m0_RIGHT_HAND_URDF),
            xform=_m0_HAND_HOME,
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
            xform=wp.transform(_m0_TABLE_POS, _m0_TABLE_ROTATION),
            hx=_m0_TABLE_HALF_EXTENTS[0],
            hy=_m0_TABLE_HALF_EXTENTS[1],
            hz=_m0_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="hand_tuning_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="hand_tuning_ground")
        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=_m0_CUBE_DENSITY, ke=_m0_CONTACT_KE, kd=_m0_CONTACT_KD, mu=_m0_CONTACT_MU, margin=_m0_CONTACT_MARGIN
        )
        cube_cfg.configure_sdf(force_sdf=True)
        self.cube_body = builder.add_body(
            xform=wp.transform(_m0_CUBE_CENTRE, wp.quat_identity()), label="tunable_rigid_cube"
        )
        self.cube_shape = builder.shape_count
        builder.add_shape_box(
            self.cube_body,
            hx=_m0_CUBE_HALF_EXTENTS[0],
            hy=_m0_CUBE_HALF_EXTENTS[1],
            hz=_m0_CUBE_HALF_EXTENTS[2],
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
        for name in _m0_HAND_JOINTS:
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
            for name in _m0_HAND_JOINTS
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
            for name in _m0_HAND_JOINTS:
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
            + [self.joint_degrees[name] for name in _m0_HAND_JOINTS]
        )

    def step_once(self):
        """Advance one real-time physical frame toward the current hand target."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        for substep in range(_m0_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m0_SIM_SUBSTEPS
            wp.launch(
                _m0__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m0__joint_velocity,
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
        self.gizmo_transform = self._copy_transform(_m0_INITIAL_HAND_ROOT)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(_m0_INITIAL_HAND_JOINTS)
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
        for row, name in enumerate(_m0_HAND_JOINTS):
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
                -_m0_POSITION_LIMIT_MM,
                _m0_POSITION_LIMIT_MM,
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
            root.after(max(1, int(1000.0 / _m0_FPS)), pump_viewer)

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


def _m0_main():
    """Launch the interactive right-hand rigid-cube recorder."""
    parser = _m0_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = _m0_Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import PneumaticConfig, PneumaticMode, SolverMuJoCoVBD

_m1_FPS = 60
_m1_SIM_SUBSTEPS = 5
_m1_VBD_ITERATIONS = 12
_m1_MAX_FINGER_SPEED_DEG_S = 90.0
_m1_MAX_FINGER_CONTACT_SPEED_DEG_S = 30.0
_m1_CAMERA_POS = wp.vec3(0.3, -3.39, 1.63)
_m1_CAMERA_FOV = 45.0
_m1_CAMERA_PITCH = -28.3
_m1_CAMERA_YAW = 124.3
_m1_BAG_SCALE = 0.36
_m1_BAG_DENSITY = 0.12
_m1_BAG_REFERENCE_ABSOLUTE_PRESSURE = 125000.0
_m1_BAG_AMBIENT_PRESSURE = 101325.0
_m1_BAG_MAX_ABSOLUTE_PRESSURE = 200000.0
_m1_BAG_BULK_DAMPING = 50.0
_m1_BAG_TARGET_VOLUME_RATIO = 1.01
_m1_BAG_PARTICLE_RADIUS = 0.002
_m1_BAG_TRI_KE = 100000.0
_m1_BAG_TRI_KA = 100000.0
_m1_BAG_TRI_KD = 80.0
_m1_BAG_EDGE_KE = 20.0
_m1_BAG_EDGE_KD = 0.5
_m1_BAG_REST_BULGE = 0.02
_m1_BAG_SOURCE_HALF_WIDTH = 0.133928
_m1_BAG_SOURCE_HALF_LENGTH = 0.18
_m1_BAG_SOURCE_FACE_HALF_THICKNESS = 0.052
_m1_BAG_SOURCE_HALF_THICKNESS = _m1_BAG_SOURCE_FACE_HALF_THICKNESS + _m1_BAG_REST_BULGE
_m1_BAG_CENTER = wp.vec3(
    float(_m0_CUBE_CENTRE[0]),
    float(_m0_CUBE_CENTRE[1]),
    _m0_TABLE_TOP_Z + _m1_BAG_SCALE * _m1_BAG_SOURCE_HALF_THICKNESS + _m1_BAG_PARTICLE_RADIUS,
)
_m1_BAG_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5) * wp.quat_from_axis_angle(
    wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5
)
_m1_SHAPE_CONTACT_MARGIN = 0.0
_m1_SOFT_CONTACT_MARGIN = 0.0
_m1_CONTACT_KE = 200000.0
_m1_CONTACT_KD = 100.0
_m1_CONTACT_MU = 2.0
_m1_RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
_m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 8192
_m1_PNEUMATIC_MODES = {"isothermal": PneumaticMode.ISOTHERMAL, "target-volume": PneumaticMode.TARGET_VOLUME}
_m1_INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.835815668106079, 1.3647105693817139),
    wp.quat(0.03812963888049126, 0.9212844967842102, -0.3854166567325592, 0.03514265641570091),
)
_m1_INITIAL_HAND_JOINTS = {
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


@dataclass(frozen=True)
class _m1__ChipBagMesh:
    """Store simulation triangles and rendering edges for the authored bag."""

    vertices: list[list[float]]
    indices: list[int]
    edges: list[tuple[int, int]]


def _m1__scaled_mesh_volume(mesh: _m1__ChipBagMesh, scale: float) -> float:
    """Compute the closed mesh volume after uniform scaling [m^3]."""
    positions = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    anchor = positions[triangles[0, 0]]
    relative = positions[triangles] - anchor
    signed_volume = np.einsum("ij,ij->i", relative[:, 0], np.cross(relative[:, 1], relative[:, 2])).sum() / 6.0
    volume = abs(float(signed_volume)) * scale**3
    if volume <= 0.0:
        raise ValueError("The inflatable-bag mesh must enclose a positive volume.")
    return volume


def _m1__make_pneumatic_config(mode_name: str, rest_volume: float) -> PneumaticConfig:
    """Create a pressure law with matching initial gauge pressure."""
    mode = _m1_PNEUMATIC_MODES[mode_name]
    target_volume = None
    volume_stiffness = 0.0
    if mode == PneumaticMode.TARGET_VOLUME:
        target_volume = _m1_BAG_TARGET_VOLUME_RATIO * rest_volume
        target_volume_delta = target_volume - rest_volume
        initial_gauge_pressure = _m1_BAG_REFERENCE_ABSOLUTE_PRESSURE - _m1_BAG_AMBIENT_PRESSURE
        volume_stiffness = initial_gauge_pressure / target_volume_delta
    return PneumaticConfig(
        mode=mode,
        reference_absolute_pressure=_m1_BAG_REFERENCE_ABSOLUTE_PRESSURE,
        ambient_pressure=_m1_BAG_AMBIENT_PRESSURE,
        target_volume=target_volume,
        volume_stiffness=volume_stiffness,
        bulk_damping=_m1_BAG_BULK_DAMPING,
        max_absolute_pressure=_m1_BAG_MAX_ABSOLUTE_PRESSURE,
    )


def _m1__load_chip_bag_mesh() -> _m1__ChipBagMesh:
    """Load, pre-bulge, and validate the closed Blender-authored bag mesh."""
    path = newton.examples.get_asset("newton_chip_bag_sealed_cylinder.obj")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as obj_file:
        for line in obj_file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                faces.append([int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]])
    if not vertices or not faces:
        raise ValueError(f"{path} does not contain a mesh.")
    triangles = [
        (face[0], face[vertex_index], face[vertex_index + 1])
        for face in faces
        for vertex_index in range(1, len(face) - 1)
    ]
    if not triangles:
        raise ValueError(f"{path} does not contain any triangle faces.")
    edge_counts = Counter(
        (
            tuple(sorted((vertex0, vertex1)))
            for triangle in triangles
            for vertex0, vertex1 in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
        )
    )
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(f"{path} must be a closed two-manifold surface after triangulation.")
    for vertex in vertices:
        x, y, z = vertex
        if abs(y) < 1e-08:
            continue
        width_phase = min(1.0, abs(x) / _m1_BAG_SOURCE_HALF_WIDTH)
        length_phase = min(1.0, abs(z) / _m1_BAG_SOURCE_HALF_LENGTH)
        width_profile = float(np.cos(0.5 * np.pi * width_phase) ** 2)
        length_profile = float(np.cos(0.5 * np.pi * length_phase) ** 2)
        face_weight = min(1.0, abs(y) / _m1_BAG_SOURCE_FACE_HALF_THICKNESS)
        vertex[1] += np.sign(y) * _m1_BAG_REST_BULGE * width_profile * length_profile * face_weight
    return _m1__ChipBagMesh(
        vertices=vertices, indices=[vertex for triangle in triangles for vertex in triangle], edges=sorted(edge_counts)
    )


@wp.kernel
def _m1__gather_edges(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array[int],
    lift: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    edge = wp.tid()
    offset = wp.vec3(0.0, 0.0, lift)
    starts[edge] = positions[edge_indices[2 * edge]] + offset
    ends[edge] = positions[edge_indices[2 * edge + 1]] + offset


@wp.kernel
def _m1__limit_finger_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    soft_contact_count: wp.array[int],
    soft_contact_shape: wp.array[int],
    hand_shape_end: int,
    free_max_step: float,
    contact_max_step: float,
    target_q: wp.array[float],
):
    finger = wp.tid()
    active_contact_count = soft_contact_count[0]
    if active_contact_count > soft_contact_shape.shape[0]:
        active_contact_count = soft_contact_shape.shape[0]
    hand_contact = bool(False)
    for contact in range(active_contact_count):
        shape = soft_contact_shape[contact]
        if shape >= 0 and shape < hand_shape_end:
            hand_contact = True
    max_step = free_max_step
    if hand_contact:
        max_step = contact_max_step
    q_index = finger_q_indices[finger]
    delta = wp.clamp(target_q[q_index] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


class _m1_Example(_m0_Example):
    """Record physical right-hand keyframes for an inflatable-bag grasp."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / _m1_FPS
        self.sim_dt = self.frame_dt / _m1_SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self._root = None
        self._status_var = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._last_target_signature: tuple[float, ...] | None = None
        self._initial_keyframe = None
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
                "iterations": _m1_VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": _m1_RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": _m1_SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
            coupling_mode="one_way",
        )
        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.hand_joint_q_indices = wp.array(
            tuple(self.hand_joint_indices.values()), dtype=int, device=self.model.device
        )
        self.max_finger_step = float(np.radians(_m1_MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self.max_finger_contact_step = float(np.radians(_m1_MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt)
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m1_INITIAL_HAND_ROOT)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(_m1_INITIAL_HAND_JOINTS)
        self._restore_initial_controls()
        self.joint_limits = self._joint_limits()
        self.target_transform = self._copy_transform(self.gizmo_transform)
        self._refresh_target()
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(_m1_CAMERA_POS, _m1_CAMERA_PITCH, _m1_CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = _m1_CAMERA_FOV
        self._store_trajectory_frame()

    def _build_scene(self):
        """Build the kinematic hand, support table, and sealed bag."""
        if not _m0_RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {_m0_RIGHT_HAND_URDF}")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = _m1_CONTACT_KE
        builder.default_shape_cfg.kd = _m1_CONTACT_KD
        builder.default_shape_cfg.mu = _m1_CONTACT_MU
        builder.default_shape_cfg.margin = _m1_SHAPE_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(_m0_RIGHT_HAND_URDF),
            xform=_m0_HAND_HOME,
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
            ke=_m1_CONTACT_KE, kd=_m1_CONTACT_KD, mu=0.9, margin=_m1_SHAPE_CONTACT_MARGIN, is_visible=True
        )
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(_m0_TABLE_POS, _m0_TABLE_ROTATION),
            hx=_m0_TABLE_HALF_EXTENTS[0],
            hy=_m0_TABLE_HALF_EXTENTS[1],
            hz=_m0_TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="inflatable_bag_recorder_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="inflatable_bag_recorder_ground")
        bag_mesh = _m1__load_chip_bag_mesh()
        bag_rest_volume = _m1__scaled_mesh_volume(bag_mesh, _m1_BAG_SCALE)
        self.pneumatic_mode_name = self.args.pneumatic_mode
        self.pneumatic_config = _m1__make_pneumatic_config(self.pneumatic_mode_name, bag_rest_volume)
        self.bag_particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=_m1_BAG_CENTER,
            rot=_m1_BAG_ROTATION,
            scale=_m1_BAG_SCALE,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=_m1_BAG_DENSITY,
            tri_ke=_m1_BAG_TRI_KE,
            tri_ka=_m1_BAG_TRI_KA,
            tri_kd=_m1_BAG_TRI_KD,
            edge_ke=_m1_BAG_EDGE_KE,
            edge_kd=_m1_BAG_EDGE_KD,
            particle_radius=_m1_BAG_PARTICLE_RADIUS,
            validate_mesh=True,
            label="graspable_sealed_chip_bag",
            config=self.pneumatic_config,
        )
        self.bag_particle_end = builder.particle_count
        bag_triangle_indices = np.asarray(bag_mesh.indices, dtype=np.int32) + self.bag_particle_start
        bag_edges = np.asarray(bag_mesh.edges, dtype=np.int32) + self.bag_particle_start
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = _m1_CONTACT_KE
        self.model.soft_contact_kd = _m1_CONTACT_KD
        self.model.soft_contact_mu = _m1_CONTACT_MU
        self.bag_triangle_indices = wp.array(bag_triangle_indices, dtype=int, device=self.model.device)
        self.bag_edges = wp.array(bag_edges.reshape(-1), dtype=int, device=self.model.device)
        self.bag_edge_starts = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_edge_ends = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)

    def _prepare_physics_frame(self):
        """Prepare graph-compatible root and finger targets for one frame."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        contacts = self.solver.contacts
        wp.launch(
            _m1__limit_finger_target_step,
            dim=self.hand_joint_q_indices.shape[0],
            inputs=[
                self.frame_q_start,
                self.hand_joint_q_indices,
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                self.hand_shape_end,
                self.max_finger_step,
                self.max_finger_contact_step,
                self.frame_q_end,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance the fixed substep sequence for one display frame."""
        for substep in range(_m1_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m1_SIM_SUBSTEPS
            wp.launch(
                _m0__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m0__joint_velocity,
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
            if _m1_SIM_SUBSTEPS % 2 != 0 and substep == _m1_SIM_SUBSTEPS - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = (self.state_1, self.state_0)

    def _advance_physics_frame(self):
        """Prepare targets and advance one graph-capturable physics frame."""
        self._prepare_physics_frame()
        self._simulate_substeps()

    def step_once(self):
        """Advance one frame while limiting finger motion through soft contact."""
        self._advance_physics_frame()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _reset_physics(self):
        """Reset the hand, bag, pressure state, and recorded keyframes."""
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(_m1_INITIAL_HAND_ROOT)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(_m1_INITIAL_HAND_JOINTS)
        self._restore_initial_controls()
        self.sim_time = 0.0
        self.frame_index = 0
        self._trajectory_frames.clear()
        self._refresh_target()
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        self._store_trajectory_frame()

    def _contact_counts(self) -> tuple[int, int]:
        """Return hand-bag and total rigid-soft contact counts."""
        contacts = self.solver.contacts
        total = int(contacts.soft_contact_count.numpy()[0])
        shape_indices = contacts.soft_contact_shape.numpy()
        active = min(total, shape_indices.shape[0])
        hand_bag = np.count_nonzero((shape_indices[:active] >= 0) & (shape_indices[:active] < self.hand_shape_end))
        return (int(hand_bag), total)

    def _capture_frame(self):
        """Capture the current hand target and pneumatic-bag summary."""
        current_q = self.state_0.joint_q.numpy()
        root_q = current_q[self.root_q_start : self.root_q_start + 7]
        bag_positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        bag_center = np.mean(bag_positions, axis=0)
        bag_min = np.min(bag_positions, axis=0)
        bag_max = np.max(bag_positions, axis=0)
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        absolute_pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        volume_rate = float(self.state_0.pneumatic.volume_rate.numpy()[cavity_index])
        clamp_flags = int(self.state_0.pneumatic.clamp_flags.numpy()[cavity_index])
        hand_bag_contacts, total_soft_contacts = self._contact_counts()
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
            "inflatable_bag": {
                "pneumatic_mode": self.pneumatic_mode_name,
                "particle_count": int(self.bag_particle_end - self.bag_particle_start),
                "center_m": [float(value) for value in bag_center],
                "aabb_min_m": [float(value) for value in bag_min],
                "aabb_max_m": [float(value) for value in bag_max],
                "volume_m3": volume,
                "rest_volume_m3": float(self.cavity.rest_volume),
                "volume_ratio": volume / self.cavity.rest_volume,
                "target_volume_m3": self.pneumatic_config.target_volume,
                "volume_stiffness_pa_per_m3": self.pneumatic_config.volume_stiffness,
                "absolute_pressure_pa": absolute_pressure,
                "gauge_pressure_pa": absolute_pressure - _m1_BAG_AMBIENT_PRESSURE,
                "volume_rate_m3_s": volume_rate,
                "clamp_flags": clamp_flags,
            },
            "hand_bag_contact_count": hand_bag_contacts,
            "total_soft_contact_count": total_soft_contacts,
        }

    def save_pose(self):
        """Save the current physical grasp pose."""
        path = self._write_json(
            self.args.pose_output,
            {"format": "newton_w1_right_hand_inflatable_bag_pose_v1", "pose": self._capture_frame()},
        )
        self._set_status(f"Saved pose: {path}")

    def save_trajectory(self):
        """Save all explicitly recorded grasp keyframes."""
        path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_inflatable_bag_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        self._set_status(f"Saved trajectory: {path}")

    def render(self):
        """Render the physical hand, pneumatic surface, and mesh edges."""
        if self._target_signature() != self._last_target_signature:
            self._refresh_target()
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo("right_hand_target", self.gizmo_transform)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/inflatable_bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        wp.launch(
            _m1__gather_edges,
            dim=len(self.bag_edge_starts),
            inputs=[self.state_0.particle_q, self.bag_edges, 0.0001],
            outputs=[self.bag_edge_starts, self.bag_edge_ends],
            device=self.model.device,
        )
        self.viewer.log_lines("/inflatable_bag/grid", self.bag_edge_starts, self.bag_edge_ends, (0.08, 0.06, 0.02))
        self.viewer.end_frame()

    def run_recorder(self):
        """Run the interactive Tk controls alongside the viewer."""
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
        root.title(f"MJVBD-v2 W1 right-hand inflatable-bag recorder ({self.pneumatic_mode_name})")
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
            value=f"Pneumatic mode: {self.pneumatic_mode_name}. Finger targets move gradually; wait for the physical hand to settle before recording."
        )
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.step_once()
            self.render()
            root.after(max(1, int(1000.0 / _m1_FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _record_keyframe_from_ui(self):
        """Record and persist one physical grasp keyframe."""
        self._store_trajectory_frame()
        keyframe_path = self._write_json(
            self.args.keyframe_output,
            {"format": "newton_w1_right_hand_inflatable_bag_keyframe_v1", "keyframe": self._trajectory_frames[-1]},
        )
        trajectory_path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_inflatable_bag_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        hand_bag_contacts, total_soft_contacts = self._contact_counts()
        self._set_status(
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; hand-bag contacts: {hand_bag_contacts}, total soft contacts: {total_soft_contacts}. Saved: {keyframe_path}, {trajectory_path}"
        )

    def _reset_from_ui(self):
        """Reset the interactive physical scene."""
        self._reset_physics()
        self._set_status("Reset the hand and inflatable bag to their initial states.")
        self.render()

    def test_final(self):
        """Verify the first keyframe and one physical step remain finite."""
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        volume = float(self.state_0.pneumatic.volume.numpy()[self.cavity.cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[self.cavity.cavity_index])
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(positions))
        assert np.isfinite(volume) and volume > self.cavity.rest_volume * 0.2
        assert np.isfinite(pressure) and pressure > 0.0
        assert self._trajectory_frames[0]["frame"] == 0
        assert self._trajectory_frames[0]["target_root_pose"] == self._transform_dict(_m1_INITIAL_HAND_ROOT)

    @staticmethod
    def create_parser():
        """Create command-line arguments for the bag-grasp recorder."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        output_dir = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2"
        parser.add_argument("--pose-output", default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_pose.json"))
        parser.add_argument(
            "--trajectory-output", default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_trajectory.json")
        )
        parser.add_argument(
            "--keyframe-output", default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_last_keyframe.json")
        )
        parser.add_argument(
            "--pneumatic-mode",
            choices=tuple(_m1_PNEUMATIC_MODES),
            default="target-volume",
            help="Pressure law for the sealed bag (default: %(default)s).",
        )
        parser.add_argument("--recorder-no-gui", action="store_true")
        return parser


def _m1_main():
    """Launch the interactive right-hand inflatable-bag recorder."""
    parser = _m1_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = _m1_Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

_m2_DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "vbd_mjvbd_v2"
    / "vbd_w1_right_hand_inflatable_bag_last_keyframe.json"
)
_m2_HAND_JOINTS = tuple(_m1_INITIAL_HAND_JOINTS)
_m2_OPEN_JOINTS = dict(_m1_INITIAL_HAND_JOINTS)
_m2_HAND_TRANSLATION_SPEED = 0.04
_m2_HAND_ANGULAR_SPEED_DEG_S = 90.0
_m2_INITIAL_POSE_HOLD_DURATION = 0.1
_m2_GRASP_SETTLE_DURATION = 0.75
_m2_SCRIPTED_LIFT_HEIGHT = 0.12
_m2_LIFTED_HOLD_DURATION = 0.1
_m2_DROP_SETTLE_DURATION = 2.0
_m2_FINGER_PHASE_PADDING = 0.25
_m2_MINIMUM_ROOT_PHASE_DURATION = 0.25
_m2_DEMO_MAX_ABSOLUTE_PRESSURE = 500000.0
_m2_RELEASE_FRICTION = 1.0
_m2__SOFT_MATERIAL_GRASP = 0
_m2__SOFT_MATERIAL_RELEASE = 1


@dataclass(frozen=True)
class _m2__RecordedGrasp:
    """Store a validated pre-lift hand target and scripted lift height."""

    root: wp.transform
    joints_degrees: dict[str, float]
    lift_height: float


@dataclass(frozen=True)
class _m2__Phase:
    """Describe one autonomous hand-motion phase."""

    name: str
    duration: float
    root_start: wp.transform
    root_end: wp.transform
    finger_target_degrees: dict[str, float]
    release: bool = False


class _m2_Example(_m1_Example):
    """Replay a recorded physical grasp, lift, and release sequence."""

    def __init__(self, viewer, args):
        self.recorded_grasp = self._load_recorded_grasp(args.grasp_keyframe)
        super().__init__(viewer, args)
        self._validate_initial_pose()
        self._raise_pressure_limit()
        self.graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.script_target_q = self.manual_target_q.numpy()
        self.soft_contact_materials = None
        self.soft_contact_material_index = None
        if self.use_graph:
            self.soft_contact_materials = wp.array(
                np.asarray(
                    (
                        (_m1_CONTACT_KE, _m1_CONTACT_KD, _m1_CONTACT_MU),
                        (_m1_CONTACT_KE, _m1_CONTACT_KD, _m2_RELEASE_FRICTION),
                    ),
                    dtype=np.float32,
                ),
                dtype=wp.vec3,
                device=self.device,
            )
            self.soft_contact_material_index = wp.full(1, _m2__SOFT_MATERIAL_GRASP, dtype=wp.int32, device=self.device)
            self.solver.vbd_solver.set_soft_contact_material_source(
                self.soft_contact_materials, self.soft_contact_material_index
            )
        self.release_material_applied = False
        self.initial_bag_center_z = self._bag_center_z()
        self.lifted_bag_center_z: float | None = None
        self.minimum_volume_ratio = 1.0
        self.maximum_pressure = _m1_BAG_REFERENCE_ABSOLUTE_PRESSURE
        self.phases = self._build_phases()
        self.script_duration = sum(phase.duration for phase in self.phases)
        self.active_phase_index = -1
        self.active_phase_name = "initialization"

    @staticmethod
    def _validated_vector(value, length: int, label: str, path: Path) -> np.ndarray:
        """Return a finite vector with the requested length."""
        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"Invalid {label} in recorded grasp: {path}")
        vector = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite {label} in recorded grasp: {path}")
        return vector

    @classmethod
    def _load_recorded_grasp(cls, path_value: str) -> _m2__RecordedGrasp:
        """Load the grasp while ignoring its recorded post-grasp Z offset."""
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Inflatable-bag grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload.get("keyframe")
        if not isinstance(keyframe, dict):
            raise ValueError(f"Missing keyframe object in recorded grasp: {path}")
        gizmo = keyframe.get("gizmo_world")
        if not isinstance(gizmo, dict):
            raise ValueError(f"Missing gizmo_world in recorded grasp: {path}")
        gizmo_position = cls._validated_vector(gizmo.get("position_m"), 3, "gizmo position", path)
        gizmo_rotation = cls._validated_vector(gizmo.get("quaternion_xyzw"), 4, "gizmo rotation", path)
        rotation_norm = float(np.linalg.norm(gizmo_rotation))
        if rotation_norm < 1e-08:
            raise ValueError(f"Zero-length gizmo quaternion in recorded grasp: {path}")
        gizmo_rotation /= rotation_norm
        position_offset = cls._validated_vector(keyframe.get("position_offset_mm"), 3, "position offset", path)
        rotation_offset = cls._validated_vector(keyframe.get("rotation_offset_deg"), 3, "rotation offset", path)
        grasp_position = gizmo_position + position_offset * np.asarray((0.001, 0.001, 0.0))
        rx, ry, rz = np.radians(rotation_offset)
        offset_rotation = _m1_Example._quat_mul(
            _m1_Example._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        grasp_rotation = _m1_Example._quat_mul(offset_rotation, wp.quat(*gizmo_rotation))
        recorded_target = keyframe.get("target_root_pose")
        if not isinstance(recorded_target, dict):
            raise ValueError(f"Missing target_root_pose in recorded grasp: {path}")
        recorded_position = cls._validated_vector(recorded_target.get("position_m"), 3, "target root position", path)
        recorded_rotation = cls._validated_vector(
            recorded_target.get("quaternion_xyzw"), 4, "target root rotation", path
        )
        recorded_rotation /= max(float(np.linalg.norm(recorded_rotation)), 1e-08)
        if not np.allclose(recorded_position[:2], grasp_position[:2], atol=2e-05):
            raise ValueError(f"Recorded target root XY is inconsistent with gizmo_world and position offsets: {path}")
        if abs(float(np.dot(recorded_rotation, grasp_rotation))) < 1.0 - 1e-05:
            raise ValueError(f"Recorded target rotation is inconsistent with rotation offsets: {path}")
        joints = keyframe.get("finger_joints_degrees")
        if not isinstance(joints, dict):
            joints = keyframe.get("target_finger_joints_degrees")
        if not isinstance(joints, dict):
            raise ValueError(f"Missing finger joints in recorded grasp: {path}")
        missing_joints = set(_m2_HAND_JOINTS) - joints.keys()
        if missing_joints:
            raise ValueError(f"Missing hand joints in recorded grasp {path}: {sorted(missing_joints)}")
        joint_targets = {name: float(joints[name]) for name in _m2_HAND_JOINTS}
        if not np.all(np.isfinite(tuple(joint_targets.values()))):
            raise ValueError(f"Non-finite finger joints in recorded grasp: {path}")
        return _m2__RecordedGrasp(
            root=wp.transform(wp.vec3(*grasp_position), grasp_rotation),
            joints_degrees=joint_targets,
            lift_height=_m2_SCRIPTED_LIFT_HEIGHT,
        )

    def _validate_initial_pose(self):
        """Verify initial and recorded targets are finite and within limits."""
        initial_q = self.state_0.joint_q.numpy()
        expected_position = np.asarray(wp.transform_get_translation(_m1_INITIAL_HAND_ROOT), dtype=np.float64)
        expected_rotation = np.asarray(wp.transform_get_rotation(_m1_INITIAL_HAND_ROOT), dtype=np.float64)
        actual_root = initial_q[self.root_q_start : self.root_q_start + 7]
        if not np.allclose(actual_root[:3], expected_position, atol=1e-06):
            raise ValueError("Initial hand root position does not match the recorder pose.")
        if abs(float(np.dot(actual_root[3:], expected_rotation))) < 1.0 - 1e-05:
            raise ValueError("Initial hand root rotation does not match the recorder pose.")
        for name, q_index in self.hand_joint_indices.items():
            lower, upper = self.joint_limits[name]
            initial_degrees = float(np.degrees(initial_q[q_index]))
            expected_degrees = float(_m2_OPEN_JOINTS[name])
            if abs(initial_degrees - expected_degrees) > 0.0001:
                raise ValueError(f"Initial joint {name} is {initial_degrees:.6f}°, expected {expected_degrees:.6f}°.")
            grasp_degrees = self.recorded_grasp.joints_degrees[name]
            if not lower - 0.0001 <= grasp_degrees <= upper + 0.0001:
                raise ValueError(f"Recorded joint {name}={grasp_degrees:.6f}° is outside [{lower:.6f}, {upper:.6f}]°.")
        cavity_index = self.cavity.cavity_index
        initial_volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        if abs(initial_volume / self.cavity.rest_volume - 1.0) > 0.0001:
            raise ValueError("Initial inflatable-bag volume does not match its authored rest volume.")

    def _raise_pressure_limit(self):
        """Avoid losing target-volume response when the grasp pressure rises."""
        pressure_limit = self.model.pneumatic.max_absolute_pressure.numpy()
        pressure_limit[self.cavity.cavity_index] = _m2_DEMO_MAX_ABSOLUTE_PRESSURE
        self.model.pneumatic.max_absolute_pressure.assign(pressure_limit)

    @staticmethod
    def _copy_transform(transform: wp.transform) -> wp.transform:
        """Copy one Warp transform."""
        return wp.transform(
            wp.vec3(*wp.transform_get_translation(transform)), wp.quat(*wp.transform_get_rotation(transform))
        )

    @staticmethod
    def _transform_duration(start: wp.transform, end: wp.transform) -> float:
        """Compute a duration bounded by linear and angular hand speeds."""
        start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
        end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
        distance = float(np.linalg.norm(end_position - start_position))
        start_rotation = np.asarray(wp.transform_get_rotation(start), dtype=np.float64)
        end_rotation = np.asarray(wp.transform_get_rotation(end), dtype=np.float64)
        cosine = float(np.clip(abs(np.dot(start_rotation, end_rotation)), 0.0, 1.0))
        angle = 2.0 * float(np.arccos(cosine))
        angular_speed = float(np.radians(_m2_HAND_ANGULAR_SPEED_DEG_S))
        return max(_m2_MINIMUM_ROOT_PHASE_DURATION, distance / _m2_HAND_TRANSLATION_SPEED, angle / angular_speed)

    def _build_phases(self) -> tuple[_m2__Phase, ...]:
        """Build validation, grasp, lift, hold, and physical-release phases."""
        initial_root = self._copy_transform(_m1_INITIAL_HAND_ROOT)
        grasp_root = self._copy_transform(self.recorded_grasp.root)
        grasp_position = wp.transform_get_translation(grasp_root)
        grasp_rotation = wp.transform_get_rotation(grasp_root)
        lifted_root = wp.transform(grasp_position + wp.vec3(0.0, 0.0, self.recorded_grasp.lift_height), grasp_rotation)
        maximum_finger_delta = max(
            abs(self.recorded_grasp.joints_degrees[name] - _m2_OPEN_JOINTS[name]) for name in _m2_HAND_JOINTS
        )
        finger_duration = maximum_finger_delta / _m1_MAX_FINGER_CONTACT_SPEED_DEG_S + _m2_FINGER_PHASE_PADDING
        return (
            _m2__Phase("validate_initial", _m2_INITIAL_POSE_HOLD_DURATION, initial_root, initial_root, _m2_OPEN_JOINTS),
            _m2__Phase(
                "approach",
                self._transform_duration(initial_root, grasp_root),
                initial_root,
                grasp_root,
                _m2_OPEN_JOINTS,
            ),
            _m2__Phase("close", finger_duration, grasp_root, grasp_root, self.recorded_grasp.joints_degrees),
            _m2__Phase(
                "grasp_settle", _m2_GRASP_SETTLE_DURATION, grasp_root, grasp_root, self.recorded_grasp.joints_degrees
            ),
            _m2__Phase(
                "lift",
                self._transform_duration(grasp_root, lifted_root),
                grasp_root,
                lifted_root,
                self.recorded_grasp.joints_degrees,
            ),
            _m2__Phase(
                "lifted_hold", _m2_LIFTED_HOLD_DURATION, lifted_root, lifted_root, self.recorded_grasp.joints_degrees
            ),
            _m2__Phase("release", finger_duration, lifted_root, lifted_root, _m2_OPEN_JOINTS, release=True),
            _m2__Phase(
                "drop_settle", _m2_DROP_SETTLE_DURATION, lifted_root, lifted_root, _m2_OPEN_JOINTS, release=True
            ),
        )

    @staticmethod
    def _interpolate_transform(start: wp.transform, end: wp.transform, alpha: float) -> wp.transform:
        """Interpolate translation and take the shortest normalized quaternion path."""
        start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
        end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
        start_rotation = np.asarray(wp.transform_get_rotation(start), dtype=np.float64)
        end_rotation = np.asarray(wp.transform_get_rotation(end), dtype=np.float64)
        if np.dot(start_rotation, end_rotation) < 0.0:
            end_rotation = -end_rotation
        rotation = start_rotation * (1.0 - alpha) + end_rotation * alpha
        rotation /= max(float(np.linalg.norm(rotation)), 1e-08)
        position = start_position * (1.0 - alpha) + end_position * alpha
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    def _sample(self, time_s: float) -> tuple[wp.transform, dict[str, float], int]:
        """Sample the current scripted root and rate-limited finger target."""
        for phase_index, phase in enumerate(self.phases):
            if time_s <= phase.duration:
                alpha = float(np.clip(time_s / phase.duration, 0.0, 1.0))
                root = self._interpolate_transform(phase.root_start, phase.root_end, alpha)
                return (root, phase.finger_target_degrees, phase_index)
            time_s -= phase.duration
        final_index = len(self.phases) - 1
        final_phase = self.phases[final_index]
        return (self._copy_transform(final_phase.root_end), final_phase.finger_target_degrees, final_index)

    def _set_hand_target(self, root: wp.transform, joints_degrees: dict[str, float]):
        """Set the next root and finger targets without teleporting bag particles."""
        target_q = self.script_target_q
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, q_index in self.hand_joint_indices.items():
            target_q[q_index] = np.radians(joints_degrees[name])
        self.manual_target_q.assign(target_q)

    def _bag_center_z(self) -> float:
        """Return the current inflatable-bag center height [m]."""
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        return float(np.mean(positions[:, 2]))

    def _apply_release_material(self):
        """Remove hand friction once finger opening begins."""
        if self.release_material_applied:
            return
        friction = self.model.shape_material_mu.numpy()
        friction[: self.hand_shape_end] = _m2_RELEASE_FRICTION
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_mu = _m2_RELEASE_FRICTION
        if self.soft_contact_material_index is not None:
            self.soft_contact_material_index.fill_(_m2__SOFT_MATERIAL_RELEASE)
        self.release_material_applied = True

    def _enter_phase(self, phase_index: int):
        """Record lift diagnostics and apply one-time phase changes."""
        if phase_index == self.active_phase_index:
            return
        self.active_phase_index = phase_index
        phase = self.phases[phase_index]
        self.active_phase_name = phase.name
        if phase.name == "release":
            self.lifted_bag_center_z = self._bag_center_z()
        if phase.release:
            self._apply_release_material()

    def _capture_simulation_graph(self):
        """Capture the warmed physics frame as one reusable CUDA graph."""
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._advance_physics_frame()
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}")

    def step(self):
        """Advance one frame, replaying a warmed CUDA graph when available."""
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
        """Render the physical hand and pneumatic bag without recorder controls."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/inflatable_bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        wp.launch(
            _m1__gather_edges,
            dim=len(self.bag_edge_starts),
            inputs=[self.state_0.particle_q, self.bag_edges, 0.0001],
            outputs=[self.bag_edge_starts, self.bag_edge_ends],
            device=self.model.device,
        )
        self.viewer.log_lines("/inflatable_bag/grid", self.bag_edge_starts, self.bag_edge_ends, (0.08, 0.06, 0.02))
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify every phase keeps the hand and pneumatic state finite."""
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        joint_q = self.state_0.joint_q.numpy()
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        volume_ratio = volume / self.cavity.rest_volume
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(joint_q))
        assert np.isfinite(volume_ratio) and volume_ratio > 0.7
        assert np.isfinite(pressure) and 0.0 < pressure <= _m2_DEMO_MAX_ABSOLUTE_PRESSURE + 1.0
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, volume_ratio)
        self.maximum_pressure = max(self.maximum_pressure, pressure)
        if self.active_phase_name == "validate_initial":
            hand_contacts, _ = self._contact_counts()
            assert hand_contacts == 0, f"Initial hand pose unexpectedly has {hand_contacts} bag contacts."

    def test_final(self):
        """Verify the hand lifts and physically releases the inflatable bag."""
        super().test_final()
        if self.use_graph:
            assert self.graph is not None, "The warmed CUDA physics frame was not captured."
        if self.sim_time + self.frame_dt < self.script_duration:
            return
        assert self.lifted_bag_center_z is not None, "The scripted lift phase did not complete."
        assert self.lifted_bag_center_z > self.initial_bag_center_z + 0.01, (
            f"The bag was not lifted: initial z={self.initial_bag_center_z:.6f}, lifted z={self.lifted_bag_center_z:.6f}."
        )
        final_bag_center_z = self._bag_center_z()
        assert final_bag_center_z < self.lifted_bag_center_z - 0.008, (
            f"The released bag did not fall: lifted z={self.lifted_bag_center_z:.6f}, final z={final_bag_center_z:.6f}."
        )
        assert self.minimum_volume_ratio > 0.85, (
            f"The grasp compressed the target-volume bag excessively: minimum ratio={self.minimum_volume_ratio:.6f}."
        )
        joint_q = self.state_0.joint_q.numpy()
        maximum_open_error = max(
            (
                abs(float(np.degrees(joint_q[q_index])) - _m2_OPEN_JOINTS[name])
                for name, q_index in self.hand_joint_indices.items()
            )
        )
        assert maximum_open_error < 2.0, f"The hand did not reopen fully: error={maximum_open_error:.3f}°."

    @staticmethod
    def create_parser():
        """Create command-line arguments for the recorded pick-and-release demo."""
        parser = _m1_Example.create_parser()
        parser.set_defaults(num_frames=720, paused=False, pneumatic_mode="target-volume")
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics frame as one CUDA graph.",
        )
        parser.add_argument(
            "--grasp-keyframe",
            default=str(_m2_DEFAULT_GRASP_KEYFRAME),
            help="Inflatable-bag grasp keyframe JSON generated by the recorder.",
        )
        return parser


def _m2_main():
    """Run the recorded inflatable-bag pick-and-release demo."""
    parser = _m2_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m2_Example(viewer, args), args)


import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCoVBD

_m3_FPS = 60
_m3_INITIAL_IK_ITERATIONS = 240
_m3_RUNTIME_IK_ITERATIONS = 24
_m3_END_EFFECTOR_POSITION_TOLERANCE = 0.0005
_m3_END_EFFECTOR_ANGLE_TOLERANCE_DEG = 0.25
_m3_HAND_CONTACT_KE = 600000.0
_m3_TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
_m3_RIGHT_J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
_m3_RIGHT_J7_TO_HAND_BASE_ROTATION = wp.quat(0.5, -0.5, 0.5, 0.5)
_m3_WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
_m3_WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
_m3_CAMERA_POS = wp.vec3(0.37, -3.15, 1.57)
_m3_CAMERA_FOV = 45.0
_m3_CAMERA_PITCH = -23.3
_m3_CAMERA_YAW = 150.4
_m3_DEFAULT_HOUSE_USD = "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
_m3__SOFT_MATERIAL_GRASP = 0
_m3__SOFT_MATERIAL_RELEASE = 1


@wp.kernel
def _m3__interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    joint_coord = wp.tid()
    out[joint_coord] = q0[joint_coord] * (1.0 - alpha) + q1[joint_coord] * alpha


@wp.kernel
def _m3__joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    joint_dof = wp.tid()
    out[joint_dof] = (q1[joint_dof] - q0[joint_dof]) * inv_dt


@wp.kernel
def _m3__lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    index = wp.tid()
    q[0, indices[index]] = values[index]


@wp.kernel
def _m3__accumulate_contact_diagnostics(
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


@wp.kernel
def _m3__copy_joint_q(source: wp.array[float], target: wp.array[float]):
    joint_coord = wp.tid()
    target[joint_coord] = source[joint_coord]


@wp.kernel
def _m3__limit_right_finger_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    desired_finger_q: wp.array[float],
    soft_contact_count: wp.array[int],
    soft_contact_shape: wp.array[int],
    right_hand_shape_mask: wp.array[int],
    free_max_step: float,
    contact_max_step: float,
    target_q: wp.array[float],
):
    finger = wp.tid()
    active_contact_count = wp.min(soft_contact_count[0], soft_contact_shape.shape[0])
    hand_contact = bool(False)
    for contact in range(active_contact_count):
        shape = soft_contact_shape[contact]
        if shape >= 0 and shape < right_hand_shape_mask.shape[0] and (right_hand_shape_mask[shape] != 0):
            hand_contact = True
    max_step = free_max_step
    if hand_contact:
        max_step = contact_max_step
    q_index = finger_q_indices[finger]
    delta = wp.clamp(desired_finger_q[finger] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


class _m3_Example:
    """Track the isolated-hand pneumatic-bag trajectory with the full W1."""

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
    _copy_transform = staticmethod(_m2_Example._copy_transform)
    _transform_duration = staticmethod(_m2_Example._transform_duration)
    _interpolate_transform = staticmethod(_m2_Example._interpolate_transform)

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / _m3_FPS
        self.sim_substeps = _m1_SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd
        self.recorded_grasp = _m2_Example._load_recorded_grasp(args.grasp_keyframe)
        self.contact_phase = None
        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="kinematic",
            contact_mode="soft",
            vbd_options=self._solver_vbd_options(),
            collision_options=self._solver_collision_options(),
            coupling_mode="one_way",
        )
        features = self.solver.features
        if (
            features.backend.value != "one_way_kinematic_soft"
            or features.vbd_core != "full"
            or features.contact_pipeline != "soft"
            or features.feedback_enabled
        ):
            raise RuntimeError(f"Unexpected pneumatic one-way backend: {features}")
        self.contacts = self.solver.contacts
        self.maximum_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.maximum_body_particle_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self._build_ik()
        self.phases = _m2_Example._build_phases(self)
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._right_hand_q()
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self._build_joint_target_cache()
        self._raise_pressure_limit()
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        right_hand_shape_mask = np.zeros(self.model.shape_count, dtype=np.int32)
        right_hand_shape_mask[self.right_hand_shapes] = 1
        self.right_hand_shape_mask = wp.array(right_hand_shape_mask, dtype=wp.int32, device=self.device)
        self.desired_finger_q = wp.zeros(self.hand_indices.shape[0], dtype=wp.float32, device=self.device)
        self.max_finger_step = float(np.radians(_m1_MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self.max_finger_contact_step = float(np.radians(_m1_MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt)
        contact_materials = np.asarray(
            ((_m1_CONTACT_KE, _m1_CONTACT_KD, _m1_CONTACT_MU), (_m1_CONTACT_KE, _m1_CONTACT_KD, _m2_RELEASE_FRICTION)),
            dtype=np.float32,
        )
        self.soft_contact_materials = wp.array(contact_materials, dtype=wp.vec3, device=self.device)
        self.soft_contact_material_index = wp.full(1, _m3__SOFT_MATERIAL_GRASP, dtype=wp.int32, device=self.device)
        self.solver.vbd_solver.set_soft_contact_material_source(
            self.soft_contact_materials, self.soft_contact_material_index
        )
        self.graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.release_material_applied = False
        self.active_phase_index = -1
        self.active_phase_name = "initialization"
        self.current_target_root = self._copy_transform(_m1_INITIAL_HAND_ROOT)
        self.initial_bag_center_z = self._bag_center_z()
        self.lifted_bag_center_z: float | None = None
        self.minimum_volume_ratio = 1.0
        self.maximum_pressure = _m1_BAG_REFERENCE_ABSOLUTE_PRESSURE
        self.maximum_root_position_error = 0.0
        self.maximum_root_angle_error = 0.0
        self.script_duration = sum(phase.duration for phase in self.phases)
        self._attach_house_usd()
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(_m3_CAMERA_POS, _m3_CAMERA_PITCH, _m3_CAMERA_YAW)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = _m3_CAMERA_FOV
        self._validate_mount_mapping()
        self._validate_initial_pose()

    def _solver_vbd_options(self):
        """Match the isolated pneumatic-bag solver configuration."""
        return {
            "iterations": _m1_VBD_ITERATIONS,
            "rigid_body_contact_buffer_size": _m1_RIGID_BODY_CONTACT_BUFFER_SIZE,
            "rigid_body_particle_contact_buffer_size": _m1_RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
            "particle_enable_self_contact": False,
        }

    def _solver_collision_options(self):
        """Match the isolated pneumatic-bag collision pipeline."""
        return {"soft_contact_margin": _m1_SOFT_CONTACT_MARGIN}

    def _robot_urdf(self) -> Path:
        """Return the configured full Dexforce W1 URDF."""
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = Path(__file__).resolve().parents[3] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"
        if path.is_file():
            return path
        raise FileNotFoundError("Dexforce W1 URDF is unavailable; pass --robot-urdf PATH.")

    def _build_scene(self):
        """Build the full W1 around the original table and pneumatic bag."""
        recorder = hand_reference.recorder
        hand_recorder = recorder.hand_recorder
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = recorder.CONTACT_KE
        builder.default_shape_cfg.kd = recorder.CONTACT_KD
        builder.default_shape_cfg.mu = recorder.CONTACT_MU
        builder.default_shape_cfg.margin = recorder.SHAPE_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMuJoCoVBD.register_custom_attributes(builder)
        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(articulation_start, builder.articulation_count))
        if not self.robot_articulations:
            raise RuntimeError("Dexforce W1 URDF did not create an articulation.")
        self.robot_body_end = builder.body_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=recorder.CONTACT_KE,
            kd=recorder.CONTACT_KD,
            mu=0.9,
            margin=recorder.SHAPE_CONTACT_MARGIN,
            is_visible=bool(self.args.show_physics_table),
        )
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(hand_recorder.TABLE_POS, hand_recorder.TABLE_ROTATION),
            hx=hand_recorder.TABLE_HALF_EXTENTS[0],
            hy=hand_recorder.TABLE_HALF_EXTENTS[1],
            hz=hand_recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="full_w1_inflatable_bag_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="full_w1_inflatable_bag_ground")
        bag_mesh = recorder._load_chip_bag_mesh()
        bag_rest_volume = recorder._scaled_mesh_volume(bag_mesh, recorder.BAG_SCALE)
        self.pneumatic_mode_name = self.args.pneumatic_mode
        self.pneumatic_config = recorder._make_pneumatic_config(self.pneumatic_mode_name, bag_rest_volume)
        self.bag_particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=recorder.BAG_CENTER,
            rot=recorder.BAG_ROTATION,
            scale=recorder.BAG_SCALE,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=recorder.BAG_DENSITY,
            tri_ke=recorder.BAG_TRI_KE,
            tri_ka=recorder.BAG_TRI_KA,
            tri_kd=recorder.BAG_TRI_KD,
            edge_ke=recorder.BAG_EDGE_KE,
            edge_kd=recorder.BAG_EDGE_KD,
            particle_radius=recorder.BAG_PARTICLE_RADIUS,
            validate_mesh=True,
            label="full_w1_graspable_sealed_chip_bag",
            config=self.pneumatic_config,
        )
        self.bag_particle_end = builder.particle_count
        bag_triangle_indices = np.asarray(bag_mesh.indices, dtype=np.int32) + self.bag_particle_start
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collision_mask = collide_shapes | collide_particles
        self.hand_shapes = []
        self.right_hand_shapes = []
        self.robot_visual_shapes = []
        for shape in range(self.robot_shape_end):
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if not is_collider:
                self.robot_visual_shapes.append(shape)
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            right_hand_shape = "right" in label and any(keyword in label for keyword in self.HAND_CONTACT_KEYWORDS)
            if right_hand_shape and is_collider:
                self.hand_shapes.append(shape)
                self.right_hand_shapes.append(shape)
                builder.shape_flags[shape] |= collide_shapes | collide_particles
            else:
                builder.shape_flags[shape] &= ~collision_mask
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = recorder.CONTACT_KE
        self.model.soft_contact_kd = recorder.CONTACT_KD
        self.model.soft_contact_mu = recorder.CONTACT_MU
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[self.right_hand_shapes] = recorder.CONTACT_MU
        shape_ke[self.right_hand_shapes] = _m3_HAND_CONTACT_KE
        shape_kd[self.right_hand_shapes] = recorder.CONTACT_KD
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.bag_triangle_indices = wp.array(bag_triangle_indices, dtype=wp.int32, device=self.model.device)

    def _build_ik(self):
        """Build the independent full-W1 arm IK model."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
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
            _m3_TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.left_home)], dtype=wp.vec3, device=self.device),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.left_home))], dtype=wp.vec4, device=self.device),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            _m3_TCP_OFFSET,
            wp.array([wp.transform_get_translation(self.right_home)], dtype=wp.vec3, device=self.device),
        )
        self.right_rot = ik.IKObjectiveRotation(
            right,
            wp.quat_identity(),
            wp.array([self._v4(wp.transform_get_rotation(self.right_home))], dtype=wp.vec4, device=self.device),
        )
        lower, upper = self._joint_limits()
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _root_to_tcp(self, root_transform: wp.transform) -> wp.transform:
        """Convert an isolated hand-root pose to the full-W1 wrist TCP."""
        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = self._quat_mul(hand_rotation, wp.quat_inverse(_m3_RIGHT_J7_TO_HAND_BASE_ROTATION))
        target_offset = _m3_TCP_OFFSET - _m3_RIGHT_J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _tcp_to_root(self, tcp_transform: wp.transform) -> wp.transform:
        """Recover the isolated hand-root pose from a full-W1 wrist TCP."""
        tcp_position = wp.transform_get_translation(tcp_transform)
        wrist_rotation = wp.transform_get_rotation(tcp_transform)
        root_position = tcp_position + wp.quat_rotate(wrist_rotation, _m3_RIGHT_J7_TO_HAND_BASE_OFFSET - _m3_TCP_OFFSET)
        root_rotation = self._quat_mul(wrist_rotation, _m3_RIGHT_J7_TO_HAND_BASE_ROTATION)
        return wp.transform(root_position, root_rotation)

    def _sample_hand_trajectory(self, time_s: float):
        """Sample the original root, finger target, and phase without rebuilding it."""
        return _m2_Example._sample(self, time_s)

    def _right_hand_q(self):
        """Return the canonical open and recorded grasp finger configurations."""
        q_start = self.model.joint_q_start.numpy()
        indices = []
        open_q = []
        grasp_q = []
        for suffix in self.HAND_SUFFIXES:
            name = f"RIGHT_{suffix}"
            joint = self._joint_index(name)
            indices.append(int(q_start[joint]))
            open_q.append(np.radians(_m2_OPEN_JOINTS[name]))
            grasp_q.append(np.radians(self.recorded_grasp.joints_degrees[name]))
        self.hand_start = wp.array(open_q, dtype=wp.float32, device=self.device)
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(open_q, dtype=wp.float32, device=self.device),
            wp.array(grasp_q, dtype=wp.float32, device=self.device),
        )

    def _build_joint_target_cache(self):
        """Initialize W1 at the isolated example's exact first hand pose."""
        initial_root = _m1_INITIAL_HAND_ROOT
        initial_tcp = self._root_to_tcp(initial_root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(initial_tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(initial_tcp)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m3_INITIAL_IK_ITERATIONS)
        wp.launch(
            _m3__lock_q,
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
        self.state_0.joint_qd.zero_()
        self.state_1.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _raise_pressure_limit(self):
        """Preserve target-volume response throughout the physical grasp."""
        pressure_limit = self.model.pneumatic.max_absolute_pressure.numpy()
        pressure_limit[self.cavity.cavity_index] = _m2_DEMO_MAX_ABSOLUTE_PRESSURE
        self.model.pneumatic.max_absolute_pressure.assign(pressure_limit)

    def _validate_mount_mapping(self):
        """Verify every phase maps to TCP and back without changing its root pose."""
        for phase in self.phases:
            for root in (phase.root_start, phase.root_end):
                recovered = self._tcp_to_root(self._root_to_tcp(root))
                position_error, angle_error = self._transform_error(recovered, root)
                if position_error > 1e-06 or angle_error > np.radians(0.0001):
                    raise ValueError(
                        f"The W1 wrist mount transform changes the isolated hand-root trajectory: position error={position_error:.9f} m, angle error={np.degrees(angle_error):.9f}°."
                    )

    def _validate_initial_pose(self):
        """Verify the initial arm, fingers, and cavity match the isolated scene."""
        actual_root = self._actual_hand_root()
        target_root = _m1_INITIAL_HAND_ROOT
        position_error, angle_error = self._transform_error(actual_root, target_root)
        if position_error > _m3_END_EFFECTOR_POSITION_TOLERANCE:
            raise ValueError(f"Initial W1 hand-root position error is {position_error:.6f} m.")
        if angle_error > np.radians(_m3_END_EFFECTOR_ANGLE_TOLERANCE_DEG):
            raise ValueError(f"Initial W1 hand-root angle error is {np.degrees(angle_error):.6f}°.")
        joint_q = self.state_0.joint_q.numpy()
        for suffix, q_index in zip(self.HAND_SUFFIXES, self.hand_indices.numpy(), strict=True):
            name = f"RIGHT_{suffix}"
            actual_degrees = float(np.degrees(joint_q[q_index]))
            expected_degrees = _m2_OPEN_JOINTS[name]
            if abs(actual_degrees - expected_degrees) > 0.0001:
                raise ValueError(f"Initial joint {name} is {actual_degrees:.6f}°, expected {expected_degrees:.6f}°.")
        cavity_index = self.cavity.cavity_index
        initial_volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        if abs(initial_volume / self.cavity.rest_volume - 1.0) > 0.0001:
            raise ValueError("Initial inflatable-bag volume does not match its authored rest volume.")

    def _actual_hand_root(self) -> wp.transform:
        """Return the full robot's current ``right_hand_base`` world pose."""
        body_transform = wp.transform(*self.state_0.body_q.numpy()[self.right_body])
        wrist_position = wp.transform_get_translation(body_transform)
        wrist_rotation = wp.transform_get_rotation(body_transform)
        root_position = wrist_position + wp.quat_rotate(wrist_rotation, _m3_RIGHT_J7_TO_HAND_BASE_OFFSET)
        root_rotation = self._quat_mul(wrist_rotation, _m3_RIGHT_J7_TO_HAND_BASE_ROTATION)
        return wp.transform(root_position, root_rotation)

    @staticmethod
    def _transform_error(actual: wp.transform, target: wp.transform) -> tuple[float, float]:
        """Return translation [m] and shortest-angle [rad] transform errors."""
        actual_position = np.asarray(wp.transform_get_translation(actual), dtype=np.float64)
        target_position = np.asarray(wp.transform_get_translation(target), dtype=np.float64)
        position_error = float(np.linalg.norm(actual_position - target_position))
        actual_rotation = np.asarray(wp.transform_get_rotation(actual), dtype=np.float64)
        target_rotation = np.asarray(wp.transform_get_rotation(target), dtype=np.float64)
        actual_rotation /= max(float(np.linalg.norm(actual_rotation)), 1e-12)
        target_rotation /= max(float(np.linalg.norm(target_rotation)), 1e-12)
        cosine = float(np.clip(abs(np.dot(actual_rotation, target_rotation)), 0.0, 1.0))
        return (position_error, 2.0 * float(np.arccos(cosine)))

    def _bag_center_z(self) -> float:
        """Return the current inflatable-bag center height [m]."""
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        return float(np.mean(positions[:, 2]))

    def _contact_counts(self) -> tuple[int, int]:
        """Return current right-hand-to-bag and total rigid-soft contacts."""
        total = int(self.contacts.soft_contact_count.numpy()[0])
        shape_indices = self.contacts.soft_contact_shape.numpy()
        active = min(total, shape_indices.shape[0])
        shape_mask = self.right_hand_shape_mask.numpy()
        active_shapes = shape_indices[:active]
        valid = (active_shapes >= 0) & (active_shapes < shape_mask.shape[0])
        safe_shapes = np.clip(active_shapes, 0, shape_mask.shape[0] - 1)
        hand_contacts = int(np.count_nonzero(valid & (shape_mask[safe_shapes] != 0)))
        return (hand_contacts, total)

    def _apply_release_material(self):
        """Use the isolated example's release friction once opening begins."""
        if self.release_material_applied:
            return
        friction = self.model.shape_material_mu.numpy()
        friction[self.right_hand_shapes] = _m2_RELEASE_FRICTION
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_mu = _m2_RELEASE_FRICTION
        self.soft_contact_material_index.fill_(_m3__SOFT_MATERIAL_RELEASE)
        self.release_material_applied = True

    def _enter_phase(self, phase_index: int):
        """Apply one-time diagnostics and material changes at phase boundaries."""
        if phase_index == self.active_phase_index:
            return
        self.active_phase_index = phase_index
        phase = self.phases[phase_index]
        self.active_phase_name = phase.name
        if phase.name == "release":
            self.lifted_bag_center_z = self._bag_center_z()
        if phase.release:
            self._apply_release_material()

    def _prepare_frame(self):
        """Solve the arm target sampled directly from the isolated trajectory."""
        script_time = self.sim_time * self.args.trajectory_time_scale
        root, finger_joints, phase_index = self._sample_hand_trajectory(script_time)
        self.current_target_root = root
        self._enter_phase(phase_index)
        tcp = self._root_to_tcp(root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(tcp))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(tcp)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=_m3_RUNTIME_IK_ITERATIONS)
        wp.launch(
            _m3__lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.launch(_m3__copy_joint_q, self.model.joint_coord_count, [self.ik_q[0], self.frame_q_end], device=self.device)
        self.desired_finger_q.assign([np.radians(finger_joints[f"RIGHT_{suffix}"]) for suffix in self.HAND_SUFFIXES])
        wp.launch(
            _m3__limit_right_finger_target_step,
            self.hand_indices.shape[0],
            [
                self.frame_q_start,
                self.hand_indices,
                self.desired_finger_q,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.right_hand_shape_mask,
                self.max_finger_step,
                self.max_finger_contact_step,
                self.frame_q_end,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance one full-W1 frame with the isolated scene's substep count."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _m3__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m3__joint_velocity,
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
                _m3__accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            if self.sim_substeps % 2 != 0 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = (self.state_1, self.state_0)

    def _capture_simulation_graph(self):
        """Capture the warmed, fixed physics substeps as one CUDA graph."""
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}.")

    def step(self):
        """Advance one frame using the exact source target and warmed graph."""
        self._prepare_frame()
        if self.graph is None:
            self._simulate_substeps()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        """Render the complete W1 and the pneumatic bag surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/inflatable_bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify finite pneumatic state and accurate hand-root tracking."""
        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        joint_q = self.state_0.joint_q.numpy()
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        volume_ratio = volume / self.cavity.rest_volume
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(joint_q))
        assert np.isfinite(volume_ratio) and volume_ratio > 0.7
        assert np.isfinite(pressure) and 0.0 < pressure <= _m2_DEMO_MAX_ABSOLUTE_PRESSURE + 1.0
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, volume_ratio)
        self.maximum_pressure = max(self.maximum_pressure, pressure)
        position_error, angle_error = self._transform_error(self._actual_hand_root(), self.current_target_root)
        self.maximum_root_position_error = max(self.maximum_root_position_error, position_error)
        self.maximum_root_angle_error = max(self.maximum_root_angle_error, angle_error)
        assert position_error <= _m3_END_EFFECTOR_POSITION_TOLERANCE, (
            f"W1 hand-root position error is {position_error:.6f} m."
        )
        assert angle_error <= np.radians(_m3_END_EFFECTOR_ANGLE_TOLERANCE_DEG), (
            f"W1 hand-root angle error is {np.degrees(angle_error):.6f}°."
        )
        if self.active_phase_name == "validate_initial":
            hand_contacts, _ = self._contact_counts()
            assert hand_contacts == 0, f"Initial W1 hand pose unexpectedly has {hand_contacts} bag contacts."

    def test_final(self):
        """Verify trajectory fidelity, physical lift, release, and bag volume."""
        assert not any("physical_pad" in label for label in self.model.shape_label)
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        visual_flags = self.model.shape_flags.numpy()[self.robot_visual_shapes]
        assert np.all(visual_flags & collision_mask == 0), "Robot visual shapes must remain non-colliding"
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        if self.use_graph:
            assert self.graph is not None, "The warmed CUDA physics graph was not captured."
        script_time = self.sim_time * self.args.trajectory_time_scale
        if script_time + self.frame_dt * self.args.trajectory_time_scale < self.script_duration:
            return
        assert self.lifted_bag_center_z is not None, "The scripted lift phase did not complete."
        assert self.lifted_bag_center_z > self.initial_bag_center_z + 0.01, (
            f"The bag was not lifted: initial z={self.initial_bag_center_z:.6f}, lifted z={self.lifted_bag_center_z:.6f}."
        )
        final_bag_center_z = self._bag_center_z()
        assert final_bag_center_z < self.lifted_bag_center_z - 0.008, (
            f"The released bag did not fall: lifted z={self.lifted_bag_center_z:.6f}, final z={final_bag_center_z:.6f}."
        )
        assert self.minimum_volume_ratio > 0.85, (
            f"The grasp compressed the target-volume bag excessively: minimum ratio={self.minimum_volume_ratio:.6f}."
        )
        joint_q = self.state_0.joint_q.numpy()
        maximum_open_error = max(
            (
                abs(float(np.degrees(joint_q[q_index])) - _m2_OPEN_JOINTS[f"RIGHT_{suffix}"])
                for suffix, q_index in zip(self.HAND_SUFFIXES, self.hand_indices.numpy(), strict=True)
            )
        )
        assert maximum_open_error < 2.0, f"The hand did not reopen fully: error={maximum_open_error:.3f}°."

    def _joint_limits(self):
        """Lock non-arm coordinates in the IK objective limits."""
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
        """Return non-arm coordinates held fixed after every IK solve."""
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

    def _joint_index(self, name: str) -> int:
        """Return a model joint index from its unprefixed asset name."""
        return next((index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + name)))

    @staticmethod
    def _body_index(labels: list[str], name: str) -> int:
        """Return a body index from its unprefixed asset name."""
        return next((index for index, label in enumerate(labels) if label.endswith("/" + name)))

    def _tcp(self, state: newton.State, body: int) -> wp.transform:
        """Return one wrist TCP world transform."""
        body_transform = wp.transform(*state.body_q.numpy()[body])
        body_rotation = wp.transform_get_rotation(body_transform)
        return wp.transform(
            wp.transform_get_translation(body_transform) + wp.quat_rotate(body_rotation, _m3_TCP_OFFSET), body_rotation
        )

    def _attach_house_usd(self):
        """Attach the visual-only WAIC house background when available."""
        if not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.isfile(self.house_visual_usd):
            print(f"WAIC house USD not found; continuing without it: {self.house_visual_usd}")
            return
        prim = self.viewer.stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    @staticmethod
    def _normal_quat(value: wp.quat) -> wp.quat:
        """Return a normalized quaternion."""
        array = np.asarray([float(value[0]), float(value[1]), float(value[2]), float(value[3])])
        array /= max(float(np.linalg.norm(array)), 1e-08)
        return wp.quat(*array)

    @staticmethod
    def _quat_mul(a: wp.quat, b: wp.quat) -> wp.quat:
        """Multiply two quaternions without relying on expression overloads."""
        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _v4(value: wp.quat) -> wp.vec4:
        """Convert a quaternion to the IK rotation target type."""
        return wp.vec4(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    @staticmethod
    def create_parser():
        """Create full-W1 options plus the canonical pneumatic-bag inputs."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=720, paused=False)
        parser.add_argument("--robot-urdf", default=None, help="Optional Dexforce W1 URDF path.")
        parser.add_argument(
            "--house-visual-usd",
            default=_m3_DEFAULT_HOUSE_USD,
            help="Optional WAIC house USD reference; it is visual-only.",
        )
        parser.add_argument(
            "--show-physics-table",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Render the physical table collider.",
        )
        parser.add_argument("--trajectory-time-scale", type=float, default=1.0)
        parser.add_argument(
            "--pneumatic-mode",
            choices=tuple(_m1_PNEUMATIC_MODES),
            default="target-volume",
            help="Pressure law for the sealed bag (default: %(default)s).",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics substeps as one CUDA graph.",
        )
        parser.add_argument(
            "--grasp-keyframe",
            default=str(_m2_DEFAULT_GRASP_KEYFRAME),
            help="Inflatable-bag grasp keyframe used by the isolated-hand trajectory.",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(_m3_WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(_m3_WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(_m3_WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(_m3_WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(_m3_WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(_m3_WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(_m3_WAIC_ROBOT_BASE_QUAT[3]))
        return parser


def _m3_main():
    """Run the full-W1 recorded inflatable-bag pick-and-release demo."""
    parser = _m3_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m3_Example(viewer, args), args)


import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples

_m4_PLASTIC_YIELD_ANGLE_DEG = 1.0
_m4_PLASTIC_FLOW_RATE = 120.0
_m4_PLASTIC_MAX_ANGLE_DEG = 60.0
_m4_PLASTIC_HARDENING = 0.0
_m4_BAG_BENDING_STIFFNESS_SCALE = 0.25
_m4_BAG_BULK_DAMPING = 2000000.0


@wp.kernel
def _m4__update_bending_plasticity(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    authored_rest_angles: wp.array[float],
    rest_angles: wp.array[float],
    authored_bending_properties: wp.array2d[float],
    bending_properties: wp.array2d[float],
    bag_particle_start: int,
    bag_particle_end: int,
    yield_angle: float,
    flow_rate: float,
    max_plastic_angle: float,
    hardening: float,
    dt: float,
):
    """Move yielded edge rest angles toward the current bag shape."""
    edge = wp.tid()
    opposite_0 = edge_indices[edge, 0]
    opposite_1 = edge_indices[edge, 1]
    edge_start = edge_indices[edge, 2]
    edge_end = edge_indices[edge, 3]
    if opposite_0 < bag_particle_start or opposite_0 >= bag_particle_end:
        return
    if opposite_1 < bag_particle_start or opposite_1 >= bag_particle_end:
        return
    if edge_start < bag_particle_start or edge_start >= bag_particle_end:
        return
    if edge_end < bag_particle_start or edge_end >= bag_particle_end:
        return
    x0 = positions[opposite_0]
    x1 = positions[opposite_1]
    x2 = positions[edge_start]
    x3 = positions[edge_end]
    normal_0 = wp.cross(x2 - x0, x3 - x0)
    normal_1 = wp.cross(x3 - x1, x2 - x1)
    edge_vector = x3 - x2
    normal_0_length = wp.length(normal_0)
    normal_1_length = wp.length(normal_1)
    edge_length = wp.length(edge_vector)
    if normal_0_length < 1e-08 or normal_1_length < 1e-08 or edge_length < 1e-08:
        return
    normal_0 /= normal_0_length
    normal_1 /= normal_1_length
    edge_direction = edge_vector / edge_length
    current_angle = wp.atan2(wp.dot(wp.cross(normal_0, normal_1), edge_direction), wp.dot(normal_0, normal_1))
    current_rest_angle = rest_angles[edge]
    angle_error = wp.atan2(wp.sin(current_angle - current_rest_angle), wp.cos(current_angle - current_rest_angle))
    absolute_error = wp.abs(angle_error)
    if absolute_error <= yield_angle:
        return
    direction = 1.0
    if angle_error < 0.0:
        direction = -1.0
    authored_rest_angle = authored_rest_angles[edge]
    plastic_offset = current_rest_angle - authored_rest_angle
    adoption_fraction = wp.min(flow_rate * dt, 1.0)
    plastic_step_magnitude = (absolute_error - yield_angle) * adoption_fraction
    plastic_step = direction * plastic_step_magnitude
    plastic_offset = wp.clamp(plastic_offset + plastic_step, -max_plastic_angle, max_plastic_angle)
    rest_angles[edge] = authored_rest_angle + plastic_offset
    hardening_angle = wp.max(yield_angle, 1e-06)
    hardening_fraction = wp.min(wp.abs(plastic_offset) / hardening_angle, 1.0)
    bending_properties[edge, 0] = authored_bending_properties[edge, 0] * (1.0 + hardening * hardening_fraction)


def _m4__validate_nonnegative(value: float, label: str) -> float:
    """Return a finite nonnegative plasticity parameter."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value}")
    return value


class _m4_Example(_m2_Example):
    """Replay the recorded grasp with a plastically bending bag shell."""

    def __init__(self, viewer, args):
        self.plasticity_enabled = bool(args.plastic)
        yield_angle_deg = _m4__validate_nonnegative(args.plastic_yield_angle_deg, "plastic yield angle")
        max_angle_deg = _m4__validate_nonnegative(args.plastic_max_angle_deg, "maximum plastic angle")
        if yield_angle_deg >= 180.0:
            raise ValueError("plastic yield angle must be less than 180 degrees")
        if max_angle_deg >= 180.0:
            raise ValueError("maximum plastic angle must be less than 180 degrees")
        self.plastic_yield_angle = math.radians(yield_angle_deg)
        self.plastic_flow_rate = _m4__validate_nonnegative(args.plastic_flow_rate, "plastic flow rate")
        self.plastic_max_angle = math.radians(max_angle_deg)
        self.plastic_hardening = _m4__validate_nonnegative(args.plastic_hardening, "plastic hardening")
        self.bag_bending_stiffness_scale = _m4__validate_nonnegative(
            args.bag_bending_stiffness_scale, "bag bending stiffness scale"
        )
        self.bag_bulk_damping = _m4__validate_nonnegative(args.bag_bulk_damping, "bag bulk damping")
        super().__init__(viewer, args)
        if (
            self.model.edge_rest_angle is None
            or self.model.edge_indices is None
            or self.model.edge_bending_properties is None
            or (self.model.edge_count == 0)
        ):
            raise RuntimeError("The inflatable bag did not create bending edges.")
        edge_bending_properties = self.model.edge_bending_properties.numpy()
        edge_bending_properties[:, 0] *= self.bag_bending_stiffness_scale
        self.model.edge_bending_properties.assign(edge_bending_properties)
        pneumatic_bulk_damping = self.model.pneumatic.bulk_damping.numpy()
        pneumatic_bulk_damping[self.cavity.cavity_index] = self.bag_bulk_damping
        self.model.pneumatic.bulk_damping.assign(pneumatic_bulk_damping)
        self.authored_edge_rest_angle = wp.clone(self.model.edge_rest_angle)
        self.authored_edge_bending_properties = wp.clone(self.model.edge_bending_properties)

    def _apply_bending_plasticity(self):
        """Evolve the bag rest curvature after one completed physics substep."""
        wp.launch(
            _m4__update_bending_plasticity,
            dim=self.model.edge_count,
            inputs=[
                self.state_0.particle_q,
                self.model.edge_indices,
                self.authored_edge_rest_angle,
                self.model.edge_rest_angle,
                self.authored_edge_bending_properties,
                self.model.edge_bending_properties,
                self.bag_particle_start,
                self.bag_particle_end,
                self.plastic_yield_angle,
                self.plastic_flow_rate,
                self.plastic_max_angle,
                self.plastic_hardening,
                self.sim_dt,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance the hand and apply plastic flow after every substep."""
        for substep in range(_m1_SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / _m1_SIM_SUBSTEPS
            wp.launch(
                _m0__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m0__joint_velocity,
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
            if _m1_SIM_SUBSTEPS % 2 != 0 and substep == _m1_SIM_SUBSTEPS - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = (self.state_1, self.state_0)
            if self.plasticity_enabled:
                self._apply_bending_plasticity()

    def _capture_simulation_graph(self):
        """Capture one frame without committing capture-time plastic flow."""
        rest_angle_backup = wp.clone(self.model.edge_rest_angle)
        bending_properties_backup = wp.clone(self.model.edge_bending_properties)
        super()._capture_simulation_graph()
        wp.copy(self.model.edge_rest_angle, rest_angle_backup)
        wp.copy(self.model.edge_bending_properties, bending_properties_backup)

    def test_final(self):
        """Verify plastic flow stays finite, bounded, and switchable."""
        super().test_final()
        plastic_offset = self.model.edge_rest_angle.numpy() - self.authored_edge_rest_angle.numpy()
        assert np.all(np.isfinite(plastic_offset))
        maximum_offset = float(np.max(np.abs(plastic_offset), initial=0.0))
        assert maximum_offset <= self.plastic_max_angle + 1e-05
        bending_stiffness = self.model.edge_bending_properties.numpy()[:, 0]
        authored_stiffness = self.authored_edge_bending_properties.numpy()[:, 0]
        assert np.all(np.isfinite(bending_stiffness))
        assert np.all(bending_stiffness >= authored_stiffness)
        assert np.all(bending_stiffness <= authored_stiffness * (1.0 + self.plastic_hardening) + 1e-05)
        if not self.plasticity_enabled:
            assert maximum_offset < 1e-07, "Disabled plasticity changed the authored rest angles."
            assert np.array_equal(bending_stiffness, authored_stiffness)
            return
        if self.sim_time + self.frame_dt < self.script_duration:
            return
        assert maximum_offset > math.radians(0.1), "The completed grasp did not yield any bag edges."

    @staticmethod
    def create_parser():
        """Create command-line arguments for the plastic bag demo."""
        parser = _m2_Example.create_parser()
        parser.add_argument(
            "--plastic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable bending plasticity for the inflatable bag.",
        )
        parser.add_argument(
            "--plastic-yield-angle-deg",
            type=float,
            default=_m4_PLASTIC_YIELD_ANGLE_DEG,
            help="Dihedral-angle error that starts plastic flow [deg].",
        )
        parser.add_argument(
            "--plastic-flow-rate",
            type=float,
            default=_m4_PLASTIC_FLOW_RATE,
            help="Rest-angle adoption rate while forming a crease [1/s].",
        )
        parser.add_argument(
            "--plastic-max-angle-deg",
            type=float,
            default=_m4_PLASTIC_MAX_ANGLE_DEG,
            help="Maximum rest-angle offset from the authored bag [deg].",
        )
        parser.add_argument(
            "--plastic-hardening",
            type=float,
            default=_m4_PLASTIC_HARDENING,
            help="Maximum added bending-stiffness multiple on yielded edges.",
        )
        parser.add_argument(
            "--bag-bending-stiffness-scale",
            type=float,
            default=_m4_BAG_BENDING_STIFFNESS_SCALE,
            help="Scale applied to the elastic bag bending stiffness before yielding.",
        )
        parser.add_argument(
            "--bag-bulk-damping",
            type=float,
            default=_m4_BAG_BULK_DAMPING,
            help="Pneumatic volume-rate damping [Pa s/m^3].",
        )
        return parser


def _m4_main():
    """Run the plastic inflatable-bag pick-and-release demo."""
    parser = _m4_Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(_m4_Example(viewer, args), args)


"Replay the plastic inflatable-bag grasp with the full Dexforce W1.\n\nThe right hand follows the exact root and finger trajectory used by the\nisolated-hand v1 example. Each display frame converts that root pose to the\nfull-W1 right-wrist TCP and solves the arm with realtime analytic-jacobian IK.\nThe bag retains the source example's bending plasticity, pneumatic damping,\ncontact-aware finger closing, and release material.\n\nCUDA devices capture the warmed physics substeps by default. The realtime IK\nsolve remains outside that graph because its target changes every frame.\n\nRun from the repository root::\n\n    uv run --extra examples -m newton.examples         mujoco_vbd_recorded_plastic_inflatable_bag_pick_release --viewer gl\n"
import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples


class Example(_m3_Example):
    """Track the v1 plastic-bag trajectory with realtime full-W1 IK."""

    def __init__(self, viewer, args):
        self.plasticity_enabled = bool(args.plastic)
        yield_angle_deg = _m4__validate_nonnegative(args.plastic_yield_angle_deg, "plastic yield angle")
        max_angle_deg = _m4__validate_nonnegative(args.plastic_max_angle_deg, "maximum plastic angle")
        if yield_angle_deg >= 180.0:
            raise ValueError("plastic yield angle must be less than 180 degrees")
        if max_angle_deg >= 180.0:
            raise ValueError("maximum plastic angle must be less than 180 degrees")
        self.plastic_yield_angle = math.radians(yield_angle_deg)
        self.plastic_flow_rate = _m4__validate_nonnegative(args.plastic_flow_rate, "plastic flow rate")
        self.plastic_max_angle = math.radians(max_angle_deg)
        self.plastic_hardening = _m4__validate_nonnegative(args.plastic_hardening, "plastic hardening")
        self.bag_bending_stiffness_scale = _m4__validate_nonnegative(
            args.bag_bending_stiffness_scale, "bag bending stiffness scale"
        )
        self.bag_bulk_damping = _m4__validate_nonnegative(args.bag_bulk_damping, "bag bulk damping")
        super().__init__(viewer, args)
        if (
            self.model.edge_rest_angle is None
            or self.model.edge_indices is None
            or self.model.edge_bending_properties is None
            or (self.model.edge_count == 0)
        ):
            raise RuntimeError("The inflatable bag did not create bending edges.")
        edge_bending_properties = self.model.edge_bending_properties.numpy()
        edge_bending_properties[:, 0] *= self.bag_bending_stiffness_scale
        self.model.edge_bending_properties.assign(edge_bending_properties)
        pneumatic_bulk_damping = self.model.pneumatic.bulk_damping.numpy()
        pneumatic_bulk_damping[self.cavity.cavity_index] = self.bag_bulk_damping
        self.model.pneumatic.bulk_damping.assign(pneumatic_bulk_damping)
        self.authored_edge_rest_angle = wp.clone(self.model.edge_rest_angle)
        self.authored_edge_bending_properties = wp.clone(self.model.edge_bending_properties)

    def _sample_hand_trajectory(self, time_s: float):
        """Sample the v10000 source trajectory without changing its hand pose."""
        return _m4_Example._sample(self, time_s)

    def _apply_bending_plasticity(self):
        """Evolve the bag rest curvature after one completed physics substep."""
        wp.launch(
            _m4__update_bending_plasticity,
            dim=self.model.edge_count,
            inputs=[
                self.state_0.particle_q,
                self.model.edge_indices,
                self.authored_edge_rest_angle,
                self.model.edge_rest_angle,
                self.authored_edge_bending_properties,
                self.model.edge_bending_properties,
                self.bag_particle_start,
                self.bag_particle_end,
                self.plastic_yield_angle,
                self.plastic_flow_rate,
                self.plastic_max_angle,
                self.plastic_hardening,
                self.sim_dt,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance realtime-IK targets and apply plastic flow each substep."""
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _m3__interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _m3__joint_velocity,
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
                _m3__accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            if self.sim_substeps % 2 != 0 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = (self.state_1, self.state_0)
            if self.plasticity_enabled:
                self._apply_bending_plasticity()

    def _capture_simulation_graph(self):
        """Capture one frame without committing capture-time plastic flow."""
        rest_angle_backup = wp.clone(self.model.edge_rest_angle)
        bending_properties_backup = wp.clone(self.model.edge_bending_properties)
        super()._capture_simulation_graph()
        wp.copy(self.model.edge_rest_angle, rest_angle_backup)
        wp.copy(self.model.edge_bending_properties, bending_properties_backup)

    def test_final(self):
        """Verify robot tracking and bounded plastic flow."""
        super().test_final()
        plastic_offset = self.model.edge_rest_angle.numpy() - self.authored_edge_rest_angle.numpy()
        assert np.all(np.isfinite(plastic_offset))
        maximum_offset = float(np.max(np.abs(plastic_offset), initial=0.0))
        assert maximum_offset <= self.plastic_max_angle + 1e-05
        bending_stiffness = self.model.edge_bending_properties.numpy()[:, 0]
        authored_stiffness = self.authored_edge_bending_properties.numpy()[:, 0]
        assert np.all(np.isfinite(bending_stiffness))
        assert np.all(bending_stiffness >= authored_stiffness)
        assert np.all(bending_stiffness <= authored_stiffness * (1.0 + self.plastic_hardening) + 1e-05)
        if not self.plasticity_enabled:
            assert maximum_offset < 1e-07, "Disabled plasticity changed the authored rest angles."
            assert np.array_equal(bending_stiffness, authored_stiffness)
            return
        script_time = self.sim_time * self.args.trajectory_time_scale
        if script_time + self.frame_dt * self.args.trajectory_time_scale < self.script_duration:
            return
        assert maximum_offset > math.radians(0.1), "The completed grasp did not yield any bag edges."

    @staticmethod
    def create_parser():
        """Create full-W1 realtime-IK and plastic-bag arguments."""
        parser = _m3_Example.create_parser()
        parser.add_argument(
            "--plastic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable bending plasticity for the inflatable bag.",
        )
        parser.add_argument(
            "--plastic-yield-angle-deg",
            type=float,
            default=_m4_PLASTIC_YIELD_ANGLE_DEG,
            help="Dihedral-angle error that starts plastic flow [deg].",
        )
        parser.add_argument(
            "--plastic-flow-rate",
            type=float,
            default=_m4_PLASTIC_FLOW_RATE,
            help="Rest-angle adoption rate while forming a crease [1/s].",
        )
        parser.add_argument(
            "--plastic-max-angle-deg",
            type=float,
            default=_m4_PLASTIC_MAX_ANGLE_DEG,
            help="Maximum rest-angle offset from the authored bag [deg].",
        )
        parser.add_argument(
            "--plastic-hardening",
            type=float,
            default=_m4_PLASTIC_HARDENING,
            help="Maximum added bending-stiffness multiple on yielded edges.",
        )
        parser.add_argument(
            "--bag-bending-stiffness-scale",
            type=float,
            default=_m4_BAG_BENDING_STIFFNESS_SCALE,
            help="Scale applied to the elastic bag bending stiffness before yielding.",
        )
        parser.add_argument(
            "--bag-bulk-damping",
            type=float,
            default=_m4_BAG_BULK_DAMPING,
            help="Pneumatic volume-rate damping [Pa s/m^3].",
        )
        return parser


def main():
    """Run the full-W1 plastic inflatable-bag pick-and-release demo."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
