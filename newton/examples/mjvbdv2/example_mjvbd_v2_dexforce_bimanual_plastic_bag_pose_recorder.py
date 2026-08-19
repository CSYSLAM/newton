# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tune two Dexforce W1 hand poses around a hanging supermarket bag.

The original bag geometry remains supported by a fixed horizontal rod while
four rigid balls remain inside it. On startup, physical rod-to-handle contact
opens the gap between the two handle films before pose tuning stops the scene.
The left and right hands are kinematic visual references: all hand collision
flags are disabled so moving a hand cannot disturb the rod, bag, balls, or the
other hand. Rod-to-bag and ball-to-bag contacts remain enabled.

Use the two transform gizmos or the Tk sliders to adjust each floating hand
root and its ten finger joints, then save both poses to one JSON file. The
recorder starts from the last saved bimanual pose when that file is available.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_dexforce_bimanual_plastic_bag_pose_recorder --viewer gl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
BAG_MESH_PATH = (
    ASSET_ROOT / "supermarket_plastic_bag_open_carry_v5_asset" / "supermarket_plastic_bag_open_carry_v5_tri.obj"
)
LEFT_HAND_URDF = ASSET_ROOT / "W1_left_hand" / "DexforceW1_left_hand.urdf"
RIGHT_HAND_URDF = ASSET_ROOT / "W1_right_hand" / "DexforceW1_right_hand.urdf"
HAND_POSE_PATH = ASSET_ROOT / "vbd_mjvbd_v2" / "vbd_w1_bimanual_plastic_bag_pose.json"

FPS = 60
SIM_SUBSTEPS = 6
VBD_ITERATIONS = 12
AUTO_SETTLE_FRAMES = 120
EXTRA_SETTLE_FRAMES = 60

BAG_POSITION = np.array((0.0, 0.0, 0.30), dtype=np.float32)
BAG_AREAL_DENSITY = 0.02
BAG_PARTICLE_RADIUS = 0.0015
BAG_TRI_KE = 3.0e6
BAG_TRI_KA = 3.0e6
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 100.0
BAG_EDGE_KD = 3.0
AIR_DRAG_RATE = 1.0  # [1/s]

HANDLE_HOLE_CENTER_Z = 0.519
ROD_RADIUS = 0.0105
ROD_HALF_LENGTH = 0.22
ROD_CONTACT_MARGIN = 0.002
ROD_CONTACT_KE = 4.0e8
ROD_CONTACT_KD = 100.0
ROD_COLOR = (0.55, 0.58, 0.62)

BALL_RADIUS = 0.040
BALL_DENSITY = 30000000.0
BALL_INITIAL_DOWNWARD_SPEED = 2.0
BALL_CONTACT_MARGIN = 0.005
BALL_CONTACT_KE = 1.0e7
BALL_CONTACT_KD = 500.0
BALL_FRICTION = 0.3
BALL_LOCAL_POSITIONS = (
    (-0.080, 0.0, 0.22),
    (0.080, 0.0, 0.22),
    (-0.040, 0.0, 0.32),
    (0.040, 0.0, 0.32),
)
BALL_COLORS = (
    (0.92, 0.18, 0.15),
    (0.10, 0.48, 0.92),
    (0.20, 0.75, 0.30),
    (0.95, 0.65, 0.10),
)

SOFT_CONTACT_KE = 1.0e7
SOFT_CONTACT_KD = 500.0
SOFT_CONTACT_FRICTION = 0.06
SOFT_CONTACT_MARGIN = 0.010
SELF_CONTACT_MARGIN = 0.003
BAG_COLOR = (0.88, 0.035, 0.025)
BAG_OPACITY = 0.48

HAND_POSITION_LIMIT_MM = 300.0


def _load_initial_hand_pose(path: Path):
    """Load the saved root transforms and finger targets for both hands."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    hands = payload.get("hands")
    if not isinstance(hands, dict):
        raise ValueError(f"Hand pose file does not contain a hands mapping: {path}")

    result = {}
    for side in ("LEFT", "RIGHT"):
        hand = hands.get(side)
        if not isinstance(hand, dict):
            raise ValueError(f"Hand pose file does not contain {side}: {path}")
        root = hand.get("target_root_world")
        joints = hand.get("finger_joints_degrees")
        if not isinstance(root, dict) or not isinstance(joints, dict):
            raise ValueError(f"Saved {side} pose is incomplete: {path}")
        position = np.asarray(root.get("position_m"), dtype=np.float32)
        rotation = np.asarray(root.get("quaternion_xyzw"), dtype=np.float32)
        if position.shape != (3,) or rotation.shape != (4,):
            raise ValueError(f"Saved {side} root pose has an invalid shape: {path}")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"Saved {side} root pose is not finite: {path}")
        norm = float(np.linalg.norm(rotation))
        if norm < 1.0e-8:
            raise ValueError(f"Saved {side} root quaternion has zero length: {path}")
        result[side] = {
            "transform": wp.transform(wp.vec3(*position), wp.quat(*(rotation / norm))),
            "joint_degrees": {str(name): float(value) for name, value in joints.items()},
        }
    return result


@wp.kernel
def _apply_particle_drag(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    drag_rate: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional air drag to all particles."""
    particle = wp.tid()
    particle_f[particle] = particle_f[particle] - drag_rate * particle_mass[particle] * particle_qd[particle]


class Example:
    """Interactively tune collision-disabled hands around a physical bag."""

    HAND_UI_SUFFIXES = (
        "HAND_THUMB1",
        "HAND_THUMB2",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )
    HAND_UI_LABELS: ClassVar[dict[str, str]] = {
        "HAND_THUMB1": "THUMB1",
        "HAND_THUMB2": "THUMB2",
        "HAND_INDEX": "INDEX",
        "INDEX_PIP": "INDEX PIP",
        "HAND_MIDDLE": "MIDDLE",
        "MIDDLE_PIP": "MIDDLE PIP",
        "HAND_RING": "RING",
        "RING_PIP": "RING PIP",
        "HAND_PINKY": "PINKY",
        "PINKY_PIP": "PINKY PIP",
    }

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self._settle_target_frame = AUTO_SETTLE_FRAMES
        self._settle_complete = False
        self._root = None
        self._status_var = None
        self._last_target_signature: tuple[float, ...] | None = None

        bag_mesh_path = Path(args.bag_mesh).expanduser().resolve()
        left_hand_urdf = Path(args.left_hand_urdf).expanduser().resolve()
        right_hand_urdf = Path(args.right_hand_urdf).expanduser().resolve()
        initial_pose_path = Path(args.initial_pose).expanduser().resolve()
        for description, path in (
            ("Plastic bag mesh", bag_mesh_path),
            ("Left-hand URDF", left_hand_urdf),
            ("Right-hand URDF", right_hand_urdf),
            ("Initial bimanual hand pose", initial_pose_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")
        self.initial_hand_poses = _load_initial_hand_pose(initial_pose_path)

        bag_mesh = newton.Mesh.create_from_file(str(bag_mesh_path), compute_inertia=False, is_solid=False)
        vertices = np.asarray(bag_mesh.vertices, dtype=np.float32)
        indices = np.asarray(bag_mesh.indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected bag vertices with shape (n, 3), got {vertices.shape}")
        if indices.ndim != 1 or indices.size % 3 != 0:
            raise ValueError(f"Expected a flat triangle index buffer, got {indices.shape}")

        self.hand_homes = {
            side: self._copy_transform(self.initial_hand_poses[side]["transform"]) for side in ("LEFT", "RIGHT")
        }

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverMJVBDV2.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(*BAG_POSITION),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            density=BAG_AREAL_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
        )

        self.rod_position = BAG_POSITION + np.array((0.0, 0.0, HANDLE_HOLE_CENTER_Z), dtype=np.float32)
        rod_cfg = newton.ModelBuilder.ShapeConfig(
            ke=ROD_CONTACT_KE,
            kd=ROD_CONTACT_KD,
            mu=0.8,
            margin=ROD_CONTACT_MARGIN,
        )
        self.rod_shape_index = builder.add_shape_capsule(
            -1,
            xform=wp.transform(
                wp.vec3(*self.rod_position),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi),
            ),
            radius=ROD_RADIUS,
            half_height=ROD_HALF_LENGTH,
            cfg=rod_cfg,
            color=ROD_COLOR,
            label="handle support rod",
        )

        ball_cfg = newton.ModelBuilder.ShapeConfig(
            density=BALL_DENSITY,
            ke=BALL_CONTACT_KE,
            kd=BALL_CONTACT_KD,
            mu=BALL_FRICTION,
            margin=BALL_CONTACT_MARGIN,
        )
        self.ball_bodies = []
        for ball_index, (local_position, color) in enumerate(zip(BALL_LOCAL_POSITIONS, BALL_COLORS, strict=True)):
            position = BAG_POSITION + np.asarray(local_position, dtype=np.float32)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                label=f"bag ball {ball_index}",
            )
            builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -BALL_INITIAL_DOWNWARD_SPEED, 0.0, 0.0, 0.0)
            builder.add_shape_sphere(
                body,
                radius=BALL_RADIUS,
                cfg=ball_cfg,
                color=color,
                label=f"bag ball {ball_index} shape",
            )
            self.ball_bodies.append(body)

        self.hand_articulations: list[int] = []
        self.hand_shape_indices: list[int] = []
        collide_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for side, hand_path in (("LEFT", left_hand_urdf), ("RIGHT", right_hand_urdf)):
            articulation_start = builder.articulation_count
            body_start = builder.body_count
            shape_start = builder.shape_count
            builder.add_urdf(
                str(hand_path),
                xform=self.hand_homes[side],
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                parse_visuals_as_colliders=False,
                force_show_colliders=False,
            )
            self.hand_articulations.extend(range(articulation_start, builder.articulation_count))
            for body in range(body_start, builder.body_count):
                builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            for shape in range(shape_start, builder.shape_count):
                builder.shape_flags[shape] &= ~collide_mask
                self.hand_shape_indices.append(shape)

        if len(self.hand_articulations) != 2:
            raise RuntimeError("Expected one articulation from each standalone hand URDF")
        builder.add_ground_plane()
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_FRICTION
        self.device = self.model.device

        joint_q = self.model.joint_q.numpy()
        joint_q_starts = self.model.joint_q_start.numpy()
        for side in ("LEFT", "RIGHT"):
            for suffix, degrees in self.initial_hand_poses[side]["joint_degrees"].items():
                label = f"{side}_{suffix}"
                joint = next(index for index, name in enumerate(self.model.joint_label) if name.endswith("/" + label))
                joint_q[int(joint_q_starts[joint])] = np.radians(degrees)
        self.model.joint_q.assign(joint_q)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.root_joint_indices = self._find_root_joints()
        self._ui_controls: dict[str, dict[str, Any]] = {}
        self._manual_target_q = wp.clone(self.model.joint_q)
        self._zero_joint_qd = wp.zeros_like(self.model.joint_qd)
        self._create_hand_controls()
        self._refresh_manual_target()
        self._apply_preview_pose()
        self.state_1.assign(self.state_0)

        self.solver = newton.solvers.SolverMJVBDV2(
            self.model,
            mujoco_articulations=tuple(self.hand_articulations),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "friction_epsilon": 1.0e-4,
                "rigid_body_contact_buffer_size": 2048,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": 48,
                "particle_edge_contact_buffer_size": 96,
                "particle_collision_detection_interval": -1,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": SELF_CONTACT_MARGIN,
                "rigid_body_particle_contact_buffer_size": 1024,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "include_static_kinematic_pairs": False,
            },
        )

        self.render_indices = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.72, -1.05, 0.66), pitch=-7.0, yaw=124.0)

    def _find_root_joints(self):
        joint_types = self.model.joint_type.numpy()
        joint_parents = self.model.joint_parent.numpy()
        joint_children = self.model.joint_child.numpy()
        roots = {}
        for side in ("LEFT", "RIGHT"):
            suffix = f"/{side.lower()}_hand_base"
            roots[side] = next(
                joint
                for joint, (joint_type, parent, child) in enumerate(
                    zip(joint_types, joint_parents, joint_children, strict=True)
                )
                if int(joint_type) == int(newton.JointType.FREE)
                and int(parent) == -1
                and self.model.body_label[int(child)].endswith(suffix)
            )
        return roots

    def _create_hand_controls(self):
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        asset_q = self.model.joint_q.numpy()
        for side in ("LEFT", "RIGHT"):
            joint_indices = {}
            joint_degrees = {}
            joint_limits = {}
            for suffix in self.HAND_UI_SUFFIXES:
                label = f"{side}_{suffix}"
                joint = next(index for index, name in enumerate(self.model.joint_label) if name.endswith("/" + label))
                q_index = int(q_starts[joint])
                qd_index = int(qd_starts[joint])
                minimum, maximum = sorted((float(np.degrees(lower[qd_index])), float(np.degrees(upper[qd_index]))))
                joint_indices[suffix] = q_index
                joint_degrees[suffix] = float(np.clip(np.degrees(asset_q[q_index]), minimum, maximum))
                joint_limits[suffix] = (minimum, maximum)
            self._ui_controls[side] = {
                "root_q_start": int(q_starts[self.root_joint_indices[side]]),
                "base_transform": self._copy_transform(self.hand_homes[side]),
                "gizmo_transform": self._copy_transform(self.hand_homes[side]),
                "target_transform": self._copy_transform(self.hand_homes[side]),
                "joint_indices": joint_indices,
                "joint_degrees": joint_degrees,
                "joint_limits": joint_limits,
                "position_mm": np.zeros(3, dtype=np.float32),
                "rotation_deg": np.zeros(3, dtype=np.float32),
            }

    @staticmethod
    def _copy_transform(transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

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

    def _offset_transform(self, control):
        base = control["gizmo_transform"]
        position = np.asarray(wp.transform_get_translation(base), dtype=np.float32)
        position += control["position_mm"] * 1.0e-3
        rx, ry, rz = np.radians(control["rotation_deg"])
        offset = self._quat_mul(
            self._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        rotation = self._quat_mul(offset, wp.transform_get_rotation(base))
        return wp.transform(wp.vec3(*position), wp.normalize(rotation))

    def _refresh_manual_target(self):
        target_q = self._manual_target_q.numpy()
        for control in self._ui_controls.values():
            target = self._offset_transform(control)
            control["target_transform"] = target
            position = wp.transform_get_translation(target)
            rotation = wp.transform_get_rotation(target)
            root = control["root_q_start"]
            target_q[root : root + 7] = (*position, *rotation)
            for suffix, index in control["joint_indices"].items():
                target_q[index] = np.radians(control["joint_degrees"][suffix])
        self._manual_target_q.assign(target_q)
        self._last_target_signature = self._target_signature()

    def _apply_preview_pose(self):
        self.state_0.joint_q.assign(self._manual_target_q)
        wp.copy(self.state_0.joint_qd, self._zero_joint_qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def _target_signature(self):
        values = []
        for control in self._ui_controls.values():
            values.extend(float(value) for value in wp.transform_get_translation(control["gizmo_transform"]))
            values.extend(float(value) for value in wp.transform_get_rotation(control["gizmo_transform"]))
            values.extend(float(value) for value in control["position_mm"])
            values.extend(float(value) for value in control["rotation_deg"])
            values.extend(float(control["joint_degrees"][suffix]) for suffix in self.HAND_UI_SUFFIXES)
        return tuple(values)

    def step_once(self):
        """Advance the bag and balls once while the hands remain visual-only."""
        for _ in range(SIM_SUBSTEPS):
            self.state_0.joint_q.assign(self._manual_target_q)
            wp.copy(self.state_0.joint_qd, self._zero_joint_qd)
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _apply_particle_drag,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.model.particle_mass, AIR_DRAG_RATE],
                outputs=[self.state_0.particle_f],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._apply_preview_pose()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _reset_physics(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._manual_target_q = wp.clone(self.model.joint_q)
        self._zero_joint_qd = wp.zeros_like(self.model.joint_qd)
        for control in self._ui_controls.values():
            control["gizmo_transform"] = self._copy_transform(control["base_transform"])
            control["position_mm"].fill(0.0)
            control["rotation_deg"].fill(0.0)
            for suffix, index in control["joint_indices"].items():
                control["joint_degrees"][suffix] = float(np.degrees(self.model.joint_q.numpy()[index]))
        self._refresh_manual_target()
        self._apply_preview_pose()
        self.state_1.assign(self.state_0)
        self.sim_time = 0.0
        self.frame_index = 0
        self._settle_target_frame = AUTO_SETTLE_FRAMES
        self._settle_complete = False

    @staticmethod
    def _transform_dict(transform):
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return {
            "position_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in rotation],
        }

    def _capture_pose(self):
        hands = {}
        for side, control in self._ui_controls.items():
            hands[side] = {
                "base_root_world": self._transform_dict(control["base_transform"]),
                "gizmo_root_world": self._transform_dict(control["gizmo_transform"]),
                "position_offset_mm": [float(value) for value in control["position_mm"]],
                "rotation_offset_deg": [float(value) for value in control["rotation_deg"]],
                "target_root_world": self._transform_dict(control["target_transform"]),
                "finger_joints_degrees": {
                    suffix: float(control["joint_degrees"][suffix]) for suffix in self.HAND_UI_SUFFIXES
                },
            }
        return {
            "format": "newton_mjvbd_v2_bimanual_plastic_bag_pose_v1",
            "hand_collisions_enabled": False,
            "rod_handle_collision_enabled": True,
            "rod_radius_m": ROD_RADIUS,
            "settled_physics_frame": self.frame_index,
            "hands": hands,
        }

    @staticmethod
    def _write_json(path_value: str, payload: dict[str, Any]):
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_pose(self):
        """Save both requested hand poses to one JSON file."""
        path = self._write_json(self.args.pose_output, self._capture_pose())
        self._set_status(f"Saved bimanual pose: {path}")

    def render(self):
        """Render the physical bag scene and both interactive hand gizmos."""
        if self._target_signature() != self._last_target_signature:
            self._refresh_manual_target()
            self._apply_preview_pose()
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            for side, control in self._ui_controls.items():
                self.viewer.log_gizmo(f"{side.lower()}_hand_root", control["gizmo_transform"])
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/plastic_bag",
            self.state_0.particle_q,
            self.render_indices,
            backface_culling=False,
            color=BAG_COLOR,
            roughness=0.65,
            metallic=0.0,
            opacity=BAG_OPACITY,
        )
        self.viewer.end_frame()

    def _set_status(self, message: str):
        if self._status_var is not None:
            self._status_var.set(message)

    def _on_control_changed(self, side: str, variables):
        control = self._ui_controls[side]
        for suffix, variable in variables["joints"].items():
            control["joint_degrees"][suffix] = float(variable.get())
        control["position_mm"] = np.asarray(
            [float(variable.get()) for variable in variables["position"]], dtype=np.float32
        )
        control["rotation_deg"] = np.asarray(
            [float(variable.get()) for variable in variables["rotation"]], dtype=np.float32
        )
        self._refresh_manual_target()
        self._apply_preview_pose()
        self.render()

    def _make_scale(self, parent, row, label, variable, minimum, maximum, command):
        import tkinter as tk  # noqa: PLC0415

        self._ttk.Label(parent, text=label, width=20).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        self._ttk.Label(parent, textvariable=variable._display_var, width=8, anchor="e").grid(
            row=row, column=1, sticky="e", padx=3
        )
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=1.0,
            orient="horizontal",
            showvalue=False,
            length=435,
            highlightthickness=0,
            command=command,
        )
        scale.grid(row=row, column=2, sticky="ew", padx=4)

        def update_display(*_):
            variable._display_var.set(f"{float(variable.get()):.1f}")

        variable.trace_add("write", update_display)
        update_display()

    def _build_hand_tab(self, notebook, side):
        import tkinter as tk  # noqa: PLC0415

        control = self._ui_controls[side]
        tab = self._ttk.Frame(notebook, padding=8)
        tab.columnconfigure(2, weight=1)
        variables = {"joints": {}, "position": [], "rotation": []}

        joints = self._ttk.LabelFrame(tab, text="Finger joint angles (degrees)", padding=5)
        joints.grid(row=0, column=0, columnspan=3, sticky="nsew")
        joints.columnconfigure(2, weight=1)
        for row, suffix in enumerate(self.HAND_UI_SUFFIXES):
            variable = tk.DoubleVar(value=control["joint_degrees"][suffix])
            variable._display_var = tk.StringVar()
            variables["joints"][suffix] = variable
            minimum, maximum = control["joint_limits"][suffix]
            self._make_scale(
                joints,
                row,
                self.HAND_UI_LABELS[suffix],
                variable,
                minimum,
                maximum,
                lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )

        root_box = self._ttk.LabelFrame(tab, text="Floating root offset relative to gizmo", padding=5)
        root_box.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        root_box.columnconfigure(2, weight=1)
        for index, label in enumerate(("Position X (mm)", "Position Y (mm)", "Position Z (mm)")):
            variable = tk.DoubleVar(value=float(control["position_mm"][index]))
            variable._display_var = tk.StringVar()
            variables["position"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -HAND_POSITION_LIMIT_MM,
                HAND_POSITION_LIMIT_MM,
                lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )
        for index, label in enumerate(("Rotation X (deg)", "Rotation Y (deg)", "Rotation Z (deg)"), start=3):
            variable = tk.DoubleVar(value=float(control["rotation_deg"][index - 3]))
            variable._display_var = tk.StringVar()
            variables["rotation"].append(variable)
            self._make_scale(
                root_box,
                index,
                label,
                variable,
                -180.0,
                180.0,
                lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )
        return tab

    def run_pose_recorder(self):
        """Run the Tk pose recorder while servicing the Newton viewer."""
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
        root.title("MJVBD-v2 bimanual plastic-bag pose recorder")
        root.geometry("700x950")
        root.minsize(650, 780)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        notebook.add(self._build_hand_tab(notebook, "LEFT"), text="LEFT")
        notebook.add(self._build_hand_tab(notebook, "RIGHT"), text="RIGHT")

        buttons = ttk.Frame(root, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Settle 60 more frames", command=self._settle_more_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset scene", command=self._reset_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save both poses JSON", command=self.save_pose).pack(side="left", padx=3)
        self._status_var = tk.StringVar(
            value=f"Rod-handle collision is ON; opening the handle gap for {AUTO_SETTLE_FRAMES} frames."
        )
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))

        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            if self.frame_index < self._settle_target_frame:
                self.step_once()
                if self.frame_index % 15 == 0:
                    self._set_status(
                        f"Rod contact is opening the handle gap: {self.frame_index}/{self._settle_target_frame}."
                    )
            elif not self._settle_complete:
                self._settle_complete = True
                self._set_status(
                    "Handle gap settled and physics paused; adjust both collision-disabled hands, then save."
                )
            self.render()
            root.after(max(1, int(1000.0 / FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _settle_more_from_ui(self):
        self._settle_target_frame = max(self._settle_target_frame, self.frame_index) + EXTRA_SETTLE_FRAMES
        self._settle_complete = False
        self._set_status(f"Continuing rod-handle settling through frame {self._settle_target_frame}.")

    def _reset_from_ui(self):
        self._reset_physics()
        self._set_status("Reset bag, balls, and both hand poses.")
        self.render()

    def test_final(self):
        """Verify finite state and collision-disabled hand shapes."""
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        shape_flags = self.model.shape_flags.numpy()[self.hand_shape_indices]
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        assert np.all((shape_flags & collision_mask) == 0)
        rod_flags = int(self.model.shape_flags.numpy()[self.rod_shape_index])
        assert rod_flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)

    @staticmethod
    def create_parser():
        """Create command-line options for the bimanual pose recorder."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        parser.add_argument("--bag-mesh", type=Path, default=BAG_MESH_PATH)
        parser.add_argument("--left-hand-urdf", type=Path, default=LEFT_HAND_URDF)
        parser.add_argument("--right-hand-urdf", type=Path, default=RIGHT_HAND_URDF)
        parser.add_argument(
            "--initial-pose",
            type=Path,
            default=HAND_POSE_PATH,
            help="Saved bimanual pose used to initialize and reset both hands.",
        )
        parser.add_argument(
            "--pose-output",
            default=str(HAND_POSE_PATH),
            help="Path for the JSON file written by Save both poses JSON.",
        )
        parser.add_argument(
            "--recorder-no-gui",
            action="store_true",
            help="Build and render one frame without opening the Tk controls.",
        )
        return parser


def main():
    """Launch the collision-disabled bimanual plastic-bag pose recorder."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_pose_recorder()


if __name__ == "__main__":
    main()
