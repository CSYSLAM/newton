# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactively tune a W1 right-hand grasp of a sealed inflatable bag.

The scene combines the floating kinematic hand controls from the right-hand
rigid-cube recorder with the closed pneumatic chip-bag model from
``vbd_inflatable_bag_v0``. The hand and bag interact only through physical
rigid-soft contact; the bag is never attached to the hand.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_inflatable_bag_recorder --viewer gl

The default ``target-volume`` mode makes the bag approximately
volume-preserving during grasping. Use ``--pneumatic-mode isothermal`` for the
compressible ideal-gas response.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_right_hand_rigid_cube_recorder as hand_recorder
from newton.solvers import PneumaticConfig, PneumaticMode, SolverMJVBDV2

FPS = 60
SIM_SUBSTEPS = 12
VBD_ITERATIONS = 40
MAX_FINGER_SPEED_DEG_S = 90.0
MAX_FINGER_CONTACT_SPEED_DEG_S = 30.0

BAG_SCALE = 0.36
BAG_DENSITY = 0.12
BAG_REFERENCE_ABSOLUTE_PRESSURE = 125_000.0
BAG_AMBIENT_PRESSURE = 101_325.0
BAG_MAX_ABSOLUTE_PRESSURE = 200_000.0
BAG_BULK_DAMPING = 50.0
BAG_TARGET_VOLUME_RATIO = 1.01
BAG_PARTICLE_RADIUS = 0.002
BAG_TRI_KE = 1.0e5
BAG_TRI_KA = 1.0e5
BAG_TRI_KD = 80.0
BAG_EDGE_KE = 20.0
BAG_EDGE_KD = 0.5

BAG_REST_BULGE = 0.020
BAG_SOURCE_HALF_WIDTH = 0.133928
BAG_SOURCE_HALF_LENGTH = 0.180
BAG_SOURCE_FACE_HALF_THICKNESS = 0.052
# Local Y becomes world Z after rotation. Include the authored rest bulge so
# the lower membrane remains one particle radius above the table.
BAG_SOURCE_HALF_THICKNESS = BAG_SOURCE_FACE_HALF_THICKNESS + BAG_REST_BULGE
BAG_CENTER = wp.vec3(
    float(hand_recorder.CUBE_CENTRE[0]),
    float(hand_recorder.CUBE_CENTRE[1]),
    hand_recorder.TABLE_TOP_Z + BAG_SCALE * BAG_SOURCE_HALF_THICKNESS + BAG_PARTICLE_RADIUS,
)
BAG_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5) * wp.quat_from_axis_angle(
    wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5
)

SHAPE_CONTACT_MARGIN = 0.0
SOFT_CONTACT_MARGIN = 0.0
CONTACT_KE = 2.0e5
CONTACT_KD = 100.0
CONTACT_MU = 2.0
RIGID_BODY_CONTACT_BUFFER_SIZE = 4096
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 8192

PNEUMATIC_MODES = {
    "isothermal": PneumaticMode.ISOTHERMAL,
    "target-volume": PneumaticMode.TARGET_VOLUME,
}

INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.835815668106079, 1.3647105693817139),
    wp.quat(0.03812963888049126, 0.9212844967842102, -0.3854166567325592, 0.03514265641570091),
)
INITIAL_HAND_JOINTS = {
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
class _ChipBagMesh:
    """Store simulation triangles and rendering edges for the authored bag."""

    vertices: list[list[float]]
    indices: list[int]
    edges: list[tuple[int, int]]


def _scaled_mesh_volume(mesh: _ChipBagMesh, scale: float) -> float:
    """Compute the closed mesh volume after uniform scaling [m^3]."""

    positions = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    anchor = positions[triangles[0, 0]]
    relative = positions[triangles] - anchor
    signed_volume = (
        np.einsum(
            "ij,ij->i",
            relative[:, 0],
            np.cross(relative[:, 1], relative[:, 2]),
        ).sum()
        / 6.0
    )
    volume = abs(float(signed_volume)) * scale**3
    if volume <= 0.0:
        raise ValueError("The inflatable-bag mesh must enclose a positive volume.")
    return volume


def _make_pneumatic_config(mode_name: str, rest_volume: float) -> PneumaticConfig:
    """Create a pressure law with matching initial gauge pressure."""

    mode = PNEUMATIC_MODES[mode_name]
    target_volume = None
    volume_stiffness = 0.0
    if mode == PneumaticMode.TARGET_VOLUME:
        target_volume = BAG_TARGET_VOLUME_RATIO * rest_volume
        target_volume_delta = target_volume - rest_volume
        initial_gauge_pressure = BAG_REFERENCE_ABSOLUTE_PRESSURE - BAG_AMBIENT_PRESSURE
        volume_stiffness = initial_gauge_pressure / target_volume_delta

    return PneumaticConfig(
        mode=mode,
        reference_absolute_pressure=BAG_REFERENCE_ABSOLUTE_PRESSURE,
        ambient_pressure=BAG_AMBIENT_PRESSURE,
        target_volume=target_volume,
        volume_stiffness=volume_stiffness,
        bulk_damping=BAG_BULK_DAMPING,
        max_absolute_pressure=BAG_MAX_ABSOLUTE_PRESSURE,
    )


def _load_chip_bag_mesh() -> _ChipBagMesh:
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
        tuple(sorted((vertex0, vertex1)))
        for triangle in triangles
        for vertex0, vertex1 in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
    )
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(f"{path} must be a closed two-manifold surface after triangulation.")

    for vertex in vertices:
        x, y, z = vertex
        if abs(y) < 1.0e-8:
            continue
        width_phase = min(1.0, abs(x) / BAG_SOURCE_HALF_WIDTH)
        length_phase = min(1.0, abs(z) / BAG_SOURCE_HALF_LENGTH)
        width_profile = float(np.cos(0.5 * np.pi * width_phase) ** 2)
        length_profile = float(np.cos(0.5 * np.pi * length_phase) ** 2)
        face_weight = min(1.0, abs(y) / BAG_SOURCE_FACE_HALF_THICKNESS)
        vertex[1] += np.sign(y) * BAG_REST_BULGE * width_profile * length_profile * face_weight

    return _ChipBagMesh(
        vertices=vertices,
        indices=[vertex for triangle in triangles for vertex in triangle],
        edges=sorted(edge_counts),
    )


@wp.kernel
def _gather_edges(
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
def _limit_finger_target_step(
    current_q: wp.array[float],
    finger_q_indices: wp.array[int],
    max_step: float,
    target_q: wp.array[float],
):
    finger = wp.tid()
    q_index = finger_q_indices[finger]
    delta = wp.clamp(target_q[q_index] - current_q[q_index], -max_step, max_step)
    target_q[q_index] = current_q[q_index] + delta


class Example(hand_recorder.Example):
    """Record physical right-hand keyframes for an inflatable-bag grasp."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.frame_index = 0
        self._root = None
        self._status_var = None
        self._trajectory_frames: list[dict[str, Any]] = []
        self._last_target_signature: tuple[float, ...] | None = None
        # Keep the recorded first pose fixed as later recordings replace the
        # last-keyframe output file.
        self._initial_keyframe = None

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": False,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )

        self.root_joint = self._root_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint])
        self.hand_joint_indices = self._hand_joint_indices()
        self.hand_joint_q_indices = wp.array(
            tuple(self.hand_joint_indices.values()),
            dtype=int,
            device=self.model.device,
        )
        self.max_finger_step = float(np.radians(MAX_FINGER_SPEED_DEG_S) * self.frame_dt)
        self.max_finger_contact_step = float(np.radians(MAX_FINGER_CONTACT_SPEED_DEG_S) * self.frame_dt)
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(INITIAL_HAND_ROOT)
        self.position_mm = np.zeros(3, dtype=np.float32)
        self.rotation_deg = np.zeros(3, dtype=np.float32)
        self.joint_degrees = dict(INITIAL_HAND_JOINTS)
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
            self.viewer.set_camera(
                hand_recorder.CAMERA_POS,
                hand_recorder.CAMERA_PITCH,
                hand_recorder.CAMERA_YAW,
            )
        self._store_trajectory_frame()

    def _build_scene(self):
        """Build the kinematic hand, support table, and sealed bag."""

        if not hand_recorder.RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {hand_recorder.RIGHT_HAND_URDF}")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CONTACT_MU
        builder.default_shape_cfg.margin = SHAPE_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(hand_recorder.RIGHT_HAND_URDF),
            xform=hand_recorder.HAND_HOME,
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
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=0.9,
            margin=SHAPE_CONTACT_MARGIN,
            is_visible=True,
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
            label="inflatable_bag_recorder_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="inflatable_bag_recorder_ground")

        bag_mesh = _load_chip_bag_mesh()
        bag_rest_volume = _scaled_mesh_volume(bag_mesh, BAG_SCALE)
        self.pneumatic_mode_name = self.args.pneumatic_mode
        self.pneumatic_config = _make_pneumatic_config(self.pneumatic_mode_name, bag_rest_volume)
        self.bag_particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=BAG_CENTER,
            rot=BAG_ROTATION,
            scale=BAG_SCALE,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=BAG_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
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
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KD
        self.model.soft_contact_mu = CONTACT_MU
        self.bag_triangle_indices = wp.array(bag_triangle_indices, dtype=int, device=self.model.device)
        self.bag_edges = wp.array(bag_edges.reshape(-1), dtype=int, device=self.model.device)
        self.bag_edge_starts = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_edge_ends = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)

    def step_once(self):
        """Advance one frame while limiting finger motion through soft contact."""

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.manual_target_q)
        hand_bag_contacts, _ = self._contact_counts()
        max_finger_step = self.max_finger_contact_step if hand_bag_contacts else self.max_finger_step
        wp.launch(
            _limit_finger_target_step,
            dim=self.hand_joint_q_indices.shape[0],
            inputs=[
                self.frame_q_start,
                self.hand_joint_q_indices,
                max_finger_step,
                self.frame_q_end,
            ],
            device=self.device,
        )
        for substep in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / SIM_SUBSTEPS
            wp.launch(
                hand_recorder._interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                hand_recorder._joint_velocity,
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
        """Reset the hand, bag, pressure state, and recorded keyframes."""

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.manual_target_q = wp.clone(self.model.joint_q)
        self.gizmo_transform = self._copy_transform(INITIAL_HAND_ROOT)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(INITIAL_HAND_JOINTS)
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
        return int(hand_bag), total

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
                "gauge_pressure_pa": absolute_pressure - BAG_AMBIENT_PRESSURE,
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
            _gather_edges,
            dim=len(self.bag_edge_starts),
            inputs=[self.state_0.particle_q, self.bag_edges, 1.0e-4],
            outputs=[self.bag_edge_starts, self.bag_edge_ends],
            device=self.model.device,
        )
        self.viewer.log_lines(
            "/inflatable_bag/grid",
            self.bag_edge_starts,
            self.bag_edge_ends,
            (0.08, 0.06, 0.02),
        )
        self.viewer.end_frame()

    def run_recorder(self):
        """Run the interactive Tk controls alongside the viewer."""

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
            value=(
                f"Pneumatic mode: {self.pneumatic_mode_name}. "
                "Finger targets move gradually; wait for the physical hand to settle before recording."
            )
        )
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
        """Record and persist one physical grasp keyframe."""

        self._store_trajectory_frame()
        keyframe_path = self._write_json(
            self.args.keyframe_output,
            {
                "format": "newton_w1_right_hand_inflatable_bag_keyframe_v1",
                "keyframe": self._trajectory_frames[-1],
            },
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
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; "
            f"hand-bag contacts: {hand_bag_contacts}, total soft contacts: {total_soft_contacts}. "
            f"Saved: {keyframe_path}, {trajectory_path}"
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
        assert self._trajectory_frames[0]["target_root_pose"] == self._transform_dict(INITIAL_HAND_ROOT)

    @staticmethod
    def create_parser():
        """Create command-line arguments for the bag-grasp recorder."""

        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        output_dir = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2"
        parser.add_argument(
            "--pose-output",
            default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_pose.json"),
        )
        parser.add_argument(
            "--trajectory-output",
            default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_trajectory.json"),
        )
        parser.add_argument(
            "--keyframe-output",
            default=str(output_dir / "vbd_w1_right_hand_inflatable_bag_last_keyframe.json"),
        )
        parser.add_argument(
            "--pneumatic-mode",
            choices=tuple(PNEUMATIC_MODES),
            default="target-volume",
            help="Pressure law for the sealed bag (default: %(default)s).",
        )
        parser.add_argument("--recorder-no-gui", action="store_true")
        return parser


def main():
    """Launch the interactive right-hand inflatable-bag recorder."""

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
