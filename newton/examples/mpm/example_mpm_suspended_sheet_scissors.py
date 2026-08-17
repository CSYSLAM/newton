# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cut a suspended volumetric MPM sheet with animated scissors.

The sheet is two particles thick and fixed along two opposite edges. A pair of
kinematic blades first presses against the sheet, then a narrow band of
particles behind the closing blades is deactivated to create a persistent cut.
This combines physical blade contact with an explicit MPM separation rule,
because standard MPM has no topological edge connectivity to sever.

Run from the repository root::

    uv run --extra examples -m newton.examples mpm_suspended_sheet_scissors
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM

SHEET_COLOR = wp.vec3(0.12, 0.42, 0.85)
ANCHOR_COLOR = wp.vec3(1.0, 0.48, 0.08)
BLADE_COLOR = wp.vec3(0.72, 0.76, 0.82)
BLADE_EDGE_COLOR = wp.vec3(0.88, 0.91, 0.95)
HANDLE_COLOR = wp.vec3(0.86, 0.10, 0.12)
HINGE_COLOR = wp.vec3(0.95, 0.66, 0.12)

BLADE_LENGTH = 0.18
BLADE_HALF_THICKNESS = 0.00125
BLADE_EDGE_HALF_HEIGHT = 0.0008
BLADE_ROOT_HEIGHT = 0.018
BLADE_TIP_HEIGHT = 0.0025
BLADE_CURVE_HEIGHT = 0.006
BLADE_SEGMENTS = 10
BLADE_Y_OFFSET = 0.00135
HANDLE_LENGTH = 0.105
HANDLE_HALF_WIDTH = 0.008
HANDLE_HALF_THICKNESS = 0.007
HANDLE_Y_OFFSET = 0.012
HANDLE_RING_MAJOR_RADIUS = 0.023
HANDLE_RING_MINOR_RADIUS = 0.0045


@wp.kernel
def set_kinematic_body_pose(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_index: int,
    xform: wp.transform,
    linear_velocity: wp.vec3,
    angular_velocity: wp.vec3,
):
    """Set one kinematic body's pose and twist."""
    body_q[body_index] = xform
    body_qd[body_index] = wp.spatial_vector(linear_velocity, angular_velocity)


def create_torus_mesh(major_radius: float, minor_radius: float, major_segments: int = 24) -> newton.Mesh:
    """Create a torus whose symmetry axis is the local y-axis."""
    minor_segments = 8
    vertices = np.empty((major_segments * minor_segments, 3), dtype=np.float32)
    for major_index in range(major_segments):
        major_angle = 2.0 * math.pi * major_index / major_segments
        major_cos = math.cos(major_angle)
        major_sin = math.sin(major_angle)
        for minor_index in range(minor_segments):
            minor_angle = 2.0 * math.pi * minor_index / minor_segments
            tube_radius = major_radius + minor_radius * math.cos(minor_angle)
            vertex_index = major_index * minor_segments + minor_index
            vertices[vertex_index] = (
                tube_radius * major_cos,
                minor_radius * math.sin(minor_angle),
                tube_radius * major_sin,
            )

    indices = []
    for major_index in range(major_segments):
        next_major = (major_index + 1) % major_segments
        for minor_index in range(minor_segments):
            next_minor = (minor_index + 1) % minor_segments
            a = major_index * minor_segments + minor_index
            b = next_major * minor_segments + minor_index
            c = next_major * minor_segments + next_minor
            d = major_index * minor_segments + next_minor
            indices.extend((a, b, c, a, c, d))

    return newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)


def blade_edge_height(x: float, side: float) -> float:
    """Return the curved cutting-edge height in blade-local coordinates."""
    phase = math.pi * np.clip(x / BLADE_LENGTH, 0.0, 1.0)
    return side * BLADE_CURVE_HEIGHT * math.sin(phase)


def create_curved_blade_mesh(side: float) -> newton.Mesh:
    """Create a thin blade with a curved cutting edge and tapered spine."""
    sample_count = BLADE_SEGMENTS + 1
    vertices = np.empty((sample_count * 4, 3), dtype=np.float32)

    def vertex_index(sample: int, outer: int, y_side: int) -> int:
        return 4 * sample + 2 * outer + y_side

    for sample in range(sample_count):
        x = BLADE_LENGTH * sample / BLADE_SEGMENTS
        alpha = sample / BLADE_SEGMENTS
        inner_z = blade_edge_height(x, side)
        blade_height = BLADE_TIP_HEIGHT + (BLADE_ROOT_HEIGHT - BLADE_TIP_HEIGHT) * (1.0 - alpha) ** 0.65
        outer_z = inner_z - side * blade_height
        for y_side, y in enumerate((-BLADE_HALF_THICKNESS, BLADE_HALF_THICKNESS)):
            vertices[vertex_index(sample, 0, y_side)] = (x, y, inner_z)
            vertices[vertex_index(sample, 1, y_side)] = (x, y, outer_z)

    indices: list[int] = []

    def add_quad(a: int, b: int, c: int, d: int) -> None:
        indices.extend((a, b, c, a, c, d))

    for sample in range(BLADE_SEGMENTS):
        next_sample = sample + 1
        add_quad(
            vertex_index(sample, 0, 0),
            vertex_index(next_sample, 0, 0),
            vertex_index(next_sample, 1, 0),
            vertex_index(sample, 1, 0),
        )
        add_quad(
            vertex_index(sample, 1, 1),
            vertex_index(next_sample, 1, 1),
            vertex_index(next_sample, 0, 1),
            vertex_index(sample, 0, 1),
        )
        add_quad(
            vertex_index(sample, 0, 1),
            vertex_index(next_sample, 0, 1),
            vertex_index(next_sample, 0, 0),
            vertex_index(sample, 0, 0),
        )
        add_quad(
            vertex_index(sample, 1, 0),
            vertex_index(next_sample, 1, 0),
            vertex_index(next_sample, 1, 1),
            vertex_index(sample, 1, 1),
        )

    add_quad(
        vertex_index(0, 0, 1),
        vertex_index(0, 0, 0),
        vertex_index(0, 1, 0),
        vertex_index(0, 1, 1),
    )
    tip = BLADE_SEGMENTS
    add_quad(
        vertex_index(tip, 0, 0),
        vertex_index(tip, 0, 1),
        vertex_index(tip, 1, 1),
        vertex_index(tip, 1, 0),
    )

    return newton.Mesh(vertices=vertices, indices=np.asarray(indices, dtype=np.int32), compute_inertia=False)


class Example:
    """Two-edge suspended MPM sheet progressively cut by kinematic scissors."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = float(args.fps)
        self.sim_substeps = int(args.substeps)
        if self.fps <= 0.0 or self.sim_substeps <= 0:
            raise ValueError("FPS and substeps must be positive")
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.sheet_width = float(args.sheet_width)
        self.sheet_length = float(args.sheet_length)
        self.anchor_height = float(args.anchor_height)
        self.cut_y = 0.5 * self.sheet_length
        self.cut_z = self.anchor_height - float(args.cut_depth)
        self.cut_half_width = float(args.cut_half_width)
        self.settle_time = float(args.settle_time)
        self.approach_time = float(args.approach_time)
        self.cut_time = float(args.cut_time)
        self.depart_time = float(args.depart_time)
        self.snip_count = int(args.snips)
        self.trajectory_curve_offset = float(args.trajectory_curve_offset)
        self.open_angle = math.radians(float(args.open_angle))
        self.closed_angle = math.radians(float(args.closed_angle))
        if min(self.settle_time, self.approach_time, self.cut_time, self.depart_time) < 0.0:
            raise ValueError("motion durations must be non-negative")
        if self.cut_time <= 0.0 or self.snip_count <= 0:
            raise ValueError("cut time and snip count must be positive")
        if not 0.0 <= self.closed_angle < self.open_angle:
            raise ValueError("expected 0 <= closed angle < open angle")
        if self.cut_half_width <= 0.0:
            raise ValueError("cut half-width must be positive")
        if abs(self.trajectory_curve_offset) + self.cut_half_width >= 0.5 * self.sheet_length:
            raise ValueError("trajectory curve must remain inside the sheet")

        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        particle_data = self._emit_sheet(builder, args)
        self.sheet_spacing = particle_data["spacing"]
        self.rest_positions_np = particle_data["positions"]
        self.fixed_indices_np = particle_data["fixed_indices"]
        self.center_indices_np = particle_data["center_indices"]
        self._add_scissors(builder)

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity(args.gravity)

        self.fixed_indices = wp.array(self.fixed_indices_np, dtype=wp.int32, device=self.model.device)
        self.model.particle_mass[self.fixed_indices].fill_(0.0)

        material = self.model.mpm
        material.young_modulus.fill_(float(args.young_modulus))
        material.poisson_ratio.fill_(float(args.poisson_ratio))
        material.damping.fill_(float(args.damping))
        material.yield_pressure.fill_(1.0e15)
        material.tensile_yield_ratio.fill_(1.0)
        material.yield_stress.fill_(0.0)
        material.hardening.fill_(0.0)
        material.dilatancy.fill_(0.0)
        material.viscosity.fill_(0.0)

        self.state_0 = self.model.state()
        self.state_0.mpm.particle_Jp.fill_(1.0)

        initial_positions = self.state_0.particle_q.numpy()
        self.fixed_positions_initial = initial_positions[self.fixed_indices_np].copy()
        self.center_height_initial = float(np.mean(initial_positions[self.center_indices_np, 2]))

        config = SolverImplicitMPM.Config()
        for key, value in vars(args).items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.warmstart_mode = "particles"
        config.collider_velocity_mode = "forward"
        self.solver = SolverImplicitMPM(self.model, config=config)

        self.particle_colors = wp.full(
            self.model.particle_count,
            value=SHEET_COLOR,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.particle_colors[self.fixed_indices].fill_(ANCHOR_COLOR)
        self.render_radii = wp.clone(self.model.particle_radius)

        self.particle_flags_np = self.model.particle_flags.numpy().copy()
        curve_progress = np.clip(
            (self.rest_positions_np[:, 0] + 0.5 * self.sheet_width) / self.sheet_width,
            0.0,
            1.0,
        )
        curve_envelope = 16.0 * curve_progress**2 * (1.0 - curve_progress) ** 2
        cut_center_y = self.cut_y + self.trajectory_curve_offset * curve_envelope
        cut_candidate_mask = np.abs(self.rest_positions_np[:, 1] - cut_center_y) <= self.cut_half_width
        cut_candidate_indices = np.flatnonzero(cut_candidate_mask).astype(np.int32)
        cut_order = np.argsort(self.rest_positions_np[cut_candidate_indices, 0], kind="stable")
        self.cut_candidate_indices_np = cut_candidate_indices[cut_order]
        self.cut_candidate_x_np = self.rest_positions_np[self.cut_candidate_indices_np, 0]
        self.cut_particle_count = 0

        self._update_scissors(0.0)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.82, -0.90, 1.72), pitch=-25.0, yaw=123.0)

        print(
            f"[newton] MPM suspended-sheet scissors: particles={self.model.particle_count}, "
            f"fixed={self.fixed_indices.shape[0]}, cut_candidates={self.cut_candidate_indices_np.size}, "
            f"thickness_layers=2"
        )

    @staticmethod
    def _emit_sheet(builder: newton.ModelBuilder, args) -> dict[str, np.ndarray]:
        width = float(args.sheet_width)
        length = float(args.sheet_length)
        thickness = float(args.sheet_thickness)
        voxel_size = float(args.voxel_size)
        particles_per_cell = int(args.particles_per_cell)
        anchor_rows = int(args.anchor_rows)
        density = float(args.density)
        anchor_height = float(args.anchor_height)
        initial_sag = float(args.initial_sag)
        initial_ripple = float(args.initial_ripple)

        if min(width, length, thickness, voxel_size, density) <= 0.0 or particles_per_cell <= 0:
            raise ValueError("sheet dimensions, voxel size, density, and particles per cell must be positive")
        if anchor_rows <= 0:
            raise ValueError("anchor rows must be positive")
        if initial_sag < 0.0 or initial_ripple < 0.0:
            raise ValueError("initial sag and ripple must be non-negative")

        target_spacing = voxel_size / particles_per_cell
        resolution = np.array(
            [
                max(int(np.ceil(width / target_spacing)), 1),
                max(int(np.ceil(length / target_spacing)), 1),
                1,
            ],
            dtype=np.int32,
        )
        if 2 * anchor_rows >= resolution[1] + 1:
            raise ValueError("anchor rows must leave free particle rows between the fixed ends")
        spacing = np.array([width, length, thickness]) / resolution

        x = np.linspace(-0.5 * width, 0.5 * width, int(resolution[0]) + 1)
        y = np.linspace(0.0, length, int(resolution[1]) + 1)
        z = np.linspace(-0.5 * thickness, 0.5 * thickness, int(resolution[2]) + 1)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

        normalized_y = grid_y / length
        center_envelope = 4.0 * normalized_y * (1.0 - normalized_y)
        ripple_phase = 3.0 * np.pi * (grid_x / width + 0.5)
        ripple = initial_ripple * np.sin(ripple_phase) * center_envelope
        positions = np.column_stack(
            (
                grid_x.ravel(),
                grid_y.ravel(),
                (anchor_height + grid_z - initial_sag * center_envelope + ripple).ravel(),
            )
        )

        cell_volume = float(np.prod(spacing))
        particle_radius = 0.55 * float(np.min(spacing))
        builder.add_particles(
            pos=positions.tolist(),
            vel=np.zeros_like(positions).tolist(),
            mass=[cell_volume * density] * len(positions),
            radius=[particle_radius] * len(positions),
        )

        anchor_width = (anchor_rows - 0.5) * spacing[1]
        fixed_indices = np.flatnonzero(
            (positions[:, 1] <= anchor_width) | (positions[:, 1] >= length - anchor_width)
        ).astype(np.int32)
        center_indices = np.flatnonzero(np.abs(positions[:, 1] - 0.5 * length) <= 0.51 * spacing[1]).astype(np.int32)
        if len(center_indices) == 0:
            raise ValueError("the center of the sheet contains no particles")
        return {
            "positions": positions,
            "spacing": spacing,
            "fixed_indices": fixed_indices,
            "center_indices": center_indices,
        }

    def _add_scissors(self, builder: newton.ModelBuilder) -> None:
        hinge_x, hinge_y, hinge_z, angle, yaw, _cut_front = self._scissor_motion(0.0)
        upper_rotation = self._blade_rotation(yaw, angle)
        lower_rotation = self._blade_rotation(yaw, -angle)
        upper_xform = wp.transform(wp.vec3(hinge_x, hinge_y, hinge_z), upper_rotation)
        lower_xform = wp.transform(wp.vec3(hinge_x, hinge_y, hinge_z), lower_rotation)

        self.upper_blade_body = builder.add_body(xform=upper_xform, label="upper_scissor_blade", is_kinematic=True)
        self.lower_blade_body = builder.add_body(xform=lower_xform, label="lower_scissor_blade", is_kinematic=True)
        self.hinge_body = builder.add_body(
            xform=wp.transform(wp.vec3(hinge_x, hinge_y, hinge_z), self._rotation_z(yaw)),
            label="scissor_hinge",
            is_kinematic=True,
        )

        blade_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=0.04,
            has_shape_collision=False,
            has_particle_collision=True,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        handle_mesh = create_torus_mesh(HANDLE_RING_MAJOR_RADIUS, HANDLE_RING_MINOR_RADIUS)

        for body, side in ((self.upper_blade_body, 1.0), (self.lower_blade_body, -1.0)):
            builder.add_shape_mesh(
                body=body,
                xform=wp.transform(wp.vec3(0.0, side * BLADE_Y_OFFSET, 0.0), wp.quat_identity()),
                mesh=create_curved_blade_mesh(side),
                cfg=visual_cfg,
                color=BLADE_COLOR,
                label="scissor_blade_surface",
            )
            for segment in range(BLADE_SEGMENTS):
                x_0 = BLADE_LENGTH * segment / BLADE_SEGMENTS
                x_1 = BLADE_LENGTH * (segment + 1) / BLADE_SEGMENTS
                z_0 = blade_edge_height(x_0, side)
                z_1 = blade_edge_height(x_1, side)
                delta_x = x_1 - x_0
                delta_z = z_1 - z_0
                segment_length = math.hypot(delta_x, delta_z)
                tangent_angle = -math.atan2(delta_z, delta_x)
                builder.add_shape_box(
                    body=body,
                    xform=wp.transform(
                        wp.vec3(
                            0.5 * (x_0 + x_1),
                            side * BLADE_Y_OFFSET,
                            0.5 * (z_0 + z_1) - side * BLADE_EDGE_HALF_HEIGHT,
                        ),
                        self._rotation_y(tangent_angle),
                    ),
                    hx=0.5 * segment_length + 0.0002,
                    hy=BLADE_HALF_THICKNESS,
                    hz=BLADE_EDGE_HALF_HEIGHT,
                    cfg=blade_cfg,
                    color=BLADE_EDGE_COLOR,
                    label="scissor_cutting_edge",
                )
            builder.add_shape_box(
                body=body,
                xform=wp.transform(
                    wp.vec3(-0.5 * HANDLE_LENGTH, side * HANDLE_Y_OFFSET, 0.0),
                    wp.quat_identity(),
                ),
                hx=0.5 * HANDLE_LENGTH,
                hy=HANDLE_HALF_WIDTH,
                hz=HANDLE_HALF_THICKNESS,
                cfg=visual_cfg,
                color=HANDLE_COLOR,
                label="scissor_handle",
            )
            builder.add_shape_mesh(
                body=body,
                xform=wp.transform(
                    wp.vec3(-HANDLE_LENGTH - HANDLE_RING_MAJOR_RADIUS, side * HANDLE_Y_OFFSET, 0.0),
                    wp.quat_identity(),
                ),
                mesh=handle_mesh,
                cfg=visual_cfg,
                color=HANDLE_COLOR,
                label="scissor_handle_ring",
            )

        builder.add_shape_sphere(
            body=self.hinge_body,
            radius=0.010,
            cfg=visual_cfg,
            color=HINGE_COLOR,
            label="scissor_hinge_pin",
        )

    @staticmethod
    def _rotation_y(angle: float) -> wp.quat:
        half_angle = 0.5 * angle
        return wp.quat(0.0, math.sin(half_angle), 0.0, math.cos(half_angle))

    @staticmethod
    def _rotation_z(angle: float) -> wp.quat:
        half_angle = 0.5 * angle
        return wp.quat(0.0, 0.0, math.sin(half_angle), math.cos(half_angle))

    @classmethod
    def _blade_rotation(cls, yaw: float, angle: float) -> wp.quat:
        return wp.mul(cls._rotation_z(yaw), cls._rotation_y(angle))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _curve_envelope(progress: float) -> tuple[float, float]:
        progress = float(np.clip(progress, 0.0, 1.0))
        complement = 1.0 - progress
        envelope = 16.0 * progress**2 * complement**2
        derivative = 32.0 * progress * complement * (1.0 - 2.0 * progress)
        return envelope, derivative

    def _scissor_motion(self, query_time: float) -> tuple[float, float, float, float, float, float]:
        x_min = -0.5 * self.sheet_width
        x_max = 0.5 * self.sheet_width
        parked_tip_x = x_min - 0.22
        entry_tip_x = x_min - 0.01
        tip_y = self.cut_y
        hinge_z = self.cut_z
        yaw = 0.0
        cut_front = x_min - self.sheet_spacing[0]

        if query_time < self.settle_time:
            tip_x = parked_tip_x
            angle = self.open_angle
        elif query_time < self.settle_time + self.approach_time:
            approach_alpha = self._smoothstep((query_time - self.settle_time) / max(self.approach_time, 1.0e-8))
            tip_x = parked_tip_x + approach_alpha * (entry_tip_x - parked_tip_x)
            angle = self.open_angle
        elif query_time < self.settle_time + self.approach_time + self.cut_time:
            cut_alpha = (query_time - self.settle_time - self.approach_time) / self.cut_time
            cut_alpha = float(np.clip(cut_alpha, 0.0, 1.0))

            snip_position = cut_alpha * self.snip_count
            snip_index = min(int(snip_position), self.snip_count - 1)
            cycle = snip_position - snip_index
            if cycle < 0.5:
                closure = 2.0 * cycle
                cut_progress = (snip_index + closure) / self.snip_count
                angle = self.open_angle + closure * (self.closed_angle - self.open_angle)
            else:
                reopen = 2.0 * (cycle - 0.5)
                cut_progress = (snip_index + 1.0) / self.snip_count
                angle = self.closed_angle + reopen * (self.open_angle - self.closed_angle)
            tip_x = entry_tip_x + cut_progress * (x_max + 0.03 - entry_tip_x)
            cut_front = x_min + cut_progress * (x_max - x_min + self.sheet_spacing[0])
            curve_envelope, curve_derivative = self._curve_envelope(cut_progress)
            tip_y += self.trajectory_curve_offset * curve_envelope
            tip_path_length = x_max + 0.03 - entry_tip_x
            yaw = math.atan2(self.trajectory_curve_offset * curve_derivative, tip_path_length)
        else:
            depart_alpha = self._smoothstep(
                (query_time - self.settle_time - self.approach_time - self.cut_time) / max(self.depart_time, 1.0e-8)
            )
            tip_x = x_max + 0.03 + 0.16 * depart_alpha
            hinge_z += 0.10 * depart_alpha
            angle = self.open_angle
            cut_front = x_max + self.sheet_spacing[0]

        horizontal_blade_length = BLADE_LENGTH * math.cos(angle)
        hinge_x = tip_x - horizontal_blade_length * math.cos(yaw)
        hinge_y = tip_y - horizontal_blade_length * math.sin(yaw)
        return hinge_x, hinge_y, hinge_z, angle, yaw, cut_front

    def _update_scissors(self, query_time: float) -> None:
        hinge_x, hinge_y, hinge_z, angle, yaw, _cut_front = self._scissor_motion(query_time)
        prev_hinge_x, prev_hinge_y, prev_hinge_z, prev_angle, prev_yaw, _prev_cut_front = self._scissor_motion(
            max(query_time - self.sim_dt, 0.0)
        )
        inv_dt = 1.0 / self.sim_dt
        linear_velocity = wp.vec3(
            (hinge_x - prev_hinge_x) * inv_dt,
            (hinge_y - prev_hinge_y) * inv_dt,
            (hinge_z - prev_hinge_z) * inv_dt,
        )
        angular_speed = (angle - prev_angle) * inv_dt
        yaw_speed = (yaw - prev_yaw) * inv_dt
        opening_axis = wp.vec3(-math.sin(yaw), math.cos(yaw), 0.0)
        yaw_velocity = wp.vec3(0.0, 0.0, yaw_speed)
        position = wp.vec3(hinge_x, hinge_y, hinge_z)

        wp.launch(
            set_kinematic_body_pose,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.upper_blade_body,
                wp.transform(position, self._blade_rotation(yaw, angle)),
                linear_velocity,
                yaw_velocity + angular_speed * opening_axis,
            ],
            device=self.model.device,
        )
        wp.launch(
            set_kinematic_body_pose,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.lower_blade_body,
                wp.transform(position, self._blade_rotation(yaw, -angle)),
                linear_velocity,
                yaw_velocity - angular_speed * opening_axis,
            ],
            device=self.model.device,
        )
        wp.launch(
            set_kinematic_body_pose,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.hinge_body,
                wp.transform(position, self._rotation_z(yaw)),
                linear_velocity,
                yaw_velocity,
            ],
            device=self.model.device,
        )

    def _advance_cut(self, cut_front: float) -> None:
        new_count = int(np.searchsorted(self.cut_candidate_x_np, cut_front, side="right"))
        if new_count <= self.cut_particle_count:
            return

        new_indices_np = self.cut_candidate_indices_np[self.cut_particle_count : new_count]
        self.particle_flags_np[new_indices_np] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(
            self.particle_flags_np,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

        new_indices = wp.array(new_indices_np, dtype=wp.int32, device=self.model.device)
        self.render_radii[new_indices].fill_(0.0)
        self.cut_particle_count = new_count

    def simulate(self) -> None:
        for substep in range(self.sim_substeps):
            query_time = self.sim_time + (substep + 1) * self.sim_dt
            self._update_scissors(query_time)
            self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
            self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)

        self.sim_time += self.frame_dt
        _hinge_x, _hinge_y, _hinge_z, _angle, _yaw, cut_front = self._scissor_motion(self.sim_time)
        self._advance_cut(cut_front)

    def step(self) -> None:
        self.simulate()

    def render(self) -> None:
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self.viewer.show_particles = False
        self.viewer.log_state(self.state_0)
        self.viewer.show_particles = show_particles
        self.viewer.log_points(
            name="/suspended_sheet",
            points=self.state_0.particle_q,
            radii=self.render_radii,
            colors=self.particle_colors,
            hidden=not show_particles,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Verify the sheet and scissors remain finite after every frame."""
        positions = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(body_q)):
            raise ValueError("suspended-sheet scissors state is not finite")

    def test_final(self) -> None:
        """Verify both edges stay fixed and the completed cut is persistent."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("suspended-sheet scissors state is not finite")
        if np.ptp(self.fixed_positions_initial[:, 1]) < 0.9 * self.sheet_length:
            raise ValueError("fixed particles do not span both ends of the sheet")
        if np.max(np.abs(positions[self.fixed_indices_np] - self.fixed_positions_initial)) > 1.0e-5:
            raise ValueError("kinematic edge particles moved")
        if np.linalg.norm(np.ptp(positions, axis=0)) > 5.0:
            raise ValueError("suspended-sheet particles became unbounded")

        cut_end_time = self.settle_time + self.approach_time + self.cut_time
        if self.sim_time >= cut_end_time and self.cut_particle_count != self.cut_candidate_indices_np.size:
            raise ValueError(
                f"cut stopped after {self.cut_particle_count} of {self.cut_candidate_indices_np.size} particles"
            )

    @staticmethod
    def create_parser():
        """Create command-line arguments for the suspended-sheet scissors demo."""
        parser = newton.examples.create_parser()
        parser.description = "Cut a two-edge suspended volumetric MPM sheet with animated scissors."
        parser.set_defaults(num_frames=420)
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--gravity", type=float, nargs=3, default=(0.0, 0.0, -9.81))
        parser.add_argument("--sheet-width", type=float, default=0.70, help="Sheet width [m].")
        parser.add_argument("--sheet-length", type=float, default=0.75, help="Distance between fixed ends [m].")
        parser.add_argument("--sheet-thickness", type=float, default=0.01, help="Two-layer sheet thickness [m].")
        parser.add_argument("--anchor-height", type=float, default=1.20, help="Fixed-edge height [m].")
        parser.add_argument("--anchor-rows", type=int, default=2, help="Particle rows fixed at each end.")
        parser.add_argument("--initial-sag", type=float, default=0.005, help="Initial center sag [m].")
        parser.add_argument("--initial-ripple", type=float, default=0.004, help="Initial cross-sheet ripple [m].")
        parser.add_argument("--cut-depth", type=float, default=0.09, help="Cut height below the fixed edges [m].")
        parser.add_argument("--cut-half-width", type=float, default=0.011, help="Half-width of the cut band [m].")
        parser.add_argument(
            "--trajectory-curve-offset",
            type=float,
            default=0.07,
            help="Maximum in-plane offset of the curved cutting trajectory [m].",
        )
        parser.add_argument("--settle-time", type=float, default=1.25, help="Sheet settling time [s].")
        parser.add_argument("--approach-time", type=float, default=0.75, help="Scissor approach time [s].")
        parser.add_argument("--cut-time", type=float, default=4.0, help="Progressive cutting time [s].")
        parser.add_argument("--depart-time", type=float, default=0.75, help="Scissor departure time [s].")
        parser.add_argument("--snips", type=int, default=8, help="Number of open-close scissor cycles.")
        parser.add_argument("--open-angle", type=float, default=18.0, help="Open half-angle of the blades [deg].")
        parser.add_argument("--closed-angle", type=float, default=2.0, help="Closed half-angle of the blades [deg].")
        parser.add_argument("--particles-per-cell", type=int, default=2)
        parser.add_argument("--density", type=float, default=100.0, help="Effective sheet density [kg/m³].")
        parser.add_argument("--young-modulus", "-ym", type=float, default=5.0e4, help="Young's modulus [Pa].")
        parser.add_argument("--poisson-ratio", "-nu", type=float, default=0.30)
        parser.add_argument("--damping", type=float, default=0.05, help="Elastic damping relaxation time [s].")
        parser.add_argument("--air-drag", type=float, default=1.5, help="Background numerical drag.")
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.02)
        parser.add_argument("--grid-type", choices=("sparse", "dense", "fixed"), default="sparse")
        parser.add_argument("--strain-basis", choices=("P0", "P1d", "Q1", "Q1d"), default="P1d")
        parser.add_argument("--max-iterations", "-it", type=int, default=150)
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-4)
        return parser


def main():
    """Run the MPM suspended-sheet scissors demo."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()
