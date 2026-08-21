# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Body Cutting
#
# Demonstrates a Newton-native cohesive-zone cutting prototype inspired by
# DiSECt. A tetrahedral block is split along a prescribed plane before model
# finalization. Coincident particle pairs keep the two halves joined until a
# descending wedge damages and releases their cohesive constraints.
#
# Command: uv run -m newton.examples softbody_cutting
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def apply_cohesive_cutting_forces(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    pair_indices: wp.array2d[int],
    pair_rest_positions: wp.array[wp.vec3],
    damage: wp.array[float],
    knife_edge_z: float,
    knife_height: float,
    knife_half_length: float,
    fracture_depth: float,
    cohesive_ke: float,
    cohesive_kd: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply cohesive forces and update damage at a prescribed cut plane."""
    pair = wp.tid()
    left = pair_indices[pair, 0]
    right = pair_indices[pair, 1]

    q_left = particle_q[left]
    q_right = particle_q[right]
    qd_left = particle_qd[left]
    qd_right = particle_qd[right]
    rest = pair_rest_positions[pair]

    pair_damage = damage[pair]
    height_above_edge = rest[2] - knife_edge_z
    inside_blade = (
        wp.abs(rest[1]) <= knife_half_length and height_above_edge >= 0.0 and height_above_edge <= knife_height
    )

    if inside_blade:
        # Fracture transitions only in the narrow region immediately behind
        # the edge; the rest of the blade continues to provide wedge contact.
        target_damage = wp.clamp(height_above_edge / fracture_depth, 0.0, 1.0)
        pair_damage = wp.max(pair_damage, target_damage)
        damage[pair] = pair_damage

    intact = 1.0 - pair_damage
    delta = q_right - q_left
    delta_velocity = qd_right - qd_left
    cohesive_force = (delta * cohesive_ke + delta_velocity * cohesive_kd) * intact
    wp.atomic_add(particle_f, left, cohesive_force)
    wp.atomic_sub(particle_f, right, cohesive_force)


@wp.kernel
def project_intact_pairs(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    pair_indices: wp.array2d[int],
    damage: wp.array[float],
):
    """Keep undamaged virtual-node pairs exactly coincident."""
    pair = wp.tid()
    if damage[pair] == 0.0:
        left = pair_indices[pair, 0]
        right = pair_indices[pair, 1]
        average_q = 0.5 * (particle_q[left] + particle_q[right])
        average_qd = 0.5 * (particle_qd[left] + particle_qd[right])
        particle_q[left] = average_q
        particle_q[right] = average_q
        particle_qd[left] = average_qd
        particle_qd[right] = average_qd


@wp.kernel
def project_knife_contact(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    pair_indices: wp.array2d[int],
    pair_rest_positions: wp.array[wp.vec3],
    damage: wp.array[float],
    knife_edge_z: float,
    knife_height: float,
    knife_half_length: float,
    knife_half_thickness: float,
    cut_plane_x: float,
    knife_end_z: float,
    retract_progress: float,
    residual_half_opening: float,
):
    """Project damaged cut faces during wedge contact and damped recovery."""
    pair = wp.tid()
    if damage[pair] == 0.0:
        return

    rest = pair_rest_positions[pair]
    height_above_edge = rest[2] - knife_edge_z
    inside_blade = (
        wp.abs(rest[1]) <= knife_half_length and height_above_edge >= 0.0 and height_above_edge <= knife_height
    )
    left = pair_indices[pair, 0]
    right = pair_indices[pair, 1]
    q_left = particle_q[left]
    q_right = particle_q[right]
    qd_left = particle_qd[left]
    qd_right = particle_qd[right]

    if retract_progress > 0.0:
        bottom_height = wp.clamp((rest[2] - knife_end_z) / knife_height, 0.0, 1.0)
        inserted_half_gap = knife_half_thickness * bottom_height
        wedge_half_gap = inserted_half_gap * (1.0 - retract_progress) + residual_half_opening * retract_progress
        center_y = 0.5 * (q_left[1] + q_right[1])
        center_z = 0.5 * (q_left[2] + q_right[2])
        center_vy = 0.5 * (qd_left[1] + qd_right[1])
        center_vz = 0.5 * (qd_left[2] + qd_right[2])
        q_left[0] = cut_plane_x - wedge_half_gap
        q_left[1] = center_y
        q_left[2] = center_z
        q_right[0] = cut_plane_x + wedge_half_gap
        q_right[1] = center_y
        q_right[2] = center_z
        qd_left[0] = 0.0
        qd_left[1] = center_vy
        qd_left[2] = center_vz
        qd_right[0] = 0.0
        qd_right[1] = center_vy
        qd_right[2] = center_vz
        particle_q[left] = q_left
        particle_q[right] = q_right
        particle_qd[left] = qd_left
        particle_qd[right] = qd_right
    elif inside_blade:
        wedge_half_gap = knife_half_thickness * height_above_edge / knife_height
        q_left[0] = cut_plane_x - wedge_half_gap
        q_right[0] = cut_plane_x + wedge_half_gap
        qd_left[0] = 0.0
        qd_right[0] = 0.0
        particle_q[left] = q_left
        particle_q[right] = q_right
        particle_qd[left] = qd_left
        particle_qd[right] = qd_right
    else:
        # Once the blade leaves, keep the disconnected faces from crossing
        # while allowing their elastic deformation to close naturally.
        if q_left[0] > cut_plane_x:
            q_left[0] = cut_plane_x
            qd_left[0] = wp.min(qd_left[0], 0.0)
            particle_q[left] = q_left
            particle_qd[left] = qd_left
        if q_right[0] < cut_plane_x:
            q_right[0] = cut_plane_x
            qd_right[0] = wp.max(qd_right[0], 0.0)
            particle_q[right] = q_right
            particle_qd[right] = qd_right


def add_presplit_block(
    builder: newton.ModelBuilder,
    *,
    half_dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_size: float,
    base_z: float,
    density: float,
    k_mu: float,
    k_lambda: float,
    k_damp: float,
    particle_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Add two tetrahedral half-blocks and return coincident cut-plane pairs."""
    half_width = half_dim_x * cell_size
    half_depth = 0.5 * dim_y * cell_size
    common = {
        "rot": wp.quat_identity(),
        "vel": wp.vec3(0.0, 0.0, 0.0),
        "dim_x": half_dim_x,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "cell_x": cell_size,
        "cell_y": cell_size,
        "cell_z": cell_size,
        "density": density,
        "k_mu": k_mu,
        "k_lambda": k_lambda,
        "k_damp": k_damp,
        "add_surface_mesh_edges": False,
        "particle_radius": particle_radius,
    }

    left_start = builder.particle_count
    builder.add_soft_grid(pos=wp.vec3(-half_width, -half_depth, base_z), **common)
    right_start = builder.particle_count
    builder.add_soft_grid(pos=wp.vec3(0.0, -half_depth, base_z), **common)

    row_size = half_dim_x + 1
    layer_size = row_size * (dim_y + 1)
    pairs = []
    rest_positions = []
    for z in range(dim_z + 1):
        for y in range(dim_y + 1):
            left = left_start + z * layer_size + y * row_size + half_dim_x
            right = right_start + z * layer_size + y * row_size
            pairs.append((left, right))
            rest_positions.append((0.0, y * cell_size - half_depth, z * cell_size + base_z))

    return np.asarray(pairs, dtype=np.int32), np.asarray(rest_positions, dtype=np.float32)


def create_knife_mesh(half_length: float, height: float, half_thickness: float) -> newton.Mesh:
    """Create a closed chef-knife blade with a sharp edge and short bevel."""
    half_height = 0.5 * height
    bevel_height = min(0.055, 0.25 * height)
    bevel_z = -half_height + bevel_height
    tip_spine_y = 0.72 * half_length
    vertices = np.asarray(
        [
            (0.0, -half_length, -half_height),
            (0.0, half_length, -half_height),
            (-half_thickness, -half_length, bevel_z),
            (-half_thickness, half_length, bevel_z),
            (-half_thickness, tip_spine_y, half_height),
            (-half_thickness, -half_length, half_height),
            (half_thickness, -half_length, bevel_z),
            (half_thickness, half_length, bevel_z),
            (half_thickness, tip_spine_y, half_height),
            (half_thickness, -half_length, half_height),
        ],
        dtype=np.float32,
    )
    indices = np.asarray(
        [
            0,
            2,
            3,
            0,
            3,
            1,
            2,
            5,
            4,
            2,
            4,
            3,
            0,
            1,
            7,
            0,
            7,
            6,
            6,
            7,
            8,
            6,
            8,
            9,
            5,
            9,
            8,
            5,
            8,
            4,
            0,
            6,
            9,
            0,
            9,
            5,
            0,
            5,
            2,
            1,
            3,
            7,
            3,
            4,
            8,
            3,
            8,
            7,
        ],
        dtype=np.int32,
    )
    mesh = newton.Mesh(vertices, indices)
    mesh.finalize()
    return mesh


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.cell_size = 0.12
        self.half_dim_x = 2
        self.dim_y = 6
        self.dim_z = 5
        self.base_z = 0.035
        self.block_top = self.base_z + self.dim_z * self.cell_size

        self.knife_height = 0.68
        self.knife_half_length = 0.5 * self.dim_y * self.cell_size + 0.055
        self.knife_half_thickness = 0.014
        self.fracture_depth = 0.020
        self.knife_start_z = self.block_top + 0.20
        self.knife_end_z = self.base_z - self.fracture_depth
        if args.cut_start_time < 0.0:
            raise ValueError("cut start time must be nonnegative")
        if args.cut_duration <= 0.0:
            raise ValueError("cut duration must be positive")
        if args.cut_hold_time < 0.0:
            raise ValueError("cut hold time must be nonnegative")
        if args.retract_duration <= 0.0:
            raise ValueError("retract duration must be positive")
        self.cut_start_time = args.cut_start_time
        self.cut_duration = args.cut_duration
        self.cut_hold_time = args.cut_hold_time
        self.retract_duration = args.retract_duration
        self.residual_half_opening = 0.0025
        self.cut_end_time = self.cut_start_time + self.cut_duration
        self.retract_start_time = self.cut_end_time + self.cut_hold_time
        self.retract_end_time = self.retract_start_time + self.retract_duration
        self.knife_edge_z = self.knife_start_z

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        pair_indices, pair_rest_positions = add_presplit_block(
            builder,
            half_dim_x=self.half_dim_x,
            dim_y=self.dim_y,
            dim_z=self.dim_z,
            cell_size=self.cell_size,
            base_z=self.base_z,
            density=350.0,
            k_mu=2.0e4,
            k_lambda=3.0e4,
            k_damp=400.0,
            particle_radius=0.018,
        )

        contact_cfg = newton.ModelBuilder.ShapeConfig(ke=5.0e4, kd=200.0, kf=0.0, mu=1.2)
        builder.add_ground_plane(cfg=contact_cfg)

        self.model = builder.finalize()
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 200.0
        self.model.soft_contact_kf = 0.0
        self.model.soft_contact_mu = 1.2

        # Coincident virtual nodes must not collide before their cohesive link fails.
        self.model.particle_grid = None

        self.pair_indices = wp.array(pair_indices, dtype=wp.int32, device=self.model.device)
        self.pair_rest_positions = wp.array(pair_rest_positions, dtype=wp.vec3, device=self.model.device)
        self.damage = wp.zeros(len(pair_indices), dtype=float, device=self.model.device)

        surface_indices = self.model.tri_indices.numpy()
        rest_positions = self.model.particle_q.numpy()
        is_cut_face = np.all(np.abs(rest_positions[surface_indices, 0]) < 1.0e-6, axis=1)
        outer_indices = surface_indices[~is_cut_face]
        cut_face_indices = surface_indices[is_cut_face]
        cut_face_min_z = np.min(rest_positions[cut_face_indices, 2], axis=1)
        pair_z = pair_rest_positions[:, 2]

        # Each path has fixed topology for the viewers. A variant welds the
        # uncut rows and exposes only cut faces above the current crack tip.
        self.surface_outer_variants = []
        self.surface_cut_variants = []
        for first_released_row in range(self.dim_z + 2):
            release_z = self.base_z + first_released_row * self.cell_size
            weld_map = np.arange(self.model.particle_count, dtype=np.int32)
            uncut_pairs = pair_z < release_z - 1.0e-6
            weld_map[pair_indices[uncut_pairs, 1]] = pair_indices[uncut_pairs, 0]

            variant_outer = weld_map[outer_indices].reshape(-1)
            revealed_cut_faces = cut_face_indices[cut_face_min_z >= release_z - 1.0e-6].reshape(-1)
            self.surface_outer_variants.append(wp.array(variant_outer, dtype=wp.int32, device=self.model.device))
            self.surface_cut_variants.append(
                wp.array(revealed_cut_faces, dtype=wp.int32, device=self.model.device)
                if len(revealed_cut_faces) > 0
                else None
            )
        self.active_surface_variant = self.dim_z + 1

        self.cohesive_ke = 2.0e5
        self.cohesive_kd = 300.0

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_margin=0.01)
        self.contacts = self.collision_pipeline.contacts()

        self.knife_mesh = create_knife_mesh(
            self.knife_half_length,
            self.knife_height,
            self.knife_half_thickness,
        )
        self.blade_color = wp.array([wp.vec3(0.72, 0.76, 0.82)], dtype=wp.vec3, device=self.model.device)
        self.handle_color = wp.array([wp.vec3(0.24, 0.055, 0.025)], dtype=wp.vec3, device=self.model.device)
        self.rivet_color = wp.array([wp.vec3(0.72, 0.75, 0.78)], dtype=wp.vec3, device=self.model.device)
        self.blade_material = wp.array([wp.vec4(0.38, 0.68, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.handle_material = wp.array([wp.vec4(0.34, 0.04, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.rivet_material = wp.array([wp.vec4(0.30, 0.75, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(wp.vec3(1.55, -2.20, 1.32), -13.0, 126.0)

    def knife_edge_height(self, time: float) -> float:
        """Return the blade-edge height for settle, cut, hold, and retract phases."""
        if time <= self.cut_start_time:
            return self.knife_start_z
        if time <= self.cut_end_time:
            phase = (time - self.cut_start_time) / self.cut_duration
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            return self.knife_start_z + (self.knife_end_z - self.knife_start_z) * smooth_phase
        if time <= self.retract_start_time:
            return self.knife_end_z
        phase = min((time - self.retract_start_time) / self.retract_duration, 1.0)
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        return self.knife_end_z + (self.knife_start_z - self.knife_end_z) * smooth_phase

    def knife_retract_progress(self, time: float) -> float:
        """Return smooth normalized progress through the upward stroke."""
        if time <= self.retract_start_time:
            return 0.0
        phase = min((time - self.retract_start_time) / self.retract_duration, 1.0)
        return phase * phase * (3.0 - 2.0 * phase)

    def surface_variant_candidate(self) -> int:
        """Return the cut-plane row implied by the current blade height."""
        release_threshold = self.knife_edge_z + self.fracture_depth
        for row in range(self.dim_z + 1):
            if self.base_z + row * self.cell_size >= release_threshold - 1.0e-6:
                return row
        return self.dim_z + 1

    def surface_variant_index(self) -> int:
        """Return the deepest irreversibly released row used for rendering."""
        return self.active_surface_variant

    def simulate(self):
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + substep * self.sim_dt
            self.knife_edge_z = self.knife_edge_height(substep_time)
            retract_progress = self.knife_retract_progress(substep_time)

            self.state_0.clear_forces()
            wp.launch(
                kernel=apply_cohesive_cutting_forces,
                dim=self.damage.shape[0],
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.pair_indices,
                    self.pair_rest_positions,
                    self.damage,
                    self.knife_edge_z,
                    self.knife_height,
                    self.knife_half_length,
                    self.fracture_depth,
                    self.cohesive_ke,
                    self.cohesive_kd,
                ],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            wp.launch(
                kernel=project_intact_pairs,
                dim=self.damage.shape[0],
                inputs=[self.state_1.particle_q, self.state_1.particle_qd, self.pair_indices, self.damage],
                device=self.model.device,
            )
            wp.launch(
                kernel=project_knife_contact,
                dim=self.damage.shape[0],
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.pair_indices,
                    self.pair_rest_positions,
                    self.damage,
                    self.knife_edge_z,
                    self.knife_height,
                    self.knife_half_length,
                    self.knife_half_thickness,
                    0.0,
                    self.knife_end_z,
                    retract_progress,
                    self.residual_half_opening,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        self.knife_edge_z = self.knife_edge_height(self.sim_time)
        self.active_surface_variant = min(self.active_surface_variant, self.surface_variant_candidate())

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        active_surface = self.surface_variant_index()
        for variant, indices in enumerate(self.surface_outer_variants):
            hidden = variant != active_surface
            self.viewer.log_mesh(
                f"/cutting_block/outer_{variant}",
                self.state_0.particle_q,
                indices,
                hidden=hidden,
                backface_culling=False,
                color=(0.82, 0.55, 0.20),
                roughness=0.72,
            )
            cut_indices = self.surface_cut_variants[variant]
            if cut_indices is not None:
                self.viewer.log_mesh(
                    f"/cutting_block/cut_{variant}",
                    self.state_0.particle_q,
                    cut_indices,
                    hidden=hidden,
                    backface_culling=False,
                    color=(0.63, 0.34, 0.12),
                    roughness=0.86,
                )

        blade_center = wp.vec3(0.0, 0.0, self.knife_edge_z + 0.5 * self.knife_height)
        blade_xform = wp.array(
            [wp.transform(blade_center, wp.quat_identity())], dtype=wp.transform, device=self.model.device
        )
        self.viewer.log_shapes(
            "/cutting_tool/blade",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            blade_xform,
            self.blade_color,
            self.blade_material,
            geo_src=self.knife_mesh,
        )

        handle_half_length = 0.18
        handle_center = wp.vec3(
            0.0,
            -self.knife_half_length - handle_half_length + 0.035,
            self.knife_edge_z + self.knife_height - 0.10,
        )
        handle_xform = wp.array(
            [wp.transform(handle_center, wp.quat_identity())], dtype=wp.transform, device=self.model.device
        )
        self.viewer.log_shapes(
            "/cutting_tool/handle",
            newton.GeoType.BOX,
            (0.038, handle_half_length, 0.060),
            handle_xform,
            self.handle_color,
            self.handle_material,
        )

        bolster_center = wp.vec3(
            0.0,
            -self.knife_half_length - 0.005,
            self.knife_edge_z + self.knife_height - 0.10,
        )
        bolster_xform = wp.array(
            [wp.transform(bolster_center, wp.quat_identity())], dtype=wp.transform, device=self.model.device
        )
        self.viewer.log_shapes(
            "/cutting_tool/bolster",
            newton.GeoType.BOX,
            (0.043, 0.045, 0.072),
            bolster_xform,
            self.rivet_color,
            self.rivet_material,
        )

        rivet_xforms = wp.array(
            [
                wp.transform(
                    wp.vec3(0.039, handle_center[1] - 0.065, handle_center[2]),
                    wp.quat_identity(),
                ),
                wp.transform(
                    wp.vec3(0.039, handle_center[1] + 0.065, handle_center[2]),
                    wp.quat_identity(),
                ),
            ],
            dtype=wp.transform,
            device=self.model.device,
        )
        self.viewer.log_shapes(
            "/cutting_tool/rivets",
            newton.GeoType.SPHERE,
            0.012,
            rivet_xforms,
            self.rivet_color,
            self.rivet_material,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify that damage follows the blade edge without premature release."""
        damage = self.damage.numpy()
        rest_positions = self.pair_rest_positions.numpy()
        if self.sim_time <= self.cut_end_time:
            unreached = rest_positions[:, 2] < self.knife_edge_z
            assert np.max(damage[unreached], initial=0.0) == 0.0

            if self.knife_edge_z > self.block_top - self.fracture_depth:
                assert self.surface_variant_index() == self.dim_z + 1

            released = rest_positions[:, 2] > self.knife_edge_z + self.fracture_depth
            if np.any(released) and np.any(unreached):
                positions = self.state_0.particle_q.numpy()
                pair_indices = self.pair_indices.numpy()
                opening = positions[pair_indices[:, 1], 0] - positions[pair_indices[:, 0], 0]
                wedge_opening = (
                    2.0 * self.knife_half_thickness * (rest_positions[:, 2] - self.knife_edge_z) / self.knife_height
                )
                assert np.mean(opening[released]) > 0.5 * np.mean(wedge_opening[released])

            if self.knife_edge_z > self.base_z:
                assert np.min(damage) < 1.0

        if self.sim_time < self.cut_start_time:
            positions = self.state_0.particle_q.numpy()
            pair_indices = self.pair_indices.numpy()
            mismatch = positions[pair_indices[:, 1]] - positions[pair_indices[:, 0]]
            assert np.max(np.linalg.norm(mismatch, axis=1)) < 1.0e-6

    def test_final(self):
        """Verify that the blade fully releases and opens the prescribed cut."""
        damage = self.damage.numpy()
        assert np.all(np.isfinite(damage))
        assert np.all((damage >= 0.0) & (damage <= 1.0))
        assert np.min(damage) > 0.99, f"not all cohesive constraints failed: min damage={np.min(damage)}"
        assert self.sim_time > self.retract_end_time
        assert self.knife_edge_z > self.block_top
        assert self.surface_variant_index() == 0

        positions = self.state_0.particle_q.numpy()
        pair_indices = self.pair_indices.numpy()
        opening = positions[pair_indices[:, 1], 0] - positions[pair_indices[:, 0], 0]
        mean_opening = np.mean(opening)
        assert mean_opening > 0.004, f"cut did not visibly open: mean opening={mean_opening}"
        assert mean_opening < 0.025, f"cut opened too far: mean opening={mean_opening}"
        assert np.max(opening) < 0.035, f"cut opened too far: max opening={np.max(opening)}"
        assert np.max(np.abs(opening - 2.0 * self.residual_half_opening)) < 1.0e-5

        tangential_offset = positions[pair_indices[:, 1], 1:] - positions[pair_indices[:, 0], 1:]
        assert np.max(np.linalg.norm(tangential_offset, axis=1)) < 1.0e-5

        bounds_lower = wp.vec3(-1.0, -1.0, -0.1)
        bounds_upper = wp.vec3(1.0, 1.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "cut block remains within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, bounds_lower, bounds_upper),
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--cut-start-time",
            type=float,
            default=0.75,
            help="Time to wait before the knife starts moving, in seconds.",
        )
        parser.add_argument(
            "--cut-duration",
            type=float,
            default=1.65,
            help="Duration of the knife's downward cutting stroke, in seconds.",
        )
        parser.add_argument(
            "--cut-hold-time",
            type=float,
            default=0.15,
            help="Time to hold the knife at the bottom before retracting, in seconds.",
        )
        parser.add_argument(
            "--retract-duration",
            type=float,
            default=1.0,
            help="Duration of the knife's upward retraction, in seconds.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
