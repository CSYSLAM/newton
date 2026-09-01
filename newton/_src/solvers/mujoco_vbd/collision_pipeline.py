# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""MJVBDV2 collision-pipeline helpers."""

from __future__ import annotations

from copy import copy

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import Contacts, Model, State
from .config import MuJoCoVBDStaticContactOwner
from .diagnostics import MuJoCoVBDDiagnostics
from .point_contact_kernels import compute_shape_world_aabbs, create_soft_contacts_with_aabb

__all__ = ["MJVBDV2SoftContactPipeline", "MuJoCoVBDCollisionPipeline"]


@wp.kernel
def _record_candidate_overflow_kernel(
    broad_count: wp.array[wp.int32],
    broad_capacity: int,
    split_query_count: wp.array[wp.int32],
    split_query_capacity: int,
    gjk_count: wp.array[wp.int32],
    gjk_capacity: int,
    split_gjk_count: wp.array[wp.int32],
    split_gjk_capacity: int,
    split_manifold_count: wp.array[wp.int32],
    split_manifold_capacity: int,
    rigid_overflow: wp.array[wp.int32],
):
    if broad_count[0] > broad_capacity:
        rigid_overflow[0] = 1
    if split_query_capacity >= 0 and split_query_count[0] > split_query_capacity:
        rigid_overflow[0] = 1
    if gjk_count[0] > gjk_capacity:
        rigid_overflow[0] = 1
    if split_gjk_capacity >= 0 and split_gjk_count[0] > split_gjk_capacity:
        rigid_overflow[0] = 1
    if split_manifold_capacity >= 0 and split_manifold_count[0] > split_manifold_capacity:
        rigid_overflow[0] = 1


@wp.kernel
def _record_contact_overflow_kernel(
    mesh_count: wp.array[wp.int32],
    mesh_capacity: int,
    triangle_count: wp.array[wp.int32],
    triangle_capacity: int,
    mesh_plane_count: wp.array[wp.int32],
    mesh_plane_capacity: int,
    mesh_mesh_count: wp.array[wp.int32],
    mesh_mesh_capacity: int,
    sdf_sdf_count: wp.array[wp.int32],
    sdf_sdf_capacity: int,
    rigid_count: wp.array[wp.int32],
    rigid_capacity: int,
    reduction_failures: wp.array[wp.int32],
    soft_count: wp.array[wp.int32],
    soft_capacity: int,
    rigid_overflow: wp.array[wp.int32],
    soft_overflow: wp.array[wp.int32],
):
    if mesh_count[0] > mesh_capacity:
        rigid_overflow[0] = 1
    if triangle_count[0] > triangle_capacity:
        rigid_overflow[0] = 1
    if mesh_plane_count[0] > mesh_plane_capacity:
        rigid_overflow[0] = 1
    if mesh_mesh_count[0] > mesh_mesh_capacity:
        rigid_overflow[0] = 1
    if sdf_sdf_count[0] > sdf_sdf_capacity:
        rigid_overflow[0] = 1
    if rigid_count[0] > rigid_capacity or reduction_failures[0] != 0:
        rigid_overflow[0] = 1
    if soft_count[0] > soft_capacity:
        soft_overflow[0] = 1


@wp.kernel
def _record_hydro_overflow_kernel(
    broad_count: wp.array[wp.int32],
    broad_capacity: int,
    iso0_count: wp.array[wp.int32],
    iso0_capacity: int,
    iso1_count: wp.array[wp.int32],
    iso1_capacity: int,
    iso2_count: wp.array[wp.int32],
    iso2_capacity: int,
    voxel_count: wp.array[wp.int32],
    voxel_capacity: int,
    face_count: wp.array[wp.int32],
    face_capacity: int,
    reduction_failures: wp.array[wp.int32],
    rigid_overflow: wp.array[wp.int32],
):
    if broad_count[0] > broad_capacity:
        rigid_overflow[0] = 1
    if iso0_count[0] > iso0_capacity:
        rigid_overflow[0] = 1
    if iso1_count[0] > iso1_capacity:
        rigid_overflow[0] = 1
    if iso2_count[0] > iso2_capacity:
        rigid_overflow[0] = 1
    if voxel_count[0] > voxel_capacity:
        rigid_overflow[0] = 1
    if face_count[0] > face_capacity or reduction_failures[0] != 0:
        rigid_overflow[0] = 1


def _empty_pairs(device: wp.context.Devicelike) -> wp.array:
    return wp.array(np.empty((0, 2), dtype=np.int32), dtype=wp.vec2i, device=device)


def _count_world_compatible_particle_shape_pairs(model: Model) -> int:
    """Count the particle-shape candidate upper bound without cross-world pairs."""
    if model.particle_count == 0 or model.shape_count == 0:
        return 0

    particle_start = np.asarray(model.particle_world_start.numpy(), dtype=np.int64)
    shape_start = np.asarray(model.shape_world_start.numpy(), dtype=np.int64)
    global_particles = int(particle_start[-1] - particle_start[-2] + particle_start[0])
    global_shapes = int(shape_start[-1] - shape_start[-2] + shape_start[0])

    total = global_particles * model.shape_count
    total += (model.particle_count - global_particles) * global_shapes
    local_worlds = slice(0, model.world_count + 1)
    total += int(np.dot(np.diff(particle_start[local_worlds]), np.diff(shape_start[local_worlds])))
    return total


def _build_particle_shape_pairs(model: Model) -> wp.array:
    if model.particle_count == 0 or model.shape_count == 0:
        return _empty_pairs(model.device)

    particle_world = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int32)
    shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32)
    shapes = np.flatnonzero((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0).astype(np.int32)
    if shapes.size == 0:
        return _empty_pairs(model.device)

    particles = np.arange(model.particle_count, dtype=np.int32)
    blocks: list[np.ndarray] = []

    global_shapes = shapes[shape_world[shapes] == -1]
    if global_shapes.size:
        blocks.append(
            np.column_stack((np.repeat(particles, global_shapes.size), np.tile(global_shapes, particles.size)))
        )

    global_particles = particles[particle_world == -1]
    local_shapes = shapes[shape_world[shapes] != -1]
    if global_particles.size and local_shapes.size:
        blocks.append(
            np.column_stack(
                (np.repeat(global_particles, local_shapes.size), np.tile(local_shapes, global_particles.size))
            )
        )

    for world in np.unique(particle_world[particle_world >= 0]):
        local_particles = particles[particle_world == world]
        world_shapes = shapes[shape_world[shapes] == world]
        if local_particles.size and world_shapes.size:
            blocks.append(
                np.column_stack(
                    (np.repeat(local_particles, world_shapes.size), np.tile(world_shapes, local_particles.size))
                )
            )

    if not blocks:
        return _empty_pairs(model.device)
    pairs = np.concatenate(blocks, axis=0)
    if model.device.is_cuda:
        pairs = pairs[np.argsort(pairs[:, 1], kind="stable")]
    return wp.array(pairs, dtype=wp.vec2i, device=model.device)


class MJVBDV2SoftContactPipeline:
    """Sparse particle-shape collision pass for the soft-only V2 path."""

    def __init__(self, model: Model, *, margin: float = 0.0):
        if margin < 0.0:
            raise ValueError("soft_contact_margin must be non-negative")
        self.model = model
        self.margin = float(margin)
        self.pairs = _build_particle_shape_pairs(model)
        self._empty_body_q = wp.empty(0, dtype=wp.transform, device=model.device)
        self.shape_aabb_lower = wp.empty(model.shape_count, dtype=wp.vec3, device=model.device)
        self.shape_aabb_upper = wp.empty(model.shape_count, dtype=wp.vec3, device=model.device)

    @property
    def pair_count(self) -> int:
        return int(self.pairs.shape[0])

    def contacts(self) -> Contacts:
        contacts = Contacts(
            rigid_contact_max=0,
            soft_contact_max=self.pair_count,
            soft_contact_tids_size=self.pair_count,
            requires_grad=self.model.requires_grad,
            device=self.model.device,
        )
        self.model._add_custom_attributes(
            contacts,
            Model.AttributeAssignment.CONTACT,
            requires_grad=self.model.requires_grad,
        )
        return contacts

    def collide(self, state: State, contacts: Contacts) -> None:
        if contacts.device != self.model.device:
            raise ValueError(f"MJVBDV2 contacts must reside on model device {self.model.device}, got {contacts.device}")
        if contacts.soft_contact_max < self.pair_count:
            raise ValueError(
                f"MJVBDV2 soft-contact buffer needs {self.pair_count} records, got {contacts.soft_contact_max}"
            )

        contacts.clear()
        if self.pair_count == 0:
            return
        if state.particle_q is None:
            raise ValueError("MJVBDV2 soft-only collision requires state.particle_q")
        body_q = state.body_q
        if body_q is None and self.model.body_count > 0:
            raise ValueError("MJVBDV2 soft-only collision requires state.body_q")
        if body_q is None:
            body_q = self._empty_body_q

        model = self.model
        wp.launch(
            kernel=compute_shape_world_aabbs,
            dim=model.shape_count,
            inputs=[
                body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_margin,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
            ],
            outputs=[self.shape_aabb_lower, self.shape_aabb_upper],
            device=model.device,
        )
        wp.launch(
            kernel=create_soft_contacts_with_aabb,
            dim=self.pair_count,
            inputs=[
                self.pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                model._shape_mesh_properties,
                model.shape_world,
                self.margin,
                model.shape_margin,
                contacts.soft_contact_max,
                model.shape_flags,
                model.shape_heightfield_index,
                model.heightfield_data,
                model.heightfield_elevations,
                self.shape_aabb_lower,
                self.shape_aabb_upper,
            ],
            outputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_tids,
            ],
            device=model.device,
        )


class MuJoCoVBDCollisionPipeline:
    """Cross-solver contact pipeline wrapper for the two-way backend (``DESIGN.md`` 12).

    Wraps the private full-surface :class:`MJVBDV2CollisionPipeline` and refreshes
    all V-V and M-V contacts every coupling iteration, never reusing stale
    narrow-phase results (``DESIGN.md`` requirement 8).
    """

    def __init__(
        self,
        model,
        ownership,
        routing,
        *,
        static_contact_owner=MuJoCoVBDStaticContactOwner.AUTO,
        **collision_options: object,
    ) -> None:
        from .full_contact_pipeline import MJVBDV2CollisionPipeline

        self.model = model
        self.ownership = ownership
        self.routing = routing
        self._construction_shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32).copy()

        # VBD owns V-V and M-V rigid contacts. Use an explicit immutable pair
        # stream so M-M/M-S contacts handled by MuJoCo cannot be emitted here.
        pair_chunks = []
        for pair_array in (routing.vbd_shape_pairs, routing.cross_shape_pairs):
            if pair_array.shape[0] > 0:
                pair_chunks.append(np.asarray(pair_array.numpy(), dtype=np.int32).reshape(-1, 2))
        pair_data = np.concatenate(pair_chunks, axis=0) if pair_chunks else np.empty((0, 2), dtype=np.int32)
        explicit_pairs = wp.array(pair_data, dtype=wp.vec2i, device=model.device)

        # Soft particles are VBD-owned, so particle-static contacts always stay
        # in VBD. All dynamic rigid shapes remain eligible for M-soft cross
        # contact and V-soft internal contact.
        collision_model = copy(model)

        options = dict(collision_options)
        options["broad_phase"] = "explicit"
        options["shape_pairs_filtered"] = explicit_pairs
        self._soft_contact_margin = float(options.pop("soft_contact_margin", 0.0) or 0.0)
        self._pipeline = MJVBDV2CollisionPipeline(collision_model, **options)
        self._contacts = self._pipeline.contacts()
        # Fixed device-side zero used for optional narrow-phase counters that
        # are not allocated for the current shape mix. Reusing an active
        # counter here would falsely report overflow in primitive-only scenes.
        self._zero_count = wp.zeros(1, dtype=wp.int32, device=model.device)
        self._matching_snapshot: list[tuple[wp.array, wp.array]] = []
        self._allocate_matching_snapshot()

    def _allocate_matching_snapshot(self) -> None:
        matcher = self._pipeline._contact_matcher
        if matcher is None:
            return
        arrays = [
            matcher._prev_sorted_keys,
            matcher._prev_count,
            matcher._prev_claim,
            matcher._prev_was_matched,
            matcher._reset_world_mask,
            matcher._sorter.scratch_pos_world,
            matcher._sorter.scratch_normal,
        ]
        for name in (
            "_prev_point0",
            "_prev_point1",
            "_prev_offset0",
            "_prev_offset1",
            "_prev_normal_sticky",
        ):
            array = getattr(matcher, name, None)
            if array is not None:
                arrays.append(array)
        self._matching_snapshot = [(array, wp.clone(array)) for array in arrays]

    def begin_substep(self) -> None:
        """Snapshot collision matching history before the first outer round."""
        for live, snapshot in self._matching_snapshot:
            wp.copy(snapshot, live)

    def contacts(self) -> Contacts:
        return self._contacts

    def validate_runtime_shape_flags(self) -> None:
        """Reject collision-topology growth beyond construction-time capacity."""
        current = np.asarray(self.model.shape_flags.numpy(), dtype=np.int32)
        collision_bits = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
        newly_enabled = (current & collision_bits) & ~(self._construction_shape_flags & collision_bits)
        changed_shapes = np.flatnonzero(newly_enabled != 0)
        if changed_shapes.size:
            raise RuntimeError(
                "Two-way collision flags enabled new construction-time candidates for shapes "
                f"{changed_shapes.tolist()}; rebuild SolverMuJoCoVBD so fixed contact streams, capacities, "
                "and CUDA Graph topology include them. Disabling or re-enabling an originally enabled "
                "collision flag remains supported."
            )

    def collide_iteration(self, state: State, contacts: Contacts, *, iteration: int) -> None:
        _ = iteration
        contacts.clear()
        self._pipeline.collide(state, contacts, soft_contact_margin=self._soft_contact_margin)

    def record_overflow(self, contacts: Contacts, diagnostics: MuJoCoVBDDiagnostics) -> None:
        """Latch every collision-pipeline overflow into public diagnostics.

        The shared collision pipeline prints warnings, but two-way coupling must
        expose the same counters programmatically so a failed outer iteration
        cannot be committed when ``fail_on_overflow`` is enabled.
        """
        rigid_flag = diagnostics.rigid_contact_overflow
        soft_flag = diagnostics.soft_contact_overflow
        if rigid_flag is None or soft_flag is None:
            return

        pipeline = self._pipeline
        narrow = pipeline.narrow_phase
        dummy = self._zero_count
        if narrow.split_gjk_mpr:
            split_query_count = (
                pipeline.broad_phase_pair_count if narrow.sparse_gjk_pairs else narrow.gjk_candidate_pairs_count
            )
            split_query_capacity = narrow.split_query_results.shape[0]
            split_gjk_count = narrow.split_gjk_work_count
            split_gjk_capacity = narrow.split_gjk_work_items.shape[0]
            split_manifold_count = narrow.split_manifold_work_count
            split_manifold_capacity = narrow.split_manifold_work_items.shape[0]
        else:
            split_query_count = dummy
            split_query_capacity = -1
            split_gjk_count = dummy
            split_gjk_capacity = -1
            split_manifold_count = dummy
            split_manifold_capacity = -1

        reducer = narrow.global_contact_reducer
        reduction_failures = reducer.ht_insert_failures if reducer is not None else dummy
        mesh_count = narrow.shape_pairs_mesh_count if narrow.shape_pairs_mesh_count is not None else dummy
        triangle_count = narrow.triangle_pairs_count if narrow.triangle_pairs_count is not None else dummy
        mesh_plane_count = (
            narrow.shape_pairs_mesh_plane_count if narrow.shape_pairs_mesh_plane_count is not None else dummy
        )
        mesh_mesh_count = (
            narrow.shape_pairs_mesh_mesh_count if narrow.shape_pairs_mesh_mesh_count is not None else dummy
        )
        sdf_sdf_count = narrow.shape_pairs_sdf_sdf_count if narrow.shape_pairs_sdf_sdf_count is not None else dummy
        wp.launch(
            _record_candidate_overflow_kernel,
            dim=1,
            inputs=[
                pipeline.broad_phase_pair_count,
                pipeline.broad_phase_shape_pairs.shape[0],
                split_query_count,
                split_query_capacity,
                narrow.gjk_candidate_pairs_count,
                narrow.gjk_candidate_pairs.shape[0],
                split_gjk_count,
                split_gjk_capacity,
                split_manifold_count,
                split_manifold_capacity,
                rigid_flag,
            ],
            device=self.model.device,
        )
        wp.launch(
            _record_contact_overflow_kernel,
            dim=1,
            inputs=[
                mesh_count,
                narrow.shape_pairs_mesh.shape[0] if narrow.shape_pairs_mesh is not None else 0,
                triangle_count,
                narrow.triangle_pairs.shape[0] if narrow.triangle_pairs is not None else 0,
                mesh_plane_count,
                narrow.shape_pairs_mesh_plane.shape[0] if narrow.shape_pairs_mesh_plane is not None else 0,
                mesh_mesh_count,
                narrow.shape_pairs_mesh_mesh.shape[0] if narrow.shape_pairs_mesh_mesh is not None else 0,
                sdf_sdf_count,
                narrow.shape_pairs_sdf_sdf.shape[0] if narrow.shape_pairs_sdf_sdf is not None else 0,
                contacts.rigid_contact_count,
                contacts.rigid_contact_max,
                reduction_failures,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                rigid_flag,
                soft_flag,
            ],
            device=self.model.device,
        )
        hydro = narrow.hydroelastic_sdf
        if hydro is not None:
            wp.launch(
                _record_hydro_overflow_kernel,
                dim=1,
                inputs=[
                    hydro.block_broad_collide_count,
                    hydro.max_num_blocks_broad,
                    hydro.iso_buffer_counts[1],
                    hydro.iso_max_dims[0],
                    hydro.iso_buffer_counts[2],
                    hydro.iso_max_dims[1],
                    hydro.iso_buffer_counts[3],
                    hydro.iso_max_dims[2],
                    hydro.iso_voxel_count,
                    hydro.max_num_iso_voxels,
                    hydro.contact_reduction.contact_count,
                    hydro.max_num_face_contacts,
                    hydro.contact_reduction.reducer.ht_insert_failures,
                    rigid_flag,
                ],
                device=self.model.device,
            )

    def restore_iteration(self, iteration: int) -> None:
        _ = iteration
        for live, snapshot in self._matching_snapshot:
            wp.copy(live, snapshot)

    def abort_substep(self) -> None:
        for live, snapshot in self._matching_snapshot:
            wp.copy(live, snapshot)

    def reset(self, world_mask: wp.array | None = None) -> None:
        self._contacts.clear()
        self._pipeline.reset_contact_matching(world_mask)

    def rebuild(self) -> None:
        rebuild = getattr(self._pipeline, "rebuild", None)
        if callable(rebuild):
            rebuild()

    def rebuild_dynamic_bvhs(self, state: State) -> None:
        rebuild = getattr(self._pipeline, "rebuild_bvh", None)
        if callable(rebuild):
            rebuild(state)
