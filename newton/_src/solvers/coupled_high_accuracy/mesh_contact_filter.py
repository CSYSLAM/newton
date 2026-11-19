# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Force selection for Newton's full-surface mesh contact records.

Newton's mesh contact detection reports every vertex-face, face-vertex and edge-edge pair within the contact band,
including several representations of one surface patch. Its contract is that solvers apply forces only to records
accepted by ``_soft_contact_force_enabled`` (feature ownership, resolved sign, active particles) and keep the rest for
penetration prevention, which this penalty solver does not use. The solver core treats every record it receives as a
force-carrying contact, so the accepted records are compacted, in record order, into buffers handed to the core; a
contact-force observable is scattered back to Newton's record indices after the step.

The selection is evaluated once per step on the positions the records were detected at, so the contact set stays
fixed during the step's Newton iterations.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from newton._src.geometry.soft_contacts_mesh import (
    _make_soft_contact_filter_data,
    _soft_contact_force_enabled,
    _SoftContactFilterData,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _select_force_records(
    data: _SoftContactFilterData,
    count: wp.array[wp.int32],
    capacity: int,
    selected: wp.array[wp.int32],
):
    i = wp.tid()
    keep = 0
    if i < wp.min(wp.max(count[0], 0), capacity):
        if _soft_contact_force_enabled(i, data):
            keep = 1
    selected[i] = keep


@wp.kernel
def _compact_force_records(
    selected: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    capacity: int,
    particle: wp.array[wp.int32],
    indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shape: wp.array[wp.int32],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    out_count: wp.array[wp.int32],
    out_particle: wp.array[wp.int32],
    out_indices: wp.array[wp.vec3i],
    out_barycentric: wp.array[wp.vec3],
    out_shape: wp.array[wp.int32],
    out_body_pos: wp.array[wp.vec3],
    out_body_vel: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
    source: wp.array[wp.int32],
):
    i = wp.tid()
    if i == capacity - 1:
        out_count[0] = offsets[i] + selected[i]
    if selected[i] == 0:
        return
    j = offsets[i]
    if particle.shape[0] > 0:
        out_particle[j] = particle[i]
    out_indices[j] = indices[i]
    out_barycentric[j] = barycentric[i]
    out_shape[j] = shape[i]
    out_body_pos[j] = body_pos[i]
    out_body_vel[j] = body_vel[i]
    out_normal[j] = normal[i]
    source[j] = i


@wp.kernel
def _scatter_soft_contact_forces(
    compact: wp.array[wp.spatial_vector],
    count: wp.array[wp.int32],
    source: wp.array[wp.int32],
    rigid_capacity: int,
    output: wp.array[wp.spatial_vector],
):
    j = wp.tid()
    if j < count[0]:
        output[rigid_capacity + source[j]] = compact[rigid_capacity + j]


class MeshContactFilter:
    """Owns the compacted soft-contact buffers for one solver."""

    def __init__(self, device: Any):
        self.device = device
        self.capacity = 0
        self._empty_int = wp.empty(0, dtype=wp.int32, device=device)

    def _reserve(self, capacity: int) -> None:
        if capacity <= self.capacity:
            return
        z = lambda dtype: wp.zeros(capacity, dtype=dtype, device=self.device)  # noqa: E731
        self.capacity = capacity
        self.selected, self.offsets, self.source = z(wp.int32), z(wp.int32), z(wp.int32)
        self.count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.particle, self.indices, self.barycentric = z(wp.int32), z(wp.vec3i), z(wp.vec3)
        self.shape, self.body_pos, self.body_vel, self.normal = z(wp.int32), z(wp.vec3), z(wp.vec3), z(wp.vec3)

    @staticmethod
    def applies(contacts: Any) -> bool:
        """Only full-surface mesh records carry the redundant representations to select from."""
        return getattr(contacts, "_soft_contact_mesh_features", None) is not None

    def compact(self, model: Any, contacts: Any, particle_q: wp.array, body_q: wp.array | None) -> None:
        capacity = int(contacts.soft_contact_max)
        self._reserve(capacity)
        data = _make_soft_contact_filter_data(model, contacts, particle_q, body_q)
        wp.launch(
            _select_force_records,
            dim=capacity,
            inputs=[data, contacts.soft_contact_count, capacity, self.selected],
            device=self.device,
        )
        wp.utils.array_scan(self.selected, self.offsets, inclusive=False)
        particle = getattr(contacts, "soft_contact_particle", None)
        wp.launch(
            _compact_force_records,
            dim=capacity,
            inputs=[
                self.selected,
                self.offsets,
                capacity,
                particle if particle is not None else self._empty_int,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                self.count,
                self.particle,
                self.indices,
                self.barycentric,
                self.shape,
                self.body_pos,
                self.body_vel,
                self.normal,
                self.source,
            ],
            device=self.device,
        )

    def scatter_forces(self, compact: wp.array, rigid_capacity: int, output: wp.array) -> None:
        """Copy rigid wrenches and move the selected soft wrenches back to their Newton record slots."""
        wp.copy(output, compact, count=rigid_capacity)
        output[rigid_capacity:].zero_()
        wp.launch(
            _scatter_soft_contact_forces,
            dim=self.capacity,
            inputs=[compact, self.count, self.source, rigid_capacity, output],
            device=self.device,
        )
