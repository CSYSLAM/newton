# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2 collision-pipeline helpers."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...geometry.kernels import create_soft_contacts
from ...sim import Contacts, Model, State

__all__ = ["MJVBDV2SoftContactPipeline"]


def _empty_pairs(device: wp.context.Devicelike) -> wp.array:
    return wp.array(np.empty((0, 2), dtype=np.int32), dtype=wp.vec2i, device=device)


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
    return wp.array(np.concatenate(blocks, axis=0), dtype=wp.vec2i, device=model.device)


class MJVBDV2SoftContactPipeline:
    """Sparse particle-shape collision pass for the soft-only V2 path."""

    def __init__(self, model: Model, *, margin: float = 0.0):
        if margin < 0.0:
            raise ValueError("soft_contact_margin must be non-negative")
        self.model = model
        self.margin = float(margin)
        self.pairs = _build_particle_shape_pairs(model)

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
        if state.body_q is None:
            raise ValueError("MJVBDV2 soft-only collision requires state.body_q")

        model = self.model
        wp.launch(
            kernel=create_soft_contacts,
            dim=self.pair_count,
            inputs=[
                self.pairs,
                state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                state.body_q,
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
