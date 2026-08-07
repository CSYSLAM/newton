# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small, MJVBD-owned particle--rigid contact pass.

This deliberately does not instantiate :class:`CollisionPipeline`: MJVBD only
needs particle-to-shape contacts after the rigid pose is known, while MuJoCo
keeps ownership of rigid-to-rigid collision detection.  Keeping the candidate
list here also means that the VBD half never sees stale, pre-step rigid poses.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...geometry.kernels import create_soft_contacts
from ...sim import Contacts, Model, State

__all__ = ["MJVBDSoftContactPipeline"]


def _empty_pairs(device: wp.context.Devicelike) -> wp.array:
    return wp.array(np.empty((0, 2), dtype=np.int32), dtype=wp.vec2i, device=device)


def _build_particle_shape_pairs(model: Model) -> wp.array:
    """Return world-compatible ``(particle, shape)`` pairs.

    Unlike the general collision pipeline this filters shapes that cannot
    collide with particles before allocating the candidate list.  Particle
    active flags intentionally remain a device-side test in ``create_soft_contacts``:
    changing them is common and must not require a host rebuild.
    """
    if model.particle_count == 0 or model.shape_count == 0:
        return _empty_pairs(model.device)

    particle_world = np.asarray(model.particle_world.numpy(), dtype=np.int32)
    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int32)
    shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32)
    shapes = np.flatnonzero((shape_flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0).astype(np.int32)
    if shapes.size == 0:
        return _empty_pairs(model.device)

    particles = np.arange(model.particle_count, dtype=np.int32)
    pair_blocks: list[np.ndarray] = []

    # Global shapes apply to every particle, and global particles apply to all
    # particle-colliding shapes.  The local/local pass then has exactly one
    # compact block per populated world.
    global_shapes = shapes[shape_world[shapes] == -1]
    if global_shapes.size:
        pair_blocks.append(
            np.column_stack((np.repeat(particles, global_shapes.size), np.tile(global_shapes, particles.size)))
        )

    global_particles = particles[particle_world == -1]
    local_shapes = shapes[shape_world[shapes] != -1]
    if global_particles.size and local_shapes.size:
        pair_blocks.append(
            np.column_stack(
                (np.repeat(global_particles, local_shapes.size), np.tile(local_shapes, global_particles.size))
            )
        )

    for world in np.unique(particle_world[particle_world >= 0]):
        local_particles = particles[particle_world == world]
        world_shapes = shapes[shape_world[shapes] == world]
        if local_particles.size and world_shapes.size:
            pair_blocks.append(
                np.column_stack(
                    (np.repeat(local_particles, world_shapes.size), np.tile(world_shapes, local_particles.size))
                )
            )

    if not pair_blocks:
        return _empty_pairs(model.device)
    return wp.array(np.concatenate(pair_blocks, axis=0), dtype=wp.vec2i, device=model.device)


class MJVBDSoftContactPipeline:
    """Generate point particle--shape contacts against a supplied rigid state.

    The supplied ``particle_state`` and ``rigid_state`` are intentionally
    separate.  MJVBD passes the old particle state and newly integrated/FK'd
    rigid state, respectively, which is the one-way coupling contract.
    """

    def __init__(self, model: Model, *, margin: float = 0.0):
        if margin < 0.0:
            raise ValueError("soft_contact_margin must be non-negative")
        self.model = model
        self.margin = float(margin)
        self.pairs = _build_particle_shape_pairs(model)

    @property
    def pair_count(self) -> int:
        return int(self.pairs.shape[0])

    def rebuild(self) -> None:
        """Rebuild static candidate pairs after a shape/world topology update."""
        self.pairs = _build_particle_shape_pairs(self.model)

    def make_contacts(self) -> Contacts:
        """Allocate an MJVBD-owned contact buffer sized for every candidate."""
        # MJVBD never consumes Newton rigid contacts: MuJoCo solves those
        # internally and the external mode receives rigid poses from the caller.
        contacts = Contacts(
            rigid_contact_max=0,
            soft_contact_max=self.pair_count,
            soft_contact_tids_size=self.pair_count,
            requires_grad=self.model.requires_grad,
            device=self.model.device,
        )
        # Match Model.contacts()/CollisionPipeline.contacts(): contact-scoped
        # custom attributes are part of the Contacts lifetime, even though the
        # first MJVBD pass itself only consumes the built-in soft fields.
        self.model._add_custom_attributes(
            contacts, Model.AttributeAssignment.CONTACT, requires_grad=self.model.requires_grad
        )
        return contacts

    def validate_contacts(self, contacts: Contacts) -> None:
        if contacts.device != self.model.device:
            raise ValueError(
                f"MJVBD contacts must reside on the model device ({self.model.device}), got {contacts.device}."
            )
        if contacts.soft_contact_max < self.pair_count:
            raise ValueError(
                "MJVBD soft-contact buffer is too small for its sparse candidate set: "
                f"need at least {self.pair_count}, got {contacts.soft_contact_max}. "
                "Pass contacts=None or allocate a larger Contacts buffer."
            )

    def generate(self, particle_state: State, rigid_state: State, contacts: Contacts) -> None:
        """Clear and fill ``contacts`` using old particles and new rigid poses."""
        self.validate_contacts(contacts)
        contacts.clear()
        if self.pair_count == 0:
            return
        if particle_state.particle_q is None:
            raise ValueError("MJVBD requires state_in.particle_q when particles are present")
        if rigid_state.body_q is None:
            raise ValueError("MJVBD requires state_out.body_q to generate particle-rigid contacts")

        model = self.model
        wp.launch(
            kernel=create_soft_contacts,
            dim=self.pair_count,
            inputs=[
                self.pairs,
                particle_state.particle_q,
                model.particle_radius,
                model.particle_flags,
                model.particle_world,
                rigid_state.body_q,
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
