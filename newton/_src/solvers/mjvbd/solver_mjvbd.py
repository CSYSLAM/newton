# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...geometry.kernels import create_soft_contacts
from ...sim import CollisionPipeline, Contacts, Control, Model, ModelBuilder, State
from .solver_mujoco import SolverMuJoCo as SolverMJVBDRigid
from .solver_vbd import SolverVBD

__all__ = ["SolverMJVBD"]


@wp.kernel
def _remap_soft_contact_shape_indices(
    soft_contact_count: wp.array[int],
    compact_shape_index_to_model_shape_index: wp.array[wp.int32],
    soft_contact_shape: wp.array[int],
):
    tid = wp.tid()
    if tid >= soft_contact_count[0]:
        return

    compact_shape_index = soft_contact_shape[tid]
    if compact_shape_index >= 0:
        soft_contact_shape[tid] = compact_shape_index_to_model_shape_index[compact_shape_index]


class SolverMJVBD:
    """Single-solver wrapper for copied MuJoCo rigid motion plus copied VBD particle solve.

    This wrapper preserves the existing MuJoCo+VBD Franka example flows while
    exposing a single solver object.

    External rigid mode:
        1. The caller updates the articulation pose in ``state_out``.
        2. MJVBD runs Newton's collision pipeline on ``state_out``.
        3. The copied VBD core advances particles from ``state_in`` to ``state_out``.

    Internal rigid mode:
        1. MJVBD temporarily disables particles and gravity.
        2. The copied MuJoCo solver advances rigid bodies from ``state_in`` to ``state_out``.
        3. MJVBD restores particle state, runs collision detection, and advances
           the copied VBD core.

    Args:
        model: Simulation model.
        rigid_contact_max: Maximum number of rigid-rigid contacts kept by the
            internal collision pipeline.
        soft_contact_margin: Particle-shape contact margin [cm].
        step_rigid_bodies: If ``True``, run the copied MuJoCo rigid solver
            inside :meth:`step`. If ``False``, assume the caller updates rigid
            bodies externally before calling :meth:`step`.
        rigid_njmax: MuJoCo constraint buffer size used when
            ``step_rigid_bodies`` is enabled.
        rigid_nconmax: MuJoCo contact buffer size used when
            ``step_rigid_bodies`` is enabled.
        **vbd_kwargs: Forwarded to the copied :class:`SolverVBD` core.
    """

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register copied MuJoCo custom attributes on a builder."""
        SolverMJVBDRigid.register_custom_attributes(builder)

    def __init__(
        self,
        model: Model,
        *,
        rigid_contact_max: int = 0,
        soft_contact_margin: float = 0.0,
        step_rigid_bodies: bool = False,
        rigid_njmax: int = 768,
        rigid_nconmax: int = 768,
        **vbd_kwargs,
    ):
        self.model = model
        self.step_rigid_bodies = step_rigid_bodies
        self.rigid_contact_max = rigid_contact_max
        self.soft_contact_margin = soft_contact_margin
        self._use_soft_contact_only_collision = True
        self.collision_pipeline = None
        self._init_soft_contact_shape_subset()
        soft_contact_max = self._soft_contact_shape_count * self.model.particle_count
        self.contacts = Contacts(
            rigid_contact_max=rigid_contact_max,
            soft_contact_max=soft_contact_max,
            requires_grad=self.model.requires_grad,
            device=self.model.device,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )
        self.model._add_custom_attributes(
            self.contacts,
            Model.AttributeAssignment.CONTACT,
            requires_grad=self.model.requires_grad,
        )
        self.particle_solver = SolverVBD(
            model,
            integrate_with_external_rigid_solver=True,
            **vbd_kwargs,
        )
        self.rigid_solver = None
        self.rigid_contacts = None
        if step_rigid_bodies:
            self.rigid_solver = SolverMJVBDRigid(
                model,
                use_mujoco_contacts=False,
                njmax=rigid_njmax,
                nconmax=rigid_nconmax,
            )
            self.rigid_contacts = Contacts(rigid_contact_max=0, soft_contact_max=0, device=model.device)

    @property
    def device(self):
        """Device used by the wrapped particle solver."""
        return self.particle_solver.device

    @property
    def particle_enable_self_contact(self) -> bool:
        """Whether particle self-contact is enabled in the copied VBD core."""
        return self.particle_solver.particle_enable_self_contact

    def rebuild_bvh(self, state: State) -> None:
        """Refresh the copied VBD core's self-contact BVH."""
        self.particle_solver.rebuild_bvh(state)

    def _init_soft_contact_shape_subset(self) -> None:
        """Compact particle-collidable shapes into a dedicated subset.

        This is behavior-preserving: ``create_soft_contacts()`` already ignores
        shapes without ``COLLIDE_PARTICLES``. Compacting here only reduces the
        launch width and the soft-contact buffer capacity.
        """

        shape_flags_np = self.model.shape_flags.numpy()
        collidable_shape_indices_np = np.flatnonzero((shape_flags_np & int(ShapeFlags.COLLIDE_PARTICLES)) != 0).astype(
            np.int32, copy=False
        )

        self._soft_contact_shape_count = int(collidable_shape_indices_np.size)
        self._compact_soft_contact_shapes = self._soft_contact_shape_count < int(self.model.shape_count)

        if self._soft_contact_shape_count == 0:
            self._soft_contact_shape_index_to_model_shape_index = wp.empty(0, dtype=wp.int32, device=self.model.device)
            self._soft_contact_shape_transform = wp.empty(0, dtype=wp.transform, device=self.model.device)
            self._soft_contact_shape_body = wp.empty(0, dtype=wp.int32, device=self.model.device)
            self._soft_contact_shape_type = wp.empty(0, dtype=wp.int32, device=self.model.device)
            self._soft_contact_shape_scale = wp.empty(0, dtype=wp.vec3, device=self.model.device)
            self._soft_contact_shape_source_ptr = wp.empty(0, dtype=wp.uint64, device=self.model.device)
            self._soft_contact_shape_world = wp.empty(0, dtype=wp.int32, device=self.model.device)
            self._soft_contact_shape_flags = wp.empty(0, dtype=wp.int32, device=self.model.device)
            self._soft_contact_shape_heightfield_index = wp.empty(0, dtype=wp.int32, device=self.model.device)
            return

        if not self._compact_soft_contact_shapes:
            self._soft_contact_shape_index_to_model_shape_index = None
            self._soft_contact_shape_transform = self.model.shape_transform
            self._soft_contact_shape_body = self.model.shape_body
            self._soft_contact_shape_type = self.model.shape_type
            self._soft_contact_shape_scale = self.model.shape_scale
            self._soft_contact_shape_source_ptr = self.model.shape_source_ptr
            self._soft_contact_shape_world = self.model.shape_world
            self._soft_contact_shape_flags = self.model.shape_flags
            self._soft_contact_shape_heightfield_index = self.model.shape_heightfield_index
            return

        self._soft_contact_shape_index_to_model_shape_index = wp.array(
            collidable_shape_indices_np, dtype=wp.int32, device=self.model.device
        )
        self._soft_contact_shape_transform = wp.array(
            self.model.shape_transform.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_transform.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_body = wp.array(
            self.model.shape_body.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_body.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_type = wp.array(
            self.model.shape_type.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_type.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_scale = wp.array(
            self.model.shape_scale.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_scale.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_source_ptr = wp.array(
            self.model.shape_source_ptr.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_source_ptr.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_world = wp.array(
            self.model.shape_world.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_world.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_flags = wp.array(
            self.model.shape_flags.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_flags.dtype,
            device=self.model.device,
        )
        self._soft_contact_shape_heightfield_index = wp.array(
            self.model.shape_heightfield_index.numpy()[collidable_shape_indices_np],
            dtype=self.model.shape_heightfield_index.dtype,
            device=self.model.device,
        )

    def _get_collision_pipeline(self) -> CollisionPipeline:
        """Lazily build the full collision pipeline for the fallback path."""
        if self.collision_pipeline is None:
            self.collision_pipeline = CollisionPipeline(
                self.model,
                rigid_contact_max=self.rigid_contact_max,
                soft_contact_max=self.contacts.soft_contact_max,
                soft_contact_margin=self.soft_contact_margin,
            )
            self.contacts = self.collision_pipeline.contacts()
        return self.collision_pipeline

    def _collide_soft_contacts_only(self, state: State) -> None:
        """Populate only particle-shape contacts for the copied MJVBD path."""
        contacts = self.contacts
        contacts.clear()

        if state.particle_q is None or self._soft_contact_shape_count == 0:
            return

        particle_count = len(state.particle_q)
        if particle_count == 0:
            return

        wp.launch(
            kernel=create_soft_contacts,
            dim=particle_count * self._soft_contact_shape_count,
            inputs=[
                state.particle_q,
                self.model.particle_radius,
                self.model.particle_flags,
                self.model.particle_world,
                state.body_q,
                self._soft_contact_shape_transform,
                self._soft_contact_shape_body,
                self._soft_contact_shape_type,
                self._soft_contact_shape_scale,
                self._soft_contact_shape_source_ptr,
                self._soft_contact_shape_world,
                self.soft_contact_margin,
                contacts.soft_contact_max,
                self._soft_contact_shape_count,
                self._soft_contact_shape_flags,
                self._soft_contact_shape_heightfield_index,
                self.model.heightfield_data,
                self.model.heightfield_elevations,
            ],
            outputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_tids,
            ],
            device=self.device,
        )

        if self._compact_soft_contact_shapes:
            wp.launch(
                kernel=_remap_soft_contact_shape_indices,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    self._soft_contact_shape_index_to_model_shape_index,
                    contacts.soft_contact_shape,
                ],
                device=self.device,
            )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        dt: float,
    ) -> None:
        """Advance one substep using the copied collision+VBD pipeline."""
        collision_state = state_out

        if self.step_rigid_bodies:
            self.rigid_solver.step(state_in, state_out, control, self.rigid_contacts, dt)
            collision_state = state_in

        if self._use_soft_contact_only_collision:
            self._collide_soft_contacts_only(collision_state)
        else:
            self._get_collision_pipeline().collide(collision_state, self.contacts)
        self.particle_solver.step(state_in, state_out, control, self.contacts, dt)
