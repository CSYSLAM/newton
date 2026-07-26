# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-way MuJoCo / VBD coupled solver.

``SolverMJVBD`` is intentionally a composition rather than a new monolithic
solver.  It advances rigid bodies either through an external FK/IK producer or
through the private MuJoCo snapshot, then creates particle contacts at that new
rigid pose and lets the private VBD snapshot advance the soft bodies.  No cloth
reaction force is sent back to the rigid solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
from ..solver import SolverBase
from .mujoco.solver_mujoco import SolverMuJoCo
from .soft_contact_pipeline import MJVBDSoftContactPipeline
from .vbd.solver_vbd import SolverVBD

__all__ = ["SolverMJVBD"]


class SolverMJVBD(SolverBase):
    """One-way rigid-to-soft MJVBD coupling.

    .. experimental::
        SolverMJVBD's API and behavior may change without prior notice.

    ``rigid_mode='external'`` expects the caller to populate the rigid and
    joint fields of ``state_out`` (normally using FK after an IK solve) before
    each call to :meth:`step`.  ``rigid_mode='mujoco'`` advances them internally
    with the private MuJoCo solver.  In both modes VBD reads particles from
    ``state_in`` and rigid poses from ``state_out``; this avoids the one-frame
    contact lag of a conventional pre-step collision pass.
    """

    def __init__(
        self,
        model: Model,
        *,
        rigid_mode: Literal["external", "mujoco"] = "external",
        soft_contact_margin: float = 0.0,
        vbd_options: Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(model)
        if rigid_mode not in ("external", "mujoco"):
            raise ValueError("rigid_mode must be either 'external' or 'mujoco'")
        self.rigid_mode: Literal["external", "mujoco"] = rigid_mode

        vbd_kwargs = dict(vbd_options or {})
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", True)
        if requested_external is not True:
            raise ValueError(
                "SolverMJVBD always uses one-way coupling; "
                "vbd_options['integrate_with_external_rigid_solver'] must be True."
            )
        self.vbd_solver = SolverVBD(model, integrate_with_external_rigid_solver=True, **vbd_kwargs)

        self.mujoco_solver = None
        if rigid_mode == "mujoco":
            mujoco_kwargs = dict(mujoco_options or {})
            if mujoco_kwargs.pop("use_mujoco_contacts", True) is not True:
                raise ValueError("SolverMJVBD requires MuJoCo native rigid contacts (use_mujoco_contacts=True).")
            if mujoco_kwargs.pop("disable_contacts", False):
                raise ValueError("SolverMJVBD requires MuJoCo rigid contacts (disable_contacts=False).")
            self.mujoco_solver = SolverMuJoCo(model, use_mujoco_contacts=True, **mujoco_kwargs)

        self.soft_contact_pipeline = MJVBDSoftContactPipeline(model, margin=soft_contact_margin)
        self._contacts = self.soft_contact_pipeline.make_contacts()

    @property
    def contacts(self) -> Contacts:
        """MJVBD-owned soft-contact buffer.

        This is useful for callers that prefer an explicit step signature,
        e.g. ``solver.step(state_a, state_b, control, solver.contacts, dt)``.
        It is recreated after a shape/model notification that changes the
        sparse candidate topology; do not retain it across such a notification.
        """
        return self._contacts

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register custom attributes required by the private solver snapshots."""
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        SolverMuJoCo.register_custom_attributes(builder)

    def _select_contacts(self, contacts: Contacts | None) -> Contacts:
        selected = self._contacts if contacts is None else contacts
        self.soft_contact_pipeline.validate_contacts(selected)
        return selected

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one substep using new rigid poses and old particle positions.

        When the rigid mode is external, this method does not copy or integrate
        body fields.  The caller owns ``state_out.body_q`` / ``body_qd`` (and
        usually joint fields) and must write them before this call.
        """
        if dt <= 0.0:
            raise ValueError("MJVBD timestep dt must be positive")
        soft_contacts = self._select_contacts(contacts)

        if self.mujoco_solver is not None:
            # Native MuJoCo contacts never enter VBD: VBD is external-rigid
            # mode and consumes only the soft records generated below.
            self.mujoco_solver.step(state_in, state_out, control, None, dt)

        # This deliberately clears any caller-provided Contacts object.  The
        # object represents this MJVBD substep's soft contacts, not a stale
        # pre-step collision pass based on state_in.body_q.
        self.soft_contact_pipeline.generate(state_in, state_out, soft_contacts)
        self.vbd_solver.step(state_in, state_out, control, soft_contacts, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.vbd_solver.reset(state, world_mask=world_mask, flags=flags)
        if self.mujoco_solver is not None:
            self.mujoco_solver.reset(state, world_mask=world_mask, flags=flags)
        self._contacts.clear()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.vbd_solver.notify_model_changed(flags)
        if self.mujoco_solver is not None:
            self.mujoco_solver.notify_model_changed(flags)
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
            # Shape flags and world assignments decide candidate topology.
            # Recreate only MJVBD-owned contacts; an externally supplied buffer
            # is revalidated on the next step.
            self.soft_contact_pipeline.rebuild()
            self._contacts = self.soft_contact_pipeline.make_contacts()

    def rebuild_bvh(self, state: State) -> None:
        """Forward a soft self-contact BVH rebuild to the private VBD solver."""
        self.vbd_solver.rebuild_bvh(state)
