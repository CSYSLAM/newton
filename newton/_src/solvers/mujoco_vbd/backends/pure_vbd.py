# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Pure VBD backends: only a private VBD core is constructed (``DESIGN.md`` 4.1).

``pure_vbd_soft`` uses the lightweight ``vbd_soft`` core; ``pure_vbd_full`` uses
the complete VBD/AVBD/pneumatic core. No MuJoCo model, coupling wrench, or
overlay is built.
"""

from __future__ import annotations

import warp as wp

from ....sim import Contacts, Control, ModelFlags, State, StateFlags
from ..config import MuJoCoVBDResolvedOptions
from ..diagnostics import allocate_diagnostics
from ..ownership import MuJoCoVBDOwnership
from .base import MuJoCoVBDBackendBase

__all__ = ["PureVBDFullBackend", "PureVBDSoftBackend"]


class _PureVBDBackend(MuJoCoVBDBackendBase):
    _core: str = "full"

    def __init__(self, model, ownership: MuJoCoVBDOwnership, options: MuJoCoVBDResolvedOptions) -> None:
        super().__init__(model, ownership, options)
        if self._core == "full":
            from ..vbd.solver_vbd import SolverVBD
        else:
            from ..vbd_soft.solver_vbd import SolverVBD

        external_rigid = not ownership.has_vbd_dynamic_bodies
        vbd_options = dict(options.vbd_options)
        requested_external = bool(vbd_options.pop("integrate_with_external_rigid_solver", external_rigid))
        if requested_external != external_rigid:
            raise ValueError(
                "integrate_with_external_rigid_solver is selected from model ownership; "
                f"expected {external_rigid}, got {requested_external}."
            )
        self._solver = SolverVBD(
            model,
            integrate_with_external_rigid_solver=external_rigid,
            **vbd_options,
        )

        contact_mode = options.contact_mode
        if contact_mode == "auto":
            contact_mode = "full" if ownership.has_vbd_dynamic_bodies or model.particle_count == 0 else "soft"
        collision_options = dict(options.collision_options)
        if contact_mode == "soft":
            from ..collision_pipeline import MJVBDV2SoftContactPipeline

            margin = float(collision_options.pop("soft_contact_margin", 0.0))
            if collision_options:
                raise ValueError(
                    "soft pure-VBD contact accepts only collision_options['soft_contact_margin']; "
                    f"unsupported keys are {sorted(collision_options)}"
                )
            self.pipeline = MJVBDV2SoftContactPipeline(model, margin=margin)
            self._contacts = self.pipeline.contacts()
        else:
            from ..full_contact_pipeline import MJVBDV2CollisionPipeline

            collision_options.setdefault("broad_phase", "nxn")
            collision_options.setdefault("include_static_kinematic_pairs", False)
            self.pipeline = MJVBDV2CollisionPipeline(model, **collision_options)
            self._contacts = self.pipeline.contacts()
        self._diagnostics = allocate_diagnostics(model, backend=None, feedback_enabled=False)

    @property
    def vbd_solver(self):
        return self._solver

    @property
    def contacts(self) -> Contacts | None:
        return self._contacts

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        selected_contacts = self._contacts if contacts is None else contacts
        self.pipeline.collide(state_in, selected_contacts)
        self._solver.step(state_in, state_out, control, selected_contacts, dt)

    def reset(self, state, world_mask: wp.array | None = None, flags: StateFlags | int | None = None) -> None:
        self._solver.reset(state, world_mask=world_mask, flags=flags)
        self._contacts.clear()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self._solver.notify_model_changed(flags)
        if int(flags) & (int(ModelFlags.SHAPE_PROPERTIES) | int(ModelFlags.MODEL_PROPERTIES)):
            rebuild = getattr(self.pipeline, "rebuild", None)
            if callable(rebuild):
                rebuild()

    def rebuild_bvh(self, state: State) -> None:
        rebuild = getattr(self._solver, "rebuild_bvh", None)
        if callable(rebuild):
            rebuild(state)


class PureVBDSoftBackend(_PureVBDBackend):
    _core = "soft"


class PureVBDFullBackend(_PureVBDBackend):
    _core = "full"
