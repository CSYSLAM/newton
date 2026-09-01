# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pure MuJoCo backend: only the private MuJoCo core is constructed (``DESIGN.md`` 4.1)."""

from __future__ import annotations

import warp as wp

from ....sim import Contacts, Control, ModelFlags, State, StateFlags
from ..config import MuJoCoVBDResolvedOptions
from ..diagnostics import allocate_diagnostics
from ..mujoco.solver_mujoco import SolverMuJoCo
from ..ownership import MuJoCoVBDOwnership
from .base import MuJoCoVBDBackendBase

__all__ = ["PureMuJoCoBackend"]


class PureMuJoCoBackend(MuJoCoVBDBackendBase):
    """Dynamic MuJoCo articulation only; no VBD, BVH, or coupling state."""

    def __init__(self, model, ownership: MuJoCoVBDOwnership, options: MuJoCoVBDResolvedOptions) -> None:
        super().__init__(model, ownership, options)
        self._solver = SolverMuJoCo(model, **dict(options.mujoco_options))
        self._diagnostics = allocate_diagnostics(model, backend=None, feedback_enabled=False)

    @property
    def mujoco_solver(self):
        return self._solver

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        self._solver.step(state_in, state_out, control, contacts, dt)

    def reset(self, state, world_mask: wp.array | None = None, flags: StateFlags | int | None = None) -> None:
        self._solver.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self._solver.notify_model_changed(flags)
