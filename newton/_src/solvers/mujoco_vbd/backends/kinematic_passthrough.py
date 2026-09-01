# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kinematic passthrough backend: no solver is constructed (``DESIGN.md`` 4.1).

Only externally prescribed articulation is present and there is no VBD dynamics,
so the step forwards the caller-provided ``state_in`` pose/velocity to
``state_out`` without integrating any dynamics.
"""

from __future__ import annotations

from ....sim import Contacts, Control, State
from ..config import MuJoCoVBDResolvedOptions
from ..ownership import MuJoCoVBDOwnership
from .base import MuJoCoVBDBackendBase

__all__ = ["KinematicPassthroughBackend"]


class KinematicPassthroughBackend(MuJoCoVBDBackendBase):
    """Forward the externally prescribed state; construct nothing else."""

    def __init__(self, model, ownership: MuJoCoVBDOwnership, options: MuJoCoVBDResolvedOptions) -> None:
        super().__init__(model, ownership, options)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if contacts is not None:
            raise ValueError("kinematic_passthrough only accepts contacts=None (DESIGN 5.2).")
        self._copy_public_state(state_out, state_in)
