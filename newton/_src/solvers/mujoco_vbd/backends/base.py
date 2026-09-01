# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Common base for the mutually-exclusive execution backends (``DESIGN.md`` 4.1)."""

from __future__ import annotations

import warp as wp

from ....sim import Contacts, Control, ModelFlags, State, StateFlags
from ...solver import SolverBase
from ..config import MuJoCoVBDResolvedOptions
from ..ownership import MuJoCoVBDOwnership

__all__ = ["MuJoCoVBDBackendBase"]


class MuJoCoVBDBackendBase(SolverBase):
    """Shared storage and default no-op properties for a single backend.

    Each concrete backend constructs only the modules it needs; unavailable
    solvers and buffers report ``None`` (``DESIGN.md`` 5.3).
    """

    def __init__(
        self,
        model,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDResolvedOptions,
    ) -> None:
        super().__init__(model)
        self.ownership = ownership
        self.options = options
        self._diagnostics = None

    # -- read-only accessors (overridden where a module exists) --

    @property
    def contacts(self) -> Contacts | None:
        return None

    @property
    def mujoco_solver(self):
        return None

    @property
    def vbd_solver(self):
        return None

    @property
    def diagnostics(self):
        return self._diagnostics

    # -- lifecycle defaults --

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        raise NotImplementedError

    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        pass

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        pass

    def rebuild_bvh(self, state: State) -> None:
        pass

    # -- helpers --

    @staticmethod
    def _copy_public_state(dst: State, src: State) -> None:
        for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
            s = getattr(src, name, None)
            d = getattr(dst, name, None)
            if s is not None and d is not None and s.shape == d.shape and s.shape[0] > 0:
                wp.copy(d, s)
