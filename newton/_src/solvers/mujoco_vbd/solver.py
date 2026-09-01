# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Public :class:`SolverMuJoCoVBD`: standalone multi-mode MuJoCo/VBD solver.

See ``DESIGN.md``. The constructor resolves ownership, discovers features, and
statically selects exactly one execution backend (pure VBD, pure MuJoCo,
kinematic passthrough, one-way, or two-way). All step/reset/model-update calls
delegate to that single backend; unused cores and buffers are never built.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import warp as wp

from ...sim import Contacts, Control, Model, ModelBuilder, ModelFlags, State, StateFlags
from ..solver import SolverBase
from .config import (
    MuJoCoVBDCouplingOptions,
    MuJoCoVBDResolvedOptions,
    validate_coupling_options,
)
from .dispatch import (
    MuJoCoVBDBackendKind,
    build_backend,
    discover_features,
    resolve_features,
    select_backend_kind,
)
from .mujoco.solver_mujoco import SolverMuJoCo
from .ownership import resolve_mujoco_vbd_ownership
from .vbd.solver_vbd import SolverVBD

__all__ = ["SolverMuJoCoVBD"]

_PURE_KINDS = frozenset(
    {
        MuJoCoVBDBackendKind.PURE_VBD_SOFT,
        MuJoCoVBDBackendKind.PURE_VBD_FULL,
        MuJoCoVBDBackendKind.PURE_MUJOCO,
        MuJoCoVBDBackendKind.KINEMATIC_PASSTHROUGH,
    }
)


class SolverMuJoCoVBD(SolverBase):
    """Standalone solver dispatching to one static MuJoCo/VBD backend (``DESIGN.md``)."""

    def __init__(
        self,
        model: Model,
        *,
        mujoco_articulations: Sequence[int] | None = None,
        mujoco_joints: Sequence[int] | None = None,
        joint_mode: Literal["dynamic", "kinematic"] = "dynamic",
        coupling_mode: Literal["auto", "one_way", "two_way"] = "auto",
        contact_mode: Literal["auto", "soft", "full"] = "auto",
        coupling_options: MuJoCoVBDCouplingOptions | Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        vbd_options: Mapping[str, object] | None = None,
        collision_options: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(model)

        if getattr(model, "requires_grad", False):
            raise NotImplementedError(
                "SolverMuJoCoVBD does not support differentiable simulation; the fixed-point "
                "coupling loop is not differentiable in this version (DESIGN 5.1/20.4)."
            )
        for name, value in (
            ("joint_mode", joint_mode),
            ("coupling_mode", coupling_mode),
            ("contact_mode", contact_mode),
        ):
            _validate_choice(name, value)

        coupling = validate_coupling_options(coupling_options)

        # -- ownership and static backend selection (DESIGN 5.2, 6, 4.1) --
        ownership = resolve_mujoco_vbd_ownership(
            model,
            mujoco_articulations=mujoco_articulations,
            mujoco_joints=mujoco_joints,
        )
        discovered = discover_features(model, ownership)
        kind = select_backend_kind(
            discovered,
            joint_mode=joint_mode,
            coupling_mode=coupling_mode,
            contact_mode=contact_mode,
        )

        if kind in _PURE_KINDS and coupling_options is not None:
            raise ValueError(
                f"backend {kind.value!r} does not use coupling options; pass coupling_options=None "
                "so the configuration is not silently ignored (DESIGN 5.1)."
            )

        self._features = resolve_features(
            kind,
            discovered,
            joint_mode=joint_mode,
            coupling_mode=coupling_mode,
            contact_mode=contact_mode,
        )
        self._ownership = ownership

        options = MuJoCoVBDResolvedOptions(
            coupling=coupling,
            joint_mode=joint_mode,
            coupling_mode=coupling_mode,
            contact_mode=contact_mode,
            mujoco_options=dict(mujoco_options or {}),
            vbd_options=dict(vbd_options or {}),
            collision_options=dict(collision_options or {}),
        )
        self._backend = build_backend(kind, model, ownership, options)

    # -- construction-time attribute registration (DESIGN 5.2) --

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register attributes for the private MuJoCo, full VBD, and soft VBD cores."""
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)
        from .vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

        SolverVBDSoft.register_custom_attributes(builder)

    # -- read-only diagnostics (DESIGN 5.3) --

    @property
    def features(self):
        return self._features

    @property
    def backend_kind(self) -> MuJoCoVBDBackendKind:
        return self._features.backend

    @property
    def ownership(self):
        return self._ownership

    @property
    def contacts(self) -> Contacts | None:
        return self._backend.contacts

    @property
    def diagnostics(self):
        return self._backend.diagnostics

    @property
    def mujoco_solver(self) -> SolverMuJoCo | None:
        return self._backend.mujoco_solver

    @property
    def vbd_solver(self) -> SolverVBD | None:
        return self._backend.vbd_solver

    # -- delegation to the selected backend --

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        self._backend.step(state_in, state_out, control, contacts, dt)

    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self._backend.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self._backend.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        self._backend.rebuild_bvh(state)


def _validate_choice(name: str, value: str) -> None:
    valid = {
        "joint_mode": {"dynamic", "kinematic"},
        "coupling_mode": {"auto", "one_way", "two_way"},
        "contact_mode": {"auto", "soft", "full"},
    }[name]
    if value not in valid:
        raise ValueError(f"{name} must be one of {sorted(valid)}, got {value!r}")
