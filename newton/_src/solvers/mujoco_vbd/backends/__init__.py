# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Mutually-exclusive execution backends for :class:`SolverMuJoCoVBD`.

``construct`` builds exactly the one backend selected by ``dispatch`` so unused
cores and buffers are never allocated (``DESIGN.md`` 4.1).
"""

from __future__ import annotations

from ..config import MuJoCoVBDResolvedOptions
from ..dispatch import MuJoCoVBDBackendKind
from ..ownership import MuJoCoVBDOwnership
from .base import MuJoCoVBDBackendBase

__all__ = ["MuJoCoVBDBackendBase", "construct"]


def construct(
    kind: MuJoCoVBDBackendKind,
    model,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDResolvedOptions,
) -> MuJoCoVBDBackendBase:
    """Instantiate the single backend for ``kind`` (``DESIGN.md`` 4.1)."""
    K = MuJoCoVBDBackendKind

    if kind is K.PURE_MUJOCO:
        from .pure_mujoco import PureMuJoCoBackend

        return PureMuJoCoBackend(model, ownership, options)
    if kind is K.PURE_VBD_SOFT:
        from .pure_vbd import PureVBDSoftBackend

        return PureVBDSoftBackend(model, ownership, options)
    if kind is K.PURE_VBD_FULL:
        from .pure_vbd import PureVBDFullBackend

        return PureVBDFullBackend(model, ownership, options)
    if kind is K.KINEMATIC_PASSTHROUGH:
        from .kinematic_passthrough import KinematicPassthroughBackend

        return KinematicPassthroughBackend(model, ownership, options)
    if kind is K.ONE_WAY_KINEMATIC_SOFT:
        from .one_way_kinematic_soft import OneWayKinematicSoftBackend

        return OneWayKinematicSoftBackend(model, ownership, options)
    if kind is K.ONE_WAY_KINEMATIC_FULL:
        from .one_way_kinematic_full import OneWayKinematicFullBackend

        return OneWayKinematicFullBackend(model, ownership, options)
    if kind is K.ONE_WAY_DYNAMIC_SOFT:
        from .one_way_dynamic_soft import OneWayDynamicSoftBackend

        return OneWayDynamicSoftBackend(model, ownership, options)
    if kind is K.ONE_WAY_DYNAMIC_FULL:
        from .one_way_dynamic_full import OneWayDynamicFullBackend

        return OneWayDynamicFullBackend(model, ownership, options)
    if kind is K.TWO_WAY:
        from .two_way import TwoWayBackend

        return TwoWayBackend(model, ownership, options)

    raise ValueError(f"unknown backend kind {kind!r}")
