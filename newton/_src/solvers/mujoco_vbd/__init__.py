# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone multi-mode MuJoCo/VBD solver package.

See ``DESIGN.md``. This package holds a fully private MuJoCo, full VBD/AVBD, and
lightweight ``vbd_soft`` core and a static backend dispatcher. It has no runtime
dependency on the shared ``coupled`` package or on ``mjvbd_v2`` (DESIGN 3.1).

Imports are lazy so that pulling in a private sub-core (for example
``mujoco_vbd.vbd``) does not force construction of the whole dispatch layer.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import (
        MuJoCoVBDCouplingMode,
        MuJoCoVBDCouplingOptions,
        MuJoCoVBDRelaxation,
        MuJoCoVBDStaticContactOwner,
    )
    from .diagnostics import MuJoCoVBDDiagnostics
    from .dispatch import MuJoCoVBDBackendKind, MuJoCoVBDFeatures
    from .solver import SolverMuJoCoVBD

__all__ = [
    "MuJoCoVBDBackendKind",
    "MuJoCoVBDCouplingMode",
    "MuJoCoVBDCouplingOptions",
    "MuJoCoVBDDiagnostics",
    "MuJoCoVBDFeatures",
    "MuJoCoVBDRelaxation",
    "MuJoCoVBDStaticContactOwner",
    "SolverMuJoCoVBD",
]

_LAZY_IMPORTS = {
    "MuJoCoVBDBackendKind": (".dispatch", "MuJoCoVBDBackendKind"),
    "MuJoCoVBDCouplingMode": (".config", "MuJoCoVBDCouplingMode"),
    "MuJoCoVBDCouplingOptions": (".config", "MuJoCoVBDCouplingOptions"),
    "MuJoCoVBDDiagnostics": (".diagnostics", "MuJoCoVBDDiagnostics"),
    "MuJoCoVBDFeatures": (".dispatch", "MuJoCoVBDFeatures"),
    "MuJoCoVBDRelaxation": (".config", "MuJoCoVBDRelaxation"),
    "MuJoCoVBDStaticContactOwner": (".config", "MuJoCoVBDStaticContactOwner"),
    "SolverMuJoCoVBD": (".solver", "SolverMuJoCoVBD"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(__all__)
