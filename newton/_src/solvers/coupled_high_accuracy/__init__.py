# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""High-accuracy coupled solver: one implicit Newton solve for particles, cloth, soft bodies, cables, granular and
fluid particles, and affine rigid bodies with joints and contacts.

The public names mirror the engineRTS ``style3d_sim`` package this solver was ported from, so that code written
against ``import style3d_sim as s3d`` works with ``from newton.solvers import coupled_high_accuracy as s3d``.
"""

from __future__ import annotations

import importlib
from typing import Any

from .core import (
    BACKEND,
    AffineBodyRigidityModel,
    Config,
    Contact,
    Control,
    InertiaMode,
    JointConstraintMode,
    LogLevel,
    Model,
    NonlinearSolver,
    Solver,
    Span,
    State,
    set_log_callback,
)

__all__ = [
    "BACKEND",
    "AffineBodyRigidityModel",
    "Config",
    "Contact",
    "Control",
    "InertiaMode",
    "JointConstraintMode",
    "LogLevel",
    "Model",
    "ModelBuilder",
    "NewtonSolver",
    "NonlinearSolver",
    "ParticleSurface",
    "ParticleSurfaceMesh",
    "Solver",
    "SolverCoupledHighAccuracy",
    "Span",
    "State",
    "configure_fluid",
    "configure_granular",
    "get_asset",
    "get_assets",
    "log_surface_mesh",
    "set_log_callback",
]

# Resolve the Newton-dependent helpers lazily so importing the core enums stays cheap.
_LAZY: dict[str, tuple[str, str]] = {
    "SolverCoupledHighAccuracy": (".solver_coupled_high_accuracy", "SolverCoupledHighAccuracy"),
    "NewtonSolver": (".solver_coupled_high_accuracy", "SolverCoupledHighAccuracy"),
    "ModelBuilder": (".builder", "ModelBuilder"),
    "ParticleSurface": (".particle_surface", "ParticleSurface"),
    "ParticleSurfaceMesh": (".particle_surface", "ParticleSurfaceMesh"),
    "log_surface_mesh": (".particle_surface", "log_surface_mesh"),
    "configure_granular": (".granular", "configure_granular"),
    "configure_fluid": (".fluid", "configure_fluid"),
    "get_asset": (".assets", "get_asset"),
    "get_assets": (".assets", "get_assets"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
