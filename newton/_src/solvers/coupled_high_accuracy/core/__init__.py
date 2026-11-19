# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver core of the high-accuracy coupled solver.

Two interchangeable backends expose the same API (``Model``, ``State``, ``Control``, ``Contact``, ``Config``,
``Span``, ``Solver`` and the enums):

* ``warp`` (default): the pure Warp implementation in this package.
* ``native``: the original CUDA extension, kept as a comparison oracle (see :mod:`.native`); it needs a local
  engineRTS build.

The backend is chosen once per process with ``NEWTON_CHA_BACKEND``.
"""

from __future__ import annotations

import os

BACKEND = os.environ.get("NEWTON_CHA_BACKEND", "warp").strip().lower()

if BACKEND == "native":
    from .native import load as _load_native

    _core = _load_native()
elif BACKEND == "warp":
    from . import api as _core
else:
    raise ValueError(f"NEWTON_CHA_BACKEND must be 'warp' or 'native', got {BACKEND!r}")

AffineBodyRigidityModel = _core.AffineBodyRigidityModel
Config = _core.Config
Contact = _core.Contact
Control = _core.Control
InertiaMode = _core.InertiaMode
JointConstraintMode = _core.JointConstraintMode
LogLevel = _core.LogLevel
Model = _core.Model
NonlinearSolver = _core.NonlinearSolver
Solver = _core.Solver
Span = _core.Span
State = _core.State
set_log_callback = _core.set_log_callback

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
    "NonlinearSolver",
    "Solver",
    "Span",
    "State",
    "set_log_callback",
]
