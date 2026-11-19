# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import coupled_high_accuracy, mujoco, style3d
    from .coupled_high_accuracy import SolverCoupledHighAccuracy
    from .featherstone import SolverFeatherstone
    from .implicit_mpm import SolverImplicitMPM
    from .kamino import SolverKamino
    from .mujoco.solver_mujoco import SolverMuJoCo
    from .semi_implicit import SolverSemiImplicit
    from .solver import SolverBase
    from .steel import SolverSteel
    from .style3d.solver_style3d import SolverStyle3D
    from .vbd import SolverVBD
    from .xpbd import SolverXPBD

__all__ = [
    "SolverBase",
    "SolverCoupledHighAccuracy",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverSemiImplicit",
    "SolverSteel",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
    "coupled_high_accuracy",
    "mujoco",
    "style3d",
]

# Maps each public symbol to the module that provides it and the attribute to
# fetch from that module (None returns the module itself). Symbols are
# resolved on first attribute access (PEP 562) so that importing Newton does
# not pay the import cost of every solver backend.
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    "SolverBase": (".solver", "SolverBase"),
    "SolverCoupledHighAccuracy": (".coupled_high_accuracy", "SolverCoupledHighAccuracy"),
    "SolverFeatherstone": (".featherstone", "SolverFeatherstone"),
    "SolverImplicitMPM": (".implicit_mpm", "SolverImplicitMPM"),
    "SolverKamino": (".kamino", "SolverKamino"),
    "SolverMuJoCo": (".mujoco.solver_mujoco", "SolverMuJoCo"),
    "SolverSemiImplicit": (".semi_implicit", "SolverSemiImplicit"),
    "SolverSteel": (".steel", "SolverSteel"),
    "SolverStyle3D": (".style3d.solver_style3d", "SolverStyle3D"),
    "SolverVBD": (".vbd", "SolverVBD"),
    "SolverXPBD": (".xpbd", "SolverXPBD"),
    "coupled_high_accuracy": (".coupled_high_accuracy", None),
    "mujoco": (".mujoco", None),
    "style3d": (".style3d", None),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
