# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver_dispatch import SolverMJVBDV2
    from .vbd.pneumatic import (
        PneumaticCavityHandle,
        PneumaticConfig,
        PneumaticMode,
        add_inflatable_mesh,
        add_pneumatic_cavity,
        register_pneumatic_attributes,
    )

__all__ = [
    "PneumaticCavityHandle",
    "PneumaticConfig",
    "PneumaticMode",
    "SolverMJVBDV2",
    "add_inflatable_mesh",
    "add_pneumatic_cavity",
    "register_pneumatic_attributes",
]

_LAZY_IMPORTS = {
    "PneumaticCavityHandle": (".vbd.pneumatic", "PneumaticCavityHandle"),
    "PneumaticConfig": (".vbd.pneumatic", "PneumaticConfig"),
    "PneumaticMode": (".vbd.pneumatic", "PneumaticMode"),
    "SolverMJVBDV2": (".solver_dispatch", "SolverMJVBDV2"),
    "add_inflatable_mesh": (".vbd.pneumatic", "add_inflatable_mesh"),
    "add_pneumatic_cavity": (".vbd.pneumatic", "add_pneumatic_cavity"),
    "register_pneumatic_attributes": (".vbd.pneumatic", "register_pneumatic_attributes"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
