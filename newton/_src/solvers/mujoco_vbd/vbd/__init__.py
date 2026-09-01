# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pneumatic import (
        PneumaticCavityHandle,
        PneumaticConfig,
        PneumaticMode,
        add_inflatable_mesh,
        add_pneumatic_cavity,
        register_pneumatic_attributes,
    )
    from .solver_vbd import SolverVBD

__all__ = [
    "PneumaticCavityHandle",
    "PneumaticConfig",
    "PneumaticMode",
    "SolverVBD",
    "add_inflatable_mesh",
    "add_pneumatic_cavity",
    "register_pneumatic_attributes",
]

_LAZY_IMPORTS = {
    "PneumaticCavityHandle": (".pneumatic", "PneumaticCavityHandle"),
    "PneumaticConfig": (".pneumatic", "PneumaticConfig"),
    "PneumaticMode": (".pneumatic", "PneumaticMode"),
    "SolverVBD": (".solver_vbd", "SolverVBD"),
    "add_inflatable_mesh": (".pneumatic", "add_inflatable_mesh"),
    "add_pneumatic_cavity": (".pneumatic", "add_pneumatic_cavity"),
    "register_pneumatic_attributes": (".pneumatic", "register_pneumatic_attributes"),
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
