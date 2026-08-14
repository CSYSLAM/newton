# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Source for the detailed solver guide: docs/solvers/index.rst
"""
Solvers integrate the dynamics of a :class:`~newton.Model` through the common
:class:`~newton.solvers.SolverBase` interface. Newton provides backends for
rigid articulated systems, maximal-coordinate constraints, particles, and
deformable simulation.

For solver-selection guidance and the feature, contact-material, joint-support,
and differentiability comparisons, see the :doc:`Solvers guide </solvers/index>`.
Installed-wheel users can use the stable hosted guide at
https://newton-physics.github.io/newton/stable/solvers/index.html.
"""

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from ._src import solvers as _solvers

if TYPE_CHECKING:
    from ._src.solvers import *  # noqa: F403

__all__ = [*_solvers.__all__, "experimental"]  # noqa: PLE0604


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(_solvers, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


class _LazySolverModule(ModuleType):
    """Expose a solver helper namespace without importing its implementation eagerly."""

    def __init__(self, name: str, solver_attr: str):
        super().__init__(name)
        self._solver_attr = solver_attr
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = getattr(_solvers, self._solver_attr)
            self.__doc__ = self._module.__doc__
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


class _LazyCoupledModule(ModuleType):
    def _load(self) -> ModuleType:
        module = importlib.import_module("._src.solvers.coupled", __package__)
        experimental.coupled = module
        sys.modules[self.__name__] = module
        return module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> list[str]:
        return dir(self._load())


experimental = ModuleType(f"{__name__}.experimental")
experimental.__doc__ = """Experimental solver namespaces.

.. experimental::
"""
experimental.__all__ = ["coupled"]
experimental.__path__ = []
experimental.coupled = _LazyCoupledModule(f"{__name__}.experimental.coupled")

for _solver_module_name in ("sph", "style3d"):
    _solver_module = _LazySolverModule(f"{__name__}.{_solver_module_name}", _solver_module_name)
    globals()[_solver_module_name] = _solver_module
    sys.modules[_solver_module.__name__] = _solver_module

sys.modules[f"{__name__}.experimental"] = experimental
sys.modules[f"{__name__}.experimental.coupled"] = experimental.coupled
