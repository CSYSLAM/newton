# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public configuration types and validation for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` section 5.1. Public enums use Python :class:`enum.Enum`; the
stable integer constants below are what the Warp kernels consume.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum

__all__ = [
    "PROXY_RESPONSE_DIRICHLET",
    "PROXY_RESPONSE_EFFECTIVE_MASS",
    "RELAXATION_AITKEN",
    "RELAXATION_FIXED",
    "STATIC_OWNER_AUTO",
    "STATIC_OWNER_MUJOCO",
    "STATIC_OWNER_VBD",
    "MuJoCoVBDCouplingMode",
    "MuJoCoVBDCouplingOptions",
    "MuJoCoVBDRelaxation",
    "MuJoCoVBDResolvedOptions",
    "MuJoCoVBDStaticContactOwner",
    "validate_coupling_options",
]


class MuJoCoVBDCouplingMode(Enum):
    """Requested coupling topology between the MuJoCo and VBD cores."""

    AUTO = "auto"
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"


class MuJoCoVBDRelaxation(Enum):
    """Interface wrench relaxation scheme across coupling iterations."""

    FIXED = "fixed"
    AITKEN = "aitken"


class MuJoCoVBDStaticContactOwner(Enum):
    """Owner for robot/VBD vs. static-shape contact pairs."""

    AUTO = "auto"
    MUJOCO = "mujoco"
    VBD = "vbd"


# Stable integer constants handed to Warp kernels. These must not change value.
PROXY_RESPONSE_EFFECTIVE_MASS = 0
PROXY_RESPONSE_DIRICHLET = 1

RELAXATION_FIXED = 0
RELAXATION_AITKEN = 1

STATIC_OWNER_AUTO = 0
STATIC_OWNER_MUJOCO = 1
STATIC_OWNER_VBD = 2

_RELAXATION_TO_INT = {
    MuJoCoVBDRelaxation.FIXED: RELAXATION_FIXED,
    MuJoCoVBDRelaxation.AITKEN: RELAXATION_AITKEN,
}

_STATIC_OWNER_TO_INT = {
    MuJoCoVBDStaticContactOwner.AUTO: STATIC_OWNER_AUTO,
    MuJoCoVBDStaticContactOwner.MUJOCO: STATIC_OWNER_MUJOCO,
    MuJoCoVBDStaticContactOwner.VBD: STATIC_OWNER_VBD,
}


@dataclass(frozen=True)
class MuJoCoVBDCouplingOptions:
    """Complete coupling configuration (see ``DESIGN.md`` section 5.1)."""

    iterations: int = 4
    relaxation: MuJoCoVBDRelaxation | str = MuJoCoVBDRelaxation.AITKEN
    relaxation_initial: float = 0.5
    relaxation_min: float = 0.05
    relaxation_max: float = 1.0
    force_absolute_tolerance: float = 1.0e-3
    force_relative_tolerance: float = 1.0e-3
    velocity_tolerance: float = 1.0e-4
    proxy_mass_scale: float = 1.0
    proxy_mass_min: float = 1.0e-6
    proxy_mass_max: float = 1.0e6
    proxy_inertia_eigenvalue_min: float = 1.0e-8
    proxy_inertia_eigenvalue_max: float = 1.0e8
    warm_start_wrench: bool = True
    deterministic: bool = False
    fail_on_overflow: bool = True
    fail_on_nonfinite: bool = True
    static_contact_owner: MuJoCoVBDStaticContactOwner | str = MuJoCoVBDStaticContactOwner.AUTO

    @property
    def relaxation_mode_int(self) -> int:
        return _RELAXATION_TO_INT[_as_relaxation(self.relaxation)]

    @property
    def static_owner_int(self) -> int:
        return _STATIC_OWNER_TO_INT[_as_static_owner(self.static_contact_owner)]


def _as_relaxation(value: MuJoCoVBDRelaxation | str) -> MuJoCoVBDRelaxation:
    if isinstance(value, MuJoCoVBDRelaxation):
        return value
    try:
        return MuJoCoVBDRelaxation(str(value))
    except ValueError as exc:
        valid = [member.value for member in MuJoCoVBDRelaxation]
        raise ValueError(f"relaxation must be one of {valid}, got {value!r}") from exc


def _as_static_owner(value: MuJoCoVBDStaticContactOwner | str) -> MuJoCoVBDStaticContactOwner:
    if isinstance(value, MuJoCoVBDStaticContactOwner):
        return value
    try:
        return MuJoCoVBDStaticContactOwner(str(value))
    except ValueError as exc:
        valid = [member.value for member in MuJoCoVBDStaticContactOwner]
        raise ValueError(f"static_contact_owner must be one of {valid}, got {value!r}") from exc


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def validate_coupling_options(
    value: MuJoCoVBDCouplingOptions | Mapping[str, object] | None,
) -> MuJoCoVBDCouplingOptions:
    """Normalize enums, reject unknown keys, and validate all finite ranges.

    See ``DESIGN.md`` section 5.1. Does not silently change user settings, but
    normalizes enum spellings and canonicalizes numeric types.
    """
    if value is None:
        options = MuJoCoVBDCouplingOptions()
    elif isinstance(value, MuJoCoVBDCouplingOptions):
        options = value
    elif isinstance(value, Mapping):
        known = {field.name for field in fields(MuJoCoVBDCouplingOptions)}
        unknown = sorted(set(value.keys()) - known)
        if unknown:
            raise ValueError(f"Unknown coupling option keys: {unknown}; valid keys are {sorted(known)}")
        options = MuJoCoVBDCouplingOptions(**dict(value))
    else:
        raise TypeError(
            f"coupling_options must be MuJoCoVBDCouplingOptions, a mapping, or None; got {type(value).__name__}"
        )

    relaxation = _as_relaxation(options.relaxation)
    static_owner = _as_static_owner(options.static_contact_owner)

    iterations = int(options.iterations)
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    relaxation_min = _require_finite("relaxation_min", options.relaxation_min)
    relaxation_max = _require_finite("relaxation_max", options.relaxation_max)
    relaxation_initial = _require_finite("relaxation_initial", options.relaxation_initial)
    if not (0.0 < relaxation_min <= relaxation_max):
        raise ValueError(
            f"require 0 < relaxation_min <= relaxation_max, got min={relaxation_min}, max={relaxation_max}"
        )
    if not (relaxation_min <= relaxation_initial <= relaxation_max):
        raise ValueError(
            "relaxation_initial must lie in [relaxation_min, relaxation_max], "
            f"got initial={relaxation_initial}, min={relaxation_min}, max={relaxation_max}"
        )

    force_absolute_tolerance = _require_finite("force_absolute_tolerance", options.force_absolute_tolerance)
    force_relative_tolerance = _require_finite("force_relative_tolerance", options.force_relative_tolerance)
    velocity_tolerance = _require_finite("velocity_tolerance", options.velocity_tolerance)
    for name, tol in (
        ("force_absolute_tolerance", force_absolute_tolerance),
        ("force_relative_tolerance", force_relative_tolerance),
        ("velocity_tolerance", velocity_tolerance),
    ):
        if tol < 0.0:
            raise ValueError(f"{name} must be non-negative, got {tol}")

    proxy_mass_scale = _require_finite("proxy_mass_scale", options.proxy_mass_scale)
    proxy_mass_min = _require_finite("proxy_mass_min", options.proxy_mass_min)
    proxy_mass_max = _require_finite("proxy_mass_max", options.proxy_mass_max)
    if proxy_mass_scale <= 0.0:
        raise ValueError(f"proxy_mass_scale must be positive, got {proxy_mass_scale}")
    if not (0.0 < proxy_mass_min <= proxy_mass_max):
        raise ValueError(
            f"require 0 < proxy_mass_min <= proxy_mass_max, got min={proxy_mass_min}, max={proxy_mass_max}"
        )

    inertia_min = _require_finite("proxy_inertia_eigenvalue_min", options.proxy_inertia_eigenvalue_min)
    inertia_max = _require_finite("proxy_inertia_eigenvalue_max", options.proxy_inertia_eigenvalue_max)
    if not (0.0 < inertia_min <= inertia_max):
        raise ValueError(
            "require 0 < proxy_inertia_eigenvalue_min <= proxy_inertia_eigenvalue_max, "
            f"got min={inertia_min}, max={inertia_max}"
        )

    return replace(
        options,
        iterations=iterations,
        relaxation=relaxation,
        relaxation_initial=relaxation_initial,
        relaxation_min=relaxation_min,
        relaxation_max=relaxation_max,
        force_absolute_tolerance=force_absolute_tolerance,
        force_relative_tolerance=force_relative_tolerance,
        velocity_tolerance=velocity_tolerance,
        proxy_mass_scale=proxy_mass_scale,
        proxy_mass_min=proxy_mass_min,
        proxy_mass_max=proxy_mass_max,
        proxy_inertia_eigenvalue_min=inertia_min,
        proxy_inertia_eigenvalue_max=inertia_max,
        static_contact_owner=static_owner,
    )


@dataclass(frozen=True)
class MuJoCoVBDResolvedOptions:
    """Fully resolved construction options handed to the selected backend.

    Bundles the validated coupling options with the mode selectors and the raw
    per-core option mappings so a backend constructs only what it needs.
    """

    coupling: MuJoCoVBDCouplingOptions
    joint_mode: str
    coupling_mode: str
    contact_mode: str
    mujoco_options: Mapping[str, object]
    vbd_options: Mapping[str, object]
    collision_options: Mapping[str, object]
