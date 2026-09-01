# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Construction-time feature discovery and static backend selection.

See ``DESIGN.md`` sections 4.1-4.3. ``discover_features`` counts topology and
dynamics without constructing a backend; ``select_backend_kind`` maps the counts
plus the requested modes to exactly one backend; ``build_backend`` constructs
only the modules that backend needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

from ...sim import BodyFlags, Model
from ..solver import SolverBase
from .config import MuJoCoVBDResolvedOptions
from .ownership import MuJoCoVBDOwnership
from .vbd.solver_vbd import _get_pneumatic_counts

__all__ = [
    "MuJoCoVBDBackendKind",
    "MuJoCoVBDFeatures",
    "MuJoCoVBDFeaturesInput",
    "build_backend",
    "discover_features",
    "select_backend_kind",
]


class MuJoCoVBDBackendKind(Enum):
    PURE_VBD_SOFT = "pure_vbd_soft"
    PURE_VBD_FULL = "pure_vbd_full"
    KINEMATIC_PASSTHROUGH = "kinematic_passthrough"
    PURE_MUJOCO = "pure_mujoco"
    ONE_WAY_KINEMATIC_SOFT = "one_way_kinematic_soft"
    ONE_WAY_KINEMATIC_FULL = "one_way_kinematic_full"
    ONE_WAY_DYNAMIC_SOFT = "one_way_dynamic_soft"
    ONE_WAY_DYNAMIC_FULL = "one_way_dynamic_full"
    TWO_WAY = "two_way"


@dataclass(frozen=True)
class MuJoCoVBDFeaturesInput:
    """Raw topology/dynamics counts discovered before backend construction."""

    mujoco_joint_count: int
    mujoco_body_count: int
    mujoco_hard_kinematic_body_count: int
    vbd_body_count: int
    vbd_dynamic_body_count: int
    particle_count: int
    triangle_count: int
    edge_count: int
    tetrahedron_count: int
    spring_count: int
    pneumatic_cavity_count: int
    pneumatic_face_count: int


@dataclass(frozen=True)
class MuJoCoVBDFeatures:
    """Fully resolved feature set exposed by ``SolverMuJoCoVBD.features``."""

    backend: MuJoCoVBDBackendKind
    coupling_mode: str
    joint_mode: str
    contact_mode: str
    vbd_core: Literal["none", "soft", "full"]
    contact_pipeline: Literal["none", "soft", "full"]
    mujoco_joint_count: int
    mujoco_body_count: int
    mujoco_hard_kinematic_body_count: int
    vbd_body_count: int
    vbd_dynamic_body_count: int
    particle_count: int
    triangle_count: int
    edge_count: int
    tetrahedron_count: int
    spring_count: int
    pneumatic_cavity_count: int
    pneumatic_face_count: int
    mujoco_solve_enabled: bool
    vbd_solve_enabled: bool
    rigid_solve_enabled: bool
    particle_solve_enabled: bool
    triangle_solve_enabled: bool
    bending_solve_enabled: bool
    tetrahedron_solve_enabled: bool
    spring_solve_enabled: bool
    pneumatic_solve_enabled: bool
    feedback_enabled: bool
    effective_mass_enabled: bool
    iteration_transaction_enabled: bool


def discover_features(model: Model, ownership: MuJoCoVBDOwnership) -> MuJoCoVBDFeaturesInput:
    """Count topology and dynamics without constructing a backend (``DESIGN.md`` 4.2)."""
    cavity, face = _get_pneumatic_counts(model)

    inv_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float64)
    body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32)
    hard_kinematic = sum(
        inv_mass[body] <= 0.0 or (int(body_flags[body]) & int(BodyFlags.KINEMATIC)) != 0
        for body in ownership.mujoco_bodies
    )

    vbd_dynamic = 0
    if ownership.vbd_bodies:
        for body in ownership.vbd_bodies:
            if inv_mass[body] > 0.0 and (int(body_flags[body]) & int(BodyFlags.KINEMATIC)) == 0:
                vbd_dynamic += 1

    return MuJoCoVBDFeaturesInput(
        mujoco_joint_count=len(ownership.mujoco_joints),
        mujoco_body_count=len(ownership.mujoco_bodies),
        mujoco_hard_kinematic_body_count=hard_kinematic,
        vbd_body_count=len(ownership.vbd_bodies),
        vbd_dynamic_body_count=vbd_dynamic,
        particle_count=int(getattr(model, "particle_count", 0)),
        triangle_count=int(getattr(model, "tri_count", 0)),
        edge_count=int(getattr(model, "edge_count", 0)),
        tetrahedron_count=int(getattr(model, "tet_count", 0)),
        spring_count=int(getattr(model, "spring_count", 0)),
        pneumatic_cavity_count=int(cavity),
        pneumatic_face_count=int(face),
    )


def _vbd_has_dynamics(d: MuJoCoVBDFeaturesInput) -> bool:
    return d.vbd_dynamic_body_count > 0 or d.particle_count > 0 or d.pneumatic_cavity_count > 0


def _resolve_contact_pipeline(
    d: MuJoCoVBDFeaturesInput,
    contact_mode: str,
    *,
    default_full: bool,
) -> Literal["soft", "full"]:
    if d.vbd_dynamic_body_count > 0:
        if contact_mode == "soft":
            raise ValueError(
                "contact_mode='soft' is invalid when VBD dynamic rigid bodies are present; "
                "edge/face contact requires the full pipeline (DESIGN 4.3)."
            )
        return "full"
    if contact_mode == "full":
        return "full"
    if contact_mode == "soft":
        return "soft"
    return "full" if default_full else "soft"


def _resolve_vbd_core(d: MuJoCoVBDFeaturesInput) -> Literal["soft", "full"]:
    # Pneumatic and dynamic rigid both require the full VBD/AVBD core.
    if d.vbd_dynamic_body_count > 0 or d.pneumatic_cavity_count > 0:
        return "full"
    return "soft"


def select_backend_kind(
    discovered: MuJoCoVBDFeaturesInput,
    *,
    joint_mode: str,
    coupling_mode: str,
    contact_mode: str,
) -> MuJoCoVBDBackendKind:
    """Return exactly one construction-time backend or raise on contradiction (``DESIGN.md`` 4.3)."""
    has_mj = discovered.mujoco_joint_count > 0
    vbd_dyn = _vbd_has_dynamics(discovered)
    kinematic_source = joint_mode == "kinematic"

    if coupling_mode == "auto":
        if not has_mj:
            resolved_mode = "pure_vbd"
        elif not vbd_dyn:
            resolved_mode = "kinematic_passthrough" if kinematic_source else "pure_mujoco"
        else:
            resolved_mode = "one_way" if kinematic_source else "two_way"
    elif coupling_mode == "one_way":
        if not has_mj:
            raise ValueError("coupling_mode='one_way' requires at least one MuJoCo articulation.")
        resolved_mode = "one_way"
    elif coupling_mode == "two_way":
        if not has_mj:
            raise ValueError("coupling_mode='two_way' requires at least one MuJoCo articulation.")
        if not vbd_dyn:
            raise ValueError("coupling_mode='two_way' requires at least one VBD dynamic body or particle.")
        if kinematic_source:
            raise ValueError(
                "coupling_mode='two_way' requires a dynamic/compliant MuJoCo source, not a hard "
                "kinematic articulation (DESIGN 2.1)."
            )
        resolved_mode = "two_way"
    else:
        raise ValueError(f"unknown coupling_mode {coupling_mode!r}")

    if resolved_mode == "two_way" and discovered.mujoco_hard_kinematic_body_count > 0:
        raise ValueError(
            "two-way coupling requires finite-mass dynamic MuJoCo bodies; "
            f"found {discovered.mujoco_hard_kinematic_body_count} hard-kinematic bodies."
        )

    if resolved_mode == "pure_vbd":
        core = _resolve_vbd_core(discovered)
        return MuJoCoVBDBackendKind.PURE_VBD_FULL if core == "full" else MuJoCoVBDBackendKind.PURE_VBD_SOFT
    if resolved_mode == "kinematic_passthrough":
        return MuJoCoVBDBackendKind.KINEMATIC_PASSTHROUGH
    if resolved_mode == "pure_mujoco":
        return MuJoCoVBDBackendKind.PURE_MUJOCO

    if resolved_mode == "one_way":
        if kinematic_source:
            pipeline = _resolve_contact_pipeline(discovered, contact_mode, default_full=False)
            return (
                MuJoCoVBDBackendKind.ONE_WAY_KINEMATIC_FULL
                if pipeline == "full"
                else MuJoCoVBDBackendKind.ONE_WAY_KINEMATIC_SOFT
            )
        pipeline = _resolve_contact_pipeline(discovered, contact_mode, default_full=False)
        return (
            MuJoCoVBDBackendKind.ONE_WAY_DYNAMIC_FULL
            if pipeline == "full"
            else MuJoCoVBDBackendKind.ONE_WAY_DYNAMIC_SOFT
        )

    return MuJoCoVBDBackendKind.TWO_WAY


def resolve_features(
    kind: MuJoCoVBDBackendKind,
    discovered: MuJoCoVBDFeaturesInput,
    *,
    joint_mode: str,
    coupling_mode: str,
    contact_mode: str,
) -> MuJoCoVBDFeatures:
    """Populate the public feature record for the selected backend."""
    K = MuJoCoVBDBackendKind
    d = discovered

    mujoco_enabled = kind in (K.PURE_MUJOCO, K.ONE_WAY_DYNAMIC_SOFT, K.ONE_WAY_DYNAMIC_FULL, K.TWO_WAY)
    vbd_enabled = kind not in (K.KINEMATIC_PASSTHROUGH, K.PURE_MUJOCO)
    feedback_enabled = kind is K.TWO_WAY
    effective_mass_enabled = kind is K.TWO_WAY
    iteration_transaction_enabled = kind is K.TWO_WAY

    if kind in (K.KINEMATIC_PASSTHROUGH, K.PURE_MUJOCO):
        vbd_core: Literal["none", "soft", "full"] = "none"
        contact_pipeline: Literal["none", "soft", "full"] = "none"
    elif kind in (K.PURE_VBD_SOFT, K.PURE_VBD_FULL):
        vbd_core = _resolve_vbd_core(d)
        contact_pipeline = _resolve_contact_pipeline(
            d,
            contact_mode,
            default_full=d.vbd_dynamic_body_count > 0 or d.particle_count == 0,
        )
    elif kind in (K.ONE_WAY_KINEMATIC_SOFT, K.ONE_WAY_DYNAMIC_SOFT):
        vbd_core = _resolve_vbd_core(d)  # may be "full" for pneumatic particle-only bags
        contact_pipeline = "soft"
    else:
        vbd_core = "full"
        contact_pipeline = "full"

    rigid_enabled = vbd_enabled and (d.vbd_dynamic_body_count > 0 or kind in (K.PURE_VBD_FULL,))
    resolved_coupling = "auto->" + coupling_mode if coupling_mode == "auto" else coupling_mode

    return MuJoCoVBDFeatures(
        backend=kind,
        coupling_mode=resolved_coupling,
        joint_mode=joint_mode,
        contact_mode=contact_mode,
        vbd_core=vbd_core,
        contact_pipeline=contact_pipeline,
        mujoco_joint_count=d.mujoco_joint_count,
        mujoco_body_count=d.mujoco_body_count,
        mujoco_hard_kinematic_body_count=d.mujoco_hard_kinematic_body_count,
        vbd_body_count=d.vbd_body_count,
        vbd_dynamic_body_count=d.vbd_dynamic_body_count,
        particle_count=d.particle_count,
        triangle_count=d.triangle_count,
        edge_count=d.edge_count,
        tetrahedron_count=d.tetrahedron_count,
        spring_count=d.spring_count,
        pneumatic_cavity_count=d.pneumatic_cavity_count,
        pneumatic_face_count=d.pneumatic_face_count,
        mujoco_solve_enabled=mujoco_enabled,
        vbd_solve_enabled=vbd_enabled,
        rigid_solve_enabled=rigid_enabled,
        particle_solve_enabled=vbd_enabled and d.particle_count > 0,
        triangle_solve_enabled=vbd_enabled and d.triangle_count > 0,
        bending_solve_enabled=vbd_enabled and d.edge_count > 0,
        tetrahedron_solve_enabled=vbd_enabled and d.tetrahedron_count > 0,
        spring_solve_enabled=vbd_enabled and d.spring_count > 0,
        pneumatic_solve_enabled=vbd_enabled and d.pneumatic_cavity_count > 0,
        feedback_enabled=feedback_enabled,
        effective_mass_enabled=effective_mass_enabled,
        iteration_transaction_enabled=iteration_transaction_enabled,
    )


def build_backend(
    kind: MuJoCoVBDBackendKind,
    model: Model,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDResolvedOptions,
) -> SolverBase:
    """Construct only the modules required by the selected kind (``DESIGN.md`` 4.1)."""
    from . import backends

    return backends.construct(kind, model, ownership, options)
