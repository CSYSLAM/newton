# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shallow MuJoCo/VBD model overlays for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` section 7. This does not build a generic ``ModelView``. It
shallow-copies the large topology arrays and only replaces the small ownership,
mass, flag, joint-enable, and collision-filter arrays.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import BodyFlags, Model, ModelFlags
from .config import MuJoCoVBDCouplingOptions, MuJoCoVBDStaticContactOwner, _as_static_owner
from .contact_routing import MuJoCoVBDContactRouting
from .ownership import MuJoCoVBDOwnership

__all__ = [
    "MuJoCoVBDModelOverlays",
    "build_model_overlays",
    "build_mujoco_overlay",
    "build_vbd_overlay",
    "refresh_model_overlays",
]


@dataclass
class MuJoCoVBDModelOverlays:
    """The two shallow model overlays consumed by the backends."""

    mujoco: Model
    vbd: Model


def _shallow_model_copy(model: Model) -> Model:
    """Copy the ``Model`` object while sharing its large device arrays."""
    return copy(model)


def build_mujoco_overlay(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
    options: MuJoCoVBDCouplingOptions,
) -> Model:
    """Overlay enabling only MuJoCo articulation dynamics (``DESIGN.md`` 7.1).

    Global body/joint ids stay stable so runtime scatter is unnecessary. VBD
    body dynamics are not disabled here; the MuJoCo backend integrates only the
    owned bodies via ownership arrays.
    """
    overlay = _shallow_model_copy(model)
    _ = routing

    # Cross contacts belong exclusively to VBD. Hiding VBD-owned shapes from
    # MuJoCo prevents the same M-V pair from being solved by both cores.
    flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32).copy()
    collision_bits = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
    for shape in ownership.vbd_shapes:
        flags[shape] &= ~collision_bits
    if _as_static_owner(options.static_contact_owner) is MuJoCoVBDStaticContactOwner.VBD:
        for shape in ownership.static_shapes:
            flags[shape] &= ~collision_bits
    overlay.shape_flags = wp.array(flags, dtype=model.shape_flags.dtype, device=model.device)
    if model.joint_count:
        joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=np.bool_).copy()
        selected = set(ownership.mujoco_joints)
        for joint in range(model.joint_count):
            if joint not in selected:
                joint_enabled[joint] = False
        overlay.joint_enabled = wp.array(joint_enabled, dtype=wp.bool, device=model.device)
    return overlay


def build_vbd_overlay(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
    options: MuJoCoVBDCouplingOptions,
) -> Model:
    """Overlay marking MuJoCo bodies as proxies for VBD (``DESIGN.md`` 7.2).

    MuJoCo-owned bodies get ``BodyFlags.PROXY`` and are given a separate
    effective mass/inertia workspace by the VBD backend. MuJoCo articulation
    joints are disabled in VBD to avoid solving the robot joints twice.
    """
    overlay = _shallow_model_copy(model)
    _ = (routing, options)

    if model.body_count and ownership.proxy_bodies:
        flags = np.asarray(model.body_flags.numpy(), dtype=np.int32).copy()
        for body in ownership.proxy_bodies:
            flags[body] = int(flags[body]) | int(BodyFlags.PROXY)
        overlay.body_flags = wp.array(flags, dtype=model.body_flags.dtype, device=model.device)

    if model.joint_count and ownership.mujoco_joints:
        joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=np.bool_).copy()
        joint_enabled[list(ownership.mujoco_joints)] = False
        overlay.joint_enabled = wp.array(joint_enabled, dtype=wp.bool, device=model.device)

    return overlay


def build_model_overlays(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
    options: MuJoCoVBDCouplingOptions,
) -> MuJoCoVBDModelOverlays:
    """Build both overlays (``DESIGN.md`` section 7)."""
    return MuJoCoVBDModelOverlays(
        mujoco=build_mujoco_overlay(model, ownership, routing, options),
        vbd=build_vbd_overlay(model, ownership, routing, options),
    )


def refresh_model_overlays(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDCouplingOptions,
    overlays: MuJoCoVBDModelOverlays,
    flags: ModelFlags | int,
) -> None:
    """Refresh mutable masked arrays without changing overlay topology."""
    changed = int(flags)
    if changed & int(ModelFlags.SHAPE_PROPERTIES):
        shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32).copy()
        collision_bits = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)
        for shape in ownership.vbd_shapes:
            shape_flags[shape] &= ~collision_bits
        if _as_static_owner(options.static_contact_owner) is MuJoCoVBDStaticContactOwner.VBD:
            for shape in ownership.static_shapes:
                shape_flags[shape] &= ~collision_bits
        overlays.mujoco.shape_flags.assign(shape_flags)

    if changed & int(ModelFlags.BODY_PROPERTIES) and model.body_count:
        body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32).copy()
        for body in ownership.proxy_bodies:
            body_flags[body] |= int(BodyFlags.PROXY)
        overlays.vbd.body_flags.assign(body_flags)

    if changed & int(ModelFlags.JOINT_PROPERTIES) and model.joint_count:
        joint_enabled = np.asarray(model.joint_enabled.numpy(), dtype=np.bool_)
        mujoco_enabled = joint_enabled.copy()
        vbd_enabled = joint_enabled.copy()
        mujoco_owned = set(ownership.mujoco_joints)
        for joint in range(model.joint_count):
            if joint not in mujoco_owned:
                mujoco_enabled[joint] = False
        if ownership.mujoco_joints:
            vbd_enabled[list(ownership.mujoco_joints)] = False
        overlays.mujoco.joint_enabled.assign(mujoco_enabled)
        overlays.vbd.joint_enabled.assign(vbd_enabled)
