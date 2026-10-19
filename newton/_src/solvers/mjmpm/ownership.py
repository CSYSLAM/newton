# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Partition complete mechanisms from MPM-owned free bodies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from newton._src.sim import JointType, Model


@dataclass(frozen=True)
class Ownership:
    """Immutable global indices for the two dynamic domains."""

    mujoco_joints: tuple[int, ...]
    mujoco_bodies: tuple[int, ...]
    mpm_bodies: tuple[int, ...]
    mpm_particles: tuple[int, ...]


def resolve_ownership(model: Model, *, mujoco_joints: Sequence[int] | None = None) -> Ownership:
    """Select whole connected joint trees while leaving standalone free bodies in MPM."""
    types = model.joint_type.numpy()
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    mechanism = {i for i, kind in enumerate(types) if kind not in (JointType.FREE, JointType.FIXED)}
    if mujoco_joints is None:
        selected = set(mechanism)
        changed = True
        while changed:
            bodies = {int(b) for j in selected for b in (parents[j], children[j]) if b >= 0}
            expanded = selected | {j for j in range(model.joint_count) if parents[j] in bodies or children[j] in bodies}
            changed = expanded != selected
            selected = expanded
    else:
        selected = set(mujoco_joints)
        if len(selected) != len(mujoco_joints):
            raise ValueError("mujoco_joints contains duplicate indices")
        if any(not isinstance(j, (int, np.integer)) or j < 0 or j >= model.joint_count for j in selected):
            raise ValueError("mujoco_joints contains invalid joint indices")
    bodies = {int(b) for j in selected for b in (parents[j], children[j]) if b >= 0}
    for j in range(model.joint_count):
        if j not in selected and (parents[j] in bodies or children[j] in bodies):
            raise ValueError("Select a complete closed joint tree; the selection splits connected bodies")
        if j not in selected and types[j] != JointType.FREE:
            raise ValueError("MPM-owned bodies must be independent free bodies; select or collapse remaining joints")
    return Ownership(
        mujoco_joints=tuple(sorted(selected)),
        mujoco_bodies=tuple(sorted(bodies)),
        mpm_bodies=tuple(i for i in range(model.body_count) if i not in bodies),
        mpm_particles=tuple(range(model.particle_count)),
    )
