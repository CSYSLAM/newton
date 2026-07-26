# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Entity ownership for the MJVBDV2 composite solver."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...sim import BodyFlags, JointType, Model

__all__ = ["MJVBDV2Ownership", "resolve_ownership"]


@dataclass(frozen=True)
class MJVBDV2Ownership:
    """Resolved MuJoCo/VBD entity ownership."""

    mujoco_articulations: tuple[int, ...]
    mujoco_joints: tuple[int, ...]
    mujoco_bodies: tuple[int, ...]
    vbd_bodies: tuple[int, ...]
    vbd_particles: tuple[int, ...]
    has_vbd_dynamic_bodies: bool


def _unique_indices(values: Sequence[int], count: int, label: str) -> tuple[int, ...]:
    indices = tuple(int(value) for value in values)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices")
    invalid = [index for index in indices if index < 0 or index >= count]
    if invalid:
        raise IndexError(f"{label} contains out-of-range indices {invalid}; valid range is [0, {count})")
    return tuple(sorted(indices))


def _joint_articulations(model: Model) -> np.ndarray:
    articulation = getattr(model, "joint_articulation", None)
    if articulation is None:
        return np.full(model.joint_count, -1, dtype=np.int32)
    return np.asarray(articulation.numpy(), dtype=np.int32)


def _selected_joints(
    model: Model,
    mujoco_articulations: Sequence[int] | None,
    mujoco_joints: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if mujoco_articulations is not None and mujoco_joints is not None:
        raise ValueError("Specify either mujoco_articulations or mujoco_joints, not both")

    articulation = _joint_articulations(model)
    available_articulations = tuple(sorted(int(value) for value in np.unique(articulation) if value >= 0))

    if mujoco_articulations is not None:
        requested = tuple(sorted({int(value) for value in mujoco_articulations}))
        missing = sorted(set(requested) - set(available_articulations))
        if missing:
            raise ValueError(
                f"mujoco_articulations contains unknown articulation ids {missing}; "
                f"available ids are {list(available_articulations)}"
            )
        joints = tuple(int(index) for index in np.flatnonzero(np.isin(articulation, requested)))
        return requested, joints

    if mujoco_joints is not None:
        joints = _unique_indices(mujoco_joints, model.joint_count, "mujoco_joints")
        articulations = tuple(sorted(int(value) for value in np.unique(articulation[list(joints)]) if value >= 0))
        return articulations, joints

    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32)
    mechanism_mask = (joint_type != int(JointType.FREE)) & (joint_type != int(JointType.FIXED))
    inferred_articulations = tuple(
        sorted(int(value) for value in np.unique(articulation[mechanism_mask]) if value >= 0)
    )
    joints = tuple(int(index) for index in np.flatnonzero(np.isin(articulation, inferred_articulations)))
    if not joints:
        return (), ()
    return inferred_articulations, joints


def _validate_closed_joint_selection(model: Model, joints: tuple[int, ...]) -> tuple[int, ...]:
    parent = np.asarray(model.joint_parent.numpy(), dtype=np.int32)
    child = np.asarray(model.joint_child.numpy(), dtype=np.int32)
    selected = set(joints)
    selected_children = {int(child[joint]) for joint in joints if int(child[joint]) >= 0}
    bodies = set(selected_children)

    for joint in joints:
        parent_body = int(parent[joint])
        if parent_body >= 0:
            bodies.add(parent_body)
            if parent_body not in selected_children:
                parent_joint = next(
                    (candidate for candidate in range(model.joint_count) if int(child[candidate]) == parent_body),
                    None,
                )
                if parent_joint is not None and parent_joint not in selected:
                    raise ValueError(
                        f"MuJoCo joint selection is not ancestor-closed: joint {joint} uses parent body "
                        f"{parent_body}, whose owning joint {parent_joint} is not selected"
                    )

    for joint in range(model.joint_count):
        if joint in selected:
            continue
        parent_body = int(parent[joint])
        child_body = int(child[joint])
        if parent_body in bodies and child_body in bodies:
            raise ValueError(
                f"Joint {joint} connects two MuJoCo-owned bodies but is not selected; "
                "select a complete articulation or a closed joint tree"
            )

    return tuple(sorted(bodies))


def resolve_ownership(
    model: Model,
    *,
    mujoco_articulations: Sequence[int] | None = None,
    mujoco_joints: Sequence[int] | None = None,
) -> MJVBDV2Ownership:
    """Resolve and validate the MJVBDV2 ownership partition."""
    articulations, joints = _selected_joints(model, mujoco_articulations, mujoco_joints)
    bodies = _validate_closed_joint_selection(model, joints)
    body_set = set(bodies)
    vbd_bodies = tuple(index for index in range(model.body_count) if index not in body_set)

    inv_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float64)
    body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32)
    has_vbd_dynamic_bodies = any(
        inv_mass[index] > 0.0 and (int(body_flags[index]) & int(BodyFlags.KINEMATIC)) == 0 for index in vbd_bodies
    )

    return MJVBDV2Ownership(
        mujoco_articulations=articulations,
        mujoco_joints=joints,
        mujoco_bodies=bodies,
        vbd_bodies=vbd_bodies,
        vbd_particles=tuple(range(model.particle_count)),
        has_vbd_dynamic_bodies=has_vbd_dynamic_bodies,
    )
