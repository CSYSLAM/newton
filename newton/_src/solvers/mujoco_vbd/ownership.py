# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Entity ownership and fixed index maps for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` section 6. Every dynamic degree of freedom has exactly one
writer. Host arrays are read only at construction; all runtime maps are
pre-allocated device arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import BodyFlags, JointType, Model

__all__ = [
    "OWNER_MUJOCO",
    "OWNER_NONE",
    "OWNER_SHARED",
    "OWNER_VBD",
    "MuJoCoVBDOwnership",
    "resolve_mujoco_vbd_ownership",
]

OWNER_NONE = 0
OWNER_MUJOCO = 1
OWNER_VBD = 2
OWNER_SHARED = 3


@dataclass(frozen=True)
class MuJoCoVBDOwnership:
    """Resolved MuJoCo/VBD/static ownership partition (``DESIGN.md`` 6.1)."""

    mujoco_articulations: tuple[int, ...]
    mujoco_joints: tuple[int, ...]
    mujoco_bodies: tuple[int, ...]
    mujoco_shapes: tuple[int, ...]
    vbd_bodies: tuple[int, ...]
    vbd_particles: tuple[int, ...]
    vbd_shapes: tuple[int, ...]
    static_shapes: tuple[int, ...]
    proxy_bodies: tuple[int, ...]

    has_vbd_dynamic_bodies: bool

    proxy_body_ids: wp.array  # int32[n_proxy]
    body_to_proxy_slot: wp.array  # int32[model.body_count], -1 when not proxy
    body_owner: wp.array  # int8[model.body_count]
    shape_owner: wp.array  # int8[model.shape_count]
    joint_owner: wp.array  # int8[model.joint_count]


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
    available = tuple(sorted(int(value) for value in np.unique(articulation) if value >= 0))

    if mujoco_articulations is not None:
        requested = tuple(sorted({int(value) for value in mujoco_articulations}))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                f"mujoco_articulations contains unknown articulation ids {missing}; available ids are {list(available)}"
            )
        joints = tuple(int(index) for index in np.flatnonzero(np.isin(articulation, requested)))
        return requested, joints

    if mujoco_joints is not None:
        joints = _unique_indices(mujoco_joints, model.joint_count, "mujoco_joints")
        articulations = tuple(sorted(int(value) for value in np.unique(articulation[list(joints)]) if value >= 0))
        return articulations, joints

    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32)
    mechanism_mask = (joint_type != int(JointType.FREE)) & (joint_type != int(JointType.FIXED))
    inferred = tuple(sorted(int(value) for value in np.unique(articulation[mechanism_mask]) if value >= 0))
    joints = tuple(int(index) for index in np.flatnonzero(np.isin(articulation, inferred)))
    if not joints:
        return (), ()
    return inferred, joints


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
        if parent_body in bodies or child_body in bodies:
            raise ValueError(
                f"Joint {joint} touches a MuJoCo-owned body but is not selected; "
                "select a complete articulation or a closed joint tree"
            )

    return tuple(sorted(bodies))


def _classify_shapes(
    model: Model,
    mujoco_bodies: set[int],
    vbd_bodies: set[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int32)
    mujoco_shapes: list[int] = []
    vbd_shapes: list[int] = []
    static_shapes: list[int] = []
    for shape in range(model.shape_count):
        body = int(shape_body[shape])
        if body < 0:
            static_shapes.append(shape)
        elif body in mujoco_bodies:
            mujoco_shapes.append(shape)
        elif body in vbd_bodies:
            vbd_shapes.append(shape)
        else:
            static_shapes.append(shape)
    return tuple(mujoco_shapes), tuple(vbd_shapes), tuple(static_shapes)


def resolve_mujoco_vbd_ownership(
    model: Model,
    *,
    mujoco_articulations: Sequence[int] | None = None,
    mujoco_joints: Sequence[int] | None = None,
) -> MuJoCoVBDOwnership:
    """Resolve and validate the MuJoCo/VBD ownership partition.

    Hard-kinematic articulation bodies remain valid ownership members for
    passthrough and one-way modes. Static dispatch rejects them only when the
    caller explicitly selects two-way coupling.
    """
    articulations, joints = _selected_joints(model, mujoco_articulations, mujoco_joints)
    mujoco_body_tuple = _validate_closed_joint_selection(model, joints)
    mujoco_body_set = set(mujoco_body_tuple)
    vbd_body_tuple = tuple(index for index in range(model.body_count) if index not in mujoco_body_set)
    vbd_body_set = set(vbd_body_tuple)

    body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32)
    inv_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float64)

    has_vbd_dynamic_bodies = any(
        inv_mass[index] > 0.0 and (int(body_flags[index]) & int(BodyFlags.KINEMATIC)) == 0 for index in vbd_body_tuple
    )

    mujoco_shapes, vbd_shapes, static_shapes = _classify_shapes(model, mujoco_body_set, vbd_body_set)

    # Proxy bodies are exactly the dynamic MuJoCo-owned bodies exposed to VBD.
    proxy_bodies = mujoco_body_tuple

    device = model.device
    body_owner_np = np.full(model.body_count, OWNER_NONE, dtype=np.int8)
    for body in mujoco_body_tuple:
        body_owner_np[body] = OWNER_MUJOCO
    for body in vbd_body_tuple:
        body_owner_np[body] = OWNER_VBD

    shape_owner_np = np.full(model.shape_count, OWNER_NONE, dtype=np.int8)
    for shape in mujoco_shapes:
        shape_owner_np[shape] = OWNER_MUJOCO
    for shape in vbd_shapes:
        shape_owner_np[shape] = OWNER_VBD
    for shape in static_shapes:
        shape_owner_np[shape] = OWNER_SHARED

    joint_owner_np = np.full(model.joint_count, OWNER_NONE, dtype=np.int8)
    for joint in joints:
        joint_owner_np[joint] = OWNER_MUJOCO
    for joint in range(model.joint_count):
        if joint not in set(joints):
            joint_owner_np[joint] = OWNER_VBD

    body_to_proxy_slot_np = np.full(model.body_count, -1, dtype=np.int32)
    for slot, body in enumerate(proxy_bodies):
        body_to_proxy_slot_np[body] = slot

    proxy_body_ids = wp.array(np.asarray(proxy_bodies, dtype=np.int32), dtype=wp.int32, device=device)
    body_to_proxy_slot = wp.array(body_to_proxy_slot_np, dtype=wp.int32, device=device)
    body_owner = wp.array(body_owner_np, dtype=wp.int8, device=device)
    shape_owner = wp.array(shape_owner_np, dtype=wp.int8, device=device)
    joint_owner = wp.array(joint_owner_np, dtype=wp.int8, device=device)

    return MuJoCoVBDOwnership(
        mujoco_articulations=articulations,
        mujoco_joints=joints,
        mujoco_bodies=mujoco_body_tuple,
        mujoco_shapes=mujoco_shapes,
        vbd_bodies=vbd_body_tuple,
        vbd_particles=tuple(range(model.particle_count)),
        vbd_shapes=vbd_shapes,
        static_shapes=static_shapes,
        proxy_bodies=proxy_bodies,
        has_vbd_dynamic_bodies=has_vbd_dynamic_bodies,
        proxy_body_ids=proxy_body_ids,
        body_to_proxy_slot=body_to_proxy_slot,
        body_owner=body_owner,
        shape_owner=shape_owner,
        joint_owner=joint_owner,
    )


# Validate ShapeFlags is importable and used for downstream routing checks.
_ = ShapeFlags
