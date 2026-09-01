# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-owner contact routing between MuJoCo, VBD, and static shapes.

See ``DESIGN.md`` section 8. Each collidable shape pair is assigned to exactly
one contact stream so a pair is never solved twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

from ...geometry import ShapeFlags
from ...sim import Model
from .config import (
    STATIC_OWNER_MUJOCO,
    STATIC_OWNER_VBD,
    MuJoCoVBDStaticContactOwner,
    _as_static_owner,
)
from .ownership import OWNER_MUJOCO, OWNER_SHARED, OWNER_VBD, MuJoCoVBDOwnership

__all__ = [
    "MuJoCoVBDContactRouting",
    "build_contact_routing",
    "validate_contact_routing",
]


@dataclass(frozen=True)
class MuJoCoVBDContactRouting:
    """Fixed contact routing produced at construction (``DESIGN.md`` 8.1)."""

    mujoco_shape_pairs: wp.array  # vec2i[n_mj_pairs]
    vbd_shape_pairs: wp.array  # vec2i[n_vbd_pairs]
    cross_shape_pairs: wp.array  # vec2i[n_cross_pairs]
    cross_shape_mask: wp.array  # uint8[shape_count]
    cross_body_mask: wp.array  # uint8[body_count]
    full_surface_shape_indices: tuple[int, ...]

    @property
    def cross_pair_count(self) -> int:
        return int(self.cross_shape_pairs.shape[0])


def _shape_collides_rigid(shape_flags: np.ndarray, shape: int) -> bool:
    """Return whether a shape participates in rigid shape-shape contact."""
    return (int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)) != 0


def _empty_pairs(device) -> wp.array:
    return wp.array(np.empty((0, 2), dtype=np.int32), dtype=wp.vec2i, device=device)


def _to_pair_array(pairs: list[tuple[int, int]], device) -> wp.array:
    if not pairs:
        return _empty_pairs(device)
    return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=device)


def _worlds_compatible(shape_world: np.ndarray, a: int, b: int) -> bool:
    wa = int(shape_world[a])
    wb = int(shape_world[b])
    return wa == -1 or wb == -1 or wa == wb


def build_contact_routing(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    *,
    collision_options: Mapping[str, object] | None = None,
    static_contact_owner: MuJoCoVBDStaticContactOwner | str = MuJoCoVBDStaticContactOwner.AUTO,
) -> MuJoCoVBDContactRouting:
    """Build the fixed per-pair routing (``DESIGN.md`` 8.2).

    ``AUTO`` splits static pairs per participant: robot-static goes to MuJoCo,
    VBD-static goes to VBD. A static shape may be visible to both backends, but
    the same pair never enters two contact streams.
    """
    _ = collision_options  # collision options influence broad phase, not routing set
    device = model.device
    owner = _as_static_owner(static_contact_owner)

    shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32)
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int32)
    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int32)
    shape_owner = np.asarray(ownership.shape_owner.numpy(), dtype=np.int8)

    collidable = [s for s in range(model.shape_count) if _shape_collides_rigid(shape_flags, s)]
    allowed_pairs = None
    if model.shape_contact_pairs is not None:
        pair_data = np.asarray(model.shape_contact_pairs.numpy(), dtype=np.int32).reshape(-1, 2)
        allowed_pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pair_data}

    mujoco_pairs: list[tuple[int, int]] = []
    vbd_pairs: list[tuple[int, int]] = []
    cross_pairs: list[tuple[int, int]] = []

    cross_shape_mask = np.zeros(model.shape_count, dtype=np.uint8)
    cross_body_mask = np.zeros(model.body_count, dtype=np.uint8)

    def route_static(other_owner: int) -> int | None:
        # VBD objects always keep their static contacts in VBD. The option
        # selects only robot-static ownership; MuJoCo cannot integrate a
        # VBD-owned body or particle on behalf of VBD.
        if other_owner == OWNER_VBD:
            return OWNER_VBD
        if other_owner == OWNER_MUJOCO:
            if owner is MuJoCoVBDStaticContactOwner.VBD:
                return OWNER_VBD
            return OWNER_MUJOCO
        return None

    for i in range(len(collidable)):
        a = collidable[i]
        owner_a = int(shape_owner[a])
        for j in range(i + 1, len(collidable)):
            b = collidable[j]
            owner_b = int(shape_owner[b])
            if not _worlds_compatible(shape_world, a, b):
                continue

            pair = (a, b)
            if allowed_pairs is not None and pair not in allowed_pairs:
                continue
            # static/static: skip (no dynamics to resolve).
            if owner_a == OWNER_SHARED and owner_b == OWNER_SHARED:
                continue

            if owner_a == OWNER_SHARED or owner_b == OWNER_SHARED:
                dynamic_owner = owner_b if owner_a == OWNER_SHARED else owner_a
                routed = route_static(dynamic_owner)
                if routed == OWNER_MUJOCO:
                    mujoco_pairs.append(pair)
                elif routed == OWNER_VBD:
                    vbd_pairs.append(pair)
                continue

            if owner_a == OWNER_MUJOCO and owner_b == OWNER_MUJOCO:
                mujoco_pairs.append(pair)
            elif owner_a == OWNER_VBD and owner_b == OWNER_VBD:
                vbd_pairs.append(pair)
            else:
                # M-V cross pair: solved by VBD, feeds MuJoCo (DESIGN 8.2).
                cross_pairs.append(pair)
                cross_shape_mask[a] = 1
                cross_shape_mask[b] = 1
                for shape in pair:
                    body = int(shape_body[shape])
                    if body >= 0:
                        cross_body_mask[body] = 1

    _ = (STATIC_OWNER_MUJOCO, STATIC_OWNER_VBD)  # referenced for stable-constant parity

    # Default full-surface shapes: robot collision shapes that can hit soft VBD.
    full_surface = tuple(
        sorted(
            {
                shape
                for pair in cross_pairs
                for shape in pair
                if int(shape_owner[shape]) == OWNER_MUJOCO
                and (int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
            }
        )
    )

    return MuJoCoVBDContactRouting(
        mujoco_shape_pairs=_to_pair_array(mujoco_pairs, device),
        vbd_shape_pairs=_to_pair_array(vbd_pairs, device),
        cross_shape_pairs=_to_pair_array(cross_pairs, device),
        cross_shape_mask=wp.array(cross_shape_mask, dtype=wp.uint8, device=device),
        cross_body_mask=wp.array(cross_body_mask, dtype=wp.uint8, device=device),
        full_surface_shape_indices=full_surface,
    )


def validate_contact_routing(
    model: Model,
    ownership: MuJoCoVBDOwnership,
    routing: MuJoCoVBDContactRouting,
) -> None:
    """Validate routing invariants (``DESIGN.md`` 8.3)."""
    shape_flags = np.asarray(model.shape_flags.numpy(), dtype=np.int32)
    shape_world = np.asarray(model.shape_world.numpy(), dtype=np.int32)
    body_world = np.asarray(model.body_world.numpy(), dtype=np.int32) if model.body_count else np.empty(0, np.int32)
    shape_body = np.asarray(model.shape_body.numpy(), dtype=np.int32)

    def as_pairs(array: wp.array) -> set[tuple[int, int]]:
        if array.shape[0] == 0:
            return set()
        data = array.numpy().reshape(-1, 2)
        return {(int(a), int(b)) for a, b in data}

    mujoco = as_pairs(routing.mujoco_shape_pairs)
    vbd = as_pairs(routing.vbd_shape_pairs)
    cross = as_pairs(routing.cross_shape_pairs)

    # No duplicate pair across streams (DESIGN 8.3).
    all_pairs = list(mujoco) + list(vbd) + list(cross)
    if len(all_pairs) != len(set(all_pairs)):
        raise ValueError("Contact routing produced a duplicate pair across MuJoCo/VBD/cross streams")

    # Visual-only shapes must not carry COLLIDE_PARTICLES (DESIGN 8.3).
    for shape in routing.full_surface_shape_indices:
        if (int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_PARTICLES)) == 0:
            raise ValueError(
                f"full-surface shape {shape} lacks ShapeFlags.COLLIDE_PARTICLES; visual-only shapes "
                "must not participate in cross contacts"
            )

    # shape world must match body world where both are local (DESIGN 8.3).
    for shape in range(model.shape_count):
        body = int(shape_body[shape])
        if body < 0 or body >= body_world.shape[0]:
            continue
        sw = int(shape_world[shape])
        bw = int(body_world[body])
        if sw != -1 and bw != -1 and sw != bw:
            raise ValueError(f"shape {shape} world {sw} disagrees with owning body {body} world {bw}")

    _ = ownership  # reserved for missing legal M-V pair checks against a candidate oracle
