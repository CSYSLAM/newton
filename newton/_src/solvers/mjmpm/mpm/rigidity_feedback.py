# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fuse small rigid mobility maps with collider feedback without an intermediate copy."""

import warp as wp

from newton._src.solvers.mjmpm.mpm.contact_solver_kernels import _independent_block, _mobility_block
from newton._src.solvers.mjmpm.rigid.response import mat66


@wp.func
def _mapped_twist(
    body: int,
    offsets: wp.array[int],
    link_slot: wp.array[int],
    link_body: wp.array2d[int],
    active_links: wp.array2d[int],
    active_count: wp.array[int],
    link_mobility: wp.array3d[float],
    enabled: wp.array[int],
    independent: wp.array[mat66],
    wrenches: wp.array[float],
):
    torque = wp.vec3(wrenches[6 * body], wrenches[6 * body + 1], wrenches[6 * body + 2])
    force = wp.vec3(wrenches[6 * body + 3], wrenches[6 * body + 4], wrenches[6 * body + 5])
    linear = force
    angular = torque
    slot = link_slot[body]
    if slot >= 0 and offsets[2 * body + 2] > offsets[2 * body]:
        world = slot // link_body.shape[1]
        link = slot % link_body.shape[1]
        linear = wp.vec3(0.0)
        angular = wp.vec3(0.0)
        for k in range(active_count[world]):
            other = active_links[world, k]
            other_body = link_body[world, other]
            other_torque = wp.vec3(wrenches[6 * other_body], wrenches[6 * other_body + 1], wrenches[6 * other_body + 2])
            other_force = wp.vec3(
                wrenches[6 * other_body + 3], wrenches[6 * other_body + 4], wrenches[6 * other_body + 5]
            )
            linear += _mobility_block(link_mobility, world, link, other, 0, 0) * other_force
            linear += _mobility_block(link_mobility, world, link, other, 0, 3) * other_torque
            angular += _mobility_block(link_mobility, world, link, other, 3, 0) * other_force
            angular += _mobility_block(link_mobility, world, link, other, 3, 3) * other_torque
    if enabled[body]:
        matrix = independent[body]
        mapped_linear = _independent_block(matrix, 0, 0) * linear + _independent_block(matrix, 0, 3) * angular
        angular = _independent_block(matrix, 3, 0) * linear + _independent_block(matrix, 3, 3) * angular
        linear = mapped_linear
    return wp.spatial_vector(linear, angular)


@wp.kernel(enable_backward=False)
def apply_mobility_feedback(
    offsets: wp.array[int],
    link_slot: wp.array[int],
    link_body: wp.array2d[int],
    active_links: wp.array2d[int],
    active_count: wp.array[int],
    link_mobility: wp.array3d[float],
    enabled: wp.array[int],
    independent: wp.array[mat66],
    wrenches: wp.array[float],
    accumulated: wp.array[wp.spatial_vector],
    twists: wp.array[float],
):
    i = wp.tid()
    if i < enabled.shape[0]:
        twist = _mapped_twist(
            i, offsets, link_slot, link_body, active_links, active_count, link_mobility, enabled, independent, wrenches
        )
        if enabled[i]:
            accumulated[i] += twist
        for k in range(3):
            twists[6 * i + k] = twist[3 + k]
            twists[6 * i + 3 + k] = twist[k]


@wp.kernel(enable_backward=False)
def apply_mobility_feedback_rows(
    offsets: wp.array[int],
    link_slot: wp.array[int],
    link_body: wp.array2d[int],
    active_links: wp.array2d[int],
    active_count: wp.array[int],
    link_mobility: wp.array3d[float],
    enabled: wp.array[int],
    independent: wp.array[mat66],
    wrenches: wp.array[float],
    accumulated: wp.array[wp.spatial_vector],
    twists: wp.array[float],
):
    """Evaluate six twist rows in parallel, with one warp per independent body."""
    body, lane = wp.tid()
    mapped = wp.tile_zeros(shape=6, dtype=float, storage="shared")
    result = wp.tile_zeros(shape=6, dtype=float, storage="shared")
    value = float(0.0)
    if lane < 6:
        # Spatial wrenches store torque first; mobility maps use force first.
        source = (lane + 3) % 6
        value = wrenches[6 * body + source]
        slot = link_slot[body]
        if slot >= 0 and offsets[2 * body + 2] > offsets[2 * body]:
            world = slot // link_body.shape[1]
            row = 6 * (slot % link_body.shape[1]) + lane
            value = float(0.0)
            # Keep the link order and the two three-component products of
            # _mapped_twist for each component, without a parallel reduction.
            for k in range(active_count[world]):
                other = active_links[world, k]
                other_body = link_body[world, other]
                force = wp.vec3(
                    wrenches[6 * other_body + 3], wrenches[6 * other_body + 4], wrenches[6 * other_body + 5]
                )
                torque = wp.vec3(wrenches[6 * other_body], wrenches[6 * other_body + 1], wrenches[6 * other_body + 2])
                force_row = wp.vec3(
                    link_mobility[world, row, 6 * other],
                    link_mobility[world, row, 6 * other + 1],
                    link_mobility[world, row, 6 * other + 2],
                )
                torque_row = wp.vec3(
                    link_mobility[world, row, 6 * other + 3],
                    link_mobility[world, row, 6 * other + 4],
                    link_mobility[world, row, 6 * other + 5],
                )
                value += wp.dot(force_row, force)
                value += wp.dot(torque_row, torque)
    wp.tile_scatter_masked(mapped, lane, value, lane < 6)
    if enabled[body] and lane < 6:
        linear = wp.vec3(wp.tile_extract(mapped, 0), wp.tile_extract(mapped, 1), wp.tile_extract(mapped, 2))
        angular = wp.vec3(wp.tile_extract(mapped, 3), wp.tile_extract(mapped, 4), wp.tile_extract(mapped, 5))
        matrix = independent[body]
        force_row = wp.vec3(matrix[lane, 0], matrix[lane, 1], matrix[lane, 2])
        torque_row = wp.vec3(matrix[lane, 3], matrix[lane, 4], matrix[lane, 5])
        value = wp.dot(force_row, linear) + wp.dot(torque_row, angular)
    wp.tile_scatter_masked(result, lane, value, lane < 6)
    if lane < 6:
        twists[6 * body + (lane + 3) % 6] = value
    if enabled[body] and lane == 0:
        accumulated[body] += wp.spatial_vector(
            wp.vec3(wp.tile_extract(result, 0), wp.tile_extract(result, 1), wp.tile_extract(result, 2)),
            wp.vec3(wp.tile_extract(result, 3), wp.tile_extract(result, 4), wp.tile_extract(result, 5)),
        )
