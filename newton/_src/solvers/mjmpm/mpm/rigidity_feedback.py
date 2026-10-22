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
