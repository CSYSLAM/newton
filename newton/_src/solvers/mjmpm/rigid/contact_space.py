# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded contact-space Gauss-Seidel with the original contact law and order.

Each CUDA lane carries point velocities instead of body twists. A precomputed
Delassus matrix propagates each sequential impulse to all contact points; the
body wrenches and twists are reconstructed after the unchanged sweeps.
"""

import warp as wp

from newton._src.solvers.mjmpm.rigid import kernels as rk
from newton._src.solvers.mjmpm.rigid.warp_sweeps import _shuffle_point, _sync_warp

CAPACITY = 64


@wp.kernel(enable_backward=False)
def compact(
    contact_count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    slots: wp.array[int],
    link_count: int,
    counts: wp.array[int],
    ids: wp.array[int],
    count: wp.array[int],
    enabled: wp.array[int],
    stats: wp.array[int],
):
    total = int(0)
    valid = bool(True)
    world = int(-1)
    for c in range(wp.min(contact_count[0], bodies.shape[0])):
        pair = bodies[c]
        if pair[0] < 0 and pair[1] < 0:
            continue
        if pair[0] == pair[1]:
            valid = False
        if total < CAPACITY:
            ids[total] = c
        else:
            valid = False
        total += 1
        a = int(-1)
        b = int(-1)
        if pair[0] >= 0:
            a = slots[pair[0]]
        if pair[1] >= 0:
            b = slots[pair[1]]
        if a >= 0 and b >= 0:
            valid = False
        for side in range(2):
            slot = a
            if side == 1:
                slot = b
            if slot >= 0:
                current = slot // link_count
                if world < 0:
                    world = current
                elif world != current:
                    valid = False
                if counts[current] > rk.MAX_RESPONSE_LINKS:
                    valid = False
    count[0] = total
    enabled[0] = int(valid)
    stats[0] += 1
    if valid:
        stats[1] += 1


@wp.func
def column_response(
    c: int,
    target: int,
    impulse: wp.vec3,
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    slots: wp.array[int],
    links: wp.array2d[int],
    contacted: wp.array2d[int],
    counts: wp.array[int],
    responses: wp.array2d[rk.mat63],
):
    result = wp.spatial_vector()
    if target >= 0:
        pair = bodies[c]
        for side in range(2):
            body = pair[side]
            if body >= 0 and mass[body] > 0.0:
                radius = ra[c]
                direction = -impulse
                if side == 1:
                    radius = rb[c]
                    direction = impulse
                slot = slots[body]
                if slot >= 0:
                    world = slot // links.shape[1]
                    for k in range(counts[world]):
                        if links[world, contacted[world, k]] == target:
                            change = responses[c, k] * direction
                            result += wp.spatial_vector(
                                change[0], change[1], change[2], change[3], change[4], change[5]
                            )
                elif target == body:
                    result += wp.spatial_vector(mass[body] * direction, inertia[body] * wp.cross(radius, direction))
    return result


@wp.kernel(enable_backward=False)
def build_delassus(
    ids: wp.array[int],
    count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    slots: wp.array[int],
    links: wp.array2d[int],
    contacted: wp.array2d[int],
    counts: wp.array[int],
    responses: wp.array2d[rk.mat63],
    out: wp.array3d[float],
    diagonal: wp.array[wp.mat33],
    enabled: wp.array[int],
):
    i, j = wp.tid()
    if not enabled[0] or i >= count[0] or j >= count[0]:
        return
    c = ids[j]
    r = ids[i]
    pair = bodies[r]
    block = wp.mat33()
    for col in range(3):
        unit = wp.vec3()
        unit[col] = 1.0
        va = column_response(
            c, pair[0], unit, bodies, ra, rb, mass, inertia, slots, links, contacted, counts, responses
        )
        vb = column_response(
            c, pair[1], unit, bodies, ra, rb, mass, inertia, slots, links, contacted, counts, responses
        )
        value = rk.point_velocity(vb, rb[r]) - rk.point_velocity(va, ra[r])
        for row in range(3):
            out[j, 3 * row + col, i] = value[row]
            block[row, col] = value[row]
    if i == j:
        diagonal[i] = block


@wp.func
def product(matrix: wp.array3d[float], j: int, i: int, value: wp.vec3):
    return wp.vec3(
        matrix[j, 0, i] * value[0] + matrix[j, 1, i] * value[1] + matrix[j, 2, i] * value[2],
        matrix[j, 3, i] * value[0] + matrix[j, 4, i] * value[1] + matrix[j, 5, i] * value[2],
        matrix[j, 6, i] * value[0] + matrix[j, 7, i] * value[1] + matrix[j, 8, i] * value[2],
    )


@wp.func
def relative_velocity(
    c: int,
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    velocities: wp.array[wp.spatial_vector],
):
    pair = bodies[c]
    va = wp.vec3()
    vb = wp.vec3()
    if pair[0] >= 0:
        va = rk.point_velocity(velocities[pair[0]], ra[c])
    if pair[1] >= 0:
        vb = rk.point_velocity(velocities[pair[1]], rb[c])
    return vb - va


@wp.kernel(enable_backward=False, module="unique")
def solve(
    ids: wp.array[int],
    count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    params: wp.array[wp.vec4],
    matrix: wp.array3d[float],
    diagonal: wp.array[wp.mat33],
    velocities: wp.array[wp.spatial_vector],
    impulses: wp.array[wp.vec3],
    iterations: int,
    predicted: wp.array[wp.vec3],
    enabled: wp.array[int],
):
    lane = wp.tid()
    if not enabled[0]:
        return
    total = count[0]
    p0 = wp.vec3()
    p1 = wp.vec3()
    if lane < total:
        p0 = relative_velocity(ids[lane], bodies, ra, rb, velocities)
    if lane + 32 < total:
        p1 = relative_velocity(ids[lane + 32], bodies, ra, rb, velocities)
    for phase in range(iterations + 1):
        for j in range(total):
            _sync_warp()
            local = p0
            if j >= 32:
                local = p1
            relative = _shuffle_point(local, j % 32)
            c = ids[j]
            old = impulses[c]
            delta = old
            if phase > 0:
                n = normals[c]
                p = params[c]
                normal_old = wp.dot(old, n)
                normal_new = wp.max(
                    0.0, normal_old + (p[0] - wp.dot(relative, n) - p[1] * normal_old) / wp.max(p[2] + p[1], 1e-12)
                )
                tangent = wp.vec3()
                if normal_new > 0.0 and p[3] > 0.0:
                    delta_n = (normal_new - normal_old) * n
                    relative = relative + diagonal[j] * delta_n
                    tangent_velocity = relative - n * wp.dot(relative, n)
                    tangent = old - normal_old * n
                    speed_squared = wp.dot(tangent_velocity, tangent_velocity)
                    if speed_squared > 1e-18:
                        response_squared = wp.dot(tangent_velocity, diagonal[j] * tangent_velocity)
                        scale = speed_squared / wp.max(response_squared, 1e-12 * speed_squared)
                        tangent = tangent - scale * tangent_velocity
                    limit = p[3] * normal_new
                    tangent_squared = wp.dot(tangent, tangent)
                    if tangent_squared > limit * limit:
                        tangent = wp.normalize(tangent) * limit
                new = normal_new * n + tangent
                delta = new - old
                if lane == 0:
                    impulses[c] = new
            if delta == wp.vec3():
                continue
            if lane < total:
                p0 += product(matrix, j, lane, delta)
            if lane + 32 < total:
                p1 += product(matrix, j, lane + 32, delta)
    if lane < total:
        predicted[lane] = p0
    if lane + 32 < total:
        predicted[lane + 32] = p1


@wp.kernel(enable_backward=False)
def finish(
    ids: wp.array[int],
    count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    slots: wp.array[int],
    links: wp.array2d[int],
    contacted: wp.array2d[int],
    counts: wp.array[int],
    responses: wp.array2d[rk.mat63],
    impulses: wp.array[wp.vec3],
    velocities: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
    enabled: wp.array[int],
):
    body = wp.tid()
    if not enabled[0] or count[0] == 0:
        return
    change = wp.spatial_vector()
    force = wp.spatial_vector()
    for j in range(count[0]):
        c = ids[j]
        impulse = impulses[c]
        change += column_response(
            c, body, impulse, bodies, ra, rb, mass, inertia, slots, links, contacted, counts, responses
        )
        pair = bodies[c]
        if pair[0] == body:
            force -= wp.spatial_vector(impulse, wp.cross(ra[c], impulse))
        if pair[1] == body:
            force += wp.spatial_vector(impulse, wp.cross(rb[c], impulse))
    velocities[body] += change
    wrench[body] += force


@wp.kernel(enable_backward=False)
def check_consistency(
    ids: wp.array[int],
    count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    params: wp.array[wp.vec4],
    velocities: wp.array[wp.spatial_vector],
    impulses: wp.array[wp.vec3],
    predicted: wp.array[wp.vec3],
    tolerance: float,
    accepted: wp.array[int],
    enabled: wp.array[int],
):
    i = wp.tid()
    if not enabled[0] or i >= count[0]:
        return
    c = ids[i]
    actual = relative_velocity(c, bodies, ra, rb, velocities)
    expected = predicted[i]
    error = wp.length(actual - expected)
    bound = tolerance * (1.0 + wp.max(wp.length(actual), wp.length(expected)))
    impulse = impulses[c]
    load = wp.dot(impulse, normals[c])
    tangent = impulse - load * normals[c]
    valid = wp.isfinite(error) and error <= bound
    valid = valid and wp.isfinite(wp.length(impulse)) and load >= -1.0e-7
    valid = valid and wp.length(tangent) <= params[c][3] * wp.max(load, 0.0) + 1.0e-7
    if not valid:
        wp.atomic_min(accepted, 0, 0)


@wp.kernel(enable_backward=False)
def record_acceptance(enabled: wp.array[int], accepted: wp.array[int], stats: wp.array[int]):
    if enabled[0]:
        if accepted[0]:
            stats[2] += 1
        else:
            stats[3] += 1


@wp.kernel(enable_backward=False)
def restore_bodies(
    accepted: wp.array[int],
    before_velocity: wp.array[wp.spatial_vector],
    before_wrench: wp.array[wp.spatial_vector],
    velocity: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    if not accepted[0]:
        velocity[i] = before_velocity[i]
        wrench[i] = before_wrench[i]


@wp.kernel(enable_backward=False)
def restore_impulses(accepted: wp.array[int], before: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    if not accepted[0]:
        out[i] = before[i]


class ContactSpaceBuffers:
    """Fixed-size CUDA workspace; unsupported buckets use the existing solver."""

    def __init__(self, body_count, contact_capacity, device):
        self.ids = wp.empty(CAPACITY, dtype=int, device=device)
        self.count = wp.zeros(1, dtype=int, device=device)
        self.enabled = wp.zeros(1, dtype=int, device=device)
        self.stats = wp.zeros(4, dtype=int, device=device)
        self.matrix = wp.empty((CAPACITY, 9, CAPACITY), dtype=float, device=device)
        self.diagonal = wp.empty(CAPACITY, dtype=wp.mat33, device=device)
        self.predicted = wp.empty(CAPACITY, dtype=wp.vec3, device=device)
        self.accepted = wp.zeros(1, dtype=int, device=device)
        self.before_velocity = wp.empty(body_count, dtype=wp.spatial_vector, device=device)
        self.before_impulses = wp.empty(contact_capacity, dtype=wp.vec3, device=device)
        self.before_wrench = wp.empty(body_count, dtype=wp.spatial_vector, device=device)
        self.velocity_tolerance = 1.0e-5

    def run(self, env, work, fallback):
        """Issue candidates and reference sweeps without pausing a forked capture.

        ``fallback`` must honor ``env._contact_space_skip`` in its cooperative
        kernel, so accepted candidates are not solved or applied twice.
        """
        wp.launch(
            compact,
            dim=1,
            inputs=[
                env.contacts.rigid_contact_count,
                env._contact_bodies,
                env._link_slot,
                env._link_count,
                env._contacted_count,
                self.ids,
                self.count,
                self.enabled,
                self.stats,
            ],
        )

        outputs = (work.body_qd, env.contact_impulses, env.rigid_impulses)
        before = (self.before_velocity, self.before_impulses, self.before_wrench)
        for target, source in zip(before, outputs, strict=True):
            wp.copy(target, source)
        shared = [
            self.ids,
            self.count,
            env._contact_bodies,
            env._contact_ra,
            env._contact_rb,
            env._contact_mass_inv,
            env._contact_inertia_inv,
            env._link_slot,
            env._link_body,
            env._contacted_links,
            env._contacted_count,
            env._link_responses,
        ]
        wp.launch(build_delassus, dim=(CAPACITY, CAPACITY), inputs=[*shared, self.matrix, self.diagonal, self.enabled])
        wp.launch(
            solve,
            dim=32,
            block_dim=32,
            inputs=[
                self.ids,
                self.count,
                env._contact_bodies,
                env._contact_ra,
                env._contact_rb,
                env._contact_normal,
                env._contact_params,
                self.matrix,
                self.diagonal,
                work.body_qd,
                env.contact_impulses,
                env.rigid_iterations,
                self.predicted,
                self.enabled,
            ],
        )
        wp.launch(
            finish,
            dim=env.model.body_count,
            inputs=[*shared, env.contact_impulses, work.body_qd, env.rigid_impulses, self.enabled],
        )
        wp.copy(self.accepted, self.enabled)
        wp.launch(
            check_consistency,
            dim=CAPACITY,
            inputs=[
                self.ids,
                self.count,
                env._contact_bodies,
                env._contact_ra,
                env._contact_rb,
                env._contact_normal,
                env._contact_params,
                work.body_qd,
                env.contact_impulses,
                self.predicted,
                self.velocity_tolerance,
                self.accepted,
                self.enabled,
            ],
        )
        wp.launch(record_acceptance, dim=1, inputs=[self.enabled, self.accepted, self.stats])

        wp.launch(
            restore_bodies,
            dim=env.model.body_count,
            inputs=[self.accepted, self.before_velocity, self.before_wrench, work.body_qd, env.rigid_impulses],
        )
        wp.launch(
            restore_impulses,
            dim=env.contact_impulses.size,
            inputs=[self.accepted, self.before_impulses, env.contact_impulses],
        )
        # Kernel predicates keep the material side stream in flight. A CUDA
        # conditional graph node cannot pause an unjoined forked capture.
        previous = env._contact_space_skip
        env._contact_space_skip = self.accepted
        try:
            fallback()
        finally:
            env._contact_space_skip = previous
