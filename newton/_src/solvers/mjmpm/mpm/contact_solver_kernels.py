# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import warp as wp
import warp.sparse as sp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def compute_collider_inv_mass(
    J_mat_offsets: wp.array[int],
    J_mat_columns: wp.array[int],
    J_mat_values: wp.array[wp.mat33],
    IJtm_mat_offsets: wp.array[int],
    IJtm_mat_columns: wp.array[int],
    IJtm_mat_values: wp.array[wp.mat33],
    collider_inv_mass: wp.array[float],
):
    i = wp.tid()

    block_beg = J_mat_offsets[i]
    block_end = J_mat_offsets[i + 1]

    w_mat = wp.mat33(0.0)

    for b in range(block_beg, block_end):
        col = J_mat_columns[b]
        transposed_block = sp.bsr_block_index(col, i, IJtm_mat_offsets, IJtm_mat_columns)
        if transposed_block == -1:
            continue

        # Mass-splitting: divide by number of nodes overlapping with this body
        multiplicity = float(IJtm_mat_offsets[col + 1] - IJtm_mat_offsets[col])

        w_mat += (J_mat_values[b] @ IJtm_mat_values[transposed_block]) * multiplicity

    _eigvecs, eigvals = wp.eig3(w_mat)
    collider_inv_mass[i] = wp.max(0.0, wp.max(eigvals))


@wp.func
def project_on_friction_cone(
    mu: float,
    nor: wp.vec3,
    r: wp.vec3,
):
    """Projects a stress vector ``r`` onto the Coulomb friction cone (non-orthogonally)."""

    r_n = wp.dot(r, nor)
    r_t = r - r_n * nor

    r_n = wp.max(0.0, r_n)
    mu_rn = mu * r_n

    r_t_n2 = wp.length_sq(r_t)
    if r_t_n2 > mu_rn * mu_rn:
        r_t *= mu_rn / wp.sqrt(r_t_n2)

    return r_n * nor + r_t


@wp.func
def solve_coulomb_isotropic(
    mu: float,
    nor: wp.vec3,
    u: wp.vec3,
):
    """Solves for the relative velocity in the Coulomb friction model,
    assuming an isotropic velocity-impulse relationship, u = r + b
    """

    u_n = wp.dot(u, nor)
    if u_n < 0.0:
        u -= u_n * nor
        tau = wp.length_sq(u)
        alpha = mu * u_n
        if tau <= alpha * alpha:
            u = wp.vec3(0.0)
        else:
            u *= 1.0 + mu * u_n / wp.sqrt(tau)

    return u


@wp.func
def filter_collider_impulse_warmstart(
    friction: float,
    nor: wp.vec3,
    adhesion: float,
    impulse: wp.vec3,
):
    """Filters the collider impulse to be within the friction cone"""

    if friction < 0.0:
        return wp.vec3(0.0)

    return project_on_friction_cone(friction, nor, impulse + adhesion * nor) - adhesion * nor


@wp.kernel
def apply_nodal_impulse_warmstart(
    collider_impulse: wp.array[wp.vec3],
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse
    velocities[i] += inv_mass[i] * impulse


@wp.kernel
def solve_nodal_friction(
    inv_mass: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_inv_mass: wp.array[float],
    velocities: wp.array[wp.vec3],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    """
    Solves for frictional impulses at nodes interacting with colliders.

    For each node (i) potentially in contact:
    1. Skips if friction coefficient is negative (no friction).
    2. Calculates the relative velocity `u0` between the particle and collider,
       accounting for the existing impulse and adhesion.
    3. Computes the effective inverse mass `w` for the interaction.
    4. Calls `solve_coulomb_isotropic` to determine the change in relative
       velocity `u` due to friction.
    5. Calculates the change in impulse `delta_impulse` required to achieve this
       change in relative velocity.
    6. Updates the total impulse, particle velocity, and collider velocity.
    """
    i = wp.tid()

    friction_coeff = collider_friction[i]
    if friction_coeff < 0.0:
        return

    n = collider_normals[i]
    u0 = velocities[i] - collider_velocities[i]

    w = inv_mass[i] + collider_inv_mass[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    delta_impulse[i] = delta_lambda
    impulse[i] += delta_lambda
    velocities[i] += inv_mass[i] * delta_lambda


@wp.kernel
def apply_subgrid_impulse(
    tr_collider_mat_offsets: wp.array[int],
    tr_collider_mat_columns: wp.array[int],
    tr_collider_mat_values: wp.array[float],
    inv_mass: wp.array[float],
    impulses: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    """
    Applies pre-computed impulses to particles and colliders.
    """

    u_i = wp.tid()
    block_beg = tr_collider_mat_offsets[u_i]
    block_end = tr_collider_mat_offsets[u_i + 1]
    if block_beg == block_end:
        return

    delta_f = wp.vec3(0.0)
    for b in range(block_beg, block_end):
        delta_f += tr_collider_mat_values[b] * impulses[tr_collider_mat_columns[b]]

    velocities[u_i] += inv_mass[u_i] * delta_f


@wp.kernel
def apply_subgrid_impulse_warmstart(
    collider_friction: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_adhesion: wp.array[float],
    collider_impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    impulse = filter_collider_impulse_warmstart(
        collider_friction[i], collider_normals[i], collider_adhesion[i], collider_impulse[i]
    )

    collider_impulse[i] = impulse
    delta_impulse[i] = impulse


@wp.kernel
def compute_collider_delassus_diagonal(
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_inv_mass: wp.array[float],
    transposed_collider_mat_offsets: wp.array[int],
    inv_volume: wp.array[float],
    delassus_diagonal: wp.array[float],
):
    i = wp.tid()

    block_beg = collider_mat_offsets[i]
    block_end = collider_mat_offsets[i + 1]

    inv_mass = collider_inv_mass[i]
    w = inv_mass

    for b in range(block_beg, block_end):
        u_i = collider_mat_columns[b]
        weight = collider_mat_values[b]

        multiplicity = transposed_collider_mat_offsets[u_i + 1] - transposed_collider_mat_offsets[u_i]

        w += weight * weight * inv_volume[u_i] * float(multiplicity)

    delassus_diagonal[i] = w


@wp.kernel
def solve_subgrid_friction(
    velocity: wp.array[wp.vec3],
    collider_mat_offsets: wp.array[int],
    collider_mat_columns: wp.array[int],
    collider_mat_values: wp.array[float],
    collider_friction: wp.array[float],
    collider_adhesion: wp.array[float],
    collider_normals: wp.array[wp.vec3],
    collider_delassus_diagonal: wp.array[float],
    collider_velocities: wp.array[wp.vec3],
    impulse: wp.array[wp.vec3],
    delta_impulse: wp.array[wp.vec3],
):
    i = wp.tid()

    w = collider_delassus_diagonal[i]
    friction_coeff = collider_friction[i]
    if w <= 0.0 or friction_coeff < 0.0:
        return

    beg = collider_mat_offsets[i]
    end = collider_mat_offsets[i + 1]

    u0 = -collider_velocities[i]
    for b in range(beg, end):
        u_i = collider_mat_columns[b]
        u0 += collider_mat_values[b] * velocity[u_i]

    n = collider_normals[i]

    u = solve_coulomb_isotropic(friction_coeff, n, u0 - (impulse[i] + collider_adhesion[i] * n) * w)

    delta_u = u - u0
    delta_lambda = delta_u / w

    impulse[i] += delta_lambda
    delta_impulse[i] = delta_lambda


@wp.func
def _mobility_block(mobility: wp.array3d[float], world: int, row_link: int, column_link: int, r0: int, c0: int):
    return wp.mat33(
        mobility[world, 6 * row_link + r0 + 0, 6 * column_link + c0 + 0],
        mobility[world, 6 * row_link + r0 + 0, 6 * column_link + c0 + 1],
        mobility[world, 6 * row_link + r0 + 0, 6 * column_link + c0 + 2],
        mobility[world, 6 * row_link + r0 + 1, 6 * column_link + c0 + 0],
        mobility[world, 6 * row_link + r0 + 1, 6 * column_link + c0 + 1],
        mobility[world, 6 * row_link + r0 + 1, 6 * column_link + c0 + 2],
        mobility[world, 6 * row_link + r0 + 2, 6 * column_link + c0 + 0],
        mobility[world, 6 * row_link + r0 + 2, 6 * column_link + c0 + 1],
        mobility[world, 6 * row_link + r0 + 2, 6 * column_link + c0 + 2],
    )


@wp.kernel
def list_active_links(
    link_body: wp.array2d[int],
    IJtm_mat_offsets: wp.array[int],
    active_links: wp.array2d[int],
    active_count: wp.array[int],
):
    """List each world's links that have collider nodes in this step's rigidity operator."""
    world = wp.tid()
    n = int(0)
    for link in range(link_body.shape[1]):
        body = link_body[world, link]
        if body >= 0 and IJtm_mat_offsets[2 * body + 2] > IJtm_mat_offsets[2 * body]:
            active_links[world, n] = link
            n += 1
    active_count[world] = n


@wp.kernel
def apply_link_mobility(
    link_body: wp.array2d[int],
    active_links: wp.array2d[int],
    active_count: wp.array[int],
    mobility: wp.array3d[float],
    wrenches: wp.array[float],
    twists: wp.array[float],
):
    """Map impulse wrenches of the active links to their twists through the mobility.

    Only links with collider nodes receive wrenches, and the collider velocity
    map reads only their twists, so the product is restricted to them.
    Rigidity-operator body blocks are ``(angular, linear)``: ``2 b`` holds the
    torque or angular velocity and ``2 b + 1`` the force or linear velocity.
    Mobility rows are ``(v_com, omega)`` and columns ``(force, torque)``.
    """
    world, row = wp.tid()
    count = active_count[world]
    if row >= count:
        return
    link = active_links[world, row]
    body = link_body[world, link]
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    for k in range(count):
        other = active_links[world, k]
        other_body = link_body[world, other]
        torque = wp.vec3(wrenches[6 * other_body + 0], wrenches[6 * other_body + 1], wrenches[6 * other_body + 2])
        force = wp.vec3(wrenches[6 * other_body + 3], wrenches[6 * other_body + 4], wrenches[6 * other_body + 5])
        linear += _mobility_block(mobility, world, link, other, 0, 0) * force
        linear += _mobility_block(mobility, world, link, other, 0, 3) * torque
        angular += _mobility_block(mobility, world, link, other, 3, 0) * force
        angular += _mobility_block(mobility, world, link, other, 3, 3) * torque
    for axis in range(3):
        twists[6 * body + axis] = angular[axis]
        twists[6 * body + 3 + axis] = linear[axis]


@wp.kernel
def articulated_collider_inv_mass(
    J_mat_offsets: wp.array[int],
    J_mat_columns: wp.array[int],
    J_mat_values: wp.array[wp.mat33],
    IJtm_mat_offsets: wp.array[int],
    IJtm_mat_columns: wp.array[int],
    IJtm_mat_values: wp.array[wp.mat33],
    body_count: int,
    link_slot: wp.array[int],
    link_count: int,
    mobility: wp.array3d[float],
    collider_inv_mass: wp.array[float],
):
    """Replace the nodal inverse mass of articulated links with their own mobility block."""
    i = wp.tid()
    block_beg = J_mat_offsets[i]
    block_end = J_mat_offsets[i + 1]
    body = int(-1)
    W = wp.mat33(0.0)
    G_angular = wp.mat33(0.0)
    G_linear = wp.mat33(0.0)
    multiplicity = float(0.0)
    for b in range(block_beg, block_end):
        col = J_mat_columns[b]
        if col >= 2 * body_count:
            continue
        candidate = col // 2
        if link_slot[candidate] < 0:
            continue
        transposed_block = sp.bsr_block_index(col, i, IJtm_mat_offsets, IJtm_mat_columns)
        if transposed_block == -1:
            continue
        body = candidate
        multiplicity = float(IJtm_mat_offsets[col + 1] - IJtm_mat_offsets[col])
        if col % 2 == 0:
            W = J_mat_values[b]
            G_angular = IJtm_mat_values[transposed_block]
        else:
            G_linear = IJtm_mat_values[transposed_block]
    if body < 0:
        return
    slot = link_slot[body]
    world = slot // link_count
    link = slot % link_count
    to_linear = _mobility_block(mobility, world, link, link, 0, 0) + W * _mobility_block(
        mobility, world, link, link, 3, 0
    )
    to_angular = _mobility_block(mobility, world, link, link, 0, 3) + W * _mobility_block(
        mobility, world, link, link, 3, 3
    )
    w_mat = (to_linear * G_linear + to_angular * G_angular) * multiplicity
    _eigvecs, eigvals = wp.eig3(0.5 * (w_mat + wp.transpose(w_mat)))
    collider_inv_mass[i] = wp.max(0.0, wp.max(eigvals))


@wp.kernel
def articulated_collider_masses(
    link_slot: wp.array[int],
    body_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    out_mass: wp.array[float],
    out_inv_inertia: wp.array[wp.mat33],
):
    """Use unit inertia for articulated links so the rigidity operator yields wrenches."""
    body = wp.tid()
    if link_slot[body] >= 0:
        out_mass[body] = 1.0
        out_inv_inertia[body] = wp.identity(n=3, dtype=float)
    else:
        out_mass[body] = body_mass[body]
        out_inv_inertia[body] = body_inv_inertia[body]


@dataclass
class ArticulatedRigidity:
    """Articulated link response appended to a collider rigidity operator.

    Attributes:
        body_count: Rigid bodies in the operator's body block columns.
        link_body: Body index of each mobility link slot, shape ``(world_count, link_count)``.
        link_slot: Mobility slot ``world * link_count + link`` of each body, or -1.
        mobility: Link mobility, shape ``(world_count, 6 link_count, 6 link_count)``.
        wrenches: Scratch copy of the body impulse wrenches.
    """

    body_count: int
    link_body: wp.array2d[int]
    link_slot: wp.array[int]
    mobility: wp.array3d[float]
    wrenches: wp.array[float] | None = None
    active_links: wp.array2d[int] | None = None
    active_count: wp.array[int] | None = None
