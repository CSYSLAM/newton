# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Granular material: frictional grain contacts and breakable plastic bonds (native ``kernels/granular.cu``).

Grains are hashed into cells of twice the largest radius (plus the bond formation gap with adhesion). Each grain
owns its contact row: up to 32 contacts are cached slot-major for the tangent product, rows with more walk the grid
again. Authored (cohesive) and dynamic (adhesive) bonds share the bond kernels; bonds commit a clamped plastic rest
length after every step and break beyond their break strain.
"""

import warp as wp

CONTACT_CACHE = wp.constant(32)


@wp.func
def fmaf(a: float, b: float, c: float):
    """Single-rounding a * b + c: the float32 product is exact in float64."""
    return float(wp.float64(a) * wp.float64(b) + wp.float64(c))


@wp.struct
class GranularModel:
    particles_per_world: int
    state_count: int
    radius: wp.array[float]
    particle_world: wp.array[wp.int32]
    granular_count: int
    granular_indices: wp.array[wp.int32]
    bond_damping: float
    bond_yield_strain: float
    bond_max_compaction: float
    bond_break_strain: float
    formation_gap: float
    formation_speed: float
    max_dynamic_bonds: int


@wp.struct
class GranularView:
    head: wp.array[wp.int32]
    next: wp.array[wp.int32]
    ids: wp.array[wp.int32]
    world: wp.array[wp.int32]
    cells: wp.array[wp.vec3i]
    positions: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    radius: wp.array[float]
    counts: wp.array[wp.int32]
    neighbours: wp.array[wp.int32]
    cache: wp.array[wp.vec4]
    count: int
    slots: int
    cell_size: float
    stiffness: float
    friction: float
    friction_damping: float
    dt: float


@wp.func
def hash_slot(cell: wp.vec3i, slots: int):
    return ((cell[0] * 73856093) ^ (cell[1] * 19349663) ^ (cell[2] * 83492791)) & (slots - 1)


@wp.kernel
def add_diagonal(granular_diagonal: wp.array[wp.mat33], diagonal: wp.array[wp.mat33]):
    i = wp.tid()
    diagonal[i] = diagonal[i] + granular_diagonal[i]


@wp.kernel
def multiply_bonds(
    bonds: wp.array[wp.vec2i],
    active: wp.array[wp.int32],
    tangent: wp.array[wp.mat33],
    p: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
):
    """z = K (p_a - p_b) is added to a and subtracted from b (active bonds only)."""
    i = wp.tid()
    if active[i] == 0:
        return
    a = bonds[i][0]
    b = bonds[i][1]
    z = tangent[i] * (p[a] - p[b])
    wp.atomic_add(product, a, z)
    wp.atomic_add(product, b, -z)


@wp.kernel
def build_grid(
    model: GranularModel,
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    view: GranularView,
    positions: wp.array[wp.vec3],
    previous_positions: wp.array[wp.vec3],
    radius: wp.array[float],
    ids: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    cells: wp.array[wp.vec3i],
    head: wp.array[wp.int32],
    next: wp.array[wp.int32],
):
    """The radius and grain list are per world; positions and ids use the replicated particle offsets."""
    i = wp.tid()
    world = i // model.granular_count
    local = i % model.granular_count
    particle = model.granular_indices[local]
    global_particle = model.particles_per_world * world + particle
    ids[i] = global_particle
    particle_world = world
    if model.state_count <= model.particles_per_world:
        particle_world = 0
        if model.particle_world.shape[0] > 0:
            particle_world = wp.max(model.particle_world[particle], 0)
    worlds[i] = particle_world
    p = x[global_particle]
    positions[i] = p
    previous_positions[i] = previous[global_particle]
    radius[i] = model.radius[particle]
    h = view.cell_size
    cell = wp.vec3i(int(wp.floor(p[0] / h)), int(wp.floor(p[1] / h)), int(wp.floor(p[2] / h)))
    cells[i] = cell
    next[i] = wp.atomic_exch(head, hash_slot(cell, view.slots), i)


@wp.func
def coincident_axis(a: int, b: int):
    axis = (wp.min(a, b) + wp.max(a, b)) % 3
    sign = float(-1.0)
    if a < b:
        sign = 1.0
    if axis == 1:
        return wp.vec3(0.0, sign, 0.0)
    if axis == 2:
        return wp.vec3(0.0, 0.0, sign)
    return wp.vec3(sign, 0.0, 0.0)


@wp.func
def effective_length(model: GranularModel, length: float, rest: float, history: float):
    return wp.max(
        rest * (1.0 - model.bond_max_compaction), wp.min(history, fmaf(rest, model.bond_yield_strain, length))
    )


@wp.kernel
def commit_bonds(
    model: GranularModel,
    x: wp.array[wp.vec3],
    bonds: wp.array[wp.vec2i],
    rest: wp.array[float],
    history: wp.array[float],
    active: wp.array[wp.int32],
):
    """Break overstretched bonds (history untouched); intact bonds commit the clamped plastic rest length."""
    i = wp.tid()
    if active[i] == 0:
        return
    difference = x[bonds[i][0]] - x[bonds[i][1]]
    length = wp.sqrt(wp.dot(difference, difference))
    if length > history[i] * (model.bond_break_strain + 1.0):
        active[i] = 0
    else:
        history[i] = effective_length(model, length, rest[i], history[i])


@wp.func
def symmetric_block(n: wp.vec3, scale: float, isotropic: float):
    """scale n n^T + isotropic I with the native fused rounding."""
    xx = fmaf(scale, n[0] * n[0], isotropic)
    yx = fmaf(isotropic, 0.0, scale * (n[1] * n[0]))
    zx = fmaf(isotropic, 0.0, scale * (n[2] * n[0]))
    yy = fmaf(scale, n[1] * n[1], isotropic)
    zy = fmaf(isotropic, 0.0, scale * (n[2] * n[1]))
    zz = fmaf(scale, n[2] * n[2], isotropic)
    return wp.mat33(xx, yx, zx, yx, yy, zy, zx, zy, zz)


@wp.kernel
def accumulate_bonds(
    model: GranularModel,
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    dt: float,
    bonds: wp.array[wp.vec2i],
    rest: wp.array[float],
    history: wp.array[float],
    active: wp.array[wp.int32],
    stiffness: float,
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    tangent: wp.array[wp.mat33],
):
    """Elastic-plastic bond force and its tangent; the tangent is reset even for inactive or broken bonds."""
    i = wp.tid()
    tangent[i] = wp.mat33(0.0)
    if active[i] == 0:
        return
    a = bonds[i][0]
    b = bonds[i][1]
    difference = x[a] - x[b]
    length = wp.sqrt(wp.dot(difference, difference))
    if length > history[i] * (model.bond_break_strain + 1.0):
        return
    normal = coincident_axis(a, b)
    if length > 1.0e-8:
        normal = difference / length
    floor_length = rest[i] * (1.0 - model.bond_max_compaction)
    effective = effective_length(model, length, rest[i], history[i])
    stretch = length - effective
    magnitude = -(stretch * stiffness)
    damping = model.bond_damping / dt
    if damping > 0.0:
        old_difference = previous[a] - previous[b]
        magnitude -= damping * (length - wp.sqrt(wp.dot(old_difference, old_difference)))
    # The plastic plateau has no axial stiffness; the compaction floor and the elastic branch do.
    axial = stiffness
    if effective > floor_length and effective < history[i]:
        axial = 0.0
    transverse = (wp.max(stretch, 0.0) * stiffness) / wp.max(length, 1.0e-8)
    block = symmetric_block(normal, (damping + axial) - transverse, transverse)
    tangent[i] = block
    f = normal * magnitude
    wp.atomic_add(force, a, f)
    wp.atomic_add(force, b, -f)
    wp.atomic_add(diagonal, a, block)
    wp.atomic_add(diagonal, b, block)


@wp.struct
class NeighbourIterator:
    origin: wp.vec3i
    cell: wp.vec3i
    world: int
    next_cell: int
    next_point: int


@wp.func
def neighbours_begin(view: GranularView, query: int):
    it = NeighbourIterator()
    c = view.cells[query]
    it.origin = wp.vec3i(c[0] - 1, c[1] - 1, c[2] - 1)
    it.world = view.world[query]
    it.next_cell = 0
    it.next_point = -1
    return it


@wp.func
def neighbours_advance(view: GranularView, it: NeighbourIterator):
    """Fluid hash order, but only exact world matches (no shared static world)."""
    while True:
        while it.next_point >= 0:
            point = it.next_point
            it.next_point = view.next[point]
            candidate = view.cells[point]
            if candidate[0] == it.cell[0] and candidate[1] == it.cell[1] and candidate[2] == it.cell[2]:
                if view.world[point] == it.world:
                    return point, it
        if it.next_cell == 27:
            return -1, it
        it.cell = wp.vec3i(
            it.origin[0] + it.next_cell % 3, it.origin[1] + (it.next_cell // 3) % 3, it.origin[2] + it.next_cell // 9
        )
        it.next_cell += 1
        it.next_point = view.head[hash_slot(it.cell, view.slots)]
    return -1, it


@wp.struct
class GrainContact:
    hit: int
    normal: wp.vec3
    force: wp.vec3
    viscosity: float


@wp.func
def grain_contact(view: GranularView, i: int, j: int):
    """Penalty normal force with regularised Coulomb friction (viscosity capped by friction_damping / dt)."""
    result = GrainContact()
    result.hit = 0
    if j == i:
        return result
    difference = view.positions[i] - view.positions[j]
    reach = view.radius[i] + view.radius[j]
    distance2 = wp.dot(difference, difference)
    if distance2 >= reach * reach:
        return result
    old_difference = view.previous[i] - view.previous[j]
    distance = wp.sqrt(distance2)
    normal = wp.vec3(0.0)
    if distance > 1.0e-8:
        normal = difference / distance
    else:
        old_length2 = wp.dot(old_difference, old_difference)
        if old_length2 > 1.0e-16:
            normal = old_difference / wp.sqrt(old_length2)
        else:
            normal = coincident_axis(i, j)
    relative = difference - old_difference
    tangential = relative - wp.dot(relative, normal) * normal
    normal_force = view.stiffness * (reach - distance)
    tangent_length = wp.sqrt(wp.dot(tangential, tangential))
    viscosity = wp.min(view.friction_damping / view.dt, (view.friction * normal_force) / wp.max(tangent_length, 1.0e-8))
    result.hit = 1
    result.normal = normal
    result.force = normal_force * normal - viscosity * tangential
    result.viscosity = viscosity
    return result


@wp.kernel
def accumulate_contacts(view: GranularView, force: wp.array[wp.vec3], diagonal: wp.array[wp.mat33]):
    """A grain owns its row: add the force sum, overwrite the diagonal, cache the first 32 contacts slot-major."""
    i = wp.tid()
    accumulated_force = wp.vec3(0.0)
    accumulated_diagonal = wp.mat33(0.0)
    found = int(0)
    it = neighbours_begin(view, i)
    j, it = neighbours_advance(view, it)
    while j >= 0:
        pair = grain_contact(view, i, j)
        if pair.hit != 0:
            if found < CONTACT_CACHE:
                slot = found * view.count + i
                view.neighbours[slot] = j
                view.cache[slot] = wp.vec4(pair.normal[0], pair.normal[1], pair.normal[2], pair.viscosity)
            accumulated_force = accumulated_force + pair.force
            accumulated_diagonal = accumulated_diagonal + symmetric_block(
                pair.normal, view.stiffness - pair.viscosity, pair.viscosity
            )
            found += 1
        j, it = neighbours_advance(view, it)
    view.counts[i] = found
    a = view.ids[i]
    force[a] = force[a] + accumulated_force
    diagonal[a] = accumulated_diagonal


@wp.func
def contact_product(normal: wp.vec3, viscosity: float, stiffness: float, difference: wp.vec3):
    scale = (stiffness - viscosity) * wp.dot(normal, difference)
    return wp.vec3(
        fmaf(viscosity, difference[0], normal[0] * scale),
        fmaf(viscosity, difference[1], normal[1] * scale),
        fmaf(viscosity, difference[2], normal[2] * scale),
    )


@wp.kernel
def multiply_contacts(view: GranularView, direction: wp.array[wp.vec3], product: wp.array[wp.vec3]):
    """Contact tangent product; overflowed rows recompute geometry instead of trusting a truncated cache."""
    i = wp.tid()
    a = view.ids[i]
    accumulated = wp.vec3(0.0)
    if view.counts[i] < CONTACT_CACHE + 1:
        for entry in range(view.counts[i]):
            slot = entry * view.count + i
            b = view.ids[view.neighbours[slot]]
            cached = view.cache[slot]
            accumulated = accumulated + contact_product(
                wp.vec3(cached[0], cached[1], cached[2]), cached[3], view.stiffness, direction[a] - direction[b]
            )
    else:
        it = neighbours_begin(view, i)
        j, it = neighbours_advance(view, it)
        while j >= 0:
            pair = grain_contact(view, i, j)
            if pair.hit != 0:
                accumulated = accumulated + contact_product(
                    pair.normal, pair.viscosity, view.stiffness, direction[a] - direction[view.ids[j]]
                )
            j, it = neighbours_advance(view, it)
    product[a] = accumulated + product[a]


@wp.func
def initial_bond(offsets: wp.array[wp.int32], columns: wp.array[wp.int32], row: int, other: int):
    """Binary search of the sorted authored-bond CSR row."""
    lower = offsets[row]
    upper = offsets[row + 1]
    end = upper
    while lower < upper:
        middle = lower + (upper - lower) // 2
        if columns[middle] < other:
            lower = middle + 1
        else:
            upper = middle
    return lower < end and columns[lower] == other


@wp.kernel
def form_bonds(
    model: GranularModel,
    view: GranularView,
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    partner: wp.array[wp.int32],
    bonds: wp.array[wp.vec2i],
    rest: wp.array[float],
    history: wp.array[float],
    active: wp.array[wp.int32],
):
    """Release separated inactive slots, then greedily fill free slots with the nearest eligible higher grain."""
    i = wp.tid()
    capacity = model.max_dynamic_bonds
    base = capacity * i
    for slot in range(capacity):
        entry = base + slot
        j = partner[entry]
        if j >= 0 and active[entry] == 0:
            reach = (view.radius[i] + view.radius[j]) + model.formation_gap
            difference = view.positions[i] - view.positions[j]
            if wp.dot(difference, difference) > reach * reach:
                partner[entry] = -1
    for slot in range(capacity):
        entry = base + slot
        if partner[entry] < 0:
            best = int(-1)
            best_distance2 = float(wp.inf)
            it = neighbours_begin(view, i)
            j, it = neighbours_advance(view, it)
            while j >= 0:
                eligible = j > i and not initial_bond(offsets, columns, i, j)
                difference = view.positions[i] - view.positions[j]
                distance2 = wp.dot(difference, difference)
                if eligible:
                    reach = (view.radius[i] + view.radius[j]) + model.formation_gap
                    eligible = distance2 > 1.0e-16 and distance2 <= reach * reach
                if eligible:
                    maximum_displacement = view.dt * model.formation_speed
                    relative = difference - (view.previous[i] - view.previous[j])
                    eligible = wp.dot(relative, relative) <= maximum_displacement * maximum_displacement
                if eligible:
                    for existing in range(capacity):
                        if partner[base + existing] == j:
                            eligible = False
                if eligible and (distance2 < best_distance2 or (distance2 == best_distance2 and j < best)):
                    best = j
                    best_distance2 = distance2
                j, it = neighbours_advance(view, it)
            if best < 0:
                return
            partner[entry] = best
            bonds[entry] = wp.vec2i(view.ids[i], view.ids[best])
            length = wp.sqrt(best_distance2)
            history[entry] = length
            rest[entry] = length
            active[entry] = 1
