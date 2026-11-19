# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle fluid: SPH density constraints and viscosity solved inside the Newton step (native ``kernels/fluid.cu``).

Grid points are the fluid particles of every world followed by each world's static boundary samples, ``stride``
points per world. A spatial hash (head/next linked lists) finds neighbours in the 27 surrounding cells. Up to 64
dynamic neighbours per row are cached for the force, tangent and preconditioner; rows with more neighbours walk the
grid again.
"""

import warp as wp

# Poly6 and spiky constants as stored by the native kernels (bit patterns 0x3fc88904 and 0xc11666c3).
POLY6 = wp.constant(1.5666814)
SPIKY_GRADIENT = wp.constant(-9.400088)
CACHE = wp.constant(64)


@wp.struct
class FluidModel:
    particles_per_world: int
    state_count: int
    mass: wp.array[float]
    particle_world: wp.array[wp.int32]
    fluid_count: int
    fluid_indices: wp.array[wp.int32]
    rest_density: float
    smoothing_radius: float
    boundary_count: int
    boundary_positions: wp.array[wp.vec3]
    boundary_volumes: wp.array[float]


@wp.struct
class FluidView:
    head: wp.array[wp.int32]
    next: wp.array[wp.int32]
    ids: wp.array[wp.int32]
    world: wp.array[wp.int32]
    cells: wp.array[wp.vec3i]
    positions: wp.array[wp.vec3]
    volumes: wp.array[float]
    counts: wp.array[wp.int32]
    neighbours: wp.array[wp.int32]
    cache: wp.array[wp.vec4]
    slots: int
    stride: int
    fluid_count: int
    rows: int
    smoothing_radius: float
    bulk_modulus: float
    viscosity: float
    dt: float


@wp.func
def hash_slot(cell: wp.vec3i, slots: int):
    return ((cell[0] * 73856093) ^ (cell[1] * 19349663) ^ (cell[2] * 83492791)) & (slots - 1)


@wp.kernel
def initialize_scale(scale: wp.array[float], value: float):
    scale[wp.tid()] = value


@wp.kernel
def add_diagonal(fluid_diagonal: wp.array[wp.mat33], diagonal: wp.array[wp.mat33]):
    i = wp.tid()
    diagonal[i] = diagonal[i] + fluid_diagonal[i]


@wp.kernel
def bound_scale(
    view: FluidView, solution: wp.array[wp.vec3], relaxation: float, max_step: float, scale: wp.array[float]
):
    """Per world, the smallest of ``relaxation`` and max_step / |dx| over its fluid particles."""
    i = wp.tid()
    point = (i // view.fluid_count) * view.stride + i % view.fluid_count
    length = wp.length(solution[view.ids[point]])
    limit = relaxation
    if max_step > 0.0 and length > max_step:
        limit = wp.min(limit, max_step / length)
    wp.atomic_min(scale, view.world[point], limit)


@wp.kernel
def scale_step(model: FluidModel, solution: wp.array[wp.vec3], scale: wp.array[float]):
    """Scale every particle's Newton step by its world's limiter (replicated or explicit world table)."""
    i = wp.tid()
    world = int(0)
    if model.state_count > model.particles_per_world:
        world = i // model.particles_per_world
    elif model.particle_world.shape[0] > 0:
        world = wp.max(model.particle_world[i], 0)
    solution[i] = solution[i] * scale[world]


@wp.kernel
def build_grid(
    model: FluidModel,
    x: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    volumes: wp.array[float],
    ids: wp.array[wp.int32],
    worlds: wp.array[wp.int32],
    cells: wp.array[wp.vec3i],
    head: wp.array[wp.int32],
    next: wp.array[wp.int32],
    slots: int,
):
    i = wp.tid()
    stride = model.fluid_count + model.boundary_count
    world = i // stride
    local = i % stride
    replicated = model.state_count > model.particles_per_world
    if local < model.fluid_count:
        particle = model.fluid_indices[local]
        ids[i] = model.particles_per_world * world + particle
        particle_world = world
        if not replicated:
            particle_world = 0
            if model.particle_world.shape[0] > 0:
                particle_world = wp.max(model.particle_world[particle], 0)
        worlds[i] = particle_world
        positions[i] = x[ids[i]]
        volumes[i] = model.mass[particle] / model.rest_density
    else:
        boundary = local - model.fluid_count
        ids[i] = -1
        if replicated:
            worlds[i] = world
        else:
            worlds[i] = -1
        positions[i] = model.boundary_positions[boundary]
        volumes[i] = model.boundary_volumes[boundary]
    p = positions[i]
    h = model.smoothing_radius
    cell = wp.vec3i(int(wp.floor(p[0] / h)), int(wp.floor(p[1] / h)), int(wp.floor(p[2] / h)))
    cells[i] = cell
    next[i] = wp.atomic_exch(head, hash_slot(cell, slots), i)


@wp.struct
class NeighbourIterator:
    origin: wp.vec3i
    cell: wp.vec3i
    world: int
    next_cell: int
    next_point: int


@wp.func
def neighbours_begin(view: FluidView, query: int):
    it = NeighbourIterator()
    c = view.cells[query]
    it.origin = wp.vec3i(c[0] - 1, c[1] - 1, c[2] - 1)
    it.world = view.world[query]
    it.next_cell = 0
    it.next_point = -1
    return it


@wp.func
def neighbours_advance(view: FluidView, it: NeighbourIterator):
    """Next point in the 27 cells (x fastest) of this world or the shared static world (-1); -1 when done."""
    while True:
        while it.next_point >= 0:
            point = it.next_point
            it.next_point = view.next[point]
            candidate = view.cells[point]
            if candidate[0] == it.cell[0] and candidate[1] == it.cell[1] and candidate[2] == it.cell[2]:
                if view.world[point] == it.world or view.world[point] == -1:
                    return point, it
        if it.next_cell == 27:
            return -1, it
        it.cell = wp.vec3i(
            it.origin[0] + it.next_cell % 3, it.origin[1] + (it.next_cell // 3) % 3, it.origin[2] + it.next_cell // 9
        )
        it.next_cell += 1
        it.next_point = view.head[hash_slot(it.cell, view.slots)]
    return -1, it


@wp.func
def row_point(view: FluidView, row: int):
    return (row // view.fluid_count) * view.stride + row % view.fluid_count


@wp.func
def smoothing_gradient(difference: wp.vec3, z: float, h5: float):
    return (difference * (z * (z * SPIKY_GRADIENT))) / h5


@wp.func
def viscosity_scale(
    view: FluidView, query: int, neighbour: int, difference: wp.vec3, distance2: float, gradient: wp.vec3
):
    h = view.smoothing_radius
    soft2 = h * (h * 0.01)
    scale = ((view.viscosity * -2.0) * view.volumes[query]) * view.volumes[neighbour]
    return ((wp.dot(difference, gradient) * scale) / (soft2 + distance2)) / view.dt


@wp.kernel
def evaluate_rows(
    view: FluidView,
    gradients: wp.array[wp.vec3],
    residuals: wp.array[float],
    weights: wp.array[float],
):
    """Density, constraint gradient and stiffness weight of every fluid row; caches up to 64 neighbours but counts
    all of them so later kernels can fall back to a second grid walk."""
    row = wp.tid()
    query = row_point(view, row)
    h = view.smoothing_radius
    h2 = h * h
    h3 = h * h2
    h5 = h * (h * h3)
    density = float(0.0)
    gradient = wp.vec3(0.0)
    found = int(0)
    it = neighbours_begin(view, query)
    j, it = neighbours_advance(view, it)
    while j >= 0:
        difference = view.positions[query] - view.positions[j]
        distance2 = wp.dot(difference, difference)
        z = wp.max(1.0 - distance2 / h2, 0.0)
        value = (z * (z * (z * POLY6))) / h3
        volume = view.volumes[j]
        density = volume * value + density
        if j != query:
            g = smoothing_gradient(difference, z, h5)
            gradient = g * volume + gradient
            if view.ids[j] >= 0 and distance2 < h2:
                if found < CACHE:
                    slot = row * CACHE + found
                    view.neighbours[slot] = j
                    viscosity = float(0.0)
                    if view.viscosity > 0.0 and j > query:
                        viscosity = viscosity_scale(view, query, j, difference, distance2, g)
                    c = g * -view.volumes[j]
                    view.cache[slot] = wp.vec4(c[0], c[1], c[2], viscosity)
                found += 1
        j, it = neighbours_advance(view, it)
    gradients[row] = gradient
    residuals[row] = wp.max(density + -1.0, 0.0)
    weight = float(0.0)
    if density > 1.0:
        weight = view.bulk_modulus * view.volumes[query]
    weights[row] = weight
    view.counts[row] = found


@wp.func
def scatter_viscosity(
    force: wp.array[wp.vec3], diagonal: wp.array[wp.mat33], a: int, b: int, difference: wp.vec3, viscosity: float
):
    f = difference * -viscosity
    wp.atomic_add(force, a, f)
    wp.atomic_add(force, b, -f)
    block = viscosity * wp.identity(n=3, dtype=float)
    wp.atomic_add(diagonal, a, block)
    wp.atomic_add(diagonal, b, block)


@wp.kernel
def accumulate_rows(
    view: FluidView,
    gradients: wp.array[wp.vec3],
    residuals: wp.array[float],
    weights: wp.array[float],
    previous: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    """Density forces and Gauss-Newton diagonal (second grid walk when the row cache overflowed)."""
    row = wp.tid()
    query = row_point(view, row)
    a = view.ids[query]
    k = weights[row]
    g = gradients[row]
    wp.atomic_add(force, a, g * -(k * residuals[row]))
    wp.atomic_add(diagonal, a, wp.outer(g, g) * k)
    if view.counts[row] < CACHE + 1:
        for entry in range(view.counts[row]):
            slot = row * CACHE + entry
            neighbour = view.neighbours[slot]
            b = view.ids[neighbour]
            cached = view.cache[slot]
            c = wp.vec3(cached[0], cached[1], cached[2])
            wp.atomic_add(force, b, c * -(k * residuals[row]))
            wp.atomic_add(diagonal, b, wp.outer(c, c) * k)
            if cached[3] > 0.0:
                dx = view.positions[query] - view.positions[neighbour]
                dy = previous[a] - previous[b]
                scatter_viscosity(force, diagonal, a, b, dx - dy, cached[3])
    else:
        h = view.smoothing_radius
        h2 = h * h
        h5 = h * (h * (h * h2))
        it = neighbours_begin(view, query)
        j, it = neighbours_advance(view, it)
        while j >= 0:
            b = view.ids[j]
            if j != query and b >= 0:
                difference = view.positions[query] - view.positions[j]
                distance2 = wp.dot(difference, difference)
                z = wp.max(1.0 - distance2 / h2, 0.0)
                gj = smoothing_gradient(difference, z, h5)
                c = gj * -view.volumes[j]
                wp.atomic_add(force, b, c * -(k * residuals[row]))
                wp.atomic_add(diagonal, b, wp.outer(c, c) * k)
                # The fallback deliberately has no radius or computed-viscosity positivity check.
                if view.viscosity > 0.0 and j > query:
                    viscosity = viscosity_scale(view, query, j, difference, distance2, gj)
                    scatter_viscosity(force, diagonal, a, b, difference - (previous[a] - previous[b]), viscosity)
            j, it = neighbours_advance(view, it)


@wp.kernel
def multiply_rows(
    view: FluidView,
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
):
    """Matrix-free density/viscosity operator."""
    row = wp.tid()
    query = row_point(view, row)
    a = view.ids[query]
    g = gradients[row]
    total = wp.dot(g, direction[a])
    if view.counts[row] < CACHE + 1:
        for entry in range(view.counts[row]):
            slot = row * CACHE + entry
            cached = view.cache[slot]
            total += wp.dot(wp.vec3(cached[0], cached[1], cached[2]), direction[view.ids[view.neighbours[slot]]])
        scale = total * weights[row]
        accumulated = g * scale
        for entry in range(view.counts[row]):
            slot = row * CACHE + entry
            b = view.ids[view.neighbours[slot]]
            cached = view.cache[slot]
            viscous = cached[3] * (direction[a] - direction[b])
            accumulated = accumulated + viscous
            wp.atomic_add(product, b, scale * wp.vec3(cached[0], cached[1], cached[2]) - viscous)
        wp.atomic_add(product, a, accumulated)
    else:
        h = view.smoothing_radius
        h2 = h * h
        h5 = h * (h * (h * h2))
        it = neighbours_begin(view, query)
        j, it = neighbours_advance(view, it)
        while j >= 0:
            b = view.ids[j]
            if j != query and b >= 0:
                difference = view.positions[query] - view.positions[j]
                z = wp.max(1.0 - wp.dot(difference, difference) / h2, 0.0)
                total -= wp.dot(smoothing_gradient(difference, z, h5), direction[b]) * view.volumes[j]
            j, it = neighbours_advance(view, it)
        scale = total * weights[row]
        wp.atomic_add(product, a, g * scale)
        it = neighbours_begin(view, query)
        j, it = neighbours_advance(view, it)
        while j >= 0:
            b = view.ids[j]
            if j != query and b >= 0:
                difference = view.positions[query] - view.positions[j]
                distance2 = wp.dot(difference, difference)
                z = wp.max(1.0 - distance2 / h2, 0.0)
                gj = smoothing_gradient(difference, z, h5)
                wp.atomic_add(product, b, gj * -(scale * view.volumes[j]))
                if view.viscosity > 0.0 and j > query:
                    viscosity = viscosity_scale(view, query, j, difference, distance2, gj)
                    viscous = (direction[a] - direction[b]) * viscosity
                    wp.atomic_add(product, a, viscous)
                    wp.atomic_add(product, b, -viscous)
            j, it = neighbours_advance(view, it)
