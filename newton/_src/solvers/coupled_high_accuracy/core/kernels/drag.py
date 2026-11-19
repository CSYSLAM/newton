# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Aerodynamic drag on cloth triangles, linearised about the step-start normal (native ``kernels/drag.cu``)."""

import warp as wp

from .common import normalized_or_zero


@wp.kernel
def initialize_triangle_drag(
    x: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    normal: wp.array[wp.vec3],
    area: wp.array[float],
    triangles_per_world: int,
    particles_per_world: int,
):
    """Unit normal (zero when degenerate) and area of every triangle of every world at the step start."""
    i = wp.tid()
    world = i // triangles_per_world
    local = triangles[i % triangles_per_world]
    offset = world * particles_per_world
    a = local[0] + offset
    b = local[1] + offset
    c = local[2] + offset
    n = wp.cross(x[b] - x[a], x[c] - x[a])
    length = wp.length(n)
    area[i] = length * 0.5
    normal[i] = normalized_or_zero(n, length, 1.0e-12)


@wp.kernel
def multiply_triangle_drag_tangent(
    p: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    normal: wp.array[wp.vec3],
    coefficient: wp.array[float],
    product: wp.array[wp.vec3],
    triangles_per_world: int,
    particles_per_world: int,
):
    """Drag tangent times p: every vertex receives c n (n . (p_a + p_b + p_c))."""
    i = wp.tid()
    world = i // triangles_per_world
    local = triangles[i % triangles_per_world]
    offset = world * particles_per_world
    a = local[0] + offset
    b = local[1] + offset
    c = local[2] + offset
    n = normal[i]
    total = p[a] + p[b] + p[c]
    f = n * (coefficient[i] * wp.dot(n, total))
    wp.atomic_add(product, a, f)
    wp.atomic_add(product, b, f)
    wp.atomic_add(product, c, f)


@wp.kernel
def evaluate_triangle_drag(
    x: wp.array[wp.vec3],
    x_previous: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    drag_coefficient: wp.array[float],
    normal: wp.array[wp.vec3],
    area: wp.array[float],
    density: float,
    wind: wp.vec3,
    inverse_dt: float,
    force: wp.array[wp.vec3],
    tangent_scalar: wp.array[float],
    triangles_per_world: int,
    particles_per_world: int,
):
    """Drag at the current iterate: vn = n . (v - wind) with v from the substep displacements, q = rho/2 Cd area.

    Each vertex receives -(q vn |vn|) / 3 n; the tangent scalar 2 q |vn| / (9 h) is stored for the preconditioner and
    the tangent product.
    """
    i = wp.tid()
    world = i // triangles_per_world
    triangle = i % triangles_per_world
    local = triangles[triangle]
    offset = world * particles_per_world
    a = local[0] + offset
    b = local[1] + offset
    c = local[2] + offset
    displacement = (x[a] - x_previous[a]) + (x[b] - x_previous[b]) + (x[c] - x_previous[c])
    relative = inverse_dt / 3.0 * displacement - wind
    n = normal[i]
    vn = wp.dot(relative, n)
    qa = density * 0.5 * drag_coefficient[triangle] * area[i]
    flux = qa * vn
    f = n * (wp.abs(vn) * flux / -3.0)
    wp.atomic_add(force, a, f)
    wp.atomic_add(force, b, f)
    wp.atomic_add(force, c, f)
    tangent_scalar[i] = wp.abs(vn) * (qa + qa) * inverse_dt / 9.0


@wp.kernel
def add_triangle_drag_preconditioner(
    triangles: wp.array[wp.vec3i],
    normal: wp.array[wp.vec3],
    tangent_scalar: wp.array[float],
    preconditioner: wp.array[wp.mat33],
    triangles_per_world: int,
    particles_per_world: int,
):
    """Drag preconditioner c n n^T on the diagonal blocks of the three vertices."""
    i = wp.tid()
    world = i // triangles_per_world
    local = triangles[i % triangles_per_world]
    offset = world * particles_per_world
    block = tangent_scalar[i] * wp.outer(normal[i], normal[i])
    wp.atomic_add(preconditioner, local[0] + offset, block)
    wp.atomic_add(preconditioner, local[1] + offset, block)
    wp.atomic_add(preconditioner, local[2] + offset, block)
