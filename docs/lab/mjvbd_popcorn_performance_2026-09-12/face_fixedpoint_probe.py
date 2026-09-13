# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Skip only bitwise-fixed Frank-Wolfe iterations, without a tolerance stop."""

import warp as wp

from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import eval_shape_sdf, optimize_edge_sdf


@wp.func
def optimize_face_fixedpoint(
    geo: int,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: int,
    table: wp.array[TextureSDFData],
    n_iter: int,
    ls_iter: int,
):
    bary = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    for _iteration in range(n_iter):
        x = bary[0] * a + bary[1] * b + bary[2] * c
        _lower, _phi, gradient = eval_shape_sdf(geo, scale, x, shape_sdf_index, table)
        da, db, dc = wp.dot(gradient, a), wp.dot(gradient, b), wp.dot(gradient, c)
        vertex = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            vertex = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            vertex = wp.vec3(0.0, 0.0, 1.0)
        target = vertex[0] * a + vertex[1] * b + vertex[2] * c
        gamma, _point, _distance, _normal = optimize_edge_sdf(geo, scale, x, target, shape_sdf_index, table, ls_iter)
        updated = (1.0 - gamma) * bary + gamma * vertex
        # Identical barycentrics yield the identical next query, vertex and line
        # search. Every remaining iteration is therefore the same fixed point.
        fixed = updated[0] == bary[0] and updated[1] == bary[1] and updated[2] == bary[2]
        bary = updated
        if fixed:
            break
    x = bary[0] * a + bary[1] * b + bary[2] * c
    _lower, phi, gradient = eval_shape_sdf(geo, scale, x, shape_sdf_index, table)
    return bary, x, phi, gradient
