# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rate-independent bending plasticity for a paper-shell demo.

Membrane stretch remains elastic. Each hinge has an elastic angle, a plastic
rest-angle offset, and accumulated plastic rotation. A scalar return mapping
with linear isotropic hardening bounds the elastic angle. This is a simple
constitutive approximation, not a calibrated model of paper damage/tearing.
"""

import warp as wp


@wp.func
def wrap_angle(angle: float):
    return wp.atan2(wp.sin(angle), wp.cos(angle))


@wp.func
def plastic_return(elastic_angle: float, accumulated: float, yield_angle: float, hardening: float):
    excess = wp.abs(elastic_angle) - yield_angle - hardening * accumulated
    increment = wp.max(0.0, excess) / (1.0 + hardening)
    return wp.vec2(wp.sign(elastic_angle) * increment, accumulated + increment)


@wp.kernel
def update_paper_hinges(
    q: wp.array[wp.vec3],
    indices: wp.array2d[int],
    reference: wp.array[float],
    plastic: wp.array[float],
    accumulated: wp.array[float],
    rest: wp.array[float],
    yield_angle: float,
    hardening: float,
):
    i = wp.tid()
    a, b, c, d = indices[i, 0], indices[i, 1], indices[i, 2], indices[i, 3]
    if a < 0 or b < 0 or c < 0 or d < 0:
        return
    n0, n1, edge = wp.cross(q[c] - q[a], q[d] - q[a]), wp.cross(q[d] - q[b], q[c] - q[b]), q[d] - q[c]
    if wp.length(n0) < 1e-10 or wp.length(n1) < 1e-10 or wp.length(edge) < 1e-8:
        return
    n0, n1, edge = wp.normalize(n0), wp.normalize(n1), wp.normalize(edge)
    angle = wp.atan2(wp.dot(wp.cross(n0, n1), edge), wp.dot(n0, n1))
    elastic = wrap_angle(angle - reference[i] - plastic[i])
    result = plastic_return(elastic, accumulated[i], yield_angle, hardening)
    plastic[i] += result[0]
    accumulated[i] = result[1]
    rest[i] = wrap_angle(reference[i] + plastic[i])
