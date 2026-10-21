# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen contact response of independent rigid bodies for material coupling."""

import warp as wp

from newton._src.sim import BodyFlags

mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
vec6d = wp.types.vector(length=6, dtype=wp.float64)


@wp.func
def inverse_inertia_world(q: wp.transform, inverse: wp.mat33):
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(q))
    return rotation * inverse * wp.transpose(rotation)


@wp.func
def response(matrix: mat66, impulse: wp.spatial_vector):
    out = wp.spatial_vector()
    for i in range(6):
        value = float(0.0)
        for j in range(6):
            value += matrix[i, j] * impulse[j]
        out[i] = value
    return out


@wp.kernel
def select_bodies(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    flags: wp.array[int],
    mass_inv: wp.array[float],
    enabled: wp.array[int],
):
    """Fall back to independent implicit response for bodies touching another free dynamic body."""
    body = wp.tid()
    use = int(mass_inv[body] > 0.0 and (flags[body] & (BodyFlags.KINEMATIC | BodyFlags.PROXY)) == 0)
    if use:
        for c in range(wp.min(count[0], shape0.shape[0])):
            if shape0[c] < 0 or shape1[c] < 0:
                continue
            a = shape_body[shape0[c]]
            b = shape_body[shape1[c]]
            other = int(-1)
            if a == body:
                other = b
            elif b == body:
                other = a
            if other >= 0 and mass_inv[other] > 0.0:
                if (flags[other] & (BodyFlags.KINEMATIC | BodyFlags.PROXY)) == 0:
                    use = 0
    enabled[body] = use


@wp.kernel
def apply_prediction(
    enabled: wp.array[int],
    impulse: wp.array[wp.spatial_vector],
    q: wp.array[wp.transform],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    velocity: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if enabled[body]:
        velocity[body] += wp.spatial_vector(
            mass_inv[body] * wp.spatial_top(impulse[body]),
            inverse_inertia_world(q[body], inertia_inv[body]) * wp.spatial_bottom(impulse[body]),
        )


@wp.func
def constrain(factor: mat66d, original: mat66d, row: vec6d, softness: float):
    transformed = wp.transpose(factor) * row
    squared = wp.dot(transformed, transformed)
    denominator = squared + wp.float64(softness)
    reference = wp.dot(row, original * row)
    if squared > wp.max(wp.float64(1.0e-12), wp.float64(1.0e-9) * reference):
        coefficient = (wp.float64(1.0) - wp.sqrt(wp.float64(softness) / denominator)) / squared
        change = factor * transformed
        for i in range(6):
            for j in range(6):
                factor[i, j] -= coefficient * change[i] * transformed[j]
    return factor


@wp.func
def factor_mobility(matrix: mat66d):
    factor = mat66d()
    for j in range(6):
        diagonal = matrix[j, j]
        for k in range(j):
            diagonal -= factor[j, k] * factor[j, k]
        factor[j, j] = wp.sqrt(wp.max(diagonal, wp.float64(1.0e-30)))
        for i in range(j + 1, 6):
            value = matrix[i, j]
            for k in range(j):
                value -= factor[i, k] * factor[j, k]
            factor[i, j] = value / factor[j, j]
    return factor


@wp.kernel(enable_backward=False)
def build_response(
    enabled: wp.array[int],
    q: wp.array[wp.transform],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    count: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra: wp.array[wp.vec3],
    rb: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    params: wp.array[wp.vec4],
    impulses: wp.array[wp.vec3],
    out: wp.array[mat66],
):
    """Eliminate active normal compliance and interior sticking rows in six dimensions."""
    body = wp.tid()
    base = mat66d()
    inverse = inverse_inertia_world(q[body], inertia_inv[body])
    for i in range(3):
        base[i, i] = wp.float64(mass_inv[body])
        for j in range(3):
            base[i + 3, j + 3] = wp.float64(inverse[i, j])
    matrix = factor_mobility(base)
    if enabled[body]:
        for phase in range(2):
            for c in range(wp.min(count[0], bodies.shape[0])):
                pair = bodies[c]
                if pair[0] != body and pair[1] != body:
                    continue
                n = normal[c]
                load = wp.dot(impulses[c], n)
                if load <= 1.0e-10:
                    continue
                radius = rb[c]
                if pair[0] == body:
                    radius = ra[c]
                if phase == 0:
                    angular = wp.cross(radius, n)
                    row = vec6d(
                        wp.float64(n[0]),
                        wp.float64(n[1]),
                        wp.float64(n[2]),
                        wp.float64(angular[0]),
                        wp.float64(angular[1]),
                        wp.float64(angular[2]),
                    )
                    matrix = constrain(matrix, base, row, params[c][1])
                else:
                    mu = params[c][3]
                    tangent = impulses[c] - load * n
                    if mu <= 0.0 or wp.length(tangent) >= 0.9999 * mu * load:
                        continue
                    axis = wp.vec3(1.0, 0.0, 0.0)
                    if wp.abs(n[0]) > 0.8:
                        axis = wp.vec3(0.0, 1.0, 0.0)
                    t0 = wp.normalize(wp.cross(n, axis))
                    t1 = wp.cross(n, t0)
                    for k in range(2):
                        direction = t0
                        if k == 1:
                            direction = t1
                        angular = wp.cross(radius, direction)
                        row = vec6d(
                            wp.float64(direction[0]),
                            wp.float64(direction[1]),
                            wp.float64(direction[2]),
                            wp.float64(angular[0]),
                            wp.float64(angular[1]),
                            wp.float64(angular[2]),
                        )
                        matrix = constrain(matrix, base, row, 0.0)
    matrix = matrix * wp.transpose(matrix)
    result = mat66()
    for i in range(6):
        for j in range(6):
            result[i, j] = wp.float32(wp.float64(0.5) * (matrix[i, j] + matrix[j, i]))
    out[body] = result


@wp.kernel
def remove_prediction(
    enabled: wp.array[int],
    mobility: wp.array[mat66],
    prediction: wp.array[wp.spatial_vector],
    velocity: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if enabled[body]:
        velocity[body] -= response(mobility[body], prediction[body])


@wp.kernel
def finish_material(
    enabled: wp.array[int],
    mobility: wp.array[mat66],
    base: wp.array[wp.spatial_vector],
    impulse: wp.array[wp.spatial_vector],
    velocity: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    if enabled[body]:
        velocity[body] = base[body] + response(mobility[body], impulse[body])


@wp.kernel
def diagnose(
    enabled: wp.array[int],
    q: wp.array[wp.transform],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    start: wp.array[wp.spatial_vector],
    accepted: wp.array[wp.spatial_vector],
    checked: wp.array[wp.spatial_vector],
    material: wp.array[wp.spatial_vector],
    contact: wp.array[wp.spatial_vector],
    momentum_residual: wp.array[wp.spatial_vector],
    velocity_residual: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    momentum = wp.spatial_vector()
    residual = wp.spatial_vector()
    if enabled[body]:
        rotation = wp.quat_to_matrix(wp.transform_get_rotation(q[body]))
        change = accepted[body] - start[body]
        momentum = wp.spatial_vector(
            mass[body] * wp.spatial_top(change),
            rotation * inertia[body] * wp.transpose(rotation) * wp.spatial_bottom(change),
        )
        momentum -= material[body] + contact[body]
        residual = accepted[body] - checked[body]
    momentum_residual[body] = momentum
    velocity_residual[body] = residual


@wp.kernel
def check_material_velocity(
    enabled: wp.array[int],
    base: wp.array[wp.spatial_vector],
    delta: wp.array[wp.spatial_vector],
    accepted: wp.array[wp.spatial_vector],
    residual: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    value = wp.spatial_vector()
    if enabled[body]:
        value = accepted[body] - (base[body] + delta[body])
    residual[body] = value
