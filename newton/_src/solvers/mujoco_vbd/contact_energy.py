# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical rigid-contact energy copied and adapted from VBD."""

from __future__ import annotations

import warp as wp

from ...sim.contacts import contact_surface_point

_SMALL_LENGTH_EPS = wp.constant(1.0e-9)


@wp.func
def evaluate_projected_isotropic_friction(
    friction_mu: float,
    normal_load: float,
    normal: wp.vec3,
    slip: wp.vec3,
    epsilon: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Evaluate VBD's regularized isotropic Coulomb friction law.

    This is intentionally copied from
    ``vbd.rigid_vbd_kernels.compute_projected_isotropic_friction``.  Keeping
    the implementation local prevents a change to the existing VBD solver
    from changing this experimental joint solver's contact behavior.
    """
    tangential_slip = slip - normal * wp.dot(normal, slip)
    slip_length = wp.length(tangential_slip)
    if slip_length > 0.0:
        if slip_length > epsilon:
            scale_over_length = 1.0 / slip_length
        else:
            scale_over_length = (-slip_length / epsilon + 2.0) / epsilon
        scale = friction_mu * normal_load * scale_over_length
        force = -(scale * tangential_slip)
        hessian = scale * (wp.identity(3, float) - wp.outer(normal, normal))
        return force, hessian
    return wp.vec3(0.0), wp.mat33(0.0)


@wp.func
def evaluate_canonical_rigid_contact(
    transform0: wp.transform,
    transform1: wp.transform,
    transform0_previous: wp.transform,
    transform1_previous: wp.transform,
    com0_local: wp.vec3,
    com1_local: wp.vec3,
    point0_local: wp.vec3,
    point1_local: wp.vec3,
    offset0_local: wp.vec3,
    offset1_local: wp.vec3,
    normal: wp.vec3,
    penetration_depth: float,
    stiffness: float,
    tangential_stiffness: float,
    damping: float,
    multiplier: wp.vec3,
    friction_mu: float,
    friction_epsilon: float,
    hard_contact: int,
    dt: float,
    friction_c0: wp.vec3,
) -> tuple[wp.vec3, wp.mat33, wp.vec3, wp.vec3]:
    """Evaluate one canonical VBD-style rigid contact.

    The implementation is adapted from
    ``evaluate_rigid_contact_from_collision`` in VBD.  Its first two return
    values are the gradient and Gauss--Newton Hessian with respect to the
    relative displacement ``x1 - x0``.  The remaining values are physical
    forces on endpoints zero and one, respectively.
    """
    multiplier_normal = wp.dot(multiplier, normal)
    if penetration_depth <= _SMALL_LENGTH_EPS and multiplier_normal <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0), wp.vec3(0.0), wp.vec3(0.0)

    normal_load = stiffness * penetration_depth + multiplier_normal
    if stiffness <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0), wp.vec3(0.0), wp.vec3(0.0)
    normal_load = wp.max(normal_load, 0.0)
    if normal_load == 0.0 and hard_contact == 0:
        return wp.vec3(0.0), wp.mat33(0.0), wp.vec3(0.0), wp.vec3(0.0)

    point0 = wp.transform_point(transform0, point0_local)
    point1 = wp.transform_point(transform1, point1_local)
    point0_previous = wp.transform_point(transform0_previous, point0_local)
    point1_previous = wp.transform_point(transform1_previous, point1_local)
    anchor0 = contact_surface_point(transform0, point0_local, offset0_local)
    anchor1 = contact_surface_point(transform1, point1_local, offset1_local)
    anchor0_previous = contact_surface_point(transform0_previous, point0_local, offset0_local)
    anchor1_previous = contact_surface_point(transform1_previous, point1_local, offset1_local)

    normal_outer = wp.outer(normal, normal)
    normal_velocity = wp.dot(normal, (point1 - point1_previous - point0 + point0_previous) / dt)
    tangential_velocity = (anchor1 - anchor1_previous - anchor0 + anchor0_previous) / dt
    tangential_velocity = tangential_velocity - normal * wp.dot(normal, tangential_velocity)

    normal_force = normal * normal_load
    normal_hessian = stiffness * normal_outer
    tangential_force = wp.vec3(0.0)
    tangential_hessian = wp.mat33(0.0)
    if hard_contact == 1:
        if friction_mu > 0.0 and normal_load > 0.0:
            tangential_displacement = -(tangential_velocity * dt)
            multiplier_tangent = multiplier - normal * multiplier_normal
            tangential_force = tangential_stiffness * (tangential_displacement + friction_c0) + multiplier_tangent
            tangential_length = wp.length(tangential_force)
            limit = friction_mu * normal_load
            if tangential_length > limit and tangential_length > 0.0:
                tangential_force = tangential_force * (limit / tangential_length)
            tangential_hessian = tangential_stiffness * (wp.identity(3, float) - normal_outer)
    elif friction_mu > 0.0 and normal_load > 0.0:
        tangential_force, tangential_hessian = evaluate_projected_isotropic_friction(
            friction_mu,
            normal_load,
            normal,
            tangential_velocity * dt,
            friction_epsilon * dt,
        )

    if damping > 0.0 and normal_velocity < 0.0 and normal_load > 0.0:
        normal_force = normal_force - damping * normal_velocity * normal
        normal_hessian = normal_hessian + (damping / dt) * normal_outer

    force1 = normal_force + tangential_force
    hessian = normal_hessian + tangential_hessian
    gradient_relative = -force1
    return gradient_relative, hessian, -force1, force1
