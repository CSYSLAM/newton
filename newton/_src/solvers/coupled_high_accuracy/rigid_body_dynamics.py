# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body inertia integrators used to construct ABD inertia terms.

All quantities are expressed in body coordinates. ``inertia`` is constant,
``angular_momentum`` is the beginning-of-step body angular momentum, and the
returned rotation increment satisfies ``R_next = R @ relative_rotation``.

The integer method codes match ``InertiaMode`` in the native binding.

The exponential-map closures use the principal SO(3) logarithm, so their
relative step rotation must stay below pi. The Cayley closure is defined when
``relative_rotation + identity`` is nonsingular. The fixed nonlinear iteration
counts are intended for the local time steps used by the ABD inertia path.
"""

from __future__ import annotations

import warp as wp

EXPLICIT_EULER = 0
IMPLICIT_EULER = 1
LIE_GROUP_MIDPOINT = 2
MIDPOINT_CAYLEY = 3
NONE = 4
VBD_SEMI_IMPLICIT = 5

__all__ = [
    "cayley_increment_from_midpoint",
    "integrate_explicit_euler",
    "integrate_implicit_euler",
    "integrate_lie_group_midpoint",
    "integrate_midpoint_cayley",
    "integrate_rigid_body",
    "integrate_vbd_semi_implicit",
    "recover_explicit_euler_end_velocity",
    "recover_implicit_euler_end_velocity",
    "recover_lie_group_midpoint_end_velocity",
    "recover_midpoint_cayley_end_velocity",
    "recover_rigid_body_end_velocity",
    "recover_vbd_semi_implicit_end_velocity",
    "solve_energy_momentum_midpoint",
]


@wp.func
def _exponential_increment(angular_velocity: wp.vec3, dt: float) -> wp.mat33:
    """Return Exp(dt [omega]_x) as a proper rotation matrix."""
    rotation_vector = dt * angular_velocity
    angle = wp.length(rotation_vector)
    if angle > 1.0e-8:
        rotation = wp.quat_from_axis_angle(rotation_vector / angle, angle)
        return wp.quat_to_matrix(rotation)
    return wp.identity(3, dtype=float)


@wp.func
def _logarithmic_velocity(relative_rotation: wp.mat33, dt: float) -> wp.vec3:
    """Return the principal angular velocity Log(Q)/dt."""
    rotation = wp.quat_from_matrix(relative_rotation)
    if rotation[3] < 0.0:
        rotation = wp.quat(
            -rotation[0],
            -rotation[1],
            -rotation[2],
            -rotation[3],
        )
    axis, angle = wp.quat_to_axis_angle(rotation)
    return axis * angle / dt


@wp.func
def integrate_explicit_euler(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Advance Euler's equation using its beginning-of-step derivative."""
    angular_velocity = wp.inverse(inertia) * angular_momentum
    next_angular_momentum = angular_momentum + dt * wp.cross(
        angular_momentum,
        angular_velocity,
    )
    relative_rotation = _exponential_increment(angular_velocity, dt)
    return next_angular_momentum, relative_rotation


@wp.func
def integrate_implicit_euler(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Advance Euler's equation using its end-of-step derivative."""
    inverse_inertia = wp.inverse(inertia)
    next_angular_momentum = angular_momentum + dt * wp.cross(
        angular_momentum,
        inverse_inertia * angular_momentum,
    )

    for _ in range(8):
        next_angular_velocity = inverse_inertia * next_angular_momentum
        residual = (
            next_angular_momentum - angular_momentum - dt * wp.cross(next_angular_momentum, next_angular_velocity)
        )
        jacobian = (
            wp.identity(3, dtype=float)
            + dt * wp.skew(next_angular_velocity)
            - dt * wp.skew(next_angular_momentum) * inverse_inertia
        )
        next_angular_momentum = next_angular_momentum - wp.inverse(jacobian) * residual

    next_angular_velocity = inverse_inertia * next_angular_momentum
    relative_rotation = _exponential_increment(next_angular_velocity, dt)
    return next_angular_momentum, relative_rotation


@wp.func
def integrate_lie_group_midpoint(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Advance with an exponential midpoint while preserving spatial momentum."""
    inverse_inertia = wp.inverse(inertia)
    midpoint_angular_velocity = inverse_inertia * angular_momentum

    for _ in range(8):
        half_rotation = _exponential_increment(midpoint_angular_velocity, 0.5 * dt)
        midpoint_angular_velocity = half_rotation * inverse_inertia * wp.transpose(half_rotation) * angular_momentum

    relative_rotation = _exponential_increment(midpoint_angular_velocity, dt)
    next_angular_momentum = wp.transpose(relative_rotation) * angular_momentum
    return next_angular_momentum, relative_rotation


@wp.func
def solve_energy_momentum_midpoint(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    """Solve the body-angular-momentum midpoint equation with Newton's method."""
    inverse_inertia = wp.inverse(inertia)
    next_angular_momentum = angular_momentum + dt * wp.cross(
        angular_momentum,
        inverse_inertia * angular_momentum,
    )

    for _ in range(8):
        midpoint_angular_momentum = 0.5 * (angular_momentum + next_angular_momentum)
        midpoint_angular_velocity = inverse_inertia * midpoint_angular_momentum
        residual = (
            next_angular_momentum
            - angular_momentum
            - dt * wp.cross(midpoint_angular_momentum, midpoint_angular_velocity)
        )
        jacobian = (
            wp.identity(3, dtype=float)
            + 0.5 * dt * wp.skew(midpoint_angular_velocity)
            - 0.5 * dt * wp.skew(midpoint_angular_momentum) * inverse_inertia
        )
        next_angular_momentum = next_angular_momentum - wp.inverse(jacobian) * residual

    midpoint_angular_momentum = 0.5 * (angular_momentum + next_angular_momentum)
    midpoint_angular_velocity = inverse_inertia * midpoint_angular_momentum
    return (
        next_angular_momentum,
        midpoint_angular_momentum,
        midpoint_angular_velocity,
    )


@wp.func
def cayley_increment_from_midpoint(
    midpoint_angular_velocity: wp.vec3,
    dt: float,
) -> wp.mat33:
    """Return Cayley(dt [omega_mid]_x)."""
    identity = wp.identity(3, dtype=float)
    generator = dt * wp.skew(midpoint_angular_velocity)
    return wp.inverse(identity - 0.5 * generator) * (identity + 0.5 * generator)


@wp.func
def integrate_midpoint_cayley(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Advance with the paired energy-momentum midpoint-Cayley method."""
    next_angular_momentum, _midpoint_momentum, midpoint_velocity = solve_energy_momentum_midpoint(
        inertia, angular_momentum, dt
    )
    relative_rotation = cayley_increment_from_midpoint(midpoint_velocity, dt)
    return next_angular_momentum, relative_rotation


@wp.func
def integrate_vbd_semi_implicit(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    body_torque: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Use VBD's explicit gyro update and normalized quaternion Euler rotation."""
    inverse_inertia = wp.inverse(inertia)
    angular_velocity = inverse_inertia * angular_momentum
    # Evaluate gyro before applying torque, as in Newton's rigid-body predictor.
    next_velocity = angular_velocity + dt * inverse_inertia * (
        body_torque - wp.cross(angular_velocity, inertia * angular_velocity)
    )
    increment = wp.normalize(wp.quat(0.5 * dt * next_velocity, 1.0))
    return inertia * next_velocity, wp.quat_to_matrix(increment)


@wp.func
def integrate_rigid_body(
    inertia: wp.mat33,
    angular_momentum: wp.vec3,
    dt: float,
    integrator: int,
) -> tuple[wp.vec3, wp.mat33]:
    """Dispatch one free-rigid-body step to the selected integrator."""
    if integrator == EXPLICIT_EULER:
        return integrate_explicit_euler(inertia, angular_momentum, dt)
    elif integrator == IMPLICIT_EULER:
        return integrate_implicit_euler(inertia, angular_momentum, dt)
    elif integrator == LIE_GROUP_MIDPOINT:
        return integrate_lie_group_midpoint(inertia, angular_momentum, dt)
    elif integrator == MIDPOINT_CAYLEY:
        return integrate_midpoint_cayley(inertia, angular_momentum, dt)
    elif integrator == VBD_SEMI_IMPLICIT:
        # This dispatcher advances free bodies; forced prediction passes torque
        # separately to avoid evaluating gyro at a torque-shifted velocity.
        return integrate_vbd_semi_implicit(inertia, angular_momentum, wp.vec3(0.0), dt)


@wp.func
def recover_explicit_euler_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
) -> wp.vec3:
    """Recover explicit Euler's end velocity from its rotation increment."""
    initial_velocity = _logarithmic_velocity(relative_rotation, dt)
    initial_momentum = inertia * initial_velocity
    next_momentum = initial_momentum + dt * wp.cross(initial_momentum, initial_velocity)
    return wp.inverse(inertia) * next_momentum


@wp.func
def recover_implicit_euler_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
) -> wp.vec3:
    """Recover implicit Euler's end velocity from its rotation increment."""
    return _logarithmic_velocity(relative_rotation, dt)


@wp.func
def recover_lie_group_midpoint_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
) -> wp.vec3:
    """Recover the endpoint velocity paired with exponential midpoint."""
    midpoint_velocity = _logarithmic_velocity(relative_rotation, dt)
    half_rotation = _exponential_increment(midpoint_velocity, 0.5 * dt)
    initial_momentum = half_rotation * inertia * wp.transpose(half_rotation) * midpoint_velocity
    next_momentum = wp.transpose(relative_rotation) * initial_momentum
    return wp.inverse(inertia) * next_momentum


@wp.func
def recover_midpoint_cayley_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
) -> wp.vec3:
    """Recover the endpoint velocity paired with midpoint-Cayley."""
    identity = wp.identity(3, dtype=float)
    generator = 2.0 * (relative_rotation - identity) * wp.inverse(relative_rotation + identity)
    generator = 0.5 * (generator - wp.transpose(generator))
    midpoint_velocity = (
        wp.vec3(
            generator[2, 1],
            generator[0, 2],
            generator[1, 0],
        )
        / dt
    )
    midpoint_momentum = inertia * midpoint_velocity
    next_momentum = midpoint_momentum + 0.5 * dt * wp.cross(
        midpoint_momentum,
        midpoint_velocity,
    )
    return wp.inverse(inertia) * next_momentum


@wp.func
def recover_vbd_semi_implicit_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
) -> wp.vec3:
    """Match VBD's shortest-arc quaternion difference after the positional solve."""
    return _logarithmic_velocity(relative_rotation, dt)


@wp.func
def recover_rigid_body_end_velocity(
    inertia: wp.mat33,
    relative_rotation: wp.mat33,
    dt: float,
    integrator: int,
) -> wp.vec3:
    """Recover the endpoint velocity paired with the selected integrator."""
    if integrator == EXPLICIT_EULER:
        return recover_explicit_euler_end_velocity(inertia, relative_rotation, dt)
    elif integrator == IMPLICIT_EULER:
        return recover_implicit_euler_end_velocity(inertia, relative_rotation, dt)
    elif integrator == LIE_GROUP_MIDPOINT:
        return recover_lie_group_midpoint_end_velocity(inertia, relative_rotation, dt)
    elif integrator == MIDPOINT_CAYLEY:
        return recover_midpoint_cayley_end_velocity(inertia, relative_rotation, dt)
    elif integrator == VBD_SEMI_IMPLICIT:
        return recover_vbd_semi_implicit_end_velocity(inertia, relative_rotation, dt)
