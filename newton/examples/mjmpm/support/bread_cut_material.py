# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Example-local, irreversible blade damage for an elastoplastic loaf.

This is a finite-width material weakening law, not a sharp crack or CPIC.
It keeps mass and elastic compressibility, removing tensile/shear cohesion
only in the swept blade-edge band. Damage is evaluated from physical poses.
"""

import warp as wp


@wp.kernel
def damage_at_blade(
    particles: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    tool_local: wp.transform,
    half_band: float,
    damage: wp.array[float],
):
    i = wp.tid()
    blade = wp.mul(body_q[body], tool_local)
    point = wp.transform_point(wp.transform_inverse(blade), particles[i])
    # Follow the straight edge and its curved tip in the physical knife frame.
    # Include the particle support around the edge, before collider contact
    # pushes the sample away. No operation deletes or repositions a sample.
    edge = float(-0.055)
    if point[0] > 0.230:
        edge = -0.032 + (point[0] - 0.230) * (0.017 / 0.010)
    elif point[0] > 0.218:
        edge = -0.045 + (point[0] - 0.218) * (0.013 / 0.012)
    elif point[0] > 0.205:
        edge = -0.052 + (point[0] - 0.205) * (0.007 / 0.013)
    elif point[0] > 0.190:
        edge = -0.055 + (point[0] - 0.190) * (0.003 / 0.015)
    if (
        point[0] >= 0.055
        and point[0] <= 0.245
        and wp.abs(point[1]) <= half_band
        and wp.abs(point[2] - edge) <= half_band
    ):
        damage[i] = 1.0


@wp.kernel
def apply_damage(
    damage: wp.array[float],
    intact_shear: float,
    intact_tension: float,
    yield_stress: wp.array[float],
    tensile_yield_ratio: wp.array[float],
):
    i = wp.tid()
    remaining = 1.0 - damage[i]
    yield_stress[i] = remaining * intact_shear
    tensile_yield_ratio[i] = remaining * intact_tension
