# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Runtime flow-control kernels for the D3Q27 LBM solver.

Ported from the Open HOME-LBM project (``envs/lbm3d/lbm_runtime.py``): uniform
inflow initialization, per-boundary velocity updates, and smooth ellipsoidal
body forces used to seed vortex shedding.
"""

import warp as wp

from .lbm_core import HomeFlow, cx_d3q27, cy_d3q27, cz_d3q27, w_d3q27


@wp.kernel
def set_uniform_flow_3d_kernel(flows: wp.array(dtype=HomeFlow), ux: float, uy: float, uz: float):
    """Initialize every lattice cell with a uniform D3Q27 equilibrium flow."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    rho = 1.0
    population = wp.types.vector(length=27, dtype=wp.float32)
    speed_squared = ux * ux + uy * uy + uz * uz
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        population[i] = w_d3q27[i] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared
        )

    inv_rho = 1.0 / rho
    pixx = population[1] + population[2] + population[7] + population[8] + population[9] + population[10] + population[13] + population[14] + population[15] + population[16] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    pixy = (population[7] + population[8] + population[19] + population[20] + population[21] + population[22]) - (population[13] + population[14] + population[23] + population[24] + population[25] + population[26])
    pixz = (population[9] + population[10] + population[19] + population[20] + population[23] + population[24]) - (population[15] + population[16] + population[21] + population[22] + population[25] + population[26])
    piyy = population[3] + population[4] + population[7] + population[8] + population[11] + population[12] + population[13] + population[14] + population[17] + population[18] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    piyz = (population[11] + population[12] + population[19] + population[20] + population[25] + population[26]) - (population[17] + population[18] + population[21] + population[22] + population[23] + population[24])
    pizz = population[5] + population[6] + population[9] + population[10] + population[11] + population[12] + population[15] + population[16] + population[17] + population[18] + population[19] + population[20] + population[21] + population[22] + population[23] + population[24] + population[25] + population[26]
    cs2_local = pixx
    pixx = pixx * inv_rho - cs2_local
    pixy = pixy * inv_rho
    pixz = pixz * inv_rho
    piyy = piyy * inv_rho - cs2_local
    piyz = piyz * inv_rho
    pizz = pizz * inv_rho - cs2_local

    flow.rho[x, y, z] = rho
    flow.rho_post[x, y, z] = rho
    flow.u[x, y, z] = wp.vec3(ux, uy, uz)
    flow.u_post[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx[x, y, z] = pixx
    flow.Sxx_post[x, y, z] = pixx
    flow.Syy[x, y, z] = piyy
    flow.Syy_post[x, y, z] = piyy
    flow.Szz[x, y, z] = pizz
    flow.Szz_post[x, y, z] = pizz
    flow.Sxy[x, y, z] = pixy
    flow.Sxy_post[x, y, z] = pixy
    flow.Sxz[x, y, z] = pixz
    flow.Sxz_post[x, y, z] = pixz
    flow.Syz[x, y, z] = piyz
    flow.Syz_post[x, y, z] = piyz
    flow.forcex[x, y, z] = 0.0
    flow.forcey[x, y, z] = 0.0
    flow.forcez[x, y, z] = 0.0


@wp.kernel
def set_local_force_3d_kernel(
    flows: wp.array(dtype=HomeFlow),
    center_x: float,
    center_y: float,
    center_z: float,
    radius_x: float,
    radius_y: float,
    radius_z: float,
    force_x: float,
    force_y: float,
    force_z: float,
):
    """Apply a smooth ellipsoidal perturbation to seed vortex shedding."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    dx = (float(x) - center_x) / wp.max(radius_x, 1.0e-6)
    dy = (float(y) - center_y) / wp.max(radius_y, 1.0e-6)
    dz = (float(z) - center_z) / wp.max(radius_z, 1.0e-6)
    weight = wp.max(0.0, 1.0 - dx * dx - dy * dy - dz * dz)
    flow.forcex[x, y, z] = force_x * weight
    flow.forcey[x, y, z] = force_y * weight
    flow.forcez[x, y, z] = force_z * weight


@wp.kernel
def set_boundary_velocity_3d_kernel(
    flows: wp.array(dtype=HomeFlow),
    boundary_idx: int,
    ux: float,
    uy: float,
    uz: float,
):
    """Update one velocity boundary for every world."""
    world_idx = wp.tid()
    flows[world_idx].bc_value[boundary_idx] = wp.vec3(ux, uy, uz)


@wp.kernel
def set_solid_velocities_3d_kernel(
    flows: wp.array(dtype=HomeFlow),
    solid_ids: wp.array(dtype=wp.int32),
    linear_v: wp.array(dtype=wp.vec3),
    angle_v: wp.array(dtype=wp.vec3),
):
    """Record rigid-body velocities into each flow for the cut-cell fallback path.

    The immersed-boundary kernel uses frame-to-frame mesh-transform differences
    to compute boundary velocity; ``linear_v`` / ``angle_v`` are the fallback
    used on the first frame before a transform history exists.
    """
    world_idx, idx = wp.tid()
    flow = flows[world_idx]
    solid_id = solid_ids[idx]
    flow.linear_v[solid_id] = linear_v[idx]
    flow.angle_v[solid_id] = angle_v[idx]
