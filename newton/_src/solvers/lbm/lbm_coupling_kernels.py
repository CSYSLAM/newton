# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Warp kernels that couple Newton rigid bodies with the D3Q27 LBM grid.

Adapted from the Open HOME-LBM coupling kernels
(``envs/lbm3d/lbm_fluid_env_3d_func.py``): body poses are read from Newton's
``State.body_q`` (body-frame origin transforms) plus ``Model.body_com`` to
recover the COM pose, and hydrodynamic wrenches are written into
``State.body_f`` (world-frame, COM-referenced) so that
:class:`newton.solvers.SolverMuJoCo` picks them up as ``xfrc_applied``.
"""

import warp as wp

from .lbm_core import HomeFlow


@wp.kernel
def extract_body_states_from_body_q(
    body_q: wp.array(dtype=wp.transform),          # flat (nbody_total,) body-frame transforms
    body_com: wp.array(dtype=wp.vec3),              # flat (nbody_total,) body-local COM [m]
    body_ids: wp.array(dtype=wp.int32),             # (n_solids,) flat body index
    positions_out: wp.array3d(dtype=wp.float32),    # (nworld, n_solids, 3) COM positions [m]
    quaternions_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 4) quats (w, x, y, z)
):
    """Extract world-frame COM pose of each registered body for every world.

    ``body_ids`` hold flat indices into the (world-major) ``body_q`` array; for
    single-world models this is simply the model body index. Global (static)
    bodies live at the start of the flat array and are shared across worlds.
    """
    world_idx, idx = wp.tid()
    q = body_q[body_ids[idx]]
    pos = wp.transform_point(q, body_com[body_ids[idx]])
    positions_out[world_idx, idx, 0] = pos[0]
    positions_out[world_idx, idx, 1] = pos[1]
    positions_out[world_idx, idx, 2] = pos[2]

    quat = q.q  # wp.quat stores (x, y, z, w)
    quaternions_out[world_idx, idx, 0] = quat[3]  # w
    quaternions_out[world_idx, idx, 1] = quat[0]  # x
    quaternions_out[world_idx, idx, 2] = quat[1]  # y
    quaternions_out[world_idx, idx, 3] = quat[2]  # z


@wp.kernel
def convert_and_update_solid_batch_3d(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,) solid IDs in LBM
    mujoco_positions: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    mujoco_quaternions: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 4)
    mujoco_origins: wp.array(dtype=wp.vec3),  # (n_solids,) - shared across all worlds
    lbm_origins: wp.array(dtype=wp.vec3),  # (n_solids,) - shared across all worlds
    scales: wp.array(dtype=wp.float32),  # (n_solids,) - shared across all worlds
):
    """
    Convert MuJoCo coordinates to LBM and update all worlds in parallel.
    Also updates mesh_transforms for ray casting.
    2D launch: (nworld, n_solids)
    """
    world_idx, body_idx = wp.tid()

    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]
    scale = scales[body_idx]

    # Convert position: (mujoco_pos - mujoco_origin) * scale + lbm_origin
    mujoco_pos_x = mujoco_positions[world_idx, body_idx, 0]
    mujoco_pos_y = mujoco_positions[world_idx, body_idx, 1]
    mujoco_pos_z = mujoco_positions[world_idx, body_idx, 2]

    mujoco_origin = mujoco_origins[body_idx]
    lbm_origin = lbm_origins[body_idx]

    lbm_x = (mujoco_pos_x - mujoco_origin[0]) * scale + lbm_origin[0]
    lbm_y = (mujoco_pos_y - mujoco_origin[1]) * scale + lbm_origin[1]
    lbm_z = (mujoco_pos_z - mujoco_origin[2]) * scale + lbm_origin[2]

    lbm_pos = wp.vec3(lbm_x, lbm_y, lbm_z)
    flow.solid_position[solid_id] = lbm_pos

    # Copy quaternion directly (w, x, y, z)
    w = mujoco_quaternions[world_idx, body_idx, 0]
    x = mujoco_quaternions[world_idx, body_idx, 1]
    y = mujoco_quaternions[world_idx, body_idx, 2]
    z = mujoco_quaternions[world_idx, body_idx, 3]

    flow.solid_quaternion[solid_id] = wp.vec4(w, x, y, z)

    # Update mesh_transforms for ray casting
    # Save current transform to last (for velocity calculation)
    is_initialized = flow.mesh_transforms_initialized[solid_id]
    if is_initialized > 0:
        flow.mesh_transforms_last[solid_id] = flow.mesh_transforms[solid_id]

    # Create new transform: wp.quat uses (x, y, z, w) order internally
    new_transform = wp.transform(lbm_pos, wp.quat(x, y, z, w))
    flow.mesh_transforms[solid_id] = new_transform

    # If first time, also set last to current (zero velocity)
    if is_initialized == 0:
        flow.mesh_transforms_last[solid_id] = new_transform
        flow.mesh_transforms_initialized[solid_id] = 1


@wp.kernel
def extract_forces_torques_physical_3d(
    flows: wp.array(dtype=HomeFlow),  # (nworld,) array of flow objects
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,) solid IDs
    force_conversion: wp.float32,  # Physical force conversion factor: rho * dx^4 / dt^2
    torque_conversion: wp.float32,  # Physical torque conversion factor: rho * dx^5 / dt^2
    forces_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    torques_out: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """
    Extract forces and torques with proper physical unit conversion.

    Physical conversion based on dimensional analysis:
    - LBM uses dimensionless units: rho_lbm=1, dx_lbm=1, dt_lbm=1
    - Force:  F_physical = F_lbm * rho_fluid * dx^4 / dt^2  [N]
    - Torque: tau_physical = tau_lbm * rho_fluid * dx^5 / dt^2  [N*m]

    2D launch: (nworld, n_solids)
    """
    world_idx, body_idx = wp.tid()

    flow = flows[world_idx]
    solid_id = solid_ids[body_idx]

    # Get raw LBM forces (dimensionless)
    force = flow.solid_force[solid_id]
    torque = flow.solid_torque[solid_id]

    # Apply physical unit conversion
    forces_out[world_idx, body_idx, 0] = force[0] * force_conversion
    forces_out[world_idx, body_idx, 1] = force[1] * force_conversion
    forces_out[world_idx, body_idx, 2] = force[2] * force_conversion

    torques_out[world_idx, body_idx, 0] = torque[0] * torque_conversion
    torques_out[world_idx, body_idx, 1] = torque[1] * torque_conversion
    torques_out[world_idx, body_idx, 2] = torque[2] * torque_conversion


@wp.kernel
def fill_body_f_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),  # flat (nbody_total,)
    body_ids: wp.array(dtype=wp.int32),  # (n_solids,) flat body index
    forces: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
    torques: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """Write hydrodynamic wrenches into the flat ``State.body_f`` array.

    ``State.body_f`` is a world-frame external wrench referenced at the body
    COM, matching the LBM solid-position convention. 2D launch: (nworld, n_solids)
    """
    world_idx, idx = wp.tid()
    body_f[body_ids[idx]] = wp.spatial_vector(
        forces[world_idx, idx, 0],
        forces[world_idx, idx, 1],
        forces[world_idx, idx, 2],
        torques[world_idx, idx, 0],
        torques[world_idx, idx, 1],
        torques[world_idx, idx, 2],
    )


@wp.kernel
def extract_all_solid_positions_3d_kernel(
    flows: wp.array(dtype=HomeFlow),  # (nworld,)
    solid_ids: wp.array(dtype=wp.int32),  # (n_solids,)
    solid_positions_dst: wp.array3d(dtype=wp.float32),  # (nworld, n_solids, 3)
):
    """Extract all solids' positions from all LBM solvers.
    2D launch: (nworld, n_solids)
    """
    world_idx, idx = wp.tid()
    solid_id = solid_ids[idx]
    flow = flows[world_idx]
    pos = flow.solid_position[solid_id]
    solid_positions_dst[world_idx, idx, 0] = pos[0]
    solid_positions_dst[world_idx, idx, 1] = pos[1]
    solid_positions_dst[world_idx, idx, 2] = pos[2]
