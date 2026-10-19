# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulated link response of the MuJoCo Warp step for contact coupling.

For an impulse wrench ``p_m`` applied at the center of mass of link ``m``,
one MuJoCo step changes the generalized velocity by ``H^-1 J_m^T p_m``, where
``H`` is the matrix MuJoCo factors to integrate velocities (``M - dt dqfrc/dqvel``
for the implicit integrators, ``M`` with implicit joint damping for Euler)
and ``J_m`` maps generalized velocities to the link's center-of-mass twist.
The resulting twist change of link ``l`` is ``Lambda^-1_lm p_m`` with the link
mobility ``Lambda^-1 = J H^-1 J^T``. Twists are ``(v_com, omega)`` and wrenches
``(force, torque about the center of mass)``, both in world coordinates.

The mobility is linear in the impulse and excludes MuJoCo constraint
activity (contacts, joint limits, equality constraints) and actuator force
clamping, which same-step coupling iterations account for.
"""

from __future__ import annotations

import numpy as np
import warp as wp

# Matches mujoco_warp.IntegratorType.
_EULER = 0
_IMPLICIT = 2
_IMPLICITFAST = 3


@wp.kernel(enable_backward=False)
def _densify_symmetric(
    rows: wp.array[int],
    cols: wp.array[int],
    values: wp.array2d[float],
    dense: wp.array3d[float],
):
    world, entry = wp.tid()
    i = rows[entry]
    j = cols[entry]
    value = values[world, entry]
    dense[world, i, j] = value
    dense[world, j, i] = value


@wp.kernel(enable_backward=False)
def _add_euler_damping(
    dt: wp.array[float],
    damping: wp.array2d[float],
    dense: wp.array3d[float],
):
    world, dof = wp.tid()
    dense[world, dof, dof] = dense[world, dof, dof] + dt[world % dt.shape[0]] * damping[world % damping.shape[0], dof]


@wp.kernel(enable_backward=False)
def _cholesky(nv: int, matrix: wp.array3d[float]):
    """Factor each world's symmetric positive definite matrix into its lower triangle in place."""
    world = wp.tid()
    for j in range(nv):
        diagonal = matrix[world, j, j]
        for k in range(j):
            diagonal -= matrix[world, j, k] * matrix[world, j, k]
        pivot = wp.sqrt(wp.max(diagonal, 1.0e-20))
        matrix[world, j, j] = pivot
        for i in range(j + 1, nv):
            value = matrix[world, i, j]
            for k in range(j):
                value -= matrix[world, i, k] * matrix[world, j, k]
            matrix[world, i, j] = value / pivot


@wp.func
def _jac_dof(
    body_isdofancestor: wp.array2d[int],
    body_rootid: wp.array[int],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    point: wp.vec3,
    body: int,
    dof: int,
    world: int,
):
    """Point and angular velocity per unit dof velocity, as MuJoCo's ``mj_jac``."""
    if body_isdofancestor[body, dof] == 0:
        return wp.vec3(0.0), wp.vec3(0.0)
    offset = point - subtree_com[world, body_rootid[body]]
    motion = cdof[world, dof]
    angular = wp.spatial_top(motion)
    return wp.spatial_bottom(motion) + wp.cross(angular, offset), angular


@wp.kernel(enable_backward=False)
def _link_jacobian(
    link_body: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    body_rootid: wp.array[int],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    xipos: wp.array2d[wp.vec3],
    jacobian: wp.array3d[float],
):
    world, link, dof = wp.tid()
    body = link_body[link]
    linear, angular = _jac_dof(body_isdofancestor, body_rootid, subtree_com, cdof, xipos[world, body], body, dof, world)
    for axis in range(3):
        jacobian[world, 6 * link + axis, dof] = linear[axis]
        jacobian[world, 6 * link + 3 + axis, dof] = angular[axis]


@wp.kernel(enable_backward=False)
def _solve_columns(
    nv: int,
    factor: wp.array3d[float],
    jacobian: wp.array3d[float],
    solution: wp.array3d[float],
):
    """Solve ``H x = J[c]^T`` for each Jacobian row ``c`` with the Cholesky factor."""
    world, column = wp.tid()
    for i in range(nv):
        value = jacobian[world, column, i]
        for k in range(i):
            value -= factor[world, i, k] * solution[world, k, column]
        solution[world, i, column] = value / factor[world, i, i]
    for step in range(nv):
        i = nv - 1 - step
        value = solution[world, i, column]
        for k in range(i + 1, nv):
            value -= factor[world, k, i] * solution[world, k, column]
        solution[world, i, column] = value / factor[world, i, i]


@wp.kernel(enable_backward=False)
def _project(
    nv: int,
    jacobian: wp.array3d[float],
    solution: wp.array3d[float],
    mobility: wp.array3d[float],
):
    world, row, column = wp.tid()
    value = float(0.0)
    for k in range(nv):
        value += jacobian[world, row, k] * solution[world, k, column]
    mobility[world, row, column] = value


class LinkMobility:
    """Evaluate the articulated link mobility of a private MuJoCo Warp solver.

    Args:
        mujoco_solver: Private ``SolverMuJoCo`` using the MuJoCo Warp backend.

    Attributes:
        link_count: Links per world with a Newton body, ``n``.
        link_body: Newton body index of each link slot in the MuJoCo solver's
            model, shape ``(world_count, n)``.
        mobility: Link mobility, shape ``(world_count, 6 n, 6 n)`` [1/kg, 1/(kg m), 1/(kg m^2)].
    """

    def __init__(self, mujoco_solver):
        if mujoco_solver.use_mujoco_cpu:
            raise ValueError("Articulated contact response requires the MuJoCo Warp backend (use_mujoco_cpu=False)")
        self._solver = mujoco_solver
        m = mujoco_solver.mjw_model
        d = mujoco_solver.mjw_data
        device = mujoco_solver.model.device
        body_to_newton = mujoco_solver.mjc_body_to_newton.numpy()
        if body_to_newton.ndim == 1:
            body_to_newton = body_to_newton[None, :]
        template = np.flatnonzero(body_to_newton[0] >= 0)
        if np.any(body_to_newton[:, template] < 0) or np.any(np.delete(body_to_newton, template, axis=1) >= 0):
            raise ValueError("Every MuJoCo world must map the same bodies to Newton")
        self.world_count = d.nworld
        self.link_count = int(template.shape[0])
        self.nv = int(m.nv)
        self.link_body = wp.array(body_to_newton[:, template].astype(np.int32), dtype=int, device=device)
        self._link_mjc_body = wp.array(template.astype(np.int32), dtype=int, device=device)
        size = 6 * self.link_count
        nv = max(self.nv, 1)
        self._h = wp.zeros((self.world_count, nv, nv), dtype=float, device=device)
        self._h_csr = wp.zeros((self.world_count, max(int(m.nC), 1)), dtype=float, device=device)
        self._jacobian = wp.zeros((self.world_count, size, nv), dtype=float, device=device)
        self._solution = wp.zeros((self.world_count, nv, size), dtype=float, device=device)
        self.mobility = wp.zeros((self.world_count, size, size), dtype=float, device=device)
        self._integrator = int(m.opt.integrator)

    def update(self) -> None:
        """Evaluate the mobility at the configuration of the latest MuJoCo step."""
        if self.nv == 0 or self.link_count == 0:
            return
        solver = self._solver
        m = solver.mjw_model
        d = solver.mjw_data
        device = solver.model.device
        with wp.ScopedDevice(device), solver._scoped_mujoco_warp_execution():
            if self._integrator in (_IMPLICIT, _IMPLICITFAST):
                from mujoco_warp._src import derivative

                derivative.deriv_smooth_vel(m, d, self._h_csr)
                values = self._h_csr
            else:
                values = d.M
            self._h.zero_()
            wp.launch(
                _densify_symmetric,
                dim=(self.world_count, int(m.nC)),
                inputs=[m.M_hinit_i, m.M_colind, values],
                outputs=[self._h],
            )
            if self._integrator == _EULER and m.dof_damping.shape[-1]:
                damping = m.dof_damping if m.dof_damping.ndim == 2 else m.dof_damping.reshape((1, -1))
                wp.launch(
                    _add_euler_damping,
                    dim=(self.world_count, self.nv),
                    inputs=[m.opt.timestep, damping],
                    outputs=[self._h],
                )
            wp.launch(_cholesky, dim=self.world_count, inputs=[self.nv], outputs=[self._h])
            wp.launch(
                _link_jacobian,
                dim=(self.world_count, self.link_count, self.nv),
                inputs=[self._link_mjc_body, m.body_isdofancestor, m.body_rootid, d.subtree_com, d.cdof, d.xipos],
                outputs=[self._jacobian],
            )
            size = 6 * self.link_count
            wp.launch(
                _solve_columns,
                dim=(self.world_count, size),
                inputs=[self.nv, self._h, self._jacobian],
                outputs=[self._solution],
            )
            wp.launch(
                _project,
                dim=(self.world_count, size, size),
                inputs=[self.nv, self._jacobian, self._solution],
                outputs=[self.mobility],
            )
