# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced-coordinate Jacobians for canonical Newton contact records."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Contacts, State
from ..coupled.solver_coupled import SolverEntry
from .articulation_predictor import MuJoCoSmoothPredictor


@wp.kernel(enable_backward=False)
def _classify_rigid_contact_q_endpoints(
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_to_q_body: wp.array[int],
    q_body_to_mujoco: wp.array[int],
    q_body_world: wp.array[int],
    endpoint0_q_body: wp.array[int],
    endpoint1_q_body: wp.array[int],
    contact_world: wp.array[int],
    active: wp.array[int],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        endpoint0_q_body[contact] = -1
        endpoint1_q_body[contact] = -1
        contact_world[contact] = -1
        active[contact] = 0
        return

    body0 = -1
    body1 = -1
    shape0_id = shape0[contact]
    shape1_id = shape1[contact]
    if shape0_id >= 0 and shape0_id < shape_to_q_body.shape[0]:
        body0 = shape_to_q_body[shape0_id]
    if shape1_id >= 0 and shape1_id < shape_to_q_body.shape[0]:
        body1 = shape_to_q_body[shape1_id]

    world = -1
    if body0 >= 0:
        world = q_body_world[body0]
    if body1 >= 0:
        if world >= 0 and q_body_world[body1] != world:
            body0 = -1
            body1 = -1
        else:
            world = q_body_world[body1]

    endpoint0_q_body[contact] = body0
    endpoint1_q_body[contact] = body1
    contact_world[contact] = world
    active[contact] = int(world >= 0 and (body0 >= 0 or body1 >= 0))


def _make_contact_relative_jacobian_kernel():
    """Create the optional MuJoCo-Warp Jacobian kernel lazily."""
    from mujoco_warp._src import support as mjw_support

    @wp.kernel(enable_backward=False)
    def _evaluate_contact_relative_jacobian(
        body_parentid: wp.array[int],
        body_rootid: wp.array[int],
        dof_bodyid: wp.array[int],
        body_isdofancestor: wp.array2d[int],
        subtree_com: wp.array2d[wp.vec3],
        cdof: wp.array2d[wp.spatial_vector],
        active: wp.array[int],
        endpoint0_q_body: wp.array[int],
        endpoint1_q_body: wp.array[int],
        contact_world: wp.array[int],
        q_body_to_mujoco: wp.array[int],
        body_q: wp.array[wp.transform],
        point0_local: wp.array[wp.vec3],
        point1_local: wp.array[wp.vec3],
        jacobian: wp.array3d[float],
    ):
        contact, dof = wp.tid()
        if active[contact] == 0:
            jacobian[contact, 0, dof] = 0.0
            jacobian[contact, 1, dof] = 0.0
            jacobian[contact, 2, dof] = 0.0
            return

        world = contact_world[contact]
        result = wp.vec3(0.0)
        body0 = endpoint0_q_body[contact]
        if body0 >= 0:
            point0 = wp.transform_point(body_q[body0], point0_local[contact])
            jacp0, _ = mjw_support.jac_dof(
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                subtree_com,
                cdof,
                point0,
                q_body_to_mujoco[body0],
                dof,
                world,
            )
            result = result - jacp0

        body1 = endpoint1_q_body[contact]
        if body1 >= 0:
            point1 = wp.transform_point(body_q[body1], point1_local[contact])
            jacp1, _ = mjw_support.jac_dof(
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                subtree_com,
                cdof,
                point1,
                q_body_to_mujoco[body1],
                dof,
                world,
            )
            result = result + jacp1

        jacobian[contact, 0, dof] = result[0]
        jacobian[contact, 1, dof] = result[1]
        jacobian[contact, 2, dof] = result[2]

    return _evaluate_contact_relative_jacobian


class ArticulationContactJacobian:
    """Evaluate q-space relative point Jacobians from ``Contacts`` records.

    The output represents the canonical contact displacement
    ``x_shape1 - x_shape0``.  A contact involving an articulated link and a
    VBD-owned body therefore has exactly one q endpoint; a self-contact of two
    articulated links has both endpoints and receives their exact Jacobian
    difference.  This is the direct coupling path used by q-block assembly,
    not a proxy-force or ADMM approximation.

    Args:
        predictor: Compact articulated MuJoCo predictor.
        articulation_entry: Ownership entry defining local/global body maps.
    """

    def __init__(self, predictor: MuJoCoSmoothPredictor, articulation_entry: SolverEntry) -> None:
        self.predictor = predictor
        self._entry = articulation_entry
        self._model = predictor.model
        solver = predictor._solver
        data = solver.mjw_data
        mjw_model = solver.mjw_model
        if data is None or mjw_model is None or solver.mjc_body_to_newton is None:
            raise RuntimeError("articulation contact Jacobian requires an initialized MuJoCo-Warp model")

        global_to_local = articulation_entry.body_global_to_local.numpy()
        shape_body = self._model.parent.shape_body.numpy()
        shape_to_q_body = np.full(self._model.parent.shape_count, -1, dtype=np.int32)
        for shape, global_body in enumerate(shape_body):
            if global_body >= 0 and global_to_local[global_body] >= 0:
                shape_to_q_body[shape] = global_to_local[global_body]

        local_body_world = articulation_entry.view.body_world.numpy().astype(np.int32, copy=True)
        # A one-world Newton model stores template bodies with world ``-1``;
        # MuJoCo-Warp materializes that template as its sole runtime world.
        if data.nworld == 1:
            local_body_world[local_body_world < 0] = 0
        elif np.any(local_body_world < 0):
            raise ValueError("a multi-world q-block cannot contain global articulated bodies")
        mjc_to_local = solver.mjc_body_to_newton.numpy()
        q_body_to_mujoco = np.full(articulation_entry.view.body_count, -1, dtype=np.int32)
        for local_body, world in enumerate(local_body_world):
            matches = np.flatnonzero(mjc_to_local[int(world)] == local_body)
            if matches.size != 1:
                raise ValueError(
                    "compact MuJoCo articulation body mapping is not one-to-one; "
                    f"local body {local_body} has {matches.size} MuJoCo bodies"
                )
            q_body_to_mujoco[local_body] = int(matches[0])

        device = self._model.device
        self._shape_to_q_body = wp.array(shape_to_q_body, dtype=int, device=device)
        self._q_body_to_mujoco = wp.array(q_body_to_mujoco, dtype=int, device=device)
        self._q_body_world = wp.array(local_body_world, dtype=int, device=device)
        self._relative_jacobian_kernel = _make_contact_relative_jacobian_kernel()
        self.capacity = 0
        self.endpoint0_q_body: wp.array[int] | None = None
        self.endpoint1_q_body: wp.array[int] | None = None
        self.contact_world: wp.array[int] | None = None
        self.active: wp.array[int] | None = None
        self.jacobian: wp.array3d[float] | None = None

    def evaluate(self, state: State, contacts: Contacts) -> wp.array3d[float]:
        """Evaluate relative Jacobians for all active canonical rigid contacts.

        Args:
            state: Current compact articulation state whose FK matches the
                predictor's MuJoCo-Warp data.
            contacts: Canonical global Newton contact buffer.

        Returns:
            Relative point Jacobians ``[rigid_contact_max, 3, nv]``.  Rows
            outside ``contacts.rigid_contact_count`` or without a q endpoint
            are zero; :attr:`active` identifies valid rows.
        """
        if state.body_q is None:
            raise ValueError("articulation contact Jacobian requires body transforms")
        if contacts.rigid_contact_max != self.capacity:
            self._allocate(contacts.rigid_contact_max)
        assert self.endpoint0_q_body is not None
        assert self.endpoint1_q_body is not None
        assert self.contact_world is not None
        assert self.active is not None
        assert self.jacobian is not None

        data = self.predictor._solver.mjw_data
        mjw_model = self.predictor._solver.mjw_model
        assert data is not None
        assert mjw_model is not None
        device = self._model.device
        wp.launch(
            _classify_rigid_contact_q_endpoints,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self._shape_to_q_body,
                self._q_body_to_mujoco,
                self._q_body_world,
            ],
            outputs=[self.endpoint0_q_body, self.endpoint1_q_body, self.contact_world, self.active],
            device=device,
        )
        wp.launch(
            self._relative_jacobian_kernel,
            dim=(self.capacity, mjw_model.nv),
            inputs=[
                mjw_model.body_parentid,
                mjw_model.body_rootid,
                mjw_model.dof_bodyid,
                mjw_model.body_isdofancestor,
                data.subtree_com,
                data.cdof,
                self.active,
                self.endpoint0_q_body,
                self.endpoint1_q_body,
                self.contact_world,
                self._q_body_to_mujoco,
                state.body_q,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
            ],
            outputs=[self.jacobian],
            device=device,
        )
        return self.jacobian

    def _allocate(self, capacity: int) -> None:
        device = self._model.device
        self.capacity = capacity
        self.endpoint0_q_body = wp.empty(capacity, dtype=int, device=device)
        self.endpoint1_q_body = wp.empty(capacity, dtype=int, device=device)
        self.contact_world = wp.empty(capacity, dtype=int, device=device)
        self.active = wp.empty(capacity, dtype=int, device=device)
        self.jacobian = wp.empty(
            (capacity, 3, self.predictor.qd_count),
            dtype=float,
            device=device,
        )
