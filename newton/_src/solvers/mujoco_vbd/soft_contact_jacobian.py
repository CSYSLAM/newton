# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced-coordinate point Jacobians for particle/rigid contact records."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Contacts, State
from ..coupled.solver_coupled import SolverEntry
from .articulation_predictor import MuJoCoSmoothPredictor


@wp.kernel(enable_backward=False)
def _classify_soft_contacts(
    contact_count: wp.array[int],
    contact_shape: wp.array[int],
    shape_to_q_body: wp.array[int],
    q_body_world: wp.array[int],
    endpoint_q_body: wp.array[int],
    contact_world: wp.array[int],
    active: wp.array[int],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        endpoint_q_body[contact] = -1
        contact_world[contact] = -1
        active[contact] = 0
        return
    shape = contact_shape[contact]
    body = shape_to_q_body[shape] if shape >= 0 and shape < shape_to_q_body.shape[0] else -1
    endpoint_q_body[contact] = body
    contact_world[contact] = q_body_world[body] if body >= 0 else -1
    active[contact] = int(body >= 0)


def _make_soft_contact_jacobian_kernel():
    from mujoco_warp._src import support as mjw_support

    @wp.kernel(enable_backward=False)
    def _evaluate_soft_contact_jacobian(
        body_parentid: wp.array[int],
        body_rootid: wp.array[int],
        dof_bodyid: wp.array[int],
        body_isdofancestor: wp.array2d[int],
        subtree_com: wp.array2d[wp.vec3],
        cdof: wp.array2d[wp.spatial_vector],
        active: wp.array[int],
        endpoint_q_body: wp.array[int],
        contact_world: wp.array[int],
        q_body_to_mujoco: wp.array[int],
        body_q: wp.array[wp.transform],
        contact_body_pos: wp.array[wp.vec3],
        jacobian: wp.array3d[float],
    ):
        contact, dof = wp.tid()
        if active[contact] == 0:
            jacobian[contact, 0, dof] = 0.0
            jacobian[contact, 1, dof] = 0.0
            jacobian[contact, 2, dof] = 0.0
            return
        body = endpoint_q_body[contact]
        point = wp.transform_point(body_q[body], contact_body_pos[contact])
        jacobian_point, _ = mjw_support.jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com,
            cdof,
            point,
            q_body_to_mujoco[body],
            dof,
            contact_world[contact],
        )
        jacobian[contact, 0, dof] = jacobian_point[0]
        jacobian[contact, 1, dof] = jacobian_point[1]
        jacobian[contact, 2, dof] = jacobian_point[2]

    return _evaluate_soft_contact_jacobian


class ArticulationSoftContactJacobian:
    """Evaluate q-space body-point Jacobians for canonical soft contacts.

    A soft contact has one rigid endpoint.  When that endpoint is an
    articulated link, the returned Jacobian is the motion of its body-surface
    point; particle motion is handled by the VBD particle block.

    Args:
        predictor: Compact articulated MuJoCo predictor.
        articulation_entry: Ownership entry defining local/global body maps.
    """

    def __init__(self, predictor: MuJoCoSmoothPredictor, articulation_entry: SolverEntry) -> None:
        self.predictor = predictor
        model = predictor.model
        solver = predictor._solver
        data = solver.mjw_data
        mjw_model = solver.mjw_model
        if data is None or mjw_model is None or solver.mjc_body_to_newton is None:
            raise RuntimeError("soft contact Jacobian requires an initialized MuJoCo-Warp model")
        global_to_local = articulation_entry.body_global_to_local.numpy()
        shape_body = model.parent.shape_body.numpy()
        shape_to_q_body = np.full(model.parent.shape_count, -1, dtype=np.int32)
        for shape, global_body in enumerate(shape_body):
            if global_body >= 0 and global_to_local[global_body] >= 0:
                shape_to_q_body[shape] = global_to_local[global_body]
        q_body_world = articulation_entry.view.body_world.numpy().astype(np.int32, copy=True)
        if data.nworld == 1:
            q_body_world[q_body_world < 0] = 0
        elif np.any(q_body_world < 0):
            raise ValueError("a multi-world q-block cannot contain global articulated bodies")
        mjc_to_local = solver.mjc_body_to_newton.numpy()
        q_body_to_mujoco = np.full(articulation_entry.view.body_count, -1, dtype=np.int32)
        for body, world in enumerate(q_body_world):
            matches = np.flatnonzero(mjc_to_local[int(world)] == body)
            if matches.size != 1:
                raise ValueError(f"compact MuJoCo body mapping is not one-to-one for local body {body}")
            q_body_to_mujoco[body] = int(matches[0])
        device = model.device
        self._model = model
        self._data = data
        self._mjw_model = mjw_model
        self._shape_to_q_body = wp.array(shape_to_q_body, dtype=int, device=device)
        self._q_body_world = wp.array(q_body_world, dtype=int, device=device)
        self._q_body_to_mujoco = wp.array(q_body_to_mujoco, dtype=int, device=device)
        self._jacobian_kernel = _make_soft_contact_jacobian_kernel()
        self.capacity = 0
        self.endpoint_q_body: wp.array[int] | None = None
        self.contact_world: wp.array[int] | None = None
        self.active: wp.array[int] | None = None
        self.jacobian: wp.array3d[float] | None = None

    def evaluate(self, state: State, contacts: Contacts) -> wp.array3d[float]:
        """Evaluate body-point Jacobians for the current soft contact buffer."""
        if contacts.soft_contact_max != self.capacity:
            self._allocate(contacts.soft_contact_max)
        assert self.endpoint_q_body is not None
        assert self.contact_world is not None
        assert self.active is not None
        assert self.jacobian is not None
        wp.launch(
            _classify_soft_contacts,
            dim=self.capacity,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                self._shape_to_q_body,
                self._q_body_world,
            ],
            outputs=[self.endpoint_q_body, self.contact_world, self.active],
            device=self._model.device,
        )
        wp.launch(
            self._jacobian_kernel,
            dim=(self.capacity, self._mjw_model.nv),
            inputs=[
                self._mjw_model.body_parentid,
                self._mjw_model.body_rootid,
                self._mjw_model.dof_bodyid,
                self._mjw_model.body_isdofancestor,
                self._data.subtree_com,
                self._data.cdof,
                self.active,
                self.endpoint_q_body,
                self.contact_world,
                self._q_body_to_mujoco,
                state.body_q,
                contacts.soft_contact_body_pos,
            ],
            outputs=[self.jacobian],
            device=self._model.device,
        )
        return self.jacobian

    def _allocate(self, capacity: int) -> None:
        self.capacity = capacity
        device = self._model.device
        self.endpoint_q_body = wp.empty(capacity, dtype=int, device=device)
        self.contact_world = wp.empty(capacity, dtype=int, device=device)
        self.active = wp.empty(capacity, dtype=int, device=device)
        self.jacobian = wp.empty((capacity, 3, self.predictor.qd_count), dtype=float, device=device)
