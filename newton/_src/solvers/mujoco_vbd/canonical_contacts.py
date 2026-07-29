# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical rigid-contact batch owned by the MuJoCo--VBD solver."""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Model, State
from ...sim.contacts import contact_surface_separation
from .contact_energy import evaluate_canonical_rigid_contact


@wp.kernel(enable_backward=False)
def _evaluate_canonical_rigid_contacts(
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_previous: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    multiplier: wp.array[wp.vec3],
    friction_c0: wp.array[wp.vec3],
    hard_contact: wp.array[int],
    tangential_stiffness_scale: float,
    friction_epsilon: float,
    dt: float,
    gradient: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
    force0: wp.array[wp.vec3],
    force1: wp.array[wp.vec3],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        gradient[contact] = wp.vec3(0.0)
        hessian[contact] = wp.mat33(0.0)
        force0[contact] = wp.vec3(0.0)
        force1[contact] = wp.vec3(0.0)
        return

    shape0_id = shape0[contact]
    shape1_id = shape1[contact]
    body0 = shape_body[shape0_id]
    body1 = shape_body[shape1_id]
    transform0 = wp.transform_identity()
    transform1 = wp.transform_identity()
    transform0_previous = wp.transform_identity()
    transform1_previous = wp.transform_identity()
    com0 = wp.vec3(0.0)
    com1 = wp.vec3(0.0)
    if body0 >= 0:
        transform0 = body_q[body0]
        transform0_previous = body_q_previous[body0]
        com0 = body_com[body0]
    if body1 >= 0:
        transform1 = body_q[body1]
        transform1_previous = body_q_previous[body1]
        com1 = body_com[body1]

    x0 = wp.transform_point(transform0, point0[contact])
    x1 = wp.transform_point(transform1, point1[contact])
    penetration = -contact_surface_separation(x0, x1, normal[contact], margin0[contact], margin1[contact])
    stiffness = 0.5 * (shape_material_ke[shape0_id] + shape_material_ke[shape1_id])
    damping = 0.5 * (shape_material_kd[shape0_id] + shape_material_kd[shape1_id])
    friction = wp.sqrt(shape_material_mu[shape0_id] * shape_material_mu[shape1_id])
    g, h, f0, f1 = evaluate_canonical_rigid_contact(
        transform0,
        transform1,
        transform0_previous,
        transform1_previous,
        com0,
        com1,
        point0[contact],
        point1[contact],
        offset0[contact],
        offset1[contact],
        normal[contact],
        penetration,
        stiffness,
        tangential_stiffness_scale * stiffness,
        damping,
        multiplier[contact],
        friction,
        friction_epsilon,
        hard_contact[contact],
        dt,
        friction_c0[contact],
    )
    gradient[contact] = g
    hessian[contact] = h
    force0[contact] = f0
    force1[contact] = f1


class CanonicalRigidContacts:
    """Evaluate one contact energy shared by all MuJoCo--VBD blocks.

    The data layout deliberately follows the existing VBD contact law but is
    local to this solver.  Existing ``SolverVBD`` contact state is neither
    read nor modified.  Multiplier and friction-anchor updates are performed
    by the future unified outer loop after a DAT-accepted trial.

    Args:
        model: Shared Newton model that owns canonical contact material data.
        tangential_stiffness_scale: Ratio of tangential to normal hard-contact
            stiffness.
        friction_epsilon: Friction regularization length [m].
    """

    def __init__(
        self,
        model: Model,
        *,
        tangential_stiffness_scale: float = 1.0,
        friction_epsilon: float = 1.0e-2,
    ) -> None:
        if tangential_stiffness_scale < 0.0:
            raise ValueError("tangential_stiffness_scale must be non-negative")
        if friction_epsilon <= 0.0:
            raise ValueError("friction_epsilon must be positive")
        self.model = model
        self.tangential_stiffness_scale = tangential_stiffness_scale
        self.friction_epsilon = friction_epsilon
        self.capacity = 0
        self.multiplier: wp.array[wp.vec3] | None = None
        self.friction_c0: wp.array[wp.vec3] | None = None
        self.hard_contact: wp.array[int] | None = None
        self.gradient: wp.array[wp.vec3] | None = None
        self.hessian: wp.array[wp.mat33] | None = None
        self.force0: wp.array[wp.vec3] | None = None
        self.force1: wp.array[wp.vec3] | None = None

    def evaluate(self, state: State, state_previous: State, contacts: Contacts, dt: float) -> None:
        """Evaluate current contact gradients and Hessians.

        Args:
            state: Current global trial state.
            state_previous: Beginning-of-step global state.
            contacts: Canonical collision records for this trial.
            dt: Timestep [s].
        """
        if dt <= 0.0:
            raise ValueError("canonical contact timestep must be positive")
        if contacts.rigid_contact_max == 0:
            return
        if state.body_q is None or state_previous.body_q is None:
            raise ValueError("canonical rigid contacts require rigid body transforms")
        if contacts.rigid_contact_max != self.capacity:
            self._allocate(contacts.rigid_contact_max)
        assert self.multiplier is not None
        assert self.friction_c0 is not None
        assert self.hard_contact is not None
        assert self.gradient is not None
        assert self.hessian is not None
        assert self.force0 is not None
        assert self.force1 is not None
        if (
            self.model.shape_material_ke is None
            or self.model.shape_material_kd is None
            or self.model.shape_material_mu is None
        ):
            raise ValueError("canonical rigid contacts require per-shape contact materials")
        wp.launch(
            _evaluate_canonical_rigid_contacts,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_material_mu,
                state.body_q,
                state_previous.body_q,
                self.model.body_com,
                self.multiplier,
                self.friction_c0,
                self.hard_contact,
                self.tangential_stiffness_scale,
                self.friction_epsilon,
                dt,
            ],
            outputs=[self.gradient, self.hessian, self.force0, self.force1],
            device=self.model.device,
        )

    def _allocate(self, capacity: int) -> None:
        self.capacity = capacity
        device = self.model.device
        self.multiplier = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.friction_c0 = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.hard_contact = wp.zeros(capacity, dtype=int, device=device)
        self.gradient = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.hessian = wp.empty(capacity, dtype=wp.mat33, device=device)
        self.force0 = wp.empty(capacity, dtype=wp.vec3, device=device)
        self.force1 = wp.empty(capacity, dtype=wp.vec3, device=device)
