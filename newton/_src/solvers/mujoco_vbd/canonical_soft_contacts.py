# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical particle/rigid contact terms for the articulated q-block."""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Model, State
from .contact_energy import evaluate_projected_isotropic_friction


@wp.func
def _evaluate_soft_contact_point(
    corners: wp.vec3i,
    bary: wp.vec3,
    particle_q: wp.array[wp.vec3],
    particle_q_previous: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_previous: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    contact_shape: int,
    contact_body_pos: wp.vec3,
    contact_body_vel: wp.vec3,
    normal: wp.vec3,
    stiffness: float,
    damping: float,
    friction: float,
    friction_epsilon: float,
    dt: float,
) -> tuple[wp.vec3, wp.mat33]:
    """Return energy gradient/Hessian with respect to the rigid endpoint."""
    particle = corners[0]
    if particle < 0:
        return wp.vec3(0.0), wp.mat33(0.0)
    soft_point = bary[0] * particle_q[particle]
    soft_previous = bary[0] * particle_q_previous[particle]
    radius = particle_radius[particle]
    for corner in range(1, 3):
        index = corners[corner]
        if index >= 0:
            soft_point += bary[corner] * particle_q[index]
            soft_previous += bary[corner] * particle_q_previous[index]
            radius = wp.max(radius, particle_radius[index])

    body = shape_body[contact_shape]
    current = wp.transform_identity()
    previous = wp.transform_identity()
    if body >= 0:
        current = body_q[body]
        previous = body_q_previous[body]
    body_point = wp.transform_point(current, contact_body_pos)
    body_previous = wp.transform_point(previous, contact_body_pos)
    margin = shape_margin[contact_shape] if shape_margin.shape[0] > 0 else 0.0
    penetration = -(wp.dot(normal, soft_point - body_point) - radius - margin)
    if penetration <= 0.0 or stiffness <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)

    relative_motion = soft_point - soft_previous
    body_velocity = (body_point - body_previous) / dt + wp.transform_vector(current, contact_body_vel)
    relative_motion = relative_motion - body_velocity * dt
    normal_force = stiffness * penetration
    force_on_soft = normal * normal_force
    hessian = stiffness * wp.outer(normal, normal)
    if wp.dot(normal, relative_motion) < 0.0 and damping > 0.0:
        damping_hessian = (damping / dt) * wp.outer(normal, normal)
        force_on_soft = force_on_soft - damping_hessian * relative_motion
        hessian = hessian + damping_hessian
    if friction > 0.0 and normal_force > 0.0:
        friction_force, friction_hessian = evaluate_projected_isotropic_friction(
            friction,
            normal_force,
            normal,
            relative_motion,
            friction_epsilon * dt,
        )
        force_on_soft = force_on_soft + friction_force
        hessian = hessian + friction_hessian
    # The body-point energy gradient is the negative physical body force,
    # which is equal to the physical force on the soft endpoint.
    return force_on_soft, hessian


@wp.kernel(enable_backward=False)
def _evaluate_canonical_soft_contacts(
    contact_count: wp.array[int],
    contact_indices: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_body_vel: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_q_previous: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_previous: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    gradient: wp.array[wp.vec3],
    hessian: wp.array[wp.mat33],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        gradient[contact] = wp.vec3(0.0)
        hessian[contact] = wp.mat33(0.0)
        return
    shape = contact_shape[contact]
    if shape < 0:
        gradient[contact] = wp.vec3(0.0)
        hessian[contact] = wp.mat33(0.0)
        return
    stiffness = 0.5 * (soft_contact_ke + shape_material_ke[shape])
    damping = 0.5 * (soft_contact_kd + shape_material_kd[shape])
    friction = wp.sqrt(soft_contact_mu * shape_material_mu[shape])
    contact_gradient, contact_hessian = _evaluate_soft_contact_point(
        contact_indices[contact],
        contact_barycentric[contact],
        particle_q,
        particle_q_previous,
        particle_radius,
        body_q,
        body_q_previous,
        shape_body,
        shape_margin,
        shape,
        contact_body_pos[contact],
        contact_body_vel[contact],
        contact_normal[contact],
        stiffness,
        damping,
        friction,
        friction_epsilon,
        dt,
    )
    gradient[contact] = contact_gradient
    hessian[contact] = contact_hessian


class CanonicalSoftContacts:
    """Evaluate particle/rigid reaction terms shared with the q-block.

    The law is copied from VBD's body-particle contact primitive.  It is
    evaluated on global canonical contacts so the q-side and VBD-side observe
    the same geometry, while only the articulated endpoint is accumulated
    here.

    Args:
        model: Shared scene model.
        friction_epsilon: Friction regularization length [m].
    """

    def __init__(self, model: Model, *, friction_epsilon: float = 1.0e-2) -> None:
        self.model = model
        self.friction_epsilon = friction_epsilon
        self.capacity = 0
        self.gradient: wp.array[wp.vec3] | None = None
        self.hessian: wp.array[wp.mat33] | None = None

    def evaluate(self, state: State, state_previous: State, contacts: Contacts, dt: float) -> None:
        """Evaluate soft-contact q-endpoint gradients and Hessians."""
        if contacts.soft_contact_max == 0:
            return
        if contacts.soft_contact_max != self.capacity:
            self.capacity = contacts.soft_contact_max
            self.gradient = wp.empty(self.capacity, dtype=wp.vec3, device=self.model.device)
            self.hessian = wp.empty(self.capacity, dtype=wp.mat33, device=self.model.device)
        assert self.gradient is not None
        assert self.hessian is not None
        wp.launch(
            _evaluate_canonical_soft_contacts,
            dim=self.capacity,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                state.particle_q,
                state_previous.particle_q,
                self.model.particle_radius,
                state.body_q,
                state_previous.body_q,
                self.model.shape_body,
                self.model.shape_margin,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_material_mu,
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
                self.friction_epsilon,
                dt,
            ],
            outputs=[self.gradient, self.hessian],
            device=self.model.device,
        )
