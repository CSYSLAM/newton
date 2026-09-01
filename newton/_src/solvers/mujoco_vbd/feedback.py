# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified point/edge/face and rigid cross-contact wrench harvest.

See ``DESIGN.md`` section 13. This removes the previous full-surface proxy
limitation (``coupling_prepare_proxy_contacts`` raised ``NotImplementedError``
for edge/face records) by consuming the unified contact layout::

    soft_contact_indices = (p, -1, -1)     point
    soft_contact_indices = (v0, v1, -1)    edge
    soft_contact_indices = (v0, v1, v2)    face

Soft feedback calls the same point/edge/face force function used by the VBD
rigid-body solve, avoiding a second approximate contact law.
"""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, State
from .ownership import MuJoCoVBDOwnership
from .vbd.rigid_vbd_kernels import _evaluate_body_particle_contact_reaction

__all__ = [
    "MuJoCoVBDFeedback",
    "harvest_rigid_proxy_wrenches_kernel",
    "harvest_unified_soft_proxy_wrenches_kernel",
]


@wp.kernel
def harvest_unified_soft_proxy_wrenches_kernel(
    dt: float,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_max: int,
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    shape_margin: wp.array[float],
    body_to_proxy_slot: wp.array[wp.int32],
    contact_penalty_k: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    friction_epsilon: float,
    out_proxy_wrench: wp.array[wp.spatial_vector],
    out_contact_force: wp.array[wp.vec3],
):
    """Harvest equal-and-opposite point/edge/face wrenches (``DESIGN.md`` 13.3)."""
    tid = wp.tid()
    if tid >= soft_contact_count[0] or tid >= soft_contact_max:
        return

    shape = soft_contact_shape[tid]
    if shape < 0:
        return
    body = shape_body[shape]
    if body < 0:
        return
    slot = body_to_proxy_slot[body]
    if slot < 0:
        return  # Only MuJoCo proxy bodies receive feedback.

    corners = soft_contact_indices[tid]
    if corners[0] < 0:
        return

    X_wb = body_q[body]
    com_world = wp.transform_point(X_wb, body_com[body])
    force_on_body, torque_on_body, _, _, _ = _evaluate_body_particle_contact_reaction(
        dt,
        tid,
        X_wb,
        body_q_prev[body],
        com_world,
        particle_q,
        particle_q_prev,
        particle_radius,
        body_q_prev,
        body_q,
        body_qd,
        body_com,
        shape_body,
        friction_epsilon,
        contact_penalty_k,
        contact_material_kd,
        contact_material_mu,
        soft_contact_indices,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_body_vel,
        soft_contact_normal,
        soft_contact_barycentric,
        shape_margin,
    )

    out_contact_force[tid] = -force_on_body
    wp.atomic_add(out_proxy_wrench, body, wp.spatial_vector(force_on_body, torque_on_body))


@wp.kernel
def harvest_rigid_proxy_wrenches_kernel(
    rigid_contact_count: wp.array[wp.int32],
    rigid_contact_max: int,
    body0: wp.array[wp.int32],
    body1: wp.array[wp.int32],
    point0_world: wp.array[wp.vec3],
    point1_world: wp.array[wp.vec3],
    force_on_body1: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_to_proxy_slot: wp.array[wp.int32],
    out_proxy_wrench: wp.array[wp.spatial_vector],
):
    """Reduce only M-V rigid-rigid cross contacts onto proxy bodies (13.4)."""
    tid = wp.tid()
    if tid >= rigid_contact_count[0] or tid >= rigid_contact_max:
        return

    b0 = body0[tid]
    b1 = body1[tid]
    f1 = force_on_body1[tid]

    slot1 = wp.int32(-1)
    if b1 >= 0:
        slot1 = body_to_proxy_slot[b1]
    if slot1 >= 0:
        com1 = wp.transform_point(body_q[b1], body_com[b1])
        torque1 = wp.cross(point1_world[tid] - com1, f1)
        wp.atomic_add(out_proxy_wrench, b1, wp.spatial_vector(f1, torque1))

    slot0 = wp.int32(-1)
    if b0 >= 0:
        slot0 = body_to_proxy_slot[b0]
    if slot0 >= 0:
        f0 = -f1
        com0 = wp.transform_point(body_q[b0], body_com[b0])
        torque0 = wp.cross(point0_world[tid] - com0, f0)
        wp.atomic_add(out_proxy_wrench, b0, wp.spatial_vector(f0, torque0))


class MuJoCoVBDFeedback:
    """Contact-native raw wrench on MuJoCo-owned proxy bodies (``DESIGN.md`` 13.5)."""

    def __init__(
        self,
        model,
        ownership: MuJoCoVBDOwnership,
        diagnostics,
        vbd_backend,
        contacts: Contacts,
        friction_epsilon: float = 1.0e-4,
    ) -> None:
        self.model = model
        self.ownership = ownership
        self.diagnostics = diagnostics
        self.vbd_backend = vbd_backend
        self.friction_epsilon = float(friction_epsilon)
        self.device = model.device
        self.out_wrench = diagnostics.feedback_wrench_raw
        self._contact_force = wp.zeros(max(int(contacts.soft_contact_max), 1), dtype=wp.vec3, device=self.device)

    def clear(self) -> None:
        self.out_wrench.zero_()

    def harvest(
        self,
        vbd_state_in: State,
        vbd_state_out: State,
        contacts: Contacts,
        dt: float,
    ) -> wp.array:
        """Return the contact-native raw wrench on proxy bodies (``DESIGN.md`` 13.5).

        Requires the VBD material/penalty state that the VBD backend exposes for
        the current contact set. Missing state is an explicit failure, never an
        aggregate-momentum fallback.
        """
        self.clear()
        penalty = self.vbd_backend.body_particle_contact_penalty_k()
        material_kd = self.vbd_backend.body_particle_contact_material_kd()
        material_mu = self.vbd_backend.body_particle_contact_material_mu()
        if penalty is None or material_kd is None or material_mu is None:
            raise RuntimeError(
                "Two-way feedback requires VBD body-particle contact material/penalty state; it is "
                "not available for the current contact set. This is a construction/topology error, "
                "not a fallback condition (DESIGN 13.5)."
            )

        soft_max = int(getattr(contacts, "soft_contact_max", 0))
        if soft_max > 0 and penalty.shape[0] >= soft_max:
            if self._contact_force.shape[0] < soft_max:
                raise RuntimeError(
                    "Two-way soft-contact capacity changed after construction; rebuild SolverMuJoCoVBD "
                    "before CUDA Graph capture."
                )
            particle_q_prev = self.vbd_backend.particle_q_prev()
            wp.launch(
                harvest_unified_soft_proxy_wrenches_kernel,
                dim=soft_max,
                inputs=[
                    dt,
                    contacts.soft_contact_count,
                    soft_max,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    vbd_state_out.particle_q,
                    particle_q_prev,
                    self.model.particle_radius,
                    vbd_state_out.body_q,
                    self.vbd_backend.body_q_prev(),
                    vbd_state_out.body_qd,
                    self.model.body_com,
                    self.model.shape_body,
                    self.model.shape_margin,
                    self.ownership.body_to_proxy_slot,
                    penalty,
                    material_kd,
                    material_mu,
                    self.friction_epsilon,
                    self.out_wrench,
                    self._contact_force,
                ],
                device=self.device,
            )

        self._harvest_rigid(vbd_state_out, contacts, dt)
        _ = vbd_state_in
        return self.out_wrench

    def _harvest_rigid(self, vbd_state_out: State, contacts: Contacts, dt: float) -> None:
        rigid_max = int(getattr(contacts, "rigid_contact_max", 0))
        if rigid_max <= 0:
            return
        body0, body1, point0, point1, force_on_body1, rigid_contact_count = (
            self.vbd_backend.collect_rigid_contact_forces(contacts, vbd_state_out, dt)
        )
        wp.launch(
            harvest_rigid_proxy_wrenches_kernel,
            dim=rigid_max,
            inputs=[
                rigid_contact_count,
                rigid_max,
                body0,
                body1,
                point0,
                point1,
                force_on_body1,
                vbd_state_out.body_q,
                self.model.body_com,
                self.ownership.body_to_proxy_slot,
                self.out_wrench,
            ],
            device=self.device,
        )

    def validate_finite(self) -> None:
        """Deferred to the solver's device-side nonfinite flag check (DESIGN 14.1)."""

    def reset(self, world_mask: wp.array | None = None) -> None:
        _ = world_mask
        self.clear()
