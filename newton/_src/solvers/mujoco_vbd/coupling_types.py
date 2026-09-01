# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private coupling endpoint enum, adapter protocol, and proxy state-transfer kernels.

Vendored from ``newton._src.solvers.coupled.interface`` and
``newton._src.solvers.coupled.proxy_utils`` so that ``mujoco_vbd`` has no runtime
dependency on the shared ``coupled`` package (DESIGN 3.1/3.2). This module only
provides the fixed endpoint enum, the hook protocol used by the private MuJoCo
and VBD cores, and the proxy sync/harvest kernels their default hooks call. It
contains no generic entry/group dispatcher.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any

import warp as wp

from ...geometry import ParticleFlags
from ...sim import BodyFlags, StateFlags

if TYPE_CHECKING:
    from ...sim import Contacts, State

# ------------------------------------------------------------------
# 1. Sync proxy states
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def sync_proxy_states_kernel(
    src_body_q: wp.array[wp.transform],
    src_body_qd: wp.array[wp.spatial_vector],
    source_local_to_proxy_local: wp.array[int],
    dst_body_q: wp.array[wp.transform],
    dst_body_qd: wp.array[wp.spatial_vector],
):
    """Copy body poses and velocities from a source solver to proxy bodies in a destination solver.

    Args:
        src_body_q: Source solver begin-of-step body transforms.
        src_body_qd: Source solver begin-of-step body velocities.
        source_local_to_proxy_local: Dense map from source-local body id to
            proxy-local body id. ``-1`` means no proxy exists for that source
            body.
        dst_body_q: Destination solver body transforms (written for proxies).
        dst_body_qd: Destination solver body velocities (written for proxies).
    """
    source_local_id = wp.tid()
    proxy_local_id = source_local_to_proxy_local[source_local_id]

    if proxy_local_id >= 0:
        dst_body_q[proxy_local_id] = src_body_q[source_local_id]
        dst_body_qd[proxy_local_id] = src_body_qd[source_local_id]


@wp.kernel(enable_backward=False)
def sync_proxy_particles_kernel(
    src_particle_q: wp.array[wp.vec3],
    src_particle_qd: wp.array[wp.vec3],
    source_local_to_proxy_local: wp.array[int],
    dst_particle_q: wp.array[wp.vec3],
    dst_particle_qd: wp.array[wp.vec3],
):
    """Copy particle positions and velocities from a source solver to proxy particles."""
    source_local_id = wp.tid()
    proxy_local_id = source_local_to_proxy_local[source_local_id]

    if proxy_local_id >= 0:
        dst_particle_q[proxy_local_id] = src_particle_q[source_local_id]
        dst_particle_qd[proxy_local_id] = src_particle_qd[source_local_id]


# ------------------------------------------------------------------
# 2. Rewind proxy velocities
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def subtract_proxy_body_forces_kernel(
    body_gravity_acceleration: wp.array[wp.vec3],
    dst_body_f: wp.array[wp.spatial_vector],
    coupling_forces: wp.array[wp.spatial_vector],
    body_local_to_proxy_global: wp.array[int],
    dst_body_mass: wp.array[float],
    dst_body_inv_mass: wp.array[float],
):
    """Subtract lagged proxy feedback and gravity from destination body force inputs.

    Args:
        body_gravity_acceleration: Per-body acceleration applied internally by
            the destination solver's gravity-like forces [m/s^2].
        dst_body_f: Destination body force inputs (written in-place).
        coupling_forces: Spatial forces previously applied to the driving solver,
            indexed by global proxy body id.
        body_local_to_proxy_global: Dense map from local body id to global
            proxy body id. ``-1`` entries are skipped.
        dst_body_mass: Destination body masses [kg].
        dst_body_inv_mass: Destination inverse masses.
    """
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    f = coupling_forces[global_id]

    inv_m = dst_body_inv_mass[local_id]
    g = body_gravity_acceleration[local_id]
    f_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        f_grav = dst_body_mass[local_id] * g

    dst_body_f[local_id] = -f - wp.spatial_vector(f_grav, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel(enable_backward=False)
def subtract_proxy_particle_forces_kernel(
    dt: float,
    particle_gravity_acceleration: wp.array[wp.vec3],
    dst_particle_f: wp.array[wp.vec3],
    coupling_forces: wp.array[wp.vec3],
    particle_local_to_proxy_global: wp.array[int],
    dst_particle_inv_mass: wp.array[float],
    dst_particle_qd: wp.array[wp.vec3],
):
    """Subtract default velocity-level feedback, particle force inputs, and gravity."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    inv_m = dst_particle_inv_mass[local_id]
    delta_v = dt * inv_m * (coupling_forces[global_id] + dst_particle_f[local_id])

    g = particle_gravity_acceleration[local_id]
    delta_v_grav = wp.vec3(0.0, 0.0, 0.0)
    if inv_m > 0.0:
        delta_v_grav = dt * g

    dst_particle_qd[local_id] = dst_particle_qd[local_id] - (delta_v + delta_v_grav)


# ------------------------------------------------------------------
# 3. Harvest proxy feedback
# ------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def harvest_proxy_momentum_forces_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.spatial_vector],
    qd_after: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    out_coupling_forces: wp.array[wp.spatial_vector],
):
    """Estimate proxy feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = body_local_to_proxy_global[local_id]
    if global_id < 0:
        return

    dv = wp.spatial_top(qd_after[local_id]) - wp.spatial_top(qd_before[local_id])
    dw = wp.spatial_bottom(qd_after[local_id]) - wp.spatial_bottom(qd_before[local_id])

    m = body_mass[local_id]
    I_body = body_inertia[local_id]
    r = wp.transform_get_rotation(body_q[local_id])

    f = m * dv / dt
    tau = wp.quat_rotate(r, I_body * wp.quat_rotate_inv(r, dw)) / dt

    wp.atomic_add(out_coupling_forces, global_id, wp.spatial_vector(f, tau))


@wp.kernel(enable_backward=False)
def harvest_proxy_particle_momentum_forces_kernel(
    dt: float,
    particle_local_to_proxy_global: wp.array[int],
    qd_before: wp.array[wp.vec3],
    qd_after: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    active_flag: int,
    out_coupling_forces: wp.array[wp.vec3],
):
    """Estimate proxy particle feedback force from destination velocity change."""
    local_id = wp.tid()
    global_id = particle_local_to_proxy_global[local_id]
    if global_id < 0:
        return
    if (particle_flags[local_id] & active_flag) == 0:
        return

    dv = qd_after[local_id] - qd_before[local_id]
    m = particle_mass[local_id]

    f = m * dv / dt
    wp.atomic_add(out_coupling_forces, global_id, f)


@wp.kernel(enable_backward=False)
def stash_proxy_forces_kernel(
    proxy_ids_global: wp.array[int],
    coupling_forces: wp.array[Any],
    out_previous_coupling_forces: wp.array[Any],
):
    """Save the current proxy feedback for a later relaxation blend."""
    i = wp.tid()
    out_previous_coupling_forces[i] = coupling_forces[proxy_ids_global[i]]


@wp.kernel(enable_backward=False)
def blend_proxy_forces_kernel(
    proxy_relaxation: float,
    proxy_ids_global: wp.array[int],
    previous_coupling_forces: wp.array[Any],
    coupling_forces: wp.array[Any],
):
    """Blend harvested proxy feedback with the saved lagged value."""
    i = wp.tid()
    global_id = proxy_ids_global[i]
    coupling_forces[global_id] = (
        proxy_relaxation * coupling_forces[global_id] + (1.0 - proxy_relaxation) * previous_coupling_forces[i]
    )


@wp.kernel(enable_backward=False)
def filter_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_inv_mass: wp.array[float],
    proxy_flag: int,
):
    """Invalidate proxy-vs-static and proxy-vs-proxy rigid contacts."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    body0 = shape_body[s0] if s0 >= 0 and s0 < shape_body.shape[0] else -1
    body1 = shape_body[s1] if s1 >= 0 and s1 < shape_body.shape[0] else -1

    is_proxy0 = 0
    if body0 >= 0 and body0 < body_flags.shape[0]:
        if (body_flags[body0] & proxy_flag) != 0:
            is_proxy0 = 1
    is_proxy1 = 0
    if body1 >= 0 and body1 < body_flags.shape[0]:
        if (body_flags[body1] & proxy_flag) != 0:
            is_proxy1 = 1

    is_static0 = 0
    if body0 < 0:
        is_static0 = 1
    elif body0 < body_inv_mass.shape[0] and body_inv_mass[body0] == 0.0:
        is_static0 = 1

    is_static1 = 0
    if body1 < 0:
        is_static1 = 1
    elif body1 < body_inv_mass.shape[0] and body_inv_mass[body1] == 0.0:
        is_static1 = 1

    discard = 0
    if is_proxy0 == 1 and is_proxy1 == 1:
        discard = 1
    if is_proxy0 == 1 and is_static1 == 1:
        discard = 1
    if is_proxy1 == 1 and is_static0 == 1:
        discard = 1

    if discard == 1:
        if s0 >= 0:
            rigid_contact_shape0[contact_id] = -s0 - 2
        if s1 >= 0:
            rigid_contact_shape1[contact_id] = -s1 - 2


@wp.kernel(enable_backward=False)
def restore_filtered_proxy_rigid_contacts_kernel(
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[wp.int32],
    rigid_contact_shape1: wp.array[wp.int32],
):
    """Restore contacts temporarily encoded by proxy contact filtering."""
    contact_id = wp.tid()
    if contact_id >= rigid_contact_count[0]:
        return

    s0 = rigid_contact_shape0[contact_id]
    s1 = rigid_contact_shape1[contact_id]
    if s0 < -1:
        rigid_contact_shape0[contact_id] = -s0 - 2
    if s1 < -1:
        rigid_contact_shape1[contact_id] = -s1 - 2


__all__ = ["CouplingInterface"]


class CouplingInterface:
    """Marker mixin for solvers that participate in coupled simulations.

    .. experimental::

    Inheriting buys into the coupling contract:

    - Override hook methods on the solver class to provide custom behavior.
      Otherwise, the mixin's generic defaults are used.
    - Override a hook and raise :class:`NotImplementedError` when no generic
      default can produce a meaningful result for the solver.

    ``EndpointKind`` stays nested because it is coupling-specific. Input update
    notifications reuse :class:`newton.StateFlags`.
    """

    class EndpointKind(IntEnum):
        """Kinds of model endpoints addressed by coupling hooks."""

        BODY = 0
        PARTICLE = 1

    def coupling_eval_effective_mass(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out: wp.array[float],
    ) -> None:
        """Evaluate scalar effective masses for coupling endpoints.

        Args:
            endpoint_kind: Endpoint kinds.
            endpoint_index: Endpoint-local body or particle ids.
            endpoint_local_pos: Body-frame endpoint positions [m].
            out: Output effective masses [kg].
        """
        del endpoint_local_pos
        if out.shape[0] == 0:
            return

        model = self.model
        body_inv_mass = getattr(model, "body_inv_mass", None)
        particle_inv_mass = getattr(model, "particle_inv_mass", None)
        if body_inv_mass is not None and particle_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_kernel,
                dim=out.shape[0],
                inputs=[
                    endpoint_kind,
                    endpoint_index,
                    body_inv_mass,
                    particle_inv_mass,
                    out,
                ],
                device=model.device,
            )
        elif body_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_body_kernel,
                dim=out.shape[0],
                inputs=[endpoint_kind, endpoint_index, body_inv_mass, out],
                device=model.device,
            )
        elif particle_inv_mass is not None:
            wp.launch(
                _coupling_eval_effective_mass_particle_kernel,
                dim=out.shape[0],
                inputs=[endpoint_kind, endpoint_index, particle_inv_mass, out],
                device=model.device,
            )
        else:
            wp.launch(_coupling_zero_mass_kernel, dim=out.shape[0], inputs=[out], device=model.device)

    def coupling_eval_effective_mass_block(
        self,
        endpoint_kind: wp.array[int],
        endpoint_index: wp.array[int],
        endpoint_local_pos: wp.array[wp.vec3],
        out_mass: wp.array[float],
        out_inertia: wp.array[wp.mat33] | None = None,
    ) -> None:
        """Evaluate effective mass and inertia blocks for coupling endpoints.

        Args:
            endpoint_kind: Endpoint kinds.
            endpoint_index: Endpoint-local body or particle ids.
            endpoint_local_pos: Body-frame endpoint positions [m].
            out_mass: Output effective masses [kg].
            out_inertia: Optional output body inertia tensors [kg m^2]. Body
                effective inertia must not be smaller than modeled inertia
                around any axis.
        """
        self.coupling_eval_effective_mass(endpoint_kind, endpoint_index, endpoint_local_pos, out_mass)
        if out_inertia is None or out_inertia.shape[0] == 0:
            return

        model = self.model
        body_mass = getattr(model, "body_mass", None)
        body_inertia = getattr(model, "body_inertia", None)
        if body_mass is None or body_inertia is None:
            wp.launch(
                _coupling_zero_inertia_kernel,
                dim=out_inertia.shape[0],
                inputs=[out_inertia],
                device=model.device,
            )
            return

        wp.launch(
            _coupling_eval_effective_inertia_kernel,
            dim=out_inertia.shape[0],
            inputs=[
                endpoint_kind,
                endpoint_index,
                body_mass,
                body_inertia,
                out_mass,
                out_inertia,
            ],
            device=model.device,
        )

    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """React to coupler-produced public input updates.

        ``flags`` uses :class:`~newton.StateFlags` bits for both kinematic
        state arrays and public force-input buffers.
        """
        del state, flags, iteration_restart, dt

    def coupling_supports_inertial_property_refresh(self) -> bool:
        """Return whether inertial property refresh is safe during graph capture.

        Solvers that read mass and inertia arrays directly, or can refresh
        their derived inertial buffers with device work only, should override
        this to return ``True`` and provide a graph-capturable implementation
        of :meth:`notify_model_changed` for BODY_INERTIAL_PROPERTIES.
        """
        return False

    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        """Return whether the solver consumes edge and face soft contacts."""
        return False

    def coupling_eval_gravity_acceleration(
        self,
        out_body_acceleration: wp.array[wp.vec3] | None,
        out_particle_acceleration: wp.array[wp.vec3] | None,
    ) -> None:
        """Evaluate solver-applied gravity-like acceleration for all local entities.

        The coupled solvers cache these arrays at initialization and refresh
        them on relevant model changes. Solvers that apply scaled or compensated
        gravity should override this hook so proxy and ADMM coupling can remove
        exactly the acceleration the sub-solver will apply internally.

        Args:
            out_body_acceleration: Optional output per local body [m/s^2].
            out_particle_acceleration: Optional output per local particle [m/s^2].
        """
        model = self.model
        if out_body_acceleration is not None and out_body_acceleration.shape[0] > 0:
            wp.launch(
                _coupling_eval_body_gravity_acceleration_kernel,
                dim=out_body_acceleration.shape[0],
                inputs=[model.gravity, model.body_world],
                outputs=[out_body_acceleration],
                device=model.device,
            )
        if out_particle_acceleration is not None and out_particle_acceleration.shape[0] > 0:
            wp.launch(
                _coupling_eval_particle_gravity_acceleration_kernel,
                dim=out_particle_acceleration.shape[0],
                inputs=[model.gravity, model.particle_world],
                outputs=[out_particle_acceleration],
                device=model.device,
            )

    def coupling_rewind_proxy_body(
        self,
        body_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.spatial_vector],
        body_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Rewind lagged proxy-body feedback, gravity acceleration and external forces
        before the destination solve, so those are not double-counted.

        Implementations may update either ``state.body_qd`` or ``state.body_f``.
        """
        del dt
        if body_local_to_proxy_global.shape[0] == 0 or state.body_f is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_body_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                body_gravity_acceleration,
                state.body_f,
                coupling_forces,
                body_local_to_proxy_global,
                model.body_mass,
                model.body_inv_mass,
            ],
            device=model.device,
        )

    def coupling_rewind_proxy_particle(
        self,
        particle_local_to_proxy_global: wp.array[int],
        state: State,
        coupling_forces: wp.array[wp.vec3],
        particle_gravity_acceleration: wp.array[wp.vec3],
        dt: float,
    ) -> None:
        """Rewind lagged proxy-body feedback, gravity acceleration and external forces
        before the destination solve, so those are not double-counted.

        Implementations may update either ``state.particle_qd`` or ``state.particle_f``.
        """
        if particle_local_to_proxy_global.shape[0] == 0 or state.particle_qd is None:
            return

        model = self.model
        wp.launch(
            subtract_proxy_particle_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                particle_gravity_acceleration,
                state.particle_f,
                coupling_forces,
                particle_local_to_proxy_global,
                model.particle_inv_mass,
                state.particle_qd,
            ],
            device=model.device,
        )

    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Accumulate proxy-body feedback from destination momentum change."""
        del state, contacts
        if body_local_to_proxy_global.shape[0] == 0:
            return
        if state_out.body_qd is None:
            raise ValueError("Default body proxy harvest requires state_out.body_qd")
        if dt <= 0.0:
            raise ValueError("Default body proxy harvest requires dt > 0")

        model = self.model
        wp.launch(
            harvest_proxy_momentum_forces_kernel,
            dim=body_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                body_local_to_proxy_global,
                body_qd_before,
                state_out.body_qd,
                model.body_mass,
                model.body_inertia,
                state_out.body_q,
                out_body_f,
            ],
            device=model.device,
        )

    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Accumulate proxy-particle feedback from destination momentum change."""
        del state, contacts
        if particle_local_to_proxy_global.shape[0] == 0:
            return
        if state_out.particle_qd is None:
            raise ValueError("Default particle proxy harvest requires state_out.particle_qd")
        if dt <= 0.0:
            raise ValueError("Default particle proxy harvest requires dt > 0")

        model = self.model
        wp.launch(
            harvest_proxy_particle_momentum_forces_kernel,
            dim=particle_local_to_proxy_global.shape[0],
            inputs=[
                float(dt),
                particle_local_to_proxy_global,
                particle_qd_before,
                state_out.particle_qd,
                model.particle_mass,
                model.particle_flags,
                int(ParticleFlags.ACTIVE),
                out_particle_f,
            ],
            device=model.device,
        )

    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Prepare contacts for a proxy destination solve.

        The generic momentum harvest treats proxy feedback as a destination
        momentum change. Proxy-static and proxy-proxy rigid contacts therefore
        must not be passed through as solver contacts because they would feed
        constraints between virtual objects back to the source.
        """
        del state, contacts_freshly_detected
        if contacts is None or contacts.rigid_contact_count is None or contacts.rigid_contact_max == 0:
            return contacts

        model = self.model
        wp.launch(
            filter_proxy_rigid_contacts_kernel,
            dim=contacts.rigid_contact_shape0.shape[0],
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                model.shape_body,
                model.body_flags,
                model.body_inv_mass,
                int(BodyFlags.PROXY),
            ],
            device=model.device,
        )
        return contacts


@wp.kernel(enable_backward=False)
def _coupling_eval_body_gravity_acceleration_kernel(
    gravity: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = gravity[body_world[i]]


@wp.kernel(enable_backward=False)
def _coupling_eval_particle_gravity_acceleration_kernel(
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    out: wp.array[wp.vec3],
):
    i = wp.tid()
    out[i] = gravity[particle_world[i]]


@wp.func
def _mass_from_inverse(inv_mass: float) -> float:
    if inv_mass == 0.0:
        return 0.0
    return 1.0 / inv_mass


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    body_inv_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    kind = endpoint_kind[i]
    index = endpoint_index[i]
    inv_mass = 0.0

    if kind == wp.static(int(CouplingInterface.EndpointKind.BODY)):
        if index >= 0 and index < body_inv_mass.shape[0]:
            inv_mass = body_inv_mass[index]
    elif kind == wp.static(int(CouplingInterface.EndpointKind.PARTICLE)):
        if index >= 0 and index < particle_inv_mass.shape[0]:
            inv_mass = particle_inv_mass[index]

    out[i] = _mass_from_inverse(inv_mass)


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_body_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    mass = 0.0
    index = endpoint_index[i]
    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.BODY)) and index >= 0:
        if index < inv_mass.shape[0]:
            mass = _mass_from_inverse(inv_mass[index])
    out[i] = mass


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_mass_particle_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    inv_mass: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    mass = 0.0
    index = endpoint_index[i]
    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.PARTICLE)) and index >= 0:
        if index < inv_mass.shape[0]:
            mass = _mass_from_inverse(inv_mass[index])
    out[i] = mass


@wp.kernel(enable_backward=False)
def _coupling_zero_mass_kernel(out: wp.array[float]):
    out[wp.tid()] = 0.0


@wp.kernel(enable_backward=False)
def _coupling_eval_effective_inertia_kernel(
    endpoint_kind: wp.array[int],
    endpoint_index: wp.array[int],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    out_mass: wp.array[float],
    out_inertia: wp.array[wp.mat33],
):
    i = wp.tid()
    index = endpoint_index[i]
    inertia = wp.mat33(0.0)

    if endpoint_kind[i] == wp.static(int(CouplingInterface.EndpointKind.BODY)) and index >= 0:
        if index < body_inertia.shape[0]:
            inertia = body_inertia[index]
            if index < body_mass.shape[0]:
                mass = body_mass[index]
                if mass > 0.0:
                    inertia = inertia * wp.max(out_mass[i] / mass, 1.0)

    out_inertia[i] = inertia


@wp.kernel(enable_backward=False)
def _coupling_zero_inertia_kernel(out_inertia: wp.array[wp.mat33]):
    out_inertia[wp.tid()] = wp.mat33(0.0)


CouplingEndpointKind = CouplingInterface.EndpointKind
