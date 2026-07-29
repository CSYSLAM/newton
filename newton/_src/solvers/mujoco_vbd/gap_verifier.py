# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side endpoint gap verification for DAT trial acceptance."""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from ...sim import Contacts, Model, State
from ...sim.contacts import contact_surface_separation


@wp.kernel(enable_backward=False)
def _interpolate_body_positions(
    start: wp.array[wp.transform], end: wp.array[wp.transform], alpha: float, output: wp.array[wp.transform]
):
    body = wp.tid()
    transform = start[body]
    wp.transform_set_translation(
        transform,
        wp.lerp(wp.transform_get_translation(start[body]), wp.transform_get_translation(end[body]), alpha),
    )
    wp.transform_set_rotation(
        transform,
        wp.quat_slerp(wp.transform_get_rotation(start[body]), wp.transform_get_rotation(end[body]), alpha),
    )
    output[body] = transform


@wp.kernel(enable_backward=False)
def _interpolate_particle_positions(
    start: wp.array[wp.vec3], end: wp.array[wp.vec3], alpha: float, output: wp.array[wp.vec3]
):
    particle = wp.tid()
    output[particle] = wp.lerp(start[particle], end[particle], alpha)


@wp.kernel(enable_backward=False)
def _verify_rigid_contact_gaps(
    contact_count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    tolerance: float,
    violation: wp.array[int],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    body0 = shape_body[shape0[contact]]
    body1 = shape_body[shape1[contact]]
    transform0 = body_q[body0] if body0 >= 0 else wp.transform_identity()
    transform1 = body_q[body1] if body1 >= 0 else wp.transform_identity()
    # Shape margins are a pre-contact buffer used by the contact law. DAT's
    # feasibility condition is physical surface separation, not clearance by
    # the sum of those solver margins; otherwise a resting table contact is
    # incorrectly retried forever despite having no geometric overlap.
    gap = contact_surface_separation(
        wp.transform_point(transform0, point0[contact]),
        wp.transform_point(transform1, point1[contact]),
        normal[contact],
        0.0,
        0.0,
    )
    if gap < -tolerance:
        wp.atomic_max(violation, 0, 1)


@wp.kernel(enable_backward=False)
def _verify_soft_contact_gaps(
    contact_count: wp.array[int],
    contact_indices: wp.array[wp.vec3i],
    contact_barycentric: wp.array[wp.vec3],
    contact_shape: wp.array[int],
    contact_body_pos: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_margin: wp.array[float],
    tolerance: float,
    violation: wp.array[int],
):
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape = contact_shape[contact]
    corners = contact_indices[contact]
    if shape < 0 or corners[0] < 0:
        return
    bary = contact_barycentric[contact]
    soft_point = bary[0] * particle_q[corners[0]]
    radius = particle_radius[corners[0]]
    for corner in range(1, 3):
        index = corners[corner]
        if index >= 0:
            soft_point += bary[corner] * particle_q[index]
            radius = wp.max(radius, particle_radius[index])
    body = shape_body[shape]
    transform = body_q[body] if body >= 0 else wp.transform_identity()
    body_point = wp.transform_point(transform, contact_body_pos[contact])
    # As above, shape_margin is a solver buffer rather than physical volume.
    # The particle radius remains part of the actual cloth/soft geometry.
    if wp.dot(contact_normal[contact], soft_point - body_point) - radius < -tolerance:
        wp.atomic_max(violation, 0, 1)


class EndpointGapVerifier:
    """Check that a trial endpoint has no reported contact penetration.

    This verifier consumes the same canonical contact records as the q and VBD
    blocks.  It is intentionally an acceptance gate, not a replacement for
    swept candidate generation: a rejected endpoint is retried through smaller
    joint/VBD substeps before any persistent contact state is committed.

    Args:
        model: Shared model owning contact geometry.
        tolerance: Allowed negative contact gap [m].
    """

    def __init__(self, model: Model, *, tolerance: float = 1.0e-5) -> None:
        if tolerance < 0.0:
            raise ValueError("gap tolerance must be non-negative")
        self.model = model
        self.tolerance = tolerance
        self._violation = wp.zeros(1, dtype=int, device=model.device)

    def has_violation(self, state: State, contacts: Contacts) -> bool:
        """Return whether any reported rigid or soft contact exceeds tolerance."""
        if self.model.device.is_capturing:
            raise RuntimeError("MuJoCo-VBD endpoint DAT verification is not CUDA-graph-capturable yet")
        self._violation.zero_()
        if contacts.rigid_contact_max:
            wp.launch(
                _verify_rigid_contact_gaps,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    self.model.shape_body,
                    state.body_q,
                    self.tolerance,
                ],
                outputs=[self._violation],
                device=self.model.device,
            )
        if contacts.soft_contact_max:
            wp.launch(
                _verify_soft_contact_gaps,
                dim=contacts.soft_contact_max,
                inputs=[
                    contacts.soft_contact_count,
                    contacts.soft_contact_indices,
                    contacts.soft_contact_barycentric,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_normal,
                    state.particle_q,
                    self.model.particle_radius,
                    state.body_q,
                    self.model.shape_body,
                    self.model.shape_margin,
                    self.tolerance,
                ],
                outputs=[self._violation],
                device=self.model.device,
            )
        return bool(self._violation.numpy()[0])


class SweptStateProbe:
    """Sample body and particle motion between accepted and trial states.

    This is a conservative rejection aid while the primitive-specific
    time-of-impact path is implemented.  Each sample obtains fresh collision
    records for an interpolated geometry state, so a high-speed pass-through
    that has no endpoint contact can still force adaptive subdivision.

    Args:
        model: Shared model owning collision geometry.
        verifier: Endpoint gap verifier used on each sampled contact buffer.
        samples: Number of equally-spaced interior geometry samples.
    """

    def __init__(
        self,
        model: Model,
        verifier: EndpointGapVerifier,
        *,
        samples: int = 1,
        collide: Callable[[State, Contacts], None] | None = None,
    ) -> None:
        if samples < 0:
            raise ValueError("swept probe sample count must be non-negative")
        self.model = model
        self.verifier = verifier
        self.samples = samples
        self._collide = model.collide if collide is None else collide
        self.state = model.state(requires_grad=False)
        self.contacts = model.contacts()

    def has_violation(self, start: State, end: State) -> bool:
        """Return whether any interior geometry sample reports penetration."""
        if self.samples == 0:
            return False
        if self.model.device.is_capturing:
            raise RuntimeError("MuJoCo-VBD swept DAT probes are not CUDA-graph-capturable yet")
        for sample in range(self.samples):
            alpha = float(sample + 1) / float(self.samples + 1)
            self._interpolate(start, end, alpha)
            self._collide(self.state, self.contacts)
            if self.verifier.has_violation(self.state, self.contacts):
                return True
        return False

    def _interpolate(self, start: State, end: State, alpha: float) -> None:
        self.interpolate_into(start, end, alpha, self.state)

    def interpolate_into(self, start: State, end: State, alpha: float, output: State) -> None:
        """Interpolate geometry into ``output`` while preserving other state fields."""
        output.assign(start)
        if start.body_q is not None and end.body_q is not None:
            wp.launch(
                _interpolate_body_positions,
                dim=start.body_q.shape[0],
                inputs=[start.body_q, end.body_q, alpha],
                outputs=[output.body_q],
                device=self.model.device,
            )
        if start.particle_q is not None and end.particle_q is not None:
            wp.launch(
                _interpolate_particle_positions,
                dim=start.particle_q.shape[0],
                inputs=[start.particle_q, end.particle_q, alpha],
                outputs=[output.particle_q],
                device=self.model.device,
            )


class VBDDATProjector:
    """Commit a VBD candidate through a conservative shared-alpha projection.

    The projector runs *inside* the VBD outer iteration. A candidate endpoint
    is never made persistent until a common alpha for its free-body and
    particle displacement has passed the same collision/gap checks. This is a
    global-island fallback: it is conservative, but keeps every owner on one
    feasible path until a sparse island implementation replaces it.

    Args:
        model: Shared model owning the trial states.
        verifier: Physical-gap acceptance checker.
        swept_probe: Conservative interior collision probe.
        collide: Collision callback for canonical contacts.
        max_bisections: Number of device-candidate bisection refinements.
    """

    def __init__(
        self,
        model: Model,
        verifier: EndpointGapVerifier,
        swept_probe: SweptStateProbe,
        *,
        collide: Callable[[State, Contacts], None],
        max_bisections: int,
    ) -> None:
        self.model = model
        self.verifier = verifier
        self.swept_probe = swept_probe
        self._collide = collide
        self.max_bisections = max_bisections
        self._candidate = model.state(requires_grad=False)
        self._safe = model.state(requires_grad=False)

    def project(self, start: State, trial: State, contacts: Contacts) -> bool:
        """Truncate ``trial`` in place to a feasible VBD candidate.

        Returns:
            Whether the unscaled candidate was feasible. A truncated candidate
            is still feasible and remains written to ``trial``.
        """
        self._candidate.assign(trial)
        if self._is_feasible(start, self._candidate, contacts):
            return True

        self._safe.assign(start)
        low = 0.0
        high = 1.0
        for _ in range(self.max_bisections):
            alpha = 0.5 * (low + high)
            self.swept_probe.interpolate_into(start, self._candidate, alpha, trial)
            if self._is_feasible(start, trial, contacts):
                low = alpha
                self._safe.assign(trial)
            else:
                high = alpha
        trial.assign(self._safe)
        self._collide(trial, contacts)
        return False

    def _is_feasible(self, start: State, candidate: State, contacts: Contacts) -> bool:
        self._collide(candidate, contacts)
        return not self.verifier.has_violation(candidate, contacts) and not self.swept_probe.has_violation(start, candidate)
