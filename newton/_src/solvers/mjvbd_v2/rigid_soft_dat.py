# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Adapter for detection-time rigid-soft DAT in the full MJVBDV2 backend."""

import numpy as np
import warp as wp

from . import rigid_soft_dat_kernels as kernels


@wp.kernel
def _difference(q: wp.array[wp.vec3], reference: wp.array[wp.vec3], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = q[i] - reference[i]


@wp.kernel
def _apply_particles(
    reference: wp.array[wp.vec3],
    displacement: wp.array[wp.vec3],
    factors: wp.array[float],
    max_displacement: float,
    q: wp.array[wp.vec3],
    chebyshev_excluded: wp.array[wp.int32],
):
    i = wp.tid()
    dx = displacement[i]
    length = wp.length(dx)
    t = factors[i]
    if length > max_displacement:
        t = wp.min(t, max_displacement / length)
    if chebyshev_excluded and t < 1.0 - 1.0e-6:
        chebyshev_excluded[i] = 1
    q[i] = reference[i] + t * dx


class RigidSoftDAT:
    """Constrain updates relative to the input contact detection of each substep.

    The MJVBDV2 coupler detects contacts on the state passed into VBD. References
    stay separate from self-contact references, which may refresh mid-substep.
    """

    def __init__(self, model, query_margin: float, relaxation: float, interval: bool):
        if not np.isfinite(query_margin) or query_margin <= 0.0:
            raise ValueError("Rigid-soft DAT requires a positive contact query margin")
        if not np.isfinite(relaxation) or not 0.0 < relaxation < 1.0:
            raise ValueError("Rigid-soft DAT relaxation must be in (0, 1)")
        self.model = model
        self.relaxation = relaxation
        self.interval = interval
        self.particle_reference = wp.clone(model.particle_q)
        self.body_reference = wp.clone(model.body_q)
        self.displacements = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.particle_factors = wp.ones(model.particle_count, dtype=float, device=model.device)
        self.body_factors = wp.ones(model.body_count, dtype=float, device=model.device)
        radii = model.particle_radius.numpy()
        minimum_radius = float(radii.min()) if radii.size else 0.0
        # Ignoring positive shape margins only tightens the budget.
        self.budget = 0.5 * relaxation * (query_margin + minimum_radius)
        bounds = np.zeros(model.body_count, dtype=np.float32)
        transforms = model.shape_transform.numpy()
        com = model.body_com.numpy()
        collision_radius = model.shape_collision_radius.numpy()
        scale = model.shape_scale.numpy()
        for shape, body in enumerate(model.shape_body.numpy()):
            if body < 0:
                continue
            local_com = np.asarray(
                wp.transform_point(wp.transform_inverse(wp.transform(*transforms[shape])), wp.vec3(*com[body]))
            )
            vertices = getattr(model.shape_source[shape], "vertices", None)
            if vertices is not None and len(vertices):
                reach = np.linalg.norm(np.asarray(vertices) * scale[shape] - local_com, axis=1).max()
            else:
                reach = np.linalg.norm(local_com) + collision_radius[shape]
            bounds[body] = max(bounds[body], reach)
        self.body_radius = wp.array(bounds, dtype=float, device=model.device)
        self.body_budget = wp.full(model.body_count, self.budget, dtype=float, device=model.device)
        self.state = None
        self.contacts = None

    def begin(self, state, contacts):
        if contacts is None:
            raise ValueError("Rigid-soft DAT requires fresh input-state contacts every substep")
        self.state = state
        self.contacts = contacts
        self.particle_reference.assign(state.particle_q)
        self.body_reference.assign(state.body_q)

    def apply(self, solver):
        state, contacts, model = self.state, self.contacts, self.model
        self.particle_factors.fill_(1.0)
        self.body_factors.fill_(1.0)
        wp.launch(
            _difference,
            model.particle_count,
            [state.particle_q, self.particle_reference, self.displacements],
            device=model.device,
        )
        wp.launch(
            kernels.apply_rigid_soft_truncation,
            4096,
            [
                contacts.soft_contact_count,
                contacts.soft_contact_indices,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_normal,
                contacts.soft_contact_barycentric,
                model.shape_body,
                self.particle_reference,
                self.displacements,
                self.body_reference,
                state.body_q,
                model.body_com,
                self.relaxation,
                self.interval,
                self.particle_factors,
                self.body_factors,
            ],
            device=model.device,
        )
        wp.launch(
            _apply_particles,
            model.particle_count,
            [
                self.particle_reference,
                self.displacements,
                self.particle_factors,
                self.budget,
                state.particle_q,
                solver.particle_chebyshev_collided if solver.particle_chebyshev_guarded else None,
            ],
            device=model.device,
        )
        wp.launch(
            kernels.apply_body_truncation_ts,
            model.body_count,
            [
                self.body_reference,
                model.body_flags,
                model.body_com,
                self.body_factors,
                self.body_radius,
                self.body_budget,
                state.body_q,
            ],
            device=model.device,
        )
        # Subsequent self-contact/color updates use their own reference frame.
        wp.launch(
            _difference,
            model.particle_count,
            [state.particle_q, solver.pos_prev_collision_detection, solver.particle_displacements],
            device=model.device,
        )
