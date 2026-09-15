# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental supported free-body island sleeping.

The wake-counter criterion follows the structure of PhysX DySleep.cpp,
without its stabilization damping or pose projection. Velocity accumulation
is time-normalized to a 60 Hz reference. This helper is not enabled by default.
"""

import warp as wp

from newton import JointType

from .vbd.rigid_vbd_kernels import _reset_world_selected


@wp.func
def _root(parent: wp.array[int], body: int):
    r = body
    while parent[r] != r:
        r = parent[r]
    return r


@wp.kernel
def _initialize(
    parent: wp.array[int],
    veto: wp.array[int],
    supported: wp.array[int],
    support: wp.array[int],
    adjacent: wp.array2d[int],
):
    b, other = wp.tid()
    adjacent[b, other] = 0
    if other == 0:
        parent[b] = b
        veto[b] = 0
        supported[b] = 0
        support[b] = 0


@wp.kernel
def _graph(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    shape_body: wp.array[int],
    p0: wp.array[wp.vec3],
    p1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    base: wp.array[float],
    parent: wp.array[int],
    adjacent: wp.array2d[int],
    support: wp.array[int],
    gravity: wp.array[wp.vec3],
    body_world: wp.array[int],
):
    row = wp.tid()
    if row == 0 and (count[0] < 0 or count[0] > shape0.shape[0]):
        # Incomplete contact topology cannot justify putting any island to sleep.
        wp.atomic_or(support, 0, 4)
    if row >= wp.min(count[0], shape0.shape[0]):
        return
    if shape0[row] < 0 or shape1[row] < 0:
        return
    a = shape_body[shape0[row]]
    b = shape_body[shape1[row]]
    if a == b:
        return
    xa = p0[row]
    xb = p1[row]
    if a >= 0:
        xa = wp.transform_point(q[a], xa)
    if b >= 0:
        xb = wp.transform_point(q[b], xb)
    if wp.dot(normal[row], xb - xa) - margin0[row] - margin1[row] > 0.0001:
        return
    da = False
    db = False
    if a >= 0:
        da = base[a] > 0.0
    if b >= 0:
        db = base[b] > 0.0
    if da and db:
        # The diagonal stores the unique-neighbor count. Count each pair once,
        # regardless of how many manifold points it contributes.
        if wp.atomic_max(adjacent, a, b, 1) == 0:
            wp.atomic_add(adjacent, a, a, 1)
        if wp.atomic_max(adjacent, b, a, 1) == 0:
            wp.atomic_add(adjacent, b, b, 1)
        ra = _root(parent, a)
        rb = _root(parent, b)
        while ra != rb:
            hi = wp.max(ra, rb)
            lo = wp.min(ra, rb)
            old = wp.atomic_cas(parent, hi, hi, lo)
            if old == hi:
                break
            ra = _root(parent, ra)
            rb = _root(parent, rb)
    elif da:
        up = wp.normalize(-gravity[wp.max(body_world[a], 0)])
        if wp.dot(normal[row], up) < -0.3:
            wp.atomic_or(support, a, 1)
        if b >= 0 and wp.dot(qd[b], qd[b]) > 1.0e-16:
            wp.atomic_or(support, a, 2)
    elif db:
        up = wp.normalize(-gravity[wp.max(body_world[b], 0)])
        if wp.dot(normal[row], up) > 0.3:
            wp.atomic_or(support, b, 1)
        if a >= 0 and wp.dot(qd[a], qd[a]) > 1.0e-16:
            wp.atomic_or(support, b, 2)


@wp.kernel
def _soft_wake(count: wp.array[int], shapes: wp.array[int], shape_body: wp.array[int], support: wp.array[int]):
    i = wp.tid()
    if i == 0 and (count[0] < 0 or count[0] > shapes.shape[0]):
        wp.atomic_or(support, 0, 4)
    if i < wp.min(count[0], shapes.shape[0]):
        s = shapes[i]
        if s >= 0:
            b = shape_body[s]
            if b >= 0:
                wp.atomic_or(support, b, 2)


@wp.kernel
def _evaluate(
    parent: wp.array[int],
    adjacent: wp.array2d[int],
    support: wp.array[int],
    supported: wp.array[int],
    veto: wp.array[int],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    forces: wp.array[wp.spatial_vector],
    inertia: wp.array[wp.mat33],
    base: wp.array[float],
    asleep: wp.array[int],
    timer: wp.array[float],
    accumulator: wp.array[wp.spatial_vector],
    dt: float,
    before: int,
):
    b = wp.tid()
    if base[b] <= 0.0:
        return
    r = _root(parent, b)
    finite = True
    for axis in range(6):
        finite = finite and wp.isfinite(qd[b][axis]) and wp.isfinite(forces[b][axis])
    for axis in range(7):
        finite = finite and wp.isfinite(q[b][axis])
    if not finite:
        wp.atomic_max(veto, r, 1)
        timer[b] = 0.4
        return
    if support[b] & 1:
        wp.atomic_max(supported, r, 1)
    if support[b] & 2 or support[0] & 4 or wp.dot(forces[b], forces[b]) > 1.0e-16:
        wp.atomic_max(veto, r, 1)
    if before != 0:
        if asleep[b] == 0 or wp.length(wp.spatial_top(qd[b])) > 1.0e-8 or wp.length(wp.spatial_bottom(qd[b])) > 1.0e-8:
            wp.atomic_max(veto, r, 1)
    else:
        v = wp.spatial_top(qd[b])
        w = wp.quat_rotate_inv(wp.transform_get_rotation(q[b]), wp.spatial_bottom(qd[b]))
        degree = adjacent[b, b]
        # PhysX non-stabilization wake counter: accumulate velocity vectors
        # during the final half of the countdown (not a mean-energy filter).
        wc = timer[b]
        if wc < 0.2 or wc < dt:
            # Integrate velocity over physical time, expressed in 60 Hz
            # reference samples rather than counting solver substeps.
            acc = accumulator[b] + wp.spatial_vector(v, w) * (dt * 60.0)
            accumulator[b] = acc
            lv = wp.spatial_top(acc)
            av = wp.spatial_bottom(acc)
            energy = 0.5 * (wp.dot(lv, lv) + wp.dot(av, inertia[b] * av) * base[b])
            # PxRigidDynamic default: 5e-5 * tolerance speed squared;
            # PxTolerancesScale's metric default speed is 10 m/s.
            threshold = 0.005 * float(1 + degree)
            if not wp.isfinite(energy) or energy >= threshold:
                accumulator[b] = wp.spatial_vector()
                wc = 0.4 + dt * float(degree)
                if wp.isfinite(energy):
                    wc = wp.min(energy / threshold, 2.0) * 0.2 + dt * float(degree)
        timer[b] = wp.max(0.0, wc - dt)
        # PhysX sleepCheck resets the filter whenever the body's counter
        # reaches zero, even while an active neighbor keeps its island awake.
        if timer[b] == 0.0:
            accumulator[b] = wp.spatial_vector()
        if timer[b] > 0.0:
            wp.atomic_max(veto, r, 1)


@wp.kernel
def _apply(
    parent: wp.array[int],
    supported: wp.array[int],
    veto: wp.array[int],
    base: wp.array[float],
    asleep: wp.array[int],
    effective: wp.array[float],
    qd: wp.array[wp.spatial_vector],
    other_qd: wp.array[wp.spatial_vector],
    timer: wp.array[float],
):
    b = wp.tid()
    if base[b] <= 0.0:
        return
    r = _root(parent, b)
    sleeping = int(veto[r] == 0 and supported[r] != 0)
    if asleep[b] != 0 and sleeping == 0:
        timer[b] = 0.4
    asleep[b] = sleeping
    effective[b] = 0.0 if sleeping != 0 else base[b]
    if sleeping != 0:
        qd[b] = wp.spatial_vector()
        other_qd[b] = wp.spatial_vector()


class RigidBodySleep:
    """Own free-body sleep state independently of physical model masses.

    Experimental: currently uses bounded dense adjacency.
    Non-free joints are always active. Shape/pose edits must call
    reset before stepping; moving supports must supply consistent velocities.
    """

    def __init__(self, solver):
        self.solver = solver
        self.model = model = solver.model
        self.device = solver.device
        n = model.body_count
        if 4 * n * n > 32 * 1024 * 1024:
            raise ValueError("Rigid sleeping adjacency exceeds the 32 MiB budget")
        self.base = wp.array(self._eligible_inverse_mass(), dtype=float, device=self.device)
        self.parent = wp.empty(n, dtype=int, device=self.device)
        self.veto = wp.zeros(n, dtype=int, device=self.device)
        self.supported = wp.zeros(n, dtype=int, device=self.device)
        self.support = wp.zeros(n, dtype=int, device=self.device)
        self.adjacent = wp.zeros((n, n), dtype=int, device=self.device)
        self.asleep = wp.zeros(n, dtype=int, device=self.device)
        self.timer = wp.full(n, 0.4, dtype=float, device=self.device)
        self.accumulator = wp.zeros(n, dtype=wp.spatial_vector, device=self.device)

    def _eligible_inverse_mass(self):
        model = self.model
        eligible = self.solver.body_inv_mass_effective.numpy().copy()
        for kind, a, b in zip(
            model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy(), strict=True
        ):
            if kind != int(JointType.FREE):
                if a >= 0:
                    eligible[a] = 0.0
                if b >= 0:
                    eligible[b] = 0.0
        return eligible

    def refresh_eligibility(self):
        """Refresh mass ownership after the solver has refreshed body properties."""
        self.base.assign(self._eligible_inverse_mass())

    def update(self, state, other, contacts, dt, *, before):
        """Update support, wake propagation and the island readiness criterion."""
        if contacts is None:
            self.reset()
            return
        n = self.model.body_count
        wp.launch(
            _initialize,
            (n, n),
            [self.parent, self.veto, self.supported, self.support, self.adjacent],
            device=self.device,
        )
        wp.launch(
            _graph,
            contacts.rigid_contact_shape0.size,
            [
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                self.model.shape_body,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                state.body_q,
                state.body_qd,
                self.base,
                self.parent,
                self.adjacent,
                self.support,
                self.model.gravity,
                self.model.body_world,
            ],
            device=self.device,
        )
        if contacts.soft_contact_shape.size:
            wp.launch(
                _soft_wake,
                contacts.soft_contact_shape.size,
                [contacts.soft_contact_count, contacts.soft_contact_shape, self.model.shape_body, self.support],
                device=self.device,
            )
        wp.launch(
            _evaluate,
            n,
            [
                self.parent,
                self.adjacent,
                self.support,
                self.supported,
                self.veto,
                state.body_q,
                state.body_qd,
                state.body_f,
                self.model.body_inertia,
                self.base,
                self.asleep,
                self.timer,
                self.accumulator,
                dt,
                int(before),
            ],
            device=self.device,
        )
        wp.launch(
            _apply,
            n,
            [
                self.parent,
                self.supported,
                self.veto,
                self.base,
                self.asleep,
                self.solver.body_inv_mass_effective,
                state.body_qd,
                other.body_qd,
                self.timer,
            ],
            device=self.device,
        )

    def reset(self, world_mask=None):
        """Wake selected bodies without altering pose, velocity or model mass."""
        wp.launch(
            _reset,
            self.model.body_count,
            [
                self.base,
                self.solver.body_inv_mass_effective,
                self.asleep,
                self.timer,
                self.accumulator,
                self.model.body_world,
                world_mask,
                world_mask is None,
                self.model.world_count,
            ],
            device=self.device,
        )


@wp.kernel
def _reset(
    base: wp.array[float],
    effective: wp.array[float],
    asleep: wp.array[int],
    timer: wp.array[float],
    accumulator: wp.array[wp.spatial_vector],
    body_world: wp.array[int],
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
):
    body = wp.tid()
    if not _reset_world_selected(body_world[body], world_mask, reset_all, world_count):
        return
    if base[body] > 0.0:
        effective[body] = base[body]
    asleep[body] = 0
    timer[body] = 0.4
    accumulator[body] = wp.spatial_vector()
