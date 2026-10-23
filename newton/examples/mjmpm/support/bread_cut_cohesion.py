# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cuttable elastic particle connections, integrated implicitly before MPM.

The MPM grid supplies bulk rheology and tool contact. This additional spring
network retains cohesion inside the fragments without eroding a volume into
loose grains. Only connections swept by the physical blade edge are removed.
No sample is attached to a prescribed pose or projected after contact.
"""

import numpy as np
import warp as wp
from warp.optim.linear import CG, LinearOperator


@wp.func
def edge_height(x: float):
    z = float(-0.055)
    if x > 0.230:
        z = -0.032 + (x - 0.230) * (0.017 / 0.010)
    elif x > 0.218:
        z = -0.045 + (x - 0.218) * (0.013 / 0.012)
    elif x > 0.205:
        z = -0.052 + (x - 0.205) * (0.007 / 0.013)
    elif x > 0.190:
        z = -0.055 + (x - 0.190) * (0.003 / 0.015)
    return z


@wp.kernel
def _cut_connections(
    positions: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body: int,
    local: wp.transform,
    previous: wp.array[wp.transform],
    pairs: wp.array[wp.vec2i],
    support: float,
    active: wp.array[int],
):
    e = wp.tid()
    if active[e] == 0:
        return
    pose = wp.mul(body_q[body], local)
    inv = wp.transform_inverse(pose)
    motion = wp.transform_vector(inv, wp.transform_get_translation(pose) - wp.transform_get_translation(previous[0]))
    # A downward cutting stroke severs connections. Pushing a completed
    # slice with the flat side or withdrawing the tool must not re-cut it.
    if motion[2] >= -1.0e-6 or -motion[2] < 0.25 * wp.abs(motion[1]):
        return
    pair = pairs[e]
    a = wp.transform_point(inv, positions[pair[0]])
    b = wp.transform_point(inv, positions[pair[1]])
    # A single cross-cut removes a connection, not the material on its ends.
    if a[1] * b[1] < 0.0:
        t = a[1] / (a[1] - b[1])
        hit = wp.lerp(a, b, t)
        if hit[0] >= 0.060 - support and hit[0] <= 0.240 + support:
            old_hit = wp.transform_point(wp.transform_inverse(previous[0]), wp.transform_point(pose, hit))
            if hit[2] >= edge_height(hit[0]) - support and old_hit[2] <= edge_height(old_hit[0]) + support:
                active[e] = 0


@wp.kernel
def _save_tool(body_q: wp.array[wp.transform], body: int, local: wp.transform, previous: wp.array[wp.transform]):
    previous[0] = wp.mul(body_q[body], local)


@wp.kernel
def _initialize(velocities: wp.array[wp.vec3], rhs: wp.array[wp.vec3], diagonal: wp.array[wp.mat33]):
    i = wp.tid()
    rhs[i] = velocities[i]
    diagonal[i] = wp.identity(n=3, dtype=float)


@wp.kernel
def _compressive_plasticity(
    positions: wp.array[wp.vec3],
    pairs: wp.array[wp.vec2i],
    active: wp.array[int],
    zone: wp.array[float],
    initial_lengths: wp.array[float],
    yield_strain: float,
    maximum_strain: float,
    lengths: wp.array[float],
):
    e = wp.tid()
    pair = pairs[e]
    if active[e] == 0 or (zone[pair[0]] <= 0.0 and zone[pair[1]] <= 0.0):
        return
    length = wp.length(positions[pair[1]] - positions[pair[0]])
    rest = lengths[e]
    # A compression return map. Only the yielded portion is permanent;
    # remaining elastic compression still recovers after the knife unloads.
    if length < (1.0 - yield_strain) * rest:
        lengths[e] = wp.max((1.0 - maximum_strain) * initial_lengths[e], length / (1.0 - yield_strain))


@wp.kernel
def _assemble(
    positions: wp.array[wp.vec3],
    pairs: wp.array[wp.vec2i],
    lengths: wp.array[float],
    active: wp.array[int],
    stiffness: float,
    damping: float,
    spacing: float,
    inv_mass: float,
    dt: float,
    directions: wp.array[wp.vec3],
    coefficients: wp.array[float],
    rhs: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    e = wp.tid()
    directions[e] = wp.vec3(0.0)
    coefficients[e] = 0.0
    if active[e] == 0:
        return
    pair = pairs[e]
    delta = positions[pair[1]] - positions[pair[0]]
    length = wp.length(delta)
    if length < 1.0e-9:
        return
    n = delta / length
    k = stiffness * spacing / lengths[e]
    c = damping * wp.sqrt(spacing / lengths[e])
    coefficient = (dt * dt * k + dt * c) * inv_mass
    directions[e] = n
    coefficients[e] = coefficient
    impulse_velocity = dt * k * (length - lengths[e]) * inv_mass * n
    wp.atomic_add(rhs, pair[0], impulse_velocity)
    wp.atomic_sub(rhs, pair[1], impulse_velocity)
    block = coefficient * wp.outer(n, n)
    wp.atomic_add(diagonal, pair[0], block)
    wp.atomic_add(diagonal, pair[1], block)


@wp.kernel
def _invert(diagonal: wp.array[wp.mat33], inverse: wp.array[wp.mat33]):
    i = wp.tid()
    inverse[i] = wp.inverse(diagonal[i])


@wp.kernel
def _identity_product(x: wp.array[wp.vec3], y: wp.array[wp.vec3], z: wp.array[wp.vec3], alpha: float, beta: float):
    i = wp.tid()
    z[i] = alpha * x[i] + beta * y[i]


@wp.kernel
def _connection_product(
    pairs: wp.array[wp.vec2i],
    directions: wp.array[wp.vec3],
    coefficients: wp.array[float],
    x: wp.array[wp.vec3],
    alpha: float,
    z: wp.array[wp.vec3],
):
    e = wp.tid()
    pair = pairs[e]
    n = directions[e]
    value = alpha * coefficients[e] * wp.dot(n, x[pair[0]] - x[pair[1]]) * n
    wp.atomic_add(z, pair[0], value)
    wp.atomic_sub(z, pair[1], value)


@wp.kernel
def _precondition(
    inverse: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    alpha: float,
    beta: float,
):
    i = wp.tid()
    z[i] = alpha * (inverse[i] @ x[i]) + beta * y[i]


def make_connections(rest, spacing):
    """Face, edge and corner neighbours of the material lattice, once each."""
    cells = np.rint((rest - rest.min(axis=0)) / spacing).astype(int)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) > (0, 0, 0)]
    pairs = []
    for i, cell in enumerate(cells):
        for offset in offsets:
            j = lookup.get(tuple(cell + offset))
            if j is not None:
                pairs.append((i, j))
    pairs = np.array(pairs, dtype=np.int32).reshape(-1, 2)
    lengths = np.linalg.norm(rest[pairs[:, 1]] - rest[pairs[:, 0]], axis=1)
    return pairs, lengths.astype(np.float32)


def count_detached_samples(positions, labels, spacing):
    """Samples outside the largest spatially connected patch of each piece."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    detached = 0
    for label in np.unique(labels):
        points = positions[labels == label]
        pairs = cKDTree(points).query_pairs(1.8 * spacing, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
        _, patch = connected_components(graph, directed=False)
        detached += len(points) - int(np.bincount(patch).max())
    return detached


class CuttableCohesion:
    """Implicit spring network with persistent, tool-triggered connection cuts.

    Uniform particle mass is required. Fixed-count CG is captured with the
    parent frame graph. Contact follows the network solve, so the final MPM
    velocity, including cohesion impulses, still satisfies tool contact.
    """

    def __init__(
        self,
        rest,
        *,
        mass,
        spacing,
        device,
        tool_pose,
        stiffness=20.0,
        iterations=24,
        plastic_zone=None,
        plastic_yield=0.03,
        plastic_limit=0.18,
    ):
        self.device = wp.get_device(device)
        self.count, self.spacing, self.mass = len(rest), spacing, mass
        self.stiffness = stiffness
        self.damping = 0.002
        pairs, lengths = make_connections(rest, spacing)
        self.pairs = wp.array(pairs, dtype=wp.vec2i, device=device)
        self.lengths = wp.array(lengths, dtype=float, device=device)
        self.initial_lengths = wp.clone(self.lengths)
        self.plastic_zone = plastic_zone
        self.plastic_yield, self.plastic_limit = plastic_yield, plastic_limit
        self.active = wp.ones(len(pairs), dtype=int, device=device)
        self.previous_tool = wp.array([tool_pose], dtype=wp.transform, device=device)
        self.directions = wp.empty(len(pairs), dtype=wp.vec3, device=device)
        self.coefficients = wp.empty(len(pairs), dtype=float, device=device)
        self.rhs = wp.empty(self.count, dtype=wp.vec3, device=device)
        self.velocity = wp.empty_like(self.rhs)
        self.diagonal = wp.empty(self.count, dtype=wp.mat33, device=device)
        self.inverse = wp.empty_like(self.diagonal)
        matrix = LinearOperator((3 * self.count, 3 * self.count), wp.mat33, self.device, self._matvec)
        preconditioner = LinearOperator(matrix.shape, wp.mat33, self.device, self._precondition)
        self.cg = CG(
            matrix,
            self.rhs,
            self.velocity,
            M=preconditioner,
            maxiter=iterations,
            tol=1.0e-6,
            atol=1.0e-8,
            check_every=0,
            use_cuda_graph=False,
        )

    def _matvec(self, x, y, z, alpha, beta):
        wp.launch(_identity_product, self.count, [x, y, z, float(alpha), float(beta)], device=self.device)
        wp.launch(
            _connection_product,
            len(self.pairs),
            [self.pairs, self.directions, self.coefficients, x, float(alpha), z],
            device=self.device,
        )

    def _precondition(self, x, y, z, alpha, beta):
        wp.launch(_precondition, self.count, [self.inverse, x, y, z, float(alpha), float(beta)], device=self.device)

    def cut(self, state, body, local):
        wp.launch(
            _cut_connections,
            len(self.pairs),
            [
                state.particle_q,
                state.body_q,
                body,
                local,
                self.previous_tool,
                self.pairs,
                self.spacing * 0.55,
                self.active,
            ],
            device=self.device,
        )
        wp.launch(_save_tool, 1, [state.body_q, body, local, self.previous_tool], device=self.device)

    def apply(self, state, dt):
        if self.plastic_zone is not None:
            wp.launch(
                _compressive_plasticity,
                len(self.pairs),
                [
                    state.particle_q,
                    self.pairs,
                    self.active,
                    self.plastic_zone,
                    self.initial_lengths,
                    self.plastic_yield,
                    self.plastic_limit,
                    self.lengths,
                ],
                device=self.device,
            )
        wp.launch(_initialize, self.count, [state.particle_qd, self.rhs, self.diagonal], device=self.device)
        wp.launch(
            _assemble,
            len(self.pairs),
            [
                state.particle_q,
                self.pairs,
                self.lengths,
                self.active,
                self.stiffness,
                self.damping,
                self.spacing,
                1.0 / self.mass,
                dt,
                self.directions,
                self.coefficients,
                self.rhs,
                self.diagonal,
            ],
            device=self.device,
        )
        wp.launch(_invert, self.count, [self.diagonal, self.inverse], device=self.device)
        wp.copy(self.velocity, state.particle_qd)
        self.cg()
        wp.copy(state.particle_qd, self.velocity)

    def components(self):
        """Return component labels and sizes for validation outside capture."""
        from scipy.sparse import coo_matrix  # noqa: PLC0415
        from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

        pairs = self.pairs.numpy()[self.active.numpy() != 0]
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(self.count, self.count))
        _, labels = connected_components(graph, directed=False)
        return labels, np.bincount(labels)
