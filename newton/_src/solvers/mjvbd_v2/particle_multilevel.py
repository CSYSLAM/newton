# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-level particle correction helpers private to MJVBDV2."""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
import warp as wp

from ...geometry import ParticleFlags

__all__ = ["ParticleMultilevelCorrection"]

_COARSE_PCG_BLOCK_DIM = 256
_AUTO_MIN_ACTIVE_PARTICLES = 1024
_AUTO_MIN_CLUSTER_COUNT = 128
_AUTO_MAX_CLUSTER_COUNT = 1500

_RUNTIME_STATUS_NONFINITE_RHS = 1
_RUNTIME_STATUS_NONFINITE_SOLUTION = 2
_RUNTIME_STATUS_RESIDUAL_NOT_REDUCED = 4
_RUNTIME_STATUS_EXCESSIVE_RADIUS_CLAMP = 8

ParticleMultilevelMode = Literal["off", "on", "auto"]


def _normalize_multilevel_mode(enabled: bool | Literal["auto"]) -> ParticleMultilevelMode:
    """Normalize the backward-compatible multilevel option."""
    if enabled is False:
        return "off"
    if enabled is True:
        return "on"
    if enabled == "auto":
        return "auto"
    raise ValueError(f"particle_enable_multilevel_correction must be False, True, or 'auto', got {enabled!r}")


def _automatic_rejection_reason(model, correction: ParticleMultilevelCorrection) -> str | None:
    """Return why the conservative automatic policy rejected a hierarchy."""
    if model.tet_count > 0:
        return "tetrahedra_present"
    if model.world_count > 1:
        return "multiple_worlds"
    if correction.active_particle_count < _AUTO_MIN_ACTIVE_PARTICLES:
        return "too_few_active_particles"
    if correction.cluster_count < _AUTO_MIN_CLUSTER_COUNT:
        return "too_few_clusters"
    if correction.cluster_count > _AUTO_MAX_CLUSTER_COUNT:
        return "too_many_clusters"
    return None


@wp.kernel(enable_backward=False)
def _restrict_particle_corrections(
    cluster_particle_offsets: wp.array[wp.int32],
    cluster_particles: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    local_correction: wp.array[wp.vec3],
    local_hessian: wp.array[wp.mat33],
    contact_hessian: wp.array[wp.mat33],
    dt: float,
    coarse_rhs: wp.array[wp.vec3],
    coarse_mass: wp.array[float],
    coarse_stiffness: wp.array[float],
    coarse_contact_stiffness: wp.array[float],
):
    cluster = wp.tid()
    force_sum = wp.vec3(0.0)
    mass_sum = float(0.0)
    stiffness_sum = float(0.0)
    contact_stiffness_sum = float(0.0)
    inv_dt_sq = 1.0 / (dt * dt)
    for slot in range(cluster_particle_offsets[cluster], cluster_particle_offsets[cluster + 1]):
        particle = cluster_particles[slot]
        mass = particle_mass[particle]
        if mass > 0.0 and (particle_flags[particle] & ParticleFlags.ACTIVE) != 0:
            hessian = local_hessian[particle]
            contact = contact_hessian[particle]
            force_sum += hessian * local_correction[particle]
            mass_sum += mass
            mean_diagonal = (hessian[0, 0] + hessian[1, 1] + hessian[2, 2]) / 3.0
            contact_mean_diagonal = (contact[0, 0] + contact[1, 1] + contact[2, 2]) / 3.0
            stiffness_sum += wp.max(mean_diagonal - mass * inv_dt_sq - contact_mean_diagonal, 0.0)
            contact_stiffness_sum += wp.max(contact_mean_diagonal, 0.0)
    coarse_rhs[cluster] = force_sum
    coarse_mass[cluster] = mass_sum
    coarse_stiffness[cluster] = stiffness_sum
    coarse_contact_stiffness[cluster] = contact_stiffness_sum


@wp.func
def _coarse_edge_weight(
    cluster: wp.int32,
    neighbor: wp.int32,
    edge_multiplicity: wp.int32,
    coarse_incident_edges: wp.array[wp.int32],
    coarse_stiffness: wp.array[float],
    coupling: float,
):
    cluster_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
    neighbor_edges = wp.max(float(coarse_incident_edges[neighbor]), 1.0)
    cluster_weight = coarse_stiffness[cluster] / cluster_edges
    neighbor_weight = coarse_stiffness[neighbor] / neighbor_edges
    return 0.5 * coupling * float(edge_multiplicity) * (cluster_weight + neighbor_weight)


@wp.kernel(enable_backward=False)
def _solve_coarse_pcg_persistent(
    coarse_count: int,
    coarse_neighbor_offsets: wp.array[wp.int32],
    coarse_neighbors: wp.array[wp.int32],
    coarse_neighbor_multiplicity: wp.array[wp.int32],
    coarse_incident_edges: wp.array[wp.int32],
    coarse_anchor_edges: wp.array[wp.int32],
    rhs: wp.array[wp.vec3],
    coarse_mass: wp.array[float],
    coarse_stiffness: wp.array[float],
    coarse_contact_stiffness: wp.array[float],
    dt: float,
    coupling: float,
    iterations: int,
    validate_residual: bool,
    minimum_residual_reduction: float,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    preconditioned_residual: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    diagonal: wp.array[float],
    runtime_status: wp.array[wp.int32],
    runtime_metrics: wp.array[float],
    runtime_counters: wp.array[wp.int32],
):
    """Solve the coarse system in one block to avoid per-PCG-step launches."""
    lane = wp.tid()
    stride = wp.block_dim()
    inv_dt_sq = 1.0 / (dt * dt)
    rz_local = float(0.0)
    initial_residual_norm_sq_local = float(0.0)
    nonfinite_rhs_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        begin = coarse_neighbor_offsets[cluster]
        end = coarse_neighbor_offsets[cluster + 1]
        diagonal_value = coarse_mass[cluster] * inv_dt_sq + coarse_contact_stiffness[cluster]
        incident_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
        diagonal_value += coupling * float(coarse_anchor_edges[cluster]) * coarse_stiffness[cluster] / incident_edges
        for slot in range(begin, end):
            diagonal_value += _coarse_edge_weight(
                cluster,
                coarse_neighbors[slot],
                coarse_neighbor_multiplicity[slot],
                coarse_incident_edges,
                coarse_stiffness,
                coupling,
            )
        diagonal_value = wp.max(diagonal_value, 1.0e-12)
        r = rhs[cluster]
        z = r / diagonal_value
        solution[cluster] = wp.vec3(0.0)
        residual[cluster] = r
        preconditioned_residual[cluster] = z
        direction[cluster] = z
        diagonal[cluster] = diagonal_value
        rz_local += wp.dot(r, z)
        initial_residual_norm_sq_local += wp.dot(r, r)
        if not (wp.isfinite(r[0]) and wp.isfinite(r[1]) and wp.isfinite(r[2])):
            nonfinite_rhs_local += 1.0
        cluster += stride

    rz = wp.tile_sum(wp.tile(rz_local))[0]
    initial_residual_norm_sq = wp.tile_sum(wp.tile(initial_residual_norm_sq_local))[0]
    nonfinite_rhs_count = wp.tile_sum(wp.tile(nonfinite_rhs_local))[0]
    if lane == 0:
        runtime_metrics[4] = rz
    for _iteration in range(iterations):
        direction_product_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            value = (coarse_mass[cluster] * inv_dt_sq + coarse_contact_stiffness[cluster]) * direction[cluster]
            incident_edges = wp.max(float(coarse_incident_edges[cluster]), 1.0)
            anchor_weight = coupling * float(coarse_anchor_edges[cluster]) * coarse_stiffness[cluster] / incident_edges
            value += anchor_weight * direction[cluster]
            begin = coarse_neighbor_offsets[cluster]
            end = coarse_neighbor_offsets[cluster + 1]
            for slot in range(begin, end):
                neighbor = coarse_neighbors[slot]
                weight = _coarse_edge_weight(
                    cluster,
                    neighbor,
                    coarse_neighbor_multiplicity[slot],
                    coarse_incident_edges,
                    coarse_stiffness,
                    coupling,
                )
                value += weight * (direction[cluster] - direction[neighbor])
            product[cluster] = value
            direction_product_local += wp.dot(direction[cluster], value)
            cluster += stride

        direction_product = wp.tile_sum(wp.tile(direction_product_local))[0]
        alpha = float(0.0)
        if direction_product > 1.0e-20:
            alpha = rz / direction_product

        new_rz_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            solution[cluster] += alpha * direction[cluster]
            r = residual[cluster] - alpha * product[cluster]
            z = r / diagonal[cluster]
            residual[cluster] = r
            preconditioned_residual[cluster] = z
            new_rz_local += wp.dot(r, z)
            cluster += stride

        new_rz = wp.tile_sum(wp.tile(new_rz_local))[0]
        if lane == 0:
            runtime_metrics[5 + _iteration] = new_rz
        beta = float(0.0)
        if rz > 1.0e-20:
            beta = new_rz / rz
        rz = new_rz

        direction_norm_local = float(0.0)
        cluster = lane
        while cluster < coarse_count:
            direction[cluster] = preconditioned_residual[cluster] + beta * direction[cluster]
            direction_norm_local += wp.dot(direction[cluster], direction[cluster])
            cluster += stride
        # This reduction is also the block-wide barrier before neighbors read
        # the direction values written above.
        direction_norm = wp.tile_sum(wp.tile(direction_norm_local))[0]
        if direction_norm <= 1.0e-30:
            rz = 0.0

    final_residual_norm_sq_local = float(0.0)
    nonfinite_solution_local = float(0.0)
    cluster = lane
    while cluster < coarse_count:
        final_residual = residual[cluster]
        coarse_solution = solution[cluster]
        final_residual_norm_sq_local += wp.dot(final_residual, final_residual)
        if not (
            wp.isfinite(final_residual[0])
            and wp.isfinite(final_residual[1])
            and wp.isfinite(final_residual[2])
            and wp.isfinite(coarse_solution[0])
            and wp.isfinite(coarse_solution[1])
            and wp.isfinite(coarse_solution[2])
        ):
            nonfinite_solution_local += 1.0
        cluster += stride

    final_residual_norm_sq = wp.tile_sum(wp.tile(final_residual_norm_sq_local))[0]
    nonfinite_solution_count = wp.tile_sum(wp.tile(nonfinite_solution_local))[0]
    if lane == 0:
        status = wp.int32(0)
        if nonfinite_rhs_count > 0.0 or not wp.isfinite(initial_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_RHS)
        if nonfinite_solution_count > 0.0 or not wp.isfinite(final_residual_norm_sq):
            status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_SOLUTION)
        if validate_residual and initial_residual_norm_sq > 1.0e-30 and wp.isfinite(final_residual_norm_sq):
            residual_reduction = 1.0 - final_residual_norm_sq / initial_residual_norm_sq
            if residual_reduction < minimum_residual_reduction:
                status = status | wp.int32(_RUNTIME_STATUS_RESIDUAL_NOT_REDUCED)
        runtime_status[0] = status
        runtime_metrics[0] = initial_residual_norm_sq
        runtime_metrics[1] = final_residual_norm_sq
        runtime_metrics[2] = 0.0
        runtime_metrics[3] = 0.0
        runtime_counters[0] = 0
        runtime_counters[1] = 0


@wp.kernel(enable_backward=False)
def _prepare_prolonged_corrections(
    active_particles: wp.array[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    particle_radius: wp.array[float],
    coarse_correction: wp.array[wp.vec3],
    relaxation: float,
    max_radius_fraction: float,
    runtime_counters: wp.array[wp.int32],
    fine_correction: wp.array[wp.vec3],
):
    particle = active_particles[wp.tid()]
    cluster = fine_to_coarse[particle]
    correction = relaxation * coarse_correction[cluster]
    if not (wp.isfinite(correction[0]) and wp.isfinite(correction[1]) and wp.isfinite(correction[2])):
        wp.atomic_add(runtime_counters, 1, 1)
        fine_correction[particle] = wp.vec3(0.0)
        return
    correction_length = wp.length(correction)
    max_length = max_radius_fraction * particle_radius[particle]
    if correction_length > max_length:
        wp.atomic_add(runtime_counters, 0, 1)
        if max_length > 0.0:
            correction *= max_length / correction_length
        else:
            correction = wp.vec3(0.0)
    fine_correction[particle] = correction


@wp.kernel(enable_backward=False)
def _commit_prolonged_corrections(
    active_particles: wp.array[wp.int32],
    active_particle_count: int,
    fine_correction: wp.array[wp.vec3],
    max_clamp_fraction: float,
    runtime_counters: wp.array[wp.int32],
    runtime_status: wp.array[wp.int32],
    runtime_metrics: wp.array[float],
    particle_displacements: wp.array[wp.vec3],
):
    active_index = wp.tid()
    status = runtime_status[0]
    clamp_fraction = float(runtime_counters[0]) / float(active_particle_count)
    if runtime_counters[1] > 0:
        status = status | wp.int32(_RUNTIME_STATUS_NONFINITE_SOLUTION)
    if clamp_fraction > max_clamp_fraction:
        status = status | wp.int32(_RUNTIME_STATUS_EXCESSIVE_RADIUS_CLAMP)

    if active_index == 0:
        runtime_status[0] = status
        runtime_metrics[2] = clamp_fraction
        runtime_metrics[3] = float(status == 0)

    if status == 0:
        particle = active_particles[active_index]
        particle_displacements[particle] += fine_correction[particle]


def _particle_topology_edges(model) -> np.ndarray:
    """Return unique undirected particle edges without crossing worlds."""
    edge_chunks: list[np.ndarray] = []
    if model.tri_count:
        triangles = np.asarray(model.tri_indices.numpy(), dtype=np.int32).reshape((-1, 3))
        edge_chunks.extend(
            (
                triangles[:, (0, 1)],
                triangles[:, (1, 2)],
                triangles[:, (2, 0)],
            )
        )
    if model.tet_count:
        tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
        edge_chunks.extend(tetrahedra[:, pair] for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    if model.spring_count:
        edge_chunks.append(np.asarray(model.spring_indices.numpy(), dtype=np.int32).reshape((-1, 2)))
    if not edge_chunks:
        return np.empty((0, 2), dtype=np.int32)

    edges = np.concatenate(edge_chunks, axis=0)
    edges.sort(axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if model.particle_world is not None:
        particle_world = np.asarray(model.particle_world.numpy(), dtype=np.int32)
        edges = edges[particle_world[edges[:, 0]] == particle_world[edges[:, 1]]]
    return np.unique(edges, axis=0)


def _build_clusters(
    model, target_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    particle_count = model.particle_count
    edges = _particle_topology_edges(model)
    neighbors: list[list[int]] = [[] for _ in range(particle_count)]
    for particle_a, particle_b in edges:
        neighbors[int(particle_a)].append(int(particle_b))
        neighbors[int(particle_b)].append(int(particle_a))

    masses = np.asarray(model.particle_mass.numpy(), dtype=np.float32)
    flags = np.asarray(model.particle_flags.numpy(), dtype=np.int32)
    movable = (masses > 0.0) & ((flags & int(ParticleFlags.ACTIVE)) != 0)
    movable &= np.fromiter((bool(particle_neighbors) for particle_neighbors in neighbors), dtype=bool)
    # Piecewise-constant cluster translations improve long-wave propagation
    # for cloth and shells, but cannot represent the affine modes required by
    # tetrahedral solids. Leave tet vertices on the original VBD path.
    if model.tet_count:
        tetrahedra = np.asarray(model.tet_indices.numpy(), dtype=np.int32).reshape((-1, 4))
        movable[np.unique(tetrahedra)] = False
    fine_to_coarse = np.full(particle_count, -1, dtype=np.int32)
    clusters: list[list[int]] = []

    for seed in np.flatnonzero(movable):
        if fine_to_coarse[seed] >= 0:
            continue
        cluster_index = len(clusters)
        cluster: list[int] = []
        frontier = deque([int(seed)])
        fine_to_coarse[seed] = cluster_index
        while frontier and len(cluster) < target_size:
            particle = frontier.popleft()
            cluster.append(particle)
            for neighbor in neighbors[particle]:
                if movable[neighbor] and fine_to_coarse[neighbor] < 0:
                    fine_to_coarse[neighbor] = cluster_index
                    frontier.append(neighbor)
        # Return overflowed frontier vertices to the unassigned pool. They
        # become seeds of adjacent clusters in the outer loop.
        while frontier:
            fine_to_coarse[frontier.popleft()] = -1
        clusters.append(cluster)

    cluster_offsets = np.zeros(len(clusters) + 1, dtype=np.int32)
    if clusters:
        cluster_offsets[1:] = np.cumsum([len(cluster) for cluster in clusters], dtype=np.int32)
    cluster_particles = (
        np.concatenate([np.asarray(cluster, dtype=np.int32) for cluster in clusters])
        if clusters
        else np.empty(0, dtype=np.int32)
    )

    coarse_edges: dict[tuple[int, int], int] = {}
    coarse_incident_edges = np.zeros(len(clusters), dtype=np.int32)
    coarse_anchor_edges = np.zeros(len(clusters), dtype=np.int32)
    for particle_a, particle_b in edges:
        cluster_a = int(fine_to_coarse[particle_a])
        cluster_b = int(fine_to_coarse[particle_b])
        if cluster_a >= 0:
            coarse_incident_edges[cluster_a] += 1
        if cluster_b >= 0:
            coarse_incident_edges[cluster_b] += 1
        if cluster_a < 0 <= cluster_b:
            coarse_anchor_edges[cluster_b] += 1
            continue
        if cluster_b < 0 <= cluster_a:
            coarse_anchor_edges[cluster_a] += 1
            continue
        if cluster_a < 0 or cluster_b < 0 or cluster_a == cluster_b:
            continue
        coarse_edge = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
        coarse_edges[coarse_edge] = coarse_edges.get(coarse_edge, 0) + 1
    coarse_neighbors: list[list[tuple[int, int]]] = [[] for _ in clusters]
    for (cluster_a, cluster_b), multiplicity in sorted(coarse_edges.items()):
        coarse_neighbors[cluster_a].append((cluster_b, multiplicity))
        coarse_neighbors[cluster_b].append((cluster_a, multiplicity))
    coarse_neighbor_offsets = np.zeros(len(clusters) + 1, dtype=np.int32)
    if clusters:
        coarse_neighbor_offsets[1:] = np.cumsum(
            [len(cluster_neighbors) for cluster_neighbors in coarse_neighbors],
            dtype=np.int32,
        )
    coarse_neighbor_indices = (
        np.concatenate(
            [
                np.asarray([neighbor for neighbor, _multiplicity in entries], dtype=np.int32)
                for entries in coarse_neighbors
            ]
        )
        if any(coarse_neighbors)
        else np.empty(0, dtype=np.int32)
    )
    coarse_neighbor_multiplicity = (
        np.concatenate(
            [
                np.asarray([multiplicity for _neighbor, multiplicity in entries], dtype=np.int32)
                for entries in coarse_neighbors
            ]
        )
        if any(coarse_neighbors)
        else np.empty(0, dtype=np.int32)
    )
    return (
        fine_to_coarse,
        cluster_offsets,
        cluster_particles,
        coarse_neighbor_offsets,
        coarse_neighbor_indices,
        coarse_neighbor_multiplicity,
        coarse_incident_edges,
        coarse_anchor_edges,
    )


class ParticleMultilevelCorrection:
    """Fixed-topology two-level propagation correction for particle VBD."""

    def __init__(
        self,
        model,
        *,
        cluster_size: int,
        coarse_iterations: int,
        coupling: float,
        relaxation: float,
        max_radius_fraction: float,
        minimum_residual_reduction: float | None,
        max_clamp_fraction: float,
    ):
        if cluster_size < 2:
            raise ValueError(f"particle multilevel cluster_size must be at least 2, got {cluster_size}")
        if coarse_iterations < 1:
            raise ValueError(f"particle multilevel coarse_iterations must be at least 1, got {coarse_iterations}")
        if coupling < 0.0:
            raise ValueError(f"particle multilevel coupling must be nonnegative, got {coupling}")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(f"particle multilevel relaxation must be in (0, 1], got {relaxation}")
        if max_radius_fraction <= 0.0:
            raise ValueError(f"particle multilevel max_radius_fraction must be positive, got {max_radius_fraction}")
        if minimum_residual_reduction is not None and not 0.0 <= minimum_residual_reduction < 1.0:
            raise ValueError(
                f"particle multilevel minimum_residual_reduction must be in [0, 1), got {minimum_residual_reduction}"
            )
        if not 0.0 <= max_clamp_fraction <= 1.0:
            raise ValueError(f"particle multilevel max_clamp_fraction must be in [0, 1], got {max_clamp_fraction}")

        (
            fine_to_coarse,
            cluster_offsets,
            cluster_particles,
            coarse_neighbor_offsets,
            coarse_neighbors,
            coarse_neighbor_multiplicity,
            coarse_incident_edges,
            coarse_anchor_edges,
        ) = _build_clusters(model, cluster_size)
        self.cluster_count = int(cluster_offsets.size - 1)
        self.active_particle_count = int(cluster_particles.size)
        self.coarse_iterations = coarse_iterations
        self.coupling = coupling
        self.relaxation = relaxation
        self.max_radius_fraction = max_radius_fraction
        self.validate_residual = minimum_residual_reduction is not None
        self.minimum_residual_reduction = 0.0 if minimum_residual_reduction is None else minimum_residual_reduction
        self.max_clamp_fraction = max_clamp_fraction
        self.fine_to_coarse = wp.array(fine_to_coarse, dtype=wp.int32, device=model.device)
        self.active_particles = wp.array(cluster_particles, dtype=wp.int32, device=model.device)
        self.cluster_particle_offsets = wp.array(cluster_offsets, dtype=wp.int32, device=model.device)
        self.cluster_particles = wp.array(cluster_particles, dtype=wp.int32, device=model.device)
        self.coarse_neighbor_offsets = wp.array(coarse_neighbor_offsets, dtype=wp.int32, device=model.device)
        self.coarse_neighbors = wp.array(coarse_neighbors, dtype=wp.int32, device=model.device)
        self.coarse_neighbor_multiplicity = wp.array(
            coarse_neighbor_multiplicity,
            dtype=wp.int32,
            device=model.device,
        )
        self.coarse_incident_edges = wp.array(coarse_incident_edges, dtype=wp.int32, device=model.device)
        self.coarse_anchor_edges = wp.array(coarse_anchor_edges, dtype=wp.int32, device=model.device)
        self.local_correction = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.local_hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.contact_forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.contact_hessians = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
        self.coarse_rhs = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_mass = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_stiffness = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_contact_stiffness = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.coarse_solution = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_residual = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_preconditioned_residual = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_direction = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_product = wp.zeros(self.cluster_count, dtype=wp.vec3, device=model.device)
        self.coarse_diagonal = wp.zeros(self.cluster_count, dtype=float, device=model.device)
        self.fine_correction = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.runtime_status = wp.zeros(1, dtype=wp.int32, device=model.device)
        # Initial/final Euclidean residual squared, radius-clamp fraction,
        # acceptance, then the initial and per-iteration preconditioned residuals.
        self.runtime_metrics = wp.zeros(5 + coarse_iterations, dtype=float, device=model.device)
        # Radius-clamp count and non-finite fine-correction count.
        self.runtime_counters = wp.zeros(2, dtype=wp.int32, device=model.device)

    @property
    def enabled(self) -> bool:
        return self.cluster_count > 1 and self.active_particle_count > 0

    def restrict_and_prolong(self, model, particle_displacements: wp.array[wp.vec3], dt: float) -> None:
        """Restrict the local system, solve its coarse approximation, and prolong."""
        if not self.enabled:
            return
        wp.launch(
            _restrict_particle_corrections,
            dim=self.cluster_count,
            inputs=[
                self.cluster_particle_offsets,
                self.cluster_particles,
                model.particle_mass,
                model.particle_flags,
                self.local_correction,
                self.local_hessians,
                self.contact_hessians,
                dt,
            ],
            outputs=[
                self.coarse_rhs,
                self.coarse_mass,
                self.coarse_stiffness,
                self.coarse_contact_stiffness,
            ],
            device=model.device,
        )
        wp.launch(
            _solve_coarse_pcg_persistent,
            dim=_COARSE_PCG_BLOCK_DIM,
            block_dim=_COARSE_PCG_BLOCK_DIM,
            inputs=[
                self.cluster_count,
                self.coarse_neighbor_offsets,
                self.coarse_neighbors,
                self.coarse_neighbor_multiplicity,
                self.coarse_incident_edges,
                self.coarse_anchor_edges,
                self.coarse_rhs,
                self.coarse_mass,
                self.coarse_stiffness,
                self.coarse_contact_stiffness,
                dt,
                self.coupling,
                self.coarse_iterations,
                self.validate_residual,
                self.minimum_residual_reduction,
            ],
            outputs=[
                self.coarse_solution,
                self.coarse_residual,
                self.coarse_preconditioned_residual,
                self.coarse_direction,
                self.coarse_product,
                self.coarse_diagonal,
                self.runtime_status,
                self.runtime_metrics,
                self.runtime_counters,
            ],
            device=model.device,
        )
        wp.launch(
            _prepare_prolonged_corrections,
            dim=self.active_particle_count,
            inputs=[
                self.active_particles,
                self.fine_to_coarse,
                model.particle_radius,
                self.coarse_solution,
                self.relaxation,
                self.max_radius_fraction,
                self.runtime_counters,
            ],
            outputs=[self.fine_correction],
            device=model.device,
        )
        wp.launch(
            _commit_prolonged_corrections,
            dim=self.active_particle_count,
            inputs=[
                self.active_particles,
                self.active_particle_count,
                self.fine_correction,
                self.max_clamp_fraction,
                self.runtime_counters,
                self.runtime_status,
                self.runtime_metrics,
            ],
            outputs=[particle_displacements],
            device=model.device,
        )
