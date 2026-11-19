# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Configure cohesive granular particles for the native Newton/PCG solver."""

import itertools

import numpy as np
import warp as wp


def _initial_bonds(points, worlds, indices, radius):
    """Build initial bonds without a dense distance matrix or an optional SciPy dependency."""
    cells = np.floor(points / radius).astype(np.int64)
    buckets = {}
    for i, (cell, world) in enumerate(zip(cells, worlds, strict=True)):
        buckets.setdefault((int(world), *cell), []).append(i)
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    pairs, lengths = [], []
    for i, (cell, world) in enumerate(zip(cells, worlds, strict=True)):
        neighbors = [
            j
            for dx, dy, dz in offsets
            for j in buckets.get((int(world), cell[0] + dx, cell[1] + dy, cell[2] + dz), ())
            if j > i
        ]
        if not neighbors:
            continue
        neighbors = np.asarray(neighbors, dtype=np.int32)
        distance = np.linalg.norm(points[neighbors] - points[i], axis=1)
        selected = (distance > 0.0) & (distance <= radius)
        neighbors, distance = neighbors[selected], distance[selected]
        if len(neighbors):
            pairs.append(np.column_stack((np.full(len(neighbors), indices[i]), indices[neighbors])))
            lengths.append(distance)
    if not pairs:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)
    return np.concatenate(pairs).astype(np.int32), np.concatenate(lengths).astype(np.float32)


def configure_granular(
    model,
    particle_indices,
    *,
    contact_stiffness: float = 2000.0,
    friction: float = 0.5,
    friction_damping: float = 20.0,
    cohesion: float = 200.0,
    bond_damping: float = 0.0,
    yield_strain: float = 0.03,
    max_compaction: float = 0.4,
    break_strain: float = 0.1,
    bond_radius: float | None = None,
    bond_indices=None,
    adhesion: float = 0.0,
    bond_formation_gap: float = 0.0,
    bond_formation_speed: float = 1.0,
    max_dynamic_bonds: int = 12,
) -> None:
    """Select granular particles before constructing ``SolverCoupledHighAccuracy``.

    Particle radii define spherical contact cores. Normal contact stiffness and
    cohesive bond stiffness (``cohesion``) are in N/m; ``friction`` is a Coulomb
    coefficient and ``friction_damping`` its low-speed regularization in N*s/m.
    Grain contacts participate in Newton/PCG even when self collision is disabled.
    Enable self collision separately for contacts with other particles or cloth.
    Shape contacts use the model's ordinary soft-contact and shape materials.

    Initial pairs within ``bond_radius`` are bonded, respecting world boundaries.
    Its default is 2.5 times the largest selected particle radius. Zero cohesion
    disables bonds. Bonds resist relative distance changes; compression beyond
    ``yield_strain`` times the initial length permanently shortens their rest
    length, up to ``max_compaction`` of the initial length. Tension beyond
    ``break_strain`` times the committed rest length breaks the bond irreversibly.
    Explicit ``bond_indices`` (N, 2) can replace the initial radius selection.
    They reference model particle IDs; their rest lengths come from initial positions.
    ``bond_damping`` is axial Kelvin-Voigt damping in N*s/m for intact initial and
    contact-created bonds. It opposes bond length changes, preserves rigid motion,
    and ceases on fracture. Its default of zero preserves undamped bond behavior.

    Positive ``adhesion`` enables new contact bonds in N/m, after accepted steps.
    Pairs must approach within their summed radii plus ``bond_formation_gap`` and
    relative speed at most ``bond_formation_speed``. New bonds start stress-free.
    Each lower-index grain owns at most ``max_dynamic_bonds`` contact bonds (1..64).
    Full slots skip additional adhesion; repulsive contacts remain uncapped.
    Authored bonds never heal. Broken contact bonds can reform after separation
    beyond the formation gap and later recontact. By default adhesion is disabled.
    This is a coarse grain model, without grain rotations,
    persistent tangential friction history, temperature, or phase changes.

    Plastic and fracture history belongs to the solver and advances once per
    substep. Reconstruct the solver to reset history; restoring only particle
    positions and velocities does not reset it. Calling this function again
    replaces the selection and authored material for subsequently created solvers.
    """
    parameters = {
        "contact_stiffness": contact_stiffness,
        "friction": friction,
        "friction_damping": friction_damping,
        "cohesion": cohesion,
        "bond_damping": bond_damping,
        "yield_strain": yield_strain,
        "max_compaction": max_compaction,
        "break_strain": break_strain,
        "adhesion": adhesion,
        "bond_formation_gap": bond_formation_gap,
        "bond_formation_speed": bond_formation_speed,
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value < 0.0 or (name == "contact_stiffness" and value == 0.0):
            raise ValueError(
                f"{name} must be finite and {'positive' if name == 'contact_stiffness' else 'non-negative'}"
            )
    for name in ("yield_strain", "max_compaction"):
        if parameters[name] >= 1.0:
            raise ValueError(f"{name} must be less than 1")
    if bond_radius is not None and (not np.isfinite(bond_radius) or bond_radius <= 0.0):
        raise ValueError("bond_radius must be finite and positive")
    if isinstance(max_dynamic_bonds, (bool, np.bool_)) or not isinstance(max_dynamic_bonds, (int, np.integer)):
        raise ValueError("max_dynamic_bonds must be an integer in [1, 64]")
    if not 1 <= max_dynamic_bonds <= 64:
        raise ValueError("max_dynamic_bonds must be an integer in [1, 64]")
    if bond_radius is not None and bond_indices is not None:
        raise ValueError("bond_radius and explicit bond_indices cannot be used together")
    indices = np.asarray(particle_indices.numpy() if isinstance(particle_indices, wp.array) else particle_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("particle_indices must be a nonempty one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= model.particle_count) or len(np.unique(indices)) != len(indices):
        raise ValueError("granular particle indices must be unique and within the model")
    indices = indices.astype(np.int32)
    fluid = getattr(model, "synreal_fluid_particle_indices", None)
    if fluid is not None and np.intersect1d(indices, fluid.numpy()).size:
        raise ValueError("granular and fluid selections must be disjoint")
    radii = model.particle_radius.numpy()[indices]
    masses = model.particle_mass.numpy()[indices]
    points = model.particle_q.numpy()[indices]
    if not np.isfinite(points).all():
        raise ValueError("granular particle positions must be finite")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("granular particle radii must be finite and positive")
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise ValueError("granular particle masses must be finite and positive")
    flags = model.particle_flags.numpy()[indices]
    if np.any((flags & 1) == 0):
        raise ValueError("granular particles must be active")
    # Authored arrays outlive the non-owning native spans; history remains private to each solver.
    bonds, lengths = np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)
    if bond_indices is not None:
        bonds = np.asarray(bond_indices.numpy() if isinstance(bond_indices, wp.array) else bond_indices)
        if bonds.ndim != 2 or bonds.shape[1] != 2 or bonds.dtype.kind not in "iu":
            raise ValueError("bond_indices must be an (N, 2) integer array")
        if not np.isin(bonds, indices).all() or np.any(bonds[:, 0] == bonds[:, 1]):
            raise ValueError("bond_indices must connect two distinct selected granular particles")
        if len(np.unique(np.sort(bonds, axis=1), axis=0)) != len(bonds):
            raise ValueError("bond_indices must not contain duplicate pairs")
        worlds = model.particle_world.numpy()
        if np.any(worlds[bonds[:, 0]] != worlds[bonds[:, 1]]):
            raise ValueError("bond_indices must remain within one world")
        initial = model.particle_q.numpy()
        lengths = np.linalg.norm(initial[bonds[:, 0]] - initial[bonds[:, 1]], axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("bond_indices require finite, positive initial lengths")
    elif cohesion > 0.0:
        worlds = model.particle_world.numpy()[indices]
        bonds, lengths = _initial_bonds(points, worlds, indices, bond_radius or 2.5 * float(radii.max()))
    model.synreal_granular_particle_indices = wp.array(indices, dtype=wp.int32, device=model.device)
    model.synreal_granular_bond_indices = wp.array(bonds, dtype=wp.vec2i, device=model.device)
    model.synreal_granular_bond_rest_lengths = wp.array(lengths, dtype=wp.float32, device=model.device)
    for name, value in parameters.items():
        setattr(model, f"synreal_granular_{name}", float(value))
    model.synreal_granular_max_dynamic_bonds = int(max_dynamic_bonds)
