# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Configure the native single-phase fluid material on a finalized Newton model."""

import numpy as np
import warp as wp


def configure_fluid(
    model,
    particle_indices,
    *,
    smoothing_radius: float,
    rest_density: float = 1000.0,
    bulk_modulus: float = 200000.0,
    viscosity: float = 0.0,
    boundary_positions=None,
    boundary_volumes=None,
) -> None:
    """Select fluid particles before constructing ``SolverCoupledHighAccuracy``.

    The particles retain their existing masses, positions, velocities, and collision
    radii. Density pressure and viscosity participate in the native Newton/PCG solve.
    ``viscosity`` is dynamic viscosity in Pa*s; ``bulk_modulus`` is in Pa. The kernel
    has compact support at ``smoothing_radius`` meters. Neighbor search excludes all
    unselected particles, including cloth and affine coordinates.

    Optional fixed boundary samples contribute their quadrature volume to density.
    Author collision shapes separately to prevent penetration. Boundary samples are
    static world-space data, not moving rigid-body or cloth boundary attachments.
    Multiple Newton worlds are isolated; fixed samples are shared world-space geometry.
    Calling this again replaces the material and its selection on the model.
    """
    for name, value, positive in (
        ("smoothing_radius", smoothing_radius, True),
        ("rest_density", rest_density, True),
        ("bulk_modulus", bulk_modulus, False),
        ("viscosity", viscosity, False),
    ):
        if not np.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    indices = np.asarray(particle_indices.numpy() if isinstance(particle_indices, wp.array) else particle_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("particle_indices must be a nonempty one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= model.particle_count) or len(np.unique(indices)) != len(indices):
        raise ValueError("fluid particle indices must be unique and within the model")
    granular = getattr(model, "synreal_granular_particle_indices", None)
    if granular is not None and np.intersect1d(indices, granular.numpy()).size:
        raise ValueError("granular and fluid selections must be disjoint")
    if (boundary_positions is None) != (boundary_volumes is None):
        raise ValueError("boundary_positions and boundary_volumes must be supplied together")
    positions = np.empty((0, 3), dtype=np.float32) if boundary_positions is None else np.asarray(boundary_positions)
    volumes = np.empty(0, dtype=np.float32) if boundary_volumes is None else np.asarray(boundary_volumes)
    if positions.ndim != 2 or positions.shape[1] != 3 or volumes.shape != (len(positions),):
        raise ValueError("boundary samples require positions of shape (N, 3) and volumes of shape (N,)")
    if not np.isfinite(positions).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("boundary positions must be finite and volumes positive and finite")
    # The model owns these allocations because native Span properties are non-owning.
    model.synreal_fluid_particle_indices = wp.array(indices, dtype=wp.int32, device=model.device)
    model.synreal_fluid_boundary_positions = wp.array(positions, dtype=wp.vec3, device=model.device)
    model.synreal_fluid_boundary_volumes = wp.array(volumes, dtype=wp.float32, device=model.device)
    model.synreal_fluid_smoothing_radius = float(smoothing_radius)
    model.synreal_fluid_rest_density = float(rest_density)
    model.synreal_fluid_bulk_modulus = float(bulk_modulus)
    model.synreal_fluid_viscosity = float(viscosity)
