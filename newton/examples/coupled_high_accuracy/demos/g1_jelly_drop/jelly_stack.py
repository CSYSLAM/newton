"""Build a freestanding pyramid of individual tetrahedral jelly cubes."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

JELLY_SIDE = 0.40
JELLY_LAYERS = 3
JELLY_GAP = 0.02
JELLY_LAYER_GAP = 0.02
JELLY_BASE_HEIGHT = 0.04
JELLY_DENSITY = 450.0
# A firmer shear response restores shape quickly; volume stiffness keeps the loaded stack from collapsing.
JELLY_MU = 4.0e3
JELLY_LAMBDA = 8.0e4
JELLY_DAMPING = 12.0
JELLY_CONTACT_RADIUS = 0.006
# The three tiers nearest the top are intentionally easy to count in the viewer:
# one blue cube, four green cubes, and nine red cubes. Lower tiers keep the original palette.
TOP_LAYER_COLORS = {1: (0.42, 0.62, 1.0), 2: (0.42, 0.95, 0.58), 3: (1.0, 0.42, 0.42)}
LAYER_COLORS = ((0.12, 0.72, 0.95), (0.23, 0.93, 0.48), (1.0, 0.65, 0.08), (0.96, 0.20, 0.58))


def signed_six_volumes(positions: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Compute oriented volume times six, using indices local to one jelly cube."""
    points = positions[tetrahedra]
    return np.einsum(
        "ij,ij->i", points[:, 1] - points[:, 0], np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 0])
    )


def jelly_cube_mesh(side: float = JELLY_SIDE, subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return a lightly rounded cube with planar load-bearing regions on every face."""
    if not np.isfinite(side) or side <= 0.0 or subdivisions < 2:
        raise ValueError("Jelly cube side must be positive and subdivisions must be at least two")
    coordinates = np.linspace(-side / 2.0, side / 2.0, subdivisions + 1, dtype=np.float64)
    vertices = np.asarray([(x, y, z) for z in coordinates for y in coordinates for x in coordinates])
    rounding = side * 0.06
    inner = np.clip(vertices, -side / 2.0 + rounding, side / 2.0 - rounding)
    offset = vertices - inner
    length = np.linalg.norm(offset, axis=1, keepdims=True)
    # Only edges and corners move inward; broad flat faces keep stacked cubes supported.
    vertices = inner + offset * np.minimum(1.0, rounding / np.maximum(length, 1.0e-12))
    row = subdivisions + 1

    def index(x, y, z):
        return z * row * row + y * row + x

    tetrahedra = []
    for z in range(subdivisions):
        for y in range(subdivisions):
            for x in range(subdivisions):
                nodes = (
                    index(x, y, z),
                    index(x + 1, y, z),
                    index(x + 1, y, z + 1),
                    index(x, y, z + 1),
                    index(x, y + 1, z),
                    index(x + 1, y + 1, z),
                    index(x + 1, y + 1, z + 1),
                    index(x, y + 1, z + 1),
                )
                # Alternate cell diagonals so neighboring tetrahedra share complete faces.
                if (x & 1) ^ (y & 1) ^ (z & 1):
                    cells = ((0, 1, 4, 3), (2, 3, 6, 1), (5, 4, 1, 6), (7, 6, 3, 4), (4, 1, 6, 3))
                else:
                    cells = ((1, 2, 5, 0), (3, 0, 7, 2), (4, 7, 0, 5), (6, 5, 2, 7), (5, 2, 7, 0))
                tetrahedra.extend(tuple(nodes[i] for i in tet) for tet in cells)
    vertices = vertices.astype(np.float32)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    if np.min(signed_six_volumes(vertices, tetrahedra)) <= 0.0:
        raise ValueError("The jelly cube mesh contains an inverted tetrahedron")
    return vertices, tetrahedra


def add_jelly_stack(
    builder: newton.ModelBuilder,
    *,
    side: float = JELLY_SIDE,
    gap: float | None = None,
    layer_gap: float = JELLY_LAYER_GAP,
    base_z: float = JELLY_BASE_HEIGHT,
    layers: int = JELLY_LAYERS,
    subdivisions: int = 3,
    center_xy: tuple[float, float] = (0.0, 0.0),
    shear_modulus: float = JELLY_MU,
    lame_lambda: float = JELLY_LAMBDA,
    damping: float = JELLY_DAMPING,
) -> dict:
    """Add a square pyramid with exactly one cube layer in each tier.

    Every cube has independent nodes and topology; there are no attachments or
    hidden support bodies. Tetrahedron indices are local to each group. Particle
    and triangle ranges address the complete caller model and are half open.
    The bottom grid has ``layers`` cubes per side; successive grids shrink by one,
    ending in a single cube. The horizontal gap and contact thickness scale with cube size.
    ``layer_gap`` is the vertical surface clearance between tiers; ``base_z`` is the bottom surface height.
    """
    if not all(np.isfinite(value) and value > 0.0 for value in (shear_modulus, lame_lambda)):
        raise ValueError("Jelly material stiffnesses must be finite and positive")
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("Jelly damping must be finite and nonnegative")
    vertices, tetrahedra = jelly_cube_mesh(side, subdivisions)
    scale = side / JELLY_SIDE
    if gap is None:
        gap = JELLY_GAP * scale
    contact_radius = JELLY_CONTACT_RADIUS * scale
    if layers < 1 or not all(np.isfinite(value) and value >= 0.0 for value in (gap, layer_gap, base_z)):
        raise ValueError("The stack needs positive tiers and finite, nonnegative gaps and base height")
    volumes = signed_six_volumes(vertices, tetrahedra)
    start = len(builder.particle_q)
    groups, centers, layer_ranges = {}, [], []
    spacing = side + gap
    # Vertical clearance is independent of horizontal spacing, so lifting tiers does not widen the pyramid.
    layer_spacing = side + layer_gap
    for layer in range(layers):
        count = layers - layer
        first = len(groups)
        for y in range(count):
            for x in range(count):
                center = (
                    center_xy[0] + (x - (count - 1) / 2.0) * spacing,
                    center_xy[1] + (y - (count - 1) / 2.0) * spacing,
                    base_z + side / 2.0 + layer * layer_spacing,
                )
                name = f"jelly_{layer}_{y}_{x}"
                particle_start = len(builder.particle_q)
                triangle_start = len(builder.tri_indices)
                tet_start = len(builder.tet_indices)
                builder.add_soft_mesh(
                    pos=wp.vec3(*center),
                    rot=wp.quat_identity(),
                    scale=1.0,
                    vel=wp.vec3(0.0),
                    vertices=vertices,
                    indices=tetrahedra.ravel(),
                    density=JELLY_DENSITY,
                    k_mu=shear_modulus,
                    k_lambda=lame_lambda,
                    k_damp=damping,
                    particle_radius=contact_radius,
                    add_surface_mesh_edges=True,
                    validate_mesh=True,
                    label=name,
                )
                # ``tier_from_top`` maps the default 1/4/9 cube tiers to blue/green/red,
                # while keeping configurable lower tiers visually distinct with the old palette.
                tier_from_top = layers - layer
                if tier_from_top in TOP_LAYER_COLORS:
                    color = np.asarray(TOP_LAYER_COLORS[tier_from_top], dtype=np.float32)
                else:
                    color = np.asarray(LAYER_COLORS[layer % len(LAYER_COLORS)], dtype=np.float32)
                groups[name] = {
                    "particle_range": (particle_start, len(builder.particle_q)),
                    "triangle_range": (triangle_start, len(builder.tri_indices)),
                    "tet_range": (tet_start, len(builder.tet_indices)),
                    "initial_positions": np.asarray(builder.particle_q[particle_start:], dtype=np.float32).copy(),
                    "tet_indices": tetrahedra.copy(),
                    "initial_signed_volumes": volumes.copy(),
                    "fixed_indices": np.empty(0, dtype=np.int32),
                    "fixed_positions": np.empty((0, 3), dtype=np.float32),
                    "kind": "solid",
                    "color": tuple(color),
                    "center": center,
                    "layer": layer,
                    "tier": layer,
                    "mass": float(np.sum(builder.particle_mass[particle_start:])),
                }
                centers.append(center)
        layer_ranges.append((first, len(groups)))
    return {
        "groups": groups,
        "centers": np.asarray(centers, dtype=np.float32),
        "layer_ranges": tuple(layer_ranges),
        "top_z": base_z + layers * side + (layers - 1) * layer_gap,
        "tier_count": layers,
        "layer_count": layers,
        "contact_radius": contact_radius,
        "block_side": side,
        "block_gap": gap,
        "block_count": len(groups),
        "layer_gap": layer_gap,
        "base_z": base_z,
        "particle_range": (start, len(builder.particle_q)),
        "fixed_indices": np.empty(0, dtype=np.int32),
        "fixed_positions": np.empty((0, 3), dtype=np.float32),
    }
