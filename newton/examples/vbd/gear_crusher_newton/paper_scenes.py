"""Geometry and assets for the paper-scale Newton scenes.

The scene dimensions are based on the authors' public Newton fork.  This
module deliberately has no Newton/Warp dependency so geometry, counts, and
asset integrity can be tested without the optional GPU environment.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ARMADILLO_COMMIT = "59fa41e66f47e0bb4a3f351e89cb56261abda9ab"
ARMADILLO_FILENAME = "Armadilo_15K.1.vtk"
ARMADILLO_SHA256 = "b0fcbe79f1e90f4010b4ad61691e8507d6b342153e96b63bcd585bbc6c8fee71"
ARMADILLO_URL = (
    "https://raw.githubusercontent.com/AnkaChan/newton/"
    f"{ARMADILLO_COMMIT}/newton/examples/mutlphysics/{ARMADILLO_FILENAME}"
)


@dataclass(frozen=True)
class LayeredClothPreset:
    """Topology parameters for one layered-cloth run."""

    layers: int
    vertices_x: int
    vertices_y: int

    @property
    def vertices(self) -> int:
        return self.layers * self.vertices_x * self.vertices_y

    @property
    def triangles(self) -> int:
        return self.layers * 2 * (self.vertices_x - 1) * (self.vertices_y - 1)

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "vertices": self.vertices, "triangles": self.triangles}


LAYERED_CLOTH_PRESETS = {
    # Fast correctness smoke test.
    "smoke": LayeredClothPreset(layers=8, vertices_x=16, vertices_y=20),
    # Interactive-oriented topology that still keeps the visually important 200 layers.
    "visual200": LayeredClothPreset(layers=200, vertices_x=20, vertices_y=24),
    # Higher-quality local topology that fits a 24 GiB GPU but is not real-time.
    "layers200": LayeredClothPreset(layers=200, vertices_x=30, vertices_y=40),
    # Exact topology in the authors' public 200-layer launcher/base scene.
    "author200": LayeredClothPreset(layers=200, vertices_x=60, vertices_y=80),
    # Inferred uniquely from the paper's exact 7.84M-triangle count.
    "paper": LayeredClothPreset(layers=200, vertices_x=141, vertices_y=141),
}


def layered_cloth_counts(layers: int, vertices_x: int, vertices_y: int) -> tuple[int, int]:
    """Return total vertex and triangle counts for independent grid layers."""

    if min(layers, vertices_x, vertices_y) <= 0:
        raise ValueError("layer and grid counts must be positive")
    if vertices_x < 2 or vertices_y < 2:
        raise ValueError("each cloth dimension must contain at least two vertices")
    vertices = layers * vertices_x * vertices_y
    triangles = layers * 2 * (vertices_x - 1) * (vertices_y - 1)
    return vertices, triangles


def generate_cloth_grid(
    vertices_x: int,
    vertices_y: int,
    *,
    size_x: float = 1.2,
    size_y: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the authors' alternating-diagonal regular cloth grid.

    The returned sheet lies in the local XY plane with Z as its normal and is
    centred at the origin.  Faces alternate their diagonal to avoid a global
    directional bias.
    """

    layered_cloth_counts(1, vertices_x, vertices_y)
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError("cloth dimensions must be positive")

    x = np.linspace(-0.5 * size_x, 0.5 * size_x, vertices_x, dtype=np.float32)
    y = np.linspace(-0.5 * size_y, 0.5 * size_y, vertices_y, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack(
        (xx.reshape(-1), yy.reshape(-1), np.zeros(vertices_x * vertices_y, dtype=np.float32))
    )

    faces = np.empty((2 * (vertices_x - 1) * (vertices_y - 1), 3), dtype=np.int32)
    face = 0
    for i in range(vertices_x - 1):
        for j in range(vertices_y - 1):
            v = i * vertices_y + j
            if (i + j) % 2:
                faces[face] = (v, v + vertices_y + 1, v + 1)
                faces[face + 1] = (v, v + vertices_y, v + vertices_y + 1)
            else:
                faces[face] = (v, v + vertices_y, v + 1)
                faces[face + 1] = (v + vertices_y, v + vertices_y + 1, v + 1)
            face += 2
    return vertices, faces


def layered_grid_coloring(
    layers: int,
    vertices_x: int,
    vertices_y: int,
    *,
    period: int = 3,
) -> list[np.ndarray]:
    """Return an analytic distance-two coloring for stacked independent grids.

    A 3x3 periodic pattern separates triangle neighbours and the opposite
    vertices of every bending hinge.  Reusing colors between disconnected
    layers avoids an expensive graph-coloring preprocessing pass.
    """

    layered_cloth_counts(layers, vertices_x, vertices_y)
    if period < 3:
        raise ValueError("period must be at least three for bending-safe coloring")

    per_layer = vertices_x * vertices_y
    layer_offsets = np.arange(layers, dtype=np.int64) * per_layer
    groups: list[np.ndarray] = []
    grid = np.arange(per_layer, dtype=np.int64).reshape(vertices_x, vertices_y)
    for i_mod in range(period):
        for j_mod in range(period):
            local = grid[i_mod::period, j_mod::period].reshape(-1)
            global_indices = (layer_offsets[:, None] + local[None, :]).reshape(-1)
            groups.append(global_indices.astype(np.int32, copy=False))
    return groups


def create_gear_cylinder_mesh(
    *,
    inner_radius: float = 0.36,
    outer_radius: float = 0.40,
    length: float = 1.6,
    num_teeth: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the extruded 16-tooth roller used by the authors' crusher scene."""

    if not (0.0 < inner_radius < outer_radius):
        raise ValueError("gear radii must satisfy 0 < inner_radius < outer_radius")
    if length <= 0.0 or num_teeth < 3:
        raise ValueError("gear length must be positive and it must have at least three teeth")

    profile: list[tuple[float, float]] = []
    tooth_angle = 2.0 * np.pi / num_teeth
    for tooth in range(num_teeth):
        base = tooth * tooth_angle
        profile.extend(
            (
                (base, inner_radius),
                (base + tooth_angle * 0.08, inner_radius),
                (base + tooth_angle * 0.15, outer_radius),
                (base + tooth_angle * 0.85, outer_radius),
                (base + tooth_angle * 0.92, inner_radius),
            )
        )

    half_length = 0.5 * length
    vertices: list[tuple[float, float, float]] = []
    for x in (-half_length, half_length):
        vertices.extend((x, radius * np.cos(angle), radius * np.sin(angle)) for angle, radius in profile)

    profile_count = len(profile)
    faces: list[tuple[int, int, int]] = []
    for i in range(profile_count):
        nxt = (i + 1) % profile_count
        faces.append((i, nxt, profile_count + nxt))
        faces.append((i, profile_count + nxt, profile_count + i))

    left_center = len(vertices)
    vertices.append((-half_length, 0.0, 0.0))
    right_center = len(vertices)
    vertices.append((half_length, 0.0, 0.0))
    for i in range(profile_count):
        nxt = (i + 1) % profile_count
        faces.append((left_center, nxt, i))
        faces.append((right_center, profile_count + i, profile_count + nxt))

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_vtk_tet_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an ASCII legacy-VTK tetrahedral mesh without an extra dependency."""

    source = Path(path)
    tokens = source.read_text(encoding="utf-8").split()
    try:
        points_at = tokens.index("POINTS")
        point_count = int(tokens[points_at + 1])
        point_values_at = points_at + 3
        point_values_end = point_values_at + 3 * point_count
        vertices = np.asarray(tokens[point_values_at:point_values_end], dtype=np.float32).reshape(-1, 3)

        cells_at = tokens.index("CELLS", point_values_end)
        cell_count = int(tokens[cells_at + 1])
        cursor = cells_at + 3
        tetrahedra: list[list[int]] = []
        for _ in range(cell_count):
            arity = int(tokens[cursor])
            cursor += 1
            cell = [int(index) for index in tokens[cursor : cursor + arity]]
            cursor += arity
            if arity == 4:
                tetrahedra.append(cell)
    except (ValueError, IndexError) as error:
        raise ValueError(f"invalid legacy VTK mesh: {source}") from error

    if len(vertices) != point_count or not tetrahedra:
        raise ValueError(f"VTK mesh has incomplete points or no tetrahedra: {source}")
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


def signed_tetrahedron_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    """Return signed volumes for a tetrahedral mesh (positive is rest orientation)."""

    points = np.asarray(vertices, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tetrahedra must have shape (M, 4)")
    if tets.size and (np.min(tets) < 0 or np.max(tets) >= len(points)):
        raise ValueError("tetrahedron index is outside the vertex array")

    a = points[tets[:, 0]]
    b = points[tets[:, 1]]
    c = points[tets[:, 2]]
    d = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def file_sha256(path: str | Path) -> str:
    """Compute a file SHA-256 without loading the complete asset in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_armadillo_path() -> Path:
    """Return the bundled Armadillo VTK path next to this module."""
    return Path(__file__).resolve().parent / ARMADILLO_FILENAME


def fetch_armadillo_asset(destination: str | Path | None = None) -> Path:
    """Fetch and verify the authors' exact 15K/60K Armadillo research asset."""

    target = Path(destination) if destination is not None else default_armadillo_path()
    if target.exists():
        actual = file_sha256(target)
        if actual == ARMADILLO_SHA256:
            return target
        raise RuntimeError(f"existing Armadillo asset has unexpected SHA-256: {actual}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    request = urllib.request.Request(ARMADILLO_URL, headers={"User-Agent": "planar-dat-reproduction/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        actual = file_sha256(temporary)
        if actual != ARMADILLO_SHA256:
            raise RuntimeError(f"downloaded Armadillo SHA-256 mismatch: {actual}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
