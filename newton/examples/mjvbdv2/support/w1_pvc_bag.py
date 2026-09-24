# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a welded, open PVC shell from the authored clear-bag surfaces.

The source uses a thickened structured film and two hollow swept tubes. Sample
one film midsurface, retain its wrinkles, weld the bottom and bridge the handle
ends into holes in the film. Do not simulate both faces of a 0.24 mm film as
independent cloth or use the 476,580 render triangles for contact.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

# Source Z stays up; source width X becomes the packing scene's width Y.
SOURCE_TO_BAG = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float64)
FILM_COLOR = (0.72, 0.86, 0.90)
HANDLE_COLOR = (0.85, 0.94, 0.97)


def load_pvc_bag(directory: Path, *, coarse_shell=False):
    """Return a connected proxy in meters plus source hashes for later rendering."""
    import trimesh  # noqa: PLC0415

    root = directory.expanduser().resolve()
    info = json.loads((root / "asset_info.json").read_text())
    if info["units"] != "meter" or not np.isclose(info["film_thickness_m"], 0.00024):
        raise ValueError("Expected the meter-scaled 0.24 mm clear PVC bag asset")
    files = ["asset_info.json", "clear_plastic_bag.glb"]

    def read(name, count):
        relative = f"source_meshes/{name}.ply"
        files.append(relative)
        mesh = trimesh.load(root / relative, process=False)
        if mesh.vertices.shape != (count, 3):
            raise ValueError(f"Unsupported clear-bag source topology: {relative}")
        return np.asarray(mesh.vertices)

    film = read("Bag_body_wrinkled_film_0p24mm", 160448)
    film = ((film[:80224] + film[80224:]) * 0.5).reshape(184, 436, 3)
    # Fewer panel vertices let the local solver propagate support from the
    # table to the mouth. Keep the fine swept handles for finger contact.
    nx, ny = (8, 4) if coarse_shell else (18, 10)
    perimeter = np.concatenate(
        [
            np.rint(np.linspace(a, b, n + 1)[:-1]).astype(int)
            for a, b, n in ((0, 138, nx), (138, 218, ny), (218, 356, nx), (356, 436, ny))
        ]
    )
    rows = np.rint(np.linspace(0, 183, 9 if coarse_shell else 25)).astype(int)
    ring = len(perimeter)
    vertices = list(film[rows][:, perimeter].reshape(-1, 3))
    faces = []
    for j in range(len(rows) - 1):
        for i in range(ring):
            a, b = j * ring + i, j * ring + (i + 1) % ring
            faces.extend(((a, b, b + ring), (a, b + ring, a + ring)))

    # Share bottom perimeter vertices with the side film, leaving only the mouth open.
    bottom = read("Bottom_crumpled_sealed_panel", 13200)
    bottom = ((bottom[:6600] + bottom[6600:]) * 0.5).reshape(66, 100, 3)
    grid = np.empty((ny + 1, nx + 1), dtype=int)
    grid[0] = np.arange(nx + 1)
    grid[:, -1] = np.arange(nx, nx + ny + 1)
    grid[-1] = np.arange(nx + ny + nx, nx + ny - 1, -1)
    grid[:, 0] = np.concatenate(([0], np.arange(ring - 1, ring - ny - 1, -1)))
    for j in range(1, ny):
        for i in range(1, nx):
            grid[j, i] = len(vertices)
            vertices.append(bottom[round(j * 65 / ny), round(i * 99 / nx)])
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]
            faces.extend(((a, c, b), (a, d, c)))
    shell_count = len(vertices)
    shell_faces = np.asarray(faces, dtype=np.int32)
    removed, handles, handle_faces = [], [], []
    centers = np.asarray(vertices)[shell_faces].mean(axis=1)
    for name in ("Front_hollow_clear_handle", "Back_hollow_clear_handle"):
        source = read(name, 7040)[:3520].reshape(160, 22, 3)
        # Keep the authored 5.3 mm outer diameter and curved centerline.
        tube = source[np.rint(np.linspace(0, 159, 41)).astype(int)][:, [0, 4, 7, 11, 15, 18]]
        ids = np.arange(len(vertices), len(vertices) + tube.size // 3).reshape(-1, 6)
        vertices.extend(tube.reshape(-1, 3))
        handles.append(ids)
        for row in range(len(ids) - 1):
            for col in range(6):
                a, b = ids[row, col], ids[row, (col + 1) % 6]
                c, d = ids[row + 1, (col + 1) % 6], ids[row + 1, col]
                handle_faces.extend(((a, b, c), (a, c, d)))
        for tube_end in (ids[0][::-1], ids[-1]):
            target = np.asarray(vertices)[tube_end].mean(axis=0)
            distance = np.linalg.norm(centers - target, axis=1)
            distance[removed] = np.inf
            hole = int(distance.argmin())
            removed.append(hole)
            patch = shell_faces[hole][::-1]
            # Rotate the ring correspondence to avoid twisted weld bridges.
            ring_points = np.asarray(vertices)[tube_end]
            patch_points = np.asarray(vertices)[np.repeat(patch, 2)]
            shift = min(range(6), key=lambda k: np.linalg.norm(np.roll(ring_points, k, axis=0) - patch_points))
            weld_ring = np.roll(tube_end, shift)
            # Triangulate a six-to-three ring bridge; global winding is fixed below.
            for j in range(3):
                a, b, c = weld_ring[2 * j], weld_ring[2 * j + 1], weld_ring[(2 * j + 2) % 6]
                u, v = patch[j], patch[(j + 1) % 3]
                handle_faces.extend(((a, b, u), (b, v, u), (b, c, v)))
    shell_faces = np.delete(shell_faces, removed, axis=0)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices) @ SOURCE_TO_BAG.T,
        faces=np.concatenate((shell_faces, handle_faces)),
        process=False,
    )
    mesh.fix_normals()
    return {
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "faces": np.asarray(mesh.faces, dtype=np.int32),
        "shell_count": shell_count,
        "shell_faces": len(shell_faces),
        "bottom": grid.ravel(),
        "rim": np.arange((len(rows) - 1) * ring, len(rows) * ring),
        "side_rings": np.arange(len(rows) * ring).reshape(len(rows), ring),
        "perimeter_corners": np.array((0, nx, nx + ny, 2 * nx + ny)),
        "handles": np.asarray(handles),
        "source": {
            "kind": "clear-pvc-v1",
            "directory": str(root),
            "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
            "sourceToBag": SOURCE_TO_BAG.tolist(),
            "sourceMeshUpAxis": "Z",
            "glbUpAxis": "Y",
            "glbToBag": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "proxyVersion": 2 if coarse_shell else 1,
            "filmThicknessMeters": info["film_thickness_m"],
        },
    }


def add_pvc_material(builder, mesh, *, handle_stiffness=1.0, reinforced=False):
    """Set elastic film, reinforced seams and much stiffer hollow PVC handles.

    These are demo tuning values, not a calibrated PVC grade. Membrane gains
    resist stretching; regional bending allows wrinkles and folding. Handle
    cross-section springs prevent a stiff tube from flattening into a ribbon.
    ``reinforced`` uses the paper-bag panel/fold/rim material profile and molded handles.
    Reinforcement stays within each handle and its small bonded root patches;
    the film body remains free to bend and fold under gravity.
    """
    vertices, faces = mesh["vertices"], mesh["faces"]
    shell_faces = mesh["shell_faces"]
    # 0.24 mm PVC at 1400 kg/m^3, with bonded double plies at the rim.
    masses = np.zeros(len(vertices), dtype=np.float64)
    for i, face in enumerate(faces):
        is_handle = i >= shell_faces
        seam = not is_handle and (np.isin(face, mesh["rim"]).any() or vertices[face, 2].max() < 0.012)
        stiffness = 2e6 * handle_stiffness if is_handle else (6e4 if seam else 2e4)
        if reinforced and not is_handle:
            stiffness = 2e5 if seam else 1e5
        damping = 0.12 if is_handle else (0.4 if reinforced else 0.04)
        builder.tri_materials[i] = (stiffness, stiffness, damping, 0.0, 0.0)
        builder.tri_color[i] = HANDLE_COLOR if is_handle else FILM_COLOR
        density = 1.1 if is_handle else 0.336 * (2 if seam else 1)
        masses[face] += density * builder.tri_areas[i] / 3
    builder.particle_mass[:] = masses.tolist()
    rim = set(mesh["rim"].tolist())
    rings = mesh["side_rings"]
    ring_size = rings.shape[1]
    corners = mesh["perimeter_corners"]
    fold_columns = set(corners) | {(corners[1] + corners[2]) // 2, (corners[3] + ring_size) // 2}
    top_band = vertices[mesh["rim"], 2].min() - 0.04
    for i, edge in enumerate(builder.edge_indices):
        if max(edge) >= mesh["shell_count"]:
            builder.edge_bending_properties[i] = (
                (300.0 if reinforced else 30.0) * handle_stiffness,
                0.2 if reinforced else 0.04,
            )
        elif reinforced:
            # Match the paper bag's local panel/fold/top-hem rules. Do not
            # emulate its stiffness with distances spanning the bag cavity.
            hinge = np.asarray(edge[2:])
            side_edge = np.all(hinge < rings.size)
            corner = side_edge and hinge[0] % ring_size == hinge[1] % ring_size in fold_columns
            bottom_fold = np.all(vertices[hinge, 2] < 0.04)
            points = vertices[np.asarray(edge)[np.asarray(edge) >= 0]]
            if points[:, 2].min() > top_band:
                builder.edge_bending_properties[i] = (240.0, 4.0)
            elif corner or bottom_fold:
                builder.edge_bending_properties[i] = (16.0, 0.6)
            else:
                builder.edge_bending_properties[i] = (60.0, 2.0)
        elif edge[2] in rim or edge[3] in rim:
            builder.edge_bending_properties[i] = (0.8, 0.004)
        else:
            builder.edge_bending_properties[i] = (0.08, 0.002)
    for tube in mesh["handles"]:
        for row in tube:
            for i in range(3):
                builder.add_spring(int(row[i]), int(row[i + 3]), 2e6 * handle_stiffness, 0.02, 0.0)
    if reinforced:
        _reinforce_handles(builder, mesh, handle_stiffness)


def _reinforce_handles(builder, mesh, handle_stiffness):
    """Preserve hollow handle curves without bracing the film across its interior."""
    pairs = {tuple(sorted(pair)) for pair in np.asarray(builder.spring_indices).reshape(-1, 2)}

    def spring(a, b, stiffness, damping=0.2):
        pair = tuple(sorted((int(a), int(b))))
        if pair[0] != pair[1] and pair not in pairs:
            builder.add_spring(*pair, stiffness, damping, 0.0)
            pairs.add(pair)

    rings = mesh["side_rings"]
    top = len(rings) - 1
    vertices = mesh["vertices"]
    for tube in mesh["handles"]:
        last, half = len(tube) - 1, (len(tube) - 1) // 2
        opposite = tube.shape[1] // 2
        # Spread each weld into the adjacent film, not the middle/bottom of
        # the bag. Long handle-to-panel braces make a lifted bag act rigid.
        for end, near in ((0, 1), (last, last - 1)):
            origin = vertices[tube[end]].mean(0)
            column = int(np.linalg.norm(vertices[rings[top], :2] - origin[:2], axis=1).argmin())
            patch = rings[[top - 1, top]][:, [(column - 1) % rings.shape[1], column, (column + 1) % rings.shape[1]]]
            for vertex in patch.ravel():
                for side in (0, 2, 4):
                    spring(tube[near, side], vertex, 1e5 * handle_stiffness)
        anchors = tube[[0, half, last]][:, [0, opposite]].ravel()
        for vertex in tube.ravel():
            for anchor in anchors:
                spring(vertex, anchor, 1e5 * handle_stiffness)
