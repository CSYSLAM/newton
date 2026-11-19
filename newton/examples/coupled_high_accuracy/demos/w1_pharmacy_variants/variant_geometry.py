"""Conforming hollow paperboard tetrahedra with explicit overlap and score-band thickness changes."""

import numpy as np


def make_carton(spec):
    size = np.asarray(spec.size)
    cells = np.asarray(spec.cells)
    vertices = []
    faces = []
    uv = []
    panels = []
    lookup = {}

    def vertex(key):
        key = tuple(key)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append((np.asarray(key) / cells - 0.5) * size)
        return lookup[key]

    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in (0, 1):
            for i in range(cells[other[0]]):
                for j in range(cells[other[1]]):
                    ids = []
                    coords = []
                    for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        key = [0, 0, 0]
                        key[axis] = side * cells[axis]
                        key[other[0]] = u
                        key[other[1]] = v
                        ids.append(vertex(key))
                        coords.append([u / cells[other[0]], v / cells[other[1]]])
                    for order in ((0, 1, 2), (0, 2, 3)):
                        tri = [ids[k] for k in order]
                        tex = [coords[k] for k in order]
                        p = np.asarray([vertices[k] for k in tri])
                        n = np.cross(p[1] - p[0], p[2] - p[0])
                        if n[axis] * (2 * side - 1) < 0:
                            tri.reverse()
                            tex.reverse()
                        faces.append(tri)
                        uv.append(tex)
                        panels.append(2 * axis + side)
    outer = np.asarray(vertices)
    wall = np.full(len(outer), spec.thickness)
    if spec.construction == "reinforced_ends":
        wall[abs(outer[:, 2]) > size[2] / 2 - 0.007] *= 2.0
    elif spec.construction == "side_overlap":
        wall[(outer[:, 0] < -size[0] / 2 + 0.009) & (outer[:, 1] > size[1] / 2 - 0.002)] *= 1.8
    elif spec.construction == "scored_corners":
        corner = (abs(outer[:, 0]) > size[0] / 2 - 0.004) & (abs(outer[:, 1]) > size[1] / 2 - 0.004)
        wall[corner] *= 0.70
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        mask = np.isclose(abs(outer[:, axis]), size[axis] / 2)
        u = outer[mask, other[0]] / size[other[0]] + 0.5
        v = outer[mask, other[1]] / size[other[1]] + 0.5
        outer[mask, axis] += 0.000025 * np.sin(np.pi * u) * np.sin(np.pi * v) * np.sin(3 * np.pi * u)
    inner = outer * ((size / 2 - wall[:, None]) / (size / 2))
    points = np.vstack((outer, inner))
    count = len(outer)
    tets = []
    for a, b, c in np.sort(faces, axis=1):
        tets.extend(((a, b, c, c + count), (a, b, b + count, c + count), (a, a + count, b + count, c + count)))
    tets = np.asarray(tets, np.int32)
    p = points[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flip = vol < 0
    tets[flip, :2] = tets[flip, :2][:, ::-1]
    if abs(vol).min() < 1e-14:
        raise ValueError("Degenerate carton tetrahedron")
    return dict(
        vertices=points.astype(np.float32),
        faces=np.asarray(faces, np.int32),
        tets=tets,
        uv=np.asarray(uv, np.float32),
        panel=np.asarray(panels, np.int32),
        outer_count=count,
        wall=wall,
        size=size,
        thickness=spec.thickness,
    )
