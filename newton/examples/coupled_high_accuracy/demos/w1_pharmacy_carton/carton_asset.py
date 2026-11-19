"""A closed, hollow paperboard shell with a conforming through-thickness tetrahedral layer."""

import numpy as np

SIZE = np.array([0.09, 0.042, 0.12])
THICKNESS = 0.00045
CENTERS = np.array([[0.47, 0.28, 0.9205], [0.47, -0.28, 0.9205]])


def make_carton(cells=(12, 6, 16), thickness=THICKNESS):
    cells = np.asarray(cells)
    vertices = []
    faces = []
    uv = []
    lookup = {}
    panel = []

    def vertex(key):
        key = tuple(key)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append((np.asarray(key) / cells - 0.5) * SIZE)
        return lookup[key]

    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for side in (0, 1):
            for i in range(cells[others[0]]):
                for j in range(cells[others[1]]):
                    ids = []
                    coords = []
                    for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        key = [0, 0, 0]
                        key[axis] = side * cells[axis]
                        key[others[0]] = u
                        key[others[1]] = v
                        ids.append(vertex(key))
                        coords.append([u / cells[others[0]], v / cells[others[1]]])
                    for order in ((0, 1, 2), (0, 2, 3)):
                        tri = [ids[k] for k in order]
                        tex = [coords[k] for k in order]
                        p = np.asarray([vertices[k] for k in tri])
                        normal = np.cross(p[1] - p[0], p[2] - p[0])
                        if normal[axis] * (2 * side - 1) < 0:
                            tri = tri[::-1]
                            tex = tex[::-1]
                        faces.append(tri)
                        uv.append(tex)
                        panel.append(axis * 2 + side)
    outer = np.asarray(vertices)
    # Small manufacturing imperfections trigger natural buckling rather than a perfectly symmetric bifurcation.
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        mask = np.isclose(abs(outer[:, axis]), SIZE[axis] / 2)
        u = outer[mask, others[0]] / SIZE[others[0]] + 0.5
        v = outer[mask, others[1]] / SIZE[others[1]] + 0.5
        outer[mask, axis] += 0.00003 * np.sin(np.pi * u) * np.sin(np.pi * v) * np.sin(3 * np.pi * u)
    inner = outer * ((SIZE / 2 - thickness) / (SIZE / 2))
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
        raise ValueError("Degenerate paperboard tetrahedron")
    return dict(
        vertices=points.astype(np.float32),
        outer_count=count,
        faces=np.asarray(faces, np.int32),
        tets=tets,
        uv=np.asarray(uv, np.float32),
        panel=np.asarray(panel, np.int32),
        size=SIZE,
        thickness=thickness,
    )
