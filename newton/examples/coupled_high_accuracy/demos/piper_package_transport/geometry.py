"""Closed film envelopes and conforming tetrahedra for two independent payloads."""

import numpy as np

# The envelope has a book-sized load and a generous flat perimeter for gripping the film alone.
PAYLOAD_SIZE = (0.170, 0.200, 0.020)
BAG_SIZE = (0.230, 0.280, 0.038)
GRASP_OFFSET = 0.135


def rounded_grid(size, cells, radius):
    size = np.asarray(size, dtype=float)
    axes = [np.linspace(-s / 2, s / 2, n + 1) for s, n in zip(size, cells)]
    points = np.array([(x, y, z) for z in axes[2] for y in axes[1] for x in axes[0]])
    inner = np.clip(points, -size / 2 + radius, size / 2 - radius)
    offset = points - inner
    points = inner + offset * np.minimum(1, radius / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-12))
    return points.astype(np.float32)


def tetrahedral_box(size=PAYLOAD_SIZE, cells=(10, 14, 2), radius=0.004):
    points = rounded_grid(size, cells, radius)
    nx, ny, nz = cells

    def index(x, y, z):
        return (z * (ny + 1) + y) * (nx + 1) + x

    tets = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                nodes = [
                    index(x, y, z),
                    index(x + 1, y, z),
                    index(x + 1, y, z + 1),
                    index(x, y, z + 1),
                    index(x, y + 1, z),
                    index(x + 1, y + 1, z),
                    index(x + 1, y + 1, z + 1),
                    index(x, y + 1, z + 1),
                ]
                if (x + y + z) % 2:
                    split = ((0, 1, 4, 3), (2, 3, 6, 1), (5, 4, 1, 6), (7, 6, 3, 4), (4, 1, 6, 3))
                else:
                    split = ((1, 2, 5, 0), (3, 0, 7, 2), (4, 7, 0, 5), (6, 5, 2, 7), (5, 2, 7, 0))
                tets.extend(tuple(nodes[i] for i in tet) for tet in split)
    return points, np.asarray(tets, dtype=np.int32)


def sealed_bag(size=BAG_SIZE, spacing=0.006):
    """Two rectangular cutting panels, opened around the load and welded only at their perimeter.

    The developed coordinates include the arc length needed to wrap the payload. There are no side-wall
    panels, and the common perimeter has zero opening rather than the vertical walls of a box mesh.
    """
    nx, ny = np.ceil(np.asarray(size[:2]) / spacing).astype(int)
    axes = [np.linspace(-s / 2, s / 2, n + 1) for s, n in zip(size[:2], (nx, ny))]
    profiles, developed = [], []
    for axis, payload in zip(axes, PAYLOAD_SIZE):
        shoulder = payload / 2 + 0.003
        t = np.clip((np.abs(axis) - shoulder) / (axis[-1] - shoulder), 0, 1)
        height = size[2] / 2 * (1 - t * t * (3 - 2 * t))
        profiles.append(height / (size[2] / 2))
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(axis), np.diff(height)))]
        developed.append((arc - arc[-1] / 2) * 1.015)
    xx, yy = np.meshgrid(*axes)
    height = size[2] / 2 * profiles[1][:, None] * profiles[0][None, :]
    # Small excess film makes real folds under gravity rather than a taut cuboid shell.
    height += 0.001 * np.sin(6 * np.pi * xx / size[0]) * np.sin(4 * np.pi * yy / size[1]) * height / (size[2] / 2)
    upper = np.stack((xx, yy, height), axis=-1).reshape(-1, 3)
    boundary = np.zeros((ny + 1, nx + 1), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    boundary = boundary.ravel()
    lower = upper.copy()
    lower[:, 2] *= -1
    bottom_ids = np.arange(len(upper))
    bottom_ids[~boundary] = len(upper) + np.arange(np.count_nonzero(~boundary))
    points = np.vstack((upper, lower[~boundary])).astype(np.float32)
    faces = []
    for y in range(ny):
        for x in range(nx):
            a = y * (nx + 1) + x
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            # A corner must connect to an interior node; a three-boundary-node triangle would
            # otherwise be duplicated by the lower panel and enclose no volume.
            if (x == nx - 1 and y == 0) or (x == 0 and y == ny - 1):
                faces.extend(((a, b, d), (b, c, d)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    top = np.asarray(faces, dtype=np.int32)
    faces = np.vstack((top, bottom_ids[top[:, ::-1]])).astype(np.int32)
    uu, vv = np.meshgrid(*developed)
    panel = np.column_stack((uu.ravel(), vv.ravel()))
    uv = np.vstack((panel, panel * (-1, 1))).astype(np.float32)
    uv_faces = np.vstack((top, top[:, ::-1] + len(upper))).astype(np.int32)
    return points, faces, uv, uv_faces


def six_volumes(points, tets):
    p = points[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
