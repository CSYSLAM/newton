"""Pillow-shaped chips bag: a welded two-panel film envelope around a soft chips-and-air core.

Local frame: x is the bag width, y its length (+y is the top crimp seal), z its thickness.
"""

import numpy as np

# The supplied Lay's model is 0.30 x 0.44 m; half scale gives a single-serve bag that fits half of the basket.
BAG_WIDTH, BAG_LENGTH, BAG_THICKNESS = 0.150, 0.220, 0.085
SIDE_SEAL, END_SEAL = 0.006, 0.015
# The core stops well short of the top seal so the closed fingers only gather film.
CORE_INSET = (0.012, 0.024)
FILM_GAP = 0.0035
SPACING = 0.005

# Texture atlas regions of the supplied snack004 material: st range times its UsdTransform2d scale 3.937.
FRONT_UV = ((0.0433, 0.409), (0.429, 0.992))
BACK_UV = ((0.512, 0.004), (0.898, 0.591))


def inflation_profile(coordinate, inflated_half, power):
    """Superelliptic cushion: nearly full thickness over the middle, closing quickly only near the edge.

    A nitrogen-filled pillow pack is a flattened film tube: rounded folds at the sides (power 3) and a short,
    steep taper into the crimp seals at the ends (power 6).
    """
    t = np.clip(np.abs(coordinate) / inflated_half, 0, 1)
    return (1 - t**power) ** 0.5


WIDTH_POWER, LENGTH_POWER = 3.0, 6.0


# Coincident panels inside the flat seals would sit inside each other's self-collision radius (1.5 mm) and chatter.
# Keep a 4 mm thick seal band that closes to zero only over the last few millimetres at the welded perimeter.
SEAL_HALF_THICKNESS, SEAL_RAMP = 0.002, 0.004


def profiles(x, y):
    return inflation_profile(x, BAG_WIDTH / 2 - SIDE_SEAL, WIDTH_POWER) * inflation_profile(
        y, BAG_LENGTH / 2 - END_SEAL, LENGTH_POWER
    )


def seal_floor(distance_to_edge):
    return SEAL_HALF_THICKNESS * np.clip(distance_to_edge / SEAL_RAMP, 0, 1)


def film_height(x, y):
    edge = np.minimum(BAG_WIDTH / 2 - np.abs(x), BAG_LENGTH / 2 - np.abs(y))
    return np.maximum(BAG_THICKNESS / 2 * profiles(x, y), seal_floor(edge))


def film_mesh(spacing=SPACING):
    """Return particles, triangles, developed panel coordinates and per-corner texture coordinates.

    Rest lengths come from separable arc lengths of the inflated profile, as in the courier envelope, so the film
    starts nearly stress-free around the inflated core. Both panels share only their perimeter vertices.
    """
    nx, ny = np.ceil(np.array((BAG_WIDTH, BAG_LENGTH)) / spacing).astype(int)
    axes = (np.linspace(-BAG_WIDTH / 2, BAG_WIDTH / 2, nx + 1), np.linspace(-BAG_LENGTH / 2, BAG_LENGTH / 2, ny + 1))
    halves = (BAG_WIDTH / 2 - SIDE_SEAL, BAG_LENGTH / 2 - END_SEAL)
    developed = []
    for axis, half, power in zip(axes, halves, (WIDTH_POWER, LENGTH_POWER)):
        height = np.maximum(
            BAG_THICKNESS / 2 * inflation_profile(axis, half, power), seal_floor(axis[-1] - np.abs(axis))
        )
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(axis), np.diff(height)))]
        # No excess film: trapped air keeps a sealed chips bag taut, and load still wrinkles it near the seals.
        developed.append(arc - arc[-1] / 2)
    xx, yy = np.meshgrid(*axes)
    upper = np.stack((xx, yy, film_height(xx, yy)), axis=-1).reshape(-1, 3)
    boundary = np.zeros((ny + 1, nx + 1), dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    boundary = boundary.ravel()
    lower = upper * (1, 1, -1)
    lower_ids = np.arange(len(upper))
    lower_ids[~boundary] = len(upper) + np.arange(np.count_nonzero(~boundary))
    points = np.vstack((upper, lower[~boundary])).astype(np.float32)
    faces = []
    for y in range(ny):
        for x in range(nx):
            a = y * (nx + 1) + x
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            # A corner triangle made of three perimeter vertices would be shared by both panels.
            if (x == nx - 1 and y == 0) or (x == 0 and y == ny - 1):
                faces.extend(((a, b, d), (b, c, d)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    top = np.asarray(faces, dtype=np.int32)
    faces = np.vstack((top, lower_ids[top[:, ::-1]])).astype(np.int32)
    uu, vv = np.meshgrid(*developed)
    panel = np.column_stack((uu.ravel(), vv.ravel()))
    panel_uv = np.vstack((panel, panel * (-1, 1))).astype(np.float32)
    panel_faces = np.vstack((top, top[:, ::-1] + len(upper))).astype(np.int32)
    texture_uv = np.vstack((atlas(panel, FRONT_UV), atlas(panel * (-1, 1), BACK_UV))).astype(np.float32)
    return points, faces, panel_uv, panel_faces, texture_uv


def atlas(panel, region):
    # The upper panel shows the front print and the mirrored lower panel the back print, both read upright.
    low, high = panel.min(0), panel.max(0)
    (u0, v0), (u1, v1) = region
    return np.column_stack(
        (
            u0 + (panel[:, 0] - low[0]) / (high[0] - low[0]) * (u1 - u0),
            v0 + (panel[:, 1] - low[1]) / (high[1] - low[1]) * (v1 - v0),
        )
    )


def core_mesh(cells=(8, 12, 2)):
    """Tetrahedral chips-and-air core that follows the pillow profile with a small clearance to the film."""
    half = np.array((BAG_WIDTH / 2 - SIDE_SEAL - CORE_INSET[0], BAG_LENGTH / 2 - END_SEAL - CORE_INSET[1]))
    nx, ny, nz = cells
    xs, ys, zs = (
        np.linspace(-half[0], half[0], nx + 1),
        np.linspace(-half[1], half[1], ny + 1),
        np.linspace(-1, 1, nz + 1),
    )
    points = np.array(
        [(x, y, z * max(BAG_THICKNESS / 2 * profiles(x, y) - FILM_GAP, 0.004)) for z in zs for y in ys for x in xs],
        dtype=np.float32,
    )

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


def six_volumes(points, tets):
    p = points[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))
