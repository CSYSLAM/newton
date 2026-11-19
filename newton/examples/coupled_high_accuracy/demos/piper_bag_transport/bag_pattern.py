"""Developed T-shirt bag panels with inward side gussets and welded heat seals."""

import numpy as np


def _open_without_stretch(points, triangles, panels, panel_triangles):
    """Fit the opened pose to the flat cutting metric before handing it to the dynamic solver.

    Linear interpolation between a flattened heat seal and an open tube would pre-stretch
    the gussets by 60%. This geometric fit permits the bottom folds to tuck inward instead.
    It changes only the initial pose; every rest length, mass, and simulation coefficient is retained.
    """
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    sides = ((0, 1), (1, 2), (2, 0))
    edges, first = np.unique(np.sort(triangles[:, sides].reshape(-1, 2), axis=1), axis=0, return_index=True)
    panel_edges = panel_triangles[:, sides].reshape(-1, 2)[first]
    lengths = np.linalg.norm(panels[panel_edges[:, 0]] - panels[panel_edges[:, 1]], axis=1)
    points = points.astype(np.float64)
    a, b = edges.T
    count, edge_count = len(points), len(edges)
    rows = np.r_[np.repeat(np.arange(edge_count), 6), edge_count + np.arange(count * 3)]
    cols = np.r_[
        np.column_stack((3 * a[:, None] + np.arange(3), 3 * b[:, None] + np.arange(3))).ravel(), np.arange(count * 3)
    ]
    pose_weight = 0.015

    def residual(flat):
        q = flat.reshape(-1, 3)
        return np.r_[np.linalg.norm(q[a] - q[b], axis=1) - lengths, pose_weight * (q - points).ravel()]

    def jacobian(flat):
        q = flat.reshape(-1, 3)
        direction = q[a] - q[b]
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1.0e-12)
        values = np.r_[np.column_stack((direction, -direction)).ravel(), np.full(count * 3, pose_weight)]
        return coo_matrix((values, (rows, cols)), shape=(edge_count + count * 3, count * 3)).tocsr()

    fit = least_squares(
        residual,
        points.ravel(),
        jac=jacobian,
        max_nfev=60,
        ftol=1.0e-6,
        gtol=1.0e-7,
        tr_options={"atol": 1.0e-5, "btol": 1.0e-5},
    )
    opened = fit.x.reshape(-1, 3)
    opened[:, 2] -= opened[:, 2].min()
    return opened.astype(np.float32)


def create_pattern(cells, cell_size, shoulder_height, handle_width_cells, handle_rise):
    """Return posed vertices/faces and independent, meter-scale flat cutting panels.

    A real gusseted bag is a tube sealed across its flattened bottom and handle tops.
    Four sheets meet along the gusset portions of those seals; there is no separate floor panel.
    Coincident seal vertices are intentionally welded, while the developed panels remain separate.
    """
    nx, ny, nz = cells
    half_width, half_gusset = np.array((nx, ny)) * cell_size * 0.5
    mouth = nz * cell_size
    shoulder = mouth + shoulder_height
    height = shoulder + handle_rise
    vertices, faces, panel_vertices, panel_faces = [], [], [], []
    vertex_ids = {}

    def vertex_id(point):
        key = tuple(np.round(point, 9))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(point)
        return vertex_ids[key]

    def add_panel(world, rest, reverse=False):
        if reverse:
            # Developed triangles must stay counterclockwise even when the outward shell normal flips.
            rest = rest.copy()
            rest[..., 0] *= -1
        rows, columns = world.shape[:2]
        grid = np.array([vertex_id(p) for p in world.reshape(-1, 3)]).reshape(rows, columns)
        panel_start = len(panel_vertices)
        panel_vertices.extend(rest.reshape(-1, 2))
        midpoints = {}
        for v in range(rows - 1):
            for u in range(columns - 1):
                corners = [(v, u), (v, u + 1), (v + 1, u + 1), (v + 1, u)]
                if reverse:
                    corners.reverse()
                center = vertex_id(np.mean([world[p] for p in corners], axis=0))
                panel_center = len(panel_vertices)
                panel_vertices.append(np.mean([rest[p] for p in corners], axis=0))
                # Eight triangles per cell retain the demo's existing film sampling density.
                for a, b in zip(corners, corners[1:] + corners[:1]):
                    midpoint = vertex_id((world[a] + world[b]) * 0.5)
                    edge = tuple(sorted((grid[a], grid[b])))
                    if edge not in midpoints:
                        midpoints[edge] = len(panel_vertices)
                        panel_vertices.append((rest[a] + rest[b]) * 0.5)
                    faces.extend(((grid[a], midpoint, center), (midpoint, grid[b], center)))
                    panel_faces.extend(
                        (
                            (panel_start + a[0] * columns + a[1], midpoints[edge], panel_center),
                            (midpoints[edge], panel_start + b[0] * columns + b[1], panel_center),
                        )
                    )

    def lip_height(x):
        inner = half_width - handle_width_cells * cell_size
        blend = np.clip((np.abs(x) - inner + 3 * cell_size) / (3 * cell_size), 0, 1)
        return mouth + shoulder_height * (1 - np.sqrt(np.maximum(0, 1 - blend**2)))

    def depth(z):
        # Start from an opened tube, retaining visibly inward folds instead of a rectangular box.
        opening = 0.92 * half_gusset
        lower = np.clip(z / 0.09, 0, 1)
        upper = np.clip((height - z) / handle_rise, 0, 1)
        return opening * np.minimum(lower, upper)

    x_values = np.linspace(-half_width, half_width, nx + 1)
    for side in (-1, 1):
        rest = np.empty((nz + 1, nx + 1, 2))
        world = np.empty((nz + 1, nx + 1, 3))
        for u, x in enumerate(x_values):
            z_values = np.linspace(0, lip_height(x), nz + 1)
            rest[:, u] = np.column_stack((np.full(nz + 1, x), z_values))
            world[:, u] = np.column_stack((np.full(nz + 1, x), side * depth(z_values), z_values))
        add_panel(world, rest, reverse=side > 0)
        for start in (0, nx - handle_width_cells):
            xs = x_values[start : start + handle_width_cells + 1]
            zs = np.linspace(shoulder, height, round(handle_rise / cell_size) + 1)
            xx, zz = np.meshgrid(xs, zs)
            world = np.stack((xx, side * depth(zz), zz), axis=-1)
            rest = np.stack((xx, zz), axis=-1)
            add_panel(world, rest, reverse=side > 0)

    # Each V-fold has two independent flat half-panels. At either seal their edges coincide
    # with the front/back heat-seal, producing the physical four-face nonmanifold junction.
    zs = np.r_[np.linspace(0, shoulder, nz + 1), np.linspace(shoulder, height, round(handle_rise / cell_size) + 1)[1:]]
    us = np.linspace(0, 1, ny // 2 + 1)
    for side in (-1, 1):
        for front in (-1, 1):
            uu, zz = np.meshgrid(us, zs)
            yy = front * (1 - uu) * depth(zz)
            fold = np.sqrt(np.maximum(0, half_gusset**2 - depth(zz) ** 2))
            xx = side * (half_width - uu * fold)
            world = np.stack((xx, yy, zz), axis=-1)
            rest = np.stack((uu * half_gusset, zz), axis=-1)
            add_panel(world, rest, reverse=side * front > 0)
    points, triangles = np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)
    panels, panel_triangles = np.asarray(panel_vertices, dtype=np.float32), np.asarray(panel_faces, dtype=np.int32)
    points = _open_without_stretch(points, triangles, panels, panel_triangles)
    return points, triangles, panels, panel_triangles
