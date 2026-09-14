# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read insertion channels and deforming ear geometry from the actual MJCF model."""

import numpy as np


def slot_bounds(model, side):
    """Return signed lateral, longitudinal and vertical bounds in the carton frame."""
    sign = -1 if side == "left" else 1
    outer = model.geom(f"carton/{side}_wall")
    inner = model.geom(f"carton/{side}_rollover")
    return np.array(
        [
            [sign * inner.pos[0] + inner.size[0], sign * outer.pos[0] - outer.size[0]],
            [inner.pos[1] - inner.size[1], min(inner.pos[1] + inner.size[1], outer.pos[1] + outer.size[1])],
            [
                max(inner.pos[2] - inner.size[2], outer.pos[2] - outer.size[2]),
                min(inner.pos[2] + inner.size[2], outer.pos[2] + outer.size[2]),
            ],
        ]
    )


def ear_vertices(model, data, side):
    """Return all collision-mesh vertices of one bent ear in carton coordinates."""
    box = data.body("carton")
    vertices = []
    for suffix in ("", "/strip1", "/strip2"):
        geom = model.geom(f"carton/{side}_ear_panel{suffix}")
        mesh = geom.dataid[0]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + count] @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
        )
        vertices.extend((world - box.xpos) @ box.xmat.reshape(3, 3))
    return np.asarray(vertices)


def aligned_ear_angle(model, data, side):
    """Find an inward fold angle centering the free ear tip on the physical channel."""
    sign = -1 if side == "left" else 1
    box, ear = data.body("carton"), data.body(f"carton/{side}_ear")
    rotation = box.xmat.reshape(3, 3)
    root = rotation.T @ (ear.xpos - box.xpos)
    axis = rotation.T @ ear.xmat.reshape(3, 3)[:, 2]
    vertices = ear_vertices(model, data, side)
    root_local = (vertices - root) @ (rotation.T @ ear.xmat.reshape(3, 3))
    tip = (vertices - root)[sign * root_local[:, 0] > 0.026].mean(axis=0)
    current = float(data.qpos[model.joint(f"crease/{side}_ear").qposadr[0]])
    target = slot_bounds(model, side)[0].mean()
    candidates = np.linspace(1.0, 2.1, 221)
    delta = -sign * candidates - current
    rotated = (
        np.cos(delta)[:, None] * tip
        + np.sin(delta)[:, None] * np.cross(axis, tip)
        + (1 - np.cos(delta))[:, None] * axis * np.dot(axis, tip)
    )
    predicted = sign * (root[0] + rotated[:, 0])
    return float(candidates[np.argmin(abs(predicted - target))])
