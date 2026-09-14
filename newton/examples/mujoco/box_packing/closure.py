# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Geometric closure checks: actual ear vertices must occupy the side channels."""

import numpy as np

from .slots import ear_vertices, slot_bounds


def closure_metrics(model, data):
    box = data.body("carton")
    rotation = box.xmat.reshape(3, 3)
    result = {}
    for side, sign in (("left", -1), ("right", 1)):
        vertices = ear_vertices(model, data, side)
        bounds = slot_bounds(model, side)
        # Only material beyond the rolled-wall entrance counts as inserted.
        deep = vertices[:, 1] > bounds[1, 0]
        x = sign * vertices[:, 0]
        tolerance = 0.0001
        within = (
            (x > bounds[0, 0] - tolerance)
            & (x < bounds[0, 1] + tolerance)
            & (vertices[:, 2] > bounds[2, 0] - tolerance)
            & (vertices[:, 2] < bounds[2, 1] + tolerance)
            & (vertices[:, 1] < bounds[1, 1] + tolerance)
        )
        fraction = float(np.mean(within[deep])) if np.any(deep) else 0.0
        depth = float(np.max(vertices[:, 1]) - bounds[1, 0])
        result[side] = {
            "depth_m": depth,
            "tip_fraction_in_slot": fraction,
            "lateral_tip_error_m": float(np.mean(x[deep]) - bounds[0].mean()) if np.any(deep) else None,
            "slot_bounds_m": bounds.tolist(),
            "inserted": bool(depth >= 0.008 and np.count_nonzero(deep) >= 6 and fraction == 1.0),
        }
    lid = float(data.qpos[model.joint("crease/lid").qposadr[0]])
    result["front_angle_rad"] = float(data.qpos[model.joint("crease/front").qposadr[0]])
    result["ear_angles_rad"] = {
        side: float(data.qpos[model.joint(f"crease/{side}_ear").qposadr[0]]) for side in ("left", "right")
    }
    result["upright_on_table"] = bool(rotation[2, 2] > 0.98 and -0.001 < box.xpos[2] < 0.005)
    wings = [float(data.qpos[model.joint(f"crease/{side}_wing").qposadr[0]]) for side in ("left", "right")]
    result["dust_wings_tucked"] = bool(1.25 < wings[0] < 2.75 and -2.75 < wings[1] < -1.25)
    result["closed"] = bool(
        result["left"]["inserted"]
        and result["right"]["inserted"]
        and abs(lid - np.pi / 2) < 0.10
        and result["upright_on_table"]
        and result["dust_wings_tucked"]
    )
    max_penetration = 0.0
    robot_contact = False
    for contact in data.contact[: data.ncon]:
        names = [model.geom(int(i)).name for i in contact.geom]
        owners = [model.body(model.geom(int(i)).bodyid[0]).name for i in contact.geom]
        if any("ear_panel" in name for name in names):
            max_penetration = max(max_penetration, float(-contact.dist))
        if any(name.startswith("carton") for name in owners) and any(
            name.startswith(("left/", "right/")) for name in owners
        ):
            robot_contact = True
    result["max_ear_penetration_m"] = max_penetration
    result["robot_contact"] = robot_contact
    result["lid_seated"] = bool(
        abs(lid - np.pi / 2) < 0.10 and result["upright_on_table"] and result["dust_wings_tucked"] and not robot_contact
    )
    result["closed"] = result["closed"] and max_penetration < 0.0005 and not robot_contact
    return result
