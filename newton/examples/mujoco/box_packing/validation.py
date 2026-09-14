# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared observation checks for interactive and headless packing runs."""

import mujoco
import numpy as np

from .closure import closure_metrics


def verify_hold(model, times, configurations, seconds=2.0):
    """Evaluate the full released hold, never just a visually convenient final frame."""
    if len(times) == 0:
        return False, {}
    data = mujoco.MjData(model)
    passed = times[-1] - times[0] >= seconds
    metrics = {}
    sample_count = 0
    failed_samples = 0
    minimum_depth = {"left": np.inf, "right": np.inf}
    for time, configuration in zip(times, configurations, strict=True):
        if time >= times[-1] - seconds:
            data.qpos[:] = configuration
            mujoco.mj_forward(model, data)
            metrics = closure_metrics(model, data)
            sample_count += 1
            failed_samples += not metrics["closed"]
            for side, depth in minimum_depth.items():
                minimum_depth[side] = min(depth, metrics[side]["depth_m"])
            passed = passed and metrics["closed"]
    metrics["hold_samples"] = sample_count
    metrics["failed_hold_samples"] = failed_samples
    metrics["minimum_hold_depth_m"] = minimum_depth
    return passed, metrics


def rear_support_contact(model, data):
    """Check real right-hand/rear-wall contact, including unnamed CAD collision meshes."""
    for contact in data.contact:
        names = [model.geom(int(g)).name for g in contact.geom]
        owners = [model.body(model.geom_bodyid[int(g)]).name for g in contact.geom]
        if "carton/back_wall" in names and any(owner.startswith("right/") for owner in owners):
            return True
    return False


def packing_fault(model, data, time):
    """Return a failure reason; never modify the simulated configuration."""
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return "non-finite physics state"
    box = data.body("carton")
    if box.xpos[2] < -0.02 or np.max(np.abs(box.xpos[:2])) > 0.65:
        return "carton left the work surface"
    if box.xmat.reshape(3, 3)[2, 2] < 0.90:
        return "carton tipped; remaining packing motion is unsafe"
    if time >= 12:
        lid = float(data.qpos[model.joint("crease/lid").qposadr[0]])
        if abs(lid - np.pi / 2) > 0.18:
            return "main lid is not seated; cannot start or continue locking-ear insertion"
    return None
