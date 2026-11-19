"""Measure the two jaws relative to their moving palm, independently of wrist motion and carton deformation."""

import numpy as np
from scipy.spatial.transform import Rotation


def measure_gripper(data, hold_start, hold_end):
    if "jaw_parents" not in data:
        return dict(passed=False, reason="Missing jaw frames; record again with the coupling telemetry enabled.")
    bodies = data["bodies"]
    metrics = []
    times = np.arange(len(bodies)) / 30
    hold = (times >= hold_start) & (times <= hold_end)
    if hold.sum() < 3:
        return dict(passed=False, reason="The recording does not include the loaded hold window.")
    for i in range(len(data["jaw_parents"])):
        coordinates = []
        for j in (0, 1):
            parent = int(data["jaw_parents"][i, j])
            child = int(data["jaw_children"][i, j])
            parent_frame = data["jaw_parent_frames"][i, j]
            child_frame = data["jaw_child_frames"][i, j]
            local_axis = Rotation.from_quat(parent_frame[3:]).apply(data["jaw_axes"][i, j])
            origin = np.broadcast_to(parent_frame[:3], (len(bodies), 3))
            axis = np.broadcast_to(local_axis, origin.shape)
            if parent >= 0:
                rotation = Rotation.from_quat(bodies[:, parent, 3:])
                origin = rotation.apply(origin) + bodies[:, parent, :3]
                axis = rotation.apply(axis)
            endpoint = (
                Rotation.from_quat(bodies[:, child, 3:]).apply(np.broadcast_to(child_frame[:3], origin.shape))
                + bodies[:, child, :3]
            )
            coordinates.append(np.einsum("ij,ij->i", endpoint - origin, axis))
        offset, scale = data["jaw_coefficients"][i]
        error = coordinates[0] - offset - scale * coordinates[1]
        # W1 has antiparallel sliders with unit mimic ratio; half the error is the jaw-center offset.
        if not np.isclose(scale, 1) or not np.isclose(offset, 0):
            raise ValueError("W1 midpoint diagnostic expects an unshifted unit-ratio mimic pair")
        center = error * 0.5 * 1000
        label = str(data["jaw_labels"][i]) if "jaw_labels" in data else f"pair_{i}"
        side = "left" if "LEFT_" in label else ("right" if "RIGHT_" in label else label)
        item = dict(
            side=side,
            maximum_mimic_error_mm=float(abs(error).max() * 1000),
            maximum_center_offset_mm=float(abs(center).max()),
            loaded_center_sway_mm=float(np.ptp(center[hold])),
            loaded_center_maximum_offset_mm=float(abs(center[hold]).max()),
        )
        item["passed"] = item["maximum_center_offset_mm"] < 0.5 and item["loaded_center_sway_mm"] < 0.5
        metrics.append(item)
    balance_error = None
    if "coupling_efforts" in data:
        passive = data["coupling_efforts"].reshape((-1, 2, 2))
        balance_error = float(abs(passive.sum(axis=2)).max())
    return dict(
        passed=all(m["passed"] for m in metrics) and (balance_error is None or balance_error < 1e-5),
        jaws=metrics,
        passive_generalized_force_sum_error_n=balance_error,
        hold_window_s=[hold_start, hold_end],
        scope="30 Hz raw finger poses relative to their palm; includes slow drift, not only high-frequency vibration.",
    )
