"""Measure actual robot motion after its commanded pose has stopped."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def measure(folder):
    data = np.load(folder / "trajectory.npz")
    body = data["bodies"][-31:, : int(data["carton"])]
    positions = body[:, :, :3]
    rotations = body[:, :, 3:]
    r0 = Rotation.from_quat(rotations[:-1].reshape(-1, 4))
    r1 = Rotation.from_quat(rotations[1:].reshape(-1, 4))
    result = dict(
        max_robot_frame_translation_mm=float(np.linalg.norm(np.diff(positions, axis=0), axis=2).max() * 1000),
        max_robot_last_second_span_mm=float(np.linalg.norm(np.ptp(positions, axis=0), axis=1).max() * 1000),
        max_robot_frame_rotation_degrees=float(np.rad2deg((r0.inv() * r1).magnitude().max())),
        last_second_command_span=float(np.ptp(data["joint_targets"][-61:], axis=0).max()),
    )
    result["passed"] = bool(
        result["max_robot_frame_translation_mm"] < 1
        and result["max_robot_last_second_span_mm"] < 5
        and result["max_robot_frame_rotation_degrees"] < 0.2
    )
    (folder / "settling_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = measure(Path(sys.argv[1]))
    raise SystemExit(0 if result["passed"] else 1)
