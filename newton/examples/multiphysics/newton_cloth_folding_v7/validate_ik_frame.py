"""Regression test for the task-space frame used by cloth_franka keyframes.

Run from a Newton checkout:
    uv run --extra examples python validate_ik_frame.py

Optional local URDF override:
    python validate_ik_frame.py --urdf path/to/fr3_franka_hand.urdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.utils

BASE = wp.transform(wp.vec3(-0.5, -0.5, 0.0), wp.quat_identity())
Q = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.0, 0.032, 0.032]


def find_suffix(labels: list[str], suffix: str) -> int:
    return next(i for i, label in enumerate(labels) if label.endswith(suffix))


def as_transform(row: np.ndarray) -> wp.transform:
    return wp.transform(wp.vec3(*map(float, row[:3])), wp.quat(*map(float, row[3:])))


def transform_row(value: wp.transform) -> np.ndarray:
    return np.asarray([*value.p, *value.q], dtype=np.float64)


def quat_error_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = abs(float(np.dot(q0, q1)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def build(urdf: Path, collapse: bool):
    builder = newton.ModelBuilder()
    builder.add_urdf(
        str(urdf),
        xform=BASE,
        floating=False,
        collapse_fixed_joints=collapse,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )
    builder.joint_q[: len(Q)] = Q
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return builder, model, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()

    urdf = args.urdf
    if urdf is None:
        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"

    old_builder, _, old_state = build(urdf, collapse=True)
    old_link_index = old_builder.body_count - 3
    old_link = as_transform(old_state.body_q.numpy()[old_link_index])
    old_frame = old_link * wp.transform(wp.vec3(0.0, 0.0, 0.22), wp.quat_identity())

    new_builder, _, new_state = build(urdf, collapse=False)
    body_q = new_state.body_q.numpy()
    link7 = as_transform(body_q[find_suffix(new_builder.body_label, "fr3_link7")])
    corrected_frame = link7 * wp.transform(wp.vec3(0.0, 0.0, 0.22), wp.quat_identity())
    hand_tcp = as_transform(body_q[find_suffix(new_builder.body_label, "fr3_hand_tcp")])

    old_row = transform_row(old_frame)
    corrected_row = transform_row(corrected_frame)
    tcp_row = transform_row(hand_tcp)
    corrected_pos_error = float(np.linalg.norm(corrected_row[:3] - old_row[:3]))
    corrected_rot_error = quat_error_deg(corrected_row[3:], old_row[3:])
    tcp_pos_error = float(np.linalg.norm(tcp_row[:3] - old_row[:3]))
    tcp_rot_error = quat_error_deg(tcp_row[3:], old_row[3:])

    print(f"URDF: {urdf}")
    print(f"legacy collapsed body: {old_builder.body_label[old_link_index]}")
    print(f"corrected link7+22cm error: position={corrected_pos_error:.9f} m, rotation={corrected_rot_error:.9f} deg")
    print(f"direct hand_tcp mismatch: position={tcp_pos_error:.6f} m, rotation={tcp_rot_error:.6f} deg")

    assert corrected_pos_error < 1.0e-6
    assert corrected_rot_error < 1.0e-4
    assert tcp_pos_error > 0.005
    assert tcp_rot_error > 30.0
    print("PASS: copied cloth_franka keyframes must target fr3_link7 + 0.22 m, not fr3_hand_tcp.")


if __name__ == "__main__":
    main()
