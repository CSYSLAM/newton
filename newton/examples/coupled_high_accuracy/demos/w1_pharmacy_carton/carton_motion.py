"""Continuous, separated W1 dual-arm grasping trajectories for the two identical cartons."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import newton
import newton.ik as ik

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "w1_courier_transport"))
from carton_asset import CENTERS
from w1_motion import TCP, coordinates, fixed_stance_asset, smooth


def targets(t):
    pick = CENTERS.copy()
    pick[:, 2] += 0.04
    hover = pick + [0, 0, 0.18]
    lift = pick + [0, 0, 0.22]
    keys = [
        (0, hover, 0.048),
        (1, hover, 0.048),
        (2.5, pick, 0.048),
        (4, pick, 0.0),
        (5.5, lift, 0.0),
        (9, lift, 0.0),
        (11, pick, 0.0),
        (12, pick, 0.048),
        (13.2, hover, 0.048),
        (15, hover, 0.048),
    ]
    a, b = keys[-2:]
    for first, second in zip(keys, keys[1:]):
        if t <= second[0]:
            a, b = first, second
            break
    u = smooth((t - a[0]) / (b[0] - a[0]))
    p = (1 - u) * a[1] + u * b[1]
    opening = (1 - u) * a[2] + u * b[2]
    return p, opening, Rotation.from_euler("y", 170, degrees=True).as_quat()


def create_plan(duration=15, fps=60):
    signature = hashlib.sha256(
        Path(__file__).read_bytes() + CENTERS.tobytes() + f"{duration}:{fps}".encode()
    ).hexdigest()[:16]
    cache = Path(__file__).resolve().parents[2] / "build" / f"pharmacy-plan-{signature}.npy"
    if cache.exists():
        return np.load(cache)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.add_urdf(str(fixed_stance_asset()), floating=False, enable_self_collisions=False, collapse_fixed_joints=False)
    coords = coordinates(b)
    for side, sign in [("LEFT", -1), ("RIGHT", 1)]:
        b.joint_q[coords[side + "_J2"]] = sign * 0.9
        b.joint_q[coords[side + "_J4"]] = sign
    model = b.finalize()
    ee = [
        next(i for i, n in enumerate(model.body_label) if n.endswith("/" + s + "_gripper_base"))
        for s in ("left", "right")
    ]
    p, _, quat = targets(0)
    positions = [ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*v)], dtype=wp.vec3)) for body, v in zip(ee, p)]
    rotations = [
        ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([wp.vec4(*quat)], dtype=wp.vec4)) for body in ee
    ]
    reference = wp.clone(model.joint_q)
    continuity = ik.IKObjectiveJointLimit(reference, reference, weight=0.04)
    mask = np.zeros(model.joint_dof_count, bool)
    for side in ("LEFT", "RIGHT"):
        for j in range(1, 8):
            mask[coords[f"{side}_J{j}"]] = True
    solver = ik.IKSolver(
        model,
        1,
        positions
        + rotations
        + [ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=10), continuity],
        joint_dof_mask=wp.array(mask, dtype=wp.bool),
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
        lambda_initial=0.03,
    )
    q = wp.clone(model.joint_q).reshape((1, -1))
    plan = []
    for frame in range(round(duration * fps) + 1):
        p, opening, quat = targets(frame / fps)
        reference.assign(q.numpy()[0])
        continuity.weight = 0.0 if frame == 0 else 0.04
        for objective, v in zip(positions, p):
            objective.set_target_position(0, wp.vec3(*v))
        for objective in rotations:
            objective.target_rotations.assign(np.array([quat], np.float32))
        solver.step(q, q, iterations=150 if frame == 0 else 12)
        values = q.numpy()[0]
        for side in ("LEFT", "RIGHT"):
            for j in (1, 2):
                values[coords[f"{side}_FINGER{j}_JOINT"]] = opening
        q.assign(values.reshape(1, -1))
        plan.append(values)
    plan = np.asarray(plan, np.float32)
    window = min(21, len(plan) // 2 * 2 - 1)
    if window > 3:
        plan[:, mask] = savgol_filter(plan[:, mask], window, 3, axis=0)
    state = model.state()
    errors = []
    for frame in range(0, len(plan), 15):
        newton.eval_fk(model, wp.array(plan[frame], dtype=float), model.joint_qd, state)
        poses = state.body_q.numpy()
        actual = np.array([wp.transform_point(wp.transform(*poses[body]), TCP) for body in ee])
        errors.append(float(np.linalg.norm(actual - targets(frame / fps)[0], axis=1).max()))
    print("Pharmacy IK maximum error:", max(errors), flush=True)
    if max(errors) > 0.008:
        raise RuntimeError("Carton grasp exceeds W1 reach")
    np.save(cache, plan)
    return plan
