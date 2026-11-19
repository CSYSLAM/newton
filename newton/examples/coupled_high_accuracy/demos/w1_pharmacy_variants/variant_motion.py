"""Continuous W1 targets, adjusted independently for each carton's top and palm clearance."""

import hashlib
from pathlib import Path

import numpy as np
import warp as wp
from catalog import DURATION, centers
from common import ROOT
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from w1_motion import TCP, coordinates, fixed_stance_asset, smooth

import newton
import newton.ik as ik


def targets(t, cases):
    pick = np.array(centers(cases))
    pick[:, 2] += [c.size[2] / 2 - 0.020 for c in cases]
    hover = pick + [0, 0, 0.17]
    lift = pick + [0, 0, 0.20]
    keys = [
        (0, hover, 0.048),
        (0.5, hover, 0.048),
        (1.8, pick, 0.048),
        (3, pick, 0.0),
        (4.2, lift, 0.0),
        (6.5, lift, 0.0),
        (8, pick, 0.0),
        (9, pick, 0.048),
        (10.2, hover, 0.048),
        (DURATION, hover, 0.048),
    ]
    a, b = keys[-2:]
    for first, second in zip(keys, keys[1:]):
        if t <= second[0]:
            a, b = first, second
            break
    blend = smooth((t - a[0]) / (b[0] - a[0]))
    return (
        (1 - blend) * a[1] + blend * b[1],
        (1 - blend) * a[2] + blend * b[2],
        Rotation.from_euler("y", 170, degrees=True).as_quat(),
    )


def create_plan(cases):
    signature = hashlib.sha256(Path(__file__).read_bytes() + np.array(centers(cases)).tobytes()).hexdigest()[:16]
    cache = ROOT / "build" / f"pharmacy-variants-plan-{signature}.npy"
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
    p, _, quat = targets(0, cases)
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
        + [ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=10.0), continuity],
        joint_dof_mask=wp.array(mask, dtype=wp.bool),
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
        lambda_initial=0.03,
    )
    q = wp.clone(model.joint_q).reshape((1, -1))
    plan = []
    for frame in range(round(DURATION * 60) + 1):
        p, opening, quat = targets(frame / 60, cases)
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
    plan[:, mask] = savgol_filter(plan[:, mask], 21, 3, axis=0)
    state = model.state()
    error = 0.0
    for frame in range(0, len(plan), 15):
        newton.eval_fk(model, wp.array(plan[frame], dtype=float), model.joint_qd, state)
        poses = state.body_q.numpy()
        actual = np.array([wp.transform_point(wp.transform(*poses[body]), TCP) for body in ee])
        error = max(error, float(np.linalg.norm(actual - targets(frame / 60, cases)[0], axis=1).max()))
    print("Variant IK max error:", error, flush=True)
    if error > 0.008:
        raise RuntimeError("Variant grasp is outside safe W1 reach")
    np.save(cache, plan)
    return plan
