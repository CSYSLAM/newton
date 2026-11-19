"""Reference-inspired dual-arm packing targets; object motion is never prescribed."""

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
from w1_motion import TCP, coordinates, fixed_stance_asset, smooth

CAN_POSITIONS = np.array([[0.48, -0.30, 0.906], [0.48, -0.42, 0.906], [0.48, -0.18, 0.906]])
LEFT_GRIP = np.array([0.5, 0.198, 1.176])


def targets(t, items=1):
    left = LEFT_GRIP.copy()
    if items == 1:
        left[2] += 0.20 * smooth((t - 9.5) / 2) * (1 - smooth((t - 13.5) / 2))
    else:
        left[2] += 0.12 * smooth((t - 26) / 2) * (1 - smooth((t - 31) / 2))
    index = min(int(max(t - 1, 0) // 8), items - 1)
    u = np.clip(t - 1 - index * 8, 0, 8)
    pick = CAN_POSITIONS[index].copy()
    following = CAN_POSITIONS[min(index + 1, items - 1)].copy()
    following[2] = 1.35
    high = pick.copy()
    high[2] = 1.35
    mouth = np.array([0.54, 0.065, 1.30])
    keys = [
        (0, high, 0.048),
        (0.6, pick + [0, 0, 0.17], 0.048),
        (1.4, pick, 0.048),
        (2.1, pick, 0.0),
        (3.2, high, 0.0),
        (4.2, mouth, 0.0),
        (4.9, mouth - [0, 0, 0.065], 0.0),
        (5.4, mouth - [0, 0, 0.065], 0.048),
        (6.2, mouth - [0, 0, 0.065], 0.048),
        (6.6, mouth, 0.048),
        (7.4, following, 0.048),
        (8, following, 0.048),
    ]
    for a, b in zip(keys, keys[1:]):
        if u <= b[0]:
            blend = smooth((u - a[0]) / (b[0] - a[0]))
            right = (1 - blend) * a[1] + blend * b[1]
            opening = (1 - blend) * a[2] + blend * b[2]
            break
    if t >= 1 + 8 * items:
        right = CAN_POSITIONS[items - 1].copy()
        right[2] = 1.35
        opening = 0.048
    right_pitch = 120 + 25 * smooth((u - 3.2) / 1.0) * (1 - smooth((u - 6.6) / 0.8))
    quat = Rotation.from_euler("y", [[120], [right_pitch]], degrees=True).as_quat()
    return np.array([left, right]), [0.0, opening], quat


def create_plan(duration=16, fps=60, items=1):
    signature = hashlib.sha256(Path(__file__).read_bytes() + f"{duration}:{fps}:{items}".encode()).hexdigest()[:16]
    cached = Path(__file__).resolve().parents[2] / "build" / f"shopping-plan-{signature}.npy"
    if cached.is_file():
        print("Reusing checked IK plan:", cached.name, flush=True)
        return np.load(cached)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_urdf(
        str(fixed_stance_asset()), floating=False, enable_self_collisions=False, collapse_fixed_joints=False
    )
    coords = coordinates(builder)
    for side, sign in [("LEFT", -1), ("RIGHT", 1)]:
        builder.joint_q[coords[side + "_J2"]] = sign * 0.9
        builder.joint_q[coords[side + "_J4"]] = sign
    model = builder.finalize()
    ee = [
        next(i for i, n in enumerate(model.body_label) if n.endswith("/" + side + "_gripper_base"))
        for side in ("left", "right")
    ]
    p, _, quat = targets(0, items)
    positions = [ik.IKObjectivePosition(b, TCP, wp.array([wp.vec3(*v)], dtype=wp.vec3)) for b, v in zip(ee, p)]
    rotations = [
        ik.IKObjectiveRotation(b, wp.quat_identity(), wp.array([wp.vec4(*q)], dtype=wp.vec4)) for b, q in zip(ee, quat)
    ]
    # Equal bounds form a quadratic posture cost. Updating them from the previous solution
    # prevents the redundant wrists from changing IK branch near a singular configuration.
    reference = wp.clone(model.joint_q)
    continuity = ik.IKObjectiveJointLimit(reference, reference, weight=0.04)
    mask = np.zeros(model.joint_dof_count, dtype=bool)
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
    errors = []
    state = model.state()
    for frame in range(round(duration * fps) + 1):
        continuity.weight = 0.0 if frame == 0 else 0.04
        reference.assign(q.numpy()[0])
        p, opening, quat = targets(frame / fps, items)
        for objective, v in zip(positions, p):
            objective.set_target_position(0, wp.vec3(*v))
        for objective, rotation in zip(rotations, quat):
            objective.target_rotations.assign(np.array([rotation], np.float32))
        solver.step(q, q, iterations=150 if frame == 0 else 12)
        values = q.numpy()[0]
        for side, finger in zip(("LEFT", "RIGHT"), opening):
            for j in (1, 2):
                values[coords[f"{side}_FINGER{j}_JOINT"]] = finger
        q.assign(values.reshape(1, -1))
        plan.append(values.copy())
        if frame % 60 == 0:
            newton.eval_fk(model, q.flatten(), model.joint_qd, state)
            pose = state.body_q.numpy()
            actual = np.array([wp.transform_point(wp.transform(*pose[b]), TCP) for b in ee])
            errors.append(float(np.linalg.norm(actual - p, axis=1).max()))
            print("IK", frame / fps, np.linalg.norm(actual - p, axis=1).round(4).tolist(), flush=True)
    print("Maximum sampled IK position error:", max(errors), flush=True)
    if max(errors) > 0.012:
        raise RuntimeError("Packing trajectory is outside W1 reach")
    plan = np.asarray(plan, np.float32)
    # Smooth the commands only; the renderer always uses the unfiltered physical states.
    # A short symmetric fit suppresses small IK null-space oscillations without shifting grasp timing.
    arms = np.flatnonzero(mask)
    plan[:, arms] = savgol_filter(plan[:, arms], 21, 3, axis=0, mode="interp")
    for frame in range(len(plan)):
        newton.eval_fk(model, wp.array(plan[frame], dtype=float), model.joint_qd, state)
        pose = state.body_q.numpy()
        actual = np.array([wp.transform_point(wp.transform(*pose[b]), TCP) for b in ee])
        if np.linalg.norm(actual - targets(frame / fps, items)[0], axis=1).max() > 0.012:
            raise RuntimeError("Smoothed packing trajectory exceeds tracking tolerance")
    np.save(cached, plan)
    return plan
