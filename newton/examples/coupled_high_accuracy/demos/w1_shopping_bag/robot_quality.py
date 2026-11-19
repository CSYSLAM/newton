"""Check cross-arm collider penetration and motion roughness in raw solver states or an IK plan."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import newton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "w1_courier_transport"))
from gripper_collision import enable_cross_arm_collisions, install
from shopping_motion import create_plan
from w1_motion import add_w1


def check(folder=None, plan_only=False):
    if folder is not None:
        data = np.load(folder / "trajectory.npz")
        plan = data["targets"]
    else:
        plan = create_plan()
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    add_w1(b, plan[0])
    install(b)
    enable_cross_arm_collisions(b)
    # This diagnostic isolates the two arms: robot/table and same-arm contacts are outside its scope.
    for shape, body in enumerate(b.shape_body):
        if body < 0:
            b.shape_flags[shape] = 0
    model = b.finalize()
    state = model.state()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=65536)
    contacts = pipeline.contacts()
    if plan_only or folder is None:
        poses = []
        for q in plan[::2]:
            newton.eval_fk(model, wp.array(q, dtype=float), model.joint_qd, state)
            poses.append(state.body_q.numpy())
        poses = np.asarray(poses)
    else:
        poses = data["bodies"][:, : model.body_count]
    worst = 0.0
    worst_pair = None
    contact_frames = 0
    for frame, pose in enumerate(poses):
        state.body_q.assign(pose)
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        if count > 65536:
            raise RuntimeError("Robot collision diagnostic contact capacity exceeded")
        if count == 0:
            continue
        shapes = [a.numpy()[:count] for a in (contacts.rigid_contact_shape0, contacts.rigid_contact_shape1)]
        valid = (shapes[0] >= 0) & (shapes[1] >= 0)
        if not valid.any():
            continue
        ids = [np.asarray(b.shape_body)[s[valid]] for s in shapes]
        local = [a.numpy()[:count][valid] for a in (contacts.rigid_contact_point0, contacts.rigid_contact_point1)]
        world = [Rotation.from_quat(pose[i, 3:]).apply(p) + pose[i, :3] for i, p in zip(ids, local)]
        normals = contacts.rigid_contact_normal.numpy()[:count][valid]
        depth = -np.einsum("ij,ij->i", world[1] - world[0], normals)
        if depth.max() > 0:
            contact_frames += 1
        if depth.max() > worst:
            k = int(depth.argmax())
            worst = float(depth[k])
            worst_pair = dict(time_s=frame / 30, links=[b.body_label[int(i[k])] for i in ids])
    arms = [
        i
        for j, i in enumerate(b.joint_q_start[:-1])
        if b.joint_label[j].rsplit("/", 1)[-1].startswith(("LEFT_J", "RIGHT_J"))
    ]
    speed = np.diff(plan[:, arms], axis=0) * 60
    acceleration = np.diff(speed, axis=0) * 60
    roughness = {}
    for side in ("right", "left"):
        body = next(i for i, n in enumerate(b.body_label) if n.endswith(f"wrist_roll_{side[0]}_j7_link"))
        # Residual from a 300 ms local polynomial separates vibration from the slow commanded motion.
        p = poses[:, body, :3]
        residual = p - savgol_filter(p, 9, 2, axis=0)
        windows = {}
        for name, lo, hi in [
            ("approach", 1, 2.4),
            ("grasp_hold", 2.4, 3.1),
            ("release_hold", 6.4, 7.1),
            ("parked", 9, 9.5),
        ]:
            x = residual[round(lo * 30) : round(hi * 30)]
            rotation = Rotation.from_quat(poses[round(lo * 30) : round(hi * 30), body, 3:])
            angles = (rotation[0].inv() * rotation).as_rotvec()
            angular_residual = angles - savgol_filter(angles, 9, 2, axis=0)
            windows[name] = dict(
                rms_mm=float(np.sqrt(np.mean(x * x)) * 1000),
                max_mm=float(np.linalg.norm(x, axis=1).max() * 1000),
                rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(angular_residual**2)))),
            )
        roughness[side] = windows
    report = dict(
        passed=worst < 0.001,
        max_cross_arm_penetration_mm=worst * 1000,
        penetrating_frames=contact_frames,
        worst_pair=worst_pair,
        max_joint_speed_rad_s=float(abs(speed).max()),
        max_joint_acceleration_rad_s2=float(abs(acceleration).max()),
        wrist_vibration=roughness,
        scope="30 Hz samples of the URDF collision hulls; adjacent-link exclusions retained.",
    )
    if folder is not None:
        name = "robot_plan_quality.json" if plan_only else "robot_quality.json"
        (folder / name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if check(args.folder, args.plan_only)["passed"] else 1)
