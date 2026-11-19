"""W1 dual-arm targets; both mailers remain free dynamic objects."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import newton
import newton.ik as ik

ASSET = Path(__file__).resolve().parents[2] / "assets/w1/robot/w1-030/robot.urdf"
TABLE_Z = 0.86
TCP = wp.vec3(0, 0, 0.108)
# Keep the complete finger hull, including contact thickness, outside the payload's near face at x=0.435.
# The film settles below its initial seal height; the pads must overlap that settled shoulder.
GRASPS = np.array(((0.390, 0.27, 0.896), (0.390, -0.27, 0.896)))
CENTERS = np.array(((0.535, 0.27, 0.882), (0.535, -0.27, 0.882)))
ROTATIONS = [Rotation.from_euler("z", -90, degrees=True).as_matrix()] * 2
APPROACH_END, CLOSE_END, LIFT_START, LIFT_END = 3.0, 4.0, 4.5, 7.0
# Move back into the original carry workspace after pinching the seal outside the payload.
# Moving the entire shake/retreat path back with the grasp drives the wrists close to their IK limits.
CARRY_OFFSET = np.array((0.020, 0.0, 0.344))
FINGER_PAD_CENTER = (0.0, -0.0285, -0.0015)
FINGER_PAD_HALF_EXTENTS = (0.014, 0.0315, 0.0015)


def fixed_stance_asset():
    """Lock unused stance/head joints at their commanded zero pose, preserving both dynamic arms."""
    tree = ET.parse(ASSET)
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in ("ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2"):
            joint.set("type", "fixed")
    for mesh in tree.getroot().iter("mesh"):
        mesh.set("filename", str((ASSET.parent / mesh.get("filename")).resolve()))
    path = Path(__file__).resolve().parents[2] / "build/w1_fixed_stance.urdf"
    path.parent.mkdir(exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def smooth(t):
    u = np.clip(t, 0, 1)
    return u**3 * (10 + u * (-15 + 6 * u))


def target(t):
    lift = GRASPS + CARRY_OFFSET
    # The finite-effort jaw drives need a stationary interval to compress the gathered film before lift-off.
    keys = [
        (0, GRASPS + (-0.05, 0, 0.15), 0.045),
        (1, GRASPS + (-0.05, 0, 0.15), 0.045),
        (APPROACH_END, GRASPS, 0.045),
        (CLOSE_END, GRASPS, 0),
        (LIFT_START, GRASPS, 0),
        (LIFT_END, lift, 0),
        (10, lift, 0),
        (10.8, lift + (-0.08, 0, 0.03), 0.045),
        (15, lift + (-0.08, 0, 0.03), 0.045),
    ]
    a, b = keys[-2:]
    for first, second in zip(keys, keys[1:]):
        if t <= second[0]:
            a, b = first, second
            break
    u = smooth((t - a[0]) / (b[0] - a[0]))
    point = (1 - u) * a[1] + u * b[1]
    finger = (1 - u) * a[2] + u * b[2]
    if t >= 10:
        # Release promptly, but let the unloaded arms retreat over a longer, smooth interval.
        finger = 0.045 * smooth((t - 10) / 0.2)
    pitch = 120.0
    if 7 <= t <= 10:
        u = (t - 7) / 3
        wave = np.sin(6 * np.pi * u) * smooth(u / 0.15) * smooth((1 - u) / 0.15)
        point = point + (0.04 * wave, 0, 0)
        pitch += 12 * wave
    return point, finger, Rotation.from_euler("y", pitch, degrees=True).as_quat()


def coordinates(builder):
    return {
        label.rsplit("/", 1)[-1]: builder.joint_q_start[j]
        for j, label in enumerate(builder.joint_label)
        if builder.joint_type[j] != newton.JointType.FIXED
    }


def create_plan(fps=60, duration=15):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_urdf(
        str(fixed_stance_asset()), floating=False, enable_self_collisions=False, collapse_fixed_joints=False
    )
    coords = coordinates(builder)
    for side, sign in [("LEFT", -1), ("RIGHT", 1)]:
        builder.joint_q[coords[side + "_J2"]] = sign * 0.9
        builder.joint_q[coords[side + "_J4"]] = sign * 1.0
    model = builder.finalize()
    ee = [
        next(i for i, name in enumerate(model.body_label) if name.endswith("/" + side + "_gripper_base"))
        for side in ["left", "right"]
    ]
    p, _, quat = target(0)
    positions = [ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*v)], dtype=wp.vec3)) for body, v in zip(ee, p)]
    rotations = [
        ik.IKObjectiveRotation(body, wp.quat_identity(), wp.array([wp.vec4(*quat)], dtype=wp.vec4)) for body in ee
    ]
    mask = np.zeros(model.joint_dof_count, dtype=bool)
    for side in ["LEFT", "RIGHT"]:
        for j in range(1, 8):
            mask[coords[f"{side}_J{j}"]] = True
    solver = ik.IKSolver(
        model,
        1,
        positions
        + rotations
        + [ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=10.0)],
        joint_dof_mask=wp.array(mask, dtype=wp.bool),
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
        lambda_initial=0.03,
    )
    q = wp.clone(model.joint_q).reshape((1, -1))
    plan, errors = [], []
    state = model.state()
    for frame in range(round(duration * fps) + 1):
        p, finger, quat = target(frame / fps)
        for objective, v in zip(positions, p):
            objective.set_target_position(0, wp.vec3(*v))
        for objective in rotations:
            objective.target_rotations.assign(np.array([quat], dtype=np.float32))
        solver.step(q, q, iterations=150 if frame == 0 else 12)
        values = q.numpy()[0]
        for side in ["LEFT", "RIGHT"]:
            for j in [1, 2]:
                values[coords[f"{side}_FINGER{j}_JOINT"]] = finger
        q.assign(values.reshape(1, -1))
        plan.append(values.copy())
        if frame % 60 == 0:
            newton.eval_fk(model, q.flatten(), model.joint_qd, state)
            pose = state.body_q.numpy()
            actual = np.array([wp.transform_point(wp.transform(*pose[b]), TCP) for b in ee])
            errors.append(float(np.max(np.linalg.norm(actual - p, axis=1))))
    print("W1 IK maximum sampled position error:", max(errors), flush=True)
    if max(errors) > 0.008:
        raise RuntimeError("W1 trajectory exceeds IK tracking tolerance")
    return np.asarray(plan, dtype=np.float32)


def add_w1(builder, initial, *, finger_pads=False):
    first_robot_shape = builder.shape_count
    builder.add_urdf(
        str(fixed_stance_asset()), floating=False, enable_self_collisions=False, collapse_fixed_joints=True
    )
    # ABD requires positive volume moments; some URDF links specify degenerate rotational inertia.
    for body, inertia in enumerate(builder.body_inertia):
        tensor = np.asarray(inertia, dtype=float).reshape(3, 3)
        moment = 0.5 * np.trace(tensor) * np.eye(3) - tensor
        values, axes = np.linalg.eigh(moment)
        floor = max(float(builder.body_mass[body]) * 1e-6, 1e-8)
        if values.min() < floor:
            moment = (axes * np.maximum(values, floor)) @ axes.T
            builder.body_inertia[body] = wp.mat33(np.trace(moment) * np.eye(3) - moment)
    joints = [j for j in range(builder.joint_count) if builder.joint_type[j] != newton.JointType.FIXED]
    for joint in joints:
        coord, dof = builder.joint_q_start[joint], builder.joint_qd_start[joint]
        target_index = coord
        builder.joint_q[coord] = initial[coord]
        builder.joint_target_q[target_index] = initial[coord]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_target_ke[dof] = 50000 if "FINGER" in builder.joint_label[joint] else 12000
        builder.joint_target_kd[dof] = 100 if "FINGER" in builder.joint_label[joint] else 350
    for shape, source in enumerate(builder.shape_source):
        builder.shape_material_mu[shape] = 4.0
        if source is not None and builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES:
            hull = source.compute_convex_hull()
            hull.build_sdf(
                max_resolution=48,
                narrow_band_range=(-0.02, 0.02),
                cache_dir=Path(__file__).resolve().parents[2] / "build/w1-sdf",
            )
            builder.shape_source[shape] = hull
            builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    if finger_pads:
        add_finger_pads(builder, first_robot_shape)
    table = builder.default_shape_cfg.copy()
    table.mu = 0.5
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.48, 0, TABLE_Z - 0.025), wp.quat_identity()),
        hx=0.36,
        hy=0.73,
        hz=0.025,
        cfg=table,
        color=(0.33, 0.36, 0.4),
        label="worktop",
    )
    for x in [0.18, 0.78]:
        for y in [-0.66, 0.66]:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(x, y, (TABLE_Z - 0.05) / 2), wp.quat_identity()),
                hx=0.025,
                hy=0.025,
                hz=(TABLE_Z - 0.05) / 2,
                cfg=table,
                color=(0.15, 0.17, 0.20),
                label="table_leg",
            )
    return joints, next(i for i, name in enumerate(builder.body_label) if name.endswith("/wrist_roll_l_j7_link"))


def add_finger_pads(builder, first_robot_shape):
    pad = builder.default_shape_cfg.copy()
    # The URDF mass already represents the complete finger. Flat rubber contact proxies avoid relying on
    # the tapered metal hull's rim to retain a few gathered film vertices; all loads still pass through contact.
    pad.density, pad.mu = 0.0, 4.0
    robot_shapes = list(range(first_robot_shape, builder.shape_count))
    for body, label in enumerate(builder.body_label):
        if label.endswith(("/left_finger1", "/left_finger2", "/right_finger1", "/right_finger2")):
            shape = builder.add_shape_box(
                body=body,
                xform=wp.transform(wp.vec3(*FINGER_PAD_CENTER), wp.quat_identity()),
                hx=FINGER_PAD_HALF_EXTENTS[0],
                hy=FINGER_PAD_HALF_EXTENTS[1],
                hz=FINGER_PAD_HALF_EXTENTS[2],
                cfg=pad,
                color=(0.025, 0.025, 0.025),
                label=label.rsplit("/", 1)[-1] + "_grip_pad",
            )
            # Pads added after URDF import must inherit its disabled robot self-collisions.
            for other in robot_shapes:
                builder.add_shape_collision_filter_pair(other, shape)
            robot_shapes.append(shape)
