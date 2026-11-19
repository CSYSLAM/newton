"""W1 robot, basket proxy and dual-arm targets: the left hand holds the basket, the right hand drops the chips in."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from snack_geometry import BAG_LENGTH, BAG_THICKNESS

import newton
import newton.ik as ik

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/w1_snack_basket"
ROBOT = ASSETS / "robot/robot.urdf"
BASKET_USD = ASSETS / "basket/model_basket_7.usdc"
SNACK_TEXTURE = ASSETS / "snack/resource/img/57dd04deef32c53a6a1c3dc3db516907.png"

TABLE_Z = 0.86
TCP = wp.vec3(0, 0, 0.108)
FINGER_PAD_CENTER = (0.0, -0.0285, -0.0015)
FINGER_PAD_HALF_EXTENTS = (0.014, 0.0315, 0.0015)

# The supplied basket is authored about 2.8 times real size; 0.36 gives a 0.44 x 0.31 x 0.19 m shopping basket.
BASKET_SCALE = 0.36
# Asset x (long side) maps to world y and the handle bars (asset y) run along world x, across the jaw axis.
BASKET_ROTATION = Rotation.from_euler("z", 90, degrees=True)
BASKET_CENTER = np.array((0.56, 0.17, TABLE_Z + 0.002))
# Measured from the source mesh, already scaled: wire floor top, floor and rim footprints, rim height, handle bars.
BASKET_FLOOR_TOP = 0.0043
BASKET_FLOOR_HALF = (0.187, 0.137)
BASKET_RIM_HALF = (0.218, 0.156)
BASKET_RIM_Z = 0.187
HANDLE_BAR_HALF = (0.0088, 0.12, 0.003)
HANDLE_BAR_Z = 0.3087
BASKET_MASS = 0.45

# Chips bag lies flat on the right with its top seal facing the robot.
BAG_ROTATION = Rotation.from_euler("z", 90, degrees=True)
BAG_CENTER = np.array((0.56, -0.24, TABLE_Z + BAG_THICKNESS / 2 + 0.0025))

PITCH = 120.0
BASKET_LIFT = 0.05
# Pinch the bars directly above the centre of mass: a jaw pinch barely resists torque about its own closing axis.
# The pad contact on the bars lies about 18 mm ahead of the TCP along the tilted tool axis.
HANDLE_GRASP = np.array((BASKET_CENTER[0] - 0.018, BASKET_CENTER[1], BASKET_CENTER[2] + HANDLE_BAR_Z))
# The fingers gather only the flat crimp seal, 5 mm outside the bag edge as in the courier grasp.
BAG_GRASP = np.array((BAG_CENTER[0] - BAG_LENGTH / 2 - 0.005, BAG_CENTER[1], BAG_CENTER[2] + 0.005))
# A gathered, pinched seal holds the inflated bag out along the jaws: it spans about 0.19 m ahead of the fingers and
# only 0.07 m below them. Place that span over the basket half nearest the right arm, clear of the handle bars and
# the left fingers, then lower it just below the rim before opening.
DROP_ABOVE = np.array(
    (BASKET_CENTER[0] - 0.125, BASKET_CENTER[1] - 0.115, BASKET_CENTER[2] + BASKET_LIFT + HANDLE_BAR_Z + 0.04)
)
DROP_RELEASE = DROP_ABOVE + (0, 0, -0.12)
OPEN = 0.045
# Each finger drive is limited to 10 N by the URDF. A stiff drive far past the bars saturates both fingers at equal
# force, so the clamped pair has no lateral centring and slides along the jaw axis. A soft drive whose target is
# only 3 mm inside the bars (about 10.9 mm per finger) gives the same squeeze while staying in its linear range.
BASKET_CLOSE = 0.008
BASKET_FINGER_KE, BASKET_FINGER_KD = 3300.0, 20.0
BAG_CLOSE = 0.0

# Phase times in seconds.
APPROACH_START, APPROACH_END, CLOSE_END, LIFT_START = 1.0, 3.0, 4.0, 4.5
BASKET_LIFT_END = 6.0
BAG_LIFT_END, BAG_CARRY_END, BAG_LOWER_END = 6.0, 8.2, 9.4
BAG_RELEASE_START, BAG_RELEASE_END = 9.8, 10.0
BAG_RETREAT_END = 11.6
BASKET_LOWER_START, BASKET_LOWER_END = 12.0, 13.4
BASKET_RELEASE_START, BASKET_RELEASE_END = 13.6, 13.8
BASKET_RETREAT_END = 15.0
DURATION = 16.0


def fixed_stance_asset():
    """Lock unused stance/head joints at zero; both arms and their four fingers stay dynamic."""
    tree = ET.parse(ROBOT)
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in ("ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2"):
            joint.set("type", "fixed")
    for mesh in tree.getroot().iter("mesh"):
        mesh.set("filename", str((ROBOT.parent / mesh.get("filename")).resolve()))
    path = ROOT / "build/w1_snack_basket_fixed_stance.urdf"
    path.parent.mkdir(exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def smooth(t):
    u = np.clip(t, 0, 1)
    return u**3 * (10 + u * (-15 + 6 * u))


def keyframes(keys, t):
    """Quintic interpolation of (time, point, finger) keys."""
    a, b = keys[-2:]
    for first, second in zip(keys, keys[1:]):
        if t <= second[0]:
            a, b = first, second
            break
    u = smooth((t - a[0]) / max(b[0] - a[0], 1e-9))
    return (1 - u) * np.asarray(a[1]) + u * np.asarray(b[1]), (1 - u) * a[2] + u * b[2]


def left_target(t):
    above = HANDLE_GRASP + (-0.06, 0, 0.10)
    lifted = HANDLE_GRASP + (0, 0, BASKET_LIFT)
    keys = [
        (0, above, OPEN),
        (APPROACH_START, above, OPEN),
        (APPROACH_END, HANDLE_GRASP, OPEN),
        (CLOSE_END, HANDLE_GRASP, BASKET_CLOSE),
        (LIFT_START, HANDLE_GRASP, BASKET_CLOSE),
        (BASKET_LIFT_END, lifted, BASKET_CLOSE),
        (BASKET_LOWER_START, lifted, BASKET_CLOSE),
        (BASKET_LOWER_END, HANDLE_GRASP, BASKET_CLOSE),
        (BASKET_RELEASE_START, HANDLE_GRASP, BASKET_CLOSE),
        (BASKET_RELEASE_END, HANDLE_GRASP, OPEN),
        (BASKET_RETREAT_END, above, OPEN),
        (DURATION, above, OPEN),
    ]
    return keyframes(keys, t)


def right_target(t):
    above = BAG_GRASP + (-0.05, 0, 0.15)
    lifted = BAG_GRASP + (-0.02, 0.06, 0.30)
    retreat = DROP_ABOVE + (-0.10, -0.12, -0.06)
    keys = [
        (0, above, OPEN),
        (APPROACH_START, above, OPEN),
        (APPROACH_END, BAG_GRASP, OPEN),
        (CLOSE_END, BAG_GRASP, BAG_CLOSE),
        (LIFT_START, BAG_GRASP, BAG_CLOSE),
        (BAG_LIFT_END, lifted, BAG_CLOSE),
        (BAG_CARRY_END, DROP_ABOVE, BAG_CLOSE),
        (BAG_LOWER_END, DROP_RELEASE, BAG_CLOSE),
        (BAG_RELEASE_START, DROP_RELEASE, BAG_CLOSE),
        (BAG_RELEASE_END, DROP_RELEASE, OPEN),
        (BAG_RETREAT_END, retreat, OPEN),
        (DURATION, retreat, OPEN),
    ]
    return keyframes(keys, t)


def target(t):
    (left, left_finger), (right, right_finger) = left_target(t), right_target(t)
    quat = Rotation.from_euler("y", PITCH, degrees=True).as_quat()
    return np.array((left, right)), (left_finger, right_finger), quat


def coordinates(builder):
    return {
        label.rsplit("/", 1)[-1]: builder.joint_q_start[j]
        for j, label in enumerate(builder.joint_label)
        if builder.joint_type[j] != newton.JointType.FIXED
    }


def create_plan(fps=60, duration=DURATION):
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
        p, fingers, quat = target(frame / fps)
        for objective, v in zip(positions, p):
            objective.set_target_position(0, wp.vec3(*v))
        for objective in rotations:
            objective.target_rotations.assign(np.array([quat], dtype=np.float32))
        solver.step(q, q, iterations=150 if frame == 0 else 12)
        values = q.numpy()[0]
        for side, finger in zip(["LEFT", "RIGHT"], fingers):
            for j in [1, 2]:
                values[coords[f"{side}_FINGER{j}_JOINT"]] = finger
        q.assign(values.reshape(1, -1))
        plan.append(values.copy())
        if frame % 30 == 0:
            newton.eval_fk(model, q.flatten(), model.joint_qd, state)
            pose = state.body_q.numpy()
            actual = np.array([wp.transform_point(wp.transform(*pose[b]), TCP) for b in ee])
            errors.append(float(np.max(np.linalg.norm(actual - p, axis=1))))
            if errors[-1] > 0.004:
                print(
                    f"  IK error {frame / fps:.1f}s (left, right) mm:",
                    (np.linalg.norm(actual - p, axis=1) * 1000).round(1),
                    flush=True,
                )
    print("W1 IK maximum sampled position error:", max(errors), flush=True)
    if max(errors) > 0.008:
        raise RuntimeError("W1 trajectory exceeds IK tracking tolerance")
    return np.asarray(plan, dtype=np.float32)


def add_w1(builder, initial):
    first_robot_shape = builder.shape_count
    builder.add_urdf(
        str(fixed_stance_asset()), floating=False, enable_self_collisions=False, collapse_fixed_joints=True
    )
    # ABD requires positive volume moments; regularise any degenerate URDF rotational inertia at import only.
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
        label = builder.joint_label[joint]
        builder.joint_target_ke[dof] = 50000 if "FINGER" in label else 12000
        builder.joint_target_kd[dof] = 100 if "FINGER" in label else 350
        if "LEFT_FINGER" in label:
            builder.joint_target_ke[dof], builder.joint_target_kd[dof] = BASKET_FINGER_KE, BASKET_FINGER_KD
    for shape, source in enumerate(builder.shape_source):
        builder.shape_material_mu[shape] = 4.0
        if source is not None and builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES:
            hull = source.compute_convex_hull()
            hull.build_sdf(max_resolution=48, narrow_band_range=(-0.02, 0.02), cache_dir=ROOT / "build/w1-sdf")
            builder.shape_source[shape] = hull
            builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    robot_shapes = add_finger_pads(builder, first_robot_shape)
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
    return joints, robot_shapes


def add_finger_pads(builder, first_robot_shape):
    pad = builder.default_shape_cfg.copy()
    # Flat rubber contact proxies on the inner finger faces; the URDF finger mass already includes them.
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
    return robot_shapes


def basket_proxy_boxes():
    """Box proxies in the basket body frame (scaled asset frame): floor, four tapered walls and the handle bars.

    The visual wire mesh has 800k triangles; the solver only needs the surfaces that the film and the jaws touch.
    """
    thickness = 0.006
    boxes = [((0, 0, BASKET_FLOOR_TOP - thickness / 2), (0, 0, 0), (*BASKET_FLOOR_HALF, thickness / 2))]
    height = BASKET_RIM_Z - BASKET_FLOOR_TOP
    for axis in (0, 1):
        spread = BASKET_RIM_HALF[axis] - BASKET_FLOOR_HALF[axis]
        tilt = np.arctan2(spread, height)
        slant = np.hypot(spread, height) / 2
        other = BASKET_RIM_HALF[1 - axis]
        for sign in (-1, 1):
            # Place the wall's inner face on the wire plane, which runs from the floor edge to the rim edge.
            mid = (BASKET_FLOOR_HALF[axis] + BASKET_RIM_HALF[axis]) / 2 + thickness / 2 * np.cos(tilt)
            centre = np.zeros(3)
            centre[axis], centre[2] = sign * mid, BASKET_FLOOR_TOP + height / 2 - thickness / 2 * np.sin(tilt)
            euler = np.zeros(3)
            # Tilt about the in-plane horizontal axis so the wall leans outward like the wire basket.
            euler[1 - axis] = sign * tilt * (1 if axis == 0 else -1)
            half = [0.0, 0.0, slant]
            half[axis], half[1 - axis] = thickness / 2, other
            boxes.append((tuple(centre), tuple(euler), tuple(half)))
    boxes.append(((0, 0, HANDLE_BAR_Z), (0, 0, 0), HANDLE_BAR_HALF))
    return boxes


def add_basket(builder):
    cfg = builder.default_shape_cfg.copy()
    cfg.mu = 1.5
    boxes = basket_proxy_boxes()
    volume = sum(8 * np.prod(half) for _, _, half in boxes)
    cfg.density = float(BASKET_MASS / volume)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*BASKET_CENTER), wp.quat(*BASKET_ROTATION.as_quat())), label="basket"
    )
    for i, (centre, euler, half) in enumerate(boxes):
        rotation = Rotation.from_euler("xyz", euler).as_quat()
        builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(*map(float, centre)), wp.quat(*map(float, rotation))),
            hx=float(half[0]),
            hy=float(half[1]),
            hz=float(half[2]),
            cfg=cfg,
            color=(0.9, 0.9, 0.9),
            label="basket_handle_bars" if i == len(boxes) - 1 else f"basket_proxy_{i}",
        )
    return body
