"""Official Unitree G1 with a floating pelvis and a compliant landing posture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

G1_URDF = Path(__file__).resolve().parents[2] / "assets" / "g1_jelly_drop" / "g1_29dof_rev_1_0.urdf"


def measure_joint_errors(model, body_q: np.ndarray, robot) -> dict:
    """Measure physical joint coherence from body poses; intended for occasional test diagnostics.

    SynReal may not rewrite generalized joint coordinates after a step, so using ``state.joint_q``
    would miss disconnected limbs. Free-root motion is deliberately excluded from these constraints.
    """
    children = model.joint_child.numpy()
    joints = np.flatnonzero(
        (model.joint_type.numpy() == int(newton.JointType.REVOLUTE)) & np.isin(children, robot.body_indices)
    )
    parents, children = model.joint_parent.numpy()[joints], children[joints]
    parent_frames, child_frames = model.joint_X_p.numpy()[joints], model.joint_X_c.numpy()[joints]

    def rotate(quaternion, vectors):
        xyz, scalar = quaternion[:, :3], quaternion[:, 3:4]
        return vectors + 2.0 * np.cross(xyz, np.cross(xyz, vectors) + scalar * vectors)

    parent_poses = body_q[parents].copy()
    # Revolute-to-world joints are supported even though the official G1 has only a free root.
    parent_poses[parents < 0] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    child_poses = body_q[children]
    parent_anchors = parent_poses[:, :3] + rotate(parent_poses[:, 3:], parent_frames[:, :3])
    child_anchors = child_poses[:, :3] + rotate(child_poses[:, 3:], child_frames[:, :3])
    gaps = np.linalg.norm(parent_anchors - child_anchors, axis=1)
    axes = model.joint_axis.numpy()[model.joint_qd_start.numpy()[joints]]
    parent_axes = rotate(parent_poses[:, 3:], rotate(parent_frames[:, 3:], axes))
    child_axes = rotate(child_poses[:, 3:], rotate(child_frames[:, 3:], axes))
    lengths = np.linalg.norm(parent_axes, axis=1) * np.linalg.norm(child_axes, axis=1)
    cosine = np.sum(parent_axes * child_axes, axis=1) / np.maximum(lengths, 1.0e-12)
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    worst = int(np.argmax(gaps))
    return {
        "max_anchor_separation": float(gaps[worst]),
        "max_hinge_axis_angle": float(np.max(angles)),
        "worst_joint": model.joint_label[int(joints[worst])],
    }


@dataclass(frozen=True)
class G1Robot:
    root_body: int
    root_coord_start: int
    body_indices: np.ndarray
    joint_dofs: np.ndarray
    joint_coords: np.ndarray
    target_indices: np.ndarray
    home_positions: np.ndarray
    total_mass: float
    asset_path: Path
    local_bounds: np.ndarray
    center_of_mass_offset: np.ndarray
    body_local_poses: np.ndarray
    motor_stiffness: np.ndarray
    motor_damping: np.ndarray


def _collision_bounds(robot, tilt_radians):
    """Measure the tilted posture analytically from primitive support extents, relative to the pelvis."""
    model = robot.finalize(device="cpu")
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    body_poses = model.body_q.numpy()
    tilt = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt_radians))
    poses = [wp.transform_multiply(tilt, wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))) for pose in body_poses]
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    for shape, flags in enumerate(robot.shape_flags):
        if not flags & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        transform = wp.transform_multiply(poses[robot.shape_body[shape]], robot.shape_transform[shape])
        center = np.asarray(wp.transform_get_translation(transform), dtype=np.float64)
        rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
        scale = np.asarray(robot.shape_scale[shape], dtype=np.float64)
        kind = robot.shape_type[shape]
        if kind == newton.GeoType.BOX:
            extent = np.abs(rotation) @ scale
        elif kind == newton.GeoType.SPHERE:
            extent = np.full(3, scale[0])
        elif kind == newton.GeoType.CYLINDER:
            axis = np.clip(rotation[:, 2], -1.0, 1.0)
            extent = scale[0] * np.sqrt(1.0 - axis * axis) + scale[1] * np.abs(axis)
        else:
            raise ValueError(f"The G1 collider {shape} was not simplified to a supported primitive")
        lower, upper = np.minimum(lower, center - extent), np.maximum(upper, center + extent)
    centers = np.asarray([wp.transform_point(pose, com) for pose, com in zip(poses, robot.body_com, strict=True)])
    center_of_mass = np.average(centers, axis=0, weights=np.asarray(robot.body_mass))
    return np.stack((lower, upper)), center_of_mass, np.asarray([np.asarray(pose) for pose in poses])


def _landing_angle(name: str) -> float:
    side = 1.0 if name.startswith("left_") else -1.0
    if "hip_pitch" in name:
        return -0.32
    if "hip_roll" in name:
        return 0.06 * side
    if "knee" in name:
        return 0.64
    if "ankle_pitch" in name:
        return -0.32
    if "shoulder_pitch" in name:
        return -0.15
    if "shoulder_roll" in name:
        return 0.72 * side
    if "elbow" in name:
        return 0.42
    return 0.0


def _filter_assembled_colliders(robot: newton.ModelBuilder) -> None:
    """Exclude contacts inside the assembled joints while keeping separate limbs collidable."""
    excluded_bodies = {(body, body) for body in range(robot.body_count)}
    # Collapsing fixed links moves their shapes, but Newton retains the old shape filter pairs.
    # Rebuild parent-child exclusions so the merged pelvis contour cannot push its own hip joints apart.
    for parent, child, filter_parent in zip(
        robot.joint_parent, robot.joint_child, robot.joint_collision_filter_parent, strict=True
    ):
        if parent >= 0 and filter_parent:
            excluded_bodies.add(tuple(sorted((parent, child))))

    bodies = {label.rsplit("/", 1)[-1]: body for body, label in enumerate(robot.body_label)}
    # The coarse pelvis and hip-roll boxes overlap across the intervening hip-pitch joints.
    # Exclude these two mechanical assemblies explicitly; distant limbs must still self-collide.
    for side in ("left", "right"):
        excluded_bodies.add(tuple(sorted((bodies["pelvis"], bodies[f"{side}_hip_roll_link"]))))

    colliders = [
        shape for shape, flags in enumerate(robot.shape_flags) if flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
    ]
    existing_pairs = set(robot.shape_collision_filter_pairs)
    for offset, shape_a in enumerate(colliders):
        body_a = robot.shape_body[shape_a]
        for shape_b in colliders[offset + 1 :]:
            body_pair = tuple(sorted((body_a, robot.shape_body[shape_b])))
            if body_pair in excluded_bodies and (shape_a, shape_b) not in existing_pairs:
                robot.add_shape_collision_filter_pair(shape_a, shape_b)


def add_g1(builder: newton.ModelBuilder, position, tilt_radians: float = 0.12) -> G1Robot:
    """Add a real 29-DOF G1 in a bent-knee, open-arm posture with a free six-DOF base.

    ``position`` locates the pelvis frame, and positive tilt pitches the chest forward about world Y.
    The returned ``home_positions`` are already assigned to motor position targets. A caller may reuse
    them at ``target_indices`` in ``control.joint_target_q``; the floating-base coordinates have no motor.
    """
    source = G1_URDF
    if not source.is_file():
        from newton import utils as newton_utils

        source = newton_utils.download_asset("unitree_g1") / "urdf" / "g1_29dof_rev_1_0.urdf"
    # A private builder lets us collapse decorative fixed links without renumbering the caller's objects.
    robot = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=builder.gravity)
    robot.default_shape_cfg = builder.default_shape_cfg.copy()
    robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=12.0, friction=1.0e-5)
    with wp.ScopedDevice("cpu"):
        robot.add_urdf(
            str(source),
            floating=True,
            collapse_fixed_joints=True,
            enable_self_collisions=True,
            ignore_inertial_definitions=False,
        )
        # Visual meshes stay detailed; only collision meshes become oriented boxes.
        robot.approximate_meshes("bounding_box", raise_on_failure=True)
    _filter_assembled_colliders(robot)
    joint_ids = [joint for joint, kind in enumerate(robot.joint_type) if kind == int(newton.JointType.REVOLUTE)]
    if len(joint_ids) != 29:
        raise ValueError(f"Expected the official 29-DOF G1 without active fingers, found {len(joint_ids)} joints")
    dofs, coords, home = [], [], []
    for joint in joint_ids:
        name = robot.joint_label[joint].rsplit("/", 1)[-1]
        dof, coord = robot.joint_qd_start[joint], robot.joint_q_start[joint]
        angle = _landing_angle(name)
        if not robot.joint_limit_lower[dof] <= angle <= robot.joint_limit_upper[dof]:
            raise ValueError(f"Landing angle for {name} violates its URDF joint limit")
        target = coord
        robot.joint_q[coord] = angle
        robot.joint_target_q[target] = angle
        robot.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
        # Small passive-like gains keep the silhouette readable while allowing impact to fold the limbs.
        is_leg = any(part in name for part in ("hip_", "knee", "ankle_"))
        robot.joint_target_ke[dof] = 70.0 if is_leg else 18.0
        robot.joint_target_kd[dof] = 5.0 if is_leg else 1.5
        dofs.append(dof)
        coords.append(coord)
        home.append(angle)
    first_body, first_dof, first_coord = builder.body_count, builder.joint_dof_count, builder.joint_coord_count
    root = next(index for index, label in enumerate(robot.body_label) if label.rsplit("/", 1)[-1] == "pelvis")
    local_bounds, center_of_mass, body_local_poses = _collision_bounds(robot, tilt_radians)
    stiffness = np.asarray(robot.joint_target_ke, dtype=np.float32)[dofs]
    damping = np.asarray(robot.joint_target_kd, dtype=np.float32)[dofs]
    builder.add_builder(
        robot,
        xform=wp.transform(wp.vec3(*position), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt_radians)),
        label_prefix="jelly_g1",
    )
    dofs = first_dof + np.asarray(dofs, dtype=np.int32)
    coords = first_coord + np.asarray(coords, dtype=np.int32)
    return G1Robot(
        root_body=first_body + root,
        root_coord_start=first_coord,
        body_indices=np.arange(first_body, builder.body_count, dtype=np.int32),
        joint_dofs=dofs,
        joint_coords=coords,
        target_indices=coords,
        home_positions=np.asarray(home, dtype=np.float32),
        total_mass=float(sum(robot.body_mass)),
        asset_path=source,
        local_bounds=local_bounds,
        center_of_mass_offset=center_of_mass,
        body_local_poses=body_local_poses,
        motor_stiffness=stiffness,
        motor_damping=damping,
    )
