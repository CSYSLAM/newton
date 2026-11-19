#!/usr/bin/env python3
"""Evaluate a SynReal G1 checkpoint in MuJoCo using its training USD asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

import newton

if __package__ in (None, ""):
    from locomotion_g1.config import (
        DEFAULT_JOINT_POS,
        HISTORY_LENGTH,
        JOINT_EFFORT_LIMIT,
        JOINT_KD,
        JOINT_KP,
        JOINT_NAMES,
        POLICY_OBSERVATION_DIM,
        POLICY_TERM_DIMS,
    )
else:
    from .config import (
        DEFAULT_JOINT_POS,
        HISTORY_LENGTH,
        JOINT_EFFORT_LIMIT,
        JOINT_KD,
        JOINT_KP,
        JOINT_NAMES,
        POLICY_OBSERVATION_DIM,
        POLICY_TERM_DIMS,
    )


VISIBLE = 1
COLLIDABLE = 2
ACTION_DIM = len(JOINT_NAMES)
MUJOCO_INTEGRATION_REFINEMENT = 2
LOCAL_UNITREE_RL_LAB_ASSET = Path(__file__).resolve().parent / "assets" / "unitree_g1_29dof" / "g1_29dof_rev_1_0.usd"


@dataclass(frozen=True)
class RunSettings:
    """Physics and policy settings captured beside a training checkpoint."""

    snapshot: Path
    training_dt: float
    training_decimation: int
    dt: float
    decimation: int
    gravity: float
    contact_stiffness: float
    contact_damping: float
    contact_friction_gain: float
    ground_friction: float
    robot_friction: float
    friction_range: tuple[float, float]
    joint_armature: float
    action_scale: float
    initial_height: float
    mesh_approximation: str
    self_collisions: bool
    nonlinear_iterations: int
    actor_hidden_dims: tuple[int, ...]
    activation: str

    @property
    def policy_dt(self) -> float:
        return self.dt * self.decimation


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run_settings(checkpoint: Path, robot_friction: float | None) -> RunSettings:
    """Load the environment snapshot written by the G1 training entry point."""

    snapshot = checkpoint.parent / "environment.json"
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing training snapshot {snapshot}; refusing an unaligned evaluation")
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    expected = {
        "environment": "synreal_g1_locomotion",
        "policy_observation_dim": POLICY_OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
    }
    mismatches = {key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value}
    if mismatches:
        raise ValueError(f"checkpoint environment ABI mismatch: {mismatches}")

    environment = data["environment_config"]
    simulation = environment["simulation"]
    friction_range = tuple(float(value) for value in environment["randomization"]["friction_range"])
    if len(friction_range) != 2:
        raise ValueError("training friction_range must have two values")
    selected_friction = (
        0.5 * (friction_range[0] + friction_range[1]) if robot_friction is None else float(robot_friction)
    )
    policy = data["rsl_rl"]["policy"]
    training_dt = float(simulation["dt"])
    training_decimation = int(simulation["decimation"])
    return RunSettings(
        snapshot=snapshot,
        training_dt=training_dt,
        training_decimation=training_decimation,
        dt=training_dt / MUJOCO_INTEGRATION_REFINEMENT,
        decimation=training_decimation * MUJOCO_INTEGRATION_REFINEMENT,
        gravity=float(simulation["gravity"]),
        contact_stiffness=float(simulation["contact_stiffness"]),
        contact_damping=float(simulation["contact_damping"]),
        contact_friction_gain=float(simulation["contact_friction_gain"]),
        ground_friction=float(simulation["ground_friction"]),
        robot_friction=selected_friction,
        friction_range=(friction_range[0], friction_range[1]),
        joint_armature=float(simulation["joint_armature_approximation"]),
        action_scale=float(environment["action_scale"]),
        initial_height=float(environment["initial_base_height"]),
        mesh_approximation=str(environment["mesh_approximation"]),
        self_collisions=bool(environment["enable_self_collisions"]),
        nonlinear_iterations=int(simulation["nonlinear_iterations"]),
        actor_hidden_dims=tuple(int(value) for value in policy["actor_hidden_dims"]),
        activation=str(policy["activation"]).lower(),
    )


def load_actor(checkpoint: Path, settings: RunSettings) -> torch.nn.Sequential:
    """Recreate and load only the deterministic actor from an RSL-RL checkpoint."""

    activations: dict[str, type[torch.nn.Module]] = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "tanh": torch.nn.Tanh,
    }
    if settings.activation not in activations:
        raise ValueError(f"unsupported actor activation {settings.activation!r}")
    dims = [POLICY_OBSERVATION_DIM, *settings.actor_hidden_dims, ACTION_DIM]
    layers: list[torch.nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(pairwise(dims)):
        layers.append(torch.nn.Linear(input_dim, output_dim))
        if index < len(dims) - 2:
            layers.append(activations[settings.activation]())
    actor = torch.nn.Sequential(*layers)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state.get("model_state_dict", state.get("model", state))
    actor_state = {
        name.removeprefix("actor."): value for name, value in state_dict.items() if name.startswith("actor.")
    }
    if not actor_state:
        raise ValueError(f"checkpoint contains no actor parameters: {checkpoint}")
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    with torch.inference_mode():
        output = actor(torch.zeros((1, POLICY_OBSERVATION_DIM)))
    if output.shape != (1, ACTION_DIM) or not torch.isfinite(output).all():
        raise ValueError("loaded actor does not satisfy the G1 policy ABI")
    return actor


def resolve_training_asset(snapshot: dict[str, Any], override: Path | None) -> tuple[Path, str, bool]:
    """Resolve the asset in the same order as the SynReal G1 environment."""

    candidates: list[tuple[Path, str]] = []
    if override is not None:
        candidates.append((override.expanduser(), "command_line"))
    resolved_asset = snapshot.get("resolved_asset")
    if resolved_asset:
        candidates.append((Path(resolved_asset).expanduser(), "training_resolved_asset"))
    saved_path = snapshot["environment_config"].get("asset_path")
    if saved_path:
        candidates.append((Path(saved_path).expanduser(), "training_snapshot"))
    environment_path = os.environ.get("G1_ASSET_PATH")
    if environment_path:
        candidates.append((Path(environment_path).expanduser(), "G1_ASSET_PATH"))
    candidates.append((LOCAL_UNITREE_RL_LAB_ASSET, "repository_local_derived_usd"))
    for path, source in candidates:
        if path.is_file():
            asset = path.resolve()
            expected_hash = snapshot.get("asset_sha256")
            if expected_hash is not None:
                actual_hash = _sha256(asset)
                if actual_hash != expected_hash:
                    raise ValueError(
                        "G1 asset SHA256 differs from the training snapshot: "
                        f"expected {expected_hash}, got {actual_hash} ({asset})"
                    )
            return asset, source, expected_hash is not None

    fallback = Path(newton.utils.download_asset("unitree_g1")) / "urdf" / "g1_29dof_rev_1_0.urdf"
    if fallback.is_file():
        asset = fallback.resolve()
        expected_hash = snapshot.get("asset_sha256")
        if expected_hash is not None and _sha256(asset) != expected_hash:
            raise ValueError(
                "the training G1 asset is unavailable and the Newton fallback does not match its recorded SHA256"
            )
        return asset, "newton_fallback", expected_hash is not None
    raise FileNotFoundError("could not resolve the G1 training asset")


def load_training_asset(asset: Path, settings: RunSettings) -> Any:
    """Parse the asset with the same options and mesh approximation as training."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=settings.joint_armature,
        limit_ke=1.0e3,
        limit_kd=10.0,
        friction=1.0e-5,
    )
    builder.default_shape_cfg.ke = settings.contact_stiffness
    builder.default_shape_cfg.kd = settings.contact_damping
    builder.default_shape_cfg.kf = settings.contact_friction_gain
    builder.default_shape_cfg.mu = settings.robot_friction
    import_options = {
        "floating": True,
        "collapse_fixed_joints": True,
        "enable_self_collisions": settings.self_collisions,
    }
    xform = wp.transform(wp.vec3(0.0, 0.0, settings.initial_height))
    if asset.suffix.lower() in {".usd", ".usda", ".usdc"}:
        builder.add_usd(
            str(asset),
            xform=xform,
            joint_ordering="dfs",
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
            **import_options,
        )
    elif asset.suffix.lower() == ".urdf":
        builder.default_shape_cfg.density = 0.0
        builder.add_urdf(str(asset), xform=xform, **import_options)
    else:
        raise ValueError(f"unsupported G1 asset type: {asset}")

    joint_by_name = {_basename(label): i for i, label in enumerate(builder.joint_label)}
    missing = sorted(set(JOINT_NAMES) - set(joint_by_name))
    if missing:
        raise ValueError(f"training asset is missing G1 joints: {missing}")
    for name in JOINT_NAMES:
        joint = joint_by_name[name]
        qd = int(builder.joint_qd_start[joint])
        builder.joint_armature[qd] = settings.joint_armature
    for shape in range(len(builder.shape_material_mu)):
        builder.shape_material_ke[shape] = settings.contact_stiffness
        builder.shape_material_kd[shape] = settings.contact_damping
        builder.shape_material_kf[shape] = settings.contact_friction_gain
        builder.shape_material_mu[shape] = settings.robot_friction
    if settings.mesh_approximation:
        builder.approximate_meshes(settings.mesh_approximation)
    return builder.finalize(device="cpu")


def _rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _transform_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rotation_xyzw(transform[3:7])
    matrix[:3, 3] = transform[:3]
    return matrix


def _pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(matrix[:3, :3]).ravel())
    return matrix[:3, 3].copy(), quaternion


def _principal_inertia(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, rotation = np.linalg.eigh(np.asarray(inertia, dtype=np.float64))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(rotation).ravel())
    return values, quaternion


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    sine = math.sin(angle)
    cosine = math.cos(angle)
    rotation = np.asarray(
        (
            (
                cosine + x * x * (1.0 - cosine),
                x * y * (1.0 - cosine) - z * sine,
                x * z * (1.0 - cosine) + y * sine,
            ),
            (
                y * x * (1.0 - cosine) + z * sine,
                cosine + y * y * (1.0 - cosine),
                y * z * (1.0 - cosine) - x * sine,
            ),
            (
                z * x * (1.0 - cosine) - y * sine,
                z * y * (1.0 - cosine) + x * sine,
                cosine + z * z * (1.0 - cosine),
            ),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


def _build_world_transforms(reference: Any, joint_positions: dict[str, float] | None = None) -> list[np.ndarray]:
    parents = reference.joint_parent.numpy()
    children = reference.joint_child.numpy()
    parent_frames = reference.joint_X_p.numpy()
    child_frames = reference.joint_X_c.numpy()
    axes = reference.joint_axis.numpy()
    dof_starts = reference.joint_qd_start.numpy()
    worlds: list[np.ndarray | None] = [None] * int(reference.body_count)
    for joint in range(int(reference.joint_count)):
        child = int(children[joint])
        parent = int(parents[joint])
        if parent < 0:
            worlds[child] = np.eye(4, dtype=np.float64)
            continue
        if worlds[parent] is None:
            raise ValueError("G1 joints are not parent-before-child ordered")
        angle = 0.0
        if joint_positions is not None:
            angle = joint_positions.get(_basename(reference.joint_label[joint]), 0.0)
        motion = _axis_angle_matrix(axes[int(dof_starts[joint])], angle)
        worlds[child] = (
            worlds[parent]
            @ _transform_matrix(parent_frames[joint])
            @ motion
            @ np.linalg.inv(_transform_matrix(child_frames[joint]))
        )
    if any(world is None for world in worlds):
        raise ValueError("G1 asset contains a body disconnected from its joint tree")
    return [world for world in worlds if world is not None]


def _quat_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray((quaternion_xyzw[3], *quaternion_xyzw[:3]), dtype=np.float64)


def _add_shapes(
    mj_spec: mujoco.MjSpec,
    reference: Any,
    bodies: list[Any],
    settings: RunSettings,
) -> list[str]:
    shape_type = reference.shape_type.numpy()
    shape_body = reference.shape_body.numpy()
    shape_scale = reference.shape_scale.numpy()
    shape_transform = reference.shape_transform.numpy()
    shape_flags = reference.shape_flags.numpy()
    shape_rgba = reference.shape_color.numpy()
    geom_names: list[str] = []
    for index, label in enumerate(reference.shape_label):
        body = int(shape_body[index])
        if body < 0:
            continue
        flags = int(shape_flags[index])
        collidable = bool(flags & COLLIDABLE)
        visible = bool(flags & VISIBLE)
        name = f"shape_{index}_{_basename(label)}"
        geom_names.append(name)
        color = np.asarray(shape_rgba[index], dtype=np.float64).reshape(-1)
        if color.size == 3:
            color = np.append(color, 1.0)
        common: dict[str, Any] = {
            "name": name,
            "pos": shape_transform[index, :3],
            "quat": _quat_wxyz(shape_transform[index, 3:7]),
            "contype": 1 if collidable else 0,
            "conaffinity": 1 if collidable else 0,
            "friction": [settings.robot_friction, 0.005, 0.0001],
            "rgba": color if visible else [0.0, 0.0, 0.0, 0.0],
            "group": 0 if collidable else 2,
            "density": 0.0,
        }
        if collidable:
            common["solref"] = [-settings.contact_stiffness, -settings.contact_damping]
        kind = int(shape_type[index])
        scale = shape_scale[index]
        if kind == int(newton.GeoType.SPHERE):
            bodies[body].add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[float(scale[0]), 0.0, 0.0],
                **common,
            )
        elif kind == int(newton.GeoType.CAPSULE):
            bodies[body].add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[float(scale[0]), float(scale[1]), 0.0],
                **common,
            )
        elif kind == int(newton.GeoType.ELLIPSOID):
            bodies[body].add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=scale, **common)
        elif kind == int(newton.GeoType.CYLINDER):
            bodies[body].add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[float(scale[0]), float(scale[1]), 0.0],
                **common,
            )
        elif kind == int(newton.GeoType.BOX):
            bodies[body].add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=scale, **common)
        elif kind in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)):
            source = reference.shape_source[index]
            if source is None:
                raise ValueError(f"G1 shape {label!r} has no mesh source")
            mesh_name = f"mesh_{index}"
            mj_spec.add_mesh(
                name=mesh_name,
                scale=scale,
                uservert=np.asarray(source.vertices, dtype=np.float64).reshape(-1),
                userface=np.asarray(source.indices, dtype=np.int32).reshape(-1),
            )
            bodies[body].add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                size=[1.0, 1.0, 1.0],
                **common,
            )
        else:
            raise NotImplementedError(f"unsupported Newton shape type {kind} for {label!r}")
    return geom_names


def _add_collision_excludes(mj_spec: mujoco.MjSpec, reference: Any) -> int:
    shape_body = reference.shape_body.numpy()
    excluded_bodies: set[tuple[int, int]] = set()
    for shape_a, shape_b in reference.shape_collision_filter_pairs_array():
        body_a = int(shape_body[int(shape_a)])
        body_b = int(shape_body[int(shape_b)])
        if body_a < 0 or body_b < 0 or body_a == body_b:
            continue
        excluded_bodies.add(tuple(sorted((body_a, body_b))))
    for index, (body_a, body_b) in enumerate(sorted(excluded_bodies)):
        mj_spec.add_exclude(
            name=f"newton_filter_{index}",
            bodyname1=_basename(reference.body_label[body_a]),
            bodyname2=_basename(reference.body_label[body_b]),
        )
    return len(excluded_bodies)


def build_mujoco_model(reference: Any, settings: RunSettings) -> tuple[mujoco.MjModel, int]:
    """Construct MuJoCo directly from the Newton model parsed from the training asset."""

    worlds = _build_world_transforms(reference)
    masses = reference.body_mass.numpy()
    centers = reference.body_com.numpy()
    inertias = reference.body_inertia.numpy()
    joint_parent = reference.joint_parent.numpy()
    joint_child = reference.joint_child.numpy()
    joint_child_frame = reference.joint_X_c.numpy()
    joint_axis = reference.joint_axis.numpy()
    joint_qd_start = reference.joint_qd_start.numpy()
    joint_lower = reference.joint_limit_lower.numpy()
    joint_upper = reference.joint_limit_upper.numpy()
    joint_armature = reference.joint_armature.numpy()
    joint_friction = reference.joint_friction.numpy()
    joint_damping = reference.joint_damping.numpy()

    mj_spec = mujoco.MjSpec()
    mj_spec.modelname = "unitree_g1_synreal_training_asset"
    mj_spec.compiler.degree = 0
    mj_spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    mj_spec.option.timestep = settings.dt
    mj_spec.option.gravity = [0.0, 0.0, settings.gravity]
    mj_spec.option.iterations = settings.nonlinear_iterations
    mj_spec.stat.center = [0.0, 0.0, 0.8]
    mj_spec.stat.extent = 2.0
    mj_spec.visual.global_.azimuth = 130.0
    mj_spec.visual.global_.elevation = -16.0

    bodies: list[Any] = [None] * int(reference.body_count)
    for body_index, label in enumerate(reference.body_label):
        inertia, iquat = _principal_inertia(inertias[body_index])
        if body_index == 0:
            parent = mj_spec.worldbody
            relative = np.eye(4, dtype=np.float64)
        else:
            matches = np.flatnonzero(joint_child == body_index)
            if len(matches) != 1:
                raise ValueError(f"body {label!r} does not have exactly one parent joint")
            parent_index = int(joint_parent[int(matches[0])])
            parent = bodies[parent_index]
            relative = np.linalg.inv(worlds[parent_index]) @ worlds[body_index]
        position, quaternion = _pose(relative)
        bodies[body_index] = parent.add_body(
            name=_basename(label),
            pos=position,
            quat=quaternion,
            mass=float(masses[body_index]),
            ipos=centers[body_index],
            iquat=iquat,
            inertia=inertia,
            explicitinertial=1,
        )

    root_joints = np.flatnonzero(joint_parent < 0)
    if len(root_joints) != 1:
        raise ValueError(f"expected one floating root joint, found {len(root_joints)}")
    root_body = int(joint_child[int(root_joints[0])])
    bodies[root_body].add_freejoint(name="floating_base")
    for joint in range(int(reference.joint_count)):
        if int(joint_parent[joint]) < 0:
            continue
        child = int(joint_child[joint])
        dof = int(joint_qd_start[joint])
        child_anchor = joint_child_frame[joint]
        axis_child = _rotation_xyzw(child_anchor[3:7]) @ joint_axis[dof]
        bodies[child].add_joint(
            name=_basename(reference.joint_label[joint]),
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=child_anchor[:3],
            axis=axis_child,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[float(joint_lower[dof]), float(joint_upper[dof])],
            armature=float(joint_armature[dof]),
            frictionloss=float(joint_friction[dof]),
            damping=float(joint_damping[dof]),
        )

    _add_shapes(mj_spec, reference, bodies, settings)
    excluded_body_pairs = _add_collision_excludes(mj_spec, reference)
    mj_spec.add_texture(
        name="ground_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.88, 0.93, 0.98],
        rgb2=[0.10, 0.32, 0.58],
        markrgb=[0.98, 0.98, 1.0],
        width=300,
        height=300,
    )
    mj_spec.add_material(
        name="ground_material",
        textures=["", "ground_checker"],
        texrepeat=[5.0, 5.0],
        texuniform=1,
        reflectance=0.15,
    )
    mj_spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        contype=1,
        conaffinity=1,
        friction=[settings.ground_friction, 0.005, 0.0001],
        solref=[-settings.contact_stiffness, -settings.contact_damping],
        material="ground_material",
    )
    mj_spec.worldbody.add_light(
        name="scene_light",
        pos=[0.0, 0.0, 2.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    for slot, name in enumerate(JOINT_NAMES):
        mj_spec.add_actuator(
            name=name.removesuffix("_joint"),
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=[-JOINT_EFFORT_LIMIT[slot], JOINT_EFFORT_LIMIT[slot]],
        )
    model = mj_spec.compile()
    constants = mujoco.MjData(model)
    mujoco.mj_setConst(model, constants)
    return model, excluded_body_pairs


def _joint_addresses(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos: list[int] = []
    qvel: list[int] = []
    actuators: list[int] = []
    for name in JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}")
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
        if len(matches) != 1:
            raise ValueError(f"expected one actuator for {name!r}, found {len(matches)}")
        qpos.append(int(model.jnt_qposadr[joint]))
        qvel.append(int(model.jnt_dofadr[joint]))
        actuators.append(int(matches[0]))
    return np.asarray(qpos), np.asarray(qvel), np.asarray(actuators)


def _geometry_digest(reference: Any) -> str:
    digest = hashlib.sha256()
    for shape, flags in enumerate(reference.shape_flags.numpy()):
        if not int(flags) & COLLIDABLE:
            continue
        digest.update(np.asarray(reference.shape_type.numpy()[shape], dtype=np.int32).tobytes())
        digest.update(np.asarray(reference.shape_body.numpy()[shape], dtype=np.int32).tobytes())
        digest.update(np.asarray(reference.shape_scale.numpy()[shape], dtype=np.float32).tobytes())
        digest.update(np.asarray(reference.shape_transform.numpy()[shape], dtype=np.float32).tobytes())
        source = reference.shape_source[shape]
        if source is not None:
            digest.update(np.asarray(source.vertices, dtype=np.float32).tobytes())
            digest.update(np.asarray(source.indices, dtype=np.int32).tobytes())
    return digest.hexdigest()


def verify_asset_contract(reference: Any, model: mujoco.MjModel) -> dict[str, Any]:
    """Fail before rollout if topology, inertials, limits, or FK drifted."""

    actual = {
        "bodies": int(reference.body_count),
        "joints": int(reference.joint_count),
        "shapes": int(reference.shape_count),
    }
    expected = {"bodies": 30, "joints": 30, "shapes": 58}
    if actual != expected:
        raise ValueError(f"unexpected G1 topology: {actual}, expected {expected}")
    if model.nbody != actual["bodies"] + 1 or model.ngeom != actual["shapes"] + 1:
        raise ValueError("MuJoCo topology differs from the parsed training asset")

    max_mass = 0.0
    max_com = 0.0
    max_inertia = 0.0
    for body, label in enumerate(reference.body_label):
        mj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _basename(label))
        if mj_body < 0:
            raise ValueError(f"MuJoCo is missing body {label!r}")
        max_mass = max(
            max_mass,
            abs(float(model.body_mass[mj_body]) - float(reference.body_mass.numpy()[body])),
        )
        max_com = max(
            max_com,
            float(np.max(np.abs(model.body_ipos[mj_body] - reference.body_com.numpy()[body]))),
        )
        source_inertia = np.sort(np.linalg.eigvalsh(reference.body_inertia.numpy()[body]))
        max_inertia = max(
            max_inertia,
            float(np.max(np.abs(np.sort(model.body_inertia[mj_body]) - source_inertia))),
        )

    max_limit = 0.0
    for joint in range(int(reference.joint_count)):
        if int(reference.joint_parent.numpy()[joint]) < 0:
            continue
        name = _basename(reference.joint_label[joint])
        mj_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = int(reference.joint_qd_start.numpy()[joint])
        source_range = np.asarray(
            (
                reference.joint_limit_lower.numpy()[dof],
                reference.joint_limit_upper.numpy()[dof],
            )
        )
        max_limit = max(max_limit, float(np.max(np.abs(model.jnt_range[mj_joint] - source_range))))

    max_fk_position = 0.0
    max_fk_rotation = 0.0
    for positions in ({}, dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS, strict=True))):
        data = mujoco.MjData(model)
        for name, value in positions.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[joint])] = value
        mujoco.mj_forward(model, data)
        worlds = _build_world_transforms(reference, positions)
        for body, label in enumerate(reference.body_label):
            mj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _basename(label))
            max_fk_position = max(
                max_fk_position,
                float(np.max(np.abs(data.xpos[mj_body] - worlds[body][:3, 3]))),
            )
            delta = worlds[body][:3, :3].T @ data.xmat[mj_body].reshape(3, 3)
            angle = math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
            max_fk_rotation = max(max_fk_rotation, angle)

    if max_mass > 1.0e-7 or max_com > 1.0e-6 or max_inertia > 1.0e-7:
        raise ValueError("MuJoCo inertials do not match the parsed training asset")
    if max_limit > 1.0e-7:
        raise ValueError("MuJoCo joint limits do not match the parsed training asset")
    if max_fk_position > 1.0e-6 or max_fk_rotation > 1.0e-6:
        raise ValueError("MuJoCo FK does not match the parsed training asset")
    collision_shapes = int(np.count_nonzero(reference.shape_flags.numpy() & COLLIDABLE))
    foot_shapes = sum(
        int(bool(int(reference.shape_flags.numpy()[shape]) & COLLIDABLE) and "ankle_roll_link" in label)
        for shape, label in enumerate(reference.shape_label)
    )
    return {
        **actual,
        "total_mass_kg": float(reference.body_mass.numpy().sum()),
        "collision_shapes": collision_shapes,
        "foot_collision_shapes": foot_shapes,
        "collision_geometry_sha256": _geometry_digest(reference),
        "max_mass_error": max_mass,
        "max_com_error_m": max_com,
        "max_principal_inertia_error": max_inertia,
        "max_joint_limit_error_rad": max_limit,
        "max_fk_position_error_m": max_fk_position,
        "max_fk_rotation_error_rad": max_fk_rotation,
    }


class TermHistory:
    """NumPy version of the training policy's term-major history buffer."""

    def __init__(self, term_dims: tuple[int, ...], length: int) -> None:
        self.buffers = [np.zeros((length, dim), dtype=np.float32) for dim in term_dims]

    def fill(self, terms: tuple[np.ndarray, ...]) -> np.ndarray:
        for buffer, value in zip(self.buffers, terms, strict=True):
            buffer[:] = value
        return self.output()

    def update(self, terms: tuple[np.ndarray, ...]) -> np.ndarray:
        for buffer, value in zip(self.buffers, terms, strict=True):
            buffer[:-1] = buffer[1:]
            buffer[-1] = value
        return self.output()

    def output(self) -> np.ndarray:
        return np.concatenate([buffer.reshape(-1) for buffer in self.buffers])


def _rpy_degrees(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    return np.rad2deg(
        (
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


@dataclass
class StepResult:
    action: np.ndarray
    torque: np.ndarray
    linear_velocity_body: np.ndarray


class G1MuJoCoPlayer:
    """One deterministic MuJoCo rollout with the G1 training policy ABI."""

    def __init__(
        self,
        settings: RunSettings,
        actor: torch.nn.Module,
        model: mujoco.MjModel,
        command: np.ndarray,
    ) -> None:
        self.settings = settings
        self.actor = actor
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos, self.qvel, self.actuators = _joint_addresses(model)
        free_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        self.root_qpos = int(model.jnt_qposadr[free_joint])
        self.root_body = int(model.jnt_bodyid[free_joint])
        self.default = np.asarray(DEFAULT_JOINT_POS, dtype=np.float64)
        self.kp = np.asarray(JOINT_KP, dtype=np.float64)
        self.kd = np.asarray(JOINT_KD, dtype=np.float64)
        self.effort = np.asarray(JOINT_EFFORT_LIMIT, dtype=np.float64)
        self.command = np.asarray(command, dtype=np.float32)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.history = TermHistory(POLICY_TERM_DIMS, HISTORY_LENGTH)
        self.observation = np.zeros(POLICY_OBSERVATION_DIM, dtype=np.float32)
        self.steps = 0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        start = self.root_qpos
        self.data.qpos[start : start + 3] = (0.0, 0.0, self.settings.initial_height)
        self.data.qpos[start + 3 : start + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos] = self.default
        self.last_action.fill(0.0)
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.observation = self.history.fill(self._observation_terms())

    def _body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity_world = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.root_body,
            velocity_world,
            0,
        )
        rotation = self.data.xmat[self.root_body].reshape(3, 3)
        linear = rotation.T @ velocity_world[3:]
        angular = rotation.T @ velocity_world[:3]
        return linear, angular

    def _observation_terms(self) -> tuple[np.ndarray, ...]:
        _linear, angular = self._body_velocity()
        rotation = self.data.xmat[self.root_body].reshape(3, 3)
        gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        terms = (
            (0.2 * angular).astype(np.float32),
            gravity.astype(np.float32),
            self.command,
            (self.data.qpos[self.qpos] - self.default).astype(np.float32),
            (0.05 * self.data.qvel[self.qvel]).astype(np.float32),
            self.last_action,
        )
        if tuple(term.size for term in terms) != POLICY_TERM_DIMS:
            raise RuntimeError("MuJoCo observation terms do not match the G1 ABI")
        return terms

    def step(self) -> StepResult:
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(self.observation).unsqueeze(0))[0].numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError("actor output contains NaN or Inf")
        target = self.default + self.settings.action_scale * action
        torque = np.zeros(ACTION_DIM, dtype=np.float64)
        for _ in range(self.settings.decimation):
            q = self.data.qpos[self.qpos]
            qd = self.data.qvel[self.qvel]
            torque = np.clip(self.kp * (target - q) - self.kd * qd, -self.effort, self.effort)
            self.data.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, self.data)
        self.last_action[:] = action
        self.observation = self.history.update(self._observation_terms())
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state contains NaN or Inf")
        linear, _angular = self._body_velocity()
        return StepResult(action=action, torque=torque, linear_velocity_body=linear)

    def root_state(self) -> tuple[np.ndarray, np.ndarray]:
        start = self.root_qpos
        return self.data.qpos[start : start + 3].copy(), _rpy_degrees(self.data.qpos[start + 3 : start + 7])

    def failure_reason(self) -> str | None:
        rotation = self.data.xmat[self.root_body].reshape(3, 3)
        gravity_body = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        tilt = math.acos(float(np.clip(-gravity_body[2], -1.0, 1.0)))
        if self.data.xpos[self.root_body, 2] < 0.2:
            return "base_height"
        if tilt > 0.8:
            return "bad_orientation"
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    settings = load_run_settings(checkpoint, args.robot_friction)
    snapshot = json.loads(settings.snapshot.read_text(encoding="utf-8"))
    asset, asset_source, asset_hash_verified = resolve_training_asset(snapshot, args.asset)
    reference = load_training_asset(asset, settings)
    model, excluded_body_pairs = build_mujoco_model(reference, settings)
    contract = verify_asset_contract(reference, model)
    actor = load_actor(checkpoint, settings)
    command = np.asarray((args.command_vx, args.command_vy, args.command_yaw), dtype=np.float32)
    player = G1MuJoCoPlayer(settings, actor, model, command)

    if args.xml_output is not None:
        xml_output = args.xml_output.expanduser().resolve()
        xml_output.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(xml_output), model)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, player.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 125.0
        viewer.cam.elevation = -16.0

    max_steps = None if args.duration == 0.0 else round(args.duration / settings.policy_dt)
    maximum_action = 0.0
    maximum_torque = 0.0
    minimum_height = float("inf")
    sum_linear = np.zeros(3, dtype=np.float64)
    first_failure_seconds: float | None = None
    first_failure_reason: str | None = None
    start = time.perf_counter()
    try:
        while (max_steps is None or player.steps < max_steps) and (viewer is None or viewer.is_running()):
            step = player.step()
            position, _rpy = player.root_state()
            maximum_action = max(maximum_action, float(np.max(np.abs(step.action))))
            maximum_torque = max(maximum_torque, float(np.max(np.abs(step.torque))))
            minimum_height = min(minimum_height, float(position[2]))
            sum_linear += step.linear_velocity_body
            reason = player.failure_reason()
            if first_failure_seconds is None and reason is not None:
                first_failure_seconds = player.steps * settings.policy_dt
                first_failure_reason = reason
            if viewer is not None:
                viewer.cam.lookat[:] = position
                viewer.cam.lookat[2] = max(float(position[2]) * 0.7, 0.25)
                viewer.sync()
                deadline = start + player.steps * settings.policy_dt / args.playback_speed
                time.sleep(max(0.0, deadline - time.perf_counter()))
    finally:
        if viewer is not None:
            viewer.close()

    position, rpy = player.root_state()
    linear, angular = player._body_velocity()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "environment_snapshot": str(settings.snapshot),
        "environment_snapshot_sha256": _sha256(settings.snapshot),
        "asset": str(asset),
        "asset_source": asset_source,
        "asset_sha256": _sha256(asset),
        "asset_hash_verified_from_training_snapshot": asset_hash_verified,
        "asset_mode": "training-asset-parsed-with-training-import-options",
        "asset_contract": contract,
        "self_collisions": settings.self_collisions,
        "translated_collision_filter_body_pairs": excluded_body_pairs,
        "contact": {
            "stiffness": settings.contact_stiffness,
            "damping": settings.contact_damping,
            "synreal_friction_gain_not_available_in_mujoco": settings.contact_friction_gain,
            "robot_friction": settings.robot_friction,
            "ground_friction": settings.ground_friction,
            "training_robot_friction_range": list(settings.friction_range),
            "note": "MuJoCo and SynReal combine friction and solve compliant contact differently.",
        },
        "integration": {
            "training_physics_dt": settings.training_dt,
            "training_decimation": settings.training_decimation,
            "mujoco_physics_dt": settings.dt,
            "mujoco_decimation": settings.decimation,
            "policy_dt": settings.policy_dt,
        },
        "simulated_seconds": player.steps * settings.policy_dt,
        "steps": player.steps,
        "command": command.tolist(),
        "mean_body_linear_velocity": (sum_linear / max(player.steps, 1)).tolist(),
        "maximum_abs_action": maximum_action,
        "maximum_abs_torque": maximum_torque,
        "minimum_base_height": minimum_height,
        "first_failure_seconds": first_failure_seconds,
        "first_failure_reason": first_failure_reason,
        "passed_training_termination_conditions": first_failure_reason is None,
        "final_position": position.tolist(),
        "final_rpy_degrees": rpy.tolist(),
        "final_body_linear_velocity": linear.tolist(),
        "final_body_angular_velocity": angular.tolist(),
        "finite": bool(np.isfinite(player.data.qpos).all() and np.isfinite(player.data.qvel).all()),
        "observation_dim": POLICY_OBSERVATION_DIM,
        "history_layout": "term-major, oldest-to-newest",
        "policy_hz": 1.0 / settings.policy_dt,
        "physics_hz": 1.0 / settings.dt,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.summary is not None:
        output = args.summary.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset", type=Path, help="override the checkpoint's training asset")
    parser.add_argument("--command-vx", type=float, default=0.3)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0, help="zero runs until the viewer closes")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--robot-friction",
        type=float,
        help="fixed MuJoCo robot friction; defaults to training-range midpoint",
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--xml-output", type=Path)
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if args.headless and args.duration == 0.0:
        parser.error("headless evaluation requires a positive --duration")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.robot_friction is not None and args.robot_friction < 0.0:
        parser.error("--robot-friction must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())
