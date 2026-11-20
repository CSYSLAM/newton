"""Validate a Unitree Go2 checkpoint in MuJoCo using the training USD asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import yaml

if __package__ in (None, ""):
    from locomotion.deployment import load_actor
    from locomotion.env import resolve_asset
    from locomotion.policy_spec import load_policy_spec

    from locomotion_unitree.config import load_config
else:
    from locomotion.deployment import load_actor
    from locomotion.env import resolve_asset
    from locomotion.policy_spec import load_policy_spec

    from .config import load_config


VISIBLE = 1
COLLIDABLE = 2


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quat_wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray((quaternion_xyzw[3], *quaternion_xyzw[:3]), dtype=np.float64)


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


def load_run_config(checkpoint: Path) -> tuple[Any, dict[str, Any], Path]:
    """Use the immutable task snapshot written beside an RSL-RL checkpoint."""
    saved = checkpoint.parent / "task.yaml"
    if not saved.is_file():
        spec, unitree_cfg = load_config()
        return spec, unitree_cfg, saved
    data = yaml.safe_load(saved.read_text(encoding="utf-8"))
    if "unitree" not in data:
        raise ValueError(f"checkpoint task snapshot has no Unitree settings: {saved}")
    return load_policy_spec(saved), data["unitree"], saved


def load_training_asset(spec: Any, asset_override: Path | None = None) -> tuple[Path, Any]:
    """Parse the exact USD and importer options used by UnitreeGo2Env."""
    import newton
    from newton.solvers import coupled_high_accuracy as cha

    asset = resolve_asset(spec, asset_override)
    builder = cha.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.01,
        limit_ke=1.0e3,
        limit_kd=10.0,
        friction=1.0e-5,
    )
    builder.default_shape_cfg.ke = spec.contact_stiffness
    builder.default_shape_cfg.kd = spec.contact_damping
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = spec.contact_friction
    builder.add_usd(
        str(asset),
        hide_collision_shapes=True,
        floating=True,
        collapse_fixed_joints=True,
        enable_self_collisions=False,
    )
    return asset, builder.finalize(device="cpu")


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
        else:
            if worlds[parent] is None:
                raise ValueError("training USD joints are not parent-before-child ordered")
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
        raise ValueError("training USD has a body disconnected from the joint tree")
    return [world for world in worlds if world is not None]


def _add_shapes(
    mj_spec: mujoco.MjSpec,
    reference: Any,
    bodies: list[Any],
) -> None:
    shape_type = reference.shape_type.numpy()
    shape_body = reference.shape_body.numpy()
    shape_scale = reference.shape_scale.numpy()
    shape_transform = reference.shape_transform.numpy()
    shape_flags = reference.shape_flags.numpy()
    shape_rgba = reference.shape_color.numpy()

    for index, label in enumerate(reference.shape_label):
        body_index = int(shape_body[index])
        if body_index < 0:
            continue
        transform = shape_transform[index]
        position = transform[:3]
        quaternion = _quat_wxyz(transform[3:7])
        flags = int(shape_flags[index])
        collidable = bool(flags & COLLIDABLE)
        visible = bool(flags & VISIBLE)
        color = np.asarray(shape_rgba[index], dtype=np.float64).reshape(-1)
        if color.size == 3:
            color = np.append(color, 1.0)
        common = {
            "name": f"shape_{index}_{_basename(label)}",
            "pos": position,
            "quat": quaternion,
            "contype": 2 if collidable else 0,
            "conaffinity": 1 if collidable else 0,
            "friction": [
                float(reference.shape_material_mu.numpy()[index]),
                0.005,
                0.0001,
            ],
            "rgba": color if visible else [0.0, 0.0, 0.0, 0.0],
            "group": 0 if collidable else 2,
            "density": 0.0,
        }
        if collidable:
            common["solref"] = [
                -float(reference.shape_material_ke.numpy()[index]),
                -float(reference.shape_material_kd.numpy()[index]),
            ]
        kind = int(shape_type[index])
        scale = shape_scale[index]
        if kind == 3:  # newton.GeoType.SPHERE
            bodies[body_index].add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[float(scale[0]), 0.0, 0.0],
                **common,
            )
        elif kind == 6:  # newton.GeoType.CYLINDER: radius, half-height
            bodies[body_index].add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[float(scale[0]), float(scale[1]), 0.0],
                **common,
            )
        elif kind == 7:  # newton.GeoType.BOX: half extents
            bodies[body_index].add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=scale, **common)
        elif kind == 8:  # newton.GeoType.MESH
            source = reference.shape_source[index]
            if source is None:
                raise ValueError(f"visual mesh {label!r} has no source data")
            mesh_name = f"mesh_{index}"
            mj_spec.add_mesh(
                name=mesh_name,
                scale=scale,
                uservert=np.asarray(source.vertices, dtype=np.float64).reshape(-1),
                userface=np.asarray(source.indices, dtype=np.int32).reshape(-1),
            )
            bodies[body_index].add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=mesh_name,
                size=[1.0, 1.0, 1.0],
                **common,
            )
        else:
            raise NotImplementedError(f"unsupported Newton shape type {kind} for {label!r}")


def build_mujoco_model(reference: Any, spec: Any) -> mujoco.MjModel:
    """Create a MuJoCo model only from the model parsed from the training USD."""
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

    mj_spec = mujoco.MjSpec()
    mj_spec.modelname = "unitree_go2_training_usd"
    mj_spec.compiler.degree = 0
    mj_spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_FALSE
    mj_spec.option.timestep = spec.dt
    mj_spec.option.gravity = [0.0, 0.0, -9.81]
    mj_spec.option.iterations = spec.solver_iterations
    mj_spec.stat.center = [0.0, 0.0, 0.1]
    mj_spec.stat.extent = 0.8
    mj_spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    mj_spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    mj_spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    mj_spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
    mj_spec.visual.global_.azimuth = -130.0
    mj_spec.visual.global_.elevation = -20.0

    bodies: list[Any] = [None] * int(reference.body_count)
    for body_index, label in enumerate(reference.body_label):
        inertia, iquat = _principal_inertia(inertias[body_index])
        if body_index == 0:
            parent = mj_spec.worldbody
            relative = np.eye(4, dtype=np.float64)
        else:
            joint = int(np.flatnonzero(joint_child == body_index)[0])
            parent_index = int(joint_parent[joint])
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

    bodies[0].add_freejoint(name="floating_base")
    for joint in range(1, int(reference.joint_count)):
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
            armature=0.01,
            frictionloss=1.0e-5,
        )

    _add_shapes(mj_spec, reference, bodies)
    mj_spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    mj_spec.add_texture(
        name="ground_checker",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mj_spec.add_material(
        name="ground_material",
        textures=["", "ground_checker"],
        texrepeat=[5.0, 5.0],
        texuniform=1,
        reflectance=0.2,
    )
    mj_spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        contype=1,
        conaffinity=2,
        friction=[spec.contact_friction, 0.005, 0.0001],
        solref=[-spec.contact_stiffness, -spec.contact_damping],
        material="ground_material",
    )
    mj_spec.worldbody.add_light(
        name="scene_light",
        pos=[0.0, 0.0, 1.5],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    for name in spec.policy_joint_names:
        mj_spec.add_actuator(
            name=name.removesuffix("_joint"),
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[1.0] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_FALSE,
        )
    model = mj_spec.compile()
    constants = mujoco.MjData(model)
    mujoco.mj_setConst(model, constants)
    return model


def _joint_addresses(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos: list[int] = []
    qvel: list[int] = []
    actuators: list[int] = []
    for name in names:
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


def verify_asset_contract(
    reference: Any,
    model: mujoco.MjModel,
    policy_joint_names: tuple[str, ...],
    default_joint_pos: tuple[float, ...],
) -> dict[str, float | int]:
    """Verify topology, inertials, limits, and zero-pose FK across engines."""
    expected = {"bodies": 13, "joints": 13, "shapes": 60}
    actual = {
        "bodies": int(reference.body_count),
        "joints": int(reference.joint_count),
        "shapes": int(reference.shape_count),
    }
    if actual != expected:
        raise ValueError(f"unexpected training asset topology: {actual}, expected {expected}")
    if model.nbody != int(reference.body_count) + 1:
        raise ValueError("MuJoCo body count differs from training USD")
    if model.ngeom != int(reference.shape_count) + 1:
        raise ValueError("MuJoCo geom count differs from training USD plus ground")

    max_mass = 0.0
    max_com = 0.0
    max_inertia = 0.0
    max_joint_limit = 0.0
    max_shape_position = 0.0
    max_shape_rotation = 0.0
    max_shape_size = 0.0
    for body_index, label in enumerate(reference.body_label):
        name = _basename(label)
        mj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        max_mass = max(
            max_mass,
            abs(float(model.body_mass[mj_body]) - float(reference.body_mass.numpy()[body_index])),
        )
        max_com = max(
            max_com,
            float(np.max(np.abs(model.body_ipos[mj_body] - reference.body_com.numpy()[body_index]))),
        )
        source_eigenvalues = np.linalg.eigvalsh(reference.body_inertia.numpy()[body_index])
        max_inertia = max(
            max_inertia,
            float(np.max(np.abs(np.sort(model.body_inertia[mj_body]) - source_eigenvalues))),
        )

    for joint in range(1, int(reference.joint_count)):
        name = _basename(reference.joint_label[joint])
        mj_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if mj_joint < 0:
            raise ValueError(f"MuJoCo model is missing training joint {name!r}")
        dof = int(reference.joint_qd_start.numpy()[joint])
        source_range = np.asarray(
            (
                reference.joint_limit_lower.numpy()[dof],
                reference.joint_limit_upper.numpy()[dof],
            )
        )
        max_joint_limit = max(
            max_joint_limit,
            float(np.max(np.abs(model.jnt_range[mj_joint] - source_range))),
        )

    kind_to_mujoco = {
        3: mujoco.mjtGeom.mjGEOM_SPHERE,
        6: mujoco.mjtGeom.mjGEOM_CYLINDER,
        7: mujoco.mjtGeom.mjGEOM_BOX,
    }
    collision_shapes = 0
    visual_meshes = 0
    for shape, label in enumerate(reference.shape_label):
        flags = int(reference.shape_flags.numpy()[shape])
        kind = int(reference.shape_type.numpy()[shape])
        collision_shapes += int(bool(flags & COLLIDABLE))
        visual_meshes += int(kind == 8 and bool(flags & VISIBLE))
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"shape_{shape}_{_basename(label)}")
        if geom < 0:
            raise ValueError(f"MuJoCo model is missing training shape {label!r}")
        if int(model.geom_bodyid[geom]) != int(reference.shape_body.numpy()[shape]) + 1:
            raise ValueError(f"MuJoCo shape {label!r} is attached to the wrong body")
        if kind not in kind_to_mujoco:
            continue
        if model.geom_type[geom] != kind_to_mujoco[kind]:
            raise ValueError(f"MuJoCo shape {label!r} has the wrong primitive type")
        transform = reference.shape_transform.numpy()[shape]
        max_shape_position = max(
            max_shape_position,
            float(np.max(np.abs(model.geom_pos[geom] - transform[:3]))),
        )
        source_rotation = _rotation_xyzw(transform[3:7])
        geom_quaternion = model.geom_quat[geom]
        geom_rotation = _rotation_xyzw(np.asarray((*geom_quaternion[1:4], geom_quaternion[0])))
        delta = source_rotation.T @ geom_rotation
        max_shape_rotation = max(
            max_shape_rotation,
            math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))),
        )
        source_size = reference.shape_scale.numpy()[shape]
        width = 1 if kind == 3 else (2 if kind == 6 else 3)
        max_shape_size = max(
            max_shape_size,
            float(np.max(np.abs(model.geom_size[geom, :width] - source_size[:width]))),
        )

    max_position = 0.0
    max_rotation = 0.0
    for positions in ({}, dict(zip(policy_joint_names, default_joint_pos))):
        data = mujoco.MjData(model)
        for name, value in positions.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[joint])] = value
        mujoco.mj_forward(model, data)
        worlds = _build_world_transforms(reference, positions)
        for body_index, label in enumerate(reference.body_label):
            mj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _basename(label))
            max_position = max(
                max_position,
                float(np.max(np.abs(data.xpos[mj_body] - worlds[body_index][:3, 3]))),
            )
            source_rotation = worlds[body_index][:3, :3]
            delta = source_rotation.T @ data.xmat[mj_body].reshape(3, 3)
            angle = math.acos(float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
            max_rotation = max(max_rotation, angle)

    if max_mass > 1.0e-7 or max_com > 1.0e-6 or max_inertia > 1.0e-7:
        raise ValueError("MuJoCo inertial data does not match the training USD")
    if max_joint_limit > 1.0e-7:
        raise ValueError("MuJoCo joint limits do not match the training USD")
    if max_shape_position > 1.0e-6 or max_shape_rotation > 1.0e-4 or max_shape_size > 1.0e-7:
        raise ValueError(
            "MuJoCo collision primitives do not match the training USD: "
            f"position={max_shape_position}, rotation={max_shape_rotation}, size={max_shape_size}"
        )
    if max_position > 1.0e-6 or max_rotation > 1.0e-6:
        raise ValueError("MuJoCo zero-pose FK does not match the training USD")
    return {
        **actual,
        "total_mass_kg": float(reference.body_mass.numpy().sum()),
        "max_mass_error": max_mass,
        "max_com_error_m": max_com,
        "max_principal_inertia_error": max_inertia,
        "max_joint_limit_error_rad": max_joint_limit,
        "collision_shapes": collision_shapes,
        "visual_meshes": visual_meshes,
        "max_primitive_position_error_m": max_shape_position,
        "max_primitive_rotation_error_rad": max_shape_rotation,
        "max_primitive_size_error_m": max_shape_size,
        "max_fk_position_error_m": max_position,
        "max_fk_rotation_error_rad": max_rotation,
    }


def torque_speed_limit(
    desired: np.ndarray,
    velocity: np.ndarray,
    *,
    x1: float,
    x2: float,
    y1: float,
    y2: float,
) -> np.ndarray:
    limit = np.where(velocity * desired > 0.0, y1, y2).astype(np.float64)
    derating = np.clip((x2 - np.abs(velocity)) / (x2 - x1), 0.0, 1.0)
    limit *= np.where(np.abs(velocity) >= x1, derating, 1.0)
    return np.clip(desired, -limit, limit)


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


class Go2MuJoCoPlayer:
    def __init__(
        self,
        spec: Any,
        unitree_cfg: dict[str, Any],
        actor: torch.nn.Module,
        model: mujoco.MjModel,
        command: np.ndarray,
    ) -> None:
        self.spec = spec
        self.actor = actor
        self.model = model
        self.data = mujoco.MjData(model)
        self.qpos, self.qvel, self.actuators = _joint_addresses(model, spec.policy_joint_names)
        free_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
        self.root_qpos = int(model.jnt_qposadr[free_joint])
        self.root_body = int(model.jnt_bodyid[free_joint])
        self.default = np.asarray(spec.default_joint_pos, dtype=np.float64)
        self.scale = np.asarray(spec.action_scale, dtype=np.float64)
        self.kp = np.asarray(spec.stiffness, dtype=np.float64)
        self.kd = np.asarray(spec.damping, dtype=np.float64)
        self.command = np.asarray(command, dtype=np.float32)
        self.last_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.torque_curve = {key: float(value) for key, value in unitree_cfg["torque_speed"].items()}
        self.steps = 0
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        start = self.root_qpos
        self.data.qpos[start : start + 3] = (0.0, 0.0, self.spec.initial_height)
        self.data.qpos[start + 3 : start + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos] = self.default
        self.last_action.fill(0.0)
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)

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
        return rotation.T @ velocity_world[3:], rotation.T @ velocity_world[:3]

    def observation(self) -> np.ndarray:
        _linear, angular = self._body_velocity()
        rotation = self.data.xmat[self.root_body].reshape(3, 3)
        gravity = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        observation = np.concatenate(
            (
                self.spec.ang_vel_scale * angular,
                gravity,
                self.command,
                self.data.qpos[self.qpos] - self.default,
                self.spec.joint_vel_scale * self.data.qvel[self.qvel],
                self.last_action,
            )
        ).astype(np.float32)
        if observation.shape != (self.spec.observation_dim,):
            raise RuntimeError(f"observation is {observation.shape}, expected {self.spec.observation_dim}D")
        return np.clip(observation, -100.0, 100.0)

    def step(self) -> StepResult:
        with torch.inference_mode():
            action = self.actor(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy()
        action = np.clip(action, -self.spec.action_clip, self.spec.action_clip)
        target = self.default + self.scale * action
        torque = np.zeros(self.spec.action_dim, dtype=np.float64)
        for _ in range(self.spec.decimation):
            q = self.data.qpos[self.qpos]
            qd = self.data.qvel[self.qvel]
            desired = self.kp * (target - q) - self.kd * qd
            torque = torque_speed_limit(desired, qd, **self.torque_curve)
            self.data.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, self.data)
        self.last_action[:] = action
        self.steps += 1
        linear, _angular = self._body_velocity()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state contains NaN or Inf")
        return StepResult(action, torque, linear)

    def root_state(self) -> tuple[np.ndarray, np.ndarray]:
        start = self.root_qpos
        return self.data.qpos[start : start + 3].copy(), _rpy_degrees(self.data.qpos[start + 3 : start + 7])

    def failure_reason(self, bad_orientation_angle: float) -> str | None:
        rotation = self.data.xmat[self.root_body].reshape(3, 3)
        gravity_body = rotation.T @ np.asarray((0.0, 0.0, -1.0))
        tilt = math.acos(float(np.clip(-gravity_body[2], -1.0, 1.0)))
        if tilt > bad_orientation_angle:
            return "bad_orientation"
        for contact in self.data.contact[: self.data.ncon]:
            bodies = self.model.geom_bodyid[np.asarray(contact.geom, dtype=np.int32)]
            if self.root_body in bodies and 0 in bodies:
                return "base_contact"
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    spec, unitree_cfg, saved_task = load_run_config(checkpoint)
    asset, reference = load_training_asset(spec, args.asset)
    model = build_mujoco_model(reference, spec)
    contract = verify_asset_contract(reference, model, spec.policy_joint_names, spec.default_joint_pos)
    actor = load_actor(spec, checkpoint)
    command = np.asarray((args.command_vx, args.command_vy, args.command_yaw), dtype=np.float32)
    player = Go2MuJoCoPlayer(spec, unitree_cfg, actor, model, command)

    viewer = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, player.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -14.0

    max_steps = None if args.duration == 0.0 else round(args.duration / spec.policy_dt)
    maximum_action = 0.0
    maximum_torque = 0.0
    minimum_height = float("inf")
    sum_linear = np.zeros(3, dtype=np.float64)
    first_failure_seconds = None
    first_failure_reason = None
    start = time.perf_counter()
    try:
        while (max_steps is None or player.steps < max_steps) and (viewer is None or viewer.is_running()):
            result = player.step()
            position, _rpy = player.root_state()
            maximum_action = max(maximum_action, float(np.max(np.abs(result.action))))
            maximum_torque = max(maximum_torque, float(np.max(np.abs(result.torque))))
            minimum_height = min(minimum_height, float(position[2]))
            sum_linear += result.linear_velocity_body
            failure_reason = player.failure_reason(float(unitree_cfg["bad_orientation_angle"]))
            if first_failure_seconds is None and failure_reason is not None:
                first_failure_seconds = player.steps * spec.policy_dt
                first_failure_reason = failure_reason
            if viewer is not None:
                viewer.cam.lookat[:] = position
                viewer.cam.lookat[2] = max(float(position[2]) * 0.65, 0.18)
                viewer.sync()
                deadline = start + player.steps * spec.policy_dt / args.playback_speed
                time.sleep(max(0.0, deadline - time.perf_counter()))
    finally:
        if viewer is not None:
            viewer.close()

    position, rpy = player.root_state()
    linear, angular = player._body_velocity()
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_task": str(saved_task) if saved_task.is_file() else None,
        "asset": str(asset),
        "asset_sha256": _sha256(asset),
        "asset_mode": "exact-training-usd-parsed-with-identical-import-options",
        "asset_contract": contract,
        "physics_note": "MuJoCo and SynReal use different constraint/contact solvers.",
        "simulated_seconds": player.steps * spec.policy_dt,
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
        "observation_dim": spec.observation_dim,
        "policy_hz": 1.0 / spec.policy_dt,
        "physics_hz": 1.0 / spec.dt,
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
    parser.add_argument("--asset", type=Path, help="override the training USD")
    parser.add_argument("--command-vx", type=float, default=0.3)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0, help="zero runs until viewer closes")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.duration < 0.0:
        parser.error("--duration must be non-negative")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
