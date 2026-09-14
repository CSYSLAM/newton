# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the matching Newton and MuJoCo box-packing models."""

from __future__ import annotations

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import newton
from newton.solvers import SolverMuJoCo

HERE = Path(__file__).resolve().parent
MENAGERIE_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_REVISION = "affef0836947b64cc06c4ab1cbf0152835693374"
CREASE_NAMES = (
    "crease/lid",
    "crease/left_wing",
    "crease/right_wing",
    "crease/front",
    "crease/left_ear",
    "crease/right_ear",
)
INITIAL_CREASES = (-0.70, 0.12, -0.12, 0.11, 0.0, 0.0)


def cached_aloha() -> Path:
    """Return the pinned MuJoCo Menagerie ALOHA checkout, fetching it once."""
    override = os.environ.get("MUJOCO_MENAGERIE")
    if override:
        aloha = Path(override).expanduser().resolve() / "aloha"
        if not (aloha / "aloha.xml").is_file():
            raise FileNotFoundError(f"MUJOCO_MENAGERIE does not contain aloha/aloha.xml: {aloha.parent}")
        return aloha

    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    checkout = cache_home / "newton" / "menagerie" / MENAGERIE_REVISION
    aloha = checkout / "aloha"
    if (aloha / "aloha.xml").is_file():
        return aloha

    checkout.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching pinned ALOHA assets into {checkout}...", flush=True)
    with tempfile.TemporaryDirectory(prefix="menagerie-", dir=checkout.parent) as temporary_directory:
        staging = Path(temporary_directory)
        subprocess.run(["git", "init", "--quiet", staging.as_posix()], check=True)
        subprocess.run(
            ["git", "fetch", "--quiet", "--depth", "1", MENAGERIE_REPOSITORY, MENAGERIE_REVISION],
            cwd=staging,
            check=True,
        )
        subprocess.run(["git", "checkout", "--quiet", "FETCH_HEAD", "--", "aloha"], cwd=staging, check=True)
        if checkout.exists():
            raise FileExistsError(f"Asset cache appeared during fetch: {checkout}")
        staging.rename(checkout)
    return aloha


def _patched_aloha_xml(aloha: Path) -> bytes:
    """Add the reference contact sites and fingertip pads to ALOHA."""
    robot = ET.fromstring((aloha / "aloha.xml").read_bytes())
    worldbody = robot.find("worldbody")
    if worldbody is None:
        raise ValueError("ALOHA model has no worldbody")
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    for geom in robot.iter("geom"):
        if geom.get("class") == "visual" and geom.get("mesh") == "vx300s_7_gripper_wrist_mount":
            geom.set("rgba", "0.9 0.9 0.91 1")
    for side in ("left", "right"):
        wrist = robot.find(f".//body[@name='{side}/gripper_link']")
        if wrist is None:
            raise ValueError(f"ALOHA model has no {side} gripper link")
        ET.SubElement(wrist, "site", name=f"{side}/contact", pos="0.145375 0 -0.00335", size="0.001", group="5")
        for finger, sign in (("left", -1), ("right", 1)):
            body = robot.find(f".//body[@name='{side}/{finger}_finger_link']")
            if body is None:
                raise ValueError(f"ALOHA model has no {side}/{finger} finger link")
            ET.SubElement(
                body,
                "geom",
                name=f"{side}/{finger}_grip_pad",
                type="box",
                pos=f"0.0176 {sign * 0.086} 0.0275",
                size="0.006 0.009 0.0008",
                mass="0.001",
                rgba="0.08 0.08 0.08 1",
                friction="1.2 0.003 0.0001",
                condim="4",
                group="1",
            )
    return ET.tostring(robot)


def load_reference_model() -> mujoco.MjModel:
    """Load the observation model used by the feedback controller."""
    aloha = cached_aloha()
    assets = {
        path.relative_to(aloha).as_posix(): path.read_bytes()
        for path in aloha.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    assets["aloha.xml"] = _patched_aloha_xml(aloha)
    assets["carton.xml"] = (HERE / "carton.xml").read_bytes()
    return mujoco.MjModel.from_xml_string((HERE / "scene.xml").read_text(), assets=assets)


def initial_configuration(model: mujoco.MjModel) -> np.ndarray:
    """Return the observed open-carton configuration."""
    qpos = model.qpos0.copy()
    if model.nkey:
        qpos[:16] = model.key_qpos[0, :16]
    for name, angle in zip(CREASE_NAMES, INITIAL_CREASES, strict=True):
        qpos[model.joint(name).qposadr[0]] = angle
    return qpos


def _resolve_mjcf_path(aloha: Path, patched_aloha: str, base_dir: str | None, path: str) -> str:
    """Resolve scene includes and Menagerie mesh assets for Newton's importer."""
    if path == "aloha.xml":
        return patched_aloha
    if path == "carton.xml":
        return str(HERE / "carton.xml")
    if base_dir is not None:
        local = (Path(base_dir) / path).resolve()
        if local.exists():
            return str(local)
    return str((aloha / path).resolve())


def build_newton_model(initial_qpos: np.ndarray, initial_qvel: np.ndarray) -> newton.Model:
    """Import the workcell through Newton and initialize it from the replay."""
    aloha = cached_aloha()
    patched_aloha = _patched_aloha_xml(aloha).decode()
    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    # MuJoCo's default geom gap is zero; Newton's generic builder default is
    # intentionally larger and would change this contact-rich manipulation.
    builder.default_shape_cfg.gap = 0.0
    builder.add_mjcf(
        str(HERE / "scene.xml"),
        collapse_fixed_joints=False,
        ctrl_direct=True,
        path_resolver=lambda base_dir, path: _resolve_mjcf_path(aloha, patched_aloha, base_dir, path),
    )

    # These geoms are both the physical surface and its appearance. The MJCF
    # importer hides colliders when separate robot visuals exist in the scene.
    for shape, label in enumerate(builder.shape_label):
        if builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES and (
            "/carton/" in label or label.endswith(("/table", "/table/back_rail", "_grip_pad"))
        ):
            builder.shape_flags[shape] |= newton.ShapeFlags.VISIBLE

    if len(builder.joint_q) != len(initial_qpos) or len(builder.joint_qd) != len(initial_qvel):
        raise ValueError("Imported Newton joint layout does not match the box-packing replay")
    newton_qpos = initial_qpos.copy()
    # MuJoCo stores free-joint quaternions as wxyz; Newton uses xyzw.
    newton_qpos[19:23] = initial_qpos[[20, 21, 22, 19]]
    builder.joint_q[:] = newton_qpos.tolist()
    builder.joint_qd[:] = initial_qvel.tolist()
    return builder.finalize()
