"""Fetch pinned official Microduck assets and convert the full-collision MJCF tree to URDF."""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "microduck"
MODEL_REVISION = "cb70b792312d559a4da09064d92009079671815f"
POLICY_REVISION = "bb29428d40ba492ce5241ec7e9e2fc980f2a039c"
POLICY_SHA256 = "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
MODEL_PREFIX = "src/mjlab_microduck/robot/microduck/"
MODEL_FILENAME = "robot_allcollisions.xml"


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _numbers(values) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _rotation(quaternion: str) -> np.ndarray:
    w, x, y, z = np.fromstring(quaternion, sep=" ")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _origin(parent, source):
    rotation = _rotation(source.get("quat", "1 0 0 0"))
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    ET.SubElement(parent, "origin", xyz=source.get("pos", "0 0 0"), rpy=_numbers((roll, pitch, yaw)))


def convert_urdf(source: Path, destination: Path) -> None:
    """Convert the pinned, single-hinge-per-body model without changing its mass or frames."""
    mjcf = ET.parse(source).getroot()
    robot = ET.Element("robot", name="microduck")
    robot.append(ET.Comment(f"Derived from Pollen Robotics {source.name}; see README.md for source and license."))
    meshes = {m.get("name", Path(m.attrib["file"]).stem): m for m in mjcf.findall("asset/mesh")}
    materials = {m.attrib["name"]: m for m in mjcf.findall("asset/material")}

    def visit(body, parent_name=None):
        name = body.attrib["name"]
        link = ET.SubElement(robot, "link", name=name)
        inertial = body.find("inertial")
        if inertial is not None:
            node = ET.SubElement(link, "inertial")
            _origin(node, inertial)
            ET.SubElement(node, "mass", value=inertial.attrib["mass"])
            values = np.fromstring(inertial.attrib["fullinertia"], sep=" ")
            ET.SubElement(
                node,
                "inertia",
                **dict(zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), (f"{v:.12g}" for v in values), strict=True)),
            )
        for geom in body.findall("geom"):
            # Preserve the full model's environment colliders on every link. URDF cannot express
            # contact masks, so omit the internal support marked exclusively for self collision.
            if geom.get("class") not in ("visual", "collision"):
                continue
            kind = "visual" if geom.get("class") == "visual" else "collision"
            node = ET.SubElement(link, kind)
            _origin(node, geom)
            geometry = ET.SubElement(node, "geometry")
            if geom.get("type") != "mesh":
                raise ValueError(f"Unexpected geometry in pinned model: {geom.attrib}")
            mesh = meshes[geom.attrib["mesh"]]
            ET.SubElement(geometry, "mesh", filename="meshes/" + mesh.attrib["file"], scale=mesh.get("scale", "1 1 1"))
            if kind == "visual" and geom.get("material") in materials:
                material = ET.SubElement(node, "material", name=geom.attrib["material"])
                ET.SubElement(material, "color", rgba=materials[geom.attrib["material"]].get("rgba", "0.7 0.7 0.7 1"))
        if parent_name is not None:
            joints = body.findall("joint")
            if len(joints) != 1 or joints[0].get("type") != "hinge" or joints[0].get("pos", "0 0 0") != "0 0 0":
                raise ValueError("The converter only supports the pinned zero-offset hinge tree")
            joint = joints[0]
            node = ET.SubElement(robot, "joint", name=joint.attrib["name"], type="revolute")
            ET.SubElement(node, "parent", link=parent_name)
            ET.SubElement(node, "child", link=name)
            _origin(node, body)
            ET.SubElement(node, "axis", xyz=joint.attrib["axis"])
            lower, upper = joint.attrib["range"].split()
            ET.SubElement(node, "limit", lower=lower, upper=upper, effort="0.96", velocity="100")
            ET.SubElement(node, "dynamics", damping="0.053", friction="0.0048")
        for child in body.findall("body"):
            visit(child, name)

    visit(mjcf.find("worldbody/body"))
    ET.indent(robot)
    ET.ElementTree(robot).write(destination, encoding="utf-8", xml_declaration=True)


def download_assets(destination: Path = ASSET_DIR) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_url = f"https://codeload.github.com/pollen-robotics/microduck_rl/zip/{MODEL_REVISION}"
    archive = zipfile.ZipFile(io.BytesIO(_download(archive_url)))
    for entry in archive.infolist():
        relative = "/".join(entry.filename.split("/")[1:])
        output = None
        if relative in ("LICENSE", "README.md"):
            output = destination / ("LICENSE.model" if relative == "LICENSE" else "README.upstream.md")
        elif relative == MODEL_PREFIX + MODEL_FILENAME:
            output = destination / MODEL_FILENAME
        elif relative.startswith(MODEL_PREFIX + "assets/") and relative.endswith(".stl"):
            output = destination / "meshes" / Path(relative).name
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(entry))
    # Preserve the official MJCF alongside the URDF, with only its local mesh directory adjusted.
    source = destination / MODEL_FILENAME
    source.write_text(
        source.read_text(encoding="utf-8").replace('meshdir="assets"', 'meshdir="meshes"'), encoding="utf-8"
    )
    convert_urdf(source, destination / "microduck.urdf")
    base = f"https://raw.githubusercontent.com/pollen-robotics/microduck/{POLICY_REVISION}/"
    policy = _download(base + "policies/alpha_walking.onnx")
    if hashlib.sha256(policy).hexdigest() != POLICY_SHA256:
        raise ValueError("Official walking policy SHA-256 does not match the pinned version")
    (destination / "policies").mkdir(exist_ok=True)
    (destination / "policies/walk.onnx").write_bytes(policy)
    (destination / "LICENSE.policy").write_bytes(_download(base + "LICENSE"))
    (destination / "policies/README.upstream.md").write_bytes(_download(base + "policies/README.md"))
    (destination / "README.md").write_text(
        f"# Microduck assets\n\nModel: https://github.com/pollen-robotics/microduck_rl/tree/{MODEL_REVISION}\n\n"
        f"Policy: https://github.com/pollen-robotics/microduck/tree/{POLICY_REVISION}/policies\n\n"
        f"walk.onnx SHA-256: `{POLICY_SHA256}`\n\n"
        "The official distribution is MJCF, not URDF. microduck.urdf is generated by examples/microduck_assets.py. "
        "It uses robot_allcollisions.xml, preserving the 15 links, 14 hinges, full inertias, visual meshes and "
        "environment collision meshes on every link, including the trunk, head, neck and legs. "
        "The internal self-collision-only support is omitted. "
        "The floating base and motor armature/gains are set by the example.\n\n"
        "Upstream code and policy repository: Apache-2.0; see LICENSE.model and LICENSE.policy. "
        "Upstream explicitly licenses 3D models as Creative Commons BY-SA-NC; see README.upstream.md. "
        "The derived URDF retains that model license and attribution.\n",
        encoding="utf-8",
    )
    print(f"Microduck URDF, meshes and walking policy saved to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    download_assets(parser.parse_args().asset_dir)
