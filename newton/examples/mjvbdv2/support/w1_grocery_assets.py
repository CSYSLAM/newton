# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read the scale-aligned grocery USDs without loading their textures."""

import hashlib
from pathlib import Path

import numpy as np

import newton


def load_grocery(root: Path, name: str, *, scale: float = 1.0):
    """Load meter-scaled visual meshes, authored collision hulls and mass.

    Vertices are centered on the combined bounds. The returned source-to-body
    transform keeps the original textured USD aligned with recorded body poses.
    Uniform scaling preserves density by scaling the authored mass cubically.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Grocery scale must be finite and positive")
    path = root.expanduser().resolve() / name / f"{name}.usd"
    if not path.is_file():
        raise FileNotFoundError(f"Missing grocery asset: {path}; set --grocery-assets to the USD bundle root")
    stage = Usd.Stage.Open(str(path))
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(f"Expected a Z-up grocery asset: {path}")
    meters = UsdGeom.GetStageMetersPerUnit(stage) * scale
    transforms = UsdGeom.XformCache()
    parts, masses = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            masses.append(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
            raise ValueError(f"Expected triangulated grocery mesh: {prim.GetPath()}")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        vertices = (points @ matrix[:3, :3] + matrix[3, :3]) * meters
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        if (mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded) != (np.linalg.det(matrix[:3, :3]) < 0):
            faces = faces[:, ::-1]
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        color, opacity = (0.7, 0.7, 0.7), 1.0
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            shader, _, _ = material.ComputeSurfaceSource()
            if shader:
                value = shader.GetInput("diffuseColor").Get()
                if value is not None:
                    color = tuple(value)
                if shader.GetInput("diffuseColor").HasConnectedSource():
                    color = {
                        "toy": (0.7, 0.7, 0.7),
                        "soda": (0.83, 0.57, 0.018),
                        "biscuit": (0.21, 0.65, 0.015),
                        "glue": (0.85, 0.90, 0.95),
                        "water_bottle": (0.10, 0.45, 0.80),
                    }[name]
                value = shader.GetInput("opacity").Get()
                if value is not None:
                    opacity = float(value)
        # The source toy uses one texture atlas; distinguish its pieces without it.
        if name == "toy" and "piece_" in prim.GetName():
            row = int(prim.GetName().split("_")[-2])
            color = ((0.55, 0.25, 0.72), (0.95, 0.36, 0.10), (0.55, 0.83, 0.56))[row]
        parts.append(
            {
                "path": str(prim.GetPath()),
                "vertices": vertices,
                "faces": faces,
                "collision": collision,
                "color": color,
                "opacity": opacity,
            }
        )
    if len(masses) != 1 or not np.isfinite(masses[0]) or masses[0] <= 0:
        raise ValueError(f"Expected one positive authored rigid-body mass: {path}")
    if not parts or not any(part["collision"] for part in parts):
        raise ValueError(f"Missing grocery mesh or authored collision hull: {path}")
    vertices = np.concatenate([part["vertices"] for part in parts])
    center = (vertices.min(0) + vertices.max(0)) / 2
    half = (vertices.max(0) - vertices.min(0)) / 2
    for part in parts:
        part["mesh"] = newton.Mesh(
            (part.pop("vertices") - center).astype(np.float32),
            part.pop("faces").ravel(),
            compute_inertia=part["collision"],
        )
    source_to_body = np.eye(4)
    source_to_body[:3, :3] *= meters
    source_to_body[:3, 3] = -center
    return {
        "name": name,
        "parts": parts,
        "half": half,
        "mass": masses[0] * scale**3,
        "source": {
            "usd": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scale": scale,
            "sourceToBody": source_to_body.tolist(),
        },
    }
