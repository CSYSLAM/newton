# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Blender-authored rice-cooker visuals driven by live MJMPM state."""

import json
from pathlib import Path

import numpy as np
import warp as wp

from newton.viewer import ViewerGL


@wp.kernel
def _rice_transforms(
    positions: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
    transforms: wp.array[wp.transform],
):
    i = wp.tid()
    transforms[i] = wp.transform(positions[i], rotations[i])


class RiceCookerVisuals:
    """Instance visual meshes without adding mass or collision shapes to the model."""

    def __init__(self, viewer, *, device, cooker_pose, rice_count, batch_opaque=True, prefix="/rice_cooker_visuals"):
        self.viewer = viewer
        self.device = device
        self.parts = []
        self.cooker_pose = wp.array([cooker_pose], dtype=wp.transform, device=device)
        self.single_scale = wp.full(1, wp.vec3(1.0), dtype=wp.vec3, device=device)
        path = Path(__file__).resolve().parents[1] / "assets/rice_cooker/realistic_visuals.npz"
        with np.load(path, allow_pickle=False) as archive:
            groups = []
            for part in json.loads(str(archive["manifest"])):
                key = part["key"]
                material_key = (
                    part["parent"],
                    tuple(part["color"]),
                    part["roughness"],
                    part["metallic"],
                    part["opacity"],
                )
                # Merge only adjacent opaque parts, preserving draw order even
                # where different materials meet at coincident triangle edges.
                group = (
                    groups[-1]
                    if batch_opaque and part["opacity"] == 1.0 and groups and groups[-1]["material_key"] == material_key
                    else None
                )
                points = archive[key + "_points"]
                indices = archive[key + "_indices"]
                normals = archive[key + "_normals"]
                if group is None:
                    group = {
                        "part": part,
                        "material_key": material_key,
                        "points": [],
                        "indices": [],
                        "normals": [],
                        "vertices": 0,
                    }
                    groups.append(group)
                group["points"].append(points)
                group["indices"].append(indices + group["vertices"])
                group["normals"].append(normals)
                group["vertices"] += len(points)
            for group in groups:
                part = group["part"]
                key = part["key"]
                name = f"{prefix}/{key}"
                viewer.log_mesh(
                    name,
                    wp.array(np.concatenate(group["points"]), dtype=wp.vec3, device=device),
                    wp.array(np.concatenate(group["indices"]), dtype=wp.int32, device=device),
                    normals=wp.array(np.concatenate(group["normals"]).astype(np.float32), dtype=wp.vec3, device=device),
                    hidden=True,
                    backface_culling=False,
                    color=tuple(part["color"]),
                    roughness=part["roughness"],
                    metallic=part["metallic"],
                    opacity=part["opacity"],
                )
                opacity = wp.full(1, part["opacity"], dtype=float, device=device)
                color = wp.full(1, wp.vec3(*part["color"]), dtype=wp.vec3, device=device)
                material = wp.full(
                    1, wp.vec4(part["roughness"], part["metallic"], 0.0, 0.0), dtype=wp.vec4, device=device
                )
                self.parts.append((name, part["parent"], opacity, color, material))

        random = np.random.default_rng(47)
        rotations = random.normal(size=(rice_count, 4))
        rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
        self.rotations = wp.array(rotations, dtype=wp.quat, device=device)
        self.rice_poses = wp.empty(rice_count, dtype=wp.transform, device=device)
        sizes = random.uniform(0.75, 0.90, size=(rice_count, 1)) * np.ones((rice_count, 3))
        self.rice_scales = wp.array(sizes, dtype=wp.vec3, device=device)
        colors = random.uniform(0.90, 1.0, size=(rice_count, 1)) * np.array([[0.97, 0.95, 0.86]])
        self.rice_colors = wp.array(colors, dtype=wp.vec3, device=device)
        self.rice_materials = wp.full(rice_count, wp.vec4(0.38, 0.0, 0.0, 0.0), dtype=wp.vec4, device=device)
        self._static_logged = False
        # ViewerGL preserves instance attributes when their update arrays are
        # omitted. These materials and opacities are immutable in this scene.
        self._reuse_instance_properties = isinstance(viewer, ViewerGL)

    def render(self, *, cup_pose, rice_positions):
        """Draw the cup at its rigid pose and one grain at each MPM sample position."""
        wp.launch(
            _rice_transforms,
            dim=rice_positions.shape[0],
            inputs=[rice_positions, self.rotations],
            outputs=[self.rice_poses],
            device=self.device,
        )
        for name, parent, opacity, color, material in self.parts:
            if parent == "cooker" and self._static_logged:
                continue
            if parent == "grain":
                poses, scales, colors, opacities = self.rice_poses, self.rice_scales, self.rice_colors, None
                materials = self.rice_materials
            else:
                poses = cup_pose if parent == "cup" else self.cooker_pose
                scales, colors, opacities, materials = self.single_scale, color, opacity, material
            if self._reuse_instance_properties and self._static_logged:
                colors, materials, opacities = None, None, None
            self.viewer.log_instances(name + "/instances", name, poses, scales, colors, materials, opacities=opacities)
        self._static_logged = True
