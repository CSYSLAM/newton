# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Blender skins embedded in the live MPM material, with no visual animation."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree


@wp.kernel
def _skin(
    particles: wp.array[wp.vec3],
    ids: wp.array2d[int],
    coefficients: wp.array2d[wp.vec4],
    rest_normals: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    position = wp.vec3(0.0)
    f0 = wp.vec3(0.0)
    f1 = wp.vec3(0.0)
    f2 = wp.vec3(0.0)
    for k in range(ids.shape[1]):
        p = particles[ids[i, k]]
        c = coefficients[i, k]
        position += c[0] * p
        f0 += c[1] * p
        f1 += c[2] * p
        f2 += c[3] * p
    n = rest_normals[i]
    normal = n[0] * wp.cross(f1, f2) + n[1] * wp.cross(f2, f0) + n[2] * wp.cross(f0, f1)
    points[i] = position
    normals[i] = wp.normalize(normal)


@wp.kernel
def _tool_pose(body_q: wp.array[wp.transform], body: int, offset: wp.transform, pose: wp.array[wp.transform]):
    pose[0] = wp.mul(body_q[body], offset)


@wp.kernel
def _expose_crumb(
    damage: wp.array[float],
    ids: wp.array2d[int],
    coefficients: wp.array2d[wp.vec4],
    rest_uv: wp.array[wp.vec2],
    uv: wp.array[wp.vec2],
):
    i = wp.tid()
    damaged = float(0.0)
    total = float(0.0)
    for k in range(ids.shape[1]):
        weight = wp.max(coefficients[i, k][0], 0.0)
        damaged += weight * damage[ids[i, k]]
        total += weight
    # Damage near the skin tears away the crust and exposes porous crumb.
    # This changes colour only; all surface positions still come from MPM.
    half = wp.where(damaged > 0.35 * total, 0.5, 0.0)
    original = rest_uv[i]
    uv[i] = wp.vec2(half + 0.5 * wp.clamp(original[0], 0.001, 0.999), original[1])


def embedding(points, rest, candidates, *, neighbors=16):
    """Affine-exact local MLS embedding; never connects across the skin patch."""
    distance, local = cKDTree(rest[candidates]).query(points, k=neighbors)
    ids = candidates[local]
    offsets = rest[ids] - points[:, None, :]
    design = np.concatenate((np.ones((*offsets.shape[:2], 1)), offsets), axis=2)
    weights = 1.0 / (distance**2 + 0.003**2)
    weighted = design * weights[:, :, None]
    gram = np.einsum("vki,vkj->vij", design, weighted)
    coefficients = np.einsum("vki,vij->vkj", weighted, np.linalg.pinv(gram, rcond=1.0e-12))
    return ids.astype(np.int32), coefficients.astype(np.float32)


class BreadCutVisuals:
    """Render authored board/tool meshes and deform the loaf skin from particles.

    The demonstrated plane has separate surface embedding patches. Material
    remains a single uncut particle continuum at initialization. This visual
    topology is specific to this example; it is not arbitrary mesh cutting.
    """

    def __init__(self, viewer, *, device, rest, center, cut_y, tool_body, tool_local, render_bread=False):
        from PIL import Image  # noqa: PLC0415

        self.viewer = viewer
        self.device = device
        self.tool_body, self.tool_local = tool_body, tool_local
        self.tool_pose = wp.empty(1, dtype=wp.transform, device=device)
        self.scale = wp.full(1, wp.vec3(1.0), dtype=wp.vec3, device=device)
        self.board_pose = wp.array(
            [wp.transform(wp.vec3(*center), wp.quat_identity())], dtype=wp.transform, device=device
        )
        self.identity = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.parts, self.skins, self.crust_uvs = [], [], []
        path = Path(__file__).resolve().parents[1] / "assets/bread_cut"
        with np.load(path / "visuals.npz", allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"]))
        manifest = [part for part in manifest if render_bread or part["parent"] != "bread"]
        names = {part["texture"] for part in manifest if part["texture"]} | {"crust.jpg", "crumb.jpg"}
        textures = {name: np.array(Image.open(path / name).convert("RGB")) for name in names}
        crust_atlas = np.concatenate((textures["crust.jpg"], textures["crumb.jpg"]), axis=1)
        with np.load(path / "visuals.npz", allow_pickle=False) as archive:
            for part in manifest:
                key = part["key"]
                points = archive[key + "_points"].copy()
                normals = archive[key + "_normals"].astype(np.float32)
                indices = wp.array(archive[key + "_indices"], dtype=int, device=device)
                name = "/bread_cut/" + key
                skin = part["parent"] == "bread"
                if skin:
                    points += center
                vertices = wp.array(points, dtype=wp.vec3, device=device)
                vertex_normals = wp.array(normals, dtype=wp.vec3, device=device)
                if skin:
                    candidates = np.flatnonzero((rest[:, 1] < cut_y) if part["side"] < 0 else (rest[:, 1] >= cut_y))
                    ids, coefficients = embedding(points, rest, candidates)
                    self.skins.append(
                        (
                            vertices,
                            vertex_normals,
                            wp.clone(vertex_normals),
                            wp.array(ids, dtype=int, device=device),
                            wp.array(coefficients, dtype=wp.vec4, device=device),
                        )
                    )
                color = wp.full(
                    1, wp.vec3(*((1.0, 1.0, 1.0) if part["texture"] else part["color"])), dtype=wp.vec3, device=device
                )
                material = wp.full(
                    1,
                    wp.vec4(part["roughness"], part["metallic"], 0.0, float(bool(part["texture"]))),
                    dtype=wp.vec4,
                    device=device,
                )
                uv = wp.array(archive[key + "_uvs"], dtype=wp.vec2, device=device)
                texture = textures[part["texture"]] if part["texture"] else None
                if skin and part["texture"] == "crust.jpg":
                    self.crust_uvs.append((self.skins[-1][3], self.skins[-1][4], wp.clone(uv), uv))
                    texture = crust_atlas
                self.parts.append(
                    (
                        name,
                        part,
                        vertices,
                        indices,
                        vertex_normals,
                        uv,
                        color,
                        material,
                        texture,
                    )
                )
        self.logged = False

    def render(self, state, damage):
        wp.launch(
            _tool_pose,
            dim=1,
            inputs=[state.body_q, self.tool_body, self.tool_local, self.tool_pose],
            device=self.device,
        )
        for vertices, normals, rest_normals, ids, coefficients in self.skins:
            wp.launch(
                _skin,
                dim=len(vertices),
                inputs=[state.particle_q, ids, coefficients, rest_normals, vertices, normals],
                device=self.device,
            )
        for ids, coefficients, rest_uv, uv in self.crust_uvs:
            wp.launch(_expose_crumb, dim=len(uv), inputs=[damage, ids, coefficients, rest_uv, uv], device=self.device)
        for name, part, vertices, indices, normals, uv, color, material, texture in self.parts:
            skin = part["parent"] == "bread"
            if skin or not self.logged:
                self.viewer.log_mesh(
                    name,
                    vertices,
                    indices,
                    normals=normals,
                    uvs=uv,
                    texture=texture,
                    color=tuple(part["color"]),
                    roughness=part["roughness"],
                    metallic=part["metallic"],
                    backface_culling=False,
                    hidden=True,
                )
            if part["parent"] == "knife":
                pose = self.tool_pose
            elif skin:
                pose = self.identity
            else:
                pose = self.board_pose
            if not self.logged or part["parent"] == "knife":
                self.viewer.log_instances(name + "/instance", name, pose, self.scale, color, material)
        self.logged = True
