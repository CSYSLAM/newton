# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the W1 conveyor's continuous belt and synchronized end drums."""

import math
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def _advance_belt_uv(base: wp.array[wp.vec2], phase: float, result: wp.array[wp.vec2]):
    i = wp.tid()
    # Path distance increases toward the picking station on the upper run.
    # Subtract travel to transport a fixed material feature in that direction.
    result[i] = base[i] - wp.vec2(0.0, phase)


class ConveyorBeltVisual:
    """Animate authored belt surfaces from the collision belt's travel [m]."""

    def __init__(self, asset_directory: Path, device):
        self.parts = []
        with np.load(asset_directory / "belt.npz", allow_pickle=False) as asset:
            self.path_length = float(asset["path_length"])
            self.radius = float(asset["radius"])
        for kind in ("belt", "roller"):
            with np.load(asset_directory / f"{kind}.npz", allow_pickle=False) as asset:
                for i, material in enumerate(asset["materials"]):
                    if not len(asset[f"indices_{i}"]):
                        continue
                    color = (1.0, 1.0, 1.0) if kind == "belt" else material[:3]
                    roughness, metallic = (0.84, 0.0) if kind == "belt" else material[3:5]
                    count = 1 if kind == "belt" else 2
                    self.parts.append(
                        (
                            kind,
                            i,
                            wp.array(asset[f"vertices_{i}"], dtype=wp.vec3, device=device),
                            wp.array(asset[f"indices_{i}"], dtype=int, device=device),
                            wp.array(asset[f"normals_{i}"], dtype=wp.vec3, device=device),
                            wp.array(asset[f"uvs_{i}"], dtype=wp.vec2, device=device),
                            wp.array([wp.vec3(*color)] * count, dtype=wp.vec3, device=device),
                            wp.array(
                                [wp.vec4(roughness, metallic, 0.0, float(kind == "belt"))] * count,
                                dtype=wp.vec4,
                                device=device,
                            ),
                        )
                    )
        self.belt_uv = wp.empty_like(self.parts[0][5])
        self.belt_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.roller_transforms = wp.empty(2, dtype=wp.transform, device=device)
        self.belt_scale = wp.array([wp.vec3(1.0)], dtype=wp.vec3, device=device)
        self.roller_scale = wp.array([wp.vec3(1.0)] * 2, dtype=wp.vec3, device=device)

        # Full-circumference texture: one splice, no repeated transverse slats.
        # The fine stipple and longitudinal wear are deliberately low contrast.
        rng = np.random.default_rng(17)
        yy, xx = np.indices((4096, 512))
        grain = rng.normal(0, 0.8, (4096, 512)) + rng.normal(0, 0.5, (1, 512))
        wear = 2.0 * np.exp(-(((xx - 145) / 60) ** 2)) + 1.5 * np.exp(-(((xx - 365) / 45) ** 2))
        rubber = 43.0 + grain + wear
        rubber[(xx < 5) | (xx > 506)] *= 0.78
        splice = np.abs(yy - (1100 + 6 * np.sin(xx * 0.12))) < 1
        rubber[splice] *= 0.70
        rubber[(xx > 9) & (xx < 13) & (yy > 1085) & (yy < 1115)] = 105
        self.texture = np.uint8(np.clip(np.stack((rubber * 0.92, rubber, rubber * 0.99), axis=-1), 0, 255))

    def render(self, viewer, travel: float):
        """Keep material advection and drum rotation synchronized, including stops."""
        phase = (travel / self.path_length) % 1.0
        angle = (travel / self.radius) % math.tau
        rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), angle)
        self.roller_transforms.assign(
            np.asarray([tuple(wp.transform(wp.vec3(0.6, y, 0.708), rotation)) for y in (-0.24, 1.76)], dtype=np.float32)
        )
        for kind, index, points, indices, normals, base_uv, colors, materials in self.parts:
            if kind == "belt":
                wp.launch(_advance_belt_uv, len(base_uv), [base_uv, phase, self.belt_uv], device=base_uv.device)
            name = f"/station/{kind}_{index}"
            viewer.log_mesh(
                name,
                points,
                indices,
                normals=normals,
                uvs=self.belt_uv if kind == "belt" else None,
                texture=self.texture if kind == "belt" else None,
                hidden=True,
            )
            viewer.log_instances(
                f"{name}/instances",
                name,
                self.belt_transform if kind == "belt" else self.roller_transforms,
                self.belt_scale if kind == "belt" else self.roller_scale,
                colors,
                materials,
            )
