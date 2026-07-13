# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice in the full WAIC house scene
#
# This is a thin wrapper around ``example_mpm_w1_burger_slice_waic_kitchen_V4``:
# the W1 pose, simplified pan collision, meat, knife, IK script, and carry
# sequence stay identical to V4. The only scene change is visual composition:
# load the complete WAIC house USD as a non-colliding background and load the
# Blender frying-pan visual as a mesh attached to the V4 pan body.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_house
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton.examples.cloth import example_cloth_dexforce_bimanual_fold_tshirt_waic_house as fold_house
from newton.examples.mpm import example_mpm_w1_burger_slice_waic_kitchen_V4 as v4


HOUSE_VISUAL_USD = (
    r"E:\csy_work\CG\assets\WAIC\house_background"
    r"\House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
)
PAN_VISUAL_USD = r"E:\csy_work\CG\assets\WAIC\house_background\waic_frying_pan.usd"


class Example(v4.Example):
    """V4 burger slicing physics placed inside the complete WAIC house visual."""

    _usd_mesh_to_newton_arrays = staticmethod(fold_house.Example._usd_mesh_to_newton_arrays)
    _usd_mesh_color = fold_house.Example._usd_mesh_color
    _material_texture_average_color = fold_house.Example._material_texture_average_color
    _average_texture_rgb = staticmethod(fold_house.Example._average_texture_rgb)
    _material_diffuse_color = staticmethod(fold_house.Example._material_diffuse_color)
    _semantic_house_color = staticmethod(fold_house.Example._semantic_house_color)

    def __init__(self, viewer, args):
        self.house_visual_usd = str(args.house_visual_usd)
        self.pan_visual_usd = str(args.pan_visual_usd)
        self.pan_visual_axis_mode = str(args.pan_visual_axis_mode)
        self.pan_visual_offset = wp.vec3(
            float(args.pan_visual_offset_x),
            float(args.pan_visual_offset_y),
            float(args.pan_visual_offset_z),
        )
        self.hide_house_visual = bool(args.hide_house_visual)
        self.house_visual_color_mode = str(args.house_visual_color_mode)
        self._house_material_color_cache: dict[str, tuple[float, float, float]] = {}
        super().__init__(viewer, args)
        self._attach_house_visual_stage()

    def _add_waic_visual_model(self, builder: newton.ModelBuilder) -> None:
        self._add_house_visual_model(builder)
        self._add_pan_visual_model(builder)

    def _add_simplified_pan(self, builder: newton.ModelBuilder) -> None:
        pan_cfg = newton.ModelBuilder.ShapeConfig()
        pan_cfg.ke = 8.0e5
        pan_cfg.kd = 1.0e-6
        pan_cfg.mu = 0.9
        pan_cfg.density = 0.0
        pan_cfg.has_particle_collision = True
        pan_cfg.is_visible = True

        self.pan_body = builder.add_body(
            xform=v4.wp.transform(self.pan_center, self.scene_rotation),
            label="waic_simplified_pan_body",
        )
        builder.add_shape_cylinder(
            body=self.pan_body,
            xform=v4.wp.transform_identity(),
            radius=v4.PAN_RADIUS,
            half_height=v4.PAN_DISK_HALF_HEIGHT,
            cfg=pan_cfg,
            color=(1.0, 0.25, 0.05),
            label="waic_visible_pan_collision_disk",
        )
        builder.add_shape_cylinder(
            body=self.pan_body,
            xform=v4.wp.transform(
                v4.PAN_HANDLE_LOCAL_POS,
                v4.wp.quat(0.0, 0.7071067690849304, 0.0, 0.7071067690849304),
            ),
            radius=v4.PAN_HANDLE_RADIUS,
            half_height=0.5 * v4.PAN_HANDLE_LENGTH,
            cfg=pan_cfg,
            color=(0.0, 0.85, 1.0),
            label="waic_visible_pan_collision_handle",
        )

    def _add_house_visual_model(self, builder: newton.ModelBuilder) -> None:
        if self.hide_house_visual or not self.house_visual_usd:
            return

        # USD viewers can reference the full stage directly after model setup.
        # GL/null viewers need the meshes baked into the Newton model.
        if hasattr(self.viewer, "stage"):
            return

        house_path = self._resolved_house_visual_path()
        original_visual_usd = self.waic_visual_usd
        try:
            self.waic_visual_usd = house_path
            fold_house.Example._add_house_visual_model(self, builder)
        finally:
            self.waic_visual_usd = original_visual_usd

    def _add_pan_visual_model(self, builder: newton.ModelBuilder) -> None:
        if not self.pan_visual_usd:
            return
        if not os.path.exists(self.pan_visual_usd):
            print(f"WAIC pan visual USD not found, skipping: {self.pan_visual_usd}")
            return

        stage = Usd.Stage.Open(self.pan_visual_usd)
        if stage is None:
            print(f"WAIC pan visual USD failed to open, skipping: {self.pan_visual_usd}")
            return

        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.density = 0.0
        visual_cfg.collision_group = 0
        visual_cfg.has_shape_collision = False
        visual_cfg.has_particle_collision = False
        visual_cfg.is_visible = True

        xform_cache = UsdGeom.XformCache()
        mesh_count = 0
        tri_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            prim_path = str(prim.GetPath())
            usd_mesh = UsdGeom.Mesh(prim)
            vertices, indices = self._usd_mesh_to_newton_arrays(usd_mesh, xform_cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue

            vertices = self._pan_visual_vertices_to_newton_world(vertices)
            vertices = self._world_vertices_to_pan_body(vertices)
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=self.pan_body,
                mesh=mesh,
                xform=wp.transform_identity(),
                cfg=visual_cfg,
                color=self._semantic_visual_color(prim_path),
                label=f"waic_pan_visual_{mesh_count:03d}",
            )
            mesh_count += 1
            tri_count += int(len(indices) // 3)

        print(
            f"Loaded WAIC pan visual: {mesh_count} meshes, {tri_count} triangles from {self.pan_visual_usd}, "
            f"axis_mode={self.pan_visual_axis_mode}, offset="
            f"({float(self.pan_visual_offset[0]):.4f}, {float(self.pan_visual_offset[1]):.4f}, "
            f"{float(self.pan_visual_offset[2]):.4f})"
        )

    def _pan_visual_vertices_to_newton_world(self, vertices: np.ndarray) -> np.ndarray:
        if self.pan_visual_axis_mode == "blender_usd":
            converted = np.empty_like(vertices)
            converted[:, 0] = vertices[:, 0]
            converted[:, 1] = vertices[:, 2]
            converted[:, 2] = -vertices[:, 1]
        elif self.pan_visual_axis_mode == "identity":
            converted = vertices.copy()
        else:
            raise ValueError(f"Unknown pan visual axis mode: {self.pan_visual_axis_mode}")

        converted[:, 0] += float(self.pan_visual_offset[0])
        converted[:, 1] += float(self.pan_visual_offset[1])
        converted[:, 2] += float(self.pan_visual_offset[2])
        return converted

    def _attach_house_visual_stage(self) -> None:
        if self.hide_house_visual or not hasattr(self.viewer, "stage"):
            return

        house_path = self._resolved_house_visual_path()
        if not os.path.exists(house_path):
            print(f"WAIC house visual USD not found, skipping background: {house_path}")
            return

        stage = self.viewer.stage
        prim = stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(house_path))

    def _resolved_house_visual_path(self) -> str:
        if os.path.exists(self.house_visual_usd):
            return self.house_visual_usd
        fallback = fold_house.HOUSE_VISUAL_USD
        if self.house_visual_usd != fallback:
            print(f"WAIC full house USD not found, falling back to: {fallback}")
        return fallback

    @staticmethod
    def create_parser():
        parser = v4.Example.create_parser()
        parser.set_defaults(waic_visual_usd=PAN_VISUAL_USD)
        parser.add_argument(
            "--house-visual-usd",
            type=str,
            default=HOUSE_VISUAL_USD,
            help="Complete WAIC house USD visual background. Visual only; no collision.",
        )
        parser.add_argument(
            "--pan-visual-usd",
            type=str,
            default=PAN_VISUAL_USD,
            help="Frying-pan USD visual mesh attached to the simplified pan body from V4.",
        )
        parser.add_argument(
            "--pan-visual-axis-mode",
            choices=("blender_usd", "identity"),
            default="blender_usd",
            help="Axis conversion for the pan USD. blender_usd maps USD (x,y,z) to Newton/Blender (x,z,-y).",
        )
        parser.add_argument("--pan-visual-offset-x", type=float, default=0.0)
        parser.add_argument("--pan-visual-offset-y", type=float, default=0.0)
        parser.add_argument("--pan-visual-offset-z", type=float, default=0.0)
        parser.add_argument(
            "--hide-house-visual",
            action="store_true",
            help="Skip loading the complete WAIC house visual background.",
        )
        parser.add_argument(
            "--house-visual-color-mode",
            choices=("texture", "material", "semantic", "gray"),
            default="texture",
            help="Color mode for the GL house background.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
