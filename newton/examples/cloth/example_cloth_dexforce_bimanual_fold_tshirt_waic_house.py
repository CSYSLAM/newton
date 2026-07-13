# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual Fold T-Shirt in the WAIC house scene
#
# Reuses the captured W1 bimanual T-shirt folding trajectory from
# ``example_cloth_dexforce_bimanual_fold_tshirt_v2`` and applies only a
# world-space translation so the whole robot/table/shirt/script lands at the
# dining table location from the WAIC Blender house scene.
#
# Command:
#   python -m newton.examples cloth_dexforce_bimanual_fold_tshirt_waic_house
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdShade

import newton
import newton.examples
import newton.usd
from newton.examples.cloth import example_cloth_dexforce_bimanual_fold_tshirt_v2 as fold_v2
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_tshirt import (
    SHIRT_ASSET,
    SHIRT_COLLISION_RADIUS,
    SHIRT_DENSITY,
    SHIRT_PRIM_PATH,
    SHIRT_ROT,
    TABLE_COLOR,
    TABLE_HALF_EXTENTS,
    TABLE_POS,
)


OLD_TABLE_POS = TABLE_POS
OLD_ROBOT_BASE_POS = wp.vec3(0.0, 0.0, 0.0)
WAIC_ROBOT_BASE_POS = wp.vec3(-0.3493143916130066, -3.246695160865784, -0.0037720240652561188)
WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.7071067690849304, 0.7071067690849304)
WAIC_TABLE_CENTER_X = None
WAIC_TABLE_CENTER_Y = None
WAIC_TABLE_TOP_Z = None
SCENE_OFFSET = wp.vec3(
    float(WAIC_ROBOT_BASE_POS[0]) - float(OLD_ROBOT_BASE_POS[0]),
    float(WAIC_ROBOT_BASE_POS[1]) - float(OLD_ROBOT_BASE_POS[1]),
    float(WAIC_ROBOT_BASE_POS[2]) - float(OLD_ROBOT_BASE_POS[2]),
)

PHYSICS_TABLE_HIDDEN_COLOR = (0.02, 0.025, 0.03)
HOUSE_VISUAL_USD = r"E:\csy_work\CG\assets\WAIC\house_background\House5_Simple2_visual.usd"
CAMERA_POS = wp.vec3(2.024085283279419, -5.666704177856445, 2.0068864822387695)
CAMERA_PITCH = -20.622435706293572
CAMERA_YAW = 126.25384166077777
CAMERA_LENS_MM = 35.0
CAMERA_RIGHT = (0.8064049482345581, 0.591363787651062, 5.51955601224563e-08)
CAMERA_UP = (-0.2082831859588623, 0.28402236104011536, 0.9359216690063477)
CAMERA_BACK = (0.5534701347351074, -0.7547318935394287, 0.3522081673145294)


class Example(fold_v2.Example):
    def __init__(self, viewer, args):
        self.waic_robot_base_pos = wp.vec3(
            float(args.waic_robot_base_x),
            float(args.waic_robot_base_y),
            float(args.waic_robot_base_z),
        )
        self.scene_rotation = self._normalize_wp_quat(
            wp.quat(
                float(args.waic_robot_base_qx),
                float(args.waic_robot_base_qy),
                float(args.waic_robot_base_qz),
                float(args.waic_robot_base_qw),
            )
        )
        self.scene_offset = wp.vec3(
            float(self.waic_robot_base_pos[0]) - float(OLD_ROBOT_BASE_POS[0]),
            float(self.waic_robot_base_pos[1]) - float(OLD_ROBOT_BASE_POS[1]),
            float(self.waic_robot_base_pos[2]) - float(OLD_ROBOT_BASE_POS[2]),
        )
        table_pos = self._offset_vec3(OLD_TABLE_POS)
        self.waic_table_pos = wp.vec3(
            float(args.waic_table_center_x) if args.waic_table_center_x is not None else float(table_pos[0]),
            float(args.waic_table_center_y) if args.waic_table_center_y is not None else float(table_pos[1]),
            float(args.waic_table_top_z) - TABLE_HALF_EXTENTS[2] if args.waic_table_top_z is not None else float(table_pos[2]),
        )
        self.show_physics_table = bool(args.show_physics_table)
        self.house_visual_usd = str(args.house_visual_usd)
        self.hide_house_visual = bool(args.hide_house_visual)
        self.house_visual_color_mode = str(args.house_visual_color_mode)
        self._house_material_color_cache: dict[str, tuple[float, float, float]] = {}
        super().__init__(viewer, args)
        self._attach_house_visual_stage()
        self._attach_initial_usd_camera()
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)

    def _robot_xform(self) -> wp.transform:
        return wp.transform(self.waic_robot_base_pos, self.scene_rotation)

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.waic_table_pos, self.scene_rotation),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR if self.show_physics_table else PHYSICS_TABLE_HIDDEN_COLOR,
            label="waic_hidden_physics_table",
        )
        builder.add_ground_plane(
            height=float(self.scene_offset[2]),
            label="waic_shifted_ground_plane",
        )
        self._add_house_visual_model(builder)

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        self.particle_radius = SHIRT_COLLISION_RADIUS
        self.soft_contact_margin = fold_v2.SHIRT_SOFT_CONTACT_MARGIN
        self.particle_self_contact_radius = fold_v2.SHIRT_SELF_CONTACT_RADIUS
        self.particle_self_contact_margin = fold_v2.SHIRT_SELF_CONTACT_MARGIN

        usd_stage = Usd.Stage.Open(newton.examples.get_asset(SHIRT_ASSET))
        usd_prim = usd_stage.GetPrimAtPath(SHIRT_PRIM_PATH)
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = self._center_shirt_vertices(shirt_mesh.vertices)

        builder.add_cloth_mesh(
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
            indices=shirt_mesh.indices,
            rot=self._quat_multiply(self.scene_rotation, SHIRT_ROT),
            pos=self._offset_vec3(fold_v2.SHIRT_POS),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=SHIRT_DENSITY,
            scale=1.0,
            tri_ke=fold_v2.TSHIRT_TRI_KE,
            tri_ka=fold_v2.TSHIRT_TRI_KA,
            tri_kd=fold_v2.TSHIRT_TRI_KD,
            edge_ke=fold_v2.TSHIRT_EDGE_KE,
            edge_kd=fold_v2.TSHIRT_EDGE_KD,
            particle_radius=self.particle_radius,
        )

    def _build_motion_segments(self):
        o = self._offset_script_tf
        return (
            (fold_v2.HOME_HOLD_TIME, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (
                fold_v2.APPROACH_TIME,
                self.left_home_tf,
                o(fold_v2.LEFT_APPROACH_TF),
                self.right_home_tf,
                o(fold_v2.RIGHT_APPROACH_TF),
                0.0,
                0.0,
            ),
            (
                fold_v2.DESCEND_TIME,
                o(fold_v2.LEFT_APPROACH_TF),
                o(fold_v2.LEFT_GRASP_TF),
                o(fold_v2.RIGHT_APPROACH_TF),
                o(fold_v2.RIGHT_GRASP_TF),
                0.0,
                0.0,
            ),
            (fold_v2.GRASP_CLOSE_TIME, o(fold_v2.LEFT_GRASP_TF), o(fold_v2.LEFT_GRASP_TF), o(fold_v2.RIGHT_GRASP_TF), o(fold_v2.RIGHT_GRASP_TF), 0.0, 1.0),
            (
                fold_v2.LIFT_TIME,
                o(fold_v2.LEFT_GRASP_TF),
                o(fold_v2.LEFT_LIFT_TF),
                o(fold_v2.RIGHT_GRASP_TF),
                o(fold_v2.RIGHT_LIFT_TF),
                1.0,
                1.0,
            ),
            (
                fold_v2.FOLD_TRAVEL_TIME,
                o(fold_v2.LEFT_LIFT_TF),
                o(fold_v2.LEFT_FOLD_TRAVEL_TF),
                o(fold_v2.RIGHT_LIFT_TF),
                o(fold_v2.RIGHT_FOLD_TRAVEL_TF),
                1.0,
                1.0,
            ),
            (
                fold_v2.FOLD_PLACE_TIME,
                o(fold_v2.LEFT_FOLD_TRAVEL_TF),
                o(fold_v2.LEFT_PLACE_TF),
                o(fold_v2.RIGHT_FOLD_TRAVEL_TF),
                o(fold_v2.RIGHT_PLACE_TF),
                1.0,
                1.0,
            ),
            (fold_v2.RELEASE_TIME, o(fold_v2.LEFT_PLACE_TF), o(fold_v2.LEFT_PLACE_TF), o(fold_v2.RIGHT_PLACE_TF), o(fold_v2.RIGHT_PLACE_TF), 1.0, fold_v2.RELEASE_CRACK_ALPHA),
            (
                fold_v2.RELEASE_LIFT_TIME,
                o(fold_v2.LEFT_PLACE_TF),
                o(fold_v2.LEFT_RELEASE_TF),
                o(fold_v2.RIGHT_PLACE_TF),
                o(fold_v2.RIGHT_RELEASE_TF),
                fold_v2.RELEASE_CRACK_ALPHA,
                0.0,
            ),
            (fold_v2.HOLD_TIME, o(fold_v2.LEFT_RELEASE_TF), o(fold_v2.LEFT_RELEASE_TF), o(fold_v2.RIGHT_RELEASE_TF), o(fold_v2.RIGHT_RELEASE_TF), 0.0, 0.0),
            (
                fold_v2.SECOND_APPROACH_TIME,
                o(fold_v2.LEFT_RELEASE_TF),
                o(fold_v2.LEFT_SECOND_GRASP_TF),
                o(fold_v2.RIGHT_RELEASE_TF),
                o(fold_v2.RIGHT_SECOND_GRASP_TF),
                0.0,
                0.0,
            ),
            (
                fold_v2.SECOND_GRASP_CLOSE_TIME,
                o(fold_v2.LEFT_SECOND_GRASP_TF),
                o(fold_v2.LEFT_SECOND_GRASP_TF),
                o(fold_v2.RIGHT_SECOND_GRASP_TF),
                o(fold_v2.RIGHT_SECOND_GRASP_TF),
                0.0,
                1.0,
            ),
            (
                fold_v2.SECOND_LIFT_TIME,
                o(fold_v2.LEFT_SECOND_GRASP_TF),
                o(fold_v2.LEFT_SECOND_LIFT_TF),
                o(fold_v2.RIGHT_SECOND_GRASP_TF),
                o(fold_v2.RIGHT_SECOND_LIFT_TF),
                1.0,
                1.0,
            ),
            (
                fold_v2.SECOND_FOLD_TRAVEL_TIME,
                o(fold_v2.LEFT_SECOND_LIFT_TF),
                o(fold_v2.LEFT_SECOND_FOLD_TRAVEL_TF),
                o(fold_v2.RIGHT_SECOND_LIFT_TF),
                o(fold_v2.RIGHT_SECOND_FOLD_TRAVEL_TF),
                1.0,
                1.0,
            ),
            (
                fold_v2.SECOND_FOLD_PLACE_TIME,
                o(fold_v2.LEFT_SECOND_FOLD_TRAVEL_TF),
                o(fold_v2.LEFT_SECOND_PLACE_TF),
                o(fold_v2.RIGHT_SECOND_FOLD_TRAVEL_TF),
                o(fold_v2.RIGHT_SECOND_PLACE_TF),
                1.0,
                1.0,
            ),
            (
                fold_v2.SECOND_RELEASE_TIME,
                o(fold_v2.LEFT_SECOND_PLACE_TF),
                o(fold_v2.LEFT_SECOND_PLACE_TF),
                o(fold_v2.RIGHT_SECOND_PLACE_TF),
                o(fold_v2.RIGHT_SECOND_PLACE_TF),
                1.0,
                fold_v2.RELEASE_CRACK_ALPHA,
            ),
            (
                fold_v2.SECOND_RELEASE_LIFT_TIME,
                o(fold_v2.LEFT_SECOND_PLACE_TF),
                o(fold_v2.LEFT_SECOND_RELEASE_TF),
                o(fold_v2.RIGHT_SECOND_PLACE_TF),
                o(fold_v2.RIGHT_SECOND_RELEASE_TF),
                fold_v2.RELEASE_CRACK_ALPHA,
                0.0,
            ),
            (
                fold_v2.RETURN_INITIAL_TIME,
                o(fold_v2.LEFT_SECOND_RELEASE_TF),
                o(fold_v2.LEFT_APPROACH_TF),
                o(fold_v2.RIGHT_SECOND_RELEASE_TF),
                o(fold_v2.RIGHT_APPROACH_TF),
                0.0,
                0.0,
            ),
            (fold_v2.HOLD_TIME, o(fold_v2.LEFT_APPROACH_TF), o(fold_v2.LEFT_APPROACH_TF), o(fold_v2.RIGHT_APPROACH_TF), o(fold_v2.RIGHT_APPROACH_TF), 0.0, 0.0),
        )

    def _offset_vec3(self, value: wp.vec3) -> wp.vec3:
        rotated = self._quat_rotate_vec3(self.scene_rotation, value)
        return wp.vec3(
            float(rotated[0]) + float(self.waic_robot_base_pos[0]),
            float(rotated[1]) + float(self.waic_robot_base_pos[1]),
            float(rotated[2]) + float(self.waic_robot_base_pos[2]),
        )

    def _offset_script_tf(self, tf: wp.transform) -> wp.transform:
        return wp.transform(
            self._offset_vec3(wp.transform_get_translation(tf)),
            self._quat_multiply(self.scene_rotation, wp.transform_get_rotation(tf)),
        )

    @staticmethod
    def _normalize_wp_quat(q: wp.quat) -> wp.quat:
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        length = (x * x + y * y + z * z + w * w) ** 0.5
        if length == 0.0:
            return wp.quat_identity()
        return wp.quat(x / length, y / length, z / length, w / length)

    @staticmethod
    def _quat_multiply(a: wp.quat, b: wp.quat) -> wp.quat:
        ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _quat_rotate_vec3(q: wp.quat, v: wp.vec3) -> wp.vec3:
        qv = wp.quat(float(v[0]), float(v[1]), float(v[2]), 0.0)
        qc = wp.quat(-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))
        rotated = Example._quat_multiply(Example._quat_multiply(q, qv), qc)
        return wp.vec3(float(rotated[0]), float(rotated[1]), float(rotated[2]))

    def _attach_house_visual_stage(self) -> None:
        if self.hide_house_visual or not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.exists(self.house_visual_usd):
            print(f"WAIC house visual USD not found, skipping background: {self.house_visual_usd}")
            return

        stage = self.viewer.stage
        prim = stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    def _add_house_visual_model(self, builder: newton.ModelBuilder) -> None:
        if self.hide_house_visual or not self.house_visual_usd or hasattr(self.viewer, "stage"):
            return
        if not os.path.exists(self.house_visual_usd):
            print(f"WAIC house visual USD not found, skipping GL background: {self.house_visual_usd}")
            return

        stage = Usd.Stage.Open(self.house_visual_usd)
        if stage is None:
            print(f"WAIC house visual USD failed to open, skipping GL background: {self.house_visual_usd}")
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

            usd_mesh = UsdGeom.Mesh(prim)
            vertices, indices = self._usd_mesh_to_newton_arrays(usd_mesh, xform_cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue

            color = self._usd_mesh_color(usd_mesh)
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=-1,
                mesh=mesh,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                cfg=visual_cfg,
                color=color,
                label=f"waic_house_visual_{mesh_count:04d}",
            )
            mesh_count += 1
            tri_count += int(len(indices) // 3)

        print(
            f"Loaded WAIC house visual for GL viewer: {mesh_count} meshes, {tri_count} triangles, "
            f"color_mode={self.house_visual_color_mode}"
        )

    @staticmethod
    def _usd_mesh_to_newton_arrays(usd_mesh: UsdGeom.Mesh, xform_cache: UsdGeom.XformCache):
        points = usd_mesh.GetPointsAttr().Get()
        face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)

        world_from_local = xform_cache.GetLocalToWorldTransform(usd_mesh.GetPrim())
        vertices = np.empty((len(points), 3), dtype=np.float32)
        for i, point in enumerate(points):
            p = world_from_local.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            vertices[i] = (float(p[0]), float(p[1]), float(p[2]))

        triangles: list[int] = []
        cursor = 0
        for count in face_counts:
            count = int(count)
            if count >= 3:
                face = face_indices[cursor : cursor + count]
                for i in range(1, count - 1):
                    triangles.extend((int(face[0]), int(face[i]), int(face[i + 1])))
            cursor += count

        return vertices, np.asarray(triangles, dtype=np.int32)

    def _usd_mesh_color(self, usd_mesh: UsdGeom.Mesh) -> tuple[float, float, float]:
        if self.house_visual_color_mode == "gray":
            return (0.62, 0.62, 0.58)

        display_color = usd_mesh.GetDisplayColorAttr().Get()
        if display_color:
            color = display_color[0]
            return (float(color[0]), float(color[1]), float(color[2]))

        material = UsdShade.MaterialBindingAPI(usd_mesh.GetPrim()).ComputeBoundMaterial()[0]
        material_key = str(material.GetPath()) if material else str(usd_mesh.GetPrim().GetPath())
        if material_key in self._house_material_color_cache:
            return self._house_material_color_cache[material_key]

        color = None
        if material and self.house_visual_color_mode == "texture":
            color = self._material_texture_average_color(material)
        if material and color is None and self.house_visual_color_mode in {"texture", "material"}:
            color = self._material_diffuse_color(material)
        if color is None:
            color = self._semantic_house_color(str(usd_mesh.GetPrim().GetPath()), material_key)

        self._house_material_color_cache[material_key] = color
        return color

    def _material_texture_average_color(self, material: UsdShade.Material) -> tuple[float, float, float] | None:
        for child in material.GetPrim().GetChildren():
            shader = UsdShade.Shader(child)
            if shader.GetPrim().IsValid() and shader.GetIdAttr().Get() != "UsdUVTexture":
                continue
            file_input = shader.GetInput("file")
            if not file_input:
                continue
            asset_path = file_input.Get()
            if asset_path is None:
                continue
            texture_path = getattr(asset_path, "resolvedPath", "") or getattr(asset_path, "authoredPath", "")
            if not texture_path:
                continue
            if not os.path.isabs(texture_path):
                texture_path = os.path.join(os.path.dirname(self.house_visual_usd), texture_path)
            if not os.path.exists(texture_path):
                continue
            lower_path = texture_path.lower()
            if not any(token in lower_path for token in ("basecolor", "base_color", "diffuse", "albedo")):
                continue
            color = self._average_texture_rgb(texture_path)
            if color is not None:
                return color
        return None

    @staticmethod
    def _average_texture_rgb(texture_path: str) -> tuple[float, float, float] | None:
        try:
            from PIL import Image

            with Image.open(texture_path) as image:
                image.thumbnail((32, 32))
                rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        except Exception:
            return None

        alpha = rgba[:, :, 3:4]
        alpha_sum = float(alpha.sum())
        if alpha_sum <= 1.0e-6:
            rgb = rgba[:, :, :3].mean(axis=(0, 1))
        else:
            rgb = (rgba[:, :, :3] * alpha).sum(axis=(0, 1)) / alpha_sum

        rgb = np.clip(rgb, 0.04, 0.96)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))

    @staticmethod
    def _material_diffuse_color(material: UsdShade.Material) -> tuple[float, float, float] | None:
        for child in material.GetPrim().GetChildren():
            shader = UsdShade.Shader(child)
            if shader.GetPrim().IsValid() and shader.GetIdAttr().Get() not in {"UsdPreviewSurface", ""}:
                continue
            diffuse_input = shader.GetInput("diffuseColor")
            if not diffuse_input:
                continue
            color = diffuse_input.Get()
            if color is not None:
                return (float(color[0]), float(color[1]), float(color[2]))
        return None

    @staticmethod
    def _semantic_house_color(prim_path: str, material_key: str) -> tuple[float, float, float]:
        text = f"{prim_path} {material_key}".lower()
        if "wood" in text or "table" in text or "chair" in text:
            return (0.55, 0.39, 0.24)
        if "wall" in text or "white" in text:
            return (0.76, 0.74, 0.68)
        if "book" in text:
            palette = ((0.72, 0.18, 0.16), (0.18, 0.32, 0.65), (0.18, 0.50, 0.30), (0.78, 0.60, 0.22))
            return palette[sum(ord(c) for c in material_key) % len(palette)]
        if "metal" in text or "aluminium" in text or "lamp" in text:
            return (0.58, 0.58, 0.55)
        if "glass" in text or "window" in text:
            return (0.42, 0.58, 0.68)
        if "vinyl" in text or "floor" in text:
            return (0.44, 0.39, 0.34)
        if "plant" in text or "leaf" in text:
            return (0.30, 0.48, 0.28)
        return (0.64, 0.61, 0.55)

    def _attach_initial_usd_camera(self) -> None:
        if not hasattr(self.viewer, "stage"):
            return

        camera = UsdGeom.Camera.Define(self.viewer.stage, "/root/waic_initial_camera")
        camera.CreateFocalLengthAttr().Set(CAMERA_LENS_MM)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRow(0, Gf.Vec4d(*CAMERA_RIGHT, 0.0))
        matrix.SetRow(1, Gf.Vec4d(*CAMERA_UP, 0.0))
        matrix.SetRow(2, Gf.Vec4d(*CAMERA_BACK, 0.0))
        matrix.SetRow(3, Gf.Vec4d(float(CAMERA_POS[0]), float(CAMERA_POS[1]), float(CAMERA_POS[2]), 1.0))
        xform.AddTransformOp().Set(matrix)

    @staticmethod
    def create_parser():
        parser = fold_v2.Example.create_parser()
        parser.add_argument(
            "--waic-robot-base-x",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[0]),
            help="World X of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-y",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[1]),
            help="World Y of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-z",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[2]),
            help="World Z of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qx",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[0]),
            help="World quaternion X of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qy",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[1]),
            help="World quaternion Y of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qz",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[2]),
            help="World quaternion Z of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qw",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[3]),
            help="World quaternion W of the WAIC Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-table-center-x",
            type=float,
            default=WAIC_TABLE_CENTER_X,
            help="Optional manual override for hidden physics table center X. Defaults to old demo table X plus the robot-base offset.",
        )
        parser.add_argument(
            "--waic-table-center-y",
            type=float,
            default=WAIC_TABLE_CENTER_Y,
            help="Optional manual override for hidden physics table center Y. Defaults to old demo table Y plus the robot-base offset.",
        )
        parser.add_argument(
            "--waic-table-top-z",
            type=float,
            default=WAIC_TABLE_TOP_Z,
            help="Optional manual override for hidden physics table top Z. Defaults to old demo table Z plus the robot-base offset.",
        )
        parser.add_argument(
            "--show-physics-table",
            action="store_true",
            help="Render the hidden physics table box visibly for alignment/debugging.",
        )
        parser.add_argument(
            "--house-visual-usd",
            type=str,
            default=HOUSE_VISUAL_USD,
            help="USD visual background exported from the WAIC Blender house scene.",
        )
        parser.add_argument(
            "--hide-house-visual",
            action="store_true",
            help="Skip loading the WAIC house visual background in the viewer.",
        )
        parser.add_argument(
            "--house-visual-color-mode",
            choices=("texture", "material", "semantic", "gray"),
            default="texture",
            help="Color mode for the GL house background. texture averages USD BaseColor textures; gray restores the old neutral look.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
