"""Export only simulated geometry and the original Piper visual meshes."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, Vt

import newton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos/piper_bag_transport"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demos/piper_package_transport"))
from export_simulation import SimulationUSD, bounds, box_mesh, set_pose


class SceneUSD(SimulationUSD):
    def __init__(self, path, sim, frames):
        self.path = Path(path)
        self.stage = Usd.Stage.CreateNew(str(path))
        self.stage.SetFramesPerSecond(30)
        self.stage.SetTimeCodesPerSecond(30)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(frames)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/World").GetPrim())
        self.materials, self.textures, self.meshes, self.bodies = {}, {}, [], []
        self.deformables = []
        colors = ((0.32, 0.35, 0.39), (0.32, 0.35, 0.39), (0.05, 0.55, 0.30))
        for group, color in zip(sim.groups, colors):
            mesh = self.add_mesh("/World/" + group["name"], group["rest"], group["faces"], color, 0.25, 0)
            self.deformables.append((group["start"], group["end"], mesh))
        paths = {-1: "/World/Piper_base"}
        for i in range(sim.model.body_count):
            path = f"/World/body_{i:02d}"
            body = UsdGeom.Xform.Define(self.stage, path)
            self.bodies.append((i, body.AddTranslateOp(), body.AddOrientOp()))
            paths[i] = path
        model = sim.model
        catalog = {}
        robot_materials = {}
        flags, parents = model.shape_flags.numpy(), model.shape_body.numpy()
        scales, transforms, colors = model.shape_scale.numpy(), model.shape_transform.numpy(), model.shape_color.numpy()
        for i, source in enumerate(model.shape_source):
            if not flags[i] & int(newton.ShapeFlags.VISIBLE):
                continue
            if source is None:
                if model.shape_type.numpy()[i] == newton.GeoType.BOX:
                    points, faces = box_mesh(scales[i])
                    path = f"/World/table_{i}" if parents[i] < 0 else f"{paths[int(parents[i])]}/pad_{i}"
                    mesh = self.add_mesh(path, points, faces, colors[i], 0.65, 0)
                    transform = UsdGeom.Xformable(mesh)
                    set_pose(transform.AddTranslateOp(), transform.AddOrientOp(), transforms[i])
                continue
            name = "carton" if parents[i] == sim.carton else f"robot_{i:03d}"
            texture = source.texture if source.uvs is not None else None
            if texture is not None and not isinstance(texture, (str, Path)):
                texture_path = self.path.parent / f"w1_texture_{i:03d}.png"
                pixels = np.asarray(texture)
                if pixels.dtype != np.uint8:
                    pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(texture_path)
                texture = str(texture_path.resolve())
            color = colors[i]
            roughness = 0.4 if source.roughness is None else source.roughness
            if name != "carton":
                color = np.asarray(color)
                color = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
                appearance = dict(
                    name=name,
                    color_linear=color.tolist(),
                    color_srgb=colors[i].tolist(),
                    roughness=roughness,
                    specular=0.5,
                )
                robot_materials[name] = dict(
                    appearance,
                    metallic=0 if source.metallic is None else float(source.metallic),
                    texture=None if texture is None else str(texture),
                )
            normals = None if source.normals is None else np.asarray(source.normals) / scales[i]
            if normals is not None:
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            mesh = self.add_mesh(
                f"{paths[int(parents[i])]}/{name}",
                source.vertices * scales[i],
                source.indices,
                color,
                roughness,
                0 if source.metallic is None else source.metallic,
                normals=normals,
                uvs=source.uvs,
                texture=texture,
            )
            if name != "carton":
                mesh.GetPrim().SetCustomDataByKey("piper:color_srgb", Gf.Vec3f(*appearance["color_srgb"]))
            transform = UsdGeom.Xformable(mesh)
            set_pose(transform.AddTranslateOp(), transform.AddOrientOp(), transforms[i])
        (self.path.parent / "piper_materials.json").write_text(
            json.dumps(dict(**catalog, meshes=robot_materials), indent=2)
        )

    def sample(self, frame, particles, bodies):
        for start, end, mesh in self.deformables:
            points = particles[start:end]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points), frame)
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(bounds(points)), frame)
        for i, translation, orientation in self.bodies:
            set_pose(translation, orientation, bodies[i], frame)
