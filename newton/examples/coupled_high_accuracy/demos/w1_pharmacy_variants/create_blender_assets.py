"""Add a new editable asset-inspection scene, preserving the user's existing Blender scenes and file."""

import importlib.util
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ASSETS = HERE.parents[1] / "assets/w1/pharmacy"
from catalog import CARTONS
from variant_geometry import make_carton

spec = importlib.util.spec_from_file_location("pharmacy_variant_asset_render", HERE / "render_scene.py")
render = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = render
spec.loader.exec_module(render)
scene = bpy.data.scenes.new("W1 pharmacy | four carton constructions")
for i, case in enumerate(CARTONS):
    asset = make_carton(case)
    mesh = bpy.data.meshes.new(case.key + " physical outer surface")
    mesh.from_pydata(asset["vertices"][: asset["outer_count"]].tolist(), [], asset["faces"].tolist())
    mesh.update()
    obj = bpy.data.objects.new(case.name, mesh)
    scene.collection.objects.link(obj)
    obj.location = ((i - 1.5) * 0.16, 0, case.size[2] / 2)
    render.dress(obj, asset, case, solidify=False)
    inner = bpy.data.meshes.new(case.key + " physical inner wall")
    inner.from_pydata(asset["vertices"][asset["outer_count"] :].tolist(), [], asset["faces"][:, ::-1].tolist())
    inner.update()
    wall = bpy.data.objects.new(case.name + " inner wall", inner)
    scene.collection.objects.link(wall)
    wall.location = obj.location
    wall.data.materials.append(render.material(case.key + " raw paperboard", (0.63, 0.59, 0.48), 0.8))
    obj["construction"] = case.construction
    obj["wall_thickness_m"] = case.thickness
    obj["physical_mesh"] = str(ASSETS / f"{case.key}_physics.npz")
output = ASSETS / "pharmacy_carton_variants.blend"
bpy.data.libraries.write(str(output), {scene})
if bpy.context.window is not None:
    bpy.context.window.scene = scene
print("Created four physical carton assets:", str(output))
