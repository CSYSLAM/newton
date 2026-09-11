# W1 sorting workstation

Original geometry authored with Blender through Blender MCP for Newton's
`mjvbd_v2_conveyor_sorting` example. The script and exported mesh are licensed
under Apache-2.0. No downloaded third-party models or textures are included.

- `build_station.py`: editable, deterministic Blender authoring source.
- `station.npz`: evaluated meshes, including bevels and corner normals, grouped
  into 15 material batches (39,320 triangles, about 439 KiB).
- `belt.npz`: closed 4 mm belt laminate, 784 triangles, continuous travel UVs.
- `roller.npz`: rotating drum and hub template, 2,792 triangles, instanced twice.

The asset includes aluminum conveyor extrusions, bearing housings, return drums,
a finned gearmotor, control cabinet and cables, operator controls, an HMI and
stack light, a folded stainless workbench, molded receiving trays, and a factory
bay wall. Dimensions are in meters, with Z up, in the demo's world frame.

To rebuild, run this in Blender's Python console, using an absolute path:

```python
path = "/home/oem/code/repos/newton/assets/conveyor_station/build_station.py"
exec(compile(open(path).read(), path, "exec"), {"__file__": path})
```

Only the `W1 Sorting Workstation` collection is replaced. The script exports
`station.npz`, `belt.npz`, and `roller.npz` beside itself. Blender is not a runtime dependency of the demo.
The NPZ files store `vertices_N`, `normals_N`, `indices_N`, `uvs_N`, and a `materials` array
whose rows contain RGB, roughness, and metallic values. Arrays contain no
pickled objects. Changes to geometry require regenerating the NPZ.

These meshes are visual surfaces. The demo retains its validated belt, tray,
and workbench collision primitives. Rounded tray corners and handles are visual
details; they do not introduce new collision features. Props and the background
wall have no collision and stay outside the parcel trajectories.

The belt outer radius is 0.0521 m, with its top at 0.7601 m. Upper and lower
runs join smoothly around the two drums at Y = -0.24 and 1.76 m. The belt file
also stores `path_length` and `radius`. UV V increases along the direction of
travel, from the far end toward the picking station on the upper run. Runtime
UV advection and drum rotation both use the demo's accumulated belt travel,
including when the belt stops. The texture spans one complete loop and includes
one splice, fine rubber grain, and faint longitudinal wear. It is generated
locally with a fixed random seed.

The take-up screws, locking nuts, end guards, and return guard pan are visual
mechanical details. The original overlapping kinematic contact sections are
retained, including their shape ordering, to preserve calibrated parcel grasps.
The rendered belt does not teleport with those hidden contact sections.
