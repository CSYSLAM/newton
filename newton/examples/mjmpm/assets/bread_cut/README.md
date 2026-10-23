# Bread cutting assets

Original procedural meshes and textures authored through Blender MCP with
`scripts/create_bread_cut_assets.py`. No downloaded or generated third-party
models are included. Meshes and textures use this repository's Apache-2.0
license. The editable Blender scene is saved under the author's Documents
folder; the example needs only `visuals.npz` and the four JPEG textures.

`visuals.npz` contains a JSON manifest plus evaluated triangle positions,
normals and UVs. Parents are `board`, `knife` and `bread`. The knife wedge
matches the collision wedge, with a 12 mm spine and 1 mm edge. This resolved
blade is thicker than a typical kitchen knife; it is a resolved chef-knife demonstration
on a 6 mm MPM grid.

The default example renders the bread as material particles and does not load
or draw either loaf surface patch. Thus the initial loaf has no predefined
visible cut seam. Knife and board use the authored meshes.

The optional loaf surface has two embedding patches at the demonstrated cut plane.
These are render surfaces; the initial MPM material has no pre-existing gap
or separate physical bodies. Positions and normals follow local affine fits
to live particles, with neighbors restricted to each visual patch. The inner
crumb faces begin hidden inside the loaf and become visible as material moves
apart. This skin does not implement arbitrary paths or topological fracture.

Recreate assets by running the authoring script inside Blender, either through
its Python console or the Blender MCP execute-code tool. Asset generation
replaces only objects in the `MJMPM Bread Cutting` collection and saves a copy
of the Blender scene. Adjust the output paths at the top for another machine.
