Bread cutting visual assets

Byte-exact copies of visuals.npz and the four JPEG textures from Newton's
newton/examples/mjmpm/assets/bread_cut at local Newton revision
bc66356640786a752c3fbc8526a827d50d83b74d (Apache-2.0).
They were authored procedurally in Blender for the MJMPM bread-cutting example.

visuals.npz holds a JSON manifest plus triangle positions, normals and UVs for
the parents board, knife and bread. examples/example_granular_bread_three_pieces.py
draws only the board and knife meshes; the loaf is rendered as solver particles.
The physical knife blade is generated in the example (thickened to the grain
spacing) and does not use these meshes.

The W1 robot for that example is the shared assets/w1/robot/w1-030 package; its
URDF and meshes match Newton's mjmpm/assets/w1_v030/w1-030 apart from line endings.
