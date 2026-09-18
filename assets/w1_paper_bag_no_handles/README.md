# W1 paper bag without handles

Independent runtime assets for `mjvbd_v2_webxr_w1_bag_packing_no_handles`.
The original handled scene continues to load `../w1_paper_bag/` unchanged.

`bag.npz` contains the original 1,197 paper-body vertices and 2,328 paper
triangles, plus eight triangles closing the four handle-root holes. The ribbon
vertices and triangles are removed. Only the bag mouth remains open.
`bag.obj` is an editable export of the same mesh in meters; the original Blender
file remains in the handled asset directory.

`placement_bounds` preserves the original asset's layout pivot so removing the
handles does not translate the bag body. Material density, stiffness, contact,
fold and hem rules are inherited from the shared packing scene. Total mass
changes only through the removed ribbons and restored wall patches.

`snacks.npz`, `snacks.json`, `dimensions.json` and `kraft.png` are independent,
byte-identical copies of the original assets. The soft cube is built by the
shared scene code. See [source provenance and dimensions](../w1_paper_bag/README.md).

Regenerate this directory from the current handled asset without modifying it:

```bash
uv run python assets/w1_paper_bag_no_handles/build_assets.py
```

New recordings store `bag_variant: "no-handles"`; the existing packing demo
selects this asset automatically during replay. Recordings without that field
continue to use the original handled bag.
