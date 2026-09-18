# W1 paper bag with downward rope handles

Independent assets for `mjvbd_v2_webxr_w1_bag_packing_rope_handles`, inspired by
the user's kraft shopping-bag reference. The original paper-ribbon and
handle-free assets remain unchanged in their own directories.

The bag retains all 1,197 original paper vertices and 2,328 paper faces. Each
new U-shaped handle has an approximately 9 mm round section, eight vertices per
ring, and shallow three-lobe helical ridges suggesting twisted jute. The handles
drop 10.5 cm toward local **-Z**, opposite the mouth at local **+Z**. This is a
direction relative to the bag, including when it starts lying on the table.

The two ropes share the original four attachment-hole boundaries with the paper
mesh. There are no separate floating strands or fixed particles. The full mesh
has 1,565 vertices and 3,080 triangles, with only the mouth left open. It uses
the shared deformable-surface solver and existing handle material parameters;
the rope is a tubular shell, not a calibrated volumetric rope model.

`bag.npz` is the runtime asset; `bag.obj` is an editable triangle-mesh export in
meters. `placement_bounds` preserves the original layout pivot. The rope-handle
teleoperation scene adds 8 cm toward the robot's left (+Y) compared with the
other variants. `handle_0_rings` and `handle_1_rings` identify the
round cross-sections for inspection. `snacks.npz`, `snacks.json`, `kraft.png`,
and `dimensions.json` are independent copies of the original assets.

The rope scene sets `_paper_stiffness_scale = 1.5` to increase paper membrane,
area, and bending stiffness, including the hem and folds. Rope stiffness,
damping, mass density, and contact parameters retain their original values.

Regenerate only this variant:

```bash
uv run python assets/w1_paper_bag_rope_handles/build_assets.py
```

Recordings store `bag_variant: "rope-handles"` so the existing packing replay
entry selects this asset automatically. Older recordings keep their original
asset choice. See [original provenance and dimensions](../w1_paper_bag/README.md).

![CPU asset preview](../../docs/images/examples/example_mjvbd_v2_webxr_w1_bag_packing_rope_handles.jpg)
