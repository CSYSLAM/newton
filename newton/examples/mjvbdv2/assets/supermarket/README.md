# Supermarket packing assets

## Active files

- `piper/`: WAIC Piper, native joint limits and two-finger geometry.
  Its README records the pinned source and license. The example adds a
  task TCP and matching visible/contact silicone pads.
- `shopping_bag.blend/json`: vest-style bag, 1,097 vertices, 2,088 triangles,
  0.30 x 0.18 x 0.27 m excluding handles. Rounded carrying apertures and
  separate racking slots rest on visible rails, with no pinned vertices.
- `checkout_decor.blend/json`: checkout and shelving decoration, with
  split normals, materials and surface-bound lettering.
- `pantry_assets.blend`, `pantry_meshes.json`, `usd/`: metric shelf groceries,
  UVs, normals and USD exports. CC0 sources, PBR maps and attribution are
  retained under `polyhaven/`, together with the floor texture.
- `bag_print.png`, `bread_crust.png`, `milk_pouch.png`: textures that follow
  the simulated surfaces. JPEG previews document their appearance.
- `asset_manifest.json`: asset roles, units and material assumptions.

The four `build_*_blender.py` scripts regenerate assets through their
`build(output_directory)` functions in separate Blender scenes. Blender
sources and PBR maps remain useful even where Newton GL uses only albedo
and scalar materials. No official SimReady validation has been run.
The unused UR5e assets are no longer bundled.

## Physics and limitations

Piper is a prescribed MuJoCo articulation coupled one-way to MJVBD V2.
Bread is elastic tetrahedral geometry; the milk-labelled pouch is pneumatic,
**not a liquid simulation**. Groceries move through contact/friction, without
attachments or state resets. The bag is pre-threaded onto visible rails:
no pinned rim or hidden bottom support. Threading itself is not simulated.

Defaults are 8 substeps / 24 sweeps and 0.001 m/s friction smoothing.
The transport guard rejects relative grocery-center changes above 18 mm,
including deformation as well as slip. A 40-second run and a full recording
passed, but another recording triggered the guard at 18.3 mm / 9 seconds.
Repeatability remains open; the actual example's guard is unchanged.

Still-air film drag uses triangle area and a positive-semidefinite implicit
update, not a post-trajectory velocity multiplier. Materials are illustrative,
not measured; background stock is noncolliding decoration. Tests do not prove
continuous nonpenetration, zero sway, SimReady compliance or photorealism.
Historical trials are in `docs/lab/supermarket_hanging_bag_2026-09-15/README.md`
at the repository root.

## Run and verify

```powershell
uv run --no-sync python -m newton.examples mjvbd_v2_supermarket_packing
uv run --no-sync python -m newton.examples mjvbd_v2_supermarket_packing --viewer null --num-frames 2400 --test
uv run --no-sync python -m unittest newton.tests.test_mjvbd_v2_supermarket
```

Functional checks cover lifting, bag placement, finite state, sampled inversion,
table crossing and transport slip. CPU tests cover geometry, exports, native
limits, pad visual/contact agreement, reachable motion, clearance and drag.
