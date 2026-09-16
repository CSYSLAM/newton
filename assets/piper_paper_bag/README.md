# PiPER paper shopping bag

Created in Blender 5.2 through Blender MCP from the two user-provided reference
images. The body dimensions were confirmed by the user: 200 mm broad face,
90 mm depth, 260 mm height, with handles extending approximately 100 mm above
the rim. Local X is depth, Y is width, and Z is height; the base is at Z = 0.

The open kraft-paper body has shallow side gussets, concave side rims, an
inside folded hem, glued bottom flaps, reinforcement patches, and two
three-strand twisted-paper handles. The editable source is
[`kraft_shopping_bag.blend`](kraft_shopping_bag.blend), scene
`PiPER kraft bag asset`. Its objects are separate from the user's other
Blender scenes.

## Physical asset

`simulation_mesh.npz` contains 1,893 vertices and 3,720 triangles. The hidden
Blender object `Simulation shell (hidden)` preserves that topology. Each cord
end joins a finite triangular patch in the paper through a six-sided sleeve;
this connection transmits both tensile load and bending moment. The only open
mesh boundary is the bag mouth.

The `paper_plies` face attribute records bonded double plies at the hem, bottom
flaps, and handle pads. Runtime material assembly accounts for their additional
mass, membrane stiffness, and bending stiffness. All particles have positive
mass and remain dynamic. The paperboard and cords use separate elastic bending
coefficients. The coefficients are an experimental demo material, not a fit to
a measured paper grade. The visible single-sheet thickness is 0.32 mm.

## Render asset

`render_mesh.npz` and `render_parts.json` contain the evaluated Blender shell,
folded details, and braided cord geometry: 32,154 vertices and 64,160 triangles.
The inside hem is extruded from the shell's mouth and shares its vertices at
the fold. Reinforcement patches and bottom flaps are subdivided to follow
local deformation; wall-sized quads would bridge a bent wall and expose gaps.
`render_binding.npz` binds every visual vertex to a physical triangle using
barycentric coordinates and a local-frame offset. The render surface therefore
follows the simulated paper and handles. `kraft_albedo.png` is an original
cellulose-grain texture authored in Blender and packed into the blend file.

The previews `preview.png` and `preview_top.png` show the Blender asset from
oblique and overhead viewpoints. They are asset previews, not simulation results.

## Re-export

Open the blend file and select its bag scene. In Blender's Python console,
or through MCP, run the following with the repository path substituted:

```python
from pathlib import Path
folder = Path('/home/oem/code/repos/newton/assets/piper_paper_bag')
namespace = {'__name__': 'asset_export'}
exec(compile((folder / 'export_blender.py').read_text(), 'export_blender.py', 'exec'), namespace)
namespace['export_asset'](folder)
```

Then rebuild the visual binding from the repository root:

```bash
uv run --offline --no-sync python assets/piper_paper_bag/build_binding.py
uv run --offline --no-sync python -m unittest newton.tests.test_mjvbd_v2_piper_paper_bag
```

The binding exporter uses NumPy only. Blender is required for editing/exporting,
but not for running the example. The two PiPER robots reuse the user-provided
assets documented in [`../piper_bag/README.md`](../piper_bag/README.md).

## Packing scene

Run `uv run --extra examples -m newton.examples mjvbd_v2_piper_paper_bag`.
The left arm approaches above the front rim before descending to support it;
the parcel travels above both cord loops before release. Handle pickup releases
the rim first and approaches the loops from their outside faces.

The scene contains one 44 mm volumetric soft cube. It uses a 5 x 5 x 5
cell tetrahedral grid with density 250 kg/m³, shear stiffness 30,000,
Lamé stiffness 80,000 and damping 0.5. There is no pneumatic pouch or rigid
ball in the active scene. The right gripper has visible silicone pads with
matching physical geometry.

Containment is checked against the fitted frame of the paper body using all
soft-parcel vertices. A parcel left on the table is not counted as contained
after the bag lifts. This envelope check assumes the paper body remains
approximately box-shaped.

Jaw apertures include the particle collision radius and fingertip margin.
The front paper rim uses a 6.6 mm jaw gap; handle pickup uses a 9 mm gap on
the bare left fingers and a 12 mm joint gap on the padded right fingers
(8 mm between its pad surfaces). This avoids crushing both contact envelopes
into an infeasible narrow gap. The stationary support phase also checks paper
RMS speed, so a small but rapidly vibrating deformation cannot pass solely
because the bag retains its overall shape.

## Solver scope and validation

Contact materials, jaw apertures, and trajectories are configured only by this
example. It explicitly enables rigid-soft DAT and particle-only Galerkin
multilevel correction in the kinematic full-contact backend. The shared solver
change permits that combination; it does not change defaults or numerical
kernels. DAT still rejects CUDA fusion, coupled body-particle translation, and
autodiff.

Run the complete pickup, containment, two-handle retention, and vibration checks:

```bash
uv run --extra examples -m newton.examples mjvbd_v2_piper_paper_bag --viewer null --test
```

The unloaded stability check remains available with `--settle-only --num-frames 1800`.
The DAT regression tests cover cloth and mixed cloth/tetrahedral models with
both multilevel operators, ordinary steps, and CUDA graph replay:

```bash
uv run --extra dev -m unittest newton.tests.test_mjvbd_v2_piper_paper_bag newton.tests.test_mjvbd_v2_rigid_soft_dat
```
