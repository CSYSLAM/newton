# Blender-authored popcorn props

Original assets authored for Newton, 2026, Apache-2.0. No downloaded assets or
third-party textures are included.

- `popcorn_props.blend`: editable metric-scale Blender scene; display layout
  separates the three props. The original user scene is preserved.
- `props.json`: simulation-frame meshes, split display normals, colors,
  roughness/metallicity and glass opacity. No runtime Blender dependency.
- `preview.png`: Blender material preview, **not a simulation validation image**.
- `scoop_detail.png`: close-up of the continuous pressed sheet and round grip.

The machine includes rounded enamel panels, glazing rails/rivets, a folded
stainless tray, ventilation details, kettle, lid and hangers. The scoop uses a
cylindrical phenolic grip, ferrule, shank and thin formed aluminum pan. The cup
has restrained brown printing on light paper and a physical rolled rim.

The cup export preserves all 841 physical particle indices and 1640 triangles.
Newton renders the current simulated particle positions, including elastic and
plastic bending, rather than a rigid cup or a pre-deformed animation. Its
material remains an experimental approximation, not measured paper stock.
No visible dent is claimed merely from creating this asset.

Rigid display meshes do not contribute inertia, forces or collision shapes.
The validated collision pan and grip remain coincident with their display
surfaces; fine cabinet manufacturing details are display-only. Newton's GL
viewer uses alpha-blended glass, not Blender's physically refractive glass.

To reproduce through an already running Blender MCP addon on localhost:9876:

```powershell
uv run --no-sync python -m newton.examples.mjvbdv2.support.build_popcorn_blender
```

The builder refuses to replace a nonempty `Newton Popcorn Assets R2` scene. Save
or rename that scene before a deliberate rebuild. Generated JSON is checked by:

```powershell
uv run --no-sync python -m unittest newton.tests.test_mjvbd_v2_popcorn_assets
```

The default enlarged pile has 160 real rigid bodies (+25%); the existing CLI
still allows up to 192. The additional layers start behind
the held pan: extending the original rectangular packing upward intersects the
initial pan and kicks it off its intended grasp attitude. The regression test
detects that old packing at grain 152.

R2 uses one continuous pressed scoop sheet instead of separately rendered
pan panels, with an outside attachment tab, flush rivets and a matte round
grip. The twelve old panels remain hidden references for collision-envelope
tests. The cup has a restrained lap-seam color band on its deforming surface.

The retained R2 paper settings use membrane stiffness 50000, bending stiffness
1 and experimental hinge yield angle 0.015 rad. A 40 s full-scene run reached
0.17225 rad maximum plastic hinge offset and 2.275 mm maximum
rigid-motion-removed RMS shape distortion (previous R1: 0.498 mm). This is a
combined elastic/plastic shape measurement, not a measurement of the unloaded
permanent dent. The cup is visibly ovalized, not fully crushed.
That run lifted 21 grains and retained 17 (81.0%), with 100% five-finger contact
coverage and 0.715 mm final shaft slip. Ring/little-finger closing travel is
expanded under the same force/rate/URDF bounds. No acceptance limit was relaxed.
