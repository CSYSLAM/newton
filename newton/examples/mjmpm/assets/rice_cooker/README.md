# Rice cooker for MJMPM

`body.obj` and `lid.obj` were modelled in Blender 5.1.2 for
`mjmpm_w1_rice_cooker`; `model_rice_cooker.py` regenerates them. Units are
metres with the origin at the bottom centre and +z up.

- Body: 24 cm diameter, 15 cm tall, with an 18 cm wide, 11 cm deep inner pot
  and a control panel on the robot side (-x). The walls (3 cm) and the bottom
  (4 cm) are at least two MPM cells of the example's 1.5 cm grid thick, so rice
  grains cannot tunnel through them.
- Lid: a shallow dome hinged at the back rim (+x) and opened 100 degrees, clear
  of the cup during the pour.

Both meshes are closed (no non-manifold edges), as the MPM collider's signed
distance needs. They are original work for this example; no external assets
were used.

## Blender product visuals

The example defaults to `--visual-style realistic`. `realistic_visuals.npz`
contains the original Blender-authored enamel housing, sage trim, metal lid
details, inner pot, control panel, clear measuring pitcher and a single rice
grain, split by material. The archive needs only NumPy to load; Blender
is not required to run the example. `export_realistic.py` regenerates it from
the authored `rice_pour_studio.blend` scene.

The cooker is scaled to the 24 cm diameter, 15 cm tall collision
model and rotated so the lid opens away from the robot. The 85 mm tall pitcher
has a thin, lightly tinted clear body, rolled pouring lip, continuous curved
handle, small silicone grip and foot, and a curved red millilitre scale.
`model_measuring_cup.py` authors these details in Blender. The printed scale is
calibrated against the tapered cavity and remains legible in Newton's viewer.
Wall normals are split at the floor and lip, so the clear material uses the
wall's optical normal instead of an average with the adjoining cap. The
export stores a separate raster-viewer tint and opacity alongside the Blender
transmission shader.

`support/rice_cooker_geometry.py` supplies both the numerical shell and a
separate 2 mm display shell. Their inner material-contact surface and floor
are identical. The numerical wall extends outward to an 80 mm diameter and
15 mm wall thickness for the MPM grid; it is not used as the visible glass.
The display body is about 54 mm across, with the same open pouring cavity.
The 12 mm wide handle has a rounded visual profile and a three-bar collision
proxy. Collision shapes determine contact, mass and inertia; the visual
meshes add no physical shapes or mass. Use
`--visual-style collision` to inspect the collision geometry.

The round cup starts with 1,478 live MPM samples (3.5 mm spacing, about 54 g
of material), compared with 343 in the earlier square cup. The denser fill
uses a `pic128` collider basis so a captured step does not overflow the old
48-sample-per-cell reservation. Sixty rigid contact sweeps resolve the open
handle's additional contact constraints (the thinner handle also uses a
stronger finger position drive to maintain the grasp load).

The cup visuals follow the free body's pose. One instanced grain follows each
live MPM sample; orientations are deterministic display variations, not an
independent rigid-grain simulation. The staged rice mound, stream, cloth and
lighting from the Blender presentation are not imported. Newton viewers use
the exported colors, normals, roughness, metallic values and pitcher opacity;
they do not reproduce Blender's procedural shaders or ray-traced refraction.

```console
uv run --no-sync python -m newton.examples mjmpm_w1_rice_cooker
uv run --no-sync python -m newton.examples mjmpm_w1_rice_cooker --viewer null --test
uv run --no-sync python -m newton.examples mjmpm_w1_rice_cooker --visual-style collision
```
