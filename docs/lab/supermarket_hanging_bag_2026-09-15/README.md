# Hanging supermarket bag: diagnosis and validation

## Scope

Only the supermarket example, authored bag asset and its tests change.
No shared MJVBD V2 contact kernel, friction law or solver default is changed.
Groceries remain free bodies/deformables, held only by visible finger contact.

## Diagnosed issues

- The old front/back loop handles and strongly narrowed bottom did not match
  a disposable supermarket vest bag. The narrowed bottom could wedge goods.
- Global soft friction also participates in gripper contact mixing. Reducing
  it from 0.8 to 0.18 improved descent but broke the pouch grasp. This was a
  failed experiment, not an accepted low-friction configuration.
- Strain-rate damping does not damp rigid-body pendulum motion. The old
  hanging bag had no still-air drag. Increasing internal damping alone is
  not equivalent to adding an external dissipative force.
- Picking the flattened pouch too high pushed its thin edges away. Lowering
  the pick also exposed a 3 mm metal-finger/table overlap; shortening the
  metal tip fixes this without changing the visible rubber pad. A regression
  test fails at a 0.71701 m finger minimum for the old tip, against the 0.72 m
  tabletop, and passes with the corrected tip throughout the robot path.

## Asset and physical changes

The Blender-authored replacement follows the integral side shoulders and
gussets in the [Everpack carrier-bag reference](https://www.everpack.net/product/market-carrier-bags-small/).
It has 1,097 vertices, 2,104 triangles and only 20 constrained vertices at
the two visible side-handle clamps. The bottom is unsupported and expanded;
there is no hidden basket or duplicated collision shell.

The scene adds area-based quadratic still-air drag. Its frozen positive
semidefinite drag tensor uses backward Euler, applied through particle forces
on every substep. The isolated-force test verifies nonpositive work, no
increase in speed, and no force on fixed vertices. This does not prove that
the complete contact solver is converged. No trajectory-dependent damping,
velocity reset, reduced gravity or prescribed grocery motion is used.

## Measurements so far

RTX 5060 Ti, Warp 1.17.0; 40 simulated seconds. Sway is the peak-to-peak
mass-weighted bag center displacement during seconds 35–40. Different meshes
and contact trajectories make this an end-to-end observation, not an isolated
algorithm speed or accuracy comparison.

| Revision | Sweeps | Sway X / Y (mm) | Mean bag kinetic energy (J) | Both items placed |
|---|---:|---:|---:|---|
| Old hanging bag | 12 | 48.90 / 31.04 | 1.00468e-4 | Yes, block remained high |
| Vest, friction 0.18, damping 1 | 12 | 7.91 / 32.41 | 1.41028e-4 | No |
| Vest, friction 0.18, damping 5 | 20 | 4.36 / 13.46 | 1.39457e-4 | No, pouch slipped |
| Vest, friction 0.4, damping 10 | 12 | 8.02 / 34.97 | 8.85320e-5 | Yes |
| Current: clear release, corrected fingers, closer pouch grip | 12 | 7.98 / 23.96 | 9.04302e-5 | Yes |

The successful 12-sweep trial placed the block and pouch at approximately
0.773 m and 0.748 m center height at 40 s, versus the old block's 0.887 m.
Their lift heights were 0.585 m and 0.591 m. The minimum sampled tetrahedral
volume ratio was 0.353, with no sampled inversion. However, residual swinging
was still visible: this is not a complete settling pass.

The old metric comparing goods to the *mean* height of the original bottom
vertices is not a valid local contact gap once the bag folds around a load.
Do not interpret its negative values as evidence of penetration or its
positive values as a rigorous clearance test.

## Reproduce

```powershell
uv run --no-sync python -m unittest newton.tests.test_mjvbd_v2_supermarket
uv run --no-sync python -m newton.examples mjvbd_v2_supermarket_packing --viewer null --num-frames 2400 --test
```

Four CPU tests cover welded bag topology, sealed pouch volume, reachable
robot motion with finger/table clearance, and isolated drag dissipation.
The demo's placement test does not yet establish asymptotic settling or
continuous zero penetration. The 20-sweep corrected-tip run did not solve
settling: the loaded bag swung toward the counter and failed placement at
40 seconds. The default remains 12 sweeps; simply increasing work is not an
accepted fix. The revised release is 85 mm higher, keeping the open fingers
above the central neck instead of withdrawing through it. A nominal
finger/neck clearance check now covers both release intervals.

## Historical corrected-grip result (before contact-only rack)

The corrected fingers, pouch slide target of 0.004 m, and above-neck release
completed 2,400 frames / 40 seconds with the placement test passing. Final
centers were block `(0.40371, 0.23671, 0.77497)` m and pouch
`(0.46502, 0.24686, 0.74772)` m. Maximum lifts were 0.58368 and 0.57941 m;
minimum sampled tet-volume ratio was 0.22541. A render of the saved final
state confirms both goods in the lower bag region, with the bag unsupported
by the counter. Asset preview and 320x320 example thumbnail were refreshed.

Residual sway is **not eliminated**: late-window X/Y peak-to-peak amplitudes
are 7.98/23.96 mm, compared with 48.90/31.04 mm for the old scene. This is
partial improvement, not proof of asymptotic convergence. No performance
claim or proof of continuous nonpenetration is made. Further settling work
must separate contact time-discretization effects from physical pendulum
motion; increasing sweeps alone was not robust in this experiment.

## Bread / milk appearance follow-up

Replaced the elastic block with a domed tetrahedral bread reference mesh and
rendered the existing pneumatic pouch with authored milk packaging. Physics
still uses an illustrative elastic bread and pneumatic pouch surrogate, not
a fluid-filled milk solver. Blender-generated UV prints deform with the
actual simulation surface. Shelf lettering now follows cylindrical facets;
Blender split normals and material properties are retained in static rendering.

The normal example command passed 1,800 frames / 30 seconds with `--test`
after the rendering changes. Lift heights were 0.57141/0.57225 m, final center
heights 0.77335/0.74761 m, and minimum sampled tet-volume ratio 0.29032.
The bread revision also passed a separate 40-second diagnostic run. Six unit
tests now include bread reference tetrahedra and exported decoration normals.
Near-field rendered previews are retained with the supermarket assets.

These appearance changes do not resolve the remaining settling limitation
reported above. Shared MJVBD V2 solver code and its defaults remain unchanged.

## Contact-only rack and prepared assets: current revision

The bag now has rounded carrying apertures and separate front/back racking
slots. Two visible capsule rails pass through the slots, with raised end stops.
All 1,097 bag vertices have positive mass: there are no pinned vertices or
hidden attachment constraints. A negative-control run disabling only the
rail/end-stop particle collisions lets the bag fall and triggers the floor
assertion. This establishes that rail contact, not an implicit anchor, supports
the bag. Initial threading onto the rack is authored; rack loading is not simulated.

A 2,400-frame / 40-second run passed placement with maximum grocery lifts
0.56426/0.58115 m. Minimum sampled tet-volume ratio was 0.11932 (positive,
but significant compression). Over 35–40 seconds, bag COM peak-to-peak
motion was 1.53/5.59/0.71 mm in X/Y/Z and mean kinetic energy was
2.580e-6 J. The preceding bread/milk setup measured 5.90/24.28 mm in X/Y.
This is a scene-level comparison including changed support geometry, not
an isolated solver convergence claim or continuous nonpenetration proof.

The final asset-integrated command passed 1,800 frames / 30 seconds with
`--test`: lifts 0.56172/0.57311 m, minimum sampled tet-volume ratio 0.32797,
minimum bag height 0.73120 m. Seven unit tests pass, including prepared
pantry asset dimensions, UVs and export presence. Final saved-state visual
checks and the example thumbnail were refreshed. Defaults remain 8 substeps
and 12 sweeps; shared solver code is unchanged.

Poly Haven CC0 Long Life Food meshes now supply actual shelf cartons and
cans, with metric bottom-centered geometry, UVs, normals, source PBR maps,
Blender source and USD exports. Floor Tiles 08 supplies the tiled floor.
Source links and licenses are recorded under the assets' `polyhaven` folder.
`asset_manifest.json` documents simulation roles and uncalibrated materials.
These assets are prepared for this scene, not officially SimReady-certified.
Newton GL uses albedo and scalar material properties here, not all source PBR
maps; the resulting real-time view is not claimed to be photorealistic.

## Piper replacement and grasp validation

Replaced the active UR5e with WAIC Piper assets pinned to
`e1176d518f831b17392fb1bf6b90de8b97f200a9`. No branch merge or shared
solver change is involved. The native limits and opposite-sign finger slide
coordinates are preserved. A visible 0.35 m pedestal raises the base to
1.07 m; transfer height is now 1.16 m. The TCP orientation and 45-degree
initial grocery orientation align the opening with each product's short side.
The robot remains kinematic, not a dynamic force-controlled arm.

The native tapered fingers initially dropped the bread after approximately
0.19 m lift. Tightening the diagonal grasp alone produced severe distortion
and a sampled inverted tetrahedron; that trial was rejected. Visible silicone
pads (26 x 32 x 4 mm) now cover the inner fingertips with matching collision
boxes. The final bread/pouch slide targets are +/−23 mm and +/−10 mm.
Their unchanged reference meshes, mass densities and elastic parameters are
retained, with only the initial rotation changed and UVs rotated consistently.

The example now uses friction smoothing velocity 0.001 m/s instead of the
solver default 0.01 m/s. This alone did not fix transport creep: 12-sweep
trials failed the new 18 mm relative-center displacement guard. A 24-sweep
diagnostic reached 25 seconds with maximum relative-center changes
16.48 mm (bread) and 8.69 mm (pouch), lifts 0.397/0.399 m, and minimum
sampled tet-volume ratio 0.263. This metric includes elastic deformation,
not only contact-point sliding. It is not a zero-slip claim. The exact
25-second placement test previously suffered floating-point cutoff rounding;
the cutoff now includes a 1 microsecond tolerance.

Default iterations are raised from 12 to 24 for this grasp, with 8 substeps
unchanged. This costs solver work: no unchanged-FPS claim is made. Monitoring
runs every 15 frames throughout closed-jaw transport, before commanded release;
no anchors, feedback teleportation, mass reduction or velocity reset is used.
Eight CPU tests cover the asset geometry, native joint limits, visual/contact
pad agreement, reachable trajectory, table clearance and existing cloth checks.

The normal default invocation then passed 2,400 frames / 40 seconds with
`--viewer null --test`. Maximum relative-center changes were 15.89/8.72 mm;
maximum lifts were 0.39532/0.39868 m. Both groceries remained in the lower bag
at the final state: centers `(0.38869, 0.27230, 0.78378)` and
`(0.46865, 0.26278, 0.75887)` m. Minimum sampled tet-volume ratio was
0.14180 and minimum bag height was 0.72747 m. No early-drop, floor or
inversion assertion fired. These are sampled checks, not continuous
nonpenetration guarantees or zero-slip claims.

## Recording repeatability and cleanup

A default-parameter recording triggered the existing grasp guard at 9 seconds,
with bread relative-center displacement 18.3 mm. Earlier successful trials do
not guarantee repeatability. The next recording completed 30 seconds with no
warnings, displacement 15.80/8.71 mm and minimum tet-volume ratio 0.30165.
The recording-only driver can log transport warnings without stopping video;
it does not alter physical state and is not part of the actual example.
The example's guard remains active and unchanged.

Cleanup moved the unused UR5e bundle (26 files, 33,790,045 bytes) outside the
repository and removed its staged additions. Recoverable assets, the previous
asset README, staged/unstaged patches and index backup are under the sibling
directory `newton_cleanup_archive/supermarket_cleanup_20260915`. Active Piper
assets, Blender sources, USD/PBR exports, licenses and tests remain. No physics
parameters or shared solver code changed during cleanup.

Before submission, the full-repository pre-commit run still reported existing
lint/typo findings in prior experimental scripts. The scoped staged-file
pre-commit run passed all applicable hooks, and eight supermarket tests passed
again on HEAD `ab8d7ea4`. Large assets use path-scoped Git LFS rules. Native
Piper OBJ files retain their upstream trailing blank lines; these generate
`git diff --check` EOF warnings but are not altered just for whitespace.
