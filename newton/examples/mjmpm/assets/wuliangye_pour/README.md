# Wuliangye pouring props

These are original procedural meshes made for the Newton example, not scans
or official product CAD. The bottle follows the silhouette and red/gold label
layout of eighth-generation Wuliangye; the name and product identity belong to their respective
owners. This demonstration is not affiliated with the manufacturer.

Reference: [eighth-generation bottle launch photographs](https://ishare.ifeng.com/c/s/7mrLuLGAnmF).
Reference photographs are not bundled or used as textures. The shoulder emblem
is an original simplified grain motif, not an exact reproduction of the logo.
Dimensions are adapted to this robot demonstration, not a 500 ml product replica.

`props.npz` stores indexed triangle meshes, normals, colors, roughness,
metallicity, display opacity, UV coordinates, and attachment groups. Optional
Blender authoring exports contain glass transmission and IOR 1.46.
Bottle and glass have closed, consistently oriented walls and open cavities.
The simulation scales the bottle horizontally to fit the W1 fingers. The
separate cap rests on the table; the example starts with the bottle open.
The curved label uses `label.jpg`, generated with the built-in image_gen tool.
The exact prompt is in `label_prompt.txt`; it is original illustrative artwork,
not an official product label. No fonts are required at runtime. Liquid remains visible as particles through translucent vessels.

`generate_assets.py` reproduces the meshes using Blender's Python API and
NumPy. Execute it with `__file__` set to the script path from Blender's Python
console. It replaces only its `Newton_Wuliangye_Props` collection. Set
`BLEND_OUTPUT` in the execution globals to export an optional editable `.blend`
file outside the checkout. Runtime geometry stays in the compact NPZ; display
positions use micrometre precision and shading normals use float16. Physics
positions retain float32 precision.

The generator and original mesh geometry use the repository's Apache-2.0
license. The liquid parameters are illustrative, not a measured constitutive
model of the commercial beverage.

The detailed display meshes add twelve-fold pressed-glass fluting, a rounded
shoulder, curved gold-bordered label, shoulder medallion, neck seal, stepped
cap, thin glass stem, turned foot and rolled cup lip. A rounded, inward-tapering
549 ml bowl replaces the earlier narrow goblet, with the same spline used for
the visible inner wall, collision cavity and particle containment checks. The
invisible outer cup wall extends 8 mm outward and downward for collision sampling; the visible wall
is 1.8 mm thick. Decorative meshes have no mass or collision. The bottle rests
on a bevelled cylindrical pedestal instead of the previous visible cube.

The example explicitly enables `ViewerGL.set_transparent_reflections(True)`.
This experimental thin-glass approximation retains environment reflections at
low base opacity and increases reflection at grazing angles, with fixed IOR
1.5. It is disabled by default in other scenes. The glass walls use low residual
haze rather than the previous milky constant-opacity shading. The darker table
provides contrast for the stem and edges without making the glass opaque.

This distinction between optical transmission and simple alpha coverage is
described in [Khronos' transmission specification](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_transmission).
The viewer approximation is not a complete implementation of that extension:
it has no background refraction, local scene reflection, thickness-dependent
absorption, or multiple scattering. Its appearance differs from a path-traced
Blender render. No painted highlights or fluid surface reconstruction are used.
Liquid remains discrete particles, so it will not resemble a continuous liquid
surface.

## Running and validation

```console
uv run --extra examples -m newton.examples mjmpm_wuliangye_pour
uv run --extra examples -m newton.examples mjmpm_wuliangye_pour --viewer null --test
```

The 22-second sequence grasps the bottle body, lifts it off its wooden rest,
pours along a 60-degree azimuth, returns it upright, opens the fingers, and
retracts. Arm targets are limited to 1.2 rad/s to prevent redundant IK branch
changes from jerking the bottle. The rest raises the pickup above an awkward
low wrist configuration. Neither the bottle pose nor liquid particle positions
are overwritten to enforce the animation.

Exterior grip patches provide rigid finger contact. The vessel walls use low
friction independently of those patches. Robot-particle contact is disabled
because the bottle walls separate the fingers from the contained liquid;
forces still reach the robot through the free bottle and its rigid contacts.
The geometry and trajectory are designed together for this example.

### Current liquid density feedback configuration

The default keeps 10,062 particles, a 6 mm grid, four substeps, PIC transfer,
particle-based contact, and CUDA graph execution. It now uses P0 strain,
GS with a 256-iteration budget, `critical_fraction=1`,
`density_correction=0.1`, and no stress warm start. The liquid explicitly
disables plastic hardening/softening rates. The particle count and radius,
vessel geometry, grasp and pouring trajectory are unchanged.

This is an experimental **volume-drift correction**, not a claim of exact
geometric volume conservation or a 25 FPS solution. The complete 1,320-frame
GL recording reached 98.6% delivery: 310.2 ml reference volume versus a 286.8 ml
level-based cavity estimate, about 7.5% below reference. The four-second
bottle still shows a larger discrepancy (about 264 ml below the particle-center
99th-percentile height versus 314 ml reference). Coarse-grid boundary and
surface errors remain. The earlier configuration's severe final contraction
to roughly 105 ml must not be accepted as a performance improvement.

The new option belongs to MJMPM's private material solver and is passed through
`SolverMJMPM(..., mpm_options=...)`. It defaults to zero; existing examples keep
their original incompressibility behavior. Positive values require P0 strain,
`critical_fraction=1`, `warmstart_mode="none"`, non-compliant material and no
hardening. Unsupported combinations raise `ValueError`. This restriction
avoids applying a cell-volume estimate to signed higher-order strain modes or
feeding numerical volume recovery into elastic/plastic material history.

For each cell, the solver compares reference particle volume with the cell
volume minus estimated collider occupancy. The existing positive void offset
is unchanged. For an overfilled cell, the added negative pressure RHS offset is
`density_correction * max(void_volume - particle_volume, -0.25 * particle_volume)`.
The contact-aware pressure solve then produces an expansion velocity; the
example does not rescale a liquid mesh, move particles after the solve, emit
particles or change their reference volumes. This numerical feedback may add
energy and its relaxation per unit time depends on the substep size.

The final acceptance check compares reference volume with the inner-cavity
volume below the 99th-percentile particle height plus one simulation radius,
with a 10% tolerance. This assumes the settled liquid fills the bowl below a
roughly level surface; it is a diagnostic, not reconstruction of splashes or
interior voids. The UI now calls the particle-count quantity **reference liquid
volume**. Retention, finiteness, capacity and bottle-return checks remain.

Rejected full-pour trials included GS/512 without density feedback (about
206 ml level estimate for 309 ml reference), and B2/PIC/batched GS/25 with
air drag 1 (136 ml for 308 ml reference). Simply increasing iterations or
changing interpolation did not solve the contraction. The first feedback
prototype restored volume but failed finiteness in `particle_Jp`; the retained
configuration uses the existing no-history path and passes that check too.
The no-history GS/128 trial failed the initial half-bottle height check, so its
faster runtime was not accepted.

On the RTX 5060 Ti at 1280x900, the final recording averaged **10.49 FPS** for
simulation plus synchronized GL after 60 warmup frames (median 9.39 FPS).
Readback and encoding are excluded. This is slower than the preceding 14.17 FPS
configuration with severe volume loss. There were 24 escaped particles (0.24%);
maximum grasp offset was 3.94 mm. Final volume, finiteness, capacity, retention
and upright-return checks passed. The 22-second video is encoded at 60 FPS;
that rate is playback speed, not real-time simulation throughput.

The fixed source version passed 72 regression tests covering material/rigid
and articulated coupling, contact, CUDA graphs, bread cutting, and the new
density option. The new tests cover stationary overcompression recovery with
no net impulse, eager versus captured execution, default signed-offset
behavior, and rejection of unsupported configurations on CPU/CUDA as relevant.
Logs are in the external `graph_audit/` directory. The new full video and its
measurement JSON are in
`E:/csy_work/CG/Engine/newton-results/wuliangye-pour/volume_feedback/`.

### Previous reduced-particle velocity-only configuration

**Known correctness issue (volume audit, 2026-10-28):** the checks below
validate particle retention, not occupied liquid volume. A full default run
retained 9,906 particles in the glass (309.56 ml reference volume), but the
99th-percentile and maximum particle heights were only 122.77 and 126.75 mm
in glass-local coordinates. Integrating the inner cavity to these heights,
including one simulation particle radius, gives approximately 105-129 ml.
A level 309.56 ml charge would reach 155.66 mm. This cavity-based estimate is
not surface reconstruction, but establishes substantial spatial contraction.
The bottle was already contracted at four seconds: its 99th-percentile
height enclosed about 264 ml versus 314 ml reference volume. The 14.17 FPS
result therefore does **not** establish an optimization that preserves volume.
The relative contributions of finite solver convergence, contact coupling and
particle advection still require controlled isolation. Logs and the probe are
in `graph_audit/volume_audit.log` and `graph_audit/volume_audit.py` under the
external results directory cited below. No volume-preservation fix is claimed.

This previous preset used 10,062 particles, approximately half of the previous
20,091. Sampling spacing is 3.150 mm; mass and simulation radius scale with
spacing, retaining about 314.4 ml nominal volume instead of halving the liquid
charge. Display radius is 2.079 mm and simulation radius is 1.575 mm.

The default uses four substeps instead of six, a 6 mm grid, P1d strain,
PIC transfer and 128 Jacobi iterations instead of 512 GS iterations. The coarser grid keeps
approximately two particles per voxel axis. A 4 mm grid with the reduced
particle density collapsed the static fill below half height. A 5 mm grid
with APIC and 512 iterations also missed the four-second height check.
The earlier GS preset retained a 114 mm liquid height; the then-current Jacobi
preset retains 94 mm at four seconds, above half the 180 mm bottle height.
These settings trade spatial and transfer detail for speed; the scene still
uses free-body grasping and does not spawn, delete or reposition liquid.

A two-substep full test reached only 28.5% delivery and was rejected, despite
passing the initial height check. Thus two substeps are not a supported default
for this trajectory.

That preset enabled CUDA graph replay, used the existing parallel
Jacobi solver, and reserves 4,096 active grid cells instead of 16,384. The
10,062 particles, four substeps, 128-iteration budget, tolerance, PIC contact
basis and trajectory are unchanged from the reduced-particle GS preset.
The smaller grid allocation passed the complete scene's capacity checks;
it is a capacity bound, not a coarser grid or permission to discard cells.
Jacobi changes convergence at a fixed iteration budget: the lower settled
height is a measured tradeoff, not numerical equivalence to GS.

The complete 1,320-frame synchronized GL recording with these defaults passed
with 9,821 / 10,062 particles (97.60%) in the glass and 8 (0.08%) outside both
vessels. Maximum grasp offset was 3.94 mm. Final finiteness, grid and particle
basis capacities, active flags, bottle return and intermediate glass-retention
checks passed. At 1280x900 on the RTX 5060 Ti, mean simulation plus GL throughput
after 60 warmup frames was **14.17 FPS**, versus **5.52 FPS** for the preceding
eager GS preset (**2.57x**). Median throughput was 12.06 FPS. Early motion reached
19-20 FPS, but the filled-glass phase ran at about 11-12 FPS; the full-run mean
must not be replaced by the faster startup measurement. Readback and video
encoding are excluded. The video plays at 60 FPS and is not real-time playback.
The requested 25 FPS has **not** been reached.

Three short GL graph launches with different extra GPU allocations and the
complete GL recording passed for the reduced-particle PIC configuration.
This does not establish a fix for the previous high-charge APIC graph fault;
`--no-cuda-graph` remains available for diagnosing other configurations.

Rejected probes: Q1 nodal contact and S2 contact reduced the initial liquid
height to 39 mm and 58 mm respectively and failed the height assertion.
The reordered GS variant was slower. Removing the generic fused rigid-feedback
row limit did not improve scene FPS, so that solver change was reverted.
No generic solver implementation change is retained from this audit.

Repeat the full per-frame validation through the normal example entry point:

```powershell
uv run --no-sync -m newton.examples mjmpm_wuliangye_pour --headless --test --report-fps
```

This also passed with 97.5% delivery and 11 escaped particles. Its viewer
counter reported about 14 FPS; use the synchronized recording metric above
for the performance comparison. Local audit logs, the measurement JSON and
the 22-second video are outside the checkout under
`E:/csy_work/CG/Engine/newton-results/wuliangye-pour/graph_audit/`;
the complete recording is `jacobi_full/wuliangye_pour.mp4`.

### Previous reduced-particle eager GS baseline

The complete 1,320-frame GL recording passed with 9,833 / 10,062 particles
(97.72%) in the glass, 205 in the returned bottle and 24 (0.24%) outside both
vessels. Maximum measured grasp offset was 3.92 mm; final state finiteness,
capacity, active-particle flags, return pose and intermediate glass-retention
checks passed. The recording retained about 114 mm liquid height at 4 seconds.

At 1280x900 on the same RTX 5060 Ti, mean simulation plus synchronized GL
throughput after 60 warmup frames was **5.52 FPS**, versus 2.14 FPS for the
20,091-particle/six-substep preset (about **2.58x**). Readback and video encoding
are excluded from both measurements. The 22-second video is encoded at 60 FPS;
that is playback speed. This configuration still does not meet the 25 FPS target.

### Previous 20,091-particle configuration

The previous high-charge configuration used 20,091 particles (4.63 times the original),
313.9 ml nominal material volume, a 4 mm grid, P1d strain, APIC transfer, six
substeps, and 512 GS iterations. Particle display radius is 1.65 mm, independent
of the 1.25 mm simulation radius. The fluid uses the solver's default near-rigid
modulus with zero friction, as in the teapot liquid reference; the earlier soft
elastic override was removed. The increased iteration budget improves pressure
convergence at this larger charge. The initial sample extends to 165 mm and is
allowed to settle physically; tests require the 99th-percentile liquid height
to remain above 90 mm at both 2 and 4 seconds. The bottle is 180 mm tall.
The initial preview is not treated as proof of the settled fill level.

Active-cell capacity is 16,384 to accommodate the larger charge. The pour TCP
is shifted back by 18 mm along the pour direction to account for the outgoing
jet's horizontal travel. At 20,091 particles, 128 and 256 iterations settled
below half height by 4 seconds (81 mm and 89 mm respectively); the 512-iteration
static trial retained 99 mm. This is an accuracy setting with a substantial
performance cost, not a solver speed improvement. Trials
with GIMP or P0 did not improve this scene and were not retained. The CR solver
does not support the active contacts here and was not used. No solver code was
modified for these trials. No surface reconstruction,
particle spawning, or particle retirement is used. Simulation stops at the
final pose while the viewer continues displaying the bottle and filled glass.

The scene assigns a private FEM temporary store to its material solver before
stepping to reuse scratch allocations without changing Warp's global defaults.
That previous high-charge preset disabled whole-frame capture: GL graph
replay produced CUDA illegal memory accesses on the development machine,
including with the persistent pool and in a serial-preparation diagnostic.
One full graph run completed, but recording still reproduced the error, so
the APIC high-charge graph path was not considered reliable. The current
reduced-particle PIC validation above does not fix this historical fault or
change graph defaults in other examples.

### High-charge validation (2026-10-27)

The complete 1,320-frame GL recording with that earlier eager configuration
passed: 19,984 / 20,091 particles (99.47%) remained in the glass, 61 remained
in the returned bottle, and 46 (0.23%) were outside both vessels. All particles
remained active. Final checks covered finite states, grid capacity, grasp
retention and upright bottle return; maximum measured grasp offset was
2.42 mm. The settled liquid height was 104 mm at 2 seconds and 97 mm at
4 seconds, above half the 180 mm bottle height. The intermediate glass-count
checks also passed. This is not a claim of zero spilling.

On the RTX 5060 Ti 8 GiB, the 1280x900 recording measured **2.14 FPS** on
average after 60 warmup frames, including simulation and synchronized GL
rendering but excluding image readback and encoding. The 22-second video is
encoded at 60 FPS; that playback rate is not simulation performance. The
high-charge configuration is substantially slower than the historical
4,342-particle configuration and does **not** meet the 25 FPS target.

Both collision meshes passed closed-edge, winding and positive-volume checks.
All 47 viewer reflection/frame/batching/layer regression tests and pre-commit
checks for the changed files passed. The final recording checks state
finiteness at completion; the earlier full GL graph trial checked each frame
but failed containment with the thinner collision wall (107 escaped particles).

## Performance audit against Isaac Lab (2026-10-27)

The [Isaac Lab v3.0.0 teapot source](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/examples/demos/teapot_fill.py)
uses an 800 Hz simulation loop and renders every two steps. Its container is
scripted, without a robot grasp. Its MPM configuration uses P0/APIC and a
100-iteration maximum. Thus each rendered frame represents 2.5 ms, whereas
this example represents 16.67 ms. If an observed 25 FPS uses that configuration,
it corresponds to 0.0625 simulation seconds per wall second, not real time;
our measured 2.14 FPS corresponds to 0.0357. This arithmetic is not a matched
hardware benchmark, and the user's particular video/configuration is unknown.

The [later main-branch robot configuration](https://github.com/isaac-sim/IsaacLab/blob/be69b949e2d30bb07ce1f4120a4699d17347b1cf/examples/demos/teapot_fill.py)
uses B2 velocity interpolation, P1d strain, PIC transfer and 25 iterations.
It is distinct from the v3.0.0 kinematic demo. Copying only its iteration count
would not reproduce its discretization or solver selection.

Local synchronized section timings on the existing Q1/APIC/512 preset put
`Strain solve` near 60 ms per substep, or about 360 ms per six-substep frame,
without rendering. This already accounts for most of the slow frame; detailed
prop rendering is not the dominant cause. A separate instrumented CPU profile
observed six graph destructors waiting about 320 ms in total. Destruction can
wait for queued GPU work, so this is not evidence of 320 ms of additional CPU
overhead, and the two measurements must not be added together.

Static 241-frame probes kept the 20,091 particles, geometry, six substeps and
timestep unchanged, with no renderer and outer graph capture disabled:

| Velocity / transfer / solver | Iterations | Height at 4 seconds | Measured loop FPS |
| --- | ---: | ---: | ---: |
| B2 / PIC / GS | 25 | 95.4 mm | 1.18 |
| Q1 / PIC / GS | 25 | 84.6 mm (fails half-height criterion) | 4.89 |

These are static probes, not complete pouring validations or like-for-like
comparisons with the full GL recording. Selecting `auto` for B2 chooses
batched GS: a separate short probe reduced section time from roughly 114 to
39 ms per substep, but did not establish four-second height or full-pour
correctness. None of these trial configurations replaces the default.

The prior decision to raise iterations to 512 based primarily on fill height
was not an adequate performance solution. The B2 probe shows that preserving
height does not intrinsically require 512 iterations. Likewise, disabling
outer graph capture is only a stability workaround, not a fix for the CUDA
fault. Further work must address discretization/convergence and graph safety,
then rerun the complete grasp/pour/return and containment checks; simply
reducing particles or iterations would not meet the requested behavior.

## Historical results for the 4,342-particle version

Before the detailed asset revision, on an RTX 5060 Ti 8 GiB, two complete null-viewer test runs passed with 97.8%
and 98.1% of particles in the glass. A separate full GL recording passed with
4,244/4,342 particles in the glass (97.7%) and a maximum grasp offset of
2.44 mm. All three runs checked finite states, grid capacity, upright bottle
return, active particle flags, and at least 99.5% containment in the two vessels.
The remaining liquid can stay in the bottle; the test does not require a
perfectly dry bottle.

The 1280x900 GL recording measured **18.52 FPS** after 60 warmup frames,
including simulation and synchronized rendering, excluding readback and video
encoding. Its playback is 60 Hz; that is not the simulation throughput.
The previous 25 FPS target is not yet met by this configuration.

A four-substep P1d trial delivered only 82.6%, and a P0 trial delivered only
59.4%; neither became the default. These results concern this new example,
not a general solver performance claim. No solver implementation was changed.

After the detailed asset revision, a complete null-viewer test passed with
98.0% in the glass and 2.3 mm maximum grasp offset. The detailed GL video
passed with 4,244/4,342 particles (97.7%) and 2.43 mm maximum grasp offset.
That recording used CPU image readback and `CUDA_LAUNCH_BLOCKING=1` after
two asynchronous recorder attempts failed with CUDA illegal-memory-access
errors. Its measured 18.22 FPS is diagnostic, not a controlled performance
comparison with the previous 18.52 FPS. No synchronization workaround was
added to the example or solver; the recording issue is not claimed fixed.
A separate complete `--headless --test` GL run, using default asynchronous
CUDA execution without the recorder, passed with 97.8% in the glass and
2.4 mm maximum grasp offset.

With thin-glass reflections, the complete recorded run passed with
4,232/4,342 particles (97.5%) in the glass and 2.42 mm maximum grasp offset.
It used the same diagnostic recording settings and measured 18.10 FPS;
this is not a controlled performance claim. Rendering regression checks
cover unchanged opaque/metal/zero-opacity surfaces, restoring the original
image on disabling reflections, and resetting the option on scene changes.
The subsequent complete default asynchronous `--headless --test` run passed
with 98.2% in the glass and 2.4 mm maximum grasp offset.
