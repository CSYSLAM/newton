# Piper bag transport demo

Five original textured apples begin above the open food bag and fall into it under gravity.
Their initial centers span 21 mm across the bag and 33 mm along its depth, with staggered horizontal offsets.
The original release heights remain unchanged, while the wider offsets reduce head-on stacking during loading.
The open Piper gripper waits at one handle while the apples fall through the center of the mouth.
Its horizontal lower finger supports the soft handle through contact, with no pinned cloth or added attachment.
The gripper closes around the measured handle and lifts the loaded bag. It first swings sideways three times,
with coordinated shoulder, elbow, and wrist motion, pauses, and then jolts vertically three times using shoulder and elbow,
briefly unloading the apples while keeping the handle's horizontal position fixed.
It slowly lowers the bag onto the platform, waits for the platform to support the load, opens, and withdraws.
There is no loading rack or sequence of robot pickups for the apples.

The bag is built from developed food-bag panels, including folded side gussets, handles, and welded seams.
The 6,007-particle, 12,000-triangle simulation mesh has no separate rectangular bottom or single-sheet arch handles.
Its 40 four-face heat-seal edges preserve every incident film panel. Ordinary hinges retain their original bending
response; welded multilayer seams transfer membrane load without inventing cross-ply bending springs.
The native self-collision path deduplicates seam edges and retains all incident face normals.
An initial geometric fit opens the flat cutting panels with less than 3% strain, without changing their rest metric.
Panel coordinates are exported as static, face-varying `film_pattern_m` UVs in meters, preserving pattern seams
and keeping the crease normal map attached to the deforming film.

The apples use `models/apple/apple.usd` from the Style3D Sim asset repository, at scale 0.58 and 200 g mass each.
Their original visual mesh, color texture, UV coordinates, and normals are exported alongside the actual rigid-body poses.
Full convex collision hulls determine film contact and platform support; they are not substituted for the render mesh.
USD records actual cloth positions and rigid poses, and Blender never substitutes a separately animated trajectory.

The Piper model is downloaded from `robots/piper/piper_with_texture.xml` in the Style3D Sim asset repository.
The robot importer and pose planner use the same cached MJCF and its relative visual and processed collision meshes.
Assets are fetched when needed; importing the demo does not download them.
Set `STYLE3D_SIM_ASSETS_DIR` to an existing asset checkout to use local files.

Physics uses **four nonlinear iterations**, 80 linear iterations, and four substeps at 60 frames per second.
Material, contact, friction, and motor settings live in `../demo/demo_piper_bag_transport.py`.
Rendering settings do not alter them.
The standalone validator rejects exports that record a nonlinear iteration count other than four.

| Time | Action |
| --- | --- |
| 0.00–0.45 s | Five separated apples fall through the clear center of the open bag. |
| 0.45–1.35 s | Close on the measured handle and hold before lifting. |
| 1.35–3.35 s | Lift to a 51 cm grip height. |
| 3.35–3.85 s | Settle before swinging. |
| 3.85–7.90 s | Three coordinated horizontal swings with a 1.35 s period and gentle wrist pitch. |
| 7.90–8.40 s | Pause at the center before changing the swing direction. |
| 8.40–11.40 s | Three 1 s vertical jolts, targeting 35–63 cm handle height with joint5 and joint6 held fixed. |
| 11.40–15.40 s | Slowly lower the loaded bag onto the platform. |
| 15.40–16.40 s | Wait for the platform to support the bag and apples. |
| 16.40–16.80 s | Open the gripper after the bag is resting on the platform. |
| 16.80–18.80 s | Center the open jaw gap, extract along the inclined fingers to clear the handle, then withdraw. |
| 18.80–22.00 s | Observe the bag and all five apples resting on the platform. |

The complete sequence contains 1,321 samples, including the initial state, over 22 seconds at 60 fps.
The horizontal and vertical swings use separate time intervals, both following Cartesian IK waypoints.
Horizontal swinging follows a 38 cm side-to-side path with a shallow 12 mm depth arc and 9 mm height arc.
The wrist turns with the motion and pitches by ±4 degrees, recruiting all six arm joints without a rigid base-only sweep.
The vertical path targets 28 cm of handle travel using only joint2 and joint3. The two joints immediately before the
gripper, joint5 and joint6, keep their entry motor targets throughout this stage; the base and wrist-roll targets also hold.
Position-only IK keeps the held ribbon on its vertical path while the gripper's world orientation follows the arm naturally.
The validator checks both constant commands and completed parent-relative joint angles, allowing small physical compliance.
Smooth ramps stop each phase before the next begins. The exported validator measures actual completed jaw positions
to limit horizontal-stage height variation and vertical-stage lateral drift. Parent-relative completed link poses
verify that both the shoulder and elbow articulate, while the actual finger extension verifies gentle wrist pitch.
It also follows the same gripped cloth vertices throughout the vertical stage, requiring the bag handle itself
to travel with the jaws, retain its grip, and remain at nearly fixed horizontal coordinates.
Each vertical cycle rises for 0.32 s, drops rapidly for 0.24 s, and recovers for 0.44 s. The short downstroke
accelerates faster than gravity while its endpoints retain zero target velocity and acceleration.
The validator uses solved apple velocities to require at least 50 ms of reduced external load in every cycle.
Mass-weighted `a - gravity` cancels forces between apples; its norm must be below 35% of gravity and its vertical
component below 25%, with at least three individual apples also approaching free fall. These are interval-average
kinematic checks, not a direct contact-force sensor. The validated intervals are recorded in `simulation_validation.json`.
These changes affect motor targets; the solver and material parameters below remain unchanged.

The retained film settings are density `0.1`, membrane stiffness `(5e3, 5e3, 2e3)`, zero area stiffness,
zero membrane damping, bending stiffness `1e-6`, and zero bending damping. Particle radius remains `1.5 mm`;
apple collision margin remains `3 mm`. Stage, gripper, and cloth contact coefficients and motor gains are unchanged.

## Run

Run PowerShell commands from the repository root. Install the example dependencies with `uv sync --extra examples`.
USD export requires `usd-core`; Blender requires USD import and Cycles support. The renderer is tested with Blender 5.2.

```powershell
uv run python .\demos\demo\demo_piper_bag_transport.py

$blenderExe = 'D:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$outputDir = '.\demos\piper_bag_transport\outputs\final'
uv run python .\demos\piper_bag_transport\workflow.py --blender $blenderExe --output-dir $outputDir
```

The workflow runs `export -> validate -> telemetry -> build -> check -> render -> encode -> verify` and stops on failure.
One simulation export supplies the scene video, motor input video, and motor feedback video in the same output directory.
The command above writes the complete bundle to `outputs/final/`.
Export length comes from the complete demo duration. Later stages read the exported metadata for the exact frame range.
`frame_end: null` in `render_config.json` selects the last USD sample. Blender is discovered on PATH or in common
Windows installation directories unless `--blender` supplies its executable.

For a complete preview video before approving a full-quality render, use a separate candidate directory:

```powershell
$previewDir = '.\demos\piper_bag_transport\outputs\preview'
uv run python .\demos\piper_bag_transport\workflow.py --blender $blenderExe --draft --output-dir $previewDir
```

This creates `piper_bag_transport_preview.mp4` at 960×540 with 16 Cycles samples per frame. The two telemetry videos
use the same simulation export. After approving the preview, reuse that export for full-quality rendering with
`--steps render encode verify` and omit `--draft`; do not rerun `export` for the approved motion.

For a short pipeline check in a separate output directory:

```powershell
uv run python .\demos\piper_bag_transport\workflow.py --blender $blenderExe --draft --num-frames 12 `
    --output-dir .\demos\piper_bag_transport\outputs\smoke
```

`--num-frames` changes only export length. A short export verifies the pipeline; it does not validate later pickup,
swinging, platform placement, or withdrawal. `--dry-run` prints commands without executing them.

Run individual groups of stages as needed:

```powershell
uv run python .\demos\piper_bag_transport\workflow.py --steps export validate --output-dir $outputDir
uv run python .\demos\piper_bag_transport\workflow.py --steps telemetry --output-dir $outputDir
uv run python .\demos\piper_bag_transport\workflow.py --blender $blenderExe --steps build check preview `
    --output-dir $outputDir
uv run python .\demos\piper_bag_transport\workflow.py --blender $blenderExe --steps render encode verify `
    --output-dir $outputDir
```

## Motor telemetry

Each export records `motor_telemetry.csv` alongside the USD, with eight rows per simulation sample (one per motor)
and timestamps in seconds. The companion `motor_telemetry.json` describes the channels and their units.
The six arm motors use position targets
in radians and torque feedback in N m. The two prismatic finger motors use position targets in meters and force feedback
in N. These quantities are plotted on separate axes within each video so their units remain explicit.

Feedback is the simulated actuator effort evaluated after each physics substep finishes. The completed joint position
and velocity determine the active PD response; the feedforward effort is added and the motor effort limit is applied.
The calculation writes an independent output array without modifying the simulation state.
Angular joint velocity comes from the completed `body_qd`, projected onto the joint axis.
`feedback_effort` records the final substep's output, and the CSV retains the mean, minimum, and maximum across the
frame's four completed substeps. This output is distinct from the nonlinear solver's internal right-hand side,
hardware measurements, and external contact or joint-constraint reactions.
Initial sample zero has blank feedback and a zero validity flag because the solver has not advanced yet.
The plots omit that unavailable feedback.

New sidecars identify this definition with `feedback_sampling: "post_step"` and
`feedback_model: "limited_pd_plus_feedforward"`. The feedback plot labels it as post-step actuator effort.
An older CSV without those markers retains a generic recorded-effort label so plotting cannot change its meaning.

The solver also provides the calculation directly:

```python
effort_buffer = wp.empty(model.joint_dof_count, dtype=wp.float32, device=model.device)
effort = solver.compute_joint_drive_effort(state_out, control, out=effort_buffer)
```

`effort` is an independent `float32[model.joint_dof_count]` GPU array; the state remains read-only.
Pass the control held during the completed substep. Calling `solver.enable_joint_drive_feedback()` enables automatic
evaluation at the end of each solver step and exposes the independent `solver.joint_drive_effort` array used by this demo.

Interactive `../demo/demo_piper_bag_transport.py` runs record this directory's `outputs/motor_telemetry.csv` by default.
Override that path with
`--telemetry-csv PATH`, or disable recording with `--no-telemetry`. Recording does not change the four nonlinear iterations
or any simulation, material, contact, or motor parameters.

The `telemetry` workflow stage creates two videos from that CSV without starting Blender or rerunning the simulation.
To plot an existing CSV directly:

```powershell
uv run python .\demos\piper_bag_transport\plot_motor_telemetry.py `
    --csv .\demos\piper_bag_transport\outputs\final\motor_telemetry.csv `
    --output-dir .\demos\piper_bag_transport\outputs\final --fps 60
```

The input video shows all motor position targets; the feedback video shows the corresponding torques and finger forces.
At each animation frame the curves include only samples up to the current time, and the horizontal axis expands from
zero to that time so the growing curves always fill its width. All four borders and their ticks remain visible.
The first frame uses a one-sample time span because a zero-width time axis cannot be displayed.

## Plastic and lighting

The render scene explicitly uses **Cycles**. Thin pale film transmits light without pretending the mesh is a thick,
solid glass shell. Its air/film/air Fresnel response uses `bag.ior = 1.46`, with about 0.068 normal-incidence reflection.
Glossy reflections and subtle tint make folds readable while the apples remain visible through the bag.
The side key keeps the 45-degree counterclockwise world-Z direction, with position and power halfway between the earlier
low-angle and raised-light previews. Its aim target and source size are unchanged, giving an intermediate shadow length.
Soft fill and strip lights preserve the arm surfaces and the thin film highlights; bag contact shadows remain visible.
The darker blue-gray floor uses metallic 0.4 and roughness 0.35; the small platform uses metallic 0.5 and roughness 0.34.
These shading settings add soft metal reflections without changing collision surfaces or friction.
The camera uses a 30-degree horizontal FOV and a longer viewing distance to retain the full motion in frame.
The 4.4 m square floor extends the visible ground, leaving approximately one third of the image as background.

- `bag.normal_map` selects the seamless 2048 by 2048 crease normal texture at `textures/crumpled_normal.png`.
  It is loaded as Non-Color data and packed into the Blender file. `bag.normal_strength` defaults to 1.0;
  `bag.normal_tile_size_m` defaults to 0.18 m. Developed panel UVs keep creases stable throughout deformation.
- Procedural waviness adds smaller shading detail using the same film UVs. Neither normal detail nor render subdivision
  changes the simulated geometry, mass, collision radius, or bending response.
- The optional studio HDR provides illumination and reflections. A camera-ray branch shows the configured plain
  background instead of the HDR image, so the environment does not appear as a large photographic backdrop.
- Camera, lights, materials, resolution, samples, and video FPS are set in `render_config.json`.
  The default final preset is 1920 by 1080 at 60 fps with 256 Cycles samples, adaptive threshold 0.01,
  denoising, and AgX. Final video encoding uses H.264 CRF 16 with the slow preset.
- GPU rendering is selected automatically. `--device CPU` forces CPU; `--device OPTIX` requires an OptiX device.

Regenerate the deterministic crease texture with:

```powershell
uv run python .\demos\piper_bag_transport\generate_film_normal.py
```

The generator also supports `--studio-output` to regenerate the bundled studio HDR.
Rebuild with `--steps build check preview` after changing materials or lighting.
Building replaces the `.blend`; save manual Blender edits separately before rebuilding.
To retain manual scene edits, save the scene and use only `--steps render encode verify --overwrite`.

## Artifacts and validation

Generated files stay under the ignored `outputs/` directory. The complete delivery uses `outputs/final/`:

Only the approved final run is retained. Its three videos are 1920 by 1080 at 60 fps, with 1,321 frames each.
The scene uses 256 Cycles samples and denoising. Previous candidates, draft renders, backups, and caches have been removed.
The final PNG sequence, sealed simulation, CSV, source snapshots, and Blender project remain available for verification
and reproduction. `render_preview_config.json` preserves the configuration used to build this scene; despite its historical
filename, it contains the final 256-sample settings. Its paths are relative to the final bundle.

- [Scene video](outputs/final/piper_bag_transport.mp4)
- [Motor input video](outputs/final/motor_inputs.mp4)
- [Motor feedback video](outputs/final/motor_feedback.mp4)
- [Per-frame motor telemetry](outputs/final/motor_telemetry.csv)

| File | Purpose |
| --- | --- |
| `simulation.usdc` | Animated cloth, static panel UVs, and rigid-body transforms. |
| `simulation_states.npz` | Raw particle positions, triangles, body poses, and timestamps. |
| `simulation_metadata.json` | Source hashes, solver settings, executed motor plan, and per-frame checks. |
| `simulation_validation.json` | Independent USD/raw-state, sequential swing, and supported platform placement verification. |
| `run_manifest.json`, `sources/` | Export identity and exact source snapshots for this simulation run. |
| `run_validation.json` | Common export identity, frame coverage, hashes, and decoding checks for all three videos. |
| `motor_telemetry.csv`, `motor_telemetry.json` | Per-sample motor targets and post-step actuator efforts, plus units. |
| `motor_inputs.mp4`, `motor_feedback.mp4` | Growing time histories of position commands and post-step actuator efforts. |
| `motor_inputs.png`, `motor_feedback.png` | Complete static motor input and feedback plots. |
| `piper_bag_transport.blend` | Editable Cycles scene with packed film and environment textures. |
| `import_validation.json` | Blender/USD geometry and transform comparison. |
| `frames/` | Complete final-quality PNG sequence and frame provenance manifest. |
| `piper_bag_transport.mp4` | Final H.264 scene video. |
| `*.video.json`, `contact_sheet.jpg` | Full-decode verification and video overview. |

Every export sample is checked for finite states and physical task assertions. Independent validation checks five
nonoverlapping apples above the bag mouth at time zero, downward loading motion, and retention until release.
Horizontal swinging must finish before vertical swinging starts. The first stage must maintain grip height, and
the second must maintain horizontal grip position. The bag and apples must reach platform support before the jaws open.
These checks use the simulated geometry and rigid poses.
The validator also checks exact initial convex-hull clearance, original visual geometry and normals, file/source hashes,
fixed topology, exact timestamps, every cloth sample, and every rigid-body transform.
Grip error is measured only during carrying; intentional handle separation during release is reported separately.
Timeline phases are `loading`, `pickup`, `carry`, `place`, `release`, and `placed`; metadata records both swing intervals.

The retained run has 38.59 cm of horizontal grip travel, followed by 29.85 cm of actual bag-handle vertical travel.
During vertical motion, joint5 and joint6 have constant motor targets; their solved angle ranges are 0.28 and 0.08 degrees.
Film-to-platform gap is 1.36 mm before opening and 1.45 mm after withdrawal, consistent with the 1.5 mm film contact radius.
The final bag and all five apple hulls remain over the platform.
Fresh GPU runs can vary; reuse the sealed export when reproducing a delivered video.

Rendering alone resumes by skipping existing PNGs. Use `--overwrite` after modifying a scene. A workflow that exports
or builds before rendering automatically overwrites frames to avoid mixing versions. Draft rendering uses separate
images and video files. Encoding checks continuity and dimensions and fully decodes the MP4 before replacing an older video.

The export seals `simulation.usdc`, `simulation_states.npz`, and `motor_telemetry.csv` into one run identity and preserves
the exact simulation source files under `sources/`. The final `verify` stage checks all three videos against that identity,
their file hashes, frame counts, frame rates, and full video decoding. It also checks the rendered PNG hash chain.
Files from different simulations cannot be combined into a verified bundle, even if their durations match.

Run that check independently without Blender:

```powershell
uv run python .\demos\piper_bag_transport\verify_run.py --output-dir $outputDir `
    --scene-video piper_bag_transport.mp4
```

Keep the output directory together when moving the project. The Blender file references `//simulation.usdc` and any
portable robot textures under `textures/`. Validation uses the preserved source snapshots for a sealed export.
Changing the demo source affects future exports; it does not change the recorded run.
GPU trajectories can vary between fresh simulations. Reuse one validated export for all three videos, and rebuild
and render the scene again after a new export. Regenerate both motor videos from that same export's CSV.
Reusable rendering and encoding scripts are documented in [../utils/README.md](../utils/README.md).
