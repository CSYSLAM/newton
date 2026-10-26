# Two-cut bread indentation and blade contact

Date: 2026-10-26. Device: RTX 5060 Ti, Warp 1.18.0.
Baseline: `0e81b4cf`, default 1,200-frame two-cut example.

## Retained changes

The two-cut scene now uses a 1,500 Pa Young modulus (previously 4,000 Pa)
and a 0.2% local connection compression yield threshold (previously 0.5%).
The existing compressive return map retains the resulting indentation after
the knife unloads. These are material changes, not a prescribed deformation
of the particle positions. MPM compressive yield pressure remains unchanged.

The handle starts 22.5 mm farther along X, and the sawing offsets change
from +15/-10 mm to +8/-5 mm, so the whole stroke retains clearance from
both the heel and curved tip on the straight 60--190 mm blade segment.
Lateral slice separation is 10 mm
instead of 18 mm. Initial withdrawal lifts 30 mm while returning 4 mm into
the open kerf, relieving the pushed face before the rest of the lift.
A 1.5 mm physical contact margin on the blade reduces subgrid contact error.

The authored display spheres have a 2.85 mm radius, versus the material
samples' 2.5 mm radius. Even exterior sample centers can therefore render
through a thin blade. A small GPU kernel bounds each exterior glyph by its
actual distance to the blade, including a 0.05 mm display clearance. It
restores the full radius after withdrawal. An interior center retains its
full glyph radius, so this adjustment cannot conceal physical penetration.

Signed distances from the same mesh query are checked every frame in test
mode, rejecting centers more than 0.1 mm inside the blade. The render and
test paths share one query per frame, with no state readback during normal
rendering. Queries cover centers inside the entire 12 mm spine. Material
positions, mass, connections, and fracture decisions are not changed by the
display kernel. The loaf remains a particle rendering; no surface is rebuilt.

The single-cut scene retains its existing material settings and trajectory.
Its separation distance is exposed as a class setting for the two-cut scene
to override. Iteration limits and simulation substeps are unchanged.

## Measurements

The [result summary](data/bread_three_pieces_deformation_20261026.json)
records the baseline and updated full trajectory, including validation.

| Measurement | Baseline | Updated, video run |
| --- | ---: | ---: |
| First retained notch | 2.12 mm | 5.19 mm |
| Second retained notch | 1.63 mm | 5.01 mm |
| First retained opening | 9.76 mm | 7.05 mm |
| Second retained opening | 10.74 mm | 7.19 mm |
| Material sample count | 7,376 | 7,376 |
| Final coherent pieces | 3 | 3 |
| Detached samples | 0 | 0 |

An independent signed-distance probe checked every one of the 1,200 frames
of the updated video run: no sample centers entered the blade, and minimum
center clearance was 0.099 mm. An independent second run with the final
per-frame acceptance check also passed: 5.29/4.97 mm retained notches and
0.265 mm minimum center clearance. These are measured trajectories, not a
general nonpenetration guarantee for arbitrary material/trajectory settings.
Earlier versions without the kerf release had occasional submillimeter
center penetrations at the start of withdrawal. A repeat also exposed a
penetration during the wider sawing stroke. The narrowed stroke, kerf
release, and per-frame acceptance check address these observed failures.
The later performance repeats below nevertheless exposed additional
transient contact errors; the initial passing runs do not establish hard
nonpenetration across repeated executions.

Both cuts must now retain at least 3 mm indentation in test mode. Existing
mass conservation, sequential 1/2/3-piece topology, no-detached-sample,
finite-state, speed, grip, and capacity checks remain active. The GPU/CPU
geometry test covers both exterior blade faces, unchanged interior glyphs,
restoration after withdrawal, and unchanged material positions. Trajectory
checks cover the straight cutting segment, kerf release, and safe traverse.

Directly reducing compressive yield pressure, broadening the damage zone,
and larger blade margins produced tiny fragments in trial runs. Those
variants were rejected; their outputs are archived outside the checkout.

Reproduce the scene, checks, stills, and 20-second simulation-time preview:

```powershell
uv run --no-sync python scripts/record_mjmpm_bread_cut.py --scene three-pieces --video --output-dir ../newton-results/bread-three-pieces
uv run --no-sync python -m unittest newton.tests.mjmpm.test_bread_cut newton.tests.mjmpm.test_bread_three_pieces newton.tests.mjmpm.test_bread_cut_cohesion
```

The preview contains 1,200 frames at 1280 x 720 and 60 encoded FPS. This is
simulation-time playback and does not measure realtime application FPS.

## Performance follow-up

The user-approved scene parameters were retained unchanged. A fresh-process
A/B comparison loaded the three original runtime modules from `0e81b4cf`
outside the checkout, sharing the unchanged assets and solver backend.
The [performance summary](data/bread_three_pieces_fps_20261026.json) records
the matched runs, source hashes, validation warnings, and kernel probe.

The frame timer covers production `step()`, GL rendering at 1280 x 720,
`glFinish()`, and device synchronization. Warmup is 60 frames, the trajectory
is 1,200 frames, and there is no video encoding in the timer. Acceptance
checks run afterwards. Only completed, finite trajectories passing final
scene checks contribute to the matched performance results.

| Matched pair | Previous FPS | Approved FPS | Change |
| --- | ---: | ---: | ---: |
| 2 | 25.88 | 25.19 | -2.68% |
| 4 | 24.93 | 25.70 | +3.10% |
| 5 | 24.40 | 24.54 | +0.54% |
| Arithmetic mean FPS | 25.07 | 25.14 | |

The geometric mean paired FPS change is +0.29%, with a run-paired 95%
interval of -6.66% to +7.77%. The measurements do not establish a stable
overall slowdown or speedup. One run can be slower even though the matched
mean is essentially unchanged; the interval also does not rule out a small
regression.

An isolated CUDA-event probe of the added glyph kernel used frozen approved
particle/tool states, 100 calls per graph, ten replays per sample, and five
samples per state. Median cost was 0.028--0.035 ms around cutting and
separation, and 0.0015 ms after full withdrawal. These GPU timings exclude
CPU submission and GL synchronization. They are small relative to the
roughly 40 ms application frame and do not explain a large FPS reduction.

Component samples used the scene's existing every-30-frame production
readbacks as a physics completion boundary, with no new phase fences. Their
mean step-return cost was 35.05/34.68 ms (previous/approved), and subsequent
render-and-wait cost was 6.40/6.52 ms. These sparse, diagnostic-frame samples
include host diagnostics and vary by run; they do not establish a separate
stable physics or rendering regression.

### Validation exceptions retained in the measurements

Five paired attempts were made. Approved runs 1 and 3 initially aborted
under strict per-frame checks: a 0.110 mm penetration exceeded the 0.1 mm
threshold, and one trajectory still had one connected piece at exactly
7.000 s. Their partial FPS values were excluded.

For additional timing runs 4 and 5, transient per-frame assertions were
recorded without aborting the production trajectory. Finite-state, excessive
speed, and grip failures would still abort, and final scene checks still
had to pass. Thus a performance-valid run does **not** imply every transient
audit passed. Run 4 recorded two contact errors (up to 0.116 mm); run 5
recorded ten contact errors (up to 0.490 mm). All three matched complete
approved runs finished with three coherent pieces, conserved 7,376 samples,
zero detached samples, and retained notches above 3 mm.

These contact residuals remain an observed repeatability limitation. The
effect, material parameters, trajectory, glyph bounds, and strict example
assertions were not relaxed during this performance check. Raw logs, partial
runs, source snapshots, and the measurement harness remain outside the
checkout in `../newton-results/bread-fps-20261026/`.
