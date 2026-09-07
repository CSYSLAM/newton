# Cloth settling investigation — 2026-09-07

## Scope and method

Investigated `example_mjvbd_v2_cloth_twist.py` and
`example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00.py`
on RTX 5090 D v2, Warp 1.17.0. No production solver or example settings
were changed for this investigation. Existing uncommitted changes remain.

`scripts/diagnose_mjvbd_v2_settling.py` disables multilevel in its own
process, advances the example 900 frames (15 seconds), then restores the
same particle/body state and time for each solver variant. Each variant
uses a fresh solver and recaptured CUDA graph, runs 180 frames, and reports
the average over the final 90. Scratch/contact history is not restored.
The first five variants below share one snapshot per scene; the final
three were run in a separate invocation and therefore do not share the
exact same snapshot with the first five. GPU contact accumulation and
long trajectories are not assumed deterministic. These are exploratory
single runs, not statistical or visual acceptance tests.

`max_speed` is the average of each frame's maximum particle speed, not
the maximum over the entire window. The second-difference metric is
RMS(q[n] - 2q[n-1] + q[n-2]); it helps distinguish frame-to-frame changes
from slow residual motion, but is not a proof of instability.
All metrics include all particles. Timings include CPU readback and
must not be presented as pure solver performance measurements.

## Results

Speed units: m/s. Second-difference units below: micrometers.

| Scene | Variant | RMS speed | Mean frame max speed | Second difference |
| --- | --- | ---: | ---: | ---: |
| T-shirt | fast, 8 sweeps | 0.003878 | 0.073648 | 47.35 |
| T-shirt | Jacobi relaxation 0.5 | 0.001535 | 0.027395 | 20.58 |
| T-shirt | GS8, caches on, Chebyshev off | 0.001244 | 0.018887 | 12.66 |
| T-shirt | GS20, caches on, Chebyshev off | 0.006849 | 0.066764 | 21.64 |
| T-shirt | GS8, caches off, Chebyshev off | 0.001258 | 0.018061 | 13.35 |
| T-shirt | GS8, contact detection each sweep | 0.000717 | 0.012960 | 8.29 |
| T-shirt | GS8, self-contact disabled | 0.001089 | 0.006448 | 1.77 |
| T-shirt | GS8, final IK frozen | 0.000859 | 0.016480 | 8.46 |
| Twist | fast, 3 sweeps | 0.013875 | 0.066170 | 24.12 |
| Twist | Jacobi relaxation 0.5 | 0.014141 | 0.036301 | 21.07 |
| Twist | GS3, caches on, Chebyshev off | 0.012744 | 0.057717 | 34.66 |
| Twist | GS20, caches on, Chebyshev off | 0.014151 | 0.037865 | 62.16 |
| Twist | GS3, caches off, Chebyshev off | 0.013805 | 0.036191 | 26.63 |
| Twist | GS3, contact detection each sweep | 0.017876 | 0.086570 | 126.31 |
| Twist | GS3, self-contact disabled | 0.046432 | 0.121411 | 188.83 |

## Findings and next steps

- Multilevel is not necessary for the residual motion: all measurements
  above have it disabled. This does not exonerate its other behavior.
- T-shirt settling is sensitive to the batched Jacobi update. Relaxation
  0.5 reduced second differences by about 57% and local maximum speeds by
  about 63% in the common-state comparison. This is the most promising
  low-cost candidate, not a validated universal fix.
- T-shirt cached and uncached GS8 results are close; no evidence here
  justifies reverting the caches. More ordinary sweeps did not uniformly
  improve settling and are not an automatic accuracy reference.
- Self-contact removal greatly reduced the T-shirt high-frequency metric,
  but changes the physical problem and permits penetration. It is only a
  diagnostic probe. Freezing IK and refreshing contacts also warrant
  controlled repeats. Neither result alone establishes a kernel bug.
- Twist responds differently: more frequent contact detection and removal
  of self-contact both worsened its second-difference metric. Do not turn
  either into a global policy based on the T-shirt result.
- Next evaluate relaxation 0.5 through complete folding/twisting histories,
  including grasp retention, interpenetration, final shape and timing
  without readback. Then inspect contact activation/normal/friction
  changes at oscillating vertices. No sleep/freeze, artificial damping,
  collision removal, or production default changes have been applied.

## Reproduction

```bash
uv run scripts/diagnose_mjvbd_v2_settling.py --scene tshirt --cases fast jacobi_half gs_cached gs20_cached gs_uncached
uv run scripts/diagnose_mjvbd_v2_settling.py --scene tshirt --cases gs_contact_each gs_no_self_contact gs_frozen_ik
uv run scripts/diagnose_mjvbd_v2_settling.py --scene twist --cases fast jacobi_half gs_cached gs20_cached gs_uncached
uv run scripts/diagnose_mjvbd_v2_settling.py --scene twist --cases gs_contact_each gs_no_self_contact
```

These commands run CUDA diagnostics, not Quest stop/reload operations.
Keep the existing CUDA guard running on the affected machine. The script
does not stop services, reset the GPU, or change the repository examples.

## Follow-up: actual self-intersections during folding

The screenshot prompted a separate geometry check, not just a velocity
measurement. `scripts/diagnose_mjvbd_v2_intersections.py` runs from the
initial state and checks all unique mesh edges against nonincident faces
on the CPU in float64, using AABB pruning followed by strict segment /
triangle-interior intersection. Coplanar overlaps, endpoint touches and
shared-vertex pairs are excluded. Counts are edge/face pairs, not distinct
holes or a penetration-depth metric. Synthetic crossing, miss and
incident-edge cases passed. The check does not identify the screenshot's
exact triangles without a matching captured state.

Each row is one independent full-history run with multilevel disabled.

| Policy | Frame 0 | Frame 120 | Frame 240 | Frame 390 |
| --- | ---: | ---: | ---: | ---: |
| Current fast8 | 0 | 264 | 277 | 182 |
| GS8, Chebyshev off, caches on | 0 | 245 | 295 | 173 |
| GS20, Chebyshev off, caches on | 0 | 65 | 39 | 117 |
| GS20, Chebyshev off, both caches off | 0 | 57 | 42 | 123 |
| Legacy20 options, no preset | 0 | 60 | 40 | 21 |
| Legacy30 options, no preset | 0 | 32 | 6 | 3 |

Fast8 additionally had 222 pairs at frame 600 and 264 at frame 900.
Legacy30 changes only the legacy20 iteration count from 20 to 30. Its
sampled vertex-triangle overflow row counts were all zero. Edge-edge
overflow affected one row at frame 120 and zero rows at frames 240 and
390. This is evidence of a capacity limit at that sample, not proof that
overflow caused the remaining intersections. At frame 390 the three
remaining edge/face pairs were (898, 10541), (1189, 4136), (1189, 4994).
Compared with the prior legacy20 run, the frame-390 count dropped from
21 to 3 (about 86%), but this is a single-run comparison, not a guarantee
of penetration-free behavior or visual equivalence.
Legacy20 reproduces the old solver *options* (including 16/20 VT/EE
buffer sizes), using today's solver and example code. It is not a checkout
of the entire historical revision and must not be called a git bisect.
GS20 retained today's default 32/64 buffers. Its sampled VT/EE overflow
row counts were zero at frames 120, 240 and 390; that does not rule out
overflow between samples. Frame-zero detector counters are uninitialized
and are deliberately not reported by the updated script.

The full-history result invalidates using the earlier settling-only test
as evidence of penetration safety. Turning off Jacobi and Chebyshev
without restoring iterations did not eliminate crossings. GS20 reduced
early crossings, but even the legacy-options path was not crossing-free.
The cache-only GS20 comparison produced similar crossing counts with both
caches disabled (57 / 42 / 123). Its sampled overflow counts were also zero.
This does not support blaming the caches for the large late difference
against legacy20. The remaining option differences and run-to-run variation
need isolation; lowering contact buffer capacity is not a safe remedy.

History inspection confirms `889ae3d4` changed the example from 20 ordinary
sweeps to 12 with multilevel. Later revisions introduced Chebyshev,
cache policies and the surface preset. `9746291e` changed that preset
from 5 sweeps plus selective polishing to 8 batched sweeps and removed
selective polishing. These are candidate policy changes, not a confirmed
first-bad-commit attribution. No production changes or rollback were made
in this investigation.

```bash
uv run scripts/diagnose_mjvbd_v2_intersections.py --case fast
uv run scripts/diagnose_mjvbd_v2_intersections.py --case gs8 --frames 390
uv run scripts/diagnose_mjvbd_v2_intersections.py --case gs20 --frames 390
uv run scripts/diagnose_mjvbd_v2_intersections.py --case legacy20 --frames 390
uv run scripts/diagnose_mjvbd_v2_intersections.py --case legacy30 --frames 390
uv run scripts/diagnose_mjvbd_v2_intersections.py --case gs20_uncached --frames 390
```
