# Parallel rigid mobility feedback

Date: 2026-10-26. Device: RTX 5060 Ti 8 GiB, Warp 1.18.0.

## Retained implementation

The existing fused material-to-rigid feedback kernel gave one thread all six
twist components of a rigid body. Up to 64 bodies consequently occupied one
CUDA block. Each thread serially loaded and multiplied the active links'
mobility blocks.

`apply_mobility_feedback_rows` assigns one 32-thread block to each body and
uses six lanes for its six twist components. Each lane evaluates the same
three-component products in the same active-link order. Two six-float shared
tiles separate articulated response from the subsequent independent-body
response. Only lane zero updates the accumulated spatial vector, avoiding
component write races. Each body still has one writer for each output.

The existing fused-path limits remain: CUDA, at most 64 bodies, at most
8,192 collider rows, and the existing sparse-layout checks. Fewer than four
bodies retain the compact kernel. CPU and unfused paths retain their previous
implementation. This change adds no material parameters, approximations,
iteration reductions, substep changes, or per-step allocation.

## Identical-input measurement

`scripts/benchmark_mjmpm_mobility_layout.py` compares the two kernels using
fixed dense mobility matrices, random wrenches, and mixed independent-body
responses. Each CUDA graph contains 100 calls, and each timing covers ten
replays. Values below are medians of three alternating pairs, per call.
The script also asserts bitwise equality of the resulting twists.

| Bodies | Active links | Compact | Parallel rows | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 4 | 5.53 us | 3.34 us | 40% |
| 24 | 12 | 11.97 us | 5.04 us | 58% |
| 24 | 24 | 21.10 us | 7.22 us | 66% |
| 64 | 64 | 74.83 us | 15.92 us | 79% |

These are kernel timings, not simulation FPS.

## Full-scene limits

The final alternating bread run measured synchronized physics steps:

| Window | Compact | Parallel rows |
| --- | ---: | ---: |
| Cutting | 36.57 ms | 36.40 ms |
| Separation | 37.07 ms | 36.86 ms |
| Whole warm trajectory | 30.48 ms | 30.89 ms |

The cutting improvement is small, and whole-trajectory timing varies with
the nonlinear solve workload. Earlier diagnostic runs also showed small
cutting improvements but mixed whole-trajectory changes. There is **no
demonstrated general increase in application FPS**. The retained benefit is
the lower cost of this exact mobility operation, especially with many active
links. The dominant colored strain solve remains unchanged.

The bread acceptance checks passed with 2,204 and 5,172 particles in the two
pieces, zero detached samples, a 1.90 mm retained notch, 3.54 mm recoil,
9.41 mm retained opening, and 0.399 mm minimum visual grip clearance.

The existing fused-feedback regression now checks both layouts against the
unfused reference, on CPU/CUDA and repeated CUDA graph replay. Its fixture
includes off-diagonal independent response, a zero response direction,
unconnected bodies, and all/free/empty active-row cases. All output bits
(collider velocity, body twist, and accumulated body response) must match.
An additional review probe with two worlds, noncontiguous link IDs, an
inactive articulated body, and mixed dense independent responses also matched
the compact kernel bit for bit in eager execution and three graph replays.
The regression additionally requires the intended compact or row kernel
to execute. Replacing the row dispatch with the original compact dispatch
in a review process made this check fail; restoring the retained dispatch
passed the complete CPU/CUDA regression.

The 68-test MJMPM suite passed, as did the updated six-body focused test
and repository pre-commit checks. Full GL teapot acceptance passed at
30.4 viewer FPS with 2,668/2,831 particles in the bowl (94%) and three
retired particles. Full rice-cooker acceptance passed with 100% of rice in
the pot, no spills, 3.34 mm cup slip, and 1.62 N loaded versus 1.10 N empty
finger load. The teapot FPS is a current-run measurement, not an A/B speedup.
The full 1,200-frame bread two-cut example also passed: three pieces of
1,992 / 2,204 / 3,180 particles, zero detached samples, and both retained
cut openings within their original acceptance bounds.

Reproduce:

```powershell
uv run --no-sync python -B scripts/benchmark_mjmpm_mobility_layout.py
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --mobility-layout compact --output artifacts/mjmpm_perf/compact.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --mobility-layout rows --validate --output artifacts/mjmpm_perf/rows.json
uv run --no-sync python -m unittest newton.tests.mjmpm.test_solver -k test_fused_rigid_mobility_matches_reference
```

## Full-scene GL A/B measurement

`scripts/benchmark_mjmpm_scene_fps.py` runs the water (900 frames), rice
(990 frames), and two-cut bread (1,200 frames) trajectories in fresh,
sequential processes. The only A/B switch is `_MOBILITY_ROW_PARALLEL`.
Resolution is 1280 x 720 with headless GL rendering. The first 60 frames
are warmup; each subsequent frame measures the production `step()`,
`render()`, `glFinish()`, and device synchronization together. No video
encoding or playback is involved. Acceptance checks execute outside this
timer, so these numbers describe synchronized simulation plus rendering,
not test-harness wall-clock throughput.

The initial plan uses three paired runs per scene, with the second pair's
order reversed. Failed runs remain in the output and are excluded from
complete, accepted trajectory comparisons. Raw per-frame timings, final
acceptance results, GPU clocks/temperature, and captured kernel dispatch
counts are saved in `artifacts/mjmpm_fps_ab/`. The protocol manifest also
records source hashes. Confidence intervals use independent run pairs,
not individual frames as independent samples.

Results on the device and software versions listed above:

The [paired result summary](data/mobility_rows_scene_fps_20261026.json) is
retained in the repository alongside this report. Full logs, per-frame
samples, and previously exported videos are archived outside the checkout
when cleaning the workspace.

| Scene | Complete accepted pairs | Compact mean FPS | Parallel rows mean FPS | Paired FPS change | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water | 3 | 31.28 | 31.55 | +0.85% | -2.81% to +4.65% |
| Rice cooker | 3 | 27.73 | 27.93 | +0.74% | -2.37% to +3.95% |
| Bread, two cuts | 3 | 26.39 | 26.54 | +0.58% | -3.93% to +5.31% |

The FPS columns are arithmetic means of accepted run FPS. Changes are
geometric means of paired FPS ratios. Intervals use a Student-t interval on
the three log ratios; individual frames are correlated and do not count as
additional independent runs. The per-pair changes were water
`+2.49%, +0.56%, -0.47%`, rice `+2.22%, +0.05%, -0.03%`, and bread
`+0.45%, +2.52%, -1.19%`.

There were 20 runs: 19 passed their scene acceptance checks. The compact
water baseline in round one failed at frame 704 (11.733 simulated seconds)
with nonfinite `body_q`, `body_qd`, `joint_q`, and stress. Its partial
31.76 FPS is **not** compared with a complete trajectory. Both reports
from that pair remain saved. One additional independent water pair was
run, in rows/compact order, giving water three accepted pairs from four
attempted pairs. This single baseline failure does not establish a
stability difference between the kernels.

All six rice runs passed pouring, slip, and finger-load checks. All six
bread runs passed sequential two-cut checks, with three coherent pieces
of 1,992 / 2,204 / 3,180 particles and zero detached samples. All seven
completed water runs passed their original acceptance checks. Captured
dispatch counters confirmed that each run used its intended layout.

The average application gains are below 1%, every interval includes zero,
and every scene has a pair that was slower with parallel rows. This test
therefore **does not establish a stable overall FPS improvement**. The
identical-input kernel speedup remains valid, but must not be presented
as a similar application speedup. The original full-scene limitation
above still applies.

Reproduce:

```powershell
uv run --no-sync python -u scripts/benchmark_mjmpm_scene_fps.py --output-dir artifacts/mjmpm_fps_ab --rounds 3
```

Use a new output directory for another measurement. Existing reports are
preserved on resume, including failures.

The additional water pair used the same command with
`--worker --scene water --layout rows --round 4`, then `--layout compact`,
and the same output directory. Call `summarize(output_dir, 4)` from the
benchmark module to include these reports. The exact extension driver
is retained with the raw results as `extend_water.py`.

## Rejected probes in this investigation

Cached GS rows and empty-row eigenvalue shortcuts did not show stable frame
improvements. Conditional sparse-grid reuse hit unsupported allocation inside
CUDA conditional graph bodies. BVH group filtering and midpoint-query splitting
either changed traversal order or failed to improve identical-pose collision
timing; those changes were removed. The original BVH path also exposed an
intermittent teapot stress/settling failure during the investigation. No broad
collision or constitutive changes are included in this optimization.
