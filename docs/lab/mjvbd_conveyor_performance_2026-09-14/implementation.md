# Conveyor computation cache implementation

The computation cache combination is **opt-in** because full-scene settling
acceptance still fails. Add `--compute-cache` to test it; normal startup keeps
the original uncached IK/volume paths.

```bash
uv run --no-sync -m newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting --compute-cache
uv run --no-sync -m newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting --no-compute-cache
```

## Implementation

- Build pressure-only particle lists for each existing elasticity color from
  pneumatic face adjacency. Keep original vertex ordering and empty color rows.
  Pressure assembly no longer scans unrelated cloth and solid particles.
- Select the existing single-cavity fused volume/pressure/force kernel using
  cavity color sizes rather than whole-scene color sizes. Every required volume
  update still occurs, including when the next color has no pressure vertices.
- Enable incremental cavity volumes in this example. Surface bending and
  self-contact truncation remain on their original evaluation paths.
- Capture each of the two IK seed buffers separately. Graph replay reads updated
  target and joint-limit arrays and executes the same 24 iterations. CPU uses
  the ordinary path. Initial seed generation remains the original 300 iterations.

The scene still uses 8 substeps and 16 VBD sweeps. There are no changes to
pressure formulas, materials, collision detection frequency, mesh resolution,
controller trajectories, or the existing cloth settling rule. This does not
enable the earlier pressure color-coupling or Chebyshev experiment.

The solver-level pressure-list compaction also applies when scene caches are
turned off; it skips vertices whose pneumatic contribution was exactly zero.
The toggle therefore compares computation caches, not an exact source checkout.

## Validation protocol

Capture a complete pre-change run before modifying the scene, then run the new
version through all four parcels and another 120 settling frames. Both runs
call `test_post_step()` every frame and `test_final()` at completion. Compare
placement, finite state, speed/limit bounds, robot clearance, cloth settling and
bag volume. Local scripts and trajectories are under
`newton/tests/outputs/conveyor_performance`.

Unit coverage checks exact force/Hessian preservation for unrelated particles,
small-cavity fusion despite a large non-cavity color, empty cavity color rows,
and IK graph reuse after targets/limits change for both seed buffers.

Timing caveat: a separate native inflatable-bag application began using the same
GPU during the full baseline run. Full-run timings are consequently not directly
comparable with the earlier isolated initial-stage probe. No other application
was stopped as part of validation.

## Rejected broader cache configuration

The first full candidate also enabled surface and self-contact truncation
caches. Both baseline and candidate completed all four placements, but the
last-second cloth RMS speed was 5.06 mm/s before versus 9.90 mm/s after; both
failed the existing 2 mm/s final-settling assertion. This did not establish
preserved settling quality, so those two scene options were removed from the
retained version. This observation alone does not isolate a causal module in
a nonlinear contact simulation. Their run is retained as `after.json` in the
local output directory; the narrowed implementation is `retained.json`.

## Final measured result

An alternating 180-frame A/B test at 1280x960, excluding frames 0–29, uses
separate identical scene instances and alternates which version runs first.
The reference disables the new caches and restores the original full-color
pressure lists. Rendering uses the NVIDIA OpenGL renderer with interop disabled
and `glFinish()` to include completed rendering work. This initial-stage test
is not a full-sequence throughput guarantee.

| Initial-stage measurement | Reference | Retained opt-in mode |
| --- | ---: | ---: |
| Step | 51.22 ms | 37.28 ms |
| Render | 15.48 ms | 15.32 ms |
| Total | 66.71 ms | 52.60 ms |
| Equivalent FPS | 14.99 | 19.01 |

Pressure work covers 216 cavity vertices instead of all 2,744 particles.
Whole-scene color sizes are `[569, 569, 569, 519, 518]`; pressure color sizes
are `[6, 43, 51, 53, 63]`. The existing small-cavity fusion is now eligible.
This saves work but does **not** achieve 30 FPS.

All three complete-task runs passed per-frame checks and placed all four
parcels. Every run continued for 120 frames after completion. Final cloth
settling failed the unchanged 2 mm/s RMS limit in all three runs:

| Run | Completion frame | Cloth RMS speed | Final bag V/V0 |
| --- | ---: | ---: | ---: |
| Before | 5372 | 5.06 mm/s | 0.98082 |
| Broader caches, rejected | 5369 | 9.90 mm/s | 0.98130 |
| Narrowed opt-in mode | 5385 | 14.41 mm/s | 0.98126 |

These results do not establish preserved settling quality; the cache combination
therefore defaults to **off**. The narrowed mode is available for explicit
experimentation, not presented as an accepted replacement. No settling threshold
or physical parameter was relaxed. The effect of nonlinear contact variability
versus individual cache changes has not been isolated by these runs.

`validation.json` retains full numerical summaries and failed assertions. The
full-run mean timings are not used as a speedup claim because another GPU
application changed during those runs. The default-duration 9000-frame final
acceptance was not run.

All 34 selected conveyor and pneumatic regression tests passed, as did changed-file
pre-commit checks. The small-cavity fusion eligibility test was also executed
against the original committed solver and failed as expected before the change.
These unit results do not override the failed scene-settling acceptance above.
