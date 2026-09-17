# Popcorn contact performance, 2026-09-12

**2026-09-13 update:** the ordinary example now uses the integrated coupled
full-space correction. Two consecutive final step-bound trials passed the
complete 48-second, 160-grain, 1920×1080-rendered sequence at **20.21 and
20.10 FPS**, retaining 22/27 and 22/25 grains respectively. No attachments,
lowered geometry quality or relaxed acceptance checks were used. See
[the chronological trial log](substep_trials.md#20-fps-acceptance-and-integration--2026-09-13)
for rejected fast variants and the scope of the experimental solver.
The earlier measurements below are historical, not the current default.

```powershell
uv run --no-sync python -m newton.examples mjvbd_v2_popcorn
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/validate_default.py
```

Target: retain the R2 elastic/plastic cup, friction-only grasp, 160 dynamic
grains, and the original complete-scene acceptance. No attachment, collision
filter, contact-capacity reduction or force-model change is used for speed.

## Measurements and experiments

RTX 5060 Ti, Warp 1.17.0, warm ViewerNull loop: 30 warm-up frames followed by
60 measured frames. Construction, compilation and rendering are excluded.
CUDA-event span includes host submission gaps; it is not pure kernel time.
The baseline is the staged R2 code, 16 substeps and 24 sweeps.

| Experiment | Wall ms/frame | Result |
| --- | ---: | --- |
| R2 baseline, four contact lanes per body | 199.214 | Previously accepted R2 scene |
| 16 shared contact lanes, unchanged solve budget | 159.674 | Timing probe only |
| 32 shared contact lanes, unchanged solve budget | 160.529 | Timing probe only |
| Separated rigid lanes, dense prefix, 20 sweeps | 138.961 | Rejected: grasp slipped |
| Final implementation, original 16 x 24 budget | 152.417 | Warm timing; see full acceptance below |

The timing probes changed both rigid/rigid and sparse rigid/particle lane
counts. The retained implementation separates the constants: only rigid/rigid
uses 32 lanes; the existing rigid/particle path remains at four lanes.
Unoccupied rigid lanes return before zero-valued atomic updates. Arithmetic
reduction order changes; bitwise identical long contact trajectories are not
claimed.

Dense rigid/particle partials now skip blocks beyond the actual clamped contact
count. The final reduction reads only that live prefix, so unused capacity
need not be cleared. Contact capacity is unchanged. A poisoned-NaN partial
buffer test with 257 contacts in 1024 slots covers the partly occupied last
block and stale unused blocks; the existing 1024-contact test remains.

**Rejected:** 12 substeps, 24 sweeps, coarse checkpoints every eight sweeps,
12 coarse PCG steps. Cup slip reached 33.9 mm at 23.583 s. This combined
experiment does not isolate which budget change caused failure. Restore
16 substeps, checkpoints every four sweeps and 32 PCG steps.

Additional regression covers 0, 1 and 37 rigid contacts, hard/soft contacts,
same/different body colors and an immovable endpoint, on CPU and CUDA.
CPU reference arrays must be copied before reusing output buffers: `.numpy()`
can alias CPU storage. The initial test failure from that alias was fixed in
the test, not by relaxing numerical tolerances.

Raw logs and trial images are outside the repo at
`E:/csy_work/CG/Engine/newton_cleanup_archive/popcorn-20260911-200805/`:
`r2-performance`, `perf16`, `perf32`, `fast12`, `fast16sweep`,
`perf-all-tests`, and `perf-dense-tests`.

Kernel census uses one uncaptured frame. Those timings identify hotspots but
must not be summed as a CUDA Graph frame-time decomposition. In the baseline,
rigid contact accumulation was 70.95 ms in that census; in the 32-lane probe it
was 22.38 ms. Only end-to-end loop timings are used for speed claims.

**Rejected:** 16 substeps, 16 sweeps, original coarse budget. The cup slipped
30.8 mm at 32.800 s, despite 19 grains having reached it. Delivery alone is
insufficient; keep the original grip stability criterion.

Reproduce warm wall time for the current defaults (not a full acceptance run):

```powershell
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/benchmark.py
```

Run timing probes alone on the GPU. The unit tests and full scene diagnostic
runs can overlap for correctness, but their wall times are not benchmarks.

**Rejected:** 16 substeps, 20 sweeps, original coarse budget. Cup slip reached
31.1 mm at 24.917 s. The attractive 138.96 ms timing is not an accepted speedup.
Restore the original 24 sweeps; all demo parameters now match staged R2.

The intermediate `fast24` run was deliberately interrupted when the final
active-prefix implementation was ready. It is not counted as a passed full run.

## Retained implementation

- Rigid/rigid accumulation: 32 strided lanes and one body per CUDA block,
  skipping lanes with no work. Sparse rigid/particle scheduling is unchanged.
- Dense rigid/particle accumulation: skip unused partial blocks and do not
  read their stale data during reduction.
- Particle-side unified contact stream: at most 4096 workers iterate over
  the actual clamped contact count instead of launching every capacity slot
  on every color/sweep. The same code handles points, edges and faces, including
  barycentric corner weights and color masks. All contacts are visited; there
  is no candidate reduction, lagged force cache or CPU count readback.
- Normal/friction formulas, capacities, DAT, paper plasticity, trajectories,
  160 grains, 16 substeps, 24 sweeps and the original coarse budget are unchanged.

The warm timing is 23.49% less wall time (1.307x throughput), not 2x and not
real-time. The initial test interval does not establish the same speedup in
every scene phase or in the GUI. Full-loop timing includes all stages but is
not directly comparable to the short baseline interval.

The mixed-contact regression exercises 4103-capacity point/edge/face records,
active counts 0/7/4097/overflow and color masks against the independent soft
kernel on CPU/CUDA. Original sparse, hard-contact and majorizer tests remain.

Full 48-second physical acceptance and timing, with no extra diagnostic viewer:

```powershell
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/benchmark.py --frames 2850 --validate
```

30 warm-up frames plus 2850 measured frames correspond to 48 simulated seconds.
The unchanged `test_final()` runs after the measured interval. Runtime slip,
IK and grain-speed guards remain active throughout.

### Full acceptance result

`perf-full48.log`, 48 simulated seconds, original defaults: **Pass**.

- Measured 2850 frames average **170.147 ms/frame, 5.877 FPS**, without rendering.
  Do not compare this whole-trajectory average directly with the baseline's
  60-frame initial interval; the comparable short interval is 199.214 to
  152.417 ms/frame, a 1.307x throughput gain.
- 22 grains lifted; 18 retained, 81.82%, 54 retention checkpoints.
- Five-finger contact coverage 100%; no tool recovery pauses or assertions.
- Maximum cup RMS distortion 1.7284 mm, plastic hinge offset 0.11212 rad.
  The material is unchanged; the contact trajectory is not bitwise identical
  to R2's 2.2754 mm / 0.17225 rad trial. Both show real elastic/plastic change.
- Cup lift maximum 109.86 mm; final lift passed the original 80 mm threshold.
  Minimum rim radius 30.61 mm, upright cosine 0.98945.
- Final shaft slip 0.704 mm; maximum IK position error 0.376 mm.

This is useful but **not a 2x or real-time result**. Lower-budget configurations
were rejected rather than used to inflate the performance claim. Further major
speedups require more work on the coupled solve, not disabling physical contacts.

No staging or commit was performed. Existing staged R2 work was preserved.

## Follow-up: DAT cache and parallel coarse restriction

The complete rigid-soft scene falls back from the external-rigid `surface-fast`
preset. Enable only its existing DAT fixed-geometry cache explicitly in this
demo; keep the original surface elasticity kernel. Coarse Galerkin restriction
now sums each CUDA cluster with 64 lanes instead of a single serial thread.
CPU restriction is unchanged. No contact, iteration, material or trajectory
budget was reduced.

The isolated 48-second DAT-cache / tiled-restriction run passed unchanged
acceptance (`dat-tile-full48.log`):

- 18 lifted grains, 15 retained (83.33%), 54 retention checkpoints.
- Five-finger coverage 100%; cup lift 109.99 mm, rim radius 30.72 mm,
  upright cosine 0.99111; no tool recovery pause.
- Cup RMS distortion 1.710 mm and plastic hinge rotation 0.10716 rad.
- 2850 measured frames: **160.648 ms/frame** (about 6.22 FPS), compared with
  the previous full-loop 170.147 ms/frame, a 5.58% time reduction.

This is still far from 30 FPS. Do not present the rejected short-window probes
as accepted acceleration. The combined surface/DAT cache failed the cup-slip
guard; the batched rigid reaction experiment failed the 70% retention test.
Both were removed from production changes. See `soft_reaction_batch.md` for
their details. The dense conditional graph prototype is only in the diagnostic
script, not a new production solver branch.

97 relevant regression tests passed after integrating the retained change,
including ragged coarse sums against double precision and CUDA graph replay
with updated plastic rest angles (`dat-tile-final-tests.log`). The full scene
was tested through the diagnostic launch adapter before integration; production
uses the same tested tiled kernel and the same DAT option directly.

Exact production-default short benchmark: **141.151 ms/frame, 7.085 FPS**
(`dat-tile-default-timing.log`), versus the preceding comparable 152.417 ms/frame.
The separate 26-test multilevel suite also passed (`coarse-main-regression.log`).

Final validation: 79 contact/material/asset regression tests pass; selected-file
Ruff checks, formatting and `git diff --check` pass. Logs:
`perf-implementation.log`, `perf-full48.log`, `perf-regression-final.log`
in the external archive above. No regression claim is made for every existing
demo; the numerical regression suite and the complete popcorn workflow were run.
