# Bread cutting performance

RTX 5060 Ti 8 GB, 2026-10-23. Reference: `bc663566`, default 660-frame
trajectory. No physical parameter, resolution or iteration-limit changes
were adopted.

## Which rice optimizations carry over?

Both examples inherit `W1MJMPMScene` and share the MJMPM backend. Frame/IK
CUDA graphs, articulated link response, collapsed robot chains, cached
force-only MuJoCo evaluation, bounded point restriction, overlapped MPM
preparation and nonblocking GL event polling apply to both. Cooperative
rigid/contact-space paths are also enabled. However, bread has no independent
free rigid bodies; its environment rigid contact count was zero at every
sampled frame. Robot contacts belong to MuJoCo, so independent rigid-contact
speedups have less impact here.

Rice-specific visual batching did not carry over because bread uses a
different asset renderer. Bread also adds an implicit cuttable elastic/plastic
connection solve absent from rice.

| Configuration | Rice | Bread |
| --- | ---: | ---: |
| Particles | 1,478 | 7,376 |
| Grid voxel size | 15 mm | 6 mm |
| Active-cell capacity | 4,096 | 8,192 |
| Substeps per frame | 6 | 6 |
| MPM iteration limit | 50 | 50 |
| Additional material connections | 0 | 85,912 |
| Connection CG iterations per substep | 0 | 24 |

Capacity is an allocation limit, not an occupied-cell count.

## Measurements

`scripts/profile_mjmpm_bread.py` follows the real trajectory, synchronizes
after physics, and finishes OpenGL rendering at 1920 x 1080 when enabled.
These are synchronized offscreen throughput measurements, not interactive
window FPS. Warm statistics exclude frames 0-59; cutting uses frames 240-419.
Existing diagnostics remain enabled. Acceptance checks run outside the timed
interval, and validation runs are separate from comparison runs.

| Run | Cut physics | Cut rendering | Cut throughput | Whole warm throughput |
| --- | ---: | ---: | ---: | ---: |
| Reference, physics only | 41.68 ms | — | 23.99 FPS | 29.19 FPS |
| Reference, full appearance | 41.75 ms | 8.86 ms | 19.76 FPS | 22.79 FPS |
| Reference, repeated | 40.74 ms | 8.85 ms | 20.17 FPS | 23.38 FPS |
| Batched appearance | 41.81 ms | 7.36 ms | 20.34 FPS | 23.88 FPS |

The visual candidate merges adjacent opaque meshes sharing a parent and
material, reducing authored mesh batches from 13 to 6. It reuses immutable
GL instance colors/materials after first upload, as rice already does.
Triangles, normals, UVs, textures and material order are preserved. Frozen
1280 x 720 renders were pixel identical before and after, both at first
upload and at the subsequent cached update. The measured rendering saving
is about 1.5 ms. Physical timing varied between runs, so whole-frame speedup
is less certain than the rendering reduction. This is a modest display
improvement, not solver acceleration.

Separate CUDA-event runs identify these cutting-phase GPU costs:

| Region | Mean per frame |
| --- | ---: |
| Entire physical frame | 41.06 ms |
| MPM main step | 23.83 ms |
| MPM rheology solve, included above | 19.39 ms |
| MPM iteration loop, included above | 18.23 ms |
| Connection solve including assembly | 4.62 ms |
| Connection CG, included above | 4.09 ms |
| MuJoCo step | about 3.25 ms |
| Overlapped MPM preparation | about 5.61 ms |

Nested regions must not be added. Preparation overlaps rigid work: removing
that rigid work cannot recover its entire measured cost. The profiler times
the enclosing MPM iteration loop; events inside a conditional loop would be
overwritten on each execution. The GS solver has 8 colors, 8,192 strain slots
and 7,376 collider slots. These are array capacities, not active-contact counts.

## Experiments not adopted

* GS-SoA: cut physics 41.90 ms; whole warm 35.25 ms versus reference 41.68
  and 34.25 ms. No benefit; retain GS.
* Device-conditional connection CG: cut 41.17 ms; whole warm 34.08 ms.
  The small difference needs repeated timing and solution-error comparisons;
  retain fixed 24 iterations.
* Skip only the environment collision detector when all bodies are proxies:
  cut 40.58 ms; whole warm 34.98 ms. The overall result is mixed and much of
  this work overlaps preparation. Do not add this special case now.

These switches exist only in the profiling script and are disabled by
default. They do not change the production solver.

## Next solver work

1. Trace kernels inside the 18.23 ms MPM loop, distinguishing colored strain
   relaxation, collider contact, articulated response and residual reductions.
   Investigate repeated setup, memory traffic and redundant launches while
   preserving iteration order, operators and convergence checks. SoA alone
   did not help this scene.
2. Profile connection matrix-vector products and assembly. They use atomic
   scatter into particle vectors. Per-particle adjacency gather is a candidate,
   trading atomics for extra reads and duplicated edge evaluation. Even halving
   the entire 4.1 ms CG region saves only about 2 ms per frame.
3. Repeat full acceptance and compare physical errors for each adopted solver
   change. Keep resolution, connection count, substeps, iteration limits,
   collision coverage and material response.

## Validation and reproduction

The visual candidate passed all 660 frames of `test_post_step` and
`test_final`: pieces of 2,204 and 5,172 samples, zero detached samples,
1.91 mm retained notch, 3.48 mm recoil, 9.76 mm retained opening and 0.399 mm
minimum visual grip clearance. Peak sampled speed was 1.01 m/s. The physical
operators are unchanged; parallel floating-point evaluation produces small
run-to-run differences. Nine bread/cohesion/rice-geometry tests passed.
The new batching test checks triangles, normals, UVs and material order exactly.

From the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --output physics.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --events --output events.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --render --visual-batching reference --output reference.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --render --output batched.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --render --validate --output validation.json
uv run --no-sync python -B -m unittest newton.tests.mjmpm.test_bread_cut newton.tests.mjmpm.test_bread_cut_cohesion newton.tests.mjmpm.test_rice_cooker_geometry
```

Local raw reports, validation results and frozen-state images:
`C:/Users/Administrator/Documents/Blender/BreadCut/perf/`.

## Follow-up: connection gather and larger fused-feedback eligibility

The connection matrix-vector product now gathers each particle's neighbors
instead of scattering contributions from edges with atomics. Slot-major
adjacency is built once from the original topology. Cuts continue to zero
the same edge coefficients, and plasticity continues to update their rest
lengths. Neighbor membership never needs rebuilding. The implicit matrix,
block preconditioner, 24 CG iterations and tolerances remain unchanged.
Gather combines the identity term and connection terms in one kernel.
It uses about 1.5 MB of additional adjacency storage for this scene.

The existing fused rigid mobility path was limited to 4,096 collider rows.
Bread has 7,376 rows, 24 body slots and 22,128 allocated matrix blocks;
therefore this particular optimization did not carry over automatically.
Raising the row eligibility limit to 8,192 enables the existing kernel while
retaining the body-count, matrix-density, CUDA and static-row guards. The
mobility kernel itself scales with bodies; its sparse input/output products
are unchanged. Other configurations keep the original fallback.

| CUDA-event run, cutting | Connection CG | MPM iteration loop | Entire GPU frame |
| --- | ---: | ---: | ---: |
| Scatter/reference, repeated | 4.105 ms | 18.268 ms | 41.051 ms |
| Gather, original feedback limit | 3.240 ms | 18.218 ms | 40.231 ms |
| Gather, 8,192-row feedback limit | 3.249 ms | 17.875 ms | 39.876 ms |

These changes save about 1.2 ms per cutting frame in the measured GPU regions.
This is a small physical acceleration; it does not establish a 28 FPS target.
Changing GS launch blocks to 32 or 16 only moved the iteration loop from
18.218 ms to 18.112/18.124 ms. That is insufficient to change the default.
Those switches remain profiling experiments.

Fresh display comparisons holding the batched appearance fixed, without
timing events or acceptance checks:

| Physical path | Cut physics | Cut render | Cut throughput | Whole warm throughput |
| --- | ---: | ---: | ---: | ---: |
| Scatter plus reference feedback | 40.76 ms | 7.20 ms | 20.85 FPS | 24.32 FPS |
| Gather plus extended fused feedback | 39.57 ms | 7.23 ms | 21.37 FPS | 25.45 FPS |

The cutting physical saving is 1.19 ms (about 2.9%). Rendering is effectively
unchanged. Whole-trajectory results also depend on nonlinear iteration counts
and run-to-run floating-point variation; use the region timings to attribute
the changes. These remain synchronized offscreen measurements at 1920 x 1080.

Validation of the combined physical changes passed all 660 frames: pieces
of 2,204 and 5,172 samples, zero detached samples, 1.911 mm retained notch,
3.448 mm recoil, 9.771 mm opening and 0.399 mm minimum grip clearance.

The added CPU/CUDA gather test compares a multi-particle operator against an
independently assembled dense matrix, including severed edges, nonunit alpha,
nonzero beta, aliased output and CUDA graph replay. The fused-feedback test
now covers 4,104 rows, active-link changes, bitwise agreement with the fallback
and captured replay. Restoring the old 4,096-row limit makes its fast-path
coverage assertion fail, as expected.

All 65 MJMPM tests passed after these changes (113.3 seconds), as did the
repository-wide pre-commit checks and explicit checks of the new files.

For a comparison holding the batched appearance fixed:

```powershell
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --render --connection-product scatter --rigidity-feedback reference --output physics_reference_display.json
uv run --no-sync python -B scripts/profile_mjmpm_bread.py --render --output physics_optimized_display.json
uv run --no-sync python -B -m unittest discover -s newton/tests/mjmpm
```
