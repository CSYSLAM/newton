# MJMPM rice-cooker physics performance log

Live log of the performance pass on `mjmpm_w1_rice_cooker`. Configuration is
held fixed throughout: 1,478 rice samples, 6 substeps, 60 rigid contact
sweeps, MPM at most 50 iterations, `rigid_material_coupling="constrained"`,
CUDA graph replay. RTX 5060 Ti 8 GB, Warp 1.18.0, Windows.

Tools (scratchpad, not committed): `rice_bench.py`

- `time`: wall time per frame for graph replay, synchronized every frame,
  frames 60-990 (first 60 are warm-up).
- `phases`: CUDA events recorded inside the captured frame graph around each
  solver phase; replay, then read elapsed times. Median of 20 frames at the
  carry (frame 300) and pour (frame 515) windows. MPM preparation runs on a
  side stream that overlaps the rigid contact phase, so its row does not add
  to the main-stream total.
- `iters`: device-side histogram of MPM solver iterations per solve.

## 1. Baseline phase breakdown (HEAD 89c2e178)

Frames 60-540, graph replay: median 59.2 ms, p90 74.4 ms. Rigid contacts per
frame: median 74, max 110.

| Phase | ms/frame | share | calls/frame |
|---|---:|---:|---:|
| frame total | 57.3 | 100% | 1 |
| environment step (all) | 49.8 | 86.9% | 6 |
| **rigid contact solve** | **28.6** | **49.9%** | 12 |
| of which GS sweeps (`solve_contact_buckets`) | 28.1 | 48.9% | 12 |
| constrained: finish (contains the 2nd rigid solve) | 14.3 | 25.0% | 6 |
| MPM step (main stream) | 12.7 | 22.2% | 6 |
| MPM iterations | 6.9 | 12.0% | 6 |
| rigid collision detection | 6.0 | 10.4% | 6 |
| MPM prepare [side stream, overlapped] | 5.5 | 9.6% | 6 |
| MPM contact: rasterize colliders | 2.4 | 4.2% | 6 |
| MPM contact: rigidity operator | 2.0 | 3.5% | 6 |
| MPM contact: plasticity system | 0.8 | 1.4% | 6 |
| constrained: prepare B_c | 0.6 | 1.1% | 6 |
| IK (per frame) | 0.6 | 1.1% | 1 |
| MPM update: save warm start | 0.5 | 0.9% | 6 |
| constrained: predict | 0.2 | 0.3% | 6 |
| MPM update: particles / save data, collect impulses, warm start | <0.1 each | | 6 |

MuJoCo plus coupling bookkeeping is the remainder: 57.3 - 49.8 - 0.6 = about
6.9 ms.

Top two:

1. Rigid GS sweeps, 28 ms (49%). One thread per world runs 60 sequential
   sweeps over ~74 contacts: about 2.3 ms per call, 0.5 us per contact update,
   so the kernel is latency-bound on a dependent chain of global loads and
   read-modify-writes (each update rewrites the velocities of every contacted
   finger link through the precomputed link responses). It runs twice per
   substep: the main solve and the reaction-recovery re-solve in
   `_finish_constrained_material`.
2. MPM, 12.7 ms main stream (22%) plus 6.0 ms collision detection.

### MPM iterations per solve (baseline configuration)

Device-side histogram over the full 990-frame run (5,934 solves; the solve
loop checks convergence every 5 iterations, so counts are 1 + 5k):

| iterations | 6 | 11 | 16 | 21 | 26 | 31 | 36 | 41 | 46 | 51 (cap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| solves | 4537 | 695 | 335 | 99 | 62 | 39 | 26 | 20 | 18 | 103 |

Mean 8.9, median 6, p90 16; 1.7% of solves reach the cap. MPM time is mostly
per-solve fixed cost (assembly, rasterization, rigidity operator), not
iterations, so lowering `max_iterations` would buy little even if it were
allowed.

## 2. Optimizations

### 2.1 Rigid GS sweeps on bucket-local state (kept)

Change: `solve_contact_buckets_local` (`rigid/kernels.py`). Per bucket, a
pre-pass packs the sweep-invariant contact data in sweep order
(`PackedContact`: bodies, inverse masses and inertias, link slots, whether
each side uses precomputed link responses, the point mobility, contact
frame and parameters) and gathers the bodies the bucket touches into
thread-local component vectors (up to 32 bodies). The 60 sweeps then run with
the reference arithmetic and order on that local state, and velocities and
body impulses are written back once. Buckets with more than 32 bodies, links
of several worlds, or links that need mobility products set a flag and are
solved by the unchanged `solve_contact_buckets`, which now skips unflagged
buckets.

Why it helps: the reference sweep issues a chain of dependent global loads
per contact (`ids -> bodies -> mass/inertia/link slot -> contacted_count ->
contacted_links -> link_body -> qd`) and read-modify-writes the same few
velocities in global memory every contact, about 10 serialized round trips.

Micro-benchmark (`gs_micro.py`, the 12 launches of one frame, inputs dumped
from the example):

| variant | frame 515 (71 contacts) | frame 300 (76-91 contacts) | bitwise |
|---|---:|---:|---|
| reference | 2.16 ms/launch | 3.06 ms/launch | - |
| local state only (matrix rows) | 3.86 | 5.57 | yes (Warp copies the whole matrix per row write) |
| packed data, global state | 1.63 | 2.33 | no (FMA contraction differs, 5e-6) |
| packed + local component vectors (V3) | 1.75 | 2.52 | yes |
| V3 + packed response flags and point mobility (V4, kept) | **1.42** | **2.00** | yes |

Phase breakdown after the change (same windows): frame 57.3 -> 47.6 ms;
rigid contact solve 28.6 -> 19.0 ms (GS sweeps 28.1 -> 18.4 ms); constrained
finish 14.3 -> 9.4 ms; everything else unchanged within noise.

Full trajectory, frames 60-990, alternating runs (A = reference sweeps via
`_use_local_rigid_buckets=False`, B = local):

| run | A median | A p90 | B median | B p90 |
|---|---:|---:|---:|---:|
| 1 | 56.90 | 69.91 | 45.65 | 56.09 |
| 2 | 57.91 | 73.60 | 46.09 | 56.29 |
| 3 | 57.19 | 72.09 | 46.43 | 56.01 |
| median | 57.19 | 72.09 | 46.09 | 56.09 |

Median frame time -19.4%, p90 -22.2% (run-to-run spread about 1 ms).

Correctness:

- New test `test_local_contact_buckets_match_reference`: inside every
  solve, sweeps the same inputs with the reference and the local kernel and
  requires bitwise equal velocities, contact impulses and body impulses, on
  CPU and CUDA, for single- and multi-world box stacks (hard and compliant
  contacts), a 40-box stack that exceeds the local capacity (fallback), and
  an arm striking a box with articulated link responses. (Whole-trajectory
  comparison is bitwise on CPU; on CUDA the detected contact order already
  varies between runs with the reference kernel, so trajectories diverge
  after ~20 steps either way.)
- Rice cooker `--test`: PASS (slip 3.39 mm / 3.12 deg, 100% in pot, 0%
  spilled, peak grain 1.94 m/s; baseline 3.34 mm / 3.28 deg, 100%, 0%, 1.93
  m/s; the example varies run to run at this level).
- CUDA graph capture and replay unchanged (the bench runs graph replay).

## 3. Breakdown after 2.1 and next candidates

Refined phases (commit 95d605ec, same windows; collision and MuJoCo split
out): frame 46.6 ms.

| Phase | ms/frame | share | calls/frame |
|---|---:|---:|---:|
| rigid GS sweeps (2 per substep) | 17.5 | 37.6% | 12 |
| MPM step (main stream) | 12.6 | 27.1% | 6 |
| of which MPM iterations | 6.7-6.9 | 14.4% | 6 |
| of which rasterize colliders | 2.5 | 5.3% | 6 |
| of which rigidity operator | 1.9 | 4.1% | 6 |
| of which plasticity system | 0.9 | 2.0% | 6 |
| collision detection | 6.2 | 13.3% | 6 |
| of which narrow phase (mesh-mesh SDF) | 5.2 | 11.2% | 6 |
| of which contact sort / match / broad phase | 0.8 | 1.6% | 6 |
| MuJoCo step (main + feedback re-solve) | 4.0 | 8.7% | 12 |
| MPM prepare [side stream, overlapped] | 5.5 | 11.8% | 6 |

MPM iterations in these windows average 12.5 per solve, so one iteration
costs about 92 us. An iteration launches 8 Gauss-Seidel color kernels (the
P0 strain graph has 8 colors) plus about 6 contact kernels (nodal friction,
two BSR products and the link/independent response maps), about 6.5 us per
graph node: the iteration is launch-latency bound, not compute bound.

Tried and rejected for the GS sweep kernel:

- statically unrolled link loop (`range(MAX_RESPONSE_LINKS)` with a guard):
  1.39 -> 1.52 ms/launch, slower.
- thread-local matrix rows for body state: 2x slower (Warp copies the whole
  matrix on every dynamic row write); per-component vectors avoid this.

Tooling notes: Nsight Systems 2024.6 records no CUDA activity with the
installed CUDA 13.3 driver; CUDA events recorded inside the captured graph
work and are what `rice_bench.py phases` uses. Kernel timing without the
graph (`wp.timing_begin`) inflates small kernels and memsets by several
times and is not usable for this scene.

Candidates, by expected gain:

1. GS sweeps are still the top item (37.6%), single-threaded and now
   compute-latency bound (~330 ns per contact update). Half of it is the
   reaction-recovery re-solve in `_finish_constrained_material`; making it
   cheaper (warm start from the first solve, fewer sweeps) changes results
   and belongs to the second batch.
2. MPM iterations (14%): fuse the 8 color launches into one single-block
   kernel that loops over colors with block-level synchronization when the
   strain node count fits one block, and fuse the contact-side launches;
   requires a capacity check and fallback to the multi-launch path.
3. Mesh-mesh SDF narrow phase (11%): shared Newton geometry code; check the
   launch dimension against the active pair count before touching it.
4. Rasterize colliders and rigidity operator (9%): rebuilt every substep;
   check which parts depend only on body poses.

### 2.2 Single-block launch for all Gauss-Seidel strain colors (measured, reverted)

Not kept: about 1% per frame did not justify a native block barrier, a
device-flag `capture_if` and about 400 added lines in the MPM solver. The
measurements below are kept for reference.

Change: `make_gs_solve_fused_kernel` and `apply_stress_gs_fused`
(`rheology_solver_kernels.py`) run all colors of the strain Gauss-Seidel
sweep in one launch of one 256-thread block, with a block barrier
(`__syncthreads` through `wp.func_native`) between colors. The per-color
body is shared with the existing kernels (`_make_gs_solve_color`,
`_apply_stress_color`), so order and arithmetic are unchanged. At the start
of each solve, `gs_fused_condition` sets a device flag when no color holds
more than 512 blocks, and `wp.capture_if` picks the fused or the per-color
path, so a captured graph stays valid if the grid grows.

Why: in this scene a color holds 4-41 blocks of one strain node each, while
the per-color launch starts 36 SMs x 2 x 64 = 4,608 threads per color.

Measured cost split inside the rheology solve (phases with `FINE=1`, frame
300 window, 14.4 iterations per solve):

| part | ms/frame |
|---|---:|
| solver loop (`_run_solver_loop`) | 3.74 |
| strain eigenbasis build | 1.84 |
| GS initial guess + solver setup | 0.34 |
| other (contact solver setup, delassus, eigenbasis apply) | ~1.2 |

So the iterations were never the whole "MPM iterations" phase: with the GS
and contact sweeps replaced by no-ops (diagnostic only) the phase still
took 5.7 ms/frame.

Effect on the solver loop (same windows, two runs each): per-color 3.76 /
4.05 ms at 12.3 / 12.8 iterations, fused 3.27 / 2.95 ms at 13.3 / 12.8
iterations; per iteration ~51 us -> ~40 us (-22%).

Full trajectory, alternating runs (A = per-color via
`FUSED_MAX_COLOR_BLOCKS = -1`, B = fused):

| run | A median | A p90 | B median | B p90 |
|---|---:|---:|---:|---:|
| 1 | 46.42 | 56.08 | 45.88 | 55.39 |
| 2 | 46.10 | 56.81 | 45.74 | 55.16 |
| 3 | 46.18 | 55.74 | 44.92 | 54.73 |
| median | 46.18 | 56.08 | 45.74 | 55.16 |

Median -1.0%, p90 -1.6%; B is faster in every pair, but the gain is close to
the run-to-run spread.

Correctness: new test `test_fused_gauss_seidel_colors_match_per_color_launches`
runs every `apply_initial_guess` and `solve` of a sand-and-box scene through
both paths on the same inputs (eager steps and graph replay, comparison on
device) and requires zero mismatches; with the fused kernel deliberately
skipping the last color it reports 23,044 mismatches. Rice cooker `--test`
PASS (slip 3.36 mm / 3.74 deg, 100% in pot, 0% spilled, peak grain 1.96
m/s).

### 2.3 Rigid GS local capacity (closed)

Same-input micro-benchmark, all bitwise identical to the reference:

| local bodies | frame 515 | frame 300 |
|---:|---:|---:|
| 8 | 1.399 ms | 1.971 ms |
| 16 | 1.408 ms | 1.987 ms |
| 32 (kept) | 1.392 ms | 1.966 ms |

No spill effect: per-component vectors only touch the slots in use. Buckets
here hold 3 bodies (two finger links and the cup) and 2 contacted links. The
remaining time is the serial dependence of Gauss-Seidel itself; further
gains need algorithm changes (second batch).

### 2.4 Mesh-mesh SDF narrow phase work (measured, change reverted)

Instrumented the narrow phase inside the captured graph (per-launch events)
and read the mesh-mesh pair and block counts at three stages:

| stage | mesh-mesh pairs | blocks with work / launched | SDF kernel ms/frame | narrow phase ms/frame |
|---|---:|---:|---:|---:|
| approach (frame 30) | 0 | 0 / 576 | 0.085 | 2.34 |
| carry (frame 300) | 2 (finger pads x cup shell) | 33 / 576 | 3.59 | 6.02 |
| pour (frame 515) | 2 | 33 / 576 | 3.75 | 6.02 |

Idle blocks are cheap (the empty launch costs 0.085 ms per frame). The cost
is the work itself on only 33 blocks for 36 SMs: `compute_block_counts_from_weights`
gives every block at least 256 edges (`wp.max(256, total_weight //
target_blocks)`), so 8,248 edges get 33 blocks instead of the 144 the
partition targets. The floor dates from #2215, where it was a per-block
triangle/vertex count.

Experiment: make the floor a parameter for the mesh-mesh path only. On
identical states the reduced contact sets are bitwise identical for floors
256/128/64/32/16 (83 contacts at frame 300, 70 at frame 515), and `collide()`
drops from 0.77-0.79 ms to 0.52-0.58 ms per call (floor 32), about 1.3 ms per
frame (2.8%).

Not kept:

- With floor 32, full-frame graph replay hangs (GPU at 100%, the frame never
  completes) shortly after the grasp starts (frames 150-190). It does not hang
  with floors 64 or 128 (240 frames), with floor 256 and only 144 launched
  blocks, in eager steps (also with 144 launched blocks), with the MPM prepare
  side stream disabled, or for single eager `collide()` calls on those
  states. The in-place exclusive scan of the block offsets checks out in graph
  replay (2,000 replays). The hanging kernel was not identified: events
  recorded in the graph do not report completion to `is_complete`, and Nsight
  Systems records nothing on this driver.
- The gain is below the 3% bar, and lowering the floor exposes what looks like
  a latent race or deadlock in shared collision code that needs its own
  investigation (worth reporting upstream with the reproduction: rice-cooker
  example, floor 32, graph replay with the material side stream).

### 2.5 Direct strain mass diagonal for one-node strain bases (kept)

Cost of `_build_strain_eigenbasis` before the change (frame 300 window,
events around its parts; the instrumentation itself adds time, 1.84 ms/frame
without it):

| part | ms/frame |
|---|---:|
| `fem.integrate(mass_form)` (BSR assembly from triplets) | 1.51 |
| `bsr_copy` to element blocks | 1.23 |
| `compute_eigenvalues` | ~0.1 |

With the P0 strain basis an element has one node, so the PIC mass matrix is
diagonal and each 1x1 "eigendecomposition" is trivial; only the node volume
and `rotated_volume` are used afterwards (the eigenvectors are applied only
for more than one node per element).

Change: for `MAX_NODES_PER_ELEMENT == 1`, integrate the diagonal as a linear
form (`node_mass_form`, `q(s) * inv_cell_volume`, which equals the bilinear
`p(s) * q(s) * inv_cell_volume` for 0/1 shape values) and finish with
`single_node_eigenvalues`, which reproduces `compute_eigenvalues` for 1x1
blocks: a zero mass stands for a block that sparse assembly prunes (volume
and rotated volume 0), a mass below the eigenvalue floor gives 1 and 1, and
otherwise the volume becomes the mass and the rotated volume `sign(s) * s` of
the previous node volume. The general path moved to
`_assemble_strain_eigenbasis`, unchanged.

Equivalence: running both paths on the same inputs at every solve of the
example in eager mode (3,360 solves, frames 0-559, up to 287 active strain
nodes) gave identical volumes and rotated volumes (relative difference 0, no
branch mismatches). New test `test_single_node_eigenbasis_matches_assembled`
does the same comparison for a sand-and-box scene on CUDA (rtol 1e-6); with
the fast path deliberately doubling the volume it fails (100 mismatches).
(On CPU, after other tests in the same process, warp.fem fails to generate
`fill_element_node_indices` for this scene; the test therefore runs on CUDA
with a fixed grid like the other material tests.)

Phase time of the eigenbasis build: 2.72 -> 0.51 ms/frame (both runs carry
the part instrumentation).

Full trajectory, alternating runs (A = previous method restored from git,
B = new). The machine ran about 3 ms slower overall in this session than in
2.1-2.2; the alternation cancels that out:

| run | A median | A p90 | B median | B p90 |
|---|---:|---:|---:|---:|
| 1 | 48.87 | 60.09 | 47.56 | 58.36 |
| 2 | 50.12 | 61.42 | 46.82 | 58.69 |
| 3 | 49.20 | 59.79 | 47.86 | 59.08 |
| median | 49.20 | 60.09 | 47.56 | 58.69 |

Median -3.3%, mean -3.3% (50.05 -> 48.41), p90 -2.3%; B is faster in every
pair. Rice cooker `--test` PASS (slip 3.34 mm / 3.48 deg, 100% in pot, 0%
spilled, peak grain 1.91 m/s). MJMPM tests 54/54.

### 2.6 Rigidity assembly and recovery warm-start probes (measured, experimental)

Probed at `2930c795`. Reproduction scripts:

- `scripts/profile_mjmpm_physics.py --output-dir <directory>` records CUDA
  events inside the frame graph and freezes contact inputs for offline GS
  comparisons. Instrumented timings are not production FPS.
- `scripts/benchmark_mjmpm_recovery.py --output-dir <directory>` runs the
  unchanged 60/60-sweep task; add `--warm-start --recovery-sweeps 20` (or
  40/60) to experiment with the recovery solve only.
- `scripts/test_mjmpm_recovery_experiment.py <report.json>` runs existing
  MJMPM tests with process-local recovery warm starts and a recovery budget
  of `min(20, configured sweeps)`. It calls the real solvers, not mocks.

All scripts run from the repository root. They leave production solver
source and defaults unchanged. These new probes complement the earlier
scratchpad tools; raw results for this session are under
`C:/Users/Administrator/Documents/Blender/RicePour/physics_optimization_probe`.

Rigidity build, median of 62 frame samples (after 60 warm-up frames, sampled
every 15 frames over the complete 990-frame trajectory):

| part | ms/frame |
|---|---:|
| `build_rigidity_operator` total | 1.886 |
| fill coupling triplets | 0.033 |
| triplets to BSR: J | 0.888 |
| triplets to BSR: IJtm | 0.851 |

Each part is called six times per frame. Nested event overhead is included;
the medians do not add exactly. Each build has 1,478 collider nodes, 25
bodies, and zero deformable collider vertices. Generic BSR construction is
the substantial fixed cost; this measurement does not isolate sorting
from the other operations inside `bsr_from_triplets`. A rigid-only direct
construction is worth a bounded experiment, retaining the general path.
IJtm includes mass/inertia and weighting, so it is not simply J transposed.

GS probes freeze both solves at the first and last substeps of 13 frames
between 120 and 960: 52 input sets. Each budget starts from the same input
and applies its initial impulses correctly. For the 26 recovery inputs,
also try the corresponding first solve's 60-sweep impulse as a warm start.
Diagnostics never write their solved states back into the live trajectory.
120 sweeps is a comparison budget, not an exact solution.

Recovery normal projection residual in frames 240-600 (median, m/s):

| guess | 20 sweeps | 40 sweeps | 60 sweeps | 120 sweeps |
|---|---:|---:|---:|---:|
| current initial guess | 0.005360 | 0.004397 | 0.003031 | 0.001311 |
| first solve's 60-sweep impulse | 0.002180 | 0.001636 | 0.001423 | 0.000436 |

The residual evaluates one contact projection at frozen final velocities;
normal impulse correction is scaled by local inverse effective mass plus
softness to obtain m/s. This is not just the last sweep's impulse difference
and is not a global convergence proof. Warm starting helps early accuracy,
but does not make a smaller iteration budget equivalent in all inputs.

Full-task trials, one run per variant followed by a repeated baseline:

| recovery | median ms | p90 ms | slip mm | turn deg | task |
|---|---:|---:|---:|---:|---|
| current, 60 sweeps | 44.48 | 53.75 | 3.36 | 4.08 | PASS |
| warm, 20 sweeps | 38.88 | 46.15 | 3.32 | 2.56 | PASS |
| warm, 40 sweeps | 42.27 | 51.33 | 3.31 | 1.52 | PASS |
| warm, 60 sweeps | 45.19 | 57.75 | 3.30 | 1.69 | PASS |
| current, 60 sweeps, repeated | 43.82 | 53.10 | 3.32 | 2.97 | PASS |

Initial solves remain at 60 sweeps. All trials pour 100% into the pot with
0% spilled, and pass the existing load checks. Timed steps include normal
test telemetry; post-step checks and extra residual downloads are outside
the timed interval. No rendering. Warm 20 saves about 11-13% against the two
baseline runs; this is exploratory, not three alternating A/B pairs.

Important limitation: successful task checks do not establish unchanged
coupling accuracy. Maxima sampled every 60 frames, not per-frame maxima:

| recovery | linear momentum residual, N s | recovery angular velocity mismatch, rad/s |
|---|---:|---:|
| current, 60 | 5.26e-5 | 0.00627 |
| warm, 20 | 8.29e-5 | 0.0377 |
| warm, 40 | 2.98e-4 | 0.0719 |
| warm, 60 | 3.77e-4 | 0.154 |
| current, 60, repeated | 6.56e-5 | 0.00709 |

MPM-versus-accepted collider velocity mismatch remains around 1e-9 or below,
but recovery-versus-accepted velocity and momentum residuals change. More
recovery iterations do not monotonically lower these trajectory peaks.
One possible mechanism is that the first solve defines the frozen B_c and
accepted material velocity, while the warm-started recovery follows a
different finite-iteration path. This explanation needs a controlled test;
CUDA contact ordering also varies between trajectories.

Current unittest discovery ran 49 MJMPM tests with the experimental path:
all PASS, zero skips (37 constrained GS environments instrumented). This
does not override the residual limitation or prove cross-scene equivalence.
Keep the warm recovery experiment out of the production default. Prioritize
rigid-only BSR construction; investigate reaction/accepted-velocity
consistency before enabling recovery warm starts or early termination.

### 2.7 Direct rigid coupling BSR assembly (kept)

Change: a rigid node has at most two distinct block columns, angular then
linear, already in sorted order. Keep `fill_collider_coupling_matrices`
unchanged, then construct compact J rows and IJtm rows directly. Integer
counts and prefix scans determine storage; a warp-sized tile compacts each
IJtm row in increasing node order. No floating-point reduction or
approximation is introduced. J and IJtm independently prune exactly zero
blocks, including singular-inertia and zero-volume cases.

The previous `3 * node_count` nnz upper bound is retained deliberately:
`bsr_mv` uses it to select its reduction kernel. Reducing that bound to the
actual two-block maximum could change floating-point reduction ordering.
The fast path is bounded to 4,096 nodes, 64 bodies, and node-by-body-row work
of at most 262,144. Larger domains and all deformable colliders keep the
unchanged triplet assembler. Both CPU and CUDA are supported.

No changes to physical parameters, contact order, warm-start policy,
60/60 rigid sweeps, MPM iteration limit/tolerances, constrained response, or
rendering. In particular, the recovery experiment in 2.6 is not enabled.

Validation:

- `test_rigid_coupling_assembly_matches_triplets` compares compact offsets,
  active columns, float32 value bits, and matrix-vector products on CPU and
  CUDA. It includes COM nodes, minimum positive float32 values, zero and
  singular inverse inertia, infinite/static mass, zero cell volume, empty
  input, deformable fallback, and both sides of each fast-path capacity
  boundary. CUDA replay changes the contact set from empty to active and
  back to a single active node.
- On a complete 990-frame rice-cooker trajectory, build the direct and
  reference operators from the same input at every substep and compare on
  device: **11,880 matrices, zero bitwise mismatches**. This instrumented
  run passes the complete task checks; its time is not used as a benchmark.
- Full MJMPM unittest discovery: **50/50 PASS** (103.7 s). Expanded boundary
  checks also pass independently after the final capacity guards.
- Ruff checks and formatting pass.

Complete-trajectory timing, three alternating pairs, frames 60-989. Same
RTX 5060 Ti, 1,478 rice samples, six substeps, CUDA graph, no rendering;
test telemetry remains enabled identically in both variants. Post-step
checks and additional residual downloads are outside the timed step.

| run | generic median ms | direct median ms | generic p90 ms | direct p90 ms |
|---|---:|---:|---:|---:|
| 1 | 44.03 | 42.45 | 53.60 | 52.80 |
| 2 | 44.11 | 42.47 | 53.07 | 52.21 |
| 3 | 44.13 | 42.23 | 53.43 | 52.44 |
| median | 44.11 | 42.45 | 53.43 | 52.44 |

Median physical step time **-3.77%** (1.66 ms), p90 **-1.87%**. Every pair
improves. The corresponding short 135-frame instrumented assembly probe
falls from 1.84 to 0.38 ms/frame; this is a short-window stage measurement,
not a new whole-task FPS measurement.

All six timed trajectories pass: 100% into the pot, 0% spilled. Generic
slip ranges 3.30-3.40 mm and turn 3.32-4.48 degrees; direct slip ranges
3.33-3.37 mm and turn 3.43-3.69 degrees. Finger loads and grain-speed checks
also pass. Independent CUDA trajectories are not claimed bitwise equal:
their existing atomic/contact ordering varies between runs. The controlled
same-input operator comparison, rather than trajectory equality, establishes
that this assembly change preserves the numerical operator.

Residual samples remain available in every run JSON. The median of each
run's sampled recovery-angular-mismatch maximum is 0.00873 rad/s for
generic and 0.00795 rad/s for direct; the largest individual sampled peak
is 0.0168 versus 0.0519 rad/s. Thus these few independent trajectories do
not establish identical residual envelopes; no tighter residual bound or
convergence improvement is claimed. The exact-operator comparison and
unchanged iteration/reduction rules distinguish this change from the
non-equivalent recovery warm-start experiment.

Reproduction: `scripts/benchmark_mjmpm_rigidity.py --assembly generic
--output <run.json>` or `--assembly direct`; add `--verify` to compare
operators during the trajectory (exclude this mode from performance
measurements). Raw JSON and logs are under
`C:/Users/Administrator/Documents/Blender/RicePour/rigid_assembly_optimization`.

### 2.8 Reusing contact preparation during reaction recovery (reverted)

Tested reusing the first rigid solve's body transforms, contact geometry,
bucket ordering, and articulated contact responses in the second solve of
the same substep. Neither MPM nor the material impulse application changes
these inputs before rigid positions are integrated. Recovery still used
the original cold start or the same projected previous-step warm start,
and all 60 sweeps. No first-solve impulse warm start was introduced.
Jacobi and the reference solver retained their existing paths.

Correctness evidence for the candidate:

- CPU and CUDA same-input comparisons passed for hard/soft contacts,
  cold/previous-step warm starts, one/three worlds, and an articulated arm
  interacting with both a free box and particles. Every comparison checked
  the float32 bits of body velocities, contact impulses, and body impulses.
- A separate full 990-frame CUDA-graph trajectory compared cached versus
  rebuilt recovery at every substep: **5,940 solves, 17,820 output arrays,
  zero bitwise mismatches**. The complete task also passed. Its doubled
  solves are excluded from performance measurements.

Performance evidence, three alternating full-trajectory pairs, using the
direct BSR path from 2.7 in both variants and the same timing/check protocol:

| run | rebuild median ms | reuse median ms | rebuild p90 ms | reuse p90 ms |
|---|---:|---:|---:|---:|
| 1 | 43.61 | 42.93 | 52.80 | 51.26 |
| 2 | 42.57 | 42.99 | 52.25 | 51.94 |
| 3 | 42.77 | 42.95 | 53.14 | 52.61 |
| median | 42.77 | 42.95 | 52.80 | 51.94 |

The median of run medians is **0.41% slower**; only one of three pairs
improves that metric. The median of run means improves 0.72% and p90
improves 1.63%, but this does not establish a consistent improvement in
typical frame time. All six timed trajectories pass, with 100% of rice
inside the pot and 0% spilled. Independent CUDA trajectories retain their
existing contact/atomic-order variation; the controlled comparisons above
establish same-input reaction equivalence, not trajectory determinism.

Decision: **revert the candidate**, including its extra warm-start storage
and production dispatch helpers. Keep the accepted direct BSR optimization
in 2.7. Small kernel-count savings alone are insufficient to claim faster
physics. The next investigation should target time inside the rigid GS
sweeps, particularly repeated articulated response loads and updates, using
frozen identical inputs before making another production change. Preserve
sweep order, iteration counts, and reaction accuracy throughout that work.

Raw results, verification counters, the candidate environment patch, and
its benchmark/test snapshots are archived at
`C:/Users/Administrator/Documents/Blender/RicePour/contact_preparation_optimization`.
To reproduce the rejected experiment in a disposable checkout containing
2.7, apply `candidate_environment.patch` and run the archived
`candidate_benchmark_mjmpm_rigidity.py` with `--preparation rebuild` or
`--preparation reuse`; use `--verify-recovery` only for correctness checks.
These options are intentionally absent from the retained benchmark script.
After the revert, the retained production tree passes all **50 MJMPM
tests** (109.4 s); Ruff checks, formatting, and `git diff --check` pass.

### 2.9 Warp-cooperative contact sweeps and display caching (checkpoint)

Checkpoint `a84a6331` was committed and pushed to
`fork/feature/mjmpm-two-way-grasp` before this work. The changes described
below form the next checkpoint on that branch. They are not yet a
verified claim of sustained 25 FPS.

Retained changes:

| Area | Implementation | Scope and qualification |
|---|---|---|
| Rigid GS | Compact prepared contacts with valid bodies, preserve their relative order, skip exactly zero impulse updates | Retain all 60 sweeps in each of the two contact solves per substep |
| Articulated contact GS | One CUDA warp per bucket; distribute body velocity and impulse updates over lanes, broadcast point velocities, precompute the contact's combined point mobility | Enabled for articulated CUDA environments; unsupported buckets fall back, and independent free-body worlds retain their scalar path |
| Link mobility | Factor matrices up to 32 DOFs using a warp and shared memory | Preserve per-entry Cholesky arithmetic order; CPU and larger matrices retain the original factorization |
| Material feedback | Fuse articulated and independent mobility maps, removing an intermediate copy | Keep both sparse matrix-vector products; retain the general path outside the bounded supported case |
| Display | Reuse immutable instance colors/materials/opacities; merge adjacent opaque parts with identical parent and material | Keep all geometry, rice instances, texture maps, transparency, and draw order |

The cooperative GS still uses the same contact ordering and physical
contact law, but its precomputed point mobility and tangent calculations
reassociate float32 arithmetic. It is **not bitwise equivalent** to the
scalar reference. The final candidate restores the original normal
division and tangent normalization rather than retaining the reciprocal
and rescaled-normalization experiment.

Configuration remains **1,478 rice particles, 6 substeps, 60/60 rigid
sweeps, MPM iteration cap 50**, with CUDA graphs enabled. No physics
tolerance was relaxed. The supported-path dispatch does not replace
Jacobi, CPU, or oversized-bucket fallback behavior.

Measured evidence, with timing protocols kept separate:

| Version and protocol | Full-trajectory FPS | Grasp through pour FPS | Pour FPS |
|---|---:|---:|---:|
| `a84a6331`, offscreen, synchronized | 18.32 | not recorded | not recorded |
| Combined cooperative/cached-display stage, offscreen, synchronized | 24.91 | 23.62 | 24.37 |
| Later factor/point-broadcast stage, offscreen, synchronized | see raw JSON | 24.28 | 24.95 |
| Visible-window stage, synchronized | see raw JSON | 23.36 | 23.91 |
| Latest candidate, visible window, native application timing | 23.66 | 22.51 | 23.35 |

These are successive development stages, not repeated alternating pairs
of one final revision. They establish improvement and its approximate
scale; they do not establish a final speedup confidence interval. The
latest visible-window run requested 1920x1080, but Windows provided a
1920x1061 framebuffer. Offscreen comparisons use 1920x1080. Record actual
framebuffer sizes in future comparisons.

The synchronized protocol inserts device synchronization and `glFinish`
to attribute GPU cost to physics versus display. Native timing omits
those per-frame diagnostic barriers. In native timing, `step_ms` measures
submission time, and outstanding GPU work can be charged to the following
render call; **5.24 ms submission time is not 5.24 ms physics**. Use total
frame time for application FPS. Never compare those component timings
across protocols as if they measured the same quantity.

Validation completed so far:

- The complete 51-test MJMPM suite passed again on the retained checkpoint
  in 111.52 s, including the final point-broadcast, shared-factor and
  arithmetic-order changes. The earlier intermediate run passed in
  146.57 s. These suite durations are not performance benchmarks.
- Cholesky tests cover 1, 7, 24, 32 and 33 DOFs, three worlds, positive
  definite and zero matrices. Factors compare bitwise against the scalar
  implementation, including repeated captured execution and large-matrix
  fallback.
- Fused mobility tests cover articulated, rank-deficient independent,
  absent active rows and empty rows, comparing output bits and graph replay.
- The latest 990-frame physics acceptance run passed: 100% rice in the
  pot, 0% spilled, cup slip 3.38 mm / 2.34 degrees, full/empty finger load
  1.64/1.10 N and peak grain speed 1.88 m/s. Median physics time was
  31.71 ms; this includes test telemetry and excludes display.
- On identical frozen states at frames 180, 300 and 515, immutable-property
  caching and adjacent batching produced **zero different pixel
  components** versus the original display. Caching reduced frozen render
  time from about 9.5-9.7 ms to 7.1-7.4 ms. The extra batching gain is small
  and inconsistent; do not attribute the main improvement to it.
- **Unresolved strict numerical comparison:** full-trajectory same-input
  cooperative/scalar comparison examined 35,640 output arrays. The latest
  version reports seven scalar components exceeding
  `atol=1e-5, rtol=1e-5`; maximum absolute differences are
  `3.07e-5` in body twist, `1.79e-6` in contact impulse and `8.77e-7` in
  body impulse. Physical task acceptance passes, but the strict comparison
  command exits with failure. Do not describe it as passed or loosen
  existing acceptance thresholds to hide it.

Rejected experiments remain outside production: shared contact/impulse
caches, a one-DOF-per-lane layout, the contact-frame variant, and a scalar
precomputed-Delassus alternative did not provide a useful measured gain.
Globally grouping nonadjacent visual parts changed four pixels and was
replaced with adjacent-only grouping. No collision-block-floor reduction,
recovery-sweep reduction, reduced precision, or texture stripping is
retained.

Reproduction tools:

- `scripts/benchmark_mjmpm_sweeps.py frozen --cooperative --allow-roundoff
  --snapshots <directory> --output-dir <directory>` compares identical
  contact inputs; `verify` instead checks the complete trajectory.
- `scripts/benchmark_mjmpm_sweeps.py full --cooperative --output-dir
  <directory>` runs physics acceptance.
- `scripts/benchmark_mjmpm_sweeps.py display --cooperative --variant
  current --visible --timing native --output-dir <directory>` measures
  visible application timing. Use `--variant baseline` with the same
  timing options for the committed reference.
- `scripts/profile_mjmpm_frame.py --output <file.json>` separates captured
  solver phases; `scripts/verify_mjmpm_visual_uploads.py --output-dir
  <directory>` checks identical-state rendered images.

Raw evidence and rejected source snapshots are under
`C:/Users/Administrator/Documents/Blender/RicePour/target_25fps`.
Remaining gates are the unresolved strict comparison and repeated
matched-protocol performance pairs. The original 25 FPS target remains open.

### 2.10 Specializing articulated triangular substitutions

The main optimization in 2.9 was committed and pushed as `ae6df576`.
Subsequent work first tested restoring scalar point-response and tangent
arithmetic inside the cooperative contact sweep. This reduced the strict
same-input threshold violations from seven to two components over 35,640
arrays, with maximum twist difference `1.10e-5`. It still failed the
comparison and made representative frozen contact solves roughly 40-60%
slower than the retained precomputed-response path. Revert that experiment;
do not relax the comparison threshold. Its source, JSON and logs remain in
`target_25fps/ordered_response_verify` and `ordered_response_micro_repeat`.

The next candidate targets `K^-1 J^T` and `H^-1 b` in the articulated
mobility calculation. Generate a CUDA kernel specialized to the model's
fixed DOF count for systems with 1-32 DOFs. Unroll the two triangular
substitutions, retain intermediate column entries in a local fixed vector,
and write the completed column once. Each multiply/subtract and division
retains scalar loop order. The factorization, matrix, constraint Hessian,
integrator and coupling law remain unchanged. CPU and larger systems use
their original kernels. Kernels are generated during solver initialization,
before graph capture; repeated execution does not generate code.

A preliminary dynamic-length local-vector variant saved about 10% in the
24-DOF microbenchmark. The retained fixed-size specialization saves more:

| DOFs | operation | scalar ms | specialized ms |
|---|---|---:|---:|
| 7 | Jacobian RHS | 0.00655 | 0.00380 |
| 7 | in-place RHS | 0.00655 | 0.00380 |
| 24 | Jacobian RHS | 0.02686 | 0.00946 |
| 24 | in-place RHS | 0.02706 | 0.00944 |
| 32 | Jacobian RHS | 0.04285 | 0.01805 |
| 32 | in-place RHS | 0.04169 | 0.01710 |

These are CUDA-event medians of seven alternating captured measurements,
ten fresh substitutions per capture, three worlds and 144 RHS columns.
Both outputs compare bitwise with the scalar reference. This is a local
linear-algebra speedup, not a 2.8x frame speedup. The verification test adds
1-DOF coverage and repeated graph replay, and uses separate out-of-place
and in-place cases.

Two alternating 990-frame physics-acceptance pairs retain all physical
settings from 2.9. Both variants use the cooperative GS and display changes
from `ae6df576`; only triangular substitution changes:

| pair | scalar median ms | specialized median ms | scalar mean ms | specialized mean ms | scalar p90 ms | specialized p90 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32.008 | 31.723 | 32.224 | 32.167 | 37.336 | 37.680 |
| 2 | 31.936 | 31.628 | 32.655 | 32.034 | 38.388 | 37.386 |

The median frame metric improves about 0.9% in each pair. Mean and p90
benefits are less consistent; pair 1 p90 regresses. All four trajectories
pass: 100% rice in pot, 0% spilled; slip 3.32-3.43 mm, turn 3.06-4.37
degrees, full finger load 1.63-1.68 N and empty load about 1.10 N.
This small optimization does not establish that the 25 FPS display target
has been met. It also does not resolve the cooperative GS strict numerical
comparison recorded in 2.9.

Reproduce complete-task comparisons with
`scripts/benchmark_mjmpm_sweeps.py full --cooperative --column-solver
reference --output-dir <directory>` and `--column-solver local`.
The microbenchmark and raw outputs are archived under `target_25fps`
(`probe_triangular.py`, `triangular_reference_1/2`,
`triangular_local_1/2` and `mjmpm_triangular_specialized.log` in the user's
home directory). The retained specialization passes all **52 MJMPM tests**
in 129.26 s, including bitwise substitution comparisons and graph replay.
All pre-commit checks and `git diff --check` pass. Suite duration includes
compilation and is not a performance measurement.
