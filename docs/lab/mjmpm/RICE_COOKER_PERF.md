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
