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
