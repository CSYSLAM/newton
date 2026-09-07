# Large surface Galerkin PCG scheduling

2026-09-07, RTX 5090 D v2, Warp 1.17.0. This is an engineering
optimization of the existing coarse solve, not a new convergence algorithm.
No example, sweep budget, material, damping or multilevel enable default
changes. Experimental contact projection remains disabled by default.

## Implementation and boundaries

The original persistent PCG executes sparse products and vector reductions
inside one 256-thread block. The split implementation uses many 64-thread
blocks for the sparse product and retains the original 256-thread block,
lane-stride traversal and summation order for each reduction/update.
Matrix, RHS, preconditioner, iteration count, residual checks and rejection
flags are unchanged. Fixed scalar workspace is allocated before capture;
there are no convergence readbacks or unconditional `wp.capture_if` calls.

Automatic selection requires CUDA, at least 512 clusters, at least 8192
stored base matrix blocks, and at least four coarse iterations. Small,
short-iteration and CPU systems retain the original implementation. The
graph operator and mixed/tetrahedral rigid-basis path are not replaced.
These thresholds are conservative results from one GPU, not a universal
optimal crossover or a claim about batched multi-world scaling.

An optional internal `ContactProjection.compact()` copies the linked rows
to CSR in their existing order. Reset invalidates its ready flag. The
first call reserves workspace outside CUDA capture. It is **not called
automatically**: its extra packing work outweighed its benefit with split
PCG on the measured real system.

## Controlled timing

Freeze one matrix, RHS and contact-row ordering. Warm and capture each
variant, then measure eight alternating-order groups of 50 CUDA graph
replays with CUDA events. Construction, JIT and host diagnostics are excluded.
Compare all output arrays before timing. All tested outputs are bitwise
equal, including residual traces, guard flags and work vectors. This is
evidence on the tested backend, not a guarantee of cross-device bit identity.

Real T-shirt state: 996 clusters, 4874 unique contact cluster pairs,
eight PCG iterations:

| Schedule | PCG plus required packing (ms) |
| --- | ---: |
| Persistent, linked rows | 0.589809 |
| Persistent, packed rows including packing | 0.522932 |
| Split, linked rows | 0.210974 |
| Split, packed rows including packing | 0.231651 |

Split linked rows reduce this PCG stage by 64.2%. This is not a claim of
64.2% complete-frame acceleration. Packing alone costs 0.047945 ms.

Synthetic eight-iteration scale scan (periodic sparse SPD blocks):

| Clusters | Base blocks | Contact pairs | Persistent (ms) | Split (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 384 | 0 | 0.0199 | 0.0471 |
| 128 | 384 | 1024 | 0.0774 | 0.1017 |
| 256 | 768 | 0 | 0.0225 | 0.0492 |
| 256 | 4352 | 2048 | 0.1498 | 0.1340 |
| 512 | 8704 | 0 | 0.1422 | 0.0828 |
| 512 | 8704 | 4096 | 0.3003 | 0.1284 |
| 1024 | 17408 | 0 | 0.2888 | 0.0938 |
| 1024 | 17408 | 8192 | 0.6400 | 0.1629 |
| 2048 | 34816 | 0 | 0.5793 | 0.1247 |
| 2048 | 34816 | 16384 | 1.3103 | 0.1911 |

At four iterations, the eligible rows also improved: split/persistent
ratios range from 0.165 to 0.608. Small matrices can regress materially,
which is why this is not an unconditional replacement.

## Complete-history checks and unresolved quality

The coupled diagnostic uses the same eight fast sweeps, coarse settings,
contact policy and 900-frame history length, followed by 120 timed frames.
The example's final checks pass for both implementations. First runs:

| Schedule | End frame (ms) | RMS speed | Second-difference RMS | Intersections at 900 |
| --- | ---: | ---: | ---: | ---: |
| Persistent | 38.496 | 0.00482045 | 0.0000326832 | 66 |
| Auto split | 24.546 | 0.00425446 | 0.0000541867 | 219 |
| Persistent repeat | 41.727 | 0.00295503 | 0.0000445524 | 160 |
| Auto split repeat | 24.053 | 0.00351456 | 0.0000345433 | 226 |

These are independently assembled GPU histories, not replay of identical
contact matrices. Despite identical frozen PCG outputs, the histories
differ substantially. They do **not** establish unchanged geometric
quality, improved settling, or a statistically stable full-frame speedup.
The persistent repeat demonstrates sizable baseline run-to-run variation,
but both split runs have more final intersections than either persistent
run. Two samples per schedule cannot dismiss that difference as harmless
nondeterminism or establish a causal regression; it remains a quality gate.
In particular, the larger intersection count must not be hidden by reporting
only the lower speed or frame time. Coupled coarse correction remains
experimental. This work does not resolve the original jitter/penetration
problem or justify enabling it in the default fast demos.

An additional auto-scheduled coupled twist run completed 900 frames plus
120 timed frames and passed the diagnostic's bounds/finite-state checks.
At frame 900 it reported source flags 24 and coarse status 32, with a zeroed
coarse residual trace: invalid contact projection was still rejected. Its
111 intersection pairs and nonzero final RMS speed (0.005368) are not a
quality pass. No twist speedup is inferred from this single run.

## Regression coverage and reproduction

86 focused tests passed, not the complete Newton suite. Coverage includes
CPU fallback, CUDA sizes crossing block boundaries, projected linked/packed
contacts, zero RHS, nonfinite RHS, negative curvature, overflow rejection,
automatic large-surface selection, correction equivalence, CUDA graph
replay, compaction/reset, and the existing multilevel/contact/Jacobi/cache
regressions. The compaction regression failed on the missing method before
implementation. Existing user-staged changes were retained; no commit or
push is part of this experiment. CUDA guard stayed active throughout.

```bash
uv run scripts/benchmark_mjvbd_v2_coarse_rows.py
uv run scripts/benchmark_mjvbd_v2_coarse_rows.py --iterations 4
uv run scripts/diagnose_mjvbd_v2_coarse_space.py --scene tshirt --profile-contact-rows
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode coupled --scene tshirt --coarse-pcg persistent
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode coupled --scene tshirt
uv run -m unittest newton.tests.test_mjvbd_v2_coarse_pcg_split newton.tests.test_mjvbd_v2_contact_rows
```

Temporary logs: `/tmp/newton-contact-row-split-profile.log`,
`/tmp/newton-coarse-split-scale*.log`,
`/tmp/newton-coarse-split-full-*.log`, and
`/tmp/newton-coarse-split-regressions-final.log`.

The next algorithmic gate remains a current-state coarse/enriched direction
that reduces the reassembled nonlinear residual and complete-history
jitter/penetration, within the saved cost budget. Frozen linear residual
improvement alone is insufficient; see `COARSE_SPACE_DIAGNOSTICS.md`.
