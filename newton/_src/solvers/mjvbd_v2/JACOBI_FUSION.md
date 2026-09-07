# Jacobi update audit and launch fusion — 2026-09-07

## Scope

This change optimizes the soft backend's existing CUDA surface-batched Jacobi
implementation. It does not change example settings, color schedules, iteration
counts, relaxation factors, contacts, constitutive laws, or DAT. It adds no
history fitting, adaptive sweeps, new public options, or CPU synchronization.
The ordinary CPU path and the complete AVBD backend retain their existing path.

The earlier uncommitted `particle_jacobi_polish_iterations` experiment is
separate from this work and remains disabled by default.

## Update dependency audit

The original per-batch sequence was:

1. Accumulate contact forces/Hessians for the selected batch.
2. Read frozen positions and solve elasticity into a cleared scratch array.
3. Add `relaxation * scratch[p]` to cumulative displacement.
4. Run DAT and publish positions for the next batch.

The fused specialization combines steps 2 and 3. The elasticity kernel reads
positions, not neighboring cumulative displacements. Batch indices contain
each selected particle exactly once, so writing each particle's displacement
does not introduce cross-thread dependencies. Position publication and the
kernel boundary before DAT are unchanged. The next batch still sees the
updated positions.

Important boundary cases:

- Jacobi relaxation applies to contact rows as well as contact-free rows.
  Ordinary surface relaxation retains its contact-free-only behavior.
- Inactive/zero-mass rows preserve their prior cumulative displacement, just
  as adding a zero scratch correction did; the ordinary kernel still clears
  these rows. Singular local systems similarly add no correction.
- DAT uses displacement relative to the collision snapshot, not a fresh
  incremental displacement. The fused kernel adds to that existing quantity.
- Contact accumulation, Chebyshev exclusions, and DAT are not fused together.
  Their dependencies remain intact.

This removes one scratch clear per sweep, one apply launch per batch, and a
12-byte-per-particle scratch allocation. For 8 sweeps and 2 batches, that is
24 fewer launches per substep. It does **not** make batched Jacobi equivalent
to ordinary Gauss–Seidel at a fixed iteration count, nor establish the cause
of the previously observed late jitter.

## Regression checks

The new integration test failed before the implementation (missing fused path
and retained scratch allocation). After implementation, 29 targeted tests
passed, covering:

- CPU and CUDA kernel-level comparison against the original solve/apply pair;
- nonzero anchor displacements, pinned/zero-mass rows, singular Hessians,
  and contact weighting at relaxation 0.5 and 1.0;
- multistep contact-free trajectories;
- per-update self-contact comparison using identical positions and forces;
- CUDA graph fused/unfused replay, existing Chebyshev graph checks, CPU
  fallback, surface caches, and cached/uncached DAT tests.

Independent self-contact trajectories can build different candidate orderings;
their difference is not used as a bitwise equivalence test. Kernel comparisons
use explicit floating-point tolerances, not a universal bitwise guarantee.

```bash
uv run -m unittest newton.tests.test_mjvbd_v2_jacobi_fusion newton.tests.test_mjvbd_v2_batched_jacobi newton.tests.test_mjvbd_v2_particle_chebyshev newton.tests.test_mjvbd_v2_surface_cache newton.tests.test_mjvbd_v2_truncation_cache newton.tests.test_mjvbd_v2_surface_relaxation
uv run scripts/benchmark_mjvbd_v2_jacobi_fusion.py
```

## Generic-model performance

RTX 5090 D v2, Warp 1.17.0. Each A/B graph resets physical inputs, executes
one complete 8-sweep solver substep, and uses the same solver/candidate policy.
The unfused graph reconstructs the original launch sequence. Its scratch
owner is retained throughout replay. Python interception is capture-only.
Each reported median contains 8 alternating-order batches of 100 replays.
Timings are synchronized wall-clock time per substep, including graph
submission; they are not end-to-end demo frame timings.

| Particles | Model | Unfused ms | Fused ms | Throughput gain |
| ---: | --- | ---: | ---: | ---: |
| 81 | Pinned surface, no contact | 0.2680 | 0.2357 | 13.7% |
| 1089 | Pinned surface, no contact | 0.2554 | 0.2247 | 13.7% |
| 4225 | Pinned surface, no contact | 0.2832 | 0.2453 | 15.4% |
| 2178 | Two layers, active self-contact | 1.8263 | 1.7843 | 2.35% |

A second complete invocation with explicit graph scratch ownership produced:

| Particles | Unfused ms | Fused ms | Throughput gain |
| ---: | ---: | ---: | ---: |
| 81 | 0.2778 | 0.2511 | 10.6% |
| 1089 | 0.2774 | 0.2334 | 18.8% |
| 4225 | 0.2841 | 0.2497 | 13.8% |
| 2178 | 1.8205 | 1.8019 | 1.04% |

The existing CUDA guard was left running; device clocks and competing load
were not locked. Absolute timings varied substantially between exploratory
runs, so compare paired A/B samples rather than mixing timings across runs.
This is a modest gain in contact-dominated work, not a claim of 15% faster
robot cloth demos; the small contact-case gain is sensitive to timing noise.
The remaining convergence problem is still open.
