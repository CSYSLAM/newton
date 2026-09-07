# Coarse direction, step length and subspace diagnostics

2026-09-07, RTX 5090 D v2, Warp 1.17.0. This follow-up changes only
diagnostic code and its tests. It does not enable contact projection,
change solver defaults, alter the examples, or add damping/history fitting.
The user's existing staged solver work is preserved.

## Controlled experiment

`scripts/diagnose_mjvbd_v2_coarse_space.py` advances the original fast scene
and intercepts the first iteration of one subsequent substep. Every branch
shares inertial targets, kinematic poses, previous positions, material state
and contact candidates. Positions, cumulative displacements, truncation
factors and Chebyshev history/flags are restored between branches.

Ordinary 30/60 sweeps from the substep's initial state provide two references,
not exact solutions. A separate branch runs six fast sweeps for T-shirt
(one for twist), applies a candidate, then two fast sweeps with Chebyshev
disabled consistently across candidates. The zero-correction control is
therefore a **controlled 6+2 schedule**, not an unmodified fast8 performance
or accuracy claim. The candidate never uses reference positions to move
particles.

The assembly probe uses the real uncached fine elasticity/contact evaluators.
Reported forces are reconstructed as `D * local_correction`, where
`local_correction = D^-1 F` from the existing local solve. This introduces
inverse/reconstruction roundoff and is not an independent exact gradient.
The local Newton norm is a preconditioned residual proxy, not an energy.
DAT and the existing 5%-radius bound are retained for all trial steps.
No field timings include these host-side measurement/least-squares loops.

## Separating the limitations

One T-shirt state after 900 frames:

| Coupled coarse8 step | Error to ordinary30 after two post-sweeps (um) | Error to ordinary60 (um) |
| --- | ---: | ---: |
| No correction | 57.146 | 109.643 |
| alpha 0.1 | 54.313 | 106.763 |
| alpha 0.5 | 43.342 | 95.147 |
| alpha 1.0 | 31.976 | 81.124 |
| alpha 2.0 | 34.899 | 58.289 |

DAT changed these proposed steps by only about 0.1 um RMS and none hit
the radius clamp. The step-1.0 local Newton norm **before post-sweeps**
increased from 5.55 to 11.44 um despite the smaller reference error.
Increasing PCG from 8 to 32 at alpha 1.0 barely changed the result
(31.98 vs 32.09 um to ordinary30). A fixed larger alpha is not accepted.

Offline least-squares fits of the ordinary60 error gave a 29.23 um
remaining error for piecewise translations and 12.19 um for local affine
fields at cluster size 8. These measure representational capacity only:
they are not computed corrections, convergence guarantees or performance
results. Enlarging clusters to 32 worsened both bounds (39.87/22.92 um).

Twist after 900 frames behaves differently. Its uncoupled direction had a
smaller useful step range (0.25/0.5 helped modestly; 1.0 worsened error).
The coupled direction was rejected with source-buffer overflow and missing
reciprocal candidate flags. The experiment preserves that rejection rather
than using a truncated matrix. No new claim about fixing twist jitter is made.

## Operator-based Ritz enrichment prototype

Let `A` denote the frozen fine approximate Hessian, `D` its vertex-block
diagonal, `F` the current reconstructed force, and `d` the computed coarse
direction. The following bases use only the **current operator/state**:

- Three directions: `d`, `D^-1 A d`, `D^-1 F`.
- Five directions: additionally `(D^-1 A)^2 d` and `(D^-1 A) D^-1 F`.

Solve `B^T A B c = B^T F` and propose `B c`. The CPU prototype normalizes
columns and rejects dependent/nonpositive reduced systems. It contains no
iterate history or target trajectory. A singleton Galerkin projection exposes
the fine matrix for this diagnostic only; production cluster-size validation
is unchanged. Matrix products run on GPU. This is not yet a production
matrix-free implementation: it explicitly builds a fine sparse operator,
copies data to the host and performs the tiny reduced solve in NumPy.

Representative independent 900-frame sample, all rows sharing its frozen state:

| Candidate | Error30 (um) | Error60 (um) | Post local Newton norm (um) | Post reconstructed force RMS |
| --- | ---: | ---: | ---: | ---: |
| No correction | 56.455 | 110.264 | 3.517 | 4.193 |
| Coarse8, alpha 0.1 | 53.628 | 107.425 | 3.507 | 4.025 |
| Coarse8, alpha 1.0 | 31.564 | 82.371 | 9.068 | 5.368 |
| Three-direction Ritz | 25.547 | 72.181 | 6.914 | 4.322 |
| Three-direction Ritz, half step | 37.725 | 90.743 | 3.605 | 3.344 |
| Five-direction Ritz | 25.634 | 66.508 | 9.681 | 4.937 |
| Five-direction Ritz, half step | 34.794 | 87.398 | 3.348 | 3.282 |

At frame 390, three/five half-step candidates also reduced the sampled
post local norm (4.107 -> 3.499/3.302 um) and reference errors. The half-step
rows are an exploratory step scan, **not a chosen universal relaxation**.
Higher degree does not uniformly improve the nonlinear residual.

### Why a cheap linear residual guard is not enough

For `l = D^-1 F` and `m = D^-1 A d`, preserving the frozen norm requires
`0 <= alpha <= 2*(l dot m)/(m dot m)` when the numerator is positive.
The implementation tests this bound independently, including nonfinite and
nonpositive cases. However, the bound often accepts alpha 1.0 while the
**reassembled nonlinear** local norm grows (see full-step rows above).
Thus this guard is rejected as sufficient production acceptance. It only
bounds a frozen model with fixed `D`, not changing contact response or the
new vertex diagonal. A real acceptance check or a tighter model is still
needed; no rollout uses this heuristic as a safety guarantee.

## Cost gate

Frozen-state CUDA graph microbenchmarks exclude construction-time allocations,
JIT and host measurement/least-squares work. Each stage is warmed, captured,
then measured over five groups of 50 replays, reporting their median.
All graph work is single-process/sequential and the CUDA guard stays active.

First measured state:

| Stage | Time (ms) |
| --- | ---: |
| One fast batched sweep, with input reset | 0.1149 |
| Coupled coarse8 pass, with input reset and DAT | 1.0555 |
| Fine operator assembly only | 0.3196 |
| One fine sparse matrix product | 0.0344 |

The existing coupled coarse pass alone costs approximately nine of these
fast sweeps. The enriched prototype builds an additional fine operator and
performs several matrix products, so positional improvement is **not** enough
to justify integration. These stage timings are not complete-frame timings
and should not be extrapolated to every scene or GPU.

An independent repeat with a finer breakdown measured: fast sweep 0.2278 ms,
coupled coarse pass 2.1992 ms, coarse assembly 0.5508 ms, and persistent PCG
alone 1.5211 ms. PCG accounts for about **69%** of that coarse pass; the
pass/sweep ratio is again about 9.7. Absolute stage times differed materially
between runs on this shared desktop. This is bottleneck evidence within the
same run, not evidence of a speedup or an isolated cause of timing variation.

## Status and next engineering gate

- Keep coupled coarse and all new enrichment candidates out of the default
  fast path. No complete folding/twisting trajectory has been accepted for
  the enrichment prototype; frozen-state improvement is not proof of settling.
- The measured bottleneck is persistent PCG. Compare
  contact-row compaction or a multi-block coarse solve at identical RHS,
  operator and iteration count before adding more basis directions.
- Share operator construction if the enriched route remains promising; do
  not ship the diagnostic's duplicate coarse/fine assembly.
- Recheck the actual nonlinear residual, geometry and complete histories
  before promotion. Source-contact overflow in twist is an independent
  limitation and must not be bypassed.

Validation: 81 focused tests passed (77 existing plus four probe tests),
including CPU/CUDA matrix products, dense Ritz equivalence, basis rescaling,
nonpositive/dependent-space rejection and the analytic residual bound.
This is not the full Newton suite.

```bash
uv run scripts/diagnose_mjvbd_v2_coarse_space.py --scene tshirt --enrichment --profile-enrichment
uv run scripts/diagnose_mjvbd_v2_coarse_space.py --scene tshirt --enrichment --warmup-frames 390
uv run scripts/diagnose_mjvbd_v2_coarse_space.py --scene twist
uv run -m unittest newton.tests.test_mjvbd_v2_coarse_space_probe
```

Session logs: `/tmp/newton-coarse-space-*`, `/tmp/newton-coarse-krylov-*`.
The logs are temporary; the tables above retain the relevant observations.

Follow-up implementation and measurements: `COARSE_PCG_SPLIT.md` compares
contact-row packing and multi-block sparse products. The latter reduces
the measured large-system PCG cost without changing its recurrence, but
does not establish improved nonlinear convergence or settled geometry.
`--profile-enrichment` explicitly keeps the original persistent solver
so that its historical per-kernel breakdown remains comparable.
