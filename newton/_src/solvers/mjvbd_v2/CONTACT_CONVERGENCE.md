# Contact convergence experiments — 2026-09-07

## Acceptance criteria

Improve the generic accelerated solver, not robot trajectories or material
parameters. Lower end-stage motion alone is insufficient: compare convergence
on an identical frozen substep, strict geometric intersections through the
full trajectory, finite states, and execution time. Keep existing accelerated
defaults, collision checks, and the CUDA guard. No sleeping particles,
trajectory fitting, added physical damping, or forced 30-sweep default.

## Rejected: simultaneous-contact diagonal majorization

A fixed linearized contact has Hessian `b b^T tensor K`, with `K` positive
semidefinite. Updating several participating vertices simultaneously with
only their original diagonal blocks can oscillate. For the n vertices in
one batch, Cauchy-Schwarz gives a local bound
`n diag(b_i^2) - b b^T >= 0`. The experiment multiplied each participating
vertex's contact Hessian by n, without changing its contact force. It changed
only batched-Jacobi EE/VT self-contact assembly, not ordinary GS, coarse
correction, materials, friction, iteration counts, or the demos.

An isolated four-vertex contact with diagonal inertia 100 and normal penalty
1000 reproduced the old two-cycle. The bounded update converged to the
analytic equilibrium within 1e-7 m after 256 iterations; the original remained
more than 1e-3 m away. CPU/CUDA EE/VT tests verified unchanged forces and
single-vertex colors. Together with existing fusion, batched Jacobi,
Chebyshev, surface/truncation cache, relaxation and contact tests, 43 tests
passed. This was **not sufficient for full-scene acceptance**.

### Complete histories, repeat with geometry checks

RTX 5090 D v2, Warp 1.17.0. Each variant starts from the original scene and
runs 900 frames. Motion averages use the last 90 frames. Timing covers a
further 120 frames without CPU geometry checks. Rows are independent runs;
GPU atomic ordering and nonlinear contact history are not deterministic.
Intersections count strict nonincident edge/triangle-interior pairs, not
distinct holes, penetration depth, coplanar overlap, or touches.

| Scene / policy | RMS speed (m/s) | Second difference (um) | Kinetic energy | Crossings frame 390 / 900 | End frame (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| T-shirt / original fast | 0.0033232 | 37.01 | 3.6542e-8 | 285 / 320 | 16.743 |
| T-shirt / majorized | 0.0022892 | 33.94 | 1.7923e-8 | 464 / 557 | 17.352 |
| Twist / original fast | 0.0112423 | 51.45 | 2.1234e-5 | 61 / 276 | 16.541 |
| Twist / majorized | 0.0103810 | 32.06 | 1.6821e-5 | 42 / 230 | 15.810 |

An earlier pair of independent runs showed much larger jitter reductions
(T-shirt 51.13 -> 25.68 um, twist 68.35 -> 9.83 um). The repeat above shows
why those single-run percentages must not be advertised as guaranteed gains.
The T-shirt crossing increase fails the quality criterion.

### Identical frozen substep after 900 original T-shirt frames

| Policy / sweeps | Extra ordinary sweep displacement (um) | RMS distance to ordinary 30 (um) |
| --- | ---: | ---: |
| Original fast / 8 | 4.108 | 56.31 |
| Majorized fast / 8 | 3.647 | 64.05 |
| Ordinary / 30 | 2.443 | 0 |
| Ordinary / 60 | 2.335 | 54.75 |

The extra-sweep displacement is a fixed-point defect, not a force residual
or proof of convergence. The contact bound reduces oscillatory motion but
slows remaining low-frequency progress here. Even ordinary 30 and 60 are
not the same state, so neither is treated as exact ground truth.

### Same-input performance control

Alternating CUDA graph replay on generic cloth grids, eight trials per
variant, 100 reset-input substeps per trial. Both policies retain the
existing fused Jacobi kernel.

| Particles / layers | Original substep (ms) | Majorized substep (ms) |
| --- | ---: | ---: |
| 81 / 1, no self-contact | 0.19345 | 0.19310 |
| 1089 / 1, no self-contact | 0.18669 | 0.18566 |
| 4225 / 1, no self-contact | 0.20131 | 0.20067 |
| 2178 / 2, active self-contact | 1.46038 | 1.46022 |

No measurable kernel-cost regression in this control does not cancel the
full-history timing and quality results. This candidate is rejected as a
default fix. Production source was restored to the pre-experiment index;
the previously staged fusion/polish work was preserved.

`contact_majorization_rejected.patch` preserves the experiment, tests and
diagnostic extensions against that staged baseline. It is an **unapplied
research artifact**, not recommended production code. Check applicability
with `git apply --check` in a disposable worktree before reproducing. Its
diagnostic `--mode majorized` opts into the candidate; `--mode fast` retains
the original policy. It must not be silently promoted to a default.

## Also rejected: filling the zero-slip friction Hessian

The regularized friction helper returns a zero Hessian exactly at zero slip,
although its positive-epsilon limit is nonzero. A candidate filling this
limit passed local derivative tests on CPU and CUDA, but, combined with the
contact-bound candidate, the full T-shirt trajectory became nonfinite at
frame 809. All four friction-helper edits were reverted. The local derivative
observation does not establish that this candidate is safe in the full
nonlinear solver, or isolate the cause of that failure.

## Remaining work

The end-stage convergence issue is **not solved** by these experiments.
Simply increasing the contact diagonal reduces overstepping but does not
resolve coupled slow modes. Further candidates need to preserve coupled
contact response and pass full-history quality checks, not only local
derivative tests or reduced motion metrics. A separate local-block precision
experiment checks whether float32 determinant/inverse error is material.

## Rejected as a settling fix: higher-precision local block inversion

The second experiment kept float32 state, force and Hessian assembly, but
solved the final 3x3 block in float64. A second variant used float64 only
when `abs(det(H)) < 1e-6 * max(abs(diag(H)))^3` or the float32 determinant
was nonfinite. This scale test is a heuristic, not an exact condition-number
estimate. Neither variant changed sweeps, contact detection, damping or
trajectory controls. Both remained opt-in throughout the experiment.

A CPU/CUDA regression used rotated positive-definite blocks with eigenvalues
near 1, 1000 and 1e7, plus diagonal inertia. The reference solves the actually
assembled float32 matrix in NumPy float64, not a different pre-rounding
matrix. Original float32 inversion exceeded 1e-3 relative error; both higher
precision variants stayed below 1e-6. This demonstrates a local numerical
weakness, **not that it is the cause of end-stage scene motion**.

| Scene / policy | RMS speed (m/s) | Second difference (um) | Crossings frame 390 / 900 | End frame (ms) |
| --- | ---: | ---: | ---: | ---: |
| T-shirt / all local solves float64 | 0.0028870 | 29.39 | 172 / 242 | 16.351 |
| T-shirt / adaptive float64 | 0.0043946 | 39.55 | 227 / 223 | 31.719 |
| Twist / all local solves float64 | 0.0118692 | 44.89 | 37 / 179 | 15.601 |
| Twist / adaptive float64 | 0.0095917 | 40.87 | 53 / 311 | 17.459 |

Compare against the independent original-fast histories above, with the
same nondeterminism caveats. No robust scene-level settling improvement was
established. In particular, the adaptive T-shirt run increased motion and
had a large timing regression, while adaptive twist increased final crossing
pairs. No cause for the large T-shirt timing change was isolated; it must
not be dismissed as noise or attributed to the local kernel without profiling.

Same-input graph replay exposed the cost of unconditional float64 even
where individual scene timings looked favorable:

| Particles / layers | Single / double substep (ms) | Single / adaptive substep (ms) |
| --- | ---: | ---: |
| 81 / 1 | 0.19494 / 0.21240 | 0.19365 / 0.19625 |
| 1089 / 1 | 0.18778 / 0.20894 | 0.18638 / 0.18846 |
| 4225 / 1 | 0.20142 / 0.27326 | 0.20098 / 0.20287 |
| 2178 / 2, active contact | 1.45910 / 1.49970 | 1.45592 / 1.45787 |

All production and diagnostic source changes for this experiment were also
reverted to the pre-experiment index. The one regression test and diagnostic
are preserved in the **unapplied** `local_precision_rejected.patch`; they
are not presented as an accepted solver implementation. Both rejected
patches apply separately against the same staged baseline, not sequentially.
The original accelerated defaults and staged Jacobi fusion remain intact.
After restoring the production source, all 41 existing tests in the seven
fusion/Jacobi/Chebyshev/cache/relaxation/contact regression modules passed
(`newton-convergence-restored-regression.log`). Both experiment patches
passed `git apply --check` separately. The CUDA guard remained active.

### Local logs

Raw logs remain under `/tmp` on the test machine, not versioned artifacts:

- `newton-original-{tshirt,twist}-verified.log`
- `newton-majorized-{tshirt,twist}-verified.log`
- `newton-majorization-frozen.log`, `newton-majorization-benchmark.log`
- `newton-double-local-{tshirt,twist}.log`
- `newton-adaptive-local-{tshirt,twist}.log`
- `newton-local-precision-benchmark.log`, `newton-adaptive-precision-benchmark.log`
- `newton-local-precision-test.log`, `newton-majorization-regression.log`

These results reject the tested diagonal and precision changes as sufficient
solutions. They do not establish a unique cause for the remaining low-frequency
error. The next algorithmic investigation should measure the coupled contact
operator and changing active sets on identical substeps; any block-coupled
correction must be evaluated against the original fast solver at comparable
cost, not accepted because a static four-vertex test converges.
