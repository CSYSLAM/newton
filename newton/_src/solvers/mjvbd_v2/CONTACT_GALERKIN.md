# Contact coupling and convergence investigation

Date: 2026-09-07. Baseline: the user's staged fast/Jacobi fusion work on
FAST_MJVBDV2. No example defaults, material parameters, sweep budgets, or
recorded trajectories are changed by this investigation.

## Experimental Galerkin contact projection (disabled by default)

The translation-only surface Galerkin operator previously retained contact
vertex diagonal blocks without projecting the corresponding cross-vertex
blocks. For a frozen contact stencil with weights `b_i` and spatial Hessian
`K`, its projected block is `alpha_c * alpha_d * K`, where
`alpha_c = sum(b_i for i in cluster c)`. The existing diagonal already
contains `sum(b_i**2) * K`; replace that contribution rather than double
counting it. A self-contact entirely within one translating cluster must
have zero stiffness under that translation. Pinned particles are omitted
from the free coarse unknowns; external contacts retain their stiffness.

The experimental implementation reuses the fine contact evaluators, including
normal stiffness, friction, damping and edge/face barycentric weights.
Repeated cluster pairs are accumulated in a bounded device hash table.
PCG scans unique coarse neighbors instead of every contact record.
Nonfinite stencils, capacity/hash overflow, source contact-buffer overflow,
and missing reciprocal active EE records reject the coarse correction.
No unconditional `wp.capture_if`, host contact-count readback, or allocation
based on a device contact count is introduced.

This is an internal diagnostic opt-in (`contact_projection_enabled`), not a
new recommended production setting. It remains **false by default**.
CPU/CUDA unit tests cover the dense projected operator, pinned vertices,
same-cluster translation, contact extraction, overflow rejection, duplicate
reduction and CUDA graph replay. The translation regression failed before
the implementation.

### Frozen T-shirt substep

All candidates start from the same substep after 900 baseline frames. The
ordinary 30-sweep result is a reference, not a demonstrated exact solution.
Errors below are micrometers; the smoothed error is a diagnostic low-frequency
proxy, not a rigorous spectral decomposition.

| Candidate | RMS vs 30 sweeps | Smoothed RMS vs 30 sweeps |
| --- | ---: | ---: |
| Fast 8 sweeps | 52.709 | 50.340 |
| Fast 8 + uncoupled coarse PCG 8 | 52.842 | 50.469 |
| Fast 8 + coupled coarse PCG 8 | 50.926 | 48.538 |
| Fast 8 + coupled coarse PCG 32 | 50.810 | 48.421 |

Increasing coarse PCG from 8 to 32 reduced its final squared residual from
7376.595 to 0.002059, but improved the fine smoothed error only about 0.24%.
Simply solving this coarse space more accurately is therefore insufficient.
Next isolate coarse step length/acceptance and coarse-space representational
error using frozen states before attempting another full-history change.

### Full-history acceptance before topology-filter experiments

900 frames, last-90-frame statistics, then 120 timed frames, RTX 5090 D v2.
Timing is wall time on a desktop shared with the CUDA guard and other apps;
these are single runs, not statistically established speedups.

| T-shirt configuration | RMS speed (m/s) | Second difference (um) | Crossings at 900 | End frame (ms) |
| --- | ---: | ---: | ---: | ---: |
| Uncoupled coarse | 0.002953 | 25.69 | 102 | 21.06 |
| Coupled coarse, raw contact lists | 0.004095 | 41.65 | 161 | 62.87 |
| Coupled coarse, hashed cluster pairs | 0.002582 | 30.98 | 158 | 31.33 |

Hashing reduces experimental overhead, but the corrected coarse operator has
**not** passed the combined quality/performance target. It is not enabled in
the demos and is not claimed to solve end-stage jitter. Crossings count strict
edge/triangle intersection pairs, not unique holes or penetration depth.

## Built-in EE topology-filter reciprocity

An independent 8x8 cloth reproduction found 644 directed-only exclusion
relationships at filter radius 2. The legacy n-ring policy subtracts edges
incident through all four bending-stencil slots, including opposite vertices.
Consequently A can filter B while B still queries A; fine EE accumulation
expects the two directed contact halves.

An initial endpoint-only subtraction makes graph-distance exclusions
symmetric, but removes additional candidates. This was rejected as a default
change: a single twist run had more intersection pairs (278 to 501 at frame
900), despite a lower second-difference statistic. Those measurements alone
do not establish repeatability or total error.

The conservative alternative retains only **mutual legacy exclusions**.
Thus no formerly allowed contact candidate is newly excluded; only missing
reverse candidates are restored. It is a construction-time operation in both
VBD backends, with no extra GPU launches or sweeps. User-supplied filtering
maps are merged afterward and are not silently rewritten.

The regression independently reconstructs the legacy policy from mesh
connectivity, verifies candidate preservation and reciprocity at radii 2/3,
and runs against both backends. With the mutual-filter return replaced by
the legacy return, all four subtests fail; the new policy passes.

**Important control:** the T-shirt demo uses filter radius 1, so this path
does not execute there. Its observed run-to-run changes (for example 42.3 to
27.8 um second difference) are NOT evidence of this filter fix helping that
demo. Long histories with atomic accumulation need repeated measurements.

### Conservative-filter full-history repeats

Twist, unchanged 3 fast sweeps, 900 frames:

| Policy / run | RMS speed (m/s) | Second difference (um) | Crossings 390 / 900 | End frame (ms) |
| --- | ---: | ---: | ---: | ---: |
| Legacy / 1 | 0.008144 | 34.68 | 43 / 278 | 38.60 |
| Mutual-only / 1 | 0.008440 | 33.65 | 26 / 176 | 19.54 |
| Legacy / 2 | 0.010919 | 35.89 | 47 / 238 | 20.93 |
| Mutual-only / 2 | 0.009971 | 51.85 | 26 / 156 | 18.95 |

Both mutual-only runs have fewer crossings, but the second has a worse
second-difference statistic. This is retained as a contact-reciprocity
correctness fix, **not** an accepted general jitter/convergence solution.
The first legacy timing is an outlier; do not claim a 2x speedup. The
construction-time postpass itself adds no per-frame work, but restored
contact candidates can change runtime collision workload.

Final focused regression run: **77 tests passed**, covering both CPU and
CUDA where applicable. Ruff check, Ruff format check, and `git diff --check`
passed. These are focused solver tests, not the entire Newton test suite.

## Reproduction

```bash
uv run -m unittest newton.tests.test_mjvbd_v2_contact_projection newton.tests.test_mjvbd_v2_edge_filter_reciprocity
uv run scripts/diagnose_mjvbd_v2_convergence.py --help
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode fast --scene twist
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode fast --scene twist --legacy-edge-filter
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode uncoupled --scene tshirt
uv run scripts/diagnose_mjvbd_v2_contact_coupling.py --mode coupled --scene tshirt
uv run scripts/benchmark_mjvbd_v2_jacobi_fusion.py --help
```

Raw diagnostics for this session are under `/tmp/newton-contact-*` and
`/tmp/newton-edge-*`; these are temporary, not committed artifacts.
