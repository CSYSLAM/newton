# Inflatable cavity color coupling experiment

This is an opt-in solver experiment, not a passed physical acceptance result.
The original grasp motion, finger controller, material, mesh, five substeps,
12 sweeps and 500 kPa absolute-pressure limit are preserved. No volume feedback
is added to the robot controller. The scene's minimum-volume acceptance remains
0.85 and **still fails**.

## Test the scene

```bash
uv run --offline --no-sync -m newton.examples.mjvbdv2.example_mjvbd_v2_inflatable_bag_grasp --pneumatic-color-coupling
```

Omit `--pneumatic-color-coupling` to use ordinary VBD. The exact local pneumatic
Hessian correction applies to both paths. The experiment requires CUDA and
activates color pressure coupling, surface caching and existing contact-aware
Chebyshev acceleration (spectral radius 0.95, two warmup and two polish sweeps).

Full diagnostic acceptance, recording failures and returning nonzero on failure:

```bash
uv run --offline --no-sync docs/lab/mjvbd_inflatable_performance_2026-09-14/validate.py --pneumatic-color-coupling --no-render --output coupled_color
```

## Algorithm

The signed closed-mesh volume and pressure force were verified independently.
The local pressure Hessian previously accumulated outer products separately per
face. It now groups incident faces by cavity and uses
`H_i = sum_c kappa_c * outer(sum_f g_if, sum_f g_if)`, including cross-face terms.
For closed surfaces volume is affine in one vertex, so this is its exact local
pressure-energy Hessian block away from pressure-law kinks.

The optional coupled path evaluates the existing elasticity, inertia and contact
blocks `A_i` without pneumatic curvature. It computes `d_i = inverse(A_i) * f_i`
and reduces `s = sum dot(g_i, d_i)`, `t = sum dot(g_i, inverse(A_i) * g_i)` over
one color. Let `p` be current gauge pressure and `p_raw` the unclamped pressure
at the current volume. For target-volume pressure with bulk damping:

```text
kappa = pressure_scale * volume_stiffness + bulk_damping / dt
p_new = clamp((p_raw - kappa * (s - p*t)) / (1 + kappa*t),
              -ambient_pressure, max_absolute_pressure - ambient_pressure)
d_i += inverse(A_i) * g_i * (p_new - p)
```

This solves the rank-one pressure coupling with both pressure bounds, including
steps that enter or leave a capped branch. It is exact for the affine volume of
a triangle-independent color and the existing local quadratic approximation of
other forces; it is **not** a global nonlinear Newton solve. Contact generation,
contact penalties and displacement truncation remain in place. The experimental
path currently accepts one target-volume surface cavity on the CUDA tile path;
unsupported configurations raise an error. Ordinary VBD retains multi-cavity
support, including shared vertices in Hessian assembly.

Separating pneumatic curvature from contact blocks also lets the existing
Chebyshev contact guard distinguish pneumatic stiffness from actual contacts.
Coupling alone made almost no improvement; the combination improves convergence.
The cached surface variant exports the already assembled local Hessian, and an
empty-edge guard avoids reading absent bending adjacency on membrane-only meshes.

## Measurements

Baseline: `e9bee362f8e380e6367543b9669cbfa690a1720c`, RTX 5090 D v2.
The following comparisons replay identical recorded original joint start/end
positions. All minima use the first 200 frames; mean step times exclude the first
30 frames and rendering. These are sequential diagnostic measurements, not a
repeated end-to-end FPS benchmark.

| Variant | Sweeps | Minimum V/V0 | Mean step ms |
| --- | ---: | ---: | ---: |
| Original | 12 | 0.79186 | 15.69 |
| Original | 16 | 0.80443 | 21.77 |
| Original | 48 | 0.82355 | 52.17 |
| Original | 96 | 0.83756 | 100.88 |
| Correct local Hessian only | 12 | 0.79160 | See results JSON |
| Color coupling only | 12 | 0.79186 | See results JSON |
| Original + Chebyshev | 12 | 0.80226 | 17.00 |
| Coupling + Chebyshev + surface cache | 12 | 0.81778 | 19.34 |

The new path costs more than original 12-sweep VBD. Its advantage is better
volume preservation than the more expensive original 16-sweep run, not a speedup
at identical sweep count. The full 608-frame fixed-motion run had no post-step
assertion failures and minimum V/V0 0.81778.

The complete scene with its original live controller reached minimum V/V0
0.81531, maximum IK position error 0.003644 mm, and bag lift from 1.19915 m to
1.31581 m. Final acceptance recorded **one failure: volume below 0.85**. Do not
interpret the unit/regression tests as passing this physical acceptance.

At zero volume rate, the existing target-volume law reaches its 500 kPa pressure
cap at approximately V/V0 = 0.84161. Bulk damping shifts this threshold during
motion. Finite capped pressure does not enforce a hard minimum volume.

## Verification and artifacts

- Independent finite differences check forces and local Hessians of two closed
  cavities sharing a vertex, with interleaved face rows. The original committed
  kernel fails this derivative test; corrected kernels pass on CPU and CUDA.
- Dense solves and force residuals verify the coupled update in interior, lower
  and upper pressure regimes, singular/fixed rows and 513-particle colors.
- Cached and uncached integration agree on a membrane-only pneumatic model.
- All 26 selected pneumatic, surface-cache and Chebyshev regression tests passed.
- All pre-commit hooks passed for the changed files.

`results.json` retains numerical summaries and the complete-scene acceptance
failure. Local detailed trajectories and diagnostic scripts remain under the
ignored `newton/tests/outputs/inflatable_volume` and `inflatable_perf` directories.
