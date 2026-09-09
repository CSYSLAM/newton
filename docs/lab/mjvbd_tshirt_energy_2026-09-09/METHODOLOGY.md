# T-shirt energy comparison

Source: `FAST_MJVBDV2`, HEAD `cfe948b9` (production code inherited from
`8e72b4b9`). Measurements made on RTX 5060 Ti, Warp 1.17.0.
Only `scripts/plot_mjvbd_tshirt_energy.py` was added; solver and demo unchanged.

## Experiment

- Default: current `surface-fast`, seven batched sweeps plus one ordinary
  sweep; current Chebyshev/cache settings, multilevel disabled.
- Ordinary20: current scene and material/contact settings with the original
  ordered GS algorithm, 20 ordinary sweeps, no acceleration preset. This is
  not a historical checkout with historical contact capacities.
- Both: 1200 frames / 20 simulated seconds, 10 substeps per frame, sample
  every five frames including initial state. Scripted motion ends at 6.55 s.
- Common-state probe: first initialized substep after default frame 1200,
  restored into both algorithms with identical inertia target, anchor,
  contact records, and initial state; record every sweep including sweep 0.
- No added damping, smoothing, or end-of-trajectory velocity attenuation.
- Both demo final checks passed. Recorded contact-row overflow was zero.
  At most one asymmetric EE pair occurred in either full trajectory; the
  common-state probe had none. Checks are at sampled states, not every step.

## Energy meanings

Full-trajectory cloth mechanical energy is kinetic + gravitational +
membrane + bending energy. Gravity uses its initial value as the zero.
Membrane rest energy is subtracted as an additive constant. Bending uses
the original solver angle kernel and rest angles; its initial 2.43319 J
is not artificially removed. The separately plotted contact contribution
integrates current normal forces over the retained detector records,
counting canonical EE pairs once.

The robot is a prescribed external boundary. These plots are NOT total
energy of a closed cloth/robot system: robot kinetic energy, actuator work,
and accumulated frictional heat are not included. Normal contact energy
is a retained-record diagnostic, not an exhaustive collision certificate;
asymmetric EE updates need not be a gradient of the canonical pair scalar.

The per-sweep fixed diagnostic includes inertia, membrane, bending,
elastic damping, normal contact potential with frozen initial stiffness,
and a frozen-contact friction/damping surrogate. Friction normals,
weights, and loads are frozen at the initial iterate. The production
solver changes contact geometry/coefficients, so this is NOT its exact
global scalar objective. The separate conservative/inertial curve omits
friction/damping and uses current body-contact stiffness. It need not
decrease each sweep. Gravity is already in the inertia target and is not
added a second time. The surrogate assumes zero local body surface
velocity (mesh collider, no conveyor); its damping omits the solver's
tiny normal-rate activation deadband.

## Results

| Metric | Default | Ordinary20 |
|---|---:|---:|
| Mechanical energy at 20 s [J] | 18.04868146 | 11.07383732 |
| Mean kinetic energy, sampled 15–20 s [J] | 1.77976177e-8 | 6.10862275e-9 |
| Common initial fixed diagnostic [J] | 21.59356364 | 21.59356364 |
| Final fixed diagnostic after 8 / 20 sweeps [J] | 18.32741006 | 18.14264678 |

Both common-state fixed diagnostic curves decreased at every measured
sweep. Ordinary20 continued decreasing after sweep 8. In the full
trajectory, the default has approximately 2.91 times the tail mean kinetic
energy. This supports greater residual motion in this run, but is not a
proof of the cause of visible jitter. Full trajectories diverge, so their
final energies alone cannot quantify same-state solver accuracy. A single
substep probe cannot establish convergence equivalence across the scene.

## Files and reproduction

- `energy_history.png`: complete process and energy decomposition.
- `energy_tail.png`: final five seconds, without smoothing.
- `energy_per_sweep.png`: common-state substep objective versus sweep.
- `*_trajectory.csv`, `common_substep_sweeps.csv`: raw measurements.

Run from the repository root:

```powershell
uv run --no-sync python scripts/plot_mjvbd_tshirt_energy.py --mode default
uv run --no-sync python scripts/plot_mjvbd_tshirt_energy.py --mode ordinary20
uv run --no-sync python scripts/plot_mjvbd_tshirt_energy.py --mode plot
uv run --no-sync python scripts/plot_mjvbd_tshirt_energy.py --mode self-test
```

Two formula tests pass: scalar contact derivatives and finite differences
of elastic/damping energy against original membrane/bending force kernels.
Observation performs CPU readback, so these runs are not timing benchmarks.
