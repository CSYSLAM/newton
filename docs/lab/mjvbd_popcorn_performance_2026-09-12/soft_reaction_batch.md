# Rejected experiment: batch independent rigid-side soft reactions

The accepted 16-substep / 24-sweep setup remains unchanged. Do not change
grain count, paper material, friction, contact coverage, IK or acceptance limits.

## Implementation

Dense body/particle accumulation now honors effective inverse mass, just like
the sparse path. Both partial generation and reduction skip immovable bodies;
otherwise the reduction could read stale partials. This does not remove
particle-side contact forces or change prescribed collider motion. The internal
rigid force scratch is not a kinematic reaction-reporting API (the sparse path
already skips it).

The new immovable regression fails before the fix: dense reaction -3392.4001
versus zero on the sparse path. It passes afterwards, including poisoned unused
partials. Finite-mass dense comparisons remain.

Rigid-side soft reactions are accumulated over all bodies once per iteration,
before the rigid color loop. Their dependencies are the associated body's own
pose and fixed particle positions. Neither changes before that body's color is
solved. Rigid/rigid forces and joint constraints retain color ordering, and all
dual updates retain their original location. No reduced iteration cadence or
lagged inter-body force cache is introduced.

A CPU/CUDA regression defers soft launches back into the color loop and checks
the resulting body and particle poses against the batched implementation, with
both rigid/rigid and rigid/cloth contacts active over four iterations.

## Measurements

Same isolated 30 warm-up + 60 measured frames, ViewerNull, RTX 5060 Ti:

- Previous accepted implementation: 152.417 ms/frame.
- Dense immovable filtering alone: 154.135 ms/frame; no speedup established.
- Filtering plus batched soft reactions: 148.477 ms/frame, 6.735 FPS.

The modest short-window difference is not evidence of 30 FPS. The full 48-second
run is for correctness: regression tests overlap a portion of that run, so its
wall-time output must not be used as an uncontended performance measurement.

Logs in the external archive referenced by README.md:
`dense-static-before.log`, `dense-static-tests.log`, `dense-static-timing.log`,
`soft-batch-timing.log`, `soft-batch-equivalence.log`, `soft-batch-full48.log`,
`soft-batch-regression.log`.

## Next experiment: fixed geometry caches

Full-scene result: **No-Go**. Five-finger coverage was 100%, cup lift 109.64 mm,
distortion 1.959 mm and no runtime slip guard fired. However, only 16 of the 26
lifted grains were retained (61.54%), below the unchanged 70% requirement.
The local equivalence test does not guarantee an identical long atomic-contact
trajectory. Both this turn's dense filtering and batched scheduling code were
removed, along with their experimental tests; prior accepted optimization code
was preserved. The 161.222 ms full-loop output is not a valid isolated benchmark.

The cache combination initially measured 143.003 ms/frame on top of the rejected
batching experiment. That is a probe, not the performance of retained defaults.
The next full run uses the accepted original color schedule plus caches only.

The current full rigid-soft scene does not satisfy the external-rigid surface
preset gate; specifying `surface-fast` therefore does not enable its caches.
`cache_probe.py --surface --dat` explicitly probes only the existing bending
anchor and DAT geometry caches, without enabling Chebyshev, batching particle
colors or reducing iterations. Production defaults are not changed by the probe.
Cache refresh must honor each substep and collision rebuild, including changes
to paper plastic rest angles. Short timing alone does not establish acceptance.

The accepted color schedule plus both caches failed at 45.950 seconds with
31.4 mm cup slip (`cache-baseline-full48.log`). This combination is not enabled
in the demo. A new CUDA replay regression verifies that changed plastic rest
angles are consumed, with nonzero bending stiffness; freezing material state
was not the explanation for this failure.

The first parallel coarse restriction probe (64 lanes per cluster) matches
double-precision sums for empty and ragged clusters, including 1025 particles
and replay with changed inputs. Combined-cache short timing with this reduction
was 140.762 ms/frame (`cache-tile-timing.log`), but this does not establish a
successful delivery. Production restriction remains serial during experiments.

Next isolated full check: original surface solver, DAT cache and tiled coarse
restriction (`dat-tile-full48.log`). Separately, `--dense-graph-gate` probes a
CUDA conditional node around the two dense reaction kernels. A device-side
threshold flag is rebuilt with the contact lists; empty dense work is skipped,
not sparse contacts. The original eager path remains unchanged. These switches
are diagnostic-script options, not new production solver modes.

## Accepted intermediate implementation

DAT cache + tiled restriction, with the original surface solve and 24 sweeps,
passed full acceptance: 18 lifted / 15 retained, 83.33%, five-finger coverage
100%, cup lift 109.99 mm. The isolated full-loop time was 160.648 ms/frame.
This is now integrated directly: tiled CUDA restriction (serial CPU fallback)
and the existing DAT-cache option in the popcorn demo. The exact production
short benchmark measured 141.151 ms/frame (`dat-tile-default-timing.log`).
97 relevant tests plus 26 multilevel tests passed. Surface caching remains off.

## Stronger global correction experiments

Cluster size 32 instead of 400, 12 sweeps with corrections every three sweeps,
unchanged 16 substeps, DAT cache and tiled restriction: **105.230 ms/frame**
over 2850 measured frames. Full acceptance failed: 18 lifted / 12 retained
(66.67%). Cup held throughout, five-finger coverage 100%, RMS distortion
2.718 mm. This is not an accepted speedup. Log: `global32-sweep12-full48.log`.
The next experiment keeps cluster size 32 and restores 16 sweeps, with a coarse
correction every four sweeps. Production defaults stay at the accepted budget
until full acceptance passes.

The 16-sweep / cluster-32 / checkpoint-4 run also failed: cup slip reached
35.2 mm at 25.900 seconds (`global32-sweep16-full48.log`). Do not enable either
reduced-sweep configuration. More coarse DOFs alone did not make the current
coupled grasp converge robustly under these lower budgets. This is an empirical
failure, not a proof that global methods cannot work.

Final retained defaults for this turn: 16 substeps, 24 sweeps, cluster size 400,
coarse correction every four sweeps, DAT cache enabled, tiled CUDA restriction,
original surface solve. The other experiment switches exist only in this lab's
probe script. The standalone dense-graph gate measured 146.165 ms/frame on the
earlier uncached baseline; it was not integrated or full-scene validated.
No stage or commit was performed. No 30-FPS claim is made.
