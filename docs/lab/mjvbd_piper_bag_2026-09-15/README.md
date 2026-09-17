# PiPER bag: retained configuration and validation

The demo uses the original FBD_03 orientation, 15 VBD sweeps and 10 substeps.
Rigid-soft DAT and contact-aware Chebyshev are enabled only in this demo;
other demos retain the solver default (DAT disabled). Both rack and ball use
SDF edge/face contacts. No pinned cloth vertices or prescribed ball motion.

The final 60-second run passed final acceptance. Diagnostics sampled every
30 frames: minimum bag/table clearance after frame 120 was 3.884 mm, final
clearance 3.975 mm, maximum enclosing-sphere penetration 0.443 mm. The sphere
metric is conservative for the original nonspherical ball mesh. Late sampled
step time averaged 94.9 ms (10.5 FPS); RMS frame displacement averaged 0.142 mm.
Residual jitter and submillimeter overlap are not claimed to be eliminated.
See `fbd03-dat-full-surface-results.json`.

An independent 600-frame recording passed all per-frame and final checks;
`fbd03-dat-full-surface-recording.json` records its settings and timing.
The video uses offline 60 FPS playback, not real-time simulation throughput.

The subsequently rejected 90-degree rotation and material/margin experiments
are not included. The unused coupled-global demo branch and fine bag asset
have been removed. Historical logs and pre-cleanup files remain locally in
`newton/tests/outputs/piper_bag/cleanup_archive_20260915/` (ignored by Git).
The retained recordings remain in `newton/tests/outputs/piper_bag/performance/`.

Retained regression coverage includes coincident cloth contact normals,
compact truncation-cache parity, DAT clipping/kinematic exclusion, guarded
Chebyshev exclusion, and bag geometry/placement checks.

## Pre-push regression check

101 tests passed across solver routing, contact invariants/optimizations, CUDA
fast paths, displacement deadband, cache parity, coincident contact, rigid-soft
DAT, guarded Chebyshev, and bag geometry. On CPU and CUDA, a self-contact cloth
fixture with DAT disabled matched the old full-VBD solver's particle positions
and velocities exactly after 20 steps. This is targeted regression evidence,
not an exhaustive guarantee for every demo. Coincident-contact regression
failed in both old backends and passed with the fix.

Staged-file checks passed. The full-repository pre-commit run reported existing
ruff/typos issues in the unrelated popcorn lab scripts dated 2026-09-12. Those
files were not changed. Required source assets larger than 500 KiB use Git LFS.

After integrating remote popcorn sleeping commit `bd6a03668`, the same 101
tests passed again, plus five rigid sleeping/contact-group tests. Both
initialization paths were retained when resolving the solver merge conflict.
