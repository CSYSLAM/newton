# Two cuts, three bread pieces

```powershell
uv run --no-sync -m newton.examples mjmpm_w1_bread_three_pieces
```

This independent 20-second example inherits the existing bread scene's
robot, knife, board, material, particle rendering and optimized solver paths.
It does not duplicate asset construction or change the single-cut trajectory.
The default run is 1,200 frames, with the same 7,376 samples, 6 substeps,
50 MPM iteration limit and 24 connection CG iterations per substep.

The first stroke and withdrawal are unchanged through 9.6 seconds. The tool
then pauses above the board, traverses from the first cut at y = -0.275 m
to the opposite side at y = -0.205 m, and cuts again. The cutting edge clears
the loaf during the traverse. The second stroke finishes at 15.7 seconds,
pushes its outer piece outward, and withdraws by 18.5 seconds. The final
1.5 seconds show the unloaded three-piece result.

The knife remains attached to the dynamic MuJoCo wrist. Both cuts use the
actual blade sweep and persistent connection history; the material starts
with one connected component. No particles are removed or repositioned,
and no predetermined slice is detached. MPM damage and connection plasticity
continue accumulating through both cuts. The optional single-cut surface
skin cannot represent this topology, so this example requires particle
rendering. Authored knife and board meshes are shared with the first scene.

## Acceptance

Every tested frame checks finite state, speeds below 3 m/s and finger/handle
clearance. Stable phases require one piece before the first stroke, two after
the first withdrawal, and three after the second cut. No component may be
smaller than 10% of the material. From 7 seconds onward, every tested frame
also rejects spatially detached samples inside each connected component.

Final checks require unchanged particle count/mass, dynamic tool ownership,
board support, exactly three coherent pieces, and observation of the full
one-to-two-to-three progression. Each cut must retain at least 2 mm average
opening, 0.5 mm lateral recoil and, with damage enabled, 1.5 mm upper notch.
Opening is measured relative to the original near-cut samples. The first
cut's recoil is measured before the second stroke; each notch is measured
in the final unloaded state against the settled pre-cut material.

RTX 5060 Ti validation, 2026-10-23: all 1,200 frames passed. The final pieces
contain 1,992, 2,204 and 3,180 particles, with zero detached samples. First
and second retained notches are 2.10 and 1.72 mm; recoil is 3.57 and 3.70 mm.
Minimum visual grip clearance is 0.399 mm. Parallel floating-point evaluation
can produce small run-to-run differences.

The hybrid material has the same limitations as the [single-cut example](BREAD_CUT.md):
it is not a generic multi-field crack/contact solver or calibrated fracture model.

## Record and test

```powershell
uv run --no-sync python -B scripts/record_mjmpm_bread_cut.py --scene three-pieces --video --width 1920 --height 1080 --output-dir output/bread_three_pieces
uv run --no-sync python -B -m unittest newton.tests.mjmpm.test_bread_three_pieces newton.tests.mjmpm.test_bread_cut newton.tests.mjmpm.test_bread_cut_cohesion
```

The recorder defaults to 1,200 frames for this scene, saves before/between/after
stills and video, and writes physical snapshots, connection histories and
acceptance metrics. Its original single-cut command remains supported.
Unit tests cover unchanged first-stroke targets, clearance during traverse,
and two physical blade sweeps producing one, then two, then three components
on CPU and available CUDA devices, preserving earlier severed connections.

All 67 MJMPM tests and pre-commit checks passed. A negative control omitting
the second blade sweep fails the three-component unit acceptance. A second
full run with continuous 1080p recording also passed: retained notches were
2.09 and 1.69 mm, with no detached samples. The recorder reuses its RGB device
buffer throughout the longer export.
