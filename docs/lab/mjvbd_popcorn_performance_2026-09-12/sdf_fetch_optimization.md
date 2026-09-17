# Popcorn SDF fetch optimization (2026-09-13)

Target: 30 FPS (33.33 ms/frame), including ordinary 1920x1080 rendering.
Keep the default scene, grains, substeps, solver iterations and physical
acceptance checks unchanged. No grasp attachment or artificial damping.

## Profiling

`profile_current.py` profiles CPU submission/rendering and separately lists
uncaptured GPU kernels. The CPU render synchronization includes outstanding
simulation time: it is not evidence that rendering alone costs 39 ms/frame.
Uncaptured GPU timings are diagnostic and must not be summed as CUDA Graph
frame timing. Compact soft face contact search was the largest uncaptured
kernel group (13.088 ms across eight launches).

## Search-local texel cache: do not enable in production

`sdf_cell_probe.py` is an isolated, explicitly installed experiment. It caches
eight immutable texels only within one golden-section search. It does not
cache contact results across time steps.

The independent line-search comparison passed 24,576 segments. The full
48-second scene experiment passed its physical checks at 44.936 ms/frame
(22.254 FPS), delivering 22 of 27 lifted grains. However, integration caused
both existing fixed-point face-search exact-equality tests to fail. Restoring
the original search made both tests pass. The cause of this interaction has
not been established; line-search equality alone is insufficient validation.
Consequently the production search and cache integration were restored. The
22.254 FPS result is NOT a validated default-path improvement.

## Paired texel fetches: do not enable in production

Instead use both channels of the existing paired-X texture storage when
reading a cell's corners. Four texture reads replace eight in paired storage;
scalar storage retains eight reads. Cell selection, quantization decoding,
interpolation, gradients and search iteration counts are unchanged.

`test_sdf_paired_corners.py` freezes the old eight-read corner sampler for
independent comparison of all eight values and three interpolation weights.
It covers scalar and paired storage, popcorn and box meshes, random and grid
boundary queries, and CUDA Graph replays. The existing fixed-point face and
search-distance tests also remain required.

Full-scene measurement uses `uv run --no-sync python
docs/lab/mjvbd_popcorn_performance_2026-09-12/validate_default.py` without
experimental installation or parameter overrides. Initialization/compilation
and 30 warmup frames are excluded, followed by 2850 timed frames.

Measured 47.853 ms/frame (20.897 FPS). All four unit tests passed, including
32,768 corner queries plus two CUDA Graph replays for each configuration.
Same-state render comparison found zero changed channels. But the final
physical acceptance failed: 16 of 23 lifted grains remained in the cup
(69.565%, below the unchanged 70% requirement). Cup distortion RMS was
1.817 mm, upright cosine 0.987, and minimum rim radius 30.458 mm.

This does not establish a mathematical error in paired reads, nor prove the
trajectory difference is only pre-existing nondeterminism. Do not dismiss the
failure or relax the acceptance threshold. The production changes were
removed. `paired_corners_experiment.patch` and the adjacent corner test retain
the bounded experiment for further investigation.

## Outcome and next steps

30 FPS has not been achieved. Neither experiment is enabled by default.
The original solver, demo budgets, grasp controller and acceptance checks
remain unchanged. No runtime performance improvement is claimed for the
restored default path.

The next larger optimization should reduce unnecessary face/shape search
work through conservative candidate pruning and workload batching, rather
than reducing physics budgets. Validate candidate coverage against the
original pipeline and use repeated complete-scene acceptance runs. At roughly
50 ms/frame, 30 FPS requires about 16.7 ms/frame saved (one third of total
frame time); small texture optimizations alone are insufficient.
