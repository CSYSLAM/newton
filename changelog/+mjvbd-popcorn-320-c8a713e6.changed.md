Optimize opt-in MJVBDV2 CUDA surface/contact execution and coupled PCG, and
double the popcorn demo's default population to 320 dynamic grains. Use
`--popcorn-count 160` to retain the previous population. The demo now uses six
substeps with sixteen coarse PCG iterations; this changes discretization, not
contact laws or grasp acceptance thresholds.
