# Conveyor sorting performance investigation

For the subsequent code change and full-sequence validation, see
[implementation.md](implementation.md). The measurements below describe the
initial diagnostic experiment.

Baseline commit: `72c275654aa811cfce7f1bc64bf15c201e294811`.
RTX 5090 D v2, 2,744 particles, 4,508 triangles, 3,000 tetrahedra,
282 shapes and 68 bodies. Preserve 8 substeps, 16 VBD sweeps and 24 IK
iterations per solve (two solves per motion plan).

## Initial-stage measurement

180 frames; means exclude the first 30. ViewerGL headless at 1280x960,
CUDA/OpenGL interop disabled. These are synchronized diagnostic step/render
measurements, not a complete-sequence or interactive-viewer FPS guarantee.

| Component | Original | Experimental graph/cache path |
| --- | ---: | ---: |
| Total step | 51.10 ms | 38.77 ms |
| Motion planning (included in step) | 14.51 ms | 4.34 ms |
| Render | 13.63 ms | 12.71 ms |
| Step + render | 64.73 ms | 51.49 ms |
| Equivalent FPS | 15.45 | 19.42 |

Original physics graph takes 36.27 ms/frame. In the experimental probe,
`graph_ms` includes the newly captured IK graphs as well as physics; it must
not be interpreted as an isolated physics measurement.

The diagnostic experiment captures the two IK solves separately and enables
existing surface, self-contact truncation and incremental pneumatic-volume
caches. It does not use the rejected color-coupling/Chebyshev experiment or
change material, collision frequency, mesh resolution, iterations or substeps.
No production code had been changed at this initial diagnostic stage. Physical equivalence and completion of
all four sorting phases have **not** been validated for this experiment.

## Remaining GPU work

A two-frame Nsight CUDA Graph node trace of the experiment reports these
fractions of instrumented GPU kernel time (not full-frame time):

- Surface elasticity: 13.9%.
- Pressure force/Hessian accumulation: 13.1%.
- Tetrahedral elasticity: 10.6%.
- Particle/body contact accumulation: 8.8%.
- Incremental cavity updates: 6.8%.
- Edge/edge and vertex/triangle self-contact detection: 9.1% combined.

The pressure path is worth inspecting: it is dispatched using whole model
particle color groups, although only the inflatable parcel belongs to the gas
cavity. Compact cavity-only particle lists could avoid processing unrelated
cloth/soft-body vertices and enable the existing small-cavity fused dispatch.
This is a proposed optimization, not a measured saving yet.

30 FPS requires <=33.33 ms including rendering. The demonstrated experiment
still needs about 18.2 ms/frame of further savings. IK optimization alone is
insufficient; physics dispatch/assembly and rendering need further work.

Numerical summaries and top kernels are in `results.json`. Local reproduction
script: `newton/tests/outputs/conveyor_performance/probe.py`; raw Nsight trace:
`/tmp/conveyor_profile.nsys-rep`. These local diagnostics are not a validated
replacement for the example.
