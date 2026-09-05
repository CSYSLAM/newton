# MJVBDV2 optimization log

This file is the performance decision record for MJVBDV2. The current solver
architecture and numerical contracts are documented in `MJVBDV2_PLAN.md`.
Entries here are append-only: rejected and reverted experiments remain visible
so that they are not repeated without new evidence.

The history before this file was introduced was reconstructed from commits and
tests. Where no controlled timing was preserved, the entry says so instead of
inferring a speedup from reduced work.

## Recording policy

Every performance change must record:

1. the date, commit or experiment identifier, and status;
2. the affected backend and representative scene topology;
3. the work removed or added and the numerical invariants that must remain;
4. hardware, graph mode, substeps, iterations, topology counts, warm-up, and
   timing statistic for each benchmark;
5. correctness tests and representative visual checks; and
6. the retain, revise, or reject decision.

Correctness tests are not performance evidence. Prefer an A/B comparison of
the exact parent and candidate revisions, multiple warmed samples, and median
GPU time. Record end-to-end frame time separately from an isolated kernel
measurement. FPS from an interactive viewer is useful supporting evidence but
must not be the only measurement.

Status values are:

- **Retained**: present in the current implementation.
- **Rejected**: measured and deliberately not retained.
- **Pending**: implemented or proposed, but not yet supported by enough data.
- **Superseded**: replaced by a later entry.

## Decision summary

| Date | Change | Affected path | Status | Evidence |
| --- | --- | --- | --- | --- |
| 2026-09-05 | Extend tetrahedral coarse clusters from six rigid modes to twelve affine modes | CUDA `vbd/` mixed and tet multilevel paths | **Rejected, reverted** | Armadillo position error improved but frame time regressed 49.0%; the mixed soft-tet-cube plus cloth-bag scene regressed 32.2% with no accuracy improvement, and cold module compilation grew from about 3 to 10 seconds |
| 2026-09-04 | Extend the particle coarse space with six-DOF rigid modes and a Galerkin tet operator | CUDA `vbd/` mixed surface/tetrahedral paths | **Retained, opt-in** | The 240-frame Armadillo run was 3.93% faster than 10 sweeps while reducing the plain-6 position error from 6.83 to 3.53 mm; the complete 1,900-frame soft-cube-plus-cloth final was 9.29% faster than 12 sweeps and passed its placement test |
| 2026-09-04 | Add conservative eligibility and device-side rejection to the particle multilevel correction | CUDA `vbd/` and `vbd_soft/` cloth/shell paths | **Retained, opt-in** | A graph scale scan bounded the current one-block PCG policy to 128--1,500 clusters; all 120 sampled T-shirt frames passed the residual/finite/clamp guards, while unsafe plastic-bag settings were detected and left on their explicitly validated policy |
| 2026-09-04 | Incrementally update pneumatic volume by unique color-cavity faces | CUDA `vbd/` pneumatic surfaces | **Retained, gated** | Two alternating 50-frame plastic inflatable-bag runs reduced mean summed GPU kernel time from 28.169 to 25.785 ms/frame (8.46%) and pneumatic kernels from 7.366 to 5.254 ms/frame (28.67%); two 300-frame CUDA Graph runs reduced mean wall time from 25.142 to 22.595 ms/frame (10.13%) |
| 2026-09-04 | Cache rigid-soft contact color masks in the optimized soft backend | CUDA `vbd_soft/` contact-major particle contact scatter | **Rejected, reverted** | An isolated mixed-contact graph improved 8.60%–10.96%, but real fixed-frame examples measured -1.0% for gear crusher, +1.0% for nonwoven bag, and -0.5% for kinematic T-shirt; the added mask build did not produce a reliable end-to-end gain |
| 2026-09-04 | Add one contact-aware two-level correction after the final surface-particle VBD sweep | CUDA `vbd/` and `vbd_soft/` cloth/shell paths | **Retained, opt-in** | The 300-frame T-shirt comparison reduced wall time 30.3% while cutting the 12-sweep error against 20 sweeps by 41.4%; the plastic-bag final improved from 17.3 to 20.1 FPS while cutting the 8-sweep error against 12 sweeps by 41.7% |
| 2026-09-04 | Temporally warm-start stable mesh-SDF face contacts, with bounded reuse and exact fallback | CUDA `full` contact backends with full-surface mesh-SDF contact | **Retained, gated** | The Armadillo frozen-state collision Graph fell from 3.653 to 2.575 ms (41.9%) with all 27,444 contact keys preserved; the 240-frame plastic-bag scene remained 17.5 FPS and passed with the cache both enabled and disabled |
| 2026-09-03 | Build particle self-contact adjacency and gather VT/EE force/Hessian per colored particle, with and without fusion into the cloth tile | CUDA `vbd_soft/` surface self-contact solve | **Rejected, reverted** | The T-shirt Graph fell from 10.1 FPS to 7.40 FPS with tile fusion and 5.39 FPS with a separate gather kernel; duplicated narrow phase, divergent lists, and tile register pressure outweighed removed atomics and row scans |
| 2026-09-03 | Traverse an overallocated rigid-soft contact stream with a fixed active-prefix worker grid | CUDA Graph `vbd_soft/` particle-side rigid-soft scatter | **Retained, gated** | The 6,436-particle T-shirt Graph improved 3.40%--4.10% in same-process A/B runs; the unchanged 31,768-capacity tablecloth path and a 300-frame cloth-twist regression passed |
| 2026-08-28 | Let one canonical EE owner evaluate both directed filter sides and write the legacy rows | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | Exact directed rows were preserved, but dense 28-by-28 two-layer EE detection regressed 46.6% without rest exclusion and 26.1% with a 0.03 m rest exclusion because dual atomic row updates outweighed the saved narrow phase |
| 2026-08-28 | Schedule VT/EE source queries in static rest-space Morton order | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | Dense two-layer cloth detection regressed 4.6%; a sparse supermarket-bag state improved only 1.7%--2.3%, about 0.01 ms, because scattered row/filter writes offset more coherent BVH traversal |
| 2026-08-28 | Skip color-irrelevant directed EE rows before force/Hessian traversal | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | A frozen four-color two-layer cloth Graph improved only 0.42%, while a 10-second supermarket-bag Graph fell from 38.1 to 37.7 FPS and timed EE/VT force launches were 4.13% slower |
| 2026-08-28 | Precompute rest-shape VT/EE exclusion CSR | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | A two-layer 28-by-28 cloth Graph made VT/EE detection 19.1% slower after CSR rows grew to 190,236/918,016 entries; a 10-second supermarket-bag frame benchmark improved only 1.7%, below the 5% gate |
| 2026-08-28 | Canonicalize directed edge-edge self-contact records with side masks | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | Dense two-layer cloth reduced 163,224 directed rows to 90,576 pairs, but full eager GPU steps regressed 23.5% (`vbd/`) and 17.5% (`vbd_soft/`) |
| 2026-08-28 | Compress retained self-contact rows into VT/EE active streams | Complete `vbd/` and optimized `vbd_soft/` self-contact | **Rejected, reverted** | Dense 24-by-24 cloth Graph replays showed only order-sensitive 1%--5% gains and eager steps were neutral; the extra kernels, fixed stream memory, and dual implementation are not justified |
| 2026-08-28 | Reject spatially remote sparse point-contact pairs before SDF evaluation | Dynamic and kinematic `soft` backends | **Pending, low impact** | Dynamic W1 fold collision Graph fell from 0.024180 to 0.017609 ms (27.18%), but only 0.006571 ms per call; CPU/CUDA contact records remain equivalent |
| 2026-08-21 | Reduce dense body-particle contacts in parallel | Complete `vbd/` AVBD path | **Retained, gated** | A 200-frame supermarket-bag CUDA Graph run fell from 37.670750 to 28.246249 ms/frame (25.02%); a 1,024-contact isolated Graph was 25.23x faster |
| 2026-08-21 | Prune zero-inverse-mass and kinematic rows from rigid color groups | Complete `vbd/` AVBD path | **Rejected** | Both end-to-end dense-contact variants included pruning, so no gain was isolated; the host-cached launch topology would require Graph recapture after runtime mass or flag changes |
| 2026-08-21 | Add another AABB pass to sparse point contacts | Dynamic and kinematic `soft` backends | **Rejected** | Rejected before implementation: the dynamic fold scans 566,368 candidates in 0.026726 ms per collision Graph; even ten calls are below 0.1% of the measured frame |
| 2026-08-21 | Move shape-major full-surface AABB rejection into a private V2 collision pipeline | All `full` contact backends with full-surface contact on CUDA | **Retained** | Frozen frame-121 collision time fell from 3.825585 to 1.643479 ms (57.04% lower, 2.33x throughput) with the same 9,838 contact keys |
| 2026-08-21 | Fuse coupled transfers and pipeline MuJoCo/VBD substeps on two CUDA streams | Dynamic coupled backend | **Rejected** | The real dynamic fold was 1.32% slower with transfer fusion and 0.90% slower with the two-stream wavefront; a 1.66% light-scene gain did not transfer to the representative workload |
| 2026-08-24 | Capture complete dynamic-fold frames on one CUDA stream | Dynamic coupled example execution | **Retained** | The dynamic T-shirt fold improved from 3.42 to 16.2 FPS (4.74x); particle state remained bitwise equal and joint differences stayed below 5.37e-7 |
| 2026-08-24 | Solve dynamic-fold IK during each captured frame | Dynamic coupled example execution | **Retained** | Realtime IK Graph replay reached 14.7 FPS; Graph/eager IK targets were exact, particle state was bitwise equal, and the complete 900-frame test passed |
| 2026-08-24 | Fuse mutually exclusive active/inactive truncation application | Optimized `vbd_soft/` self-contact path | **Retained** | An isolated 1,800-iteration Graph was 46.6%--46.8% faster for both selector values; outputs were bitwise equal and the 450-frame dynamic fold passed |
| 2026-08-21 | Selectively port PR 3995 soft-contact traversal and capacity sizing | Both private VBD implementations and sparse contact helpers | **Retained, batch-gated** | A 1,024-world CUDA Graph benchmark is 4.81% faster; world-compatible preallocation removes a 1,024x capacity overestimate in that setup |
| 2026-08-21 | Cache a particle-color membership mask per rigid-soft contact | Complete `vbd/` particle-side rigid-soft solve | **Retained** | Frozen contact solve is 5.39% faster including mask construction; full handoff graph is 6.99% faster in a consecutive run |
| 2026-08-24 | Compact AABB-active full-surface candidates before SDF optimization | CUDA `full` contact backends with full-surface contact | **Retained, gated** | Handoff collision Graph median is 10.58% faster and full frame median is 6.58% faster; pneumatic and 100%-active collision Graphs are 60.54% and 47.21% faster |
| 2026-08-25 | Remove rigid-only `shape_gap` from the private full-surface AABB mask | CUDA `full` contact backends with full-surface contact | **Retained** | The 300-frame Armadillo grasp improved from 6.34 to 7.00 FPS; its close-phase window improved from 4.10 to 5.51 FPS with unchanged contact thresholds |
| 2026-08-25 | Retune self-contact block size, rebuild cadence, and reference-distance ordering | Private V2 self-contact | **Rejected** | Block size 16 remained fastest; rebuilding changed 3.312 to 3.319 ms; deferred reference tests reduced the 300-frame result from 7.00 to 6.80 FPS |
| 2026-08-21 | Shape-major conservative AABB mask before full-surface SDF optimization | Shared full-contact edge/face generation | **Rejected** | Frozen-state collision graph was 39.4% faster, but the implementation changed Newton's shared collision pipeline and violated MJVBDV2's standalone migration boundary |
| 2026-08-21 | Exact particle CSR and per-color compact contact lists | Complete `vbd/` particle-side rigid-soft solve | **Rejected** | Per-particle CSR was 71.7% slower; compact color lists improved the full graph only 0.21%, within noise |
| 2026-08-20 | Full-VBD device-selected truncation fast path | Complete `vbd/` particle iterations | **Rejected** | 0.59% in the small pneumatic bag and no repeatable gain in the supermarket bag |
| 2026-08-20 | Copy particle output once per step; specialize surface-only CUDA tile solves (`685aa797`) | Complete and soft VBD particle iterations | **Retained** | 3.3% to 8.0% end-to-end gain in two supermarket-bag A/B runs; combined contribution only |
| 2026-08-19 | Traverse active self-contact records instead of allocated capacity (`5ebbf77f`) | All Newton VBD variants, including both V2 implementations | **Retained** | User-visible gain reported; no standardized timing archive |
| 2026-08-13 | Device-selected self-contact path, sparse soft-contact launch, and scalar fallback for small color groups (`bb2791da`) | Optimized `vbd_soft/` path | **Retained** | Regression coverage exists; no standardized timing archive |
| 2026-08-13 | Device-resident soft-contact material selector (`a299dfc1`) | Captured full and soft VBD graphs | **Retained** | Removes host selection and graph re-recording; not a general per-frame speedup |
| 2026-08-13 | Optional MuJoCo sleeping and shared collision-query improvements (`24b3479e`) | `pure_mujoco` sleeping; applicable full-contact collision pipelines | **Retained** | Sleeping intentionally excluded from coupled paths; no V2-specific timing archive |
| 2026-08-10 | Articulation-only dispatch shortcut (`89002b52`) | `pure_mujoco` and `kinematic_passthrough` | **Retained** | Structurally removes VBD construction and execution; no standardized timing archive |

## Detailed experiments

### 2026-09-04: retain guarded particle multilevel correction

**Problem.** The initial two-level correction was deliberately opt-in, but it
had no automatic size policy and committed a coarse candidate before checking
whether PCG converged or most fine corrections hit their radius cap. This made
it too easy to apply the single-block coarse solve outside its measured range
or to silently accept an unhelpful correction.

**Implementation.** The solver now accepts
`particle_enable_multilevel_correction="auto"`. Automatic mode requires CUDA,
non-differentiable and non-deterministic execution, one particle world, no
tetrahedra, at least 1,024 topologically active surface particles, and 128 to
1,500 coarse clusters. Self-contact and pneumatic scenes still require an
explicit enable because their contact-growth and cavity-volume rejection tests
are not yet transactional. Isolated particles are excluded from the hierarchy.

The persistent PCG writes a device-side status and squared initial/final
residuals. Prolongation first writes a candidate buffer and counts non-finite
and radius-clamped corrections; it mutates the real displacement only if the
residual, finite-value, and configurable clamp-fraction checks all pass. An
optional `particle_multilevel_fallback_iterations` captures additional ordinary
VBD sweeps and selects them with `wp.capture_if` when the device status rejects
the candidate. The fallback runs in the same substep, which is more
conservative than waiting until the next substep and does not add a host
readback or change CUDA Graph topology.

**Scale evidence.** On an NVIDIA GeForce RTX 5090 D v2, Warp
1.17.0.dev20260807, CUDA Toolkit 12.9, and Driver 13.2, one correction was
captured and replayed after 20 warm-ups. Each result below is the mean of 500
replays through 512 clusters and 200 replays above that size.

| Particles | Clusters | Correction Graph |
| ---: | ---: | ---: |
| 520 | 65 | 0.042430 ms |
| 1,032 | 128 | 0.049689 ms |
| 2,056 | 257 | 0.056633 ms |
| 4,104 | 512 | 0.071493 ms |
| 8,008 | 1,001 | 0.138156 ms |
| 12,008 | 1,500 | 0.184706 ms |
| 16,008 | 2,000 | 0.237782 ms |
| 24,008 | 3,000 | 0.360834 ms |

The near-linear scan shows that the current block remains fast beyond 1,500
clusters on this GPU, but it does not establish good occupancy or batched-world
scaling there. Automatic mode therefore keeps the conservative 1,500-cluster
ceiling pending a multi-block PCG comparison.

**Runtime evidence.** The 6,436-particle, 996-cluster T-shirt scene passed all
120 sampled frames: status was zero and no fine correction hit the radius cap.
Its median final-to-initial squared residual ratio was `3.44e-5`, with a
95th percentile of `1.23e-3`. That measured example explicitly enables the
guards and falls back from 12 to 20 ordinary sweeps on rejection. Applying the
same provisional guards to the 5,886-particle plastic-bag/rod scene rejected 77
of 120 frames: its median clamp fraction was 0.728 and its Euclidean residual
was not monotone. That example therefore retains the previously measured
explicit correction behavior instead of silently changing its trajectory.

**Correctness evidence.** Nine focused tests pass across the complete and soft
private VBD implementations. They cover invalid modes, CPU/gradient/tet
fallback, scale eligibility, conservative self-contact rejection, fixed
anchors, long-range propagation, transactional clamp rejection, extra-sweep
fallback, and CUDA Graph capture/replay. A forced clamp rejection is bitwise
equal to ordinary VBD, including through the conditional Graph branch.

**Decision.** Retain automatic eligibility and transactional residual/finite/
radius-clamp checks as an opt-in policy. Do not broaden automatic mode to
self-contact, pneumatic, multi-world, tetrahedral, or more than 1,500-cluster
models until their specific rejection checks and scaling paths are measured.

### 2026-09-04: retain incremental pneumatic volume updates

**Problem.** The plastic inflatable-bag profile evaluated every cavity face and
pressure law once per particle color. In an eager 3-warm-up/5-sample profile,
volume accumulation and pressure evaluation consumed 14.86% of total GPU
kernel time; including pressure force/Hessian accumulation, pneumatics consumed
29.54%. This exceeds the 5% implementation gate.

**Implementation.** Construction builds a unique CSR row for each
`(particle color, cavity)` pair, plus a cavity-to-face CSR. At the beginning of
each VBD iteration, one block per cavity freezes the numerical anchor, computes
the full volume, and caches one contribution per face. After a color moves,
only its unique incident faces recompute their contribution and add the delta
to that cavity before the next color reads pressure. A face containing two
vertices of the same color appears only once in that row, preserving
Gauss--Seidel pressure semantics.

Initialization fuses full volume and pressure evaluation. Single-cavity models
whose largest color contains at most 512 particles additionally fuse the
previous color's volume update, pressure evaluation, and the next color's
pressure force/Hessian pass into one block. Multiple cavities retain one block
per cavity and the ordinary parallel force kernel. The option is available
only on non-differentiable, non-deterministic CUDA models; every other model
uses the original full recomputation. The measured full-W1 plastic inflatable-
bag example enables it.

**Performance.** On the same RTX 5090 D v2 software stack, the full-W1 plastic
inflatable-bag scene used 216 particles, 428 cavity faces, one cavity, eight
colors, five substeps, and 12 VBD iterations. Runs disabled CUDA Graph capture
to expose individual kernel costs, discarded 15 frames, and timed the next 50
frames with `wp.ScopedTimer(cuda_filter=wp.TIMING_KERNEL)`. Baseline and
candidate processes were alternated.

| Run | Full recomputation | Incremental |
| --- | ---: | ---: |
| A, summed GPU kernels | 28.220272 ms/frame | 25.679576 ms/frame |
| B, summed GPU kernels | 28.118220 ms/frame | 25.890692 ms/frame |
| Mean | 28.169246 ms/frame | 25.785134 ms/frame |
| Mean pneumatic kernels | 7.366125 ms/frame | 5.254412 ms/frame |

The candidate reduced summed GPU kernel time by 8.46% and pneumatic time by
28.67%. The first unfused prototype regressed because it replaced arithmetic
with the same number of tiny launches; it was superseded by the fused path and
must not be restored.

The production CUDA Graph path was measured separately with 90 warm-up frames
followed by 300 timed frames spanning the lift through release phases. Reverse-
ordered runs gave 24.871605 and 25.412919 ms/frame for full recomputation versus
22.569273 and 22.621135 ms/frame for the incremental path. Mean end-to-end wall
time fell from 25.142262 to 22.595204 ms/frame, a 10.13% reduction including IK,
collision detection, and Python frame orchestration.

**Correctness evidence.** A sealed-shell regression matches ten eager steps to
`2e-6`, and a two-world/two-cavity regression verifies color-cavity indexing.
A 120-frame dual-instance full-W1 comparison bounded maximum particle-position
difference to 1.08 mm, cavity-volume relative difference to 0.093%, and
pressure relative difference to 0.63%; this nondeterministic path changes
floating-point reduction order. The candidate completed the full 720-frame
CUDA Graph trajectory. The current branch's pre-existing example assertions
also fail with full recomputation: both variants first exceed the 0.5 mm IK
tolerance at frame 265 with the same 0.535 mm error, and their minimum volume
ratios are respectively 0.792163 and 0.792113 against a stale 0.85 threshold.

**Decision.** Retain the CUDA-gated incremental path and enable it in the
measured plastic inflatable-bag example. Keep full recomputation as the solver
default and as the deterministic, differentiable, CPU, and unsupported-size
fallback.

### 2026-09-04: final00 guarded-acceleration applicability sweep

The guarded multilevel and incremental-pneumatic paths were checked across all
five `mjvbdv2/*final00.py` examples on the same RTX 5090 D v2. CUDA Graph wall
times below use 60 warm-up plus 180 measured frames, except for the inflatable
bag's two reverse-ordered 90-warm-up/300-measured-frame pairs.

| Example | Reference | Candidate | Result | Decision |
| --- | ---: | ---: | ---: | --- |
| W1 T-shirt fold | 20 ordinary sweeps, 72.829 ms/frame | 12 sweeps + guarded multilevel, 50.643 ms/frame | 30.5% lower wall time | Keep; the complete 900-frame test passes |
| W1 plastic bag + rod | 12 ordinary sweeps, 36.190 ms/frame | 8 sweeps + multilevel, 28.722 ms/frame | 20.6% lower wall time | Keep explicit policy; the built-in 240-frame test passes |
| W1 inflatable plastic bag | Full cavity recomputation, 25.142 ms/frame | Incremental/fused cavity path, 22.595 ms/frame | 10.13% lower wall time | Keep; the complete 720-frame CUDA Graph trajectory runs |
| Soft then rigid cube into bag | 12 ordinary sweeps, 40.347 ms/frame | 8 sweeps + multilevel, 35.313 ms/frame | 12.5% lower than 12 sweeps, but 1.6% slower than plain 8 sweeps | Reject; the coarse correction worsened 240-frame RMS error |
| Armadillo gear crusher | Existing 10 sweeps, 95.793 ms/frame | Automatic multilevel | No active coarse clusters | Reject; all 15,228 particles belong to 62,770 tetrahedra |

For the T-shirt, two guarded 180-frame windows averaged 49.863 ms/frame and
two otherwise identical unguarded windows averaged 48.916 ms/frame. The new
transactional checks therefore cost about 1.9% in this scene; they are a safety
mechanism, not an additional speedup. The guarded configuration still retains
roughly the previously measured 30% advantage over the 20-sweep reference.

The mixed soft/rigid-cube candidate built 255 clusters only for the surface bag
while leaving the 720 tetrahedra on ordinary VBD. Against a 12-sweep reference
at frame 240, plain 8 sweeps had 3.114 mm all-particle RMS error and the coarse
candidate had 3.775 mm, 21.2% worse. Its bag-only RMS likewise increased from
3.293 to 3.993 mm, while the tetrahedral cube changed by less than 1%. Do not
enable multilevel for this example merely to reduce its sweep count.

The Armadillo probe confirmed `tetrahedra_present` as the automatic rejection
reason and produced zero coarse clusters. The example remains finite through
the measured 240-frame window. These two negative results validate keeping
tetrahedral and mixed volumetric scenes outside the automatic policy.

### 2026-09-04: reject rigid-soft color masks in the optimized soft backend

**Hypothesis.** Port the complete `vbd/` backend's per-contact particle-color
mask into `vbd_soft/`, building one `uint32` mask per contact refresh and using
it to reject contacts before loading their geometry and material data during
each color sweep.

**Isolated result.** On an RTX 5090 D v2, a synthetic CUDA Graph containing
5,886 particles, 9,880 mixed point/edge/face contacts, eight colors, and 12
VBD iterations improved the contact stage by 8.60% on the ordinary traversal
and 10.96% on the overallocated persistent traversal. Masked and unmasked
force/Hessian tests matched on CPU and CUDA.

**Real-example result.** Fixed-frame headless runs alternated the unmodified
commit `37ad664c` and the mask implementation. Medians were 24.25 versus 24.50
FPS for the 360-frame gear crusher (-1.0%), 19.4 versus 19.2 FPS for the
180-frame nonwoven-bag table drop (+1.0%), and 21.7 versus 21.8 FPS for the
180-frame kinematic T-shirt fold (-0.5%). Individual runs varied by roughly
the same magnitude as the measured differences, so none established the 5%
end-to-end threshold. The dynamic T-shirt run was stopped after these results
already rejected the broad enablement rule.

**Decision.** Revert the mask storage, construction kernel, solver wiring,
tests, and changelog fragment. The isolated workload overrepresented
edge/face contacts and contact-stage share; current `vbd_soft/` examples use
mostly point contacts, where rebuilding masks adds work without removing
enough whole-frame cost. Reconsider only with a real full-surface
`vbd_soft/` workload and a runtime policy based on measured contact mix, not
particle-color count alone.

### 2026-09-04: retain a contact-aware surface multilevel correction

**Problem.** Particle VBD communicates information only through successive
colored vertex sweeps. Stiff cloth therefore needs many sweeps before a load or
contact response reaches distant vertices. Simply reducing the iteration count
improves throughput but changes large-scale motion even when local stretch
remains small.

The retained prototype builds a fixed rest-topology hierarchy at solver
construction. Breadth-first clusters contain eight movable surface particles
by default and never cross particle worlds. Zero-mass vertices remain coarse
anchors. Vertices belonging to tetrahedra are deliberately excluded: the
current piecewise-constant translation basis cannot represent volumetric
rotation and affine deformation accurately. Spring endpoints participate in
the topology graph.

After the final ordinary VBD sweep, contacts are evaluated once at the final
particle positions. A local elasticity solve supplies the fine residual and
block-diagonal Hessian. Restriction forms cluster residual, mass, elastic
stiffness, and contact-diagonal terms. Eight fixed PCG steps solve a scalar
mass-plus-topology-Laplacian approximation for three displacement components,
then prolong one relaxed translation per cluster. The default relaxation is
0.1 and each fine correction is capped at 5% of particle radius. Existing DAT
runs on the combined fine and coarse displacement.

The coarse PCG is one persistent 256-thread CUDA block. Each lane processes a
grid-stride subset of clusters and tile reductions provide its block-wide
reductions and barriers. This produced exactly the same coarse result as the
original 42-launch implementation in a 71-cluster comparison (`max_abs=0`),
while avoiding the small-kernel scheduling cost. Storage and launches are
fixed at construction, so CUDA Graph topology remains fixed. The feature is
off by default and falls back to ordinary VBD on CPU, `requires_grad`, and
deterministic runs.

**Rejected intermediate variants.** Applying four contact-blind corrections
per substep caused late self-contact growth in the complete 900-frame T-shirt
trajectory: even a capped conservative variant fell to 5.62 FPS. A single
contact-blind correction completed at 13.3 FPS but differed from the 20-sweep
reference by 38.7 mm RMS after 300 frames, versus 19.6 mm for ordinary
12-sweep VBD. Re-evaluating rigid-soft and self-contact force/Hessian at the
final iterate, treating contact stiffness as a coarse diagonal anchor, and
applying only one correction removed this failure. These failed schedules are
not exposed as options.

**Measurements.** All measurements used an NVIDIA GeForce RTX 5060 Ti, CUDA
Graphs, and the same scripted trajectory within each comparison. Viewer-null
benchmarks discard the first three frames. State-error comparisons ran the
variants consecutively in one process and compare particle positions at the
same frame against the original higher-sweep configuration.

| Scene | Topology and settings | Reference | Retained configuration | Performance result | State result |
| --- | --- | --- | --- | --- | --- |
| W1 T-shirt fold final00 | 6,436 particles, 12,736 triangles, 19,174 edges, 88 shapes, 10 substeps | 20 sweeps | 12 sweeps + one coarse correction | 300-frame wall time 31.579 to 22.001 s, 30.3% lower | At frame 300, RMS error was 11.46 mm; ordinary 12 sweeps was 19.56 mm. Mean triangle-edge error was 0.251 mm versus 0.418 mm. |
| W1 plastic-bag rod final00 | 5,886 particles, 11,512 triangles, 17,399 edges, 90 shapes, 6 substeps | 12 sweeps | 8 sweeps + one coarse correction | 300-frame benchmark 17.3 to 20.1 FPS, 16.2% faster | At frame 300, RMS error was 20.60 mm; ordinary 8 sweeps was 35.30 mm. Mean triangle-edge error was 0.160 mm versus 0.302 mm. |
| Armadillo gear-crusher final00 | 15,228 tet vertices, 62,770 tetrahedra, 120 frames | 10 sweeps | Experimental 8 sweeps + coarse translation | 5.12 to 5.36 FPS before the tet gate | RMS error regressed from 2.11 to 25.25 mm, so tet vertices are excluded and this scene retains its original solver path. |

**Correctness evidence.** Dedicated tests verify fixed-vertex anchors,
tetrahedral exclusion, improved long-range response against a 100-sweep
cantilever reference, deterministic fallback, and CUDA Graph capture/replay in
both private VBD implementations. All 12 existing MJVBDV2 contact-optimization
tests also pass. The two validated final00 examples enable the feature while
reducing their ordinary sweep counts; unmeasured examples keep their existing
configuration.

**Decision.** Retain as an opt-in surface correction and enable it only in the
two measured final00 examples. Do not claim a volumetric multigrid method: a
tet-capable coarse basis needs at least rotational/affine modes and a matching
Galerkin operator. Keep ordinary VBD as the default until more cloth and shell
scenes establish a broadly safe automatic iteration policy.

### 2026-09-04: extend the coarse space to tetrahedral meshes

**Problem.** The translation-only cluster basis used by the first retained
surface prototype cannot represent a tetrahedral object's large-scale
rotation. On the Armadillo scene, an early translation-only tet experiment
reduced the 10-sweep frame cost slightly but increased the particle RMS error
to 25.25 mm. The surface implementation therefore gated out tet particles.

The retained extension assigns every movable tet particle to a fixed
topological cluster and uses six coarse coordinates per tet cluster:

```text
delta_x_i = translation_c + rotation_c cross (x_i - centroid_c)
```

The centroid and rotational basis are evaluated at the current particle
positions. Restriction applies the transpose of this basis to the local VBD
residual. The coarse diagonal contains the Galerkin projection of particle
mass and contact Hessians. For every tetrahedron and directed cluster block,
the implementation projects the same positive-semidefinite Neo-Hookean
deformation-gradient Hessian used by fine VBD. Objective metric damping is
projected as well; terms whose spectral scale is below one part per million of
the elastic block use the elastic-only fast path. Surface clusters retain only
their three translation modes, and construction prevents a cluster from
mixing surface-only and tet particles.

The six-DOF system uses a Cholesky block-Jacobi preconditioner and a fixed-step
persistent PCG kernel. Its reported diagnostic is the true Euclidean residual
ratio `||r_k|| / ||r_0||`. Fixed particles remain coarse boundary conditions,
the final correction retains the existing relaxation and particle-radius cap,
and DAT still truncates the combined fine-plus-coarse displacement. All
buffers and block topology are fixed at construction, preserving CUDA Graph
topology. CPU, `requires_grad`, and deterministic modes still use ordinary
VBD. The solver-wide option remains disabled by default.

**Measurements.** Tests used an NVIDIA GeForce RTX 5060 Ti, Warp
1.17.0.dev20260807, CUDA Toolkit 12.9, Driver 13.3, viewer-null CUDA Graphs,
and discarded the first three frame timings. Each comparison used the same
scripted input and substep count.

| Scene | Reference | Six-DOF configuration | Performance | Accuracy and function |
| --- | --- | --- | --- | --- |
| Armadillo gear-crusher final00, 15,228 particles and 62,770 tets, 240 frames | 10 fine sweeps, 201.678 ms mean | 6 fine sweeps, 32-particle clusters, 4 coarse PCG steps, 0.025 relaxation | 193.746 ms mean, 3.93% faster; p95 224.953 to 208.883 ms | Against 10 sweeps, particle RMS 3.535 mm, max 15.668 mm, tet-edge MAE 0.0519 mm. Plain 6 sweeps had 6.826 mm RMS, 32.914 mm max, and 0.0646 mm edge MAE. The complete 1,500-frame test passed physical grasp, lift, carry, release, and gear-contact assertions. |
| Dexforce soft-then-rigid cube into cloth bag final00, 1,900 frames | 12 fine sweeps, 40.338 ms mean | 8 fine sweeps, 32-particle clusters, 4 coarse PCG steps, 0.025 relaxation | 36.590 ms mean, 9.29% faster | Before soft-cube release, bag RMS against 12 sweeps was 0.75--2.40 mm versus 3.27--4.48 mm for plain 8; tet-cube differences remained at or below 1.74 mm. The complete placement test passed. |

The mixed scene is not run-to-run deterministic. Two identical accelerated
480-frame runs differed by at most 1.1 mm before soft release but diverged by
28--37 mm after later rigid-contact phases. Consequently, post-release pointwise
RMS is not used as accuracy evidence; phase-local pre-release error and the
scene's physical completion assertions are the meaningful checks.

**Correctness evidence.** Unit tests cover tet cluster assignment, explicit
separation of connected surface and tet cluster types, six-DOF rotation
response, finite coarse residuals, CUDA Graph replay in both private VBD
implementations, deterministic and differentiable fallbacks, and the existing
surface long-range propagation result. The mixed and Armadillo final00 demos
enable their measured configurations. Unmeasured scenes retain the original
solver because the global default remains off.

**Decision.** Retain the six-DOF extension. It resolves the tet accuracy
failure that forced the original gate and provides a measurable benefit in a
mixed cloth/tet workload. It is not an affine coarse space: cluster-scale
shear and stretch still require fine sweeps, so larger relaxation or radius
caps remain unsupported without separate validation.

### 2026-09-04: retain bounded temporal mesh-SDF face warm starts

**Hypothesis.** Full-surface face contacts solve the same 24-step
Frank--Wolfe problem, with a 16-step SDF line search at every step, on every
collision call. Consecutive substeps usually leave a retained mesh-SDF face
near its previous minimizer. Reusing that barycentric point after a short
warm-start refinement should remove most texture-SDF queries without changing
the candidate set, contact threshold, output stream, or CUDA Graph topology.

The retained path stores one `vec3` barycentric coordinate and one byte of
state per shape-major face pair. A cached point receives two Frank--Wolfe
refinement steps and is accepted only when its evaluated SDF is at least
0.5 mm inside the contact threshold and its stationarity gap is at most
0.25 mm. A failure always executes the original 24-by-16 optimizer; the cache
never rejects a candidate. At most three consecutive calls can reuse a result
before a mandatory full refresh. AABB-inactive and disabled-shape pairs clear
their state.

The optimization is private to MJVBDV2's CUDA full-surface pipeline and only
handles non-analytic texture SDFs. CPU, `requires_grad`, shared Newton
collision, analytic primitives, and every non-full contact backend retain the
old path. Cache storage is fixed at construction and capped at 64 MiB; models
above the cap fall back completely. No count, allocation, or launch is added
during Graph replay.

**Isolated performance.** The representative Armadillo crusher scene was
advanced for 240 frames, then both variants captured collision-only Graphs at
the same frozen state. The NVIDIA GeForce RTX 5060 Ti 8 GiB used Warp
1.17.0.dev20260807, CUDA Toolkit 12.9, and Driver 13.3. Each result is the
median of seven samples of 80 warmed Graph replays.

| Face optimizer | Collision Graph | Change |
| --- | ---: | ---: |
| Original 24-by-16 solve on every active pair | 3.652919 ms | baseline |
| Bounded two-step temporal warm start | 2.575027 ms | 41.86% faster |

Both variants emitted 27,444 contacts with identical sorted
`(shape, particle, indices)` keys. Across the complete stream, the 99th
percentile differences in shape-surface position and normal were respectively
`8.83e-9 m` and `3.22e-7`; the means were `4.37e-6 m` and `5.27e-4`. Rare
texture-SDF local-minimum switches reached `5.11e-3 m` and `0.474`, so bounded
reuse and periodic exact refresh are part of the retained correctness guard,
not optional tuning.

**End-to-end cross-check.** The 240-frame W1 plastic-bag/rod full-contact demo
measured 237 post-warm-up frames in separate processes. Cache enabled and
disabled both rounded to 17.5 FPS (13.55 and 13.57 seconds). Both configurations
also completed their full null-viewer functional test. This scene establishes
no speedup, but it rules out a measurable regression at its current topology.

The focused MJVBDV2 optimization suite passes all 12 CPU/CUDA tests. New
coverage compares shared and private mesh-SDF contact keys across a static
repeat, a particle displacement, and complete AABB separation; it also checks
that analytic primitives do not use the cache and inactive pairs invalidate
state. The representative Armadillo run remained finite through frame 240.

**Decision.** Retain the guarded cache. Its isolated collision gain is large
enough even though collision is only part of the frame, and the neutral bag
result is preferable to imposing a full-frame percentage gate on a local
hotspot. Do not enable it for analytic, differentiable, CPU, or over-capacity
configurations without separate numerical and performance evidence.

### 2026-09-03: reject particle-centric self-contact gather and tile fusion

**Hypothesis.** Preserve the existing per-vertex and per-edge detector rows,
then build a device linked list from every retained directed EE row to its two
owner-edge vertices and from every retained VT row to its four incident
vertices. Each graph color could gather only the rows incident to its particles
instead of launching four threads per primitive and filtering the complete row
set. Gathering directly inside `solve_surface_elasticity_tile` could also remove
one force/Hessian launch and the intermediate self-contact writes.

The prototype kept directed EE ownership, asymmetric filters, current-position
contact reevaluation, material selection, color order, overflow clamping, and
the independent planar DAT path. The adjacency used deterministic slots backed
by the existing fixed row capacities, a device-side head array, and atomic
linked-list construction after each detector refresh. It was gated to CUDA,
non-gradient, nondeterministic, surface-only tiled solves and capped adjacency
storage at 64 MiB; all other configurations retained the original scatter.

**Numerical check.** A frozen two-layer 12-by-12 cloth used one detector result
for both implementations. After accumulating every color, gathered force
differed from scatter by at most `9.281559e-5` against a maximum force norm of
`2.0054703e2`. Hessian entries differed by at most `0.625` against a maximum
absolute entry of `1.5345574e6`. These relative differences are consistent
with the changed CUDA summation order. A one-step 24-by-24 two-layer smoke test
compiled the fused Warp kernel, reported active contact, and produced finite
particle positions.

**End-to-end Go/No-Go.** The representative 6,436-particle T-shirt example ran
with its normal ten substeps, 20 iterations, nine colors, CUDA Graphs, null
viewer, and 120 requested frames on an NVIDIA GeForce RTX 5060 Ti 8 GiB with
Warp 1.17.0.dev20260807, CUDA Toolkit 12.9, and Driver 13.3. The benchmark
measured 117 post-warm-up frames. Runs were separate processes and are therefore
not precision performance evidence, but the regressions are much larger than
the retain threshold.

| Self-contact force path | Wall time | Rate | Change from legacy |
| --- | ---: | ---: | ---: |
| Legacy primitive-row scatter | 11.62 s / 117 frames | 10.1 FPS | baseline |
| Particle gather fused into 16-thread elasticity tile | 15.81 s / 117 frames | 7.40 FPS | 36.1% slower per frame |
| Separate particle gather before the existing tile | 21.70 s / 117 frames | 5.39 FPS | 86.7% slower per frame |

The fused kernel serializes each particle's contacts on tile lane zero while
the other lanes wait, and the added contact code increases register pressure
for the elasticity kernel. The separate gather avoids that coupling but still
recomputes a VT narrow phase once per incident particle rather than once per
distinct active color; its linked-list traversal is also divergent and
noncoalesced. Those costs dominate the saved color checks, atomics, and
primitive-row traversal in this single-scene cloth workload.

**Decision.** Reject and completely revert both forms. A future particle-
centric attempt needs a compact contiguous CSR plus a contact representation
that shares one narrow-phase evaluation across incident vertices; rebuilding
the same linked-list gather or embedding the current large evaluators in the
elasticity tile is not justified. DAT remains unchanged.

### 2026-09-03: traverse sparse rigid-soft contacts with persistent workers

**Problem.** During CUDA Graph capture, `vbd_soft/` cannot read the
device-side soft-contact count to reduce a launch dimension. Its particle-side
rigid-soft force/Hessian kernel therefore launched one thread for every
allocated contact slot once per particle color and VBD iteration. In the
realtime T-shirt fold this meant 289,620 threads per launch for only
3,177--5,730 active records (1.10%--1.98% occupancy), repeated for nine colors,
20 iterations, and ten substeps per displayed frame.

**Retained implementation.** The legacy per-contact body is shared by the old
capacity-sized kernel and a CUDA-only active-prefix kernel. The latter launches
4,096 fixed workers and grid-strides over
`min(soft_contact_count[0], soft_contact_max)`. It does not compact or reorder
the contact stream, add host readback, allocate per-frame storage, or change
CUDA Graph topology. Contact evaluation, per-color Gauss--Seidel order,
barycentric distribution, atomics, material state, and overflow clamping are
unchanged.

Dispatch is deliberately conservative. The persistent traversal is selected
only while capturing a CUDA Graph, only for non-differentiable models in
`NOT_GUARANTEED` deterministic mode, and only when the captured capacity is at
least eight times the 4,096-worker grid (32,768 records). Eager execution keeps
the existing host-count-sized launch. CPU, `requires_grad`, deterministic, and
smaller-capacity Graphs keep the legacy kernel. The complete `vbd/` backend is
also unchanged; this experiment only covers the `vbd_soft/` scatter path that
was measured.

**Measurements.** Tests used an NVIDIA GeForce RTX 5060 Ti 8 GiB with Warp
1.17.0.dev20260807, CUDA Toolkit 12.9, and Driver 13.3. The representative
T-shirt scene had 6,436 particles, 12,736 triangles, 19,174 edges, nine particle
colors, ten substeps per displayed frame, and 20 VBD iterations per substep.
Both variants ran in one process with separately constructed examples and
captured Graphs. The candidate ran first; both variants passed the example's
final-state checks. The 120-frame comparison discarded three warm-up frames;
the 360-frame comparison did the same and covers the full accelerated script.

| T-shirt A/B window | Active-prefix wall time | Legacy wall time | Change |
| --- | ---: | ---: | ---: |
| 120 frames | 96.829577 ms/frame | 100.968758 ms/frame | 4.10% faster |
| 360 frames | 110.060740 ms/frame | 113.929294 ms/frame | 3.40% faster |

An isolated frozen-frame Graph measured one complete nine-color contact-force
pass. The active count was 3,783 and the capacity was 289,620. Outputs from all
tested worker widths were exactly equal to the legacy force and Hessian arrays.

| Frozen nine-color pass | GPU time | Change |
| --- | ---: | ---: |
| Legacy capacity scan | 0.057044 ms | baseline |
| 4,096-worker active prefix | 0.030767 ms | 46.07% faster |

Earlier full-trajectory forward and reverse-order A/B runs with a wider
persistent grid measured 3.74%--4.69% end-to-end gains, supporting that the
same-process result is not solely an ordering artifact.

**Generality check.** The 361-particle tablecloth example allocates 31,768
soft-contact records, just below the gate, so both nominal variants dispatched
the exact legacy kernel. Its 90-frame runs measured 17.273018 and 16.915756
ms/frame and both passed `test_final()`; the difference is treated as run-order
noise, not as an optimization result. The independent 300-frame cloth-twist
example also passed in CUDA Graph test mode. This demonstrates that ordinary
small scenes do not pay for the T-shirt specialization.

**Correctness evidence.** Focused CUDA tests compare the new kernel with the
legacy scatter for empty, partial, full, and overflow-clamped active prefixes.
The existing MJVBDV2 contact-optimization suite passes all eleven tests. Dispatch
tests cover the 32,768-record threshold and verify that eager, deterministic,
and differentiable paths remain on the legacy implementation. The T-shirt and
tablecloth final-state checks and the cloth-twist Graph regression passed.

At frame 360, candidate-versus-legacy particle displacement had a 5.02 mm
mean, 23.68 mm 95th percentile, and 92.53 mm maximum; the cloth centroid and
AABB extrema differed by 2.46 mm and at most 8.85 mm. Two independent legacy
runs already differed by 7.34 mm mean, 30.64 mm 95th percentile, and 80.89 mm
maximum, with a 2.22 mm centroid difference and up to 20.31 mm AABB difference.
The candidate drift is therefore within the existing run-to-run variation of
this `NOT_GUARANTEED` atomic simulation rather than evidence of a changed fold.

**Decision.** Retain with the automatic gate. The representative Graph gain is
large enough to keep, while the fallback preserves solver coverage and avoids
the small-scene regression risk observed during broader evaluation.

### 2026-08-28: reject canonical-owner detection with directed-row compatibility

**Hypothesis.** Let only the smaller edge ID evaluate each unordered BVH
candidate, query both directional topology/external filter rows, perform the
current and optional rest closest-point tests once, then emit whichever
directed sides are enabled. Keep the existing directed per-edge buffers so
force/Hessian, DAT, conservative bounds, overflow handling, and CUDA Graph
consumers remain unchanged during a detector-only Go/No-Go test.

**Candidate work.** An uncommitted V2-private CUDA kernel skipped candidates
whose target ID was not greater than the source ID. For a retained canonical
pair it atomically updated the count and minimum distance of each enabled
source row, then wrote `(source, target)` to the old fixed-capacity row. The
reverse row was allowed independently, preserving asymmetric external
filters. CPU, differentiable, and deterministic solver configurations kept
the shared detector. No force/Hessian or DAT kernel was changed.

The compatibility layout requires an initialization pass because row counts
and minimum distances receive cross-thread atomics. It also turns the old
single-owner sequential row writes into two potentially scattered atomic row
updates per canonical pair. These costs are part of this detector-only design,
not part of a future direct compact-stream implementation.

**Controlled A/B.** Tests used an NVIDIA GeForce RTX 5060 Ti with CUDA 12.9
and Warp 1.17.0.dev20260807. The frozen topology contained two disconnected,
quarter-cell-offset 28-by-28 cloth grids separated by 0.01 m: 1,682 particles,
4,816 edges, n-ring filtering threshold 2, a 0.04 m query margin, and 256
directed row slots per edge. Each kernel was captured separately against the
same positions and BVH, warmed for 300 replays, then measured with the median
of seven samples of 500 synchronized Graph replays.

| EE detector case | Directed-row baseline | Canonical owner | Change |
| --- | ---: | ---: | ---: |
| No rest exclusion; 439,800 directed rows | 0.946510 ms | 1.387774 ms | 46.62% slower |
| 0.03 m rest exclusion; 188,610 directed rows | 1.144414 ms | 1.443326 ms | 26.12% slower |

**Correctness evidence.** In both cases, sorted directed pair keys, every
per-edge count, every minimum distance, and all overflow flags matched the
shared detector exactly. A smaller 8-by-8 compile/smoke case also matched all
four outputs, but regressed 36.4%.

**Decision.** Rejected and reverted before commit. The detector-only gate was
25% faster; instead both representative variants regressed by more than 25%,
so do not extend this compatibility design into force/Hessian or DAT. This
does not measure a detector that directly emits a compact canonical stream:
such a design must remove the legacy row initialization, dual row counts, and
directed row writes, and must be evaluated together with all downstream
consumers rather than reintroducing a compatibility scatter.

### 2026-08-28: reject static Morton self-contact query scheduling

**Hypothesis.** Warp already constructs CUDA BVHs with LBVH, but the VT and EE
detector kernels still launch source vertices and edges in original ID order.
Sort source queries once by `(world, rest-space Morton code)` so neighboring
threads traverse more similar BVH nodes, while continuing to write collision
results into the original source CSR rows.

**Candidate work.** An uncommitted V2-private module generated 10-bit-per-axis
Morton permutations for rest vertices and rest edge midpoints. Private copies
of the shared VT and EE detector kernels changed only
`source = query_order[tid]`; topology filters, current and rest distance
tests, row capacities, output IDs, minimum distances, and overflow behavior
were otherwise unchanged. Both private VBD implementations selected the new
kernels only for ordinary CUDA execution. CPU, differentiable, and
deterministic configurations retained the shared kernels. No dynamic sort,
buffer compression, host readback, or additional Graph kernel was added.

**Controlled A/B.** Tests used an NVIDIA GeForce RTX 5060 Ti with CUDA 12.9
and Warp 1.17.0.dev20260807. Each pair of kernels was captured against the
same detector, BVHs, positions, buffers, and capacities; only the Python-side
kernel selection changed before capture. Each variant warmed 300 replays and
reported seven samples of 500 synchronized Graph replays in alternating
order.

The dense case used two disconnected 28-by-28-cell cloth grids separated by
0.01 m: 1,682 particles, 3,136 triangles, 4,816 edges, 79,098 retained VT
rows, 428,316 retained directed EE rows, and no overflow.

| Dense two-layer detection | Original order | Morton order | Change |
| --- | ---: | ---: | ---: |
| VT-only median | 0.454972 ms | 0.471008 ms | 3.5% slower |
| EE-only median | 0.971624 ms | 1.018857 ms | 4.9% slower |
| Combined, order A | 1.419446 ms | 1.484442 ms | 4.6% slower |
| Combined, reverse order | 1.416001 ms | 1.481688 ms | 4.6% slower |

A second frozen state used the standard supermarket bag after 100 captured
frames. It contained only 152 VT and 1,442 directed EE rows with no overflow.
Original-order medians were 0.606674 and 0.613190 ms in the two run orders;
Morton medians were 0.596230 and 0.599069 ms. This is only 1.7%--2.3% faster,
or about 0.01--0.014 ms per complete detection, far below the isolated gate
and below 0.1% of the approximately 26 ms representative frame.

**Related layout checks.** Static Morton sorting inside the four particle
color groups of a 96-by-96 cloth changed a ten-iteration frozen Graph from
1.476465/1.477377 ms to 1.487719/1.487336 ms, about 0.7% slower. Changing the
dense detector BVHs from the default LBVH to SAH was about 1.5% slower.
LBVH leaf sizes 1, 2, 4, and 8 measured 1.396, 2.708, 2.250, and 1.961 ms;
retain leaf size 1.

**Correctness evidence.** Dense-cloth and supermarket-bag comparisons matched
the raw VT and EE row buffers, counts, minimum distances, and overflow flags
exactly. The focused CPU self-contact activity test passed on the unchanged
fallback. The candidate compiled and replayed captured Graphs in both
`vbd_soft/` and complete `vbd/`.

**Decision.** Rejected and reverted before commit. Morton ordering reduces
the spatial spread of queries, but it turns source positions, row offsets,
counts, filter CSR reads, and output rows from contiguous accesses into
indirect scattered accesses. That cost dominates in the dense workload, while
the sparse real-scene gain is immaterial. Do not reorder cloth primitives,
BVH inputs, color groups, or detector sources by Morton without a different
memory layout that preserves coalesced source-row access.

### 2026-08-28: reject directed EE source-color row gating

**Hypothesis.** A directed edge-edge force/Hessian row writes only the two
vertices of its source edge. Move the source indices and particle-color loads
outside the variable-length row loop, and avoid reading the row offsets,
count, and target edges when neither source endpoint belongs to the current
Gauss--Seidel color. The candidate changed both private VBD implementations
without changing detector records, asymmetric filters, contact evaluation,
DAT, material selection, launch dimensions, or CUDA Graph topology.

**Candidate work.** The uncommitted prototype added only the outer source-color
gate to `accumulate_self_contact_force_and_hessian`. The existing target-edge
order, per-color current-position force evaluation, directed source-side
atomics, and VT traversal were unchanged. No stream, counter, buffer, build
step, or runtime compression was added.

**Controlled A/B.** Measurements used an NVIDIA GeForce RTX 5060 Ti with CUDA
12.9 and Warp 1.17.0.dev20260807. A frozen `vbd_soft/` benchmark used two
disconnected 28-by-28-cell cloth grids separated by 0.01 m: 1,682 particles,
3,136 triangles, 4,816 edges, four particle colors, one VBD iteration, a
0.02 m contact radius, and a 0.04 m margin. Each separate process warmed 1,000
captured replays, then reported the median of seven samples of 500 replays.

| Frozen `vbd_soft/` step | Baseline | Source-color gate | Change |
| --- | ---: | ---: | ---: |
| Median Graph time | 2.941983 ms | 2.929634 ms | 0.42% faster |

The complete `vbd/` representative check ran the standard supermarket-bag
CUDA Graph in separate processes with the built-in three-frame warm-up and a
10-second null-viewer sample at default process priority. Baseline completed
381 frames at 38.1 FPS; the candidate completed 377 frames at 37.7 FPS,
approximately 1.0% slower by frame time. Supporting event timing over one
eager frame measured the 576 force/Hessian launches at 20.311520 ms total for
the baseline and 21.151072 ms for the candidate, a 4.13% regression. The
event-wrapped eager result is not used alone for the decision, but it agrees
with the representative Graph result.

**Correctness evidence.** The focused CPU self-contact activity test passed.
The changed kernels also compiled and executed on CUDA in both `vbd_soft/`
and complete `vbd/`, including captured Graph replay. Because the experiment
was rejected, no new permanent regression test was added.

**Decision.** Rejected and reverted before commit. The only controlled gain
was 0.42%, within the observed clock and run-order drift, while the
representative full-frame and event-timed results regressed. CUDA compilation
can already hoist invariant source data, and avoiding the remaining
color-irrelevant row metadata does not remove contact evaluation, VT,
detection, or DAT work. Revisit only with construction-time per-color source
edge lists that materially reduce the launch domain, and require the usual
15% isolated and 5% representative gates.

### 2026-08-28: reject rest-shape exclusion CSR precomputation

**Hypothesis.** Rest positions, rest-shape exclusion radius, topology, and
external filter maps are fixed after solver construction. Build exact
vertex-triangle (VT) and directed edge-edge (EE) rest-near pair CSR rows once,
union them with the detector's final static filters, and disable the per-frame
rest-pose closest-point calculations. The candidate preserved the strict
`distance_rest < radius` predicate, BVH world grouping, EE directionality,
topological and asymmetric external filters, current-position force order,
and CUDA Graph topology.

**Candidate work.** A V2-private module used the construction-time rest BVHs
to count, scan, and fill exact VT and EE rows on CUDA. It merged and sorted
the entries with the existing CSR because the detector binary-searches each
row. A combined eight-million-entry cap left the existing filters and the
runtime reference-distance path untouched when exceeded. CPU, `requires_grad`,
and deterministic configurations also retained the legacy path. CUDA tests
compared sorted detector rows, counters, minimum distances, overflow flags,
one-step DAT values, final particle positions and velocities, asymmetric
external filters, cap fallback, and Graph replay.

**Controlled A/B.** The candidate was an uncommitted experiment measured on
an NVIDIA GeForce RTX 5060 Ti (CUDA 12.9, Warp 1.17.0.dev20260807). The
isolated CUDA Graph benchmark used two disconnected 28-by-28-cell cloth grids
at rest-plane separation 0.01 m: 1,682 particles, 3,136 triangles, 4,816
edges, a 0.03 m rest exclusion, a 0.05 m detection margin, and n-ring filter
threshold 2. Both variants warmed 30 replays, then ran 300 synchronized
VT+EE-detection Graph replays. The legacy filter held 27,112 VT and 97,056 EE
entries; the candidate grew those rows to 190,236 and 918,016 entries.

| Measurement | Runtime rest filter | Static CSR | Change |
| --- | ---: | ---: | ---: |
| VT + EE self-detection | 6.0853 ms/pass | 7.2488 ms/pass | 19.1% slower |

The proposal removes rest closest-point arithmetic, but it replaces it with
large, divergent global-memory binary searches. In this rest-near workload
the additional CSR traffic dominates the saved geometry work.

As an end-to-end representative check, separate-process null-viewer CUDA
Graph runs of `mjvbd_v2_supermarket_plastic_bag` used the standard model,
three-frame warm-up, and a 10.01-second sample at default process priority. The
candidate completed 422 frames (42.2 FPS); a same-worktree baseline that
disabled CSR installation completed 415 (41.5 FPS). This is only a 1.7%
frame-time reduction, below the 5% gate and near desktop timing variance.

**Decision.** Rejected and reverted before commit. Do not restore a general
rest-near CSR with per-row binary search. Reconsider only with a more compact
membership representation that does not expand dense rest-near rows, and only
after it clears at least 15% isolated self-detection and 5% representative
frame-time improvement.

### 2026-08-28: reject post-detection canonical edge-edge pairs

**Hypothesis.** Merge the directed edge-edge (EE) self-contact rows into one
canonical `(min(edge_a, edge_b), max(edge_a, edge_b))` pair and retain two side
bits. Force/Hessian and proxy harvest would still process every enabled side,
preserving asymmetric external topology filtering and graph-color
Gauss--Seidel semantics. Planar DAT, which constrains all four vertices, would
run once per pair instead of once per directed row.

**Candidate work.** After the existing EE detector wrote its directed CSR, a
new device kernel scanned it to build a fixed-capacity canonical-pair stream.
It reverse-searched the CSR to decide whether each side was present. Both
private VBD implementations consumed the new stream; the directed CSR was
retained for conservative bounds and contact activity. CPU/CUDA regressions
covered symmetric and asymmetric filters, force/Hessian equality against the
legacy directed reference, DAT, proxy harvest, and CUDA Graph capture.

**Controlled A/B.** The exact parent was `4c3cb08f`; the candidate was an
uncommitted experiment and was measured before reverting. Tests ran eagerly
on an NVIDIA GeForce RTX 5060 Ti (CUDA 12.9, Warp 1.17.0.dev20260807), with
one VBD iteration and self-contact radius/margin of 0.02/0.04 m. The topology
was two disconnected, quarter-cell-offset 28-by-28 cloth grids: 1,682
particles, 4,816 edges, one particle color, and 163,224 stored directed EE
rows. The candidate produced 90,576 canonical pairs, a 44.5% reduction rather
than an ideal half because the input contained one-sided rows. Each sample
synchronized the GPU; solver input positions were reset before full-step
samples. Complete `vbd/` used 20 warm-ups and 100 samples; `vbd_soft/` used
10 warm-ups and 50 samples. Values below are medians.

| Measurement | Complete parent | Complete candidate | Change |
| --- | ---: | ---: | ---: |
| EE detection | 0.677500 ms | 0.876250 ms | 29.3% slower |
| EE force/Hessian | 0.085700 ms | 0.101150 ms | 18.0% slower |
| EE DAT | 0.181700 ms | 0.154700 ms | 14.9% faster |
| Full self-contact step | 2.123400 ms | 2.623400 ms | 23.5% slower |

| Measurement | Soft parent | Soft candidate | Change |
| --- | ---: | ---: | ---: |
| EE detection | 0.729350 ms | 0.866850 ms | 18.9% slower |
| EE force/Hessian | 0.086950 ms | 0.094750 ms | 9.0% slower |
| EE DAT | 0.084450 ms | 0.086800 ms | 2.8% slower |
| Full self-contact step | 2.234900 ms | 2.627100 ms | 17.5% slower |

The absolute isolated-kernel numbers are small and have normal desktop-GPU
variance, but both end-to-end measurements regress materially. DAT's reduced
work is insufficient to offset canonical-stream construction. The reverse CSR
lookups add detector work, while force/Hessian cannot remove directional
evaluations without changing the preserved side semantics.

**Decision.** Rejected and reverted before commit. Do not restore this
post-processing canonical stream. Revisit only if broad phase can emit a
unique pair directly, without first materializing and reverse-searching the
directed CSR, and if a new controlled A/B proves end-to-end benefit.

### 2026-08-28: reject row-buffer active contact streams

**Hypothesis.** Keep the detector's fixed per-vertex VT and per-edge directed
EE rows, then append each retained row to fixed-capacity device-side streams.
Force/Hessian and planar DAT would use fixed persistent workers over the
counts, avoiding the four-thread primitive-row scans. Directed EE entries were
intentionally not canonicalized: asymmetric filters, both EE force sides, and
the existing Gauss--Seidel color semantics remain unchanged.

**Candidate work.** Both private VBD detectors allocated streams sized exactly
to their total retained row capacities and atomically compressed the clamped
VT/EE prefixes after detection. The CUDA/non-gradient/nondeterministic solver
path used those streams for force/Hessian and DAT, with a static
`2 * SM * 128` worker bound. CPU, `requires_grad`, and deterministic modes
kept their original row kernels. The detector's existing resize flags remained
authoritative because stream capacity cannot be exceeded by clamped rows.

**Correctness.** CUDA tests verified that sorted VT and directed EE streams
equal every retained detector row, that one-step positions match the legacy
row path to `2e-5`, and that complete and soft solvers capture and replay one
CUDA Graph. CPU, autodiff, and deterministic construction were verified to
select the legacy path.

**Controlled A/B.** The experiment ran on an NVIDIA GeForce RTX 5060 Ti
(CUDA 12.9, Warp 1.17.0.dev20260807). The topology was one 24-by-24-cell
cloth grid (625 particles and 1,152 triangles), one VBD iteration, radius
0.02 m, margin 0.03 m, and default VT/EE row capacities of 32/64. At the
measured state, streams held roughly 9.7k--9.9k VT and 60.5k--61.5k directed
EE entries. The baseline disabled both stream consumption *and* detector
compression; it otherwise used the same warmed worktree. CUDA Graph samples
used 50 warm-ups, 11 synchronized samples, and 200 replays per sample.

| Path | Baseline-first Graph result | Stream-first Graph result | Eager full-step result |
| --- | ---: | ---: | ---: |
| `vbd_soft/` | 1.913282 -> 1.858652 ms (2.86% faster) | 1.991650 -> 1.972545 ms (0.96% faster) | 2.025850 -> 2.046293 ms (1.01% slower) |
| Complete `vbd/` | 1.915529 -> 1.824162 ms (4.77% faster) | 2.000077 -> 1.945915 ms (2.71% faster) | 2.021113 -> 2.000897 ms (1.00% faster) |

The small Graph effect was sensitive to run order and did not translate into
a repeatable eager improvement. It is insufficient to retain duplicated
kernels, two compression launches, and the fixed stream allocations. The
planned direct-detector output may remove part of that cost, but it is a
different experiment and must be measured against the row baseline.

**Decision.** Rejected and reverted before commit. Do not retain a separate
post-detection active stream. Revisit only together with direct compact output
from the detector and a controlled end-to-end speedup that exceeds timing
variance.

### 2026-08-28: revisit sparse point-contact AABB rejection

The earlier experiment below rejected another sparse point-contact broad phase
because the measured dynamic-fold collision Graph took only 0.026726 ms. That
result remains valid for that captured state. The optimization is revisited for
full-robot scenes whose world-compatible candidate table contains many remote
mesh collision shapes, to measure whether an explicit solver-private spatial
rejection can make that already-small pass cheaper without changing contacts.

**Pending implementation.** Both MJVBDV2-private `soft` contact entry points
now update one conservative world AABB per shape, then scan their unchanged
world-compatible particle/shape table. A pair outside the shape bound expanded
by the runtime soft-contact margin and particle radius returns before shape
transform inversion, SDF evaluation, or mesh query. Shape margin remains in
the shape bound. Infinite planes remain unconditionally eligible because their
signed half-space has no finite scene-independent AABB.

This is deliberately not a dynamic BVH or compact pair list. Candidate
capacity, launch dimensions, pair order, replay tids, counter semantics,
contact fields, contact thresholds, and CUDA Graph topology remain fixed.
Runtime particle positions, body poses, shape margins, and collision flags are
reread on every pass. All implementation code is private to `mjvbd_v2`; no
shared Newton collision or solver module changes.

**Controlled A/B.** The self-contained dynamic W1 T-shirt example ran on an
NVIDIA GeForce RTX 5090 D v2 at its initial state. The VBD destination view had
6,436 particles and 88 shapes: 86 triangle meshes, one box, and one plane. Its
world-compatible table contained 566,368 pairs, of which the conservative
AABB test admitted 6,444 (1.138%); both variants emitted eight contacts. Each
collision path was captured as a separate CUDA Graph, warmed for 20 launches,
then measured in alternating order over nine samples of 1,000 synchronized
replays. The table reports the median complete collision-Graph time.

| Sparse point collision | Time | Change |
| --- | ---: | ---: |
| Original direct SDF/mesh-query scan | 0.024180 ms | baseline |
| Per-shape AABB plus pair early return | 0.017609 ms | 27.18% faster |

The absolute saving is only 0.006571 ms per collision. Ten collision refreshes
would therefore save about 0.066 ms per displayed frame, below 0.1% of the
roughly 68 ms realtime-IK folding frame previously measured. This confirms
that the AABB rejects remote robot meshes correctly, but also confirms the
earlier conclusion that sparse point collision is not the dynamic fold's
frame-time bottleneck.

**Correctness evidence.** A CPU/CUDA reference regression compares both private
entry points against the original `create_soft_contacts` kernel. It covers a
moving kinematic sphere, static primitive contact, a watertight mesh whose
vertices are offset from its authored shape origin, a spatially remote mesh,
runtime body motion, and runtime `COLLIDE_PARTICLES` changes. Sorted particle
and shape IDs, unified indices, barycentrics, body-local points, body
velocities, and normals agree at `1e-6`. The same test captures both paths on
CUDA and changes shape margin before replay. It passed on an NVIDIA GeForce RTX
5090 D v2 as well as the CPU path.

**Decision.** Pending, low impact. The representative mesh-heavy collision
pass is faster and exact, but the expected end-to-end gain is below measurement
noise while the private contact kernel adds maintenance cost. Keep this as an
explicitly measured candidate until the affected W1 bag/cloth demos are timed;
revert it if they show no stable frame gain or if primitive-heavy scenes
regress.

### 2026-08-25: remove rigid-only gap from private soft-surface AABBs

The retained private full-surface mask reused shape AABBs produced by the
shared rigid broad phase. Those AABBs include `shape_margin + shape_gap`.
Full-surface particle contact, however, uses the soft-contact margin, incident
particle radius, and `shape_margin`; `shape_gap` only broadens rigid pair
detection. The builder default is 0.1 m, so an Armadillo only 60 mm above its
table initially admitted most table/hand edge and face pairs to SDF narrow
phase even though they could not emit a soft contact.

**Retained implementation.** The two MJVBDV2-private mask kernels tighten the
shared AABB by each shape's positive gap before testing an edge or face. They
leave `shape_margin` in the shape bound and continue to expand the soft feature
by the runtime soft-contact margin and maximum incident particle radius. A
1e-6 m safety margin keeps the rejection conservative. Negative gaps retain
their original deliberately reduced broad-phase bound. Shape transforms,
shape flags, contact thresholds, pair IDs, SDF iterations, capacities, and
Graph launch topology are unchanged. No shared collision or VBD file changed.

**End-to-end measurement.** An NVIDIA GeForce RTX 5090 D v2 ran
`example_vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher_final00` with a null
viewer, complete-frame CUDA Graph, ten substeps, ten VBD iterations, 15,228
particles, 62,770 tetrahedra, 20,000 surface triangles, 30,000 surface edges,
and 26 shapes. Both separate-process runs used three warm-up frames and 297
measured frames at ordinary process priority.

| Private full-surface bound | Time / 297 frames | Throughput |
| --- | ---: | ---: |
| Rigid AABB including 0.1 m `shape_gap` | 46.82 s | 6.34 FPS |
| Same AABB tightened to the soft-contact bound | 42.42 s | 7.00 FPS |

Synchronized 20-frame windows show that the gain grows when the hand creates
many contacts:

| Phase window | Before | After | Frame-time reduction |
| --- | ---: | ---: | ---: |
| Initial hold | 153.572 ms | 142.428 ms | 7.26% |
| Pre-grasp hold | 135.747 ms | 122.639 ms | 9.66% |
| Finger close | 244.154 ms | 181.465 ms | 25.68% |
| Grasp settle | 261.006 ms | 209.131 ms | 19.88% |

At a representative close state, the private compact pass admitted 48,360
edge pairs and 33,302 face pairs. Before tightening, a comparable close state
admitted 658,776 and 439,767 pairs respectively. These evolving-state counts
explain the hotspot reduction but are not an exact frozen-state comparison.
After the change, one eager close frame spent 19.897 ms in compact face SDF
generation and 1.485 ms in edge SDF generation; before it spent 53.238 and
4.437 ms respectively.

**Correctness evidence.** The focused CUDA regression places a second rigid
box inside the default 0.1 m rigid gap but outside every full-surface soft
threshold. Its pairs are rejected while the private and shared pipelines
still emit identical sorted shape/particle keys, barycentrics, body-local
points, and normals. The same test changes runtime margins, captures and
replays the Graph, and toggles `COLLIDE_PARTICLES`. A 320-frame Armadillo trace
remained finite through initial hold, approach, close, and grasp settle.

**Decision.** Retain. This removes only a rigid-broad-phase expansion that is
not part of the soft-contact equation, gives a material end-to-end gain, and
keeps the implementation within the MJVBDV2 migration boundary.

### 2026-08-25: reject additional self-contact tuning

After the retained AABB change, an eager close-frame GPU profile measured
192.727 ms. Edge-edge detection used 59.259 ms, vertex-triangle detection
29.615 ms, self-contact force/Hessian evaluation 28.097 ms, planar truncation
29.501 ms, rigid-soft SDF generation 21.382 ms, and volumetric elasticity
10.394 ms. Self-contact is therefore the remaining dominant cost, but the
following exact-semantics experiments did not improve it:

- CUDA collision block sizes 16, 32, 64, 128, and 256 measured 3.368, 3.899,
  3.630, 3.709, and 3.842 ms per complete self-detection pass. Keep 16.
- Rebuilding both BVHs after the close deformation cost 0.477 ms and changed a
  repeated detection pass from 3.312 to 3.319 ms. Refit quality was not the
  cause of the slowdown.
- Testing current distance before the rest-shape exclusion is algebraically
  equivalent, but its extra branch/register pressure reduced the 300-frame
  Graph result from 7.00 to 6.80 FPS. The shared prototype was reverted.

Do not reduce self-contact cadence, radius, exclusion distance, substeps, or
VBD iterations under the label of a no-effect optimization. Further work
needs a contact-set-preserving algorithmic change and a new controlled A/B.

### 2026-08-24: capture the dynamic fold as a single-stream frame graph

The dynamic T-shirt example previously overrode `capture()` to disable CUDA
Graph execution unconditionally. One displayed frame submits ten coupled
MuJoCo/VBD substeps, and every substep runs 20 colored VBD iterations. The
uncaptured path also read the active soft-contact count inside each VBD step
and copied the shape-friction array through the host once per frame. This made
kernel submission and synchronization overhead large even though the
MuJoCo-to-VBD order was already correct.

**Retained implementation.** The dynamic example now records one complete
single-stream frame graph for each scripted material tuple. A device counter
loads both cached PD-target endpoints before the coupled solve, so Graph replay
preserves the original target-to-target interpolation instead of using the
kinematic example's actual-state-to-target interpolation. The counter
saturates with both endpoints at the final cache row, avoiding a persistent
nonzero target velocity after the script ends. The standalone example keeps
the original two-stage initialization inside one file: its file-local base
creates the same temporary kinematic setup, then the final dynamic coupled
solver records only after its contact buffers exist.

The uncaptured fallback now updates shape friction with a device kernel instead
of copying the array to NumPy and back. Contact-count diagnostics run only in
`--test` mode. No solver source, trajectory, PD gain, contact coefficient,
substep count, VBD iteration count, collision cadence, or one-way coupling
rule changed. CUDA uses Graph by default; `--no-graph-capture` retains the
ordinary execution path.

**Local A/B measurement.** The example ran on an NVIDIA GeForce RTX 5090 D v2
with its then-default trajectory time scale of 2, ten substeps, 20 VBD iterations,
cloth self-contact enabled, and a null viewer. Each separate process used three
warm-up frames followed by 17 measured frames at ordinary process priority.
The timing excludes offline IK-cache construction and Graph recording.

| Execution | Measured time | Throughput |
| --- | ---: | ---: |
| Uncaptured single stream | 4.97 s / 17 frames | 3.42 FPS |
| Complete-frame CUDA Graph | 1.05 s / 17 frames | 16.2 FPS |

This is approximately 4.74x throughput and 78.9% lower measured frame time.
It is a short end-to-end viewer benchmark rather than a multi-sample median,
but the margin is much larger than the run-to-run noise observed in prior
single-digit-percent experiments.

**Correctness evidence.** A five-frame Graph/non-Graph comparison exercised
material changes and final-cache saturation. `particle_q` and `particle_qd`
were bitwise identical. Maximum absolute differences were 1.49e-8 for
`joint_q` and 5.37e-7 for `joint_qd`, consistent with floating-point launch
rounding and far below visible or control-relevant scale. A CUDA `--test` run
also captured and replayed the optional contact-count diagnostic. The CPU
uncaptured path completed a separate construction-and-step smoke test.

**Decision.** Retain the complete single-stream Graph in the dynamic example.
Do not interpret this result as support for same-GPU MuJoCo/VBD concurrency:
the rejected two-stream experiment competed for GPU resources, while this
change preserves strict sequential dependencies and removes submission and
readback overhead.

### 2026-08-24: solve dynamic-fold IK inside the frame graph

The dynamic T-shirt example originally generated its complete joint-target
trajectory during construction and replayed adjacent cache rows as MuJoCo
position/velocity-drive targets. That kept IK outside the measured frame but
could not respond to a target producer that changes at runtime.

**Retained implementation.** Both folding examples now own persistent device
buffers for the current Cartesian TCP poses and grip command. Every displayed
frame updates those inputs and executes 24 fixed IK iterations. The kinematic
example interpolates from its current prescribed joint state. The dynamic
example instead retains the preceding IK solution as the start of its MuJoCo
PD-target interval; using the lagging simulated joint state there would turn
tracking error into an incorrect target velocity. MuJoCo still integrates the
robot and VBD still consumes one-way link proxies. No dynamic joint position is
written directly.

The complete CUDA Graph contains target unpacking, IK, material selection, ten
coupled substeps, and 20 VBD iterations per substep. The examples no longer
contain an offline joint-target cache or a device cache-row counter. Cartesian
trajectory sampling remains a host-side input producer and can be replaced by
another realtime pose source without changing the captured computation.

**Performance evidence.** A short null-viewer run on an NVIDIA GeForce RTX
5090 D v2 used the then-default trajectory time scale of 2, ten substeps, 20 VBD
iterations, and three warm-up frames. The following 17 frames ran at 14.7 FPS.
The earlier cached-target measurement in the preceding section was 16.2 FPS;
the roughly 9% throughput difference is supporting evidence from separate
runs, not a controlled revision A/B, and represents the expected cost of 24
IK iterations now executed every frame.

**Correctness evidence.** A five-frame Graph/eager comparison at trajectory
time scale 100 produced bitwise-identical IK solutions, PD targets, particle
positions, and particle velocities. Maximum absolute MuJoCo differences were
4.77e-7 rad in joint position and 1.53e-5 rad/s in joint velocity. Neither
instance created `cached_joint_targets`. Separate two-frame Graph and eager
tests crossed accelerated material phases. Finally, the default 900-frame
Graph run passed all per-frame finite-state checks and the final dynamic
backend, joint motion, tracking-error, cloth-contact, and folded-area checks.

The standalone dynamic example uses the same authored TCP, grip, timing, and
material schedule as the kinematic example. Its former release override
opened each hand from fully closed to fully open while stationary at the first
and second place poses, causing the finger geometry to sweep through the shirt
before lift-away. The first shared replacement still began lift-away at 0.75 grip.
A joint-tracking trace showed that this was a geometric clamp rather than a
dynamic-IK lag: the driven finger coordinates tracked within about 0.003 rad,
while the larger 0.06 rad error belonged to the passive PIP mimic and opened
the finger rather than closing it. The retained trajectory therefore relaxes
grip from 1.0 to 0.25 while the TCP is stationary, then opens from 0.25 to 0.0
during lift-away. TCP poses, segment durations, friction phases, solver
iterations, and contact parameters are unchanged. A former dynamic-only
material override used a condition that stayed active after the first place
pose, unintentionally forcing every later open-hand phase—including the final
retreat after the second fold—to zero friction instead of restoring the shared
0.25 value.

A separate tracking diagnostic showed that the `ANKLE`, `KNEE`, and
`BUTTOCK` support drives contributed a correlated 16.6 mm TCP displacement
with the original 8,000/400 gains. Raising only the non-folding support gains
to 50,000/1,000 reduced the displacement to 4.3 mm; doubling stiffness again
recovered only another 1.2 mm. The conservative 50,000/1,000 values are
therefore retained. One complete 900-frame default Graph test passed after the
trajectory and support-drive corrections. Two subsequent repetitions became
non-finite at frames 715 and 724 with respectively zero and 0.25 final-retreat
finger friction. That late-fold sensitivity is therefore not attributed to
the material phase; it remains a separate robustness issue in this dense,
nondeterministic self-contact trajectory.

**Decision.** Retain realtime IK in both folding examples. Keep it inside the
single-stream frame Graph so the functional requirement adds IK computation
without restoring per-iteration Python launch overhead.

### 2026-08-25: defer dynamic-fold parity and second-pass penetration

The kinematic and dynamic folding examples each solve realtime IK once per
displayed frame with the same Cartesian targets and fixed iteration count.
This does not make their simulated hand trajectories equivalent. The
kinematic path writes the interpolated joint coordinates into every physics
substep and evaluates FK directly, so its link poses exactly follow the IK
trajectory. The dynamic path interpolates MuJoCo PD targets from the preceding
IK solution, then MuJoCo integrates the actual joints with finite drive gains,
damping, and effort limits. MJVBDV2 synchronizes those actual MuJoCo link
poses, not the IK targets, into VBD as one-way moving collision proxies.

Consequently the dynamic links may lag or overshoot even when both examples
produce the same IK solution. The cloth state also diverges during the first
fold, so the shared second-pass trajectory is no longer guaranteed to match
the deformed cloth. The authored first grasp contains a separate hover
approach followed by a descent, while the second pass moves directly from the
first release pose to the second grasp pose. That diagonal open-loop segment
has no second-pass clearance waypoint and can sweep a dynamic hand through
the already-folded shirt. The finite-stiffness, discrete soft-contact model is
not a hard nonpenetration constraint and cannot by itself guarantee that this
motion remains intersection-free.

The standalone dynamic example now keeps the pre-existing initialization,
trajectory, material schedule, Graph capture, and solver order inside one
file; this refactor does not claim to correct the second-pass penetration.
Further dynamic-example tuning is deferred. The next targeted experiment
should add a second-pass hover waypoint and vertical descent, then measure
per-stage actual-to-target joint and TCP error before changing drive gains,
phase timing, substeps, or contact parameters. Increasing IK frequency alone
does not address the identified difference.

### 2026-08-24: fuse device-selected truncation application

The optimized `vbd_soft/` self-contact path keeps its active-contact selector
on the device so it remains CUDA Graph compatible. For each particle color
and VBD iteration, it formerly launched two mutually exclusive kernels: the
active branch applied planar truncation to every particle, and the inactive
branch applied identity truncation only to the selected color. One kernel did
all useful work while every thread in the other returned. The dynamic T-shirt
frame launches each family 1,800 times (ten substeps, 20 iterations, and nine
color passes), so the redundant Graph node was measurable even when it did no
arithmetic.

**Retained implementation.** One solver-local kernel reads the same device
selector. When active, its particle-count launch uses each thread index and
the computed truncation factor. When inactive, threads beyond the selected
color count return and the remaining threads use the selected particle IDs
with an identity factor. It preserves the all-particle active path, selected
inactive path, displacement clamp, output writes, fixed capture topology across
runtime selector values, and Gauss--Seidel order. The complete `vbd/` solver
is unchanged because it does not use this two-kernel selector.

**Controlled A/B.** An NVIDIA GeForce RTX 5090 D v2 benchmark used 5,886
particles, 736 selected particles, and a captured 1,800-iteration Graph. Each
variant received identical positions, displacements, truncation factors, and
selector values. Ten warm-up replays preceded 21 timed replays; the table
reports the median complete-Graph GPU time.

| Device selector | Two kernels | Fused kernel | Reduction |
| --- | ---: | ---: | ---: |
| No active self-contact | 8.742705 ms | 4.665207 ms | 46.64% |
| Active self-contact | 7.166376 ms | 3.814676 ms | 46.77% |

The benchmark compared displacement and position outputs bitwise for both
selector values. A representative evolving-frame trace also reduced the two
truncation-application families from about 13.6--16.0 ms to about 6.5 ms, but
that cross-run figure is supporting evidence rather than the controlled A/B.
The full 450-frame dynamic folding example passed its finite-state, realtime
IK, dynamic backend, contact, tracking-error, and folded-area assertions.

**Decision.** Retain the fused private `vbd_soft/` kernel. It removes one
kernel family without changing collision detection, contact evaluation,
iteration count, material behavior, or any backend outside MJVBDV2.

### 2026-08-21: parallelize dense rigid-side soft contacts

The complete private VBD solver previously launched four threads per rigid
body. Each thread walked one quarter of that body's particle, edge, and face
contacts serially, then atomically added five accumulated force/Hessian blocks.
This underutilized the GPU in scenes with a few rigid bodies and hundreds of
soft contacts per body.

**Retained implementation.** On nondeterministic CUDA models that do not
require gradients, a per-body contact capacity of at least 512 enables a hybrid
path. Bodies below 128 active contacts stay on the four-thread kernel. Dense
bodies use 64-contact blocks, fixed-capacity partial buffers, and one
block-level final reduction. CPU, deterministic, differentiable,
external-rigid, and small-buffer configurations retain the legacy
accumulation. All dispatch and scratch storage live under `mjvbd_v2/vbd/`.
The solver continues to iterate the model's original rigid color groups;
existing kernel guards reject static and kinematic bodies.

An isolated CUDA Graph benchmark used one dynamic body on an NVIDIA GeForce RTX
5090 D v2. Each sample followed 100 warm-up replays and averaged 2,000 graph
replays. Both graphs cleared the five body outputs. The dense graph also
included the sparse-path gate, chunk evaluation, and final reduction.

| Active contacts / capacity | Four-thread path | Hybrid path | Throughput |
| ---: | ---: | ---: | ---: |
| 128 / 512 | 74.913 us | 21.232 us | 3.53x |
| 256 / 512 | 135.350 us | 21.295 us | 6.36x |
| 512 / 512 | 260.071 us | 21.391 us | 12.16x |
| 1,024 / 1,024 | 540.278 us | 21.413 us | 25.23x |
| 4,096 / 4,096 | 2,434.821 us | 23.298 us | 104.51x |

The end-to-end A/B used
`example_mjvbd_v2_supermarket_plastic_bag.py`: four dynamic balls, 5,886
particles, 11,512 triangles, 17,399 bending edges, eight particle colors, six
substeps, 12 VBD iterations, full contact, self-contact, and a captured frame
graph. Construction and capture were excluded. Each separate run advanced 200
frames, synchronized every 20 frames, and averaged all step batches.

| Rigid-side accumulation | Frame time | Change |
| --- | ---: | ---: |
| Legacy four-thread path | 37.670750 ms | baseline |
| Gated chunk reduction | 28.246249 ms | 25.02% faster, 1.33x throughput |

Both runs passed the example's final stability, containment, handle-contact,
and ball/bag penetration checks. A focused CUDA regression compares force,
torque, and all three Hessian blocks for 1,024 contacts. The path also captured
and replayed as part of the complete example frame graph. The comparison uses
floating-point tolerance because the retained path is restricted to
nondeterministic execution and deliberately changes parallel reduction order.

**Rejected companion change.** Both end-to-end variants above used the same
construction-time pruning of zero-inverse-mass and kinematic rigid color rows,
so the measurement isolates dense reduction and provides no evidence that
pruning helps. Pruning also cached Python launch topology from mutable mass and
body flags, requiring both `notify_model_changed()` and CUDA Graph recapture
when either changed. It was removed before commit; device-side inverse-mass
guards retain the original runtime behavior and color groups.

**Decision.** Retained with all gates above. Do not enable it for deterministic
or differentiable execution, and do not replace the sparse four-thread path.
The isolated speedup is a hotspot result; use the measured 25.02% only for this
representative full-contact bag topology.

### 2026-08-21: reject another sparse point-contact broad phase

The dynamic T-shirt fold's private soft pipeline contains 566,368
world-compatible particle/shape candidates and emitted 16 contacts at the
measured first-frame state. Despite the large static capacity, one captured
collision pass took a median 0.026726 ms on an NVIDIA GeForce RTX 5090 D v2
after ten warm-up launches. The timing used nine samples of twenty synchronized
Graph launches. Ten such passes account for less than 0.3 ms beside the roughly
300 ms representative frame measured below.

**Decision.** Rejected before implementation. A second AABB pass would require
extra buffers and either a private copy of the point-contact narrow phase or a
new shared hook. Even eliminating the measured collision work entirely cannot
materially improve this scene. The Cartesian candidate count is therefore a
memory-capacity concern for replicated worlds, handled by the retained
world-compatible sizing and large-batch paths, not the single-scene dynamic
fold's runtime bottleneck.

### 2026-08-21: reject a specialized coupled frame executor

**Hypothesis.** The generic proxy coupler performs separate source-state
distribution, MuJoCo-to-Newton conversion, forward kinematics, proxy-state
injection, and destination-state reconciliation for every substep. A private
MJVBDV2 frame executor could fuse those transfers and overlap MuJoCo substep
`i + 1` with collision and VBD substep `i` on a second CUDA stream. The
experiment preserved the existing one-way coupling order: every VBD substep
consumed the matching, newly computed MuJoCo pose and velocity. It did not read
MuJoCo's derived `xpos`, `xquat`, or `cvel` before they were refreshed.

The candidate contained three cumulative variants:

1. fused global-to-view and view-to-global kernels plus a fused joint-control
   copy;
2. one frame-level API that executed all internal substeps on one stream; and
3. double-buffered proxy states with CUDA events and a MuJoCo/VBD two-stream
   wavefront.

All buffers and events were persistent. No host readback, dynamic allocation,
iteration reduction, stale-pose approximation, or feedback suppression was
introduced.

**A/B measurements.** Tests used an NVIDIA GeForce RTX 5090 D v2 without CUDA
Graph capture. The representative dynamic fold contained one 40-body,
40-joint articulation, 88 shapes, 6,436 particles, 12,736 triangles, and
19,174 edges. It used 10 substeps and 20 VBD iterations. Initialization used a
shortened offline IK trajectory cache, but physics topology, substeps,
iterations, controls, and simulated frame states were identical between each
pair. Alternating-order measurements discarded the first three frames and
reported the median of nine synchronized frame samples.

| Candidate | Specialized path | Repeated public `step()` | Result |
| --- | ---: | ---: | ---: |
| Fused transfers and control copy | 293.170226 ms | 289.295990 ms | 1.32% slower |
| Single-stream frame executor | 298.339913 ms | 299.887690 ms | 0.52% faster, within run-to-run noise |
| Two-stream wavefront | 303.976088 ms | 301.250214 ms | 0.90% slower |

A deliberately small coupled scene with one revolute MuJoCo link, one
VBD-owned rigid body, one particle, four substeps, and one VBD iteration
measured 16.852946 ms for the wavefront and 17.132567 ms for repeated steps, a
1.66% gain. This establishes that the executor can hide submission work when
the physics workload is tiny, but the gain is not representative of the
solver's target scenes.

**Correctness evidence.** A three-substep CPU comparison covered nonzero body
and particle forces and changing joint forces, position targets, velocity
targets, and actuator inputs. All compared state and internal-control arrays
were exact. CUDA comparisons checked joint and particle state after every
frame in both the light and representative scenes; all passed at the existing
GPU tolerance. The complete MJVBDV2 unit-test module passed before timing.

**Decision.** Rejected and reverted. In the representative scene, VBD and
collision dominate the frame. MuJoCo and VBD also compete for the same GPU, so
event overhead and resource contention consume the small amount of work that
could overlap. Transfer fusion removes about twenty submissions per substep
but does not remove enough GPU work to produce a stable end-to-end gain. Do
not add a specialized frame API or make examples depend on it. Reconsider only
for a CPU-MuJoCo/GPU-VBD backend, a many-environment workload with measured
transfer dominance, or hardware where a trace demonstrates real concurrent
execution. Optimize the private collision and VBD paths first.

### 2026-08-21: retain private full-surface AABB rejection

This revisits the rejected shared-pipeline experiment below without changing
files outside `newton/_src/solvers/mjvbd_v2/`. Every V2 `full` contact backend
now constructs `MJVBDV2CollisionPipeline`, a private layer over the shared
pipeline. CPU and scenes without full-surface rigid-soft contact use the
original implementation directly.

On CUDA, construction stores the unchanged world-compatible edge/shape and
face/shape candidate sets in stable shape-major order. Each collision pass
first runs the ordinary rigid and particle contact work, which also refreshes
the rigid shape AABBs. Two private kernels then mark whether each soft
feature's world AABB overlaps the current rigid shape AABB. The feature bound
is expanded by the runtime soft-contact margin and maximum incident particle
radius; the rigid bound already contains its shape margin and gap. Masked
edge/face kernels return before shape transforms or SDF evaluation.

The implementation does not compact candidates or resize buffers at runtime.
Candidate capacity, replay-tid offsets, contact thresholds, optimizer
iterations, contact record fields, and graph launch dimensions remain fixed.
Runtime particle positions, body poses, shape flags, and contact-margin
overrides are read on every launch.

**Controlled A/B.** The benchmark loaded the same saved frame-121 state from
`example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff.py` into an
exact shared `CollisionPipeline` and the private candidate in one process. It
used an NVIDIA GeForce RTX 5090 D v2, 5,886 particles, 11,512 triangles,
17,399 edges, 24 selected full-surface shapes, 417,576 edge/shape pairs, and
276,288 face/shape pairs. Both paths were captured as separate CUDA Graphs,
warmed for ten launches, then measured in alternating order over nine samples
of twenty synchronized launches.

| Measurement | Shared baseline | Private V2 pipeline | Result |
| --- | ---: | ---: | ---: |
| Frozen frame-121 collision graph | 3.825585 ms | 1.643479 ms | 57.04% lower time; 2.33x throughput |

The conservative masks retained 38,522 edge pairs and 25,343 face pairs for
narrow phase, rejecting 90.77% and 90.83% respectively. Both pipelines
emitted 9,838 contacts, and their sorted `(shape, particle indices)` keys were
identical. This is an isolated collision measurement; it is not presented as
an equal end-to-end frame gain. The earlier full-graph measurements in the
next entry remain supporting evidence that this hotspot affects the complete
handoff scene, but they used the prior shared prototype and are not relabeled
as measurements of this revision.

**Correctness evidence.** A CUDA unit scene compared the shared and private
contact keys, particle ids, barycentrics, body-local points, and normals, then
captured and replayed the private collision graph three times. Shape-major
masks contained both accepted and rejected candidates, the full-surface
buffer marker remained enabled, and all values agreed with relative and
absolute tolerances of `1e-6`. All 44 tests in `test_mjvbd_v2` and
`test_mjvbd_v2_contact_optimizations` passed on CPU/CUDA. A new architecture
test also requires every V2 full-contact backend to use the private module.

**Decision.** Retained. The controlled hotspot gain is large, the emitted
contact set is preserved, the path remains graph-capturable, and the entire
implementation lies inside MJVBDV2. Re-evaluate only if a future shared
pipeline exposes an equivalent mask hook or compact broad phase without
changing contact semantics.

### 2026-08-24: compact active full-surface candidates

The retained AABB pass above still launched the expensive edge and face SDF
kernels over their complete candidate capacities. Rejected pairs returned
immediately, but a CUDA Graph replay continued to schedule every logical pair,
and mixed active/inactive warps retained avoidable branch and grid-stride
overhead.

**Retained implementation.** The same private AABB kernels now append passing
original pair indices into fixed-capacity device arrays while writing the
existing masks. A two-element device counter is reset once per collision.
Edge and face contact generation use fixed 128-thread persistent kernels with
at most two blocks per SM; workers stride only over the device-side active
count. The original pair index is still passed to the contact emitter, so
replay-tid ranges, edge/face offsets, contact capacity, SDF iterations,
thresholds, and record fields are unchanged. Runtime particle positions, body
poses, shape flags, margins, and AABBs are reread on every collision, including
inside a captured Graph.

Compaction is enabled only for CUDA full-surface pipelines that do not require
gradients. CPU, non-full-surface, and differentiable pipelines retain the
previous path. Candidate and contact buffers remain construction-time fixed;
there is no host readback, allocation, topology mutation, or Graph recapture.
The prior masked kernels remain as the fallback and as an A/B reference.

**Controlled A/B.** Tests used an NVIDIA GeForce RTX 5090 D v2 and captured
single-stream CUDA Graphs. Collision measurements used ten warm-up replays and
1,000 timed replays per process. The handoff full-frame measurement used three
warm-up replays and 30 timed replays; three separate masked and compact runs
are summarized by their median. No solver iteration, substep, contact margin,
buffer size, or scene parameter changed.

| Scene and topology | Masked | Compact | Result |
| --- | ---: | ---: | ---: |
| Handoff frame 130 collision; 417,576 edge and 276,288 face pairs; about 8.6% AABB-active | 2.291320 ms | 2.048838 ms | 10.58% faster |
| Handoff captured frame; 6 substeps and 12 VBD iterations | 45.602193 ms | 42.603617 ms | 6.58% faster |
| Pneumatic bag frame 10 collision; 7,680 edge and 5,120 face pairs; 56.25% and 57.50% active | 0.681671 ms | 0.268998 ms | 60.54% faster |
| Synthetic 96x96 cloth/plane collision; all 27,840 edge and 18,432 face pairs active | 0.187894 ms | 0.099183 ms | 47.21% faster |

The handoff collision medians came from masked samples of 2.291320, 2.321063,
and 2.129393 ms and compact samples of 2.007445, 2.104492, and 2.048838 ms.
The full-frame samples were 44.772257, 45.623088, and 45.602193 ms masked and
41.329248, 45.097961, and 42.603617 ms compact. The synthetic all-active case
guards against a regression when conservative AABBs cannot reject any pair.

Two exact `v10000` demo spot checks used the same GPU and each demo's default
captured physics graph. Collision medians came from five samples of 500 graph
replays. Full-frame medians came from five samples of ten graph replays after
three warm-ups. Frame 1 checks the warmed initial state; frame 240 checks the
named active interaction phase. These are representative-phase A/B results,
not complete 720- or 1,900-frame demo regressions.

| Exact demo and phase | Collision masked / compact | Collision result | Frame masked / compact | Frame result |
| --- | ---: | ---: | ---: | ---: |
| `right_hand_recorded_plastic_inflatable_bag_pick_release_v10000`, frame 1 (`validate_initial`) | 0.601839 / 0.279699 ms | 53.53% faster | 21.200090 / 19.325244 ms | 8.84% faster |
| Same demo, frame 240 (`lift`) | 1.772287 / 1.363240 ms | 23.08% faster | 27.690378 / 25.499519 ms | 7.91% faster |
| `dexforce_recorded_soft_then_rigid_cube_into_bag_v10000`, frame 1 (`soft_wait`) | 2.196659 / 1.831552 ms | 16.62% faster | 53.287305 / 50.258893 ms | 5.68% faster |
| Same demo, frame 240 (`soft_grasp`) | 3.561637 / 3.236834 ms | 9.12% faster | 66.660876 / 65.249060 ms | 2.12% faster |

**Correctness evidence.** The focused CUDA regression compares compact and
masked records as well as the shared pipeline's sorted shape, particle,
barycentric, body-point, and normal fields. It also checks that compacted pair
indices exactly equal the AABB masks, changes contact margins, captures and
replays the Graph, and disables then restores `COLLIDE_PARTICLES` without
recapture. A ten-frame pneumatic rollout remained finite. Its compact/masked
state difference was below the old masked path's masked/masked variation in
the existing nondeterministic atomic mode (particle maximum 0.000547 versus
0.001597 m; body-array maximum 0.002328 versus 0.006729; cavity-volume maximum
6.90e-7 versus 1.12e-6 m^3). The compact path also completed all 720 default
frames of `right_hand_recorded_plastic_inflatable_bag_pick_release_v10000`
and all 1,900 default frames of
`dexforce_recorded_soft_then_rigid_cube_into_bag_v10000`; both demos passed
their own final-state assertions under the Null viewer.

**Decision.** Retained with the CUDA/non-gradient gates above. The compact
path preserves the contact set and runtime collision controls, improves both
sparse and fully active candidate distributions, and remains entirely inside
MJVBDV2. Keep the masked fallback for gradients and for future A/B checks.

### 2026-08-20: profile full-surface bag contact work

An Nsight Systems CUDA-node trace measured one captured frame after frame 121
of `example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff.py`. The frame
used an NVIDIA GeForce RTX 5090 D v2, 6 substeps, 12 VBD iterations, 5,886
particles, 11,512 triangles, 17,399 edges, and 8 particle colors. The solver
was in the post-handoff `vbd_kinematic_full` phase with hand contact enabled.

| GPU work | Frame share |
| --- | ---: |
| Full-surface face contact generation | 30.1% |
| Full-surface edge contact generation | 3.3% |
| Legacy particle contact generation | 2.1% |
| Rigid-side body-particle contact accumulation | 18.2% |
| Particle-side body-particle contact accumulation | 7.9% |
| Surface elasticity | 9.8% |
| Self-contact force accumulation | 8.5% |
| Penetration-free planar truncation | 7.1% |

The collision pipeline precomputes only world-compatible feature/shape pairs.
For this one-world scene, its full-surface arrays are Cartesian products:

- 24 selected full-surface shapes, including 22 hand collision shapes;
- 276,288 triangle/shape pairs (`11,512 * 24`); and
- 417,576 edge/shape pairs (`17,399 * 24`).

Every face pair currently evaluates the shape SDF at the triangle centroid
before the first geometric rejection. For hand mesh SDFs, that query is much
more expensive than a world-space AABB comparison.

**Experimental implementation.** Full-surface edge and face pairs were sorted
shape-major at pipeline construction. Two lightweight kernels write one-byte
active masks by comparing each soft feature's world AABB, expanded by particle
radius and the exact contact margins, with the current rigid-shape AABB. The
original SDF kernels read the mask before shape transforms or SDF samples.
Static candidate counts, replay-tid ranges, contact emission, SDF iteration
counts, and CUDA Graph launch dimensions remained unchanged. Pipelines with
full-surface contact disabled allocated no masks and launched no new kernels.

The first two layouts were rejected during development:

- Computing the AABB test inside the SDF kernels raised their register
  footprint and made the full graph about 4.2% slower.
- A separate mask with the original feature-major pair order made frozen-state
  collision 5.7% slower. With 24 shapes per feature, almost every warp retained
  one nearby shape and diverged through the expensive branch.

Shape-major ordering lets far shapes retire whole warps and makes transform and
SDF texture access coherent. At the recorded post-handoff state it rejected
408,695 of 417,576 edge pairs (97.87%) and 270,154 of 276,288 face pairs
(97.78%). The two mask kernels together took about 0.052 ms per six-substep
frame in a CUDA Graph node trace.

**A/B measurements.** Tests used an NVIDIA GeForce RTX 5090 D v2 with the
handoff scene's 6 substeps, 12 VBD iterations, 5,886 particles, 11,512
triangles, 17,399 edges, and 24 selected full-surface shapes. The primary test
loaded the same saved frame-121 particle positions, body transforms, and shape
flags into the exact parent and candidate, captured only the collision graph,
warmed ten launches, and reported the median of nine samples of twenty
launches. Both versions generated exactly 9,838 soft contacts.

| Measurement | Exact parent | Candidate | Result |
| --- | ---: | ---: | ---: |
| Frozen frame-121 collision graph | 3.433520 ms | 2.079930 ms | 39.42% faster |
| Full evolving graph, baseline-before order | 61.028250 ms | 58.809458 ms | 3.64% faster |
| Full evolving graph, baseline-after order | 66.358963 ms | 58.809458 ms | 11.38% faster |

The full-graph figures are supporting evidence only because concurrent contact
emission and the evolving trajectory changed the active contact count between
processes. The frozen-state comparison controls both input geometry and active
count and is the retain/reject measurement.

**Correctness evidence.** All 48 CPU/CUDA full-surface collision tests passed,
including infinite planes, per-shape margin, nonuniform mesh SDFs, replay tids,
and CUDA Graph capture. All 36 MJVBDV2 tests passed. The 240-frame bimanual rod
handoff example completed its `test_final()` checks, including hand contacts,
contact capacities, finite state, ground penetration, and retained handles.

**Decision.** Rejected and reverted. The measured speedup is real, but the
implementation modified the shared geometry contact kernels,
`CollisionPipeline`, and its shared tests. MJVBDV2 must remain independently
migratable, so its runtime optimization boundary is
`newton/_src/solvers/mjvbd_v2/`. Reuse the measurements only when designing a
private MJVBDV2 contact pipeline; do not restore the shared implementation.

### 2026-08-20--21: reduce repeated rigid-soft contact solve scans

The particle-side contact kernel launches once per particle color and filters
records inside the kernel. A graph launch uses `soft_contact_max`, not the
active count. Device snapshots after warmed graph replays showed:

| Scene | Active / capacity | Current active record/color visits | Color-relevant visits | Repetition |
| --- | ---: | ---: | ---: | ---: |
| Ordinary supermarket bag, frame 12 | 857 / 35,316 | 6,856 | 857 | 8.00x |
| Bimanual handoff, frame 81 | 9,571 / 12,288 | 76,568 | 20,271 | 3.78x |
| Bimanual handoff, frame 121 | 9,964 / 12,288 | 79,712 | 21,172 | 3.77x |

The first row understates launched padding: its captured graph actually starts
`35,316 * 8 = 282,528` contact threads per VBD iteration before active-count
guards, while only 857 records are relevant to one color. In the post-handoff
snapshot, per-particle contact incidence reached 52 with a 99th percentile of
44, so a small fixed per-particle capacity is not safe.

Three solver-local layouts were measured while treating Newton's shared
`Contacts` buffers as read-only input:

1. An exact CSR with at most three entries per contact launched one thread per
   particle and traversed that particle's contacts serially. It removed padded
   scans but reduced GPU parallelism; the full handoff graph regressed from
   71.499767 ms to 122.728243 ms (71.65% slower) and was reverted.
2. Per-color compact contact-index arrays preserved one contact per thread, but
   their construction and random gathers offset the saved evaluations. The
   full graph measured 71.347331 ms against the same 71.499767 ms baseline,
   only 0.21% faster, and the layout was reverted.
3. The retained layout stores one `wp.uint32` membership mask per contact. A
   graph-captured kernel builds masks once per contact refresh. Each color still
   scans the original contiguous contact buffer, but unrelated active records
   return after one mask test and before material loads or rigid-soft force
   evaluation. Models with more than 32 graph colors use the original path.

**A/B measurements.** Tests used an NVIDIA GeForce RTX 5090 D v2, 6 substeps,
12 VBD iterations, 5,886 particles, 8 colors, and a 12,288-record contact
capacity. The frozen-state graph contained one mask build plus all particle-side
contact launches for 12 iterations and 8 colors. It used the same 9,880 active
records, positions, body poses, and contact data for both variants.

| Measurement | Original scan | Color mask | Result |
| --- | ---: | ---: | ---: |
| Frozen particle-side contact graph | 0.668677 ms | 0.632664 ms | 5.39% faster |
| Full evolving handoff graph | 71.499767 ms | 66.498750 ms | 6.99% faster |

The full-graph result is supporting evidence because independently evolved
runs ended with 9,816 and 9,767 active records. The frozen-state comparison is
the controlled retain/reject measurement.

**Correctness evidence.** On the frozen state, masked and original force and
Hessian arrays agreed within `rtol=1e-5, atol=1e-2`; differences were limited
to existing GPU atomic summation order. All 36 MJVBDV2 unit tests passed. The
real bimanual handoff scene captured and replayed the changed path in one CUDA
Graph and passed its 240-frame `test_final()`. CPU smoke tests covered both the
masked complete-VBD path and the original fallback for a synthetic 33-color
model.

**Decision.** Retained. The implementation changes only files under
`newton/_src/solvers/mjvbd_v2/vbd/`, adds no host readback, and leaves all
non-complete-VBD backends unchanged.

### 2026-08-20: reject full-VBD device-selected truncation

**Hypothesis.** Port the optimized `vbd_soft/` self-contact selector to the
complete `vbd/` implementation. With no active VT/EE candidate, update only
the current graph-color group instead of scanning every particle after every
color solve.

**Candidate work.** The experiment added a detector-counter reduction, a
device flag, selected-color truncation kernels, and active/inactive guarded
launches. The active-contact path retained the all-particle truncation pass so
that a contact plane could constrain vertices outside the current color.

**A/B measurements.** Tests used an NVIDIA GeForce RTX 5090 D v2 on CUDA and
replayed one captured graph after warm-up.

| Scene | Topology | Baseline | Candidate | Result |
| --- | --- | ---: | ---: | ---: |
| `example_vbd_inflatable_bag_v1.py` | 514 particles, 4 colors, self-contact disabled | 18.910584 ms/frame | 18.799767 ms/frame | 0.586% faster |
| `example_mjvbd_v2_supermarket_plastic_bag.py` | 5,886 particles, 8 colors, active self-contact | 37.944188 ms/frame | 37.046115 and 39.174723 ms/frame in repeat runs | No stable gain outside run-to-run noise |

The supermarket-bag candidate also adds one reduction kernel per collision
detection and guarded or empty launches on the active-contact path. That cost
is not justified by the inactive-path hypothesis in the measured flagship
scene.

**Correctness evidence.** The candidate passed all 37
`newton.tests.test_mjvbd_v2` tests, targeted pre-commit checks, and a CUDA graph
test that switched active and inactive self-contact states in one graph.

**Decision.** Rejected and reverted before commit. The implementation cost and
possible active-contact regression outweigh a sub-percent gain in the only
scene that benefited. Reconsider only with a compact detector-produced active
count or a fused kernel that does not add an extra full-array reduction.

### 2026-08-20: retain final-copy hoisting and surface-only tile solve

Commit `685aa797` contains two changes:

- copy the final solved particle positions from the mutable working state to
  the output once per step instead of once per VBD iteration; and
- split color groups at construction so particles without adjacent tetrahedra
  use a surface-only CUDA tile kernel, while volumetric particles retain the
  generic triangle/edge/tetrahedron kernel.

Both changes preserve force models, graph-color ordering, iteration count, and
the final state contract. They remove memory traffic and irrelevant
tetrahedron traversal without adding per-frame host synchronization.

An exact-parent (`685aa797^`) versus exact-commit A/B used the captured
`example_mjvbd_v2_supermarket_plastic_bag.py` step on an NVIDIA GeForce RTX
5090 D v2. The scene had 5,886 particles and 8 colors. Each process warmed six
graph replays, then reported the median of seven samples of ten replays.

| Run order | Parent | `685aa797` | Gain |
| --- | ---: | ---: | ---: |
| parent, then candidate | 38.614856 ms/frame | 37.353617 ms/frame | 3.27% |
| candidate, then parent | 38.590663 ms/frame | 35.513681 ms/frame | 7.97% |

**Decision.** Retained. The end-to-end gain is modest but repeatable across
both run orders, and the change removes work without adding runtime branches.
The benchmark measures the combined commit and cannot attribute the gain to
one of its two optimizations. If either implementation becomes difficult to
maintain, benchmark the two changes independently before keeping it solely for
performance.

### 2026-08-21: selectively port PR 3995 contact traversal and sizing

[Newton PR 3995](https://github.com/newton-physics/newton/pull/3995)
optimizes upstream VBD for large replicated workloads. MJVBDV2 already has a
surface-only elasticity specialization and a retained contact-color mask, so
the upstream patch cannot be copied wholesale.

**Retained work.** The two MJVBDV2-owned sparse particle-shape builders use a
stable shape-major candidate order on CUDA. The candidate set is unchanged;
the layout improves shape-transform and SDF locality.

The private VBD constructors size their initial body-particle contact state
from the world-compatible pair count instead of the full
`particle_count * shape_count` Cartesian product. The first step can still grow
the arrays for a larger externally supplied `Contacts` stream. In the measured
1,024-world setup, the old capacity was 268,435,456 records and the private
pipeline needed 262,144: a 1,024x reduction. For the four float contact-state
arrays in `vbd_soft/`, that changes the initial allocation from 4 GiB to 4 MiB.

Both private VBD implementations also contain the PR's linked per-particle
contact gather and grid-stride body-particle dual update, with stricter V2
dispatch. They activate only when at least one particle graph-color group has
at least `SM count * 128` particles. Small and ordinary single-scene models
retain their previous kernels and launch dimensions. Gather is additionally
disabled for deterministic execution, differentiable models, and unified
full-surface edge/face contact streams. Once a model qualifies, all colors use
gather so a mixed-size coloring cannot fall through to an uninitialized
contact-color mask. Its linked-list storage is allocated lazily from the
runtime `Contacts` capacity, not from the model's Cartesian particle-shape
upper bound.

The batch gate is intentional. A per-particle linked gather serializes all
contacts incident to one particle. That trades contact-level parallelism for
removing repeated capacity scans and is beneficial only when the particle
color group itself provides enough GPU work. The active-prefix dual update has
the same gate because its grid-stride loop was slower than the legacy
one-thread-per-capacity launch in the measured single scene.

**Local A/B measurements.** A CUDA Graph run of
`example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00.py` used an
NVIDIA GeForce RTX 5090 D v2, 10 substeps, 20 VBD iterations, 6,436 particles,
12,736 triangles, 19,174 edges, 88 shapes, 566,368 particle-shape candidates,
and about 5,050 active contacts. Each process warmed 100 frames and reported
the mean wall time of the next 30 frames. These are evolving-trajectory,
separate-process results and therefore supporting evidence rather than a
controlled frozen-state comparison.

| Candidate | Frame time | Change from legacy |
| --- | ---: | ---: |
| Legacy contact order, dual launch, and surface tile | 91.022263 ms | baseline |
| Shape-major sparse candidates only | 89.897158 ms | 1.24% faster |
| Shape-major plus active-prefix dual | 90.958515 ms | 1.18% slower than shape-major only |
| Shape-major plus two-particle surface tile | 98.299060 ms | 9.35% slower than shape-major only |
| All initially ported candidates | 98.559647 ms | 8.28% slower than legacy |

The final implementation therefore keeps shape-major ordering, disables the
new dual and gather paths for this topology, and retains the existing
surface-only tile kernel. PR 3995 reports 1.66x cloth and 1.32x soft-volume
throughput for its complete patch at 1,024 environments on RTX 4090/L40-class
hardware. Those upstream aggregate figures motivate the batch-only path but
are not attributed to any one MJVBDV2 optimization.

A separate frozen CUDA Graph benchmark replicated a 15-by-15-cell cloth and
one static shape into 1,024 worlds: 262,144 particles, 262,144 active contacts,
three color groups of 87,040--88,064 particles, eight VBD iterations, and an
NVIDIA GeForce RTX 5090 D v2. Both variants used shape-major candidates. After
20 warmups, each result is the median of seven samples of 40 graph replays.

| Contact traversal | Frame time | Change |
| --- | ---: | ---: |
| Legacy capacity scan and dual launch | 8.402016 ms | baseline |
| Batch-gated gather and active-prefix dual | 7.998294 ms | 4.81% faster |

This measures the combined gather/dual dispatch, not either kernel in
isolation. It supports retaining the large-color gate but does not justify
enabling the path for smaller single-scene models.

**Rejected or deferred work.** The two-particle surface tile is rejected
because MJVBDV2's existing surface-only tile was faster. Tet-only elasticity
and split SDF value/gradient helpers are not ported: open PR review identified
respectively a stale construction-time specialization after material edits and
a possible missed contact for nonuniformly scaled mesh SDFs. Full-surface
feature-AABB rejection remains the highest-value follow-up, but it requires a
private MJVBDV2 full-contact pipeline; changing Newton's shared
`CollisionPipeline` would violate the migration boundary recorded above.

**Correctness evidence.** Focused CPU and CUDA tests compare gather
force/Hessian output against both legacy private VBD scatter kernels over
empty, partial, full, and overflow-clamped active prefixes. Separate CPU and
CUDA tests cover the dual update for both private rigid kernels. CUDA tests
also verify both sparse pair builders are shape-major and that a small color
model retains the legacy dispatch. A forced-dispatch integration test executes
both private solvers and captures/replays their gather paths in a CUDA Graph.
Another test verifies two isolated worlds preallocate two compatible records,
not the four-record Cartesian product. Existing MJVBDV2 CPU and CUDA Graph
regressions remain required before commit.

**Decision.** Retain the shape-major layout and the batch-gated gather/dual
paths. Do not enable the latter for ordinary single-world demos without new
controlled evidence. Revisit the threshold only with additional MJVBDV2
multiworld A/B measurements across batch sizes; do not infer a per-scene
speedup from PR 3995's 1,024-environment aggregate result.

### 2026-09-04: reject fixed-point iteration accelerators

VBD's fixed particle-color order gives every nonlinear Gauss--Seidel sweep the
same propagation direction.  A prototype alternated forward and backward
color traversal between iterations without changing launch count, per-color
contact refresh, DAT ordering, or CUDA Graph topology.  Short surface tests
looked promising, but complete trajectories and tetrahedral tests rejected
the change.  All prototype code was removed.

An NVIDIA GeForce RTX 5060 Ti A/B of the 62,770-tet Armadillo crusher scene
showed that reducing from 10 to 8 iterations increased the position/edge error
from 4.714 mm/0.036 mm with forward traversal to 7.207 mm/0.080 mm with
alternating traversal.  Restricting the prototype to surface-only systems
avoided that regression, but did not solve the long-trajectory surface issue.

Surface results were measured by comparing a lower iteration count with a
higher-count reference using the same traversal policy.  These are evolving
nonlinear trajectories, so the position RMS is a convergence proxy rather
than an error against a known analytic solution.

| Scene and frames | Traversal | Candidate/reference | Wall time | Position RMS | Edge-length MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| T-shirt, 100 | forward | 12/20 | 6.522 s | 6.137 mm | 0.155 mm |
| T-shirt, 100 | alternating | 12/20 | 6.674 s | 4.477 mm | 0.126 mm |
| T-shirt, 100 | alternating | 10/20 | 5.883 s | 5.789 mm | 0.160 mm |
| Plastic bag, 120 | forward | 8/12 | 4.751 s | 21.164 mm | 0.187 mm |
| Plastic bag, 120 | alternating | 8/12 | 4.695 s | 10.584 mm | 0.074 mm |
| Plastic bag, 120 | alternating | 7/12 | 4.480 s | 13.974 mm | 0.098 mm |

The 300-frame T-shirt check is the limiting case.  Alternating 12/20 produced
11.248 mm position RMS and 0.325 mm edge MAE in 22.736/33.653 s.  Dropping to
11 iterations increased those errors to 15.477 mm and 0.354 mm, while 10
iterations reached 16.225 mm and 0.392 mm in a separate run.  Therefore the
T-shirt demo remains at 12 iterations.  The shorter tests demonstrate better
surface propagation, but do not justify a global iteration-count reduction.

The complete 240-frame plastic-bag manipulation reversed the short-window
result.  Forward 8/12 produced 9.189 mm position RMS and 0.109 mm edge MAE;
alternating 8/12 produced 25.060 mm and 0.204 mm.  Their eight-iteration wall
times were respectively 10.219 s and 10.241 s, so there was no performance
gain to offset the accuracy loss.  Alternating 7/12 reached 22.365 mm and
0.188 mm in 9.815 s and was also rejected.

The following accelerators were implemented and removed during the same
investigation:

- A six-DOF translation/rotation cluster basis improved synthetic twist and
  bend tests, but on the T-shirt its 12-iteration position RMS regressed from
  7.41 mm to 7.79 mm against the same 20-iteration reference.  Smoothed
  aggregation and post-smoothing did not recover the loss.  The approximate
  rotational operator was not accurate enough to retain.
- A device-side adaptive `capture_while` loop stopped the T-shirt at six
  iterations with a loose relative-update threshold and saved about 12%, but
  introduced 2.504 mm RMS and 0.119 mm edge error relative to fixed 12.  Strict
  thresholds ran all 12 iterations and were slower because of reductions and
  conditional-graph overhead.
- Successive over-relaxation at 1.10 let a 10-iteration, 100-frame T-shirt run
  reach 5.870 mm RMS versus 6.292 mm for ordinary 12 iterations, and reduced
  wall time from 6.668 s to 5.610 s.  Stronger settings produced NaNs in a
  multi-step nonlinear cloth stress test.  Without a cheap monotonicity
  safeguard, fixed SOR is not sufficiently general and was removed.
- Applying the translation-only coarse correction both mid-solve and at the
  end improved the 100-frame 10/20 T-shirt proxy to 5.299 mm, but regressed to
  17.817 mm over 300 frames.  Multiple applications amplify coarse-operator
  error during long contact sequences, so the retained multilevel path still
  performs one final correction.

**Decision.** Reject all four prototypes and retain the original forward
color order, fixed iteration count, one final translation-only coarse
correction, and unit local Newton step.  Do not lower a demo's iteration count
solely from a short trajectory; require its complete manipulation sequence to
meet both position and strain metrics.

### 2026-09-04: reject adaptive sweep plus multilevel control

A second adaptive prototype tested block-level CUDA Graph conditionals rather
than the earlier `capture_while` maximum-update loop.  It retained the configured
12/20-iteration ceiling, grouped two complete color sweeps per conditional
branch, required two consecutive passes, and always ran one coarse correction
after the conditional blocks.  The first checkpoint initialized history and
could not stop the solve.

The prototype measured the untruncated local Newton candidate written by the
elasticity solve, the change in the force-element normal-correction scale, a
device flag set by any significant DAT or maximum-displacement truncation, and
pneumatic volume change/clamp validity.  All decisions and statistics stayed on
the device.  Warp 1.17's `capture_if` was verified separately to skip every CUDA
node in a false branch on the local CUDA 12.9 system; this was a real graph-node
skip, not a per-kernel inactive return.

For scenes containing VBD-owned dynamic rigid bodies, an inactive particle
branch continued the original number of AVBD and body-particle dual iterations.
This avoided using a particle residual as a rigid convergence test.  The final
coarse particle correction therefore saw the latest rigid pose.  CPU,
differentiable, deterministic, uncaptured, and non-multilevel paths retained the
ordinary fixed loop throughout the prototype.

The captured T-shirt scene used 10 substeps per frame and 12 maximum particle
iterations.  Timings below cover graph replay after construction and capture on
an NVIDIA GeForce RTX 5060 Ti.  Contact atomic ordering is non-deterministic: a
100-frame fixed/fixed repeat differed by 1.464 mm position RMS and 0.046 mm
edge-length MAE, so position differences near that level are noise rather than
an adaptive accuracy claim.

| T-shirt run | Fixed | Adaptive | Mean sweeps | Position RMS | Edge MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 frames, local 0.005, contact check disabled | 6.608 s | 6.058 s | 9.996/12 | 1.796 mm | 0.055 mm |
| 300 frames, local 0.005, contact 0.005 | 22.364 s | 23.551 s | 11.958/12 | 11.523 mm | 0.230 mm |

The loose 100-frame diagnostic showed that node skipping can save 9.1% when
contact convergence is deliberately ignored: 298 of 1,000 substeps stopped at
six sweeps.  It is not a valid production configuration.  With all checks
enabled over 300 frames, only 21 of 3,000 substeps stopped early and the added
checkpoint/conditional nodes made the run 5.3% slower.  The long-horizon state
difference is reported for completeness, but nonlinear non-deterministic
trajectory divergence prevents attributing it to the 21 early stops.

The full-contact plastic-bag/rod scene used an eight-sweep ceiling.  During a
30-frame captured check, all 174 measured substeps ran all eight particle
sweeps.  Fixed and adaptive wall times were 1.503 s and 1.481 s; the 1.4%
difference is treated as noise because no particle work was skipped.  Position
RMS and edge-length MAE were 0.164 mm and 0.003 mm.

**Decision.** No-Go; all runtime, kernel, option, test, and benchmark prototype
code was removed.  Conservative residuals correctly refuse to stop during the
representative continuously driven contact phases, so per-checkpoint graph work
adds overhead without removing sweeps.  Do not restore this design merely by
loosening contact or local thresholds.  A future tiered-graph attempt first
needs evidence that a cheap prior-substep signal predicts sustained 4/8/12/20
sweep regimes; otherwise graph selection only moves the same overhead outside
the loop.

### 2026-09-05: reject twelve-DOF affine tet clusters

A complete twelve-DOF tetrahedral coarse basis was prototyped to address the
six-DOF basis limitation. Each volumetric cluster represented

```text
u_i = t_c + omega_c x (x_i - x_bar_c) + S_c (x_i - x_bar_c),
```

where the symmetric `S_c` contributed six strain modes in addition to three
translations and three infinitesimal rotations. Surface clusters in mixed
models remained three-DOF translation clusters. Restriction included mass and
contact Hessians; tet groups used the projected Neo-Hookean Galerkin operator
and objective damping metric; a persistent block-Cholesky PCG solve preserved
fixed CUDA Graph topology. A synthetic symmetric-strain test confirmed that
the affine basis represented a correction that the rigid basis could not.

The representative measurements did not establish a general production
benefit:

| Scene and window | Six-DOF rigid | Twelve-DOF affine | Accuracy observation |
| --- | ---: | ---: | --- |
| 62,770-tet Armadillo, 240 frames | 201.829 ms/frame | 300.759 ms/frame | Position RMS versus ten sweeps: 4.727 -> 3.319 mm; maximum: 20.583 -> 11.533 mm; edge MAE: 0.0530 -> 0.0559 mm |
| Soft tet cube plus cloth bag, 480 frames | 43.274 ms/frame | 57.224 ms/frame | Bag RMS versus twelve sweeps: 0.758 -> 0.862 mm; soft-cube RMS: 0.465 -> 0.557 mm |

The Armadillo position result was consistent with the larger coarse space, but
the trajectory uses nondeterministic contact and therefore the single-run
percentage is only a convergence proxy. The mixed scene showed no quality
improvement and was 32.2% slower. Exact 12-by-12 tet-group assembly has four
times as many scalar entries as 6-by-6 assembly. Increasing cluster size from
32 to 64 and reducing PCG from four iterations to two made Armadillo slower
still at 349.901 ms/frame because it reduced assembly parallelism. A
16-particle cluster variant reached 325.720 ms/frame and ended with a poor
coarse residual ratio of 2.598.

The extra kernels also increased the cold `particle_multilevel` CUDA compile
from roughly three seconds to roughly ten seconds on the test machine, even
when the six-DOF path was selected, because both paths occupied one Warp
module.

**Decision.** No-Go; the affine kernels, storage, solver option, demo switches,
and affine-specific tests were removed. Retain the six-DOF production path and
this result. A future affine attempt must isolate compilation and demonstrate
a benefit on both a large tet model and the mixed tet/cloth acceptance scene;
an Armadillo-only accuracy result is insufficient.

## Reconstructed retained history

The following changes predate standardized performance logging. They remain
because they either prune an entire absent module or support a required graph
execution contract. Their throughput effect must be re-measured before using
it as a performance claim.

### Scene-specialized dispatch

Commit `89002b52` added construction-time `pure_mujoco` and
`kinematic_passthrough` paths. Articulation-only scenes do not allocate or run
VBD. Later dispatch work selects `vbd_soft/` for ordinary particle-only scenes
and the complete `vbd/` implementation only for dynamic VBD rigid bodies,
pneumatics, or explicitly full contact.

### Optimized soft-particle execution

Commit `bb2791da` retained the self-contact activity flag on the device for
CUDA graph compatibility, selected current color groups when self-contact is
inactive, skipped sparse soft-contact stages in uncaptured execution when the
active count justified the readback, and selected scalar elasticity for color
groups too small to occupy a CUDA tile. These choices apply to `vbd_soft/`;
they must not be assumed beneficial in complete `vbd/` without an A/B test.

### Active collision-buffer traversal

Commit `5ebbf77f` changed VT/EE processing to use detector active counts rather
than iterating the full allocated collision-buffer capacity. The same bounded
traversal was synchronized across Newton VBD, MJVBD, and both MJVBDV2 VBD
implementations. Preserve capacity-sized graph allocation, but do not restore
capacity-sized per-thread loops.

### Runtime material selection in one graph

Commit `a299dfc1` added a device material table and selector so a captured graph
can change cached soft-contact material phases in stream order. Its value is
avoiding a CPU/GPU round trip and graph re-recording in phase-changing demos;
it is not expected to make a fixed-material graph faster.

### MuJoCo sleeping and shared collision queries

Commit `24b3479e` made MuJoCo sleeping optional for the `pure_mujoco` backend
and brought in shared SDF/half-space collision-query improvements. Sleeping is
deliberately rejected in the one-way coupled backend because VBD has no wake
signal for MuJoCo. Measure collision-query changes only in scenes that actually
select the corresponding full collision path and shape types.

## Next measurement targets

Re-profile the complete `vbd/` path after dense rigid-side accumulation before
choosing another hotspot. Pneumatic cavity-volume updates and compact
self-contact/color traversal remain candidates, but both must preserve
per-color Gauss-Seidel pressure and contact semantics. Any implementation must
remain entirely under `newton/_src/solvers/mjvbd_v2/`; shared collision and
simulation modules are inputs, not optimization targets. Do not infer a gain
for particle-only cloth or coupled articulation scenes from the rigid-ball bag
measurement because those backends do not execute this dense rigid path.
