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
| 2026-09-07 | Keep multilevel correction off the rotating fixed-boundary cloth twist | `mjvbd_v2_cloth_twist` CUDA surface solve | **Retained, gated** | Disabling only multilevel removed the corner spike (transverse/inward ratio 1.760 to 0.858) while the remaining three-sweep schedule reduced a matched 300-frame mean from 17.791 to 9.830 ms/frame (44.74%) |
| 2026-09-07 | Honor runtime Chebyshev disablement | Both private MJVBDV2 VBD backends | **Retained** | A CPU regression fails in both backends without the boolean gate and passes with it; this restores genuinely ordinary guarded fallbacks and benchmark references |
| 2026-09-07 | Batch independent-set colors into rotating topology-aware Jacobi partitions | CUDA `vbd_soft/` cached triangle-surface solve | **Retained, opt-in** | The matched W1 T-shirt 900-frame mean fell from 28.244 to 18.672 ms/frame (33.89%); after fixing the disabled-Chebyshev reference, the former ordinary-30 accuracy claim is withdrawn because the corrected ratios are 1.643 / 1.632 / 2.132 |
| 2026-09-07 | Split edge-edge and vertex-triangle self-contact accumulation into dedicated kernels | CUDA `vbd_soft/` surface self-contact solve | **Rejected, reverted** | A matched 100-frame W1 T-shirt CUDA Graph run regressed from 22.220 to 23.493 ms/frame (5.73% slower); the extra launch and VT grid outweighed any register-pressure reduction |
| 2026-09-05 | Promote guarded cached Chebyshev to the W1 T-shirt default | T-shirt final00 example | **Retained** | The final DAT-safe policy passed 92 focused regressions and a complete 900-frame task, improved common-state accuracy over `cached13`, and reduced the same-run 900-frame mean by 7.59% |
| 2026-09-05 | Promote the validated `cached13` policy to the W1 T-shirt default | T-shirt final00 example | **Superseded** | After merging `20d5d8fb` with the collision-aware Chebyshev work, an exact configuration assertion and the complete 900-frame null-viewer test passed; the later guarded policy keeps `cached13` selectable |
| 2026-09-05 | Cache fixed bending anchors and DAT geometry; use 13 contact-free relaxed sweeps | CUDA `vbd/` and `vbd_soft/`; T-shirt `cached13` mode | **Retained, opt-in** | Two 900-frame runs reduced frame time 5.01% / 6.34% versus 12 + graph, with lower position/edge errors at all nine checkpoints; the native common-state mean position error fell 38.04%; defaults unchanged |
| 2026-09-04 | Project the current particle Hessian into a block-sparse aggregate Galerkin operator | CUDA `vbd/` and `vbd_soft/` cloth/shell paths | **Retained, opt-in** | On the 6,436-particle T-shirt, 12 sweeps plus Galerkin reduced the 300-frame position/edge error versus 30 sweeps by 43.7%/36.3% relative to the existing graph operator, with 5.1% overhead over that operator and 48.9% lower frame time than 30 ordinary sweeps |
| 2026-09-05 | Add collision-aware Chebyshev acceleration to particle VBD sweeps | CUDA `vbd/` and `vbd_soft/` particle paths | **Retained, opt-in** | The 6,436-particle W1 T-shirt fold reduced its validated policy from 12 to 8 sweeps and improved paired 300-frame GPU and wall time by 23.3% while keeping repeated-run RMS error ranges comparable; reducing the timestep count or using only 7 sweeps was rejected |
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

### 2026-09-05: retain contact-aware Chebyshev particle sweeps

**Motivation.** The W1 T-shirt fold originally used 20 ordinary VBD sweeps per
substep. The guarded multilevel correction reduced that to 12 fine sweeps plus
one coarse correction, but fine-level convergence remained the dominant
solver cost. Reducing substeps directly is numerically different from reducing
nonlinear iterations: [Small Steps in Physics Simulation][small-steps]
predicts that fewer, larger timesteps generally lose contact and
time-integration accuracy even when each step receives more solver iterations.
The newer [Augmented VBD][avbd] method is a stronger candidate for hard
constraints and extreme stiffness ratios, but migrating the cloth contact
objective to augmented-Lagrangian state would be a substantially larger
algorithmic change. Collision-aware acceleration from the [original VBD
method][vbd-paper] is the controlled first step because it preserves the
existing local objective and contact model.

**Implementation.** Both private particle VBD implementations now expose an
opt-in `particle_chebyshev_spectral_radius`. The recurrence matches the
Chebyshev semi-iterative acceleration described in the original VBD paper:

```text
omega_1 = 1
omega_2 = 2 / (2 - rho^2)
omega_n = 4 / (4 - rho^2 omega_(n-1))
x_n = x_(n-2) + omega_n (x_bar_n - x_(n-2)).
```

The accelerator stores two fine iterates and adds the extrapolation through
the existing displacement/DAT path, so self-contact truncation and CUDA Graph
topology remain intact. Any particle with a nonzero contact-force Hessian in
any sweep of the current timestep is permanently excluded from acceleration,
including contacts detected in the first unit-weight sweep. This implements
the VBD paper's collision-aware safety rule instead of applying a global SOR
factor to contacting particles. Spring Hessians use the same accumulation
buffer and therefore conservatively disable acceleration for their endpoints.
A per-sweep particle-radius clamp bounds the candidate before DAT. The feature
is disabled by default and for `requires_grad` models; therefore existing
solver and demo paths do not change unless they opt in.

**Benchmark.** Measurements used an RTX 5060 Ti, CUDA Graph replay, 6,436
particles, 60 displayed frames/s, ten substeps, and the scripted W1 bimanual
fold. Wall time excludes construction and graph capture. Accuracy compares the
final particle state to a 20-ordinary-sweep run at the same substep count. The
300-frame window covers grasp, lift, transport, placement, release, and most
of the second fold.

| Policy | Wall time | GPU stream time | Position RMS | Position p95 | Position maximum | Velocity RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 ordinary sweeps, reference | 111.115 ms/frame | not recorded | 0 | 0 | 0 | 0 |
| 12 sweeps + multilevel | 85.210 ms/frame | 85.212 ms/frame | 10.566 mm | 21.231 mm | 47.176 mm | 0.05781 m/s |
| 8 sweeps + multilevel + Chebyshev `rho=0.90` | 65.325 ms/frame | 65.327 ms/frame | 11.695 mm | 23.316 mm | 53.756 mm | 0.07219 m/s |
| 7 sweeps + multilevel + Chebyshev `rho=0.90` | 61.130 ms/frame | not recorded | 19.232 mm | 43.626 mm | 74.044 mm | 0.08045 m/s |
| 9 substeps, 8 sweeps + multilevel + Chebyshev `rho=0.90` | 61.197 ms/frame | not recorded | 17.995 mm | 37.552 mm | 78.988 mm | 0.07567 m/s |

The retained policy is 23.3% faster than the paired existing 12-sweep policy
in both GPU and wall time and 41.2% faster than the earlier 20-sweep reference.
Atomic contact order makes the trajectory nondeterministic: across repeated
300-frame runs, the retained policy produced 10.180--11.695 mm position RMS,
while the 12-sweep path produced 10.566--11.732 mm. Thus the small paired-run
error increase is within the observed trajectory spread and is not reported
as an accuracy improvement. A separate 100-frame spectral-radius scan rejected
`rho=0.95` and selected `rho=0.90`. The default 900-frame run completed at
73.582 ms/frame with finite positions and velocities and a final maximum
particle speed of 0.0282 m/s.

**Substep decision.** No-Go for reducing the T-shirt default below ten
substeps. Nine substeps was only 6.3% faster in the 300-frame window while its
RMS error rose 53.9% versus the paired retained run. Eight substeps with 20
ordinary sweeps had already produced 21.169 mm RMS error, so adding more
nonlinear work cannot recover the lost temporal resolution efficiently. Seven
fine sweeps is also rejected because its 6.4% extra speed over the retained
policy increased RMS error by 64.4%.

**Validation and scope.** Unit tests cover option validation, exact recurrence,
default-off allocation, and CUDA Graph execution in both private VBD solvers.
The W1 example explicitly selects eight sweeps and `rho=0.90`; its guarded
multilevel fallback still captures up to 20 ordinary sweeps. Other examples
retain their prior iteration behavior. Treat `rho` as a scene-policy parameter,
not a universal default, until additional cloth, tet, pneumatic, and
self-contact scenes establish their own safe values.

Post-change regression ran 68 focused tests covering both particle solvers,
multilevel, contact optimizations, pneumatic state, kinematic passthrough,
pure MuJoCo, and coupled dispatch. The unchanged W1 plastic-bag-and-rod example
passed its complete 240-frame test. Three-frame CUDA Graph smoke runs passed
for that scene, the pneumatic inflatable bag, and the mixed soft-tet-cube plus
cloth-bag scene; the 15,228-particle Armadillo tet scene passed a two-frame
Graph smoke run. These examples leave Chebyshev disabled and therefore
allocate no Chebyshev state or accelerator Graph nodes.

[small-steps]: https://matthias-research.github.io/pages/publications/smallsteps.pdf
[vbd-paper]: https://www.cemyuksel.com/research/papers/vbd-siggraph2024.pdf
[avbd]: https://www.cemyuksel.com/research/papers/Augmented_VBD-SIGGRAPH25.pdf

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

### 2026-09-04: retain an aggregate Galerkin coarse operator

**Historical measurements:** this prototype still contained a membrane
off-diagonal assembly error. The 2026-09-05 follow-up below corrects it and
supersedes these numbers for the retained implementation.

**Problem.** The original multilevel correction restricts a frozen local
Newton residual into piecewise-constant translation aggregates, but solves it
with a scalar mass-plus-topology Laplacian. Parameter scans over cluster size,
coupling, relaxation, PCG count, and repeated corrections improved short runs
but could not consistently approach a 30-sweep reference. The limitation was
the coarse operator rather than its default parameters.

**Implementation.** A new opt-in `particle_multilevel_operator="galerkin"`
path assembles the block-sparse aggregate projection (P^T H P) at the final
ordinary sweep. The diagonal starts with the exact local VBD blocks, including
inertia and diagonal contact terms. Off-diagonal blocks use the same projected
membrane, dihedral-bending, damping, and spring Hessians as the fine solve.
Each unordered element pair writes a block and its transpose, halving element
evaluation work and keeping the matrix symmetric to floating-point atomic
ordering. Block-Jacobi PCG and the existing finite, residual, radius-clamp,
DAT, and conditional-fallback guards remain in force. The inexpensive
`"graph"` operator remains the default.

Uncaptured fallback now reads the device status and takes an ordinary Python
branch. `wp.capture_if()` is used only while a CUDA Graph is actually being
captured, so eager execution does not depend on a conditional-capture API and
CPU-disabled multilevel configurations retain the original VBD path.

**Measurements.** The primary benchmark was the scripted 6,436-particle,
12,736-triangle, 19,174-edge W1 T-shirt scene with self-contact, 10 substeps,
CUDA Graph replay, Warp 1.17.0.dev20260807, CUDA Toolkit 12.9, Driver 13.2,
and an NVIDIA GeForce RTX 5090 D v2. Runs were consecutive in one process and
included all 300 displayed frames.

| Configuration | Mean frame time | Position RMS vs. 30 sweeps | Edge-length MAE vs. 30 sweeps |
| --- | ---: | ---: | ---: |
| 30 ordinary sweeps | 88.866 ms | reference | reference |
| Independent 30-sweep repeat | 89.470 ms | 2.004 mm | 0.082 mm |
| 12 sweeps + graph operator | 43.192 ms | 17.096 mm | 0.313 mm |
| 12 sweeps + Galerkin operator | 45.384 ms | 9.624 mm | 0.199 mm |

The Galerkin path adds 5.1% over the previous correction, reduces its
position error by 43.7% and edge error by 36.3%, and remains 48.9% faster than
30 ordinary sweeps. A separate 100-frame run measured 40.867 ms versus
79.815 ms for 30 sweeps; its Galerkin position/edge errors were 3.833/0.112 mm
versus 8.225/0.175 mm for the graph operator. Contact atomics make independent
long trajectories diverge, so the 30/30 repeat is reported as the noise floor
and endpoint position RMS is not treated as an analytic error bound.

The configured Galerkin T-shirt example completed its full 900-frame test.
Focused tests cover both private VBD implementations, CUDA Graph capture and
replay, block-pattern and runtime-matrix symmetry, fixed anchors, transactional
fallback, and equality between the Galerkin element diagonal blocks and the
fine VBD projected Hessians.

**Rejected variants.** One- and two-step smoothed aggregation with a scalar
topological operator was slower and less accurate. Replacing that operator
with the physical Hessian improved it only marginally in an offline exact
coarse solve. Applying corrections mid-sweep, using two or three coarse cycles,
and adding post-smoothing helped a contact-free cantilever but regressed the
real T-shirt contact trajectory. Bounded Anderson and Chebyshev fixed-point
prototypes likewise failed to improve the low-sweep cantilever consistently.
A 6-DOF aggregate subspace with three translation and three infinitesimal
rotation coordinates was also implemented and tested, rather than inferred.
On the 100-frame T-shirt comparison, its best tested 12-sweep result took
48.479 ms/frame with 4.472 mm position RMS and 0.109 mm edge MAE; the retained
translation Galerkin path took 39.492 ms/frame with 3.791 mm and 0.096 mm.
The affine coarse residual was also substantially harder to reduce. None of
these rejected prototypes remains in source.

**Decision.** Retain the physical Galerkin operator as an explicit accuracy
option and use it for the measured T-shirt example at 12 sweeps, with a
30-sweep fallback on rejected corrections. Do not enable it globally or claim
30-sweep equivalence: it materially closes the gap, but the remaining
long-trajectory error is above the measured repeat floor. The next accuracy
step should include contact-aware off-diagonal blocks or a guarded nonlinear
reduced solve, not another sweep/relaxation parameter grid.

### 2026-09-05: correct Galerkin cross blocks and compare against 30 sweeps

**Correctness first.** Reviewing the full element matrix, rather than only its
3-by-3 vertex diagonals, exposed a transposed outer product in the projected
membrane cross blocks. In the cross-product Jacobian term, `D_i.T @ D_j`
requires `outer(w_j, w_i)`, not `outer(w_i, w_j)`. Both expressions agree on
the diagonal, which is why the previous diagonal-equality test missed this.
At stretch factors 1.5, 2, and 4, the old assembled element matrix had negative
eigenvalues despite its positive-semidefinite construction being intended.
The new symmetry, translation-nullspace, and positive-semidefiniteness test
failed at all three stretches before the fix and passed after it. Fine VBD
forces, local Hessians, materials, and collision geometry are unchanged.

The Galerkin PCG now rejects nonpositive preconditioned curvature or search
curvature even when residual-reduction validation is disabled. A two-cluster
indefinite-system regression failed before this guard and passed afterward.
A separate positive-definite block test matches a dense NumPy solve and
checks that a valid correction is not spuriously rejected. Nonzero status
prevents prolongation from committing the candidate and selects the existing
fallback when configured. Normal DAT still truncates committed displacements;
this is not a new nonlinear contact, cavity-volume, or energy acceptance test.

**Reproducible trajectory benchmark.** Run:

```bash
uv run scripts/benchmark_mjvbd_v2_multilevel.py --frames 300 --sweeps 8 12
```

The script runs independent 30-sweep reference/repeat trajectories, ordinary
low-sweep trajectories, and graph/Galerkin candidates. It reports JSON records
with whole-frame times, errors at frames 100/200/300, finite checks, and the
example's `test_final()`. Construction, compilation/capture, and checkpoint
readbacks are outside the timer; all simulation and realtime IK frames are
included. Device and scene are the same RTX 5090 D v2 / 6,436-particle T-shirt
configuration as above. The CUDA guard was left running throughout.

| Configuration | Mean frame time | RMS at 100 | RMS at 200 | RMS at 300 | Edge MAE at 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 ordinary sweeps | 93.303 ms | reference | reference | reference | reference |
| Independent 30-sweep repeat | 93.504 ms | 2.280 mm | 2.029 mm | 13.080 mm | 0.194 mm |
| 8 ordinary sweeps | 32.124 ms | 16.634 mm | 23.185 mm | 32.508 mm | 0.561 mm |
| 8 sweeps + graph | 34.256 ms | 9.936 mm | 13.334 mm | 26.988 mm | 0.453 mm |
| 8 sweeps + Galerkin | 37.958 ms | 5.676 mm | 5.237 mm | 15.225 mm | 0.300 mm |
| 12 ordinary sweeps | 43.869 ms | 11.360 mm | 13.035 mm | 16.546 mm | 0.393 mm |
| 12 sweeps + graph | 45.553 ms | 7.692 mm | 8.173 mm | 16.586 mm | 0.325 mm |
| 12 sweeps + Galerkin | 47.766 ms | 3.768 mm | 4.819 mm | 10.935 mm | 0.207 mm |

The corrected 12-sweep Galerkin run uses 48.8% less frame time than 30 ordinary
sweeps, at an 8.9% cost over ordinary 12 sweeps. Eight sweeps with Galerkin are
faster, but are not promoted to the example default: earlier checkpoints are
still appreciably farther from the reference. The 13.080 mm 30/30 difference
also demonstrates why a favorable endpoint alone cannot establish equivalent
accuracy. Checkpoint runtime statuses were zero for these candidates; these
samples are not an all-substep rejection count or a guarantee about contacts.

**Same-substep accuracy diagnostic.** At the end of sweep 12, save the
reference particle positions and displacements, apply the candidate correction
using that exact contact state, save its result, restore both arrays, then
continue ordinary sweeps 13 through 30. This prevents candidate trajectories
from influencing subsequent reference steps. The rigid bodies are externally
driven; the candidate only modifies the two restored physical arrays and
temporary coarse/DAT buffers. Sample the final substep of each of 300 frames.
This comparison does not time the candidate or pretend to sample every
substep. The candidate is measured before extra fallback sweeps.

| Candidate from identical state | Mean position RMS | 95th percentile RMS | Mean edge MAE |
| --- | ---: | ---: | ---: |
| Ordinary 12 sweeps | 0.014872 mm | 0.025311 mm | 0.001359 mm |
| 12 sweeps + Galerkin | 0.015477 mm | 0.042187 mm | 0.001944 mm |

Two of 300 candidate samples were rejected; these samples stay at the
12-sweep state in this diagnostic. Production fallback would instead run
through sweep 30. Even without interpreting long-trajectory divergence, the
candidate does not consistently improve the local approximation. In
particular, the position tail and edge error regress. A reproducible version
of this diagnostic is included in the benchmark:

```bash
uv run scripts/benchmark_mjvbd_v2_multilevel.py --same-substep --frames 300 --sweeps 12
```

Re-running the checked-in script independently reproduced the conclusion:
ordinary 12 sweeps had mean/P95 position RMS 0.014807/0.024515 mm and edge
MAE 0.001366 mm, versus 0.016327/0.053890 mm and 0.001986 mm for Galerkin.
Again, two sampled candidates were rejected. This repeat does not support
treating the favorable independent-trajectory endpoint as an accuracy upgrade.

**Additional subspace/solver experiments, not retained.** The following were
implemented as isolated prototypes without changing the fine forces:

- A metric Gauss-Newton membrane tangent and an exact membrane tangent were
  compared with the projected tangent. Neither improved the real contact
  trajectory consistently. Those preliminary runs preceded the cross-block
  fix and are not used to validate the retained implementation.
- A nonlinear reduced correction re-evaluated the original fine residual,
  accepted only a decrease in a frozen local inverse-Hessian norm, and tried
  half/quarter steps before rollback. After the cross-block fix, a 150-frame
  run measured 44.105 ms / 5.266 mm RMS for ordinary Galerkin, versus
  47.299 ms / 6.901 mm for one guarded cycle and 53.058 ms / 5.881 mm for two.
  The two-cycle edge MAE improved from 0.155 to 0.138 mm, but position error
  and runtime did not improve together. This norm is a diagnostic, not the
  frictional-contact objective or a complete nonlinear safety certificate.
- A contact-compliance-weighted aggregate basis used
  `B_i = I - 0.9 inverse(H_i) H_contact_i` and transformed restriction,
  membrane/bending blocks, and prolongation consistently. A separate
  150-frame comparison measured position RMS 8.831 mm / edge MAE 0.205 mm,
  versus 7.394 mm / 0.217 mm for ordinary Galerkin. This is a tradeoff, not
  evidence of better accuracy; it is not retained.

The existing 6-DOF aggregate and scalar-smoothed subspace experiments are also
not reintroduced. More coarse cycles, looser guards, or a changed constitutive
force are not accepted as substitutes for improving the approximation.

**Decision.** No-Go as a default T-shirt accuracy upgrade. Keep the corrected
Galerkin operator as an explicit experiment, but restore the T-shirt's
pre-experiment graph-operator configuration, including its original fallback
budget. Do not claim equivalence to 30 sweeps or reduce the demo to eight
sweeps. The benchmark and mathematical regressions are retained; the
inconclusive nonlinear and contact-basis prototypes are not part of the solver
API. Subsequent work needs a coarse model/subspace that improves same-state
contact and deformation errors together, not merely a more favorable endpoint.

**Verification.** All 15 focused multilevel tests pass, covering CPU-disabled
operation, both private CUDA VBD backends, graph capture/replay, eager fallback
with `wp.capture_if` patched to fail if called, and the new matrix regressions.
The restored default T-shirt configuration passes its 900-frame `--test` run.
The experimental Galerkin configuration also passes a separate 900-frame
finite-state/`test_final()` run (49.058 ms/frame; all nine sampled final-substep
statuses zero). This long run is not a 900-frame comparison to 30 sweeps.
These are finite-state/material/API checks, not visual certification that every
fold or grasp is identical. Changed-file pre-commit hooks pass. No remote
commit or push was made for this experiment.

### 2026-09-05: compare contact-free surface relaxation to the original 20 sweeps

**Target.** Use ordinary 20 sweeps as the accuracy reference and the existing
12-sweep-plus-one-graph-correction T-shirt as the speed/accuracy baseline.
`889ae3d4^` confirms that this example originally used `VBD_ITERATIONS = 20`.
Use current identical materials, geometry, realtime IK, substeps, and collision
parameters for all variants; this is not a comparison between different
historical physics implementations. The exact demo baseline has a **5%**
particle-radius coarse cap, not the 20% cap used in earlier operator scans.

**Rejected before retention.** Isolated 300-frame same-substep comparisons
tested corrected translation Galerkin, post-smoothing, and a genuinely larger
four-dimensional per-aggregate basis: three translations plus a normalized
local-Newton-residual deviation from the aggregate mean. The latter assembled
consistent 4-by-4 projected blocks, restriction, and prolongation and used a
block-PCG solve. Neither the expanded basis nor fewer ordinary sweeps with
translation Galerkin improved position and edge errors over the exact current
baseline. Unconditional local SOR and alternating forward/reverse color order
also regressed contact errors. None of those prototypes was added to the
production solver.

**Retained experimental candidate.** Perform 12 ordinary color sweeps without
the terminal multilevel correction. In sweeps 2 through 9, multiply the local
Newton displacement by 1.3 only if the particle's accumulated contact/spring
Hessian is exactly zero. Sweep 1 and sweeps 10 through 12 remain ordinary.
Nonzero off-diagonal contributions also exclude acceleration; the production
predicate tests the complete matrix, not just its trace. Existing DAT and
displacement bounds still apply after each color. This changes the numerical
iteration, not the constitutive force, stiffness, damping, friction, collision
geometry, or timestep. A constrained row is not extrapolated locally; this
does not mean its later trajectory is independent of neighboring free rows.

The extra work is fused into the existing surface solve, with no scratch
position copies, extra per-color launches, host synchronization, or conditional
graph API. Both private MJVBDV2 backends expose
`particle_surface_relaxation=1.0` by default. Non-differentiable,
non-deterministic CUDA surface tiles can opt in; scalar/CPU and volumetric
paths remain ordinary. Scene-specific validation is required. The T-shirt
default remains the existing 12-sweep graph configuration.

**Prototype evidence (RTX 5090 D v2).** In a matched 300-frame full-trajectory
run, ordinary 20 sweeps took 66.424 ms/frame, its repeat 65.906 ms/frame,
the exact demo baseline 45.566 ms/frame, and the fused 12-sweep candidate
43.169 ms/frame (5.3% less time than the baseline).

| Frame | Baseline RMS vs. 20 | Candidate RMS vs. 20 | Baseline edge MAE | Candidate edge MAE |
| --- | ---: | ---: | ---: | ---: |
| 100 | 3.859 mm | 2.794 mm | 0.103 mm | 0.088 mm |
| 200 | 5.075 mm | 3.482 mm | 0.056 mm | 0.047 mm |
| 300 | 11.057 mm | 8.820 mm | 0.257 mm | 0.195 mm |

The 20/20 repeat position RMS was 0.989/1.795/8.256 mm at these checkpoints;
endpoint differences remain subject to nonlinear contact divergence. In a
separate 900-frame diagnostic, both candidates were forked from identical
substep states along an ordinary 20-sweep history. One final-substep sample
was taken per frame, avoiding inter-trajectory divergence:

| Candidate | Mean position RMS vs. 20 | P95 position RMS | Mean edge MAE |
| --- | ---: | ---: | ---: |
| Exact 12 + graph baseline | 0.008317 mm | 0.013298 mm | 0.000738 mm |
| 12 contact-free, final 3 ordinary | 0.005798 mm | 0.009367 mm | 0.000660 mm |

This is approximately 30% lower mean/P95 position error and 11% lower edge
error. An 11-sweep candidate was faster (40.106 ms/frame in the 300-frame
run), but its 900-sample edge error increased; it is not the retained demo
candidate. The residual-basis and SOR probes above were isolated prototypes;
the checked-in implementation and native demo modes are measured separately.

**Checked-in 900-frame timing/trajectory validation.** Ordinary 20 sweeps
took 67.992 ms/frame (repeat 67.806), the unchanged default mode took
47.613 ms/frame, and `contact-free` took 46.113 ms/frame: **3.15% less time**
than the existing 12 + multilevel baseline. This supersedes the short prototype
timing as the retained implementation's throughput estimate. All four runs
passed the finite-state and existing `test_final()` checks.

| Frame | Baseline RMS vs. 20 | Candidate RMS vs. 20 | Baseline edge MAE | Candidate edge MAE |
| --- | ---: | ---: | ---: | ---: |
| 100 | 4.545 mm | 3.182 mm | 0.117 mm | 0.092 mm |
| 200 | 4.423 mm | 6.170 mm | 0.055 mm | 0.054 mm |
| 300 | 11.376 mm | 10.270 mm | 0.251 mm | 0.196 mm |
| 400 | 14.424 mm | 17.392 mm | 0.157 mm | 0.153 mm |
| 500 | 14.524 mm | 18.886 mm | 0.158 mm | 0.142 mm |
| 600 | 14.730 mm | 19.163 mm | 0.149 mm | 0.136 mm |
| 700 | 15.068 mm | 19.258 mm | 0.144 mm | 0.135 mm |
| 800 | 15.460 mm | 19.340 mm | 0.141 mm | 0.132 mm |
| 900 | 16.440 mm | 18.309 mm | 0.141 mm | 0.133 mm |

Edge errors improve at all nine checkpoints, but long-trajectory position
errors do not. The independent 20/20 repeat itself differed by 5.102 mm at
frame 300, 13.259 mm at 500, and 9.419 mm at 900. This variability is a reason
to report common-history substep errors separately, not a reason to dismiss
the worse trajectory checkpoints or claim equivalent grasp/folding behavior.
Retain the mode as an explicit experiment; do not replace the demo default.

**Checked-in same-substep validation.** The native diagnostic completed 900
frames, taking one final-substep sample per frame along a common ordinary
20-sweep history. Neither trial feeds its positions into the reference's next
substep. The baseline correction was accepted in all 900 sampled substeps
(this is not a rejection-rate measurement over all substeps).

| Candidate | Mean position RMS vs. 20 | P95 position RMS | Mean edge MAE |
| --- | ---: | ---: | ---: |
| Exact 12 + graph baseline | 0.008289 mm | 0.013445 mm | 0.000736 mm |
| Checked-in contact-free mode | 0.005857 mm | 0.009165 mm | 0.000669 mm |

Mean position error is 29.3% lower, P95 is 31.8% lower, and edge error is
9.0% lower. These confirm a local convergence improvement, not uniformly
better full-trajectory accuracy. The 20-sweep reference is a practical
comparison target, not a proven exact solution.

**Verification.** All 20 focused multilevel/surface-relaxation tests pass.
The new invalid-factor regression failed before the option was implemented
and passes afterward. Coverage includes both private backends, exact
preservation of contact/anchor updates, nonzero off-diagonal contact Hessians,
the warm-up/final-sweep schedule, CUDA capture versus eager execution,
deterministic/differentiable opt-out, and unchanged CPU execution with
`wp.capture_if` patched to raise. The four independent 900-frame runs and the
900-frame same-substep diagnostic pass finite-state/`test_final()` checks.
These are not visual certification of equivalent folding or grasp retention.
No default was changed, and no commit or push was made for this experiment.

**Commands.** The default is unchanged. For visual comparison, append exactly
one mode to the existing T-shirt command:

```bash
uv run -m newton.examples.mjvbdv2.example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 --particle-solver-mode baseline
uv run -m newton.examples.mjvbdv2.example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 --particle-solver-mode reference20
uv run -m newton.examples.mjvbdv2.example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 --particle-solver-mode contact-free
```

Reproduce the native 20/reference-repeat/baseline/candidate comparison and
the independent same-substep diagnostic without a viewer:

```bash
uv run scripts/benchmark_mjvbd_v2_multilevel.py --compare-demo --frames 900
uv run scripts/benchmark_mjvbd_v2_multilevel.py --compare-demo --same-substep --frames 900
```

The existing CUDA guard stays active throughout these measurements; the
benchmark does not stop a demo, reset the GPU, or terminate the guard.

### 2026-09-05: cached13 improves the T-shirt speed/accuracy tradeoff

**Acceptance target.** Beat the existing 12 + graph mode in throughput and
accuracy against ordinary 20 sweeps, including the independent grasp/folding
trajectory, not only a same-state local norm. Keep defaults unchanged while
testing candidates. Do not disable collision, change materials, or weaken
the original displacement/DAT limits to obtain a speedup.

**Whole-sweep accelerators rejected.** Implemented isolated Chebyshev and
two-dimensional Anderson prototypes using the actual fine-sweep history;
neither requires a coarse matrix or mesh hierarchy. Anderson uses a fused
single-block Gram reduction, finite/conditioning/coefficient guards, and
existing DAT after mixing. These are experiments inspired by the
[VBD paper](https://arxiv.org/abs/2403.06321) and
[periodic Anderson paper](https://graphics.cs.uh.edu/wp-content/papers/2026/2026-HPG-PAA-PhysicalSim.pdf),
not a reproduction of the latter's 100-iteration tetrahedral test setup.
That paper explicitly reports contact-dominated failure cases; its speed and
energy claims do not establish a safe cloth-grasping acceleration here.

In a 300-frame common-history diagnostic (one final-substep sample/frame),
the exact baseline had mean position RMS **0.009461 mm**, edge MAE
**0.001152 mm**. Chebyshev at 10/12 sweeps and spectral estimates 0.7/0.9
increased both errors. Anderson with 11 sweeps / period 3 had
0.012399 / 0.001395 mm; damping its mixing to 0.5 still gave
0.012071 / 0.001359 mm. A 12-sweep period-2 trial and a contact-free
period-3 trial also failed to improve both metrics. None is added to the
production solver API.

**Investigate per-sweep cost instead.** A kernel-event diagnostic after 300
T-shirt frames (one subsequent eager frame, not a graph throughput estimate)
attributed approximately 13.17 ms to self-contact accumulation, 10.17 ms to
surface elasticity, 9.77 ms to planar truncation, 5.94 ms to body-particle
contact, 3.87 ms to applying truncation, and 3.17 ms to builtin fills. Thus
the earlier bending-only idea cannot account for the whole bottleneck.

Isolated bending prototypes evaluate only the selected angle gradient in
closed form and cache the fixed damping anchor angle once per substep.
Against the original matrix-chain derivative on 1,000 random hinges,
maximum relative force/Hessian differences were 2.35e-5 / 4.01e-5
(mean 1.65e-7 / 2.78e-7); algebraic equivalence is not bitwise equivalence.
An independent 300-frame run with cached bending and 14 contact-free SOR
sweeps took **44.481 ms/frame**, versus **45.103** for the baseline.
Position RMS at frames 100/200/300 was 1.821/2.895/5.211 mm versus
4.055/4.261/10.874 mm; corresponding edge errors were lower at all three
checkpoints. This modest short-run result is not sufficient for acceptance.

Subsequent experiments cache only the frozen closest-point geometry used by DAT
(division-plane offsets still respond to the current displacement), fuse
the reset of truncation factors with applying them, and separate EE/VT
self-contact kernels. The synthetic CPU cached-DAT comparison matched
the original truncation factors bitwise, including a degenerate pair.
The EE/VT split matched the original CPU result within tolerance but made
the full frame slower; it is not retained. Fourteen cached relaxed sweeps
had too small a long-run speed advantage. Thirteen was the best tested
compromise; this is not a claim of a globally optimal iteration count.

**Retained as opt-in, not a new default.** `--particle-solver-mode cached13`
uses 13 fine sweeps, no multilevel correction, contact-free relaxation 1.3,
and both caches. The first and last three sweeps are ordinary. Only rows
with exactly zero accumulated contact/spring Hessian are relaxed; ordinary
DAT and the original displacement bound still apply. Materials, contact
generation, robot trajectory, time step, and substeps are unchanged.

`particle_enable_surface_cache` caches only the fixed damping angle from
`particle_q_prev`, refreshed every substep, and uses the same bending force
law with a selected-vertex closed-form derivative. Current elastic angles
are still recomputed for every color. `particle_enable_truncation_cache`
refreshes closest-point/normal geometry after **every** self-collision
detection, including detections inside an iteration. Its division plane
still depends on the current displacement. Applying truncation also resets
the consumed factors to 1, eliminating the next color's full-array fill.
Neither feature is enabled for CPU, differentiable, or deterministic
execution. Surface scalar/volumetric solves are unchanged. DAT cache
allocation is capped at 256 MiB; capture-time growth raises instead of
silently reallocating pointers used by a CUDA Graph.

**Independent 900-frame trajectory measurements.** RTX 5090 D v2, Warp
1.17.0.dev20260807, CUDA Graphs, headless, the same continuously running CUDA
guard. Timing includes physics and realtime IK but excludes construction,
compilation, graph capture, and checkpoint readback. No other GPU benchmark
or regression suite ran concurrently with these timed measurements.

| Run | Ordinary 20 | 12 + graph baseline | cached13 | Frame-time reduction |
| --- | ---: | ---: | ---: | ---: |
| Prototype | 68.511 ms | 47.243 ms | 44.877 ms | 5.01% |
| Native repository implementation | 67.778 ms | 47.535 ms | 44.519 ms | 6.34% |

The native baseline/candidate correspond to 21.04 / 22.46 frames/s.
Native results against that run's ordinary-20 reference:

| Frame | Baseline position RMS (mm) | cached13 position RMS (mm) | Baseline edge-length MAE (mm) | cached13 edge-length MAE (mm) |
| --- | ---: | ---: | ---: | ---: |
| 100 | 4.2172 | 1.7806 | 0.09860 | 0.06487 |
| 200 | 5.7364 | 5.6575 | 0.06425 | 0.03789 |
| 300 | 11.0322 | 4.7453 | 0.24224 | 0.13948 |
| 400 | 12.3164 | 6.1921 | 0.17008 | 0.09648 |
| 500 | 13.5504 | 8.4636 | 0.16953 | 0.10051 |
| 600 | 16.6881 | 12.6804 | 0.17144 | 0.11101 |
| 700 | 16.0192 | 10.3073 | 0.16660 | 0.10122 |
| 800 | 15.9779 | 9.9892 | 0.16439 | 0.09475 |
| 900 | 15.8116 | 10.4810 | 0.16246 | 0.09711 |
| Checkpoint mean | 12.3722 | 7.8108 | 0.15662 | 0.09370 |

Position and edge errors improve at all nine checkpoints in both runs.
For the native run, checkpoint means fall 36.87% / 40.17%. However, an
ordinary-20 repeat itself differed by 0.617--15.495 mm at these checkpoints
(68.376 ms/frame): nonlinear contact trajectories are not reproducible
ground truth. The numbers support a measured improvement, not bitwise
equivalence, a guarantee for every frame, or visual certification of an
identical fold. The old 12-sweep contact-free candidate also improved in
this native run (45.057 ms/frame), despite its late-trajectory regression
in the preceding experiment; this variability is why a common-state test
is also required.

The prototype common-state test used an ordinary-20 history and one final
substep sample per frame for 900 frames. Baseline / cached13 mean position
RMS was 0.008022 / 0.004938 mm; P95 was 0.012990 / 0.010200 mm, and mean
edge-length MAE was 0.0007194 / 0.0006339 mm. No baseline coarse rejection
occurred in the sampled substeps. A native reproduction is included in
`scripts/benchmark_mjvbd_v2_multilevel.py`; neither candidate feeds its state
into the next reference substep.

The **native 900-frame common-state reproduction** also passed:

| Metric | 12 + graph baseline | cached13 |
| --- | ---: | ---: |
| Mean local position RMS (mm) | 0.00855777 | 0.00530261 |
| P95 local position RMS (mm) | 0.01429923 | 0.01130051 |
| Mean local edge-length MAE (mm) | 0.00074771 | 0.00068813 |

These reduce the baseline errors by 38.04%, 20.97%, and 7.97%, respectively;
all 900 sampled baseline coarse corrections were accepted. The older
contact-free12 candidate has a lower local P95/edge MAE than cached13 in
this run (0.00984599 / 0.00068074 mm), although its mean local position error
is higher (0.00590760 mm). Do not describe cached13 as strictly dominating
every candidate or metric. It improves the requested 12 + graph baseline
on measured throughput, both independent trajectory metrics, and all three
reported same-state metrics.

**Regression coverage.** All **81 tests** in the six MJVBDV2 modules below
passed on CPU/CUDA. The new cache tests compare the bending force/Hessian
against the original law and the gradient against double-precision finite
differences, exercise the +/-pi damping wrap and degenerate/boundary hinges,
and compare DAT against both original backends on synthetic VT/EE pairs.
They cover repeated factor consumption/reset, active/inactive transitions,
selected-color updates, changing collision snapshots, native per-iteration
collision refresh, eager/Graph replay, allocation limits, and CPU/gradient/
deterministic opt-out. The CPU option regression was checked to fail before
the cache options were introduced. The first synthetic DAT fixture exposed
a test-owned buffer lifetime bug (device structs contain pointers, not
ownership); keeping its host struct alive fixed the test, without changing
solver code. Changed-file pre-commit hooks and `git diff --check` pass.

```bash
uv run --extra dev -m unittest \
  newton.tests.test_mjvbd_v2_surface_cache \
  newton.tests.test_mjvbd_v2_truncation_cache \
  newton.tests.test_mjvbd_v2_surface_relaxation \
  newton.tests.test_mjvbd_v2_particle_multilevel \
  newton.tests.test_mjvbd_v2_contact_optimizations \
  newton.tests.test_mjvbd_v2
```

**Reproduce.** Defaults remain unchanged. To try the candidate visually:

```bash
uv run -m newton.examples.mjvbdv2.example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 --particle-solver-mode cached13
```

For the complete native comparison and common-substep accuracy diagnostic:

```bash
uv run scripts/benchmark_mjvbd_v2_multilevel.py --compare-demo --frames 900
uv run scripts/benchmark_mjvbd_v2_multilevel.py --compare-demo --same-substep --frames 900
```

No default, VR process, GPU power configuration, or CUDA guard lifetime was
changed. These measurements validate this T-shirt scenario only, not all
`final00` examples. No commit or push is part of this tuning request.

### 2026-09-05: integrate the surface experiments with remote tet correction

**Integration base.** The remote `FAST_MJVBDV2` advanced to `ff29013f0`
while the surface experiments were local. Integrate both implementations
instead of replacing either side. The original staged surface work is
retained in a local recovery stash while the integrated revision is tested.

**Dispatch contract.** Keep the remote six-DOF path whenever a correction
contains movable tetrahedral clusters, including mixed cloth/tet models.
`particle_multilevel_operator` selects graph versus block-Galerkin only for
surface-only corrections; it does not replace the mixed/tet basis. Skip
unused three-DOF matrix allocation in the six-DOF case. Preserve the default
`operator="graph"` on the internal correction constructor for the remote
callers. CPU, differentiable, and deterministic solver eligibility remains
unchanged. Automatic mode retains its existing conservative tet exclusion;
explicit tet enablement remains available.

AST comparisons confirm all 32 remote top-level multilevel functions and
its `_restrict_and_prolong_rigid` method are unchanged. All 19 local
top-level functions other than `_build_clusters` are unchanged; clustering
now retains the remote surface/tet partition and tetrahedral assignment.
The two remote final00 examples and their measured defaults are unchanged
relative to `ff29013f0`. The T-shirt still defaults to 12 sweeps plus graph
correction, with `cached13` opt-in.

**Integration regression.** Add a test that rejects unnecessary three-DOF
matrix storage for mixed/tet corrections: without the dispatch-specific
allocation gate it failed with 133 blocks instead of zero, and with the gate
it passed. A second test exercises actual surface caches together with the
six-DOF correction, both surface operator values, and eager/CUDA Graph
execution in both private backends. Its fixture includes enough tets to
meet the pre-existing per-color tile threshold; tiny mixed fixtures correctly
stay scalar and therefore do not exercise the cache. All **87 tests** in the
six MJVBDV2 modules pass after integration.

**Repository checks.** Run `uvx pre-commit run -a` on an isolated exact index
snapshot so its autofixes cannot touch unrelated workspace files. Full-tree
Ruff reports 737 errors in unrelated historical files; the unmodified
`ff29013f0` tree reproduces the same 737 errors. Format the changed test file
and run all hooks on this change's files separately. Do not sweep unrelated
formatting changes into the integration commit.

**Full-length scenario checks after integration.** Run each command below
with `--viewer null --test`. All four runs exit successfully and execute
their existing final-state assertions; these are functional regressions,
not new speed measurements or a claim of bitwise trajectory equivalence.

| `newton.examples.mjvbdv2` module | Configuration | Frames | Result |
| --- | --- | ---: | --- |
| `example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00` | Default `baseline` (12 + graph) | 900 | Pass |
| `example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00` | Opt-in `cached13` | 900 | Pass |
| `example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00` | Remote default (8 sweeps + six-DOF correction) | 1900 | Pass |
| `example_vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher_final00` | Remote default | 1500 | Pass |

The T-shirt checks finite cloth/IK state and collision-only robot geometry.
The cube/bag and Armadillo checks additionally cover completed placement or
physical lift/release/contact phases. The CUDA guard remains active during
and after the tests; no VR stop/reload scripts or GPU power changes are used.
The Towncrier draft also succeeds. Commit and push this integrated revision
at the user's request, retaining the pre-integration stash as local recovery.

### 2026-09-05: promote cached13 to the T-shirt default

At the user's request, the validated `cached13` policy is now the default for
`example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00`.
The historical entries above correctly describe the earlier opt-in decision;
this entry records the later promotion rather than rewriting that history.

Mode construction is explicit and keeps the two retained experiments
separate. Default `cached13` selects 13 sweeps, contact-free surface relaxation
of 1.3, the fixed bending-anchor cache, and the DAT geometry cache, with both
multilevel and Chebyshev disabled. `chebyshev8` preserves the separately
validated eight-sweep collision-aware Chebyshev plus multilevel policy.
`baseline`, `reference20`, and `contact-free` retain their original meanings.
Low-level CLI overrides remain available without changing a mode's defaults.

After fast-forwarding to `20d5d8fb` and restoring the local Chebyshev work, an
exact live-solver assertion confirmed the default configuration. The complete
900-frame CUDA Graph run with `--viewer null --test` passed. Ten surface cache
and relaxation tests, four truncation-cache tests, and three Chebyshev tests
also passed. The source formatter, focused Ruff check, and `git diff --check`
reported no errors.

### 2026-09-05: combine Chebyshev with the surface and DAT caches

Add an explicit `cached-chebyshev8` T-shirt mode to test whether the two
independent optimizations compose. It uses eight collision-aware Chebyshev
sweeps, the guarded graph multilevel correction, and both exact geometry
caches. Surface relaxation remains 1.0: combining Chebyshev with the existing
1.3 surface SOR is a separate over-relaxation and is not part of the retained
mode. The validated `cached13` policy remains the example default.

A 100-frame CUDA Graph comparison on the RTX 5060 Ti measured the retained
combined mode at 51.43--52.05 ms/frame, versus 57.47--57.81 ms/frame for
uncached `chebyshev8` and 58.58--59.33 ms/frame for `cached13`. Thus the caches
reduce the eight-sweep Chebyshev path by about 10.5%, and the combined path is
about 11.8--12.3% faster than `cached13` in this short run.

The speedup is real, but this run does not justify an equal-accuracy claim.
At frame 100 the combined trajectory differed from ordinary 20 sweeps by
5.86--6.21 mm position RMS and 0.101--0.122 mm edge-length MAE; `cached13`
measured 1.70--1.91 mm and 0.060--0.067 mm. Independent ordinary-20 repeats
already differed by 0.68--1.15 mm, so these are nonlinear trajectory metrics,
not direct solver residuals, but the combined deviation is still materially
larger. Nine and ten combined sweeps did not improve it, which is consistent
with contact-driven trajectory divergence rather than simple under-solving.

An exploratory eight-sweep SOR scan was rejected. Relaxation 1.1, 1.2, and
1.3 reduced time to 50.88, 49.92, and 48.78 ms/frame, respectively, while the
frame-100 position RMS increased to 5.76, 6.94, and 7.49 mm. Those variants
are not exposed. Keep `cached-chebyshev8` opt-in for throughput-oriented
experiments; use default `cached13` when matching the accepted 20-sweep
trajectory more closely is the priority. A complete 900-frame CUDA Graph run
of the retained combined mode passed the example's finite-state and scene
invariant checks.

### 2026-09-05: guard cached Chebyshev with contact-local ordinary sweeps

The raw `cached-chebyshev8` mode is fast, but its independent trajectory
diverges from ordinary 20-sweep VBD more than `cached13`. Add an opt-in
`guarded-cached-chebyshev8` policy instead of changing the accepted default:

1. run two ordinary warm-up sweeps;
2. run four Chebyshev sweeps only outside a persistent contact/DAT exclusion;
3. run two ordinary polish sweeps;
4. apply the existing graph multilevel correction; and
5. conditionally run five more ordinary sweeps, reaching 13 total, when the
   device-side post-coarse convergence estimate exceeds 3% of particle radius.

The exact exclusion records particles touched by contact, springs, pneumatic
Hessians, or a DAT truncation at any point in the substep. A compact, immutable
particle-neighbor CSR expands that mask by two elasticity-topology rings before
each accelerated update. The mask remains local to the current substep and the
two private VBD backends use the same schedule. The ordinary and unguarded
Chebyshev paths are unchanged when the new options are disabled.

The cleanup estimate is

```text
max_i ||local_correction_i - accepted_coarse_correction_i|| / radius_i
```

over active particles. This is a cheap linearized estimate of the correction
that remains after the coarse update. Nonfinite values, a rejected coarse
solve, or any DAT/magnitude truncation of the final coarse update always
requests cleanup. A first version tested the pre-coarse local
correction directly; it needlessly requested fallback for error that the coarse
solve subsequently removed. That version reduced the short-run advantage to
about 0.4% and was rejected. The retained scalar status is consumed by
`wp.capture_if`, so inactive cleanup sweeps do not launch their Graph nodes and
there is no per-substep CPU readback. CUDA Graph topology remains fixed.

An initial topology-ring implementation traversed the full mesh adjacency in
each Graph replay and made the benchmark impractically slow. Replace it with a
construction-time bidirectional CSR containing triangle, tet, and spring
connectivity. The retained expansion is one thread per particle and one
short CSR row per requested ring.

**RTX 5060 Ti, Warp 1.17.0.dev20260807, CUDA Graphs, headless.** Timing includes
physics and realtime IK, but excludes construction, compilation, Graph capture,
and checkpoint readback. Each comparison is from one process and one reference
trajectory; absolute times differed between the short and long runs, so only
same-run ratios are used.

| Run | `cached13` | guarded cached Chebyshev | Reduction |
| --- | ---: | ---: | ---: |
| 100 frames | 60.169 ms/frame | 57.284 ms/frame | 4.80% |
| 900 frames | 77.165 ms/frame | 71.308 ms/frame | 7.59% |

The 900-frame number is not a regression from the roughly 60 ms short result:
the first 100 frames contain much less folded-cloth self-contact than the later
trajectory. A standalone 900-frame `cached13` replay measured 76.432 ms/frame,
with consecutive 100-frame blocks at 59.961, 69.861, 77.219, 80.836, 80.431,
79.866, 80.745, 79.617, and 79.353 ms/frame. The earlier roughly 55 ms figure
was the guarded candidate in a short run, not `cached13`; after the final DAT
safety condition that short candidate is 57.284 ms/frame. Compare modes within
the same trajectory length and process rather than mixing these two workloads.

Both 900-frame modes passed `test_final()`. At frame 900 their independent
position RMS differences from that run's ordinary-20 trajectory were 19.702 mm
and 18.179 mm, respectively; edge-length MAE was 0.1437 mm and 0.2088 mm.
These nonlinear, atomic contact trajectories are not reproducible ground truth:
the guarded mode is closer in position but farther in edge length, so neither
trajectory metric alone establishes a strict accuracy ordering.

Use a common-state diagnostic for that distinction. For every frame it forks
all candidates from the same final-substep state of an ordinary-20 history,
then restores all particle state before continuing the reference. The complete
900-frame result was:

| Mode | Mean position RMS | P95 position RMS | Mean edge-length MAE |
| --- | ---: | ---: | ---: |
| 12 + graph baseline | 0.009101 mm | 0.015985 mm | 0.000826 mm |
| `cached13` | 0.031274 mm | 0.112559 mm | 0.001821 mm |
| guarded cached Chebyshev | 0.012310 mm | 0.020483 mm | 0.001180 mm |

No sampled baseline coarse correction was rejected. The guarded policy is
slightly less accurate than the 12 + graph baseline on this local diagnostic,
but it is materially closer to ordinary-20 than `cached13` in all three
reported metrics and remains at hundredth-millimetre position error. Retain it
as an explicit performance/accuracy mode; keep `cached13` as the demo default
until broader scenario validation justifies promotion.

Parameter validation covers invalid spectral windows, topology rings, and
cleanup configurations. CUDA Graph tests execute both private VBD backends with
the guarded schedule and conditional fallback. The compact masks and CSR are
allocated only when guarded Chebyshev is enabled; CPU, differentiable,
deterministic, and ordinary defaults are unaffected because every new option is
disabled by default. Differentiable models remain on ordinary particle VBD, and
the existing unguarded modes retain their prior behavior.

All 92 tests in the seven focused MJVBDV2 modules passed after the final DAT
safety path was added. A final focused re-run also passed the five Chebyshev and
four DAT-cache tests after adding explicit invalid-configuration and truncation
status assertions. Ruff format/check and `git diff --check` pass on every
changed Python file. The required full-tree pre-commit run passes all hooks
except repository-wide Ruff, which reports the same 737 unrelated historical
errors already documented above; none points to a file changed by this work.

### 2026-09-05: promote guarded cached Chebyshev to the T-shirt default

At the user's request, promote `guarded-cached-chebyshev8` after the retained
implementation passed its numerical and performance gates. This supersedes the
earlier `cached13` default decision without removing that mode. The example's
implicit Python fallback and command-line parser now select the guarded policy;
explicit `--particle-solver-mode cached13` reproduces the previous default.

This promotion is local to the W1 T-shirt example. All low-level MJVBDV2
Chebyshev, topology guard, multilevel, cache, and cleanup options remain disabled
by default, so no other example or solver construction changes behavior.

### 2026-09-06: stage one graph correction inside six cached sweeps

**Goal.** Test whether a multilevel V-cycle can approach 30 ordinary particle
sweeps while remaining faster than the retained eight-sweep guarded cached
Chebyshev policy in the W1 T-shirt fold. The experiment keeps ten substeps,
all contact and DAT settings, and the scripted realtime IK trajectory fixed.

**Implementation.** Both private MJVBDV2 particle solvers accept an optional
exact `particle_multilevel_checkpoints` sequence. `None` preserves the legacy
single correction after the final fine iteration. An explicit sequence is the
complete schedule; it does not implicitly append a terminal correction. The
T-shirt performance mode runs six cached, collision-aware Chebyshev sweeps and
applies the graph correction after sweep three, leaving three fine sweeps to
smooth the prolonged update. CPU, differentiable, deterministic, disabled
multilevel, and all examples that omit the option keep their previous path.

The first prototype accidentally interpreted `(3,)` as corrections after both
sweeps three and six. Correcting this to the intended `3 fine -> coarse -> 3
fine` schedule reduced a matched 300-frame run from 53.612 to 51.001 ms/frame.
Its independent position RMS versus ordinary 30 sweeps improved at frames
50/100/150 from 9.091/11.645/13.024 mm to 6.658/10.048/10.601 mm. At frame 300
the two schedules were effectively tied at 18.418/18.630 mm. These trajectory
numbers are convergence proxies, not deterministic error bounds.

**30-sweep accuracy decision: No-Go.** In the native 300-frame comparison,
ordinary 30 sweeps took 152.626 ms/frame and its independent repeat differed
by 0.015, 1.838, and 6.543 mm at frames 50, 100, and 300. The original staged
candidate took 53.612 ms/frame but differed from the same reference by 9.091,
11.645, and 18.418 mm. The corrected single-midpoint schedule improves the
early trajectory and throughput but does not close the remaining gap. It must
not be described as equivalent to 30 ordinary sweeps.

Parameter probes over checkpoint placement, two or three coarse corrections,
coarse relaxation, seven through twelve fine sweeps, guarded windows, and
spectral radii did not consistently improve both position and edge error.
More or stronger frozen coarse corrections usually amplified nonlinear contact
error. A singleton/full-fine Galerkin defect-correction prototype was also not
retained: aggressive parameters diverged, and an attempted in-solver fork did
not transactionally restore all hidden solver/contact history, eventually
corrupting its diagnostic reference. The temporary singleton allowance was
removed; none of those exploratory results changes the production API.

**Long run.** The corrected six-sweep schedule completed the full 900-frame
CUDA Graph headless test on an RTX 5060 Ti in 59.043 ms/frame. Consecutive
100-frame blocks were 44.957, 51.746, 57.987, 62.598, 62.129, 62.511,
62.567, 63.859, and 63.031 ms/frame. Particle positions and velocities stayed
finite, the final graph-correction status was zero, and the example's existing
`test_final()` passed. This establishes stability and throughput for that run;
the test does not certify visual identity or 30-sweep convergence.

### 2026-09-06: residual-selected frozen-contact Schwarz polish

**Goal.** Improve on the six-sweep staged mode while meeting a stricter
criterion than independent trajectory similarity: from an identical substep
state, the candidate should be no farther from an ordinary 60-sweep proxy
solution than ordinary 30 sweeps. The position mean/P95 and edge-length mean
error ratios must each remain at or below 1.10, while end-to-end time must be
lower than the retained six-sweep mode.

Two broader prototypes were rejected first. Reusing the previous coarse PCG
solution through a one-vector residual-minimizing projection did not reduce
end-to-end time at three or four new PCG iterations and worsened edge error.
Restricting complete fine sweeps to a residual mask also lost performance:
even with only 15--47% of particles active, it still rebuilt contact forces,
self-contact forces, and DAT state for the whole candidate set. Neither
prototype remains in the implementation.

The retained `residual-schwarz5` T-shirt mode uses the following schedule:

1. run five cached collision-aware Chebyshev fine sweeps;
2. after sweep three, run the existing guarded graph coarse correction and
   mark particles whose local Newton correction exceeds 0.001 particle radii;
3. after sweep five, run two multiplicative colored surface passes only for
   the marked particles;
4. reuse the last complete fine sweep's contact force/Hessian during those
   passes and clamp every accepted local update to 0.001 particle radii; and
5. run one final DAT truncation over the accumulated displacement.

The last two passes are deliberately not incomplete full VBD iterations. They
are a frozen-contact nonlinear Schwarz smoother: updated elastic positions are
visible to later colors, but expensive body-contact/self-contact force and
Hessian assembly is not repeated. The tight trust region bounds the error of
that frozen contact linearization. A rejected/nonfinite coarse solve takes the
existing device-side fallback branch to 20 iterations and skips the polish.
CUDA Graphs express polish as the false branch of `wp.capture_if`, so fallback
and polish have fixed captured topology without a per-substep CPU readback.

The low-level feature is opt-in and defaults to zero polish iterations in both
private VBD backends. Its initial implementation is intentionally limited to
the cached pure-surface tile path; a requested mixed/tetrahedral configuration
fails explicitly. CPU, differentiable, deterministic, volumetric, and every
example that omits the new options retain their previous paths and allocation
behavior.

**Strict common-state accuracy.** A new benchmark diagnostic runs an ordinary
60-sweep history and, at the same last-substep initial state of each of 100
frames, forks both ordinary 30 sweeps and the retained candidate. Ordinary 60
is only a numerical proxy for the converged state, but this comparison removes
independent atomic-contact trajectory divergence.

| Trial versus ordinary 60 | Mean position RMS | P95 position RMS | Mean edge-length MAE |
| --- | ---: | ---: | ---: |
| Ordinary 30 | 0.037830 mm | 0.105278 mm | 0.002550 mm |
| `residual-schwarz5` | 0.027789 mm | 0.104203 mm | 0.002768 mm |
| Candidate / ordinary 30 | 0.735 | 0.990 | 1.085 |

All three error ratios pass the 1.10 gate. The last-substep coarse status was
accepted in all 100 samples and the mean active set was 314 of 6,436 particles.
This validates 30-sweep-level convergence for the sampled W1 T-shirt states;
it is not a claim that five sweeps are universally equivalent to 30 for other
materials, topologies, or contact histories.

**Performance and long-run validation.** On the RTX 5060 Ti with Warp
1.17.0.dev20260807, ten substeps, CUDA Graphs, and headless realtime IK, a
matched 300-frame process measured the old six-sweep mode at 52.468 ms/frame
and the candidate at 47.564 ms/frame, a 9.35% reduction. The candidate's
complete 900-frame run took 54.961 ms/frame; a fresh old-mode run under the
same source tree took 59.493 ms/frame, a 7.62% reduction. Candidate 100-frame
blocks were 41.475, 47.541, 53.186, 58.175, 59.496, 59.120, 58.666, 58.164,
and 58.824 ms/frame. All nine checkpoint statuses were zero, active sets were
433, 304, 721, 206, 197, 169, 151, 183, and 209 particles, finite-state checks
passed, and the example's `test_final()` passed.

The benchmark now exposes both retained six-sweep and residual-Schwarz modes
and a strict `--same-substep --strict-residual-schwarz` diagnostic. Unit tests
cover option validation, threshold/ring mask construction, explicit volumetric
rejection, and eager plus CUDA Graph execution in both private VBD backends.

### 2026-09-06: consolidate the validated surface policy into one preset

The T-shirt demo had accumulated more than twenty solver switches while
comparing superseded acceleration experiments. That made the final example
look as though users needed to understand and reproduce an implementation
schedule involving Chebyshev warm-up, cleanup rings, multilevel checkpoints,
selective polish, caches, and fallback iterations.

`SolverMJVBDV2` now accepts `vbd_preset="surface-fast"`. The preset owns the
validated five-sweep residual-Schwarz schedule and its 20-sweep transactional
fallback. `vbd_options` remains an expert override layer and takes precedence
over preset values. Outside externally driven CUDA triangle-only surface
solves—including differentiable, deterministic, spring, tetrahedral,
pneumatic, and VBD-dynamic-rigid models—the preset resolves to 20 ordinary
sweeps instead of silently running five sweeps on an unvalidated path.

The final T-shirt demo now selects that single preset and specifies only its
scene-scale self-contact distances and topology filtering. It uses the safe
default row capacities instead of exposing memory-tuning knobs. Historical
and rejected policies remain available only in the dedicated benchmark
script; they are no longer advertised as normal demo configuration choices.

A 300-frame headless run after consolidation measured 47.672 ms/frame in
100-frame blocks of 41.804, 47.916, and 53.297 ms/frame. The nearby
pre-consolidation measurement was 47.564 ms/frame, so the difference is timing
noise rather than a material regression. All three multilevel statuses were
zero and `test_final()` passed. The public dispatch suite passed 42 tests, the
preset-specific CUDA/fallback tests passed, and a direct 10-frame invocation
of the simplified demo completed successfully.

### 2026-09-07: reject split EE and VT self-contact kernels

**Hypothesis.** The combined self-contact force/Hessian kernel contains both
edge-edge (EE) and vertex-triangle (VT) narrow phase code. Splitting those
paths could lower register pressure enough to improve occupancy, despite
adding one CUDA launch per color.

**Prototype.** An opt-in `vbd_soft/` path copied the existing EE and VT
branches into separate kernels. It retained the directed detector rows,
material selection, color filtering, contact equations, atomic accumulation,
block limits, and current-position reevaluation. The normal combined kernel
remained available for a matched A/B run. No shared Newton collision code or
production preset was changed.

**Measurement.** The scripted 6,436-particle W1 T-shirt scene used 12,736
triangles, 19,174 edges, nine particle colors, ten substeps, the retained
five-sweep residual-Schwarz schedule, CUDA Graph capture, Warp 1.17.0, and an
NVIDIA GeForce RTX 5090 D v2. Both cases ran consecutively in one process for
100 evolving frames after construction and graph capture:

| Self-contact accumulation | Mean frame time | Change |
| --- | ---: | ---: |
| Combined EE + VT kernel | 22.220485 ms | baseline |
| Dedicated EE and VT kernels | 23.493046 ms | 5.73% slower |

Both paths completed with finite state, zero multilevel rejection status, and
the example's `test_final()` passed. The split path's extra VT launch and grid
work cost more than any occupancy benefit. The prototype was fully reverted;
do not repeat this split without a fused scheduling mechanism that removes at
least one other synchronization boundary.

### 2026-09-07: retain topology-aware color-batched Jacobi

**Problem.** The previous fastest surface policy used five complete colored
sweeps. Every sweep traversed all nine independent-set colors, so contact,
self-contact, elasticity, and DAT work paid nine launch and synchronization
boundaries. Simply reducing the sweep count no longer met the established
ordinary-30 accuracy gate, while merging fixed pairs of colors made the
edge-length error too sensitive to the selected partition.

**Implementation.** The opt-in CUDA surface path now merges the original
colors into two ordered batches. A batch evaluates every particle correction
from the same frozen displacement and commits those corrections together;
the second batch still sees the first batch's committed state. Construction
scores balanced partitions by triangle and bending-edge coupling, then rotates
among low-coupling partitions over eight sweeps. The rotation minimizes the
maximum number of times any original color pair shares a batch, rather than
leaving one strongly coupled pair in Jacobi form for the entire solve.

The retained `surface-fast` policy uses eight of these two-batch sweeps,
Chebyshev spectral radius 0.8, and one guarded graph multilevel correction
after sweep four. A rejected multilevel correction takes the already captured
20-sweep fallback, with both batched Jacobi and Chebyshev disabled so that the
fallback remains ordinary colored Gauss--Seidel. The CUDA Graph topology is
fixed and no per-frame host synchronization was added.

The feature is disabled by default in the low-level solver. A direct opt-in
requires a nondifferentiable, nondeterministic CUDA triangle surface using the
cached tiled solve, with no tetrahedra or springs and no isolated particles.
Unsupported direct configurations fail explicitly. The high-level preset
continues to resolve CPU, differentiable, deterministic, pneumatic,
tetrahedral, spring, and dynamic-rigid configurations to 20 ordinary sweeps.

**Strict common-state accuracy.** On an NVIDIA GeForce RTX 5090 D v2, Warp
1.17.0, CUDA Toolkit 12.9, and Driver 13.2, the diagnostic advanced an
ordinary 60-sweep history and forked ordinary 30 sweeps and the candidate from
the identical final-substep initial state of each of 300 frames. The scene had
6,436 particles, 12,736 triangles, 19,174 bending edges, nine original colors,
996 coarse clusters, and ten substeps per 60 Hz frame. Ordinary 60 is a
numerical convergence proxy rather than an exact solution.

| Trial versus ordinary 60 | Mean position RMS | P95 position RMS | Mean edge-length MAE |
| --- | ---: | ---: | ---: |
| Ordinary 30 | 0.026622 mm | 0.046863 mm | 0.001700 mm |
| Two-batch eight-sweep candidate | 0.020966 mm | 0.040641 mm | 0.001849 mm |
| Candidate / ordinary 30 | 0.788 | 0.867 | 1.088 |

All three ratios pass the pre-established 1.10 accuracy gate. None of the 300
sampled final-substep corrections was rejected, all values remained finite,
and the example's final test passed. A fixed two-batch partition was faster
but failed the edge gate at 1.207; a rotating affine partition also failed at
1.147; and three batches passed accuracy but improved frame time by only
23.5%. These alternatives were not retained.

**Performance and long-run validation.** A matched process captured both
policies as CUDA Graphs and advanced the full scene, including realtime IK.
Over 900 frames, the previous five-sweep residual-Schwarz policy averaged
28.244172 ms/frame and the candidate averaged 18.671711 ms/frame, a 33.89%
reduction. Candidate 100-frame blocks were 15.012, 17.721, 19.490, 19.988,
19.358, 19.104, 19.174, 19.078, and 19.121 ms/frame. All nine multilevel
statuses were zero and `test_final()` passed. A separate direct 900-frame
invocation through `vbd_preset="surface-fast"` also passed, confirming that the
measured benchmark configuration and the production entry point agree.

### 2026-09-07: correct the disabled-Chebyshev reference

The strict result immediately above used `particle_chebyshev_enabled=False`
to form its ordinary 30- and 60-sweep references while leaving the previously
built recurrence weights allocated. The iteration kernels checked only the
weight-array length before applying acceleration, so those references still
ran Chebyshev. The performance comparison between the enabled production
policies remains valid, but the stated ordinary-30 accuracy gate did not.

Both private VBD backends now require the runtime enable flag as well as a
valid recurrence index. A focused CPU regression constructs a solver with
weights, disables it at runtime, and compares one step with a solver that never
allocated weights. Without the fix, 80% of particle-position components differ
and the maximum error is 0.199 m in the small forced-cloth fixture; with the
fix, both backends match within `rtol=2e-6`, `atol=2e-7`.

The corrected 100-frame common-state T-shirt diagnostic gives:

| Trial versus ordinary 60 | Mean position RMS | P95 position RMS | Mean edge-length MAE |
| --- | ---: | ---: | ---: |
| Ordinary 30 | 0.014054 mm | 0.027881 mm | 0.000958 mm |
| Two-batch eight-sweep candidate | 0.023095 mm | 0.045508 mm | 0.002042 mm |
| Candidate / ordinary 30 | 1.643 | 1.632 | 2.132 |

The two-batch implementation remains an explicit performance option, but its
former claim of ordinary-30-equivalent T-shirt accuracy is withdrawn. Future
surface presets must use the corrected reference gate.

### 2026-09-07: exclude multilevel from rotating anchored cloth twist

Enabling the surface preset on the 2,500-particle, three-color cloth-twist
scene exposed a narrow triangular spike beside a rotating fixed corner. A
geometric diagnostic measured the transverse displacement of each fixed
corner's first free neighbor divided by its inward displacement. Ordinary four
sweeps stayed at 0.750 or below, while the three-sweep preset reached 1.760 at
opposite corners. The existing finite, residual, global clamp-fraction, and
velocity guards all passed, so none detected this local directional bias.

Matched 300-frame option ablations isolated the coarse correction:

| Three-sweep surface configuration | Maximum corner ratio | Result |
| --- | ---: | --- |
| Batched + Chebyshev + multilevel | 1.760 | Visible spike |
| Disable only multilevel | 0.858 | Normal boundary |
| Disable only Chebyshev | 1.954 | Visible spike |
| Disable only batched Jacobi | 1.776 | Visible spike |
| Batched only | 0.661 | Normal boundary |
| Chebyshev only | 0.754 | Normal boundary |
| Cached ordinary Gauss--Seidel | 0.717 | Normal boundary |

Commit `889ae3d47` introduced the opt-in multilevel mechanism, but the cloth
twist did not exercise it until the surface preset was selected. Its
translation-only coarse clusters do not move fixed particles, yet clusters
beside the rotating anchors can repeatedly prolong a small transverse update
to their active particles. The global clamp-fraction check dilutes this local
failure among all 2,400 active particles. Suppressing one or two coarse rings
was insufficient; three rings removed the spike but excluded 174 of 308
clusters, so that scene-specific solver modification was rejected.

The example now disables only multilevel correction on its accelerated CUDA
path. It retains three two-batch Chebyshev sweeps plus the surface and DAT
caches; CPU, differentiable, and deterministic execution retain the former
four ordinary sweeps. In an alternating ABBA run on an NVIDIA GeForce RTX
5090 D v2 with Warp 1.17.0, CUDA Toolkit 12.9, Driver 13.2, CUDA Graphs, ten
substeps, and three 100-frame synchronized blocks per case, ordinary four
sweeps averaged 17.790773 ms/frame and the anchored-safe configuration averaged
9.830389 ms/frame, a 44.744% reduction. Repeated accelerated runs had maximum
corner ratios of 0.855 and 0.860. The example's five-second final test now
rejects a ratio of 1.2 or greater so the visual regression cannot silently
return.

At frame 300, ordinary/accelerated mean edge strain was 8.129%/8.003%, P95
edge strain was 21.698%/21.872%, mean area strain was 6.841%/6.855%, and P95
area strain was 14.081%/14.372%. RMS particle speed was 0.1711/0.1713 m/s and
maximum speed was 0.4952/0.4975 m/s. These constraint and velocity statistics
remain comparable while the visible corner failure is removed.

### 2026-09-08: local acceptance trial of nonlinear coarse/fine Newton

At the user's request, port the candidate from `exp/mjvbd-defect-polish`
to the local `FAST_MJVBDV2` working tree. This is an uncommitted visual
acceptance trial, not a declaration of general solver equivalence.
The T-shirt example now defaults to `--vbd-preset surface-global`;
`--vbd-preset surface-fast` restores its previous accelerated schedule,
including its ordinary final sweep. Other example defaults are unchanged.

The experimental preset performs a four-particle-cluster Galerkin predictor
(eight PCG iterations), followed by a particle-resolution dynamic tangent
solve (five PCG iterations). CUDA conditional nodes can select up to two
additional nonlinear fine solves when the radius-normalized candidate
correction exceeds 0.049. Each correction keeps the existing DAT path and
0.05-radius bound. This criterion measures the bounded candidate update,
not an exact nonlinear residual or the accepted post-DAT displacement;
it is not a convergence proof. Missing reciprocal EE records invalidate
the projected operator and trigger the ordinary-sweep fallback rather
than changing directed fine-level contact forces.

CPU, requires-grad, deterministic, tet, pneumatic, spring, and full-VBD
dynamic-rigid configurations retain ordinary-sweep fallback. Unsupported
schedule overrides are rejected explicitly. No material, friction,
damping, prescribed trajectory, substep count or collision frequency is
changed. The global algorithm is not automatically enabled for cloth twist
or the loaded plastic bag.

Prior matched 900-frame measurements in the experimental worktree:
RTX 5060 Ti, Warp 1.17.0, CUDA 12.9, driver 13.3, ten substeps/frame.
These numbers are historical benchmark evidence, not local migration
smoke-test timings.

| Mode | Wall ms/frame | Mean / P95 absolute edge strain | Frame-900 RMS speed | Strict edge/face crossing pairs, frame 900 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary 30 sweeps | 142.666 | 2.010% / 6.614% | 0.001210 m/s | 48 |
| Existing surface-fast | 38.017 | 3.729% / 11.165% | 0.003939 m/s | 335 |
| Adaptive global candidate | 35.609 | 1.903% / 6.285% | 0.007928 m/s | 89 |

Candidate trajectory RMS relative to ordinary30 was 22.728 mm at frame
900 (surface-fast: 31.433 mm). Strain improves, but terminal velocity is
worse and self-intersections remain. Crossing counts exclude touching and
coplanar overlap; they are not counts of visible holes. Neither baseline
nor candidate establishes collision freedom. A frozen-substep probe also
did not establish ordinary30-equivalent nonlinear convergence.

Cloth twist's prior 300-frame trial passed its corner check but regressed
from 10.077 to 20.798 ms/frame, so its default remains unchanged. The loaded
bag uses dynamic rigid bodies and the full VBD core; this surface-only
algorithm has not been validated there. Do not advertise a universal
speedup. Unconditional extra ordinary smoothing was also rejected after
its 900-frame trial took 43.535 ms/frame without recovering baseline
terminal speed or the candidate's deformation/crossing measures.

Local integration checks cover preset fallback, conflicting options,
CUDA graph replay, asymmetric contact rejection and clearing adaptive
flags between graph launches. Twelve focused unit tests passed. The
T-shirt default completed a 120-frame startup run and a 900-frame `--test`
run; the old preset passed a separate 120-frame `--test` run. These checks
do not test visual self-intersections. The remaining acceptance is visual.

### 2026-09-08: post-DAT nonlinear residual backtracking for settling

The user reported persistent terminal motion in the local global prototype.
The earlier strain comparison did not establish ordinary30-equivalent
nonlinear convergence. A same-state frame-900 probe confirmed a remaining
local Newton defect; merely increasing fine PCG from five to ten or twenty
iterations did not improve it consistently. Removing the coarse predictor
made the measured defect worse. This points away from treating the linear
iteration budget as the sole problem.

Add a private residual-globalization controller to `surface-global`:

- Assemble the original force and block Hessian at the actual post-DAT
  position. Measure `sum(delta_i^T H_ii delta_i)`, where the local Newton
  correction is `delta_i = H_ii^-1 f_i`, plus radius-normalized squared
  local corrections. This is a preconditioned force-residual merit, not
  physical energy, and does not establish convergence by itself.
- Check the coarse predictor and each fine Newton step against that merit.
  On a failed check, try half and quarter steps. Restore both positions and
  cumulative DAT displacements together and reapply existing DAT to shortened
  trials. If no trial is accepted, restore the base state and run an ordinary
  directed-contact sweep; that fallback is not claimed to monotonically
  decrease the global merit.
- Reuse the newly assembled local/contact data for the next fine solve.
  Skip subsequent solve blocks only when merit has decreased by 100x AND
  RMS radius-normalized local correction is at most 0.001. Keep at most
  three fine solves. Those are bounded-work stopping rules, not a guarantee
  that an unconverged substep reaches tolerance.
- Keep decisions on device with flat CUDA conditional blocks and persistent
  storage. No frame number, robot trajectory phase, velocity decay, material
  retuning, substep change or weakened DAT is used.

Matched 1200-frame full-trajectory trials, same RTX 5060 Ti / Warp 1.17.0
environment as above. Sample speed after every ten frames in the final
300-frame window, outside timed blocks. Wall/GPU times exclude these
diagnostic readbacks and initialization. Each example final check passed.

| Mode | Wall / GPU ms per frame | Terminal-window mean / P95 RMS speed (m/s) | Final mean / P95 absolute edge strain |
| --- | ---: | ---: | ---: |
| Previous global | 34.670 / 34.663 | 0.011034 / 0.013489 | 1.755% / 5.806% |
| Coarse + fine residual backtracking, selected | 38.416 / 38.409 | 0.004423 / 0.006276 | 2.242% / 6.971% |
| Ordinary30 | 142.474 / 142.469 | 0.002383 / 0.005704 | 1.893% / 6.301% |

The selected trial reduces window-mean speed by about 60% with 10.8% more
frame time, but its mean/P95 strain is worse than the previous global
trajectory and is not identical to ordinary30. Terminal-window mean speed
is still 1.86x ordinary30. These are complete trajectories, not matched-state
convergence measurements. Do not describe this as eliminating jitter,
achieving ordinary30 accuracy, or proving zero self-intersection. No new
independent intersection count was collected in this trial.

Rejected/default-off ablations (all 1200 frames):

| Ablation | Wall ms/frame | Window-mean RMS speed (m/s) | Final mean / P95 edge strain |
| --- | ---: | ---: | ---: |
| Fine-only backtracking, strict AND stopping | 48.065 | 0.003459 | 1.705% / 5.573% |
| Fine-only, either relative 0.01 or absolute RMS 0.001 | 34.734 | 0.008155 | 1.964% / 6.277% |
| Fine-only, either relative 0.001 or absolute RMS 0.0005 | 36.827 | 0.006829 | 1.998% / 6.346% |

Fine-only strict stopping executes 35,817 fine solves in 12,000 substeps.
The selected coarse+fine controller executes 20,721 fine solves and 7,323
backtracking trials. Looser stopping largely recovers speed but loses much
of the settling improvement, so it is not the local default.

A separate integration defect was found while recapturing graphs: lazily
allocated contact-projection arrays could belong to the first graph, then
be reused after that graph was destroyed. Compute Sanitizer reported an
out-of-bounds `_reset` write before the fix. Allocate both coarse and fine
contact projections during preset construction, before capture. Tests now
assert allocation before first capture and reuse across destroyed/recaptured
graphs. Benchmark ablations configure their mode before first capture rather
than replacing live graph inputs. Thirteen focused tests cover CPU/CUDA
residual backtracking, restoration of cumulative displacements, rejection,
converged-block skipping, graph replay, preset fallback and asymmetric EE.

Reproduction: `scripts/benchmark_mjvbd_v2_global_settling.py --frames 1200`
compares the previous global scheduler, selected residual controller and
ordinary30. `scripts/probe_mjvbd_v2_global_settling.py --frames 900` performs
the matched-state diagnostic. Raw local benchmark records are retained in
ignored `newton/tests/outputs/mjvbd_global_settling/`; the `balanced` and
`tight` records use the historical OR tolerances listed above, not the
current strict default. Nothing is staged or committed for this trial.

Validation limitation: Compute Sanitizer 12.8 on this machine still exits
with a host-side access violation inside `wp_cuda_graph_launch` on the full
new graph, while reporting zero device memory errors after the allocation
fix. This is NOT a successful sanitizer run and its cause is not established.
Normal CUDA replay and the focused CPU/CUDA tests complete; do not infer a
general absence of memory bugs from those checks.
The final local default also completed a separate direct example invocation
with `--viewer null --num-frames 1200 --test` (exit code 0).

### Follow-up: reject frozen-metric search and repair full-backend cache calls

The next experiment kept the same force equations, DAT, stopping tolerances,
coarse/fine budgets and trajectory. It froze each particle's initial local
Hessian inverse for the substep and measured `f(x)^T H(x_initial)^-1 f(x)`
instead of refreshing the weight at every trial. This makes the trial
comparison use one fixed metric, but does not make it a physical energy.

Matched 1200-frame trials on the same machine, with final checks passing:

| Mode | Wall / GPU ms per frame | Terminal-window mean / P95 RMS speed (m/s) | Final mean / P95 absolute edge strain |
| --- | ---: | ---: | ---: |
| Current residual controller | 38.503 / 38.497 | 0.004099 / 0.005949 | 1.999% / 6.468% |
| Frozen initial-Hessian metric, rejected | 40.967 / 40.961 | 0.004093 / 0.005890 | 1.982% / 6.428% |

The frozen metric costs 6.4% more frame time for essentially unchanged
terminal-window mean speed. Fine solves increase from 20,681 to 26,085;
backtracking trials increase from 8,132 to 14,271. These single-trajectory
results do not establish a meaningful accuracy gain, especially with
nondeterministic contact ordering. No-Go: remove the experimental metric
buffer, kernel arguments and benchmark mode; retain the previously selected
controller. Both timed arms included the initial inverse calculation in
the experimental implementation, so this comparison is not a separate timing
of the restored production version. Raw records: local ignored
`newton/tests/outputs/mjvbd_global_settling/fixed_metric_trial.json`.

An equation audit and independent CPU/CUDA finite-difference tests establish:

- The membrane force, including metric damping, agrees with its scalar
  energy gradient on compressed, sheared and stretched nondegenerate
  triangles. The existing projected/Gauss-Newton Hessian is an approximation,
  not evidence that the force uses a different energy.
- Fully refreshed friction loads do not in general admit a joint scalar
  normal/tangential potential with the unchanged normal equation. A simple
  sliding contact gives `dF_t/dgap = mu*k` but `dF_n/dslip = 0`. A future
  energy-based globalizer must explicitly define load lagging / outer
  iterations or change the equations; it must not silently claim an exact
  energy for the current coupled force field. This observation alone does
  not identify the cause of the demo's remaining jitter.

Broader regression testing found an actual integration bug in the staged
surface-cache signature change: the complete private `vbd` backend still
passed 24 arguments to a 26-argument kernel. Ordinary cached surface updates
in mixed tet/cloth models and selective polish were affected. Pass null
optional local-correction/Hessian outputs at both full-backend call sites;
no residual collection, scheduling or material change is needed there.
The existing mixed-model and selective-polish CUDA graph tests failed before
the fix and pass afterward. Update the Jacobi-fusion reference harness too:
its trailing argument index had started overwriting `selective_active`
instead of relaxation. Preserve the original equivalence assertions.

Full MJVBD_V2 test discovery also exposed experimental PCG dispatch leaking
into the established multilevel path. `SplitCoarsePCG.parallel` had become
true by default, and even its nonparallel path always used the new fused
float64 product reduction. Ten subcases in the original split-PCG tests
failed: output rounding differed, residual histories changed, and overflow
scratch was no longer preserved. Restore block-ordered reductions as the
default and explicitly opt in to fused fine-PCG / parallel coarse-PCG only
inside `surface-global`. Keep the global preset's selected numerical route;
do not relax the old bitwise array-equality assertions. All four split-PCG
tests now pass, including their CPU/CUDA, linked/packed-contact, rejection
and repeated-capture subcases.

After these fixes, full discovery
`uv run python -m unittest discover -s newton/tests -p 'test_mjvbd_v2*.py' -q`
passes all 136 tests. This is wider regression coverage, not a claim that
all full-length demos have been rerun or ordinary30 accuracy is achieved.

A separate post-fix 1200-frame T-shirt residual-mode run passes its final
check: 38.977 wall / 38.971 GPU ms/frame, terminal-window mean/P95 RMS speed
0.004295 / 0.005880 m/s, final mean/P95 edge strain 2.168% / 7.005%.
Fine solves: 20,629; backtracking trials: 9,061. This remains in the prior
roughly 39 ms/frame range, but trajectory statistics vary; it is not a new
speedup or an accuracy improvement claim. The retained changes in this
follow-up are compatibility fixes and validation, not a new settling method.
Record: `newton/tests/outputs/mjvbd_global_settling/compatibility_fix_validation.json`.

Sanitizer isolation is now reproducible independently of Newton using
`scripts/probe_cuda_conditional_graph.py`. On this machine:

- Uninstrumented conditional graph replay passes.
- Compute Sanitizer 12.8 ordinary graph replay passes, while conditional
  replay fails with CUDA error 999 at the following device-to-host copy.
- A portable official CUDA 13.0.85 sanitizer, downloaded without installing
  or changing the system toolchain, reproduces the same conditional failure.
- That 13.0 sanitizer passes the two new objective tests plus the mixed
  tet/surface ordinary-graph replay test, with zero reported memory errors.

This narrows the conditional-graph failure to an interaction reproducible
without the solver, but does not identify the responsible Warp/driver/tool
component or certify the full conditional solver graph. Do not discard
conditional execution merely to obtain a clean instrumentation result.
The T-shirt default, ordinary force laws, friction, damping and DAT remain
unchanged by this follow-up. Existing staged work is preserved; new fixes,
tests and this record remain unstaged.

### 2026-09-08: nonlinear direction and settling diagnostics (in progress)

The acceptance remains unmet: ordinary30-like same-state convergence and
substantially less terminal jitter, with a small performance cost. Neither
similar final strain nor passing the demo's broad final check is sufficient.
No trajectory-dependent damping, velocity decay, material retuning or DAT
relaxation is introduced in these experiments.

Rejected preliminary directions, all using the unchanged force equations:

- One-history Anderson mixing of the preconditioned PCG displacement map:
  38.812 ms/frame, terminal mean/P95 RMS speed 0.004155/0.005691 m/s;
  allowing four fine corrections gives 41.168 ms/frame and
  0.004239/0.006031 m/s. There is no convincing settling gain over the
  roughly 39 ms residual baseline. Remove this private implementation and
  its benchmark modes, rather than exposing another solver switch.
- Replacing only the isotropic friction tangent with its exact fixed-load
  derivative is harmful: 118.819 ms/frame, terminal mean/P95 RMS speed
  0.204870/0.235159 m/s and mean/P95 edge strain 6.537%/27.666%.
  The current refreshed normal/friction problem is not a symmetric
  fixed-load minimization. Revert this incomplete derivative change.
- A CPU sparse-direct diagnostic of the assembled global matrix does not
  establish that more accurate PCG solves would fix nonlinear convergence.
  In one shared frame-900 substep, residual mode has a local defect RMS
  3.702 um and displacement RMS 13.967 um relative to ordinary30;
  direct four/eight-step corrections have 7.892 um and 438.875 um,
  respectively. This is an untimed diagnostic, not a proposed CPU path.
- Initial-state finite-difference Newton/GMRES with a sparse-direct
  preconditioner also fails the combined force-defect acceptance. A trial
  gives 9.803 um displacement RMS relative to ordinary30, but force RMS
  7.328 N versus 1.018 N for ordinary30. A closer displacement alone is
  not sufficient. A further diagnostic tests this direction as refinement
  of the fast solution instead of replacing its initial solve.
- Six-vector nonlinear residual mixing with an ordinary GS preconditioner
  reduces the local defect in an eight-step diagnostic, but does not
  match ordinary30: local RMS 1.703 versus 0.890 um, displacement RMS
  26.818 um relative to ordinary30. The CPU prototype is removed; no
  N-GMRES path is enabled in the solver.

One independent tangent consistency fix is under regression validation:
at exactly zero slip with positive regularization distance, friction force
is zero but its tangent is `2*mu*N/eps * P`, not zero. Both private soft/full
helper copies now retain this tangent. The eps=0, slip=0 branch remains
unchanged. Independent CPU/CUDA tests verify zero force and the finite
tangent. This fixes a discontinuity in the linearization; it is not by
itself evidence of eliminating terminal motion.

The new spectral membrane experiment projects all six deformation-gradient
eigenmodes before mapping to vertex blocks. Unlike clamping the scalar
stress coefficient, this preserves the stress-free rotational nullspace.
It replaces only the global membrane tangent (including its diagonal), not
the force or the established local VBD operator. CPU/CUDA tests compare
against an independent finite-difference Hessian and eigenprojection, and
check pinned vertices, merged clusters and preservation of other diagonal
terms. Degenerate triangles retain the existing conservative tangent.
This is diagnostic-only, not selected as the preset default.

Matched 1200-frame trials now also record terminal-window net displacement
and the path sampled every ten frames. These distinguish slow net motion
from oscillatory motion but are not a full frequency analysis:

| Mode | Wall / GPU ms/frame | Mean / P95 RMS speed (m/s) | Net / sampled-path RMS (mm) |
| --- | ---: | ---: | ---: |
| Spectral tangent, original stopping | 38.836 / 38.830 | 0.005425 / 0.013335 | 7.732 / 10.400 |
| Current residual baseline | 38.492 / 38.486 | 0.004893 / 0.006438 | 3.413 / 4.540 |
| Ordinary30 | 142.232 / 142.229 | 0.001518 / 0.002169 | 4.834 / 5.376 |

All final checks pass; none proves freedom from self-intersection. Bodies
also move by up to 0.000109 in transform-component units in that window,
identically in all three runs, so these are not perfectly fixed-boundary
static-equilibrium experiments. Full trajectories have nondeterministic
contact ordering; compare matched-state force defects separately.

Original tangent with stricter stopping (`relative=0.001`, normalized local
RMS=0.0001) costs 49.971 ms/frame with three fine solves, terminal speed
0.003106/0.004816 m/s. Five fine solves cost 80.519 ms/frame and give
0.004394/0.009199 m/s. Merely raising the bounded nonlinear work is not a
successful default: the latter is slower without a stable settling gain.
Further spectral/strict trials and force-balance refinement remain under
evaluation; do not report this section as a completed optimization.

Raw local records are in ignored `newton/tests/outputs/mjvbd_global_settling/`:
`krylov_trial.json`, `krylov4_trial.json`, `exact_friction_tangent_trial.json`,
`jfnk_probe.log`, `ngmres_gs_probe.log`,
`spectral_membrane_trial.json`, `spectral_probe.log`, and
`strict_spectral_trial.json`. The first direct-solve probe was recorded in
the console, not in a separate raw file. Solver/demo defaults and the user's existing
staged work are preserved; nothing is staged or committed in this trial.

### 2026-09-09: nonlinear smoothing experiments (not a selected default)

The acceptance goal is still **not met**. All figures below use the unchanged
T-shirt trajectory, ten substeps, and 1,200 frames. Timings exclude setup,
capture and diagnostic readbacks. The terminal window is frames 901--1,200.
These are separate nondeterministic trajectories, not matched-state errors.

| Experimental schedule | Wall ms/frame | Terminal speed mean / P95 (m/s) | Net / path RMS (mm) |
| --- | ---: | ---: | ---: |
| Residual-selected GS, four scans | 56.004 | 0.003019 / 0.005238 | 3.768 / 4.749 |
| Residual-selected GS, six scans | 66.189 | 0.002773 / 0.004047 | 2.726 / 3.662 |
| Central-difference right-preconditioned JFNK, four vectors | 57.603 | 0.003501 / 0.006023 | 6.473 / 9.155 |
| Nonlinear GS-GMRES, four scans, four-vector history | 61.124 | 0.002623 / 0.002972 | 2.319 / 3.176 |
| Nonlinear GS-GMRES, six scans, four-vector history | 79.776 | 0.002032 / 0.002583 | 2.133 / 3.061 |
| GS-GMRES, six scans, frozen norm and force-only queries | 71.641 | 0.003038 / 0.003505 | 2.929 / 3.671 |
| GS-GMRES, six scans, color-local contact gather | 131.864 | 0.003862 / 0.011704 | 6.953 / 9.072 |
| GS-GMRES, six scans, no projection in norm queries, one mixing trial | 73.406 | 0.002619 / 0.003029 | 2.216 / 2.945 |
| Same reduced-work schedule, eight scans | 88.404 | 0.002488 / 0.003130 | 3.141 / 3.676 |

All scene final checks passed, but these checks do not certify self-contact
safety or absence of jitter. None of the above justifies replacing the current
default: lower terminal motion alone is insufficient at this cost.

The separate `ngmres_gs_probe.log` matched-state test gave:

| Schedule | Local Newton defect RMS (um) | Force RMS (N) | Position RMS relative to 30 sweeps (um) |
| --- | ---: | ---: | ---: |
| Current residual globalization | 4.681 | 6.747 | 14.251 |
| Nonlinear GS-GMRES, four scans | 1.738 | 1.358 | 19.820 |
| Nonlinear GS-GMRES, six scans | 1.276 | 1.079 | 17.625 |
| Ordinary 30 sweeps | 0.890 | 2.104 | 0 |

This is evidence of improved local force balance, **not** evidence that all
solution-error measures improve. A 100-sweep same-state diagnostic is being
added to distinguish local defect from global position error.

The color-local gather prototype used fixed detector-slot-indexed links,
four VT endpoint links and two owning-edge EE links. CPU/CUDA and graph replay
tests matched original forces/Hessians, including asymmetric EE and a device
material table. Nevertheless the full-frame cost increased sharply. Its two
new source/test files and solver hooks were removed; the raw benchmark record
is retained in `ngmres_gather_trial.json`. It is not a retained solver option.

The existing temporal warm-start switch also failed this trajectory:
27.700 ms/frame but mean/P95 terminal speed 0.385/0.405 m/s. It remains disabled.
The right-preconditioned matrix-free experiment did not reproduce the CPU
left-preconditioned oracle: even with 20 inner PCG steps the estimated Krylov
residual squared stayed about 0.98--0.996 of its initial value. The Arnoldi
orthogonality check was small (about 1e-7--3e-6); simply increasing the inner
iteration budget was not a successful remedy.

Current follow-up experiments compare full-history acceleration of global
Newton steps and inexact **left** preconditioning. They remain private research
hooks, with unchanged demo/default configuration. No velocity damping, phase
trigger, material change or weakened DAT was introduced by these experiments.

Follow-up: full-history acceleration of global steps also failed the cost
gate: four/six steps cost 110.373/169.492 ms per frame. Terminal speed mean/P95
was 0.004125/0.007044 and 0.002677/0.004967 m/s respectively. The six-step
matched-state probe had looked promising (1.171 um local defect versus 1.032
um for 30 sweeps; error relative to 100 sweeps 47.506 versus 55.050 um), but
that single-state observation did not translate into an acceptable trajectory.
Block-reducing the GS-GMRES history Gram matrix gave 75.427 ms/frame and
0.002535/0.003784 m/s, still No-Go. The history, selective nonlinear smoother,
and spectral membrane experimental modules/tests/hooks were removed after
these trials; raw JSON/log files remain. Do not try to reproduce removed
schedules using the remaining benchmark's current mode list.

Two additional hypotheses were checked before further solver changes:

- 4,096 synthetic near-parallel directed EE pairs on CPU had relative net
  action/reaction error at most 1.57e-7 and no one-sided activation. This test
  does not support changing canonical pair semantics to fix this jitter.
- On one shared T-shirt state, local 3x3 Hessian condition numbers were roughly
  median 8--17, P95 32--39, maximum 205--426. Against an independent double
  precision solve using freshly queried forces, local-step RMS discrepancies
  were about 5.6e-6 um. Local matrix inversion precision is not the bottleneck
  in that sample; no mixed-precision solver change was made.

Changing only the experimental residual acceptance norm from F^T H^-1 F to
the normalized squared local position defect reduced matched-state defect
3.741 to 2.681 um and error relative to 30 sweeps 15.321 to 12.599 um. It is
not yet equivalent to the 30-sweep result (0.862 um local defect). A full
trajectory check is running; the default acceptance norm is unchanged.

### 2026-09-09: Lagged-dissipation and within-step secant research (in progress)

The position-norm acceptance trial finished at 37.527 ms/frame with terminal
mean/P95 RMS speed 0.004550/0.005863 m/s. It does not meet the settling goal.
The default residual merit is unchanged.

A research-only energy difference query freezes friction normal load,
tangent and interpolation weights at each line-search base, not over time.
Normal contact and elasticity are refreshed. This is a lagged-dissipation
surrogate, not a scalar potential for the fully refreshed Coulomb system.
Original forces are still queried between steps. DAT applies to every
accepted motion and retry; rejection restores both position and cumulative
DAT displacement. Overflow and nonreciprocal directed EE rows invalidate
the energy query; ordinary directed GS remains the rejection fallback.

The fixed-base body/EF energy implementation initially lost accuracy by
subtracting interpolated absolute positions. It now interpolates particle
increments in double precision; base force semantics are unchanged. CPU and
CUDA tests check VT/EE and body point/EF energy gradients. A CPU/CUDA Graph
replay test checks backtracking, invalid-query rejection and q/D rollback.
These are targeted checks, not full solver/general-demo certification.

| Research schedule | Wall / GPU ms/frame | Terminal mean / P95 RMS speed (m/s) |
| --- | ---: | ---: |
| Energy fine4 + ordinary polish4 | 66.003 / 65.997 | 0.002171 / 0.003898 |
| Energy fine3 + ordinary polish2 | 53.874 / 53.868 | 0.002803 / 0.004089 |
| Energy fine1 + ordinary polish4 | 50.936 / 50.930 | 0.002372 / 0.004816 |
| Energy coarse only + ordinary polish6 | 54.400 / 54.395 | 0.002366 / 0.003488 |

All rows use 1200 unchanged T-shirt frames and the last 300-frame window.
The first two preceded the increment-precision and reciprocal-row guard
fixes; do not attribute differences between these rows solely to schedules.
Final checks pass, but do not certify no self-intersection or no jitter.
No new preset/demo default is selected. The goal is not achieved.

In one matched-state probe, fine4/polish4 gave local correction RMS/P95
0.954/1.680 um versus ordinary30 0.884/1.635 um, but force RMS was still
2.056 versus 1.689 N. Other same-state short schedules gave local RMS
1.130--1.249 um versus ordinary30 0.857 um. These probes are not trajectory
or throughput evidence; raw logs are under the ignored global_settling
output directory.

The next research direction uses within-substep L-BFGS history to avoid
repeated global matrix assembly/PCG. Its fixed eight-entry compact Gram
recursion matches an independent two-loop implementation on CPU/CUDA,
including ring wrap. History is discarded per substep and on a rejected
step; no velocity/trajectory/material adjustment is involved. A CPU-led
same-state diagnostic with twelve secant corrections and two GS sweeps gave
local RMS 0.998 um versus ordinary30 1.106 um and force RMS 0.647 versus
2.602 N. A GPU implementation and full-trajectory validation are in progress;
this does not yet establish a general accuracy or speed improvement.

### Follow-up: failed accelerators and contact-motion invariance

The L-BFGS full-trajectory experiment did not reproduce the promising
single-state result: six/twelve corrections cost 80.934/117.262 ms/frame
and terminal mean RMS speeds were 0.005135/0.003736 m/s. Contact-centered
four-vertex Woodbury patches also failed the speed/settling target:
four/eight corrections cost 56.207/69.446 ms/frame with terminal speeds
0.003164/0.002398 m/s. These are research paths, not preset defaults.
Interpolated energy backtracking reduced retries in a 300-frame run, but
that window contains active manipulation and is not settling evidence.

A separate regularization diagnostic changed friction epsilon from 0.01
to 0.001 m/s. Ordinary30 terminal speed fell to 0.001049 m/s, whereas the
fast residual solver still gave 0.004689 m/s. Thus regularized friction
micro-slip does not explain all fast-solver jitter. This parameter change
is not an acceleration and was not applied to the demo/default solver.
Likewise, an ordinary1000 single-state oracle continued moving and did not
give a smaller local residual; it must not be called converged truth.

New regression tests exposed concrete floating-point contact errors:

- VT used two differently evaluated world-space interpolations for current
  and previous contact points. With identical particle positions and
  anchors, the spurious friction/damping force reached 1.487 N on CPU and
  1.481 N on CUDA in the constructed test (ke=300000, radius=0.002 m).
- EE similarly produced up to 1.283 N under an exactly represented common
  translation because current/previous world points were interpolated
  independently.
- Body point/face contacts produced about 0.714 N under common translation.
  The full backend additionally imported the public rigid helpers instead
  of its private copies, bypassing private fixes.

The correction evaluates material-point motion from per-vertex increments
and separates body translation from rotated local-point increments. It
preserves the mathematical force law, contact geometry, material values,
DAT and trajectory. Both particle and rigid reaction paths are updated.
CPU/CUDA soft/full rest/common-translation regressions pass after the fix;
expanded regression and trajectory validation are still in progress.

VT-only 1200-frame timing was 39.194 ms/frame (residual) versus 139.944
ms/frame (ordinary30), with terminal speeds 0.004061 versus 0.001383 m/s.
This does not meet the goal, nor does it establish a precise improvement
over separate nondeterministic trajectories. EE/body fixes were not loaded
in that run. No new demo or preset default is selected by this research.

The all-material-motion follow-up gave residual 38.513 ms/frame and
ordinary30 139.322 ms/frame, with terminal mean speeds 0.007682 and
0.001277 m/s respectively. The fast trajectory did not improve its
settling target; correcting a floating-point invariance error is not a
convergence guarantee. A further EE normal-force test found 0.03224 N
variation under an exactly represented common translation. Evaluating
the edge separation locally fixes that test; this last geometry fix was
not loaded in the all-material-motion benchmark above.

The expanded contact tests (15 tests, CPU/CUDA) pass. The dense-reduction
fixture is offset by 1 mm to avoid exact cancellation of its angular-linear
block: the finite zero-slip tangent caused different float32 reduction
trees to differ by 0.001465 at a mathematically zero entry. Existing
relative/absolute comparison tolerances were not loosened. Full discovery
before this fixture adjustment ran 148 tests with that one failure.

### Analytic frozen-geometry friction Jacobian (research)

`friction_jacobian.py` builds the radial friction tangent and normal-load
cross derivative missing from the isotropic SPD approximation. This is
not the complete geometrically differentiated contact Jacobian. Its
nonsymmetric operator is solved by right-preconditioned GMRES, not CG;
the original force query, DAT and nonlinear acceptance remain authoritative.
All contact storage is fixed-capacity with device counts and overflow
rejection. It does not introduce a new friction/material model.

Independent finite differences of a frozen-normal/load contact law and
an independently assembled nonsymmetric stencil product pass on CPU/CUDA.
An initial implementation incorrectly rejected inactive asymmetric EE
rows; this was corrected before the full trajectory run. Only active
contact stencils require reciprocal rows, matching the base projection.

The analytic6 1200-frame result is still No-Go: 71.070 ms/frame, terminal
mean/P95 speed 0.004279/0.006857 m/s, final 0.006077 m/s. A matched-state
probe reduced force RMS from 2.287 to 1.373 N, but local correction RMS
worsened from 2.085 to 3.678 um (ordinary30: 0.759 um). This demonstrates
why lower total force-work residual alone is not sufficient acceptance.
A joint force-work/displacement-defect acceptance filter is being tested;
it is research-only and does not change the default merit policy.

Follow-up 1200-frame trials remain below acceptance:

| Research variant | Wall ms/frame | Terminal mean / P95 speed (m/s) |
| --- | ---: | ---: |
| Joint residual filter, original PCG | 38.337 | 0.004016 / 0.004992 |
| Joint residual filter, analytic GMRES6 | 69.088 | 0.009431 / 0.020477 |
| Fixed substep residual metric, original PCG | 41.448 | 0.003926 / 0.005670 |
| Fixed substep residual metric, analytic GMRES6 | 67.183 | 0.007678 / 0.013708 |

The fixed-metric test verifies that increasing the current Hessian alone
cannot improve the fixed residual norm. This is a mathematical acceptance
property, not evidence that it solves the trajectory problem. All tests
for analytic tangents, nonsymmetric products, the joint filter and the
fixed metric pass on CPU/CUDA. None of these modes becomes a default.

Warp's `closest_point_edge_edge` treats epsilon as a squared-edge-length
degeneracy threshold (confirmed in installed `warp/native/intersect.h`),
not solely a near-parallel test as the private solver parameter describes.
At the current 1e-5 value, 39 of 19174 rest-shape T-shirt edges are treated
as points. A diagnostic setting epsilon=1e-12 consistently for detection,
force and DAT costs 38.475 ms/frame on the fast path and 139.603 on
ordinary30. Terminal speeds remain 0.005340 and 0.001468 m/s respectively.
This does not establish a jitter remedy; no default epsilon change was
made. In a separate deformed-state probe, 40 edges met the degeneracy
threshold, but the largest local residual was not incident on such an edge.

A CPU-only frozen-elastic/nonlinear-contact inner solve was also checked
before committing to a GPU implementation. Four/ten inner iterations
reduced original force RMS to about 0.894 N (ordinary30 0.856 N), but local
defect remained 2.448/2.447 um (ordinary30 0.730 um), and position error
against ordinary30 was about 115 um. The second outer correction was
rejected. The normal load was linearized and damping phase frozen inside
this oracle; actual force and DAT were checked outside. This is not a
successful full-force solver and has not been ported to GPU.

Next bounded trial: regularize only the global linear step in its local
block Hessian metric, to control weak global modes. This is Newton-step
regularization, not physical or velocity damping. The original matrix is
restored exactly before the nonlinear acceptance check. CPU/CUDA tests,
including repeated CUDA Graph replay, verify the diagonal modification
and bitwise restoration. Full trajectory validation is recorded below.

### Further globalization experiments (not enabled by a preset)

The full 1200-frame results still do not meet the settling target. In
particular, one-state accuracy must not be reported as full-trajectory
equivalence to ordinary30. All runs retain the trajectory, material,
friction, substep count and DAT; no terminal velocity decay is applied.

| Variant | Wall ms/frame | Terminal mean / P95 speed (m/s) |
| --- | ---: | ---: |
| Diagonal metric shift 0.001 | 37.906 | 0.003801 / 0.004881 |
| Diagonal metric shift 0.01 | 37.602 | 0.004181 / 0.005569 |
| Diagonal metric shift 0.1 | 38.628 | 0.004478 / 0.008784 |
| Shift 0.1, strict five-step limit | 48.924 | 0.003099 / 0.007288 |
| Shift 0.1, twelve-step maximum-defect guard | 65.419 | 0.004673 / 0.012883 |
| Frozen-tangent modified Newton6 | 97.393 | 0.010170 / 0.020313 |
| Frozen tangent with shift 0.1 | 77.878 | 0.006896 / 0.010017 |
| Refactor rejected tangent, Newton6 | 120.790 | 0.006910 / 0.016446 |
| Refactor rejected tangent, shift 0.1 | 73.282 | 0.004402 / 0.006923 |
| Radial friction tangent, shift 0.1, three steps | 65.727 | 0.029411 / 0.081552 |
| Radial friction tangent, shift 0.1, five steps | 89.736 | 0.014161 / 0.017142 |
| Global plus up to sixteen residual-controlled GS sweeps | 87.144 | 0.004149 / 0.008029 |

The strict shift-0.1 matched-state correction RMS was 0.715 um against
ordinary30's 0.727 um, but that did not translate into sufficient settling.
The independent force-only query agrees with assembled force to about
2.7e-7 N RMS in the modified-Newton comparison. Tangent reuse failures are
therefore not explained by a different force law in that query. Adaptive
refactoring performed 35622 / 21989 rebuilds without / with the shift;
the old local metric was retained for comparisons across each rebuild.

The radial tangent keeps normal load fixed only in the linearization,
retains the original force in residual checks, and uses SPD PCG. Matrix
diagonals and raw contact heads/count/overflow are restored after solving.
Six CPU/CUDA derivative/restoration tests passed, including repeated Graph
replay. This correctness coverage does not establish trajectory quality:
the full runs above failed, and no default selects the radial tangent.

The residual-controlled GS experiment used 72342 additional sweeps over
12000 substeps. Its decisions contain no trajectory time or terminal-frame
test. Neither this run nor the radial runs exhibited sampled bending
degeneracy-threshold switches; that particular mechanism did not explain
their terminal motion.

A CPU-only TALS trial used the frozen contact frames and closest approach
to the regularized stiction disk, with a pi/3 angular bound, following
Castro et al., *A Transition-Aware Method for the Simulation of Compliant
Contact with Regularized Friction*, Section III-A
(https://arxiv.org/abs/1909.05700). It avoided inner rejected steps in the
sampled run but worsened original local defect: 3.702 um versus 2.870 um
without TALS and 0.702 um for ordinary30. Position difference from
ordinary30 was 360 / 210 um. It has not been promoted into the GPU solver.

Current bounded checks separate two remaining hypotheses: a symmetric
color-triangular preconditioner to improve the linear direction, and a
constant-velocity nonlinear initial guess that leaves the inertial target
unchanged. Independent CPU/CUDA dense-matrix and Graph-replay checks for
the symmetric preconditioner pass. Energy-merit and initialization
trajectory checks remain research, not validated defaults.

### Follow-up: reject more approximate directions; accelerate exact work

The following additional 1200-frame experiments remain No-Go. They do not
select a preset or change the demo. Velocity-guess runs also visibly change
the fold outcome; their short runtimes are not acceptable speedups.

| Variant | Wall ms/frame | Terminal mean / P95 RMS speed (m/s) |
| --- | ---: | ---: |
| Energy2 + two GS sweeps, radial tangent | 60.535 | 0.004154 / 0.005779 |
| Energy2 + two GS sweeps, colored SSOR | 112.337 | 0.002804 / 0.004781 |
| Constant-velocity initial guess, residual controller | 39.995 | 0.113506 / 0.136680 |
| Previous-position initial guess, residual controller | 73.207 | 0.001585 / 0.001908 |
| Constant-velocity guess, energy1 + four GS sweeps | 23.416 | 0.077484 / 0.097213 |
| Energy-selected initial guess, residual controller | 80.907 | 0.001266 / 0.001784 |
| Energy-selected initial guess, energy1 + four GS sweeps | 56.550 | 0.002960 / 0.007617 |
| Energy-selected initial guess, energy2 + two GS sweeps | 71.618 | 0.004711 / 0.008176 |
| Nonlinear merit only, energy1 + four GS sweeps | 50.828 | 0.002869 / 0.007313 |
| Nonlinear merit only, residual controller | 37.553 | 0.004774 / 0.006940 |

The Euclidean linear residual is not monotone under preconditioned CG.
An independent 2-block SPD counterexample passes on CPU/CUDA: one step
reduces quadratic energy while growing the Euclidean residual tenfold.
Disabling that gate was tested only behind actual nonlinear acceptance;
finite values, curvature, overflow, clamp and DAT guards remain. This did
not establish the requested trajectory improvement and is not a default.

The research energy query now computes EE geometry using local differences,
and rejects asymmetric EE rows only when active at the base or trial.
Inactive asymmetric rows must not invalidate an otherwise valid energy
query. Focused CPU/CUDA tests cover inactive, active and activating cases.

The frozen-contact CPU oracle's directional derivative audit gives errors
around 1e-7 and linear backerrors around 1e-13. Broyden inverse secants on
the actual force did not repair the outer problem: local defects of about
2.12 um versus ordinary30's 0.677 um in one matched-state run; another
run worsened to 6.01 um. These CPU oracles have not been ported to GPU.

Two independent diagnostics also failed to explain/remove the jitter:
translating the complete physical scene to the table-centered origin gave
37.973 ms/frame and 0.004097 m/s for the residual controller; ordinary30
gave 138.050 ms/frame and 0.001580 m/s. Twenty substeps with one fine
global correction gave 52.189 ms/frame and 0.004522 m/s. Neither the world
origin nor substep count is changed in the demo.

Terminal diagnostics now separate velocity fluctuations from mean drift.
Smooth regularized friction can sustain micro-slip; total path length alone
is not an oscillation measure. No artificial velocity decay or freezing is
used, and ordinary30 is a reference budget, not a converged exact solution.

New bounded implementation experiment: a **color-bucket contact stream**,
not the previously rejected serial particle gather or unbucketed stream.
Each directed EE row is present once for each distinct owning-endpoint
color; each VT row once per distinct endpoint color. Detection count/fill
uses fixed worst-case capacity and device prefix offsets. At each color,
the original contact function is evaluated at current positions. DAT is
unchanged. CPU/CUDA and repeated Graph tests cover repeated endpoint
colors, asymmetric EE, device material changes and clamped overflow rows.

The first 1200-frame ordinary30 run with buckets costs 98.171 ms/frame,
terminal mean/P95 speed 0.001336/0.001763 m/s, final speed 0.000995 m/s.
This is promising compared with recent unmodified ordinary30 runs near
139 ms/frame, but paired order-swapped timing and more scene coverage are
still required. No preset installs the research module.

A separate DAT identity certificate is under test. It bounds distance to
every cached division plane in the allowed blend interval [0.05, 0.95],
with conservative roundoff slack. Only when every displacement ball stays
on its side does it skip pair evaluation; otherwise it executes original
DAT. The original global displacement cap and output rounding still apply.
Randomized VT/EE CPU/CUDA tests, including unsafe fallback and Graph replay,
are bitwise equal to the cached DAT reference. Full-scene hit rate and
cost are pending; this is not a general no-self-intersection certificate.

### Follow-up: exact-work throughput and coloring experiments (not defaults)

The global conditional DAT certificate was rejected: although roughly 78%
of queries certified identity, the extra check/flag/branch nodes increased
ordinary30 with color buckets to 139.454 ms/frame. Moving the sufficient
certificate inside each existing cached row instead gave 91.188 ms/frame
(90.971 in a later run). Certified rows skip plane evaluation only; unsafe
rows still execute the original DAT and the original displacement cap stays.
Randomized safe/unsafe VT/EE tests remain bitwise equal on CPU/CUDA and Graph.

Additional local/global schedules failed to establish a quality/cost win:

| Schedule, with buckets and row certificate | ms/frame | Terminal mean / P95 speed (m/s) |
| --- | ---: | ---: |
| Six GS + coarse/fine + four GS | 67.729 | 0.001991 / 0.004661 |
| Ten GS + coarse/fine + four GS | 106.862 | 0.004507 / 0.008658 |
| Six GS + fine only + four GS | 56.383 | 0.002841 / 0.005546 |
| Six GS + coarse only + four GS | 62.155 | 0.002114 / 0.004866 |
| Ordinary12 | 50.729 | 0.001755 / 0.003395 |
| Ordinary16 | 60.293 | 0.001600 / 0.005236 |

In a common frame-900 state, six-GS/coarse/fine/four-GS gave local correction
RMS 0.830 um and force RMS 1.154 N, versus ordinary30 0.748 um / 0.670 N.
The smaller position defect alone does not demonstrate equivalent convergence.

The body EF scatter now skips records with no corner of the current color
before evaluating contact. Both private backends retain the same forces.
Twenty-two focused contact/projection/global tests passed; the separate
full-trajectory benefit of this early predicate is not established.

A CUDA-only research launch packs independent 16-thread surface groups into
32/64/128-thread blocks. Inactive/padded groups participate with zero values;
16-lane shuffle reductions do not cross particles. The initial tile-axis
implementation did not compile for vector/matrix tiles and was replaced.
CPU/gradient/default kernels retain the original path. Padded/inactive/
selective tests plus the existing Jacobi fusion tests pass (three tests).
Packing four groups yielded ordinary30 88.975 ms and ordinary12 51.176 ms;
thus packing alone is only a small candidate gain, not a convergence fix.

Fusing unchanged body and self-contact accumulation into one per-color
launch gives ordinary30 84.822 ms and ordinary12 49.291 ms in first runs.
CPU/CUDA tests include particle/edge/face body records, directed EE, duplicate
colors, device material changes, count overflow and Graph replay. Atomic
accumulation order changes, so complete trajectories can diverge; final-state
speed alone is not a paired numerical-equivalence measurement.

Construction-only DSATUR coloring reduced the T-shirt's nine structural
colors to eight; bounded tabu repair subsequently found a valid seven-color
assignment. Every original triangle/hinge/tet dependency is retained and
checked. No material, trajectory or substep is changed. Eight colors with
the preceding exact-work optimizations gave ordinary30 79.262 ms and
ordinary12 45.361 ms. The latter still had a different final strain and is
not an accepted ordinary30 substitute. The first seven-color ordinary30
run gives 73.853 ms, terminal mean speed 0.001369 m/s and fluctuation RMS
0.001075 m/s. Paired accuracy and independent trajectory checks are pending.

None of these measurements establishes the full user goal of near-ordinary30
accuracy at the current fastest cost with no visible terminal oscillation.
No demo/preset was changed, and no velocity decay or sleeping was introduced.

Alternating forward/reverse seven-color sweeps was also rejected. At a
shared frame-600 state, 30 alternating sweeps had local defect 1.715 um and
force RMS 4.578 N, versus original ordinary30 0.848 um / 1.420 N. The scan
order hook and command-line switch were removed after this diagnostic.

Seven-color ordinary20 gives 58.309 ms/frame over 1200 frames, terminal
mean/P95 speed 0.001085/0.001819 m/s, last speed 0.000584 m/s, and fluctuation
RMS 0.001060 m/s. Same-substep frame-600 local RMS is 0.756 um versus
ordinary30 0.738 um; frame-900 0.954 versus 1.830 um. The latter reference
has an outlier, so its RMS must not be treated as a clean convergence target.
At frame 300 both have large unresolved outliers (about 283 um RMS); P95
is 4.530 um for the candidate versus 3.712 um for ordinary30.

Budget redistribution to 15 substeps with ten seven-color sweeps gives
54.428 ms/frame, terminal mean/P95 0.000942/0.001159 m/s and fluctuation
RMS 0.000646 m/s. This follows the conditioning motivation of
[Small Steps in Physics Simulation](https://matthias-research.github.io/pages/publications/smallsteps.pdf),
but that paper's XPBD results do not prove equivalence for this VBD schedule.
Collision detection still runs each substep and DAT remains unchanged.

**The full-frame check is stricter and has not passed.** A new diagnostic
forks a complete frame, including source IK and every substep, from the same
state. At frame 900 seven-color 20 differs from original ordinary30 by
0.291 mm RMS (0.574 mm P95); 15x10 differs by 0.402 mm RMS, and differs from
its own same-dt 15x30 reference by 0.834 mm RMS. Original ordinary30 A/A
repeats give 0--0.00172 mm in the sampled checkpoints. Therefore a close
single-substep residual and lower terminal speed are insufficient acceptance
evidence. Neither schedule is promoted; the iteration error accumulates
through the updated velocities over the frame.

An additional frame-900 fork includes ordinary60 as a larger-budget reference
(not an exact physical solution). Original ordinary30 differs from ordinary60
by 0.662 mm RMS; seven-color ordinary30 differs by 0.675 mm, while its
difference from original ordinary30 is 0.0319 mm. Seven-color ordinary20
differs from ordinary60 by 0.950 mm and the current fast residual path by
0.886 mm. Thus retaining the actual 30 sweeps is presently the more reliable
precision direction. Seventy-five focused existing regression tests passed
before the following execution experiment; this is not a full repository
or three-demo certification.

### Resident cooperative GS execution experiment (not a preset)

The RTX 5060 Ti reports cooperative-launch support. Official CUDA
`cooperative_groups::this_grid().sync()` is used, with driver occupancy
validation; no software spin barrier is introduced. A standalone 64-round
cross-block exchange test is bitwise correct for 1031, 8193 and 9217 entries,
including ten CUDA Graph replays. This research adapter currently requires
installed CUDA SDK headers; unsupported devices/backends are not enabled.

The fused kernel retains all color/sweep boundaries, recomputes contacts at
the current iterate and applies DAT after every color. Importantly, the
original body-particle penalty update precedes the particle solve each sweep.
Membrane, bending, contact and DAT math reuse existing private functions;
there is no trajectory-dependent velocity decay or material/time-step change.

The initial seven-color resident ordinary30 run is **No-Go for performance**:
151.414 ms/frame over 1200 frames, terminal mean speed 0.001581 m/s and
fluctuation RMS 0.001347 m/s. Fusion alone is substantially slower than the
73.853 ms split-kernel candidate. A second trial increases resident occupancy
and applies the previously tested sufficient DAT identity certificate inside
the grid, avoiding standalone certificate/check/conditional Graph nodes.
Failure to certify still executes original DAT; the global displacement cap
is unchanged. Its full-trajectory result is pending.

The new resident test compares 1/4/30 sweeps with ordinary launches on pinned,
nearly coincident cloth sheets with active self contact and ground friction,
including three reset/replay executions. It passes with a 5 um componentwise
position tolerance (ordinary30 A/A alone differs by up to 2.3 um on this
atomic-contact fixture), and the corresponding 0.003 m/s velocity tolerance
at dt=1/600. Existing color-contact and identity-certificate tests also pass.
These tests do not prove absence of visible terminal jitter or equivalence
over a complete robot trajectory. Demo/preset defaults remain unchanged.

**Correction to the initial resident trajectory measurements:** the command
cache initially keyed only state/contact identities and dt, omitting captured
scalar material values. The T-shirt captures several friction phases against
the same buffers, so both the 151.414 ms run above and a subsequent 75.563 ms
certificate/occupancy run used stale self-friction in later phases. Neither
is a valid same-material trajectory comparison. An isolated two-sheet,
self-contact-only multi-material Graph regression fails before the key fix
(346/546 coordinate components outside 2 um tolerance) and passes afterward.
The earlier test with a ground collider did not detect this because its
separately initialized body-contact material changed with the scalar.

After the fix, 16 focused contact/identity/resident tests pass. The corrected
1200-frame resident ordinary30 run is 75.273 ms/frame, terminal mean/P95
speed 0.001534/0.001983 m/s and fluctuation RMS 0.001167 m/s. This still does
not improve the 73.853 ms split candidate or meet the roughly 38 ms fast-path
target. A complete-frame-900 fork (which installed the experiment only after
selecting the current material, hence was not affected by the multi-Graph
cache issue) differed from ordinary30 by 0.00214 mm RMS with original colors;
seven-color resident30 differed by 0.0426 mm, versus seven-color split30
0.0409 mm. No default is changed; this execution experiment is not promoted.

A higher-budget full trajectory with seven-color split ordinary120 costs
219.556 ms/frame and does not settle cleanly throughout frames 900--1200:
mean speed 0.015015 m/s, fluctuation RMS 0.034122 m/s, last speed
0.000914 m/s. Its final cloth bounds and strain differ substantially from
ordinary30, so this separate trajectory cannot prove that more iterations
are intrinsically worse. A common-state terminal continuation is needed to
separate solver jitter from a different fold/release event.

An offline static-friction feasibility query was added, not a solver path.
It freezes geometry, removes velocity-dependent forces, adds gravity, and
uses projected FISTA to fit tangential tractions within the current penalty
normal-load disks. Analytic sticking/sliding block checks pass. At a
fast-path frame-1200 snapshot, fixed-normal static tractions only reduce the
mass-weighted squared force residual by about 2% (force RMS 4.150 to 4.108 N).
Thus absence of set-valued static friction alone is not established as the
main cause. A negative-load record made the initial reported disk-violation
metric incorrect; the optimization already clipped the load to zero, and
the diagnostic now uses the same clipped bound. A free-normal cone query
is explicitly counterfactual: it ignores gap complementarity and is not a
hard-contact or simulation-accuracy certificate. No material or friction
model has been replaced on the basis of these diagnostics.

### September 9: terminal continuation and complete contact derivatives

These are offline experiments, not an accepted preset or a claim that the
requested accuracy, speed and settling target has been achieved.

* The corrected free-normal cone query at a frame-1200 snapshot reduces
  force RMS from 4.82818 to 0.18187 N, and mass-weighted squared residual to
  0.0013604 of its original value. Fixed-normal tangential disks only reduce
  it to 0.970638. The free-normal query ignores gap complementarity and
  cannot certify a physically valid static equilibrium.
* Replacing the inner linear normal load with the original reciprocal-gap
  law passes an analytic scalar-root regression but brings almost no
  additional reduction in the frame-900 frozen-contact experiment. With
  extended backtracking both variants reach about 0.918 um local correction
  RMS and 1.292 N force RMS, versus ordinary30's 0.717 um and 1.187 N.
* The full geometric bending tangent passes independent double-precision
  energy differences, symmetry and rigid-translation checks. Adding it to
  the frozen-contact sparse oracle does not reach ordinary30 accuracy:
  local correction RMS 1.354 versus 0.718 um in that paired run.
* A CUDA local-diagonal version of the same bending tangent also fails to
  accelerate convergence. At frame900 its 12/20-sweep position errors from
  ordinary30 are 15.899/8.578 um, versus unmodified 15.891/8.570 um.
  The original bending force is bitwise unchanged and CPU/CUDA diagonal
  derivatives match the independent reference. Correct derivatives alone
  are not an improvement certificate; this kernel option is not promoted.

The complete-frame probe now also supports a common-state continuation.
Every candidate starts at exactly the same fast-path frame1200 with the
same particle states, robot inputs and material. The following 300 frames
retain all original motion, contact and damping laws; no velocity reset,
sleeping or terminal decay is introduced. Optimized traversal/packing is
used for ordinary sweeps without changing the coloring.

| Continuation | Mean speed (m/s) | Fluctuation RMS (m/s) | Last speed (m/s) |
| --- | ---: | ---: | ---: |
| ordinary30 | 0.001651 | 0.001322 | 0.001631 |
| ordinary60 | 0.002827 | 0.002017 | 0.001962 |
| ordinary120 | 0.025625 | 0.044255 | 0.001721 |
| fast residual | 0.004206 | 0.004162 | 0.004080 |
| ordinary30 A/A repeat | 0.001757 | 0.001504 | 0.003782 |

Ordinary120 releases substantial stored deformation (63.37 mm net RMS
motion); it is not a clean static-accuracy reference. Ordinary30 A/A final
positions differ by 0.125 mm RMS after the continuation, and instantaneous
last speed is noisy. These results reject the assumption that merely
raising the sweep budget necessarily eliminates terminal jitter; they do
not identify a unique cause. Performance figures from this diagnostic are
not a release benchmark (CPU-only analysis overlapped part of the run).

An offline contact oracle differentiates the actual private VT/EE force
evaluators, including changing closest-point weights, normals, friction
load and contact damping. CPU finite differences agree within 0.5% relative
matrix norm on nondegenerate test contacts. Warp deliberately bounds the EE
adjoint near parallelism, so that region must not be called an exact tangent.
The resulting matrix is nonsymmetric and is solved by sparse LU, not CG.
At one frame900 fork, six outer trials plus ordinary polishing reduce force
RMS to 0.264 N versus ordinary30's 0.956 N, but position differs by 121.9 um;
this fails the near-ordinary30 position test. A second fork with 100/1000
sweeps confirms that lower force residual is not sufficient: ordinary1000
itself retains 0.678 N force RMS and differs from ordinary30 by 418.8 um.
No runtime solver or material is replaced by this offline experiment.

A separate pending experiment targets simultaneous same-color contacts.
Elastic graph coloring does not forbid contact between two same-color
vertices. For a frozen contact Hessian with blocks b_i*b_j*H, Cauchy-Schwarz
gives the block-diagonal upper bound n*diag(b_i^2*H), where n counts vertices
of that contact updated simultaneously. Only those contact diagonal blocks
are enlarged; forces, non-contact blocks, material, time step and DAT stay
unchanged. This bound is not a global bound on the moving-geometry nonlinear
force. CPU/CUDA tests cover original-force equality, directed EE ownership,
per-color multiplicity and Graph replay; a common-state settling comparison
is pending. No preset selects it.

The same-color bound is **No-Go as a settling solution**. In its 300-frame
common-state continuation, ordinary30 mean/fluctuation speed is
0.003411/0.003754 m/s and bounded30 is 0.003339/0.003610 m/s. Bounded12
ends at 0.000615 m/s but differs from ordinary30 by 9.99 mm RMS after the
continuation. The lower final speed does not establish comparable dynamics.
Bounded30 still differs by 1.57 mm (ordinary30 A/A is 0.050 mm). This is not
the requested accuracy/performance/stability improvement.

Another ablation freezes the actual self-friction basis, barycentric weights
and load at the physical substep anchor, following the lagged friction
potential in the [IPC technical supplement, section 9](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf).
Normal force and contact damping still use current geometry; body friction
is unchanged. This changes finite-iteration friction dynamics, unlike a
mere search-tangent adjustment. Force equality at the reference, zero-slip
invariance and nonpositive slip work pass on CPU/CUDA. Candidate membership
and DAT stay unchanged, but lagged tangential force can persist for one
inner solve after normal separation, as in the lagged potential; this is
an approximation, not an exact current-load friction evaluation.

Its common-state 300-frame results are also **No-Go**:

| Variant | ms/frame | Mean / fluctuation speed (m/s) | ordinary30 final q RMS (mm) |
| --- | ---: | ---: | ---: |
| ordinary30 | 87.30 | 0.001321 / 0.001074 | 0 |
| lagged12 | 56.61 | 0.001812 / 0.001948 | 4.983 |
| lagged20 | 73.96 | 0.001116 / 0.000938 | 2.669 |
| lagged30 | 93.24 | 0.001423 / 0.001054 | 0.808 |
| ordinary30 A/A | 87.30 | 0.001354 / 0.001111 | 0.029 |

No default, demo, material or trajectory is changed on the strength of
either trial. The unsuccessful local geometric-bending kernel option has
been removed; its double-precision offline derivative reference remains.

Next investigation: formulate the existing finite-stiffness normal
potential through an auxiliary gap and a compliant augmented Lagrangian.
For penalty rho and dual load lambda, eliminate the gap by solving
N = N_original(d + (N-lambda)/rho). Then the primal tangent is
k_original*rho/(k_original+rho), with dual update lambda <- N after a
complete primal sweep. At a fixed point lambda=N and the original force
law is recovered. This is not the existing experimental hard-contact ALM
(which changes the target contact law). A runtime implementation and
validation are still required; no benefit is claimed.

### 2026-09-09: archive all unaccepted experiments; retain only this log

**User decision:** stop the experiments and remove the accumulated code
changes. Restore solver, demo, test, benchmark, and changelog files to
`8e72b4b91e1c9028bc7ca43e8510abb64b04383e` on `FAST_MJVBDV2`.
Retain the optimization history in this document only. This includes
removing the staged `surface-global` implementation, not just its later
unstaged experimental extensions. It is an archival cleanup, not a claim
that the requested accuracy, settling, and performance goals were met.

All earlier entries describing these uncommitted prototypes as "retained",
"pending", "in progress", or available through an option are historical
experiment descriptions. They do **not** describe the post-cleanup source.
Previously committed optimizations remain in HEAD and are not rolled back.
References above to experimental source/test/probe files are archival:
those files are intentionally absent after this cleanup.

#### Inventory and final disposition

The detailed measurements and qualifications in the preceding September
8--9 entries are preserved. The following index covers the code removed
in this cleanup, including variants that only reached a formula/oracle test.

| Attempt / removed implementation family | Finding and disposition |
| --- | --- |
| Nonlinear coarse/fine Newton, `surface-global`, `global_newton`, fine-block Schwarz, extended multilevel and split PCG | Some same-state residual/position improvements, but no combined proof of near-30-sweep accuracy, low terminal jitter, and acceptable frame cost. Remove the preset, integration, demo switch, and associated tests/changelog. |
| Residual/backtracking, energy and lagged-contact merits, force-only residual queries, energy initialization | Lower residual or accepted energy proxy did not reliably imply comparable positions or settling. Remove controllers and diagnostic hooks. |
| Frozen/fixed metrics, regularized global directions, maximum guards and shifted solves | Tradeoffs remained scene/state-dependent; no robust joint quality/performance win. Remove. |
| Matrix-free Newton / FD-JFNK / GMRES and frozen-geometry friction Jacobian | Krylov convergence or total runtime insufficient; analytic contact tangents did not solve the nonlinear settling problem. Remove. |
| Modified Newton, frozen tangent reuse and refactor-on-rejection | Tangent reuse worsened convergence or required costly refactorization. Remove. |
| L-BFGS, Anderson/history, GS-GMRES and local/global cycles | Promising local probes did not reproduce the required full-history result. Remove remaining code. Earlier removed variants stay documented only. |
| Radial friction solve, colored SSOR, contact-patch Newton / overlapping contact corrections | Did not achieve the requested accuracy/speed/stability combination. Remove. |
| Nonlinear/selective polishing and previous-position / velocity / temporal initial guesses | Extra work or altered transient convergence without a robust terminal improvement. Remove experimental paths; keep only HEAD's existing behavior. |
| Spectral membrane and geometric bending tangents, geometric tangent oracle | Correct derivative checks did not establish a better full simulation. Runtime spectral/geometric trials had already been rejected; remove remaining offline helpers. |
| Full moving-contact derivative oracle / sparse LU | Force residual could decrease while position error against ordinary30 increased. Near-parallel EE adjoint is bounded, not an exact derivative there. Remove oracle/probe. |
| Static-friction-cone / frozen-contact CPU oracle and exact radial friction-law checks | Diagnostic only; no validated full complementarity/static-friction implementation or settling fix. Remove. |
| Contact-majorized same-color blocks | Preserves original force for the frozen test, but short-budget continuation diverges from the reference; lower last speed is not acceptance. Remove. |
| Lagged self-friction load, basis and barycentric weights | Changed finite-iteration friction behavior and did not meet the reference/settling gate. Remove. |
| Compliant normal ALM auxiliary-gap formula | Scalar-root/derivative tests only; no integrated runtime or scene benefit. Stop and remove; do not label as a completed solver improvement. |
| Color contact stream, body-contact fusion and early EF color filtering | Exact-work throughput candidates, but the combined requested acceptance is missing. Remove the uncommitted stream/fusion/early-return changes. |
| Surface packing (2/4/8 particles), DAT identity/row certificate and worker refactors | Local equivalence/Graph tests and throughput results exist, but no accepted end-to-end configuration meeting the task. Remove all uncommitted variants. |
| Recoloring / DSATUR / tabular repair and fewer-substep schedules | Reduced launch cost, but lower sweep/substep counts failed matched precision checks. Remove helpers and modes. |
| Resident cooperative CUDA GS and conditional-graph experiments | Valid launch/material-isolation tests do not overcome measured throughput No-Go. Remove native-launch helpers and probes. |
| VT/EE/body contact-motion invariance and finite zero-slip tangent fixes | Reproducible formula issues, but not a demonstrated settling solution; HEAD-isolated results below. Per user request, remove these uncommitted fixes too and preserve evidence here. |
| Full-backend cache-call, PCG dispatch and test-fixture compatibility edits | Fixes/support for the discarded working-tree experiment, not a reason to keep its dependencies. Restore HEAD along with the experiment. |
| Benchmarks, per-frame/EE-reciprocity probes, derivative tests and new regression modules | Archive together with the code; no leftover imports or public options referencing removed implementations. |

#### Baseline correction: committed HEAD is not the experimental default

The user explicitly clarified that "current default" excludes the working
tree changes. HEAD `8e72b4b9` uses `surface-fast`, multilevel disabled, and
seven batched sweeps followed by one ordinary sweep in the T-shirt demo.
It does not include the working tree's `surface-global` default.
Measurements of the latter must not be attributed to the former.

An independent detached worktree was created at HEAD. Its harness asserted
the imported Newton package path belonged to that worktree. On RTX 5060 Ti,
Warp 1.17.0, each full-history run used 1500 frames, 10 substeps/frame,
unchanged materials/trajectory, and CUDA Graph replay. The scripted motion
ends at simulation time 6.55 s; the terminal window below is approximately
20--25 s. Velocities/positions were sampled every ten frames.

| HEAD-isolated run | Wall ms/frame | Terminal RMS speed (mm/s) | Fluctuating speed RMS (mm/s) | Position range P95 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Unmodified HEAD default | 37.236 | 2.458 | 2.413 | 1.779 |
| Ordinary20, same caches and detection settings | 104.541 | 1.885 | 1.695 | 8.130 |
| Ordinary20 without preset, constructor defaults | 138.417 | 35.205 | 9.316 | 6.905 |
| HEAD + VT relative-increment fix only | 37.194 | 3.823 | 3.756 | 3.446 |
| HEAD + VT fix and finite zero-slip friction tangent | 36.683 | 2.977 | 2.902 | 2.887 |

The no-preset row also changes self-detection interval from -1 to 0 and
disables caches; it is **not** an iteration-only comparison. Position range
includes slow drift and is not pure vibration amplitude. These are single
independent histories, not a statistical benchmark or same-state accuracy
proof. Existing final-state and finite-value checks passed, but do not
certify intersection freedom or absence of jitter.

The VT synthetic regression failed in all eight CPU/CUDA, soft/full,
stationary/translating cases before the relative-increment fix and passed
afterward. At radius 0.002 m and stiffness 300000 N/m, it exposed a maximum
spurious dissipative-force component of about 1.48 N. That is a synthetic
test measurement, not the actual shirt's measured spurious force.

For mu=0.4, normal load=12 N and smoothing distance=0.001 m, the existing
smooth friction law's zero-slip tangent should be diag(9600, 9600, 0) N/m.
HEAD returns zero. Four CPU/CUDA and soft/full finite-difference subcases
failed before the tangent fix and passed afterward. Neither isolated fix
demonstrated a terminal jitter improvement in the full scene; no promotion.

#### Same-state terminal schedule diagnostic on unmodified HEAD

After saving and removing the formula changes, a diagnostic ran HEAD for
1200 frames, saved 440 reachable Newton objects and 519 Warp arrays, and
restored the same particle positions/velocities exactly before each case.
Each continuation ran 180 frames; the metrics use the final 90 frames,
except net displacement, which spans the full continuation. Opaque BVH
handles are not serialized; normal initialization refits/rebuilds them.
An A/A repetition measures remaining reproducibility rather than assuming
the entire contact history is deterministic.

| Schedule | RMS speed (mm/s) | Frame second-difference RMS (um) | Net displacement RMS (mm) | ms/frame including readbacks |
| --- | ---: | ---: | ---: | ---: |
| HEAD default | 2.116 | 24.070 | 0.745 | 33.679 |
| Disable Chebyshev only | 1.958 | 22.464 | 1.188 | 32.716 |
| Ordinary GS8 | 1.058 | 13.297 | 0.951 | 56.421 |
| Ordinary GS20 | 3.172 | 18.212 | 11.415 | 102.049 |
| Ordinary GS30 | 8.823 | 27.595 | 15.021 | 142.573 |
| HEAD default A/A repeat | 2.178 | 29.525 | 0.713 | 33.802 |

The default A/A first-frame positions match exactly; after 180 frames,
their RMS separation is 0.163 mm. First-frame position differences against
GS30 are 1.173 mm (default), 1.090 mm (GS8), and 0.421 mm (GS20). This is
a complete-frame, common-initial-state diagnostic, not a frozen-substep
Newton-residual measurement. None proves near-30-sweep convergence.

Disabling Chebyshev alone did not remove jitter. GS8 reduced the measured
high-frequency motion, but at a substantial cost. GS20/30 also produced
large slow motion; neither can be called a stationary ground truth. The
data support investigating simultaneous batch updates, not a unique cause
or a production fix. Timings including per-frame readback are diagnostic.

A second common-state batch was stopped at the user's cleanup request.
Completed cases from that batch were:

| Schedule | RMS speed (mm/s) | Second-difference RMS (um) | Net displacement RMS (mm) |
| --- | ---: | ---: | ---: |
| Default | 1.478 | 15.261 | 0.590 |
| Jacobi relaxation 0.5 | 1.542 | 15.178 | 3.453 |
| Jacobi relaxation 0.25 | 2.581 | 32.233 | 5.459 |
| Two final ordinary sweeps, total budget still eight | 1.181 | 13.110 | 0.601 |
| Four final ordinary sweeps, total budget still eight | 1.347 | 20.254 | 0.519 |

The GS30 reference and final A/A repeat in this second batch did not
complete; no accuracy or repeatability acceptance is inferred. Its warmup
history differs from the first batch, so rows cannot be cross-compared as
one shared snapshot. Two final ordinary sweeps are at most a partial
diagnostic lead, not a validated improvement. No default change is retained.

#### Cleanup / recovery

All unaccepted code is archived in local Git stashes before removal from
the main modification area. The HEAD-isolated helper scripts, tests,
formula patch and notes are separately archived. Raw measurements under
the ignored `newton/tests/outputs/mjvbd_global_settling/` directory are left
untouched; they are not source changes. No assets, committed optimization,
remote branch, or pre-existing worktree is deleted. No commit or push is
performed by this cleanup. The sole intended main-worktree change is this
optimization log, unstaged.

Recovery archives (local stash commits; indices may change as new stashes
are created):

- Main staged/unstaged/untracked research:
  `4e3216e9dff3f65a79b43755185ca90f23efb7ee`, message
  `Archive unaccepted MJVBD research before log-only cleanup 2026-09-09`.
- HEAD-isolated formula patch, regression tests and terminal diagnostics:
  `c9962ab7fb3f992ba396309de0713be7af1ea337`, message
  `Archive HEAD-isolated settling diagnostics 2026-09-09`.

The main cleanup covers 76 code/change paths, excluding this log. Recovery
should be selective and deliberate; applying the entire archive would
reintroduce the experimental preset and rejected branches removed here.

### 2026-09-09: T-shirt default versus ordinary GS20 energy diagnostics

Archived the figures, raw CSV measurements, configuration, formula caveats,
and reproduction commands in
[the local energy report](../../../../docs/lab/mjvbd_tshirt_energy_2026-09-09/README.md).
Baseline: `cfe948b9`; no production solver or demo changes.

Both runs cover 1200 frames (20 seconds), with identical scene parameters
and 10 substeps per frame. Default is seven batched sweeps plus one ordinary
sweep; the comparison uses 20 ordinary GS sweeps without the acceleration
preset. Sampled mean kinetic energy over 15–20 seconds is 1.77976e-8 J
(default) versus 6.10862e-9 J (GS20), approximately 2.91 times higher.

A shared initialized substep after default frame 1200 starts at 21.59356 J
in the fixed diagnostic objective, ending at 18.32741 J after default eight
sweeps versus 18.14265 J after ordinary twenty. Both diagnostic curves
decrease at each sampled sweep. Frozen-contact friction/damping terms make
this a surrogate, not the changing-contact solver's exact global objective.
Full-trajectory final energies cannot establish same-state accuracy, and
these measurements do not prove a unique jitter cause or near-30-sweep
convergence. Two energy formula tests and both demo final checks pass.
No new optimization, default change, or timing acceptance is claimed.
