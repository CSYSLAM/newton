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
| 2026-08-21 | Reduce dense body-particle contacts in parallel | Complete `vbd/` AVBD path | **Retained, gated** | A 200-frame supermarket-bag CUDA Graph run fell from 37.670750 to 28.246249 ms/frame (25.02%); a 1,024-contact isolated Graph was 25.23x faster |
| 2026-08-21 | Prune zero-inverse-mass and kinematic rows from rigid color groups | Complete `vbd/` AVBD path | **Rejected** | Both end-to-end dense-contact variants included pruning, so no gain was isolated; the host-cached launch topology would require Graph recapture after runtime mass or flag changes |
| 2026-08-21 | Add another AABB pass to sparse point contacts | Dynamic and kinematic `soft` backends | **Rejected** | Rejected before implementation: the dynamic fold scans 566,368 candidates in 0.026726 ms per collision Graph; even ten calls are below 0.1% of the measured frame |
| 2026-08-21 | Move shape-major full-surface AABB rejection into a private V2 collision pipeline | All `full` contact backends with full-surface contact on CUDA | **Retained** | Frozen frame-121 collision time fell from 3.825585 to 1.643479 ms (57.04% lower, 2.33x throughput) with the same 9,838 contact keys |
| 2026-08-21 | Fuse coupled transfers and pipeline MuJoCo/VBD substeps on two CUDA streams | Dynamic coupled backend | **Rejected** | The real dynamic fold was 1.32% slower with transfer fusion and 0.90% slower with the two-stream wavefront; a 1.66% light-scene gain did not transfer to the representative workload |
| 2026-08-21 | Selectively port PR 3995 soft-contact traversal and capacity sizing | Both private VBD implementations and sparse contact helpers | **Retained, batch-gated** | A 1,024-world CUDA Graph benchmark is 4.81% faster; world-compatible preallocation removes a 1,024x capacity overestimate in that setup |
| 2026-08-21 | Cache a particle-color membership mask per rigid-soft contact | Complete `vbd/` particle-side rigid-soft solve | **Retained** | Frozen contact solve is 5.39% faster including mask construction; full handoff graph is 6.99% faster in a consecutive run |
| 2026-08-24 | Compact AABB-active full-surface candidates before SDF optimization | CUDA `full` contact backends with full-surface contact | **Retained, gated** | Handoff collision Graph median is 10.58% faster and full frame median is 6.58% faster; pneumatic and 100%-active collision Graphs are 60.54% and 47.21% faster |
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
`example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff0000.py` into an
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
`example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house.py` used an
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
