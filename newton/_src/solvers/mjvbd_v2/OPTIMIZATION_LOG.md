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
| 2026-08-21 | Cache a particle-color membership mask per rigid-soft contact | Complete `vbd/` particle-side rigid-soft solve | **Retained** | Frozen contact solve is 5.39% faster including mask construction; full handoff graph is 6.99% faster in a consecutive run |
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

Re-profile the complete `vbd/` path after the retained color-mask change. The
next candidate should target the rigid-side per-body contact accumulation while
preserving contact-level GPU parallelism. It must remain entirely under
`newton/_src/solvers/mjvbd_v2/`; shared collision and simulation modules are
inputs, not optimization targets. The ordinary supermarket bag has a different
profile dominated by self-contact force accumulation, planar truncation, and
self-contact collision detection, so handoff-scene gains must not be applied to
it without a separate measurement.
