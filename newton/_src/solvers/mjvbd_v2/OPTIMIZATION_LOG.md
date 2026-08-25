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
`example_vbd_mjvbd_v2_right_hand_armadillo_into_gear_crusher` with a null
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
