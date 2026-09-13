# Reduced-substep experiments

Target: at least 20 FPS without changing the original physical acceptance,
grain count, friction grasp, deformable cup, or contact material.
No configuration in this log has yet met that target.

All probes use `cache_probe.py` and remain separate from demo defaults.
Raw logs are in `E:/csy_work/CG/Engine/newton_cleanup_archive/popcorn-20260911-200805`.

| Probe log | Budget | Result |
| --- | --- | --- |
| `substep4-full48.log` | 4 substeps, 24 sweeps, cluster 400 | Cup slip 30.7 mm at 19.217 s |
| `substep2-sweep48-full48.log` | 2 substeps, 48 sweeps | Shaft slip 12.8 mm at 0.167 s |
| `substep4-global32-full48.log` | 4 substeps, 24 sweeps, cluster 32 | Shaft slip 25.5 mm at 11.533 s |
| `substep4-global32-history-full48.log` | Same, existing rigid contact history enabled | Cup slip 32.4 mm at 20.133 s |
| `substep4-global2-sweep8-full48.log` | 4 substeps, 8 sweeps, cluster 2, history | Shaft grasp failed |
| `heterogeneous6-r24-p8-global2-full48.log` | 6 substeps, 24 rigid / 8 particle sweeps, cluster 2, history | Cup slip 31.4 mm at 20.433 s |
| `projected4-full48.log` | 4 substeps, 24 sweeps, cluster 32, history, projected surface contacts | Grain speed 32.7 m/s; rejected |
| `projected8-r12-full48.log` | 8 substeps, 12 sweeps, cluster 400, projected surface contacts | Cup slip 32.6 mm at 19.783 s |
| `coarse-audit4-full48.log` | 4 substeps, 24 sweeps, cluster 400 | Cup slip 30.5 mm at 19.617 s |
| `global4-r24-p4-full48.log` | Singleton particle clusters, 4 substeps, 24 rigid / 4 particle sweeps, PCG 64, projected contacts | Cup slip 30.1 mm at 19.833 s |
| `global4-r24-p8-pcg256-full48.log` | Singleton particle clusters, 4 substeps, 24 rigid / 8 particle sweeps, PCG 256 | Shaft slip 43.8 mm at 8.900 s |
| `dense-single8-full48.log` | Single-block reduction, 8 substeps, 24 sweeps | Cup slip 32.4 mm at 24.617 s |
| `cup24x6-sub4-full48.log` | 24-segment / 6-ring shell, unchanged material and areal density, 4 substeps, 24 sweeps | Cup slip 31.7 mm at 20.250 s |

The heterogeneous schedule preserves alternating coupling and reaction forces,
but did not preserve the grasp at this budget. It is not a production option.
The projection probe adds the missing barycentric edge/face contact cross blocks
to the Galerkin matrix, including the same-cluster diagonal correction. It does
not project self-contact or rigid degrees of freedom. It remains experimental.

## Fixed soft-contact penalty update

For `rigid_linear_beta == 0`, initialization already puts each soft contact at
its material penalty ceiling. Subsequent penalty-ramp launches do not change
that value. Skip those launches in both rigid-solve and skipped-rigid branches.
Positive-beta behavior is unchanged. This does not skip force evaluation.

`test_fixed_soft_penalty_skips_redundant_updates` failed before the change
(3 launches instead of 0) and passed after it, for static and dynamic shapes.
The four-test contact-penalty module passed. Full-scene and broader validation
of this follow-up change is still pending; do not claim 20 FPS from this change.

## Coarse and ownership audit

The four-substep coarse audit attempted 28,248 corrections and accepted 28,244.
Only 66,439 individual particle corrections reached the radius clamp across
all those solves. Wholesale coarse rejection is not the observed failure.

Current VBD ownership already excludes the robot from movable body groups:
the three groups have (total, movable) counts `(1, 1), (80, 80), (80, 80)`.
Skipping wholly immovable color groups cannot accelerate this scene. The
follow-up run was cancelled after this was established; no timing is claimed.

## Single-block dense contact reduction

`dense_single_probe.py` evaluates all active contacts for a dense body with
one block, striding over its live prefix, then reduces directly into the
body force, torque and Hessian. It replaces the capacity-sized chunk grid
and second reduction launch. Contact records, materials and solve budgets
remain unchanged. Floating-point summation order changes.

`dense-single-full48.log` passed original full-scene acceptance at 16 x 24:
17 grains lifted, 14 retained (82.35%), five-finger coverage 100%, cup lift
109.92 mm, RMS distortion 2.00 mm, plastic hinge offset 0.1171 rad.
2850 measured frames averaged **149.162 ms/frame** (6.70 FPS, no rendering).
This result includes the fixed-penalty no-op elimination. It is not an
isolated attribution of the entire improvement to the dense kernel, and is
not the requested 20 FPS. The new dense kernel remains diagnostic pending
numerical regression and integration.

Follow-up: the reduction is now integrated for per-body capacities up to 4096;
larger capacities keep parallel chunk reduction. Twenty selected regression
tests passed (`dense-integrated-regression.log`), including comparison with
independent rigid reactions and two CUDA Graph replays. The exact integrated
full-scene run remains pending. Existing staged work was not changed.

The self-contact conditional graph probe (`self-gate-short.log`) took
158.901 ms/frame in the initial interval, so it is not an accepted speedup.
It scans all VT/EE counters after detection, retains detection and the DAT
displacement bound, and conditionally skips only empty-contact work.

## Frozen-color rigid fusion prototype

`rigid_fusion_probe.py` combines rigid/particle contact evaluation,
rigid/rigid evaluation, and the free-body 6x6 pose solve into one block per
body. A copied color-start pose array prevents same-color read/write races.
Dual updates remain outside the fused kernel, at their original cadence.
Only solve groups without constrained joints are supported by this prototype;
non-free joints affecting a solved body are rejected, not silently ignored.

The mixed-contact same-state oracle passed (`rigid-fusion-numerical3.log`).
Full-scene quality and performance are not yet established. An initial
general-joint prototype was cancelled during excessive compilation; the
current probe has backward compilation disabled and only free-body solves.

Follow-up full run `rigid-fusion-full48b.log` passed at unchanged 16 x 24:
136.133 ms/frame without rendering, 19 lifted, 15 retained (78.95%), all five
fingers in contact, cup RMS 2.035 mm and plastic hinge offset 0.12819 rad.
This remains diagnostic, not a general articulated-body implementation.

## Particle fusion and rigid Ritz experiments

Gathering all point/edge/face contact corners directly in the surface tile
passes same-state numerical checks. The test contact buffer was increased
from 256 to 1024 because 429 contacts overflowed the original test capacity;
comparison tolerances were not changed. The complete combined fusion run
passed, but 139.895 ms/frame is slower than rigid fusion alone. It is not
retained as a demonstrated performance improvement. At 8 substeps and 12
sweeps, the cup slipped at 21.017 s.

Direct reuse of the existing tet rigid basis for cloth was rejected: its
assembly omits cloth membrane/bending terms. The replacement Ritz prototype
projects the fully assembled fine surface operator into cluster translation
and rotation modes. It includes edge/face contact cross blocks when requested.
A dense double-precision projection/solve oracle, with and without contact
cross terms, passed. The diagnostic explicitly checks the projected residual
and retains the original DAT and coarse acceptance conditions.

Eight substeps / 12 sweeps failed tool slip at 15.233 s without contact history;
with history it failed cup slip at 20.233 s. Neither is an accepted setting.
Sixteen substeps / 12 sweeps is under test. No 20 FPS claim is supported.

Follow-up: that run failed cup slip at 20.767 s; its partial initial timing
was about 201 ms/frame. The projection was subsequently parallelized across
matrix columns (instead of only three cluster blocks). The double-precision
oracle still passes; no complete-scene benefit is claimed from that change.

The cached-elasticity + rigid fusion 16 x 8 trial failed cup slip at 22.617 s
(initial partial timing 93.837 ms/frame). Full-space PCG with 8 substeps,
12 sweeps and 32 PCG iterations failed cup slip at 20.633 s (initial partial
timing 78.739 ms/frame). Both remain rejected, not new demo defaults.

## Scheduling and two-level preconditioning follow-up

Five rigid work-array clears per iteration can be omitted in the diagnostic
fused overwrite path. The mixed-contact numerical oracle passes, with exactly
20 omitted clears over four sweeps. This is scoped to that phase; unrelated
array resets are untouched.

The device-side periodic sweep-loop probe preserves all sweeps, explicit first
and last blocks, first-sweep detection, and periodic coarse checkpoints. It
rejects unsupported non-periodic features. A 16-sweep cloth/contact test with
non-unit middle relaxation passes three graph replays against an unrolled
graph. Combined fusion/reset-pruning/loop short timing is 123.566 ms/frame,
versus 126.308 ms/frame for fusion alone. This is modest, not a major speedup.

`two_level_probe.py` keeps the complete fine solve and uses the rigid Ritz
space only in the additive SPD preconditioner D^-1 + P (P^T A P)^-1 P^T.
It does not discard internal deformation modes. Double-precision random SPD
systems, with and without edge/face contact cross terms, pass solution and
true residual checks, including two graph replays. Full-scene validation of
8 substeps / 12 sweeps / 8 preconditioned PCG steps is in progress.

Follow-up: it ran the complete 48 seconds without a slip assertion, at
73.731 ms/frame, but lifted/delivered zero grains and failed final acceptance.
It is not a successful faster demo. Increasing rigid continuation beta by 2
while retaining material ceilings failed cup slip at 21.333 s; the last
reported relative linear residual was about 0.64. Keeping 24 rigid sweeps but
only eight particle sweeps/four coarse corrections also failed cup slip.

## Unified translation experiment (not production)

`coupled_translation_probe.py` augments the complete fine cloth operator with
translation unknowns for the free dynamic bodies. Barycentric contact rows
include a negative body weight, so the linear solve contains particle/body
cross blocks and equal/opposite reactions, plus rigid/rigid contact blocks.
Rigid orientation still uses the ordinary local six-DOF solve; this prototype
is not a full articulated or full six-DOF global Newton implementation.

The diagnostic uses a common Newton step scale bounded by the existing fine
particle correction radius and a 1 mm body-translation trust bound. It applies
body corrections only when the original coarse commit accepts the step. DAT
still processes the particle update. No attachment, velocity overwrite,
contact-capacity reduction or acceptance relaxation was added. Unsupported
hard-contact and constrained-body configurations are rejected.

Point/edge/face augmented contact matrices match the analytic k*w*w^T system,
including the negative body weight. The first short interval was 74.068
ms/frame with 8 substeps, 12 sweeps and 16 PCG iterations. Full-scene physical
acceptance is in progress; short timing alone is not an accepted result.

`table_guard_probe.py` performs conservative mesh/triangle AABB filtering on
the GPU, using double-precision coordinates and outward rounding. Candidate
triangles still receive the original CPU separating-axis test. Random rotated
mesh tests and graph replay preserve every CPU-positive triangle. It does not
disable the robot/table clearance guard. The full-scene experiment includes
this diagnostic path; its isolated timing benefit has not yet been measured.

### September 13: coupled full-run and rendering results

The augmented surface/body translation operator now also passes an independent
NumPy dense-solve oracle on a cloth plus two dynamic spheres, including two
CUDA Graph replays. This validates its assembled linear system, not the
full nonlinear trajectory or a full rotational/articulated Newton method.

Full 48-second trials, with unchanged trajectories, geometry and acceptance:

| Configuration | Wall ms/frame | Lifted / retained | Result |
| --- | ---: | ---: | --- |
| 8 substeps, 12 sweeps, 16 PCG, headless | 84.560 | 28 / 14 | Fail: 50% retention |
| 6 substeps, 8 sweeps, 16 PCG, 1080p rendering | 86.244 | 24 / 11 | Fail: 45.83% retention |
| 4 substeps, 8 sweeps, 16 PCG, cached props, 1080p | 55.318 | 26 / 12 | Fail: 46.15% retention |

These trials hold the deformable cup and complete the trajectory, but do not
meet the existing 70% payload-retention requirement. None is an accepted new
default, and none proves 20 FPS with physical acceptance. Increasing only the
rigid budget to 24 sweeps at 4 substeps (particle cadence 3, checkpoint 6)
instead lost the cup at 17.600 seconds; that configuration is rejected.

`prop_render_probe.py` retains persistent static prop instances and updates only
transforms for dynamic prop meshes. Colors, opacity and geometry are unchanged.
The same-state 1920x1080 framebuffer comparison is pixel-identical. A short
6-substep/8-sweep trial with this cache measured 62.255 ms/frame. This is not a
complete physical acceptance result. Further appearance-upload change detection
is diagnostic-only and must preserve in-place color/opacity edits.

Appearance change detection passes the in-place color/opacity and model
invalidation test. Same-state framebuffer equality also passes. A 4-substep
repeat nevertheless generated a 23.5 m/s grain, so the 4-substep configuration
is not robust enough. The 6-substep/8-sweep/8-PCG trial measured 52.512 ms/frame
but failed physical acceptance (19/35 retention and final cup lift below the
original threshold). Enabling surface cache for that budget also failed with
a 24.2 m/s grain; surface cache remains off for these experiments.

Detailed payload tracing found a trajectory issue separate from solver cost:
35 grains in the scoop fell to 23 during lift/retract, before transport or
pouring; 19 of those reached the cup without recorded cup exits. A separately
labelled motion experiment advances the existing carry pitch transition from
11.25--12.5 s to 10.5--11.25 s. It only changes robot targets, not object states,
materials, forces or acceptance. This motion change is not claimed as solver
acceleration.

**Full physical pass:** 8 substeps, 6 sweeps, 8 PCG iterations, checkpoint 2,
coupled translation, original surface elasticity, DAT cache, rigid fusion,
contact history, GPU table guard, cached prop/appearance uploads, batched shape
transforms, and the explicitly labelled earlier carry pitch. The 48-second
1920x1080 rendered run measured **59.644 ms/frame (16.77 FPS)**, with 28 lifted,
25 retained (89.29%), five-finger contact 100%, cup RMS deformation 1.749 mm,
plastic hinge rotation 0.331 rad, rim radius 30.586 mm, and shaft slip 0.807 mm.
All existing final guards passed. **20 FPS has not yet been achieved.**

The batched shape transform kernel matches the original kernel exactly for
parented/static shapes, world offsets, layer transforms and changed body poses.
Its full-scene same-state framebuffer comparison also passes. Model/shape-batch
replacement is explicitly rejected by this diagnostic adapter rather than
silently using stale data. Preconditioner Cholesky factorization is now reused
within each linear PCG solve (the matrix is fixed there); the augmented dense
oracle and two CUDA Graph replays still pass.

**Batched-transform diagnostic removed:** inspection of the concrete
`ViewerGL.log_state` override (not only the base class) shows that the CUDA
backend already performs packed transform updates. The extra experimental
kernel is redundant on this backend. Its passing arithmetic test did not
establish a performance benefit. The diagnostic adapter/flag/test were removed;
the existing production packed path remains unchanged.

The CPU profile's 32 ms inside appearance-change readback is largely waiting
for the asynchronously launched physics graph, not 32 ms of appearance
processing. Do not report that wait as removable rendering work. Actual
unnecessary rendering includes updates of hidden debug triangle meshes, while
the custom four-material cup remains visible and deforming. A diagnostic now
skips repeated hidden updates, refreshing on visibility/model changes.

The 8-substep/6-sweep/checkpoint-2 run with equivalent cross-product Ritz
restriction/prolongation and cached Cholesky, hidden-triangle refresh gating
and mesh CUDA interop passed again: **57.879 ms/frame**, 20/22 retained
(90.91%), five-finger contact 100%. Four local sweeps were rejected: cup slip
at 26.850 seconds despite approaching the timing target.

**Conditional IK experiment removed:** a small accepted joint update can mean
an LM proposal was rejected rather than converged. A single checkpoint and
then two consecutive checkpoints both failed the shadow 32-iteration joint
comparison (0.000211 and 0.000481 rad). Adding the undamped gradient and
proposed-step checks preserved the reference in audited frames but ran all
32 iterations, offering no demonstrated gain. The adapter, flags and predicate
test were removed; retain the original 32-iteration IK and all wrist guards.

With 8 substeps, 6 sweeps and checkpoint **3**, the 48-second rendered run reached
**49.101 ms/frame (20.37 FPS)**, but failed payload acceptance: 11/18 retained,
four still in the scoop. This is a timing result, **not completed acceptance**.
An explicitly labelled 40-degree pour-angle trial (original is 35 degrees)
is testing physical draining; it changes only the wrist trajectory, not grain
state, friction, cup geometry, contact stiffness or acceptance thresholds.

`face_fixedpoint_probe.py` ends Frank-Wolfe search only when every barycentric
component is bitwise unchanged by the original update. For fixed input
geometry every remaining iteration then repeats exactly; this is not a
tolerance-based early exit. Analytic box/sphere barycentrics, query positions,
distance and gradient match the 24-iteration reference exactly, including two
CUDA Graph replays. A mesh-SDF check is also required before integration.

## Continued 20 FPS acceptance work

The actual popcorn convex mesh SDF, including mirrored/nonuniform scale, now
passes the fixed-point search comparison (`face-fixedpoint-mesh-test.log`).
The 40-degree pour trial passed the complete 48-second rendered sequence:
**52.524 ms/frame (19.04 FPS)**, 21/25 retained, five-finger contact 100%,
cup RMS deformation 2.092 mm and plastic hinge rotation 0.223 rad.
Reducing PCG from eight to six iterations was rejected: cup slip at 23.717 s.
Keep the eight-iteration linear budget.

Default versus mesh CUDA interop did not establish an improvement. Default
interop passed at **52.603 ms/frame**, 22/24 retained (91.67%), five-finger
contact 100%, RMS deformation 1.850 mm, plastic hinge rotation 0.175 rad.
These are still below the 20 FPS acceptance target.

The same-state framebuffer audit sometimes differs in one 8-bit channel by
one level out of 6,220,800 channels. The diagnostic reports changed-channel
counts and renders an uncached repeat on a mismatch. It tolerates at most one
level in 0.001% of channels, not a broad image-quality tolerance. Physical
acceptance thresholds remain unchanged.

The first fused PCG update prototype failed full-scene grasp acceptance.
A new 1002-row, three-cluster test reproduced a cross-thread shared-tile
construction error that the small one-cluster dense test missed. The explicit
shared writes were replaced by Warp's collective tile construction/slice.
The new regression fails before this change and passes after it. Full-scene
validation is required again; this prototype has not changed production defaults.

The corrected fused update passed the full sequence at **52.247 ms/frame**,
20/24 retained. Cross-product Ritz restriction plus full CUDA interop passed
at **52.485 ms/frame**, 22/25 retained; no meaningful interop win was shown.
Using a 24-column padded factor for the actual 18-column Ritz space passed
three independent update/operator tests and full acceptance at **51.890
ms/frame**, 21/25 retained. No basis columns were removed.

Bending-gradient reuse matched all random/deformed/boundary/collapsed-slot
cross blocks, but its full-scene trial slipped at 24.133 s. Do not enable it.
One global assembly with 16 PCG iterations instead of two assemblies with
eight iterations failed five-finger lift acceptance. Six substeps with eight
local sweeps also failed lift acceptance. Seven substeps were not simulated:
the demo explicitly requires even substeps for its single captured graph.

Batching the four cup material downloads without changing normal computation
measured **51.818 ms/frame**. It delivered 23/23 grains but failed the final
current cup-lift guard, so this run is not successful acceptance. The report's
maximum historical cup lift is not the final lift. Do not weaken that guard.

A further rendering-only diagnostic gathers each material's normals in one
kernel and uses one shared CUDA/OpenGL vertex buffer while retaining four
separate material meshes/VAOs. Original geometry, smoothing domains, colors,
backface selection, resolution and physics are unchanged. Its initial
same-state framebuffer comparison is exact; full acceptance is pending.

The first gathered-normal/shared-VBO run retained 25/30 but failed final cup
height (about 51.4 ms/frame). Reducing only dummy Ritz padding and reduction
storage from 32/24 to the actual 18 columns passed four matrix/render-normal
tests and full acceptance at **50.903 ms/frame**, 17/21 retained. A 128-thread
PCG block passed but was slower (**51.056 ms/frame**, 18/22 retained), so retain
256 threads. Neither result reaches 20 FPS.

Read-only body/particle snapshots within the controller phase remove redundant
downloads, with invalidation before eager physics and before post-graph
measurements. The phase/slice regression passes. One full trial nevertheless
slipped at 37.650 s; performance or a prior passing trajectory is not sufficient
evidence for acceptance.

Moving the same two global corrections to sweeps **2 and 4**, followed by two
ordinary local sweeps, passed the full sequence at **50.600 ms/frame**, 21/25
retained, five-finger contact 100%, RMS deformation 1.824 mm, plastic hinge
rotation 0.239 rad. This is normal post-smoothing, not extra damping or a force
on the cup. All physical budgets and acceptance checks are unchanged.

A 64-thread compact surface-contact block retains the exact global worker
count (four blocks/SM instead of two 128-thread blocks/SM), candidate set and
grid-stride traversal. Shared/private contact and temporal-cache regressions
both pass. Full-scene speed and stability are being measured.

## 20 FPS acceptance and integration — 2026-09-13

The 64-thread surface block passed at 50.765 ms/frame, but did not improve
performance. Retain the original 128-thread block and unchanged worker count.

Masked IK compaction eliminates exactly zero Jacobian columns (40 to 14 DOFs),
not iterations. Original LM damping, all 32 iterations, limits, wrist targets
and elbow objectives remain. Dense NumPy and original 40-DOF comparisons pass.
Single-robot serial objective scheduling removes cross-stream events between
tiny kernels. Shadow comparisons every 120 frames gave exactly zero joint
difference against the original 32-iteration graph through 46 seconds.

The six-sweep version reached 47.416 ms/frame but lost the cup late in the
sequence. The seven-sweep version also slipped at 31.55 seconds. These are
**rejected**, despite their speed. Coarse rejection fallback did not help:
all 46,080 coarse solves were accepted, with zero clamped particles. Keep
ordinary post-smoothing instead of treating coarse residual acceptance as
proof that contact and shell equilibrium are settled.

Accepted budget: **8 substeps, 8 local sweeps, global corrections after
sweeps 2 and 4, 8 PCG iterations per correction, then 4 local post-smoothing
sweeps**. Full particle unknowns and free-body translation contact cross
blocks are solved together; body rotations retain their local 6-DOF solves.
The 18-column translation/rotation Ritz space is a preconditioner, not a
replacement for the full fine operator. Padding elimination removes no modes.

Every result below includes 1920×1080 OpenGL rendering, 160 convex-mesh/SDF
dynamic grains, unchanged 841-particle elastoplastic cup, 30 warmup frames and
2,850 timed frames (48 simulated seconds total). No reduced geometry,
resolution, attachments, artificial particle damping or relaxed physical
acceptance thresholds were used.

| Path | ms/frame | FPS | Lifted / retained | Full acceptance |
| --- | ---: | ---: | ---: | --- |
| Eight-sweep experimental repeat 1 | 49.645 | 20.14 | 30 / 23 | Pass |
| Eight-sweep experimental repeat 2 | 49.566 | 20.17 | 25 / 19 | Pass |
| Owned native solver, rendering diagnostics | 49.803 | 20.08 | 25 / 21 | Pass |
| Ordinary demo defaults, no solver/render monkeypatch | 49.640 | 20.15 | 26 / 25 | Pass |

The default run retained five-finger contact for 100% of the measured grasp,
kept the current cup lift about 88 mm at 40.5 s, rim radius 30.37 mm, shell RMS
deformation 1.91 mm and plastic hinge rotation 0.188 rad. Contact-order
nondeterminism still changes individual grain paths; retained counts are
reported per run, not claimed deterministic.

Motion changes are explicitly separate from solver performance: carry pitch
starts at 10.5 s and finishes at 11.25 s (formerly 11.25–12.5 s), and pour
pitch is 40 rather than 35 degrees. This affects only robot wrist targets and
helps keep/drain a spoonful. Cup/grain forces and material parameters were not
altered. These motion changes are not counted as a solver optimization.

Library integration lives in `coupled_free_body`, owned per solver and gated
by experimental `particle_enable_coupled_translation=False` by default.
It currently supports CUDA, unpinned cloth and finite-mass free solved bodies
with soft contacts. It rejects unsupported tet, pneumatic, external-rigid,
hard-contact and articulation cases rather than silently applying a partial
operator. Other demos retain their solver path. This is not a claim that this
particular preconditioner supports every Newton scene.

Rendering now uses explicit immutable prop instances, per-material gathered
normals and a shared vertex upload. Exact appearance dirty detection and
hidden-triangle caching are opt-in for this demo. Controller snapshots are
read-only and invalidated before every physical advance. The GPU table guard
is a conservative candidate filter; the original triangle/box SAT still
decides whether a pose is rejected.

79 integrated contact, dense-operator, multi-width PCG, masked IK, material
normal, paper shell and asset tests passed (`integrated-regressions.log`).
The full-size PCG test covers 1,002 rows and independent 18/6/18-column kernel
specializations, including the collective tile regression that caught the
earlier failed fused implementation.

Reproduce the unmodified default with:

```powershell
uv run --no-sync python -m newton.examples mjvbd_v2_popcorn
uv run --no-sync python docs/lab/mjvbd_popcorn_performance_2026-09-12/validate_default.py
```

FPS is end-to-end average after initialization/JIT, not a promise that every
frame or every machine exceeds 20 FPS. The margin above 20 is small; 30 FPS
has not been achieved. Nothing has been staged or committed by this work.

### Repeat-stability qualification

The second ordinary-default run slipped at 29.0 s. The first successful
20.15 FPS run does **not** establish repeat stability. A 9-local/7-PCG trial
also slipped at 24.233 s and is rejected. Neither the hand controller nor
material/friction settings were changed to hide this failure.

Reducing the common global step bound from 0.5 to **0.25 particle radii**
(unchanged operator, PCG count, DAT and local sweeps) passed at **49.489
ms/frame, 20.21 FPS**, including three saved screenshots. It retained 22/27
grains, five-finger contact 100%, plastic hinge rotation 0.216 rad, RMS shell
deformation 2.264 mm, final rim radius 29.716 mm. Current cup lift was about
80.5 mm at 40.5 s, close to the unchanged 80 mm acceptance threshold; repeat
validation is still required. Same-state cached/uncached image difference
was exactly zero across all channels. This step-bound trial is not yet the
default until repeated successfully.

The repeat passed at **49.762 ms/frame (20.10 FPS)**, retained 22/25 grains,
five-finger contact 100%, current cup lift 85.34 mm at 40.5 s, RMS deformation
2.009 mm and plastic hinge rotation 0.191 rad. The existing bounded IK recovery
paused the path clock for 0.083 s; physics continued, and no rejected wrist
pose was executed. The logged maximum IK error includes rejected attempts.
The same-state image audit again had **zero changed channels**. The 0.25-radius
bound is now the ordinary demo default; the 0.5-radius and 9/7 trials are not.

An additional **65 tests passed**, covering the integrated operator and
independent kernel specializations, all existing IK modes (CPU/CUDA), viewer
geometry/broadcasting, appearance-cache invalidation and exact Frank-Wolfe
fixed-point termination. The latter compares against a frozen pre-change
24-iteration implementation, not against the newly optimized function itself.
This count overlaps the earlier 79-test batch and must not be reported as
144 distinct tests. Raw logs: `native-quarter-radius.log`,
`native-quarter-repeat.log`, `integrated-regressions2.log` in the archive above.
