# 28 FPS target: ongoing experiments, 2026-09-13

User requires persistence until 28 FPS, without demo tricks. Measurement is
the unchanged default popcorn scene including 1920x1080 rendering, 30 warmup
frames and 2850 timed frames (48 simulated seconds total). Keep 160 grains,
8 substeps, 8 local sweeps, 8 coupled PCG iterations, geometry, material,
grasp feedback, and the existing acceptance thresholds unchanged.

No experiment below is currently enabled in production. No commit requested.

## Latest active direction: speculative, path-identical golden search

Instead of accepting a different contact witness, eight lanes cooperate on
one face query. Given the current golden-section bracket and values, the next
branch is known. Seven lanes pre-evaluate the one/two/four possible new sample
locations for the next three iterations. Shuffle broadcasts then replay exactly
the original comparisons and bracket updates, discarding unused samples.
The iteration count, scalar SDF, sample coordinates and comparison rules remain
unchanged. This trades extra parallel reads for fewer dependent texture-fetch
round trips. Contact emission and cache stores are performed only by the group
leader; all eight lanes follow the same accepted search trajectory.

The first implementation exposed a Warp SSA alias pitfall: assigning scratch
brackets directly from live brackets before a dynamic loop allowed loop-carried
assignments to overwrite the live state. Explicit scalar copies fix it.
262,144 tests now reproduce original search parameter, position, distance, and
normal bitwise, across two shapes, uniform/mirrored/nonuniform scales, eight
iteration counts, zero-length and very short segments. Full-pipeline frozen
contact/cache comparisons now pass at frames 240, 900 and 1800: contact
identities, positions, barycentrics and temporal caches are equal; normals
use the existing 2e-7 normalization-rounding tolerance. The full rendered
benchmark is running; no performance result is claimed yet.

Additional completed probes:

- 16-lane rigid reductions: three frozen scene states pass force/Hessian/pose
  comparisons; updated poses are identical in these fixtures. Not a universal
  bitwise reduction guarantee.
- Convex query + launch fusion: 36.98 ms/frame through 20.5 s, but slips at
  37.683 s (30.1 mm). Reject.
- Audited cull, geometry seed using existing cache predicate, IK and rigid
  launch fusion: 46.650 ms/frame; retained 25/31, full acceptance passes.
- Strict 2 um stationary-witness shortcut plus empty-candidate graph branch:
  45.665 ms/frame, retained 29/31, full acceptance passes. Scalar objective
  regression against two original refinements stays below 2 um in 65,536
  random tests. Too few witnesses meet the shortcut for useful acceleration.
- Test geometry seed before historical cache: approximately 42.79 ms/frame
  early, cup slip at 20.717 s (36.5 mm). Reject.

All are experiments, not adopted defaults. The initial empty-contact graph
experiment was ineligible for the scene's interval=0 setting. The later probe
allows non-refresh iterations at interval=0 while retaining each scheduled
collision detection, DAT position publication and displacement cap.

| Experiment | Result | Disposition |
| --- | --- | --- |
| CUDA Graph conditional skip when current self-contact candidate counts are zero | 49.323 ms/frame, 20.274 FPS; 22/28 retained, acceptance passes | Insufficient gain; isolated probe only |
| Persistent single-block two-level PCG, unchanged 8 iterations/preconditioner | Dense-system/unit tests pass; approximately 50.607 ms/frame through 20.5 s; cup slips 31.1 mm at 26.45 s | Reject; slower and full-scene failure |
| Compute-kernel substitution for small device copies | Approximately 49.454 ms/frame through 30.5 s; final lift/clearance acceptance fails | Reject |
| Triangle/expanded-AABB SAT | 50.306 ms/frame, 19.878 FPS; 25/32 retained, acceptance passes | No acceleration |
| Mesh-only specialization of face-contact geometry dispatch | 49.082 ms/frame, 20.374 FPS; 25/29 retained, acceptance passes | Insufficient gain |
| Analytic cubic minimization inside one interpolation cell | 49.877 ms/frame, 20.049 FPS; 23/26 retained, acceptance passes | Too few applicable queries |
| Add analytic grid-exterior segment searches | 47.377 ms/frame, 21.107 FPS; 17/22 retained, acceptance passes | Does not meet target; isolated probe |
| Affine minorants fitted to actual SDF interpolation cells | Pointwise lower-bound tests pass; 52.433 ms/frame; final lift/five-finger-grasp acceptance fails | Reject |
| Projected BFGS triangle search with original FW fallback | 52.876 ms/frame, 18.912 FPS; full scene passes, but random triangle objective test fails by 70.36 micrometers | Reject; slower and insufficient search equivalence |
| Contact-gather / surface tile fusion, retaining DAT publication boundaries | 49.801 ms/frame, 20.080 FPS; 22/27 retained, acceptance passes; local 8-sweep coordinate difference 5.82e-11 m | No meaningful acceleration |
| Packed actual SDF cell corners | 49.489 ms/frame, 20.206 FPS; 26/30 retained, acceptance passes; 32,768 point/edge/face comparisons bitwise equal | No meaningful acceleration; adds 531,555,872 bytes |
| Precomputed immutable rest-distance exclusions in bitsets | Approximately 53.0 ms/frame; final cup-lift acceptance fails | Reject; slower and full-scene failure |
| Exact convex exterior GJK with support-plane certificate and original SDF fallback | 47.963 ms/frame, 20.850 FPS; 24/24 lifted grains retained, acceptance passes | Insufficient gain; isolated probe |

The convex exterior probe changes the proximity algorithm, not the supplied
convex hull geometry: use actual hull/triangle separation instead of the sampled
SDF exterior. Support-plane lower/upper distance agreement must be within 2 um;
otherwise fall back. This is not bitwise equivalence to the sampled SDF model.
Independent box-plane tests cover 2048 exterior cases, mirrored/nonuniform
scaling, normals, barycentric witnesses, and overlap fallback. Accepted/fallback
coverage is measured separately from correctness; conservative rejection of an
accurate GJK point is allowed, accepting an inaccurate point is not.

The next isolated convex-MPR probe also evaluates penetrating convex contacts
using the repository's MPR algorithm. Validate witness location, supporting
plane, depth consistency and barycentrics; keep the SDF fallback when these
checks fail. Independent 4096 signed box/triangle cases pass with distance
error <=2 um and normal error <=2e-3. This is a standard convex contact algorithm,
not fewer physical iterations, altered geometry, attached grains or grasp forces.
Full-scene result pending.

Captured external CUDA events at 18--19 seconds, rather than an uncaptured
kernel sum, measured approximately 41.858 ms simulation, 19.685 ms collision,
4.669 ms self detection, 4.592 ms rigid iterations, 14.080 ms particle iterations
(inclusive of 6.989 ms multilevel). Some intervals overlap; do not sum these
inclusive categories. Instrumented end-to-end time was 51.483 ms/frame.
These event nodes have overhead and the measured wall time is diagnostic,
not an accepted benchmark improvement.

The projected-search experiment exposed a distinction between the original
contact-normal convention and the scalar SDF gradient outside grid bounds.
The former deliberately uses the outward extension direction; substituting
it as an optimization gradient is not mathematically equivalent. A corrected
scalar gradient still did not yield an acceptable performance/accuracy result.
The later geometry-size gate in that probe remains unvalidated, not adopted.

The surface fusion experiment combines barycentric contact gathering and
the existing 16-lane surface elasticity calculation. Position publication
cannot be fused: current coloring permits same-color opposite vertices in
some bending elements. Retain the original DAT publication kernel and its
launch boundary. The probe is limited to current empty self-contact candidate sets,
with an ordinary-path conditional fallback for live self contact, contact
adjacency overflow, or same-color contact dependence. The displacement bound
relative to the last collision-detection state is retained. All physical
parameters, substeps, local iterations, coupled PCG iterations, render settings,
and acceptance tests remain unchanged. No production integration yet.

Next experiment: pack actual texture cell corner values into contiguous GPU
memory, replacing repeated texture fetches with indexed reads. Preserve the
software float32 interpolation, scalar field, quantization de-scaling, normals,
and all original search iteration counts. Validate point, line, and triangle
search outputs before full-scene measurement. No resampling or lower-resolution
collision approximation is intended.

At a frozen state after 1080 frames, baseline collision Graph measurements
were approximately 2.38 ms/substep. Changing face/edge block dimensions among
32, 64, 128, 256 did not produce a substantial gain. There were zero VT/EE
self-contact candidates, 3976 active edge pairs, 2964 active face pairs, and
364 cached contact faces. These isolated measurements are not demo FPS.

## Analytic search details

### Convex geometry prototypes (not adopted)

The original convex collision hulls were queried directly instead of treating
their sampled SDF as the exact geometry. This preserves the hulls but is not
bitwise equivalent to the sampled contact model. Contacts require a 2 micron
support-plane/witness certificate; uncertified queries retain the SDF path.

- Exterior GJK: 47.963 ms/frame, 20.850 FPS; 24/24 grains retained,
  full acceptance passed. Analytic box tests passed.
- MPR plus GJK: 48.513 ms/frame, 20.613 FPS; 19/20 retained,
  full acceptance passed. Signed box-plane tests passed.
- Accelerated support LUT/walk: 48.422 ms/frame, 20.652 FPS;
  21/26 retained, full acceptance passed. Query counters: 100,368,987
  total, 7 accepted MPR, 70,931,875 accepted GJK, 29,437,105 SDF
  fallbacks. Diagnostic counters are included in this timing.

The next experiment separates conservative non-contact rejection from accurate
contact reconstruction. A separating support plane beyond the contact margin
can reject a pair without converging to its closest points. It also caches
static scaled hull centers and tries GJK before MPR. No iteration, particle,
rendering, friction, or acceptance budgets are reduced. It completed at
46.763 ms/frame (21.384 FPS), with 21/23 retained and full acceptance passed.
The independent analytic test covers 16,384 signed box-plane queries, including
mirrored scales and conservative rejection. No true contact was culled in
those cases; contact depth/normal/barycentric checks passed.

Keeping the SDF temporal cache before the convex query did not qualify: the
cup slipped 30.0 mm at 27.950 s. Do not adopt this combination.

An independent two-stage SDF search prototype compacts unresolved queries;
the first version reran the temporal refinement for unresolved pairs. Contacts
and cache arrays were bitwise equal at frames 240, 900 and 1800 after sorting
by shape and particle indices (soft_contact_tids is an inverse lookup, not a
per-contact identity). That version was slower: 74.6--77.4 ms/frame through
20.5 s, so its benchmark was stopped. A revised second stage removes the
redundant failed temporal refinement; tests and performance remain pending.

Analytic IK objective fusion retains all 32 LM iterations, limits, weights,
and damping. An active short 600-frame probe measured 46.728 ms/frame during
the initial approach/grasp only. This is NOT a complete scene or an achieved
speedup claim. Its objective functions share one launch and write disjoint
residual/Jacobian rows. An independent four-problem, mixed position/rotation/
joint-limit test subsequently produced bitwise identical joint coordinates
and residuals after 32 LM iterations.

The installed Nsight Compute 2025.1 cannot initialize its counter library
against the current driver (`LibraryNotLoaded`). No driver/system changes
were made. CUDA-event and end-to-end measurements remain available. A frozen
collision Graph experiment found no register-budget win: unrestricted 2.34 ms,
64 registers 3.63 ms, 96 registers 3.36 ms, 128 registers 2.87 ms (repeat 2).

### Small-simplex GJK finding

At the frozen baseline state at frame 1080, 2942 convex queries yielded 246
fallbacks. All 246 reported positive separation, none reported an MPR hit;
226 even had a positive support-plane lower bound. Thus penetrating-contact
LP/EPA work would not address this bottleneck. A CPU epigraph-LP exploratory
test passed 200 comparisons against HiGHS but is not integrated or pursued.

The GJK prototype inherited a dimensional issue from its simplex helper:
the same absolute `EPSILON = 1e-8` was used for squared edge length, squared
triangle area and tetrahedron determinant. Valid millimeter-scale simplex
triangles can consequently be discarded as degenerate. The isolated
`relative_gjk_probe.py` changes only those degeneracy tests to scale-relative
ones; the 40-iteration cap, 1 um convergence tolerance and 2 um contact
certificate remain unchanged. Production GJK and SDF code are unmodified.

Independent 512 triangle/box regressions compare against a float64 convex
minimizer with its own primal/duality-gap certificate. Old maximum distance
error: 1.0343 mm. Relative-degeneracy maximum error: 1.025e-9 m; all 512 meet
the 2 um support-plane certificate. Existing signed box-plane tests also pass.
The full rendered scene benchmark is running; this finding alone is not a
28 FPS or scene-acceptance claim.

The first full run labeled relative GJK still statically bound the original
GJK caller (`wp.static(gjk.core)` had already been resolved). Its 46.743 ms
result and 25/29 retention are NOT relative-GJK results. Rebuild the caller,
not merely its Python global, to activate the intended function.

The actually active run reached 36.05 ms/frame through 10.5 s and 36.79 ms/frame
through 20.5 s, then failed cup slip at 29.733 s (30.4 mm). This is a failed
acceptance run and has not been adopted. Next isolate convex and generic SDF
branches in separate kernels to reduce register pressure without changing
their contact formulas, then repeat full acceptance. Do not loosen slip limits.

### Follow-up full-scene results (not adopted)

- Split convex/SDF kernels: 37.518 ms/frame, 26.654 FPS; retained 18/27
  (66.67%), failing the unchanged 70% requirement.
- Convex witness followed by original SDF refinement, without convex culling:
  approximately 50.76 ms/frame early; cup slip failed at 26.583 s.
- Convex non-contact rejection with an audit of all actual SDF interpolation
  corners: 45.737 ms/frame, 21.864 FPS; only four grains lifted, failing delivery.
  The audit alone does not establish equality of all legacy contacts: the
  legacy accurate-distance evaluator has a zero-gradient special case.
- Combined audited culling and strict SDF witness refinement: 45.639 ms/frame,
  21.911 FPS; retained 26/31 (83.87%), complete acceptance passed. Still far
  below 28 FPS; not a completed optimization.
- Fresh unmodified baseline control: 50.814 ms/frame, 19.679 FPS; retained
  22/25 (88%), complete acceptance passed.

The IK objective launch-fusion test preserves joint solutions and residuals
exactly in four 32-iteration two-link cases. A separate rigid subwarp probe
will compare force/Hessian/pose on frozen inputs before any scene benchmark.
No scene constants, physical acceptance thresholds, or production defaults
were changed to obtain these numbers.

`cubic_search_probe.py` preserves the SDF field, using exact cell polynomial
coefficients. Within one clamping region, grid-exterior extension consists of
that polynomial plus distance to the clamped point. Corner regions reduce
to point/segment distance; face regions to a quadratic plus linear term;
edge regions to a linear term plus the norm of an affine vector. Crossing
regions/cells falls back to the original search. A cancellation-safe quadratic
root formula was required; the naive formula failed the objective test.
32,768 tested segments, including mirrored/nonuniform scale and tiny segments,
passed the scalar-objective check. This is not a universal bitwise equality
claim and does not by itself validate contact normals or entire demos.

## Affine lower-bound principle

For each fixed direction with norm at most one, fit an offset below every
corner value of every actual fine or coarse interpolation cell. Trilinear
interpolation is a convex combination of these corner values and reproduces
an affine function, hence the resulting plane remains a lower bound inside
the cell. Outside the grid, the existing added Euclidean distance preserves
the bound by Cauchy-Schwarz. The minimum affine value over a triangle/segment
occurs at a vertex. Only positive-gap rejection is used, after scale and
rigid-frame transformation and a conservative floating-point allowance.

The fit includes all cells, not sampled mesh geometry or an assumed exact
unit-Lipschitz SDF. Startup fitting is separate from timing. Further required
checks include transformed/mirrored bounds, candidate coverage against the
unmodified pipeline, and repeated full-scene acceptance before adoption.

## Follow-up: exact speculative searches and preconditioner probes

No production defaults or physical acceptance limits have changed.

- Eight-lane speculative golden search reproduces the original samples and
  outputs bitwise in 262,144 tested line queries. Frozen whole-pipeline tests
  at frames 240, 900 and 1800 also pass (normal rounding tolerance 2e-7).
  Standalone full scene nevertheless fails cup slip at 29.083 s; not adopted.
- Four-lane speculative version: 46.020 ms/frame, 21.730 FPS. Final cup lift
  fails despite 23/28 retained grains; not adopted.
- Original centroid prefilter followed by eight-lane speculative searches:
  44.192 ms/frame, 22.628 FPS, 19/25 retained; complete acceptance passes.
  This remains below the 28 FPS requirement.
- Inverse Ritz factor reuse passes three independent dense-system tests.
  Combined with speculative prefilter, IK fusion, subwarp rigid reduction and
  empty-contact graph branching, it fails shaft slip at 16.950 s (10.1 mm).
  No acceptance threshold was relaxed to accept it.
- Relative-dimensional rigid GJK degeneracy checks: 49.415 ms/frame,
  20.237 FPS, 15/19 retained; complete acceptance passes, no major speed gain.
- Cooperative exact SDF corner reads pass the 262,144-query bitwise tests,
  but the complete scene costs 56.711 ms/frame (17.633 FPS), with 22/30 retained.
  Reject this implementation for performance.

Instrumented captured-graph profile of the passing speculative-prefilter
variant at 18--19 s: collision 11.550 ms, self detection 4.698 ms, rigid
6.308 ms, particle-inclusive 19.527 ms, multilevel-inclusive 8.762 ms.
The 769 event pairs impose overhead (52.871 ms instrumented wall time);
these inclusive intervals are not additive, and are not the acceptance FPS.

The complete piecewise-trilinear segment minimizer passes 32,768 random/short
line objective comparisons after accounting for discontinuous quantized
fine/coarse interfaces. Maximum measured objective increase: 2.608e-8 m.
Its first full-scene interval costs 63.014 ms/frame through 10.5 s. The run
was stopped for this clear performance regression, not counted as a complete
acceptance pass. The original SDF and material remain unchanged in this probe.

## Conservative empty self-contact certificates

The scene currently runs 16 self-contact searches per frame (before particle
prediction and at the first sweep of each of eight substeps). These are not
duplicate snapshots and cannot simply be removed.

The new prototype queries an expanded radius and keeps all original immutable
rest/topology/world filters. A separating support plane certifies each pair's
clearance; the approximate closest-point witness only chooses that plane and
is never assumed to give a distance lower bound. Unqueried pairs are protected
by the expanded AABB. The certificate is valid only while every vertex's
Euclidean displacement is less than 0.49 times the certified clearance.
Actual query radius, contact laws, DAT publication and isotropic movement
limits remain unchanged. Public detector BVHs are still refit every call.

- Expanded AABBs alone: no reusable certificates in this cup; reject that
  version for overhead (44.762 ms/frame in the short test).
- Adding support-plane separation: 37.784 ms/frame, 26.467 FPS over the first
  600 frames; 216 rebuilds and 9,864 reuses. This is NOT full delivery acceptance.
- A later refinement stores the minimum positive certified clearance instead
  of rejecting an entire certificate when a pair lies within the maximum skin.
  Tests cover new overlaps, close parallel layers, random rotations, large
  translations, and subsequent small displacements. 105 pose comparisons pass;
  contact lists/counts/minima, DAT snapshots and reset factors match the original.
- Combined cache, eight-lane speculative prefilter, empty-contact scheduling
  and IK fusion: 36.862 ms/frame through 10.5 s, 38.222 through 20.5 s, then
  cup slip at 24.783 s. Failed acceptance; not adopted.
- Sixteen-lane speculative searches preserve all outputs bitwise in the same
  262,144 line-query tests, but the full combined run costs 42.021 ms/frame
  (23.798 FPS) and fails final cup lift. Thirty of 32 lifted grains remain;
  retention does not override the failed cup criterion.
- Zero-basis Ritz block elision matches the original GPU matrix/RHS bitwise
  for one through five clusters and matches an independent dense CPU product.
  Combined with the eight-lane prefilter: 44.447 ms/frame, 22.499 FPS, 24/30
  retained, complete acceptance passed. No meaningful overall gain demonstrated.

The existing native DAT collision-refresh regression also passes with the
certificate and empty-contact scheduling wrappers. No production changes
have been enabled, and the 28 FPS target remains unmet.

## Cached speculative queries and rigid work distribution

- Eight-lane speculative queries with per-lane exact cell caching: 35.724
  ms/frame (27.993 FPS), first 600 frames only. Not full acceptance.
- A faster cache-hit path avoids repeated fine-cell addressing and slot
  reads. The 262,144 independent line queries remain bitwise identical.
  Full combined run: 37.805 ms/frame (26.451 FPS), 18/20 grains retained,
  but final cup-lift assertion failed. Reject as an accepted default.
- Frozen rigid-color tests compare all three color groups. Multiwarp
  reductions meet the original pose tolerance and a per-body normwise
  force/Hessian tolerance of 3e-6 (entrywise relative error is unsuitable
  for cancellation-dominated off-diagonals). Single-scoop kernel costs
  22.567/17.812/12.881/12.319 us for 32/64/128/256 lanes. Two 80-body
  groups cost 18.425/17.434/19.377/26.612 and
  18.435/16.268/16.507/25.966 us respectively. These are frozen single
  kernel timings, not scene FPS or physical acceptance.
- An alternative Ritz application stores Z=L^-1 and applies Z^T(Z r),
  retaining the positive Gram structure rather than forming C^-1.
  Independent tests at widths 6/18/30 and condition numbers 1--1e6 pass:
  positive preconditioning and accuracy comparable with the original
  Cholesky solve. Full coupled-system and scene tests remain necessary.

Follow-up checks:

- Fast cell cache plus speculative prefilter passes the full frozen-contact
  and cache comparison at frames 240, 900 and 1800 (original 2e-7 normal
  tolerance; all other compared outputs exact).
- The triangular-factor Ritz implementation passes all three existing
  coupled translation/dense-system regressions.
- Occupancy-selected rigid workers (128 for underfilled colors, 64 otherwise)
  plus inverse-factor Ritz and certified cached queries: 35.643 ms/frame
  through 20.5 s, then cup slip at 25.650 s. Reject.
- Adding contact-gather/surface fusion: full 37.512 ms/frame, 26.658 FPS;
  23/33 grains retained (69.697%), below the unchanged 70% requirement.
  Cup geometry and lift pass, but delivery does not. Not accepted.
- Packing two independent 16-lane surface updates into a full warp passes
  the eight-sweep contact/elasticity/DAT regression, maximum coordinate
  difference 5.821e-11 m. A first draft used an unsupported Warp array `.size`
  attribute; compilation caught it and it was corrected to `.shape[0]`.
  Full-scene measurement of this layout is in progress.

Further completed measurements (still no production adoption):

- Two-particle subwarp surface layout: full 38.796 ms/frame, 25.776 FPS,
  19/24 retained, all acceptance checks passed. Slower than the 16-thread
  tile layout; do not retain based on occupancy intuition alone.
- Select the surface/self-contact branch once per eight-sweep block:
  full 38.666 ms/frame, 25.862 FPS, 24/32 retained, acceptance passed.
  Independent two-layer tests exercise both no-self-contact and active
  self-contact fallback; positions/velocities remain within original test
  tolerances. Reduced graph-branch count alone did not improve full timing.
- Retain a lane's SDF cell across all line searches of one face:
  frozen contact/cache comparisons pass at frames 240/900/1800;
  full 38.089 ms/frame, 26.254 FPS, 19/29 retained (65.52%), failed retention.
- Surface adjacency construction now uses at most 4096 workers and strides
  over every active row instead of launching one thread per 65536 capacity
  slot. A 20,000-row regression checks all counts and every adjacency entry;
  both this and the original eight-sweep fusion test pass.
- Analytic / warm-SDF / cold-SDF queue grouping and exact LM fixed-point
  elision are new unaccepted probes. They may not change contact predicates,
  solver tolerances or physical acceptance thresholds.

## Complete near-target result

Coherent analytic/warm/cold query queues, face-lifetime cell caching,
inverse-factor Ritz, occupancy-selected rigid workers, certified self-contact
reuse, fused surface gathering, block-wise graph selection, and exact LM
fixed-point elision complete the unchanged 2850-frame rendered benchmark at
**36.148 ms/frame (27.664 FPS)**. All physical acceptance checks pass:
21/23 grains retained (91.30%), five-finger contact fraction 1.0,
cup distortion RMS 2.511 mm, rim minimum radius 29.452 mm,
upright cosine 0.9908, shaft-center slip 0.660 mm.
This is not 28 FPS and must not be rounded into a success claim.

The coherent queues pass all three frozen contact/cache snapshots. Exact LM
fixed-point checks compare q, residuals, damping, costs, accept flags,
proposed q and dq bitwise against the complete original iterations for
batches 1/4, iteration counts 8/12/31/32/80, normal/zero step sizes and
reachable/unreachable targets. W1-specific shadow checks remain required.
This optimization uses no convergence tolerance; state must repeat exactly.

Next probe removes only two provable no-ops: the entry force/Hessian clears
when the fused path overwrites every particle, and joint dual kernels when
the solver has zero constraint slots (all joints FREE). Later coarse-solve
clears remain intact. Poisoned NaN scratch tests and all three coupled
translation regressions pass. This run also omits the previously unhelpful
block-wise graph grouping. No production code has been enabled yet.

That no-op-elision/per-sweep variant completes at 37.388 ms/frame,
26.747 FPS, 17/24 retained; acceptance passes, but performance regresses.
An alternating-order microbenchmark at the same frozen robot command
separates layout cost from trajectory differences: per-sweep graph branches
average 25.760 GPU ms, per-block selection 24.918 GPU ms (100 samples each).
This is not a rendered FPS measurement; it supports retaining block grouping.

W1 IK executes 11,228 of 11,778 requested four-iteration blocks through frame
1080. Exact fixed-point elision therefore skips only 4.67% of these blocks,
and adds comparison/copy/conditional overhead. The next candidate omits it
and retains the full original 32-iteration IK command solve. It combines
the other queue/cache/solver optimizations, no-op elision, and block grouping.
The 28 FPS target remains unmet; production defaults are still untouched.

## 2026-09-13 — Freeze the user-accepted 27.47 FPS combination

The user accepted 27.47 FPS and requested cleanup rather than further tuning.
`run_popcorn_fast.py` now fixes: IK exact fixed-point elision, face-lifetime
cell caching, coherent search queues with 32-thread blocks, rigid occupancy,
inverse-factor Ritz, default certificate skin (0.5), surface fusion, analytic
IK objective fusion, and whole-sweep block graph grouping. No no-op elision,
packed controller snapshots, expanded certificate skin, batched target updates,
or rigid dual stride is enabled.

The previous query32 full run was 36.408884 ms/frame / 27.465824 FPS,
23/23 retained, all physical assertions passed. A separate later snapshot/dual
experiment reached 36.308438 ms / 27.541808 FPS, 15/18 retained; not selected.

The cleaned launcher was rerun with 30 warmup + 2850 timed rendered 1080p
frames: **35.968823 ms/frame / 27.801855 FPS**, 19/21 retained (90.48%).
All physical assertions passed. Final grip slip was 0.374 mm, cup deformation
RMS 1.841 mm, minimum rim radius 30.426 mm. Same-state cached/public renderer
pixel difference was zero. These are full-run results, not isolated GPU timings.
Contact reduction order can change trajectories; FPS differences between runs
are not proof that cleanup itself accelerated anything.

Unused newly created probes were moved, not destroyed, into the external
`newton_cleanup_archive/popcorn_rejected_2026-09-13` directory. Historical
tracked files were preserved. Unused standalone main entrypoints were removed
from retained implementation helpers. See `TESTING_FAST.md` for the one visual
test entrypoint and its explicit experimental/process-global limitations.
Production defaults remain untouched; this is not a claim of general solver
integration or validation of every other demo. No commit or push was performed.

Additional cleanup validation: 96 same-state self-contact shadow comparisons
over the full 48 seconds all matched the original detector. Certificate counters
were 676 rebuilds, 676 certified-empty rebuilds, 45,500 reuses. The default-skin
105-pose certificate unit test also passed. This audit is not an FPS benchmark.
Exactly 69 unused newly created experiment files were archived recoverably.
All seven isolated retained regression scripts passed after cleanup: IK fixed
point, IK objective fusion, inverse-factor Ritz, self-contact certificate,
surface fusion, batched surface graph, and speculative SDF search. The visual
launcher argument parser was checked with `--help`; visual acceptance is left
to the user. `git diff --check` passed. No simulator process remains running.

## 2026-09-13 — Integrate into solver code (supersedes isolated entrypoint)

The user approved the visual result and requested core integration and cleanup.
The code now lives in MJVBD V2 `fast_kernels.py` / `fast_path.py`, the existing
rigid fusion / two-level PCG implementation, and IK `analytic_row_kernels.py` /
`cuda_fast.py`. Solver and pipeline state is instance-owned. No `wp.launch`,
array-method, or solver-class monkeypatch is installed. IK uses a typed kernel
factory for heterogeneous argument lists, with statically defined math functions;
no runtime AST rewriting or experimental-file imports remain in the core.

The original popcorn example explicitly enables the new runtime options.
Other examples' defaults remain unchanged. CPU, differentiable and deterministic
configurations fall back. Surface batching retains the empty-contact/capacity
guards; cache memory is bounded with fallback. Model notifications and explicit
BVH rebuilding invalidate the empty-contact certificate. Captured graphs must
be rebuilt after model-property changes. These execution options are runtime-only
and intentionally do not add a USD asset schema.

All 32 remaining newly created experiment Python files were moved to
`newton_cleanup_archive/popcorn_integrated_probes_2026-09-13`; unit coverage was
migrated into normal `newton/tests` modules. Tracked historical lab files remain.

Full integrated validation attempts, all with unchanged physical settings:

1. `integrated-full-01.log`: failed at 22.683 s, cup slip 31.7 mm.
2. `integrated-full-02.log`: passed 2850 timed rendered frames, 36.793996 ms/frame
   (27.178347 FPS), 22/29 grains retained (75.86%), grip slip 0.957 mm, cup
   deformation RMS 1.966 mm. Same-state render pixel comparison was exact.
3. `integrated-full-final.log`: failed at 31.833 s, cup slip 30.2 mm.

These are one pass and two failures, NOT evidence of reliable scene completion.
Contact roundoff/order sensitivity is a possible mechanism, not a demonstrated
exoneration of the integration. No passing runs were substituted for the failed
ones, and no grasp/material/trajectory/acceptance changes were made. Further
scene stability investigation remains necessary.

The first combined core regression run passed 28 tests covering IK exact state,
SDF search, warm/cold pipeline outputs, DAT/certificate fallback, PCG, original
contact optimizations, and popcorn diagnostics. Additional fallback/invalidation
tests were subsequently added. Ruff and formatting checks pass. No commit/push.

Final combined regression: **30 tests passed in 6.138 s** after adding CPU/AD
fallback and certificate invalidation checks. A further IK run explicitly
included position + rotation + joint-limit objectives: 3 tests passed, including
bit-identical LM state at batches 1/4 and iteration counts 8/12/31/32/80.
The ordinary example's direct-file `--help` entrypoint also passed with all
experimental Python scripts removed from the working tree. The two pre-existing
geometry files still have no semantic diff and were not restored or staged.
