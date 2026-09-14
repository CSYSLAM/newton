# 320-grain popcorn optimization (pending user acceptance)

## Pre-push review

The scoped review finds no blocking issue; the change fits the opt-in
MJVBDV2 performance work without adding a public solver option. Re-ran the
19 targeted tests plus 23 contact-optimization, invariant and truncation-cache
tests: all 42 passed. An external replay of the committed fixed-128 adjacency
kernel fails the new 129-record regression as expected; the new kernel passes.
All staged-file pre-commit hooks pass. The required full-repository run still
reports existing Ruff/typos findings in unrelated experimental scripts; these
are not silently fixed or included. The unrelated geometry SDF file is not
staged. Full-scene timing and remaining slip limitations are recorded below.

## Revised target: at least 27 FPS

The user relaxed the target to 27 FPS or better, prioritizing working grasp
and delivery. Candidate defaults are now 320 grains, six substeps, eight
local sweeps and sixteen PCG iterations. No trajectory, friction coefficient,
contact law or acceptance threshold was changed for this candidate.

| Initial cup x perturbation | Wall ms/frame | FPS | Retained/lifted | Five-finger fraction |
| --- | ---: | ---: | ---: | ---: |
| +0.5 mm | 34.807 | 28.730 | 31/36 | .996813 |
| -0.5 mm | 35.009 | 28.564 | 28/33 | .998805 |

Both full 48-second rendered runs pass unchanged acceptance. Cup slip at
40.5 s is 8.908 / 6.444 mm respectively: this is not a zero-slip claim.
The lower substep count changes discretization and plastic response; it is
not an exactly equivalent kernel-only speedup. The sixteen-iteration solve
addresses convergence within the larger timestep. The direct-file-entrypoint
test of these defaults also completed 48 seconds and passed final acceptance:
12/12 grains retained, five-finger fraction 1.0, cup RMS 1.678 mm, final rim
radius 30.667 mm, and final wrist-local cup slip 5.6 mm. Delivered counts vary
with contact trajectories; twelve is the existing minimum lifted-count gate,
so this is a pass, not evidence of consistently full scoops.

Final regression run passed all 19 targeted tests; Ruff check/format and
`git diff --check` pass. The PowerShell redirected test commands report
`NativeCommandError` for stderr warnings/progress, but the unit log ends in
`Ran 19 tests ... OK`; the direct example has no traceback and prints its
final report after the unchanged assertions. No rejected experimental
preconditioner or contact-law path was installed. Nothing has been staged or
committed during this follow-up. The current defaults are ready for visual
user acceptance, not a claim of universal stability or zero slip.

## Follow-up: restore the committed grip controller

The following supersedes the controller selection described below. Holding
320 grains and eight substeps fixed, restore only the committed grip force
targets `(4, 1, 1, 1, 1) N` and committed bounded controller. Keep the solver,
collision, and controller-I/O performance work. Full 48-second checks passed:

| Cup x offset | Wall ms/frame | FPS | Retained/lifted | Cup RMS mm | Final rim mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| +0.5 mm | 40.480 | 24.704 | 31/39 | 1.836 | 30.460 |
| -0.5 mm | 40.576 | 24.645 | 25/32 | 1.898 | 30.397 |
| nominal | 40.425 | 24.737 | 32/35 | 1.842 | 30.357 |

All three have five-finger contact fraction 1.0 and pass the original final
height, shape, and delivery checks. This supports removing the uncommitted
higher-preload/faster-acquisition/serial-release controller combination; it
does not isolate a single parameter as the cause or prove all trajectories.
Production now restores that committed controller. Tests specific to the
rejected serial-release policy are replaced by bounded-feedback checks.
The five popcorn diagnostic tests pass. Eight substeps remains the default;
30 FPS is still pending, and no worktree changes have been staged or committed.

An independent exact lagged-normal soft-friction tangent prototype passes
finite-difference checks in both regularized and sliding regions. It has not
been installed in the solver: the full-scene test failed to establish the
opposing grasp before lift. A more exact derivative does not alone establish
globalized nonlinear-solver stability; retain the original friction majorizer.

Further isolated budget tests with the restored controller:

- Six substeps, eight local/eight PCG iterations: failed five-finger contact
  during lift at 12.283 s; index force approached zero at its travel limit.
- Eight substeps, four local/four PCG iterations: slipped 30.8 mm at 23.100 s.
- Eight substeps, eight local/four PCG iterations: slipped 31.7 mm at 21.750 s.

None is installed. A separate 128-thread PCG reduction-block experiment retains
all eight/eight iterations and passes both component-connectivity and isolated
linear-system tests. It showed no early-frame timing benefit and slipped
30.1 mm at 24.500 s; the production block remains 256 threads.

Soft-contact-only friction regularization of 0.003 m/s (rigid friction unchanged)
with four PCG iterations slipped 31.4 mm at 26.850 s. This does not establish a
roundoff-related fix and is not installed. Experimental scripts live outside
the repository and add no production options.

Final targeted regression bundle: 19 tests passed, covering component PCG,
cached bending, controller snapshots, table clearance, fast CUDA paths,
diagnostics, and exact SDF search distances. Direct file-entrypoint test mode
completed 48 seconds without the cup-slip assertion, ending with 9.8 mm cup
slip, 2.193 mm cup RMS, 30.028 mm rim radius, and five-finger fraction 1.0.
However, it retained only 23/37 grains (62.16%), failing the unchanged 70%
delivery criterion. Thus the three passing rendered benchmark runs do not
establish full example acceptance. An external lower-outlet trajectory trial
(12 mm instead of 25 mm above the observed rim at final pitch) delivered 35
grains but slipped 30.3 mm at 47.383 s. It is not installed.

### Frozen-state numerical audits

External diagnostic scripts compare fused and ordinary surface scheduling
with the same captured arrays restored before each two-substep replay. At
14 s the cross-path maximum difference was about 0.27 micrometers, comparable
to repeated-path variation. At 24 s, cross-path maximum/RMS differences were
15.71/1.245 micrometers versus within-path maximum differences 8.02--9.73
micrometers. These limited samples do not show a gross fused-path discrepancy.
They are not proof of trajectory equivalence: internal BVH handles are not
array snapshots and contact reductions remain nondeterministic.

For the assembled 24 s linear system (841 particle + 321 free-body vector
rows), a float64 sparse direct reference had relative residual 4.72e-16.
Current GPU PCG gave the following relative linear energy-norm errors:

| PCG iterations | 4 | 8 | 12 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Relative energy-norm error | .7422 | .3034 | .1684 | .1199 | .03494 |

These are errors in one frozen linear solve, not percentages of simulation
error. All sampled step clamps were 1.0, so clipping was not the limiting
factor for those solves. Fewer PCG iterations need a better preconditioner,
not simply a lower iteration setting.

A CPU-only harmonic extension of the existing geometric coarse basis into
contacting free bodies was tested with 1, 2, and 4 block-Jacobi extension
sweeps. Four extensions only changed the eight-iteration error from .303419
to .302096, and the four-iteration error from .742168 to .741162. This is a
no-go: the extra projection/setup cost is not justified. No production code
or public option was added.

An eight-direction recycling prototype, using Krylov directions recorded from
the preceding real coarse solve and projecting them against the *current*
operator, reduced the eight-iteration error only from .530648 to .504022 in
another 24 s sample. Four iterations gave .639840, worse than the original
eight. This does not justify GPU integration or fewer iterations.

A saved, separate 24 s operator permits CPU-only preconditioner comparisons:
ordinary block-Jacobi + Ritz gives .295922 at eight iterations. Symmetric
color-Gauss-Seidel + Ritz gives .311609 at four, .248231 at six and .196109 at
eight, but requires forward/reverse passes over seven colors. Disjoint RCM
patches of 8/16/32/64 vector rows give .352436/.339883/.328315/.307745 at four
iterations. None yet demonstrates enough iteration savings to pay for its
setup and application cost; none is installed. These comparisons preserve
the captured matrix and RHS and do not change contact forces or thresholds.

The full eight-substep / sixteen-PCG diagnostic (+0.5 mm cup offset) passed
48 simulated seconds at 43.119 ms/frame (23.192 FPS): 23/26 grains retained,
five-finger fraction 1.0, cup RMS 1.804 mm and rim radius 30.492 mm. This is
slower, not a proposed default, and one pass does not prove that linear error
is the unique cause of slip. A six-substep / sixteen-PCG combination will
check whether fewer collision passes can fund stronger per-substep solves.
Its first full +0.5 mm run passed at 34.807 ms/frame (28.730 FPS), retaining
31/36 grains, five-finger fraction .996813, cup RMS 1.774 mm and final rim
radius 30.488 mm. Cup slip at 40.5 s was 8.908 mm: residual drift remains.
Maximum plastic hinge rotation was .31548 rad versus .18544 in the 8/16 run;
changing timestep is not an exactly equivalent numerical optimization.
The negative-placement test is still required before changing any default.

## Earlier investigation after the reported slip (historical)

The 30 FPS target is **not yet accepted**. Six-substep configurations passed
some runs but failed others, including +/-0.5 mm initial cup-placement checks.
The default now uses eight substeps and eight local sweeps, with 320 dynamic
grains. Earlier six-substep timing and acceptance statements below are history,
not the current release status.

The holding controller uses a continuous overload band, releases only the most
relatively overloaded finger at a time, and reduces holding gain relative to
acquisition. These are physical controller changes, not solver speedups. They
did not alone make six substeps robust. Force targets, materials, slip limits,
and physical acceptance thresholds were not relaxed.

A dense-contact fast-path allocation now permits 256 records per particle
where this fits the existing 256 MiB budget, otherwise retaining 128. Overflow
still triggers the original fallback; no contacts are discarded. The regression
for 129 records failed before this change. Capacity and overflow checks pass
after it, and the current targeted regression bundle passes all 19 tests.

Eight-substep negative-placement full-run results (48 simulated seconds,
1080p rendering included, initialization excluded):

- Original 128-entry allocation: 41.585 ms/frame, 24.05 FPS; 26/32 grains
  retained. A sampled 141-contact row triggered fallback.
- Adaptive 256-entry allocation: 40.257 ms/frame, 24.84 FPS; 24/29 grains
  retained, five-finger contact fraction 1.0, cup RMS deformation 2.275 mm,
  final rim radius 29.896 mm. Cup slip at 40.5 s was 4.338 mm. Full validation
  passed. These are different contact trajectories, not an isolated timing
  attribution or proof of robust acceptance.

The subsequent nominal-placement run also passed all 48 seconds: 40.528
ms/frame (24.67 FPS), 24/28 grains retained, five-finger fraction 1.0, cup RMS
2.627 mm, final rim radius 29.641 mm. Cup slip at 40.5 s was 4.622 mm.

**The positive-placement check subsequently failed the final cup-height test.**
It retained 26/30 grains but cup height fell to 74.660 mm and wrist-local slip
reached 17.101 mm at 40.5 s. Wall time was 40.709 ms/frame (24.56 FPS).
This run also loaded the value-only centroid-cull experiment: it uses the
existing exact SDF lower-bound evaluator rather than computing an unused
gradient. It is not a clean isolated performance comparison. Thus neither
eight substeps nor the controller changes establish robust acceptance yet.

Checked-branch graph profiling over 960 substeps measured 13.413 ms collision,
3.716 ms face search (included in collision), 5.496 ms coarse PCG, and 30.860
ms total simulation GPU time. The diagnostic verifies all recorded fallback
predicates are zero; it is not the ordinary acceptance timing path.

The exact value-only SDF query and fast-path suite passed six tests, including
a new same-color-contact fallback check. This supplements the 19-test bundle
above; it is not six additional distinct tests.

An external bounded slip-to-grip-pressure controller experiment (no prop-state
edits, maximum target multiplier 1.5) held the cup at eight substeps with the
positive placement perturbation: 4.291 mm slip at 40.5 s, final rim 29.663 mm.
However, it lifted only 10 grains and retained 6 (60%), failing full delivery
acceptance at 25.22 FPS. It is not installed in the example.
The same external controller at six substeps failed at 25.733 s with 30.3 mm
cup slip and nearly zero index-finger force. Reject this as a six-substep fix;
do not merge it on the strength of the eight-substep holding result.

Rejected experiments include stronger/weaker preload, wider force bands,
holding joint offsets fixed, contact-loss reflexes, coupled finger curl,
additional local sweeps, more coarse PCG iterations, altered coarse checkpoints,
smaller coarse step bounds, and increased friction regularization. Each failed
at least one stability or transfer check; none is retained as a claimed fix.
Eight-substep 20-local-sweep reference passed but cost 50.321 ms/frame.

Raw logs remain in the external cleanup archive. No staging, commit, or push
has been performed during this investigation.

Starting commit: `9ae5cff9` on `FAST_MJVBDV2`.

The requested target is 320 dynamic grains and at least 30 FPS. The staged
implementation was committed locally before this work; new changes remain
unstaged. No remote push is authorized.

## User regression after the first handoff

The user reproduced a six-substep/eight-sweep cup slip at 10.333 s (34.2 mm).
The previous passing runs do not establish robust acceptance; the physical
quality claim is reopened. Do not relax the slip assertion or treat the earlier
30 FPS runs as proof this failure is fixed.

The reported forces `[12.3321, 4.6580, 44.9872, 14.8090, 4.2467]` make the old
independent overload controller open **all five digits at once**. A regression
test using that exact case fails before the change (five opening commands).
The pending controller change unloads only the most relatively overloaded digit
per update after grasp establishment; other loaded digits hold, and underloaded
digits retain their closing feedback. Force targets, joint/speed bounds,
materials, contact pipeline and failure thresholds are unchanged in this fix.
This corrects a controller behavior; it does not by itself prove the full slip
failure has been eliminated. Full-run repetitions are in progress.

Follow-up diagnostics (not accepted fixes):

- Sequential overload release passed one nominal full run at 31.92 FPS,
  retaining 25 of 33 lifted grains. A +0.5 mm initial cup placement test
  failed the unchanged final lift check (77.9 mm at 40.5 s). Thus the
  controller change alone is insufficient.
- Increasing linear PCG iterations to 12 with the same placement perturbation
  slipped 34.6 mm at 8.783 s. More linear iterations alone are not a fix.
- Fixed the diagnostic harness to resize its per-iteration metric buffer
  when overriding linear iterations. An earlier 12-iteration probe without
  this resize was stopped and is invalid; ordinary default runs were not
  affected by that diagnostic-only issue.
- The partitioned linear solve still used a global correction clamp.
  An experimental domain-local clamp preserves the same displacement caps
  and uniform scaling on contact-connected unknowns, while preventing
  detached grains from reducing the particle step. CPU/CUDA regression
  tests verify that invariant. However its +0.5 mm full-scene run still
  failed the final lift check (75.4 mm at 40.5 s): it is not a demonstrated
  solution to the grasp failure.
- The same domain-bound candidate with 8 substeps held the cup at 91.2 mm
  lift and 2.74 mm slip at 40.5 s, but transferred only 7 grains and ran at
  25.63 FPS. This is a time-resolution diagnostic, not acceptance.
- Doubling physical grip-force targets at 6 substeps preserved support but
  crushed the cup opening (15.3 mm final minimum radius; 10.5 mm distortion),
  and retained only 16/24 grains. Reject this preload change. It was tested
  only through an external script, not installed in the example defaults.

Physical grasp robustness remains unresolved. No slip/lift threshold has
been loosened, and no object attachment or velocity override has been added.

Further isolation:

- 0.75x preload with domain-local bounds passed the +0.5 mm run at
  31.36 FPS but had little lift margin. The nominal run held the cup but
  lifted only 8 grains, failing full acceptance. With the original global
  bound, the -0.5 mm run slipped at 21.967 s. Do not install this preload.
- Removed the domain-local step-bound experiment and its dedicated test;
  it did not establish an end-to-end quality benefit. The earlier independent
  PCG reductions remain unchanged.
- With original force targets, original global bound and sequential overload
  release, **10 local sweeps** (6 substeps, unchanged 8 linear iterations)
  passed the +0.5 mm run: 32.985 ms/frame / 30.32 FPS, 31/33 grains retained,
  85.4 mm cup lift and 5.99 mm slip at 40.5 s. This suggests local nonlinear
  contact convergence matters more than simply increasing the linear budget.
  Additional placement/repetition tests are required before accepting it.

The 10-sweep negative-placement repetition subsequently slipped; do not
install the larger iteration budget on the strength of the first pass.
A second controller discontinuity was reproduced in a failing unit test:
when the last weak digit crosses its target, the other digits' overload
threshold abruptly falls from `max(4*target, 8 N)` to `2*target`. A supporting
middle finger at 3 N is consequently commanded open merely because the pinky
changes from 0.99 N to 1.01 N. Keep the established-grasp load band fixed,
with the same finite overload limits; force targets and speed bounds remain
unchanged. This candidate still requires full-scene verification.

Holding-controller candidate:

- Fixed load band alone still slipped in the negative-placement test.
- Reducing the gain during acquisition also failed to maintain five-finger
  opposition before lift; a diagnostic showed zero index contact at its
  authored 12-degree extra-travel limit. No URDF or travel limit was changed.
- Retain acquisition gain 0.15 and use 0.04 only after grasp establishment,
  together with the fixed overload band and sequential release. Force targets,
  materials, trajectories, substeps (6), local iterations (8), linear budget,
  and safety/acceptance thresholds stay unchanged.
- +0.5 mm placement: full acceptance passed, **31.48 FPS**, 25/33 grains
  retained, 2.56 mm distortion, 29.26 mm rim radius, five-finger fraction 1.0.
- -0.5 mm placement: cup held for all 48 s (88.2 mm lift / 4.47 mm slip at
  40.5 s), **32.65 FPS**, but only 5/12 grains remained in the cup. This is
  explicitly a full-scene acceptance failure despite improved grasp stability.
- Nominal placement repetition is pending. Do not describe the complete
  320-grain scene as robustly accepted from these two runs.

Subsequent results supersede that pending note: nominal placement also lost
the required lift margin. Position-only finger holding, low-preload contact
recovery, coupled MCP/PIP closure, and eight additional particle-only sweeps
were each tested externally and did not eliminate late cup slip. These
experimental branches are not installed in the solver/demo.

The current controller candidate instead uses a fixed **2x proportional**
force band (no 8 N floor or weak-digit switching), sequential overload release,
and reduced post-acquisition gain. Its nominal run passed at 31.65 FPS with
27/30 retained grains, but the negative-placement repetition still slipped.
This is not robust acceptance.

A higher-budget reference (8 substeps, 20 coupled fine sweeps, retaining the
existing two 8-iteration global solves) passed the negative-placement case:
19.87 FPS, 23/27 retained grains, 2.29 mm distortion, 29.82 mm minimum rim
radius, five-finger fraction 1.0. It is a diagnostic reference, not the default.

Numerical-resolution hypothesis under investigation: at meter-scale float32
coordinates, the displacement required by a light cup's near-static friction
balance can be smaller than one position ULP when the smoothing velocity is
1e-4 m/s. The solver's ordinary default is 1e-2 m/s. Increasing the *global*
value to 0.003 m/s caused unstable grain motion (32.4 m/s at 12 s), so that
change is rejected. An external soft-contact-only force-law experiment is
being tested, preserving equal-and-opposite reactions and the original
rigid-rigid law. No smoothing change is installed in production code yet.

## Acceptance

Use `validate_default.py`: 1920 x 1080, VSync disabled, 30 warmup frames and
2850 measured step-plus-render frames. The initial scheduling experiments
retained 8 substeps, 8 VBD sweeps and 8 coarse PCG iterations. The final candidate
uses 6 substeps after the separate time-discretization audits below; this is a
numerical-budget change, not a claim of bit-exact equivalence to 8 substeps.
Retain 8 VBD sweeps, 8 coarse PCG iterations, the existing material/contact
parameters, robot trajectory, and all grasp, containment, and transfer
assertions. Initialization and compilation are excluded. A failed run or a
short sample is not acceptance.

## Experiments

| Experiment | Observed partial wall time | Full physical result |
| --- | --- | --- |
| Only double 160 to 320 grains | 41.061 ms/frame at 20.5 s | Cup slip 32.7 mm at 25.183 s |
| Unroll Ritz indices / avoid dynamic local-vector indexing | 41.055 ms/frame at 20.5 s | Cup slip 30.2 mm at 22.583 s |
| Above plus 64-thread PCG update | 42.409 ms/frame at 10.5 s | Cup slip; rejected |
| Geometric hinge gradient, uncached anchors | 40.498 ms/frame at 10.5 s | Cup slip 30.1 mm at 24.333 s |
| Reuse existing geometric bending and immutable substep anchor cache; cull diagnostic rays outside shell bounds | 39.332 ms/frame at 20.5 s | Cup slip 30.9 mm at 29.317 s |

The PCG scheduling experiments have been removed. The standalone duplicate
geometric bending implementation has also been removed in favor of the
existing `particle_surface_cache` functions. The latest retained candidate is
not yet approved for handoff: neither performance nor grasp robustness meets
the target.

Captured phase profiling of the 320-grain baseline around 18--20 s measured
13.705 ms/frame in collision and 30.909 ms/frame for the whole simulation
graph. The surface fast branch was eligible and active (fallback 0, maximum
per-particle contact adjacency 33). Uncaptured kernel timings are only used
to identify expensive operations, not as captured-graph performance claims.

## Numerical checks

- Geometric hinge force/Hessian comparison covers 4096 random hinges,
  damping on/off, boundary and collapsed hinges, on CPU and CUDA.
- A separate central-difference energy-gradient test covers 48 hinges.
- These tests and the three existing CUDA fast-path tests passed together
  after integrating the existing fixed-anchor implementation (5 tests).
- Ray classification is checked against the identical ray calculation with
  expanded unused bounds vertices, including deformed shells and boundary
  samples. The virtual cap remains measurement-only, never a collider.

Logs are kept outside the source tree in
`E:/csy_work/CG/Engine/newton_cleanup_archive/popcorn320_*.log`.

## Further experiments (not accepted)

- Reuse SDF interpolation corners for inner search gradients; retain the
  original final contact evaluation. An initial replacement of that final
  evaluation changed one contact coordinate by one ULP and was removed.
  Both strict SDF fast-path tests then passed, including 262144 searches.
- Batch controller state reads and four wrist-target writes without changing
  their values. CPU/CUDA snapshot and target tests pass. CPU snapshots must
  copy because Warp's CPU NumPy view aliases the source buffer.
- Place the additional 160 free dynamic grains in two side banks. The first
  160 spawn positions are unchanged. This changes stock layout, not solver
  performance semantics; compare it separately from kernel optimizations.
- Adjust only physical finger joint pressure feedback: recover a digit below
  its target, and use thumb/index targets of 5/2 N. These are controller
  changes, not solver speedups or object attachments. The prior 4/1 N targets
  lost index contact and eventually let the cup slip under its payload.
- The combined `popcorn320_io.log` run completed all 2850 timed frames at
  **39.326211 ms/frame / 25.428333 FPS**, with five-finger contact fraction 1.0
  and late cup slip approximately 5.3 mm. It lifted only 7 grains and retained
  6, so it FAILED the unchanged delivery requirement. This is not acceptance.
- Routing provisioned convex rigid pairs to the existing GJK/analytic path
  failed with 32.4 mm cup slip at 11.450 s (`popcorn320_convex.log`). Removed
  the experimental routing flag from the demo and shared collision classes.
- Current captured profiling around 18--20 s: collision 13.039 ms, including
  soft SDF search 3.307 ms; local sweeps 15.057 ms; whole simulation 30.080 ms.
  Nested values must not be added to the whole-graph total. Surface fallback
  remained zero. Event recording inside conditional child graphs is not
  supported, so no per-PCG event result is claimed.
- Testing contact CSR compaction before the coupled translation PCG while
  retaining linked-row summation order and every contact. The first trial
  exposed a temporary allocation in Warp's scan that conditional graphs do
  not allow. Replaced that scan with the existing scratch-backed geometry
  scan and added conditional-graph coverage. Full scene results are pending.

No trial lowers physical acceptance thresholds, adds object attachments,
freezes grains, or reduces substeps, sweep counts, geometry, or rendering
resolution. The 30 FPS objective remains unmet.

## Later physical and solver checks

- CSR compaction had no demonstrated end-to-end gain and the scene slipped
  at 33.283 s. Removed its changes from contact storage, Ritz projection,
  coupled translation, and the contact-row test; scratch-backed scans are
  not part of the retained candidate.
- Defer self-contact BVH refitting while the empty-set certificate is valid.
  Every real query still refits first. The motion/jump/new-contact comparison
  and CUDA surface tests pass. IK feedback backups now remain on the GPU;
  CPU/CUDA tests cover accepted retries and restoration after all retries fail.
- Cache coarse SDF interpolation corners over their full demoted subgrid,
  not just one fine cell. Exact search/contact tests still pass.
- Increasing thumb/index targets to 6/3 N and adding distal curl did not
  prevent slip; both changes were removed. Keeping 5/2 N but raising tactile
  feedback gain to 0.15 and bounding closure at 0.18 rad/s (opening remains
  limited to 0.06 rad/s) passed the full run in `popcorn320_fastgrip.log`:
  **39.762731 ms / 25.149178 FPS**, 29/34 retained, five-digit contact 1.0,
  late slip about 4.4 mm. This is a physical joint-controller change, not
  a solver speedup. The performance target still fails.
- Experimental component PCG conservatively groups all particles and their
  contact-connected free bodies together; detached rigid islands form a
  second independent block. Each block performs all 8 PCG iterations with
  its own step lengths. No contact or matrix block is removed. Frozen tests
  show exact isolation from detached loads and reduced quadratic error.
  A fully connected contact graph matches the original solve.
- `popcorn320_component.log`: **40.125642 ms / 24.921720 FPS**, 31/33 retained,
  five-digit contact 1.0. Same-state PCG benchmark: global 0.302167 ms with
  particle residual 0.930975; component 0.323542 ms with residual 0.572835.
  This demonstrates a quality benefit, not a performance benefit.
- Experimental surface solve/commit fusion uses ping-pong position buffers.
  Non-selected particles still receive exactly the original displacement
  clamp; no inter-particle reads observe partially committed colors. Reject
  inconsistent/incomplete color partitions. CUDA reference tests pass.
  Its first full run slipped; it is not approved for handoff.
- Persistent component PCG matched split execution exactly in the frozen
  unit tests, but was slower on the real operator: global 0.304951 ms,
  split component 0.335879 ms, persistent component 0.407068 ms. Particle
  residuals were 0.729303, 0.330489, and 0.330489 respectively. Serializing
  sparse products within a CTA outweighs saved launches. Do not enable the
  persistent variant as the default on this evidence.

## Further exact-work scheduling probes

- Removed the rejected persistent-PCG implementation and its unused option;
  retained the split-component connectivity/quality regression tests.
- Removed a duplicate face-centroid SDF bound already evaluated on the same
  frozen geometry during pair gathering. Search/controller/CUDA tests: 8 pass.
- Full current combination: `popcorn320_bound_reuse_full.log`, 40.520608 ms /
  24.678800 FPS, 30/35 grains retained, five-finger fraction 1.0. Target fails.
- External-only 64-register limit: 41.419861 ms / 24.143007 FPS, 31/34 retained.
  No demonstrated speedup; production register limits remain unchanged.
- External-only doubling of mesh-SDF work chunks to 32 per SM did not pass
  the complete grasp run. Do not change the production work partition.
- Selected-body angular contact blocks pass 4096 hard/soft contact cases
  bit-for-bit against the full two-body calculation on CPU and CUDA. The
  surface scratch-poison regression also passes. These are correctness
  checks, not proof of a performance gain.
- Testing a direct short-chunk SDF path: for a chunk fitting in one block,
  retain each thread's culling result instead of pushing/popping a queue.
  Longer chunks keep cooperative compaction. Small-mesh deterministic
  contact comparisons pass across separated, touching, penetrating, and
  rotated poses. This experimental factory option is off by default.

- Direct short chunks initially hung in the full scene: skipping the queue
  reset also removed a required block barrier before reusing shared progress.
  Restoring the barrier passed small/large chunk comparisons, but the full
  scene slipped at 21.633 s and the short timing was not faster (37.726 ms).
  Removed this factory option and its probe tests. Removed selected-body
  angular specialization too: correctness passed, speed benefit unproven.
  Shared SDF/contact-force code is unchanged again. Probe logs remain outside
  the repository under `newton_cleanup_archive`.
- Capture the post-IK forward-kinematics update in the existing IK graph;
  retain the identical CPU reachability checks. Standalone IK checks still
  perform their own forward kinematics.
- Diagnostic only: test six local sweeps with component PCG (eight substeps
  and eight linear iterations unchanged). This deliberately relaxes the
  original fixed-work benchmark rule to investigate whether the improved
  preconditioner can reduce work; it is not an accepted default or proof of
  equal accuracy. Production still defaults to eight local sweeps.

## Collision profiling and rejected near-target result

- Six sweeps failed the grasp at 31.800 s (30.1 mm slip). It is not enabled.
- Native convex/convex-only routing, leaving generic mesh and handle routing
  unchanged, reached 34.093609 ms / 29.331011 FPS in a complete timing run,
  but lifted only two grains and retained one. This FAILS acceptance. A second
  diagnostic lifted twenty grains by 12 s; a subsequent complete run slipped
  at 9.667 s. Removed the experimental routing option; it is not a proven
  equivalent replacement for the SDF contact manifold.
- Nsight Systems 2024.6 cannot profile this installed CUDA 13.3 driver:
  diagnostics report unsupported driver and CUPTI_ERROR_INVALID_DEVICE.
  Empty Nsight traces are not evidence of absent GPU work.
- An external CUDA-event probe recorded the empty-self-contact branch and
  checked its actual predicate at every substep (960 checks, no fallback
  requested). It measured collision 14.5273 ms/frame, including face search
  4.7926 ms; rigid sweeps 4.2808 ms; particles including coarse 10.6831 ms;
  coarse including PCG 7.8399 ms; PCG including Ritz 5.6775 ms. Event-instrumented
  runs are diagnostics, not the end-to-end performance acceptance.
- Table-clearance FK and conservative broad phase are captured together.
  The exact CPU triangle/table SAT and rejection are retained. Regression
  alternates safe/penetrating commands on CPU and CUDA, including device-side
  command copies. Explicit kernel devices fix the CPU fallback; test passes.
- Frozen face-search launch scan: 1171 candidates, identical contact counts;
  32 threads / 64 workers per SM took 0.478659 ms per query pass, 64 threads /
  64 workers per SM took 0.455992 ms. This small microbenchmark difference
  does not meet the target and has not changed the default block size.
- Next exact-work experiment: four-lane, two-level speculative golden search
  instead of eight-lane, three-level search. The serial golden-search samples,
  iteration budget, and acceptance predicates must remain unchanged.
- Four-lane search passed exact-search and cached-contact tests, and delivered
  30/32 grains with all five fingers in contact. Complete timing was
  40.673952 ms / 24.585760 FPS: no gain. Reverted the four-lane change.
- Next numerical-budget diagnostic uses six substeps and eight local sweeps,
  leaving production defaults at eight/eight. Unlike the scheduling changes,
  this changes time discretization and needs physical-quality and contact-depth
  validation. It is not to be described as a bit-exact kernel optimization.
- Six substeps / eight sweeps completed at 32.765009 ms / 30.520364 FPS:
  35/38 grains retained, five-finger fraction 1.0, cup RMS deformation 2.798 mm.
  Increasing to nine sweeps did not establish robustness: that run slipped at
  28.183 s. No nine-sweep default is retained.
- A separate six/eight read-only depth audit passed (23/28 retained, all five
  fingers, cup RMS 2.581 mm). Across 30-frame samples the maximum hand/cup
  geometric overlap against the emitted contact tangent planes was zero;
  maximum penalty-radius overlap was 0.06457 mm. This is a sampled contact-row
  diagnostic, NOT an exhaustive mesh-intersection or CCD proof. Maximum sampled
  grain speed was 4.8524 m/s. An eight-substep reference audit is next.
- Eight-substep depth reference also passed (31/33 retained, five-finger
  fraction 1.0). Sampled geometric overlap against contact planes was zero,
  maximum penalty overlap 0.06506 mm and maximum sampled grain speed 4.8271 m/s.
  Cup RMS was 2.370 mm versus 2.581 mm at six substeps. Maximum plastic hinge
  angle was 0.160 versus 0.430 rad: plastic response is measurably different,
  despite similar overall deformation and no increased sampled contact depth.
  Do not claim identical deformation or temporal accuracy.
- Set the pending demo default to six substeps, leaving the solver defaults
  and all material/contact settings unchanged. The ordinary 320-grain default
  is undergoing another complete step-plus-render timing with screenshots.

## Final pending default

- 320 independently simulated massive grains; six substeps, eight local sweeps,
  eight coarse PCG iterations. No grain freezing, attachment, teleportation,
  disabled contacts, smaller contact capacities, or relaxed failure thresholds.
- Ordinary-default complete run `popcorn320_final6.log`: **31.980964 ms/frame,
  31.268601 FPS**, including 1080p rendering and screenshot writes. The earlier
  same-budget run measured **30.520364 FPS**. Initialization is excluded in both.
- Final run lifted 18 grains and retained 15 (83.3%); five-finger fraction 1.0,
  cup RMS 2.555 mm, final rim radius 29.537 mm, upright cosine 0.9876. All existing
  transfer/grasp/clearance assertions passed. Different runs have different
  grain trajectories; this does not establish deterministic replay.
- Six substeps trade time resolution for speed. Sampled contact-depth audits
  above passed, but plastic hinge rotations differ from eight substeps. The
  change must not be advertised as maintaining identical temporal accuracy.
- Retained solver changes: cached previous hinge angles, fused color updates
  and DAT commit with separate read/write positions, conservative empty-contact
  BVH reuse, SDF interpolation-cell reuse, and independent PCG step lengths for
  the particle-connected domain versus detached rigid islands.
- Retained controller changes: batched immutable state downloads and wrist
  targets, device-side IK retry backup, captured FK/table broad phase. Tactile
  preload/gain changes use real contact forces and bounded finger commands;
  they are physical controller tuning, not pure solver scheduling.
- Shared geometry routing and SDF/contact kernels from rejected experiments
  were restored. Other demos' substep/material defaults are untouched. New
  solver scheduling stays behind the existing CUDA-fast eligibility gates.
- Experiment scripts, full logs and screenshots reside outside the repository
  at `E:/csy_work/CG/Engine/newton_cleanup_archive/`; final screenshots are in
  `popcorn320_final6/`. Preserve the pre-existing user change in `sdf_texture.py`.
- No staging, commit, or push performed.
- Final targeted regression bundle: **16 tests passed** (`popcorn320_final_tests.log`),
  covering search/cache equivalence, fused-surface scratch poisoning and reference
  comparison, self-contact certificates, component PCG, cached bending, controller
  snapshots/IK retries, table-guard replay, diagnostics and doubled spawn layout.
  This is targeted coverage, not a claim that every repository demo was rerun.

Test with:

  ```powershell
  uv run --no-sync python -m newton.examples mjvbd_v2_popcorn
  ```

  Eight-substep comparison:

  ```powershell
  uv run --no-sync python -m newton.examples mjvbd_v2_popcorn --substeps 8
  ```
