# W1 popcorn demo: retained implementation and experiment record

## Status

Latest (2026-09-12): the coordinated-grasp/two-grain-color version passed two
40-second full-scene runs, including one rendered run of the real demo class.
The runs retained 11/13 and 13/14 lifted grains, respectively. All 98 related
regression tests pass. Physical delivery/retention acceptance has passed in
these runs; this is not a general guarantee of collision-free grasping.
Performance acceptance remains open: the first full no-render run averaged
199.57 ms/frame after warm-up (about 5 FPS, not real time). No target minimum
FPS has yet been specified. The robot remains a prescribed moving boundary.

Earlier failed runs and superseded configurations below are historical
evidence, not claims about the latest run. In particular, one grain color was
rejected, and two colors without coordinated grasp control failed a repeat.

## Retained code

- `newton/examples/mjvbdv2/example_mjvbd_v2_popcorn.py`: original W1 hand
  geometry and joint limits; dynamic cup, scoop, and grains; contact-based
  grasping; measured failure guards. No hidden friction bodies, attachments,
  prescribed prop poses, or post-solve velocity scaling.
- `support/paper_shell_material.py`: experimental elastoplastic bending
  return with elastic membrane response; not calibrated paper material.
- The existing SDF search optimization avoids unused texture gradients while
  preserving value arithmetic, search iterations, and final gradient evaluation.
  `test_soft_contact_search_distance.py` checks search-distance compatibility.
- Both private VBD backends retain `rigid_avbd_contact_beta`: an optional
  body-body penalty ramp independent of joint/body-particle penalties.
  Omitted values retain legacy behavior; `test_mjvbd_v2_contact_penalty.py`
  checks compatibility and parameter validation.
- `test_mjvbd_v2_paper_shell.py`: geometry, plasticity, trajectory continuity,
  force-feedback bounds, original joint limits, and overhand-orientation checks.

## Current coordinated workstation

- Right-hand roll: -135 degrees, palm above the unchanged cylindrical shaft.
- Handle position: `(0.70, -0.28, 1.03)` m; station yaw: 0 degrees.
  The machine's complete collision geometry moves with the station frame.
- Left hand transports the cup to `(0.70, 0.0, CUP.z + 0.11)` m.
- Pour yaw: 60 degrees. Withdraw laterally 12 cm and upward 2 cm before
  leveling, instead of the old high upward arc.
- Keep the existing robot base and leg posture after testing alternatives.
  A lower-elbow seed did not preserve the full trajectory and was rejected.
- Right thumb joint: 20.5 degrees rather than 11.8035 degrees. The original
  thumb missed the shaft by 4.241 mm; the new sampled surface gap is -0.120 mm,
  with 29.912 mm clearance from the metal neck. The little finger still has
  a 5.479 mm gap: this is **not** a verified five-finger right-hand grasp.
- Four-finger extra PIP squeeze remains 0.8 degrees. Default contact stiffness,
  friction, and pressure-feedback gains are unchanged by the rejected trials.

## Evidence and rejected trials

| Retained evidence | Result |
| --- | --- |
| `overhand-coordinated-preflight.log` | 201 nominal path samples over 25 s; maximum wrist position error 1.311 mm; original 4 mm / 5 degree preflight limits pass. |
| `overhand-thumb-tests.log` | 34 targeted tests pass; not a physical-delivery acceptance. |
| `overhand-thumb-rest-audit.log` | Sampled original hand/shaft/neck geometry checks; not a continuous collision guarantee. |
| `overhand-coordinated.log` | Zero payload collected; shaft slips 10.4 mm at 29.817 s. |
| `overhand-squeeze2.log` | Raising extra finger closure to 2 degrees loses the shaft at 15.000 s; rejected. |
| `overhand-thumb.log` and `.npz` | At 14.000 s, `popcorn_32` (body 73) has nonfinite pose/velocity; other bodies and all particles remain finite in the snapshot. Cause not established. |
| `overhand-thumb-majorized.log` | Experimental same-color Hessian majorizer loses five-finger lift contact at frame 785; not adopted. |
| `overhand-thumb-soft-grip.log` | Majorizer plus left-hand ke=40000 loses lift contact at frame 796; not adopted. |
| `overhand-thumb-slow-release.log` | Majorizer plus opening gain 0.01 reaches a 6.0 mm right-wrist error at 11.350 s; not adopted. |

The nominal IK path does not include the full live tool-attitude correction
excursion. Further planning needs workspace/joint-limit reserve for this
feedback, not a relaxation of the 5 mm runtime guard. Diagnostic wall timings
include readbacks and rendering and must not be reported as a GPU benchmark.

## Reproduction

Run from the repository root:

```powershell
uv run --no-sync python -m newton.examples mjvbd_v2_popcorn
uv run --no-sync python -m newton.examples mjvbd_v2_popcorn --viewer null --num-frames 2400 --test
uv run --no-sync python -m unittest newton.tests.test_mjvbd_v2_paper_shell newton.tests.test_mjvbd_v2_contact_penalty newton.tests.test_soft_contact_search_distance
```

The full-scene command is expected to expose the unresolved failures; do not
remove guards to label it a pass. No CPU full-delivery acceptance is claimed.

## Cleanup (2026-09-11)

Retain implementation, regression tests, this summary, nine evidence logs,
and one nonfinite-state snapshot. Remove the unused parameter-scan scripts,
monkey-patched solver experiments, old images, repeated run dumps, and caches
from the worktree. No production solver behavior changes in this cleanup.

The complete original 2000-file lab directory, including its detailed history,
is recoverable outside the repository at:
`E:\csy_work\CG\Engine\newton_cleanup_archive\popcorn-20260911-200805\lab`.
Nothing is staged, committed, or pushed by this cleanup.

Post-cleanup verification: all 38 tests across the three retained test modules
pass; targeted Ruff checks and `git diff --check` pass. The cleaned lab folder
contains 11 files (approximately 0.6 MB, down from 314.6 MB).

## Runtime IK and empty-scoop investigation (2026-09-11)

The reported 10.733 s wrist failure is not a reason to relax the 5 mm guard.
The nominal path omitted the accumulated live tool-attitude feedback. Change
the overhand roll from -135 to -125 degrees for more planning reserve, and
backtrack only the current, unexecuted feedback increment when IK rejects it.
Retain the previously accepted correction and robot state; do not unwind the
whole correction abruptly, move the prop, or suppress an unreachable target.
Update the grasp-wait clock before generating wrist targets.

The empty-scoop run also showed a nose-up blade under load. Add a 5-degree
entry pitch before entering the pile, compensating handle height using the
actual panel vertices so the lowest blade clearance is unchanged. This is a
robot trajectory change, not a prescribed scoop pose.

Physical experiments (128 grains, 24 sweeps, 8 substeps):

- Roll/feedback change alone: zero payload, shaft-slip guard at 15.100 s.
- Entry pitch, original contact solve: one grain in the scoop at 14 s, then
  a 56 m/s grain ejection near 15 s. Not a successful scoop or delivery.
- Entry pitch plus archived same-color contact-Hessian majorizer: 19 grains
  physically lifted, then only 2 retained when the cup-slip guard stopped the
  run at 19.150 s. This experimental solver path is **not** the demo default.

These experiments used the earlier whole-correction feedback backtrack;
they do not validate the final, safer increment-only implementation. The
full task remains unaccepted. The majorizer is not adopted on the strength
of an isolated successful scoop: cup retention and full delivery still fail.
Evidence is outside the worktree alongside the cleanup archive, under the
`feedback-reserve`, `entry-pitch`, and `entry-pitch-majorized` prefixes.

Current regression coverage: 42 tests across the three retained modules pass;
Ruff and whitespace checks pass. New tests cover feedback rollback, preserving
the established correction, rejecting an unreachable path, and blade-height
compensation. Unit-test success does not establish physical scene acceptance.
No staging, commit, or push is performed for this investigation.

Final increment-only/default-solver rerun (`increment-feedback.log`, same
external archive directory) stops at physical time 12.783 s / trajectory time
4.370 s: five-finger contact was not maintained during lift. No grain was
scooped. This rerun does not reach the reported IK failure phase and therefore
cannot prove that failure eliminated. Grasp repeatability remains an additional
blocker; do not advertise this working-tree revision as a completed fix.

## Table penetration and grain ejection follow-up

The reported left-arm overlap is with the tabletop, not evidence of wrist
self-collision. The prescribed robot cannot be displaced by table reactions.
Move the real tabletop front edge to x=0.57 m (cup center x=0.62 m), adjusting
its support legs as well. Add a pre-execution arm/finger command screen using
world geometry bounds followed by triangle/box separating-axis tests against
the solid tabletop with 1 mm clearance. Visual and collision meshes participate.
Bounds alone produced false positives on the slanted wrist; add a regression
for that case and for triangles crossing the box with all vertices outside.
This endpoint check is not swept-mesh CCD or general obstacle-avoidance IK.

Integrate the same-color contact-block majorizer into the private full VBD
backend for **soft contacts only**. For simultaneous finite-mass endpoints,
double each diagonal contact Hessian block without changing forces, friction,
mass, or velocities. The bound follows from
`||a+b||_K^2 <= 2||a||_K^2 + 2||b||_K^2` for the positive linearized contact
metric. This is not a proof of nonlinear collision-free trajectories. Static,
kinematic, and different-color pairs keep their previous update. Applying the
same scaling to hard contacts changed an existing five-iteration regression;
the hard-contact AL path is deliberately unchanged.

With the adjusted table and integrated majorizer, the `table-majorizer` run
lifted 12 grains and reached pouring without the former high-speed ejection,
but the cup slipped at 19.667 s. The stronger-preload `table-preload` run
passed the new table screen and lifted the cup, then hit a right-wrist IK
failure at 12.617 s. Neither is a full-scene pass. Evidence remains outside
the repository in the cleanup archive parent directory.

Further trial: double tactile preload to 4 N thumb / 1 N each finger, retain
bounded physical joint actuation, lower the receiving position by 2 cm, and
use a -121 degree overhand roll. The -110 degree trial was rejected because
it did not retain the existing downward-palm orientation criterion; that
criterion is not weakened. Full-scene acceptance remains pending.

The -121 degree / stronger-preload run passed nominal IK and all table screens
through pouring, lifted 17 grains and delivered 7, but still slipped at
18.967 s (35 mm). Five-digit contact coverage was 100%; this rules out simply
missing a digit as the whole explanation. Do not describe the larger preload
as a proven retention fix. A subsequent one-cluster cup-correction trial was
stopped by the conservative left-wrist/table screen at 0.267 s; move the edge
farther out from 0.57 to 0.595 m, preserving support under the cup base, and
repeat. Latest completed regression suite: 93 tests passed.

The 2048-particle/one-cluster trial subsequently produced 18.7 mm cup RMS
deformation during initial settling and a left-wrist tracking failure at
0.350 s. Revert to cluster size 400; do not keep this as a grasp fix. Restore
the table edge to 0.57 m now that triangle/box tests resolve bounds-only false
positives. A 48-sweep run is a convergence diagnostic, not an accepted faster
default. No full-delivery success has been established by these trials.

## Delivery and withdrawal follow-up

Keep the cup elastic in stretch and elastoplastic in bending. Reduce the
particle/shape velocity damping coefficient from 30 to 1; do not alter grain
velocities, attach props, or disable the slip/instability assertions. Add a
finite tactile load deadband after grasp acquisition so that a transient
increase in load does not immediately command the fingers to open. This alone
did not fix retention: 8-substep trials still lost the cup during pouring.

The actual scoop outlet can differ significantly from the commanded outlet
because the utensil is friction-held. Aim using its observed orientation with
a 50 mm correction cap and 30 mm/s target correction limit. Integrate feedback
once per physical frame, not once per IK retry. Withdraw from the observed
utensil location, clear the cup laterally, then level over five seconds.
These operations change robot targets only. A bounded right-wrist recovery
can pause the path clock while reducing attitude compensation by 0.25 degrees
per frame; physical simulation continues and the original IK tolerance stays
unchanged. Recovery has a two-second total budget and still raises on failure.

`official-substeps16.log` delivered and retained all 12 lifted grains through
the drain and early withdrawal. The cup remained held, but the shaft slipped
10 mm at 27.000 s and the original guard stopped the run. This is partial
progress, **not full acceptance**. The next trial gradually closes right
THUMB1 by a further 2 degrees during initial grasp closure (within the
unchanged URDF range), with 16 substeps / 24 sweeps. The default substep count
is raised from 8 to 16 for validation; this is a stability/performance tradeoff,
not a claimed speed improvement.

An isolated free paper cup on a plane receiving 16 initially separated rigid
grains completed two seconds with maximum grain speed 1.961 m/s and final
speed 0.054 m/s (`probe-loaded-cup-valid.log`). No contact overflow occurred.
Earlier overlapping-initialization and overflow trials are excluded from
this conclusion. This isolation test does not establish friction-grasp
stability. All new raw logs and diagnostic scripts remain in the external
`newton_cleanup_archive/popcorn-20260911-200805` directory.

The subsequent `official-thumb-preload.log` run delivered all 13 lifted
grains, but again lost the shaft at 27.100 s. Its contact diagnostics show
`machine_side_glass` immediately before loss: the old return-to-HANDLE path
levels the long bowl through the side glass. More grip force is not an
adequate fix for an obstructed path. Move the final withdrawal handle target
16 cm toward the robot, preserving the lateral cup-clearance phase. A new
test samples all pan vertices along the nominal withdrawal, requiring 10 mm
clearance in front of the glass. This test would reject the former endpoint.
The revised full scene is being verified separately (`clear-withdraw.log`).

### Completed 48-second rendering/physics validation

`clear-withdraw.log` completed 2880 frames and the unchanged `test_final`
assertions with 128 grains, 16 substeps and 24 sweeps. No traceback, contact
overflow, table-screen failure, or withdrawal obstacle contact was logged.
Maximum **logged** grain speed was 1.51 m/s (sampled, not a continuous bound).
Final results: 16 lifted and 16 retained, 59 retention checkpoints, five-finger
contact fraction 1.0, cup maximum distortion RMS 0.480 mm, upright cosine
0.99205, minimum rim radius 32.31 mm, and shaft slip 0.630 mm. Maximum accepted
IK position error was 0.871 mm; no tool recovery pause was needed. Final cup,
grip, overall and intermediate rendered images were inspected from the
external archive. The run includes about 20 seconds of post-withdrawal hold.

All 95 tests passed across `test_mjvbd_v2`, `test_mjvbd_v2_paper_shell`,
`test_mjvbd_v2_contact_majorizer`, `test_mjvbd_v2_contact_penalty`, and
`test_soft_contact_search_distance`. These tests do not prove all other demos
unchanged or sweep-level collision freedom. Standard null-viewer repeat is
recorded separately in `clear-withdraw-repeat.log`; no clean FPS comparison
is claimed while diagnostic runs overlap on the GPU. Staged user work remains
untouched; these follow-up edits have not been staged or committed.

### Standard null-viewer repeat

The exact current default configuration also completed
`uv run --no-sync python -m newton.examples mjvbd_v2_popcorn --viewer null
--num-frames 2400 --test` with the unchanged final assertions. At 40 seconds:
12 lifted, 11 retained (91.67%), 39 retention checkpoints, five-finger coverage
1.0, maximum cup distortion RMS 0.513 mm, upright cosine 0.99537, rim radius
32.36 mm, final shaft slip 0.668 mm, maximum accepted IK error 0.326 mm,
and zero tool-recovery wait. The different payload count illustrates that
frictional contact results are not bitwise deterministic. Both current-code
runs passed, but this is not a guarantee over arbitrary robot trajectories.
The cup remains a free deformable shell; neither prop is attached to the hand.
Performance remains unbenchmarked and 16 substeps cost more than the former 8.

## Follow-up: performance acceptance is still open

The two completed deliveries do not by themselves establish the requested
interactive performance. Profile the current default rather than claiming
the whole task complete. In a 180-frame cProfile sample after 60 warm-up
frames (construction excluded), baseline wall time was 373.35 ms/frame and
the mesh/table screen took 68.23 ms/frame. Cache each mesh's local bounds,
rotate the bounds with `abs(R) * half_extent`, and only transform full meshes
whose conservative bounds overlap the table. Preserve the old exact test,
with outward rounding on the broad-phase bounds. After this change the same
profile window measured 314.51 ms/frame and the screen took 13.31 ms/frame.
These numbers include profiler overhead. A 100-pose randomized regression
compares the optimized decision to the full-mesh triangle/box decision.

An isolated physics Graph launch measured about 209 ms. Instrumented
uncaptured per-kernel timings identify the nine sequential rigid color groups
as a large cost (one utensil plus eight initial-packing grain colors). Trial
four grain colors outside production, retaining 16 substeps, 24 sweeps and
all acceptance assertions. This changes simultaneous contact updates and is
not an equivalent-code optimization; do not enable it without full delivery
and retention testing. Diagnostic scripts/results stay in the external archive.

The four-color trial completed 40 seconds with 14 lifted, 12 retained
(85.71%), five-finger coverage 1.0, maximum distortion RMS 0.581 mm,
upright cosine 0.99414 and shaft slip 0.576 mm. No recovery wait was needed.
Its full post-warm-up wall average was 249.33 ms/frame including normal test
measurements. A matched short-window comparison without cProfile, run
sequentially, measured 310.36 ms/frame for eight grain colors and 242.31
ms/frame for four (21.9% lower time). Keep four colors in the demo; this is
not a claim of equal iterates or universal speedups. Contact filtering and
the number of substeps/sweeps are unchanged. A separate small-thread-block
experiment is still outside production, pending timing and regression tests.

The 32-thread body/body accumulation launch measured 230.98 ms/frame versus
242.31 ms/frame for default blocks in the same short window. Do not integrate
this single-run 4.7% result into the shared solver without repeat measurements
and wider scene testing. No production block-size change was made. A two-color
grain trial is evaluated separately; until it passes, four colors remain the
validated demo configuration. No stage/commit/push is performed.

Two grain colors also completed 40 seconds: 17 lifted / 14 retained (82.35%),
five-finger coverage 99.75%, maximum cup distortion RMS 0.773 mm, upright
cosine 0.99253 and shaft slip 0.626 mm. Full post-warm-up average was 205.76
ms/frame, with the original assertions intact. Try a final single grain group
with a separate utensil group; simultaneous soft-pair majorization remains
essential. These averages exclude setup, include normal test measurements,
and do not include rendering. No real-time FPS claim is made.

Single-grain-group trial is **No-Go**: despite delivering 14 grains, the cup
slipped 35.9 mm at 29.950 s during/after withdrawal. The original guard raised;
do not remove it or keep the faster grouping. Select two grain groups and
repeat through the real demo class with rendered diagnostics before acceptance.

The two-color rendered repeat also failed: at 24.983 s the cup slipped
30.7 mm, after all 13 carried grains were delivered. Therefore withdraw the
two-color acceptance claim and restore the production eight-color grouping
while diagnosing repeatability. The table broad-phase optimization remains.
Immediately before failure, ring/pinky forces fell to 0.15/0.10 N while index
load rose to about 5 N. The independent load controller commanded the index
to open as the other digits lost contact, aggravating the roll/slip.

Add a regression for this load-transfer case: it fails on the old controller
(index update -0.000833 rad despite weak opposing contacts). When any digit
has less than half its preload, keep supporting digits closed unless their
force exceeds a finite emergency ceiling (16 N thumb / 8 N other digits).
Weak digits still close under the existing speed and URDF travel limits;
normal overload release resumes once opposition recovers. This changes only
robot commands, never cup/tool poses, velocities, or friction. Validate the
coordinated controller in an external two-color full-scene trial before
enabling that faster grouping again.

The first coordinated-grasp trial stopped before pouring at 18.083 s:
right-wrist error 5.1 mm, then 5.4 mm after the old one-shot recovery.
Change recovery to hold the robot's actual joint coordinates if the right
target remains unreachable, while retaining the 0.25-degree-per-frame
feedback unwind and two-second total path-pause budget. Suspend attitude
integration during the hold so it cannot immediately wind back up. Physics
and tactile finger control keep running; no unreachable IK iterate is sent
to the robot and no prop state is overwritten. Left-wrist failures still
raise. Add a regression checking that failed IK coordinates are discarded,
physical state is untouched, and only the path clock pauses.

### Latest coordinated-controller validation

`coordinated-hold-final.log` completed 40 seconds: 13 lifted, 11 retained
(84.62%), five-finger coverage 1.0, maximum cup distortion RMS 0.468 mm,
upright cosine 0.99486, shaft slip 0.683 mm, and 199.57 ms/frame averaged after
the first 60 warm-up frames. No recovery pause was used in that run.

Then select two grain colors in production and repeat through the unchanged
real `Example` class with rendering (`coordinated-production.log`). It
completed 40 seconds: 14 lifted, 13 retained (92.86%), five-finger coverage
1.0, maximum cup distortion RMS 0.400 mm, upright cosine 0.99237, rim radius
32.39 mm and shaft slip 0.640 mm. All original final assertions passed.
The bounded IK recovery was exercised for 0.05 s. Maximum **attempted** IK
error was 7.84 mm, not an executed tracking error: unreachable iterates were
rejected, actual robot coordinates held, and the original 5 mm threshold was
not widened. Final cup and round-handle grip images were inspected. Sampled
hand/cup triangle crossings and interior checks were empty. These checks are
not continuous collision detection. All 98 related tests, Ruff and whitespace
checks pass. No staged changes, commits or pushes were made in this follow-up.

Final matched short-window no-profiler timing (`bench-coordinated-two.log`):
196.38 ms/frame after 60 warm-up frames, versus 310.36 ms/frame for the
earlier eight-color version with the same bounds optimization (36.7% lower
wall time). The later version also includes coordinated grasp control, so
this is an end-to-end configuration comparison, not an isolated attribution
to coloring. Render cost and initialization are excluded. About 5.1 FPS is
still not real time; obtain a concrete minimum FPS target before calling the
overall performance requirement satisfied.
