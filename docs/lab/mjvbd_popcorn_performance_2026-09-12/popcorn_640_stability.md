# 640-grain stability investigation (2026-09-14)

## Final configuration (2026-09-15)

The final 640-grain configuration passed two complete 48-second validations
at 29.67 and 28.97 FPS (1080p rendering), with 19/20 and 17/18 lifted grains
retained. Use `uv run --no-sync python -m newton.examples mjvbd_v2_popcorn`.
The runtime-only experimental sleep/cache/grouping options are private to
MJVBD V2, off by default for other callers, and are not USD-authored material
properties. Bounds and reset requirements are documented on the options.
The dated investigation below retains rejected trials for reproducibility;
its intermediate status statements are not the final configuration.

## Original investigation status

Initial status: in progress, **not accepted**. The requested target is twice the
accepted 320-grain population without reducing end-to-end performance.
The accepted 320-grain baseline is commit `377979034c`, with 6 substeps,
8 local sweeps and 16 coarse PCG iterations (28.56–28.73 FPS in the recorded
perturbed full-length tests). These are not measurements of the 640-grain work.

## Initial geometry

The old 320-grain layout places 16 hulls through the rear baffle when the
actual hull vertices and 0.8 mm combined contact margin are included.
Measured expanded overlap is 0.892–2.553 mm. This is an initial-condition
defect, not a reason to suppress contact forces or freeze grains.

The experimental bottom-up 640-grain layout avoids the tray walls, floor
and the initial scoop envelope. An actual narrow-phase contact audit found
5,739 initial candidate rows, zero penetrating grain rows, and a minimum
grain contact gap of approximately 0.720 mm. This is a discrete initial
contact audit, not a CCD/no-intersection proof for the subsequent simulation.

## Serial GPU experiments

All tests below use the same experimental 640-grain layout and retain the
cup, robot and existing assertions. Unless noted, they trace the first
11 simulation seconds, not a complete transfer/performance acceptance run.
Full logs and temporary scripts are outside the repository in
`E:/csy_work/CG/Engine/newton_cleanup_archive/`.

| Experiment | Result | Production decision |
| --- | --- | --- |
| New layout, otherwise accepted solver | Early escapes; body 273 became nonfinite at 10.067 s, no nonfinite cup particles | Not accepted |
| Exclude detached rigid bodies from the extra surface coarse correction | Escapes remain; grain speed assertion at 21.9 m/s | Rejected, not installed |
| Grain normal damping 0.02 → 0.2 | Escapes remain, sampled speed reaches 10.6 m/s | Rejected, not installed |
| Float64 arithmetic in the same rigid 6×6 LDLT solve | Escapes remain; sampled speed 12.4 m/s at 10.017 s | Rejected, not installed |
| Route convex-hull pairs through GJK/MPR, retain SDF for other contacts | Contacts reduced substantially, but many more escaped grains | Rejected; shared narrow-phase change reverted |
| Reference 32-lane rigid fusion instead of cooperative 64/128-lane fusion | Escapes remain | Does not isolate the fault to cooperative fusion; not installed |

No sleep, mass changes, force/velocity clamps, contact removal, grain
attachments, weakened assertions, staged changes or commits were made.
An escaping grain alone is not proof of a numerical instability: this is
an open-front tray. Nonfinite states and the existing high-speed failure
must nevertheless be resolved before accepting the larger scene.

## Isolated stationary-tray diagnostics

The reduced diagnostic keeps the grain mesh, density, contact coefficients,
2 mm detection gap and 6-substep/8-sweep budget, but omits the robot, cup,
scoop and global coupled-translation correction. Its floor is near z=0,
so its escaped-grain speeds/timings are not directly comparable to the
full elevated machine. It is a diagnostic, not an acceptance demo.

- A single grain settles to approximately 1e-5 m/s. Its mass is 0.175 g;
  neither zero mass nor an obviously invalid inertia explains the result.
- A 640-grain sloped initial mound avoids immediate spill in the first
  second, but sustained motion and later escapes remain.
- Eight static colors lower late RMS velocity versus two colors, but raise
  isolated step cost to about 19 ms/frame and do not eliminate escapes.
- Twenty local iterations also cost about 19 ms/frame and still do not
  eliminate sustained motion. These timings exclude the robot and rendering.
- Hard-contact mode, a constant 0.5 Newton relaxation, and a fixed material
  penalty did not provide an acceptable replacement. Hard mode additionally
  conflicts with the current soft-contact-only coupled translation operator.
- Disabling friction *only in the isolated diagnostic* makes motion much
  worse; removing friction is not a proposed production fix.

These experiments do not identify a proven single root cause for dense-pile
instability. In particular, they do not establish that sleep would fix it.

## Retained scope

The unaccepted 640-grain default and experimental layouts were removed from
the working demo. Their patch/scripts remain in the external archive.
The demo retains **320 grains and the accepted solver/material budgets**.
Only the side banks are shifted 10 mm forward to remove the known initial
rear-baffle overlap, preserving their internal spacing and heights.

The updated diagnostics test checks actual rotated hull extents plus the
grain/tray contact margins, rather than merely checking center positions.
All five popcorn diagnostic tests pass. Substituting the exact HEAD spawn
function into the new hull test reproduces a rear-baffle failure (expanded
x extent 0.099061 m versus the 0.0975 m inner plane); the retained fix passes.

The complete 48-second 320-grain regression passed all existing assertions:
`popcorn320_baffle_fix_full.log`, 2,850 timed frames after 30 warm-up frames,
1920×1080 OpenGL rendering included, initialization excluded:

- **34.8274 ms/frame, 28.7130 FPS**.
- 33 grains lifted, 29 retained in the cup (87.9%).
- Five-finger contact fraction 1.0; cup RMS deformation 1.884 mm.
- Cached/uncached same-state rendered image difference: zero.
- Final sampled maximum grain speed: 0.0256 m/s. This is not zero motion;
  do not claim the persistent-jitter issue is resolved.

The timing is within the previously recorded 320-grain range, but it is one
full run, not a statistical proof of performance equivalence across runs.
Ruff checks and `git diff --check` pass. Nothing is committed or pushed.
**The requested 640-grain stability and non-regressing performance target
has not been achieved.**

## Follow-up: contact coloring and sleep feasibility (2026-09-14)

The staged 320-grain fix remains unchanged. The following experiments are
external diagnostic scripts in `../newton_cleanup_archive/`, not production
solver options. No sleep implementation is enabled in the demo.

All rows below use the isolated stationary tray, 640 grains, 6 substeps,
8 local sweeps, and 1,200 frames (20 simulated seconds). Times exclude
rendering and robot/cup/tool work, and include periodic diagnostic readbacks.
They are single-run observations, not statistical regression guarantees.

| Experiment | ms/frame | Final escaped grains | Final RMS linear speed (m/s) |
| --- | ---: | ---: | ---: |
| Cached serial dynamic 8-color grouping | 32.1 | 0 | 0.0095 |
| Parallel endpoint preparation, serial 8-color assignment | 28.8 | 1 | 0.0093 |
| Penetration-weighted serial 4-color assignment | 25.2 | 0 | 0.0106 |
| Parallel balanced color swaps | 15.2 | 144 | 0.1930 |
| Parallel swaps with contact-conflict objective rejection | 16.8 | 94 | 0.2495 |

The promising result is weighted serial coloring, not the parallel swaps.
Reducing a graph-conflict objective is insufficient to guarantee nonlinear
contact-solve stability. Neither parallel variant is accepted. A single
zero-escape run does not establish determinism or eliminate persistent jitter.

### Sleep experiment, not a production feature

The isolated prototype builds rigid contact components, requires static
support, and only sleeps a whole component when every dynamic member is
ready. An awake contacting body wakes its component before integration.
Model masses are unchanged; the prototype masks solver-effective inverse
mass while asleep. This alone is insufficient for production: moving
kinematic support, external force, soft contact, reset/model changes and
the coupled translation correction still require explicit integration and
regression tests. It must not be used in the full popcorn scene as-is.

Two rest criteria were tested with dynamic 8-color grouping:

- Continuous low instantaneous mass-normalized kinetic energy for 0.5 s:
  zero sleeping grains at 20 s, approximately 29.5 ms/frame.
- A velocity-accumulation/wake-counter criterion adapted from the
  non-stabilization branch of local PhysX `DySleep.cpp`: four sleeping
  grains at 20 s, approximately 28.7 ms/frame. No PhysX stabilization
  damping or pose freezing was copied. The prototype's reset factor is
  conservative, not a line-for-line PhysX implementation. Its per-substep
  velocity accumulation is timestep-sensitive and needs further analysis.

Both runs had zero escaped grains, but most grains remained awake.
Consequently neither demonstrates material speedup from sleeping. This
does not prove that properly integrated sleep is useless; it shows why
simply adding a timer to this still-moving pile is not an accepted fix.
Do not loosen thresholds merely to report more sleeping bodies.

Lowering grain contact stiffness to 1,000 N/m still left two escaped grains
and 0.0199 m/s RMS motion after 20 s. Increasing grain normal damping to
2.6 N s/m made instability substantially worse. These material experiments
are not retained and do not establish a proven root cause.

### Full-scene 640-grain check

The sloped mound with static spatial 8-color groups and 65,536 rigid-contact
capacity was tested for 48 s including 1920x1080 rendering:
`popcorn640_spatial8_full.log`, 2,850 timed frames after warm-up.

- **49.785 ms/frame, 20.086 FPS**: slower than the accepted 320-grain result.
- 84 grains lifted and 74 retained, versus the accepted run's 33 and 29.
- The final current cup lift fell below the existing 0.08 m requirement.
  The original final assertion failed; it was not weakened.

The 640-grain option/layout was therefore removed from the working demo.
Default count, solver budgets, material values and assertions remain the
staged accepted version. Further work must address both dense-pile solve
cost/stability and the heavier payload's grasp, followed by full-scene
validation. No improvement to the default demo is claimed by this follow-up.

## Continued investigation (2026-09-15, not accepted as default)

The optional 640-grain mound is being tested again; the default remains 320.
The index/staged version is untouched. Scratch runners and raw logs are in
`E:/csy_work/CG/Engine/newton_cleanup_archive/`.

### Independent correction limits

The augmented solve already separates detached rigid bodies from the
particle-connected domain for PCG dot products. Its correction clamp still
used one global minimum, however: a detached grain could reduce the paper
cup's update even without a contact path connecting them. The working fix
uses the contact-island union-find roots for this clamp. Bodies connected to
particles still share their clamp; the non-component fallback preserves
the previous shared limit. No displacement limit is increased.

Detached/connected limit tests pass on CPU and CUDA. The detached regression
was also checked to fail with the original shared-clamp behavior. The whole
coupled-translation module passed five tests. A further optimization omits
the identically zero Ritz preconditioner contribution for detached bodies;
the combined component-PCG and coupled-translation suite passes seven tests.

### Full 640-grain result before final island-clamp refinement

`popcorn640_domainclamp_pcg12.log`: 48 simulated seconds, 2,850 timed frames,
1920x1080 rendering, 6 substeps, 8 local sweeps, 12 PCG iterations.
External shared-memory contact coloring uses four grain groups, with zero
adjacency overflows. The scoop's pile insertion is 5 cm shallower for 640
grains. This run used a two-domain clamp (subsequently refined to actual
contact islands), not the final working implementation.

- All existing final assertions passed: 28 lifted, 21 retained, five-finger
  grasp fraction 1.0, final cup slip 1.287 mm, cup deformation RMS 1.694 mm.
- All 640 grains remained in the machine at 10.5 s.
- **47.166 ms/frame, 21.202 FPS**: performance target is NOT met.
- Final maximum grain speed was 0.0651 m/s; complete settling is not proven.

Four substeps failed the original five-finger lift condition, including a
12-sweep trial. Keep six substeps. Do not weaken the grasp assertions.

### Rejected contact experiments

- Lagging friction load to previous-substep geometry caused 77 escaped
  grains in the isolated pile. Previous-sweep lagging kept the pile inside
  but did not improve RMS motion or speed enough. Neither is retained.
- Rolling/torsional friction prototypes did not demonstrate a convincing
  stability/performance benefit. An exact saturated-spin tangent was unstable;
  an IRLS tangent was stable but not a useful improvement. All prototype
  changes to the shared rigid contact kernel were removed.
- Suppressing detached-island global corrections failed the full scene
  at 13 s with 40.3 m/s grain motion. This is not retained.

An uncaptured diagnostic identifies mesh-SDF contact generation, rigid local
solves, PCG, and repeated surface/DAT kernels as the principal GPU costs.
Its per-node sums are NOT captured-graph frame timings. DAT truncation writes
atomic minima consumed by a following kernel: directly fusing these kernels
would remove a required global synchronization boundary and is not safe.

Latest actual-island clamp plus detached Ritz bypass full run:
`popcorn640_island_fastpcg_full.log` passed 48 s at **45.653 ms/frame,
21.904 FPS**, lifting 30 and retaining 26, five-finger fraction 1.0,
final slip 3.209 mm and cup deformation RMS 1.742 mm. This still does not
meet the performance target. Native convex/convex routing was retried after
this clamp correction; it produced a nonfinite state at 3 s and was removed
again (`popcorn640_island_convex_full.log`). Shared geometry routing remains
unchanged.

The sleep diagnostic's velocity accumulator was changed to integrate actual
elapsed time in 60 Hz reference samples, and its energy scale was checked
against local PhysX `PxRigidDynamic.h` and `PxTolerancesScale.h`: the default
is `5e-5 * speed^2`, with metric default speed 10 m/s, not simply `5e-5`.
This remains an adaptation, not a copied PhysX sleep implementation.
`popcorn_sleep_time_scaled.log` still had only four sleeping bodies at 20 s,
although 630 bodies were individually ready. An active member keeps its
connected island awake; skipping that requirement would not be a valid fix.
There were zero escaped grains, RMS speed 0.01262 m/s, and 19.689 ms/frame
in the isolated diagnostic (not full-scene FPS). No production sleep feature
is claimed from this result.

### Sleep-filter lifecycle correction and wake tests

The prototype missed the filter reset in PhysX `DySleep.cpp::sleepCheck`
when an individual body's counter reaches zero. It must reset even when
other members keep the contact island awake. Otherwise its accumulated
velocity grows indefinitely and repeatedly revokes its ready state.
After this correction, `popcorn_sleep_filter_reset.log` reached 640 sleeping
grains, zero escaped grains, at 14.737 ms/frame in the 20 s isolated run.
No sleep-stabilization damping or additional pose projection was applied.

`popcorn_sleep_wake_inputs.log` applies a downward velocity to one sleeping
grain at 10 s: it wakes, the contacting island wakes, and all 640 grains
subsequently settle and sleep again, without escaping. The prototype also
vetoes sleep for external body forces, soft contact, and contact with a
moving non-sleepable body. Non-free joints are excluded. Reset lifecycle,
moving-support tests and bounded-memory adjacency remain prerequisites for
production integration; this is still external diagnostic code.

Coarse assembly now treats effective-inverse-mass-zero rows as prescribed
boundaries: zero RHS, nonsingular identity diagonal, and no body unknown in
contact cross-blocks. New tests cover inactive rows and point/edge/face
contact against an inactive body. Combined component/translation tests:
eight passed. All original active-body tests remain enabled.

`popcorn640_sleep_full.log`: full 48 s validation passed, 640 grains,
**42.630 ms/frame, 23.458 FPS**, 26 lifted and 25 retained, final slip
2.131 mm. This remains below the target. A fixed-topology stationary-contact
cache diagnostic retains previous contacts only for repeated candidate pairs
with exactly unchanged shape transforms/data/gaps and zero speculative
velocities. Isolated wake/re-settle test passes at 9.538 ms/frame. This cache
is not yet a general pipeline feature: topology/material mutation/reset
invalidation and full-scene acceptance are still required.

Full stationary-cache trial `popcorn640_sleep_cache_full.log` reached
34.896 ms/frame (28.656 FPS), but retained only 19 of 36 lifted grains;
it FAILS the unchanged 70% delivery requirement. A 1.5 cm shallower pickup
for the 640-grain scene, with coloring omitting inactive-body constraints,
passed all checks in `popcorn640_sleep_cache_shallow_full.log`: 17 lifted,
17 retained, zero initial escape at 10.5 s, final slip 4.885 mm. Performance
was 35.746 ms/frame (27.975 FPS), still short of the accepted 320-grain
28.713 FPS reference. Both use 6 substeps, 8 sweeps and 12 PCG iterations;
the default 320-grain pickup trajectory is unchanged.

Independent CPU wake-condition tests pass for static support, moving
support, explicit velocity, external force, removed support and soft contact
(`popcorn_wake_conditions.log`). These kernel checks supplement, rather
than replace, the captured wake/re-settle test and full-scene validation.

### Repository integration and repeatability audit (2026-09-15)

Experimental `rigid_enable_sleep=False` and
`stationary_rigid_contact_cache=False` options now own the sleep/cache
lifecycle in the private solver and pipeline. Existing defaults remain
unchanged. Sleep excludes non-free joints, uses per-world gravity for support,
and restores effective inverse mass on reset; physical model mass is unchanged.
Contact caching requires exactly unchanged shape and body frames, geometry
metadata and margins, with zero speculative velocity. Geometry/material edits
require explicit cache reset. Candidate overflow must not be hidden by cache
compaction. Both features remain experimental, with bounded dense storage.

Contact storage now copies only valid rows in one kernel, including the pose
and geometry snapshots. The sleep contact graph counts unique neighbors while
deduplicating manifold pairs rather than rescanning every body row. Neither
change adjusts contact coefficients, collision geometry, or solver budgets.

Recent full 48 s / 2850 timed-frame results, including 1080p rendering:

| External diagnostic configuration | FPS | Lifted / retained | Result |
| --- | ---: | ---: | --- |
| Core sleep, 4 groups, 8 PCG | 30.080 | 20 / 8 | Reject: delivery |
| Core sleep, 6 groups, 8 PCG | 29.868 | 8 / 6 | Reject: insufficient scoop |
| Core sleep, 6 groups, 12 PCG, 20 mm upstream aim | 28.316 | 15 / 10 | Reject: delivery |
| Integrated cache/sleep, 6 groups, 12 PCG, 27 mm aim | 27.620 | 20 / 19 | Full assertions pass, below performance reference |
| Repeat with unique-neighbor counter | 27.677 | 32 / 22 | Reject: 68.75% retention |

Logs are `popcorn640_core_sleep_pcg8_full.log`,
`popcorn640_core_sleep_color6_full.log`,
`popcorn640_core_sleep_color6_pcg12_full.log`,
`popcorn640_integrated_color6_full.log`, and
`popcorn640_degree_count_full.log` in the external cleanup archive.
These results do NOT establish repeatable acceptance. The balanced contact
grouping still lives in diagnostic wrappers; the plain demo does not yet
enable this complete experimental configuration. It is not a guaranteed
proper graph coloring, and should not be advertised as a general solution.

The alternative mound layout and persistent sleeping-island shortcut were
rejected (escaped grains and an unstable-velocity failure respectively).
The original 640-point mound layout and rebuilding contact islands remain.
The 10.5 s machine count occurs after scoop entry: it is not sufficient to
attribute every escaped grain to initial settling. A separate 5 s check is
being added to diagnostic runs before scoop entry.

CPU/CUDA Graph contact-cache movement/reset checks and existing coarse-solve
tests passed (nine tests before the neighbor-count test was added). After
snapshot fusion, the four then-existing sleep/cache tests passed again.
The staged 320-grain baseline has not been modified or committed.

Further trials:

- `popcorn640_snapshot_color6_full.log`: 28.194 FPS, 18 lifted / 16
  retained, full assertions pass. The 5 s pre-scoop check reports all 640
  grains inside the machine and zero maximum speed.
- `popcorn640_snapshot_color5_full.log`: 28.613 FPS, 24 lifted / 19
  retained, full assertions pass. Same pre-scoop check passes.
- `popcorn320_current_core_reference.log`: 30.673 FPS with the current
  coarse changes and default 320-grain settings, but 19/31 retained fails.
  Do not treat this as an accepted functional baseline or claim regression
  freedom. A committed-core comparison is prepared outside the worktree.
- `popcorn640_native_entrypoint_full.log`: 29.168 FPS, 7/20 retained,
  fails. `popcorn640_private_groups_diagnostics.log`: 28.547 FPS, 7/19
  retained, fails. These repository-entrypoint tests supersede any inference
  that the successful external-wrapper run established readiness.

The experimental five-group scheduler now has solver-owned membership and
color arrays, leaving shared Model coloring unchanged. Its SM80+ path uses
a bounded 48 KiB shared-memory design (5--704 selected free bodies). On
adjacency overflow it retains complete previous membership rather than
discarding physical contact rows. It is opt-in and not exact graph coloring.
The 640-grain demo path enables these experimental features; the default
population remains 320 pending acceptance. No staged content was changed.

Actual-pose diagnostics identified delayed discharge: at simulation 21--22 s,
the scoop is only at approximately 31 degrees while the target is 40 degrees.
15 and then 13 grains remain in the scoop, and **none of those grains nor
the scoop is sleeping**. Withdrawal starts while material is still leaving.
This is not evidence of erroneous sleeping on the moving scoop. A 50-degree
robot target for the 640-grain case is under test, with matching outlet and
withdrawal geometry; no grain pose, force, velocity, or friction is changed.

Eleven core/demo unit tests pass, including private coloring ownership,
overflow fallback, CPU/CUDA Graph contact reuse with latest-contact matching,
and existing popcorn diagnostics. Full-scene repeatability and performance
acceptance remain outstanding.

### September 15: capacity audit and local/global redistribution

The 45/50-degree pour targets and shifted receiving-cup experiments were
rejected after runtime IK failures. The original 40-degree target and cup
location are restored. Instead, the 640-grain robot controller holds its pour
pose while observed material remains in the scoop (bounded to approximately
6 seconds). This is robot sequencing, not a solver acceleration: it changes
neither material velocities nor contact forces. Wait duration is reported.

Rejected performance paths, archived outside the repository:

- Persistent two-CTA component PCG: bitwise-equal frozen-system tests, but
  only 25.97 FPS for the full scene. Removed from production code.
- Existing 32-lane rigid kernel: 26.21 FPS, no improvement; not adopted.
- Surface cache: 27.78 FPS, no improvement; remains disabled in this demo.

GPU counters measured contact occupancy after every substep. A 32768-row
rigid capacity peaked at 30309, leaving insufficient headroom. The selected
capacities are 40960 rigid / 8192 soft; measured peaks across subsequent
trials were below 31k / 3.2k. A retained peak counter and fatal overflow check
are now in the demo; reducing capacity must not silently discard contacts.

The private scheduler reuses membership if the unordered active neighbor
topology has not changed. Contact row order and weight changes alone do not
require recoloring; physical contact rows are still solved. Sleeping bodies
retain their prior adjacency. Overflow retains complete previous membership.

Full 48 simulated seconds, 2850 timed frames after 30 warm-up frames,
1920x1080 rendering, 640 unchanged dynamic grain meshes, six substeps:

| Local sweeps / PCG | FPS | Lifted / retained | Result |
| --- | ---: | ---: | --- |
| 8 / 12, sized buffers | 27.876 | 24 / 21 | Pass |
| 6 / 16 | 28.565 | 22 / 20 | Pass |
| 6 / 14 | 29.416 | 20 / 19 | Pass |
| 6 / 14 repeat | 28.736 | 21 / 19 | Pass |

The last two runs waited 5.25 and 0 seconds respectively. Their contact peaks
were 29855/2784 and 30755/2666. Cached-vs-reference rendering was unchanged.
The repeat had a moving spilled grain late in the run: passing delivery does
not establish that every grain outside the bin has settled. Initial bin
settling and late spilled-grain behavior must not be conflated.

The 6/14 configuration and measured capacities are now the 640-grain demo
defaults, pending direct-entrypoint confirmation. Explicit 320-grain mode
retains eight sweeps and sixteen PCG iterations. No staged content was
replaced and no commit or push was performed.

A committed-core 320-grain comparison passed at 28.710 FPS (35 lifted / 31
retained), whereas the earlier modified-core trial failed retention. This
does not isolate a single cause. As a compatibility precaution, independent
island step limits are now enabled only with experimental rigid sleeping;
ordinary callers retain the established global step limit. A fresh 320-grain
regression run remains required. Do not claim universal regression freedom.

Direct-entrypoint follow-up invalidated acceptance of the 6/14 defaults:
`popcorn640_native_defaults_full.log` ran at 28.767 FPS and retained 29/33,
but the final cup lift fell to approximately 76 mm, below the unchanged
80 mm requirement. The pinky had reached its 20-degree controller offset
limit while producing only 0.475 N versus its 1 N target.

A physical closure-range trial (pinky offset 26 degrees, same final URDF
limits, force targets, rate limits and overload release) passed at 29.075 FPS
with 22/20 lifted/retained. However the native repeat subsequently failed
cup slip at 21.5 s. The pinky had not yet saturated in that second failure;
extra travel alone is therefore not a sufficient fix. Seven local sweeps
are under test. No failed run is counted as meeting the performance target.

Twenty-two targeted tests passed after integration, covering sleep wake-up,
contact topology, cache reset/overflow, PCG, coupled translation and demo
diagnostics. The new pinky-range regression has been added since that run.

The seven-sweep trial passed once at 28.988 FPS (16/14), then failed grasp
at 10.883 s on repeat. Thus adding a sweep did not establish robustness.
Reducing the coarse displacement limit from 0.25 to 0.10 particle radii
tests the hypothesis that the newly independent cup island was taking
over-large steps for its nonlinear contact linearization. This limits the
solver correction, not physical velocity, contact force or material motion.

- Seven sweeps / fourteen PCG / 0.10 radius: 28.213 FPS, 16/14, pass.
- Six sweeps / fourteen PCG / 0.10 radius: 29.671 FPS, 20/19, pass;
  30167/2525 peak contacts, final grain maximum speed 0.00344 m/s.

The latter settings are in the 640-grain default for direct repeat testing.
The explicit 320-grain configuration keeps its original 0.25 radius limit.
Eight demo-level tests, including the new closure-range test, pass.

### Final-candidate repeat (six sweeps / fourteen PCG / 0.10 radius)

`popcorn640_radius01_native_repeat.log` uses the repository defaults with
read-only pose diagnostics and three saved screenshots; no solver/controller
monkeypatch is applied. Full validation passed at **28.972 FPS**:

- All 640 grains were inside the bin at the 5-second pre-scoop check.
- 18 grains lifted, 17 retained (94.44%); all five digits maintained contact.
- Final maximum grain speed 0.000970 m/s; cup rim minimum radius 30.78 mm.
- Contact peaks 30924 rigid / 2410 soft, below 40960 / 8192 capacities.
- Drain wait 0.567 s; no material motion or contact forces are prescribed.
- Cached rendering is pixel-identical to its same-state reference.

Together with the prior 29.671 FPS / 20-to-19 pass, this meets the measured
320-grain reference of 28.71 FPS on this machine. This is two full successful
runs, not a guarantee over arbitrary trajectories or hardware. Raw logs and
15/21/45-second images reside in the sibling `newton_cleanup_archive`
directory, outside the repository. The 320-grain compatibility run and final
regression sweep are still in progress at this checkpoint.

Final checks completed:

- `popcorn320_compatibility_full.log`: full validation passed, 40 lifted /
  34 retained, 28.413 FPS. This is approximately 1.0% below the 28.710 FPS
  committed-core run with a different contact trajectory; it is not proof
  of identical performance for all existing scenes.
- `popcorn_final_regressions.log`: all 23 targeted tests passed.
- Ruff checks and `git diff --check` passed. No experimental persistent PCG,
  warp-width or surface-cache trial was retained in the repository.
- Index remains the original four staged files (123 insertions, 3 deletions).
  No commit, push or staging operation was performed.

Test the new 640-grain default with:

```powershell
uv run --no-sync python -m newton.examples mjvbd_v2_popcorn
```

Use `--popcorn-count 320` for the preserved smaller-scene configuration.
Sleeping, cached stationary rigid contacts and balanced rigid scheduling
remain opt-in solver/pipeline features; other demos do not enable them
automatically. New contact-peak diagnostics detect capacity exhaustion
instead of accepting a clipped contact set. The final two 640-grain runs
passed all original grasp, delivery, table, shape and motion assertions.

## Submission checks on HEAD `894f69ad`

The fork's `FAST_MJVBDV2` tip was fetched and matched the local HEAD before
submission. The 640-grain default passed another full validation at 29.486
FPS, with 23 lifted / 20 retained and contact peaks 30628 / 2758. The robot
used the bounded drain wait for approximately six seconds in this run.
All 23 targeted tests passed again after final cleanup.

An isolated CPU regression using the HEAD version of `initialize_bodies`
failed as expected: a zero-effective-inverse-mass row produced RHS
`[99.99999, 0, 0]` instead of zero. The current implementation passed the
same probe, confirming that the test detects the inactive-body defect.

`uvx pre-commit run -a` was executed. Existing unrelated Ruff violations in
experimental scripts/examples and typos findings on NumPy's `writeable`
attribute prevent a clean repository-wide result. Every file in this commit
passes `pre-commit run --files ...`; unrelated code was not changed. The
final review fits the requested bounded experimental MJVBD V2 scope, not an
unrestricted new default for every VBD solver. No broader regression-free
or exact-trajectory-equivalence claim is made.
