# Blender popcorn asset integration

## Scope

Rebuild the commercial warmer, round-handled metal scoop and light paper cup
through the running Blender MCP addon (localhost:9876). Preserve the user's
existing Blender scene and the staged simulation baseline. No solver-core,
robot geometry, attachment, prop pose override or artificial grain velocity
clamp is added.

Assets live in `newton/examples/mjvbdv2/assets/popcorn`. The dedicated Blender
scene contains 60 authored prop parts. Blender split normals and material
roughness/metallicity are exported and used by the Newton viewer. Glass in GL
is alpha-blended rather than refractive. Manufacturing details are visual;
the established pan, handle and enclosure collision envelopes are retained.
The paper display keeps the physical shell's 841 vertices/1640 faces and uses
the current particle state, not a static Blender mesh.

## Increased pile: discovered failure and fix

Increasing 128 to 192 grains by simply extending the original packing creates
initial overlap with the held scoop. Two initial runs missed the pile after
the pan settled at a different attitude. The geometry regression reproduces
the old packing failure at grain 152. Extra layers now start behind the pan;
all 192 grains participate in dynamics and collisions.

The robot's tray-height compensation now uses the observed pan orientation
when available, rather than assuming that the finite-contact grasp follows
the commanded angle exactly. This changes a robot wrist target, never the
dynamic scoop state. Floor-clearance tests cover positive and negative slip.

The render-only path avoids adding display meshes to the physical model.
Neither display normals nor material changes alter contact stiffness/mass.

## Validation status

- Asset topology, split normals, pan correspondence, initial pile separation
  and observed floor clearance tests pass.
- Paper-shell plus asset suite: 54 tests pass at this checkpoint.
- The 192-grain original-paper run completed 40 s with 14 retained, but its
  initial lifted set was 25 (not the 15 remaining during transport). Retention
  was 56%, below the unchanged 70% criterion: **No-Go**. Five-finger contact
  fraction was 100%, cup RMS distortion 0.530 mm, shaft slip 0.455 mm.
- Bending stiffness 2.5 / yield angle 0.2 rad completed 40 s, but retained only
  11/22 (50%) and showed zero plastic rotation. RMS distortion was 0.578 mm.
  **No-Go**: no meaningful permanent crease improvement.
- Earlier tip-back (10.5--11.25 s) with that material slipped the cup at 28 s
  (33.5 mm). **No-Go**, original timing restored.
- Retained default: 160 grains (+25%), original bending stiffness 5,
  experimental hinge yield 0.04 rad, original carry timing. **Pass**, full 40 s:
  23 lifted, 17 retained (73.91%); 100% five-finger coverage; final shaft slip
  0.673 mm; cup upright cosine 0.9952; minimum rim radius 32.36 mm; maximum cup
  RMS distortion 0.498 mm; maximum plastic hinge offset 0.06869 rad (3.94 deg).
  No tool IK recovery pause. Sampled hand/cup crossing and interior checks were
  empty. The deformation is light, not the reference photograph's deep crush.
  These are one full-run results, not a guarantee for all contact trajectories.
- No acceptance threshold has been relaxed. The Blender preview alone is not
  evidence of physical grasp/deformation. GPU timing was not benchmarked.

Diagnostic logs/images remain outside the repository in
`E:/csy_work/CG/Engine/newton_cleanup_archive/popcorn-20260911-200805/`, with
`blender-clear-spawn`, `blender-paper-crease`, `blender-carry-early` and
`blender-160-paper` prefixes. Runs overlapping on
the GPU are correctness checks, not comparable performance measurements.

## R2: continuous scoop sheet and larger cup deformation

The Blender MCP-authored R2 scene preserves R1. A single shared-vertex thin
sheet replaces the twelve separately rendered pan panels. Its surface vertices
match the original collision envelope; a regression checks that correspondence.
The original panels remain in the export as hidden references. Satin aluminum,
a matte round grip, an outside shank tab, flush rivets and an end plug improve
manufacturing detail without changing the physical scoop or its inertia.
The cup adds a subtle printed lap-seam band, rendered on the actual particles.
No dent is baked into the geometry and no grasp attachment is added.

Material trials keep 160 dynamic grains and the original final acceptance:

- Membrane 100000, bending 2, yield 0.02 rad, original finger travel:
  full 40 s, RMS distortion 0.997 mm, plastic hinge offset 0.08973 rad,
  14/21 retained (66.7%), five-finger coverage 78.9%: **No-Go**.
  The little finger reached its 12-degree compensation limit.
- Same material, ring/little compensation limits increased to 16/20 degrees:
  full 40 s, RMS distortion 0.975 mm, plastic offset 0.06355 rad,
  14/17 retained (82.35%), five-finger coverage 100%: **Pass**.
  Final shaft slip 0.707 mm. Force targets, finger-speed limits and URDF
  joint limits are unchanged. Only available force-controlled closing travel
  is increased; dynamic props are never repositioned to fake a grasp.

The existing controller already tracks the observed cup rim from 15 s.
An accidentally duplicated rim-tracking block was removed: after 16.5 s its
contribution was multiplied by zero by the original tracking blend. It is not
part of the retained implementation.

R2 logs use `paper-larger-dent`, `popcorn-r2-grip` and
`popcorn-r2-strong-crease` prefixes in the external archive above.

### Retained R2 result

Membrane 50000, bending 1, yield 0.015 rad, hardening unchanged at 2;
ring/little compensation limits 16/20 degrees. Full 40 s **Pass**:

- 21 lifted, 17 retained (80.95%); 100% five-finger contact coverage.
- Maximum rigid-motion-removed RMS deformation 2.2754 mm, versus R1 0.4984 mm
  (4.57 times). This includes elastic deformation; unloaded residual geometry
  was not measured. Maximum accumulated plastic hinge offset 0.17225 rad
  (9.87 degrees) confirms yielding, not a prebaked dent.
- Final cup lift 109.78 mm; minimum rim radius 30.19 mm; upright cosine 0.9904.
- Final shaft slip 0.715 mm; maximum IK position error 0.370 mm;
  no tool recovery pause, no runtime/final assertion failure.
- Sampled hand/cup crossing and interior checks were empty. Sampling is not
  proof of continuous non-intersection.
- 55 paper-shell/asset tests pass; selected-file Ruff checks and formatting
  pass. Feedback regression now allows ring/little travel beyond 12 degrees
  while retaining the original force, rate and saturation assertions.
- No solver-core changes, no staged-index changes, no commit in this revision.

`cup_r2.png` and `cup_r2_top.png` are actual final simulation screenshots.
The asset directory's `scoop_detail.png` is a Blender appearance preview only.
Newton GL and Blender lighting/material rendering differ. No FPS improvement
is claimed; these trials were for visual and physical acceptance.
