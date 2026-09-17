# W1 conveyor motion validation

The working-tree version passes a complete four-parcel sequence with the full
MJVBDV2 kinematic VBD backend, 8 substeps, 16 sweeps, and 24 IK iterations.
Sorting completes at 89.68 simulation seconds; the recorded test continues to
91.68 seconds (5,501 physical frames). This version is ready for visual acceptance.

## Motion changes

Joint position and velocity limits are read directly from the source URDF.
Both arms retain a three-degree position-limit margin. Every requested joint
endpoint is executed over enough physical frames to stay below 90% of the
configured speed limit; controller time stretches while physics retains its
60 Hz frame rate and original substeps.

URDF finger velocities are zero. The user has no hardware finger speed
specification; `--finger-speed 90` is a configurable demo fallback in
degrees/second, not a measured hardware limit.

The hand retains its release orientation during withdrawal. Reorientation
finishes at the empty-hand feed stage before another approach can begin, even
if the next parcel arrives early. The cloth-to-bag transition also restores a
known home IK reference seed; executed joints still pass through the retimer.
A regression test fails when the early-arrival transition guard is removed.
Coordinated ankle/knee/hip offsets lower the shoulders for the hanging cloth
while preserving their summed pitch. Waist turns assist cloth and bag placement.

## Contact and workstation changes

The original flat belt collision boxes extended beyond the visible end curve.
Flat sections now stop at the tangent points, with rotating cylinder colliders
at both curved ends. The Blender worktop has a pickup notch matching its
collision geometry. Receiving trays are arranged closer to the robot, and the
soft block and bag are released higher to give opening fingers rim clearance.

Cloth sampling changes from 10 mm to 5 mm with unchanged total mass. The hand
enters below the hanging edge and pinches a cloth node between thumb and index.
The calibrated grasp uses thumb flexion 0.61 rad, index flexion 1.02 rad,
index PIP 0.85 rad, and thumb opposition 0.74 rad. A higher lift and transfer
clear the intervening trays. The bag grasp is raised 6 mm to clear the belt;
its horizontal TCP offset is remeasured after deformation during lift.
Thumb opposition remains fixed while opening the cloth and bag grasps.

Full cloth, soft-solid, gas-pressure, and contact dynamics remain enabled.
There are no grasp attachments or frozen grasp particles. The existing
scene-local cloth displacement damping remains active after verified placement
and hand withdrawal. It explains the zero final cloth speed reported below.

## Full-sequence results

| Metric | Result |
| --- | ---: |
| Maximum actual joint speed / configured limit | 0.899569 |
| Minimum arm position-limit margin | 2.999996 degrees (float precision) |
| Worst sampled robot separation | -0.598 mm |
| Last-second mean cloth RMS speed | 0 mm/s |
| Final inflatable / initial volume | 0.981140 |

The clearance tolerance is 1 mm; the negative separation above is a small
penetration within that tolerance, not positive clearance everywhere.

| Parcel | Maximum center lift | Final center X, Y, Z (m) | Placement |
| --- | ---: | --- | --- |
| Rigid | 0.23943 m | 0.59317, -0.57314, 0.71200 | Passed |
| Soft | 0.25149 m | 0.32495, -0.37824, 0.71688 | 100% particles inside tray XY bounds |
| Cloth | 0.28521 m | 0.08351, -0.41282, 0.69370 | 100% particles inside tray XY bounds |
| Inflatable | 0.23122 m | 0.33372, -0.55047, 0.71315 | 100% particles inside tray XY bounds |

Actual right wrist `RIGHT_J6` peak speed is 44.927 deg/s against its source
50 deg/s limit. The prior committed demo audit recorded 579.430 deg/s. This
compares complete demo versions, rather than isolating one algorithm change.

Ten unit tests pass for timing, source limits, conveyor geometry, clearance,
gaze, torso profiles, and early-arrival sequencing. Changed-file pre-commit
checks and `git diff --check` pass.

Local evidence under `newton/tests/outputs/conveyor_motion_20260911/`:

- `validation_transition/preview.mp4`: full 1280×960 recording, 15 fps.
- `validation_transition/metrics.json`: final numerical results.
- `validation_transition/trajectory.npz`: actual joint positions, source speed
  limits, coordinate mapping, timestamps, and final body/particle states.
- `validation_transition/last.png`, `cloth_carry.png`, `bag_release.png`:
  inspected final and intermediate frames.
- `validation_transition.log`: complete passing phase and validation log.

These local artifacts are ignored by Git. Earlier focused and failed-run logs
remain in the same directory for diagnosis; only `validation_transition`
provides the final full-sequence result.

## Reproduce

From the repository root, run the interactive example:

```bash
uv run --offline --no-sync -m newton.examples mjvbd_v2_conveyor_sorting --num-frames 9000
```

To run the example's assertions, add `--test --viewer null`. Focused tests:

```bash
uv run --offline --no-sync -m unittest \
  newton.tests.test_mjvbd_v2_conveyor_motion \
  newton.tests.test_mjvbd_v2_conveyor_clearance \
  newton.tests.test_mjvbd_v2_conveyor_waist \
  newton.tests.test_mjvbd_v2_conveyor_gaze
```

## Scope of validation

The robot remains kinematic. Joint velocity and position checks do not validate
arm acceleration, jerk, torque, or real hardware finger limits. Cartesian IK
residuals remain during some constrained placement poses; successful sorting
is checked using actual object positions, not assumed perfect TCP tracking.

The independent clearance query includes arm/torso, nonadjacent arm, opposing
arms, and arm/workstation geometry even when simulation filters disable those
pairs. Same-hand finger pairs, adjacent links, and intentional parcel contacts
are excluded. This is sampled-frame checking, not continuous collision
detection or hardware certification.

## Workstation layout

Tray centers and half extents use meters in the scene world frame. Blender
geometry and simulation primitives use the same values.

| Parcel | Center X, Y | Half extents X, Y | Nominal drop-center Z |
| --- | --- | --- | --- |
| Rigid | 0.65, -0.60 | 0.125, 0.125 | 0.76 |
| Soft | 0.28, -0.34 | 0.095, 0.125 | 0.81 |
| Cloth | 0.04, -0.48 | 0.125, 0.20 | 0.76 |
| Inflatable | 0.34, -0.60 | 0.125, 0.125 | 0.81 |

Drop-center Z is a controller target before release. Contact dynamics and
object deformation determine final resting height.
