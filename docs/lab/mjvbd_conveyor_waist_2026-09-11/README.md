# W1 conveyor torso coordination

The waist turns toward the receiving trays during transfer, lowering, release,
and clearance: -30 degrees for cloth and -15 degrees for the inflatable bag.
It returns smoothly while the hand retreats. The torso profile limits speed to
0.45 rad/s and acceleration to 0.8 rad/s². These bounds apply to the waist only.

The right-hand world-space position and orientation objectives remain active.
A separate, unturned IK reference preserves the recorded grasp branch. The
actual solution stays warm-started during manipulation and gradually approaches
the reference during empty-hand feed/settle phases. The left arm follows the
torso in its existing idle posture; head tracking remains torso-relative.

## Validation

- A 4,200-frame / 70-second rendered run passed all existing sorting and final
  checks: all four parcels sorted, cloth last-second mean speed 0 mm/s, and
  inflatable volume/rest-volume ratio 0.9905.
- Three torso/gaze unit tests passed, including waist speed, acceleration,
  return-to-neutral, and tracking relative to a rotated torso.
- Changed-file pre-commit checks passed.
- Local recording and joint traces are under
  `newton/tests/outputs/conveyor_waist_20260911/audit/` (ignored artifacts).

## Hardware reachability remains unresolved

This is a simulated posture improvement, not a hardware-executable trajectory.
The audit found right-wrist limit saturation and a peak finite-difference J6
speed of 579.43 degrees/s, versus 50 degrees/s in the source URDF. The largest
peak occurs during the cloth retreat. Other right-arm joints also exceed their
URDF velocity limits. The imported model reports a default 1e6 rad/s velocity
limit, so the audit reads the source URDF instead of trusting that field.
Robot self-collision is disabled in the existing scene; successful sorting does
not establish arm/torso collision clearance or hardware feasibility.

Hardware-oriented follow-up needs explicit arm speed constraints, posture and
joint-limit margins during IK, collision-aware planning, and trajectory timing.
The current passing simulation must not be reported as validating those items.
