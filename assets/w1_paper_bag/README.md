# W1 paper bag and snack packing

This scene adapts the user-provided video `飞书20260916-115427.mp4`. Around
16–19 seconds, the robot's left hand leads the lift while the other hand is
near the opposite edge and assists. This demo uses a single left-hand rim grip
to turn the bag up. As in the later reference frames, it then releases, clears
the rim, and regrips from the side while the right hand waits clear. The right
hand subsequently packs snacks.

The table is 0.92 m high. Its near edge is at x = 0.36 m, leaving the wrist
and finger tails outside the table during the inclined rim approach. The lower
body starts with ankle, knee, and hip angles of 25, -50, and 25 degrees. This
lowers the torso by about 6.4 cm without leaning it or shifting it horizontally.
The crouch is held throughout the sequence. These are scene choices inspired by
the reference, not measured video dimensions or a balance controller.

The robot is the same W1 V030 URDF, mesh geometry, parallel grippers, and wrist
cameras used by `mjvbd_v2_w1_pick_place`. Scene-local display colors approximate
the video's yellow shell and black joints. Finger visual assets have repaired
face winding and normals; their geometry and separate collision meshes are unchanged.

## Blender assets

`build_blender.py` authors an independent Blender scene and exports:

- `w1_bag_and_snacks.blend`: editable bag and snack models, with packed paper grain.
- `bag.npz`: an open, welded paper shell with side gussets, a pressed bottom fold,
  and raised multilayer paper handles extending above the mouth.
- `snacks.npz` and `snacks.json`: colored geometry for a green crisp can and red
  biscuit carton, including rolled metal rims, lettering, and food illustrations.
- `kraft.png`: original procedural cellulose grain.
- `dimensions.json`: estimated dimensions in metres.

The video has no scale calibration. The bag is approximately 320 mm wide,
130 mm deep, and 250 mm high; the can is 65 mm in diameter and 117 mm tall;
the carton is 48 x 60 x 118 mm. Packaging graphics are original approximations,
not extracted brand artwork. The table and the trajectories also approximate
the reference, rather than reconstructing its camera calibration or human motion.

Rebuild in Blender's Python console, substituting the repository path:

```python
from pathlib import Path
p = Path('/home/oem/code/repos/newton/assets/w1_paper_bag/build_blender.py')
scope = {'__name__': 'w1_assets', '__file__': str(p)}
exec(compile(p.read_text(), str(p), 'exec'), scope)
scope['build'](p.parent)
```

This adds a separate scene and writes only that scene's dependencies to the
asset blend file. Blender is required to edit/re-export, not to run the demo.

## Physics and validation

The robot is a fixed-base kinematic moving boundary, with IK limited to its
14 arm joints. The two neck joints separately track the operating hand with
a smoothed yaw/pitch command, a 0.7 rad/s speed cap, and angles inside the
source joint limits. The bag uses dynamic cloth particles, elastic membrane and
bending forces, and a reinforced folded rim. Broad panels use bending stiffness
30 and damping 1.0 (scene units); the authored corner, gusset, and bottom fold
hinges use 8 and 0.3. This lets the existing folds bend while limiting broad
panel dents. The double-ply top hem uses bending stiffness 120 and damping 2.0.
All particles have positive mass. Handles start outside the paper surface,
with 2.6 mm folded thickness and a 1.5 mm rest-contact exclusion distance,
so nearby handle-wall contacts remain active.
Handle arches and their welded connections use bending stiffness 0.3 and
damping 0.01 in both the automatic and teleoperated scenes.
The snacks are free rigid bodies. Only contact and friction stand the bag up
and lift the snacks; there are no grasp attachments, table anchors, or scripted
object poses. Materials are demonstration parameters, not measured paper or
hardware calibration.

The full-contact MJVBDV2 scene explicitly uses rigid-soft DAT and particle-only
multilevel correction. No shared solver code or default is changed by this demo.

```bash
uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing
uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing --viewer null --test
uv run --extra dev -m unittest newton.tests.test_mjvbd_v2_w1_bag_packing
```

### Record once and replay

```bash
uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing --record --test
uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing --replay --loop
```

Recording defaults to the null viewer and saves the initial state plus every
60 Hz simulation frame. The default directory is `newton/tests/outputs/w1_bag_packing`
in this checkout, independent of the working directory. Both `--record DIRECTORY`
and `--replay DIRECTORY` accept another location. Existing directories are never
overwritten; choose a new directory to record another take. A short recording
(`--num-frames 120`, for example) or an interrupted recording remains playable,
but its metadata marks it as incomplete until final packing validation passes.

Replay builds only the scene geometry and materials, then reads saved robot,
snack, and paper poses directly. It does not initialize IK, mesh SDFs, collision
detection, or either physics solver. Recorded scene options, including snack
count and robot setback, are restored automatically; incompatible geometry is
rejected. These are visual recordings, not checkpoints for resuming dynamics.

The replay sidebar offers pause, a frame slider, restart, and looping. Camera
controls remain available. `--start-frame 1440` starts at 24 seconds;
`--unthrottled` removes the default 60 FPS cap for benchmarking. Without looping,
the interactive GL viewer holds the final frame; headless playback exits after
the last frame. Arrays are streamed through memory-mapped files, so recording
and seeking do not require keeping the entire sequence in RAM.

### Simulation options

`--snacks 1` limits packing to the green can. Test mode checks an upward-facing
mouth (up to 30 degrees of tilt), broad-panel mouth bowing (at most 30 mm),
handle-wall triangle intersections, the bottom remaining within 1 cm of the
table while tipping, physical snack pickup, release and full-object
containment in the fitted moving bag frame. The envelope test assumes the bag
retains approximately its rectangular shape. Every test step also checks the
robot's visual and collision mesh triangles against the finite tabletop volume,
including a 3 mm clearance above its surface. This catches wrist and forearm
intersections even when the tool-center point is above the table.

The default sequence lasts 50 simulated seconds (3,000 frames). For a shorter
one-snack run, use `--snacks 1 --num-frames 2280`. Joint commands keep all 14 arm joints at least 10 degrees inside their
source position limits and cap arm speed at 4 rad/s; coordinated IK updates
are scaled together to avoid sudden wrist flips. The path is approximate and
allows up to 4 cm of transient TCP tracking lag during reorientation. Released
objects must settle, and the can's containment check uses cylindrical bounds
so rotation about its own axis cannot create fictitious box corners.

The tipping path keeps the bottom pivot near the same place on the table.
The left hand grips the upper side edge of the laid bag nearer the finger
roots. The wrist rotates from 60 to 150 degrees and guides the rim forward
over the bottom, preserving the full turn while ending at an oblique angle. A weak elbow objective
keeps both elbows lower and closer to the torso throughout the motion, including
release and side support. The robot stands 3 cm farther from the table by default;
`--robot-setback` sets this distance in meters.
Table clearance feedback adjusts the lifting hand to keep the bottom in contact.
The right hand waits at the body's side with a nearly horizontal wrist and partly
closed fingers, then opens as it approaches the snack. Test mode rejects a
left elbow raised to shoulder level and a head looking away from the active hand.

After tipping, the left hand opens and raises its inner jaw clear of the rim,
withdraws sideways, then translates while rotating to a forward-facing grip.
It regrips the middle of the settled side rim using a one-time measured bag
frame, avoiding the corner contact that tips the bottom under load. A lower elbow
target keeps the supporting forearm forward instead of sustaining the
shoulder-high, downward-facing tipping posture. Test mode checks the left
gripper faces sideways during packing. The right hand approaches
each snack from above with open fingers; the first pickup goes directly to the
approach point without first raising the empty hand to carry height.
The single-sided support allows a modest lean and
local folds; the shape checks reject gross collapse, not all deformation.
The handles arch toward the bag opening, with their tops about 30 mm above
the rim in the upright asset, making them accessible for grasping. This remains
an elastic approximation; it does not model permanent paper creasing or damage.

Packing targets account for the supported bag's tilt and leave space for each
snack to settle toward the bottom as the shell takes its weight. The right hand
grasps each snack 35 mm above its centre so the base enters the mouth while
the finger bodies stay above the rim during release. This reduces rim impacts
and tumbling. It then raises the open gripper before withdrawing.
