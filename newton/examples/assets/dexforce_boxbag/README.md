# DexForce W1 VBD box-bag example assets

`DexforceW1V021/` is copied unchanged from the local `FAST_MJVBDV2` branch
(commit `49bb6220`). It contains the full W1 URDF, collision meshes, visual
meshes, and textures. Keeping the asset locally avoids a runtime dependency
on that branch or an external robot download.

`robot_targets.npz` contains only the first 40 (robot) joint coordinates from
`assets/vbd_mjvbd_v2/vbd_mjvbd_v2_dexforce_soft_then_rigid_cube_into_bag.npz`
at the same commit: 1900 frames at 60 Hz, plus the matching URDF joint names.
The current importer was checked against the original recording: forward
kinematics at the first frame agrees within 2.4e-7. No recorded body poses,
object coordinates, particle positions, or particle velocities are included.

The source workflow is
`example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_final00.py`:
soft-cube grasp, lift, transport, release, followed by the same sequence for
a rigid cube. Both are deposited into a five-sided cloth bag with a fixed rim.

Run from the repository root:

```bash
uv run --extra examples -m newton.examples vbd_dexforce_boxbag
uv run --extra examples -m newton.examples vbd_dexforce_boxbag --viewer null --num-frames 1900 --test
```

The bag is partly transparent so both deposited cubes remain visible. Use
`--bag-opacity 1` for the original opaque appearance. The default runs 12
substeps with 24 VBD iterations each; 1900 frames represent 31.7 seconds of
simulation, and take longer than real time on the tested GPU.

This port uses `SolverVBD` for the dynamic robot, rigid cube, tetrahedral soft
cube, and cloth bag. The recording drives position and velocity targets.
URDF mimic relations are disabled because the recorded finger PIP targets
were edited independently. The default `block_sparse_joints` solve has zero
joint friction, as required by this branch's sparse articulation solver.

The original stage-dependent contact setup is retained, including unusually
high friction during grasp and reduced friction at release. It is a scripted
contact demonstration, not a calibrated grasp controller. The robot uses the
original hand collision meshes without auxiliary pads or object attachments.
The original kinematic robot, MJVBDV2 solver, multilevel particle correction,
and decorative house background are replaced by dynamic VBD joint drives,
the current particle solver, and a visible physical table.
