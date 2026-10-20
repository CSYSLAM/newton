# W1 left hand assets for MJMPM

`left_hand.urdf` is the `left_hand_base` subtree (hand base, thumb, and four
fingers; the wrist camera is dropped) of
`FAST_MJVBDV2:assets/W1-hand-obj/DexforceW1V021.urdf` at revision
`da7026d8e7db6c4f9a50264feb0ef20cea411195`. Joint definitions, limits, mimic
couplings, inertials, and collision geometry are preserved; collision mesh paths
are rewritten to `collision/`. The 11 collision meshes are copied unchanged.
`SOURCE_MANIFEST.json` records the source URDF and mesh hashes.

The visual meshes of the source (about 25 MB) are not copied; the example
displays the collision meshes. The URDF gives the finger joints zero effort
limits, so `mjmpm_hand_grasp` sets its own drive gains and torque limits.

The reference asset was supplied by the user as the DexForce W1 V021 robot with
an anthropomorphic hand. No additional license is asserted for these
user-provided assets.
