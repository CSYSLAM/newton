# W1 V030 assets for MJMPM

The robot URDF and its 62 referenced meshes are copied from
`FAST_MJVBDV2:newton/examples/mjvbdv2/assets/w1_v030/` at revision
`2af3c8c0b496394de88e287c1efbd617c8b6eb12`.
`SOURCE_MANIFEST.json` records source hashes. Mesh payloads were resolved from
the existing local Git LFS objects; URDF geometry and joint definitions are
preserved.

The reference asset was supplied by the user from
`AssembleW1-030/DexforceW1V030/w1-030/`. Its URDF identifies DexForce Technology
Co., Ltd., assembly manager V2.0, generated July 14, 2026. No additional license
is asserted for these user-provided assets. Finger visual winding repairs from
the reference branch are retained. Some original GLB materials lack UVs; the
Newton importer falls back to their material color. Newton also applies its
standard inertia validation when finalizing the imported models.

`mjmpm_w1_squeeze_grasp` uses a dynamic, fixed-base robot. MuJoCo integrates
the robot under gravity compensation and joint PD drives; IK only produces
drive targets. The MPM environment integrates the free rigid block and the
elastic particle cube and returns contact forces to the robot. There are no
grasp attachment joints or object pose resets. See the example for the
demonstration control, material, and friction parameters.
