# Piper source asset

Imported from [SynReal/newton, WAIC](https://github.com/SynReal/newton/tree/WAIC/style3d/examples/assets/piper),
pinned to commit `e1176d518f831b17392fb1bf6b90de8b97f200a9`.
Only meshes referenced by `piper_with_texture.xml` are bundled. That XML and
the referenced mesh files are unmodified. `LICENSE.md` is the source
repository's Apache-2.0 license; no separate asset license was present in
the inspected Piper directory.

The supermarket example adapts this asset in memory: sets the mesh directory
and raised mounting position, removes an unused material referencing an absent
scene texture, names geometries, and adds a task TCP and visible silicone pads.
The native arm, gripper geometry, joint axes and limits are retained. The two
slide coordinates have opposite signs. Silicone pad visual and collision boxes
are identical; visual meshes never add duplicate contact geometry.

The robot remains a prescribed moving boundary in MJVBD V2. The food has
positive mass and is transported by contact/friction, without attachment,
teleportation or velocity resets. The pad friction coefficient (1.1) is
illustrative rather than a measured material calibration.
