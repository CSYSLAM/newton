# G1 29-DoF Asset Record

The locomotion task uses a repository-local USD package at:

```text
examples/locomotion_g1/assets/unitree_g1_29dof/
|-- g1_29dof_rev_1_0.usd
`-- configuration/
    |-- g1_29dof_rev_1_0_base.usd
    |-- g1_29dof_rev_1_0_base.usd.orig
    |-- g1_29dof_rev_1_0_physics.usd
    `-- g1_29dof_rev_1_0_sensor.usd
```

This directory is intentionally ignored by Git. The files are large, and the
redistribution terms of this particular USD package have not been established
in this repository. A checkout without the package remains usable with
Newton's downloadable URDF fallback, but that fallback is not asset-equivalent.

## Provenance

The local package was copied from:

```text
/home/ubuntu/trampoline_sim/assets/unitree_g1_29dof/
```

It was introduced in the `trampoline_sim` repository by commit
`69e2c549a0fd8d4e5bfd92195b19dc19b89e300c` (`add asset: g1`). That commit does
not record an upstream download URL, release identifier, or official checksum.
The model follows the Unitree RL Lab G1 29-DoF layout, but the available
provenance is insufficient to claim byte-for-byte identity with an official
NVIDIA Nucleus or Unitree release.

## Collision Differences

The package includes `g1_29dof_rev_1_0_base.usd.orig` as its comparison
baseline. A semantic USD diff against the active base layer shows that the
articulation, joint layout, body mass, center of mass, and inertia data are
retained. The active base layer changes both ankle-roll links:

- `left_ankle_roll_link/mesh_0` through `mesh_5` are deactivated.
- `right_ankle_roll_link/mesh_0` through `mesh_5` are deactivated.
- Each ankle-roll link receives one collision mesh named `mesh_collision`.
- The new meshes use `PhysicsCollisionAPI`, `PhysicsMeshCollisionAPI`, and
  `physics:approximation = "none"` in the USD source.
- The SynReal/Newton loader subsequently applies the configured
  `convex_hull` mesh approximation for training.

After the training import path and convex-hull approximation, the derived USD
produces 30 bodies, 30 joints (one floating root plus 29 actuated joints), 58
shapes, 23 collidable shapes, and a total mass of `33.341140747 kg`. There are
two collidable foot shapes, one per ankle-roll link.

Newton's downloadable G1 URDF produces the same body and joint counts and the
same total mass, but it produces 71 shapes and 36 collidable shapes under the
same importer settings. It is therefore suitable only as an explicit
compatibility fallback, not for reproducing training or sim2sim results.

## File Manifest

The manifest paths, relative to the package root, are:

```text
g1_29dof_rev_1_0.usd
configuration/g1_29dof_rev_1_0_base.usd
configuration/g1_29dof_rev_1_0_base.usd.orig
configuration/g1_29dof_rev_1_0_physics.usd
configuration/g1_29dof_rev_1_0_sensor.usd
```

Their SHA256 values, in the same order, are:

```text
cfed730bb043c42708260f25bf3bdf49ac80deed5d5a49384721b8e01059a96e
a248d55f97faf91bd7e2e7e63025623ccf5094bdc4c26a1166e5aded2e4c887c
d9768c942783ae0932f0c3db3558d5283b9a93ba0e561d15d8984f548aa2a65d
cde4ff0378183e4e2063df1fdcad4060df8247eaf1da777f45e2adea0b6dde22
5115364b53bffbb37007af0fd1038336679b66abdae134bc87d0fca9829f9b60
```

The top-level USD is a small composition layer. Its hash alone does not cover
changes to the referenced files under `configuration/`; compare the complete
manifest when moving the package between machines.

## Installation

Copy the complete directory, preserving its relative layout. The top-level USD
will not load correctly if only that file is copied.

The default location needs no additional configuration. For another location,
set:

```bash
export G1_ASSET_PATH=/absolute/path/to/unitree_g1_29dof/g1_29dof_rev_1_0.usd
```

The `--asset` option on `train.py` and `play.py` overrides both the environment
variable and the repository-local default.
