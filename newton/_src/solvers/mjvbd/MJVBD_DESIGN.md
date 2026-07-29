# MJVBD private solver snapshots

The `mujoco/` and `vbd/` directories in this package are private source
snapshots for the MJVBD composite solver.  They were copied from Newton commit
`e6ea232a`.

## Snapshot scope

- `mujoco/`: every Python source file from `newton/_src/solvers/mujoco`.
- `vbd/`: every Python source file from `newton/_src/solvers/vbd`.

The copies intentionally retain the solver class names `SolverMuJoCo` and
`SolverVBD`, but they are not exported by `newton.solvers`.  Internal imports
are package-relative and resolve to these snapshots; dependencies on shared
Newton core, simulation, coupling, XPBD, and base-solver modules continue to
use their current public implementations.

## Private VBD specializations

The MJVBD VBD snapshot additionally keeps its cloth-oriented optimizations
private: it detects whether VT/EE self-contact buffers are active, skips the
self-contact and full-array truncation paths when they are empty, partitions
every particle color into surface and volumetric groups, and uses a surface
tile kernel that contains no tetrahedral work.  Uniform active/material flags
are cached for the surface tile fast path and refreshed after relevant model
notifications.  The body-particle material, force, and dual kernels launch
only over active soft contacts outside graph capture; capture conservatively
uses the buffer capacity because it cannot read the device counter back.
CUDA graph capture similarly uses the conservative self-contact path.

## Maintenance rule

Changes to the public MuJoCo or VBD solvers do not automatically update this
package.  A future synchronization must explicitly copy the desired upstream
changes and reapply MJVBD-specific modifications, together with focused
numerical and import-isolation tests.
