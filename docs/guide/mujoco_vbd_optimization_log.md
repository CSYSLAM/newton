# MuJoCo + VBD Optimization Log

This file tracks the recent optimization work around the MuJoCo + VBD cloth
pipeline, with emphasis on changes that were actually kept in the tree.

The longer RST note at `docs/guide/mujoco_vbd_shirt_performance.rst` remains
the narrative performance write-up. This markdown log is a concise engineering
record of adopted changes, rejected experiments, and the current generic
optimization direction.

## Scope

Current profiling and optimization work started from
`newton.examples.cloth.example_cloth_franka_mujoco_shirt`, but the long-term
goal is broader:

- arbitrary cloth assets and garments
- shared MuJoCo + VBD coupling rules
- future volumetric elastic bodies and soft solids

The working rule is to keep optimizations that preserve behavior and scale to
general VBD particle systems, not just the current shirt asset.

## Adopted Changes

These changes were validated and kept.

### Example-level pipeline changes

1. Solve IK once per rendered frame, then interpolate joint positions across
   cloth substeps instead of solving IK every substep.
2. Replace per-substep `state_1.assign(state_0)` with targeted device copies of
   only the arrays needed before collision and VBD.
3. Set `rigid_contact_max=0` in the shirt and cloth examples when rigid contact
   generation is not needed.
4. Reduce IK iterations from `24` to `8` for the current Franka examples.
5. Cache the scripted joint target trajectory and replay it at runtime instead
   of re-solving the same fixed target sequence every frame.
6. Add CUDA graph capture to the shirt and cloth MuJoCo + VBD examples by
   replaying the whole `substeps x (collide + solver.step)` sequence with one
   `capture_launch` per rendered frame.
7. Move cached joint-target staging into the captured shirt and cloth graph
   path with a device-side frame counter. This removes the remaining per-frame
   host-side joint target copies from the graph replay path while preserving
   the eager fallback. On a 120-frame RTX 3060 sample, this cut the graph path
   versus a host-staged replay variant by about `1.84%` for the shirt example
   (`22.072 ms` to `21.666 ms`) and about `0.28%` for the cloth example
   (`6.879 ms` to `6.860 ms`).

### VBD solver changes kept in the shared path

1. Add a no-active-self-contact fast path so VBD skips self-contact force
   accumulation and full truncation scans when detection reports no active
   self-contact primitives.
2. Replace the no-active-self-contact per-color full-particle truncation update
   with a narrower color-subset update, then fuse that writeback into
   `solve_elasticity_tile` and `solve_elasticity`.
3. Keep body-particle contact buffers sized for correctness, but process them
   with a fixed host-side launch width and device-side grid-stride iteration
   over the active `soft_contact_count`.
4. Narrow the body-particle fixed launch width heuristic to
   `max(256, min(2048, sm_count * 32))`. This reduced overlaunch in
   contact-heavy shirt frames while preserving state.
5. Increase `TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE` from `16` to `32`. The shirt
   mesh has average / p95 adjacency of about `11.9 / 16` faces and
   `23.8 / 32` edges per active particle, so the old tile width under-served
   the edge-bending pass. The new width passed shirt and cloth headless tests
   and improved contact-heavy shirt samples.
6. Add a surface-only tile elasticity kernel and dispatch it by per-particle
   tet adjacency rather than a model-wide switch. In mixed scenes, particles
   with no adjacent tetrahedra use the lighter cloth path while particles with
   tet contributions still use the original generic `tri + edge + tet` kernel.
   This keeps mixed cloth + volumetric models correct while still letting the
   surface subset avoid dormant tet work. On the shirt contact-heavy 90-frame
   sample, this kept the last-10-frame average in the roughly `33-35 ms` band,
   below the earlier tile-32-only baseline of about `37.9 ms`.
7. Make VBD self-contact detection graph-safe by avoiding host reads of the
   active VT/EE counts during CUDA graph capture. In captured paths the solver
   now keeps the self-contact branch enabled and lets the kernels early-out on
   device when no active self-contact primitives exist.
8. Replace the fixed body-particle contact launch width with a per-step
   adaptive width on uncaptured paths, using the current active
   `soft_contact_count` to tighten the launch while keeping a fixed-width
   fallback during CUDA graph capture.
9. Fuse the self-contact "is anything active?" bookkeeping directly into the
   VT and EE detection kernels. The shared solver path only needs a boolean
   gate for the expensive self-contact force/truncation branch, not separate
   active vertex and edge totals, so the detector now writes that flag during
   query instead of running a post-detection scan. On the shirt profile, the
   old scan stage was about `0.121 ms/substep`; after fusing the flag write
   into the queries, that standalone stage disappeared.

## Rejected Or Rolled-back Experiments

These directions were explored and should not be retried without a stronger
reason or measurement plan.

1. Extending the fixed-width no-host-sync idea to rigid body-body VBD contacts.
   The rigid contact capacity was not large enough relative to active contacts
   to create useful headroom.
2. Reading `soft_contact_count` back to the host unconditionally, including
   CUDA-graph-captured paths, to size launches exactly. This breaks graph
   capture and is not kept. The retained variant only adapts launches on
   uncaptured paths and falls back to fixed-width launches during capture.
3. Replacing the shirt soft-contact capacity with explicit smaller caps. This
   did not produce a stable speed win.
4. The external-rigid lazy-generation body-particle material scheme. It was
   rolled back after proving fragile, and one revert accidentally removed the
   grid-stride increment in `accumulate_particle_body_contact_force_and_hessian`,
   causing a contact-phase infinite loop.
5. Fusing `init_body_particle_contacts` into the first external-rigid particle
   accumulation pass. This variant did not show a stable benefit.
6. Disabling the tile solve in favor of the scalar `solve_elasticity` path.
   On the shirt contact-heavy sample, `use_particle_tile_solve=False` was much
   slower than the tile path.
7. Runtime monkeypatch sweeps of `TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE`. Warp did
   not tolerate changing this constant on the fly and produced illegal memory
   access. Tile-width experiments must be done by editing source and
   recompiling kernels.
8. Replacing the particle 3x3 solve with an LDLT-based helper without a stable
   measured gain. The change was reverted rather than left in the tree.
9. Splitting volumetric particles into an extra `tet-only` launch path. In both
   the tile and non-tile paths, the added per-color kernel launches cost more
   than the small amount of skipped surface-branch work on the tested softbody
   scenes, so this direction was rolled back.

## Current Hotspots

In the contact-heavy shirt regime, the dominant cost is still the particle-side
elasticity solve, especially `solve_elasticity_tile`.

Recent profiling on the RTX 3060 laptop GPU showed that:

- `solve_elasticity_tile` remains the main recurring hot kernel in contact-heavy
  frames.
- `init_body_particle_contacts`,
  `accumulate_particle_body_contact_force_and_hessian`, and
  `update_duals_body_particle_contacts` matter, but they are smaller than the
  total elasticity solve cost.
- For the current shirt and cloth examples, wrapping the whole substep loop in
   one CUDA graph produces a much larger win than trying to shave a few more
   host calls inside the already-captured `collide + solver.step` sequence.
- After graph capture is in place, moving the cached joint-target staging into
   the captured path still buys a small extra win on shirt and is effectively
   neutral on the lighter cloth case.
- `use_particle_tile_solve=True` is still clearly preferable to the non-tile
  path for the current shirt asset.
- For cloth-only assets, removing the dormant tetrahedral branch from the tile
   kernel produced another measurable win without changing the volumetric path.

## Generic Optimization Direction

The next stage should prefer changes that generalize across cloth meshes,
garments, and volumetric elastic bodies.

### High-value directions

1. Further specialize the particle elasticity solve by material topology:
   surface cloth (`tri + edge`) and volumetric elastic (`tet`) do not need the
   same kernel structure or batching assumptions.
2. Reduce wasted work inside `solve_elasticity_tile`, especially in the
   edge-bending pass where adjacency is higher than the old tile width.
3. Keep the captured MuJoCo + VBD path free of host-side count reads and CPU
   round-trips. On uncaptured paths, small host reads can still be worth
   keeping when they pay for themselves clearly in frame time.
4. Favor optimizations that preserve one common external-rigid coupling path for
   cloth today and volumetric elastic bodies later.

### Constraints

1. Do not trade correctness for speed without an explicit behavior check.
2. Avoid optimizations that depend on a single shirt-specific parameterization.
3. Keep accepted changes graph-safe and compatible with large buffer capacities.

## Current Baseline To Keep

If future experiments regress, the stable baseline to return to is:

- cached IK + substep interpolation
- one CUDA graph replay per rendered frame in the shirt and cloth examples
- device-side cached joint-target staging inside the captured shirt/cloth path
- targeted state copies
- `rigid_contact_max=0` in the example path when rigid contacts are unused
- no-active-self-contact fast path and fused truncation writeback
- adaptive body-particle launch sizing on uncaptured paths with fixed-width
   fallback during capture
- `TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE = 32`
- surface-only tile elasticity dispatch for particles with zero tet adjacency