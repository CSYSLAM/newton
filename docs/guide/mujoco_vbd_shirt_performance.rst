.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

MuJoCo And VBD Shirt Performance Notes
======================================

This note tracks validated performance work on
``newton.examples.cloth.example_cloth_franka_mujoco_shirt``.
The goal is to preserve the current folding behavior while reducing the
rendered frame time on the CUDA path.

Current Runtime Structure
-------------------------

The current shirt example does not step :class:`~newton.solvers.SolverMuJoCo`
inside the runtime loop. The hot path is:

1. Solve IK once per rendered frame.
2. Interpolate joint positions over cloth substeps.
3. Run FK each substep to update rigid body transforms.
4. Run :class:`~newton.CollisionPipeline`.
5. Run :class:`~newton.solvers.SolverVBD` with
   ``integrate_with_external_rigid_solver=True``.

That means the example is currently a one-way coupling pipeline where the rigid
side provides body transforms and VBD consumes them for cloth-body contact.

Validated Optimizations
-----------------------

The changes below were kept because they showed a measurable win without
breaking the headless shirt test.

1. Solve IK once per rendered frame instead of once per substep, then linearly
   interpolate the joint positions across substeps.
2. Replace per-substep ``state_1.assign(state_0)`` with targeted device copies
   of only the arrays needed before collision and VBD.
3. In :class:`~newton.solvers.SolverVBD`, add a global fast path that skips
   particle self-contact force accumulation and planar truncation full scans
   when collision detection reports zero active self-contact primitives.
4. Configure the shirt example's collision pipeline with ``rigid_contact_max=0``
   because rigid-rigid contact generation is not needed on this path.
5. Reduce the shirt example's IK iteration count from ``24`` to ``8``.
6. Precompute the fixed scripted IK trajectory into a frame-aligned cache and
   load cached joint targets at runtime instead of re-solving the same target
   sequence every rendered frame. This path was kept only after validating it
   against the sequential runtime ordering, not against sparse standalone IK
   queries.

Measured Findings
-----------------

The following findings were repeatedly observed on the RTX 3060 laptop GPU in
headless timing snippets:

1. ``SolverMuJoCo`` is constructed in the shirt example but was not part of the
   active runtime loop before these notes were written.
2. In the current shirt trajectory, warm-frame ``rigid_contact_count`` stayed
   at zero while the generic collision pipeline still spent time on rigid broad
   phase and narrow phase.
3. Switching the shirt example to ``rigid_contact_max=0`` removed unnecessary
   rigid-contact capacity from the example configuration, but the larger
   retained wins came from the VBD-side and runtime orchestration changes below.
4. After the collision-path tuning work, the dominant remaining cost shifted back
   to ``cloth_solver.step()``. The collision slice dropped to roughly
   ``0.002 s`` in instrumented warm frames while VBD still dominated the total.
5. In same-code A/B timing on the current shirt setup, cached joint targets
   outperformed live IK solves: about ``0.0570 s`` warm frame time for cached
   targets versus about ``0.0742 s`` for live IK in a 12-frame sample.
6. A sparse cache-versus-live IK comparison at isolated timestamps initially
   looked alarming, but it was the wrong validation because live IK depends on
   sequential warm-start state. When compared in the actual frame-by-frame
   runtime order, cached and live IK stayed close in joint space
   (about ``0.0399`` max absolute difference and ``0.0109`` mean absolute
   difference across the first 120 frames), and the first 120 simulated frames
   produced identical cloth particle positions and identical ``soft_contact``
   counts.
7. With cached IK enabled, an instrumented warm frame breakdown was roughly
   ``0.1078 s`` total, with about ``0.0001 s`` in joint-target setup,
   ``0.0025 s`` in ``collision_pipeline.collide()``, and about ``0.1001 s`` in
   ``cloth_solver.step()``.
8. Inside ``cloth_solver.step()``, the dominant slice was
   ``_solve_particle_iteration()`` at about ``0.0780 s`` per warm frame. The
   next largest slice was ``_initialize_particles()`` at about ``0.0115 s``;
   rigid-side work inside VBD was small in comparison.
9. Forcing the no-self-contact path in profiling only reduced warm frame time
   from about ``0.1074 s`` to about ``0.0991 s``. Self-contact empty-work is
   therefore no longer the primary bottleneck in the current shirt path.
10. The default shirt pipeline allocates ``soft_contact_max`` as
    ``shape_count * particle_count``. In the profiled run that capacity was
    ``360416`` while the active ``soft_contact_count`` stayed below ``1928``.
    That over-allocation is real, but the obvious ways of exploiting it did not
    yet produce a stable speed win.
11. A robust no-host-sync solution for body-particle coupling was to stop
   launching VBD soft-contact kernels at ``contacts.soft_contact_max`` and
   instead use a fixed solver-side launch width with device-side grid-stride
   iteration over ``soft_contact_count``. This was applied to body-particle
   contact-list building, material initialization, particle contact force
   accumulation, and dual updates.
12. In same-code A/B timing on the shirt path, that fixed-width grid-stride
   scheme reduced the warm last-6 frame average from about ``0.0851 s`` to
   about ``0.0739 s`` while keeping the first 120 frames bit-identical in
   cloth particle positions, body poses, and ``soft_contact_count``.

Negative Results
----------------

These experiments regressed or failed to produce a stable win and should not be
repeated blindly:

1. Low-frequency ``rebuild_bvh()`` in the shirt example.
2. Compact active self-contact primitive lists for the reverted shirt path.
3. Filtering soft-contact launches down to only ``COLLIDE_PARTICLES`` shapes.
4. Forcing ``contacts.soft_contact_max = 0`` in the hot path when the active
   soft-contact count was already zero.
5. Using sparse standalone IK queries to validate the cached IK trajectory.
   That check overstates the difference because it ignores the live solver's
   sequential warm-start state.
6. Reading ``soft_contact_count`` back to the host every substep and shrinking
   VBD launch dimensions to that exact count. Even though the resulting cloth
   state matched the baseline over the first 120 frames, the added host sync
   made the example slower overall.
7. Switching the shirt example from tile elasticity solve to the non-tile path.
   Timing was inconclusive and the first 120-frame cloth state diverged.
8. Replacing the shirt example's default soft-contact capacity with smaller
   explicit values without a stronger measurement protocol. A ``4096``-contact
   cap matched the first 120-frame cloth state but did not produce a stable
   timing win, while a looser ``8192`` cap diverged.
9. Host-side exact launch sizing for body-particle contact kernels. The stable
   fix was not to read back contact counts each step, but to keep launch sizing
   fixed on the host and let each kernel grid-stride over the active count on
   device.

Implications For MuJoCo And VBD Coupling
----------------------------------------

The biggest architectural gap is that the shirt example still pays for a custom
IK plus FK front end instead of reusing MuJoCo's articulated solve. A stronger
coupling path should:

1. Let MuJoCo advance the rigid articulation each cloth substep.
2. Keep ``use_mujoco_contacts=False`` so cloth-body contacts still come from the
   Newton collision pipeline when needed.
3. Feed MuJoCo-updated body transforms directly into VBD through
   ``integrate_with_external_rigid_solver=True``.
4. Avoid redundant state copies between MuJoCo data, Newton state, and VBD
   input state.

Recent Coupling Probe
---------------------

A direct probe of :class:`~newton.solvers.SolverMuJoCo` on the current shirt
model showed:

1. Warm MuJoCo substeps with ``use_mujoco_contacts=False`` and
   ``update_data_interval=0`` still cost about ``0.0091 s`` per rigid substep.
2. Warm MuJoCo frame-sized steps cost about ``0.0099 s`` per frame.
3. Because the shirt example's task script is authored in end-effector space,
   MuJoCo does not remove the need for IK by itself. It can only consume the
   resulting joint-space targets.
4. A focused probe of :meth:`SolverMuJoCo._update_newton_state` on the shirt
   model showed that Newton state writeback was only about ``0.0002 s`` inside
   a warm frame-sized MuJoCo step of about ``0.0055 s``. That means joint/body
   state export from MuJoCo is not the main current bottleneck in the coupling
   chain.

That makes a full MuJoCo solve on every cloth substep a weak speed tradeoff for
the current example. The more promising coupling path is still:

1. Keep cached IK as the task-space to joint-space front end.
2. Use MuJoCo only as a lower-frequency authoritative rigid solver when rigid
   dynamics are needed.
3. Interpolate the resulting rigid trajectory across cloth substeps before
   feeding those body transforms into VBD.

The next optimization stage should focus on this lower-frequency MuJoCo
coupling path and on the particle-side VBD kernels themselves. The current
breakdowns say the remaining dominant cost is the particle solve, not MuJoCo,
not collision broad phase, and not the cached IK front end.

General Coupling Direction
--------------------------

For a reusable MuJoCo + VBD pipeline, the body-particle coupling rule should be:

1. Keep the collision/contact buffers sized for correctness and graph safety.
2. Avoid host-side contact-count reads in the hot path.
3. Process body-particle contacts with a fixed host launch width chosen from a
   solver-side heuristic, and let each kernel grid-stride over the active
   ``soft_contact_count`` on device.
4. Reuse the same scheme for both external-rigid coupling
   (``integrate_with_external_rigid_solver=True``) and internal rigid VBD,
   so the coupling path does not fork into separate optimization strategies.

This keeps the coupling general enough for cloth today and volumetric elastic
examples later, because it removes capacity-driven empty work without changing
how contacts are generated or which contacts are solved.