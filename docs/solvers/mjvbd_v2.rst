.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

MJVBDV2
========

:class:`~newton.solvers.SolverMJVBDV2` combines selected MuJoCo
articulations with VBD/AVBD rigid bodies, cloth, tetrahedral soft bodies, and
pneumatic shells. It specializes itself to the model at construction, so a
scene that does not need MuJoCo, rigid-body VBD, or pneumatics does not execute
those branches.

.. experimental::
   :class:`~newton.solvers.SolverMJVBDV2`'s public API and behavior may
   change without prior notice.

The dynamic mixed path is deliberately one-way:

.. code-block:: text

   selected joints -- MuJoCo --> link poses as moving VBD colliders
                                      |
                                      v
                            VBD rigid/soft/cloth solve

VBD contact impulses do not feed back into MuJoCo. This is useful for a robot
driving deformable or VBD rigid objects when the object's reaction is not
intended to change the robot trajectory. Use a different coupling design when
two-way reaction forces are required.

Quick start
-----------

Register the solver attributes before finalizing the model, color the systems
that VBD will solve, then construct the solver:

.. code-block:: python

   import newton

   builder = newton.ModelBuilder()
   newton.solvers.SolverMJVBDV2.register_custom_attributes(builder)

   # Add articulations, free rigid bodies, cloth, or soft bodies here.
   builder.color()
   model = builder.finalize()

   solver = newton.solvers.SolverMJVBDV2(
       model,
       joint_mode="dynamic",
       contact_mode="auto",
       vbd_options={"iterations": 12},
   )

   state_in = model.state()
   state_out = model.state()
   control = model.control()

   solver.step(state_in, state_out, control, None, 1.0 / 120.0)
   print(solver.features.backend)

The solver owns the contact buffer and pipeline required by its selected
backend. Passing ``None`` for the ``contacts`` argument is therefore the usual
choice.

Entity ownership
----------------

MJVBDV2 partitions the model once at construction:

- MuJoCo owns selected joints and the complete closed tree of bodies connected
  to them.
- VBD owns every other rigid body and every particle.
- Cloth triangles, bending edges, tetrahedra, springs, and pneumatic cavities
  follow particle ownership.
- Static world shapes remain available to the backend that needs them.

Specify either ``mujoco_articulations`` or ``mujoco_joints``, but not both.
An explicit joint selection must form a closed joint tree. When neither option
is provided, MJVBDV2 selects articulations containing at least one joint that
is neither free nor fixed. Standalone free rigid bodies consequently stay in
VBD, and a joint-free model skips MuJoCo entirely.

Backend selection
-----------------

The selected backend is exposed as ``solver.features.backend``:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Backend
     - Scene condition
     - Execution
   * - ``pure_vbd``
     - No joints are assigned to MuJoCo.
     - VBD/AVBD only.
   * - ``kinematic_passthrough``
     - Kinematic selected joints and no particles or dynamic VBD bodies.
     - Preserves externally prescribed output; no physics solver is run.
   * - ``pure_mujoco``
     - Dynamic selected joints and no particles or dynamic VBD bodies.
     - MuJoCo only, including MuJoCo rigid contacts.
   * - ``mjvbd_kinematic_soft``
     - Kinematic links, particles, no dynamic VBD rigid body, and soft or
       automatic contact.
     - Sparse particle-shape contacts and VBD; MuJoCo is skipped.
   * - ``vbd_kinematic_full``
     - Kinematic links plus dynamic VBD rigid bodies, or explicit full contact.
     - Full VBD/AVBD with selected links as kinematic colliders.
   * - ``coupled``
     - Dynamic selected joints plus particles or dynamic VBD rigid bodies.
     - MuJoCo, one-way link synchronization, then VBD/AVBD.

``solver.features`` also reports entity and element counts and which MuJoCo,
rigid, particle, cloth, bending, tetrahedral, spring, and pneumatic branches
are active. Use it to confirm that a model took the intended path.

Dynamic and kinematic joints
----------------------------

With ``joint_mode="dynamic"``, selected joints are integrated by MuJoCo. A
mixed scene then uses their newly computed link transforms as zero-inverse-mass
moving colliders in VBD.

With ``joint_mode="kinematic"``, the application supplies the selected joint
and link state. MJVBDV2 does not construct MuJoCo:

- A particle-only interaction uses old particle positions and the newly
  prescribed rigid transforms when generating contacts.
- A scene with dynamic VBD rigid bodies uses the full VBD backend and treats
  selected articulation links as kinematic colliders.
- An articulation-only scene is a passthrough.

Do not pass ``mujoco_options`` to a kinematic backend. MJVBDV2 rejects options
that would be ignored.

Contact modes
-------------

``contact_mode`` has three values:

``"auto"``
   Use full contact when VBD owns a dynamic rigid body; otherwise use sparse
   particle-shape contact.

``"soft"``
   Generate point contacts only between particles and shapes marked to collide
   with particles. Candidate pairs are filtered by world and shape flags at
   construction. This mode cannot resolve contacts between VBD-owned dynamic
   rigid bodies.

``"full"``
   Use :class:`~newton.CollisionPipeline` and full VBD/AVBD contacts. This is
   required for mutually interacting VBD rigid bodies and for full-surface
   rigid-soft contact.

Soft paths accept ``collision_options={"soft_contact_margin": value}``, where
the margin is measured in meters. Full paths accept
:class:`~newton.CollisionPipeline` constructor options. For example:

.. code-block:: python

   solver = newton.solvers.SolverMJVBDV2(
       model,
       contact_mode="full",
       collision_options={
           "broad_phase": "nxn",
           "enable_rigid_soft_full_surface_contact": True,
           "rigid_soft_full_surface_shape_indices": interacting_shapes,
       },
   )

Restricting ``rigid_soft_full_surface_shape_indices`` to shapes that actually
touch a deformable object reduces candidate and contact-buffer work.

Pneumatic shells
----------------

Pneumatic pressure is an opt-in extension of the complete VBD backend. Author
one cavity from an existing closed triangular shell:

.. code-block:: python

   config = newton.solvers.PneumaticConfig(
       mode=newton.solvers.PneumaticMode.ISOTHERMAL,
       reference_absolute_pressure=120_000.0,
       ambient_pressure=101_325.0,
       bulk_damping=2.0e6,
   )
   cavity = newton.solvers.add_pneumatic_cavity(
       builder,
       triangle_indices,
       config=config,
   )

The helper reuses the shell's particles and triangles. It validates that the
selected faces form one closed, orientable, two-manifold surface, orients them
outward, computes the rest volume, and registers the required model, state, and
control attributes.

Supported pressure laws are:

- :attr:`~newton.solvers.PneumaticMode.ISOTHERMAL`
- :attr:`~newton.solvers.PneumaticMode.ADIABATIC`
- :attr:`~newton.solvers.PneumaticMode.TARGET_VOLUME`
- :attr:`~newton.solvers.PneumaticMode.PRESCRIBED_GAUGE_PRESSURE`

Create state and control objects through ``model.state()`` and
``model.control()`` after finalization. Pneumatic observables include volume,
absolute pressure, volume rate, and clamp flags. Solver reset restores cavity
history and honors world masks.

For a new shell, :func:`~newton.solvers.add_inflatable_mesh` combines cloth
creation and cavity registration. For an existing cloth mesh, use
:func:`~newton.solvers.add_pneumatic_cavity`.

Performance and CUDA Graph capture
----------------------------------

MJVBDV2 prunes work from absent features. In particular:

- joint-free models skip MuJoCo;
- kinematic models skip MuJoCo;
- particle-only ordinary scenes use the optimized soft VBD implementation;
- rigid-only scenes do not allocate particle modules;
- non-pneumatic scenes do not allocate pneumatic state or load pneumatic
  kernels;
- missing cloth, bending, tetrahedral, and spring topology skips the
  corresponding modules.

For CUDA Graph capture, allocate all contact capacity before capture. Construct
the solver and its owned contact pipeline, run one uncaptured warm-up step, and
then capture a fixed step. The warm-up ensures that contact history and other
lazy capacity cannot grow inside the graph.

MuJoCo sleeping may be enabled only when ``features.backend`` is
``"pure_mujoco"``. The dynamic coupled path rejects sleeping because VBD has no
feedback channel with which to wake MuJoCo bodies.

Limitations
-----------

- Dynamic mixed coupling is strictly MuJoCo-to-VBD; it has no reaction force
  from VBD to the selected articulation.
- Sleeping is unavailable in the coupled backend.
- ``contact_mode="soft"`` does not solve dynamic VBD rigid-body contacts or
  full-surface rigid-soft contacts.
- Explicit MuJoCo joint lists must be complete closed trees.
- VBD systems must satisfy the same coloring requirements as
  :class:`~newton.solvers.SolverVBD`.
- Backend-specific option dictionaries are validated; contradictory or unused
  options raise instead of being silently ignored.

Examples
--------

Representative commands include:

.. code-block:: console

   python -m newton.examples mjvbd_v2_cloth_twist
   python -m newton.examples vbd_inflatable_bag_v0
   python -m newton.examples vbd_inflatable_bag_v1
   python -m newton.examples mjvbd_v2_supermarket_plastic_bag

See the MJVBDV2 section in the repository ``README`` for the full demo list.
