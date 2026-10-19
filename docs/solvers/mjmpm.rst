.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

MJMPM
=====

.. experimental::

   :class:`~newton.solvers.SolverMJMPM` is an experimental independent
   MuJoCo/MPM backend. Its API and numerical implementation may change.

Ownership
---------

MuJoCo integrates complete selected articulation trees and their links. The
private MPM environment integrates all remaining free rigid bodies and material
particles. A standalone free joint does not enable MuJoCo. Free rigid bodies
retain their mass, inertia, and rigid transforms; they are not approximated by
high-stiffness particles.

The implementation owns private MuJoCo adapters, material kernels, rigid contact
integration, collision orchestration, and proxy coupling. It shares Newton's
model/state contracts and foundational geometry routines. It does not instantiate
the shared MuJoCo, implicit MPM, VBD, XPBD, or coupled solvers.

Quick start
-----------

.. code-block:: python

   import newton
   from newton.solvers import SolverMJMPM

   builder = newton.ModelBuilder()
   SolverMJMPM.register_custom_attributes(builder)
   # Add complete articulations, independent free rigid bodies, and particles.
   model = builder.finalize()
   solver = SolverMJMPM(
       model,
       coupling=SolverMJMPM.Coupling.TWO_WAY,
       mpm_options={"grid_type": "fixed", "max_active_cell_count": 256},
   )
   state = model.state()
   control = model.control()
   newton.eval_fk(model, state.joint_q, state.joint_qd, state)
   solver.step(state, state, control, None, 1.0 / 600.0)

Pass ``contacts=None`` because the solver owns collision detection. Inspect
``solver.features`` and ``solver.ownership`` to verify the selected domains.
``mujoco_joints`` may explicitly select complete joint trees; joints left in
the MPM domain must be standalone free joints.

Run the press example with:

.. code-block:: console

   uv run -m newton.examples mjmpm_press
   uv run -m newton.examples mjmpm_press --coupling one_way

The blue press holds above the yellow material for one second, compresses it
over three seconds, holds the load for one second, and retracts over three
seconds. Its weight is compensated so uncommanded sag does not obscure the
contact response. Gravity still acts on the material. The UI shows the phase
and commanded/measured joint travel, as well as the surface clearance. The example contains no auxiliary falling
rigid box; free-rigid behavior is covered by the numerical tests.

Dynamic W1 squeeze and grasp
----------------------------

.. code-block:: console

   uv run -m newton.examples mjmpm_w1_squeeze_grasp
   uv run -m newton.examples mjmpm_w1_squeeze_grasp --viewer null --test
   uv run -m newton.examples mjmpm_w1_squeeze_grasp --coupling one_way

This scene copies the W1 V030 asset, table, object sizes, trajectory, and joint
drive parameters from the ``FAST_MJVBDV2`` squeeze/grasp example. MuJoCo owns
the complete dynamic, fixed-base robot; IK generates drive targets only.
The MPM environment owns the left free rigid block and the right elastic
particle cube. The robot closes both grippers, lifts both objects by 0.15 m,
holds, lowers, and releases them. Finger effort limits remain 10 N.

The soft cube uses 288 particles with the reference density and equivalent
small-strain Young's modulus and Poisson ratio. It uses a discontinuous linear
strain basis and Gauss-Seidel iteration to resolve the localized squeeze.
The scene selects ``rigid_contact_hard=False`` to use implicit spring/damper
contacts from the shapes' ``ke`` and ``kd``, as in the reference scene. Hard
rigid contacts remain the solver default. Compliant rigid contacts do not
transmit forces across a positive surface gap.

The example's tests check ownership, dynamic robot flags, pre-contact material
stability, grip width, hold jitter, object lift, table support, and recovery
after release. A full test run requires 720 frames. Asset provenance and
import behavior are documented beside the copied URDF.

Material contacts
-----------------

With the default PIC integration scheme, the environment samples contact at
material particles and activates constraints at their physical radii. It uses
``collider_basis="pic"`` by default. Contact impulses are then transferred
through the MPM grid and returned to rigid bodies and articulated links.
This avoids constraining empty grid nodes above the material surface. Explicit
grid collider bases such as ``S2`` retain the migrated grid contact behavior,
whose contact location depends on grid resolution and support.

A plain ``pic`` basis reads the maximum cell occupancy on the host at
construction and during uncaptured steps, and grows a per-cell capacity, a
multiple of 8 with at least one particle of headroom, so every particle remains
a contact sample. Captured steps reuse the capacity reached before capture. An explicit ``picN``
basis fixes the capacity to ``N`` in every step. Both cases record the
observed occupancy on the device: call
:meth:`~newton.solvers.SolverMJMPM.check_capacity_status` after graph replay
to detect cells whose extra particles were ignored. The press validation
records the maximum observed cell occupancy alongside contact clearance and
impulses.

Same-step coupling
------------------

By default, two-way coupling makes one MuJoCo/MPM pass per step: MuJoCo
advances the links with the contact feedback harvested in the previous step,
and the MPM environment harvests new feedback against link proxies with
approximate independent inertia (``mass_scale`` times MuJoCo's effective body
inertia).

``coupling_iterations`` repeats the pass from the same step-start state. Each
repetition restores the MPM warm starts and previous collider poses, MuJoCo's
acceleration warm start, actuator activations, and time, and reuses the
step's contact detection; material history comes from the step-start state.
At the fixed point the MPM proxy and MuJoCo link end velocities agree, so the
proxy inertia only affects convergence. After the last pass MuJoCo is
re-solved with that pass's unrelaxed feedback, so both domains apply the same
contact impulse once and momentum exchanged across the interface is
conserved even before convergence. ``relaxation_mode="aitken"`` adapts the
feedback relaxation within each step, and ``coupling_tolerance`` [m/s, rad/s]
ends uncaptured steps early. ``coupling_residuals`` holds each pass's largest
linear and angular interface velocity residual, and ``coupling_passes`` the
passes run.

Articulated contact response
----------------------------

``articulated_response=True`` replaces the independent proxy inertia with the
link mobility of the MuJoCo Warp step. For an impulse wrench on link ``m`` the
step changes the twist of link ``l`` by ``Lambda^-1_lm p_m`` with
``Lambda^-1 = J H^-1 J^T``, where ``J`` stacks the link center-of-mass
Jacobians and ``H`` is the matrix MuJoCo factors to integrate velocities
(``M - dt dqfrc/dqvel`` for the implicit integrators, which includes joint
damping and actuator velocity gains). The mobility is evaluated after every
MuJoCo step. Rigid environment contacts update every contacted link of the
world, and the MPM collider rigidity operator maps nodal impulses to link
wrenches and then to coupled link twists. MuJoCo is re-solved with the
harvested feedback, so a single pass already applies contact feedback within
the step. In regression scenes, one articulated pass matches the converged
same-step solution of 30 to 40 proxy passes to within a few millimeters per
second, including the motion transmitted to links that are not in contact.

The mobility is linear and excludes MuJoCo constraints (joint limits, MuJoCo
contacts, equality constraints) and actuator force clamping; additional
coupling iterations correct them. It requires the MuJoCo Warp backend
(``use_mujoco_cpu=False``), which also runs on CPU devices, and stores a
``6 n x 6 n`` matrix per world for ``n`` articulated links.

Current limits
--------------

- Same-step iterations repeat the full MPM solve per pass, so they multiply
  the material cost. Articulated response removes most of the need for them
  in the linear regime but does not include MuJoCo constraint activity.
- Rigid contacts use a sequential projected-impulse solver with optional
  implicit compliance. Worlds are solved in parallel, but the contacts of
  one world are swept by one thread, so very contact-rich worlds are slow.
  Restitution coefficients, rolling/torsional friction,
  prescribed contact surface velocity, and continuous collision detection are
  not implemented by that solver.
- Sleeping is unsupported in the coupled backend.
- Thin-shell cloth is not implicitly supported by assigning it to MPM.
- Material support follows the private implicit MPM implementation. Air drag
  defaults to its existing value; set ``air_drag=0`` for closed-system momentum
  tests.
- CUDA Graph capture requires a positive ``max_active_cell_count`` with a
  fixed grid or a rebuildable sparse grid (``grid_padding=0``), memory pools,
  and a successful uncaptured warm-up. Captured steps cannot grow reserved
  capacities; check them with ``check_capacity_status()`` after replay.
  Graph validation covers a small coupled scene containing an articulation,
  a free rigid body, and elastic particles.

Validation
----------

Run the numerical acceptance cases and source isolation audit with:

.. code-block:: console

   uv run --no-sync python scripts/audit_mjmpm.py --run-tests
   uv run --no-sync python scripts/diagnose_mjmpm_press_contact.py --screenshots

The audit records backend versions, device names, source provenance, shared
solver import violations, and numerical test results in
``docs/lab/mjmpm/validation.json``.
