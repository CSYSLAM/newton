# MJVBDV2 current design and implementation status

This document is the source of truth for the implementation currently in this
directory. Despite the historical filename, it describes the as-built solver,
not a future implementation plan. User-facing behavior belongs in
`docs/solvers/mjvbd_v2.rst`; this file records internal ownership, dispatch,
data-flow, and maintenance constraints.

## 1. Purpose and non-negotiable invariants

MJVBDV2 combines selected generalized-coordinate articulations with VBD/AVBD
objects without paying for solver branches that are absent from a scene.

The design invariants are:

1. MuJoCo owns only the explicitly selected articulation joints and their
   closed set of link bodies.
2. VBD owns every unselected free rigid body and every particle. Cloth,
   tetrahedra, springs, and pneumatic cavities follow particle ownership.
3. VBD-owned rigid bodies, cloth, and soft bodies interact mutually through
   one VBD solve.
4. In the dynamic coupled backend, MuJoCo link poses are one-way moving
   colliders for VBD. VBD contact impulses never feed back into MuJoCo.
5. Kinematic joint modes do not instantiate MuJoCo when prescribed poses are
   already available.
6. Missing scene features must prune their modules, storage, kernels, and
   collision work. Pneumatics must be completely dormant in ordinary demos.
7. MJVBDV2 never falls back to `SolverMJVBD`. Its fast paths are private V2
   backends selected from the same public `SolverMJVBDV2` entry point.
8. The solver owns the contact pipeline required by its selected backend.
   Passing `contacts=None` therefore remains the normal call pattern.
9. MJVBDV2-specific behavior and performance changes stay inside this package.
   Shared Newton `Model`, `State`, and `Contacts` data are inputs; migrating V2
   must not require patches to Newton's shared geometry, collision, or solver
   modules.

These constraints preserve existing MJVBDV2 demos while allowing cloth-only,
rigid-only, articulation-only, and pneumatic scenes to avoid unrelated work.

## 2. Package structure

```text
mjvbd_v2/
├── solver_dispatch.py       public entry point and six-backend dispatch
├── ownership.py             MuJoCo/VBD entity partition and validation
├── solver_mjvbd_v2.py       dynamic one-way coupled backend
├── collision_pipeline.py    sparse particle-shape pass for current-state paths
├── soft_contact_pipeline.py sparse old-particle/new-rigid kinematic pass
├── full_contact_pipeline.py private full-surface AABB rejection on CUDA
├── mujoco/                  private MuJoCo copy used by V2
├── vbd/                     complete VBD/AVBD implementation and pneumatics
├── vbd_soft/                optimized external-rigid particle implementation
├── BASELINES.md             historical demo hash audit
├── OPTIMIZATION_LOG.md      measured performance decisions and rejected trials
└── MJVBDV2_PLAN.md          this as-built design record
```

There is deliberately no `state_sync.py`. Dynamic composition uses Newton's
generic coupled-solver state distribution/reconciliation machinery. Pneumatic
state needs an extra compact gather/scatter layer and is implemented by
`_SolverMJVBDV2Pneumatic` in `solver_mjvbd_v2.py`.

The package boundary is also the optimization boundary. Upstream Newton
improvements may benefit V2, but they are optional dependencies rather than
part of the standalone MJVBDV2 implementation. An optimization that requires a
shared-module change must be rejected here or reimplemented privately.

The only user-facing imports are public symbols from `newton.solvers`:

- `SolverMJVBDV2`
- `PneumaticMode`, `PneumaticConfig`, and `PneumaticCavityHandle`
- `add_pneumatic_cavity()`, `add_inflatable_mesh()`, and
  `register_pneumatic_attributes()`

Examples and documentation must not import this internal package.

## 3. Ownership model

`resolve_ownership()` computes one immutable `MJVBDV2Ownership` before backend
selection.

### 3.1 Joint selection

`mujoco_articulations` and `mujoco_joints` are mutually exclusive:

- `mujoco_articulations` selects every joint belonging to the requested
  articulation IDs.
- `mujoco_joints` accepts a joint list only when it forms a complete, closed
  joint tree. A selected child may not omit its owning ancestor, and an
  unselected joint may not touch a MuJoCo-owned body.
- When neither is supplied, V2 infers articulation IDs containing at least one
  joint whose type is neither `FREE` nor `FIXED`. It then selects every joint
  in those articulations. Standalone free bodies therefore stay in VBD.
- When inference finds no mechanism articulation, MuJoCo owns no joints and
  dispatch proceeds directly to the pure-VBD backend.

This default is intentionally not “all joints.” It prevents a model containing
only free rigid bodies from constructing MuJoCo.

### 3.2 Body and particle ownership

MuJoCo bodies are the selected joints' child bodies plus connected parent
bodies required by the closed tree. Every other body is VBD-owned. All
particles are VBD-owned.

The partition is disjoint. A selected MuJoCo link is represented inside VBD
only as a zero-inverse-mass proxy collider; this proxy is not a second dynamic
owner.

`has_vbd_dynamic_bodies` is true only if at least one VBD-owned body has
positive inverse mass and is not kinematic. Static shapes do not activate the
VBD rigid solver.

## 4. Scene-specialized dispatch

`SolverMJVBDV2` resolves ownership, counts active model features, and selects
exactly one backend at construction. Selection order is significant:

| Backend | Selection condition | Work performed |
| --- | --- | --- |
| `pure_vbd` | No joints are MuJoCo-owned | VBD/AVBD only; no MuJoCo allocation |
| `kinematic_passthrough` | Selected joints, `joint_mode="kinematic"`, and no particles or dynamic VBD bodies | No solve; preserves externally authored output |
| `pure_mujoco` | Selected joints, `joint_mode="dynamic"`, and no particles or dynamic VBD bodies | MuJoCo only, including its rigid contacts |
| `mjvbd_kinematic_soft` | Kinematic selected joints, particles, no dynamic VBD rigid bodies, and `contact_mode` is `auto` or `soft` | Sparse particle-shape contacts plus VBD; no MuJoCo |
| `vbd_kinematic_full` | Kinematic selected joints with dynamic VBD rigid bodies, or explicit full contact | Full VBD/AVBD; selected links are kinematic colliders; no MuJoCo |
| `coupled` | Dynamic selected joints plus particles or dynamic VBD bodies | MuJoCo step, one-way proxy synchronization, then VBD/AVBD |

“No VBD dynamics” means no particles and no dynamic VBD-owned rigid bodies.
Static VBD-owned geometry may still be present.

### 4.1 Feature audit

`solver.features` is a frozen `SolverMJVBDV2.Features` snapshot. It exposes:

- the selected backend;
- MuJoCo joint, VBD body, dynamic VBD body, and particle counts;
- triangle, bending-edge, tetrahedron, spring, pneumatic-cavity, and
  pneumatic-face counts;
- booleans for every solve branch.

Use it in tests, examples, and performance diagnostics instead of inferring the
path from private backend types.

## 5. Per-backend data flow

### 5.1 Dynamic coupled path

One substep is:

```text
global state_in
    │
    ├── gather selected joints/bodies ──> MuJoCo step
    │                                      │
    │                                      └── new link poses/velocities
    │
    └── gather VBD bodies/particles ───────────────┐
                                                   ▼
                          zero-mass proxy links + collision generation
                                                   │
                                                   ▼
                                            VBD/AVBD step
                                                   │
                                                   ▼
                     reconcile MuJoCo joints + VBD objects into state_out
```

The proxy mapping uses one staggered iteration. Proxy effective masses are
disabled in the VBD view, proxy relaxation is zero, and feedback buffers are
not allocated. Coupling forces are kept zero. This is a strict one-way
contract, not an iterative two-way coupling approximation.

Consequences:

- a robot may push cloth, soft bodies, or VBD rigid bodies;
- those objects do not alter the robot's joint trajectory;
- VBD contacts cannot wake a sleeping MuJoCo articulation.

### 5.2 Kinematic particle path

The caller supplies current link transforms in `state_out`. The sparse
`MJVBDSoftContactPipeline` combines particles from `state_in` with those new
rigid transforms, then VBD advances only the particles. This preserves the
established kinematic MJVBD ordering without constructing MuJoCo or the full
collision pipeline.

### 5.3 Kinematic full-VBD path

A shallow model overlay shares topology arrays with the original model while:

- assigning zero inverse mass and the kinematic flag to selected links;
- disabling their selected joints in the VBD view;
- leaving unselected free rigid bodies dynamic.

Full VBD writes particle and body outputs. The wrapper preserves the prescribed
joint coordinates and velocities from `state_in`.

### 5.4 Pure paths

- `pure_mujoco` creates a compact one-entry coupled view containing only
  selected articulation bodies, joints, their shapes, and world shapes.
- `pure_vbd` chooses the complete VBD implementation when rigid integration,
  pneumatics, or a particle-free model requires it. Particle-only ordinary
  scenes use `vbd_soft`.
- `kinematic_passthrough` validates a positive time step and otherwise leaves
  caller-authored output untouched.

## 6. Contact ownership and modes

`contact_mode` controls particle/rigid and VBD rigid contact generation:

- `auto` selects `full` when VBD owns a dynamic rigid body and `soft`
  otherwise.
- `soft` uses a compact, world-filtered particle-shape candidate array. Shapes
  without `COLLIDE_PARTICLES` are removed before device allocation. Each pass
  updates conservative world shape AABBs and rejects spatially remote pairs
  before SDF or mesh evaluation. This mode cannot solve VBD-owned dynamic
  rigid-body contacts.
- `full` uses the private `MJVBDV2CollisionPipeline`, layered over
  `CollisionPipeline`, and full VBD/AVBD contact handling. Its defaults are
  `broad_phase="nxn"` and `include_static_kinematic_pairs=False` unless the
  caller overrides them.

The two sparse contact helpers have intentionally different state ordering:

- `MJVBDV2SoftContactPipeline.collide()` consumes a single current state and is
  used by pure-VBD paths.
- `MJVBDSoftContactPipeline.generate()` consumes old particles and new rigid
  transforms and is used by the kinematic soft path.

Both allocate exactly one soft-contact record per world-compatible
particle-shape candidate. Active particle flags remain device-side so changing
activity does not force a host rebuild.

The sparse AABB pass does not resize or compact that candidate table at
runtime. Pair order, capacity, replay tids, and CUDA Graph launch topology stay
fixed. Shape margin is included in each shape bound, while the pair test adds
the runtime soft-contact margin and particle radius. Infinite planes remain
eligible for every compatible particle because their signed half-space cannot
be enclosed by a finite scene-independent AABB.

The private VBD constructors use the same world-compatibility count as their
initial body-particle contact-state capacity. They do not preallocate the full
`particle_count * shape_count` Cartesian product, which grows quadratically
with replicated world count. A larger externally supplied `Contacts` buffer
still grows the state lazily before graph capture.

On CUDA, both sparse helpers store the same candidate set in stable
shape-major order. This is a construction-time layout choice: it adds no
per-step sorting or allocation and improves locality for shape transforms and
shape queries. CPU keeps the original construction order.

The coupled backend keeps the MuJoCo-native collision-mode requirement while
setting `disable_contacts=True`, so MuJoCo does not solve a duplicate contact
set. VBD owns contacts involving VBD objects after proxy synchronization. The
pure-MuJoCo backend can retain MuJoCo's native contact solve.

Full-surface rigid-soft contact remains a full-pipeline feature. All V2
full-contact backends construct the private pipeline so performance changes do
not modify Newton's shared collision implementation. When the feature is
enabled on CUDA, edge/shape and face/shape candidates are stored in stable
shape-major order. After the ordinary rigid and per-particle passes have
updated shape AABBs, two lightweight kernels compare each soft feature's
world AABB against its rigid shape AABB. The feature AABB is expanded by the
runtime soft margin and maximum incident particle radius. The shared rigid
AABB contains both `shape_margin` and the rigid-only `shape_gap`; the private
test removes any positive gap with a 1e-6 m conservative safety margin, leaving
the shape margin that participates in the soft-contact threshold. Negative
gaps retain their shared deliberately reduced bound. Non-overlapping pairs
return before transforms or SDF samples. Candidate capacities, replay-tid
ranges, contact thresholds, SDF iteration counts, and emitted record fields
are unchanged, so the path remains CUDA-Graph-capturable and supports runtime
shape-flag and gap changes. CPU and disabled/full-surface-empty scenes use the
shared implementation without allocating masks.

When examples restrict full-surface contact to selected collision shapes,
candidate and contact buffers should be sized for those shapes rather than the
entire scene.

## 7. VBD implementations and branch pruning

MJVBDV2 contains two private VBD implementations:

- `vbd/` is the complete VBD/AVBD solver. It supports dynamic rigid bodies,
  rigid contacts, particles, cloth, bending, tetrahedra, springs, and pneumatic
  cavities.
- `vbd_soft/` is the optimized external-rigid particle path. It excludes
  pneumatic code and avoids rigid solver modules when the scene does not need
  them.

Construction inspects topology counts and only initializes present modules.
Examples:

- a rigid-only pure-VBD scene has no particle force arrays or particle kernels;
- cloth without bending edges skips the bending module;
- a tetrahedral scene without triangles skips surface elasticity;
- a non-pneumatic scene has no pneumatic state, buffers, or kernel module;
- small CUDA graph-color groups use scalar elasticity kernels when tiled
  launches would be under-subscribed.

Both private VBD implementations split graph-color groups once at construction.
Vertices without adjacent tetrahedra use a surface-only CUDA tile kernel for
membrane and bending elasticity; vertices incident to tetrahedra retain the
generic volumetric kernel. The split removes per-vertex tetrahedron work from
cloth and pneumatic-shell iterations without changing the force model or color
ordering.

Particle positions in `state_in` are the mutable VBD working buffer during a
step. The final positions are copied to `state_out` once after all iterations,
then velocity and pneumatic observables are finalized from that output. Do not
copy the full position array inside the iteration loop: no iterative rigid or
particle branch consumes the intermediate output array.

The complete `vbd/` path caches one 32-bit particle-color membership mask per
active rigid-soft contact when the model has at most 32 graph colors. The mask
is rebuilt on contact refresh and lets each color return before rigid-soft force
evaluation when a contact cannot affect it. Contact generation, contact order,
force laws, color solve order, launch dimensions, and CUDA Graph topology stay
unchanged. Models with more than 32 colors retain the original scan path.

Large replicated CUDA models may instead use a linked per-particle gather for
point rigid-soft contacts. The path is selected only when at least one particle
color group has `SM count * 128` particles, execution is nondeterministic, and
the model does not require gradients. It builds adjacency over the
device-resident active contact prefix, then launches one thread per particle in
each color instead of rescanning the full contact capacity for every color.
Unified full-surface edge/face streams retain the contact-major path. All
colors use the same gather decision so mixed-size color groups cannot observe
partially initialized contact acceleration data. The linked-list storage is
allocated lazily from the supplied `Contacts` capacity rather than the model's
Cartesian particle-shape upper bound, avoiding a second oversized allocation.

The body-particle AVBD dual update uses a graph-stable grid-stride active-prefix
launch under the same large-color threshold. Smaller models keep the previous
one-thread-per-capacity launch because the grid-stride form measured slower in
the representative single-world cloth scene. Both choices read the active
count on device and clamp it to capacity, so CUDA Graph replay does not require
a host synchronization.

Dense rigid-side body-particle contacts use a hybrid reduction on CUDA. Models
requiring gradients, deterministic execution, CPU execution, contact buffers
smaller than 512 records per body, and bodies with fewer than 128 active
contacts retain the four-thread strided kernel. Above those gates, 64-thread
blocks evaluate independent contact chunks into solver-owned partial buffers;
a second block reduces the chunks once per body. The contact force law,
per-contact materials, Gauss-Seidel color order, and final rigid solve are
unchanged. Only floating-point reduction order changes in the explicitly
nondeterministic path. Chunk counts and scratch capacity are fixed before
capture, so runtime contact counts remain device-side and CUDA Graph topology
does not change. The rigid solve continues to consume
`model.body_color_groups` directly. Existing device-side inverse-mass checks
reject static and kinematic rows, avoiding a host-cached topology that would
require CUDA Graph recapture after body mass or flag changes.

The optimized `vbd_soft/` path selects self-contact activity on device. Do not
add per-frame host reads merely to choose a branch inside a captured graph. The
complete `vbd/` path does not currently share this fast path; its measured port
was rejected and is recorded in `OPTIMIZATION_LOG.md`.

## 8. CUDA Graph and runtime material changes

Both VBD implementations are graph-capture compatible when all capacity is
allocated before capture. Build the collision pipeline before the solver or
run one uncaptured warm-up step so contact-history arrays cannot grow inside
capture.

`set_soft_contact_material_source(materials, material_index)` binds:

- a device `wp.array[wp.vec3]` table whose rows are `[ke, kd, mu]`;
- a one-element device `wp.array[wp.int32]` row selector.

Changing the selector in stream order allows one captured graph to replay
different cached soft-contact materials without a CPU/GPU round trip or graph
re-recording. Passing both arguments as `None` restores model material arrays.

This is a low-level VBD capability exposed through `solver.vbd_solver`. It must
be configured before capture.

## 9. Pneumatic cavities

Pneumatics are opt-in custom attributes authored on an existing closed
triangular shell. `add_pneumatic_cavity()`:

1. validates one closed, orientable, two-manifold surface;
2. infers consistent outward face signs unless signs are supplied;
3. computes positive rest volume;
4. registers model, state, and control attributes;
5. reuses existing particles and triangles rather than duplicating geometry.

Supported pressure laws are:

- isothermal: `p V = constant`;
- adiabatic: `p V^gamma = constant`;
- target volume: a quadratic volume energy;
- prescribed gauge pressure: a control-provided pressure difference.

Persistent state contains volume, absolute pressure, volume rate, and clamp
flags. Controls include pressure scale, prescribed gauge pressure, and target
volume scale. `reset()` restores cavity observables and previous-volume
history, including world-masked resets.

Any cavity forces use of the complete `vbd/` implementation, including on an
otherwise soft-only kinematic or coupled scene. A model with zero cavities
must not import or allocate the pneumatic execution module.

## 10. Sleeping

MuJoCo sleeping is allowed only in `pure_mujoco`. It is rejected in the dynamic
`coupled` backend because the one-way contract has no VBD-to-MuJoCo wake-up
signal. Kinematic backends do not construct MuJoCo, so `mujoco_options` are
rejected there rather than silently ignored.

Do not enable coupled sleeping until a wake propagation design exists and has
tests for VBD contact, control input, and world-masked reset.

## 11. Public construction contract

Applications must register the union of possible private solver attributes
before finalizing the model:

```python
builder = newton.ModelBuilder()
newton.solvers.SolverMJVBDV2.register_custom_attributes(builder)
# Add bodies, joints, cloth, soft bodies, or cavities.
builder.color()
model = builder.finalize()

solver = newton.solvers.SolverMJVBDV2(
    model,
    mujoco_articulations=[0],
    joint_mode="dynamic",
    contact_mode="auto",
    vbd_options={"iterations": 12},
    mujoco_options={},
    collision_options={},
)
```

The wrapper exposes these inspection attributes:

- `features`: selected path and active branch snapshot;
- `contacts`: backend-owned contact buffer, or `None`;
- `mujoco_solver`: private solver or `None`;
- `vbd_solver`: private solver or `None`;
- `ownership`: resolved entity partition.

Option dictionaries are backend-specific. V2 raises when an option is
incompatible with the selected path; it does not silently construct an unused
solver or accept a contradictory rigid-integration mode.

## 12. Validation coverage

`newton/tests/test_mjvbd_v2.py` currently verifies:

- ownership partitioning and rejection of incomplete joint trees;
- all six dispatch paths;
- pure-MuJoCo contacts and its isolated sleeping option;
- coupled sleeping rejection and strict no-feedback behavior;
- kinematic soft numerical agreement with the original MJVBD path;
- dynamic VBD rigid-body contact;
- scene-based module pruning and optimized-soft device-side self-contact
  selection;
- sparse world-shape contacts;
- sparse point-contact AABB equivalence for moving primitives, offset-origin
  meshes, runtime shape flags, and CUDA Graph replay;
- private full-contact dispatch, contact equivalence, runtime margins, and
  CUDA Graph replay;
- device material selection and single-graph replay;
- one final particle-output copy per step and full-VBD surface group selection;
- pneumatic authoring, pressure, reset, world masks, coupled state transfer,
  and zero allocation for non-pneumatic scenes.

Scenario demos remain complementary validation, not automated guarantees.
Visual similarity, application FPS, and long-horizon stability for the
DexForce T-shirt and bag scenes must be measured explicitly when a change
touches dispatch, proxy synchronization, collision generation, or VBD kernels.
The historical reference hashes are recorded in `BASELINES.md`.

## 13. Representative examples

- `newton/examples/mjvbdv2/example_mjvbd_v2_cloth_twist.py`:
  joint-free pure VBD.
- `newton/examples/mjvbdv2/example_vbd_inflatable_bag_v0.py` and
  `example_vbd_inflatable_bag_v1.py`: pneumatic shells.
- `newton/examples/mjvbdv2/example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00.py`:
  prescribed bimanual kinematic cloth interaction.
- `newton/examples/mjvbdv2/example_cloth_mjvbd_v2_dynamic_dexforce_bimanual_fold_tshirt_waic_house.py`:
  dynamic MuJoCo articulation with one-way VBD cloth coupling.
- `newton/examples/mjvbdv2/example_mjvbd_v2_supermarket_plastic_bag.py`:
  joint-free cloth and VBD rigid bodies.
- `newton/examples/mjvbdv2/example_mjvbd_v2_dexforce_bimanual_plastic_bag_rod_handoff.py`:
  kinematic hands, bag contacts, and staged support handoff.

## 14. Maintenance rules and extension boundary

When extending MJVBDV2:

1. Preserve the ownership partition before adding a new interaction.
2. Add a backend only when an existing backend cannot express the required
   state ordering without unnecessary modules.
3. Keep the dispatch decision at construction time and expose it through
   `features.backend`.
4. Reject incompatible options rather than changing their meaning by path.
5. Keep ordinary, non-pneumatic scenes free of pneumatic allocation and kernel
   loading.
6. Do not add VBD-to-MuJoCo feedback under the current solver name. True
   bidirectional coupling changes physics semantics, proxy iteration, sleeping,
   and stability requirements and needs a separate explicit design.
7. Add focused unit tests for the new path and scenario measurements for every
   affected flagship demo.
8. Record every performance experiment, including rejected candidates, in
   `OPTIMIZATION_LOG.md` with its affected path, A/B setup, result, and
   numerical constraints.

The current design intentionally optimizes a one-way robot/environment
interaction model. It should not be described as a general bidirectional
MuJoCo/VBD multiphysics solver.
