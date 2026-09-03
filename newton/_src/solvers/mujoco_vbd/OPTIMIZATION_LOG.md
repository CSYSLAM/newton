# Optimization Log

Independent performance-decision record for `SolverMuJoCoVBD`, separate from the
`mjvbd_v2` log (`DESIGN.md` sections 3.3 and 4).

## Retained optimizations (ported with the private core)

These MJVBD_V2 fast paths are carried in the copied private core and must stay
(DESIGN 3.3):

- shape-major full-surface AABB rejection
- AABB-active edge/face compaction and persistent worker
- dense rigid-side body-particle parallel reduction
- particle-color contact membership mask
- active self-contact record traversal
- surface-only CUDA tile specialization
- device-resident material selector
- world-compatible contact capacity sizing
- active-prefix/batch-gated dual update
- pneumatic device state/control
- fixed topology for one-stream whole-frame CUDA Graph

## Rejected approaches (do not reintroduce)

Judged to have no benefit in the MJVBD_V2 log and intentionally not restored:

- canonical EE pair
- post-detection VT/EE active stream
- rest-shape exclusion CSR
- Morton query order
- source-color EE row gate

## New coupling-path decisions

### Reject finite-mass one-way proxies

Removed the `one_way_kinematic_massive` backend and its public mode selector.
A finite-mass proxy that moves inside VBD while both its motion and reaction are
discarded at reconciliation is not momentum-consistent and can introduce
penetration, reset impulses, and frame-to-frame jitter. This was rejected for
correctness rather than benchmark performance.

The retained contracts are:

- one-way uses a zero-inverse-mass moving collider and allocates no feedback or
  effective-mass state;
- two-way uses MuJoCo articulated effective mass, harvests equal-and-opposite
  VBD contact wrench, reruns MuJoCo, and resynchronizes the proxy;
- strict target tracking with physical compliance uses a finite-gain MuJoCo
  drive/servo in two-way mode, never a discarded local proxy response or a
  post-VBD IK overwrite.

### Retain transactional two-way fixed-point solve

The two-way path restores the same public input and private persistent history
before every outer round. MuJoCo coordinates, actuator activation, warm-start,
applied-force inputs and `_step` are transacted; VBD pose, AVBD dual/penalty,
contact-match, Dahl and pneumatic histories use fixed device snapshots. Contact
buffers and feedback scratch are preallocated before graph capture. A failed or
non-finite round restores both cores and leaves public state unchanged.

The proxy rewind uses the full articulated inverse block: force rewinds linear
velocity and world-space torque rewinds angular velocity through the body-frame
inverse inertia. Principal moments are eigenvalue-clamped before installation.

### Retain single-owner explicit contact routing

The VBD collision pipeline uses the construction-time explicit pair stream
`V-V union M-V`; MuJoCo collision flags hide VBD-owned shapes. MuJoCo-owned
joints are disabled in the VBD overlay and VBD-owned joints are disabled in the
MuJoCo overlay. This prevents duplicate cross-contact and duplicate joint solves
while keeping stable global body ids.

Soft point/edge/face feedback calls the exact VBD contact reaction function;
rigid feedback calls AVBD's contact-force collector. Both write Newton spatial
wrenches in `(force, torque)` order.

### Retain fixed-topology overflow diagnostics

Every collision capacity involved in the two-way path is latched into public
device-side diagnostics: broad phase, split GJK/manifold work, mesh/triangle
narrow phase, contact reduction, hydroelastic intermediates, final rigid/soft
contacts, and VBD rigid/body-particle adjacency. Candidate, contact, hydro, and
VBD groups use a small fixed number of fused kernels per outer round.

Optional narrow-phase counters use a construction-time device zero instead of
host-side branching or an active counter placeholder. The latter was tested and
rejected because primitive-only scenes were falsely reported as overflowing.
The resulting launch topology is stable under CUDA Graph capture.

### Retain transactional contact matching and live overlays

Persistent contact matching keys, claims, sticky geometry, and sorter scratch
are restored before every outer round and committed only for the selected final
round. Runtime shape/body/joint property notifications refresh the existing
overlay arrays in place, preserving ownership masks without freezing mutable
model properties or reallocating graph-visible buffers.

Collision flags may disable candidates or re-enable bits that were present at
construction. Enabling a collision bit that had no construction-time candidate
is rejected explicitly by `notify_model_changed`; rebuilding the solver is
required because silently growing fixed contact streams would invalidate their
capacities and CUDA Graph topology.

### Retain correct CPU multi-world fallback

Native MuJoCo-C owns one template-world `MjData` and cannot advance all worlds
of a two-way Newton model. If a CPU multi-world solve explicitly requests
`use_mujoco_cpu=True`, the adapter emits a warning and selects MuJoCo Warp on
CPU with `separate_worlds=True`. This keeps every world independent and applies
feedback to every articulation; it also avoids maintaining an array of native
MuJoCo solvers in the coupling layer.

Validation retained with the change:

- CPU zero-contact 1-vs-4-round transaction equivalence;
- CPU point-soft contact with non-zero robot joint response;
- CPU rigid M-V contact with motion on both sides;
- CPU point/edge/face feedback, contact-filter/static routing, live shape-flag
  refresh, and two-world equal-response coverage;
- CPU non-finite feedback abort/rollback;
- CPU collision and VBD overflow detection with pre-commit rollback;
- CUDA point-soft two-round smoke on RTX 5060 Ti, including non-zero feedback,
  robot response, contact matching, zero overflow flags, and successful CUDA
  Graph capture plus two replays.

## MJVBD_V2 demo parity ports (2026-09-01)

The MJVBD_V2 plug-insertion commit `79cceb2c` and tablecloth-placement commit
`3f29191a` were first cherry-picked unchanged. Independent copies then replaced
only the public solver integration with explicit one-way `SolverMuJoCoVBD`
dispatch; neither copy imports the original demo or the MJVBD_V2 solver.

Warm-cache RTX 5060 Ti benchmarks used the authored scene parameters, CUDA
Graph enabled, null viewer, 300-frame limit, and a 3-second measurement window:

- realtime plug insertion: MJVBD_V2 52.5 FPS; SolverMuJoCoVBD 52.1 FPS
  (-0.8%);
- bimanual tablecloth placement: MJVBD_V2 54.9 FPS; SolverMuJoCoVBD 55.7 FPS
  (+1.5%).

Numerical A/B checks used identical models, inputs, and CUDA device:

- plug, eight eager frames at two substeps/two VBD iterations: rigid contact
  count 5 vs. 5; all body poses, joint coordinates/velocities, plug pose, and
  plug velocity matched exactly. The only difference was `4.32e-5` in one
  prescribed robot-link output velocity, which does not feed the dynamic plug;
- tablecloth, three eager authored frames: soft contact count 361 vs. 361 and
  particle position, particle velocity, joint position, and joint velocity all
  matched exactly.

The short benchmark window is intended as a regression A/B, not an absolute
throughput claim. Both results satisfy the parity decision: no material
performance regression and unchanged simulated-object/contact behavior.

Full authored-trajectory acceptance was also run with CUDA Graph enabled and
the null viewer. Both the original and independent plug demos passed their
660-frame grab/raise/align/insert/release/retract assertions. Both the original
and independent tablecloth demos passed their complete placement and final
geometry assertions. The measured wall times were 130.6 s vs. 58.9 s for the
plug pair and 135.2 s vs. 119.0 s for the tablecloth pair; these include process
startup and Warp compilation/cache effects and are therefore recorded as
acceptance timing, not as the controlled performance comparison above.

## Additional parity ports and normalized names (2026-09-01)

The five initial acceptance-demo filenames and commands dropped the temporary
`final00` suffix. References in the README, structural tests, and design matrix
were updated together; the MJVBD_V2 reference filenames remain unchanged.

Three more MJVBD_V2 scenes were copied into standalone MuJoCo/VBD modules. The
cloth-twist and gear-crusher scenes explicitly select the no-articulation
`pure_vbd_soft` branch. Realtime chair pushing explicitly selects
`one_way_kinematic_full`. None imports an MJVBD_V2 demo, solver, or helper; the
two cloth boundary-motion kernels are local to the new cloth module.

All three complete CUDA Graph trajectories passed their original final-state
assertions: 300 frames for cloth twist, 360 frames for gear crushing, and 900
frames for realtime chair pushing. Warm-cache RTX 5060 Ti measurements used
the authored settings, null viewer, and a 3-second benchmark window:

- cloth twist: MJVBD_V2 67.6 FPS; SolverMuJoCoVBD 67.6 FPS (0.0%);
- gear crusher: MJVBD_V2 17.0 FPS; SolverMuJoCoVBD 16.9 FPS (-0.6%);
- realtime chair push: MJVBD_V2 52.7 FPS; SolverMuJoCoVBD 53.3 FPS (+1.1%).

These results meet the parity decision: all behavior assertions pass and each
independent solver path remains within normal short-window measurement noise of
the MJVBD_V2 reference.

## Two-way soft-contact robustness (2026-09-02)

Retained three coupled changes: reconstruct VBD proxy begin poses from MuJoCo
end states so both cores solve one physical interval, append velocity-aware
speculative point contacts for separated M-V pairs predicted to cross during
the substep, and persist scalar augmented-Lagrangian normal multipliers through
the stable `soft_contact_tids` candidate map. The multiplier history participates
in the outer-round transaction and therefore commits only once per substep.

The deterministic acceptance scene is a finite-mass prismatic MuJoCo finger
moving at 3 m/s toward a 2x2x2-cell tetrahedral VBD block. It starts with a
35 mm gap and travels 50 mm during the first 1/60 s step:

- legacy penalty path: 0 first-step contacts and less than 1e-6 m particle
  displacement;
- speculative + AL path: 9 first-step contacts, more than 1e-4 m deformation,
  smaller measured first-step penetration, nonzero multiplier history, and a
  second-step equal-and-opposite feedback force that reduces joint velocity
  below 2.5 m/s;
- speculative-only A/B: second-step joint velocity was 2.318 m/s, versus
  1.097 m/s with AL warm starting; this isolates the retained multiplier's
  effect from speculative candidate generation.

Warm-cache RTX 5060 Ti null-viewer measurements used the dedicated acceptance
scene, four coupling rounds, four VBD iterations, and a 3-second window:

- robust path: 51.3 FPS;
- `--legacy-contact`: 52.3 FPS;
- measured cost: 1.9% in this deliberately tiny launch-bound scene.

The fixed contact capacity is unchanged because actual and speculative
predicates are disjoint for every candidate. CUDA Graph capture and two replays
pass with the new launches and persistent history buffers. The first version is
intentionally limited to particle-shape point candidates used by volumetric
soft bodies; edge/face speculative full-surface contact remains unchanged.

## Standalone multiphysics scene reproduction (2026-09-03)

Added an independent `SolverMuJoCoVBD` acceptance scene derived from the
generic multiphysics proxy-coupled example. It does not import
`SolverCoupledProxy` or its example module.

The proxy-coupling reproduction preserves the original three free rigid boxes,
three-link pendulum, 30x30 pinned cloth, three tetrahedral bodies, eight
substeps, and finite-mass two-way contact. A 30-frame CUDA Graph test observed
19 simultaneous soft contacts and a nonzero peak MuJoCo feedback-force norm.
It also rejects nonfinite state, excessive ground penetration, and deformable
bounds explosions.

The robot-hand/table impact scene exercises the rigid M-V path separately with
the complete 40-body Dexforce W1 URDF. The gravity-compensated, fixed-base
robot starts crouched with its right arm extended over a fixed VBD-owned
table. `RIGHT_J2` receives one initial 0.35 rad/s velocity and is then passive;
finite-stiffness drives hold the remaining joints. Only bounded convex parts
derived from the W1 URDF's right-hand collision meshes collide, with no
scripted velocity reversal or hidden contact proxy. Device-side substep
reductions preserve the acceptance metrics under CUDA Graph capture. With 24
substeps, eight coupling rounds, and compliant rigid contact, the 60-frame
acceptance measured 5.39 mm maximum signed contact penetration, an 800 N peak
feedback-force norm, and a 0.39 m/s upward wrist rebound. The wrist never
passed the tabletop, and the VBD table's fixed-joint pose drift remained below
0.01 mm.

## Microduck soft-ball kick tuning (2026-09-03)

Added a fixed-base Microduck with fourteen dynamic MuJoCo servo joints kicking
a 163-particle, 320-tetrahedron VBD ball. The visible `sole_right.stl` is also
the sole particle collider; the scene has no hidden contact proxy or scripted
ball force. Acceptance requires a right-sole contact, nonzero two-way feedback,
0.5--12 mm deformation, at least 30 mm forward travel, and no ground
penetration beyond 6 mm.

Warm-cache RTX 5060 Ti measurements used CUDA Graph, a null viewer, and 180
frames. The tuple is `(substeps, coupling iterations, VBD iterations)`:

- `(8, 5, 12)`: about 4 FPS, 3.14 mm deformation, 207 N peak feedback,
  94.6 mm forward travel;
- `(4, 3, 8)`: 24.7 FPS, 7.95 mm deformation, 13.1 N peak feedback,
  83.4 mm forward travel;
- `(4, 2, 8)`: 36.7 FPS, 6.38 mm deformation, 15.4 N peak feedback,
  98.7 mm forward travel;
- `(3, 2, 6)` and `(2, 2, 6)`: 61.7--91.9 FPS, but 19.7--25.4 mm collapse
  and 193--302 N peak feedback, so both were rejected;
- `(3, 3, 8)`: 33.0 FPS, but 14.5 mm deformation and a 904 N feedback spike,
  so it was rejected.

The retained default is `(4, 2, 8)`: it is roughly nine times faster than the
first conservative configuration while preserving the cleanest measured
impact response and passing all contact, deformation, travel, and penetration
checks.
