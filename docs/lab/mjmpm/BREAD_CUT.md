# W1 MJMPM bread cutting

```powershell
uv run --no-sync -m newton.examples mjmpm_w1_bread_cut
uv run --no-sync -m newton.examples mjmpm_w1_bread_cut --viewer null --test
uv run --no-sync python -m unittest newton.tests.mjmpm.test_bread_cut newton.tests.mjmpm.test_bread_cut_cohesion
uv run --no-sync python scripts/record_mjmpm_bread_cut.py --video --width 1920 --height 1080 --output-dir output/bread_cut
```

The 11-second trajectory closes the right gripper on a mounted knife, moves
above the loaf, cuts with a small sawing stroke, separates the pieces with
the flat side and withdraws. The robot remains dynamic in MuJoCo; material
reaction feeds back through articulated mobility. The knife is rigidly
mounted on the wrist, including its inertia. This example does not pick up
a free knife through friction.

Bread defaults to live material particles. Crust and crumb colour depend
only on the original loaf boundary, not the planned cut. There is no initial
visible seam or predefined physical split. Knife and board use Blender
meshes. `--visual-style particles` also shows their collision geometry;
`--bread-render surface` enables the older, optional surface embedding
patches for comparison, which are not loaded by the default renderer.

## Physical model

This is an experimental **MPM plus cuttable elastoplastic cohesion** model,
not a general fracture implementation inside the Newton MPM solver.

There are 7,376 particles at 5 mm spacing and density 280 kg/m^3. MPM handles
bulk elasticity/plasticity and tool/board contact on a 6 mm grid. The default
modulus is 4 kPa, Poisson ratio 0.25, pressure yield 5 kPa, shear yield 3 kPa
and tensile yield ratio 0.3. There are six substeps per 60 Hz frame and up to
50 MPM GS iterations per substep.

The original WAIC-style volume weakening alone created loose grains at the
cut. The revised model also connects each particle to its face, edge and
corner lattice neighbours. The spring stiffness is 20 N/m for face neighbours
and scales inversely with the connection rest length for diagonals.
Relative-velocity damping is 0.002 N*s/m for face neighbours. Each substep
uses a linearized backward-Euler velocity solve, with block Jacobi
preconditioning and 24 fixed CG iterations. The MPM solve follows those
internal impulses, so contact sees the cohesion response. No post-contact
position projection, prescribed particle poses or particle deletion is used.

The network begins as one connected body. A physical downward blade-edge
sweep severs intersected connections. The cutter follows the curved blade
edge, with a half-particle support allowance. A sideways flat-face push and
withdrawal do not cut additional connections. This distinction prevents the
completed slice from being cut into extra fragments while it is moved aside.
All connections remaining inside each piece retain cohesion.

MPM tensile and shear yield strengths are weakened in a 5.5 mm band around
the physical edge. The additional connections retain within-piece cohesion
there, instead of allowing those particles to become loose grains.
`--no-blade-damage` disables this rheology weakening and the local plastic
compaction, while retaining the cuttable network, for comparison.

PIC transfer is used for this split particle-impulse integration. A trial
with APIC produced a high-frequency instability during the side push; this
example does not integrate separate affine momentum for the cohesion stage.
MPM and CG iterations are graph-capturable; there is no host convergence read
inside a captured step. Damage, connection and plastic histories belong to
the example and require a new example instance to reset.

## Grip and permanent indentation

The authored handle and collision handle share a 9 mm half-width. The fingers
close to a geometry-sized opening rather than zero. The opening includes the
actual W1 visual pad overhang, rivet envelope and drive tracking allowance.
At initialization the example measures pad/handle geometry in the tool frame;
test mode checks the resulting clearance throughout the trajectory.

In the marked cutting zone, compressed connections use a return map with a
0.5% elastic compression threshold. Plastic shortening is irreversible and
capped at 60% of the initial length. Broken connections are excluded. This
retains a small crushed notch after unloading, while the elastic part still
recovers. Rest-length changes arise from measured compression; particle
positions are not directly lowered or animated.

The local values are configurable with `--cut-plastic-yield` and
`--cut-plastic-limit`. The new defaults retain about 2 mm of upper indentation;
the earlier 1.5% threshold and 30% cap retained about 1.36 mm. Higher caps do
not translate linearly to geometric notch depth. Thresholds below apply to
the default scene; alternate materials and plasticity settings need their
own validation.

## Acceptance

`test_post_step` rejects nonfinite state, particle speeds over 3 m/s and
finger/handle penetration. From 7 seconds onward it also requires exactly
two topological material components and zero spatially detached samples in
each component, at every tested frame. Spatial connectivity uses a radius of
1.8 particle spacings and counts samples outside each piece's largest patch.

After 660 frames, `test_final` additionally checks capacities, support by the
board, dynamic MuJoCo ownership, and unchanged particle count/mass. Both
components must contain at least 10% of the samples. Retained average opening
must exceed 2 mm, lateral elastic recovery 0.5 mm, and the retained upper
notch 1.5 mm when cut-zone plasticity is enabled.

Opening is twice the mean signed lateral displacement of initially near-cut
samples, not a minimum crack gap. Recovery compares the loaded opening at
7.8-8.6 seconds with 10-11 seconds. Notch depth compares the same physically
affected upper samples with their settled positions at 3 seconds, before
tool contact.

Unit tests check the exact two-particle implicit response, internal momentum
conservation, removal of severed-connection forces, captured solves, a
one-to-two component edge cut, the absence of extra cuts during a flat push,
detached-sample detection, bounded irreversible compaction, and absence of
position projection. Additional tests cover the authored blade geometry and
default particle rendering. CPU and available CUDA tests are included.

The recorder produces an 11-second video, seven stills, `metrics.json` and
`states.npz`. The JSON includes validation measurements. The state archive
includes actual particle positions/stress at sampled frames, final connection
topology, plastic rest lengths and particle damage. Stills and video are raw
Newton renders; Blender is used for asset authoring.

The shared MPM grid can still exchange momentum between nearby cut surfaces.
This is not CPIC, a multi-field crack/contact solver, or a calibrated
fracture-energy model. Acceptance applies to this trajectory and material.

Latest validation on RTX 5060 Ti, 2026-10-23: all 660 frames passed. The two
pieces contain 2,204 and 5,172 samples; detached samples remain zero at every
tested post-cut frame. The retained notch is 1.94 mm, lateral recovery is
3.56 mm, retained average opening is 9.67 mm, and the minimum visual grip
clearance is 0.399 mm. All eight relevant unit/geometry tests and pre-commit
checks passed. The existing rice-cooker scene also passed a six-frame CUDA
graph smoke check with finite state and valid capacities.
