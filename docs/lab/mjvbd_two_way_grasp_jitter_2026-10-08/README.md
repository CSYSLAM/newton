# Two-way MJVBDV2 soft-grasp jitter

The `mjvbd_v2_w1_squeeze_grasp` example commands both W1 V030 grippers fully
shut on a rigid block and a tetrahedral soft cube. With two-way coupling at
commit `8896857f`, the gripper on the soft cube visibly pulsed while holding
the cube in the air: it was pushed open and then clamped back down, about
15 times per second. The rigid-block gripper was steady.

This note records the measurements, the cause, the accepted fix, and the
approaches that were tried and rejected. Raw numbers are in
[`results.json`](results.json). Measurements were taken on an NVIDIA GeForce
RTX 5060 Ti with the example's 10 substeps per 60 Hz frame and 12 VBD
iterations.

## Measure

[`jitter_probe.py`](jitter_probe.py) runs the example without CUDA graph
capture, records the finger-1 opening of both grippers after every substep,
and reports statistics over the lifted hold from 4.6 s to 7.3 s. Arguments are
forwarded to the example parser:

```bash
uv run docs/lab/mjvbd_two_way_grasp_jitter_2026-10-08/jitter_probe.py
uv run docs/lab/mjvbd_two_way_grasp_jitter_2026-10-08/jitter_probe.py --finger-damping 0 --proxy-mass-scale 1
```

Before the fix, the soft-side finger oscillated as a clean sine between about
23 and 31 mm with a 0.066 s period (40 substeps, about 15 Hz): 8.88 mm peak to
peak, 1.83 mm standard deviation. The rigid side moved 0.24 mm peak to peak.

## Cause

Two effects combine:

1. **The finger is undamped while it squeezes.** Its position target is fully
   closed, so at the 26 mm contact position the drive asks for about 26 N and
   is clipped to the URDF's 10 N effort limit. MuJoCo clips the whole drive
   output, including the `kd * velocity` damping term, so the finger acts as a
   constant 10 N push with no damping. The 15 Hz frequency matches the finger's
   0.33 kg (link plus 0.3 kg armature) on a spring of about 2.9e3 N/m,
   consistent with the soft cube's elastic stiffness.
2. **The feedback is late and the proxy bounces.** VBD contact wrenches reach
   MuJoCo one substep late and are blended with the previous value
   (`proxy_relaxation=0.5`). At 15 Hz this lag acts like a small negative
   damping (estimated at about -10 N*s/m) that keeps feeding the cycle. Inside
   VBD the finger proxy carries only the finger's MuJoCo effective inertia
   (about 0.4 kg), so it is thrown back by the stiff particle contacts within a
   substep, which makes the harvested force spiky.

The rigid block does not show the mode: its stiff contact pushes the frequency
high enough for VBD's contact damping to absorb it.

## Fix

The W1 two-way scenes (`newton/examples/mjvbdv2/support/w1_two_way.py`) now
default to:

- `--finger-damping 50`: 50 N*s/m of passive joint damping through
  `model.joint_damping`, which MuJoCo integrates as `dof_damping`. It sits
  outside the effort limit, like the gearbox and backdrive friction of a real
  geared gripper. With a 10 N effort it limits free closing to 0.2 m/s, within
  the URDF's 1 m/s velocity limit.
- `--proxy-mass-scale 4`: the VBD finger proxies carry four times the MuJoCo
  effective inertia (`coupling_options={"mass_scale": 4.0}`). The heavier proxy
  is no longer thrown off the particle contacts within a substep, so VBD
  resolves the squeeze mostly by deforming the cube. The fed-back force is
  still the true contact impulse: the momentum harvest measures the proxy's
  momentum change, and with gravity removed and no joints in VBD, only
  contacts act on the proxy.

Result: soft-side finger 0.60-0.65 mm peak to peak (0.09-0.11 mm std) in
three runs, rigid side unchanged at about 0.22 mm. The soft cube is indented
4.4-4.5 mm per side instead of about 3 mm. The table push, squeeze grasp, and
two-way pick-and-place examples all pass, and the example cost is unchanged
because only model parameters changed.

The squeeze example's test now requires both fingers to stay within 0.5 mm
peak to peak from 6.0 s to 7.3 s, sampled per frame. The old settings fail it
at 1.48 and 2.78 mm in two runs; the new defaults pass at 0.01 mm.

## Rejected approaches

| Change (soft side, from the 8.88 mm baseline) | Result | Why rejected |
| --- | --- | --- |
| Finger damping 20 / 50 / 100 alone | 5.01 / 3.13 / 3.06 mm | Plateaus near 3 mm; the bouncing proxy still excites the finger. |
| `proxy_relaxation=1.0` (no blending) | 6.94 mm | Less lag helps slightly but does not stop the cycle. |
| `proxy_relaxation=0.25` (more blending) | 157 mm | Unstable: the extra lag opens the fingers completely. |
| Mass scale 4 alone | 2.41 mm | Halves the jitter, still visible. |
| Mass scale 10 alone | Fingers close to 0 mm | The heavy proxy crushes the 93 g rigid block out of the grasp. |
| Mass scale 30 alone | Rigid finger at 0 mm | Same failure on the rigid side. |
| Mass scale 10 + damping 50 | 0.20 mm | Best jitter, but next to the mass-scale-10 failure. |
| Mass scale 10 + damping 50 + relaxation 1.0 | Rigid grip collapses | Confirms that the mass-scale-10 region is fragile. |

Damping 50 with mass scale 4 was chosen over the steadier mass scale 10 to
keep a margin from the rigid-grasp failure.

## Earlier two-way findings that shaped this

These came from building the two-way coupling itself and constrain the
remedies above:

- **Contact-force feedback** (re-evaluating penalty contacts at the solved
  pose) reported nearly zero force on gripper fingers, because the solve had
  already separated the light proxy from the block. Fingers crushed through.
  Replaced by the momentum harvest.
- **Lagged proxy mode** with one iteration chattered and let fingers close
  through blocks, even with relaxation 0.5 or Aitken relaxation. Two lagged
  iterations worked but halved the frame rate. Staggered mode is the default.
- **Two proxy iterations in staggered mode** let fingers close straight
  through a box; this looks like a bug in the iteration restart and is not
  fixed.
- **Stiff contacts with long substeps** are unstable: with contact
  `ke = 2e4` N/m, a two-finger gripper exploded at a 1/240 s substep and
  oscillated at 1/600 s until relaxation 0.5 was added. Light links also need
  joint armature; the W1 fingers use 0.3 kg.

## Limits of the fix

Finger damping is a physical model parameter, chosen empirically rather than
from a gripper datasheet. The proxy mass scale is a numerical coupling
parameter, not physics, and its working range is narrow and scene-dependent:
1 jitters, 4 works, 10 fails here. Other grippers and objects may need a
different value, so `SolverMJVBDV2` keeps its default of 1 and the recipe is
documented in `docs/solvers/mjvbd_v2.rst`. The underlying cause, explicit
feedback that arrives one substep late, remains. A complete fix would couple
the finger implicitly, for example through working multi-iteration staggered
coupling.
