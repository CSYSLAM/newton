# SynReal Unitree G1 Locomotion

This directory is a standalone implementation of the Unitree RL Lab G1
29-DoF velocity task. It does not import any of the repository's other
locomotion examples.

The policy and critic observation ABIs are 480 and 495 values. Five samples
are stored oldest-to-newest for each observation term, then the terms are
concatenated in Unitree RL Lab order. Physics runs at 200 Hz and the policy at
50 Hz. Reward weights and PPO settings match the upstream task.

Three contact rewards use an explicit approximation. Ground contacts are
classified geometrically every 5 ms, require two consecutive samples, and
retain three substeps of history. They are not described as a 1 N contact
sensor. Self-collision is enabled by default. Per-environment or global mass
randomization is intentionally absent. The official startup friction
randomization is retained as 64 buckets in the `0.3` to `1.0` range. Newton
provides one Coulomb coefficient rather than PhysX's separate static and
dynamic coefficients, so this is also recorded as an explicit approximation.

The contact material currently uses `ke=50000`, `kd=500`, `kf=1000`, and
ground `mu=1.0`. Robot collision-shape
`mu` values are then replaced by the retained 64-bucket randomization, while
the ground shape remains at `1.0`. The solver uses augmented-Lagrangian joint
constraints with two primal iterations, six-spring affine rigidity,
midpoint-Cayley inertia, Newton nonlinear solves, 30 nonlinear iterations, and
10 linear iterations. The shared constraint alpha remains `200`.

All commands that construct the solver must use the authenticated launcher:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py
```

Run a one-world simulation smoke test before training:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --smoke-steps 20 --num-envs 1 --disable-cuda-graph
```

Exercise the same unit-Gaussian action scale used at PPO initialization:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --smoke-steps 24 --smoke-action-std 1.0 --num-envs 2 \
  --smoke-report-interval 4
```

Start the requested 512-world training configuration:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --num-envs 512 --iterations 50000
```

The forward-only actor warm-start procedure and its recorded source checkpoint are documented in
[VELOCITY_BOOTSTRAP.md](VELOCITY_BOOTSTRAP.md).

The phase-free straight-line and bilateral-balance follow-up preset is documented in
[STRAIGHT_SYMMETRY.md](STRAIGHT_SYMMETRY.md). It is split into two explicit stages and does not add
mass randomization or hard deployment action constraints.

The subsequent actuator, latency, and push schedule is documented in
[ROBUSTNESS.md](ROBUSTNESS.md). It preserves the policy ABI and keeps mass/COM randomization and
hard action constraints disabled.

Completed stage results, rejected checkpoints, MuJoCo outcomes, and promotion decisions are
recorded in [TRAINING_HISTORY.md](TRAINING_HISTORY.md).

Inspect the model with zero actions:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/play.py \
  --seconds 20
```

Play a checkpoint deterministically:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/play.py \
  --checkpoint runs/synreal_g1_locomotion/<run>/model_1000.pt \
  --cmd-x 0.3
```

Validate the same checkpoint in native MuJoCo. The evaluator rebuilds the
MuJoCo model from the training USD and checks topology, inertials, joint
limits, and zero/default-pose forward kinematics before rollout:

```bash
uv run examples/locomotion_g1/mujoco_eval.py \
  --checkpoint runs/synreal_g1_locomotion/<run>/model_1000.pt \
  --command-vx 0.3 --duration 20
```

Use `--headless --summary result.json` for automated evaluation. A single
MuJoCo world cannot reproduce the training-time friction distribution, so it
uses the midpoint of the recorded range by default; select a deterministic
slice explicitly with `--robot-friction`.

The strict-transfer MuJoCo baseline preserves the recorded 50 Hz policy
interval, PD control, effort limits, and direct contact-parameter mapping. It
refines only the MuJoCo integration from the training snapshot's `5 ms x 4`
to `2.5 ms x 8`; armature semantics remain unchanged and are not calibrated.

New training runs record the resolved asset path and SHA256 in
`environment.json`. The MuJoCo evaluator refuses to run if that asset hash no
longer matches. Older checkpoints without the hash still receive topology,
inertial, joint-limit, and forward-kinematics contract checks.

The repository-local derived USD is selected automatically when the ignored
asset package has been installed under
`examples/locomotion_g1/assets/unitree_g1_29dof`. See [ASSET.md](ASSET.md) for
its provenance, file manifest, modified foot collisions, and installation
layout. Set `G1_ASSET_PATH` or pass `--asset` to use another location. If the
USD is unavailable, the loader falls back to Newton's downloaded 29-DoF URDF
and reports that source change. The fallback has different collision geometry
and must not be treated as a reproduction of a USD-trained checkpoint.

Useful diagnostic switches are `--disable-self-collisions`,
`--disable-cuda-graph`, and `--mesh-approximation bounding_box`. A run using
any of these switches should record the changed configuration rather than be
treated as physics parity with the default run.
