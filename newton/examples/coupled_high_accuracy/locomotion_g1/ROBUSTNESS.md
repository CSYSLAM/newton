# G1 Robustness Training Plan

This document records the `robustness_v1` implementation and the staged path from the completed
velocity bootstrap to a candidate for hardware-oriented evaluation. It follows the randomization
semantics used by the repository's Unitree locomotion environments while preserving the G1 480D
actor observation and 29D action ABI.

The first stage-1 attempt and the subsequent effort-bootstrap recovery have completed and were both
rejected by their gates. See [TRAINING_HISTORY.md](TRAINING_HISTORY.md) for measured results and the
current promotion decision.

## Starting Checkpoint

The completed 5090 bootstrap run is:

```text
runs/synreal_g1_velocity_bootstrap/2026-09-10_09-26-05/model_3999.pt
SHA256 c40e66f544c9b4aec7bdab9a8264dc011e47c9c716f9c5613eea77a40ec8f81f
```

Its iteration 3800-3999 means were:

| Metric | Value |
| --- | ---: |
| Episode length | `992.9 / 1000` steps |
| Body forward velocity | `0.501 m/s` |
| Forward velocity absolute error | `0.052 m/s` |
| Alternating touchdown fraction | `97.9%` |
| Left/right air fraction | `42.3% / 55.9%` |
| Raw action over 3 fraction | `3.56%` |
| Mean episode maximum raw action | `4.47` |
| Torque saturation fraction | `0.21%` |

The checkpoint is a good walking bootstrap, but the 13.6 percentage-point air-time imbalance and
growing raw action magnitude make it unsuitable as the direct input to strong randomization. It
must first pass both straight-symmetry stages.

If enabling the actual effort envelope together with Stage 1 randomization collapses locomotion,
run `--effort-bootstrap` from the selected straight-symmetry Stage 2 actor. This transition keeps
the forward-only command distribution and nominal actuator parameters, enables only the effort
limiter, and starts exploration at `0.10`. It is a recovery bridge, not an additional randomization
stage; its failed predecessor and recovery result must both remain in the enhancement report.

## Invariants

- No mass or COM randomization.
- No raw action clipping or target slew limit.
- No actor gait phase and no hidden phase reward.
- Policy timing remains 50 Hz with 5 ms physics and decimation 4.
- Observation noise and friction randomization remain enabled as in the preceding training.
- Each stage loads actor tensors only. Critic, optimizer, PPO iteration, and exploration state are
  fresh; robustness exploration standard deviation starts at `0.20`.

## Implemented Semantics

Actuator parameters are sampled independently per environment, joint, and episode. Randomized PD
gains are written to the Newton model. The desired joint position is converted to a capture-safe
limited target so motor-strength variation changes the actual effort envelope, not only the torque
diagnostic. This limiter is active only in robustness presets, leaving earlier checkpoints on their
original control path.

Joint-zero offset is added to both the actuator neutral target and observation reference. Action
delay is sampled per environment and selects from a FIFO of raw policy outputs. The actor still
observes the current raw output as its previous-action term, matching the repository's Unitree
deployment convention.

Pushes add a sampled horizontal world-frame velocity disturbance at a separately sampled interval
for each environment. They are enabled only in robustness stage 3.

## Training Sequence

### A. Straight-Symmetry Stage 1

Start from `model_3999.pt` and run 1000 to 1200 iterations:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --straight-symmetry-stage 1 --num-envs 512 --iterations 1200 \
  --actor-init runs/synreal_g1_velocity_bootstrap/2026-09-10_09-26-05/model_3999.pt \
  --asset /absolute/path/to/g1_29dof_rev_1_0.usd
```

Advance only if mean forward error stays below `0.10 m/s`, episode length exceeds 950 steps,
left/right air-fraction difference falls below 8 percentage points, and torque saturation remains
below 1%.

### B. Straight-Symmetry Stage 2

Actor-initialize from the selected stage-1 checkpoint and run 1000 to 1500 iterations. This stage
introduces small mirrored lateral and yaw corrections. Advance when air-fraction difference is below
5 percentage points and deterministic SynReal plus strict MuJoCo rollouts satisfy the gates in
`STRAIGHT_SYMMETRY.md`.

### C. Robustness Stage 1

| Parameter | Range |
| --- | ---: |
| Stiffness scale | `[0.95, 1.05]` |
| Damping scale | `[0.90, 1.10]` |
| Motor strength | `[0.95, 1.05]` |
| Joint-zero offset | `[-0.010, 0.010] rad` |
| Action delay | `0-1` policy ticks |
| Push | disabled |

Run 800 to 1000 iterations from the selected straight stage-2 actor. This is also the first stage
that enforces the configured effort envelope in the simulation control path. Reject checkpoints if
torque saturation exceeds 2%, episode length falls below 900 steps, or straightness degrades by more
than 25% relative to the stage-2 source.

### D. Robustness Stage 2

| Parameter | Range |
| --- | ---: |
| Stiffness scale | `[0.90, 1.10]` |
| Damping scale | `[0.85, 1.15]` |
| Motor strength | `[0.90, 1.10]` |
| Joint-zero offset | `[-0.015, 0.015] rad` |
| Action delay | `0-2` policy ticks |
| Push | disabled |

Run 1000 to 1200 iterations from the selected robustness stage-1 actor. Evaluate fixed slices at
minimum and maximum motor strength, gain scale, and delay rather than relying only on randomized
training means.

### E. Robustness Stage 3

| Parameter | Range |
| --- | ---: |
| Stiffness scale | `[0.80, 1.20]` |
| Damping scale | `[0.80, 1.20]` |
| Motor strength | `[0.85, 1.15]` |
| Joint-zero offset | `[-0.020, 0.020] rad` |
| Action delay | `0-2` policy ticks |
| Push interval | `[8, 12] s` |
| Push velocity | each horizontal axis in `[-0.20, 0.20] m/s` |

Run 1200 to 1500 iterations from the selected robustness stage-2 actor. Do not increase pushes to
the Unitree reference range of `[-0.5, 0.5] m/s` until this stage recovers the pre-push straightness
and action metrics.

## Robustness Launch Template

Replace the source with the checkpoint selected from the preceding stage:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --robustness-stage 1 --num-envs 512 --iterations 1000 \
  --actor-init /absolute/path/to/selected_previous_stage.pt \
  --asset /absolute/path/to/g1_29dof_rev_1_0.usd
```

Use `--robustness-stage 2` and `--robustness-stage 3` only after the preceding gate passes. Every
run writes its preset, full randomization ranges, source checkpoint path, and source SHA256 to
`environment.json`.

## Low-Frequency Automatic Sequence

The repository includes `auto_train.py` for unattended stage transitions. It sleeps for the
estimated duration of each stage, checks once at the end (with a 15-minute safety recheck only if
the process is still alive), runs a 30-second headless MuJoCo evaluation, appends a JSON record to
`runs/g1_enhancement_report.md`, and starts the next stage only when the gate passes. The controller
does not change the asset, add mass/COM randomization, clip actions, or add target slew limits.

Example on the 5090 host, starting from the already-running Stage 1:

```bash
tmux new-session -d -s g1_auto_controller \
  '.venv/bin/python examples/locomotion_g1/auto_train.py --initial-wait 3300 \
   > runs/g1_auto_controller.log 2>&1'
```

The initial wait should be set to the remaining Stage 1 estimate. For a new complete sequence,
omit `--initial-wait`; the controller starts with the bootstrap checkpoint and waits for each
stage it launches.

## Required Review

For every 100-iteration checkpoint, compare forward/lateral/yaw error, episode length, bilateral
contact duty and air time, repeated touchdowns, raw and arm action magnitude, delayed-action delta,
joint velocity limits, torque saturation, and push recovery. A robustness reward increase is not a
sufficient promotion criterion if action magnitude or left/right imbalance regresses.
