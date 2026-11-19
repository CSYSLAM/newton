# G1 Straight And Symmetric Walking Plan

This document records the `straight_symmetry_v1` training preset implemented after the
forward-velocity bootstrap. Both stages have now been run on the 5090 host. Their checkpoints,
metrics, MuJoCo results, and promotion decisions are recorded in
[TRAINING_HISTORY.md](TRAINING_HISTORY.md).

## Objective

The velocity bootstrap established sustained forward locomotion, but `model_1500.pt` exposed two
deployment-relevant defects:

- The SynReal rollout had unequal left and right air fractions.
- The strict-transfer MuJoCo rollout remained finite for 60 seconds but accumulated substantial
  lateral and yaw drift under a zero lateral/yaw command.

This preset targets straightness and bilateral balance before adding stronger randomization. It
does not add action clipping, action target rate limits, mass randomization, COM randomization, or
an externally supplied gait phase.

## Invariants

- Policy input remains the Unitree RL Lab-compatible 480D term-major, five-frame history.
- Policy output remains 29 unbounded raw joint-position actions with scale `0.25`.
- The phase gait reward remains disabled because phase is not an actor observation.
- The training asset, PD gains, effort limits, self-collisions, physics step, and policy step are
  unchanged from the bootstrap.
- Existing light action and arm-action L2 terms remain monitoring-oriented regularizers, not hard
  deployment constraints.
- Torso mass and COM are not randomized.

## Initialization

Each stage requires `--actor-init`. Only actor tensors are loaded; critic, optimizer, PPO iteration,
and exploration state are fresh. The initial exploration standard deviation is `0.30`.

The source checkpoint must be selected by deterministic SynReal and MuJoCo evaluation rather than
iteration number alone. Candidate bootstrap checkpoints should be compared for forward-speed
error, lateral/yaw drift, bilateral air-time balance, action magnitude, and torque saturation. The
selected path and SHA256 are written to the new run's `environment.json` at launch.

## Stage 1: Straight-Line Correction

Recommended duration: 800 to 1200 PPO iterations with 512 environments.

| Command mode | Fraction | Range |
| --- | ---: | --- |
| Straight | 85% | `vx=[0.25, 0.70]`, `vy=0`, `yaw=0` |
| Slow | 10% | `vx=[0.05, 0.25]`, `vy=0`, `yaw=0` |
| Standing | 5% | all commands zero |

The command sampler creates mirrored pairs. Paired commands share `vx` while lateral and yaw signs
are reversed. Friction remains in `[0.3, 1.0]`, but all collision shapes within one environment use
the same sampled coefficient during this stage. This removes a persistent left/right material
imbalance while retaining the official friction range.

Reward changes relative to the velocity bootstrap are:

| Term | Weight |
| --- | ---: |
| `track_lin_vel_xy` | `4.0` |
| `track_ang_vel_z` | `1.5` |
| `lateral_vel_error_l2` | `-0.5` |
| `yaw_rate_error_l2` | `-0.5` |
| `foot_duty_imbalance` | `-0.25` |
| `repeated_touchdown` | `-0.5` |
| `feet_clearance` | `0.5` |

Foot-duty imbalance compares average left and right contact over a 0.8-second sliding window. It
does not compare instantaneous joint poses or demand simultaneous foot contact, so an alternating
gait is still valid. Repeated-touchdown applies only when two consecutive single-foot touchdown
events use the same foot.

Example launch after checkpoint selection:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --straight-symmetry-stage 1 --num-envs 512 --iterations 1200 \
  --actor-init runs/synreal_g1_velocity_bootstrap/<run>/<checkpoint>.pt \
  --asset /absolute/path/to/g1_29dof_rev_1_0.usd
```

## Stage 2: Bidirectional Correction

Stage 2 begins only after stage 1 satisfies the straight-line gates. Recommended duration: 1000 to
1500 PPO iterations, actor-initialized from the selected stage-1 checkpoint.

| Command mode | Fraction | Range |
| --- | ---: | --- |
| Straight | 60% | `vx=[0.20, 0.75]`, `vy=0`, `yaw=0` |
| Slow | 15% | `vx=[0.05, 0.25]`, `vy=0`, `yaw=0` |
| Standing | 10% | all commands zero |
| Turn correction | 10% | `vx=[0.20, 0.60]`, `yaw=[-0.15, 0.15]` |
| Lateral correction | 5% | `vx=[0.20, 0.60]`, `vy=[-0.08, 0.08]` |

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --straight-symmetry-stage 2 --num-envs 512 --iterations 1500 \
  --actor-init runs/synreal_g1_straight_symmetry_s1/<run>/<checkpoint>.pt \
  --asset /absolute/path/to/g1_29dof_rev_1_0.usd
```

## Diagnostics And Gates

The environment logs the existing walking and action metrics plus:

- body `vy`, `vy` absolute command error, and yaw-rate absolute command error;
- integrated lateral velocity error and integrated yaw-rate error;
- absolute integrated yaw error per forward meter;
- left/right rolling contact duty and their imbalance;
- episode left/right air-fraction imbalance;
- alternating and repeated single-foot touchdown fractions.

A checkpoint advances only after deterministic SynReal and strict-transfer MuJoCo evaluation at
multiple seeds and friction slices. The nominal targets are:

| Gate | Target |
| --- | ---: |
| Continuous rollout | at least 60 seconds |
| Mean forward-speed absolute error | below `0.10 m/s` |
| Mean lateral-speed absolute error | below `0.05 m/s` |
| Mean yaw-rate absolute error | below `0.04 rad/s` |
| Heading drift | below 10 degrees per 10 meters |
| Left/right episode air-fraction difference | below 5 percentage points |
| Repeated same-foot touchdown fraction | below 2% |
| Nominal torque saturation | below 1% |

Action clipping and target slew limits remain deferred. Raw action-over-3, maximum action, arm
action, joint-velocity-limit, and torque-saturation metrics must still be reviewed so this stage
does not silently make the eventual deployment envelope worse.

## Later Robustness Work

After both stages pass, continue with the explicit actuator, latency, and push presets described in
[ROBUSTNESS.md](ROBUSTNESS.md). Mass and COM randomization remain excluded by project decision.
