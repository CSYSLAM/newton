# G1 Forward-Velocity Bootstrap Run

This document records the forward-only optimization started on 2026-09-10. The purpose of this
run is to move the policy out of its stable-standing optimum without changing the 480-dimensional
policy ABI, robot asset, solver settings, self-collisions, or action scale.

## Initialization

- Host: `synreal-5090-admin`
- Source run: `runs/synreal_g1_locomotion/2026-09-09_11-59-03`
- Actor checkpoint: `model_13400.pt`
- Checkpoint SHA256: `ee0ce710f5fe7c740d970b1ec9204058e2d401d54efa63f11aa5b26dc7ef46db`
- Initialization mode: actor weights only
- Critic, optimizer, PPO iteration, and action exploration standard deviation: freshly initialized
- Initial action exploration standard deviation: `0.5`

The actor-only loader deliberately does not call `OnPolicyRunner.load()`, because that method also
restores the critic and policy exploration parameter even when optimizer loading is disabled.

## Bootstrap Preset

The `velocity_bootstrap_v1` preset changes only the task and reward settings below. Defaults remain
the official Unitree RL Lab parity configuration unless `--velocity-bootstrap` is passed.

| Setting | Bootstrap value |
| --- | --- |
| Forward command | Uniform `[0.35, 0.65] m/s` |
| Lateral/yaw command | `0` |
| Standing environments | `0%` |
| Velocity curriculum | Disabled |
| Five-second velocity pushes | Disabled |
| Linear velocity tracking | `4.0` |
| Alive reward | `0.05` |
| Waist deviation | `-0.5` |
| Hip roll/yaw deviation | `-0.5` |
| Base height | `-5.0` |
| Phase gait reward | Disabled |
| Global action L2 | `-0.001` |
| Arm action L2 | `-0.003` |
| Action clipping | Disabled |

The clearance reward is replaced by a supported-swing signal. A foot receives clearance reward
only when it is moving, out of contact, near the 0.1 m target height, and the opposite foot is in
contact. A stationary foot or a flight phase receives zero clearance reward.

## Launch

The 5090 launch command is:

```bash
/home/ubuntu-jushen/.local/bin/synreal-run examples/locomotion_g1/train.py \
  --run --velocity-bootstrap --num-envs 512 --iterations 4000 \
  --actor-init \
  runs/synreal_g1_locomotion/2026-09-09_11-59-03/model_13400.pt \
  --asset \
  /home/ubuntu-jushen/newton_style3d/newton/examples/assets/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
```

- Output run: recorded after launch below.
- Log file: recorded after launch below.

## Monitoring And Gates

No action clipping is applied during this run. The environment records the raw policy action used
to generate joint targets. TensorBoard and console episode metrics include:

- `mean_body_vx`, `mean_vx_abs_error`, and integrated `forward_displacement`
- left/right air fraction, flight fraction, and single-support fraction
- alternating touchdown fraction
- fraction of raw actions with `|action| > 3`
- maximum absolute action and maximum absolute arm action
- estimated torque saturation fraction

The `|action| > 3` threshold is monitoring-only. It does not alter control. Before deployment, the
run should normally satisfy action-over-3 below 1% and estimated torque saturation below 5%.
Persistent action-over-3 above 5%, growing maximum action, or high wrist/arm action is a deployment
risk and should stop checkpoint promotion even if velocity tracking improves.

Review gates:

- By iteration 500: deterministic `vx=0.5 m/s` evaluation should exceed `0.15 m/s` and show a
  supported swing phase.
- By iteration 1500: mean forward velocity should exceed `0.30 m/s`, with low flight fraction and
  meaningful left/right touchdown alternation.
- Do not continue blindly to iteration 4000 if locomotion metrics remain flat at the first gate.

## Actual Run

- Output run: `runs/synreal_g1_velocity_bootstrap/2026-09-10_09-26-05`
- Console log: `runs/synreal_g1_velocity_bootstrap/bootstrap_actor13400_2026-09-10.log`
- Python process at launch: PID `3598896`
- Asset SHA256: `cfed730bb043c42708260f25bf3bdf49ac80deed5d5a49384721b8e01059a96e`
- Remote unit tests: 10 passed
- Pre-launch SynReal smoke: 12 steps, finite state, no reset, and no contact overflow

At iteration 23, the early stochastic rollouts reported mean body `vx=0.352 m/s`, but mean episode
length was only about 50 policy steps. This is not accepted as walking because it may represent a
forward fall. Action-over-3 was 1.56%, mean episode maximum action was 4.64, mean arm maximum was
3.77, and estimated torque saturation was 3.36%. Training remains active pending the iteration-500
deterministic checkpoint gate.
