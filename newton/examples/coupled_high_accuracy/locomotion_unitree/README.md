# Unitree RL Lab Go2 on SynReal

This package ports the active Go2 velocity task from Unitree RL Lab commit
`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` to SynReal/Newton.

See [SIM2REAL_PLAN.md](SIM2REAL_PLAN.md) for the staged actuator
randomization, validation, deployment-export, and Go2 hardware safety plan.

It preserves the 45D actor observation, 60D privileged critic, 50 Hz action
rate, Unitree Go2HV torque-speed curve, and upstream PPO architecture. The
current aggressive training configuration intentionally changes the initial
command range and several reward weights; those differences are recorded in
the sim2real plan. Base-mass randomization is intentionally disabled.
Contact-force thresholds are binary geometric contact checks because SynReal
does not currently expose solved per-contact impulses.

Validate configuration:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/train.py --num-envs 512
```

Validate observations, contacts, rewards, and termination signals:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/validate.py
```

Train headless with 512 environments:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/train.py \
  --num-envs 512 --friction-randomization batch --run
```

`per-env` friction matches Unitree RL Lab but disables SynReal's shared-world
material layout and does not fit 512 worlds on a 12 GB GPU. `batch` samples one
coefficient for the whole run and retains scalable shared topology. `fixed`
uses coefficient 1.0. Other reset, observation, and push randomization remains
per environment in every mode.

Unitree RL Lab enables self-collision. On the current 12 GB GPU, combining it
with randomized friction makes the native solver fail during a 512-world PPO
rollout, so this port disables it by default. The closer upstream mode remains
available for smaller runs:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/train.py \
  --num-envs 64 --friction-randomization per-env \
  --enable-self-collisions --run
```

## MuJoCo sim2sim

`mujoco_eval.py` does not substitute a third-party Go2 URDF. It parses the
exact USD used by training with the same fixed-joint collapse and collision
options, then constructs an equivalent MuJoCo model from the parsed bodies,
inertials, joints, collision primitives, and visual meshes. Before playback it
checks the asset topology, joint limits, primitive geometry, inertials, and FK
at zero and default joint positions.

Run a finite headless validation and save its machine-readable summary:

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/mujoco_eval.py \
  --checkpoint /path/to/model_1800.pt --headless --duration 10 \
  --command-vx 0.3 --summary artifacts/go2_mujoco_1800.json
```

Open the interactive MuJoCo viewer (close the window to stop):

```bash
/home/ubuntu/.local/bin/synreal-run examples/locomotion_unitree/mujoco_eval.py \
  --checkpoint /path/to/model_1800.pt --duration 0 --command-vx 0.3
```

The robot asset and policy/control ABI are shared, but MuJoCo and SynReal use
different constraint and contact solvers. That remaining physics difference is
the quantity this sim2sim test is intended to expose.
