# Locomotion

This directory is the original phase-free Go2/G1 locomotion pipeline restored
from the 2026-09-06 17:08 training snapshot. The actor receives only robot
proprioception and velocity commands; it has no gait clock, Fourier reference,
or reference action injected outside the neural network.

Go2 uses a 45D observation and 12D bounded action. Its execution contract keeps
the Newton asset, 50 Hz policy, 80/2 PD gains, 23.5 Nm torque limits, joint
target limits, hard 15000/250 contact, and SynReal nonlinear/linear solver
iterations of 30/10.

Validate configuration without starting PPO:

```bash
.venv/bin/python -m examples.locomotion.train --robot go2
```

On this machine, use `synreal-run` for authenticated GPU execution:

```bash
synreal-run examples/locomotion/train.py \
  --robot go2 --num-envs 512 --command-vx 0.5 --run
```

The code does not depend on `synreal-run`; other machines with a compatible
CUDA environment and a normal SynReal login can run the Python module directly.

Checkpoints from `locomotion3` are not compatible: those actors use a 47D input
and contain an embedded Fourier reference.
