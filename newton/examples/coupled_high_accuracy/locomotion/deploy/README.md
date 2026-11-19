# Unitree C++ deployment integration

`deployment.py` produces the directory expected by Unitree `unitree_rl_lab`:

```text
policy/
  exported/policy.onnx
  params/deploy.yaml
  manifest.json
```

`policy.onnx` is always self-contained. Checkpoint export embeds all weights,
and `--policy-onnx` rewrites models with external tensor data into the same
single-file format.

Point `FSM.Velocity.policy_dir` at that directory. The existing Unitree
`Passive -> FixStand -> Velocity` FSM, `mode_machine=5` for G1, state timeout,
and bad-orientation fallback remain the production controller.

Before hardware use, add the guards listed in `manifest.json`: command slew,
q/dq/tau hard limits, NaN/Inf, inference deadline, temperature/battery, and a
physical e-stop. This repository intentionally does not vendor or fork the
Unitree SDK deployment tree.
