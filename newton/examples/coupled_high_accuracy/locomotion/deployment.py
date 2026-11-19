"""Export RSL-RL checkpoints and Unitree C++ deployment configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .policy_spec import PolicySpec, load_policy_spec


def deploy_config(spec: PolicySpec) -> dict[str, Any]:
    history = spec.history_length
    config = {
        "joint_ids_map": list(spec.joint_ids_map),
        "step_dt": spec.policy_dt,
        "stiffness": list(spec.stiffness),
        "damping": list(spec.damping),
        "default_joint_pos": list(spec.default_joint_pos),
        "commands": {"base_velocity": {"ranges": spec.command_ranges}},
        "actions": {
            "JointPositionAction": {
                "clip": None,
                "joint_names": [".*"],
                "scale": list(spec.action_scale),
                "offset": list(spec.default_joint_pos),
                "joint_ids": None,
            }
        },
        "observations": {
            "base_ang_vel": {
                "params": {},
                "clip": None,
                "scale": [spec.ang_vel_scale] * 3,
                "history_length": history,
            },
            "projected_gravity": {
                "params": {},
                "clip": None,
                "scale": [1.0] * 3,
                "history_length": history,
            },
            "velocity_commands": {
                "params": {"command_name": "base_velocity"},
                "clip": None,
                "scale": [1.0] * 3,
                "history_length": history,
            },
            "joint_pos_rel": {
                "params": {},
                "clip": None,
                "scale": [1.0] * spec.action_dim,
                "history_length": history,
            },
            "joint_vel_rel": {
                "params": {},
                "clip": None,
                "scale": [spec.joint_vel_scale] * spec.action_dim,
                "history_length": history,
            },
            "last_action": {
                "params": {},
                "clip": None,
                "scale": [1.0] * spec.action_dim,
                "history_length": history,
            },
        },
    }
    if spec.use_gym_history:
        config["observations"]["use_gym_history"] = True
    return config


def write_deploy_config(spec: PolicySpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(deploy_config(spec), sort_keys=False), encoding="utf-8")


def build_actor(spec: PolicySpec) -> torch.nn.Sequential:
    activation = {"elu": torch.nn.ELU}[str(spec.ppo["activation"]).lower()]
    dims = [spec.observation_dim, *spec.ppo["actor_hidden_dims"], spec.action_dim]
    layers: list[torch.nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(torch.nn.Linear(input_dim, output_dim))
        if index < len(dims) - 2:
            layers.append(activation())
    if spec.ppo.get("bounded_actions", False):
        layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def load_actor(spec: PolicySpec, checkpoint: Path) -> torch.nn.Sequential:
    actor = build_actor(spec)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state.get("model_state_dict", state.get("model", state))
    actor_state = {}
    for name, value in state_dict.items():
        if name.startswith("actor."):
            actor_state[name.removeprefix("actor.")] = value
    if not actor_state:
        raise ValueError(f"checkpoint does not contain actor.* parameters: {checkpoint}")
    actor.load_state_dict(actor_state, strict=True)
    return actor.eval()


def export_actor_onnx(actor: torch.nn.Module, spec: PolicySpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, spec.observation_dim), dtype=torch.float32)
    try:
        torch.onnx.export(
            actor,
            example,
            path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
            external_data=False,
        )
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires onnx and onnxscript; install them with `uv sync --extra locomotion`"
        ) from error


def _require_onnx():
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX deployment requires onnx; install it with `uv sync --extra locomotion`") from error
    return onnx


def copy_onnx_as_single_file(source: Path, destination: Path) -> None:
    """Copy an ONNX model while embedding any external tensor data."""

    onnx = _require_onnx()
    model = onnx.load(source, load_external_data=True)
    onnx.save_model(model, destination, save_as_external_data=False)


def validate_onnx(path: Path, spec: PolicySpec) -> None:
    """Check static deployment names and dimensions."""

    onnx = _require_onnx()
    model = onnx.load(path)
    onnx.checker.check_model(model)
    inputs = {item.name: item for item in model.graph.input}
    outputs = {item.name: item for item in model.graph.output}
    if set(inputs) != {"obs"} or set(outputs) != {"actions"}:
        raise ValueError("ONNX deployment ABI requires input 'obs' and output 'actions'")
    input_shape = [dim.dim_value for dim in inputs["obs"].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in outputs["actions"].type.tensor_type.shape.dim]
    if input_shape != [1, spec.observation_dim]:
        raise ValueError(f"ONNX input shape {input_shape} != [1, {spec.observation_dim}]")
    if output_shape != [1, spec.action_dim]:
        raise ValueError(f"ONNX output shape {output_shape} != [1, {spec.action_dim}]")


def write_manifest(spec: PolicySpec, directory: Path, checkpoint: Path | None) -> None:
    manifest = {
        "format_version": 1,
        "robot": spec.name,
        "abi_sha256": spec.abi_sha256,
        "observation_dim": spec.observation_dim,
        "frame_observation_dim": spec.frame_observation_dim,
        "history_length": spec.history_length,
        "history_layout": ("time-major" if spec.use_gym_history else "term-major, oldest-to-newest within each term"),
        "action_dim": spec.action_dim,
        "onnx_input": "obs",
        "onnx_output": "actions",
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "safety": {
            "required_fsm": ["Passive", "FixStand", "Velocity"],
            "required_external_guards": [
                "physical_estop",
                "state_timeout",
                "command_slew_limit",
                "joint_position_velocity_torque_limits",
                "nan_guard",
                "inference_deadline",
                "temperature_and_battery_guard",
            ],
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go2", "g1"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-onnx", type=Path, help="copy an already exported policy")
    parser.add_argument("--config-only", action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.checkpoint and args.policy_onnx:
        raise SystemExit("choose either --checkpoint or --policy-onnx")
    if not args.config_only and not (args.checkpoint or args.policy_onnx):
        raise SystemExit("provide --checkpoint, --policy-onnx, or --config-only")
    spec = load_policy_spec(args.robot)
    exported = args.output / "exported"
    params = args.output / "params"
    exported.mkdir(parents=True, exist_ok=True)
    params.mkdir(parents=True, exist_ok=True)
    write_deploy_config(spec, params / "deploy.yaml")
    if args.checkpoint:
        policy_path = exported / "policy.onnx"
        export_actor_onnx(load_actor(spec, args.checkpoint), spec, policy_path)
        validate_onnx(policy_path, spec)
    elif args.policy_onnx:
        policy_path = exported / "policy.onnx"
        copy_onnx_as_single_file(args.policy_onnx, policy_path)
        validate_onnx(policy_path, spec)
    write_manifest(spec, args.output, args.checkpoint)
    print(f"Unitree deployment package: {args.output.resolve()}")


if __name__ == "__main__":
    main()
