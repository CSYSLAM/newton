#!/usr/bin/env python3
"""Low-frequency controller for the staged G1 enhancement training sequence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ASSET = Path("/home/ubuntu-jushen/newton_style3d/newton/examples/assets/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd")
BOOTSTRAP = Path("runs/synreal_g1_velocity_bootstrap/2026-09-10_09-26-05/model_3999.pt")
STAGES = (
    ("straight_symmetry_s1", "--straight-symmetry-stage 1", 1200, 0.0),
    ("straight_symmetry_s2", "--straight-symmetry-stage 2", 1500, 0.0),
    ("robustness_s1", "--robustness-stage 1", 1000, 0.0),
    ("robustness_s2", "--robustness-stage 2", 1200, 0.0),
    ("robustness_s3", "--robustness-stage 3", 1500, 0.0),
)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def latest_run(root: Path, experiment: str) -> Path:
    candidates = sorted(
        (path for path in (root / f"synreal_g1_{experiment}").glob("*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no run directory for {experiment}")
    return candidates[-1]


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    return max(
        checkpoints,
        key=lambda path: int(re.search(r"model_(\d+)", path.name).group(1)),
    )


def scalar_tail(run_dir: Path, window: int = 100) -> dict[str, float]:
    script = """
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json, sys
events = EventAccumulator(sys.argv[1]).Reload()
result = {}
for tag in events.Tags().get("scalars", []):
    values = events.Scalars(tag)[-int(sys.argv[2]):]
    if values:
        result[tag] = sum(item.value for item in values) / len(values)
print(json.dumps(result))
"""
    result = run([".venv/bin/python", "-c", script, str(run_dir), str(window)])
    return json.loads(result.stdout)


def evaluate(checkpoint: Path, output: Path) -> dict[str, object]:
    command = [
        ".venv/bin/python",
        "examples/locomotion_g1/mujoco_eval.py",
        "--checkpoint",
        str(checkpoint),
        "--asset",
        str(ASSET),
        "--duration",
        "30",
        "--headless",
        "--summary",
        str(output),
    ]
    result = run(command, check=False)
    if result.returncode != 0:
        return {"returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}
    return json.loads(output.read_text(encoding="utf-8"))


def gate(
    stage: str,
    scalars: dict[str, float],
    mujoco: dict[str, object],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    episode_length = scalars.get("Episode/length", 0.0)
    vx_error = scalars.get("Episode/mean_vx_abs_error", float("inf"))
    air_imbalance = scalars.get("Episode/foot_air_fraction_imbalance", float("inf"))
    torque = scalars.get("Episode/torque_saturation_fraction", float("inf"))
    if episode_length < 900.0:
        reasons.append(f"episode_length={episode_length:.3f} < 900")
    if vx_error > 0.15:
        reasons.append(f"vx_error={vx_error:.3f} > 0.15")
    if air_imbalance > 0.10:
        reasons.append(f"air_imbalance={air_imbalance:.3f} > 0.10")
    if torque > 0.02:
        reasons.append(f"torque_saturation={torque:.3f} > 0.02")
    if not mujoco.get("finite", False) or mujoco.get("first_failure_reason") is not None:
        reasons.append(f"mujoco_failure={mujoco.get('first_failure_reason')}")
    return not reasons, reasons


def append_report(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n## " + str(record["stage"]) + "\n\n")
        stream.write("```json\n" + json.dumps(record, indent=2) + "\n```\n")


def wait_for_stage(
    log_path: Path,
    expected_seconds: float,
    final_iteration: int,
) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if not re.search(rf"Learning iteration {final_iteration - 1}/{final_iteration}", text):
        time.sleep(max(expected_seconds, 60.0))
    while True:
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if re.search(rf"Learning iteration {final_iteration - 1}/{final_iteration}", text):
            return
        process = subprocess.run(
            [
                "pgrep",
                "-f",
                "/home/ubuntu-jushen/.local/share/synreal/run_with_login.py examples/locomotion_g1/train.py",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
        )
        if process.returncode != 0:
            return
        completed = re.findall(r"Learning iteration (\d+)/\d+", text)
        iteration_times = re.findall(r"Iteration time: ([0-9.]+)s", text)
        if completed and iteration_times:
            remaining = max(final_iteration - int(completed[-1]) - 1, 1)
            next_wait = remaining * float(iteration_times[-1]) + 60.0
        else:
            next_wait = 900.0
        time.sleep(max(next_wait, 60.0))


def evaluate_stage(
    repo: Path,
    report: Path,
    stage: str,
    experiment: str,
) -> tuple[bool, Path, dict[str, object]]:
    run_dir = latest_run(repo / "runs", experiment)
    checkpoint = latest_checkpoint(run_dir)
    scalars = scalar_tail(run_dir)
    summary_path = run_dir / "mujoco_eval_30s.json"
    mujoco = evaluate(checkpoint, summary_path)
    passed, reasons = gate(stage, scalars, mujoco)
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": run(["sha256sum", str(checkpoint)]).stdout.split()[0],
        "scalars_last_100_mean": scalars,
        "mujoco_30s": mujoco,
        "gate_passed": passed,
        "gate_reasons": reasons,
        "asset": str(ASSET),
        "self_collisions": True,
        "mass_randomization": False,
        "com_randomization": False,
        "raw_action_clipping": False,
        "target_slew": False,
    }
    append_report(report, record)
    return passed, checkpoint, record


def start_stage(
    repo: Path,
    stage: str,
    options: str,
    iterations: int,
    checkpoint: Path,
) -> Path:
    log = repo / "runs" / f"g1_{stage}_auto.log"
    session = f"g1_{stage}_auto"
    command = (
        f"cd {repo} && exec /home/ubuntu-jushen/.local/bin/synreal-run "
        f"examples/locomotion_g1/train.py --run {options} --num-envs 512 "
        f"--iterations {iterations} --actor-init {checkpoint} --asset {ASSET} "
        f"> {log} 2>&1"
    )
    run(["tmux", "new-session", "-d", "-s", session, command])
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("runs/g1_enhancement_report.md"))
    parser.add_argument("--initial-wait", type=float, default=0.0)
    parser.add_argument("--wait-log", type=Path)
    parser.add_argument("--wait-experiment", type=str)
    parser.add_argument("--wait-iterations", type=int, default=0)
    parser.add_argument("--actor-init", type=Path)
    args = parser.parse_args()
    if not 0 <= args.start_index < len(STAGES):
        raise SystemExit("start index is outside the staged sequence")

    report = args.repo / args.report
    previous = args.actor_init or BOOTSTRAP
    if args.wait_log is not None:
        if not args.wait_experiment or args.wait_iterations <= 0:
            raise SystemExit("wait-log requires wait-experiment and wait-iterations")
        wait_for_stage(
            args.repo / args.wait_log,
            args.initial_wait or args.wait_iterations * 6.7,
            args.wait_iterations,
        )
        passed, previous, record = evaluate_stage(
            args.repo,
            report,
            args.wait_experiment,
            args.wait_experiment,
        )
        if not passed:
            print(json.dumps(record, indent=2), flush=True)
            return
        args.actor_init = previous
        args.initial_wait = 0.0
    for index, (stage, options, iterations, _) in enumerate(STAGES[args.start_index :], args.start_index):
        if index == args.start_index and args.actor_init is not None:
            log = start_stage(args.repo, stage, options, iterations, previous)
            args.actor_init = None
        if index == 0 and args.start_index == 0:
            log = args.repo / "runs/g1_straight_s1_20260911.log"
            expected = args.initial_wait if args.initial_wait > 0.0 else 1200 * 6.7
            args.initial_wait = 0.0
        else:
            log = args.repo / "runs" / f"g1_{stage}_auto.log"
            expected = iterations * 6.7
        wait_for_stage(log, expected, iterations)
        passed, checkpoint, record = evaluate_stage(args.repo, report, stage, stage)
        if not passed:
            print(json.dumps(record, indent=2), flush=True)
            return
        if index + 1 >= len(STAGES):
            print(json.dumps(record, indent=2), flush=True)
            return
        previous = checkpoint
        next_stage, next_options, next_iterations, _ = STAGES[index + 1]
        start_stage(args.repo, next_stage, next_options, next_iterations, previous)


if __name__ == "__main__":
    main()
