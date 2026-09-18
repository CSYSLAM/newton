#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Collect W1 crash evidence without starting CUDA or changing the machine.

uv run python scripts/collect_quest_webxr_w1_diagnostics.py

Run after reboot, before clearing the teleoperation active-run marker. Output
defaults to recordings/diagnostics/<timestamp>-<pid>. No GPU queries, process
termination, driver changes, or uploads are performed.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_command(destination: Path, command: list[str]) -> None:
    """Save bounded command output, including failures and permission errors."""
    with destination.open("w", encoding="utf-8") as stream:
        stream.write(f"Command: {command!r}\n\n")
        stream.flush()
        try:
            result = subprocess.run(
                command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=20, check=False
            )
            stream.write(f"\nExit status: {result.returncode}\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            stream.write(f"\nCollection error: {error}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or (
        REPO_ROOT / "recordings/diagnostics" / f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    )
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    commands = {
        "boots.txt": ["journalctl", "--list-boots", "--no-pager"],
        "previous-kernel.txt": ["journalctl", "-b", "-1", "-k", "--no-pager"],
        "previous-boot-tail.txt": ["journalctl", "-b", "-1", "--no-pager", "-n", "400"],
        "current-kernel.txt": ["journalctl", "-b", "0", "-k", "--no-pager"],
        "historical-errors.txt": [
            "journalctl",
            "--no-pager",
            "_TRANSPORT=kernel",
            "-g",
            "NVRM: Xid|Completion Timeout|fallen off|hard LOCKUP|soft lockup|Out of memory|oom-kill|Hardware Error|AER:.*error",
        ],
        "pci-devices.txt": ["lspci", "-nn"],
        "git-head.txt": ["git", "log", "-1", "--format=%H %s"],
        "git-status.txt": ["git", "status", "--short"],
        "w1-working-changes.patch": [
            "git",
            "diff",
            "--",
            "newton/examples/mjvbdv2/example_mjvbd_v2_w1_bag_packing.py",
            "newton/examples/mjvbdv2/example_mjvbd_v2_webxr_w1_bag_packing.py",
            "newton/examples/mjvbdv2/_webxr_w1_head.py",
        ],
    }
    for name, command in commands.items():
        collect_command(destination / name, command)
    versions = {"python": platform.python_version(), "kernel": platform.release()}
    for package in ("warp-lang", "numpy", "mujoco-warp"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed in collector environment"
    (destination / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    with (destination / "system-state.txt").open("w", encoding="utf-8") as stream:
        for path in (
            "/proc/driver/nvidia/version",
            "/proc/cmdline",
            "/proc/meminfo",
            "/proc/sys/kernel/nmi_watchdog",
            "/proc/sys/kernel/sysrq",
            "/sys/bus/pci/devices/0000:01:00.0/power/control",
            "/sys/bus/pci/devices/0000:01:00.0/power/runtime_status",
            "/sys/bus/pci/devices/0000:01:00.0/current_link_speed",
        ):
            stream.write(f"\n{path}:\n")
            try:
                stream.write(Path(path).read_text())
            except OSError as error:
                stream.write(f"Unavailable: {error}\n")
        try:
            entries = list(Path("/sys/fs/pstore").iterdir())
            stream.write(f"\npstore entries: {[entry.name for entry in entries]}\n")
            for entry in entries:
                if entry.is_file():
                    shutil.copyfile(entry, destination / f"pstore-{entry.name}")
        except OSError as error:
            stream.write(f"\npstore unavailable: {error}\n")
    state_root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    state = state_root / "newton-webxr-w1-bag-packing-teleop"
    logs = sorted(state.glob("w1-bag-packing-*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in [state / "active-run", *logs[:3], *(REPO_ROOT / "recordings/diagnostics").glob("*.log")]:
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
    (destination / "README.txt").write_text(
        "Snapshot collected without CUDA or nvidia-smi. previous-* files refer to boot -1.\n"
        "Missing Xid/panic records do not rule out a driver, PCIe, or hardware failure.\n"
        "Host-return checkpoints do not prove GPU completion. Timed Python stacks can\n"
        "also indicate compilation or a paused simulation. A full system lockup may\n"
        "prevent even the watchdog thread from running. No report was uploaded.\n",
        encoding="utf-8",
    )
    print(f"Saved diagnostics: {destination}")


if __name__ == "__main__":
    main()
