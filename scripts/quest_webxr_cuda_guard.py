#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep a small CUDA workload alive while a WebXR scene process is replaced."""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import warp as wp


@wp.kernel
def _heartbeat(counter: wp.array[wp.float32]):
    index = wp.tid()
    counter[index] = counter[index] + 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.01 <= args.interval_seconds <= 1.0:
        raise ValueError("--interval-seconds must be between 0.01 and 1.0")

    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError(f"CUDA guard requires a CUDA device, got {device.alias!r}")

    running = True

    def request_exit(_signal_number, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_exit)
    signal.signal(signal.SIGTERM, request_exit)

    counter = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(_heartbeat, dim=1, inputs=[counter], device=device)
    wp.synchronize_device(device)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(
        f"pid={os.getpid()}\ndevice={device.alias}\n",
        encoding="utf-8",
    )
    print(f"Quest WebXR CUDA reload guard ready on {device.alias} (PID {os.getpid()})", flush=True)
    try:
        while running:
            wp.launch(_heartbeat, dim=1, inputs=[counter], device=device)
            wp.synchronize_device(device)
            time.sleep(args.interval_seconds)
    finally:
        wp.synchronize_device(device)
        args.ready_file.unlink(missing_ok=True)
        print("Quest WebXR CUDA reload guard stopped", flush=True)


if __name__ == "__main__":
    main()
