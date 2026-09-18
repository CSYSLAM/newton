# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in host checkpoints for diagnosing W1 startup and GPU wait hangs."""

import faulthandler
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_stream = None


def checkpoint(phase: str) -> None:
    """Persist a host checkpoint without synchronizing or querying the GPU."""
    if _stream is None:
        return
    _stream.write(f"{datetime.now().astimezone().isoformat()} pid={os.getpid()} {phase}\n")
    _stream.flush()
    os.fsync(_stream.fileno())
    # A slow compilation can also trigger this timer; a stack is evidence of
    # the current call, not proof that the process or driver has deadlocked.
    faulthandler.dump_traceback_later(15, repeat=True, file=_stream)


@contextmanager
def diagnostics(path: Path | None):
    """Write checkpoints and timed stacks to a new file for this process."""
    global _stream  # noqa: PLW0603
    if path is None:
        yield
        return
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", buffering=1) as stream:
        _stream = stream
        try:
            checkpoint("diagnostics.begin; host checkpoints only; no added CUDA synchronization")
            yield
        finally:
            try:
                checkpoint("diagnostics.exit")
            finally:
                faulthandler.cancel_dump_traceback_later()
                _stream = None
