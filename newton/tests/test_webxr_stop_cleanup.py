# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise process cleanup with disposable workers and a fake GPU inventory."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestWebXRStopCleanup(unittest.TestCase):
    def test_cleanup_terminates_orphans_and_preserves_unrelated_gpu_jobs(self):
        """Kill matching workers, escalate ignored TERM, and tolerate unavailable NVML."""
        root = Path(__file__).resolve().parents[2]
        for gpu_available in (True, False):
            with self.subTest(gpu_available=gpu_available), tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                example = f"mjvbd_v2_webxr_cleanup_{temp.name}"
                workers = []
                for name, ignore_term in (
                    (f"example_{example}.py", False),
                    (f"example_{example}.py", True),
                    ("other.py", False),
                ):
                    folder = temp / str(len(workers))
                    folder.mkdir()
                    script = folder / name
                    script.write_text(
                        "import signal, time\n"
                        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
                        + "print('ready', flush=True)\ntime.sleep(60)\n"
                    )
                    worker = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
                    workers.append(worker)
                    self.assertEqual(worker.stdout.readline().strip(), "ready")
                # Merely mentioning the example in another program's arguments must not match.
                workers.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", example]))
                try:
                    binaries = temp / "bin"
                    binaries.mkdir()
                    actions = temp / "actions"
                    for name, body in {
                        "nvidia-smi": (
                            "printf 'gpu-query\\n' >> \"$TEST_ACTIONS\"\n"
                            + (
                                f"printf '%s\\n' {' '.join(str(p.pid) for p in workers)} {workers[0].pid}\n"
                                if gpu_available
                                else "exit 1\n"
                            )
                        ),
                        "systemctl": (
                            'printf \'systemctl %s\\n\' "$*" >> "$TEST_ACTIONS"\n'
                            f'if [[ "$*" == *MainPID* && -d /proc/{workers[0].pid} ]]; then echo {workers[0].pid}; fi\n'
                        ),
                        "curl": "exit 0\n",
                        "adb": "exit 0\n",
                    }.items():
                        executable = binaries / name
                        executable.write_text("#!/usr/bin/env bash\n" + body)
                        executable.chmod(0o755)
                    environment = dict(
                        os.environ,
                        PATH=f"{binaries}{os.pathsep}{os.environ['PATH']}",
                        TEST_ACTIONS=str(actions),
                        XDG_RUNTIME_DIR=str(temp / "run"),
                        XDG_STATE_HOME=str(temp / "state"),
                        NEWTON_WEBXR_EXAMPLE=example,
                        NEWTON_WEBXR_UNIT="newton-webxr-cleanup-test.service",
                        NEWTON_WEBXR_CLEANUP_PROCESSES="1",
                        NEWTON_WEBXR_PORT="18765",
                    )
                    result = subprocess.run(
                        ["bash", str(root / "scripts/stop_quest_webxr_teleop.sh")],
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIsNotNone(workers[0].poll(), result.stdout)
                    self.assertEqual(workers[1].poll(), -signal.SIGKILL, result.stdout)
                    self.assertIsNone(workers[2].poll())
                    self.assertIsNone(workers[3].poll())
                    self.assertIn("gpu-query", actions.read_text())
                    self.assertIn("--no-block stop newton-webxr-cleanup-test.service", actions.read_text())
                    self.assertIn("SIGKILL", result.stdout)
                    repeated = subprocess.run(
                        ["bash", str(root / "scripts/stop_quest_webxr_teleop.sh")],
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(repeated.returncode, 0, repeated.stderr)
                finally:
                    for worker in workers:
                        if worker.poll() is None:
                            worker.kill()
                        worker.wait(timeout=5)
                        if worker.stdout:
                            worker.stdout.close()

    def test_w1_stop_enables_cleanup_by_default(self):
        """Make the W1 stop command release processes without an extra environment flag."""
        root = Path(__file__).resolve().parents[2]
        self.assertIn(
            'export NEWTON_WEBXR_CLEANUP_PROCESSES="1"',
            (root / "scripts/stop_quest_webxr_w1_bag_packing_teleop.sh").read_text(),
        )
