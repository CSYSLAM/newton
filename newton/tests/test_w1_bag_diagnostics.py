# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test host-only crash tracing without initializing CUDA."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from newton.examples.mjvbdv2.support import w1_bag_diagnostics as tracing


class TestW1BagDiagnostics(unittest.TestCase):
    def test_trace_is_opt_in_and_cleans_up_after_failure(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(tracing.faulthandler, "dump_traceback_later") as arm,
            patch.object(tracing.faulthandler, "cancel_dump_traceback_later") as cancel,
        ):
            with tracing.diagnostics(None):
                tracing.checkpoint("disabled")
            arm.assert_not_called()
            path = Path(directory) / "trace.log"
            with self.assertRaisesRegex(ValueError, "test failure"), tracing.diagnostics(path):
                tracing.checkpoint("physics.begin")
                raise ValueError("test failure")
            self.assertIn("physics.begin", path.read_text())
            self.assertIn("diagnostics.exit", path.read_text())
            self.assertEqual(arm.call_count, 3)
            cancel.assert_called_once()
            self.assertIsNone(tracing._stream)
            with self.assertRaises(FileExistsError), tracing.diagnostics(path):
                self.fail("Existing crash evidence must not be overwritten")


if __name__ == "__main__":
    unittest.main()
