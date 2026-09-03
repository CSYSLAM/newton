# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``--render-fps`` example CLI option."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import newton.examples as examples_module
from newton.examples import _throttle_render_fps, create_parser


class TestRenderFPSCLI(unittest.TestCase):
    """Tests for render FPS parsing and throttling."""

    def test_parser_has_render_fps_arg(self):
        """The base parser should include --render-fps."""
        parser = create_parser()
        args = parser.parse_known_args(["--render-fps", "30"])[0]
        self.assertEqual(args.render_fps, 30.0)

    def test_default_render_fps_none(self):
        """Render FPS should be uncapped by default."""
        parser = create_parser()
        args = parser.parse_known_args([])[0]
        self.assertIsNone(args.render_fps)

    def test_render_fps_rejects_non_positive_values(self):
        """Non-positive render FPS limits should be rejected."""
        parser = create_parser()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "-10"])
            with self.assertRaises(SystemExit):
                parser.parse_known_args(["--render-fps", "nan"])
        self.assertIn("must be a finite value greater than 0", stderr.getvalue())

    def test_throttle_sleeps_for_remaining_frame_time(self):
        """Throttle should sleep for the remaining frame period."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=20.0,
            time_fn=lambda: 10.02,
            sleep_fn=sleeps.append,
        )

        self.assertAlmostEqual(slept, 0.03)
        self.assertEqual(sleeps, [slept])

    def test_throttle_skips_sleep_when_frame_is_slow(self):
        """Throttle should not sleep once a frame exceeds the target period."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=20.0,
            time_fn=lambda: 10.07,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(slept, 0.0)
        self.assertEqual(sleeps, [])

    def test_throttle_skips_sleep_without_render_fps(self):
        """No cap should be applied when render FPS is None."""
        sleeps = []

        slept = _throttle_render_fps(
            frame_start_time=10.0,
            render_fps=None,
            time_fn=lambda: 10.0,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(slept, 0.0)
        self.assertEqual(sleeps, [])

    def test_run_throttles_idle_frames(self):
        """The main run loop should throttle empty idle frames."""

        class DummyViewer:
            def __init__(self):
                self._running = [True, True, False]
                self.frames = []
                self.closed = False

            def is_running(self):
                return self._running.pop(0)

            def begin_frame(self, dt):
                self.frames.append(("begin", dt))

            def end_frame(self):
                self.frames.append(("end",))

            def should_step(self):
                raise AssertionError("idle branch should skip stepping")

            def close(self):
                self.closed = True

        class DummyBrowser:
            def __init__(self):
                self.switch_target = object()
                self._reset_requested = False

            def switch(self, example_class):
                self.switch_target = None
                return None, example_class

        viewer = DummyViewer()
        example = SimpleNamespace(viewer=viewer)
        args = SimpleNamespace(render_fps=30.0, test=False)
        browser = DummyBrowser()
        throttle_calls = []

        def record_throttle(frame_start_time, render_fps):
            throttle_calls.append((frame_start_time, render_fps))
            return 0.0

        with (
            patch.object(examples_module, "_ExampleBrowser", return_value=browser),
            patch.object(examples_module, "_throttle_render_fps", side_effect=record_throttle),
            patch.object(examples_module.time, "perf_counter", side_effect=[10.0, 11.0]),
        ):
            examples_module.run(example, args)

        self.assertEqual(viewer.frames, [("begin", 0.0), ("end",)])
        self.assertEqual(throttle_calls, [(11.0, 30.0)])
        self.assertTrue(viewer.closed)

    def test_run_uses_opt_in_in_place_reset(self):
        """A scene can preserve external sessions when the toolbar requests Reset."""

        class DummyViewer:
            def __init__(self):
                self._running = [True, True, False]
                self.closed = False

            def is_running(self):
                return self._running.pop(0)

            def should_step(self):
                return False

            def close(self):
                self.closed = True

        class DummyExample:
            reset_in_place = True

            def __init__(self, viewer):
                self.viewer = viewer
                self.reset_count = 0
                self.render_count = 0

            def reset_physics(self, *, source):
                self.reset_count += 1
                self.reset_source = source

            def render(self):
                self.render_count += 1

        class DummyBrowser:
            switch_target = None

            def __init__(self):
                self._reset_requested = True

            def reset(self, _example_class):
                raise AssertionError("opt-in reset must not reconstruct the example")

        viewer = DummyViewer()
        example = DummyExample(viewer)
        browser = DummyBrowser()
        args = SimpleNamespace(render_fps=None, test=False)

        with (
            patch.object(examples_module, "_ExampleBrowser", return_value=browser),
            patch.object(examples_module, "_throttle_render_fps", return_value=0.0),
        ):
            examples_module.run(example, args)

        self.assertEqual(example.reset_count, 1)
        self.assertEqual(example.reset_source, "viewer")
        self.assertEqual(example.render_count, 1)
        self.assertTrue(viewer.closed)

    def test_run_honors_example_exit_request(self):
        """An example can leave the loop so the viewer closes synchronously."""

        class DummyViewer:
            def __init__(self):
                self.closed = False

            def is_running(self):
                return True

            def should_step(self):
                raise AssertionError("exit request must stop before stepping")

            def close(self):
                self.closed = True

        viewer = DummyViewer()
        example = SimpleNamespace(viewer=viewer, exit_requested=True)
        args = SimpleNamespace(render_fps=None, test=False)
        browser = SimpleNamespace(switch_target=None, _reset_requested=False)

        with patch.object(examples_module, "_ExampleBrowser", return_value=browser):
            examples_module.run(example, args)

        self.assertTrue(viewer.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
