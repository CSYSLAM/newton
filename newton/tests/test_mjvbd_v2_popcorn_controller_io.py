# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify controller batching preserves exact state and command values."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import Example
from newton.examples.mjvbdv2.support.popcorn_controller_io import ControllerSnapshot, write_wrist_targets


class TestPopcornControllerIO(unittest.TestCase):
    def test_ik_feedback_backup(self):
        """Restore the original GPU joint command before retries and on failure."""
        original = np.arange(12, dtype=np.float32).reshape(1, -1) / 17
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for succeed in (True, False):
                example = Example.__new__(Example)
                example.ik_q = wp.array(original, device=device)
                example.tool_orientation_correction = wp.quat_identity()
                example._set_wrist_targets = lambda **kwargs: ([], 1.0)
                calls = []

                def solve(_targets, *, calls=calls, example=example, succeed=succeed):
                    calls.append(example.ik_q.numpy().copy())
                    example.ik_q.fill_(7.0)
                    if not succeed or len(calls) < 3:
                        raise RuntimeError("Rejected test iterate")

                example._solve_wrist_ik = solve
                if succeed:
                    example._solve_wrist_ik_with_feedback([])
                    np.testing.assert_array_equal(example.ik_q.numpy(), 7.0)
                    self.assertEqual(len(calls), 3)
                else:
                    with self.assertRaisesRegex(RuntimeError, "Rejected test iterate"):
                        example._solve_wrist_ik_with_feedback([])
                    np.testing.assert_array_equal(example.ik_q.numpy(), original)
                    self.assertEqual(len(calls), 4)
                for values in calls:
                    np.testing.assert_array_equal(values, original)

    def test_snapshot_values_and_lifetime(self):
        """Copy exact state values and keep earlier snapshots immutable."""
        rng = np.random.default_rng(793)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            state = SimpleNamespace(
                particle_q=wp.array(rng.normal(size=(97, 3)), dtype=wp.vec3, device=device),
                body_q=wp.array(rng.normal(size=(121, 7)), dtype=wp.transform, device=device),
                joint_q=wp.array(rng.normal(size=105), dtype=float, device=device),
            )
            snapshot = ControllerSnapshot(state)
            first = snapshot.read(state)
            saved = {name: value.copy() for name, value in first.items()}
            for name, value in first.items():
                np.testing.assert_array_equal(value.view(np.uint32), getattr(state, name).numpy().view(np.uint32))
                self.assertFalse(value.flags["W"])
            state.particle_q.zero_()
            second = snapshot.read(state)
            np.testing.assert_array_equal(second["particle_q"], 0)
            for name in first:
                np.testing.assert_array_equal(first[name], saved[name])

    def test_target_bits(self):
        """Publish all four targets unchanged and preserve unrelated entries."""
        values = (wp.vec3(-0.0, 0.3, -0.7), wp.vec3(0.1, -0.2, 0.9), wp.vec4(0, 0, 0, 1), wp.vec4(0.1, 0.2, 0.3, 0.4))
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            outputs = [wp.zeros(2, dtype=dtype, device=device) for dtype in (wp.vec3, wp.vec3, wp.vec4, wp.vec4)]
            wp.launch(write_wrist_targets, 1, [*values, *outputs], device=device)
            for expected, output in zip(values, outputs, strict=True):
                result = output.numpy()
                np.testing.assert_array_equal(
                    result[0].view(np.uint32), np.asarray(expected, dtype=np.float32).view(np.uint32)
                )
                np.testing.assert_array_equal(result[1], 0)


if __name__ == "__main__":
    unittest.main()
