# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check phase-local download reuse does not hide a newly simulated state."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from state_read_probe import install_state_read_cache


class TestStateReadCache(unittest.TestCase):
    def test_phase_invalidation(self):
        """Refresh after graph-style physics writes, and distinguish array slices."""
        state = SimpleNamespace(body_q=wp.ones(3, device="cpu"), particle_q=wp.ones(4, device="cpu"))
        example = SimpleNamespace(state_0=state)
        collected = []
        example._simulate = lambda: state.body_q.fill_(2.0)
        example._measure = lambda: collected.append(state.body_q.numpy().copy())

        def step():
            first = state.body_q.numpy()
            self.assertIs(first, state.body_q.numpy())
            self.assertFalse(first.flags.writeable)
            self.assertEqual(state.body_q[:1].numpy().shape, (1,))
            state.body_q.fill_(4.0)  # A captured graph does not call Python _simulate.
            example._measure()
            example._simulate()
            np.testing.assert_array_equal(state.body_q.numpy(), 2.0)

        example.step = step
        install_state_read_cache(example)
        example.step()
        np.testing.assert_array_equal(collected[0], 4.0)
        self.assertTrue(state.body_q.numpy().flags.writeable)


if __name__ == "__main__":
    unittest.main()
