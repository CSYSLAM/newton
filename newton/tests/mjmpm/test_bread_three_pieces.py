# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sequential blade strokes and safe clearance between opposite-side cuts."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import BOARD_TOP, CUT_Y, LOAF_CENTER
from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import Example as SingleCutExample
from newton.examples.mjmpm.example_mjmpm_w1_bread_three_pieces import SECOND_CUT_Y, TOTAL_FRAMES, Example
from newton.examples.mjmpm.support.bread_cut_cohesion import CuttableCohesion, count_detached_samples


class TestBreadThreePieces(unittest.TestCase):
    def test_first_stroke_preserved_and_traverse_clears_the_loaf(self):
        example = object.__new__(Example)
        example.home = example.initial_targets()
        frames = example._keyframes()
        first = SingleCutExample._keyframes(example)[:-1]
        for actual, expected in zip(frames, first, strict=False):
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertEqual(actual[2], expected[2])
        self.assertLess(CUT_Y, LOAF_CENTER[1])
        self.assertGreater(SECOND_CUT_Y, LOAF_CENTER[1])
        # The straight edge is 55 mm below the tool; horizontal transfer
        # must clear the whole 65 mm loaf rather than sweep through it.
        for time in np.linspace(9.6, 11.6, 31):
            targets, _ = example.plan(float(time))
            self.assertGreater(targets[1, 2] - 0.055, BOARD_TOP + 0.065)
        self.assertEqual(frames[-1][0], TOTAL_FRAMES / 60)
        args = Example.create_parser().parse_args([])
        self.assertEqual(args.num_frames, TOTAL_FRAMES)
        self.assertEqual(args.bread_render, "particles")

    def test_actual_blade_sweeps_split_one_then_two_then_three_components(self):
        spacing = 0.005
        rest = np.stack(
            np.meshgrid(
                np.arange(4) * spacing + 0.1,
                np.arange(8) * spacing - 0.0175,
                np.arange(4) * spacing + 0.001,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                pose = wp.transform(wp.vec3(0, -0.010, 0.20), wp.quat_identity())
                cohesion = CuttableCohesion(rest, mass=3.5e-5, spacing=spacing, device=device, tool_pose=pose)
                state = SimpleNamespace(
                    particle_q=wp.array(rest, dtype=wp.vec3), body_q=wp.array([pose], dtype=wp.transform)
                )
                self.assertEqual(len(cohesion.components()[1]), 1)

                def move(y, z, state=state, cohesion=cohesion):
                    state.body_q.assign(
                        wp.array([wp.transform(wp.vec3(0, y, z), wp.quat_identity())], dtype=wp.transform)
                    )
                    cohesion.cut(state, 0, wp.transform_identity())

                move(-0.010, 0.055)
                self.assertEqual(len(cohesion.components()[1]), 2)
                severed = cohesion.active.numpy() == 0
                move(-0.010, 0.20)
                move(0.010, 0.20)
                self.assertEqual(len(cohesion.components()[1]), 2)
                move(0.010, 0.055)
                labels, sizes = cohesion.components()
                np.testing.assert_array_equal(np.sort(sizes), [32, 32, 64])
                np.testing.assert_array_equal(cohesion.active.numpy()[severed], 0)
                self.assertEqual(count_detached_samples(rest, labels, spacing), 0)
                np.testing.assert_array_equal(state.particle_q.numpy(), rest.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
