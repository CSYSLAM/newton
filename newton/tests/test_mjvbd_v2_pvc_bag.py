# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the standalone PVC inspection scene on CPU without WebXR or W1 assets."""

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2.example_mjvbd_v2_pvc_bag import Example
from newton.viewer import ViewerNull


@unittest.skipUnless(
    (Path.home() / "下载/clear_plastic_bag/clear_plastic_bag.glb").is_file(), "Requires the PVC bag asset"
)
class TestPVCBagDemo(unittest.TestCase):
    def test_film_support_stays_local_and_handles_retain_curve(self):
        """Allow the sheet to fall under gravity without turning its reinforcement into a cage."""
        with wp.ScopedDevice("cpu"):
            scene = Example(ViewerNull(), Example.create_parser().parse_args(["--no-task", "--pose", "lying"]))
            # CPU .numpy() can alias the simulation state; keep a true rest copy.
            rest = scene.state_0.particle_q.numpy().copy()
            springs = scene.model.spring_indices.numpy().reshape(-1, 2)
            shell = springs < scene.bag["shell_count"]
            self.assertFalse(np.any(np.all(shell, axis=1)), "Film must not have interior spanning braces")
            roots = springs[np.any(shell, axis=1)]
            lengths = np.linalg.norm(rest[roots[:, 0]] - rest[roots[:, 1]], axis=1)
            self.assertLess(lengths.max(), 0.04, "Handle reinforcement must stay local to each root")
            # Free fall checks actual gravity response, rather than a material label.
            elevated = rest + np.array((0, 0, 0.20), dtype=np.float32)
            for state in (scene.state_0, scene.state_1):
                state.particle_q.assign(elevated)
                scene.solver.reset(state, flags=0)
            for _ in range(20):
                scene.step()
            q = scene.state_0.particle_q.numpy()
            scene.test_final()
            masses = scene.model.particle_mass.numpy()
            fall = np.average(elevated[:, 2] - q[:, 2], weights=masses)
            self.assertGreater(fall, 0.04, "An unsupported PVC bag must fall under gravity")
            for handle in scene.bag["handles"]:
                initial = rest[handle.ravel()]
                current = q[handle.ravel()]
                initial -= initial.mean(0)
                current -= current.mean(0)
                u, _, vt = np.linalg.svd(initial.T @ current)
                correction = np.diag((1.0, 1.0, np.linalg.det(u @ vt)))
                error = initial @ u @ correction @ vt - current
                self.assertLess(np.sqrt(np.mean(np.sum(error**2, axis=1))), 0.01)
            self.assertTrue(np.all(scene.model.particle_inv_mass.numpy() > 0))

    def test_lying_task_grips_before_loading_and_resets(self):
        """Stage a dynamic load outside a lying bag and lift before withdrawing its support."""
        with wp.ScopedDevice("cpu"):
            scene = Example(ViewerNull(), Example.create_parser().parse_args([]))
            flags = scene.model.body_flags.numpy()
            self.assertEqual(scene.args.pose, "lying")
            self.assertEqual(len(scene.payloads), 2)
            self.assertEqual(len(scene.jaws), 2)
            self.assertTrue(np.all(flags[scene.driven_bodies] & int(newton.BodyFlags.KINEMATIC)))
            self.assertFalse(np.any(flags[scene.payloads] & int(newton.BodyFlags.KINEMATIC)))
            self.assertTrue(np.all(scene.model.particle_inv_mass.numpy() > 0))
            np.testing.assert_allclose(scene.model.body_mass.numpy()[scene.payloads], (0.020, 0.025), rtol=1e-5)
            initial = scene.state_0.body_q.numpy().copy()
            rest = scene.state_0.particle_q.numpy().copy()
            direction = rest[scene.bag["rim"]].mean(0) - rest[scene.bag["bottom"]].mean(0)
            direction /= np.linalg.norm(direction)
            self.assertLess(direction[1], -0.99)
            self.assertLess(abs(direction[2]), 0.05)
            self.assertGreater(initial[scene.payloads, 0].min(), rest[:, 0].max())
            self.assertGreater(rest[scene.bag["handles"][scene.handle_index], 2].mean(), rest[:, 2].mean())
            for _ in range(2):
                scene.step()
            scene.sim_time = 0.6
            scene._prepare_task()
            closed = scene.driven_end.numpy().copy()
            np.testing.assert_allclose(np.linalg.norm(closed[1, :3] - closed[0, :3]), 0.0415, atol=1e-5)
            np.testing.assert_allclose(closed[2, :3], scene.tray_home, atol=1e-6)
            scene.sim_time = 2.3 - scene.frame_dt
            scene._prepare_task()
            raised = scene.driven_end.numpy().copy()
            np.testing.assert_allclose(raised[:2, :3] - closed[:2, :3], ((0, 0, 0.25),) * 2, atol=2e-5)
            scene.sim_time = 3.8
            scene._prepare_task()
            lifted = scene.driven_end.numpy().copy()
            np.testing.assert_allclose(lifted[:2, :3] - closed[:2, :3], ((0, 0, 0.5),) * 2, atol=2e-5)
            # Gravity must orient the bag; no scripted wrist turn or horizontal sweep.
            np.testing.assert_array_equal(lifted[:2, 3:], closed[:2, 3:])
            scene.sim_time = 5.4 - scene.frame_dt
            scene._prepare_task()
            over_mouth = scene.driven_end.numpy().copy()
            scene.sim_time = 5.65
            scene._prepare_task()
            released = scene.driven_end.numpy()
            np.testing.assert_allclose(released[2, :3] - over_mouth[2, :3], (0, 0.22, -0.20), atol=1e-6)
            np.testing.assert_array_equal(released[:2], over_mouth[:2])
            # Planning never writes particle/payload positions or changes ownership.
            self.assertFalse(np.any(scene.model.body_flags.numpy()[scene.payloads] & int(newton.BodyFlags.KINEMATIC)))
            scene.reset_physics()
            np.testing.assert_array_equal(scene.state_0.body_q.numpy(), initial)
            np.testing.assert_array_equal(scene.state_0.particle_q.numpy(), rest)
            np.testing.assert_array_equal(scene.driven_end.numpy(), scene.initial_driven_poses)
            self.assertIsNone(scene.grip_center)
            self.assertIsNone(scene.lift_baseline)
            self.assertIsNone(scene.delivery_target)

    def test_both_poses_rest_above_table_and_reset_exactly(self):
        """Run both ordinary-window poses and restore positions and velocities after settling."""
        with wp.ScopedDevice("cpu"):
            for pose in ("upright", "lying"):
                with self.subTest(pose=pose):
                    args = Example.create_parser().parse_args(["--no-task", "--pose", pose])
                    self.assertFalse(args.no_cuda_graph)
                    example = Example(ViewerNull(), args)
                    initial = example.state_0.particle_q.numpy().copy()
                    self.assertAlmostEqual(float(initial[:, 2].min()), example.table_height + 0.002, places=6)
                    self.assertEqual(example.model.body_count, 1)
                    self.assertFalse(hasattr(example, "webxr_server"))
                    self.assertGreater(np.ptp(initial[:, 2]) if pose == "upright" else np.ptp(initial[:, 1]), 0.49)
                    for _ in range(3):
                        example.step()
                    example.test_final()
                    example.render()
                    self.assertFalse(np.array_equal(initial, example.state_0.particle_q.numpy()))
                    example.reset_physics()
                    np.testing.assert_array_equal(example.state_0.particle_q.numpy(), initial)
                    np.testing.assert_array_equal(example.state_1.particle_q.numpy(), initial)
                    np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), 0)
                    self.assertEqual(example.sim_time, 0)
                    example.step()
                    example.test_final()


if __name__ == "__main__":
    unittest.main()
