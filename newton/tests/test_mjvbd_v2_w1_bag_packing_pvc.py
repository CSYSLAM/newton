# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that standalone PVC inspection shares the teleop scene without XR resources."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_bag_packing_pvc import Example
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_pick_place import _prescribe
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_pvc import Example as TeleopExample
from newton.viewer import ViewerNull


class TestPVCPackingDefaults(unittest.TestCase):
    def test_first_physics_targets_and_reset_hold_solved_pose(self):
        """Keep the solved joint pose through the actual prescription kernel and reset."""
        with wp.ScopedDevice("cpu"):
            solved = np.array([0.65, -0.4], dtype=np.float32)

            class State:
                def __init__(self):
                    self.joint_q = wp.zeros(2, dtype=float)

                def assign(self, other):
                    wp.copy(self.joint_q, other.joint_q)

            def build_scene(scene, viewer, args):
                scene.model = SimpleNamespace(state=State)
                scene.state_0, scene.state_1 = State(), State()
                scene.state_0.joint_q.assign(solved)
                scene.state_1.assign(scene.state_0)
                scene.ik_q = wp.array(solved.reshape(1, -1), dtype=float)
                # The shared constructor seeds these from the pre-IK model.
                scene.frame_start = wp.zeros(2, dtype=float)
                scene.frame_end = wp.zeros(2, dtype=float)
                scene.solver = SimpleNamespace(reset=lambda *args, **kwargs: None)

            with patch.object(packing.Example, "__init__", build_scene):
                scene = Example(None, None)
            starts = wp.array([0, 1, 2], dtype=int)
            joint_q, joint_qd = wp.zeros(2, dtype=float), wp.zeros(2, dtype=float)
            for alpha in (1 / 6, 1.0):
                wp.launch(
                    _prescribe,
                    2,
                    [scene.frame_start, scene.frame_end, starts, starts, alpha, joint_q, joint_qd],
                )
                np.testing.assert_array_equal(joint_q.numpy(), solved)
                np.testing.assert_array_equal(joint_qd.numpy(), [0, 0])
            scene.frame_start.zero_()
            scene.frame_end.zero_()
            scene.frame, scene.sim_time = 20, 1.0
            scene.reset_physics()
            np.testing.assert_array_equal(scene.frame_start.numpy(), solved)
            np.testing.assert_array_equal(scene.frame_end.numpy(), solved)
            self.assertEqual((scene.frame, scene.sim_time), (0, 0.0))

    def test_shared_physics_defaults_and_no_xr_options(self):
        """Share current physical defaults and expose only scene controls in the standalone demo."""
        demo = Example.create_parser().parse_args([])
        teleop = TeleopExample.create_parser().parse_args([])
        for args in (demo, teleop):
            self.assertEqual((args.substeps, args.iterations, args.pvc_handle_stiffness), (6, 16, 10.0))
            self.assertEqual(args.pvc_solver, "reference")
        for name in ("pvc_assets", "grocery_assets", "robot_setback", "bag_variant", "snacks", "soft_cube"):
            self.assertEqual(getattr(demo, name), getattr(teleop, name))
        self.assertFalse(hasattr(demo, "webxr_server"))
        self.assertFalse(hasattr(demo, "trajectory_output"))

    def test_step_does_not_enter_teleoperation_or_automatic_planning(self):
        """Advance physics directly without asking either controller for a target."""
        scene = Example.__new__(Example)
        scene.frame, scene.sim_time, scene.frame_dt = 0, 0.0, 1 / 60
        with (
            patch.object(scene, "_advance_physics") as physics,
            patch.object(scene, "_prepare_frame", side_effect=AssertionError("XR input accessed")),
            patch.object(scene, "_plan", side_effect=AssertionError("Automatic task started")),
        ):
            scene.step()
        physics.assert_called_once_with()
        self.assertEqual(scene.frame, 1)
        self.assertEqual(scene.sim_time, 1 / 60)


@unittest.skipUnless(
    (Path.home() / "下载/clear_plastic_bag/clear_plastic_bag.glb").is_file()
    and (Path.home() / "下载/scale_aligned_usd_minimal_20260918/glue/glue.usd").is_file(),
    "Requires authored PVC and grocery assets",
)
class TestPVCPackingScene(unittest.TestCase):
    def test_same_geometry_material_contacts_and_solver_configuration(self):
        """Compare both scene models on CPU without creating solvers, IK or services."""
        with wp.ScopedDevice("cpu"):
            demo = Example.create_render_scene(ViewerNull(), Example.create_parser().parse_args([]))
            teleop = TeleopExample.create_render_scene(ViewerNull(), TeleopExample.create_parser().parse_args([]))
        for name in (
            "particle_q",
            "particle_mass",
            "particle_radius",
            "tri_indices",
            "tri_materials",
            "edge_indices",
            "edge_bending_properties",
            "body_q",
            "body_mass",
            "body_flags",
            "shape_transform",
            "shape_scale",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
        ):
            np.testing.assert_array_equal(getattr(demo.model, name).numpy(), getattr(teleop.model, name).numpy())
        self.assertEqual(demo._vbd_options(), teleop._vbd_options())
        self.assertEqual(demo.model.spring_count, 0)
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            self.assertEqual(getattr(demo.model, name), getattr(teleop.model, name))
        np.testing.assert_array_equal(demo.pick, teleop.pick)
        self.assertEqual(demo.table_z, teleop._table_height)
        self.assertEqual(demo.kinds, ("soda", "glue"))
        self.assertIs(Example._build_simulation, packing.Example._build_simulation)
        # Exercise the standalone constructor with only the expensive physical
        # build substituted; no teleoperation constructor may be invoked.
        instance = Example.__new__(Example)
        instance.model, instance.state_0 = demo.model, demo.state_0
        with wp.ScopedDevice("cpu"):
            instance.frame_start = wp.zeros_like(demo.state_0.joint_q)
            instance.frame_end = wp.zeros_like(demo.state_0.joint_q)
        with (
            patch.object(packing.Example, "__init__") as physical_init,
            patch.object(TeleopExample, "__init__", side_effect=AssertionError("Teleoperation runtime created")),
        ):
            Example.__init__(instance, demo.viewer, demo.args)
        physical_init.assert_called_once_with(instance, demo.viewer, demo.args)
        self.assertFalse(hasattr(instance, "xr_state"))
        self.assertFalse(hasattr(instance, "trajectory_recorder"))
        instance.close()


if __name__ == "__main__":
    unittest.main()
