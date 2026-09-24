# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check PVC topology, independent layout, material, controls and replay on CPU."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2._webxr_teleop import JsonlTrajectoryRecorder
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_pvc import Example
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example as RopeExample
from newton.examples.mjvbdv2.support.w1_bag_recording import (
    SCENE_OPTIONS,
    TeleopRecordingReader,
    load_replay_scene,
    scene_signature,
)
from newton.solvers import SolverMJVBDV2
from newton.viewer import ViewerNull


@unittest.skipUnless(
    (Path.home() / "下载/clear_plastic_bag/clear_plastic_bag.glb").is_file()
    and (Path.home() / "下载/scale_aligned_usd_minimal_20260918/glue/glue.usd").is_file(),
    "Requires the authored PVC and grocery assets",
)
class TestPVCPacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the authored scene without GPU, IK, SDF baking or a server."""
        with wp.ScopedDevice("cpu"):
            cls.scene = Example.create_render_scene(ViewerNull(), Example.create_parser().parse_args([]))

    def test_welded_surface_has_only_one_open_mouth(self):
        """Keep all handles and the bottom joined with consistently wound, positive-area faces."""
        import trimesh

        s = self.scene
        mesh = trimesh.Trimesh(s.rest, s.faces, process=False)
        edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(len(mesh.split(only_watertight=False)), 1)
        self.assertLessEqual(counts.max(), 2)
        np.testing.assert_array_equal(np.unique(edges[counts == 1]), s.rim)
        self.assertGreater(mesh.area_faces.min(), 1e-7)
        self.assertLess(len(s.rest), 2500)
        np.testing.assert_array_equal(np.unique(s.faces), np.arange(len(s.rest)))
        self.assertTrue(np.all(s.model.particle_mass.numpy() > 0))

    def test_mouth_alignment_and_right_hand_props(self):
        """Match rope mouth XY and direction while clearing the table with the larger PVC bag."""
        s = self.scene
        reference = RopeExample.__new__(RopeExample)
        reference.table_z = s.table_z
        with np.load(packing.ROPE_HANDLE_ASSETS / "bag.npz") as data:
            reference._placement_rest = data["vertices"]
        position, rotation = reference._initial_bag_transform()
        matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
        mouth = position + matrix @ np.array((0, 0, packing.HEIGHT))
        q = s.state_0.particle_q.numpy()
        np.testing.assert_allclose(s.pvc_initial_transform[3:], rotation, atol=1e-7)
        np.testing.assert_allclose(q[s.rim].mean(0)[:2], mouth[:2], atol=3e-7)
        np.testing.assert_allclose(matrix @ (0, 0, 1), (0, -1, 0), atol=1e-6)
        opening = q[s.rim].mean(0) - q[s.bottom].mean(0)
        self.assertLess(abs(opening[2]), 0.001)
        self.assertLess(opening[1], -0.3)
        self.assertAlmostEqual(float(q[:, 2].min()), s.table_z + 0.002, places=6)
        self.assertTrue(np.all(s.pick[:, 1] < q[:, 1].min()))
        np.testing.assert_allclose(s.pick[:, 2] - s.half[:, 2], s.table_z + 0.002)
        np.testing.assert_allclose(s.pick[:, :2], [(0.46, -0.44), (0.46, -0.23)])
        self.assertEqual(s.kinds, ("soda", "glue"))
        self.assertEqual(RopeExample._grocery_names, ("toy", "soda", "biscuit"))
        self.assertTrue(np.all(q[:, :2] >= (s.table_center - packing.TABLE_HALF)[:2]))
        self.assertTrue(np.all(q[:, :2] <= (s.table_center + packing.TABLE_HALF)[:2]))

    def test_glb_and_source_mesh_alignment(self):
        """Record distinct transforms for the Y-up GLB and Z-up simulation source meshes."""
        import trimesh

        metadata = self.scene.pvc["source"]
        root = Path(metadata["directory"])
        glb = trimesh.load(root / "clear_plastic_bag.glb", process=False)
        ply = trimesh.load(root / "source_meshes/Bag_body_wrinkled_film_0p24mm.ply", process=False)
        source = ply.vertices @ np.asarray(metadata["sourceToBag"]).T
        rendered = glb.geometry["Bag_body_wrinkled_film_0p24mm"].vertices @ np.asarray(metadata["glbToBag"]).T
        np.testing.assert_allclose(rendered.min(0), source.min(0), atol=1e-7)
        np.testing.assert_allclose(rendered.max(0), source.max(0), atol=1e-7)

    def test_supported_shell_without_support_springs(self):
        """Stiffen shell bending while retaining paper membrane, mass and flexible handles."""
        s = self.scene
        materials = s.model.tri_materials.numpy()
        np.testing.assert_array_equal(np.unique(materials[: s.paper_faces, 0]), [1.5e5, 3e5])
        np.testing.assert_array_equal(materials[:, 0], materials[:, 1])
        np.testing.assert_allclose(
            materials[s.paper_faces :, :3], np.tile([1e6, 1e6, 0.4], (len(s.faces) - s.paper_faces, 1))
        )
        hem = s.rest[s.faces[: s.paper_faces], 2].min(axis=1) > s.rest[s.rim, 2].min() - 0.032
        np.testing.assert_allclose(materials[: s.paper_faces, 2], np.where(hem, 0.8, 0.4))
        edges = s.model.edge_indices.numpy()
        bending = s.model.edge_bending_properties.numpy()
        handles = edges.max(axis=1) >= s.paper_count
        np.testing.assert_allclose(np.unique(bending[~handles], axis=0), [[1440, 4]])
        np.testing.assert_allclose(bending[handles], np.tile([300, 0.1], (handles.sum(), 1)))
        self.assertEqual(s.model.spring_count, 0)
        areas = s.model.tri_areas.numpy()
        masses = np.zeros(len(s.rest))
        face_mass = areas * packing.BAG_SURFACE_DENSITY
        face_mass[: s.paper_faces] *= np.where(hem, 2, 1)
        np.add.at(masses, s.faces.ravel(), np.repeat(face_mass / 3, 3))
        np.testing.assert_allclose(s.model.particle_mass.numpy(), masses, rtol=2e-6)
        self.assertEqual(s._recording_extras()["pvcBagAsset"]["materialModel"], "elastic")

    def test_reference_solver_settings_and_explicit_paper_comparison(self):
        """Disable CUDA-only solve paths by default while retaining paper material and contacts."""
        s = self.scene
        self.assertEqual((s.args.substeps, s.args.iterations), (6, 16))
        self.assertEqual(s.args.pvc_solver, "reference")
        expected = packing.Example._vbd_options(s)
        expected.update(
            particle_enable_multilevel_correction=False,
            particle_enable_tile_solve=False,
            particle_enable_surface_cache=False,
            particle_enable_truncation_cache=False,
            enable_cuda_fast_path=False,
        )
        self.assertEqual(s._vbd_options(), expected)
        reference = Example.__new__(Example)
        reference.__dict__.update(s.__dict__)
        reference.args = Example.create_parser().parse_args(["--pvc-solver", "paper"])
        expected = packing.Example._vbd_options(reference)
        expected.update(particle_enable_surface_cache=True, particle_enable_truncation_cache=True)
        self.assertEqual(reference._vbd_options(), expected)

    def test_cpu_settling_preserves_handle_clearance(self):
        """Keep a free handle segment graspable after three seconds without pinning the bag."""
        with wp.ScopedDevice("cpu"):
            scene = Example.__new__(Example)
            scene.args = Example.create_parser().parse_args([])
            scene.table_z = scene._table_height
            builder = newton.ModelBuilder()
            scene._add_bag(builder)
            # A prescribed support selects W1's full-contact kinematic backend.
            table = builder.add_body(
                xform=wp.transform((0.66, -0.01, scene.table_z - 0.022), wp.quat_identity()),
                mass=1.0,
                is_kinematic=True,
            )
            builder.add_shape_box(table, hx=0.3, hy=0.66, hz=0.022, cfg=builder.ShapeConfig(ke=4e4, kd=80, mu=0.65))
            builder.color(include_bending=True)
            scene.model = builder.finalize()
            scene.model.soft_contact_ke, scene.model.soft_contact_kd, scene.model.soft_contact_mu = 2e5, 10.0, 0.6
            scene.soft_cube_start = scene.soft_cube_end = len(scene.rest)
            scene.soft_cube_faces = np.empty((0, 3), dtype=np.int32)
            solver = SolverMJVBDV2(
                scene.model,
                mujoco_articulations=(0,),
                joint_mode="kinematic",
                contact_mode="full",
                vbd_options=scene._vbd_options(),
                collision_options={
                    "enable_rigid_soft_full_surface_contact": True,
                    "rigid_soft_full_surface_shape_indices": [0],
                    "soft_contact_margin": 0.004,
                    "soft_contact_max": 131072,
                },
            )
            current, following = scene.model.state(), scene.model.state()
            self.assertFalse(solver.vbd_solver.use_particle_tile_solve)
            self.assertIsNone(solver.vbd_solver.particle_multilevel)
            self.assertIsNone(solver.vbd_solver._surface_cached_kernel)
            self.assertIsNone(solver.vbd_solver._particle_truncation_cache)
            control = scene.model.control()
            rest = current.particle_q.numpy().copy()
            mass = scene.model.particle_mass.numpy()
            initial_com_z = np.average(rest[:, 2], weights=mass)
            for frame in range(180):
                for _ in range(scene.args.substeps):
                    current.clear_forces()
                    solver.step(current, following, control, None, 1 / (60 * scene.args.substeps))
                    current, following = following, current
                q = current.particle_q.numpy()
                self.assertTrue(np.isfinite(q).all())
                self.assertGreater(q[:, 2].min(), scene.table_z - 0.002)
                self.assertLess(q[:, 2].max(), scene.table_z + 0.18)
                self.assertLess(np.average(q[:, 2], weights=mass), initial_com_z + 0.005)
                opening = q[scene.rim].mean(0) - q[scene.bottom].mean(0)
                self.assertLess(abs(opening[2]), 0.075)
                self.assertLess(opening[1], -0.3)
                if frame >= 29 and (frame + 1) % 30 == 0:
                    # A drooping crown is fine. Require a continuous 2 cm free
                    # segment at least 4 cm above the table, away from welds.
                    longest = 0.0
                    for handle in scene.pvc["handles"]:
                        centers = q[handle].mean(axis=1)[2:-2]
                        clear = centers[:, 2] > scene.table_z + 0.04
                        run = 0.0
                        for length, a, b in zip(
                            np.linalg.norm(np.diff(centers, axis=0), axis=1), clear[:-1], clear[1:], strict=True
                        ):
                            run = run + length if a and b else 0.0
                            longest = max(longest, run)
                    self.assertGreater(longest, 0.02, f"No graspable handle segment at frame {frame + 1}")
            self.assertGreater(np.linalg.norm(q - rest, axis=1).max(), 1e-4)
            self.assertTrue(np.all(scene.model.particle_mass.numpy() > 0))
            self.assertEqual(scene.model.spring_count, 0)

    def test_arms_unlimited_and_both_grippers_rate_limited(self):
        """Follow both arms immediately and limit both jaw directions without overshoot."""
        s = Example.__new__(Example)
        s.frame_dt = 1 / 60
        s.args = Example.create_parser().parse_args([])
        s.arm_indices = np.array([0, 1])
        s.finger_indices = {"left": np.array([2, 3]), "right": np.array([4, 5])}
        s.inputs = {side: SimpleNamespace(jaws=np.zeros(2)) for side in s.finger_indices}
        previous = np.full(6, 0.04)
        solved = np.full(6, 0.8)
        q = s._limit_joint_targets(previous, solved)
        np.testing.assert_array_equal(q[:2], solved[:2])
        np.testing.assert_allclose(q[2:], 0.04 - 0.035 / 60)
        for control in s.inputs.values():
            control.jaws[:] = 0.05
        np.testing.assert_allclose(s._limit_joint_targets(previous, solved)[2:], 0.04 + 0.035 / 60)
        for control in s.inputs.values():
            control.jaws[:] = [0.0399, 0.0401]
        np.testing.assert_allclose(s._limit_joint_targets(previous, solved)[2:], [0.0399, 0.0401] * 2)

    def test_webxr_translucent_bag_geometry(self):
        """Export two translucent PVC parts and no paper-colored bag."""
        s = self.scene
        s._bag_meshes, s._soft_cube_mesh, s._static_boxes = [], None, []
        payload = s._build_webxr_geometry()
        size = struct.unpack_from("<I", payload, 4)[0]
        header = json.loads(payload[8 : 8 + size])
        shapes = [shape for shape in header["shapes"] if shape["role"] == "bag"]
        self.assertEqual(len(shapes), 2)
        self.assertEqual([shape["opacity"] for shape in shapes], [0.32, 0.72])
        self.assertTrue(all(shape["doubleSided"] for shape in shapes))

    def test_recording_selects_pvc_scene_and_exact_states(self):
        """Round-trip the new recording through the shared replay without constructing physics."""
        self.assertEqual(self.scene.pvc["source"]["proxyVersion"], 2)
        self._check_recording_round_trip(self.scene)

    def test_legacy_pvc_recording_retains_fine_shell(self):
        """Rebuild the original proxy when replaying a version-one recording."""
        args = Example.create_parser().parse_args([])
        args.pvc_proxy_version = 1
        with wp.ScopedDevice("cpu"):
            scene = Example.create_render_scene(ViewerNull(), args)
        self.assertEqual(len(scene.rest), 2045)
        self.assertEqual(len(scene.faces), 4040)
        self.assertEqual(scene.pvc["source"]["proxyVersion"], 1)
        self._check_recording_round_trip(scene)

    def _check_recording_round_trip(self, s):
        """Verify metadata selects the recorded mesh and restores every particle exactly."""
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            path = Path(directory) / "pvc.jsonl"
            recorder = JsonlTrajectoryRecorder(
                path,
                {
                    "scene": "w1-bag-packing",
                    "recordingKind": "full-state",
                    "frameDtSeconds": 1 / 60,
                    "sceneSignature": scene_signature(s),
                    "sceneOptions": {name: getattr(s.args, name) for name in SCENE_OPTIONS},
                    **s._recording_extras(),
                },
            )
            recorder.start()
            recorder.append(
                {
                    "simulationTimeSeconds": 0,
                    **{
                        key: getattr(s.state_0, field).numpy().tolist()
                        for field, key in TeleopRecordingReader._fields.items()
                    },
                }
            )
            recorder.close()
            args = packing.Example.create_parser().parse_args(["--replay", str(path)])
            rendered, playback = load_replay_scene(packing.Example, ViewerNull(), args)
            self.assertEqual(rendered.kinds, s.kinds)
            self.assertFalse(hasattr(rendered, "solver"))
            playback.recording.restore(rendered, 0)
            rendered.render()
            np.testing.assert_array_equal(rendered.state_0.particle_q.numpy(), s.state_0.particle_q.numpy())


if __name__ == "__main__":
    unittest.main()
