# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU geometry and robot-path checks for the supermarket example."""

import json
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import warp as wp


@unittest.skipUnless(find_spec("mujoco") is not None, "Requires the MuJoCo example dependency")
class TestSupermarketGeometry(unittest.TestCase):
    def test_piper_native_limits_and_pads(self):
        """Preserve source limits and match visible pads to contact geometry."""
        from newton.examples.mjvbdv2.example_mjvbd_v2_supermarket_packing import ASSETS, robot_xml  # noqa: PLC0415

        source = ET.parse(ASSETS / "piper/piper_with_texture.xml").getroot()
        adapted = ET.fromstring(robot_xml())
        for joint in source.findall("worldbody/.//joint"):
            current = adapted.find(f"worldbody/.//joint[@name='{joint.get('name')}']")
            for attribute in ("axis", "range", "type"):
                self.assertEqual(current.get(attribute), joint.get(attribute))
        for name in ("link7", "link8"):
            visual = adapted.find(f".//geom[@name='{name}_rubber_pad_visual']")
            contact = adapted.find(f".//geom[@name='{name}_rubber_pad_collision']")
            for attribute in ("size", "pos", "type"):
                self.assertEqual(visual.get(attribute), contact.get(attribute))

    def test_prepared_pantry_assets(self):
        """Verify prepared pantry meshes have metric pivots and usable UVs."""
        assets = Path(__file__).resolve().parents[1] / "examples/mjvbdv2/assets/supermarket"
        groups = json.loads((assets / "pantry_meshes.json").read_text())
        self.assertEqual(len(groups), 4)
        for name, group in groups.items():
            vertices, uv = np.asarray(group["vertices"]), np.asarray(group["uvs"])
            self.assertEqual(uv.shape, (len(vertices), 2))
            self.assertTrue(np.isfinite(vertices).all())
            self.assertAlmostEqual(vertices[:, 2].min(), 0, places=6)
            np.testing.assert_allclose(vertices[:, :2].min(0) + vertices[:, :2].max(0), 0, atol=1e-6)
            self.assertTrue((assets / group["texture"]).is_file())
            self.assertTrue((assets / "usd" / (name + ".usdc")).is_file())
            self.assertGreaterEqual(uv.min(), -1e-5)
            self.assertLessEqual(uv.max(), 1 + 1e-5)

    def test_decor_normals(self):
        """Verify Blender split normals survive the batched decoration export."""
        assets = Path(__file__).resolve().parents[1] / "examples/mjvbdv2/assets/supermarket"
        groups = json.loads((assets / "checkout_decor.json").read_text())
        for group in groups.values():
            vertices, normals = np.asarray(group["vertices"]), np.asarray(group["normals"])
            indices = np.asarray(group["indices"])
            self.assertEqual(vertices.shape, normals.shape)
            self.assertTrue(np.isfinite(vertices).all())
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-6)
            self.assertGreaterEqual(indices.min(), 0)
            self.assertLess(indices.max(), len(vertices))

    def test_bread_reference_mesh(self):
        """Verify the domed bread has positive tetrahedra and a flat base."""
        from newton.examples.mjvbdv2.example_mjvbd_v2_supermarket_packing import bread_mesh  # noqa: PLC0415

        vertices, tets = bread_mesh()
        a, b, c, d = (vertices[tets[:, i]] for i in range(4))
        volume = np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6
        self.assertGreater(volume.min(), 0)
        self.assertGreater(vertices[:, 2].max(), 0.025)
        self.assertAlmostEqual(float(vertices[:, 2].min()), -0.030, places=6)
        self.assertGreater(np.unique(np.round(vertices[:, 2], 6)).size, 6)

    def test_bag_surface(self):
        """Verify the hanging film bag is welded and has no rim supports."""
        assets = Path(__file__).resolve().parents[1] / "examples/mjvbdv2/assets/supermarket"
        bag = json.loads((assets / "shopping_bag.json").read_text())
        vertices = np.asarray(bag["vertices"])
        triangles = np.asarray(bag["triangles"])
        areas = np.linalg.norm(
            np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            ),
            axis=1,
        )
        self.assertGreater(areas.min(), 1e-8)
        edges = Counter(
            tuple(sorted((int(a), int(b)))) for face in triangles for a, b in zip(face, np.roll(face, -1), strict=True)
        )
        self.assertEqual(max(edges.values()), 2)
        self.assertTrue(any(count == 1 for count in edges.values()))  # Open mouth/handles.
        self.assertEqual(bag["handle_support_vertices"], [])
        self.assertEqual(bag["support_mode"], "contact_only_slotted_handles")
        neighbours = [set() for _ in vertices]
        for a, b in edges:
            neighbours[a].add(b)
            neighbours[b].add(a)
        visited, pending = set(), [0]
        while pending:
            i = pending.pop()
            if i not in visited:
                visited.add(i)
                pending.extend(neighbours[i] - visited)
        self.assertEqual(len(visited), len(vertices))

    def test_sealed_pouch(self):
        """Verify the snack pouch is a closed positive-volume membrane."""
        # Defer the example's optional MuJoCo dependency until after the skip.
        from newton.examples.mjvbdv2.example_mjvbd_v2_supermarket_packing import pouch_mesh  # noqa: PLC0415

        vertices, triangles = pouch_mesh()
        edges = Counter(
            tuple(sorted((int(a), int(b)))) for face in triangles for a, b in zip(face, np.roll(face, -1), strict=True)
        )
        self.assertTrue(all(count == 2 for count in edges.values()))
        volume = (
            np.einsum(
                "ij,ij->i", vertices[triangles[:, 0]], np.cross(vertices[triangles[:, 1]], vertices[triangles[:, 2]])
            ).sum()
            / 6
        )
        self.assertGreater(volume, 0)

    def test_robot_path(self):
        """Verify the raised transfer clears the hanging rack and is reachable."""
        import mujoco

        # Defer the example's optional MuJoCo dependency until after the skip.
        from newton.examples.mjvbdv2.example_mjvbd_v2_supermarket_packing import (  # noqa: PLC0415
            BAG_POSITION,
            OPEN,
            ROBOT_INITIAL,
            Example,
            robot_xml,
            target,
        )

        robot = Example.__new__(Example)
        robot.mj_model = mujoco.MjModel.from_xml_string(robot_xml())
        robot.mj_data = mujoco.MjData(robot.mj_model)
        robot.site = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "packing_tcp")
        robot.jac_p = np.zeros((3, robot.mj_model.nv))
        robot.jac_r = np.zeros_like(robot.jac_p)
        robot.mj_data.qpos[:] = ROBOT_INITIAL
        for time in np.linspace(0, 26, 261):
            position, opening, _ = target(time)
            robot._ik(position, opening)
            mujoco.mj_forward(robot.mj_model, robot.mj_data)
            for i in range(robot.mj_model.ngeom):
                name = mujoco.mj_id2name(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                if name.startswith(("link7_", "link8_")):
                    if robot.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                        mesh = robot.mj_model.geom_dataid[i]
                        start = robot.mj_model.mesh_vertadr[mesh]
                        count = robot.mj_model.mesh_vertnum[mesh]
                        vertices = robot.mj_model.mesh_vert[start : start + count]
                    else:
                        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                        vertices = vertices * robot.mj_model.geom_size[i]
                    world = vertices @ robot.mj_data.geom_xmat[i].reshape(3, 3).T + robot.mj_data.geom_xpos[i]
                    self.assertGreater(world[:, 2].min(), 0.72)
                    if 10 <= time <= 11 or 22 <= time <= 23:
                        self.assertGreater(world[:, 2].min(), BAG_POSITION[2] + 0.27)
            self.assertLess(np.linalg.norm(robot.mj_data.site_xpos[robot.site] - position), 0.001)
            np.testing.assert_allclose(
                robot.mj_data.site_xmat[robot.site].reshape(3, 3), np.diag([1.0, -1.0, -1.0]), atol=0.002
            )
        for i in range(robot.mj_model.ngeom):
            name = mujoco.mj_id2name(robot.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and ("_visual_" in name or name.endswith("_visual")):
                self.assertEqual(robot.mj_model.geom_contype[i], 0)
                self.assertEqual(robot.mj_model.geom_conaffinity[i], 0)
        self.assertAlmostEqual(robot.mj_data.qpos[6], OPEN)
        self.assertAlmostEqual(robot.mj_data.qpos[7], -OPEN)

    def test_film_drag_dissipates_energy(self):
        """Verify still-air drag removes energy without moving fixed vertices."""
        from newton.examples.mjvbdv2.example_mjvbd_v2_supermarket_packing import (  # noqa: PLC0415
            accumulate_film_drag,
            apply_film_drag,
        )

        q = wp.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=wp.vec3, device="cpu")
        v = wp.array([(1, -2, 3)] * 3, dtype=wp.vec3, device="cpu")
        faces = wp.array([[0, 1, 2]], dtype=wp.int32, device="cpu")
        inv_mass = wp.array([0, 100, 100], dtype=float, device="cpu")
        drag = wp.zeros(3, dtype=wp.mat33, device="cpu")
        force = wp.zeros(3, dtype=wp.vec3, device="cpu")
        wp.launch(accumulate_film_drag, 1, [faces, q, v, drag, 1.2, 1.2, 0.02], device="cpu")
        wp.launch(apply_film_drag, 3, [v, inv_mass, drag, 0.5, force], device="cpu")
        force_np, velocity = force.numpy(), v.numpy()
        after = velocity + 0.5 * inv_mass.numpy()[:, None] * force_np
        np.testing.assert_array_equal(force_np[0], np.zeros(3))
        self.assertTrue(np.all(np.sum(force_np * velocity, axis=1) <= 0))
        self.assertTrue(np.all(np.linalg.norm(after, axis=1) <= np.linalg.norm(velocity, axis=1)))


if __name__ == "__main__":
    unittest.main()
