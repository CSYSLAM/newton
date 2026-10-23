# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Physical blade damage and affine-exact live surface embedding."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjmpm.support.bread_cut_geometry import blade_mesh
from newton.examples.mjmpm.support.bread_cut_material import apply_damage, damage_at_blade
from newton.examples.mjmpm.support.bread_cut_visuals import BreadCutVisuals, _skin, embedding, group_visual_parts


class TestBreadCut(unittest.TestCase):
    def test_opaque_batching_preserves_triangles_and_material_order(self):
        asset = Path(__file__).resolve().parents[2] / "examples/mjmpm/assets/bread_cut/visuals.npz"
        with np.load(asset) as archive:
            manifest = [p for p in json.loads(str(archive["manifest"])) if p["parent"] != "bread"]
            reference = group_visual_parts(archive, manifest, batch_opaque=False)
            batched = group_visual_parts(archive, manifest)
        self.assertLess(len(batched), len(reference))
        for slot in (1, 2, 3):
            expected = np.concatenate([g[slot][g[4]] for g in reference])
            actual = np.concatenate([g[slot][g[4]] for g in batched])
            np.testing.assert_array_equal(actual, expected)

        def materials(groups):
            return [
                (p["parent"], tuple(p["color"]), p["roughness"], p["metallic"], p["texture"])
                for p, _v, _n, _uv, indices in groups
                for _ in range(len(indices) // 3)
            ]

        self.assertEqual(materials(reference), materials(batched))

    def test_default_assets_do_not_render_a_predefined_bread_cut(self):
        class MeshRecorder:
            def __init__(self):
                self.meshes = set()

            def log_mesh(self, name, *args, **kwargs):
                self.meshes.add(name)

            def log_instances(self, *args, **kwargs):
                pass

        viewer = MeshRecorder()
        # Empty material is deliberate: the default asset renderer must not
        # construct either of the predefined bread surface embeddings.
        visuals = BreadCutVisuals(
            viewer,
            device="cpu",
            rest=np.empty((0, 3)),
            center=np.zeros(3),
            cut_y=-0.275,
            tool_body=0,
            tool_local=wp.transform_identity(),
        )
        state = SimpleNamespace(body_q=wp.array([wp.transform_identity()], dtype=wp.transform, device="cpu"))
        visuals.render(state, wp.empty(0, dtype=float, device="cpu"))
        asset = Path(__file__).resolve().parents[2] / "examples/mjmpm/assets/bread_cut/visuals.npz"
        with np.load(asset) as archive:
            manifest = json.loads(str(archive["manifest"]))
        forbidden = {"/bread_cut/" + p["key"] for p in manifest if p["parent"] == "bread"}
        self.assertTrue(viewer.meshes)
        self.assertFalse(viewer.meshes & forbidden)

    def test_visual_and_physical_blade_match(self):
        points, faces = blade_mesh()
        # Every physical edge belongs to exactly two triangles, and outward
        # orientation gives the inertia builder a positive enclosed volume.
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        volume = (
            np.einsum("ij,ij->i", points[faces[:, 0]], np.cross(points[faces[:, 1]], points[faces[:, 2]])).sum() / 6
        )
        self.assertGreater(volume, 0)
        asset = Path(__file__).resolve().parents[2] / "examples/mjmpm/assets/bread_cut/visuals.npz"
        with np.load(asset) as archive:
            part = next(
                p for p in json.loads(str(archive["manifest"])) if p["name"] == "Knife / resolved spine and sharp edge"
            )
            visual = np.unique(archive[part["key"] + "_points"], axis=0)
        np.testing.assert_allclose(visual, np.unique(points, axis=0), atol=1.0e-7)

    def test_blade_damage_is_local_and_irreversible(self):
        devices = [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]
        for device in devices:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                local = np.array([[0.12, 0.0, -0.055], [0.12, 0.02, -0.055], [0.12, 0.0, 0.02], [0.30, 0.0, -0.055]])
                pose = wp.transform(wp.vec3(0.4, -0.2, 0.9), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7))
                offset = wp.transform(wp.vec3(0.02, 0.0, 0.01), wp.quat_identity())
                world = wp.mul(pose, offset)
                points = wp.array([wp.transform_point(world, wp.vec3(*p)) for p in local], dtype=wp.vec3)
                body = wp.array([pose], dtype=wp.transform)
                damage = wp.zeros(4, dtype=float)
                shear = wp.empty(4, dtype=float)
                tension = wp.empty(4, dtype=float)

                def update(points=points, body=body, offset=offset, damage=damage, shear=shear, tension=tension):
                    wp.launch(damage_at_blade, 4, [points, body, 0, offset, 0.011, damage])
                    wp.launch(apply_damage, 4, [damage, 3000.0, 0.3, shear, tension])

                update()
                np.testing.assert_array_equal(damage.numpy(), [1, 0, 0, 0])
                np.testing.assert_allclose(shear.numpy(), [0, 3000, 3000, 3000])
                np.testing.assert_allclose(tension.numpy(), [0, 0.3, 0.3, 0.3])
                # Withdrawing the blade cannot heal the cut. Replay the same
                # captured operations with a different device pose buffer.
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        update()
                    body.assign(
                        wp.array([wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())], dtype=wp.transform)
                    )
                    wp.capture_launch(capture.graph)
                else:
                    body.assign(
                        wp.array([wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())], dtype=wp.transform)
                    )
                    update()
                np.testing.assert_array_equal(damage.numpy(), [1, 0, 0, 0])
                np.testing.assert_allclose(shear.numpy(), [0, 3000, 3000, 3000])

    def test_skin_tracks_affine_material_deformation(self):
        rng = np.random.default_rng(41)
        rest = rng.uniform(-0.05, 0.05, (128, 3))
        points = rng.uniform(-0.045, 0.045, (19, 3))
        # A patch must follow only its own side, including at the cut surface.
        candidates = np.flatnonzero(rest[:, 1] < 0)
        ids, coefficients = embedding(points, rest, candidates)
        self.assertTrue(np.all(rest[ids, 1] < 0))
        deformation = np.array([[1.1, 0.1, 0.0], [0.0, 0.7, 0.2], [0.05, 0.0, 0.9]])
        translation = np.array([0.4, -0.2, 0.9])
        current = rest @ deformation.T + translation
        normal = np.array([0.0, 0.0, 1.0])
        expected_normal = np.linalg.inv(deformation).T @ normal
        expected_normal /= np.linalg.norm(expected_normal)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                vertices = wp.empty(len(points), dtype=wp.vec3)
                normals = wp.empty(len(points), dtype=wp.vec3)
                wp.launch(
                    _skin,
                    len(points),
                    [
                        wp.array(current, dtype=wp.vec3),
                        wp.array(ids, dtype=int),
                        wp.array(coefficients, dtype=wp.vec4),
                        wp.array(np.tile(normal, (len(points), 1)), dtype=wp.vec3),
                        vertices,
                        normals,
                    ],
                )
                np.testing.assert_allclose(vertices.numpy(), points @ deformation.T + translation, atol=2.0e-6)
                np.testing.assert_allclose(normals.numpy(), np.tile(expected_normal, (len(points), 1)), atol=3.0e-5)


if __name__ == "__main__":
    unittest.main()
