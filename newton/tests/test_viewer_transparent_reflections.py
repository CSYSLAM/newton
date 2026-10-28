# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render checks for opt-in thin-glass reflections and default isolation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip


class TestTransparentReflections(unittest.TestCase):
    def test_reflections_only_change_translucent_dielectrics(self):
        with wp.ScopedDevice("cpu"):
            viewer = _make_headless_viewer_gl_or_skip(self, width=160, height=160)
        self.addCleanup(viewer.close)
        self.assertFalse(viewer.renderer.transparent_reflections)
        viewer.set_camera(wp.vec3(2.0, 0.0, 0.8), pitch=-20.0, yaw=180.0)

        def render(state):
            viewer.begin_frame(0.0)
            viewer.log_state(state)
            viewer.end_frame()
            return viewer.get_frame().numpy().copy()

        for opacity, metallic, should_change in (
            (0.08, 0.0, True),
            (1.0, 0.0, False),
            (0.0, 0.0, False),
            (0.08, 1.0, False),
        ):
            with self.subTest(opacity=opacity, metallic=metallic):
                builder = newton.ModelBuilder()
                mesh = newton.Mesh.create_sphere(radius=0.5)
                mesh.metallic = metallic
                mesh.roughness = 0.12
                builder.add_shape_mesh(-1, mesh=mesh, opacity=opacity, color=(0.9, 0.95, 1.0))
                model = builder.finalize(device="cpu")
                viewer.set_model(model)
                state = model.state()
                viewer.set_transparent_reflections(False)
                baseline = render(state)
                viewer.set_transparent_reflections(True)
                reflected = render(state)
                if should_change:
                    changed = np.any(baseline != reflected, axis=2)
                    self.assertGreater(int(changed.sum()), 100)
                else:
                    np.testing.assert_array_equal(reflected, baseline)
                viewer.set_transparent_reflections(False)
                np.testing.assert_array_equal(render(state), baseline)

        viewer.set_transparent_reflections(True)
        viewer.clear_all_layers()
        self.assertFalse(viewer.renderer.transparent_reflections)


if __name__ == "__main__":
    unittest.main()
