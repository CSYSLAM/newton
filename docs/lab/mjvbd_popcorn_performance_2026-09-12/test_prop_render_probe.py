# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check render upload suppression still observes in-place appearance edits."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import warp as wp
from prop_render_probe import AppearanceCache, HiddenTriangleCache


class TestAppearanceCache(unittest.TestCase):
    def test_hidden_triangle_refresh(self):
        """Refresh triangle geometry when showing it again after hidden frames."""
        original = Mock()
        viewer = SimpleNamespace(
            show_triangles=False,
            model_changed=False,
            _layer_force_hidden=lambda: False,
            _log_triangles=original,
        )
        cache = HiddenTriangleCache(viewer)
        viewer._log_triangles(None)
        viewer._log_triangles(None)
        self.assertEqual(original.call_count, 1)
        viewer.show_triangles = True
        viewer._log_triangles(None)
        viewer._log_triangles(None)
        self.assertEqual(original.call_count, 3)
        viewer.show_triangles = False
        viewer._log_triangles(None)
        viewer._log_triangles(None)
        self.assertEqual(original.call_count, 4)
        viewer.model_changed = True
        viewer._log_triangles(None)
        self.assertEqual(original.call_count, 5)
        viewer.model_changed = False
        cache.enabled = False
        viewer._log_triangles(None)
        self.assertEqual(original.call_count, 6)

    def test_mutations(self):
        """Preserve first upload, in-place changes, and explicit model invalidation."""
        colors, opacity = Mock(), Mock()
        model = SimpleNamespace(
            shape_color=wp.array([[1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device="cpu"),
            shape_opacity=wp.array([1, 0.5], dtype=float, device="cpu"),
        )
        viewer = SimpleNamespace(
            model=model,
            device="cpu",
            model_changed=True,
            _sync_shape_colors_from_model=colors,
            _sync_shape_opacities_from_model=opacity,
        )
        cache = AppearanceCache(viewer)

        def sync():
            viewer._sync_shape_colors_from_model()
            viewer._sync_shape_opacities_from_model()

        sync()
        viewer.model_changed = False
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (1, 1))
        model.shape_color.assign([[0, 0, 1], [0, 1, 0]])
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (2, 1))
        model.shape_opacity.assign([0.25, 0.5])
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (2, 2))
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (2, 2))
        viewer.model_changed = True
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (3, 3))
        viewer.model_changed = False
        cache.enabled = False
        sync()
        self.assertEqual((colors.call_count, opacity.call_count), (4, 4))


if __name__ == "__main__":
    unittest.main()
