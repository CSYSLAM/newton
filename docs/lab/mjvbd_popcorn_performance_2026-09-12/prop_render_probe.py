# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain immutable prop uploads and update only moving instance transforms."""

import warp as wp

import newton
from newton.viewer import ViewerGL


@wp.kernel(enable_backward=False)
def detect_appearance_changes(
    colors: wp.array[wp.vec3],
    opacity: wp.array[float],
    previous_colors: wp.array[wp.vec3],
    previous_opacity: wp.array[float],
    changed: wp.array[int],
):
    i = wp.tid()
    for channel in range(3):
        if colors[i][channel] != previous_colors[i][channel]:
            wp.atomic_max(changed, 0, 1)
    if opacity[i] != previous_opacity[i]:
        wp.atomic_max(changed, 1, 1)
    previous_colors[i] = colors[i]
    previous_opacity[i] = opacity[i]


class AppearanceCache:
    """Detect actual model changes before triggering per-batch GL uploads."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.colors = viewer._sync_shape_colors_from_model
        self.opacity = viewer._sync_shape_opacities_from_model
        self.old_colors = wp.clone(viewer.model.shape_color)
        self.old_opacity = wp.clone(viewer.model.shape_opacity)
        self.changed = wp.zeros(2, dtype=int, device=viewer.device)
        self.enabled = True
        self.flags = (True, True)
        viewer._sync_shape_colors_from_model = self.sync_colors
        viewer._sync_shape_opacities_from_model = self.sync_opacity

    def sync_colors(self):
        self.changed.zero_()
        wp.launch(
            detect_appearance_changes,
            dim=self.old_colors.size,
            inputs=[
                self.viewer.model.shape_color,
                self.viewer.model.shape_opacity,
                self.old_colors,
                self.old_opacity,
                self.changed,
            ],
            device=self.viewer.device,
        )
        self.flags = self.changed.numpy()
        if not self.enabled or self.viewer.model_changed or self.flags[0]:
            self.colors()

    def sync_opacity(self):
        if not self.enabled or self.viewer.model_changed or self.flags[1]:
            self.opacity()


class HiddenTriangleCache:
    """Refresh debug triangle meshes only when visible or changing visibility."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.original = viewer._log_triangles
        self.was_hidden = False
        self.enabled = True
        viewer._log_triangles = self.log_triangles

    def log_triangles(self, state):
        hidden = not self.viewer.show_triangles or self.viewer._layer_force_hidden()
        if not self.enabled or not hidden or not self.was_hidden or self.viewer.model_changed:
            self.original(state)
        self.was_hidden = hidden


class PropRenderCache:
    def __init__(self, example):
        if not isinstance(example.viewer, ViewerGL):
            raise ValueError("Prop caching requires the persistent OpenGL viewer")
        self.viewer = example.viewer
        self.original = self.viewer.log_shapes
        self.props = {"/props/" + name: body for name, body, *_ in example.prop_render_data}
        self.meshes = {}
        self.enabled = True
        self.viewer.log_shapes = self.log_shapes

    def log_shapes(self, name, geo_type, geo_scale, xforms, *args, **kwargs):
        if not self.enabled or name not in self.props:
            return self.original(name, geo_type, geo_scale, xforms, *args, **kwargs)
        if geo_type != newton.GeoType.MESH or tuple(geo_scale) != (1.0, 1.0, 1.0) or len(xforms) != 1:
            raise ValueError("Prop cache only accepts the demo's immutable single-mesh assets")
        qualified = self.viewer._qualify(name)
        if name not in self.meshes or qualified not in self.viewer.objects:
            result = self.original(name, geo_type, geo_scale, xforms, *args, **kwargs)
            self.meshes[name] = self.viewer._populate_geometry(
                int(geo_type), tuple(geo_scale), 0.0, True, geo_src=kwargs["geo_src"]
            )
            return result
        if self.props[name] < 0:
            return None
        # None preserves previously uploaded colors, materials and opacity.
        return self.viewer.log_instances(name, self.meshes[name], xforms, None, None, None)
