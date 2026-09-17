# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact GPU appearance change detection for persistent OpenGL instances."""

import warp as wp


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
