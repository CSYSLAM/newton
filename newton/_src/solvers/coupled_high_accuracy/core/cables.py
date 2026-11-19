# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host side of cables (native ``precompute_cables``, ``cable_view``, the penalty ramp and its reset/copy API)."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .api import Span
from .kernels import cables as kernels


class CableHost:
    """Mixin of :class:`~.solver.Solver` with the cable host logic."""

    def _precompute_cables(self) -> None:
        model = self.model
        self.cable_count = model.cable_count * (self.count // self.per_world)
        empty_m, empty_i, empty_v4 = self._empty(wp.mat33), self._empty(wp.int32), self._empty(wp.vec4)
        self.cable_linear = self.cable_angular = empty_m
        self.cable_indices = self.cable_sizes = self.cable_initialized = empty_i
        self.cable_penalty = empty_v4
        if not self.cable_count:
            return
        n = model.cable_count
        for field in (
            "cable_body_indices",
            "cable_parent_transforms",
            "cable_child_transforms",
            "cable_stiffnesses",
            "cable_dampings",
            "cable_rest_bend_twist",
        ):
            self._require(getattr(model, field), n, field)
        self._require(model.body_centers_of_mass, model.body_count, "body_centers_of_mass")
        self._require(model.body_inertia_tensors, model.body_count, "body_inertia_tensors")
        if model.cable_enabled.size:
            self._require(model.cable_enabled, n, "cable_enabled")
        for field in ("cable_angular_bias", "cable_angular_hessian"):
            span = getattr(model, field)
            if span.size:
                self._require(span, self.cable_count, field)
        pairs = self._read(model.cable_body_indices, wp.vec2i, n)
        if np.any(pairs < -1) or np.any(pairs >= model.body_count):
            raise ValueError("cable references an invalid body")
        stiffness = self._read(model.cable_stiffnesses, wp.vec4, n)
        damping = self._read(model.cable_dampings, wp.vec4, n)
        if (
            not np.all(np.isfinite(stiffness))
            or np.any(stiffness < 0.0)
            or not np.all(np.isfinite(damping))
            or np.any(damping < 0.0)
        ):
            raise ValueError("cable stiffness and damping must be finite and nonnegative")
        self.cable_linear = self._zeros(self.cable_count * 8, wp.mat33)
        self.cable_angular = self._zeros(self.cable_count * 8, wp.mat33)
        self.cable_indices = self._zeros(self.cable_count * 8, wp.int32)
        self.cable_sizes = self._zeros(self.cable_count, wp.int32)
        self.cable_penalty = self._zeros(self.cable_count, wp.vec4)
        self.cable_initialized = self._zeros(self.cable_count, wp.int32)

    def _cable_ramping(self) -> bool:
        return self._config.cable_linear_beta > 0.0 or self._config.cable_angular_beta > 0.0

    def _cable_view(self) -> Any:
        model = self.model
        view = kernels.CableView()
        view.body_indices = self._view(model.affine_body_indices, wp.vec4i, model.body_count)
        view.body_com = self._view(model.body_centers_of_mass, wp.vec3, model.body_count)
        view.inertia = self._view(model.body_inertia_tensors, wp.float32, 9 * model.body_count)
        n = model.cable_count
        view.body_pairs = self._view(model.cable_body_indices, wp.vec2i, n)
        view.frame0 = self._view(model.cable_parent_transforms, wp.transform, n)
        view.frame1 = self._view(model.cable_child_transforms, wp.transform, n)
        view.base = self._view(model.cable_stiffnesses, wp.vec4, n)
        view.penalty = self.cable_penalty if self._cable_ramping() else self._empty(wp.vec4)
        view.damping = self._view(model.cable_dampings, wp.vec4, n)
        view.rest = self._view(model.cable_rest_bend_twist, wp.vec3, n)
        view.state = self._view(model.cable_angular_bias, wp.vec3)
        view.bias = self._view(model.cable_angular_hessian, wp.vec3)
        view.active = self._view(model.cable_enabled, wp.int32, n if model.cable_enabled.size else None)
        view.linear = self.cable_linear
        view.angular = self.cable_angular
        view.indices = self.cable_indices
        view.count = self.cable_sizes
        view.per_world = max(n, 1)
        view.total_count = self.cable_count
        view.particles_per_world = self.per_world
        return view

    def _prepare_cable_penalties(self, reset: bool, rescale: bool, mask: wp.array | None = None) -> None:
        config, model = self._config, self.model
        self._launch(
            "cable_penalty_prepare",
            kernels.prepare_cable_penalties,
            self.cable_count,
            [
                self._view(model.cable_stiffnesses, wp.vec4, model.cable_count),
                self.cable_penalty,
                self.cable_initialized,
                model.cable_count,
                config.cable_linear_beta,
                config.cable_angular_beta,
                config.cable_penalty_gamma,
                config.cable_linear_k_start,
                config.cable_angular_k_start,
                int(reset),
                int(rescale),
                mask if mask is not None else self._empty(wp.int32),
            ],
        )

    def reset_cable_penalty_state(self, stream: Any = None, mask: Span = Span()) -> None:  # noqa: B008
        self._validate_cable_config()
        cable_count = self.cable_count if self._initialized else 0
        if mask.size and mask.size != cable_count:
            raise ValueError("cable penalty reset mask must match the cable state count")
        if not self._initialized or not self.cable_count:
            return
        with self._stream_scope(stream):
            self._prepare_cable_penalties(True, False, self._view(mask, wp.int32) if mask.size else None)

    def copy_cable_penalties(self, output: Span, stream: Any = None) -> None:
        self._validate_cable_config()
        cable_count = self.cable_count if self._initialized else 0
        if output.size != cable_count:
            raise ValueError("cable penalty output must match the cable state count")
        if not self._initialized or not self.cable_count:
            return
        with self._stream_scope(stream):
            self._prepare_cable_penalties(False, False)
            wp.copy(self._view(output, wp.vec4, self.cable_count), self.cable_penalty)

    def _evaluate_cables(self, x: wp.array, inverse_dt: float, direction=None, force=None, diagonal=None) -> None:
        self._launch(
            "cable_evaluate",
            kernels.evaluate_cables,
            self.cable_count,
            [
                self._cable_view(),
                x,
                self.previous,
                inverse_dt,
                direction if direction is not None else self._empty(wp.vec3),
                force if force is not None else self._empty(wp.vec3),
                diagonal if diagonal is not None else self._empty(wp.mat33),
            ],
        )
