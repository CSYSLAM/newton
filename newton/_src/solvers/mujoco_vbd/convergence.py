# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interface-wrench relaxation, residual, and divergence protection.

See ``DESIGN.md`` section 14. Convergence is diagnostic only; the fixed
``iterations`` count keeps eager and CUDA Graph launch topology identical.
"""

from __future__ import annotations

import warp as wp

from ...sim import Model
from .config import MuJoCoVBDCouplingOptions
from .diagnostics import MuJoCoVBDDiagnostics
from .ownership import MuJoCoVBDOwnership

__all__ = ["MuJoCoVBDConvergence"]


@wp.kernel
def _accumulate_world_aitken_terms_kernel(
    proxy_body_ids: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    residual_current: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
    world_dot_r_dr: wp.array[float],
    world_dot_dr_dr: wp.array[float],
):
    """Accumulate Aitken numerator/denominator per world (``DESIGN.md`` 14)."""
    slot = wp.tid()
    body = proxy_body_ids[slot]
    world = body_world[body]
    r = residual_current[body]
    r_prev = residual_previous[body]
    dr = r - r_prev
    wp.atomic_add(world_dot_r_dr, world, wp.dot(r_prev, dr))
    wp.atomic_add(world_dot_dr_dr, world, wp.dot(dr, dr))


@wp.kernel
def _finalize_world_aitken_omega_kernel(
    relaxation_min: float,
    relaxation_max: float,
    eps: float,
    world_dot_r_dr: wp.array[float],
    world_dot_dr_dr: wp.array[float],
    aitken_omega: wp.array[float],
):
    """omega[k] = clamp(-omega[k-1] * <r_{k-1},dr> / <dr,dr>) (``DESIGN.md`` 14)."""
    world = wp.tid()
    denom = wp.max(world_dot_dr_dr[world], eps)
    omega_prev = aitken_omega[world]
    omega = -omega_prev * world_dot_r_dr[world] / denom
    aitken_omega[world] = wp.clamp(omega, relaxation_min, relaxation_max)


@wp.kernel
def _reduce_residual_norms_kernel(
    proxy_body_ids: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    residual_current: wp.array[wp.spatial_vector],
    wrench_raw: wp.array[wp.spatial_vector],
    wrench_relaxed: wp.array[wp.spatial_vector],
    out_residual_l2: wp.array[float],
    out_wrench_norm: wp.array[float],
):
    """Accumulate squared residual and reference wrench norms per world."""
    slot = wp.tid()
    body = proxy_body_ids[slot]
    world = body_world[body]
    r = residual_current[body]
    wp.atomic_add(out_residual_l2, world, wp.dot(r, r))
    raw = wrench_raw[body]
    rel = wrench_relaxed[body]
    wp.atomic_add(out_wrench_norm, world, wp.max(wp.dot(raw, raw), wp.dot(rel, rel)))


@wp.kernel
def _finalize_convergence_kernel(
    abs_tol: float,
    rel_tol: float,
    residual_l2_sq: wp.array[float],
    wrench_norm_sq: wp.array[float],
    residual_velocity_max: wp.array[float],
    velocity_tol: float,
    out_residual_l2: wp.array[float],
    out_residual_relative: wp.array[float],
    out_converged: wp.array[wp.bool],
):
    """Produce per-world residual norms and converged flags (``DESIGN.md`` 14)."""
    world = wp.tid()
    residual = wp.sqrt(residual_l2_sq[world])
    reference = wp.sqrt(wrench_norm_sq[world])
    out_residual_l2[world] = residual
    denom = wp.max(reference, 1.0e-30)
    out_residual_relative[world] = residual / denom
    force_ok = residual <= abs_tol + rel_tol * reference
    velocity_ok = residual_velocity_max[world] <= velocity_tol
    out_converged[world] = force_ok and velocity_ok


@wp.kernel
def _reduce_velocity_residual_kernel(
    proxy_body_ids: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    source_qd: wp.array[wp.spatial_vector],
    destination_qd: wp.array[wp.spatial_vector],
    out_velocity_max: wp.array[float],
):
    slot = wp.tid()
    body = proxy_body_ids[slot]
    world = body_world[body]
    difference = destination_qd[body] - source_qd[body]
    wp.atomic_max(out_velocity_max, world, wp.sqrt(wp.dot(difference, difference)))


@wp.kernel
def _detect_coupling_failure_kernel(
    iteration: int,
    residual_current: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
    wrench_raw: wp.array[wp.spatial_vector],
    proxy_body_ids: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    divergence_ratio: float,
    nonfinite_flag: wp.array[wp.int32],
    diverged_flag: wp.array[wp.int32],
):
    """Flag NaN/Inf and residual blow-up per world (``DESIGN.md`` 14.1)."""
    slot = wp.tid()
    body = proxy_body_ids[slot]
    world = body_world[body]
    w = wrench_raw[body]
    for i in range(6):
        value = w[i]
        if not (value == value) or wp.abs(value) > 1.0e30:
            wp.atomic_add(nonfinite_flag, world, 1)
    r = residual_current[body]
    r_prev = residual_previous[body]
    if iteration > 0 and wp.dot(r, r) > divergence_ratio * divergence_ratio * wp.dot(r_prev, r_prev):
        wp.atomic_add(diverged_flag, world, 1)


@wp.kernel
def _clamp_diverged_relaxation_kernel(
    relaxation_min: float,
    diverged_flag: wp.array[wp.int32],
    aitken_omega: wp.array[float],
):
    world = wp.tid()
    if diverged_flag[world] != 0:
        aitken_omega[world] = relaxation_min


@wp.kernel
def _reset_masked_convergence_kernel(
    world_mask: wp.array[wp.bool],
    relaxation_initial: float,
    aitken_omega: wp.array[float],
    world_dot_r_dr: wp.array[float],
    world_dot_dr_dr: wp.array[float],
    residual_l2_sq: wp.array[float],
    wrench_norm_sq: wp.array[float],
):
    world = wp.tid()
    if world_mask[world]:
        aitken_omega[world] = relaxation_initial
        world_dot_r_dr[world] = 0.0
        world_dot_dr_dr[world] = 0.0
        residual_l2_sq[world] = 0.0
        wrench_norm_sq[world] = 0.0


class MuJoCoVBDConvergence:
    """Owns relaxation state and residual diagnostics (``DESIGN.md`` section 14)."""

    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDCouplingOptions,
        diagnostics: MuJoCoVBDDiagnostics,
    ) -> None:
        self.model = model
        self.ownership = ownership
        self.options = options
        self.diagnostics = diagnostics
        self.device = model.device
        self.world_count = max(int(model.world_count), 1)

        self.divergence_ratio = 4.0
        self.aitken_eps = 1.0e-12

        self._world_dot_r_dr = wp.zeros(self.world_count, dtype=float, device=self.device)
        self._world_dot_dr_dr = wp.zeros(self.world_count, dtype=float, device=self.device)
        self._residual_l2_sq = wp.zeros(self.world_count, dtype=float, device=self.device)
        self._wrench_norm_sq = wp.zeros(self.world_count, dtype=float, device=self.device)
        self.aitken_omega = wp.zeros(self.world_count, dtype=float, device=self.device)

    def begin_substep(self, warm_start: bool) -> None:
        """Reset iteration-local Aitken state (``DESIGN.md`` 17.1)."""
        self.aitken_omega.fill_(self.options.relaxation_initial)
        self._world_dot_r_dr.zero_()
        self._world_dot_dr_dr.zero_()
        _ = warm_start

    def update_aitken_omega(
        self,
        residual_current: wp.array,
        residual_previous: wp.array,
        iteration: int,
    ) -> None:
        """Refresh the per-world Aitken factor from the current residuals."""
        if iteration == 0 or self.ownership.proxy_body_ids.shape[0] == 0:
            return
        self._world_dot_r_dr.zero_()
        self._world_dot_dr_dr.zero_()
        wp.launch(
            _accumulate_world_aitken_terms_kernel,
            dim=self.ownership.proxy_body_ids.shape[0],
            inputs=[
                self.ownership.proxy_body_ids,
                self.model.body_world,
                residual_current,
                residual_previous,
                self._world_dot_r_dr,
                self._world_dot_dr_dr,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_world_aitken_omega_kernel,
            dim=self.world_count,
            inputs=[
                self.options.relaxation_min,
                self.options.relaxation_max,
                self.aitken_eps,
                self._world_dot_r_dr,
                self._world_dot_dr_dr,
                self.aitken_omega,
            ],
            device=self.device,
        )

    def update_velocity_residual(
        self,
        source_qd: wp.array,
        destination_qd: wp.array,
    ) -> None:
        """Measure the maximum proxy velocity mismatch for the current round."""
        self.diagnostics.residual_velocity_max.zero_()
        if self.ownership.proxy_body_ids.shape[0] == 0:
            return
        wp.launch(
            _reduce_velocity_residual_kernel,
            dim=self.ownership.proxy_body_ids.shape[0],
            inputs=[
                self.ownership.proxy_body_ids,
                self.model.body_world,
                source_qd,
                destination_qd,
                self.diagnostics.residual_velocity_max,
            ],
            device=self.device,
        )

    def finalize(
        self,
        residual_current: wp.array,
        wrench_raw: wp.array,
        wrench_relaxed: wp.array,
    ) -> None:
        """Compute final residual norms and converged flags (``DESIGN.md`` 14)."""
        if self.ownership.proxy_body_ids.shape[0] == 0:
            self.diagnostics.converged.fill_(True)
            return
        self._residual_l2_sq.zero_()
        self._wrench_norm_sq.zero_()
        wp.launch(
            _reduce_residual_norms_kernel,
            dim=self.ownership.proxy_body_ids.shape[0],
            inputs=[
                self.ownership.proxy_body_ids,
                self.model.body_world,
                residual_current,
                wrench_raw,
                wrench_relaxed,
                self._residual_l2_sq,
                self._wrench_norm_sq,
            ],
            device=self.device,
        )
        wp.launch(
            _finalize_convergence_kernel,
            dim=self.world_count,
            inputs=[
                self.options.force_absolute_tolerance,
                self.options.force_relative_tolerance,
                self._residual_l2_sq,
                self._wrench_norm_sq,
                self.diagnostics.residual_velocity_max,
                self.options.velocity_tolerance,
                self.diagnostics.residual_force_l2,
                self.diagnostics.residual_force_relative,
                self.diagnostics.converged,
            ],
            device=self.device,
        )

    def detect_failure(
        self,
        residual_current: wp.array,
        residual_previous: wp.array,
        iteration: int,
    ) -> None:
        """Populate nonfinite/diverged flags (``DESIGN.md`` 14.1)."""
        if self.ownership.proxy_body_ids.shape[0] == 0:
            return
        wp.launch(
            _detect_coupling_failure_kernel,
            dim=self.ownership.proxy_body_ids.shape[0],
            inputs=[
                iteration,
                residual_current,
                residual_previous,
                self.diagnostics.feedback_wrench_raw,
                self.ownership.proxy_body_ids,
                self.model.body_world,
                self.divergence_ratio,
                self.diagnostics.nonfinite_flag,
                self.diagnostics.diverged_flag,
            ],
            device=self.device,
        )
        wp.launch(
            _clamp_diverged_relaxation_kernel,
            dim=self.world_count,
            inputs=[
                self.options.relaxation_min,
                self.diagnostics.diverged_flag,
                self.aitken_omega,
            ],
            device=self.device,
        )

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Clear relaxation and residual state (``DESIGN.md`` 18.1)."""
        if world_mask is None:
            self.aitken_omega.fill_(self.options.relaxation_initial)
            self._world_dot_r_dr.zero_()
            self._world_dot_dr_dr.zero_()
            self._residual_l2_sq.zero_()
            self._wrench_norm_sq.zero_()
            return
        wp.launch(
            _reset_masked_convergence_kernel,
            dim=self.world_count,
            inputs=[
                world_mask,
                self.options.relaxation_initial,
                self.aitken_omega,
                self._world_dot_r_dr,
                self._world_dot_dr_dr,
                self._residual_l2_sq,
                self._wrench_norm_sq,
            ],
            device=self.device,
        )
