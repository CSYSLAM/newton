# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-step iteration of the MuJoCo/MPM proxy interface."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _record_interface_residual(
    proxy_ids_global: wp.array[int],
    proxy_ids_local: wp.array[int],
    harvested: wp.array[wp.spatial_vector],
    applied: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    dt: float,
    iteration: int,
    residuals: wp.array2d[float],
):
    """Record the largest proxy velocity mismatch caused by the feedback residual.

    In the lagged scheme the MPM proxy ends with the MuJoCo end velocity plus
    ``(harvested - applied) * dt`` mapped through the proxy inertia, so this
    is the interface velocity residual between the two domains [m/s, rad/s].
    """
    i = wp.tid()
    body = proxy_ids_local[i]
    residual = harvested[proxy_ids_global[i]] - applied[i]
    linear = body_inv_mass[body] * dt * wp.spatial_top(residual)
    rotation = wp.transform_get_rotation(body_q[body])
    angular = wp.quat_rotate(
        rotation, body_inv_inertia[body] * wp.quat_rotate_inv(rotation, dt * wp.spatial_bottom(residual))
    )
    wp.atomic_max(residuals, iteration, 0, wp.length(linear))
    wp.atomic_max(residuals, iteration, 1, wp.length(angular))


@wp.kernel(enable_backward=False)
def _record_articulated_residual(
    link_body: wp.array2d[int],
    mobility: wp.array3d[float],
    body_to_proxy: wp.array[int],
    proxy_ids_global: wp.array[int],
    harvested: wp.array[wp.spatial_vector],
    applied: wp.array[wp.spatial_vector],
    dt: float,
    iteration: int,
    residuals: wp.array2d[float],
):
    """Record the link velocity mismatch implied through the articulated link mobility."""
    world, link = wp.tid()
    if link_body[world, link] < 0:
        return
    linear = wp.vec3(0.0)
    angular = wp.vec3(0.0)
    for other in range(link_body.shape[1]):
        body = link_body[world, other]
        if body < 0:
            continue
        i = body_to_proxy[body]
        if i < 0:
            continue
        residual = dt * (harvested[proxy_ids_global[i]] - applied[i])
        for r in range(3):
            for c in range(6):
                linear[r] += mobility[world, 6 * link + r, 6 * other + c] * residual[c]
                angular[r] += mobility[world, 6 * link + 3 + r, 6 * other + c] * residual[c]
    wp.atomic_max(residuals, iteration, 0, wp.length(linear))
    wp.atomic_max(residuals, iteration, 1, wp.length(angular))


class SameStepIteration:
    """Repeat the proxy pass from the same step-start state until the interface agrees.

    Each pass advances MuJoCo with the current feedback guess, then re-solves
    the MPM environment from the restored step-start state with the predicted
    link motion and harvests new contact feedback. At a fixed point the MPM
    proxy end velocity equals the MuJoCo link end velocity, so the proxy
    inertia only affects convergence.

    After several passes, MuJoCo is re-solved from the step start with the
    last pass's unrelaxed feedback while the MPM side keeps its last pass.
    Both domains then apply the same contact impulse once, which conserves
    the momentum exchanged across the interface; the remaining kinematic
    mismatch is the recorded residual. The harvested feedback warm-starts the
    next step.

    Args:
        device: Device of the residual buffer.
        iterations: Maximum passes per step.
        tolerance: Interface velocity residual [m/s, rad/s] below which
            uncaptured steps stop early. Zero runs every pass.
        solvers: Sub-solvers whose cross-step history must be restored before
            each repeated pass. Each provides ``coupling_save_iteration_state``
            and ``coupling_restore_iteration_state``.
        resolve_sources: Re-solve MuJoCo with the final harvested feedback.
            Defaults to ``iterations > 1``. Single-pass articulated coupling
            enables it, because its MPM pass already predicts the link
            response to the feedback it harvests.
    """

    def __init__(
        self,
        device,
        *,
        iterations: int,
        tolerance: float,
        solvers: Sequence[object],
        resolve_sources: bool | None = None,
    ):
        self.device = device
        self.iterations = iterations
        self.tolerance = tolerance
        self.solvers = tuple(solvers)
        self._resolve_sources = iterations > 1 if resolve_sources is None else resolve_sources
        self.residuals = wp.zeros((iterations, 2), dtype=float, device=device)
        """Per-pass maximum linear [m/s] and angular [rad/s] interface residual of the latest step."""
        self.passes = 0
        """Passes run in the latest step (all passes while capturing)."""
        self._harvested = {}
        self._body_to_proxy = {}

    def begin_step(self, iterations: int) -> None:
        self.residuals.zero_()
        self.passes = 0
        if iterations > 1 or self._resolve_sources:
            for solver in self.solvers:
                solver.coupling_save_iteration_state()

    def restart(self) -> None:
        for solver in self.solvers:
            solver.coupling_restore_iteration_state()

    def record_body_feedback(self, proxy, destination, iteration: int, dt: float) -> None:
        if proxy.coupling_forces_previous is None or proxy.proxy_ids_global.shape[0] == 0:
            return
        if self._resolve_sources:
            # Keep the unrelaxed harvest for the final source solve.
            harvested = self._harvested.get(id(proxy))
            if harvested is None:
                harvested = wp.empty_like(proxy.coupling_forces)
                self._harvested[id(proxy)] = harvested
            wp.copy(harvested, proxy.coupling_forces)
        view = destination.view
        environment = destination.solver
        if getattr(environment, "_mobility", None) is not None:
            body_to_proxy = self._body_to_proxy.get(id(proxy))
            if body_to_proxy is None:
                mapping = np.full(view.body_count, -1, dtype=np.int32)
                mapping[proxy.proxy_ids_local.numpy()] = np.arange(proxy.proxy_ids_local.shape[0], dtype=np.int32)
                body_to_proxy = wp.array(mapping, dtype=int, device=self.device)
                self._body_to_proxy[id(proxy)] = body_to_proxy
            wp.launch(
                _record_articulated_residual,
                dim=environment._link_body.shape,
                inputs=[
                    environment._link_body,
                    environment._mobility,
                    body_to_proxy,
                    proxy.proxy_ids_global,
                    proxy.coupling_forces,
                    proxy.coupling_forces_previous,
                    dt,
                    iteration,
                ],
                outputs=[self.residuals],
                device=self.device,
            )
            return
        wp.launch(
            _record_interface_residual,
            dim=proxy.proxy_ids_global.shape[0],
            inputs=[
                proxy.proxy_ids_global,
                proxy.proxy_ids_local,
                proxy.coupling_forces,
                proxy.coupling_forces_previous,
                destination.state_0.body_q,
                view.body_inv_mass,
                view.body_inv_inertia,
                dt,
                iteration,
            ],
            outputs=[self.residuals],
            device=self.device,
        )

    def finished(self, iteration: int) -> bool:
        self.passes = iteration + 1
        if self.tolerance <= 0.0 or self.passes >= self.iterations or self.device.is_capturing:
            return False
        return float(self.residuals.numpy()[iteration].max()) <= self.tolerance

    def resolve_sources(self) -> bool:
        """Whether sources must be re-solved with the final harvested feedback."""
        return self._resolve_sources

    def load_harvested_feedback(self, proxy) -> None:
        harvested = self._harvested.get(id(proxy))
        if harvested is not None:
            wp.copy(proxy.coupling_forces, harvested)

    def restore_source(self, solver) -> None:
        solver.coupling_restore_iteration_state()
