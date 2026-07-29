# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dense reduced-coordinate articulation block for the VBD corrector."""

from __future__ import annotations

import warp as wp

from ..featherstone.kernels import dense_subs, eval_dense_cholesky_batched
from .articulation_predictor import ArticulationPredictorResult, MuJoCoSmoothPredictor


@wp.func
def _packed_mass_entry(
    row: int,
    column: int,
    dof_parent: wp.array[int],
    mass_rownnz: wp.array[int],
    mass_rowadr: wp.array[int],
    mass: wp.array2d[float],
    world: int,
) -> float:
    """Read one symmetric entry from MuJoCo-Warp's tree-packed mass matrix."""
    stored_row = wp.max(row, column)
    target_column = wp.min(row, column)
    packed_index = mass_rowadr[stored_row] + mass_rownnz[stored_row] - 1
    current_column = stored_row
    while current_column >= 0:
        if current_column == target_column:
            return mass[world, packed_index]
        current_column = dof_parent[current_column]
        packed_index -= 1
    return 0.0


@wp.kernel(enable_backward=False)
def _initialize_hessian_from_mass(
    dof_parent: wp.array[int],
    mass_rownnz: wp.array[int],
    mass_rowadr: wp.array[int],
    mass: wp.array2d[float],
    dt: float,
    hessian: wp.array3d[float],
    gradient: wp.array2d[float],
):
    world, row, column = wp.tid()
    hessian[world, row, column] = _packed_mass_entry(row, column, dof_parent, mass_rownnz, mass_rowadr, mass, world) / (
        dt * dt
    )
    if column == 0:
        gradient[world, row] = 0.0


@wp.kernel(enable_backward=False)
def _accumulate_contact_terms(
    active_count: wp.array[int],
    contact_active: wp.array[int],
    contact_world: wp.array[int],
    contact_jacobian: wp.array3d[float],
    contact_gradient: wp.array[wp.vec3],
    contact_hessian: wp.array[wp.mat33],
    hessian: wp.array3d[float],
    gradient: wp.array2d[float],
):
    contact = wp.tid()
    if contact >= active_count[0] or contact_active[contact] == 0:
        return
    world = contact_world[contact]
    dof_count = contact_jacobian.shape[2]
    contact_f = contact_gradient[contact]
    contact_K = contact_hessian[contact]

    for row in range(dof_count):
        jacobian_row = wp.vec3(
            contact_jacobian[contact, 0, row],
            contact_jacobian[contact, 1, row],
            contact_jacobian[contact, 2, row],
        )
        wp.atomic_add(gradient, world, row, wp.dot(jacobian_row, contact_f))
        contact_K_jacobian_row = contact_K * jacobian_row
        for column in range(dof_count):
            jacobian_column = wp.vec3(
                contact_jacobian[contact, 0, column],
                contact_jacobian[contact, 1, column],
                contact_jacobian[contact, 2, column],
            )
            wp.atomic_add(hessian, world, row, column, wp.dot(jacobian_column, contact_K_jacobian_row))


@wp.kernel(enable_backward=False)
def _solve_negative_gradient(
    dof_count: int,
    factor: wp.array[float],
    gradient: wp.array[float],
    delta: wp.array[float],
):
    world = wp.tid()
    gradient_start = world * dof_count
    factor_start = world * dof_count * dof_count
    for dof in range(dof_count):
        delta[gradient_start + dof] = -gradient[gradient_start + dof]
    dense_subs(dof_count, factor_start, gradient_start, factor, delta, delta)


class ArticulationQBlock:
    """Assemble and solve one complete reduced-coordinate articulation block.

    The block's endpoint objective is

    ``0.5 / h^2 * ||delta_q||_M^2 + E_contact(delta_q)``.

    Contact callers provide a point Jacobian and the contact evaluator's
    gradient/Hessian with respect to point displacement.  This class performs
    the exact q-space pullback ``J.T f`` and ``J.T K J`` before one dense
    Cholesky solve per MuJoCo world.  It is deliberately bounded to a small
    articulation block; larger systems must use the sparse q-block path rather
    than silently falling back to per-link 6D solves.

    Args:
        predictor: Smooth predictor supplying the current packed mass matrix.
        max_dofs: Maximum supported tangent DOFs per world.
        trust_region: Positive diagonal regularization in q-space.
    """

    def __init__(
        self,
        predictor: MuJoCoSmoothPredictor,
        *,
        max_dofs: int = 64,
        trust_region: float = 1.0e-8,
    ) -> None:
        if max_dofs < 1:
            raise ValueError("max_dofs must be positive")
        if trust_region <= 0.0:
            raise ValueError("trust_region must be positive")
        self.predictor = predictor
        data = predictor._solver.mjw_data
        model = predictor._solver.mjw_model
        assert data is not None
        assert model is not None
        if model.nv > max_dofs:
            raise ValueError(
                f"MuJoCo-VBD dense q-block supports at most {max_dofs} DOFs per world, got {model.nv}; "
                "use the sparse q-block implementation."
            )

        self._data = data
        self._model = model
        self._dof_count = model.nv
        self._world_count = data.nworld
        self._trust_region = trust_region
        shape = (self._world_count, self._dof_count, self._dof_count)
        self.hessian = wp.empty(shape, dtype=float, device=predictor.model.device)
        self.gradient = wp.empty((self._world_count, self._dof_count), dtype=float, device=predictor.model.device)
        self._factor = wp.empty(shape, dtype=float, device=predictor.model.device)
        self._regularization = wp.full(
            self._world_count * self._dof_count,
            trust_region,
            dtype=float,
            device=predictor.model.device,
        )
        self._starts = wp.array(
            [world * self._dof_count * self._dof_count for world in range(self._world_count)],
            dtype=int,
            device=predictor.model.device,
        )
        self._dimensions = wp.full(self._world_count, self._dof_count, dtype=int, device=predictor.model.device)
        self._gradient_starts = wp.array(
            [world * self._dof_count for world in range(self._world_count)],
            dtype=int,
            device=predictor.model.device,
        )
        self.delta = wp.empty((self._world_count, self._dof_count), dtype=float, device=predictor.model.device)
        self._all_contact_active_masks: dict[int, wp.array[int]] = {}

    @property
    def dof_count(self) -> int:
        """Number of tangent degrees of freedom in each world block."""
        return self._dof_count

    def initialize(self, result: ArticulationPredictorResult, dt: float) -> None:
        """Reset the block to ``M(q_n) / h^2`` and zero its gradient.

        Args:
            result: Result from the immediately preceding smooth prediction.
            dt: Timestep [s] used for that prediction.
        """
        if dt <= 0.0:
            raise ValueError("q-block timestep dt must be positive")
        if result.mass is not self._data.M:
            raise ValueError("q-block result must come from this predictor")
        wp.launch(
            _initialize_hessian_from_mass,
            dim=(self._world_count, self._dof_count, self._dof_count),
            inputs=[
                self._model.dof_parentid,
                self._model.M_rownnz,
                self._model.M_rowadr,
                result.mass,
                dt,
            ],
            outputs=[self.hessian, self.gradient],
            device=self.predictor.model.device,
        )

    def accumulate_contact_terms(
        self,
        active_count: wp.array[int],
        contact_world: wp.array[int],
        contact_jacobian: wp.array3d[float],
        contact_gradient: wp.array[wp.vec3],
        contact_hessian: wp.array[wp.mat33],
        contact_active: wp.array[int] | None = None,
    ) -> None:
        """Pull contact-space Gauss-Newton terms into the articulation block.

        Args:
            active_count: Number of active compacted contact records.
            contact_world: MuJoCo world index for every contact record.
            contact_jacobian: Point Jacobian ``[contact, xyz, dof]``.
            contact_gradient: Contact-space energy gradients [N].
            contact_hessian: Contact-space energy Hessians [N/m].
            contact_active: Optional per-record q ownership mask.  Entries
                set to zero are skipped even when they belong to the canonical
                global contact buffer's active prefix.
        """
        count = contact_world.shape[0]
        if active_count.shape != (1,):
            raise ValueError("active_count must have shape (1,)")
        if contact_jacobian.shape != (count, 3, self._dof_count):
            raise ValueError(
                f"contact_jacobian must have shape ({count}, 3, {self._dof_count}), got {contact_jacobian.shape}"
            )
        if contact_gradient.shape != (count,) or contact_hessian.shape != (count,):
            raise ValueError("contact gradient and Hessian arrays must contain one entry per contact")
        if contact_active is None:
            contact_active = self._all_contact_active_masks.get(count)
            if contact_active is None:
                contact_active = wp.ones(count, dtype=int, device=self.predictor.model.device)
                self._all_contact_active_masks[count] = contact_active
        elif contact_active.shape != (count,):
            raise ValueError("contact_active must contain one entry per contact")
        if count:
            wp.launch(
                _accumulate_contact_terms,
                dim=count,
                inputs=[
                    active_count,
                    contact_active,
                    contact_world,
                    contact_jacobian,
                    contact_gradient,
                    contact_hessian,
                ],
                outputs=[self.hessian, self.gradient],
                device=self.predictor.model.device,
            )

    def solve(self) -> wp.array:
        """Solve the current q-block and return the tangent correction."""
        hessian_flat = self.hessian.flatten()
        factor_flat = self._factor.flatten()
        gradient_flat = self.gradient.flatten()
        delta_flat = self.delta.flatten()
        wp.launch(
            eval_dense_cholesky_batched,
            dim=self._world_count,
            inputs=[
                self._starts,
                self._dimensions,
                self._gradient_starts,
                hessian_flat,
                self._regularization,
            ],
            outputs=[factor_flat],
            device=self.predictor.model.device,
        )
        wp.launch(
            _solve_negative_gradient,
            dim=self._world_count,
            inputs=[self._dof_count, factor_flat, gradient_flat],
            outputs=[delta_flat],
            device=self.predictor.model.device,
        )
        return self.delta
