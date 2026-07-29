# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One reduced-coordinate block update inside the unified VBD sweep."""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Control
from ..coupled.solver_coupled import _copy_control_to_entry
from .articulation_predictor import ArticulationPredictorResult, MuJoCoSmoothPredictor
from .articulation_q_block import ArticulationQBlock
from .canonical_contacts import CanonicalRigidContacts
from .canonical_soft_contacts import CanonicalSoftContacts
from .contact_jacobian import ArticulationContactJacobian
from .ownership_partition import MuJoCoVBDOwnershipPartition
from .q_limits import ArticulationLimitProjector
from .soft_contact_jacobian import ArticulationSoftContactJacobian
from .trial_workspace import MuJoCoVBDTrialWorkspace


@wp.kernel(enable_backward=False)
def _scale_q_correction(delta: wp.array2d[float], alpha: wp.array[float], scaled: wp.array2d[float]):
    world, dof = wp.tid()
    scaled[world, dof] = alpha[world] * delta[world, dof]


class ArticulationCorrector:
    """Apply a smooth MuJoCo predictor and contact q-block correction.

    The class is intentionally one block in the larger symmetric VBD sweep. It
    does not advance free bodies or particles, does not regenerate contacts,
    and does not commit multipliers.  Those responsibilities belong to the
    unified scratch/DAT layer so every participant observes one trial state.

    Args:
        partition: Static ownership partition.
        max_dofs: Dense q-block tangent-DOF limit.
    """

    def __init__(self, partition: MuJoCoVBDOwnershipPartition, *, max_dofs: int = 64) -> None:
        self.partition = partition
        self.entry = partition.articulation_entry
        self.predictor = MuJoCoSmoothPredictor(self.entry.view, solver=self.entry.solver)
        self.q_block = ArticulationQBlock(self.predictor, max_dofs=max_dofs)
        self.contact_jacobian = ArticulationContactJacobian(self.predictor, self.entry)
        self.soft_contact_jacobian = ArticulationSoftContactJacobian(self.predictor, self.entry)
        self.limit_projector = ArticulationLimitProjector(self.predictor)
        self._scaled_delta = wp.empty_like(self.q_block.delta)

    def predict(
        self,
        workspace: MuJoCoVBDTrialWorkspace,
        control: Control | None,
        dt: float,
    ) -> ArticulationPredictorResult:
        """Predict ``q_hat`` and expose its FK as the VBD contact boundary.

        Args:
            workspace: Accepted/trial state owner for the current sweep.
            control: Global caller control, or None.
            dt: Timestep [s].

        Returns:
            Smooth MuJoCo prediction whose metric is reused by
            :meth:`correct_from_prediction`.
        """
        local_control = _copy_control_to_entry(control, self.entry)
        result = self.predictor.predict(self.entry.state_0, local_control, dt)
        workspace.scatter_articulation(result.state_hat)
        return result

    def correct_from_prediction(
        self,
        workspace: MuJoCoVBDTrialWorkspace,
        result: ArticulationPredictorResult,
        contacts: Contacts,
        canonical_contacts: CanonicalRigidContacts,
        canonical_soft_contacts: CanonicalSoftContacts,
        dt: float,
    ) -> None:
        """Write one q-corrected trial state into ``workspace``.

        ``workspace.trial`` must already hold the smooth predicted articulation
        state plus the current free-rigid/particle trial state before canonical
        contact energy is evaluated.  ``contacts`` must be generated for that
        same trial geometry; stale pre-predict contacts are rejected by the
        outer collision/DAT layer rather than silently reused here.

        Args:
            workspace: Accepted/trial state owner for the current sweep.
            result: Smooth prediction returned by :meth:`predict` for this
                same trial timestep.
            contacts: Canonical rigid contacts for the smooth q trial.
            canonical_contacts: Contact gradient/Hessian batch already
                evaluated at ``workspace.trial``.
            canonical_soft_contacts: Particle/rigid contact terms evaluated
                against the same global trial state.
            dt: Timestep [s].
        """
        self.q_block.initialize(result, dt)
        if contacts.rigid_contact_max:
            canonical_contacts.evaluate(workspace.trial, workspace.accepted, contacts, dt)
            jacobian = self.contact_jacobian.evaluate(result.state_hat, contacts)
            self.q_block.accumulate_contact_terms(
                contacts.rigid_contact_count,
                self.contact_jacobian.contact_world,
                jacobian,
                canonical_contacts.gradient,
                canonical_contacts.hessian,
                self.contact_jacobian.active,
            )
        if contacts.soft_contact_max:
            canonical_soft_contacts.evaluate(workspace.trial, workspace.accepted, contacts, dt)
            soft_jacobian = self.soft_contact_jacobian.evaluate(result.state_hat, contacts)
            self.q_block.accumulate_contact_terms(
                contacts.soft_contact_count,
                self.soft_contact_jacobian.contact_world,
                soft_jacobian,
                canonical_soft_contacts.gradient,
                canonical_soft_contacts.hessian,
                self.soft_contact_jacobian.active,
            )
        delta = self.q_block.solve()
        alpha = self.limit_projector.compute_alpha(result.state_hat, delta)
        wp.launch(
            _scale_q_correction,
            dim=delta.shape,
            inputs=[delta, alpha],
            outputs=[self._scaled_delta],
            device=self.predictor.model.device,
        )
        workspace.scatter_articulation(self.predictor.apply_tangent_delta(self._scaled_delta, dt))

    def correct(
        self,
        workspace: MuJoCoVBDTrialWorkspace,
        control: Control | None,
        contacts: Contacts,
        canonical_contacts: CanonicalRigidContacts,
        canonical_soft_contacts: CanonicalSoftContacts,
        dt: float,
    ) -> None:
        """Predict then correct, retained as a convenience operation.

        New joint VBD scheduling should call :meth:`predict` before its VBD
        block and :meth:`correct_from_prediction` after VBD has updated the
        non-articulated endpoints.
        """
        result = self.predict(workspace, control, dt)
        self.correct_from_prediction(
            workspace,
            result,
            contacts,
            canonical_contacts,
            canonical_soft_contacts,
            dt,
        )
