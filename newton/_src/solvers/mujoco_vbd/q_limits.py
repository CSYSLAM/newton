# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative scalar joint-limit acceptance for q-block trials."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import JointType, State
from .articulation_predictor import MuJoCoSmoothPredictor


@wp.kernel(enable_backward=False)
def _limit_q_block_step(
    q_hat: wp.array[float],
    delta: wp.array2d[float],
    mujoco_dof_to_qcoord: wp.array2d[int],
    lower: wp.array[float],
    upper: wp.array[float],
    alpha: wp.array[float],
):
    world, dof = wp.tid()
    qcoord = mujoco_dof_to_qcoord[world, dof]
    if qcoord < 0:
        return
    value = q_hat[qcoord]
    direction = delta[world, dof]
    candidate_alpha = 1.0
    if direction > 0.0 and value + direction > upper[qcoord]:
        candidate_alpha = (upper[qcoord] - value) / direction
    elif direction < 0.0 and value + direction < lower[qcoord]:
        candidate_alpha = (lower[qcoord] - value) / direction
    alpha[world] = wp.min(alpha[world], wp.clamp(candidate_alpha, 0.0, 1.0))


class ArticulationLimitProjector:
    """Compute safe q-block alpha values for hinge and slide limits.

    Ball and free joints intentionally have no component-wise limit handling:
    their limits require manifold-specific semantics and are rejected by the
    first solver version when authored.  The result is combined with DAT and
    trust-region alpha before a unified trial is committed.

    Args:
        predictor: Compact articulated MuJoCo predictor.
    """

    def __init__(self, predictor: MuJoCoSmoothPredictor) -> None:
        self.predictor = predictor
        solver = predictor._solver
        if solver.mjc_dof_to_newton_dof is None:
            raise RuntimeError("q-limit projector requires MuJoCo-to-Newton DOF mappings")
        model = predictor.model
        local_qd_to_qcoord = np.full(model.joint_dof_count, -1, dtype=np.int32)
        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        for joint, joint_kind in enumerate(joint_type):
            if joint_kind not in (int(JointType.REVOLUTE), int(JointType.PRISMATIC)):
                continue
            qd_index = int(qd_start[joint])
            local_qd_to_qcoord[qd_index] = int(q_start[joint])

        mjc_dof_to_newton = solver.mjc_dof_to_newton_dof.numpy()
        mjc_dof_to_qcoord = np.full_like(mjc_dof_to_newton, -1)
        for world in range(mjc_dof_to_newton.shape[0]):
            for dof in range(mjc_dof_to_newton.shape[1]):
                qd_index = int(mjc_dof_to_newton[world, dof])
                if 0 <= qd_index < local_qd_to_qcoord.size:
                    mjc_dof_to_qcoord[world, dof] = local_qd_to_qcoord[qd_index]

        device = model.device
        self._mujoco_dof_to_qcoord = wp.array(mjc_dof_to_qcoord, dtype=int, device=device)
        self.alpha = wp.ones(solver.mjw_data.nworld, dtype=float, device=device)

    def compute_alpha(self, state_hat: State, delta: wp.array2d[float]) -> wp.array[float]:
        """Return the largest scalar-limit-feasible fraction of ``delta``.

        Args:
            state_hat: Smooth q endpoint used as the correction center.
            delta: Unscaled q-block correction shaped ``[world, nv]``.

        Returns:
            Per-world accepted fractions in ``[0, 1]``.
        """
        if state_hat.joint_q is None:
            raise ValueError("q-limit projector requires joint coordinates")
        data = self.predictor._solver.mjw_data
        model = self.predictor._solver.mjw_model
        assert data is not None
        assert model is not None
        if delta.shape != (data.nworld, model.nv):
            raise ValueError(f"q correction must have shape {(data.nworld, model.nv)}, got {delta.shape}")
        self.alpha.fill_(1.0)
        wp.launch(
            _limit_q_block_step,
            dim=(data.nworld, model.nv),
            inputs=[
                state_hat.joint_q,
                delta,
                self._mujoco_dof_to_qcoord,
                self.predictor.model.joint_limit_lower,
                self.predictor.model.joint_limit_upper,
            ],
            outputs=[self.alpha],
            device=self.predictor.model.device,
        )
        return self.alpha
