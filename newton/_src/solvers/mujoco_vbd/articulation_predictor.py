# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smooth reduced-coordinate prediction for the MuJoCo--VBD solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import warp as wp

from ...sim import Control, Model, State
from ..mujoco import SolverMuJoCo


@wp.kernel(enable_backward=False)
def _scale_tangent_delta(delta: wp.array2d[float], inverse_dt: float, out: wp.array2d[float]):
    world, dof = wp.tid()
    out[world, dof] = delta[world, dof] * inverse_dt


@wp.kernel(enable_backward=False)
def _add_tangent_velocity(qvel: wp.array2d[float], correction_velocity: wp.array2d[float]):
    world, dof = wp.tid()
    qvel[world, dof] += correction_velocity[world, dof]


@dataclass(frozen=True)
class ArticulationPredictorResult:
    """Scratch result of one smooth articulation prediction.

    ``state_hat`` is expressed in Newton's joint-coordinate layout. ``mass``
    and ``mass_factor`` use MuJoCo-Warp's packed reduced-coordinate storage;
    callers must use :meth:`MuJoCoSmoothPredictor.solve_mass` rather than
    treating either array as a dense matrix.
    """

    state_hat: State
    mass: wp.array
    mass_factor: wp.array
    mass_factor_diagonal_inverse: wp.array
    smooth_acceleration: wp.array


class MuJoCoSmoothPredictor:
    """Evaluate MuJoCo smooth dynamics without contacts or EFC constraints.

    The predictor deliberately implements only MuJoCo-Warp's smooth pipeline:
    kinematics, CRB mass matrix, passive/bias/actuator forces, and a
    semi-implicit Euler endpoint.  Joint limits, equality constraints, and
    all contacts belong to the later unified corrector, so this class never
    calls MuJoCo-Warp ``forward()``, ``solver.solve()``, or ``step()``.

    Args:
        model: Articulated model used by the predictor.
        mujoco_options: Supported :class:`SolverMuJoCo` construction options.
            ``integrator`` must be ``"euler"`` because the endpoint corrector
            uses the predictor's semi-implicit Euler inertial center.
        solver: Preconfigured contact-free MuJoCo-Warp solver for ``model``.
            This lets the joint solver retain one compact articulation model
            and its backend buffers for its full lifetime.
    """

    def __init__(
        self,
        model: Model,
        *,
        mujoco_options: Mapping[str, object] | None = None,
        solver: SolverMuJoCo | None = None,
    ) -> None:
        self.model = model
        if solver is not None and mujoco_options is not None:
            raise ValueError("pass either mujoco_options or a preconfigured MuJoCo solver, not both")
        options = dict(mujoco_options or {})
        integrator = options.pop("integrator", "euler")
        if integrator != "euler":
            raise ValueError("MuJoCoSmoothPredictor currently requires integrator='euler'")
        if options.pop("use_mujoco_cpu", False):
            raise ValueError("MuJoCoSmoothPredictor requires the MuJoCo-Warp backend")
        if options.pop("use_mujoco_contacts", False):
            raise ValueError("MuJoCoSmoothPredictor never evaluates MuJoCo contacts")
        if options.pop("disable_contacts", True) is not True:
            raise ValueError("MuJoCoSmoothPredictor requires disable_contacts=True")

        if solver is None:
            solver = SolverMuJoCo(
                model,
                integrator="euler",
                use_mujoco_contacts=False,
                disable_contacts=True,
                update_data_interval=1,
                **options,
            )
        elif solver.model is not model:
            raise ValueError("the preconfigured MuJoCo solver must use the predictor model view")
        elif solver.use_mujoco_cpu or solver._use_mujoco_contacts:
            raise ValueError("the preconfigured MuJoCo solver must use the contact-free MuJoCo-Warp backend")

        self._solver = solver
        if self._solver.use_mujoco_cpu or self._solver.mjw_model is None or self._solver.mjw_data is None:
            raise RuntimeError("MuJoCoSmoothPredictor could not initialize a MuJoCo-Warp state")

        self.state_hat = model.state(requires_grad=False)
        self.state_trial = model.state(requires_grad=False)
        data = self._solver.mjw_data
        assert data is not None
        self._correction_velocity = wp.empty_like(data.qvel)

    @property
    def qd_count(self) -> int:
        """Number of tangent generalized velocities per all simulation worlds."""
        return self.model.joint_dof_count

    def predict(self, state: State, control: Control | None, dt: float) -> ArticulationPredictorResult:
        """Predict the smooth endpoint and cache its reduced-coordinate metric.

        Args:
            state: Current Newton state.
            control: Applied joint/actuator controls, or None for no control.
            dt: Timestep [s].

        Returns:
            Mutable scratch result valid until the next call to :meth:`predict`.
        """
        if dt <= 0.0:
            raise ValueError("MuJoCo smooth predictor timestep dt must be positive")

        # These imports remain local so importing Newton without MuJoCo-Warp
        # keeps the existing optional-dependency behavior.
        from mujoco_warp._src import forward as mjw_forward
        from mujoco_warp._src import smooth as mjw_smooth

        solver = self._solver
        mjw_model = solver.mjw_model
        mjw_data = solver.mjw_data
        assert mjw_model is not None
        assert mjw_data is not None

        with wp.ScopedDevice(self.model.device), solver._scoped_mujoco_warp_execution():
            solver._apply_mjc_control(self.model, state, control, mjw_data)
            solver._update_mjc_data(mjw_data, self.model, state)
            mjw_model.opt.timestep.fill_(dt)

            # This is the complete smooth-only subset of MuJoCo-Warp's
            # forward pipeline.  Do not replace it with fwd_position(): that
            # function creates EFC rows, which the unified corrector owns.
            mjw_forward.fwd_kinematics(mjw_model, mjw_data)
            mjw_smooth.crb(mjw_model, mjw_data)
            mjw_smooth.tendon_armature(mjw_model, mjw_data)
            mjw_smooth.factor_m(mjw_model, mjw_data)
            mjw_smooth.transmission(mjw_model, mjw_data)
            mjw_forward.fwd_velocity(mjw_model, mjw_data)
            mjw_forward.fwd_actuation(mjw_model, mjw_data)
            mjw_forward.fwd_acceleration(mjw_model, mjw_data, factorize=False)

            # MuJoCo's Euler integrator updates qvel before qpos.  _advance
            # also advances stateful actuators, preserving the same actuator
            # timeline as the final solver will use for this substep.
            wp.copy(mjw_data.qacc, mjw_data.qacc_smooth)
            mjw_forward._advance(mjw_model, mjw_data, mjw_data.qacc_smooth)
            solver._update_newton_state(self.model, self.state_hat, mjw_data, state_prev=state)

        return ArticulationPredictorResult(
            state_hat=self.state_hat,
            mass=mjw_data.M,
            mass_factor=mjw_data.qLD,
            mass_factor_diagonal_inverse=mjw_data.qLDiagInv,
            smooth_acceleration=mjw_data.qacc_smooth,
        )

    def apply_tangent_delta(self, delta: wp.array, dt: float) -> State:
        """Retract a q-block correction from ``q_hat`` and refresh FK.

        The correction uses MuJoCo's own quaternion-aware ``qpos`` integration,
        not raw Newton-coordinate addition.  The resulting velocity is
        ``qd_hat + delta / dt``; later corrector sweeps may replace that with a
        manifold difference if they compose more than one correction.

        Args:
            delta: Tangent endpoint correction shaped ``[world_count, nv]``.
            dt: Timestep [s].

        Returns:
            Mutable trial state valid until the next predictor call.
        """
        if dt <= 0.0:
            raise ValueError("MuJoCo tangent retraction timestep dt must be positive")
        data = self._solver.mjw_data
        model = self._solver.mjw_model
        assert data is not None
        assert model is not None
        expected_shape = (data.nworld, model.nv)
        if delta.shape != expected_shape:
            raise ValueError(f"tangent correction must have shape {expected_shape}, got {delta.shape}")

        from mujoco_warp._src import forward as mjw_forward

        with wp.ScopedDevice(self.model.device), self._solver._scoped_mujoco_warp_execution():
            wp.launch(
                _scale_tangent_delta,
                dim=expected_shape,
                inputs=[delta, 1.0 / dt],
                outputs=[self._correction_velocity],
                device=self.model.device,
            )
            wp.launch(
                mjw_forward._next_position,
                dim=(data.nworld, model.njnt),
                inputs=[
                    model.opt.timestep,
                    model.jnt_type,
                    model.jnt_qposadr,
                    model.jnt_dofadr,
                    data.qpos,
                    self._correction_velocity,
                    1.0,
                ],
                outputs=[data.qpos],
                device=self.model.device,
            )
            wp.launch(
                _add_tangent_velocity,
                dim=expected_shape,
                inputs=[data.qvel, self._correction_velocity],
                device=self.model.device,
            )
            mjw_forward.fwd_kinematics(model, data)
            self._solver._update_newton_state(self.model, self.state_trial, data, state_prev=self.state_hat)
        return self.state_trial

    def solve_mass(self, right_hand_side: wp.array, out: wp.array) -> None:
        """Solve ``M(q_n) out = right_hand_side`` using the cached factor.

        Args:
            right_hand_side: Generalized force vectors [N or N*m], shaped
                ``[world_count, nv]`` in MuJoCo-Warp tangent-DOF order.
            out: Output generalized accelerations [m/s^2 or rad/s^2], with
                the same shape as ``right_hand_side``.
        """
        from mujoco_warp._src import smooth as mjw_smooth

        data = self._solver.mjw_data
        model = self._solver.mjw_model
        assert data is not None
        assert model is not None
        expected_shape = (data.nworld, model.nv)
        if right_hand_side.shape != expected_shape or out.shape != expected_shape:
            raise ValueError(
                f"mass solve arrays must have shape {expected_shape}, got {right_hand_side.shape} and {out.shape}"
            )
        with wp.ScopedDevice(self.model.device), self._solver._scoped_mujoco_warp_execution():
            mjw_smooth.solve_m(model, data, out, right_hand_side)

    def point_jacobian(
        self,
        points: wp.array[wp.vec3],
        mujoco_bodies: wp.array[int],
        out: wp.array3d[float],
    ) -> None:
        """Evaluate translational point Jacobians at the cached start state.

        Args:
            points: One world-space point [m] for each MuJoCo world.
            mujoco_bodies: MuJoCo body id for each point.
            out: Jacobians shaped ``[world_count, 3, nv]``.
        """
        from mujoco_warp._src import support as mjw_support

        data = self._solver.mjw_data
        model = self._solver.mjw_model
        assert data is not None
        assert model is not None
        if points.shape != (data.nworld,) or mujoco_bodies.shape != (data.nworld,):
            raise ValueError("points and mujoco_bodies must contain one entry per MuJoCo world")
        if out.shape != (data.nworld, 3, model.nv):
            raise ValueError(f"point Jacobian output must have shape {(data.nworld, 3, model.nv)}")
        mjw_support.jac(model, data, out, None, points, mujoco_bodies)
