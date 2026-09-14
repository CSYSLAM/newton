# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Elastoplastic scored-fold forces for Newton's MuJoCo solver backend."""

from __future__ import annotations

import mujoco
import warp as wp

import newton

from .model import CREASE_NAMES, initial_configuration


@wp.kernel(enable_backward=False)
def _apply_mujoco_elastoplastic_creases(
    opt_timestep: wp.array[wp.float32],
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    qpos_address: wp.array[wp.int32],
    dof_address: wp.array[wp.int32],
    linear_stiffness: wp.array[wp.float32],
    cubic_stiffness: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    yield_angle: wp.array[wp.float32],
    plastic_rate: wp.array[wp.float32],
    damage_rate: wp.array[wp.float32],
    forward_yield_angle: wp.array[wp.float32],
    qfrc_passive: wp.array2d[wp.float32],
    rest_angle: wp.array[wp.float32],
    plastic_slip: wp.array[wp.float32],
    damage: wp.array[wp.float32],
):
    world_id, crease_id = wp.tid()
    q = qpos[world_id, qpos_address[crease_id]]
    velocity = qvel[world_id, dof_address[crease_id]]
    rest = rest_angle[crease_id]
    delta = q - rest
    current_damage = damage[crease_id]
    torque = (
        -(1.0 - current_damage)
        * (linear_stiffness[crease_id] * delta + cubic_stiffness[crease_id] * delta * delta * delta)
        - damping[crease_id] * velocity
    )
    qfrc_passive[world_id, dof_address[crease_id]] += torque

    threshold = yield_angle[crease_id]
    if delta > 0.0:
        threshold = forward_yield_angle[crease_id]
    excess = wp.abs(delta) - threshold
    if excess > 0.0:
        increment = wp.sign(delta) * plastic_rate[crease_id] * excess * opt_timestep[world_id]
        rest_angle[crease_id] = rest + increment
        accumulated_slip = plastic_slip[crease_id] + wp.abs(increment)
        plastic_slip[crease_id] = accumulated_slip
        damage[crease_id] = wp.min(0.42, damage_rate[crease_id] * accumulated_slip)


class CreaseModel:
    """Store scored-fold history and apply its generalized forces."""

    def __init__(self, model: newton.Model, reference_model: mujoco.MjModel):
        labels = model.joint_label
        joint_ids = []
        for name in CREASE_NAMES:
            matches = [index for index, label in enumerate(labels) if label.endswith(f"/{name}")]
            if len(matches) != 1:
                raise ValueError(f"Expected one Newton joint ending in {name!r}, found {len(matches)}")
            joint_ids.append(matches[0])
        q_starts = model.joint_q_start.numpy()
        qd_starts = model.joint_qd_start.numpy()
        device = model.device
        self.qpos_address = wp.array(q_starts[joint_ids], dtype=wp.int32, device=device)
        self.dof_address = wp.array(qd_starts[joint_ids], dtype=wp.int32, device=device)
        self.linear_stiffness = wp.array([0.20, 0.004, 0.004, 0.12, 0.004, 0.004], dtype=wp.float32, device=device)
        self.cubic_stiffness = wp.array([0.012, 0.0004, 0.0004, 0.012, 0.0004, 0.0004], dtype=wp.float32, device=device)
        self.damping = wp.array([0.003, 0.0004, 0.0004, 0.002, 0.0003, 0.0003], dtype=wp.float32, device=device)
        self.yield_angle = wp.array([0.23, 0.30, 0.30, 0.20, 0.30, 0.30], dtype=wp.float32, device=device)
        self.plastic_rate = wp.array([7.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=wp.float32, device=device)
        self.damage_rate = wp.array([0.04, 0.06, 0.06, 0.05, 0.06, 0.06], dtype=wp.float32, device=device)
        self.forward_yield_angle = wp.array([0.08, 0.30, 0.30, 0.10, 0.30, 0.30], dtype=wp.float32, device=device)

        reference_joint_ids = [
            mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CREASE_NAMES
        ]
        equilibrium = mujoco.MjData(reference_model)
        equilibrium.qpos[:] = initial_configuration(reference_model)
        mujoco.mj_forward(reference_model, equilibrium)
        stiffness = self.linear_stiffness.numpy()
        cubic = self.cubic_stiffness.numpy()
        gravity = equilibrium.qfrc_bias[reference_model.jnt_dofadr[reference_joint_ids]]
        preload = gravity / stiffness
        for _ in range(8):
            preload -= (stiffness * preload + cubic * preload**3 - gravity) / (stiffness + 3.0 * cubic * preload**2)
        self.initial_rest = equilibrium.qpos[reference_model.jnt_qposadr[reference_joint_ids]] + preload
        self.rest_angle = wp.array(self.initial_rest, dtype=wp.float32, device=device)
        self.plastic_slip = wp.zeros_like(self.rest_angle)
        self.damage = wp.zeros_like(self.rest_angle)

    def install_mujoco_callback(self, solver) -> None:
        """Install the fold law at MuJoCo's passive-force evaluation point."""

        def passive_callback(mujoco_model, mujoco_data):
            wp.launch(
                _apply_mujoco_elastoplastic_creases,
                dim=(mujoco_data.nworld, len(CREASE_NAMES)),
                inputs=[
                    mujoco_model.opt.timestep,
                    mujoco_data.qpos,
                    mujoco_data.qvel,
                    self.qpos_address,
                    self.dof_address,
                    self.linear_stiffness,
                    self.cubic_stiffness,
                    self.damping,
                    self.yield_angle,
                    self.plastic_rate,
                    self.damage_rate,
                    self.forward_yield_angle,
                ],
                outputs=[mujoco_data.qfrc_passive, self.rest_angle, self.plastic_slip, self.damage],
                device=self.rest_angle.device,
            )

        solver.mjw_model.callback.passive = passive_callback
