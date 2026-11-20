"""Unitree RL Lab Go2 MDP implemented with SynReal/Newton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from locomotion.env import SynRealLocomotionEnv
from locomotion.randomization import uniform

from .config import CONFIG_PATH, load_config
from .contacts import LEGS, classify_shapes
from .observations import actor_observation, critic_observation
from .rewards import unitree_go2_reward


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    twice_cross = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + quaternion[..., 3:4] * twice_cross + torch.linalg.cross(q_xyz, twice_cross, dim=-1)


class UnitreeGo2Env(SynRealLocomotionEnv):
    """Vectorized Go2 velocity task matching Unitree RL Lab's active terms."""

    def __init__(
        self,
        *,
        num_envs: int = 512,
        device: str = "cuda:0",
        asset: Path | None = None,
        seed: int = 42,
        randomize: bool = True,
        auto_reset: bool = True,
        viewer: Any | None = None,
        use_cuda_graph: bool = True,
        command_vx: float | None = None,
        friction_randomization: str = "batch",
        enable_self_collisions: bool = False,
    ) -> None:
        if friction_randomization not in {"per-env", "batch", "fixed"}:
            raise ValueError("friction_randomization must be per-env, batch, or fixed")
        self.friction_randomization = friction_randomization
        self.enable_self_collisions = enable_self_collisions
        _, self.unitree_cfg = load_config()
        super().__init__(
            CONFIG_PATH,
            num_envs=num_envs,
            device=device,
            asset=asset,
            seed=seed,
            randomize=randomize,
            auto_reset=auto_reset,
            viewer=viewer,
            use_cuda_graph=use_cuda_graph,
            command_vx=command_vx,
        )

    def _asset_options(self) -> dict[str, bool]:
        return {
            "floating": True,
            "collapse_fixed_joints": True,
            "enable_self_collisions": self.enable_self_collisions,
        }

    def _configure_solver(self) -> None:
        cha = self.cha
        config = self.solver.config
        config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SIX_SPRINGS
        config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        config.joint_constraint_primal_iterations = 2
        config.nonlinear_iterations = 30
        config.linear_iterations = 10

    def _configure_model_before_solver(self) -> None:
        shape_body = self.wp.to_torch(self.model.shape_body).long()
        robot_shapes = shape_body >= 0
        if int(self.model.body_count) % self.num_envs:
            raise RuntimeError("replicated model bodies are not world-major")
        body_per_env = int(self.model.body_count) // self.num_envs
        shape_env = torch.div(shape_body[robot_shapes], body_per_env, rounding_mode="floor")
        friction = torch.ones(self.num_envs, device=self.device)
        if self.randomize and self.friction_randomization == "per-env":
            friction.copy_(
                uniform(
                    (self.num_envs,),
                    self.spec.randomization.dynamic_friction,
                    self.device,
                    self.generator,
                )
            )
        elif self.randomize and self.friction_randomization == "batch":
            friction.fill_(
                float(
                    uniform(
                        (1,),
                        self.spec.randomization.dynamic_friction,
                        self.device,
                        self.generator,
                    )[0]
                )
            )
        shape_friction = self.wp.to_torch(self.model.shape_material_mu)
        shape_friction[torch.nonzero(robot_shapes, as_tuple=False).squeeze(-1)] = friction[shape_env]
        self.friction = friction
        torch.cuda.current_stream(self.device).synchronize()

    def _configure_task_buffers(self) -> None:
        self.num_privileged_obs = 60
        classification = classify_shapes(
            shape_labels=list(self.model.shape_label),
            shape_bodies=self.model.shape_body.numpy().tolist(),
            body_labels=list(self.model.body_label),
            body_per_env=self.body_per_env,
            num_envs=self.num_envs,
        )
        self.undesired_per_env = classification.undesired_per_env
        self.shape_to_foot = torch.tensor(classification.shape_to_foot, dtype=torch.int32, device=self.device)
        self.shape_to_undesired = torch.tensor(classification.shape_to_undesired, dtype=torch.int32, device=self.device)
        self.shape_to_base_env = torch.tensor(classification.shape_to_base_env, dtype=torch.int32, device=self.device)

        foot_shape = (self.num_envs, len(LEGS))
        self.foot_contact = torch.zeros(foot_shape, dtype=torch.int32, device=self.device)
        self.foot_contact_any = torch.zeros_like(self.foot_contact)
        self.previous_foot_contact = torch.zeros_like(self.foot_contact)
        self.first_foot_contact = torch.zeros_like(self.foot_contact)
        self.current_air_time = torch.zeros(foot_shape, device=self.device)
        self.current_contact_time = torch.zeros_like(self.current_air_time)
        self.last_air_time = torch.zeros_like(self.current_air_time)
        self.last_contact_time = torch.zeros_like(self.current_air_time)
        self.undesired_contact = torch.zeros(
            (self.num_envs, self.undesired_per_env),
            dtype=torch.int32,
            device=self.device,
        )
        self.base_contact = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.applied_torque = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_joint_velocity = torch.zeros_like(self.applied_torque)

        lower = (
            self.wp.to_torch(self.model.joint_limit_lower)
            .view(self.num_envs, self.qd_per_env)[0]
            .index_select(0, self.policy_qd)
        )
        upper = (
            self.wp.to_torch(self.model.joint_limit_upper)
            .view(self.num_envs, self.qd_per_env)[0]
            .index_select(0, self.policy_qd)
        )
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        factor = float(self.unitree_cfg["soft_joint_pos_limit_factor"])
        self.soft_joint_lower = midpoint - factor * half_range
        self.soft_joint_upper = midpoint + factor * half_range

        first_body_labels = self.model.body_label[: self.body_per_env]
        body_by_name = {_basename(label): body for body, label in enumerate(first_body_labels)}
        self.calf_bodies = torch.tensor(
            [body_by_name[f"{leg}_calf"] for leg in LEGS],
            dtype=torch.long,
            device=self.device,
        )
        self.foot_offset = torch.tensor(self.unitree_cfg["foot_offset"], dtype=torch.float32, device=self.device)
        self.reward_weights = {key: float(value) for key, value in self.unitree_cfg["rewards"].items()}
        self.episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.reward_weights}
        self.diagnostic_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in ("abs_command_vx", "command_aligned_vx", "abs_vx_error")
        }
        self.curriculum_score_sum = 0.0
        self.curriculum_episode_count = 0
        self.command_ranges = {
            key: list(self.spec.command_ranges[key]) for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
        }
        self.command_limits = {
            key: tuple(float(value) for value in values) for key, values in self.unitree_cfg["command_limits"].items()
        }

    def _bind_capture_arrays(self) -> None:
        super()._bind_capture_arrays()
        wp = self.wp
        self._shape_to_foot_wp = wp.from_torch(self.shape_to_foot)
        self._shape_to_undesired_wp = wp.from_torch(self.shape_to_undesired)
        self._shape_to_base_env_wp = wp.from_torch(self.shape_to_base_env)
        self._foot_contact_wp = wp.from_torch(self.foot_contact.flatten())
        self._foot_contact_any_wp = wp.from_torch(self.foot_contact_any.flatten())
        self._previous_foot_contact_wp = wp.from_torch(self.previous_foot_contact.flatten())
        self._first_foot_contact_wp = wp.from_torch(self.first_foot_contact.flatten())
        self._current_air_time_wp = wp.from_torch(self.current_air_time.flatten())
        self._current_contact_time_wp = wp.from_torch(self.current_contact_time.flatten())
        self._last_air_time_wp = wp.from_torch(self.last_air_time.flatten())
        self._last_contact_time_wp = wp.from_torch(self.last_contact_time.flatten())
        self._undesired_contact_wp = wp.from_torch(self.undesired_contact.flatten())
        self._base_contact_wp = wp.from_torch(self.base_contact)
        self._applied_torque_wp = wp.from_torch(self.applied_torque)

    def _frame(self, corrupt: bool = True) -> torch.Tensor:
        _, _, angular, gravity, q, qd = self._state()
        return actor_observation(
            angular_velocity_body=angular,
            gravity_body=gravity,
            commands=self.commands,
            joint_position=q,
            joint_velocity=qd,
            last_action=self.actions,
            default_joint_position=self.default_joint_pos,
            corrupt=corrupt and self.randomize,
            generator=self.generator,
        )

    def _critic_observation(self) -> torch.Tensor:
        _, linear, angular, gravity, q, qd = self._state()
        return critic_observation(
            linear_velocity_body=linear,
            angular_velocity_body=angular,
            gravity_body=gravity,
            commands=self.commands,
            joint_position=q,
            joint_velocity=qd,
            joint_effort=self.applied_torque,
            last_action=self.actions,
            default_joint_position=self.default_joint_pos,
        )

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        observation = self.history.value()
        return observation, {"observations": {"critic": self._critic_observation()}}

    def get_privileged_observations(self) -> torch.Tensor:
        return self._critic_observation()

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        for index, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            self.commands[env_ids, index] = uniform(
                (len(env_ids),),
                tuple(self.command_ranges[key]),
                self.device,
                self.generator,
            )
        standing = (
            torch.rand((len(env_ids),), device=self.device, generator=self.generator) < self.spec.standing_probability
        )
        self.commands[env_ids[standing]] = 0.0
        if self.command_vx is not None:
            self.commands[env_ids, 0] = self.command_vx

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        observation = super().reset(env_ids)
        if not len(env_ids):
            return observation
        if self.randomize:
            root = self.joint_q_0[env_ids, :3]
            root[:, :2] += uniform(
                (len(env_ids), 2),
                tuple(self.unitree_cfg["reset_position_xy"]),
                self.device,
                self.generator,
            )
            yaw = uniform(
                (len(env_ids),),
                tuple(self.unitree_cfg["reset_yaw"]),
                self.device,
                self.generator,
            )
            self.joint_q_0[env_ids, 3:7] = torch.stack(
                (
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(0.5 * yaw),
                    torch.cos(0.5 * yaw),
                ),
                dim=-1,
            )
            velocity = uniform(
                (len(env_ids), self.num_actions),
                tuple(self.unitree_cfg["reset_joint_velocity"]),
                self.device,
                self.generator,
            )
            self.joint_qd_0[env_ids[:, None], self.policy_qd] = velocity
            self.joint_q_1[env_ids] = self.joint_q_0[env_ids]
            self.joint_qd_1[env_ids] = self.joint_qd_0[env_ids]

        for tensor in (
            self.foot_contact,
            self.foot_contact_any,
            self.previous_foot_contact,
            self.first_foot_contact,
            self.current_air_time,
            self.current_contact_time,
            self.last_air_time,
            self.last_contact_time,
            self.undesired_contact,
        ):
            tensor[env_ids] = 0
        self.base_contact[env_ids] = 0
        self.applied_torque[env_ids] = 0.0
        qd = self.joint_qd_0.index_select(1, self.policy_qd)
        self.previous_joint_velocity[env_ids] = qd[env_ids]
        for episode_sum in self.episode_sums.values():
            episode_sum[env_ids] = 0.0
        for diagnostic_sum in self.diagnostic_sums.values():
            diagnostic_sum[env_ids] = 0.0

        torch.cuda.current_stream(self.device).synchronize()
        with self.wp.ScopedStream(self.stream):
            self.newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.solver.sync_affine_body_state(self.state_0)
            self.solver.sync_affine_body_state(self.state_1)
        self.wp.synchronize_stream(self.stream)
        frame = self._frame(corrupt=False)
        self.history.reset(env_ids, frame)
        return self.history.value()

    def _launch_limited_joint_targets(self) -> None:
        from .simulation_kernels import update_unitree_joint_targets

        motor = self.unitree_cfg["torque_speed"]
        self.wp.launch(
            update_unitree_joint_targets,
            dim=self.num_envs * self.num_actions,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._desired_targets_wp,
                self._policy_q_wp,
                self._policy_qd_wp,
                self._policy_targets_wp,
                self.num_actions,
                self.q_per_env,
                self.qd_per_env,
                self.target_per_env,
                float(self.spec.stiffness[0]),
                float(self.spec.damping[0]),
                float(motor["x1"]),
                float(motor["x2"]),
                float(motor["y1"]),
                float(motor["y2"]),
            ],
            outputs=[self.control.joint_target_q, self._applied_torque_wp],
            device=self.wp_device,
            stream=self.stream,
        )

    def _launch_contact_aggregation(self) -> None:
        from .simulation_kernels import (
            aggregate_rigid_contacts,
            update_contact_timers,
        )

        self.wp.launch(
            aggregate_rigid_contacts,
            dim=len(self.contacts.rigid_contact_shape0),
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.state_0.body_q,
                self._shape_to_foot_wp,
                self._shape_to_undesired_wp,
                self._shape_to_base_env_wp,
            ],
            outputs=[
                self._foot_contact_wp,
                self._foot_contact_any_wp,
                self._undesired_contact_wp,
                self._base_contact_wp,
            ],
            device=self.wp_device,
            stream=self.stream,
        )
        self.wp.launch(
            update_contact_timers,
            dim=self.num_envs * len(LEGS),
            inputs=[self._foot_contact_wp, self.spec.dt],
            outputs=[
                self._previous_foot_contact_wp,
                self._first_foot_contact_wp,
                self._current_air_time_wp,
                self._current_contact_time_wp,
                self._last_air_time_wp,
                self._last_contact_time_wp,
            ],
            device=self.wp_device,
            stream=self.stream,
            block_dim=256,
        )

    def _simulate_uncaptured(self, *, keep_state_buffers: bool = False) -> None:
        from .simulation_kernels import clear_policy_contacts, clear_substep_contacts

        self.wp.launch(
            clear_policy_contacts,
            dim=max(
                self.num_envs * len(LEGS),
                self.num_envs * self.undesired_per_env,
            ),
            inputs=[],
            outputs=[
                self._foot_contact_any_wp,
                self._first_foot_contact_wp,
                self._undesired_contact_wp,
                self._base_contact_wp,
            ],
            device=self.wp_device,
            stream=self.stream,
        )
        for substep in range(self.spec.decimation):
            self.wp.launch(
                clear_substep_contacts,
                dim=self.num_envs * len(LEGS),
                inputs=[],
                outputs=[self._foot_contact_wp],
                device=self.wp_device,
                stream=self.stream,
            )
            self._launch_limited_joint_targets()
            self.newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self._launch_contact_aggregation()
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.spec.dt,
            )
            is_last_odd_substep = (
                keep_state_buffers and self.spec.decimation % 2 == 1 and substep == self.spec.decimation - 1
            )
            if is_last_odd_substep:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self._bind_views()

    def _foot_velocity_xy(self) -> torch.Tensor:
        calf_pose = self.body_q_0.index_select(1, self.calf_bodies)
        calf_velocity = self.body_qd_0.index_select(1, self.calf_bodies)
        quaternion = torch.nn.functional.normalize(calf_pose[:, :, 3:7], dim=-1)
        offset = self.foot_offset.expand(self.num_envs, len(LEGS), 3)
        world_offset = _quat_rotate(quaternion, offset)
        foot_velocity = calf_velocity[:, :, :3] + torch.linalg.cross(calf_velocity[:, :, 3:], world_offset, dim=-1)
        return foot_velocity[:, :, :2]

    def _resample_due_commands(self) -> None:
        interval = round(float(self.unitree_cfg["command_resampling_time_s"]) / self.spec.policy_dt)
        due = (self.episode_length_buf > 0) & (self.episode_length_buf.remainder(interval) == 0)
        if torch.any(due):
            self._sample_commands(torch.nonzero(due, as_tuple=False).squeeze(-1))

    def _update_command_curriculum(self, done: torch.Tensor) -> None:
        env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if len(env_ids):
            self.curriculum_score_sum += float(self.episode_sums["track_lin_vel_xy"][env_ids].sum().item())
            self.curriculum_episode_count += len(env_ids)
        if self.policy_step_count % self.max_episode_length:
            return
        if not self.curriculum_episode_count:
            return
        score = self.curriculum_score_sum / self.curriculum_episode_count
        score /= self.spec.episode_length_s
        self.curriculum_score_sum = 0.0
        self.curriculum_episode_count = 0
        threshold = self.reward_weights["track_lin_vel_xy"] * float(self.unitree_cfg["command_curriculum_threshold"])
        if score <= threshold:
            return
        delta = float(self.unitree_cfg["command_curriculum_step"])
        for key in ("lin_vel_x", "lin_vel_y"):
            lower, upper = self.command_ranges[key]
            limit_lower, limit_upper = self.command_limits[key]
            self.command_ranges[key] = [
                max(limit_lower, lower - delta),
                min(limit_upper, upper + delta),
            ]

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"expected actions {(self.num_envs, self.num_actions)}, got {actions.shape}")
        _, _, _, _, _, qd_before = self._state()
        self.previous_joint_velocity.copy_(qd_before)
        actions = actions.to(self.device).clamp(-self.action_clip, self.action_clip)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self.desired_targets.copy_(self.default_joint_pos + self.action_scale * actions)
        self._push_due_envs()

        torch.cuda.current_stream(self.device).synchronize()
        with self.wp.ScopedStream(self.stream):
            self._simulate()
            self.newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        self.wp.synchronize_stream(self.stream)
        self.sim_time += self.spec.policy_dt
        self.render()
        self.episode_length_buf += 1
        self.steps_since_reset_buf += 1
        self.policy_step_count += 1

        root, linear, angular, gravity, q, qd = self._state()
        joint_acceleration = (qd - self.previous_joint_velocity) / self.spec.policy_dt
        reward, terms = unitree_go2_reward(
            commands=self.commands,
            linear_velocity_body=linear,
            angular_velocity_body=angular,
            gravity_body=gravity,
            joint_position_error=q - self.default_joint_pos,
            joint_velocity=qd,
            joint_acceleration=joint_acceleration,
            joint_torque=self.applied_torque,
            actions=self.actions,
            previous_actions=self.previous_actions,
            soft_joint_lower=self.soft_joint_lower,
            soft_joint_upper=self.soft_joint_upper,
            joint_position=q,
            first_foot_contact=self.first_foot_contact,
            last_air_time=self.last_air_time,
            last_contact_time=self.last_contact_time,
            foot_contact=self.foot_contact_any,
            foot_velocity_xy=self._foot_velocity_xy(),
            undesired_contact=self.undesired_contact,
            weights=self.reward_weights,
            step_dt=self.spec.policy_dt,
        )
        finite = (
            torch.isfinite(root).all(-1)
            & torch.isfinite(q).all(-1)
            & torch.isfinite(qd).all(-1)
            & torch.isfinite(self.applied_torque).all(-1)
        )
        tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
        self.reset_buf = (self.base_contact != 0) | (tilt > float(self.unitree_cfg["bad_orientation_angle"])) | ~finite
        self.timeout_buf = self.episode_length_buf >= self.max_episode_length
        done = self.reset_buf | self.timeout_buf
        reward = torch.nan_to_num(reward, nan=-100.0, posinf=100.0, neginf=-100.0)
        reward.clamp_(-100.0, 100.0)
        self.episode_return += reward
        for name, contribution in terms.items():
            self.episode_sums[name] += contribution
        self.diagnostic_sums["abs_command_vx"] += self.commands[:, 0].abs() * self.spec.policy_dt
        self.diagnostic_sums["command_aligned_vx"] += self.commands[:, 0].sign() * linear[:, 0] * self.spec.policy_dt
        self.diagnostic_sums["abs_vx_error"] += (self.commands[:, 0] - linear[:, 0]).abs() * self.spec.policy_dt

        self._resample_due_commands()
        self._update_command_curriculum(done)
        self.extras = {
            "time_outs": self.timeout_buf.clone(),
            "reward_terms": {name: value.mean() for name, value in terms.items()},
            "curriculum": {
                "lin_vel_x_max": self.command_ranges["lin_vel_x"][1],
                "lin_vel_y_max": self.command_ranges["lin_vel_y"][1],
            },
        }
        has_done = bool(torch.any(done))
        if has_done:
            self.extras["episode"] = {
                "return": self.episode_return[done].mean(),
                "length": self.episode_length_buf[done].float().mean(),
                **{
                    f"rew_{name}": values[done].mean() / self.spec.episode_length_s
                    for name, values in self.episode_sums.items()
                },
                **{
                    name: (values[done] / (self.episode_length_buf[done] * self.spec.policy_dt)).mean()
                    for name, values in self.diagnostic_sums.items()
                },
                "curriculum_lin_vel_x_min": torch.tensor(self.command_ranges["lin_vel_x"][0], device=self.device),
                "curriculum_lin_vel_x_max": torch.tensor(self.command_ranges["lin_vel_x"][1], device=self.device),
            }

        frame = self._frame(corrupt=True)
        observation = self.history.append(frame)
        if self.auto_reset and has_done:
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            self.reset(env_ids)
            observation = self.history.value()
        self.extras["observations"] = {"critic": self._critic_observation()}
        return observation, reward, done, self.extras
