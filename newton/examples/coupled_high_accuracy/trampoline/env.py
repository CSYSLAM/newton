# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Vectorized Unitree G1 jumping environment on a deformable trampoline."""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp
import yaml
from newton.examples.coupled_high_accuracy.mesh import boundary_vertex_indices, load_trampoline_mesh, triangle_mesh_area

import newton
import newton.examples
import newton.utils
import newton.viewer
from newton import JointTargetMode, ParticleFlags
from newton.solvers import coupled_high_accuracy as cha
from newton.solvers import style3d

CONTROL_FPS = 50
SIM_SUBSTEPS = 6
ACTION_SCALE = 0.5
TRAMPOLINE_HEIGHT = 1.0
TARGET_BASE_HEIGHT = TRAMPOLINE_HEIGHT + 0.76
JUMP_TARGET_HEIGHT = 0.10
JUMP_LANDING_HEIGHT = 0.04
JUMP_REFERENCE_HEIGHT = 0.12
JUMP_FREQUENCY = 1.0


@wp.kernel
def apply_particle_world_offsets(
    particle_q: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    output_q: wp.array[wp.vec3],
) -> None:
    particle = wp.tid()
    world = particle_world[particle]
    position = particle_q[particle]
    if world >= 0 and world < world_offsets.shape[0]:
        position += world_offsets[world]
    output_q[particle] = position


def quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by inverse xyzw quaternions."""
    q_xyz = -quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    first_cross = torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + 2.0 * (q_w * first_cross + torch.linalg.cross(q_xyz, first_cross, dim=-1))


class TrampolineEnv:
    """Batched RL environment preserving the cloth-trampoline example physics."""

    def __init__(
        self,
        num_envs: int = 1,
        *,
        device: str = "cuda:0",
        viewer: Any | None = None,
        episode_length: int = 1200,
        seed: int = 42,
        sim_substeps: int = SIM_SUBSTEPS,
        contact_stiffness: float = 1.0e5,
        contact_damping: float = 1.0e3,
        shape_friction: float = 0.75,
        soft_contact_friction: float = 0.6,
        particle_radius: float = 0.025,
        foot_contact_margin: float = 0.0,
        spawn_height_offset: float = 0.0,
        enable_joint_drives: bool = True,
        auto_reset: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if sim_substeps <= 0:
            raise ValueError("sim_substeps must be positive")
        if contact_stiffness < 0.0 or contact_damping < 0.0:
            raise ValueError("contact stiffness and damping must be non-negative")
        if shape_friction < 0.0 or soft_contact_friction < 0.0:
            raise ValueError("contact friction must be non-negative")
        if particle_radius <= 0.0:
            raise ValueError("particle_radius must be positive")
        if foot_contact_margin < 0.0:
            raise ValueError("foot_contact_margin must be non-negative")
        if not torch.cuda.is_available():
            raise RuntimeError("TrampolineEnv requires a CUDA-enabled PyTorch build")

        self.num_envs = num_envs
        self.episode_length = episode_length
        self.auto_reset = auto_reset
        self.frame_dt = 1.0 / CONTROL_FPS
        self.sim_dt = self.frame_dt / sim_substeps
        self.sim_substeps = sim_substeps
        self.target_base_height = TARGET_BASE_HEIGHT + spawn_height_offset
        self.sim_time = 0.0
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise RuntimeError("TrampolineEnv requires a CUDA Warp device")
        wp.set_device(self.device)
        self.torch_device = torch.device(str(self.device))
        torch.cuda.set_device(self.torch_device)
        torch.manual_seed(seed)
        self.generator = torch.Generator(device=self.torch_device).manual_seed(seed)
        self.stream = wp.get_stream(self.device)

        (
            builder,
            controlled_dofs,
            policy_dofs,
            policy_action_indices,
        ) = self._build_scene(
            num_envs,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
            shape_friction=shape_friction,
            particle_radius=particle_radius,
            foot_contact_margin=foot_contact_margin,
            initial_base_height=self.target_base_height,
            enable_joint_drives=enable_joint_drives,
        )

        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = contact_stiffness
        self.model.soft_contact_kd = contact_damping
        self.model.soft_contact_mu = soft_contact_friction

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.constraint_alpha = 200.0
        self.solver.config.nonlinear_iterations = 15
        self.solver.config.linear_iterations = 6

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
        )
        self.state_1.assign(self.state_0)

        self.joint_q_per_env = len(self.state_0.joint_q) // num_envs
        self.joint_qd_per_env = len(self.state_0.joint_qd) // num_envs
        self.particles_per_env = len(self.state_0.particle_q) // num_envs
        self.bodies_per_env = len(self.state_0.body_q) // num_envs
        self.targets_per_env = len(self.control.joint_target_q) // num_envs
        if any(
            count * num_envs != total
            for count, total in (
                (self.joint_q_per_env, len(self.state_0.joint_q)),
                (self.joint_qd_per_env, len(self.state_0.joint_qd)),
                (self.particles_per_env, len(self.state_0.particle_q)),
                (self.bodies_per_env, len(self.state_0.body_q)),
                (self.targets_per_env, len(self.control.joint_target_q)),
            )
        ):
            raise RuntimeError("Replicated model arrays are not world-contiguous")

        self.controlled_dofs = torch.tensor(controlled_dofs, dtype=torch.long, device=self.torch_device)
        self.policy_dofs = torch.tensor(policy_dofs, dtype=torch.long, device=self.torch_device)
        self.policy_action_indices = torch.tensor(policy_action_indices, dtype=torch.long, device=self.torch_device)
        self.controlled_q = self.controlled_dofs + 7
        self.controlled_qd = self.controlled_dofs + 6
        self.controlled_targets = self.controlled_dofs + 6
        self.policy_q = self.policy_dofs + 7
        self.policy_qd = self.policy_dofs + 6
        self.action_dim = len(controlled_dofs)
        self.policy_action_dim = len(policy_dofs)
        if self.action_dim != 29:
            raise RuntimeError(f"Expected 29 non-hand G1 joints, found {self.action_dim}")
        if self.policy_action_dim != 43:
            raise RuntimeError(f"Expected 43 pretrained-policy joints, found {self.policy_action_dim}")
        self.observation_dim = 16 + 3 * self.policy_action_dim

        self._bind_torch_views()
        self.initial_joint_q = self.joint_q_0.clone()
        self.initial_joint_qd = self.joint_qd_0.clone()
        self.initial_particle_q = self.particle_q_0.clone()
        self.initial_particle_qd = self.particle_qd_0.clone()
        self.initial_body_q = self.body_q_0.clone()
        self.initial_body_qd = self.body_qd_0.clone()
        self.default_targets = self.control_targets.clone()
        self.default_joint_positions = self.initial_joint_q.index_select(1, self.controlled_q)
        self.default_policy_joint_positions = self.initial_joint_q.index_select(1, self.policy_q)

        self.previous_actions = torch.zeros((num_envs, self.action_dim), device=self.torch_device)
        self.policy_previous_actions = torch.zeros((num_envs, self.policy_action_dim), device=self.torch_device)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.torch_device)
        self.episode_returns = torch.zeros(num_envs, device=self.torch_device)
        self.jump_armed = torch.zeros(num_envs, dtype=torch.bool, device=self.torch_device)
        self.jump_cycle_base_height = self.initial_joint_q[:, 2].clone()
        self.jump_cycle_peak = torch.zeros(num_envs, device=self.torch_device)
        self.jump_peak_height = torch.zeros(num_envs, device=self.torch_device)
        self.jump_counts = torch.zeros(num_envs, dtype=torch.long, device=self.torch_device)
        self.steps_since_jump = torch.zeros(num_envs, dtype=torch.long, device=self.torch_device)
        self.last_metrics: dict[str, torch.Tensor] = {}

        self.viewer = viewer if viewer is not None else newton.viewer.ViewerNull()
        self.viewer.set_model(self.model)
        if num_envs > 1:
            self.viewer.set_world_offsets((3.0, 3.0, 0.0))
        self.render_state = self.model.state() if num_envs > 1 else None
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(3.8, -4.2, 2.7), pitch=-12.0, yaw=42.0)

        self.graph = None
        with wp.ScopedStream(self.stream):
            with wp.ScopedCapture(stream=self.stream) as capture:
                self._simulate()
            self.graph = capture.graph
        self.reset()

    @staticmethod
    def _build_scene(
        num_envs: int,
        *,
        contact_stiffness: float,
        contact_damping: float,
        shape_friction: float,
        particle_radius: float,
        foot_contact_margin: float,
        initial_base_height: float,
        enable_joint_drives: bool,
    ) -> tuple[newton.ModelBuilder, list[int], list[int], list[int]]:
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(scene)
        scene.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0,
        )
        scene.default_shape_cfg.ke = contact_stiffness
        scene.default_shape_cfg.kd = contact_damping
        scene.default_shape_cfg.kf = 1.0e3
        scene.default_shape_cfg.mu = shape_friction

        asset_path = newton.utils.download_asset("unitree_g1")
        with (asset_path / "rl_policies" / "g1_29dof.yaml").open(encoding="utf-8") as config_file:
            policy_config = yaml.safe_load(config_file)
        scene.add_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0.0, 0.0, initial_base_height)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )

        joint_by_name = {
            scene.joint_label[joint].rsplit("/", maxsplit=1)[-1]: joint for joint in range(1, len(scene.joint_label))
        }
        policy_joint_names = list(policy_config["mjw_joint_names"])
        missing_joints = sorted(set(policy_joint_names) - set(joint_by_name))
        if missing_joints:
            raise ValueError(f"G1 asset is missing pretrained-policy joints: {missing_joints}")

        policy_dofs: list[int] = []
        controlled_dofs: list[int] = []
        policy_action_indices: list[int] = []
        for policy_index, joint_name in enumerate(policy_joint_names):
            joint = joint_by_name[joint_name]
            q_index = int(scene.joint_q_start[joint])
            qd_index = int(scene.joint_qd_start[joint])
            dof = qd_index - 6
            policy_dofs.append(dof)
            scene.joint_q[q_index] = policy_config["mjw_joint_pos"][policy_index]
            scene.joint_target_ke[qd_index] = (
                policy_config["mjw_joint_stiffness"][policy_index] if enable_joint_drives else 0.0
            )
            scene.joint_target_kd[qd_index] = (
                policy_config["mjw_joint_damping"][policy_index] if enable_joint_drives else 0.0
            )
            scene.joint_armature[qd_index] = policy_config["mjw_joint_armature"][policy_index]
            scene.joint_target_mode[qd_index] = int(
                JointTargetMode.POSITION if enable_joint_drives else JointTargetMode.NONE
            )
            if "_hand_" not in joint_name:
                controlled_dofs.append(dof)
                policy_action_indices.append(policy_index)

        # Initialize at the pretrained standing pose, translated onto the cloth.
        scene.joint_q[:3] = [0.0, 0.0, initial_base_height]
        scene.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]

        scene.approximate_meshes("convex_hull")
        for shape_index in range(len(scene.shape_body)):
            scene.shape_material_ke[shape_index] = contact_stiffness
            scene.shape_material_kd[shape_index] = contact_damping
            if "/ankle_roll_link/collisions/" in scene.shape_label[shape_index]:
                scene.shape_margin[shape_index] = foot_contact_margin

        vertices, indices = load_trampoline_mesh()
        trampoline_start = len(scene.particle_q)
        style3d.add_cloth_mesh(
            scene,
            pos=wp.vec3(0.0, 0.0, TRAMPOLINE_HEIGHT),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=vertices,
            indices=indices,
            density=5.0 / triangle_mesh_area(vertices, indices),
            tri_aniso_ke=wp.vec3(2.0e5, 2.0e5, 2.0e5),
            edge_aniso_ke=wp.vec3(2.0e-4, 2.0e-4, 2.0e-4),
            tri_kd=5.0,
            edge_kd=1.0,
            particle_radius=particle_radius,
            label="trampoline",
        )
        for vertex_index in boundary_vertex_indices(indices):
            particle = trampoline_start + vertex_index
            scene.particle_flags[particle] &= ~ParticleFlags.ACTIVE
            scene.particle_mass[particle] = 0.0

        if num_envs == 1:
            builder = scene
        else:
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            builder.default_shape_cfg.ke = contact_stiffness
            builder.default_shape_cfg.kd = contact_damping
            builder.default_shape_cfg.kf = 1.0e3
            builder.default_shape_cfg.mu = shape_friction
            builder.replicate(scene, num_envs)
        builder.add_ground_plane()
        return builder, controlled_dofs, policy_dofs, policy_action_indices

    def _bind_torch_views(self) -> None:
        def view(array: Any, rows: int) -> torch.Tensor:
            return wp.to_torch(array).view(self.num_envs, rows, -1).squeeze(-1)

        self.joint_q_0 = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self.joint_q_per_env)
        self.joint_q_1 = wp.to_torch(self.state_1.joint_q).view(self.num_envs, self.joint_q_per_env)
        self.joint_qd_0 = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, self.joint_qd_per_env)
        self.joint_qd_1 = wp.to_torch(self.state_1.joint_qd).view(self.num_envs, self.joint_qd_per_env)
        self.particle_q_0 = view(self.state_0.particle_q, self.particles_per_env)
        self.particle_q_1 = view(self.state_1.particle_q, self.particles_per_env)
        self.particle_qd_0 = view(self.state_0.particle_qd, self.particles_per_env)
        self.particle_qd_1 = view(self.state_1.particle_qd, self.particles_per_env)
        self.body_q_0 = view(self.state_0.body_q, self.bodies_per_env)
        self.body_q_1 = view(self.state_1.body_q, self.bodies_per_env)
        self.body_qd_0 = view(self.state_0.body_qd, self.bodies_per_env)
        self.body_qd_1 = view(self.state_1.body_qd, self.bodies_per_env)
        self.control_targets = wp.to_torch(self.control.joint_target_q).view(self.num_envs, self.targets_per_env)

    @torch.no_grad()
    def _observations(self) -> torch.Tensor:
        root_position = self.joint_q_0[:, :3]
        root_quaternion = torch.nn.functional.normalize(self.joint_q_0[:, 3:7], dim=-1)
        linear_velocity = quat_rotate_inverse(root_quaternion, self.joint_qd_0[:, :3])
        angular_velocity = quat_rotate_inverse(root_quaternion, self.joint_qd_0[:, 3:6])
        gravity = torch.zeros_like(linear_velocity)
        gravity[:, 2] = -1.0
        projected_gravity = quat_rotate_inverse(root_quaternion, gravity)
        joint_positions = self.joint_q_0.index_select(1, self.policy_q)
        joint_velocities = self.joint_qd_0.index_select(1, self.policy_qd)
        command = torch.zeros((self.num_envs, 3), device=self.torch_device)
        jump_height = ((root_position[:, 2] - self.jump_cycle_base_height) / JUMP_TARGET_HEIGHT).clamp(0.0, 3.5)
        jump_phase = 2.0 * math.pi * JUMP_FREQUENCY * self.frame_dt * self.episode_steps.float()
        return torch.cat(
            (
                linear_velocity,
                angular_velocity,
                projected_gravity,
                command,
                joint_positions - self.default_policy_joint_positions,
                joint_velocities,
                self.policy_previous_actions,
                jump_height.unsqueeze(-1),
                self.jump_armed.float().unsqueeze(-1),
                torch.sin(jump_phase).unsqueeze(-1),
                torch.cos(jump_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

    @torch.no_grad()
    def reset(self, done: torch.Tensor | None = None) -> torch.Tensor:
        if done is None:
            done = torch.ones(self.num_envs, dtype=torch.bool, device=self.torch_device)
        else:
            done = done.to(device=self.torch_device, dtype=torch.bool)
        if not torch.any(done):
            return self._observations()

        reset_joint_q = self.initial_joint_q[done].clone()
        reset_joint_q[:, :2] += 0.04 * (
            2.0
            * torch.rand(
                (int(done.sum().item()), 2),
                device=self.torch_device,
                generator=self.generator,
            )
            - 1.0
        )
        controlled = reset_joint_q.index_select(1, self.controlled_q)
        controlled += 0.025 * torch.randn(controlled.shape, device=self.torch_device, generator=self.generator)
        reset_joint_q[:, self.controlled_q] = controlled

        self.joint_q_0[done] = reset_joint_q
        self.joint_q_1[done] = reset_joint_q
        self.joint_qd_0[done] = self.initial_joint_qd[done]
        self.joint_qd_1[done] = self.initial_joint_qd[done]
        self.particle_q_0[done] = self.initial_particle_q[done]
        self.particle_q_1[done] = self.initial_particle_q[done]
        self.particle_qd_0[done] = self.initial_particle_qd[done]
        self.particle_qd_1[done] = self.initial_particle_qd[done]
        self.body_q_0[done] = self.initial_body_q[done]
        self.body_q_1[done] = self.initial_body_q[done]
        self.body_qd_0[done] = self.initial_body_qd[done]
        self.body_qd_1[done] = self.initial_body_qd[done]
        self.previous_actions[done] = 0.0
        self.policy_previous_actions[done] = 0.0
        self.episode_steps[done] = 0
        self.episode_returns[done] = 0.0
        self.jump_armed[done] = False
        self.jump_cycle_base_height[done] = reset_joint_q[:, 2]
        self.jump_cycle_peak[done] = 0.0
        self.jump_peak_height[done] = 0.0
        self.jump_counts[done] = 0
        self.steps_since_jump[done] = 0
        self.control_targets.copy_(self.default_targets)
        torch.cuda.current_stream(device=self.torch_device).synchronize()
        with wp.ScopedStream(self.stream):
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
            )
            newton.eval_fk(
                self.model,
                self.state_1.joint_q,
                self.state_1.joint_qd,
                self.state_1,
            )
        wp.synchronize_stream(self.stream)
        return self._observations()

    def _simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected actions {(self.num_envs, self.action_dim)}, received {tuple(actions.shape)}")
        actions = actions.to(self.torch_device).clamp(-1.0, 1.0)
        self.control_targets.copy_(self.default_targets)
        self.control_targets[:, self.controlled_targets] = self.default_joint_positions + ACTION_SCALE * actions

        torch.cuda.current_stream(device=self.torch_device).synchronize()
        with wp.ScopedStream(self.stream):
            wp.capture_launch(self.graph, stream=self.stream)
            newton.eval_ik(
                self.model,
                self.state_0,
                self.state_0.joint_q,
                self.state_0.joint_qd,
            )
        wp.synchronize_stream(self.stream)
        self.sim_time += self.frame_dt
        self.episode_steps += 1

        root_position = self.joint_q_0[:, :3]
        root_quaternion = torch.nn.functional.normalize(self.joint_q_0[:, 3:7], dim=-1)
        gravity = torch.zeros((self.num_envs, 3), device=self.torch_device)
        gravity[:, 2] = -1.0
        projected_gravity = quat_rotate_inverse(root_quaternion, gravity)
        upright = (-projected_gravity[:, 2]).clamp(0.0, 1.0)
        center_distance_sq = torch.sum(root_position[:, :2].square(), dim=-1)
        seeking_takeoff = ~self.jump_armed
        self.jump_cycle_base_height[seeking_takeoff] = torch.minimum(
            self.jump_cycle_base_height[seeking_takeoff],
            root_position[seeking_takeoff, 2],
        )
        jump_height = root_position[:, 2] - self.jump_cycle_base_height
        vertical_velocity = self.joint_qd_0[:, 2]
        joint_velocity = self.joint_qd_0.index_select(1, self.controlled_qd)
        action_rate = actions - self.previous_actions

        # Reward new height within each jump, rather than rewarding hovering high.
        # A jump only counts after crossing 10 cm and returning near the trampoline.
        bounded_height = jump_height.clamp(0.0, 0.35)
        peak_gain = (bounded_height - self.jump_cycle_peak).clamp_min(0.0)
        self.jump_cycle_peak = torch.maximum(self.jump_cycle_peak, bounded_height)
        self.jump_peak_height = torch.maximum(self.jump_peak_height, bounded_height)
        newly_reached_target = (~self.jump_armed) & (jump_height >= JUMP_TARGET_HEIGHT)
        self.jump_armed |= newly_reached_target
        completed_jump = self.jump_armed & (jump_height <= JUMP_LANDING_HEIGHT) & (vertical_velocity < 0.0)
        self.jump_counts += completed_jump.long()
        sustained_jump = completed_jump & (self.jump_counts >= 2)
        self.steps_since_jump += 1
        self.steps_since_jump[completed_jump] = 0
        self.jump_armed[completed_jump] = False
        self.jump_cycle_base_height[completed_jump] = root_position[completed_jump, 2]
        self.jump_cycle_peak[completed_jump] = 0.0

        phase_velocity = torch.where(
            self.jump_armed,
            (-vertical_velocity).clamp(0.0, 1.0),
            vertical_velocity.clamp(0.0, 1.0),
        )
        stalled = ((self.steps_since_jump.float() - CONTROL_FPS) / CONTROL_FPS).clamp(0.0, 1.0)
        jump_phase = 2.0 * math.pi * JUMP_FREQUENCY * self.frame_dt * self.episode_steps.float()
        desired_jump_height = 0.5 * JUMP_REFERENCE_HEIGHT * (1.0 - torch.cos(jump_phase))
        desired_vertical_velocity = math.pi * JUMP_FREQUENCY * JUMP_REFERENCE_HEIGHT * torch.sin(jump_phase)
        height_tracking_error = (jump_height - desired_jump_height).square()
        velocity_tracking_error = (vertical_velocity - desired_vertical_velocity).square()
        reward = (
            0.5 * upright.square()
            + 0.25 * torch.exp(-6.0 * center_distance_sq)
            - (1.0 - upright)
            - 0.5 * center_distance_sq
            - 100.0 * height_tracking_error
            - velocity_tracking_error
            + 10.0 * peak_gain / JUMP_TARGET_HEIGHT
            + 8.0 * newly_reached_target.float()
            + 20.0 * completed_jump.float()
            + 0.5 * phase_velocity
            - 0.5 * stalled
            - 2.0e-2 * self.joint_qd_0[:, :2].square().sum(-1)
            - 2.0e-3 * actions.square().sum(-1)
            - 2.0e-3 * action_rate.square().sum(-1)
            - 1.0e-4 * joint_velocity.square().sum(-1)
        )
        finite = torch.isfinite(self.joint_q_0).all(dim=-1) & torch.isfinite(self.joint_qd_0).all(dim=-1)
        terminated = (root_position[:, 2] < 1.25) | (upright < 0.35) | (center_distance_sq > 0.9**2) | ~finite
        timeout = self.episode_steps >= self.episode_length
        done = terminated | timeout
        reward = torch.where(terminated, reward - 100.0, reward)
        self.episode_returns += reward
        completed_returns = torch.where(done, self.episode_returns, torch.zeros_like(self.episode_returns))
        completed_lengths = torch.where(done, self.episode_steps, torch.zeros_like(self.episode_steps))
        self.last_metrics = {
            "base_height": root_position[:, 2].detach().clone(),
            "jump_height": jump_height.detach().clone(),
            "jump_reference_height": self.jump_cycle_base_height.detach().clone(),
            "jump_peak_height": self.jump_peak_height.detach().clone(),
            "jump_count": self.jump_counts.detach().clone(),
            "jump_completed": completed_jump.detach().clone(),
            "sustained_jump_completed": sustained_jump.detach().clone(),
            "above_jump_target": (jump_height >= JUMP_TARGET_HEIGHT).detach().clone(),
            "vertical_velocity": vertical_velocity.detach().clone(),
            "desired_jump_height": desired_jump_height.detach().clone(),
            "desired_vertical_velocity": desired_vertical_velocity.detach().clone(),
            "upright": upright.detach().clone(),
            "center_distance": torch.sqrt(center_distance_sq).detach().clone(),
        }
        info = {
            "terminated": terminated,
            "timeout": timeout,
            "episode_return": completed_returns,
            "episode_length": completed_lengths,
            **self.last_metrics,
        }
        self.previous_actions.copy_(actions)
        self.policy_previous_actions.zero_()
        self.policy_previous_actions[:, self.policy_action_indices] = actions
        if self.auto_reset and torch.any(done):
            self.reset(done)
        return self._observations(), reward, done, info

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        render_state = self.state_0
        world_offsets = getattr(self.viewer, "world_offsets", None)
        if self.render_state is not None and world_offsets is not None:
            wp.copy(self.render_state.body_q, self.state_0.body_q)
            wp.launch(
                apply_particle_world_offsets,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.particle_q,
                    self.model.particle_world,
                    world_offsets,
                ],
                outputs=[self.render_state.particle_q],
                device=self.device,
            )
            render_state = self.render_state
        self.viewer.log_state(render_state)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def close(self) -> None:
        self.viewer.close()


class StandaloneExample:
    """Viewer example with an unactuated G1 on the deformable trampoline."""

    def __init__(self, viewer: Any, args: Any) -> None:
        self.viewer = viewer
        self.env = TrampolineEnv(
            num_envs=args.num_envs,
            device=str(wp.get_device()),
            viewer=viewer,
            episode_length=max(args.num_frames, 1),
            seed=args.seed,
            sim_substeps=args.sim_substeps,
            contact_stiffness=args.contact_stiffness,
            contact_damping=args.contact_damping,
            shape_friction=args.shape_friction,
            soft_contact_friction=args.soft_contact_friction,
            particle_radius=args.particle_radius,
            foot_contact_margin=args.foot_contact_margin,
            spawn_height_offset=args.spawn_height_offset,
            enable_joint_drives=False,
            auto_reset=False,
        )
        self.frame_dt = self.env.frame_dt
        self.sim_time = 0.0
        self.actions = torch.zeros(
            (self.env.num_envs, self.env.action_dim),
            device=self.env.torch_device,
        )

    def step(self) -> None:
        self.env.step(self.actions)
        self.sim_time = self.env.sim_time

    def render(self) -> None:
        self.env.render()

    def test_final(self) -> None:
        if not torch.isfinite(self.env.joint_q_0).all():
            raise ValueError("Standalone trampoline simulation became non-finite")


def create_standalone_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sim-substeps", type=int, default=8)
    parser.add_argument("--contact-stiffness", type=float, default=1.0e5)
    parser.add_argument("--contact-damping", type=float, default=1.0e3)
    parser.add_argument("--shape-friction", type=float, default=0.75)
    parser.add_argument("--soft-contact-friction", type=float, default=0.6)
    parser.add_argument("--particle-radius", type=float, default=0.025)
    parser.add_argument("--foot-contact-margin", type=float, default=0.005)
    parser.add_argument("--spawn-height-offset", type=float, default=0.06)
    parser.set_defaults(num_frames=600)
    return parser


if __name__ == "__main__":
    standalone_viewer, standalone_args = newton.examples.init(create_standalone_parser())
    newton.examples.run(
        StandaloneExample(standalone_viewer, standalone_args),
        standalone_args,
    )
