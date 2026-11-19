# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example walking-policy robot interacting with a falling cloth
#
# Shows how to control robots pretrained in IsaacLab with RL. Policies are
# loaded from ONNX files and run with Warp-NN's Warp-backed runtime, so PyTorch
# is not required for policy inference.
# The policy observes robot state and movement commands. An unpinned sheet is
# released above the robot to demonstrate coupled cloth and articulated-body contact.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples cha_robot_cloth_policy --robot g1_29dof
# python -m newton.examples cha_robot_cloth_policy --robot g1_23dof
# python -m newton.examples cha_robot_cloth_policy --robot go2
# python -m newton.examples cha_robot_cloth_policy --robot anymal
# python -m newton.examples cha_robot_cloth_policy --robot anymal --physx
###########################################################################

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import warp as wp
import yaml
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str


ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.onnx", "physx": "rl_policies/physx_anymal.onnx"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.onnx", "physx": "rl_policies/physx_go2.onnx"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.onnx"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.onnx", "physx": "rl_policies/physx_g1_23DOF.onnx"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_pos_initial: wp.array[float],
    physx_to_mjc_idx: wp.array[int],
    gravity_w: wp.vec3,
    command: wp.vec3,
    prev_act: wp.array2d[float],
    num_dofs: int,
    obs: wp.array2d[float],
):
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])

    lin_w = wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2])
    ang_w = wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5])

    vel_b = wp.quat_rotate_inv(q, lin_w)
    avel_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, gravity_w)

    obs[0, 0] = vel_b[0]
    obs[0, 1] = vel_b[1]
    obs[0, 2] = vel_b[2]
    obs[0, 3] = avel_b[0]
    obs[0, 4] = avel_b[1]
    obs[0, 5] = avel_b[2]
    obs[0, 6] = grav_b[0]
    obs[0, 7] = grav_b[1]
    obs[0, 8] = grav_b[2]
    obs[0, 9] = command[0]
    obs[0, 10] = command[1]
    obs[0, 11] = command[2]

    for k in range(num_dofs):
        idx = physx_to_mjc_idx[k]
        obs[0, 12 + k] = joint_q[7 + idx] - joint_pos_initial[idx]
        obs[0, 12 + num_dofs + k] = joint_qd[6 + idx]
        obs[0, 12 + 2 * num_dofs + k] = prev_act[0, k]


@wp.kernel
def _build_joint_target_q_kernel(
    act: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    reorder: wp.array[int],
    action_scale: float,
    target_indices: wp.array[int],
    out: wp.array[float],
):
    i = wp.tid()
    idx = reorder[i]
    out[target_indices[i]] = joint_pos_initial[i] + action_scale * act[0, idx]


def policy_target_indices(model, num_dofs: int, joint_pos_slice: slice):
    """Map policy coordinates to the model's position-target layout."""
    kinds = model.joint_type.numpy()
    coordinates = model.joint_q_start.numpy()
    joints = np.flatnonzero((kinds == newton.JointType.REVOLUTE) | (kinds == newton.JointType.PRISMATIC))
    joints = joints[np.argsort(coordinates[joints])]
    expected = np.arange(model.joint_coord_count)[joint_pos_slice]
    if len(expected) != num_dofs or not np.array_equal(coordinates[joints], expected):
        raise ValueError("Robot policy positions must correspond to the robot's scalar joints in coordinate order.")
    # A free base occupies seven position coordinates but only six velocity DOFs.
    return model.joint_target_q_start.numpy()[joints].astype(np.int32)


def place_cloth_above_robot(model, particle_count: int, clearance: float = 0.05):
    """Translate the sheet above the posed robot's visible and collision geometry."""
    bodies = model.body_q.numpy()
    parents = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    lower, upper = model.shape_collision_aabb_lower.numpy(), model.shape_collision_aabb_upper.numpy()
    margins, flags = model.shape_margin.numpy(), model.shape_flags.numpy()
    corners = np.asarray(list(product((False, True), repeat=3)))
    bounds = []
    for shape, body in enumerate(parents):
        if body < 0 or flags[shape] == 0:
            continue
        transform = wp.transform_multiply(wp.transform(*bodies[body]), wp.transform(*transforms[shape]))
        local = np.where(corners, upper[shape], lower[shape])
        world = np.asarray([wp.transform_point(transform, wp.vec3(*point)) for point in local])
        margin = max(float(margins[shape]), 0.0)
        bounds.append((world.min(axis=0) - margin, world.max(axis=0) + margin))
    if not bounds:
        raise ValueError("The falling-cloth example needs bounded robot geometry")
    robot_lower = np.min([bound[0] for bound in bounds], axis=0)
    robot_upper = np.max([bound[1] for bound in bounds], axis=0)
    points = model.particle_q.numpy()
    sheet = points[:particle_count]
    radius = float(model.particle_radius.numpy()[:particle_count].max())
    root = int(model.joint_child.numpy()[0])
    center = 0.5 * (sheet.min(axis=0) + sheet.max(axis=0))
    # Initial joint FK must run first: body origins do not bound the head, shoulders, or offset visual meshes.
    shift = np.array(
        [
            bodies[root, 0] - center[0],
            bodies[root, 1] - center[1],
            robot_upper[2] + radius + clearance - sheet[:, 2].min(),
        ]
    )
    points[:particle_count] += shift
    model.particle_q.assign(points)
    return robot_lower, robot_upper


def add_cloth_collision_shells(builder):
    """Cover visual shells with closed cloth colliders without changing articulated mass or rigid contacts."""
    original_count = builder.shape_count
    shells, covered_bodies = [], set()
    cfg = builder.default_shape_cfg.copy()
    cfg.density = 0.0
    cfg.is_visible = False
    cfg.has_shape_collision = False
    cfg.has_particle_collision = True
    cfg.configure_sdf(force_sdf=True)
    for shape in range(original_count):
        body = builder.shape_body[shape]
        source = builder.shape_source[shape]
        if body < 0 or not (builder.shape_flags[shape] & newton.ShapeFlags.VISIBLE):
            continue
        if not isinstance(source, newton.Mesh):
            continue
        # Imported rigid proxies can sit centimetres inside the render mesh. A closed hull also
        # seals mechanical openings where a sheet should ride over the outside of the shell.
        # Keep every hull vertex: a vertex budget can cut into the visible shoulder and elbow.
        mesh = newton.Mesh(
            np.asarray(source.vertices) * np.asarray(builder.shape_scale[shape]),
            source.indices,
            compute_inertia=False,
            maxhullvert=0,
        ).compute_convex_hull()
        shells.append(
            builder.add_shape_mesh(
                body,
                xform=builder.shape_transform[shape],
                mesh=mesh,
                cfg=cfg,
                label=builder.shape_label[shape] + "_cloth_shell",
            )
        )
        covered_bodies.add(body)
    for shape in range(original_count):
        if builder.shape_body[shape] in covered_bodies:
            # Avoid duplicate forces from the old coarse proxies, while retaining all rigid contacts.
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        elif builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES:
            source = builder.shape_source[shape]
            if isinstance(source, newton.Mesh):
                builder.shape_force_sdf[shape] = True
    return shells


def load_policy_and_setup_arrays(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load ONNX policy and setup device buffers for the policy step."""
    print("[INFO] Loading policy from:", policy_path)
    example.policy = OnnxRuntime(policy_path, device=example.device)
    example.policy_input_name = example.policy.inputs[0].name
    example.policy_output_name = example.policy.outputs[0].name

    if example.state_0.joint_q is not None:
        example._joint_pos_initial_wp = wp.clone(example.state_0.joint_q[joint_pos_slice])
    else:
        example._joint_pos_initial_wp = wp.zeros(num_dofs, dtype=wp.float32, device=example.device)

    expected = 12 + 3 * num_dofs
    obs_dim = int(example.config["num_observations"]) if "num_observations" in example.config else expected
    if obs_dim != expected:
        raise ValueError(
            f"load_policy_and_setup_arrays: config num_observations={obs_dim} does not match the expected "
            f"layout (12 + 3*num_dofs = {expected})"
        )
    validate_policy_io_shapes(
        policy_path,
        example.policy_input_name,
        example.policy_output_name,
        obs_width=obs_dim,
        action_width=num_dofs,
        context="load_policy_and_setup_arrays",
    )
    example._obs_wp = wp.zeros((1, obs_dim), dtype=wp.float32, device=example.device)
    example._prev_act_wp = wp.zeros((1, num_dofs), dtype=wp.float32, device=example.device)

    example._physx_to_mjc_wp = wp.array(
        np.asarray(example.physx_to_mjc_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._mjc_to_physx_wp = wp.array(
        np.asarray(example.mjc_to_physx_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._num_dofs = num_dofs
    example._joint_target_indices_wp = wp.array(
        policy_target_indices(example.model, num_dofs, joint_pos_slice), dtype=wp.int32, device=example.device
    )


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(self, viewer, args):
        if args.robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot: {args.robot}. Available: {list(ROBOT_CONFIGS.keys())}")
        robot_config = ROBOT_CONFIGS[args.robot]
        print(f"[INFO] Selected robot: {args.robot}")

        asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
        print(f"[INFO] Asset directory: {asset_directory}")

        yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        num_dofs = config["num_dofs"]
        print(f"[INFO] Loaded config with {num_dofs} DOFs")

        mjc_to_physx = list(range(num_dofs))
        physx_to_mjc = list(range(num_dofs))

        if args.physx:
            if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
                physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
                raise ValueError(
                    f"PhysX policy not available for robot '{args.robot}'. Robots with PhysX support: {physx_robots}"
                )
            policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(
                config["mjw_joint_names"], config["physx_joint_names"]
            )
            if len(mjc_to_physx) != num_dofs or len(physx_to_mjc) != num_dofs:
                missing_mjw = sorted(set(config["physx_joint_names"]) - set(config["mjw_joint_names"]))
                missing_physx = sorted(set(config["mjw_joint_names"]) - set(config["physx_joint_names"]))
                raise ValueError(
                    "PhysX/MJWarp joint mapping is incomplete: "
                    f"expected {num_dofs} DOFs, got {len(mjc_to_physx)} MJWarp-to-PhysX and "
                    f"{len(physx_to_mjc)} PhysX-to-MJWarp entries. "
                    f"Missing from MJWarp: {missing_mjw}; missing from PhysX: {missing_physx}"
                )
        else:
            policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

        # Match the bundled walking policy: 200 Hz physics and one action every four physics ticks.
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.solver_type = args.solver

        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config

        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        else:
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.rigid_gap = 0.0

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            # Synreal creates affine proxies for dynamic bodies. Merge fixed
            # frame links so they do not add redundant proxy constraints.
            collapse_fixed_joints=self.solver_type == "synreal",
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")
        if self.solver_type == "synreal":
            self.cloth_collision_shells = add_cloth_collision_shells(builder)

        builder.add_ground_plane()

        self.cloth_particle_count = 0
        if self.solver_type == "synreal":
            cloth_dim_x = 64
            cloth_dim_y = 32
            cloth_total_mass = 0.5
            # Author the sheet metric here, then translate it above the robot after evaluating its initial joint pose.
            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
                vel=wp.vec3(0.0),
                dim_x=cloth_dim_x,
                dim_y=cloth_dim_y,
                cell_x=0.015,
                cell_y=0.015,
                mass=cloth_total_mass / ((cloth_dim_x + 1) * (cloth_dim_y + 1)),
                fix_left=False,
                particle_radius=0.005,
                tri_aniso_ke=wp.vec3(1.0e3, 1.0e3, 1.0e2),
                edge_aniso_ke=wp.vec3(2.0e-6, 1.0e-6, 5.0e-6),
            )
            self.cloth_particle_count = builder.particle_count

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 2.0e1
        self.model.soft_contact_mu = 0.25

        # Synreal initializes affine proxies from model.body_q rather than an
        # articulation state, so populate the authored pose before construction.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        if self.cloth_particle_count:
            self.initial_robot_bounds = place_cloth_above_robot(self.model, self.cloth_particle_count)
            cloth_bottom = self.model.particle_q.numpy()[: self.cloth_particle_count, 2].min()
            print(
                f"[INFO] Cloth released above robot: bottom={cloth_bottom:.3f} m, "
                f"robot top={self.initial_robot_bounds[1][2]:.3f} m, surface clearance=0.050 m"
            )

        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=self.use_mujoco,
                solver="newton",
                nconmax=30,
                njmax=100,
            )
        else:
            self.solver = s3d.SolverCoupledHighAccuracy(self.model)
            # Resolve the articulated constraints before feeding joint velocities back into the policy.
            # A ten-iteration linear solve leaves the small finger links with unstable angular motion.
            self.solver.config.constraint_alpha = 200
            self.solver.config.joint_constraint_primal_iterations = 3
            self.solver.config.nonlinear_iterations = 5
            self.solver.config.linear_iterations = 50
            self.solver.config.enable_collision_hessian_non_diagonal = True

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.solver_type == "mujoco":
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.006
            )
            self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)
        self._initial_particle_q = wp.clone(self.state_0.particle_q)
        self._initial_particle_qd = wp.clone(self.state_0.particle_qd)

        self.physx_to_mjc_indices = np.asarray(physx_to_mjc, dtype=np.int64)
        self.mjc_to_physx_indices = np.asarray(mjc_to_physx, dtype=np.int64)
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self._command = wp.vec3(0.0, 0.0, 0.0)
        self._reset_key_prev = False

        self.policy = None
        self.policy_input_name = None
        self.policy_output_name = None
        self._joint_pos_initial_wp = None
        self._obs_wp = None
        self._prev_act_wp = None
        self._physx_to_mjc_wp = None
        self._mjc_to_physx_wp = None
        self._num_dofs = None

        load_policy_and_setup_arrays(self, policy_path, config["num_dofs"], slice(7, None))

        self.capture()

    def capture(self):
        self.graph = None
        self.use_graph = False
        if self.device.is_cpu or self.device.is_mempool_enabled:
            print("[INFO] Using graph capture")
            self.use_graph = True
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        need_state_copy = self.use_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            if self.solver_type == "synreal":
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

        if self.solver_type == "mujoco":
            self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_0.particle_q, self._initial_particle_q)
        wp.copy(self.state_0.particle_qd, self._initial_particle_qd)
        wp.copy(self.state_1.particle_q, self._initial_particle_q)
        wp.copy(self.state_1.particle_qd, self._initial_particle_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        if self.solver_type == "synreal":
            self.solver.sync_affine_body_state(self.state_0)
            self.solver.sync_affine_body_state(self.state_1)
        if self._prev_act_wp is not None:
            self._prev_act_wp.zero_()
        self.sim_time = 0.0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self._command = wp.vec3(float(fwd), float(lat), float(rot))
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        if self.solver_type == "synreal":
            # The policy consumes generalized coordinates, while Synreal advances
            # affine maximal state. Recover the current joints before observation.
            newton.eval_ik(
                self.model,
                self.state_0,
                self.state_0.joint_q,
                self.state_0.joint_qd,
            )

        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._physx_to_mjc_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._num_dofs,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self.policy_input_name: self._obs_wp})
        act_wp = out[self.policy_output_name]

        wp.launch(
            _build_joint_target_q_kernel,
            dim=self._num_dofs,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mjc_to_physx_wp,
                float(self.config["action_scale"]),
                self._joint_target_indices_wp,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

        wp.copy(self._prev_act_wp, act_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.cycle_time

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        # Finite, positive heights alone also accept a robot that has been launched tens of metres away.
        bodies = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        if not np.isfinite(bodies).all() or not np.isfinite(velocities).all():
            raise ValueError("Robot state contains non-finite values")
        if bodies[:, 2].min() < -0.05 or bodies[:, 2].max() > 3.0:
            raise ValueError("Robot body heights left the supported ground-level workspace")
        if np.linalg.norm(velocities[:, :3], axis=1).max() > 30.0:
            raise ValueError("Robot body speed exceeds 30 m/s")
        if np.linalg.norm(velocities[:, 3:], axis=1).max() > 300.0:
            raise ValueError("Robot angular speed exceeds 300 rad/s")
        if self.cloth_particle_count:
            points = self.state_0.particle_q.numpy()[: self.cloth_particle_count]
            speeds = self.state_0.particle_qd.numpy()[: self.cloth_particle_count]
            if not np.isfinite(points).all() or not np.isfinite(speeds).all():
                raise ValueError("Cloth state contains non-finite values")
            if np.linalg.norm(speeds, axis=1).max() > 100.0:
                raise ValueError("Cloth particle speed exceeds 100 m/s")

    def test_final(self):
        self.test_post_step()
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--robot",
            type=str,
            default="g1_29dof",
            choices=list(ROBOT_CONFIGS.keys()),
            help="Robot name to load",
        )
        parser.add_argument(
            "--physx",
            action="store_true",
            help="Run physX policy instead of MJWarp.",
        )
        parser.add_argument(
            "--solver",
            type=str,
            default="synreal",
            choices=["synreal", "mujoco"],
            help="Solver type: synreal (default) or mujoco.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
