"""Run the official Microduck walking policy with SynReal or MuJoCo.

python -m newton.examples cha_robot_microduck --solver synreal --policy walk
python -m newton.examples cha_robot_microduck --solver mujoco --policy walk
"""

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.coupled_high_accuracy.microduck_assets import ASSET_DIR, MODEL_FILENAME, download_assets


class WalkPolicy:
    """Official 61-D observation contract, with normalization embedded in ONNX."""

    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "Microduck requires ONNX Runtime for policy inference. Install it in this interpreter with: "
                f'uv pip install --python "{sys.executable}" "onnxruntime>=1.20"'
            ) from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 61] or len(outputs) != 1 or outputs[0].shape != [1, 14]:
            raise ValueError("Microduck walk requires an ONNX policy with shape [1,61] -> [1,14]")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.joint_names = metadata["joint_names"].split(",")
        self.default_pose = np.fromstring(metadata["default_joint_pos"], sep=",", dtype=np.float32)
        self.action_scale = float(metadata["action_scale"])
        expected_observations = (
            "base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command,head_command,body_command"
        )
        if metadata.get("observation_names") != expected_observations:
            raise ValueError("Unsupported Microduck observation layout")
        if len(self.joint_names) != 14 or len(set(self.joint_names)) != 14 or self.default_pose.shape != (14,):
            raise ValueError("Invalid Microduck joint metadata")
        self.previous_action = np.zeros(14, dtype=np.float32)
        self.observation = np.zeros((1, 61), dtype=np.float32)

    def infer(self, quaternion, angular_velocity, positions, velocities, command):
        orientation = wp.quat(*quaternion)
        self.observation[0, :3] = wp.quat_rotate_inv(orientation, wp.vec3(*angular_velocity))
        self.observation[0, 3:6] = wp.quat_rotate_inv(orientation, wp.vec3(0.0, 0.0, -1.0))
        self.observation[0, 6:20] = positions - self.default_pose
        self.observation[0, 20:34] = velocities
        self.observation[0, 34:48] = self.previous_action
        self.observation[0, 48:51] = command
        # Neutral head/body commands occupy the final ten entries, in the training order.
        if not np.isfinite(self.observation).all():
            raise FloatingPointError("Microduck observations contain non-finite simulation state")
        action = self.session.run([self.output_name], {self.input_name: self.observation})[0][0]
        if not np.isfinite(action).all():
            raise FloatingPointError("Microduck policy produced non-finite actions")
        self.previous_action[:] = action
        return self.default_pose + self.action_scale * action

    def reset(self):
        self.previous_action.fill(0.0)
        self.observation.fill(0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.solver_type = args.solver
        self.device = wp.get_device()
        self.frame_dt = 0.01
        # The small feet and stiff contacts need a finer physics tick than the 100 Hz policy tick.
        self.sim_substeps = 12
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.command = np.asarray(args.command, dtype=np.float32)
        self._reset_key_prev = False
        self.graph = None
        if self.solver_type == "synreal" and not self.device.is_cuda:
            raise ValueError("SynReal requires CUDA; use --solver mujoco --device cpu for CPU simulation")

        asset_dir = Path(args.asset_dir)
        if not all((asset_dir / name).is_file() for name in ("microduck.urdf", MODEL_FILENAME, "policies/walk.onnx")):
            download_assets(asset_dir)
        self.policy = WalkPolicy(asset_dir / "policies/walk.onnx")

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.solver_type == "mujoco":
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=100.0, limit_kd=1.0)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.kf = 1000.0
        builder.default_shape_cfg.mu = 0.8
        # The builder's 10 cm speculative range is comparable to this robot's height.
        # Keep contact searches local to the touching geometry at this scale.
        builder.rigid_gap = 0.0
        builder.add_urdf(
            str(asset_dir / "microduck.urdf"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            joint_damping_mode="passive",
        )
        joint_ids = {name.rsplit("/", 1)[-1]: index for index, name in enumerate(builder.joint_label)}
        missing = set(self.policy.joint_names) - joint_ids.keys()
        if missing:
            raise ValueError(f"Microduck URDF is missing policy joints: {sorted(missing)}")
        self.q_indices = np.array([builder.joint_q_start[joint_ids[name]] for name in self.policy.joint_names])
        self.qd_indices = np.array([builder.joint_qd_start[joint_ids[name]] for name in self.policy.joint_names])
        policy_joint_ids = [joint_ids[name] for name in self.policy.joint_names]
        self.root_body = next(i for i, name in enumerate(builder.body_label) if name.rsplit("/", 1)[-1] == "trunk_base")
        for index, q_index, qd_index in zip(range(14), self.q_indices, self.qd_indices, strict=True):
            builder.joint_q[q_index] = float(self.policy.default_pose[index])
            # Use the upstream XML's identified XL330 approximation. The ONNX stiffness metadata
            # describes a BAM wrapper (unit gain), not the physical servo's PD stiffness.
            builder.joint_target_ke[qd_index] = 0.55
            builder.joint_target_kd[qd_index] = 0.0
            builder.joint_damping[qd_index] = 0.053
            builder.joint_armature[qd_index] = 0.0018
            builder.joint_effort_limit[qd_index] = 0.96
            builder.joint_target_mode[qd_index] = int(newton.JointTargetMode.POSITION)
        builder.approximate_meshes("convex_hull")
        builder.add_ground_plane()
        self.model = builder.finalize()
        # SynReal constructs affine proxies from model.body_q, so initialize that pose first.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        if self.solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=self.device.is_cpu,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                iterations=50,
                # A fallen robot can contact the ground through many body meshes at once.
                njmax=512,
                nconmax=256,
            )
        else:
            from newton.solvers import coupled_high_accuracy as cha

            self.solver = cha.SolverCoupledHighAccuracy(self.model)
            config = self.solver.config
            config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
            config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
            config.nonlinear_solver = cha.NonlinearSolver.NEWTON
            config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
            config.joint_constraint_primal_iterations = 3
            config.constraint_alpha = 200.0
            config.nonlinear_iterations = 5
            config.linear_iterations = 50
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.initial_state = self.model.state()
        self.initial_state.assign(self.state_0)
        self.target_indices = self.model.joint_target_q_start.numpy()[policy_joint_ids]
        self.targets = np.zeros(len(self.control.joint_target_q), dtype=np.float32)
        self.targets[self.target_indices] = self.policy.default_pose
        self.control.joint_target_q.assign(self.targets)
        if self.solver_type == "mujoco":
            self.contacts = None
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()
        viewer.set_model(self.model)
        if not args.disable_cuda_graph and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self.reset()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.solver_type == "synreal":
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self):
        # State.assign also restores SynReal's authoritative affine proxy particles.
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.policy.reset()
        self.targets.fill(0.0)
        self.targets[self.target_indices] = self.policy.default_pose
        self.control.joint_target_q.assign(self.targets)
        self.sim_time = 0.0

    def step(self):
        command = self.command.copy()
        if hasattr(self.viewer, "is_key_down"):
            key = self.viewer.is_key_down
            command += np.array([0.5 * (key("i") - key("k")), 0.2 * (key("j") - key("l")), 0.5 * (key("u") - key("o"))])
            reset_down = bool(key("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down
        if self.solver_type == "synreal":
            # Generalized coordinates are derived from the maximal-body simulation for policy feedback.
            newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        q, qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
        root_pose = self.state_0.body_q.numpy()[self.root_body]
        root_velocity = self.state_0.body_qd.numpy()[self.root_body]
        target = self.policy.infer(root_pose[3:7], root_velocity[3:6], q[self.q_indices], qd[self.qd_indices], command)
        self.targets[self.target_indices] = target
        self.control.joint_target_q.assign(self.targets)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(self.state_0.body_qd.numpy()).all():
            raise AssertionError("Microduck simulation produced non-finite state")
        if poses[self.root_body, 2] < 0.07:
            raise AssertionError("Microduck fell below its standing height")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1000)
        parser.add_argument("--solver", choices=("synreal", "mujoco"), default="synreal")
        parser.add_argument("--policy", choices=("walk",), default="walk")
        parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
        parser.add_argument(
            "--disable-cuda-graph", action="store_true", help="Run simulation steps without CUDA capture"
        )
        parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    try:
        newton.examples.run(Example(viewer, args), args)
    except BaseException:
        viewer.close()
        raise
