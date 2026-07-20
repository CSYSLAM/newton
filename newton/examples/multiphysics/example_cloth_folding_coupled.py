# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Franka T-shirt folding with MuJoCo/VBD proxy coupling (V8 edge-following pinch grasp).
#
# Robot/articulation: SolverMuJoCo
# Garment particles:  SolverVBD
# Bidirectional robot-garment interaction: SolverCoupledProxy
# Motion generation: Newton GPU IK + an open-loop folding keyframe sequence
#
# Install this file as:
#   newton/examples/multiphysics/example_cloth_folding_coupled.py
#
# Run from the Newton repository root:
#   uv run --extra examples -m newton.examples cloth_folding_coupled
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy


# The original Newton shirt asset and folding trajectory are authored in cm.
# This example converts both to SI units while leaving the Franka URDF at scale=1.
CM_TO_M = 0.01
FRANKA_BASE_POS = wp.vec3(-0.50, -0.50, 0.0)
TABLE_POS = wp.vec3(0.0, -0.50, 0.10)
TABLE_HALF_EXTENTS = (0.40, 0.40, 0.10)

GRIP_OPEN = 0.032
DEFAULT_GRIP_CLOSE = 0.004

# Use the home pose from Newton's cloth_franka example, converted to the
# current 7-arm-coordinate + 2-finger-coordinate FR3 URDF layout.  The
# previously borrowed cable-manipulation pose placed the hand inside the shirt.
FRANKA_Q = [
    0.0,
    0.0,
    0.0,
    -1.59695,
    0.0,
    2.5307,
    0.0,
    GRIP_OPEN,
    GRIP_OPEN,
]

# Keep the authored shirt transform exactly consistent with cloth_franka.
# Do not lower the mesh by its minimum z: the asset is a two-sided garment,
# and that heuristic can place the hand/table inside its initial volume.
SHIRT_POS = wp.vec3(0.0, 0.70, 0.30)
GRIP_FORCE = 1000.0
GRIP_STIFFNESS = 1000.0
GRIP_DAMPING = 100.0


@wp.kernel
def set_gripper_q(
    joint_q: wp.array2d[float],
    finger_pos: wp.array[float],
    idx0: int,
    idx1: int,
):
    world_idx = wp.tid()
    joint_q[world_idx, idx0] = finger_pos[world_idx]
    joint_q[world_idx, idx1] = finger_pos[world_idx]


@wp.kernel
def set_task_targets(
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    finger_pos: wp.array[float],
    pos: wp.vec3,
    rot: wp.vec4,
    grip_width: float,
):
    world_idx = wp.tid()
    target_positions[world_idx] = pos
    target_rotations[world_idx] = rot
    finger_pos[world_idx] = grip_width


def find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


def normalized_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float32).copy()
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        raise ValueError("Encountered a zero-length task-space quaternion")
    q /= norm
    return q


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_graph = bool(args.graph_capture)
        self.motion_speed = max(float(args.motion_speed), 1.0e-3)
        self.grip_close = float(args.grip_close)

        # VBD uses the radius as the actual self-contact distance and the margin
        # as the broader candidate-generation distance. The latter must never be
        # smaller than the radius. When omitted, choose Newton's recommended
        # lower bound of 1.5x the radius.
        self.self_contact_radius = float(args.self_contact_radius)
        self.self_contact_margin = (
            1.5 * self.self_contact_radius
            if args.self_contact_margin is None
            else float(args.self_contact_margin)
        )
        if bool(args.cloth_self_contact) and self.self_contact_margin < self.self_contact_radius:
            raise ValueError(
                "--self-contact-margin must be >= --self-contact-radius. "
                "A value between 1.5x and 2.0x the radius is recommended; for example, "
                f"use --self-contact-margin {1.5 * self.self_contact_radius:.6g}."
            )

        self._build_scene()
        self.use_graph = self.use_graph and self.device.is_cuda
        print(
            "[cloth_folding_coupled] "
            f"substeps={self.sim_substeps}, vbd_iterations={args.vbd_iterations}, "
            f"proxy_iterations={args.proxy_iterations}, self_contact_interval={args.particle_collision_detection_interval}, "
            f"graph_capture={self.use_graph}, settle_time={args.settle_time}s"
        )
        if self.device.is_cuda and not self.use_graph:
            print(
                "[cloth_folding_coupled] WARNING: CUDA graph capture is disabled. "
                "This cloth scene can become tens of times slower; use --no-graph-capture only for debugging."
            )

        self.control = self.model.control()
        self._build_solver(args)
        self._build_ik()
        print(
            "[cloth_folding_coupled] grasp setup: "
            f"ik_frame={self.ik_frame_label}, finger_bodies="
            f"{[self.model.body_label[i] for i in self.gripper_bodies]}, "
            f"open={GRIP_OPEN:.3f} m/finger, close={self.grip_close:.3f} m/finger, "
            f"cloth_radius={self.args.cloth_radius:.3f} m, contact_margin={self.args.contact_margin:.3f} m, "
            f"finger_mu={self.args.contact_mu:.2f}"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=float(args.contact_margin),
            contact_matching="latest",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        newton.examples.configure_coupled_view(self, args)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(0.90, -1.35, 1.05), pitch=-32.0, yaw=122.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, -0.48, 0.22))

        self.capture()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    @staticmethod
    def _add_franka(builder: newton.ModelBuilder) -> None:
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(FRANKA_BASE_POS, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = float(self.args.contact_gap)

        # Solver-specific metadata must be registered before finalizing.
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        # Franka robot, owned by MuJoCo.
        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count
        self._add_franka(builder)

        # Use the same MuJoCo position-drive tuning as Newton's current
        # coupled Franka examples.  The previous V3 values came from a different
        # IK example and were unnecessarily stiff for the proxy-coupled solve.
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS, GRIP_STIFFNESS]
        builder.joint_target_kd[7:9] = [GRIP_DAMPING, GRIP_DAMPING]
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [GRIP_FORCE, GRIP_FORCE]
        builder.joint_armature[:7] = [0.05] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))
        self.franka_shapes = list(range(franka_shape_start, builder.shape_count))

        # The current coupled Franka examples use body-level MuJoCo gravity
        # compensation.  Do not also enable actuator gravity compensation here:
        # applying both can double-compensate the same arm load.
        body_gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if body_gravcomp.values is None:
            body_gravcomp.values = {}
        for body in self.franka_bodies:
            body_gravcomp.values[body] = 1.0

        # Couple only the two finger links into VBD.  Including fr3_hand makes
        # the palm/hand housing push the cloth before the fingertips can form a
        # pinch, which is especially harmful for the edge-grasp trajectory used
        # by cloth_franka.
        self.gripper_bodies = [
            body for body in self.franka_bodies if "finger" in builder.body_label[body]
        ]
        if len(self.gripper_bodies) != 2:
            labels = [builder.body_label[body] for body in self.franka_bodies]
            raise RuntimeError(
                "Expected exactly two Franka finger bodies for proxy coupling; "
                f"found {len(self.gripper_bodies)} in {labels}"
            )

        # Static table and floor are shared environment geometry.
        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=float(self.args.contact_ke),
            kd=float(self.args.contact_kd),
            mu=float(self.args.contact_mu),
            margin=0.0,
            gap=float(self.args.contact_gap),
        )
        self.table_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            label="folding_table",
        )
        self.ground_shape = builder.add_ground_plane(cfg=table_cfg, label="ground")

        # T-shirt mesh, owned by VBD as particles/triangles.
        self.shirt_particle_start = builder.particle_count
        self._add_shirt(builder)
        self.shirt_particle_end = builder.particle_count
        if self.shirt_particle_end <= self.shirt_particle_start:
            raise RuntimeError("The shirt asset produced no particles")

        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.device = self.model.device

        # This is essential for the unisex_shirt asset.  Its authored mesh has
        # non-zero dihedral angles; keeping them as bending rest angles makes the
        # garment try to preserve a puffy 3-D shape instead of settling flat.
        if self.model.edge_rest_angle is not None:
            self.model.edge_rest_angle.zero_()

        # Assign high friction only to the two fingertips.  The table should be
        # less sticky so the grabbed patch can slide while the arm folds it.
        shape_body = self.model.shape_body.numpy()
        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        finger_bodies = set(self.gripper_bodies)
        static_shapes = {int(self.table_shape), int(self.ground_shape)}
        for shape_idx in range(self.model.shape_count):
            body_idx = int(shape_body[shape_idx])
            if body_idx in finger_bodies:
                shape_ke[shape_idx] = float(self.args.contact_ke)
                shape_kd[shape_idx] = float(self.args.contact_kd)
                shape_mu[shape_idx] = float(self.args.contact_mu)
            elif shape_idx in static_shapes:
                shape_ke[shape_idx] = float(self.args.contact_ke)
                shape_kd[shape_idx] = float(self.args.contact_kd)
                shape_mu[shape_idx] = float(self.args.table_mu)
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.device
        )

        # VBD's soft_contact_mu is also used by cloth self-contact, so keep it
        # separate from the much higher fingertip friction.
        self.model.soft_contact_ke = float(self.args.contact_ke)
        self.model.soft_contact_kd = float(self.args.contact_kd)
        self.model.soft_contact_mu = float(self.args.self_contact_mu)

        particle_q = self.model.particle_q.numpy()[self.shirt_particle_start : self.shirt_particle_end]
        initial_span = np.ptp(particle_q[:, :2], axis=0)
        self.initial_planar_aabb_area = float(max(initial_span[0] * initial_span[1], 1.0e-8))
        initial_min = np.min(particle_q, axis=0)
        initial_max = np.max(particle_q, axis=0)
        print(
            "[cloth_folding_coupled] initial shirt AABB "
            f"min={initial_min.tolist()} max={initial_max.tolist()} "
            f"franka_q={FRANKA_Q[:7]}"
        )

        # Track a small material patch around the first authored grasp point.
        # --first-grasp-only turns this into an automated pass/fail regression:
        # the tracked patch must finish clearly above the table, not merely be
        # pushed sideways by the gripper.
        # The first pinch closes while translating from outside the hem toward
        # the measured free boundary (approximately y=-0.56 m). Track the
        # material patch at the final pinch center, not the outer approach pose.
        first_grasp_xy = np.array([0.32, -0.56], dtype=np.float32)
        d2 = np.sum((particle_q[:, :2] - first_grasp_xy[None, :]) ** 2, axis=1)
        local_ids = np.argsort(d2)[:64]
        self.first_grasp_particle_ids = local_ids + self.shirt_particle_start
        self.first_grasp_initial_p75_z = float(np.percentile(particle_q[local_ids, 2], 75.0))

        self._build_keyframes()

    def _add_shirt(self, builder: newton.ModelBuilder) -> None:
        try:
            from pxr import Usd
        except ImportError as exc:
            raise RuntimeError(
                "The shirt asset loader requires the Newton examples dependencies. "
                "Install/run with `uv sync --extra examples` and `uv run --extra examples ...`."
            ) from exc

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        if usd_stage is None:
            raise RuntimeError("Failed to open Newton example asset: unisex_shirt.usd")
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices_np = np.asarray(shirt_mesh.vertices, dtype=np.float32)
        vertices = [wp.vec3(v) for v in vertices_np]
        shirt_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi)

        # Preserve the transform used by Newton's cloth_franka example.
        # The garment mesh is not a single flat sheet, so computing a placement
        # from min(z) is not a reliable way to put it on the table.
        shirt_pos = SHIRT_POS

        particle_start = builder.particle_count
        # Use unit areal density to construct topology, then normalize the complete
        # garment to an explicit physical mass. This avoids scale-dependent mass
        # changes when the centimetre-authored mesh is converted to metres.
        builder.add_cloth_mesh(
            vertices=vertices,
            indices=shirt_mesh.indices,
            pos=shirt_pos,
            rot=shirt_rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=1.0,
            scale=CM_TO_M,
            tri_ke=float(self.args.cloth_tri_ke),
            tri_ka=float(self.args.cloth_tri_ka),
            tri_kd=float(self.args.cloth_tri_kd),
            edge_ke=float(self.args.cloth_edge_ke),
            edge_kd=float(self.args.cloth_edge_kd),
            particle_radius=float(self.args.cloth_radius),
            label="folding_shirt",
        )
        particle_end = builder.particle_count
        raw_mass = float(sum(builder.particle_mass[particle_start:particle_end]))
        if raw_mass <= 0.0:
            raise RuntimeError("The shirt mesh generated zero mass")
        mass_scale = float(self.args.cloth_mass) / raw_mass
        for particle in range(particle_start, particle_end):
            builder.particle_mass[particle] *= mass_scale
        self.actual_cloth_mass = float(sum(builder.particle_mass[particle_start:particle_end]))

    # ------------------------------------------------------------------
    # Coupled solver
    # ------------------------------------------------------------------

    def _build_solver(self, args) -> None:
        shirt_particles = list(range(self.shirt_particle_start, self.shirt_particle_end))

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(
                        model=view,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=int(args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        njmax=512,
                        nconmax=256,
                    ),
                    bodies=self.franka_bodies,
                    joints=self.franka_joints,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=int(args.vbd_iterations),
                        friction_epsilon=float(args.friction_epsilon),
                        rigid_avbd_beta=float(args.vbd_rigid_avbd_beta),
                        rigid_contact_k_start=float(args.vbd_rigid_contact_k_start),
                        # Do not enable VBD rigid body-body contact history here.
                        # The VBD destination view receives MuJoCo finger proxies through
                        # SolverCoupledProxy, and its proxy CollisionPipeline is constructed
                        # after the destination SolverVBD. Enabling history therefore forces
                        # a lazy allocation during CUDA graph capture. It also does not
                        # warm-start particle-vs-finger contacts, so it provides no grasp
                        # benefit in this scene. Newton's coupled VBD examples keep it off.
                        rigid_contact_history=False,
                        particle_enable_self_contact=bool(args.cloth_self_contact),
                        particle_self_contact_radius=self.self_contact_radius,
                        particle_self_contact_margin=self.self_contact_margin,
                        particle_topological_contact_filter_threshold=int(
                            args.topological_contact_filter_threshold
                        ),
                        particle_rest_shape_contact_exclusion_radius=float(
                            args.rest_shape_contact_exclusion_radius
                        ),
                        particle_vertex_contact_buffer_size=int(args.vertex_contact_buffer_size),
                        particle_edge_contact_buffer_size=int(args.edge_contact_buffer_size),
                        rigid_body_particle_contact_buffer_size=int(args.rigid_particle_contact_buffer_size),
                        particle_collision_detection_interval=int(
                            args.particle_collision_detection_interval
                        ),
                    ),
                    bodies=[],
                    particles=shirt_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.gripper_bodies,
                        mass_scale=float(args.mass_scale),
                        mode=args.coupling_mode,
                        collision_pipeline=lambda model: newton.CollisionPipeline(
                            model,
                            broad_phase="explicit",
                            soft_contact_margin=float(args.contact_margin),
                            contact_matching="latest",
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(args.proxy_iterations),
            ),
        )

    # ------------------------------------------------------------------
    # Newton GPU IK
    # ------------------------------------------------------------------

    def _build_ik(self) -> None:
        # A Franka-only view keeps the cloth particles out of the IK problem.
        ik_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._add_franka(ik_builder)
        self.ik_model = ik_builder.finalize(device=self.device)

        self.world_count = 1
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((1, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, -1))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(1, GRIP_OPEN, dtype=float, device=self.device)

        # IMPORTANT: the task-space keyframes come from cloth_franka.py.  That
        # example imports the URDF with collapse_fixed_joints=True and defines
        # its controlled frame as ``body_count - 3`` plus a 22 cm local-z
        # offset.  In the current FR3 asset, body_count - 3 is fr3_link7.
        #
        # Driving fr3_hand_tcp directly is NOT equivalent: the TCP is 9.6 mm
        # behind that legacy point and its fixed hand joint adds -45 degrees
        # about local z.  Using the copied quaternion on hand_tcp therefore
        # rotates the physical gripper by 45 degrees and makes the fingertips
        # miss the shirt edge.  Reproduce the source frame exactly instead.
        hand_body = find_label_index(self.ik_model.body_label, "fr3_link7")
        self.ik_tcp_offset = wp.vec3(0.0, 0.0, 0.22)
        self.ik_frame_label = "fr3_link7 + (0, 0, 0.22 m)"
        first = self.targets[0]
        self.ik_target_positions = wp.array(
            [wp.vec3(*first[:3].tolist())], dtype=wp.vec3, device=self.device
        )
        self.ik_target_rotations = wp.array(
            [wp.vec4(*first[3:7].tolist())], dtype=wp.vec4, device=self.device
        )

        self.pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=self.ik_tcp_offset,
            target_positions=self.ik_target_positions,
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((1, -1))[:, : self.n_coords])
        joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((1, -1))[:, : self.n_coords])
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = int(self.args.ik_iterations)

    # ------------------------------------------------------------------
    # Folding motion
    # ------------------------------------------------------------------

    def _build_keyframes(self) -> None:
        open_ = GRIP_OPEN
        close = self.grip_close
        qa = (0.8536, -0.3536, 0.3536, -0.1464)
        qb = (0.9239, -0.3827, 0.0, 0.0)

        # duration[s], legacy task-frame position[cm], quaternion[xyzw], finger position[m]
        # Sequence: left sleeve -> left lower corner -> right sleeve -> right lower
        # corner -> bottom hem. Positions are adapted from Newton's cloth_franka demo.
        motion_poses_cm = np.array(
            [
                # First pickup: approach from outside the lower hem, descend only
                # 4 mm below the table-top task height, and translate +4 cm in Y
                # while closing. A fixed-center close retracts the inner finger
                # away from this free edge and was the reason V7 merely wrinkled it.
                [4.0, 31.0, -60.0, 40.0, *qa, open_],
                [1.5, 31.0, -60.0, 20.0, *qa, open_],
                [0.8, 32.0, -57.0, 19.6, *qa, open_],
                [1.2, 32.0, -56.0, 19.6, *qa, close],
                [2.0, 32.0, -56.0, 30.0, *qa, close],
                [1.2, 26.0, -60.0, 26.0, *qa, close],
                [2.0, 12.0, -60.0, 31.0, *qa, close],
                [3.0, -6.0, -60.0, 31.0, *qa, close],
                [1.0, -6.0, -60.0, 31.0, *qa, open_],
                [2.0, 15.0, -33.0, 31.0, *qa, open_],
                [3.0, 15.0, -33.0, 21.0, *qa, open_],
                [3.0, 15.0, -33.0, 21.0, *qa, close],
                [2.0, 15.0, -33.0, 28.0, *qa, close],
                [3.0, -2.0, -33.0, 28.0, *qa, close],
                [1.0, -2.0, -33.0, 28.0, *qa, open_],
                [2.0, -28.0, -60.0, 28.0, *qb, open_],
                [2.0, -28.0, -60.0, 20.0, *qb, open_],
                [2.0, -28.0, -60.0, 20.0, *qb, close],
                [2.0, -18.0, -60.0, 31.0, *qb, close],
                [3.0, 5.0, -60.0, 31.0, *qb, close],
                [1.0, 5.0, -60.0, 31.0, *qb, open_],
                [3.0, -18.0, -30.0, 20.5, *qb, open_],
                [3.0, -18.0, -30.0, 20.5, *qb, close],
                [2.0, -3.0, -30.0, 31.0, *qb, close],
                [3.0, -3.0, -30.0, 31.0, *qb, close],
                [2.0, -3.0, -30.0, 31.0, *qb, open_],
                [2.0, 0.0, -20.0, 30.0, *qb, open_],
                [2.0, 0.0, -20.0, 19.5, *qb, open_],
                [2.0, 0.0, -20.0, 19.5, *qb, close],
                [2.0, 0.0, -20.0, 35.0, *qb, close],
                [1.0, 0.0, -30.0, 35.0, *qb, close],
                [1.5, 0.0, -30.0, 35.0, *qb, close],
                [1.5, 0.0, -40.0, 35.0, *qb, close],
                [1.5, 0.0, -40.0, 35.0, *qb, open_],
                [2.0, -28.0, -60.0, 28.0, *qb, open_],
            ],
            dtype=np.float32,
        )

        # Add a true closed-grip dwell after each grasp.  The source controller
        # prescribed joint velocity directly; this version uses a dynamic
        # MuJoCo position drive, so a dwell is needed for both fingers to build
        # contact force before the lift segment starts.
        hold = max(float(self.args.grasp_hold_time), 0.0)
        if hold > 0.0:
            # Insert a dwell after every open->closed transition. Detect the
            # transitions from the gripper column instead of hard-coding row
            # numbers so trajectory edits cannot silently put dwell at the wrong pose.
            close_threshold = 0.5 * (open_ + close)
            close_rows = {
                row_idx
                for row_idx in range(1, len(motion_poses_cm))
                if motion_poses_cm[row_idx - 1, 8] > close_threshold
                and motion_poses_cm[row_idx, 8] <= close_threshold
            }
            expanded = []
            for row_idx, row in enumerate(motion_poses_cm):
                expanded.append(row.copy())
                if row_idx in close_rows:
                    dwell = row.copy()
                    dwell[0] = hold
                    expanded.append(dwell)
            motion_poses_cm = np.asarray(expanded, dtype=np.float32)

        if bool(self.args.first_grasp_only):
            # Isolate the first grasp and finish while still holding the cloth.
            # This avoids confusing a failed pickup with later folding motion.
            motion_poses_cm = np.array(
                [
                    [2.0, 31.0, -60.0, 40.0, *qa, open_],
                    [1.5, 31.0, -60.0, 20.0, *qa, open_],
                    # Move the open gap over the hem and descend 4 mm.
                    [0.8, 32.0, -57.0, 19.6, *qa, open_],
                    # Close while translating the gripper center to the measured
                    # cloth boundary, then dwell before any lift motion.
                    [1.2, 32.0, -56.0, 19.6, *qa, close],
                    [max(hold, 1.0), 32.0, -56.0, 19.6, *qa, close],
                    [2.0, 32.0, -56.0, 30.0, *qa, close],
                    [1.5, 32.0, -56.0, 30.0, *qa, close],
                ],
                dtype=np.float32,
            )

        # Convert only task-frame xyz from cm to m; gripper coordinates are already metres.
        motion_poses = motion_poses_cm.copy()
        motion_poses[:, 1:4] *= CM_TO_M
        motion_poses[:, 0] /= self.motion_speed
        for row in motion_poses:
            row[4:8] = normalized_quaternion_xyzw(row[4:8])

        # Hold the actual FK pose while the shirt settles, then interpolate from that
        # pose to the first folding waypoint. Previously interval zero was already the
        # first cloth waypoint, so IK made a discontinuous jump on the first frame.
        start_pos, start_rot = self._initial_tcp_pose()
        settle_pose = np.array(
            [[float(self.args.settle_time), *start_pos.tolist(), *start_rot.tolist(), open_]],
            dtype=np.float32,
        )
        poses = np.concatenate([settle_pose, motion_poses], axis=0)
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])
        self.task_duration = float(self.key_times[-1])

    def _initial_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        # Use the same legacy task frame as _build_ik().  The initial hold pose
        # must be expressed in the controlled frame, otherwise the first
        # interpolation silently reintroduces a frame discontinuity.
        hand_body = find_label_index(self.model.body_label, "fr3_link7")
        local_offset = wp.vec3(0.0, 0.0, 0.22)
        hand_q = state.body_q.numpy()[hand_body]
        pos = wp.vec3(float(hand_q[0]), float(hand_q[1]), float(hand_q[2]))
        rot = wp.quat(float(hand_q[3]), float(hand_q[4]), float(hand_q[5]), float(hand_q[6]))
        tcp_pos = pos + wp.quat_rotate(rot, local_offset)
        return (
            np.array([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])], dtype=np.float32),
            normalized_quaternion_xyzw(
                np.array([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])], dtype=np.float32)
            ),
        )

    def update_ik_targets(self) -> None:
        t = min(self.sim_time, self.task_duration - 1.0e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        t_end = self.key_times[interval]
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))
        # Cubic smoothstep gives zero endpoint velocity and avoids exciting both the
        # joint PD controller and the cloth at every keyframe transition.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        cur = self.targets[interval]
        prev = self.targets[interval - 1] if interval > 0 else cur

        # Keep shortest-path quaternion interpolation, then renormalize. A normalized
        # lerp is sufficient for these slowly changing task-space keyframes.
        prev_q = prev[3:7].copy()
        cur_q = cur[3:7].copy()
        if float(np.dot(prev_q, cur_q)) < 0.0:
            cur_q *= -1.0
        pos = (1.0 - alpha) * prev[:3] + alpha * cur[:3]
        quat = normalized_quaternion_xyzw((1.0 - alpha) * prev_q + alpha * cur_q)
        grip = float((1.0 - alpha) * prev[-1] + alpha * cur[-1])

        wp.launch(
            set_task_targets,
            dim=1,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_pos_buf,
                wp.vec3(*pos.tolist()),
                wp.vec4(*quat.tolist()),
                grip,
            ],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def capture(self) -> None:
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.device}")
            self.graph = capture.graph

    def simulate(self) -> None:
        # IK result drives MuJoCo's joint-space PD targets.
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.device,
        )
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self.update_ik_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_final(self) -> None:
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"

        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()[self.shirt_particle_start : self.shirt_particle_end]
        assert np.all(np.isfinite(body_q)), "Robot body transforms contain NaN or inf"
        assert np.all(np.isfinite(particle_q)), "Shirt particles contain NaN or inf"

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = float(np.linalg.norm(max_pos - min_pos))
        assert bbox_size < 5.0, f"Shirt particle bounding box exploded: {bbox_size:.3f} m"
        assert min_pos[2] > -0.25, f"Shirt penetrated excessively below the floor: z={min_pos[2]:.4f} m"

        final_span = np.ptp(particle_q[:, :2], axis=0)
        final_area = float(final_span[0] * final_span[1])
        ratio = final_area / self.initial_planar_aabb_area

        patch_global = np.asarray(self.first_grasp_particle_ids, dtype=np.int64)
        patch_local = patch_global - self.shirt_particle_start
        patch_z = particle_q[patch_local, 2]
        patch_p75_z = float(np.percentile(patch_z, 75.0))
        table_top = float(TABLE_POS[2] + TABLE_HALF_EXTENTS[2])
        patch_clearance = patch_p75_z - table_top
        print(
            "[cloth_folding_coupled] first-grasp patch: "
            f"initial_p75_z={self.first_grasp_initial_p75_z:.4f} m, "
            f"final_p75_z={patch_p75_z:.4f} m, "
            f"table_clearance={patch_clearance:.4f} m"
        )
        if bool(self.args.first_grasp_only):
            required_clearance = float(self.args.grasp_validation_clearance)
            assert patch_clearance >= required_clearance, (
                "First grasp failed: the tracked shirt patch was not lifted. "
                f"Measured p75 clearance={patch_clearance:.4f} m, "
                f"required={required_clearance:.4f} m."
            )

        print(
            f"[cloth_folding_coupled] task_duration={self.task_duration:.2f}s, "
            f"cloth_mass={self.actual_cloth_mass:.3f} kg, "
            f"final/initial planar AABB area={ratio:.3f}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)

        parser.add_argument("--substeps", type=int, default=8, help="Coupled substeps per rendered frame.")
        parser.add_argument("--ik-iterations", type=int, default=12, help="Newton GPU IK iterations per frame.")
        parser.add_argument(
            "--motion-speed",
            type=float,
            default=1.0,
            help="Keyframe playback speed multiplier; values >1 shorten the task but reduce grasp robustness.",
        )
        parser.add_argument(
            "--grip-close",
            type=float,
            default=DEFAULT_GRIP_CLOSE,
            help="Per-finger closed target [m]. Increase if the cloth is squeezed through the fingers.",
        )

        parser.add_argument("--proxy-iterations", type=int, default=2, help="Proxy coupling passes per substep; two passes improve pinch-force transfer.")
        parser.add_argument("--mass-scale", type=float, default=1.0, help="Proxy effective mass/inertia scale.")
        parser.add_argument(
            "--coupling-mode",
            choices=["lagged", "staggered"],
            default="staggered",
            help="SolverCoupledProxy transfer mode.",
        )
        parser.add_argument("--mujoco-iterations", type=int, default=12)
        parser.add_argument("--mujoco-ls-iterations", type=int, default=25)
        parser.add_argument("--vbd-iterations", type=int, default=8)
        parser.add_argument("--vbd-rigid-avbd-beta", type=float, default=0.0)
        parser.add_argument("--vbd-rigid-contact-k-start", type=float, default=1.0e2)

        parser.add_argument("--contact-ke", type=float, default=5.0e4)
        parser.add_argument("--contact-kd", type=float, default=50.0)
        parser.add_argument("--contact-mu", type=float, default=1.5)
        parser.add_argument("--table-mu", type=float, default=0.6)
        parser.add_argument("--self-contact-mu", type=float, default=0.25)
        parser.add_argument("--contact-gap", type=float, default=0.002)
        parser.add_argument("--contact-margin", type=float, default=0.008)
        parser.add_argument("--friction-epsilon", type=float, default=0.01)

        parser.add_argument("--cloth-mass", type=float, default=0.18, help="Total shirt mass [kg].")
        parser.add_argument(
            "--cloth-settle-clearance",
            type=float,
            default=0.002,
            help="Deprecated compatibility option; the V3 scene uses the shirt asset's authored transform.",
        )
        parser.add_argument("--settle-time", type=float, default=0.1, help="Initial robot/cloth hold time [s].")
        parser.add_argument(
            "--grasp-hold-time",
            type=float,
            default=1.0,
            help="Closed-gripper dwell at each pickup pose before lifting [s].",
        )
        parser.add_argument(
            "--first-grasp-only",
            action="store_true",
            help="Run only approach/close/hold/lift for the first shirt corner and validate pickup.",
        )
        parser.add_argument(
            "--grasp-validation-clearance",
            type=float,
            default=0.04,
            help="Required tracked-patch p75 height above the table in --first-grasp-only mode [m].",
        )
        parser.add_argument("--cloth-tri-ke", type=float, default=1.0e4)
        parser.add_argument("--cloth-tri-ka", type=float, default=1.0e4)
        parser.add_argument("--cloth-tri-kd", type=float, default=1.5e-2)
        parser.add_argument("--cloth-edge-ke", type=float, default=0.08)
        parser.add_argument("--cloth-edge-kd", type=float, default=0.02)
        parser.add_argument("--cloth-radius", type=float, default=0.008)
        parser.add_argument("--self-contact-radius", type=float, default=0.002)
        parser.add_argument(
            "--self-contact-margin",
            type=float,
            default=None,
            help="Self-contact broad-phase margin [m]. Defaults to 1.5x --self-contact-radius.",
        )
        parser.add_argument("--rest-shape-contact-exclusion-radius", type=float, default=0.005)
        parser.add_argument(
            "--topological-contact-filter-threshold",
            type=int,
            default=1,
            help="Suppress self-contact between nearby mesh primitives; matches cloth_franka by default.",
        )
        parser.add_argument(
            "--particle-collision-detection-interval",
            type=int,
            default=-1,
            help="Build self-contact candidates once per VBD step, matching cloth_franka and reducing cost.",
        )
        parser.add_argument("--vertex-contact-buffer-size", type=int, default=32)
        parser.add_argument("--edge-contact-buffer-size", type=int, default=64)
        parser.add_argument(
            "--rigid-particle-contact-buffer-size",
            type=int,
            default=1024,
            help="Maximum VBD rigid-body/particle contacts retained per solve.",
        )
        parser.add_argument(
            "--no-cloth-self-contact",
            action="store_false",
            dest="cloth_self_contact",
            default=True,
            help="Disable cloth self-contact for faster debugging.",
        )
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    # Full trajectory duration includes one-second closed-grip dwells.
    parser.set_defaults(num_frames=4500)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
