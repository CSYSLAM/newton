# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka MJVBD Style3D
#
# Reuses the Franka + MJVBD folding flow from the shirt example, but swaps in
# the Style3D Women_Sweatshirt garment mesh. The garment is recentered and
# rescaled to lie flat over the table in the centimeter-based Franka scene.
#
# With --enable-franka: Franka follows the scripted grasp trajectory through MJVBD.
# Without: Franka stays frozen, cloth drapes freely.
#
# Command: python -m newton.examples.cloth.example_cloth_franka_mjvbd_style3d
#          python -m newton.examples.cloth.example_cloth_franka_mjvbd_style3d --disable-franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.mjvbd
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_target: float,
):
    joint_targets[0] = ik_solution[0, 0]
    joint_targets[1] = ik_solution[0, 1]
    joint_targets[2] = ik_solution[0, 2]
    joint_targets[3] = ik_solution[0, 3]
    joint_targets[4] = ik_solution[0, 4]
    joint_targets[5] = ik_solution[0, 5]
    joint_targets[6] = ik_solution[0, 6]
    joint_targets[7] = gripper_target
    joint_targets[8] = gripper_target


@wp.kernel
def update_joint_velocity_kernel(
    joint_q_prev: wp.array[wp.float32],
    joint_q_next: wp.array[wp.float32],
    inv_dt: float,
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    joint_qd[i] = (joint_q_next[i] - joint_q_prev[i]) * inv_dt


@wp.kernel
def interpolate_joint_positions_kernel(
    joint_q_start: wp.array[wp.float32],
    joint_q_end: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = joint_q_start[i] * (1.0 - alpha) + joint_q_end[i] * alpha


@wp.kernel
def load_cached_joint_targets_kernel(
    cached_joint_targets: wp.array2d[wp.float32],
    frame_index: int,
    joint_targets: wp.array[wp.float32],
):
    i = wp.tid()
    joint_targets[i] = cached_joint_targets[frame_index, i]


@wp.kernel
def copy_joint_positions_kernel(src: wp.array[wp.float32], dst: wp.array[wp.float32]):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def load_cached_joint_targets_from_counter_kernel(
    cached_joint_targets: wp.array2d[wp.float32],
    frame_counter: wp.array[wp.int32],
    max_frame_index: int,
    joint_targets: wp.array[wp.float32],
):
    i = wp.tid()
    frame_index = wp.min(frame_counter[0], max_frame_index)
    joint_targets[i] = cached_joint_targets[frame_index, i]


@wp.kernel
def advance_frame_counter_kernel(frame_counter: wp.array[wp.int32], max_frame_index: int):
    if wp.tid() == 0:
        frame_counter[0] = wp.min(frame_counter[0] + 1, max_frame_index)


TARGET_CUBE_SIZE = 1.0  # cm, half-extents
TARGET_CUBE_COLOR = wp.vec3(1.0, 0.2, 0.2)
GRIPPER_OPEN = 3.2
GRIPPER_CLOSED = 0.4
GRASP_QUAT = [0.88806, -0.45973, 0.0, 0.0]

GARMENT_USD_NAME = "Women_Sweatshirt"
GARMENT_SCALE = 75.0
GARMENT_DENSITY = 0.02
GARMENT_POS = wp.vec3(0.0, -50.0, 30.0)

CLOTH_COLOR = (0.72, 0.72, 0.76)

TABLE_HX_CM = 40.0
TABLE_HY_CM = 40.0
TABLE_HZ_CM = 1.0
TABLE_POS_CM = wp.vec3(0.0, -50.0, 2.0)
TABLE_COLOR = wp.vec3(0.16, 0.55, 0.78)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.enable_franka = getattr(args, "enable_franka", True) and not getattr(args, "disable_franka", False)

        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.self_contact_bvh_rebuild_interval_frames = 30

        self.viz_scale = 0.01

        # Scale Style3D collision widths down to match the denser garment mesh.
        self.cloth_particle_radius = 0.35
        self.cloth_body_contact_margin = 0.4
        self.particle_self_contact_radius = 0.1
        self.particle_self_contact_margin = 0.1

        self.soft_contact_ke = 1.0e4
        self.soft_contact_kd = 1.0e-2

        self.robot_contact_ke = 5.0e4
        self.robot_contact_kd = 1.0e-3
        self.robot_contact_mu = 1.5
        self.self_contact_friction = 0.25

        self.tri_ke = 1.0e3
        self.tri_ka = 1.0e3
        self.tri_kd = 1.0e-5

        self.bending_ke = 1.0
        self.bending_kd = 0.1

        self.scene = ModelBuilder(gravity=-981.0)

        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count

        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            wp.transform(TABLE_POS_CM, wp.quat_identity()),
            hx=TABLE_HX_CM,
            hy=TABLE_HY_CM,
            hz=TABLE_HZ_CM,
        )

        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(str(asset_path / "garments" / f"{GARMENT_USD_NAME}.usd"))
        usd_prim = usd_stage.GetPrimAtPath(f"/Root/{GARMENT_USD_NAME}/Root_Garment")
        garment_mesh = newton.usd.get_mesh(usd_prim)
        garment_vertices = self._center_garment_vertices(garment_mesh.vertices)

        self.scene.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in garment_vertices],
            indices=garment_mesh.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=GARMENT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=GARMENT_DENSITY,
            scale=GARMENT_SCALE,
            tri_ke=self.tri_ke,
            tri_ka=self.tri_ka,
            tri_kd=self.tri_kd,
            edge_ke=self.bending_ke,
            edge_kd=self.bending_kd,
            particle_radius=self.cloth_particle_radius,
        )

        self.scene.color(include_bending=True)
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(TABLE_POS_CM[0]) * self.viz_scale,
                        float(TABLE_POS_CM[1]) * self.viz_scale,
                        float(TABLE_POS_CM[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            TABLE_HX_CM * self.viz_scale,
            TABLE_HY_CM * self.viz_scale,
            TABLE_HZ_CM * self.viz_scale,
        )
        self.table_viz_color = wp.array([TABLE_COLOR], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.viz_state = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.graph_frame_index = wp.array([1], dtype=wp.int32, device=self.model.device)
        self.cached_joint_targets = None
        self.cached_joint_target_frame_count = 0

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=self.cloth_body_contact_margin,
            iterations=self.iterations,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.5,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )

        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.setup_ik()
        if self.enable_franka:
            self.build_joint_target_cache()

        self.graph = None
        self.capture()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.72, 0.62, 1.28), -41.0, -60.0)

    def _refresh_self_contact_bvh(self) -> None:
        if not self.solver.particle_enable_self_contact:
            return

        if self.frame_index > 0 and self.frame_index % self.self_contact_bvh_rebuild_interval_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

    def _prepare_substep_state(self) -> None:
        # Only copy the arrays needed before the MJVBD substep.
        wp.copy(self.state_1.particle_q, self.state_0.particle_q)

        if not self.enable_franka:
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

    def _center_garment_vertices(self, vertices: np.ndarray) -> np.ndarray:
        vertices_np = np.asarray(vertices, dtype=np.float32)
        bbox_min = vertices_np.min(axis=0)
        bbox_max = vertices_np.max(axis=0)
        bbox_center = 0.5 * (bbox_min + bbox_max)
        return vertices_np - bbox_center

    def create_articulation(self, builder: ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-50.0, -50.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=100.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:7] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.7854]
        builder.joint_q[7:9] = [GRIPPER_OPEN, GRIPPER_OPEN]
        builder.joint_target_pos[:9] = [*builder.joint_q[:9]]
        builder.joint_target_ke[:9] = [4000.0] * 7 + [12000.0, 12000.0]
        builder.joint_target_kd[:9] = [400.0] * 7 + [1200.0, 1200.0]
        builder.joint_effort_limit[:7] = [300.0] * 7
        builder.joint_effort_limit[7:9] = [2000.0, 2000.0]
        builder.joint_armature[:7] = [0.2] * 7
        builder.joint_armature[7:9] = [0.5, 0.5]

        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.vec3(0.0, 0.0, 22.0)
        self.robot_key_poses = np.array(
            [
                [3.0, 20.0, -60.0, 40.0, *GRASP_QUAT, GRIPPER_OPEN],
                [1.0, 31.1052, -31.6017, 26.0012, 0.8639, -0.4991, 0.0624, -0.0270 , GRIPPER_OPEN],
                [1.0, 31.1052, -31.6017, 16.0012, 0.8639, -0.4991, 0.0624, -0.0270 , GRIPPER_OPEN],
                [1.0, 31.1052, -31.6017, 5.8, 0.8639, -0.4991, 0.0624, -0.0270 , 3.0],
                [2.0, 31.1052, -31.6017, 5.8, 0.8639, -0.4991, 0.0624, -0.0270 , 0.2],
                [1.0, 31.1052, -31.6017, 12.0012, 0.8639, -0.4991, 0.0624, -0.0270 , GRIPPER_CLOSED],
                [1.0, 31.1052, -31.6017, 15.0012, 0.8639, -0.4991, 0.0624, -0.0270 , GRIPPER_CLOSED],
                [1.0, 26.1844, -47.4308, 20.9441, 0.8639, -0.4991, 0.0623, -0.0270, GRIPPER_CLOSED],
                [1.0, 23.9451, -56.8466, 15.4145, 0.8639, -0.4991, 0.0624, -0.0270, GRIPPER_CLOSED],
                [1.0, 7.2415, -65.1912, 12.0000, 0.8639, -0.4991, 0.0623, -0.0270, GRIPPER_CLOSED],
                [1.0, 7.2415, -65.1912, 6.0000, 0.8639, -0.4991, 0.0623, -0.0270, GRIPPER_CLOSED],
                [1.0, 7.2415, -65.1912, 6.0000, 0.8639, -0.4991, 0.0623, -0.0270, 2.1],
                [1.0, 7.2415, -65.1912, 12.0000, 0.8639, -0.4991, 0.0623, -0.0270, GRIPPER_OPEN],
                [1.0, 7.2415, -65.1912, 20.0000, 0.8639, -0.4991, 0.0623, -0.0270, GRIPPER_OPEN],

                [1.0, -28.9794, -31.0225, 20.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_OPEN],
                [1.0, -28.9794, -31.0225, 12.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_OPEN],
                [1.0, -28.9794, -31.0225, 8.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_OPEN],
                [1.0, -28.9794, -31.0225, 8.0000, 0.9982, -0.0065, 0.0556, -0.0194, 0.2],
                [1.0, -28.9794, -31.0225, 12.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -28.9794, -31.0225, 20.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -17.8216, -49.8613, 20.1037, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -11.0285, -63.3218, 21.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -11.0285, -63.3218, 12.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -11.0285, -63.3218, 7.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_CLOSED],
                [1.0, -11.0285, -63.3218, 7.0000, 0.9982, -0.0065, 0.0556, -0.0194, 2.1],
                [1.0, -11.0285, -63.3218, 12.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_OPEN],
                [1.0, -11.0285, -63.3218, 20.0000, 0.9982, -0.0065, 0.0556, -0.0194, GRIPPER_OPEN],

                [1.0, -1.3491, -35.6835, 20.0000, 0.9285, -0.3667, 0.0588, 0.0020, GRIPPER_OPEN],
                [1.0, -1.3491, -35.6835, 12.0000, 0.9285, -0.3667, 0.0588, 0.0020, GRIPPER_OPEN],
                [2.0, -1.3491, -33.7312, 6.0000, 0.9285, -0.3667, 0.0588, 0.0020, GRIPPER_OPEN],
                [2.0, -1.3491, -33.7312, 6.0000, 0.9285, -0.3667, 0.0588, 0.0020, 0.1],
                [1.0, -1.3491, -33.7312, 12.0000, 0.9285, -0.3667, 0.0588, 0.0020, GRIPPER_CLOSED],
                [1.0, -1.3491, -49.9879, 20.2274, 0.9285, -0.3667, 0.0588, 0.0020, GRIPPER_CLOSED]
            ],
            dtype=np.float32,
        )
        self.robot_targets = self.robot_key_poses[:, 1:]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

    def setup_ik(self) -> None:
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=self.endeffector_offset,
            target_positions=wp.array([wp.vec3(*self.robot_targets[0][:3].tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [wp.vec4(*self.robot_targets[0][3:7].tolist())],
                dtype=wp.vec4,
            ),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 8
        self.current_gripper_target = float(self.robot_targets[0][-1])

    def interpolated_target(self, query_time: float | None = None) -> np.ndarray:
        sample_time = self.sim_time if query_time is None else query_time

        if sample_time >= self.robot_key_poses_time[-1]:
            return self.robot_targets[-1]

        interval = int(np.searchsorted(self.robot_key_poses_time, sample_time))
        t_start = float(self.robot_key_poses_time[interval - 1]) if interval > 0 else 0.0
        t_end = float(self.robot_key_poses_time[interval])
        alpha = float(np.clip((sample_time - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        target_cur = self.robot_targets[interval]
        target_prev = self.robot_targets[interval - 1] if interval > 0 else target_cur
        return (1.0 - alpha) * target_prev + alpha * target_cur

    def set_joint_targets(self, query_time: float | None = None) -> None:
        if self.cached_joint_targets is not None:
            sample_time = self.sim_time if query_time is None else query_time
            frame_index = int(np.clip(np.round(sample_time * self.fps), 0, self.cached_joint_target_frame_count))
            wp.launch(
                load_cached_joint_targets_kernel,
                dim=9,
                inputs=[self.cached_joint_targets, frame_index],
                outputs=[self.control.joint_target_pos],
                device=self.model.device,
            )
            return

        self._set_joint_targets_via_ik(query_time)

    def _set_joint_targets_via_ik(self, query_time: float | None = None) -> None:
        target = self.interpolated_target(query_time)
        self.pos_obj.set_target_position(0, wp.vec3(*target[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target[3:7].tolist()))
        self.current_gripper_target = float(target[-1])
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=1,
            inputs=[self.ik_joint_q, self.control.joint_target_pos, self.current_gripper_target],
            device=self.model.device,
        )

    def build_joint_target_cache(self) -> None:
        cache_frame_count = int(np.ceil(self.robot_key_poses_time[-1] * self.fps))
        cache = np.zeros((cache_frame_count + 1, 9), dtype=np.float32)
        cache[0] = self.model.joint_q.numpy()[:9]

        for frame_index in range(1, cache_frame_count + 1):
            query_time = min(frame_index * self.frame_dt, float(self.robot_key_poses_time[-1]))
            self._set_joint_targets_via_ik(query_time)
            wp.synchronize_device()
            cache[frame_index] = self.control.joint_target_pos.numpy()[:9]

        self.cached_joint_targets = wp.array(cache, dtype=wp.float32, device=self.model.device)
        self.cached_joint_target_frame_count = cache_frame_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])
        self.current_gripper_target = float(self.robot_targets[0][-1])

    def _prepare_frame_inputs(self) -> None:
        self._refresh_self_contact_bvh()

        if self.enable_franka:
            wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
            self.set_joint_targets(self.sim_time + self.frame_dt)
            wp.copy(self.frame_joint_q_end, self.control.joint_target_pos)

    def _simulate_substeps(self) -> None:
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self._prepare_substep_state()

            if self.enable_franka:
                substep_alpha = float((substep + 1) / self.sim_substeps)
                wp.launch(
                    interpolate_joint_positions_kernel,
                    dim=self.model.joint_coord_count,
                    inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                    outputs=[self.state_1.joint_q],
                    device=self.model.device,
                )
                wp.launch(
                    update_joint_velocity_kernel,
                    dim=self.model.joint_dof_count,
                    inputs=[self.state_0.joint_q, self.state_1.joint_q, 1.0 / self.sim_dt],
                    outputs=[self.state_1.joint_qd],
                    device=self.model.device,
                )
                eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self) -> None:
        if self.model.device.is_cuda:
            state_0_backup = self.model.state()
            state_1_backup = self.model.state()
            state_0_backup.assign(self.state_0)
            state_1_backup.assign(self.state_1)
            control_joint_target_backup = wp.zeros_like(self.control.joint_target_pos)
            wp.copy(control_joint_target_backup, self.control.joint_target_pos)
            wp.copy(self.frame_joint_q_start, self.model.joint_q)
            wp.copy(self.frame_joint_q_end, self.model.joint_q)
            self.graph_frame_index.fill_(1)

            with wp.ScopedCapture() as capture:
                if self.enable_franka:
                    wp.launch(
                        copy_joint_positions_kernel,
                        dim=self.model.joint_coord_count,
                        inputs=[self.state_0.joint_q],
                        outputs=[self.frame_joint_q_start],
                        device=self.model.device,
                    )
                    wp.launch(
                        load_cached_joint_targets_from_counter_kernel,
                        dim=9,
                        inputs=[
                            self.cached_joint_targets,
                            self.graph_frame_index,
                            self.cached_joint_target_frame_count,
                        ],
                        outputs=[self.frame_joint_q_end],
                        device=self.model.device,
                    )
                self._simulate_substeps()
                if self.enable_franka:
                    wp.launch(
                        advance_frame_counter_kernel,
                        dim=1,
                        inputs=[self.graph_frame_index, self.cached_joint_target_frame_count],
                        device=self.model.device,
                    )
            self.graph = capture.graph

            self.state_0.assign(state_0_backup)
            self.state_1.assign(state_1_backup)
            wp.copy(self.control.joint_target_pos, control_joint_target_backup)
            wp.copy(self.frame_joint_q_start, self.model.joint_q)
            wp.copy(self.frame_joint_q_end, self.model.joint_q)
            self.graph_frame_index.fill_(1)
        else:
            self.graph = None

    def simulate(self) -> None:
        self._prepare_frame_inputs()
        self._simulate_substeps()
        self.sim_time += self.frame_dt

    def step(self) -> None:
        if self.graph:
            self._refresh_self_contact_bvh()
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.simulate()
        self.frame_index += 1

    def render(self) -> None:
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.viz_state)
        self.viewer.log_mesh(
            "/model/triangles",
            self.viz_state.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
            color=CLOTH_COLOR,
        )

        self.viewer.log_shapes(
            "/table",
            newton.GeoType.BOX,
            self.table_viz_scale,
            self.table_viz_xform,
            self.table_viz_color,
        )

        target = self.interpolated_target()
        cube_pos_m = wp.vec3(
            float(target[0]) * self.viz_scale,
            float(target[1]) * self.viz_scale,
            float(target[2]) * self.viz_scale,
        )
        self.viewer.log_shapes(
            "/target_cube",
            newton.GeoType.BOX,
            (TARGET_CUBE_SIZE * self.viz_scale, TARGET_CUBE_SIZE * self.viz_scale, TARGET_CUBE_SIZE * self.viz_scale),
            wp.array(
                [wp.transform(cube_pos_m, wp.quat(float(target[3]), float(target[4]), float(target[5]), float(target[6])))],
                dtype=wp.transform,
            ),
            wp.array([TARGET_CUBE_COLOR], dtype=wp.vec3),
        )

        self.viewer.end_frame()

        self.model.shape_transform = self.sim_shape_transform
        self.model.shape_scale = self.sim_shape_scale

    def test_final(self) -> None:
        p_lower = wp.vec3(-46.0, -96.0, -5.0)
        p_upper = wp.vec3(46.0, 8.0, 60.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 250.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--enable-franka", action="store_true", default=True, help="Enable Franka IK tracking to target cube (default: True)")
        parser.add_argument("--disable-franka", action="store_true", help="Disable Franka, keep it frozen")
        parser.set_defaults(num_frames=3850)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
