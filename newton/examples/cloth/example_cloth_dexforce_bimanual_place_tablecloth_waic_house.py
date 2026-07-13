# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual Place Tablecloth in the WAIC house scene
#
# Reuses ``example_cloth_dexforce_bimanual_place_tablecloth`` and applies one
# world-space rigid transform anchored at the latest W1 pose from Blender.
#
# Command:
#   python -m newton.examples cloth_dexforce_bimanual_place_tablecloth_waic_house
#
###########################################################################

from __future__ import annotations

import warp as wp
from pxr import Gf, UsdGeom

import newton
import newton.examples
from newton.examples.cloth import example_cloth_dexforce_bimanual_fold_tshirt_waic_house as fold_waic
from newton.examples.cloth import example_cloth_dexforce_bimanual_place_tablecloth as place


OLD_ROBOT_BASE_POS = wp.vec3(0.0, place.TABLECLOTH_START_Y, 0.0)
WAIC_ROBOT_BASE_POS = wp.vec3(3.3669776916503906, 1.2771245241165161, -0.0037720240652561188)
WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.0, 1.0)

PHYSICS_TABLE_HIDDEN_COLOR = fold_waic.PHYSICS_TABLE_HIDDEN_COLOR
HOUSE_VISUAL_USD = fold_waic.HOUSE_VISUAL_USD
CAMERA_POS = wp.vec3(6.134052276611328, -0.2173839807510376, 1.3652291297912598)
CAMERA_PITCH = -11.199997684740863
CAMERA_YAW = 142.39995142654632
CAMERA_LENS_MM = 35.0
CAMERA_RIGHT = (0.6108649847336679, 0.7917351130931174, 0.0)
CAMERA_UP = (0.1537677568147261, -0.11865019313872387, 0.9809518419261425)
CAMERA_BACK = (0.7772001028060913, -0.5985257029533386, 0.19423431158065796)


@wp.kernel
def set_free_root_motion_waic_kernel(
    root_q_start: int,
    waic_root_pos: wp.vec3,
    scene_rotation: wp.quat,
    old_root_pos: wp.vec3,
    old_root_rot: wp.quat,
    base_time: float,
    retreat_distance: float,
    retreat_time: float,
    turn_out_radians: float,
    turn_out_time: float,
    center_shift_y: float,
    center_move_time: float,
    turn_in_radians: float,
    turn_in_time: float,
    approach_distance: float,
    approach_time: float,
    joint_q_out: wp.array[wp.float32],
):
    t0 = retreat_time
    t1 = t0 + turn_out_time
    t2 = t1 + center_move_time
    t3 = t2 + turn_in_time

    retreat_u = place.smoothstep(base_time / wp.max(retreat_time, 1.0e-6))
    turn_out_u = place.smoothstep((base_time - t0) / wp.max(turn_out_time, 1.0e-6))
    center_u = place.smoothstep((base_time - t1) / wp.max(center_move_time, 1.0e-6))
    turn_in_u = place.smoothstep((base_time - t2) / wp.max(turn_in_time, 1.0e-6))
    approach_u = place.smoothstep((base_time - t3) / wp.max(approach_time, 1.0e-6))

    yaw = turn_out_radians * turn_out_u + turn_in_radians * turn_in_u
    q_yaw = wp.quat(0.0, 0.0, wp.sin(0.5 * yaw), wp.cos(0.5 * yaw))
    old_q = q_yaw * old_root_rot
    new_q = scene_rotation * old_q

    old_p = wp.vec3(
        old_root_pos[0] - retreat_distance * retreat_u + approach_distance * approach_u,
        old_root_pos[1] + center_shift_y * center_u,
        old_root_pos[2],
    )
    new_p = waic_root_pos + wp.quat_rotate(scene_rotation, old_p - old_root_pos)

    joint_q_out[root_q_start + 0] = new_p[0]
    joint_q_out[root_q_start + 1] = new_p[1]
    joint_q_out[root_q_start + 2] = new_p[2]
    joint_q_out[root_q_start + 3] = new_q[0]
    joint_q_out[root_q_start + 4] = new_q[1]
    joint_q_out[root_q_start + 5] = new_q[2]
    joint_q_out[root_q_start + 6] = new_q[3]


class Example(place.Example):
    _normalize_wp_quat = staticmethod(fold_waic.Example._normalize_wp_quat)
    _quat_multiply = staticmethod(fold_waic.Example._quat_multiply)
    _quat_rotate_vec3 = staticmethod(fold_waic.Example._quat_rotate_vec3)
    _attach_house_visual_stage = fold_waic.Example._attach_house_visual_stage
    _add_house_visual_model = fold_waic.Example._add_house_visual_model
    _usd_mesh_to_newton_arrays = staticmethod(fold_waic.Example._usd_mesh_to_newton_arrays)
    _usd_mesh_color = fold_waic.Example._usd_mesh_color
    _material_texture_average_color = fold_waic.Example._material_texture_average_color
    _average_texture_rgb = staticmethod(fold_waic.Example._average_texture_rgb)
    _material_diffuse_color = staticmethod(fold_waic.Example._material_diffuse_color)
    _semantic_house_color = staticmethod(fold_waic.Example._semantic_house_color)

    def __init__(self, viewer, args):
        self.waic_robot_base_pos = wp.vec3(
            float(args.waic_robot_base_x),
            float(args.waic_robot_base_y),
            float(args.waic_robot_base_z),
        )
        self.scene_rotation = self._normalize_wp_quat(
            wp.quat(
                float(args.waic_robot_base_qx),
                float(args.waic_robot_base_qy),
                float(args.waic_robot_base_qz),
                float(args.waic_robot_base_qw),
            )
        )
        self.show_physics_table = bool(args.show_physics_table)
        self.house_visual_usd = str(args.house_visual_usd)
        self.hide_house_visual = bool(args.hide_house_visual)
        self.house_visual_color_mode = str(args.house_visual_color_mode)
        self._house_material_color_cache: dict[str, tuple[float, float, float]] = {}
        self.waic_table_pos = self._waic_vec3(place.TABLE_POS)
        self.waic_ground_height = float(self._waic_vec3(wp.vec3(0.0, 0.0, 0.0))[2])
        self.house_shape_start = 0
        self.house_shape_end = 0
        self._waic_runtime_ready = False

        super().__init__(viewer, args)

        self._set_state_root_to_waic_initial()
        self._waic_runtime_ready = True
        self._attach_house_visual_stage()
        self._attach_initial_usd_camera()
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self._report_pose(force=True)

    def _robot_xform(self) -> wp.transform:
        return wp.transform(OLD_ROBOT_BASE_POS, wp.quat_identity())

    def _report_pose(self, force: bool = False) -> None:
        if not getattr(self, "_waic_runtime_ready", False):
            return
        super()._report_pose(force=force)

    def _set_state_root_to_waic_initial(self) -> None:
        root_q = place.np.array(
            [
                float(self.waic_robot_base_pos[0]),
                float(self.waic_robot_base_pos[1]),
                float(self.waic_robot_base_pos[2]),
                float(self.scene_rotation[0]),
                float(self.scene_rotation[1]),
                float(self.scene_rotation[2]),
                float(self.scene_rotation[3]),
            ],
            dtype=place.np.float32,
        )

        for state in (self.state_0, self.state_1):
            joint_q = state.joint_q.numpy()
            joint_q[self.root_q_start : self.root_q_start + 7] = root_q
            state.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.model.device)

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _attach_initial_usd_camera(self) -> None:
        if not hasattr(self.viewer, "stage"):
            return

        camera = UsdGeom.Camera.Define(self.viewer.stage, "/root/waic_initial_camera")
        camera.CreateFocalLengthAttr().Set(CAMERA_LENS_MM)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))

        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRow(0, Gf.Vec4d(*CAMERA_RIGHT, 0.0))
        matrix.SetRow(1, Gf.Vec4d(*CAMERA_UP, 0.0))
        matrix.SetRow(2, Gf.Vec4d(*CAMERA_BACK, 0.0))
        matrix.SetRow(3, Gf.Vec4d(float(CAMERA_POS[0]), float(CAMERA_POS[1]), float(CAMERA_POS[2]), 1.0))
        xform.AddTransformOp().Set(matrix)

    def _waic_vec3(self, value: wp.vec3) -> wp.vec3:
        local = wp.vec3(
            float(value[0]) - float(OLD_ROBOT_BASE_POS[0]),
            float(value[1]) - float(OLD_ROBOT_BASE_POS[1]),
            float(value[2]) - float(OLD_ROBOT_BASE_POS[2]),
        )
        rotated = self._quat_rotate_vec3(self.scene_rotation, local)
        return wp.vec3(
            float(self.waic_robot_base_pos[0]) + float(rotated[0]),
            float(self.waic_robot_base_pos[1]) + float(rotated[1]),
            float(self.waic_robot_base_pos[2]) + float(rotated[2]),
        )

    def _waic_transform(self, tf: wp.transform) -> wp.transform:
        return wp.transform(
            self._waic_vec3(wp.transform_get_translation(tf)),
            self._quat_multiply(self.scene_rotation, wp.transform_get_rotation(tf)),
        )

    def _apply_root_motion(self, joint_q_out: wp.array, query_time: float) -> None:
        wp.launch(
            set_free_root_motion_waic_kernel,
            dim=1,
            inputs=[
                self.root_q_start,
                self.waic_robot_base_pos,
                self.scene_rotation,
                OLD_ROBOT_BASE_POS,
                wp.quat_identity(),
                self._base_motion_time(query_time),
                place.RETREAT_DISTANCE,
                place.RETREAT_TIME,
                place.math.radians(place.TURN_OUT_DEGREES),
                place.TURN_OUT_TIME,
                place.CENTER_SHIFT_Y,
                place.CENTER_MOVE_TIME,
                place.math.radians(place.TURN_IN_DEGREES),
                place.TURN_IN_TIME,
                place.APPROACH_DISTANCE,
                place.APPROACH_TIME,
            ],
            outputs=[joint_q_out],
            device=self.model.device,
        )

    def _root_motion_transform(self, query_time: float) -> wp.transform:
        return self._waic_transform(self._canonical_root_motion_transform(query_time))

    def _canonical_root_motion_transform(self, query_time: float) -> wp.transform:
        base_time = self._base_motion_time(query_time)

        def smooth(u: float) -> float:
            x = float(place.np.clip(u, 0.0, 1.0))
            return x * x * (3.0 - 2.0 * x)

        t0 = place.RETREAT_TIME
        t1 = t0 + place.TURN_OUT_TIME
        t2 = t1 + place.CENTER_MOVE_TIME
        t3 = t2 + place.TURN_IN_TIME
        retreat_u = smooth(base_time / max(place.RETREAT_TIME, 1.0e-6))
        turn_out_u = smooth((base_time - t0) / max(place.TURN_OUT_TIME, 1.0e-6))
        center_u = smooth((base_time - t1) / max(place.CENTER_MOVE_TIME, 1.0e-6))
        turn_in_u = smooth((base_time - t2) / max(place.TURN_IN_TIME, 1.0e-6))
        approach_u = smooth((base_time - t3) / max(place.APPROACH_TIME, 1.0e-6))

        yaw = place.math.radians(place.TURN_OUT_DEGREES) * turn_out_u + place.math.radians(place.TURN_IN_DEGREES) * turn_in_u
        old_q = wp.quat(0.0, 0.0, place.math.sin(0.5 * yaw), place.math.cos(0.5 * yaw))
        old_p = wp.vec3(
            float(OLD_ROBOT_BASE_POS[0]) - place.RETREAT_DISTANCE * retreat_u + place.APPROACH_DISTANCE * approach_u,
            float(OLD_ROBOT_BASE_POS[1]) + place.CENTER_SHIFT_Y * center_u,
            float(OLD_ROBOT_BASE_POS[2]),
        )
        return wp.transform(old_p, old_q)

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.waic_table_pos, self.scene_rotation),
            hx=place.TABLE_HALF_EXTENTS[0],
            hy=place.TABLE_HALF_EXTENTS[1],
            hz=place.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=place.TABLE_COLOR if self.show_physics_table else PHYSICS_TABLE_HIDDEN_COLOR,
            label="waic_hidden_tablecloth_physics_table",
        )
        builder.add_ground_plane(
            height=self.waic_ground_height,
            label="waic_shifted_ground_plane",
        )
        self.house_shape_start = builder.shape_count
        self._add_house_visual_model(builder)
        self.house_shape_end = builder.shape_count

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=self._waic_vec3(place.CLOTH_POS),
            rot=self.scene_rotation,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=place.CLOTH_DIM_X,
            dim_y=place.CLOTH_DIM_Y,
            cell_x=place.CLOTH_CELL_X,
            cell_y=place.CLOTH_CELL_Y,
            mass=0.003,
            tri_ke=7.0e2,
            tri_ka=7.0e2,
            tri_kd=5.0e-5,
            edge_ke=0.35,
            edge_kd=0.12,
            particle_radius=self.particle_radius,
            label="waic_bimanual_place_tablecloth",
        )

    def _configure_particle_contacts(self) -> None:
        super()._configure_particle_contacts()
        if self.house_shape_end <= self.house_shape_start:
            return
        flags = self.model.shape_flags.numpy()
        visual_only_mask = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[self.house_shape_start : self.house_shape_end] &= ~visual_only_mask
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _build_motion_segments(self):
        return super()._build_motion_segments()

    @staticmethod
    def create_parser():
        parser = place.Example.create_parser()
        parser.add_argument(
            "--waic-robot-base-x",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[0]),
            help="World X of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-y",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[1]),
            help="World Y of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-z",
            type=float,
            default=float(WAIC_ROBOT_BASE_POS[2]),
            help="World Z of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qx",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[0]),
            help="World quaternion X of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qy",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[1]),
            help="World quaternion Y of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qz",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[2]),
            help="World quaternion Z of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--waic-robot-base-qw",
            type=float,
            default=float(WAIC_ROBOT_BASE_QUAT[3]),
            help="World quaternion W of the current Blender W1 base anchor.",
        )
        parser.add_argument(
            "--show-physics-table",
            action="store_true",
            help="Render the hidden physics table box visibly for alignment/debugging.",
        )
        parser.add_argument(
            "--house-visual-usd",
            type=str,
            default=HOUSE_VISUAL_USD,
            help="USD visual background exported from the WAIC Blender house scene.",
        )
        parser.add_argument(
            "--hide-house-visual",
            action="store_true",
            help="Skip loading the WAIC house visual background in the viewer.",
        )
        parser.add_argument(
            "--house-visual-color-mode",
            choices=("texture", "material", "semantic", "gray"),
            default="texture",
            help="Color mode for the GL house background.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
