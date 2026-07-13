# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
"""DexSim-native W1 bimanual cloth grasp demo via Newton MJVBD.

This recreates Newton's ``example_cloth_dexforce_bimanual_grasp_cloth.py``
scene using DexSim scene APIs.  The W1 motion is generated in this file from
scripted dual-hand target poses, rather than loaded from an external npz cache.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import dexsim
import dexsim.render as R
import newton
import newton.ik as ik
import numpy as np
import warp as wp
from dexsim.engine.newton_physics import (
    NewtonCfg,
    NewtonClothAttr,
    get_newton_manager,
)
from dexsim.engine.newton_physics.solvers_cfg import MJVBDSolverCfg
from dexsim.types import (
    ActorType,
    ArticulationFlag,
    DriveType,
    PhysicalAttr,
    RigidBodyShape,
)
from dexsim.utility import Color as NvtxColor
from dexsim.utility import scope as nvtx_scope

REPO_ROOT = Path(__file__).resolve().parents[4]
W1_URDF = (
    REPO_ROOT
    / "resources"
    / "robots"
    / "W1-hand-obj"
    / "DexforceW1V021_visual_collision.urdf"
)

DEFAULT_DT = 1.0 / 60.0
DEFAULT_STEPS = 720
GROUND_SIZE = 8.0

TABLE_POS = np.array([0.60, 0.0, 1.15], dtype=np.float32)
TABLE_HALF_EXTENTS = np.array([0.32, 0.78, 0.025], dtype=np.float32)
TABLE_TOP_Z = float(TABLE_POS[2] + TABLE_HALF_EXTENTS[2])

CLOTH_DIM_X = 24
CLOTH_DIM_Y = 36
CLOTH_CELL_X = 0.022
CLOTH_CELL_Y = 0.025
CLOTH_COLLISION_RADIUS = 0.010
SOFT_CONTACT_MARGIN = 0.020
CLOTH_START_CLEARANCE = CLOTH_COLLISION_RADIUS + SOFT_CONTACT_MARGIN + 0.002
CLOTH_CENTER = np.array(
    [TABLE_POS[0], 0.0, TABLE_TOP_Z + CLOTH_START_CLEARANCE],
    dtype=np.float32,
)
CLOTH_POS = CLOTH_CENTER - np.array(
    [0.5 * CLOTH_DIM_X * CLOTH_CELL_X, 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y, 0.0],
    dtype=np.float32,
)
CLOTH_COLOR = np.array([0.78, 0.12, 0.10, 1.0], dtype=np.float32)

SOFT_CONTACT_KE = 3.0e5
SOFT_CONTACT_KD = 1.0e-4
SOFT_CONTACT_MU = 1.5
SELF_CONTACT_RADIUS = 0.010
SELF_CONTACT_MARGIN = 0.012
SOLVER_ITERATIONS = 24

RIGID_CONTACT_KE = 2.0e5
RIGID_CONTACT_KD = 1.0e-4
RIGID_CONTACT_MU = 2.0
HAND_CONTACT_KE = 3.0e5
HAND_CONTACT_KD = 1.0e-4
HAND_CONTACT_MU = 2.2

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)
RIGHT_APPROACH_POS = np.array([0.3363, -0.6980, 1.3628], dtype=np.float32)
RIGHT_APPROACH_ROT = np.array([0.0510, 0.7059, 0.7047, 0.0499], dtype=np.float32)
RIGHT_GRASP_POS = np.array([0.5897, -0.4651, 1.2055], dtype=np.float32)
RIGHT_GRASP_ROT = np.array([0.0245, 0.6878, 0.7139, -0.1294], dtype=np.float32)
GRASP_HEIGHT_OFFSET = 0.04
LIFT_HEIGHT = 0.22

ARM_JOINTS = (
    "LEFT_J1",
    "LEFT_J2",
    "LEFT_J3",
    "LEFT_J4",
    "LEFT_J5",
    "LEFT_J6",
    "LEFT_J7",
    "RIGHT_J1",
    "RIGHT_J2",
    "RIGHT_J3",
    "RIGHT_J4",
    "RIGHT_J5",
    "RIGHT_J6",
    "RIGHT_J7",
)
HAND_TARGETS = {
    "HAND_THUMB2": 0.84,
    "HAND_THUMB1": 0.46,
    "HAND_INDEX": 0.70,
    "INDEX_PIP": 0.90,
}
HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DexSim-native Dexforce W1 bimanual cloth grasp via Newton MJVBD."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ik-device",
        default="cpu",
        help="Warp device used for offline IK trajectory generation.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--ik-iterations", type=int, default=24)
    parser.add_argument("--static-w1", action="store_true")
    parser.add_argument("--disable-w1-collision", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument(
        "--debug-cloth-state",
        action="store_true",
        help="Print cloth particle finite/AABB diagnostics around each simulation step.",
    )
    parser.add_argument(
        "--debug-cloth-interval",
        type=int,
        default=30,
        help="Frame interval for --debug-cloth-state output.",
    )
    parser.add_argument(
        "--debug-cuda-sync",
        action="store_true",
        help="Force Warp CUDA synchronization before and after each update.",
    )
    parser.add_argument(
        "--debug-warp-verify",
        action="store_true",
        help="Enable Warp CUDA verification for the simulation loop. Experimental.",
    )
    return parser.parse_args()


def _setup_world(args: argparse.Namespace):
    config = dexsim.WorldConfig()
    config.open_windows = not args.headless
    config.renderer = dexsim.types.Renderer.HYBRID
    # config.backend = dexsim.types.Backend.VULKAN
    # config.raytrace_config.open_denoise = False
    # pp = config.postprocess_config
    # pp.bloom_enabled = False
    # pp.tone_mapping_enabled = False
    # pp.auto_exposure_enabled = False
    # pp.ca_enabled = False
    # pp.dof_enabled = False
    # pp.vignette_enabled = False


    cfg = NewtonCfg()
    cfg.device = args.device
    cfg.dt = args.dt
    cfg.num_substeps = 12
    cfg.use_cuda_graph = False
    cfg.solver_cfg = MJVBDSolverCfg(
        iterations=SOLVER_ITERATIONS,
        particle_enable_self_contact=True,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_topological_contact_filter_threshold=1,
        particle_rest_shape_contact_exclusion_radius=0.03,
        particle_vertex_contact_buffer_size=96,
        particle_edge_contact_buffer_size=128,
        particle_collision_detection_interval=-1,
        self_contact_bvh_rebuild_interval_frames=15,
        rigid_contact_max=0,
        step_rigid_bodies=False,
        soft_contact_margin=SOFT_CONTACT_MARGIN,
        soft_contact_ke=SOFT_CONTACT_KE,
        soft_contact_kd=SOFT_CONTACT_KD,
        soft_contact_mu=SOFT_CONTACT_MU,
    )
    config.newton_cfg = cfg

    world = dexsim.World(config)
    world.show_coordinate_axis(False)
    env = world.get_env()
    return world, env


def _make_contact_attr(mu: float, ke: float, kd: float) -> PhysicalAttr:
    attr = PhysicalAttr()
    attr.dynamic_friction = mu
    attr.static_friction = mu
    attr.contact_stiffness = ke
    attr.contact_damping = kd
    return attr


def _copy_physical_attr(src) -> PhysicalAttr:
    attr = PhysicalAttr()
    if src is None:
        return attr
    for name, value in src.as_dict().items():
        if hasattr(attr, name):
            setattr(attr, name, value)
    return attr


def _debug_sync_cuda(device: str, *, step: int, phase: str) -> None:
    try:
        if str(device).startswith("cuda"):
            wp.synchronize_device(device)
        else:
            wp.synchronize()
    except Exception as exc:
        print(
            f"[debug] CUDA/Warp sync failed at step={step + 1}, phase={phase}: {exc}",
            flush=True,
        )
        raise


def _debug_cloth_state(cloth, *, step: int, phase: str, force: bool) -> bool:
    try:
        positions = np.asarray(cloth.get_particle_positions(), dtype=np.float32)
    except Exception as exc:
        print(
            f"[debug] cloth particle read failed at step={step + 1}, phase={phase}: {exc}",
            flush=True,
        )
        raise

    if positions.size == 0:
        if force:
            print(f"[debug] step={step + 1} phase={phase} cloth has no particles", flush=True)
        return True

    finite = np.isfinite(positions)
    valid = bool(finite.all())
    if valid:
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        max_abs = float(np.max(np.abs(positions)))
        span = pmax - pmin
        if force:
            print(
                "[debug] "
                f"step={step + 1} phase={phase} "
                f"min={pmin.tolist()} max={pmax.tolist()} "
                f"span={span.tolist()} max_abs={max_abs:.6f}",
                flush=True,
            )
        return True

    bad_indices = np.flatnonzero(~finite.all(axis=1))[:8]
    print(
        "[debug] "
        f"INVALID cloth particles at step={step + 1} phase={phase} "
        f"bad_indices={bad_indices.tolist()}",
        flush=True,
    )
    return False


def _setup_render(world, env) -> None:
    env.set_env_light_intensity(90.0)

    key = env.create_light("key_light", R.LightType.POINT)
    if key is not None:
        key.set_location(1.8, -2.4, 4.0)
        key.set_intensity(160.0)
        key.set_shadow(False)

    fill = env.create_light("fill_light", R.LightType.POINT)
    if fill is not None:
        fill.set_location(-2.5, 2.0, 3.0)
        fill.set_intensity(60.0)
        fill.set_shadow(False)

    world.open_window()
    window = world.get_windows()
    if window is not None:
        window.set_look_at(
            eye=np.array([1.45, -1.65, 1.65], dtype=np.float32),
            look_at=np.array([0.55, 0.0, 1.12], dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )


def _create_ground(env):
    ground = env.create_plane(0.0, GROUND_SIZE)
    ground.set_name("ground_plane")
    ground.add_rigidbody(
        ActorType.STATIC,
        RigidBodyShape.PLANE,
        _make_contact_attr(RIGID_CONTACT_MU, RIGID_CONTACT_KE, RIGID_CONTACT_KD),
    )
    if hasattr(env, "create_color_material"):
        ground.set_material(env.create_color_material([0.35, 0.35, 0.35, 1.0], "ground_mat"))
    return ground


def _create_table(env):
    size = 2.0 * TABLE_HALF_EXTENTS
    table = env.create_cube(float(size[0]), float(size[1]), float(size[2]))
    table.set_name("cloth_table")
    table.set_location(float(TABLE_POS[0]), float(TABLE_POS[1]), float(TABLE_POS[2]))
    table.add_rigidbody(
        ActorType.STATIC,
        RigidBodyShape.BOX,
        _make_contact_attr(RIGID_CONTACT_MU, RIGID_CONTACT_KE, RIGID_CONTACT_KD),
    )
    if hasattr(env, "create_color_material"):
        table.set_material(env.create_color_material([0.35, 0.42, 0.48, 1.0], "table_mat"))
    return table


def _compute_grid_uv(mesh_obj) -> np.ndarray:
    vertices = np.asarray(mesh_obj.get_vertices(), dtype=np.float32)
    if vertices.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    uv = vertices[:, :2].copy()
    mins = uv.min(axis=0)
    spans = uv.max(axis=0) - mins
    spans = np.where(spans > 1.0e-8, spans, 1.0)
    uv = (uv - mins) / spans
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv.astype(np.float32)


def _create_cloth(env):
    cloth_mesh = env.create_grid_2d(
        start_point=CLOTH_POS.tolist(),
        width=CLOTH_DIM_X * CLOTH_CELL_X,
        height=CLOTH_DIM_Y * CLOTH_CELL_Y,
        rows=CLOTH_DIM_Y,
        cols=CLOTH_DIM_X,
    )
    cloth_mesh.set_name("bimanual_grasp_cloth_mesh")
    uv = _compute_grid_uv(cloth_mesh)
    if uv.shape[0] == np.asarray(cloth_mesh.get_vertices()).shape[0]:
        cloth_mesh.set_uv_mapping(uv)

    cloth_attr = NewtonClothAttr(
        mass=0.002,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=1.0e-5,
        edge_ke=1.0,
        edge_kd=0.05,
        particle_radius=CLOTH_COLLISION_RADIUS,
    )
    cloth = cloth_mesh.add_clothbody(cloth_attr, name="bimanual_grasp_cloth")
    if hasattr(env, "create_color_material"):
        material = env.create_color_material(CLOTH_COLOR.tolist(), "bimanual_grasp_cloth_mat")
        material.set_roughness(0.8)
        cloth.set_material(material)
    return cloth


def _load_w1(env, *, enable_collision: bool):
    if not W1_URDF.exists():
        raise FileNotFoundError(f"Dexforce W1 URDF not found: {W1_URDF}")

    w1 = env.load_urdf(
        str(W1_URDF),
        fix_root_link=True,
    )
    w1.set_articulation_flag(ArticulationFlag.FIX_BASE, True)
    w1.set_articulation_flag(ArticulationFlag.DISABLE_SELF_COLLISION, True)
    w1.enable_collision(enable_collision)

    for link_name in w1.get_link_names():
        attr = _copy_physical_attr(w1.get_physical_attr(link_name))
        attr.dynamic_friction = RIGID_CONTACT_MU
        attr.static_friction = RIGID_CONTACT_MU
        attr.contact_stiffness = RIGID_CONTACT_KE
        attr.contact_damping = RIGID_CONTACT_KD
        if any(keyword in link_name.lower() for keyword in HAND_CONTACT_KEYWORDS):
            attr.dynamic_friction = HAND_CONTACT_MU
            attr.static_friction = HAND_CONTACT_MU
            attr.contact_stiffness = HAND_CONTACT_KE
            attr.contact_damping = HAND_CONTACT_KD
        w1.set_physical_attr(attr, link_name=link_name)

    return w1


def _configure_w1_particle_only_collision(w1) -> None:
    mgr = getattr(w1, "_mgr", None)
    model = getattr(mgr, "_model", None)
    if model is None:
        return

    flags = model.shape_flags.numpy()
    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for link_meta in w1.dexsim_meta_links["links"].values():
        for shape_id in link_meta.shape_ids:
            flags[shape_id] |= collide_particles
            flags[shape_id] &= ~collide_shapes
    model.shape_flags.assign(flags)


def _short_name(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _slerp_quat_xyzw(quat_a: np.ndarray, quat_b: np.ndarray, alpha: float) -> np.ndarray:
    qa = _normalize_quat(quat_a)
    qb = _normalize_quat(quat_b)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quat(qa * (1.0 - alpha) + qb * alpha)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    scale_a = np.sin(theta_0 - theta) / sin_theta_0
    scale_b = np.sin(theta) / sin_theta_0
    return _normalize_quat(qa * scale_a + qb * scale_b)


def _mirror_y_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([pos[0], -pos[1], pos[2]], dtype=np.float32),
        _normalize_quat(np.array([-quat[0], quat[1], -quat[2], quat[3]], dtype=np.float32)),
    )


def _offset_pose(pos: np.ndarray, quat: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (pos + offset).astype(np.float32), _normalize_quat(quat)


def _interpolate_pose(
    start: tuple[np.ndarray, np.ndarray],
    end: tuple[np.ndarray, np.ndarray],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = start[0] * (1.0 - alpha) + end[0] * alpha
    quat = _slerp_quat_xyzw(start[1], end[1], alpha)
    return pos.astype(np.float32), quat


def _build_motion_segments(left_home, right_home):
    right_approach = (RIGHT_APPROACH_POS, _normalize_quat(RIGHT_APPROACH_ROT))
    left_approach = _mirror_y_pose(*right_approach)
    right_grasp = _offset_pose(
        RIGHT_GRASP_POS,
        RIGHT_GRASP_ROT,
        np.array([0.0, 0.0, GRASP_HEIGHT_OFFSET], dtype=np.float32),
    )
    left_grasp = _mirror_y_pose(*right_grasp)
    right_lift = _offset_pose(
        right_grasp[0],
        right_grasp[1],
        np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32),
    )
    left_lift = _offset_pose(
        left_grasp[0],
        left_grasp[1],
        np.array([0.0, 0.0, LIFT_HEIGHT], dtype=np.float32),
    )
    return (
        (0.8, left_home, left_home, right_home, right_home, 0.0, 0.0),
        (2.0, left_home, left_approach, right_home, right_approach, 0.0, 0.0),
        (2.0, left_approach, left_grasp, right_approach, right_grasp, 0.0, 0.0),
        (2.0, left_grasp, left_grasp, right_grasp, right_grasp, 0.0, 1.0),
        (3.0, left_grasp, left_lift, right_grasp, right_lift, 1.0, 1.0),
        (2.0, left_lift, left_lift, right_lift, right_lift, 1.0, 1.0),
    )


def _sample_script(segments, query_time: float):
    remaining = query_time
    for duration, left_start, left_end, right_start, right_end, grasp_start, grasp_end in segments:
        if remaining <= duration:
            alpha = float(np.clip(remaining / duration, 0.0, 1.0))
            left = _interpolate_pose(left_start, left_end, alpha)
            right = _interpolate_pose(right_start, right_end, alpha)
            grasp = grasp_start * (1.0 - alpha) + grasp_end * alpha
            return left, right, float(grasp)
        remaining -= duration

    _, _, left_end, _, right_end, _, grasp_end = segments[-1]
    return left_end, right_end, float(grasp_end)


def _quat_to_vec4(quat: np.ndarray) -> wp.vec4:
    quat = _normalize_quat(quat)
    return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _find_label_index(labels, name: str, kind: str) -> int:
    for index, label in enumerate(labels):
        if label == name or _short_name(str(label)) == name:
            return index
    raise ValueError(f"Newton {kind} not found: {name}")


def _current_tcp_pose(state, body_index: int, offset: wp.vec3) -> tuple[np.ndarray, np.ndarray]:
    body_q_np = state.body_q.numpy()
    body_tf = wp.transform(*body_q_np[body_index])
    body_pos = wp.transform_get_translation(body_tf)
    body_rot = wp.transform_get_rotation(body_tf)
    tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
    return (
        np.array([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])], dtype=np.float32),
        np.array([float(body_rot[0]), float(body_rot[1]), float(body_rot[2]), float(body_rot[3])], dtype=np.float32),
    )


def _build_ik_model() -> newton.Model:
    with nvtx_scope("W1Demo::IK::build_model", color=NvtxColor.ARTICULATION):
        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = RIGID_CONTACT_KE
        builder.default_shape_cfg.kd = RIGID_CONTACT_KD
        builder.default_shape_cfg.mu = RIGID_CONTACT_MU
        with nvtx_scope("W1Demo::IK::add_urdf", color=NvtxColor.ARTICULATION):
            builder.add_urdf(
                str(W1_URDF),
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                floating=False,
                hide_visuals=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                force_show_colliders=False,
            )
        with nvtx_scope("W1Demo::IK::builder_finalize", color=NvtxColor.ARTICULATION):
            return builder.finalize(requires_grad=False)


def _trajectory_cache_order(w1) -> list[str]:
    ordered_infos = w1._ordered_joint_infos()
    return [
        _short_name(info.name)
        for info in ordered_infos
        if w1.dexsim_meta_joints[info.name].is_active
    ]


def _remap_trajectory(
    trajectory: np.ndarray,
    source_joint_names: list[str],
    target_joint_names: list[str],
) -> np.ndarray:
    column_by_name = {
        _short_name(name): index for index, name in enumerate(source_joint_names)
    }
    missing = [name for name in target_joint_names if name not in column_by_name]
    if missing:
        raise ValueError(
            "W1 trajectory is missing required joints: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    remapped = np.empty((trajectory.shape[0], len(target_joint_names)), dtype=np.float32)
    for dst_index, joint_name in enumerate(target_joint_names):
        remapped[:, dst_index] = trajectory[:, column_by_name[joint_name]]
    return np.ascontiguousarray(remapped, dtype=np.float32)


def _trajectory_row_to_joint_order(
    trajectory_row: np.ndarray,
    source_joint_names: list[str],
    target_joint_names: list[str],
) -> np.ndarray:
    return _remap_trajectory(
        trajectory_row.reshape(1, -1),
        source_joint_names,
        [_short_name(name) for name in target_joint_names],
    )[0]


def _build_w1_joint_target_trajectory(
    args: argparse.Namespace,
    w1,
) -> tuple[np.ndarray, int, list[str]]:
    with nvtx_scope("W1Demo::IK::trajectory", color=NvtxColor.ARTICULATION):
        with nvtx_scope(f"W1Demo::IK::ScopedDevice({args.ik_device})",
                        color=NvtxColor.ARTICULATION):
            with wp.ScopedDevice(args.ik_device):
                return _build_w1_joint_target_trajectory_on_current_device(args, w1)


def _build_w1_joint_target_trajectory_on_current_device(
    args: argparse.Namespace,
    w1,
) -> tuple[np.ndarray, int, list[str]]:
    model = _build_ik_model()
    source_joint_names = [_short_name(str(label)) for label in model.joint_label]
    qpos = np.zeros(len(source_joint_names), dtype=np.float32)
    with nvtx_scope("W1Demo::IK::initial_fk", color=NvtxColor.ARTICULATION):
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    left_ee_index = _find_label_index(model.body_label, "left_j7", "body")
    right_ee_index = _find_label_index(model.body_label, "right_j7", "body")
    left_home = _current_tcp_pose(state, left_ee_index, TCP_OFFSET)
    right_home = _current_tcp_pose(state, right_ee_index, TCP_OFFSET)
    segments = _build_motion_segments(left_home, right_home)
    total_duration = sum(segment[0] for segment in segments)
    fps = max(1, int(round(1.0 / float(args.dt))))
    frame_count = max(args.steps, int(np.ceil(total_duration * fps)) + 1)

    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_q_home = model.joint_q.numpy()
    controlled_joint_names = set(ARM_JOINTS)
    locked_q_indices = []
    locked_q_values = []
    for joint_idx, label in enumerate(model.joint_label):
        short = _short_name(str(label))
        if short in controlled_joint_names:
            continue
        q_idx = int(joint_q_start[joint_idx])
        locked_q_indices.append(q_idx)
        locked_q_values.append(float(joint_q_home[q_idx]))

    hand_q_indices: list[int] = []
    hand_open_values: list[float] = []
    hand_grasp_values: list[float] = []
    joint_index_by_short = {
        _short_name(str(label)): joint_idx for joint_idx, label in enumerate(model.joint_label)
    }
    for side in ("LEFT", "RIGHT"):
        for suffix, close_value in HAND_TARGETS.items():
            joint_idx = joint_index_by_short.get(f"{side}_{suffix}")
            if joint_idx is None:
                continue
            q_idx = int(joint_q_start[joint_idx])
            hand_q_indices.append(q_idx)
            hand_open_values.append(float(joint_q_home[q_idx]))
            hand_grasp_values.append(float(close_value))

    lower = model.joint_limit_lower.numpy().copy()
    upper = model.joint_limit_upper.numpy().copy()
    for joint_idx, label in enumerate(model.joint_label):
        short = _short_name(str(label))
        if short in controlled_joint_names:
            continue
        q_idx = int(joint_q_start[joint_idx])
        dof_idx = int(joint_qd_start[joint_idx])
        lower[dof_idx] = float(joint_q_home[q_idx]) - 1.0e-4
        upper[dof_idx] = float(joint_q_home[q_idx]) + 1.0e-4

    with nvtx_scope("W1Demo::IK::create_solver", color=NvtxColor.ARTICULATION):
        ik_joint_q = wp.array(model.joint_q, shape=(1, model.joint_coord_count))
        left_pos_obj = ik.IKObjectivePosition(
            link_index=left_ee_index,
            link_offset=TCP_OFFSET,
            target_positions=wp.array([wp.vec3(*left_home[0].tolist())], dtype=wp.vec3),
        )
        left_rot_obj = ik.IKObjectiveRotation(
            link_index=left_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(left_home[1])], dtype=wp.vec4),
        )
        right_pos_obj = ik.IKObjectivePosition(
            link_index=right_ee_index,
            link_offset=TCP_OFFSET,
            target_positions=wp.array([wp.vec3(*right_home[0].tolist())], dtype=wp.vec3),
        )
        right_rot_obj = ik.IKObjectiveRotation(
            link_index=right_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([_quat_to_vec4(right_home[1])], dtype=wp.vec4),
        )
        limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=model.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=model.device),
            weight=25.0,
        )
        solver = ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[left_pos_obj, left_rot_obj, right_pos_obj, right_rot_obj, limit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    model_joint_to_source = []
    source_index_by_name = {
        name: index for index, name in enumerate(source_joint_names)
    }
    for joint_idx, label in enumerate(model.joint_label):
        source_index = source_index_by_name.get(_short_name(str(label)))
        if source_index is not None:
            model_joint_to_source.append((joint_idx, source_index))

    source_targets = np.zeros((frame_count, len(source_joint_names)), dtype=np.float32)
    with nvtx_scope("W1Demo::IK::sample_frames", color=NvtxColor.ARTICULATION):
        for frame in range(frame_count):
            query_time = min(frame / float(fps), total_duration)
            left_target, right_target, grasp_alpha = _sample_script(segments, query_time)

            left_pos_obj.set_target_position(0, wp.vec3(*left_target[0].tolist()))
            left_rot_obj.set_target_rotation(0, _quat_to_vec4(left_target[1]))
            right_pos_obj.set_target_position(0, wp.vec3(*right_target[0].tolist()))
            right_rot_obj.set_target_rotation(0, _quat_to_vec4(right_target[1]))
            with nvtx_scope("W1Demo::IK::solver_step", color=NvtxColor.ARTICULATION):
                solver.step(ik_joint_q, ik_joint_q, iterations=args.ik_iterations)
            with nvtx_scope("W1Demo::IK::synchronize", color=NvtxColor.GPU_TRANSFER):
                wp.synchronize()

            model_q = ik_joint_q.numpy()[0].astype(np.float32)
            if locked_q_indices:
                model_q[np.asarray(locked_q_indices, dtype=np.int32)] = np.asarray(
                    locked_q_values, dtype=np.float32
                )
            if hand_q_indices:
                hand_indices = np.asarray(hand_q_indices, dtype=np.int32)
                hand_open = np.asarray(hand_open_values, dtype=np.float32)
                hand_close = np.asarray(hand_grasp_values, dtype=np.float32)
                model_q[hand_indices] = hand_open * (1.0 - grasp_alpha) + hand_close * grasp_alpha
            ik_joint_q.assign(model_q.reshape(1, model.joint_coord_count))

            for joint_idx, source_index in model_joint_to_source:
                qpos[source_index] = model_q[int(joint_q_start[joint_idx])]
            source_targets[frame] = qpos

    target_joint_names = _trajectory_cache_order(w1)
    with nvtx_scope("W1Demo::IK::remap_trajectory", color=NvtxColor.ARTICULATION):
        return (
            _remap_trajectory(source_targets, source_joint_names, target_joint_names),
            fps,
            target_joint_names,
        )


def _configure_w1_drive(w1) -> None:
    dof = int(w1.get_dof())
    stiffness = np.full(dof, 650.0, dtype=np.float32)
    damping = np.full(dof, 65.0, dtype=np.float32)
    max_force = np.full(dof, 180.0, dtype=np.float32)
    max_velocity = np.full(dof, 4.0, dtype=np.float32)

    active_joint_names = list(w1.get_actived_joint_names(False))
    for i, name in enumerate(active_joint_names[:dof]):
        short = _short_name(name)
        if "HAND" in short or short.endswith("_PIP"):
            stiffness[i] = 950.0
            damping[i] = 75.0
            max_force[i] = 45.0

    w1.set_drive(
        stiffness=stiffness,
        damping=damping,
        max_force=max_force,
        max_velocity=max_velocity,
        drive_type=DriveType.FORCE,
    )


def run_demo(args: argparse.Namespace) -> None:
    world = None
    env = None
    w1 = None
    mgr = None
    cloth = None
    try:
        with nvtx_scope("W1Demo::setup_world", color=NvtxColor.NEWTON_SIM):
            world, env = _setup_world(args)
        if not args.headless:
            with nvtx_scope("W1Demo::setup_render", color=NvtxColor.RENDER):
                _setup_render(world, env)

        with nvtx_scope("W1Demo::create_scene", color=NvtxColor.NEWTON_CLOTH):
            _create_ground(env)
            _create_table(env)
            w1 = _load_w1(env, enable_collision=not args.disable_w1_collision)
            mgr = get_newton_manager(world)
            if mgr is not None and not args.disable_w1_collision:
                mgr.add_on_model_ready_callback(lambda: _configure_w1_particle_only_collision(w1))
            cloth = _create_cloth(env)
            _configure_w1_drive(w1)

        with nvtx_scope("W1Demo::build_trajectory", color=NvtxColor.ARTICULATION):
            trajectory, trajectory_fps, trajectory_joint_names = (
                _build_w1_joint_target_trajectory(args, w1)
            )
        if trajectory.shape[1] != int(w1.get_dof()):
            raise ValueError(
                "W1 trajectory DOF does not match DexSim articulation DOF: "
                f"trajectory={trajectory.shape[1]}, articulation={int(w1.get_dof())}"
            )

        zero_qvel = np.zeros(int(w1.get_dof()), dtype=np.float32)
        initial_api_qpos = _trajectory_row_to_joint_order(
            trajectory[0],
            trajectory_joint_names,
            list(w1.get_actived_joint_names(False)),
        )
        with nvtx_scope("W1Demo::install_trajectory", color=NvtxColor.JOINT):
            w1.set_current_qpos(initial_api_qpos)
            w1.set_current_qvel(zero_qvel)
            w1.set_target_qpos(initial_api_qpos)
            w1.set_target_qvel(zero_qvel)
            if not args.static_w1:
                w1.set_joint_target_trajectory(
                    trajectory,
                    fps=trajectory_fps,
                    mode="kinematic_substep",
                    hold_last=True,
                )

        print(
            "trajectory: generated-key-poses | "
            f"driver: {'static-w1' if args.static_w1 else 'kinematic_substep'} | "
            f"sim_device: {args.device} | ik_device: {args.ik_device} | "
            f"w1_collision: {not args.disable_w1_collision} | "
            f"frames: {len(trajectory)} | fps: {trajectory_fps} | "
            f"dof: {int(w1.get_dof())} | "
            f"cloth particles: {cloth.particle_count}",
            flush=True,
        )

        if args.debug_warp_verify:
            wp.config.verify_cuda = True
            print("[debug] Warp CUDA verification enabled for simulation loop", flush=True)

        start_time = time.perf_counter()
        for step in range(args.steps):
            should_debug_cloth = (
                args.debug_cloth_state
                and (step % max(1, args.debug_cloth_interval) == 0
                     or step == args.steps - 1)
            )
            if args.debug_cuda_sync:
                _debug_sync_cuda(args.device, step=step, phase="pre-update")
            if args.debug_cloth_state:
                _debug_cloth_state(
                    cloth,
                    step=step,
                    phase="pre-update",
                    force=should_debug_cloth,
                )

            with nvtx_scope("W1Demo::world_update", color=NvtxColor.NEWTON_SIM):
                world.update(args.dt)

            if args.debug_cuda_sync:
                _debug_sync_cuda(args.device, step=step, phase="post-update")
            if args.debug_cloth_state:
                _debug_cloth_state(
                    cloth,
                    step=step,
                    phase="post-update",
                    force=should_debug_cloth,
                )

            if step % 120 == 0 or step == args.steps - 1:
                elapsed = time.perf_counter() - start_time
                fps = (step + 1) / elapsed if elapsed > 0.0 else 0.0
                print(f"step={step + 1}/{args.steps}, fps={fps:.1f}", flush=True)
            if args.real_time:
                time.sleep(args.dt)
    finally:
        cloth = None
        mgr = None
        w1 = None
        env = None
        if world is not None:
            world.quit()


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()