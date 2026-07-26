# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# 版权声明；本文件用 SPDX 标记版权与许可证（Apache-2.0）。
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo Franka + Rigid Chain ADMM Pick-and-Place
#
# 【整体功能】固定基座的 Franka 机械臂按 IK 抓取-放置序列运动（MuJoCo 驱动），
# 同时一根短刚性负载链（默认用 XPBD 模拟）与之交互；原始的 VBD 线缆负载作为
# 备选模式用于 A/B 对比测试。SolverCoupledADMM 从模型碰撞对中检测机械臂与负载
# 之间的刚-刚接触，并把同一模板复制到多个世界以测试 ADMM 接触的并行扩展性。
#
# A fixed-base Franka arm tracks a pick-and-place IK sequence through MuJoCo
# position targets while a short rigid payload chain is simulated by XPBD by
# default. The original VBD cable payload is kept as an alternate mode for A/B
# testing. SolverCoupledADMM detects rigid-rigid contacts between the robot and
# the payload from the model collision pairs, and the same template is
# replicated across many worlds to exercise ADMM contact scaling.
#
# Command: python -m newton.examples mujoco_franka_vbd_cable_admm_solver
#
###########################################################################

# 启用 PEP 563 的"延迟注解求值"，让类型注解（如 Example 自引用）在运行时不立即求值。
from __future__ import annotations

# Callable 类型，用于 _capture_frame_graph 的回调参数注解。
from collections.abc import Callable

# numpy：用于关键帧数组、下标计算、断言等主机端数值处理。
import numpy as np
# Warp：Newton 的 GPU 计算后端，提供 array / kernel / capture 等。
import warp as wp
# 耦合求解器：SolverCoupled（基类）、SolverCoupledADMM（ADMM 耦合子类）。
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

# Newton 主包：提供 Model / ModelBuilder / CollisionPipeline / eval_fk / eval_ik 等。
import newton
# examples 子包：提供 init / run / create_parser 等示例运行框架与通用工具。
import newton.examples
# Newton 的逆运动学模块：IKSolver、各类 IKObjective、雅可比类型。
import newton.ik as ik
# Newton 工具：download_asset 下载机器人资产、create_straight_cable_points_and_quaternions 生成线缆几何。
import newton.utils
# 三个子求解器：MuJoCo（机械臂）、VBD（弹性杆/线缆）、XPBD（刚体链）。
from newton.solvers import SolverMuJoCo, SolverVBD, SolverXPBD

# 负载中心点（世界坐标）：x=0.5、y=0、z=0.256。机械臂与负载都在此附近交互。
PAYLOAD_CENTER = wp.vec3(0.5, 0.0, 0.256)
# 负载总长度 [m]，沿 +x 方向铺设。
PAYLOAD_LENGTH = 0.42

# 夹爪朝下的姿态：绕世界 x 轴转 180°，使手的 z 轴指向 -z（朝下抓物）。
# Top-down gripper orientation: 180 deg about world x flips the hand z-axis to -z.
# 注意 Newton 四元数顺序为 (qx, qy, qz, qw)，故 (1,0,0,0) 表示绕 x 转 180°。
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)  # (qx, qy, qz, qw)

# 夹爪张开时两手指关节目标位置 [m]（每个手指相对零位的位移）。
GRIP_OPEN = 0.04
# 夹爪完全闭合时的关节目标位置（0 表示手指合拢）。
GRIP_CLOSE = 0.0
# 抓持时的夹持宽度因子：乘以负载半径得到"持物"夹爪宽度；0 表示完全贴合。
GRIP_HOLD_FACTOR = 0.0
# 手指关节力矩上限 [N·m]，用于限制夹持力。
GRIP_FORCE = 1000.0
# 手指关节位置目标的 PD 刚度（target_ke）[N/m]。
GRIP_STIFFNESS = 1000.0

# 抬臂、张爪的初始关节构型：7 个 Franka 关节 + 2 个手指关节（张开）。
# Raised-arm, open-gripper starting configuration.
FRANKA_Q = [
    0.0,        # joint1
    -0.569,     # joint2
    0.0,        # joint3
    -2.810,     # joint4
    0.0,        # joint5
    3.037,      # joint6
    0.741,      # joint7
    GRIP_OPEN,  # 手指1（左）
    GRIP_OPEN,  # 手指2（右）
]


# GPU kernel：把"当前期望的手指宽度"写入每个世界的关节目标数组（两个手指同一宽度）。
# joint_q 形状 [world_count, n_coords]，idx0/idx1 为两手指在该坐标中的下标。
@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    world_idx = wp.tid()                                    # 当前世界索引
    joint_q[world_idx, idx0] = finger_pos[world_idx]       # 左手指目标 = 期望宽度
    joint_q[world_idx, idx1] = finger_pos[world_idx]       # 右手指目标 = 期望宽度


# GPU kernel：把插值得到的 IK 目标位姿（位置/姿态/夹爪宽度）广播写入每个世界。
# 在 update_ik_targets 里每帧调用一次，更新所有世界的目标缓冲。
@wp.kernel
def set_task_targets(
    target_positions: wp.array[wp.vec3],   # 输出：每世界目标位置
    target_rotations: wp.array[wp.vec4],   # 输出：每世界目标姿态四元数
    finger_pos: wp.array[float],           # 输出：每世界夹爪宽度
    pos: wp.vec3,                          # 输入：本帧插值目标位置
    rot: wp.vec4,                          # 输入：本帧插值目标姿态
    grip_width: float,                     # 输入：本帧插值夹爪宽度
):
    world_idx = wp.tid()                       # 当前世界索引
    target_positions[world_idx] = pos          # 广播位置
    target_rotations[world_idx] = rot          # 广播姿态
    finger_pos[world_idx] = grip_width         # 广播夹爪宽度


# 用 CUDA Graph 录制一帧的 simulate() 调用，返回录好的 graph 供后续重放。
# enabled=False 时直接返回 None（不录制）。录制需在 model.device 上进行。
def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool = True):
    if not enabled:                          # 未启用图捕获
        return None

    with wp.ScopedDevice(model.device):      # 切到模型所在设备（CPU/CUDA）
        with wp.ScopedCapture() as capture:  # 开始捕获一段 Warp 操作
            simulate()                       # 跑一次 simulate，所有 kernel 启动被录进 graph

    if capture.graph is None:                # 录制失败（例如设备不支持）
        raise RuntimeError(f"Graph capture failed on device {model.device}")
    return capture.graph                     # 返回可重放的图


# 重放已录制的帧 graph；返回是否成功重放。graph 为 None 时返回 False（调用方退回即时执行）。
def _launch_frame_graph(model: newton.Model, graph) -> bool:
    if graph is None:                        # 没有可用的图
        return False

    with wp.ScopedDevice(model.device):      # 切到模型设备
        wp.capture_launch(graph)             # 一键重放整帧的 kernel 序列
    return True


# 在标签列表中找以 suffix 结尾的标签，返回其下标；找不到则报错。
# 用于按名字定位 body（如 "fr3_hand"）。
def _find_label_index(labels: list[str], suffix: str) -> int:
    for index, label in enumerate(labels):
        if label.endswith(suffix):           # 后缀匹配（容忍前缀变化）
            return index
    raise ValueError(f"Could not find label ending in {suffix!r}")


# 示例主类：Newton 的 Example 框架约定类名必须为 Example，并实现 step/render/test_final 等。
class Example:
    # 构造函数：完成"建模 + 求解器 + 碰撞管线 + IK + 图捕获"的全部初始化。
    def __init__(self, viewer, args):
        self.viewer = viewer                 # 可视化查看器
        self.sim_time = 0.0                  # 当前仿真时间 [s]
        self.fps = 60                        # 渲染帧率
        self.frame_dt = 1.0 / self.fps       # 每帧时长 ≈ 16.67ms
        self.sim_substeps = max(1, int(args.substeps))   # 每帧的物理子步数（默认16）
        self.sim_dt = self.frame_dt / self.sim_substeps  # 每子步时长 ≈ 1.04ms
        self.use_graph = bool(args.graph_capture)        # 是否启用 CUDA Graph
        self.world_count = max(1, int(args.world_count)) # 并行世界数（默认8）
        self.payload_kind = str(args.payload_kind)       # 负载类型：xpbd-chain 或 vbd-cable
        self.payload_segments = max(2, int(args.payload_segments))  # 负载分段数（默认11）
        self.payload_radius = float(args.payload_radius)            # 负载半径 [m]（默认0.012）
        # 地面高度 = 负载中心 z − 负载半径，使负载恰好"躺"在地面上。
        self.surface_z = float(PAYLOAD_CENTER[2]) - self.payload_radius
        # 抓持时的夹爪宽度 = clamp(GRIP_HOLD_FACTOR*radius, GRIP_CLOSE, GRIP_OPEN)。
        # 当前因子为 0，所以 grip_hold=0（完全贴合抓物）。
        self.grip_hold = min(GRIP_OPEN, max(GRIP_CLOSE, GRIP_HOLD_FACTOR * self.payload_radius))

        # ---- 1) 构建单世界模板 ----
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))  # 带重力的建模器
        template.rigid_gap = 0.005            # 刚体碰撞间隙（margin 之外的额外间隙）
        SolverMuJoCo.register_custom_attributes(template)          # 注册 MuJoCo 需要的自定义属性（如 gravcomp）
        if self.payload_kind == "vbd-cable":  # 仅 VBD 模式需要注册其自定义属性
            SolverVBD.register_custom_attributes(template, dahl_defaults_enabled=False)
        self._emit_template(template)         # 往模板里加 Franka + 负载，并记录所有权区间

        # 记录单世界规模，供后续把局部下标展开为跨世界全局下标用。
        bodies_per_world = template.body_count
        joints_per_world = template.joint_count
        shapes_per_world = template.shape_count

        # ---- 2) 复制模板成多世界模型 ----
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))  # 主建模器
        builder.replicate(template, world_count=self.world_count) # 把模板复制 N 份
        self._expand_world_indices(bodies_per_world, joints_per_world, shapes_per_world)  # 局部下标→全局下标
        self.ground_shapes = [self._emit_ground_plane(builder)]   # 在每世界同高度加一个地面平面

        builder.color()                       # 给模型上色（便于可视化区分）
        self.model = builder.finalize()       # 冻结模型，分配 GPU 缓冲
        self.device = self.model.device       # 模型所在设备
        self.use_graph = self.use_graph and self.device.is_cuda  # Graph 仅在 CUDA 上启用
        self._count_admm_shape_pairs_per_world()                 # 统计每世界 Franka-负载接触对数（用于校验）

        # ---- 3) 构建 ADMM 耦合求解器 ----
        mujoco_contact_budget = max(64, 16 * self.world_count)   # MuJoCo 接触容量预算（按世界数缩放）
        payload_name = "vbd" if self.payload_kind == "vbd-cable" else "xpbd"  # 负载 entry 名
        payload_solver = self._make_payload_solver(args)         # 负载求解器工厂（lambda）
        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                # Entry 1：MuJoCo 驱动的 Franka 机械臂。
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda v: SolverMuJoCo(
                        model=v,
                        solver="newton",          # MuJoCo 求解器类型
                        integrator="implicitfast",# 隐式快速积分器
                        iterations=int(args.mujoco_iterations),      # 求解迭代数（默认12）
                        ls_iterations=int(args.mujoco_ls_iterations),# 线搜索迭代数（默认25）
                        use_mujoco_contacts=False, # 【关键】MuJoCo 不自处理接触，交由耦合器！
                        njmax=max(256, 64 * self.world_count),      # 最大 Jacobian 行数
                        nconmax=mujoco_contact_budget,              # 最大接触数
                    ),
                    bodies=self.franka_bodies,   # 该 entry 拥有的 body
                    joints=self.franka_joints,   # 该 entry 拥有的 joint
                ),
                # Entry 2：VBD 或 XPBD 驱动的负载。
                SolverCoupled.Entry(
                    name=payload_name,
                    solver=payload_solver,
                    bodies=self.payload_bodies,
                    joints=self.payload_joints,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=int(args.admm_iterations),   # 每子步 ADMM 迭代轮数（默认5）
                rho=float(args.rho),                     # ADMM 罚参数 ρ（默认200）
                gamma=float(args.gamma),                 # 近端质量缩放 γ（默认0.001）
                baumgarte=float(args.baumgarte),         # 位置误差修正比例 β（默认0.5）
                rigid_contact_matching=str(args.rigid_contact_matching),           # 跨帧接触匹配模式
                contact_matching_pos_threshold=args.contact_matching_pos_threshold,   # 匹配中点距离阈值
                contact_matching_normal_dot_threshold=args.contact_matching_normal_dot_threshold, # 法向点积阈值
                contact_matching_force_scale=args.contact_matching_force_scale,       # warm-start λ 缩放
                # 启用 mjc ↔ payload 之间的刚-刚接触耦合。
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="mjc",
                        destination=payload_name,
                    ),
                ],
            ),
        )

        # ---- 4) 状态、碰撞管线、控制、IK ----
        self.state_0 = self.model.state()     # 输入/当前状态（双缓冲 ping-pong）
        self.state_1 = self.model.state()     # 输出状态
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,                        # 用户主管线：检测全量接触（同域+跨域）
        )
        self.contacts = self.collision_pipeline.contacts()  # 分配接触缓冲
        self.solver.prepare_contacts(self.contacts)         # 为每 entry 预分配过滤后接触缓冲（图捕获友好）
        self.control = self.model.control()    # 控制输入（关节位置目标等）
        self._build_keyframes()                # 构建抓取-放置关键帧序列
        self._build_ik()                       # 构建 IK 模型与求解器

        # ---- 5) 可视化配置 ----
        newton.examples.configure_coupled_view(self, args)  # 配置耦合视图（高亮接触等）
        self.viewer.set_world_offsets((1.1, 1.1, 0.0))      # 各世界在视图中的间距偏移
        if isinstance(self.viewer, newton.viewer.ViewerGL): # GL 查看器额外调相机
            scale = max(1.0, float(np.sqrt(self.world_count)))  # 相机距离随世界数增大
            self.viewer.set_camera(pos=wp.vec3(0.9 * scale, -1.7 * scale, 0.95 * scale), pitch=-18.0, yaw=120.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.45, 0.0, 0.28))  # 看向负载中心附近

        # 用初始关节角做一次正向运动学，把 body 状态填进 state_0/state_1。
        # （此处 eval_ik 实为 eval_fk 的别名/封装：由关节角推 body 位姿）
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.capture()                         # 录制 CUDA Graph（若启用）

    # 根据负载类型返回对应的子求解器工厂（lambda，接受模型视图 v）。
    def _make_payload_solver(self, args):
        if self.payload_kind == "vbd-cable":            # VBD 弹性杆/线缆
            vbd_iterations = int(args.vbd_iterations)
            return lambda v: SolverVBD(
                model=v,
                iterations=vbd_iterations,
                rigid_contact_history=False,             # 不保留刚体接触历史（耦合器自管）
            )
        if self.payload_kind == "xpbd-chain":           # XPBD 刚体链（默认）
            xpbd_iterations = int(args.xpbd_iterations)
            joint_linear_relaxation = float(args.xpbd_joint_linear_relaxation)
            joint_angular_relaxation = float(args.xpbd_joint_angular_relaxation)
            return lambda v: SolverXPBD(
                model=v,
                iterations=xpbd_iterations,
                joint_linear_relaxation=joint_linear_relaxation,  # 关节线松弛
                joint_angular_relaxation=joint_angular_relaxation,# 关节角松弛
                angular_damping=0.02,                            # 角阻尼，稳定链摆
            )
        raise ValueError(f"Unsupported payload kind {self.payload_kind!r}")

    # 往模板 builder 里排放 Franka + 负载，并记录各部件在单世界内的下标区间与所有权。
    def _emit_template(self, builder: newton.ModelBuilder) -> None:
        # 记录 Franka 加入前的 body/joint/shape 数量，作为其起始下标。
        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        franka_shape_start = builder.shape_count

        self._add_franka(builder, self.surface_z)   # 加载 Franka URDF
        # 设置 Franka 7 个旋转关节的位置目标 PD 增益（ke=刚度, kd=阻尼）。
        builder.joint_target_ke[:7] = [900.0] * 7
        builder.joint_target_kd[:7] = [90.0] * 7
        # 两个手指关节的 PD 增益（更硬，以便稳定夹持）。
        builder.joint_target_ke[7:9] = [GRIP_STIFFNESS, GRIP_STIFFNESS]
        builder.joint_target_kd[7:9] = [100.0, 100.0]
        # 关节力矩上限：7 臂关节 80 N·m，2 手指 1000 N·m。
        builder.joint_effort_limit[:7] = [80.0] * 7
        builder.joint_effort_limit[7:9] = [GRIP_FORCE, GRIP_FORCE]
        # 关节转子惯量（armature）：臂关节 0.05 提高数值稳定性，手指 0。
        builder.joint_armature[:7] = [0.05] * 7
        builder.joint_armature[7:9] = [0.0, 0.0]

        # 记录 Franka 加入后的数量，得到其结束下标。
        franka_body_end = builder.body_count
        franka_joint_end = builder.joint_count
        franka_shape_end = builder.shape_count
        franka_bodies = list(range(franka_body_start, franka_body_end))   # Franka 的 body 列表

        # 为每个 Franka body 开启 MuJoCo 重力补偿，抵消自重、减小 IK 跟踪误差。
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in franka_bodies:
            gravcomp.values[body] = 1.0       # 1.0 = 完全补偿该 body 所受重力

        # 记录负载加入前的 shape 数量，作为负载 shape 起始下标。
        payload_shape_start = builder.shape_count
        if self.payload_kind == "vbd-cable":
            payload_bodies, payload_joints = self._emit_vbd_cable(builder)   # VBD 线缆
        else:
            payload_bodies, payload_joints = self._emit_xpbd_chain(builder)  # XPBD 链（默认）

        # 把单世界内的所有权区间存为实例属性（后面 _expand_world_indices 会展开为全局）。
        self.franka_bodies = franka_bodies
        self.franka_joints = list(range(franka_joint_start, franka_joint_end))
        self.franka_shapes = list(range(franka_shape_start, franka_shape_end))
        self.payload_bodies = payload_bodies
        self.payload_joints = payload_joints
        self.payload_shapes = list(range(payload_shape_start, builder.shape_count))
        self.payload_body_count_per_world = len(payload_bodies)        # 每世界负载体数
        self.payload_mid_body_offset = self.payload_body_count_per_world // 2  # 中段偏移（可视化/抓取用）

    # 静态方法：加载 Franka FR3 + 手的 URDF，设置初始关节角与目标。
    @staticmethod
    def _add_franka(builder: newton.ModelBuilder, base_z: float) -> None:
        builder.add_urdf(
            # 下载（必要时）franka 资产并加载其 URDF。
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            # 基座位姿：放在 (0,0,base_z)，无旋转。
            xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
            floating=False,                  # 固定基座（不浮动）
            enable_self_collisions=False,    # 关闭自碰撞（简化）
            parse_visuals_as_colliders=False,# 可视化网格不作碰撞体
            force_show_colliders=False,      # 不强制显示碰撞体
        )
        # 设置初始关节角（前7臂关节+2手指）。
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        # 初始位置目标 = 初始关节角（静止起步）。
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

    # 在地面高度 surface_z 加一个无限大平面作为负载支撑面，返回其 shape 下标。
    def _emit_ground_plane(self, builder: newton.ModelBuilder) -> int:
        # 地面材质：ke=接触刚度, kd=接触阻尼, mu=摩擦系数, margin/gap=碰撞余量。
        plane_cfg = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=2.0e1, mu=0.8, margin=0.001, gap=0.002)
        return builder.add_ground_plane(
            height=self.surface_z,           # 平面高度
            cfg=plane_cfg,
            label="payload_ground_plane",    # 标签，便于后续按名查找
        )

    # 生成 VBD 弹性杆（线缆）负载：返回 (body 列表, joint 列表)。
    def _emit_vbd_cable(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        stretch_stiffness = 2.0e5            # 拉伸刚度（很大，近似不可拉伸）
        bend_stiffness = 0.08                # 弯曲刚度（较小，可弯）
        # 线缆碰撞材质：密度、接触刚度/阻尼、摩擦、余量。
        cable_cfg = newton.ModelBuilder.ShapeConfig(
            density=1400.0, ke=5.0e4, kd=1.0e1, mu=0.9, margin=0.001, gap=0.002,
        )
        # 生成一条直线的离散点与朝向：从负载中心左侧沿 +x 铺设 PAYLOAD_LENGTH 长。
        points, quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=PAYLOAD_CENTER - wp.vec3(0.5 * PAYLOAD_LENGTH, 0.0, 0.0),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=PAYLOAD_LENGTH,
            num_segments=self.payload_segments,
            twist_total=0.0,                 # 无初始扭转
        )
        # add_rod 返回 (bodies, joints)：建立离散弹性杆模型。
        return builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=self.payload_radius,       # 杆半径
            body_frame_origin="start",        # body 坐标原点取在起点
            cfg=cable_cfg,
            stretch_stiffness=stretch_stiffness,
            stretch_damping=2.0e-2,           # 拉伸阻尼
            bend_stiffness=bend_stiffness,
            bend_damping=2.0e-2 * bend_stiffness,  # 弯曲阻尼（按刚度缩放）
            label="vbd_cable",
        )

    # 生成 XPBD 刚体胶囊链负载：返回 (body 列表, joint 列表)。默认模式。
    def _emit_xpbd_chain(self, builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
        chain_length = PAYLOAD_LENGTH                          # 链总长
        segment_length = chain_length / float(self.payload_segments)  # 每段长
        segment_half_length = 0.5 * segment_length
        # 胶囊半高：至少 0.25*半径，且不超过"半段长−半径"（避免相邻胶囊重叠）。
        capsule_half_height = max(0.25 * self.payload_radius, segment_half_length - self.payload_radius)
        # 链起点：负载中心左侧，z 比中心略低 2mm（贴地）。
        start = PAYLOAD_CENTER - wp.vec3(0.5 * PAYLOAD_LENGTH, 0.0, -0.002)
        direction = wp.vec3(1.0, 0.0, 0.0)                     # 铺设方向 +x
        # 胶囊朝向：绕 y 轴转 90°，使胶囊长轴沿 x（与链方向一致）。
        capsule_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * wp.pi)
        shape_xform = wp.transform(p=wp.vec3(0.0), q=capsule_rot)  # shape 相对 body 的变换
        # 链节碰撞材质：密度比线缆小，接触刚度/阻尼/摩擦/余量。
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=900.0, ke=6.0e4, kd=1.5e1, mu=0.9, margin=0.001, gap=0.002,
        )

        bodies = []                              # 收集每段 body
        joints = []                              # 收集每段 joint
        for segment in range(self.payload_segments):
            # 每段中心 = 起点 + (段号+0.5)*段长，沿 +x。
            center = start + direction * ((float(segment) + 0.5) * segment_length)
            body = builder.add_link(
                xform=wp.transform(p=center, q=wp.quat_identity()),  # 无旋转的链节
                label=f"xpbd_chain_link_{segment}",
            )
            # 给该链节加一个胶囊碰撞体。
            builder.add_shape_capsule(
                body,
                xform=shape_xform,
                radius=self.payload_radius,
                half_height=capsule_half_height,
                cfg=shape_cfg,
                label=f"xpbd_chain_capsule_{segment}",
            )
            bodies.append(body)
            if segment == 0:
                # 第一段：自由关节（6 自由度），让整条链可以自由运动。
                joints.append(builder.add_joint_free(child=body, label="xpbd_chain_root"))
                continue

            # 后续段：用球关节（3 自由度转动）连到上一段，形成链条。
            joints.append(
                builder.add_joint_ball(
                    parent=bodies[segment - 1],
                    child=body,
                    friction=0.02,              # 关节摩擦
                    # 关节在 parent 坐标系下的位置：沿 +x 偏移半段长（链节右端）。
                    parent_xform=wp.transform(p=wp.vec3(segment_half_length, 0.0, 0.0), q=wp.quat_identity()),
                    # 关节在 child 坐标系下的位置：沿 -x 偏移半段长（链节左端）。
                    child_xform=wp.transform(p=wp.vec3(-segment_half_length, 0.0, 0.0), q=wp.quat_identity()),
                    collision_filter_parent=True,  # 过滤相邻链节间的碰撞（避免抖动）
                    label=f"xpbd_chain_joint_{segment - 1}_{segment}",
                )
            )

        # 把所有关节组成一个 articulation（铰接链），便于 XPBD 统一求解。
        builder.add_articulation(joints, label="xpbd_chain")
        return bodies, joints

    # 把单世界内的局部下标展开为跨世界的全局下标：global = world*stride + local。
    def _expand_world_indices(self, bodies_per_world: int, joints_per_world: int, shapes_per_world: int) -> None:
        def expand(ids: list[int], stride: int) -> list[int]:
            # 对每个世界，把 ids 中每个局部下标加上 world*stride，得到全局下标。
            return [world * stride + id_ for world in range(self.world_count) for id_ in ids]

        self.franka_bodies = expand(self.franka_bodies, bodies_per_world)
        self.franka_joints = expand(self.franka_joints, joints_per_world)
        self.franka_shapes = expand(self.franka_shapes, shapes_per_world)
        self.payload_bodies = expand(self.payload_bodies, bodies_per_world)
        self.payload_joints = expand(self.payload_joints, joints_per_world)
        self.payload_shapes = expand(self.payload_shapes, shapes_per_world)

    # 统计每个世界里"Franka ↔ 负载"的 ADMM 接触对数，用于校验多世界复制正确性。
    def _count_admm_shape_pairs_per_world(self) -> None:
        shape_body = self.model.shape_body.numpy()      # shape → 所属 body
        shape_world = self.model.shape_world.numpy()    # shape → 所属 world
        franka_bodies = set(self.franka_bodies)         # 转集合加速查找
        payload_bodies = set(self.payload_bodies)
        counts = np.zeros(self.world_count, dtype=np.int32)  # 每世界计数

        for pair in self.model.shape_contact_pairs.numpy():  # 遍历所有可碰撞 shape 对
            shape_a = int(pair[0])
            shape_b = int(pair[1])
            body_a = int(shape_body[shape_a])
            body_b = int(shape_body[shape_b])
            owner_a = self._body_owner(body_a, franka_bodies, payload_bodies)  # shape_a 的所有者
            owner_b = self._body_owner(body_b, franka_bodies, payload_bodies)  # shape_b 的所有者
            # 只统计跨所有者（一方 mjc、一方 payload）的接触对——这才是 ADMM 要管的。
            if {owner_a, owner_b} != {"mjc", "payload"}:
                continue
            world_a = int(shape_world[shape_a])
            world_b = int(shape_world[shape_b])
            if world_a != world_b:                     # 跨世界的接触对是错误（不允许）
                raise RuntimeError("Cross-world Franka-payload contact pair was generated")
            if 0 <= world_a < self.world_count:
                counts[world_a] += 1                   # 该世界计数+1

        self.admm_shape_pairs_per_world = counts       # 保存，供 test_final 断言

    # 静态方法：判断一个 body 属于谁——mjc / payload / None（其他，如地面）。
    @staticmethod
    def _body_owner(body: int, franka_bodies: set[int], payload_bodies: set[int]) -> str | None:
        if body in franka_bodies:
            return "mjc"
        if body in payload_bodies:
            return "payload"
        return None

    # 收集"负载 ↔ 地面"的接触 shape 对（本例中地面也参与碰撞）。返回 vec2i 数组。
    # 注：本例主流程未直接调用此方法，但保留以便扩展/调试使用。
    def _payload_ground_shape_pairs(self) -> wp.array:
        payload_shapes = set(self.payload_shapes)
        ground_shapes = set(self.ground_shapes)
        # 筛出同时含负载 shape 和地面 shape 的接触对。
        pairs = [
            (shape_a, shape_b)
            for shape_a, shape_b in self.model.shape_contact_pairs.numpy()
            if ({int(shape_a), int(shape_b)} & payload_shapes) and ({int(shape_a), int(shape_b)} & ground_shapes)
        ]
        if not pairs:
            raise RuntimeError("No payload-ground contact pairs were generated")
        return wp.array(np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.model.device)

    # 构建逆运动学（IK）模型与求解器。关键：IK 跑在"只有 Franka"的独立模型上，
    # 这样负载坐标不会进入 IK 求解，机械臂目标完全由关键帧决定。
    def _build_ik(self) -> None:
        # IK runs on a Franka-only model so payload coordinates do not enter the solve.
        ik_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))  # 独立建模器
        self._add_franka(ik_builder, self.surface_z)                 # 只加 Franka
        self.ik_model = ik_builder.finalize(device=self.device)       # 冻结 IK 模型

        self.n_coords = self.ik_model.joint_coord_count              # Franka 关节坐标数（7+2=9）
        # 从主模型关节角切出每世界前 n_coords 个，作为 IK 输入/输出缓冲（每世界一份）。
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, : self.n_coords])
        # 控制目标的视图：每世界的关节位置目标，前 n_coords 列对应 Franka。
        self.control_joint_target_q = self.control.joint_target_q.reshape((self.world_count, -1))
        self.finger_idx0 = self.n_coords - 2    # 左手指在坐标中的下标
        self.finger_idx1 = self.n_coords - 1    # 右手指在坐标中的下标
        # 每世界的手指目标宽度缓冲，初值张开。
        self.finger_pos_buf = wp.full(self.world_count, GRIP_OPEN, dtype=float, device=self.device)
        # 按标签找"fr3_hand"（末端手爪 body）的下标。
        hand_body = _find_label_index(self.ik_model.body_label, "fr3_hand")

        # 用第一个关键帧初始化目标位姿，并广播到所有世界。
        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())
        self.ik_target_positions = wp.array([target_pos] * self.world_count, dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array([target_rot] * self.world_count, dtype=wp.vec4, device=self.device)

        # 位置目标：让 hand_body 上偏移 z+0.107 处（TCP 点）到达目标位置。
        self.pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),   # TCP 相对手的偏移
            target_positions=self.ik_target_positions,
        )
        # 姿态目标：让 hand_body 的姿态（无额外偏移旋转）对齐目标姿态。
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        # 关节限位目标：从主模型取出每世界关节限位，作为软约束（权重10）防止超限。
        joint_limit_lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, : self.n_coords])
        joint_limit_upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, : self.n_coords])
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=joint_limit_lower.flatten(),
            joint_limit_upper=joint_limit_upper.flatten(),
            weight=10.0,
        )
        # IK 求解器：每世界一个独立问题，三个目标，解析雅可比，LM 初始 λ=0.05。
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=self.world_count,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24                      # 每次 IK 求解的迭代数

    # 构建抓取-放置关键帧序列：10 段位姿，每段带持续时间。
    def _build_keyframes(self) -> None:
        cx, cy, cz = PAYLOAD_CENTER[0], PAYLOAD_CENTER[1], PAYLOAD_CENTER[2]
        approach_z = cz + 0.20                  # 接近高度（高于负载 20cm）
        grasp_z = cz                            # 抓取高度（负载所在高度）
        tx, ty = 0.4, 0.25                      # 放置目标 xy

        start_pos, start_rot = self._initial_tcp_pose()  # 初始 TCP 位姿（由当前关节角 FK 得到）
        qx, qy, qz, qw = GRIPPER_DOWN           # 朝下姿态
        self.place_target_xy = (tx, ty)         # 记录放置目标（可供外部查询）
        # 每行 = [持续时间, x, y, z, qx, qy, qz, qw, 夹爪宽度]
        # 流程：起始→接近负载→下降→闭合抓持→抬起→移到放置点→下降→保持→松开→抬起
        poses = np.array(
            [
                [0.25, *start_pos.tolist(), *start_rot.tolist(), GRIP_OPEN],          # 起始
                [0.5, cx, cy, approach_z, qx, qy, qz, qw, GRIP_OPEN],                 # 接近负载上方
                [0.5, cx, cy, grasp_z, qx, qy, qz, qw, GRIP_OPEN],                    # 下降到负载
                [1.0, cx, cy, grasp_z, qx, qy, qz, qw, self.grip_hold],               # 闭合抓持
                [1.0, cx, cy, approach_z, qx, qy, qz, qw, self.grip_hold],            # 抬起负载
                [1.0, tx, ty, approach_z, qx, qy, qz, qw, self.grip_hold],            # 移到放置点上方
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, self.grip_hold],               # 下降到放置
                [0.5, tx, ty, grasp_z, qx, qy, qz, qw, self.grip_hold],               # 保持（稳定放置）
                [1.0, tx, ty, grasp_z, qx, qy, qz, qw, GRIP_OPEN],                    # 松开
                [0.5, tx, ty, approach_z, qx, qy, qz, qw, GRIP_OPEN],                 # 抬起离开
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]             # 去掉持续时间列，只留位姿+夹爪
        self.key_times = np.cumsum(poses[:, 0]) # 累加持续时间得到各关键帧时间点

    # 由当前关节角做正向运动学，得到末端 TCP 的初始位姿（位置+四元数）。
    def _initial_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)  # FK
        hand_body = _find_label_index(self.model.body_label, "fr3_hand")            # 手 body 下标
        hand_q = state.body_q.numpy()[hand_body]    # body 位姿 [x,y,z, qx,qy,qz,qw]

        pos = wp.vec3(float(hand_q[0]), float(hand_q[1]), float(hand_q[2]))        # 位置
        rot = wp.quat(float(hand_q[3]), float(hand_q[4]), float(hand_q[5]), float(hand_q[6]))  # 姿态
        # TCP 位置 = 手位置 + 手姿态旋转 (0,0,0.107)。
        tcp_pos = pos + wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 0.107))

        return (
            np.array([float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])], dtype=np.float32),
            np.array([float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])], dtype=np.float32),
        )

    # 每帧调用：按当前 sim_time 在关键帧间线性插值，并写入 IK 目标缓冲（图启动前）。
    def update_ik_targets(self) -> None:
        """Interpolate keyframes and update IK target arrays before graph launch."""
        # 把 sim_time 限制在最后一个关键帧之前（避免越界）。
        t = min(self.sim_time, float(self.key_times[-1]) - 1.0e-6)
        # 二分查找当前时间落在哪个区间。
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0  # 区间起点
        t_end = self.key_times[interval]                                 # 区间终点
        # 插值因子 alpha ∈ [0,1]。
        alpha = float(np.clip((t - t_start) / max(t_end - t_start, 1.0e-6), 0.0, 1.0))

        cur = self.targets[interval]                    # 当前帧目标
        prev = self.targets[interval - 1] if interval > 0 else cur  # 上一帧目标
        interp = (1.0 - alpha) * prev + alpha * cur     # 线性插值

        # 用 kernel 把插值结果广播到每个世界的目标缓冲。
        wp.launch(
            set_task_targets,
            dim=self.world_count,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_pos_buf,
                wp.vec3(*interp[:3].tolist()),          # 插值位置
                wp.vec4(*interp[3:7].tolist()),         # 插值姿态
                float(interp[-1]),                      # 插值夹爪宽度
            ],
            device=self.device,
        )

    # 录制（或重录）CUDA Graph：把 simulate() 录成一个可重放的图。
    def capture(self):
        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=self.use_graph)

    # 一帧的物理仿真主体：IK 求解 + 写控制 + 多子步（碰撞+ADMM求解+FK）。
    def simulate(self):
        # 1) IK 求解：根据目标位姿解出 Franka 关节角（in/out 同一缓冲，原地迭代）。
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        # 2) 把期望夹爪宽度写入 IK 解的两手指下标。
        wp.launch(
            set_gripper_q,
            dim=self.world_count,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.device,
        )
        # 3) 把 IK 解出的关节角拷到控制缓冲，作为 MuJoCo 的位置目标。
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)

        # 4) 子步循环：每帧拆成 sim_substeps 个小步，提高碰撞/耦合稳定性。
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()                 # 清空本步外力
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)  # 施加鼠标交互力
            # 全量碰撞检测（用户主管线），结果写 self.contacts。
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            # ADMM 耦合求解一步：内含跨求解器接触重算 + 多轮 ADMM 迭代 + 各子求解器步进。
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # 由 body 状态反推关节坐标，保证下一子步输入一致（eval_ik 实为 FK 回填）。
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0  # ping-pong 交换缓冲

    # 每渲染帧入口：更新 IK 目标 → 重放图（或即时 simulate）→ 推进时间。
    def step(self):
        self.update_ik_targets()                        # 插值本帧目标
        if not _launch_frame_graph(self.model, self.graph):  # 优先重放图
            self.simulate()                             # 无图则即时执行
        self.sim_time += self.frame_dt                  # 推进仿真时间

    # 仿真结束后的验证：确保状态有效、耦合在所有世界都生效。
    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        # 位置/速度无 NaN/Inf，说明数值稳定。
        assert np.all(np.isfinite(body_q)), "Body positions contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "Body velocities contain NaN or inf values"
        # 每个世界都应有 Franka-负载 ADMM 接触对（耦合确实生效）。
        assert np.all(self.admm_shape_pairs_per_world > 0), "Each world should have Franka-payload ADMM contact pairs"
        # 各世界接触对数应一致（多世界复制正确）。
        assert np.all(self.admm_shape_pairs_per_world == self.admm_shape_pairs_per_world[0]), (
            "Franka-payload ADMM contact pair counts should be identical across replicated worlds"
        )
        # 若启用了图捕获，graph 必须非空。
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"

    # 每帧渲染：开始帧 → 记录耦合视图（接触等）→ 结束帧。
    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)  # 把接触/状态送入查看器
        self.viewer.end_frame()

    # 静态方法：构建命令行参数解析器（含所有 ADMM/负载/子求解器参数）。
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()       # 基础参数
        newton.examples.add_coupled_view_args(parser)  # 耦合视图参数
        newton.examples.add_world_count_arg(parser)    # 世界数参数
        parser.set_defaults(world_count=8)             # 默认 8 个世界
        parser.add_argument("--substeps", type=int, default=16, help="Coupled substeps per rendered frame.")          # 每帧子步
        parser.add_argument("--admm-iterations", type=int, default=5, help="ADMM iterations per coupled substep.")    # ADMM 轮数
        parser.add_argument("--rho", type=float, default=200.0, help="ADMM penalty parameter.")                       # 罚参数 ρ
        parser.add_argument("--gamma", type=float, default=0.001, help="ADMM proximal metric scale.")                 # 近端 γ
        parser.add_argument("--baumgarte", type=float, default=0.5, help="Position error correction fraction.")       # Baumgarte β
        parser.add_argument(
            "--rigid-contact-matching",
            choices=["disabled", "latest", "sticky"],
            default="latest",
            help="ADMM Franka-payload rigid contact matching mode.",                                                   # 接触匹配模式
        )
        parser.add_argument(
            "--contact-matching-pos-threshold",
            type=float,
            default=None,
            help="ADMM rigid contact matching midpoint distance threshold [m]; omitted uses CollisionPipeline default.",  # 匹配距离阈值
        )
        parser.add_argument(
            "--contact-matching-normal-dot-threshold",
            type=float,
            default=None,
            help="ADMM rigid contact matching normal dot-product threshold; omitted uses CollisionPipeline default.",   # 法向点积阈值
        )
        parser.add_argument(
            "--contact-matching-force-scale",
            type=float,
            default=0.9,
            help="Multiplier for matched previous-step ADMM rigid contact lambda warm-starts.",                          # warm-start λ 缩放
        )
        parser.add_argument(
            "--payload-kind",
            choices=["xpbd-chain", "vbd-cable"],
            default="xpbd-chain",
            help="Payload simulated by the non-MuJoCo solver.",                                                          # 负载类型
        )
        parser.add_argument("--payload-segments", type=int, default=11, help="Number of payload rigid/cable segments.")  # 负载段数
        parser.add_argument("--payload-radius", type=float, default=0.012, help="Payload capsule/cable radius [m].")     # 负载半径
        parser.add_argument("--xpbd-iterations", type=int, default=16, help="XPBD iterations per coupled substep.")      # XPBD 迭代
        parser.add_argument(
            "--xpbd-joint-linear-relaxation",
            type=float,
            default=0.9,
            help="XPBD joint linear relaxation for the rigid-chain payload.",                                             # XPBD 线松弛
        )
        parser.add_argument(
            "--xpbd-joint-angular-relaxation",
            type=float,
            default=0.5,
            help="XPBD joint angular relaxation for the rigid-chain payload.",                                            # XPBD 角松弛
        )
        parser.add_argument("--vbd-iterations", type=int, default=8, help="VBD iterations per coupled substep.")          # VBD 迭代
        parser.add_argument("--mujoco-iterations", type=int, default=12, help="MuJoCo solver iterations.")                # MuJoCo 迭代
        parser.add_argument("--mujoco-ls-iterations", type=int, default=25, help="MuJoCo line-search iterations.")        # MuJoCo 线搜索
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable graph capture.",                                                                                 # 关闭图捕获
        )
        return parser


# 脚本入口：解析参数 → 初始化查看器与示例 → 进入运行循环。
if __name__ == "__main__":
    parser = Example.create_parser()                   # 构建参数解析器
    viewer, args = newton.examples.init(parser)        # 初始化查看器、解析参数
    example = Example(viewer, args)                    # 构建示例（建模+求解器+IK+图）
    newton.examples.run(example, args)                 # 进入主循环（反复 step/render）
