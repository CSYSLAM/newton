"""One Style3D solver couples the W1 dual arms, two courier mailers and rigid/soft payloads."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

import newton
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d

DIRECTORY = Path(__file__).resolve().parents[1] / "w1_courier_transport"
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(ROOT / "demos/utils"))
sys.path.insert(0, str(ROOT / "demos/piper_bag_transport"))
sys.path.insert(0, str(ROOT / "demos/piper_package_transport"))
from bag_bending import rebuild_bag_bending
from geometry import BAG_SIZE, GRASP_OFFSET, PAYLOAD_SIZE, sealed_bag, six_volumes, tetrahedral_box
from particle_forces import viscous_drag
from w1_motion import (
    APPROACH_END,
    CARRY_OFFSET,
    CLOSE_END,
    FINGER_PAD_CENTER,
    FINGER_PAD_HALF_EXTENTS,
    GRASPS,
    LIFT_START,
    TCP,
)
from w1_motion import CENTERS as W1_CENTERS
from w1_motion import ROTATIONS as W1_ROTATIONS
from w1_motion import add_w1 as add_piper
from w1_motion import create_plan as w1_plan

GRIP_POINTS = (tuple(TCP), tuple(TCP))
SHAKE_START, SHAKE_END, SHAKE_CYCLES = 7.0, 10.0, 3
DROP_RELEASE_START, DROP_RELEASE_END, RELEASE_OPENING = 10.0, 10.2, 0.045
LIFT_HEIGHT, SHAKE_DIRECTION, SHAKE_AMPLITUDE = float(CARRY_OFFSET[2]), (1.0, 0.0, 0.0), 0.04
SHAKE_PITCH_CENTER, SHAKE_PITCH_AMPLITUDE = 120.0, 12.0


create_plan = w1_plan
CENTERS = W1_CENTERS
ROTATIONS = W1_ROTATIONS


@wp.kernel
def drive(
    clock: wp.array[int],
    targets: wp.array2d[float],
    indices: wp.array[int],
    substeps: int,
    dof_indices: wp.array[int],
    output: wp.array[float],
    velocity: wp.array[float],
):
    row = wp.min(clock[0] // substeps, targets.shape[0] - 2)
    blend = float(clock[0] % substeps) / float(substeps)
    for j in range(indices.shape[0]):
        output[indices[j]] = wp.lerp(targets[row, j], targets[row + 1, j], blend)
        velocity[dof_indices[j]] = (targets[row + 1, j] - targets[row, j]) * 60.0
    clock[0] += 1


class Simulation:
    def __init__(self, args):
        self.substeps, self.dt = args.substeps, 1 / (60 * args.substeps)
        self.plan = create_plan(duration=args.seconds)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        joints, self.wrist = add_piper(builder, self.plan[0], finger_pads=True)
        self.groups = []
        points, faces, uv, uv_faces = sealed_bag()
        self.film_panels, self.film_panel_faces = uv, uv_faces
        for i in range(2):
            start, edge = builder.particle_count, len(builder.edge_indices)
            world = points @ ROTATIONS[i].T + CENTERS[i]
            style3d.add_cloth_mesh(
                builder,
                pos=wp.vec3(0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0),
                vertices=world,
                indices=faces.ravel(),
                panel_verts=uv,
                panel_indices=uv_faces.ravel(),
                density=0.1,
                particle_radius=0.0015,
                tri_aniso_ke=wp.vec3(1200, 1200, 500),
                tri_ka=0,
                tri_kd=0,
                edge_aniso_ke=wp.vec3(1e-7),
                edge_kd=2e-7,
                validate_mesh=True,
                label=f"film_{i}",
            )
            rebuild_bag_bending(
                builder,
                faces,
                uv,
                uv_faces,
                particle_start=start,
                edge_start=edge,
                edge_aniso_ke=wp.vec3(1e-7),
                edge_kd=2e-7,
            )
            self.groups.append(
                dict(name=f"film_{i}", start=start, end=builder.particle_count, faces=faces, rest=world, local=points)
            )
        carton, _ = tetrahedral_box(cells=(2, 2, 2))
        hull = ConvexHull(carton)
        tri = hull.simplices.copy()
        for row, normal in zip(tri, hull.equations[:, :3]):
            if np.dot(np.cross(carton[row[1]] - carton[row[0]], carton[row[2]] - carton[row[0]]), normal) < 0:
                row[1], row[2] = row[2], row[1]
        mesh = newton.Mesh(carton, tri.ravel(), maxhullvert=0)
        mesh.build_sdf(max_resolution=64, narrow_band_range=(-0.02, 0.02), cache_dir=ROOT / "build/package-sdf")
        self.carton_points = carton
        cfg = builder.default_shape_cfg.copy()
        cfg.density, cfg.mu, cfg.margin = 120, 3.0, 0.006
        self.carton = builder.add_body(
            xform=wp.transform(wp.vec3(*CENTERS[0]), wp.quat(*Rotation.from_matrix(ROTATIONS[0]).as_quat())),
            label="carton",
        )
        builder.add_shape_convex_hull(self.carton, mesh=mesh, cfg=cfg, color=(0.48, 0.26, 0.10), label="carton")
        jelly, self.tets = tetrahedral_box()
        self.jelly_start = builder.particle_count
        first_tri = len(builder.tri_indices)
        builder.add_soft_mesh(
            pos=wp.vec3(*CENTERS[1]),
            rot=wp.quat(*Rotation.from_matrix(ROTATIONS[1]).as_quat()),
            scale=1.0,
            vel=wp.vec3(0),
            vertices=jelly,
            indices=self.tets.ravel(),
            density=450,
            k_mu=1500,
            k_lambda=15000,
            k_damp=1.0,
            particle_radius=0.0015,
            add_surface_mesh_edges=True,
            validate_mesh=True,
            label="jelly",
        )
        jelly_faces = np.asarray(builder.tri_indices[first_tri:]) - self.jelly_start
        self.groups.append(
            dict(
                name="jelly",
                start=self.jelly_start,
                end=builder.particle_count,
                faces=jelly_faces,
                rest=jelly @ ROTATIONS[1].T + CENTERS[1],
                local=jelly,
            )
        )
        # A wet gel adheres to its liner. Finite rest-length springs retain the initial interface clearance;
        # both endpoints remain dynamic, and no interface spring connects to the robot or the world.
        surface = np.unique(jelly_faces)
        tree = cKDTree(self.groups[1]["local"])
        _, neighbors = tree.query(jelly[surface], k=3)
        for node, near in zip(surface, neighbors):
            for cloth in near:
                builder.add_spring(
                    self.jelly_start + int(node), self.groups[1]["start"] + int(cloth), ke=80.0, kd=0.03, control=0.0
                )
        self.rest_volumes = six_volumes(jelly, self.tets)
        floor = builder.default_shape_cfg.copy()
        floor.ke, floor.kd, floor.mu = 1.0e4, 10, 0.5
        builder.add_ground_plane(cfg=floor)
        self.model = builder.finalize()
        self.gripper_bodies = [i for i, label in enumerate(self.model.body_label) if "finger" in label.lower()]
        # URDF traversal visits the right hand first, whereas the bag groups start with the left hand.
        self.gripper_bag_indices = [0 if "/left_" in self.model.body_label[i] else 1 for i in self.gripper_bodies[::2]]
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.soft_contact_ke = 2e3
        self.model.soft_contact_kd = 1
        self.model.soft_contact_kf = 1e3
        self.model.soft_contact_mu = 4.0
        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        cfg = self.solver.config
        cfg.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        cfg.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        cfg.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        cfg.inertia_mode = s3d.InertiaMode.NONE
        cfg.joint_constraint_primal_iterations = 1
        cfg.nonlinear_iterations, cfg.linear_iterations = args.iterations, args.linear_iterations
        cfg.enable_collision_hessian_non_diagonal = True
        cfg.self_collision_radius, cfg.enable_self_collision = 0.0015, True
        cfg.self_collision_edge_face_stiffness = 0.1
        # cfg.self_collision_vertex_face_stiffness = 0.2
        # cfg.self_collision_edge_edge_stiffness = 0.01
        cfg.constraint_alpha = 5e2
        self.state, self.next_state, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=int)
        self.targets = wp.array(self.plan, dtype=float)
        self.target_indices = wp.array(self.model.joint_target_q_start.numpy()[joints], dtype=int)
        self.target_dof_indices = wp.array(self.model.joint_qd_start.numpy()[joints], dtype=int)
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=8192, soft_contact_gap=0.006, enable_rigid_soft_full_surface_contact=True
        )
        self.contacts = self.pipeline.contacts()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.substeps):
            wp.launch(
                drive,
                dim=1,
                inputs=[self.clock, self.targets, self.target_indices, self.substeps, self.target_dof_indices],
                outputs=[self.control.joint_target_q, self.control.joint_target_qd],
            )
            self.state.clear_forces()
            # Modest mass-proportional drag dissipates unresolved film vibration while retaining free payload motion.
            wp.launch(
                viscous_drag,
                dim=self.groups[-1]["end"],
                inputs=[self.state.particle_qd, self.model.particle_mass, 3.0],
                outputs=[self.state.particle_f],
            )
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
            self.state, self.next_state = self.next_state, self.state

    def step(self):
        wp.capture_launch(self.graph)


def boolean_argument(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15)
    # Resolve film squeezed between the side-closing pads and floor, as well as the airborne jelly's landing impact.
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--linear-iterations", type=int, default=50)
    parser.add_argument(
        "--viewer",
        type=boolean_argument,
        nargs="?",
        const=True,
        default=True,
        metavar="true|false",
        help="Show Newton ViewerGL (default: true); false records headlessly for rendering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build/w1-courier",
        help="Recording directory; only used with --viewer=false.",
    )
    args = parser.parse_args(argv)
    if args.substeps < 2 or args.substeps % 2 or not 0 < args.seconds <= 30:
        parser.error("Use an even positive substep count and a duration in (0, 30].")
    if args.iterations < 1 or args.linear_iterations < 1:
        parser.error("Solver iteration counts must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    print(
        f"Payload dimensions (mm): {np.asarray(PAYLOAD_SIZE) * 1000}; "
        f"bag dimensions (mm): {np.asarray(BAG_SIZE) * 1000}",
        flush=True,
    )
    if not args.viewer:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"Headless render recording: {args.output.resolve()}", flush=True)
    sim = Simulation(args)
    viewer = None
    viewer_meshes = []
    if args.viewer:
        viewer = newton.viewer.ViewerGL(vsync=True)
        viewer.set_model(sim.model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.set_camera(pos=wp.vec3(2.3, -2.4, 2.0), pitch=-20.0, yaw=135.0)
        colors = ((0.32, 0.35, 0.39), (0.32, 0.35, 0.39), (0.05, 0.55, 0.30))
        for group, color in zip(sim.groups, colors):
            indices = wp.array(group["faces"].astype(np.int32).ravel(), dtype=wp.int32, device=sim.model.device)
            viewer_meshes.append((group, indices, color))
    usd = None
    if not args.viewer:
        # Interactive playback must not construct a USD stage or retain a full trajectory in host memory.
        from w1_export import SceneUSD

        usd = SceneUSD(args.output / "simulation.usdc", sim, round(args.seconds * 30))
    particle_frames, body_frames = [], []
    start = time.perf_counter()
    completed_frame = 0
    try:
        for frame in range(round(args.seconds * 60) + 1):
            if viewer is not None and not viewer.is_running():
                break
            if frame:
                sim.step()
            completed_frame = frame
            if (not args.viewer and frame % 2 == 0) or frame % 60 == 0:
                particles, bodies = sim.state.particle_q.numpy(), sim.state.body_q.numpy()
                if not np.isfinite(particles).all() or not np.isfinite(bodies).all():
                    raise RuntimeError(f"Non-finite state at frame {frame}")
            if usd is not None and frame % 2 == 0:
                usd.sample(frame // 2, particles, bodies)
                particle_frames.append(particles)
                body_frames.append(bodies)
            if viewer is not None:
                viewer.begin_frame(frame / 60.0)
                viewer.log_state(sim.state)
                for group, indices, color in viewer_meshes:
                    viewer.log_mesh(
                        f"/simulation/{group['name']}",
                        sim.state.particle_q[group["start"] : group["end"]],
                        indices,
                        color=color,
                        roughness=0.65,
                        backface_culling=False,
                    )
                viewer.end_frame()
            if frame % 60 == 0:
                jelly = particles[sim.jelly_start : sim.groups[-1]["end"]]
                ratio = six_volumes(jelly, sim.tets) / sim.rest_volumes
                carton_normal = Rotation.from_quat(bodies[sim.carton, 3:]).as_matrix()[2, 2]
                fingers = bodies[sim.gripper_bodies, :3].reshape(2, 2, 3)
                gaps = np.linalg.norm(fingers[:, 0] - fingers[:, 1], axis=1) * 1000
                print(
                    f"{frame / 60:.1f}s: carton={bodies[sim.carton, :3].round(3)} "
                    f"carton_normal_z={carton_normal:.3f} "
                    f"jelly={jelly.mean(0).round(3)} floor={particles[: sim.groups[-1]['end'], 2].min():.4f} "
                    f"min_tet={ratio.min():.3f} jaw_gap_mm(right,left)={gaps.round(1)} "
                    f"elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    finally:
        if viewer is not None:
            viewer.close()
    if usd is None:
        return
    usd.stage.SetEndTimeCode(len(particle_frames) - 1)
    usd.stage.GetRootLayer().Save()
    np.savez_compressed(
        args.output / "trajectory.npz",
        particles=particle_frames,
        bodies=body_frames,
        carton=sim.carton,
        jelly_start=sim.jelly_start,
        tets=sim.tets,
        rest_volumes=sim.rest_volumes,
        carton_points=sim.carton_points,
        film_faces=sim.groups[0]["faces"],
        film_panel_vertices=sim.film_panels,
        film_panel_faces=sim.film_panel_faces,
        wrist=sim.wrist,
        gripper_bodies=sim.gripper_bodies,
        grip_local=np.asarray(GRIP_POINTS),
        gripper_bag_indices=sim.gripper_bag_indices,
        seal_patch_indices=np.flatnonzero(
            (np.abs(sim.groups[0]["local"][:, 0]) < 0.015)
            & (np.abs(sim.groups[0]["local"][:, 1] + GRASP_OFFSET) < 0.007)
        ),
        joint_targets=sim.plan,
        group_ranges=np.array([(g["start"], g["end"]) for g in sim.groups]),
    )
    (args.output / "run.json").write_text(
        json.dumps(
            dict(
                seconds=completed_frame / 60,
                fps=30,
                substeps=args.substeps,
                nonlinear=args.iterations,
                pcg=args.linear_iterations,
                particles=sim.model.particle_count,
                tetrahedra=len(sim.tets),
                payload_size_m=PAYLOAD_SIZE,
                bag_size_m=BAG_SIZE,
                bag_construction="two_rectangular_panels_perimeter_welded",
                robot="W1 V030",
                solver="s3d.SolverCoupledHighAccuracy",
                motion="simultaneous_dual_arm_grip_shake_drop",
                shake_local_seconds=(SHAKE_START, SHAKE_END),
                shake_cycles=SHAKE_CYCLES,
                shake_direction_world=SHAKE_DIRECTION,
                shake_amplitude_m=SHAKE_AMPLITUDE,
                shake_pitch_axis_world=(0, 1, 0),
                shake_pitch_center_degrees=SHAKE_PITCH_CENTER,
                shake_pitch_amplitude_degrees=SHAKE_PITCH_AMPLITUDE,
                seal_grasp_offset_m=GRASP_OFFSET,
                grasp_tcp_world_m=GRASPS.tolist(),
                approach_end_seconds=APPROACH_END,
                close_end_seconds=CLOSE_END,
                lift_start_seconds=LIFT_START,
                carry_offset_world_m=CARRY_OFFSET.tolist(),
                finger_pad_center_m=FINGER_PAD_CENTER,
                finger_pad_size_m=(np.asarray(FINGER_PAD_HALF_EXTENTS) * 2).tolist(),
                jaw_closing_axis_world=(0, 1, 0),
                lift_height_m=LIFT_HEIGHT,
                release_local_seconds=(DROP_RELEASE_START, DROP_RELEASE_END),
                release_opening_m=RELEASE_OPENING,
                wall_seconds=time.perf_counter() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
