"""W1 holds a shopping basket with the left gripper and drops a deformable chips bag into it with the right."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

import newton
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d

DIRECTORY = Path(__file__).resolve().parents[1] / "w1_snack_basket"
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(ROOT / "demos/utils"))
sys.path.insert(0, str(ROOT / "demos/piper_bag_transport"))
sys.path.insert(0, str(ROOT / "demos/w1_courier_transport"))
import scene
from bag_bending import rebuild_bag_bending
from snack_geometry import core_mesh, film_mesh, six_volumes

FILM_DENSITY = 0.1
# Metallised BOPP/PET laminate is stiffer in bending than mailer polyethylene; it also keeps the bag's
# dog-ear corners from flapping.
FILM_BENDING = wp.vec3(5e-7)
# Chips plus trapped air: a light core with low shear stiffness and a much higher bulk modulus, so a squeezed
# pillow bulges elsewhere instead of losing volume.
CORE_DENSITY, CORE_MU, CORE_LAMBDA = 45.0, 800.0, 20000.0
# Settle the resting bag and basket under heavy drag before recording; the robot holds its start pose meanwhile.
SETTLE_FRAMES, SETTLE_DRAG, DRAG = 60, 40.0, 3.0


@wp.kernel
def viscous_drag(velocity: wp.array[wp.vec3], mass: wp.array[float], rate: wp.array[float], force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = -rate[0] * mass[i] * velocity[i]


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
        self.plan = scene.create_plan(duration=args.seconds)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        joints, _ = scene.add_w1(builder, self.plan[0])
        self.basket = scene.add_basket(builder)
        # Chips bag film: two welded panels with the supplied print mapped onto the front and back.
        points, faces, panel_uv, panel_faces, texture_uv = film_mesh()
        rotation = scene.BAG_ROTATION.as_matrix()
        world = points @ rotation.T + scene.BAG_CENTER
        start, edge = builder.particle_count, len(builder.edge_indices)
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            vertices=world,
            indices=faces.ravel(),
            panel_verts=panel_uv,
            panel_indices=panel_faces.ravel(),
            density=FILM_DENSITY,
            particle_radius=0.0015,
            tri_aniso_ke=wp.vec3(1200, 1200, 500),
            tri_ka=0,
            tri_kd=0,
            edge_aniso_ke=FILM_BENDING,
            edge_kd=2e-7,
            validate_mesh=True,
            label="snack_film",
        )
        rebuild_bag_bending(
            builder,
            faces,
            panel_uv,
            panel_faces,
            particle_start=start,
            edge_start=edge,
            edge_aniso_ke=FILM_BENDING,
            edge_kd=2e-7,
        )
        self.film = dict(
            name="snack_film", start=start, end=builder.particle_count, faces=faces, rest=world, local=points
        )
        self.panel_uv, self.panel_faces, self.texture_uv = panel_uv, panel_faces, texture_uv
        core, self.tets = core_mesh()
        self.core_start = builder.particle_count
        first_tri = len(builder.tri_indices)
        builder.add_soft_mesh(
            pos=wp.vec3(*scene.BAG_CENTER),
            rot=wp.quat(*scene.BAG_ROTATION.as_quat()),
            scale=1.0,
            vel=wp.vec3(0),
            vertices=core,
            indices=self.tets.ravel(),
            density=CORE_DENSITY,
            k_mu=CORE_MU,
            k_lambda=CORE_LAMBDA,
            k_damp=2.0,
            particle_radius=0.0015,
            add_surface_mesh_edges=True,
            validate_mesh=True,
            label="snack_core",
        )
        core_faces = np.asarray(builder.tri_indices[first_tri:]) - self.core_start
        self.core = dict(
            name="snack_core",
            start=self.core_start,
            end=builder.particle_count,
            faces=core_faces,
            rest=core @ rotation.T + scene.BAG_CENTER,
            local=core,
        )
        self.groups = [self.film, self.core]
        # Weak finite-length springs keep the loose chips roughly centred in the envelope; both ends are dynamic
        # and nothing connects the bag to the robot or the world.
        surface = np.unique(core_faces)
        _, neighbors = cKDTree(points).query(core[surface], k=3)
        for node, near in zip(surface, neighbors):
            for cloth in near:
                builder.add_spring(self.core_start + int(node), start + int(cloth), ke=30.0, kd=0.02, control=0.0)
        self.rest_volumes = six_volumes(core, self.tets)
        floor = builder.default_shape_cfg.copy()
        floor.ke, floor.kd, floor.mu = 1.0e4, 10, 0.5
        builder.add_ground_plane(cfg=floor)
        self.model = builder.finalize()
        self.gripper_bodies = [i for i, label in enumerate(self.model.body_label) if "finger" in label.lower()]
        # The courier exporter names a single rigid payload "carton"; there is none here.
        self.carton = -2
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
        cfg.constraint_alpha = 5e2
        self.state, self.next_state, self.control = self.model.state(), self.model.state(), self.model.control()
        self.clock = wp.zeros(1, dtype=int)
        # Device-side drag rate so the captured graph can run both the settling and the recorded phases.
        self.drag = wp.array([DRAG], dtype=float)
        self.targets = wp.array(self.plan, dtype=float)
        self.target_indices = wp.array(self.model.joint_target_q_start.numpy()[joints], dtype=int)
        self.target_dof_indices = wp.array(self.model.joint_qd_start.numpy()[joints], dtype=int)
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=16384, soft_contact_gap=0.006, enable_rigid_soft_full_surface_contact=True
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
            # Mass-proportional drag damps unresolved film flutter while leaving the bag free to fall and swing.
            wp.launch(
                viscous_drag,
                dim=self.groups[-1]["end"],
                inputs=[self.state.particle_qd, self.model.particle_mass, self.drag],
                outputs=[self.state.particle_f],
            )
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
            self.state, self.next_state = self.next_state, self.state

    def step(self):
        wp.capture_launch(self.graph)

    def settle(self):
        # The plan holds its first pose for over a second, so these steps only let the payloads come to rest;
        # the drive clock is rewound so the recorded motion still starts at plan frame 0.
        self.drag.assign([SETTLE_DRAG])
        for _ in range(SETTLE_FRAMES):
            self.step()
        self.drag.assign([DRAG])
        self.clock.zero_()
        for state in (self.state, self.next_state):
            state.particle_qd.zero_()


def boolean_argument(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=scene.DURATION)
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
        default=ROOT / "build/w1-snack-basket",
        help="Recording directory; only used with --viewer=false.",
    )
    args = parser.parse_args(argv)
    if args.substeps < 2 or args.substeps % 2 or not 0 < args.seconds <= 30:
        parser.error("Use an even positive substep count and a duration in (0, 30].")
    return args


def report(sim, frame, particles, bodies, start):
    film = particles[sim.film["start"] : sim.film["end"]]
    core = particles[sim.core_start : sim.core["end"]]
    ratio = six_volumes(core, sim.tets) / sim.rest_volumes
    fingers = bodies[sim.gripper_bodies, :3].reshape(2, 2, 3)
    gaps = np.linalg.norm(fingers[:, 0] - fingers[:, 1], axis=1) * 1000
    basket = bodies[sim.basket]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (basket[3] ** 2 + basket[4] ** 2), -1, 1)))
    print(
        f"{frame / 60:.1f}s: bag={film.mean(0).round(3)} bag_z=[{film[:, 2].min():.3f},{film[:, 2].max():.3f}] "
        f"basket={basket[:3].round(3)} tilt={tilt:.1f}deg min_tet={ratio.min():.3f} "
        f"core_volume={ratio.mean():.3f} jaw_gap_mm(right,left)={gaps.round(1)} "
        f"elapsed={time.perf_counter() - start:.1f}s",
        flush=True,
    )


def main(argv=None):
    args = parse_args(argv)
    if not args.viewer:
        args.output.mkdir(parents=True, exist_ok=True)
        print(f"Headless render recording: {args.output.resolve()}", flush=True)
    sim = Simulation(args)
    sim.settle()
    viewer = None
    viewer_meshes = []
    if args.viewer:
        viewer = newton.viewer.ViewerGL(vsync=True)
        viewer.set_model(sim.model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.set_camera(pos=wp.vec3(2.3, -2.4, 2.0), pitch=-20.0, yaw=135.0)
        for group, color in zip(sim.groups, ((0.85, 0.12, 0.18), (0.95, 0.75, 0.3))):
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
            if (not args.viewer and frame % 2 == 0) or frame % 30 == 0:
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
                        roughness=0.4,
                        backface_culling=False,
                    )
                viewer.end_frame()
            if frame % 30 == 0:
                report(sim, frame, particles, bodies, start)
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
        basket=sim.basket,
        core_start=sim.core_start,
        tets=sim.tets,
        rest_volumes=sim.rest_volumes,
        film_faces=sim.film["faces"],
        film_panel_vertices=sim.panel_uv,
        film_panel_faces=sim.panel_faces,
        film_texture_uv=sim.texture_uv,
        gripper_bodies=sim.gripper_bodies,
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
                robot="W1 V030 (simready, ~/下载/assets/w1-030)",
                solver="s3d.SolverCoupledHighAccuracy",
                motion="left_hand_holds_basket_right_hand_drops_chips_bag",
                basket_body=sim.basket,
                basket_scale=scene.BASKET_SCALE,
                basket_center_m=scene.BASKET_CENTER.tolist(),
                basket_rotation_xyzw=scene.BASKET_ROTATION.as_quat().tolist(),
                basket_mass_kg=scene.BASKET_MASS,
                film_density_kg_m2=FILM_DENSITY,
                core_density=CORE_DENSITY,
                core_mu=CORE_MU,
                core_lambda=CORE_LAMBDA,
                bag_release_seconds=(scene.BAG_RELEASE_START, scene.BAG_RELEASE_END),
                basket_lift_m=scene.BASKET_LIFT,
                wall_seconds=time.perf_counter() - start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
