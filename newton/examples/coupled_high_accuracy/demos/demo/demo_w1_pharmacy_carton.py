"""W1 pharmacy: two identical hollow cartons, gentle grasp versus force-limited plastic crushing."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import coupled_high_accuracy as cha

HERE = Path(__file__).resolve().parents[1] / "w1_pharmacy_carton"
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(HERE.parent / "utils"),
    str(HERE.parent / "w1_shopping_bag"),
    str(HERE.parent / "w1_courier_transport"),
    str(HERE.parent / "piper_bag_transport"),
]
from carton_asset import CENTERS, make_carton
from carton_motion import create_plan
from gripper_collision import enable_cross_arm_collisions, install
from parallel_gripper import ParallelJawCoupling
from particle_forces import viscous_drag
from plasticity import PaperPlasticity
from sdf_candidate_filter import ContactFilter, filtered_kernels
from w1_export import SceneUSD
from w1_motion import add_w1


def install_rubber_pads(builder):
    from scipy.spatial import ConvexHull

    size = np.array([0.026, 0.044, 0.003])
    vertices = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32) * size
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for face, normal in zip(faces, hull.equations[:, :3]):
        p = vertices[face]
        if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), normal) < 0:
            face[:] = face[::-1]
    mesh = newton.Mesh(vertices, faces.ravel(), maxhullvert=0)
    mesh.build_sdf(max_resolution=32, narrow_band_range=(-0.008, 0.008), cache_dir=ROOT / "build/pharmacy-pad-sdf")
    pads = []
    for body, name in enumerate(builder.body_label):
        if name.endswith(("left_finger1", "left_finger2", "right_finger1", "right_finger2")):
            cfg = builder.default_shape_cfg.copy()
            cfg.density = 0
            cfg.mu = 1.0
            cfg.margin = 0.0002
            pads.append(
                builder.add_shape_convex_hull(
                    body,
                    mesh=mesh,
                    xform=wp.transform(wp.vec3(0, -0.032, -0.0015), wp.quat_identity()),
                    cfg=cfg,
                    color=(0.015, 0.018, 0.02),
                    label="Rubber finger pad",
                )
            )
    for pad in pads:
        for other, body in enumerate(builder.shape_body):
            if body >= 0 and other != pad:
                builder.add_shape_collision_filter_pair(pad, other)


@wp.kernel
def drive(
    clock: wp.array[int],
    targets: wp.array2d[float],
    rates: wp.array2d[float],
    indices: wp.array[int],
    dofs: wp.array[int],
    fingers: wp.array[int],
    substeps: int,
    normal_force: float,
    crush_force: float,
    output: wp.array[float],
    velocity: wp.array[float],
    limits: wp.array[float],
):
    row = wp.min(clock[0] // substeps, targets.shape[0] - 2)
    u = float(clock[0] % substeps) / float(substeps)
    for j in range(indices.shape[0]):
        a = targets[row, j]
        b = targets[row + 1, j]
        va = rates[row, j] / 60.0
        vb = rates[row + 1, j] / 60.0
        output[indices[j]] = (
            (2.0 * u * u * u - 3.0 * u * u + 1.0) * a
            + (u * u * u - 2.0 * u * u + u) * va
            + (-2.0 * u * u * u + 3.0 * u * u) * b
            + (u * u * u - u * u) * vb
        )
        velocity[dofs[j]] = 60.0 * (
            (6.0 * u * u - 6.0 * u) * a
            + (3.0 * u * u - 4.0 * u + 1.0) * va
            + (-6.0 * u * u + 6.0 * u) * b
            + (3.0 * u * u - 2.0 * u) * vb
        )
    t = float(clock[0]) / float(60 * substeps)
    blend = wp.clamp((t - 5.5) / 1.2, 0.0, 1.0)
    blend = blend * blend * blend * (10.0 + blend * (-15.0 + 6.0 * blend))
    for i in range(4):
        force = normal_force
        if i >= 2:
            force = normal_force + (crush_force - normal_force) * blend
        limits[fingers[i]] = force
    clock[0] += 1


class Simulation:
    def __init__(self, args):
        self.substeps = args.substeps
        self.plan = create_plan(max(args.seconds, 15))
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
        joints, self.wrist = add_w1(b, self.plan[0])
        install(b)
        install_rubber_pads(b)
        enable_cross_arm_collisions(b)
        finger_joints = []
        for side in ("LEFT", "RIGHT"):
            for finger in (1, 2):
                j = next(j for j in joints if b.joint_label[j].endswith(f"{side}_FINGER{finger}_JOINT"))
                dof = b.joint_qd_start[j]
                finger_joints.append(j)
                b.joint_target_ke[dof] = 5000
                b.joint_target_kd[dof] = 50
                b.joint_effort_limit[dof] = args.normal_force
        self.groups = []
        self.carton = -999
        self.asset = make_carton()
        self.tet_ranges = []
        for i, center in enumerate(CENTERS):
            start = b.particle_count
            tet_start = len(b.tet_indices)
            poisson = 0.25
            young = args.young
            b.add_soft_mesh(
                pos=wp.vec3(*center),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0),
                vertices=self.asset["vertices"],
                indices=self.asset["tets"].ravel(),
                density=650,
                k_mu=young / (2 * (1 + poisson)),
                k_lambda=young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
                k_damp=0.03,
                particle_radius=0.0003,
                add_surface_mesh_edges=True,
                label=("carton_left_gentle", "carton_right_crush")[i],
            )
            self.tet_ranges.append((tet_start, len(b.tet_indices)))
            self.groups.append(
                dict(
                    name=("carton_left_gentle", "carton_right_crush")[i],
                    start=start,
                    end=b.particle_count,
                    rest=self.asset["vertices"] + center,
                    faces=self.asset["faces"],
                )
            )
        self.n = b.particle_count
        cfg = b.default_shape_cfg.copy()
        cfg.mu = 0.5
        b.add_ground_plane(cfg=cfg)
        self.model = b.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.soft_contact_ke = 6e4
        self.model.soft_contact_kd = 10
        self.model.soft_contact_kf = 1000
        self.model.soft_contact_mu = 0.65
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        cfg = self.solver.config
        cfg.nonlinear_solver = cha.NonlinearSolver.NEWTON
        cfg.nonlinear_iterations = args.iterations
        cfg.linear_iterations = args.linear_iterations
        cfg.tangent_streams = args.tangent_streams
        cfg.inertia_mode = cha.InertiaMode.NONE
        cfg.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        cfg.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        cfg.joint_constraint_primal_iterations = 1
        cfg.constraint_alpha = 500
        cfg.enable_collision_hessian_non_diagonal = True
        cfg.enable_self_collision = True
        cfg.self_collision_radius = 0.00018
        cfg.self_collision_edge_face_stiffness = 0.5
        self.material = PaperPlasticity(self.model, args.yield_strain)
        self.solver.enable_joint_drive_feedback()
        self.state, self.next = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.coupling = ParallelJawCoupling(self.model)
        self.clock = wp.zeros(1, dtype=int)
        self.targets = wp.array(self.plan, dtype=float)
        self.rates = wp.array(np.gradient(self.plan, 1 / 60, axis=0).astype(np.float32), dtype=float)
        self.indices = wp.array(self.model.joint_target_q_start.numpy()[joints], dtype=int)
        self.dofs = wp.array(self.model.joint_qd_start.numpy()[joints], dtype=int)
        self.fingers = wp.array(self.model.joint_qd_start.numpy()[finger_joints], dtype=int)
        self.finger_dofs = self.fingers.numpy()
        self.forces = (args.normal_force, args.crush_force)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=8192,
            soft_contact_max=65536,
            soft_contact_gap=0.0015,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        cache = ROOT / "build/pharmacy-contact-filter"
        cache.mkdir(exist_ok=True)
        self.filter = ContactFilter(self, filtered_kernels(cache))
        self.filter.enabled = True
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        dt = 1 / (60 * self.substeps)
        for _ in range(self.substeps):
            wp.launch(
                drive,
                dim=1,
                inputs=[
                    self.clock,
                    self.targets,
                    self.rates,
                    self.indices,
                    self.dofs,
                    self.fingers,
                    self.substeps,
                    *self.forces,
                ],
                outputs=[self.control.joint_target_q, self.control.joint_target_qd, self.model.joint_effort_limit],
            )
            self.state.clear_forces()
            wp.launch(
                viscous_drag,
                dim=self.n,
                inputs=[self.state.particle_qd, self.model.particle_mass, 8.0],
                outputs=[self.state.particle_f],
            )
            self.pipeline.collide(self.state, self.contacts)
            self.coupling.apply(self.state, self.control, dt)
            self.solver.step(self.state, self.next, self.control, self.contacts, dt)
            self.state, self.next = self.next, self.state
            self.material.step(self.state, dt)


def live(sim, seconds, close_on_end=False):
    from newton.viewer import ViewerGL

    viewer = ViewerGL(vsync=True)
    try:
        viewer.set_model(sim.model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.set_camera(pos=wp.vec3(1.8, -2.2, 1.6), pitch=-14.0, yaw=135.0)
        faces = wp.array(sim.asset["faces"].ravel(), dtype=int)
        frame = 0
        end = round(seconds * 60)
        while viewer.is_running():
            viewer.begin_frame(frame / 60)
            viewer.log_state(sim.state)
            for i, g in enumerate(sim.groups):
                viewer.log_mesh(
                    g["name"],
                    sim.state.particle_q[g["start"] : g["end"]],
                    faces,
                    color=(0.8, 0.9, 0.85) if i == 0 else (0.9, 0.8, 0.65),
                    backface_culling=False,
                )
            viewer.end_frame()
            if frame >= end and close_on_end:
                break
            if frame < end and not viewer.is_paused():
                wp.capture_launch(sim.graph)
                frame += 1
            else:
                time.sleep(0.01)
    finally:
        viewer.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=15)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--linear-iterations", type=int, default=120)
    # Concurrent PCG tangent terms; 1 restores the serial kernel order.
    p.add_argument("--tangent-streams", type=int, default=8)
    p.add_argument("--young", type=float, default=8e6)
    p.add_argument("--yield-strain", type=float, default=0.04)
    p.add_argument("--normal-force", type=float, default=2.5)
    p.add_argument("--crush-force", type=float, default=30.0)
    p.add_argument("--output", type=Path, default=ROOT / "build/w1-pharmacy-cartons")
    p.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the interactive viewer by default; --no-viewer records headlessly.",
    )
    p.add_argument("--close-on-end", action="store_true")
    args = p.parse_args()
    sim = Simulation(args)
    if args.viewer:
        live(sim, args.seconds, args.close_on_end)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    usd = SceneUSD(args.output / "simulation.usdc", sim, round(args.seconds * 30))
    particles = []
    bodies = []
    plastic = []
    forces = []
    coupling_efforts = []
    start = time.perf_counter()
    for frame in range(round(args.seconds * 60) + 1):
        if frame:
            wp.capture_launch(sim.graph)
        if frame % 2 == 0:
            q = sim.state.particle_q.numpy()
            body = sim.state.body_q.numpy()
            history = sim.material.accumulated.numpy()
            if not np.isfinite(q).all() or not np.isfinite(body).all():
                raise RuntimeError("Nonfinite paperboard state")
            sim.filter.counts_checked()
            if int(sim.contacts.soft_contact_count.numpy()[0]) > sim.contacts.soft_contact_max:
                raise RuntimeError("Soft contact capacity exceeded")
            particles.append(q[: sim.n])
            bodies.append(body)
            plastic.append(history)
            passive = sim.control.joint_f.numpy()[sim.finger_dofs]
            coupling_efforts.append(passive)
            forces.append(sim.solver.joint_drive_effort.numpy()[sim.finger_dofs] - passive)
            usd.sample(frame // 2, q, body)
        if frame % 30 == 0:
            centers = [q[g["start"] : g["end"]].mean(0).round(4).tolist() for g in sim.groups]
            strain = [float(history[a:b].max()) for a, b in sim.tet_ranges]
            print(
                f"{frame / 60:.1f}s centers={centers} plastic={np.round(strain, 4).tolist()} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    usd.stage.GetRootLayer().Save()
    np.savez_compressed(
        args.output / "trajectory.npz",
        particles=particles,
        bodies=bodies,
        plastic_strain=plastic,
        coupling_efforts=coupling_efforts,
        **sim.coupling.archive(),
        forces=forces,
        targets=sim.plan,
        faces=sim.asset["faces"],
        tets=sim.model.tet_indices.numpy(),
        ranges=[(g["start"], g["end"]) for g in sim.groups],
        tet_ranges=sim.tet_ranges,
        outer_count=sim.asset["outer_count"],
        uv=sim.asset["uv"],
        panel=sim.asset["panel"],
        rest=np.concatenate([g["rest"] for g in sim.groups]),
    )
    report = dict(
        solver="cha.SolverCoupledHighAccuracy",
        seconds=args.seconds,
        fps=30,
        particles=sim.n,
        tetrahedra=sim.model.tet_count,
        substeps=args.substeps,
        nonlinear_iterations=args.iterations,
        linear_iterations=args.linear_iterations,
        tangent_streams=args.tangent_streams,
        mass_damping_rate_per_s=8.0,
        plastic_hardening=0.035,
        plastic_viscosity_s=0.015,
        thickness_m=sim.asset["thickness"],
        young_pa=args.young,
        yield_strain=args.yield_strain,
        normal_finger_force_limit_n=args.normal_force,
        crush_finger_force_limit_n=args.crush_force,
        gripper_coupling=sim.coupling.report(),
        wall_seconds=time.perf_counter() - start,
        constitutive_model="Isotropic thin tetrahedral paperboard; split viscoplastic J2 log-strain return map.",
        scope="Estimated paperboard parameters; closed end-flap approximation, no tearing or adhesive failure.",
    )
    (args.output / "run.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
