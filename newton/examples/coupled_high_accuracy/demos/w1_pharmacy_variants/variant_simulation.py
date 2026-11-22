"""Dynamic W1, force-limited jaws and heterogeneous hollow elastoplastic pharmacy cartons."""

import json
import time

import numpy as np
import warp as wp
from catalog import centers
from common import ROOT, base
from variant_geometry import make_carton
from variant_motion import create_plan

import newton
from newton.solvers import coupled_high_accuracy as cha


@wp.kernel
def drive(
    clock: wp.array[int],
    targets: wp.array2d[float],
    rates: wp.array2d[float],
    indices: wp.array[int],
    dofs: wp.array[int],
    fingers: wp.array[int],
    peaks: wp.array[float],
    substeps: int,
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
    blend = wp.clamp((t - 4.2) / 1.1, 0.0, 1.0)
    blend = blend * blend * blend * (10.0 + blend * (-15.0 + 6.0 * blend))
    for i in range(4):
        limits[fingers[i]] = 2.5 + (peaks[i] - 2.5) * blend
    clock[0] += 1


class Simulation:
    def __init__(self, cases, args):
        self.cases = cases
        self.substeps = args.substeps
        self.plan = create_plan(cases)
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(b)
        joints, self.wrist = base.add_w1(b, self.plan[0])
        base.install(b)
        base.install_rubber_pads(b)
        base.enable_cross_arm_collisions(b)
        finger_joints = []
        for side in ("LEFT", "RIGHT"):
            for finger in (1, 2):
                j = next(j for j in joints if b.joint_label[j].endswith(f"{side}_FINGER{finger}_JOINT"))
                dof = b.joint_qd_start[j]
                finger_joints.append(j)
                b.joint_target_ke[dof] = 5000
                b.joint_target_kd[dof] = 50
                b.joint_effort_limit[dof] = 2.5
        self.groups = []
        self.assets = []
        self.tet_ranges = []
        self.carton = -999
        for i, (case, center) in enumerate(zip(cases, centers(cases))):
            asset = make_carton(case)
            self.assets.append(asset)
            start = b.particle_count
            tet_start = len(b.tet_indices)
            young = case.young
            b.add_soft_mesh(
                pos=wp.vec3(*center),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0),
                vertices=asset["vertices"],
                indices=asset["tets"].ravel(),
                density=650,
                k_mu=young / 2.5,
                k_lambda=young * 0.4,
                k_damp=0.03,
                particle_radius=0.0003,
                add_surface_mesh_edges=True,
                label=case.key,
            )
            self.tet_ranges.append((tet_start, len(b.tet_indices)))
            self.groups.append(
                dict(
                    name=f"carton_{i}",
                    start=start,
                    end=b.particle_count,
                    rest=asset["vertices"] + center,
                    faces=asset["faces"],
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
        cfg.inertia_mode = cha.InertiaMode.NONE
        cfg.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        cfg.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        cfg.joint_constraint_primal_iterations = 1
        cfg.constraint_alpha = 500
        cfg.enable_collision_hessian_non_diagonal = True
        cfg.enable_self_collision = True
        cfg.self_collision_radius = min(c.thickness for c in cases) * 0.35
        cfg.self_collision_edge_face_stiffness = 0.5
        self.material = base.PaperPlasticity(self.model)
        self.material.yield_strain.assign(
            np.concatenate([np.full(b - a, c.yield_strain, np.float32) for c, (a, b) in zip(cases, self.tet_ranges)])
        )
        self.solver.enable_joint_drive_feedback()
        self.state, self.next = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.coupling = base.ParallelJawCoupling(self.model)
        self.clock = wp.zeros(1, dtype=int)
        self.targets = wp.array(self.plan, dtype=float)
        self.rates = wp.array(np.gradient(self.plan, 1 / 60, axis=0).astype(np.float32), dtype=float)
        self.indices = wp.array(self.model.joint_target_q_start.numpy()[joints], dtype=int)
        self.dofs = wp.array(self.model.joint_qd_start.numpy()[joints], dtype=int)
        self.fingers = wp.array(self.model.joint_qd_start.numpy()[finger_joints], dtype=int)
        self.finger_dofs = self.fingers.numpy()
        self.peaks = wp.array(np.repeat([c.peak_force * args.force_scale for c in cases], 2).astype(np.float32))
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=8192,
            soft_contact_max=65536,
            soft_contact_gap=0.0015,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        cache = ROOT / "build/pharmacy-variant-contact-filter"
        cache.mkdir(exist_ok=True)
        self.filter = base.ContactFilter(self, base.filtered_kernels(cache))
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
                    self.peaks,
                    self.substeps,
                ],
                outputs=[self.control.joint_target_q, self.control.joint_target_qd, self.model.joint_effort_limit],
            )
            self.state.clear_forces()
            wp.launch(
                base.viscous_drag,
                dim=self.n,
                inputs=[self.state.particle_qd, self.model.particle_mass, 8.0],
                outputs=[self.state.particle_f],
            )
            self.pipeline.collide(self.state, self.contacts)
            self.coupling.apply(self.state, self.control, dt)
            self.solver.step(self.state, self.next, self.control, self.contacts, dt)
            self.state, self.next = self.next, self.state
            self.material.step(self.state, dt)


def record(sim, args, folder):
    folder.mkdir(parents=True, exist_ok=True)
    usd = base.SceneUSD(folder / "simulation.usdc", sim, round(args.seconds * 30))
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
                raise RuntimeError("Nonfinite variant state")
            sim.filter.counts_checked()
            if int(sim.contacts.soft_contact_count.numpy()[0]) > sim.contacts.soft_contact_max:
                raise RuntimeError("Soft-contact output overflow")
            if int(sim.contacts.rigid_contact_count.numpy()[0]) > sim.contacts.rigid_contact_max:
                raise RuntimeError("Rigid-contact output overflow")
            particles.append(q[: sim.n])
            bodies.append(body)
            plastic.append(history)
            passive = sim.control.joint_f.numpy()[sim.finger_dofs]
            coupling_efforts.append(passive)
            forces.append(sim.solver.joint_drive_effort.numpy()[sim.finger_dofs] - passive)
            usd.sample(frame // 2, q, body)
        if frame % 60 == 0:
            c = [q[g["start"] : g["end"]].mean(0).round(4).tolist() for g in sim.groups]
            print(f"{frame / 60:.1f}s centers={c} elapsed={time.perf_counter() - start:.1f}s", flush=True)
    usd.stage.GetRootLayer().Save()
    arrays = dict(
        particles=particles,
        bodies=bodies,
        plastic_strain=plastic,
        forces=forces,
        targets=sim.plan,
        coupling_efforts=coupling_efforts,
        **sim.coupling.archive(),
        ranges=[(g["start"], g["end"]) for g in sim.groups],
        tet_ranges=sim.tet_ranges,
        outer_counts=[a["outer_count"] for a in sim.assets],
        tets=sim.model.tet_indices.numpy(),
        rest=np.concatenate([g["rest"] for g in sim.groups]),
    )
    for i, a in enumerate(sim.assets):
        for field in ("faces", "uv", "panel", "wall"):
            arrays[f"{field}_{i}"] = a[field]
    np.savez_compressed(folder / "trajectory.npz", **arrays)
    report = dict(
        solver="cha.SolverCoupledHighAccuracy",
        seconds=args.seconds,
        fps=30,
        cases=[c.report() for c in sim.cases],
        particles=sim.n,
        tetrahedra=sim.model.tet_count,
        substeps=args.substeps,
        nonlinear_iterations=args.iterations,
        linear_iterations=args.linear_iterations,
        mass_damping_rate_per_s=8.0,
        force_scale=args.force_scale,
        wall_seconds=time.perf_counter() - start,
        gripper_coupling=sim.coupling.report(),
        internal_contents_modeled=False,
        scope="Estimated effective paperboard; welded overlap/end-flap approximation, no tearing or adhesive failure.",
    )
    (folder / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def live(sim, args, close_on_end=False):
    from newton.viewer import ViewerGL

    viewer = ViewerGL(vsync=True)
    try:
        viewer.set_model(sim.model)
        viewer.show_particles = False
        viewer.show_triangles = False
        viewer.set_camera(pos=wp.vec3(1.8, -2.2, 1.6), pitch=-14.0, yaw=135.0)
        faces = [wp.array(a["faces"].ravel(), dtype=int) for a in sim.assets]
        frame = 0
        end = round(args.seconds * 60)
        while viewer.is_running():
            viewer.begin_frame(frame / 60)
            viewer.log_state(sim.state)
            for case, g, f in zip(sim.cases, sim.groups, faces):
                viewer.log_mesh(
                    g["name"],
                    sim.state.particle_q[g["start"] : g["end"]],
                    f,
                    color=tuple(v / 255 for v in case.color),
                    backface_culling=False,
                )
            viewer.end_frame()
            if frame >= end and close_on_end:
                return True
            if frame < end and not viewer.is_paused():
                wp.capture_launch(sim.graph)
                frame += 1
            else:
                time.sleep(0.01)
        return frame >= end
    finally:
        viewer.close()
