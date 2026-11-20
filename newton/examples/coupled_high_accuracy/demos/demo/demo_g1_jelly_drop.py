"""G1 JELLY DROP: a G1 drops onto independent FEM jelly cubes.

Run: python demos/demo/demo_g1_jelly_drop.py
Check: python demos/demo/demo_g1_jelly_drop.py --viewer null --num-frames 1200 --test
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.kernels import sdf_box, sdf_cylinder, sdf_sphere

if __package__:
    from ..g1_jelly_drop.g1_robot import add_g1, measure_joint_errors
    from ..g1_jelly_drop.jelly_stack import (
        JELLY_BASE_HEIGHT,
        JELLY_DAMPING,
        JELLY_LAMBDA,
        JELLY_LAYER_GAP,
        JELLY_LAYERS,
        JELLY_MU,
        JELLY_SIDE,
        add_jelly_stack,
        signed_six_volumes,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "g1_jelly_drop"))
    from g1_robot import add_g1, measure_joint_errors
    from jelly_stack import (
        JELLY_BASE_HEIGHT,
        JELLY_DAMPING,
        JELLY_LAMBDA,
        JELLY_LAYER_GAP,
        JELLY_LAYERS,
        JELLY_MU,
        JELLY_SIDE,
        add_jelly_stack,
        signed_six_volumes,
    )


@wp.kernel
def observe_contact(
    count: wp.array[int],
    shapes: wp.array[int],
    soft_indices: wp.array[wp.vec3i],
    barycentric: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    particle_radius: wp.array[float],
    capacity: int,
    particle_start: int,
    particle_stop: int,
    observations: wp.array[int],
    particles: wp.array[wp.vec3],
    bodies: wp.array[wp.transform],
    record_hits: bool,
    penetration: wp.array[float],
):
    # Exclude affine proxy particles appended by the solver from robot-to-jelly measurements.
    hits = wp.int32(0)
    largest_depth = wp.float32(0.0)
    for i in range(wp.tid(), wp.min(count[0], capacity), 128):
        shape = shapes[i]
        particle = soft_indices[i][0]
        if shape >= 0 and shape_body[shape] >= 0 and particle >= particle_start and particle < particle_stop:
            soft_point = wp.vec3(0.0)
            radius = wp.float32(0.0)
            for corner in range(3):
                node = soft_indices[i][corner]
                if node >= 0:
                    soft_point += barycentric[i][corner] * particles[node]
                    radius = wp.max(radius, particle_radius[node])
            # Re-evaluate the primitive at the current pose: stored world normals become stale after the solve.
            pose = wp.transform_multiply(bodies[shape_body[shape]], shape_transform[shape])
            local_point = wp.transform_point(wp.transform_inverse(pose), soft_point)
            scale = shape_scale[shape]
            kind = shape_type[shape]
            distance = wp.float32(1.0e20)
            if kind == wp.static(int(newton.GeoType.BOX)):
                distance = sdf_box(local_point, scale[0], scale[1], scale[2])
            elif kind == wp.static(int(newton.GeoType.SPHERE)):
                distance = sdf_sphere(local_point, scale[0])
            elif kind == wp.static(int(newton.GeoType.CYLINDER)):
                distance = sdf_cylinder(local_point, scale[0], scale[1], wp.static(int(newton.Axis.Z)))
            # The broad phase's 5 cm search margin is not physical contact thickness.
            if record_hits and distance < radius + shape_margin[shape]:
                hits += 1
            depth = wp.max(0.0, -distance)
            largest_depth = wp.max(largest_depth, depth)
    wp.atomic_max(penetration, 0, largest_depth)
    if wp.tid() == 0 and count[0] > capacity:
        penetration[1] = 1.0
    if hits > 0:
        previous_hits = wp.atomic_add(observations, 1, hits)
        if previous_hits == 0:
            observations[2] = observations[0]


@wp.kernel
def advance_contact_clock(observations: wp.array[int]):
    observations[0] += 1


@wp.kernel
def material_lattice(
    q: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    rest_lengths: wp.array[float],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    i = wp.tid()
    a, b = q[edges[i][0]], q[edges[i][1]]
    strain = wp.clamp(wp.abs(wp.length(b - a) / rest_lengths[i] - 1.0) * 4.0, 0.0, 1.0)
    starts[i] = a
    ends[i] = b
    colors[i] = wp.lerp(wp.vec3(0.03, 0.20, 0.40), wp.vec3(1.0, 0.16, 0.04), strain)


@wp.kernel
def gather_surface_points(
    particles: wp.array[wp.vec3],
    source_indices: wp.array[wp.int32],
    surface_points: wp.array[wp.vec3],
):
    i = wp.tid()
    surface_points[i] = particles[source_indices[i]]


@wp.kernel
def record_descent(
    velocities: wp.array[wp.spatial_vector],
    root: int,
    contacts: wp.array[int],
    descent: wp.array[float],
):
    speed = wp.max(0.0, -velocities[root][2])
    if contacts[1] == 0:
        descent[0] = wp.max(descent[0], speed)
    elif descent[2] == 0.0:
        descent[1] = speed
        descent[2] = 1.0


def prepare_surface_batches(triangles: np.ndarray, groups: dict) -> tuple[np.ndarray, list[dict]]:
    """Group disconnected surface meshes by exact color while retaining their original particle references."""
    by_color = {}
    for group in groups.values():
        first, last = group["triangle_range"]
        source, indices = np.unique(triangles[first:last], return_inverse=True)
        by_color.setdefault(tuple(group["color"]), []).append((source, indices))
    sources, batches = [], []
    total_points = 0
    for color, meshes in by_color.items():
        point_count, batch_indices = 0, []
        for source, indices in meshes:
            # Adjacent cubes must keep distinct render vertices so separation and deformation remain visible.
            sources.append(source)
            batch_indices.append(indices.ravel() + point_count)
            point_count += len(source)
        batches.append(
            {
                "name": f"jelly_surface_{len(batches):02d}",
                "color": color,
                "point_range": (total_points, total_points + point_count),
                "indices": np.concatenate(batch_indices).astype(np.int32),
            }
        )
        total_points += point_count
    return np.concatenate(sources).astype(np.int32), batches


class Example:
    def __init__(self, viewer, args):
        from newton.solvers import coupled_high_accuracy as cha

        if not wp.get_device().is_cuda:
            raise ValueError("Jelly Drop requires a CUDA device")
        if args.substeps < 1 or args.layers < 1:
            raise ValueError("substeps and layers must be positive")
        if not math.isfinite(args.contact_kd) or args.contact_kd < 0.0:
            raise ValueError("contact-kd must be finite and nonnegative")
        if not math.isfinite(args.jelly_width) or args.jelly_width <= 0.0:
            raise ValueError("jelly-width must be positive and finite")
        if not math.isfinite(args.drop_height) or not 0.0 <= args.drop_height <= 50.0:
            raise ValueError("drop-height must be between 0 and 50 metres")
        if not math.isfinite(args.tilt) or not -45.0 <= args.tilt <= 45.0:
            raise ValueError("tilt must be between -45 and 45 degrees")
        self.viewer, self.args = viewer, args
        self.frame_dt = 1.0 / 60.0
        self.substeps = args.substeps
        self.dt = self.frame_dt / self.substeps
        self.sim_time = 0.0
        self.show_lattice = False
        self.follow_camera = True
        self.reset_requested = False
        self.reset_key_down = False

        builder = cha.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=wp.vec3(0.0, 0.0, -9.81),
            affine_body_particle_radius=0.0,
        )
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        # Keep rigid-to-soft contact at the same scale as the global implicit contact solve.
        # Lower shape values are selected by min(global, shape) and let the robot or stack tunnel through.
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = args.contact_kd
        builder.default_shape_cfg.mu = 0.2
        floor = builder.default_shape_cfg.copy()
        floor.is_visible = False
        builder.add_ground_plane(cfg=floor)
        self.jelly_stack = add_jelly_stack(
            builder,
            side=args.jelly_width,
            layers=args.layers,
            layer_gap=args.layer_gap,
            base_z=args.base_height,
            shear_modulus=args.jelly_mu,
            lame_lambda=args.jelly_lambda,
            damping=args.jelly_damping,
        )
        self.groups = self.jelly_stack["groups"]
        self.robot = add_g1(builder, (0.02, 0.0, 0.0), tilt_radians=math.radians(args.tilt))
        # Clearance refers to the lowest collision point in the actual bent-knee, tilted pose, not the pelvis.
        builder.joint_q[self.robot.root_coord_start + 2] = (
            self.jelly_stack["top_z"] + args.drop_height - self.robot.local_bounds[0, 2]
        )
        span = args.layers * args.jelly_width + (args.layers - 1) * self.jelly_stack["block_gap"]
        self.scene_top = float(builder.joint_q[self.robot.root_coord_start + 2] + self.robot.local_bounds[1, 2])
        # Fit both the configurable pyramid and the fixed-size robot throughout the drop.
        self.scene_scale = max(span / 2.5225, self.scene_top / 4.568, 0.45)
        self.impact_scale = max(span / 2.5225, 0.75)
        self.zone_radius = max(1.0, math.sqrt(2.0) * span / 2.0 + 0.7)
        self.model = builder.finalize()
        # A firm contact response keeps limbs supported by the FEM blocks during landing.
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = args.contact_kd
        self.model.soft_contact_mu = 0.3
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        cfg = self.solver.config
        cfg.air_density = 0.0
        cfg.wind_velocity = (0.0, 0.0, 0.0)
        cfg.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        cfg.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        cfg.nonlinear_solver = cha.NonlinearSolver.NEWTON
        cfg.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        cfg.enable_collision_hessian_non_diagonal = True
        cfg.joint_constraint_primal_iterations = 4
        cfg.nonlinear_iterations = 20
        cfg.linear_iterations = 15
        cfg.enable_self_collision = True
        cfg.self_collision_radius = self.jelly_stack["contact_radius"]
        cfg.self_collision_vertex_face_stiffness = 2.0
        cfg.self_collision_edge_edge_stiffness = 0.25
        cfg.self_collision_edge_face_stiffness = 1.0
        cfg.self_collision_particle_particle_stiffness = 1.0
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
            self.solver.sync_affine_body_state(state)
        self.control = self.model.control()
        self.initial_particles = self.state_0.particle_q.numpy().copy()
        self.initial_bodies = self.state_0.body_q.numpy().copy()
        self.state_buffers = (self.state_0, self.state_1)
        self.reset_arrays = {
            name: wp.clone(getattr(self.state_0, name))
            for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd")
        }
        self.observations = wp.zeros(3, dtype=int, device=self.model.device)
        self.descent = wp.zeros(3, dtype=float, device=self.model.device)
        self.penetration = wp.zeros(2, dtype=float, device=self.model.device)
        # Generate candidates before a fast limb crosses the surface during the next discrete substep.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.05,
            enable_rigid_soft_full_surface_contact=True,
            soft_contact_max=max(32768, 4 * self.model.particle_count),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.setup_rendering()
        self.graphs = []
        self.graph_index = 0
        if args.cuda_graph:
            original = self.state_0, self.state_1
            try:
                for _ in range(2 if self.substeps % 2 else 1):
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        self.simulate()
                    self.graphs.append(capture.graph)
            finally:
                self.state_0, self.state_1 = original

    def camera(self, name):
        height = self.jelly_stack["top_z"]
        if name == "Stage":
            distance = 10.65 * self.scene_scale
            yaw = math.radians(130.0)
            self.viewer.set_camera(
                pos=wp.vec3(
                    -distance * math.cos(yaw),
                    -distance * math.sin(yaw),
                    0.4 * self.scene_top + distance * math.tan(math.radians(20.0)),
                ),
                pitch=-20.0,
                yaw=130.0,
            )
        else:
            self.viewer.set_camera(
                pos=wp.vec3(4.6 * self.impact_scale, -5.8 * self.impact_scale, height + 1.7 * self.impact_scale),
                pitch=-19.0,
                yaw=129.0,
            )

    def setup_rendering(self):
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.camera("Stage")
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.set_title("G1 JELLY DROP")
            renderer.sky_upper = (0.28, 0.32, 0.42)
            renderer.sky_lower = (0.69, 0.73, 0.79)
            renderer.ambient_sky = (0.72, 0.78, 0.90)
            renderer.ambient_ground = (0.30, 0.29, 0.36)
            renderer.exposure = 1.15
            renderer.shadow_radius = 1.5
            renderer.shadow_extents = max(9.0, 2.0 * self.zone_radius)
            self.viewer.register_ui_callback(self.render_ui, position="side")
        device = self.model.device
        self.viewer.log_mesh(
            "studio_floor",
            wp.array(
                [(-80, -80, -0.008), (80, -80, -0.008), (80, 80, -0.008), (-80, 80, -0.008)],
                dtype=wp.vec3,
                device=device,
            ),
            wp.array([0, 1, 2, 0, 2, 3], dtype=int, device=device),
            color=(0.56, 0.59, 0.66),
            roughness=0.72,
            metallic=0.08,
        )
        triangles = self.model.tri_indices.numpy()
        source_indices, batches = prepare_surface_batches(triangles, self.groups)
        self.surface_source_indices = wp.array(source_indices, dtype=wp.int32, device=device)
        self.surface_points = wp.empty(len(source_indices), dtype=wp.vec3, device=device)
        self.surface_batches = []
        for batch in batches:
            start, stop = batch["point_range"]
            self.surface_batches.append(
                (
                    batch["name"],
                    self.surface_points[start:stop],
                    wp.array(batch["indices"], dtype=wp.int32, device=device),
                    batch["color"],
                )
            )
        edge_sets = []
        for group in self.groups.values():
            start, _ = group["particle_range"]
            tets = group["tet_indices"]
            edges = np.concatenate([tets[:, pair] for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
            edge_sets.append(np.unique(np.sort(edges, axis=1), axis=0) + start)
        edges = np.concatenate(edge_sets)
        lengths = np.linalg.norm(self.initial_particles[edges[:, 1]] - self.initial_particles[edges[:, 0]], axis=1)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=device)
        self.rest_lengths = wp.array(lengths, dtype=float, device=device)
        self.edge_starts = wp.empty(len(edges), dtype=wp.vec3, device=device)
        self.edge_ends = wp.empty_like(self.edge_starts)
        self.edge_colors = wp.empty_like(self.edge_starts)

    def track_descent(self):
        pelvis = self.state_0.body_q.numpy()[self.robot.root_body, :3]
        center = pelvis.copy()
        # Frame both the falling robot and its landing area.
        height = float(pelvis[2]) + 0.6
        center[2] = 0.5 * max(height, 1.0)
        distance = max(7.0, height * 1.5)
        pitch, yaw = math.radians(18.0), math.radians(130.0)
        self.viewer.set_camera(
            pos=wp.vec3(
                center[0] - distance * math.cos(yaw),
                center[1] - distance * math.sin(yaw),
                center[2] + distance * math.tan(pitch),
            ),
            pitch=-18.0,
            yaw=130.0,
        )

    def step(self):
        pressed = self.viewer.is_key_down("r")
        if self.reset_requested or (pressed and not self.reset_key_down):
            self.reset()
            self.reset_key_down = pressed
            return
        self.reset_key_down = pressed
        if self.graphs:
            wp.capture_launch(self.graphs[self.graph_index])
            if self.substeps % 2:
                self.state_0, self.state_1 = self.state_1, self.state_0
                self.graph_index = 1 - self.graph_index
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def reset(self):
        # Restore buffers in place: captured graphs and native solver spans retain their original allocations.
        self.state_0, self.state_1 = self.state_buffers
        for state in self.state_buffers:
            for name, initial in self.reset_arrays.items():
                wp.copy(getattr(state, name), initial)
            state.clear_forces()
            self.solver.sync_affine_body_state(state)
        self.observations.zero_()
        self.descent.zero_()
        self.penetration.zero_()
        self.graph_index = 0
        self.sim_time = 0.0
        self.reset_requested = False

    def simulate(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            contact_inputs = [
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_indices,
                self.contacts.soft_contact_barycentric,
                self.model.shape_body,
                self.model.shape_type,
                self.model.shape_transform,
                self.model.shape_scale,
                self.model.shape_margin,
                self.model.particle_radius,
                self.contacts.soft_contact_max,
                *self.jelly_stack["particle_range"],
                self.observations,
            ]
            # Preserve initial overlap and the converged overlap, counting effective contacts only once per substep.
            wp.launch(
                observe_contact,
                dim=128,
                inputs=[*contact_inputs, self.state_0.particle_q, self.state_0.body_q, False, self.penetration],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            wp.launch(advance_contact_clock, dim=1, inputs=[self.observations], device=self.model.device)
            wp.launch(
                observe_contact,
                dim=128,
                inputs=[*contact_inputs, self.state_1.particle_q, self.state_1.body_q, True, self.penetration],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            record_descent,
            dim=1,
            inputs=[self.state_0.body_qd, self.robot.root_body, self.observations, self.descent],
            device=self.model.device,
        )

    def render_ui(self, imgui):
        imgui.text("J E L L Y  D R O P")
        imgui.text("G1 / JELLY MOUNTAIN")
        imgui.separator()
        if imgui.button("Restart drop [R]"):
            self.reset_requested = True
        _, self.show_lattice = imgui.checkbox("FEM lattice / strain", self.show_lattice)
        _, self.follow_camera = imgui.checkbox("Follow descent", self.follow_camera)
        if imgui.button("Stage"):
            self.follow_camera = False
            self.camera("Stage")
        imgui.same_line()
        if imgui.button("Impact"):
            self.follow_camera = False
            self.camera("Impact")
        imgui.separator()
        hits = self.observations.numpy()
        pelvis = self.state_0.body_q.numpy()[self.robot.root_body, 2]
        imgui.text(f"Time             {self.sim_time:5.2f} s")
        imgui.text(f"Pelvis height    {pelvis:5.2f} m")
        imgui.text(f"Foot clearance   {self.args.drop_height:5.2f} m")
        imgui.text(f"Robot mass       {self.robot.total_mass:5.2f} kg")
        imgui.text(f"Peak descent     {self.descent.numpy()[0]:5.2f} m/s")
        imgui.text(f"Peak penetration {1000.0 * self.penetration.numpy()[0]:5.1f} mm")
        imgui.text(f"Jelly width      {self.jelly_stack['block_side']:5.2f} m")
        imgui.text(f"Pyramid layers   {self.jelly_stack['layer_count']:5d}")
        imgui.text(f"Free jelly cubes {self.jelly_stack['block_count']:5d}")
        imgui.text(f"Tetrahedra       {sum(len(g['tet_indices']) for g in self.groups.values()):5d}")
        imgui.text(f"Shear modulus   {self.args.jelly_mu:g}")
        imgui.text(f"Material damping {self.args.jelly_damping:g} | Contact damping {self.args.contact_kd:g}")
        imgui.text(f"First contact    {hits[2] * self.dt:.2f} s" if hits[1] else "First contact    falling...")
        imgui.separator()
        imgui.text("R : reset the complete drop")
        imgui.text("Space : pause / resume")
        imgui.text("WASD + mouse : camera")

    def render(self):
        pressed = self.viewer.is_key_down("r")
        if self.reset_requested or (pressed and not self.reset_key_down):
            self.reset()
        self.reset_key_down = pressed
        if self.follow_camera:
            self.track_descent()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            gather_surface_points,
            dim=len(self.surface_source_indices),
            inputs=[self.state_0.particle_q, self.surface_source_indices],
            outputs=[self.surface_points],
            device=self.model.device,
        )
        for name, points, indices, color in self.surface_batches:
            self.viewer.log_mesh(
                name,
                points,
                indices,
                color=color,
                roughness=0.19,
                metallic=0.03,
                backface_culling=False,
                hidden=self.show_lattice,
            )
        if self.show_lattice:
            wp.launch(
                material_lattice,
                dim=len(self.edges),
                inputs=[self.state_0.particle_q, self.edges, self.rest_lengths],
                outputs=[self.edge_starts, self.edge_ends, self.edge_colors],
                device=self.model.device,
            )
            self.viewer.log_lines("jelly_lattice", self.edge_starts, self.edge_ends, self.edge_colors)
        else:
            self.viewer.log_lines("jelly_lattice", None, None, None)
        self.viewer.end_frame()

    def test_final(self):
        points, bodies = self.state_0.particle_q.numpy(), self.state_0.body_q.numpy()
        arrays = (points, bodies, self.state_0.particle_qd.numpy(), self.state_0.body_qd.numpy())
        if not all(np.isfinite(a).all() for a in arrays):
            raise AssertionError("The jelly drop contains a nonfinite state")
        penetration = self.penetration.numpy()
        if penetration[1] != 0.0:
            raise AssertionError("Soft contacts exceeded the collision buffer capacity")
        if penetration[0] > 0.03:
            raise AssertionError(f"A jelly contact point penetrated a G1 collider by {penetration[0]:.4f}m")
        joint_errors = measure_joint_errors(self.model, bodies, self.robot)
        if joint_errors["max_anchor_separation"] > 0.06 or joint_errors["max_hinge_axis_angle"] > 0.15:
            raise AssertionError(f"G1 joint constraints separated: {joint_errors}")
        minimum = 1.0
        max_slide = 0.0
        for name, group in self.groups.items():
            a, b = group["particle_range"]
            ratios = signed_six_volumes(points[a:b], group["tet_indices"]) / group["initial_signed_volumes"]
            if ratios.min() <= 0.0:
                raise AssertionError(f"{name} contains an inverted tetrahedron: {ratios.min():.5f}")
            minimum = min(minimum, float(ratios.min()))
            max_slide = max(
                max_slide, float(np.linalg.norm(points[a:b].mean(axis=0)[:2] - np.asarray(group["center"])[:2]))
            )
        a, b = self.jelly_stack["particle_range"]
        jelly_min_z = float(points[a:b, 2].min())
        pelvis_z = float(bodies[self.robot.root_body, 2])
        if jelly_min_z < -0.06 or pelvis_z < 0.0:
            raise AssertionError(
                f"Ground penetration: lowest jelly node z={jelly_min_z:.4f}m, pelvis z={pelvis_z:.4f}m"
            )
        hits = self.observations.numpy()
        if self.sim_time > max(8.0, self.args.drop_height):
            if hits[1] == 0:
                raise AssertionError("G1 never contacted the jelly during descent")
        if self.sim_time > 2.0:
            if self.initial_bodies[self.robot.root_body, 2] - bodies[self.robot.root_body, 2] < 0.3:
                raise AssertionError("The floating G1 did not drop")
        print(
            f"t={self.sim_time:.2f}s; min tet ratio={minimum:.3f}; max cube slide={max_slide:.3f}m; "
            f"G1 contacts={hits[1]}; first contact={hits[2] * self.dt:.3f}s; "
            f"joint gap={joint_errors['max_anchor_separation']:.5f}m; "
            f"peak descent={self.descent.numpy()[0]:.2f}m/s; max penetration={penetration[0] * 1000.0:.1f}mm"
        )

    def test_post_step(self):
        if round(self.sim_time / self.frame_dt) % 10 == 0:
            self.test_final()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        parser.add_argument(
            "--drop-height", type=float, default=0.5, help="Initial foot clearance above the jelly, in metres"
        )
        parser.add_argument("--tilt", type=float, default=0.0, help="Initial G1 forward pitch in degrees")
        parser.add_argument(
            "--jelly-width",
            type=float,
            default=JELLY_SIDE,
            help="Edge length of each jelly cube in metres",
        )
        parser.add_argument(
            "--layers",
            type=int,
            default=JELLY_LAYERS,
            help="Pyramid layers; each tier contains one cube layer and shrinks by one cube per side",
        )
        parser.add_argument(
            "--layer-gap",
            type=float,
            default=JELLY_LAYER_GAP,
            help="Initial vertical surface gap between jelly layers, in metres",
        )
        parser.add_argument(
            "--base-height",
            type=float,
            default=JELLY_BASE_HEIGHT,
            help="Initial clearance between the bottom jelly surface and the ground, in metres",
        )
        parser.add_argument(
            "--jelly-mu", type=float, default=JELLY_MU, help="FEM shear modulus; higher restores shape faster."
        )
        parser.add_argument(
            "--jelly-lambda", type=float, default=JELLY_LAMBDA, help="FEM volume stiffness (Lame lambda)."
        )
        parser.add_argument(
            "--jelly-damping", type=float, default=JELLY_DAMPING, help="FEM viscosity; lower retains wobble."
        )
        parser.add_argument(
            "--contact-kd", type=float, default=25.0, help="Normal contact damping for jelly/ground/G1."
        )
        # Smaller steps resolve the lightly damped rebound and reduce implicit integration's numerical damping.
        parser.add_argument("--substeps", type=int, default=2)
        parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
