# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure cloth mechanical energy and a common-substep diagnostic objective.

This observer never changes the solver implementation or demo defaults.
Contact potentials use retained detector records, not an exhaustive collision
oracle. Frozen friction is a diagnostic surrogate, not the changing-contact
solver's exact scalar objective. All reported energies use joules.
"""

import argparse
import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2.particle_surface_cache import _prepare_anchor_angles
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as kernels
from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 as scene,
)


@wp.kernel(enable_backward=False)
def self_geometry(
    q: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array2d[int],
    pairs: wp.array2d[int],
    vt_count: int,
    epsilon: float,
    ids: wp.array2d[int],
    weights: wp.array[wp.vec4],
    separation: wp.array[wp.vec3],
    distances: wp.array[float],
):
    i = wp.tid()
    a, b = pairs[i, 0], pairs[i, 1]
    if i < vt_count:
        i0, i1, i2 = triangles[b, 0], triangles[b, 1], triangles[b, 2]
        cp, bary, _feature = kernels.triangle_closest_point(q[i0], q[i1], q[i2], q[a])
        ids[i, 0] = i0
        ids[i, 1] = i1
        ids[i, 2] = i2
        ids[i, 3] = a
        weights[i] = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        separation[i] = q[a] - cp
        distances[i] = wp.length(q[a] - cp)
    else:
        i0, i1, i2, i3 = edges[a, 2], edges[a, 3], edges[b, 2], edges[b, 3]
        st = wp.closest_point_edge_edge(q[i0], q[i1], q[i2], q[i3], epsilon)
        ids[i, 0] = i0
        ids[i, 1] = i1
        ids[i, 2] = i2
        ids[i, 3] = i3
        weights[i] = wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1])
        separation[i] = q[i0] + st[0] * (q[i1] - q[i0]) - q[i2] - st[1] * (q[i3] - q[i2])
        distances[i] = st[2]


def normal_energy(distance, radius, stiffness):
    """Integrate the exact piecewise self-normal force, with E(radius)=0."""
    d = np.asarray(distance, dtype=np.float64)
    tau, minimum = radius * 0.5, 1.0e-5
    scale = stiffness * tau * tau
    value = 0.5 * stiffness * np.maximum(radius - d, 0.0) ** 2
    logarithmic = d < tau
    value = np.where(logarithmic, 0.5 * scale + scale * np.log(tau / np.maximum(d, minimum)), value)
    offset = np.minimum(d - minimum, 0.0)
    return value - scale / minimum * offset + 0.5 * scale / minimum**2 * offset**2


def normal_load(distance, radius, stiffness):
    """Return the positive normal load from the same piecewise potential."""
    d = np.asarray(distance, dtype=np.float64)
    tau, minimum = radius * 0.5, 1.0e-5
    load = stiffness * np.maximum(radius - d, 0.0)
    load = np.where(d < tau, stiffness * tau**2 / np.maximum(d, minimum), load)
    return np.where(d < minimum, stiffness * tau**2 * (2 * minimum - d) / minimum**2, load)


def smooth_friction(length, epsilon):
    """Integrate smooth slip force with the diagnostic convention D(0)=0."""
    return np.where(length < epsilon, length**2 / epsilon - length**3 / (3 * epsilon**2), length - epsilon / 3)


def save_csv(path, rows):
    """Write numeric samples without interpolation or smoothing."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EnergyObserver:
    """Evaluate elastic energy in float64 from the solver's model arrays."""

    def __init__(self, model):
        self.model = model
        self.mass = model.particle_mass.numpy().astype(np.float64)
        self.gravity = model.gravity.numpy()[0].astype(np.float64)
        self.faces = model.tri_indices.numpy()
        self.poses = model.tri_poses.numpy().astype(np.float64)
        self.areas = model.tri_areas.numpy().astype(np.float64)
        self.materials = model.tri_materials.numpy().astype(np.float64)
        self.edges = model.edge_indices.numpy()
        self.valid_edges = (self.edges[:, 0] >= 0) & (self.edges[:, 1] >= 0)
        self.bending = model.edge_bending_properties.numpy().astype(np.float64)
        self.rest_angles = model.edge_rest_angle.numpy().astype(np.float64)
        self.rest_lengths = model.edge_rest_length.numpy().astype(np.float64)
        self.initial_gravity = -np.sum(self.mass[:, None] * model.particle_q.numpy() * self.gravity)
        self.angle_positions = wp.clone(model.particle_q)
        self.angle_values = wp.empty(model.edge_count, dtype=wp.vec2, device=model.device)

    def metric(self, q):
        """Return the three independent components of F-transpose-F."""
        a, b, c = (q[self.faces[:, i]] for i in range(3))
        f = np.einsum("nki,nij->nkj", np.stack((b - a, c - a), axis=2), self.poses)
        c00 = np.sum(f[:, :, 0] ** 2, axis=1)
        c11 = np.sum(f[:, :, 1] ** 2, axis=1)
        c01 = np.sum(f[:, :, 0] * f[:, :, 1], axis=1)
        return np.column_stack((c00, c11, c01))

    def angles(self, q):
        """Use the original bending orientation and degeneracy threshold."""
        self.angle_positions.assign(np.asarray(q, dtype=np.float32))
        wp.launch(
            _prepare_anchor_angles,
            dim=self.model.edge_count,
            inputs=[self.angle_positions, self.model.edge_indices],
            outputs=[self.angle_values],
            device=self.model.device,
        )
        values = self.angle_values.numpy()
        return values[:, 0].astype(np.float64), values[:, 1] > 0

    def elastic(self, q, anchor=None, dt=None):
        """Return zero-at-rest membrane energy, bending and optional damping potentials."""
        c = self.metric(q)
        mu, lam = self.materials[:, 0], self.materials[:, 0] + self.materials[:, 1]
        alpha = 1 + mu / np.maximum(lam, 1e-6)
        jacobian = np.sqrt(np.maximum(c[:, 0] * c[:, 1] - c[:, 2] ** 2, 1e-20))
        psi = 0.5 * mu * (c[:, 0] + c[:, 1] - 2) + 0.5 * lam * ((jacobian - alpha) ** 2 - (1 - alpha) ** 2)
        angle, valid = self.angles(q)
        bend = 0.5 * self.bending[:, 0] * self.rest_lengths * (angle - self.rest_angles) ** 2
        values = {"membrane_J": float(np.sum(self.areas * psi)), "bending_J": float(np.sum(bend[valid]))}
        if anchor is not None:
            dc = c - self.metric(anchor)
            values["membrane_damping_J"] = float(
                np.sum(self.areas * self.materials[:, 2] / (2 * dt) * np.sum(dc**2 * [1, 1, 2], axis=1))
            )
            old_angle, old_valid = self.angles(anchor)
            delta = (angle - old_angle + np.pi) % (2 * np.pi) - np.pi
            values["bending_damping_J"] = float(
                np.sum((0.5 * self.bending[:, 1] * self.rest_lengths / dt * delta**2)[valid & old_valid])
            )
        return values

    def mechanical(self, state):
        """Measure cloth mechanical energy; the robot is an external moving boundary."""
        q, v = state.particle_q.numpy().astype(np.float64), state.particle_qd.numpy().astype(np.float64)
        values = self.elastic(q)
        values["kinetic_J"] = float(np.sum(0.5 * self.mass[:, None] * v**2))
        values["gravity_relative_J"] = float(-np.sum(self.mass[:, None] * q * self.gravity) - self.initial_gravity)
        values["cloth_mechanical_J"] = sum(values.values())
        return values


class ContactObserver:
    """Keep one diagnostic contact set, and optionally lag its friction data."""

    def __init__(self, core, state, contacts):
        self.core, self.model, self.contacts = core, core.model, contacts
        self.radius = core.particle_self_contact_radius
        self.ke, self.kd, self.mu = self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu
        if core._use_soft_contact_material_source:
            material = core._soft_contact_materials.numpy()[int(core._soft_contact_material_index.numpy()[0])]
            self.ke, self.kd, self.mu = map(float, material)
        detector = core.trimesh_collision_detector
        data, self.overflow = [], 0
        for prefix in ("vertex_colliding_triangles", "edge_colliding_edges"):
            counts = getattr(detector, prefix + "_count").numpy()
            offsets = getattr(detector, prefix + "_offsets").numpy()
            capacity = np.diff(offsets)
            self.overflow += int(np.count_nonzero(counts > capacity))
            counts = np.minimum(counts, capacity)
            slots = np.concatenate(
                [np.arange(offset, offset + count) for offset, count in zip(offsets[:-1], counts, strict=True)]
            )
            records = getattr(detector, prefix).numpy().reshape(-1, 2)[slots]
            data.append(np.unique(records, axis=0))
        self.vt_count = len(data[0])
        ee = data[1]
        canonical, multiplicity = np.unique(np.sort(ee, axis=1), axis=0, return_counts=True)
        self.asymmetric_ee = int(np.count_nonzero(multiplicity != 2))
        pairs = np.concatenate((data[0], canonical))
        self.pairs = wp.array(pairs, dtype=int, device=self.model.device)
        n = len(pairs)
        self.ids = wp.empty((n, 4), dtype=int, device=self.model.device)
        self.weights = wp.empty(n, dtype=wp.vec4, device=self.model.device)
        self.separation = wp.empty(n, dtype=wp.vec3, device=self.model.device)
        self.distances = wp.empty(n, dtype=float, device=self.model.device)
        self.body_q = state.body_q.numpy().copy()
        self.shape_body = self.model.shape_body.numpy()
        count = int(contacts.soft_contact_count.numpy()[0])
        count = min(count, contacts.soft_contact_shape.shape[0])
        self.shapes = contacts.soft_contact_shape.numpy()[:count]
        self.corners = contacts.soft_contact_indices.numpy()[:count]
        self.body_indices = np.arange(count)
        self.bary = contacts.soft_contact_barycentric.numpy()[:count]
        self.body_points = contacts.soft_contact_body_pos.numpy()[:count]
        self.normals = contacts.soft_contact_normal.numpy()[:count].astype(np.float64)
        self.margin = self.model.shape_margin.numpy()[self.shapes]
        self.particle_radius = self.model.particle_radius.numpy()
        self.body_ke = core.body_particle_contact_penalty_k.numpy()[self.body_indices].astype(np.float64)
        self.body_mu = core.body_particle_contact_material_mu.numpy()[self.body_indices].astype(np.float64)
        self.body_kd = core.body_particle_contact_material_kd.numpy()[self.body_indices].astype(np.float64)
        self.anchor = core.particle_q_prev.numpy().astype(np.float64)
        distance = self.geometry(state.particle_q)
        self.frozen_ids, self.frozen_weights = self.ids.numpy(), self.weights.numpy().astype(np.float64)
        sep = self.separation.numpy().astype(np.float64)
        self.frozen_normals = sep / np.maximum(np.linalg.norm(sep, axis=1, keepdims=True), 1e-30)
        self.frozen_load = normal_load(distance, self.radius, self.ke)
        self.body_world = np.zeros_like(self.body_points, dtype=np.float64)
        for i, (shape, local) in enumerate(zip(self.shapes, self.body_points, strict=True)):
            body = self.shape_body[shape]
            transform = wp.transform_identity() if body < 0 else wp.transform(*self.body_q[body])
            self.body_world[i] = wp.transform_point(transform, wp.vec3(*local))
        self.body_load = self.body_ke * np.maximum(-self.body_gap(state.particle_q.numpy()), 0)

    def geometry(self, q):
        """Evaluate existing geometry primitives without updating the detector."""
        if self.pairs.shape[0]:
            wp.launch(
                self_geometry,
                dim=self.pairs.shape[0],
                inputs=[
                    q,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.pairs,
                    self.vt_count,
                    self.core.trimesh_collision_detector.edge_edge_parallel_epsilon,
                ],
                outputs=[self.ids, self.weights, self.separation, self.distances],
                device=self.model.device,
            )
        return self.distances.numpy().astype(np.float64)

    def body_point(self, q):
        """Decode the point or barycentric edge/face contact record."""
        valid = self.corners >= 0
        return np.sum(q[np.maximum(self.corners, 0)] * (self.bary * valid)[:, :, None], axis=1)

    def body_gap(self, q):
        radius = np.max(np.where(self.corners >= 0, self.particle_radius[np.maximum(self.corners, 0)], 0), axis=1)
        return np.sum(self.normals * (self.body_point(q) - self.body_world), axis=1) - radius - self.margin

    def potential(self, q, *, frozen=False, dt=None):
        """Sum unique self-pair potentials; report asymmetry separately."""
        distance = self.geometry(q)
        q_np = q.numpy().astype(np.float64)
        ke = self.body_ke if frozen else self.core.body_particle_contact_penalty_k.numpy()[self.body_indices]
        values = {
            "self_normal_J": float(np.sum(normal_energy(distance, self.radius, self.ke))),
            "body_normal_J": float(np.sum(0.5 * ke * np.maximum(-self.body_gap(q_np), 0) ** 2)),
        }
        if frozen:
            relative = np.sum((q_np - self.anchor)[self.frozen_ids] * self.frozen_weights[:, :, None], axis=1)
            normal_motion = np.sum(relative * self.frozen_normals, axis=1)
            slip = np.linalg.norm(relative - normal_motion[:, None] * self.frozen_normals, axis=1)
            values["lagged_self_friction_J"] = float(
                np.sum(self.mu * self.frozen_load * smooth_friction(slip, self.core.friction_epsilon * dt))
            )
            values["lagged_self_damping_J"] = float(
                np.sum(0.5 * self.kd / dt * np.minimum(normal_motion, 0) ** 2 * (self.frozen_load > 0))
            )
            # This probe uses a terminal substep, where the prescribed boundary
            # is nearly stationary. Still include exact finite pose translation.
            body_prev = self.core._external_body_q_prev
            previous = body_prev.numpy() if body_prev is not None else self.body_q
            motion = np.zeros_like(self.body_world)
            for i, (shape, local) in enumerate(zip(self.shapes, self.body_points, strict=True)):
                body = self.shape_body[shape]
                if body >= 0:
                    before = wp.transform_point(wp.transform(*previous[body]), wp.vec3(*local))
                    motion[i] = self.body_world[i] - np.asarray(before)
            relative = self.body_point(q_np - self.anchor) - motion
            dn = np.sum(relative * self.normals, axis=1)
            slip = np.linalg.norm(relative - dn[:, None] * self.normals, axis=1)
            values["lagged_body_friction_J"] = float(
                np.sum(self.body_mu * self.body_load * smooth_friction(slip, self.core.friction_epsilon * dt))
            )
            values["lagged_body_damping_J"] = float(
                np.sum(0.5 * self.body_kd / dt * np.minimum(dn, 0) ** 2 * (self.body_load > 0))
            )
        return values


class ProbeComplete(Exception):
    """End the disposable example after its common-state substep diagnostic."""


@wp.kernel(enable_backward=False)
def elastic_force_reference(
    q: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    faces: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    edges: wp.array2d[int],
    rest_angles: wp.array[float],
    rest_lengths: wp.array[float],
    bending: wp.array2d[float],
    force: wp.array[wp.vec3],
):
    vertex = wp.tid()
    total = wp.vec3(0.0)
    for face in range(faces.shape[0]):
        for order in range(3):
            if faces[face, order] == vertex:
                f, _h = kernels.evaluate_neo_hookean_membrane_force_hessian(
                    face,
                    order,
                    q,
                    anchor,
                    faces,
                    poses[face],
                    areas[face],
                    materials[face, 0],
                    materials[face, 1],
                    materials[face, 2],
                    0.01,
                )
                total += f
    for edge in range(edges.shape[0]):
        for order in range(4):
            if edges[edge, order] == vertex:
                f, _h = kernels.evaluate_dihedral_angle_based_bending_force_hessian(
                    edge, order, q, anchor, edges, rest_angles, rest_lengths, bending[edge, 0], bending[edge, 1], 0.01
                )
                total += f
    force[vertex] = total


class EnergyFormulaTests(unittest.TestCase):
    def test_normal_integral_and_friction_derivative(self):
        """Check continuity and force derivatives of the scalar contact integrals."""
        r, k = 0.002, 300000.0
        d = np.array([-1e-5, 5e-6, 2e-5, 0.0006, 0.0015, 0.003])
        h = 1e-9
        derivative = (normal_energy(d + h, r, k) - normal_energy(d - h, r, k)) / (2 * h)
        np.testing.assert_allclose(derivative, -normal_load(d, r, k), rtol=1e-5, atol=1e-4)
        for boundary in (1e-5, r / 2, r):
            self.assertLess(abs(float(normal_energy(boundary + h, r, k) - normal_energy(boundary - h, r, k))), 1e-4)
        self.assertEqual(float(normal_energy(r, r, k)), 0.0)
        u, eps = np.array([1e-6, 5e-4, 0.002]), 0.001
        slope = (smooth_friction(u + h, eps) - smooth_friction(u - h, eps)) / (2 * h)
        np.testing.assert_allclose(slope, np.where(u < eps, (2 - u / eps) * u / eps, 1), rtol=1e-5)

    def test_elastic_objective_matches_original_forces(self):
        """Differentiate the measured membrane/bending/damping energy against original kernels."""
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            vertices=[wp.vec3(0, 0, 0), wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(1, 1, 0)],
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0),
            indices=[0, 1, 2, 2, 1, 3],
            density=0.02,
            tri_ke=1500,
            tri_ka=1500,
            tri_kd=1e-5,
            edge_ke=1.2,
            edge_kd=0.1,
        )
        model = builder.finalize(device="cpu")
        observer = EnergyObserver(model)
        anchor = model.particle_q.numpy().astype(np.float64)
        q = anchor + np.array([[0.01, -0.01, 0.02], [0.04, 0.01, 0], [0, 0.02, -0.01], [0.02, 0, 0.12]])
        force = wp.empty(4, dtype=wp.vec3, device="cpu")
        wp.launch(
            elastic_force_reference,
            dim=4,
            inputs=[
                wp.array(q, dtype=wp.vec3, device="cpu"),
                model.particle_q,
                model.tri_indices,
                model.tri_poses,
                model.tri_areas,
                model.tri_materials,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
            ],
            outputs=[force],
            device="cpu",
        )
        fd = np.zeros_like(q)
        h = 1e-4
        for i in range(4):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[i, axis] += h
                minus[i, axis] -= h
                fd[i, axis] = -(
                    sum(observer.elastic(plus, anchor, 0.01).values())
                    - sum(observer.elastic(minus, anchor, 0.01).values())
                ) / (2 * h)
        np.testing.assert_allclose(force.numpy(), fd, rtol=3e-3, atol=3e-3)


def sweep_probe(example, observer):
    """Fork the first initialized substep into default eight and ordinary twenty sweeps."""
    core = example.solver.vbd_solver
    original = core._solve_particle_iteration
    rows = []

    def probe(state_in, state_out, contacts, dt, iteration):
        if iteration != 0:
            raise RuntimeError("Probe must begin before the first sweep")
        attributes = dict(vars(core))
        arrays = [(v, wp.clone(v)) for v in attributes.values() if isinstance(v, wp.array)]
        states = [(state_in, example.model.state()), (state_out, example.model.state())]
        for state, backup in states:
            backup.assign(state)
        energy = ContactObserver(core, state_out, contacts)
        # Iterations update state_in.particle_q, while the prescribed body pose
        # belongs to state_out. Rebuild frozen friction at the actual iterate.
        distance = energy.geometry(state_in.particle_q)
        energy.frozen_ids, energy.frozen_weights = energy.ids.numpy(), energy.weights.numpy().astype(np.float64)
        sep = energy.separation.numpy().astype(np.float64)
        energy.frozen_normals = sep / np.maximum(np.linalg.norm(sep, axis=1, keepdims=True), 1e-30)
        energy.frozen_load = normal_load(distance, energy.radius, energy.ke)
        energy.body_load = energy.body_ke * np.maximum(-energy.body_gap(state_in.particle_q.numpy()), 0)
        target, anchor = core.inertia.numpy().astype(np.float64), core.particle_q_prev.numpy().astype(np.float64)
        for mode, count in (("default", 8), ("ordinary20", 20)):
            vars(core).update(attributes)
            for dst, src in arrays:
                dst.assign(src)
            for state, backup in states:
                state.assign(backup)
            if mode == "ordinary20":
                core.iterations = count
                core.particle_enable_batched_jacobi = False
                core.particle_chebyshev_enabled = False
                core.particle_jacobi_polish_iterations = 0
                core._surface_cached_kernel = None
                core.surface_anchor_angles = None
                core._particle_truncation_cache = None
            for sweep in range(count + 1):
                q = state_in.particle_q.numpy().astype(np.float64)
                value = observer.elastic(q, anchor, dt)
                value["inertia_J"] = float(np.sum(0.5 * observer.mass[:, None] * (q - target) ** 2) / dt**2)
                value.update(energy.potential(state_in.particle_q, frozen=True, dt=dt))
                value["fixed_diagnostic_objective_J"] = sum(value.values())
                value["conservative_inertial_J"] = sum(
                    value[k] for k in ("membrane_J", "bending_J", "inertia_J", "self_normal_J", "body_normal_J")
                )
                current_ke = core.body_particle_contact_penalty_k.numpy()[energy.body_indices]
                value["current_body_normal_J"] = float(
                    np.sum(0.5 * current_ke * np.maximum(-energy.body_gap(q), 0) ** 2)
                )
                value["current_conservative_inertial_J"] = (
                    value["conservative_inertial_J"] - value["body_normal_J"] + value["current_body_normal_J"]
                )
                rows.append(
                    {
                        "mode": mode,
                        "sweep": sweep,
                        **value,
                        "overflow_rows": energy.overflow,
                        "asymmetric_ee_pairs": energy.asymmetric_ee,
                    }
                )
                if sweep < count:
                    if sweep > 0:
                        core._solve_rigid_body_iteration(state_in, state_out, example.control, contacts, dt)
                    original(state_in, state_out, contacts, dt, sweep)
        raise ProbeComplete

    core._solve_particle_iteration = probe
    example.use_graph = False
    try:
        example.step()
    except ProbeComplete:
        pass
    finally:
        core._solve_particle_iteration = original
    return rows


def run(mode, args):
    """Run the unmodified trajectory, changing only the solver policy in this process."""
    options = newton.examples.default_args(scene.Example.create_parser())
    options.graph_capture = False
    constructor = scene.SolverMJVBDV2

    def configure(*positional, **kwargs):
        if mode == "ordinary20":
            kwargs["vbd_preset"] = None
            kwargs["vbd_options"] = dict(
                kwargs["vbd_options"],
                iterations=20,
                particle_jacobi_polish_iterations=0,
                particle_collision_detection_interval=-1,
            )
        return constructor(*positional, **kwargs)

    configure.register_custom_attributes = constructor.register_custom_attributes
    with patch.object(scene, "SolverMJVBDV2", configure):
        example = scene.Example(newton.viewer.ViewerNull(), options)
    observer = EnergyObserver(example.model)
    core = example.solver.vbd_solver
    print(
        json.dumps(
            {
                "mode": mode,
                "iterations": core.iterations,
                "jacobi": core.particle_enable_batched_jacobi,
                "chebyshev": core.particle_chebyshev_enabled,
                "detection_interval": core.particle_collision_detection_interval,
            }
        ),
        flush=True,
    )
    example.use_graph = True
    example.capture()
    rows = []
    for frame in range(args.frames + 1):
        if frame % args.sample_interval == 0 or frame == args.frames:
            values = observer.mechanical(example.state_0)
            if frame:
                contact = ContactObserver(core, example.state_0, example.contacts)
                values.update(contact.potential(example.state_0.particle_q))
                overflow, asymmetric = contact.overflow, contact.asymmetric_ee
            else:
                values.update(self_normal_J=0.0, body_normal_J=0.0)
                overflow = asymmetric = 0
            values["mechanical_plus_contact_J"] = (
                values["cloth_mechanical_J"] + values["self_normal_J"] + values["body_normal_J"]
            )
            rows.append(
                {
                    "frame": frame,
                    "time_s": frame * example.frame_dt,
                    **values,
                    "overflow_rows": overflow,
                    "asymmetric_ee_pairs": asymmetric,
                }
            )
        if frame % 300 == 0:
            print(json.dumps({"mode": mode, "frame": frame, **values}), flush=True)
        if frame < args.frames:
            example.step()
    example.test_final()
    save_csv(args.output / (mode + "_trajectory.csv"), rows)
    if mode == "default":
        save_csv(args.output / "common_substep_sweeps.csv", sweep_probe(example, observer))
    return rows


def plots(output):
    """Render unsmoothed curves, keeping kinetic energy and contact terms visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {
        mode: np.genfromtxt(output / (mode + "_trajectory.csv"), delimiter=",", names=True)
        for mode in ("default", "ordinary20")
    }
    colors = {"default": "#2374ab", "ordinary20": "#d55e00"}
    labels = {"default": "当前默认:7 次批量 + 1 次普通", "ordinary20": "原始有序 GS:20 次普通 sweep"}
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fields = [
        ("cloth_mechanical_J", "布料机械能(重力势能以初始值为零点)"),
        ("kinetic_J", "动能(对数刻度,显示末段残余运动)"),
        ("membrane_J", "膜拉伸 / 面积形变能"),
        ("bending_J", "弯曲能"),
        ("gravity_relative_J", "重力势能变化"),
        ("mechanical_plus_contact_J", "机械能 + 已记录接触的法向势能"),
    ]
    for ax, (key, title) in zip(axes.flat, fields, strict=True):
        for mode, samples in data.items():
            ax.plot(samples["time_s"], samples[key], color=colors[mode], lw=1.2, label=labels[mode])
        ax.axvline(6.55, color="0.5", ls=":", lw=1)
        ax.set(title=title, xlabel="仿真时间 [s]", ylabel="能量 [J]")
        ax.grid(alpha=0.2)
        if key == "kinetic_J":
            ax.set_yscale("symlog", linthresh=1e-10)
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "叠衣服过程的能量走势 | 机器人按规定轨迹运动,系统不封闭\n竖虚线:轨迹结束(6.55 s); 原始采样,无平滑、无末段速度衰减",
        fontsize=14,
    )
    fig.savefig(output / "energy_history.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for mode, samples in data.items():
        tail = samples["time_s"] >= max(samples["time_s"]) - 5
        for ax, key in zip(axes, ("kinetic_J", "cloth_mechanical_J"), strict=True):
            ax.plot(samples["time_s"][tail], samples[key][tail], color=colors[mode], label=labels[mode])
            ax.set(xlabel="仿真时间 [s]", ylabel="能量 [J]")
            ax.grid(alpha=0.2)
    axes[0].set_title("最后 5 秒:动能放大")
    axes[1].set_title("最后 5 秒:布料机械能")
    axes[0].legend()
    fig.savefig(output / "energy_tail.png", dpi=180)
    plt.close(fig)
    with (output / "common_substep_sweeps.csv").open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for mode in data:
        samples = [r for r in records if r["mode"] == mode]
        x = [int(r["sweep"]) for r in samples]
        for ax, key in zip(axes, ("fixed_diagnostic_objective_J", "current_conservative_inertial_J"), strict=True):
            y = np.array([float(r[key]) for r in samples])
            ax.plot(x, y - y[0], ".-", color=colors[mode], label=labels[mode])
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set(xlabel="完成的 sweep 数(0 = 相同初始化状态)", ylabel="相对 sweep 0 的目标变化 [J]")
            ax.grid(alpha=0.2)
    axes[0].set_title("固定诊断目标:包含冻结接触的摩擦 / 阻尼势")
    axes[1].set_title("保守势 + 惯性项(使用当轮接触系数)")
    axes[0].legend()
    fig.suptitle(
        f"第 {int(data['default']['frame'][-1])} 帧后的首个 substep:相同状态对照\n"
        "冻结摩擦目标是诊断代理,不等同于动态接触求解器的严格全局势能",
        fontsize=13,
    )
    fig.savefig(output / "energy_per_sweep.png", dpi=180)
    plt.close(fig)


def main():
    """Write measurements and figures without staging or changing production code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("default", "ordinary20", "plot", "self-test"), required=True)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("newton/tests/outputs/mjvbd_energy"))
    args = parser.parse_args()
    if args.frames < 1 or args.sample_interval < 1:
        parser.error("--frames and --sample-interval must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "self-test":
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(EnergyFormulaTests)
        )
        if not result.wasSuccessful():
            raise SystemExit(1)
    elif args.mode == "plot":
        plots(args.output)
    else:
        run(args.mode, args)


if __name__ == "__main__":
    main()
