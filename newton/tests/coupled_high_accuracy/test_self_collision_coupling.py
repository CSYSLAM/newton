"""Self-contact tangents must couple both sides of each deformable contact."""

from dataclasses import dataclass

import numpy as np
import pytest

from newton.solvers import coupled_high_accuracy as s3d

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Self-contact coupling requires CUDA")


@dataclass
class ContactCase:
    positions: np.ndarray
    masses: np.ndarray
    weights: np.ndarray
    direction: np.ndarray
    correction: float
    stiffness: float
    triangles: tuple = ()
    edges: tuple = ()
    particles: tuple = ()


DT = 0.01
FACTOR = 20.0


def _case(kind: str) -> ContactCase:
    if kind == "particle_pair":
        positions = ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))
        masses = (0.5, 1.5)
        return ContactCase(
            np.asarray(positions),
            np.asarray(masses),
            np.asarray((-1.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            0.1,
            FACTOR * 0.5 * 1.5 / 2.0 / DT**2,
            particles=(0, 1),
        )
    if kind == "edge_edge":
        positions = ((-1.0, 0.0, 0.02), (1.0, 0.0, 0.02), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        return ContactCase(
            np.asarray(positions),
            np.asarray((1.0, 1.0, 2.0, 2.0)),
            np.asarray((0.5, 0.5, -0.5, -0.5)),
            np.asarray((0.0, 0.0, 1.0)),
            0.02,
            FACTOR * 2.0 / 3.0 / DT**2,
            edges=((-1, -1, 0, 1), (-1, -1, 2, 3)),
        )
    face = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    if kind == "edge_face":
        positions = (*face, (1.0 / 3.0, 1.0 / 3.0, -0.1), (1.0 / 3.0, 1.0 / 3.0, 0.1), (4.0 / 3.0, 1.0 / 3.0, 0.0))
        return ContactCase(
            np.asarray(positions),
            np.asarray((1.0, 1.0, 1.0, 2.0, 2.0, 1.0)),
            np.asarray((-1.0 / 3.0,) * 3 + (0.5, 0.5, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            0.08,
            FACTOR * 2.0 / 3.0 / DT**2,
            triangles=((0, 1, 2),),
            edges=((5, -1, 3, 4),),
        )
    positions = (*face, (1.0 / 3.0, 1.0 / 3.0, 0.03))
    masses = (1.0, 1.0, 1.0, 2.0)
    weights = (-1.0 / 3.0,) * 3 + (1.0,)
    if kind == "surface_vertex_face":
        # The edge registers a surface vertex without introducing a second nearby triangle.
        positions += ((3.0, 3.0, 1.0),)
        masses += (1.0,)
        weights += (0.0,)
    return ContactCase(
        np.asarray(positions),
        np.asarray(masses),
        np.asarray(weights),
        np.asarray((0.0, 0.0, 1.0)),
        0.01 if kind == "surface_vertex_face" else 0.09,
        FACTOR * 2.0 / 3.0 / DT**2,
        triangles=((0, 1, 2),),
        edges=((-1, -1, 3, 4),) if kind == "surface_vertex_face" else (),
        particles=(3,) if kind == "particle_vertex_face" else (),
    )


def _expected(case: ContactCase, full_hessian: bool) -> np.ndarray:
    inertia = case.masses / DT**2
    load = case.stiffness * case.correction * case.weights
    if full_hessian:
        # Sherman-Morrison solves the relative contact mode while preserving the translation nullspace.
        displacement = load / inertia / (1.0 + case.stiffness * np.sum(case.weights**2 / inertia))
    else:
        displacement = load / (inertia + case.stiffness * case.weights**2)
    return case.positions + displacement[:, None] * case.direction


def _span(array: wp.array) -> s3d.Span:
    return s3d.Span(ptr=array.ptr, size=len(array))


class ContactScene:
    def __init__(
        self,
        kind: str,
        full_hessian: bool,
        method: s3d.NonlinearSolver,
        world_count: int = 1,
        rotated: bool = False,
        fixed_indices: tuple[int, ...] = (),
        fixed_by_flags: bool = False,
    ) -> None:
        self.case = _case(kind)
        if rotated:
            rotation = np.asarray(((0.36, -0.8, 0.48), (0.48, 0.6, 0.64), (-0.8, 0.0, 0.6)))
            self.case.positions = self.case.positions @ rotation.T
            self.case.direction = rotation @ self.case.direction
        self.initial = np.tile(self.case.positions, (world_count, 1)).astype(np.float32)
        self.arrays = []

        def array(values, dtype):
            result = wp.array(values, dtype=dtype, device="cuda:0")
            self.arrays.append(result)
            return result

        self.position_in = array(self.initial, wp.vec3)
        self.velocity_in = array(np.zeros_like(self.initial), wp.vec3)
        self.force_in = array(np.zeros_like(self.initial), wp.vec3)
        self.position_out = array(np.zeros_like(self.initial), wp.vec3)
        self.velocity_out = array(np.zeros_like(self.initial), wp.vec3)
        self.model = s3d.Model()
        self.model.world_count = world_count
        self.model.particle_count = len(self.case.masses)
        self.model.particle_state_count = len(self.initial)
        masses = self.case.masses.copy()
        if fixed_by_flags:
            flags = np.ones(len(masses), dtype=np.int32)
            flags[list(fixed_indices)] = 0
            self.model.particle_flags = _span(array(flags, wp.int32))
        else:
            masses[list(fixed_indices)] = 0.0
        self.model.particle_masses = _span(array(masses, wp.float32))
        self.model.particle_radii = _span(array(np.full(len(self.case.masses), 0.1), wp.float32))
        self.model.particle_rest_positions = _span(self.position_in)
        for name, values, dtype in (
            ("triangle", self.case.triangles, wp.vec3i),
            ("edge", self.case.edges, wp.vec4i),
            ("particle", self.case.particles, wp.int32),
        ):
            if values:
                setattr(self.model, f"collision_{name}_count", len(values))
                setattr(self.model, f"collision_{name}_indices", _span(array(values, dtype)))
        self.state_in, self.state_out = s3d.State(), s3d.State()
        self.state_in.particle_q = _span(self.position_in)
        self.state_in.particle_qd = _span(self.velocity_in)
        self.state_in.particle_f = _span(self.force_in)
        self.state_out.particle_q = _span(self.position_out)
        self.state_out.particle_qd = _span(self.velocity_out)
        self.solver = s3d.Solver()
        self.solver.initialize(self.model)
        self.solver.config.nonlinear_solver = method
        self.solver.config.nonlinear_iterations = 1
        self.solver.config.linear_iterations = 50
        self.solver.config.enable_self_collision = True
        self.solver.config.enable_collision_hessian_non_diagonal = full_hessian
        self.solver.config.self_collision_radius = 0.02
        self.solver.config.self_collision_vertex_face_stiffness = FACTOR if "vertex_face" in kind else 0.0
        self.solver.config.self_collision_particle_particle_stiffness = FACTOR if kind == "particle_pair" else 0.0
        self.solver.config.self_collision_edge_edge_stiffness = FACTOR if kind == "edge_edge" else 0.0
        self.solver.config.self_collision_edge_face_stiffness = FACTOR if kind == "edge_face" else 0.0

    def step(self) -> None:
        self.solver.step(self.state_in, self.state_out, DT, wp.get_stream("cuda:0"))


CONTACT_KINDS = ("particle_pair", "particle_vertex_face", "surface_vertex_face", "edge_edge", "edge_face")


@pytest.mark.parametrize("kind", CONTACT_KINDS)
@pytest.mark.parametrize("full_hessian", (False, True))
@pytest.mark.parametrize("method", (s3d.NonlinearSolver.PD_NEWTON, s3d.NonlinearSolver.NEWTON))
def test_self_contact_step_matches_coupled_normal_system(
    kind: str, full_hessian: bool, method: s3d.NonlinearSolver
) -> None:
    scene = ContactScene(kind, full_hessian, method)
    scene.step()
    result = scene.position_out.numpy()
    np.testing.assert_allclose(result, _expected(scene.case, full_hessian), atol=2.0e-6, rtol=0.0)
    if full_hessian:
        # Internal collision forces cannot accelerate the system's center of mass, even at one Newton iteration.
        momentum = np.sum(scene.case.masses[:, None] * scene.velocity_out.numpy(), axis=0)
        np.testing.assert_allclose(momentum, np.zeros(3), atol=2.0e-4, rtol=0.0)


@pytest.mark.parametrize("kind", CONTACT_KINDS[:-1])
@pytest.mark.parametrize("iterations", (1, 4))
def test_moving_contact_converges_without_oscillating_between_newton_iterations(kind: str, iterations: int) -> None:
    scene = ContactScene(kind, True, s3d.NonlinearSolver.NEWTON)
    velocities = 0.3 * (scene.case.weights / scene.case.masses)[:, None] * scene.case.direction
    scene.velocity_in.assign(velocities.astype(np.float32))
    scene.solver.config.nonlinear_iterations = iterations
    scene.step()

    # Each side translates along the normal, keeping the face/edge interpolation weights fixed.
    scene.case.positions = scene.case.positions + DT * velocities
    scene.case.correction -= DT * np.dot(scene.case.weights, velocities @ scene.case.direction)
    expected = _expected(scene.case, True)
    np.testing.assert_allclose(scene.position_out.numpy(), expected, atol=2.0e-6, rtol=0.0)
    initial_momentum = np.sum(scene.case.masses[:, None] * velocities, axis=0)
    final_momentum = np.sum(scene.case.masses[:, None] * scene.velocity_out.numpy(), axis=0)
    np.testing.assert_allclose(final_momentum, initial_momentum, atol=2.0e-4, rtol=0.0)


@pytest.mark.parametrize(
    "kind,fixed_indices,fixed_by_flags",
    (
        ("particle_pair", (0,), False),
        ("particle_pair", (0, 1), True),
        ("particle_vertex_face", (0, 1, 2), False),
        ("surface_vertex_face", (3,), True),
        ("surface_vertex_face", (0,), True),
        ("edge_edge", (0, 2), False),
        ("edge_face", (0, 3), True),
    ),
)
def test_self_contact_coupling_preserves_fixed_particles(
    kind: str,
    fixed_indices: tuple[int, ...],
    fixed_by_flags: bool,
) -> None:
    scene = ContactScene(
        kind,
        True,
        s3d.NonlinearSolver.NEWTON,
        fixed_indices=fixed_indices,
        fixed_by_flags=fixed_by_flags,
    )
    scene.step()
    effective_mass = scene.case.masses.copy()
    effective_mass[list(fixed_indices)] = 0.0
    first_mass = np.mean(effective_mass[scene.case.weights > 0.0])
    second_mass = np.mean(effective_mass[scene.case.weights < 0.0])
    contact_mass = first_mass + second_mass
    if first_mass > 0.0 and second_mass > 0.0:
        contact_mass = first_mass * second_mass / contact_mass
    scene.case.stiffness = FACTOR * contact_mass / DT**2
    # Eliminating fixed degrees of freedom leaves the same rank-one contact on the free particles.
    scene.case.weights[list(fixed_indices)] = 0.0
    result = scene.position_out.numpy()
    np.testing.assert_allclose(result, _expected(scene.case, True), atol=2.0e-6, rtol=0.0)
    np.testing.assert_array_equal(result[list(fixed_indices)], scene.initial[list(fixed_indices)])


@pytest.mark.parametrize("kind", CONTACT_KINDS)
def test_self_contact_cuda_capture_keeps_replicated_worlds_independent(kind: str) -> None:
    scene = ContactScene(kind, True, s3d.NonlinearSolver.NEWTON, world_count=3, rotated=True)
    with wp.ScopedCapture(device="cuda:0") as capture:
        scene.step()
    wp.capture_launch(capture.graph)
    # Coincident worlds intentionally overlap: each must receive exactly its own contact response.
    expected = np.tile(_expected(scene.case, True), (3, 1))
    np.testing.assert_allclose(scene.position_out.numpy(), expected, atol=2.0e-6, rtol=0.0)

    # Replaying a graph with separated contacts must clear all previously cached contact tangents.
    separated = scene.initial.copy().reshape(3, -1, 3)
    separated += 3.0 * (scene.case.weights > 0.0)[None, :, None] * scene.case.direction
    separated = separated.reshape(-1, 3)
    scene.position_in.assign(separated)
    scene.velocity_in.zero_()
    force = np.tile(100.0 * scene.case.weights[:, None] * scene.case.direction, (3, 1))
    scene.force_in.assign(force.astype(np.float32))
    wp.capture_launch(capture.graph)
    expected = separated + DT**2 * force / np.tile(scene.case.masses, 3)[:, None]
    np.testing.assert_allclose(scene.position_out.numpy(), expected, atol=2.0e-6, rtol=0.0)
