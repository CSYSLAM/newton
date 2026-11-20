"""Regressions for multiple developed film sheets sharing a welded edge."""

import numpy as np
import pytest

from newton.examples.coupled_high_accuracy.demos.piper_bag_transport.bag_bending import rebuild_bag_bending
from newton.solvers import coupled_high_accuracy as cha

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")


def _welded_sheets(*, separate_plies: bool = False):
    from newton.solvers import style3d

    points = np.array(((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)), dtype=float)
    triangles = np.array(((0, 2, 3), (1, 3, 2), (4, 2, 3), (5, 3, 2)), dtype=np.int32)
    panel = np.array(((-1, 0), (1, 0), (0, -1), (0, 1)) * 2, dtype=float)
    panel_triangles = np.array(((0, 2, 3), (1, 3, 2), (4, 6, 7), (5, 7, 6)), dtype=np.int32)
    if separate_plies:
        panel = panel[panel_triangles].reshape(-1, 2)
        panel_triangles = np.arange(12, dtype=np.int32).reshape(-1, 3)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    with pytest.warns(UserWarning, match="non-manifold"):
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=points,
            indices=triangles.ravel(),
            panel_verts=panel,
            panel_indices=panel_triangles.ravel(),
            density=1.0,
            tri_aniso_ke=wp.vec3(0.0),
            edge_aniso_ke=wp.vec3(0.3),
            edge_kd=0.0,
        )
    metadata = rebuild_bag_bending(
        builder,
        triangles,
        panel,
        panel_triangles,
        particle_start=0,
        edge_start=0,
        edge_aniso_ke=wp.vec3(0.3),
        edge_kd=0.0,
    )
    return builder, metadata


@pytest.mark.parametrize("separate_plies", (False, True))
def test_welded_seam_preserves_every_developed_sheet_face(separate_plies: bool) -> None:
    builder, metadata = _welded_sheets(separate_plies=separate_plies)
    assert metadata["physical_edge_count"] == 9
    assert metadata["nonmanifold_edge_count"] == 1
    assert metadata["maximum_incident_faces"] == 4
    assert metadata["represented_face_sides"] == 12
    assert metadata["nonmanifold_sheet_hinge_count"] == (0 if separate_plies else 2)
    assert metadata["bending_hinge_count"] == (0 if separate_plies else 2)
    model = builder.finalize(device="cpu")
    assert model.edge_count == (12 if separate_plies else 10)
    assert np.isfinite(model.style3d.edge_bending_cot.numpy()).all()
    assert np.count_nonzero(model.style3d.edge_rest_area.numpy()) == (0 if separate_plies else 2)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_collision_edges_are_unique_while_sheet_hinges_remain_distinct() -> None:
    builder, _ = _welded_sheets()
    model = builder.finalize(device="cuda:0")
    solver = cha.SolverCoupledHighAccuracy(model)
    assert solver.synreal_model.edge_count == 10
    assert solver.synreal_model.edge_indices.ptr == model.edge_indices.ptr
    assert solver.synreal_model.collision_edge_count == 9
    edges = solver._collision_edge_indices.numpy()
    assert len(np.unique(np.sort(edges[:, 2:4], axis=1), axis=0)) == 9


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Native solver requires CUDA")
@pytest.mark.parametrize("nonlinear_solver", (cha.NonlinearSolver.PD_NEWTON, cha.NonlinearSolver.NEWTON))
@pytest.mark.parametrize("collapsed_row", (False, True))
def test_native_nonmanifold_bending_matches_sum_of_sheet_energies(nonlinear_solver, collapsed_row: bool) -> None:
    # Two sheets have the same endpoints but independent opposite vertices. The
    # exact implicit quadratic solve detects dropped or double-counted hinges.
    points = np.array(((-1, 0, 0.2), (1, 0, 0.1), (0, -1, 0), (0, 1, 0), (0, 0.3, -1), (0, 0.1, 1)), dtype=np.float32)
    edges = [(0, 1, 2, 3), (4, 5, 2, 3)]
    if collapsed_row:
        edges.append((0, 0, 2, 3))
    arrays = {
        "particle_masses": wp.ones(6, dtype=wp.float32, device="cuda:0"),
        "edge_indices": wp.array(edges, dtype=wp.vec4i, device="cuda:0"),
        "edge_rest_areas": wp.full(len(edges), 2.0, dtype=wp.float32, device="cuda:0"),
        "edge_bending_cots": wp.full(len(edges), wp.vec4(0.5), dtype=wp.vec4, device="cuda:0"),
        "edge_bending_properties": wp.full(len(edges), wp.vec2(5.0, 0.0), dtype=wp.vec2, device="cuda:0"),
    }
    model = cha.Model()
    model.particle_count = len(points)
    model.edge_count = len(edges)
    for name, array in arrays.items():
        setattr(model, name, cha.Span(ptr=array.ptr, size=len(array)))
    position_in = wp.array(points, dtype=wp.vec3, device="cuda:0")
    position_out = wp.zeros_like(position_in)
    velocity_in, velocity_out = wp.zeros_like(position_in), wp.zeros_like(position_in)
    state_in, state_out = cha.State(), cha.State()
    for state, position, velocity in ((state_in, position_in, velocity_in), (state_out, position_out, velocity_out)):
        state.particle_q = cha.Span(ptr=position.ptr, size=len(position))
        state.particle_qd = cha.Span(ptr=velocity.ptr, size=len(velocity))
    solver = cha.Solver()
    solver.initialize(model)
    solver.config.nonlinear_solver = nonlinear_solver
    solver.config.nonlinear_iterations = 4
    solver.config.linear_iterations = 80
    solver.config.enable_self_collision = False
    dt = 0.04
    solver.step(state_in, state_out, dt)
    stiffness = np.zeros((6, 6))
    for edge in edges[:2]:
        weight = np.zeros(6)
        weight[list(edge)] = (-1.0, -1.0, 1.0, 1.0)
        stiffness += 7.5 * np.outer(weight, weight)
    expected = np.linalg.solve(np.eye(6) + dt * dt * stiffness, points)
    np.testing.assert_allclose(position_out.numpy(), expected, atol=3.0e-6, rtol=1.0e-5)
    np.testing.assert_allclose(position_out.numpy().sum(axis=0), points.sum(axis=0), atol=1.0e-6)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Native solver requires CUDA")
def test_nonmanifold_edge_face_recovery_uses_every_incident_ply() -> None:
    points = np.array(
        (
            (0, 0, -0.03),
            (0, 0, 0.03),
            (0.1, 0, 0),
            (0.1, 0.1, 0),
            (0.1, -0.1, 0),
            (0.2, 0.01, 0),
            (-1, -1, 0),
            (1, -1, 0),
            (0, 1, 0),
        ),
        dtype=np.float32,
    )
    faces = [(0, 1, opposite) for opposite in range(2, 6)] + [(6, 7, 8)]
    results = []
    # The edge record can retain any two plies after collision deduplication. The
    # untangling direction must depend on all four actual incident triangles.
    for opposites in ((-1, -1), (2, -1), (2, 3), (4, 5)):
        arrays = {
            "particle_masses": wp.ones(len(points), dtype=wp.float32, device="cuda:0"),
            "particle_flags": wp.array([1, 1] + [0] * 7, dtype=wp.int32, device="cuda:0"),
            "particle_rest_positions": wp.array(points, dtype=wp.vec3, device="cuda:0"),
            "collision_edge_indices": wp.array([(*opposites, 0, 1)], dtype=wp.vec4i, device="cuda:0"),
            "collision_triangle_indices": wp.array(faces, dtype=wp.vec3i, device="cuda:0"),
        }
        model = cha.Model()
        model.particle_count = len(points)
        model.collision_edge_count = 1
        model.collision_triangle_count = len(faces)
        for name, array in arrays.items():
            setattr(model, name, cha.Span(ptr=array.ptr, size=len(array)))
        position_in = arrays["particle_rest_positions"]
        position_out = wp.zeros_like(position_in)
        velocity_in, velocity_out = wp.zeros_like(position_in), wp.zeros_like(position_in)
        state_in, state_out = cha.State(), cha.State()
        for state, position, velocity in (
            (state_in, position_in, velocity_in),
            (state_out, position_out, velocity_out),
        ):
            state.particle_q = cha.Span(ptr=position.ptr, size=len(position))
            state.particle_qd = cha.Span(ptr=velocity.ptr, size=len(velocity))
        solver = cha.Solver()
        solver.initialize(model)
        solver.config.nonlinear_iterations = 4
        solver.config.linear_iterations = 80
        solver.config.enable_self_collision = True
        solver.config.self_collision_radius = 0.01
        solver.config.self_collision_vertex_face_stiffness = 0.0
        solver.config.self_collision_edge_edge_stiffness = 0.0
        solver.config.self_collision_edge_face_stiffness = 5.0
        solver.step(state_in, state_out, 0.01)
        results.append(position_out.numpy())
    assert np.linalg.norm(results[0][:2] - points[:2]) > 1.0e-5
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=1.0e-7, rtol=0.0)


def test_food_bag_has_consistent_winding_and_flat_sheet_metric() -> None:
    from collections import defaultdict

    from newton.examples.coupled_high_accuracy.demos.demo.demo_piper_bag_transport import create_bag_mesh

    points, triangles, panel, panel_triangles = create_bag_mesh()
    developed = panel[panel_triangles]
    a, b = developed[:, 1] - developed[:, 0], developed[:, 2] - developed[:, 0]
    assert np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > 0.0)
    assert np.all(
        np.linalg.norm(
            np.cross(
                points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]
            ),
            axis=1,
        )
        > 1.0e-12
    )
    adjacency = defaultdict(list)
    for face, triangle in enumerate(triangles):
        for corner in range(3):
            first, second = int(triangle[corner]), int(triangle[(corner + 1) % 3])
            length = np.linalg.norm(developed[face, corner] - developed[face, (corner + 1) % 3])
            adjacency[tuple(sorted((first, second)))].append((first < second, length))
    assert any(len(adjacent) == 4 for adjacent in adjacency.values())
    for adjacent in adjacency.values():
        assert len(adjacent) in (1, 2, 4)
        if len(adjacent) == 2:
            assert adjacent[0][0] != adjacent[1][0]
        np.testing.assert_allclose([entry[1] for entry in adjacent], adjacent[0][1], atol=1.0e-6)
