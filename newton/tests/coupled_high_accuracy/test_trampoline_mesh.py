"""A missing optional OBJ must not prevent the demo from running or pin interior vertices."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from newton.examples.coupled_high_accuracy.trampoline import mesh


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.8])
def test_generated_trampoline_is_a_complete_oriented_disk(radius):
    vertices, indices = mesh.generate_trampoline_mesh(radius=radius)
    points = np.asarray(vertices)
    triangles = np.asarray(indices).reshape(-1, 3)
    rim = mesh.boundary_vertex_indices(indices)
    assert len(rim) == 64
    np.testing.assert_allclose(points[:, 2], 0)
    np.testing.assert_allclose(np.linalg.norm(points[rim, :2], axis=1), radius)
    assert np.linalg.norm(points[:, :2], axis=1).max() <= radius + 1e-12
    np.testing.assert_array_equal(np.unique(triangles), np.arange(len(points)))
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    assert np.cross(b - a, c - a)[:, 2].min() > 1e-8
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert counts.max() == 2
    assert (counts == 1).sum() == 64
    assert len(points) - len(edges) + len(triangles) == 1
    assert mesh.triangle_mesh_area(vertices, indices) == pytest.approx(32 * radius**2 * np.sin(2 * np.pi / 64))


def test_rim_detection_survives_vertex_reordering():
    vertices, indices = mesh.generate_trampoline_mesh(resolution=8, boundary_segments=32)
    old_rim = mesh.boundary_vertex_indices(indices)
    permutation = np.random.default_rng(7).permutation(len(vertices))
    inverse = np.argsort(permutation)
    reordered = inverse[np.asarray(indices)]
    actual = mesh.boundary_vertex_indices(reordered)
    np.testing.assert_array_equal(actual, np.sort(inverse[old_rim]))
    assert not np.array_equal(actual, np.arange(32))


def test_missing_default_asset_works_outside_the_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(mesh, "DEFAULT_OBJ", tmp_path / "absent.obj")
    monkeypatch.chdir(tmp_path)
    vertices, indices = mesh.load_trampoline_mesh()
    assert len(vertices) > 500
    assert len(mesh.boundary_vertex_indices(indices)) == 64
    assert mesh.triangle_mesh_area(vertices, indices) > 3.1


def test_original_and_explicit_obj_are_honored(monkeypatch, tmp_path):
    original = tmp_path / "original.obj"
    original.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1 -3/2 -2/3 -1/4 # quad\n")
    monkeypatch.setattr(mesh, "DEFAULT_OBJ", original)
    for path in (None, original):
        vertices, indices = mesh.load_trampoline_mesh(path)
        assert len(vertices) == 4 and len(indices) == 6
        assert mesh.triangle_mesh_area(vertices, indices) == pytest.approx(1.0)
    with pytest.raises(FileNotFoundError):
        mesh.load_trampoline_mesh(tmp_path / "explicit-missing.obj")


@pytest.mark.parametrize("index", [0, 4, -4])
def test_obj_rejects_invalid_face_indices(tmp_path, index):
    path = tmp_path / "bad.obj"
    path.write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 {index}\n")
    with pytest.raises(ValueError, match="Invalid vertex index"):
        mesh.load_obj_triangles(path)


def test_rim_detection_rejects_closed_and_nonmanifold_surfaces():
    with pytest.raises(ValueError, match="open boundary"):
        mesh.boundary_vertex_indices([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    with pytest.raises(ValueError, match="nonmanifold"):
        mesh.boundary_vertex_indices([0, 1, 2, 1, 0, 3, 0, 1, 4])


def test_zero_area_trampoline_is_rejected():
    with pytest.raises(ValueError, match="surface area"):
        mesh.triangle_mesh_area([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [0, 1, 2])


@pytest.mark.parametrize("full_surface", [False, True])
def test_small_foot_is_caught_between_cloth_vertices(full_surface):
    wp = pytest.importorskip("warp")
    newton = pytest.importorskip("newton")
    from newton.solvers import coupled_high_accuracy as cha
    from newton.solvers import style3d

    if not wp.is_cuda_available():
        pytest.skip("Coupled trampoline contact requires CUDA")
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.1, 0.075, 1.3)), mass=0.5, inertia=wp.mat33(np.eye(3) * 0.0001)
        )
        cfg = builder.default_shape_cfg.copy()
        cfg.density, cfg.ke, cfg.kd = 0.0, 1e5, 1e3
        builder.add_shape_box(body, hx=0.015, hy=0.015, hz=0.03, cfg=cfg)
        vertices, indices = mesh.generate_trampoline_mesh(resolution=4, boundary_segments=32)
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0, 0, 1),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            vertices=vertices,
            indices=indices,
            density=5 / mesh.triangle_mesh_area(vertices, indices),
            tri_aniso_ke=wp.vec3(2e5),
            edge_aniso_ke=wp.vec3(2e-4),
            particle_radius=0.005,
        )
        rim = mesh.boundary_vertex_indices(indices)
        for vertex in rim:
            builder.particle_flags[vertex] &= ~newton.ParticleFlags.ACTIVE
            builder.particle_mass[vertex] = 0.0
        model = builder.finalize()
        model.soft_contact_ke, model.soft_contact_kd = 1e5, 1e3
        solver = cha.SolverCoupledHighAccuracy(model)
        solver.config.nonlinear_iterations, solver.config.linear_iterations = 5, 50
        solver.config.enable_collision_hessian_non_diagonal = True
        first, second, control = model.state(), model.state(), model.control()
        initial_rim = first.particle_q.numpy()[rim].copy()
        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=full_surface)
        contacts = pipeline.contacts()

        def step_pair():
            for current, following in ((first, second), (second, first)):
                current.clear_forces()
                pipeline.collide(current, contacts)
                solver.step(current, following, control, contacts, 0.00125)

        step_pair()
        with wp.ScopedCapture() as captured:
            step_pair()
        upward_speed, surface_contacts = 0.0, 0
        for _ in range(200):
            wp.capture_launch(captured.graph)
            upward_speed = max(upward_speed, float(first.body_qd.numpy()[body, 2]))
            count = int(contacts.soft_contact_count.numpy()[0])
            surface_contacts = max(surface_contacts, int((contacts.soft_contact_particle.numpy()[:count] == -1).sum()))
        z = float(first.body_q.numpy()[body, 2])
        np.testing.assert_allclose(first.particle_q.numpy()[rim], initial_rim, atol=1e-7)
        if full_surface:
            assert surface_contacts > 0
            assert upward_speed > 0.05
            assert z > 0.95
        else:
            assert z < 0.7
