"""Shared particle surface API covers fluid meshes and material coordinate transport."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ParticleSurface, log_surface_mesh
from newton.examples.coupled_high_accuracy.example_cha_fluid import Example, particle_layout

requires_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="Particle surface reconstruction requires CUDA")


@requires_cuda
@pytest.mark.parametrize("resolution", [8, 16, 32])
def test_fluid_surface_is_closed_and_has_outward_winding(resolution):
    points, spacing = particle_layout(resolution)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    mesh = ParticleSurface(spacing, positions, voxel_size=0.5 * spacing).rebuild(positions)
    vertices, faces = mesh.vertices.numpy(), mesh.indices.numpy().reshape(-1, 3)
    assert len(vertices) and len(faces)
    assert mesh.rest_positions is None
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    triangles = vertices[faces].astype(np.float64) - vertices.mean(axis=0)
    volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    assert volume > 0


@requires_cuda
def test_reference_coordinates_follow_separate_material_pieces():
    points = np.array([[-0.005, 0, 0], [0.005, 0, 0]], dtype=np.float32)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    bonds = wp.array([[0, 1]], dtype=wp.vec2i, device="cuda:0")
    broken = wp.ones(1, dtype=int, device="cuda:0")
    surface = ParticleSurface(0.01, positions, bonds=bonds, broken=broken, interpolate_rest_positions=True)
    moved = points + np.array([0.2, 0.3, 0.4], dtype=np.float32)
    positions.assign(moved)
    mesh = surface.rebuild(positions)
    vertices, rest = mesh.vertices.numpy(), mesh.rest_positions.numpy()
    left = vertices[:, 0] < moved[:, 0].mean()
    assert np.any(left) and np.any(~left)
    # Nearby foreign particles must not contaminate either piece's reference coordinates.
    np.testing.assert_allclose(rest[left], np.broadcast_to(points[0], rest[left].shape), atol=1.0e-7)
    np.testing.assert_allclose(rest[~left], np.broadcast_to(points[1], rest[~left].shape), atol=1.0e-7)


@requires_cuda
def test_empty_selection_returns_empty_mesh():
    positions = wp.empty(0, dtype=wp.vec3, device="cuda:0")
    mesh = ParticleSurface(0.01, positions, interpolate_rest_positions=True).rebuild(positions)
    assert len(mesh.vertices) == len(mesh.indices) == len(mesh.rest_positions) == 0


@requires_cuda
def test_rebuild_rejects_changed_selection_and_nonfinite_positions():
    positions = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(0.01, positions)
    with pytest.raises(ValueError, match="particle selection"):
        surface.rebuild(positions[:1])
    positions.assign([[np.nan, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="non-finite"):
        surface.rebuild(positions)


@requires_cuda
def test_untextured_usd_export_updates_topology(tmp_path):
    usd = pytest.importorskip("pxr.Usd")
    usd_geom = pytest.importorskip("pxr.UsdGeom")
    from newton.viewer import ViewerUSD

    points, spacing = particle_layout(8)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(spacing, positions)
    output = tmp_path / "fluid_surface.usda"
    viewer = ViewerUSD(str(output), fps=60, num_frames=2)
    expected = []
    try:
        for frame in (1, 2):
            if frame == 2:
                moved = points.copy()
                moved[points[:, 0] < 0.56, 0] -= 3.0 * spacing
                positions.assign(moved)
            mesh = surface.rebuild(positions)
            expected.append((mesh.vertices.numpy().copy(), mesh.indices.numpy().copy()))
            viewer.begin_frame(frame / 60.0)
            log_surface_mesh(viewer, "fluid_surface", mesh, color=(0.025, 0.20, 0.65), roughness=0.18)
            viewer.end_frame()
    finally:
        viewer.close()
    assert len(expected[0][1]) != len(expected[1][1])
    stage = usd.Stage.Open(str(output))
    mesh = usd_geom.Mesh(stage.GetPrimAtPath("/root/fluid_surface"))
    for frame, (vertices, indices) in enumerate(expected, start=1):
        np.testing.assert_array_equal(mesh.GetPointsAttr().Get(frame), vertices)
        np.testing.assert_array_equal(mesh.GetFaceVertexIndicesAttr().Get(frame), indices)
        assert len(mesh.GetFaceVertexCountsAttr().Get(frame)) == len(indices) // 3


def test_fluid_surface_flag_is_opt_in():
    parser = Example.create_parser()
    assert not parser.parse_args([]).build_surface
    assert parser.parse_args(["--build-surface"]).build_surface
