"""Beef surface reconstruction stays closed while particles move and separate."""

import numpy as np
import pytest

pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ParticleSurface
from newton.examples.coupled_high_accuracy.example_cha_granular_beaf import (
    BeefSurfaceMaterial,
    Example,
    particle_layout,
)

requires_cuda = pytest.mark.skipif(not wp.is_cuda_available(), reason="Beef surface reconstruction requires CUDA")


def assert_closed_mesh(vertices, indices):
    points = vertices.numpy().astype(np.float64)
    faces = indices.numpy().reshape(-1, 3)
    assert len(points) > 0 and len(faces) > 0
    assert np.isfinite(points).all()
    assert faces.min() >= 0 and faces.max() < len(points)
    assert np.all(faces[:, 0] != faces[:, 1])
    assert np.all(faces[:, 1] != faces[:, 2])
    assert np.all(faces[:, 2] != faces[:, 0])

    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert np.all(edge_counts == 2), "The reconstructed surface must have no boundary or nonmanifold edges"

    triangles = points[faces]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.linalg.norm(area_vectors, axis=1) > 0.0), "The mesh must have no zero-area faces"

    # Centering preserves volume but avoids cancellation after a large translation.
    centered = triangles - points.mean(axis=0)
    volume = np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6.0
    assert volume > 0.0, "Triangle winding must face outward"
    return points, faces


def lattice_axis_bonds(points, spacing):
    cells = np.rint((points - points.min(axis=0)) / spacing).astype(int)
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    bonds = []
    for i, cell in enumerate(cells):
        for axis in range(3):
            neighbor = cell.copy()
            neighbor[axis] += 1
            j = lookup.get(tuple(neighbor))
            if j is not None:
                bonds.append((i, j))
    return np.asarray(bonds, dtype=np.int32).reshape(-1, 2)


def mesh_component_labels(points, faces):
    parent = np.arange(len(points))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Derive connectivity only from output triangles, independently of particle bond labeling.
    for a, b, c in faces:
        parent[root(a)] = root(b)
        parent[root(b)] = root(c)
    return np.array([root(i) for i in range(len(points))])


@requires_cuda
@pytest.mark.parametrize("resolution", [12, 32, 64])
def test_initial_beef_surface_is_closed_and_outward(resolution):
    points, spacing = particle_layout(resolution)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(spacing, positions, interpolate_rest_positions=True)
    mesh = surface.rebuild(positions)
    vertices, indices, _ = mesh
    uvs = BeefSurfaceMaterial().build_uvs(mesh)
    mesh_points, _ = assert_closed_mesh(vertices, indices)
    assert len(uvs) == len(mesh_points)
    assert np.isfinite(uvs.numpy()).all()
    assert np.all(mesh_points.min(axis=0) < points.min(axis=0))
    assert np.all(mesh_points.max(axis=0) > points.max(axis=0))


@requires_cuda
def test_beef_surface_follows_particles_outside_original_domain():
    points, spacing = particle_layout(32)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(spacing, positions)
    first_vertices, first_indices, _ = surface.rebuild(positions)
    first_points, _ = assert_closed_mesh(first_vertices, first_indices)

    offset = np.array([2.1, -1.8, 0.9], dtype=np.float32)
    positions.assign(points + offset)
    moved_vertices, moved_indices, _ = surface.rebuild(positions)
    moved_points, _ = assert_closed_mesh(moved_vertices, moved_indices)
    assert np.all(moved_points.min(axis=0) < (points + offset).min(axis=0))
    assert np.all(moved_points.max(axis=0) > (points + offset).max(axis=0))
    np.testing.assert_allclose(moved_points.min(axis=0) - first_points.min(axis=0), offset, atol=0.6 * spacing)
    np.testing.assert_allclose(moved_points.max(axis=0) - first_points.max(axis=0), offset, atol=0.6 * spacing)


@requires_cuda
@pytest.mark.parametrize("offset", [1.0, 1.25, 3.0])
def test_separated_beef_halves_have_closed_cut_faces_without_bridging(offset):
    points, spacing = particle_layout(32)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(spacing, positions)
    separated = points.copy()
    separated[:, 0] += np.where(points[:, 0] < 0.0, -offset, offset).astype(np.float32) * spacing
    positions.assign(separated)
    vertices, indices, _ = surface.rebuild(positions)
    mesh_points, faces = assert_closed_mesh(vertices, indices)
    x = mesh_points[faces, 0]
    left, right = np.all(x < 0.0, axis=1), np.all(x > 0.0, axis=1)
    assert np.any(left) and np.any(right)
    assert np.all(left | right), "The rebuilt mesh must not seal the gap between separated pieces"


@requires_cuda
def test_isolated_beef_grain_retains_a_closed_surface():
    center = np.array([[0.08, -0.02, 0.03]], dtype=np.float32)
    positions = wp.array(center, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(0.01, positions)
    vertices, indices, _ = surface.rebuild(positions)
    mesh_points, _ = assert_closed_mesh(vertices, indices)
    assert np.all(mesh_points.min(axis=0) < center[0])
    assert np.all(mesh_points.max(axis=0) > center[0])
    distances = np.linalg.norm(mesh_points - center[0], axis=1)
    np.testing.assert_allclose(distances, surface.radius, atol=surface.voxel_size)


@requires_cuda
def test_broken_beef_cut_stays_closed_and_separate_when_halves_touch_again():
    points, spacing = particle_layout(32)
    pairs = lattice_axis_bonds(points, spacing)
    cross = points[pairs[:, 0], 0] * points[pairs[:, 1], 0] < 0.0
    assert np.any(cross)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    bonds = wp.array(pairs, dtype=wp.vec2i, device="cuda:0")
    broken = wp.zeros(len(pairs), dtype=int, device="cuda:0")
    surface = ParticleSurface(spacing, positions, bonds=bonds, broken=broken)

    vertices, indices, _ = surface.rebuild(positions)
    initial_points, initial_faces = assert_closed_mesh(vertices, indices)
    assert len(np.unique(mesh_component_labels(initial_points, initial_faces))) == 1

    broken.assign(cross.astype(np.int32))
    # Fracture must open render topology immediately, even before physical grains move apart.
    for offset in (0.0, 1.0, 0.0):
        moved = points.copy()
        moved[:, 0] += np.where(points[:, 0] < 0.0, -offset, offset).astype(np.float32) * spacing
        positions.assign(moved)
        vertices, indices, _ = surface.rebuild(positions)
        mesh_points, faces = assert_closed_mesh(vertices, indices)
        labels = mesh_component_labels(mesh_points, faces)
        components = np.unique(labels)
        assert len(components) == 2, "Disconnected material must keep separate render topology after recontact"

        x = mesh_points[faces, 0]
        assert np.all(np.all(x < 0.0, axis=1) | np.all(x > 0.0, axis=1))
        left, right = mesh_points[:, 0] < 0.0, mesh_points[:, 0] > 0.0
        assert np.any(left) and np.any(right) and np.all(left | right)
        assert mesh_points[right, 0].min() > mesh_points[left, 0].max()
        for component in components:
            selected_faces = faces[labels[faces[:, 0]] == component]
            triangles = mesh_points[selected_faces]
            centered = triangles - mesh_points[labels == component].mean(axis=0)
            volume = np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6.0
            assert volume > 0.0, "Each reconstructed cut piece must have outward winding"


@requires_cuda
def test_oblique_broken_beef_cut_remains_separate_after_recontact():
    points, spacing = particle_layout(32)
    pairs = lattice_axis_bonds(points, spacing)
    cross = points[pairs[:, 0], 0] * points[pairs[:, 1], 0] < 0.0
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    bonds = wp.array(pairs, dtype=wp.vec2i, device="cuda:0")
    broken = wp.array(cross.astype(np.int32), dtype=int, device="cuda:0")
    surface = ParticleSurface(spacing, positions, bonds=bonds, broken=broken)

    azimuth, elevation = np.deg2rad([31.0, 23.0])
    ca, sa, ce, se = np.cos(azimuth), np.sin(azimuth), np.cos(elevation), np.sin(elevation)
    rotation_z = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    rotation_y = np.array([[ce, 0.0, se], [0.0, 1.0, 0.0], [-se, 0.0, ce]])
    rotation = rotation_z @ rotation_y
    for offset in (1.0, 0.0):
        moved = points.copy()
        moved[:, 0] += np.where(points[:, 0] < 0.0, -offset, offset).astype(np.float32) * spacing
        positions.assign((moved @ rotation.T).astype(np.float32))
        vertices, indices, _ = surface.rebuild(positions)
        mesh_points, faces = assert_closed_mesh(vertices, indices)
        labels = mesh_component_labels(mesh_points, faces)
        components = np.unique(labels)
        assert len(components) == 2

        # Compare against the cut in material coordinates, independently of the extraction grid orientation.
        projected_x = (mesh_points @ rotation)[:, 0]
        face_x = projected_x[faces]
        assert np.all(np.all(face_x < 0.0, axis=1) | np.all(face_x > 0.0, axis=1))
        left, right = projected_x < 0.0, projected_x > 0.0
        assert np.any(left) and np.any(right) and np.all(left | right)
        assert projected_x[right].min() > projected_x[left].max()
        for component in components:
            selected_faces = faces[labels[faces[:, 0]] == component]
            triangles = mesh_points[selected_faces]
            centered = triangles - mesh_points[labels == component].mean(axis=0)
            volume = np.einsum("ij,ij->i", centered[:, 0], np.cross(centered[:, 1], centered[:, 2])).sum() / 6.0
            assert volume > 0.0


@requires_cuda
def test_usd_surface_export_tracks_animated_topology_and_uvs(tmp_path):
    usd = pytest.importorskip("pxr.Usd")
    usd_geom = pytest.importorskip("pxr.UsdGeom")
    from newton.viewer import ViewerUSD

    points, spacing = particle_layout(32)
    positions = wp.array(points, dtype=wp.vec3, device="cuda:0")
    surface = ParticleSurface(spacing, positions, interpolate_rest_positions=True)
    material = BeefSurfaceMaterial()
    output_path = tmp_path / "beef_surface.usda"
    viewer = ViewerUSD(str(output_path), fps=60, num_frames=2)
    expected = []
    try:
        for frame in (1, 2):
            if frame == 2:
                separated = points.copy()
                separated[:, 0] += np.where(points[:, 0] < 0.0, -1.0, 1.0).astype(np.float32) * spacing
                positions.assign(separated)
            mesh = surface.rebuild(positions)
            assert_closed_mesh(*mesh[:2])
            uvs = material.build_uvs(mesh)
            expected.append(tuple(array.numpy().copy() for array in (mesh.vertices, mesh.indices, uvs)))
            viewer.begin_frame(frame / 60.0)
            material.log_mesh(viewer, mesh, uvs)
            viewer.end_frame()
    finally:
        viewer.close()

    assert len(expected[0][0]) != len(expected[1][0])
    assert len(expected[0][1]) != len(expected[1][1])
    stage = usd.Stage.Open(str(output_path))
    mesh = usd_geom.Mesh(stage.GetPrimAtPath("/root/beef_surface"))
    points_attr = mesh.GetPointsAttr()
    counts_attr = mesh.GetFaceVertexCountsAttr()
    indices_attr = mesh.GetFaceVertexIndicesAttr()
    uvs_attr = usd_geom.PrimvarsAPI(mesh).GetPrimvar("st")
    for attr in (points_attr, counts_attr, indices_attr, uvs_attr):
        assert attr.GetTimeSamples() == [1.0, 2.0]
    for frame, (vertices, indices, uvs) in enumerate(expected, start=1):
        np.testing.assert_array_equal(points_attr.Get(frame), vertices)
        np.testing.assert_array_equal(counts_attr.Get(frame), np.full(len(indices) // 3, 3))
        np.testing.assert_array_equal(indices_attr.Get(frame), indices)
        np.testing.assert_array_equal(uvs_attr.Get(frame), uvs)
        assert len(uvs_attr.Get(frame)) == len(points_attr.Get(frame))


def test_build_surface_flag_is_opt_in():
    parser = Example.create_parser()
    assert not parser.parse_args([]).build_surface
    assert parser.parse_args(["--build-surface"]).build_surface
