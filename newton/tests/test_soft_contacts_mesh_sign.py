# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pseudonormal inside/outside sign of full-surface mesh contacts on convex shells."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.flags import MeshSignMethod
from newton._src.geometry.soft_contacts_mesh import _feature_mesh_sign, _mesh_face_pseudonormals
from newton.tests.unittest_utils import NewtonTestCase, add_function_test, get_test_devices

# Unit tetrahedron with outward winding.
_TET_VERTICES = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
_TET_FACES = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))


def _mesh(vertices, faces):
    return newton.Mesh(np.asarray(vertices, np.float32), np.asarray(faces, np.int32).ravel(), compute_inertia=False)


def _icosahedron():
    t = (1.0 + 5.0**0.5) / 2.0
    vertices = np.array(
        (
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        ),
        dtype=np.float64,
    )
    faces = np.array(
        (
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        )
    )
    return vertices / np.linalg.norm(vertices[0]), faces


def test_pseudonormals_accept_convex_shells(test, device):
    """Convex, consistently wound shells get an outward table, including inward-wound ones."""
    table = _mesh_face_pseudonormals(_mesh(_TET_VERTICES, _TET_FACES))
    test.assertIsNotNone(table)
    # Face 0 is the bottom (z = 0) face; its face pseudonormal (feature 6) points down.
    np.testing.assert_allclose(table[6], (0.0, 0.0, -1.0), atol=1.0e-6)
    # A consistently inward-wound shell is reoriented outward.
    inward = _mesh_face_pseudonormals(_mesh(_TET_VERTICES, _TET_FACES[:, ::-1]))
    test.assertIsNotNone(inward)
    np.testing.assert_allclose(inward[6], (0.0, 0.0, -1.0), atol=1.0e-6)
    test.assertIsNotNone(_mesh_face_pseudonormals(_mesh(*_icosahedron())))


def test_pseudonormals_reject_unsafe_shells(test, device):
    """Shells whose nearest-feature side is not their inside/outside keep the parity query."""
    # One face wound against the others: every edge still borders two faces.
    flipped = _TET_FACES.copy()
    flipped[0] = flipped[0, ::-1]
    test.assertIsNone(_mesh_face_pseudonormals(_mesh(_TET_VERTICES, flipped)))
    # Two separate shells, one wound inward: no single orientation is correct.
    two = np.concatenate((_TET_VERTICES, _TET_VERTICES + 3.0))
    test.assertIsNone(_mesh_face_pseudonormals(_mesh(two, np.concatenate((_TET_FACES, _TET_FACES[:, ::-1] + 4)))))
    # Nested outward shells: parity treats the inner solid as outside.
    nested = np.concatenate((3.0 * _TET_VERTICES - 1.0, _TET_VERTICES))
    test.assertIsNone(_mesh_face_pseudonormals(_mesh(nested, np.concatenate((_TET_FACES, _TET_FACES + 4)))))
    # A sliver face that Warp's nearest-point query skips.
    sliver = np.array(((-10.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 1.0e-5)))
    test.assertIsNone(_mesh_face_pseudonormals(_mesh(sliver, _TET_FACES)))
    # A closed, consistently wound but non-convex shell (a tetrahedron with a dented face).
    dented = np.concatenate((_TET_VERTICES, [(0.25, 0.25, 0.25)]))
    dented_faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4)))
    test.assertIsNone(_mesh_face_pseudonormals(_mesh(dented, dented_faces)))


@wp.kernel
def _mesh_signs(
    mesh: wp.uint64,
    scale: wp.vec3,
    points: wp.array[wp.vec3],
    pseudonormals: wp.array[wp.vec3],
    offset: int,
    signs: wp.array[wp.int32],
):
    i = wp.tid()
    signs[i] = _feature_mesh_sign(
        mesh, wp.transform_identity(), scale, points[i], MeshSignMethod.PARITY, 1.0, pseudonormals, offset
    )


def _inside_convex(vertices, faces, local):
    """Exact inside test of a convex, outward-wound shell: every face plane has the point behind it."""
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    heights = local @ normals.T - np.einsum("ij,ij->i", normals, vertices[faces[:, 0]])
    return heights.max(axis=1)


# Inside the icosahedron, 6 cm behind every face, where the parity ray of the CUDA build grazes shared features
# and reports the point as outside.
_PARITY_DEGENERATE_POINT = (-0.36583563685417175, 0.5795437097549438, 0.32185590267181396)


def _signs(device, vertices, faces, local, scale):
    table = _mesh_face_pseudonormals(_mesh(vertices, faces))
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.ravel(), dtype=wp.int32, device=device),
    )
    points = wp.array(local * np.array(scale), dtype=wp.vec3, device=device)
    out = wp.zeros(len(local), dtype=wp.int32, device=device)
    pseudonormals = wp.array(table, dtype=wp.vec3, device=device)
    wp.launch(_mesh_signs, len(local), [mesh.id, wp.vec3(*scale), points, pseudonormals, 0, out], device=device)
    return out.numpy()


def test_pseudonormal_sign_is_exact_on_convex_shells(test, device):
    """The pseudonormal sign classifies points around a convex shell exactly, with and without scaling.

    The reference is the half-space test, not the parity query: a parity ray that passes through shared edges or
    vertices can misclassify a point deep inside, differently on CPU and CUDA.
    """
    vertices, faces = _icosahedron()
    rng = np.random.default_rng(7)
    # Points in a band around the unit-radius shell, near vertices and edges, and the parity-degenerate point.
    directions = rng.normal(size=(4000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = rng.uniform(0.7, 1.2, size=(len(directions), 1))
    local = np.concatenate((directions * radii, vertices * 1.01, vertices * 0.99, [_PARITY_DEGENERATE_POINT]))
    height = _inside_convex(vertices, faces, local)
    clear = np.abs(height) > 1.0e-3
    for scale in ((1.0, 1.0, 1.0), (2.0, 1.0, 0.5)):
        signs = _signs(device, vertices, faces, local, scale)
        np.testing.assert_array_equal(signs[clear], np.where(height[clear] < 0.0, -1, 1))
        test.assertEqual(int(signs[-1]), -1)


def test_degenerate_parity_contact_points_outward(test, device):
    """A face-vertex contact at the parity-degenerate point is flagged inside, so its normal pushes the cloth out."""
    vertices, faces = _icosahedron()
    point = np.array(_PARITY_DEGENERATE_POINT)
    corner = int(np.argmin(np.linalg.norm(vertices - point, axis=1)))
    # A small cloth triangle through the point, facing the nearest hull vertex, so that vertex projects onto it.
    axis = vertices[corner] - point
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    triangle = [point + 0.02 * u, point - 0.01 * u + 0.02 * w, point - 0.01 * u - 0.02 * w]

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_shape_convex_hull(-1, mesh=_mesh(vertices, faces))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=[wp.vec3(*x) for x in triangle],
        indices=[0, 1, 2],
        density=1.0,
        particle_radius=1.0e-3,
    )
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, soft_contact_gap=0.5, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)

    data = pipeline._soft_mesh_contact_data
    test.assertGreaterEqual(int(data.face_pseudonormal_offsets.numpy()[0]), 0)
    count = int(contacts.soft_contact_count.numpy()[0])
    features = contacts._soft_contact_mesh_features.numpy()[:count]
    rigid_vertices = data.rigid_features[0].numpy()
    corners = np.asarray(faces).ravel()[rigid_vertices[:, 1]]
    rows = [
        i
        for i, feature in enumerate(features)
        if feature[0] >= 0 and feature[0] & 7 == 1 and corners[feature[2]] == corner
    ]
    test.assertEqual(len(rows), 1)
    feature = features[rows[0]]
    test.assertTrue(feature[0] & 8, "the contact point lies inside the hull")
    test.assertFalse(feature[0] & 32, "the side is resolved")
    # Inside, the normal points from the soft point to the hull vertex, i.e. out of the solid.
    normal = contacts.soft_contact_normal.numpy()[rows[0]]
    test.assertGreater(float(normal @ axis), 0.99)


class TestSoftContactsMeshSign(NewtonTestCase):
    def setUp(self):
        self.allowOutputRegex(r"^Module .* load on device .*\n?", stream="stdout")


for fn in (test_pseudonormals_accept_convex_shells, test_pseudonormals_reject_unsafe_shells):
    add_function_test(TestSoftContactsMeshSign, fn.__name__, fn, devices=None)
for fn in (test_pseudonormal_sign_is_exact_on_convex_shells, test_degenerate_parity_contact_points_outward):
    add_function_test(TestSoftContactsMeshSign, fn.__name__, fn, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
