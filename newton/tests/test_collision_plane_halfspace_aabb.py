# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Half-space AABBs for infinite planes: surface clamping vs. residual tilt."""

from __future__ import annotations

import math
import unittest

import warp as wp

import newton

TILT = math.radians(0.05)
FAR_X = -600.0
TILT_TABLE_DEG = (0.0, 0.05, 0.081, 0.1, 1.0)
LATERAL_OFFSETS = (-100.0, -600.0)
PLANE_OFFSETS = (0.0, -25.0)


def _surface_z(normal, plane_d: float, x: float) -> float:
    """Return the plane surface height above ``x`` for ``n . p + plane_d = 0``."""
    return (-plane_d - normal[0] * x) / normal[2]


def _build(device, *, tilted: bool, shape_z_offset: float = 0.0):
    """Build one infinite plane and one sphere near its far-offset surface."""
    builder = newton.ModelBuilder()
    normal = (math.sin(TILT), 0.0, math.cos(TILT)) if tilted else (0.0, 0.0, 1.0)
    builder.add_shape_plane(plane=(*normal, 0.0), width=0.0, length=0.0)

    surface_z = _surface_z(normal, 0.001, FAR_X)
    center_z = surface_z + 0.5 / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(FAR_X, 0.0, center_z + shape_z_offset)))
    builder.add_shape_sphere(body, radius=0.5)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return pipeline, contacts


def _build_resting_box(device, *, tilt_deg: float, lateral_offset: float, plane_d: float = 0.0):
    """Build one infinite plane and a unit box resting far from its anchor."""
    builder = newton.ModelBuilder()
    tilt = math.radians(tilt_deg)
    normal = (math.sin(tilt), 0.0, math.cos(tilt))
    builder.add_shape_plane(plane=(*normal, plane_d), width=0.0, length=0.0)

    half = 0.5
    support = half * (abs(normal[0]) + abs(normal[2]))
    center_z = (-plane_d + support - 0.001 - normal[0] * lateral_offset) / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(lateral_offset, 0.0, center_z)))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return pipeline, contacts


def _plane_aabb_top(pipeline) -> float:
    """Return the upper Z bound of the infinite plane."""
    return float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][2])


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneHalfSpaceAABB(unittest.TestCase):
    """Verify half-space pruning without dropping far-offset contacts."""

    DEVICE = "cuda:0"

    def test_resting_contact_far_from_anchor_on_slightly_tilted_plane(self):
        """Retain a resting contact far from a slightly tilted plane's anchor."""
        _, contacts = _build(self.DEVICE, tilted=True)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_tilted_plane_bound_clears_the_true_surface(self):
        """Keep the tilted plane bound above every admitted surface point."""
        normal = (math.sin(TILT), 0.0, math.cos(TILT))
        pipeline, _ = _build(self.DEVICE, tilted=True)
        self.assertGreater(_plane_aabb_top(pipeline), _surface_z(normal, 0.0, FAR_X))

    def test_axis_aligned_plane_keeps_halfspace_pruning(self):
        """Clamp aligned ground at its surface while retaining resting contact."""
        pipeline, contacts = _build(self.DEVICE, tilted=False)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(_plane_aabb_top(pipeline), 1.0)

    def test_axis_aligned_plane_prunes_hovering_shape(self):
        """Prune a sphere hovering far above flat ground."""
        pipeline, contacts = _build(self.DEVICE, tilted=False, shape_z_offset=50.0)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(_plane_aabb_top(pipeline), 1.0)


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneTiltTableRestingBox(unittest.TestCase):
    """Verify conservative clamping across representative plane tilts."""

    DEVICE = "cuda:0"

    def test_resting_box_retained_across_tilt_table(self):
        """Keep far-offset resting boxes in contact across tilt and offset cases."""
        for plane_d in PLANE_OFFSETS:
            for tilt_deg in TILT_TABLE_DEG:
                for offset in LATERAL_OFFSETS:
                    with self.subTest(plane_d=plane_d, tilt_deg=tilt_deg, offset_m=offset):
                        pipeline, contacts = _build_resting_box(
                            self.DEVICE, tilt_deg=tilt_deg, lateral_offset=offset, plane_d=plane_d
                        )
                        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

                        tilt = math.radians(tilt_deg)
                        normal = (math.sin(tilt), 0.0, math.cos(tilt))
                        anchor_z = _surface_z(normal, plane_d, 0.0)
                        top = _plane_aabb_top(pipeline)
                        if tilt_deg == 0.0:
                            self.assertLess(top, anchor_z + 1.0)
                        else:
                            self.assertGreater(top, _surface_z(normal, plane_d, offset))


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneNearlyAlignedNormal(unittest.TestCase):
    """Verify nearly axis-aligned normals still enable pruning."""

    DEVICE = "cuda:0"

    def test_non_z_axis_plane_still_clamps(self):
        """Clamp a plane whose rotated normal contains roundoff residue."""
        builder = newton.ModelBuilder()
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, 0.0), width=0.0, length=0.0)
        model = builder.finalize(device=self.DEVICE)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertLess(float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][0]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
