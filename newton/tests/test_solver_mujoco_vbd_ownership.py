# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ownership and contact-routing tests for :class:`SolverMuJoCoVBD` (``DESIGN.md`` 23.4).

These are pure construction-time host checks and run on any device.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry import ShapeFlags
from newton._src.solvers.mujoco_vbd.contact_routing import build_contact_routing
from newton._src.solvers.mujoco_vbd.ownership import (
    OWNER_MUJOCO,
    OWNER_VBD,
    resolve_mujoco_vbd_ownership,
)


def _single_pendulum(mass: float = 1.0) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    link = b.add_link(mass=mass)
    b.add_shape_capsule(body=link, radius=0.05, half_height=0.2)
    joint = b.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
    b.add_articulation([joint])
    return b


def _two_link_chain() -> tuple[newton.ModelBuilder, list[int]]:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    base = b.add_link(mass=2.0)
    tip = b.add_link(mass=1.0)
    j0 = b.add_joint_revolute(parent=-1, child=base, axis=newton.Axis.Z)
    j1 = b.add_joint_revolute(
        parent=base,
        child=tip,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j0, j1])
    return b, [j0, j1]


def _mixed_scene() -> newton.ModelBuilder:
    """MuJoCo articulation + a free VBD dynamic body + a static ground shape."""
    b, _ = _two_link_chain()
    free_body = b.add_body()
    b.add_shape_sphere(body=free_body, radius=0.1)
    b.add_joint_free(child=free_body)
    b.add_shape_box(body=-1, hx=5.0, hy=5.0, hz=0.1)  # static ground
    return b


class TestMuJoCoVBDOwnership(unittest.TestCase):
    def test_closed_articulation_ownership(self):
        model = _single_pendulum().finalize()
        own = resolve_mujoco_vbd_ownership(model, mujoco_articulations=None, mujoco_joints=None)
        self.assertEqual(len(own.mujoco_bodies), 1)
        self.assertEqual(own.mujoco_joints, (0,))
        self.assertEqual(own.vbd_bodies, ())
        self.assertEqual(int(own.body_owner.numpy()[0]), OWNER_MUJOCO)

    def test_multiple_articulations(self):
        b, joints = _two_link_chain()
        model = b.finalize()
        own = resolve_mujoco_vbd_ownership(model, mujoco_joints=list(joints))
        self.assertEqual(sorted(own.mujoco_joints), sorted(joints))
        self.assertEqual(len(own.mujoco_bodies), 2)

    def test_reject_partial_joint_tree(self):
        b, joints = _two_link_chain()
        model = b.finalize()
        with self.assertRaises((ValueError, IndexError)):
            resolve_mujoco_vbd_ownership(model, mujoco_joints=[joints[1]])  # tip without base

    def test_reject_specifying_both_selectors(self):
        model = _single_pendulum().finalize()
        with self.assertRaises(ValueError):
            resolve_mujoco_vbd_ownership(model, mujoco_articulations=[0], mujoco_joints=[0])

    def test_hard_kinematic_is_valid_ownership_input(self):
        """Ownership is mode-independent; dispatch rejects two-way use later."""
        model = _single_pendulum(mass=0.0).finalize()
        ownership = resolve_mujoco_vbd_ownership(model, mujoco_joints=[0])
        self.assertEqual(ownership.mujoco_bodies, (0,))

    def test_mixed_scene_partitions_bodies(self):
        model = _mixed_scene().finalize()
        own = resolve_mujoco_vbd_ownership(model, mujoco_joints=[0, 1])
        owner = own.body_owner.numpy()
        self.assertTrue(any(int(o) == OWNER_MUJOCO for o in owner))
        self.assertTrue(any(int(o) == OWNER_VBD for o in owner))

    def test_contact_pair_routing_has_no_duplicates(self):
        model = _mixed_scene().finalize()
        own = resolve_mujoco_vbd_ownership(model, mujoco_joints=[0, 1])
        routing = build_contact_routing(model, own)

        def _pairs(arr) -> set[tuple[int, int]]:
            data = arr.numpy()
            return {tuple(sorted((int(a), int(b)))) for a, b in data} if data.size else set()

        mj, vbd, cross = (
            _pairs(routing.mujoco_shape_pairs),
            _pairs(routing.vbd_shape_pairs),
            _pairs(routing.cross_shape_pairs),
        )
        self.assertEqual(mj & vbd, set())
        self.assertEqual(mj & cross, set())
        self.assertEqual(vbd & cross, set())

    def test_visual_shapes_do_not_enter_cross_contacts(self):
        b = _mixed_scene()
        visual = b.add_shape_sphere(body=b.add_body(), radius=0.2)
        flags = int(b.shape_flags[visual])
        b.shape_flags[visual] = (
            (flags | int(ShapeFlags.VISIBLE)) & ~int(ShapeFlags.COLLIDE_SHAPES) & ~int(ShapeFlags.COLLIDE_PARTICLES)
        )
        model = b.finalize()
        own = resolve_mujoco_vbd_ownership(model, mujoco_joints=[0, 1])
        routing = build_contact_routing(model, own)
        cross = routing.cross_shape_pairs.numpy()
        if cross.size:
            self.assertNotIn(visual, np.unique(cross))


if __name__ == "__main__":
    unittest.main(verbosity=2)
