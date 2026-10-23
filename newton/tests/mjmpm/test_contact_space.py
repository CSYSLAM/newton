# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Verify bounded contact-space sweeps and captured reference fallback."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.rigid import contact_space, kernels, warp_sweeps


def _cross(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _fixture(device, softness, warm):
    rng = np.random.default_rng(17)
    capacity = 65
    pairs = np.array([[i % 3 - 1, 2] for i in range(capacity)], dtype=np.int32)
    ra = rng.uniform(-0.05, 0.05, (capacity, 3)).astype(np.float32)
    rb = rng.uniform(-0.05, 0.05, (capacity, 3)).astype(np.float32)
    normals = rng.normal(size=(capacity, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    matrix = rng.normal(size=(12, 12))
    mobility = (0.003 * matrix @ matrix.T + 0.1 * np.eye(12)).astype(np.float32)
    mass = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    inertia = np.array([np.eye(3), np.eye(3), np.diag([3.0, 4.0, 5.0])], dtype=np.float32)
    params = np.zeros((capacity, 4), dtype=np.float32)
    params[:, 0] = rng.uniform(0.0, 0.01, capacity)
    params[:, 1] = softness
    params[:, 3] = 0.8
    for c, pair in enumerate(pairs):
        point = mass[2] * np.eye(3) - _cross(rb[c]) @ inertia[2] @ _cross(rb[c])
        if pair[0] >= 0:
            link = pair[0]
            jacobian = np.concatenate((np.eye(3), -_cross(ra[c])), axis=1)
            point += jacobian @ mobility[6 * link : 6 * link + 6, 6 * link : 6 * link + 6] @ jacobian.T
        params[c, 2] = normals[c] @ point @ normals[c]

    def array(data, dtype):
        return wp.array(data, dtype=dtype, device=device)

    env = SimpleNamespace(
        model=SimpleNamespace(body_count=3, device=device),
        contacts=SimpleNamespace(rigid_contact_count=array([capacity], int)),
        _contact_bodies=array(pairs, wp.vec2i),
        _contact_ra=array(ra, wp.vec3),
        _contact_rb=array(rb, wp.vec3),
        _contact_normal=array(normals, wp.vec3),
        _contact_params=array(params, wp.vec4),
        _contact_mass_inv=array(mass, float),
        _contact_inertia_inv=array(inertia, wp.mat33),
        _link_slot=array([0, 1, -1], int),
        _link_body=array([[0, 1]], int),
        _link_count=2,
        _mobility=array(mobility[None], float),
        _contacted_links=array([[0, 1]], int),
        _contacted_count=array([2], int),
        _point_mobility=wp.zeros(capacity, dtype=wp.mat33, device=device),
        _link_responses=wp.zeros((capacity, kernels.MAX_RESPONSE_LINKS), dtype=kernels.mat63, device=device),
        rigid_iterations=60,
        _contact_space_skip=None,
    )
    initial = normals * (0.001 if warm else 0.0)
    env.contact_impulses = array(initial, wp.vec3)
    env.rigid_impulses = array(rng.uniform(-1.0e-4, 1.0e-4, (3, 6)), wp.spatial_vector)
    work = SimpleNamespace(body_qd=array(rng.uniform(-0.1, 0.1, (3, 6)), wp.spatial_vector))
    wp.launch(
        kernels.prepare_link_contact_responses,
        dim=(capacity, kernels.MAX_RESPONSE_LINKS),
        inputs=[
            env.contacts.rigid_contact_count,
            env._contact_bodies,
            env._contact_ra,
            env._contact_rb,
            env._contact_mass_inv,
            env._link_slot,
            env._link_count,
            env._mobility,
            env._contacted_links,
            env._contacted_count,
        ],
        outputs=[env._point_mobility, env._link_responses],
        device=device,
    )
    ids = array(np.arange(capacity), int)
    begin = array([0], int)
    packed = wp.empty(capacity, dtype=kernels.PackedContact, device=device)
    fallback_flags = wp.zeros(1, dtype=int, device=device)

    def reference():
        wp.launch(
            warp_sweeps.solve_contact_buckets_cooperative,
            dim=(1, 32),
            block_dim=32,
            inputs=[
                begin,
                env.contacts.rigid_contact_count,
                ids,
                env._contact_bodies,
                env._contact_ra,
                env._contact_rb,
                env._contact_normal,
                env._contact_params,
                env._contact_mass_inv,
                env._contact_inertia_inv,
                env._link_slot,
                env._link_body,
                env._mobility,
                env._contacted_links,
                env._contacted_count,
                env._point_mobility,
                env._link_responses,
                work.body_qd,
                env.contact_impulses,
                env.rigid_impulses,
                env.rigid_iterations,
                packed,
                fallback_flags,
                env._contact_space_skip,
            ],
            device=device,
        )

    return env, work, reference


class TestContactSpace(unittest.TestCase):
    def test_sweeps_and_device_capacity_fallback(self):
        """Keep hard/soft friction reactions and handle changing counts in one graph."""
        for device in wp.get_cuda_devices():
            for softness, warm in ((0.0, False), (0.0, True), (1.35, False), (1.35, True)):
                with self.subTest(device=device, softness=softness, warm=warm):
                    env, work, reference = _fixture(device, softness, warm)
                    workspace = contact_space.ContactSpaceBuffers(3, 65, device)
                    outputs = (work.body_qd, env.contact_impulses, env.rigid_impulses)
                    initial = [wp.clone(v) for v in outputs]
                    actual = [wp.empty_like(v) for v in outputs]
                    expected = [wp.empty_like(v) for v in outputs]

                    def pair(
                        env=env,
                        work=work,
                        reference=reference,
                        workspace=workspace,
                        outputs=outputs,
                        initial=initial,
                        actual=actual,
                        expected=expected,
                    ):
                        for a, b in zip(outputs, initial, strict=True):
                            wp.copy(a, b)
                        workspace.run(env, work, reference)
                        for a, b in zip(actual, outputs, strict=True):
                            wp.copy(a, b)
                        for a, b in zip(outputs, initial, strict=True):
                            wp.copy(a, b)
                        reference()
                        for a, b in zip(expected, outputs, strict=True):
                            wp.copy(a, b)

                    pair()
                    with wp.ScopedCapture(device=device) as capture:
                        pair()
                    for count in (19, 35, 65, 0):
                        env.contacts.rigid_contact_count.assign([count])
                        workspace.stats.zero_()
                        wp.capture_launch(capture.graph)
                        for k, (a, b) in enumerate(zip(actual, expected, strict=True)):
                            if count > contact_space.CAPACITY:
                                np.testing.assert_array_equal(a.numpy(), b.numpy())
                            else:
                                np.testing.assert_allclose(
                                    a.numpy(), b.numpy(), atol=1.0e-4 if k == 0 else 1.0e-5, rtol=1.0e-5
                                )
                        stats = workspace.stats.numpy()
                        self.assertEqual(stats[0], 1)
                        self.assertEqual(stats[1], int(count <= contact_space.CAPACITY))
                        self.assertEqual(stats[2], int(count <= contact_space.CAPACITY))
                        if 0 < count <= contact_space.CAPACITY:
                            impulse = actual[1].numpy()[:count]
                            normals = env._contact_normal.numpy()[:count]
                            load = np.sum(impulse * normals, axis=1)
                            tangent = np.linalg.norm(impulse - load[:, None] * normals, axis=1)
                            self.assertTrue((load >= -1.0e-7).all())
                            self.assertTrue((tangent <= 0.8 * load + 1.0e-7).all())

                    # Force the consistency guard to reject, and require exact
                    # restoration before the original solve is replayed.
                    env.contacts.rigid_contact_count.assign([19])
                    workspace.velocity_tolerance = -1.0
                    workspace.stats.zero_()
                    with wp.ScopedCapture(device=device) as capture:
                        pair()
                    wp.capture_launch(capture.graph)
                    self.assertEqual(workspace.stats.numpy()[3], 1)
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(a.numpy(), b.numpy())


if __name__ == "__main__":
    unittest.main()
