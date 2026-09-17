# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check same-color soft-contact majorization and unchanged contact forces."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.vbd.rigid_vbd_kernels import (
    _NUM_RIGID_CONTACT_THREADS_PER_BODY,
    accumulate_body_body_contacts_per_body,
)


class TestContactMajorizer(unittest.TestCase):
    def test_only_simultaneous_dynamic_pair_hessians_change(self):
        """Preserve forces and off-color or immovable contact responses."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for colors, inverse_mass, hard_contacts, expected in (
                ([0, 0], [1.0, 1.0], 0, 2.0),
                ([0, 1], [1.0, 1.0], 0, 1.0),
                ([0, 0], [1.0, 0.0], 0, 1.0),
                ([0, 0], [1.0, 1.0], 1, 1.0),
            ):
                with self.subTest(device=device, colors=colors, inverse_mass=inverse_mass):

                    def array(values, dtype, device=device):
                        return wp.array(values, dtype=dtype, device=device)

                    inputs = [
                        0.01,
                        array([0, 1], int),
                        array([[0, 0, 0, 0, 0, 0, 1], [0.015, -0.001, 0, 0, 0, 0, 1]], wp.transform),
                        array([[0, 0, 0, 0, 0, 0, 1], [0.015, 0, 0, 0, 0, 0, 1]], wp.transform),
                        array([[0, 0, 0]] * 2, wp.vec3),
                        array(inverse_mass, float),
                        0.001,
                        array([1000], float),
                        array([1000], float),
                        array([0.2], float),
                        array([0.5], float),
                        array([[0, 0, 0]], wp.vec3),
                        array([[0, 0, 0]], wp.vec3),
                        0.9,
                        hard_contacts,
                        array([1], int),
                        array([0], int),
                        array([1], int),
                        array([[0.01, 0, 0]], wp.vec3),
                        array([[-0.01, 0, 0]], wp.vec3),
                        array([[0, 0, 0]], wp.vec3),
                        array([[0, 0, 0]], wp.vec3),
                        array([[1, 0, 0]], wp.vec3),
                        array([0], float),
                        array([0], float),
                        array([0, 1], int),
                        1,
                        array([1, 1], int),
                        array([0, 0], int),
                    ]
                    output = []
                    for pair_colors in ([0, 1], colors):
                        values = list(inputs)
                        values.insert(6, array(pair_colors, int))
                        fields = [
                            wp.zeros(2, dtype=dtype, device=device)
                            for dtype in (wp.vec3, wp.vec3, wp.mat33, wp.mat33, wp.mat33)
                        ]
                        wp.launch(
                            accumulate_body_body_contacts_per_body,
                            2 * _NUM_RIGID_CONTACT_THREADS_PER_BODY,
                            block_dim=_NUM_RIGID_CONTACT_THREADS_PER_BODY,
                            inputs=values,
                            outputs=fields,
                            device=device,
                        )
                        output.append([field.numpy().copy() for field in fields])
                        # Exercise both an empty adjacency and more contacts
                        # than one warp; every strided lane must sum exactly once.
                        for count in (0, 37):
                            repeated = list(values)
                            for slot in (*range(8, 14), *range(17, 26)):
                                source = values[slot]
                                repeated[slot] = array(np.repeat(source.numpy(), 37, axis=0), source.dtype)
                            repeated[16] = array([count], int)
                            repeated[27] = 37
                            repeated[28] = array([count, count], int)
                            repeated[29] = array(np.tile(np.arange(37), 2), int)
                            for field in fields:
                                field.zero_()
                            wp.launch(
                                accumulate_body_body_contacts_per_body,
                                2 * _NUM_RIGID_CONTACT_THREADS_PER_BODY,
                                block_dim=_NUM_RIGID_CONTACT_THREADS_PER_BODY,
                                inputs=repeated,
                                outputs=fields,
                                device=device,
                            )
                            for single, field in zip(output[-1], fields, strict=True):
                                np.testing.assert_allclose(field.numpy(), count * single, rtol=3e-6, atol=1e-5)
                    self.assertGreater(np.linalg.norm(output[0][0]), 0.0)
                    for index, (before, after) in enumerate(zip(*output, strict=True)):
                        np.testing.assert_allclose(
                            after, before * (expected if index >= 2 else 1.0), rtol=2e-6, atol=1e-6
                        )


if __name__ == "__main__":
    unittest.main()
