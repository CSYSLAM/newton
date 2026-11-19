"""Operator-split, isochoric J2 log-strain return mapping for thin paperboard tetrahedra."""

import numpy as np
import warp as wp


@wp.kernel
def update_rest_metric(
    positions: wp.array[wp.vec3],
    tets: wp.array2d[int],
    rest_inverse: wp.array[wp.mat33],
    yield_strain: wp.array[float],
    hardening: float,
    relaxation: float,
    accumulated: wp.array[float],
):
    t = wp.tid()
    ids = wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3])
    p = positions[ids[0]]
    ds = wp.matrix_from_cols(positions[ids[1]] - p, positions[ids[2]] - p, positions[ids[3]] - p)
    elastic = ds * rest_inverse[t]
    if wp.determinant(elastic) > 0.05:
        u = wp.mat33()
        s = wp.vec3()
        v = wp.mat33()
        wp.svd3(elastic, u, s, v)
        log_s = wp.vec3(wp.log(wp.max(s[0], 0.05)), wp.log(wp.max(s[1], 0.05)), wp.log(wp.max(s[2], 0.05)))
        mean = (log_s[0] + log_s[1] + log_s[2]) / 3.0
        dev = log_s - wp.vec3(mean)
        equivalent = wp.sqrt((2.0 / 3.0) * wp.dot(dev, dev))
        limit = yield_strain[t] + hardening * accumulated[t]
        if equivalent > limit:
            increment = relaxation * (equivalent - limit) / (1.0 + hardening)
            retained = wp.max(0.0, 1.0 - increment / equivalent)
            ratio = wp.vec3(
                wp.exp((retained - 1.0) * dev[0]), wp.exp((retained - 1.0) * dev[1]), wp.exp((retained - 1.0) * dev[2])
            )
            # Positions are never edited. Only the stress-free metric changes after accepted loading.
            rest_inverse[t] = rest_inverse[t] * v * wp.diag(ratio) * wp.transpose(v)
            accumulated[t] = accumulated[t] + increment


class PaperPlasticity:
    def __init__(self, model, threshold=0.018, hardening=0.035, viscosity=0.015):
        self.model = model
        self.hardening = hardening
        self.viscosity = viscosity
        self.yield_strain = wp.full(model.tet_count, threshold, dtype=float)
        self.accumulated = wp.zeros(model.tet_count, dtype=float)
        self.original_inverse = wp.clone(model.tet_poses)

    def step(self, state, dt):
        wp.launch(
            update_rest_metric,
            dim=self.model.tet_count,
            inputs=[
                state.particle_q,
                self.model.tet_indices,
                self.model.tet_poses,
                self.yield_strain,
                self.hardening,
                float(1 - np.exp(-dt / self.viscosity)),
            ],
            outputs=[self.accumulated],
        )
