# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Articulated effective mass from MuJoCo installed as VBD proxy preconditioner.

See ``DESIGN.md`` section 11. The effective mass is only a preconditioner for
the partitioned solve; it never changes the public model mass.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import State
from .config import PROXY_RESPONSE_EFFECTIVE_MASS, MuJoCoVBDCouplingOptions
from .kernels import install_proxy_effective_inertia_kernel
from .ownership import MuJoCoVBDOwnership

__all__ = ["MuJoCoVBDEffectiveMass"]

# CouplingEndpointKind.BODY stable constant (see coupled/interface.py). Using the
# integer directly keeps this module free of the forbidden coupled imports.
_ENDPOINT_KIND_BODY = 0


class MuJoCoVBDEffectiveMass:
    """Evaluate articulated body blocks and install proxy preconditioners."""

    def __init__(
        self,
        mujoco_solver,
        vbd_backend,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDCouplingOptions,
    ) -> None:
        self.mujoco_solver = mujoco_solver
        self.vbd_backend = vbd_backend
        self.ownership = ownership
        self.options = options
        self.device = ownership.proxy_body_ids.device
        self.n_proxy = int(ownership.proxy_body_ids.shape[0])
        self._valid = True

        # Construction-time endpoint description (DESIGN 11.1).
        proxy_ids = np.asarray(ownership.proxy_body_ids.numpy(), dtype=np.int32)
        self.endpoint_kind = wp.array(
            np.full(self.n_proxy, _ENDPOINT_KIND_BODY, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self.endpoint_index = wp.array(proxy_ids, dtype=wp.int32, device=self.device)
        self.endpoint_local_pos = wp.zeros(max(self.n_proxy, 1), dtype=wp.vec3, device=self.device)

        self.mass = wp.zeros(max(self.n_proxy, 1), dtype=float, device=self.device)
        self.inertia = wp.zeros(max(self.n_proxy, 1), dtype=wp.mat33, device=self.device)

    def invalidate(self) -> None:
        """Mark cached blocks stale after a body inertial property change."""
        self._valid = False

    def update(self, mujoco_state: State, vbd_state: State) -> None:
        """Evaluate blocks and install them as proxy preconditioners (DESIGN 11)."""
        if self.n_proxy == 0:
            return
        _ = (mujoco_state, vbd_state)

        # Ask MuJoCo for the articulated effective mass/inertia at each proxy.
        self.mujoco_solver.coupling_eval_effective_mass_block(
            self.endpoint_kind,
            self.endpoint_index,
            self.endpoint_local_pos,
            self.mass,
            self.inertia,
        )

        self.vbd_backend.set_proxy_effective_inertia(self.mass, self.inertia)
        self._valid = True

    def install(
        self,
        out_inv_mass_effective: wp.array,
        out_inv_inertia_effective: wp.array,
        nonfinite_flag: wp.array,
    ) -> None:
        """Scatter clamped inverse mass/inertia onto proxy slots (DESIGN 11.2)."""
        if self.n_proxy == 0:
            return
        wp.launch(
            install_proxy_effective_inertia_kernel,
            dim=self.n_proxy,
            inputs=[
                self.ownership.proxy_body_ids,
                self.mass,
                self.inertia,
                self.options.proxy_mass_scale,
                self.options.proxy_mass_min,
                self.options.proxy_mass_max,
                self.options.proxy_inertia_eigenvalue_min,
                self.options.proxy_inertia_eigenvalue_max,
                PROXY_RESPONSE_EFFECTIVE_MASS,
                out_inv_mass_effective,
                out_inv_inertia_effective,
                nonfinite_flag,
            ],
            device=self.device,
        )
