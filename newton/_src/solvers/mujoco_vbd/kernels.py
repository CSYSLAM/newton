# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-map copy, sync, restore, and wrench-scatter Warp kernels.

See ``DESIGN.md`` sections 10, 11.2, and 14. All kernels use global stable body
ids; there is no generic entry-local mapping.
"""

from __future__ import annotations

import warp as wp

from .config import PROXY_RESPONSE_DIRICHLET, RELAXATION_AITKEN
from .ownership import OWNER_MUJOCO, OWNER_VBD

__all__ = [
    "compose_mujoco_body_force_kernel",
    "copy_body_state_to_backends_kernel",
    "copy_particle_state_to_backends_kernel",
    "gather_proxy_effective_inverse_kernel",
    "install_proxy_effective_inertia_kernel",
    "reconcile_owned_body_state_kernel",
    "reconcile_owned_joint_state_kernel",
    "reconcile_owned_particle_state_kernel",
    "sync_and_rewind_proxy_bodies_kernel",
    "update_relaxed_wrench_kernel",
]


@wp.kernel
def copy_body_state_to_backends_kernel(
    body_owner: wp.array[wp.int8],
    state_body_q: wp.array[wp.transform],
    state_body_qd: wp.array[wp.spatial_vector],
    mujoco_body_q: wp.array[wp.transform],
    mujoco_body_qd: wp.array[wp.spatial_vector],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
):
    """Distribute the shared initial body state to both backends (DESIGN 10.1)."""
    body = wp.tid()
    q = state_body_q[body]
    qd = state_body_qd[body]
    # Both backends see the same initial state; ownership decides who advances it.
    mujoco_body_q[body] = q
    mujoco_body_qd[body] = qd
    vbd_body_q[body] = q
    vbd_body_qd[body] = qd


@wp.kernel
def copy_particle_state_to_backends_kernel(
    state_particle_q: wp.array[wp.vec3],
    state_particle_qd: wp.array[wp.vec3],
    vbd_particle_q: wp.array[wp.vec3],
    vbd_particle_qd: wp.array[wp.vec3],
):
    """Distribute the shared initial particle state to VBD (DESIGN 10.1)."""
    particle = wp.tid()
    vbd_particle_q[particle] = state_particle_q[particle]
    vbd_particle_qd[particle] = state_particle_qd[particle]


@wp.kernel
def sync_and_rewind_proxy_bodies_kernel(
    dt: float,
    proxy_body_ids: wp.array[wp.int32],
    source_body_q: wp.array[wp.transform],
    source_body_qd: wp.array[wp.spatial_vector],
    body_gravity_acceleration: wp.array[wp.vec3],
    wrench_relaxed: wp.array[wp.spatial_vector],
    proxy_inv_mass: wp.array[float],
    proxy_inv_inertia: wp.array[wp.mat33],
    response_mode: int,
    destination_body_q: wp.array[wp.transform],
    destination_body_qd: wp.array[wp.spatial_vector],
    proxy_qd_before: wp.array[wp.spatial_vector],
):
    """Copy the solved MuJoCo pose to the VBD proxy and lagged-rewind velocity.

    See ``DESIGN.md`` section 10.2. ``DIRICHLET`` copies pose/velocity verbatim
    and forbids VBD from updating the proxy. ``EFFECTIVE_MASS`` undoes the
    previous iteration's coupling wrench so the same wrench is not applied twice.
    """
    slot = wp.tid()
    body = proxy_body_ids[slot]
    q = source_body_q[body]
    qd = source_body_qd[body]

    destination_body_q[body] = q
    proxy_qd_before[body] = qd

    if response_mode == PROXY_RESPONSE_DIRICHLET:
        destination_body_qd[body] = qd
        return

    # EFFECTIVE_MASS: remove the previously applied coupling impulse so VBD's
    # effective-mass response can re-derive the interface velocity increment.
    w = wrench_relaxed[body]
    force = wp.spatial_top(w)
    torque = wp.spatial_bottom(w)
    rotation = wp.transform_get_rotation(q)
    angular_acceleration = wp.quat_rotate(
        rotation,
        proxy_inv_inertia[slot] * wp.quat_rotate_inv(rotation, torque),
    )
    rewound = qd - wp.spatial_vector(
        force * proxy_inv_mass[slot] * dt,
        angular_acceleration * dt,
    )
    destination_body_qd[body] = rewound


@wp.kernel
def compose_mujoco_body_force_kernel(
    body_owner: wp.array[wp.int8],
    external_body_f: wp.array[wp.spatial_vector],
    coupling_body_f: wp.array[wp.spatial_vector],
    out_body_f: wp.array[wp.spatial_vector],
):
    """``out = external + coupling`` without overwriting user forces (DESIGN 10.3)."""
    body = wp.tid()
    external = external_body_f[body]
    if wp.int32(body_owner[body]) == wp.int32(OWNER_MUJOCO):
        out_body_f[body] = external + coupling_body_f[body]
    else:
        out_body_f[body] = external


@wp.kernel
def reconcile_owned_body_state_kernel(
    body_owner: wp.array[wp.int8],
    mujoco_body_q: wp.array[wp.transform],
    mujoco_body_qd: wp.array[wp.spatial_vector],
    vbd_body_q: wp.array[wp.transform],
    vbd_body_qd: wp.array[wp.spatial_vector],
    out_body_q: wp.array[wp.transform],
    out_body_qd: wp.array[wp.spatial_vector],
):
    """Write each body's public state from its single owner (DESIGN 10.4)."""
    body = wp.tid()
    if wp.int32(body_owner[body]) == wp.int32(OWNER_VBD):
        out_body_q[body] = vbd_body_q[body]
        out_body_qd[body] = vbd_body_qd[body]
    else:
        # MuJoCo owns proxy bodies; VBD proxy output never overwrites them.
        out_body_q[body] = mujoco_body_q[body]
        out_body_qd[body] = mujoco_body_qd[body]


@wp.kernel
def reconcile_owned_joint_state_kernel(
    joint_owner: wp.array[wp.int8],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_coord_count: int,
    joint_dof_count: int,
    mujoco_joint_q: wp.array[float],
    mujoco_joint_qd: wp.array[float],
    vbd_joint_q: wp.array[float],
    vbd_joint_qd: wp.array[float],
    out_joint_q: wp.array[float],
    out_joint_qd: wp.array[float],
):
    """Reconcile reduced coordinates with one writer per joint."""
    joint = wp.tid()
    q_begin = joint_q_start[joint]
    q_end = joint_coord_count
    qd_begin = joint_qd_start[joint]
    qd_end = joint_dof_count
    if joint + 1 < joint_owner.shape[0]:
        q_end = joint_q_start[joint + 1]
        qd_end = joint_qd_start[joint + 1]

    use_mujoco = wp.int32(joint_owner[joint]) == wp.int32(OWNER_MUJOCO)
    for coordinate in range(q_begin, q_end):
        out_joint_q[coordinate] = mujoco_joint_q[coordinate] if use_mujoco else vbd_joint_q[coordinate]
    for dof in range(qd_begin, qd_end):
        out_joint_qd[dof] = mujoco_joint_qd[dof] if use_mujoco else vbd_joint_qd[dof]


@wp.kernel
def reconcile_owned_particle_state_kernel(
    vbd_particle_q: wp.array[wp.vec3],
    vbd_particle_qd: wp.array[wp.vec3],
    out_particle_q: wp.array[wp.vec3],
    out_particle_qd: wp.array[wp.vec3],
):
    """Particles are VBD-owned; copy their final state (DESIGN 10.4)."""
    particle = wp.tid()
    out_particle_q[particle] = vbd_particle_q[particle]
    out_particle_qd[particle] = vbd_particle_qd[particle]


@wp.func
def _symmetrize_and_clamp_inertia(
    inertia: wp.mat33,
    eig_min: float,
    eig_max: float,
) -> wp.mat33:
    """Symmetrize inertia and clamp all principal moments."""
    sym = 0.5 * (inertia + wp.transpose(inertia))
    eigenvectors, eigenvalues = wp.eig3(sym)
    principal = wp.mat33(
        wp.clamp(eigenvalues[0], eig_min, eig_max),
        0.0,
        0.0,
        0.0,
        wp.clamp(eigenvalues[1], eig_min, eig_max),
        0.0,
        0.0,
        0.0,
        wp.clamp(eigenvalues[2], eig_min, eig_max),
    )
    return eigenvectors * principal * wp.transpose(eigenvectors)


@wp.kernel
def install_proxy_effective_inertia_kernel(
    proxy_body_ids: wp.array[wp.int32],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    mass_scale: float,
    mass_min: float,
    mass_max: float,
    inertia_eigenvalue_min: float,
    inertia_eigenvalue_max: float,
    response_mode: int,
    out_inv_mass_effective: wp.array[float],
    out_inv_inertia_effective: wp.array[wp.mat33],
    nonfinite_flag: wp.array[wp.int32],
):
    """Install clamped effective inverse mass/inertia on proxy slots (DESIGN 11.2)."""
    slot = wp.tid()
    body = proxy_body_ids[slot]

    if response_mode == PROXY_RESPONSE_DIRICHLET:
        out_inv_mass_effective[body] = 0.0
        out_inv_inertia_effective[body] = wp.mat33(0.0)
        return

    m = mass[slot] * mass_scale
    if not (m == m) or m <= 0.0:  # NaN or non-positive
        wp.atomic_add(nonfinite_flag, 0, 1)
        out_inv_mass_effective[body] = 0.0
        out_inv_inertia_effective[body] = wp.mat33(0.0)
        return

    m = wp.clamp(m, mass_min, mass_max)
    out_inv_mass_effective[body] = 1.0 / m

    inertia_is_finite = True
    for row in range(3):
        for column in range(3):
            value = inertia[slot][row, column]
            if not (value == value) or wp.abs(value) > 1.0e30:
                inertia_is_finite = False
    if not inertia_is_finite:
        wp.atomic_add(nonfinite_flag, 0, 1)
        out_inv_inertia_effective[body] = wp.mat33(0.0)
        return

    clamped = _symmetrize_and_clamp_inertia(inertia[slot], inertia_eigenvalue_min, inertia_eigenvalue_max)
    det = wp.determinant(clamped)
    if not (det == det) or det <= 0.0:
        wp.atomic_add(nonfinite_flag, 0, 1)
        out_inv_inertia_effective[body] = wp.mat33(0.0)
        return
    out_inv_inertia_effective[body] = wp.inverse(clamped)


@wp.kernel
def gather_proxy_effective_inverse_kernel(
    proxy_body_ids: wp.array[wp.int32],
    body_inv_mass_effective: wp.array[float],
    body_inv_inertia_effective: wp.array[wp.mat33],
    out_proxy_inv_mass: wp.array[float],
    out_proxy_inv_inertia: wp.array[wp.mat33],
):
    """Gather installed effective inverse inertia into fixed proxy-slot arrays."""
    slot = wp.tid()
    body = proxy_body_ids[slot]
    out_proxy_inv_mass[slot] = body_inv_mass_effective[body]
    out_proxy_inv_inertia[slot] = body_inv_inertia_effective[body]


@wp.kernel
def update_relaxed_wrench_kernel(
    iteration: int,
    relaxation_mode: int,
    relaxation_initial: float,
    relaxation_min: float,
    relaxation_max: float,
    body_world: wp.array[wp.int32],
    proxy_body_ids: wp.array[wp.int32],
    wrench_raw: wp.array[wp.spatial_vector],
    wrench_previous: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
    aitken_omega: wp.array[float],
    out_wrench_relaxed: wp.array[wp.spatial_vector],
    out_residual_current: wp.array[wp.spatial_vector],
):
    """Per-proxy fixed/Aitken relaxation update (``DESIGN.md`` section 14).

    ``r[k] = Wraw[k] - W[k-1]``; ``W[k] = W[k-1] + omega * r[k]``. The Aitken
    ``omega`` is a per-world scalar produced by a separate reduction; this kernel
    reads it and applies the update. The reduction is done by the convergence
    module.
    """
    slot = wp.tid()
    body = proxy_body_ids[slot]
    world = body_world[body]

    w_prev = wrench_previous[body]
    raw = wrench_raw[body]
    finite = True
    for component in range(6):
        value = raw[component]
        if not (value == value) or wp.abs(value) > 1.0e30:
            finite = False
    if not finite:
        out_residual_current[body] = wp.spatial_vector()
        out_wrench_relaxed[body] = w_prev
        return

    r = raw - w_prev
    out_residual_current[body] = r

    omega = relaxation_initial
    if relaxation_mode == RELAXATION_AITKEN and iteration > 0:
        omega = wp.clamp(aitken_omega[world], relaxation_min, relaxation_max)

    out_wrench_relaxed[body] = w_prev + omega * r


# Referenced so the owner constants remain a single source of truth.
_ = (OWNER_VBD,)
