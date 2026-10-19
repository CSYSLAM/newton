# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Own free-rigid and material dynamics inside the private MPM domain."""

from __future__ import annotations

import warp as wp

from newton._src.core.reset import reset_world_selected
from newton._src.sim import ModelFlags, StateFlags
from newton._src.solvers.mjmpm.base import SolverBase
from newton._src.solvers.mjmpm.collision.pipeline import CollisionPipeline
from newton._src.solvers.mjmpm.coupling.interface import CouplingInterface
from newton._src.solvers.mjmpm.coupling.solver_coupled import _copy_forces, _copy_same_view_state
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM
from newton._src.solvers.mjmpm.rigid import kernels


@wp.kernel
def _reset_bodies(
    worlds: wp.array[int],
    mask: wp.array[wp.bool],
    world_count: int,
    flags: int,
    initial_q: wp.array[wp.transform],
    initial_qd: wp.array[wp.spatial_vector],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    forces: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    if reset_world_selected(worlds[i], mask, world_count):
        if (flags & int(StateFlags.BODY_Q)) != 0:
            q[i] = initial_q[i]
        if (flags & int(StateFlags.BODY_QD)) != 0:
            qd[i] = initial_qd[i]
        if (flags & int(StateFlags.BODY_F)) != 0:
            forces[i] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))


class Environment(SolverBase, CouplingInterface):
    """Integrate free rigid bodies and implicit MPM without any MuJoCo backend.

    The initial rigid contact implementation is a sequential projected-impulse
    reference path. It supports hard or implicitly compliant normal contacts
    and sliding friction impulses, with no restitution coefficient, rolling
    friction, or rigid joints.
    """

    def __init__(self, model, *, mpm_options=None, rigid_iterations=20, rigid_contact_hard=True):
        super().__init__(model)
        self.rigid_iterations = rigid_iterations
        self.rigid_contact_hard = rigid_contact_hard
        config = SolverImplicitMPM.Config()
        config.separate_worlds = True
        # Sample contact at material particles, whose physical extent is given
        # by their radius. Grid nodes outside the material must not transmit
        # impulses through an empty gap to an articulation proxy.
        config.collider_basis = "pic"
        for name, value in (mpm_options or {}).items():
            if not hasattr(config, name):
                raise ValueError(f"Unknown MPM option {name!r}")
            setattr(config, name, value)
        if config.collider_velocity_mode != "forward":
            raise ValueError("MJMPM currently requires collider_velocity_mode='forward'")
        self.material = SolverImplicitMPM(model, config=config) if model.particle_count else None
        bounded_point_bases = all(
            not basis.startswith("pic") or (basis[3:].isdigit() and int(basis[3:]) > 0)
            for basis in (config.collider_basis, config.strain_basis)
        )
        self.capture_ready = self.material is None or (
            config.grid_type == "fixed" and config.max_active_cell_count > 0 and bounded_point_bases
        )
        self.detector = CollisionPipeline(model, broad_phase="nxn", soft_contact_max=0)
        self.contacts = self.detector.contacts()
        self.contact_impulses = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
        self._init_contact_buffers()
        self.rigid_impulses = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.material_impulses = wp.zeros_like(self.rigid_impulses)
        self.work = model.state()
        self._all_worlds = wp.full(model.world_count + 1, True, dtype=wp.bool, device=model.device)

    def _init_contact_buffers(self):
        """Allocate the per-world contact solve workspace."""
        model = self.model
        device = model.device
        capacity = self.contacts.rigid_contact_max
        body_count = max(model.body_count, 1)
        # Worlds share no dynamic bodies, so each world sweeps its contacts on
        # its own thread. Global bodies may couple worlds (their mass can be
        # changed later), so they force one sequential bucket, as does a
        # single-world model.
        global_bodies = bool(model.body_count and (model.body_world.numpy() < 0).any())
        self._single_bucket = model.world_count <= 1 or global_bodies
        self._bucket_count = 1 if self._single_bucket else model.world_count + 1
        self._use_reference_rigid_solve = False
        self._contact_center = wp.zeros(body_count, dtype=wp.vec3, device=device)
        self._contact_mass_inv = wp.zeros(body_count, dtype=float, device=device)
        self._contact_inertia_inv = wp.zeros(body_count, dtype=wp.mat33, device=device)
        self._contact_bodies = wp.zeros(capacity, dtype=wp.vec2i, device=device)
        self._contact_ra = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._contact_rb = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._contact_normal = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._contact_params = wp.zeros(capacity, dtype=wp.vec4, device=device)
        # Radix sorting needs twice the sorted length.
        self._contact_keys = wp.zeros(2 * max(capacity, 1), dtype=int, device=device)
        self._contact_ids = wp.zeros(2 * max(capacity, 1), dtype=int, device=device)
        self._bucket_begin = wp.zeros(self._bucket_count, dtype=int, device=device)
        self._bucket_end = wp.zeros(self._bucket_count, dtype=int, device=device)

    def _solve_rigid_contacts(self, work, dt):
        model = self.model
        c = self.contacts
        if self._use_reference_rigid_solve:
            wp.launch(
                kernels.solve_contacts,
                dim=1,
                inputs=[
                    c.rigid_contact_count,
                    c.rigid_contact_shape0,
                    c.rigid_contact_shape1,
                    c.rigid_contact_point0,
                    c.rigid_contact_point1,
                    c.rigid_contact_normal,
                    c.rigid_contact_margin0,
                    c.rigid_contact_margin1,
                    model.shape_body,
                    model.shape_material_mu,
                    model.shape_material_ke,
                    model.shape_material_kd,
                    self.rigid_contact_hard,
                    work.body_q,
                    work.body_qd,
                    model.body_com,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.body_flags,
                    self.contact_impulses,
                    self.rigid_impulses,
                    self.rigid_iterations,
                    dt,
                ],
                device=model.device,
            )
            return
        capacity = c.rigid_contact_max
        if capacity == 0:
            return
        wp.launch(
            kernels.prepare_contact_bodies,
            dim=model.body_count,
            inputs=[work.body_q, model.body_com, model.body_inv_mass, model.body_inv_inertia, model.body_flags],
            outputs=[self._contact_center, self._contact_mass_inv, self._contact_inertia_inv],
            device=model.device,
        )
        wp.launch(
            kernels.prepare_contacts,
            dim=capacity,
            inputs=[
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                model.shape_body,
                model.shape_material_mu,
                model.shape_material_ke,
                model.shape_material_kd,
                self.rigid_contact_hard,
                work.body_q,
                model.body_world,
                model.body_flags,
                self._contact_center,
                self._contact_mass_inv,
                self._contact_inertia_inv,
                model.world_count,
                self._single_bucket,
                dt,
            ],
            outputs=[
                self._contact_bodies,
                self._contact_ra,
                self._contact_rb,
                self._contact_normal,
                self._contact_params,
                self._contact_keys,
                self._contact_ids,
                self.contact_impulses,
            ],
            device=model.device,
        )
        if self._single_bucket:
            wp.launch(
                kernels.single_bucket_range,
                dim=1,
                inputs=[c.rigid_contact_count, capacity],
                outputs=[self._bucket_begin, self._bucket_end],
                device=model.device,
            )
        else:
            wp.utils.radix_sort_pairs(self._contact_keys, self._contact_ids, capacity)
            self._bucket_begin.zero_()
            self._bucket_end.zero_()
            wp.launch(
                kernels.contact_bucket_ranges,
                dim=capacity,
                inputs=[self._contact_keys, capacity, self._bucket_count],
                outputs=[self._bucket_begin, self._bucket_end],
                device=model.device,
            )
        wp.launch(
            kernels.solve_contact_buckets,
            dim=self._bucket_count,
            inputs=[
                self._bucket_begin,
                self._bucket_end,
                self._contact_ids,
                self._contact_bodies,
                self._contact_ra,
                self._contact_rb,
                self._contact_normal,
                self._contact_params,
                self._contact_mass_inv,
                self._contact_inertia_inv,
                work.body_qd,
                self.contact_impulses,
                self.rigid_impulses,
                self.rigid_iterations,
            ],
            device=model.device,
        )

    def notify_model_changed(self, flags):
        if self.material is not None:
            self.material.notify_model_changed(flags)
            if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
                self.material.setup_collider(body_q=self.model.body_q)

    def coupling_harvest_proxy_wrenches(
        self, body_local_to_proxy_global, out_body_f, *, body_qd_before, state, state_out, contacts, dt
    ):
        out_body_f.zero_()
        wp.launch(
            kernels.harvest_feedback,
            dim=self.model.body_count,
            inputs=[body_local_to_proxy_global, self.rigid_impulses, self.material_impulses, dt, out_body_f],
            device=self.model.device,
        )

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        self.validate_observables(observables, contacts)
        model = self.model
        _copy_same_view_state(state_in, self.work)
        _copy_forces(state_in, self.work)
        work = self.work
        self.rigid_impulses.zero_()
        self.material_impulses.zero_()
        self.contact_impulses.zero_()
        if model.body_count:
            wp.launch(
                kernels.kick,
                dim=model.body_count,
                inputs=[
                    work.body_q,
                    work.body_qd,
                    work.body_f,
                    model.body_com,
                    model.body_inv_mass,
                    model.body_inertia,
                    model.body_inv_inertia,
                    model.body_flags,
                    model.body_world,
                    model.gravity,
                    dt,
                ],
                device=model.device,
            )
            self.detector.collide(work, self.contacts)
            self._solve_rigid_contacts(work, dt)
        if self.material is not None:
            self.material.step(work, work, control, None, dt)
            if model.body_count:
                impulses, points, ids = self.material.collect_collider_impulses(work)
                wp.launch(
                    kernels.collect_material_impulses,
                    dim=ids.shape[0],
                    inputs=[
                        impulses,
                        points,
                        ids,
                        self.material.collider_body_index,
                        work.body_q,
                        model.body_com,
                        self.material_impulses,
                    ],
                    device=model.device,
                )
                wp.launch(
                    kernels.apply_material_impulses,
                    dim=model.body_count,
                    inputs=[
                        self.material_impulses,
                        work.body_q,
                        work.body_qd,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.body_flags,
                    ],
                    device=model.device,
                )
        if model.body_count:
            wp.launch(
                kernels.drift,
                dim=model.body_count,
                inputs=[work.body_q, work.body_qd, model.body_com, model.body_inv_mass, model.body_flags, dt],
                device=model.device,
            )
        _copy_same_view_state(work, state_out)

    def reset(self, state, world_mask=None, flags=None):
        mask = self._normalize_reset_world_mask(world_mask)
        if mask is None:
            mask = self._all_worlds
        bits = int(StateFlags.ALL if flags is None else flags)
        if self.model.body_count:
            wp.launch(
                _reset_bodies,
                dim=self.model.body_count,
                inputs=[
                    self.model.body_world,
                    mask,
                    self.model.world_count,
                    bits,
                    self.model.body_q,
                    self.model.body_qd,
                    state.body_q,
                    state.body_qd,
                    state.body_f,
                ],
                device=self.model.device,
            )
        if self.material is not None:
            self.material.reset(state, world_mask=mask, flags=flags)
        self.detector.reset_contact_matching(world_mask=world_mask)
