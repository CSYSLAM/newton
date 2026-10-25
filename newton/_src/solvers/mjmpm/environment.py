# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Own free-rigid and material dynamics inside the private MPM domain."""

from __future__ import annotations

import numpy as np
import warp as wp

from newton._src.core.reset import reset_world_selected
from newton._src.sim import BodyFlags, ModelFlags, StateFlags
from newton._src.solvers.mjmpm.base import SolverBase
from newton._src.solvers.mjmpm.collision.pipeline import CollisionPipeline
from newton._src.solvers.mjmpm.coupling.interface import CouplingInterface
from newton._src.solvers.mjmpm.coupling.solver_coupled import _copy_forces, _copy_same_view_state
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM
from newton._src.solvers.mjmpm.rigid import kernels, response


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

    def __init__(
        self,
        model,
        *,
        mpm_options=None,
        rigid_iterations=20,
        rigid_contact_hard=True,
        rigid_warm_start=False,
        rigid_solver="gauss_seidel",
        rigid_material_coupling="constrained",
    ):
        super().__init__(model)
        if rigid_solver not in ("gauss_seidel", "jacobi"):
            raise ValueError("rigid_solver must be 'gauss_seidel' or 'jacobi'")
        if rigid_material_coupling not in ("implicit", "lagged", "constrained"):
            raise ValueError("rigid_material_coupling must be 'implicit', 'lagged', or 'constrained'")
        self.rigid_iterations = rigid_iterations
        self.rigid_contact_hard = rigid_contact_hard
        self.rigid_warm_start = rigid_warm_start
        self.rigid_solver = rigid_solver
        self.rigid_material_coupling = rigid_material_coupling
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
        # Plain "pic" bases adapt their capacity during uncaptured steps.
        self.capture_ready = self.material is None or (
            self.material.max_active_cell_count > 0
            and (config.grid_type == "fixed" or self.material._sparse_rebuildable)
        )
        # Warm starts carry each contact's impulse to its match in the next step.
        self.detector = CollisionPipeline(
            model,
            broad_phase="nxn",
            soft_contact_max=0,
            contact_matching="latest" if rigid_warm_start else "disabled",
        )
        self._exclude_proxy_pairs()
        self.contacts = self.detector.contacts()
        self.contact_impulses = wp.zeros(self.contacts.rigid_contact_max, dtype=wp.vec3, device=model.device)
        self._previous_impulses = wp.zeros_like(self.contact_impulses) if rigid_warm_start else None
        self._init_contact_buffers()
        self._reuse_contacts = False
        # Articulated link response; see enable_articulated_response().
        self._link_mobility = None
        self._mobility = None
        self._link_slot = None
        self._link_body = None
        self._link_count = 0
        self._contacted_links = None
        self._contacted_count = None
        self._contact_space = None
        self._contact_space_skip = None
        self._recovering_contacts = False
        self.rigid_impulses = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.material_impulses = wp.zeros_like(self.rigid_impulses)
        self.work = model.state()
        self._all_worlds = wp.full(model.world_count + 1, True, dtype=wp.bool, device=model.device)
        # Lagged rigid-material coupling: the material solve sees free rigid
        # bodies as kinematic, and its reaction reaches them at the start of the
        # next step, before the rigid contact solve.
        self._lagged_material = None
        if self.material is not None and model.body_count and rigid_material_coupling == "lagged":
            self._lagged_material = wp.zeros_like(self.rigid_impulses)
            self._lagged_material_saved = wp.zeros_like(self.rigid_impulses)
            self._immediate_material = wp.zeros_like(self.rigid_impulses)
            self.material.setup_collider(body_q=model.body_q, body_mass=self._material_body_mass())
        self._constrained_material = (
            self.material is not None and model.body_count > 0 and rigid_material_coupling == "constrained"
        )
        if self._constrained_material:
            self._response_enabled = wp.zeros(model.body_count, dtype=int, device=model.device)
            self._body_response = wp.zeros(model.body_count, dtype=response.mat66, device=model.device)
            self._material_prediction = wp.zeros_like(self.material_impulses)
            self._saved_prediction = wp.zeros_like(self.material_impulses)
            self._response_start = wp.zeros_like(self.material_impulses)
            self._response_base = wp.zeros_like(self.material_impulses)
            self._response_accepted = wp.zeros_like(self.material_impulses)
            self.material_velocity_residual = wp.zeros_like(self.material_impulses)
            self.momentum_residual = wp.zeros_like(self.material_impulses)
            self.constraint_velocity_residual = wp.zeros_like(self.material_impulses)
            self.material.enable_independent_response(self._response_enabled, self._body_response)
        # The material terms that read only particles overlap the latency-bound
        # rigid contact solve on a second stream.
        self._material_stream = None
        if self.material is not None and model.body_count and model.device.is_cuda:
            self._material_stream = wp.Stream(model.device)
            self._material_fork = wp.Event(model.device)
            self._material_join = wp.Event(model.device)

    def _exclude_proxy_pairs(self):
        """Skip shape pairs whose contacts the rigid solve always discards.

        MuJoCo owns articulation self-contact and articulation-static contact,
        so the environment never solves proxy-proxy or proxy-static pairs.
        Proxy flags are fixed at construction, unlike kinematic flags, so
        those pairs are excluded from broad-phase candidates up front.
        """
        model = self.model
        if model.shape_count == 0 or model.body_count == 0:
            return
        shape_body = model.shape_body.numpy()
        shape_world = model.shape_world.numpy()
        proxy_body = (model.body_flags.numpy() & int(BodyFlags.PROXY)) != 0
        proxy = np.flatnonzero((shape_body >= 0) & proxy_body[np.maximum(shape_body, 0)])
        if proxy.shape[0] == 0:
            return
        static = np.flatnonzero(shape_body < 0)
        global_static = static[shape_world[static] < 0]
        global_proxy = proxy[shape_world[proxy] < 0]
        chunks = []
        # Pair shapes world by world so the candidate list stays linear in the world count.
        for world in np.unique(shape_world[proxy]):
            if world < 0:
                world_proxy = global_proxy
                others = np.concatenate((proxy, static))
            else:
                world_proxy = proxy[shape_world[proxy] == world]
                others = np.concatenate(
                    (world_proxy, global_proxy, static[shape_world[static] == world], global_static)
                )
            a = np.repeat(world_proxy, others.shape[0])
            b = np.tile(others, world_proxy.shape[0])
            keep = a != b
            chunks.append(np.stack((np.minimum(a, b), np.maximum(a, b)), axis=1)[keep])
        pairs = np.concatenate(chunks)
        existing = model.shape_collision_filter_pairs_array()
        if existing.shape[0]:
            pairs = np.concatenate((pairs, np.asarray(existing, dtype=pairs.dtype).reshape(-1, 2)))
        pairs = np.unique(pairs.astype(np.int32), axis=0)
        self.detector.shape_pairs_excluded = wp.array(pairs, dtype=wp.vec2i, device=model.device)
        self.detector.shape_pairs_excluded_count = pairs.shape[0]

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
        # Gauss-Seidel sweeps on bucket-local body state, with a reference fallback.
        self._packed_contacts = wp.zeros(capacity, dtype=kernels.PackedContact, device=device)
        self._bucket_fallback = wp.zeros(self._bucket_count, dtype=int, device=device)
        self._use_local_rigid_buckets = True
        self._use_cooperative_rigid_buckets = True
        self._use_contact_space = True
        self._contact_space_recovery_only = False
        # Jacobi sweeps: contact counts for mass splitting and two wrench
        # accumulators, each cleared by the sweep that does not write it.
        self._body_contacts = wp.zeros(body_count, dtype=int, device=device)
        self._world_link_contacts = wp.zeros(1, dtype=int, device=device)
        self._delta_wrench = wp.zeros((2, body_count), dtype=wp.spatial_vector, device=device)

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
        reuse_prepared = self._reuse_recovery_contact_geometry()
        # The material solve changes velocities, but body poses and contact
        # geometry are unchanged until drift. Reuse their prepared data for
        # the current-load reaction solve.
        if not reuse_prepared:
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
                    self._link_slot,
                    self._link_count,
                    self._mobility,
                    c.rigid_contact_match_index if self.rigid_warm_start else None,
                    self._previous_impulses,
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
        if self._mobility is not None:
            wp.copy(self._qd_before_contacts, work.body_qd)
            if not reuse_prepared:
                self._contacted.zero_()
                wp.launch(
                    kernels.mark_contacted_links,
                    dim=capacity,
                    inputs=[c.rigid_contact_count, self._contact_bodies, self._link_slot],
                    outputs=[self._contacted],
                    device=model.device,
                )
                wp.launch(
                    kernels.compact_contacted_links,
                    dim=self._link_body.shape[0],
                    inputs=[self._contacted, self._link_count],
                    outputs=[self._contacted_links, self._contacted_count],
                    device=model.device,
                )
                if self.rigid_solver == "gauss_seidel":
                    wp.launch(
                        kernels.prepare_link_contact_responses,
                        dim=(capacity, kernels.MAX_RESPONSE_LINKS),
                        inputs=[
                            c.rigid_contact_count,
                            self._contact_bodies,
                            self._contact_ra,
                            self._contact_rb,
                            self._contact_mass_inv,
                            self._link_slot,
                            self._link_count,
                            self._mobility,
                            self._contacted_links,
                            self._contacted_count,
                        ],
                        outputs=[self._point_mobility, self._link_responses],
                        device=model.device,
                    )
        if self.rigid_solver == "jacobi":
            self._solve_contacts_jacobi(work, capacity)
        else:
            self._solve_contacts_gauss_seidel(work)
        if self._mobility is not None:
            wp.launch(
                kernels.finish_articulated_contacts,
                dim=self._link_body.shape,
                inputs=[self._link_body, self._mobility, self._qd_before_contacts, self.rigid_impulses],
                outputs=[work.body_qd],
                device=model.device,
            )

    def _solve_contacts_gauss_seidel(self, work):
        if (
            self._use_contact_space
            and self._constrained_material
            and self._contact_space is not None
            and self._use_local_rigid_buckets
            and self._use_cooperative_rigid_buckets
            and (self._recovering_contacts or not self._contact_space_recovery_only)
        ):
            self._contact_space.run(
                self,
                work,
                lambda: self._solve_contacts_gauss_seidel_sweeps(work),
                reuse_geometry=self._reuse_recovery_contact_geometry(),
            )
            return
        self._solve_contacts_gauss_seidel_sweeps(work)

    def _reuse_recovery_contact_geometry(self):
        return (
            self._recovering_contacts
            and self._use_contact_space
            and self._constrained_material
            and self._contact_space is not None
            and self._use_local_rigid_buckets
            and self._use_cooperative_rigid_buckets
            and self.rigid_solver == "gauss_seidel"
            and not self._contact_space_recovery_only
        )

    def _solve_contacts_gauss_seidel_sweeps(self, work):
        model = self.model
        c = self.contacts
        capacity = c.rigid_contact_max
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
        inputs = [
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
            self._link_slot,
            self._link_body,
            self._mobility,
            self._contacted_links,
            self._contacted_count,
            self._point_mobility if self._mobility is not None else None,
            self._link_responses if self._mobility is not None else None,
            work.body_qd,
            self.contact_impulses,
            self.rigid_impulses,
            self.rigid_iterations,
        ]
        if self._use_local_rigid_buckets:
            if self._use_cooperative_rigid_buckets and self._mobility is not None and model.device.is_cuda:
                from newton._src.solvers.mjmpm.rigid.warp_sweeps import (  # noqa: PLC0415
                    solve_contact_buckets_cooperative,
                )

                wp.launch(
                    solve_contact_buckets_cooperative,
                    dim=(self._bucket_count, 32),
                    inputs=[*inputs, self._packed_contacts, self._bucket_fallback, self._contact_space_skip],
                    device=model.device,
                    block_dim=32,
                )
            else:
                wp.launch(
                    kernels.solve_contact_buckets_local,
                    dim=self._bucket_count,
                    inputs=[*inputs, self._packed_contacts, self._bucket_fallback],
                    device=model.device,
                )
        else:
            self._bucket_fallback.fill_(1)
        wp.launch(
            kernels.solve_contact_buckets,
            dim=self._bucket_count,
            inputs=[*inputs, self._bucket_fallback],
            device=model.device,
        )

    def _solve_contacts_jacobi(self, work, capacity):
        """Sweep all contacts in parallel with mass splitting instead of in sequence."""
        model = self.model
        device = model.device
        body_count = model.body_count
        self._body_contacts.zero_()
        self._world_link_contacts.zero_()
        self._delta_wrench.zero_()
        wp.launch(
            kernels.count_split_contacts,
            dim=capacity,
            inputs=[self._contact_bodies, self._contact_mass_inv, self._link_slot, self._link_count],
            outputs=[self._body_contacts, self._world_link_contacts],
            device=device,
        )
        buffers = (self._delta_wrench[0], self._delta_wrench[1])

        def apply(delta, other):
            wp.launch(
                kernels.apply_contact_wrenches_jacobi,
                dim=body_count,
                inputs=[
                    delta,
                    self._contact_mass_inv,
                    self._contact_inertia_inv,
                    self._link_slot,
                    self._link_body,
                    self._mobility,
                    self._contacted_links,
                    self._contacted_count,
                ],
                outputs=[work.body_qd, other],
                device=device,
            )

        # Warm-started impulses act on the bodies before the first sweep.
        wp.launch(
            kernels.scatter_initial_contact_impulses,
            dim=capacity,
            inputs=[
                self._contact_bodies,
                self._contact_ra,
                self._contact_rb,
                self._contact_mass_inv,
                self.contact_impulses,
            ],
            outputs=[buffers[0], self.rigid_impulses],
            device=device,
        )
        apply(buffers[0], buffers[1])
        for sweep in range(self.rigid_iterations):
            delta, other = buffers[(sweep + 1) % 2], buffers[sweep % 2]
            wp.launch(
                kernels.solve_contacts_jacobi,
                dim=capacity,
                inputs=[
                    self._contact_bodies,
                    self._contact_ra,
                    self._contact_rb,
                    self._contact_normal,
                    self._contact_params,
                    self._contact_mass_inv,
                    self._contact_inertia_inv,
                    self._link_slot,
                    self._link_count,
                    self._mobility,
                    self._body_contacts,
                    self._world_link_contacts,
                ],
                outputs=[work.body_qd, self.contact_impulses, delta, self.rigid_impulses],
                device=device,
            )
            apply(delta, other)

    def enable_articulated_response(self, link_mobility, link_body):
        """Let articulation proxies respond through the MuJoCo link mobility.

        Args:
            link_mobility: :class:`~newton._src.solvers.mjmpm.coupling.articulated.LinkMobility`
                refreshed after each MuJoCo step.
            link_body: Proxy body index in this environment of each mobility
                link slot, shape ``(world_count, link_count)``.
        """
        model = self.model
        device = model.device
        link_body = np.asarray(link_body, dtype=np.int32)
        slots = np.full(model.body_count, -1, dtype=np.int32)
        for world in range(link_body.shape[0]):
            for link in range(link_body.shape[1]):
                body = link_body[world, link]
                if body >= 0:
                    slots[body] = world * link_body.shape[1] + link
        self._link_mobility = link_mobility
        self._mobility = link_mobility.mobility
        self._link_count = int(link_body.shape[1])
        self._link_body = wp.array(link_body, dtype=int, device=device)
        self._link_slot = wp.array(slots, dtype=int, device=device)
        self._contacted = wp.zeros(link_body.size, dtype=int, device=device)
        self._contacted_links = wp.zeros(link_body.shape, dtype=int, device=device)
        self._contacted_count = wp.zeros(link_body.shape[0], dtype=int, device=device)
        self._qd_before_contacts = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
        self._world_link_contacts = wp.zeros(link_body.shape[0], dtype=int, device=device)
        capacity = self.contacts.rigid_contact_max
        self._point_mobility = wp.zeros(capacity, dtype=wp.mat33, device=device)
        self._link_responses = wp.zeros((capacity, kernels.MAX_RESPONSE_LINKS), dtype=kernels.mat63, device=device)
        if device.is_cuda and self._single_bucket and self._constrained_material:
            from newton._src.solvers.mjmpm.rigid.contact_space import ContactSpaceBuffers  # noqa: PLC0415

            self._contact_space = ContactSpaceBuffers(model.body_count, capacity, device)
        if self.material is not None:
            self.material.enable_articulated_response(self._mobility, self._link_body, self._link_slot)

    def coupling_rewind_proxy_body(
        self, body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt
    ):
        """Rewind lagged feedback through the link mobility when articulated response is enabled."""
        if self._link_mobility is None:
            super().coupling_rewind_proxy_body(
                body_local_to_proxy_global, state, coupling_forces, body_gravity_acceleration, dt
            )
            return
        # MuJoCo has just stepped; evaluate the response at this step's configuration.
        self._link_mobility.update()
        wp.launch(
            kernels.rewind_articulated_proxies,
            dim=self._link_body.shape,
            inputs=[
                self._link_body,
                body_local_to_proxy_global,
                coupling_forces,
                self._mobility,
                self.model.body_mass,
                body_gravity_acceleration,
                dt,
            ],
            outputs=[state.body_qd, state.body_f],
            device=self.model.device,
        )

    def _material_body_mass(self):
        """Body masses seen by the material solve: zero for free rigid bodies under lagged coupling."""
        model = self.model
        mass = model.body_mass.numpy().copy()
        flags = model.body_flags.numpy()
        free = (flags & (int(BodyFlags.KINEMATIC) | int(BodyFlags.PROXY))) == 0
        mass[free] = 0.0
        mass[(flags & int(BodyFlags.KINEMATIC)) != 0] = 0.0
        return wp.array(mass, dtype=float, device=model.device)

    def coupling_save_iteration_state(self):
        """Save history overwritten by a step before the same step is re-solved."""
        self._reuse_contacts = False
        if self._lagged_material is not None:
            wp.copy(self._lagged_material_saved, self._lagged_material)
        if self._constrained_material:
            wp.copy(self._saved_prediction, self._material_prediction)
        if self.material is not None:
            self.material.coupling_save_iteration_state()

    def coupling_restore_iteration_state(self):
        """Restore the step-start history for another pass over the same step.

        Contacts depend only on step-start poses, so the repeated pass reuses
        the first pass's detection.
        """
        self._reuse_contacts = True
        if self._lagged_material is not None:
            wp.copy(self._lagged_material, self._lagged_material_saved)
        if self._constrained_material:
            wp.copy(self._material_prediction, self._saved_prediction)
        if self.material is not None:
            self.material.coupling_restore_iteration_state()

    def _predict_constrained_material(self, work):
        model = self.model
        wp.copy(self._response_start, work.body_qd)
        wp.launch(
            response.select_bodies,
            dim=model.body_count,
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                model.shape_body,
                model.body_flags,
                model.body_inv_mass,
            ],
            outputs=[self._response_enabled],
            device=model.device,
        )
        wp.launch(
            response.apply_prediction,
            dim=model.body_count,
            inputs=[
                self._response_enabled,
                self._material_prediction,
                work.body_q,
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[work.body_qd],
            device=model.device,
        )

    def _prepare_constrained_material(self, work):
        model = self.model
        wp.launch(
            response.build_response,
            dim=model.body_count,
            inputs=[
                self._response_enabled,
                work.body_q,
                model.body_inv_mass,
                model.body_inv_inertia,
                self.contacts.rigid_contact_count,
                self._contact_bodies,
                self._contact_ra,
                self._contact_rb,
                self._contact_normal,
                self._contact_params,
                self.contact_impulses,
            ],
            outputs=[self._body_response],
            device=model.device,
        )
        wp.launch(
            response.remove_prediction,
            dim=model.body_count,
            inputs=[self._response_enabled, self._body_response, self._material_prediction],
            outputs=[work.body_qd],
            device=model.device,
        )
        wp.copy(self._response_base, work.body_qd)

    def _finish_constrained_material(self, work, dt):
        model = self.model
        wp.launch(
            response.finish_material,
            dim=model.body_count,
            inputs=[self._response_enabled, self._body_response, self._response_base, self.material_impulses],
            outputs=[work.body_qd],
            device=model.device,
        )
        wp.launch(
            response.check_material_velocity,
            dim=model.body_count,
            inputs=[
                self._response_enabled,
                self._response_base,
                self.material._independent_rigidity.velocity_delta,
                work.body_qd,
            ],
            outputs=[self.material_velocity_residual],
            device=model.device,
        )
        wp.copy(self._response_accepted, work.body_qd)
        # Re-solve only the cheap contact problem to recover current-load reactions.
        # Its velocities are diagnostic: the accepted material boundary is retained.
        wp.copy(work.body_qd, self._response_start)
        wp.launch(
            response.apply_prediction,
            dim=model.body_count,
            inputs=[
                self._response_enabled,
                self.material_impulses,
                work.body_q,
                model.body_inv_mass,
                model.body_inv_inertia,
            ],
            outputs=[work.body_qd],
            device=model.device,
        )
        self.rigid_impulses.zero_()
        self._recovering_contacts = True
        try:
            if self._reuse_recovery_contact_geometry():
                # The first contact-space solve saved the warm-start impulses
                # before updating them. Match the original recovery solve's
                # starting point while keeping its unchanged geometry.
                wp.copy(self.contact_impulses, self._contact_space.before_impulses)
            self._solve_rigid_contacts(work, dt)
        finally:
            self._recovering_contacts = False
        wp.launch(
            response.diagnose,
            dim=model.body_count,
            inputs=[
                self._response_enabled,
                work.body_q,
                model.body_mass,
                model.body_inertia,
                self._response_start,
                self._response_accepted,
                work.body_qd,
                self.material_impulses,
                self.rigid_impulses,
            ],
            outputs=[self.momentum_residual, self.constraint_velocity_residual],
            device=model.device,
        )
        wp.copy(work.body_qd, self._response_accepted)
        wp.copy(self._material_prediction, self.material_impulses)

    def check_capacity_status(self):
        """Raise if a captured step exceeded a sparse-grid or particle-basis capacity."""
        if self.material is not None:
            self.material.check_sparse_grid_rebuild_status()
            self.material.check_point_basis_capacity()

    def notify_model_changed(self, flags):
        if self.material is not None:
            self.material.notify_model_changed(flags)
            if flags & ModelFlags.BODY_INERTIAL_PROPERTIES:
                body_mass = self._material_body_mass() if self._lagged_material is not None else None
                self.material.setup_collider(body_q=self.model.body_q, body_mass=body_mass)

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
        if self._previous_impulses is not None and not self._reuse_contacts:
            # A repeated pass over the same step reuses its detection, and so
            # also the previous step's impulses saved by the first pass.
            wp.copy(self._previous_impulses, self.contact_impulses)
        self.rigid_impulses.zero_()
        self.material_impulses.zero_()
        self.contact_impulses.zero_()
        if self._material_stream is not None:
            # Bodies change below; the prepared material terms read only particles.
            stream = wp.get_stream(model.device)
            stream.record_event(self._material_fork)
            self._material_stream.wait_event(self._material_fork)
            with wp.ScopedStream(self._material_stream, sync_enter=False):
                self.material.prepare_step(work, dt)
            self._material_stream.record_event(self._material_join)
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
            if self._lagged_material is not None:
                # Last step's material reaction on free bodies, so that the
                # contact solve below holds against it.
                wp.launch(
                    kernels.apply_material_impulses,
                    dim=model.body_count,
                    inputs=[
                        self._lagged_material,
                        work.body_q,
                        work.body_qd,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.body_flags,
                    ],
                    device=model.device,
                )
            if not self._reuse_contacts:
                self.detector.collide(work, self.contacts)
            self._reuse_contacts = False
            if self._constrained_material:
                self._predict_constrained_material(work)
            self._solve_rigid_contacts(work, dt)
            if self._constrained_material:
                self._prepare_constrained_material(work)
        if self.material is not None:
            if self._material_stream is not None:
                stream.wait_event(self._material_join)
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
                if self._mobility is not None:
                    wp.copy(self._qd_before_contacts, work.body_qd)
                applied = self.material_impulses
                if self._lagged_material is not None:
                    wp.launch(
                        kernels.split_lagged_material_impulses,
                        dim=model.body_count,
                        inputs=[self.material_impulses, model.body_inv_mass, model.body_flags],
                        outputs=[self._immediate_material, self._lagged_material],
                        device=model.device,
                    )
                    applied = self._immediate_material
                wp.launch(
                    kernels.apply_material_impulses,
                    dim=model.body_count,
                    inputs=[
                        applied,
                        work.body_q,
                        work.body_qd,
                        model.body_inv_mass,
                        model.body_inv_inertia,
                        model.body_flags,
                    ],
                    device=model.device,
                )
                if self._mobility is not None:
                    # Links move through the mobility, as inside the material solve.
                    wp.launch(
                        kernels.finish_articulated_contacts,
                        dim=self._link_body.shape,
                        inputs=[self._link_body, self._mobility, self._qd_before_contacts, self.material_impulses],
                        outputs=[work.body_qd],
                        device=model.device,
                    )
        if self._constrained_material:
            self._finish_constrained_material(work, dt)
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
        if self._lagged_material is not None:
            self._lagged_material.zero_()
        if self._constrained_material:
            self._material_prediction.zero_()
        self.detector.reset_contact_matching(world_mask=world_mask)
