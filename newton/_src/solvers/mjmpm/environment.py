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

    def __init__(
        self, model, *, mpm_options=None, rigid_iterations=20, rigid_contact_hard=True, rigid_warm_start=False
    ):
        super().__init__(model)
        self.rigid_iterations = rigid_iterations
        self.rigid_contact_hard = rigid_contact_hard
        self.rigid_warm_start = rigid_warm_start
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
        self.rigid_impulses = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        self.material_impulses = wp.zeros_like(self.rigid_impulses)
        self.work = model.state()
        self._all_worlds = wp.full(model.world_count + 1, True, dtype=wp.bool, device=model.device)

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
        if self._mobility is not None:
            wp.copy(self._qd_before_contacts, work.body_qd)
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
                self._link_slot,
                self._link_body,
                self._mobility,
                self._contacted_links,
                self._contacted_count,
                work.body_qd,
                self.contact_impulses,
                self.rigid_impulses,
                self.rigid_iterations,
            ],
            device=model.device,
        )
        if self._mobility is not None:
            wp.launch(
                kernels.finish_articulated_contacts,
                dim=self._link_body.shape,
                inputs=[self._link_body, self._mobility, self._qd_before_contacts, self.rigid_impulses],
                outputs=[work.body_qd],
                device=model.device,
            )

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

    def coupling_save_iteration_state(self):
        """Save history overwritten by a step before the same step is re-solved."""
        self._reuse_contacts = False
        if self.material is not None:
            self.material.coupling_save_iteration_state()

    def coupling_restore_iteration_state(self):
        """Restore the step-start history for another pass over the same step.

        Contacts depend only on step-start poses, so the repeated pass reuses
        the first pass's detection.
        """
        self._reuse_contacts = True
        if self.material is not None:
            self.material.coupling_restore_iteration_state()

    def check_capacity_status(self):
        """Raise if a captured step exceeded a sparse-grid or particle-basis capacity."""
        if self.material is not None:
            self.material.check_sparse_grid_rebuild_status()
            self.material.check_point_basis_capacity()

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
        if self._previous_impulses is not None and not self._reuse_contacts:
            # A repeated pass over the same step reuses its detection, and so
            # also the previous step's impulses saved by the first pass.
            wp.copy(self._previous_impulses, self.contact_impulses)
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
            if not self._reuse_contacts:
                self.detector.collide(work, self.contacts)
            self._reuse_contacts = False
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
                if self._mobility is not None:
                    wp.copy(self._qd_before_contacts, work.body_qd)
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
                if self._mobility is not None:
                    # Links move through the mobility, as inside the material solve.
                    wp.launch(
                        kernels.finish_articulated_contacts,
                        dim=self._link_body.shape,
                        inputs=[self._link_body, self._mobility, self._qd_before_contacts, self.material_impulses],
                        outputs=[work.body_qd],
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
