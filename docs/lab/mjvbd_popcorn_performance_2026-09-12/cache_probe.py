# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe fixed-geometry caches without changing the production demo defaults."""

import argparse
import cProfile
import json
import math
import time
from pathlib import Path
from unittest import mock

import numpy as np
import warp as wp
from coupled_translation_probe import CoupledTranslationPCG
from dense_single_probe import accumulate_body_particle_contact_dense_single
from face_fixedpoint_probe import optimize_face_fixedpoint
from iteration_loop_probe import install_iteration_loop
from particle_fusion_probe import ParticleFusionAdapter
from payload_trace_probe import install_payload_trace
from prop_render_probe import AppearanceCache, HiddenTriangleCache, PropRenderCache
from rigid_fusion_probe import RigidFusionAdapter, prune_fused_resets
from rigid_ritz_probe import RigidRitz
from table_guard_probe import TableGuardProbe
from two_level_probe import TwoLevelPCG

from newton._src.solvers.mjvbd_v2 import full_contact_pipeline
from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjection, ContactProjectionData, append_contact
from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as pk
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk
from newton.examples.mjvbdv2 import example_mjvbd_v2_popcorn as scene
from newton.viewer import ViewerGL, ViewerNull


@wp.kernel
def _mark_dense(counts: wp.array[int], active: wp.array[int]):
    if counts[wp.tid()] >= 128:
        wp.atomic_max(active, 0, 1)


@wp.kernel
def _audit_coarse(status: wp.array[int], counters: wp.array[int], totals: wp.array[int]):
    wp.atomic_add(totals, 0, 1)
    if status[0] == 0:
        wp.atomic_add(totals, 1, 1)
    wp.atomic_add(totals, 2, counters[0])


@wp.kernel
def _detect_self_contacts(info_array: wp.array[pk.TriMeshCollisionInfo], active: wp.array[int]):
    index = wp.tid()
    info = info_array[0]
    if index < info.edge_colliding_edges_count.shape[0] and info.edge_colliding_edges_count[index] > 0:
        wp.atomic_max(active, 0, 1)
    if index < info.vertex_colliding_triangles_count.shape[0] and info.vertex_colliding_triangles_count[index] > 0:
        wp.atomic_max(active, 0, 1)


@wp.kernel
def _project_soft_contacts(
    dt: float,
    anchor: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    friction_epsilon: float,
    radius: wp.array[float],
    indices: wp.array[wp.vec3i],
    count: wp.array[int],
    capacity: int,
    ke: wp.array[float],
    kd: wp.array[float],
    mu: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shapes: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    margin: wp.array[float],
    barycentric: wp.array[wp.vec3],
    clusters: wp.array[int],
    projection: ContactProjectionData,
):
    for contact in range(wp.tid(), min(count[0], capacity), min(capacity, 4096)):
        corners = indices[contact]
        if corners[1] >= 0:
            bary = barycentric[contact]
            _force, hessian, _point = rk._eval_soft_ef_contact(
                contact,
                corners,
                bary,
                pos,
                anchor,
                radius,
                ke[contact],
                kd[contact],
                mu[contact],
                friction_epsilon,
                shape_body,
                body_q,
                body_prev,
                body_qd,
                body_com,
                shapes,
                body_pos,
                body_vel,
                normals,
                margin,
                dt,
            )
            append_contact(
                wp.vec4i(corners[0], corners[1], corners[2], -1),
                wp.vec4(bary[0], bary[1], bary[2], 0.0),
                hessian,
                clusters,
                projection,
            )


class _SolverFactory:
    def __init__(self, original, surface, dat, cluster_size, checkpoint_interval, contact_history, coarse_iterations):
        self.original = original
        self.surface = surface
        self.dat = dat
        self.cluster_size = cluster_size
        self.checkpoint_interval = checkpoint_interval
        self.contact_history = contact_history
        self.coarse_iterations = coarse_iterations

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, *args, **kwargs):
        options = dict(kwargs.get("vbd_options", {}))
        options.update(particle_enable_surface_cache=self.surface, particle_enable_truncation_cache=self.dat)
        options["particle_multilevel_cluster_size"] = self.cluster_size
        options["rigid_contact_history"] = self.contact_history
        options["particle_multilevel_coarse_iterations"] = self.coarse_iterations
        options["particle_multilevel_checkpoints"] = tuple(
            range(self.checkpoint_interval, options["iterations"] + 1, self.checkpoint_interval)
        )
        return self.original(*args, **{**kwargs, "vbd_options": options})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--dat", action="store_true")
    parser.add_argument("--tiled-coarse", action="store_true")
    parser.add_argument("--dense-graph-gate", action="store_true")
    parser.add_argument("--dense-single", action="store_true")
    parser.add_argument("--rigid-fusion", action="store_true")
    parser.add_argument("--prune-fused-resets", action="store_true")
    parser.add_argument("--iteration-loop", action="store_true")
    parser.add_argument("--gpu-table-guard", action="store_true")
    parser.add_argument("--particle-fusion", action="store_true")
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--substeps", type=int, default=16)
    parser.add_argument("--render", action="store_true", help="Include 1920x1080 OpenGL rendering")
    parser.add_argument("--screenshots-dir", type=Path)
    parser.add_argument("--cache-prop-render", action="store_true")
    parser.add_argument("--cache-appearance", action="store_true")
    parser.add_argument("--cache-hidden-triangles", action="store_true")
    parser.add_argument("--face-fixedpoint", action="store_true")
    parser.add_argument("--bending-assembly", action="store_true")
    parser.add_argument("--compact-ritz", action="store_true")
    parser.add_argument("--ritz-width", type=int, choices=(18, 24, 32), default=24)
    parser.add_argument("--batch-cup-render", action="store_true")
    parser.add_argument("--cup-render-interop", action="store_true")
    parser.add_argument("--cup-gather-normals", action="store_true")
    parser.add_argument("--cache-state-reads", action="store_true")
    parser.add_argument("--surface-block", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--compact-ik", action="store_true")
    parser.add_argument("--ik-shadow", action="store_true")
    parser.add_argument("--serial-ik", action="store_true")
    parser.add_argument("--cpu-profile", type=Path)
    parser.add_argument("--interop", choices=("default", "meshes", "all"), default="default")
    parser.add_argument("--carry-pitch-begin", type=float)
    parser.add_argument("--carry-pitch-end", type=float)
    parser.add_argument("--pour-pitch-deg", type=float)
    parser.add_argument("--payload-trace", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Time one uncaptured diagnostic frame after acceptance")
    parser.add_argument("--contact-history", action="store_true")
    parser.add_argument("--rigid-beta-scale", type=float, default=1.0)
    parser.add_argument("--project-contacts", action="store_true")
    parser.add_argument("--prune-fixed-colors", action="store_true")
    parser.add_argument("--coarse-audit", action="store_true")
    parser.add_argument("--coarse-radius", type=float, default=0.5)
    parser.add_argument("--global-particles", action="store_true")
    parser.add_argument("--ritz6", action="store_true")
    parser.add_argument("--two-level", action="store_true")
    parser.add_argument("--coupled-translation", action="store_true")
    parser.add_argument("--native-solver", action="store_true")
    parser.add_argument("--self-contact-gate", action="store_true")
    parser.add_argument("--cup-segments", type=int, default=40)
    parser.add_argument("--cup-rings", type=int, default=12)
    parser.add_argument("--coarse-iterations", type=int, default=32)
    parser.add_argument("--particle-interval", type=int, default=1)
    parser.add_argument("--cluster-size", type=int, default=400)
    parser.add_argument("--checkpoint-interval", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--fallback-iterations", type=int)
    args = parser.parse_args()
    if args.screenshots_dir is not None and not args.render:
        parser.error("Screenshots require rendering")
    if args.cache_prop_render and not args.render:
        parser.error("Prop render caching requires rendering")
    if args.coupled_translation:
        if not args.rigid_fusion or args.project_contacts:
            parser.error("Coupled translation requires rigid fusion and assembles its own contact cross terms")
        args.two_level = True
    if args.prune_fused_resets and not args.rigid_fusion:
        parser.error("Reset pruning requires the fused rigid overwrite kernel")
    if args.ritz6 or args.two_level:
        args.global_particles = True
    print(json.dumps({"probe_options": vars(args)}, default=str), flush=True)
    if args.frames < 1 or args.warmup < 1:
        parser.error("Frame counts must be positive")
    if args.iterations < 1 or args.cluster_size < 2 or not 1 <= args.checkpoint_interval <= args.iterations:
        parser.error("Invalid iteration or cluster budget")
    original_launch = wp.launch
    dense_active = None
    pending = None
    projection = None
    cluster_ids = None
    fixed_groups = set()
    audit = None
    self_active = None
    truncation_kernel = None
    fusion_adapter = None
    particle_fusion_adapter = None
    ritz = None
    coupled = None

    def launch(*positional, **kwargs):
        nonlocal dense_active, pending, projection, cluster_ids
        kernel = kwargs.get("kernel", positional[0] if positional else None)
        if args.bending_assembly and kernel is ml._assemble_bending_energy_galerkin:
            from bending_assembly_probe import assemble_bending

            kwargs["kernel"] = assemble_bending
            kwargs["dim"] //= 6
            return original_launch(**kwargs)
        if coupled is not None and kernel is ml._commit_prolonged_corrections:
            result = original_launch(*positional, **kwargs)
            coupled.commit()
            if audit is not None:
                values = kwargs["inputs"]
                original_launch(_audit_coarse, dim=1, inputs=[values[5], values[4], audit], device=kwargs["device"])
            return result
        if ritz is not None and kernel is ml._solve_energy_galerkin_pcg_persistent:
            ritz.solve(original_launch, kwargs["inputs"], kwargs["outputs"])
            return None
        if particle_fusion_adapter is not None and (
            kernel in (rk.init_body_particle_contacts, pk.solve_surface_elasticity_tile)
            or (kernel is pk.accumulate_particle_body_contact_force_and_hessian and kwargs["inputs"][1] >= 0)
        ):
            return particle_fusion_adapter(*positional, **kwargs)
        if fusion_adapter is not None and kernel in (
            rk.accumulate_body_particle_contacts_per_body,
            rk.accumulate_body_body_contacts_per_body,
            rk.accumulate_body_particle_contact_dense_partials,
            rk.accumulate_body_particle_contact_dense_reduction,
            rk.accumulate_body_particle_contact_dense_single,
            rk.solve_rigid_body,
        ):
            return fusion_adapter(*positional, **kwargs)
        if (
            self_active is not None
            and self_active.device.is_capturing
            and kernel
            in (
                pk.accumulate_self_contact_force_and_hessian,
                truncation_kernel,
            )
        ):
            wp.capture_if(self_active, on_true=lambda: original_launch(*positional, **kwargs))
            return None
        if args.dense_single:
            if kernel is rk.accumulate_body_particle_contact_dense_partials:
                pending = dict(kwargs)
                return None
            if kernel is rk.accumulate_body_particle_contact_dense_reduction:
                values = pending["inputs"]
                result = original_launch(
                    accumulate_body_particle_contact_dense_single,
                    dim=values[1].size * rk._BODY_PARTICLE_CONTACT_BLOCK_DIM,
                    block_dim=rk._BODY_PARTICLE_CONTACT_BLOCK_DIM,
                    inputs=values,
                    outputs=kwargs["outputs"],
                    device=kwargs["device"],
                )
                pending = None
                return result
        if args.coarse_audit and kernel is ml._commit_prolonged_corrections:
            result = original_launch(*positional, **kwargs)
            values = kwargs["inputs"]
            original_launch(_audit_coarse, dim=1, inputs=[values[5], values[4], audit], device=kwargs["device"])
            return result
        if args.prune_fixed_colors and kernel in (
            rk.accumulate_body_particle_contacts_per_body,
            rk.accumulate_body_particle_contact_dense_partials,
            rk.accumulate_body_particle_contact_dense_reduction,
            rk.accumulate_body_body_contacts_per_body,
            rk.solve_rigid_body,
        ):
            group_index = 0 if kernel is rk.accumulate_body_particle_contact_dense_reduction else 1
            if kwargs["inputs"][group_index].ptr in fixed_groups:
                return None
        if args.project_contacts and kernel is pk.accumulate_particle_body_contact_force_and_hessian:
            values = kwargs["inputs"]
            if values[1] == -1 and projection is not None:
                projection.reset()
                result = original_launch(*positional, **kwargs)
                selected = [
                    values[i]
                    for i in (0, 2, 3, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
                ]
                original_launch(
                    _project_soft_contacts,
                    dim=kwargs["dim"],
                    inputs=[*selected, cluster_ids, projection.data],
                    device=kwargs["device"],
                )
                return result
        if args.dense_graph_gate:
            if kernel is rk.build_body_particle_contact_lists:
                result = original_launch(*positional, **kwargs)
                counts = kwargs["outputs"][0]
                if dense_active is None:
                    dense_active = wp.zeros(1, dtype=int, device=counts.device)
                dense_active.zero_()
                original_launch(
                    _mark_dense, dim=counts.size, inputs=[counts], outputs=[dense_active], device=counts.device
                )
                return result
            if dense_active is not None and dense_active.device.is_capturing:
                if kernel is rk.accumulate_body_particle_contact_dense_partials:
                    pending = (positional, dict(kwargs))
                    return None
                if kernel is rk.accumulate_body_particle_contact_dense_reduction:
                    partial_args, partial_kwargs = pending

                    def run_dense():
                        original_launch(*partial_args, **partial_kwargs)
                        original_launch(*positional, **kwargs)

                    wp.capture_if(dense_active, on_true=run_dense)
                    pending = None
                    return None
        if kernel in (ml._restrict_energy_galerkin, ml._restrict_energy_galerkin_tiled):
            clusters = kwargs["dim"] // (64 if kernel is ml._restrict_energy_galerkin_tiled else 1)
            target = ml._restrict_energy_galerkin_tiled if args.tiled_coarse else ml._restrict_energy_galerkin
            if positional:
                positional = (target, *positional[1:])
            else:
                kwargs["kernel"] = target
            kwargs["dim"] = clusters * (64 if args.tiled_coarse else 1)
            kwargs["block_dim"] = 64 if args.tiled_coarse else 256
        return original_launch(*positional, **kwargs)

    original_clusters = ml._build_clusters
    original_cup = scene.cup_mesh
    original_load = json.load

    def cup_mesh():
        return original_cup(args.cup_segments, args.cup_rings)

    def load_assets(stream, *positional, **kwargs):
        assets = original_load(stream, *positional, **kwargs)
        if args.cup_segments == 40 and args.cup_rings == 12:
            return assets
        if isinstance(assets, dict) and assets.get("version") == 1 and "meshes" in assets:
            for asset in assets["meshes"]:
                if asset.get("group") == "cup":
                    old = np.asarray(asset["vertices"])[np.asarray(asset["faces"])].mean(axis=1)
                    points, faces = cup_mesh()
                    centers = points[faces].mean(axis=1)
                    nearest = ((centers[:, None] - old[None, :]) ** 2).sum(axis=2).argmin(axis=1)
                    asset["material_indices"] = np.asarray(asset["material_indices"])[nearest].tolist()
                    asset["vertices"], asset["faces"] = points.tolist(), faces.tolist()
        return assets

    def build_clusters(model, target_size):
        return original_clusters(model, 1 if args.global_particles else target_size)

    with (
        mock.patch.object(wp, "launch", launch),
        mock.patch.object(ml, "_build_clusters", build_clusters),
        mock.patch.object(scene, "cup_mesh", cup_mesh),
        mock.patch.object(json, "load", load_assets),
    ):

        def configure(solver):
            nonlocal ritz, coupled
            nonlocal \
                projection, \
                cluster_ids, \
                audit, \
                self_active, \
                truncation_kernel, \
                fusion_adapter, \
                particle_fusion_adapter
            solver.particle_multilevel.max_radius_fraction = args.coarse_radius
            if args.fallback_iterations is not None:
                if args.fallback_iterations < solver.iterations:
                    raise ValueError("Fallback budget cannot be smaller than ordinary iterations")
                solver.particle_multilevel_fallback_iterations = args.fallback_iterations
            if args.checkpoints is not None:
                if any(value < 1 or value > solver.iterations for value in args.checkpoints):
                    raise ValueError("Checkpoints must be within the local iteration budget")
                solver.particle_multilevel_checkpoints = tuple(sorted(set(args.checkpoints)))
            if not np.isfinite(args.rigid_beta_scale) or args.rigid_beta_scale <= 0:
                raise ValueError("Rigid continuation multiplier must be positive and finite")
            solver.rigid_contact_beta *= args.rigid_beta_scale
            if args.native_solver:
                return
            if args.rigid_fusion:
                fusion_adapter = RigidFusionAdapter(original_launch, solver.model)
                if args.prune_fused_resets:
                    prune_fused_resets(solver)
            if args.particle_fusion:
                particle_fusion_adapter = ParticleFusionAdapter(original_launch, solver.model)
            if args.ritz6 or args.two_level:
                correction = solver.particle_multilevel
                ritz = RigidRitz(solver.model, correction, args.cluster_size)
                if args.two_level:
                    correction.coarse_use_split_pcg = True
                    correction._split_coarse_pcg = TwoLevelPCG(ritz)
                if args.coupled_translation:
                    coupled = CoupledTranslationPCG(solver, correction, ritz, fusion_adapter)
                    correction._split_coarse_pcg = coupled
                original_restrict = correction.restrict_and_prolong

                def restrict(model, particle_q, particle_displacements, dt):
                    ritz.q = particle_q
                    original_restrict(model, particle_q, particle_displacements, dt)

                correction.restrict_and_prolong = restrict
            if args.self_contact_gate:
                cache = solver._particle_truncation_cache
                if cache is None:
                    raise ValueError("Self-contact gate probe requires DAT cache")
                self_active = wp.zeros(1, dtype=int, device=solver.device)
                cache._active = self_active
                truncation_kernel = cache._truncate
                original_rebuild = cache.rebuild

                def rebuild(owner):
                    self_active.zero_()
                    original_launch(
                        _detect_self_contacts,
                        dim=max(owner.model.particle_count, owner.model.edge_count),
                        inputs=[owner.trimesh_collision_info],
                        outputs=[self_active],
                        device=owner.device,
                    )
                    original_rebuild(owner)

                cache.rebuild = rebuild
            if args.coarse_audit:
                audit = wp.zeros(3, dtype=int, device=solver.device)
            if args.prune_fixed_colors:
                inverse_mass = solver.body_inv_mass_effective.numpy()
                sizes = []
                for group in solver.model.body_color_groups:
                    dynamic = int((inverse_mass[group.numpy()] > 0).sum())
                    sizes.append((group.size, dynamic))
                    if dynamic == 0:
                        fixed_groups.add(group.ptr)
                print(f"Rigid color groups (total, movable): {sizes}", flush=True)
            if args.project_contacts:
                correction = solver.particle_multilevel
                projection = ContactProjection(correction.cluster_count, 1000000, solver.device)
                cluster_ids = correction.fine_to_coarse
                correction.contact_projection = projection

        try:
            run_probe(args, configure)
        finally:
            if audit is not None:
                print(f"Coarse totals (attempted, accepted, clamped particles): {audit.numpy().tolist()}", flush=True)


def run_probe(args, configure):
    full_contact_pipeline._COMPACT_SOFT_SURFACE_BLOCK_DIM = args.surface_block
    # Preserve the same global worker count and grid-stride contact traversal.
    full_contact_pipeline._COMPACT_SOFT_SURFACE_BLOCKS_PER_SM = 256 // args.surface_block
    if args.compact_ritz:
        import two_level_probe

        two_level_probe.RITZ_WIDTH = wp.constant(args.ritz_width)
        two_level_probe.Vec32 = wp.types.vector(length=args.ritz_width, dtype=wp.float32)
    if args.pour_pitch_deg is not None:
        if not 35.0 <= args.pour_pitch_deg <= 45.0:
            raise ValueError("Pour-angle diagnostic is limited to 35--45 degrees")
        scene.POUR_PITCH = math.radians(args.pour_pitch_deg)
    if args.face_fixedpoint:
        full_contact_pipeline.optimize_face_sdf = optimize_face_fixedpoint
    if args.carry_pitch_begin is not None or args.carry_pitch_end is not None:
        if args.carry_pitch_begin is None or args.carry_pitch_end is None:
            raise ValueError("Specify both carry-pitch times for a trajectory diagnostic")
        if not 10.0 <= args.carry_pitch_begin < args.carry_pitch_end <= 12.5:
            raise ValueError("Carry-pitch diagnostic must remain within the scoop/lift/retract phase")
        scene.PITCH_BEGIN, scene.PITCH_END = args.carry_pitch_begin, args.carry_pitch_end
    factory = _SolverFactory(
        scene.SolverMJVBDV2,
        args.surface,
        args.dat,
        args.cluster_size,
        args.checkpoint_interval,
        args.contact_history,
        args.coarse_iterations,
    )
    if args.native_solver:
        original_factory = factory

        def factory(*positional, **kwargs):
            options = dict(kwargs.get("vbd_options", {}))
            options["particle_enable_coupled_translation"] = True
            return original_factory(*positional, **{**kwargs, "vbd_options": options})

        factory.register_custom_attributes = original_factory.register_custom_attributes
    with mock.patch.object(scene, "SolverMJVBDV2", factory):
        options = scene.Example.create_parser().parse_args(
            ["--vbd-iterations", str(args.iterations), "--substeps", str(args.substeps)]
        )
        interop = {
            "default": ViewerGL.CudaInterop.DYNAMIC_MESH,
            "meshes": ViewerGL.CudaInterop.DYNAMIC_MESH | ViewerGL.CudaInterop.STATIC_MESH,
            "all": ViewerGL.CudaInterop.ALL,
        }[args.interop]
        viewer = (
            ViewerGL(width=1920, height=1080, headless=True, vsync=False, enable_cuda_interop=interop)
            if args.render
            else ViewerNull()
        )
        example = scene.Example(viewer, options)
    solver = example.solver.vbd_solver
    render_cache = PropRenderCache(example) if args.cache_prop_render else None
    appearance_cache = AppearanceCache(viewer) if args.cache_appearance else None
    hidden_cache = HiddenTriangleCache(viewer) if args.cache_hidden_triangles else None
    cup_cache = None
    if args.batch_cup_render:
        from cup_render_probe import CupRenderBatch

        cup_cache = CupRenderBatch(viewer, shared_vbo=args.cup_render_interop, gather_normals=args.cup_gather_normals)
    if args.payload_trace:
        install_payload_trace(example)
    if args.screenshots_dir is not None:
        args.screenshots_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_table_guard:
        example._check_robot_table = TableGuardProbe(example)
    configure(solver)
    if args.compact_ik:
        from ik_compaction_probe import install_compact_ik

        install_compact_ik(example, scene.lock_joints, shadow=args.ik_shadow, serial_objectives=args.serial_ik)
    if args.cache_state_reads:
        from state_read_probe import install_state_read_cache

        install_state_read_cache(example)
    if args.iteration_loop:
        if args.particle_interval != 1:
            raise ValueError("Loop diagnostic requires the normal particle cadence")
        install_iteration_loop(solver, args.checkpoint_interval)
    particle_iteration = solver._solve_particle_iteration

    def solve_particle_iteration(state_in, state_out, control, contacts, dt, iteration):
        if (iteration + 1) % args.particle_interval == 0 or iteration == solver.iterations - 1:
            return particle_iteration(state_in, state_out, control, contacts, dt, iteration)

    if args.particle_interval < 1:
        raise ValueError("particle interval must be positive")
    solver._solve_particle_iteration = solve_particle_iteration
    assert (solver._surface_cached_kernel is not None) == args.surface
    assert (solver._particle_truncation_cache is not None) == args.dat
    for _ in range(args.warmup):
        example.step()
        if args.render:
            example.render()
    if args.render:
        from pyglet import gl  # noqa: PLC0415 - OpenGL is optional for headless physics probes.

        gl.glFinish()
        if render_cache is not None:
            render_cache.enabled = False
            if cup_cache is not None:
                cup_cache.enabled = False
            if hidden_cache is not None:
                hidden_cache.enabled = False
            if appearance_cache is not None:
                appearance_cache.enabled = False
            example.render()
            original_pixels = viewer.get_frame().numpy()
            render_cache.enabled = True
            if cup_cache is not None:
                cup_cache.enabled = True
            if hidden_cache is not None:
                hidden_cache.enabled = True
            if appearance_cache is not None:
                appearance_cache.enabled = True
            example.render()
            cached_pixels = viewer.get_frame().numpy()
            difference = np.abs(original_pixels.astype(np.int16) - cached_pixels.astype(np.int16))
            changed = int(np.count_nonzero(difference))
            print(
                json.dumps(
                    {"prop_cache_pixel_max_difference": int(difference.max()), "prop_cache_changed_channels": changed}
                ),
                flush=True,
            )
            if changed:
                # Atomic normal accumulation can change the final 8-bit rounding
                # of isolated channels even when repeatedly rendering one state.
                render_cache.enabled = False
                if cup_cache is not None:
                    cup_cache.enabled = False
                if hidden_cache is not None:
                    hidden_cache.enabled = False
                if appearance_cache is not None:
                    appearance_cache.enabled = False
                example.render()
                repeated_pixels = viewer.get_frame().numpy()
                repeated = np.abs(original_pixels.astype(np.int16) - repeated_pixels.astype(np.int16))
                print(
                    json.dumps(
                        {
                            "uncached_repeat_pixel_max_difference": int(repeated.max()),
                            "uncached_repeat_changed_channels": int(np.count_nonzero(repeated)),
                        }
                    ),
                    flush=True,
                )
                render_cache.enabled = True
                if cup_cache is not None:
                    cup_cache.enabled = True
                if hidden_cache is not None:
                    hidden_cache.enabled = True
                if appearance_cache is not None:
                    appearance_cache.enabled = True
            if int(difference.max()) > 1 or changed > max(1, original_pixels.size // 100000):
                raise AssertionError("Render cache changed more than isolated 8-bit normal-rounding noise")
    wp.synchronize_device(example.model.device)
    start = time.perf_counter()
    cpu_profile = cProfile.Profile() if args.cpu_profile is not None else None
    if cpu_profile is not None:
        cpu_profile.enable()
    for frame in range(args.frames):
        try:
            example.step()
        except (AssertionError, RuntimeError):
            print(
                json.dumps(
                    {
                        "failure_time": example.sim_time,
                        "trajectory_time": example.trajectory_time,
                        "coarse_status": solver.particle_multilevel.runtime_status.numpy().tolist(),
                        "coarse_metrics": solver.particle_multilevel.runtime_metrics.numpy().tolist(),
                        "failed_probe_wall_ms_per_frame": 1000 * (time.perf_counter() - start) / (frame + 1),
                        "grasp_forces": example.grasp_force_filtered.tolist(),
                        "grasp_offsets_deg": np.degrees(example.grasp_joint_offset).tolist(),
                        "particle_bounds": [
                            example.state_0.particle_q.numpy().min(axis=0).tolist(),
                            example.state_0.particle_q.numpy().max(axis=0).tolist(),
                        ],
                    }
                ),
                flush=True,
            )
            raise
        if args.render:
            example.render()
            if args.screenshots_dir is not None and int(round(example.sim_time * 60)) in (900, 1080, 1260, 1620):
                from PIL import Image  # noqa: PLC0415 - Optional diagnostic image output.

                Image.fromarray(viewer.get_frame().numpy()).save(
                    args.screenshots_dir / f"frame-{example.sim_time:.1f}.png"
                )
        if args.validate and (frame + 1) % 600 == 0:
            elapsed_ms = 1000 * (time.perf_counter() - start) / (frame + 1)
            print(f"Validation progress: {example.sim_time:.2f} s; partial wall {elapsed_ms:.3f} ms/frame", flush=True)
            metrics = solver.particle_multilevel.runtime_metrics.numpy()
            print(
                json.dumps(
                    {
                        "coarse_status": solver.particle_multilevel.runtime_status.numpy().tolist(),
                        "coarse_relative_residual": float(np.sqrt(max(metrics[1], 0) / max(metrics[0], 1e-30))),
                        "trajectory_time": example.trajectory_time,
                        "max_scoop_count": example.max_scoop_count,
                        "loaded_lift_count": example.loaded_lift_count,
                        "delivered_inside": example.delivered_inside,
                        "cup_current_lift_m": example.current_cup_lift,
                        "cup_grip_slip_m": getattr(example, "cup_grip_slip", None),
                        "finger_contact_force_N": example.grasp_force_filtered.tolist(),
                        "grip_offsets_deg": np.degrees(example.grasp_joint_offset).tolist(),
                    }
                ),
                flush=True,
            )
    wp.synchronize_device(example.model.device)
    if args.render:
        gl.glFinish()
    if cpu_profile is not None:
        cpu_profile.disable()
        cpu_profile.dump_stats(str(args.cpu_profile))
    print(
        json.dumps(
            {
                "surface_cache": args.surface,
                "dat_cache": args.dat,
                "tiled_coarse": args.tiled_coarse,
                "dense_graph_gate": args.dense_graph_gate,
                "frames": args.frames,
                "iterations": args.iterations,
                "substeps": args.substeps,
                "rendering": args.render,
                "contact_history": args.contact_history,
                "particle_interval": args.particle_interval,
                "cluster_size": args.cluster_size,
                "checkpoint_interval": args.checkpoint_interval,
                "wall_ms_per_frame": 1000 * (time.perf_counter() - start) / args.frames,
            }
        ),
        flush=True,
    )
    if args.validate:
        example.test_final()
    if args.profile:
        # Separate from trajectory timing: per-launch events perturb scheduling.
        wp.timing_begin(wp.TIMING_ALL)
        example._simulate()
        records = wp.timing_end()
        totals = {}
        for record in records:
            total, count = totals.get(record.name, (0.0, 0))
            totals[record.name] = (total + record.elapsed, count + 1)
        print(
            json.dumps({"uncaptured_kernel_profile_ms": sorted(totals.items(), key=lambda item: -item[1][0])}),
            flush=True,
        )


if __name__ == "__main__":
    main()
