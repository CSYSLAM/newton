# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Separate coarse direction, step-length and approximation-space errors.

This synchronized diagnostic is not a benchmark or a simulation policy.
Reference-error fits are offline upper bounds, never applied to the solver.
"""

import argparse
import json
import time
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2 import contact_projection
from newton._src.solvers.mjvbd_v2 import particle_multilevel as multilevel
from newton._src.solvers.mjvbd_v2.particle_multilevel import ParticleMultilevelCorrection
from newton.examples.mjvbdv2 import example_mjvbd_v2_cloth_twist as twist
from newton.examples.mjvbdv2 import example_mjvbd_v2_tshirt_fold as tshirt
from scripts.benchmark_mjvbd_v2_coarse_rows import benchmark_rows
from scripts.diagnose_mjvbd_v2_convergence import ProbeComplete, rms


def emit(**values):
    print(json.dumps(values), flush=True)


@wp.kernel(enable_backward=False)
def _matrix_product(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    blocks: wp.array[wp.mat33],
    contacts: contact_projection.ContactProjectionData,
    vector: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    row = wp.tid()
    value = wp.vec3(0.0)
    for slot in range(offsets[row], offsets[row + 1]):
        value += blocks[slot] * vector[columns[slot]]
    result[row] = value + contact_projection.off_diagonal_product(row, vector, contacts)


def ritz_step(directions, products, force):
    """Minimize the frozen quadratic in current-operator directions, not iterate history."""
    scale = np.array([max(np.linalg.norm(d), 1e-30) for d in directions])
    basis = np.array(directions) / scale[:, None, None]
    images = np.array(products) / scale[:, None, None]
    reduced = np.einsum("aij,bij->ab", basis, images)
    symmetric = (reduced + reduced.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if eigenvalues[0] <= 1e-8 * max(eigenvalues[-1], 1e-30):
        raise ValueError("Dependent or nonpositive Ritz subspace")
    rhs = np.einsum("aij,ij->a", basis, force)
    weights = np.linalg.solve(symmetric, rhs)
    return np.einsum("a,aij->ij", weights, basis), weights / scale, eigenvalues


def local_residual_step_bound(local, preconditioned_product):
    """Bound alpha in [0, 1] so the frozen local residual norm cannot grow."""
    dot = float(np.sum(local * preconditioned_product))
    norm_sq = float(np.sum(preconditioned_product**2))
    if not np.isfinite(dot) or not np.isfinite(norm_sq):
        return 0.0
    if norm_sq <= 1e-30:
        return 1.0
    return min(1.0, max(0.0, 2.0 * dot / norm_sq))


def graph_milliseconds(function, device):
    """Time repeated same-input GPU work, excluding allocation and compilation."""
    function()
    with wp.ScopedCapture(device=device) as capture:
        function()
    for _ in range(5):
        wp.capture_launch(capture.graph)
    timings = []
    for _ in range(5):
        wp.synchronize_device(device)
        start = time.perf_counter()
        for _ in range(50):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        timings.append((time.perf_counter() - start) * 1000.0 / 50)
    return float(np.median(timings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("tshirt", "twist"), default="tshirt")
    parser.add_argument("--warmup-frames", type=int, default=900)
    parser.add_argument("--enrichment", action="store_true", help="Probe operator-based coarse Krylov enrichment")
    parser.add_argument(
        "--profile-contact-rows", action="store_true", help="Compare linked and packed PCG contact rows"
    )
    parser.add_argument(
        "--profile-enrichment", action="store_true", help="Measure frozen GPU stages (requires enrichment)"
    )
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    if args.profile_enrichment and not args.enrichment:
        parser.error("--profile-enrichment requires --enrichment")
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    scene = tshirt if args.scene == "tshirt" else twist
    scene_args = newton.examples.default_args(tshirt.Example.create_parser()) if scene is tshirt else None
    example = scene.Example(newton.viewer.ViewerNull(), scene_args)
    for _ in range(args.warmup_frames):
        example.step()
    backend = example.solver.vbd_solver
    if backend.particle_multilevel is not None or backend.particle_collision_detection_interval != -1:
        raise RuntimeError("Requires coarse disabled and frozen per-substep contact candidates")
    iteration = backend._solve_particle_iteration
    active = (example.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
    active &= example.model.particle_mass.numpy() > 0
    radius = example.model.particle_radius.numpy()
    operators = {}
    for size, coupled, pcg in ((8, False, 8), (8, True, 8), (8, True, 32), (32, True, 32)):
        correction = ParticleMultilevelCorrection(
            example.model,
            operator="galerkin",
            cluster_size=size,
            coarse_iterations=pcg,
            coupling=0.5,
            minimum_residual_reduction=1e-4,
            relaxation=1.0,
            max_radius_fraction=0.05,
            max_clamp_fraction=1.0,
        )
        correction.contact_projection_enabled = coupled
        if args.profile_enrichment:
            # Keep historical stage profiles on their original persistent path;
            # --profile-contact-rows explicitly compares both implementations.
            correction.coarse_use_split_pcg = False
        operators[f"cluster{size}_{'coupled' if coupled else 'uncoupled'}_pcg{pcg}"] = correction

    def probe(state_in, state_out, contacts, dt, iter_num):
        if iter_num != 0:
            raise RuntimeError("Expected first substep iteration")
        attrs = {
            name: getattr(backend, name)
            for name in (
                "particle_enable_batched_jacobi",
                "particle_chebyshev_enabled",
                "_particle_truncation_cache",
                "_surface_cached_kernel",
                "surface_anchor_angles",
                "particle_multilevel",
                "particle_multilevel_checkpoints",
            )
        }

        def snapshot():
            return [
                wp.clone(array)
                for array in (
                    state_in.particle_q,
                    backend.particle_displacements,
                    backend.particle_chebyshev_older,
                    backend.particle_chebyshev_collided,
                    backend.truncation_ts,
                )
            ]

        def restore(saved):
            for array, original in zip(
                (
                    state_in.particle_q,
                    backend.particle_displacements,
                    backend.particle_chebyshev_older,
                    backend.particle_chebyshev_collided,
                    backend.truncation_ts,
                ),
                saved,
                strict=True,
            ):
                array.assign(original)
            if backend._particle_truncation_cache is not None:
                backend._particle_truncation_cache.rebuild(backend)

        def ordinary():
            backend.particle_enable_batched_jacobi = False
            backend.particle_chebyshev_enabled = False
            backend._particle_truncation_cache = None
            backend._surface_cached_kernel = None
            backend.surface_anchor_angles = None

        def assemble(correction):
            # Exercise the actual fine assembly, but do not solve/prolong/move.
            backend.particle_multilevel = correction
            with (
                patch.object(correction, "restrict_and_prolong", lambda *a: None),
                patch.object(backend, "_penetration_free_truncation", lambda *a: None),
            ):
                backend._apply_particle_multilevel_correction(
                    state_in, contacts, state_out.body_q, backend._external_body_q_prev, state_out.body_qd, dt
                )
            local = correction.local_correction.numpy().astype(np.float64)
            diagonal = correction.local_hessians.numpy().astype(np.float64)
            # Reconstructed force, not an independent exact gradient: H * (H^-1 f).
            force = np.einsum("nij,nj->ni", diagonal, local)
            return local, force

        initial = snapshot()
        try:
            ordinary()
            refs = {}
            for i in range(60):
                iteration(state_in, state_out, contacts, dt, i)
                if i + 1 in (30, 60):
                    refs[i + 1] = state_in.particle_q.numpy().astype(np.float64)
            for name, value in attrs.items():
                setattr(backend, name, value)
            restore(initial)
            pre_sweeps = max(1, backend.iterations - 2)
            for i in range(pre_sweeps):
                iteration(state_in, state_out, contacts, dt, i)
            frozen = snapshot()
            q0 = frozen[0].numpy().astype(np.float64)
            delta0 = frozen[1].numpy()
            error = refs[60] - q0
            emit(
                scene=args.scene,
                kind="reference",
                pre_sweeps=pre_sweeps,
                error30=rms((refs[30] - q0)[active]),
                error60=rms(error[active]),
                reference30_to60=rms((refs[60] - refs[30])[active]),
            )
            for name, correction in operators.items():
                restore(frozen)
                local0, force0 = assemble(correction)
                backend.particle_multilevel_checkpoints = ()
                with patch.object(backend, "_penetration_free_truncation", lambda *a: None):
                    correction.restrict_and_prolong(
                        example.model, state_in.particle_q, backend.particle_displacements, dt
                    )
                status = int(correction.runtime_status.numpy()[0])
                mapping = correction.fine_to_coarse.numpy()
                direction = np.zeros_like(q0)
                direction[active] = correction.coarse_solution.numpy()[mapping[active]]
                projection = correction.contact_projection
                emit(
                    kind="direction",
                    case=name,
                    status=status,
                    overflow=int(projection.data.overflow.numpy()[0]) if projection is not None else 0,
                    coarse_metrics=correction.runtime_metrics.numpy().tolist(),
                    direction_rms=rms(direction[active]),
                    local_newton_rms=rms(local0[active]),
                    reconstructed_force_rms=rms(force0[active]),
                    force_dot_direction=float(np.sum(force0 * direction)),
                    oracle_alpha=float(np.sum(error * direction) / max(np.sum(direction**2), 1e-30)),
                )
                # Best fit of the reference error measures space expressiveness
                # independently of PCG and step length. Never use it as a step.
                fit_constant = np.zeros_like(q0)
                fit_affine = np.zeros_like(q0)
                for cluster in range(correction.cluster_count):
                    ids = np.flatnonzero(mapping == cluster)
                    positions = q0[ids] - np.mean(q0[ids], axis=0)
                    scale = max(float(np.max(np.linalg.norm(positions, axis=1))), 1e-12)
                    basis = np.column_stack((np.ones(len(ids)), positions / scale))
                    fit_constant[ids] = np.mean(error[ids], axis=0)
                    fit_affine[ids] = basis @ np.linalg.lstsq(basis, error[ids], rcond=1e-5)[0]
                emit(
                    kind="space_bound",
                    case=name,
                    constant_error=rms((error - fit_constant)[active]),
                    affine_error=rms((error - fit_affine)[active]),
                )
                if status:
                    continue
                if args.profile_contact_rows and name == "cluster8_coupled_pcg8":
                    emit(
                        kind="contact_row_timings", **benchmark_rows(correction, example.model, state_in.particle_q, dt)
                    )
                candidates = [(f"alpha{alpha}", alpha, alpha * direction) for alpha in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
                if args.enrichment and name == "cluster8_coupled_pcg8":
                    restore(frozen)
                    # A singleton Galerkin projection exposes the fine matrix.
                    # Only this offline builder is overridden; do not relax
                    # production's minimum cluster-size validation.
                    singleton = multilevel._build_clusters(example.model, 1)
                    with patch.object(multilevel, "_build_clusters", return_value=singleton):
                        fine = ParticleMultilevelCorrection(
                            example.model,
                            operator="galerkin",
                            cluster_size=2,
                            coarse_iterations=1,
                            relaxation=0.1,
                            max_radius_fraction=0.05,
                            max_clamp_fraction=1.0,
                            coupling=0.5,
                            minimum_residual_reduction=0.0,
                        )
                    fine.contact_projection_enabled = True
                    fine.coarse_use_split_pcg = False
                    assemble(fine)
                    fine.restrict_and_prolong(example.model, state_in.particle_q, wp.zeros_like(frozen[1]), dt)
                    if fine.contact_projection is not None and int(fine.contact_projection.data.overflow.numpy()[0]):
                        raise RuntimeError("Cannot use incomplete fine contact operator for Ritz experiment")
                    fine_ids = fine.cluster_particles.numpy()
                    if fine.cluster_count != len(fine_ids):
                        raise RuntimeError("Expected singleton fine operator")
                    data = (
                        fine.contact_projection.data
                        if fine.contact_projection
                        else contact_projection.ContactProjectionData()
                    )
                    if args.profile_enrichment:
                        v = wp.array(direction[fine_ids], dtype=wp.vec3, device=example.model.device)
                        output = wp.empty_like(v)
                        multiply_ms = graph_milliseconds(
                            lambda fine=fine, data=data, v=v, output=output: wp.launch(
                                _matrix_product,
                                dim=fine.cluster_count,
                                inputs=[
                                    fine.coarse_matrix_offsets,
                                    fine.coarse_matrix_columns,
                                    fine.coarse_matrix_blocks,
                                    data,
                                    v,
                                ],
                                outputs=[output],
                                device=example.model.device,
                            ),
                            example.model.device,
                        )
                        real_launch = wp.launch

                        def assembly_launch(kernel, *launch_args, real_launch=real_launch, **launch_kwargs):
                            if kernel not in (
                                multilevel._solve_energy_galerkin_pcg_persistent,
                                multilevel._prepare_prolonged_corrections,
                                multilevel._commit_prolonged_corrections,
                                contact_projection.reject_overflow,
                            ):
                                return real_launch(kernel, *launch_args, **launch_kwargs)
                            return None

                        def apply_current():
                            backend._apply_particle_multilevel_correction(
                                state_in,
                                contacts,
                                state_out.body_q,
                                backend._external_body_q_prev,
                                state_out.body_qd,
                                dt,
                            )

                        backend.particle_multilevel = fine
                        with (
                            patch.object(wp, "launch", assembly_launch),
                            patch.object(backend, "_penetration_free_truncation", lambda *a: None),
                        ):
                            fine_assembly_ms = graph_milliseconds(apply_current, example.model.device)
                        backend.particle_multilevel = correction
                        with (
                            patch.object(wp, "launch", assembly_launch),
                            patch.object(backend, "_penetration_free_truncation", lambda *a: None),
                        ):
                            coarse_assembly_ms = graph_milliseconds(apply_current, example.model.device)

                        def coarse_pass():
                            restore(frozen)
                            apply_current()

                        coarse_ms = graph_milliseconds(coarse_pass, example.model.device)
                        pcg_calls = []

                        def record_pcg(kernel, *launch_args, calls=pcg_calls, launch=real_launch, **launch_kwargs):
                            if kernel == multilevel._solve_energy_galerkin_pcg_persistent:
                                calls.append((launch_args, launch_kwargs))
                            return launch(kernel, *launch_args, **launch_kwargs)

                        restore(frozen)
                        with patch.object(wp, "launch", record_pcg):
                            apply_current()
                        pcg_args, pcg_kwargs = pcg_calls[0]
                        pcg_ms = graph_milliseconds(
                            lambda pcg_args=pcg_args, pcg_kwargs=pcg_kwargs: wp.launch(
                                multilevel._solve_energy_galerkin_pcg_persistent, *pcg_args, **pcg_kwargs
                            ),
                            example.model.device,
                        )
                        backend.particle_multilevel = None
                        backend.particle_chebyshev_enabled = False

                        def fine_sweep():
                            restore(frozen)
                            iteration(state_in, state_out, contacts, dt, pre_sweeps)

                        sweep_ms = graph_milliseconds(fine_sweep, example.model.device)
                        restore(frozen)
                        emit(
                            kind="stage_timings",
                            fine_operator_build_ms=fine_assembly_ms,
                            fine_matrix_product_ms=multiply_ms,
                            coupled_coarse_pass_ms=coarse_ms,
                            coarse_operator_build_ms=coarse_assembly_ms,
                            coarse_pcg_ms=pcg_ms,
                            fast_sweep_ms=sweep_ms,
                        )

                    def product(vector, fine=fine, fine_ids=fine_ids, data=data):
                        v = wp.array(vector[fine_ids], dtype=wp.vec3, device=example.model.device)
                        result = wp.empty_like(v)
                        wp.launch(
                            _matrix_product,
                            dim=fine.cluster_count,
                            inputs=[
                                fine.coarse_matrix_offsets,
                                fine.coarse_matrix_columns,
                                fine.coarse_matrix_blocks,
                                data,
                                v,
                            ],
                            outputs=[result],
                            device=example.model.device,
                        )
                        output = np.zeros_like(q0)
                        output[fine_ids] = result.numpy()
                        return output

                    ad = product(direction)
                    diagonal = fine.local_hessians.numpy().astype(np.float64)
                    jacobi_image = np.zeros_like(q0)
                    jacobi_image[active] = np.linalg.solve(diagonal[active], ad[active, :, None])[:, :, 0]
                    aj = product(jacobi_image)
                    al = product(local0)
                    second_image = np.zeros_like(q0)
                    second_image[active] = np.linalg.solve(diagonal[active], aj[active, :, None])[:, :, 0]
                    local_image = np.zeros_like(q0)
                    local_image[active] = np.linalg.solve(diagonal[active], al[active, :, None])[:, :, 0]
                    asecond = product(second_image)
                    alocal_image = product(local_image)
                    for label, directions, images in (
                        ("ritz_coarse", [direction], [ad]),
                        ("ritz_coarse_local", [direction, local0], [ad, al]),
                        ("ritz_coarse_krylov", [direction, jacobi_image], [ad, aj]),
                        ("ritz_three", [direction, jacobi_image, local0], [ad, aj, al]),
                        (
                            "ritz_five",
                            [direction, jacobi_image, second_image, local0, local_image],
                            [ad, aj, asecond, al, alocal_image],
                        ),
                    ):
                        try:
                            candidate, weights, eigenvalues = ritz_step(directions, images, force0)
                        except ValueError as exception:
                            emit(kind="ritz_rejected", case=label, reason=str(exception))
                            continue
                        emit(
                            kind="ritz",
                            case=label,
                            coefficients=weights.tolist(),
                            eigenvalues=eigenvalues.tolist(),
                            predicted_force_rms=rms(
                                (force0 - sum(w * image for w, image in zip(weights, images, strict=True)))[active]
                            ),
                        )
                        candidates.append((label, 1.0, candidate))
                        if label in ("ritz_three", "ritz_five"):
                            candidates.append((label + "_half", 0.5, 0.5 * candidate))
                            combined = sum(w * image for w, image in zip(weights, images, strict=True))
                            guard_image = np.linalg.solve(diagonal[active], combined[active, :, None])[:, :, 0]
                            bounded_alpha = local_residual_step_bound(local0[active], guard_image)
                            candidates.append((label + "_guarded", bounded_alpha, bounded_alpha * candidate))
                for label, alpha, proposed in candidates:
                    restore(frozen)
                    step = proposed.copy()
                    norm = np.linalg.norm(step, axis=1)
                    cap = radius * 0.05
                    step *= np.minimum(1.0, cap / np.maximum(norm, 1e-30))[:, None]
                    backend.particle_displacements.assign(delta0 + step.astype(np.float32))
                    backend._penetration_free_truncation(state_in.particle_q)
                    q = state_in.particle_q.numpy().astype(np.float64)
                    accepted = q - q0
                    local, force = assemble(correction)
                    before_post = {
                        "error30": rms((q - refs[30])[active]),
                        "error60": rms((q - refs[60])[active]),
                        "local_newton_rms": rms(local[active]),
                        "reconstructed_force_rms": rms(force[active]),
                        "force_dot_direction": float(np.sum(force * direction)),
                        "accepted_step_rms": rms(accepted[active]),
                        "proposed_step_rms": rms(step[active]),
                        "dat_change_rms": rms((accepted - step)[active]),
                        "clamp_fraction": float(np.mean(norm[active] > cap[active])),
                    }
                    # Two existing fast sweeps, with extrapolation history reset
                    # after a coarse step just as in the coarse integration path.
                    backend.particle_multilevel = None
                    backend.particle_chebyshev_enabled = False
                    for i in range(2):
                        iteration(state_in, state_out, contacts, dt, pre_sweeps + i)
                    q_post = state_in.particle_q.numpy().astype(np.float64)
                    post_local, post_force = assemble(correction)
                    emit(
                        kind="step",
                        case=name,
                        candidate=label,
                        alpha=alpha,
                        before_post=before_post,
                        post_error30=rms((q_post - refs[30])[active]),
                        post_error60=rms((q_post - refs[60])[active]),
                        post_local_newton_rms=rms(post_local[active]),
                        post_reconstructed_force_rms=rms(post_force[active]),
                    )
        finally:
            for name, value in attrs.items():
                setattr(backend, name, value)
            restore(initial)
        raise ProbeComplete

    with patch.object(backend, "_solve_particle_iteration", probe):
        try:
            example.simulate()
        except ProbeComplete:
            pass


if __name__ == "__main__":
    main()
