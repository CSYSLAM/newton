# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Architecture tests for :class:`SolverMuJoCoVBD` (``DESIGN.md`` section 23.1).

These tests enforce the independence, private-baseline, and static-dispatch
contracts. They are pure host logic plus an AST scan, so they run on any device
without constructing a GPU backend.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco_vbd import SolverMuJoCoVBD
from newton._src.solvers.mujoco_vbd.config import (
    PROXY_RESPONSE_EFFECTIVE_MASS,
    MuJoCoVBDCouplingOptions,
    validate_coupling_options,
)
from newton._src.solvers.mujoco_vbd.dispatch import (
    MuJoCoVBDBackendKind,
    discover_features,
    select_backend_kind,
)
from newton._src.solvers.mujoco_vbd.kernels import sync_and_rewind_proxy_bodies_kernel
from newton._src.solvers.mujoco_vbd.ownership import resolve_mujoco_vbd_ownership

_PACKAGE_ROOT = pathlib.Path(newton.__file__).parent / "_src" / "solvers" / "mujoco_vbd"
_NEWTON_ROOT = pathlib.Path(newton.__file__).parent.parent

FORBIDDEN_PRODUCTION_IMPORT_PREFIXES = (
    "newton._src.solvers.coupled",
    "newton._src.solvers.mjvbd_v2",
    "newton._src.solvers.solver_mujoco",
    "newton._src.solvers.solver_vbd",
    "newton._src.solvers.vbd.",
    "newton._src.solvers.mujoco.",
)


def _module_name_for(path: pathlib.Path) -> str:
    rel = path.relative_to(_NEWTON_ROOT).with_suffix("")
    return ".".join(("newton", *rel.parts[1:])) if rel.parts[0] == "newton" else ".".join(rel.parts)


def _resolve_relative(package_parts: list[str], level: int, module: str | None) -> str:
    base = package_parts[: len(package_parts) - (level - 1)] if level > 1 else package_parts
    tail = [module] if module else []
    return ".".join([*base, *tail])


def _iter_imported_modules(tree: ast.AST, package_parts: list[str]):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                yield _resolve_relative(package_parts, node.level, node.module)
            elif node.module:
                yield node.module


# -- particle / articulation fixtures --------------------------------------


def _cloth_only_builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    for i in range(4):
        b.add_particle(pos=wp.vec3(0.1 * i, 0.0, 1.0), vel=wp.vec3(0.0), mass=1.0)
    b.color()
    return b


def _articulation_builder(with_particles: bool = False) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    link = b.add_link(mass=1.0)
    b.add_shape_capsule(body=link, radius=0.05, half_height=0.2)
    joint = b.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z)
    b.add_articulation([joint])
    if with_particles:
        for i in range(4):
            b.add_particle(pos=wp.vec3(0.1 * i, 0.0, 1.0), vel=wp.vec3(0.0), mass=1.0)
        b.color()
    return b


def _select(model, *, joint_mode, coupling_mode="auto", contact_mode="auto", joints=None):
    ownership = resolve_mujoco_vbd_ownership(model, mujoco_articulations=None, mujoco_joints=joints)
    discovered = discover_features(model, ownership)
    return select_backend_kind(
        discovered,
        joint_mode=joint_mode,
        coupling_mode=coupling_mode,
        contact_mode=contact_mode,
    )


class TestMuJoCoVBDArchitecture(unittest.TestCase):
    def test_soft_contact_robustness_options_validate(self):
        options = validate_coupling_options(None)
        self.assertGreater(options.soft_contact_speculative_distance, 0.0)
        self.assertTrue(options.soft_contact_augmented_lagrangian)
        with self.assertRaisesRegex(ValueError, "soft_contact_speculative_distance"):
            validate_coupling_options({"soft_contact_speculative_distance": -1.0})
        with self.assertRaisesRegex(ValueError, "soft_contact_al_rho_scale"):
            validate_coupling_options({"soft_contact_al_rho_scale": 0.0})
        with self.assertRaisesRegex(ValueError, "soft_contact_lambda_decay"):
            validate_coupling_options({"soft_contact_lambda_decay": 1.1})

    # -- 23.1 independence --

    def test_production_package_has_no_forbidden_solver_imports(self):
        offenders: list[str] = []
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            package_parts = _module_name_for(path).split(".")[:-1]
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module in _iter_imported_modules(tree, package_parts):
                if module and module.startswith(FORBIDDEN_PRODUCTION_IMPORT_PREFIXES):
                    offenders.append(f"{path.name}: {module}")
        self.assertEqual(offenders, [], f"forbidden imports found: {offenders}")

    def test_private_baseline_manifest_is_complete(self):
        manifest = (_PACKAGE_ROOT / "PRIVATE_BASELINE.md").read_text(encoding="utf-8")
        for token in ("source branch", "source commit", "copy date", "mujoco/", "vbd/", "vbd_soft/"):
            self.assertIn(token, manifest, f"PRIVATE_BASELINE.md missing {token!r}")

    def test_private_cores_are_physically_present(self):
        for rel in ("mujoco/solver_mujoco.py", "vbd/solver_vbd.py", "vbd_soft/solver_vbd.py", "coupling_types.py"):
            self.assertTrue((_PACKAGE_ROOT / rel).is_file(), f"missing private file {rel}")

    # -- 23.1 static dispatch matrices --

    def test_auto_backend_selection_matrix(self):
        K = MuJoCoVBDBackendKind
        cloth = _cloth_only_builder().finalize()
        self.assertEqual(_select(cloth, joint_mode="dynamic"), K.PURE_VBD_SOFT)

        arti = _articulation_builder().finalize()
        self.assertEqual(_select(arti, joint_mode="dynamic"), K.PURE_MUJOCO)
        self.assertEqual(_select(arti, joint_mode="kinematic"), K.KINEMATIC_PASSTHROUGH)

        mixed = _articulation_builder(with_particles=True).finalize()
        self.assertEqual(_select(mixed, joint_mode="dynamic"), K.TWO_WAY)
        self.assertEqual(_select(mixed, joint_mode="kinematic"), K.ONE_WAY_KINEMATIC_SOFT)

    def test_explicit_backend_selection_matrix(self):
        K = MuJoCoVBDBackendKind
        mixed = _articulation_builder(with_particles=True).finalize()
        self.assertEqual(
            _select(mixed, joint_mode="kinematic", coupling_mode="one_way"),
            K.ONE_WAY_KINEMATIC_SOFT,
        )
        self.assertEqual(
            _select(mixed, joint_mode="dynamic", coupling_mode="one_way"),
            K.ONE_WAY_DYNAMIC_SOFT,
        )
        self.assertEqual(_select(mixed, joint_mode="dynamic", coupling_mode="two_way"), K.TWO_WAY)

    def test_one_way_exposes_no_finite_mass_proxy_mode(self):
        """Keep finite-mass proxies exclusive to two-way coupling."""
        parameters = inspect.signature(SolverMuJoCoVBD.__init__).parameters
        self.assertNotIn("one_way_proxy_response", parameters)
        self.assertFalse((_PACKAGE_ROOT / "backends" / "one_way_kinematic_massive.py").exists())
        self.assertNotIn("ONE_WAY_KINEMATIC_MASSIVE", MuJoCoVBDBackendKind.__members__)

    def test_reject_incompatible_mode_options(self):
        mixed = _articulation_builder(with_particles=True).finalize()
        with self.assertRaises(ValueError):
            _select(mixed, joint_mode="kinematic", coupling_mode="two_way")  # hard kinematic two-way

        cloth = _cloth_only_builder().finalize()
        with self.assertRaises(ValueError):
            _select(cloth, joint_mode="dynamic", coupling_mode="one_way")  # no MuJoCo articulation

    # -- 23.1 clean branches --

    def test_pure_vbd_does_not_allocate_mujoco_or_coupling(self):
        model = _cloth_only_builder().finalize()
        solver = SolverMuJoCoVBD(model)
        self.assertEqual(solver.backend_kind, MuJoCoVBDBackendKind.PURE_VBD_SOFT)
        self.assertIsNone(solver.mujoco_solver)
        self.assertIsNotNone(solver.vbd_solver)
        self.assertFalse(solver.features.feedback_enabled)
        self.assertFalse(solver.features.effective_mass_enabled)
        self.assertIsNone(solver.diagnostics.feedback_wrench_raw)

    def test_backend_kind_is_immutable_after_construction(self):
        model = _cloth_only_builder().finalize()
        solver = SolverMuJoCoVBD(model)
        kind = solver.backend_kind
        st0, st1 = model.state(), model.state()
        solver.step(st0, st1, model.control(), None, 1.0 / 60.0)
        wp.synchronize()
        self.assertEqual(solver.backend_kind, kind)

    def test_pure_backend_rejects_explicit_coupling_options(self):
        model = _cloth_only_builder().finalize()
        with self.assertRaises(ValueError):
            SolverMuJoCoVBD(model, coupling_options=MuJoCoVBDCouplingOptions(iterations=3))

    def test_two_way_proxy_sync_reconstructs_substep_begin_pose(self):
        """Prevent VBD from advancing the MuJoCo interval twice."""
        device = "cpu"
        dt = 0.25
        proxy_body_ids = wp.array([0], dtype=wp.int32, device=device)
        source_body_q = wp.array(
            [wp.transform(wp.vec3(1.5, 2.75, 4.0), wp.quat_identity())], dtype=wp.transform, device=device
        )
        source_body_qd = wp.array(
            [wp.spatial_vector(2.0, 3.0, 4.0, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector, device=device
        )
        body_com = wp.zeros(1, dtype=wp.vec3, device=device)
        body_inertia = wp.array([wp.mat33(1.0)], dtype=wp.mat33, device=device)
        gravity = wp.array([wp.vec3(0.0, 0.0, -10.0)], dtype=wp.vec3, device=device)
        wrench = wp.array([wp.spatial_vector(4.0, 5.0, 6.0, 7.0, 8.0, 9.0)], dtype=wp.spatial_vector, device=device)
        inverse_mass = wp.array([0.5], dtype=float, device=device)
        destination_body_q = wp.zeros(1, dtype=wp.transform, device=device)
        destination_body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        destination_body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        destination_body_q_prev = wp.zeros(1, dtype=wp.transform, device=device)
        destination_body_q_prev_snapshot = wp.zeros(1, dtype=wp.transform, device=device)
        proxy_qd_before = wp.zeros(1, dtype=wp.spatial_vector, device=device)

        wp.launch(
            sync_and_rewind_proxy_bodies_kernel,
            dim=1,
            inputs=[
                dt,
                proxy_body_ids,
                source_body_q,
                source_body_qd,
                body_com,
                body_inertia,
                gravity,
                wrench,
                inverse_mass,
                PROXY_RESPONSE_EFFECTIVE_MASS,
                destination_body_q,
                destination_body_qd,
                destination_body_f,
                destination_body_q_prev,
                destination_body_q_prev_snapshot,
                proxy_qd_before,
            ],
            device=device,
        )

        expected_pose = np.asarray([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(destination_body_q.numpy(), expected_pose)
        np.testing.assert_allclose(destination_body_q_prev.numpy(), expected_pose)
        np.testing.assert_allclose(destination_body_q_prev_snapshot.numpy(), expected_pose)
        np.testing.assert_allclose(destination_body_qd.numpy(), np.asarray([[2, 3, 4, 0, 0, 0]]))
        np.testing.assert_allclose(proxy_qd_before.numpy(), np.asarray([[2, 3, 4, 0, 0, 0]]))
        np.testing.assert_allclose(destination_body_f.numpy(), np.asarray([[-4, -5, 14, -7, -8, -9]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
