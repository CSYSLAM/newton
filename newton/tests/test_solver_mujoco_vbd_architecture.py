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

import warp as wp

import newton
from newton._src.solvers.mujoco_vbd import SolverMuJoCoVBD
from newton._src.solvers.mujoco_vbd.config import MuJoCoVBDCouplingOptions
from newton._src.solvers.mujoco_vbd.dispatch import (
    MuJoCoVBDBackendKind,
    discover_features,
    select_backend_kind,
)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
