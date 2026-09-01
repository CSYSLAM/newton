# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural acceptance tests for the standalone MuJoCo/VBD demos."""

from __future__ import annotations

import ast
import importlib
import pathlib
import unittest

import newton
import newton.examples

_ROOT = pathlib.Path(newton.__file__).parent / "examples" / "mujoco_vbd"
_MODULES = {
    "tshirt": "example_mujoco_vbd_bimanual_fold_tshirt",
    "plastic_bag_rod": "example_mujoco_vbd_bimanual_plastic_bag_rod",
    "pneumatic_bag": "example_mujoco_vbd_recorded_plastic_inflatable_bag_pick_release",
    "soft_then_rigid_cube": "example_mujoco_vbd_recorded_soft_then_rigid_cube_into_bag",
    "armadillo_crusher": "example_mujoco_vbd_right_hand_armadillo_into_gear_crusher",
    "bimanual_nut_bolt": "example_mujoco_vbd_bimanual_nut_bolt",
    "plug_socket": "example_mujoco_vbd_dexforce_realtime_plug_socket",
    "tablecloth": "example_mujoco_vbd_dexforce_bimanual_place_tablecloth_waic_house",
    "cloth_twist": "example_mujoco_vbd_cloth_twist",
    "push_chair": "example_mujoco_vbd_dexforce_realtime_push_chair",
    "gear_crusher": "example_mujoco_vbd_gear_crusher",
}
_FORBIDDEN_IMPORTS = (
    "newton.examples.mjvbdv2",
    "newton._src.solvers.mjvbd_v2",
    "newton._src.solvers.coupled",
)
_FORBIDDEN_NAMES = {"SolverMJVBDV2", "SolverCoupled", "SolverCoupledProxy"}


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _solver_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "SolverMuJoCoVBD":
            yield node
        elif isinstance(function, ast.Attribute) and function.attr == "SolverMuJoCoVBD":
            yield node


def _is_main_guard(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


class TestMuJoCoVBDExamples(unittest.TestCase):
    def _assert_demo(self, key: str) -> None:
        """Verify that one demo is importable, testable, and registered."""
        short_name = _MODULES[key]
        module = importlib.import_module(f"newton.examples.mujoco_vbd.{short_name}")
        self.assertTrue(hasattr(module, "Example"))
        self.assertTrue(hasattr(module.Example, "test_final") or hasattr(module.Example, "test_post_step"))
        self.assertTrue(callable(module.Example.create_parser))
        self.assertIn(short_name.removeprefix("example_"), newton.examples.get_examples())

    def test_examples_have_no_mjvbd_runtime_imports(self):
        """Reject runtime dependencies on the legacy solver and demos."""
        offenders: list[str] = []
        for path in sorted(_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for imported in _imports(tree):
                if imported.startswith(_FORBIDDEN_IMPORTS):
                    offenders.append(f"{path.name}: import {imported}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
                    offenders.append(f"{path.name}: name {node.id}")
        self.assertEqual(offenders, [])

    def test_each_demo_has_one_public_example_and_no_example_imports(self):
        """Require each standalone module to own one directly executable example."""
        for short_name in _MODULES.values():
            path = _ROOT / f"{short_name}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            public_examples = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Example"]
            self.assertEqual(len(public_examples), 1, path.name)
            imported_examples = [name for name in _imports(tree) if name.startswith("newton.examples.")]
            self.assertEqual(imported_examples, [], path.name)
            public_main = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
            self.assertEqual(len(public_main), 1, f"{path.name} must be directly executable")
            self.assertTrue(any(_is_main_guard(node) for node in tree.body), f"{path.name} has no __main__ guard")

    def test_every_local_solver_call_has_explicit_mode(self):
        """Require explicit pure-VBD or one-way dispatch at every solver call."""
        calls = []
        for path in sorted(_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in _solver_calls(tree):
                calls.append((path, call))
                keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
                is_one_way = "mujoco_articulations" in keywords or "mujoco_joints" in keywords
                expected = (
                    {"joint_mode": "kinematic", "coupling_mode": "one_way"}
                    if is_one_way
                    else {"joint_mode": "dynamic", "coupling_mode": "auto"}
                )
                for name, value in expected.items():
                    self.assertIn(name, keywords, f"{path.name} omits {name}")
                    self.assertIsInstance(keywords[name], ast.Constant)
                    self.assertEqual(keywords[name].value, value)
        self.assertGreater(len(calls), 0)

    def test_pneumatic_bag_uses_full_core_soft_pipeline_contract(self):
        """Keep the pneumatic bag on the full-core, soft-contact branch."""
        path = _ROOT / f"{_MODULES['pneumatic_bag']}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = list(_solver_calls(tree))
        contact_modes = []
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
            contact_modes.append(ast.literal_eval(keywords["contact_mode"]))
        self.assertIn("soft", contact_modes)
        source = path.read_text(encoding="utf-8")
        self.assertRegex(source, r"features\.vbd_core != [\"']full[\"']")
        self.assertRegex(source, r"features\.contact_pipeline != [\"']soft[\"']")

    def test_tshirt_one_way_parity(self):
        """Register the standalone T-shirt folding demo."""
        self._assert_demo("tshirt")

    def test_plastic_bag_rod_one_way_parity(self):
        """Register the standalone plastic-bag rod demo."""
        self._assert_demo("plastic_bag_rod")

    def test_pneumatic_bag_one_way_parity(self):
        """Register the standalone pneumatic-bag demo."""
        self._assert_demo("pneumatic_bag")

    def test_soft_then_rigid_cube_one_way_parity(self):
        """Register the standalone soft/rigid cube demo."""
        self._assert_demo("soft_then_rigid_cube")

    def test_armadillo_crusher_one_way_parity(self):
        """Register the standalone hand-and-Armadillo crusher demo."""
        self._assert_demo("armadillo_crusher")

    def test_bimanual_nut_bolt_one_way_parity(self):
        """Register the standalone nut-and-bolt demo."""
        self._assert_demo("bimanual_nut_bolt")

    def test_plug_socket_one_way_parity(self):
        """Register the standalone plug-and-socket demo."""
        self._assert_demo("plug_socket")

    def test_tablecloth_one_way_parity(self):
        """Register the standalone tablecloth demo."""
        self._assert_demo("tablecloth")

    def test_cloth_twist_pure_vbd_parity(self):
        """Register the standalone pure-VBD cloth-twist demo."""
        self._assert_demo("cloth_twist")

    def test_push_chair_one_way_parity(self):
        """Register the standalone realtime chair-pushing demo."""
        self._assert_demo("push_chair")

    def test_gear_crusher_pure_vbd_parity(self):
        """Register the standalone pure-VBD gear-crusher demo."""
        self._assert_demo("gear_crusher")


if __name__ == "__main__":
    unittest.main(verbosity=2)
