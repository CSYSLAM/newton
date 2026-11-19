"""Keep the W1 passive jaw relation intact across Newton's sparse-to-joint metadata migration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "examples" / "coupled_high_accuracy" / "demos/utils/parallel_gripper.py"
spec = importlib.util.spec_from_file_location("w1_mimic_compat_under_test", path)
gripper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gripper)


def array(values):
    return SimpleNamespace(numpy=lambda: np.asarray(values))


def sparse(enabled=True):
    return {
        "constraint_mimic_" + key: array(value)
        for key, value in {
            "joint0": [1],
            "joint1": [0],
            "coef0": [0.02],
            "coef1": [-1.0],
            "enabled": [enabled],
        }.items()
    }


def test_joint_owned_mimic_survives_empty_legacy_arrays():
    model = SimpleNamespace(
        joint_mimic_joint=array([-1, 0, -1, 2]),
        joint_mimic_coeffs=array([[0, 1], [0.02, -1], [0, 1], [0, 1]]),
        **{"constraint_mimic_" + key: array([]) for key in ("joint0", "joint1", "coef0", "coef1", "enabled")},
    )
    assert gripper.mimic_constraints(model) == [(1, 0, 0.02, -1.0), (3, 2, 0.0, 1.0)]


def test_legacy_disabled_rows_stay_disabled():
    assert gripper.mimic_constraints(SimpleNamespace(**sparse(False))) == []
    assert gripper.mimic_constraints(SimpleNamespace(**sparse())) == [(1, 0, 0.02, -1.0)]


def test_conflicting_metadata_cannot_silently_change_jaw_relation():
    model = SimpleNamespace(joint_mimic_joint=array([-1, 0]), joint_mimic_coeffs=array([[0, 1], [0, 1]]), **sparse())
    with pytest.raises(ValueError, match="metadata disagree"):
        gripper.mimic_constraints(model)
