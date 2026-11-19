"""CPU checks for the mass metric used by generated joints and distance constraints."""

from types import SimpleNamespace

import numpy as np
import pytest

newton = pytest.importorskip("newton")
pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver


def _normalize(mass):
    pairs = np.transpose(np.nonzero(np.triu(mass, k=1)))

    def array(values):
        return SimpleNamespace(numpy=lambda: np.asarray(values))

    # This host-only normalization does not need a native solver or CUDA state.
    adapter = SimpleNamespace(
        _affine_body_indices=array(np.arange(8).reshape(2, 4)),
        _affine_body_com=array(np.zeros((2, 3))),
        _particle_flags=array(np.full(8, int(newton.ParticleFlags.ACTIVE), dtype=np.int32)),
        _particle_mass_diagonal=array(np.diag(mass)),
        _particle_mass_non_diagonal_indices=array(pairs),
        _particle_mass_non_diagonal_values=array(mass[pairs[:, 0], pairs[:, 1]]),
        model=SimpleNamespace(particle_world=array(np.zeros(8, dtype=np.int32))),
    )
    springs, lines = NewtonSolver._normalize_joint_constraint_stiffnesses(
        adapter,
        [(0, -1), (0, 1)],
        [(0.1, 0.2, -0.3), (0.0, 0.2, 0.1)],
        [(0.0, 0.0, 0.0), (0.3, -0.1, 0.2)],
        [(1, -1)],
        [(0.2, 0.1, 0.3)],
        [(-1.0, 0.0, 0.0)],
        [(1.0, 0.0, 0.0)],
    )
    return np.array(springs + lines)


@pytest.mark.parametrize("small_mass", (1.0e-6, 1.0e-3, 1.0))
def test_joint_mass_normalization_preserves_unit_spectral_response_across_mass_scales(small_mass):
    mass = np.diag([0.25, 1.0e-4, 2.0e-4, 3.0e-4, small_mass, *([small_mass * 5.0e-4] * 3)])
    # Coupled rotational terms expose incorrect transposes in the whitening map.
    mass[1, 2] = mass[2, 1] = 5.0e-5
    mass[5, 6] = mass[6, 5] = small_mass * 1.0e-4
    stiffnesses = _normalize(mass)
    coefficients = np.array(
        [
            [-1.0, -0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -0.2, -0.1, 1.0, 0.3, -0.1, 0.2],
            [0.0, 0.0, 0.0, 0.0, -1.0, -0.2, -0.1, -0.3],
        ]
    )
    responses = np.diag(coefficients @ np.linalg.solve(mass, coefficients.T))
    metric = coefficients.T @ np.diag(stiffnesses) @ coefficients
    spectrum = np.linalg.eigvals(np.linalg.solve(mass, metric))

    assert np.isfinite(stiffnesses).all() and np.all(stiffnesses > 0.0)
    np.testing.assert_allclose(stiffnesses * responses, (stiffnesses * responses)[0], rtol=1.0e-10)
    assert np.max(spectrum.real) == pytest.approx(1.0, rel=1.0e-10)
    assert np.max(np.abs(spectrum.imag)) < 1.0e-10


@pytest.mark.parametrize("invalid", ("zero", "negative", "nonfinite", "singular", "indefinite", "near-singular"))
def test_joint_mass_normalization_still_rejects_invalid_mass(invalid):
    mass = np.eye(8)
    if invalid in ("zero", "negative", "nonfinite"):
        mass[1, 1] = {"zero": 0.0, "negative": -1.0, "nonfinite": np.nan}[invalid]
    else:
        mass[1, 2] = mass[2, 1] = {"singular": 1.0, "indefinite": 2.0, "near-singular": 1.0 - 1.0e-7}[invalid]
    with pytest.raises(ValueError, match="singular proxy mass matrix"):
        _normalize(mass)
