import importlib

import numpy as np
import pytest

from neuroboros.ha import ridge, ridge_path
ridge_module = importlib.import_module("neuroboros.ha.ridge")


def _augmented_lstsq(X, Y, alpha):
    """Ridge solution via least squares on the augmented system."""
    p = X.shape[1]
    A = np.vstack((X, np.sqrt(alpha) * np.eye(p)))
    b = np.vstack((Y, np.zeros((p, Y.shape[1]))))
    return np.linalg.lstsq(A, b, rcond=None)[0]


@pytest.mark.parametrize("shape", [(50, 8), (8, 50), (12, 12)])
@pytest.mark.parametrize("solver", ["eigh", "svd"])
def test_path_matches_augmented_least_squares(shape, solver):
    rng = np.random.default_rng(41)
    X = rng.normal(size=shape) + 2  # no implicit centering
    X[:, -1] = X[:, 0]  # rank deficient
    Y = rng.normal(size=(shape[0], 7)) - 1
    alphas = [0.01, 1000.0, 0.01]
    betas = ridge_path(X, Y, alphas, solver=solver)
    assert betas.shape == (3, shape[1], 7)
    for alpha, beta in zip(alphas, betas):
        np.testing.assert_allclose(
            beta, _augmented_lstsq(X, Y, alpha), atol=2e-10, rtol=2e-9
        )
    np.testing.assert_array_equal(betas[0], betas[2])


@pytest.mark.parametrize("shape", [(1210, 150), (150, 600)])
def test_solvers_agree(shape):
    rng = np.random.default_rng(3)
    X = rng.normal(size=shape)
    Y = rng.normal(size=(shape[0], 40))
    a = ridge(X, Y, 1e3, solver="eigh")
    b = ridge(X, Y, 1e3, solver="svd")
    np.testing.assert_allclose(a, b, atol=1e-13, rtol=1e-10)


def test_default_solver_is_eigh(monkeypatch):
    calls = []
    original = ridge_module.eigh
    monkeypatch.setattr(
        ridge_module, "eigh", lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    )
    ridge(np.eye(5), np.eye(5), 1.0)
    assert len(calls) == 1


def test_ridge_matches_path():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 9))
    Y = rng.normal(size=(30, 4))
    np.testing.assert_array_equal(ridge(X, Y, 2.5), ridge_path(X, Y, [2.5])[0])


def test_eigh_decomposition_reused(monkeypatch):
    calls = []
    original = ridge_module.eigh
    monkeypatch.setattr(
        ridge_module, "eigh", lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    )
    ridge_path(np.eye(5), np.eye(5), [0.01, 1.0, 1000.0])
    assert len(calls) == 1


def test_ill_conditioned_gram_falls_back_to_svd(monkeypatch):
    calls = []
    original = ridge_module.safe_svd
    monkeypatch.setattr(
        ridge_module,
        "safe_svd",
        lambda *a, **k: (calls.append(1), original(*a, **k))[1],
    )
    X = np.diag([1.0, 1e-6, 1e-12])
    Y = np.eye(3)
    result = ridge_path(X, Y, [1e-16])
    assert len(calls) == 1
    np.testing.assert_allclose(result[0], ridge(X, Y, 1e-16, solver="svd"), rtol=1e-12)


@pytest.mark.parametrize("shape", [(30, 7), (7, 30)])
@pytest.mark.parametrize("solver", ["eigh", "svd"])
def test_zero_alpha_is_minimum_norm(shape, solver):
    rng = np.random.default_rng(23)
    X = rng.normal(size=shape)
    X[:, -1] = X[:, 0]
    Y = rng.normal(size=(shape[0], 4))
    np.testing.assert_allclose(
        ridge_path(X, Y, [0.0], solver=solver)[0],
        np.linalg.lstsq(X, Y, rcond=None)[0],
        atol=1e-10,
    )


def test_float32_and_zero_matrix():
    X = np.zeros((5, 3), dtype=np.float32)
    Y = np.ones((5, 2))
    np.testing.assert_array_equal(ridge_path(X, Y, [0, 1]), 0)
    assert np.isfinite(ridge(X, Y, 0)).all()


@pytest.mark.parametrize("alphas", [[], [-1], [np.nan], [np.inf], [[1]], 1.0])
def test_invalid_alphas(alphas):
    with pytest.raises(ValueError, match="alphas"):
        ridge_path(np.eye(3), np.eye(3), alphas)


def test_invalid_arguments():
    X, Y = np.eye(5), np.ones((5, 2))
    with pytest.raises(ValueError, match="solver"):
        ridge(X, Y, 1, solver="typo")
    with pytest.raises(ValueError, match="scalar"):
        ridge(X, Y, [1])
    with pytest.raises(ValueError, match="samples"):
        ridge(X, Y[:4], 1)
    with pytest.raises(ValueError, match="nonempty"):
        ridge(X[:, :0], Y, 1)


@pytest.mark.parametrize("solver", ["eigh", "svd"])
@pytest.mark.parametrize("complex_X", [False, True])
def test_complex_values(solver, complex_X):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(20, 6))
    if complex_X:
        X = X + 1j * rng.normal(size=X.shape)
    Y = rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))
    for alpha in (1.0, 100.0):
        beta = ridge(X, Y, alpha, solver=solver)
        assert np.iscomplexobj(beta)
        np.testing.assert_allclose(beta, _augmented_lstsq(X, Y, alpha), atol=1e-10)
    X, Y = np.eye(2), np.array([[1 + 2j], [3 + 4j]])
    np.testing.assert_allclose(
        ridge(X, Y, 1.0, solver=solver), [[0.5 + 1j], [1.5 + 2j]], atol=1e-12
    )


@pytest.mark.parametrize("solver", ["eigh", "svd"])
def test_searchlight_ridge_solver(solver):
    from neuroboros.ha import searchlight_ridge

    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 9))
    Y = rng.normal(size=(40, 9))
    sls = [np.array([3, 0, 2]), np.array([1, 3, 4]), np.array([4, 5, 6])]
    expected = np.zeros((9, 9))
    for sl in sls:
        expected[np.ix_(sl, sl)] += _augmented_lstsq(X[:, sl], Y[:, sl], 10.0)
    actual = searchlight_ridge(X, Y, sls, alpha=10.0, solver=solver)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    with pytest.raises(ValueError, match="solver"):
        searchlight_ridge(X, Y, sls, solver="typo")
