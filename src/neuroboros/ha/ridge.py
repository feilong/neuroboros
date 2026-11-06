from datetime import datetime

import numpy as np
from scipy.linalg import eigh

from ..linalg import safe_svd


def _check_alphas(alphas):
    alphas = np.asarray(alphas, dtype=float)
    if (
        alphas.ndim != 1
        or not len(alphas)
        or not np.isfinite(alphas).all()
        or (alphas < 0).any()
    ):
        raise ValueError(
            "`alphas` must be a nonempty 1D array of finite nonnegative values."
        )
    return alphas


def _adjoint(A):
    """Conjugate transpose of a 2D array (plain transpose for real arrays)."""
    return A.conj().T if np.iscomplexobj(A) else A.T


def _svd_path(X, Y, alphas):
    U, s, Vh = safe_svd(X, remove_mean=False)
    projected = _adjoint(U) @ Y
    cutoff = np.finfo(s.dtype).eps * max(X.shape) * s.max()
    result = []
    for alpha in alphas:
        if alpha == 0:
            factors = np.divide(1.0, s, out=np.zeros_like(s), where=s > cutoff)
        else:
            factors = s / (s**2 + alpha)
        result.append(_adjoint(Vh) @ (factors[:, np.newaxis] * projected))
    return np.stack(result)


def _eigh_path(X, Y, alphas):
    # Work in double precision, keeping complex inputs complex.
    X = X.astype(np.result_type(X, np.float64), copy=False)
    Y = Y.astype(np.result_type(Y, np.float64), copy=False)
    primal = X.shape[1] <= X.shape[0]
    gram = _adjoint(X) @ X if primal else X @ _adjoint(X)
    eigenvalues, eigenvectors = eigh(
        gram, driver="evd", overwrite_a=True, check_finite=False
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if alphas.min() <= np.sqrt(np.finfo(np.float64).eps) * eigenvalues.max():
        # Weak regularization relative to the Gram matrix scale, where the
        # roundoff of the normal equations would be amplified.
        return _svd_path(X, Y, alphas)
    if primal:
        left = eigenvectors
        projected = _adjoint(eigenvectors) @ (_adjoint(X) @ Y)
    else:
        left = _adjoint(X) @ eigenvectors
        projected = _adjoint(eigenvectors) @ Y
    return np.stack(
        [
            left @ (projected / (eigenvalues + alpha)[:, np.newaxis])
            for alpha in alphas
        ]
    )


def ridge_path(X, Y, alphas, *, solver="eigh"):
    """Fit ridge transforms for several regularization strengths.

    For each alpha, minimizes ``||X @ B - Y||_F**2 + alpha * ||B||_F**2``
    without centering, intercept, or scaling, and without dividing ``alpha``
    by the number of samples. The decomposition is reused across alphas.
    See `ridge_grid` to additionally vary the number of retained principal
    components.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        Source data, real or complex.
    Y : ndarray of shape (n_samples, n_features_Y)
        Target data, real or complex. Must be 2D even for a single target.
    alphas : array-like of shape (n_alphas,)
        Finite nonnegative regularization strengths. Order and duplicates
        are preserved. Zero gives the minimum-norm least-squares solution.
    solver : {'eigh', 'svd'}, default='eigh'
        ``'eigh'`` decomposes ``XᴴX`` for tall or square matrices and ``XXᴴ``
        for wide matrices, using double precision. It falls back to SVD when
        regularization is weak relative to the largest Gram eigenvalue.
        ``'svd'`` decomposes ``X`` directly without forming a Gram matrix.

    Returns
    -------
    betas : ndarray of shape (n_alphas, n_features_X, n_features_Y)
        ``X @ betas[i]`` approximates ``Y`` with regularization ``alphas[i]``.

    Notes
    -----
    The ``'eigh'`` solver promotes ``X`` and ``Y`` to double precision
    (float64 or complex128) and is typically faster than ``'svd'`` for tall
    searchlight matrices. It falls back to SVD when the smallest alpha is at
    or below ``sqrt(eps) * lambda_max``, where ``eps`` is float64 machine
    precision and ``lambda_max`` is the largest Gram eigenvalue.
    """
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("`X` and `Y` must be 2D with the same number of samples.")
    if not all(X.shape) or not all(Y.shape):
        raise ValueError("`X` and `Y` must have nonempty sample and feature axes.")
    alphas = _check_alphas(alphas)
    if solver == "eigh":
        return _eigh_path(X, Y, alphas)
    if solver == "svd":
        return _svd_path(X, Y, alphas)
    raise ValueError(f"`solver` must be 'eigh' or 'svd', got {solver!r}.")


def ridge(X, Y, alpha, *, solver="eigh"):
    """Fit a ridge transform ``B`` such that ``X @ B`` approximates ``Y``.

    Minimizes ``||X @ B - Y||_F**2 + alpha * ||B||_F**2`` without centering,
    intercept, or scaling. See `ridge_path` for the solvers and for fitting
    several alphas from one decomposition.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        Source data, real or complex.
    Y : ndarray of shape (n_samples, n_features_Y)
        Target data, real or complex. Must be 2D even for a single target.
    alpha : float
        Finite nonnegative regularization strength. Zero gives the
        minimum-norm least-squares solution.
    solver : {'eigh', 'svd'}, default='eigh'
        See `ridge_path`.

    Returns
    -------
    beta : ndarray of shape (n_features_X, n_features_Y)
        The ridge transform.
    """
    if np.ndim(alpha) != 0:
        raise ValueError("`alpha` must be a scalar; use `ridge_path` for several.")
    return ridge_path(X, Y, [alpha], solver=solver)[0]


def ridge_grid(X, y, alphas, npcs, train_idx=None):
    """Fit ridge coefficients over a grid of alphas and numbers of components.

    Unlike `ridge_path`, which varies only the regularization strength,
    this varies both the regularization strength and the number of
    retained principal components of ``X``, for a single target ``y``.
    Inputs are not centered: the components are those of the supplied
    ``X``, so center the data beforehand for conventional PCA.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Source data.
    y : ndarray of shape (n_samples,)
        Target data.
    alphas : ndarray of shape (n_alphas,)
        Regularization strengths.
    npcs : sequence of int
        Numbers of principal components, in increasing order. Each entry
        retains the leading components of ``X``.
    train_idx : ndarray or None, default=None
        Indices of the samples used for fitting. If None, all samples are
        used.

    Returns
    -------
    betas : ndarray of shape (n_features, n_alphas, n_npcs)
        ``betas[:, i, j]`` is the fit with ``alphas[i]`` and ``npcs[j]``
        components.
    """
    if train_idx is not None:
        X = X[train_idx]
        y = y[train_idx]
    # Assumes data have zero mean
    U, s, Vt = safe_svd(X, remove_mean=False)
    n_npcs = len(npcs)

    d = s[:, np.newaxis] / (alphas[np.newaxis, :] + (s**2)[:, np.newaxis])  # (k, n_alphas)
    UT_y = np.dot(U.T, y)  # (k, )
    d_UT_y = d * UT_y[..., np.newaxis]  # (k, n_alphas)
    betas = np.zeros((X.shape[1], len(alphas), n_npcs))

    npcs = [0] + list(npcs)
    for i in range(n_npcs):
        slc = slice(npcs[i], npcs[i+1])
        betas[..., i] = np.tensordot(Vt.T[:, slc], d_UT_y[slc], axes=(1, 0))

    betas = np.cumsum(betas, axis=2)

    return betas


def ensemble_ridge(X, y, alphas, npcs, train_idx_li, test_idx_li, msg=None):
    nt, nv = X.shape
    n_alphas, n_npcs = len(alphas), len(npcs)

    weights_test = np.zeros((nt, nv, n_alphas, n_npcs))
    count_test = np.zeros((nt, ), dtype=int)
    n_models = len(train_idx_li)
    weights = np.zeros((n_models, nv, n_alphas, n_npcs))
    count = 0

    for i, (train_idx, test_idx) in enumerate(zip(train_idx_li, test_idx_li)):
        betas = ridge_grid(X[train_idx], y[train_idx], alphas, npcs)
        weights_test[test_idx] += betas[np.newaxis]
        weights[i] = betas
        count_test[test_idx] += 1
        count += 1

    weights_test /= count_test[:, np.newaxis, np.newaxis, np.newaxis]
    weights = np.mean(weights, axis=0)

    pred = np.sum(X[:, :, np.newaxis, np.newaxis] * weights_test, axis=1)
    cost = np.sum((y[:, np.newaxis, np.newaxis] - pred)**2, axis=0)
    i, j = np.unravel_index(np.argmin(cost, axis=None), cost.shape)
    alpha = alphas[i]
    npc = npcs[j]
    weights = weights[:, i, j]
    r2 = 1. - cost[i, j] / np.sum(y**2)
    if msg is not None:
        print(datetime.now(), msg, f'{r2:8.5f}, {np.log10(alpha):4.1f}', npc)
    return weights, pred[:, i, j], r2, alpha, npc
