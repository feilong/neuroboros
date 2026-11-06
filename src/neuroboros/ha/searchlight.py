import functools

import numpy as np
import scipy.sparse as sparse
from joblib import Parallel, delayed

from .local_template import compute_template
from .procrustes import procrustes
from .ridge import ridge
from .sparse import csc_block_offsets
from .transform import Transform


def compute_searchlight_weights(sls, dists=None, radius=None, return_sparse=False):
    """
    Compute the weights used for combining searchlight models.
    If `dists` is None, uniform weights are used.
    If `dists` is provided, distance-based weights are computed, where the
    weight for each vertex is highest at the center of the searchlight and
    decreases linearly to zero at the edge of the searchlight.

    Typical usage:
    >>> weights = compute_searchlight_weights(sls, dists, radius)
    >>> mat = compute_searchlight_weights(
    ...     sls, dists, radius, return_sparse=True)

    Parameters
    ----------
    sls : list of ndarrays
        Searchlight indices, where each ndarray contains the indices of the
        vertices in the searchlight.
    dists : list of ndarrays or None, default=None
        Each array contains the distances of the vertices in the searchlight
        to the center of the searchlight, used for distance-based weighting.
        If None, uniform weights are used.
    radius : float or None, default=None
        The radius of the searchlight, used for distance-based weighting.

    Returns
    -------
    weights : list of ndarrays, optional
        Each ndarray contains the weights for the vertices in the corresponding
        searchlight. The weights sum to 1 for each vertex. Returned only if
        `return_sparse` is False.
    mat : sparse matrix, optional
        A sparse matrix where each row corresponds to a searchlight and each
        column corresponds to a vertex. Returned only if `return_sparse` is
        True.
    """
    indices = np.concatenate(sls) if len(sls) else np.zeros(0, dtype=np.int64)
    nv = int(indices.max()) + 1 if len(indices) else 0
    weights_sum = np.zeros((nv,))
    weights = []

    if dists is None:
        for sl in sls:
            weights_sum[sl] += 1

        for sl in sls:
            w = 1.0 / weights_sum[sl]
            weights.append(w)

    else:
        # If dists is provided, use distance-based weighting
        assert radius is not None, "Radius must be provided if distances are given."
        for sl, d in zip(sls, dists):
            w = (radius - d) / radius
            weights_sum[sl] += w

        for sl, d in zip(sls, dists):
            w = (radius - d) / radius
            w /= weights_sum[sl]
            weights.append(w)

    if return_sparse:
        mat = sparse.lil_array((len(sls), nv))
        for i, w in enumerate(weights):
            mat[i, sls[i]] = w
        mat = mat.tocsr()
        return mat

    return weights


def searchlight_hyperalignment(
    X, Y, sls, sls_out=None, sl_func=None, mat0=None, weights=None
):
    """
    Searchlight hyperalignment.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_out : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    sl_func : callable
        A function that takes two arrays and computes the transformation
        between them.
    mat0 : Transform, sparse matrix or None, default=None
        An all-zero transform with the sparsity structure of the searchlights,
        as returned by `Transform.from_searchlights` or
        `initialize_sparse_matrix`. It is copied, not modified. If None,
        a dense array of zeros is used instead.
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.

    Returns
    -------
    mat : Transform, sparse matrix or ndarray
        The resulting transformation matrix, of the same kind as ``mat0``.
        Each row corresponds to a vertex in ``X`` and each column to a vertex
        in ``Y``, so that ``X @ mat`` approximates ``Y``.
    """
    if sls_out is None:
        sls_out = sls
    if weights is None:
        weights = [None] * len(sls)

    if mat0 is None:
        mat = np.zeros((X.shape[1], Y.shape[1]))

        def add_block(sl_X, sl_Y, t):
            mat[np.ix_(sl_X, sl_Y)] += t

    elif isinstance(mat0, Transform):
        mat = mat0.copy()
        add_block = mat.add_block

    else:
        # Write directly into the ``data`` array of the preallocated CSC
        # matrix. Indexing the sparse matrix itself is far slower, because
        # scipy validates the entire matrix on every write.
        mat = mat0.tocsc(copy=True)
        mat.sum_duplicates()
        indptr, indices, data = mat.indptr, mat.indices, mat.data

        def add_block(sl_X, sl_Y, t):
            offsets = csc_block_offsets(indptr, indices, sl_X, sl_Y)
            data[offsets] += t.T.ravel()

    for sl_X, sl_Y, w in zip(sls, sls_out, weights):
        t = sl_func(X[:, sl_X], Y[:, sl_Y])
        if w is not None:
            t = t * w[np.newaxis]
        add_block(sl_X, sl_Y, t)
    return mat


def searchlight_procrustes(
    X, Y, sls, sls_out=None, mat0=None, reflection=True, scaling=False, weights=None
):
    """
    Searchlight hyperalignment using orthogonal Procrustes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_out : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    mat0 : Transform, sparse matrix or None, default=None
        An all-zero transform with the sparsity structure of the searchlights,
        as returned by `Transform.from_searchlights` or
        `initialize_sparse_matrix`. It is copied, not modified. If None,
        a dense array of zeros is used instead.
    reflection : bool, default=True
        Whether allows reflection in the transformation (True) or not (False).
    scaling : bool, default=False
        Whether allows global scaling (True) or not (False).
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.

    Returns
    -------
    mat : Transform, sparse matrix or ndarray
        The resulting transformation matrix, of the same kind as ``mat0``.
        Each row corresponds to a vertex in ``X`` and each column to a vertex
        in ``Y``, so that ``X @ mat`` approximates ``Y``.
    """
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    xfm = searchlight_hyperalignment(
        X, Y, sls, sls_out=sls_out, sl_func=sl_func, mat0=mat0, weights=weights
    )
    return xfm


def searchlight_ridge(
    X, Y, sls, sls_out=None, mat0=None, alpha=1e3, weights=None, *, solver="eigh"
):
    """
    Searchlight hyperalignment using ridge regression.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The first data matrix, to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The second data matrix, the target for alignment.
    sls : list of ndarrays
        Each ndarray contains the indices of the vertices in the searchlight
        for the first data matrix.
    sls_out : list of ndarrays or None, default=None
        Searchlight indices for the second data matrix, if different from
        ``sls``.
    mat0 : Transform, sparse matrix or None, default=None
        An all-zero transform with the sparsity structure of the searchlights,
        as returned by `Transform.from_searchlights` or
        `initialize_sparse_matrix`. It is copied, not modified. If None,
        a dense array of zeros is used instead.
    alpha : float, default=1e3
        The regularization parameter for ridge regression.
    weights : list of ndarrays or None, default=None
        Weights for combining searchlight transformations. If None, simply add
        the transformations without weighting.
    solver : {'eigh', 'svd'}, default='eigh'
        Solver of the local ridge regressions; see `ridge_path`.

    Returns
    -------
    mat : Transform, sparse matrix or ndarray
        The resulting transformation matrix, of the same kind as ``mat0``.
        Each row corresponds to a vertex in ``X`` and each column to a vertex
        in ``Y``, so that ``X @ mat`` approximates ``Y``.
    """
    sl_func = functools.partial(ridge, alpha=alpha, solver=solver)
    xfm = searchlight_hyperalignment(
        X, Y, sls, sls_out=sls_out, sl_func=sl_func, mat0=mat0, weights=weights
    )
    return xfm


def searchlight_template(dms, sls, dists, radius, n_jobs=1, tpl_kind="pca"):
    weights = compute_searchlight_weights(sls, dists, radius)
    tpl = np.zeros_like(dms[0])

    if n_jobs == 1:
        for sl, w in zip(sls, weights):
            local_template = compute_template(
                dms, sl=sl, kind=tpl_kind, max_npc=len(sl), common_topography=True
            )
            tpl[:, sl] += local_template * w[np.newaxis]
    else:
        with Parallel(n_jobs=n_jobs, batch_size=1, verbose=1) as parallel:
            local_templates = parallel(
                delayed(compute_template)(
                    dms, sl=sl, kind=tpl_kind, max_npc=len(sl), common_topography=True
                )
                for sl in sls
            )

        for local_template, w, sl in zip(local_templates, weights, sls):
            tpl[:, sl] += local_template * w[np.newaxis]
    return tpl
