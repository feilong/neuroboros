import functools
import operator

import numpy as np
import scipy.sparse as sparse
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .local_template import compute_template
from .procrustes import procrustes
from .ridge import _check_alphas, ridge_path
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


def _check_n_jobs(n_jobs):
    try:
        n_jobs = operator.index(n_jobs)
    except TypeError:
        raise TypeError(f"`n_jobs` must be an integer, got {n_jobs!r}.") from None
    if n_jobs == 0:
        raise ValueError("`n_jobs` must be a nonzero integer; -1 uses all CPUs.")
    return n_jobs


def _check_max_blas_threads(max_blas_threads):
    if max_blas_threads is None:
        return None
    try:
        max_blas_threads = operator.index(max_blas_threads)
    except TypeError:
        raise TypeError(
            f"`max_blas_threads` must be an integer or None, got {max_blas_threads!r}."
        ) from None
    if max_blas_threads < 1:
        raise ValueError("`max_blas_threads` must be a positive integer or None.")
    return max_blas_threads


def _check_data(X, Y):
    X, Y = np.asarray(X), np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("`X` and `Y` must be 2D with the same number of samples.")
    for name, arr in (("X", X), ("Y", Y)):
        if not np.isfinite(arr).all():
            raise ValueError(f"`{name}` contains NaN or infinite values.")
    return X, Y


def _check_nv(name, sls, n_features):
    """Check that the searchlights of ``name`` are defined for the data."""
    nv = getattr(sls, "nv", None)
    if nv is not None and nv != n_features:
        raise ValueError(
            f"`{name}` is defined for {nv} vertices but the data has "
            f"{n_features} features."
        )


def _default_weights(sls):
    """Combining weights of a searchlight set.

    Distance-based weights from `Searchlights.weights` when available, and
    uniform weights otherwise.
    """
    if hasattr(sls, "weights"):
        return sls.weights()
    return compute_searchlight_weights(sls)


def _default_structure(sls, sls_out, shape):
    """The sparsity structure of a transform between the searchlight sets.

    The structure cached on a `Searchlights` object is reused when the
    transform is square within that space. Without ``sls_out``, the same
    searchlights apply to the output space, which may have a different
    number of vertices.
    """
    if shape[0] == shape[1] and sls_out is None:
        if hasattr(sls, "structure"):
            return sls.structure()
        return Transform.from_searchlights(sls, nv=shape[0])
    if sls_out is None:
        sls_out = sls
    return Transform.from_searchlights(sls, sls_out, nv=shape[0], nv_out=shape[1])


def _searchlight_fit(
    X,
    Y,
    sls,
    sl_func,
    *,
    sls_out,
    weights,
    structure,
    n_models,
    n_jobs,
    max_blas_threads,
):
    """Fit local models in each searchlight and combine them into transforms.

    ``sl_func(X[:, sl], Y[:, sl_out])`` returns ``n_models`` blocks of shape
    ``(len(sl), len(sl_out))``, as a list or a stacked array. Block ``k`` of
    every searchlight, scaled by the searchlight's weights, is added to the
    ``k``-th transform. The transforms are copies of ``structure``.

    The blocks are added in searchlight order regardless of ``n_jobs``, so
    the number of jobs does not change the result. Jobs are threads; the
    local fits release the GIL in their linear algebra, and the data is
    shared rather than copied. The BLAS thread pools are limited to
    ``max_blas_threads`` while fitting, which avoids oversubscription when
    ``n_jobs > 1``.
    """
    n_jobs = _check_n_jobs(n_jobs)
    max_blas_threads = _check_max_blas_threads(max_blas_threads)
    X, Y = _check_data(X, Y)
    shape = (X.shape[1], Y.shape[1])
    _check_nv("sls", sls, shape[0])
    if sls_out is not None:
        _check_nv("sls_out", sls_out, shape[1])
        if len(sls_out) != len(sls):
            raise ValueError("`sls` and `sls_out` must have the same length.")
    if weights is None:
        weights = _default_weights(sls if sls_out is None else sls_out)
    elif len(weights) != len(sls):
        raise ValueError("`weights` must have one entry per searchlight.")
    if structure is None:
        structure = _default_structure(sls, sls_out, shape)
    elif not isinstance(structure, Transform):
        raise TypeError("`structure` must be a Transform or None.")
    if structure.shape != shape:
        raise ValueError(
            f"`structure` has shape {structure.shape}, but the data has "
            f"{shape[0]} input and {shape[1]} output features."
        )
    if structure.internal_matrix.data.any():
        raise ValueError(
            "`structure` must be all zeros; it holds a fitted transform. Use "
            "`Searchlights.structure`, `Transform.from_searchlights` or "
            "`Transform.load_structure`."
        )
    if sls_out is None:
        sls_out = sls

    def fit(sl_in, sl_out, w):
        if len(sl_in) == 0 or len(sl_out) == 0:
            return None
        w = np.asarray(w)
        if w.shape != (len(sl_out),):
            raise ValueError(
                "Each entry of `weights` must have one weight per vertex of "
                "the output searchlight."
            )
        blocks = sl_func(X[:, sl_in], Y[:, sl_out])
        if len(blocks) != n_models:
            raise ValueError(f"Expected {n_models} blocks, got {len(blocks)}.")
        offsets = structure.block_offsets(sl_in, sl_out)
        flat = []
        for block in blocks:
            block = np.asarray(block)
            if block.shape != (len(sl_in), len(sl_out)):
                raise ValueError(
                    f"Expected a block of shape {(len(sl_in), len(sl_out))}, "
                    f"got {block.shape}."
                )
            # ``w`` scales the columns, i.e., the output vertices. The
            # flattened block follows the column-major layout of the offsets.
            flat.append((block * w).T.ravel())
        return offsets, flat

    tasks = zip(sls, sls_out, weights)
    if n_jobs == 1:
        results = (fit(*task) for task in tasks)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(fit)(*task) for task in tasks
        )

    mats = [structure.copy() for _ in range(n_models)]
    datas = [mat.internal_matrix.data for mat in mats]
    with threadpool_limits(limits=max_blas_threads):
        for result in results:
            if result is None:
                continue
            offsets, flat = result
            for data, values in zip(datas, flat):
                data[offsets] += values
    return mats


def searchlight_hyperalignment(
    X,
    Y,
    sls,
    sl_func,
    *,
    sls_out=None,
    weights=None,
    structure=None,
    n_jobs=1,
    max_blas_threads=1,
):
    """Searchlight hyperalignment with a custom local model.

    A local transform is fitted in each searchlight with ``sl_func``, and
    the local transforms are combined into one sparse transform using
    ``weights``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The data to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The target of the alignment.
    sls : Searchlights or list of ndarrays
        Each entry holds the indices of the vertices in a searchlight.
    sl_func : callable
        ``sl_func(X[:, sl], Y[:, sl_out])`` returns the local transform, an
        array of shape ``(len(sl), len(sl_out))``, such that
        ``X[:, sl] @ t`` approximates ``Y[:, sl_out]``.
    sls_out : Searchlights, list of ndarrays or None, default=None
        Searchlights in the space of ``Y``, if different from ``sls``. Must
        have the same length as ``sls``; the ``i``-th searchlights of ``sls``
        and ``sls_out`` are fitted together.
    weights : list of ndarrays or None, default=None
        Weights for combining the local transforms. Entry ``i`` has one
        weight per vertex of the ``i``-th output searchlight. If None, they
        are computed from the output searchlights (``sls``, or ``sls_out``
        when given): distance-based weights for a `Searchlights` object
        with distances, uniform weights otherwise. Computed weights sum to
        1 for each vertex across the searchlights containing it; supplied
        weights are used as they are.
    structure : Transform or None, default=None
        An all-zero transform with the sparsity structure of the
        searchlights, e.g., from `Searchlights.structure`,
        `Transform.from_searchlights` or `Transform.load_structure`. It is
        copied, not modified. If None, the structure cached on ``sls`` is
        used when ``sls`` is a `Searchlights` object and the transform is
        square, and otherwise one is built from the searchlights.
    n_jobs : int, default=1
        The number of threads fitting local models, following joblib
        conventions; -1 uses all CPUs. The local transforms are combined in a
        fixed order, so the result does not depend on it as long as
        ``sl_func`` is deterministic and safe to call from several
        threads at once.
    max_blas_threads : int or None, default=1
        The maximum number of BLAS threads while fitting, whatever
        ``n_jobs`` is, set with threadpoolctl; None keeps the current
        setting. One thread avoids oversubscription with several ``n_jobs``
        and reduces numerical variation caused by BLAS threading. It has no
        effect on Apple's Accelerate, which only the
        ``VECLIB_MAXIMUM_THREADS`` environment variable controls.

    Returns
    -------
    xfm : Transform
        The transform, of shape ``(n_features_X, n_features_Y)``, such that
        ``X @ xfm`` approximates ``Y``.
    """
    return _searchlight_fit(
        X,
        Y,
        sls,
        lambda a, b: [sl_func(a, b)],
        sls_out=sls_out,
        weights=weights,
        structure=structure,
        n_models=1,
        n_jobs=n_jobs,
        max_blas_threads=max_blas_threads,
    )[0]


def searchlight_procrustes(
    X,
    Y,
    sls,
    *,
    reflection=True,
    scaling=False,
    sls_out=None,
    weights=None,
    structure=None,
    n_jobs=1,
    max_blas_threads=1,
):
    """Searchlight hyperalignment using orthogonal Procrustes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The data to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The target of the alignment.
    sls : Searchlights or list of ndarrays
        Each entry holds the indices of the vertices in a searchlight.
    reflection : bool, default=True
        Whether the local transforms may include reflections.
    scaling : bool, default=False
        Whether the local transforms may include a scale factor, one per
        searchlight.
    sls_out : Searchlights, list of ndarrays or None, default=None
        Searchlights in the space of ``Y``, if different from ``sls``. Must
        have the same length as ``sls``; the ``i``-th searchlights of ``sls``
        and ``sls_out`` are fitted together.
    weights : list of ndarrays or None, default=None
        Weights for combining the local transforms. Entry ``i`` has one
        weight per vertex of the ``i``-th output searchlight. If None, they
        are computed from the output searchlights (``sls``, or ``sls_out``
        when given): distance-based weights for a `Searchlights` object
        with distances, uniform weights otherwise. Computed weights sum to
        1 for each vertex across the searchlights containing it; supplied
        weights are used as they are.
    structure : Transform or None, default=None
        An all-zero transform with the sparsity structure of the
        searchlights, e.g., from `Searchlights.structure`,
        `Transform.from_searchlights` or `Transform.load_structure`. It is
        copied, not modified. If None, the structure cached on ``sls`` is
        used when ``sls`` is a `Searchlights` object and the transform is
        square, and otherwise one is built from the searchlights.
    n_jobs : int, default=1
        The number of threads fitting local models, following joblib
        conventions; -1 uses all CPUs. The local transforms are combined in a
        fixed order, so the result does not depend on it.
    max_blas_threads : int or None, default=1
        The maximum number of BLAS threads while fitting, whatever
        ``n_jobs`` is, set with threadpoolctl; None keeps the current
        setting. One thread avoids oversubscription with several ``n_jobs``
        and reduces numerical variation caused by BLAS threading. It has no
        effect on Apple's Accelerate, which only the
        ``VECLIB_MAXIMUM_THREADS`` environment variable controls.

    Returns
    -------
    xfm : Transform
        The transform, of shape ``(n_features_X, n_features_Y)``, such that
        ``X @ xfm`` approximates ``Y``.
    """
    sl_func = functools.partial(procrustes, reflection=reflection, scaling=scaling)
    return searchlight_hyperalignment(
        X,
        Y,
        sls,
        sl_func,
        sls_out=sls_out,
        weights=weights,
        structure=structure,
        n_jobs=n_jobs,
        max_blas_threads=max_blas_threads,
    )


def searchlight_ridge(
    X,
    Y,
    sls,
    alpha=1e3,
    *,
    solver="eigh",
    sls_out=None,
    weights=None,
    structure=None,
    n_jobs=1,
    max_blas_threads=1,
):
    """Searchlight hyperalignment using ridge regression.

    In each searchlight, the local transform minimizes
    ``||X[:, sl] @ t - Y[:, sl_out]||_F**2 + alpha * ||t||_F**2`` without
    centering, intercept, or scaling, and without dividing ``alpha`` by the
    number of samples. See `searchlight_ridge_path` to fit several values
    of ``alpha`` without repeating the local decompositions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The data to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The target of the alignment.
    sls : Searchlights or list of ndarrays
        Each entry holds the indices of the vertices in a searchlight.
    alpha : float, default=1e3
        The regularization strength, finite and nonnegative.
    solver : {'eigh', 'svd'}, default='eigh'
        Solver of the local ridge regressions; see `ridge_path`.
    sls_out : Searchlights, list of ndarrays or None, default=None
        Searchlights in the space of ``Y``, if different from ``sls``. Must
        have the same length as ``sls``; the ``i``-th searchlights of ``sls``
        and ``sls_out`` are fitted together.
    weights : list of ndarrays or None, default=None
        Weights for combining the local transforms. Entry ``i`` has one
        weight per vertex of the ``i``-th output searchlight. If None, they
        are computed from the output searchlights (``sls``, or ``sls_out``
        when given): distance-based weights for a `Searchlights` object
        with distances, uniform weights otherwise. Computed weights sum to
        1 for each vertex across the searchlights containing it; supplied
        weights are used as they are.
    structure : Transform or None, default=None
        An all-zero transform with the sparsity structure of the
        searchlights, e.g., from `Searchlights.structure`,
        `Transform.from_searchlights` or `Transform.load_structure`. It is
        copied, not modified. If None, the structure cached on ``sls`` is
        used when ``sls`` is a `Searchlights` object and the transform is
        square, and otherwise one is built from the searchlights.
    n_jobs : int, default=1
        The number of threads fitting local models, following joblib
        conventions; -1 uses all CPUs. The local transforms are combined in a
        fixed order, so the result does not depend on it.
    max_blas_threads : int or None, default=1
        The maximum number of BLAS threads while fitting, whatever
        ``n_jobs`` is, set with threadpoolctl; None keeps the current
        setting. One thread avoids oversubscription with several ``n_jobs``
        and reduces numerical variation caused by BLAS threading. It has no
        effect on Apple's Accelerate, which only the
        ``VECLIB_MAXIMUM_THREADS`` environment variable controls.

    Returns
    -------
    xfm : Transform
        The transform, of shape ``(n_features_X, n_features_Y)``, such that
        ``X @ xfm`` approximates ``Y``.
    """
    if np.ndim(alpha) != 0:
        raise ValueError(
            "`alpha` must be a scalar; use `searchlight_ridge_path` for several."
        )
    return searchlight_ridge_path(
        X,
        Y,
        sls,
        [alpha],
        solver=solver,
        sls_out=sls_out,
        weights=weights,
        structure=structure,
        n_jobs=n_jobs,
        max_blas_threads=max_blas_threads,
    )[0]


def searchlight_ridge_path(
    X,
    Y,
    sls,
    alphas,
    *,
    solver="eigh",
    sls_out=None,
    weights=None,
    structure=None,
    n_jobs=1,
    max_blas_threads=1,
):
    """Searchlight ridge transforms for several regularization strengths.

    The local decomposition is reused across regularization strengths,
    avoiding repeated decompositions. Each value of ``alphas`` still gets
    its own transform, so memory grows with their number. See
    `searchlight_ridge` for the local model.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features_X)
        The data to be aligned to ``Y``.
    Y : ndarray of shape (n_samples, n_features_Y)
        The target of the alignment.
    sls : Searchlights or list of ndarrays
        Each entry holds the indices of the vertices in a searchlight.
    alphas : array-like of shape (n_alphas,)
        Finite nonnegative regularization strengths. Order and duplicates
        are preserved.
    solver : {'eigh', 'svd'}, default='eigh'
        Solver of the local ridge regressions; see `ridge_path`.
    sls_out : Searchlights, list of ndarrays or None, default=None
        Searchlights in the space of ``Y``, if different from ``sls``. Must
        have the same length as ``sls``; the ``i``-th searchlights of ``sls``
        and ``sls_out`` are fitted together.
    weights : list of ndarrays or None, default=None
        Weights for combining the local transforms. Entry ``i`` has one
        weight per vertex of the ``i``-th output searchlight. If None, they
        are computed from the output searchlights (``sls``, or ``sls_out``
        when given): distance-based weights for a `Searchlights` object
        with distances, uniform weights otherwise. Computed weights sum to
        1 for each vertex across the searchlights containing it; supplied
        weights are used as they are.
    structure : Transform or None, default=None
        An all-zero transform with the sparsity structure of the
        searchlights, e.g., from `Searchlights.structure`,
        `Transform.from_searchlights` or `Transform.load_structure`. It is
        copied, not modified. If None, the structure cached on ``sls`` is
        used when ``sls`` is a `Searchlights` object and the transform is
        square, and otherwise one is built from the searchlights.
    n_jobs : int, default=1
        The number of threads fitting local models, following joblib
        conventions; -1 uses all CPUs. The local transforms are combined in a
        fixed order, so the result does not depend on it.
    max_blas_threads : int or None, default=1
        The maximum number of BLAS threads while fitting, whatever
        ``n_jobs`` is, set with threadpoolctl; None keeps the current
        setting. One thread avoids oversubscription with several ``n_jobs``
        and reduces numerical variation caused by BLAS threading. It has no
        effect on Apple's Accelerate, which only the
        ``VECLIB_MAXIMUM_THREADS`` environment variable controls.

    Returns
    -------
    xfms : list of Transform
        One transform per value of ``alphas``, in the same order, each of
        shape ``(n_features_X, n_features_Y)`` such that ``X @ xfms[i]``
        approximates ``Y`` with regularization ``alphas[i]``.
    """
    alphas = _check_alphas(alphas)
    if solver not in ("eigh", "svd"):
        raise ValueError(f"`solver` must be 'eigh' or 'svd', got {solver!r}.")
    sl_func = functools.partial(ridge_path, alphas=alphas, solver=solver)
    return _searchlight_fit(
        X,
        Y,
        sls,
        sl_func,
        sls_out=sls_out,
        weights=weights,
        structure=structure,
        n_models=len(alphas),
        n_jobs=n_jobs,
        max_blas_threads=max_blas_threads,
    )


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
