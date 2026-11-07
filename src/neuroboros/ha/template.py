"""Computing hyperalignment templates from several individuals' data.

`compute_template` computes a template for a set of features, such as the
vertices of a brain region, and `compute_searchlight_template` combines local
templates across searchlights into a template for a whole space. The
HyperBrain templates distributed with neuroboros are loaded with `nb.tpl`
from the top-level `neuroboros.templates` module.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.stats import zscore
from threadpoolctl import threadpool_limits

from ..linalg import safe_svd
from .procrustes import procrustes
from .searchlight import (
    _check_max_blas_threads,
    _check_n_jobs,
    _check_nv,
    _default_weights,
)


def _pca_decomposition(
    dms, *, max_components=None, solver="eigh", normalize_subjects=False
):
    """Decompose the data of several subjects, concatenated along features.

    The decomposition is the SVD of the concatenated data, of shape
    ``(n_samples, n_subjects * n_features)``, computed in double precision.
    Column means are not removed before the decomposition.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    max_components : int or None, default=None
        The maximum number of components, at most the smaller of the number
        of samples and the total number of features across subjects. If
        None, all components are kept.
    solver : {'eigh', 'svd'}, default='eigh'
        ``'eigh'`` decomposes the smaller of the two Gram matrices of the
        concatenated data, which is faster. Its singular values are accurate
        to about the square root of machine precision relative to the
        largest, and components below that are treated as zero. ``'svd'``
        decomposes the concatenated data directly and keeps components
        down to machine precision. Both cutoffs are scaled by the larger
        dimension of the concatenated data, as in ``numpy.linalg.matrix_rank``.
    normalize_subjects : bool, default=False
        Whether to divide the scores by the square root of the number of
        subjects, so that their variance does not grow with it.

    Returns
    -------
    scores : ndarray of shape (n_samples, n_components)
        The principal component scores, i.e., ``U * s``, in decreasing
        order of variance.
    loadings : ndarray of shape (n_components, n_subjects, n_features)
        The loadings of each component on each subject's features, i.e.,
        the rows of ``Vt``. Components with zero variance, which occur
        when the data is rank deficient, have zero loadings.
    """
    ns, nt, nv = dms.shape
    X = dms.transpose(1, 0, 2).reshape(nt, ns * nv).astype(np.float64)
    k = min(X.shape)
    if max_components is not None:
        k = min(k, max_components)
    if solver == "eigh":
        if X.shape[0] <= X.shape[1]:
            w, U = eigh(X @ X.T)
            idx = np.argsort(w)[::-1][:k]
            s = np.sqrt(np.clip(w[idx], 0, None))
            U = U[:, idx]
            nonzero = s > 0
            Vt = np.zeros((k, X.shape[1]))
            Vt[nonzero] = (X.T @ U[:, nonzero] / s[nonzero]).T
        else:
            w, V = eigh(X.T @ X)
            idx = np.argsort(w)[::-1][:k]
            s = np.sqrt(np.clip(w[idx], 0, None))
            Vt = V[:, idx].T
            nonzero = s > 0
            U = np.zeros((X.shape[0], k))
            U[:, nonzero] = X @ Vt[nonzero].T / s[nonzero]
    elif solver == "svd":
        U, s, Vt = safe_svd(X, remove_mean=False)
        U, s, Vt = U[:, :k], s[:k], Vt[:k]
    else:
        raise ValueError(f"`solver` must be 'eigh' or 'svd', got {solver!r}.")
    # Components with negligible variance carry no information; zero them so
    # rank-deficient data gives no spurious loadings. The cutoff is relative
    # to the largest singular value: ``max(X.shape) * eps`` for the SVD, and
    # its square root for the Gram eigendecomposition, whose eigenvalues are
    # accurate to ``eps`` relative to the largest, so their square roots are
    # accurate to ``sqrt(eps)``. The solvers may therefore differ on
    # components between the two cutoffs.
    rel = max(X.shape) * np.finfo(float).eps
    tol = s.max(initial=0) * (np.sqrt(rel) if solver == "eigh" else rel)
    null = s <= tol
    s[null] = 0
    Vt[null] = 0
    scores = U * s
    if normalize_subjects:
        scores /= np.sqrt(ns)
    loadings = Vt.reshape(-1, ns, nv)
    return scores, loadings


def _pca_template(dms, sl=None, *, max_components=None, solver="eigh"):
    """PCA template: the scores of `_pca_decomposition`, normalized by subjects.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use. If None, all features are used.
    max_components : int or None, default=None
        The maximum number of components; see `_pca_decomposition`.
    solver : {'eigh', 'svd'}, default='eigh'
        See `_pca_decomposition`.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_components)
        The template, with one column per component.
    """
    if sl is not None:
        dms = dms[:, :, sl]
    scores, _ = _pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=True
    )
    return scores


def _pca_mean_template(dms, sl=None, *, max_components=None, solver="eigh"):
    """PCA template with components scaled by their mean loading norm.

    Like `_pca_template`, but instead of normalizing by the number
    of subjects, each component is scaled by the arithmetic mean across
    subjects of the norm of its loadings on the subject's features.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use. If None, all features are used.
    max_components : int or None, default=None
        The maximum number of components; see `_pca_decomposition`.
    solver : {'eigh', 'svd'}, default='eigh'
        See `_pca_decomposition`.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_components)
        The template, with one column per component.
    """
    if sl is not None:
        dms = dms[:, :, sl]
    scores, loadings = _pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=False
    )
    w = np.sqrt(np.sum(loadings**2, axis=2)).mean(axis=1)
    return scores * w[np.newaxis]


def _pca_geometric_mean_template(dms, sl=None, *, max_components=None, solver="eigh"):
    """PCA template with components scaled by their geometric mean loading norm.

    Like `_pca_mean_template`, with the geometric instead of the
    arithmetic mean across subjects of the loading norms.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use. If None, all features are used.
    max_components : int or None, default=None
        The maximum number of components; see `_pca_decomposition`.
    solver : {'eigh', 'svd'}, default='eigh'
        See `_pca_decomposition`.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_components)
        The template, with one column per component.
    """
    if sl is not None:
        dms = dms[:, :, sl]
    scores, loadings = _pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=False
    )
    # Zero-variance components have zero loadings; their scale is zero too.
    with np.errstate(divide="ignore"):
        w = np.exp(0.5 * np.log(np.sum(loadings**2, axis=2)).mean(axis=1))
    return scores * w[np.newaxis]


def _gpa_template(
    dms,
    sl=None,
    *,
    reflection=True,
    scaling=False,
    zscore_common=True,
    n_refinement_iters=1,
    dms2=None,
):
    """Template by generalized Procrustes analysis, starting from the mean.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use. If None, all features are used.
    reflection, scaling : bool
        Whether the Procrustes transforms may include reflections and a
        scale factor; see `procrustes`.
    zscore_common : bool, default=True
        Whether to z-score the aligned data and the template.
    n_refinement_iters : int, default=1
        The number of refinement iterations, sometimes called level-2
        iterations, in which each subject is realigned to the average of
        the other aligned subjects.
    dms2 : ndarray or None, default=None
        Test data of the subjects, of shape ``(n_subjects, n_samples2,
        n_features)``. If given, it is aligned with the final transforms
        and a test template is computed from it as well.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_features)
        The template.
    tpl2 : ndarray of shape (n_samples2, n_features)
        The test template, returned only if ``dms2`` is given.
    """
    if sl is not None:
        dms = dms[:, :, sl]

    tpl = np.mean(dms, axis=0)
    if zscore_common:
        tpl = np.nan_to_num(zscore(tpl, axis=0))

    aligned_dms = []
    for dm in dms:
        T = procrustes(dm, tpl, reflection=reflection, scaling=scaling)
        aligned_dm = dm @ T
        if zscore_common:
            aligned_dm = np.nan_to_num(zscore(aligned_dm, axis=0))
        aligned_dms.append(aligned_dm)

    aligned_dms2 = []
    for iteration in range(n_refinement_iters):
        tpl = np.zeros_like(dms[0])
        for dm in aligned_dms:
            tpl += dm
        for i, dm in enumerate(dms):
            reference = (tpl - aligned_dms[i]) / (len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i] @ T)
            aligned_dms[i] = dm @ T

    tpl = np.sum(aligned_dms, axis=0)
    if zscore_common:
        tpl = np.nan_to_num(zscore(tpl, axis=0))
    else:
        tpl /= len(dms)
    if dms2 is not None:
        tpl2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            tpl2 += dm
        if zscore_common:
            tpl2 = np.nan_to_num(zscore(tpl2, axis=0))
        else:
            tpl2 /= len(dms)
        return tpl, tpl2

    return tpl


def _procrustes_template(
    dms,
    sl=None,
    *,
    reflection=True,
    scaling=False,
    zscore_common=True,
    n_refinement_iters=1,
    dms2=None,
):
    """Template by Procrustes alignment, starting from the first subject.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use. If None, all features are used.
    reflection, scaling : bool
        Whether the Procrustes transforms may include reflections and a
        scale factor; see `procrustes`.
    zscore_common : bool, default=True
        Whether to z-score the aligned data and the template.
    n_refinement_iters : int, default=1
        The number of refinement iterations, sometimes called level-2
        iterations, in which each subject is realigned to the average of
        the other aligned subjects.
    dms2 : ndarray or None, default=None
        Test data of the subjects, of shape ``(n_subjects, n_samples2,
        n_features)``. If given, it is aligned with the final transforms
        and a test template is computed from it as well.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_features)
        The template.
    tpl2 : ndarray of shape (n_samples2, n_features)
        The test template, returned only if ``dms2`` is given.
    """
    if sl is not None:
        dms = dms[:, :, sl]
    tpl = dms[0].copy()
    aligned_dms = [dms[0]]
    for dm in dms[1:]:
        T = procrustes(dm, tpl, reflection=reflection, scaling=scaling)
        aligned_dm = dm @ T
        if zscore_common:
            aligned_dm = np.nan_to_num(zscore(aligned_dm, axis=0))
        aligned_dms.append(aligned_dm)
        tpl = (tpl + aligned_dm) * 0.5
        if zscore_common:
            tpl = np.nan_to_num(zscore(tpl, axis=0))

    aligned_dms2 = []
    for iteration in range(n_refinement_iters):
        tpl = np.zeros_like(dms[0])
        for dm in aligned_dms:
            tpl += dm
        for i, dm in enumerate(dms):
            reference = (tpl - aligned_dms[i]) / (len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i] @ T)
            aligned_dms[i] = dm @ T

    tpl = np.sum(aligned_dms, axis=0)
    if zscore_common:
        tpl = np.nan_to_num(zscore(tpl, axis=0))
    else:
        tpl /= len(dms)
    if dms2 is not None:
        tpl2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            tpl2 += dm
        if zscore_common:
            tpl2 = np.nan_to_num(zscore(tpl2, axis=0))
        else:
            tpl2 /= len(dms)
        return tpl, tpl2

    return tpl


def _warn_if_not_centered(dms):
    """Warn when the columns of ``dms`` do not have zero means."""
    # Accumulate in double precision; squares of integer or float32 values
    # may overflow in the input dtype.
    means = dms.mean(axis=1, dtype=np.float64)
    squares = np.einsum("stv,stv->sv", dms, dms, dtype=np.float64)
    rms = np.sqrt(squares / dms.shape[1])
    if (np.abs(means) > 1e-3 * rms).any():
        warnings.warn(
            "The columns of `dms` do not have zero means. Column means are "
            "not removed; center the data of each subject before computing "
            "a template.",
            UserWarning,
            stacklevel=3,
        )


def compute_template(
    dms,
    sl=None,
    *,
    kind="pca",
    max_components=None,
    solver="eigh",
    common_topography=None,
):
    """Compute a template from the data of several subjects.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects.
    sl : ndarray or None, default=None
        Indices of the features to use, e.g., the vertices of a
        searchlight. If None, all features are used.
    kind : {'pca', 'procrustes'}, default='pca'
        ``'pca'`` takes the principal components of the subjects' data
        concatenated along features, scaled so that the variance does not
        grow with the number of subjects. Column means are not removed
        before the decomposition, so the data of each subject should be
        centered; a warning is issued otherwise. ``'procrustes'``, kept for
        backward compatibility, aligns the subjects to each other with
        orthogonal Procrustes, starting from the first subject and refining
        against the average of the others, and z-scores the average of the
        aligned data.
    max_components : int or None, default=None
        The maximum number of principal components for ``'pca'``, at most
        the smaller of the number of samples and the total number of
        features across subjects. If None, all components are kept. Ignored
        for ``'procrustes'``.
    solver : {'eigh', 'svd'}, default='eigh'
        The solver of the decomposition for ``'pca'``. ``'eigh'``
        decomposes the smaller Gram matrix of the concatenated data, which
        is faster; ``'svd'`` decomposes the data directly. Components with
        negligible variance are treated as zero, with a cutoff that is
        larger for ``'eigh'``, so the solvers may differ in those
        components. Ignored for ``'procrustes'``.
    common_topography : bool or None, default=None
        Whether to map the template onto the features by the orthogonal
        Procrustes transform that brings it closest to the subjects' data
        stacked along samples. If None, this is done for ``'pca'``, whose
        columns are otherwise components, and not for ``'procrustes'``,
        whose columns are already features.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_components) or (n_samples, n_features)
        The template, with one column per feature if ``common_topography``
        is True (or None for ``'pca'``), and one per component otherwise.
        For ``'procrustes'`` the columns are always features.
    """
    if kind not in ("pca", "procrustes"):
        raise ValueError(f"`kind` must be 'pca' or 'procrustes', got {kind!r}.")
    if common_topography is None:
        common_topography = kind == "pca"
    dms = np.asarray(dms)
    if sl is not None:
        dms = dms[:, :, sl]
    _warn_if_not_centered(dms)

    if kind == "pca":
        tpl = _pca_template(dms, max_components=max_components, solver=solver)
    else:
        tpl = _procrustes_template(
            dms, reflection=True, scaling=False, zscore_common=True
        )
    if common_topography:
        ns, nt, nv = dms.shape
        T = procrustes(np.tile(tpl, (ns, 1)), dms.reshape(ns * nt, nv))
        tpl = tpl @ T
    return tpl


def compute_searchlight_template(
    dms,
    sls,
    *,
    kind="pca",
    solver="eigh",
    weights=None,
    n_jobs=1,
    max_blas_threads=1,
):
    """Build a template by combining local templates across searchlights.

    In each searchlight, a local template is computed from the data of all
    subjects with `compute_template`. For ``'pca'``, its columns are
    components rather than vertices, so it is mapped back to the
    searchlight's vertices by Procrustes alignment to the subjects' data.
    The local templates are then combined vertex by vertex into one
    template using ``weights``.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
        The data of the subjects, with samples in correspondence across
        subjects. The columns should have zero means; see
        `compute_template`.
    sls : Searchlights or list of ndarrays
        Each entry holds the indices of the vertices in a searchlight.
    kind : {'pca', 'procrustes'}, default='pca'
        The kind of local template; see `compute_template`. ``'pca'``
        keeps at most as many components as the searchlight has vertices.
    solver : {'eigh', 'svd'}, default='eigh'
        The solver of the local decompositions for ``'pca'``; see
        `compute_template`. Ignored for ``'procrustes'``.
    weights : list of ndarrays or None, default=None
        Weights for combining the local templates. Entry ``i`` has one
        weight per vertex of the ``i``-th searchlight. If None, they are
        computed from ``sls``: distance-based weights for a `Searchlights`
        object with distances, uniform weights otherwise. Computed weights
        sum to 1 for each vertex across the searchlights containing it;
        supplied weights are used as they are.
    n_jobs : int, default=1
        The number of threads computing local templates, following joblib
        conventions; -1 uses all CPUs. The local templates are combined in
        a fixed order, so the result does not depend on it.
    max_blas_threads : int or None, default=1
        The maximum number of BLAS threads while computing, whatever
        ``n_jobs`` is, set with threadpoolctl; None keeps the current
        setting. One thread avoids oversubscription with several ``n_jobs``
        and reduces numerical variation caused by BLAS threading. It has no
        effect on Apple's Accelerate, which only the
        ``VECLIB_MAXIMUM_THREADS`` environment variable controls.

    Returns
    -------
    tpl : ndarray of shape (n_samples, n_features)
        The template, with the dtype of ``dms``.
    """
    n_jobs = _check_n_jobs(n_jobs)
    max_blas_threads = _check_max_blas_threads(max_blas_threads)
    dms = np.asarray(dms)
    if dms.ndim != 3:
        raise ValueError(
            "`dms` must be a 3D array of shape (n_subjects, n_samples, n_features)."
        )
    if not np.isfinite(dms).all():
        raise ValueError("`dms` contains NaN or infinite values.")
    _check_nv("sls", sls, dms.shape[2])
    if weights is None:
        weights = _default_weights(sls)
    elif len(weights) != len(sls):
        raise ValueError("`weights` must have one entry per searchlight.")

    def fit(sl, w):
        if len(sl) == 0:
            return None
        w = np.asarray(w)
        if w.shape != (len(sl),):
            raise ValueError(
                "Each entry of `weights` must have one weight per vertex of "
                "the searchlight."
            )
        local = compute_template(
            dms, sl, kind=kind, max_components=len(sl), solver=solver
        )
        return local * w[np.newaxis]

    tpl = np.zeros_like(dms[0])
    tasks = zip(sls, weights)
    with threadpool_limits(limits=max_blas_threads):
        if n_jobs == 1:
            results = (fit(*task) for task in tasks)
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
                delayed(fit)(*task) for task in tasks
            )
        for sl, local in zip(sls, results):
            if local is not None:
                tpl[:, sl] += local
    return tpl
