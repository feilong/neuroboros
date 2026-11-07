import warnings

import numpy as np
from scipy.linalg import LinAlgError
from scipy.stats import zscore
from sklearn.utils.extmath import randomized_svd

from ..linalg import safe_svd
from .procrustes import procrustes


def pca_decomposition(
    dms, max_components=None, solver="randomized", normalize_subjects=False
):
    """Decompose the data of several subjects, concatenated along features.

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
    solver : {'randomized', 'full'}, default='randomized'
        ``'randomized'`` uses scikit-learn's randomized SVD with a fixed
        seed, falling back to the full SVD if it fails; ``'full'`` uses the
        full SVD.
    normalize_subjects : bool, default=False
        Whether to divide the scores by the square root of the number of
        subjects, so that their variance does not grow with it.

    Returns
    -------
    scores : ndarray of shape (n_samples, n_components)
        The principal component scores, i.e., ``U * s``.
    loadings : ndarray of shape (n_components, n_subjects, n_features)
        The loadings of each component on each subject's features.
    """
    ns, nt, nv = dms.shape
    X = dms.transpose(1, 0, 2).reshape(nt, ns * nv)
    if max_components is not None:
        max_components = min(max_components, min(X.shape[0], X.shape[1]))
    if solver == "randomized":
        try:
            U, s, Vt = randomized_svd(
                X,
                (max_components if max_components is not None else min(X.shape)),
                random_state=0,
            )
        except LinAlgError:
            return pca_decomposition(
                dms,
                max_components=max_components,
                solver="full",
                normalize_subjects=normalize_subjects,
            )
    elif solver == "full":
        U, s, Vt = safe_svd(X, remove_mean=False)
    else:
        raise ValueError(f"`solver` must be 'randomized' or 'full', got {solver!r}.")
    scores = U[:, :max_components] * s[np.newaxis, :max_components]
    if normalize_subjects:
        scores /= np.sqrt(ns)
    loadings = Vt[:max_components].reshape(-1, ns, nv)
    return scores, loadings


def compute_pca_template(dms, sl=None, max_components=None, solver="randomized"):
    if sl is not None:
        dms = dms[:, :, sl]
    scores, loadings = pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=True
    )
    return scores


def compute_pca_var1_template(dms, sl=None, max_components=None, solver="randomized"):
    if sl is not None:
        dms = dms[:, :, sl]
    scores, loadings = pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=False
    )
    w = np.sqrt(np.sum(loadings**2, axis=2)).mean(axis=1)
    scores *= w[np.newaxis]
    return scores


def compute_pca_var2_template(dms, sl=None, max_components=None, solver="randomized"):
    if sl is not None:
        dms = dms[:, :, sl]
    scores, loadings = pca_decomposition(
        dms, max_components=max_components, solver=solver, normalize_subjects=False
    )
    # w = np.sqrt(np.sum(loadings**2, axis=2)).mean(axis=1)
    w = np.exp(0.5 * np.log(np.sum(loadings**2, axis=2)).mean(axis=1))
    scores *= w[np.newaxis]
    return scores


def compute_gpa_template(
    dms,
    sl=None,
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
        aligned_dm = dm.dot(T)
        if zscore_common:
            aligned_dm = np.nan_to_num(zscore(aligned_dm, axis=0))
        aligned_dms.append(aligned_dm)

    aligned_dms2 = []
    for iteration in range(n_refinement_iters):
        tpl = np.zeros_like(dms[0])
        for dm in aligned_dms:
            tpl += dm
        for i, dm in enumerate(dms):
            reference = (tpl - aligned_dms[i]) / float(len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i].dot(T))
            aligned_dms[i] = dm.dot(T)

    tpl = np.sum(aligned_dms, axis=0)
    if zscore_common:
        tpl = np.nan_to_num(zscore(tpl, axis=0))
    else:
        tpl /= float(len(dms))
    if dms2 is not None:
        tpl2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            tpl2 += dm
        if zscore_common:
            tpl2 = np.nan_to_num(zscore(tpl2, axis=0))
        else:
            tpl2 /= float(len(dms))
        return tpl, tpl2

    return tpl


def compute_procrustes_template(
    dms,
    sl=None,
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
    tpl = np.copy(dms[0])
    aligned_dms = [dms[0]]
    for dm in dms[1:]:
        T = procrustes(dm, tpl, reflection=reflection, scaling=scaling)
        aligned_dm = dm.dot(T)
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
            reference = (tpl - aligned_dms[i]) / float(len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i].dot(T))
            aligned_dms[i] = dm.dot(T)

    # tpl = np.zeros_like(dms[0])
    # for dm in aligned_dms:
    #     tpl += dm
    tpl = np.sum(aligned_dms, axis=0)
    if zscore_common:
        tpl = np.nan_to_num(zscore(tpl, axis=0))
    else:
        tpl /= float(len(dms))
    if dms2 is not None:
        tpl2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            tpl2 += dm
        if zscore_common:
            tpl2 = np.nan_to_num(zscore(tpl2, axis=0))
        else:
            tpl2 /= float(len(dms))
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
    dms, sl=None, *, kind="pca", max_components=None, common_topography=None
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
        tpl = compute_pca_template(dms, max_components=max_components)
    else:
        tpl = compute_procrustes_template(
            dms, reflection=True, scaling=False, zscore_common=True
        )
    if common_topography:
        ns, nt, nv = dms.shape
        T = procrustes(np.tile(tpl, (ns, 1)), dms.reshape(ns * nt, nv))
        tpl = tpl @ T
    return tpl


if __name__ == "__main__":
    from scipy.stats import zscore

    from ..linalg.base import svd_pca

    rng = np.random.default_rng(0)
    dms = rng.standard_normal((1, 200, 100))
    dms = zscore(dms, axis=1)
    dms = np.tile(dms, (10, 1, 1))
    scores0 = compute_pca_template(dms)
    scores1 = compute_pca_var1_template(dms)
    scores2 = compute_pca_var2_template(dms)
    np.testing.assert_allclose(scores0, scores1, atol=1e-7)
    np.testing.assert_allclose(scores0, scores2, atol=1e-7)
    scores3 = compute_procrustes_template(dms)
    np.testing.assert_allclose(dms[0], scores3, atol=1e-7)
    scores3 = svd_pca(scores3)
    np.testing.assert_allclose(np.abs(scores0[:, :100]), np.abs(scores3), atol=1e-7)
