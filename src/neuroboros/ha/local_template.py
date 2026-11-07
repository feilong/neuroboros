import numpy as np
from scipy.linalg import LinAlgError
from scipy.stats import zscore
from sklearn.utils.extmath import randomized_svd

from ..linalg import safe_svd
from .procrustes import procrustes


def pca_decomposition(
    dms, max_components=None, solver="randomized", normalize_subjects=False
):
    """Decompose concatenated data matrices using SVD.

    Column means are not removed before decomposition.

    Parameters
    ----------
    dms : ndarray of shape (n_subjects, n_samples, n_features)
    max_components : integer or None
    solver : {'randomized', 'full'}
        'randomized' uses randomized SVD, 'full' uses full SVD.
    normalize_subjects : bool
        Whether to adjust the variance of the output so that it doesn't increase with the number of subjects.

    Returns
    -------
    scores : ndarray of shape (n_samples, n_components)
    loadings : ndarray of shape (n_components, n_subjects, n_features)
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
        raise NotImplementedError
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
    **kwargs
):
    if sl is not None:
        dms = dms[:, :, sl]

    common_space = np.mean(dms, axis=0)
    if zscore_common:
        common_space = np.nan_to_num(zscore(common_space, axis=0))

    aligned_dms = []
    for dm in dms:
        T = procrustes(dm, common_space, reflection=reflection, scaling=scaling)
        aligned_dm = dm.dot(T)
        if zscore_common:
            aligned_dm = np.nan_to_num(zscore(aligned_dm, axis=0))
            aligned_dms.append(aligned_dm)

    aligned_dms2 = []
    for iteration in range(n_refinement_iters):
        common_space = np.zeros_like(dms[0])
        for dm in aligned_dms:
            common_space += dm
        for i, dm in enumerate(dms):
            reference = (common_space - aligned_dms[i]) / float(len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i].dot(T))
            aligned_dms[i] = dm.dot(T)

    common_space = np.sum(aligned_dms, axis=0)
    if zscore_common:
        common_space = np.nan_to_num(zscore(common_space, axis=0))
    else:
        common_space /= float(len(dms))
    if dms2 is not None:
        common_space2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            common_space2 += dm
        if zscore_common:
            common_space2 = np.nan_to_num(zscore(common_space2, axis=0))
        else:
            common_space2 /= float(len(dms))
        return common_space, common_space2

    return common_space


def compute_procrustes_template(
    dms,
    sl=None,
    reflection=True,
    scaling=False,
    zscore_common=True,
    n_refinement_iters=1,
    dms2=None,
    **kwargs
):
    if sl is not None:
        dms = dms[:, :, sl]
    common_space = np.copy(dms[0])
    aligned_dms = [dms[0]]
    for dm in dms[1:]:
        T = procrustes(dm, common_space, reflection=reflection, scaling=scaling)
        aligned_dm = dm.dot(T)
        if zscore_common:
            aligned_dm = np.nan_to_num(zscore(aligned_dm, axis=0))
        aligned_dms.append(aligned_dm)
        common_space = (common_space + aligned_dm) * 0.5
        if zscore_common:
            common_space = np.nan_to_num(zscore(common_space, axis=0))

    aligned_dms2 = []
    for iteration in range(n_refinement_iters):
        common_space = np.zeros_like(dms[0])
        for dm in aligned_dms:
            common_space += dm
        for i, dm in enumerate(dms):
            reference = (common_space - aligned_dms[i]) / float(len(dms) - 1)
            if zscore_common:
                reference = np.nan_to_num(zscore(reference, axis=0))
            T = procrustes(dm, reference, reflection=reflection, scaling=scaling)
            if iteration == n_refinement_iters - 1 and dms2 is not None:
                aligned_dms2.append(dms2[i].dot(T))
            aligned_dms[i] = dm.dot(T)

    # common_space = np.zeros_like(dms[0])
    # for dm in aligned_dms:
    #     common_space += dm
    common_space = np.sum(aligned_dms, axis=0)
    if zscore_common:
        common_space = np.nan_to_num(zscore(common_space, axis=0))
    else:
        common_space /= float(len(dms))
    if dms2 is not None:
        common_space2 = np.zeros_like(dms2[0])
        for dm in aligned_dms2:
            common_space2 += dm
        if zscore_common:
            common_space2 = np.nan_to_num(zscore(common_space2, axis=0))
        else:
            common_space2 /= float(len(dms))
        return common_space, common_space2

    return common_space


def compute_template(
    dms, sl=None, kind="procrustes", max_components=None, common_topography=False
):
    mapping = {
        "pca": compute_pca_template,
        "pcav1": compute_pca_var1_template,
        "pcav2": compute_pca_var2_template,
        "cls": compute_procrustes_template,
    }
    if kind == "procrustes":
        tpl = compute_procrustes_template(
            dms=dms, sl=sl, reflection=True, scaling=False, zscore_common=True
        )
    elif kind in mapping:
        tpl = mapping[kind](dms=dms, sl=sl, max_components=max_components)
    else:
        raise ValueError

    if common_topography:
        if sl is not None:
            dms = dms[:, :, sl]
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
