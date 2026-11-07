import warnings

import numpy as np
import pytest
from scipy.stats import zscore

from neuroboros.ha.local_template import (
    compute_gpa_template,
    compute_pca_template,
    compute_pca_var1_template,
    compute_pca_var2_template,
    compute_procrustes_template,
    compute_template,
    pca_decomposition,
)
from neuroboros.linalg.base import svd_pca


def low_rank_dss(rng, ns=3, nt=50, nv=20, rank=5, offset=2.0):
    """Rank-``rank + 1`` data with a nonzero mean."""
    X = rng.standard_normal((nt, rank)) @ rng.standard_normal((rank, ns * nv))
    X += offset
    return X.reshape(nt, ns, nv).transpose(1, 0, 2)


def centered(dss):
    """Remove the column means of each subject."""
    return dss - dss.mean(axis=1, keepdims=True)


def uncentered_svd(dss, npc):
    ns, nt, nv = dss.shape
    X = dss.transpose(1, 0, 2).reshape(nt, ns * nv)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :npc] * s[np.newaxis, :npc], Vt[:npc].reshape(-1, ns, nv)


def assert_equal_up_to_sign(a, b, **kwargs):
    np.testing.assert_allclose(np.abs(a), np.abs(b), **kwargs)


class TestPCADecomposition:
    @pytest.mark.parametrize("solver", ["eigh", "svd"])
    def test_nonzero_mean_not_removed(self, solver):
        rng = np.random.default_rng(0)
        dss = low_rank_dss(rng)
        scores, loadings = pca_decomposition(dss, max_components=6, solver=solver)
        ref_scores, ref_loadings = uncentered_svd(dss, 6)
        assert scores.shape == (50, 6) and loadings.shape == (6, 3, 20)
        assert_equal_up_to_sign(scores, ref_scores, atol=1e-8)
        assert_equal_up_to_sign(loadings, ref_loadings, atol=1e-8)
        # Centering would give a different decomposition.
        X = dss.transpose(1, 0, 2).reshape(50, 60)
        centered = svd_pca(X)[:, :6]
        assert not np.allclose(np.abs(scores), np.abs(centered), atol=1e-3)

    @pytest.mark.parametrize("solver", ["eigh", "svd"])
    def test_normalize_subjects(self, solver):
        rng = np.random.default_rng(1)
        dss = low_rank_dss(rng)
        scores, loadings = pca_decomposition(
            dss, max_components=5, solver=solver, normalize_subjects=False
        )
        scores_adj, loadings_adj = pca_decomposition(
            dss, max_components=5, solver=solver, normalize_subjects=True
        )
        np.testing.assert_allclose(scores_adj, scores / np.sqrt(3), atol=1e-8)
        np.testing.assert_array_equal(loadings_adj, loadings)

    @pytest.mark.parametrize("shape", [(3, 50, 20), (2, 50, 10), (6, 30, 20)])
    @pytest.mark.parametrize("full_rank", [True, False])
    def test_solvers_agree(self, shape, full_rank):
        # Tall and wide concatenated data, full rank and rank deficient.
        rng = np.random.default_rng(2)
        ns, nt, nv = shape
        k = min(nt, ns * nv)
        if full_rank:
            dss = rng.standard_normal((ns, nt, nv)) + 2.0
            rank = k
        else:
            dss = low_rank_dss(rng, ns=ns, nt=nt, nv=nv, rank=k - 2)
            rank = k - 1
        scores_e, loadings_e = pca_decomposition(dss, solver="eigh")
        scores_s, loadings_s = pca_decomposition(dss, solver="svd")
        assert scores_e.shape == (nt, k) and loadings_e.shape == (k, ns, nv)
        assert_equal_up_to_sign(scores_e, scores_s, atol=1e-8)
        assert_equal_up_to_sign(loadings_e, loadings_s, atol=1e-8)
        # The nonzero loadings are orthonormal, and they reconstruct the data.
        V = loadings_e.reshape(k, -1)
        V = V[np.abs(V).sum(axis=1) > 0]
        assert len(V) == rank
        np.testing.assert_allclose(V @ V.T, np.eye(rank), atol=1e-10)
        X = dss.transpose(1, 0, 2).reshape(nt, -1)
        np.testing.assert_allclose(scores_e @ loadings_e.reshape(k, -1), X, atol=1e-8)

    @pytest.mark.parametrize("solver", ["eigh", "svd"])
    def test_zero_variance_components(self, solver):
        # Rank-deficient data: the null components have zero scores and
        # loadings instead of arbitrary vectors.
        rng = np.random.default_rng(3)
        dss = low_rank_dss(rng, ns=2, nt=50, nv=10, rank=3)
        scores, loadings = pca_decomposition(dss, solver=solver)
        assert scores.shape == (50, 20)
        assert np.isfinite(scores).all() and np.isfinite(loadings).all()
        assert not scores[:, 4:].any() and not loadings[4:].any()
        assert np.abs(loadings[:4]).sum(axis=(1, 2)).all()
        scores, loadings = pca_decomposition(dss, max_components=3, solver=solver)
        assert scores.shape == (50, 3) and loadings.shape == (3, 2, 10)

    def test_unknown_solver(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="solver"):
            pca_decomposition(low_rank_dss(rng), solver="other")


class TestTemplates:
    def test_pca_templates_agree_on_identical_subjects(self):
        rng = np.random.default_rng(0)
        dss = zscore(rng.standard_normal((1, 200, 100)), axis=1)
        dss = np.tile(dss, (10, 1, 1))
        scores0 = compute_pca_template(dss)
        scores1 = compute_pca_var1_template(dss)
        scores2 = compute_pca_var2_template(dss)
        np.testing.assert_allclose(scores0, scores1, atol=1e-7)
        np.testing.assert_allclose(scores0, scores2, atol=1e-7)
        scores3 = compute_procrustes_template(dss)
        np.testing.assert_allclose(dss[0], scores3, atol=1e-7)
        assert_equal_up_to_sign(scores0[:, :100], svd_pca(scores3), atol=1e-7)

    @pytest.mark.parametrize(
        "func", [compute_procrustes_template, compute_gpa_template]
    )
    def test_procrustes_templates_without_zscore(self, func):
        rng = np.random.default_rng(9)
        dss = centered(low_rank_dss(rng))
        tpl = func(dss, zscore_common=False)
        assert tpl.shape == dss.shape[1:]
        assert np.isfinite(tpl).all()
        # The test template of the training data is the template itself.
        tpl2 = func(dss, zscore_common=False, dms2=dss)[1]
        np.testing.assert_allclose(tpl2, tpl, atol=1e-10)

    def test_compute_template_pca(self):
        rng = np.random.default_rng(4)
        dss = centered(low_rank_dss(rng))
        sl = np.arange(5, 15)
        components = compute_template(
            dss, sl=sl, kind="pca", max_components=4, common_topography=False
        )
        assert components.shape == (50, 4)
        assert_equal_up_to_sign(
            components, compute_pca_template(dss, sl=sl, max_components=4), atol=1e-10
        )
        # By default, PCA templates are mapped onto the features by the
        # orthogonal transform closest to the subjects' data.
        tpl = compute_template(dss, sl=sl, max_components=4)
        assert tpl.shape == (50, 10)
        np.testing.assert_allclose(
            tpl, compute_template(dss, sl=sl, max_components=4, common_topography=True)
        )
        U, s, Vt = np.linalg.svd(
            components.T @ dss[:, :, sl].sum(axis=0), full_matrices=False
        )
        np.testing.assert_allclose(components @ (U @ Vt), tpl, atol=1e-10)

    def test_compute_template_procrustes(self):
        rng = np.random.default_rng(5)
        dss = centered(low_rank_dss(rng))
        sl = np.arange(5, 15)
        # By default, Procrustes templates are used as computed.
        tpl = compute_template(dss, sl=sl, kind="procrustes")
        np.testing.assert_allclose(
            tpl, compute_procrustes_template(dss, sl=sl), atol=1e-12
        )
        rotated = compute_template(
            dss, sl=sl, kind="procrustes", common_topography=True
        )
        assert rotated.shape == tpl.shape
        assert not np.allclose(rotated, tpl)

    def test_compute_template_unknown_kind(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="'cls'"):
            compute_template(low_rank_dss(rng), kind="cls")

    @pytest.mark.parametrize("kind", ["pca", "procrustes"])
    def test_compute_template_warns_when_not_centered(self, kind):
        rng = np.random.default_rng(7)
        dss = low_rank_dss(rng)
        with pytest.warns(UserWarning, match="zero means"):
            compute_template(dss, kind=kind)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_template(centered(dss), kind=kind)
            # Zero columns are centered.
            compute_template(np.zeros_like(dss), kind=kind, common_topography=False)
        # Only the selected features are checked.
        partial = centered(dss)
        partial[:, :, 10:] += 2.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_template(partial, sl=np.arange(5), kind=kind)
        with pytest.warns(UserWarning, match="zero means"):
            compute_template(partial, sl=np.arange(5, 15), kind=kind)

    @pytest.mark.parametrize("dtype, value", [(np.int16, 100), (np.float32, 1e20)])
    def test_centering_check_does_not_overflow(self, dtype, value):
        # Squares that overflow the input dtype must not hide the warning.
        dss = np.full((2, 20, 6), value, dtype=dtype)
        with pytest.warns(UserWarning, match="zero means"):
            compute_template(dss, kind="pca", common_topography=False)
