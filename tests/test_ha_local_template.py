import warnings

import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy.stats import zscore

from neuroboros.ha import local_template
from neuroboros.ha.local_template import (
    pca_decomposition,
    compute_pca_template,
    compute_pca_var1_template,
    compute_pca_var2_template,
    compute_gpa_template,
    compute_procrustes_template,
    compute_template,
)
from neuroboros.linalg.base import svd_pca


def low_rank_dss(rng, ns=3, nt=50, nv=20, rank=5, offset=2.0):
    """Rank-``rank`` data with a nonzero mean, so randomized SVD is exact."""
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
    @pytest.mark.parametrize("solver", ["randomized", "full"])
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

    @pytest.mark.parametrize("solver", ["randomized", "full"])
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

    def test_linalg_error_falls_back_to_svd(self, monkeypatch):
        rng = np.random.default_rng(2)
        dss = low_rank_dss(rng)
        calls = []

        def failing_randomized_svd(*args, **kwargs):
            calls.append(1)
            raise LinAlgError("forced")

        monkeypatch.setattr(local_template, "randomized_svd", failing_randomized_svd)
        scores, loadings = pca_decomposition(
            dss, max_components=5, solver="randomized", normalize_subjects=True
        )
        ref_scores, ref_loadings = pca_decomposition(
            dss, max_components=5, solver="full", normalize_subjects=True
        )
        assert calls == [1]
        np.testing.assert_array_equal(scores, ref_scores)
        np.testing.assert_array_equal(loadings, ref_loadings)

    def test_unknown_solver(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="solver"):
            pca_decomposition(low_rank_dss(rng), solver="other")


class TestTemplates:
    def test_pca_templates_agree_on_identical_subjects(self):
        # Same check as the module's self-test block.
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
