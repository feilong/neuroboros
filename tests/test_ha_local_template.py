import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy.stats import zscore

from neuroboros.ha import local_template
from neuroboros.ha.local_template import (
    PCA_decomposition,
    compute_PCA_template,
    compute_PCA_var1_template,
    compute_PCA_var2_template,
    compute_procrustes_template,
    compute_template,
)
from neuroboros.linalg.base import svd_pca


def low_rank_dss(rng, ns=3, nt=50, nv=20, rank=5, offset=2.0):
    """Rank-``rank`` data with a nonzero mean, so randomized SVD is exact."""
    X = rng.standard_normal((nt, rank)) @ rng.standard_normal((rank, ns * nv))
    X += offset
    return X.reshape(nt, ns, nv).transpose(1, 0, 2)


def uncentered_svd(dss, npc):
    ns, nt, nv = dss.shape
    X = dss.transpose(1, 0, 2).reshape(nt, ns * nv)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :npc] * s[np.newaxis, :npc], Vt[:npc].reshape(-1, ns, nv)


def assert_equal_up_to_sign(a, b, **kwargs):
    np.testing.assert_allclose(np.abs(a), np.abs(b), **kwargs)


class TestPCADecomposition:
    @pytest.mark.parametrize("flavor", ["sklearn", "svd"])
    def test_nonzero_mean_not_removed(self, flavor):
        rng = np.random.default_rng(0)
        dss = low_rank_dss(rng)
        XX, cc = PCA_decomposition(dss, max_npc=6, flavor=flavor)
        ref_XX, ref_cc = uncentered_svd(dss, 6)
        assert XX.shape == (50, 6) and cc.shape == (6, 3, 20)
        assert_equal_up_to_sign(XX, ref_XX, atol=1e-8)
        assert_equal_up_to_sign(cc, ref_cc, atol=1e-8)
        # Centering would give a different decomposition.
        X = dss.transpose(1, 0, 2).reshape(50, 60)
        centered = svd_pca(X)[:, :6]
        assert not np.allclose(np.abs(XX), np.abs(centered), atol=1e-3)

    @pytest.mark.parametrize("flavor", ["sklearn", "svd"])
    def test_adjust_ns(self, flavor):
        rng = np.random.default_rng(1)
        dss = low_rank_dss(rng)
        XX, cc = PCA_decomposition(dss, max_npc=5, flavor=flavor, adjust_ns=False)
        XX_adj, cc_adj = PCA_decomposition(dss, max_npc=5, flavor=flavor, adjust_ns=True)
        np.testing.assert_allclose(XX_adj, XX / np.sqrt(3), atol=1e-8)
        np.testing.assert_array_equal(cc_adj, cc)

    def test_linalg_error_falls_back_to_svd(self, monkeypatch):
        rng = np.random.default_rng(2)
        dss = low_rank_dss(rng)
        calls = []

        def failing_randomized_svd(*args, **kwargs):
            calls.append(1)
            raise LinAlgError("forced")

        monkeypatch.setattr(local_template, "randomized_svd", failing_randomized_svd)
        XX, cc = PCA_decomposition(dss, max_npc=5, flavor="sklearn", adjust_ns=True)
        ref_XX, ref_cc = PCA_decomposition(dss, max_npc=5, flavor="svd", adjust_ns=True)
        assert calls == [1]
        np.testing.assert_array_equal(XX, ref_XX)
        np.testing.assert_array_equal(cc, ref_cc)

    def test_unknown_flavor(self):
        rng = np.random.default_rng(3)
        with pytest.raises(NotImplementedError):
            PCA_decomposition(low_rank_dss(rng), flavor="other")


class TestTemplates:
    def test_pca_templates_agree_on_identical_subjects(self):
        # Same check as the module's self-test block.
        rng = np.random.default_rng(0)
        dss = zscore(rng.standard_normal((1, 200, 100)), axis=1)
        dss = np.tile(dss, (10, 1, 1))
        XX0 = compute_PCA_template(dss)
        XX1 = compute_PCA_var1_template(dss)
        XX2 = compute_PCA_var2_template(dss)
        np.testing.assert_allclose(XX0, XX1, atol=1e-7)
        np.testing.assert_allclose(XX0, XX2, atol=1e-7)
        XX3 = compute_procrustes_template(dss)
        np.testing.assert_allclose(dss[0], XX3, atol=1e-7)
        assert_equal_up_to_sign(XX0[:, :100], svd_pca(XX3), atol=1e-7)

    @pytest.mark.parametrize("kind", ["pca", "pcav1", "pcav2"])
    def test_compute_template_pca_kinds(self, kind):
        rng = np.random.default_rng(4)
        dss = low_rank_dss(rng)
        sl = np.arange(5, 15)
        tpl = compute_template(dss, sl=sl, kind=kind, max_npc=4)
        assert tpl.shape == (50, 4)
        tpl = compute_template(dss, sl=sl, kind=kind, max_npc=4, common_topography=True)
        assert tpl.shape == (50, 10)
        assert np.isfinite(tpl).all()
