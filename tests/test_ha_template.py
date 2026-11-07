import warnings

import numpy as np
import pytest
from scipy.stats import zscore

import neuroboros as nb
from neuroboros.ha import compute_template, compute_searchlight_template
from neuroboros.ha.template import (
    _gpa_template,
    _pca_decomposition,
    _pca_template,
    _pca_mean_template,
    _pca_geometric_mean_template,
    _procrustes_template,
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
        scores, loadings = _pca_decomposition(dss, max_components=6, solver=solver)
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
        scores, loadings = _pca_decomposition(
            dss, max_components=5, solver=solver, normalize_subjects=False
        )
        scores_adj, loadings_adj = _pca_decomposition(
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
        scores_e, loadings_e = _pca_decomposition(dss, solver="eigh")
        scores_s, loadings_s = _pca_decomposition(dss, solver="svd")
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
        scores, loadings = _pca_decomposition(dss, solver=solver)
        assert scores.shape == (50, 20)
        assert np.isfinite(scores).all() and np.isfinite(loadings).all()
        assert not scores[:, 4:].any() and not loadings[4:].any()
        assert np.abs(loadings[:4]).sum(axis=(1, 2)).all()
        scores, loadings = _pca_decomposition(dss, max_components=3, solver=solver)
        assert scores.shape == (50, 3) and loadings.shape == (3, 2, 10)

    def test_unknown_solver(self):
        rng = np.random.default_rng(3)
        with pytest.raises(ValueError, match="solver"):
            _pca_decomposition(low_rank_dss(rng), solver="other")


class TestTemplates:
    def test_pca_templates_agree_on_identical_subjects(self):
        rng = np.random.default_rng(0)
        dss = zscore(rng.standard_normal((1, 200, 100)), axis=1)
        dss = np.tile(dss, (10, 1, 1))
        scores0 = _pca_template(dss)
        scores1 = _pca_mean_template(dss)
        scores2 = _pca_geometric_mean_template(dss)
        np.testing.assert_allclose(scores0, scores1, atol=1e-7)
        np.testing.assert_allclose(scores0, scores2, atol=1e-7)
        scores3 = _procrustes_template(dss)
        np.testing.assert_allclose(dss[0], scores3, atol=1e-7)
        assert_equal_up_to_sign(scores0[:, :100], svd_pca(scores3), atol=1e-7)

    @pytest.mark.parametrize("func", [_procrustes_template, _gpa_template])
    def test_procrustes_templates_without_zscore(self, func):
        rng = np.random.default_rng(9)
        dss = centered(low_rank_dss(rng))
        tpl = func(dss, zscore_common=False)
        assert tpl.shape == dss.shape[1:]
        assert np.isfinite(tpl).all()
        # The test template of the training data is the template itself.
        tpl2 = func(dss, zscore_common=False, dms2=dss)[1]
        np.testing.assert_allclose(tpl2, tpl, atol=1e-10)

    def test_local_template_pca(self):
        rng = np.random.default_rng(4)
        dss = centered(low_rank_dss(rng))
        sl = np.arange(5, 15)
        components = compute_template(
            dss, sl=sl, kind="pca", max_components=4, common_topography=False
        )
        assert components.shape == (50, 4)
        assert_equal_up_to_sign(
            components, _pca_template(dss, sl=sl, max_components=4), atol=1e-10
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

    def test_local_template_solver(self):
        rng = np.random.default_rng(10)
        dss = centered(low_rank_dss(rng))
        a = compute_template(dss, max_components=4, solver="eigh")
        b = compute_template(dss, max_components=4, solver="svd")
        np.testing.assert_allclose(a, b, atol=1e-8)
        with pytest.raises(ValueError, match="solver"):
            compute_template(dss, solver="typo")
        # Ignored for Procrustes templates.
        np.testing.assert_array_equal(
            compute_template(dss, kind="procrustes", solver="typo"),
            compute_template(dss, kind="procrustes"),
        )

    def test_local_template_procrustes(self):
        rng = np.random.default_rng(5)
        dss = centered(low_rank_dss(rng))
        sl = np.arange(5, 15)
        # By default, Procrustes templates are used as computed.
        tpl = compute_template(dss, sl=sl, kind="procrustes")
        np.testing.assert_allclose(tpl, _procrustes_template(dss, sl=sl), atol=1e-12)
        rotated = compute_template(
            dss, sl=sl, kind="procrustes", common_topography=True
        )
        assert rotated.shape == tpl.shape
        assert not np.allclose(rotated, tpl)

    def test_local_template_unknown_kind(self):
        rng = np.random.default_rng(6)
        with pytest.raises(ValueError, match="'cls'"):
            compute_template(low_rank_dss(rng), kind="cls")

    @pytest.mark.parametrize("kind", ["pca", "procrustes"])
    def test_local_template_warns_when_not_centered(self, kind):
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


def make_searchlights(rng, nv=40, n_sls=15, max_size=9):
    """Unsorted searchlights of varying size, one of them empty."""
    sls = [rng.permutation(nv)[: rng.integers(3, max_size)] for _ in range(n_sls)]
    sls[4] = np.array([], dtype=np.int64)
    return sls


def low_rank_dms(rng, ns=3, nt=50, nv=40, rank=5):
    """Centered rank-``rank`` data, so randomized SVD is exact."""
    X = rng.standard_normal((nt, rank)) @ rng.standard_normal((rank, ns * nv))
    dms = X.reshape(nt, ns, nv).transpose(1, 0, 2)
    return dms - dms.mean(axis=1, keepdims=True)


def reference_template(dms, sls, weights, kind="pca"):
    # PCA templates are mapped to the vertices; Procrustes templates are
    # already in vertex space and are used as they are.
    tpl = np.zeros(dms.shape[1:])
    for sl, w in zip(sls, weights):
        if len(sl):
            local = compute_template(
                dms,
                sl=sl,
                kind=kind,
                max_components=len(sl),
                common_topography=kind == "pca",
            )
            tpl[:, sl] += local * w
    return tpl


class TestSearchlightTemplate:
    def test_distance_weights(self):
        rng = np.random.default_rng(13)
        nv, radius = 40, 10.0
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * radius for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=radius, nv=nv)
        dms = low_rank_dms(rng, nv=nv)
        expected = reference_template(dms, sls, sls.weights())
        tpl = compute_searchlight_template(dms, sls)
        assert tpl.shape == dms.shape[1:]
        np.testing.assert_allclose(tpl, expected, atol=1e-10)
        # Uncovered vertices stay zero; covered ones do not.
        covered = np.zeros(nv, dtype=bool)
        for sl in sls:
            covered[sl] = True
        assert not tpl[:, ~covered].any()
        assert np.abs(tpl[:, covered]).min(axis=0).all()

    @pytest.mark.parametrize("kind", ["pca", "procrustes"])
    def test_uniform_and_supplied_weights(self, kind):
        rng = np.random.default_rng(14)
        nv = 40
        sls = make_searchlights(rng, nv)
        dms = low_rank_dms(rng, nv=nv)
        counts = np.zeros(nv)
        for sl in sls:
            counts[sl] += 1
        uniform = [1.0 / counts[sl] for sl in sls]
        np.testing.assert_allclose(
            compute_searchlight_template(dms, sls, kind=kind),
            reference_template(dms, sls, uniform, kind),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            compute_searchlight_template(dms, sls, kind=kind, solver="svd"),
            reference_template(dms, sls, uniform, kind),
            atol=1e-8,
        )
        weights = [rng.random(len(sl)) * 3 for sl in sls]
        np.testing.assert_allclose(
            compute_searchlight_template(dms, sls, kind=kind, weights=weights),
            reference_template(dms, sls, weights, kind),
            atol=1e-10,
        )

    def test_threads_and_dtype(self):
        rng = np.random.default_rng(15)
        sls = make_searchlights(rng, nv=40)
        dms = low_rank_dms(rng)
        serial = compute_searchlight_template(dms, sls)
        for n_jobs in (2, -1):
            np.testing.assert_array_equal(
                compute_searchlight_template(dms, sls, n_jobs=n_jobs), serial
            )
        # The BLAS thread count may change the rounding, not the result.
        np.testing.assert_allclose(
            compute_searchlight_template(dms, sls, n_jobs=2, max_blas_threads=None),
            serial,
            rtol=1e-9,
        )
        single = compute_searchlight_template(dms.astype(np.float32), sls)
        assert single.dtype == np.float32
        np.testing.assert_allclose(single, serial, rtol=1e-4, atol=1e-4)

    def test_errors(self):
        rng = np.random.default_rng(16)
        nv = 40
        sls = make_searchlights(rng, nv)
        dms = low_rank_dms(rng, nv=nv)
        cases = [
            (dict(n_jobs=0), ValueError, "nonzero"),
            (dict(max_blas_threads=0), ValueError, "max_blas_threads"),
            (
                dict(weights=[np.ones(len(sl)) for sl in sls][:-1]),
                ValueError,
                "one entry",
            ),
            (
                dict(weights=[np.ones(len(sl) + 1) for sl in sls]),
                ValueError,
                "one weight per vertex",
            ),
        ]
        for kwargs, error, match in cases:
            with pytest.raises(error, match=match):
                compute_searchlight_template(dms, sls, **kwargs)
        with pytest.raises(ValueError, match="3D"):
            compute_searchlight_template(dms[0], sls)
        with pytest.raises(ValueError, match="NaN"):
            compute_searchlight_template(
                np.r_[dms[:-1], [np.full(dms.shape[1:], np.nan)]], sls
            )
        with pytest.raises(ValueError, match="defined for 40 vertices"):
            compute_searchlight_template(dms[:, :, :-1], nb.Searchlights(sls, nv=nv))
        with pytest.raises(TypeError):
            compute_searchlight_template(dms, sls, np.ones(nv), 10.0)
