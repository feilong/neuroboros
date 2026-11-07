import numpy as np
import pytest

import neuroboros as nb
from neuroboros.ha import (
    Transform,
    searchlight_hyperalignment,
    searchlight_procrustes,
    searchlight_ridge,
    searchlight_ridge_path,
    searchlight_template,
)
from neuroboros.ha.local_template import compute_template
from neuroboros.ha.procrustes import procrustes
from neuroboros.ha.ridge import ridge

lstsq = lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]


def make_searchlights(rng, nv=40, n_sls=15, max_size=9):
    """Unsorted searchlights of varying size, one of them empty."""
    sls = [rng.permutation(nv)[: rng.integers(3, max_size)] for _ in range(n_sls)]
    sls[4] = np.array([], dtype=np.int64)
    return sls


def reference_fit(X, Y, sls, sl_func, weights, sls_out=None):
    sls_out = sls if sls_out is None else sls_out
    mat = np.zeros((X.shape[1], Y.shape[1]))
    for sl, sl_out, w in zip(sls, sls_out, weights):
        if len(sl) and len(sl_out):
            mat[np.ix_(sl, sl_out)] += sl_func(X[:, sl], Y[:, sl_out]) * w
    return mat


class TestDefaults:
    def test_distance_weights(self):
        # Weights decrease linearly with distance and sum to 1 per vertex.
        rng = np.random.default_rng(0)
        nv, radius = 40, 10.0
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * radius for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=radius, nv=nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))

        raw = [(radius - d) / radius for d in dists]
        total = np.zeros(nv)
        for sl, w in zip(sls, raw):
            total[sl] += w
        weights = [w / total[sl] for sl, w in zip(sls, raw)]
        expected = reference_fit(X, Y, sls, lambda a, b: ridge(a, b, 10.0), weights)

        xfm = searchlight_ridge(X, Y, sls, 10.0)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-12)
        assert not sls.structure().internal_matrix.data.any()
        assert np.shares_memory(
            xfm.internal_matrix.indices, sls.structure().internal_matrix.indices
        )

    def test_uniform_weights_for_plain_lists(self):
        rng = np.random.default_rng(1)
        nv = 40
        sls = make_searchlights(rng, nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        counts = np.zeros(nv)
        for sl in sls:
            counts[sl] += 1
        weights = [1.0 / counts[sl] for sl in sls]
        expected = reference_fit(X, Y, sls, lstsq, weights)
        xfm = searchlight_hyperalignment(X, Y, sls, lstsq)
        assert isinstance(xfm, Transform)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-12)

    def test_supplied_weights_used_as_given(self):
        rng = np.random.default_rng(2)
        nv = 40
        sls = make_searchlights(rng, nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        weights = [rng.random(len(sl)) * 3 for sl in sls]
        expected = reference_fit(X, Y, sls, lstsq, weights)
        xfm = searchlight_hyperalignment(X, Y, sls, lstsq, weights=weights)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-12)

    def test_structure_must_be_zero(self):
        # A fitted transform, e.g., loaded from a file, is not a structure.
        rng = np.random.default_rng(2)
        nv = 40
        sls = make_searchlights(rng, nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        xfm = searchlight_ridge(X, Y, sls)
        with pytest.raises(ValueError, match="all zeros"):
            searchlight_ridge(X, Y, sls, structure=xfm)
        xfm.internal_matrix.data[:] = 0
        again = searchlight_ridge(X, Y, sls, structure=xfm)
        assert again.internal_matrix.data.any()


class TestRidgePath:
    @pytest.mark.parametrize("solver", ["eigh", "svd"])
    def test_matches_augmented_least_squares(self, solver):
        # Independent reference: ridge as least squares on the augmented
        # system, with alpha=0 the minimum-norm solution. Rank-5 data makes
        # the larger searchlights rank deficient.
        rng = np.random.default_rng(12)
        nv = 40
        sls = make_searchlights(rng, nv)
        X = rng.standard_normal((50, 5)) @ rng.standard_normal((5, nv))
        Y = rng.standard_normal((50, nv))
        counts = np.zeros(nv)
        for sl in sls:
            counts[sl] += 1
        weights = [1.0 / counts[sl] for sl in sls]
        alphas = [0.0, 1.0, 50.0]

        def augmented(a, b, alpha):
            n = a.shape[1]
            return lstsq(
                np.vstack([a, np.sqrt(alpha) * np.eye(n)]),
                np.vstack([b, np.zeros((n, b.shape[1]))]),
            )

        xfms = searchlight_ridge_path(X, Y, sls, alphas, solver=solver)
        for alpha, xfm in zip(alphas, xfms):
            expected = reference_fit(
                X, Y, sls, lambda a, b: augmented(a, b, alpha), weights
            )
            np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-9)

    def test_matches_single_fits(self):
        rng = np.random.default_rng(9)
        nv = 40
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * 10 for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=10, nv=nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        alphas = [0.1, 1e3, 0.1]
        xfms = searchlight_ridge_path(X, Y, sls, alphas)
        assert isinstance(xfms, list) and len(xfms) == len(alphas)
        for alpha, xfm in zip(alphas, xfms):
            assert isinstance(xfm, Transform)
            single = searchlight_ridge(X, Y, sls, alpha)
            np.testing.assert_allclose(
                xfm.internal_matrix.data, single.internal_matrix.data, rtol=1e-10
            )
        # Duplicated alphas give identical, independent transforms.
        np.testing.assert_array_equal(
            xfms[0].internal_matrix.data, xfms[2].internal_matrix.data
        )
        assert not np.shares_memory(
            xfms[0].internal_matrix.data, xfms[2].internal_matrix.data
        )
        assert not sls.structure().internal_matrix.data.any()

    def test_rectangular_and_threads(self):
        rng = np.random.default_rng(10)
        sls = make_searchlights(rng, nv=40)
        sls_out = [rng.permutation(25)[: len(sl)] for sl in sls]
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 25))
        alphas = [1.0, 100.0]
        serial = searchlight_ridge_path(X, Y, sls, alphas, sls_out=sls_out)
        threaded = searchlight_ridge_path(
            X, Y, sls, alphas, sls_out=sls_out, n_jobs=2, solver="svd"
        )
        for alpha, a, b in zip(alphas, serial, threaded):
            assert a.shape == (40, 25)
            expected = searchlight_ridge(X, Y, sls, alpha, sls_out=sls_out)
            np.testing.assert_allclose(
                a.internal_matrix.data, expected.internal_matrix.data, rtol=1e-10
            )
            np.testing.assert_allclose(
                b.internal_matrix.data, expected.internal_matrix.data, rtol=1e-8
            )

    def test_errors(self):
        rng = np.random.default_rng(11)
        sls = make_searchlights(rng, nv=40)
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 40))
        for alphas in ([], [[1.0]], [1.0, -1.0], [np.nan], 1.0):
            with pytest.raises(ValueError, match="nonnegative"):
                searchlight_ridge_path(X, Y, sls, alphas)
        with pytest.raises(ValueError, match="solver"):
            searchlight_ridge_path(X, Y, sls, [1.0], solver="typo")
        with pytest.raises(ValueError, match="searchlight_ridge_path"):
            searchlight_ridge(X, Y, sls, [1.0, 2.0])


class TestRectangular:
    def test_same_indices_different_feature_counts(self):
        # Without ``sls_out``, the searchlights index both X and Y, and Y
        # may have more columns than X.
        rng = np.random.default_rng(3)
        sls = make_searchlights(rng, nv=40)
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 55))
        counts = np.zeros(40)
        for sl in sls:
            counts[sl] += 1
        weights = [1.0 / counts[sl] for sl in sls]
        expected = reference_fit(X, Y, sls, lstsq, weights)
        xfm = searchlight_hyperalignment(X, Y, sls, lstsq)
        assert xfm.shape == (40, 55)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-12)

    def test_distinct_output_searchlights(self):
        rng = np.random.default_rng(4)
        sls = make_searchlights(rng, nv=40)
        sls_out = [rng.permutation(25)[: len(sl)] for sl in sls]
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 25))
        counts = np.zeros(25)
        for sl in sls_out:
            counts[sl] += 1
        weights = [1.0 / counts[sl] for sl in sls_out]
        expected = reference_fit(X, Y, sls, lstsq, weights, sls_out)
        xfm = searchlight_hyperalignment(X, Y, sls, lstsq, sls_out=sls_out)
        assert xfm.shape == (40, 25)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), expected, atol=1e-12)
        explicit = Transform.from_searchlights(sls, sls_out, nv=40, nv_out=25)
        xfm2 = searchlight_hyperalignment(
            X, Y, sls, lstsq, sls_out=sls_out, structure=explicit
        )
        np.testing.assert_allclose(xfm2.to_scipy().toarray(), expected, atol=1e-12)


class TestThreads:
    @pytest.mark.parametrize("n_jobs", [2, -1])
    def test_same_result_as_serial(self, n_jobs):
        rng = np.random.default_rng(5)
        nv = 40
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * 10 for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=10, nv=nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        for fit in (
            lambda **kw: searchlight_ridge(X, Y, sls, 10.0, solver="eigh", **kw),
            lambda **kw: searchlight_ridge(X, Y, sls, 10.0, solver="svd", **kw),
            lambda **kw: searchlight_procrustes(X, Y, sls, scaling=True, **kw),
        ):
            serial, threaded = fit(), fit(n_jobs=n_jobs)
            np.testing.assert_array_equal(
                threaded.internal_matrix.data, serial.internal_matrix.data
            )
            assert not np.shares_memory(
                threaded.internal_matrix.data, serial.internal_matrix.data
            )

    @pytest.mark.parametrize("max_blas_threads", [None, 2])
    def test_max_blas_threads(self, max_blas_threads):
        rng = np.random.default_rng(8)
        sls = make_searchlights(rng, nv=40)
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 40))
        default = searchlight_ridge(X, Y, sls, 10.0)
        other = searchlight_ridge(
            X, Y, sls, 10.0, n_jobs=2, max_blas_threads=max_blas_threads
        )
        # The BLAS thread count may change the rounding, not the result.
        np.testing.assert_allclose(
            other.internal_matrix.data, default.internal_matrix.data, rtol=1e-9
        )

    def test_empty_searchlights_contribute_nothing(self):
        rng = np.random.default_rng(6)
        sls = make_searchlights(rng, nv=40)
        X = rng.standard_normal((50, 40))
        Y = rng.standard_normal((50, 40))
        with_empty = searchlight_procrustes(X, Y, sls, n_jobs=2)
        without = searchlight_procrustes(X, Y, [sl for sl in sls if len(sl)])
        np.testing.assert_array_equal(
            with_empty.to_scipy().toarray(), without.to_scipy().toarray()
        )


class TestErrors:
    def test_arguments(self):
        rng = np.random.default_rng(7)
        nv = 40
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * 10 for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=10, nv=nv)
        X = rng.standard_normal((50, nv))
        Y = rng.standard_normal((50, nv))
        cases = [
            (dict(n_jobs=0), ValueError, "nonzero"),
            (dict(n_jobs=1.5), TypeError, "integer"),
            (dict(solver="typo"), ValueError, "solver"),
            (dict(alpha=[1.0, 2.0]), ValueError, "scalar"),
            (dict(weights=sls.weights()[:-1]), ValueError, "one entry per searchlight"),
            (
                dict(weights=[np.ones(len(sl) + 1) for sl in sls]),
                ValueError,
                "one weight per vertex",
            ),
            (dict(structure=sls.structure().to_scipy()), TypeError, "Transform"),
            (dict(sls_out=list(sls)[:-1]), ValueError, "same length"),
            (dict(max_blas_threads=0), ValueError, "max_blas_threads"),
            (dict(max_blas_threads=2.0), TypeError, "max_blas_threads"),
        ]
        for kwargs, error, match in cases:
            with pytest.raises(error, match=match):
                searchlight_ridge(X, Y, sls, **kwargs)
        with pytest.raises(ValueError, match="NaN"):
            searchlight_ridge(np.r_[X[:-1], [np.full(nv, np.nan)]], Y, sls)
        with pytest.raises(ValueError, match="same number of samples"):
            searchlight_ridge(X, Y[:-1], sls)
        with pytest.raises(ValueError, match="defined for 40 vertices"):
            searchlight_ridge(X[:, :-1], Y, sls)
        with pytest.raises(ValueError, match="structure.*shape"):
            searchlight_ridge(X, Y[:, :-1], sls, structure=sls.structure())
        with pytest.raises(ValueError, match="Expected a block of shape"):
            searchlight_hyperalignment(X, Y, sls, lambda a, b: np.zeros((2, 2)))


def low_rank_dms(rng, ns=3, nt=50, nv=40, rank=5):
    """Rank-``rank`` data with a nonzero mean, so randomized SVD is exact."""
    X = rng.standard_normal((nt, rank)) @ rng.standard_normal((rank, ns * nv))
    return (X + 2.0).reshape(nt, ns, nv).transpose(1, 0, 2)


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
                max_npc=len(sl),
                common_topography=kind == "pca",
            )
            tpl[:, sl] += local * w
    return tpl


class TestTemplate:
    def test_distance_weights(self):
        rng = np.random.default_rng(13)
        nv, radius = 40, 10.0
        sls = make_searchlights(rng, nv)
        dists = [rng.random(len(sl)) * radius for sl in sls]
        sls = nb.Searchlights(sls, dists, radius=radius, nv=nv)
        dms = low_rank_dms(rng, nv=nv)
        expected = reference_template(dms, sls, sls.weights())
        tpl = searchlight_template(dms, sls)
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
            searchlight_template(dms, sls, kind=kind),
            reference_template(dms, sls, uniform, kind),
            atol=1e-10,
        )
        weights = [rng.random(len(sl)) * 3 for sl in sls]
        np.testing.assert_allclose(
            searchlight_template(dms, sls, kind=kind, weights=weights),
            reference_template(dms, sls, weights, kind),
            atol=1e-10,
        )

    def test_threads_and_dtype(self):
        rng = np.random.default_rng(15)
        sls = make_searchlights(rng, nv=40)
        dms = low_rank_dms(rng)
        serial = searchlight_template(dms, sls)
        for n_jobs in (2, -1):
            np.testing.assert_array_equal(
                searchlight_template(dms, sls, n_jobs=n_jobs), serial
            )
        # The BLAS thread count may change the rounding, not the result.
        np.testing.assert_allclose(
            searchlight_template(dms, sls, n_jobs=2, max_blas_threads=None),
            serial,
            rtol=1e-9,
        )
        single = searchlight_template(dms.astype(np.float32), sls)
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
                searchlight_template(dms, sls, **kwargs)
        with pytest.raises(ValueError, match="3D"):
            searchlight_template(dms[0], sls)
        with pytest.raises(ValueError, match="NaN"):
            searchlight_template(np.where(dms > 5, np.nan, dms), sls)
        with pytest.raises(ValueError, match="defined for 40 vertices"):
            searchlight_template(dms[:, :, :-1], nb.Searchlights(sls, nv=nv))
        with pytest.raises(TypeError):
            searchlight_template(dms, sls, np.ones(nv), 10.0)
