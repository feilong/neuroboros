import numpy as np
import pytest

import neuroboros as nb
from neuroboros.ha import (
    Transform,
    searchlight_hyperalignment,
    searchlight_procrustes,
    searchlight_ridge,
)
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
