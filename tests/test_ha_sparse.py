import numpy as np
import pytest
from scipy import sparse

from neuroboros.ha import initialize_sparse_matrix, searchlight_hyperalignment
from neuroboros.ha.sparse import csc_block_offsets


def make_searchlights(rng, nv=60, n_sls=25, size=8):
    return [np.sort(rng.choice(nv, size, replace=False)) for _ in range(n_sls)]


class TestCSCBlockOffsets:
    def test_matches_dense_block(self):
        rng = np.random.default_rng(0)
        sls = make_searchlights(rng)
        mat = initialize_sparse_matrix(sls, nv=60)
        rows, cols = sls[3], sls[3][::-1]
        offsets = csc_block_offsets(mat.indptr, mat.indices, rows, cols)
        block = rng.standard_normal((len(rows), len(cols)))
        mat.data[offsets] += block.T.ravel()
        arr = mat.toarray()
        np.testing.assert_array_equal(arr[np.ix_(rows, cols)], block)
        arr[np.ix_(rows, cols)] = 0
        assert not arr.any(), "entries outside the block must stay zero"

    def test_outside_structure(self):
        rng = np.random.default_rng(1)
        sls = make_searchlights(rng)
        mat = initialize_sparse_matrix(sls, nv=60)
        with pytest.raises(ValueError):
            csc_block_offsets(mat.indptr, mat.indices, np.arange(60), np.arange(60))


class TestSearchlightHyperalignment:
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize("weighted", [True, False])
    def test_sparse_equals_dense(self, dtype, weighted):
        rng = np.random.default_rng(2)
        sls = make_searchlights(rng)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 60))
        weights = [rng.random(len(sl)) for sl in sls] if weighted else None
        sl_func = lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]

        dense = searchlight_hyperalignment(X, Y, sls, sl_func=sl_func, weights=weights)
        mat0 = initialize_sparse_matrix(sls, nv=60, dtype=dtype)
        xfm = searchlight_hyperalignment(
            X, Y, sls, sl_func=sl_func, mat0=mat0, weights=weights
        )

        assert sparse.issparse(xfm)
        assert xfm.dtype == dtype
        np.testing.assert_allclose(xfm.toarray(), dense, rtol=1e-5, atol=1e-6)
        assert mat0.nnz == xfm.nnz
        assert np.all(mat0.data == 0), "mat0 must not be modified"

    def test_rectangular(self):
        rng = np.random.default_rng(3)
        sls_X = make_searchlights(rng, nv=60)
        sls_out = make_searchlights(rng, nv=40, size=5)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 40))
        sl_func = lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]

        dense = searchlight_hyperalignment(
            X, Y, sls_X, sls_out=sls_out, sl_func=sl_func
        )
        structure = sparse.lil_matrix((60, 40))
        for sl_X, sl_Y in zip(sls_X, sls_out):
            structure[np.ix_(sl_X, sl_Y)] = 1.0
        mat0 = structure.tocsr()  # also exercises non-CSC input
        mat0.data[:] = 0
        xfm = searchlight_hyperalignment(
            X, Y, sls_X, sls_out=sls_out, sl_func=sl_func, mat0=mat0
        )
        np.testing.assert_allclose(xfm.toarray(), dense)
