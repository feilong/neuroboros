import numpy as np
import pytest
from scipy import sparse

from neuroboros.ha.sparse import csc_block_offsets, initialize_sparse_matrix


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
