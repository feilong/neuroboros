import numpy as np
import pytest
from scipy import sparse

from neuroboros.ha import (
    Transform,
    initialize_sparse_matrix,
    searchlight_hyperalignment,
)
import neuroboros as nb
from neuroboros.ha.sparse import _rcm_order, permute_matrix


def make_searchlights(rng, nv=60, n_sls=25, size=8):
    return [np.sort(rng.choice(nv, size, replace=False)) for _ in range(n_sls)]


lstsq = lambda a, b: np.linalg.lstsq(a, b, rcond=None)[0]


class TestStructure:
    def test_rectangular_structure(self):
        rng = np.random.default_rng(0)
        sls = make_searchlights(rng, nv=60)
        sls_out = make_searchlights(rng, nv=40, size=5)
        mat = initialize_sparse_matrix(sls, nv=60, sls_out=sls_out, nv_out=40)
        expected = np.zeros((60, 40), dtype=bool)
        for sl, sl_Y in zip(sls, sls_out):
            expected[np.ix_(sl, sl_Y)] = True
        structure = mat.copy()
        structure.data[:] = 1
        np.testing.assert_array_equal(structure.toarray().astype(bool), expected)
        assert mat.nnz == expected.sum()

    @pytest.mark.parametrize(
        "factory", [initialize_sparse_matrix, Transform.from_searchlights]
    )
    @pytest.mark.parametrize("nv_out", [3, 4])
    def test_nv_out_requires_sls_out(self, factory, nv_out):
        with pytest.raises(ValueError, match="`nv_out` requires `sls_out`"):
            factory([np.array([0, 1])], nv=3, nv_out=nv_out)

    def test_permute_keeps_structure(self):
        rng = np.random.default_rng(1)
        sls = make_searchlights(rng)
        mat = initialize_sparse_matrix(sls, nv=60)
        order = _rcm_order(mat)
        assert np.array_equal(np.sort(order), np.arange(60))
        permuted = permute_matrix(mat, order, order)
        assert permuted.nnz == mat.nnz, "explicit zeros must be kept"
        assert permuted.has_canonical_format
        ones = mat.copy()
        ones.data[:] = 1
        ones_p = permuted.copy()
        ones_p.data[:] = 1
        np.testing.assert_array_equal(
            ones_p.toarray(), ones.toarray()[np.ix_(order, order)]
        )


class TestTransform:
    def test_combine_and_apply(self):
        rng = np.random.default_rng(2)
        sls = make_searchlights(rng)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 60))
        weights = [rng.random(len(sl)) for sl in sls]

        dense = searchlight_hyperalignment(X, Y, sls, sl_func=lstsq, weights=weights)
        xfm0 = Transform.from_searchlights(sls, nv=60)
        xfm = searchlight_hyperalignment(
            X, Y, sls, sl_func=lstsq, mat0=xfm0, weights=weights
        )

        assert isinstance(xfm, Transform)
        assert xfm.shape == (60, 60)
        assert not xfm0.internal_matrix.data.any(), "mat0 must not be modified"
        np.testing.assert_allclose(xfm.to_scipy().toarray(), dense)
        np.testing.assert_allclose(X @ xfm, X @ dense)
        np.testing.assert_allclose(xfm.apply(X), X @ dense)
        np.testing.assert_allclose(X[0] @ xfm, X[0] @ dense)

    def test_rectangular(self):
        rng = np.random.default_rng(3)
        sls = make_searchlights(rng, nv=60)
        sls_out = make_searchlights(rng, nv=40, size=5)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 40))

        dense = searchlight_hyperalignment(X, Y, sls, sls_out=sls_out, sl_func=lstsq)
        xfm0 = Transform.from_searchlights(sls, sls_out=sls_out, nv=60, nv_out=40)
        xfm = searchlight_hyperalignment(
            X, Y, sls, sls_out=sls_out, sl_func=lstsq, mat0=xfm0
        )
        assert xfm.shape == (60, 40)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), dense)
        np.testing.assert_allclose(X @ xfm, X @ dense)

    def test_to_scipy_keeps_structure_and_dtype(self):
        rng = np.random.default_rng(11)
        sls = make_searchlights(rng)
        xfm = Transform.from_searchlights(sls, nv=60, dtype=np.float32)
        out = xfm.to_scipy()
        assert out.dtype == np.float32
        assert out.nnz == xfm.nnz
        structure = initialize_sparse_matrix(sls, nv=60)
        np.testing.assert_array_equal(out.indptr, structure.indptr)
        np.testing.assert_array_equal(out.indices, structure.indices)

    def test_apply_checks_features(self):
        rng = np.random.default_rng(12)
        xfm = Transform.from_searchlights(make_searchlights(rng), nv=60)
        with pytest.raises(ValueError, match="shape"):
            xfm.apply(rng.standard_normal((5, 61)))
        with pytest.raises(ValueError, match="shape"):
            xfm.apply(rng.standard_normal((5, 59)))
        with pytest.raises(ValueError, match="shape"):
            xfm.apply(rng.standard_normal((2, 5, 60)))

    def test_noncanonical_input(self):
        # Build CSC arrays directly: column 0 is unsorted and repeats row 1.
        data = np.array([1., 2., 3., 5., 6.])
        indices = np.array([1, 0, 1, 1, 0])
        indptr = np.array([0, 3, 5])
        messy = sparse.csc_matrix((data, indices, indptr), shape=(2, 2))
        assert not messy.has_canonical_format
        assert messy.nnz == 5

        xfm = Transform(messy, np.arange(2))
        assert xfm.internal_matrix.has_canonical_format
        assert xfm.nnz == 4
        expected = np.array([[2., 6.], [4., 5.]])
        np.testing.assert_array_equal(xfm.to_scipy().toarray(), expected)
        # Canonicalizing the transform must not modify the supplied matrix.
        np.testing.assert_array_equal(messy.data, [1., 2., 3., 5., 6.])
        np.testing.assert_array_equal(messy.indices, [1, 0, 1, 1, 0])
        np.testing.assert_array_equal(messy.indptr, [0, 3, 5])
        sl = np.arange(2)
        xfm.add_block(sl, sl, np.ones((2, 2)))
        np.testing.assert_array_equal(xfm.to_scipy().toarray(), expected + 1)

    def test_add_block_outside_structure(self):
        rng = np.random.default_rng(4)
        sls = make_searchlights(rng)
        xfm = Transform.from_searchlights(sls, nv=60)
        with pytest.raises(ValueError):
            xfm.add_block(np.arange(60), np.arange(60), np.ones((60, 60)))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            Transform(sparse.eye(4, format="csc"), np.arange(3))


def fitted_transform(rng, nv=60):
    sls = make_searchlights(rng, nv=nv)
    X = rng.standard_normal((30, nv))
    Y = rng.standard_normal((30, nv))
    xfm0 = Transform.from_searchlights(sls, nv=nv)
    return xfm0, searchlight_hyperalignment(X, Y, sls, sl_func=lstsq, mat0=xfm0), X


class TestSaveLoad:
    def test_full(self, tmp_path):
        rng = np.random.default_rng(5)
        xfm0, xfm, X = fitted_transform(rng)
        fn = str(tmp_path / "xfm.npz")
        xfm.save(fn, full=True)
        loaded_mat = nb.load(fn)
        assert sparse.issparse(loaded_mat)
        np.testing.assert_allclose(loaded_mat.toarray(), xfm.to_scipy().toarray())
        loaded = Transform.load(fn)
        assert isinstance(loaded, Transform)
        np.testing.assert_allclose(X @ loaded, X @ xfm)
        np.testing.assert_allclose(
            loaded.to_scipy().toarray(), xfm.to_scipy().toarray()
        )

    @pytest.mark.parametrize("subclass", [False, True])
    def test_nb_save(self, tmp_path, subclass):
        rng = np.random.default_rng(6)
        xfm0, xfm, X = fitted_transform(rng)
        if subclass:
            class SubjectTransform(Transform):
                pass

            xfm = SubjectTransform(
                xfm.internal_matrix, xfm.order_in, xfm.order_out
            )
        fn = str(tmp_path / "xfm.npz")
        nb.save(fn, xfm)
        np.testing.assert_allclose(nb.load(fn).toarray(), xfm.to_scipy().toarray())

    def test_data_only(self, tmp_path):
        rng = np.random.default_rng(7)
        xfm0, xfm, X = fitted_transform(rng)
        fn = str(tmp_path / "xfm.npz")
        xfm.save(fn)
        loaded = Transform.load(fn, structure=xfm0)
        np.testing.assert_array_equal(
            loaded.internal_matrix.data, xfm.internal_matrix.data
        )
        np.testing.assert_allclose(X @ loaded, X @ xfm)
        assert np.shares_memory(
            loaded.internal_matrix.indices, xfm0.internal_matrix.indices
        ), "structure is shared"
        assert (
            loaded.structure_fingerprint()
            == xfm.structure_fingerprint()
            == xfm0.structure_fingerprint()
        )

    def test_data_only_requires_matching_structure(self, tmp_path):
        rng = np.random.default_rng(8)
        xfm0, xfm, X = fitted_transform(rng)
        fn = str(tmp_path / "xfm.npz")
        xfm.save(fn)
        with pytest.raises(ValueError, match="structure"):
            Transform.load(fn)
        other = Transform.from_searchlights(make_searchlights(rng), nv=60)
        with pytest.raises(ValueError, match="does not match"):
            Transform.load(fn, structure=other)

    def test_from_scipy_rectangular(self):
        rng = np.random.default_rng(9)
        sls = make_searchlights(rng, nv=60)
        sls_out = make_searchlights(rng, nv=40, size=5)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 40))
        dense = searchlight_hyperalignment(X, Y, sls, sls_out=sls_out, sl_func=lstsq)
        xfm = Transform.from_scipy(sparse.csc_matrix(dense))
        assert xfm.shape == (60, 40)
        np.testing.assert_allclose(X @ xfm, X @ dense)
        np.testing.assert_allclose(xfm.to_scipy().toarray(), dense)

    @pytest.mark.parametrize("compressed", [True, False])
    def test_structure_file(self, tmp_path, compressed):
        rng = np.random.default_rng(10)
        sls = make_searchlights(rng)
        X = rng.standard_normal((30, 60))
        Y = rng.standard_normal((30, 60))
        xfm0 = Transform.from_searchlights(sls, nv=60)
        structure_fn = str(tmp_path / "structure.npz")
        xfm0.save_structure(structure_fn, compressed=compressed)
        structure = Transform.load_structure(structure_fn)
        assert structure.structure_fingerprint() == xfm0.structure_fingerprint()
        assert structure.dtype == xfm0.dtype
        assert not structure.internal_matrix.data.any()

        xfm = searchlight_hyperalignment(X, Y, sls, sl_func=lstsq, mat0=xfm0)
        xfm_from_file = searchlight_hyperalignment(
            X, Y, sls, sl_func=lstsq, mat0=structure
        )
        np.testing.assert_array_equal(
            xfm_from_file.internal_matrix.data, xfm.internal_matrix.data
        )

        fn = str(tmp_path / "xfm.npz")
        xfm.save(fn)
        loaded = Transform.load(fn, structure=structure_fn)
        np.testing.assert_allclose(X @ loaded, X @ xfm)
        with pytest.raises(ValueError, match="not a saved Transform structure"):
            Transform.load_structure(fn)
