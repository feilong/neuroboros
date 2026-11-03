import os
import numpy as np
from scipy import sparse


def initialize_sparse_matrix(sls, nv=None, dtype=np.float64, cache_fn=None):
    """Initialize a sparse matrix based on the searchlights.

    Parameters
    ----------
    sls : list
        A list of searchlights. Each entry is an integer array comprising the indices of a searchlight.
    nv : int, optional
        Number of vertices, by default None. The sparse matrix has a shape of (nv, nv).
    dtype : dtype, optional
        The dtype of the generated sparse matrix, by default np.float64
    cache_fn : {None, str}
        The file name of the cached sparse matrix. If the file exists, it will
        be loaded to avoid repeated computations. If it is `None`, no cached
        file will be used and the matrix will be computed.

    Returns
    -------
    mat : csc_matrix
        The initialized sparse matrix which allows fast computations for searchlight algorithms. All of its elements are 0's.
    """
    if cache_fn is not None and os.path.exists(cache_fn):
        return sparse.load_npz(cache_fn)
    if len(sls):
        indices = np.concatenate(sls).astype(np.int64)
    else:
        indices = np.zeros(0, dtype=np.int64)
    if nv is None:
        if not len(indices):
            raise ValueError("`nv` must be specified when `sls` is empty.")
        nv = int(indices.max()) + 1
    # The CSR constructor below does not check bounds, so validate them here.
    if len(indices) and (indices.min() < 0 or indices.max() >= nv):
        raise IndexError(f"Searchlight indices must be in [0, {nv}) for nv={nv}.")

    # Membership matrix: row i is the indicator of searchlight i. The product
    # B.T @ B is nonzero exactly where two vertices share a searchlight, which
    # is the sparsity pattern of a searchlight model.
    indptr = np.zeros(len(sls) + 1, dtype=np.int64)
    np.cumsum([len(sl) for sl in sls], out=indptr[1:])
    data = np.ones(len(indices), dtype=np.float32)
    B = sparse.csr_matrix((data, indices, indptr), shape=(len(sls), nv))

    mat = (B.T @ B).tocsc()
    mat.sum_duplicates()
    mat.data = np.zeros(mat.nnz, dtype=dtype)
    if cache_fn is not None:
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        sparse.save_npz(cache_fn, mat)
    return mat


def csc_block_offsets(indptr, indices, rows, cols):
    """Locate the entries of a block in the ``data`` array of a CSC matrix.

    Only the columns in ``cols`` are searched, so the cost scales with the
    size of the block rather than with the number of nonzeros of the matrix.
    This is what makes it possible to accumulate many small blocks into a
    large sparse matrix without going through ``scipy.sparse`` indexing,
    which validates the entire matrix on every write.

    Parameters
    ----------
    indptr : ndarray of shape (n_cols + 1,)
        The ``indptr`` array of the CSC matrix.
    indices : ndarray of shape (nnz,)
        The ``indices`` array of the CSC matrix. The row indices within each
        column must be sorted and free of duplicates, which is the canonical
        format of ``scipy.sparse`` matrices.
    rows : ndarray of shape (n_rows_block,)
        Row indices of the block. Must not contain duplicates.
    cols : ndarray of shape (n_cols_block,)
        Column indices of the block. Must not contain duplicates.

    Returns
    -------
    offsets : ndarray of shape (n_rows_block * n_cols_block,)
        Offsets into ``data`` of the entries of the block, in column-major
        order. For a dense block ``t`` of shape ``(n_rows_block,
        n_cols_block)``, ``data[offsets] += t.T.ravel()`` adds the block to
        the matrix.

    Raises
    ------
    ValueError
        If an entry of the block is not in the sparsity pattern.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    # Gather the stored entries of the touched columns, in order.
    starts = indptr[cols].astype(np.int64)
    lens = indptr[cols + 1] - starts
    n = int(lens.sum())
    col_rank = np.repeat(np.arange(len(cols)), lens)
    pos = np.arange(n) - np.repeat(np.cumsum(lens) - lens, lens)
    pos += np.repeat(starts, lens)
    stored = indices[pos]

    # Combine (column rank, row) into one sortable key. The stride only has
    # to exceed every row index involved, so it is taken from the gathered
    # entries and the block rather than from the whole matrix.
    stride = int(max(stored.max() if n else 0, rows.max() if len(rows) else 0)) + 1
    # Rows are sorted within each column, so these keys are sorted overall.
    keys = col_rank * stride + stored

    queries = np.repeat(np.arange(len(cols)), len(rows)) * stride
    queries += np.tile(rows, len(cols))
    loc = np.searchsorted(keys, queries)
    found = loc < n
    found[found] = keys[loc[found]] == queries[found]
    if not found.all():
        raise ValueError(
            "Some entries of the block are not in the sparsity pattern of the "
            "matrix. The matrix must be initialized with the same searchlights."
        )
    return pos[loc]
