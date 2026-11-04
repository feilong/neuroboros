import os
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee


def _membership_matrix(sls, nv=None):
    """Sparse matrix with one row per searchlight and a 1 for each member."""
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
    indptr = np.zeros(len(sls) + 1, dtype=np.int64)
    np.cumsum([len(sl) for sl in sls], out=indptr[1:])
    data = np.ones(len(indices), dtype=np.float32)
    return sparse.csr_matrix((data, indices, indptr), shape=(len(sls), nv))


def initialize_sparse_matrix(
    sls, nv=None, dtype=np.float64, cache_fn=None, sls_out=None, nv_out=None
):
    """Initialize a sparse matrix based on the searchlights.

    Parameters
    ----------
    sls : list
        Input searchlights. Each entry is an integer array containing the
        vertices of a searchlight.
    nv : int, optional
        Number of input vertices. If None, inferred from the largest
        input searchlight index.
    dtype : dtype, optional
        The dtype of the generated sparse matrix, by default np.float64
    cache_fn : {None, str}
        The file name of the cached sparse matrix. If the file exists, it will
        be loaded to avoid repeated computations. If it is `None`, no cached
        file will be used and the matrix will be computed.
    sls_out : list or None, default=None
        Searchlights of the output space, if different from ``sls``. Must
        have the same length as ``sls``. The matrix then has a shape of
        (nv, nv_out), and entry (i, j) is in the sparsity structure when vertex
        ``i`` of ``sls[k]`` and vertex ``j`` of ``sls_out[k]`` share a
        searchlight ``k``.
    nv_out : int, optional
        Number of output vertices when ``sls_out`` is provided. If None,
        inferred from the largest output searchlight index. Must be None
        when ``sls_out`` is None; the output space then matches the input.

    Returns
    -------
    mat : csc_matrix
        An all-zero matrix with the stored entries needed to accumulate
        searchlight transforms. Shape is (nv, nv_out) when ``sls_out`` is
        provided and (nv, nv) otherwise.

    Raises
    ------
    ValueError
        If ``nv_out`` is provided without ``sls_out``.
    """
    if sls_out is None and nv_out is not None:
        raise ValueError("`nv_out` requires `sls_out`.")
    if cache_fn is not None and os.path.exists(cache_fn):
        return sparse.load_npz(cache_fn)

    # Membership matrix: row k is the indicator of searchlight k. The product
    # B.T @ B is nonzero exactly where two vertices share a searchlight, which
    # is the sparsity structure of a searchlight model.
    B = _membership_matrix(sls, nv)
    if sls_out is None:
        B_Y = B
    else:
        if len(sls_out) != len(sls):
            raise ValueError("`sls` and `sls_out` must have the same length.")
        B_Y = _membership_matrix(sls_out, nv_out)

    mat = (B.T @ B_Y).tocsc()
    mat.sum_duplicates()
    mat.data = np.zeros(mat.nnz, dtype=dtype)
    if cache_fn is not None:
        os.makedirs(os.path.dirname(cache_fn), exist_ok=True)
        sparse.save_npz(cache_fn, mat)
    return mat


def _rcm_order(mat):
    """Compute a reverse Cuthill-McKee vertex order from an adjacency matrix.

    Multiplying data by a sparse transform reads, for every nonzero, the
    data of one vertex. With the reverse Cuthill-McKee ordering of the
    searchlight adjacency, the vertices read while processing one row have
    nearby indices, which keeps the reads in cache and makes the product
    about three times faster than in the original vertex order.

    Parameters
    ----------
    mat : sparse matrix of shape (nv, nv)
        The symmetric sparsity structure of the searchlights, as returned by
        `initialize_sparse_matrix`.

    Returns
    -------
    order : ndarray of shape (nv,)
        ``order[i]`` is the original index of the vertex at position ``i`` in
        the new order.
    """
    return reverse_cuthill_mckee(mat, symmetric_mode=True).astype(np.int64)


def permute_matrix(mat, order_in, order_out):
    """Reorder the rows and columns of a sparse matrix.

    All stored entries are kept, including explicit zeros, so a sparsity
    structure stays complete after reordering.

    Parameters
    ----------
    mat : sparse matrix of shape (n_in, n_out)
    order_in : ndarray of shape (n_in,)
        ``order_in[i]`` is the row of ``mat`` that becomes row ``i``.
    order_out : ndarray of shape (n_out,)
        ``order_out[j]`` is the column of ``mat`` that becomes column ``j``.

    Returns
    -------
    permuted : csc_matrix of shape (n_in, n_out)
        In canonical format (sorted indices, no duplicates).
    """
    permuted = sparse.csc_matrix(mat)[order_in][:, order_out]
    permuted.sum_duplicates()
    return permuted


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
        If an entry of the block is not in the sparsity structure.
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
            "Some entries of the block are not in the sparsity structure of the "
            "matrix. The matrix must be initialized with the same searchlights."
        )
    return pos[loc]
