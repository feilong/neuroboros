import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .sparse import (
    _rcm_order,
    csc_block_offsets,
    initialize_sparse_matrix,
    permute_matrix,
)

DATA_FORMAT = "neuroboros-transform-data"
STRUCTURE_FORMAT = "neuroboros-transform-structure"


class Transform:
    """A sparse linear transform between two vertex spaces.

    The transform maps data of shape ``(n_samples, nv)`` to
    ``(n_samples, nv_out)`` with ``data @ xfm``. Internally the matrix is
    stored in CSC format with the vertices in an order that keeps vertices
    sharing searchlights close together, which makes the product about three
    times faster than in the original vertex order. ``data`` and the result
    are always in the original vertex order.

    Parameters
    ----------
    mat : sparse matrix of shape (nv, nv_out)
        The transform in the internal vertex order.
    order_in : ndarray of shape (nv,)
        ``order_in[i]`` is the original index of the input vertex stored at
        row ``i`` of ``mat``.
    order_out : ndarray of shape (nv_out,) or None, default=None
        Same for the output vertices and the columns of ``mat``. If None,
        ``order_in`` is used for both.

    Attributes
    ----------
    internal_matrix : csc_matrix
        The stored matrix, in the internal vertex order. Use `to_scipy` to
        obtain a matrix in the original vertex order. Transforms with the
        same sparsity structure and vertex ordering can store different
        values in their ``data`` arrays.
    """

    # Let ``ndarray @ Transform`` fall through to ``__rmatmul__``.
    __array_ufunc__ = None

    def __init__(self, mat, order_in, order_out=None):
        mat = sparse.csc_matrix(mat)
        if not mat.has_canonical_format:
            # `block_offsets` relies on sorted, duplicate-free indices.
            mat = mat.copy()
            mat.sum_duplicates()
        self.internal_matrix = mat
        self.order_in = np.asarray(order_in, dtype=np.int64)
        if order_out is None:
            self.order_out = self.order_in
        else:
            self.order_out = np.asarray(order_out, dtype=np.int64)
        if self.internal_matrix.shape != (len(self.order_in), len(self.order_out)):
            raise ValueError(
                f"Matrix shape {self.internal_matrix.shape} does not match the orders "
                f"({len(self.order_in)}, {len(self.order_out)})."
            )
        # Position of each original vertex in the internal order.
        self._inverse_order_in = np.argsort(self.order_in)
        self._inverse_order_out = np.argsort(self.order_out)

    @classmethod
    def from_searchlights(
        cls, sls, sls_out=None, nv=None, nv_out=None, dtype=np.float64
    ):
        """Create an all-zero transform with the sparsity of the searchlights.

        Parameters
        ----------
        sls : list of ndarrays
            Searchlights in the input space.
        sls_out : list of ndarrays or None, default=None
            Searchlights in the output space, if different from ``sls``.
        nv : int or None, default=None
            Number of input vertices. If None, taken from ``sls.nv`` when
            ``sls`` is a `Searchlights` object, and otherwise inferred from
            the largest input searchlight index.
        nv_out : int or None, default=None
            Number of output vertices when ``sls_out`` is provided. If None,
            taken from ``sls_out.nv`` when available, and otherwise inferred
            from the largest output searchlight index.
        dtype : dtype, default=np.float64
            The dtype of the stored values.

        Returns
        -------
        xfm : Transform
        """
        if nv is None:
            nv = getattr(sls, "nv", None)
        if nv_out is None and sls_out is not None:
            nv_out = getattr(sls_out, "nv", None)
        structure = initialize_sparse_matrix(
            sls, nv=nv, sls_out=sls_out, nv_out=nv_out, dtype=dtype
        )
        if sls_out is None:
            order_in = _rcm_order(structure)
            order_out = None
            permuted = permute_matrix(structure, order_in, order_in)
        else:
            order_in = _rcm_order(
                initialize_sparse_matrix(sls, nv=structure.shape[0])
            )
            order_out = _rcm_order(
                initialize_sparse_matrix(sls_out, nv=structure.shape[1])
            )
            permuted = permute_matrix(structure, order_in, order_out)
        return cls(permuted, order_in, order_out)

    @classmethod
    def from_scipy(cls, mat, order_in=None, order_out=None):
        """Create a transform from a sparse matrix in the original vertex order.

        Parameters
        ----------
        mat : sparse matrix of shape (nv, nv_out)
            The transform in the original vertex order, e.g., as saved by
            ``save(fn, full=True)``.
        order_in, order_out : ndarrays or None, default=None
            The internal vertex orders to use. If None, they are computed
            from the sparsity structure of ``mat`` (reverse Cuthill-McKee).

        Returns
        -------
        xfm : Transform
        """
        mat = sparse.csc_matrix(mat)
        if order_in is None and order_out is None:
            if mat.shape[0] == mat.shape[1]:
                order_in = _rcm_order(mat)
            else:
                # Order the bipartite graph of rows and columns together,
                # then split the order into its row and column parts.
                nv = mat.shape[0]
                graph = sparse.bmat([[None, mat], [mat.T, None]], format="csr")
                order = reverse_cuthill_mckee(graph, symmetric_mode=True).astype(
                    np.int64
                )
                order_in = order[order < nv]
                order_out = order[order >= nv] - nv
        elif order_in is None or order_out is None:
            raise ValueError("Specify both `order_in` and `order_out`, or neither.")
        if order_out is None:
            order_out = order_in
        return cls(permute_matrix(mat, order_in, order_out), order_in, order_out)

    @property
    def shape(self):
        return self.internal_matrix.shape

    @property
    def dtype(self):
        return self.internal_matrix.dtype

    @property
    def nnz(self):
        return self.internal_matrix.nnz

    def copy(self):
        """Copy the values. The sparsity structure is shared, not copied."""
        mat = sparse.csc_matrix(
            (
                self.internal_matrix.data.copy(),
                self.internal_matrix.indices,
                self.internal_matrix.indptr,
            ),
            shape=self.shape,
        )
        return Transform(mat, self.order_in, self.order_out)

    def structure_fingerprint(self):
        """A hash of the sparsity structure and vertex orders.

        The hash excludes the stored values and their dtype. `load` checks
        it to ensure that saved values are combined with matching sparsity
        structure and vertex ordering.
        """
        h = hashlib.blake2b(digest_size=16)
        for a in (
            self.order_in,
            self.order_out,
            self.internal_matrix.indptr,
            self.internal_matrix.indices,
        ):
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        h.update(np.asarray(self.shape, dtype=np.int64).tobytes())
        return h.hexdigest()

    def save(self, fn, full=False):
        """Save the transform to an npz file.

        With ``full=False``, loading requires matching sparse indices and
        vertex ordering, supplied through the ``structure`` argument to
        `load`. Use ``full=True`` to save a self-contained sparse matrix in
        the original vertex order.

        Parameters
        ----------
        fn : str
            The file name.
        full : bool, default=False
            If False, save the stored values, vertex orders, shape, and
            structure fingerprint without compression. The sparse indices
            are omitted; save them separately with `save_structure` for use
            with `load`. If True, save the full sparse matrix with
            ``scipy.sparse.save_npz``. `load` returns it as a Transform,
            while ``nb.load`` returns it as a plain sparse matrix.

        Examples
        --------
        Save a structure once and use it when loading saved values:

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> xfm = Transform.from_searchlights([np.array([0, 1])], nv=2)
        >>> with TemporaryDirectory() as directory:
        ...     structure_fn = Path(directory) / "structure.npz"
        ...     transform_fn = Path(directory) / "transform.npz"
        ...     xfm.save_structure(structure_fn)
        ...     xfm.save(transform_fn)
        ...     restored = Transform.load(transform_fn, structure=structure_fn)
        """
        if full:
            sparse.save_npz(fn, self.to_scipy())
        else:
            np.savez(
                fn,
                format=DATA_FORMAT,
                fingerprint=self.structure_fingerprint(),
                shape=np.asarray(self.shape),
                data=self.internal_matrix.data,
                order_in=self.order_in,
                order_out=self.order_out,
            )

    def save_structure(self, fn, compressed=True):
        """Save the sparsity structure and vertex orders to an npz file.

        The file contains the sparse indices, column pointers, matrix
        shape, value dtype, vertex orders, and structure fingerprint. Stored
        values are omitted. One file can be reused for transforms with
        matching sparsity structure and vertex ordering. `load_structure`
        restores an all-zero transform for use as ``structure`` in new fits
        and in `load`.

        Parameters
        ----------
        fn : str
            The file name.
        compressed : bool, default=True
            Whether to compress the file. The indices compress well, to
            less than half the size, at the cost of a slower save.
        """
        savez = np.savez_compressed if compressed else np.savez
        savez(
            fn,
            format=STRUCTURE_FORMAT,
            fingerprint=self.structure_fingerprint(),
            shape=np.asarray(self.shape),
            dtype=str(self.dtype),
            indptr=self.internal_matrix.indptr,
            indices=self.internal_matrix.indices,
            order_in=self.order_in,
            order_out=self.order_out,
        )

    @classmethod
    def load_structure(cls, fn):
        """Load a sparsity structure saved by `save_structure`.

        Returns
        -------
        structure : Transform
            An all-zero transform with the saved sparsity structure and vertex
            orders.
        """
        with np.load(fn) as f:
            if "format" not in f or str(f["format"]) != STRUCTURE_FORMAT:
                raise ValueError(f"{fn} is not a saved Transform structure.")
            shape = tuple(f["shape"])
            indptr, indices = f["indptr"], f["indices"]
            data = np.zeros(len(indices), dtype=str(f["dtype"]))
            mat = sparse.csc_matrix((data, indices, indptr), shape=shape)
            return cls(mat, f["order_in"], f["order_out"])

    @classmethod
    def load(cls, fn, structure=None):
        """Load a transform saved by `save`.

        Parameters
        ----------
        fn : str
            The file name.
        structure : Transform, str or None, default=None
            A Transform with sparsity structure and vertex ordering that
            match the saved values, or the name of a file saved by
            `save_structure`. Required for a file saved with ``full=False``
            and ignored otherwise. The sparse indices and column pointers
            are shared with the structure, not copied.

        Returns
        -------
        xfm : Transform
        """
        with np.load(fn) as f:
            # Files written by ``scipy.sparse.save_npz`` also have a "format"
            # entry, holding the matrix format such as "csc".
            fmt = str(f["format"]) if "format" in f else None
            if fmt == STRUCTURE_FORMAT:
                raise ValueError(f"{fn} is a saved structure, use `load_structure`.")
            if fmt != DATA_FORMAT:
                return cls.from_scipy(sparse.load_npz(fn))
            if structure is None:
                raise ValueError(
                    "This file holds only the values of the transform. Pass "
                    "`structure`, a Transform with matching sparsity structure "
                    "and vertex ordering, or a file saved by `save_structure`."
                )
            if not isinstance(structure, Transform):
                structure = cls.load_structure(structure)
            if str(f["fingerprint"]) != structure.structure_fingerprint():
                raise ValueError(
                    "The sparsity structure or vertex ordering of `structure` "
                    "does not match the saved transform."
                )
            mat = sparse.csc_matrix(
                (
                    f["data"],
                    structure.internal_matrix.indices,
                    structure.internal_matrix.indptr,
                ),
                shape=tuple(f["shape"]),
            )
        return cls(mat, structure.order_in, structure.order_out)

    def block_offsets(self, sl_in, sl_out):
        """Locate a block in ``internal_matrix.data``.

        The block contains input vertices ``sl_in`` and output vertices
        ``sl_out``.

        ``sl_in`` and ``sl_out`` are in the original vertex order. See
        `csc_block_offsets` for the layout of the offsets.
        """
        return csc_block_offsets(
            self.internal_matrix.indptr,
            self.internal_matrix.indices,
            self._inverse_order_in[sl_in],
            self._inverse_order_out[sl_out],
        )

    def add_block(self, sl_in, sl_out, block):
        """Add a dense block to the transform in place.

        Parameters
        ----------
        sl_in, sl_out : ndarrays
            Input and output vertices of the block, in the original order.
        block : ndarray of shape (len(sl_in), len(sl_out))
        """
        block = np.asarray(block)
        self.internal_matrix.data[self.block_offsets(sl_in, sl_out)] += block.T.ravel()

    def apply(self, X):
        """Apply the transform to a data matrix or a single vector.

        Parameters
        ----------
        X : ndarray of shape (n_samples, nv) or (nv,)
            Input data in the original vertex order.

        Returns
        -------
        Y : ndarray of shape (n_samples, nv_out) or (nv_out,)
            Transformed data in the original vertex order. A one-dimensional
            input returns a one-dimensional output.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return self.apply(X[np.newaxis])[0]
        if X.ndim != 2 or X.shape[1] != self.shape[0]:
            raise ValueError(
                f"Expected data of shape (n_samples, {self.shape[0]}), got {X.shape}."
            )
        Y = X[:, self.order_in] @ self.internal_matrix
        return Y[:, self._inverse_order_out]

    def __rmatmul__(self, X):
        return self.apply(X)

    def to_scipy(self):
        """The transform as a CSC matrix in the original vertex order.

        The sparsity structure and dtype are preserved, including stored
        zeros, so an all-zero transform retains its stored-entry locations.
        """
        return permute_matrix(
            self.internal_matrix, self._inverse_order_in, self._inverse_order_out
        )

    def __repr__(self):
        return (
            f"<Transform {self.shape[0]}x{self.shape[1]}, "
            f"{self.nnz} stored, {self.dtype}>"
        )
