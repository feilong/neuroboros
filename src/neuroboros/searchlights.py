import os

import numpy as np

from .io import core_dataset
from .spaces import get_mask


def load_npz(npz_fn):
    npz = np.load(npz_fn)
    sls = np.array_split(npz["concatenated_searchlights"], npz["boundaries"])
    dists = np.array_split(npz["concatenated_distances"], npz["boundaries"])
    return sls, dists


def _concatenate(arrays, dtype):
    if len(arrays):
        return np.concatenate(arrays)
    return np.zeros(0, dtype=dtype)


def _split(packed, offsets):
    """Split a concatenated array into views.

    ``offsets`` has one more entry than the number of pieces, and piece
    ``i`` is ``packed[offsets[i]:offsets[i + 1]]``.
    """
    return tuple(packed[a:b] for a, b in zip(offsets[:-1], offsets[1:]))


def _freeze(arr):
    """Make an array read-only and return it."""
    arr.flags.writeable = False
    return arr


def _freeze_structure(obj):
    """Make the values of the cached structure read-only, if cached."""
    if obj._structure is not None:
        _freeze(obj._structure.internal_matrix.data)


def _unpack(cls, packed, offsets, state):
    """Rebuild a `Searchlights` object from its pickled form."""
    obj = cls.__new__(cls)
    obj._packed = {k: _freeze(v) for k, v in packed.items()}
    obj._offsets = offsets
    list.extend(obj, obj._split(packed["indices"]))
    obj._dists = obj._split(packed["dists"]) if "dists" in packed else None
    obj._weights = obj._split(packed["weights"]) if "weights" in packed else None
    obj.__dict__.update(state)
    _freeze_structure(obj)
    return obj


def _unpack_plain(cls, items, state):
    """Rebuild an `_UnpackedSearchlights` object from its pickled form."""
    obj = cls.__new__(cls)
    list.extend(obj, (_freeze(sl) for sl in items))
    obj.__dict__.update(state)
    for arrays in (obj._dists, obj._weights):
        if arrays is not None:
            for arr in arrays:
                _freeze(arr)
    _freeze_structure(obj)
    return obj


def _immutable(self, *args, **kwargs):
    raise TypeError(f"{type(self).__name__} objects are immutable.")


class Searchlights(list):
    """Searchlights of a surface space, with their metadata.

    A list with one integer array per searchlight holding the indices of the
    vertices in that searchlight. It behaves like a plain list of arrays, so
    ``len(sls)``, ``sls[i]`` and ``for sl in sls`` work. It additionally
    carries the distances to the searchlight centers and the number of
    vertices in the space, which are needed to combine searchlight results.

    Parameters
    ----------
    sls : list of ndarray
        Each entry is an ndarray of integers, which are the indices of the
        vertices in a searchlight.
    dists : list of ndarray or None, default=None
        Each entry is an ndarray of float numbers, which are the distances
        between vertices in a searchlight and the center of the searchlight,
        in the same order as the corresponding entry of ``sls``.
    radius : int or float or None, default=None
        The searchlight radius in mm.
    nv : int or None, default=None
        The number of vertices in the space, i.e., the number of columns of
        the data arrays the searchlights apply to. If None, it is inferred
        from the largest searchlight index.
    space : str or None, default=None
        The space in which the searchlights are defined.
    center_space : str or None, default=None
        The space in which the searchlight centers are defined.
    lr : {'l', 'r', 'lr'} or None, default=None
        The hemisphere(s).

    Notes
    -----
    The indices, distances and weights are each stored as one concatenated
    array, the entries of the list are views into them, and pickling
    transfers the concatenated arrays and the offsets of the searchlights
    within them, in the same way as the searchlight files on disk. This is
    much cheaper than pickling a list of arrays when passing the
    searchlights to joblib workers, because joblib shares large arrays
    between workers through memory-mapped files but copies small ones for
    every task.

    The object is immutable: the list cannot be modified, its arrays are
    read-only, and its attributes cannot be reassigned. This keeps the cached
    weights and structure valid and makes pickled copies, such as those sent
    to joblib workers, identical to the original.

    `weights` and `structure` cache their results on the object, and the
    cached results are pickled together with it. The sparsity structure is
    expensive to build (about 20 s for ``onavg-ico64`` searchlights with
    ``onavg-ico32`` centers), so build it once in the parent process before
    passing the searchlights to joblib workers. With joblib's default
    ``loky`` backend, the large arrays of the structure are then written to
    a memory-mapped file once and shared by all workers.
    """

    # Whether to use the concatenated storage described above. The
    # alternative is kept in `_UnpackedSearchlights` for comparison.
    _packed_storage = True

    def __init__(
        self,
        sls,
        dists=None,
        radius=None,
        nv=None,
        space=None,
        center_space=None,
        lr=None,
    ):
        sls = [np.asarray(sl) for sl in sls]
        if dists is not None:
            dists = [np.asarray(d) for d in dists]
            if len(dists) != len(sls):
                raise ValueError(
                    f"`dists` has {len(dists)} entries but `sls` has {len(sls)}."
                )
            if any(len(d) != len(sl) for d, sl in zip(dists, sls)):
                raise ValueError("Entries of `dists` and `sls` differ in length.")
        if self._packed_storage:
            self._offsets = np.zeros(len(sls) + 1, dtype=np.int64)
            np.cumsum([len(sl) for sl in sls], out=self._offsets[1:])
            self._packed = {"indices": _freeze(_concatenate(sls, np.int64))}
            super().__init__(self._split(self._packed["indices"]))
            if dists is not None:
                self._packed["dists"] = _freeze(_concatenate(dists, np.float64))
                dists = self._split(self._packed["dists"])
        else:
            self._offsets = None
            self._packed = None
            super().__init__(_freeze(sl.copy()) for sl in sls)
            if dists is not None:
                dists = tuple(_freeze(d.copy()) for d in dists)
        self._dists = dists
        self._radius = radius
        if nv is None:
            nv = max((int(sl.max()) + 1 for sl in self if len(sl)), default=0)
        self._nv = int(nv)
        self._space = space
        self._center_space = center_space
        self._lr = lr
        self._weights = None
        self._structure = None

    # The list itself cannot be modified.
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable
    append = extend = insert = pop = remove = clear = reverse = sort = _immutable

    @property
    def dists(self):
        """Distances of the vertices to the searchlight centers, or None."""
        return self._dists

    @property
    def radius(self):
        """The searchlight radius in mm."""
        return self._radius

    @property
    def nv(self):
        """The number of vertices in the space."""
        return self._nv

    @property
    def space(self):
        """The space in which the searchlights are defined."""
        return self._space

    @property
    def center_space(self):
        """The space in which the searchlight centers are defined."""
        return self._center_space

    @property
    def lr(self):
        """The hemisphere(s)."""
        return self._lr

    def _split(self, packed):
        """Split a concatenated array into a tuple of read-only views."""
        return _split(packed, self._offsets)

    def __reduce_ex__(self, protocol):
        state = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_packed", "_offsets", "_dists", "_weights")
        }
        return (_unpack, (type(self), self._packed, self._offsets, state))

    def __repr__(self):
        return (
            f"{type(self).__name__}(n={len(self)}, radius={self.radius!r}, "
            f"nv={self.nv}, space={self.space!r}, "
            f"center_space={self.center_space!r}, lr={self.lr!r})"
        )

    def weights(self):
        """Weights for combining searchlight results.

        For each searchlight, an array of weights for its vertices. The
        weights of a vertex sum to 1 across the searchlights containing it.
        They decrease linearly with the distance to the searchlight center
        when ``dists`` is available, and are uniform otherwise. The result
        is cached.

        Returns
        -------
        weights : tuple of ndarray
        """
        if self._weights is None:
            from .ha.searchlight import compute_searchlight_weights

            if self.dists is None:
                weights = compute_searchlight_weights(self)
            else:
                if self.radius is None:
                    raise ValueError(
                        "`radius` is required for distance-based weights."
                    )
                weights = compute_searchlight_weights(self, self.dists, self.radius)
            if self._packed_storage:
                self._packed["weights"] = _freeze(_concatenate(weights, np.float64))
                weights = self._split(self._packed["weights"])
            else:
                weights = tuple(_freeze(w) for w in weights)
            self._weights = weights
        return self._weights

    def structure(self):
        """The sparsity structure of searchlight transforms.

        An all-zero `Transform` of shape ``(nv, nv)`` whose stored entries
        are the pairs of vertices that share a searchlight. It is the
        starting point for combining local transforms across searchlights,
        and the same object is reused for every participant. The result is
        cached, and its values are read-only; use ``copy()`` to obtain a
        transform that can be filled in.

        Returns
        -------
        structure : Transform
        """
        if self._structure is None:
            from .ha.transform import Transform

            self._structure = Transform.from_searchlights(self, nv=self.nv)
            _freeze(self._structure.internal_matrix.data)
        return self._structure


class _UnpackedSearchlights(Searchlights):
    """`Searchlights` whose entries are independent arrays.

    The object is pickled as an ordinary list of arrays. It behaves the same
    as `Searchlights` and is kept for comparison and testing of the packed
    storage; it is not part of the public API.
    """

    _packed_storage = False

    def __reduce_ex__(self, protocol):
        return (_unpack_plain, (type(self), list(self), self.__dict__))


def load_searchlights(lr, radius, space, center_space=None, **kwargs):
    species = "macaque" if space.startswith("mkavg-") else "human"
    if species == "human":
        group, dist_type, avg_type = "on1031", "dijkstra", "trimmed"
    elif species == "macaque":
        group, dist_type, avg_type = "mk12", "geodesic", "average"
    else:
        raise ValueError(f"Unknown species: {species}")
    group = kwargs.get("group", group)
    dist_type = kwargs.get("dist_type", dist_type)
    avg_type = kwargs.get("avg_type", avg_type)
    if center_space is None:
        center_space = space
    npz_fn = os.path.join(
        space,
        "searchlights",
        f"{center_space}_center",
        f"{lr}h",
        f"{group}_{avg_type}",
        f"{dist_type}_{radius}mm.npz",
    )
    sls, dists = core_dataset.get(npz_fn, load_func=load_npz, on_missing="raise")
    return sls, dists


def convert_searchlights(sls, dists, radius, mask, center_mask):
    """
    Convert larger-radius/unmasked searchlights to smaller-radius/masked.

    Parameters
    ----------
    sls : list of ndarray
        Each entry is an ndarray of integers, which are the indices of the
        vertices in a searchlight.
    dists : list of ndarray
        Each entry is an ndarray of float numbers, which are the distances
        between vertices in a searchlight and the center of the
        searchlight. The order of vertices are the same as ``sls``.
    radius : int or float
        The searchlight radius in mm.
    mask : ndarray
        A boolean ndarray indicating whether a vertex belongs to the cortex
        (True) or not (False).
    center_mask : ndarray
        A boolean ndarray indicating whether a searchlight center belongs to
        the cortex (True) or not (False). Mainly used in sparse searchlights
        where the number of searchlights is smaller than the number of
        vertices.

    Returns
    -------
    sls_new : list of ndarray
        The new list of searchlight indices after conversion.
    dists_new : list of ndarray
        The new list of distances to searchlight center after conversion.
    """
    if mask is not False:
        cortical_indices = np.where(mask)[0]
        mapping = np.cumsum(mask) - 1

    sls_new, dists_new = [], []
    for i, (sl, d) in enumerate(zip(sls, dists)):
        if center_mask is False or center_mask[i]:
            m = d <= radius
            if mask is not False:
                m = np.logical_and(np.isin(sl, cortical_indices), m)
                sls_new.append(mapping[sl[m]])
            else:
                sls_new.append(sl[m])
            dists_new.append(d[m])

    return sls_new, dists_new


def get_searchlights(
    lr,
    radius,
    space="onavg-ico32",
    center_space=None,
    mask=True,
    center_mask=None,
    return_dists=False,
    **kwargs,
):
    """
    Get a list of vertex indices for searchlight analysis.

    Parameters
    ----------
    lr : {'l', 'r', 'lr'}
        'l' for left hemisphere, 'r' for right hemisphere, 'lr' for both.
    radius : int or float
        The searchlight radius in mm.
    space : str, default='onavg-ico32'
        The space in which the searchlights are defined. It should match the
        space of the data to be analyzed.
    center_space : str, default=None
        The space in which the searchlight centers are defined. If None, it
        will be the same as `space`.
    mask : bool or ndarray, default=True
        If True, use the default mask for the specified space. If False, no
        mask is applied. If an ndarray, it should be a boolean array indicating
        which vertices are included in the searchlight. The mask should match
        the mask applied to the data to be analyzed.
    center_mask : bool or ndarray, default=None
        If True, use the default mask for the center space. If False, no mask
        is applied. If an ndarray, it should be a boolean array indicating
        which vertices are included in the searchlight center.
    return_dists : bool, default=False
        If True, return the distances of the vertices in the searchlight to
        the center of the searchlight.
    **kwargs : dict
        Additional keyword arguments passed to the searchlight loading function.

    Returns
    -------
    sls : Searchlights
        A list of ndarrays of integers, which are the indices of the vertices
        in each searchlight, with the distances and other metadata as
        attributes (see `Searchlights`).
    dists : tuple of ndarray, optional
        Each entry is an ndarray of float numbers, which are the distances
        between vertices in a searchlight and the center of the searchlight.
        Only returned if `return_dists` is True; also available as
        ``sls.dists``.
    """
    if lr == "lr":
        nv = 0
        if isinstance(mask, np.ndarray):
            raise NotImplementedError
        m = get_mask("l", space, **kwargs)
        nv = m.sum() if mask else m.size
        kwargs.update(
            {
                "radius": radius,
                "space": space,
                "center_space": center_space,
                "mask": mask,
                "center_mask": center_mask,
                "return_dists": True,
            }
        )
        sls_l, dists_l = get_searchlights("l", **kwargs)
        sls_r, dists_r = get_searchlights("r", **kwargs)

        nv_total = sls_l.nv + sls_r.nv
        # Ensure dtype is sufficient using int64
        sls_r = [s.astype(np.int64) + nv for s in sls_r]
        sls = Searchlights(
            sls_l + sls_r,
            dists_l + dists_r,
            radius,
            nv=nv_total,
            space=space,
            center_space=space if center_space is None else center_space,
            lr=lr,
        )

        if return_dists:
            return sls, sls.dists

        return sls

    radius_ = 20
    if radius > radius_:
        raise ValueError(
            f"Searchlight radius {radius}mm is larger than the maximum "
            f"supported radius {radius_}mm. Please use a smaller radius."
        )

    if center_mask is None:
        center_mask = mask

    if not isinstance(mask, np.ndarray) and mask:
        mask = get_mask(lr, space, **kwargs)

    if center_space is None:
        center_space = space
        center_mask = mask
    else:
        if center_mask is True:
            center_mask = get_mask(lr, center_space, **kwargs)

    sls, dists = load_searchlights(
        lr, radius_, space, center_space=center_space, **kwargs
    )
    sls, dists = convert_searchlights(sls, dists, radius, mask, center_mask)

    if isinstance(mask, np.ndarray):
        nv = int(mask.sum())
    else:
        nv = get_mask(lr, space, **kwargs).size
    sls = Searchlights(
        sls, dists, radius, nv=nv, space=space, center_space=center_space, lr=lr
    )

    if return_dists:
        return sls, sls.dists
    return sls
