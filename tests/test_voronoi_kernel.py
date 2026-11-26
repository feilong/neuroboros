import sys
import types

import numpy as np
import pytest
from scipy.spatial import ConvexHull

import neuroboros as nb
from neuroboros.surface import voronoi
from neuroboros.surface.voronoi import VoronoiSurface

MODULE = "neuroboros.surface._voronoi_kernel"
A = 1.0 + 2.0**-27


def test_multiply_add_is_not_fused():
    kernel = pytest.importorskip(MODULE)
    # The exact result is 2**-26 + 2**-54, which a fused multiply-add returns.
    assert kernel.multiply_add(A, A, -1.0) == A * A - 1.0 == 2.0**-26
    assert kernel.multiply_add(3.0, 0.5, 0.25) == 1.75


def test_load_kernel(monkeypatch):
    kernel = pytest.importorskip(MODULE)
    assert voronoi._load_kernel() is kernel
    monkeypatch.setenv("NEUROBOROS_VORONOI_KERNEL", "0")
    assert voronoi._load_kernel() is None
    monkeypatch.setenv("NEUROBOROS_VORONOI_KERNEL", "1")
    assert voronoi._load_kernel() is kernel


def test_load_kernel_not_built(monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, None)
    assert voronoi._load_kernel() is None


def test_load_kernel_refuses_old_build(monkeypatch):
    kernel = pytest.importorskip(MODULE)
    for missing in ("multiply_add", "label_samples", "count_labels"):
        names = {"multiply_add", "label_samples", "count_labels"} - {missing}
        old = types.SimpleNamespace(**{name: getattr(kernel, name) for name in names})
        monkeypatch.setitem(sys.modules, MODULE, old)
        with pytest.warns(RuntimeWarning, match="older version"):
            assert voronoi._load_kernel() is None


def test_load_kernel_refuses_fused_multiply_add(monkeypatch):
    fused = types.SimpleNamespace(
        multiply_add=lambda a, b, c: 2.0**-26 + 2.0**-54,
        label_samples=None,
        count_labels=None,
    )
    monkeypatch.setitem(sys.modules, MODULE, fused)
    with pytest.warns(RuntimeWarning, match="fused multiply-add"):
        assert voronoi._load_kernel() is None


@pytest.fixture
def mesh():
    """A convex mesh whose faces list their vertices in every order."""
    rng = np.random.default_rng(0)
    sphere = rng.standard_normal((30, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    faces = ConvexHull(sphere).simplices
    faces = np.array([f[rng.permutation(3)] for f in faces])
    return sphere * rng.uniform(0.8, 1.2, (30, 1)), faces


@pytest.fixture
def square():
    """A flat unit square of two faces, where many scores tie exactly."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    return coords, np.array([[0, 1, 2], [1, 3, 2]])


def _random_partition(surface, n, seed=0):
    rng = np.random.default_rng(seed)
    return surface.partition(
        face_indices=rng.integers(0, len(surface.faces), n),
        weights=rng.dirichlet(np.ones(3), n),
    )


def _kernel_arguments(surface, partitions, face_indices):
    single = {p: surface._single_labels(p._point_set) for p in partitions}
    shape = (len(partitions), len(face_indices), len(surface._ww))
    return [
        surface._ww,
        surface.coords,
        surface.faces,
        surface._node_coords,
        surface._face_nodes,
        face_indices,
        *surface._kernel_arrays(partitions, single),
        np.full(shape, -2, dtype=np.int32),
    ]


def _kernel_labels(surface, partitions, face_indices):
    kernel = pytest.importorskip(MODULE)
    args = _kernel_arguments(surface, partitions, face_indices)
    assert kernel.label_samples(*args) is None
    return args[-1]


def _numpy_labels(surface, partitions, face_indices):
    boundary = surface._boundary_distances(face_indices)
    return np.stack(
        [
            surface._face_labels(
                partition._point_set,
                surface._single_labels(partition._point_set),
                face_indices,
                *boundary,
            )
            for partition in partitions
        ]
    )


@pytest.mark.parametrize("n_div", [1, 2, 8])
@pytest.mark.parametrize("t_div", [1, 2, 8])
def test_label_samples(mesh, n_div, t_div):
    surface = VoronoiSurface(*mesh, edge_divisions=n_div, sample_divisions=t_div)
    partitions = [
        surface.native,
        _random_partition(surface, 3),
        _random_partition(surface, 10),
        _random_partition(surface, 200),
        surface.native,
    ]
    faces = np.arange(len(surface.faces))
    expected = _numpy_labels(surface, partitions, faces)
    labels = _kernel_labels(surface, partitions, faces)
    assert labels.shape == (5, len(faces), t_div**2)
    np.testing.assert_array_equal(labels, expected)
    # Mixed and single faces of the same partition are both covered, and
    # faces with no, one and several points inside.
    single = surface._single_labels(partitions[1]._point_set)
    assert 0 < np.count_nonzero(single < 0) < len(faces)
    assert {0, 1} <= set(np.diff(partitions[2]._point_set.inside_ptr))
    assert np.diff(partitions[3]._point_set.inside_ptr).max() > 2

    # A partition has the same labels alone, and so does any choice of faces.
    for i, partition in enumerate(partitions):
        alone = _kernel_labels(surface, [partition], faces)
        np.testing.assert_array_equal(alone[0], expected[i])
    chosen = faces[[5, 0, 5, 2]]
    np.testing.assert_array_equal(
        _kernel_labels(surface, partitions, chosen), expected[:, [5, 0, 5, 2]]
    )
    assert _kernel_labels(surface, partitions, faces[:0]).shape[1] == 0


@pytest.mark.parametrize("n_div", [1, 2, 4])
@pytest.mark.parametrize("t_div", [2, 4, 8])
def test_label_samples_ties(square, n_div, t_div):
    """Samples that are as far from two candidates take the first."""
    surface = VoronoiSurface(*square, edge_divisions=n_div, sample_divisions=t_div)
    # Points at vertices 1 and 2 of both faces, twice at the center of the
    # square, and a pair of points mirrored across the diagonal in face 0.
    f_indices = np.array([0, 1, 0, 1, 0, 0, 0, 0])
    weights = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0.5, 0.5],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0, 0.5, 0.5],
        ]
    )
    points = surface.partition(face_indices=f_indices, weights=weights)
    partitions = [surface.native, points]
    faces = np.arange(2)
    expected = _numpy_labels(surface, partitions, faces)
    np.testing.assert_array_equal(_kernel_labels(surface, partitions, faces), expected)
    # Of the two points at the center, the second one has no sample.
    assert 7 not in expected[1]
    if t_div > 2:
        assert 4 in expected[1]


def test_label_samples_many_inside(mesh):
    """One face with more points inside than samples, and one with a point."""
    surface = VoronoiSurface(*mesh, sample_divisions=4)
    rng = np.random.default_rng(0)
    f_indices = np.concatenate([np.full(300, 3), [7]])
    weights = rng.dirichlet(np.ones(3), len(f_indices))
    partition = surface.partition(face_indices=f_indices, weights=weights)
    faces = np.arange(len(surface.faces))
    expected = _numpy_labels(surface, [partition], faces)
    np.testing.assert_array_equal(_kernel_labels(surface, [partition], faces), expected)
    assert len(np.unique(expected[0, 3])) > 10


def test_label_samples_onavg():
    """Every face of a real mesh, with its own vertices and two templates."""
    coords, faces = nb.geometry("midthickness", "l", "onavg-ico32")
    sphere = nb.geometry("sphere.reg", "l", "onavg-ico32", vertices_only=True)
    surface = VoronoiSurface(coords, faces, sphere)
    partitions = [surface.native]
    for space in ("onavg-ico8", "onavg-ico32"):
        points = nb.geometry("sphere.reg", "l", space, vertices_only=True)
        partitions.append(surface.partition(points))
    for start in range(0, len(faces), 2048):
        chunk = np.arange(start, min(start + 2048, len(faces)))
        np.testing.assert_array_equal(
            _kernel_labels(surface, partitions, chunk),
            _numpy_labels(surface, partitions, chunk),
        )


def _replace(args, index, value):
    return [value if i == index else arg for i, arg in enumerate(args)]


def test_label_samples_rejects_invalid(mesh):
    kernel = pytest.importorskip(MODULE)
    surface = VoronoiSurface(*mesh, sample_divisions=2)
    faces = np.arange(4)
    partitions = [surface.native, _random_partition(surface, 10)]
    args = _kernel_arguments(surface, partitions, faces)
    kernel.label_samples(*args)
    expected = args[-1].copy()

    with pytest.raises(TypeError):
        kernel.label_samples(*args[:-1])
    for index in range(len(args)):
        array = args[index]
        other = np.float32 if array.dtype == np.float64 else np.float64
        with pytest.raises(TypeError, match="-byte"):
            kernel.label_samples(*_replace(args, index, array.astype(other)))
        with pytest.raises(TypeError, match="-D array"):
            kernel.label_samples(*_replace(args, index, array[np.newaxis]))
        if array.ndim > 1:
            # Not contiguous, or contiguous with another shape.
            with pytest.raises((BufferError, ValueError), match="contiguous|shapes"):
                kernel.label_samples(*_replace(args, index, array.T))
        if index != 5:
            # One row less for the arrays whose last axis has no fixed size.
            smaller = array[:-1] if index in (2, 3, 4, 11) else array[..., :-1]
            with pytest.raises(ValueError, match="shapes"):
                kernel.label_samples(*_replace(args, index, smaller.copy()))
    read_only = args[-1].copy()
    read_only.flags.writeable = False
    with pytest.raises((BufferError, ValueError), match="read-only|writable"):
        kernel.label_samples(*_replace(args, 12, read_only))

    n_faces, n_nodes = len(surface.faces), len(surface._node_coords)
    inside_ptr = args[9]
    decreasing = inside_ptr.copy()
    decreasing[1, 1:3] = [5, 3]
    out_of_range = [
        (5, np.array([0, n_faces, 1, 2])),
        (5, np.array([0, -1, 1, 2])),
        (2, np.full_like(surface.faces, len(surface.coords))),
        (4, np.full_like(surface._face_nodes, n_nodes)),
        (4, np.full_like(surface._face_nodes, -1)),
        (9, inside_ptr + 100),
        (9, inside_ptr - 1),
        (9, decreasing),
    ]
    for index, array in out_of_range:
        with pytest.raises(IndexError, match="out of range"):
            kernel.label_samples(*_replace(args, index, array))
    # The valid call still gives the same labels.
    kernel.label_samples(*args)
    np.testing.assert_array_equal(args[-1], expected)


def test_label_samples_without_partitions(mesh):
    kernel = pytest.importorskip(MODULE)
    surface = VoronoiSurface(*mesh, sample_divisions=2)
    args = _kernel_arguments(surface, [surface.native], np.arange(4))
    for index in (6, 7, 8, 9, 12):
        args[index] = args[index][:0].copy()
    assert kernel.label_samples(*args) is None


def _numpy_counts(first, second, n_second, face_indices, selected):
    """The entries as the NumPy code of ``VoronoiSurface._entries`` counts them."""
    keys = first[selected].astype(np.int64)
    if second is not None:
        keys = keys * n_second + second[selected]
    keys = np.sort(keys, axis=1)
    is_run_start = np.ones(keys.shape, dtype=bool)
    is_run_start[:, 1:] = keys[:, 1:] != keys[:, :-1]
    run_starts = np.flatnonzero(is_run_start)
    return (
        face_indices[selected][run_starts // max(keys.shape[1], 1)],
        keys.ravel()[run_starts],
        np.diff(run_starts, append=keys.size),
    )


def _kernel_counts(first, second, n_second, face_indices, selected, extra=0):
    kernel = pytest.importorskip(MODULE)
    capacity = np.count_nonzero(selected) * first.shape[1] + extra
    out = (
        np.full(capacity, -7, dtype=np.intp),
        np.full(capacity, -7, dtype=np.int64),
        np.full(capacity, -7, dtype=np.int64),
    )
    n = kernel.count_labels(first, second, n_second, face_indices, selected, *out)
    for array in out:
        # Nothing is written after the entries.
        assert np.all(array[n:] == -7)
    return tuple(array[:n] for array in out)


@pytest.mark.parametrize("n_samples", [1, 4, 64, 1024])
@pytest.mark.parametrize("n_labels", [1, 3, 31, 33, 70000])
def test_count_labels(n_samples, n_labels):
    rng = np.random.default_rng(n_samples + n_labels)
    k = 20
    # Labels in runs, as on a face, and scattered labels.
    first = np.repeat(rng.integers(0, n_labels, (k, 4)), -(-n_samples // 4), axis=1)
    first = first[:, :n_samples].astype(np.int32)
    second = rng.integers(0, n_labels, (k, n_samples)).astype(np.int32)
    face_indices = rng.permutation(1000)[:k]
    for selected in [np.ones(k, bool), rng.random(k) < 0.5, np.zeros(k, bool)]:
        for a, b in [(first, None), (second, None), (first, second), (second, second)]:
            expected = _numpy_counts(a, b, n_labels, face_indices, selected)
            counts = _kernel_counts(a, b, n_labels, face_indices, selected, extra=3)
            for x, y in zip(counts, expected):
                assert x.dtype == y.dtype
                np.testing.assert_array_equal(x, y)
            assert counts[2].sum() == np.count_nonzero(selected) * n_samples
            if n_labels == 70000 and b is not None and selected.all():
                # The keys of pairs do not fit in 32 bits.
                assert expected[1].max() > np.iinfo(np.int32).max


def test_count_labels_rejects_invalid():
    kernel = pytest.importorskip(MODULE)
    first = np.zeros((3, 4), dtype=np.int32)
    selected = np.ones(3, dtype=bool)
    faces = np.arange(3)
    out = [np.empty(12, dtype=np.intp), np.empty(12, np.int64), np.empty(12, np.int64)]
    assert kernel.count_labels(first, None, 1, faces, selected, *out) == 3
    assert kernel.count_labels(first, first, 5, faces, selected, *out) == 3
    with pytest.raises(TypeError):
        kernel.count_labels(first.astype(np.int64), None, 1, faces, selected, *out)
    with pytest.raises(TypeError):
        kernel.count_labels(first, None, 1, faces, selected.astype(np.int8), *out)
    with pytest.raises(TypeError):
        kernel.count_labels(first, first[0], 1, faces, selected, *out)
    for args in [
        (first, first[:2], 1, faces, selected, *out),
        (first, None, 1, faces[:2], selected, *out),
        (first, None, 1, faces, selected[:2], *out),
        (first, None, 1, faces, selected, out[0][:11], out[1], out[2]),
        (first, None, 1, faces, selected, out[0], out[1][:11], out[2]),
        (first, None, 1, faces, selected, out[0], out[1], out[2][:11]),
    ]:
        with pytest.raises(ValueError, match="shapes"):
            kernel.count_labels(*args)
    # Room for the selected faces is enough.
    selected[1] = False
    assert (
        kernel.count_labels(first, None, 1, faces, selected, *[x[:8] for x in out]) == 2
    )
