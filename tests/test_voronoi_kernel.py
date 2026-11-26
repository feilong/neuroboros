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


def test_load_kernel_refuses_fused_multiply_add(monkeypatch):
    fused = types.SimpleNamespace(multiply_add=lambda a, b, c: 2.0**-26 + 2.0**-54)
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
