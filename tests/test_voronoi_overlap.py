import warnings

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from neuroboros.surface import Sphere, Surface, voronoi
from neuroboros.surface.voronoi import VoronoiOverlap


@pytest.fixture
def mesh():
    """A convex mesh whose faces list their vertices in every order."""
    rng = np.random.default_rng(0)
    sphere = rng.standard_normal((30, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    faces = ConvexHull(sphere).simplices
    faces = np.array([f[rng.permutation(3)] for f in faces])
    coords = sphere * rng.uniform(0.8, 1.2, (30, 1))
    return coords, faces, sphere


@pytest.mark.parametrize("n_div", [1, 2, 3, 8])
def test_face_nodes_match_e_mapping(mesh, n_div):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces, n_div=n_div)
    assert ov._face_nodes.shape == (len(faces), 3 * n_div)
    _, e_mapping, graph = voronoi.subdivide_edges(coords, faces, n_div)
    for f, nodes in zip(faces, ov._face_nodes):
        a, b, c = sorted(f)
        expected = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        np.testing.assert_array_equal(nodes, expected)
    assert (ov._graph != graph).nnz == 0


def test_constructor(mesh):
    coords, faces, sphere = mesh
    ov = VoronoiOverlap(coords.astype(np.float32), faces, sphere)
    surf = Surface(coords.astype(np.float32), faces)
    assert ov.names == ("native",)
    assert ov.coords.dtype == np.float64
    np.testing.assert_array_equal(ov.coords, surf.coords)
    np.testing.assert_array_equal(ov.face_areas, surf.face_areas)
    new_coords = voronoi.subdivide_edges(surf.coords, faces, 8)[0]
    np.testing.assert_array_equal(
        ov._node_coords, np.concatenate([surf.coords, new_coords])
    )


@pytest.mark.parametrize("dtype", [np.uint64, np.uint32, np.int32, np.int16])
def test_constructor_face_dtypes(mesh, dtype):
    coords, faces, _ = mesh
    expected = VoronoiOverlap(coords, faces)
    ov = VoronoiOverlap(coords, faces.astype(dtype))
    assert ov.faces.dtype == np.intp
    assert ov._face_nodes.dtype == np.intp
    np.testing.assert_array_equal(ov._face_nodes, expected._face_nodes)
    assert (ov._graph != expected._graph).nnz == 0
    ov._node_coords[ov._face_nodes]


def test_constructor_copies_inputs(mesh):
    coords, faces, sphere = (x.copy() for x in mesh)
    ov = VoronoiOverlap(coords, faces, sphere)
    expected = [x.copy() for x in (coords, faces, sphere)]
    coords[:] = 0
    faces[:] = 0
    sphere[:] = 0
    for x, y in zip((ov.coords, ov.faces, ov.sphere), expected):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", ["coords", "sphere"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_constructor_rejects_non_finite(mesh, name, value):
    kwargs = dict(zip(["coords", "faces", "sphere"], (x.copy() for x in mesh)))
    kwargs[name][3, 1] = value
    with pytest.raises(ValueError, match=f"{name} must contain only finite"):
        VoronoiOverlap(**kwargs)


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(coords=np.zeros((30, 2))), "coords must have shape"),
        (dict(faces=np.zeros((5, 4), dtype=int)), "faces must have shape"),
        (dict(faces=np.zeros((5, 3))), "faces must be integers"),
        (dict(faces=np.array([[0, 1, 30]])), "out of range"),
        (dict(faces=np.array([[0, 1, -1]])), "out of range"),
        (dict(sphere=np.zeros((29, 3))), "sphere must have the shape"),
        (dict(n_div=0), "n_div must be a positive integer"),
        (dict(t_div=2.0), "t_div must be a positive integer"),
    ],
)
def test_constructor_rejects_invalid(mesh, change, match):
    kwargs = dict(zip(["coords", "faces", "sphere"], mesh))
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        VoronoiOverlap(**kwargs)


@pytest.fixture
def points():
    """Points on the unit sphere, several of them in the same faces."""
    rng = np.random.default_rng(1)
    points = rng.standard_normal((40, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _old_inputs(ov):
    _, e_mapping, graph = voronoi.subdivide_edges(ov.coords, ov.faces, ov.n_div)
    return ov._node_coords, ov.faces, e_mapping, graph


@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_native_labels(mesh, n_div):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces, n_div=n_div)
    assert ov._native is None
    labels = ov._get("native")
    nn, nnd = voronoi.native_voronoi(*_old_inputs(ov))
    assert labels.nn.dtype == np.int32
    np.testing.assert_array_equal(labels.nn, nn)
    np.testing.assert_array_equal(labels.nnd, nnd)
    assert labels.size == len(coords)
    np.testing.assert_array_equal(labels.inside_ptr, np.zeros(len(faces) + 1))
    assert labels.inside_labels.size == 0


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_add_labels(mesh, points, n_div):
    coords, faces, sphere = mesh
    ov = VoronoiOverlap(coords, faces, sphere, n_div=n_div)
    ov.add("points", points)
    labels = ov._get("points")
    f_indices, weights = Sphere(sphere, faces).barycentric(
        points, eps=1e-7, return_sparse=False
    )
    assert len(np.unique(f_indices)) < len(points)
    nn, nnd = voronoi.subdivision_voronoi(*_old_inputs(ov), f_indices, weights)
    assert labels.nn.dtype == np.int32
    np.testing.assert_array_equal(labels.nn, nn)
    np.testing.assert_array_equal(labels.nnd, nnd)
    assert labels.size == len(points)

    f_inv = voronoi.inverse_face_mapping(f_indices, weights, ov.coords, faces)
    for f in range(len(faces)):
        inside = slice(labels.inside_ptr[f], labels.inside_ptr[f + 1])
        expected = f_inv.get(f, [])
        assert labels.inside_labels[inside].tolist() == [i for i, _ in expected]
        np.testing.assert_array_equal(
            labels.inside_coords[inside],
            np.reshape([c for _, c in expected], (-1, 3)),
        )

    ov.add("weights", f_indices=f_indices, weights=weights)
    for key, value in vars(labels).items():
        np.testing.assert_array_equal(getattr(ov._get("weights"), key), value)
    assert ov.names == ("native", "points", "weights")


def test_add_warns_for_shared_faces(mesh):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces)
    weights = np.full((3, 3), 1 / 3)
    with pytest.warns(UserWarning, match="3 template vertices are mapped to 2"):
        ov.add("a", f_indices=[0, 5, 0], weights=weights)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ov.add("b", f_indices=[0, 5, 1], weights=weights)


def test_add_copies_inputs(mesh):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces)
    f_indices, weights = np.array([0, 5]), np.full((2, 3), 1 / 3)
    ov.add("a", f_indices=f_indices, weights=weights)
    expected = {k: v.copy() for k, v in vars(ov._get("a")).items() if k != "size"}
    f_indices[:] = 1
    weights[:] = 0
    for key, value in expected.items():
        np.testing.assert_array_equal(getattr(ov._get("a"), key), value)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(name="native", f_indices=[0], weights=[[1, 0, 0]]), "already exists"),
        (dict(name="a", f_indices=[0], weights=[[1, 0, 0]]), "already exists"),
        (dict(name="b"), "either coords or both"),
        (dict(name="b", f_indices=[0]), "either coords or both"),
        (dict(name="b", weights=[[1, 0, 0]]), "either coords or both"),
        (dict(name="b", coords=[[1, 0, 0]], f_indices=[0]), "not both"),
        (dict(name="b", coords=[[1, 0, 0]], weights=[[1, 0, 0]]), "not both"),
        (dict(name="b", f_indices=[0.0], weights=[[1, 0, 0]]), "1-D array of int"),
        (dict(name="b", f_indices=[[0]], weights=[[1, 0, 0]]), "1-D array of int"),
        (dict(name="b", f_indices=[0], weights=[[1, 0]]), "weights must have"),
        (dict(name="b", f_indices=[0, 1], weights=[[1, 0, 0]]), "weights must have"),
        (dict(name="b", f_indices=[-1], weights=[[1, 0, 0]]), "out of range"),
        (dict(name="b", f_indices=[10**6], weights=[[1, 0, 0]]), "out of range"),
        (dict(name="b", f_indices=[0], weights=[[np.nan, 0, 0]]), "finite"),
    ],
)
def test_add_rejects_invalid(mesh, kwargs, match):
    coords, faces, sphere = mesh
    ov = VoronoiOverlap(coords, faces, sphere)
    ov.add("a", f_indices=[0], weights=[[1, 0, 0]])
    with pytest.raises(ValueError, match=match):
        ov.add(**kwargs)
    assert ov.names == ("native", "a")


@pytest.mark.parametrize(
    "coords, match",
    [([[np.inf, 0, 0]], "coords must contain only finite"), ([1, 0, 0], "shape")],
)
def test_add_rejects_invalid_coords(mesh, coords, match):
    ov = VoronoiOverlap(*mesh)
    with pytest.raises(ValueError, match=match):
        ov.add("a", coords)


def test_add_requires_sphere(mesh, points):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces)
    with pytest.raises(ValueError, match="requires sphere"):
        ov.add("a", points)


def test_discard(mesh):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces)
    ov.add("a", f_indices=[0], weights=[[1, 0, 0]])
    ov.add("b", f_indices=[1], weights=[[1, 0, 0]])
    ov.discard("a")
    assert ov.names == ("native", "b")
    with pytest.raises(KeyError, match="No point set named 'a'"):
        ov._get("a")
    with pytest.raises(KeyError, match="No point set named 'a'"):
        ov.discard("a")
    ov.add("a", f_indices=[2], weights=[[1, 0, 0]])
    assert ov.names == ("native", "b", "a")

    nn = ov._get("native").nn
    ov.discard("native")
    assert ov._native is None
    assert ov.names == ("native", "b", "a")
    np.testing.assert_array_equal(ov._get("native").nn, nn)
