import warnings

import numpy as np
import pytest
import voronoi_reference as reference
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
    _, e_mapping, graph = reference.subdivide_edges(coords, faces, n_div)
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
    new_coords = reference.subdivide_edges(surf.coords, faces, 8)[0]
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
    _, e_mapping, graph = reference.subdivide_edges(ov.coords, ov.faces, ov.n_div)
    return ov._node_coords, ov.faces, e_mapping, graph


@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_native_labels(mesh, n_div):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces, n_div=n_div)
    assert ov._native is None
    point_set = ov._get("native")
    nn, nnd = reference.native_voronoi(*_old_inputs(ov))
    assert point_set.nn.dtype == np.int32
    np.testing.assert_array_equal(point_set.nn, nn)
    np.testing.assert_array_equal(point_set.nnd, nnd)
    assert point_set.size == len(coords)
    np.testing.assert_array_equal(point_set.inside_ptr, np.zeros(len(faces) + 1))
    assert point_set.inside_labels.size == 0


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_add_labels(mesh, points, n_div):
    coords, faces, sphere = mesh
    ov = VoronoiOverlap(coords, faces, sphere, n_div=n_div)
    ov.add("points", points)
    point_set = ov._get("points")
    f_indices, weights = Sphere(sphere, faces).barycentric(
        points, eps=1e-7, return_sparse=False
    )
    assert len(np.unique(f_indices)) < len(points)
    nn, nnd = reference.subdivision_voronoi(*_old_inputs(ov), f_indices, weights)
    assert point_set.nn.dtype == np.int32
    np.testing.assert_array_equal(point_set.nn, nn)
    np.testing.assert_array_equal(point_set.nnd, nnd)
    assert point_set.size == len(points)

    f_inv = reference.inverse_face_mapping(f_indices, weights, ov.coords, faces)
    for f in range(len(faces)):
        inside = slice(point_set.inside_ptr[f], point_set.inside_ptr[f + 1])
        expected = f_inv.get(f, [])
        assert point_set.inside_labels[inside].tolist() == [i for i, _ in expected]
        np.testing.assert_array_equal(
            point_set.inside_coords[inside],
            np.reshape([c for _, c in expected], (-1, 3)),
        )

    ov.add("weights", f_indices=f_indices, weights=weights)
    for key, value in vars(point_set).items():
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


def _expected_samples(ov, name, f_indices=None, weights=None):
    """The sample labels of every face from compute_occupation."""
    point_set = ov._get(name)
    _, e_mapping, _ = reference.subdivide_edges(ov.coords, ov.faces, ov.n_div)
    f_inv = {}
    if f_indices is not None:
        f_inv = reference.inverse_face_mapping(f_indices, weights, ov.coords, ov.faces)
    expected = np.empty((len(ov.faces), len(ov._ww)), dtype=int)
    for f_idx, f in enumerate(ov.faces):
        a, b, c = sorted(f)
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        occupation = reference.compute_occupation(
            f_idx,
            f,
            ov._node_coords,
            indices,
            point_set.nn,
            point_set.nnd,
            f_inv,
            ov._ww,
        )
        for u, mask in occupation.items():
            expected[f_idx, mask] = u
    return expected


def _labelled(ov, name, chunk_size=64):
    """The sample labels of a point set, as the pass of ``_entries`` stores them."""
    ov._entries([(name,)], chunk_size=chunk_size)
    return ov._sample_labels_cache[name]


def _decoded_samples(ov, name, sample_labels):
    decoded = np.repeat(sample_labels.single[:, np.newaxis], len(ov._ww), axis=1)
    decoded[sample_labels.mixed_faces] = ov._decode(
        ov._get(name), sample_labels.mixed_faces, sample_labels.local
    )
    return decoded


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_div", [1, 2, 8])
@pytest.mark.parametrize("t_div", [1, 2, 8])
@pytest.mark.parametrize("n_points", [0, 5, 40])
def test_sample_labels(mesh, n_div, t_div, n_points):
    coords, faces, sphere = mesh
    ov = VoronoiOverlap(coords, faces, n_div=n_div, t_div=t_div)
    name, placement = "native", {}
    if n_points:
        rng = np.random.default_rng(n_points)
        weights = rng.dirichlet(np.ones(3), n_points)
        f_indices = rng.integers(0, len(faces), n_points)
        placement = dict(f_indices=f_indices, weights=weights)
        name = "points"
        ov.add(name, **placement)
    sample_labels = _labelled(ov, name)
    expected = _expected_samples(ov, name, **placement)

    np.testing.assert_array_equal(_decoded_samples(ov, name, sample_labels), expected)
    is_single = sample_labels.single >= 0
    np.testing.assert_array_equal(np.flatnonzero(~is_single), sample_labels.mixed_faces)
    assert not np.any(is_single & (np.diff(ov._get(name).inside_ptr) > 0))
    assert sample_labels.local.shape == (len(sample_labels.mixed_faces), t_div**2)
    assert sample_labels.local.dtype == np.uint8
    if n_points == 5:
        assert 0 < is_single.sum() < len(faces)


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_div", [1, 2, 4])
@pytest.mark.parametrize("t_div", [2, 4])
def test_sample_labels_ties(n_div, t_div):
    """A flat, symmetric mesh where many scores tie exactly."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ov = VoronoiOverlap(coords, faces, n_div=n_div, t_div=t_div)
    # Points at vertex 1 and 2 of both faces, at the center of the square,
    # and a pair of points mirrored across the diagonal in face 0.
    f_indices = np.array([0, 1, 0, 1, 0, 0, 0])
    weights = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0.5, 0.5],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
        ]
    )
    ov.add("points", f_indices=f_indices, weights=weights)
    for name, placement in [
        ("native", {}),
        ("points", dict(f_indices=f_indices, weights=weights)),
    ]:
        sample_labels = _labelled(ov, name)
        expected = _expected_samples(ov, name, **placement)
        np.testing.assert_array_equal(
            _decoded_samples(ov, name, sample_labels), expected
        )


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_sample_labels_chunk_size(mesh, points, chunk_size):
    ov = VoronoiOverlap(*mesh, t_div=4)
    ov.add("points", points)
    for name in ov.names:
        expected = vars(_labelled(ov, name)).copy()
        ov.clear_cache()
        sample_labels = _labelled(ov, name, chunk_size=chunk_size)
        for key, value in expected.items():
            np.testing.assert_array_equal(getattr(sample_labels, key), value)


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_inside, dtype", [(232, np.uint8), (233, np.uint16)])
def test_sample_labels_dtype(mesh, n_inside, dtype):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces, t_div=4)
    rng = np.random.default_rng(0)
    others = np.delete(np.arange(len(faces)), 3)
    f_indices = np.concatenate([np.full(n_inside, 3), others])
    weights = rng.dirichlet(np.ones(3), len(f_indices))
    ov.add("points", f_indices=f_indices, weights=weights)
    sample_labels = _labelled(ov, "points")
    assert sample_labels.local.dtype == dtype
    expected = _expected_samples(ov, "points", f_indices, weights)
    np.testing.assert_array_equal(
        _decoded_samples(ov, "points", sample_labels), expected
    )


def test_sample_labels_limit(mesh):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(coords, faces, n_div=1, t_div=2)
    weights = np.full((65534, 3), 1 / 3)
    with pytest.warns(UserWarning, match="mapped to"):
        ov.add("points", f_indices=np.zeros(65534, dtype=int), weights=weights)
    with pytest.raises(ValueError, match="65537 candidates in one face"):
        ov._sample_labels(ov._get("points"))


@pytest.mark.parametrize(
    "n_candidates, dtype", [(1, np.uint8), (256, np.uint8), (257, np.uint16)]
)
def test_local_dtype(n_candidates, dtype):
    assert voronoi._local_dtype(n_candidates) == dtype
    assert voronoi._local_dtype(65536) == np.uint16


@pytest.mark.parametrize("t_div", [1, 2, 3, 32])
def test_split_triangle(t_div):
    ww = voronoi._split_triangle(t_div)
    np.testing.assert_array_equal(ww, reference.split_triangle(t_div))
    assert ww.shape == (t_div**2, 3)
    np.testing.assert_allclose(ww.sum(axis=1), 1)
    assert np.all(ww > 0)
    np.testing.assert_allclose(ww.mean(axis=0), 1 / 3)
    if t_div == 1:
        np.testing.assert_array_equal(ww, [[1 / 3, 1 / 3, 1 / 3]])


def _assert_identical(mat, expected):
    """Assert exact equality of sparse shapes, indices, and values."""
    mat, expected = mat.tocsr(), expected.tocsr()
    for m in (mat, expected):
        m.sort_indices()
    assert mat.shape == expected.shape
    np.testing.assert_array_equal(mat.indptr, expected.indptr)
    np.testing.assert_array_equal(mat.indices, expected.indices)
    np.testing.assert_array_equal(mat.data, expected.data)


def _reference(ov, placements, source, target):
    """The overlap from compute_overlap, with the labels of ``ov``."""
    _, e_mapping, _ = reference.subdivide_edges(ov.coords, ov.faces, ov.n_div)
    args = []
    for name in (source, target):
        point_set = ov._get(name)
        f_inv = {}
        if name != "native":
            f_inv = reference.inverse_face_mapping(
                *placements[name], ov.coords, ov.faces
            )
        args.append((point_set.nn, point_set.nnd, f_inv))
    return reference.compute_overlap(
        ov.faces,
        ov.face_areas,
        e_mapping,
        ov._node_coords,
        *args[0],
        *args[1],
        ov._get(source).size,
        ov._get(target).size,
        t_div=ov.t_div,
    )


def _with_point_sets(
    mesh, n_div=8, t_div=4, sizes=(("few", 5), ("many", 40)), cache_samples=True
):
    coords, faces, _ = mesh
    ov = VoronoiOverlap(
        coords, faces, n_div=n_div, t_div=t_div, cache_samples=cache_samples
    )
    placements = {}
    for name, n in sizes:
        rng = np.random.default_rng(n)
        placements[name] = (
            rng.integers(0, len(faces), n),
            rng.dirichlet(np.ones(3), n),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ov.add(name, f_indices=placements[name][0], weights=placements[name][1])
    return ov, placements


@pytest.mark.parametrize("cache_samples", [True, False])
@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_transform_matches_compute_overlap(mesh, n_div, t_div, cache_samples):
    ov, placements = _with_point_sets(mesh, n_div, t_div, cache_samples=cache_samples)
    for source in ov.names:
        for target in ov.names:
            mat = ov.transform(source, target)
            _assert_identical(mat, _reference(ov, placements, source, target))
            _assert_identical(ov.transform(target, source), mat.T)
            np.testing.assert_allclose(mat.sum(axis=1).A1, ov.areas(source))
            np.testing.assert_allclose(mat.sum(), ov.face_areas.sum())


@pytest.mark.parametrize("cache_samples", [True, False])
@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_areas(mesh, n_div, t_div, cache_samples):
    ov, placements = _with_point_sets(mesh, n_div, t_div, cache_samples=cache_samples)
    for name in ["native", "few", "many"]:
        expected = np.zeros(ov._get(name).size)
        occupation = _expected_samples(ov, name, *placements.get(name, ()))
        for f_idx, labels in enumerate(occupation):
            for u in np.unique(labels):
                expected[u] += (labels == u).mean() * ov.face_areas[f_idx]
        np.testing.assert_array_equal(ov.areas(name), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_div, t_div", [(8, 32), (1, 1), (2, 3)])
def test_vertex_areas_nn(mesh, dtype, n_div, t_div):
    coords, faces, _ = mesh
    surf = Surface(coords.astype(dtype), faces)
    ov = VoronoiOverlap(coords.astype(dtype), faces, n_div=n_div, t_div=t_div)
    areas = surf.vertex_areas_nn(n_div=n_div, t_div=t_div)
    np.testing.assert_array_equal(areas, ov.areas("native"))
    np.testing.assert_allclose(areas.sum(), surf.face_areas.sum())
    if (n_div, t_div) == (8, 32):
        np.testing.assert_array_equal(surf.vertex_areas_nn(), areas)


@pytest.mark.parametrize("cache_samples", [True, False])
@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_entries_chunk_size(mesh, chunk_size, cache_samples):
    requests = [("native",), ("many",), ("native", "few"), ("many", "native")]
    expected, _ = _with_point_sets(mesh, cache_samples=cache_samples)
    ov, _ = _with_point_sets(mesh, cache_samples=cache_samples)
    # Once together and, with the cache, once more from the cached labels.
    for _ in range(2):
        entries = ov._entries(requests, chunk_size=chunk_size)
        for request, request_entries in zip(requests, entries):
            for a, b in zip(request_entries, expected._entries([request])[0]):
                np.testing.assert_array_equal(a, b)


def test_pass_shares_distances(mesh, monkeypatch):
    """Point sets labelled in one pass share the distances of each face, and
    cached labels need none."""
    ov, _ = _with_point_sets(mesh)
    calls = []
    boundary_distances = ov._boundary_distances
    monkeypatch.setattr(
        ov,
        "_boundary_distances",
        lambda faces: calls.append(len(faces)) or boundary_distances(faces),
    )
    ov._entries([("few", "many"), ("many", "few"), ("few",)])
    is_mixed = {name: ov._sample_labels_cache[name].single < 0 for name in ov.names[1:]}
    assert 0 < is_mixed["few"].sum() < is_mixed["many"].sum()
    assert sum(calls) == (is_mixed["few"] | is_mixed["many"]).sum()
    calls.clear()
    ov._entries([("few", "many")])
    assert calls == []
    # Only the faces that the new point set needs.
    ov.clear_cache("few")
    ov._entries([("few", "many")])
    assert sum(calls) == is_mixed["few"].sum()


def test_transform_returns_new_arrays(mesh):
    ov, _ = _with_point_sets(mesh)
    mat = ov.transform("native", "many")
    areas = ov.areas("many")
    expected_mat, expected_areas = mat.copy(), areas.copy()
    mat.data[:] = 0
    areas[:] = 0
    _assert_identical(ov.transform("native", "many"), expected_mat)
    _assert_identical(ov.transform("many", "native"), expected_mat.T)
    np.testing.assert_array_equal(ov.areas("many"), expected_areas)


def test_transform_large_labels(mesh):
    """Pair keys exceed the int32 range."""
    ov, placements = _with_point_sets(mesh, n_div=1, t_div=2, sizes=[("large", 70000)])
    assert ov._get("large").size ** 2 > np.iinfo(np.int32).max
    mat = ov.transform("large", "large")
    _assert_identical(mat, _reference(ov, placements, "large", "large"))
    assert mat.indices.max() > 65536


def test_transform_unreachable_faces(mesh):
    coords, faces, _ = mesh
    # A second copy of the mesh, not connected to the first.
    ov = VoronoiOverlap(
        np.concatenate([coords, coords + 10]),
        np.concatenate([faces, faces + len(coords)]),
    )
    ov.add("points", f_indices=[0], weights=[[1, 0, 0]])
    ov.transform("native", "native")
    with pytest.raises(ValueError, match="cannot be reached"):
        ov.transform("native", "points")
    with pytest.raises(ValueError, match="cannot be reached"):
        ov.areas("points")


def test_discard_releases_samples(mesh):
    ov, _ = _with_point_sets(mesh)
    ov.transform("native", "many")
    assert set(ov._sample_labels_cache) == {"native", "many"}
    ov.discard("many")
    ov.discard("native")
    assert ov._sample_labels_cache == {}
    with pytest.raises(KeyError):
        ov.transform("native", "many")
    _assert_identical(ov.transform("native", "few"), ov.transform("few", "native").T)


def test_uncached_transform(mesh, monkeypatch):
    ov, _ = _with_point_sets(mesh, cache_samples=False)
    cached, _ = _with_point_sets(mesh)
    calls = []
    boundary_distances = ov._boundary_distances
    monkeypatch.setattr(
        ov,
        "_boundary_distances",
        lambda faces: calls.append(len(faces)) or boundary_distances(faces),
    )
    for names in [("native", "many"), ("few", "many"), ("many",)]:
        calls.clear()
        (entries,) = ov._entries([names], chunk_size=7)
        (expected,) = cached._entries([names], chunk_size=7)
        for a, b in zip(entries, expected):
            np.testing.assert_array_equal(a, b)
        # The two point sets share the distances of each chunk.
        is_mixed = np.logical_or.reduce(
            [cached._get_sample_labels(name).single < 0 for name in names]
        )
        assert sum(calls) == is_mixed.sum()
        assert max(calls) == 7
    assert ov._sample_labels_cache == {}


def test_clear_cache(mesh):
    ov, _ = _with_point_sets(mesh)
    expected = ov.transform("native", "many")
    expected_areas = ov.areas("few")
    assert set(ov._sample_labels_cache) == {"native", "many", "few"}
    ov.clear_cache("many")
    assert set(ov._sample_labels_cache) == {"native", "few"}
    native = ov._get("native")
    _assert_identical(ov.transform("native", "many"), expected)
    assert ov._get("native") is native
    ov.clear_cache()
    assert ov._sample_labels_cache == {}
    _assert_identical(ov.transform("native", "many"), expected)
    np.testing.assert_array_equal(ov.areas("few"), expected_areas)
    ov.clear_cache("native")
    assert set(ov._sample_labels_cache) == {"many", "few"}
    with pytest.raises(KeyError, match="No point set named 'other'"):
        ov.clear_cache("other")
