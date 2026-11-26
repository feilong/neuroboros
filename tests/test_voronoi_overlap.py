import numpy as np
import pytest
import voronoi_reference as reference
from scipy.spatial import ConvexHull

from neuroboros.surface import Sphere, Surface, voronoi
from neuroboros.surface.voronoi import VoronoiSurface


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
    ov = VoronoiSurface(coords, faces, edge_divisions=n_div)
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
    ov = VoronoiSurface(coords.astype(np.float32), faces, sphere)
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
    expected = VoronoiSurface(coords, faces)
    ov = VoronoiSurface(coords, faces.astype(dtype))
    assert ov.faces.dtype == np.intp
    assert ov._face_nodes.dtype == np.intp
    np.testing.assert_array_equal(ov._face_nodes, expected._face_nodes)
    assert (ov._graph != expected._graph).nnz == 0
    ov._node_coords[ov._face_nodes]


def test_constructor_copies_inputs(mesh):
    coords, faces, sphere = (x.copy() for x in mesh)
    ov = VoronoiSurface(coords, faces, sphere)
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
        VoronoiSurface(**kwargs)


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(coords=np.zeros((30, 2))), "coords must have shape"),
        (dict(faces=np.zeros((5, 4), dtype=int)), "faces must have shape"),
        (dict(faces=np.zeros((5, 3))), "faces must be integers"),
        (dict(faces=np.array([[0, 1, 30]])), "out of range"),
        (dict(faces=np.array([[0, 1, -1]])), "out of range"),
        (dict(sphere=np.zeros((29, 3))), "sphere must have the shape"),
        (dict(edge_divisions=0), "edge_divisions must be a positive integer"),
        (dict(sample_divisions=2.0), "sample_divisions must be a positive integer"),
    ],
)
def test_constructor_rejects_invalid(mesh, change, match):
    kwargs = dict(zip(["coords", "faces", "sphere"], mesh))
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        VoronoiSurface(**kwargs)


@pytest.fixture
def points():
    """Points on the unit sphere, several of them in the same faces."""
    rng = np.random.default_rng(1)
    points = rng.standard_normal((40, 3))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _old_inputs(ov):
    _, e_mapping, graph = reference.subdivide_edges(
        ov.coords, ov.faces, ov.edge_divisions
    )
    return ov._node_coords, ov.faces, e_mapping, graph


@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_native_labels(mesh, n_div):
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces, edge_divisions=n_div)
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
    ov = VoronoiSurface(coords, faces, sphere, edge_divisions=n_div)
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


def test_add_shared_faces_do_not_warn(mesh, recwarn):
    """Several points in one face are supported and common."""
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces)
    ov.add("a", f_indices=[0, 5, 0], weights=np.full((3, 3), 1 / 3))
    assert len(recwarn) == 0


def test_add_copies_inputs(mesh):
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces)
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
    ov = VoronoiSurface(coords, faces, sphere)
    ov.add("a", f_indices=[0], weights=[[1, 0, 0]])
    with pytest.raises(ValueError, match=match):
        ov.add(**kwargs)
    assert ov.names == ("native", "a")


@pytest.mark.parametrize(
    "coords, match",
    [([[np.inf, 0, 0]], "coords must contain only finite"), ([1, 0, 0], "shape")],
)
def test_add_rejects_invalid_coords(mesh, coords, match):
    ov = VoronoiSurface(*mesh)
    with pytest.raises(ValueError, match=match):
        ov.add("a", coords)


def test_add_requires_sphere(mesh, points):
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces)
    with pytest.raises(ValueError, match="requires sphere"):
        ov.add("a", points)


def test_discard(mesh):
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces)
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
    _, e_mapping, _ = reference.subdivide_edges(ov.coords, ov.faces, ov.edge_divisions)
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


def _sample_labels(ov, name):
    """The label of every sample of every face, as the pass computes them."""
    point_set = ov._get(name)
    faces = np.arange(len(ov.faces))
    return ov._face_labels(
        point_set,
        ov._single_labels(point_set),
        faces,
        *ov._boundary_distances(faces),
    )


@pytest.mark.parametrize("n_div", [1, 2, 8])
@pytest.mark.parametrize("t_div", [1, 2, 8])
@pytest.mark.parametrize("n_points", [0, 5, 40])
def test_sample_labels(mesh, n_div, t_div, n_points):
    coords, faces, sphere = mesh
    ov = VoronoiSurface(coords, faces, edge_divisions=n_div, sample_divisions=t_div)
    name, placement = "native", {}
    if n_points:
        rng = np.random.default_rng(n_points)
        weights = rng.dirichlet(np.ones(3), n_points)
        f_indices = rng.integers(0, len(faces), n_points)
        placement = dict(f_indices=f_indices, weights=weights)
        name = "points"
        ov.add(name, **placement)
    labels = _sample_labels(ov, name)
    assert labels.shape == (len(faces), t_div**2)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, _expected_samples(ov, name, **placement))
    is_single = ov._single_labels(ov._get(name)) >= 0
    assert not np.any(is_single & (np.diff(ov._get(name).inside_ptr) > 0))
    if n_points == 5:
        assert 0 < is_single.sum() < len(faces)


@pytest.mark.parametrize("n_div", [1, 2, 4])
@pytest.mark.parametrize("t_div", [2, 4])
def test_sample_labels_ties(n_div, t_div):
    """A flat, symmetric mesh where many scores tie exactly."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ov = VoronoiSurface(coords, faces, edge_divisions=n_div, sample_divisions=t_div)
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
        expected = _expected_samples(ov, name, **placement)
        np.testing.assert_array_equal(_sample_labels(ov, name), expected)


def test_sample_labels_many_inside(mesh):
    """One face with more inside points than samples."""
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces, sample_divisions=4)
    rng = np.random.default_rng(0)
    others = np.delete(np.arange(len(faces)), 3)
    f_indices = np.concatenate([np.full(300, 3), others])
    weights = rng.dirichlet(np.ones(3), len(f_indices))
    ov.add("points", f_indices=f_indices, weights=weights)
    expected = _expected_samples(ov, "points", f_indices, weights)
    np.testing.assert_array_equal(_sample_labels(ov, "points"), expected)


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
    _, e_mapping, _ = reference.subdivide_edges(ov.coords, ov.faces, ov.edge_divisions)
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
        t_div=ov.sample_divisions,
    )


def _with_point_sets(mesh, n_div=8, t_div=4, sizes=(("few", 5), ("many", 40))):
    coords, faces, _ = mesh
    ov = VoronoiSurface(coords, faces, edge_divisions=n_div, sample_divisions=t_div)
    placements = {}
    for name, n in sizes:
        rng = np.random.default_rng(n)
        placements[name] = (
            rng.integers(0, len(faces), n),
            rng.dirichlet(np.ones(3), n),
        )
        ov.add(name, f_indices=placements[name][0], weights=placements[name][1])
    return ov, placements


@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_transform_matches_compute_overlap(mesh, n_div, t_div):
    ov, placements = _with_point_sets(mesh, n_div, t_div)
    for source in ov.names:
        for target in ov.names:
            mat = ov.transform(source, target)
            _assert_identical(mat, _reference(ov, placements, source, target))
            _assert_identical(ov.transform(target, source), mat.T)
            np.testing.assert_allclose(mat.sum(axis=1).A1, ov.areas(source))
            np.testing.assert_allclose(mat.sum(), ov.face_areas.sum())


@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_areas(mesh, n_div, t_div):
    ov, placements = _with_point_sets(mesh, n_div, t_div)
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
    ov = VoronoiSurface(
        coords.astype(dtype), faces, edge_divisions=n_div, sample_divisions=t_div
    )
    areas = surf.vertex_areas_nn(n_div=n_div, t_div=t_div)
    np.testing.assert_array_equal(areas, ov.areas("native"))
    np.testing.assert_allclose(areas.sum(), surf.face_areas.sum())
    if (n_div, t_div) == (8, 32):
        np.testing.assert_array_equal(surf.vertex_areas_nn(), areas)


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_entries_chunk_size(mesh, chunk_size):
    """Several requests in one pass give the entries of each request alone."""
    requests = [("native",), ("many",), ("native", "few"), ("many", "native")]
    ov, _ = _with_point_sets(mesh)
    entries = ov._entries(requests, chunk_size=chunk_size)
    for request, request_entries in zip(requests, entries):
        for a, b in zip(request_entries, ov._entries([request])[0]):
            np.testing.assert_array_equal(a, b)


SIX_SETS = (("a", 3), ("b", 5), ("c", 17), ("d", 40), ("e", 200))


def test_transforms_all_pairs(mesh):
    """All ordered pairs of six point sets in one pass."""
    ov, _ = _with_point_sets(mesh, sizes=SIX_SETS)
    single, _ = _with_point_sets(mesh, sizes=SIX_SETS)
    pairs = [(a, b) for a in ov.names for b in ov.names]
    mats = ov.transforms(pairs)
    assert len(mats) == 36
    for pair, mat in zip(pairs, mats):
        _assert_identical(mat, single.transform(*pair))


def test_transforms_composition_and_order(mesh):
    """A matrix does not depend on the other pairs of the call or on their
    order."""
    single, _ = _with_point_sets(mesh, sizes=SIX_SETS)
    names = single.names
    expected = {(a, b): single.transform(a, b) for a in names for b in names}
    ov, _ = _with_point_sets(mesh, sizes=SIX_SETS)
    rng = np.random.default_rng(0)
    for _ in range(10):
        # With repeated pairs.
        indices = rng.integers(0, len(expected), rng.integers(1, 8))
        pairs = [list(expected)[i] for i in indices]
        mats = ov.transforms(pairs)
        assert len(mats) == len(pairs)
        for pair, mat in zip(pairs, mats):
            _assert_identical(mat, expected[pair])


def test_transforms_special_requests(mesh):
    ov, _ = _with_point_sets(mesh)
    assert ov.transforms([]) == []
    assert ov._native is None

    pairs = [("native", "many"), ("many", "native"), ("native", "many")]
    mat, reverse, repeated = ov.transforms(pairs + [("many", "many")])[:3]
    _assert_identical(reverse, mat.T)
    _assert_identical(repeated, mat)
    expected = mat.copy()
    for x in (repeated, reverse):
        x.data[:] = 0
        x.indices[:] = 0
        x.indptr[:] = 0
    _assert_identical(mat, expected)
    _assert_identical(ov.transform("native", "many"), expected)
    # A list is accepted as a pair, and any iterable as the pairs.
    _assert_identical(ov.transforms(iter([["native", "many"]]))[0], expected)


def test_self_overlap(mesh):
    ov, _ = _with_point_sets(mesh)
    for name, mat in zip(ov.names, ov.transforms([(n, n) for n in ov.names])):
        diagonal = mat.diagonal()
        assert mat.nnz == np.count_nonzero(diagonal)
        np.testing.assert_array_equal(diagonal, ov.areas(name))


@pytest.mark.parametrize(
    "pairs, error",
    [
        ([("native",)], TypeError),
        ([("native", "few", "many")], TypeError),
        (["native"], TypeError),
        (("native", "many"), TypeError),
        ([("native", 1)], TypeError),
        ([("native", "many"), None], TypeError),
        ([("native", "other")], KeyError),
        ([("native", "many"), ("other", "many")], KeyError),
    ],
)
def test_transforms_rejects_invalid(mesh, pairs, error):
    ov, _ = _with_point_sets(mesh)
    with pytest.raises(error, match="pair of names|No point set named 'other'"):
        ov.transforms(pairs)
    # Raised before anything is computed.
    assert ov._native is None


@pytest.mark.parametrize("scale", [2.0, 0.25])
def test_scaling_by_power_of_two_is_exact(mesh, scale):
    coords, faces, sphere = mesh
    ov, _ = _with_point_sets(mesh)
    scaled, _ = _with_point_sets((coords * scale, faces, sphere))
    pairs = [("native", "many"), ("few", "many")]
    for mat, scaled_mat in zip(ov.transforms(pairs), scaled.transforms(pairs)):
        mat.data *= scale**2
        _assert_identical(scaled_mat, mat)
    np.testing.assert_array_equal(scaled.areas("many"), ov.areas("many") * scale**2)


def test_pass_shares_distances(mesh, monkeypatch):
    """The point sets of a call share the distances of each face."""
    ov, _ = _with_point_sets(mesh, sizes=[("few", 5), ("some", 10)])
    calls = []
    boundary_distances = ov._boundary_distances
    monkeypatch.setattr(
        ov,
        "_boundary_distances",
        lambda faces: calls.append(len(faces)) or boundary_distances(faces),
    )
    is_mixed = {name: ov._single_labels(ov._get(name)) < 0 for name in ("few", "some")}
    n_visited = (is_mixed["few"] | is_mixed["some"]).sum()
    assert max(is_mixed["few"].sum(), is_mixed["some"].sum()) < n_visited
    for requests in [[("few", "some")], [("few", "some"), ("some", "few"), ("few",)]]:
        calls.clear()
        ov._entries(requests, chunk_size=7)
        assert sum(calls) == n_visited < len(ov.faces)
        assert max(calls) == 7


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
    ov = VoronoiSurface(
        np.concatenate([coords, coords + 10]),
        np.concatenate([faces, faces + len(coords)]),
    )
    ov.add("points", f_indices=[0], weights=[[1, 0, 0]])
    ov.transform("native", "native")
    with pytest.raises(ValueError, match="cannot be reached"):
        ov.transform("native", "points")
    with pytest.raises(ValueError, match="cannot be reached"):
        ov.areas("points")
