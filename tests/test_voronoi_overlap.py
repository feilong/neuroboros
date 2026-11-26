import gc
import types
import weakref

import numpy as np
import pytest
import scipy.sparse as sparse
import voronoi_reference as reference
from scipy.sparse import csgraph
from scipy.spatial import ConvexHull

from neuroboros.surface import Sphere, Surface, voronoi
from neuroboros.surface.voronoi import VoronoiPartition, VoronoiSurface


@pytest.fixture(params=["kernel", "numpy"], autouse=True)
def labelling_path(request, monkeypatch):
    """Run every test with the compiled kernel and with NumPy."""
    if request.param == "numpy":
        monkeypatch.setattr(voronoi, "_kernel", None)
    elif voronoi._kernel is None:
        pytest.skip("The compiled kernel is not built.")
    return request.param


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
    surface = VoronoiSurface(coords, faces, edge_divisions=n_div)
    assert surface._face_nodes.shape == (len(faces), 3 * n_div)
    _, e_mapping, graph = reference.subdivide_edges(coords, faces, n_div)
    for f, nodes in zip(faces, surface._face_nodes):
        a, b, c = sorted(f)
        expected = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        np.testing.assert_array_equal(nodes, expected)
    assert (surface._graph != graph).nnz == 0


def test_constructor(mesh):
    coords, faces, sphere = mesh
    surface = VoronoiSurface(coords.astype(np.float32), faces, sphere)
    surf = Surface(coords.astype(np.float32), faces)
    assert surface.coords.dtype == np.float64
    np.testing.assert_array_equal(surface.coords, surf.coords)
    np.testing.assert_array_equal(surface.face_areas, surf.face_areas)
    new_coords = reference.subdivide_edges(surf.coords, faces, 8)[0]
    np.testing.assert_array_equal(
        surface._node_coords, np.concatenate([surf.coords, new_coords])
    )


@pytest.mark.parametrize("dtype", [np.uint64, np.uint32, np.int32, np.int16])
def test_constructor_face_dtypes(mesh, dtype):
    coords, faces, _ = mesh
    expected = VoronoiSurface(coords, faces)
    surface = VoronoiSurface(coords, faces.astype(dtype))
    assert surface.faces.dtype == np.intp
    assert surface._face_nodes.dtype == np.intp
    np.testing.assert_array_equal(surface._face_nodes, expected._face_nodes)
    assert (surface._graph != expected._graph).nnz == 0
    surface._node_coords[surface._face_nodes]


def test_constructor_copies_inputs(mesh):
    coords, faces, sphere = (x.copy() for x in mesh)
    surface = VoronoiSurface(coords, faces, sphere)
    expected = [x.copy() for x in (coords, faces, sphere)]
    coords[:] = 0
    faces[:] = 0
    sphere[:] = 0
    for x, y in zip((surface.coords, surface.faces, surface.sphere), expected):
        np.testing.assert_array_equal(x, y)


def test_surface_is_read_only(mesh):
    surface = VoronoiSurface(*mesh)
    assert repr(surface) == "<VoronoiSurface with 30 vertices and 56 faces>"
    assert (surface.edge_divisions, surface.sample_divisions) == (8, 32)
    names = ["coords", "faces", "sphere", "face_areas"]
    for name in names + ["edge_divisions", "sample_divisions", "native"]:
        with pytest.raises(AttributeError):
            setattr(surface, name, getattr(surface, name))
    for name in names:
        with pytest.raises(ValueError, match="read-only"):
            getattr(surface, name)[0] = 0
    assert VoronoiSurface(*mesh[:2]).sphere is None
    # So are the arrays of a partition.
    partition = surface.partition(mesh[2][:3])
    for point_set in (surface.native._point_set, partition._point_set):
        for value in vars(point_set).values():
            assert isinstance(value, int) or not value.flags.writeable
    partition.overlap(surface.native)


def test_constructor_memory_order(mesh, points):
    """Arrays in Fortran order, or views with steps, give the same results."""
    coords, faces, sphere = mesh
    expected = VoronoiSurface(coords, faces, sphere, sample_divisions=4)
    expected = expected.native.overlap(expected.partition(points))
    doubled = np.repeat(coords, 2, axis=1)
    for args in [
        [np.asfortranarray(x) for x in mesh],
        [doubled[:, ::2], np.asfortranarray(faces)[::1], sphere.T.copy().T],
    ]:
        surface = VoronoiSurface(*args, sample_divisions=4)
        for array in (surface.coords, surface.faces, surface.sphere):
            assert array.flags.c_contiguous
        partition = surface.partition(np.asfortranarray(points))
        _assert_identical(surface.native.overlap(partition), expected)
        f_indices, weights = Sphere(sphere, faces).barycentric(
            points, eps=1e-7, return_sparse=False
        )
        partition = surface.partition(
            face_indices=f_indices, weights=np.asfortranarray(weights)
        )
        _assert_identical(surface.native.overlap(partition), expected)


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


def _old_inputs(surface):
    _, e_mapping, graph = reference.subdivide_edges(
        surface.coords, surface.faces, surface.edge_divisions
    )
    return surface._node_coords, surface.faces, e_mapping, graph


def _point_connections(surface, partition):
    """The boundary nodes that each point is connected to, and the lengths."""
    point_set = partition._point_set
    f_indices = np.repeat(np.arange(len(surface.faces)), np.diff(point_set.inside_ptr))
    nodes = np.empty((point_set.size, surface._face_nodes.shape[1]), dtype=int)
    nodes[point_set.inside_labels] = surface._face_nodes[f_indices]
    positions = np.empty((point_set.size, 3))
    positions[point_set.inside_labels] = point_set.inside_coords
    diff = positions[:, np.newaxis] - surface._node_coords[nodes]
    return nodes, np.sqrt(np.sum(diff**2, axis=2))


def _lowest_labels(graph, distances, labels, point_nodes=(), point_distances=()):
    """The tie rule of ``voronoi._lowest_labels``, one connection at a time."""
    graph, labels = graph.tocoo(), labels.copy()
    connections = [
        (None, v, w, p)
        for p, (nodes, lengths) in enumerate(zip(point_nodes, point_distances))
        for v, w in zip(nodes, lengths)
    ]
    connections += [
        (u, v, w, None) for u, v, w in zip(graph.row, graph.col, graph.data)
    ]
    changed = True
    while changed:
        changed = False
        for u, v, w, label in connections:
            start = 0.0 if u is None else distances[u]
            label = labels[u] if label is None else label
            if start + w == distances[v] and -1 < label < labels[v]:
                labels[v] = label
                changed = True
    return labels


@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_native_labels(mesh, n_div):
    coords, faces, _ = mesh
    surface = VoronoiSurface(coords, faces, edge_divisions=n_div)
    assert surface._native is None
    point_set = surface.native._point_set
    nn, nnd = reference.native_voronoi(*_old_inputs(surface))
    assert point_set.nn.dtype == np.int32
    # SciPy's distances, and the lowest index where SciPy had a choice.
    np.testing.assert_array_equal(point_set.nn, _lowest_labels(surface._graph, nnd, nn))
    np.testing.assert_array_equal(point_set.nnd, nnd)
    assert point_set.size == len(coords)
    np.testing.assert_array_equal(point_set.inside_ptr, np.zeros(len(faces) + 1))
    assert point_set.inside_labels.size == 0


@pytest.mark.filterwarnings("ignore:.*template vertices are mapped")
@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_partition_labels(mesh, points, n_div):
    coords, faces, sphere = mesh
    surface = VoronoiSurface(coords, faces, sphere, edge_divisions=n_div)
    partition = surface.partition(points)
    point_set = partition._point_set
    f_indices, weights = Sphere(sphere, faces).barycentric(
        points, eps=1e-7, return_sparse=False
    )
    assert len(np.unique(f_indices)) < len(points)
    nn, nnd = reference.subdivision_voronoi(*_old_inputs(surface), f_indices, weights)
    assert point_set.nn.dtype == np.int32
    assert point_set.size == len(points)
    # The reference places the points with a matrix product, which rounds
    # differently, so the distances agree only up to rounding. The exact
    # values are checked in test_labels_do_not_depend_on_scipy.
    np.testing.assert_allclose(point_set.nnd, nnd, rtol=1e-13)
    # No node is at the same distance from two of these points.
    np.testing.assert_array_equal(point_set.nn, nn)

    f_inv = reference.inverse_face_mapping(f_indices, weights, surface.coords, faces)
    for f in range(len(faces)):
        inside = slice(point_set.inside_ptr[f], point_set.inside_ptr[f + 1])
        expected = f_inv.get(f, [])
        assert point_set.inside_labels[inside].tolist() == [i for i, _ in expected]
        np.testing.assert_allclose(
            point_set.inside_coords[inside],
            np.reshape([c for _, c in expected], (-1, 3)),
            rtol=1e-13,
        )
        positions = voronoi._barycentric_points(
            weights[point_set.inside_labels[inside]], surface.coords[faces[f]]
        )
        np.testing.assert_array_equal(point_set.inside_coords[inside], positions)


def test_partition_placement_routes(mesh, points):
    """Points on the sphere, or faces and weights, give the same partition."""
    coords, faces, sphere = mesh
    surface = VoronoiSurface(coords, faces, sphere)
    f_indices, weights = Sphere(sphere, faces).barycentric(
        points, eps=1e-7, return_sparse=False
    )
    placed = surface.partition(points)
    located = surface.partition(face_indices=f_indices, weights=weights)
    assert placed is not located
    assert placed.n_cells == located.n_cells == len(points)
    expected = {k: np.copy(v) for k, v in vars(placed._point_set).items()}
    # The inputs are copied.
    f_indices[:] = 0
    weights[:] = 0
    for key, value in expected.items():
        np.testing.assert_array_equal(getattr(located._point_set, key), value)
    _assert_identical(placed.overlap(surface.native), located.overlap(surface.native))


def test_partition_shared_faces_do_not_warn(mesh, recwarn):
    """Several points in one face are supported and common."""
    coords, faces, _ = mesh
    surface = VoronoiSurface(coords, faces)
    surface.partition(face_indices=[0, 5, 0], weights=np.full((3, 3), 1 / 3))
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(), "either coords or both"),
        (dict(face_indices=[0]), "either coords or both"),
        (dict(weights=[[1, 0, 0]]), "either coords or both"),
        (dict(coords=[[1, 0, 0]], face_indices=[0]), "not both"),
        (dict(coords=[[1, 0, 0]], weights=[[1, 0, 0]]), "not both"),
        (dict(face_indices=[0.0], weights=[[1, 0, 0]]), "1-D array of int"),
        (dict(face_indices=[[0]], weights=[[1, 0, 0]]), "1-D array of int"),
        (dict(face_indices=[0], weights=[[1, 0]]), "weights must have"),
        (dict(face_indices=[0, 1], weights=[[1, 0, 0]]), "weights must have"),
        (dict(face_indices=[-1], weights=[[1, 0, 0]]), "out of range"),
        (dict(face_indices=[10**6], weights=[[1, 0, 0]]), "out of range"),
        (dict(face_indices=[0], weights=[[np.nan, 0, 0]]), "finite"),
    ],
)
def test_partition_rejects_invalid(mesh, kwargs, match):
    surface = VoronoiSurface(*mesh)
    with pytest.raises(ValueError, match=match):
        surface.partition(**kwargs)


@pytest.mark.parametrize(
    "coords, match",
    [([[np.inf, 0, 0]], "coords must contain only finite"), ([1, 0, 0], "shape")],
)
def test_partition_rejects_invalid_coords(mesh, coords, match):
    surface = VoronoiSurface(*mesh)
    with pytest.raises(ValueError, match=match):
        surface.partition(coords)


def test_partition_requires_sphere(mesh, points):
    coords, faces, _ = mesh
    surface = VoronoiSurface(coords, faces)
    with pytest.raises(ValueError, match="requires sphere"):
        surface.partition(points)


@pytest.fixture
def square():
    """A flat unit square of two faces, where many distances tie exactly."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    return coords, np.array([[0, 1, 2], [1, 3, 2]])


def _node_at(surface, xy):
    """The index of the graph node at a position of the unit square."""
    (node,) = np.flatnonzero(np.all(surface._node_coords[:, :2] == xy, axis=1))
    return node


@pytest.mark.parametrize("n_div", [2, 4, 8])
def test_lowest_index_wins_ties(square, n_div):
    coords, faces = square
    surface = VoronoiSurface(coords, faces, edge_divisions=n_div)
    labels = surface.native._point_set.nn
    # The middle of an edge is as far from both ends, and the center of the
    # square from all four vertices.
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        assert labels[_node_at(surface, (coords[a, :2] + coords[b, :2]) / 2)] == a
    assert labels[_node_at(surface, (0.5, 0.5))] == 0

    # Two points mirrored across the line y = x. The nodes on that line are
    # as far from both points, and the second face is reached through one of
    # them, the center of the square.
    weights = np.array([[0.4, 0.4, 0.2], [0.4, 0.2, 0.4]])
    tied = [_node_at(surface, xy) for xy in [(0, 0), (0.5, 0.5), (1, 1)]]
    areas = []
    for order in ([0, 1], [1, 0]):
        partition = surface.partition(face_indices=[0, 0], weights=weights[order])
        np.testing.assert_array_equal(partition._point_set.nn[tied], 0)
        areas.append(partition.areas())
    # The first point takes the tied part, whichever of the two it is.
    assert areas[0][0] > 0.5 > areas[0][1]
    np.testing.assert_allclose(areas[0], areas[1], atol=0.01)


def _raw_labels(surface, partition=None, graph=None):
    """SciPy's distances and labels for a partition, before the tie rule."""
    n_nodes = len(surface._node_coords)
    graph = surface._graph if graph is None else graph
    sources = np.unique(surface.faces)
    if partition is not None:
        nodes, lengths = _point_connections(surface, partition)
        n = len(nodes)
        indptr = np.arange(0, nodes.size + 1, nodes.shape[1])
        point_rows = sparse.csr_matrix(
            (lengths.ravel(), nodes.ravel(), indptr), shape=(n, n_nodes + n)
        )
        mesh_rows = sparse.csr_matrix(
            (graph.data, graph.indices, graph.indptr), shape=(n_nodes, n_nodes + n)
        )
        graph = sparse.vstack([mesh_rows, point_rows], format="csr")
        sources = np.arange(n_nodes, n_nodes + n)
    distances, _, labels = csgraph.dijkstra(
        graph, indices=sources, return_predecessors=True, min_only=True
    )
    if partition is not None:
        labels = np.where(labels >= 0, labels - n_nodes, -1)
    return distances[:n_nodes], labels[:n_nodes].astype(np.int32)


@pytest.mark.parametrize("n_div", [1, 2, 4])
@pytest.mark.parametrize("flat", [True, False])
def test_labels_do_not_depend_on_scipy(mesh, square, n_div, flat):
    """Not on which tied source SciPy returns, and not on the order of the
    entries of the graph."""
    coords, faces = square if flat else mesh[:2]
    surface = VoronoiSurface(coords, faces, edge_divisions=n_div)
    rng = np.random.default_rng(0)
    f_indices = rng.integers(0, len(faces), 12)
    weights = rng.dirichlet(np.ones(3), 12)
    if flat:
        # Points on vertices, edges and the diagonal, some of them twice.
        weights = rng.choice([0, 0.25, 0.5], (12, 3))
        weights[:, 2] = 1 - weights[:, :2].sum(axis=1)
    points = surface.partition(face_indices=f_indices, weights=weights)

    graph = surface._graph
    n_ties = 0
    for partition, connections in [
        (None, ()),
        (points, _point_connections(surface, points)),
    ]:
        point_set = (partition or surface.native)._point_set
        distances, labels = _raw_labels(surface, partition)
        np.testing.assert_array_equal(point_set.nnd, distances)
        np.testing.assert_array_equal(
            point_set.nn, _lowest_labels(graph, distances, labels, *connections)
        )
        # Any other valid choice gives the same labels. The highest tied
        # source is the lowest one with the labels in reverse order.
        n = point_set.size
        reversed_labels = voronoi._lowest_labels(
            graph, distances, n - 1 - labels, *[x[::-1] for x in connections]
        )
        highest = n - 1 - reversed_labels
        n_ties += np.count_nonzero(highest != point_set.nn)
        resolved = voronoi._lowest_labels(graph, distances, highest, *connections)
        np.testing.assert_array_equal(resolved, point_set.nn)

        # The same graph with the entries of each row in another order.
        shuffled = graph.copy()
        for start, stop in zip(shuffled.indptr[:-1], shuffled.indptr[1:]):
            order = start + rng.permutation(stop - start)
            shuffled.indices[start:stop] = graph.indices[order]
            shuffled.data[start:stop] = graph.data[order]
        distances2, labels2 = _raw_labels(surface, partition, shuffled)
        np.testing.assert_array_equal(distances2, distances)
        resolved = voronoi._lowest_labels(shuffled, distances2, labels2, *connections)
        np.testing.assert_array_equal(resolved, point_set.nn)
    if n_div > 1:
        assert n_ties > 0


def test_lowest_labels_unreachable(mesh):
    coords, faces, _ = mesh
    # A second copy of the mesh, not connected to the first.
    surface = VoronoiSurface(
        np.concatenate([coords, coords + 10]),
        np.concatenate([faces, faces + len(coords)]),
        edge_divisions=2,
    )
    point_set = surface._label_points(np.array([0, 3]), np.full((2, 3), 1 / 3))
    n_reached = np.count_nonzero(point_set.nn >= 0)
    assert 0 < n_reached < len(point_set.nn)
    np.testing.assert_array_equal(np.isfinite(point_set.nnd), point_set.nn >= 0)
    np.testing.assert_array_equal(np.unique(point_set.nn), [-1, 0, 1])


def _expected_samples(partition, f_indices=None, weights=None):
    """The sample labels of every face from compute_occupation."""
    surface, point_set = partition.surface, partition._point_set
    _, e_mapping, _ = reference.subdivide_edges(
        surface.coords, surface.faces, surface.edge_divisions
    )
    f_inv = {}
    if f_indices is not None:
        f_inv = reference.inverse_face_mapping(
            f_indices, weights, surface.coords, surface.faces
        )
    expected = np.empty((len(surface.faces), len(surface._ww)), dtype=int)
    for f_idx, f in enumerate(surface.faces):
        a, b, c = sorted(f)
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        occupation = reference.compute_occupation(
            f_idx,
            f,
            surface._node_coords,
            indices,
            point_set.nn,
            point_set.nnd,
            f_inv,
            surface._ww,
        )
        for u, mask in occupation.items():
            expected[f_idx, mask] = u
    return expected


def _sample_labels(partition):
    """The label of every sample of every face, as the pass computes them."""
    surface, point_set = partition.surface, partition._point_set
    faces = np.arange(len(surface.faces))
    return surface._face_labels(
        point_set,
        surface._single_labels(point_set),
        faces,
        *surface._boundary_distances(faces),
    )


@pytest.mark.parametrize("n_div", [1, 2, 8])
@pytest.mark.parametrize("t_div", [1, 2, 8])
@pytest.mark.parametrize("n_points", [0, 5, 40])
def test_sample_labels(mesh, n_div, t_div, n_points):
    coords, faces, sphere = mesh
    surface = VoronoiSurface(
        coords, faces, edge_divisions=n_div, sample_divisions=t_div
    )
    partition, placement = surface.native, ()
    if n_points:
        rng = np.random.default_rng(n_points)
        weights = rng.dirichlet(np.ones(3), n_points)
        f_indices = rng.integers(0, len(faces), n_points)
        placement = (f_indices, weights)
        partition = surface.partition(face_indices=f_indices, weights=weights)
    labels = _sample_labels(partition)
    assert labels.shape == (len(faces), t_div**2)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, _expected_samples(partition, *placement))
    point_set = partition._point_set
    is_single = surface._single_labels(point_set) >= 0
    assert not np.any(is_single & (np.diff(point_set.inside_ptr) > 0))
    if n_points == 5:
        assert 0 < is_single.sum() < len(faces)


@pytest.mark.parametrize("n_div", [1, 2, 4])
@pytest.mark.parametrize("t_div", [2, 4])
def test_sample_labels_ties(square, n_div, t_div):
    """A flat, symmetric mesh where many scores tie exactly."""
    coords, faces = square
    surface = VoronoiSurface(
        coords, faces, edge_divisions=n_div, sample_divisions=t_div
    )
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
    points = surface.partition(face_indices=f_indices, weights=weights)
    for partition, placement in [(surface.native, ()), (points, (f_indices, weights))]:
        expected = _expected_samples(partition, *placement)
        np.testing.assert_array_equal(_sample_labels(partition), expected)


def test_sample_labels_many_inside(mesh):
    """One face with more inside points than samples."""
    coords, faces, _ = mesh
    surface = VoronoiSurface(coords, faces, sample_divisions=4)
    rng = np.random.default_rng(0)
    others = np.delete(np.arange(len(faces)), 3)
    f_indices = np.concatenate([np.full(300, 3), others])
    weights = rng.dirichlet(np.ones(3), len(f_indices))
    partition = surface.partition(face_indices=f_indices, weights=weights)
    expected = _expected_samples(partition, f_indices, weights)
    np.testing.assert_array_equal(_sample_labels(partition), expected)


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


def test_barycentric_points():
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(3), (5, 7))
    corners = rng.standard_normal((5, 1, 3, 3)) * 100
    points = voronoi._barycentric_points(weights, corners)
    assert points.shape == (5, 7, 3)
    np.testing.assert_allclose(points, np.matmul(weights, corners[:, 0]), rtol=1e-13)
    # Every operation is rounded on its own, from left to right.
    for i, j, k in [(0, 0, 0), (4, 6, 2), (2, 3, 1)]:
        w, c = weights[i, j].tolist(), corners[i, 0, :, k].tolist()
        assert points[i, j, k] == w[0] * c[0] + w[1] * c[1] + w[2] * c[2]
    corners = np.eye(3)
    np.testing.assert_array_equal(
        voronoi._barycentric_points(corners, corners), corners
    )


def _assert_identical(mat, expected):
    """Assert exact equality of sparse shapes, indices, and values."""
    mat, expected = mat.tocsr(), expected.tocsr()
    for m in (mat, expected):
        m.sort_indices()
    assert mat.shape == expected.shape
    np.testing.assert_array_equal(mat.indptr, expected.indptr)
    np.testing.assert_array_equal(mat.indices, expected.indices)
    np.testing.assert_array_equal(mat.data, expected.data)


def _reference(partitions, placements, source, target):
    """The overlap from compute_overlap, with the labels of the partitions."""
    surface = partitions[source].surface
    _, e_mapping, _ = reference.subdivide_edges(
        surface.coords, surface.faces, surface.edge_divisions
    )
    args = []
    for name in (source, target):
        point_set = partitions[name]._point_set
        f_inv = {}
        if name != "native":
            f_inv = reference.inverse_face_mapping(
                *placements[name], surface.coords, surface.faces
            )
        args.append((point_set.nn, point_set.nnd, f_inv))
    return reference.compute_overlap(
        surface.faces,
        surface.face_areas,
        e_mapping,
        surface._node_coords,
        *args[0],
        *args[1],
        partitions[source].n_cells,
        partitions[target].n_cells,
        t_div=surface.sample_divisions,
    )


def _partitions(mesh, n_div=8, t_div=4, sizes=(("few", 5), ("many", 40))):
    """The native partition and others by name, and the placements of those."""
    coords, faces, _ = mesh
    surface = VoronoiSurface(
        coords, faces, edge_divisions=n_div, sample_divisions=t_div
    )
    partitions, placements = {"native": surface.native}, {}
    for name, n in sizes:
        rng = np.random.default_rng(n)
        placements[name] = (
            rng.integers(0, len(faces), n),
            rng.dirichlet(np.ones(3), n),
        )
        partitions[name] = surface.partition(
            face_indices=placements[name][0], weights=placements[name][1]
        )
    return partitions, placements


@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_overlap_matches_compute_overlap(mesh, n_div, t_div):
    partitions, placements = _partitions(mesh, n_div, t_div)
    face_areas = partitions["native"].surface.face_areas
    for source, a in partitions.items():
        for target, b in partitions.items():
            mat = a.overlap(b)
            _assert_identical(mat, _reference(partitions, placements, source, target))
            _assert_identical(b.overlap(a), mat.T)
            assert mat.shape == (a.n_cells, b.n_cells)
            np.testing.assert_allclose(mat.sum(axis=1).A1, a.areas())
            np.testing.assert_allclose(mat.sum(), face_areas.sum())


@pytest.mark.parametrize("n_div", [1, 8])
@pytest.mark.parametrize("t_div", [1, 2, 3, 8])
def test_areas(mesh, n_div, t_div):
    partitions, placements = _partitions(mesh, n_div, t_div)
    face_areas = partitions["native"].surface.face_areas
    for name, partition in partitions.items():
        expected = np.zeros(partition.n_cells)
        occupation = _expected_samples(partition, *placements.get(name, ()))
        for f_idx, labels in enumerate(occupation):
            for u in np.unique(labels):
                expected[u] += (labels == u).mean() * face_areas[f_idx]
        np.testing.assert_array_equal(partition.areas(), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_div, t_div", [(8, 32), (1, 1), (2, 3)])
def test_vertex_areas_nn(mesh, dtype, n_div, t_div):
    coords, faces, _ = mesh
    surf = Surface(coords.astype(dtype), faces)
    surface = VoronoiSurface(
        coords.astype(dtype), faces, edge_divisions=n_div, sample_divisions=t_div
    )
    areas = surf.vertex_areas_nn(n_div=n_div, t_div=t_div)
    np.testing.assert_array_equal(areas, surface.native.areas())
    np.testing.assert_allclose(areas.sum(), surf.face_areas.sum())
    if (n_div, t_div) == (8, 32):
        np.testing.assert_array_equal(surf.vertex_areas_nn(), areas)


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
def test_entries_chunk_size(mesh, chunk_size):
    """Several requests in one pass give the entries of each request alone."""
    partitions, _ = _partitions(mesh)
    native, few, many = partitions.values()
    requests = [(native,), (many,), (native, few), (many, native)]
    entries = native.surface._entries(requests, chunk_size=chunk_size)
    for request, request_entries in zip(requests, entries):
        for a, b in zip(request_entries, native.surface._entries([request])[0]):
            np.testing.assert_array_equal(a, b)


SIX_SETS = (("a", 3), ("b", 5), ("c", 17), ("d", 40), ("e", 200))


def test_overlaps_all_pairs(mesh):
    """All ordered pairs of six partitions in one pass."""
    partitions, _ = _partitions(mesh, sizes=SIX_SETS)
    pairs = [(a, b) for a in partitions.values() for b in partitions.values()]
    mats = partitions["native"].surface.overlaps(pairs)
    assert len(mats) == 36
    for (a, b), mat in zip(pairs, mats):
        _assert_identical(mat, a.overlap(b))


def test_overlaps_composition_and_order(mesh):
    """A matrix does not depend on the other pairs of the call or on their
    order."""
    partitions, _ = _partitions(mesh, sizes=SIX_SETS)
    pairs = [(a, b) for a in partitions.values() for b in partitions.values()]
    expected = {(a, b): a.overlap(b) for a, b in pairs}
    rng = np.random.default_rng(0)
    for _ in range(10):
        # With repeated pairs.
        indices = rng.integers(0, len(pairs), rng.integers(1, 8))
        request = [pairs[i] for i in indices]
        mats = partitions["native"].surface.overlaps(request)
        assert len(mats) == len(request)
        for pair, mat in zip(request, mats):
            _assert_identical(mat, expected[pair])


def test_overlaps_special_requests(mesh):
    partitions, _ = _partitions(mesh)
    native, many = partitions["native"], partitions["many"]
    pairs = [(native, many), (many, native), (native, many), (many, many)]
    mat, reverse, repeated = native.surface.overlaps(pairs)[:3]
    _assert_identical(reverse, mat.T)
    _assert_identical(repeated, mat)
    expected = mat.copy()
    for x in (repeated, reverse):
        x.data[:] = 0
        x.indices[:] = 0
        x.indptr[:] = 0
    _assert_identical(mat, expected)
    _assert_identical(native.overlap(many), expected)
    # A list is accepted as a pair, and any iterable as the pairs.
    _assert_identical(native.surface.overlaps(iter([[native, many]]))[0], expected)


def test_self_overlap(mesh):
    partitions, _ = _partitions(mesh)
    pairs = [(partition, partition) for partition in partitions.values()]
    for (partition, _), mat in zip(pairs, partitions["native"].surface.overlaps(pairs)):
        diagonal = mat.diagonal()
        assert mat.nnz == np.count_nonzero(diagonal)
        np.testing.assert_array_equal(diagonal, partition.areas())


def test_overlaps_rejects_invalid(mesh):
    coords, faces, _ = mesh
    surface, other = VoronoiSurface(coords, faces), VoronoiSurface(coords, faces)
    kwargs = dict(face_indices=[0, 5], weights=np.full((2, 3), 1 / 3))
    a, b = surface.partition(**kwargs), other.partition(**kwargs)
    for pairs in [[a], [(a,)], [(a, a, a)], (a, a), [(a, "native")], [(a, a), None]]:
        with pytest.raises(TypeError, match="pair of partitions"):
            surface.overlaps(pairs)
    for pairs in [[(a, b)], [(b, a)], [(a, a), (b, b)], [(a, other.native)]]:
        with pytest.raises(ValueError, match="belongs to another surface"):
            surface.overlaps(pairs)
    with pytest.raises(ValueError, match="belongs to another surface"):
        a.overlap(b)
    assert surface.overlaps([]) == []
    # Nothing is computed for an invalid or empty request.
    assert surface._native is None


@pytest.mark.parametrize("scale", [2.0, 0.25])
def test_scaling_by_power_of_two_is_exact(mesh, scale):
    coords, faces, sphere = mesh
    partitions, _ = _partitions(mesh)
    scaled, _ = _partitions((coords * scale, faces, sphere))
    for source, target in [("native", "many"), ("few", "many")]:
        mat = partitions[source].overlap(partitions[target])
        mat.data *= scale**2
        _assert_identical(scaled[source].overlap(scaled[target]), mat)
    np.testing.assert_array_equal(
        scaled["many"].areas(), partitions["many"].areas() * scale**2
    )


def test_pass_shares_distances(mesh, monkeypatch, labelling_path):
    """The partitions of a call share the distances of each face."""
    partitions, _ = _partitions(mesh, sizes=[("few", 5), ("some", 10)])
    _, few, some = partitions.values()
    surface = few.surface
    # Count the faces whose distances are computed, in either path.
    calls = []
    if labelling_path == "kernel":
        label_samples = voronoi._kernel.label_samples

        def counting(*args):
            calls.append(len(args[5]))
            return label_samples(*args)

        kernel = types.SimpleNamespace(
            label_samples=counting, count_labels=voronoi._kernel.count_labels
        )
        monkeypatch.setattr(voronoi, "_kernel", kernel)
    else:
        boundary_distances = surface._boundary_distances
        monkeypatch.setattr(
            surface,
            "_boundary_distances",
            lambda faces: calls.append(len(faces)) or boundary_distances(faces),
        )
    is_mixed = [surface._single_labels(p._point_set) < 0 for p in (few, some)]
    n_visited = (is_mixed[0] | is_mixed[1]).sum()
    assert max(is_mixed[0].sum(), is_mixed[1].sum()) < n_visited
    for requests in [[(few, some)], [(few, some), (some, few), (few,)]]:
        calls.clear()
        surface._entries(requests, chunk_size=7)
        assert sum(calls) == n_visited < len(surface.faces)
        assert max(calls) == 7


def test_paths_give_the_same_entries(mesh, monkeypatch, labelling_path):
    """The compiled kernel and NumPy, for several requests in one pass."""
    if labelling_path == "numpy":
        pytest.skip("Compares the two paths once.")
    partitions, _ = _partitions(mesh, sizes=SIX_SETS)
    surface = partitions["native"].surface
    requests = [(a, b) for a in partitions.values() for b in partitions.values()]
    requests += [(a,) for a in partitions.values()]
    entries = surface._entries(requests, chunk_size=5)
    monkeypatch.setattr(voronoi, "_kernel", None)
    for a, b in zip(entries, surface._entries(requests, chunk_size=5)):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("n_jobs", [2, 3, -1])
def test_threads_do_not_change_results(mesh, n_jobs):
    partitions, _ = _partitions(mesh, sizes=SIX_SETS)
    surface = partitions["native"].surface
    requests = [(a, b) for a in partitions.values() for b in partitions.values()]
    requests += [(a,) for a in partitions.values()]
    expected = surface._entries(requests, chunk_size=5)
    for a, b in zip(surface._entries(requests, chunk_size=5, n_jobs=n_jobs), expected):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    native, d = partitions["native"], partitions["d"]
    pairs = [(native, d), (d, d), (d, native)]
    for mat, expected in zip(
        surface.overlaps(pairs, n_jobs=n_jobs), surface.overlaps(pairs)
    ):
        _assert_identical(mat, expected)
    _assert_identical(native.overlap(d, n_jobs=n_jobs), native.overlap(d))
    np.testing.assert_array_equal(d.areas(n_jobs=n_jobs), d.areas())
    with pytest.raises(ValueError, match="n_jobs"):
        native.areas(n_jobs=0)


def test_outputs_are_new_arrays(mesh):
    partitions, _ = _partitions(mesh)
    native, many = partitions["native"], partitions["many"]
    mat = native.overlap(many)
    areas = many.areas()
    expected_mat, expected_areas = mat.copy(), areas.copy()
    mat.data[:] = 0
    areas[:] = 0
    _assert_identical(native.overlap(many), expected_mat)
    _assert_identical(many.overlap(native), expected_mat.T)
    np.testing.assert_array_equal(many.areas(), expected_areas)


def test_overlap_large_labels(mesh):
    """Pair keys exceed the int32 range."""
    partitions, placements = _partitions(
        mesh, n_div=1, t_div=2, sizes=[("large", 70000)]
    )
    large = partitions["large"]
    assert large.n_cells**2 > np.iinfo(np.int32).max
    mat = large.overlap(large)
    _assert_identical(mat, _reference(partitions, placements, "large", "large"))
    assert mat.indices.max() > 65536


def test_unreachable_faces(mesh):
    coords, faces, _ = mesh
    # A second copy of the mesh, not connected to the first.
    surface = VoronoiSurface(
        np.concatenate([coords, coords + 10]),
        np.concatenate([faces, faces + len(coords)]),
    )
    native = surface.native
    np.testing.assert_allclose(native.areas().sum(), surface.face_areas.sum())
    # A partition needs a point in each connected part.
    weights = [[1, 0, 0]]
    with pytest.raises(ValueError, match="cannot be reached"):
        surface.partition(face_indices=[0], weights=weights)
    both = surface.partition(face_indices=[0, len(faces)], weights=weights * 2)
    np.testing.assert_allclose(both.areas(), surface.face_areas.sum() / 2)


def test_empty_surface():
    """An empty mesh has no cell areas or overlaps."""
    surface = VoronoiSurface(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    native = surface.native
    assert native.n_cells == 0
    assert native.areas().shape == (0,)
    overlap = native.overlap(native)
    assert overlap.shape == (0, 0)
    assert overlap.nnz == 0


def test_vertex_without_face(mesh):
    """A vertex that no face uses has an empty cell."""
    coords, faces, _ = mesh
    surface = VoronoiSurface(np.concatenate([coords, [[5.0, 5.0, 5.0]]]), faces)
    expected = VoronoiSurface(coords, faces)
    areas = surface.native.areas()
    np.testing.assert_array_equal(areas[:-1], expected.native.areas())
    assert areas[-1] == 0
    partition = surface.partition(face_indices=[0, 3], weights=np.full((2, 3), 1 / 3))
    assert partition.overlap(surface.native).shape == (2, len(coords) + 1)


def test_native_partition(mesh):
    coords, faces, _ = mesh
    surface = VoronoiSurface(coords, faces)
    assert surface._native is None
    native = surface.native
    assert isinstance(native, VoronoiPartition)
    assert native is surface.native
    assert native.surface is surface
    assert native.n_cells == len(coords)
    assert repr(native) == "<VoronoiPartition with 30 cells>"
    for attribute in ("surface", "n_cells"):
        with pytest.raises(AttributeError):
            setattr(native, attribute, None)
    with pytest.raises(AttributeError):
        native.name = "native"


def test_surface_is_released_with_its_partitions(mesh):
    """No reference cycle keeps the graph until the next garbage collection."""
    coords, faces, _ = mesh
    gc.collect()
    gc.disable()
    try:
        surface = VoronoiSurface(coords, faces)
        native = surface.native
        partition = surface.partition(face_indices=[0], weights=[[1, 0, 0]])
        native.overlap(partition)
        released = weakref.ref(surface)
        # The surface does not keep the partitions that it creates.
        dropped = weakref.ref(surface.partition(face_indices=[1], weights=[[1, 0, 0]]))
        assert dropped() is None
        del surface
        # Each partition keeps its surface.
        assert partition.surface is native.surface is released()
        del native
        assert released() is not None
        del partition
        assert released() is None
    finally:
        gc.enable()
