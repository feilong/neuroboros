import pickle
import warnings

import numpy as np
import pytest

from neuroboros.surface import Sphere, Surface, voronoi, voronoi_legacy

# The deprecated functions keep their behaviour and are tested here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:neuroboros.surface.voronoi:DeprecationWarning"
)


@pytest.fixture
def subdivided_mesh():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    new_coords, mapping, neighbors = voronoi.subdivide_edges(coords, faces, 3)
    return np.concatenate([coords, new_coords]), faces, mapping, neighbors


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("max_dist", [None, 0.01])
@pytest.mark.parametrize("verbose", [False, True])
def test_subdivision_voronoi_legacy_arguments(
    subdivided_mesh, positional, max_dist, verbose
):
    args = (*subdivided_mesh, np.array([0]), np.array([[0.2, 0.3, 0.5]]))
    expected_nn, expected_nnd = voronoi.subdivision_voronoi(*args)
    if positional:
        nn, nnd = voronoi.subdivision_voronoi(*args, max_dist, verbose)
    else:
        nn, nnd = voronoi.subdivision_voronoi(*args, max_dist=max_dist, verbose=verbose)
    np.testing.assert_array_equal(nn, expected_nn)
    np.testing.assert_allclose(nnd, expected_nnd)
    assert np.all(nn == 0)
    assert np.all(np.isfinite(nnd))


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
def test_native_voronoi_legacy_arguments(subdivided_mesh, positional, verbose):
    expected_nn, expected_nnd = voronoi.native_voronoi(*subdivided_mesh)
    if positional:
        nn, nnd = voronoi.native_voronoi(*subdivided_mesh, verbose)
    else:
        nn, nnd = voronoi.native_voronoi(*subdivided_mesh, verbose=verbose)
    np.testing.assert_array_equal(nn, expected_nn)
    np.testing.assert_allclose(nnd, expected_nnd)
    np.testing.assert_array_equal(nn[:4], np.arange(4))
    np.testing.assert_array_equal(nnd[:4], np.zeros(4))


def test_dijkstra_distances_original_import_path():
    from neuroboros.surface.voronoi import dijkstra_distances

    neighbors = {0: {1: 1.0}, 1: {0: 1.0, 2: 2.0}, 2: {1: 2.0}}
    np.testing.assert_array_equal(
        dijkstra_distances(3, [(0.0, 0)], neighbors), [0.0, 1.0, 3.0]
    )
    np.testing.assert_array_equal(
        dijkstra_distances(3, [(0.0, 0)], neighbors, max_dist=1.5),
        [0.0, 1.0, np.inf],
    )


@pytest.mark.parametrize("n_div", [1, 2, 3, 8])
def test_subdivide_edges_matches_legacy(n_div):
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((4, 3))
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    new_coords, mapping, neighbors = voronoi.subdivide_edges(coords, faces, n_div)
    expected = voronoi_legacy.subdivide_edges(coords, faces, n_div)
    np.testing.assert_array_equal(new_coords, expected[0])
    assert mapping.keys() == expected[1].keys()
    for e, indices in mapping.items():
        np.testing.assert_array_equal(indices, expected[1][e])
    nv = coords.shape[0] + new_coords.shape[0]
    expected_graph = voronoi.neighbors_to_graph(expected[2], nv)
    assert (neighbors != expected_graph).nnz == 0


def test_subdivide_edges_integer_coords():
    coords = np.array(
        [[0, 0, 0], [50000, 0, 0], [0, 50000, 0], [0, 0, 50000]], dtype=np.int32
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    neighbors = voronoi.subdivide_edges(coords, faces, 2)[2]
    expected = voronoi.subdivide_edges(coords.astype(float), faces, 2)[2]
    assert np.all(np.isfinite(neighbors.data))
    assert (neighbors != expected).nnz == 0


@pytest.mark.parametrize("duplicate", [[0, 1, 2], [2, 1, 0]])
def test_subdivide_edges_duplicate_faces(duplicate):
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((4, 3))
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    expected = voronoi.subdivide_edges(coords, faces, 3)
    new_coords, mapping, neighbors = voronoi.subdivide_edges(
        coords, np.vstack([faces, duplicate]), 3
    )
    np.testing.assert_array_equal(new_coords, expected[0])
    assert mapping.keys() == expected[1].keys()
    assert (neighbors != expected[2]).nnz == 0


@pytest.mark.parametrize("n_div", [1, 2, 8])
def test_subdivided_graph_arrays(n_div):
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((4, 3))
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    edges, face_edge_ids, edge_points, new_coords, neighbors = (
        voronoi._subdivided_graph(coords, faces, n_div)
    )
    expected = voronoi.subdivide_edges(coords, faces, n_div)
    np.testing.assert_array_equal(new_coords, expected[0])
    assert (neighbors != expected[2]).nnz == 0
    assert edges.shape == (6, 2)
    assert edge_points.shape == (6, n_div - 1)
    assert list(map(tuple, edges.tolist())) == list(expected[1])
    for e, indices in zip(edges, edge_points):
        np.testing.assert_array_equal(indices, expected[1][tuple(e)])
    pairs = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    np.testing.assert_array_equal(edges[face_edge_ids], pairs)


DEPRECATED = [
    "subdivide_edges",
    "neighbors_to_coo",
    "neighbors_to_graph",
    "subdivision_voronoi",
    "native_voronoi",
    "inverse_face_mapping",
    "split_triangle",
    "compute_occupation",
    "compute_overlap",
    "overlap_transform",
    "dijkstra_distances",
]


@pytest.fixture
def deprecated_calls():
    """A call of each deprecated function on a tetrahedron."""
    sphere = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    new_coords, e_mapping, graph = voronoi._subdivide_edges(sphere, faces, 2)
    coords = np.concatenate([sphere, new_coords])
    nn, nnd = voronoi._native_voronoi(coords, faces, e_mapping, graph)
    legacy_graph = voronoi_legacy.subdivide_edges(sphere, faces, 2)[2]
    point = sphere[:3].mean(axis=0, keepdims=True)
    f_indices, weights = np.array([0]), np.full((1, 3), 1 / 3)
    surf = Surface(sphere, faces)
    edge_nodes = [e_mapping[(0, 1)], e_mapping[(0, 2)], e_mapping[(1, 2)]]
    indices = np.concatenate(edge_nodes + [[0, 1, 2]])
    ww = voronoi._split_triangle(2)
    native = (nn, nnd, {})
    return {
        "subdivide_edges": lambda f: f(sphere, faces, 2),
        "neighbors_to_coo": lambda f: f(legacy_graph),
        "neighbors_to_graph": lambda f: f(legacy_graph, len(coords)),
        "subdivision_voronoi": lambda f: f(
            coords, faces, e_mapping, graph, f_indices, weights
        ),
        "native_voronoi": lambda f: f(coords, faces, e_mapping, graph),
        "inverse_face_mapping": lambda f: f(f_indices, weights, sphere, faces),
        "split_triangle": lambda f: f(2),
        "compute_occupation": lambda f: f(0, faces[0], coords, indices, *native, ww),
        "compute_overlap": lambda f: f(
            faces, surf.face_areas, e_mapping, coords, *native, *native, 4, 4, t_div=2
        ),
        "overlap_transform": lambda f: f(
            surf, Sphere(sphere, faces), point, coords, e_mapping, graph, nn, nnd, 2
        ),
        "dijkstra_distances": lambda f: f(2, [(0.0, 0)], {0: {1: 1.0}, 1: {0: 1.0}}),
    }


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecation_warning(deprecated_calls, name):
    func = getattr(voronoi, name)
    assert func.__name__ == name
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        deprecated_calls[name](func)
    # One warning, attributed to the caller: internal calls do not warn.
    assert [r.category for r in records] == [DeprecationWarning]
    message = str(records[0].message)
    assert message.startswith(f"neuroboros.surface.voronoi.{name} is deprecated")
    assert records[0].filename == __file__


@pytest.mark.parametrize("name", DEPRECATED)
def test_deprecated_pickle(name):
    func = getattr(voronoi, name)
    assert pickle.loads(pickle.dumps(func)) is func
