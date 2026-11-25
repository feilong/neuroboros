import numpy as np
import pytest

from neuroboros.surface import voronoi


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
        nn, nnd = voronoi.subdivision_voronoi(
            *args, max_dist=max_dist, verbose=verbose
        )
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
    from neuroboros.surface.voronoi_legacy import (
        dijkstra_distances as legacy_dijkstra_distances,
    )

    assert dijkstra_distances is legacy_dijkstra_distances
    neighbors = {0: {1: 1.0}, 1: {0: 1.0, 2: 2.0}, 2: {1: 2.0}}
    np.testing.assert_array_equal(
        dijkstra_distances(3, [(0.0, 0)], neighbors), [0.0, 1.0, 3.0]
    )
    np.testing.assert_array_equal(
        dijkstra_distances(3, [(0.0, 0)], neighbors, max_dist=1.5),
        [0.0, 1.0, np.inf],
    )
