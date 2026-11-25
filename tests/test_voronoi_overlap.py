import numpy as np
import pytest
from scipy.spatial import ConvexHull

from neuroboros.surface import Surface, voronoi
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
