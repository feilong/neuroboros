import functools
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from .properties import compute_face_areas
from .voronoi_legacy import dijkstra_distances as _legacy_dijkstra_distances


def _row_norms(x):
    """Compute row norms using the dot product used by single-row norms."""
    if not np.issubdtype(x.dtype, np.inexact):
        x = x.astype(float)
    squared_norms = np.matmul(x[:, np.newaxis, :], x[:, :, np.newaxis])
    return np.sqrt(squared_norms[:, 0, 0])


def _mesh_edges(faces, nv):
    """Return unique edges in encounter order and each face's three edge IDs.

    Edge endpoints are sorted by vertex index. The returned arrays have
    shapes (n_edges, 2) and (n_faces, 3), respectively; face edges follow the
    local vertex pairs (0, 1), (1, 2), (2, 0).
    """
    face_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    edge_keys = face_edges[:, 0].astype(np.int64) * nv + face_edges[:, 1]
    _, first_occurrence, sorted_edge_ids = np.unique(
        edge_keys, return_index=True, return_inverse=True
    )

    # np.unique numbers edges by key. Remap those IDs to encounter order to
    # preserve the vertex numbering used by the legacy implementation.
    encounter_order = np.argsort(first_occurrence)
    edges = face_edges[first_occurrence[encounter_order]]
    sorted_to_encounter = np.empty_like(encounter_order)
    sorted_to_encounter[encounter_order] = np.arange(len(encounter_order))
    face_edge_ids = sorted_to_encounter[sorted_edge_ids].reshape(-1, 3)
    return edges, face_edge_ids


def _connection_distances(coords, rows, cols):
    """Compute connection lengths in chunks to limit temporary coordinate arrays."""
    distances = np.empty(rows.shape)
    chunk_size = 2**22
    for start in range(0, len(rows), chunk_size):
        chunk = slice(start, start + chunk_size)
        distances[chunk] = _row_norms(coords[rows[chunk]] - coords[cols[chunk]])
    return distances


def _subdivided_graph(coords, faces, n_div):
    """Subdivide each edge into ``n_div`` parts and build the distance graph.

    Returns
    -------
    edges : ndarray of shape (n_edges, 2)
        Unique edges in encounter order, with sorted vertex indices.
    face_edge_ids : ndarray of shape (nf, 3)
        The edges of each face, for the local vertex pairs (0, 1), (1, 2),
        (2, 0).
    edge_points : ndarray of shape (n_edges, n_div - 1)
        The indices of the new vertices on each edge, from the lower to the
        higher endpoint index. Indices start from ``nv``.
    new_coords : ndarray of shape (nv_new, 3)
        The coordinates of the new vertices.
    neighbors : scipy.sparse.csr_matrix of shape (nv + nv_new, nv + nv_new)
        The graph described in ``subdivide_edges``.
    """
    nv = coords.shape[0]
    n_edge_points = n_div - 1
    steps = np.arange(1, n_div)[:, np.newaxis]

    edges, face_edge_ids = _mesh_edges(faces, nv)

    # Each row contains the new vertex indices along one edge, from low to
    # high endpoint index. Interpolation has shape (n_edges, n_edge_points, 3).
    edge_points = nv + np.arange(len(edges) * n_edge_points).reshape(
        len(edges), n_edge_points
    )
    start_coords = coords[edges[:, 0], np.newaxis]
    end_coords = coords[edges[:, 1], np.newaxis]
    new_coords = (end_coords * steps + start_coords * (n_div - steps)) / n_div
    new_coords = new_coords.reshape(-1, 3).astype(coords.dtype)

    # Collect each connection once; add its reverse when building the graph.
    # Start with consecutive points along each edge, including both endpoints.
    chain = np.concatenate([edges[:, :1], edge_points, edges[:, 1:]], axis=1)
    row_blocks = [chain[:, :-1].ravel()]
    col_blocks = [chain[:, 1:].ravel()]
    segment_lengths = _row_norms(coords[edges[:, 1]] - coords[edges[:, 0]]) / n_div
    distance_blocks = [np.repeat(segment_lengths, n_div)]

    # Deduplicate faces so CSR construction does not sum repeated connections.
    _, first_face = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    first_face = np.sort(first_face)
    # Shape: (n_unique_faces, 3 edges, n_edge_points).
    face_points = edge_points[face_edge_ids[first_face]]
    all_coords = np.concatenate([coords, new_coords])

    # Connect each edge's points to the face vertex opposite that edge.
    opposite_vertices = faces[first_face][:, [2, 0, 1], np.newaxis]
    row_blocks.append(face_points.ravel())
    col_blocks.append(np.broadcast_to(opposite_vertices, face_points.shape).ravel())

    # Connect every pair of points on different edges of the same face. The
    # cyclic edge pairs (0, 1), (1, 2), (2, 0) cover each pair of edges once.
    pair_shape = face_points.shape + (n_edge_points,)
    row_blocks.append(np.broadcast_to(face_points[..., np.newaxis], pair_shape).ravel())
    col_blocks.append(
        np.broadcast_to(face_points[:, [1, 2, 0], np.newaxis], pair_shape).ravel()
    )

    # Segment lengths are already known; compute lengths for face connections.
    for rows, cols in zip(row_blocks[1:], col_blocks[1:]):
        distance_blocks.append(_connection_distances(all_coords, rows, cols))

    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    distances = np.concatenate(distance_blocks)
    # Release the blocks before allocating the symmetric graph entries.
    del row_blocks, col_blocks, distance_blocks
    n_vertices = all_coords.shape[0]
    neighbors = sparse.csr_matrix(
        (
            np.concatenate([distances, distances]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n_vertices, n_vertices),
    )
    return edges, face_edge_ids, edge_points, new_coords, neighbors


def _subdivide_edges(coords, faces, n_div):
    """Subdivide each edge into ``n_div`` parts.

    This function adds new vertices to the mesh. The new vertices are placed
    on each edge of the mesh. The number of new vertices on each edge is
    ``n_div - 1``, which means each edge is now divided into ``n_div`` parts.

    The new vertices are usually used to increase the accuracy of the
    estimated geodesic distances using Dijkstra's algorithm.

    Parameters
    ----------
    coords : ndarray of shape (nv, 3)
        The coordinates of the vertices.
    faces : ndarray of shape (nf, 3)
        The indices of the vertices of each triangle face.
    n_div : int
        The number of segments to divide each edge into.


    Returns
    -------
    new_coords : ndarray of shape (nv_new, 3)
        The coordinates of the new vertices. It should be concatenated to the
        original coordinates to get the full set of vertex coordinates.
    e_mapping : dict
        The mapping from the original edges to the new vertices. The keys are
        tuples of sorted vertex indices of the original edges. The values are
        the indices of the new vertices. Vertex indices start from ``nv``.
    neighbors : scipy.sparse.csr_matrix of shape (nv + nv_new, nv + nv_new)
        The graph connecting the original and new vertices. Each stored entry
        is the distance between two neighboring vertices. Consecutive points
        along each edge are connected, and each new vertex is connected to the
        new vertices on the other edges and to the opposite vertex of each
        face it is on.
    """
    edges, _, edge_points, new_coords, neighbors = _subdivided_graph(
        coords, faces, n_div
    )
    e_mapping = dict(zip(map(tuple, edges.tolist()), edge_points))
    return new_coords, e_mapping, neighbors


def _neighbors_to_coo(neighbors):
    """Convert the neighbors of each vertex to a graph in COO format.

    Parameters
    ----------
    neighbors : dict
        The neighbors of each vertex. The keys are the vertex indices. The
        values are dictionaries. The keys of the dictionaries are the indices
        of the neighboring vertices. The values of the dictionaries are the
        distances between the vertex and the neighboring vertices.

    Returns
    -------
    rows : ndarray of shape (n_edges,)
        The index of the first vertex of each edge.
    cols : ndarray of shape (n_edges,)
        The index of the second vertex of each edge.
    dists : ndarray of shape (n_edges,)
        The length of each edge.
    """
    counts = np.fromiter(
        (len(adjacent) for adjacent in neighbors.values()), int, len(neighbors)
    )
    n_connections = counts.sum()
    rows = np.repeat(np.fromiter(neighbors.keys(), int, len(neighbors)), counts)
    cols = np.fromiter(
        (vertex for adjacent in neighbors.values() for vertex in adjacent),
        int,
        n_connections,
    )
    dists = np.fromiter(
        (distance for adjacent in neighbors.values() for distance in adjacent.values()),
        float,
        n_connections,
    )
    return rows, cols, dists


def _neighbors_to_graph(neighbors, nv):
    """Convert the neighbors of each vertex to a graph in CSR format.

    Parameters
    ----------
    neighbors : scipy.sparse matrix or dict
        The graph returned by ``subdivide_edges``, or the neighbors of each
        vertex in the format of ``voronoi_legacy.subdivide_edges``.
    nv : int
        The number of vertices.

    Returns
    -------
    graph : scipy.sparse.csr_matrix of shape (nv, nv)
        Each stored entry is the distance between two neighboring vertices.
    """
    if sparse.issparse(neighbors):
        return sparse.csr_matrix(neighbors)
    rows, cols, dists = _neighbors_to_coo(neighbors)
    return sparse.csr_matrix((dists, (rows, cols)), shape=(nv, nv))


def _subdivision_voronoi(
    coords,
    faces,
    e_mapping,
    neighbors,
    f_indices,
    weights,
    max_dist=None,
    verbose=False,
):
    """Voronoi diagram on a subdivided mesh.

    This function estimates the nearest vertex on the target mesh for each
    vertex of the subdivided mesh. The subdivided mesh is used to increase the
    accuracy of the estimated geodesic distances using Dijkstra's algorithm.

    Barycentric interpolation is used to map the vertices of the target mesh
    to the subdivided mesh.

    Parameters
    ----------
    coords : ndarray of shape (nv, 3)
        The coordinates of the vertices of the subdivided mesh.
    faces : ndarray of shape (nf, 3)
        The vertex indices of each triangle face of the original mesh.
    e_mapping : dict
        The mapping from the original edges to the new vertices. The keys are
        tuples of sorted vertex indices of the original edges. The values are
        the indices of the new vertices.
    neighbors : scipy.sparse.csr_matrix or dict
        The graph returned by ``subdivide_edges``. The format of
        ``voronoi_legacy.subdivide_edges`` is also accepted.
    f_indices : ndarray of shape (nv_target,)
        Each element is the index of the triangle face on the original mesh
        that contains the corresponding vertex on the target mesh.
    weights : ndarray of shape (nv_target, 3)
        Each row is the barycentric weights of the three vertices of the face
        for the corresponding vertex on the target mesh.
    max_dist : float or None, default=None
        Retained for backward compatibility and ignored. The multi-source
        search computes distances without an initial search radius.
    verbose : bool, default=False
        Retained for backward compatibility and ignored. The multi-source
        search does not report per-source progress.

    Returns
    -------
    nn : ndarray of shape (nv,)
        The indices of the nearest vertex on the target mesh for each vertex
        of the subdivided mesh.
    nnd : ndarray of shape (nv,)
        The estimated geodesic distances between each vertex of the subdivided
        mesh and the nearest vertex on the target mesh.
    """
    nv, nt = coords.shape[0], len(f_indices)
    n_target_faces = len(np.unique(f_indices))
    if n_target_faces != nt:
        warnings.warn(f"{nt} template vertices are mapped to {n_target_faces} faces.")

    # Each target vertex becomes a source node with outgoing connections to
    # the original and subdivided vertices on its face's boundary.
    target_cols, target_distances = [], []
    for f_idx, face_weights in zip(f_indices, weights):
        target_coord = face_weights @ coords[faces[f_idx]]
        a, b, c = sorted(faces[f_idx])
        boundary_indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        target_cols.append(boundary_indices)
        target_distances.append(
            cdist(target_coord[np.newaxis], coords[boundary_indices], "euclidean").ravel()
        )
    target_indptr = np.concatenate([[0], np.cumsum([len(cols) for cols in target_cols])])

    # Widen the mesh graph with empty columns for the target nodes, then append
    # one row per target. Connections point out of targets only, so paths
    # cannot use a target node as a shortcut through a face.
    mesh_graph = _neighbors_to_graph(neighbors, nv)
    mesh_rows = sparse.csr_matrix(
        (mesh_graph.data, mesh_graph.indices, mesh_graph.indptr), shape=(nv, nv + nt)
    )
    target_rows = sparse.csr_matrix(
        (np.concatenate(target_distances), np.concatenate(target_cols), target_indptr),
        shape=(nt, nv + nt),
    )
    graph = sparse.vstack([mesh_rows, target_rows], format="csr")
    del mesh_graph, mesh_rows, target_rows

    dists, _, sources = dijkstra(
        graph,
        indices=np.arange(nv, nv + nt),
        return_predecessors=True,
        min_only=True,
    )
    nn = np.where(sources[:nv] >= 0, sources[:nv] - nv, -1)
    nnd = dists[:nv]
    return nn, nnd


def _native_voronoi(coords, faces, e_mapping, neighbors, verbose=False):
    """Voronoi diagram on the original mesh.

    This function estimates the nearest vertex on the original mesh for each
    vertex of the subdivided mesh. The subdivided mesh is used to increase the
    accuracy of the estimated geodesic distances using Dijkstra's algorithm.

    Parameters
    ----------
    coords : ndarray of shape (nv, 3)
        The coordinates of the vertices of the subdivided mesh.
    faces : ndarray of shape (nf, 3)
        The vertex indices of each triangle face of the original mesh.
    e_mapping : dict
        The mapping from the original edges to the new vertices. The keys are
        tuples of sorted vertex indices of the original edges. The values are
        the indices of the new vertices.
    neighbors : scipy.sparse.csr_matrix or dict
        The graph returned by ``subdivide_edges``. The format of
        ``voronoi_legacy.subdivide_edges`` is also accepted.
    verbose : bool, default=False
        Retained for backward compatibility and ignored. The multi-source
        search does not report per-source progress.

    Returns
    -------
    nn : ndarray of shape (nv,)
        The indices of the nearest vertex on the original mesh for each vertex
        of the subdivided mesh.
    nnd : ndarray of shape (nv,)
        The estimated geodesic distances between each vertex of the subdivided
        mesh and the nearest vertex on the original mesh.
    """
    nv = coords.shape[0]
    graph = _neighbors_to_graph(neighbors, nv)
    nnd, _, nn = dijkstra(
        graph,
        indices=np.unique(faces),
        return_predecessors=True,
        min_only=True,
    )
    nn = np.where(nn >= 0, nn, -1)

    return nn, nnd


def _inverse_face_mapping(f_indices, weights, coords, faces):
    """Inverse face mapping.

    This function computes the mapping from the original faces to the target
    vertices. Given a triangle face on the original mesh, the mapping returns
    the indices of the target vertices inside the triangle face and the
    coordinates of the target vertices.

    Parameters
    ----------
    f_indices : ndarray of shape (nv_target,)
        Each element is the index of the triangle face on the original mesh
        that contains the corresponding vertex on the target mesh.
    weights : ndarray of shape (nv_target, 3)
        Each row is the barycentric weights of the three vertices of the face
        for the corresponding vertex on the target mesh.
    coords : ndarray of shape (nv, 3)
        The coordinates of the vertices of the original mesh.
    faces : ndarray of shape (nf, 3)
        The vertex indices of each triangle face of the original mesh.

    Returns
    -------
    f_inv : dict
        The inverse mapping from the original faces to the target vertices.
        The keys are the indices of the original faces. The values are lists
        of pairs of the indices of the target vertices and the coordinates of
        the target vertices.
    """
    f_inv = {}
    for i, f_idx in enumerate(f_indices):
        if f_idx not in f_inv:
            f_inv[f_idx] = []
        f_inv[f_idx].append([i, weights[i] @ coords[faces[f_idx]]])
    return f_inv


def _split_triangle(t_div):
    """Return barycentric coordinates of the ``t_div**2`` subtriangle centroids."""
    ww1 = []
    for i in range(t_div):
        for j in range(t_div - i):
            k = t_div - i - j - 1
            ww1.append([i, j, k])
    ww1 = (np.array(ww1) + 1.0 / 3) / t_div
    ww2 = []
    for i in range(t_div - 1):
        for j in range(t_div - i - 1):
            k = t_div - i - j - 1
            ww2.append([i, j, k])
    # Reshape so that t_div=1, which has no inverted sub-triangles, gives an
    # empty (0, 3) array.
    ww2 = (np.reshape(ww2, (-1, 3)) + np.array([[2 / 3, 2 / 3, -1 / 3]])) / t_div
    ww = np.concatenate([ww1, ww2])
    return ww


def _compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww):
    """
    Parameters
    ----------
    f_idx : int
        The index of the triangular face on the original mesh.
    f : ndarray of shape (3,)
        The vertex indices of the triangular face.
    coords : ndarray of shape (nv, 3)
        The coordinates of the vertices of the subdivided mesh.
    indices : ndarray of shape (n,)
        The indices of all vertices of the subdivided mesh that are on the
        edges of the triangular face.
    nn : ndarray of shape (nv,)
        The indices of the nearest vertex on the target mesh for each vertex
        of the subdivided mesh.
    nnd : ndarray of shape (nv,)
        The geodesic distances between each vertex of the subdivided mesh and
        the nearest vertex on the target mesh.
    f_inv : dict
        The inverse mapping from the original faces to the target vertices.
    ww : ndarray of shape (m, 3)
        The barycentric weights of the points on the triangular face. Usually
        the output of the ``split_triangle`` function.

    Returns
    -------

    """
    nn1 = [nn[indices]]
    u = np.unique(nn1)

    # The same vertex of the target mesh is the nearest for all vertices on
    # the edges of the triangular face, and no vertex of the target mesh is
    # inside the face.
    if len(u) == 1 and f_idx not in f_inv:
        return {u[0]: np.ones((ww.shape[0],), dtype=bool)}

    cc1 = [coords[indices]]
    dd1 = [nnd[indices]]
    # Vertices of the target mesh that are within the triangular face.
    if f_idx in f_inv:
        for i, c in f_inv[f_idx]:
            cc1.append([c])
            dd1.append([0.0])
            nn1.append([i])
    cc1 = np.concatenate(cc1)
    dd1 = np.concatenate(dd1)
    nn1 = np.concatenate(nn1)
    cc2 = ww @ coords[f]
    d = cdist(cc1, cc2, "euclidean") + dd1[:, np.newaxis]
    idx = np.argmin(d, axis=0)
    occupation = nn1[idx]
    return {u: occupation == u for u in np.unique(occupation)}


def _compute_overlap(
    faces,
    face_areas,
    e_mapping,
    coords,
    nn,
    nnd,
    f_inv,
    nn2,
    nnd2,
    f_inv2,
    nv1,
    nv2,
    t_div=32,
):
    mat = sparse.lil_matrix((nv1, nv2))
    ww = _split_triangle(t_div)
    for f_idx, f in enumerate(faces):
        a, b, c = sorted(faces[f_idx])
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        uu1 = _compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww)
        uu2 = _compute_occupation(f_idx, f, coords, indices, nn2, nnd2, f_inv2, ww)
        for u1, m1 in uu1.items():
            for u2, m2 in uu2.items():
                overlap = np.logical_and(m1, m2).mean()
                if overlap:
                    mat[u1, u2] += overlap * face_areas[f_idx]
    return mat.tocsr()


def _overlap_transform(
    anat,
    sphere,
    tpl_coords,
    coords,
    e_mapping,
    neighbors,
    native_nn,
    native_nnd,
    t_div=32,
):
    f_indices, weights = sphere.barycentric(tpl_coords, eps=1e-7, return_sparse=False)
    nn, nnd = _subdivision_voronoi(
        coords, anat.faces, e_mapping, neighbors, f_indices, weights
    )
    f_inv = _inverse_face_mapping(f_indices, weights, anat.coords, anat.faces)
    mat = _compute_overlap(
        anat.faces,
        anat.face_areas,
        e_mapping,
        coords,
        native_nn,
        native_nnd,
        {},
        nn,
        nnd,
        f_inv,
        anat.nv,
        tpl_coords.shape[0],
        t_div=t_div,
    )
    return mat


class VoronoiOverlap:
    """Overlap areas between the geodesic Voronoi cells of point sets.

    The point sets are the vertices of the mesh (``"native"``) and the point
    sets added with ``add``. ``transform(a, b)[i, j]`` is the area of the
    intersection of cell ``i`` of ``a`` and cell ``j`` of ``b``, measured on
    ``coords``. ``transforms`` computes several of them in one pass over the
    faces, which is faster than one call of ``transform`` for each.

    Parameters
    ----------
    coords : ndarray of shape (nv, 3)
        The surface on which areas and geodesic distances are measured,
        usually the midthickness.
    faces : ndarray of shape (nf, 3)
        The vertex indices of each triangle face.
    sphere : ndarray of shape (nv, 3) or None
        The registered sphere (e.g., sphere.reg) with the same faces, used by
        ``add`` to place point sets given by their coordinates.
    n_div : int
        The number of segments to divide each edge into for the graph.
    t_div : int
        Each face is sampled with ``t_div**2`` points.
    cache_samples : bool
        Whether to keep the sample labels of each point set once computed
        (about 1 byte per sample of each face). Without the cache, every
        ``transforms``, ``transform`` and ``areas`` call labels the samples
        again. In both cases, the point sets that are labelled in the same
        call share the sample distances.
    """

    def __init__(
        self, coords, faces, sphere=None, n_div=8, t_div=32, cache_samples=True
    ):
        coords = _finite_copy(coords, "coords")
        faces = np.array(faces)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (nv, 3), not {coords.shape}.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (nf, 3), not {faces.shape}.")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError(f"faces must be integers, not {faces.dtype}.")
        if faces.size and (faces.min() < 0 or faces.max() >= coords.shape[0]):
            raise ValueError("faces contain vertex indices out of range.")
        # Unsigned faces would turn node indices into floats when combined
        # with signed indices.
        faces = faces.astype(np.intp, copy=False)
        if sphere is not None:
            sphere = _finite_copy(sphere, "sphere")
            if sphere.shape != coords.shape:
                raise ValueError(
                    f"sphere must have the shape of coords {coords.shape}, "
                    f"not {sphere.shape}."
                )
        for name, value in [("n_div", n_div), ("t_div", t_div)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer, not {value!r}.")

        self.coords = coords
        self.faces = faces
        self.sphere = sphere
        self.n_div = int(n_div)
        self.t_div = int(t_div)
        self.cache_samples = bool(cache_samples)
        self.face_areas = compute_face_areas(coords, faces)
        self._ww = _split_triangle(self.t_div)  # Barycentric sample coordinates.

        _, face_edge_ids, edge_points, new_coords, graph = _subdivided_graph(
            coords, faces, self.n_div
        )
        self._node_coords = np.concatenate([coords, new_coords])
        self._graph = graph
        self._face_nodes = _face_nodes(faces, face_edge_ids, edge_points)
        self._native = None
        self._point_sets = {}
        self._placement = None
        self._sample_labels_cache = {}

    @property
    def names(self):
        """The names of the point sets, starting with ``"native"``."""
        return ("native", *self._point_sets)

    def add(self, name, coords=None, *, f_indices=None, weights=None):
        """Add a point set and label the graph nodes with its points.

        Give either the coordinates of the points on the unit sphere (placed
        on ``sphere`` with ``Sphere.barycentric``), or the face that contains
        each point and its barycentric weights.

        Parameters
        ----------
        name : str
            The name of the point set. It must be new and not ``"native"``.
        coords : ndarray of shape (n, 3) or None
            The coordinates of the points on the unit sphere.
        f_indices : ndarray of shape (n,) or None
            The face that contains each point.
        weights : ndarray of shape (n, 3) or None
            The barycentric weights of each point in its face.
        """
        if name in self.names:
            raise ValueError(f"A point set named {name!r} already exists.")
        if coords is not None:
            if f_indices is not None or weights is not None:
                raise ValueError(
                    "Give either coords or f_indices and weights, not both."
                )
            if self.sphere is None:
                raise ValueError("Placing points by coordinates requires sphere.")
            coords = _finite_copy(coords, "coords")
            if coords.ndim != 2 or coords.shape[1] != 3:
                raise ValueError(f"coords must have shape (n, 3), not {coords.shape}.")
            if self._placement is None:
                from . import Sphere

                self._placement = Sphere(self.sphere, self.faces)
            f_indices, weights = self._placement.barycentric(
                coords, eps=1e-7, return_sparse=False
            )
        elif f_indices is None or weights is None:
            raise ValueError("Give either coords or both f_indices and weights.")

        f_indices = np.array(f_indices)
        weights = _finite_copy(weights, "weights")
        if f_indices.ndim != 1 or not np.issubdtype(f_indices.dtype, np.integer):
            raise ValueError("f_indices must be a 1-D array of integers.")
        if weights.shape != (len(f_indices), 3):
            raise ValueError(
                f"weights must have shape ({len(f_indices)}, 3), not {weights.shape}."
            )
        if f_indices.size and (
            f_indices.min() < 0 or f_indices.max() >= len(self.faces)
        ):
            raise ValueError("f_indices contain face indices out of range.")
        f_indices = f_indices.astype(np.intp, copy=False)

        n_faces = len(np.unique(f_indices))
        if n_faces != len(f_indices):
            warnings.warn(
                f"{len(f_indices)} template vertices are mapped to {n_faces} faces."
            )
        self._point_sets[name] = self._label_points(f_indices, weights)

    def discard(self, name):
        """Remove a point set and its labels.

        For ``"native"``, only the labels are removed. They are computed
        again when needed.
        """
        if name == "native":
            self._native = None
        else:
            self._get(name)
            del self._point_sets[name]
        self._sample_labels_cache.pop(name, None)

    def clear_cache(self, name=None):
        """Release the cached sample labels of a point set, or of all point
        sets if ``name`` is None.

        Node labels are kept. Sample labels are computed again when needed.
        """
        if name is None:
            self._sample_labels_cache.clear()
        elif name in self.names:
            self._sample_labels_cache.pop(name, None)
        else:
            raise KeyError(f"No point set named {name!r}.")

    def transforms(self, pairs):
        """Overlap areas for several pairs of point sets.

        All pairs are computed in one pass over the faces, and the point sets
        share the work that does not depend on them. Each matrix is the one
        that ``transform`` returns for that pair.

        Parameters
        ----------
        pairs : sequence of (str, str)
            The names of the source and target point sets of each matrix. A
            pair can be given more than once, and so can its reverse.

        Returns
        -------
        mats : list of scipy.sparse.csr_matrix
            One new matrix for each pair, in the order of ``pairs``. The
            matrices do not share memory.
        """
        pairs = list(pairs)
        for pair in pairs:
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(name, str) for name in pair)
            ):
                raise TypeError(
                    f"Each item of pairs must be a (source, target) pair of "
                    f"names, not {pair!r}."
                )
        pairs = [tuple(pair) for pair in pairs]
        for name in dict.fromkeys(name for pair in pairs for name in pair):
            if name not in self.names:
                raise KeyError(f"No point set named {name!r}.")

        # A pair that is given several times is computed once.
        requests = list(dict.fromkeys(pairs))
        entries = dict(zip(requests, self._entries(requests)))
        mats = []
        for source, target in pairs:
            n_source, n_target = self._get(source).size, self._get(target).size
            _, keys, values = entries[source, target]
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            totals = np.zeros(len(unique_keys))
            # Accumulate individual face contributions in face order.
            np.add.at(totals, inverse, values)

            # Sorted keys encode row * n_target + column. Each pair is unique,
            # so CSR construction does not need to sort or sum duplicates.
            rows, cols = np.divmod(unique_keys, n_target)
            row_counts = np.bincount(rows, minlength=n_source)
            indptr = np.concatenate([[0], np.cumsum(row_counts)])
            mat = sparse.csr_matrix((totals, cols, indptr), shape=(n_source, n_target))
            mat.eliminate_zeros()
            mats.append(mat)
        return mats

    def transform(self, source, target):
        """Overlap areas between the Voronoi cells of two point sets.

        Parameters
        ----------
        source, target : str
            The names of the point sets.

        Returns
        -------
        mat : scipy.sparse.csr_matrix of shape (n_source, n_target)
            ``mat[i, j]`` is the area of the intersection of cell ``i`` of
            ``source`` and cell ``j`` of ``target``. The matrix is computed
            on every call. ``transform(b, a)`` equals ``transform(a, b).T``
            exactly, and ``transform(a, a)`` is diagonal, with ``areas(a)``
            on its diagonal.
        """
        return self.transforms([(source, target)])[0]

    def areas(self, name):
        """The area of the Voronoi cell of each point of a point set.

        Return a new array of shape (n_points,). For ``"native"``, this
        equals ``Surface.vertex_areas_nn`` exactly. Transform row sums can
        differ slightly because they group the additions differently.
        """
        ((_, keys, values),) = self._entries([(name,)])
        areas = np.zeros(self._get(name).size)
        np.add.at(areas, keys, values)
        return areas

    def _entries(self, requests, chunk_size=64):
        """Return the area contributions of every face for each request.

        A request is a tuple with the names of one or two point sets. Its
        result is three parallel arrays: face indices, keys and areas. Each
        face contributes one entry per label of the point set (or per pair of
        labels of the two point sets) that its samples have, with the key
        ``label`` (or ``label * n_second + second_label``). The area is
        ``count / n_samples * face_area``, as in ``compute_overlap``. Entries
        are sorted by face, so that ``np.add.at`` adds the areas of each key
        in face order, as the in-place additions of ``compute_overlap`` do.

        All requests are served by one pass over the faces. The point sets
        that are labelled in this pass use the same sample distances.
        """
        names = list(dict.fromkeys(name for request in requests for name in request))
        point_sets = {name: self._get(name) for name in names}
        sample_labels = {name: self._get_sample_labels(name) for name in names}
        # The point sets without cached labels are labelled in this pass.
        new = [name for name in names if name not in self._sample_labels_cache]
        is_mixed = {name: sample_labels[name].single < 0 for name in names}
        n_samples = len(self._ww)

        def encode_labels(request, labels):
            # Labels are int32. Convert them to int64 before combining them,
            # because int32 times a Python int stays int32 and could wrap.
            keys = labels[0].astype(np.int64)
            if len(request) == 2:
                keys = keys * point_sets[request[1]].size + labels[1]
            return keys

        # Faces where every set of a request has one label give one entry.
        parts = []
        for request in requests:
            is_single = ~np.logical_or.reduce([is_mixed[name] for name in request])
            single_faces = np.flatnonzero(is_single)
            keys = encode_labels(
                request, [sample_labels[name].single[single_faces] for name in request]
            )
            counts = np.full(len(single_faces), n_samples)
            parts.append(([single_faces], [keys], [counts]))

        any_mixed = np.zeros(len(self.faces), dtype=bool)
        for name in names:
            any_mixed |= is_mixed[name]
        visited_faces = np.flatnonzero(any_mixed)
        for start in range(0, len(visited_faces), chunk_size):
            face_indices = visited_faces[start : start + chunk_size]
            # Distances are needed for the faces that are labelled now.
            labelled = np.zeros(len(face_indices), dtype=bool)
            for name in new:
                labelled |= is_mixed[name][face_indices]
            boundary = None
            if labelled.any():
                boundary = (
                    np.cumsum(labelled) - 1,
                    *self._boundary_distances(face_indices[labelled]),
                )
            labels = {
                name: self._face_labels(
                    point_sets[name],
                    sample_labels[name],
                    face_indices,
                    boundary if name in new else None,
                )
                for name in names
            }
            for request, (face_parts, key_parts, count_parts) in zip(requests, parts):
                # The faces where both sets have one label are counted above.
                rows = np.zeros(len(face_indices), dtype=bool)
                for name in request:
                    rows |= is_mixed[name][face_indices]
                if not rows.any():
                    continue
                sorted_keys = np.sort(
                    encode_labels(request, [labels[name][rows] for name in request]),
                    axis=1,
                )
                # Each row is one face. Mark the start of every run of equal
                # keys, including the first sample of each face, even if
                # adjacent faces share a key. Differences between flattened
                # starts give counts.
                is_run_start = np.ones(sorted_keys.shape, dtype=bool)
                is_run_start[:, 1:] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
                run_starts = np.flatnonzero(is_run_start)
                face_parts.append(face_indices[rows][run_starts // n_samples])
                key_parts.append(sorted_keys.ravel()[run_starts])
                count_parts.append(np.diff(run_starts, append=sorted_keys.size))

        if self.cache_samples:
            for name in new:
                self._sample_labels_cache[name] = sample_labels[name]

        # Merge single and mixed faces back into face order. Keep every face's
        # contribution separate so chunk boundaries cannot change rounding.
        entries = []
        for face_parts, key_parts, count_parts in parts:
            faces = np.concatenate(face_parts)
            order = np.argsort(faces, kind="stable")
            faces = faces[order]
            keys = np.concatenate(key_parts)[order]
            counts = np.concatenate(count_parts)[order]
            entries.append((faces, keys, counts / n_samples * self.face_areas[faces]))
        return entries

    def _face_labels(self, point_set, sample_labels, face_indices, boundary=None):
        """Return the labels of the samples of the faces.

        The result has shape (len(face_indices), n_samples). Without
        ``boundary``, the labels of the mixed faces are read from
        ``sample_labels.local``. With ``boundary``, they are computed here,
        and stored in ``sample_labels.local`` unless it is None. ``boundary``
        has the row of each face in the arrays that follow (only used for
        the mixed faces), and the output of ``_boundary_distances`` for
        those rows.
        """
        decoded = np.repeat(
            sample_labels.single[face_indices, np.newaxis], len(self._ww), axis=1
        )
        is_mixed = sample_labels.single[face_indices] < 0
        if not is_mixed.any():
            return decoded
        mixed_indices = face_indices[is_mixed]
        if boundary is None or sample_labels.local is not None:
            cache_rows = np.searchsorted(sample_labels.mixed_faces, mixed_indices)
        if boundary is None:
            local = sample_labels.local[cache_rows]
        else:
            rows, samples, distances = boundary
            rows = rows[is_mixed]
            # Rows are increasing, so they are all rows if as many.
            if len(rows) < len(samples):
                samples, distances = samples[rows], distances[rows]
            local = self._local_labels(point_set, mixed_indices, samples, distances)
            if sample_labels.local is not None:
                sample_labels.local[cache_rows] = local
        decoded[is_mixed] = self._decode(point_set, mixed_indices, local)
        return decoded

    def _get_sample_labels(self, name):
        """Return the cached sample labels of a point set, or new ones that
        the pass of ``_entries`` fills in."""
        if name in self._sample_labels_cache:
            return self._sample_labels_cache[name]
        return self._sample_labels(self._get(name))

    def _get(self, name):
        """Return the labels of a point set, labelling ``"native"`` if needed."""
        if name == "native":
            if self._native is None:
                self._native = self._label_vertices()
            return self._native
        if name not in self._point_sets:
            raise KeyError(f"No point set named {name!r}.")
        return self._point_sets[name]

    def _label_vertices(self):
        """Label the graph nodes with the nearest mesh vertex."""
        nnd, _, nn = dijkstra(
            self._graph,
            indices=np.unique(self.faces),
            return_predecessors=True,
            min_only=True,
        )
        nn = np.where(nn >= 0, nn, -1).astype(np.int32)
        return _PointSet(
            size=len(self.coords),
            nn=nn,
            nnd=nnd,
            inside_ptr=np.zeros(len(self.faces) + 1, dtype=np.intp),
            inside_labels=np.zeros(0, dtype=np.int32),
            inside_coords=np.zeros((0, 3)),
        )

    def _label_points(self, f_indices, weights):
        """Label the graph nodes with the nearest point of a point set.

        Each point becomes an extra node with one-way connections to the
        boundary nodes of its face, as in ``subdivision_voronoi``.
        """
        n_nodes, n = len(self._node_coords), len(f_indices)
        # The batched matmul rounds like ``weights[i] @ coords[faces[f]]``.
        corners = self.coords[self.faces[f_indices]]
        positions = np.matmul(weights[:, np.newaxis], corners)[:, 0]
        boundary = self._face_nodes[f_indices]
        # The sum of squares rounds like ``cdist``.
        diff = positions[:, np.newaxis] - self._node_coords[boundary]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        del diff

        # Connections point out of the point nodes only, so paths cannot use
        # a point node as a shortcut through a face.
        graph = self._graph
        mesh_rows = sparse.csr_matrix(
            (graph.data, graph.indices, graph.indptr), shape=(n_nodes, n_nodes + n)
        )
        point_rows = sparse.csr_matrix(
            (
                distances.ravel(),
                boundary.ravel(),
                np.arange(0, boundary.size + 1, boundary.shape[1]),
            ),
            shape=(n, n_nodes + n),
        )
        graph = sparse.vstack([mesh_rows, point_rows], format="csr")
        del mesh_rows, point_rows
        dists, _, sources = dijkstra(
            graph,
            indices=np.arange(n_nodes, n_nodes + n),
            return_predecessors=True,
            min_only=True,
        )
        del graph
        nn = np.where(sources[:n_nodes] >= 0, sources[:n_nodes] - n_nodes, -1)

        # The points inside each face, in increasing index.
        order = np.argsort(f_indices, kind="stable")
        counts = np.bincount(f_indices, minlength=len(self.faces))
        return _PointSet(
            size=n,
            nn=nn.astype(np.int32),
            nnd=dists[:n_nodes],
            inside_ptr=np.concatenate([[0], np.cumsum(counts)]),
            inside_labels=order.astype(np.int32),
            inside_coords=positions[order],
        )

    def _sample_labels(self, point_set):
        """Find the faces whose samples all have the same label.

        A face whose boundary nodes all have the same label, and that has no
        point inside, takes that label (``single``). For the other faces
        (``mixed_faces``), the pass of ``_entries`` finds each sample's
        nearest candidate. Candidates are the face's boundary nodes in
        ``_face_nodes`` order, followed by its inside points in index order.
        With ``cache_samples``, ``local`` is an empty array for the indices
        of those candidates; otherwise it is None.
        """
        n_boundary = self._face_nodes.shape[1]
        boundary_labels = point_set.nn[self._face_nodes]
        if np.any(boundary_labels < 0):
            raise ValueError(
                "Some faces cannot be reached from any point. Each connected "
                "part of the mesh needs at least one point."
            )
        inside_counts = np.diff(point_set.inside_ptr)
        is_single = np.all(boundary_labels == boundary_labels[:, :1], axis=1)
        is_single &= inside_counts == 0
        single = np.where(is_single, boundary_labels[:, 0], -1).astype(np.int32)
        mixed_faces = np.flatnonzero(~is_single)
        local = None
        if self.cache_samples:
            dtype = _local_dtype(n_boundary + inside_counts.max(initial=0))
            local = np.empty((len(mixed_faces), len(self._ww)), dtype=dtype)
        return _SampleLabels(single=single, mixed_faces=mixed_faces, local=local)

    def _boundary_distances(self, face_indices):
        """Return sample coordinates and distances to each face's boundary nodes.

        For k face indices, the shapes are (k, n_samples, 3) and
        (k, n_samples, n_boundary). Neither array depends on the point set.
        """
        # The batched matmul rounds like ``ww @ coords[f]``.
        samples = np.matmul(self._ww, self.coords[self.faces[face_indices]])
        boundary = self._node_coords[self._face_nodes[face_indices]]
        distances = np.empty((len(face_indices), len(self._ww), boundary.shape[1]))
        for i in range(len(face_indices)):
            distances[i] = cdist(samples[i], boundary[i])
        return samples, distances

    def _local_labels(self, point_set, face_indices, samples, distances):
        """Return the index of each sample's nearest candidate in its face.

        A boundary node's score is its graph distance to the nearest point
        plus its Euclidean distance to the sample. An inside point's score
        is its Euclidean distance to the sample. Ties go to the first
        candidate, as in ``compute_occupation``.
        """
        boundary_nnd = point_set.nnd[self._face_nodes[face_indices]]
        scores = distances + boundary_nnd[:, np.newaxis]
        local = np.argmin(scores, axis=2)
        best_scores = np.take_along_axis(scores, local[..., np.newaxis], axis=2)[..., 0]
        del scores

        # The inside points win only if strictly closer, so ties still go to
        # the boundary nodes, which come first.
        n_boundary = self._face_nodes.shape[1]
        ptr = point_set.inside_ptr
        sample_indices = np.arange(samples.shape[1])
        has_inside = ptr[face_indices + 1] > ptr[face_indices]
        for i in np.flatnonzero(has_inside):
            face = face_indices[i]
            inside = point_set.inside_coords[ptr[face] : ptr[face + 1]]
            inside_distances = cdist(samples[i], inside)
            nearest = np.argmin(inside_distances, axis=1)
            closer = inside_distances[sample_indices, nearest] < best_scores[i]
            local[i, closer] = n_boundary + nearest[closer]
        return local

    def _decode(self, point_set, face_indices, local):
        """Convert local candidate indices to point labels for the given faces.

        ``local`` and the result have shape (len(face_indices), n_samples).
        """
        n_boundary = self._face_nodes.shape[1]
        # Widen compact unsigned indices before computing array offsets.
        local = local.astype(np.intp)
        sample_faces = np.broadcast_to(face_indices[:, np.newaxis], local.shape)
        decoded = np.empty(local.shape, dtype=np.int32)
        on_boundary = local < n_boundary
        decoded[on_boundary] = point_set.nn[
            self._face_nodes[sample_faces[on_boundary], local[on_boundary]]
        ]
        inside = ~on_boundary
        decoded[inside] = point_set.inside_labels[
            point_set.inside_ptr[sample_faces[inside]] + local[inside] - n_boundary
        ]
        return decoded


@dataclass
class _SampleLabels:
    """The labels of the samples of each face for one point set.

    For faces outside ``mixed_faces``, ``single[f]`` is the label of every
    sample. For mixed faces, ``single`` is -1 and unused; ``local[i]`` holds
    the candidate index of each sample of ``mixed_faces[i]``, or ``local``
    is None if the indices are not kept.
    """

    single: np.ndarray
    mixed_faces: np.ndarray
    local: np.ndarray


@dataclass
class _PointSet:
    """The labels of one point set.

    ``nn`` and ``nnd`` are the nearest point and its distance for each graph
    node. The points inside face ``f`` are
    ``inside_labels[inside_ptr[f]:inside_ptr[f + 1]]``, at ``inside_coords``.
    """

    size: int
    nn: np.ndarray
    nnd: np.ndarray
    inside_ptr: np.ndarray
    inside_labels: np.ndarray
    inside_coords: np.ndarray


def _local_dtype(n_candidates):
    """Choose uint8 or uint16 for indices from 0 to n_candidates - 1."""
    if n_candidates <= 256:
        return np.uint8
    if n_candidates <= 65536:
        return np.uint16
    raise ValueError(
        f"{n_candidates} candidates in one face exceed the limit of 65536."
    )


def _finite_copy(x, name):
    """Copy an array as float64, checking that every value is finite."""
    x = np.array(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")
    return x


def _face_nodes(faces, face_edge_ids, edge_points):
    """Return the boundary nodes of each face in candidate order.

    For a face with sorted vertices a < b < c, the order is the points on the
    edges (a, b), (a, c) and (b, c), then a, b and c, as in
    ``compute_overlap``. The result has shape (nf, 3 * n_div).
    """
    # Position of a, b and c in each face.
    order = np.argsort(faces, axis=1, kind="stable")
    rows = np.arange(len(faces))[:, np.newaxis]
    # face_edge_ids follows the local pairs (0, 1), (1, 2), (2, 0), so the
    # edge that leaves out position k is local edge (k + 1) % 3. Edges (a, b),
    # (a, c) and (b, c) leave out the positions of c, b and a.
    local_edges = (order[:, ::-1] + 1) % 3
    sorted_edge_ids = face_edge_ids[rows, local_edges]
    edge_nodes = edge_points[sorted_edge_ids].reshape(
        len(faces), 3 * edge_points.shape[1]
    )
    return np.concatenate([edge_nodes, faces[rows, order]], axis=1)


"""
Similar to https://github.com/mojocorp/geodesic subdivision
"""


def _deprecated(func, name, alternative="VoronoiOverlap"):
    """Wrap ``func`` so that calling it by its former public ``name`` warns."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"neuroboros.surface.voronoi.{name} is deprecated and will be "
            f"removed in a future version; use {alternative} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    # Bound in this module, so that pickle finds it here.
    wrapper.__module__ = __name__
    wrapper.__name__ = wrapper.__qualname__ = name
    return wrapper


# The former public functions, unchanged in behaviour. Calls within this
# module use the private functions, so each public call warns only once.
subdivide_edges = _deprecated(_subdivide_edges, "subdivide_edges")
neighbors_to_coo = _deprecated(_neighbors_to_coo, "neighbors_to_coo")
neighbors_to_graph = _deprecated(_neighbors_to_graph, "neighbors_to_graph")
subdivision_voronoi = _deprecated(_subdivision_voronoi, "subdivision_voronoi")
native_voronoi = _deprecated(_native_voronoi, "native_voronoi")
inverse_face_mapping = _deprecated(_inverse_face_mapping, "inverse_face_mapping")
split_triangle = _deprecated(_split_triangle, "split_triangle")
compute_occupation = _deprecated(_compute_occupation, "compute_occupation")
compute_overlap = _deprecated(_compute_overlap, "compute_overlap")
overlap_transform = _deprecated(_overlap_transform, "overlap_transform")
# Formerly re-exported here; the function stays in voronoi_legacy.
dijkstra_distances = _deprecated(
    _legacy_dijkstra_distances,
    "dijkstra_distances",
    "neuroboros.surface.voronoi_legacy.dijkstra_distances",
)
