import warnings

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from .properties import compute_face_areas

# Preserve the original import path for callers of the legacy helper.
from .voronoi_legacy import dijkstra_distances as dijkstra_distances


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


def subdivide_edges(coords, faces, n_div):
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


def neighbors_to_coo(neighbors):
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


def neighbors_to_graph(neighbors, nv):
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
    rows, cols, dists = neighbors_to_coo(neighbors)
    return sparse.csr_matrix((dists, (rows, cols)), shape=(nv, nv))


def subdivision_voronoi(
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
    mesh_graph = neighbors_to_graph(neighbors, nv)
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


def native_voronoi(coords, faces, e_mapping, neighbors, verbose=False):
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
    graph = neighbors_to_graph(neighbors, nv)
    nnd, _, nn = dijkstra(
        graph,
        indices=np.unique(faces),
        return_predecessors=True,
        min_only=True,
    )
    nn = np.where(nn >= 0, nn, -1)

    return nn, nnd


def inverse_face_mapping(f_indices, weights, coords, faces):
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


def split_triangle(t_div):
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
    ww2 = (np.array(ww2) + np.array([[2 / 3, 2 / 3, -1 / 3]])) / t_div
    ww = np.concatenate([ww1, ww2])
    return ww


def compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww):
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


def compute_overlap(
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
    ww = split_triangle(t_div)
    for f_idx, f in enumerate(faces):
        a, b, c = sorted(faces[f_idx])
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        uu1 = compute_occupation(f_idx, f, coords, indices, nn, nnd, f_inv, ww)
        uu2 = compute_occupation(f_idx, f, coords, indices, nn2, nnd2, f_inv2, ww)
        for u1, m1 in uu1.items():
            for u2, m2 in uu2.items():
                overlap = np.logical_and(m1, m2).mean()
                if overlap:
                    mat[u1, u2] += overlap * face_areas[f_idx]
    return mat.tocsr()


def overlap_transform(
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
    nn, nnd = subdivision_voronoi(
        coords, anat.faces, e_mapping, neighbors, f_indices, weights
    )
    f_inv = inverse_face_mapping(f_indices, weights, anat.coords, anat.faces)
    mat = compute_overlap(
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
    ``coords``.

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
        Whether to keep the sample labels of each point set.
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
        self.cache_samples = cache_samples
        self.face_areas = compute_face_areas(coords, faces)

        _, face_edge_ids, edge_points, new_coords, graph = _subdivided_graph(
            coords, faces, self.n_div
        )
        self._node_coords = np.concatenate([coords, new_coords])
        self._graph = graph
        self._face_nodes = _face_nodes(faces, face_edge_ids, edge_points)
        self._point_sets = {}

    @property
    def names(self):
        """The names of the point sets, starting with ``"native"``."""
        return ("native", *self._point_sets)


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
