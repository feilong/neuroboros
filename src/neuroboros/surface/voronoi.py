import heapq
import warnings

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist


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
    neighbors : dict
        The neighbors of each vertex. The keys are the vertex indices. The
        values are dictionaries. The keys of the dictionaries are the indices
        of the neighboring vertices. The values of the dictionaries are the
        distances between the vertex and the neighboring vertices.
    """
    n_edges = faces.shape[0] * 3 // 2
    nv_new = n_edges * (n_div - 1)
    new_coords = np.zeros((nv_new, 3), dtype=coords.dtype)
    neighbors = {}

    nv = coords.shape[0]
    count = 0
    e_mapping = {}
    arng = np.arange(1, n_div)
    for f in faces:
        for i in f:
            if i not in neighbors:
                neighbors[i] = {}

        steiner = []
        for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 0, 1]]:
            a, b, c = f[[i, j, k]]
            e = (a, b) if a < b else (b, a)
            if e not in e_mapping:
                cc = (
                    coords[[e[1]]] * arng[:, np.newaxis]
                    + coords[e[0]] * (n_div - arng[:, np.newaxis])
                ) / n_div
                new_coords[count : count + n_div - 1] = cc
                indices = (count + nv - 1) + arng
                e_mapping[e] = indices
                count += n_div - 1

                for idx in indices:
                    neighbors[idx] = {}
                d = np.linalg.norm(coords[e[1]] - coords[e[0]]) / n_div
                chain = [e[0], *indices, e[1]]
                for idx1, idx2 in zip(chain[:-1], chain[1:]):
                    neighbors[idx1][idx2] = d
                    neighbors[idx2][idx1] = d
            else:
                indices = e_mapping[e]
                cc = new_coords[indices - nv]
            steiner.append([cc, indices])

            # connecting points on an edge and the opposite point
            idx2, c2 = c, coords[c]
            for idx1, c1 in zip(indices, cc):
                if idx1 not in neighbors[idx2]:
                    d = np.linalg.norm(c1 - c2)
                    neighbors[idx2][idx1] = d
                    neighbors[idx1][idx2] = d

        # connecting points on an edge and points on another edge
        for i, j in [[0, 1], [1, 2], [2, 0]]:
            cc1, indices1 = steiner[i]
            cc2, indices2 = steiner[j]
            for idx1, c1 in zip(indices1, cc1):
                for idx2, c2 in zip(indices2, cc2):
                    if idx1 not in neighbors[idx2]:
                        d = np.linalg.norm(c1 - c2)
                        neighbors[idx2][idx1] = d
                        neighbors[idx1][idx2] = d

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
    counts = np.fromiter((len(_) for _ in neighbors.values()), int, len(neighbors))
    n = counts.sum()
    rows = np.repeat(np.fromiter(neighbors.keys(), int, len(neighbors)), counts)
    cols = np.fromiter((j for _ in neighbors.values() for j in _), int, n)
    dists = np.fromiter((d for _ in neighbors.values() for d in _.values()), float, n)
    return rows, cols, dists


def dijkstra_distances(nv, candidates, neighbors, max_dist=None):
    """Dijkstra's algorithm to estimate the geodesic distances.

    This function estimates the geodesic distances between a vertex and all
    vertices in the mesh using Dijkstra's algorithm. If the vertex is part of
    the mesh, the ``candidates`` list should contain only one tuple
    ``(0.0, idx)`` where ``idx`` is the index of the vertex. If the vertex is
    not part of the mesh, the ``candidates`` list should contain tuples
    ``(d, idx)`` where ``d`` is the distance between the vertex and the
    neighboring vertices and ``idx`` is the index of the neighboring vertices.

    Parameters
    ----------
    nv : int
        The number of vertices.
    candidates : list
        The list of candidate vertices. Each element is a tuple of the
        distance and the vertex index.
    neighbors : dict
        The neighbors of each vertex. The keys are the vertex indices. The
        values are dictionaries. The keys of the dictionaries are the indices
        of the neighboring vertices. The values of the dictionaries are the
        distances between the vertex and the neighboring vertices.
    max_dist : float or None, default=None
        The maximum distance to search. If None, the maximum distance is
        infinity.

    Returns
    -------
    dists : ndarray of shape (nv,)
        The estimated geodesic distances.
    """
    dists = np.full((nv,), np.inf)
    finished = np.zeros((nv,), dtype=bool)
    for d, idx in candidates:
        dists[idx] = d

    while candidates:
        d, idx = heapq.heappop(candidates)
        if finished[idx]:
            continue

        for nbr, nbr_d in neighbors[idx].items():
            new_d = d + nbr_d
            if max_dist is not None and new_d > max_dist:
                continue
            if new_d < dists[nbr]:
                dists[nbr] = new_d
                heapq.heappush(candidates, (new_d, nbr))
        finished[idx] = True

    return dists


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
    neighbors : dict
        The neighbors of each vertex. The keys are the vertex indices. The
        values are dictionaries. The keys of the dictionaries are the indices
        of the neighboring vertices. The values of the dictionaries are the
        distances between the vertex and the neighboring vertices.
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
    n1 = len(np.unique(f_indices))
    if n1 != nt:
        warnings.warn(f"{nt} template vertices are mapped to {n1} faces.")

    # Each vertex of the target mesh is added to the graph as a new node,
    # which is connected to the vertices on the edges of its face.
    rows, cols, dists = neighbors_to_coo(neighbors)
    rows, cols, dists = [rows], [cols], [dists]
    for i, (f_idx, w) in enumerate(zip(f_indices, weights)):
        cc = w @ coords[faces[f_idx]]
        a, b, c = sorted(faces[f_idx])
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        rows.append(np.full(indices.shape, nv + i))
        cols.append(indices)
        dists.append(cdist(cc[np.newaxis], coords[indices], "euclidean").ravel())
    graph = sparse.csr_matrix(
        (np.concatenate(dists), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nv + nt, nv + nt),
    )

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
    neighbors : dict
        The neighbors of each vertex. The keys are the vertex indices. The
        values are dictionaries. The keys of the dictionaries are the indices
        of the neighboring vertices. The values of the dictionaries are the
        distances between the vertex and the neighboring vertices.
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
    rows, cols, dists = neighbors_to_coo(neighbors)
    graph = sparse.csr_matrix((dists, (rows, cols)), shape=(nv, nv))
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


"""
Similar to https://github.com/mojocorp/geodesic subdivision
"""
