"""
Previous implementations of functions in ``voronoi.py``, written with Python
loops and dicts; the Voronoi diagrams run Dijkstra's algorithm once per
source vertex. They give the same results up to ties and rounding, but are
much slower.
"""

import heapq
import warnings
from datetime import datetime

import numpy as np
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
        The maximum distance of the initial search. If None, the maximum
        distance is computed based on the number of vertices on the target
        mesh.
    verbose : bool, default=False
        Whether to print the progress or not.

    Returns
    -------
    nn : ndarray of shape (nv,)
        The indices of the nearest vertex on the target mesh for each vertex
        of the subdivided mesh.
    nnd : ndarray of shape (nv,)
        The estimated geodesic distances between each vertex of the subdivided
        mesh and the nearest vertex on the target mesh.
    """
    nv = coords.shape[0]
    n1, n2 = len(np.unique(f_indices)), len(f_indices)
    if n1 != n2:
        warnings.warn(f"{n2} template vertices are mapped to {n1} faces.")
    if max_dist is None:
        max_dist = 4.0 * np.sqrt(10242 / f_indices.shape[0]) + 2.0
    log_step = f_indices.shape[0] // 100 + 1
    nn = np.full((nv,), -1, dtype=int)
    nnd = np.full((nv,), np.inf)
    while np.any(np.isinf(nnd)):
        for i, (f_idx, w) in enumerate(zip(f_indices, weights)):
            cc = w @ coords[faces[f_idx]]
            a, b, c = sorted(faces[f_idx])
            indices = np.concatenate(
                [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
            )
            dd = cdist(cc[np.newaxis], coords[indices], "euclidean").ravel()
            candidates = []
            for d, idx in zip(dd, indices):
                heapq.heappush(candidates, (d, idx))
            d = dijkstra_distances(nv, candidates, neighbors, max_dist=max_dist)
            mask = d < nnd
            nn[mask] = i
            nnd[mask] = d[mask]
            if verbose and i % log_step == 0:
                print(
                    datetime.now(),
                    i,
                    mask.sum(),
                    np.isfinite(d).sum(),
                    d.shape,
                    d.max(),
                    d.min(),
                    len(candidates),
                    np.isinf(nnd).sum(),
                )
        max_dist *= 1.5
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
        Whether to print the progress or not.

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
    nn = np.full((nv,), -1, dtype=int)
    nnd = np.full((nv,), np.inf)
    max_dist = np.max(
        [
            np.linalg.norm(coords[faces[:, 0]] - coords[faces[:, 1]], axis=1).max(),
            np.linalg.norm(coords[faces[:, 1]] - coords[faces[:, 2]], axis=1).max(),
            np.linalg.norm(coords[faces[:, 2]] - coords[faces[:, 0]], axis=1).max(),
        ]
    )
    max_dist = max_dist * 0.5 + 1e-3

    seeds = []

    for f in faces:
        f = np.sort(f)
        cc1 = coords[f]
        a, b, c = f
        for i, e in enumerate([(b, c), (a, c), (a, b)]):
            indices = e_mapping[e]
            cc2 = coords[indices]
            d = cdist(cc1, cc2)

            min_idx = np.argmin(d, axis=0)
            # wide short triangle
            if np.any(min_idx == i):
                seeds.append(f[i])
            d = d.min(axis=0)
            mask = d < nnd[indices]

            nnd[indices[mask]] = d[mask]
            nn[indices[mask]] = f[min_idx[mask]]
        nn[f] = f
        nnd[f] = 0.0
    if verbose:
        print(np.isfinite(nnd).mean(), nnd.max())

    seeds = np.unique(seeds)
    if verbose:
        print(len(seeds), seeds[:10])
    for i, seed in enumerate(seeds):
        candidates = [(0.0, seed)]
        d = dijkstra_distances(nv, candidates, neighbors, max_dist=max_dist)
        mask = d < nnd
        nn[mask] = seed
        nnd[mask] = d[mask]
        if verbose and i % 10000 == 0:
            print(
                datetime.now(),
                i,
                seed,
                mask.sum(),
                nnd.max(),
                np.isfinite(d).sum(),
                d.shape,
                d.max(),
                d.min(),
                np.isinf(nnd).sum(),
            )

    return nn, nnd
