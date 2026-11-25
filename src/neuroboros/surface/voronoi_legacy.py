"""
Previous implementations of the Voronoi diagrams in ``voronoi.py``, which
run Dijkstra's algorithm in pure Python once per source vertex. They give
the same results up to ties, but are much slower.
"""

import heapq
import warnings
from datetime import datetime

import numpy as np
from scipy.spatial.distance import cdist

from .voronoi import dijkstra_distances


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
