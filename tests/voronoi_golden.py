"""Golden reference for the Voronoi overlap transforms.

Running this file as a script computes every case with the frozen functions
of ``voronoi_reference.py`` and writes ``data/voronoi_golden.json``: the
shape, sum and digest of each output, the environment, and the digests of
the inputs.
Tied shortest paths may be resolved differently with other versions of
Python or SciPy, so the digests are only valid in the recorded environment
and with the recorded inputs.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from functools import lru_cache

import numpy as np
import scipy
import scipy.sparse as sparse
from voronoi_reference import (
    compute_occupation,
    compute_overlap,
    inverse_face_mapping,
    native_voronoi,
    split_triangle,
    subdivide_edges,
    subdivision_voronoi,
)

import neuroboros as nb
from neuroboros.surface import Sphere, Surface
from neuroboros.surface.voronoi import VoronoiSurface

GOLDEN_FN = os.path.join(os.path.dirname(__file__), "data", "voronoi_golden.json")
LR = "l"
SLOW_SPACES = {"onavg-ico64", "onavg-ico128"}


def _overlap(native, source, target, n_div=8, t_div=32):
    return dict(
        kind="overlap",
        native=native,
        source=source,
        target=target,
        n_div=n_div,
        t_div=t_div,
    )


CASES = [
    _overlap("onavg-ico32", "native", "onavg-ico8"),
    *[
        _overlap("onavg-ico32", "native", "onavg-ico8", n_div, t_div)
        for n_div in [1, 2, 4]
        for t_div in [8, 16]
    ],
    _overlap("onavg-ico64", "native", "onavg-ico32"),
    _overlap("onavg-ico64", "native", "onavg-ico64"),
    _overlap("onavg-ico64", "onavg-ico32", "onavg-ico64"),
    _overlap("onavg-ico128", "native", "onavg-ico32"),
    _overlap("onavg-ico128", "native", "onavg-ico64"),
    _overlap("onavg-ico128", "native", "onavg-ico128"),
    dict(kind="areas", native="onavg-ico32"),
    dict(kind="areas", native="onavg-ico64"),
]


def case_id(case):
    if case["kind"] == "areas":
        return f"areas-{case['native']}"
    return "overlap-{native}-{source}-{target}-n{n_div}-t{t_div}".format(**case)


def _hash(*arrays):
    """SHA-256 of the shape and little-endian contents of each array."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a)
        dtype = "<f8" if np.issubdtype(a.dtype, np.inexact) else "<i8"
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
    return h.hexdigest()


def digest(result):
    """Digest of an array, or of a sparse matrix in canonical CSR form,
    including its shape."""
    if not sparse.issparse(result):
        return _hash(result)
    mat = sparse.csr_matrix(result, copy=True)
    mat.sort_indices()
    return _hash(np.asarray(mat.shape), mat.indptr, mat.indices, mat.data)


def record(result):
    rec = dict(shape=list(result.shape), sum=float(result.sum()))
    if sparse.issparse(result):
        rec["nnz"] = int(result.nnz)
    rec["sha256"] = digest(result)
    return rec


def check(result, expected):
    """Assert that ``result`` matches a record, checking the shape first."""
    shape = list(result.shape)
    assert shape == expected["shape"], f"shape {shape} != {expected['shape']}"
    assert record(result) == expected


def environment():
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        byteorder=sys.byteorder,
    )


def inputs(case):
    """Digests of the input arrays of a case."""
    native = case["native"]
    coords, faces = nb.geometry("midthickness", LR, native)
    digests = {
        f"{native}/midthickness": _hash(coords),
        f"{native}/faces": _hash(faces),
    }
    templates = {case.get("source"), case.get("target")} - {None, "native"}
    # The native sphere places the template points on the native mesh.
    for space in sorted(templates | {native} if templates else []):
        sphere = nb.geometry("sphere.reg", LR, space, vertices_only=True)
        digests[f"{space}/sphere.reg"] = _hash(sphere)
    return digests


@lru_cache(maxsize=1)
def _prepare(native, n_div):
    coords, faces = nb.geometry("midthickness", LR, native)
    surf = Surface(coords, faces)
    new_coords, e_mapping, graph = subdivide_edges(surf.coords, faces, n_div)
    coords = np.concatenate([surf.coords, new_coords])
    return surf, coords, e_mapping, graph


def _labels(native, n_div, name):
    surf, coords, e_mapping, graph = _prepare(native, n_div)
    if name == "native":
        nn, nnd = native_voronoi(coords, surf.faces, e_mapping, graph)
        return nn, nnd, {}, surf.nv
    sphere = nb.geometry("sphere.reg", LR, native, vertices_only=True)
    points = nb.geometry("sphere.reg", LR, name, vertices_only=True)
    f_indices, weights = Sphere(sphere, surf.faces).barycentric(
        points, eps=1e-7, return_sparse=False
    )
    nn, nnd = subdivision_voronoi(
        coords, surf.faces, e_mapping, graph, f_indices, weights
    )
    f_inv = inverse_face_mapping(f_indices, weights, coords, surf.faces)
    return nn, nnd, f_inv, points.shape[0]


def _vertex_areas(native, n_div=8, t_div=32):
    """``Surface.vertex_areas_nn`` as implemented with the functions."""
    surf, coords, e_mapping, graph = _prepare(native, n_div)
    nn, nnd = native_voronoi(coords, surf.faces, e_mapping, graph)
    areas = np.zeros((surf.nv,))
    ww = split_triangle(t_div)
    for f_idx, f in enumerate(surf.faces):
        a, b, c = sorted(f)
        indices = np.concatenate(
            [e_mapping[(a, b)], e_mapping[(a, c)], e_mapping[(b, c)], [a, b, c]]
        )
        uu = compute_occupation(f_idx, f, coords, indices, nn, nnd, {}, ww)
        for u, m1 in uu.items():
            areas[u] += m1.mean() * surf.face_areas[f_idx]
    return areas


def compute(case):
    """Compute a case with the reference functions."""
    if case["kind"] == "areas":
        return _vertex_areas(case["native"])
    native, n_div = case["native"], case["n_div"]
    surf, coords, e_mapping, _ = _prepare(native, n_div)
    nn1, nnd1, f_inv1, n1 = _labels(native, n_div, case["source"])
    nn2, nnd2, f_inv2, n2 = _labels(native, n_div, case["target"])
    return compute_overlap(
        surf.faces,
        surf.face_areas,
        e_mapping,
        coords,
        nn1,
        nnd1,
        f_inv1,
        nn2,
        nnd2,
        f_inv2,
        n1,
        n2,
        t_div=case["t_div"],
    )


@lru_cache(maxsize=1)
def _overlap_object(native, n_div, t_div):
    coords, faces = nb.geometry("midthickness", LR, native)
    sphere = nb.geometry("sphere.reg", LR, native, vertices_only=True)
    return VoronoiSurface(
        coords, faces, sphere, edge_divisions=n_div, sample_divisions=t_div
    )


def compute_class(case):
    """Compute a case with ``VoronoiSurface``."""
    if case["kind"] == "areas":
        ov = _overlap_object(case["native"], 8, 32)
        return ov.areas("native")
    ov = _overlap_object(case["native"], case["n_div"], case["t_div"])
    for name in [case["source"], case["target"]]:
        if name not in ov.names:
            points = nb.geometry("sphere.reg", LR, name, vertices_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ov.add(name, points)
    return ov.transform(case["source"], case["target"])


def compute_vertex_areas_nn(case):
    """Compute an areas case with ``Surface.vertex_areas_nn``."""
    coords, faces = nb.geometry("midthickness", LR, case["native"])
    return Surface(coords, faces).vertex_areas_nn()


def compute_class_batch(case):
    """Compute a case with ``VoronoiSurface.transforms``, among other pairs."""
    if case["kind"] == "areas":
        return compute_class(case)
    compute_class(case)
    ov = _overlap_object(case["native"], case["n_div"], case["t_div"])
    source, target = case["source"], case["target"]
    pairs = [(target, target), (target, source), (source, target)]
    return ov.transforms(pairs)[2]


def load():
    with open(GOLDEN_FN) as f:
        return json.load(f)


def main():
    source = subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    ).stdout.strip()
    golden = dict(source=source, environment=environment(), inputs={}, cases={})
    for case in CASES:
        print(case_id(case), flush=True)
        golden["inputs"].update(inputs(case))
        golden["cases"][case_id(case)] = dict(case=case, result=record(compute(case)))
    os.makedirs(os.path.dirname(GOLDEN_FN), exist_ok=True)
    with open(GOLDEN_FN, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
