import os

import numpy as np
import pytest
import scipy.sparse as sparse
import voronoi_golden as vg

SLOW = os.environ.get("NEUROBOROS_SLOW_TESTS") == "1"


def test_digest_includes_shape():
    a = sparse.csr_matrix(np.eye(2))
    b = sparse.csr_matrix((a.data, a.indices, a.indptr), shape=(2, 3))
    assert vg.digest(a) != vg.digest(b)
    with pytest.raises(AssertionError, match="shape"):
        vg.check(b, vg.record(a))


def _skip_if_slow(case):
    if case["native"] in vg.SLOW_SPACES and not SLOW:
        pytest.skip("Slow; set NEUROBOROS_SLOW_TESTS=1 to run.")


AREAS_CASES = [case for case in vg.CASES if case["kind"] == "areas"]


def _check_golden(case, compute):
    """Check a result against the golden reference in its environment."""
    _skip_if_slow(case)
    golden = vg.load()
    env = vg.environment()
    diff = {k: (v, env[k]) for k, v in golden["environment"].items() if env[k] != v}
    if diff:
        pytest.skip(f"Not the reference environment (expected, actual): {diff}")
    diff = [k for k, v in vg.inputs(case).items() if golden["inputs"].get(k) != v]
    if diff:
        pytest.skip(f"Inputs differ from the reference: {diff}")
    vg.check(compute(case), golden["cases"][vg.case_id(case)]["result"])


@pytest.mark.parametrize(
    "compute",
    [vg.compute, vg.compute_class, vg.compute_class_uncached],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("case", vg.CASES, ids=vg.case_id)
def test_golden(case, compute):
    _check_golden(case, compute)


@pytest.mark.parametrize("case", AREAS_CASES, ids=vg.case_id)
def test_vertex_areas_nn_golden(case):
    _check_golden(case, vg.compute_vertex_areas_nn)


@pytest.mark.parametrize("case", AREAS_CASES, ids=vg.case_id)
def test_vertex_areas_nn_matches_functions(case):
    _skip_if_slow(case)
    expected = vg.record(vg.compute(case))
    assert vg.record(vg.compute_vertex_areas_nn(case)) == expected


@pytest.mark.parametrize(
    "compute",
    [vg.compute_class, vg.compute_class_uncached],
    ids=lambda f: f.__name__,
)
@pytest.mark.parametrize("case", vg.CASES, ids=vg.case_id)
def test_class_matches_functions(case, compute):
    """Compare the class with the functions in any environment."""
    _skip_if_slow(case)
    assert vg.record(compute(case)) == vg.record(vg.compute(case))
