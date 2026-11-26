import sys
import types

import pytest

from neuroboros.surface import voronoi

MODULE = "neuroboros.surface._voronoi_kernel"
A = 1.0 + 2.0**-27


def test_multiply_add_is_not_fused():
    kernel = pytest.importorskip(MODULE)
    # The exact result is 2**-26 + 2**-54, which a fused multiply-add returns.
    assert kernel.multiply_add(A, A, -1.0) == A * A - 1.0 == 2.0**-26
    assert kernel.multiply_add(3.0, 0.5, 0.25) == 1.75


def test_load_kernel(monkeypatch):
    kernel = pytest.importorskip(MODULE)
    assert voronoi._load_kernel() is kernel
    monkeypatch.setenv("NEUROBOROS_VORONOI_KERNEL", "0")
    assert voronoi._load_kernel() is None
    monkeypatch.setenv("NEUROBOROS_VORONOI_KERNEL", "1")
    assert voronoi._load_kernel() is kernel


def test_load_kernel_not_built(monkeypatch):
    monkeypatch.setitem(sys.modules, MODULE, None)
    assert voronoi._load_kernel() is None


def test_load_kernel_refuses_fused_multiply_add(monkeypatch):
    fused = types.SimpleNamespace(multiply_add=lambda a, b, c: 2.0**-26 + 2.0**-54)
    monkeypatch.setitem(sys.modules, MODULE, fused)
    with pytest.warns(RuntimeWarning, match="fused multiply-add"):
        assert voronoi._load_kernel() is None
