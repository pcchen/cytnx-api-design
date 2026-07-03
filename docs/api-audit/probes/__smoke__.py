"""Smoke test: confirms the venv, cytnx import, and probe helpers work."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view


def make():
    return cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)


def mutate(handle):
    handle[0, 0] = 42.0


def read(src):
    return float(src[0, 0].item())


# permute() shares the underlying storage: mutating the permuted handle is
# observable on the source -> view semantics -> returns_view is True.
is_view = returns_view(make, lambda src: src.permute(1, 0), mutate, read)
report("Tensor.permute() returns a view (mutation visible on source)", is_view is True)

# clone() makes an independent deep copy: mutating the clone must not affect
# the source -> copy semantics -> returns_view is False.
is_copy = returns_view(make, lambda src: src.clone(), mutate, read)
report("Tensor.clone() returns a copy (mutation NOT visible on source)", is_copy is False)

print("smoke ok")
