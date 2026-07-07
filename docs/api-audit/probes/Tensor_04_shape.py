"""Behavioral probe for the Tensor audit, category 04 (shape / layout),
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/Tensor/04-shape-layout.md is backed by a
report(...) assertion here. Members covered: permute, permute_, reshape,
reshape_, contiguous, contiguous_, flatten, flatten_, and the leaked raw
make_contiguous shim.

The binding-fidelity findings (T-S3 *args erasure, T-S5 contiguous_ identity
drop, T-S7 flatten_ None) additionally have their raw-C++ side verified by
probes/cpp/Tensor_04_shape.cpp against a source-built libcytnx.

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_04_shape.py
"""
import sys, os, inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

Tensor = cytnx.Tensor

report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")


def _mk():
    """A fresh, contiguous rank-3 Double tensor [2,3,4] = arange(24)."""
    return cytnx.arange(24).reshape(2, 3, 4)


# =========================================================================
# T-S1: permute / reshape return a DISTINCT object that SHARES storage with
# the receiver — a metadata-only VIEW, not a copy (numpy divergence for those
# expecting reshape to sometimes copy, but here it aliases).
# =========================================================================
src = _mk()
p = src.permute(2, 0, 1)
report("permute returns a DISTINCT object (pure — the receiver is untouched)",
       p is not src)
report("permute returns a shared-data VIEW (same_data(src) is True)",
       p.same_data(src) is True)
report("permute VIEW is live: mutating the source shows through the permute",
       returns_view(_mk, lambda s: s.permute(2, 0, 1),
                    lambda h: h.__setitem__((0, 0, 0), 777),  # write through view
                    lambda s: float(s[0, 0, 0].item())) is True)

src = _mk()
r = src.reshape(6, 4)
report("reshape returns a DISTINCT object (pure)", r is not src)
report("reshape returns a shared-data VIEW (same_data(src) is True)",
       r.same_data(src) is True)
report("reshape VIEW is live: a write through the reshape is visible on the source",
       returns_view(_mk, lambda s: s.reshape(6, 4),
                    lambda h: h.__setitem__((0, 0), 555),
                    lambda s: float(s[0, 0, 0].item())) is True)

# =========================================================================
# T-S2: permute_ / reshape_ are correct in-place methods that return SELF
# (the same Python object) — chainable. C++ returns Tensor& ; the pybind
# lambdas return &self (tensor_py.cpp:179-198).
# =========================================================================
a = _mk()
ret = a.permute_(2, 0, 1)
report("permute_ permutes in place and returns SELF (ret is a)", ret is a)
report("permute_ actually reordered the receiver's shape to [4,2,3]",
       a.shape() == [4, 2, 3])

b = _mk()
ret = b.reshape_(6, 4)
report("reshape_ reshapes in place and returns SELF (ret is b)", ret is b)
report("reshape_ actually reshaped the receiver to [6,4]", b.shape() == [6, 4])

# =========================================================================
# T-S3: permute / reshape (and their _ forms) are bound as (*args) pybind
# lambdas — the typed C++ (mapper/new_shape, ...) signature is ERASED, so
# inspect.signature() raises and the docstring reads "(*args)".
# =========================================================================
for name in ("permute", "permute_", "reshape", "reshape_"):
    m = getattr(Tensor, name)
    raised = False
    try:
        inspect.signature(m)
    except ValueError:
        raised = True
    report(f"{name} is bound as a (*args) pybind lambda — inspect.signature() raises ValueError",
           raised)
    report(f"{name}'s docstring exposes the erased (*args) signature, not a typed one",
           "*args" in (m.__doc__ or "").splitlines()[0])

# =========================================================================
# T-S4: contiguous is a conti.py wrapper (Tensor_conti.py:50-55) that
# SHORT-CIRCUITS to self when already contiguous, else forwards to the leaked
# raw make_contiguous shim (a distinct, contiguous COPY).
# =========================================================================
c = _mk()
report("a freshly-built tensor is contiguous", c.is_contiguous() is True)
report("contiguous() short-circuits to SELF when the tensor is already contiguous",
       c.contiguous() is c)

nc = _mk().permute(2, 0, 1)
report("a permuted tensor is NON-contiguous", nc.is_contiguous() is False)
cc = nc.contiguous()
report("contiguous() on a non-contiguous tensor returns a DISTINCT object",
       cc is not nc)
report("contiguous() on a non-contiguous tensor returns an INDEPENDENT copy "
       "(same_data is False) that is contiguous",
       cc.same_data(nc) is False and cc.is_contiguous() is True)

# =========================================================================
# T-S5: contiguous_ is the in-place layout coalesce, BUT the binding returns a
# DISTINCT shared-data wrapper, NOT self — unlike permute_/reshape_. Root
# cause is the C++ signature `Tensor contiguous_()` (returns *this BY VALUE,
# not Tensor&), bound as a plain method pointer (tensor_py.cpp:193). The raw
# C++ side is verified by the C++ probe.
# =========================================================================
d = _mk().permute(2, 0, 1)
report("contiguous_ target is non-contiguous before the call", d.is_contiguous() is False)
ret = d.contiguous_()
report("contiguous_ coalesces the RECEIVER's storage in place (d is now contiguous)",
       d.is_contiguous() is True)
report("contiguous_ returns a DISTINCT object, NOT self (ret is not d) — the "
       "in-place self-return convention is broken (contrast permute_/reshape_)",
       ret is not d)
report("contiguous_'s returned wrapper still SHARES data with the receiver "
       "(same_data(d) is True) — a shared-data view, not an independent copy",
       ret.same_data(d) is True)

# =========================================================================
# T-S6: flatten is a PURE 1-D collapse returning an INDEPENDENT COPY
# (C++ flatten() = clone + contiguous_ + reshape_, Tensor.hpp:1416).
# =========================================================================
e = _mk()
fl = e.flatten()
report("flatten returns a DISTINCT object collapsed to rank 1 ([24])",
       fl is not e and fl.shape() == [24])
report("flatten returns an INDEPENDENT COPY (same_data(src) is False) — the "
       "receiver is untouched (still rank 3)",
       fl.same_data(e) is False and e.shape() == [2, 3, 4])

# =========================================================================
# T-S7: flatten_ is the in-place 1-D collapse — but it returns None, NOT self
# (v1 Tensor C4). Root cause is the C++ signature `void flatten_()`
# (Tensor.hpp:1430), bound as a plain method pointer (tensor_py.cpp:191).
# =========================================================================
g = _mk()
ret = g.flatten_()
report("flatten_ collapses the RECEIVER to rank 1 in place ([24])", g.shape() == [24])
report("flatten_ returns None, NOT self — breaks the in-place self-return "
       "convention its siblings permute_/reshape_ keep (v1 C4)",
       ret is None)

# =========================================================================
# T-S8: the raw make_contiguous shim LEAKS into public dir(Tensor). It is the
# raw C++ contiguous() bound under a renamed name (tensor_py.cpp:192, comment
# "this will be rename by python side conti"); unlike the public contiguous()
# wrapper it does NOT short-circuit — on an already-contiguous tensor it
# returns a NEW wrapper that still SHARES storage (a view).
# =========================================================================
report("the raw make_contiguous shim LEAKS into public dir(Tensor)",
       "make_contiguous" in dir(Tensor))
h = _mk()
mk = h.make_contiguous()
report("make_contiguous does NOT short-circuit: on an already-contiguous "
       "tensor it returns a DISTINCT wrapper (mk is not h)",
       mk is not h)
report("make_contiguous's wrapper still SHARES storage (same_data is True) — a "
       "view, unlike contiguous()'s identity short-circuit",
       mk.same_data(h) is True)

print("Tensor 04 probe ok")
