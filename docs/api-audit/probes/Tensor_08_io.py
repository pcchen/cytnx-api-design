"""Behavioral probe for Tensor category 08 — I/O.

Members: `Save` (member), `Load` (static), `Tofile` (member), `Fromfile`
(static), and the pickle hooks `__getstate__` (present, default) / `__setstate__`
(absent).

Every behavioral claim in 08-io.md cites one report(...) below. Contrasts the
UniTensor cat-11 defects: the broken pickle protocol (UT-IO2) reappears
identically here, but the `_Load` NAME heap over-read (UT-IO5) has NO analog —
a dense `Tensor` carries no name field, so its Save/Load round-trip is
deterministically clean.

Run: `source tools/env.sh && $PY docs/api-audit/probes/Tensor_08_io.py`
"""
import contextlib  # noqa: F401
import copy
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from probe_helper import report  # noqa: E402

import cytnx  # noqa: E402

T = cytnx.Tensor


def _flat(t):
    """Row-major element list of a (contiguous) tensor via its storage."""
    st = t.contiguous().storage()
    return [st[i] for i in range(len(st))]


def _build():
    return cytnx.arange(6).reshape(2, 3).astype(cytnx.Type.Double)


def _raises(fn):
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001
        return True


# ---------------------------------------------------------------------------
# Membership / N-casing (T-IO1)
# ---------------------------------------------------------------------------
report("`Save`/`Load`/`Tofile`/`Fromfile` are Tensor members (Capitalized — "
       "wrong per N-casing; should be `save`/`load`/`tofile`/`fromfile`)",
       all(m in dir(T) for m in ("Save", "Load", "Tofile", "Fromfile")))
report("`Save`/`Tofile` are instance methods; `Load`/`Fromfile` are static "
       "constructors (Load(fname) -> Tensor, Fromfile(fname, dtype, count) -> "
       "Tensor)",
       isinstance(T.__dict__["Load"], staticmethod)
       and isinstance(T.__dict__["Fromfile"], staticmethod)
       and not isinstance(T.__dict__["Save"], staticmethod)
       and not isinstance(T.__dict__["Tofile"], staticmethod))


# ---------------------------------------------------------------------------
# Broken pickle (T-IO2) — identical to UniTensor UT-IO2
# ---------------------------------------------------------------------------
report("`hasattr(Tensor, '__getstate__') is True` — but it is the DEFAULT "
       "`object.__getstate__` (doc 'Helper for pickle.'), not a real cytnx "
       "pickle stub",
       hasattr(T, "__getstate__") is True and T.__getstate__ is object.__getstate__)
report("`hasattr(Tensor, '__setstate__') is False` — no restore hook, so the "
       "pickle protocol is only half-implemented",
       hasattr(T, "__setstate__") is False)

t = _build()
dumps_error = None
try:
    pickle.dumps(t)
except Exception as e:  # noqa: BLE001
    dumps_error = type(e).__name__
report("`pickle.dumps(t)` RAISES TypeError ('cannot pickle ...Tensor object') — "
       "pybind11 registered no py::pickle(...), so pickle is a broken stub (same "
       "defect as UniTensor UT-IO2)",
       dumps_error == "TypeError")

deepcopy_error = None
try:
    copy.deepcopy(t)
except Exception as e:  # noqa: BLE001
    deepcopy_error = type(e).__name__
report("`copy.deepcopy(t)` also RAISES TypeError — its __deepcopy__ is bound to "
       "clone (cat 07) with no memo parameter, so copy cannot reach the pickle "
       "fallback (cross-ref cat 07)",
       deepcopy_error == "TypeError")


# ---------------------------------------------------------------------------
# Save -> Load value round-trip (T-IO3) — the working I/O path, and it is
# deterministically clean: a dense Tensor has NO name field, so the UniTensor
# `_Load` name heap over-read (UT-IO5) has NO analog here.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "t_roundtrip.cytn")
    src = _build()
    src.Save(path)
    report("`Save` writes the tensor to the given path (file created on disk)",
           os.path.exists(path))
    dst = T.Load(path)
    report("`Save`->`Load` round-trip is value-equal (all elements preserved)",
           _flat(dst) == _flat(src))
    report("`Save`->`Load` round-trip preserves shape and dtype",
           list(dst.shape()) == list(src.shape()) and dst.dtype() == src.dtype())

    # T-IO3 (no UT-IO5 analog): round-trip is clean across MANY tensors,
    # including non-contiguous (permuted) and complex/int dtypes — a dense
    # Tensor has no name field, so nothing to over-read. Contrast UniTensor,
    # where >=3 of 5 non-empty names corrupt EVERY run.
    dtypes = [cytnx.Type.Double, cytnx.Type.ComplexDouble, cytnx.Type.Int64]
    clean = 0
    total = 0
    for i in range(9):
        a = cytnx.arange((i + 1) * 6).reshape(2, 3, i + 1).astype(dtypes[i % 3])
        if i % 2:
            a = a.permute(2, 0, 1)  # non-contiguous view
        p = os.path.join(d, f"rt{i}.cytn")
        a.Save(p)
        b = T.Load(p)
        total += 1
        if (_flat(b) == _flat(a) and list(b.shape()) == list(a.shape())
                and b.dtype() == a.dtype()
                and b.is_contiguous() == a.is_contiguous()):
            clean += 1
    report("`Save`->`Load` round-trips EVERY tensor cleanly (9/9: values, shape, "
           "dtype, is_contiguous) — Tensor has NO name field, so the UniTensor "
           "`_Load` name heap over-read (UT-IO5) has NO analog here",
           clean == total and total == 9)

    # T-IO4: Save auto-appends the extension when missing (deprecated), while
    # Load opens the path verbatim and does NOT auto-append.
    base = os.path.join(d, "noext")
    src.Save(base)  # -> writes noext.cytn (deprecation warning to stderr)
    report("`Save` with no extension appends `.cytn` (deprecated) — NOTE the "
           "extension is `.cytn`, NOT UniTensor's `.cytnx` (cross-class "
           "inconsistency)",
           os.path.exists(base + ".cytn"))
    report("`Load` does NOT auto-append an extension — Load(base_without_ext) "
           "RAISES (asymmetric with Save's auto-append)",
           _raises(lambda: T.Load(base)))


# ---------------------------------------------------------------------------
# Tofile / Fromfile raw-binary round-trip (T-IO5) — and the shape-inference
# hazard: Fromfile needs dtype (+count) and CANNOT recover the rank/shape.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    src = _build()  # shape (2, 3), Double
    raw = os.path.join(d, "raw.bin")
    src.Tofile(raw)
    report("`Tofile` writes raw element bytes (headerless) to disk (file created)",
           os.path.exists(raw))

    got = cytnx.Tensor.Fromfile(raw, cytnx.Type.Double)  # count defaults to -1 (all)
    report("`Tofile`->`Fromfile` raw round-trip preserves the VALUES (all 6 "
           "elements, row-major)",
           _flat(got) == _flat(src))
    report("`Fromfile` CANNOT infer shape — it returns a FLAT rank-1 Tensor "
           "([6]) from a (2,3) source; the caller must reshape (shape-inference "
           "hazard, unlike Save/Load which stores the shape)",
           list(got.shape()) == [6] and got.rank() == 1)

    partial = cytnx.Tensor.Fromfile(raw, cytnx.Type.Double, 4)
    report("`Fromfile(count=n)` reads only the first n elements ([4] from 6) — "
           "count is required to read a prefix; there is no shape metadata",
           list(partial.shape()) == [4])

    report("`Fromfile` REQUIRES a dtype argument (raw bytes carry no type tag) — "
           "Fromfile(fname) with no dtype RAISES",
           _raises(lambda: cytnx.Tensor.Fromfile(raw)))


print("Tensor 08 probe ok")
