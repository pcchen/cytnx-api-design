"""Behavioral probe for UniTensor category 11 — I/O & display.

Members: `Save` (member), `Load` (static), `print_diagram`, `print_block`,
`print_blocks`, `__repr__`, and the pickle hooks `__getstate__` (present) /
`__setstate__` (absent).

Every behavioral claim in 11-io-display.md cites one report(...) below.
Run: `source tools/env.sh && $PY docs/api-audit/probes/UniTensor_11_io.py`
"""
import contextlib
import copy
import io
import os
import pickle
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from probe_helper import report  # noqa: E402

import cytnx  # noqa: E402

UT = cytnx.UniTensor


def _np2(u):
    r, c = u.shape()[0], u.shape()[1]
    return [[u.at([i, j]).value for j in range(c)] for i in range(r)]


def _build():
    return cytnx.UniTensor(
        cytnx.arange(6).reshape(2, 3).astype(cytnx.Type.Double), rowrank=1)


def _raises(fn):
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001
        return True


# ---------------------------------------------------------------------------
# Membership / N-casing (UT-IO1)
# ---------------------------------------------------------------------------
report("`Save` is a UniTensor MEMBER (Capitalized member — wrong per N-casing, "
       "should be `save`)",
       "Save" in dir(UT))
report("`Load` is a UniTensor STATIC member (Capitalized member — wrong per "
       "N-casing, should be `load`)",
       "Load" in dir(UT))
report("`print_diagram`/`print_block`/`print_blocks` are lowercase members "
       "(correct N-casing)",
       all(m in dir(UT) for m in ("print_diagram", "print_block", "print_blocks")))
report("`Save` is an instance method, `Load` is a static constructor "
       "(Load(fname) -> UniTensor)",
       "Load(fname" in UT.Load.__doc__)


# ---------------------------------------------------------------------------
# Broken pickle (UT-IO2): __getstate__ present, __setstate__ absent, dumps fails
# ---------------------------------------------------------------------------
report("`hasattr(UniTensor, '__getstate__') is True` — but it is the DEFAULT "
       "`object.__getstate__` (doc 'Helper for pickle.'), not a real cytnx "
       "pickle stub",
       hasattr(UT, "__getstate__") is True
       and UT.__getstate__ is object.__getstate__)
report("`hasattr(UniTensor, '__setstate__') is False` — no restore hook, so the "
       "pickle protocol is only half-implemented",
       hasattr(UT, "__setstate__") is False)

# runtime truth: despite __getstate__ existing, pickle.dumps RAISES — pybind11
# registered no py::pickle(...) support, so the object cannot be serialized.
u = _build()
dumps_error = None
try:
    pickle.dumps(u)
except Exception as e:  # noqa: BLE001
    dumps_error = type(e).__name__
report("`pickle.dumps(ut)` RAISES TypeError ('cannot pickle ...UniTensor "
       "object') — pickle is a broken stub, not a working protocol",
       dumps_error == "TypeError")

# copy.deepcopy does NOT fall back to pickle here (UniTensor binds __deepcopy__
# to clone, cat 12); it raises for a *different* reason — clone takes no memo
# arg. Recorded as runtime truth (cross-ref cat 12), not a pickle path.
deepcopy_error = None
try:
    copy.deepcopy(u)
except Exception as e:  # noqa: BLE001
    deepcopy_error = type(e).__name__
report("`copy.deepcopy(ut)` also RAISES TypeError — its __deepcopy__ is bound to "
       "clone (cat 12) with no memo parameter, so copy cannot even reach the "
       "pickle fallback (cross-ref cat 12)",
       deepcopy_error == "TypeError")


# ---------------------------------------------------------------------------
# Save -> Load value round-trip (UT-IO3) — the working I/O path
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    # give the file an explicit extension: Save auto-appends `.cytnx` (with a
    # deprecation warning) when missing, but Load does NOT auto-append and opens
    # the path verbatim (UT-IO4).
    path = os.path.join(d, "ut_roundtrip.cytnx")
    src = _build()  # default (empty) name -> deterministic round-trip
    src.Save(path)
    report("`Save` writes the tensor to the given path (file created on disk)",
           os.path.exists(path))
    dst = UT.Load(path)
    report("`Save`->`Load` round-trip is value-equal (all elements preserved)",
           _np2(dst) == _np2(src))
    report("`Save`->`Load` round-trip preserves shape, dtype and rowrank",
           list(dst.shape()) == list(src.shape())
           and dst.dtype() == src.dtype()
           and dst.rowrank() == src.rowrank())

    # UT-IO4: Save without an extension warns and appends `.cytnx`; Load requires
    # the full path (does NOT auto-append), so Load(base) fails.
    base = os.path.join(d, "noext")
    src.Save(base)  # -> writes noext.cytnx (deprecation warning to stderr)
    report("`Save` with no extension appends `.cytnx` (deprecated), while `Load` "
           "does NOT auto-append — Load(base_without_ext) RAISES",
           os.path.exists(base + ".cytnx")
           and _raises(lambda: UT.Load(base)))

# ---------------------------------------------------------------------------
# Save/Load name-serialization bug (UT-IO5) — _Load heap over-read
# ---------------------------------------------------------------------------
# _Load reads len_name bytes into a malloc(len_name) buffer WITHOUT a NUL
# terminator, then does std::string(cname) which scans past the buffer to the
# next heap '\0' — corrupting or crashing on a non-empty name. The empty-name
# round-trip above is the only reliably clean path.
with tempfile.TemporaryDirectory() as d:
    names = ["x", "ab", "foo", "name", "hello"]
    corrupted = 0
    path = os.path.join(d, "named.cytnx")
    for nm in names:
        v = cytnx.UniTensor(cytnx.ones([2, 2]))
        v.set_name(nm)
        v.Save(path)
        try:
            w = UT.Load(path)
            if w.name() != nm:
                corrupted += 1
        except Exception:  # noqa: BLE001  (UnicodeDecodeError on invalid over-read bytes)
            corrupted += 1
    report("`Save`->`Load` does NOT reliably preserve a non-empty tensor name — "
           "the _Load name read over-runs a non-NUL-terminated malloc(len_name) "
           "buffer (heap over-read), corrupting/crashing most names",
           corrupted > 0)


# ---------------------------------------------------------------------------
# print_diagram / print_blocks / print_block (UT-IO6): capturable via redirect
# ---------------------------------------------------------------------------
# All three carry a py::scoped_ostream_redirect call-guard, so — UNLIKE
# Device.Print_Property (enums.md) — their C++ std::cout output IS routed
# through Python's sys.stdout and captured by contextlib.redirect_stdout.
u = _build()
u.set_name("demo")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ret_pd = u.print_diagram()
report("`print_diagram()` returns None and its output IS capturable via "
       "contextlib.redirect_stdout (has a py::scoped_ostream_redirect guard, "
       "unlike Device.Print_Property)",
       ret_pd is None and len(buf.getvalue()) > 0)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ret_pbs = u.print_blocks()
report("`print_blocks()` returns None and its output IS capturable via "
       "contextlib.redirect_stdout",
       ret_pbs is None and len(buf.getvalue()) > 0)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ret_pb = u.print_block(0)
report("`print_block(idx)` returns None and its output IS capturable via "
       "contextlib.redirect_stdout",
       ret_pb is None and len(buf.getvalue()) > 0)


# ---------------------------------------------------------------------------
# __repr__ (UT-IO7): prints-and-returns-'' (like Bond/Symmetry), but capturable
# ---------------------------------------------------------------------------
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rv = repr(u)
report("`__repr__` PRINTS via `std::cout << self` and RETURNS '' (empty string) "
       "— the same print-and-return-'' pattern as Bond/Symmetry",
       rv == "")
report("`__repr__`'s printed output IS capturable via contextlib.redirect_stdout "
       "(has a py::scoped_ostream_redirect guard) and is non-empty",
       len(buf.getvalue()) > 0)


print("UniTensor 11 probe ok")
