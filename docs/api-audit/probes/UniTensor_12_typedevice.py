"""Behavioral probe for UniTensor category 12 — type & device conversion,
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/12-type-device-conversion.md is
backed by a report(...) assertion here. The raw-C++ side of the binding-fidelity
finding (UT-T1: the conti.py `is self` no-op short-circuit vs C++'s always-fresh
object) is verified by probes/cpp/UniTensor_12_typedevice.cpp.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_12_typedevice.py
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

UT = cytnx.UniTensor


def mk():
    """A fresh rank-2 Dense (bosonic) UniTensor, contiguous, filled 1."""
    return UT.ones([2, 2])


def val(u, loc):
    return complex(u.at(loc).value)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# UT-T1 — astype / to bind via conti.py over the raw astype_different_type /
#         to_different_device shims. The conti.py wrapper SHORT-CIRCUITS to
#         `return self` on a no-op (same dtype / same device), so the returned
#         object IS the same Python object — a binding-introduced identity that
#         raw C++ does NOT have (C++ always mints a fresh UniTensor; see the C++
#         probe). A REAL conversion forwards to the raw shim and returns a
#         distinct, independent object.
# =========================================================================

u = mk()
report("astype(same dtype) returns `self` (is self) — the conti.py no-op "
       "short-circuit (`if self.dtype()==dtype: return self`)",
       u.astype(u.dtype()) is u)

u = mk()
a = u.astype(cytnx.Type.ComplexDouble)
report("astype(different dtype) forwards to the raw astype_different_type shim "
       "and returns a DISTINCT, independent object with the new dtype (data "
       "copied, not shared)",
       a is not u and a.dtype() == cytnx.Type.ComplexDouble
       and not u.same_data(a))

u = mk()
report("to(same device) returns `self` (is self) — the conti.py no-op "
       "short-circuit (`if self.device()==device: return self`)",
       u.to(u.device()) is u)

report("astype/to are conti.py wrappers: the raw astype_different_type / "
       "to_different_device shims they forward to LEAK into public dir(UniTensor)",
       "astype_different_type" in dir(UT) and "to_different_device" in dir(UT))

# =========================================================================
# UT-T2 — to_ is bound as `.def(\"to_\", &UniTensor::to_)` with NO py::arg, so
#         its parameter name is erased to `arg0`: keyword calls fail.
# =========================================================================

u = mk()
kw_raises = False
try:
    u.to_(device=u.device())
except TypeError:
    kw_raises = True
report("to_ has an ERASED parameter name (bound without py::arg): calling "
       "to_(device=...) by keyword RAISES TypeError — the param is `arg0`",
       kw_raises)

u = mk()
r = u.to_(u.device())
report("to_ called POSITIONALLY moves the tensor in place and returns self",
       r is u)

# =========================================================================
# UT-T3 — clone / __copy__ produce an INDEPENDENT (deep) copy, not a view.
# =========================================================================

report("clone() returns an INDEPENDENT copy — mutating the clone is NOT visible "
       "through the source (returns_view -> False)",
       returns_view(mk, lambda s: s.clone(),
                    lambda h: h.__setitem__((0, 0), 9.0),
                    lambda s: val(s, [0, 0])) is False)

u = mk()
c = u.clone()
report("clone() returns a distinct object that does NOT share data with the "
       "source (not same_data)",
       c is not u and not u.same_data(c))

# __copy__ is bound to clone -> copy.copy(ut) yields an independent object.
report("copy.copy(ut) (the __copy__ hook, bound to clone) returns an "
       "INDEPENDENT copy (mutation not shared)",
       returns_view(mk, lambda s: copy.copy(s),
                    lambda h: h.__setitem__((0, 0), 5.0),
                    lambda s: val(s, [0, 0])) is False)

# =========================================================================
# UT-T4 — __deepcopy__ is bound to `clone` (arity: self only, NO `memo`
#         parameter), so copy.deepcopy — which calls __deepcopy__(self, memo) —
#         RAISES TypeError. Broken deep-copy hook (cross-ref cat 11 UT-IO2).
# =========================================================================

report("__deepcopy__ is bound to `clone` with NO `memo` parameter "
       "(signature is `__deepcopy__(self) -> UniTensor`)",
       "memo" not in (UT.__deepcopy__.__doc__ or ""))

u = mk()
dc_raises = False
try:
    copy.deepcopy(u)
except TypeError:
    dc_raises = True
report("copy.deepcopy(ut) RAISES TypeError — the __deepcopy__ hook (=clone) "
       "rejects the `memo` argument copy.deepcopy passes (cross-ref cat 11 "
       "UT-IO2)",
       dc_raises)

u = mk()
memo_raises = False
try:
    u.__deepcopy__({})
except TypeError:
    memo_raises = True
report("ut.__deepcopy__({}) RAISES TypeError directly — confirming the hook "
       "takes no memo dict",
       memo_raises)

# =========================================================================
# UT-T5 — numpy bridge GAP: UniTensor has no `.numpy()` method and no
#         `from_numpy` static (pairs with cat-01 UT-C3). The module-level
#         cytnx.from_numpy builds a Tensor, not a UniTensor.
# =========================================================================

report("UniTensor has NO `.numpy()` method and NO `from_numpy` static — the "
       "numpy bridge is absent on UniTensor (capability gap; pairs cat-01 UT-C3)",
       "numpy" not in dir(UT) and "from_numpy" not in dir(UT))

report("the numpy bridge exists only at MODULE level (cytnx.from_numpy builds a "
       "Tensor), never on UniTensor — so there is no numpy round-trip for a "
       "UniTensor",
       hasattr(cytnx, "from_numpy") and not hasattr(UT, "from_numpy"))

# =========================================================================
# UT-T6 — the leaked plumbing set for this category: the raw
#         astype_different_type / to_different_device shims and the raw `cfrom`
#         binding (the actual C++ convert_from) all leak into public dir.
# =========================================================================

leaked = ["astype_different_type", "to_different_device", "cfrom"]
present = [m for m in leaked if m in dir(UT)]
report("the raw plumbing bindings astype_different_type / to_different_device / "
       "cfrom all LEAK into public dir(UniTensor)",
       present == leaked)

# =========================================================================
# UT-T7 — convert_from is a conti.py wrapper over the raw `cfrom` binding; it
#         copies data from a UniTensor of a different structure IN PLACE and
#         returns self.
# =========================================================================

fp = cytnx.Symmetry.U1()
Bi = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [fp])
Bo = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [fp])
blk = cytnx.UniTensor([Bi, Bo], labels=["a", "b"])
dense_src = UT.zeros([2, 2], labels=["a", "b"])
dense_src.at([0, 0]).value = 2.0
r = blk.convert_from(dense_src)
report("convert_from (conti.py wrapper over raw cfrom) copies data from a "
       "different-structure UniTensor IN PLACE and returns self",
       r is blk and complex(blk.at([0, 0]).value) == (2 + 0j))

print("UniTensor 12 probe ok")
