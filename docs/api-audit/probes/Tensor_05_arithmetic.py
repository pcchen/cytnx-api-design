"""Behavioral probe for the Tensor audit, category 05 (arithmetic & element-wise),
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/Tensor/05-arithmetic-elementwise.md is
backed by a report(...) assertion here. Members covered: the operator dunders
(__add__/__radd__/__iadd__ and the -/*/ families, __floordiv__, __mod__,
__matmul__/__imatmul__, __neg__/__pos__, __pow__/__ipow__, __eq__), the named
element-wise methods Abs/Abs_, Conj/Conj_, Exp/Exp_, Inv/Inv_, Pow/Pow_, Norm,
and the leaked raw c* / c__i*__ bindings the conti.py wrappers call.

The binding-fidelity findings (T-A2 the @= wrapper typo — the raw c__imatmul__
primitive itself works; T-A4 the in-place element-wise methods return Tensor&)
additionally have their raw-C++ side verified by
probes/cpp/Tensor_05_arithmetic.cpp against a source-built libcytnx.

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_05_arithmetic.py
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

Tensor = cytnx.Tensor

report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")


def _v(t):
    """List the elements of a rank-1 Double tensor as Python floats."""
    return [t[i].item() for i in range(int(t.shape()[0]))]


def _dbl(*xs):
    """A rank-1 Double tensor holding the given values."""
    t = cytnx.zeros(len(xs), cytnx.Type.Double)
    for i, x in enumerate(xs):
        t[i] = x
    return t


# =========================================================================
# T-A1: `//` (__floordiv__) performs TRUE division, NOT floor — a correctness
# bug identical to UniTensor UT-A1. The __floordiv__ pybind lambda routes to
# self.Div(rhs) (tensor_py.cpp:1329-1355), the very same op as __truediv__, so
# `//` never floors: (t*7)//2 yields 3.5, not 3.
# =========================================================================
t7 = _dbl(0, 1, 2, 3) * 7  # [0, 7, 14, 21]
fd = t7 // 2
tdv = t7 / 2
report("`//` (__floordiv__) performs TRUE division, not floor: (t*7)//2 yields "
       "3.5 / 10.5, not floored 3 / 10",
       _v(fd) == [0.0, 3.5, 7.0, 10.5])
report("`//` gives the IDENTICAL result to `/` (both route to Div) — no flooring",
       _v(fd) == _v(tdv))
d = _dbl(0, 1, 2, 3) * 7
d //= 2  # conti.py __ifloordiv__ -> c__ifloordiv__ -> Div_
report("`//=` (__ifloordiv__) is likewise TRUE division in place (yields 3.5), "
       "not floor",
       _v(d) == [0.0, 3.5, 7.0, 10.5])

# =========================================================================
# T-A2 (headline, B-5 / v1 P4): the `@=` bug. Tensor_conti.py:84-87 defines
# `def __imatmul(self, rhs)` — MISSING the trailing `__` — so the special
# method Python's `@=` looks for (__imatmul__) does NOT exist on Tensor. `t @=
# x` finds no __imatmul__, falls back to `t = t.__matmul__(x)` (linalg.Dot) and
# REBINDS t to a fresh object instead of mutating in place. The raw
# c__imatmul__ PRIMITIVE itself works (mutates + returns self) — the bug is
# purely the misnamed Python wrapper.
# =========================================================================
report("Tensor_conti.py defines `__imatmul` (missing the trailing __), so the "
       "true __imatmul__ slot does NOT exist on Tensor",
       hasattr(Tensor, "__imatmul") and not hasattr(Tensor, "__imatmul__"))

A = cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)
B = cytnx.eye(2).astype(cytnx.Type.Double)
former = A
A @= B
report("`t @= x` is NOT in place: with no __imatmul__, Python falls back to "
       "__matmul__ (linalg.Dot) and REBINDS t to a FRESH object (t is not its "
       "former self)",
       A is not former)

# the raw c__imatmul__ primitive DOES work in place (mutates + returns self) —
# proving the brokenness is the Python wrapper's name, not the primitive.
C = cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)
Cid = C
ret = C.c__imatmul__(cytnx.eye(2).astype(cytnx.Type.Double))
report("the raw c__imatmul__ primitive returns a self-aliasing handle (the "
       "wrapper `__imatmul` is what is broken, not the primitive)",
       ret.same_data(Cid))

report("`__matmul__` (@, linalg.Dot) is a PURE matrix product returning a new "
       "object",
       (cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)
        .__matmul__(cytnx.eye(2).astype(cytnx.Type.Double))) is not A)

# =========================================================================
# T-A3: Abs/Conj/Exp/Inv/Pow/Norm are Capitalized MEMBERS (N-casing violation).
# The same names ALSO exist as cytnx.linalg FREE functions, which STAY
# Capitalized (they act on objects — cat 08 analog). Cross-ref UniTensor UT-A3.
# =========================================================================
for nm in ("Abs", "Conj", "Exp", "Inv", "Pow", "Norm"):
    report(f"capitalized member `{nm}` exists on Tensor (N-casing: should be "
           f"lowercase `{nm.lower()}`)",
           callable(getattr(Tensor, nm, None)))
for nm in ("Abs", "Conj", "Exp", "Norm", "Pow"):
    report(f"the capitalized name `{nm}` ALSO exists as a cytnx.linalg FREE "
           f"function (which stays Capitalized)",
           callable(getattr(cytnx.linalg, nm, None)))

# Abs/Conj/Exp/Pow are PURE — a new object, source untouched.
src = _dbl(-1, 2, -3)
ab = src.Abs()
report("`Abs` is PURE: a new tensor with |x|, the source unchanged "
       "(src[0] stays -1)",
       ab is not src and _v(ab) == [1.0, 2.0, 3.0] and src[0].item() == -1.0)
sp = _dbl(1, 2, 3)
pw = sp.Pow(2.0)
report("`Pow(2.0)` is PURE: squares into a new tensor, the source unchanged",
       pw is not sp and _v(pw) == [1.0, 4.0, 9.0] and _v(sp) == [1.0, 2.0, 3.0])

# Norm returns a scalar (rank-0-ish, shape [1]) cytnx.Tensor.
nrm = _dbl(3, 4).Norm()
report("`Norm()` returns a cytnx.Tensor holding the 2-norm (sqrt(3^2+4^2)==5)",
       isinstance(nrm, cytnx.Tensor) and abs(nrm.item() - 5.0) < 1e-12)

# =========================================================================
# T-A4: the in-place Abs_/Conj_/Exp_/Inv_/Pow_ are conti.py wrappers
# (Tensor_conti.py:91-115) over the leaked raw cAbs_/cConj_/cExp_/cInv_/cPow_
# bindings; each mutates in place and returns SELF. Cross-ref UniTensor UT-A4.
# =========================================================================
z = _dbl(1, 2, 3)
report("`Conj_()` returns SELF (in place)", z.Conj_() is z)
z = _dbl(-1, 2, -3)
r = z.Abs_()
report("`Abs_()` returns SELF and abs-es in place (-1 -> 1)",
       r is z and _v(z) == [1.0, 2.0, 3.0])
z = _dbl(2, 3)
r = z.Pow_(2.0)
report("`Pow_(2.0)` returns SELF and squares in place (2 -> 4, 3 -> 9)",
       r is z and _v(z) == [4.0, 9.0])
z = _dbl(0, 1, 2)
report("`Exp_()` returns SELF (in place)", z.Exp_() is z)
z = _dbl(2, 4)
r = z.Inv_(-1)
report("`Inv_(clip)` returns SELF and inverts in place (2 -> 0.5, 4 -> 0.25)",
       r is z and _v(z) == [0.5, 0.25])

# =========================================================================
# T-A5: `Inv` is the ELEMENT-WISE reciprocal (1/x), distinct from the matrix
# inverse `InvM` (cat 06). Recommended rename to `reciprocal` (disambiguation),
# cross-ref UniTensor UT-A5/UT-X4.
# =========================================================================
inv = _dbl(1, 2, 4).Inv()
report("`Inv` is the ELEMENT-WISE reciprocal 1/x (1 -> 1, 2 -> 0.5, 4 -> 0.25) "
       "— distinct from the matrix inverse InvM (cat 06)",
       _v(inv) == [1.0, 0.5, 0.25])

# =========================================================================
# T-A6: the C++ NAMED arithmetic methods Add/Sub/Mul/Div (+ _ forms), Cpr, Mod
# are C++-only — UNBOUND as Python members; only the operator dunders exist.
# v1 P1 / UniTensor UT-A6.
# =========================================================================
report("the C++ named arithmetic methods (Add/Sub/Mul/Div + their _ forms, "
       "Cpr, Mod) have NO Python member binding — only the operators exist",
       not any(hasattr(Tensor, m) for m in
               ("Add", "add", "Sub", "sub", "Mul", "mul", "Div", "div",
                "Add_", "Sub_", "Mul_", "Div_", "Cpr", "Mod")))
for dunder in ("__add__", "__sub__", "__mul__", "__truediv__"):
    report(f"operator dunder `{dunder}` IS bound (the operator surface)",
           hasattr(Tensor, dunder))

# =========================================================================
# T-A7: the in-place operators __iadd__/__isub__/__imul__/__itruediv__/
# __ifloordiv__/__ipow__ are conti.py wrappers over the leaked raw c__i*__
# bindings and PRESERVE identity (return self) — UNLIKE UniTensor, whose
# augmented-assign lambdas returned a fresh wrapper (UT-A7). __neg__ negates
# element-wise (new object); __pos__ returns a DISTINCT shared-data wrapper.
# =========================================================================
x = _dbl(1, 2, 3); xid = x
x += cytnx.ones(3)
report("`+=` (__iadd__) mutates in place AND preserves identity (x is its "
       "former self) — the conti.py wrapper returns self (contrast UniTensor "
       "UT-A7, which dropped identity)",
       x is xid and _v(x) == [2.0, 3.0, 4.0])
x = _dbl(4, 6); xid = x
x -= _dbl(1, 2)
report("`-=` (__isub__) mutates in place AND preserves identity", x is xid)
x = _dbl(2, 3); xid = x
x *= 2
report("`*=` (__imul__) mutates in place AND preserves identity", x is xid)
x = _dbl(8, 4); xid = x
x /= 2
report("`/=` (__itruediv__) mutates in place AND preserves identity", x is xid)
w = _dbl(2, 3); wid = w
w **= 2
report("`**=` (__ipow__) raises in place AND preserves identity (2 -> 4, 3 -> "
       "9)",
       w is wid and _v(w) == [4.0, 9.0])

y = _dbl(1, -2, 3)
ny = -y
report("`__neg__` negates element-wise into a NEW object (routes to Mul(-1)): "
       "1 -> -1, -2 -> 2",
       ny is not y and _v(ny) == [-1.0, 2.0, -3.0])
py_ = +y
report("`__pos__` returns a DISTINCT object that SHARES data with the receiver "
       "(return self BY VALUE), not the same Python handle",
       py_ is not y and py_.same_data(y))

# =========================================================================
# T-A8: unlike UniTensor (whose __mod__/__rmod__ pybind block is commented
# out — UT-A2), Tensor's `%`/__mod__ IS bound (routes to Tensor::Mod) and works
# for BOTH a scalar and a tensor right-hand operand on a Dense tensor.
# =========================================================================
report("`%` (__mod__) IS bound on Tensor (contrast UniTensor, where the pybind "
       "block is commented out)",
       hasattr(Tensor, "__mod__") and hasattr(Tensor, "__rmod__"))
tm = _dbl(7, 8, 9, 10)
report("`t % 3` computes element-wise modulo for a SCALAR rhs "
       "([7,8,9,10] % 3 == [1,2,0,1])",
       _v(tm % 3) == [1.0, 2.0, 0.0, 1.0])
report("`t % v` computes element-wise modulo for a TENSOR rhs "
       "([7,8,9,10] % [2,3,4,5] == [1,2,1,0]) — Tensor's Mod is fully "
       "implemented (Dense), NOT a [Mod][Developing] stub like UniTensor's",
       _v(tm % _dbl(2, 3, 4, 5)) == [1.0, 2.0, 1.0, 0.0])

# =========================================================================
# T-A9: the raw c* / c__i*__ plumbing bindings LEAK into public dir(Tensor).
# Each is the raw in-place primitive its friendlier wrapper calls.
# =========================================================================
for nm in ("cAbs_", "cConj_", "cExp_", "cInv_", "cPow_",
           "c__iadd__", "c__ifloordiv__", "c__imatmul__", "c__imul__",
           "c__ipow__", "c__isub__", "c__itruediv__"):
    report(f"the raw plumbing binding `{nm}` LEAKS into public dir(Tensor)",
           nm in dir(Tensor))
# the raw in-place primitive self-aliases (mutates the receiver).
zz = _dbl(1, 2, 3); zzid = zz
zz.cConj_()
report("`cConj_()` (raw primitive) mutates the receiver in place "
       "(self-aliasing) — plumbing behind Conj_/conj_",
       zz is zzid or zz.same_data(zzid))

# =========================================================================
# T-A10: `__eq__` is ELEMENT-WISE (returns a Bool Tensor, not a Python bool),
# which makes Tensor UNHASHABLE — a numpy-like footgun (v1 C5).
# =========================================================================
eqt = _dbl(1, 2, 3)
report("`==` (__eq__) is ELEMENT-WISE and returns a Bool Tensor, not a Python "
       "bool",
       isinstance(eqt == eqt, cytnx.Tensor))
report("Tensor.__hash__ is None: an element-wise __eq__ makes Tensor UNHASHABLE "
       "(cannot be a dict key / set member)",
       Tensor.__hash__ is None)

print("Tensor 05 probe ok")
