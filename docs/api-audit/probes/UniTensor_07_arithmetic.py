"""Behavioral probe for UniTensor category 07 — arithmetic & element-wise,
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/07-arithmetic-elementwise.md is
backed by a report(...) assertion here. The raw-C++ side of the
binding-fidelity findings is verified by probes/cpp/UniTensor_07_arithmetic.cpp.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_07_arithmetic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

UT = cytnx.UniTensor


def scalar(v, dtype=cytnx.Type.Double):
    """A fresh 1-element Dense UniTensor holding the value `v`."""
    u = UT.zeros([1, 1], dtype=dtype)
    u.set_elem([0, 0], v)
    return u


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# UT-A1 — CORRECTNESS: `//` (__floordiv__) performs TRUE division, not floor.
#          The pybind lambda routes __floordiv__ straight to Div (unitensor_py
#          .cpp:1223), so `x // y` == `x / y`: it does NOT floor.
# =========================================================================

# Build a tensor holding 7.0, then compute (u * 7.0) // 2.0 elementwise.
u = scalar(1.0)
r = (u * 7.0) // 2.0
report("`//` (__floordiv__) performs TRUE division, not floor: (u*7.0)//2.0 "
       "yields 3.5 elementwise (Python's // would floor to 3.0) — the binding "
       "routes __floordiv__ to Div",
       r.get_elem([0, 0]) == 3.5)

# Same with a UniTensor divisor: 7 // 2 -> 3.5, not 3.
x, y = scalar(7.0), scalar(2.0)
report("`//` with a UniTensor divisor is also true division: 7.0 // 2.0 -> 3.5 "
       "elementwise (not 3)",
       (x // y).get_elem([0, 0]) == 3.5)

# __ifloordiv__ likewise divides truly in place.
z = scalar(7.0)
z //= 2.0
report("`//=` (__ifloordiv__) is likewise true division in place: 7.0 //= 2.0 "
       "leaves 3.5, not 3",
       z.get_elem([0, 0]) == 3.5)

# =========================================================================
# UT-A2 — BINDING GAP: `%` (__mod__/__rmod__) is ABSENT from Python, though the
#          C++ side has linalg::Mod and cytnx::operator% for UniTensor. The
#          pybind __mod__ block is COMMENTED OUT (unitensor_py.cpp:1311-1363).
# =========================================================================

report("`%` is absent: UniTensor has NO __mod__ (the pybind block is commented "
       "out) — hasattr(UniTensor, '__mod__') is False",
       hasattr(UT, "__mod__") is False)
report("`%` is absent on the reflected side too: no __rmod__",
       hasattr(UT, "__rmod__") is False)

mod_raises = False
try:
    scalar(7.0) % 2.0
except TypeError:
    mod_raises = True
report("using `%` on a UniTensor raises TypeError (unsupported operand) — the "
       "operator is unbound (its C++ linalg::Mod/operator% exist; see the C++ "
       "probe)",
       mod_raises)

# =========================================================================
# UT-A3 — N-CASING: the element-wise members are Capitalized (Conj/Trace/Norm/
#          Pow/Transpose/Dagger and their _ forms) — they should be lowercase.
#          They ALSO exist as linalg FREE functions that stay Capitalized
#          (cross-ref cat 08).
# =========================================================================

CAP_MEMBERS = ["Conj", "Conj_", "Trace", "Trace_", "Norm", "Pow", "Pow_",
               "Transpose", "Transpose_", "Dagger", "Dagger_", "Inv"]
for m in CAP_MEMBERS:
    report(f"capitalized member `{m}` exists as a UniTensor method (N-casing: "
           f"should be lowercase)",
           hasattr(UT, m))

report("`normalize`/`normalize_` are already lowercase (N-conformant) — no rename",
       hasattr(UT, "normalize") and hasattr(UT, "normalize_"))

# The same names ALSO exist as linalg FREE functions, which stay Capitalized
# per N-casing (they act on objects) — the cat-08 cross reference.
report("the capitalized names ALSO exist as `linalg` FREE functions "
       "(Conj/Trace/Norm/Pow/Inv), which stay Capitalized per N-casing (cat 08)",
       all(f in dir(cytnx.linalg) for f in
           ("Conj", "Trace", "Norm", "Pow", "Inv")))

# Norm returns a Tensor (the 2-norm scalar), not a UniTensor.
report("`Norm()` returns a cytnx.Tensor (the 2-norm), not a UniTensor",
       type(UT.ones([2, 2]).Norm()).__name__ == "Tensor")

# =========================================================================
# UT-A4 — BINDING FIDELITY: the in-place Conj_/Trace_/Transpose_/Dagger_/
#          normalize_/Pow_ (and __ipow__) are conti.py wrappers over LEAKED raw
#          `c*` bindings. The wrapper returns self; the raw C++ returns &*this.
# =========================================================================

# The public in-place forms return self (the conti.py wrapper does `return self`).
t = UT.ones([2, 2]); report("Conj_() returns self (in place)", t.Conj_() is t)
t = UT.ones([2, 2]); report("Transpose_() returns self (in place)", t.Transpose_() is t)
t = UT.ones([2, 2]); report("Dagger_() returns self (in place)", t.Dagger_() is t)
t = UT.ones([2, 2]); report("normalize_() returns self (in place)", t.normalize_() is t)
t = UT.ones([2, 2]); report("Trace_(0,1) returns self (in place)", t.Trace_(0, 1) is t)
p = scalar(3.0); report("Pow_(2.0) returns self and squares in place (3->9)",
                        p.Pow_(2.0) is p and p.get_elem([0, 0]) == 9.0)
q = scalar(3.0); q **= 2.0
report("`**=` (__ipow__) raises in place (conti.py wrapper over c__ipow__): "
       "3 **= 2 leaves 9",
       q.get_elem([0, 0]) == 9.0)

# The raw `c*` bindings LEAK into dir(UniTensor) — plumbing that should be hidden.
LEAKED = ["cConj_", "cDagger_", "cPow_", "cTrace_", "cTranspose_",
          "cnormalize_", "cInv_", "c__ipow__"]
for c in LEAKED:
    report(f"the raw plumbing binding `{c}` LEAKS into public dir(UniTensor) "
           f"(the conti.py wrapper calls it)",
           c in dir(UT))

# =========================================================================
# UT-A5 — REDUNDANCY: `Inv` (pure) exists, but there is NO public `Inv_` —
#          only the leaked raw `cInv_`. C++ has UniTensor &Inv_(clip) (probe).
# =========================================================================

report("`Inv` (pure element-wise inverse) is a public member",
       hasattr(UT, "Inv"))
report("there is NO public in-place `Inv_` — only the leaked raw `cInv_` "
       "(C++ has UniTensor &Inv_(clip); see the C++ probe)",
       (not hasattr(UT, "Inv_")) and "cInv_" in dir(UT))

# =========================================================================
# UT-A6 — OPERATORS vs NAMED METHODS: the named Add/Sub/Mul/Div (and their _
#          forms) are C++-only — NOT bound as Python members; only the operator
#          dunders are exposed (they route to linalg::Add/Sub/Mul/Div).
# =========================================================================

for name in ["Add", "Sub", "Mul", "Div", "Add_", "Sub_", "Mul_", "Div_"]:
    report(f"named method `{name}` is NOT a public Python member (C++-only; "
           f"reached only through the operator dunders)",
           not hasattr(UT, name))

# The operator dunders themselves are all present.
DUNDERS = ["__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__",
           "__mul__", "__rmul__", "__imul__", "__truediv__", "__rtruediv__",
           "__itruediv__", "__floordiv__", "__rfloordiv__", "__ifloordiv__",
           "__neg__", "__pos__", "__pow__", "__ipow__"]
for d in DUNDERS:
    report(f"operator dunder `{d}` is bound", hasattr(UT, d))

# =========================================================================
# UT-A7 — the operator dunders' behavior: pure `__add__` etc. return a new
#          object; `__neg__` negates (routes to Mul(-1)); `__pos__` returns the
#          tensor unchanged (shared data). In-place `__iadd__`/`__imul__`/… mutate
#          the receiver but the binding returns a NEW wrapper (identity dropped,
#          same pattern as twist_ / UT-S7).
# =========================================================================

a = scalar(2.0)
report("`__add__` is pure: a + a leaves a unchanged and returns a new tensor "
       "(2+2 -> 4, source still 2)",
       (a + a).get_elem([0, 0]) == 4.0 and a.get_elem([0, 0]) == 2.0)

n = scalar(5.0)
report("`__neg__` negates element-wise (routes to Mul(-1)): -5 -> -5.0",
       (-n).get_elem([0, 0]) == -5.0)

p0 = UT.ones([1, 1])
pp = +p0
pp.set_elem([0, 0], 9.0)
report("`__pos__` returns the tensor unchanged, SHARING data (mutating the "
       "result shows through the source)",
       p0.get_elem([0, 0]) == 9.0)

src = scalar(2.0)
alias = src
src += UT.ones([1, 1])
report("`__iadd__` mutates the receiver in place (the aliased handle sees 3.0) "
       "but the binding returns a NEW wrapper — identity is dropped (same "
       "pattern as twist_/UT-S7)",
       alias.get_elem([0, 0]) == 3.0 and (src is not alias))

print("UniTensor 07 probe ok")
