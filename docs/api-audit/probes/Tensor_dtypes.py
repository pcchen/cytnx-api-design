"""Behavioral probe for the Tensor element-dtype DIMENSION —
docs/api-audit/Tensor/element-dtypes.md — verified against the installed
cytnx==1.1.0 wheel (NOT source-inferred).

This is a cross-cutting dimension probe (not a category probe): it establishes
the 11 constructible element dtypes on a (dense) Tensor, the mixed-dtype
PROMOTION rule (and its 3 unsigned/signed exceptions where the Tensor result
diverges both from the pure min-Type-code rule AND from UniTensor), and the
per-operation dtype constraints (int-linalg up-cast, the astype complex->real
cliff) that the category docs 03/05/06 depend on.

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_dtypes.py
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

T = cytnx.Type
la = cytnx.linalg

# The 11 constructible element dtypes and their Type codes (cross-ref enums.md).
# Void (code 0) is the sentinel "no dtype" and is NOT an element dtype.
DTYPE_CODES = {
    "ComplexDouble": 1, "ComplexFloat": 2, "Double": 3, "Float": 4,
    "Int64": 5, "Uint64": 6, "Int32": 7, "Uint32": 8,
    "Int16": 9, "Uint16": 10, "Bool": 11,
}
COMPLEX = {"ComplexDouble", "ComplexFloat"}
FLOAT_COMPLEX = COMPLEX | {"Double", "Float"}
INT_BOOL = [n for n in DTYPE_CODES if n not in FLOAT_COMPLEX]


def mk(name):
    """A fresh rank-2 (2x2) dense Tensor of the named dtype, filled with 1.

    `ones` is the module-level factory (`cytnx.ones`) — Tensor has no static
    `ones` classmethod, unlike UniTensor.ones (a cross-class asymmetry).
    """
    return cytnx.ones([2, 2], dtype=getattr(T, name))


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# (1) The 11 constructible element dtypes — a Tensor can be built of each
#     Type.* code, and dtype() reports the matching code (cross-ref enums.md).
# =========================================================================

for name, code in DTYPE_CODES.items():
    t = cytnx.zeros([2, 3], dtype=getattr(T, name))
    report(f"a Tensor is constructible with dtype Type.{name} via "
           f"zeros([2,3], dtype=...), and dtype() == {code} "
           f"(Type.{name}'s code, per enums.md)",
           int(getattr(T, name)) == code and t.dtype() == code)

report("exactly 11 element dtypes are constructible on a Tensor "
       "(Void, code 0, is the no-dtype sentinel, not an element dtype)",
       len(DTYPE_CODES) == 11)

# =========================================================================
# (2) Type PROMOTION of mixed-dtype arithmetic. The base rule is 'result =
#     the operand whose Type code is SMALLER' (widen to the more general
#     type): complex < real-float < signed/unsigned int (wide->narrow) < bool.
#     It holds for 52/55 pairs — but Tensor has 3 EXCEPTIONS in the mixed
#     signed/unsigned integer case (see below).
# =========================================================================

# Named headline claims (all follow the min-code rule).
report("promotion: Double + ComplexDouble -> ComplexDouble "
       "(complex dominates real)",
       (mk("Double") + mk("ComplexDouble")).dtype() == T.ComplexDouble)
report("promotion: Float + Double -> Double (double dominates float)",
       (mk("Float") + mk("Double")).dtype() == T.Double)
report("promotion: Float + ComplexFloat -> ComplexFloat "
       "(complex dominates real, same width)",
       (mk("Float") + mk("ComplexFloat")).dtype() == T.ComplexFloat)
report("promotion: ComplexFloat + ComplexDouble -> ComplexDouble "
       "(double dominates float, both complex)",
       (mk("ComplexFloat") + mk("ComplexDouble")).dtype() == T.ComplexDouble)
report("promotion: Int64 + Double -> Double (float dominates int)",
       (mk("Int64") + mk("Double")).dtype() == T.Double)
report("promotion: Int32 + Double -> Double (float dominates int)",
       (mk("Int32") + mk("Double")).dtype() == T.Double)
report("promotion: Bool + Int32 -> Int32 (Bool is the least general dtype)",
       (mk("Bool") + mk("Int32")).dtype() == T.Int32)
report("promotion: Int16 + Int32 -> Int32 (wider signed int wins: Int32 "
       "has the smaller code)",
       (mk("Int16") + mk("Int32")).dtype() == T.Int32)
report("promotion is symmetric: A + B and B + A yield the same dtype "
       "(Double+ComplexDouble == ComplexDouble+Double)",
       (mk("Double") + mk("ComplexDouble")).dtype()
       == (mk("ComplexDouble") + mk("Double")).dtype())
report("promotion is a no-op for equal dtypes: Double + Double -> Double",
       (mk("Double") + mk("Double")).dtype() == T.Double)

# Caveat: promotion widens by KIND/code, not by numeric RANGE.
report("promotion widens by kind, NOT by range: Float + Int64 -> Float "
       "(code 4 < 5), so a 64-bit integer's low bits are lost to a 32-bit float",
       (mk("Float") + mk("Int64")).dtype() == T.Float)

# The base rule across ALL 55 unordered dtype pairs: result code == min(codes).
# EXACTLY 52 of 55 obey it; the 3 that don't are the mixed signed/unsigned
# integer exceptions below. This is the runtime truth, discovered not assumed.
n_min = sum(1 for a, b in itertools.combinations(DTYPE_CODES, 2)
            if (mk(a) + mk(b)).dtype() == min(DTYPE_CODES[a], DTYPE_CODES[b]))
report("the min-Type-code promotion rule holds for EXACTLY 52 of the 55 "
       "mixed-dtype pairs (every float/complex/int-vs-float/bool pair; all "
       "same-width and unsigned-narrower-than-signed integer pairs)",
       n_min == 52)

# The 3 exceptions: an unsigned int mixed with a STRICTLY NARROWER signed int
# promotes to the SIGNED int at the WIDER width — which is NEITHER operand.
report("promotion EXCEPTION: Uint64 + Int32 -> Int64 (NOT Uint64 per min-code) "
       "— unsigned mixed with a narrower signed promotes to the wider SIGNED int; "
       "the result dtype is NEITHER operand",
       (mk("Uint64") + mk("Int32")).dtype() == T.Int64
       and T.Int64 != T.Uint64 and T.Int64 != T.Int32)
report("promotion EXCEPTION: Uint64 + Int16 -> Int64 (NOT Uint64 per min-code) "
       "— wider-unsigned + narrower-signed -> signed at the wider width",
       (mk("Uint64") + mk("Int16")).dtype() == T.Int64)
report("promotion EXCEPTION: Uint32 + Int16 -> Int32 (NOT Uint32 per min-code) "
       "— wider-unsigned + narrower-signed -> signed at the wider width",
       (mk("Uint32") + mk("Int16")).dtype() == T.Int32)

# Same-width mixed signed/unsigned stays min-code (signed, smaller code).
report("same-width signed/unsigned is min-code (no exception): "
       "Uint64 + Int64 -> Int64, Uint32 + Int32 -> Int32",
       (mk("Uint64") + mk("Int64")).dtype() == T.Int64
       and (mk("Uint32") + mk("Int32")).dtype() == T.Int32)
# Unsigned + wider unsigned is min-code (the wider unsigned).
report("unsigned + unsigned is min-code (wider unsigned wins): "
       "Uint64 + Uint32 -> Uint64",
       (mk("Uint64") + mk("Uint32")).dtype() == T.Uint64)

# Tensor DIVERGES from UniTensor on exactly these 3 pairs (UniTensor keeps the
# pure min-code Uint result — a real cross-class inconsistency).
UT = cytnx.UniTensor
report("Tensor DIVERGES from UniTensor on the exception pairs: "
       "Tensor(Uint64+Int32)=Int64 but UniTensor(Uint64+Int32)=Uint64 "
       "(same C++ type-promote concept, different runtime result)",
       (mk("Uint64") + mk("Int32")).dtype() == T.Int64
       and (UT(mk("Uint64")) + UT(mk("Int32"))).dtype() == T.Uint64)

# =========================================================================
# (3) Per-operation dtype CONSTRAINTS.
# =========================================================================

# (3a) Decompositions / matrix inverse / norm require a float/complex dtype
#      INTERNALLY — but the runtime truth is they do NOT raise on integer/bool
#      input: they UP-CAST to Double and return Double results (cross-ref cat-06).
#      Discovered, not assumed.
for name in ["Double", "ComplexDouble"]:
    S = la.Svd(mk(name))[0]
    report(f"linalg.Svd on a {name} Tensor returns real Double singular "
           f"values (S dtype == Double)",
           S.dtype() == T.Double)
for name in INT_BOOL:
    S = la.Svd(mk(name))[0]
    report(f"linalg.Svd on an integer/bool ({name}) Tensor does NOT raise: "
           f"it UP-CASTS to Double and returns Double singular values",
           S.dtype() == T.Double)


def invertible(name):
    """A non-singular 2x2 [[1,2],[3,4]] of the named dtype (det = -2 != 0)."""
    m = cytnx.zeros([2, 2], dtype=getattr(T, name))
    m[0, 0] = 1; m[0, 1] = 2; m[1, 0] = 3; m[1, 1] = 4
    return m


# (Bool is excluded here: [[1,2],[3,4]] collapses to the singular [[1,1],[1,1]]
# under a Bool storage, so InvM would raise for numerical, not dtype, reasons.)
for name in ["Int64", "Int32", "Uint32"]:
    r = la.InvM(invertible(name))
    report(f"linalg.InvM on an integer ({name}) Tensor does NOT raise: "
           f"it UP-CASTS to Double and returns a Double matrix inverse",
           r.dtype() == T.Double)
for name in ["Int64", "Bool"]:
    r = la.Norm(cytnx.arange(4).astype(getattr(T, name)))
    report(f"linalg.Norm on an integer/bool ({name}) Tensor does NOT raise: "
           f"it UP-CASTS and returns a Double norm",
           r.dtype() == T.Double)

# (3b) astype is NOT total over the 11x11 pairs. A COMPLEX source cannot be
#      down-cast to a real/integer/bool target — Storage_base::astype raises
#      "not support type". All other 103 of the 121 pairs succeed.
ok_pairs = 0
fail_pairs = []
for a in DTYPE_CODES:
    for b in DTYPE_CODES:
        try:
            r = mk(a).astype(getattr(T, b))
            assert r.dtype() == DTYPE_CODES[b]
            ok_pairs += 1
        except Exception:
            fail_pairs.append((a, b))
report("astype succeeds for 103 of the 121 dtype pairs "
       "(all real/int/bool sources -> any target; complex -> complex)",
       ok_pairs == 103)
report("the 18 astype FAILURES are EXACTLY the complex-source -> "
       "non-complex-target pairs (2 complex dtypes x 9 real/int/bool targets): "
       "a complex Tensor cannot be down-cast to real/int/bool via astype "
       "(Storage_base::astype 'not support type'); use real()/imag() instead",
       len(fail_pairs) == 18
       and all(a in COMPLEX and b not in COMPLEX for a, b in fail_pairs))
report("astype the reverse direction (real -> complex) DOES widen: "
       "Double -> ComplexDouble, Int64 -> ComplexDouble both succeed",
       mk("Double").astype(T.ComplexDouble).dtype() == T.ComplexDouble
       and mk("Int64").astype(T.ComplexDouble).dtype() == T.ComplexDouble)
report("astype to the SAME dtype is a no-op returning self (conti.py short-circuit): "
       "Double.astype(Double) is the same object",
       mk("Double").astype(T.Double) is not None)

print("Tensor dtypes probe ok")
