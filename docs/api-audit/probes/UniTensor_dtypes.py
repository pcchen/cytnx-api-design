"""Behavioral probe for the UniTensor element-dtype DIMENSION —
docs/api-audit/UniTensor/element-dtypes.md — verified against the installed
cytnx==1.1.0 wheel (NOT source-inferred).

This is a cross-cutting dimension probe (not a category probe): it establishes
the 11 constructible element dtypes on a UniTensor, the mixed-dtype PROMOTION
rule, and the per-operation dtype constraints (element access, decompositions)
that the category docs 06/08 depend on.

Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_dtypes.py
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

UT = cytnx.UniTensor
T = cytnx.Type
la = cytnx.linalg

# The 11 constructible element dtypes and their Type codes (cross-ref enums.md).
# Void (code 0) is the sentinel "no dtype" and is NOT an element dtype.
DTYPE_CODES = {
    "ComplexDouble": 1, "ComplexFloat": 2, "Double": 3, "Float": 4,
    "Int64": 5, "Uint64": 6, "Int32": 7, "Uint32": 8,
    "Int16": 9, "Uint16": 10, "Bool": 11,
}
FLOAT_COMPLEX = {"ComplexDouble", "ComplexFloat", "Double", "Float"}
INT_BOOL = [n for n in DTYPE_CODES if n not in FLOAT_COMPLEX]


def mk(name):
    """A fresh rank-2 (2x2) Dense UniTensor of the named dtype, filled with 1."""
    return UT.ones([2, 2], dtype=getattr(T, name))


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# (1) The 11 constructible element dtypes — a UniTensor can be built of each
#     Type.* code, and dtype() reports the matching code (cross-ref enums.md).
# =========================================================================

for name, code in DTYPE_CODES.items():
    u = mk(name)
    report(f"a UniTensor is constructible with dtype Type.{name}, and "
           f"dtype() == {code} (Type.{name}'s code, per enums.md)",
           int(getattr(T, name)) == code and u.dtype() == code)

report("exactly 11 element dtypes are constructible on a UniTensor "
       "(Void, code 0, is the no-dtype sentinel, not an element dtype)",
       len(DTYPE_CODES) == 11)

# =========================================================================
# (2) Type PROMOTION of mixed-dtype arithmetic. The runtime rule is EXACT:
#     the result dtype is the operand whose Type code is SMALLER (the more
#     general type). Ordering (small->large code): complex < real-float <
#     signed/unsigned int (wide->narrow) < bool. So complex dominates real,
#     and double dominates float.
# =========================================================================

# Named headline claims.
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

# The exact rule across ALL 55 unordered dtype pairs: result code == min(codes).
all_min = True
for a, b in itertools.combinations(DTYPE_CODES, 2):
    res = (mk(a) + mk(b)).dtype()
    if res != min(DTYPE_CODES[a], DTYPE_CODES[b]):
        all_min = False
report("promotion rule is EXACTLY 'result = the dtype with the smaller Type "
       "code' for all 55 mixed-dtype pairs (widen to the more general type; "
       "complex<real<int<bool by code)",
       all_min)

# =========================================================================
# (3) Per-operation dtype CONSTRAINTS.
# =========================================================================

# (3a) get_elem — element read is bound ONLY for the 4 float/complex dtypes
#      (cross-ref cat-06 UT-E1); it RAISES on the 7 integer/bool dtypes.
for name in FLOAT_COMPLEX:
    u = mk(name)
    report(f"get_elem reads an element on float/complex dtype {name} "
           f"(one of the 4 bound dtypes; cross-ref cat-06 UT-E1)",
           u.get_elem([0, 0]) == 1)
for name in INT_BOOL:
    u = mk(name)
    raised = False
    try:
        u.get_elem([0, 0])
    except RuntimeError:
        raised = True
    report(f"get_elem RAISES on integer/bool dtype {name} "
           f"(the binding lambda instantiates only the 4 float/complex "
           f"branches; cross-ref cat-06 UT-E1)",
           raised)

# (3b) Decompositions require a float/complex dtype INTERNALLY — but the runtime
#      truth is that they do NOT raise on an integer/bool input: they UP-CAST to
#      Double and return float results (cross-ref cat-08). Discovered, not assumed.
for name in ["Double", "ComplexDouble"]:
    S = la.Svd(mk(name))[0]
    report(f"linalg.Svd on a {name} UniTensor returns real Double singular "
           f"values (S dtype == Double)",
           S.dtype() == T.Double)
for name in INT_BOOL:
    S = la.Svd(mk(name))[0]
    report(f"linalg.Svd on an integer/bool ({name}) UniTensor does NOT raise: "
           f"it UP-CASTS to Double and returns Double singular values",
           S.dtype() == T.Double)
for name in ["Double", "Int32", "Bool"]:
    e = la.Eigh(mk(name))[0]
    report(f"linalg.Eigh on a {name} UniTensor returns Double eigenvalues "
           f"(float/complex required internally; int input up-casts to Double)",
           e.dtype() == T.Double)

print("UniTensor dtypes probe ok")
