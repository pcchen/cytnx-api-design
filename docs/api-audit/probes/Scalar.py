"""Behavioral probe for the Scalar class (Cytnx 1.1.0).

Scalar is Cytnx's dtype-tagged single-number value type (a tagged union over
the 11 numeric dtype codes of cytnx.Type). Every behavioral claim made in
docs/api-audit/per-class/Scalar.md's Parity and Consistency findings sections
is backed by a report() assertion here. Run with:
  source tools/env.sh && $PY docs/api-audit/probes/Scalar.py

Note: several assertions below deliberately trigger C++-side cytnx_error_msg
failures (RuntimeError). Those print a stack trace to stderr; that is the
error being *caught* and asserted on, not a probe failure -- the [PASS] line
still prints and the script still exits 0.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

S = cytnx.Scalar
Type = cytnx.Type


def raises(fn, exc=Exception):
    """Return the exception type name if fn() raises `exc`, else 'NO-RAISE'."""
    try:
        fn()
        return "NO-RAISE"
    except exc as e:
        return type(e).__name__


# --- construction & dtype tagging ----------------------------------------
# The Python-visible dtype of a freshly constructed Scalar is decided by
# pybind overload resolution, which does NOT match C++/numpy dtype intuition
# for Python-native ints and bools (see the two findings below).

report("Scalar(3.0) (a Python float) is tagged Double (Type code 3)",
       S(3.0).dtype() == Type.Double and S(3.0).dtype() == 3)
report("Scalar(1+2j) (a Python complex) is tagged ComplexDouble (Type code 1)",
       S(1 + 2j).dtype() == Type.ComplexDouble and S(1 + 2j).dtype() == 1)

report("Scalar(2) (a Python int) is tagged Uint64 (code 6), NOT Int64 (5): "
       "pybind tries the SupportsInt constructor overloads in registration "
       "order and uint64 is registered first, so a plain Python int silently "
       "becomes an *unsigned* Scalar",
       S(2).dtype() == Type.Uint64 and S(2).dtype() == 6)
report("Scalar(True) (a Python bool) is ALSO tagged Uint64 (6), NOT Bool "
       "(11): bool is a subclass of int, so it matches the same uint64 "
       "overload before ever reaching the explicit bool overload -- a Python "
       "caller cannot construct a Bool-dtype Scalar via Scalar(True)",
       S(True).dtype() == Type.Uint64)

report("The C++ 2-arg constructor Scalar(value, dtype) is NOT exposed to "
       "Python (only single-argument inits are bound); Scalar(3.0, Type.Double) "
       "raises TypeError, so Python callers cannot directly pick a dtype at "
       "construction and must round-trip through numpy scalars or astype()",
       raises(lambda: S(3.0, Type.Double)) == "TypeError")

# numpy scalars are the only way to construct specific real dtypes directly:
f = S(np.float32(2.0))   # Float   (code 4)
i64 = S(np.int64(5))     # Int64   (code 5)
d = S(3.0)               # Double  (code 3)
cd = S(1 + 1j)           # ComplexDouble (code 1)
report("numpy-typed scalars construct the matching dtype directly: "
       "Scalar(np.float32(2)) is Float(4), Scalar(np.int64(5)) is Int64(5)",
       f.dtype() == Type.Float and i64.dtype() == Type.Int64)

# --- B3: dtype promotion widens to the LOWER Type code (= more general) ---
# cytnx.Type orders codes ComplexDouble=1 < ComplexFloat=2 < Double=3 <
# Float=4 < Int64=5 < ... < Bool=11 (see enums.md); a binary Scalar op
# promotes both operands to min(code) -- the *more general* type -- by
# up-casting the higher-code operand first, so real+complex never hits the
# per-subtype "cannot operate real and complex" error.

report("Double + ComplexDouble promotes to ComplexDouble (code 1): the result "
       "widens to the lower Type code (the more general dtype), matching B3",
       (d + cd).dtype() == Type.ComplexDouble)
report("Double(3) + Float(4) promotes to Double(3): widen to the lower code",
       (d + f).dtype() == Type.Double)
report("Float(4) + Int64(5) promotes to Float(4): widen to the lower code",
       (f + i64).dtype() == Type.Float)
report("promotion is value-correct as well as dtype-correct: "
       "complex(Double(3) + ComplexDouble(1+1j)) == (4+1j)",
       complex(d + cd) == (4 + 1j))

# --- conversion to Python scalars: float()/int()/complex() ---------------

report("float(Scalar) and int(Scalar) extract a real Scalar's value: "
       "float(Scalar(3.0)) == 3.0, int(Scalar(3.0)) == 3",
       float(d) == 3.0 and int(d) == 3)
report("complex(Scalar) extracts a complex Scalar: complex(Scalar(1+1j)) == (1+1j)",
       complex(cd) == (1 + 1j))

report("float() of a complex Scalar RAISES RuntimeError (no silent lossy "
       "real-part extraction): float(Scalar(1+1j)) fails via "
       "ComplexDoubleScalar::to_cytnx_double's cytnx_error_msg (B4)",
       raises(lambda: float(cd)) == "RuntimeError")
report("int() of a complex Scalar likewise RAISES RuntimeError",
       raises(lambda: int(cd)) == "RuntimeError")

report("BUG: complex() of a *real* Scalar RAISES RuntimeError -- the __complex__ "
       "binding builds its result from s.real() AND s.imag(), but imag() is "
       "undefined for real scalars (see below), so complex(Scalar(3.0)) fails "
       "instead of returning (3+0j)",
       raises(lambda: complex(d)) == "RuntimeError")

# --- B5: arithmetic operators (+ - * /) ----------------------------------

report("+, -, *, / on two Scalars compute the expected values: "
       "3.0+2.0==5, 3.0-2.0==1, 3.0*2.0==6, 3.0/2.0==1.5",
       float(S(3.0) + S(2.0)) == 5.0 and float(S(3.0) - S(2.0)) == 1.0
       and float(S(3.0) * S(2.0)) == 6.0 and float(S(3.0) / S(2.0)) == 1.5)

report("a Scalar on the LEFT accepts a Python-native or numpy number on the "
       "right (implicit conversion): Scalar(3.0)+2.0, Scalar(3.0)+2, and "
       "Scalar(3.0)+np.float32(2) all succeed and stay Double(3)",
       float(d + 2.0) == 5.0 and (d + 2.0).dtype() == Type.Double
       and float(d + 2) == 5.0 and float(d + np.float32(2)) == 5.0)

report("ASYMMETRY: a Python-native number on the LEFT of a Scalar RAISES "
       "TypeError -- 2.0 + Scalar(3.0) is not commutative-usable (the reflected "
       "__radd__ is bound only for numpy-scalar right-hand operands, not "
       "Python floats), so mixed number+Scalar expressions are order-sensitive",
       raises(lambda: 2.0 + d) == "TypeError")
report("...and even a numpy scalar on the left fails: np.float64(2) + Scalar(3.0) "
       "raises TypeError too (numpy coerces the Scalar before pybind's reflected "
       "overload can fire -- see the scalar_py.cpp FOR_EACH_NUMPY_RTYPE comment)",
       raises(lambda: np.float64(2) + d) == "TypeError")

report("Scalar exposes arithmetic ONLY through operators, not named methods: "
       "there is no .add()/.sub()/.mul()/.div() (the C++ radd/rsub/rmul/rdiv "
       "are unbound), so B5's 'operator == named method' pairing is vacuous here",
       not any(hasattr(d, m) for m in ("add", "sub", "mul", "div",
                                       "radd", "rsub", "rmul", "rdiv")))

# --- N2/B1: in-place arithmetic and mutation semantics -------------------

x = S(3.0)
xid = id(x)
x += S(2.0)
report("in-place += mutates and returns the SAME Python object (identity "
       "preserved): after x += Scalar(2.0), id(x) is unchanged and x == 5.0",
       id(x) == xid and float(x) == 5.0)

report("ASYMMETRY vs '+': in-place real += complex RAISES RuntimeError. The "
       "binary '+' promotes the real operand up to complex first (so Double + "
       "ComplexDouble works, above), but __iadd__ calls the real subtype's "
       "iadd(complex) directly with no promotion, hitting 'Cannot operate real "
       "and complex values'",
       raises(lambda: S(3.0).__iadd__(cd)) == "RuntimeError")

y = S(-4.0)
iabs_ret = y.iabs()
report("iabs() is in-place and returns None (not a new Scalar): after "
       "Scalar(-4.0).iabs() the receiver is 4.0 and the call returned None. "
       "NOTE the name uses an 'i' PREFIX, not the trailing-underscore in-place "
       "convention (N2) -- its pure counterpart is abs()",
       iabs_ret is None and float(y) == 4.0)
z = S(9.0)
isqrt_ret = z.isqrt()
report("isqrt() is in-place and returns None: Scalar(9.0).isqrt() leaves the "
       "receiver at 3.0. Same 'i'-prefix-not-'_'-suffix N2 issue; pure "
       "counterpart is sqrt()",
       isqrt_ret is None and float(z) == 3.0)

# --- B5: comparison operators (< <= > >= == !=) ---------------------------

report("all six comparison operators work on real Scalars: 3.0<5.0, 3.0<=3.0, "
       "5.0>3.0, 3.0>=3.0, 3.0==3.0, 3.0!=5.0 all True; 3.0>5.0 is False",
       (S(3.0) < S(5.0)) and (S(3.0) <= S(3.0)) and (S(5.0) > S(3.0))
       and (S(3.0) >= S(3.0)) and (S(3.0) == S(3.0)) and (S(3.0) != S(5.0))
       and not (S(3.0) > S(5.0)))
report("comparison across dtypes works (operands are promoted first): "
       "Scalar(3.0) [Double] < Scalar(np.int64(5)) [Int64] is True",
       (S(3.0) < S(np.int64(5))) is True)
report("ordering comparison on complex Scalars RAISES RuntimeError (complex "
       "has no total order): Scalar(1+1j) < Scalar(2+2j) fails via "
       "'comparison not supported for complex type' (B4)",
       raises(lambda: cd < S(2 + 2j)) == "RuntimeError")

# --- conj / real / imag --------------------------------------------------

cc = S(1 + 2j)
cc_conj = cc.conj()
report("conj() returns a NEW Scalar with the conjugated value and does NOT "
       "mutate the receiver (pure, B1): Scalar(1+2j).conj() == (1-2j) while the "
       "original is still (1+2j)",
       complex(cc_conj) == (1 - 2j) and complex(cc) == (1 + 2j))
report("conj() of a real Scalar is a value-preserving no-op: "
       "Scalar(3.0).conj() == 3.0",
       float(S(3.0).conj()) == 3.0)

report("real()/imag() of a complex Scalar return the parts as real (Double) "
       "Scalars: Scalar(1+1j).real()==1.0, .imag()==1.0, both dtype Double(3)",
       float(cd.real()) == 1.0 and float(cd.imag()) == 1.0
       and cd.real().dtype() == Type.Double and cd.imag().dtype() == Type.Double)

a = S(5.0)
b = a.real()
b += S(1.0)
report("real() returns an INDEPENDENT copy, not a view (B2): mutating the "
       "result of Scalar(5.0).real() leaves the source at 5.0",
       float(a) == 5.0 and float(b) == 6.0)

report("real() of a real Scalar is defined (returns the value), but imag() of a "
       "real Scalar RAISES RuntimeError ('real type Scalar does not have imag "
       "part') -- real()/conj() are total, imag() is partial, and this partiality "
       "is exactly what breaks complex() on real scalars above",
       float(S(3.0).real()) == 3.0 and raises(lambda: S(3.0).imag()) == "RuntimeError")

# --- abs / sqrt (pure counterparts of iabs / isqrt) ----------------------

report("abs() returns a new Scalar with the magnitude: Scalar(-2.0).abs()==2.0",
       float(S(-2.0).abs()) == 2.0)
report("abs() of a complex Scalar returns the (real, Double) magnitude: "
       "Scalar(3+4j).abs() == 5.0 with dtype Double(3)",
       float(S(3 + 4j).abs()) == 5.0 and S(3 + 4j).abs().dtype() == Type.Double)
report("sqrt() returns a new Scalar: Scalar(4.0).sqrt() == 2.0",
       float(S(4.0).sqrt()) == 2.0)
report("sqrt() of a NEGATIVE real Scalar returns NaN (it stays real -- there is "
       "no automatic real->complex promotion): Scalar(-1.0).sqrt() is nan",
       np.isnan(float(S(-1.0).sqrt())))

# --- astype --------------------------------------------------------------

report("astype() converts between numeric dtypes: Scalar(3.0).astype(Type.Float) "
       "is dtype Float(4)",
       S(3.0).astype(Type.Float).dtype() == Type.Float)
report("astype() from complex to a real dtype RAISES RuntimeError (documented "
       "@attention in Scalar.hpp; use real()/imag() instead): "
       "Scalar(1+1j).astype(Type.Double) fails (B4)",
       raises(lambda: cd.astype(Type.Double)) == "RuntimeError")

# --- maxval / minval (static factories) ----------------------------------

report("maxval(dtype)/minval(dtype) are static factories returning the dtype's "
       "extremal value: maxval(Int16)==32767, minval(Int16)==-32768",
       int(S.maxval(Type.Int16)) == 32767 and int(S.minval(Type.Int16)) == -32768)
report("maxval() of a complex dtype RAISES RuntimeError ('maxval not supported "
       "for complex type') -- these extremal factories are real-dtype-only (B4)",
       raises(lambda: S.maxval(Type.ComplexDouble)) == "RuntimeError")

# --- dtype / print / __repr__ --------------------------------------------

report("dtype() returns the integer Type code (not a Type enum member): "
       "Scalar(3.0).dtype() == 3 == int(Type.Double)",
       d.dtype() == 3 and d.dtype() == int(Type.Double))

report("print() returns None and writes the value + dtype label to stdout "
       "(it is a side-effecting method, not a getter)",
       S(3.0).print() is None)

report("UNLIKE Symmetry (whose __repr__ returns '' and only prints as a side "
       "effect, Symmetry.md P5), Scalar.__repr__ RETURNS a useful non-empty "
       "string built via an ostringstream: repr(Scalar(3.0)) contains the value "
       "and the dtype name",
       repr(S(3.0)) != "" and "3" in repr(S(3.0)) and "Double" in repr(S(3.0)))

print("Scalar probe ok")
