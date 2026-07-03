"""Behavioral probe for the Symmetry class (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/Symmetry.md's Parity
and Consistency findings sections is backed by a report() assertion here. Run
with: source tools/env.sh && $PY docs/api-audit/probes/Symmetry.py
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

Symmetry = cytnx.Symmetry

# --- construction: static factories ------------------------------------

u1 = Symmetry.U1()
report("U1() constructs stype U(-1) with n == 1",
       u1.stype() == cytnx.SymType.U and u1.n() == 1 and u1.stype_str() == "U1")

z3 = Symmetry.Zn(3)
report("Zn(3) constructs stype Z(0) with n == 3",
       z3.stype() == cytnx.SymType.Z and z3.n() == 3 and z3.stype_str() == "Z3")

fp = Symmetry.FermionParity()
fn = Symmetry.FermionNumber()
report("FermionParity()/FermionNumber() are both fermionic; U1()/Zn() are not",
       fp.is_fermionic() is True and fn.is_fermionic() is True
       and u1.is_fermionic() is False and z3.is_fermionic() is False)

# --- P4: default (no-arg) constructor always yields U1; the 2-arg C++ ------
# constructor is not exposed to Python at all (py::init<>() is the only
# registered constructor; symmetry_py.cpp:60 comments out the 2-arg ctor).

d0 = Symmetry()
report("Symmetry() default constructor equals Symmetry.U1() (stype/n both match)",
       d0 == u1 and d0.stype_str() == "U1")

try:
    Symmetry(cytnx.SymType.U, 1)
    two_arg_ctor_raised = False
except TypeError:
    two_arg_ctor_raised = True
report("Symmetry(stype, n) (the C++ 2-arg constructor) is NOT exposed to Python; "
       "calling it raises TypeError", two_arg_ctor_raised)

# --- equality: by value (stype, n), not identity ------------------------

a, b = Symmetry.U1(), Symmetry.U1()
report("__eq__ compares Symmetry by value (stype, n), not identity",
       (a == b) is True and (a is b) is False)

c, dd, e = Symmetry.Zn(2), Symmetry.Zn(3), Symmetry.Zn(2)
report("Zn(2) != Zn(3) (different n) but Zn(2) == Zn(2) (same stype/n)",
       (c == dd) is False and (c == e) is True)

report("FermionParity() != FermionNumber() (different stype, despite both fermionic)",
       (fp == fn) is False)

report("__ne__ actually evaluates (not just __eq__): U1() != Zn(3) is True; "
       "U1() != U1() (a fresh, value-equal instance) is False",
       (u1 != z3) is True and (u1 != Symmetry.U1()) is False)

# --- C5: n()'s per-subtype sentinel values --------------------------------

report("FermionParity().n() == -2 and FermionNumber().n() == -1: internal "
       "sentinel values (Symmetry_base's raw storage, reused as a type "
       "discriminant), not a meaningful qnum-range size for these subtypes",
       fp.n() == -2 and fn.n() == -1)

# --- clone(): independent wrapper, equal by value ------------------------
# Symmetry has no Python-reachable mutating method at all (see P3 below), so
# clone()'s copy-vs-view distinction cannot be demonstrated by mutation the
# way Bond.clone() was; we instead confirm the only observable half: a fresh,
# distinct Python object that still compares equal by value.

cl = u1.clone()
report("clone() returns a distinct wrapper object that is still value-equal to the source",
       (cl is not u1) and (cl == u1))

# --- Save/Load round trip -------------------------------------------------

import tempfile
with tempfile.TemporaryDirectory() as d:
    fname = os.path.join(d, "sym")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # missing-extension deprecation notice, not under test
        z3.Save(fname)
    loaded = Symmetry.Load(os.path.join(d, "sym.cysym"))
    report("Save()/Load() round trip preserves value equality (Zn(3))",
           loaded == z3 and loaded.stype_str() == "Z3")

# --- combine_rule / reverse_rule: correct math for U1 and Zn -------------

report("U1.combine_rule(2, 3) == 5 (Q1 + Q2)", u1.combine_rule(2, 3) == 5)
report("U1.combine_rule(2, 3, is_reverse=True) == -5 (combine then negate)",
       u1.combine_rule(2, 3, True) == -5)
report("Zn(3).combine_rule(2, 2) == 1 ((2+2) % 3)", z3.combine_rule(2, 2) == 1)

# --- P1 (headline): Zn out-of-range qnums raise in raw C++ (ValidateZnQnum /
# cytnx_error_msg, cytnx_src/src/Symmetry.cpp) but the Python binding's
# NormalizeZnInput wrapper (symmetry_py.cpp) intercepts BEFORE reaching C++,
# silently reduces the qnum modulo n, and only ever emits a FutureWarning --
# so a value the class's own check_qnum() rejects is nonetheless accepted
# (with just a warning) by combine_rule()/reverse_rule(). Contrast with
# Zn(1) construction and FermionParity.get_fermion_parity(2), both of which
# DO raise a catchable RuntimeError elsewhere in this same class, proving
# Symmetry is not silent-by-default -- this Zn path specifically opts out.

report("Zn(3).check_qnum(4) is False: 4 is genuinely out of the valid [0,3) range",
       z3.check_qnum(4) is False)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out_of_range_result = z3.combine_rule(4, 4)
report("Zn(3).combine_rule(4, 4) does NOT raise for an out-of-range qnum (unlike "
       "check_qnum's own verdict); it silently normalizes (4->1) and returns (1+1)%3 == 2",
       out_of_range_result == 2)
report("...and the only signal of the out-of-range input is a FutureWarning "
       "('deprecated and will be rejected in v2.0.0'), not an exception",
       len(w) == 2 and all(issubclass(x.category, FutureWarning) for x in w)
       and all("deprecated" in str(x.message) and "v2.0.0" in str(x.message) for x in w))

with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    rev_out_of_range = z3.reverse_rule(5)
report("Zn(3).reverse_rule(5) (out-of-range) likewise normalizes-with-warning instead "
       "of raising: 5 -> 2, then reverse_rule(2) == (3-2)%3 == 1",
       rev_out_of_range == 1 and len(w2) == 1 and issubclass(w2[0].category, FutureWarning))

try:
    Symmetry.Zn(1)
    zn1_raised = False
except RuntimeError:
    zn1_raised = True
report("By contrast, Zn(1) (an invalid modulus) DOES raise a catchable RuntimeError -- "
       "proving this class raises hard errors elsewhere, so the combine_rule/reverse_rule "
       "silent-normalize path above is a deliberate carve-out, not a blanket policy",
       zn1_raised)

# --- P2: combine_rule's C++ vector<int64> overload has no Python binding --
# Only the scalar (qnL: int, qnR: int, is_reverse: bool) lambda overload is
# registered in symmetry_py.cpp; Symmetry_base::combine_rule(vector&, vector&)
# is unreachable from Python.

try:
    u1.combine_rule([1, 2], [3, 4])
    list_combine_worked = True
except TypeError:
    list_combine_worked = False
report("combine_rule() rejects list arguments (TypeError): the C++ batch/vector "
       "overload is not bound to Python, only the scalar form is",
       list_combine_worked is False)

# --- P3: combine_rule_/reverse_rule_ (C++ out-param mutating forms) are ---
# entirely absent from the Python binding (not merely mis-bound, as Bond's
# group_duplicates_ was -- here the attribute doesn't exist at all).

report("Symmetry has no combine_rule_ attribute at all (C++ out-param form unbound)",
       not hasattr(Symmetry, "combine_rule_"))
report("Symmetry has no reverse_rule_ attribute at all (C++ out-param form unbound)",
       not hasattr(Symmetry, "reverse_rule_"))
try:
    u1.combine_rule_(0, 1, 2)
    combine_rule_underscore_callable = True
except AttributeError:
    combine_rule_underscore_callable = False
report("...confirmed by direct call: sym.combine_rule_(...) raises AttributeError",
       combine_rule_underscore_callable is False)

# --- P5: __repr__/__str__ return '' ; the actual info text is a print() ---
# side effect (std::cout, redirected into Python's stdout only for the
# duration of the call via py::scoped_ostream_redirect), and the underlying
# C++ Symmetry::print_info() has no direct Python binding of its own.

repr_value = repr(u1)
str_value = str(u1)
report("repr(sym) evaluates to the empty string '' (not a useful representation)",
       repr_value == "")
report("str(sym) is likewise ''", str_value == "")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    repr(u1)
captured = buf.getvalue()
report("...yet calling repr(sym) DOES print human-readable info as a side effect "
       "(captured from stdout), showing the info exists but isn't returned",
       "[Symmetry]" in captured and "type : Abelian, U1" in captured)

report("print_info (the C++ method backing the above side-effect print) has no "
       "direct Python binding of its own -- unreachable except via __repr__/__str__",
       not hasattr(Symmetry, "print_info"))

# --- P6: Python enum name 'SymType' diverges from C++ 'SymmetryType' by ---
# more than the N1 casing rule (an abbreviation), so per 00-methodology.md's
# N3 this is a parity finding, not merely a naming nit.

report("The Python-visible enum is named SymType (not SymmetryType, the C++ name) "
       "-- an N3-relevant abbreviation, not just a casing difference",
       hasattr(cytnx, "SymType") and not hasattr(cytnx, "SymmetryType"))

# --- get_fermion_parity: correct per-type classification -----------------

report("FermionParity: get_fermion_parity(0) == EVEN, get_fermion_parity(1) == ODD",
       fp.get_fermion_parity(0) == cytnx.fermionParity.EVEN
       and fp.get_fermion_parity(1) == cytnx.fermionParity.ODD)
report("FermionNumber: get_fermion_parity(2) == EVEN, get_fermion_parity(3) == ODD",
       fn.get_fermion_parity(2) == cytnx.fermionParity.EVEN
       and fn.get_fermion_parity(3) == cytnx.fermionParity.ODD)
report("Non-fermionic U1.get_fermion_parity(anything) always returns EVEN "
       "(Symmetry_base's default, un-overridden implementation)",
       u1.get_fermion_parity(5) == cytnx.fermionParity.EVEN
       and u1.get_fermion_parity(-3) == cytnx.fermionParity.EVEN)

try:
    fp.get_fermion_parity(2)
    fp_invalid_raised = False
except RuntimeError:
    fp_invalid_raised = True
report("FermionParity.get_fermion_parity(2) raises RuntimeError: 2 is not a valid "
       "parity qnum (only 0/1 are)", fp_invalid_raised)

# --- Consistency finding: FermionParitySymmetry::reverse_rule_ is ----------
# internally self-inconsistent -- it can produce a qnum that the SAME
# object's own check_qnum() rejects. (out = -in + 2, so in=0 -> out=2,
# but check_qnum only accepts [0,2).) This is identical in C++ and Python
# (no Python-side wrapper touches fPar), so it is a Consistency finding
# about FermionParitySymmetry's own internal coherence, not a parity gap.

report("FermionParity.reverse_rule(0) == 2, a value FermionParity.check_qnum() itself "
       "rejects -- reverse_rule_'s formula (-in + 2) is internally inconsistent with "
       "check_qnum's own valid range [0, 2)",
       fp.reverse_rule(0) == 2 and fp.check_qnum(2) is False)

# --- Consistency finding: the Zn-only deprecation-warning safety net is ---
# not applied to FermionParity, even though FermionParity.combine_rule_ also
# implicitly wraps (mod 2) out-of-range input -- silently, with no warning
# at all (not even the FutureWarning Zn gets).

with warnings.catch_warnings(record=True) as w3:
    warnings.simplefilter("always")
    fp_wrapped = fp.combine_rule(5, 5)
report("FermionParity.combine_rule(5, 5) silently wraps out-of-range input "
       "((5+5)%2 == 0) with NO warning at all, unlike Zn's combine_rule -- "
       "the safety net (and its warning) is Zn-only, not applied uniformly",
       fp_wrapped == 0 and len(w3) == 0)

# --- check_qnum/check_qnums: correct per-type range validation -----------

report("U1.check_qnum() accepts any integer (no valid-range restriction)",
       u1.check_qnum(-100) is True and u1.check_qnum(100) is True)
report("Zn(3).check_qnums([0, 1, 4]) is False: batch check fails if ANY element "
       "is out of range", z3.check_qnums([0, 1, 4]) is False)
report("Zn(3).check_qnums([0, 1, 2]) is True: all elements in range",
       z3.check_qnums([0, 1, 2]) is True)

# --- BUG (new): FermionParitySymmetry::check_qnums compares against the -----
# sentinel this->n (-2) instead of the literal bound 2
# (cytnx_src/src/Symmetry.cpp:167-174, specifically line 170:
# `qnums[i] < this->n`), so `qnums[i] < -2` is never true for any qnum >= 0
# -- check_qnums therefore returns False for EVERY non-empty input on a
# FermionParity symmetry, even qnums that are genuinely valid (0 and 1) and
# that the *singular* check_qnum happily accepts. This is a real,
# self-contradictory bug on the same object: check_qnum(0) is True but
# check_qnums([0]) is False for the identical value.

report("BUG: FermionParity.check_qnum(0) is True (0 is a valid parity qnum) "
       "but FermionParity.check_qnums([0]) is False for the SAME value -- "
       "check_qnums (Symmetry.cpp:170) wrongly compares against the sentinel "
       "n() (-2) instead of the literal bound 2, so it rejects every "
       "non-empty input regardless of validity",
       fp.check_qnum(0) is True and fp.check_qnums([0]) is False)
report("...same contradiction for qnum 1: check_qnum(1) is True, "
       "check_qnums([1]) is False",
       fp.check_qnum(1) is True and fp.check_qnums([1]) is False)

print("Symmetry probe ok")
