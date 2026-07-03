"""Behavioral probe for the Bond class (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/Bond.md's Parity and
Consistency findings sections is backed by a report() assertion here. Run
with: source tools/env.sh && $PY docs/api-audit/probes/Bond.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view


def u1_bond(qnums_degs):
    """A symmetric U(1) Bond built from a list of (qnum, degeneracy) pairs."""
    qs = [cytnx.Qs(q) >> d for q, d in qnums_degs]
    return cytnx.Bond(cytnx.BD_IN, qs, [cytnx.Symmetry.U1()])


# --- construction -----------------------------------------------------

b_reg = cytnx.Bond(4)
report("Bond(dim) constructs a regular bond with the given dim",
       b_reg.dim() == 4 and b_reg.type() == cytnx.bondType.BD_REG)

b_ket = cytnx.Bond(3, cytnx.bondType.BD_KET)
report("Bond(dim, bond_type) sets the requested bondType",
       b_ket.type() == cytnx.bondType.BD_KET)

b_sym = u1_bond([(0, 2), (1, 3)])
report("Bond(bond_type, qnums, symmetries) builds a symmetric bond with dim == sum(degs)",
       b_sym.dim() == 5 and b_sym.Nsym() == 1)
report("qnums() reflects the constructed quantum numbers",
       b_sym.qnums() == [[0], [1]])

# --- combineBond (pure) vs combineBond_ (in-place) [N2 / B1] ----------

b1 = cytnx.Bond(2)
combined = b1.combineBond(cytnx.Bond(3))
report("combineBond() returns a new Bond and leaves the receiver unchanged",
       b1.dim() == 2 and combined.dim() == 6)

b1c = cytnx.Bond(2)
ret = b1c.combineBond_(cytnx.Bond(3))
report("combineBond_() mutates the receiver in place", b1c.dim() == 6)
report("combineBond_() returns None (not self, unlike redirect_/set_type)", ret is None)

# combineBond/combineBond_ also accept a list of bonds; combineBonds/combineBonds_
# (C++ [[deprecated]]) are exposed to Python under separate names but produce the
# identical result -- confirming they are pure duplication, not distinct behavior.
b2 = cytnx.Bond(2)
via_combine = b2.combineBond([cytnx.Bond(3), cytnx.Bond(2)])
b3 = cytnx.Bond(2)
via_deprecated = b3.combineBonds([cytnx.Bond(3), cytnx.Bond(2)])
report("combineBond(list) and the deprecated combineBonds(list) give identical results",
       via_combine.dim() == via_deprecated.dim() == 12)

# --- redirect (pure) vs redirect_ (in-place) [N2 / B1] -----------------

r = cytnx.Bond(2, cytnx.bondType.BD_KET)
rr = r.redirect()
report("redirect() returns a new Bond and leaves the receiver's type unchanged",
       r.type() == cytnx.bondType.BD_KET and rr.type() == cytnx.bondType.BD_BRA)

r2 = cytnx.Bond(2, cytnx.bondType.BD_KET)
rret = r2.redirect_()
report("redirect_() mutates the receiver's type in place",
       r2.type() == cytnx.bondType.BD_BRA)
report("redirect_() returns the identical (self) wrapper object", rret is r2)

# --- equality: by value, not identity ----------------------------------

e1 = cytnx.Bond(3, cytnx.bondType.BD_REG)
e2 = cytnx.Bond(3, cytnx.bondType.BD_REG)
report("__eq__ compares Bonds by value, not by identity",
       (e1 == e2) is True and (e1 is e2) is False)

# --- clone() independence ------------------------------------------------

is_view = returns_view(
    make=lambda: cytnx.Bond(2, cytnx.bondType.BD_KET),
    derive=lambda src: src.clone(),
    mutate=lambda h: h.redirect_(),
    read=lambda src: src.type(),
)
report("clone() produces an independent Bond (mutating the clone leaves the source untouched)",
       is_view is False)

# --- qnums()/getDegeneracies()/syms(): Python always returns a copy [B2] ------
# C++ has a non-const `qnums()`/`getDegeneracies()`/`syms()` overload that
# returns a mutable reference (a view onto the Bond's internal state);
# pybind binds all three call sites through STL-container-by-value
# conversion, so the Python object handed back is always an independent
# list.

is_view_qnums = returns_view(
    make=lambda: u1_bond([(0, 2), (1, 3)]),
    derive=lambda src: src.qnums(),
    mutate=lambda h: h.__setitem__(0, [999]),
    read=lambda src: src.qnums(),
)
report("Bond.qnums() returns a copy in Python, not the C++-side mutable-reference view",
       is_view_qnums is False)

is_view_degs = returns_view(
    make=lambda: u1_bond([(0, 2), (1, 3)]),
    derive=lambda src: src.getDegeneracies(),
    mutate=lambda h: h.__setitem__(0, 999),
    read=lambda src: src.getDegeneracies(),
)
report("Bond.getDegeneracies() returns a copy in Python (same B2 pattern as qnums())",
       is_view_degs is False)

is_view_syms = returns_view(
    make=lambda: u1_bond([(0, 2), (1, 3)]),
    derive=lambda src: src.syms(),
    mutate=lambda h: h.__setitem__(0, cytnx.Symmetry.Zn(2)),
    read=lambda src: src.syms(),
)
report("Bond.syms() returns a copy in Python (same B2 pattern as qnums()/getDegeneracies())",
       is_view_syms is False)

# --- set_type: mutates in place despite lacking a trailing `_` [N2] -----

st = cytnx.Bond(3, cytnx.bondType.BD_REG)
st_ret = st.set_type(cytnx.bondType.BD_KET)
report("set_type() mutates the receiver in place even though its name has no trailing '_'",
       st.type() == cytnx.bondType.BD_KET)
report("set_type() returns the identical (self) wrapper object, like redirect_()",
       st_ret is st)

rt = cytnx.Bond(3, cytnx.bondType.BD_REG)
rt_ret = rt.retype(cytnx.bondType.BD_KET)
report("retype() is the true pure counterpart: receiver's type is left unchanged",
       rt.type() == cytnx.bondType.BD_REG and rt_ret.type() == cytnx.bondType.BD_KET)

# --- group_duplicates_ / group_duplicates: the underscore is backwards [B1/N2] --
# Native `group_duplicates_(mapper)` is bound to C++'s *const*, clone-returning
# `Bond::group_duplicates(mapper)` -- despite the trailing underscore, it does
# NOT mutate the receiver, and the `mapper` out-argument (a plain
# std::vector<uint64_t>&) is silently not written back into the Python list
# passed in, because pybind converts it by value rather than aliasing it.
# The Python-only wrapper `group_duplicates()` (no trailing underscore, added
# in Bond_conti.py) correctly returns (new_bond, mapper) and IS the pure form
# -- but Python has no working binding at all for C++'s true in-place
# `Bond_impl::group_duplicates_()`.

gd = u1_bond([(0, 2), (0, 3), (1, 1)])
before = gd.qnums()
mapper_arg = []
gd_out = gd.group_duplicates_(mapper_arg)
report("group_duplicates_() does NOT mutate the receiver, despite the trailing '_'",
       gd.qnums() == before)
report("group_duplicates_() returns a new (grouped) Bond, distinct from the receiver",
       gd_out is not gd and gd_out.qnums() == [[0], [1]])
report("group_duplicates_()'s `mapper` out-argument is silently left unfilled",
       mapper_arg == [])

gd2 = u1_bond([(0, 2), (0, 3), (1, 1)])
before2 = gd2.qnums()
gd2_out, mapper2 = gd2.group_duplicates()
report("group_duplicates() (no underscore) is the correctly-behaving pure/copy form: "
       "receiver unchanged, new Bond grouped, mapper correctly populated",
       gd2.qnums() == before2 and gd2_out.qnums() == [[0], [1]] and mapper2 == [0, 0, 1])

# --- getDegeneracy: BROKEN in the installed 1.1.0 wheel [B4] -----------
# cytnx_src/cytnx/Bond_conti.py (repo source) defines a *fixed*
# `getDegeneracy(self, qnum, return_indices=False)` that shadows the native
# pybind overload correctly. But the installed .venv wheel's
# site-packages/cytnx/Bond_conti.py instead has TWO stacked `@add_method`
# definitions of `getDegeneracy`; the second (which wins, since add_method
# monkey-patches Bond.getDegeneracy via setattr each time) requires
# `return_indices` positionally (no default) and references an undefined
# name `lqnum` in its body. The net effect: every call to the public
# `getDegeneracy` raises, regardless of arguments -- the only presently
# working path to a degeneracy lookup is the "internal" `c_getDegeneracy_refarg`.

bd = u1_bond([(0, 2), (1, 3)])

try:
    bd.getDegeneracy([0])
    got_type_error = False
except TypeError:
    got_type_error = True
report("getDegeneracy(qnum) with no return_indices raises TypeError "
       "(installed wheel's wrapper has no default for return_indices)", got_type_error)

try:
    bd.getDegeneracy([0], False)
    got_name_error = False
except NameError:
    got_name_error = True
report("getDegeneracy(qnum, return_indices) ALSO raises (NameError: 'lqnum' is not "
       "defined) -- the public method is unconditionally broken in installed 1.1.0",
       got_name_error)

inds = []
out = bd.c_getDegeneracy_refarg([0], inds)
report("c_getDegeneracy_refarg(qnum, indices) is the only currently-working "
       "degeneracy lookup (mutates the passed-in `indices` python list in place)",
       out == 2 and inds == [0])

# --- operator*/operator*= : declared in C++, NOT bound to Python [B5] --
# Bond.hpp declares `operator*` (== combineBond) and `operator*=`
# (== combineBond_), but bond_py.cpp never binds __mul__/__imul__, so the
# operator form is entirely unavailable from Python even though the named
# method it should mirror (combineBond/combineBond_) works fine.

mul_missing = not hasattr(cytnx.Bond, "__mul__")
imul_missing = not hasattr(cytnx.Bond, "__imul__")
report("Bond has no __mul__ in Python even though C++ defines operator* (== combineBond)",
       mul_missing)
report("Bond has no __imul__ in Python even though C++ defines operator*= (== combineBond_)",
       imul_missing)

mop1 = cytnx.Bond(2)
mop2 = cytnx.Bond(3)
try:
    mop1 * mop2
    mul_raised = False
except TypeError:
    mul_raised = True
report("b1 * b2 raises TypeError (operator* not bound in Python)", mul_raised)

mop3 = cytnx.Bond(2)
mop4 = cytnx.Bond(3)
try:
    mop3 *= mop4
    imul_raised = False
except TypeError:
    imul_raised = True
report("b1 *= b2 raises TypeError (operator*= not bound in Python)", imul_raised)

# --- operator!= : not explicitly bound, but works via Python's default ----
# __ne__ delegation to __eq__ (C++ declares operator!= explicitly; Python
# relies on the implicit default rather than a dedicated binding).
ne1 = cytnx.Bond(3, cytnx.bondType.BD_REG)
ne2 = cytnx.Bond(4, cytnx.bondType.BD_REG)
report("__ne__ works correctly via Python's default delegation to __eq__, "
       "even though bond_py.cpp never binds it explicitly",
       (ne1 != ne2) is True and (ne1 != cytnx.Bond(3, cytnx.bondType.BD_REG)) is False)

print("Bond probe ok")
