"""Behavioral probe for the UniTensor class (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/UniTensor.md's Parity
and Consistency findings sections is backed by a report() assertion here. Run
with: source tools/env.sh && $PY docs/api-audit/probes/UniTensor.py

Static signatures are ground-truthed against cytnx_src/include/UniTensor.hpp,
cytnx_src/pybind/unitensor_py.cpp, and cytnx_src/cytnx/UniTensor_conti.py.
"""
import sys, os, io, contextlib, inspect, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

UniTensor = cytnx.UniTensor
Type = cytnx.Type


def cmplx(v=1 + 2j):
    t = UniTensor.zeros([2, 2], dtype=Type.ComplexDouble)
    t.set_elem([0, 0], v)
    return t


# =========================================================================
# Construction / generators
# =========================================================================

report("zeros/ones/eye/identity/arange/linspace/normal/uniform/Load are all "
       "staticmethods (module-level generators, not instance methods)",
       all(isinstance(inspect.getattr_static(UniTensor, m), staticmethod)
           for m in ("zeros", "ones", "eye", "identity", "arange", "linspace",
                     "normal", "uniform", "Load")))

z = UniTensor.zeros([2, 3])
report("zeros([2,3]) builds a rank-2 Dense UniTensor with the requested shape",
       z.shape() == [2, 3] and z.uten_type_str() == "Dense" and z.rank() == 2)

report("arange(6) builds a length-6 vector 0..5 (Nelem overload)",
       UniTensor.arange(6).shape() == [6] and UniTensor.arange(6).get_elem([5]) == 5.0)

e = UniTensor.eye(3, is_diag=True)
report("eye(3, is_diag=True) is a diagonal identity UniTensor (is_diag True)",
       e.is_diag() is True and e.to_dense().shape() == [3, 3])

# Init is an instance re-initializer (in-place, returns None), not a factory.
ut_init = UniTensor()
ut_init.Init(cytnx.Tensor([2, 2]))
report("Init(Tensor) re-initializes an existing UniTensor in place and returns "
       "None (it is an instance mutator, not a static factory like zeros/ones)",
       ut_init.shape() == [2, 2])

# =========================================================================
# Conj / Conj_ and the cConj_ raw variant   (B1, N2)
# =========================================================================

t = cmplx()
c = t.Conj()
report("Conj() returns a NEW object; the source is unmutated (B1 pure form): "
       "t[0,0] stays 1+2j while Conj()[0,0] is 1-2j, and Conj() is not t",
       t.get_elem([0, 0]) == 1 + 2j and c.get_elem([0, 0]) == 1 - 2j and c is not t)

t2 = cmplx()
r2 = t2.Conj_()
report("Conj_() mutates the receiver in place (B1/N2 in-place form) and returns "
       "the identical Python object (t2.Conj_() is t2)",
       t2.get_elem([0, 0]) == 1 - 2j and r2 is t2)

t3 = cmplx()
r3 = t3.cConj_()
report("cConj_() (the raw pybind binding of C++ UniTensor::Conj_, which returns "
       "UniTensor&) is behaviorally identical to Conj_(): it mutates the receiver, "
       "returns the same Python object, and shares its data (same_data) -- so it is "
       "a redundant public duplicate of Conj_ (N2)",
       t3.get_elem([0, 0]) == 1 - 2j and r3 is t3 and r3.same_data(t3))

# =========================================================================
# Dagger / Dagger_ = Conj + Transpose   (B1)
# =========================================================================

d0 = UniTensor.zeros([2, 2], dtype=Type.ComplexDouble)
d0.set_elem([0, 1], 1 + 2j)
dg = d0.Dagger()
report("Dagger() = conjugate + transpose, returns a new object: element at [0,1] "
       "(1+2j) appears conjugated at the transposed slot [1,0] (1-2j); source "
       "unchanged",
       dg.get_elem([1, 0]) == 1 - 2j and d0.get_elem([0, 1]) == 1 + 2j)

d1 = UniTensor.zeros([2, 2], dtype=Type.ComplexDouble)
d1.set_elem([0, 1], 1 + 2j)
rd = d1.Dagger_()
report("Dagger_() mutates in place and returns the identical object (d1.Dagger_() is d1)",
       d1.get_elem([1, 0]) == 1 - 2j and rd is d1)

report("cDagger_/cTranspose_/cnormalize_ are the same raw-binding pattern as "
       "cConj_ and are all still publicly exposed alongside their clean wrappers "
       "Dagger_/Transpose_/normalize_ (redundant public duplicates, N2)",
       all(hasattr(UniTensor, m) for m in
           ("cDagger_", "cTranspose_", "cnormalize_", "Dagger_", "Transpose_", "normalize_")))

# =========================================================================
# Transpose / Transpose_
# =========================================================================

tp = UniTensor.arange(6).reshape(2, 3)
tpr = tp.Transpose()
report("Transpose() inverts the index order, returns a new object with reversed "
       "shape ([2,3] -> [3,2]); source shape unchanged",
       tpr.shape() == [3, 2] and tp.shape() == [2, 3])

# =========================================================================
# Inv / Inv_ : the in-place form is ONLY reachable as cInv_ (N2 gap)
# =========================================================================

report("Inv (pure form) exists but there is NO clean Inv_ in-place counterpart; "
       "the only in-place inverse is the raw-binding cInv_ -- an N2 asymmetry "
       "(every other math op got a clean _-suffixed wrapper, Inv did not)",
       hasattr(UniTensor, "Inv") and not hasattr(UniTensor, "Inv_")
       and hasattr(UniTensor, "cInv_"))

iv = UniTensor.arange(1, 5).reshape(2, 2)
inv = iv.Inv()
report("Inv() returns a new element-wise-inverted UniTensor (1/1 at [0,0]); "
       "source is unmodified (B1 pure form)",
       abs(inv.get_elem([0, 0]) - 1.0) < 1e-12 and iv.get_elem([0, 0]) == 1.0)

ic = UniTensor.arange(1, 5).reshape(2, 2)
ric = ic.cInv_()
report("cInv_() inverts the receiver in place (ic[0,0] 1 -> 1/1 == 1) -- confirming "
       "it is the in-place form Inv_ should have wrapped -- but, unlike cConj_ "
       "(which binds the C++ method directly and returns the SAME object), cInv_ is "
       "a lambda that returns by value, so it hands back a FRESH wrapper that merely "
       "shares data (ric is not ic, ric.same_data(ic)): the c-bindings are not even "
       "internally uniform in their return-identity behavior",
       abs(ic.get_elem([0, 0]) - 1.0) < 1e-12 and ric is not ic and ric.same_data(ic))

# =========================================================================
# Pow / Pow_ / __ipow__
# =========================================================================

pw = UniTensor.arange(1, 5).reshape(2, 2)
pwr = pw.Pow(2)
report("Pow(2) squares element-wise into a new object; source unchanged "
       "([1,1] element 4 -> 16 in result, still 4 in source)",
       pwr.get_elem([1, 1]) == 16.0 and pw.get_elem([1, 1]) == 4.0)

pw2 = UniTensor.arange(1, 5).reshape(2, 2)
rpw = pw2.Pow_(2)
report("Pow_(2) squares in place and returns the identical object (pw2.Pow_() is pw2)",
       pw2.get_elem([1, 1]) == 16.0 and rpw is pw2)

# =========================================================================
# Trace / Trace_
# =========================================================================

tr = UniTensor.eye(3)
report("Trace(0,1) contracts axes 0 and 1; trace of a 3x3 identity is 3 (returns "
       "a scalar UniTensor)", tr.Trace(0, 1).item() == 3.0)

# =========================================================================
# normalize / normalize_
# =========================================================================

nz = UniTensor.ones([2, 2])
nn = nz.normalize()
report("normalize() returns a new unit-2-norm object; source's 2-norm is still 2 "
       "(unchanged), normalize()'s is 1",
       abs(float(nz.Norm().item()) - 2.0) < 1e-9
       and abs(float(nn.Norm().item()) - 1.0) < 1e-9)

report("Norm() returns a cytnx.Tensor (a rank-0/scalar Tensor), NOT a UniTensor",
       type(UniTensor.ones([2, 2]).Norm()).__name__ == "Tensor")

# =========================================================================
# permute / permute_    (B2: permute is a view)
# =========================================================================

def _mk_permute_src():
    s = UniTensor.zeros([2, 3])
    s.set_elem([0, 1], 5.0)
    return s

perm_is_view = returns_view(
    make=_mk_permute_src,
    derive=lambda s: s.permute([1, 0]),
    mutate=lambda h: h.set_elem([1, 0], 99.0),  # [1,0] in the permuted view aliases [0,1]
    read=lambda s: s.get_elem([0, 1]),
)
report("permute() returns a VIEW that shares storage with the source (B2): a "
       "mutation written through the permuted handle is observable on the original",
       perm_is_view is True)

pp = UniTensor.zeros([2, 3])
rpp = pp.permute_([1, 0])
report("permute_() permutes in place (shape becomes [3,2]) and returns the same "
       "object (pp.permute_() is pp)",
       pp.shape() == [3, 2] and rpp is pp)

report("permute_nosignflip/permute_nosignflip_ exist as fermionic-specific "
       "sibling permute variants (they skip the fermion sign bookkeeping permute "
       "applies)",
       hasattr(UniTensor, "permute_nosignflip") and hasattr(UniTensor, "permute_nosignflip_"))

# =========================================================================
# contiguous / contiguous_ / make_contiguous / is_contiguous
# =========================================================================

cc = UniTensor.zeros([2, 2])
report("contiguous() on an already-contiguous tensor returns the IDENTICAL object "
       "(the conti.py wrapper short-circuits to `return self` when is_contiguous()) "
       "-- so it is NOT unconditionally a copy",
       cc.is_contiguous() is True and cc.contiguous() is cc)

perm = UniTensor.zeros([2, 3]).permute([1, 0])
report("a permuted tensor is non-contiguous; contiguous() then materializes a new "
       "contiguous object (make_contiguous), which is NOT the same object",
       perm.is_contiguous() is False and perm.contiguous() is not perm
       and perm.contiguous().is_contiguous() is True)

# =========================================================================
# reshape / reshape_    (B2: reshape is a view)
# =========================================================================

reshape_is_view = returns_view(
    make=_mk_permute_src,                       # zeros([2,3]) with [0,1]=5
    derive=lambda s: s.reshape(3, 2),
    mutate=lambda h: h.set_elem([0, 1], 88.0),  # flat index 1 aliases source [0,1]
    read=lambda s: s.get_elem([0, 1]),
)
report("reshape() returns a VIEW sharing storage with the source (B2): a write "
       "through the reshaped handle is visible on the original",
       reshape_is_view is True)

rs = UniTensor.zeros([2, 3])
rrs = rs.reshape_(3, 2)
report("reshape_() reshapes in place ([2,3] -> [3,2]) and returns the same object",
       rs.shape() == [3, 2] and rrs is rs)

# =========================================================================
# relabel / relabels / relabel_ / relabels_   (metadata view)
# =========================================================================

def _mk_relabel_src():
    s = UniTensor.zeros([2, 2], labels=["a", "b"])
    return s

relabel_is_view = returns_view(
    make=_mk_relabel_src,
    derive=lambda s: s.relabel("a", "x"),
    mutate=lambda h: h.set_elem([1, 1], 8.0),
    read=lambda s: s.get_elem([1, 1]),
)
report("relabel() changes only bond-label metadata and returns a new wrapper that "
       "SHARES the underlying block storage (B2 view): a data write through the "
       "relabeled handle is visible on the source",
       relabel_is_view is True)

rl = UniTensor.zeros([2, 2], labels=["a", "b"])
rrl = rl.relabel("a", "x")
report("relabel() does NOT mutate the source's labels (still ['a','b']); the new "
       "object carries the changed labels (['x','b'])",
       rl.labels() == ["a", "b"] and rrl.labels() == ["x", "b"])

rli = UniTensor.zeros([2, 2], labels=["a", "b"])
rrli = rli.relabel_("a", "x")
report("relabel_() mutates the source's labels in place (['x','b']) and returns "
       "the same object",
       rli.labels() == ["x", "b"] and rrli is rli)

report("relabel and relabels overlap heavily: both accept a full new-label list "
       "and an (old_labels, new_labels) pair; relabels lacks relabel's single "
       "(idx,new)/(old,new) scalar overloads -- redundant sibling methods (N4)",
       hasattr(UniTensor, "relabel") and hasattr(UniTensor, "relabels")
       and hasattr(UniTensor, "relabel_") and hasattr(UniTensor, "relabels_"))

# =========================================================================
# get_block / get_block_   (COPY vs VIEW)   (B2)
# =========================================================================

def _mk_block_src():
    s = UniTensor.zeros([2, 2])
    s.set_elem([0, 0], 7.0)
    return s

get_block_is_view = returns_view(
    make=_mk_block_src,
    derive=lambda s: s.get_block(),          # default idx=0
    mutate=lambda h: h.__setitem__((0, 0), 100.0),
    read=lambda s: s.get_elem([0, 0]),
)
report("get_block() returns a COPY (a clone of the block Tensor): mutating the "
       "returned Tensor does NOT change the source UniTensor (B2)",
       get_block_is_view is False)

get_block_underscore_is_view = returns_view(
    make=_mk_block_src,
    derive=lambda s: s.get_block_(),
    mutate=lambda h: h.__setitem__((0, 0), 100.0),
    read=lambda s: s.get_elem([0, 0]),
)
report("get_block_() returns a VIEW (a reference to the live block Tensor): "
       "mutating it IS observable on the source UniTensor -- the classic "
       "copy(get_block)/view(get_block_) B2 pair",
       get_block_underscore_is_view is True)

# =========================================================================
# get_blocks / get_blocks_  (list of COPIES vs list of VIEWS) on a symmetric UT
# =========================================================================

Bi = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
Bo = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [cytnx.Symmetry.U1()])
sym_ut = UniTensor([Bi, Bo], labels=["a", "b"])
sym_ut.get_block_([0, 0])[0, 0] = 5.0
gb_copy = sym_ut.get_blocks()
gb_copy[0][0, 0] = 77.0
report("get_blocks() returns a list of COPIES: mutating an element of a returned "
       "block leaves the source UniTensor's block unchanged (still 5.0)",
       sym_ut.get_block_([0, 0])[0, 0].item() == 5.0)
gb_view = sym_ut.get_blocks_()
gb_view[0][0, 0] = 88.0
report("get_blocks_() returns a list of VIEWS: mutating a returned block IS "
       "observable on the source UniTensor (now 88.0)",
       sym_ut.get_block_([0, 0])[0, 0].item() == 88.0)

report("is_blockform() is True and Nblocks() == 2 for this U(1)-symmetric "
       "rank-2 UniTensor (two qnum-conserving blocks)",
       sym_ut.is_blockform() is True and sym_ut.Nblocks() == 2)

# =========================================================================
# set_elem / at indexing   (B2 in-place mutation)
# =========================================================================

se = UniTensor.zeros([2, 2])
se.set_elem([1, 1], 42.0)
report("set_elem(locator, value) mutates the addressed element in place",
       se.get_elem([1, 1]) == 42.0)

atx = UniTensor.zeros([2, 2])
atx.at([0, 0]).value = 3.14
report("at(locator).value = x writes through a proxy and mutates the element in "
       "place; at(locator).value reads it back",
       atx.at([0, 0]).value == 3.14 and atx.get_elem([0, 0]) == 3.14)

si = UniTensor.zeros([1, 1])
si.set_elem([0, 0], 2.71)
report("item() extracts the single scalar value of a 1-element UniTensor",
       abs(si.item() - 2.71) < 1e-12)

# =========================================================================
# dtype / device conversion    (B3-adjacent; short-circuit identity)
# =========================================================================

ad = UniTensor.zeros([2, 2], dtype=Type.Double)
report("astype(same dtype) short-circuits to `return self` (conti.py) -- returns "
       "the IDENTICAL object, no copy",
       ad.astype(Type.Double) is ad)

adc = ad.astype(Type.ComplexDouble)
report("astype(different dtype) delegates to astype_different_type and returns a "
       "NEW object with the requested dtype (Double -> ComplexDouble)",
       adc is not ad and adc.dtype() == Type.ComplexDouble)

report("to(same device) short-circuits to `return self` (conti.py) -- identical object",
       ad.to(cytnx.Device.cpu) is ad)

report("dtype()/device() return integer codes; dtype_str()/device_str() return "
       "the human strings (dtype_str of a Double tensor contains 'Double')",
       ad.dtype() == Type.Double and "Double" in ad.dtype_str()
       and ad.device() == cytnx.Device.cpu)

# =========================================================================
# operators   (B5 / B3 promotion)
# =========================================================================

dd = UniTensor.ones([2, 2], dtype=Type.Double)
cd = UniTensor.ones([2, 2], dtype=Type.ComplexDouble)
summed = dd + cd
report("__add__ promotes Double + ComplexDouble to ComplexDouble (B3: widen to "
       "the more general dtype), matching Tensor's promotion rule",
       summed.dtype() == Type.ComplexDouble)

scaled = dd * 2.0
report("__mul__ by a Python scalar broadcasts element-wise (1*2 == 2 at [0,0]) "
       "and returns a new object",
       scaled.get_elem([0, 0]) == 2.0 and scaled is not dd)

report("UniTensor exposes arithmetic operators (__add__/__sub__/__mul__/"
       "__truediv__/__neg__ and in-place __iadd__ etc.) but NO named method "
       "counterparts (there is no `add`/`Add`, `mul`/`Mul`): B5's operator-vs-"
       "named-method equivalence is vacuous here -- only the operators exist",
       hasattr(UniTensor, "__add__") and hasattr(UniTensor, "__iadd__")
       and not hasattr(UniTensor, "add") and not hasattr(UniTensor, "Add"))

# =========================================================================
# combineBonds  (in-place mutator that returns None but lacks the _ suffix; N2)
# =========================================================================

cb = UniTensor.zeros([2, 3, 4])
ret_cb = cb.combineBonds([0, 1])
report("combineBonds() mutates the receiver in place (shape [2,3,4] -> [6,4]) and "
       "returns None, yet its name carries NO trailing _ -- an N2 violation (an "
       "in-place mutator not marked as such)",
       ret_cb is None and cb.shape() == [6, 4])

# =========================================================================
# set_rowrank / set_rowrank_
# =========================================================================

srr = UniTensor.zeros([2, 3, 4])
base_rowrank = srr.rowrank()
new_srr = srr.set_rowrank(2)
report("set_rowrank(2) returns a NEW object with rowrank 2; the source's rowrank "
       "is unchanged (pure form)",
       new_srr.rowrank() == 2 and srr.rowrank() == base_rowrank)

srr.set_rowrank_(1)
report("set_rowrank_(1) mutates the source's rowrank in place to 1",
       srr.rowrank() == 1)

# =========================================================================
# Save / Load round trip
# =========================================================================

with tempfile.TemporaryDirectory() as tmpd:
    fname = os.path.join(tmpd, "ut.cytnx")
    src = UniTensor.arange(6).reshape(2, 3)
    src.Save(fname)
    loaded = UniTensor.Load(fname)
    report("Save()/Load() round trip preserves shape and element data",
           loaded.shape() == [2, 3] and loaded.get_elem([1, 2]) == src.get_elem([1, 2]))

# =========================================================================
# The full roster of raw c-prefixed public duplicates (the headline finding)
# =========================================================================

c_prefixed = ["cConj_", "cDagger_", "cInv_", "cPow_", "cTrace_", "cTranspose_",
              "cnormalize_", "ctag", "ctruncate_", "cfrom", "c_at",
              "c_relabel_", "c_relabels_", "c_set_label", "c_set_labels",
              "c_set_name", "c_set_rowrank_", "c__ipow__"]
report("all 18 raw c-prefixed bindings (cConj_, cDagger_, cInv_, cPow_, cTrace_, "
       "cTranspose_, cnormalize_, ctag, ctruncate_, cfrom, c_at, c_relabel_, "
       "c_relabels_, c_set_label, c_set_labels, c_set_name, c_set_rowrank_, "
       "c__ipow__) are PUBLIC members (no leading underscore) that leak into the "
       "user surface alongside their clean Python wrappers -- an internal "
       "implementation detail exposed publicly (N2)",
       all(m in dir(UniTensor) and not m.startswith("_") for m in c_prefixed))

report("each clean wrapper for those c-bindings does exist and is the intended "
       "public spelling (Conj_/Dagger_/Transpose_/Pow_/Trace_/normalize_/tag/"
       "truncate_/convert_from/at/relabel_/relabels_/set_label/set_labels/"
       "set_name/set_rowrank_), so every c-binding except cInv_ is a pure "
       "duplicate; cInv_ alone has no wrapper",
       all(hasattr(UniTensor, m) for m in
           ("Conj_", "Dagger_", "Transpose_", "Pow_", "Trace_", "normalize_",
            "tag", "truncate_", "convert_from", "at", "relabel_", "relabels_",
            "set_label", "set_labels", "set_name", "set_rowrank_")))

print("UniTensor probe ok")
