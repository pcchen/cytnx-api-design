"""Behavioral probe for UniTensor category 06 — element & block access,
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/UniTensor/06-element-block-access.md is
backed by a report(...) assertion here. The raw-C++ side of the
binding-fidelity findings is verified by probes/cpp/UniTensor_06_element.cpp.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_06_element.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

UT = cytnx.UniTensor

# The 11 element dtypes cytnx supports.
ALL_DTYPES = [
    cytnx.Type.ComplexDouble, cytnx.Type.ComplexFloat, cytnx.Type.Double,
    cytnx.Type.Float, cytnx.Type.Int64, cytnx.Type.Uint64, cytnx.Type.Int32,
    cytnx.Type.Uint32, cytnx.Type.Int16, cytnx.Type.Uint16, cytnx.Type.Bool,
]
FLOAT_COMPLEX = {cytnx.Type.ComplexDouble, cytnx.Type.ComplexFloat,
                 cytnx.Type.Double, cytnx.Type.Float}


def mk_scalar(dtype):
    """A fresh 1-element Dense UniTensor of the given dtype (for item/get/set)."""
    return UT.zeros([1, 1], dtype=dtype)


def mk_dense():
    """A fresh rank-2 Dense (bosonic) UniTensor filled 1."""
    return UT.ones([2, 2])


def mk_block():
    """A fresh rank-2 Block (U1-symmetric) UniTensor with two 1x1 blocks."""
    sym = cytnx.Symmetry.U1()
    Bi = cytnx.Bond(cytnx.BD_IN, [[0], [1]], [1, 1], [sym])
    Bo = cytnx.Bond(cytnx.BD_OUT, [[0], [1]], [1, 1], [sym])
    return cytnx.UniTensor([Bi, Bo], labels=["a", "b"])


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# UT-E1 — get_elem binds only the 4 float/complex dtypes, while item/set_elem
#          cover all 11 (binding fidelity: the pybind template coverage differs)
# =========================================================================

# get_elem works on the 4 float/complex dtypes.
for dt in FLOAT_COMPLEX:
    u = mk_scalar(dt)
    v = u.get_elem([0, 0])
    report(f"get_elem returns a value on float/complex dtype {dt.name}",
           v == 0)

# get_elem RAISES on the 7 integer/bool dtypes (the pybind lambda has no branch
# for them and falls through to `[ERROR] try to get element from a void Storage`).
int_bool = [dt for dt in ALL_DTYPES if dt not in FLOAT_COMPLEX]
for dt in int_bool:
    u = mk_scalar(dt)
    raised = False
    try:
        u.get_elem([0, 0])
    except RuntimeError:
        raised = True
    report(f"get_elem RAISES on integer/bool dtype {dt.name} — "
           f"the binding lambda only instantiates the 4 float/complex branches",
           raised)

# item() and set_elem() cover ALL 11 dtypes — including the integer/bool ones
# get_elem rejects. This is the binding-fidelity asymmetry (C++ get_elem<T> is a
# generic template covering all 11; the Python get_elem lambda picks only 4).
for dt in ALL_DTYPES:
    u = mk_scalar(dt)
    report(f"item() reads a value on dtype {dt.name} (all 11 covered)",
           u.item() == 0)
    u.set_elem([0, 0], 1)
    report(f"set_elem writes a value on dtype {dt.name} (all 11 covered)",
           u.item() == 1)

# =========================================================================
# UT-E2 — the C++ `get`/`set` accessor methods are UNBOUND in Python; they are
#          reached only through __getitem__/__setitem__ (binding fidelity)
# =========================================================================

report("the raw C++ accessor methods `get`/`set` are NOT public Python members "
       "— they are reachable only via __getitem__/__setitem__",
       "get" not in dir(UT) and "set" not in dir(UT))
report("__getitem__ and __setitem__ ARE bound (the only public path to C++ get/set)",
       "__getitem__" in dir(UT) and "__setitem__" in dir(UT))

# =========================================================================
# UT-E3 — get_block (copy) vs get_block_ (shared-data view) on a Dense tensor
# =========================================================================

u = mk_dense()
ref = u.get_block_()          # a view onto the tensor's storage
copy = u.get_block()          # a clone
report("get_block returns a COPY: its data is NOT shared with the tensor's "
       "block storage (same_data is False)",
       not ref.same_data(copy))
report("get_block_ returns a shared-data VIEW: its data IS shared with the "
       "tensor's block storage (same_data is True)",
       ref.same_data(u.get_block_()))

# =========================================================================
# UT-E4 — get_blocks (copy) vs get_blocks_ (shared-data view) on a Block tensor.
#          get_blocks(_) ERROR on a Dense tensor (use get_block(_) instead).
# =========================================================================

bt = mk_block()
ref_blocks = bt.get_blocks_()
copies = bt.get_blocks()
views = bt.get_blocks_()
report("get_blocks returns COPIES: none of the returned tensors share data with "
       "the tensor's block storage",
       len(copies) == len(ref_blocks) and
       not any(ref_blocks[i].same_data(copies[i]) for i in range(len(copies))))
report("get_blocks_ returns shared-data VIEWS: every returned tensor shares data "
       "with the tensor's block storage",
       len(views) == len(ref_blocks) and
       all(ref_blocks[i].same_data(views[i]) for i in range(len(views))))

dense_err = False
try:
    mk_dense().get_blocks()
except RuntimeError:
    dense_err = True
report("get_blocks errors on a Dense tensor ('use get_block() instead') — the "
       "plural forms are for Block/BlockFermionic tensors only",
       dense_err)

# =========================================================================
# UT-E5 — the misspelled `slient` kwarg on get_blocks_ warns (FutureWarning) and
#          still forwards; the correctly-spelled `silent` does not warn
# =========================================================================

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = mk_block().get_blocks_(slient=True)
report("the misspelled `slient` kwarg on get_blocks_ emits a FutureWarning "
       "('deprecated ... use silent instead') yet still forwards the call",
       any(issubclass(x.category, FutureWarning) and "slient" in str(x.message)
           for x in w))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mk_block().get_blocks_(silent=True)
report("the correctly-spelled `silent` kwarg on get_blocks_ emits NO warning",
       len(w) == 0)

# =========================================================================
# UT-E6 — put_block(..., force=...) / put_block_(..., force=...) are DEPRECATED
#          (FutureWarning); the force-free overloads are the current form
# =========================================================================

bt = mk_block()
blk = bt.get_block([0, 0])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bt.put_block(blk, [0, 0], True)          # positional force -> deprecated overload
report("put_block(in, qidx, force) emits a FutureWarning — the `force` argument "
       "is deprecated ('use put_block(in, qnum) without force instead')",
       any(issubclass(x.category, FutureWarning) and "force" in str(x.message)
           for x in w))

bt = mk_block()
blk = bt.get_block([0, 0])
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    bt.put_block_(blk, [0, 0], True)
report("put_block_(in, qidx, force) likewise emits a FutureWarning for the "
       "deprecated `force` argument",
       any(issubclass(x.category, FutureWarning) and "force" in str(x.message)
           for x in w))

# =========================================================================
# UT-E7 — put_block (copy-in) vs put_block_ (view-in): a correct N-underscore
#          pair. Both return None (they mutate the receiver in place).
# =========================================================================

u = mk_dense()
t = cytnx.ones([2, 2])
ret = u.put_block(t)
report("put_block COPIES the input tensor into the block (the receiver's block "
       "does NOT share data with the input) and returns None",
       ret is None and not u.get_block_().same_data(t))

u = mk_dense()
t = cytnx.ones([2, 2])
ret = u.put_block_(t)
report("put_block_ makes the input tensor a shared-data VIEW of the block (the "
       "receiver's block SHARES data with the input) and returns None",
       ret is None and u.get_block_().same_data(t))

# =========================================================================
# UT-E8 — __getitem__/__setitem__ slice a Dense tensor; they ERROR on a
#          Block/BlockFermionic tensor (use at()/get_block() instead)
# =========================================================================

u = UT.arange(6).reshape(2, 3)
sl = u[0, :]
report("__getitem__ slices a Dense tensor, returning a UniTensor sub-block "
       "(u[0, :] has shape [3])",
       isinstance(sl, UT) and sl.shape() == [3])

u = UT.arange(6).reshape(2, 3)
u[0, 0] = 9.0
report("__setitem__ assigns into a Dense tensor (u[0,0] = 9.0 sets the element)",
       u.get_elem([0, 0]) == 9.0)

block_getitem_err = False
try:
    mk_block()[0, 0]
except RuntimeError:
    block_getitem_err = True
report("__getitem__ ERRORS on a Block tensor ('Cannot get element using [] from "
       "Block/BlockFermionicUniTensor. Use at() instead.')",
       block_getitem_err)

# =========================================================================
# UT-E9 — at()/item() element access. at() returns a proxy usable for read AND
#          write; it works on BOTH Dense and Block tensors (unlike []).
# =========================================================================

u = mk_dense()
h = u.at([0, 0])
report("at([0,0]) on a Dense tensor returns a proxy whose .value reads the "
       "element",
       h.value == 1.0)
h.value = 3.0
report("the at() proxy is writable: setting proxy.value mutates the element in "
       "place",
       u.get_elem([0, 0]) == 3.0)

bt = mk_block()
hb = bt.at([0, 0])
report("at([0,0]) works on a Block tensor too (where [] is rejected): the proxy "
       "reports whether the element exists",
       hb.exists())

# =========================================================================
# UT-E10 — elem_exists is a Block-only predicate: it reports whether a symmetry
#           block-element exists, and ERRORS on a Dense tensor
# =========================================================================

bt = mk_block()
report("elem_exists reports True for an allowed block-element [0,0] and False "
       "for a symmetry-forbidden one [0,1]",
       bt.elem_exists([0, 0]) and not bt.elem_exists([0, 1]))

dense_ee_err = False
try:
    mk_dense().elem_exists([0, 0])
except RuntimeError:
    dense_ee_err = True
report("elem_exists ERRORS on a Dense tensor ('can only be used on UniTensor "
       "with Symmetry') — it is a Block-only predicate",
       dense_ee_err)

print("UniTensor 06 probe ok")
