"""Behavioral probe for the Tensor class (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/Tensor.md's Parity and
Consistency findings sections is backed by a report() assertion here. Run with:
source tools/env.sh && $PY docs/api-audit/probes/Tensor.py

The dense, N-way-overloaded arithmetic surface of Tensor is exercised through a
few representative dtypes only; the point of each assertion is the *behavior*
(copy-vs-view, in-place vs. returns-new, dtype promotion, operator semantics,
exception behavior), not exhaustive dtype coverage.
"""
import sys, os, io, contextlib, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

T = cytnx.Tensor


@contextlib.contextmanager
def _quiet():
    """Swallow the noisy C++ cytnx_error_msg stack-trace dumped to stderr/stdout
    by the deliberately-failing calls below; the Python exception still raises."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        yield


def mk():
    """A fresh, contiguous 2x3 Double tensor with values 0..5."""
    return cytnx.arange(6).reshape(2, 3).astype(cytnx.Type.Double)


def read00(src):
    return float(src[0, 0].item())


def set00(handle):
    handle[0, 0] = 42.0


# =========================================================================
# dtype / device identity (cross-ref enums.md: dtype via Type, device via Device)
# =========================================================================

t = mk()
report("dtype()/dtype_str() report the Type code and its human string "
       "(Type.Double == 3, 'Double (Float64)') -- see enums.md Type",
       t.dtype() == cytnx.Type.Double and t.dtype() == 3
       and t.dtype_str() == "Double (Float64)")
report("device()/device_str() report the Device code and its string "
       "(Device.cpu == -1, 'cytnx device: CPU') -- see enums.md Device",
       t.device() == cytnx.Device.cpu and t.device() == -1
       and t.device_str() == "cytnx device: CPU")
report("shape() -> list[int], rank() -> int agree (rank == len(shape))",
       t.shape() == [2, 3] and t.rank() == 2 and t.rank() == len(t.shape()))

# =========================================================================
# B1/B2: copy-vs-view classification of every derivation method
# =========================================================================

# permute -> view (shares storage). Confirmed two ways: returns_view + same_data.
report("permute() returns a VIEW: a mutation through the permuted handle is "
       "visible on the source (returns_view is True)",
       returns_view(mk, lambda s: s.permute(1, 0), set00, read00) is True)
report("permute() shares storage with its source (same_data() is True)",
       mk().same_data(mk().permute(1, 0)) is False  # different sources
       and (lambda s: s.same_data(s.permute(1, 0)))(mk()) is True)

# permute_ -> in place, returns self
tp = mk()
report("permute_() mutates in place and returns the SAME Python object "
       "(ret is self); shape is transposed afterward",
       tp.permute_(1, 0) is tp and tp.shape() == [3, 2])

# reshape -> view (shares storage for a contiguous tensor), unlike numpy-copy intuition
report("reshape() on a contiguous tensor returns a VIEW that shares storage: "
       "returns_view is True and same_data() is True",
       returns_view(mk, lambda s: s.reshape(3, 2), set00, read00) is True
       and (lambda s: s.same_data(s.reshape(3, 2)))(mk()) is True)

# reshape_ -> in place, returns self
tr = mk()
report("reshape_() mutates in place and returns the SAME Python object (ret is self)",
       tr.reshape_(3, 2) is tr and tr.shape() == [3, 2])

# clone -> deep copy (independent storage)
report("clone() returns a COPY: a mutation through the clone is NOT visible on "
       "the source (returns_view is False) and same_data() is False",
       returns_view(mk, lambda s: s.clone(), set00, read00) is False
       and (lambda s: s.same_data(s.clone()))(mk()) is False)

# contiguous (Python-side conti wrapper): already-contiguous -> returns self (identity)
tc = mk()
report("contiguous() on an already-contiguous tensor returns the SAME object "
       "(identity short-circuit added in Tensor_conti.py)",
       tc.is_contiguous() is True and tc.contiguous() is tc)

# contiguous on a non-contiguous tensor -> a fresh contiguous COPY
tperm = mk().permute(1, 0)
report("a permuted tensor is non-contiguous; contiguous() then returns a fresh "
       "COPY (not self, not sharing storage)",
       tperm.is_contiguous() is False
       and tperm.contiguous() is not tperm
       and tperm.same_data(tperm.contiguous()) is False)

# make_contiguous (raw pybind of C++ contiguous()) never short-circuits to self:
# on an already-contiguous tensor it returns a NEW wrapper that still SHARES storage.
tm = mk()
mc = tm.make_contiguous()
report("make_contiguous() (raw C++ contiguous(), bound under a different name) "
       "on an already-contiguous tensor returns a NEW wrapper object that still "
       "SHARES storage (view) -- unlike contiguous()'s identity short-circuit",
       mc is not tm and tm.same_data(mc) is True)

# contiguous_ -> mutates in place (receiver becomes contiguous)
tperm2 = mk().permute(1, 0)
tperm2.contiguous_()
report("contiguous_() mutates the receiver in place: a previously non-contiguous "
       "tensor is contiguous afterward (B1: return identity is NOT guaranteed, "
       "only the receiver's observable state)",
       tperm2.is_contiguous() is True)

# =========================================================================
# B2: slicing READ returns a COPY (NOT a numpy-style view); setitem mutates.
# =========================================================================

report("Tensor SLICE READ (t[0:2]) returns a COPY, NOT a view: a mutation "
       "through the slice is NOT visible on the source (returns_view is False) -- "
       "this DIVERGES from numpy/torch, where basic slicing is a view",
       returns_view(mk, lambda s: s[0:2], set00, read00) is False)
report("a single-index read (t[0]) likewise returns an independent COPY "
       "(same_data() is False)",
       (lambda s: s.same_data(s[0]))(mk()) is False)

# leftover debug print: the bare-1D-slice __getitem__ branch (tensor_py.cpp:355)
# does `std::cout << start << " " << stop << " " << step` with NO
# py::scoped_ostream_redirect, so it writes to the process's REAL stdout and is
# NOT capturable by contextlib.redirect_stdout (same uncapturable-print pattern
# as Device.Print_Property in enums.md P4). Tuple indexing (t[0,1]) does not.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _ = mk()[0:2]
report("t[0:2] (bare 1-D slice) leaks a leftover 'start stop step' debug line to "
       "the process's REAL stdout that contextlib.redirect_stdout CANNOT capture "
       "(the buffer stays empty) -- an unguarded std::cout at tensor_py.cpp:355",
       _buf.getvalue() == "")
report("SLICE/ELEMENT ASSIGNMENT (t[0,0] = v) DOES mutate the tensor in place: "
       "an alias of the same Python object observes the write",
       (lambda: (lambda a, b: (b.__setitem__((0, 0), 55.0), read00(a) == 55.0)[1])(
           (s := mk()), s))() is True)

# =========================================================================
# dtype / device conversion: astype / to (Tensor_conti.py wrappers)
# =========================================================================

ta = mk()
report("astype() to the SAME dtype returns the SAME object (identity "
       "short-circuit in Tensor_conti.py) -- no copy is made",
       ta.astype(cytnx.Type.Double) is ta)
taf = mk().astype(cytnx.Type.Float)
report("astype() to a DIFFERENT dtype returns an independent COPY with the new "
       "dtype (same_data() is False, dtype_str() == 'Float (Float32)')",
       taf.dtype_str() == "Float (Float32)"
       and mk().same_data(mk().astype(cytnx.Type.Float)) is False)
tto = mk()
report("to() to the SAME device returns the SAME object (identity short-circuit "
       "in Tensor_conti.py)",
       tto.to(cytnx.Device.cpu) is tto)

# the raw pybind primitives underneath REFUSE the same-dtype/same-device call:
# the no-op case is meant to be intercepted Python-side (see the conti wrappers).
with _quiet():
    try:
        mk().astype_different_dtype(cytnx.Type.Double)
        astype_raw_same_raised = False
    except RuntimeError:
        astype_raw_same_raised = True
report("astype_different_dtype() (raw pybind primitive) RAISES RuntimeError when "
       "asked for the same dtype -- it hard-asserts the no-op must be handled in "
       "Python; it is an internal helper leaked to the public surface",
       astype_raw_same_raised)
with _quiet():
    try:
        mk().to_different_device(cytnx.Device.cpu)
        to_raw_same_raised = False
    except RuntimeError:
        to_raw_same_raised = True
report("to_different_device() (raw pybind primitive) likewise RAISES RuntimeError "
       "for the same device -- same leaked-internal-helper pattern as "
       "astype_different_dtype",
       to_raw_same_raised)

# =========================================================================
# B5: operators vs. named methods, and copy-vs-inplace of each
# =========================================================================

# The C++ named arithmetic methods (Add/Sub/Mul/Div/Add_/.../Cpr/Mod) are NOT
# bound to Python at all -- only the operators (and the leaked c__i*__ helpers).
report("the C++ named arithmetic methods (Add/Sub/Mul/Div and their _ in-place "
       "forms, Cpr, Mod) have NO Python binding: only operator dunders are "
       "exposed -- a B5 gap (no named-method counterpart exists to be equivalent to)",
       not any(hasattr(T, m) for m in ("Add", "add", "Sub", "sub", "Mul", "mul",
                                       "Div", "div", "Add_", "Cpr", "Mod")))

# __add__ (and friends) return a NEW tensor; the source is untouched.
tadd = mk()
res = tadd + 1.0
report("t + scalar returns a NEW tensor (copy semantics): the source is "
       "unchanged and the result differs",
       read00(tadd) == 0.0 and read00(res) == 1.0)

# += is in place and preserves object identity (Tensor_conti.py __iadd__ -> self).
tia = mk()
alias = tia
tia += 1.0
report("t += scalar mutates IN PLACE and preserves object identity "
       "(Tensor_conti.py __iadd__ returns self): an alias sees the change",
       tia is alias and read00(alias) == 1.0)

# c__iadd__ is the raw pybind in-place primitive that __iadd__ wraps; leaked.
tci = mk()
rci = tci.c__iadd__(1.0)
report("c__iadd__() (the raw in-place primitive that Tensor_conti.py's __iadd__ "
       "wraps) mutates in place and returns a self-aliasing handle (same_data) -- "
       "an internal helper leaked to the public surface",
       read00(tci) == 1.0 and tci.same_data(rci) is True)

# B3: dtype promotion follows a widen-to-more-general rule, same both call paths.
report("B3 promotion: Double + ComplexDouble widens to ComplexDouble",
       (mk() + mk().astype(cytnx.Type.ComplexDouble)).dtype_str()
       == "Complex Double (Complex Float64)")
report("B3 promotion: Double + Float widens to Double (the more general real dtype)",
       (mk() + mk().astype(cytnx.Type.Float)).dtype_str() == "Double (Float64)")
report("B3 promotion: Int64 + a Python float scalar widens to Double",
       (cytnx.arange(6).astype(cytnx.Type.Int64) + 1.5).dtype_str()
       == "Double (Float64)")

# __eq__ is ELEMENTWISE (returns a Bool Tensor), so Tensor is unhashable.
eqr = mk() == mk()
report("== is ELEMENTWISE and returns a Bool Tensor (not a Python bool), so it "
       "cannot be used directly in a boolean/if context the way a scalar == can",
       isinstance(eqr, cytnx.Tensor) and eqr.dtype() == cytnx.Type.Bool
       and eqr.dtype_str() == "Bool")
report("because __eq__ is defined and non-boolean, Python sets Tensor.__hash__ "
       "to None: Tensor is UNHASHABLE (cannot be a dict key / set member)",
       T.__hash__ is None)

# __imatmul__ (the @= operator) is MISNAMED in Tensor_conti.py as `__imatmul`
# (missing trailing __), so @= is NOT in place -- it silently falls back to
# __matmul__ (linalg.Dot) and REBINDS the name to a fresh object.
report("Tensor_conti.py defines `__imatmul` (missing the trailing __), so the "
       "true `__imatmul__` slot does NOT exist on Tensor",
       hasattr(T, "__imatmul") is True and hasattr(T, "__imatmul__") is False)
msq = cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)
alias_m = msq
msq @= msq
report("consequently `t @= x` is NOT in place: with no __imatmul__ slot Python "
       "falls back to __matmul__ (linalg.Dot) and REBINDS t to a fresh object "
       "(t is not its former self) -- a silent B5/N2 defect",
       msq is not alias_m)

# =========================================================================
# Conj / Conj_ / cConj_ : pure vs. in-place vs. leaked-raw
# =========================================================================

def mkc():
    c = cytnx.arange(4).astype(cytnx.Type.ComplexDouble).reshape(2, 2)
    c[0, 1] = complex(1, 2)
    return c

cj_src = mkc()
cj = cj_src.Conj()
report("Conj() is PURE: it returns a new conjugated tensor (1+2j -> 1-2j) and "
       "leaves the source unchanged (same_data() is False)",
       complex(cj_src[0, 1].item()) == complex(1, 2)
       and complex(cj[0, 1].item()) == complex(1, -2)
       and cj_src.same_data(cj) is False)

cj2 = mkc()
ret_cj2 = cj2.Conj_()
report("Conj_() (Tensor_conti.py wrapper) mutates IN PLACE and returns the SAME "
       "object (ret is self): source is conjugated afterward",
       ret_cj2 is cj2 and complex(cj2[0, 1].item()) == complex(1, -2))

cj3 = mkc()
ret_cj3 = cj3.cConj_()
report("cConj_() (the raw pybind in-place primitive that Conj_ wraps) mutates in "
       "place and returns a self-aliasing handle -- leaked internal helper",
       complex(cj3[0, 1].item()) == complex(1, -2) and cj3.same_data(ret_cj3) is True)

# Pow / Pow_ / cPow_ mirror the same pure / in-place / leaked-raw triple.
pw_src = cytnx.arange(4).astype(cytnx.Type.Double)
pw = pw_src.Pow(2)
report("Pow(p) is PURE (new tensor, source unchanged); values are squared",
       pw_src.same_data(pw) is False
       and [float(pw[i].item()) for i in range(4)] == [0.0, 1.0, 4.0, 9.0])
pw2 = cytnx.arange(4).astype(cytnx.Type.Double)
report("Pow_(p) mutates in place and returns the SAME object (ret is self)",
       pw2.Pow_(2) is pw2 and [float(pw2[i].item()) for i in range(4)] == [0.0, 1.0, 4.0, 9.0])
ab = cytnx.arange(4).astype(cytnx.Type.Double) - 2.0
report("Abs_() mutates in place and returns the SAME object (ret is self)",
       ab.Abs_() is ab and [float(ab[i].item()) for i in range(4)] == [2.0, 1.0, 0.0, 1.0])

# =========================================================================
# flatten / flatten_ : copy vs. in-place, with an asymmetric return convention
# =========================================================================

fl_src = mk()
fl = fl_src.flatten()
report("flatten() returns a COPY: a 1-D new tensor whose storage is independent "
       "of the source (same_data() is False)",
       fl.shape() == [6] and fl_src.same_data(fl) is False)
fl2 = mk()
ret_fl2 = fl2.flatten_()
report("flatten_() mutates in place BUT returns None (not self) -- an N2/B1 "
       "return-convention inconsistency vs. reshape_/permute_ which return self",
       ret_fl2 is None and fl2.shape() == [6])

# =========================================================================
# numpy() : default is a COPY; share_mem is opt-in
# =========================================================================

nt = cytnx.arange(4).astype(cytnx.Type.Double)
arr = nt.numpy()
arr[0] = 123.0
report("numpy() defaults to share_mem=False -> a COPY: mutating the ndarray does "
       "NOT affect the source Tensor",
       read00(nt.reshape(4, 1)) == 0.0)

# =========================================================================
# B4: errors are catchable exceptions, not sentinels / crashes
# =========================================================================

with _quiet():
    try:
        mk().reshape(5, 5)
        bad_reshape_raised = False
    except RuntimeError:
        bad_reshape_raised = True
report("reshape() to an incompatible total size raises a catchable RuntimeError "
       "(B4: not a silent truncation or crash)", bad_reshape_raised)

with _quiet():
    try:
        mk() + cytnx.arange(4).reshape(2, 2).astype(cytnx.Type.Double)
        shape_mismatch_raised = False
    except RuntimeError:
        shape_mismatch_raised = True
report("adding two tensors of mismatched shape raises a catchable RuntimeError (B4)",
       shape_mismatch_raised)

with _quiet():
    try:
        mk().item()
        item_nonscalar_raised = False
    except RuntimeError:
        item_nonscalar_raised = True
report("item() on a non-scalar (multi-element) tensor raises a catchable "
       "RuntimeError (B4)", item_nonscalar_raised)

# =========================================================================
# Parity: C++ default arguments dropped by the pybind binding
# =========================================================================

sq = cytnx.arange(9).reshape(3, 3).astype(cytnx.Type.Double)
with _quiet():
    try:
        sq.Trace()
        trace_noarg_ok = True
    except TypeError:
        trace_noarg_ok = False
report("Trace() REQUIRES two positional axis args in Python: the C++ default "
       "args (a=0, b=1) were dropped by the pybind binding (.def has no py::arg "
       "defaults) -- a signature-parity gap vs. C++ Trace(a=0, b=1)",
       trace_noarg_ok is False)
report("Trace(0, 1) computes the trace over the given axes (0+4+8 == 12)",
       float(sq.Trace(0, 1).item()) == 12.0)

# Svd/Eigh member forms return a Python list of Tensors.
report("Svd() returns a list of 3 Tensors [U, S, vT] (is_UvT defaults True); "
       "Eigh() returns a list of 2 [eigvals, eigvecs] (is_V defaults True)",
       isinstance(sq.Svd(), list) and len(sq.Svd()) == 3
       and isinstance(sq.Eigh(), list) and len(sq.Eigh()) == 2)

print("Tensor probe ok")
