"""Behavioral probe for the Tensor audit, category 07 (dtype / device
conversion), verified against the installed cytnx==1.1.0 wheel (NOT
source-inferred).

Every runtime claim in docs/api-audit/Tensor/07-type-device-conversion.md is
backed by a report(...) assertion here. Members covered: the type/device
converters `astype` / `to` / `to_`, the deep-copy `clone`, and the two leaked
raw plumbing shims `astype_different_dtype` / `to_different_device`.

Binding-fidelity headline (T-T1, cross-ref UniTensor UT-T1): `astype`/`to`
SHORT-CIRCUIT to `is self` on a no-op conversion — a Python identity introduced
by the conti.py wrappers (cytnx/Tensor_conti.py:36-48: `if self.dtype()==dtype:
return self` / `if self.device()==device: return self`). Raw C++
`astype`/`to` return a FRESH object every call; that raw-C++ side is verified by
probes/cpp/Tensor_07_typedevice.cpp.

Two of the brief's assumptions are CORRECTED here against runtime truth (cat 04
taught us the brief can be wrong):
  * `to_` returns **None**, NOT self — C++ `void to_(const int&)` (Tensor.hpp:683)
    is bound directly (tensor_py.cpp:177), so unlike UniTensor's `to_` (which
    returns `UniTensor&`) Tensor's returns None. An in-place-return asymmetry
    (T-T4), sibling to the cat-04 `flatten_`-returns-None finding.
  * `to_`'s parameter is NAMED `device`, NOT erased — it is bound WITH
    `py::arg("device")` (tensor_py.cpp:177), so `to_(device=...)` works and
    `to_(arg0=...)` fails. This is the POSITIVE contrast to UniTensor's UT-T3
    (whose `to_` param was erased to `arg0`).

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_07_typedevice.py
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

Tensor = cytnx.Tensor
Type = cytnx.Type
Device = cytnx.Device

report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")


def _mat():
    """A fresh 2x2 Double Tensor of ones."""
    return cytnx.ones((2, 2), Type.Double)


# =========================================================================
# T-T1: astype / to SHORT-CIRCUIT to `is self` on a no-op conversion.
# This Python identity is BINDING-INTRODUCED by cytnx/Tensor_conti.py:36-48
# (`if self.dtype()==dtype: return self` / `if self.device()==device: return
# self`). Raw C++ astype/to return a FRESH object every call — see
# probes/cpp/Tensor_07_typedevice.cpp. Cross-ref UniTensor UT-T1.
# =========================================================================
t = _mat()
report("`astype(same dtype)` returns `self` (is self) -- a no-op short-circuit "
       "added by the conti.py wrapper, NOT raw-C++ behavior (T-T1; xref UT-T1)",
       t.astype(t.dtype()) is t)
report("`to(same device)` returns `self` (is self) -- the device no-op "
       "short-circuit from the conti.py wrapper. On this CPU-only wheel `to(cpu)` "
       "is ALWAYS the no-op path (T-T1; xref UT-T1)",
       t.to(t.device()) is t and t.to(Device.cpu) is t)

# =========================================================================
# T-T2: astype to a DIFFERENT dtype -> a DISTINCT, independent tensor of the
# requested dtype (data copied; NOT same_data).
# =========================================================================
t = _mat()
a = t.astype(Type.ComplexDouble)
report("`astype(different dtype)` returns a DISTINCT object of the requested "
       "dtype (ComplexDouble) whose data is COPIED (not same_data) -- the real "
       "conversion path (T-T2)",
       a is not t and a.dtype() == Type.ComplexDouble and not t.same_data(a))

# =========================================================================
# T-T3: clone is a DEEP copy -- an independent tensor, not a view.
# =========================================================================
report("`clone()` returns an INDEPENDENT deep copy: mutating the clone is NOT "
       "visible through the source (returns_view -> False) (T-T3)",
       returns_view(
           make=_mat,
           derive=lambda s: s.clone(),
           mutate=lambda h: h.__setitem__((0, 0), 99.0),
           read=lambda s: s[0, 0].item(),
       ) is False)
t = _mat()
c = t.clone()
report("`clone()` does NOT share storage with the source (not same_data) (T-T3)",
       c is not t and not t.same_data(c))

# =========================================================================
# T-T4: `to_` is an IN-PLACE device move that returns **None**, NOT self.
# C++ `void to_(const int&)` (Tensor.hpp:683) bound directly (tensor_py.cpp:177).
# This diverges from UniTensor's `to_` (returns UniTensor&/self); it is the same
# in-place-return asymmetry as the cat-04 `flatten_`-returns-None finding.
# =========================================================================
t = _mat()
ret = t.to_(t.device())
report("`to_(device)` moves in place and returns **None** (NOT self) -- C++ "
       "`void to_` bound directly; diverges from UniTensor's self-returning `to_` "
       "(T-T4)",
       ret is None)
report("`to_` leaves the receiver on the target device (in-place move; CPU-only "
       "here, so device stays Device.cpu) (T-T4)",
       t.device() == Device.cpu)

# =========================================================================
# T-T5: `to_`'s parameter IS named `device` -- bound WITH py::arg("device")
# (tensor_py.cpp:177). The POSITIVE contrast to UniTensor UT-T3 (erased arg0).
# =========================================================================
t = _mat()
report("`to_(device=...)` works -- the parameter is correctly named `device` "
       "(py::arg present), UNLIKE UniTensor's erased `arg0` (T-T5; xref UT-T3)",
       t.to_(device=Device.cpu) is None)
t = _mat()
raised = False
try:
    t.to_(arg0=Device.cpu)
except TypeError:
    raised = True
report("`to_(arg0=...)` RAISES TypeError -- confirming the real keyword is "
       "`device`, not the pybind default `arg0` (T-T5)",
       raised)

# =========================================================================
# T-T2: the raw plumbing shims `astype_different_dtype` / `to_different_device`
# LEAK into public dir(Tensor); each HARD-ASSERTS the argument differs (raises
# on a no-op -- they exist only so the conti.py wrapper handles the no-op) and
# does the real conversion on a genuine change. Cross-ref UniTensor UT-T2.
# =========================================================================
report("the raw plumbing bindings `astype_different_dtype` / `to_different_device` "
       "both LEAK into public dir(Tensor) (T-T2; xref UT-T2)",
       "astype_different_dtype" in dir(Tensor)
       and "to_different_device" in dir(Tensor))

t = _mat()
raised_a = False
try:
    t.astype_different_dtype(t.dtype())  # no-op -> hard error
except RuntimeError:
    raised_a = True
raised_d = False
try:
    t.to_different_device(t.device())    # no-op -> hard error
except RuntimeError:
    raised_d = True
report("`astype_different_dtype(same dtype)` / `to_different_device(same device)` "
       "each RAISE RuntimeError on a no-op -- the hard assert that forces the "
       "conti.py wrapper to intercept same-dtype/same-device calls (T-T2)",
       raised_a and raised_d)
report("`astype_different_dtype(ComplexDouble)` does the REAL conversion (the "
       "path `astype` forwards to after its short-circuit) (T-T2)",
       t.astype_different_dtype(Type.ComplexDouble).dtype() == Type.ComplexDouble)

print("Tensor 07 probe ok")
