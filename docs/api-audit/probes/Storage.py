"""Behavioral probe for the Storage class (Cytnx 1.1.0).

Every behavioral claim made in docs/api-audit/per-class/Storage.md's Parity
and Consistency findings sections is backed by a report() assertion here. Run
with: source tools/env.sh && $PY docs/api-audit/probes/Storage.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
import numpy as np
from probe_helper import report, returns_view

Storage = cytnx.Storage


def mk():
    """A fresh, independent Double Storage [1, 2, 3]."""
    return Storage.from_pylist([1.0, 2.0, 3.0])


# --- construction / metadata ---------------------------------------------

s = mk()
report("from_pylist([1.,2.,3.]) builds a Double Storage on cpu, size 3",
       s.dtype() == cytnx.Type.Double and s.dtype_str() == "Double (Float64)"
       and s.device() == cytnx.Device.cpu and s.device_str() == "cytnx device: CPU"
       and s.size() == 3 and len(s) == 3)

report("size() (no. of elements) and capacity() (allocated slots) are distinct: "
       "a size-3 Storage rounds capacity up to a multiple of STORAGE_DEFT_SZ (==2), "
       "so capacity() == 4 >= size() == 3",
       s.size() == 3 and s.capacity() == 4)

c = Storage(4, cytnx.Type.Float)
report("Storage(size, dtype) constructor builds a typed, sized Storage",
       c.size() == 4 and c.dtype_str() == "Float (Float32)")

# --- C7: the empty default constructor is a half-initialized object -------
# Storage() sets _impl to the abstract Storage_base (not a typed
# StorageImplementation), whose virtual size() throws "Not implemented".

e = Storage()
report("Storage() default constructor yields a Void-dtype half-object",
       e.dtype() == cytnx.Type.Void and e.dtype_str() == "Void")
try:
    e.size()
    empty_size_ok = True
except RuntimeError:
    empty_size_ok = False
report("C7: Storage().size() RAISES RuntimeError ('Not implemented') instead of "
       "returning 0 -- the default-constructed _impl is the abstract Storage_base, "
       "so an empty Storage cannot even be queried for its size",
       empty_size_ok is False)

# --- Init: in-place re-initialization ------------------------------------

si = mk()
si.Init(5, cytnx.Type.Int64, -1, True)
report("Init(size, dtype, device, init_zero) re-initializes the SAME wrapper "
       "in place: dtype/size change and elements are zeroed",
       si.size() == 5 and si.dtype_str() == "Int64"
       and [si[i] for i in range(5)] == [0, 0, 0, 0, 0])

# --- P1 (headline): numpy() returns a COPY, not a view aliased to the -----
# Storage buffer. The binding clones (or moves-to-cpu-clones) then release()s
# ownership INTO the ndarray, so the array shares no memory with the Storage.
# This diverges from the near-universal NumPy/PyTorch `.numpy()` zero-copy
# convention.

def numpy_aliases_storage():
    return returns_view(
        make=mk,
        derive=lambda src: src.numpy(),
        mutate=lambda arr: arr.__setitem__(0, 99.0),
        read=lambda src: src[0],
    )
report("P1: numpy() is a COPY, not a view -- mutating the returned ndarray does "
       "NOT change the source Storage (returns_view is False)",
       numpy_aliases_storage() is False)

s_np = mk()
arr = s_np.numpy()
s_np[1] = 88.0
report("P1 (other direction): mutating the source Storage does NOT change a "
       "previously-returned numpy array either -- the two own separate buffers",
       arr[1] == 2.0 and isinstance(arr, np.ndarray) and arr.dtype == np.float64)

sc = Storage.from_pylist([1 + 2j, 3 + 4j])
arr_c = sc.numpy()
arr_c[0] = 0
report("P1 holds for complex dtype too: numpy() of a ComplexDouble Storage is a "
       "complex128 copy; mutating it leaves the source's element unchanged",
       arr_c.dtype == np.complex128 and sc[0] == (1 + 2j))

# --- P2: astype() returns SELF (identical object) when the target dtype ----
# equals the current dtype, but a fresh independent Storage otherwise. This
# self-return is implemented in the Python augmentation layer
# (cytnx/Storage_conti.py: `if self.dtype()==new_type: return self`); the
# pybind `astype` is commented out (only astype_different_type is bound).

s2 = mk()
report("P2: astype(same_dtype) returns the IDENTICAL Python object (astype(D) is s), "
       "a pure method that aliases its receiver on a no-op conversion",
       s2.astype(cytnx.Type.Double) is s2)

s3 = mk()
d3 = s3.astype(cytnx.Type.Float)
d3[0] = 7.0
report("P2: astype(different_dtype) returns a NEW, independent Storage (not self); "
       "mutating it leaves the source unchanged",
       (d3 is s3) is False and d3.dtype_str() == "Float (Float32)" and s3[0] == 1.0)

# --- P3: to() has the same self-return pattern (Storage_conti.py) ---------

s4 = mk()
report("P3: to(same_device) returns the IDENTICAL Python object (to(cpu) is s)",
       s4.to(cytnx.Device.cpu) is s4)
report("P3: to_(device) is the in-place device move and returns None (void)",
       s4.to_(cytnx.Device.cpu) is None and s4.size() == 3)

# --- P4: the internal escape-hatch helpers astype_different_type / ---------
# to_different_device are bound as public methods and RAISE when handed the
# no-op (same dtype/device) case the Python wrappers are meant to intercept.

s5 = mk()
try:
    s5.astype_different_type(cytnx.Type.Double)
    astype_diff_same_ok = True
except RuntimeError:
    astype_diff_same_ok = False
report("P4: astype_different_type(same_dtype) RAISES RuntimeError (the internal "
       "helper cannot handle the no-op; astype() in Storage_conti.py intercepts it "
       "first) -- an implementation detail leaked into the public surface",
       astype_diff_same_ok is False)
try:
    s5.to_different_device(cytnx.Device.cpu)
    to_diff_same_ok = True
except RuntimeError:
    to_diff_same_ok = False
report("P4: to_different_device(same_device) RAISES RuntimeError for the same reason "
       "-- another leaked internal helper",
       to_diff_same_ok is False)

# --- P5: the 11 c_pylist_<dtype> methods are type-specific internal --------
# accessors leaked into the public surface; the wrong one for the Storage's
# dtype RAISES. The pylist() wrapper (Storage_conti.py) dispatches by dtype.

s6 = mk()
report("P5: c_pylist_double() (matching the Double dtype) returns a Python list copy",
       s6.c_pylist_double() == [1.0, 2.0, 3.0])
try:
    s6.c_pylist_int64()
    mismatch_ok = True
except RuntimeError:
    mismatch_ok = False
report("P5: c_pylist_int64() on a Double Storage (dtype MISMATCH) RAISES RuntimeError "
       "-- these 11 raw accessors require the caller to know the dtype and pick the "
       "right one; pylist() is the dtype-dispatching wrapper that hides them",
       mismatch_ok is False)
report("pylist() returns the correct Python list copy without the caller naming a dtype",
       s6.pylist() == [1.0, 2.0, 3.0] and isinstance(s6.pylist(), list))
report("__iter__ iterates elements (via StorageIterator in Storage_conti.py)",
       [x for x in mk()] == [1.0, 2.0, 3.0])

# --- P6: __repr__/__str__ return '' ; info is a std::cout side effect, but --
# here it IS captured via py::scoped_ostream_redirect, and print_info() is
# also bound and capturable (a positive contrast with Device.Print_Property,
# enums P4, which is NOT capturable, and Symmetry.print_info, which is unbound).

s7 = mk()
report("P6: repr(storage) and str(storage) both evaluate to the empty string '' "
       "(the info is printed as a side effect, not returned)",
       repr(s7) == "" and str(s7) == "")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    repr(s7)
report("P6: ...yet repr(storage) DOES print human-readable info as a side effect, "
       "and (unlike Device.Print_Property) it IS capturable via redirect_stdout",
       "dtype" in buf.getvalue() and "Double (Float64)" in buf.getvalue())
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    s7.print_info()
report("P6: print_info() is directly bound AND capturable (scoped_ostream_redirect) "
       "-- Storage does what Device.Print_Property (enums P4) fails to, and binds "
       "what Symmetry.print_info leaves unbound",
       "dtype" in buf2.getvalue() and "size" in buf2.getvalue())

# --- P7: the __getitem__/__setitem__ bounds guard is off-by-one -----------
# (`idx > self.size()` instead of `>=`), but idx==size is independently caught
# by the inner at() bounds check, so both raise and no OOB read occurs.
# Negative indices raise TypeError (unsigned param; no Python-style negatives).

s8 = mk()
try:
    _ = s8[3]  # idx == size: pybind guard (3 > 3) is False, but at() catches it
    oob_ok = True
except RuntimeError:
    oob_ok = False
report("P7: storage[size] (idx == size) RAISES RuntimeError -- the pybind guard "
       "'idx > size' is off-by-one and lets it through, but the inner at() bounds "
       "check catches it independently, so no out-of-bounds read actually happens",
       oob_ok is False)
try:
    _ = s8[4]  # idx > size: pybind guard rejects
    far_ok = True
except RuntimeError:
    far_ok = False
report("P7: storage[size+1] also RAISES RuntimeError (rejected by the pybind guard)",
       far_ok is False)
try:
    _ = s8[-1]
    neg_ok = True
except TypeError:
    neg_ok = False
report("P7: storage[-1] RAISES TypeError -- the index param is unsigned, so there is "
       "NO Python-style negative indexing (unlike list/ndarray)",
       neg_ok is False)

# --- element get/set mutation: __setitem__ mutates in place ---------------

def setitem_mutates():
    return returns_view(
        make=mk,
        derive=lambda src: src,               # the handle IS the source
        mutate=lambda h: h.__setitem__(0, 55.0),
        read=lambda src: src[0],
    )
report("__setitem__ (storage[i] = v) mutates the Storage in place (get after set "
       "reflects the new value)",
       setitem_mutates() is True)

# --- clone(): deep, independent copy --------------------------------------

def clone_aliases():
    return returns_view(
        make=mk,
        derive=lambda src: src.clone(),
        mutate=lambda h: h.__setitem__(0, 42.0),
        read=lambda src: src[0],
    )
s9 = mk()
report("clone() is a deep copy: mutating the clone does NOT touch the source "
       "(returns_view is False), and the clone is a distinct object",
       clone_aliases() is False and (s9.clone() is s9) is False)

# --- fill / set_zeros: in-place whole-buffer writes -----------------------

s10 = mk()
s10.fill(5.0)
report("fill(val) sets every element in place (returns None)",
       [s10[i] for i in range(3)] == [5.0, 5.0, 5.0])
s10.set_zeros()
report("set_zeros() zeros every element in place",
       [s10[i] for i in range(3)] == [0.0, 0.0, 0.0])

# --- append: in-place grow by one ----------------------------------------

s11 = mk()
s11.append(9.0)
report("append(val) grows the Storage in place by one element",
       s11.size() == 4 and s11[3] == 9.0)

# --- resize: zero-fills a FRESH grow, but NOT reused capacity (C4) ---------

s12 = mk()
s12.resize(5)
report("C4: resize(5) that grows beyond capacity zero-fills the new region "
       "(calloc): elements become [1, 2, 3, 0, 0]",
       s12.size() == 5 and [s12[i] for i in range(5)] == [1.0, 2.0, 3.0, 0.0, 0.0])

s13 = Storage.from_pylist([1.0, 2.0, 3.0, 4.0])  # size 4, capacity 4
s13.resize(1)   # shrink: size_=1, capacity stays 4
s13.resize(4)   # grow back WITHIN capacity: no realloc, no re-zero
report("C4: resize DOWN then UP within the old capacity does NOT re-zero the "
       "reused slots -- stale data [1, 2, 3, 4] reappears, unlike the fresh-grow "
       "path above; resize's zero-fill is therefore not guaranteed",
       [s13[i] for i in range(4)] == [1.0, 2.0, 3.0, 4.0])

# --- astype complex->real is refused (B3): must use real()/imag() ---------

sc2 = Storage.from_pylist([1 + 2j, 3 + 4j])
try:
    sc2.astype(cytnx.Type.Double)
    c2r_ok = True
except RuntimeError:
    c2r_ok = False
report("B3: astype() REFUSES a complex->real conversion (RuntimeError); the real "
       "and imaginary parts must be taken explicitly via real()/imag()",
       c2r_ok is False)

# --- real()/imag(): complex-only, each returns a NEW real Storage (C8) -----

report("real()/imag() on a complex Storage return NEW real-typed Storages holding "
       "the respective parts",
       [sc2.real()[i] for i in range(2)] == [1.0, 3.0]
       and [sc2.imag()[i] for i in range(2)] == [2.0, 4.0])
sr = mk()
try:
    sr.real()
    real_on_real_ok = True
except RuntimeError:
    real_on_real_ok = False
report("C8: real() on a REAL Storage RAISES RuntimeError ('can only be called from "
       "complex type') -- asymmetric with numpy, where ndarray.real of a real array "
       "returns the array itself",
       real_on_real_ok is False)

# --- C5: operator== RAISES on dtype mismatch instead of returning False ---

report("== returns True for value-equal same-dtype Storages (deep value compare, "
       "not identity)",
       (mk() == mk()) is True and (mk() == Storage.from_pylist([1.0, 2.0, 9.0])) is False)
try:
    _ = Storage.from_pylist([1.0]) == Storage.from_pylist([1])  # Double vs Uint64
    eq_cross_ok = True
except RuntimeError:
    eq_cross_ok = False
report("C5: comparing two Storages of DIFFERENT dtype with == RAISES RuntimeError "
       "instead of returning False -- violates Python's convention that == is total "
       "and never throws across types (a real footgun: `if a == b:` can raise)",
       eq_cross_ok is False)

# --- C6: from_pylist dtype inference is value- and overload-order-dependent -

report("C6: from_pylist([1, 2]) (all-positive ints) infers UNSIGNED Uint64, while "
       "from_pylist([1, -2]) infers Int64, and from_pylist([1., 2.]) infers Double "
       "-- same-shaped integer input yields a sign-dependent dtype, and an "
       "all-positive int list silently becomes unsigned (so later subtraction "
       "underflows); there is no explicit dtype= parameter to override this",
       Storage.from_pylist([1, 2]).dtype_str() == "Uint64"
       and Storage.from_pylist([1, -2]).dtype_str() == "Int64"
       and Storage.from_pylist([1.0, 2.0]).dtype_str() == "Double (Float64)")

# --- __copy__/__deepcopy__ are bound to clone (deep copy) -----------------

import copy
s14 = mk()
cp = copy.copy(s14)
cp[0] = 13.0
report("copy.copy()/copy.deepcopy() are both bound to clone(), so even a shallow "
       "copy.copy() is a DEEP copy: mutating it leaves the source unchanged",
       (cp is s14) is False and s14[0] == 1.0)

print("Storage probe ok")
