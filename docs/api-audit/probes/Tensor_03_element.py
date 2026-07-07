"""Behavioral probe for the Tensor audit, category 03 (element & storage access),
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/Tensor/03-element-storage-access.md is
backed by a report(...) assertion here. Members covered: item, storage, fill,
append, real, imag, numpy, and the __getitem__ / __setitem__ dunders.

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_03_element.py
"""
import sys, os, contextlib, io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

Tensor = cytnx.Tensor
Type = cytnx.Type


@contextlib.contextmanager
def _capture_c_stdout():
    """Capture the process's REAL fd-1 stdout (what C++ std::cout writes to),
    which contextlib.redirect_stdout cannot see. Yields a callable returning the
    bytes written to fd 1 while the block was active."""
    sys.stdout.flush()
    saved = os.dup(1)
    r, w = os.pipe()
    os.dup2(w, 1)
    os.close(w)
    captured = {}

    def _read():
        return captured.get("data", b"")

    try:
        yield _read
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        captured["data"] = os.read(r, 65536)
        os.close(r)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# Category 03 — Element & storage access
# =========================================================================

# -------------------------------------------------------------------------
# T-E1: the numpy() bridge. Default is a COPY; share_mem=True — despite
# promising a zero-copy view — ALSO returns an independent copy (the resulting
# ndarray reports OWNDATA=True and a write to it does NOT propagate to the
# tensor). The flag only enforces the contiguity precondition; the final
# py::array construction copies regardless. It is a pure pybind lambda
# (tensor_py.cpp:66-144); no C++ Tensor::numpy member exists.
# -------------------------------------------------------------------------
t = cytnx.arange(6)
nd = t.numpy()
nd[0] = 999.0
report("numpy() default is a COPY: writing the ndarray does NOT change the tensor",
       t.storage()[0] == 0.0)

t = cytnx.arange(6)  # contiguous Double
nd = t.numpy(share_mem=True)
report("numpy(share_mem=True) ndarray OWNS its buffer (OWNDATA True) — it is NOT a view",
       nd.flags["OWNDATA"] is True)
nd[0] = 999.0
report("numpy(share_mem=True) is ALSO a COPY: the promised zero-copy view is "
       "non-functional (write does not propagate to the tensor)",
       t.storage()[0] == 0.0)

# share_mem=True still enforces contiguity: it raises on a non-contiguous tensor.
tnc = cytnx.arange(24).reshape(2, 3, 4).permute(2, 0, 1)
def _raises(call):
    try:
        call(); return False
    except RuntimeError:
        return True
report("numpy(share_mem=True) RAISES on a non-contiguous tensor (contiguity precondition)",
       tnc.is_contiguous() is False and _raises(lambda: tnc.numpy(share_mem=True)))

# The numpy bridge (and from_storage) EXIST on Tensor — closing the gap that
# UniTensor has (UT-C3/UT-T6): UniTensor has no numpy() export.
report("the numpy bridge EXISTS on Tensor (numpy + from_storage) — closes the "
       "UniTensor UT-C3/UT-T6 gap (UniTensor has NO numpy())",
       hasattr(Tensor, "numpy") and hasattr(Tensor, "from_storage")
       and not hasattr(cytnx.UniTensor, "numpy"))

# -------------------------------------------------------------------------
# T-E2: storage() returns a shared-data VIEW — mutating the returned Storage is
# visible through the tensor. Thin pass-through of C++ Storage& storage().
# -------------------------------------------------------------------------
t = cytnx.arange(6)
s = t.storage()
s[0] = 888.0
report("storage() returns a shared-data VIEW: writing the Storage shows through "
       "the tensor (t[0] becomes 888)", t.storage()[0] == 888.0)

# -------------------------------------------------------------------------
# T-E3: real()/imag() return independent COPIES (complex-only). same_data is
# False and a write to the returned tensor does not propagate.
# -------------------------------------------------------------------------
tc = cytnx.arange(6).astype(Type.ComplexDouble)
re = tc.real()
report("real() returns a COPY, not a view (same_data() is False)",
       tc.same_data(re) is False)
re[0] = 777.0
report("real() COPY: writing the returned tensor does not change the source",
       tc.storage()[0].real == 0.0)
im = tc.imag()
report("imag() likewise returns a COPY (same_data() is False)",
       tc.same_data(im) is False)
# real()/imag() are complex-only: they raise on a real tensor.
report("real() RAISES on a non-complex tensor (complex-only precondition)",
       _raises(lambda: cytnx.arange(6).real()))

# -------------------------------------------------------------------------
# T-E4: slice READ is a COPY, not a numpy-style view. t[0:2] / t[0] return an
# independent tensor (C++ get() "does not share memory", Tensor.hpp:1023). This
# is a B2 hazard: numpy code expecting t[0:2] to alias t silently gets a copy.
# -------------------------------------------------------------------------
t = cytnx.arange(6).reshape(2, 3)
sub_slice = t[0:1]
report("slice READ t[0:1] returns an independent COPY (same_data() is False) — "
       "NOT a numpy-style view (B2 hazard)", t.same_data(sub_slice) is False)
sub_int = t[0]
report("integer-index READ t[0] likewise returns a COPY (same_data() is False)",
       t.same_data(sub_int) is False)
# Confirm the copy is detached: writing the slice does not change the source.
sub_slice[0, 0] = 424242.0
report("the slice-read COPY is detached: mutating it does not change the source",
       t[0, 0].item() == 0.0)

# -------------------------------------------------------------------------
# T-E5: slice / element ASSIGN mutates the tensor IN PLACE (the assignment
# target is the tensor's own storage). t[0,0]=v and t[0:1]=rhs are seen through
# an alias.
# -------------------------------------------------------------------------
t = cytnx.arange(6).reshape(2, 3)
alias = t
t[0, 0] = 555.0
report("element ASSIGN t[0,0]=v mutates in place (an alias observes the write)",
       alias[0, 0].item() == 555.0)
t[1:2] = cytnx.zeros((1, 3)) + 7.0
report("slice ASSIGN t[1:2]=rhs mutates in place (an alias observes the write)",
       alias[1, 0].item() == 7.0)

# -------------------------------------------------------------------------
# T-E6: item() extracts the sole scalar of a 1-element tensor; it RAISES on a
# multi-element tensor. The pybind lambda dispatches all 11 dtypes.
# -------------------------------------------------------------------------
report("item() extracts the sole scalar of a 1-element tensor",
       cytnx.arange(1).item() == 0.0)
report("item() extracts across dtypes (Int64 1-element tensor -> 5)",
       (cytnx.arange(1).astype(Type.Int64) + 5).item() == 5)
report("item() RAISES on a multi-element tensor (B4)",
       _raises(lambda: cytnx.arange(6).item()))

# -------------------------------------------------------------------------
# T-E7: fill(val) sets every element in place and returns None.
# -------------------------------------------------------------------------
t = cytnx.zeros((2, 2))
ret = t.fill(3.0)
report("fill(val) sets EVERY element in place",
       t[0, 0].item() == 3.0 and t[1, 1].item() == 3.0)
report("fill(val) returns None (in-place, no self-return)", ret is None)

# -------------------------------------------------------------------------
# T-E8: append(val) grows the tensor along axis 0 in place (scalar / Tensor /
# Storage overloads); returns None.
# -------------------------------------------------------------------------
t = cytnx.arange(3)  # shape [3]
ret = t.append(9.0)
report("append(scalar) grows a 1-D tensor along axis 0 ([3] -> [4]) in place",
       t.shape() == [4] and t[3].item() == 9.0)
report("append(scalar) returns None (in-place)", ret is None)
t2 = cytnx.arange(6).reshape(2, 3)  # shape [2,3]
t2.append(cytnx.arange(3))          # append a [3] row
report("append(Tensor) grows axis 0 by a matching sub-tensor ([2,3] -> [3,3])",
       t2.shape() == [3, 3])

# -------------------------------------------------------------------------
# T-E9: the bare-1-D-slice __getitem__ branch leaks a leftover std::cout debug
# line ("start stop step") to the process's REAL stdout (tensor_py.cpp:355) with
# NO scoped_ostream_redirect guard — so contextlib.redirect_stdout CANNOT see it
# (buffer stays empty), yet the line hits fd 1. A correctness/hygiene defect.
# -------------------------------------------------------------------------
buf = io.StringIO()
t = cytnx.arange(6)
with contextlib.redirect_stdout(buf):
    _ = t[0:2]
report("t[0:2] (bare 1-D slice) debug line is UNCAPTURABLE by "
       "contextlib.redirect_stdout (the Python buffer stays empty)",
       buf.getvalue() == "")

t = cytnx.arange(6)
with _capture_c_stdout() as read_fd1:
    _ = t[0:2]              # leaks "0 2 1\n" to real fd 1
    _ = t[1:4]              # leaks "1 4 1\n"
leaked = read_fd1().decode()
report("t[0:2] leaks a leftover 'start stop step' std::cout debug line to the "
       "process's REAL fd-1 stdout (captured at the fd level: '0 2 1')",
       "0 2 1" in leaked and "1 4 1" in leaked)

# Tuple indexing takes a different branch and does NOT leak the debug line.
t = cytnx.arange(6).reshape(2, 3)
with _capture_c_stdout() as read_fd1:
    _ = t[0, 1]
report("tuple indexing t[0,1] takes a different branch and does NOT leak the "
       "debug line", read_fd1().decode().strip() == "")

print("Tensor 03 probe ok")
