"""Behavioral probe for the Tensor audit, category 02 (metadata & introspection),
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/Tensor/02-metadata-introspection.md is
backed by a report(...) assertion here.
Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_02_metadata.py
"""
import sys, os, contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

Tensor = cytnx.Tensor
Type, Device = cytnx.Type, cytnx.Device


@contextlib.contextmanager
def _mute_c_stdout():
    """Swallow C++-level stdout — a failed pybind overload (e.g. rejecting a
    keyword call) prints the whole tensor via std::cout — so the keyword-
    rejection check stays quiet."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# Category 02 — Metadata & introspection
# A small rank-3 Double/cpu tensor is the fixture for the whole category.
# =========================================================================
t = cytnx.arange(24).reshape(2, 3, 4)

# M1: shape() lists the extent of each axis; rank() == len(shape()).
report("shape() lists the extent of each axis ([2, 3, 4])", t.shape() == [2, 3, 4])
report("rank() is the number of axes and equals len(shape()) (3)",
       t.rank() == 3 and t.rank() == len(t.shape()))

# M2: dtype()/device() return integer Type/Device codes; the _str forms name them.
report("dtype() returns the int Type code Type.Double (3)",
       t.dtype() == int(Type.Double))
report("dtype_str() names the element type ('Double (Float64)')",
       t.dtype_str() == "Double (Float64)")
report("device() returns the int Device code Device.cpu (-1)",
       t.device() == int(Device.cpu))
report("device_str() names the device ('cytnx device: CPU')",
       t.device_str() == "cytnx device: CPU")

# M3: is_contiguous() is a predicate — True for a freshly built tensor, and
# False after a permute produces a non-contiguous view (same storage, reordered
# strides). This is the canonical way to make a Tensor non-contiguous.
report("is_contiguous() is True for a freshly-built (arange/reshape) tensor",
       t.is_contiguous() is True)
p = t.permute(2, 0, 1)
report("is_contiguous() is False after a non-contiguous permute (a strided view)",
       p.is_contiguous() is False)

# M4: same_data(other) is the view-vs-copy oracle — True iff the two tensors
# share storage. self shares with itself; a clone() allocates independent storage.
report("same_data(self) is True (a tensor shares storage with itself)",
       t.same_data(t) is True)
report("same_data(clone()) is False (clone() allocates independent storage)",
       t.same_data(t.clone()) is False)
# The permuted view shares storage with its source — same_data confirms the view.
report("same_data(permute view) is True (permute returns a storage-sharing view)",
       t.same_data(p) is True)

# M5 (PC1): same_data's argument name is erased to `arg0` — the pybind
# `.def("same_data", &Tensor::same_data)` registers no py::arg, so the operand is
# positional-only and a keyword call same_data(other=...) is REJECTED.
def _rejects_kw(call):
    try:
        with _mute_c_stdout():
            call()
        return False
    except TypeError:
        return True

report("same_data's arg is erased to arg0 — same_data(other=...) is REJECTED (PC1)",
       _rejects_kw(lambda: t.same_data(other=t)))

print("Tensor 02 probe ok")
