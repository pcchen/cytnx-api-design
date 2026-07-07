"""Behavioral probe for the Tensor audit, category 01 (construction & init),
verified against the installed cytnx==1.1.0 wheel (NOT source-inferred).

Every runtime claim in docs/api-audit/Tensor/01-construction-init.md is backed
by a report(...) assertion here.
Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_01_construction.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report, returns_view

Tensor = cytnx.Tensor
Type, Device = cytnx.Type, cytnx.Device


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# Category 01 — Construction & init
# =========================================================================

# C1: the no-arg constructor yields a Void / un-initialized rank-0 Tensor.
e = Tensor()
report("Tensor() is an un-initialized (Void) rank-0 tensor",
       e.rank() == 0 and e.dtype_str() == "Void")

# C2 (B2 view): the copy constructor Tensor(other) SHARES the source's storage
# (C++ `Tensor(const Tensor&) { _impl = rhs._impl; }`) -- mutating the source is
# observed through the copy, and same_data() is True.
a = cytnx.arange(6).reshape(2, 3)
b = Tensor(a)
a[0, 0] = 99.0
report("Tensor(other) copy-constructor SHARES storage (view): a mutation to the "
       "source shows through, same_data() is True",
       float(b[0, 0].item()) == 99.0 and a.same_data(b) is True)

# C3: the shape constructor Tensor(shape, ...) owns fresh storage and defaults to
# dtype=Type.Double, device=Device.cpu, init_zero=True (zero-filled).
t = Tensor([2, 3])
report("Tensor(shape) defaults to dtype Double / device cpu and zero-fills "
       "(init_zero default True)",
       t.dtype() == Type.Double and t.device() == Device.cpu
       and t.dtype_str() == "Double (Float64)"
       and float(t[1, 2].item()) == 0.0)

# C4: Init is a PUBLIC method that re-initializes the object in place. Unlike the
# UniTensor Init (two overloads), Tensor.Init takes ONLY the shape form -- it
# duplicates the shape constructor (overload 3), returning None.
z = Tensor()
ret = z.Init([2, 2])
report("Init is public and re-initializes in place, returning None (duplicates "
       "the shape constructor)",
       hasattr(Tensor, "Init") and ret is None
       and z.rank() == 2 and float(z[1, 1].item()) == 0.0)

# C5 (B2 view): from_storage(sin) DEFAULT (is_clone=False) SHARES the Storage's
# buffer -- mutating the Storage is seen in the Tensor.
s = cytnx.arange(4).storage()
fs = Tensor.from_storage(s)
s[0] = 77.0
report("from_storage(sin) DEFAULT shares the Storage's buffer (view): a mutation "
       "to the Storage shows through to the Tensor",
       float(fs[0].item()) == 77.0)

# The returns_view oracle confirms the share bidirectionally: source is the
# Storage, the derived handle is the Tensor; mutating the Tensor is visible in
# the source Storage, so the derivation is a view (returns True).
report("from_storage default is a VIEW of the Storage (returns_view oracle: "
       "a Tensor mutation is visible in the source Storage)",
       returns_view(lambda: cytnx.arange(4).storage(),
                    lambda st: Tensor.from_storage(st),
                    lambda tn: tn.__setitem__(0, 5.0),
                    lambda st: float(st[0])) is True)

# C6: from_storage(sin, is_clone=True) COPIES the Storage's buffer -- later
# mutations to the Storage are NOT seen in the Tensor. The `is_clone` kwarg is a
# Python-binding-only parameter (the pybind lambda clones sin before wrapping).
s2 = cytnx.arange(4).storage()
fs2 = Tensor.from_storage(s2, is_clone=True)
s2[1] = 55.0
report("from_storage(sin, is_clone=True) COPIES the Storage buffer: a later "
       "Storage mutation is NOT seen in the Tensor",
       float(fs2[1].item()) == 1.0)
report("from_storage exposes the Python-binding-only `is_clone` kwarg (default "
       "False)", "is_clone" in Tensor.from_storage.__doc__)

print("Tensor 01 probe ok")
