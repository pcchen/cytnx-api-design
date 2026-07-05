"""Behavioral probe for the superset-method PILOT of the UniTensor audit:
category 01 (construction & init) and 02 (static generators), verified against
the installed cytnx==1.1.0 wheel (NOT source-inferred, NOT a 1.0.0 wheel).

Every runtime claim in docs/api-audit/UniTensor/01-construction-init.md and
02-static-generators.md is backed by a report(...) assertion here.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_cat01_02.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

UT = cytnx.UniTensor
Type, Device = cytnx.Type, cytnx.Device


def val(u, loc):
    return complex(u.at(loc).value)


report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")

# =========================================================================
# Category 01 — Construction & init
# =========================================================================

# C1: the empty constructor yields a Void / un-initialized rank-0 UniTensor.
e = UT()
report("UniTensor() is an un-initialized (Void) rank-0 tensor",
       e.rank() == 0 and "un-initialize" in e.uten_type_str())

# C2 (B2 view): UniTensor(Tensor) SHARES memory with the source Tensor --
# mutating the Tensor is observed through the UniTensor.
t = cytnx.zeros([2, 2])
u = UT(t)
t[0, 0] = 9.0
report("UniTensor(Tensor) shares memory: mutating the Tensor shows in the UniTensor",
       val(u, [0, 0]) == 9 + 0j)

# C3: Init is a PUBLIC method that re-initializes the object in place
# (a duplicate of the constructor path).
u2 = UT()
u2.Init(cytnx.ones([2, 2]))
report("Init is public and re-initializes in place (duplicates the constructor)",
       hasattr(UT, "Init") and val(u2, [1, 1]) == 1 + 0j)

# C4: bond-based construction defaults to dtype=Double, device=cpu.
b = cytnx.Bond(2)
ub = UT([b, b.redirect()])
report("UniTensor(bonds) defaults to dtype Double / device cpu",
       ub.dtype() == Type.Double and ub.device() == Device.cpu)

# =========================================================================
# Category 02 — Static generators
# =========================================================================

# G1: zeros/ones produce the requested dense content; default dtype is Double.
z = UT.zeros([2, 2])
o = UT.ones([2, 3])
report("zeros() fills 0, ones() fills 1, default dtype is Double(Float64)",
       val(z, [0, 0]) == 0 and val(o, [1, 2]) == 1 + 0j
       and z.dtype_str() == "Double (Float64)")

# G2: eye is an exact elementwise alias of identity.
idn, ey = UT.identity(3), UT.eye(3)
report("eye(d) is an exact elementwise alias of identity(d)",
       all(val(idn, [i, j]) == val(ey, [i, j]) for i in range(3) for j in range(3))
       and val(idn, [1, 1]) == 1 and val(idn, [0, 1]) == 0)

# G3: arange(N) yields 0..N-1.
ar = UT.arange(5)
report("arange(5) yields 0,1,2,3,4",
       [val(ar, [i]).real for i in range(5)] == [0.0, 1.0, 2.0, 3.0, 4.0])

# G4 (per-overload inconsistency): the 1-arg arange(Nelem, ...) overload has NO
# dtype/device kwarg, while the (start,end,...) overload does -- so
# arange(Nelem, dtype=...) raises.
try:
    UT.arange(5, dtype=Type.Double)
    g4 = False
except TypeError:
    g4 = True
report("arange(Nelem, dtype=...) is REJECTED: the 1-arg overload drops dtype/device",
       g4)
# ...but the (start, end) overload DOES accept dtype.
try:
    UT.arange(0, 5, 1, [], Type.Double)
    g4b = True
except TypeError:
    g4b = False
report("arange(start, end, step, labels, dtype) IS accepted (overload asymmetry)",
       g4b)

# G5: normal/uniform are seed-reproducible; a different seed differs.
a = UT.normal([50], 0.0, 1.0, seed=123)
b2 = UT.normal([50], 0.0, 1.0, seed=123)
c = UT.normal([50], 0.0, 1.0, seed=124)
report("normal(seed=s) is reproducible; a different seed produces different samples",
       all(abs(val(a, [i]) - val(b2, [i])) < 1e-12 for i in range(50))
       and not all(abs(val(a, [i]) - val(c, [i])) < 1e-12 for i in range(50)))

# G6 (F21): normal/uniform name the label kwarg `in_labels`, NOT `labels` --
# calling with labels= raises, proving the divergent name.
try:
    UT.normal([4], 0.0, 1.0, labels=["a", "b", "c", "d"])
    g6 = False
except TypeError:
    g6 = True
report("normal(labels=...) is REJECTED: the kwarg is named in_labels, unlike every "
       "other generator (F21 / parameter-consistency)", g6)

# G7 (N2 violation): the in-place fills normal_/uniform_ mutate self but return
# None, unlike the audit's other in-place methods which return self.
w = UT.zeros([4])
r = w.normal_(0.0, 1.0, seed=7)
report("normal_ fills in place but returns None (not self) -- breaks the "
       "in-place-returns-self convention", r is None
       and any(abs(val(w, [i])) > 0 for i in range(4)))

# G7 (positional naming): the extent operand exists as both an int-count form
# (`Nelem`) and a list form (`shape`) -- one concept, two overloads/names.
z_int = UT.zeros(6)
z_list = UT.zeros([6])
report("zeros accepts both the int-count (Nelem) and list (shape) forms -- one "
       "extent concept split across two overload names (UT-G7)",
       z_int.shape() == [6] and z_list.shape() == [6])

# G8 (positional order): linspace puts the count operand THIRD (start, end, Nelem)
# -- the numpy convention -- unlike the extent-first shape generators.
ls = UT.linspace(0.0, 1.0, 5)
report("linspace(start, end, Nelem) puts the count 3rd (numpy order), unlike "
       "shape-first generators (UT-G8)", ls.shape() == [5])

# G9 (positional order): distributions put `shape` FIRST (normal(shape, mean, std)),
# deliberately diverging from numpy's size-last convention but matching zeros/ones.
nz = UT.normal([4], 0.0, 1.0, seed=1)
report("normal(shape, mean, std) puts shape FIRST (internally consistent with "
       "zeros/ones; deliberate divergence from numpy size-last) (UT-G9)",
       nz.shape() == [4])

print("UniTensor cat01/02 probe ok")
