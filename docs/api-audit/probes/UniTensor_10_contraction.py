"""Behavioral probe for UniTensor category 10 — contraction & networks.

Members: `contract` (UniTensor member, lowercase) and the free functions
`Contract`, `Contracts`, `ncon` (documented here as they act on UniTensor).
`Network` is a separate class (cross-referenced from per-class/network.md, not
re-audited here).

Every behavioral claim in 10-contraction-networks.md cites one report(...) below.
Run: `source tools/env.sh && $PY docs/api-audit/probes/UniTensor_10_contraction.py`
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
from probe_helper import report  # noqa: E402

import cytnx  # noqa: E402


def _np2(u):
    """Read a rank-2 UniTensor into a nested Python list (row-major)."""
    r, c = u.shape()[0], u.shape()[1]
    return [[u.at([i, j]).value for j in range(c)] for i in range(r)]


# --- build two rank-2 UniTensors A (2x3) and B (3x2) ------------------------
A = cytnx.UniTensor(cytnx.arange(6).reshape(2, 3).astype(cytnx.Type.Double), rowrank=1)
B = cytnx.UniTensor(cytnx.arange(6).reshape(3, 2).astype(cytnx.Type.Double), rowrank=1)

# hand contraction A @ B (matmul over the shared middle dim) -----------------
An = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
Bn = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
hand = [[sum(An[i][k] * Bn[k][j] for k in range(3)) for j in range(2)] for i in range(2)]
#   hand == [[10, 13], [28, 40]]


# ---------------------------------------------------------------------------
# Membership / N-casing (UT-N1)
# ---------------------------------------------------------------------------
report("`contract` is a UniTensor MEMBER (lowercase — correct per N-casing)",
       hasattr(cytnx.UniTensor, "contract"))
report("there is NO Capitalized `Contract` member on UniTensor (it is a free function)",
       not hasattr(cytnx.UniTensor, "Contract"))
report("`Contract` is a Capitalized FREE function (acts on objects — correct per N-casing)",
       hasattr(cytnx, "Contract") and not hasattr(cytnx.UniTensor, "Contract"))
report("`Contracts` is a Capitalized FREE function (deprecated plural spelling)",
       hasattr(cytnx, "Contracts"))
report("`ncon` is a lowercase FREE function (community-standard TN name — kept lowercase)",
       hasattr(cytnx, "ncon"))

# discover the real signatures at runtime
sig_contract = cytnx.UniTensor.contract.__doc__.strip().splitlines()[0]
report("member `contract` signature is contract(self, inR, mv_elem_self=False, mv_elem_rhs=False)",
       "mv_elem_self" in sig_contract and "mv_elem_rhs" in sig_contract and "inR" in sig_contract)
report("free `Contract` is overloaded: pairwise (Tl,Tr,cacheL,cacheR) and list (TNs,order,optimal)",
       "Tl" in cytnx.Contract.__doc__ and "TNs" in cytnx.Contract.__doc__)
report("free `ncon` signature exposes connect_list_in + check_network/optimize/cont_order/out_labels",
       all(k in cytnx.ncon.__doc__ for k in
           ("connect_list_in", "check_network", "optimize", "cont_order", "out_labels")))


# ---------------------------------------------------------------------------
# ncon index convention correctness (UT-N4) — value-verify against hand result
# ---------------------------------------------------------------------------
# A legs [-1, 1], B legs [1, -2]: positive 1 is the contracted shared bond,
# negatives -1,-2 are the open output legs.  This IS the matmul A @ B.
r = cytnx.ncon([A, B], [[-1, 1], [1, -2]])
report("ncon([A,B],[[-1,1],[1,-2]]) returns a rank-2 UniTensor of shape [2,2]",
       list(r.shape()) == [2, 2])
report("ncon([A,B],[[-1,1],[1,-2]]) equals the hand contraction A@B elementwise "
       "(positives=contracted bond, negatives=open legs)",
       _np2(r) == hand)

# open-leg ordering: the output is ordered by -1, -2, ... regardless of which
# tensor/position carries each negative label.  Swap so B carries -1 and A -2.
r2 = cytnx.ncon([A, B], [[-2, 1], [1, -1]])
#   now output leg -1 is B's dim-2 leg, -2 is A's dim-2 leg => the transpose of `hand`
handT = [[hand[i][j] for i in range(2)] for j in range(2)]
report("ncon orders open output legs by -1,-2,... (leg -1 first) regardless of tensor position",
       _np2(r2) == handT)

# fully-contracted (all-positive) network -> a rank-1 scalar (the trace of A@B)
tr = cytnx.ncon([A, B], [[1, 2], [2, 1]])
report("ncon with an all-positive network fully contracts to a rank-1 scalar",
       list(tr.shape()) == [1])
report("the fully-contracted scalar equals trace(A@B) == 50.0",
       abs(tr.item() - sum(hand[i][i] for i in range(2))) < 1e-9)

# check_network validates the index convention (each positive label exactly twice)
raised = False
try:
    cytnx.ncon([A, B], [[-1, 1], [2, -2]], check_network=True)
except RuntimeError:
    raised = True
report("ncon(check_network=True) RAISES when a positive label does not appear exactly twice",
       raised)

# ncon does not mutate its inputs
Abefore = _np2(A)
cytnx.ncon([A, B], [[-1, 1], [1, -2]])
report("ncon does NOT mutate its input UniTensors", _np2(A) == Abefore)


# ---------------------------------------------------------------------------
# member `contract` == free `Contract` == hand (UT-N2, idiom split)
# ---------------------------------------------------------------------------
# member/free contract match legs by COMMON LABELS (not ncon integer lists)
Al = A.relabel(["a", "k"])
Bl = B.relabel(["k", "b"])
mc = Al.contract(Bl)                    # member idiom
fc = cytnx.Contract(Al, Bl)             # free pairwise idiom
fl = cytnx.Contract([Al, Bl])           # free list idiom
report("member `contract` contracts the common-label leg `k`, giving open legs [a,b] shape [2,2]",
       list(mc.labels()) == ["a", "b"] and list(mc.shape()) == [2, 2])
report("member `contract` result equals the hand contraction A@B elementwise",
       _np2(mc) == hand)
report("free `Contract(Tl,Tr)` produces the SAME result as the member `contract` on the same pair "
       "(idiom split — two spellings, one operation)",
       _np2(fc) == _np2(mc))
report("free `Contract([TNs])` (list overload) also produces the same result",
       _np2(fl) == _np2(mc))

# member contract is pure (returns a new object; does not consume the operands)
report("member `contract` is pure — returns a NEW UniTensor and leaves the operands intact",
       mc is not Al and mc is not Bl and _np2(A) == Abefore)


# ---------------------------------------------------------------------------
# Contracts is deprecated -> Contract (UT-N3)
# ---------------------------------------------------------------------------
cs = cytnx.Contracts([Al, Bl], order="", optimal=True)
report("`Contracts` still runs and returns the SAME result as `Contract` (deprecated alias)",
       _np2(cs) == _np2(fl))

# runtime truth: the C++ [[deprecated]] attribute is COMPILE-TIME only — calling
# Contracts from Python emits NO DeprecationWarning/FutureWarning.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cytnx.Contracts([Al, Bl], order="", optimal=True)
    py_warnings = [x.category.__name__ for x in w]
report("`Contracts` emits NO Python runtime warning — the C++ [[deprecated]] is compile-time only, "
       "so the deprecation never reaches a Python user",
       py_warnings == [])


print("UniTensor 10 probe ok")
