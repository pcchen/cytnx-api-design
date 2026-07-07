"""Behavioral probe for the Tensor audit, category 06 (linear algebra members &
reductions), verified against the installed cytnx==1.1.0 wheel (NOT
source-inferred).

Every runtime claim in docs/api-audit/Tensor/06-linalg-reductions.md is backed by
a report(...) assertion here. Members covered: the member decompositions Svd /
Eigh, the matrix inverse InvM / InvM_, the reductions Trace / Max / Min, and the
leaked raw plumbing binding cInvM_.

`Norm` is NOT in this category — it is owned by cat 05 (the 2-norm scalar).

These members are Capitalized `.def`-ed pass-throughs to the C++ Tensor methods
(tensor_py.cpp:1781-1798), which themselves forward to the cytnx::linalg:: free
functions. The only conti.py wrapper is InvM_ (Tensor_conti.py:99-101 -> return
self over the raw cInvM_) — the identical in-place-returns-self mechanism verified
on the raw-C++ side in cat 05 (probes/cpp/Tensor_05_arithmetic.cpp, finding T-A4);
no NEW binding-fidelity finding surfaces here, so this category carries no separate
C++ probe.

Run: source tools/env.sh && $PY docs/api-audit/probes/Tensor_06_linalg.py
"""
import sys, os, io, contextlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import cytnx
from probe_helper import report

Tensor = cytnx.Tensor
linalg = cytnx.linalg

report("wheel under test is 1.1.0 (runtime, not source-inferred)",
       cytnx.__version__ == "1.1.0")


def _mat(rows):
    """A Double cytnx.Tensor from a list-of-lists."""
    r = len(rows); c = len(rows[0])
    t = cytnx.zeros((r, c), cytnx.Type.Double)
    for i in range(r):
        for j in range(c):
            t[i, j] = float(rows[i][j])
    return t


def _np2(t):
    """A rank-2 Double cytnx.Tensor -> nested python-float lists."""
    r, c = int(t.shape()[0]), int(t.shape()[1])
    return [[t[i, j].item() for j in range(c)] for i in range(r)]


def _np1(t):
    return [t[i].item() for i in range(int(t.shape()[0]))]


def _matmul(A, B):
    """Plain python matrix product of two nested-list matrices."""
    r, k, c = len(A), len(B), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(c)]
            for i in range(r)]


def _close(A, B, tol=1e-9):
    return all(abs(A[i][j] - B[i][j]) < tol
               for i in range(len(A)) for j in range(len(A[0])))


# =========================================================================
# T-X1: Svd/Eigh/InvM/InvM_/Trace/Max/Min are Capitalized MEMBER methods
# (N-casing violation -> should be lowercase svd/eigh/inv_m/inv_m_/trace/max/
# min). The SAME names ALSO exist as cytnx.linalg FREE functions, which STAY
# Capitalized (they act on objects — cross-ref UniTensor UT-X1 / cat 08). The
# lowercase member spellings are NOT bound today.
# =========================================================================
for nm in ("Svd", "Eigh", "InvM", "InvM_", "Trace", "Max", "Min"):
    report(f"capitalized member `{nm}` exists on Tensor (N-casing: should be "
           f"lowercase member `{nm.lower() if nm != 'InvM' else 'inv_m'}`)",
           callable(getattr(Tensor, nm, None)))
for nm in ("Svd", "Eigh", "InvM", "InvM_", "Trace", "Max", "Min"):
    report(f"the capitalized name `{nm}` ALSO exists as a cytnx.linalg FREE "
           f"function (which correctly STAYS Capitalized)",
           callable(getattr(linalg, nm, None)))
for lo in ("svd", "eigh", "inv_m", "trace", "max", "min"):
    report(f"the lowercase member `{lo}` is NOT bound today (rename target)",
           not hasattr(Tensor, lo))

# =========================================================================
# T-X2: the decompositions return a POSITIONAL list[Tensor] whose length &
# slot meaning depend on the toggle flag. Svd(is_UvT=True) -> [S, U, vT]
# (S FIRST, unlike numpy's U,S,Vh) — value-verified by reconstruction
# M ~= U @ diag(S) @ vT; with is_UvT=False -> [S] only. Eigh(is_V=True) ->
# [eigvals, eigvecs] (length 2); is_V=False -> [eigvals]. Cross-ref UniTensor
# UT-X3 (which recommends a named result object).
# =========================================================================
M = cytnx.arange(6).reshape(3, 2).astype(cytnx.Type.Double)  # 3x2, full col rank
Mn = _np2(M)
res = M.Svd()
report("`Svd()` returns a length-3 list [S, U, vT] (default is_UvT=True)",
       isinstance(res, list) and len(res) == 3)
S, U, vT = res
report("`Svd` return ORDER is [S, U, vT] with S FIRST (a rank-1 singular-value "
       "vector), U (m x k) and vT (k x n) — unlike numpy's U,S,Vh",
       S.rank() == 1 and U.shape()[0] == 3 and vT.shape()[1] == 2)
Sd = [[S[i].item() if i == j else 0.0 for j in range(len(_np1(S)))]
      for i in range(len(_np1(S)))]
recon = _matmul(_matmul(_np2(U), Sd), _np2(vT))
report("`Svd` order VALUE-VERIFIED: M ~= U . diag(S) . vT reconstructs the input "
       "(confirms the [S, U, vT] slot assignment, not [U, S, vT])",
       _close(recon, Mn))
report("`Svd(is_UvT=False)` returns a length-1 list [S] — flag-dependent "
       "positional arity",
       len(M.Svd(False)) == 1)

H = _mat([[2, 1], [1, 3]])  # symmetric (Hermitian) 2x2
eig = H.Eigh()
report("`Eigh()` returns a length-2 list [eigvals, eigvecs] (default is_V=True)",
       isinstance(eig, list) and len(eig) == 2)
evals = _np1(eig[0])
report("`Eigh` slot 0 is the eigenvalues (ascending): for [[2,1],[1,3]] they are "
       "2-/+ (2 - sqrt2/... ) — sum(eigvals) == trace == 5",
       abs(sum(evals) - 5.0) < 1e-9 and eig[1].rank() == 2)
report("`Eigh(is_V=False)` returns a length-1 list [eigvals] — flag-dependent "
       "positional arity",
       len(H.Eigh(is_V=False)) == 1)

# =========================================================================
# T-X3: `InvM` is the MATRIX inverse (pure) — value-verified M @ InvM(M) ~= I.
# It is a NEAR-NAME COLLISION with the ELEMENT-WISE reciprocal `Inv` (cat 05):
# one letter separates two very different operations. On the same non-diagonal
# B they give DIFFERENT results. Cross-ref UniTensor UT-X4.
# =========================================================================
B = _mat([[1, 2], [3, 4]])
Bi = B.InvM()
report("`InvM` is PURE: returns a NEW tensor, source not aliased "
       "(same_data is False)",
       Bi is not B and not Bi.same_data(B))
report("`InvM` is the MATRIX inverse: B @ InvM(B) ~= I (value-verified)",
       _close(_matmul(_np2(B), _np2(Bi)), [[1.0, 0.0], [0.0, 1.0]]))
report("`InvM` (matrix inverse) and `Inv` (element-wise reciprocal, cat 05) give "
       "DIFFERENT results on the same B: InvM(B)[0,1] == 1.0 but "
       "Inv(B)[0,1] == 1/2 == 0.5 — the near-name collision hides two ops",
       abs(B.InvM()[0, 1].item() - 1.0) < 1e-9
       and abs(B.Inv()[0, 1].item() - 0.5) < 1e-9)

# =========================================================================
# T-X4: `InvM_` is the ONLY conti.py wrapper in this category
# (Tensor_conti.py:99-101 -> self.cInvM_(); return self) over the LEAKED raw
# `cInvM_` binding (tensor_py.cpp:1783). InvM_ inverts IN PLACE and returns SELF;
# the raw cInvM_ primitive LEAKS into public dir(Tensor) and self-aliases. This
# is the identical in-place-returns-self mechanism as the cat-05 element-wise
# family (finding T-A4), whose raw-C++ Tensor& return is verified there — no NEW
# C++ probe needed here.
# =========================================================================
B2 = _mat([[1, 2], [3, 4]])
b2id = B2
ret = B2.InvM_()
report("`InvM_()` returns SELF (conti.py wrapper) and inverts the MATRIX in "
       "place: B2 @ (former B2) ~= I",
       ret is B2)
report("`InvM_` mutated the operand in place to the matrix inverse "
       "(value-verified against the original B)",
       _close(_matmul(_np2(B2), _np2(B)), [[1.0, 0.0], [0.0, 1.0]]))
report("the raw plumbing binding `cInvM_` LEAKS into public dir(Tensor) "
       "(the C-prefixed primitive InvM_ wraps)",
       "cInvM_" in dir(Tensor))
B3 = _mat([[1, 2], [3, 4]])
b3id = B3
r3 = B3.cInvM_()
report("`cInvM_()` (raw primitive) mutates the receiver in place "
       "(self-aliasing) — plumbing behind InvM_/inv_m_",
       r3.same_data(b3id))

# =========================================================================
# T-X5: `Trace` DROPS the C++ default arguments (a=0, b=1). It is bound with no
# py::arg defaults (tensor_py.cpp:1798), so Python `Trace()` REQUIRES two
# positional axis args (named arg0/arg1 — the meaningful C++ names a/b are lost).
# v1 P2.
# =========================================================================
G = cytnx.arange(9).reshape(3, 3).astype(cytnx.Type.Double)
try:
    # pybind builds the TypeError's "Invoked with:" clause from repr(self), and
    # cytnx.Tensor.__repr__ dumps the tensor to stdout (a general repr concern,
    # not a cat-06 finding) — suppress it so the probe output stays clean.
    with contextlib.redirect_stdout(io.StringIO()):
        G.Trace()
    _trace_noarg_raises = False
except TypeError:
    _trace_noarg_raises = True
report("`Trace()` with NO args raises TypeError — the C++ defaults (a=0, b=1) "
       "were dropped by the binding; two positional axes are required",
       _trace_noarg_raises)
report("`Trace(0, 1)` sums the diagonal over the two given axes: "
       "trace(arange(9).reshape(3,3)) == 0 + 4 + 8 == 12",
       abs(G.Trace(0, 1).item() - 12.0) < 1e-9)

# =========================================================================
# T-X6: `Max` / `Min` are reductions returning the extremal element as a scalar
# cytnx.Tensor.
# =========================================================================
V = cytnx.arange(5).astype(cytnx.Type.Double)  # [0,1,2,3,4]
mx, mn = V.Max(), V.Min()
report("`Max()` returns the maximum element as a scalar cytnx.Tensor "
       "(max([0..4]) == 4)",
       isinstance(mx, cytnx.Tensor) and abs(mx.item() - 4.0) < 1e-9)
report("`Min()` returns the minimum element as a scalar cytnx.Tensor "
       "(min([0..4]) == 0)",
       isinstance(mn, cytnx.Tensor) and abs(mn.item() - 0.0) < 1e-9)

print("Tensor 06 probe ok")
