"""Behavioral probe for UniTensor category 08 — linalg operations (the
cytnx.linalg FREE functions that take/return UniTensor), verified against the
installed cytnx==1.1.0 wheel (NOT source-inferred).

These are cytnx.linalg free functions, NOT UniTensor members. Every runtime
claim in docs/api-audit/UniTensor/08-linalg-operations.md is backed by a
report(...) assertion here. No C++ probe accompanies this category: the linalg
free functions are bound as direct pass-through pybind lambdas to the C++
cytnx::linalg:: overloads (pybind/linalg_py.cpp), with no conti.py wrapper — so
there is no binding-fidelity finding to verify on a separate raw-C++ side.
Run: source tools/env.sh && $PY docs/api-audit/probes/UniTensor_08_linalg.py
"""
import sys, os, io, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import numpy as np
import cytnx
from cytnx import linalg as la
from probe_helper import report

UT = cytnx.UniTensor


def mat(a):
    """A fresh rank-2 Dense UniTensor (rowrank 1) holding the numpy matrix `a`."""
    return cytnx.UniTensor(cytnx.from_numpy(np.array(a, dtype=float).copy()), rowrank=1)


def raises_typeerror(call):
    """True if `call` raises TypeError; stdout noise from pybind's overload
    error repr is swallowed so probe output stays clean."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            call()
        return False
    except TypeError:
        return True


# ---------------------------------------------------------------------------
# 1. Reachability at runtime — which brief members are present / absent.
# ---------------------------------------------------------------------------
present = {m for m in dir(la) if not m.startswith("_")}

for m in ["Svd", "Svd_truncate", "Gesvd", "Gesvd_truncate", "Rsvd", "Hosvd",
          "Qr", "Qdr", "Eig", "Eigh", "ExpH", "ExpM", "InvM", "InvM_",
          "Inv", "Inv_", "Trace", "Pow", "Pow_", "Conj", "Conj_", "Norm"]:
    report(f"linalg.{m} is reachable at runtime", m in present)

# Absent: listed in the brief member set but NOT in the cytnx.linalg namespace.
report("linalg.Rsvd_truncate is ABSENT from cytnx.linalg (Rsvd exists, but the "
       "truncated variant is not bound)", "Rsvd_truncate" not in present)
for m in ["Add", "Sub", "Mul", "Div", "Mod"]:
    report(f"linalg.{m} is ABSENT from cytnx.linalg (only the operator dunders "
           f"exist on UniTensor; the named free function is unbound)", m not in present)

# ---------------------------------------------------------------------------
# 2. UniTensor overload support — which reachable functions accept a UniTensor.
#    Determined from the pybind-generated overload docstring.
# ---------------------------------------------------------------------------
def has_ut_overload(name):
    return "UniTensor" in (getattr(la, name).__doc__ or "")

for m in ["Svd", "Svd_truncate", "Gesvd", "Gesvd_truncate", "Rsvd", "Hosvd",
          "Qr", "Qdr", "Eig", "Eigh", "ExpH", "ExpM", "InvM", "InvM_",
          "Inv", "Inv_", "Trace", "Pow", "Pow_", "Conj", "Conj_", "Norm"]:
    report(f"linalg.{m} has a UniTensor overload", has_ut_overload(m))

# ExpH/ExpM carry BOTH a UniTensor and a Tensor overload (matrix-exponential
# family), same as the SVD/QR/Eig functions — no reverse parity gap.
report("linalg.ExpH accepts BOTH a UniTensor and a Tensor overload",
       "Tin: cytnx.cytnx.UniTensor" in (la.ExpH.__doc__ or "")
       and "Tin: cytnx.cytnx.Tensor" in (la.ExpH.__doc__ or ""))
report("linalg.ExpM accepts BOTH a UniTensor and a Tensor overload",
       "Tin: cytnx.cytnx.UniTensor" in (la.ExpM.__doc__ or "")
       and "Tin: cytnx.cytnx.Tensor" in (la.ExpM.__doc__ or ""))

# ---------------------------------------------------------------------------
# 3. Decomposition return arity/order — Svd(ut) -> [S, U, vT] in THAT order.
#    Verified by reconstructing M ~= U . diag(S) . vT.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
M0 = rng.normal(size=(4, 3))
u = mat(M0)

out = la.Svd(u)
report("Svd(ut) returns a list of length 3 ([S, U, vT])",
       isinstance(out, list) and len(out) == 3)
S, U, vT = out
report("Svd(ut)[0] is S, a diagonal UniTensor of shape [k, k]",
       S.is_diag() and S.shape() == [3, 3])
report("Svd(ut)[1] is U with shape [m, k] = [4, 3]", U.shape() == [4, 3])
report("Svd(ut)[2] is vT with shape [k, n] = [3, 3]", vT.shape() == [3, 3])

U.set_labels(["a", "i"]); S.set_labels(["i", "j"]); vT.set_labels(["j", "b"])
rec = cytnx.Contract(cytnx.Contract(U, S), vT)
rec.set_rowrank_(1)
report("Svd order verified: M ~= U . diag(S) . vT reconstructs the input",
       np.allclose(u.get_block().numpy(), rec.get_block().numpy(), atol=1e-9))

# Svd_truncate with return_err appends the truncation error UniTensor -> len 4.
tout = la.Svd_truncate(mat(M0), keepdim=2, return_err=1)
report("Svd_truncate(ut, keepdim, return_err=1) returns a length-4 list "
       "([S, U, vT, err]) — flag-dependent positional arity",
       isinstance(tout, list) and len(tout) == 4)
report("Svd_truncate(ut, keepdim, return_err=0) returns a length-3 list "
       "(no trailing err) — the same call yields different arity by flag",
       len(la.Svd_truncate(mat(M0), keepdim=2, return_err=0)) == 3)

# Qr -> [Q, R]; Eigh -> [eigvals, eigvecs].
report("Qr(ut) returns a length-2 list ([Q, R])", len(la.Qr(mat(M0))) == 2)
H = mat([[2., 1., 0.], [1., 2., 0.], [0., 0., 3.]])
report("Eigh(ut) returns a length-2 list ([eigvals, eigvecs])",
       len(la.Eigh(H)) == 2)

# ---------------------------------------------------------------------------
# 4. SVD toggle inconsistency — Svd/Svd_truncate take ONE is_UvT and REJECT
#    is_U; Gesvd/Rsvd take SEPARATE is_U/is_vT and reject is_UvT.
# ---------------------------------------------------------------------------
report("Svd accepts the single is_UvT toggle", la.Svd(mat(M0), is_UvT=True) is not None)
report("Svd REJECTS the finer is_U keyword (single-toggle family)",
       raises_typeerror(lambda: la.Svd(mat(M0), is_U=True)))
report("Svd_truncate REJECTS is_U (single is_UvT toggle only)",
       raises_typeerror(lambda: la.Svd_truncate(mat(M0), keepdim=2, is_U=True)))

report("Gesvd accepts the separate is_U / is_vT toggles",
       la.Gesvd(mat(M0), is_U=True, is_vT=True) is not None)
report("Gesvd REJECTS the single is_UvT keyword (finer-toggle family)",
       raises_typeerror(lambda: la.Gesvd(mat(M0), is_UvT=True)))
report("Rsvd accepts the separate is_U / is_vT toggles",
       la.Rsvd(mat(M0), keepdim=2, is_U=True, is_vT=True) is not None)
report("Rsvd REJECTS the single is_UvT keyword",
       raises_typeerror(lambda: la.Rsvd(mat(M0), keepdim=2, is_UvT=True)))

# ---------------------------------------------------------------------------
# 5. Inv (element-wise 1/x) vs InvM (matrix inverse) are DIFFERENT ops.
# ---------------------------------------------------------------------------
B = [[4., 1., 0.], [0., 4., 0.], [0., 0., 2.]]
inv_ew = la.Inv(mat(B)).get_block().numpy()
invm = la.InvM(mat(B)).get_block().numpy()
report("Inv is ELEMENT-WISE reciprocal: Inv(B)[0,1] == 1/1 == 1.0",
       np.isclose(inv_ew[0, 1], 1.0))
report("InvM is the MATRIX inverse: InvM(B)[0,1] == -1/16 == -0.0625",
       np.isclose(invm[0, 1], -0.0625))
report("Inv (element-wise) and InvM (matrix inverse) give DIFFERENT results "
       "— the near-name collision hides two distinct operations",
       not np.allclose(inv_ew, invm))

# In-place forms mutate and return None.
d = mat([[2., 0., 0.], [0., 4., 0.], [0., 0., 5.]])
report("InvM_ is in-place: returns None", la.InvM_(d) is None)
report("InvM_ mutated the operand in place (diag now [0.5, 0.25, 0.2])",
       np.allclose(np.diag(d.get_block().numpy()), [0.5, 0.25, 0.2]))
report("Inv_ is in-place: returns None",
       la.Inv_(mat([[2., 0., 0.], [0., 4., 0.], [0., 0., 5.]])) is None)
report("Pow_ is in-place: returns None", la.Pow_(mat(B), 2.0) is None)
report("Conj_ is in-place: returns None", la.Conj_(mat(B)) is None)

# ---------------------------------------------------------------------------
# 6. Return-type sanity for the scalar-ish members.
# ---------------------------------------------------------------------------
report("Norm(ut) returns a cytnx.Tensor (the 2-norm scalar), NOT a UniTensor",
       isinstance(la.Norm(mat(B)), cytnx.Tensor))
report("Trace(ut) returns a UniTensor", isinstance(la.Trace(mat(B)), UT))
report("Pow(ut, p) returns a UniTensor", isinstance(la.Pow(mat(B), 2.0), UT))
report("Conj(ut) returns a UniTensor", isinstance(la.Conj(mat(B)), UT))
report("ExpH(ut, a) returns a UniTensor",
       isinstance(la.ExpH(mat([[2., 0.], [0., 3.]]), 1.0), UT))

# ---------------------------------------------------------------------------
# 7. N-casing demonstration — the free functions are Capitalized at runtime.
#    (Positive demonstration that the convention keeps object-acting free
#    functions Capitalized; contrast the lowercased members of cat 07.)
# ---------------------------------------------------------------------------
report("the linalg free functions are Capitalized at runtime (Svd/Gesvd/Qr/"
       "Eigh/InvM/Trace present as capitalized names)",
       all(hasattr(la, n) for n in ["Svd", "Gesvd", "Qr", "Eigh", "InvM", "Trace"]))
report("their lowercase spellings are NOT bound (no snake_case duplicate) — "
       "e.g. linalg.svd / linalg.qr / linalg.eigh are absent",
       not any(hasattr(la, n) for n in ["svd", "qr", "eigh", "gesvd", "invm"]))

# ---------------------------------------------------------------------------
# 8. Appendix parity gap — Tensor-only linalg functions reject a UniTensor.
# ---------------------------------------------------------------------------
report("Matmul is Tensor-only: rejects a UniTensor argument",
       raises_typeerror(lambda: la.Matmul(mat(B), mat(B))))
report("Det is Tensor-only: rejects a UniTensor argument",
       raises_typeerror(lambda: la.Det(mat(B))))
report("Kron is Tensor-only: rejects a UniTensor argument",
       raises_typeerror(lambda: la.Kron(mat(B), mat(B))))

print("UniTensor 08 probe ok")
